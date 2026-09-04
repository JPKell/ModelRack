"""Provider adapter — a second real runtime, reached over the OpenAI-compatible chat API.

Imports :mod:`baseaicore`, this package's own types, ``httpx`` (through :mod:`_http`) and the
standard library; performs real network I/O, which is why every test in
``tests/unit/test_openai_compatible_adapter.py`` runs against a recorded transport rather than a
live server. Every unit and contract test in this repository runs with no network access at all
(``tests/conftest.py``'s socket guard).

The second real adapter, after :mod:`modelrack.providers.ollama` (development plan Phase 4,
"OpenAICompatibleProvider and capability honesty"). Its job is not merely "support one more
server" — it is to prove that the vocabulary Phase 1
designed and Phase 3 exercised against Ollama is not secretly Ollama-shaped
(ADR-0007 rule 1). Nothing in
:mod:`modelrack.types`, :mod:`modelrack.streaming` or :mod:`modelrack.provider` needed to change to
support this adapter — a fact worth stating plainly because it is the acceptance test the design
itself was under.

**Honest capability declaration is this module's whole point** (spec §11.10, and
ADR-0007 rule 2). An OpenAI-compatible chat
completions endpoint has:

* No digest for anything it serves — ``/v1/models`` lists an ``id`` and nothing that identifies
  the bytes behind it, so every identity here is
  :attr:`~baseaicore.IdentityConfidence.NAME_ONLY`, permanently
  (ADR-0024 §2).
* No endpoint to load, unload or query residency — ``force_unload`` and ``residency_query`` are
  declared ``False``, and :meth:`~OpenAICompatibleProvider.load`,
  :meth:`~OpenAICompatibleProvider.unload` and
  :meth:`~OpenAICompatibleProvider.list_resident` refuse immediately rather than pretending an
  HTTP call could answer a question the protocol has no field for.
* No per-request field to set a served context length — unlike Ollama's ``options.num_ctx``, there
  is nothing in the chat completions request body that changes what context a *running* server
  serves, so ``context_configurable`` is declared ``False`` and a request naming one is refused
  before a byte is sent (spec §11.10: this is load-bearing, not informational — a caller must be
  able to tell it may not assume a context it asked for was actually honoured).
* No backend timing breakdown — no analogue of Ollama's ``load_duration``/``prompt_eval_duration``/
  ``eval_duration``. Every :class:`~modelrack.types.Timing` this adapter builds carries only
  ``client_*`` fields; every ``backend_*`` field is
  :data:`~baseaicore.UNSUPPORTED`, never a guess.

**A structured error code, for once.** Ollama's context-overflow detection
(:mod:`modelrack.providers._ollama_wire`) has to sniff prose because the runtime gives no error
code for it. The OpenAI error shape's ``error.code`` is ``"context_length_exceeded"`` on every
server that follows the convention, which :func:`_is_context_overflow` checks first; the same
marker-phrase sniffing Ollama uses is kept only as a fallback for a server that reports the
condition with a message but no matching code.

**SSE, not NDJSON.** A streamed response here is `Server-Sent Events
<https://html.spec.whatwg.org/multipage/server-sent-events.html>`_: ``data: <json>`` lines
separated by blank lines, multi-line data joined with ``\\n``, ``:``-prefixed comment lines
(servers send these as keep-alives) ignored, and a final ``data: [DONE]`` sentinel.
:func:`iter_sse_events` is the whole parser — deliberately small, because the format itself is.
Reuses :func:`modelrack.providers._http.iter_capped_lines` on the raw lines underneath it, the same
per-chunk size cap :mod:`modelrack.providers.ollama` applies to NDJSON lines.

**Tool calls arrive pre-fragmented even in a single chunk.** Unlike Ollama, whose ``arguments`` is
already a parsed JSON object, this protocol's ``function.arguments`` is always a JSON *string* —
one that streams a few characters at a time across many chunks and is not valid JSON until the
last fragment lands. :attr:`~modelrack.types.ToolCall.raw_arguments` exists in the Phase-1
vocabulary precisely for this shape; :func:`tool_call_from_parts` is the one place a fragment
that never became valid JSON is preserved rather than discarded, so a malformed-arguments failure
is diagnosable rather than silently an empty mapping.
"""

from __future__ import annotations

import json
import logging
import ssl
from collections.abc import Mapping
from io import StringIO
from typing import TYPE_CHECKING, Any, Final

import httpx
from baseaicore import (
    UNSUPPORTED,
    ModelDescriptor,
    ModelIdentity,
    ProviderKind,
    TokenCount,
    TokenUsage,
    elapsed_ms,
    is_supported,
    monotonic_ns,
    utc_now,
)

from modelrack.cache import (
    DEFAULT_METADATA_TTL_SECONDS,
    CacheStats,
    MetadataCache,
    MetadataSnapshot,
)
from modelrack.errors import (
    ContextLimitExceeded,
    GenerationCancelled,
    ModelNotFound,
    ProviderError,
    ProviderProtocolError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from modelrack.events import EventEmitter
from modelrack.provider import (
    LoadResult,
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    ResidentModel,
    refuse_capability,
)
from modelrack.providers._http import (
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    build_client,
    iter_capped_lines,
    read_capped_json,
    translate_stream_interruption,
    translate_transport_error,
    truncated_text,
    validate_base_url,
)
from modelrack.providers._openai_wire import (
    extract_error,
    finish_reason_for,
    first_choice,
    iter_sse_events,
    message_payload,
    parse_tool_calls,
    request_tool_definitions,
    response_format_payload,
    tool_call_fragment,
    tool_call_from_parts,
    tool_call_index,
)
from modelrack.residency import refuse_force_unload, refuse_residency_query
from modelrack.streaming import (
    StreamCompleted,
    StreamFailed,
    TokenDelta,
    ToolCallDelta,
)
from modelrack.types import (
    GenerationResult,
    GenerationUsage,
    Message,
    Role,
    Timing,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence
    from datetime import datetime

    from baseaicore import RuntimeProfile

    from modelrack.adapters import AdapterRegistration, AdapterState
    from modelrack.events import EventCallback
    from modelrack.streaming import StreamEvent
    from modelrack.types import GenerationRequest

__all__ = ["OpenAICompatibleProvider"]

logger = logging.getLogger(__name__)

_CAPABILITIES: Final[ProviderCapabilities] = ProviderCapabilities(
    streaming=True,
    tool_calling=True,
    structured_output=True,
    json_mode=True,
    token_counts=True,
    token_level_chunks=False,
    thinking_control=False,
    logprobs=False,
    force_unload=False,
    residency_query=False,
    kv_metrics=False,
    context_configurable=False,
    embedding=False,
)
"""What a representative OpenAI-compatible chat completions server can do.

``token_level_chunks`` is ``False`` on principle, not measurement: nothing in the chat completions
streaming format promises one delta per model token — a ``delta.content`` fragment is whatever
size the server's own tokenizer-to-text boundary produced, exactly the "batches tokens into
transport chunks" shape [spec §11.4](../../../docs/packages/modelrack/spec.md) warns a caller not
to relabel as per-token latency. ``thinking_control`` is ``False``: the chat completions shape has
no reasoning-content field this adapter reads (some servers add one under a private key; declaring
the flag ``True`` on the strength of an undocumented extension is exactly the "capability nobody
tested" ADR-0007 rule 2 warns against).
"""

_REQUEST_HEADERS: Final[dict[str, str]] = {"Content-Type": "application/json"}

_MODELS_CACHE_KEY: Final[str] = "models"
"""The metadata cache's only key. This protocol has one discovery endpoint and no per-model
metadata call, so there is nothing else to hold."""

_CONTEXT_OVERFLOW_CODES: Final[frozenset[str]] = frozenset({"context_length_exceeded"})

# A message-text fallback for a server that reports the condition without the structured code
# above — the same discipline modelrack.providers._ollama_wire applies, kept here as the fallback
# rather than the primary signal, because this protocol has one.
_CONTEXT_OVERFLOW_MARKERS: Final[tuple[str, ...]] = (
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
)


class OpenAICompatibleProvider:
    """A real :class:`~modelrack.provider.Provider`, reached over an OpenAI-compatible chat API.

    Owns one pooled :class:`httpx.Client` for its whole lifetime (spec §15). Every method raises
    the typed errors in :mod:`modelrack.errors`, never a raw ``httpx`` exception (spec §11.7).

    Args:
        base_url: Where the server is listening, e.g. ``http://127.0.0.1:8080``. Validated to an
            ``http``/``https`` scheme with a host; a non-loopback host is permitted but flagged as
            remote on every :class:`~modelrack.provider.ProviderHealth` result (spec §14).
        api_key: Sent as ``Authorization: Bearer <api_key>`` on every request when supplied, and
            nowhere else — never logged, never in ``raw``, never in an error's ``details``
            (spec §14). ``None`` sends no ``Authorization`` header at all, the common case for a
            local server with no credential configured.
        timeout: The default applied to a call that names none (spec §12). Never ``None``.
        headers: Sent on every request in addition to ``Content-Type`` and, if supplied,
            ``Authorization``.
        client: An already-constructed :class:`httpx.Client` to use instead of building one.
        verify: TLS verification, passed to :class:`httpx.Client` unchanged.
        max_response_bytes: The total-body cap for a non-streamed response (spec §14).
        max_chunk_bytes: The per-line cap for a streamed response (spec §14).
        metadata_ttl_seconds: How long a ``/v1/models`` body may be reused (spec §10, default
            300 s). ``0`` disables the cache. This protocol has no per-model metadata endpoint,
            so the listing is the only thing there is to cache.
        on_event: An optional observer called as requests start, stream and finish (spec §17).
            Receives no prompt, no generated text and — the reason it matters here rather than on
            the Ollama adapter — no ``api_key``: :class:`~modelrack.events.ProviderEvent` has no
            field a credential could reach, and a test asserts it stays out of the event stream
            as well as out of ``raw``, ``details`` and the DEBUG log (spec §14).
        clock: Where a :class:`~baseaicore.ModelDescriptor`'s ``observed_at`` comes from, injected
            so a test can freeze it (coding standards §5). Read when the listing is fetched and
            stored with it, so a cache hit reports when the server actually answered.

    Raises:
        ValidationError: If ``base_url`` has no host or uses a scheme other than ``http``/
            ``https``, or if ``metadata_ttl_seconds`` is negative.
    """

    kind: ProviderKind = ProviderKind.OPENAI_COMPATIBLE

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT_SECONDS,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
        verify: ssl.SSLContext | str | bool = True,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        metadata_ttl_seconds: float = DEFAULT_METADATA_TTL_SECONDS,
        on_event: EventCallback | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        """Validate the base URL and construct or adopt the pooled HTTP client."""
        validated_url, is_remote = validate_base_url(base_url)
        self._base_url = validated_url
        self._is_remote = is_remote
        self._max_response_bytes = max_response_bytes
        self._max_chunk_bytes = max_chunk_bytes
        self._clock = clock
        self._cache: MetadataCache[MetadataSnapshot] = MetadataCache(
            ttl_seconds=metadata_ttl_seconds
        )
        self._events = EventEmitter(on_event, provider_kind=self.kind)
        request_headers = {**_REQUEST_HEADERS, **(headers or {})}
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or build_client(
            base_url=validated_url, timeout=timeout, headers=request_headers, verify=verify
        )

    # ------------------------------------------------------------------------------- protocol

    def health(self) -> ProviderHealth:
        """Probe the provider and report whether it can be used.

        ``/v1/models`` is the one endpoint every OpenAI-compatible server implements, so it stands
        in for a dedicated health check the way it stands in for discovery. Reached through
        :meth:`_get_json`, which already translates a transport failure into one of the typed
        errors caught below — unlike
        :meth:`modelrack.providers.ollama.OllamaProvider.health`, which calls the client directly
        and so catches the raw ``httpx`` exception itself.

        Returns:
            :attr:`~modelrack.provider.ProviderStatus.OK` with the model count on success;
            :attr:`~modelrack.provider.ProviderStatus.UNAVAILABLE` when the provider cannot be
            reached at all; :attr:`~modelrack.provider.ProviderStatus.DEGRADED` when it answered
            and refused. The third case matters more for this adapter than for Ollama's: a wrong
            or expired ``api_key`` is a 401 from a server that is running perfectly well, and
            reporting that as "unreachable" would send an operator to check the wrong thing
            entirely. **Never a raised exception**, whichever of the three it is.
        """
        start_ns = monotonic_ns()
        try:
            payload = self._get_json("/v1/models")
        except (ProviderUnavailable, ProviderTimeout, ProviderProtocolError):
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                base_url=self._base_url,
                is_remote=self._is_remote,
                detail="unreachable",
                latency_ms=elapsed_ms(start_ns),
            )
        except ProviderError as exc:
            # The server's own message is deliberately not repeated in a health detail — see
            # `modelrack.providers.ollama.OllamaProvider.health`. It matters more here: this is
            # the adapter with a credential, and a health document is rendered into a UI.
            return ProviderHealth(
                status=ProviderStatus.DEGRADED,
                base_url=self._base_url,
                is_remote=self._is_remote,
                detail=f"reachable, but refused the probe ({exc.code})",
                latency_ms=elapsed_ms(start_ns),
            )
        count = len(self._model_entries(payload))
        return ProviderHealth(
            status=ProviderStatus.OK,
            base_url=self._base_url,
            is_remote=self._is_remote,
            detail=f"openai-compatible, {count} models",
            # No documented field on /v1/models reports the server's own version; declaring one
            # here would be a guess (spec §14's own "never assume" applies as much to a health
            # detail as to a capability flag).
            provider_version=None,
            model_count=count,
            latency_ms=elapsed_ms(start_ns),
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report what this adapter can do — the static declaration, no request made."""
        return _CAPABILITIES

    # ------------------------------------------------------------------------- metadata cache

    @property
    def metadata_cache_ttl_seconds(self) -> float:
        """How long a cached ``/v1/models`` body is reused, in seconds.

        Returns:
            The lifetime this adapter was constructed with; ``0.0`` when caching is disabled.
        """
        return self._cache.ttl_seconds

    def metadata_cache_stats(self) -> CacheStats:
        """Report what the metadata cache has done since construction or the last clear.

        Returns:
            A consistent snapshot of the hit, miss, expiry and store counters plus the current
            entry count. Metadata only — no generation result ever enters the cache (spec §3).
        """
        return self._cache.stats()

    def clear_metadata_cache(self) -> None:
        """Drop the cached model listing and reset the cache counters (spec §10)."""
        self._cache.clear()

    def list_models(self, *, refresh: bool = False) -> Sequence[ModelDescriptor]:
        """List the models the server is serving.

        Args:
            refresh: Re-read ``/v1/models``, ignoring anything cached (spec §10). The
                explicit escape hatch a TTL alone cannot provide, for a caller that has just
                loaded a new model into the server.

        Returns:
            One descriptor per model. ``/v1/models`` carries an ``id`` and little else on most
            servers, so every field beyond ``identity``, ``observed_at`` and ``raw`` is
            :data:`~baseaicore.UNSUPPORTED` or ``None`` — this adapter never invents architecture
            metadata a discovery endpoint did not report (spec §11.2, "limited metadata").

        Raises:
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        snapshot = self._models_snapshot(refresh=refresh)
        entries = self._model_entries(snapshot.payload)
        return tuple(
            _build_descriptor(entry, observed_at=snapshot.observed_at) for entry in entries
        )

    def inspect_model(self, identity: ModelIdentity, *, refresh: bool = False) -> ModelDescriptor:
        """Fetch metadata for one model.

        Args:
            identity: The model to inspect, matched on
                :attr:`~baseaicore.ModelIdentity.provider_model_name`.
            refresh: Re-read ``/v1/models``, ignoring anything cached (spec §10). The
                explicit escape hatch a TTL alone cannot provide, for a caller that has just
                loaded a new model into the server.

        Returns:
            The descriptor.

        Raises:
            ModelNotFound: If the server does not have it.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        descriptors = self.list_models(refresh=refresh)
        for descriptor in descriptors:
            if descriptor.identity.provider_model_name == identity.provider_model_name:
                return descriptor
        raise ModelNotFound(
            f"No model named {identity.provider_model_name!r} is served by this provider.",
            details={
                "reference": identity.provider_model_name,
                "known_model_count": len(descriptors),
            },
        )

    def resolve(self, reference: str, *, refresh: bool = False) -> ModelIdentity:
        """Resolve a user-supplied model reference to a concrete identity.

        Tries an exact ``id`` match, then a unique prefix over every known ``id``. Unlike
        :meth:`~modelrack.providers.ollama.OllamaProvider.resolve`, there is no ``:latest``-suffix
        convention to try — that is Ollama's own tagging scheme, not part of this protocol.

        Args:
            reference: What the user typed.
            refresh: Re-read ``/v1/models``, ignoring anything cached (spec §10). The
                explicit escape hatch a TTL alone cannot provide, for a caller that has just
                loaded a new model into the server.

        Returns:
            The resolved identity — always :attr:`~baseaicore.IdentityConfidence.NAME_ONLY`.

        Raises:
            ModelNotFound: If nothing matches, or a prefix matches more than one model.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        descriptors = self.list_models(refresh=refresh)
        names = [descriptor.identity.provider_model_name for descriptor in descriptors]
        resolved = self._resolve_name(reference, names)
        identity = _identity_for(resolved)
        if reference != resolved:
            logger.debug(
                "openai_compatible.model.resolved",
                extra={"reference": reference, "resolved_to": resolved},
            )
        return identity

    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult:
        """Refuse: this protocol has no endpoint to load a model explicitly.

        Raises:
            CapabilityUnsupported: Always — ``force_unload`` is declared ``False``, and the
                normative capability set has no separate "can load" flag
                (ADR-0007 rule 2).
        """
        refuse_force_unload(action="load a model on demand")

    def unload(self, identity: ModelIdentity) -> bool:
        """Refuse: this protocol has no endpoint to evict a model on demand.

        Raises:
            CapabilityUnsupported: Always — ``force_unload`` is declared ``False``.
        """
        refuse_force_unload(action="evict a model on demand")

    def list_resident(self) -> Sequence[ResidentModel]:
        """Refuse: this protocol has no endpoint to report what is currently loaded.

        Raises:
            CapabilityUnsupported: Always — ``residency_query`` is declared ``False``.
        """
        refuse_residency_query()

    # ------------------------------------------------------------------------ generation

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one non-streamed generation and return the complete result.

        Args:
            request: What to generate, and how.

        Returns:
            The complete outcome. ``client_ttft_ms`` and every ``backend_*`` timing field are
            :data:`~baseaicore.UNSUPPORTED` — there is no first-token moment on a blocking call,
            and this protocol reports no backend timing breakdown at all.

        Raises:
            CapabilityUnsupported: If ``request.runtime_profile.context_size`` is set —
                ``context_configurable`` is declared ``False`` (spec §11.10).
            ModelNotFound: If the requested model is not available.
            ContextLimitExceeded: If the server reports the request needed more context than it
                serves.
            ProviderRejected: If the server understood the request and refused it.
            ProviderProtocolError: If the response cannot be parsed.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        body = self._build_body(request, stream=False)
        start_ns = monotonic_ns()
        self._events.started(
            operation="generate",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
        )
        try:
            result = self._generate_once(request, body, start_ns)
        except ProviderError as exc:
            self._events.failed(
                operation="generate",
                model_name=request.identity.provider_model_name,
                metadata=request.metadata,
                error_code=exc.code,
                elapsed_ms=elapsed_ms(start_ns),
            )
            raise
        self._events.completed(
            operation="generate",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
            elapsed_ms=result.timing.client_wall_ms,
            output_tokens=result.usage.tokens.output_tokens,
            finish_reason=result.finish_reason.value,
        )
        return result

    def _generate_once(
        self, request: GenerationRequest, body: Mapping[str, Any], start_ns: int
    ) -> GenerationResult:
        """Run the round trip :meth:`generate` wraps in its event pair.

        Split out for the same reason
        :meth:`modelrack.providers.ollama.OllamaProvider._generate_once` is: so that every
        ``raise`` below is reported as a failure without each one having to remember to say so.
        """
        payload = self._post_json(
            "/v1/chat/completions",
            body,
            model_reference=request.identity.provider_model_name,
            timeout=request.timeout_seconds,
        )
        if not isinstance(payload, dict):
            raise ProviderProtocolError(
                f"The provider at {self._base_url} returned something other than a JSON object.",
                details={"base_url": self._base_url, "body": truncated_text(json.dumps(payload))},
            )
        message, code = extract_error(payload)
        if message is not None:
            raise self._build_message_error(message, code=code, status_code=200)
        choice = first_choice(payload)
        response_message = choice.get("message")
        response_message = response_message if isinstance(response_message, Mapping) else {}
        text = response_message.get("content")
        text = text if isinstance(text, str) else ""
        call_prefix = f"oai-{start_ns}"
        tool_calls = parse_tool_calls(response_message.get("tool_calls"), call_prefix=call_prefix)
        return GenerationResult(
            text=text,
            identity=request.identity,
            finish_reason=finish_reason_for(
                choice.get("finish_reason"), has_tool_calls=bool(tool_calls)
            ),
            usage=_read_usage(payload, text=text),
            timing=Timing(client_wall_ms=elapsed_ms(start_ns), client_ttft_ms=UNSUPPORTED),
            tool_calls=tool_calls,
            thinking=UNSUPPORTED,
            provider_version=None,
            raw=dict(payload),
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        """Run one generation, yielding events as they arrive over the server-sent-event stream.

        Everything that can fail before a byte of the stream arrives raises from this call; every
        failure after that, including the caller's own cancellation, is delivered as
        :class:`~modelrack.streaming.StreamFailed`, with the connection closed either way (see
        :mod:`modelrack.providers.ollama`'s ``stream`` docstring — the same generator-ownership
        guarantee applies here, unchanged, because it belongs to the protocol, not the wire format).

        Args:
            request: What to generate, and how. Its ``cancel`` token is honoured between SSE
                events. A token already set when this is called yields one terminal
                :class:`~modelrack.streaming.StreamFailed` and opens no connection at all.

        Yields:
            Deltas, then one terminal event.

        Raises:
            CapabilityUnsupported: If ``request.runtime_profile.context_size`` is set.
            ModelNotFound: If the requested model is not available.
            ProviderRejected: If the server understood the request and refused it before streaming.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        body = self._build_body(request, stream=True)
        if request.cancel is not None and request.cancel.is_cancelled:
            return iter((self._already_cancelled(request),))
        start_ns = monotonic_ns()
        self._events.started(
            operation="stream",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
        )
        prepared = self._client.build_request(
            "POST", "/v1/chat/completions", json=body, timeout=self._timeout_for(request)
        )
        try:
            response = self._client.send(prepared, stream=True)
        except httpx.HTTPError as exc:
            error = translate_transport_error(exc, base_url=self._base_url)
            self._events.failed(
                operation="stream",
                model_name=request.identity.provider_model_name,
                metadata=request.metadata,
                error_code=error.code,
                elapsed_ms=elapsed_ms(start_ns),
            )
            raise error from exc
        try:
            if response.status_code >= 400:
                self._raise_for_status(
                    response, model_reference=request.identity.provider_model_name
                )
        except ProviderError as exc:
            response.close()
            self._events.failed(
                operation="stream",
                model_name=request.identity.provider_model_name,
                metadata=request.metadata,
                error_code=exc.code,
                elapsed_ms=elapsed_ms(start_ns),
            )
            raise
        except BaseException:
            response.close()
            raise
        return self._walk(request, response, start_ns)

    def _already_cancelled(self, request: GenerationRequest) -> StreamFailed:
        """Return the terminal event for a stream whose token was already set before it began.

        Opens no connection at all — see
        :meth:`modelrack.providers.ollama.OllamaProvider._already_cancelled`, which this mirrors
        exactly, because cancellation semantics belong to the protocol rather than to a wire
        format (spec §11.6).
        """
        self._events.started(
            operation="stream",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
        )
        event = self._cancelled("")
        self._events.failed(
            operation="stream",
            model_name=request.identity.provider_model_name,
            metadata=request.metadata,
            error_code=event.error.code,
        )
        return event

    def _walk(
        self, request: GenerationRequest, response: httpx.Response, start_ns: int
    ) -> Iterator[StreamEvent]:
        """Observe every event :meth:`_drain` produces, then hand it on unchanged.

        Mirrors :meth:`modelrack.providers.ollama.OllamaProvider._walk`, including the explicit
        ``events.close()``: without it, the inner generator's ``finally`` — the one holding
        ``response.close()`` — would run only when the garbage collector reached it.
        """
        model_name = request.identity.provider_model_name
        events = self._drain(request, response, start_ns)
        try:
            for event in events:
                self._observe(event, request, start_ns, model_name)
                yield event
        finally:
            events.close()

    def _observe(
        self, event: StreamEvent, request: GenerationRequest, start_ns: int, model_name: str
    ) -> None:
        """Emit the observability event matching one stream event, if anyone is listening."""
        if not self._events.is_observed:
            return
        if isinstance(event, StreamCompleted):
            self._events.completed(
                operation="stream",
                model_name=model_name,
                metadata=request.metadata,
                elapsed_ms=event.result.timing.client_wall_ms,
                output_tokens=event.result.usage.tokens.output_tokens,
                finish_reason=event.result.finish_reason.value,
            )
        elif isinstance(event, StreamFailed):
            self._events.failed(
                operation="stream",
                model_name=model_name,
                metadata=request.metadata,
                error_code=event.error.code,
                elapsed_ms=elapsed_ms(start_ns),
            )
        else:
            self._events.chunk(
                operation="stream",
                model_name=model_name,
                metadata=request.metadata,
                chunk_index=event.index,
                elapsed_ms=elapsed_ms(start_ns),
            )

    def _drain(
        self, request: GenerationRequest, response: httpx.Response, start_ns: int
    ) -> Generator[StreamEvent, None, None]:
        """Drain one SSE stream, owning ``response`` for its entire remaining lifetime.

        See :meth:`modelrack.providers.ollama.OllamaProvider._drain` for why this is a generator:
        ``GeneratorExit`` on abandonment runs the same ``finally`` a normal return does, which is
        what makes draining, breaking early and walking away all close the connection identically.
        """
        cancel = request.cancel
        answer = StringIO()
        tool_fragments: dict[int, dict[str, Any]] = {}
        tool_order: list[int] = []
        first_delta_ns: int | None = None
        delta_index = 0
        finish_reason_raw: Any = None
        usage_payload: Mapping[str, Any] = {}
        seen_done = False
        try:
            lines = iter_capped_lines(
                response.iter_lines(),
                max_chunk_bytes=self._max_chunk_bytes,
                base_url=self._base_url,
            )
            for data in iter_sse_events(lines):
                if cancel is not None and cancel.is_cancelled:
                    yield self._cancelled(answer.getvalue())
                    return
                if data == "[DONE]":
                    seen_done = True
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    yield StreamFailed(
                        error=ProviderProtocolError(
                            f"A streamed event from {self._base_url} was not valid JSON.",
                            details={"base_url": self._base_url, "body": truncated_text(data)},
                        ),
                        partial_text=answer.getvalue(),
                    )
                    return
                if not isinstance(payload, dict):
                    yield StreamFailed(
                        error=ProviderProtocolError(
                            f"A streamed event from {self._base_url} was not a JSON object.",
                            details={"base_url": self._base_url, "body": truncated_text(data)},
                        ),
                        partial_text=answer.getvalue(),
                    )
                    return
                message, code = extract_error(payload)
                if message is not None:
                    yield StreamFailed(
                        error=self._build_message_error(
                            message, code=code, status_code=response.status_code
                        ),
                        partial_text=answer.getvalue(),
                    )
                    return

                usage = payload.get("usage")
                if isinstance(usage, Mapping):
                    usage_payload = usage
                choice = first_choice(payload)
                delta = choice.get("delta")
                delta = delta if isinstance(delta, Mapping) else {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    first_delta_ns = first_delta_ns or monotonic_ns()
                    answer.write(content)
                    yield TokenDelta(text=content, index=delta_index)
                    delta_index += 1
                raw_tool_calls = delta.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    for fragment in raw_tool_calls:
                        if not isinstance(fragment, Mapping):
                            continue
                        first_delta_ns = first_delta_ns or monotonic_ns()
                        call_index = tool_call_index(fragment, fallback=len(tool_order))
                        if call_index not in tool_fragments:
                            tool_fragments[call_index] = {"id": None, "name": None, "arguments": []}
                            tool_order.append(call_index)
                        entry = tool_fragments[call_index]
                        fragment_id, fragment_name, fragment_arguments = tool_call_fragment(
                            fragment
                        )
                        if fragment_id is not None:
                            entry["id"] = fragment_id
                        if fragment_name is not None:
                            entry["name"] = fragment_name
                        if fragment_arguments is not None:
                            entry["arguments"].append(fragment_arguments)
                        yield ToolCallDelta(
                            call_index=call_index,
                            id=fragment_id,
                            name=fragment_name,
                            arguments_fragment=fragment_arguments,
                            index=delta_index,
                        )
                        delta_index += 1
                choice_finish = choice.get("finish_reason")
                if choice_finish is not None:
                    finish_reason_raw = choice_finish
            else:
                yield StreamFailed(
                    error=ProviderProtocolError(
                        f"The stream from {self._base_url} ended without a [DONE] sentinel.",
                        details={"base_url": self._base_url},
                    ),
                    partial_text=answer.getvalue(),
                )
                return

            if not seen_done:  # pragma: no cover — the `else` above always returns first
                raise AssertionError("stream loop exited without seeing [DONE] or returning")
            if cancel is not None and cancel.is_cancelled:  # pragma: no cover — see below
                # Reachable only if cancel() fires from another thread in the narrow window
                # between the top-of-loop check on the [DONE] event and this one — `[DONE]` is
                # always its own SSE event, never sharing an iteration with a content delta the
                # way Ollama's terminal NDJSON line does, so a single-threaded test cannot land
                # here deterministically. Kept as the honest defensive check for that race.
                yield self._cancelled(answer.getvalue())
                return
            tool_calls = tuple(
                tool_call_from_parts(
                    call_id=tool_fragments[index]["id"],
                    name=tool_fragments[index]["name"],
                    raw_arguments="".join(tool_fragments[index]["arguments"]) or None,
                    fallback_id=f"oai-{start_ns}-{position}",
                )
                for position, index in enumerate(tool_order)
            )
            wall_ms = elapsed_ms(start_ns)
            ttft_ms = (
                elapsed_ms(start_ns, first_delta_ns) if first_delta_ns is not None else UNSUPPORTED
            )
            text = answer.getvalue()
            yield StreamCompleted(
                result=GenerationResult(
                    text=text,
                    identity=request.identity,
                    finish_reason=finish_reason_for(
                        finish_reason_raw, has_tool_calls=bool(tool_calls)
                    ),
                    usage=_read_usage({"usage": usage_payload}, text=text),
                    timing=Timing(client_wall_ms=wall_ms, client_ttft_ms=ttft_ms),
                    tool_calls=tool_calls,
                    thinking=UNSUPPORTED,
                    provider_version=None,
                    raw={"usage": dict(usage_payload)} if usage_payload else {},
                )
            )
        except httpx.HTTPError as exc:
            yield StreamFailed(
                error=translate_stream_interruption(exc, base_url=self._base_url),
                partial_text=answer.getvalue(),
            )
        except ProviderProtocolError as exc:
            # Raised by `iter_capped_lines` when a line exceeds the per-chunk cap — already a
            # typed error, delivered as a terminal event rather than re-raised because the stream
            # has already begun (spec §13, and modelrack.streaming's own terminal-event rule).
            yield StreamFailed(error=exc, partial_text=answer.getvalue())
        finally:
            response.close()

    def _cancelled(self, partial_text: str) -> StreamFailed:
        """Return the terminal event for a stream the caller stopped, its output attached."""
        return StreamFailed(
            error=GenerationCancelled(
                "Generation was cancelled by the caller's token.",
                details={"partial_text": partial_text},
            ),
            partial_text=partial_text,
        )

    def _build_message_error(
        self, message: str, *, code: str | None, status_code: int
    ) -> ContextLimitExceeded | ProviderRejected:
        """Classify an error the server sent as content, the same way for a 4xx and an in-band
        streamed error — see :meth:`modelrack.providers.ollama.OllamaProvider._build_message_error`
        for why this is classified by content rather than by status code alone.
        """
        if _is_context_overflow(message, code):
            return ContextLimitExceeded(
                message, details={"requested_tokens": UNSUPPORTED, "maximum_tokens": UNSUPPORTED}
            )
        return ProviderRejected(
            message, details={"status_code": status_code, "provider_message": message}
        )

    # ------------------------------------------------------------------------- transport calls

    def _timeout_for(self, request: GenerationRequest) -> float | httpx._client.UseClientDefault:
        """Return the per-request timeout override, or the client's own default."""
        if request.timeout_seconds is not None:
            return request.timeout_seconds
        return httpx.USE_CLIENT_DEFAULT

    def _raise_for_status(self, response: httpx.Response, *, model_reference: str | None) -> None:
        """Translate a non-2xx response into the typed error spec §13 names for it, and raise it."""
        try:
            payload = read_capped_json(
                response, max_bytes=self._max_response_bytes, base_url=self._base_url
            )
            raw_text = json.dumps(payload)
        except ProviderProtocolError as exc:
            if "limit_bytes" in exc.details:
                raise
            payload, raw_text = None, str(exc.details.get("body", ""))
        message, code = extract_error(payload)
        if response.status_code == 404 and model_reference is not None:
            known = len(self.list_models())
            raise ModelNotFound(
                message or f"Model {model_reference!r} not found.",
                details={"reference": model_reference, "known_model_count": known},
            )
        if message is not None:
            raise self._build_message_error(message, code=code, status_code=response.status_code)
        raise ProviderProtocolError(
            f"The provider at {self._base_url} returned status {response.status_code} with an "
            "unexpected body.",
            details={
                "base_url": self._base_url,
                "status_code": response.status_code,
                "body": truncated_text(raw_text),
            },
        )

    def _post_json(
        self,
        path: str,
        body: Mapping[str, Any],
        *,
        model_reference: str | None = None,
        timeout: float | None = None,
    ) -> Any:  # noqa: ANN401 — the provider's own JSON shape
        """POST a JSON body and return the parsed JSON response, translating every failure."""
        try:
            with self._client.stream(
                "POST",
                path,
                json=body,
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            ) as response:
                if response.status_code >= 400:
                    self._raise_for_status(response, model_reference=model_reference)
                return read_capped_json(
                    response, max_bytes=self._max_response_bytes, base_url=self._base_url
                )
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc, base_url=self._base_url) from exc

    def _models_snapshot(self, *, refresh: bool) -> MetadataSnapshot:
        """Return ``/v1/models``'s body with the instant it was read, cached under one key.

        The clock is read *after* the fetch returns, so ``observed_at`` names the moment the
        server had answered rather than the moment this process decided to ask.
        """
        if not refresh:
            cached = self._cache.get(_MODELS_CACHE_KEY)
            if cached is not None:
                return cached
        payload = self._get_json("/v1/models")
        snapshot = MetadataSnapshot(
            observed_at=self._clock(), payload=payload if isinstance(payload, dict) else {}
        )
        self._cache.put(_MODELS_CACHE_KEY, snapshot)
        return snapshot

    def _get_json(self, path: str) -> Any:  # noqa: ANN401 — the provider's own JSON shape
        """GET a path and return the parsed JSON response, translating every failure."""
        try:
            with self._client.stream("GET", path) as response:
                if response.status_code >= 400:
                    self._raise_for_status(response, model_reference=None)
                return read_capped_json(
                    response, max_bytes=self._max_response_bytes, base_url=self._base_url
                )
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc, base_url=self._base_url) from exc

    @staticmethod
    def _model_entries(payload: Any) -> list[dict[str, Any]]:  # noqa: ANN401 — provider JSON
        """Return ``/v1/models``'s ``data`` array as dicts, or an empty list if malformed."""
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    @staticmethod
    def _resolve_name(reference: str, names: Sequence[str]) -> str:
        """Resolve ``reference`` to one served ``id``: exact match, then a unique prefix."""
        if reference in names:
            return reference
        prefixed = [name for name in names if name.startswith(reference)]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            raise ModelNotFound(
                f"{reference!r} is a prefix of {len(prefixed)} models; it names none of them. "
                "Give enough of the name to pick one — resolving an ambiguous reference by "
                "choosing would run weights you did not ask for.",
                details={
                    "reference": reference,
                    "known_model_count": len(names),
                    "matched_model_count": len(prefixed),
                },
            )
        raise ModelNotFound(
            f"No model matching {reference!r} is served by this provider.",
            details={"reference": reference, "known_model_count": len(names)},
        )

    def list_adapters(self) -> Sequence[AdapterState]:
        """Refuse: this provider has no adapter mechanism.

        Raises:
            CapabilityUnsupported: Always — ``adapter_hot_swap`` is declared ``False``. Raised
                rather than answering ``()``, because "no adapters registered" and "adapters are
                not a thing here" are different facts and a caller that conflated them would
                report a misconfiguration as an empty registry.
        """
        refuse_capability("adapter_hot_swap", action="report registered adapters")

    def register_adapters(self, adapters: Sequence[AdapterRegistration]) -> None:
        """Refuse: this provider has no adapter mechanism.

        Args:
            adapters: Ignored — the refusal comes first, so nothing is ever half-registered.

        Raises:
            CapabilityUnsupported: Always — ``adapter_hot_swap`` is declared ``False``.
        """
        refuse_capability("adapter_hot_swap", action="register an adapter")

    def _build_body(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        """Build the ``/v1/chat/completions`` request body.

        A completion-style request (``request.prompt``) is sent as a single user message: this
        protocol exposes one generation endpoint, ``/v1/chat/completions``, and there is no
        legacy-completions call in this adapter's scope
        ([development plan](../../../docs/packages/modelrack/development-plan.md) Phase 4's Work
        list names only the chat endpoint) — unlike
        :meth:`modelrack.providers.ollama.OllamaProvider._build_request`, there is therefore no
        endpoint-specific restriction to enforce here: tools and a completion-style prompt are not
        in tension the way they are for Ollama's ``/api/generate``.
        """
        if request.adapter is not None:
            refuse_capability("adapter_hot_swap", action="run a request under a LoRA adapter")
        if request.runtime_profile.context_size is not None:
            refuse_capability("context_configurable", action="serve a caller-chosen context")
        messages = request.messages or (Message(role=Role.USER, content=request.prompt or ""),)
        body: dict[str, Any] = {
            "model": request.identity.provider_model_name,
            "messages": [message_payload(message) for message in messages],
            "stream": stream,
        }
        sampling = request.sampling
        if sampling.temperature is not None:
            body["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            body["top_p"] = sampling.top_p
        if sampling.top_k is not None:
            # Not part of the OpenAI API itself, but accepted as an extension by every local
            # OpenAI-compatible server this adapter targets (llama.cpp server, vLLM); sent
            # directly rather than folded into provider_options, the same way Ollama's adapter
            # sends its own provider-specific extensions as named fields, not an escape hatch.
            body["top_k"] = sampling.top_k
        if sampling.seed is not None:
            body["seed"] = sampling.seed
        if sampling.max_output_tokens is not None:
            body["max_tokens"] = sampling.max_output_tokens
        if sampling.stop:
            body["stop"] = list(sampling.stop)
        if sampling.repeat_penalty is not None:
            # Two spellings for one knob, because the servers this adapter targets disagree:
            # vLLM reads `repetition_penalty` and llama-server reads `repeat_penalty`, and each
            # ignores the other. Sending both is the only way a caller's penalty reaches whichever
            # is on the other end, and neither server objects to an unknown key.
            body["repetition_penalty"] = sampling.repeat_penalty
            body["repeat_penalty"] = sampling.repeat_penalty
        if request.tools:
            body["tools"] = request_tool_definitions(request.tools)
        if request.response_format is not None:
            body["response_format"] = response_format_payload(request.response_format)
        # provider_options is the caller's own escape hatch (spec §12) and is merged last, so it
        # wins on any overlapping key — the same rule _ollama_wire.generation_options documents.
        body.update(request.runtime_profile.provider_options)
        return body


def _identity_for(model_id: str) -> ModelIdentity:
    """Build the identity for one served model — always name-only.

    No OpenAI-compatible discovery endpoint reports a content digest for what it is serving, so
    every identity from this adapter carries :attr:`~baseaicore.IdentityConfidence.NAME_ONLY`
    unconditionally, never a fabricated one
    (ADR-0024 §2).
    """
    return ModelIdentity(ProviderKind.OPENAI_COMPATIBLE, model_id)


def _build_descriptor(entry: Mapping[str, Any], *, observed_at: datetime) -> ModelDescriptor:
    """Build a :class:`~baseaicore.ModelDescriptor` from one ``/v1/models`` entry.

    Everything beyond identity and the raw payload stays unset: ``/v1/models`` on a representative
    server reports an ``id``, an ``object`` marker and an ``owned_by`` string, none of which is
    architecture metadata this vocabulary has a field for — inventing one from ``id`` text (for
    example, guessing a parameter count from a model name containing ``"8b"``) is exactly the kind
    of guess spec §11.9's digest discipline and ADR-0016 both forbid elsewhere in this package.
    """
    model_id = entry.get("id")
    model_id = model_id if isinstance(model_id, str) else ""
    return ModelDescriptor(
        identity=_identity_for(model_id), observed_at=observed_at, raw=dict(entry)
    )


def _as_token_count(value: object) -> Any:  # noqa: ANN401 — mirrors baseaicore.TokenCount
    """Return ``value`` if it is a non-negative whole number, else ``UNSUPPORTED``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return UNSUPPORTED
    return value


def _reconcile_prompt_tokens(usage: Mapping[str, Any]) -> tuple[TokenCount, TokenCount]:
    """Split the wire's prompt figure into the disjoint ``input`` and ``cache read`` classes.

    ADR-0030 makes the adapter the only
    layer that knows a provider's convention, and this protocol's convention is that
    ``prompt_tokens`` is a *total* which already contains any cached tokens reported beside it in
    ``prompt_tokens_details.cached_tokens``. Subtracting is therefore not an adjustment but the
    translation into :class:`~baseaicore.TokenUsage`'s disjoint classes; without it a cached
    prefix is billed twice, once at the input rate and once at the cache-read rate.

    Args:
        usage: A non-empty wire ``usage`` object.

    Returns:
        ``(input_tokens, cache_read_tokens)``, in three cases:

        * **No ``prompt_tokens_details`` key.** The server does no cache accounting, so nothing
          could have been billed as a cache read: the pair is ``(prompt_tokens, 0)``
          (ADR-0070 decision 2).
        * **A readable ``cached_tokens``** that does not exceed ``prompt_tokens``: the pair is
          ``(prompt_tokens - cached_tokens, cached_tokens)``.
        * **A details object that is present but unreadable** — not a mapping (``null``
          included), no ``cached_tokens`` key, a malformed or negative figure, or a
          ``cached_tokens`` larger than ``prompt_tokens``, which is a server contradicting itself
          — the pair is ``(UNSUPPORTED, UNSUPPORTED)``.

    The third case is where this function refuses rather than guesses, and it refuses on *both*
    classes together because they are the two halves of one subtraction: a server that sent a
    details object has told us it does cache accounting, so a cache-read ``0`` would be the
    fabricated zero ADR-0016 forbids
    rather than the honest one ADR-0070 licenses, and an ``input_tokens`` of ``prompt_tokens``
    beside an unknown cached figure would not be disjoint from it. Clamping the subtraction at
    zero instead — the shape :class:`~modelrack.testing.FakeProvider` uses for a *scripted*
    count — is rejected here: it would turn a self-contradicting response into a confident
    ``input_tokens`` of ``0`` for a call that certainly had input.
    """
    prompt_tokens = _as_token_count(usage.get("prompt_tokens"))
    if "prompt_tokens_details" not in usage:
        return prompt_tokens, 0
    details = usage.get("prompt_tokens_details")
    cached = (
        _as_token_count(details.get("cached_tokens"))
        if isinstance(details, Mapping)
        else UNSUPPORTED
    )
    if not is_supported(cached) or not is_supported(prompt_tokens) or cached > prompt_tokens:
        return UNSUPPORTED, UNSUPPORTED
    return prompt_tokens - cached, cached


def _read_usage(payload: Mapping[str, Any], *, text: str) -> GenerationUsage:
    """Build a :class:`~modelrack.types.GenerationUsage` from the wire ``usage`` object.

    **What this protocol can bill.** ``prompt_tokens`` and ``completion_tokens`` are the input and
    output classes. Cached input is expressible — ``usage.prompt_tokens_details.cached_tokens`` —
    and is read here and reconciled by :func:`_reconcile_prompt_tokens`. A cache *write* is not
    expressible anywhere in this protocol: there is no field by which a server following this wire
    format could charge for one, so ``cache_write_tokens`` is ``0`` whenever a usage object is
    present. That is a statement about the protocol, not about the response
    (ADR-0070 decision 2). Its known
    limit is recorded in that ADR's *Revisit when*: a provider that bills cache writes under this
    shape without reporting them would make the zero wrong, and would need a per-provider
    override rather than this protocol-level rule.

    **Absent is not empty, and neither is zero.** A response with no usage object — the key
    missing, ``null``, not a mapping, or an empty ``{}`` — reports every class ``UNSUPPORTED``,
    the third of ADR-0070's three cases. The empty mapping belongs with the absent ones for a
    concrete reason: :meth:`OpenAICompatibleProvider.stream` accumulates usage chunks into a dict
    that stays ``{}`` when the stream carried none, and passes ``{"usage": usage_payload}`` here.
    Folding that into "present but without cache detail" would report cache classes of ``0`` for
    a stream that reported no usage at all — a fabricated zero produced by a default value.

    Character, word and byte counts are observations this process can make regardless of what the
    provider counted, so they are always present.
    """
    raw_usage = payload.get("usage")
    usage = raw_usage if isinstance(raw_usage, Mapping) and raw_usage else None
    if usage is None:
        tokens = TokenUsage()
    else:
        input_tokens, cache_read_tokens = _reconcile_prompt_tokens(usage)
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=_as_token_count(usage.get("completion_tokens")),
            cache_write_tokens=0,
            cache_read_tokens=cache_read_tokens,
        )
    return GenerationUsage(
        tokens=tokens,
        output_chars=len(text),
        output_words=len(text.split()),
        output_bytes=len(text.encode("utf-8")),
    )


def _is_context_overflow(message: str, code: str | None) -> bool:
    """Report whether an error names a context-window overflow.

    Checks the structured ``error.code`` first — see the module docstring on why this protocol's
    ``"context_length_exceeded"`` is a real signal where Ollama has only prose to sniff — and
    falls back to the same conservative marker-phrase match for a server that omits the code.
    """
    if code in _CONTEXT_OVERFLOW_CODES:
        return True
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)
