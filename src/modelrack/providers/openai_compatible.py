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
([ADR-0007](../../../docs/adr/0007-provider-abstraction.md) rule 1). Nothing in
:mod:`modelrack.types`, :mod:`modelrack.streaming` or :mod:`modelrack.provider` needed to change to
support this adapter — a fact worth stating plainly because it is the acceptance test the design
itself was under.

**Honest capability declaration is this module's whole point** (spec §11.10, and
[ADR-0007](../../../docs/adr/0007-provider-abstraction.md) rule 2). An OpenAI-compatible chat
completions endpoint has:

* No digest for anything it serves — ``/v1/models`` lists an ``id`` and nothing that identifies
  the bytes behind it, so every identity here is
  :attr:`~baseaicore.IdentityConfidence.NAME_ONLY`, permanently
  ([ADR-0024 §2](../../../docs/adr/0024-canonical-id-and-model-references.md)).
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
:func:`_iter_sse_events` is the whole parser — deliberately small, because the format itself is.
Reuses :func:`modelrack.providers._http.iter_capped_lines` on the raw lines underneath it, the same
per-chunk size cap :mod:`modelrack.providers.ollama` applies to NDJSON lines.

**Tool calls arrive pre-fragmented even in a single chunk.** Unlike Ollama, whose ``arguments`` is
already a parsed JSON object, this protocol's ``function.arguments`` is always a JSON *string* —
one that streams a few characters at a time across many chunks and is not valid JSON until the
last fragment lands. :attr:`~modelrack.types.ToolCall.raw_arguments` exists in the Phase-1
vocabulary precisely for this shape; :func:`_tool_call_from_parts` is the one place a fragment
that never became valid JSON is preserved rather than discarded, so a malformed-arguments failure
is diagnosable rather than silently an empty mapping.
"""

from __future__ import annotations

import json
import logging
import ssl
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx
from baseaicore import (
    UNSUPPORTED,
    ModelDescriptor,
    ModelIdentity,
    ProviderKind,
    TokenUsage,
    elapsed_ms,
    monotonic_ns,
    utc_now,
)

from modelrack.errors import (
    CapabilityUnsupported,
    ContextLimitExceeded,
    GenerationCancelled,
    ModelNotFound,
    ProviderProtocolError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from modelrack.provider import (
    LoadResult,
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    ResidentModel,
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
from modelrack.streaming import (
    StreamCompleted,
    StreamFailed,
    TokenDelta,
    ToolCallDelta,
)
from modelrack.types import (
    FinishReason,
    GenerationResult,
    GenerationUsage,
    Message,
    ResponseFormatKind,
    Role,
    Timing,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from datetime import datetime
    from typing import NoReturn

    from baseaicore import RuntimeProfile

    from modelrack.streaming import StreamEvent
    from modelrack.types import GenerationRequest, ResponseFormat, ToolCall, ToolDefinition

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
tested" [ADR-0007](../../../docs/adr/0007-provider-abstraction.md) rule 2 warns against).
"""

_REQUEST_HEADERS: Final[dict[str, str]] = {"Content-Type": "application/json"}

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

_FINISH_REASON_MAP: Final[dict[str, FinishReason]] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


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
        clock: Where a :class:`~baseaicore.ModelDescriptor`'s ``observed_at`` comes from, injected
            so a test can freeze it (coding standards §5).

    Raises:
        ValidationError: If ``base_url`` has no host or uses a scheme other than ``http``/
            ``https``.
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
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        """Validate the base URL and construct or adopt the pooled HTTP client."""
        validated_url, is_remote = validate_base_url(base_url)
        self._base_url = validated_url
        self._is_remote = is_remote
        self._max_response_bytes = max_response_bytes
        self._max_chunk_bytes = max_chunk_bytes
        self._clock = clock
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
            :attr:`~modelrack.provider.ProviderStatus.UNAVAILABLE` — never a raised exception —
            when the provider cannot be reached at all.
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

    def list_models(self) -> Sequence[ModelDescriptor]:
        """List the models the server is serving.

        Returns:
            One descriptor per model. ``/v1/models`` carries an ``id`` and little else on most
            servers, so every field beyond ``identity``, ``observed_at`` and ``raw`` is
            :data:`~baseaicore.UNSUPPORTED` or ``None`` — this adapter never invents architecture
            metadata a discovery endpoint did not report (spec §11.2, "limited metadata").

        Raises:
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        entries = self._model_entries(self._get_json("/v1/models"))
        observed_at = self._clock()
        return tuple(_build_descriptor(entry, observed_at=observed_at) for entry in entries)

    def inspect_model(self, identity: ModelIdentity) -> ModelDescriptor:
        """Fetch metadata for one model.

        Args:
            identity: The model to inspect, matched on
                :attr:`~baseaicore.ModelIdentity.provider_model_name`.

        Returns:
            The descriptor.

        Raises:
            ModelNotFound: If the server does not have it.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        descriptors = self.list_models()
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

    def resolve(self, reference: str) -> ModelIdentity:
        """Resolve a user-supplied model reference to a concrete identity.

        Tries an exact ``id`` match, then a unique prefix over every known ``id``. Unlike
        :meth:`~modelrack.providers.ollama.OllamaProvider.resolve`, there is no ``:latest``-suffix
        convention to try — that is Ollama's own tagging scheme, not part of this protocol.

        Args:
            reference: What the user typed.

        Returns:
            The resolved identity — always :attr:`~baseaicore.IdentityConfidence.NAME_ONLY`.

        Raises:
            ModelNotFound: If nothing matches, or a prefix matches more than one model.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        descriptors = self.list_models()
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
                ([ADR-0007](../../../docs/adr/0007-provider-abstraction.md) rule 2).
        """
        self._require_capability("force_unload", "load a model on demand")

    def unload(self, identity: ModelIdentity) -> bool:
        """Refuse: this protocol has no endpoint to evict a model on demand.

        Raises:
            CapabilityUnsupported: Always — ``force_unload`` is declared ``False``.
        """
        self._require_capability("force_unload", "evict a model on demand")

    def list_resident(self) -> Sequence[ResidentModel]:
        """Refuse: this protocol has no endpoint to report what is currently loaded.

        Raises:
            CapabilityUnsupported: Always — ``residency_query`` is declared ``False``.
        """
        self._require_capability("residency_query", "report which models are resident")

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
        message, code = _extract_error(payload)
        if message is not None:
            raise self._build_message_error(message, code=code, status_code=200)
        choice = _first_choice(payload)
        response_message = choice.get("message")
        response_message = response_message if isinstance(response_message, Mapping) else {}
        text = response_message.get("content")
        text = text if isinstance(text, str) else ""
        call_prefix = f"oai-{start_ns}"
        tool_calls = _parse_tool_calls(response_message.get("tool_calls"), call_prefix=call_prefix)
        return GenerationResult(
            text=text,
            identity=request.identity,
            finish_reason=_finish_reason_for(
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
                events.

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
        start_ns = monotonic_ns()
        prepared = self._client.build_request(
            "POST", "/v1/chat/completions", json=body, timeout=self._timeout_for(request)
        )
        try:
            response = self._client.send(prepared, stream=True)
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc, base_url=self._base_url) from exc
        try:
            if response.status_code >= 400:
                self._raise_for_status(
                    response, model_reference=request.identity.provider_model_name
                )
        except BaseException:
            response.close()
            raise
        return self._walk(request, response, start_ns)

    def _walk(
        self, request: GenerationRequest, response: httpx.Response, start_ns: int
    ) -> Iterator[StreamEvent]:
        """Drain one SSE stream, owning ``response`` for its entire remaining lifetime.

        See :meth:`modelrack.providers.ollama.OllamaProvider._walk` for why this is a generator:
        ``GeneratorExit`` on abandonment runs the same ``finally`` a normal return does, which is
        what makes draining, breaking early and walking away all close the connection identically.
        """
        cancel = request.cancel
        answer_parts: list[str] = []
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
            for data in _iter_sse_events(lines):
                if cancel is not None and cancel.is_cancelled:
                    yield self._cancelled("".join(answer_parts))
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
                        partial_text="".join(answer_parts),
                    )
                    return
                if not isinstance(payload, dict):
                    yield StreamFailed(
                        error=ProviderProtocolError(
                            f"A streamed event from {self._base_url} was not a JSON object.",
                            details={"base_url": self._base_url, "body": truncated_text(data)},
                        ),
                        partial_text="".join(answer_parts),
                    )
                    return
                message, code = _extract_error(payload)
                if message is not None:
                    yield StreamFailed(
                        error=self._build_message_error(
                            message, code=code, status_code=response.status_code
                        ),
                        partial_text="".join(answer_parts),
                    )
                    return

                usage = payload.get("usage")
                if isinstance(usage, Mapping):
                    usage_payload = usage
                choice = _first_choice(payload)
                delta = choice.get("delta")
                delta = delta if isinstance(delta, Mapping) else {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    first_delta_ns = first_delta_ns or monotonic_ns()
                    answer_parts.append(content)
                    yield TokenDelta(text=content, index=delta_index)
                    delta_index += 1
                raw_tool_calls = delta.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    for fragment in raw_tool_calls:
                        if not isinstance(fragment, Mapping):
                            continue
                        first_delta_ns = first_delta_ns or monotonic_ns()
                        call_index = _tool_call_index(fragment, fallback=len(tool_order))
                        if call_index not in tool_fragments:
                            tool_fragments[call_index] = {"id": None, "name": None, "arguments": []}
                            tool_order.append(call_index)
                        entry = tool_fragments[call_index]
                        fragment_id, fragment_name, fragment_arguments = _tool_call_fragment(
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
                    partial_text="".join(answer_parts),
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
                yield self._cancelled("".join(answer_parts))
                return
            tool_calls = tuple(
                _tool_call_from_parts(
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
            text = "".join(answer_parts)
            yield StreamCompleted(
                result=GenerationResult(
                    text=text,
                    identity=request.identity,
                    finish_reason=_finish_reason_for(
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
                partial_text="".join(answer_parts),
            )
        except ProviderProtocolError as exc:
            # Raised by `iter_capped_lines` when a line exceeds the per-chunk cap — already a
            # typed error, delivered as a terminal event rather than re-raised because the stream
            # has already begun (spec §13, and modelrack.streaming's own terminal-event rule).
            yield StreamFailed(error=exc, partial_text="".join(answer_parts))
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
        message, code = _extract_error(payload)
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

    def _require_capability(self, capability: str, action: str) -> NoReturn:
        """Raise :class:`CapabilityUnsupported` naming ``capability``; it is never declared."""
        raise CapabilityUnsupported(
            f"This provider does not declare {capability!r} and cannot {action}. Check "
            "capabilities() and branch, rather than assuming.",
            details={"capability": capability},
        )

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
        if request.runtime_profile.context_size is not None:
            self._require_capability("context_configurable", "serve a caller-chosen context")
        messages = request.messages or (Message(role=Role.USER, content=request.prompt or ""),)
        body: dict[str, Any] = {
            "model": request.identity.provider_model_name,
            "messages": [_message_payload(message) for message in messages],
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
            body["repetition_penalty"] = sampling.repeat_penalty
        if request.tools:
            body["tools"] = _request_tool_definitions(request.tools)
        if request.response_format is not None:
            body["response_format"] = _response_format_payload(request.response_format)
        # provider_options is the caller's own escape hatch (spec §12) and is merged last, so it
        # wins on any overlapping key — the same rule _ollama_wire.generation_options documents.
        body.update(request.runtime_profile.provider_options)
        return body


def _identity_for(model_id: str) -> ModelIdentity:
    """Build the identity for one served model — always name-only.

    No OpenAI-compatible discovery endpoint reports a content digest for what it is serving, so
    every identity from this adapter carries :attr:`~baseaicore.IdentityConfidence.NAME_ONLY`
    unconditionally, never a fabricated one
    ([ADR-0024 §2](../../../docs/adr/0024-canonical-id-and-model-references.md)).
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


def _first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``payload["choices"][0]``, or ``{}`` when the array is missing or empty."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return choices[0]
    return {}


def _message_payload(message: Message) -> dict[str, Any]:
    """Build one OpenAI-shaped chat message from a :class:`~modelrack.types.Message`."""
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, sort_keys=True),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    return payload


def _request_tool_definitions(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """Build the OpenAI-shaped tool list from :class:`~modelrack.types.ToolDefinition` values."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }
        for tool in tools
    ]


def _response_format_payload(response_format: ResponseFormat) -> dict[str, Any]:
    """Build the OpenAI-shaped ``response_format`` object for a text/JSON/schema request."""
    if response_format.kind is ResponseFormatKind.JSON:
        return {"type": "json_object"}
    if response_format.kind is ResponseFormatKind.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "modelrack_response",
                "schema": dict(response_format.schema or {}),
                "strict": True,
            },
        }
    return {"type": "text"}


def _tool_call_from_parts(
    *, call_id: str | None, name: str | None, raw_arguments: str | None, fallback_id: str
) -> ToolCall:
    """Assemble one :class:`~modelrack.types.ToolCall` from its wire pieces.

    Shared between the non-streaming path (one complete entry) and the streaming path (fragments
    accumulated across many chunks) — both eventually have the same three pieces: an id the
    provider may or may not have sent, a name, and an ``arguments`` string that may or may not be
    valid JSON. A string that will not parse is kept as ``raw_arguments`` and yields empty
    ``arguments`` rather than raising: a malformed-arguments response is a real failure mode this
    package's :mod:`modelrack.testing` scripts on purpose, and dropping it here would make that
    scripted case untestable against a real adapter.
    """
    from modelrack.types import ToolCall  # noqa: PLC0415 — avoids a module-level cycle

    arguments: dict[str, Any] = {}
    if raw_arguments:
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            arguments = parsed
    return ToolCall(
        id=call_id if call_id else fallback_id,
        name=name if name else "unknown",
        arguments=arguments,
        raw_arguments=raw_arguments,
    )


def _parse_tool_calls(raw_calls: Any, *, call_prefix: str) -> tuple[ToolCall, ...]:  # noqa: ANN401
    """Parse a non-streaming response's ``message.tool_calls`` into the vocabulary's tuple."""
    if not isinstance(raw_calls, list):
        return ()
    calls: list[ToolCall] = []
    for offset, entry in enumerate(raw_calls):
        if not isinstance(entry, Mapping):
            continue
        call_id = entry.get("id")
        function = entry.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name")
        raw_arguments = function.get("arguments")
        calls.append(
            _tool_call_from_parts(
                call_id=call_id if isinstance(call_id, str) and call_id else None,
                name=name if isinstance(name, str) and name else None,
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
                fallback_id=f"{call_prefix}-{offset}",
            )
        )
    return tuple(calls)


def _tool_call_index(fragment: Mapping[str, Any], *, fallback: int) -> int:
    """Return a streamed tool-call fragment's ``index``, or ``fallback`` if it is absent/invalid."""
    index = fragment.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return fallback
    return index


def _tool_call_fragment(
    fragment: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Return ``(id, name, arguments_fragment)`` from one streamed tool-call delta entry."""
    call_id = fragment.get("id")
    function = fragment.get("function")
    function = function if isinstance(function, Mapping) else {}
    name = function.get("name")
    arguments = function.get("arguments")
    return (
        call_id if isinstance(call_id, str) and call_id else None,
        name if isinstance(name, str) and name else None,
        arguments if isinstance(arguments, str) and arguments else None,
    )


def _finish_reason_for(raw: Any, *, has_tool_calls: bool) -> FinishReason:  # noqa: ANN401
    """Map the wire ``finish_reason`` string to the vocabulary's finish reason.

    A message carrying tool calls wins regardless of what ``finish_reason`` said, the same
    defensive precedence :func:`modelrack.providers._ollama_wire.finish_reason_for` applies —
    :class:`~modelrack.types.GenerationResult` requires ``TOOL_CALLS`` whenever tool calls are
    present, and a server that populated ``tool_calls`` but reported a stale ``finish_reason``
    should not be able to violate that invariant.
    """
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    if isinstance(raw, str):
        return _FINISH_REASON_MAP.get(raw, FinishReason.UNKNOWN)
    return FinishReason.UNKNOWN


def _as_token_count(value: object) -> Any:  # noqa: ANN401 — mirrors baseaicore.TokenCount
    """Return ``value`` if it is a non-negative whole number, else ``UNSUPPORTED``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return UNSUPPORTED
    return value


def _read_usage(payload: Mapping[str, Any], *, text: str) -> GenerationUsage:
    """Build a :class:`~modelrack.types.GenerationUsage` from the wire ``usage`` object.

    ``prompt_tokens``/``completion_tokens`` map to the billing vocabulary's disjoint classes; this
    protocol reports no cache-aware billing, so both cache classes stay
    :data:`~baseaicore.UNSUPPORTED` rather than an invented zero
    ([ADR-0016](../../../docs/adr/0016-unavailable-is-not-zero.md)). Character, word and byte
    counts are observations this process can make regardless of what the provider counted.
    """
    usage = payload.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    return GenerationUsage(
        tokens=TokenUsage(
            input_tokens=_as_token_count(usage.get("prompt_tokens")),
            output_tokens=_as_token_count(usage.get("completion_tokens")),
        ),
        output_chars=len(text),
        output_words=len(text.split()),
        output_bytes=len(text.encode("utf-8")),
    )


def _extract_error(payload: Any) -> tuple[str | None, str | None]:  # noqa: ANN401 — provider JSON
    """Return ``(message, code)`` from a parsed error body, or ``(None, None)`` if there is none.

    Handles both documented error shapes: ``{"error": {"message": ..., "code": ...}}`` (the
    OpenAI-originated convention every server here follows) and the bare-string
    ``{"error": "..."}`` a few minimal servers still send.
    """
    if not isinstance(payload, Mapping):
        return None, None
    error = payload.get("error")
    if isinstance(error, str) and error:
        return error, None
    if isinstance(error, Mapping):
        message = error.get("message")
        code = error.get("code")
        return (
            message if isinstance(message, str) and message else None,
            code if isinstance(code, str) else None,
        )
    return None, None


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


def _iter_sse_events(lines: Iterable[str]) -> Iterator[str]:
    """Group raw SSE lines into event ``data`` payloads.

    Implements just the subset of the `Server-Sent Events grammar
    <https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation>`_
    this protocol uses: a ``data:`` field line (its value joined across consecutive ``data:``
    lines with ``\\n``, per the spec), a blank line dispatching whatever has been buffered, and a
    ``:``-prefixed line — a comment, sent by some servers as a keep-alive — ignored outright. Any
    other field name (``event:``, ``id:``, ``retry:``) is ignored: nothing this adapter reads uses
    them. A final event with no trailing blank line is still dispatched, defensively, since a
    stream ending exactly on ``data: [DONE]`` with no newline after it is a shape worth surviving
    rather than silently dropping the last event.
    """
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[len("data:") :]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)
