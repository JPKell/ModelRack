"""Provider adapter — a real local runtime, reached over Ollama's HTTP API.

Imports :mod:`baseaicore`, this package's own types, ``httpx`` (through :mod:`_http`) and the
standard library; performs real network I/O, which is why every test in
``tests/unit/test_ollama_adapter.py`` runs against a recorded transport rather than a live
process, and why the socket guard in ``tests/conftest.py`` would fail any test here that forgot
to mock one.

The second adapter, after :mod:`modelrack.providers.fake` — first in shipping order, not in
architectural importance (ADR-0007 rule 6). This
module is where the vocabulary Phase 1 designed and Phase 2 exercised against a fake meets a real
runtime's actual, occasionally inconvenient wire format for the first time.

**Where things live**, mirroring the split :mod:`modelrack.providers.fake` established: this
module owns the client, the HTTP calls and the streaming state machine — everything that touches
the network or a clock — and delegates every piece of parsing to
:mod:`modelrack.providers._ollama_wire`, which is pure and independently testable.
:mod:`modelrack.providers._http` supplies the transport plumbing every real adapter shares
(client construction, size caps, exception translation), so Phase 4's OpenAI-compatible adapter
does not re-derive it.

**Two hard requirements this module exists to satisfy, both named in its own tests:**

* **NDJSON chunk boundaries never corrupt a delta.** A streamed response is one JSON object per
  line, and neither the line breaks nor a multi-byte character inside one line is guaranteed to
  land inside a single TCP read. :meth:`httpx.Response.iter_lines` is used for exactly this
  reason — verified directly against this httpx version to reassemble a UTF-8 character split
  across two raw byte chunks before this module ever sees a line — rather than any manual
  byte-buffering, which would have to reimplement that reassembly and would be the one place a
  subtle decoding bug could hide.
* **Backend and client timings never merge.**
  :func:`~modelrack.providers._ollama_wire.read_backend_timing` reads only what Ollama reported
  about its own work (``load_duration``, ``prompt_eval_duration``, ``eval_duration``,
  ``total_duration`` — all nanoseconds, converted once, in one place). This
  module measures ``client_wall_ms`` and ``client_ttft_ms`` itself, with
  :func:`baseaicore.monotonic_ns`, from outside the call. Every :class:`~modelrack.types.Timing`
  built here sets backend fields from one source and client fields from the other, never both from
  the same reading.

**Name-based, defensive parsing** is this module's whole strategy against
risk register E1 ("Ollama API changes"): every
field is read by name, a missing or reshaped one degrades to
:data:`~baseaicore.UNSUPPORTED` rather than raising or guessing, and every adapter method still
raises a typed error for what it cannot parse at all — never a raw ``httpx`` exception
([spec §11.7](../../../docs/packages/modelrack/spec.md)).
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
    ModelIdentity,
    ProviderKind,
    elapsed_ms,
    is_supported,
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
from modelrack.providers._ollama_wire import (
    as_measurement,
    build_descriptor,
    build_resident_model,
    extract_error_message,
    find_context_overflow,
    finish_reason_for,
    generation_options,
    identity_for,
    parse_tool_calls,
    read_backend_timing,
    read_usage,
    request_tool_definitions,
)
from modelrack.streaming import (
    StreamCompleted,
    StreamFailed,
    ThinkingDelta,
    TokenDelta,
    ToolCallDelta,
)
from modelrack.types import GenerationResult, ResponseFormatKind, Timing

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import datetime

    from baseaicore import Measurement, ModelDescriptor, RuntimeProfile

    from modelrack.streaming import StreamEvent
    from modelrack.types import GenerationRequest, ToolCall

__all__ = ["OllamaProvider"]

logger = logging.getLogger(__name__)

_CAPABILITIES: Final[ProviderCapabilities] = ProviderCapabilities(
    streaming=True,
    tool_calling=True,
    structured_output=True,
    json_mode=True,
    token_counts=True,
    token_level_chunks=True,
    thinking_control=True,
    logprobs=False,
    force_unload=True,
    residency_query=True,
    kv_metrics=False,
    context_configurable=True,
    embedding=False,
)
"""What every Ollama server this adapter has been built against can do.

``logprobs`` and ``kv_metrics`` stay ``False``: neither ``/api/chat`` nor ``/api/generate``
exposes per-token log probabilities or KV-cache counters in any documented response field, so
declaring either would be exactly the "capability nobody tested"
ADR-0007 rule 2 warns against. ``embedding`` is
``False`` because embeddings are out of this package's scope until
[spec §21](../../../docs/packages/modelrack/spec.md) — Ollama has ``/api/embed``, but nothing in
the ``Provider`` protocol calls it yet.
"""

_REQUEST_HEADERS: Final[dict[str, str]] = {"Content-Type": "application/json"}

# Ollama accepts a duration string ("5m"), a number of seconds, or -1 for "never expire" as
# `keep_alive`; 0 is its documented spelling for "evict immediately", which unload() needs
# regardless of what a caller's runtime profile otherwise says about keep-alive.
_UNLOAD_KEEP_ALIVE: Final[int] = 0


class OllamaProvider:
    """A real :class:`~modelrack.provider.Provider`, reached over Ollama's HTTP API.

    Owns one pooled :class:`httpx.Client` for its whole lifetime (spec §15: "connection pooling
    via a shared ``httpx.Client``"). Not a context manager itself — callers that own the client's
    lifecycle close it directly, or inject one they already manage via ``client=``.

    Every method raises the typed errors in :mod:`modelrack.errors`, never a raw ``httpx``
    exception (spec §11.7); every unavailable measurement is
    :data:`~baseaicore.UNSUPPORTED`, never ``0``
    (ADR-0016).

    Args:
        base_url: Where Ollama is listening. Validated to an ``http``/``https`` scheme with a
            host; a non-loopback host is permitted but flagged as remote on every
            :class:`~modelrack.provider.ProviderHealth` result (spec §14).
        timeout: The default applied to a call that names none. A plain number sets connect,
            read, write and pool limits uniformly; pass a full :class:`httpx.Timeout` for
            distinct ones (spec §12). Never ``None`` — there is no "no timeout" (spec §14).
        headers: Sent on every request. Ollama needs no credential by default; this exists for a
            reverse proxy or an authenticating gateway placed in front of it.
        client: An already-constructed :class:`httpx.Client` to use instead of building one —
            for a caller that wants to share connection pooling with other traffic, or a test
            that injects a mock transport. When supplied, ``base_url``, ``timeout``, ``headers``
            and ``verify`` describe it for diagnostics but do not reconstruct it.
        verify: TLS verification, passed to :class:`httpx.Client` unchanged. Irrelevant to the
            default loopback ``http://`` URL; present for a remote ``https://`` deployment.
        max_response_bytes: The total-body cap for a non-streamed response (spec §14, default
            64 MiB).
        max_chunk_bytes: The per-line cap for a streamed response (spec §14, default 8 MiB).
        clock: Where a :class:`~baseaicore.ModelDescriptor`'s ``observed_at`` comes from — the
            one reading of the real clock this adapter takes, injected so a test can freeze it
            (coding standards §5).

    Raises:
        ValidationError: If ``base_url`` has no host or uses a scheme other than ``http``/
            ``https``.
    """

    kind: ProviderKind = ProviderKind.OLLAMA

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
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
        self._client = client or build_client(
            base_url=validated_url,
            timeout=timeout,
            headers={**_REQUEST_HEADERS, **(headers or {})},
            verify=verify,
        )

    # ------------------------------------------------------------------------------- protocol

    def health(self) -> ProviderHealth:
        """Probe the provider and report whether it can be used.

        Returns:
            :attr:`~modelrack.provider.ProviderStatus.OK` with the server's version and model
            count on success; :attr:`~modelrack.provider.ProviderStatus.UNAVAILABLE` — never a
            raised exception — when the provider cannot be reached at all, because "is it up?"
            is a question whose negative answer is not exceptional (see
            :meth:`~modelrack.provider.Provider.health`'s own contract).
        """
        start_ns = monotonic_ns()
        try:
            version_response = self._client.get("/api/version")
            version_response.raise_for_status()
            version_payload = read_capped_json(
                version_response, max_bytes=self._max_response_bytes, base_url=self._base_url
            )
            models = self._list_tags()
        except (httpx.HTTPError, ProviderProtocolError):
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                base_url=self._base_url,
                is_remote=self._is_remote,
                detail="unreachable",
                latency_ms=elapsed_ms(start_ns),
            )
        version = version_payload.get("version") if isinstance(version_payload, dict) else None
        version = version if isinstance(version, str) else None
        return ProviderHealth(
            status=ProviderStatus.OK,
            base_url=self._base_url,
            is_remote=self._is_remote,
            detail=f"ollama {version or '(unknown version)'}, {len(models)} models",
            provider_version=version,
            model_count=len(models),
            latency_ms=elapsed_ms(start_ns),
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report what this adapter can do.

        Returns:
            The static declaration in this module — cheap and non-probing, as the protocol
            requires: no request is made to answer this.
        """
        return _CAPABILITIES

    def list_models(self) -> Sequence[ModelDescriptor]:
        """List the models Ollama is serving, each enriched with full architecture metadata.

        Calls ``/api/show`` once per model after ``/api/tags`` — the reason spec §15 budgets
        cold ``list_models`` at seconds rather than milliseconds ("dominated by per-model
        ``show`` calls"): ``/api/tags`` alone carries no layer count, no head counts and no
        embedding width, and those are exactly what FreeWeight's KV-cache benchmark needs.

        Returns:
            One descriptor per model. Empty when nothing is pulled, which is a real state.

        Raises:
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        entries = self._list_tags()
        return tuple(self._describe_entry(entry) for entry in entries)

    def inspect_model(self, identity: ModelIdentity) -> ModelDescriptor:
        """Fetch full metadata for one model.

        Two calls, unavoidably: ``/api/tags`` for the current digest (so a retag is caught rather
        than echoed back — spec §11.8) and ``/api/show`` for everything else, since neither
        endpoint alone carries both.

        Args:
            identity: The model to inspect, matched on
                :attr:`~baseaicore.ModelIdentity.provider_model_name`.

        Returns:
            The descriptor, with both raw payloads preserved in
            :attr:`~baseaicore.ModelDescriptor.raw`.

        Raises:
            ModelNotFound: If Ollama does not have it.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        entries = self._list_tags()
        entry = self._find_entry(entries, identity.provider_model_name)
        return self._describe_entry(entry)

    def resolve(self, reference: str) -> ModelIdentity:
        """Resolve a user-supplied model reference to a concrete identity.

        Tries, in order: an exact name; Ollama's own ``:latest``-suffix convention for a bare name
        with no tag; a unique prefix over every known name. An ambiguous prefix is refused rather
        than resolved by picking one — a caller who meant a different model than the one chosen
        would have no way to notice (spec §11.8).

        Args:
            reference: What the user typed.

        Returns:
            The resolved identity, digest-confident when Ollama reports one.

        Raises:
            ModelNotFound: If nothing matches, or a prefix matches more than one model.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        entries = self._list_tags()
        entry = self._resolve_entry(entries, reference)
        entry_name = str(entry.get("name") or entry.get("model") or "")
        identity = identity_for(entry_name, entry.get("digest"))[0]
        if reference != identity.provider_model_name:
            logger.debug(
                "ollama.model.resolved",
                extra={
                    "reference": reference,
                    "resolved_to": identity.provider_model_name,
                    "identity_confidence": identity.identity_confidence.value,
                },
            )
        return identity

    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult:
        """Ask Ollama to load a model under a runtime profile.

        Checks ``/api/ps`` first so :attr:`~modelrack.provider.LoadResult.already_resident` is
        honest — a warm model measured as a cold start would be a figure an order of magnitude
        wrong, which is exactly the distinction a caller sets FreeWeight's cold/warm marker from.
        Only when the model is not already resident is a load actually issued: Ollama has no
        dedicated "load" endpoint, so a ``POST /api/generate`` naming a model but no ``prompt``
        is the documented way to preload one without generating anything.

        Args:
            identity: The model to load.
            profile: How it should be loaded and served.

        Returns:
            What happened. ``load_ms`` prefers Ollama's own reported ``load_duration``; when the
            provider's response does not carry one (some versions omit it on a load-only call),
            this process's own observed wall time is used instead — never ``0``.

        Raises:
            ModelNotFound: If Ollama does not have it.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        resident = self._list_ps()
        if self._find_optional_entry(resident, identity.provider_model_name) is not None:
            return LoadResult(
                identity=identity,
                already_resident=True,
                load_ms=UNSUPPORTED,
                profile_hash=profile.profile_hash,
            )
        body = self._load_body(identity, profile)
        start_ns = monotonic_ns()
        payload = self._post_json(
            "/api/generate", body, model_reference=identity.provider_model_name
        )
        backend = read_backend_timing(payload if isinstance(payload, dict) else {})
        load_ms = self._first_supported(
            backend.backend_load_ms, backend.backend_total_ms, elapsed_ms(start_ns)
        )
        return LoadResult(
            identity=identity,
            already_resident=False,
            load_ms=load_ms,
            profile_hash=profile.profile_hash,
        )

    def unload(self, identity: ModelIdentity) -> bool:
        """Ask Ollama to evict a model from memory immediately.

        Args:
            identity: The model to evict.

        Returns:
            ``True`` if it was resident and has been evicted, ``False`` if it was not resident to
            begin with — the state the caller wanted, not a failure, so no request is even made.

        Raises:
            ModelNotFound: If Ollama does not have it.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        resident = self._list_ps()
        if self._find_optional_entry(resident, identity.provider_model_name) is None:
            return False
        self._post_json(
            "/api/generate",
            {"model": identity.provider_model_name, "keep_alive": _UNLOAD_KEEP_ALIVE},
            model_reference=identity.provider_model_name,
        )
        return True

    def list_resident(self) -> Sequence[ResidentModel]:
        """List the models Ollama currently holds in memory.

        Returns:
            One entry per resident model, sorted by name so two calls agree. Ollama reports one
            VRAM figure per model rather than per device; that figure is passed through as
            ``vram_bytes`` unchanged — this package summing across devices itself is what
            ADR-0027 forbids, not a provider
            choosing to report one number of its own.

        Raises:
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        entries = self._list_ps()
        built = [build_resident_model(entry) for entry in entries]
        built.sort(key=lambda item: item[0].provider_model_name)
        return tuple(
            ResidentModel(identity=identity, vram_bytes=vram, total_bytes=total, expires_at=expires)
            for identity, vram, total, expires in built
        )

    # ------------------------------------------------------------------------ generation

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one non-streamed generation and return the complete result.

        A cancellation token on the request has no effect: a blocking round trip offers no
        boundary at which it could take effect (spec §13).

        Args:
            request: What to generate, and how.

        Returns:
            The complete outcome. ``client_ttft_ms`` is ``UNSUPPORTED`` — there is no first-token
            moment to observe on a blocking call.

        Raises:
            ModelNotFound: If the requested model is not available.
            ContextLimitExceeded: If Ollama reports the request needed more context than served.
            ProviderRejected: If Ollama understood the request and refused it.
            ProviderProtocolError: If the response cannot be parsed.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        path, body = self._build_request(request, stream=False)
        start_ns = monotonic_ns()
        payload = self._post_json(
            path,
            body,
            model_reference=request.identity.provider_model_name,
            context_size=request.runtime_profile.context_size,
            timeout=request.timeout_seconds,
        )
        if not isinstance(payload, dict):
            raise ProviderProtocolError(
                f"The provider at {self._base_url} returned something other than a JSON object.",
                details={"base_url": self._base_url, "body": truncated_text(json.dumps(payload))},
            )
        message = extract_error_message(payload)
        if message is not None:
            raise self._build_message_error(
                message, status_code=200, context_size=request.runtime_profile.context_size
            )
        text, thinking_text, raw_tool_calls, saw_thinking = self._extract_content(payload)
        tool_calls = parse_tool_calls(raw_tool_calls, call_prefix=f"ollama-{start_ns}")
        return self._build_result(
            request,
            text=text,
            thinking=thinking_text if saw_thinking else UNSUPPORTED,
            tool_calls=tool_calls,
            terminal_payload=payload,
            client_wall_ms=elapsed_ms(start_ns),
            client_ttft_ms=UNSUPPORTED,
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        """Run one generation, yielding events as they arrive over Ollama's NDJSON stream.

        Everything that can fail before a byte of the stream arrives — connection, model lookup,
        a rejected request — raises from this call. Every failure after that, including the
        caller's own cancellation, is delivered as
        :class:`~modelrack.streaming.StreamFailed`, the terminal event, with the connection
        closed either way: the response this call opens is owned by the returned generator's
        ``finally`` block for the rest of its life, so draining it, breaking out of it early or
        abandoning it all close the same way.

        Args:
            request: What to generate, and how. Its ``cancel`` token is honoured within one
                NDJSON line.

        Yields:
            Deltas, then one terminal event.

        Raises:
            ModelNotFound: If the requested model is not available.
            ContextLimitExceeded: If Ollama reports the request needed more context than served,
                before any content arrives.
            ProviderRejected: If Ollama understood the request and refused it before streaming.
            ProviderUnavailable: If the provider cannot be reached.
            ProviderTimeout: If it does not answer in time.
        """
        path, body = self._build_request(request, stream=True)
        start_ns = monotonic_ns()
        prepared = self._client.build_request(
            "POST", path, json=body, timeout=self._timeout_for(request)
        )
        try:
            response = self._client.send(prepared, stream=True)
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc, base_url=self._base_url) from exc
        try:
            if response.status_code >= 400:
                self._raise_for_status(
                    response,
                    model_reference=request.identity.provider_model_name,
                    context_size=request.runtime_profile.context_size,
                )
        except BaseException:
            response.close()
            raise
        return self._walk(request, response, start_ns)

    def _walk(
        self, request: GenerationRequest, response: httpx.Response, start_ns: int
    ) -> Iterator[StreamEvent]:
        """Drain one NDJSON stream, owning ``response`` for its entire remaining lifetime.

        The generator this method returns is what makes the connection-closing guarantee real:
        Python calls ``.close()`` on an abandoned generator, which raises ``GeneratorExit`` at
        whatever ``yield`` it is paused on, and the ``finally`` below runs on that path exactly as
        it does on a normal return — so draining the stream, breaking out of the caller's loop
        early, and never touching it again all release the connection the same way.
        """
        cancel = request.cancel
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        saw_thinking_key = False
        tool_calls: tuple[ToolCall, ...] = ()
        first_delta_ns: int | None = None
        delta_index = 0
        terminal_payload: dict[str, Any] | None = None
        try:
            lines = iter_capped_lines(
                response.iter_lines(),
                max_chunk_bytes=self._max_chunk_bytes,
                base_url=self._base_url,
            )
            for line in lines:
                if not line.strip():
                    continue
                if cancel is not None and cancel.is_cancelled:
                    yield self._cancelled("".join(answer_parts))
                    return
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    yield StreamFailed(
                        error=ProviderProtocolError(
                            f"A streamed line from {self._base_url} was not valid JSON.",
                            details={"base_url": self._base_url, "body": truncated_text(line)},
                        ),
                        partial_text="".join(answer_parts),
                    )
                    return
                if not isinstance(payload, dict):
                    yield StreamFailed(
                        error=ProviderProtocolError(
                            f"A streamed line from {self._base_url} was not a JSON object.",
                            details={"base_url": self._base_url, "body": truncated_text(line)},
                        ),
                        partial_text="".join(answer_parts),
                    )
                    return
                message = extract_error_message(payload)
                if message is not None:
                    yield StreamFailed(
                        error=self._build_message_error(
                            message,
                            status_code=response.status_code,
                            context_size=request.runtime_profile.context_size,
                        ),
                        partial_text="".join(answer_parts),
                    )
                    return

                text_piece, thinking_piece, raw_tool_calls, saw_thinking = self._extract_content(
                    payload
                )
                if saw_thinking:
                    saw_thinking_key = True
                if thinking_piece:
                    first_delta_ns = first_delta_ns or monotonic_ns()
                    thinking_parts.append(thinking_piece)
                    yield ThinkingDelta(text=thinking_piece, index=delta_index)
                    delta_index += 1
                if text_piece:
                    first_delta_ns = first_delta_ns or monotonic_ns()
                    answer_parts.append(text_piece)
                    yield TokenDelta(text=text_piece, index=delta_index)
                    delta_index += 1
                if raw_tool_calls:
                    first_delta_ns = first_delta_ns or monotonic_ns()
                    batch = parse_tool_calls(
                        raw_tool_calls,
                        call_prefix=f"ollama-{start_ns}",
                        start_index=len(tool_calls),
                    )
                    tool_calls = tool_calls + batch
                    for call_index, call in enumerate(batch):
                        yield ToolCallDelta(
                            call_index=call_index, id=call.id, name=call.name, index=delta_index
                        )
                        delta_index += 1
                        arguments_text = json.dumps(call.arguments, sort_keys=True)
                        yield ToolCallDelta(
                            call_index=call_index,
                            arguments_fragment=arguments_text,
                            index=delta_index,
                        )
                        delta_index += 1

                if payload.get("done") is True:
                    terminal_payload = payload
                    break
            else:
                yield StreamFailed(
                    error=ProviderProtocolError(
                        f"The stream from {self._base_url} ended without a terminal chunk.",
                        details={"base_url": self._base_url},
                    ),
                    partial_text="".join(answer_parts),
                )
                return

            if cancel is not None and cancel.is_cancelled:
                yield self._cancelled("".join(answer_parts))
                return
            if terminal_payload is None:  # pragma: no cover — the `else` above always returns first
                raise AssertionError("stream loop exited without a terminal payload or a return")
            wall_ms = elapsed_ms(start_ns)
            ttft_ms = (
                elapsed_ms(start_ns, first_delta_ns) if first_delta_ns is not None else UNSUPPORTED
            )
            yield StreamCompleted(
                result=self._build_result(
                    request,
                    text="".join(answer_parts),
                    thinking="".join(thinking_parts) if saw_thinking_key else UNSUPPORTED,
                    tool_calls=tool_calls,
                    terminal_payload=terminal_payload,
                    client_wall_ms=wall_ms,
                    client_ttft_ms=ttft_ms,
                )
            )
        except httpx.HTTPError as exc:
            yield StreamFailed(
                error=translate_stream_interruption(exc, base_url=self._base_url),
                partial_text="".join(answer_parts),
            )
        except ProviderProtocolError as exc:
            # Raised by `iter_capped_lines` itself when a line exceeds the per-chunk cap — a
            # typed error already, not a transport exception, so it is delivered unchanged
            # rather than re-translated. Caught here rather than let it escape the generator:
            # the stream has already begun, so this is a `StreamFailed`, never a raise
            # (spec §13, and modelrack.streaming's own terminal-event rule).
            yield StreamFailed(error=exc, partial_text="".join(answer_parts))
        finally:
            response.close()

    def _cancelled(self, partial_text: str) -> StreamFailed:
        """Return the terminal event for a stream the caller stopped, its output attached.

        Delivered, not raised: a raise mid-drain would end the generator with no terminal event,
        indistinguishable from the truncated-stream case this same method's caller already
        detects separately.
        """
        return StreamFailed(
            error=GenerationCancelled(
                "Generation was cancelled by the caller's token.",
                details={"partial_text": partial_text},
            ),
            partial_text=partial_text,
        )

    # --------------------------------------------------------------------- content extraction

    def _extract_content(self, payload: dict[str, Any]) -> tuple[str, str, Any, bool]:
        """Pull the answer text, reasoning text, raw tool calls and a thinking-key flag.

        Handles both shapes Ollama sends: ``/api/chat``'s ``message.{content,thinking,tool_calls}``
        and ``/api/generate``'s flat ``response``. The fourth element distinguishes "this payload
        said nothing about reasoning" from "this payload explicitly reported empty reasoning" —
        the difference between :data:`~baseaicore.UNSUPPORTED` and ``""`` on the eventual result
        ([types.py](../types.py)'s ``GenerationResult.thinking`` docstring is explicit that the two
        are not the same claim).
        """
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            text = content if isinstance(content, str) else ""
            saw_thinking = "thinking" in message and isinstance(message.get("thinking"), str)
            thinking = message.get("thinking") if saw_thinking else ""
            return text, thinking or "", message.get("tool_calls"), saw_thinking
        response_text = payload.get("response")
        text = response_text if isinstance(response_text, str) else ""
        return text, "", None, False

    def _build_result(
        self,
        request: GenerationRequest,
        *,
        text: str,
        thinking: str | Any,
        tool_calls: tuple[ToolCall, ...],
        terminal_payload: Mapping[str, Any],
        client_wall_ms: Measurement,
        client_ttft_ms: Measurement,
    ) -> GenerationResult:
        """Assemble the shared result shape both ``generate`` and a completed stream produce.

        The one place that joins what was said (``text``/``thinking``/``tool_calls``, already
        extracted by the caller — directly for ``generate``, accumulated across deltas for
        ``stream``) with what the terminal payload reports about cost
        (:func:`~modelrack.providers._ollama_wire.read_backend_timing`,
        :func:`~modelrack.providers._ollama_wire.read_usage`) and what this process itself
        measured (``client_wall_ms``, ``client_ttft_ms`` — never read from ``terminal_payload``,
        spec §11.3).
        """
        backend = read_backend_timing(terminal_payload)
        return GenerationResult(
            text=text,
            identity=request.identity,
            finish_reason=finish_reason_for(
                terminal_payload.get("done_reason"), has_tool_calls=bool(tool_calls)
            ),
            usage=read_usage(terminal_payload, text=text),
            timing=Timing(
                client_wall_ms=client_wall_ms,
                client_ttft_ms=client_ttft_ms,
                backend_load_ms=backend.backend_load_ms,
                backend_prompt_eval_ms=backend.backend_prompt_eval_ms,
                backend_decode_ms=backend.backend_decode_ms,
                backend_total_ms=backend.backend_total_ms,
            ),
            tool_calls=tool_calls,
            thinking=thinking,
            # Neither /api/chat nor /api/generate reports the server's own version in a
            # generation response — only /api/version does, and calling it on every generation
            # would blow spec §15's per-request overhead budget for one field. `None` here is
            # honest: this adapter genuinely was not told.
            provider_version=None,
            raw=dict(terminal_payload),
        )

    def _build_message_error(
        self, message: str, *, status_code: int, context_size: int | None
    ) -> ContextLimitExceeded | ProviderRejected:
        """Classify an error message Ollama sent as content, not as a raw transport failure.

        Shared between a pre-stream 4xx (a real HTTP status) and a mid-stream in-band error line
        (HTTP 200 already sent, the failure signalled only in the body) — Ollama uses both, and a
        caller should not have to know which one just happened to distinguish
        :class:`~modelrack.errors.ContextLimitExceeded` from
        :class:`~modelrack.errors.ProviderRejected`.

        Classified by the *presence of a message*, not by the status code's class. Spec §13's
        table names "4xx with a provider message" for :class:`~modelrack.errors.ProviderRejected`,
        but Ollama is not rigorously HTTP-semantic about which status accompanies which failure —
        trusting the status code's exact number over the message it carries would be exactly the
        version-fragility risk register E1 warns
        this adapter to avoid. A 503 that says why is more useful classified the same way a
        400 that says why is than downgraded to an opaque
        :class:`~modelrack.errors.ProviderProtocolError` for having the "wrong" status class.
        """
        if find_context_overflow(message):
            return ContextLimitExceeded(
                message,
                details={
                    "requested_tokens": UNSUPPORTED,
                    "maximum_tokens": context_size if context_size is not None else UNSUPPORTED,
                },
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

    def _raise_for_status(
        self, response: httpx.Response, *, model_reference: str | None, context_size: int | None
    ) -> None:
        """Translate a non-2xx response into the typed error spec §13 names for it, and raise it.

        Reads the body under the same size cap as a successful response — an oversize error body
        is still a protocol error, the one case this re-raises unchanged rather than reclassifying.
        """
        try:
            payload = read_capped_json(
                response, max_bytes=self._max_response_bytes, base_url=self._base_url
            )
            raw_text = json.dumps(payload)
        except ProviderProtocolError as exc:
            if "limit_bytes" in exc.details:
                raise
            payload, raw_text = None, str(exc.details.get("body", ""))
        message = extract_error_message(payload)
        if response.status_code == 404 and model_reference is not None:
            entries = self._list_tags()
            raise ModelNotFound(
                message or f"Model {model_reference!r} not found.",
                details={"reference": model_reference, "known_model_count": len(entries)},
            )
        if message is not None:
            raise self._build_message_error(
                message, status_code=response.status_code, context_size=context_size
            )
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
        context_size: int | None = None,
        timeout: float | None = None,
    ) -> Any:  # noqa: ANN401 — the provider's own JSON shape, never this adapter's business alone
        """POST a JSON body and return the parsed JSON response, translating every failure."""
        try:
            with self._client.stream(
                "POST",
                path,
                json=body,
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            ) as response:
                if response.status_code >= 400:
                    self._raise_for_status(
                        response, model_reference=model_reference, context_size=context_size
                    )
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
                    self._raise_for_status(response, model_reference=None, context_size=None)
                return read_capped_json(
                    response, max_bytes=self._max_response_bytes, base_url=self._base_url
                )
        except httpx.HTTPError as exc:
            raise translate_transport_error(exc, base_url=self._base_url) from exc

    def _list_tags(self) -> list[dict[str, Any]]:
        """Return ``/api/tags``'s ``models`` array, or an empty list if the shape is unexpected."""
        return self._model_entries(self._get_json("/api/tags"))

    def _list_ps(self) -> list[dict[str, Any]]:
        """Return ``/api/ps``'s ``models`` array, or an empty list if the shape is unexpected."""
        return self._model_entries(self._get_json("/api/ps"))

    @staticmethod
    def _model_entries(payload: Any) -> list[dict[str, Any]]:  # noqa: ANN401 — provider JSON
        """Return a listing payload's ``models`` array as dicts, or an empty list if malformed."""
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return []
        return [entry for entry in models if isinstance(entry, dict)]

    def _show(self, name: str) -> dict[str, Any]:
        """Return ``/api/show``'s body for one model."""
        payload = self._post_json("/api/show", {"model": name}, model_reference=name)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _entry_name(entry: Mapping[str, Any]) -> str:
        """Return a listing entry's model name, reading either field Ollama has used for it."""
        name = entry.get("name") or entry.get("model") or ""
        return str(name)

    def _find_optional_entry(
        self, entries: list[dict[str, Any]], name: str
    ) -> dict[str, Any] | None:
        """Return the entry matching ``name`` exactly, or ``None`` if there is none."""
        for entry in entries:
            if self._entry_name(entry) == name:
                return entry
        return None

    def _find_entry(self, entries: list[dict[str, Any]], name: str) -> dict[str, Any]:
        """Return the entry matching ``name`` exactly, or raise :class:`ModelNotFound`."""
        entry = self._find_optional_entry(entries, name)
        if entry is not None:
            return entry
        raise ModelNotFound(
            f"No model named {name!r} is served by this provider.",
            details={"reference": name, "known_model_count": len(entries)},
        )

    def _resolve_entry(self, entries: list[dict[str, Any]], reference: str) -> dict[str, Any]:
        """Resolve a reference to one entry: exact name, then Ollama's ``:latest`` convention,
        then a unique prefix.
        """
        names = [self._entry_name(entry) for entry in entries]
        if reference in names:
            return entries[names.index(reference)]
        if ":" not in reference:
            latest = f"{reference}:latest"
            if latest in names:
                return entries[names.index(latest)]
        prefixed = [
            entry for entry, name in zip(entries, names, strict=True) if name.startswith(reference)
        ]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            raise ModelNotFound(
                f"{reference!r} is a prefix of {len(prefixed)} models; it names none of them. "
                "Give enough of the name to pick one — resolving an ambiguous reference by "
                "choosing would run weights you did not ask for.",
                details={
                    "reference": reference,
                    "known_model_count": len(entries),
                    "matched_model_count": len(prefixed),
                },
            )
        raise ModelNotFound(
            f"No model matching {reference!r} is served by this provider.",
            details={"reference": reference, "known_model_count": len(entries)},
        )

    def _describe_entry(self, entry: dict[str, Any]) -> ModelDescriptor:
        """Enrich one ``/api/tags`` entry with ``/api/show`` metadata into a full descriptor."""
        name = self._entry_name(entry)
        show = self._show(name)
        return build_descriptor(
            name=name,
            digest=entry.get("digest"),
            size=as_measurement(entry.get("size")),
            show=show,
            observed_at=self._clock(),
        )

    @staticmethod
    def _first_supported(*candidates: Measurement) -> Measurement:
        """Return the first candidate that is a real measurement, else ``UNSUPPORTED``."""
        for candidate in candidates:
            if is_supported(candidate):
                return candidate
        return UNSUPPORTED

    def _load_body(self, identity: ModelIdentity, profile: RuntimeProfile) -> dict[str, Any]:
        """Build the load-only ``/api/generate`` body: a model name, no ``prompt`` key at all."""
        body: dict[str, Any] = {"model": identity.provider_model_name}
        if profile.keep_alive is not None:
            body["keep_alive"] = profile.keep_alive
        options = generation_options(
            temperature=None,
            top_p=None,
            top_k=None,
            seed=None,
            max_output_tokens=None,
            stop=(),
            repeat_penalty=None,
            context_size=profile.context_size,
            gpu_layers=profile.gpu_layers,
            threads=profile.threads,
            batch_size=profile.batch_size,
            provider_options=profile.provider_options,
        )
        if options:
            body["options"] = options
        return body

    def _build_request(
        self, request: GenerationRequest, *, stream: bool
    ) -> tuple[str, dict[str, Any]]:
        """Build the request path and JSON body for a generation call.

        Chat-style (``messages``) reaches ``/api/chat``; completion-style (``prompt``) reaches
        ``/api/generate``, which has no concept of tools at all — a request combining ``prompt``
        with ``tools`` is refused here rather than silently dropping the tools, which is the same
        dishonesty ADR-0007 rule 2 forbids for
        an undeclared capability, just discovered at the *combination* rather than the adapter
        level.
        """
        body: dict[str, Any] = {"model": request.identity.provider_model_name, "stream": stream}
        if request.runtime_profile.keep_alive is not None:
            body["keep_alive"] = request.runtime_profile.keep_alive
        options = generation_options(
            temperature=request.sampling.temperature,
            top_p=request.sampling.top_p,
            top_k=request.sampling.top_k,
            seed=request.sampling.seed,
            max_output_tokens=request.sampling.max_output_tokens,
            stop=request.sampling.stop,
            repeat_penalty=request.sampling.repeat_penalty,
            context_size=request.runtime_profile.context_size,
            gpu_layers=request.runtime_profile.gpu_layers,
            threads=request.runtime_profile.threads,
            batch_size=request.runtime_profile.batch_size,
            provider_options=request.runtime_profile.provider_options,
        )
        if options:
            body["options"] = options
        if request.response_format is not None:
            if request.response_format.kind is ResponseFormatKind.JSON:
                body["format"] = "json"
            elif request.response_format.kind is ResponseFormatKind.JSON_SCHEMA:
                body["format"] = dict(request.response_format.schema or {})
        if request.prompt is not None:
            if request.tools:
                raise CapabilityUnsupported(
                    "Ollama's completion endpoint (/api/generate) has no concept of tools; use "
                    "messages instead of prompt to call one.",
                    details={"capability": "tool_calling"},
                )
            body["prompt"] = request.prompt
            return "/api/generate", body
        body["messages"] = [self._message_payload(message) for message in request.messages]
        if request.tools:
            body["tools"] = request_tool_definitions(request.tools)
        return "/api/chat", body

    @staticmethod
    def _message_payload(message: Any) -> dict[str, Any]:  # noqa: ANN401 — modelrack.types.Message
        """Build one Ollama chat message from a :class:`~modelrack.types.Message`."""
        payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {"function": {"name": call.name, "arguments": dict(call.arguments)}}
                for call in message.tool_calls
            ]
        if message.name is not None:
            payload["name"] = message.name
        return payload
