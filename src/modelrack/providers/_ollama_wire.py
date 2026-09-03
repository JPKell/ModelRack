"""Domain module — Ollama's wire format, translated to this package's provider-neutral shapes.

Imports :mod:`baseaicore` and this package's own types; performs no I/O and reads no clock (the
one exception, :func:`build_descriptor`'s ``observed_at``, is a parameter, never
:func:`~baseaicore.utc_now` called directly). Every function here is pure: given the same JSON a
recorded fixture captured, it produces the same result forever, which is what lets
``tests/unit/test_ollama_adapter.py`` assert against fixtures without a running Ollama.

**Name-based, defensive parsing throughout** — the mitigation
risk register E1 names for "Ollama API changes":
every field is read by name with ``.get()``, a missing or malformed one becomes
:data:`~baseaicore.UNSUPPORTED` rather than a guess, and :attr:`ModelDescriptor.raw` always keeps
the untouched payload so a normalizer gap is diagnosable rather than silently lossy.

**The architecture-prefixed metadata block.** Ollama's ``/api/show`` response nests most of a
model's numeric architecture under keys prefixed with the model's own architecture name —
``"qwen3.attention.head_count"``, ``"llama.block_count"``, ``"gemma3.embedding_length"`` — because
one GGUF file can describe any architecture llama.cpp knows about, and the field *names* vary with
it while their *meaning* (layer count, embedding width, …) does not.
:func:`_architecture_value` reads ``general.architecture`` first and looks up every other field
under that prefix, which is what makes this parser work across model families it has never seen
without a single hard-coded architecture name.

**Never a guessed parameter count.** ``details.parameter_size`` is a human string such as
``"9.0B"`` — imprecise by construction, and FreeWeight's KV-cache benchmark computes a theoretical
bytes-per-token from :attr:`ModelDescriptor.parameter_count` that a guess would silently corrupt.
This module reads only ``model_info["general.parameter_count"]``, the exact integer, and leaves
the field :data:`~baseaicore.UNSUPPORTED` when that key is absent — the same choice
:class:`~modelrack.providers.fake.FakeModel`'s docstring makes for the same reason.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from baseaicore import (
    UNSUPPORTED,
    Measurement,
    ModelCapabilityFlag,
    ModelDescriptor,
    ModelIdentity,
    ProviderKind,
    TokenCount,
    from_rfc3339,
    normalize_digest,
)

from modelrack.types import FinishReason, GenerationUsage, Timing, ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

__all__ = [
    "as_measurement",
    "build_descriptor",
    "build_resident_model",
    "extract_error_message",
    "find_context_overflow",
    "finish_reason_for",
    "generation_options",
    "identity_for",
    "parse_tool_calls",
    "read_backend_timing",
    "read_usage",
    "request_tool_definitions",
]

_NANOSECONDS_PER_MILLISECOND: Final[int] = 1_000_000

_DECLARED_CAPABILITY_NAMES: Final[dict[str, ModelCapabilityFlag]] = {
    "tools": ModelCapabilityFlag.TOOLS,
    "vision": ModelCapabilityFlag.VISION,
    "thinking": ModelCapabilityFlag.THINKING,
    "embedding": ModelCapabilityFlag.EMBEDDING,
}

# Ollama does not return a distinct error code for a request that overflows the served context —
# only a human-readable message, and its exact wording has already changed across versions. This
# is name-based parsing's honest ceiling: matched conservatively, and a message that does not
# contain one of these phrases is left as ProviderRejected rather than mis-typed as this error.
_CONTEXT_OVERFLOW_MARKERS: Final[tuple[str, ...]] = (
    "context length",
    "context window",
    "exceeds context",
    "context size",
)


def _architecture_value(
    model_info: Mapping[str, Any], architecture: str | None, suffix: str
) -> Any:  # noqa: ANN401 — mirrors an arbitrary provider payload
    """Return ``model_info[f"{architecture}.{suffix}"]``, or ``None`` when either half is absent."""
    if not architecture:
        return None
    return model_info.get(f"{architecture}.{suffix}")


def _architecture_measurement(
    model_info: Mapping[str, Any], architecture: str | None, suffix: str
) -> Measurement:
    """Return an architecture-prefixed field as a whole-number :data:`~baseaicore.Measurement`."""
    return _as_int_measurement(_architecture_value(model_info, architecture, suffix))


def as_measurement(value: object) -> Measurement:
    """Return ``value`` if it is a real number, else ``UNSUPPORTED``.

    Never coerces a numeric-looking string: a provider that sent ``"32"`` where a number was
    expected sent something this adapter does not trust enough to parse, not a number spelled
    unusually. Public — the adapter uses it directly on ``/api/tags`` and ``/api/ps`` entries
    before they reach :func:`build_descriptor` and :func:`build_resident_model`.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return UNSUPPORTED
    return value


def _as_int_measurement(value: object) -> Measurement:
    """Return ``value`` if it is a whole number, else ``UNSUPPORTED``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return UNSUPPORTED
    return value


def _as_token_count(value: object) -> TokenCount:
    """Return ``value`` if it is a non-negative whole number, else ``UNSUPPORTED``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return UNSUPPORTED
    return value


def identity_for(name: str, digest: str | None) -> tuple[ModelIdentity, str | None]:
    """Build the identity for one catalogue entry, normalizing whatever digest was reported.

    Args:
        name: The provider's own name for the model, exactly as reported.
        digest: The provider's digest in whatever shape it arrived — bare hex, prefixed, upper or
            lower case — or ``None`` when the provider reported none.

    Returns:
        A ``(identity, discarded_reason)`` pair. ``discarded_reason`` is ``None`` unless a digest
        was present but would not normalize, in which case the identity is ``name_only`` and the
        reason is set so the loss is diagnosable rather than silent
        (ADR-0024 §2).
    """
    normalized = normalize_digest(digest)
    if digest is not None and normalized is None:
        reason = (
            f"provider reported digest {digest!r}, which is not 'sha256:' followed by 64 hex "
            "characters; discarded, identity is name_only"
        )
        return ModelIdentity(ProviderKind.OLLAMA, name), reason
    return ModelIdentity(ProviderKind.OLLAMA, name, artifact_digest=normalized), None


def build_descriptor(
    *,
    name: str,
    digest: str | None,
    size: Measurement,
    show: Mapping[str, Any],
    observed_at: datetime,
) -> ModelDescriptor:
    """Build a :class:`~baseaicore.ModelDescriptor` from a ``/api/tags`` entry and ``/api/show``.

    Two calls, because Ollama's two model-listing endpoints are not redundant: ``/api/tags`` is
    where the digest and on-disk size live, ``/api/show`` is where the architecture metadata
    lives, and neither includes the other's information. Merging is this function's whole job.

    Args:
        name: The provider's model name.
        digest: The digest as ``/api/tags`` (or ``/api/ps``) reported it, unnormalized.
        size: The on-disk size in bytes, from the same listing.
        show: The full parsed ``/api/show`` response body.
        observed_at: When this snapshot was read — the caller's clock, never this module's.

    Returns:
        The descriptor, with ``raw`` carrying both source payloads and, when the digest could not
        be normalized, the reason it was discarded.
    """
    identity, discarded_reason = identity_for(name, digest)
    details = show.get("details")
    details = details if isinstance(details, Mapping) else {}
    model_info = show.get("model_info")
    model_info = model_info if isinstance(model_info, Mapping) else {}
    architecture = model_info.get("general.architecture")
    architecture = architecture if isinstance(architecture, str) and architecture else None
    family = details.get("family")
    quantization = details.get("quantization_level")
    weight_format = details.get("format")
    license_text = show.get("license")
    raw: dict[str, Any] = {
        "tags_entry": {"name": name, "digest": digest, "size": size},
        "show": dict(show),
    }
    if discarded_reason is not None:
        raw["digest_discarded_reason"] = discarded_reason

    def arch(suffix: str) -> Measurement:
        return _architecture_measurement(model_info, architecture, suffix)

    return ModelDescriptor(
        identity=identity,
        observed_at=observed_at,
        family=family if isinstance(family, str) else None,
        architecture=architecture,
        parameter_count=_as_int_measurement(model_info.get("general.parameter_count")),
        expert_count=arch("expert_count"),
        quantization=quantization if isinstance(quantization, str) else None,
        weight_format=weight_format if isinstance(weight_format, str) else None,
        size_bytes=size,
        max_context=arch("context_length"),
        embedding_dim=arch("embedding_length"),
        layers=arch("block_count"),
        attention_heads=arch("attention.head_count"),
        kv_heads=arch("attention.head_count_kv"),
        head_dim=arch("attention.key_length"),
        vocab_size=arch("vocab_size"),
        sliding_window=arch("attention.sliding_window"),
        declared_capabilities=_declared_capabilities(show.get("capabilities")),
        license_text=license_text if isinstance(license_text, str) else None,
        raw=raw,
    )


def _declared_capabilities(value: Any) -> frozenset[ModelCapabilityFlag]:  # noqa: ANN401 — provider JSON
    """Map Ollama's ``capabilities`` string list to the vocabulary's flags, skipping the rest.

    Ollama's list includes entries this vocabulary has no flag for (``"completion"``,
    ``"insert"``) — skipped rather than raising, because a provider adding a new capability string
    is an additive change this adapter should survive, not an error to translate.
    """
    if not isinstance(value, list):
        return frozenset()
    return frozenset(
        _DECLARED_CAPABILITY_NAMES[name] for name in value if name in _DECLARED_CAPABILITY_NAMES
    )


def build_resident_model(
    entry: Mapping[str, Any],
) -> tuple[ModelIdentity, Measurement, Measurement, datetime | None, Measurement]:
    """Parse one ``/api/ps`` entry into the pieces :class:`~modelrack.provider.ResidentModel` needs.

    Returns:
        ``(identity, vram_bytes, total_bytes, expires_at, context_length)``. Ollama reports one
        VRAM figure per model, not per device — passed through as the single number the provider
        gave, which is not the same thing as this package summing across devices itself
        (ADR-0027 constrains *this package's*
        arithmetic, not what a provider chooses to report as one figure).

        ``context_length`` is the context the model is **actually being served at**, which is not
        the same as the context its descriptor advertises and is the only way to know the
        difference without having asked for one. A consumer resolving a served context prefers a
        reported value over an assumed one for exactly this reason
        (ADR-0023 §4).
    """
    name = entry.get("name") or entry.get("model") or ""
    identity = identity_for(str(name), entry.get("digest"))[0]
    expires_at_raw = entry.get("expires_at")
    expires_at = from_rfc3339(expires_at_raw) if isinstance(expires_at_raw, str) else None
    return (
        identity,
        as_measurement(entry.get("size_vram")),
        as_measurement(entry.get("size")),
        expires_at,
        as_measurement(entry.get("context_length")),
    )


def generation_options(
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    max_output_tokens: int | None,
    stop: Sequence[str],
    repeat_penalty: float | None,
    context_size: int | None,
    gpu_layers: int | None,
    threads: int | None,
    batch_size: int | None,
    provider_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Build Ollama's ``options`` object from sampling parameters and a runtime profile.

    ``flash_attention`` and ``kv_cache_precision`` are deliberately **not** translated: Ollama
    configures both at server startup (``OLLAMA_FLASH_ATTENTION``, ``OLLAMA_KV_CACHE_TYPE``), not
    per request, and there is no ``options`` key that would make setting one here take effect.
    Inventing one and sending it would claim a promise this adapter cannot keep — the same
    dishonesty ADR-0007 rule 2 forbids for a
    capability flag. A caller needing either configures the server directly.

    ``provider_options`` is merged last and wins on any overlapping key, which is what makes it an
    escape hatch: a caller who knows the exact Ollama option name for something this function does
    not translate can always reach it.
    """
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k
    if seed is not None:
        options["seed"] = seed
    if max_output_tokens is not None:
        options["num_predict"] = max_output_tokens
    if stop:
        options["stop"] = list(stop)
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    if context_size is not None:
        options["num_ctx"] = context_size
    if gpu_layers is not None:
        options["num_gpu"] = gpu_layers
    if threads is not None:
        options["num_thread"] = threads
    if batch_size is not None:
        options["num_batch"] = batch_size
    options.update(provider_options)
    return options


def request_tool_definitions(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Build Ollama's tool list from :class:`~modelrack.types.ToolDefinition` values.

    Ollama adopted the OpenAI function-calling shape for tools
    (``{"type": "function", "function": {...}}``); this is the one translation, kept beside the
    other request-building functions rather than inlined in the adapter.
    """
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


def finish_reason_for(done_reason: Any, *, has_tool_calls: bool) -> FinishReason:  # noqa: ANN401 — provider JSON
    """Map Ollama's ``done_reason`` string to the vocabulary's finish reason.

    A message carrying tool calls wins regardless of what ``done_reason`` said, because Ollama's
    own ``done_reason`` for a tool-calling turn is ``"stop"`` — the same string an ordinary answer
    ends with — and :class:`~modelrack.types.GenerationResult` requires ``TOOL_CALLS`` whenever
    tool calls are present (its own ``__post_init__`` enforces the reverse).
    """
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    mapped = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "load": FinishReason.STOP,
        "unload": FinishReason.STOP,
    }
    return (
        mapped.get(done_reason, FinishReason.UNKNOWN)
        if isinstance(done_reason, str)
        else (FinishReason.UNKNOWN)
    )


def parse_tool_calls(
    raw_calls: Any, *, call_prefix: str, start_index: int = 0
) -> tuple[ToolCall, ...]:  # noqa: ANN401 — provider JSON
    """Parse Ollama's ``message.tool_calls`` into the vocabulary's :class:`ToolCall` tuple.

    Ollama's tool calls carry no ``id`` — unlike OpenAI's shape it borrowed the rest of, a call
    here is just ``{"function": {"name": ..., "arguments": {...}}}``, with ``arguments`` already a
    parsed JSON object rather than a string to parse. An id is synthesized from ``call_prefix`` and
    the call's position, because a multi-call turn cannot be answered without one
    ([types.py](../types.py)'s own ``ToolCall.__post_init__`` requires a non-blank one).

    Args:
        raw_calls: The provider's ``tool_calls`` value, expected to be a list.
        call_prefix: A string unique to this generation, so ids from two different calls never
            collide.
        start_index: Where numbering begins — non-zero when a stream has already parsed an
            earlier batch of calls and this one must not reuse those ids.
    """
    if not isinstance(raw_calls, list):
        return ()
    calls: list[ToolCall] = []
    for offset, entry in enumerate(raw_calls):
        function = entry.get("function") if isinstance(entry, Mapping) else None
        function = function if isinstance(function, Mapping) else {}
        name = function.get("name")
        arguments = function.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        calls.append(
            ToolCall(
                id=f"{call_prefix}-{start_index + offset}",
                name=name if isinstance(name, str) and name else "unknown",
                arguments=dict(arguments),
            )
        )
    return tuple(calls)


def read_backend_timing(payload: Mapping[str, Any]) -> Timing:
    """Extract the provider's own account of a call's cost, converting nanoseconds to milliseconds.

    Ollama reports durations in **nanoseconds** — ``total_duration``, ``load_duration``,
    ``prompt_eval_duration``, ``eval_duration`` — and this is the only place that division happens.
    Client-observed fields are never set here: this function reads only what the provider claimed
    about its own work, never what this process measured, which is spec §11.3's separation applied
    ([types.py](../types.py)'s ``Timing`` has no field this function is allowed to fill from the
    client side).
    """

    def ms(key: str) -> Measurement:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return UNSUPPORTED
        return value / _NANOSECONDS_PER_MILLISECOND

    return Timing(
        backend_load_ms=ms("load_duration"),
        backend_prompt_eval_ms=ms("prompt_eval_duration"),
        backend_decode_ms=ms("eval_duration"),
        backend_total_ms=ms("total_duration"),
    )


def read_usage(payload: Mapping[str, Any], *, text: str) -> GenerationUsage:
    """Build a :class:`~modelrack.types.GenerationUsage` from Ollama's count fields and the answer.

    **What this protocol can bill.** Ollama's generation responses carry exactly two count fields,
    ``prompt_eval_count`` and ``eval_count``, mapped to the billing vocabulary's ``input_tokens``
    and ``output_tokens``. There is no cache-billing vocabulary anywhere in the protocol — no
    field, in any response shape, by which Ollama could charge for a cache read or a cache write —
    so both cache classes are ``0``: not an invented zero, but the statement that nothing could
    have been billed under those headings
    (ADR-0070 decision 3, which is
    ADR-0016's rule applied rather than
    reversed — a measurement this adapter cannot obtain is still ``UNSUPPORTED``, and the two
    cases are distinguished below).

    **What ``prompt_eval_count`` counts: the whole prompt, not the tokens evaluated.** Ollama
    reuses its KV cache across requests that share a prefix, which raises the question of whether
    this field reports the work done or the prompt submitted. It reports the prompt submitted.
    Measured against Ollama 0.32.13 on ``/api/chat`` with ``gemma3:latest``: two back-to-back
    requests sharing a 5 400-token prefix and differing only in a short tail both reported
    ``prompt_eval_count`` 5 410, while ``prompt_eval_duration`` fell from 885 ms to 126 ms — the
    cache demonstrably served the prefix, and the count did not move. So ``input_tokens`` here is
    the prompt length, the same quantity every other adapter reports, and a caller comparing token
    counts across providers is comparing like with like. A token brake reading this field brakes
    on prompt size, not on work performed, and a cached prefix costs the brake full price.

    **No counts at all is not zero counts.** A terminal payload carrying neither count field —
    Ollama's analogue of a response with no ``usage`` object — reports every class
    ``UNSUPPORTED``, including the cache classes. Nothing was reported, so nothing is known; that
    is the third of ADR-0070's three cases and the one where a zero would be a fabrication.

    Character, word and byte counts are observations this process can make about the string it is
    holding regardless of what the provider counted, so they are always present.
    """
    from baseaicore import TokenUsage  # noqa: PLC0415 — avoids a module-level cycle with types.py

    reports_counts = "prompt_eval_count" in payload or "eval_count" in payload
    tokens = (
        TokenUsage(
            input_tokens=_as_token_count(payload.get("prompt_eval_count")),
            output_tokens=_as_token_count(payload.get("eval_count")),
            cache_write_tokens=0,
            cache_read_tokens=0,
        )
        if reports_counts
        else TokenUsage()
    )
    return GenerationUsage(
        tokens=tokens,
        output_chars=len(text),
        output_words=len(text.split()),
        output_bytes=len(text.encode("utf-8")),
    )


def extract_error_message(payload: Any) -> str | None:  # noqa: ANN401 — provider JSON
    """Return Ollama's own error message from a parsed error body, or ``None`` if there is none.

    Every documented Ollama error shape is ``{"error": "<message>"}``; this is the one place that
    is read, so a future shape change is a one-line fix rather than a hunt through the adapter.
    """
    if not isinstance(payload, Mapping):
        return None
    message = payload.get("error")
    return message if isinstance(message, str) and message else None


def find_context_overflow(message: str) -> bool:
    """Report whether an error message names a context-window overflow.

    See the module docstring's note on this constant: Ollama gives no distinct error code for
    this condition, only prose, and the prose has already changed across versions. Matched
    conservatively against known phrasings; a message that does not match is left as
    :class:`~modelrack.errors.ProviderRejected` rather than mis-typed as
    :class:`~modelrack.errors.ContextLimitExceeded`.
    """
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)
