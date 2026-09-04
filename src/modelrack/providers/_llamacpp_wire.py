"""Domain module — llama.cpp's server wire format, translated to this package's shapes.

Imports :mod:`baseaicore`, this package's own types and the two sibling wire modules; performs no
I/O and reads no clock (:func:`build_descriptor`'s ``observed_at`` is a parameter). Every
function here is pure, so ``tests/unit/test_llamacpp_adapter.py`` asserts against recorded
fixtures without a server, and :mod:`modelrack.providers.llamacpp` owns everything that touches a
process, a file or a socket.

llama-server speaks two request shapes and this module translates both:

* **The native completion API** (``POST /completion``): a raw ``prompt``, llama.cpp's own
  sampling names (``n_predict``, ``repeat_penalty``), a ``json_schema`` field for constrained
  output, and a response carrying ``tokens_evaluated``, ``tokens_predicted``, ``stop_type`` and a
  ``timings`` object. A completion-style :class:`~modelrack.types.GenerationRequest` goes here.
* **The chat API** (``POST /v1/chat/completions``): the OpenAI chat-completions shape — which
  :mod:`modelrack.providers._openai_wire` already translates — with llama.cpp's extensions on
  top: the same ``timings`` object, ``system_fingerprint`` carrying the build, a
  ``reasoning_content`` field, and the ``lora`` request field Phase 7 will use. A chat-style
  request goes here because this is where the server applies the model's chat template and
  parses tool calls, neither of which this package does itself (spec §3: no templating).

**What this protocol can bill, and the rule it is read to** (ADR-0070 decision 4). Both shapes
report the prompt total (``tokens_evaluated`` / ``usage.prompt_tokens``) and the output count
(``tokens_predicted`` / ``usage.completion_tokens``). Both can report **cached input**:
``timings.cache_n`` is the number of prompt tokens the server reused from its KV cache, and the
chat shape repeats it as ``usage.prompt_tokens_details.cached_tokens``. Neither shape has any
field by which a cache *write* could be charged, so ``cache_write_tokens`` is ``0`` whenever
counts are present — a statement about the protocol, not the response. The three cases:

| Response | ``input_tokens`` | ``cache_read_tokens`` |
|---|---|---|
| counts, and a readable cached figure ≤ the prompt total | total − cached | cached |
| counts, and no cached-input field at all | total | ``0`` |
| counts, and a cached figure unreadable or > total | ``UNSUPPORTED`` | ``UNSUPPORTED`` |

and a response with no counts at all reports every class ``UNSUPPORTED``. The second row is a
build that predates ``timings.cache_n``: such a server may still reuse its cache but cannot say
so, which is ADR-0070's "the protocol has no way to bill it" — ``0`` is honest there, and the
consequence is exactly Ollama's: ``input_tokens`` is the prompt submitted, never less. The third
row refuses both halves of one subtraction together, for the reason the OpenAI-compatible adapter
gives: an ``input_tokens`` reported beside an unknown cached figure would not be disjoint from it.

**The trap this module exists to step around.** The native response also carries
``tokens_cached``, which reads like the cached-input count and is not: llama-server sets it to
the slot's *whole* cache after generation — prompt plus output (``slot.prompt.n_tokens()`` in
``server-context.cpp``). Read as a cache-read count it would report every token of every call as
a cached hit and bill the caller for nothing. It is never read here; the recorded fixtures carry
it with its real value (prompt + output) so a regression that started reading it would fail the
reconciliation arithmetic rather than pass a type check.

Verified against the llama.cpp server source at build ``b10792`` (``tools/server/server-task.cpp``,
``server-common.cpp``, ``server-context.cpp``), which is the build the recorded fixtures represent.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from baseaicore import (
    UNSUPPORTED,
    Measurement,
    ModelDescriptor,
    ModelIdentity,
    ProviderKind,
    TokenCount,
    TokenUsage,
    is_supported,
    normalize_digest,
)

from modelrack.providers._gguf import ArraySummary
from modelrack.providers._openai_wire import (
    message_payload,
    request_tool_definitions,
    response_format_payload,
)
from modelrack.types import FinishReason, GenerationUsage, ResponseFormatKind, Timing

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from baseaicore import RuntimeProfile

    from modelrack.providers._gguf import GgufHeader
    from modelrack.types import GenerationRequest

__all__ = [
    "DEFAULT_SERVER_EXECUTABLE",
    "FORBIDDEN_LAUNCH_FLAGS",
    "FORBIDDEN_REQUEST_KEYS",
    "SLOT_PINNING_KEYS",
    "ServerAdapter",
    "adapter_launch_flags",
    "LlamaCppError",
    "build_chat_body",
    "build_completion_body",
    "build_descriptor",
    "build_launch_argv",
    "completion_finish_reason",
    "header_kind",
    "identity_for",
    "is_shard",
    "launch_flags",
    "lora_field",
    "model_name_for",
    "read_lora_adapters",
    "quantization_name",
    "read_backend_timing",
    "read_build_info",
    "read_chat_usage",
    "read_completion_usage",
    "read_error",
    "read_served_context",
    "request_options",
]

DEFAULT_SERVER_EXECUTABLE: Final[str] = "llama-server"
"""The binary llama.cpp installs; resolved on ``PATH`` unless a path is given."""

_LAUNCH_FLAG_PREFIX: Final[str] = "--"

FORBIDDEN_REQUEST_KEYS: Final[frozenset[str]] = frozenset({"lora"})
"""Request keys a caller may not set through ``provider_options``.

``lora`` is the adapter selection itself. Reaching it through the escape hatch would change the
weights that answer without changing :attr:`~modelrack.types.GenerationRequest.adapter`, so the
result would be recorded against a subject that did not run — the fabricated comparability
ADR-0058 exists to prevent. The supported channel is the request field, which the provider
resolves, verifies and reports back on the result.
"""

SLOT_PINNING_KEYS: Final[frozenset[str]] = frozenset({"id_slot", "slot_id"})
"""Request keys that bind a request to one of the server's slots, and so to that slot's cache.

llama-server clears a slot's prompt cache when a task's adapter set differs from the slot's
(``lora_should_clear_cache``), which is what makes prefix reuse across an adapter switch
impossible **as long as slot selection stays the server's**. Pinning a slot is the one lever that
reaches past that rule, so this adapter never sends one and refuses a caller that tries
(ADR-0062 decision 4).
"""

FORBIDDEN_LAUNCH_FLAGS: Final[frozenset[str]] = frozenset(
    {"--lora", "--lora-scaled", "--lora-init-without-apply"}
)
"""Launch flags a caller may not set through ``provider_options``.

An adapter registered behind this package's back has no
:class:`~modelrack.adapters.AdapterRegistration`, so it has no digest, no verified base and no
name a result could report — it would be a weights delta the suite cannot name. Registration goes
through :meth:`~modelrack.provider.Provider.register_adapters`, which verifies the base first.
"""


@dataclass(frozen=True, slots=True)
class ServerAdapter:
    """One adapter a running server has actually registered, as the server itself reports it.

    Read back from ``GET /lora-adapters`` after the server is healthy rather than inferred from
    argv order: the id in a request's ``lora`` field is the server's own index, and assuming it
    from the command line would put a whole class of off-by-one wrong-adapter bugs one refactor
    away — the failure mode that produces plausible, confident, wrong output.

    Attributes:
        server_id: The index llama-server assigned, and the ``id`` a request sends.
        path: The artifact path the server reports, matched against what was launched.
        name: The registration's name, once the path has been matched to one.
    """

    server_id: int
    path: str
    name: str


_LOOPBACK_HOST: Final[str] = "127.0.0.1"
_SHARD_PATTERN: Final[re.Pattern[str]] = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
_MODEL_KIND: Final[str] = "model"

_CONTEXT_OVERFLOW_TYPE: Final[str] = "exceed_context_size_error"
_UNAVAILABLE_TYPE: Final[str] = "unavailable_error"
_CONTEXT_OVERFLOW_MARKERS: Final[tuple[str, ...]] = (
    "context size",
    "context length",
    "exceeds the available context",
)

_STOP_TYPES: Final[dict[str, FinishReason]] = {
    "eos": FinishReason.STOP,
    "word": FinishReason.STOP,
    "limit": FinishReason.LENGTH,
    "none": FinishReason.UNKNOWN,
}

# llama.cpp's `llama_ftype` enum, as stored in `general.file_type`. Removed members (4–6, 33–35)
# are absent on purpose: a file carrying one is a file this build of llama.cpp would not load.
_FILE_TYPES: Final[dict[int, str]] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
}


# ------------------------------------------------------------------------------- discovery


def model_name_for(path: Path, *, root: Path) -> str:
    """Return the name a GGUF file is served under: its path below ``root``, without ``.gguf``.

    ``root/qwen/qwen3-14b.Q4_K_M.gguf`` is ``qwen/qwen3-14b.Q4_K_M`` — the same shape as an
    Ollama name with a namespace, and one that round-trips to the file. The name is what a
    caller types and what ``--alias`` tells the server to answer to; the path is a locator and
    never part of the identity.
    """
    return PurePosixPath(path.relative_to(root).as_posix()).with_suffix("").as_posix()


def is_shard(path: Path) -> bool:
    """Report whether a file is one shard of a split GGUF (``…-00002-of-00003.gguf``).

    Split files are not served in this phase: their identity is a hash over several files and
    llama-server is handed only the first, so listing each shard as a model would be wrong twice
    over. They are skipped at discovery with a DEBUG log rather than misdescribed.
    """
    return _SHARD_PATTERN.search(path.name) is not None


def header_kind(header: GgufHeader) -> str:
    """Return what kind of GGUF this is: ``"model"``, ``"adapter"``, ``"mmproj"``, ….

    ``general.type`` is absent on older files, which are all base models; a LoRA adapter says
    ``adapter`` and a vision projector says ``mmproj``, and neither is a base this adapter can
    serve on its own. Adapters are Phase 7's to register.
    """
    kind = header.metadata.get("general.type")
    return kind if isinstance(kind, str) and kind else _MODEL_KIND


def quantization_name(file_type: object) -> str | None:
    """Map ``general.file_type`` to llama.cpp's quantization label, or ``None`` if unknown."""
    if isinstance(file_type, bool) or not isinstance(file_type, int):
        return None
    return _FILE_TYPES.get(file_type)


def identity_for(name: str, digest: str | None) -> ModelIdentity:
    """Build the identity for one served file.

    Args:
        name: The served name, from :func:`model_name_for`.
        digest: The content digest from :func:`~modelrack.providers._gguf.sha256_of_file`, or
            ``None`` when it has not been computed. Normalized on the way in, so a digest that
            somehow will not normalize yields a name-only identity rather than a malformed one
            (spec §11.9).
    """
    return ModelIdentity(ProviderKind.LLAMACPP, name, artifact_digest=normalize_digest(digest))


def _as_int(value: object) -> Measurement:
    """Return ``value`` if it is a whole number, else ``UNSUPPORTED``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return UNSUPPORTED
    return value


def _per_layer_or_scalar(value: object) -> Measurement:
    """Read a metadata field that is a scalar on most models and a per-layer array on a few.

    A per-layer array whose entries all agree is that one number — exact, not a guess. One whose
    entries differ has no single honest value, and is ``UNSUPPORTED`` with the array still in
    ``raw`` for a consumer that wants the detail.
    """
    if isinstance(value, tuple):
        entries = {_as_int(entry) for entry in value}
        if len(entries) == 1:
            return entries.pop()
        return UNSUPPORTED
    return _as_int(value)


def _raw_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Render header metadata as plain JSON-shaped data for a descriptor's ``raw``."""
    rendered: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, ArraySummary):
            rendered[key] = {"array": {"element_type": value.element_type, "length": value.length}}
        elif isinstance(value, tuple):
            rendered[key] = list(value)
        else:
            rendered[key] = value
    return rendered


def build_descriptor(
    header: GgufHeader, *, name: str, digest: str | None, observed_at: datetime
) -> ModelDescriptor:
    """Build a :class:`~baseaicore.ModelDescriptor` from a GGUF header.

    The architecture fields live under keys prefixed with the file's own architecture name
    (``qwen3.block_count``, ``gemma3.attention.head_count_kv``), the same convention Ollama's
    ``/api/show`` inherits from these files — so the lookup is by ``general.architecture`` plus a
    suffix, and works for architectures this code has never seen.

    Args:
        header: The parsed header.
        name: The served name.
        digest: The content digest, or ``None`` if not computed.
        observed_at: When the header was read — the caller's clock.

    Returns:
        The descriptor. ``parameter_count`` is the exact tensor-element sum from the header;
        ``vocab_size`` is the tokenizer array's length; ``size_bytes`` is the file's size;
        ``quantization`` is llama.cpp's own label for ``general.file_type``. Anything the file
        does not state is ``UNSUPPORTED``, and the whole metadata block is kept in ``raw``.
    """
    metadata = header.metadata
    architecture = metadata.get("general.architecture")
    architecture = architecture if isinstance(architecture, str) and architecture else None

    def arch(suffix: str) -> Measurement:
        if architecture is None:
            return UNSUPPORTED
        return _per_layer_or_scalar(metadata.get(f"{architecture}.{suffix}"))

    vocab = metadata.get("tokenizer.ggml.tokens")
    vocab_size: Measurement
    if isinstance(vocab, ArraySummary):
        vocab_size = vocab.length
    elif isinstance(vocab, tuple):
        vocab_size = len(vocab)
    else:
        vocab_size = arch("vocab_size")
    rope = {
        key: value
        for key, value in _raw_metadata(metadata).items()
        if architecture is not None and key.startswith(f"{architecture}.rope.")
    }
    license_text = metadata.get("general.license")
    return ModelDescriptor(
        identity=identity_for(name, digest),
        observed_at=observed_at,
        family=architecture,
        architecture=architecture,
        parameter_count=header.parameter_count if header.parameter_count > 0 else UNSUPPORTED,
        expert_count=arch("expert_count"),
        quantization=quantization_name(metadata.get("general.file_type")),
        weight_format="gguf",
        size_bytes=header.stamp.size_bytes,
        max_context=arch("context_length"),
        embedding_dim=arch("embedding_length"),
        layers=arch("block_count"),
        attention_heads=arch("attention.head_count"),
        kv_heads=arch("attention.head_count_kv"),
        head_dim=arch("attention.key_length"),
        vocab_size=vocab_size,
        rope_config=rope or None,
        sliding_window=arch("attention.sliding_window"),
        license_text=license_text if isinstance(license_text, str) else None,
        raw={
            "path": str(header.path),
            "gguf_version": header.version,
            "tensor_count": header.tensor_count,
            "metadata": _raw_metadata(metadata),
        },
    )


# ---------------------------------------------------------------------------------- launch


def launch_flags(profile: RuntimeProfile) -> tuple[str, ...]:
    """Translate a runtime profile into llama-server command-line flags.

    Every field a profile can carry becomes a launch flag, because on llama-server every one of
    them is a launch-time property: ``context_size`` → ``--ctx-size``, ``gpu_layers`` →
    ``--n-gpu-layers``, ``kv_cache_precision`` → ``--cache-type-k`` and ``--cache-type-v``,
    ``flash_attention`` → ``--flash-attn on|off``, ``threads`` → ``--threads``, ``batch_size`` →
    ``--batch-size``. A field left ``None`` sends no flag, so the server's own default applies —
    "provider defaults" is what a default profile means (ADR-0023 §1), and inventing a value
    here would record a profile the run did not use.

    ``keep_alive`` has no translation and is stated rather than silently dropped: a supervised
    server stays loaded until ``unload()`` and has no idle-eviction timer in this phase.

    ``provider_options`` keys that start with ``--`` are passed as further flags, sorted by name
    so two equal profiles produce one argv: ``True`` sends the bare flag, ``False`` or ``None``
    sends nothing, and any other value is sent as the flag's argument. This is how a
    chat-template override (``--chat-template-file``), ``--parallel`` or ``--cache-reuse``
    reach the server without this package naming each one. Keys without the prefix are request
    options (:func:`request_options`), not launch flags.
    """
    flags: list[str] = []
    if profile.context_size is not None:
        flags += ["--ctx-size", str(profile.context_size)]
    if profile.gpu_layers is not None:
        flags += ["--n-gpu-layers", str(profile.gpu_layers)]
    if profile.kv_cache_precision is not None:
        flags += ["--cache-type-k", profile.kv_cache_precision]
        flags += ["--cache-type-v", profile.kv_cache_precision]
    if profile.flash_attention is not None:
        flags += ["--flash-attn", "on" if profile.flash_attention else "off"]
    if profile.threads is not None:
        flags += ["--threads", str(profile.threads)]
    if profile.batch_size is not None:
        flags += ["--batch-size", str(profile.batch_size)]
    for key in sorted(profile.provider_options):
        if not key.startswith(_LAUNCH_FLAG_PREFIX):
            continue
        value = profile.provider_options[key]
        if value is None or value is False:
            continue
        if value is True:
            flags.append(key)
        else:
            flags += [key, str(value)]
    return tuple(flags)


def request_options(profile: RuntimeProfile) -> dict[str, Any]:
    """Return the ``provider_options`` entries that are per-request fields, not launch flags.

    The escape hatch spec §12 gives a caller who knows llama.cpp's own request vocabulary —
    ``min_p``, ``cache_prompt``, ``n_probs`` — merged last into the request body so it wins on
    any overlapping key, the same rule the Ollama adapter documents for its ``options``.
    """
    return {
        key: value
        for key, value in profile.provider_options.items()
        if not key.startswith(_LAUNCH_FLAG_PREFIX)
    }


def adapter_launch_flags(artifact_paths: Sequence[Path]) -> tuple[str, ...]:
    """Return the flags that pre-register ``artifact_paths`` without applying any of them.

    ``--lora`` once per artifact, in the order given — which is the order the server assigns ids
    in — followed by ``--lora-init-without-apply`` so nothing is active until a request asks for
    it (ADR-0062 decision 1). Empty for no adapters, so a server launched without any has an argv
    byte-for-byte identical to Phase 6's.

    Args:
        artifact_paths: The adapter artifacts to register, already verified against the base.

    Returns:
        The flags, or ``()`` when there are none.
    """
    if not artifact_paths:
        return ()
    flags: list[str] = []
    for path in artifact_paths:
        flags += ["--lora", str(path)]
    flags.append("--lora-init-without-apply")
    return tuple(flags)


def read_lora_adapters(payload: object, *, by_path: Mapping[str, str]) -> tuple[ServerAdapter, ...]:
    """Read ``GET /lora-adapters`` into the ids this adapter will send.

    Args:
        payload: The server's response — an array of ``{id, path, scale, …}`` objects.
        by_path: Artifact path to registration name, for the adapters that were launched.

    Returns:
        One :class:`ServerAdapter` per entry whose ``path`` matches a launched registration, in
        server-id order. An entry the launch did not ask for is dropped rather than trusted: it
        would be an adapter this package cannot name, and naming is the whole point.
    """
    if not isinstance(payload, list):
        return ()
    found: list[ServerAdapter] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        server_id = entry.get("id")
        path = entry.get("path")
        if (
            not isinstance(server_id, int)
            or isinstance(server_id, bool)
            or not isinstance(path, str)
        ):
            continue
        name = by_path.get(path)
        if name is None:
            continue
        found.append(ServerAdapter(server_id=server_id, path=path, name=name))
    return tuple(sorted(found, key=lambda adapter: adapter.server_id))


def lora_field(
    registered: Sequence[ServerAdapter], *, selected: str | None
) -> list[dict[str, Any]] | None:
    """Build the ``lora`` request field: the complete adapter configuration, always.

    **Complete, not minimal, and this is the correctness decision of Phase 7.** llama-server treats
    an *absent* ``lora`` field as "restore the launch-time set" — ``slot.lora =
    params_base.lora_adapters`` in ``launch_slot_with_task`` — and takes that branch **without
    consulting** ``lora_should_clear_cache``. Since ``--lora`` registers an adapter at scale
    ``1.0`` and ``--lora-init-without-apply`` changes only whether the set is applied at *init*, a
    bare-base request that sent no ``lora`` field to an adapter-registered server would run with
    **every** registered adapter applied, against a prompt cache built under whatever ran last. So
    a request to such a server always states the whole configuration, and a request to a server
    with no adapters registered sends no ``lora`` key at all — byte-for-byte Phase 6's body.

    At most one entry is ever enabled, at exactly ``1.0`` (ADR-0063): the others are present at
    ``0.0``, which is a disable and not a composition. There is no per-request scale, and there is
    nowhere for one to be passed.

    Args:
        registered: What the server has registered, in server-id order.
        selected: The registration name to run under, or ``None`` for the bare base. Assumed to
            name one of ``registered`` — the provider resolves and refuses before this is reached.

    Returns:
        The complete list, or ``None`` when the server has no adapters registered and the key
        must be absent.
    """
    if not registered:
        return None
    return [
        {"id": adapter.server_id, "scale": 1.0 if adapter.name == selected else 0.0}
        for adapter in registered
    ]


def build_launch_argv(
    *,
    server_path: str,
    model_path: Path,
    alias: str,
    port: int,
    profile: RuntimeProfile,
    adapter_paths: Sequence[Path] = (),
) -> tuple[str, ...]:
    """Build the complete command line for one supervised server.

    Loopback only: ``--host 127.0.0.1`` is fixed, because a server this package spawns is reached
    by this process and nothing else, and a supervised server that listened on every interface
    would be an egress path nobody configured. ``--alias`` makes the server answer to the same
    name this adapter serves the file under; ``--jinja`` selects the template engine tool
    calling needs; ``--no-webui`` drops the browser UI nothing here uses. Profile flags follow,
    and ``provider_options`` flags last, so a caller's explicit flag wins on a conflict.

    ``adapter_paths`` become ``--lora`` flags and one ``--lora-init-without-apply``, placed
    **before** the profile flags so that the id the server assigns each adapter depends only on
    the registration order and never on which profile flags a request happened to set. With no
    adapters the argv is byte-for-byte what Phase 6 built.
    """
    return (
        server_path,
        "--model",
        str(model_path),
        "--alias",
        alias,
        "--host",
        _LOOPBACK_HOST,
        "--port",
        str(port),
        "--jinja",
        "--no-webui",
        *adapter_launch_flags(adapter_paths),
        *launch_flags(profile),
    )


# -------------------------------------------------------------------------------- requests


def _sampling_fields(request: GenerationRequest, *, max_tokens_key: str) -> dict[str, Any]:
    """The sampling parameters both request shapes share, under the given output-limit name."""
    sampling = request.sampling
    body: dict[str, Any] = {}
    if sampling.temperature is not None:
        body["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        body["top_p"] = sampling.top_p
    if sampling.top_k is not None:
        body["top_k"] = sampling.top_k
    if sampling.seed is not None:
        body["seed"] = sampling.seed
    if sampling.max_output_tokens is not None:
        body[max_tokens_key] = sampling.max_output_tokens
    if sampling.stop:
        body["stop"] = list(sampling.stop)
    if sampling.repeat_penalty is not None:
        # llama.cpp's own spelling; `repetition_penalty` is vLLM's and llama-server ignores it.
        body["repeat_penalty"] = sampling.repeat_penalty
    return body


def build_completion_body(
    request: GenerationRequest, *, stream: bool, lora: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build the native ``POST /completion`` body for a completion-style request.

    The prompt is sent raw — no template is applied, which is what a caller choosing ``prompt``
    over ``messages`` asked for. JSON mode is ``json_schema: {"type": "object"}``, the way to ask
    llama.cpp's grammar engine for any object; a schema is passed under the same key verbatim.
    Tool definitions are not expressible on this endpoint at all, and the adapter refuses them
    before this function is reached.
    """
    body: dict[str, Any] = {"prompt": request.prompt, "stream": stream}
    if lora is not None:
        body["lora"] = lora
    body.update(_sampling_fields(request, max_tokens_key="n_predict"))
    if request.response_format is not None:
        if request.response_format.kind is ResponseFormatKind.JSON:
            body["json_schema"] = {"type": "object"}
        elif request.response_format.kind is ResponseFormatKind.JSON_SCHEMA:
            body["json_schema"] = dict(request.response_format.schema or {})
    body.update(request_options(request.runtime_profile))
    return body


def build_chat_body(
    request: GenerationRequest,
    *,
    alias: str,
    stream: bool,
    lora: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``POST /v1/chat/completions`` body for a chat-style request.

    The OpenAI chat shape, with two llama.cpp-specific choices: ``repeat_penalty`` rather than
    ``repetition_penalty`` (the name llama-server actually reads), and, when streaming,
    ``stream_options.include_usage`` — without it llama-server's final chunk carries no usage
    object and every count would honestly be ``UNSUPPORTED`` for a stream that had one to give.
    """
    body: dict[str, Any] = {
        "model": alias,
        "messages": [message_payload(message) for message in request.messages],
        "stream": stream,
    }
    if lora is not None:
        body["lora"] = lora
    if stream:
        body["stream_options"] = {"include_usage": True}
    body.update(_sampling_fields(request, max_tokens_key="max_tokens"))
    if request.tools:
        body["tools"] = request_tool_definitions(request.tools)
    if request.response_format is not None:
        body["response_format"] = response_format_payload(request.response_format)
    body.update(request_options(request.runtime_profile))
    return body


# --------------------------------------------------------------------------------- responses


def _as_token_count(value: object) -> TokenCount:
    """Return ``value`` if it is a non-negative whole number, else ``UNSUPPORTED``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return UNSUPPORTED
    return value


def _as_duration_ms(value: object) -> Measurement:
    """Return a non-negative finite millisecond figure, else ``UNSUPPORTED``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return UNSUPPORTED
    if not math.isfinite(value) or value < 0:
        return UNSUPPORTED
    return value


def _cached_from_timings(payload: Mapping[str, Any]) -> TokenCount | None:
    """Return ``timings.cache_n`` if the key is present (readable or not), else ``None``.

    ``None`` means *this response carries no cached-input figure*; ``UNSUPPORTED`` means it
    carries one this adapter could not read. The two are the second and third rows of the
    module docstring's table, and conflating them is the fabricated zero ADR-0016 forbids.
    """
    timings = payload.get("timings")
    if isinstance(timings, Mapping) and "cache_n" in timings:
        return _as_token_count(timings.get("cache_n"))
    return None


def _usage(
    *,
    has_counts: bool,
    prompt_tokens: TokenCount,
    output_tokens: TokenCount,
    cached_tokens: TokenCount | None,
    text: str,
) -> GenerationUsage:
    """Apply the module docstring's table to one response's figures."""
    if not has_counts:
        tokens = TokenUsage()
    else:
        input_tokens: TokenCount
        cache_read: TokenCount
        if cached_tokens is None:
            input_tokens, cache_read = prompt_tokens, 0
        elif (
            not is_supported(cached_tokens)
            or not is_supported(prompt_tokens)
            or cached_tokens > prompt_tokens
        ):
            input_tokens, cache_read = UNSUPPORTED, UNSUPPORTED
        else:
            input_tokens, cache_read = prompt_tokens - cached_tokens, cached_tokens
        tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=0,
            cache_read_tokens=cache_read,
        )
    return GenerationUsage(
        tokens=tokens,
        output_chars=len(text),
        output_words=len(text.split()),
        output_bytes=len(text.encode("utf-8")),
    )


def read_completion_usage(payload: Mapping[str, Any], *, text: str) -> GenerationUsage:
    """Read usage from a native ``/completion`` response, to the rule in the module docstring.

    ``tokens_evaluated`` is the prompt total and ``tokens_predicted`` the output; the cached
    figure is ``timings.cache_n`` and **never** ``tokens_cached`` (see the module docstring for
    why that field is a trap). A payload carrying neither count key is this shape's "no usage
    object" and reports every class ``UNSUPPORTED``.
    """
    has_counts = "tokens_evaluated" in payload or "tokens_predicted" in payload
    return _usage(
        has_counts=has_counts,
        prompt_tokens=_as_token_count(payload.get("tokens_evaluated")),
        output_tokens=_as_token_count(payload.get("tokens_predicted")),
        cached_tokens=_cached_from_timings(payload) if has_counts else None,
        text=text,
    )


def read_chat_usage(payload: Mapping[str, Any], *, text: str) -> GenerationUsage:
    """Read usage from a chat response, to the rule in the module docstring.

    ``usage.prompt_tokens`` and ``usage.completion_tokens`` are the totals. The cached figure is
    ``timings.cache_n`` where the response carries timings; otherwise
    ``usage.prompt_tokens_details.cached_tokens``, the same number under the OpenAI spelling,
    which llama-server writes from the same server-side counter. An absent, ``null``, non-mapping
    or empty ``usage`` is "no usage object" — the empty case matters because the streaming path
    accumulates usage into a mapping that stays empty when no usage chunk ever arrived.
    """
    raw_usage = payload.get("usage")
    usage = raw_usage if isinstance(raw_usage, Mapping) and raw_usage else None
    if usage is None:
        return _usage(
            has_counts=False,
            prompt_tokens=UNSUPPORTED,
            output_tokens=UNSUPPORTED,
            cached_tokens=None,
            text=text,
        )
    cached = _cached_from_timings(payload)
    if cached is None and "prompt_tokens_details" in usage:
        details = usage.get("prompt_tokens_details")
        cached = (
            _as_token_count(details.get("cached_tokens"))
            if isinstance(details, Mapping)
            else UNSUPPORTED
        )
    return _usage(
        has_counts=True,
        prompt_tokens=_as_token_count(usage.get("prompt_tokens")),
        output_tokens=_as_token_count(usage.get("completion_tokens")),
        cached_tokens=cached,
        text=text,
    )


def read_backend_timing(payload: Mapping[str, Any]) -> Timing:
    """Extract the server's own account of a call's cost from its ``timings`` object.

    ``prompt_ms`` is prompt evaluation and ``predicted_ms`` is decoding, both already in
    milliseconds. llama-server reports no load time on a generation and no total, so those stay
    ``UNSUPPORTED`` rather than being summed here (spec §11.3: never this package's arithmetic
    in the provider's column). Client-observed fields are never set here.
    """
    timings = payload.get("timings")
    if not isinstance(timings, Mapping):
        return Timing()
    return Timing(
        backend_prompt_eval_ms=_as_duration_ms(timings.get("prompt_ms")),
        backend_decode_ms=_as_duration_ms(timings.get("predicted_ms")),
    )


def completion_finish_reason(payload: Mapping[str, Any]) -> FinishReason:
    """Map a native response's ``stop_type`` to the vocabulary's finish reason.

    ``eos`` and ``word`` are the model ending its turn (or a stop sequence) — ``STOP``;
    ``limit`` is ``n_predict`` reached — ``LENGTH``; ``none`` and anything unrecognised are
    ``UNKNOWN``, never assumed to be a clean stop.
    """
    stop_type = payload.get("stop_type")
    if isinstance(stop_type, str):
        return _STOP_TYPES.get(stop_type, FinishReason.UNKNOWN)
    return FinishReason.UNKNOWN


def read_build_info(props: Mapping[str, Any]) -> str | None:
    """Return ``/props``'s ``build_info`` — ``"b10792-3e1f9a2c"`` — or ``None`` if absent."""
    value = props.get("build_info")
    return value if isinstance(value, str) and value else None


def read_served_context(props: Mapping[str, Any]) -> Measurement:
    """Return the context size a server reports serving, from ``/props``.

    ``default_generation_settings.n_ctx`` is the per-slot context the server actually built —
    the *reported* served context ADR-0023 §4 prefers over an assumed one — and ``UNSUPPORTED``
    when the key is missing or malformed.
    """
    settings = props.get("default_generation_settings")
    if not isinstance(settings, Mapping):
        return UNSUPPORTED
    return _as_int(settings.get("n_ctx"))


# ------------------------------------------------------------------------------------ errors


@dataclass(frozen=True, slots=True)
class LlamaCppError:
    """One error object as llama-server sends it, in either of its two shapes.

    Attributes:
        message: The server's own message.
        error_type: ``invalid_request_error``, ``exceed_context_size_error``,
            ``unavailable_error``, ``server_error``, … — or ``None`` for the bare-string shape.
        status_code: The HTTP status the server put inside the object, where it did.
        n_prompt_tokens: On a context overflow, how many tokens the request needed.
        n_ctx: On a context overflow, how many the server serves.
    """

    message: str
    error_type: str | None = None
    status_code: int | None = None
    n_prompt_tokens: int | None = None
    n_ctx: int | None = None

    @property
    def is_context_overflow(self) -> bool:
        """Whether this names a context-window overflow.

        The structured ``exceed_context_size_error`` type is the primary signal; the message
        markers are the fallback for a build that reports the condition as a plain
        ``server_error`` with prose, matched conservatively.
        """
        if self.error_type == _CONTEXT_OVERFLOW_TYPE:
            return True
        lowered = self.message.lower()
        return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)

    @property
    def is_not_ready(self) -> bool:
        """Whether the server answered that it is still loading (``unavailable_error``)."""
        return self.error_type == _UNAVAILABLE_TYPE


def read_error(payload: object) -> LlamaCppError | None:
    """Return the error a parsed body carries, or ``None`` if it carries none.

    Handles both shapes llama-server uses: ``{"error": {"code", "message", "type", …}}`` from
    the server proper, and ``{"error": "<text>"}`` from a few paths that predate it. The
    context-overflow fields ``n_prompt_tokens`` and ``n_ctx`` are read when present, which is
    what lets :class:`~modelrack.errors.ContextLimitExceeded` carry real numbers here where the
    Ollama adapter can only carry ``UNSUPPORTED``.
    """
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if isinstance(error, str) and error:
        return LlamaCppError(message=error)
    if not isinstance(error, Mapping):
        return None
    message = error.get("message")
    if not isinstance(message, str) or not message:
        return None
    error_type = error.get("type")
    code = error.get("code")
    n_prompt = error.get("n_prompt_tokens")
    n_ctx = error.get("n_ctx")
    return LlamaCppError(
        message=message,
        error_type=error_type if isinstance(error_type, str) else None,
        status_code=code if isinstance(code, int) and not isinstance(code, bool) else None,
        n_prompt_tokens=n_prompt
        if isinstance(n_prompt, int) and not isinstance(n_prompt, bool)
        else None,
        n_ctx=n_ctx if isinstance(n_ctx, int) and not isinstance(n_ctx, bool) else None,
    )
