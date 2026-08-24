"""Domain module — the declarative vocabulary that describes what ``FakeProvider`` should do.

Imports :mod:`baseaicore` and this package's own types; performs no I/O, reads no clock and
generates nothing. Every type here is a frozen value object a test writes down; the behaviour that
reads them lives in :mod:`modelrack.providers.fake`.

The development plan names one file for Phase 2. These types are separated from the provider that
interprets them because together they are a thousand-line module — the "god module" the
[coding standards](../../../docs/standards/coding-standards.md) §13 name as an anti-pattern.
:mod:`modelrack.providers.fake` re-exports every name defined here, and
:mod:`modelrack.testing` is the supported import path, so the split is invisible to callers.

**A script cannot describe a dishonest provider.** Every cross-field rule below refuses a
combination the fake could only honour by lying: reasoning content scripted onto a provider that
declares it cannot report reasoning, token counts scripted onto one that declares it reports none,
a capability declared that the fake has no way to perform. The alternative — accepting the setting
and quietly dropping it — is the exact failure
[ADR-0007](../../../docs/adr/0007-provider-abstraction.md) rule 2 forbids in a real adapter, and a
fake that permitted it would teach three applications to expect it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from baseaicore import (
    UNSUPPORTED,
    Measurement,
    ModelCapabilityFlag,
    TokenCount,
    ValidationError,
    is_supported,
)

from modelrack.provider import ProviderCapabilities, ProviderStatus
from modelrack.types import FinishReason, Timing

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DEFAULT_MODEL",
    "FULL_CAPABILITIES",
    "MINIMAL_CAPABILITIES",
    "SIMULATED_TOKEN_CHARACTERS",
    "FakeFailure",
    "FakeFailureMode",
    "FakeGeneration",
    "FakeModel",
    "FakeScript",
    "FakeToolCall",
]

FULL_CAPABILITIES: Final[ProviderCapabilities] = ProviderCapabilities(
    streaming=True,
    tool_calling=True,
    structured_output=True,
    json_mode=True,
    token_counts=True,
    token_level_chunks=True,
    thinking_control=True,
    force_unload=True,
    residency_query=True,
    context_configurable=True,
)
"""Everything the fake can actually do — and nothing it cannot.

``logprobs``, ``kv_metrics`` and ``embedding`` stay ``False`` because this package's vocabulary has
nowhere to put a log probability, a KV-cache counter or an embedding vector: a fake declaring them
would be advertising a capability whose result no caller could read. :class:`FakeScript` refuses
them for the same reason.

This is the default declaration, so an unscripted :class:`~modelrack.providers.fake.FakeProvider`
behaves like a capable local runtime. It is also the more forgiving of the two constants, and the
[audit](../../../docs/reviews/final_architecture_audit.md) §11.3 names a too-forgiving fake as this
package's residual risk — which is why :data:`MINIMAL_CAPABILITIES` exists one line away.
"""

MINIMAL_CAPABILITIES: Final[ProviderCapabilities] = ProviderCapabilities()
"""The weakest provider a caller must survive: every flag ``False``.

The shape a bare OpenAI-compatible endpoint will declare in Phase 4 — no streaming, no tools, no
schema, no token counts, no residency control, and no ability to set a served context. A consumer
that has only ever been tested against :data:`FULL_CAPABILITIES` has not been tested against the
degradation matrix, and swapping this constant in is what makes that one line of work.
"""

# 64 hex characters spelling something obviously synthetic: a digest a reader recognises as the
# fake's own rather than mistaking for a real model's, while still normalizing cleanly through
# `baseaicore.normalize_digest` (ADR-0024 §2).
_DEFAULT_DIGEST: Final[str] = "sha256:" + "fa4e" * 16

_MAXIMUM_CHUNK_CHARACTERS: Final[int] = 1 << 20
_CLIENT_TIMING_FIELDS: Final[tuple[str, ...]] = ("client_wall_ms", "client_ttft_ms")

SIMULATED_TOKEN_CHARACTERS: Final[int] = 4
"""How many characters the fake counts as one token.

The fake runs no model and owns no tokenizer, so its token counts are a documented simulation:
every count it reports is the character length of the text divided by this number and rounded up.
Published rather than hidden because a consumer asserting an expected token count needs the same
arithmetic the fake used, and a test that hard-codes ``13`` because that is what came out is a test
that will be "fixed" the first time the constant changes.

It is also the width of one streamed delta by default, which is what lets the fake declare
``token_level_chunks`` truthfully: one :class:`~modelrack.streaming.TokenDelta` per simulated
token, so the count of deltas and the reported output token count of the answer agree exactly.
"""


class FakeFailureMode(StrEnum):
    """A failure a script can ask the fake to produce, one per row of the spec's error table.

    Each member maps to exactly one typed error from :mod:`modelrack.errors` with the ``details``
    keys [spec §13](../../../docs/packages/modelrack/spec.md) requires of it, so
    "every error row is produced by a test" (spec §20 criterion 6) is reachable without a provider.

    Two rows of that table have deliberately **no** member here, because they are not the
    provider's to script:

    * ``CapabilityUnsupported`` is raised by the fake's own gating when a request asks for
      something the declared :class:`~modelrack.provider.ProviderCapabilities` refuse. Scripting it
      separately would let a fake raise it while declaring the capability, which is the
      contradiction the whole capability mechanism exists to prevent.
    * ``GenerationCancelled`` comes from the caller's own
      :class:`~modelrack.streaming.CancellationToken`. A provider cannot decide to be cancelled.
    """

    UNAVAILABLE = "unavailable"
    """Nothing is listening. Produces ``ProviderUnavailable`` with ``base_url`` and ``reason``."""

    TIMEOUT = "timeout"
    """The provider never answered. Produces ``ProviderTimeout`` with elapsed and limit."""

    UNPARSEABLE_BODY = "unparseable_body"
    """Not JSON at all. Produces ``ProviderProtocolError`` with a truncated ``body``."""

    UNEXPECTED_SHAPE = "unexpected_shape"
    """Valid JSON of the wrong shape — the half of spec §13's parse row that still parses.

    Kept apart from :attr:`UNPARSEABLE_BODY` because the two need different responses: a body that
    is not JSON usually means an error page or a wrong port, while JSON missing a field usually
    means a provider version this adapter has not been taught. Both produce
    ``ProviderProtocolError``; only the ``details`` differ.
    """

    TRUNCATED_STREAM = "truncated_stream"
    """The stream stopped without its terminal chunk. Produces ``ProviderProtocolError``.

    Only meaningful for :meth:`~modelrack.provider.Provider.stream`; from
    :meth:`~modelrack.provider.Provider.generate` it is indistinguishable from any other
    mid-response failure and produces the same error.
    """

    MODEL_NOT_FOUND = "model_not_found"
    """The provider does not have it. Produces ``ModelNotFound`` with the reference and count."""

    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    """The request needed more context than would be served. Produces ``ContextLimitExceeded``."""

    REJECTED = "rejected"
    """The provider understood and refused. Produces ``ProviderRejected`` with its own message."""


@dataclass(frozen=True, slots=True)
class FakeFailure:
    """When a scripted generation fails, and how.

    Attributes:
        mode: Which failure to produce.
        after_chunks: How many stream deltas to emit before failing. ``None`` means the call fails
            before producing anything, which is the one case
            :meth:`~modelrack.provider.Provider.stream` **raises** rather than yielding
            :class:`~modelrack.streaming.StreamFailed` — there is no stream to terminate, exactly
            as :mod:`modelrack.streaming` documents. Any number, ``0`` included, means the stream
            began, so the failure arrives as the terminal event with the partial text beside it.
            :meth:`~modelrack.provider.Provider.generate` raises either way: a blocking call has no
            deltas and therefore no boundary at which a partial result could be handed back.
        message: The error message. ``None`` uses a documented default naming the mode.
        details: Overrides merged over the ``details`` the mode produces by default. The defaults
            already satisfy [spec §13](../../../docs/packages/modelrack/spec.md); this is for a
            test that needs a *particular* status code or token limit rather than a plausible one.
    """

    mode: FakeFailureMode
    after_chunks: int | None = None
    message: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the failure point.

        Raises:
            ValidationError: If ``after_chunks`` is negative or not a whole number. A failure
                before the stream starts is ``None``, not ``-1`` — a magic negative would be the
                "no magic return values" rule broken in the type that describes failures.
        """
        if self.after_chunks is None:
            return
        if isinstance(self.after_chunks, bool) or not isinstance(self.after_chunks, int):
            raise ValidationError(
                f"FakeFailure.after_chunks must be a whole number of deltas or None; got "
                f"{self.after_chunks!r}.",
                details={"field": "after_chunks", "value": repr(self.after_chunks)},
            )
        if self.after_chunks < 0:
            raise ValidationError(
                f"FakeFailure.after_chunks must not be negative; got {self.after_chunks}. Use "
                "None for a failure that happens before the stream produces anything.",
                details={"field": "after_chunks", "value": self.after_chunks},
            )


@dataclass(frozen=True, slots=True)
class FakeToolCall:
    """A tool call the fake should request, including the ways a real model gets it wrong.

    Attributes:
        name: The tool the model asks for. Need not exist in the request's ``tools`` — models ask
            for tools that were never offered, and a caller that assumed otherwise would crash on
            the first hallucinated call.
        arguments: The parsed arguments. Left empty and paired with an unparseable
            ``raw_arguments``, this is the malformed-argument case: the fake reports no arguments
            and preserves the text, which is what a real adapter does and what FreeWeight scores
            as a failure it must be able to see.
        raw_arguments: The argument text as the provider would have sent it. ``None`` renders
            :attr:`arguments` as canonical JSON. Supplied alone, it is parsed — and when it will
            not parse, :attr:`arguments` stays empty rather than the call being dropped.
        id: The provider's call identifier. ``None`` synthesizes a deterministic one, because a
            turn with two outstanding calls cannot match results to calls without one.
    """

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        """Validate the tool name and, when supplied, the call id.

        Raises:
            ValidationError: If ``name`` is blank, or ``id`` is supplied and blank.
        """
        if not self.name or not self.name.strip():
            raise ValidationError(
                f"FakeToolCall.name must name the tool the model asks for; got {self.name!r}.",
                details={"field": "name", "value": self.name},
            )
        if self.id is not None and not self.id.strip():
            raise ValidationError(
                f"FakeToolCall.id must be non-empty when supplied; got {self.id!r}. Use None to "
                "have the fake synthesize a deterministic one.",
                details={"field": "id", "value": self.id},
            )


@dataclass(frozen=True, slots=True)
class FakeModel:
    """One entry in the fake's model catalogue: an identity, a digest, and what a provider says.

    The metadata fields mirror :class:`~baseaicore.ModelDescriptor` one for one, and deliberately
    so. ``layers``, ``kv_heads`` and ``head_dim`` are what FreeWeight's KV-cache benchmark computes
    a theoretical bytes-per-token from; a catalogue that could not express one of them individually
    could not be used to test that the benchmark returns ``UNSUPPORTED`` instead of a number built
    from a guess — which is the whole behaviour under test.

    Attributes:
        name: The model name exactly as this provider reports it.
        digest: The digest **as a provider would report it** — bare hex, ``sha256:``-prefixed,
            uppercase, truncated, or not hexadecimal at all. It is normalized on the way out
            through :func:`baseaicore.normalize_digest`, and one that will not normalize is
            discarded with a recorded reason, yielding a ``name_only`` identity rather than a
            malformed one
            ([ADR-0024 §2](../../../docs/adr/0024-canonical-id-and-model-references.md)).
            Scripting a bad digest here is how a consumer's ``name_only`` handling gets tested.
        aliases: Other names this model answers to. Resolving through one is recorded, never
            hidden ([spec §11.8](../../../docs/packages/modelrack/spec.md)).
        family: The model family, e.g. ``"qwen3.5"``.
        architecture: The architecture name, e.g. ``"transformer"``.
        parameter_count: Total parameters.
        active_parameter_count: Mixture-of-experts active parameters per token.
        expert_count: Number of experts.
        quantization: Weight quantization, e.g. ``"Q8_0"``.
        weight_format: File format, e.g. ``"gguf"``.
        size_bytes: On-disk size of the weights.
        max_context: The context the model advertises — never the context actually served,
            which is a runtime concern
            ([ADR-0023 §4](../../../docs/adr/0023-runtime-profile-resolution.md)).
        embedding_dim: Hidden dimension.
        layers: Transformer layer count.
        attention_heads: Attention head count.
        kv_heads: Key/value head count.
        head_dim: Dimension of each attention head.
        vocab_size: Tokenizer vocabulary size.
        rope_config: RoPE scaling configuration in the provider's own shape.
        sliding_window: Sliding-attention window size.
        declared_capabilities: What the provider *claims* this model can do — never what it has
            been measured doing.
        license_text: The model's licence, where a provider exposes one.
        vram_bytes: Device memory this model occupies once resident. Per device, never summed
            ([ADR-0027](../../../docs/adr/0027-multi-gpu-semantics.md)).
        total_bytes: Device and host memory together once resident.
        load_ms: How long this model takes to load. ``UNSUPPORTED`` produces a
            :class:`~modelrack.provider.LoadResult` that says so rather than one claiming an
            instantaneous load.
        raw: The provider payload to preserve on the descriptor. Empty synthesizes one naming the
            catalogue entry and, when a digest was discarded, why.
    """

    name: str
    digest: str | None = None
    aliases: tuple[str, ...] = ()
    family: str | None = None
    architecture: str | None = None
    parameter_count: Measurement = UNSUPPORTED
    active_parameter_count: Measurement = UNSUPPORTED
    expert_count: Measurement = UNSUPPORTED
    quantization: str | None = None
    weight_format: str | None = None
    size_bytes: Measurement = UNSUPPORTED
    max_context: Measurement = UNSUPPORTED
    embedding_dim: Measurement = UNSUPPORTED
    layers: Measurement = UNSUPPORTED
    attention_heads: Measurement = UNSUPPORTED
    kv_heads: Measurement = UNSUPPORTED
    head_dim: Measurement = UNSUPPORTED
    vocab_size: Measurement = UNSUPPORTED
    rope_config: Mapping[str, Any] | None = None
    sliding_window: Measurement = UNSUPPORTED
    declared_capabilities: frozenset[ModelCapabilityFlag] = frozenset()
    license_text: str | None = None
    vram_bytes: Measurement = UNSUPPORTED
    total_bytes: Measurement = UNSUPPORTED
    load_ms: Measurement = UNSUPPORTED
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the name and the aliases.

        Raises:
            ValidationError: If ``name`` is blank, if an alias is blank, or if an alias repeats the
                model's own name — an alias that resolves to the thing it names tells a reader the
                catalogue has two ways to reach one model when it has one.
        """
        if not self.name or not self.name.strip():
            raise ValidationError(
                f"FakeModel.name must be the name the provider reports; got {self.name!r}.",
                details={"field": "name", "value": self.name},
            )
        for alias in self.aliases:
            if not alias or not alias.strip():
                raise ValidationError(
                    f"FakeModel.aliases must not contain a blank alias; got {alias!r} on "
                    f"{self.name!r}.",
                    details={"field": "aliases", "model": self.name},
                )
            if alias == self.name:
                raise ValidationError(
                    f"FakeModel {self.name!r} lists its own name as an alias. Remove it: an alias "
                    "exists to name a model that is reached under another name.",
                    details={"field": "aliases", "model": self.name, "alias": alias},
                )


DEFAULT_MODEL: Final[FakeModel] = FakeModel(
    name="fake-model:8b-q8_0",
    digest=_DEFAULT_DIGEST,
    aliases=("fake-model:latest",),
    family="fake-model",
    architecture="transformer",
    parameter_count=8_030_000_000,
    active_parameter_count=8_030_000_000,
    quantization="Q8_0",
    weight_format="gguf",
    size_bytes=8_540_000_000,
    max_context=32_768,
    embedding_dim=4_096,
    layers=32,
    attention_heads=32,
    kv_heads=8,
    head_dim=128,
    vocab_size=151_936,
    declared_capabilities=frozenset(
        {
            ModelCapabilityFlag.TOOLS,
            ModelCapabilityFlag.THINKING,
            ModelCapabilityFlag.STRUCTURED_OUTPUT,
        }
    ),
    license_text="Apache-2.0",
    vram_bytes=9_100_000_000,
    total_bytes=9_400_000_000,
    load_ms=1_850.0,
)
"""The single model an unscripted fake serves.

Fully populated on purpose. A default catalogue entry whose architecture fields were all
``UNSUPPORTED`` would make every downstream test that reads metadata exercise only the absent
path, and the absent path is the easy one — the arithmetic that has to be right is the one that
runs when the numbers are there. ``kv_heads`` differs from ``attention_heads`` for the same
reason: grouped-query attention is the case a KV-cache calculation gets wrong.
"""


@dataclass(frozen=True, slots=True)
class FakeGeneration:
    """What one call to :meth:`generate` or :meth:`stream` does.

    A script holds a sequence of these, consumed one per call, so a multi-stage workflow can be
    handed a different answer — or a different failure — at each step.

    Every field is deterministic. ``text`` left ``None`` produces pseudo-text derived from the
    seed, the model identity and the prompt, so two different prompts get two different answers
    and the same prompt gets the same answer forever. That variation matters: a scorer under test
    against a fake that answers every prompt identically is a scorer that has been tested against
    one string.

    Attributes:
        text: The exact text to produce. Wins over every other setting, including
            ``response_format`` — which is how a test scripts the case that matters most about
            structured output: a model that was asked for JSON and returned prose.
        chunks: The exact stream deltas to produce, joined to form the text. Mutually exclusive
            with ``text``. This is the only way to place a split *inside* a grapheme cluster or an
            emoji sequence, which a caller assembling deltas for display has to survive.
        word_count: How many words of pseudo-text to generate. Only meaningful when neither
            ``text`` nor ``chunks`` is given, and rejected alongside them rather than ignored.
        chunk_size: Characters per stream delta. Defaults to
            :data:`SIMULATED_TOKEN_CHARACTERS`, which is what makes one delta exactly one
            simulated token and lets a script declare ``token_level_chunks`` honestly. Any other
            width — and any hand-placed ``chunks`` — requires that flag to be ``False``, because
            a caller is entitled to divide by the delta count and call the result per-token
            latency ([spec §11.4](../../../docs/packages/modelrack/spec.md)).
        first_chunk_delay_ms: Simulated time before the first delta: the slow-first-token case.
            Becomes ``Timing.client_ttft_ms`` on a streamed call.
        chunk_delay_ms: Simulated time between subsequent deltas. Per *token* while
            ``token_level_chunks`` is declared, since a delta is a token there, and per *chunk*
            otherwise — which is the whole distinction the flag exists to keep honest.
        thinking: Reasoning content, emitted as :class:`~modelrack.streaming.ThinkingDelta` before
            the answer. Requires the script to declare ``thinking_control``.
        tool_calls: Tool calls to request. Requires the script to declare ``tool_calling``.
        finish_reason: Why generation stopped. ``None`` derives it — ``TOOL_CALLS`` when tool calls
            were requested, ``LENGTH`` when ``max_output_tokens`` truncated the answer, otherwise
            ``STOP``.
        failure: A scripted failure, or ``None`` for a call that succeeds.
        input_tokens: Prompt tokens to report. ``None`` derives a count from the rendered prompt.
        output_tokens: Generated tokens to report, reasoning and tool syntax included. ``None``
            derives it from what was actually produced.
        cache_read_tokens: Tokens billed at the cache-hit rate. Subtracted from the derived
            ``input_tokens`` so the four billing classes stay **disjoint**
            ([ADR-0030](../../../docs/adr/0030-model-cost-and-pricing.md)) — reconciling a
            provider's overlapping figures is the adapter's job, and a fake that double-counted
            them would let a consumer's cost arithmetic ship wrong.
        cache_write_tokens: Tokens billed at the cache-creation rate.
        backend_timing: What the provider claims it spent. ``None`` reports every ``backend_*``
            field as ``UNSUPPORTED``, which is the honest default: the fake ran no model and has no
            account of work it did not do. Client timings are never taken from here — they are
            what the fake observed, and they come from the delays above.
    """

    text: str | None = None
    chunks: tuple[str, ...] | None = None
    word_count: int | None = None
    chunk_size: int = SIMULATED_TOKEN_CHARACTERS
    first_chunk_delay_ms: float = 0.0
    chunk_delay_ms: float = 0.0
    thinking: str | None = None
    tool_calls: tuple[FakeToolCall, ...] = ()
    finish_reason: FinishReason | None = None
    failure: FakeFailure | None = None
    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    cache_read_tokens: TokenCount | None = None
    cache_write_tokens: TokenCount | None = None
    backend_timing: Timing | None = None

    def __post_init__(self) -> None:
        """Validate that the generation describes one coherent call.

        Raises:
            ValidationError: If ``text`` and ``chunks`` are both given; if ``word_count`` is given
                alongside either; if ``chunk_size`` is not a positive whole number within the
                per-chunk size cap; if a delay is negative or not finite; if a scripted token count
                is not a whole non-negative number; or if ``backend_timing`` carries a client
                measurement, which would be the fake reporting an observation it did not make.
        """
        if self.text is not None and self.chunks is not None:
            raise ValidationError(
                "FakeGeneration takes text or chunks, never both: chunks already spell the text "
                "out, and two sources for one string can disagree.",
                details={"field": "chunks"},
            )
        if self.word_count is not None:
            if self.text is not None or self.chunks is not None:
                raise ValidationError(
                    "FakeGeneration.word_count generates text and cannot be combined with text "
                    "or chunks, which supply it. Drop whichever one is not the intent.",
                    details={"field": "word_count"},
                )
            if isinstance(self.word_count, bool) or not isinstance(self.word_count, int):
                raise ValidationError(
                    f"FakeGeneration.word_count must be a whole number of words; got "
                    f"{self.word_count!r}.",
                    details={"field": "word_count", "value": repr(self.word_count)},
                )
            if self.word_count < 0:
                raise ValidationError(
                    f"FakeGeneration.word_count must not be negative; got {self.word_count}. Use "
                    "0 for a response with no text, which is what a tool-call-only turn is.",
                    details={"field": "word_count", "value": self.word_count},
                )
        self._validate_chunking()
        self._validate_delays()
        self._validate_counts()
        self._validate_backend_timing()

    def _validate_chunking(self) -> None:
        """Raise unless the delta size is a positive whole number within the size cap."""
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise ValidationError(
                f"FakeGeneration.chunk_size must be a whole number of characters; got "
                f"{self.chunk_size!r}.",
                details={"field": "chunk_size", "value": repr(self.chunk_size)},
            )
        if self.chunk_size < 1:
            raise ValidationError(
                f"FakeGeneration.chunk_size must be at least 1; got {self.chunk_size}. A delta of "
                "zero characters would never finish a response.",
                details={"field": "chunk_size", "value": self.chunk_size},
            )
        if self.chunk_size > _MAXIMUM_CHUNK_CHARACTERS:
            raise ValidationError(
                f"FakeGeneration.chunk_size must not exceed {_MAXIMUM_CHUNK_CHARACTERS} "
                f"characters; got {self.chunk_size}. The per-chunk cap exists so a fake cannot be "
                "scripted past the size limits spec §14 puts on a real adapter.",
                details={"field": "chunk_size", "value": self.chunk_size},
            )

    def _validate_delays(self) -> None:
        """Raise unless every simulated delay is a finite, non-negative number of milliseconds."""
        for field_name in ("first_chunk_delay_ms", "chunk_delay_ms"):
            value: float = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValidationError(
                    f"FakeGeneration.{field_name} must be a number of milliseconds; got {value!r}.",
                    details={"field": field_name, "value": repr(value)},
                )
            if not math.isfinite(value) or value < 0:
                raise ValidationError(
                    f"FakeGeneration.{field_name} must be finite and not negative; got {value!r}. "
                    "A delay of infinity is a hung test, not a slow provider — script a "
                    "FakeFailureMode.TIMEOUT instead.",
                    details={"field": field_name, "value": repr(value)},
                )

    def _validate_counts(self) -> None:
        """Raise unless every scripted token count is a whole, non-negative number."""
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            value: TokenCount | None = getattr(self, field_name)
            if value is None or value is UNSUPPORTED:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(
                    f"FakeGeneration.{field_name} must be a whole number of tokens, UNSUPPORTED, "
                    f"or None to derive one; got {value!r}.",
                    details={"field": field_name, "value": repr(value)},
                )
            if value < 0:
                raise ValidationError(
                    f"FakeGeneration.{field_name} must not be negative; got {value}.",
                    details={"field": field_name, "value": value},
                )

    def _validate_backend_timing(self) -> None:
        """Raise if the scripted backend account also carries a client-observed measurement."""
        if self.backend_timing is None:
            return
        for field_name in _CLIENT_TIMING_FIELDS:
            if is_supported(getattr(self.backend_timing, field_name)):
                raise ValidationError(
                    f"FakeGeneration.backend_timing.{field_name} is a client-observed value and "
                    "the fake measures its own. Script first_chunk_delay_ms and chunk_delay_ms "
                    "instead; backend_timing carries only what a provider claims about itself "
                    "(spec §11.3).",
                    details={"field": f"backend_timing.{field_name}"},
                )


@dataclass(frozen=True, slots=True)
class FakeScript:
    """A complete declarative description of a fake provider: what it serves and what it does.

    Everything a :class:`~modelrack.providers.fake.FakeProvider` needs beyond its seed. Frozen, so
    a script shared between tests cannot be mutated by one of them, and cheap to vary with
    :func:`dataclasses.replace`.

    Attributes:
        models: The catalogue. May be empty — a runtime with nothing pulled is a real state, and
            the error a caller gets from it should name the count.
        capabilities: What this provider declares. Callers branch on it; they never assume
            ([ADR-0007](../../../docs/adr/0007-provider-abstraction.md) rule 2).
        generations: What successive calls do, consumed one per :meth:`generate` or
            :meth:`stream`.
        repeat_final_generation: Whether the last generation answers every call after it. ``True``
            is the obvious reading of a one-generation script. ``False`` makes exhaustion an
            error, which is what a workflow test wants: a stage that quietly made two extra model
            calls is a defect, and a fake that kept answering would hide it.
        health_status: What :meth:`health` reports. ``UNAVAILABLE`` also makes every other call
            raise ``ProviderUnavailable`` — the whole "Ollama is not running" row of the
            degradation matrix, in one field.
        health_latency_ms: How long the health probe took.
        provider_version: The provider's own version. Changing it between two runs is how a
            consumer's environment-drift handling gets tested
            ([ADR-0017](../../../docs/adr/0017-benchmark-confidence-and-freshness.md)).
        base_url: What :meth:`health` reports it contacted. The default names no socket, because
            the fake opens none.
        is_remote: Whether that URL is somewhere other than loopback. Carried so a caller can
            surface egress rather than discover it from a firewall log.
    """

    models: tuple[FakeModel, ...] = (DEFAULT_MODEL,)
    capabilities: ProviderCapabilities = FULL_CAPABILITIES
    generations: tuple[FakeGeneration, ...] = (FakeGeneration(),)
    repeat_final_generation: bool = True
    health_status: ProviderStatus = ProviderStatus.OK
    health_latency_ms: Measurement = UNSUPPORTED
    provider_version: str | None = "fake-1.0"
    base_url: str = "fake://in-process"
    is_remote: bool = False

    def __post_init__(self) -> None:
        """Validate the catalogue, the declaration, and that neither contradicts the generations.

        Raises:
            ValidationError: If the catalogue names one model twice or reuses a name as another
                model's alias; if the declared capabilities include one the fake cannot perform;
                if ``generations`` is empty; if a generation scripts behaviour the declared
                capabilities refuse; or if ``base_url`` is blank.
        """
        if not self.base_url or not self.base_url.strip():
            raise ValidationError(
                f"FakeScript.base_url must name where the provider would be contacted; got "
                f"{self.base_url!r}. A health result that cannot say where it probed cannot be "
                "acted on.",
                details={"field": "base_url", "value": self.base_url},
            )
        if not self.generations:
            raise ValidationError(
                "FakeScript.generations must describe at least one call; got none. A provider "
                "that cannot answer once is not a provider a test can use.",
                details={"field": "generations"},
            )
        self._validate_catalogue()
        self._validate_declared_capabilities()
        self._validate_generations_against_capabilities()

    def _validate_catalogue(self) -> None:
        """Raise unless every name and alias in the catalogue reaches exactly one model."""
        seen: dict[str, str] = {}
        for model in self.models:
            for reference in (model.name, *model.aliases):
                owner = seen.get(reference)
                if owner is not None:
                    raise ValidationError(
                        f"FakeScript.models reaches {reference!r} through both {owner!r} and "
                        f"{model.name!r}. A reference that names two models cannot be resolved, "
                        "and a provider that picked one would run weights the caller did not ask "
                        "for.",
                        details={"field": "models", "reference": reference},
                    )
                seen[reference] = model.name

    def _validate_declared_capabilities(self) -> None:
        """Raise if the declaration claims something this package's types cannot even carry."""
        unperformable = [
            name
            for name in ("logprobs", "kv_metrics", "embedding")
            if getattr(self.capabilities, name)
        ]
        if unperformable:
            raise ValidationError(
                f"FakeScript.capabilities declares {unperformable!r}, which the fake cannot "
                "perform: this package's result types carry no log probabilities, no KV-cache "
                "counters and no embeddings, so a caller could not read one if it were produced. "
                "A capability nobody can honour is the dishonesty the capability mechanism exists "
                "to prevent (ADR-0007 rule 2).",
                details={"field": "capabilities", "capabilities": unperformable},
            )

    def _validate_generations_against_capabilities(self) -> None:
        """Raise if a generation scripts output the declared capabilities say cannot exist."""
        for index, generation in enumerate(self.generations):
            if generation.thinking is not None and not self.capabilities.thinking_control:
                raise ValidationError(
                    f"FakeScript.generations[{index}] scripts reasoning content while "
                    "capabilities.thinking_control is False. Declare the capability or drop the "
                    "content: a provider that cannot report reasoning reporting some is exactly "
                    "the lie this fake must not teach consumers to expect.",
                    details={"field": "thinking", "generation_index": index},
                )
            if generation.tool_calls and not self.capabilities.tool_calling:
                raise ValidationError(
                    f"FakeScript.generations[{index}] scripts tool calls while "
                    "capabilities.tool_calling is False.",
                    details={"field": "tool_calls", "generation_index": index},
                )
            self._validate_counts_against_capabilities(index, generation)
            self._validate_chunking_against_capabilities(index, generation)

    def _validate_chunking_against_capabilities(
        self, index: int, generation: FakeGeneration
    ) -> None:
        """Raise if a generation chunks in a way that contradicts its ``token_level_chunks`` claim.

        The flag means one delta is one token, and it gates **any** per-token latency figure a
        caller derives ([spec §11.4](../../../docs/packages/modelrack/spec.md)). The fake honours
        it exactly — one :class:`~modelrack.streaming.TokenDelta` per
        :data:`SIMULATED_TOKEN_CHARACTERS` of answer text — which it can only do while the delta
        width is that unit. Hand-placed chunks and any other width make a delta an arbitrary
        fragment, so the claim has to go with them.
        """
        if not self.capabilities.token_level_chunks:
            return
        if generation.chunks is None and generation.chunk_size == SIMULATED_TOKEN_CHARACTERS:
            return
        raise ValidationError(
            f"FakeScript.generations[{index}] chunks the answer into fragments that are not one "
            f"simulated token each, while capabilities.token_level_chunks claims every delta is "
            f"one token. A caller is entitled to divide by the delta count and call the result "
            f"per-token latency (spec §11.4). Declare the truth with "
            f"`dataclasses.replace(FULL_CAPABILITIES, token_level_chunks=False)`, or leave "
            f"chunk_size at SIMULATED_TOKEN_CHARACTERS ({SIMULATED_TOKEN_CHARACTERS}).",
            details={
                "field": "token_level_chunks",
                "generation_index": index,
                "chunk_size": generation.chunk_size,
                "explicit_chunks": generation.chunks is not None,
            },
        )

    def _validate_counts_against_capabilities(self, index: int, generation: FakeGeneration) -> None:
        """Raise if token counts are scripted onto a provider that declares it reports none."""
        if self.capabilities.token_counts:
            return
        scripted = [
            name
            for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
            if getattr(generation, name) is not None
        ]
        if scripted:
            raise ValidationError(
                f"FakeScript.generations[{index}] scripts {scripted!r} while "
                "capabilities.token_counts is False, where every count must be UNSUPPORTED. A "
                "count reported by a provider that declares it counts nothing is a fabricated "
                "measurement (ADR-0016).",
                details={"field": "token_counts", "generation_index": index, "scripted": scripted},
            )
