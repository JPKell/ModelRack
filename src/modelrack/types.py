"""Domain module — the provider-neutral request and result vocabulary.

Imports :mod:`baseaicore` and the standard library; performs no I/O and reads no clock. These are
the types three applications exchange with a model runtime, and their whole purpose is that
FreeWeight, LoadCoach and IdeaPress never see provider JSON
(ADR-0007 rule 1). Every one is a frozen value
object, because a result that some later code mutates no longer describes the call it came from.

Two rules shape almost every field here:

* **Unavailable is never zero** (ADR-0016). Every
  count and duration defaults to :data:`~baseaicore.UNSUPPORTED`, so a provider that reported
  nothing produces a result that says so. A generation recorded as having taken ``0 ms`` or used
  ``0`` tokens is the exact failure this suite is built to refuse.
* **Backend and client measurements never merge**
  ([spec §11.3](../../docs/packages/modelrack/spec.md)). What the provider *said* it spent and
  what the caller *observed* are different facts that
  disagree for real reasons — queueing, transport, the client's own scheduling — and averaging them
  produces a number describing nothing. :class:`Timing` gives each its own prefix and offers no
  combined field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from baseaicore import (
    UNSUPPORTED,
    Measurement,
    ModelIdentity,
    RuntimeProfile,
    TokenCount,
    TokenUsage,
    Unsupported,
    ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from baseaicore import AdapterIdentity, IdentityConfidence

    # Imported for typing only: `streaming` imports `GenerationResult` from this module at
    # runtime, so a runtime import here would close the cycle. `from __future__ import
    # annotations` makes every annotation lazy, so the name is never needed at import time.
    from modelrack.streaming import CancellationToken

__all__ = [
    "FinishReason",
    "GenerationRequest",
    "GenerationResult",
    "GenerationUsage",
    "Message",
    "ResponseFormat",
    "ResponseFormatKind",
    "Role",
    "SamplingParameters",
    "Timing",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
]

_MAXIMUM_PROBABILITY = 1.0


def _validate_quantity(value: Measurement, *, owner: str, field_name: str) -> None:
    """Raise unless a measurement is a non-negative finite number or ``UNSUPPORTED``."""
    if value is UNSUPPORTED:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(
            f"{owner}.{field_name} must be a number or UNSUPPORTED; got {value!r}. Use "
            "UNSUPPORTED when the provider reported nothing — never 0, which is a real "
            "measurement of nothing happening (ADR-0016).",
            details={"field": field_name, "value": repr(value)},
        )
    if not math.isfinite(value):
        raise ValidationError(
            f"{owner}.{field_name} must be finite; got {value!r}. A nan or infinity here is a "
            "failed calculation in an adapter, not a measurement.",
            details={"field": field_name, "value": repr(value)},
        )
    if value < 0:
        raise ValidationError(
            f"{owner}.{field_name} must not be negative; got {value!r}.",
            details={"field": field_name, "value": value},
        )


def _validate_count(value: TokenCount, *, owner: str, field_name: str) -> None:
    """Raise unless a token count is a non-negative whole number or ``UNSUPPORTED``."""
    if value is UNSUPPORTED:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"{owner}.{field_name} must be a whole number of tokens or UNSUPPORTED; got "
            f"{value!r}. A fractional token count is a parsing error in the adapter.",
            details={"field": field_name, "value": repr(value)},
        )
    if value < 0:
        raise ValidationError(
            f"{owner}.{field_name} must not be negative; got {value}.",
            details={"field": field_name, "value": value},
        )


class Role(StrEnum):
    """Who authored a message.

    A ``StrEnum`` so it serializes and logs readably (coding standards §2). The four roles are the
    ones every supported provider understands; a provider that names them differently is the
    adapter's problem to translate, not the caller's.
    """

    SYSTEM = "system"
    """Instructions that frame the whole exchange."""

    USER = "user"
    """Input from the person or application driving the conversation."""

    ASSISTANT = "assistant"
    """Output from the model, including any tool calls it requested."""

    TOOL = "tool"
    """The result of running a tool the assistant asked for, fed back to the model."""


class FinishReason(StrEnum):
    """Why generation stopped.

    Recorded rather than inferred: a response truncated at the token limit and a response the
    model chose to end are the same string of text with entirely different meanings, and a
    benchmark that scored them alike would reward a model for being cut off at a plausible point.
    """

    STOP = "stop"
    """The model ended its turn on its own, or hit a caller-supplied stop sequence."""

    LENGTH = "length"
    """The output token limit was reached. The response is truncated."""

    TOOL_CALLS = "tool_calls"
    """The model ended its turn to request one or more tool calls."""

    CONTENT_FILTER = "content_filter"
    """The provider stopped generation on its own content policy."""

    CANCELLED = "cancelled"
    """A caller's cancellation token fired."""

    ERROR = "error"
    """Generation failed part-way; any text present is partial."""

    UNKNOWN = "unknown"
    """The provider reported a reason this adapter does not recognise, or reported none.

    Explicitly present rather than defaulted to :attr:`STOP`: guessing "the model finished
    normally" is the assumption that silently converts a truncation into a complete answer.
    """


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool offered to the model, described in the provider-neutral shape.

    ``parameters`` is passed to the provider **unmodified** (spec §14): ModelRack does not
    validate, rewrite or infer a schema, and it never executes a tool — a requested call is
    returned to the caller, which owns the decision to run it.

    Attributes:
        name: The tool's name, as the model will refer to it.
        description: What the tool does, in the words the model sees. This is prompt content, and
            it is the caller's to write.
        parameters: A JSON Schema object describing the tool's arguments, passed through verbatim.
    """

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the tool name.

        Raises:
            ValidationError: If ``name`` is empty or only whitespace. A nameless tool cannot be
                matched to the call the model makes for it.
        """
        if not self.name or not self.name.strip():
            raise ValidationError(
                f"ToolDefinition.name must be a non-empty tool name; got {self.name!r}.",
                details={"field": "name", "value": self.name},
            )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation the model requested.

    Attributes:
        id: The provider's identifier for this call, used to correlate the eventual
            :attr:`Role.TOOL` message back to it. Providers that emit no id get one synthesized by
            the adapter, because a multi-call turn cannot be answered without one.
        name: The tool the model wants to run.
        arguments: The parsed arguments. Empty when the model called a tool with none, which is
            different from failing to parse — see :attr:`raw_arguments`.
        raw_arguments: The unparsed argument text exactly as the provider sent it, kept when the
            provider streams arguments as a string. Present so a malformed-argument case is
            diagnosable rather than silently an empty mapping: models do emit invalid JSON here,
            and FreeWeight scores that as a failure it must be able to see.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None

    def __post_init__(self) -> None:
        """Validate the call's identity.

        Raises:
            ValidationError: If ``id`` or ``name`` is empty or only whitespace.
        """
        for field_name in ("id", "name"):
            value: str = getattr(self, field_name)
            if not value or not value.strip():
                raise ValidationError(
                    f"ToolCall.{field_name} must be non-empty; got {value!r}. A tool call that "
                    "cannot be named or correlated cannot be answered.",
                    details={"field": field_name, "value": value},
                )


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a conversation.

    Attributes:
        role: Who authored this turn.
        content: The text. May be empty on an assistant turn that only requested tools — that is
            a real and common response, and a model that rejected it could not express a
            tool-call-only turn at all.
        tool_calls: Tool invocations requested by an assistant turn.
        tool_call_id: On a :attr:`Role.TOOL` message, the :attr:`ToolCall.id` this result answers.
            Required there: a conversation with two outstanding calls cannot match results to
            calls without it.
        name: The tool's name on a :attr:`Role.TOOL` message, where a provider wants it echoed.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate the message against the constraints its role implies.

        Raises:
            ValidationError: If a tool result carries no ``tool_call_id``, if a non-assistant turn
                carries tool calls, or if a turn carries neither content nor tool calls.
        """
        if self.role is Role.TOOL and not self.tool_call_id:
            raise ValidationError(
                "A tool message must carry the tool_call_id it answers; got None. Without it a "
                "turn with two outstanding calls cannot match results to calls.",
                details={"field": "tool_call_id", "role": self.role.value},
            )
        if self.tool_calls and self.role is not Role.ASSISTANT:
            raise ValidationError(
                f"Only an assistant message may carry tool_calls; got role {self.role.value!r}. "
                "A tool call is something the model requests, never something sent to it.",
                details={"field": "tool_calls", "role": self.role.value},
            )
        if not self.content and not self.tool_calls:
            raise ValidationError(
                f"A {self.role.value} message must carry content or tool_calls; got neither. An "
                "empty turn tells the model nothing and costs context to send.",
                details={"field": "content", "role": self.role.value},
            )


class ResponseFormatKind(StrEnum):
    """What shape a caller is asking the model to answer in."""

    TEXT = "text"
    """Ordinary free text. The default."""

    JSON = "json"
    """Valid JSON, with the shape left to the prompt. Providers call this "JSON mode"."""

    JSON_SCHEMA = "json_schema"
    """JSON conforming to a supplied schema, where the provider can enforce it."""


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """How the model should shape its answer.

    Requesting a format a provider has not declared is refused by the adapter with
    :class:`~modelrack.errors.CapabilityUnsupported` rather than silently downgraded, because a
    caller that asked for JSON and received prose would discover it while parsing, one layer too
    late (ADR-0007 rule 2).

    Attributes:
        kind: Which shape is being requested.
        schema: The JSON Schema to enforce. Required for :attr:`ResponseFormatKind.JSON_SCHEMA`
            and forbidden otherwise — a schema attached to a plain-text request is a caller
            mistake that would otherwise be silently ignored.
    """

    kind: ResponseFormatKind = ResponseFormatKind.TEXT
    schema: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate that a schema is present exactly when the kind needs one.

        Raises:
            ValidationError: If ``JSON_SCHEMA`` carries no schema, or another kind carries one.
        """
        if self.kind is ResponseFormatKind.JSON_SCHEMA and self.schema is None:
            raise ValidationError(
                "ResponseFormat(kind=JSON_SCHEMA) requires a schema; got None. Use "
                "ResponseFormatKind.JSON for 'valid JSON, shape unspecified'.",
                details={"field": "schema", "kind": self.kind.value},
            )
        if self.kind is not ResponseFormatKind.JSON_SCHEMA and self.schema is not None:
            raise ValidationError(
                f"ResponseFormat(kind={self.kind.value}) must not carry a schema; a schema here "
                "would be silently ignored. Use ResponseFormatKind.JSON_SCHEMA to enforce it.",
                details={"field": "schema", "kind": self.kind.value},
            )


@dataclass(frozen=True, slots=True)
class SamplingParameters:
    """How the model should sample its output.

    Every field defaults to ``None``, meaning "do not send this — use the provider's default".
    That is deliberately distinct from sending an explicit value that happens to match the
    default: a run that pinned ``temperature=0.0`` and one that inherited whatever the provider
    chose are not the same experiment, and only the first is reproducible
    (Machine Identity §6
    requires the *effective* parameters be recorded).

    Sampling is **not** part of a :class:`~baseaicore.RuntimeProfile`: these change per request,
    while a runtime profile describes how the model was loaded and served
    (ADR-0023).

    Attributes:
        temperature: Randomness. ``0`` is the greedy, most reproducible setting.
        top_p: Nucleus sampling mass, in ``[0, 1]``.
        top_k: Number of candidate tokens considered.
        seed: The sampling seed. Recording one is what makes a run repeatable; a run without one
            records ``"nondeterministic"`` downstream rather than a fabricated number.
        max_output_tokens: The generation limit. Reaching it produces
            :attr:`FinishReason.LENGTH`.
        stop: Sequences that end generation when produced.
        repeat_penalty: Penalty applied to already-seen tokens.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] = ()
    repeat_penalty: float | None = None

    def __post_init__(self) -> None:
        """Validate every supplied parameter against its defensible range.

        Raises:
            ValidationError: If a value is outside the range that gives it meaning. These are
                caught here rather than at the provider because a 4xx from a runtime names the
                provider's own option spelling, not the caller's field.
        """
        if self.temperature is not None and (
            not math.isfinite(self.temperature) or self.temperature < 0
        ):
            raise ValidationError(
                f"temperature must be a finite, non-negative number; got {self.temperature!r}.",
                details={"field": "temperature", "value": repr(self.temperature)},
            )
        if self.top_p is not None and not (0 <= self.top_p <= _MAXIMUM_PROBABILITY):
            raise ValidationError(
                f"top_p is a probability mass and must lie in [0, 1]; got {self.top_p!r}.",
                details={"field": "top_p", "value": repr(self.top_p)},
            )
        for field_name in ("top_k", "max_output_tokens"):
            value: int | None = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValidationError(
                    f"{field_name} must be at least 1 when set; got {value}. Use None to leave "
                    "it to the provider rather than 0, which would ask for nothing at all.",
                    details={"field": field_name, "value": value},
                )
        if self.repeat_penalty is not None and (
            not math.isfinite(self.repeat_penalty) or self.repeat_penalty <= 0
        ):
            raise ValidationError(
                f"repeat_penalty must be a finite number above 0; got {self.repeat_penalty!r}.",
                details={"field": "repeat_penalty", "value": repr(self.repeat_penalty)},
            )


@dataclass(frozen=True, slots=True)
class Timing:
    """What the call cost in time, with the provider's account and the caller's kept apart.

    The separation is [spec §11.3](../../docs/packages/modelrack/spec.md) and it is load-bearing.
    ``backend_*`` is what the provider reported about its own work; ``client_*`` is what this
    process observed from the outside, measured with a monotonic counter. They differ by
    queueing, transport and scheduling, and both are true. There is deliberately **no** combined
    or unprefixed duration field: the moment one exists, callers reach for it and a benchmark
    starts comparing one runtime's self-report against another's wall clock.

    Every field is a :data:`~baseaicore.Measurement`, so a provider that reports no durations
    yields ``UNSUPPORTED`` rather than zeros that would average away real throughput.

    Attributes:
        client_wall_ms: Total elapsed time this process observed for the call.
        client_ttft_ms: Time this process observed before the first token arrived. Meaningful only
            for a streamed call; a non-streaming call has no first-token moment to observe.
        backend_load_ms: Time the provider reported loading the model.
        backend_prompt_eval_ms: Time the provider reported evaluating the prompt.
        backend_decode_ms: Time the provider reported generating output tokens.
        backend_total_ms: The provider's own total. Not a sum of the other three — a provider may
            account for time none of them covers, and recomputing it here would overwrite the
            provider's account with this package's arithmetic.
    """

    client_wall_ms: Measurement = UNSUPPORTED
    client_ttft_ms: Measurement = UNSUPPORTED
    backend_load_ms: Measurement = UNSUPPORTED
    backend_prompt_eval_ms: Measurement = UNSUPPORTED
    backend_decode_ms: Measurement = UNSUPPORTED
    backend_total_ms: Measurement = UNSUPPORTED

    def __post_init__(self) -> None:
        """Validate every duration.

        Raises:
            ValidationError: If a duration is negative, non-finite, or not a number. A negative
                duration means two readings came from different clocks, which is a defect worth
                failing on rather than reporting.
        """
        for field_name in (
            "client_wall_ms",
            "client_ttft_ms",
            "backend_load_ms",
            "backend_prompt_eval_ms",
            "backend_decode_ms",
            "backend_total_ms",
        ):
            _validate_quantity(getattr(self, field_name), owner="Timing", field_name=field_name)


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    """What the call consumed: the billable counts, plus what was observed about the output.

    Two kinds of fact live here, and the split is the reason this type exists rather than
    :class:`~baseaicore.TokenUsage` alone:

    * :attr:`tokens` is BaseAiCore's billing vocabulary, whose four classes are **disjoint** by
      definition so that a cost can be computed without double-billing a cached call
      (ADR-0030). Reconciling each provider's
      convention into that shape is the adapter's job — ADR-0030 names it as a conformance test
      case — and a consumer storing a cost stores this object directly.
    * The remaining fields are **observations**, not billing: breakdowns and output sizes the three
      applications persist per sample
      (FreeWeight's ``samples`` table, LoadCoach's and IdeaPress's job rows).

    ``thinking_tokens`` is therefore a *breakdown of* :attr:`TokenUsage.output_tokens`, not a
    fifth disjoint class — every provider that exposes reasoning tokens bills them at its output
    rate, which is exactly why BaseAiCore declines to give them a billing field. Adding them to a
    total computed from :attr:`tokens` would count them twice.

    Attributes:
        tokens: The disjoint, billable counts.
        thinking_tokens: Reasoning tokens, already included in ``tokens.output_tokens``.
        tool_tokens: Tokens spent on tool-call syntax, already included in
            ``tokens.output_tokens``.
        output_chars: Characters generated.
        output_words: Whitespace-delimited words generated.
        output_bytes: UTF-8 bytes generated. Distinct from ``output_chars`` because a multi-byte
            response is larger on the wire than its character count suggests.
    """

    tokens: TokenUsage = field(default_factory=TokenUsage)
    thinking_tokens: TokenCount = UNSUPPORTED
    tool_tokens: TokenCount = UNSUPPORTED
    output_chars: Measurement = UNSUPPORTED
    output_words: Measurement = UNSUPPORTED
    output_bytes: Measurement = UNSUPPORTED

    def __post_init__(self) -> None:
        """Validate the observation counts.

        ``tokens`` validates itself: :class:`~baseaicore.TokenUsage` rejects a negative,
        fractional or boolean count in its own ``__post_init__``.

        Raises:
            ValidationError: If a count is negative, fractional where it must be whole, or not a
                number.
        """
        for field_name in ("thinking_tokens", "tool_tokens"):
            _validate_count(
                getattr(self, field_name), owner="GenerationUsage", field_name=field_name
            )
        for field_name in ("output_chars", "output_words", "output_bytes"):
            _validate_quantity(
                getattr(self, field_name), owner="GenerationUsage", field_name=field_name
            )


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Everything one call to a model needs, in provider-neutral form.

    Supply **either** ``messages`` (chat-style) **or** ``prompt`` (completion-style), never both
    and never neither: they map to different provider endpoints, and an adapter handed both would
    have to pick one and silently discard the other.

    Attributes:
        identity: Which weights to run
            (ADR-0008).
        messages: The conversation, for a chat-style call.
        prompt: The raw prompt, for a completion-style call.
        runtime_profile: How the model should be loaded and served. Defaults to provider defaults,
            which is itself a legal, hashable profile — there is no "no profile" state
            (ADR-0023 §1).
        sampling: How the model should sample. Per-request, unlike ``runtime_profile``.
        tools: Tools the model may call. Passing any to a provider that has not declared
            ``tool_calling`` raises :class:`~modelrack.errors.CapabilityUnsupported`.
        response_format: The shape the answer should take.
        timeout_seconds: The limit for this call. ``None`` means "use the adapter's configured
            default", never "no timeout" (spec §14) — an unbounded generation is a hung
            application.
        cancel: A token that stops a streamed generation. Effective only through
            :meth:`~modelrack.provider.Provider.stream`; see
            :class:`~modelrack.errors.GenerationCancelled`.
        adapter: The **name** of a registered LoRA adapter to run this request under, or ``None``
            for the bare base. A pin, with ``model``-override semantics: the provider resolves it
            against what it has registered and raises
            :class:`~modelrack.errors.AdapterNotFound` rather than falling back. One adapter, never
            two — the field is single-valued because composition would need an identity for a
            weighted set, which ADR-0063 declines to
            invent — and its scale is fixed at ``1.0`` and is deliberately not a parameter.
            Passing one to a provider that has not declared ``adapter_hot_swap`` raises
            :class:`~modelrack.errors.CapabilityUnsupported`.
        metadata: Caller correlation IDs. **Never sent to the provider** — it travels with the
            request so an ``on_event`` callback and a returned result can be tied back to the
            caller's own run or job, and putting it on the wire would leak internal identifiers.
    """

    identity: ModelIdentity
    messages: tuple[Message, ...] = ()
    prompt: str | None = None
    adapter: str | None = None
    runtime_profile: RuntimeProfile = field(default_factory=RuntimeProfile)
    sampling: SamplingParameters = field(default_factory=SamplingParameters)
    tools: tuple[ToolDefinition, ...] = ()
    response_format: ResponseFormat | None = None
    timeout_seconds: float | None = None
    cancel: CancellationToken | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the request names exactly one input and a usable timeout.

        Raises:
            ValidationError: If both or neither of ``messages`` and ``prompt`` are given, or if
                ``timeout_seconds`` is not a positive finite number.
        """
        has_prompt = self.prompt is not None
        if bool(self.messages) == has_prompt:
            supplied = "both messages and prompt" if has_prompt else "neither messages nor prompt"
            raise ValidationError(
                f"A GenerationRequest carries exactly one input; got {supplied}. Chat-style and "
                "completion-style calls reach different provider endpoints, so an adapter given "
                "both would have to discard one silently.",
                details={"has_messages": bool(self.messages), "has_prompt": has_prompt},
            )
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValidationError(
                f"timeout_seconds must be a finite number above 0 when set; got "
                f"{self.timeout_seconds!r}. Use None for the adapter's default — never 0, and "
                "never infinity: a call with no ceiling is a hung application (spec §14).",
                details={"field": "timeout_seconds", "value": repr(self.timeout_seconds)},
            )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The complete outcome of one non-streamed call, or of an assembled stream.

    Attributes:
        text: The generated text. Legitimately empty when the model answered only with tool calls.
        identity: The weights that produced this. Carried on the result, not merely known by the
            caller, so a stored result stays attributable after it leaves the call site.
        finish_reason: Why generation stopped.
        usage: What the call consumed.
        timing: What it cost in time.
        tool_calls: Tool invocations the model requested.
        thinking: Reasoning content, where the provider exposes it. ``UNSUPPORTED`` when the
            provider cannot report it — distinct from ``""``, which would claim the model reasoned
            and produced nothing.
        provider_version: The provider's own version, recorded because a provider upgrade is an
            environment drift signal that reduces confidence in evidence measured before it
            (ADR-0017).
        adapter: The adapter axis of the subject that actually ran, or ``None`` when none was
            applied. Carried for the same reason ``identity`` is: a stored result must name its
            **whole** subject, because evidence measured on ``(base, adapterA)`` applies to nothing
            else (ADR-0058 §4), and a
            result that named only its base would let an adapter's numbers be read as the base's.
            ``None`` is the byte-for-byte-unchanged case — a subject with no adapter is exactly
            what it was before the axis existed.
        adapter_base_confidence: How well this adapter's claim about its base was proved:
            ``DIGEST`` when the manifest declared a digest and it matched the base actually served,
            ``NAME_ONLY`` when it declared none and only the names agreed. ``None`` when no adapter
            ran. A ``NAME_ONLY`` result is a **permanent caveat** that every surface naming this
            subject must show (ADR-0058 rule 5) — it is about the *base claim*, never about the
            adapter's own identity, which is always digest-bound.
        raw: The provider's untouched response, for **diagnostics only**. Reading it for business
            logic is a boundary violation (ADR-0007
            rule 1); it exists so a surprising result can be explained, and adapters must keep API
            keys out of it (spec §14).
    """

    text: str
    identity: ModelIdentity
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: GenerationUsage = field(default_factory=GenerationUsage)
    timing: Timing = field(default_factory=Timing)
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: str | Unsupported = UNSUPPORTED
    provider_version: str | None = None
    adapter: AdapterIdentity | None = None
    adapter_base_confidence: IdentityConfidence | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the finish reason and the tool calls tell the same story.

        Raises:
            ValidationError: If ``finish_reason`` is :attr:`FinishReason.TOOL_CALLS` but no tool
                call is present — a caller told the model wants to call a tool, with no tool to
                call, has nothing it can do next.
        """
        if self.finish_reason is FinishReason.TOOL_CALLS and not self.tool_calls:
            raise ValidationError(
                "finish_reason is TOOL_CALLS but tool_calls is empty. A caller cannot act on a "
                "tool request that names no tool; if the provider signalled tool use without "
                "supplying a parseable call, that is a ProviderProtocolError in the adapter.",
                details={"field": "tool_calls", "finish_reason": self.finish_reason.value},
            )
