"""Domain module — how one scripted call's content is derived, deterministically.

Imports :mod:`baseaicore`, this package's own types and the standard library; performs no I/O and
reads no clock. Pure functions and two internal value objects: given a script, a seed and a
request, they produce the exact text, deltas, tool calls and counts one call will emit, and
nothing about *when* or *whether* it is delivered — that is
:class:`~modelrack.providers.fake.FakeProvider`'s.

Separated from the provider so neither module is the thousand-line "god module" the
[coding standards](../../../docs/standards/coding-standards.md) §13 name as an anti-pattern, and
because the seam is real: everything here is a pure function of its arguments and can be reasoned
about — and tested — without a provider, a call counter or a clock.

**Every value is derived through SHA-256 over a canonical string.** Not :func:`random.Random`:
its core generator is reproducible across releases but the derived helpers are not, and
"identical across processes and platforms" has to survive a Python upgrade to mean anything. Byte
order is stated explicitly at every conversion, and no result depends on dict ordering, locale,
float formatting or the process's hash seed.
"""

from __future__ import annotations

import hashlib
import json
import math

# Imported at runtime, not only for typing: `isinstance(value, Mapping)` is how a supplied JSON
# Schema is inspected, and a TYPE_CHECKING-only import would fail there.
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Final

from baseaicore import UNSUPPORTED, Measurement, ModelIdentity, Unsupported

from modelrack.providers._fake_script import (
    SIMULATED_TOKEN_CHARACTERS,
    FakeFailure,
    FakeGeneration,
    FakeModel,
)
from modelrack.streaming import ThinkingDelta, TokenDelta, ToolCallDelta
from modelrack.types import (
    FinishReason,
    GenerationRequest,
    GenerationUsage,
    ResponseFormatKind,
    Timing,
    ToolCall,
)

MILLISECONDS_PER_SECOND: Final[float] = 1000.0
"""Milliseconds in a second.

Lives here rather than beside either caller because both of them need it and neither owns it: the
provider converts a timeout in seconds into the millisecond budget a plan is measured in, and the
error translator converts that budget back for the ``details`` spec §13 states in seconds.
"""

_MAXIMUM_BODY_CHARACTERS: Final[int] = 512
_MAXIMUM_SCHEMA_DEPTH: Final[int] = 6
_SCHEMA_ARRAY_LENGTH: Final[int] = 2

# Long enough that a default response exercises chunking, streaming and a token count worth
# asserting on; short enough that a failing test prints something a human can read.
_DEFAULT_WORD_COUNT: Final[int] = 24

# Fixed, ordered, and load-bearing: every generated response is drawn from this tuple by index, so
# reordering or extending it changes the golden values in tests/unit/test_fake_provider.py. Treat
# it the way the suite treats a wire format, not the way it treats a word list. The non-ASCII
# entries are here on purpose — a caller that assembles deltas for display has to survive a split
# inside a multi-byte character or a combining sequence, and a vocabulary of plain ASCII would
# never produce one.
_VOCABULARY: Final[tuple[str, ...]] = tuple(
    (
        "about across after again against almost already although always among answer around "
        "because before behind below between beyond cache context decode during either enough "
        "every except further gradient however inference instead kernel latency layer memory "
        "model neither network neurone often output parameter perhaps precision prompt quantized "
        "rather residency runtime sample several since still tensor though throughput token "
        "toward unless until café naïve þing 日本語"
    ).split()
)


@dataclass(frozen=True, slots=True)
class _Step:
    """One streamed delta and the simulated time that passes before it arrives."""

    delay_ms: float
    event: TokenDelta | ThinkingDelta | ToolCallDelta


@dataclass(frozen=True, slots=True)
class _Plan:
    """Everything one call will produce, computed once and walked by ``generate`` and ``stream``.

    Sharing the plan is what keeps the two methods from drifting: a streamed call and a blocking
    call to the same script must produce the same text, the same usage and the same finish reason,
    differing only in ``client_ttft_ms`` — which a blocking call has no moment at which to observe.
    """

    identity: ModelIdentity
    model: FakeModel
    base_url: str
    model_count: int
    steps: tuple[_Step, ...]
    text: str
    thinking: str | Unsupported
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: GenerationUsage
    backend_timing: Timing
    prompt_tokens: int
    provider_version: str | None
    failure: FakeFailure | None
    failure_step: int | None
    raw: Mapping[str, Any]

    @property
    def total_delay_ms(self) -> float:
        """Return the simulated time the whole call takes."""
        return math.fsum(step.delay_ms for step in self.steps)

    @property
    def first_delay_ms(self) -> Measurement:
        """Return the simulated time before the first delta, or ``UNSUPPORTED`` if there is none.

        A stream that produced no delta has no first-token moment to report, and reporting ``0``
        would claim one arrived instantly.
        """
        return self.steps[0].delay_ms if self.steps else UNSUPPORTED

    def elapsed_ms_before(self, step_index: int) -> float:
        """Return the simulated time elapsed once ``step_index`` deltas have been delivered."""
        return math.fsum(step.delay_ms for step in self.steps[:step_index])


def _digest_bytes(*parts: object) -> bytes:
    """Return SHA-256 over the parts joined by a separator that cannot occur inside one."""
    material = "\x00".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).digest()


def _digest_int(*parts: object) -> int:
    """Return a stable non-negative integer derived from the parts, big-endian everywhere."""
    return int.from_bytes(_digest_bytes(*parts)[:8], "big")


def _simulated_token_count(text: str) -> int:
    """Return the fake's token count for a string: characters divided by the published width."""
    return math.ceil(len(text) / SIMULATED_TOKEN_CHARACTERS)


def _generated_text(seed_material: str, word_count: int) -> str:
    """Return deterministic pseudo-text of ``word_count`` words drawn from the fixed vocabulary."""
    words = [
        _VOCABULARY[_digest_int(seed_material, index) % len(_VOCABULARY)]
        for index in range(word_count)
    ]
    return " ".join(words)


def _split_into_chunks(text: str, size: int) -> tuple[str, ...]:
    """Return ``text`` split into fixed-width pieces, splitting on characters rather than words."""
    return tuple(text[start : start + size] for start in range(0, len(text), size))


def _truncate_body(body: str) -> str:
    """Return a body short enough to travel in an error's ``details`` (spec §13)."""
    if len(body) <= _MAXIMUM_BODY_CHARACTERS:
        return body
    return body[:_MAXIMUM_BODY_CHARACTERS] + "…"


def _render_prompt(request: GenerationRequest) -> str:
    """Return one canonical string standing for everything the provider would have been sent.

    Used for two things at once, deliberately: it seeds the generated answer, so two different
    prompts get two different answers, and it is what the simulated input token count is derived
    from. Tool definitions are part of it because offering a tool really does cost context, and a
    consumer measuring the price of a large tool set against a fake that ignored them would
    measure nothing.

    ``metadata`` is excluded. It is the caller's own correlation data and is never sent to a
    provider ([spec §7](../../../docs/packages/modelrack/spec.md)); letting it change the answer
    would make it observable, which is the same leak by a slower route.
    """
    parts: list[str] = []
    if request.prompt is not None:
        parts.append(request.prompt)
    for message in request.messages:
        parts.append(f"{message.role.value}: {message.content}")
        for call in message.tool_calls:
            parts.append(
                f"tool_call: {call.name} {json.dumps(dict(call.arguments), sort_keys=True)}"
            )
    for tool in request.tools:
        parameters = json.dumps(dict(tool.parameters), sort_keys=True)
        parts.append(f"tool: {tool.name} {tool.description} {parameters}")
    return "\n".join(parts)


def _schema_value(schema: Mapping[str, Any], seed_material: str, depth: int) -> Any:  # noqa: ANN401 — mirrors an arbitrary JSON Schema
    """Return a deterministic value matching a schema's *shape*.

    Honours ``enum``, ``type``, ``properties``, ``required`` and ``items``, which is what an
    application's stage schemas are built from. It does **not** honour ``minimum``, ``pattern``,
    ``minItems`` or the rest, and says so loudly: a fake that claimed full JSON Schema conformance
    would mean a consumer's own validator never ran against a violation, and validating structured
    output is the thing IdeaPress exists to do.

    Beyond :data:`_MAXIMUM_SCHEMA_DEPTH` levels it returns ``None`` rather than descending, so a
    self-referential schema produces a shallow document instead of a stack overflow.
    """
    if depth > _MAXIMUM_SCHEMA_DEPTH:
        return None
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[_digest_int(seed_material, "enum", depth) % len(enum)]
    declared = schema.get("type")
    kind = declared[0] if isinstance(declared, list) and declared else declared
    if kind == "object" or (kind is None and "properties" in schema):
        return _schema_object(schema, seed_material, depth)
    if kind == "array":
        items = schema.get("items")
        item_schema: Mapping[str, Any] = items if isinstance(items, Mapping) else {}
        return [
            _schema_value(item_schema, f"{seed_material}\x00item{index}", depth + 1)
            for index in range(_SCHEMA_ARRAY_LENGTH)
        ]
    if kind == "integer":
        return _digest_int(seed_material, "integer") % 1000
    if kind == "number":
        return (_digest_int(seed_material, "number") % 10000) / 100
    if kind == "boolean":
        return _digest_int(seed_material, "boolean") % 2 == 0
    if kind == "null":
        return None
    return _generated_text(f"{seed_material}\x00string", 3)


def _schema_object(schema: Mapping[str, Any], seed_material: str, depth: int) -> dict[str, Any]:
    """Return an object carrying a schema's required properties, or all of them if none are named.

    All of them when ``required`` is absent, because a schema that names no required property is
    usually one whose author expected every property — and an object that came back empty would
    pass a shape check while telling the caller nothing.
    """
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return {}
    required = schema.get("required")
    names = (
        [name for name in properties if name in set(required)]
        if isinstance(required, list) and required
        else list(properties)
    )
    return {
        name: _schema_value(
            properties[name] if isinstance(properties[name], Mapping) else {},
            f"{seed_material}\x00{name}",
            depth + 1,
        )
        for name in sorted(names)
    }


def _json_text(schema: Mapping[str, Any] | None, seed_material: str, answer: str) -> str:
    """Return a JSON document: schema-shaped when a schema is supplied, else the answer wrapped."""
    if schema is None:
        return json.dumps({"answer": answer}, sort_keys=True, ensure_ascii=False)
    return json.dumps(
        _schema_value(schema, seed_material, depth=0), sort_keys=True, ensure_ascii=False
    )


def _parsed_arguments(raw: str) -> Mapping[str, Any]:
    """Return the parsed argument object, or an empty mapping when the text will not parse.

    Empty rather than an exception, because a model emitting invalid JSON for a tool call is an
    ordinary event that a caller has to see and score, not a provider fault. The unparsed text is
    kept beside it on :class:`~modelrack.types.ToolCall`, so "the model called a tool with no
    arguments" stays distinguishable from "the arguments could not be read".
    """
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _truncate_chunks(chunks: tuple[str, ...], character_budget: int) -> tuple[str, ...]:
    """Return the leading chunks that fit the budget, cutting the last one mid-chunk if needed.

    Trims the existing boundaries rather than re-splitting the truncated text: a script that
    placed a boundary inside a grapheme cluster placed it there on purpose, and re-splitting would
    quietly move it somewhere safe.
    """
    starts = accumulate((len(chunk) for chunk in chunks), initial=0)
    return tuple(
        chunk[: character_budget - start]
        for chunk, start in zip(chunks, starts, strict=False)
        if start < character_budget
    )


def _planned_text(
    request: GenerationRequest, generation: FakeGeneration, seed_material: str
) -> tuple[str, tuple[str, ...], bool]:
    """Return the answer, its stream deltas, and whether the output limit truncated it.

    Explicit ``chunks`` win, then explicit ``text``, then generation. The precedence matters for
    the case that matters most about structured output: a script that names its own text produces
    that text even when the request asked for JSON, which is how a consumer tests the model that
    ignored the format — the failure its own validator exists to catch.

    ``max_output_tokens`` truncates whatever was produced, scripted text included, because that is
    what a runtime does. Truncating schema-shaped output leaves invalid JSON behind, which is also
    what a runtime does, and a caller that never met it would parse the happy path forever.
    """
    if generation.chunks is not None:
        chunks = generation.chunks
        text = "".join(chunks)
    elif generation.text is not None:
        text = generation.text
        chunks = _split_into_chunks(text, generation.chunk_size)
    else:
        words = generation.word_count if generation.word_count is not None else _DEFAULT_WORD_COUNT
        body = _generated_text(seed_material, words)
        response_format = request.response_format
        if response_format is None or response_format.kind is ResponseFormatKind.TEXT:
            text = body
        elif response_format.kind is ResponseFormatKind.JSON:
            text = _json_text(None, seed_material, body)
        else:
            text = _json_text(response_format.schema, seed_material, body)
        chunks = _split_into_chunks(text, generation.chunk_size)
    limit = request.sampling.max_output_tokens
    if limit is None or _simulated_token_count(text) <= limit:
        return text, chunks, False
    budget = limit * SIMULATED_TOKEN_CHARACTERS
    return text[:budget], _truncate_chunks(chunks, budget), True


def _planned_tool_calls(
    generation: FakeGeneration, generation_index: int
) -> tuple[tuple[ToolCall, ...], tuple[str, ...]]:
    """Return the tool calls to request and the argument text each one travelled as.

    The wire text is kept beside the parsed arguments on every call, not only the malformed ones.
    A caller that wants to know what the model actually emitted should not have to guess whether
    the adapter felt the parse went well enough to discard it.
    """
    calls: list[ToolCall] = []
    argument_texts: list[str] = []
    for index, specification in enumerate(generation.tool_calls):
        if specification.raw_arguments is not None:
            raw_text = specification.raw_arguments
            arguments = (
                dict(specification.arguments)
                if specification.arguments
                else _parsed_arguments(raw_text)
            )
        else:
            raw_text = json.dumps(dict(specification.arguments), sort_keys=True, ensure_ascii=False)
            arguments = dict(specification.arguments)
        calls.append(
            ToolCall(
                id=specification.id or f"fake-tool-call-{generation_index}-{index}",
                name=specification.name,
                arguments=arguments,
                raw_arguments=raw_text,
            )
        )
        argument_texts.append(raw_text)
    return tuple(calls), tuple(argument_texts)


def _planned_steps(
    generation: FakeGeneration,
    chunks: tuple[str, ...],
    thinking: str | Unsupported,
    tool_calls: tuple[ToolCall, ...],
    argument_texts: tuple[str, ...],
) -> tuple[_Step, ...]:
    """Return every delta the stream will emit, in arrival order, each with the delay before it.

    Reasoning first, then the answer, then tool calls — the order every supported runtime uses.
    A tool call's identity arrives in its own delta ahead of its argument fragments, because that
    is how providers send it and a caller assembling calls has to cope with knowing the name
    before it knows the arguments.

    ``index`` counts across all three delta types, so a caller can order a stream it is buffering
    without having to know which kinds it saw.
    """
    events: list[TokenDelta | ThinkingDelta | ToolCallDelta] = []
    if isinstance(thinking, str) and thinking:
        events.extend(
            ThinkingDelta(text=piece, index=len(events))
            for piece in _split_into_chunks(thinking, generation.chunk_size)
        )
    for piece in chunks:
        events.append(TokenDelta(text=piece, index=len(events)))
    for call_index, (call, argument_text) in enumerate(
        zip(tool_calls, argument_texts, strict=True)
    ):
        events.append(
            ToolCallDelta(call_index=call_index, id=call.id, name=call.name, index=len(events))
        )
        for piece in _split_into_chunks(argument_text, generation.chunk_size):
            events.append(
                ToolCallDelta(call_index=call_index, arguments_fragment=piece, index=len(events))
            )
    return tuple(
        _Step(
            delay_ms=generation.first_chunk_delay_ms
            if position == 0
            else generation.chunk_delay_ms,
            event=event,
        )
        for position, event in enumerate(events)
    )
