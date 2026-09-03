"""Tests for :mod:`modelrack.providers.fake` — the deterministic, scriptable provider.

Three of these classes carry a Phase 2 acceptance criterion rather than an ordinary invariant.
:class:`TestDeterminism` proves "same seed ⇒ identical text, chunking and token counts, twice and
across processes"; :class:`TestScriptedFailureModes` proves "every scripted failure mode is
reachable and produces the documented error", which is also
[spec §20](../../docs/packages/modelrack/spec.md) criterion 6 for every row of §13 a provider can
be scripted into; and :class:`TestStreamContract` proves the terminal-event and cancellation
rules.

The golden values in :class:`TestDeterminism` are contracts, not conveniences. A golden test that
fails is indistinguishable at the terminal from a golden test that needs updating, and the
discipline of never "updating a golden to make CI green" is what makes "identical across processes
and platforms" mean anything at all — the vocabulary and the derivation are as much a published
shape as a wire format (audit §11.3).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from baseaicore import (
    UNSUPPORTED,
    IdentityConfidence,
    ModelCapabilityFlag,
    ModelIdentity,
    ProviderKind,
    RuntimeProfile,
    ValidationError,
    is_supported,
)

from modelrack import (
    CancellationToken,
    CapabilityUnsupported,
    ContextLimitExceeded,
    FinishReason,
    GenerationCancelled,
    GenerationRequest,
    Message,
    ModelNotFound,
    Provider,
    ProviderError,
    ProviderProtocolError,
    ProviderRejected,
    ProviderStatus,
    ProviderTimeout,
    ProviderUnavailable,
    ResponseFormat,
    ResponseFormatKind,
    Role,
    SamplingParameters,
    StreamCompleted,
    StreamFailed,
    ThinkingDelta,
    Timing,
    TokenDelta,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
)
from modelrack.testing import (
    DEFAULT_MODEL,
    FULL_CAPABILITIES,
    MINIMAL_CAPABILITIES,
    SIMULATED_TOKEN_CHARACTERS,
    FakeFailure,
    FakeFailureMode,
    FakeGeneration,
    FakeModel,
    FakeProvider,
    FakeScript,
    FakeToolCall,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from modelrack import StreamEvent

_KNOWN_MODEL = "fake-model:8b-q8_0"

# Pinned outputs for seed 7 and the request built by `_request` below. Changing the vocabulary,
# the derivation or the seed material changes these; that is the point. Never update one to make
# CI green — a changed golden means every downstream expectation built on this fake has silently
# moved too.
_GOLDEN_TEXT = (
    "almost prompt except neither neurone token neither although often before 日本語 neither "
    "below though almost 日本語 already café gradient cache sample gradient toward though"
)
_GOLDEN_SCHEMA_JSON = (
    '{"answer": "almost however between", "confidence": 19.04, '
    '"sources": ["since every several", "below decode always"]}'
)
_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence", "sources"],
}
_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _request(identity: ModelIdentity, **overrides: Any) -> GenerationRequest:
    """Build the standard chat request the goldens above were captured from."""
    fields: dict[str, Any] = {
        "identity": identity,
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def _provider(script: FakeScript | None = None, *, seed: int = 7, **kwargs: Any) -> FakeProvider:
    """Build a fake provider, defaulting to the seed the goldens were captured with."""
    return FakeProvider(script, seed=seed, **kwargs)


def _identity(provider: FakeProvider = None) -> ModelIdentity:  # type: ignore[assignment]
    """Return the identity of the default catalogue's only model."""
    return (provider or _provider()).resolve(_KNOWN_MODEL)


def _text_deltas(events: Sequence[StreamEvent]) -> list[TokenDelta]:
    """Return only the answer-text deltas from a drained stream."""
    return [event for event in events if isinstance(event, TokenDelta)]


class TestDeterminism:
    """Acceptance criterion: identical output for a seed, twice and across processes."""

    def test_the_same_seed_and_call_index_produce_the_same_text(self) -> None:
        first = _provider().generate(_request(_identity()))
        second = _provider().generate(_request(_identity()))

        assert first.text == second.text
        assert first.usage == second.usage

    def test_generated_text_is_the_recorded_golden(self) -> None:
        """A contract, not a convenience. See this module's docstring before changing it."""
        assert _provider().generate(_request(_identity())).text == _GOLDEN_TEXT

    def test_schema_shaped_output_is_the_recorded_golden(self) -> None:
        result = _provider().generate(
            _request(
                _identity(),
                response_format=ResponseFormat(
                    kind=ResponseFormatKind.JSON_SCHEMA, schema=_ANSWER_SCHEMA
                ),
            )
        )

        assert result.text == _GOLDEN_SCHEMA_JSON

    def test_a_different_seed_answers_differently(self) -> None:
        seven = _provider(seed=7).generate(_request(_identity()))
        eight = _provider(seed=8).generate(_request(_identity()))

        assert seven.text != eight.text

    def test_a_different_prompt_answers_differently(self) -> None:
        """A consumer's scorer tested against one string has been tested against one string."""
        identity = _identity()
        first = _provider().generate(_request(identity))
        second = _provider().generate(
            _request(identity, messages=(Message(role=Role.USER, content="Something else."),))
        )

        assert first.text != second.text

    def test_a_different_sampling_seed_answers_differently(self) -> None:
        identity = _identity()
        pinned = _provider().generate(_request(identity, sampling=SamplingParameters(seed=1)))
        other = _provider().generate(_request(identity, sampling=SamplingParameters(seed=2)))

        assert pinned.text != other.text

    @pytest.mark.parametrize("hash_seed", ["0", "1", "524287"])
    def test_output_survives_another_process_and_another_hash_seed(self, hash_seed: str) -> None:
        """The failure mode this guards is nondeterminism creeping in via dict or set ordering.

        ``PYTHONHASHSEED`` randomizes string hashing, and therefore set iteration order, per
        process. Running the same generation under three of them and comparing against the value
        this process computed is what turns "no dependence on dict ordering" from an intention
        into a test.
        """
        program = (
            "import sys;"
            "from modelrack.testing import FakeProvider;"
            "from modelrack import GenerationRequest, Message, Role;"
            "p = FakeProvider(seed=7);"
            "i = p.resolve('fake-model:8b-q8_0');"
            "r = p.generate(GenerationRequest(identity=i, messages="
            "(Message(role=Role.USER, content='Explain KV caching.'),)));"
            "sys.stdout.write(r.text)"
        )
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell, no external input
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            # Inherited rather than replaced: the child needs the same interpreter
            # environment this process was started with (a virtualenv, a CI cache path) to
            # import the package at all. Only the hash seed is forced.
            env={**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONIOENCODING": "utf-8"},
        )

        assert completed.stdout == _GOLDEN_TEXT

    def test_chunking_and_counts_are_stable_across_two_streams(self) -> None:
        first = list(_provider().stream(_request(_identity())))
        second = list(_provider().stream(_request(_identity())))

        assert [delta.text for delta in _text_deltas(first)] == [
            delta.text for delta in _text_deltas(second)
        ]
        assert isinstance(first[-1], StreamCompleted)
        assert isinstance(second[-1], StreamCompleted)
        assert first[-1].result.usage == second[-1].result.usage


class TestGeneratedContent:
    """What the fake produces when nothing pins the text down, and when something does."""

    def test_word_count_controls_how_much_is_generated(self) -> None:
        script = FakeScript(generations=(FakeGeneration(word_count=5),))

        result = _provider(script).generate(_request(_identity()))

        assert len(result.text.split()) == 5

    def test_a_zero_word_response_is_legal_and_empty(self) -> None:
        """A tool-call-only turn really does generate no text."""
        script = FakeScript(generations=(FakeGeneration(word_count=0),))

        assert _provider(script).generate(_request(_identity())).text == ""

    def test_explicit_text_wins(self) -> None:
        script = FakeScript(generations=(FakeGeneration(text="exactly this"),))

        assert _provider(script).generate(_request(_identity())).text == "exactly this"

    def test_explicit_text_overrides_a_requested_format(self) -> None:
        """The case that matters most about structured output: the model ignored it."""
        script = FakeScript(generations=(FakeGeneration(text="I am afraid I cannot do that."),))

        result = _provider(script).generate(
            _request(_identity(), response_format=ResponseFormat(kind=ResponseFormatKind.JSON))
        )

        assert result.text == "I am afraid I cannot do that."
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.text)

    def test_json_mode_produces_parseable_json(self) -> None:
        result = _provider().generate(
            _request(_identity(), response_format=ResponseFormat(kind=ResponseFormatKind.JSON))
        )

        assert isinstance(json.loads(result.text), dict)

    def test_a_schema_shapes_the_document(self) -> None:
        result = _provider().generate(
            _request(
                _identity(),
                response_format=ResponseFormat(
                    kind=ResponseFormatKind.JSON_SCHEMA, schema=_ANSWER_SCHEMA
                ),
            )
        )
        document = json.loads(result.text)

        assert set(document) == {"answer", "confidence", "sources"}
        assert isinstance(document["answer"], str)
        assert isinstance(document["confidence"], float)
        assert document["sources"] and all(isinstance(item, str) for item in document["sources"])

    def test_a_schema_enum_is_honoured(self) -> None:
        schema = {"type": "object", "properties": {"verdict": {"enum": ["pass", "fail"]}}}

        result = _provider().generate(
            _request(
                _identity(),
                response_format=ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema=schema),
            )
        )

        assert json.loads(result.text)["verdict"] in {"pass", "fail"}

    def test_a_self_referential_schema_terminates(self) -> None:
        """A recursive schema produces a shallow document rather than a stack overflow."""
        node: dict[str, Any] = {"type": "object", "properties": {}}
        node["properties"]["child"] = node

        result = _provider().generate(
            _request(
                _identity(),
                response_format=ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema=node),
            )
        )

        assert json.loads(result.text) is not None

    def test_the_output_limit_truncates_and_says_so(self) -> None:
        result = _provider().generate(
            _request(_identity(), sampling=SamplingParameters(max_output_tokens=4))
        )

        assert result.finish_reason is FinishReason.LENGTH
        assert len(result.text) == 4 * SIMULATED_TOKEN_CHARACTERS

    def test_an_output_limit_above_the_answer_does_not_truncate(self) -> None:
        result = _provider().generate(
            _request(_identity(), sampling=SamplingParameters(max_output_tokens=10_000))
        )

        assert result.finish_reason is FinishReason.STOP
        assert result.text == _GOLDEN_TEXT

    def test_truncation_keeps_hand_placed_chunk_boundaries(self) -> None:
        script = FakeScript(
            capabilities=dataclasses.replace(FULL_CAPABILITIES, token_level_chunks=False),
            generations=(FakeGeneration(chunks=("abc", "defgh", "ijklmnop")),),
        )

        events = list(
            _provider(script).stream(
                _request(_identity(), sampling=SamplingParameters(max_output_tokens=2))
            )
        )

        assert [delta.text for delta in _text_deltas(events)] == ["abc", "defgh"]

    def test_a_scripted_finish_reason_wins(self) -> None:
        script = FakeScript(
            generations=(FakeGeneration(finish_reason=FinishReason.CONTENT_FILTER),)
        )

        assert (
            _provider(script).generate(_request(_identity())).finish_reason
            is FinishReason.CONTENT_FILTER
        )


class TestChunking:
    """How the answer is cut into deltas, and the honesty ``token_level_chunks`` requires."""

    def test_one_delta_is_one_simulated_token_when_that_is_declared(self) -> None:
        """The claim spec §11.4 gates: a caller may divide by the delta count."""
        events = list(_provider().stream(_request(_identity())))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert len(_text_deltas(events)) == len(
            terminal.result.text
        ) // SIMULATED_TOKEN_CHARACTERS + (
            1 if len(terminal.result.text) % SIMULATED_TOKEN_CHARACTERS else 0
        )

    def test_hand_placed_chunks_are_emitted_exactly(self) -> None:
        script = FakeScript(
            capabilities=dataclasses.replace(FULL_CAPABILITIES, token_level_chunks=False),
            generations=(FakeGeneration(chunks=("Hello, ", "wor", "ld!")),),
        )

        events = list(_provider(script).stream(_request(_identity())))

        assert [delta.text for delta in _text_deltas(events)] == ["Hello, ", "wor", "ld!"]

    def test_a_grapheme_cluster_can_be_split_across_two_deltas(self) -> None:
        """A caller assembling deltas for display has to survive this; so it must be scriptable."""
        script = FakeScript(
            capabilities=dataclasses.replace(FULL_CAPABILITIES, token_level_chunks=False),
            generations=(FakeGeneration(chunks=("e", "́clair")),),
        )

        events = list(_provider(script).stream(_request(_identity())))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "éclair"
        assert [delta.text for delta in _text_deltas(events)] == ["e", "́clair"]

    def test_the_generated_vocabulary_reaches_beyond_ascii(self) -> None:
        """A vocabulary of plain ASCII would never produce the split above by accident."""
        text = _provider().generate(_request(_identity())).text

        assert len(text.encode("utf-8")) > len(text)


class TestUsageAccounting:
    """What the fake claims the call consumed, and what it refuses to claim."""

    def test_counts_are_derived_from_what_was_produced(self) -> None:
        result = _provider().generate(_request(_identity()))

        assert result.usage.tokens.output_tokens == 42
        assert result.usage.output_chars == len(_GOLDEN_TEXT)
        assert result.usage.output_words == len(_GOLDEN_TEXT.split())
        assert result.usage.output_bytes == len(_GOLDEN_TEXT.encode("utf-8"))

    def test_scripted_counts_win(self) -> None:
        script = FakeScript(generations=(FakeGeneration(input_tokens=11, output_tokens=13),))

        usage = _provider(script).generate(_request(_identity())).usage

        assert usage.tokens.input_tokens == 11
        assert usage.tokens.output_tokens == 13

    def test_an_unscripted_cache_class_is_zero_rather_than_unsupported(self) -> None:
        """ADR-0070 decision 5: the fake plays a protocol that bills no cache tier.

        `0` here is the same statement the two real adapters make about their own wire formats —
        nothing could have been billed under these headings — and it is what lets a consumer's
        tests exercise a *totalled* cost against the fake at all, which was impossible while every
        fake response reported two unavailable classes.
        """
        tokens = _provider().generate(_request(_identity())).usage.tokens

        assert tokens.cache_read_tokens == 0
        assert tokens.cache_write_tokens == 0
        assert is_supported(tokens.total_tokens)

    def test_unsupported_cache_classes_stay_scriptable(self) -> None:
        """The escape hatch decision 5 requires the fake to keep once its default became `0`.

        LoadLedger and PromptCadence both need a response whose cache classes were never reported
        — the shape a real adapter produces from a response with no usage object — and a fake that
        could only produce zeros would quietly stop them testing their own ``UNSUPPORTED``
        branches.
        """
        script = FakeScript(
            generations=(
                FakeGeneration(cache_read_tokens=UNSUPPORTED, cache_write_tokens=UNSUPPORTED),
            )
        )

        tokens = _provider(script).generate(_request(_identity())).usage.tokens

        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)
        assert not is_supported(tokens.total_tokens)

    def test_cached_input_is_not_also_billed_as_input(self) -> None:
        """ADR-0030: the four billing classes are disjoint, and reconciling them is adapter work."""
        plain = _provider().generate(_request(_identity())).usage.tokens
        script = FakeScript(generations=(FakeGeneration(cache_read_tokens=3),))

        cached = _provider(script).generate(_request(_identity())).usage.tokens

        assert cached.cache_read_tokens == 3
        assert cached.input_tokens == plain.input_tokens - 3

    def test_cached_input_never_drives_a_count_below_zero(self) -> None:
        script = FakeScript(generations=(FakeGeneration(cache_read_tokens=10_000),))

        usage = _provider(script).generate(_request(_identity())).usage

        assert usage.tokens.input_tokens == 0

    def test_an_undeclared_counter_reports_nothing_rather_than_zero(self) -> None:
        script = FakeScript(capabilities=MINIMAL_CAPABILITIES)

        usage = _provider(script).generate(_request(_identity())).usage

        assert not is_supported(usage.tokens.input_tokens)
        assert not is_supported(usage.tokens.output_tokens)
        assert not is_supported(usage.thinking_tokens)
        assert not is_supported(usage.tool_tokens)
        # ADR-0070 does not reach here: a provider that declares it counts nothing has not told
        # us that nothing was billed, only that it is not counting. This is the fake's analogue
        # of a response with no usage object, and all four classes stay unavailable.
        assert not is_supported(usage.tokens.cache_read_tokens)
        assert not is_supported(usage.tokens.cache_write_tokens)

    def test_observations_survive_an_undeclared_counter(self) -> None:
        """Characters are not the provider's count — this process is holding the string."""
        script = FakeScript(capabilities=MINIMAL_CAPABILITIES)

        usage = _provider(script).generate(_request(_identity())).usage

        assert is_supported(usage.output_chars)
        assert usage.output_chars == len(_GOLDEN_TEXT)

    def test_reasoning_tokens_are_absent_when_no_reasoning_was_reported(self) -> None:
        """Reporting a count of zero for something not reported at all is a contradiction."""
        result = _provider().generate(_request(_identity()))

        assert not is_supported(result.thinking)
        assert not is_supported(result.usage.thinking_tokens)

    def test_reasoning_tokens_are_a_breakdown_of_output_tokens(self) -> None:
        script = FakeScript(generations=(FakeGeneration(word_count=0, thinking="let me think"),))

        usage = _provider(script).generate(_request(_identity())).usage

        assert usage.thinking_tokens == 3
        assert usage.tokens.output_tokens == 3


class TestTimings:
    """Backend and client measurements, kept apart and neither invented."""

    def test_scripted_delays_become_the_client_measurements(self) -> None:
        script = FakeScript(
            generations=(FakeGeneration(word_count=2, first_chunk_delay_ms=90, chunk_delay_ms=10),)
        )

        events = list(_provider(script).stream(_request(_identity())))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.timing.client_ttft_ms == 90
        assert terminal.result.timing.client_wall_ms == 90 + 10 * (len(_text_deltas(events)) - 1)

    def test_a_blocking_call_reports_no_first_token_moment(self) -> None:
        script = FakeScript(generations=(FakeGeneration(first_chunk_delay_ms=90),))

        timing = _provider(script).generate(_request(_identity())).timing

        assert not is_supported(timing.client_ttft_ms)
        assert is_supported(timing.client_wall_ms)

    def test_backend_timings_are_absent_unless_scripted(self) -> None:
        """The fake ran no model and refuses to account for work it did not do."""
        timing = _provider().generate(_request(_identity())).timing

        assert not is_supported(timing.backend_decode_ms)
        assert not is_supported(timing.backend_total_ms)

    def test_a_scripted_backend_total_need_not_equal_its_phases(self) -> None:
        """A real total covers time its phases do not; recomputing it would erase that."""
        backend = Timing(
            backend_load_ms=100,
            backend_prompt_eval_ms=20,
            backend_decode_ms=300,
            backend_total_ms=500,
        )
        script = FakeScript(generations=(FakeGeneration(backend_timing=backend),))

        timing = _provider(script).generate(_request(_identity())).timing

        assert timing.backend_total_ms == 500
        assert timing.backend_load_ms == 100

    def test_delays_cost_no_real_time_unless_a_sleep_is_injected(self) -> None:
        slept: list[float] = []
        script = FakeScript(generations=(FakeGeneration(word_count=1, first_chunk_delay_ms=250),))

        _provider(script).generate(_request(_identity()))
        _provider(script, sleep=slept.append).generate(_request(_identity()))

        assert slept and sum(slept) == pytest.approx(0.25, abs=0.001)


class TestThinking:
    """Reasoning content, and the capability that has to be declared before it can exist."""

    def test_reasoning_is_streamed_before_the_answer(self) -> None:
        script = FakeScript(generations=(FakeGeneration(word_count=2, thinking="first I check"),))

        events = list(_provider(script).stream(_request(_identity())))

        kinds = [type(event).__name__ for event in events]
        assert kinds.index("ThinkingDelta") < kinds.index("TokenDelta")
        assert (
            "".join(event.text for event in events if isinstance(event, ThinkingDelta))
            == "first I check"
        )

    def test_reasoning_is_kept_out_of_the_answer(self) -> None:
        script = FakeScript(generations=(FakeGeneration(text="answer", thinking="working"),))

        result = _provider(script).generate(_request(_identity()))

        assert result.text == "answer"
        assert result.thinking == "working"

    def test_reasoning_cannot_be_scripted_onto_a_provider_that_declares_none(self) -> None:
        with pytest.raises(ValidationError) as raised:
            FakeScript(
                capabilities=dataclasses.replace(FULL_CAPABILITIES, thinking_control=False),
                generations=(FakeGeneration(thinking="working"),),
            )

        assert raised.value.details["field"] == "thinking"


class TestToolCalls:
    """Requested calls, including the ways a real model gets them wrong."""

    def test_a_single_call_is_parsed_and_ends_the_turn(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    word_count=0,
                    tool_calls=(FakeToolCall(name="get_weather", arguments={"city": "Berlin"}),),
                ),
            )
        )

        result = _provider(script).generate(_request(_identity(), tools=(_WEATHER_TOOL,)))

        assert result.finish_reason is FinishReason.TOOL_CALLS
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"city": "Berlin"}

    def test_several_calls_get_distinct_synthesized_ids(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    word_count=0,
                    tool_calls=(FakeToolCall(name="one"), FakeToolCall(name="two")),
                ),
            )
        )

        result = _provider(script).generate(_request(_identity(), tools=(_WEATHER_TOOL,)))

        ids = [call.id for call in result.tool_calls]
        assert len(set(ids)) == 2
        assert all(identifier for identifier in ids)

    def test_a_scripted_id_is_used_unchanged(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(word_count=0, tool_calls=(FakeToolCall(name="a", id="x1"),)),
            )
        )

        result = _provider(script).generate(_request(_identity(), tools=(_WEATHER_TOOL,)))

        assert result.tool_calls[0].id == "x1"

    def test_malformed_arguments_are_preserved_rather_than_dropped(self) -> None:
        """FreeWeight scores this as a failure it must be able to see."""
        script = FakeScript(
            generations=(
                FakeGeneration(
                    word_count=0,
                    tool_calls=(
                        FakeToolCall(name="get_weather", raw_arguments='{"city": "Berlin"'),
                    ),
                ),
            )
        )

        call = (
            _provider(script).generate(_request(_identity(), tools=(_WEATHER_TOOL,))).tool_calls[0]
        )

        assert call.arguments == {}
        assert call.raw_arguments == '{"city": "Berlin"'

    def test_a_json_array_of_arguments_is_not_mistaken_for_a_mapping(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    word_count=0, tool_calls=(FakeToolCall(name="a", raw_arguments="[1]"),)
                ),
            )
        )

        call = (
            _provider(script).generate(_request(_identity(), tools=(_WEATHER_TOOL,))).tool_calls[0]
        )

        assert call.arguments == {}
        assert call.raw_arguments == "[1]"

    def test_calls_stream_as_an_identity_then_argument_fragments(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    word_count=0,
                    tool_calls=(FakeToolCall(name="get_weather", arguments={"city": "Berlin"}),),
                ),
            )
        )

        events = list(_provider(script).stream(_request(_identity(), tools=(_WEATHER_TOOL,))))
        deltas = [event for event in events if isinstance(event, ToolCallDelta)]

        assert deltas[0].name == "get_weather"
        assert deltas[0].arguments_fragment is None
        assert json.loads("".join(delta.arguments_fragment or "" for delta in deltas)) == {
            "city": "Berlin"
        }

    def test_offering_a_tool_to_a_provider_that_declares_none_is_refused(self) -> None:
        script = FakeScript(capabilities=MINIMAL_CAPABILITIES)

        with pytest.raises(CapabilityUnsupported) as raised:
            _provider(script).generate(_request(_identity(), tools=(_WEATHER_TOOL,)))

        assert raised.value.details["capability"] == "tool_calling"

    def test_a_model_may_call_a_tool_that_was_never_offered(self) -> None:
        """Models do this, and a caller that assumed otherwise crashes on the first one."""
        script = FakeScript(
            generations=(FakeGeneration(word_count=0, tool_calls=(FakeToolCall(name="ghost"),)),)
        )

        result = _provider(script).generate(_request(_identity(), tools=(_WEATHER_TOOL,)))

        assert result.tool_calls[0].name == "ghost"


class TestStreamContract:
    """The terminal-event rule, and what cancellation does to it."""

    def test_a_stream_ends_with_exactly_one_terminal_event(self) -> None:
        events = list(_provider().stream(_request(_identity())))

        terminal = [event for event in events if isinstance(event, StreamCompleted | StreamFailed)]
        assert len(terminal) == 1
        assert events[-1] is terminal[0]

    def test_delta_indices_run_across_every_delta_type(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(word_count=2, thinking="hm", tool_calls=(FakeToolCall(name="a"),)),
            )
        )

        events = list(_provider(script).stream(_request(_identity(), tools=(_WEATHER_TOOL,))))

        indices = [
            event.index
            for event in events
            if isinstance(event, TokenDelta | ThinkingDelta | ToolCallDelta)
        ]
        assert indices == list(range(len(indices)))

    def test_cancelling_before_the_first_delta_still_terminates_the_stream(self) -> None:
        token = CancellationToken()
        token.cancel()

        events = list(_provider().stream(_request(_identity(), cancel=token)))

        assert len(events) == 1
        assert isinstance(events[0], StreamFailed)
        assert isinstance(events[0].error, GenerationCancelled)
        assert events[0].partial_text == ""

    def test_cancelling_mid_stream_stops_within_one_delta_and_keeps_the_partial(self) -> None:
        token = CancellationToken()
        events: list[StreamEvent] = []
        for event in _provider().stream(_request(_identity(), cancel=token)):
            events.append(event)
            if len(_text_deltas(events)) == 3 and not token.is_cancelled:
                token.cancel()

        assert len(_text_deltas(events)) == 3
        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == _GOLDEN_TEXT[: 3 * SIMULATED_TOKEN_CHARACTERS]
        assert terminal.error.details["partial_text"] == terminal.partial_text

    def test_cancellation_is_delivered_not_raised(self) -> None:
        """A raise mid-drain is how a *truncated* stream looks; the two must stay apart."""
        token = CancellationToken()
        token.cancel()

        events = list(_provider().stream(_request(_identity(), cancel=token)))

        assert isinstance(events[-1], StreamFailed)

    def test_a_cancellation_token_does_nothing_to_a_blocking_call(self) -> None:
        token = CancellationToken()
        token.cancel()

        result = _provider().generate(_request(_identity(), cancel=token))

        assert result.text == _GOLDEN_TEXT
        assert result.finish_reason is FinishReason.STOP

    def test_a_real_sleep_is_spent_before_each_delta(self) -> None:
        """Proves the cancellation check sits after the wait, where a token set during it lands."""
        slept: list[float] = []
        script = FakeScript(
            generations=(FakeGeneration(word_count=2, first_chunk_delay_ms=5, chunk_delay_ms=1),)
        )

        events = list(_provider(script, sleep=slept.append).stream(_request(_identity())))

        assert len(slept) == len(_text_deltas(events))
        assert slept[0] == pytest.approx(0.005)

    def test_streaming_is_refused_when_it_is_not_declared(self) -> None:
        script = FakeScript(capabilities=MINIMAL_CAPABILITIES)

        with pytest.raises(CapabilityUnsupported) as raised:
            list(_provider(script).stream(_request(_identity())))

        assert raised.value.details["capability"] == "streaming"


class TestScriptedFailureModes:
    """Acceptance criterion: every scripted failure is reachable and documented.

    Together with :class:`TestCapabilityGating` and :class:`TestStreamContract`, this covers every
    row of [spec §13](../../docs/packages/modelrack/spec.md) — the eight a provider can be
    scripted into here, plus ``CapabilityUnsupported`` and ``GenerationCancelled``, which come
    from the fake's own gating and the caller's own token rather than from a script.
    """

    _EXPECTED: dict[FakeFailureMode, tuple[type[ProviderError], str, tuple[str, ...]]] = {
        FakeFailureMode.UNAVAILABLE: (
            ProviderUnavailable,
            "PROVIDER_UNAVAILABLE",
            ("base_url", "reason"),
        ),
        FakeFailureMode.TIMEOUT: (
            ProviderTimeout,
            "PROVIDER_TIMEOUT",
            ("elapsed_seconds", "limit_seconds"),
        ),
        FakeFailureMode.UNPARSEABLE_BODY: (
            ProviderProtocolError,
            "PROVIDER_PROTOCOL_ERROR",
            ("body",),
        ),
        FakeFailureMode.UNEXPECTED_SHAPE: (
            ProviderProtocolError,
            "PROVIDER_PROTOCOL_ERROR",
            ("body",),
        ),
        FakeFailureMode.TRUNCATED_STREAM: (
            ProviderProtocolError,
            "PROVIDER_PROTOCOL_ERROR",
            ("chunks_received",),
        ),
        FakeFailureMode.MODEL_NOT_FOUND: (
            ModelNotFound,
            "MODEL_NOT_FOUND",
            ("reference", "known_model_count"),
        ),
        FakeFailureMode.CONTEXT_LIMIT_EXCEEDED: (
            ContextLimitExceeded,
            "CONTEXT_LIMIT_EXCEEDED",
            ("requested_tokens", "maximum_tokens"),
        ),
        FakeFailureMode.REJECTED: (
            ProviderRejected,
            "PROVIDER_REJECTED",
            ("status_code", "provider_message"),
        ),
    }

    def test_every_mode_is_covered_by_this_table(self) -> None:
        """A new failure mode arrives here or it arrives untested."""
        assert set(self._EXPECTED) == set(FakeFailureMode)

    @pytest.mark.parametrize("mode", list(FakeFailureMode), ids=lambda mode: mode.value)
    def test_a_blocking_call_raises_the_documented_error(self, mode: FakeFailureMode) -> None:
        expected_type, expected_code, expected_details = self._EXPECTED[mode]
        script = FakeScript(generations=(FakeGeneration(failure=FakeFailure(mode=mode)),))

        with pytest.raises(expected_type) as raised:
            _provider(script).generate(_request(_identity()))

        assert raised.value.code == expected_code
        assert set(expected_details) <= set(raised.value.details)

    @pytest.mark.parametrize("mode", list(FakeFailureMode), ids=lambda mode: mode.value)
    def test_a_stream_delivers_the_documented_error_as_its_terminal_event(
        self, mode: FakeFailureMode
    ) -> None:
        expected_type, expected_code, expected_details = self._EXPECTED[mode]
        script = FakeScript(
            generations=(FakeGeneration(failure=FakeFailure(mode=mode, after_chunks=2)),)
        )

        events = list(_provider(script).stream(_request(_identity())))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, expected_type)
        assert terminal.error.code == expected_code
        assert set(expected_details) <= set(terminal.error.details)
        assert len(_text_deltas(events)) == 2
        assert terminal.partial_text == _GOLDEN_TEXT[: 2 * SIMULATED_TOKEN_CHARACTERS]

    def test_a_failure_before_the_stream_starts_raises_instead(self) -> None:
        """There is no stream to terminate, which is exactly what `modelrack.streaming` says."""
        script = FakeScript(
            generations=(FakeGeneration(failure=FakeFailure(mode=FakeFailureMode.UNAVAILABLE)),)
        )

        with pytest.raises(ProviderUnavailable):
            _provider(script).stream(_request(_identity()))

    def test_a_failure_at_delta_zero_is_delivered_not_raised(self) -> None:
        """The boundary: the stream began, so it must end with a terminal event."""
        script = FakeScript(
            generations=(
                FakeGeneration(
                    failure=FakeFailure(mode=FakeFailureMode.UNAVAILABLE, after_chunks=0)
                ),
            )
        )

        events = list(_provider(script).stream(_request(_identity())))

        assert len(events) == 1
        assert isinstance(events[0], StreamFailed)
        assert events[0].partial_text == ""

    def test_a_failure_scheduled_past_the_end_fires_where_the_stream_would_have_completed(
        self,
    ) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    word_count=2,
                    failure=FakeFailure(mode=FakeFailureMode.REJECTED, after_chunks=10_000),
                ),
            )
        )

        events = list(_provider(script).stream(_request(_identity())))

        assert isinstance(events[-1], StreamFailed)
        assert isinstance(events[-1].error, ProviderRejected)

    def test_scripted_details_override_the_defaults(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    failure=FakeFailure(
                        mode=FakeFailureMode.REJECTED,
                        message="model requires more VRAM than is free",
                        details={"status_code": 507},
                    )
                ),
            )
        )

        with pytest.raises(ProviderRejected) as raised:
            _provider(script).generate(_request(_identity()))

        assert raised.value.details["status_code"] == 507
        assert raised.value.details["provider_message"] == "unknown option 'num_ctx'"
        assert "VRAM" in str(raised.value)

    def test_a_context_failure_asks_for_more_than_the_ceiling(self) -> None:
        """Two numbers that contradicted each other would be a fake teaching a wrong shape."""
        script = FakeScript(
            generations=(
                FakeGeneration(failure=FakeFailure(mode=FakeFailureMode.CONTEXT_LIMIT_EXCEEDED)),
            )
        )

        with pytest.raises(ContextLimitExceeded) as raised:
            _provider(script).generate(_request(_identity()))

        assert raised.value.details["requested_tokens"] > raised.value.details["maximum_tokens"]

    def test_a_context_failure_may_name_no_ceiling_at_all(self) -> None:
        """A provider that refuses without saying how much it would have allowed is real."""
        script = FakeScript(
            models=(FakeModel(name="bare"),),
            generations=(
                FakeGeneration(failure=FakeFailure(mode=FakeFailureMode.CONTEXT_LIMIT_EXCEEDED)),
            ),
        )
        provider = _provider(script)

        with pytest.raises(ContextLimitExceeded) as raised:
            provider.generate(_request(provider.resolve("bare")))

        assert not is_supported(raised.value.details["maximum_tokens"])


class TestTimeouts:
    """A default exists so that ``None`` can mean "the default" and never "no timeout"."""

    def test_a_request_limit_is_enforced_against_the_simulated_clock(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(word_count=4, first_chunk_delay_ms=100, chunk_delay_ms=100),
            )
        )

        with pytest.raises(ProviderTimeout) as raised:
            _provider(script).generate(_request(_identity(), timeout_seconds=0.2))

        assert raised.value.details["limit_seconds"] == pytest.approx(0.2)
        assert raised.value.details["elapsed_seconds"] > 0.2

    def test_a_stream_that_runs_over_fails_where_the_budget_ran_out(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(word_count=4, first_chunk_delay_ms=100, chunk_delay_ms=100),
            )
        )

        events = list(_provider(script).stream(_request(_identity(), timeout_seconds=0.25)))

        assert len(_text_deltas(events)) == 2
        assert isinstance(events[-1], StreamFailed)
        assert isinstance(events[-1].error, ProviderTimeout)

    def test_the_adapter_default_applies_when_a_request_names_none(self) -> None:
        script = FakeScript(generations=(FakeGeneration(word_count=1, first_chunk_delay_ms=200),))

        with pytest.raises(ProviderTimeout):
            _provider(script, default_timeout_seconds=0.1).generate(_request(_identity()))

    def test_a_default_of_zero_is_refused(self) -> None:
        with pytest.raises(ValidationError) as raised:
            FakeProvider(default_timeout_seconds=0)

        assert raised.value.details["field"] == "default_timeout_seconds"

    def test_a_default_of_infinity_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FakeProvider(default_timeout_seconds=float("inf"))


class TestCatalogue:
    """Discovery, resolution and the digest rules that decide what an identity can claim."""

    def test_the_default_catalogue_is_fully_populated(self) -> None:
        """A default whose architecture fields were absent would exercise only the easy path."""
        descriptor = _provider().list_models()[0]

        assert descriptor.layers == DEFAULT_MODEL.layers
        assert descriptor.kv_heads != descriptor.attention_heads
        assert ModelCapabilityFlag.TOOLS in descriptor.declared_capabilities

    def test_an_empty_catalogue_is_a_real_state(self) -> None:
        provider = _provider(FakeScript(models=()))

        assert list(provider.list_models()) == []
        with pytest.raises(ModelNotFound) as raised:
            provider.resolve("anything")
        assert raised.value.details["known_model_count"] == 0

    def test_a_twenty_model_catalogue_lists_in_order(self) -> None:
        models = tuple(FakeModel(name=f"model-{index:02d}") for index in range(20))

        descriptors = _provider(FakeScript(models=models)).list_models()

        assert [descriptor.identity.provider_model_name for descriptor in descriptors] == [
            model.name for model in models
        ]

    def test_an_exact_name_resolves(self) -> None:
        assert _provider().resolve(_KNOWN_MODEL).provider_model_name == _KNOWN_MODEL

    def test_an_alias_resolves_to_the_providers_own_name(self) -> None:
        assert _provider().resolve("fake-model:latest").provider_model_name == _KNOWN_MODEL

    def test_a_unique_prefix_resolves(self) -> None:
        assert _provider().resolve("fake-model:8b").provider_model_name == _KNOWN_MODEL

    def test_an_ambiguous_prefix_is_an_error_rather_than_a_choice(self) -> None:
        """Picking one would run weights the caller did not ask for, with no way to tell."""
        script = FakeScript(models=(FakeModel(name="qwen:7b"), FakeModel(name="qwen:14b")))

        with pytest.raises(ModelNotFound) as raised:
            _provider(script).resolve("qwen")

        assert raised.value.details["matched_model_count"] == 2

    def test_resolution_through_an_alias_is_recorded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec §11.8: never resolve silently in a way that hides a retag."""
        with caplog.at_level(logging.DEBUG, logger="modelrack.providers.fake"):
            _provider().resolve("fake-model:latest")

        assert any(record.message == "fake.model.resolved" for record in caplog.records)

    def test_the_library_says_nothing_at_info_or_above(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec §17: a library must not configure or spam the host's logs."""
        with caplog.at_level(logging.INFO, logger="modelrack"):
            provider = _provider()
            provider.resolve("fake-model:latest")
            provider.generate(_request(_identity(provider)))

        assert caplog.records == []

    @pytest.mark.parametrize(
        ("digest", "expected"),
        [
            (None, IdentityConfidence.NAME_ONLY),
            ("a" * 64, IdentityConfidence.DIGEST),
            ("sha256:" + "b" * 64, IdentityConfidence.DIGEST),
            ("SHA256:" + "C" * 64, IdentityConfidence.DIGEST),
            ("  " + "d" * 64 + "  ", IdentityConfidence.DIGEST),
            ("e" * 12, IdentityConfidence.NAME_ONLY),
            ("z" * 64, IdentityConfidence.NAME_ONLY),
            ("md5:" + "f" * 32, IdentityConfidence.NAME_ONLY),
        ],
        ids=[
            "absent",
            "bare-hex",
            "prefixed",
            "uppercase",
            "padded",
            "truncated",
            "non-hex",
            "wrong-algorithm",
        ],
    )
    def test_a_reported_digest_normalizes_or_is_discarded(
        self, digest: str | None, expected: IdentityConfidence
    ) -> None:
        """ADR-0024 §2: a digest that will not normalize yields name_only, never a malformed one."""
        provider = _provider(FakeScript(models=(FakeModel(name="m", digest=digest),)))

        identity = provider.resolve("m")

        assert identity.identity_confidence is expected
        if expected is IdentityConfidence.DIGEST:
            assert identity.artifact_digest is not None
            assert identity.artifact_digest.startswith("sha256:")
            assert identity.artifact_digest.islower()

    def test_a_discarded_digest_leaves_a_reason_behind(self) -> None:
        """Spec §11.9: discarded *with a recorded reason*, so the loss is diagnosable."""
        provider = _provider(FakeScript(models=(FakeModel(name="m", digest="not-a-digest"),)))

        descriptor = provider.list_models()[0]

        assert "not-a-digest" in descriptor.raw["digest_discarded_reason"]

    def test_a_discarded_digest_is_recorded_even_over_a_scripted_payload(self) -> None:
        provider = _provider(
            FakeScript(models=(FakeModel(name="m", digest="bad", raw={"from": "the script"}),))
        )

        descriptor = provider.list_models()[0]

        assert descriptor.raw["from"] == "the script"
        assert "digest_discarded_reason" in descriptor.raw

    def test_inspect_returns_the_digest_the_catalogue_holds_now(self) -> None:
        """A retag is surfaced by comparison, never hidden by echoing the request back."""
        provider = _provider()
        stale = ModelIdentity(ProviderKind.FAKE, _KNOWN_MODEL, artifact_digest="sha256:" + "0" * 64)

        descriptor = provider.inspect_model(stale)

        assert descriptor.identity.artifact_digest != stale.artifact_digest
        assert descriptor.identity == provider.resolve(_KNOWN_MODEL)

    def test_inspect_uses_the_injected_clock(self) -> None:
        instant = datetime(2026, 8, 22, 14, 3, 11, 250_000, tzinfo=UTC)
        provider = FakeProvider(clock=lambda: instant)

        assert provider.inspect_model(provider.resolve(_KNOWN_MODEL)).observed_at == instant

    def test_inspecting_an_unknown_model_names_the_reference_and_the_count(self) -> None:
        provider = _provider()

        with pytest.raises(ModelNotFound) as raised:
            provider.inspect_model(ModelIdentity(ProviderKind.FAKE, "nope"))

        assert raised.value.details == {"reference": "nope", "known_model_count": 1}

    def test_a_catalogue_cannot_reach_one_reference_through_two_models(self) -> None:
        with pytest.raises(ValidationError) as raised:
            FakeScript(models=(FakeModel(name="a", aliases=("shared",)), FakeModel(name="shared")))

        assert raised.value.details["reference"] == "shared"


class TestResidency:
    """Load, unload and query, each gated by the flag that declares residency controllable."""

    def test_loading_reports_that_nothing_was_resident_yet(self) -> None:
        provider = _provider()
        identity = provider.resolve(_KNOWN_MODEL)

        loaded = provider.load(identity, RuntimeProfile(context_size=8192))

        assert loaded.already_resident is False
        assert loaded.load_ms == DEFAULT_MODEL.load_ms
        assert loaded.profile_hash == RuntimeProfile(context_size=8192).profile_hash

    def test_loading_a_resident_model_reports_no_load_at_all(self) -> None:
        """Never 0: a warm model measured as a cold start is a figure an order of magnitude out."""
        provider = _provider()
        identity = provider.resolve(_KNOWN_MODEL)
        provider.load(identity, RuntimeProfile())

        again = provider.load(identity, RuntimeProfile())

        assert again.already_resident is True
        assert not is_supported(again.load_ms)

    def test_unloading_says_whether_anything_was_evicted(self) -> None:
        provider = _provider()
        identity = provider.resolve(_KNOWN_MODEL)

        assert provider.unload(identity) is False
        provider.load(identity, RuntimeProfile())
        assert provider.unload(identity) is True

    def test_resident_models_are_listed_in_a_stable_order(self) -> None:
        models = (FakeModel(name="b"), FakeModel(name="a"))
        provider = _provider(FakeScript(models=models))
        for name in ("b", "a"):
            provider.load(provider.resolve(name), RuntimeProfile())

        assert [entry.identity.provider_model_name for entry in provider.list_resident()] == [
            "a",
            "b",
        ]

    def test_resident_entries_carry_per_device_memory(self) -> None:
        provider = _provider()
        provider.load(provider.resolve(_KNOWN_MODEL), RuntimeProfile())

        entry = provider.list_resident()[0]

        assert entry.vram_bytes == DEFAULT_MODEL.vram_bytes
        assert entry.expires_at is None

    def test_resident_entries_report_a_scripted_served_context(self) -> None:
        """ADR-0023 §4's *reported* served context, only when the script declares one.

        Parity with the real adapters: a field a real provider can report, the fake can report —
        and one the script does not declare stays ``UNSUPPORTED``, never an invented number and
        never ``max_context`` standing in for it.
        """
        models = (
            FakeModel(name="says", context_length=2048),
            FakeModel(name="silent"),
        )
        provider = _provider(FakeScript(models=models))
        for name in ("says", "silent"):
            provider.load(provider.resolve(name), RuntimeProfile())

        by_name = {entry.identity.provider_model_name: entry for entry in provider.list_resident()}

        assert by_name["says"].context_length == 2048
        assert by_name["silent"].context_length is UNSUPPORTED

    def test_reset_evicts_everything(self) -> None:
        provider = _provider()
        provider.load(provider.resolve(_KNOWN_MODEL), RuntimeProfile())

        provider.reset()

        assert list(provider.list_resident()) == []

    @pytest.mark.parametrize(
        ("capability", "action"),
        [("force_unload", "load"), ("force_unload", "unload"), ("residency_query", "list")],
    )
    def test_residency_is_refused_when_it_is_not_declared(
        self, capability: str, action: str
    ) -> None:
        provider = _provider(FakeScript(capabilities=MINIMAL_CAPABILITIES))
        identity = provider.resolve(_KNOWN_MODEL)
        calls = {
            "load": lambda: provider.load(identity, RuntimeProfile()),
            "unload": lambda: provider.unload(identity),
            "list": provider.list_resident,
        }

        with pytest.raises(CapabilityUnsupported) as raised:
            calls[action]()

        assert raised.value.details["capability"] == capability


class TestHealthAndAvailability:
    """The degradation matrix row a fake has to be able to produce: the runtime is not running."""

    def test_a_healthy_provider_describes_what_it_serves(self) -> None:
        health = _provider().health()

        assert health.status is ProviderStatus.OK
        assert health.model_count == 1
        assert health.provider_version == "fake-1.0"
        assert health.is_remote is False

    def test_a_degraded_provider_still_answers_calls(self) -> None:
        provider = _provider(FakeScript(health_status=ProviderStatus.DEGRADED))

        assert provider.health().status is ProviderStatus.DEGRADED
        assert provider.generate(_request(provider.resolve(_KNOWN_MODEL))).text

    def test_an_unavailable_provider_reports_rather_than_raises(self) -> None:
        provider = _provider(FakeScript(health_status=ProviderStatus.UNAVAILABLE))

        health = provider.health()

        assert health.status is ProviderStatus.UNAVAILABLE
        assert health.base_url == "fake://in-process"

    def test_an_unavailable_provider_claims_no_knowledge_of_what_it_serves(self) -> None:
        """Something that cannot be reached cannot be asked what it is running."""
        provider = _provider(FakeScript(health_status=ProviderStatus.UNAVAILABLE))

        health = provider.health()

        assert not is_supported(health.model_count)
        assert health.provider_version is None

    @pytest.mark.parametrize(
        "call",
        ["list_models", "resolve", "inspect_model", "generate", "stream", "list_resident"],
    )
    def test_every_other_call_raises_provider_unavailable(self, call: str) -> None:
        provider = _provider(FakeScript(health_status=ProviderStatus.UNAVAILABLE))
        identity = ModelIdentity(ProviderKind.FAKE, _KNOWN_MODEL)
        calls: dict[str, Callable[[], object]] = {
            "list_models": provider.list_models,
            "resolve": lambda: provider.resolve(_KNOWN_MODEL),
            "inspect_model": lambda: provider.inspect_model(identity),
            "generate": lambda: provider.generate(_request(identity)),
            "stream": lambda: provider.stream(_request(identity)),
            "list_resident": provider.list_resident,
        }

        with pytest.raises(ProviderUnavailable) as raised:
            calls[call]()

        assert raised.value.details["base_url"] == "fake://in-process"
        assert raised.value.details["reason"] == "connection_refused"

    def test_capabilities_answer_even_while_unreachable(self) -> None:
        """A caller decides whether it *may* stream before it discovers whether it *can* connect."""
        provider = _provider(FakeScript(health_status=ProviderStatus.UNAVAILABLE))

        assert provider.capabilities() == FULL_CAPABILITIES

    def test_a_remote_provider_is_flagged_rather_than_hidden(self) -> None:
        provider = _provider(FakeScript(base_url="http://gpu-box.lan:11434", is_remote=True))

        health = provider.health()

        assert health.is_remote is True
        assert health.base_url == "http://gpu-box.lan:11434"


class TestCapabilityGating:
    """A capability that was not declared is refused, never accepted and quietly dropped."""

    @pytest.mark.parametrize(
        ("capability", "overrides"),
        [
            ("json_mode", {"response_format": ResponseFormat(kind=ResponseFormatKind.JSON)}),
            (
                "structured_output",
                {
                    "response_format": ResponseFormat(
                        kind=ResponseFormatKind.JSON_SCHEMA, schema=_ANSWER_SCHEMA
                    )
                },
            ),
            ("tool_calling", {"tools": (_WEATHER_TOOL,)}),
            ("context_configurable", {"runtime_profile": RuntimeProfile(context_size=4096)}),
        ],
    )
    def test_an_undeclared_capability_is_named_in_the_refusal(
        self, capability: str, overrides: dict[str, Any]
    ) -> None:
        provider = _provider(FakeScript(capabilities=MINIMAL_CAPABILITIES))

        with pytest.raises(CapabilityUnsupported) as raised:
            provider.generate(_request(provider.resolve(_KNOWN_MODEL), **overrides))

        assert raised.value.details["capability"] == capability

    def test_plain_text_needs_no_capability_at_all(self) -> None:
        provider = _provider(FakeScript(capabilities=MINIMAL_CAPABILITIES))

        result = provider.generate(
            _request(
                provider.resolve(_KNOWN_MODEL),
                response_format=ResponseFormat(kind=ResponseFormatKind.TEXT),
            )
        )

        assert result.text

    def test_a_declared_context_is_served_and_enforced(self) -> None:
        """Spec §11.10's other half: a provider that accepts a context can be asked for too much."""
        provider = _provider()
        identity = provider.resolve(_KNOWN_MODEL)

        with pytest.raises(ContextLimitExceeded) as raised:
            provider.generate(
                _request(
                    identity,
                    runtime_profile=RuntimeProfile(context_size=8),
                    sampling=SamplingParameters(max_output_tokens=64),
                )
            )

        assert raised.value.details["maximum_tokens"] == 8
        assert raised.value.details["requested_tokens"] > 8

    def test_a_context_that_fits_is_simply_served(self) -> None:
        provider = _provider()

        result = provider.generate(
            _request(
                provider.resolve(_KNOWN_MODEL), runtime_profile=RuntimeProfile(context_size=4096)
            )
        )

        assert result.text == _GOLDEN_TEXT

    def test_a_refused_call_does_not_consume_a_scripted_generation(self) -> None:
        """A call that never reached a provider must not eat a step of the workflow's script."""
        provider = _provider(
            FakeScript(
                capabilities=MINIMAL_CAPABILITIES,
                generations=(FakeGeneration(text="the only answer"),),
                repeat_final_generation=False,
            )
        )
        identity = provider.resolve(_KNOWN_MODEL)

        with pytest.raises(CapabilityUnsupported):
            provider.generate(_request(identity, tools=(_WEATHER_TOOL,)))

        assert provider.generations_consumed == 0
        assert provider.generate(_request(identity)).text == "the only answer"


class TestScriptSequencing:
    """Successive calls, and what happens at the end of a script."""

    def test_generations_are_consumed_one_per_call(self) -> None:
        provider = _provider(
            FakeScript(
                generations=(
                    FakeGeneration(text="first"),
                    FakeGeneration(text="second"),
                    FakeGeneration(text="third"),
                )
            )
        )
        identity = provider.resolve(_KNOWN_MODEL)

        answers = [provider.generate(_request(identity)).text for _ in range(3)]

        assert answers == ["first", "second", "third"]
        assert provider.generations_consumed == 3

    def test_streaming_consumes_a_generation_too(self) -> None:
        provider = _provider(
            FakeScript(generations=(FakeGeneration(text="first"), FakeGeneration(text="second")))
        )
        identity = provider.resolve(_KNOWN_MODEL)
        list(provider.stream(_request(identity)))

        assert provider.generate(_request(identity)).text == "second"

    def test_the_final_generation_answers_every_call_after_it_by_default(self) -> None:
        provider = _provider(FakeScript(generations=(FakeGeneration(text="only"),)))
        identity = provider.resolve(_KNOWN_MODEL)

        assert [provider.generate(_request(identity)).text for _ in range(3)] == ["only"] * 3

    def test_exhausting_a_strict_script_is_loud(self) -> None:
        """A stage that quietly made an extra model call is the defect under test."""
        provider = _provider(
            FakeScript(generations=(FakeGeneration(text="only"),), repeat_final_generation=False)
        )
        identity = provider.resolve(_KNOWN_MODEL)
        provider.generate(_request(identity))

        with pytest.raises(ValidationError) as raised:
            provider.generate(_request(identity))

        assert raised.value.details == {"generation_count": 1, "call_index": 1}

    def test_reset_rewinds_to_the_start_of_the_script(self) -> None:
        provider = _provider(
            FakeScript(generations=(FakeGeneration(text="first"), FakeGeneration(text="second")))
        )
        identity = provider.resolve(_KNOWN_MODEL)
        provider.generate(_request(identity))

        provider.reset()

        assert provider.generations_consumed == 0
        assert provider.generate(_request(identity)).text == "first"

    def test_concurrent_calls_each_consume_their_own_generation(self) -> None:
        """The mutable state is a lock away from being a shared-counter bug."""
        provider = _provider(
            FakeScript(generations=tuple(FakeGeneration(text=str(index)) for index in range(20)))
        )
        identity = provider.resolve(_KNOWN_MODEL)
        answers: list[str] = []
        lock = threading.Lock()

        def call() -> None:
            text = provider.generate(_request(identity)).text
            with lock:
                answers.append(text)

        threads = [threading.Thread(target=call) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(answers, key=int) == [str(index) for index in range(20)]
        assert provider.generations_consumed == 20


class TestScriptValidation:
    """A script cannot describe a provider the fake could only imitate by lying."""

    @pytest.mark.parametrize("capability", ["logprobs", "kv_metrics", "embedding"])
    def test_a_capability_the_types_cannot_carry_is_refused(self, capability: str) -> None:
        with pytest.raises(ValidationError) as raised:
            FakeScript(capabilities=dataclasses.replace(FULL_CAPABILITIES, **{capability: True}))

        assert raised.value.details["capabilities"] == [capability]

    def test_token_level_chunks_cannot_be_claimed_over_arbitrary_fragments(self) -> None:
        """A caller is entitled to divide by the delta count and call it per-token latency."""
        with pytest.raises(ValidationError) as raised:
            FakeScript(generations=(FakeGeneration(chunk_size=64),))

        assert raised.value.details["field"] == "token_level_chunks"

    def test_hand_placed_chunks_cannot_be_claimed_as_tokens_either(self) -> None:
        with pytest.raises(ValidationError):
            FakeScript(generations=(FakeGeneration(chunks=("ab", "cd")),))

    def test_counts_cannot_be_scripted_onto_a_provider_that_counts_nothing(self) -> None:
        with pytest.raises(ValidationError) as raised:
            FakeScript(
                capabilities=MINIMAL_CAPABILITIES,
                generations=(FakeGeneration(output_tokens=5),),
            )

        assert raised.value.details["scripted"] == ["output_tokens"]

    def test_tool_calls_cannot_be_scripted_onto_a_provider_that_declares_none(self) -> None:
        with pytest.raises(ValidationError) as raised:
            FakeScript(
                capabilities=dataclasses.replace(FULL_CAPABILITIES, tool_calling=False),
                generations=(FakeGeneration(word_count=0, tool_calls=(FakeToolCall(name="a"),)),),
            )

        assert raised.value.details["field"] == "tool_calls"

    def test_a_script_must_describe_at_least_one_call(self) -> None:
        with pytest.raises(ValidationError):
            FakeScript(generations=())

    def test_a_blank_base_url_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FakeScript(base_url="  ")

    def test_text_and_chunks_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError):
            FakeGeneration(text="a", chunks=("a",))

    def test_a_word_count_cannot_accompany_text_that_supplies_it(self) -> None:
        with pytest.raises(ValidationError):
            FakeGeneration(text="a", word_count=3)

    @pytest.mark.parametrize("word_count", [-1, 1.5, True])
    def test_an_impossible_word_count_is_refused(self, word_count: Any) -> None:
        with pytest.raises(ValidationError):
            FakeGeneration(word_count=word_count)

    @pytest.mark.parametrize("chunk_size", [0, -3, 1 << 21, 2.5])
    def test_an_impossible_chunk_size_is_refused(self, chunk_size: Any) -> None:
        with pytest.raises(ValidationError):
            FakeGeneration(chunk_size=chunk_size)

    @pytest.mark.parametrize("delay", [-1.0, float("inf"), float("nan"), "soon"])
    def test_an_impossible_delay_is_refused(self, delay: Any) -> None:
        with pytest.raises(ValidationError):
            FakeGeneration(chunk_delay_ms=delay)

    @pytest.mark.parametrize("count", [-1, 2.5])
    def test_an_impossible_token_count_is_refused(self, count: Any) -> None:
        with pytest.raises(ValidationError):
            FakeGeneration(output_tokens=count)

    def test_a_client_measurement_cannot_be_scripted_as_a_backend_claim(self) -> None:
        """The fake measures its own; a provider only ever claims what it spent (spec §11.3)."""
        with pytest.raises(ValidationError) as raised:
            FakeGeneration(backend_timing=Timing(client_wall_ms=100))

        assert raised.value.details["field"] == "backend_timing.client_wall_ms"

    @pytest.mark.parametrize("after_chunks", [-1, 1.5])
    def test_an_impossible_failure_point_is_refused(self, after_chunks: Any) -> None:
        with pytest.raises(ValidationError):
            FakeFailure(
                mode=FakeFailureMode.TIMEOUT,
                after_chunks=after_chunks,
            )

    def test_a_nameless_tool_call_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FakeToolCall(name="   ")

    def test_a_blank_call_id_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FakeToolCall(name="a", id=" ")

    def test_a_nameless_model_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FakeModel(name="")

    def test_a_blank_alias_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FakeModel(name="a", aliases=("",))

    def test_a_model_cannot_alias_its_own_name(self) -> None:
        with pytest.raises(ValidationError):
            FakeModel(name="a", aliases=("a",))

    @pytest.mark.parametrize("seed", [1.5, True, "seven"])
    def test_a_non_integer_seed_is_refused(self, seed: Any) -> None:
        with pytest.raises(ValidationError):
            FakeProvider(seed=seed)


class TestPublicSurface:
    """What a downstream repository imports, and what it must not have to import."""

    def test_the_documented_import_path_works(self) -> None:
        """Phase 2 acceptance criterion 1, asserted rather than assumed."""
        from modelrack.testing import FakeProvider as ExportedProvider  # noqa: PLC0415
        from modelrack.testing import FakeScript as ExportedScript  # noqa: PLC0415

        assert isinstance(ExportedProvider(ExportedScript()), Provider)

    def test_the_fake_satisfies_the_protocol_structurally(self) -> None:
        assert isinstance(_provider(), Provider)

    def test_the_test_double_is_not_in_the_production_namespace(self) -> None:
        """One autocomplete away from production code is one refactor away from inside it."""
        import modelrack  # noqa: PLC0415

        assert "FakeProvider" not in modelrack.__all__
        assert not hasattr(modelrack, "FakeProvider")

    def test_the_seed_and_script_are_readable_back(self) -> None:
        script = FakeScript(provider_version="fake-9")
        provider = FakeProvider(script, seed=99)

        assert provider.script is script
        assert provider.seed == 99

    def test_the_published_token_width_is_what_the_counts_use(self) -> None:
        """A consumer asserting an expected count needs the arithmetic the fake used."""
        script = FakeScript(
            generations=(FakeGeneration(text="a" * (SIMULATED_TOKEN_CHARACTERS * 5)),)
        )

        usage = _provider(script).generate(_request(_identity())).usage

        assert usage.tokens.output_tokens == 5


class TestDiagnosticsAndSecrets:
    """What travels on a result, and what deliberately does not."""

    def test_caller_metadata_never_reaches_the_provider_payload(self) -> None:
        marker = "run-7f3c-correlation"

        result = _provider().generate(_request(_identity(), metadata={"run_id": marker}))

        assert marker not in json.dumps(dict(result.raw))

    def test_the_prompt_never_reaches_the_provider_payload(self) -> None:
        confidential = "the-quiet-part"

        result = _provider().generate(
            _request(_identity(), messages=(Message(role=Role.USER, content=confidential),))
        )

        assert confidential not in json.dumps(dict(result.raw))

    def test_generated_text_is_not_accumulated_twice(self) -> None:
        """Spec §15: memory per active stream stays flat regardless of response length."""
        result = _provider().generate(_request(_identity()))

        assert result.text not in json.dumps(dict(result.raw), ensure_ascii=False)

    def test_the_payload_explains_the_result_it_came_with(self) -> None:
        result = _provider().generate(_request(_identity()))

        assert result.raw["finish_reason"] == FinishReason.STOP.value
        assert result.raw["seed"] == 7
        assert result.raw["model"] == _KNOWN_MODEL

    def test_the_provider_version_travels_on_every_result(self) -> None:
        """ADR-0017: a provider upgrade is drift that reduces confidence in earlier evidence."""
        provider = _provider(FakeScript(provider_version="fake-2.0"))

        assert (
            provider.generate(_request(provider.resolve(_KNOWN_MODEL))).provider_version
            == "fake-2.0"
        )


class TestCompletionStyleRequests:
    """The other half of the request vocabulary: a raw prompt rather than a conversation."""

    def test_a_prompt_is_answered_like_a_conversation(self) -> None:
        result = _provider().generate(
            GenerationRequest(identity=_identity(), prompt="Explain KV caching.")
        )

        assert result.text
        assert result.finish_reason is FinishReason.STOP

    def test_a_prompt_and_a_conversation_are_different_inputs(self) -> None:
        identity = _identity()

        prompted = _provider().generate(GenerationRequest(identity=identity, prompt="hello"))
        chatted = _provider().generate(_request(identity))

        assert prompted.text != chatted.text

    def test_a_longer_prompt_costs_more_input_tokens(self) -> None:
        identity = _identity()

        short = _provider().generate(GenerationRequest(identity=identity, prompt="hi"))
        long = _provider().generate(GenerationRequest(identity=identity, prompt="hi " * 200))

        assert long.usage.tokens.input_tokens > short.usage.tokens.input_tokens

    def test_offering_tools_costs_context(self) -> None:
        """A consumer measuring the price of a large tool set needs the fake to charge for it."""
        identity = _identity()

        without = _provider().generate(_request(identity))
        with_tools = _provider().generate(_request(identity, tools=(_WEATHER_TOOL,)))

        assert with_tools.usage.tokens.input_tokens > without.usage.tokens.input_tokens

    def test_a_prior_tool_call_in_the_conversation_costs_context(self) -> None:
        identity = _identity()
        answered = ToolCall(id="c1", name="get_weather", arguments={"city": "Berlin"})
        history = (
            Message(role=Role.USER, content="weather in Berlin?"),
            Message(role=Role.ASSISTANT, tool_calls=(answered,)),
            Message(role=Role.TOOL, content="17C", tool_call_id="c1"),
        )

        without = _provider().generate(_request(identity, messages=history[:1]))
        with_history = _provider().generate(_request(identity, messages=history))

        assert with_history.usage.tokens.input_tokens > without.usage.tokens.input_tokens


class TestSchemaShapes:
    """What the schema-shaped generator honours, and what it deliberately does not."""

    def test_every_scalar_type_is_produced_with_its_python_type(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "enabled": {"type": "boolean"},
                "note": {"type": "string"},
                "nothing": {"type": "null"},
            },
        }

        document = json.loads(self._generate(schema))

        assert isinstance(document["count"], int)
        assert isinstance(document["ratio"], float)
        assert isinstance(document["enabled"], bool)
        assert isinstance(document["note"], str)
        assert document["nothing"] is None

    def test_a_union_type_takes_the_first_member(self) -> None:
        schema = {"type": "object", "properties": {"value": {"type": ["integer", "null"]}}}

        assert isinstance(json.loads(self._generate(schema))["value"], int)

    def test_an_object_with_no_properties_is_empty(self) -> None:
        assert json.loads(self._generate({"type": "object"})) == {}

    def test_a_typeless_schema_with_properties_is_still_an_object(self) -> None:
        schema = {"properties": {"name": {"type": "string"}}}

        assert set(json.loads(self._generate(schema))) == {"name"}

    def test_only_required_properties_appear_when_some_are_named(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }

        assert set(json.loads(self._generate(schema))) == {"a"}

    def test_a_constraint_the_generator_does_not_honour_is_left_to_the_caller(self) -> None:
        """Claiming full JSON Schema conformance would mean a consumer's validator never ran."""
        schema = {"type": "object", "properties": {"count": {"type": "integer", "minimum": 5000}}}

        assert json.loads(self._generate(schema))["count"] < 5000

    @staticmethod
    def _generate(schema: dict[str, Any]) -> str:
        provider = _provider()
        return provider.generate(
            _request(
                provider.resolve(_KNOWN_MODEL),
                response_format=ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema=schema),
            )
        ).text


class TestCancellationRaces:
    """Cancellation arriving at the two moments a simple loop-driven test cannot reach."""

    def test_a_token_set_while_the_caller_waits_lands_on_this_delta(self) -> None:
        """The injected sleep stands in for the thread that cancels during the wait."""
        token = CancellationToken()
        script = FakeScript(
            generations=(FakeGeneration(word_count=6, first_chunk_delay_ms=1, chunk_delay_ms=1),)
        )
        provider = _provider(script, sleep=lambda _seconds: token.cancel())

        events = list(provider.stream(_request(_identity(), cancel=token)))

        assert len(events) == 1
        assert isinstance(events[0], StreamFailed)
        assert isinstance(events[0].error, GenerationCancelled)

    def test_cancelling_on_the_final_delta_still_yields_a_cancelled_terminal(self) -> None:
        """The check after the loop: the answer was complete, but the caller asked to stop."""
        token = CancellationToken()
        script = FakeScript(generations=(FakeGeneration(text="abcdefgh"),))
        events: list[StreamEvent] = []

        for event in _provider(script).stream(_request(_identity(), cancel=token)):
            events.append(event)
            if len(_text_deltas(events)) == 2:
                token.cancel()

        assert len(_text_deltas(events)) == 2
        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == "abcdefgh"


class TestErrorPayloadLimits:
    """An error object is not a place to move an unbounded response (spec §13)."""

    def test_a_scripted_body_is_capped(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    failure=FakeFailure(
                        mode=FakeFailureMode.UNPARSEABLE_BODY, details={"body": "x" * 5_000}
                    )
                ),
            )
        )

        with pytest.raises(ProviderProtocolError) as raised:
            _provider(script).generate(_request(_identity()))

        body = raised.value.details["body"]
        assert len(body) < 1_000
        assert body.endswith("…")

    def test_a_body_that_is_not_text_passes_through_untouched(self) -> None:
        """A script may record a parsed payload; the character cap has nothing to say about it."""
        script = FakeScript(
            generations=(
                FakeGeneration(
                    failure=FakeFailure(
                        mode=FakeFailureMode.UNEXPECTED_SHAPE,
                        details={"body": {"done": False}},
                    )
                ),
            )
        )

        with pytest.raises(ProviderProtocolError) as raised:
            _provider(script).generate(_request(_identity()))

        assert raised.value.details["body"] == {"done": False}

    def test_a_short_body_survives_unchanged(self) -> None:
        script = FakeScript(
            generations=(
                FakeGeneration(
                    failure=FakeFailure(
                        mode=FakeFailureMode.UNPARSEABLE_BODY, details={"body": "upstream reset"}
                    )
                ),
            )
        )

        with pytest.raises(ProviderProtocolError) as raised:
            _provider(script).generate(_request(_identity()))

        assert raised.value.details["body"] == "upstream reset"
