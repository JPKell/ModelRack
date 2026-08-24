"""Tests for :mod:`modelrack.types` — the provider-neutral request and result vocabulary.

Two of these classes carry a Phase 1 acceptance criterion rather than an ordinary invariant:
:class:`TestTimingKeepsBackendAndClientApart` proves criterion 2 ("no field named in a way that
would let a caller confuse backend and client timings"), and
:class:`TestToolCallOnlyResponses` proves the result can express the response the plan names as a
likely failure mode.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
from baseaicore import UNSUPPORTED, ModelIdentity, RuntimeProfile, TokenUsage, ValidationError

from modelrack import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    Message,
    ResponseFormat,
    ResponseFormatKind,
    Role,
    SamplingParameters,
    Timing,
    ToolCall,
    ToolDefinition,
)

_ALL_TIMING_FIELDS = [f.name for f in dataclasses.fields(Timing)]


class TestTimingKeepsBackendAndClientApart:
    """Acceptance criterion 2, and spec §11.3.

    The separation is the point: what a provider reported about its own work and what this
    process observed disagree for real reasons, and a benchmark comparing one runtime's
    self-report against another's wall clock is comparing nothing.
    """

    def test_every_field_declares_which_clock_it_came_from(self) -> None:
        assert all(f.startswith(("backend_", "client_")) for f in _ALL_TIMING_FIELDS)

    def test_there_is_no_combined_duration_field(self) -> None:
        """The moment an unprefixed total exists, callers reach for it."""
        unprefixed = [f for f in _ALL_TIMING_FIELDS if not f.startswith(("backend_", "client_"))]
        assert unprefixed == []

    def test_both_families_are_present(self) -> None:
        assert any(f.startswith("backend_") for f in _ALL_TIMING_FIELDS)
        assert any(f.startswith("client_") for f in _ALL_TIMING_FIELDS)

    def test_the_fields_consumers_persist_all_exist(self) -> None:
        """FreeWeight's `samples` table names exactly these; a missing one is unrecordable."""
        assert set(_ALL_TIMING_FIELDS) == {
            "client_wall_ms",
            "client_ttft_ms",
            "backend_load_ms",
            "backend_prompt_eval_ms",
            "backend_decode_ms",
            "backend_total_ms",
        }

    def test_backend_total_is_not_recomputed_from_the_parts(self) -> None:
        """A provider may account for time the phases do not cover; its total is its own."""
        timing = Timing(
            backend_load_ms=100,
            backend_prompt_eval_ms=50,
            backend_decode_ms=300,
            backend_total_ms=500,
        )
        assert timing.backend_total_ms == 500

    def test_every_duration_defaults_to_unsupported_not_zero(self) -> None:
        timing = Timing()
        assert all(getattr(timing, f) is UNSUPPORTED for f in _ALL_TIMING_FIELDS)

    @pytest.mark.parametrize("field_name", _ALL_TIMING_FIELDS)
    def test_a_negative_duration_is_rejected(self, field_name: str) -> None:
        """A negative duration means two readings came from different clocks."""
        with pytest.raises(ValidationError, match=field_name):
            Timing(**{field_name: -1.0})

    def test_a_non_finite_duration_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Timing(client_wall_ms=float("nan"))

    def test_a_bool_is_not_a_duration(self) -> None:
        with pytest.raises(ValidationError):
            Timing(client_wall_ms=True)

    def test_zero_is_accepted_as_a_real_reading(self) -> None:
        """`0` is refused as a *stand-in for absent*, never as a genuine measurement."""
        assert Timing(backend_load_ms=0).backend_load_ms == 0


class TestGenerationUsage:
    """Billable counts and observation counts, kept in one object without being confused."""

    def test_billable_counts_are_baseaicore_token_usage(self) -> None:
        """A consumer storing a cost stores this object directly (ADR-0030)."""
        assert isinstance(GenerationUsage().tokens, TokenUsage)

    def test_every_count_defaults_to_unsupported(self) -> None:
        usage = GenerationUsage()
        assert usage.thinking_tokens is UNSUPPORTED
        assert usage.tool_tokens is UNSUPPORTED
        assert usage.output_chars is UNSUPPORTED
        assert usage.tokens.input_tokens is UNSUPPORTED

    def test_the_fields_consumers_persist_all_exist(self) -> None:
        """FreeWeight's `samples` row, LoadCoach's and IdeaPress's job rows."""
        names = {f.name for f in dataclasses.fields(GenerationUsage)}
        assert {
            "thinking_tokens",
            "tool_tokens",
            "output_chars",
            "output_words",
            "output_bytes",
        } <= (names)

    def test_thinking_tokens_is_a_breakdown_not_a_fifth_billing_class(self) -> None:
        """It lives outside `tokens` precisely so a total cannot double-count it."""
        billing_fields = {f.name for f in dataclasses.fields(TokenUsage)}
        assert "thinking_tokens" not in billing_fields

    def test_a_total_over_billable_counts_excludes_thinking(self) -> None:
        usage = GenerationUsage(
            tokens=TokenUsage(
                input_tokens=100, output_tokens=50, cache_write_tokens=0, cache_read_tokens=0
            ),
            thinking_tokens=20,
        )
        assert usage.tokens.total_tokens == 150  # not 170

    @pytest.mark.parametrize(
        ("field_name", "build"),
        [
            ("thinking_tokens", lambda: GenerationUsage(thinking_tokens=1.5)),  # type: ignore[arg-type]
            ("tool_tokens", lambda: GenerationUsage(tool_tokens=1.5)),  # type: ignore[arg-type]
        ],
    )
    def test_a_fractional_token_count_is_rejected(
        self, field_name: str, build: Callable[[], GenerationUsage]
    ) -> None:
        """A fractional token count is a parsing error in the adapter, never a real reading."""
        with pytest.raises(ValidationError, match=field_name):
            build()

    @pytest.mark.parametrize(
        ("field_name", "build"),
        [
            ("thinking_tokens", lambda: GenerationUsage(thinking_tokens=-1)),
            ("tool_tokens", lambda: GenerationUsage(tool_tokens=-1)),
            ("output_chars", lambda: GenerationUsage(output_chars=-1)),
            ("output_bytes", lambda: GenerationUsage(output_bytes=-1)),
        ],
    )
    def test_a_negative_count_is_rejected(
        self, field_name: str, build: Callable[[], GenerationUsage]
    ) -> None:
        with pytest.raises(ValidationError, match=field_name):
            build()

    def test_a_negative_billable_count_is_rejected_by_the_domain_type(self) -> None:
        """baseaicore.TokenUsage validates itself; this is not reimplemented here."""
        with pytest.raises(ValidationError):
            GenerationUsage(tokens=TokenUsage(input_tokens=-1))


class TestMessage:
    """A turn is valid only in the shapes its role actually permits."""

    def test_a_plain_user_turn(self) -> None:
        assert Message(role=Role.USER, content="hi").content == "hi"

    def test_an_assistant_turn_may_carry_tool_calls_with_empty_content(self) -> None:
        call = ToolCall(id="c1", name="search")
        assert Message(role=Role.ASSISTANT, tool_calls=(call,)).content == ""

    def test_a_tool_result_must_name_the_call_it_answers(self) -> None:
        with pytest.raises(ValidationError, match="tool_call_id"):
            Message(role=Role.TOOL, content="42")

    def test_a_tool_result_with_its_id_is_valid(self) -> None:
        assert Message(role=Role.TOOL, content="42", tool_call_id="c1").tool_call_id == "c1"

    @pytest.mark.parametrize("role", [Role.USER, Role.SYSTEM, Role.TOOL])
    def test_only_an_assistant_may_carry_tool_calls(self, role: Role) -> None:
        """A tool call is something the model requests, never something sent to it."""
        with pytest.raises(ValidationError, match="tool_calls"):
            Message(
                role=role, content="x", tool_call_id="c1", tool_calls=(ToolCall(id="c", name="n"),)
            )

    def test_an_entirely_empty_turn_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role=Role.USER)

    def test_a_message_is_immutable(self) -> None:
        message = Message(role=Role.USER, content="hi")
        with pytest.raises(dataclasses.FrozenInstanceError):
            message.content = "changed"  # type: ignore[misc]


class TestToolTypes:
    """Definitions are passed through untouched; calls must be correlatable."""

    def test_parameters_are_not_rewritten(self) -> None:
        """Spec §14: tool definitions pass through unmodified."""
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        assert ToolDefinition(name="search", parameters=schema).parameters == schema

    def test_a_nameless_tool_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            ToolDefinition(name="  ")

    @pytest.mark.parametrize(
        ("field_name", "build"),
        [
            ("id", lambda: ToolCall(id="", name="search")),
            ("name", lambda: ToolCall(id="c1", name="")),
        ],
    )
    def test_a_tool_call_must_be_nameable_and_correlatable(
        self, field_name: str, build: Callable[[], ToolCall]
    ) -> None:
        with pytest.raises(ValidationError, match=field_name):
            build()

    def test_unparsed_arguments_are_preserved_for_diagnosis(self) -> None:
        """Models do emit invalid JSON here, and FreeWeight scores that as a visible failure."""
        call = ToolCall(id="c1", name="search", raw_arguments='{"q": "unterminated')
        assert call.arguments == {}
        assert call.raw_arguments == '{"q": "unterminated'


class TestResponseFormat:
    """A schema is required exactly where it is enforceable, and refused where it is not."""

    def test_text_is_the_default(self) -> None:
        assert ResponseFormat().kind is ResponseFormatKind.TEXT

    def test_json_schema_requires_a_schema(self) -> None:
        with pytest.raises(ValidationError, match="schema"):
            ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA)

    @pytest.mark.parametrize("kind", [ResponseFormatKind.TEXT, ResponseFormatKind.JSON])
    def test_other_kinds_refuse_a_schema(self, kind: ResponseFormatKind) -> None:
        """A schema that would be silently ignored is a caller mistake worth naming."""
        with pytest.raises(ValidationError, match="schema"):
            ResponseFormat(kind=kind, schema={"type": "object"})

    def test_json_schema_with_a_schema_is_valid(self) -> None:
        fmt = ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema={"type": "object"})
        assert fmt.schema == {"type": "object"}


class TestSamplingParameters:
    """None means "provider default"; a supplied value must be in a range that means something."""

    def test_everything_defaults_to_none(self) -> None:
        """Distinct from pinning a value that happens to match: only the pin is reproducible."""
        params = SamplingParameters()
        assert params.temperature is None
        assert params.seed is None
        assert params.stop == ()

    def test_a_pinned_greedy_configuration_is_valid(self) -> None:
        params = SamplingParameters(temperature=0.0, seed=42, max_output_tokens=512)
        assert (params.temperature, params.seed) == (0.0, 42)

    @pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf")])
    def test_an_impossible_temperature_is_rejected(self, temperature: float) -> None:
        with pytest.raises(ValidationError, match="temperature"):
            SamplingParameters(temperature=temperature)

    @pytest.mark.parametrize("top_p", [-0.01, 1.01])
    def test_top_p_outside_a_probability_mass_is_rejected(self, top_p: float) -> None:
        with pytest.raises(ValidationError, match="top_p"):
            SamplingParameters(top_p=top_p)

    @pytest.mark.parametrize(
        ("field_name", "build"),
        [
            ("top_k", lambda: SamplingParameters(top_k=0)),
            ("max_output_tokens", lambda: SamplingParameters(max_output_tokens=0)),
        ],
    )
    def test_zero_is_rejected_where_none_is_the_way_to_say_nothing(
        self, field_name: str, build: Callable[[], SamplingParameters]
    ) -> None:
        """`0` would ask for nothing at all; `None` is how a caller defers to the provider."""
        with pytest.raises(ValidationError, match=field_name):
            build()

    def test_a_non_positive_repeat_penalty_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="repeat_penalty"):
            SamplingParameters(repeat_penalty=0)


class TestGenerationRequest:
    """Exactly one input, and a timeout that is a real ceiling."""

    def test_a_chat_request(self, identity: ModelIdentity) -> None:
        request = GenerationRequest(
            identity=identity, messages=(Message(role=Role.USER, content="hi"),)
        )
        assert request.prompt is None

    def test_a_completion_request(self, identity: ModelIdentity) -> None:
        assert GenerationRequest(identity=identity, prompt="once upon").messages == ()

    def test_supplying_both_inputs_is_rejected(self, identity: ModelIdentity) -> None:
        """They reach different provider endpoints; an adapter would discard one silently."""
        with pytest.raises(ValidationError, match="both"):
            GenerationRequest(
                identity=identity, messages=(Message(role=Role.USER, content="hi"),), prompt="x"
            )

    def test_supplying_neither_input_is_rejected(self, identity: ModelIdentity) -> None:
        with pytest.raises(ValidationError, match="neither"):
            GenerationRequest(identity=identity)

    def test_a_default_runtime_profile_is_a_real_hashable_profile(
        self, identity: ModelIdentity
    ) -> None:
        """ADR-0023 §1: "provider defaults" is a legal profile, not a "no profile" state."""
        request = GenerationRequest(identity=identity, prompt="x")
        assert isinstance(request.runtime_profile, RuntimeProfile)
        assert len(request.runtime_profile.profile_hash) == 16

    def test_two_requests_do_not_share_a_mutable_default(self, identity: ModelIdentity) -> None:
        """The classic dataclass trap: a shared dict would leak one caller's metadata to another."""
        first = GenerationRequest(identity=identity, prompt="a")
        second = GenerationRequest(identity=identity, prompt="b")
        assert first.metadata is not second.metadata

    @pytest.mark.parametrize("timeout", [0, -1.0, float("inf"), float("nan")])
    def test_a_timeout_that_is_not_a_ceiling_is_rejected(
        self, identity: ModelIdentity, timeout: float
    ) -> None:
        """Spec §14: None means "the adapter's default", never "no timeout"."""
        with pytest.raises(ValidationError, match="timeout_seconds"):
            GenerationRequest(identity=identity, prompt="x", timeout_seconds=timeout)

    def test_none_timeout_means_the_adapters_default(self, identity: ModelIdentity) -> None:
        assert GenerationRequest(identity=identity, prompt="x").timeout_seconds is None

    def test_metadata_is_caller_correlation_and_is_carried_verbatim(
        self, identity: ModelIdentity
    ) -> None:
        request = GenerationRequest(identity=identity, prompt="x", metadata={"run_id": "01J9"})
        assert request.metadata == {"run_id": "01J9"}


class TestToolCallOnlyResponses:
    """The plan's named failure mode: "a result that cannot express a tool-call-only response"."""

    def test_a_result_may_have_empty_text_and_tool_calls(self, identity: ModelIdentity) -> None:
        result = GenerationResult(
            text="",
            identity=identity,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "kv cache"}),),
        )
        assert result.text == ""
        assert result.tool_calls[0].name == "search"

    def test_claiming_tool_calls_without_any_is_rejected(self, identity: ModelIdentity) -> None:
        """A caller told the model wants a tool, with no tool named, has nothing to do next."""
        with pytest.raises(ValidationError, match="tool_calls"):
            GenerationResult(text="", identity=identity, finish_reason=FinishReason.TOOL_CALLS)

    def test_tool_calls_alongside_text_are_allowed(self, identity: ModelIdentity) -> None:
        """A model may explain itself and then call a tool."""
        result = GenerationResult(
            text="Let me look that up.",
            identity=identity,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(ToolCall(id="c1", name="search"),),
        )
        assert result.text


class TestGenerationResult:
    """Defaults are honest, and the result stays attributable after it leaves the call site."""

    def test_unknown_is_the_default_finish_reason(self, identity: ModelIdentity) -> None:
        """Defaulting to STOP would convert a truncation into a complete answer."""
        assert GenerationResult(text="hi", identity=identity).finish_reason is FinishReason.UNKNOWN

    def test_thinking_defaults_to_unsupported_not_empty_string(
        self, identity: ModelIdentity
    ) -> None:
        """`""` would claim the model reasoned and produced nothing."""
        assert GenerationResult(text="hi", identity=identity).thinking is UNSUPPORTED

    def test_the_result_carries_its_own_identity(self, identity: ModelIdentity) -> None:
        result = GenerationResult(text="hi", identity=identity)
        assert result.identity.canonical_id == identity.canonical_id

    def test_usage_and_timing_default_to_fully_unsupported(self, identity: ModelIdentity) -> None:
        result = GenerationResult(text="hi", identity=identity)
        assert result.usage.tokens.input_tokens is UNSUPPORTED
        assert result.timing.client_wall_ms is UNSUPPORTED

    def test_two_results_do_not_share_a_mutable_raw(self, identity: ModelIdentity) -> None:
        assert GenerationResult(text="a", identity=identity).raw is not (
            GenerationResult(text="b", identity=identity).raw
        )

    def test_a_result_is_immutable(self, identity: ModelIdentity) -> None:
        result = GenerationResult(text="hi", identity=identity)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.text = "changed"  # type: ignore[misc]


class TestEnumsSerializeReadably:
    """Domain enums are StrEnum so a log line and a stored row read as the name."""

    @pytest.mark.parametrize("member", list(FinishReason))
    def test_finish_reasons_are_lowercase_strings(self, member: FinishReason) -> None:
        assert member.value == member.value.lower()
        assert isinstance(member, str)

    def test_every_spec_finish_reason_exists(self) -> None:
        assert {m.value for m in FinishReason} == {
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "cancelled",
            "error",
            "unknown",
        }

    def test_every_role_exists(self) -> None:
        assert {m.value for m in Role} == {"system", "user", "assistant", "tool"}
