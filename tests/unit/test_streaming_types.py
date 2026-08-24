"""Tests for :mod:`modelrack.streaming` — the event union and the cancellation token.

The token is the only mutable thing in this package's vocabulary, and it is set from a different
thread than the one reading it, so its invariants are asserted rather than assumed.
"""

from __future__ import annotations

import dataclasses
import threading
import typing

import pytest
from baseaicore import ModelIdentity, ValidationError

from modelrack import (
    CancellationToken,
    GenerationResult,
    ProviderTimeout,
    StreamCompleted,
    StreamEvent,
    StreamFailed,
    ThinkingDelta,
    TokenDelta,
    ToolCallDelta,
)
from modelrack.errors import GenerationCancelled


class TestCancellationToken:
    """One-way, idempotent, thread-safe, and it never discards the caller's own output."""

    def test_a_new_token_is_not_cancelled(self) -> None:
        assert CancellationToken().is_cancelled is False

    def test_cancelling_sets_it(self) -> None:
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_cancelling_is_idempotent(self) -> None:
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled is True

    def test_there_is_no_way_to_un_cancel(self) -> None:
        """A resumable stop would let a stream continue after the caller was told it ended."""
        assert not hasattr(CancellationToken(), "reset")

    def test_raise_if_cancelled_is_a_no_op_while_active(self) -> None:
        CancellationToken().raise_if_cancelled(partial_text="anything")

    def test_raise_if_cancelled_raises_once_cancelled(self) -> None:
        token = CancellationToken()
        token.cancel()
        with pytest.raises(GenerationCancelled):
            token.raise_if_cancelled()

    def test_partial_output_is_returned_to_the_caller(self) -> None:
        """Centralised here so every adapter preserves it identically."""
        token = CancellationToken()
        token.cancel()
        with pytest.raises(GenerationCancelled) as caught:
            token.raise_if_cancelled(partial_text="Local models are ")
        assert caught.value.details["partial_text"] == "Local models are "

    def test_it_can_be_cancelled_from_another_thread(self) -> None:
        """The ordinary case: a request handler stopping a background generation."""
        token = CancellationToken()
        started = threading.Event()

        def _cancel() -> None:
            started.wait(timeout=5)
            token.cancel()

        worker = threading.Thread(target=_cancel)
        worker.start()
        started.set()
        worker.join(timeout=5)
        assert token.is_cancelled is True

    def test_repr_names_the_state(self) -> None:
        token = CancellationToken()
        assert "active" in repr(token)
        token.cancel()
        assert "cancelled" in repr(token)

    def test_it_carries_no_per_instance_dict(self) -> None:
        """Slotted: one token per in-flight generation, and they should stay cheap."""
        assert not hasattr(CancellationToken(), "__dict__")


class TestDeltas:
    """Fragments are ordered, and a delta is not promised to be a token."""

    def test_a_token_delta_carries_text_and_position(self) -> None:
        delta = TokenDelta(text="Local ", index=3)
        assert (delta.text, delta.index) == ("Local ", 3)

    def test_an_empty_delta_is_allowed(self) -> None:
        """Some providers emit keep-alive chunks; rejecting them would break a real stream."""
        assert TokenDelta(text="").text == ""

    @pytest.mark.parametrize("delta_type", [TokenDelta, ThinkingDelta])
    def test_a_negative_index_is_rejected(self, delta_type: type) -> None:
        with pytest.raises(ValidationError, match="index"):
            delta_type(text="x", index=-1)

    def test_a_bool_index_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="index"):
            TokenDelta(text="x", index=True)

    def test_thinking_never_matches_the_answer_branch(self) -> None:
        """Concatenating the two would show a model's working to a user who wanted its answer."""
        match ThinkingDelta(text="the user seems to want..."):
            case TokenDelta():
                pytest.fail("reasoning content matched the answer-text branch")
            case ThinkingDelta() as reasoning:
                assert reasoning.text.startswith("the user")

    def test_a_tool_call_delta_carries_unparsed_argument_fragments(self) -> None:
        """A fragment is rarely valid JSON alone; parsing each would fail on all but the last."""
        delta = ToolCallDelta(call_index=0, name="search", arguments_fragment='{"q": "kv')
        assert delta.arguments_fragment == '{"q": "kv'

    def test_tool_call_deltas_distinguish_which_call_they_belong_to(self) -> None:
        """A model may request several calls at once, and their fragments interleave."""
        first = ToolCallDelta(call_index=0, index=1)
        second = ToolCallDelta(call_index=1, index=2)
        assert first.call_index != second.call_index

    def test_a_negative_call_index_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="call_index"):
            ToolCallDelta(call_index=-1)

    def test_deltas_are_immutable(self) -> None:
        delta = TokenDelta(text="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            delta.text = "y"  # type: ignore[misc]


class TestTerminalEvents:
    """Every stream ends with exactly one of these, and both say what happened."""

    def test_completed_carries_the_whole_result(self, identity: ModelIdentity) -> None:
        """So a streaming caller records a run exactly as a non-streaming one would."""
        result = GenerationResult(text="hello", identity=identity)
        assert StreamCompleted(result=result).result.text == "hello"

    def test_failed_carries_a_typed_error(self) -> None:
        event = StreamFailed(error=ProviderTimeout("too slow"))
        assert event.error.code == "PROVIDER_TIMEOUT"

    def test_failed_preserves_partial_output(self) -> None:
        """A sample that failed halfway is evidence; discarding the text loses the only record."""
        event = StreamFailed(error=ProviderTimeout("slow"), partial_text="Local models ")
        assert event.partial_text == "Local models "

    def test_failed_defaults_to_no_partial_text(self) -> None:
        assert StreamFailed(error=ProviderTimeout("slow")).partial_text == ""


class TestStreamEventUnion:
    """A closed union, so a caller can match exhaustively and be told when that breaks."""

    def test_the_union_contains_exactly_the_five_spec_members(self) -> None:
        members = set(typing.get_args(StreamEvent.__value__))
        assert members == {
            TokenDelta,
            ThinkingDelta,
            ToolCallDelta,
            StreamCompleted,
            StreamFailed,
        }

    @pytest.mark.parametrize(
        "event",
        [
            TokenDelta(text="a"),
            ThinkingDelta(text="b"),
            ToolCallDelta(name="search"),
            StreamFailed(error=ProviderTimeout("slow")),
        ],
    )
    def test_every_member_is_matchable_by_type(self, event: object) -> None:
        """The shape a consumer actually writes: a structural match over the union."""
        match event:
            case TokenDelta() | ThinkingDelta() | ToolCallDelta():
                matched = "delta"
            case StreamCompleted() | StreamFailed():
                matched = "terminal"
            case _:  # pragma: no cover — a new union member would land here
                matched = "unmatched"
        assert matched in {"delta", "terminal"}
