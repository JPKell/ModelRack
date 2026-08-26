"""Tests for :mod:`modelrack.events` — the optional observability hook and its silences.

Spec §17 asks for a callback that reports "request started/completed/failed, chunk received". The
tests that carry the most weight here are the ones about what an event must **not** contain: no
prompt, no generated text, no tool arguments, no API key. An event stream is a fourth channel a
credential could escape through, alongside ``raw``, error ``details`` and the DEBUG log — spec §14
names the first three and a test asserts them; this file closes the fourth.

The other load-bearing property is that **a host's failing callback cannot destroy a generation**.
A completed result thrown away because a metrics hook raised would be a far worse outcome than a
missing log line, and it is a failure mode that only shows up in production.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import (
    UNSUPPORTED,
    ModelIdentity,
    ProviderKind,
    RuntimeProfile,
    ValidationError,
    is_supported,
)

from modelrack import (
    GenerationRequest,
    Message,
    ProviderEvent,
    ProviderEventKind,
    Role,
    StreamCompleted,
    StreamFailed,
    TokenDelta,
)
from modelrack.events import EventEmitter, emit
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider
from modelrack.streaming import CancellationToken
from modelrack.testing import (
    FakeFailure,
    FakeFailureMode,
    FakeGeneration,
    FakeProvider,
    FakeScript,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_OLLAMA_URL = "http://127.0.0.1:11434"
_OPENAI_URL = "http://127.0.0.1:8080"
_MODEL = "qwen3.5:9b-q8_0"
# A fixture value asserted absent from the event stream, not a real credential.
_SECRET = "sk-do-not-leak-this-anywhere"  # noqa: S105
_PROMPT = "Explain KV caching, mentioning zebrafish."


class _Recorder:
    """A host callback that keeps what it was handed."""

    def __init__(self) -> None:
        self.events: list[ProviderEvent] = []

    def __call__(self, event: ProviderEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[ProviderEventKind]:
        return [event.kind for event in self.events]

    def of(self, kind: ProviderEventKind) -> list[ProviderEvent]:
        return [event for event in self.events if event.kind is kind]


def _request(**overrides: Any) -> GenerationRequest:
    """Build the standard request, carrying a caller correlation ID in ``metadata``."""
    fields: dict[str, Any] = {
        "identity": ModelIdentity(ProviderKind.OLLAMA, _MODEL),
        "messages": (Message(role=Role.USER, content=_PROMPT),),
        "metadata": {"run_id": "01JABCDEF0123456789ABCDEFG"},
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


class TestProviderEvent:
    def test_an_event_names_the_call_that_produced_it(self) -> None:
        event = ProviderEvent(
            kind=ProviderEventKind.REQUEST_STARTED,
            operation="generate",
            provider_kind=ProviderKind.FAKE,
        )

        assert event.operation == "generate"
        assert event.kind is ProviderEventKind.REQUEST_STARTED

    @pytest.mark.parametrize("operation", ["", "   "])
    def test_an_event_that_cannot_say_which_call_produced_it_is_refused(
        self, operation: str
    ) -> None:
        with pytest.raises(ValidationError) as raised:
            ProviderEvent(
                kind=ProviderEventKind.REQUEST_STARTED,
                operation=operation,
                provider_kind=ProviderKind.FAKE,
            )

        assert raised.value.details["field"] == "operation"

    def test_a_negative_chunk_index_is_refused(self) -> None:
        with pytest.raises(ValidationError) as raised:
            ProviderEvent(
                kind=ProviderEventKind.CHUNK_RECEIVED,
                operation="stream",
                provider_kind=ProviderKind.FAKE,
                chunk_index=-1,
            )

        assert raised.value.details["field"] == "chunk_index"

    def test_unmeasured_fields_are_unsupported_rather_than_zero(self) -> None:
        """ADR-0016 applies to an event as much as to a result: a start has no duration yet."""
        event = ProviderEvent(
            kind=ProviderEventKind.REQUEST_STARTED,
            operation="generate",
            provider_kind=ProviderKind.FAKE,
        )

        assert event.elapsed_ms is UNSUPPORTED
        assert event.chunk_index is UNSUPPORTED
        assert event.output_tokens is UNSUPPORTED

    def test_an_unsupported_chunk_index_passes_validation(self) -> None:
        """The sentinel must not be compared against zero by the negative-index check."""
        event = ProviderEvent(
            kind=ProviderEventKind.REQUEST_COMPLETED,
            operation="generate",
            provider_kind=ProviderKind.FAKE,
            chunk_index=UNSUPPORTED,
        )

        assert is_supported(event.chunk_index) is False

    def test_an_event_carries_no_field_content_could_reach(self) -> None:
        """The enforcement is structural: there is no field a caller could put text in."""
        text_fields = {"operation", "model_name", "finish_reason", "error_code"}
        fields = {field.name for field in ProviderEvent.__dataclass_fields__.values()}

        assert fields - text_fields == {
            "kind",
            "provider_kind",
            "metadata",
            "elapsed_ms",
            "chunk_index",
            "output_tokens",
        }

    def test_an_event_is_frozen(self) -> None:
        event = ProviderEvent(
            kind=ProviderEventKind.REQUEST_STARTED,
            operation="generate",
            provider_kind=ProviderKind.FAKE,
        )

        with pytest.raises(AttributeError):
            event.operation = "stream"  # type: ignore[misc]


class TestEmit:
    def test_no_callback_is_a_no_op(self) -> None:
        emit(
            None,
            ProviderEvent(
                kind=ProviderEventKind.REQUEST_STARTED,
                operation="x",
                provider_kind=ProviderKind.FAKE,
            ),
        )

    def test_a_callback_receives_the_event(self) -> None:
        recorder = _Recorder()
        event = ProviderEvent(
            kind=ProviderEventKind.REQUEST_STARTED, operation="x", provider_kind=ProviderKind.FAKE
        )

        emit(recorder, event)

        assert recorder.events == [event]

    def test_a_raising_callback_does_not_propagate(self) -> None:
        def explode(_event: ProviderEvent) -> None:
            raise RuntimeError("the host's metrics hook has a bug")

        emit(
            explode,
            ProviderEvent(
                kind=ProviderEventKind.REQUEST_STARTED,
                operation="x",
                provider_kind=ProviderKind.FAKE,
            ),
        )

    def test_a_raising_callback_is_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Not silently dropped: a host whose callback is failing can find out."""

        def explode(_event: ProviderEvent) -> None:
            raise RuntimeError("boom")

        with caplog.at_level(logging.DEBUG, logger="modelrack.events"):
            emit(
                explode,
                ProviderEvent(
                    kind=ProviderEventKind.REQUEST_FAILED,
                    operation="stream",
                    provider_kind=ProviderKind.FAKE,
                ),
            )

        assert any(record.message == "modelrack.event.callback_failed" for record in caplog.records)
        assert all(record.levelno == logging.DEBUG for record in caplog.records)


class TestEventEmitter:
    def test_an_unobserved_emitter_says_so(self) -> None:
        assert EventEmitter(None, provider_kind=ProviderKind.FAKE).is_observed is False

    def test_an_observed_emitter_says_so(self) -> None:
        assert EventEmitter(_Recorder(), provider_kind=ProviderKind.FAKE).is_observed is True

    def test_every_helper_is_a_no_op_without_a_callback(self) -> None:
        emitter = EventEmitter(None, provider_kind=ProviderKind.FAKE)

        emitter.started(operation="generate", model_name="m", metadata={})
        emitter.chunk(operation="stream", model_name="m", metadata={}, chunk_index=0)
        emitter.completed(operation="generate", model_name="m", metadata={})
        emitter.failed(operation="generate", model_name="m", metadata={}, error_code="X")

    def test_the_provider_kind_is_stamped_on_every_event(self) -> None:
        recorder = _Recorder()
        emitter = EventEmitter(recorder, provider_kind=ProviderKind.OLLAMA)

        emitter.started(operation="generate", model_name="m", metadata={})
        emitter.completed(operation="generate", model_name="m", metadata={})

        assert {event.provider_kind for event in recorder.events} == {ProviderKind.OLLAMA}


class TestFakeProviderEvents:
    def test_a_generation_reports_started_then_completed(self) -> None:
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        identity = provider.resolve("fake-model:8b-q8_0")

        provider.generate(_request(identity=identity))

        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_COMPLETED,
        ]

    def test_a_completion_reports_the_finish_reason_and_token_count(self) -> None:
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        identity = provider.resolve("fake-model:8b-q8_0")

        result = provider.generate(_request(identity=identity))

        completed = recorder.of(ProviderEventKind.REQUEST_COMPLETED)[0]
        assert completed.finish_reason == result.finish_reason.value
        assert completed.output_tokens == result.usage.tokens.output_tokens

    def test_the_callers_correlation_metadata_is_passed_through_unread(self) -> None:
        """The mapping that is never sent to the provider is exactly what a host joins on."""
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        identity = provider.resolve("fake-model:8b-q8_0")

        provider.generate(_request(identity=identity))

        assert all(
            event.metadata["run_id"] == "01JABCDEF0123456789ABCDEFG" for event in recorder.events
        )

    def test_a_stream_reports_a_chunk_per_delta_between_the_terminal_pair(self) -> None:
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        identity = provider.resolve("fake-model:8b-q8_0")

        events = list(provider.stream(_request(identity=identity)))

        deltas = sum(1 for event in events if not isinstance(event, StreamCompleted))
        assert recorder.kinds()[0] is ProviderEventKind.REQUEST_STARTED
        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_COMPLETED
        assert len(recorder.of(ProviderEventKind.CHUNK_RECEIVED)) == deltas

    def test_chunk_indices_are_the_delta_indices_they_describe(self) -> None:
        """One event per delta, in order, carrying that delta's own position in the stream — so a
        host measuring inter-chunk latency is measuring the gaps it thinks it is.
        """
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        identity = provider.resolve("fake-model:8b-q8_0")

        events = list(provider.stream(_request(identity=identity)))

        delta_indices = [
            event.index for event in events if not isinstance(event, StreamCompleted | StreamFailed)
        ]
        chunk_indices = [
            event.chunk_index for event in recorder.of(ProviderEventKind.CHUNK_RECEIVED)
        ]
        assert chunk_indices == delta_indices

    def test_a_failed_generation_reports_the_error_code_and_no_completion(self) -> None:
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        unknown = ModelIdentity(ProviderKind.FAKE, "no-such-model:v0")

        with pytest.raises(Exception, match="No model named"):
            provider.generate(_request(identity=unknown))

        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_FAILED,
        ]
        assert recorder.of(ProviderEventKind.REQUEST_FAILED)[0].error_code == "MODEL_NOT_FOUND"

    def test_a_cancelled_stream_reports_a_failure_not_a_completion(self) -> None:
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        identity = provider.resolve("fake-model:8b-q8_0")
        token = CancellationToken()
        token.cancel()

        list(provider.stream(_request(identity=identity, cancel=token)))

        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_FAILED,
        ]
        assert recorder.of(ProviderEventKind.REQUEST_FAILED)[0].error_code == "GENERATION_CANCELLED"

    def test_a_stream_cancelled_part_way_reports_a_failure_after_its_chunks(self) -> None:
        """The mid-stream path, distinct from a token already set before the call: chunks are
        reported first, and the terminal event is still a failure rather than a completion.
        """
        recorder = _Recorder()
        provider = FakeProvider(on_event=recorder)
        identity = provider.resolve("fake-model:8b-q8_0")
        token = CancellationToken()

        for event in provider.stream(_request(identity=identity, cancel=token)):
            if isinstance(event, TokenDelta) and event.index == 1:
                token.cancel()

        assert recorder.of(ProviderEventKind.CHUNK_RECEIVED)
        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_FAILED
        assert recorder.of(ProviderEventKind.REQUEST_FAILED)[0].error_code == "GENERATION_CANCELLED"

    def test_a_stream_that_fails_before_it_starts_reports_a_failure(self) -> None:
        """A failure scripted with no chunk offset raises rather than streaming — and still says
        so through the event channel, so a host's "started" is never left unmatched.
        """
        recorder = _Recorder()
        provider = FakeProvider(
            FakeScript(
                generations=(
                    FakeGeneration(failure=FakeFailure(mode=FakeFailureMode.UNAVAILABLE)),
                ),
            ),
            on_event=recorder,
        )
        identity = ModelIdentity(ProviderKind.FAKE, "fake-model:8b-q8_0")

        with pytest.raises(Exception, match="."):
            list(provider.stream(_request(identity=identity)))

        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_FAILED,
        ]

    def test_a_raising_callback_does_not_break_the_generation(self) -> None:
        def explode(_event: ProviderEvent) -> None:
            raise RuntimeError("the host's metrics hook has a bug")

        provider = FakeProvider(on_event=explode)
        identity = provider.resolve("fake-model:8b-q8_0")

        result = provider.generate(_request(identity=identity))

        assert result.text

    def test_a_raising_callback_does_not_break_a_stream(self) -> None:
        def explode(_event: ProviderEvent) -> None:
            raise RuntimeError("boom")

        provider = FakeProvider(on_event=explode)
        identity = provider.resolve("fake-model:8b-q8_0")

        events = list(provider.stream(_request(identity=identity)))

        assert isinstance(events[-1], StreamCompleted)

    def test_no_event_quotes_the_prompt_or_the_generated_text(self) -> None:
        recorder = _Recorder()
        provider = FakeProvider(
            FakeScript(generations=(FakeGeneration(text="zebrafish absolutely everywhere"),)),
            on_event=recorder,
        )
        identity = provider.resolve("fake-model:8b-q8_0")

        result = provider.generate(_request(identity=identity))

        rendered = " ".join(repr(event) for event in recorder.events)
        assert "zebrafish" not in rendered
        assert result.text not in rendered


class TestOllamaProviderEvents:
    @respx.mock
    def test_a_generation_reports_started_then_completed(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        )
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        provider.generate(_request())

        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_COMPLETED,
        ]

    @respx.mock
    def test_a_stream_reports_chunks_between_the_terminal_pair(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=load_ollama_fixture("chat_stream.ndjson"))
        )
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        list(provider.stream(_request()))

        assert recorder.kinds()[0] is ProviderEventKind.REQUEST_STARTED
        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_COMPLETED
        assert recorder.of(ProviderEventKind.CHUNK_RECEIVED)

    @respx.mock
    def test_an_unreachable_provider_reports_a_failure(self) -> None:
        respx.post(f"{_OLLAMA_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        with pytest.raises(Exception, match="Cannot reach"):
            provider.generate(_request())

        assert recorder.of(ProviderEventKind.REQUEST_FAILED)[0].error_code == "PROVIDER_UNAVAILABLE"

    @respx.mock
    def test_a_stream_that_cannot_connect_reports_a_failure(self) -> None:
        respx.post(f"{_OLLAMA_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        with pytest.raises(Exception, match="Cannot reach"):
            list(provider.stream(_request()))

        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_FAILED,
        ]

    @respx.mock
    def test_a_rejected_stream_reports_a_failure(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(400, json=load_ollama_fixture("error_bad_option.json"))
        )
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        with pytest.raises(Exception, match="."):
            list(provider.stream(_request()))

        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_FAILED

    @respx.mock
    def test_load_and_unload_report_the_residency_change(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """Observability standards §: a model residency change is a state transition worth an
        event, which is why load and unload emit and discovery calls do not.
        """
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_empty.json"))
        )
        respx.post(f"{_OLLAMA_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_load.json"))
        )
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        provider.load(ModelIdentity(ProviderKind.OLLAMA, _MODEL), RuntimeProfile())

        assert [event.operation for event in recorder.events] == ["load", "load"]
        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_COMPLETED,
        ]

    @respx.mock
    def test_a_stream_that_fails_mid_flight_reports_a_failure(self) -> None:
        """The terminal event is a ``StreamFailed``, not a raise — and the observer sees it as a
        failure rather than as a completion with odd contents.
        """
        respx.post(f"{_OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                content=b'{"message":{"content":"tok "},"done":false}\nnot json at all\n',
            )
        )
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        list(provider.stream(_request()))

        assert recorder.of(ProviderEventKind.CHUNK_RECEIVED)
        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_FAILED
        assert (
            recorder.of(ProviderEventKind.REQUEST_FAILED)[0].error_code == "PROVIDER_PROTOCOL_ERROR"
        )

    @respx.mock
    def test_a_failed_load_reports_the_error_code(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_empty.json"))
        )
        respx.post(f"{_OLLAMA_URL}/api/generate").mock(side_effect=httpx.ConnectError("refused"))
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        with pytest.raises(Exception, match="Cannot reach"):
            provider.load(ModelIdentity(ProviderKind.OLLAMA, _MODEL), RuntimeProfile())

        assert recorder.kinds() == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_FAILED,
        ]
        assert recorder.of(ProviderEventKind.REQUEST_FAILED)[0].operation == "load"

    @respx.mock
    def test_a_failed_unload_reports_the_error_code(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )
        respx.post(f"{_OLLAMA_URL}/api/generate").mock(side_effect=httpx.ConnectError("refused"))
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        with pytest.raises(Exception, match="Cannot reach"):
            provider.unload(ModelIdentity(ProviderKind.OLLAMA, _MODEL))

        assert recorder.of(ProviderEventKind.REQUEST_FAILED)[0].operation == "unload"

    @respx.mock
    def test_a_successful_unload_reports_the_residency_change(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )
        respx.post(f"{_OLLAMA_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_unload.json"))
        )
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        provider.unload(ModelIdentity(ProviderKind.OLLAMA, _MODEL))

        assert [event.operation for event in recorder.events] == ["unload", "unload"]
        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_COMPLETED

    @respx.mock
    def test_discovery_is_not_evented(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        """Cheap, cached and carrying no run correlation — a caller wanting these logs them."""
        respx.get(f"{_OLLAMA_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )
        respx.post(f"{_OLLAMA_URL}/api/show").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("show_qwen.json"))
        )
        recorder = _Recorder()
        provider = OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder)

        provider.list_models()

        assert recorder.events == []


class TestOpenAICompatibleProviderEvents:
    @respx.mock
    def test_an_api_key_never_reaches_the_event_stream(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """Spec §14 names ``raw``, error ``details`` and the DEBUG log. The event stream is the
        fourth channel, and it is closed by construction — there is no field to put one in.
        """
        respx.post(f"{_OPENAI_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )
        recorder = _Recorder()
        provider = OpenAICompatibleProvider(
            base_url=_OPENAI_URL, api_key=_SECRET, on_event=recorder
        )

        provider.generate(_request())

        rendered = " ".join(repr(event) for event in recorder.events)
        assert _SECRET not in rendered
        assert recorder.events

    @respx.mock
    def test_a_stream_reports_chunks_between_the_terminal_pair(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_OPENAI_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=load_openai_compatible_fixture("chat_stream.sse")
            )
        )
        recorder = _Recorder()
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL, on_event=recorder)

        list(provider.stream(_request()))

        assert recorder.kinds()[0] is ProviderEventKind.REQUEST_STARTED
        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_COMPLETED
        assert recorder.of(ProviderEventKind.CHUNK_RECEIVED)

    @respx.mock
    def test_the_provider_kind_distinguishes_two_adapters_in_one_stream(
        self,
        load_ollama_fixture: Callable[[str], Any],
        load_openai_compatible_fixture: Callable[[str], Any],
    ) -> None:
        """A host running several providers tells them apart from the event, not from the object
        reference it would otherwise have to keep beside every callback.
        """
        respx.post(f"{_OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        )
        respx.post(f"{_OPENAI_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )
        recorder = _Recorder()

        OllamaProvider(base_url=_OLLAMA_URL, on_event=recorder).generate(_request())
        OpenAICompatibleProvider(base_url=_OPENAI_URL, on_event=recorder).generate(_request())

        assert {event.provider_kind for event in recorder.events} == {
            ProviderKind.OLLAMA,
            ProviderKind.OPENAI_COMPATIBLE,
        }

    @respx.mock
    def test_a_stream_that_fails_mid_flight_reports_a_failure(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_OPENAI_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=load_openai_compatible_fixture("chat_stream_malformed.sse")
            )
        )
        recorder = _Recorder()
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL, on_event=recorder)

        list(provider.stream(_request()))

        assert recorder.kinds()[-1] is ProviderEventKind.REQUEST_FAILED

    @respx.mock
    def test_the_cache_report_is_available_on_this_adapter_too(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """Spec §10 wants the cache inspectable from whichever adapter holds it."""
        respx.get(f"{_OPENAI_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        provider.list_models()
        provider.list_models()

        assert provider.metadata_cache_stats().hits == 1
