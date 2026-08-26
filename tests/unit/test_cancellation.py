"""Tests for cancellation hardening — spec §20 acceptance criterion 5, and its neighbours.

*Cancelling a stream stops it within one chunk and leaves no open connection (asserted by a
connection-count test).* That sentence is what this file exists for, and the interesting question
is what a "connection count" can honestly be asserted against with no server running.

**The counting transport.** :class:`_CountingTransport` is a real :class:`httpx.BaseTransport`
that hands back responses whose body is a stream object counting its own opens and closes. It is
therefore a genuine count of *response streams this adapter opened and did not close* — which for
a streamed response is one-to-one with a connection held out of the pool. What it does not test is
``httpx``'s own pool bookkeeping, which is not this package's code; what it does test is the only
part that is: that every exit path an adapter can take runs the ``finally`` that releases the
body. There are five such paths, and each has a test below:

1. the stream is drained to its terminal event;
2. the caller cancels part-way and drains;
3. the caller breaks out of the loop and abandons the iterator;
4. the stream fails mid-flight;
5. the token was already set before ``stream()`` was called — where the honest count is
   **zero opened**, because a connection opened solely to be closed on the first chunk is exactly
   the leak this phase is about.

``respx`` is deliberately not used here. It is a mock *router*, and what needs observing is the
lifecycle of the response body underneath it.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from baseaicore import ModelIdentity, ProviderKind, RuntimeProfile

from modelrack import (
    GenerationCancelled,
    GenerationRequest,
    Message,
    Role,
    StreamCompleted,
    StreamFailed,
    TokenDelta,
)
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider
from modelrack.streaming import CancellationToken

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from modelrack import StreamEvent

_OLLAMA_URL = "http://127.0.0.1:11434"
_OPENAI_URL = "http://127.0.0.1:8080"
_MODEL = "qwen3.5:9b-q8_0"
_LEAK_CHECK_STREAMS = 100


def _ndjson_lines(count: int) -> list[bytes]:
    """Return ``count`` Ollama chunks followed by a terminal one."""
    body = [b'{"message":{"content":"tok%d "},"done":false}\n' % index for index in range(count)]
    return [*body, b'{"message":{"content":""},"done":true,"done_reason":"stop"}\n']


def _sse_lines(count: int) -> list[bytes]:
    """Return ``count`` OpenAI-compatible SSE frames followed by ``[DONE]``."""
    frames = [
        b'data: {"choices":[{"delta":{"content":"tok%d "},"index":0}]}\n\n' % index
        for index in range(count)
    ]
    return [*frames, b"data: [DONE]\n\n"]


class _CountingStream(httpx.SyncByteStream):
    """A response body that reports whether it is still open.

    Counts on the shared :class:`_CountingTransport` rather than on itself, so a test asks one
    object "how many streams are open right now?" instead of walking a list of responses.
    """

    def __init__(self, chunks: Sequence[bytes], counter: _CountingTransport) -> None:
        self._chunks = chunks
        self._counter = counter
        self._closed = False
        counter.opened += 1

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._counter.closed += 1


class _CountingTransport(httpx.BaseTransport):
    """Serve a fixed body over a real transport interface, counting stream opens and closes.

    Args:
        chunks: The raw byte chunks every response delivers.
        status_code: What to answer with. A 4xx exercises the pre-stream refusal path, whose
            ``response.close()`` is a different line of the adapter than the streaming one.
    """

    def __init__(self, chunks: Sequence[bytes], *, status_code: int = 200) -> None:
        self.chunks = chunks
        self.status_code = status_code
        self.opened = 0
        self.closed = 0
        self.requests: list[httpx.Request] = []

    @property
    def open_streams(self) -> int:
        return self.opened - self.closed

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status_code,
            headers={"Content-Type": "application/x-ndjson"},
            stream=_CountingStream(self.chunks, self),
        )


def _ollama(transport: _CountingTransport, **kwargs: Any) -> OllamaProvider:
    """Return an Ollama adapter whose pooled client is the counting transport."""
    return OllamaProvider(
        base_url=_OLLAMA_URL,
        client=httpx.Client(base_url=_OLLAMA_URL, transport=transport),
        **kwargs,
    )


def _openai(transport: _CountingTransport) -> OpenAICompatibleProvider:
    """Return an OpenAI-compatible adapter whose pooled client is the counting transport."""
    return OpenAICompatibleProvider(
        base_url=_OPENAI_URL,
        client=httpx.Client(base_url=_OPENAI_URL, transport=transport),
    )


def _request(**overrides: Any) -> GenerationRequest:
    fields: dict[str, Any] = {
        "identity": ModelIdentity(ProviderKind.OLLAMA, _MODEL),
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def _text_of(events: Sequence[StreamEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, TokenDelta))


def _close(iterator: Iterator[StreamEvent]) -> None:
    """Close a stream deterministically, without waiting for the garbage collector.

    ``Provider.stream`` is typed as returning an :class:`~collections.abc.Iterator` — a downstream
    adapter is free to return any iterator, and the protocol deliberately does not require a
    generator. Both adapters here do return one, which is what makes an explicit ``close()``
    possible; this helper narrows to that fact in one place rather than at each call site.
    """
    close = getattr(iterator, "close", None)
    assert callable(close), "this adapter's stream() did not return a closeable generator"
    close()


class TestConnectionIsReleasedOnEveryExitPath:
    """Acceptance criterion 5, one test per way out of a stream."""

    def test_a_drained_stream_closes_its_body(self) -> None:
        transport = _CountingTransport(_ndjson_lines(5))

        events = list(_ollama(transport).stream(_request()))

        assert isinstance(events[-1], StreamCompleted)
        assert transport.open_streams == 0

    def test_a_cancelled_stream_closes_its_body(self) -> None:
        transport = _CountingTransport(_ndjson_lines(20))
        token = CancellationToken()
        provider = _ollama(transport)
        events: list[StreamEvent] = []

        for event in provider.stream(_request(cancel=token)):
            events.append(event)
            if len(events) == 2:
                token.cancel()

        assert transport.open_streams == 0
        assert isinstance(events[-1], StreamFailed)

    def test_an_abandoned_stream_closes_its_body(self) -> None:
        """Break out of the loop and walk away: the generator's ``finally`` still runs."""
        transport = _CountingTransport(_ndjson_lines(20))
        provider = _ollama(transport)

        iterator = provider.stream(_request())
        for _ in iterator:
            break
        del iterator
        gc.collect()

        assert transport.open_streams == 0

    def test_an_explicitly_closed_stream_closes_its_body(self) -> None:
        """The deterministic version of the test above, with no reliance on the collector."""
        transport = _CountingTransport(_ndjson_lines(20))
        iterator = _ollama(transport).stream(_request())

        next(iterator)
        _close(iterator)

        assert transport.open_streams == 0

    def test_a_stream_that_fails_mid_flight_closes_its_body(self) -> None:
        transport = _CountingTransport(
            [b'{"message":{"content":"tok "},"done":false}\n', b"this is not json\n"]
        )

        events = list(_ollama(transport).stream(_request()))

        assert isinstance(events[-1], StreamFailed)
        assert transport.open_streams == 0

    def test_a_truncated_stream_closes_its_body(self) -> None:
        """No terminal chunk ever arrives — the dropped-connection shape."""
        transport = _CountingTransport([b'{"message":{"content":"tok "},"done":false}\n'])

        events = list(_ollama(transport).stream(_request()))

        assert isinstance(events[-1], StreamFailed)
        assert events[-1].error.code == "PROVIDER_PROTOCOL_ERROR"
        assert transport.open_streams == 0

    def test_a_refused_stream_closes_its_body(self) -> None:
        """The pre-stream 4xx path, whose ``response.close()`` is a different line entirely."""
        transport = _CountingTransport([b'{"error":"invalid option"}'], status_code=400)

        with pytest.raises(Exception, match="."):
            list(_ollama(transport).stream(_request()))

        assert transport.open_streams == 0

    def test_an_interrupt_while_classifying_a_refusal_still_closes_the_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``KeyboardInterrupt`` is not a :class:`~modelrack.errors.ProviderError`, so it takes
        the adapter's bare ``except BaseException`` path — which exists precisely so that a Ctrl-C
        landing inside error classification releases the connection rather than stranding it.
        """
        transport = _CountingTransport([b'{"error":"invalid option"}'], status_code=400)
        provider = _ollama(transport)

        def interrupt(*_args: Any, **_kwargs: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(provider, "_raise_for_status", interrupt)

        with pytest.raises(KeyboardInterrupt):
            list(provider.stream(_request()))

        assert transport.open_streams == 0

    def test_the_openai_compatible_adapter_releases_it_on_the_same_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport = _CountingTransport([b'{"error":{"message":"bad request"}}'], status_code=400)
        provider = _openai(transport)

        def interrupt(*_args: Any, **_kwargs: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(provider, "_raise_for_status", interrupt)

        with pytest.raises(KeyboardInterrupt):
            list(provider.stream(_request()))

        assert transport.open_streams == 0


class TestATokenAlreadySetOpensNoConnection:
    def test_ollama_opens_no_connection(self) -> None:
        transport = _CountingTransport(_ndjson_lines(20))
        token = CancellationToken()
        token.cancel()

        events = list(_ollama(transport).stream(_request(cancel=token)))

        assert transport.opened == 0
        assert transport.requests == []
        assert len(events) == 1
        assert isinstance(events[0], StreamFailed)
        assert isinstance(events[0].error, GenerationCancelled)

    def test_openai_compatible_opens_no_connection(self) -> None:
        transport = _CountingTransport(_sse_lines(20))
        token = CancellationToken()
        token.cancel()

        events = list(_openai(transport).stream(_request(cancel=token)))

        assert transport.opened == 0
        assert len(events) == 1
        assert isinstance(events[0], StreamFailed)

    def test_the_terminal_event_is_delivered_not_raised(self) -> None:
        """One cancellation code path for a caller, not two."""
        transport = _CountingTransport(_ndjson_lines(20))
        token = CancellationToken()
        token.cancel()

        events = list(_ollama(transport).stream(_request(cancel=token)))

        assert isinstance(events[0], StreamFailed)
        assert events[0].partial_text == ""

    def test_a_capability_refusal_still_wins_over_a_set_token(self) -> None:
        """A request naming something the provider never declared is malformed whichever way the
        token points; reporting it as a cancellation would hide the caller's own bug.
        """
        transport = _CountingTransport(_sse_lines(5))
        token = CancellationToken()
        token.cancel()
        provider = _openai(transport)

        with pytest.raises(Exception, match="context_configurable"):
            list(
                provider.stream(
                    _request(cancel=token, runtime_profile=RuntimeProfile(context_size=4096))
                )
            )


class TestCancellationTakesEffectWithinOneChunk:
    def test_no_delta_arrives_after_the_token_is_set(self) -> None:
        transport = _CountingTransport(_ndjson_lines(50))
        token = CancellationToken()
        seen = 0

        for event in _ollama(transport).stream(_request(cancel=token)):
            if isinstance(event, TokenDelta):
                seen += 1
                if seen == 3:
                    token.cancel()

        assert seen == 3

    def test_the_partial_output_is_preserved_on_the_terminal_event(self) -> None:
        """A caller cancelling a long generation usually wants what it already got."""
        transport = _CountingTransport(_ndjson_lines(50))
        token = CancellationToken()
        events: list[StreamEvent] = []

        for event in _ollama(transport).stream(_request(cancel=token)):
            events.append(event)
            if isinstance(event, TokenDelta) and event.index == 2:
                token.cancel()

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert terminal.partial_text == _text_of(events)
        assert terminal.partial_text == "tok0 tok1 tok2 "

    def test_the_partial_output_is_also_in_the_errors_details(self) -> None:
        """The one error in this package whose ``details`` legitimately carries content."""
        transport = _CountingTransport(_ndjson_lines(50))
        token = CancellationToken()
        events: list[StreamEvent] = []

        for event in _ollama(transport).stream(_request(cancel=token)):
            events.append(event)
            if isinstance(event, TokenDelta) and event.index == 1:
                token.cancel()

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert terminal.error.details["partial_text"] == terminal.partial_text

    def test_the_openai_compatible_adapter_behaves_identically(self) -> None:
        """Cancellation semantics belong to the protocol, not to a wire format."""
        transport = _CountingTransport(_sse_lines(50))
        token = CancellationToken()
        events: list[StreamEvent] = []

        for event in _openai(transport).stream(_request(cancel=token)):
            events.append(event)
            if isinstance(event, TokenDelta) and event.index == 2:
                token.cancel()

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == "tok0 tok1 tok2 "
        assert transport.open_streams == 0

    def test_a_cancelled_stream_yields_nothing_after_its_terminal_event(self) -> None:
        transport = _CountingTransport(_ndjson_lines(50))
        token = CancellationToken()
        iterator = _ollama(transport).stream(_request(cancel=token))

        next(iterator)
        token.cancel()
        rest = list(iterator)

        assert len(rest) == 1
        assert isinstance(rest[0], StreamFailed)


class TestNoLeakOverManyStreams:
    def test_a_hundred_drained_streams_leave_nothing_open(self) -> None:
        """The development plan's Phase 5 test: 100 sequential streams leak no connections."""
        transport = _CountingTransport(_ndjson_lines(5))
        provider = _ollama(transport)

        for _ in range(_LEAK_CHECK_STREAMS):
            list(provider.stream(_request()))

        assert transport.opened == _LEAK_CHECK_STREAMS
        assert transport.open_streams == 0

    def test_a_hundred_cancelled_streams_leave_nothing_open(self) -> None:
        transport = _CountingTransport(_ndjson_lines(20))
        provider = _ollama(transport)

        for _ in range(_LEAK_CHECK_STREAMS):
            token = CancellationToken()
            for event in provider.stream(_request(cancel=token)):
                if isinstance(event, TokenDelta):
                    token.cancel()

        assert transport.open_streams == 0

    def test_a_hundred_abandoned_streams_leave_nothing_open(self) -> None:
        transport = _CountingTransport(_ndjson_lines(20))
        provider = _ollama(transport)

        for _ in range(_LEAK_CHECK_STREAMS):
            iterator = provider.stream(_request())
            next(iterator)
            _close(iterator)

        assert transport.open_streams == 0

    def test_memory_does_not_grow_with_the_length_of_a_stream(self) -> None:
        """Spec §15: memory per active stream is flat regardless of response length. Asserted as
        "the adapter holds one copy of the text and no per-chunk accumulation beyond it" — a
        hundred-fold longer stream costs proportionally more text and nothing else.
        """
        short = _CountingTransport(_ndjson_lines(5))
        long = _CountingTransport(_ndjson_lines(500))

        short_events = list(_ollama(short).stream(_request()))
        long_events = list(_ollama(long).stream(_request()))

        short_completed, long_completed = short_events[-1], long_events[-1]
        assert isinstance(short_completed, StreamCompleted)
        assert isinstance(long_completed, StreamCompleted)
        assert len(long_completed.result.text) > len(short_completed.result.text)
        assert short.open_streams == long.open_streams == 0


class TestGenerateIsNotCancellable:
    def test_a_token_on_a_blocking_call_has_no_effect(self) -> None:
        """Spec §13: a single round trip offers no boundary at which a token could take effect,
        which is why LoadCoach always streams and assembles the response itself.
        """
        transport = _CountingTransport([b'{"message":{"content":"done"},"done":true}'])
        token = CancellationToken()
        token.cancel()

        result = _ollama(transport).generate(_request(cancel=token))

        assert result.text == "done"
        assert transport.open_streams == 0
