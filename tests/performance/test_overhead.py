"""Spec §15's overhead budgets, measured against a transport that costs nothing.

Marked ``performance`` and excluded from the default run (``addopts``), because a wall-clock
assertion on shared CI hardware is a flake generator, not a gate. Run deliberately:

    pytest -m performance

**What is being measured.** Spec §15 budgets *ModelRack's own* overhead — "excluding provider
time". A budget measured against a real provider would be measuring the model. So every test here
runs against a transport that returns a pre-built body with no I/O and no sleeping, which makes
everything the clock sees this package's own work: request construction, JSON encoding, response
parsing, normalization, event dispatch.

**Why the budgets are divided by a repeat count rather than asserted per call.** A single
sub-millisecond measurement on a machine with other tenants is noise. Each test runs the operation
many times and asserts the *mean*, which is the figure spec §15 is actually stating, and reports
the observed value in the failure message so a regression says how far it missed by.
"""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from baseaicore import ModelIdentity, ProviderKind, monotonic_ns

from modelrack import (
    GenerationRequest,
    Message,
    ProviderEvent,
    Role,
    TokenDelta,
)
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

pytestmark = pytest.mark.performance

_BASE_URL = "http://127.0.0.1:11434"
_OPENAI_URL = "http://127.0.0.1:8080"
_MODEL = "qwen3.5:9b-q8_0"

# Spec §15, verbatim.
_REQUEST_BUDGET_MS = 5.0
_CHUNK_BUDGET_MS = 1.0
_WARM_DISCOVERY_BUDGET_MS = 10.0
_STREAM_MEMORY_BUDGET_BYTES = 1024 * 1024
_LONG_STREAM_CHUNKS = 20_000

_REQUEST_REPEATS = 200
_STREAM_REPEATS = 20
_DISCOVERY_REPEATS = 100
_CHUNKS_PER_STREAM = 200
_MODEL_COUNT = 20


class _StaticTransport(httpx.BaseTransport):
    """Answer every request with a pre-built body, doing no I/O and no sleeping.

    The point of the exercise: whatever the clock measures around this is ModelRack's own work,
    which is exactly what spec §15's "excluding provider time" means.
    """

    def __init__(self, content: bytes) -> None:
        self._content = content

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=self._content)


class _RoutedTransport(httpx.BaseTransport):
    """Answer each path with its own pre-built body, for the multi-endpoint discovery budget."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self._bodies = bodies

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=self._bodies[request.url.path])


def _provider(transport: httpx.BaseTransport, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(
        base_url=_BASE_URL,
        client=httpx.Client(base_url=_BASE_URL, transport=transport),
        **kwargs,
    )


def _request(**overrides: Any) -> GenerationRequest:
    fields: dict[str, Any] = {
        "identity": ModelIdentity(ProviderKind.OLLAMA, _MODEL),
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def _complete_body() -> bytes:
    return (
        b'{"model":"qwen3.5:9b-q8_0","created_at":"2026-08-23T14:10:00.000000000Z",'
        b'"message":{"role":"assistant","content":"KV caching stores attention keys."},'
        b'"done":true,"done_reason":"stop","total_duration":1200000000,'
        b'"load_duration":100000000,"prompt_eval_count":18,'
        b'"prompt_eval_duration":200000000,"eval_count":42,"eval_duration":900000000}'
    )


def _stream_body(chunks: int) -> bytes:
    lines = [b'{"message":{"content":"tok%d "},"done":false}\n' % index for index in range(chunks)]
    lines.append(
        b'{"message":{"content":""},"done":true,"done_reason":"stop",'
        b'"total_duration":1200000000,"eval_count":%d,"eval_duration":900000000}\n' % chunks
    )
    return b"".join(lines)


def _tags_body(count: int) -> bytes:
    entries = ",".join(
        f'{{"name":"model-{index}:8b-q8_0","model":"model-{index}:8b-q8_0",'
        f'"size":9000000000,"digest":"{"a" * 64}"}}'
        for index in range(count)
    )
    return f'{{"models":[{entries}]}}'.encode()


def _show_body() -> bytes:
    return (
        b'{"details":{"family":"qwen3","parameter_size":"9.0B","quantization_level":"Q8_0"},'
        b'"model_info":{"general.architecture":"qwen3","qwen3.context_length":32768,'
        b'"qwen3.block_count":48,"qwen3.attention.head_count":40,'
        b'"qwen3.attention.head_count_kv":8,"qwen3.embedding_length":5120}}'
    )


class _ChunkedStream(httpx.SyncByteStream):
    """Deliver a body as the caller listed it, one chunk per network read."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


class _ChunkedTransport(httpx.BaseTransport):
    """Answer every request with a body delivered in realistically-sized pieces.

    Distinct from :class:`_StaticTransport`, and the distinction is the whole reason this class
    exists. A transport that hands ``httpx`` the entire body in one read makes ``httpx``'s own
    ``LineDecoder`` return every line of it as one list — nearly two megabytes of *its* memory on
    a long stream, none of it this package's, and none of it a state a real socket ever produces.
    Measuring an adapter's per-stream memory against that would be measuring the test harness.
    """

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = chunks

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_ChunkedStream(self._chunks))


def _stream_lines(chunks: int) -> list[bytes]:
    """Return one NDJSON line per network read, plus the terminal chunk."""
    lines = [b'{"message":{"content":"tok%d "},"done":false}\n' % index for index in range(chunks)]
    lines.append(
        b'{"message":{"content":""},"done":true,"done_reason":"stop",'
        b'"total_duration":1200000000,"eval_count":%d,"eval_duration":900000000}\n' % chunks
    )
    return lines


def _live_bytes_above_the_answer(chunks: int) -> int:
    """Return traced bytes alive at the last delta of one stream, less the answer so far.

    **Live, not peak.** ``tracemalloc``'s peak counts every transient allocation the loop made —
    one parsed payload, one delta and one index per chunk, all unreachable the instant the next
    iteration begins — so peak grows with the number of chunks on any implementation, correct or
    not, and would fail a package that holds nothing at all. What spec §15 budgets is what an
    *active stream* is holding, which is the live set.

    The answer text is subtracted because an adapter legitimately holds it once: that is what
    ``StreamCompleted.result.text`` will be made of. What must not scale is anything else.
    """
    provider = OllamaProvider(
        base_url=_BASE_URL,
        client=httpx.Client(base_url=_BASE_URL, transport=_ChunkedTransport(_stream_lines(chunks))),
    )
    request = _request()
    for _ in provider.stream(request):  # warm every code path before measuring
        pass

    tracemalloc.start()
    try:
        live_bytes = 0
        answer_bytes = 0
        seen = 0
        for event in provider.stream(request):
            if not isinstance(event, TokenDelta):
                continue
            seen += 1
            answer_bytes += len(event.text.encode("utf-8"))
            if seen == chunks:
                snapshot = tracemalloc.take_snapshot()
                live_bytes = sum(stat.size for stat in snapshot.statistics("filename"))
    finally:
        tracemalloc.stop()

    assert seen == chunks, "the stream produced fewer deltas than the body carried"
    return live_bytes - answer_bytes


def _sse_lines(chunks: int) -> list[bytes]:
    """Return one SSE frame per network read, plus the terminal ``[DONE]``."""
    frames = [
        b'data: {"choices":[{"delta":{"content":"tok%d "},"index":0}]}\n\n' % index
        for index in range(chunks)
    ]
    frames.append(b"data: [DONE]\n\n")
    return frames


def _openai_live_bytes_above_the_answer(chunks: int) -> int:
    """The same measurement as :func:`_live_bytes_above_the_answer`, over SSE.

    Both adapters accumulate the answer the same way and both were changed the same way, so the
    budget is asserted against both rather than against whichever one happened to be measured.
    """
    provider = OpenAICompatibleProvider(
        base_url=_OPENAI_URL,
        client=httpx.Client(base_url=_OPENAI_URL, transport=_ChunkedTransport(_sse_lines(chunks))),
    )
    request = _request()
    for _ in provider.stream(request):
        pass

    tracemalloc.start()
    try:
        live_bytes = 0
        answer_bytes = 0
        seen = 0
        for event in provider.stream(request):
            if not isinstance(event, TokenDelta):
                continue
            seen += 1
            answer_bytes += len(event.text.encode("utf-8"))
            if seen == chunks:
                snapshot = tracemalloc.take_snapshot()
                live_bytes = sum(stat.size for stat in snapshot.statistics("filename"))
    finally:
        tracemalloc.stop()

    assert seen == chunks, "the stream produced fewer deltas than the body carried"
    return live_bytes - answer_bytes


def _mean_ms(durations_ns: Sequence[int]) -> float:
    return sum(durations_ns) / len(durations_ns) / 1_000_000


class TestNonStreamingRequestOverhead:
    def test_a_generation_costs_less_than_the_budget(self) -> None:
        provider = _provider(_StaticTransport(_complete_body()))
        request = _request()
        provider.generate(request)  # warm the client and the code paths

        start_ns = monotonic_ns()
        for _ in range(_REQUEST_REPEATS):
            provider.generate(request)
        mean_ms = (monotonic_ns() - start_ns) / _REQUEST_REPEATS / 1_000_000

        assert mean_ms <= _REQUEST_BUDGET_MS, (
            f"spec §15 budgets {_REQUEST_BUDGET_MS} ms per non-streaming request; "
            f"observed {mean_ms:.3f} ms"
        )

    def test_an_observed_generation_stays_within_the_same_budget(self) -> None:
        """The ``on_event`` hook is optional but not free-to-forget: a host that turns it on must
        not fall out of the budget the spec states without one.
        """
        seen: list[ProviderEvent] = []
        provider = _provider(_StaticTransport(_complete_body()), on_event=seen.append)
        request = _request()
        provider.generate(request)

        start_ns = monotonic_ns()
        for _ in range(_REQUEST_REPEATS):
            provider.generate(request)
        mean_ms = (monotonic_ns() - start_ns) / _REQUEST_REPEATS / 1_000_000

        assert seen, "the callback was never called, so this measured nothing"
        assert mean_ms <= _REQUEST_BUDGET_MS, (
            f"spec §15 budgets {_REQUEST_BUDGET_MS} ms per request; observed {mean_ms:.3f} ms "
            "with an event callback attached"
        )


class TestStreamedChunkOverhead:
    def test_a_streamed_chunk_costs_less_than_the_budget(self) -> None:
        provider = _provider(_StaticTransport(_stream_body(_CHUNKS_PER_STREAM)))
        request = _request()
        list(provider.stream(request))

        start_ns = monotonic_ns()
        deltas = 0
        for _ in range(_STREAM_REPEATS):
            for event in provider.stream(request):
                if isinstance(event, TokenDelta):
                    deltas += 1
        mean_ms = (monotonic_ns() - start_ns) / deltas / 1_000_000

        assert deltas == _STREAM_REPEATS * _CHUNKS_PER_STREAM
        assert mean_ms <= _CHUNK_BUDGET_MS, (
            f"spec §15 budgets {_CHUNK_BUDGET_MS} ms per streamed chunk; observed {mean_ms:.3f} ms"
        )

    def test_an_observed_stream_stays_within_the_same_budget(self) -> None:
        counter = {"events": 0}

        def count(_event: ProviderEvent) -> None:
            counter["events"] += 1

        provider = _provider(_StaticTransport(_stream_body(_CHUNKS_PER_STREAM)), on_event=count)
        request = _request()
        list(provider.stream(request))

        start_ns = monotonic_ns()
        deltas = 0
        for _ in range(_STREAM_REPEATS):
            for event in provider.stream(request):
                if isinstance(event, TokenDelta):
                    deltas += 1
        mean_ms = (monotonic_ns() - start_ns) / deltas / 1_000_000

        assert counter["events"] > deltas
        assert mean_ms <= _CHUNK_BUDGET_MS, (
            f"spec §15 budgets {_CHUNK_BUDGET_MS} ms per chunk; observed {mean_ms:.3f} ms "
            "with an event callback attached"
        )


class TestWarmDiscoveryOverhead:
    def test_a_cached_listing_costs_less_than_the_budget(self) -> None:
        """Spec §15: ``list_models`` with metadata cached, ≤ 10 ms. The cold figure it is
        contrasted with is dominated by twenty ``/api/show`` round trips, which is why the cache
        exists at all.
        """
        provider = _provider(
            _RoutedTransport({"/api/tags": _tags_body(_MODEL_COUNT), "/api/show": _show_body()})
        )
        provider.list_models()  # cold, filling the cache

        start_ns = monotonic_ns()
        for _ in range(_DISCOVERY_REPEATS):
            provider.list_models()
        mean_ms = (monotonic_ns() - start_ns) / _DISCOVERY_REPEATS / 1_000_000

        assert mean_ms <= _WARM_DISCOVERY_BUDGET_MS, (
            f"spec §15 budgets {_WARM_DISCOVERY_BUDGET_MS} ms for a cached list_models; "
            f"observed {mean_ms:.3f} ms over {_MODEL_COUNT} models"
        )

    def test_the_warm_listing_is_faster_than_the_cold_one(self) -> None:
        """The cache earns its place, stated as a comparison rather than an absolute — the
        absolute is the test above; this one fails if caching silently stopped happening.
        """
        provider = _provider(
            _RoutedTransport({"/api/tags": _tags_body(_MODEL_COUNT), "/api/show": _show_body()})
        )

        cold_start_ns = monotonic_ns()
        provider.list_models()
        cold_ns = monotonic_ns() - cold_start_ns

        warm_start_ns = monotonic_ns()
        provider.list_models()
        warm_ns = monotonic_ns() - warm_start_ns

        assert warm_ns < cold_ns, (
            f"a warm listing took {warm_ns / 1e6:.3f} ms against a cold {cold_ns / 1e6:.3f} ms — "
            "the metadata cache is not being consulted"
        )


class TestStreamMemory:
    """Spec §15: ≤ 1 MiB per active stream, *flat regardless of response length*.

    That budget and "streaming never accumulates the full response more than once" are the same
    statement seen from two sides. An adapter must hold the assembled answer — that is what
    ``StreamCompleted.result.text`` is made of — so the assertion is not that the live set is
    under 1 MiB in absolute terms, which a long answer makes impossible, but that it is under
    1 MiB **above** the answer itself, at any length.

    This test has teeth, and it earned them: the accumulator these adapters shipped through Phase
    4 was a ``list[str]`` of per-chunk fragments, which costs about 49 bytes of object header for
    every 8 bytes of text. On a 20 000-chunk generation that is roughly 993 KB of overhead against
    a 1 MiB budget — a violation on any long answer, invisible to every other test in the suite.
    """

    @pytest.mark.parametrize("chunks", [50, _LONG_STREAM_CHUNKS])
    def test_what_an_active_stream_holds_stays_within_the_budget(self, chunks: int) -> None:
        overhead_bytes = _live_bytes_above_the_answer(chunks)

        assert overhead_bytes <= _STREAM_MEMORY_BUDGET_BYTES, (
            f"spec §15 budgets {_STREAM_MEMORY_BUDGET_BYTES} bytes per active stream above the "
            f"answer itself; observed {overhead_bytes} bytes over {chunks} chunks"
        )

    def test_the_budget_is_flat_rather_than_merely_large_enough(self) -> None:
        """The other half of the sentence. A implementation that retained one small object per
        chunk would pass the absolute assertion above on a short stream and fail here, which is
        exactly the regression that shipped once already.
        """
        short_overhead = _live_bytes_above_the_answer(50)
        long_overhead = _live_bytes_above_the_answer(_LONG_STREAM_CHUNKS)

        assert long_overhead <= short_overhead * 2, (
            f"overhead grew from {short_overhead} to {long_overhead} bytes across a "
            f"{_LONG_STREAM_CHUNKS // 50}x longer stream — something is retained per chunk"
        )

    def test_the_openai_compatible_adapter_is_flat_too(self) -> None:
        """Both adapters accumulate the answer the same way, so both are held to the budget."""
        short_overhead = _openai_live_bytes_above_the_answer(50)
        long_overhead = _openai_live_bytes_above_the_answer(_LONG_STREAM_CHUNKS)

        assert long_overhead <= _STREAM_MEMORY_BUDGET_BYTES
        assert long_overhead <= short_overhead * 2, (
            f"overhead grew from {short_overhead} to {long_overhead} bytes over SSE — something "
            "is retained per frame"
        )
