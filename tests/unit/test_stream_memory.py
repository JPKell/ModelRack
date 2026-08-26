"""Spec §15's per-stream memory budget: ≤ 1 MiB, flat regardless of response length.

**Not** in ``tests/performance/``, and the reason is worth stating. That directory is excluded
from the default run because a *wall-clock* assertion on a shared CI runner under PR load is a
flake generator — the budget is real, the measurement is not reliable. Nothing about that applies
here. What this file measures is the **live traced allocation set**, which is deterministic for a
given interpreter: the same code produces the same number to within a few hundred bytes, run after
run, on a loaded machine or an idle one.

A deterministic assertion that catches a real regression belongs in the suite that runs on every
push. It has already caught one: the accumulator both adapters shipped through Phase 4 was a
``list[str]`` of per-chunk fragments, costing roughly 49 bytes of object header per 8 bytes of
text — about 993 KB of overhead on a 20 000-chunk generation, against a 1 MiB budget for the whole
stream. Nightly would have found that eventually. Every push finds it immediately.

The development plan groups it this way too: "100 sequential streams leak no connections **and keep
memory flat**" is one Phase 5 test bullet, and the connection half already lives beside this file
in ``test_cancellation.py``.
"""

from __future__ import annotations

import tracemalloc
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from baseaicore import ModelIdentity, ProviderKind

from modelrack import GenerationRequest, Message, Role, TokenDelta
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_BASE_URL = "http://127.0.0.1:11434"
_OPENAI_URL = "http://127.0.0.1:8080"
_MODEL = "qwen3.5:9b-q8_0"

_STREAM_MEMORY_BUDGET_BYTES = 1024 * 1024
"""Spec §15: memory per active stream, "≤ 1 MiB, flat regardless of response length"."""

_LONG_STREAM_CHUNKS = 20_000
"""A long but entirely ordinary generation. Short enough to run in a second, long enough that
anything retained per chunk shows up against the budget rather than hiding under it."""


def _request(**overrides: Any) -> GenerationRequest:
    fields: dict[str, Any] = {
        "identity": ModelIdentity(ProviderKind.OLLAMA, _MODEL),
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


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


def _stream_lines(chunks: int, *, chunk_chars: int) -> list[bytes]:
    """Return one NDJSON line per network read, plus the terminal chunk.

    ``chunk_chars`` exists so two streams can be built carrying *identical total text* in
    different numbers of pieces — see :meth:`TestStreamMemory.test_nothing_is_retained_per_chunk`.
    """
    piece = ("x" * chunk_chars).encode("utf-8")
    lines = [b'{"message":{"content":"%s"},"done":false}\n' % piece for _ in range(chunks)]
    lines.append(
        b'{"message":{"content":""},"done":true,"done_reason":"stop",'
        b'"total_duration":1200000000,"eval_count":%d,"eval_duration":900000000}\n' % chunks
    )
    return lines


def _live_bytes_above_the_answer(chunks: int, *, chunk_chars: int = 8) -> int:
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
        client=httpx.Client(
            base_url=_BASE_URL,
            transport=_ChunkedTransport(_stream_lines(chunks, chunk_chars=chunk_chars)),
        ),
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


def _sse_lines(chunks: int, *, chunk_chars: int) -> list[bytes]:
    """Return one SSE frame per network read, plus the terminal ``[DONE]``."""
    piece = ("x" * chunk_chars).encode("utf-8")
    frames = [
        b'data: {"choices":[{"delta":{"content":"%s"},"index":0}]}\n\n' % piece
        for _ in range(chunks)
    ]
    frames.append(b"data: [DONE]\n\n")
    return frames


def _openai_live_bytes_above_the_answer(chunks: int, *, chunk_chars: int = 8) -> int:
    """The same measurement as :func:`_live_bytes_above_the_answer`, over SSE.

    Both adapters accumulate the answer the same way and both were changed the same way, so the
    budget is asserted against both rather than against whichever one happened to be measured.
    """
    provider = OpenAICompatibleProvider(
        base_url=_OPENAI_URL,
        client=httpx.Client(
            base_url=_OPENAI_URL,
            transport=_ChunkedTransport(_sse_lines(chunks, chunk_chars=chunk_chars)),
        ),
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

    def test_nothing_is_retained_per_chunk(self) -> None:
        """The "flat regardless of response length" half, isolated.

        Both streams carry **identical total text** — 160 000 characters — in 500 pieces and in
        20 000. Holding the text constant is what makes the comparison mean something: it cancels
        out the one thing an adapter is entitled to hold, so the only variable left is the number
        of chunks. Anything retained per chunk shows up as a 40x difference and nothing else can.

        A comparison between a short stream and a long one, which is the obvious way to write
        this, does not isolate that: the longer stream legitimately holds more text, plus the
        buffer slack that goes with it, so the threshold has to absorb a real difference and stops
        being sensitive to the fake one. Against the ``list[str]`` accumulator that shipped
        through Phase 4, this form produces a 40x signal against a 2x threshold.
        """
        coarse = _live_bytes_above_the_answer(500, chunk_chars=320)
        fine = _live_bytes_above_the_answer(_LONG_STREAM_CHUNKS, chunk_chars=8)

        assert fine <= coarse * 2, (
            f"the same {_LONG_STREAM_CHUNKS * 8} characters cost {fine} bytes of overhead in "
            f"{_LONG_STREAM_CHUNKS} chunks against {coarse} bytes in 500 — something is retained "
            "per chunk"
        )

    def test_the_openai_compatible_adapter_retains_nothing_per_frame_either(self) -> None:
        """Both adapters accumulate the answer the same way, so both are held to the budget."""
        coarse = _openai_live_bytes_above_the_answer(500, chunk_chars=320)
        fine = _openai_live_bytes_above_the_answer(_LONG_STREAM_CHUNKS, chunk_chars=8)

        assert fine <= _STREAM_MEMORY_BUDGET_BYTES
        assert fine <= coarse * 2, (
            f"the same text cost {fine} bytes of overhead over {_LONG_STREAM_CHUNKS} SSE frames "
            f"against {coarse} bytes over 500 — something is retained per frame"
        )
