"""Domain module — streamed generation events and the token that stops one.

Imports :mod:`baseaicore`, :mod:`modelrack.errors` and :mod:`modelrack.types`; performs no I/O.
:mod:`modelrack.types` imports :class:`CancellationToken` from here for typing only, which is why
that import is guarded — the runtime dependency runs one way, from this module to ``types``.

**Every stream ends with exactly one terminal event**, :class:`StreamCompleted` or
:class:`StreamFailed`, and yields nothing after it. A caller draining the iterator therefore always
learns how the stream ended, and a truncated stream — one that simply stops, which is what a
dropped connection looks like from here — is detectable as the absence of a terminal event rather
than being indistinguishable from a short answer
([spec §13](../../docs/packages/modelrack/spec.md)).

Failure is delivered as :class:`StreamFailed` rather than raised **once streaming has begun**, so a
caller that has already received deltas gets the reason through the same channel as the content,
without wrapping every iteration step in a ``try``. A failure that prevents the stream from
starting at all — a refused connection, an unknown model — still raises, because there is no stream
to terminate. The typed error is the same either way; only where it surfaces differs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from baseaicore import ValidationError

from modelrack.errors import GenerationCancelled

if TYPE_CHECKING:
    from modelrack.errors import ProviderError
    from modelrack.types import GenerationResult

__all__ = [
    "CancellationToken",
    "StreamCompleted",
    "StreamEvent",
    "StreamFailed",
    "ThinkingDelta",
    "TokenDelta",
    "ToolCallDelta",
]


class CancellationToken:
    """A caller's request to stop a streamed generation, safe to set from another thread.

    Mutable by design — it is the one place in this package's vocabulary that carries changing
    state, because its entire purpose is to be flipped after the call it governs has started.
    Backed by :class:`threading.Event`, so the thread driving the stream and the thread deciding
    to stop it need no lock of their own: a web request handler cancelling a background generation
    is the ordinary case, not an exotic one.

    Effective only through :meth:`~modelrack.provider.Provider.stream`. A non-streaming
    :meth:`~modelrack.provider.Provider.generate` has no boundary at which a token could take
    effect — the call is a single blocking round trip — which is why LoadCoach always streams and
    assembles the response itself ([spec §13](../../docs/packages/modelrack/spec.md)).

    Cancelling is one-way and idempotent: there is no ``reset``. A token that could be un-cancelled
    would let a stream resume after a caller had already been told it stopped, and every consumer
    of a cancelled generation treats that outcome as final.

    Invariants:
        * Once cancelled, always cancelled.
        * Safe to read and to set from any thread.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """Create a token that has not been cancelled."""
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested.

        Returns:
            ``True`` once :meth:`cancel` has been called, from any thread.
        """
        return self._event.is_set()

    def cancel(self) -> None:
        """Request that the generation stop. Idempotent, and safe from any thread."""
        self._event.set()

    def raise_if_cancelled(self, *, partial_text: str = "") -> None:
        """Raise :class:`~modelrack.errors.GenerationCancelled` if cancellation was requested.

        Provided here rather than left to each adapter so that every adapter preserves the partial
        output the same way. An adapter that checked the flag itself and raised a bare error would
        discard whatever the model had already produced, and a caller cancelling a long generation
        usually wants what it got.

        Args:
            partial_text: Whatever has been generated so far. Attached to the error's ``details``
                so the caller receives its own output back rather than losing it. This is the one
                error in this package whose ``details`` legitimately carries generated content.

        Raises:
            GenerationCancelled: If :meth:`cancel` has been called.
        """
        if self._event.is_set():
            raise GenerationCancelled(
                "Generation was cancelled by the caller's token.",
                details={"partial_text": partial_text},
            )

    def __repr__(self) -> str:
        """Return a representation naming the state, so a debugger session reads clearly."""
        state = "cancelled" if self._event.is_set() else "active"
        return f"CancellationToken({state})"


def _validate_index(index: int, *, owner: str) -> None:
    """Raise unless a stream sequence index is a non-negative whole number."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValidationError(
            f"{owner}.index must be a whole number; got {index!r}.",
            details={"field": "index", "value": repr(index)},
        )
    if index < 0:
        raise ValidationError(
            f"{owner}.index must not be negative; got {index}. Indices order a stream, and a "
            "negative one would sort before the beginning of it.",
            details={"field": "index", "value": index},
        )


@dataclass(frozen=True, slots=True)
class TokenDelta:
    """A fragment of generated text.

    Called a *delta* rather than a token deliberately. Whether one of these corresponds to exactly
    one model token is a property of the provider, declared as
    :attr:`~modelrack.provider.ProviderCapabilities.token_level_chunks`. When that flag is
    ``False``, the gap between two deltas is inter-chunk latency and nothing more — a caller must
    not relabel it as per-token latency ([spec §11.4](../../docs/packages/modelrack/spec.md)),
    because a provider that batches ten tokens into one chunk would otherwise appear ten times
    slower per token than it is.

    Attributes:
        text: The fragment. May legitimately be empty — some providers emit keep-alive chunks —
            and may split a multi-byte character or a grapheme cluster across two deltas, so a
            caller displaying output concatenates before decoding assumptions.
        index: This delta's position in the stream, counting from ``0`` across all delta types.
    """

    text: str
    index: int = 0

    def __post_init__(self) -> None:
        """Validate the stream index.

        Raises:
            ValidationError: If ``index`` is negative or not a whole number.
        """
        _validate_index(self.index, owner="TokenDelta")


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """A fragment of reasoning content, where the provider exposes it separately.

    Kept distinct from :class:`TokenDelta` rather than merged into the answer: reasoning is not
    part of the response, and a caller that concatenated the two would show a model's working to a
    user who asked only for its conclusion. IdeaPress and LoadCoach both surface it separately or
    not at all.

    Attributes:
        text: The fragment of reasoning content.
        index: This delta's position in the stream, counting from ``0`` across all delta types.
    """

    text: str
    index: int = 0

    def __post_init__(self) -> None:
        """Validate the stream index.

        Raises:
            ValidationError: If ``index`` is negative or not a whole number.
        """
        _validate_index(self.index, owner="ThinkingDelta")


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A fragment of a tool call being assembled across chunks.

    Providers stream tool calls in pieces: the name arrives in one chunk and the arguments across
    several more, as raw text that is not valid JSON until the last fragment lands. This type
    carries the fragments; the caller — or the adapter assembling a
    :class:`~modelrack.types.GenerationResult` — joins them.

    Attributes:
        call_index: Which tool call this fragment belongs to, counting from ``0`` within the turn.
            A model may request several calls at once, and their fragments interleave.
        id: The provider's identifier for the call, once it has sent one.
        name: The tool's name, once it has arrived.
        arguments_fragment: A piece of the argument text, to be concatenated in arrival order.
            Deliberately unparsed: a fragment is rarely valid JSON on its own, and attempting to
            parse each one would fail on every chunk but the last.
        index: This delta's position in the stream, counting from ``0`` across all delta types.
    """

    call_index: int = 0
    id: str | None = None
    name: str | None = None
    arguments_fragment: str | None = None
    index: int = 0

    def __post_init__(self) -> None:
        """Validate both indices.

        Raises:
            ValidationError: If either index is negative or not a whole number.
        """
        _validate_index(self.index, owner="ToolCallDelta")
        _validate_index(self.call_index, owner="ToolCallDelta.call_index")


@dataclass(frozen=True, slots=True)
class StreamCompleted:
    """The terminal event of a stream that finished: the assembled result.

    Carries a full :class:`~modelrack.types.GenerationResult` rather than only a finish reason, so
    a caller that streamed gets exactly what a caller that did not would have received — the same
    usage, the same timings, the same tool calls. Without it, streaming and non-streaming callers
    would need two different code paths to record one run, and FreeWeight records both.

    Attributes:
        result: The complete outcome, assembled from the deltas plus whatever the provider's
            terminal chunk reported.
    """

    result: GenerationResult


@dataclass(frozen=True, slots=True)
class StreamFailed:
    """The terminal event of a stream that began and then failed.

    Attributes:
        error: The typed failure. Never a raw transport exception
            ([spec §11.7](../../docs/packages/modelrack/spec.md)).
        partial_text: Whatever had been generated before the failure, so a caller can store it
            alongside the reason. FreeWeight keeps it: a sample that failed halfway is evidence
            about the failure, and discarding the text would lose the only record of what the
            model was doing when it stopped.
    """

    error: ProviderError
    partial_text: str = ""


type StreamEvent = TokenDelta | ThinkingDelta | ToolCallDelta | StreamCompleted | StreamFailed
"""Everything :meth:`~modelrack.provider.Provider.stream` can yield.

A closed union, so a caller can exhaustively match on it and a type checker will say when a new
member makes that match incomplete. Adding a member is a breaking change to every consumer that
matches exhaustively, which is why the set is small and each member earns its place.
"""
