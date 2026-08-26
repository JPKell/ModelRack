"""Domain module — the optional observability hook, and what it is allowed to say.

Imports :mod:`baseaicore` and this package's own types; performs no I/O and holds no state
beyond the callback it was handed. [Spec §17](../../docs/packages/modelrack/spec.md) asks for
exactly one thing here: *an optional* ``on_event`` *callback (request started/completed/failed,
chunk received) lets an application emit its own structured logs and events without ModelRack
knowing about them* — the inversion that keeps a library out of the host's logging configuration
while still letting the host see what the library is doing.

**An event never carries content.** Not the prompt, not the generated text, not a tool call's
arguments, not a credential. Spec §17 permits DEBUG logs "of request shape (never content)" and
spec §14 requires an API key to be absent from ``raw``, from error ``details`` and from every log;
an event stream is the fourth channel that discipline has to hold on, and the field set below is
the enforcement — there is no field a caller could put text in. What an event does carry is the
correlation metadata the caller itself attached to the request
(:attr:`~modelrack.types.GenerationRequest.metadata`, which is never sent to the provider), so a
host can join an event to its own run without ModelRack knowing what a run is.

**A callback that raises does not break a generation.** :meth:`EventEmitter.failed` and its
siblings swallow whatever the host's callback raised and log it at DEBUG under ``modelrack.events``
instead. This is the one place in this package where an exception is deliberately not propagated,
and it is not a violation of "never swallowed" (spec §13) but its complement: that rule governs
*provider* failures, which are the caller's result. A host's own logging bug is not a provider
failure, and a completed generation destroyed by a typo in a metrics callback would be a far worse
outcome than a missing log line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from baseaicore import UNSUPPORTED, Measurement, TokenCount, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from baseaicore import ProviderKind

__all__ = [
    "EventCallback",
    "EventEmitter",
    "ProviderEvent",
    "ProviderEventKind",
    "emit",
]

logger = logging.getLogger(__name__)


class ProviderEventKind(StrEnum):
    """The four moments spec §17 names, and no others.

    A closed set, for the same reason :data:`~modelrack.streaming.StreamEvent` is a closed union:
    a host writing an exhaustive ``match`` over these should be told by its type checker when a
    new member makes that match incomplete, rather than silently dropping a kind it has never
    seen.
    """

    REQUEST_STARTED = "request_started"
    """A provider call has been prepared and is about to leave this process."""

    CHUNK_RECEIVED = "chunk_received"
    """One streamed delta has been handed to the caller. Emitted per delta, so a host can measure
    inter-chunk latency itself — but only *label* it per-token when the adapter declares
    :attr:`~modelrack.provider.ProviderCapabilities.token_level_chunks`
    ([spec §11.4](../../docs/packages/modelrack/spec.md))."""

    REQUEST_COMPLETED = "request_completed"
    """A provider call finished and produced a result."""

    REQUEST_FAILED = "request_failed"
    """A provider call ended in a typed error, including the caller's own cancellation."""


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    """One thing that happened inside an adapter, described without quoting anything.

    Attributes:
        kind: Which of the four moments this is.
        operation: The provider method that produced it — ``"generate"``, ``"stream"``,
            ``"load"`` or ``"unload"``. A plain string rather than an enum: a future adapter that
            grows an operation should not need a release of this module to report it, and nothing
            in this package branches on the value.
        provider_kind: Which sort of provider emitted it, so a host running several can tell them
            apart without holding a reference to each adapter.
        model_name: The provider-side model name the call named. A name, never a digest —
            :class:`~baseaicore.ModelIdentity` is the object that carries provenance, and an event
            is a log line, not a record of what ran.
        metadata: The caller's own correlation identifiers, copied from
            :attr:`~modelrack.types.GenerationRequest.metadata` — the mapping that is never sent
            to the provider (spec §7). This is how a host joins an event to its own run, job or
            request ID. Passed through unread; ModelRack attaches no meaning to any key.
        elapsed_ms: Milliseconds since the operation started, on a terminal event. ``UNSUPPORTED``
            on :attr:`ProviderEventKind.REQUEST_STARTED`, where no duration exists yet — never
            ``0``, which would claim an instantaneous call
            (ADR-0016).
        chunk_index: Which delta this is, on a :attr:`ProviderEventKind.CHUNK_RECEIVED`, counting
            from ``0``. ``UNSUPPORTED`` on every other kind.
        output_tokens: How many tokens the provider reported generating, on a completion.
            ``UNSUPPORTED`` when the provider does not report token counts, which is a different
            statement from "it generated none".
        finish_reason: How a generation ended, on a completion — the value of the result's
            :class:`~modelrack.types.FinishReason`, as a string so a host need not import it.
        error_code: The typed error's ``code``, on a failure. The code alone: an error *message*
            can contain a provider's echo of the prompt, and this type does not carry content.
    """

    kind: ProviderEventKind
    operation: str
    provider_kind: ProviderKind
    model_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    elapsed_ms: Measurement = UNSUPPORTED
    chunk_index: Measurement = UNSUPPORTED
    output_tokens: TokenCount = UNSUPPORTED
    finish_reason: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        """Validate the operation name and the chunk index.

        Raises:
            ValidationError: If ``operation`` is empty — an event that cannot say which call
                produced it is not usable as a log line — or if ``chunk_index`` is a negative
                number, which would sort before the beginning of the stream it orders.
        """
        if not self.operation.strip():
            raise ValidationError(
                f"ProviderEvent.operation must name the provider call; got {self.operation!r}.",
                details={"field": "operation", "value": self.operation},
            )
        if isinstance(self.chunk_index, int | float) and self.chunk_index < 0:
            raise ValidationError(
                f"ProviderEvent.chunk_index must not be negative; got {self.chunk_index}.",
                details={"field": "chunk_index", "value": self.chunk_index},
            )


type EventCallback = Callable[[ProviderEvent], None]
"""What a caller passes as ``on_event``.

Called synchronously, on the thread driving the generation, inside the streaming loop. A callback
that blocks makes the stream it is observing slower — a host that needs to do real work per event
enqueues it and returns.
"""


def emit(callback: EventCallback | None, event: ProviderEvent) -> None:
    """Deliver one event to a caller's callback, and never let it break the call it describes.

    Args:
        callback: The host's callback, or ``None`` when none was supplied — the overwhelmingly
            common case, checked first so an unobserved stream pays nothing per chunk beyond a
            null test (spec §15 budgets 1 ms per chunk for *everything* this package does).
        event: What happened.

    Returns:
        Nothing. In particular, nothing about whether the callback succeeded: a host that wants to
        know its own callback is failing reads its DEBUG log, and a provider call's outcome does
        not depend on an observer.
    """
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 — a host's observer must not destroy a provider result;
        # see this module's docstring. Logged at DEBUG rather than re-raised or dropped silently.
        logger.debug(
            "modelrack.event.callback_failed",
            exc_info=True,
            extra={"operation": event.operation, "kind": event.kind.value},
        )


class EventEmitter:
    """One provider's binding of a callback, so an adapter emits in a single line.

    Holds the two things every event from one adapter repeats — the callback and the provider
    kind — so a call site names only what differs. Constructed once per adapter instance; not
    thread-confined, because it holds nothing mutable and the callback's own thread-safety is the
    host's business.

    Args:
        callback: The host's ``on_event``, or ``None``.
        provider_kind: Stamped on every event this emitter produces.
    """

    __slots__ = ("_callback", "_provider_kind")

    def __init__(self, callback: EventCallback | None, *, provider_kind: ProviderKind) -> None:
        """Bind a callback to one provider kind."""
        self._callback = callback
        self._provider_kind = provider_kind

    @property
    def is_observed(self) -> bool:
        """Whether anything is listening.

        Returns:
            ``True`` when a callback was supplied. Read by a call site that would otherwise do
            work purely to build an event nobody will receive.
        """
        return self._callback is not None

    def started(self, *, operation: str, model_name: str, metadata: Mapping[str, Any]) -> None:
        """Emit :attr:`ProviderEventKind.REQUEST_STARTED` for one call."""
        if self._callback is None:
            return
        emit(
            self._callback,
            ProviderEvent(
                kind=ProviderEventKind.REQUEST_STARTED,
                operation=operation,
                provider_kind=self._provider_kind,
                model_name=model_name,
                metadata=metadata,
            ),
        )

    def chunk(
        self,
        *,
        operation: str,
        model_name: str,
        metadata: Mapping[str, Any],
        chunk_index: int,
        elapsed_ms: Measurement = UNSUPPORTED,
    ) -> None:
        """Emit :attr:`ProviderEventKind.CHUNK_RECEIVED` for one delta."""
        if self._callback is None:
            return
        emit(
            self._callback,
            ProviderEvent(
                kind=ProviderEventKind.CHUNK_RECEIVED,
                operation=operation,
                provider_kind=self._provider_kind,
                model_name=model_name,
                metadata=metadata,
                chunk_index=chunk_index,
                elapsed_ms=elapsed_ms,
            ),
        )

    def completed(
        self,
        *,
        operation: str,
        model_name: str,
        metadata: Mapping[str, Any],
        elapsed_ms: Measurement = UNSUPPORTED,
        output_tokens: TokenCount = UNSUPPORTED,
        finish_reason: str | None = None,
    ) -> None:
        """Emit :attr:`ProviderEventKind.REQUEST_COMPLETED` for one finished call."""
        if self._callback is None:
            return
        emit(
            self._callback,
            ProviderEvent(
                kind=ProviderEventKind.REQUEST_COMPLETED,
                operation=operation,
                provider_kind=self._provider_kind,
                model_name=model_name,
                metadata=metadata,
                elapsed_ms=elapsed_ms,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            ),
        )

    def failed(
        self,
        *,
        operation: str,
        model_name: str,
        metadata: Mapping[str, Any],
        error_code: str,
        elapsed_ms: Measurement = UNSUPPORTED,
    ) -> None:
        """Emit :attr:`ProviderEventKind.REQUEST_FAILED` for one failed call."""
        if self._callback is None:
            return
        emit(
            self._callback,
            ProviderEvent(
                kind=ProviderEventKind.REQUEST_FAILED,
                operation=operation,
                provider_kind=self._provider_kind,
                model_name=model_name,
                metadata=metadata,
                elapsed_ms=elapsed_ms,
                error_code=error_code,
            ),
        )
