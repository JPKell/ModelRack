"""Domain module — the one cache this package is allowed to have.

Imports :mod:`baseaicore` and the standard library; performs no I/O.
[Spec §3](../../docs/packages/modelrack/spec.md) forbids caching in general and then carves out
exactly one exception: *no caching beyond a documented in-memory metadata cache with an explicit
TTL and a* ``clear()``. Spec §10 adds that it is "inspectable and clearable" and "never survives
the process". This module is that carve-out and nothing more.

**Metadata only — never a generation.** A model's descriptor is a fact about what the provider is
serving, and re-deriving it costs a ``/api/show`` round trip per model (spec §15 budgets a cold
20-model discovery at seconds for exactly that reason). A *generation* is not a fact about
anything; two identical requests to the same model are two different runs, and a cache that
returned the first result for the second would fabricate a measurement FreeWeight would then
record as real. Nothing in this package puts a :class:`~modelrack.types.GenerationResult` in here,
and a test asserts it.

**Why a monotonic clock.** A TTL measured against the wall clock is extended or expired by an NTP
correction or a DST-adjacent system-time change — the entry would outlive its own expiry through
no fault of the caller's. :func:`baseaicore.monotonic_ns` cannot go backwards, so an entry expires
after the time that actually passed. The clock is injected for the same reason every clock in this
suite is: a TTL test that has to sleep for 300 seconds is a test nobody runs.

**Why the TTL is not the whole answer.** A tag such as ``qwen3.5:latest`` can be repointed at any
moment, so a cached descriptor's digest can be stale the instant after it is stored — the
development plan names this as Phase 5's likely failure mode. The TTL bounds how long that can go
unnoticed; the ``refresh=True`` argument on every adapter method that reads metadata is what lets
a caller who *knows* a model was re-pulled bypass it immediately. Both exist because neither alone
is enough.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from baseaicore import ValidationError, monotonic_ns

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime
    from typing import Any

__all__ = [
    "DEFAULT_METADATA_TTL_SECONDS",
    "CacheStats",
    "MetadataCache",
    "MetadataSnapshot",
]

DEFAULT_METADATA_TTL_SECONDS: Final[float] = 300.0
"""Spec §10's default: five minutes."""

_NANOS_PER_SECOND: Final[int] = 1_000_000_000


@dataclass(frozen=True, slots=True)
class CacheStats:
    """What a cache has done since it was created or last cleared.

    Spec §10 requires the cache to be *inspectable*, and the development plan requires "cache-hit
    reporting" — this is that report. Counters are cumulative and are reset by
    :meth:`MetadataCache.clear`, so a caller measuring one discovery pass clears first and reads
    after.

    Attributes:
        hits: Reads that found a live entry.
        misses: Reads that found nothing — never stored, or already dropped.
        expirations: Reads that found an entry whose TTL had passed. Counted **in addition to**
            the miss they also produce: a cache whose misses are all expirations is a cache whose
            TTL is too short, and one whose misses are never expirations is being asked for keys
            it was never given. Those are opposite problems and a single counter cannot tell them
            apart.
        stores: Values written.
        entries: How many live-or-expired entries are held right now. Not a rate — a size.
    """

    hits: int = 0
    misses: int = 0
    expirations: int = 0
    stores: int = 0
    entries: int = 0

    @property
    def lookups(self) -> int:
        """Total reads, hits and misses together.

        Returns:
            ``hits + misses``. Offered because the hit *rate* is what a caller actually wants and
            deriving it from two fields invites the off-by-one of forgetting expirations are
            already counted inside ``misses``.
        """
        return self.hits + self.misses


@dataclass(frozen=True, slots=True)
class MetadataSnapshot:
    """One provider payload together with the instant it was actually read.

    The pair exists because caching the payload alone would falsify
    :attr:`~baseaicore.ModelDescriptor.observed_at`, whose documented meaning is *when this
    snapshot was read from the provider*. An adapter that served a five-minute-old ``show`` body
    and stamped it with the current clock would report a reading that never happened at that
    instant — the same class of error as reporting an unmeasured value as ``0``
    (ADR-0016), and one that would quietly
    corrupt FreeWeight's freshness accounting.

    Attributes:
        observed_at: When the provider answered. Timezone-aware, UTC — it comes from the
            adapter's injected clock, not from :func:`~baseaicore.monotonic_ns`, because it is a
            point on the calendar a caller will store and compare, not a duration.
        payload: The provider's own JSON body, unmodified. ``Any``-valued because it is provider
            JSON: this is the same untouched shape that reaches
            :attr:`~baseaicore.ModelDescriptor.raw`, and narrowing it here would mean parsing it
            here, which is the adapter's job.
    """

    observed_at: datetime
    payload: Mapping[str, Any]


class MetadataCache[ValueT]:
    """An in-memory, TTL-bounded store for provider metadata, and nothing else.

    Generic over what it holds so one implementation serves both a list of descriptors and a
    single ``show`` payload without either being widened to ``Any`` at a boundary the coding
    standards forbid one at.

    Thread-safe. An adapter is a plain object a caller may share across threads — the suite's own
    web layer dispatches synchronous provider work to a threadpool — and a dict mutated from two
    threads mid-resize is the kind of failure that appears once a month in production and never in
    a test. The lock is held only around the dictionary operations, never across a callback or an
    HTTP call.

    It is deliberately **not** a single-flight cache: two threads that miss the same key at the
    same moment both fetch, and the second store wins. Holding the lock across the fetch would
    serialize every caller behind one slow round trip and make an adapter's own timeout the whole
    process's, which is a far worse failure than one duplicated ``/api/show``. What this cache
    exists to remove is the *steady-state* cost of re-reading metadata, not a thundering herd.

    Args:
        ttl_seconds: How long an entry stays live. ``0`` disables the cache entirely: every read
            misses, every write is dropped, and the counters still tell the truth about what was
            asked for — the honest way to spell "no caching" without a second code path in every
            caller.
        clock: Where "now" comes from, as a monotonic nanosecond reading. Injected so a TTL test
            can advance time instead of sleeping through it (coding standards §5).

    Raises:
        ValidationError: If ``ttl_seconds`` is negative. A negative lifetime has no meaning, and
            silently treating it as ``0`` would hide a caller's unit mistake — seconds passed
            where a negative sentinel was intended is a bug worth surfacing at construction.
    """

    __slots__ = (
        "_clock",
        "_entries",
        "_expirations",
        "_hits",
        "_lock",
        "_misses",
        "_stores",
        "_ttl_ns",
    )

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_METADATA_TTL_SECONDS,
        clock: Callable[[], int] = monotonic_ns,
    ) -> None:
        """Create an empty cache with the given lifetime."""
        if ttl_seconds < 0:
            raise ValidationError(
                f"MetadataCache ttl_seconds must not be negative; got {ttl_seconds}. Pass 0 to "
                "disable caching.",
                details={"field": "ttl_seconds", "value": ttl_seconds},
            )
        self._ttl_ns = int(ttl_seconds * _NANOS_PER_SECOND)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[int, ValueT]] = {}
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._stores = 0

    @property
    def ttl_seconds(self) -> float:
        """How long an entry stays live, in seconds.

        Returns:
            The lifetime this cache was constructed with. ``0.0`` when caching is disabled.
        """
        return self._ttl_ns / _NANOS_PER_SECOND

    @property
    def is_enabled(self) -> bool:
        """Whether this cache stores anything at all.

        Returns:
            ``False`` when the TTL is ``0``, in which case every :meth:`get` misses and every
            :meth:`put` is dropped.
        """
        return self._ttl_ns > 0

    def get(self, key: str) -> ValueT | None:
        """Return the live value stored under ``key``, or ``None``.

        Args:
            key: What the value was stored under.

        Returns:
            The value if it is present and its TTL has not passed; ``None`` otherwise. An expired
            entry is dropped on the way out rather than left to accumulate, so a long-lived
            adapter that asks for one model repeatedly does not grow a graveyard of the others.

        Note:
            ``None`` is the miss signal, which means this cache cannot hold ``None`` as a *value*.
            That is deliberate rather than an oversight: every value it exists to hold is a
            descriptor or a payload, and "the provider has no metadata for this model" is a
            :class:`~modelrack.errors.ModelNotFound`, not a cacheable answer.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_ns, value = entry
            if self._clock() - stored_ns >= self._ttl_ns:
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                return None
            self._hits += 1
            return value

    def put(self, key: str, value: ValueT) -> None:
        """Store ``value`` under ``key``, starting its TTL now.

        Args:
            key: What to store it under. An existing entry is replaced and its lifetime restarts —
                a re-read of metadata is fresher than what it replaces, and keeping the older
                entry's expiry would discard that freshness for no reason.
            value: What to store. Dropped without comment when the cache is disabled.
        """
        if self._ttl_ns <= 0:
            return
        with self._lock:
            self._entries[key] = (self._clock(), value)
            self._stores += 1

    def invalidate(self, key: str) -> bool:
        """Drop one entry.

        Args:
            key: The entry to drop.

        Returns:
            ``True`` if an entry was there — live or expired — and has been removed. Used by an
            adapter that has just learned one model's metadata changed and has no reason to
            discard the other nineteen.
        """
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Drop every entry and reset the counters.

        Spec §10's required escape hatch: a caller who has re-pulled a model, or who simply does
        not trust what is held, gets a guaranteed-cold next read. Counters reset with the
        contents, because a hit rate that spans a clear describes two different caches.
        """
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._expirations = 0
            self._stores = 0

    def stats(self) -> CacheStats:
        """Report what this cache has done.

        Returns:
            A snapshot. Taken under the lock so the counters are mutually consistent — a report
            whose ``hits`` and ``entries`` came from either side of a concurrent write would
            describe a state the cache was never in.
        """
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                expirations=self._expirations,
                stores=self._stores,
                entries=len(self._entries),
            )

    def __len__(self) -> int:
        """Return how many entries are held, live or expired.

        Counts expired-but-not-yet-read entries too: they occupy memory until something asks for
        them, and a length that pretended otherwise would understate what the process is holding.
        """
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        """Return a representation naming the TTL and the size, for a debugger session."""
        return f"MetadataCache(ttl_seconds={self.ttl_seconds:g}, entries={len(self)})"
