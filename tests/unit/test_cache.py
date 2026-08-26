"""Tests for :mod:`modelrack.cache` and for the one use every adapter makes of it.

Two halves, deliberately. The first exercises :class:`~modelrack.cache.MetadataCache` directly:
TTL expiry against an injected clock rather than a ``sleep``, the counters spec §10 requires to be
inspectable, and ``clear()``. The second asserts the property that actually matters to a caller —
**metadata is cached and a generation never is** (spec §3) — by counting the requests each adapter
makes through a recorded transport, which is the only way to tell a cache hit from a fast miss
from outside.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import ModelIdentity, ProviderKind, ValidationError

from modelrack import GenerationRequest, Message, Role
from modelrack.cache import DEFAULT_METADATA_TTL_SECONDS, MetadataCache, MetadataSnapshot
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

_OLLAMA_URL = "http://127.0.0.1:11434"
_OPENAI_URL = "http://127.0.0.1:8080"
_MODEL = "qwen3.5:9b-q8_0"

_NANOS_PER_SECOND = 1_000_000_000


class _ManualClock:
    """A monotonic nanosecond counter a test moves by hand.

    The whole reason :class:`~modelrack.cache.MetadataCache` takes its clock as an argument: a
    five-minute TTL asserted with a real ``sleep`` is a test nobody runs, and one asserted with a
    shortened TTL proves the shortened TTL rather than the default.
    """

    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def advance_seconds(self, seconds: float) -> None:
        self.now_ns += int(seconds * _NANOS_PER_SECOND)


def _cache(clock: _ManualClock, *, ttl_seconds: float = 300.0) -> MetadataCache[str]:
    """Return a string-valued cache driven by ``clock``."""
    return MetadataCache(ttl_seconds=ttl_seconds, clock=clock)


class TestConstruction:
    def test_the_default_ttl_is_the_five_minutes_the_spec_names(self) -> None:
        assert DEFAULT_METADATA_TTL_SECONDS == 300.0
        assert MetadataCache[str]().ttl_seconds == DEFAULT_METADATA_TTL_SECONDS

    def test_a_negative_ttl_is_refused_at_construction(self) -> None:
        """A negative lifetime has no meaning; treating it as 0 would hide a caller's unit slip."""
        with pytest.raises(ValidationError) as raised:
            MetadataCache[str](ttl_seconds=-1.0)

        assert raised.value.details["field"] == "ttl_seconds"

    def test_a_zero_ttl_reports_itself_disabled(self) -> None:
        cache = MetadataCache[str](ttl_seconds=0.0)

        assert cache.is_enabled is False
        assert cache.ttl_seconds == 0.0

    def test_a_positive_ttl_reports_itself_enabled(self) -> None:
        assert MetadataCache[str](ttl_seconds=1.0).is_enabled is True


class TestHitAndMiss:
    def test_a_stored_value_is_returned(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock)

        cache.put("tags", "payload")

        assert cache.get("tags") == "payload"

    def test_an_unknown_key_misses(self) -> None:
        assert _cache(_ManualClock()).get("nothing-here") is None

    def test_hits_and_misses_are_counted_separately(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock)
        cache.put("tags", "payload")

        cache.get("tags")
        cache.get("tags")
        cache.get("absent")

        stats = cache.stats()
        assert (stats.hits, stats.misses, stats.stores) == (2, 1, 1)
        assert stats.lookups == 3

    def test_a_repeated_store_replaces_rather_than_accumulates(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock)

        cache.put("tags", "first")
        cache.put("tags", "second")

        assert cache.get("tags") == "second"
        assert len(cache) == 1


class TestExpiry:
    def test_an_entry_survives_up_to_the_ttl(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock, ttl_seconds=300.0)
        cache.put("tags", "payload")

        clock.advance_seconds(299.9)

        assert cache.get("tags") == "payload"

    def test_an_entry_is_gone_at_the_ttl(self) -> None:
        """At exactly the TTL, not after it: an entry whose lifetime has elapsed has elapsed."""
        clock = _ManualClock()
        cache = _cache(clock, ttl_seconds=300.0)
        cache.put("tags", "payload")

        clock.advance_seconds(300.0)

        assert cache.get("tags") is None

    def test_an_expiry_is_counted_as_both_an_expiry_and_a_miss(self) -> None:
        """Two counters, because "TTL too short" and "asked for a key never stored" are opposite
        problems that a single miss counter cannot tell apart.
        """
        clock = _ManualClock()
        cache = _cache(clock, ttl_seconds=10.0)
        cache.put("tags", "payload")
        clock.advance_seconds(11.0)

        cache.get("tags")
        cache.get("never-stored")

        stats = cache.stats()
        assert (stats.expirations, stats.misses) == (1, 2)

    def test_an_expired_entry_is_dropped_rather_than_left_to_accumulate(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock, ttl_seconds=10.0)
        cache.put("tags", "payload")
        clock.advance_seconds(11.0)

        cache.get("tags")

        assert len(cache) == 0

    def test_a_replaced_entry_restarts_its_lifetime(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock, ttl_seconds=10.0)
        cache.put("tags", "first")
        clock.advance_seconds(9.0)

        cache.put("tags", "second")
        clock.advance_seconds(9.0)

        assert cache.get("tags") == "second"

    def test_the_clock_is_monotonic_nanoseconds_not_a_wall_clock(self) -> None:
        """A wall-clock TTL is extended or expired by an NTP correction; a monotonic one is not."""
        readings: list[int] = []

        def counter() -> int:
            readings.append(len(readings))
            return len(readings) * _NANOS_PER_SECOND

        cache: MetadataCache[str] = MetadataCache(ttl_seconds=10.0, clock=counter)
        cache.put("tags", "payload")

        assert cache.get("tags") == "payload"
        assert readings, "the injected clock was never consulted"


class TestDisabled:
    def test_a_disabled_cache_stores_nothing(self) -> None:
        cache = MetadataCache[str](ttl_seconds=0.0)

        cache.put("tags", "payload")

        assert cache.get("tags") is None
        assert len(cache) == 0

    def test_a_disabled_cache_still_counts_what_was_asked_for(self) -> None:
        """ "No caching" spelled honestly: the counters describe the traffic, not the storage."""
        cache = MetadataCache[str](ttl_seconds=0.0)
        cache.put("tags", "payload")

        cache.get("tags")

        stats = cache.stats()
        assert (stats.hits, stats.misses, stats.stores) == (0, 1, 0)


class TestInvalidateAndClear:
    def test_invalidate_drops_one_entry_and_reports_it_was_there(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock)
        cache.put("a", "1")
        cache.put("b", "2")

        assert cache.invalidate("a") is True
        assert cache.get("a") is None
        assert cache.get("b") == "2"

    def test_invalidate_of_an_absent_key_reports_false(self) -> None:
        assert _cache(_ManualClock()).invalidate("absent") is False

    def test_clear_drops_every_entry(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock)
        cache.put("a", "1")
        cache.put("b", "2")

        cache.clear()

        assert len(cache) == 0

    def test_clear_resets_the_counters_too(self) -> None:
        """A hit rate that spanned a clear would describe two different caches."""
        clock = _ManualClock()
        cache = _cache(clock)
        cache.put("a", "1")
        cache.get("a")
        cache.get("absent")

        cache.clear()

        assert cache.stats() == type(cache.stats())()

    def test_repr_names_the_ttl_and_the_size(self) -> None:
        clock = _ManualClock()
        cache = _cache(clock, ttl_seconds=30.0)
        cache.put("a", "1")

        assert repr(cache) == "MetadataCache(ttl_seconds=30, entries=1)"


class TestMetadataSnapshot:
    def test_a_snapshot_pairs_a_payload_with_when_it_was_read(self, fixed_now: datetime) -> None:
        snapshot = MetadataSnapshot(observed_at=fixed_now, payload={"models": []})

        assert snapshot.observed_at == fixed_now
        assert snapshot.payload == {"models": []}

    def test_a_snapshot_is_frozen(self, fixed_now: datetime) -> None:
        snapshot = MetadataSnapshot(observed_at=fixed_now, payload={})

        with pytest.raises(AttributeError):
            snapshot.observed_at = fixed_now  # type: ignore[misc]


def _mock_ollama(load_ollama_fixture: Callable[[str], Any]) -> dict[str, respx.Route]:
    """Route every Ollama endpoint these tests touch, returning the routes for call counting."""
    return {
        "tags": respx.get(f"{_OLLAMA_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        ),
        "show": respx.post(f"{_OLLAMA_URL}/api/show").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("show_qwen.json"))
        ),
        "chat": respx.post(f"{_OLLAMA_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        ),
        "ps": respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_empty.json"))
        ),
    }


class TestOllamaMetadataCaching:
    """The cache seen from outside: request counts against a recorded transport."""

    @respx.mock
    def test_a_second_discovery_makes_no_further_requests(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        routes = _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        provider.list_models()
        calls_after_cold = routes["tags"].call_count + routes["show"].call_count
        provider.list_models()

        assert routes["tags"].call_count + routes["show"].call_count == calls_after_cold

    @respx.mock
    def test_a_warm_listing_is_identical_to_the_cold_one(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        cold = provider.list_models()
        warm = provider.list_models()

        assert cold == warm

    @respx.mock
    def test_the_hit_is_reported_in_the_stats(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        provider.list_models()
        assert provider.metadata_cache_stats().hits == 0
        provider.list_models()

        assert provider.metadata_cache_stats().hits > 0

    @respx.mock
    def test_refresh_bypasses_the_cache(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        """The escape hatch a TTL alone cannot provide: a re-pulled tag has a new digest now."""
        routes = _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        provider.list_models()
        before = routes["tags"].call_count
        provider.list_models(refresh=True)

        assert routes["tags"].call_count == before + 1

    @respx.mock
    def test_clearing_the_cache_makes_the_next_read_cold(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        routes = _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        provider.list_models()
        before = routes["tags"].call_count
        provider.clear_metadata_cache()
        provider.list_models()

        assert routes["tags"].call_count == before + 1
        assert provider.metadata_cache_stats().hits == 0

    @respx.mock
    def test_a_zero_ttl_adapter_caches_nothing(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        routes = _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL, metadata_ttl_seconds=0.0)

        provider.list_models()
        before = routes["tags"].call_count
        provider.list_models()

        assert routes["tags"].call_count == before + 1
        assert provider.metadata_cache_ttl_seconds == 0.0

    @respx.mock
    def test_generation_is_never_cached(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        """Spec §3. Two identical requests are two runs, and serving the first result for the
        second would fabricate a measurement FreeWeight would record as real.
        """
        routes = _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL)
        request = GenerationRequest(
            identity=ModelIdentity(ProviderKind.OLLAMA, _MODEL),
            messages=(Message(role=Role.USER, content="Explain KV caching."),),
        )

        provider.generate(request)
        provider.generate(request)

        assert routes["chat"].call_count == 2
        assert provider.metadata_cache_stats().entries == 0

    @respx.mock
    def test_residency_is_never_cached(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        """`/api/ps` is live state: a stale answer would tell a scheduler a model is loaded that
        was evicted a minute ago.
        """
        routes = _mock_ollama(load_ollama_fixture)
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        provider.list_resident()
        provider.list_resident()

        assert routes["ps"].call_count == 2

    @respx.mock
    def test_health_is_never_served_from_the_cache(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """A health probe that could answer from a five-minute-old body would report a provider
        healthy after it had stopped — the one thing health() exists to notice.
        """
        routes = _mock_ollama(load_ollama_fixture)
        respx.get(f"{_OLLAMA_URL}/api/version").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("version.json"))
        )
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        provider.health()
        provider.health()

        assert routes["tags"].call_count == 2

    @respx.mock
    def test_a_cached_descriptor_keeps_the_instant_the_provider_answered(
        self, load_ollama_fixture: Callable[[str], Any], frozen_clock: Callable[[], datetime]
    ) -> None:
        """`observed_at` means "when this snapshot was read from the provider". Stamping a cache
        hit with the current clock would report a reading that never happened at that instant.
        """
        _mock_ollama(load_ollama_fixture)
        reads = 0

        def ticking_clock() -> datetime:
            """Return a different instant on every reading, so a re-stamp cannot go unnoticed."""
            nonlocal reads
            reads += 1
            return frozen_clock() + timedelta(seconds=reads)

        provider = OllamaProvider(base_url=_OLLAMA_URL, clock=ticking_clock)

        cold = provider.list_models()
        reads_after_cold = reads
        warm = provider.list_models()

        assert warm[0].observed_at == cold[0].observed_at
        assert reads == reads_after_cold, "a cache hit read the clock, which means it re-stamped"


class TestOpenAICompatibleMetadataCaching:
    @respx.mock
    def test_a_second_listing_makes_no_further_request(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.get(f"{_OPENAI_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        provider.list_models()
        provider.list_models()

        assert route.call_count == 1

    @respx.mock
    def test_refresh_bypasses_the_cache(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.get(f"{_OPENAI_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        provider.list_models()
        provider.list_models(refresh=True)

        assert route.call_count == 2

    @respx.mock
    def test_resolve_and_inspect_share_the_one_cached_listing(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.get(f"{_OPENAI_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        identity = provider.resolve("qwen3.5-9b-instruct")
        provider.inspect_model(identity)

        assert route.call_count == 1

    @respx.mock
    def test_clearing_the_cache_makes_the_next_read_cold(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.get(f"{_OPENAI_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        provider.list_models()
        provider.clear_metadata_cache()
        provider.list_models()

        assert route.call_count == 2
        assert provider.metadata_cache_ttl_seconds == DEFAULT_METADATA_TTL_SECONDS

    @respx.mock
    def test_health_is_never_served_from_the_cache(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.get(f"{_OPENAI_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        provider.health()
        provider.health()

        assert route.call_count == 2
