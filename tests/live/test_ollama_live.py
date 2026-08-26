"""Live smoke tests: real discovery, generation, streaming and unload against a real Ollama.

Marked ``@pytest.mark.live`` and deselected by default (``addopts = "-m 'not live and not
performance'"``) — the suite's default run has no GPU, no Ollama and no network, and these tests
are the one place that premise is deliberately broken. Run manually or in a nightly job on a
machine with the hardware:

.. code-block:: bash

    pytest -m live

These assert **shape and plausibility**, never exact content
(testing standards §3): a response arrived, token
counts are real numbers, backend timings are non-negative and roughly self-consistent, streaming
reassembles to what a blocking call would have produced. What they must not do is assert on the
literal text a model generated — a real model is nondeterministic and this suite never treats one
as the oracle.

Unreachable is a **skip** by default, not a failure — a live test failing because nobody started
Ollama on this machine is a false alarm, not a defect. Set ``MODELRACK_REQUIRE_OLLAMA=1`` to turn
that skip into a failure instead, the same escape hatch
``WEIGHTSDB_REQUIRE_POSTGRES`` gives WeightsDB's own conditionally-skipped dialect tests
(testing standards §2): a silently skipped provider is an untested provider, and CI's nightly
hardware job sets it so a broken Ollama integration cannot hide behind "well, it just skipped".
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from baseaicore import RuntimeProfile, is_supported

from modelrack import (
    FinishReason,
    GenerationRequest,
    Message,
    ProviderEvent,
    ProviderEventKind,
    ProviderStatus,
    Role,
    StreamCompleted,
    StreamFailed,
    TokenDelta,
    find_resident,
    is_resident,
    residency_support,
)
from modelrack.providers.ollama import OllamaProvider
from modelrack.streaming import CancellationToken

if TYPE_CHECKING:
    from baseaicore import ModelDescriptor

pytestmark = pytest.mark.live

_BASE_URL = os.environ.get("MODELRACK_OLLAMA_URL", "http://127.0.0.1:11434")
_REQUIRE_OLLAMA = os.environ.get("MODELRACK_REQUIRE_OLLAMA") == "1"
_LIVE_PROMPT = "Name one thing a KV cache stores."


def _skip_or_fail(reason: str) -> None:
    """Skip, unless ``MODELRACK_REQUIRE_OLLAMA=1`` says an unreachable provider must fail loudly."""
    if _REQUIRE_OLLAMA:
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def provider() -> OllamaProvider:
    """Return a provider pointed at a real Ollama, skipping the module if it cannot be reached."""
    instance = OllamaProvider(base_url=_BASE_URL)
    health = instance.health()
    if health.status is not ProviderStatus.OK:
        _skip_or_fail(
            f"No reachable Ollama at {_BASE_URL} (status={health.status.value}). Start one, or "
            f"set MODELRACK_OLLAMA_URL to point elsewhere. Set MODELRACK_REQUIRE_OLLAMA=1 to "
            f"make this a failure instead of a skip."
        )
    return instance


@pytest.fixture(scope="module")
def a_model(provider: OllamaProvider) -> ModelDescriptor:
    """Return the first model this Ollama is serving, skipping if nothing is pulled."""
    descriptors = provider.list_models()
    if not descriptors:
        _skip_or_fail(
            f"Ollama at {_BASE_URL} is reachable but has no models pulled. Run "
            f"`ollama pull qwen3.5:9b-q8_0` (or any small model) and retry."
        )
    return descriptors[0]


class TestDiscovery:
    def test_health_reports_ok_with_a_version_and_a_model_count(
        self, provider: OllamaProvider
    ) -> None:
        health = provider.health()

        assert health.status is ProviderStatus.OK
        assert health.provider_version
        assert is_supported(health.model_count)
        assert health.model_count >= 1

    def test_list_models_describes_at_least_the_one_model_pulled(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        descriptors = provider.list_models()

        assert any(
            d.identity.provider_model_name == a_model.identity.provider_model_name
            for d in descriptors
        )

    def test_resolve_round_trips_the_models_own_name(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        identity = provider.resolve(a_model.identity.provider_model_name)

        assert identity.provider_model_name == a_model.identity.provider_model_name

    def test_inspect_model_returns_consistent_metadata(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        descriptor = provider.inspect_model(a_model.identity)

        assert descriptor.identity.provider_model_name == a_model.identity.provider_model_name
        assert descriptor.observed_at.tzinfo is not None


class TestGeneration:
    def test_generate_produces_plausible_text_and_timings(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        request = GenerationRequest(
            identity=a_model.identity,
            messages=(Message(role=Role.USER, content="Say hello in one short sentence."),),
        )

        result = provider.generate(request)

        assert result.text.strip()
        assert result.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
        assert is_supported(result.timing.client_wall_ms)
        assert result.timing.client_wall_ms >= 0
        if is_supported(result.timing.backend_total_ms):
            assert result.timing.backend_total_ms >= 0
        if is_supported(result.usage.tokens.output_tokens):
            assert result.usage.tokens.output_tokens > 0

    def test_stream_reassembles_to_a_plausible_result(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        request = GenerationRequest(
            identity=a_model.identity,
            messages=(Message(role=Role.USER, content="Count from one to three."),),
        )

        events = list(provider.stream(request))
        terminal = events[-1]
        deltas = [e for e in events if isinstance(e, TokenDelta)]

        assert isinstance(terminal, StreamCompleted)
        assert "".join(d.text for d in deltas) == terminal.result.text
        assert terminal.result.text.strip()
        assert is_supported(terminal.result.timing.client_ttft_ms)
        assert terminal.result.timing.client_ttft_ms <= terminal.result.timing.client_wall_ms

    def test_cancelling_a_real_stream_reports_cancellation(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        """No wall-clock assertion here on purpose.

        How long "one more chunk" takes depends entirely on this machine's own token rate — slow
        on a CPU fallback, fast on a GPU — and the *precise* "within one delta" claim already has
        a deterministic proof against a recorded transport (``tests/unit/test_ollama_adapter.py``,
        ``TestStreaming::test_cancelling_mid_stream_stops_within_one_delta``). What only a real
        server can prove is that cancellation is honoured *at all* over an actual socket: the
        stream stops and the caller is told why, rather than running to completion regardless.
        """
        request = GenerationRequest(
            identity=a_model.identity,
            messages=(Message(role=Role.USER, content="Write a very long story about a river."),),
            cancel=(token := CancellationToken()),
        )

        events = []
        for event in provider.stream(request):
            events.append(event)
            if isinstance(event, TokenDelta):
                token.cancel()

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert terminal.error.code == "GENERATION_CANCELLED"


class TestResidency:
    def test_load_then_unload_round_trips_residency(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        loaded = provider.load(a_model.identity, RuntimeProfile())
        assert loaded.identity.provider_model_name == a_model.identity.provider_model_name

        resident = provider.list_resident()
        assert any(
            entry.identity.provider_model_name == a_model.identity.provider_model_name
            for entry in resident
        )

        assert provider.unload(a_model.identity) is True

        resident_after = provider.list_resident()
        assert not any(
            entry.identity.provider_model_name == a_model.identity.provider_model_name
            for entry in resident_after
        )

    def test_residency_helpers_read_a_real_report(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        """The helpers LoadCoach's scheduler branches on, against a real ``/api/ps`` rather than
        a fixture — name matching has to hold when the identity a caller holds is digest-confident
        and the provider's report is not (ADR-0024 §2).
        """
        assert residency_support(provider.capabilities()).is_manageable is True
        provider.load(a_model.identity, RuntimeProfile())

        assert is_resident(provider.list_resident(), a_model.identity) is True
        entry = find_resident(provider.list_resident(), a_model.identity)
        assert entry is not None
        assert is_supported(entry.vram_bytes)

        provider.unload(a_model.identity)
        assert is_resident(provider.list_resident(), a_model.identity) is False


class TestMetadataCache:
    def test_a_warm_discovery_reuses_what_a_cold_one_read(self, provider: OllamaProvider) -> None:
        """Spec §10 against a real server: the second listing is identical and, crucially, keeps
        the instant the *first* one was read rather than re-stamping itself as fresh.
        """
        cold = provider.list_models(refresh=True)
        warm = provider.list_models()

        assert [descriptor.identity for descriptor in warm] == [
            descriptor.identity for descriptor in cold
        ]
        assert [descriptor.observed_at for descriptor in warm] == [
            descriptor.observed_at for descriptor in cold
        ]
        assert provider.metadata_cache_stats().hits > 0

    def test_clearing_makes_the_next_read_cold(self, provider: OllamaProvider) -> None:
        provider.list_models()
        provider.clear_metadata_cache()

        assert provider.metadata_cache_stats().entries == 0
        assert provider.list_models()


class TestObservability:
    def test_a_real_stream_reports_started_chunks_and_completed(
        self, provider: OllamaProvider, a_model: ModelDescriptor
    ) -> None:
        """Spec §17 against a real model: the event stream a host would log from, with no prompt
        and no generated text anywhere in it.
        """
        seen: list[ProviderEvent] = []
        observed = OllamaProvider(base_url=_BASE_URL, on_event=seen.append)
        request = GenerationRequest(
            identity=a_model.identity,
            messages=(Message(role=Role.USER, content=_LIVE_PROMPT),),
            metadata={"run_id": "live-smoke"},
        )

        list(observed.stream(request))

        kinds = [event.kind for event in seen]
        assert kinds[0] is ProviderEventKind.REQUEST_STARTED
        assert kinds[-1] is ProviderEventKind.REQUEST_COMPLETED
        assert ProviderEventKind.CHUNK_RECEIVED in kinds
        assert all(event.metadata["run_id"] == "live-smoke" for event in seen)
        assert _LIVE_PROMPT not in " ".join(repr(event) for event in seen)
