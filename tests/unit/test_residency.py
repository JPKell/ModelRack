"""Tests for :mod:`modelrack.residency` — what is loaded, and who may do something about it.

The operations themselves belong to the adapters and are exercised in their own suites; what is
proven here is everything about residency that has to be the *same* from every adapter, because
LoadCoach's scheduler branches on it:

* An adapter that cannot manage residency **refuses**, with a typed error naming the flag —
  never a silent no-op that would leave a scheduler believing it had evicted something
  (ADR-0007 rule 2, and the development plan's Phase 5 test list names this case explicitly).
* :func:`~modelrack.residency.residency_support` turns that refusal into a branch a caller takes
  *before* spending a call, which is how "providers without residency control simply skip all of
  this" (LoadCoach queue and scheduling §6) becomes one line rather than a ``try``.
* Membership is matched on the provider-side model name, which is the only field a provider's
  residency report and a caller's :class:`~baseaicore.ModelIdentity` genuinely agree on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import UNSUPPORTED, ModelIdentity, ProviderKind, RuntimeProfile

from modelrack import (
    FORCE_UNLOAD,
    RESIDENCY_QUERY,
    CapabilityUnsupported,
    ProviderCapabilities,
    ResidencySupport,
    ResidentModel,
    find_resident,
    is_resident,
    residency_support,
)
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider
from modelrack.residency import (
    refuse_force_unload,
    refuse_residency_query,
    require_force_unload,
    require_residency_query,
)
from modelrack.testing import MINIMAL_CAPABILITIES, FakeProvider, FakeScript

if TYPE_CHECKING:
    from collections.abc import Callable

_OLLAMA_URL = "http://127.0.0.1:11434"
_OPENAI_URL = "http://127.0.0.1:8080"
_MODEL = "qwen3.5:9b-q8_0"


def _identity(name: str = _MODEL) -> ModelIdentity:
    return ModelIdentity(ProviderKind.OLLAMA, name)


def _resident(name: str = _MODEL) -> ResidentModel:
    return ResidentModel(identity=_identity(name), vram_bytes=9_895_000_000)


class TestFlagNames:
    def test_the_flag_names_match_the_capability_fields_they_gate(self) -> None:
        """A typo here would be a refusal a caller could not match on: the name reaches the
        error's ``details`` and the conformance suite asserts on it.
        """
        fields = set(ProviderCapabilities.__dataclass_fields__)

        assert FORCE_UNLOAD in fields
        assert RESIDENCY_QUERY in fields

    def test_query_and_control_are_separate_powers(self) -> None:
        """A provider can plausibly report what it holds without letting anyone change it."""
        assert FORCE_UNLOAD != RESIDENCY_QUERY


class TestResidencySupport:
    def test_both_powers_declared_is_manageable(self) -> None:
        support = residency_support(ProviderCapabilities(force_unload=True, residency_query=True))

        assert support == ResidencySupport(can_query=True, can_control=True)
        assert support.is_manageable is True

    def test_neither_power_declared_is_not_manageable(self) -> None:
        support = residency_support(ProviderCapabilities())

        assert support.is_manageable is False

    @pytest.mark.parametrize(
        "capabilities",
        [
            ProviderCapabilities(residency_query=True),
            ProviderCapabilities(force_unload=True),
        ],
    )
    def test_one_power_alone_is_not_manageable(self, capabilities: ProviderCapabilities) -> None:
        """Evicting blind is guesswork; observing without acting is a report nobody can use."""
        assert residency_support(capabilities).is_manageable is False

    def test_the_default_support_claims_nothing(self) -> None:
        assert ResidencySupport() == ResidencySupport(can_query=False, can_control=False)

    def test_reading_the_declaration_makes_no_request(self) -> None:
        """No ``respx`` mock is installed, and the conftest socket guard is armed: if this
        touched the network at all, it would fail rather than pass slowly.
        """
        provider = OllamaProvider(base_url=_OLLAMA_URL)

        assert residency_support(provider.capabilities()).is_manageable is True


class TestCapabilityGates:
    def test_require_passes_when_the_flag_is_declared(self) -> None:
        require_force_unload(
            ProviderCapabilities(force_unload=True), action="evict a model on demand"
        )
        require_residency_query(ProviderCapabilities(residency_query=True))

    def test_require_force_unload_names_the_flag_it_refused_on(self) -> None:
        with pytest.raises(CapabilityUnsupported) as raised:
            require_force_unload(ProviderCapabilities(), action="evict a model on demand")

        assert raised.value.details["capability"] == FORCE_UNLOAD
        assert "evict a model on demand" in str(raised.value)

    def test_require_residency_query_names_the_flag_it_refused_on(self) -> None:
        with pytest.raises(CapabilityUnsupported) as raised:
            require_residency_query(ProviderCapabilities())

        assert raised.value.details["capability"] == RESIDENCY_QUERY

    def test_refuse_always_raises(self) -> None:
        """For an adapter whose protocol has no such endpoint under any configuration."""
        with pytest.raises(CapabilityUnsupported):
            refuse_force_unload(action="load a model on demand")
        with pytest.raises(CapabilityUnsupported):
            refuse_residency_query()

    def test_every_adapter_refuses_in_the_same_words(self) -> None:
        """A downstream test matching on the message must not pass against one adapter and fail
        against another.
        """
        fake = FakeProvider(FakeScript(capabilities=MINIMAL_CAPABILITIES))
        openai_compatible = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        with pytest.raises(CapabilityUnsupported) as from_fake:
            fake.list_resident()
        with pytest.raises(CapabilityUnsupported) as from_adapter:
            openai_compatible.list_resident()

        assert str(from_fake.value) == str(from_adapter.value)
        assert from_fake.value.details == from_adapter.value.details


class TestMembership:
    def test_a_resident_model_is_found_by_name(self) -> None:
        entry = _resident()

        assert find_resident([entry], _identity()) is entry
        assert is_resident([entry], _identity()) is True

    def test_a_model_that_is_not_loaded_is_not_found(self) -> None:
        assert find_resident([_resident()], _identity("llama3.3:70b")) is None
        assert is_resident([_resident()], _identity("llama3.3:70b")) is False

    def test_an_empty_report_finds_nothing(self) -> None:
        assert find_resident([], _identity()) is None

    def test_a_digest_pinned_identity_still_matches_a_name_only_report(self) -> None:
        """The reason matching is on the name alone. A provider's residency report does not
        re-derive a digest, so comparing whole identities would report a digest-pinned identity as
        *not* resident against the very entry running it (ADR-0024 §2).
        """
        pinned = ModelIdentity(
            ProviderKind.OLLAMA, _MODEL, artifact_digest="sha256:" + "1f3a9c4e2b70" + "0" * 52
        )

        assert is_resident([_resident()], pinned) is True

    def test_the_entry_rather_than_a_bool_is_returned_so_its_memory_can_be_read(self) -> None:
        found = find_resident([_resident()], _identity())

        assert found is not None
        assert found.vram_bytes == 9_895_000_000


class TestFakeProviderResidency:
    def test_load_then_unload_round_trips(self) -> None:
        provider = FakeProvider()
        identity = provider.resolve("fake-model:8b-q8_0")

        loaded = provider.load(identity, RuntimeProfile())

        assert loaded.already_resident is False
        assert is_resident(provider.list_resident(), identity) is True
        assert provider.unload(identity) is True
        assert is_resident(provider.list_resident(), identity) is False

    def test_loading_an_already_resident_model_reports_no_load_happened(self) -> None:
        """A warm model measured as a cold start is a figure an order of magnitude wrong."""
        provider = FakeProvider()
        identity = provider.resolve("fake-model:8b-q8_0")
        provider.load(identity, RuntimeProfile())

        again = provider.load(identity, RuntimeProfile())

        assert again.already_resident is True
        assert again.load_ms is UNSUPPORTED

    def test_unloading_a_model_that_was_not_resident_is_not_a_failure(self) -> None:
        provider = FakeProvider()
        identity = provider.resolve("fake-model:8b-q8_0")

        assert provider.unload(identity) is False

    def test_a_provider_without_the_flag_refuses_rather_than_silently_doing_nothing(self) -> None:
        """The development plan's Phase 5 test, stated exactly: unload on a provider without
        ``force_unload`` raises, not a silent no-op.
        """
        provider = FakeProvider(FakeScript(capabilities=MINIMAL_CAPABILITIES))
        identity = provider.resolve("fake-model:8b-q8_0")

        with pytest.raises(CapabilityUnsupported) as raised:
            provider.unload(identity)

        assert raised.value.details["capability"] == FORCE_UNLOAD

    def test_the_profile_the_model_was_loaded_under_is_recorded(self) -> None:
        """Same weights under a different profile are a different measurement subject."""
        provider = FakeProvider()
        identity = provider.resolve("fake-model:8b-q8_0")
        profile = RuntimeProfile(context_size=8192)

        loaded = provider.load(identity, profile)

        assert loaded.profile_hash == profile.profile_hash


class TestOllamaResidency:
    @respx.mock
    def test_a_resident_model_is_reported_with_its_memory(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )

        resident = OllamaProvider(base_url=_OLLAMA_URL).list_resident()

        assert len(resident) == 1
        assert resident[0].vram_bytes == 9_895_000_000
        assert resident[0].expires_at is not None

    @respx.mock
    def test_nothing_loaded_is_an_empty_report_not_a_failure(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_empty.json"))
        )

        assert OllamaProvider(base_url=_OLLAMA_URL).list_resident() == ()

    @respx.mock
    def test_loading_an_already_resident_model_issues_no_load(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )
        generate = respx.post(f"{_OLLAMA_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_load.json"))
        )

        loaded = OllamaProvider(base_url=_OLLAMA_URL).load(_identity(), RuntimeProfile())

        assert loaded.already_resident is True
        assert loaded.load_ms is UNSUPPORTED
        assert generate.call_count == 0

    @respx.mock
    def test_unloading_a_model_that_is_not_resident_makes_no_request(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """The state the caller wanted, not a failure — so no request is even made."""
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_empty.json"))
        )
        generate = respx.post(f"{_OLLAMA_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_unload.json"))
        )

        assert OllamaProvider(base_url=_OLLAMA_URL).unload(_identity()) is False
        assert generate.call_count == 0

    @respx.mock
    def test_unloading_a_resident_model_evicts_it_immediately(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )
        generate = respx.post(f"{_OLLAMA_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_unload.json"))
        )

        assert OllamaProvider(base_url=_OLLAMA_URL).unload(_identity()) is True
        assert generate.call_count == 1
        assert generate.calls[0].request.read() == b'{"model":"qwen3.5:9b-q8_0","keep_alive":0}'

    @respx.mock
    def test_a_digest_pinned_identity_is_recognised_as_resident(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """LoadCoach holds digest-confident identities; Ollama's `/api/ps` reports names."""
        respx.get(f"{_OLLAMA_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )
        respx.post(f"{_OLLAMA_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_load.json"))
        )
        pinned = ModelIdentity(
            ProviderKind.OLLAMA, _MODEL, artifact_digest="sha256:" + "1f3a9c4e2b70" + "0" * 52
        )

        loaded = OllamaProvider(base_url=_OLLAMA_URL).load(pinned, RuntimeProfile())

        assert loaded.already_resident is True


class TestOpenAICompatibleResidency:
    @pytest.mark.parametrize("flag", [FORCE_UNLOAD, RESIDENCY_QUERY])
    def test_this_protocol_declares_neither_power(self, flag: str) -> None:
        capabilities = OpenAICompatibleProvider(base_url=_OPENAI_URL).capabilities()

        assert getattr(capabilities, flag) is False

    def test_a_caller_can_branch_before_spending_a_call(self) -> None:
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        assert residency_support(provider.capabilities()).is_manageable is False

    @respx.mock
    def test_every_residency_call_is_refused_before_any_http_request(self) -> None:
        """Refused *before* the wire: no route is mocked, so a request would fail the test with a
        connection error rather than a capability error.
        """
        provider = OpenAICompatibleProvider(base_url=_OPENAI_URL)

        with pytest.raises(CapabilityUnsupported) as load_refusal:
            provider.load(_identity(), RuntimeProfile())
        with pytest.raises(CapabilityUnsupported) as unload_refusal:
            provider.unload(_identity())
        with pytest.raises(CapabilityUnsupported) as query_refusal:
            provider.list_resident()

        assert load_refusal.value.details["capability"] == FORCE_UNLOAD
        assert unload_refusal.value.details["capability"] == FORCE_UNLOAD
        assert query_refusal.value.details["capability"] == RESIDENCY_QUERY
        assert respx.calls.call_count == 0
