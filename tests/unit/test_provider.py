"""Tests for :mod:`modelrack.provider` — the protocol and the types describing a provider.

Phase 1 acceptance criterion 1 is "the protocol is satisfiable by a stub". The real proof of that
is ``mypy --strict`` over this file, not anything asserted at runtime: :func:`_typecheck_stub`
assigns the stub to a :class:`~modelrack.Provider` variable, and if any signature drifted from the
protocol the type check fails. The runtime assertions below are a coarser second net —
:func:`typing.runtime_checkable` verifies that names exist and nothing about their signatures.

This file is not in the development plan's Phase 1 list, which names only ``test_types``,
``test_streaming_types`` and ``test_errors``. It exists because the same plan asks for "a
structural Protocol conformance test [that] compiles against a stub implementation", and
``provider.py`` is its own module — putting its tests inside ``test_types.py`` would have hidden
them from anyone looking for them.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING

import pytest
from baseaicore import (
    UNSUPPORTED,
    ModelDescriptor,
    ModelIdentity,
    ProviderKind,
    RuntimeProfile,
    ValidationError,
    utc_now,
)

from modelrack import (
    AdapterRegistration,
    AdapterState,
    GenerationRequest,
    GenerationResult,
    LoadResult,
    Provider,
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    ResidentModel,
    StreamCompleted,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from modelrack import StreamEvent


class _StubProvider:
    """A minimal structural implementation of the protocol — no inheritance, no I/O.

    Deliberately does not subclass anything from :mod:`modelrack`: a downstream repository must be
    able to write a test double without importing a base class, which is the whole point of the
    protocol being structural.
    """

    kind: ProviderKind = ProviderKind.FAKE

    def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.OK, base_url="http://127.0.0.1:11434")

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True)

    def list_models(self, *, refresh: bool = False) -> Sequence[ModelDescriptor]:
        return ()

    def inspect_model(self, identity: ModelIdentity, *, refresh: bool = False) -> ModelDescriptor:
        return ModelDescriptor(identity=identity, observed_at=utc_now())

    def resolve(self, reference: str, *, refresh: bool = False) -> ModelIdentity:
        return ModelIdentity(ProviderKind.FAKE, reference)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(text="", identity=request.identity)

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        yield StreamCompleted(result=self.generate(request))

    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult:
        return LoadResult(identity=identity, profile_hash=profile.profile_hash)

    def unload(self, identity: ModelIdentity) -> bool:
        return False

    def list_resident(self) -> Sequence[ResidentModel]:
        return ()

    def list_adapters(self) -> Sequence[AdapterState]:
        return ()

    def register_adapters(self, adapters: Sequence[AdapterRegistration]) -> None:
        return None


def _typecheck_stub() -> Provider:
    """Return the stub as a :class:`~modelrack.Provider`.

    This function is the acceptance criterion. It is never called for its value — ``mypy --strict``
    checking that ``_StubProvider`` is assignable to ``Provider`` is the assertion, and a stub
    whose signature drifted from the protocol would fail the ``types`` CI job rather than any test.
    """
    return _StubProvider()


class TestProtocolIsSatisfiable:
    """Acceptance criterion 1, from both directions."""

    def test_a_structural_stub_satisfies_the_protocol_at_runtime(self) -> None:
        assert isinstance(_StubProvider(), Provider)

    def test_the_stub_is_assignable_to_the_protocol(self) -> None:
        """The runtime half of what `_typecheck_stub` proves statically."""
        provider: Provider = _typecheck_stub()
        assert provider.kind is ProviderKind.FAKE

    def test_satisfying_the_protocol_requires_no_inheritance(self) -> None:
        """A downstream test double must not need to import a base class from here."""
        assert Provider not in _StubProvider.__mro__

    def test_an_incomplete_implementation_does_not_satisfy_it(self) -> None:
        class _MissingMethods:
            kind: ProviderKind = ProviderKind.FAKE

            def health(self) -> ProviderHealth:  # pragma: no cover — never called
                raise NotImplementedError

        assert not isinstance(_MissingMethods(), Provider)

    def test_every_spec_method_is_declared(self) -> None:
        """Spec §7's Provider protocol, method for method."""
        expected = {
            "health",
            "capabilities",
            "list_models",
            "inspect_model",
            "resolve",
            "generate",
            "stream",
            "load",
            "unload",
            "list_resident",
            "list_adapters",
            "register_adapters",
        }
        assert expected <= set(dir(Provider))

    def test_the_stub_can_actually_be_driven(self, identity: ModelIdentity) -> None:
        """A protocol nothing can implement usefully is a protocol nobody has tried."""
        provider: Provider = _typecheck_stub()
        request = GenerationRequest(identity=identity, prompt="hi")
        assert provider.generate(request).identity == identity
        events = list(provider.stream(request))
        assert len(events) == 1
        assert isinstance(events[0], StreamCompleted)


class TestProviderCapabilities:
    """The normative flag set, and the honest default."""

    def test_the_flag_set_matches_the_spec_exactly(self) -> None:
        """ADR-0007 rule 2 defers to spec §7 for this list; this dataclass is that list."""
        assert {f.name for f in dataclasses.fields(ProviderCapabilities)} == {
            "streaming",
            "tool_calling",
            "structured_output",
            "json_mode",
            "token_counts",
            "token_level_chunks",
            "thinking_control",
            "logprobs",
            "force_unload",
            "residency_query",
            "kv_metrics",
            "context_configurable",
            "embedding",
            "adapter_hot_swap",
        }

    def test_every_flag_defaults_to_false(self) -> None:
        """A capability that appears by omission is one nobody tested."""
        capabilities = ProviderCapabilities()
        assert not any(
            getattr(capabilities, f.name) for f in dataclasses.fields(ProviderCapabilities)
        )

    def test_every_flag_is_a_bool(self) -> None:
        capabilities = ProviderCapabilities()
        assert all(
            isinstance(getattr(capabilities, f.name), bool)
            for f in dataclasses.fields(ProviderCapabilities)
        )

    def test_context_configurable_is_declarable_independently(self) -> None:
        """It gates whether a caller may set a context or must record one as assumed
        (ADR-0023 §4), so it must not be implied by any other flag."""
        capabilities = ProviderCapabilities(streaming=True, token_counts=True)
        assert capabilities.context_configurable is False

    def test_capabilities_are_immutable(self) -> None:
        capabilities = ProviderCapabilities()
        with pytest.raises(dataclasses.FrozenInstanceError):
            capabilities.streaming = True  # type: ignore[misc]


class TestProviderHealth:
    """A verdict a caller can act on, that always says where it probed."""

    def test_the_status_vocabulary_matches_the_suite_health_document(self) -> None:
        """So an application maps it straight into GET /api/v1/health without translating."""
        assert {m.value for m in ProviderStatus} == {"ok", "degraded", "unavailable"}

    def test_not_configured_is_deliberately_absent(self) -> None:
        """That is an application's statement about a provider it never constructed."""
        assert "not_configured" not in {m.value for m in ProviderStatus}

    def test_a_healthy_provider_reports_its_detail(self) -> None:
        health = ProviderHealth(
            status=ProviderStatus.OK,
            base_url="http://127.0.0.1:11434",
            detail="ollama 0.32.13, 11 models",
            provider_version="0.32.13",
            model_count=11,
        )
        assert health.detail == "ollama 0.32.13, 11 models"
        assert health.model_count == 11

    def test_remoteness_is_carried_so_egress_is_never_silent(self) -> None:
        """Spec §14: remote providers are permitted but always surfaced."""
        health = ProviderHealth(
            status=ProviderStatus.OK, base_url="http://gpu-box.lan:11434", is_remote=True
        )
        assert health.is_remote is True

    def test_loopback_is_not_remote_by_default(self) -> None:
        assert (
            ProviderHealth(status=ProviderStatus.OK, base_url="http://127.0.0.1:11434").is_remote
            is False
        )

    def test_an_unavailable_provider_still_names_where_it_probed(self) -> None:
        health = ProviderHealth(
            status=ProviderStatus.UNAVAILABLE,
            base_url="http://127.0.0.1:11434",
            detail="connection refused",
        )
        assert health.base_url

    @pytest.mark.parametrize("base_url", ["", "   "])
    def test_a_health_result_without_a_base_url_is_rejected(self, base_url: str) -> None:
        with pytest.raises(ValidationError, match="base_url"):
            ProviderHealth(status=ProviderStatus.OK, base_url=base_url)

    def test_unreported_measurements_default_to_unsupported(self) -> None:
        health = ProviderHealth(status=ProviderStatus.OK, base_url="http://127.0.0.1:11434")
        assert health.model_count is UNSUPPORTED
        assert health.latency_ms is UNSUPPORTED


class TestResidencyTypes:
    """What loading produced, and what is currently held."""

    def test_a_load_result_distinguishes_a_warm_model_from_a_fast_load(
        self, identity: ModelIdentity
    ) -> None:
        """The difference between a real cold-start figure and one an order of magnitude wrong."""
        assert LoadResult(identity=identity, already_resident=True).already_resident is True

    def test_an_unreported_load_time_is_unsupported_not_zero(self, identity: ModelIdentity) -> None:
        assert LoadResult(identity=identity).load_ms is UNSUPPORTED

    def test_a_load_result_records_the_profile_it_loaded_under(
        self, identity: ModelIdentity
    ) -> None:
        """Same weights under a different profile are a different measurement subject."""
        profile = RuntimeProfile(context_size=8192)
        result = LoadResult(identity=identity, profile_hash=profile.profile_hash)
        assert result.profile_hash == profile.profile_hash

    def test_a_resident_model_reports_memory_per_device(self, identity: ModelIdentity) -> None:
        """ADR-0027: per device, never summed across a machine."""
        resident = ResidentModel(identity=identity, vram_bytes=15_000_000_000)
        assert resident.vram_bytes == 15_000_000_000

    def test_unreported_residency_figures_are_unsupported(self, identity: ModelIdentity) -> None:
        resident = ResidentModel(identity=identity)
        assert resident.vram_bytes is UNSUPPORTED
        assert resident.total_bytes is UNSUPPORTED
        assert resident.expires_at is None


class TestTypedContractIsPublished:
    """The protocol is only enforceable downstream if the package ships its typing marker.

    Without ``py.typed``, mypy treats every ``modelrack`` import in a consumer as ``Any``: the
    protocol still exists, but nothing checks that FreeWeight's or LoadCoach's use of it is
    correct, and the ``Typing :: Typed`` classifier in ``pyproject.toml`` becomes a claim the
    distribution does not honour. Cheap to lose in a refactor, invisible when lost, and the whole
    value of Phase 1 rests on it.
    """

    def test_the_package_ships_a_py_typed_marker(self) -> None:
        import modelrack

        package_root = pathlib.Path(modelrack.__file__).parent
        assert (package_root / "py.typed").is_file()
