"""Tests for :mod:`modelrack.adapters` — the value objects an application hands this package.

The point of these is not coverage of a small dataclass. It is that **every refusal happens at
construction**, so no provider has to re-check a registration and no code path can hold an adapter
whose digest, name or format could not be applied honestly (ADR-0058 rule 1).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from baseaicore import AdapterIdentity, DataClassification, ValidationError

from modelrack import AdapterRegistration, AdapterState, AdapterStatus
from modelrack.adapters import GGUF_ADAPTER_FORMAT

_ARTIFACT = "9e2b41d07c55" + "0" * 52
_SOURCE = "1f3a9c4e2b70" + "0" * 52
_BASE = "aa11bb22cc33" + "0" * 52


def _registration(**overrides: object) -> AdapterRegistration:
    fields: dict[str, object] = {
        "name": "factcheck",
        "artifact_path": Path("/models/adapters/factcheck.gguf"),
        "artifact_sha256": _ARTIFACT,
        "base_model_name": "qwen3.5-9b-q8_0",
        "data_classification": DataClassification.INTERNAL,
    }
    fields.update(overrides)
    return AdapterRegistration(**fields)  # type: ignore[arg-type]  # a test's kwargs bag


class TestIdentityIsNotRedefined:
    def test_the_identity_is_baseaicores_value_object(self) -> None:
        """ADR-0058 §1: the definition lives in the domain foundation, and is used, not copied."""
        registration = _registration(source_sha256=_SOURCE)

        assert isinstance(registration.identity, AdapterIdentity)
        assert registration.identity == AdapterIdentity(
            name="factcheck", artifact_digest=f"sha256:{_ARTIFACT}"
        )

    def test_the_canonical_suffix_comes_from_the_shared_rule(self) -> None:
        assert _registration().identity.canonical_suffix == "+factcheck@sha256:9e2b41d07c55"

    def test_the_identity_is_stable_across_calls(self) -> None:
        """Cached, and the cache is invisible: two reads are the same value."""
        registration = _registration()

        assert registration.identity is registration.identity

    def test_lineage_never_changes_the_identity(self) -> None:
        """``source_digest`` records where an adapter came from, never what it is."""
        assert _registration(source_sha256=_SOURCE).identity == _registration().identity


class TestDigestsAreNormalized:
    def test_a_bare_hex_artifact_digest_is_normalized(self) -> None:
        assert _registration().artifact_sha256 == f"sha256:{_ARTIFACT}"

    def test_an_already_prefixed_digest_is_unchanged(self) -> None:
        assert _registration(artifact_sha256=f"sha256:{_ARTIFACT}").artifact_sha256 == (
            f"sha256:{_ARTIFACT}"
        )

    def test_an_uppercase_digest_is_lowered(self) -> None:
        assert _registration(artifact_sha256=_ARTIFACT.upper()).artifact_sha256 == (
            f"sha256:{_ARTIFACT}"
        )

    def test_the_optional_digests_are_normalized_too(self) -> None:
        registration = _registration(source_sha256=_SOURCE, base_artifact_digest=_BASE)

        assert registration.source_sha256 == f"sha256:{_SOURCE}"
        assert registration.base_artifact_digest == f"sha256:{_BASE}"

    def test_an_absent_optional_digest_stays_absent(self) -> None:
        registration = _registration()

        assert registration.source_sha256 is None
        assert registration.base_artifact_digest is None


class TestItRefusesWhatCouldNotBeAppliedHonestly:
    @pytest.mark.parametrize("field", ["artifact_sha256", "source_sha256", "base_artifact_digest"])
    def test_a_digest_that_will_not_normalize_is_a_refusal(self, field: str) -> None:
        """Never a ``name_only`` adapter: an adapter is content-addressed or it is nothing."""
        with pytest.raises(ValidationError) as raised:
            _registration(**{field: "not-a-digest"})

        assert raised.value.details["field"] == field

    @pytest.mark.parametrize("name", ["Factcheck", "9lives", "a", "fact check", "x" * 65, ""])
    def test_a_name_outside_the_manifest_shape_is_refused(self, name: str) -> None:
        """The shape is BaseAiCore's, reached by constructing the identity — never restated."""
        with pytest.raises(ValidationError) as raised:
            _registration(name=name)

        assert raised.value.details["field"] == "name"

    def test_a_format_other_than_gguf_is_refused(self) -> None:
        with pytest.raises(ValidationError) as raised:
            _registration(adapter_format="safetensors")

        assert raised.value.details["field"] == "adapter_format"
        assert "gguf" in str(raised.value)

    def test_gguf_is_the_default(self) -> None:
        assert _registration().adapter_format == GGUF_ADAPTER_FORMAT


class TestItCarriesWhatServingNeeds:
    def test_a_string_path_becomes_a_path(self) -> None:
        assert _registration(artifact_path="/models/a.gguf").artifact_path == Path("/models/a.gguf")

    def test_the_classification_is_required_and_carried(self) -> None:
        """ADR-0065 rule 1: no default, because a fail-open default would be filled in silently."""
        with pytest.raises(TypeError):
            AdapterRegistration(  # type: ignore[call-arg]  # the omission is the assertion
                name="factcheck",
                artifact_path=Path("/a.gguf"),
                artifact_sha256=_ARTIFACT,
                base_model_name="base",
            )
        assert _registration().data_classification is DataClassification.INTERNAL

    def test_two_registrations_of_the_same_bytes_are_equal(self) -> None:
        assert _registration() == _registration()

    def test_different_bytes_under_one_name_are_a_different_registration(self) -> None:
        """A rescan that found new bytes is a new subject, not an update to an old one."""
        assert _registration() != _registration(artifact_sha256=_SOURCE)


class TestStateIsActionable:
    def test_a_plain_registration_state_carries_no_reason(self) -> None:
        state = AdapterState(
            adapter=_registration(),
            status=AdapterStatus.REGISTERED,
            base_model_name="qwen3.5-9b-q8_0",
            server_id=0,
        )

        assert state.reason is None
        assert state.base_confidence is None

    def test_every_status_is_one_of_the_four(self) -> None:
        """Each is actionable and distinct: two resolve on their own, two need a person."""
        assert {status.value for status in AdapterStatus} == {
            "registered",
            "pending_restart",
            "awaiting_base",
            "incompatible",
        }
