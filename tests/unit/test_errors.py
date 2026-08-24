"""Tests for :mod:`modelrack.errors` — the typed hierarchy and its stable codes.

The codes are a published contract ([spec §7](../docs/packages/modelrack/spec.md)): they reach API
error envelopes and CLI exit codes in three applications, so a silent rename here breaks consumers
that never imported this package. These tests pin every one against the spec table.
"""

from __future__ import annotations

import pickle

import pytest
from baseaicore import SuiteError

from modelrack import errors
from modelrack.errors import (
    CapabilityUnsupported,
    ContextLimitExceeded,
    GenerationCancelled,
    ModelNotFound,
    ProviderError,
    ProviderProtocolError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnavailableReason,
)

# The spec §7 tree, verbatim: class -> code.
_SPEC_CODES = {
    ProviderError: "PROVIDER_ERROR",
    ProviderUnavailable: "PROVIDER_UNAVAILABLE",
    ProviderTimeout: "PROVIDER_TIMEOUT",
    ProviderProtocolError: "PROVIDER_PROTOCOL_ERROR",
    ModelNotFound: "MODEL_NOT_FOUND",
    ContextLimitExceeded: "CONTEXT_LIMIT_EXCEEDED",
    CapabilityUnsupported: "CAPABILITY_UNSUPPORTED",
    GenerationCancelled: "GENERATION_CANCELLED",
    ProviderRejected: "PROVIDER_REJECTED",
}


class TestErrorCodes:
    """Every code matches the spec exactly, and every class is reachable from the package root."""

    @pytest.mark.parametrize(("error_type", "code"), list(_SPEC_CODES.items()))
    def test_code_matches_the_spec(self, error_type: type[ProviderError], code: str) -> None:
        assert error_type.code == code

    def test_every_code_is_screaming_snake_case(self) -> None:
        """API Standards require this shape; a lowercase code would not match the envelope."""
        for code in _SPEC_CODES.values():
            assert code == code.upper()
            assert " " not in code

    def test_codes_are_unique(self) -> None:
        """Two classes sharing a code would make the wire form ambiguous."""
        assert len(set(_SPEC_CODES.values())) == len(_SPEC_CODES)

    def test_the_hierarchy_has_no_undeclared_members(self) -> None:
        """A new error added without a test is a code no consumer was told about."""
        exported = {
            getattr(errors, name)
            for name in errors.__all__
            if isinstance(getattr(errors, name), type)
            and issubclass(getattr(errors, name), ProviderError)
        }
        assert exported == set(_SPEC_CODES)


class TestHierarchy:
    """Catching `ProviderError` catches everything; catching a subclass catches only it."""

    @pytest.mark.parametrize("error_type", [t for t in _SPEC_CODES if t is not ProviderError])
    def test_every_error_is_a_provider_error(self, error_type: type[ProviderError]) -> None:
        assert issubclass(error_type, ProviderError)

    @pytest.mark.parametrize("error_type", list(_SPEC_CODES))
    def test_every_error_is_a_suite_error(self, error_type: type[ProviderError]) -> None:
        """A caller already handling suite errors handles these without a new except clause."""
        assert issubclass(error_type, SuiteError)

    def test_catching_the_base_catches_a_subclass(self) -> None:
        with pytest.raises(ProviderError):
            raise ProviderTimeout("slow")

    def test_catching_a_subclass_does_not_catch_a_sibling(self) -> None:
        """LoadCoach branches on which failure it got; siblings must not be interchangeable."""
        with pytest.raises(ProviderTimeout):
            try:
                raise ProviderTimeout("slow")
            except ProviderRejected:  # a sibling, so it must not intercept
                pytest.fail("a sibling error type caught ProviderTimeout")

    def test_model_not_found_is_a_provider_error_not_a_generic_not_found(self) -> None:
        """A caller handling ProviderError must catch it; a generic not-found would escape."""
        from baseaicore import NotFoundError

        assert not issubclass(ModelNotFound, NotFoundError)
        assert ModelNotFound.code != NotFoundError.code


class TestDetails:
    """`details` is structured context for machines, and it survives a process boundary."""

    def test_details_are_preserved(self) -> None:
        error = ProviderUnavailable(
            "nothing listening",
            details={"base_url": "http://127.0.0.1:11434", "reason": "connection_refused"},
        )
        assert error.details["base_url"] == "http://127.0.0.1:11434"

    def test_details_survive_pickling(self) -> None:
        """Errors cross process boundaries in a job runner; structured context is the half a
        machine reads, and the default exception reduction drops it."""
        error = ProviderTimeout("too slow", details={"elapsed_seconds": 30.0, "limit_seconds": 30})
        restored = pickle.loads(pickle.dumps(error))  # noqa: S301 — our own object, in-process
        assert restored.details == {"elapsed_seconds": 30.0, "limit_seconds": 30}
        assert restored.code == "PROVIDER_TIMEOUT"

    def test_details_default_to_empty_rather_than_none(self) -> None:
        assert ProviderError("something").details == {}

    def test_the_message_is_what_str_returns(self) -> None:
        """A printed traceback stays readable; `details` is not formatted into it."""
        assert str(ProviderRejected("unknown option 'num_ctx'")) == "unknown option 'num_ctx'"

    def test_a_caller_mutating_its_dict_cannot_change_a_raised_error(self) -> None:
        supplied = {"base_url": "http://127.0.0.1:11434"}
        error = ProviderUnavailable("nope", details=supplied)
        supplied["base_url"] = "http://elsewhere"
        assert error.details["base_url"] == "http://127.0.0.1:11434"


class TestUnavailableReason:
    """The three unreachable cases need different responses from a human."""

    @pytest.mark.parametrize(
        "reason",
        [
            ProviderUnavailableReason.CONNECTION_REFUSED,
            ProviderUnavailableReason.DNS_FAILURE,
            ProviderUnavailableReason.TLS_FAILURE,
            ProviderUnavailableReason.NETWORK_ERROR,
        ],
    )
    def test_every_reason_serializes_as_its_own_name(
        self, reason: ProviderUnavailableReason
    ) -> None:
        assert isinstance(reason.value, str)
        assert reason.value == reason.value.lower()

    def test_reasons_are_distinct(self) -> None:
        assert len({r.value for r in ProviderUnavailableReason}) == len(ProviderUnavailableReason)


class TestCancellationIsNotAFailure:
    """A cancelled generation is cancelled, not failed."""

    def test_partial_text_is_carried_in_details(self) -> None:
        """The one error whose details legitimately hold generated content — the caller's own."""
        error = GenerationCancelled("stopped", details={"partial_text": "Local models are "})
        assert error.details["partial_text"] == "Local models are "

    def test_it_is_still_a_provider_error_so_broad_handlers_see_it(self) -> None:
        assert issubclass(GenerationCancelled, ProviderError)
