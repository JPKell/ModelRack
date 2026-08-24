"""Domain module — translating a scripted failure into the typed error the spec names for it.

Imports :mod:`baseaicore`, this package's own types and the standard library; performs no I/O.
Pure functions from a :class:`~modelrack.providers._fake_generation._Plan` to a
:class:`~modelrack.errors.ProviderError`, kept apart from the provider because error translation
is its own concern — the one every adapter in this package will do, and the one
[spec §11.7](../../../docs/packages/modelrack/spec.md) exists to make non-negotiable: *no adapter
raises a raw transport exception*.

Each branch fills the ``details`` keys [spec §13](../../../docs/packages/modelrack/spec.md) names
for its row of the error table, so a consumer's own test — "this provider failure maps to a
documented code carrying the fields I persist, not to INTERNAL_ERROR" — has something to assert
against without a provider running.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from baseaicore import is_supported

from modelrack.errors import (
    ContextLimitExceeded,
    ModelNotFound,
    ProviderError,
    ProviderProtocolError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnavailableReason,
)
from modelrack.providers._fake_generation import MILLISECONDS_PER_SECOND, _truncate_body
from modelrack.providers._fake_script import FakeFailureMode

if TYPE_CHECKING:
    from modelrack.providers._fake_generation import _Plan
    from modelrack.types import GenerationRequest

__all__ = ["failure_error", "timeout_error"]


def timeout_error(elapsed_ms: float, limit_ms: float) -> ProviderTimeout:
    """Return the timeout error, carrying both numbers spec §13 requires of it.

    Both, because "it timed out" without them cannot distinguish a limit set too low from a
    model genuinely stalled — and the first is a configuration fix while the second is not.
    """
    return ProviderTimeout(
        f"The provider did not finish within {limit_ms / MILLISECONDS_PER_SECOND:g}s.",
        details={
            "elapsed_seconds": elapsed_ms / MILLISECONDS_PER_SECOND,
            "limit_seconds": limit_ms / MILLISECONDS_PER_SECOND,
        },
    )


def _with_truncated_body(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge scripted overrides over the defaults, capping whatever ``body`` survives.

    The cap applies to a scripted body as well as the default one. An error object is not a place
    to move an unbounded response — the full body reaches the caller as the artifact it stores,
    not as an exception field — and a fake that let a script past the limit would let a consumer
    build an expectation a real adapter can never meet.
    """
    merged = {**dict(defaults), **dict(overrides)}
    body = merged.get("body")
    if isinstance(body, str):
        merged["body"] = _truncate_body(body)
    return merged


def failure_error(
    plan: _Plan,
    request: GenerationRequest,
    *,
    elapsed_ms: float,
    chunks_delivered: int,
    limit_ms: float,
) -> ProviderError:
    """Turn a scripted failure into the typed error spec §13 says that condition produces.

    Every branch fills the ``details`` keys the spec's table names for its row, so a consumer
    test asserting "this maps to a documented code with the fields I store" has something to
    assert against without a provider. ``FakeFailure.details`` is merged over the top, for the
    test that needs one particular status code rather than a plausible one.
    """
    failure = plan.failure
    if failure is None:  # pragma: no cover — callers check `failure is not None` first
        raise AssertionError("failure_error called for a generation with no scripted failure")
    defaults: dict[str, Any]
    error: ProviderError
    match failure.mode:
        case FakeFailureMode.UNAVAILABLE:
            defaults = {
                "base_url": plan.base_url,
                "reason": ProviderUnavailableReason.CONNECTION_REFUSED.value,
            }
            error = ProviderUnavailable(
                failure.message or f"Cannot reach the provider at {plan.base_url}.",
                details={**defaults, **dict(failure.details)},
            )
        case FakeFailureMode.TIMEOUT:
            defaults = {
                "elapsed_seconds": elapsed_ms / MILLISECONDS_PER_SECOND,
                "limit_seconds": limit_ms / MILLISECONDS_PER_SECOND,
            }
            error = ProviderTimeout(
                failure.message or "The provider stopped answering part-way through the call.",
                details={**defaults, **dict(failure.details)},
            )
        case FakeFailureMode.UNPARSEABLE_BODY:
            defaults = {"body": "<html><head><title>502 Bad Gateway</title>"}
            error = ProviderProtocolError(
                failure.message or "The provider's response was not JSON.",
                details=_with_truncated_body(defaults, failure.details),
            )
        case FakeFailureMode.UNEXPECTED_SHAPE:
            defaults = {
                "body": json.dumps({"unexpected": True}, sort_keys=True),
                "missing_field": "message",
            }
            error = ProviderProtocolError(
                failure.message
                or "The provider's response parsed as JSON but had an unexpected shape.",
                details=_with_truncated_body(defaults, failure.details),
            )
        case FakeFailureMode.TRUNCATED_STREAM:
            defaults = {"chunks_received": chunks_delivered}
            error = ProviderProtocolError(
                failure.message or "The stream ended without a terminal chunk.",
                details={**defaults, **dict(failure.details)},
            )
        case FakeFailureMode.MODEL_NOT_FOUND:
            defaults = {
                "reference": request.identity.provider_model_name,
                "known_model_count": plan.model_count,
            }
            error = ModelNotFound(
                failure.message
                or f"No model named {request.identity.provider_model_name!r} is served.",
                details={**defaults, **dict(failure.details)},
            )
        case FakeFailureMode.CONTEXT_LIMIT_EXCEEDED:
            # The two numbers have to tell one story: a request that overflowed asked for more
            # than the ceiling. Derived from the model's advertised context plus the prompt, so
            # `requested_tokens > maximum_tokens` always holds. When the model advertises no
            # context, both stay honest in the other direction — the provider refused without
            # saying how much it would have allowed, which is the case ContextLimitExceeded
            # documents and which leaves the caller to find the ceiling by bisection.
            maximum = plan.model.max_context
            defaults = {
                "requested_tokens": (
                    maximum + plan.prompt_tokens if is_supported(maximum) else plan.prompt_tokens
                ),
                "maximum_tokens": maximum,
            }
            error = ContextLimitExceeded(
                failure.message or "The request needs more context than the provider serves.",
                details={**defaults, **dict(failure.details)},
            )
        case FakeFailureMode.REJECTED:
            defaults = {"status_code": 400, "provider_message": "unknown option 'num_ctx'"}
            error = ProviderRejected(
                failure.message or "The provider understood the request and refused it.",
                details={**defaults, **dict(failure.details)},
            )
        case _:  # pragma: no cover — every FakeFailureMode is matched above
            raise AssertionError(f"No error is documented for failure mode {failure.mode!r}.")
    return error
