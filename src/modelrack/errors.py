"""Domain module — the typed errors every provider adapter raises instead of leaking a transport.

Imports :mod:`baseaicore` only; performs no I/O and raises nothing itself. Every error here
derives from :class:`baseaicore.SuiteError`, so a caller that already handles suite errors handles
these too, and every ``code`` is part of the public contract
([spec §7](../../docs/packages/modelrack/spec.md)): adding a code is a minor change, changing what
one means is a major one, because these codes reach API error envelopes and CLI exit codes in
three applications.

The hierarchy exists to make one rule enforceable rather than aspirational —
[spec §11.7](../../docs/packages/modelrack/spec.md): *no adapter raises a raw* ``httpx``
*exception*. An application that sees a ``httpx.ConnectError`` has been handed a transport detail
it cannot act on and cannot map to a documented code, and the audit behind
ADR-0007 found exactly that in the prior
implementations. Translation happens once, here, at the only layer that knows what the transport
was doing.

**Never retried, never swallowed, never converted into an empty result**
([spec §13](../../docs/packages/modelrack/spec.md)). ModelRack surfaces what happened and the
caller decides: retry policy is LoadCoach's, and a benchmark that silently substituted an empty
string for a failed generation would score a model on a response it never produced.

**Security.** ``details`` travels into API error envelopes, so it must never carry an API key, a
prompt, or generated content (security standards; spec §14). Adapters put
the *shape* of a failure here — a URL, a limit, a status code — never its content. The one
deliberate exception is :class:`GenerationCancelled`, whose partial text is the caller's own
output being handed back to it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from baseaicore import SuiteError

__all__ = [
    "CapabilityUnsupported",
    "ContextLimitExceeded",
    "GenerationCancelled",
    "ModelNotFound",
    "ProviderError",
    "ProviderProtocolError",
    "ProviderRejected",
    "ProviderTimeout",
    "ProviderUnavailable",
    "ProviderUnavailableReason",
]


class ProviderUnavailableReason(StrEnum):
    """Why a provider could not be reached, when the adapter can tell them apart.

    Recorded in :class:`ProviderUnavailable`'s ``details`` because the three cases need different
    responses from a human: a refused connection usually means "start the runtime", a DNS failure
    means "the address is wrong", and a TLS failure means "the certificate is wrong". Collapsing
    them into one message makes the user guess which.
    """

    CONNECTION_REFUSED = "connection_refused"
    """Nothing is listening at the address."""

    DNS_FAILURE = "dns_failure"
    """The host name does not resolve."""

    TLS_FAILURE = "tls_failure"
    """The connection was made but the TLS handshake failed."""

    NETWORK_ERROR = "network_error"
    """Reachability failed for a reason the adapter could not classify further."""

    LAUNCH_FAILED = "launch_failed"
    """A provider this package supervises could not be started at all.

    The binary is missing or not executable, or no port in the configured range was free.
    Nothing ran, so there is no exit code and no captured output — ``details`` names what was
    attempted instead (ADR-0062).
    """

    PROCESS_EXITED = "process_exited"
    """A provider this package supervises started and then exited.

    Either during startup, before it ever answered a health probe, or later, between two calls.
    ``details`` carries ``exit_code`` and ``stderr_tail`` — the captured end of the process's own
    output — because "it did not start" with no diagnosis is the failure the adapter roadmap
    names as the third supervision risk (ADR-0062).
    """

    NOT_READY = "not_ready"
    """The provider answered, but said it is still loading and cannot serve yet."""


class ProviderError(SuiteError):
    """Base for every failure that originates with a provider or the transport reaching it.

    Catching this catches every error ModelRack raises from a provider call, which is what a
    caller wanting "the provider failed, whatever the reason" should do. Callers that branch on
    *which* failure — LoadCoach deciding between retry and fallback — catch the subclasses.

    Attributes:
        code: ``"PROVIDER_ERROR"``. Raised directly only when a failure genuinely fits no
            subclass; an adapter reaching for it routinely is a sign the hierarchy is missing a
            case.
    """

    code: ClassVar[str] = "PROVIDER_ERROR"


class ProviderUnavailable(ProviderError):
    """The provider could not be reached at all: refused, unresolvable, or otherwise unreachable.

    ``details`` carries ``base_url`` — spec §13 requires it, and
    Graceful Degradation requires that a
    *remote* provider's failure name the host so egress is obvious rather than silent. It also
    carries ``reason`` (a :class:`ProviderUnavailableReason`) where the adapter could classify the
    failure.

    Distinct from :class:`ProviderTimeout`: nothing was listening, as opposed to something
    listening too slowly. An application maps this to a ``provider: unavailable`` health component
    and, in FreeWeight's case, refuses to start a run rather than recording zeros.
    """

    code: ClassVar[str] = "PROVIDER_UNAVAILABLE"


class ProviderTimeout(ProviderError):
    """The provider accepted the connection but did not answer within the limit.

    ``details`` carries ``elapsed_seconds`` and ``limit_seconds`` (spec §13), because "it timed
    out" without both numbers cannot distinguish a limit set too low from a model genuinely
    stalled — and the first is a configuration fix while the second is not.

    A timed-out sample is recorded as ``timeout`` and **never as a score of zero**
    (Graceful Degradation;
    ADR-0016).
    """

    code: ClassVar[str] = "PROVIDER_TIMEOUT"


class ProviderProtocolError(ProviderError):
    """The provider answered, but not with something this adapter can parse.

    Covers a non-JSON body, JSON of an unexpected shape, and a stream that ended without its
    terminal chunk (spec §13). ``details`` carries a **truncated** ``body`` so the caller can
    store it as an artifact and diagnose the provider — FreeWeight does exactly that, keeping the
    raw body alongside the failed sample.

    Truncated rather than complete on purpose: an error object is not a place to move an
    unbounded response, and the full body is available to the caller through the artifact it
    stores.
    """

    code: ClassVar[str] = "PROVIDER_PROTOCOL_ERROR"


class ModelNotFound(ProviderError):
    """The provider does not have the model that was asked for.

    ``details`` carries ``reference`` — what the caller asked for, before resolution — and
    ``known_model_count`` (spec §13). The count is there because "model not found" against a
    provider serving zero models is a different problem from the same message against a provider
    serving eleven: the first is an empty runtime, the second is a typo or a retag.

    Deliberately not :class:`baseaicore.NotFoundError`: this is a provider failure and callers
    handling ``ProviderError`` must catch it, which a generic not-found would escape.
    """

    code: ClassVar[str] = "MODEL_NOT_FOUND"


class ContextLimitExceeded(ProviderError):
    """The request needed more context than the provider would serve.

    ``details`` carries ``requested_tokens`` and ``maximum_tokens`` where the provider reports
    them (spec §13). Either may be absent — a provider that refuses without saying how much it
    would have allowed leaves the caller to discover the ceiling by bisection, and pretending to a
    number here would be a fabricated measurement.
    """

    code: ClassVar[str] = "CONTEXT_LIMIT_EXCEEDED"


class CapabilityUnsupported(ProviderError):
    """Something was requested that this provider has declared it cannot do.

    ``details`` carries ``capability``, the field name from
    :class:`~modelrack.provider.ProviderCapabilities` (spec §13). Raised rather than silently
    degraded: an adapter that accepted a tool definition and quietly dropped it would produce a
    result whose ``finish_reason`` never mentions tools, and the caller would conclude the model
    chose not to call one.

    This is the error a caller should never see if it checked
    :meth:`~modelrack.provider.Provider.capabilities` first — it exists to make the failure loud
    for callers that did not (ADR-0007 rule 2).
    """

    code: ClassVar[str] = "CAPABILITY_UNSUPPORTED"


class GenerationCancelled(ProviderError):
    """A caller's cancellation token fired and the generation was stopped.

    ``details`` carries ``partial_text``: whatever had been generated before the stop, handed back
    rather than discarded (spec §13). This is the one error whose ``details`` legitimately
    contains generated content, because that content is the caller's own — it is being returned to
    the process that asked for it, not exported or logged by this package.

    Reachable only from :meth:`~modelrack.provider.Provider.stream`. A non-streaming
    :meth:`~modelrack.provider.Provider.generate` offers no boundary at which a token could take
    effect, which is why LoadCoach always streams and assembles the response itself (spec §13).

    An expected outcome, not a defect: a cancelled job is cancelled, not failed, and a consumer
    that mapped this to a failure would report a user's own stop request as an error.
    """

    code: ClassVar[str] = "GENERATION_CANCELLED"


class ProviderRejected(ProviderError):
    """The provider understood the request and refused it — a 4xx with something to say.

    ``details`` carries ``status_code`` and ``provider_message``, the latter **verbatim**
    (spec §13). Verbatim because a provider's own wording ("unknown option 'num_ctx'") is usually
    the fastest route to the fix, and paraphrasing it into a house style would discard the
    specific token the user needs to search for.

    Distinct from :class:`ProviderProtocolError`: the provider was understood perfectly and said
    no. Distinct from :class:`CapabilityUnsupported`: that one is ModelRack refusing on the
    provider's declared behalf, before a request is sent.
    """

    code: ClassVar[str] = "PROVIDER_REJECTED"
