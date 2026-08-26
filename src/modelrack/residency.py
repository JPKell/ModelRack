"""Domain module — what is loaded, what a caller may do about it, and how to ask honestly.

Imports :mod:`baseaicore` and this package's own provider types; performs no I/O. The operations
themselves — :meth:`~modelrack.provider.Provider.load`,
:meth:`~modelrack.provider.Provider.unload`, :meth:`~modelrack.provider.Provider.list_resident` —
belong to the adapters, because each speaks its own provider's wire protocol. What belongs *here*
is everything about residency that is the same for all of them: the capability gate that decides
whether the question may be asked at all, and the identity-matching a caller does to the answer.

**Policy lives in LoadCoach, not here.** Eviction order, idle timeouts, ``max_resident_models``
per device, affinity batching — all of that is
[LoadCoach's queue and scheduling §6](../../docs/packages/loadcoach/queue-and-scheduling.md), and
putting any of it in this package would be handing a capability package an application's
responsibility. ModelRack answers *what is loaded* and *load this* / *unload that*; deciding which
model should be resident is somebody else's job
([spec §3](../../docs/packages/modelrack/spec.md): no routing).

**An unsupported provider is a branch, not a failure.** LoadCoach's rule is that "providers without
residency control simply skip all of this; the behaviour degrades to load-on-demand with a recorded
reason". :func:`residency_support` is what makes that a one-line read of a declaration rather than
a ``try``/``except`` around an operation the caller already knew would be refused — and
:func:`require_residency_query` is what guarantees the refusal is a typed
:class:`~modelrack.errors.CapabilityUnsupported` naming the flag, from every adapter, rather than
a silent no-op that would leave a scheduler believing it had evicted something
(ADR-0007 rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NoReturn

from modelrack.provider import refuse_capability, require_capability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from baseaicore import ModelIdentity

    from modelrack.provider import ProviderCapabilities, ResidentModel

__all__ = [
    "FORCE_UNLOAD",
    "RESIDENCY_QUERY",
    "ResidencySupport",
    "find_resident",
    "is_resident",
    "refuse_force_unload",
    "refuse_residency_query",
    "require_force_unload",
    "require_residency_query",
    "residency_support",
]

FORCE_UNLOAD: Final[str] = "force_unload"
"""The capability flag governing :meth:`~modelrack.provider.Provider.load` and
:meth:`~modelrack.provider.Provider.unload`.

Named once here rather than spelled as a literal at each of the six adapter call sites: the flag
name reaches a caller inside a :class:`~modelrack.errors.CapabilityUnsupported`'s ``details``, and
the conformance suite asserts on it, so a typo in one adapter would be a refusal a caller could
not match on.
"""

RESIDENCY_QUERY: Final[str] = "residency_query"
"""The capability flag governing :meth:`~modelrack.provider.Provider.list_resident`.

Deliberately distinct from :data:`FORCE_UNLOAD`. A provider can plausibly report what it is
holding without letting anyone change it — the read and the write are separate powers, and
collapsing them would force an adapter to claim one to offer the other.
"""


@dataclass(frozen=True, slots=True)
class ResidencySupport:
    """What a caller may do about residency on one provider, read from its declaration.

    A projection of two flags out of :class:`~modelrack.provider.ProviderCapabilities`, so a
    scheduler's residency code takes one small object rather than the whole capability set and
    the two field names it happens to need.

    Attributes:
        can_query: Whether :meth:`~modelrack.provider.Provider.list_resident` will answer.
        can_control: Whether :meth:`~modelrack.provider.Provider.load` and
            :meth:`~modelrack.provider.Provider.unload` will act.
    """

    can_query: bool = False
    can_control: bool = False

    @property
    def is_manageable(self) -> bool:
        """Whether a caller can run a residency policy at all against this provider.

        Returns:
            ``True`` only when both powers are declared. Managing residency needs both halves:
            evicting without being able to see what is loaded is guesswork, and observing without
            being able to act is a report nobody can use. A provider offering one is the
            load-on-demand branch, the same as one offering neither.
        """
        return self.can_query and self.can_control


def residency_support(capabilities: ProviderCapabilities) -> ResidencySupport:
    """Read one provider's residency powers out of its capability declaration.

    Args:
        capabilities: What the adapter declared, from
            :meth:`~modelrack.provider.Provider.capabilities`.

    Returns:
        The two flags that matter, projected. Cheap and non-probing, exactly as the declaration
        it reads is: no request is made, so a scheduler may call this per job without cost.
    """
    return ResidencySupport(
        can_query=capabilities.residency_query,
        can_control=capabilities.force_unload,
    )


def require_residency_query(capabilities: ProviderCapabilities) -> None:
    """Raise unless this provider declares it can report what is resident.

    Args:
        capabilities: What the adapter declared.

    Raises:
        CapabilityUnsupported: If :attr:`RESIDENCY_QUERY` is not declared, with the flag name in
            ``details["capability"]``.
    """
    require_capability(capabilities, RESIDENCY_QUERY, action="report which models are resident")


def require_force_unload(capabilities: ProviderCapabilities, *, action: str) -> None:
    """Raise unless this provider declares it can load and evict on demand.

    Args:
        capabilities: What the adapter declared.
        action: What the caller was trying to do, in the infinitive — ``"load a model on demand"``
            or ``"evict a model on demand"``. Both operations are gated by the one flag (the
            normative capability set has no separate "can load" — ADR-0007 rule 2), so the message
            is what tells a caller which of the two it was refused.

    Raises:
        CapabilityUnsupported: If :attr:`FORCE_UNLOAD` is not declared, with the flag name in
            ``details["capability"]``.
    """
    require_capability(capabilities, FORCE_UNLOAD, action=action)


def find_resident(
    resident: Iterable[ResidentModel], identity: ModelIdentity
) -> ResidentModel | None:
    """Return the resident entry for ``identity``, or ``None``.

    Matched on :attr:`~baseaicore.ModelIdentity.provider_model_name` alone, deliberately. A
    provider's residency report names what it loaded by the name it was asked for; it does not
    re-derive a digest, and it has no notion of the identity confidence the caller's own
    :class:`~baseaicore.ModelIdentity` carries. Comparing whole identities would therefore report
    a digest-pinned identity as *not* resident against the very entry that is running it — the
    name is the only field both sides genuinely agree on
    (ADR-0024 §2).

    Args:
        resident: What :meth:`~modelrack.provider.Provider.list_resident` returned.
        identity: The model to look for.

    Returns:
        The matching entry, or ``None`` when the model is not loaded. The entry rather than a
        bool, because the caller that wants to know *whether* usually then wants to know how much
        memory it is occupying or when the provider intends to evict it.
    """
    for entry in resident:
        if entry.identity.provider_model_name == identity.provider_model_name:
            return entry
    return None


def is_resident(resident: Iterable[ResidentModel], identity: ModelIdentity) -> bool:
    """Report whether ``identity`` is currently loaded.

    Args:
        resident: What :meth:`~modelrack.provider.Provider.list_resident` returned.
        identity: The model to look for.

    Returns:
        ``True`` if a resident entry names it, matched exactly as :func:`find_resident` matches.
    """
    return find_resident(resident, identity) is not None


def refuse_force_unload(*, action: str) -> NoReturn:
    """Refuse a load or unload outright, for a provider whose protocol has no such endpoint.

    Separate from :func:`require_force_unload` only in its type: an adapter that can *never*
    support this — an OpenAI-compatible server has no residency endpoint under any configuration —
    needs a call that a type checker knows does not return, so the refusal can be the whole body
    of a method declaring a return type.

    Args:
        action: What the caller was trying to do, in the infinitive.

    Raises:
        CapabilityUnsupported: Always.
    """
    refuse_capability(FORCE_UNLOAD, action=action)


def refuse_residency_query() -> NoReturn:
    """Refuse a residency query outright, for a provider whose protocol cannot answer one.

    Raises:
        CapabilityUnsupported: Always.
    """
    refuse_capability(RESIDENCY_QUERY, action="report which models are resident")
