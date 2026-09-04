"""Domain module — the LoRA adapters an application hands this package to serve.

Imports :mod:`baseaicore` and this package's own errors; performs no I/O.

**This package never reads the adapter directory and never imports `setspec`.**
[ADR-0061](../../docs/adr/0061-the-adapter-registry-is-a-directory-and-a-manifest.md) rule 3 draws
that line: FreeWeight reads the directory to enumerate benchmark subjects, LoadCoach reads it to
build routing rows, and ModelRack "receives manifests from the application constructing it, and
validates and mounts them". So the wire shape — `model.adapter_manifest`, a SetSpec payload — stays
in the applications that read it, and the value object below is what an application converts one
*into*. That keeps this package's runtime dependencies at two
([master architecture §2](../../docs/architecture/master-architecture.md) permits only `mirrorwall`
and `commissioner` to depend on `setspec`), and it keeps the conversion — including any field this
package has no use for — a decision the application takes in the open.

Identity is **not** redefined here. :class:`baseaicore.AdapterIdentity` is the definition
(ADR-0058); :class:`AdapterRegistration` is a serving concern that *carries* one, together with the
path to load and the base claim to verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from baseaicore import AdapterIdentity, DataClassification, ValidationError, normalize_digest

if TYPE_CHECKING:
    from baseaicore import IdentityConfidence

__all__ = [
    "GGUF_ADAPTER_FORMAT",
    "AdapterRegistration",
    "AdapterState",
    "AdapterStatus",
]

GGUF_ADAPTER_FORMAT = "gguf"
"""The only artifact format v1 accepts — what llama.cpp serves (ADR-0061 rule 1)."""


def _normalized(value: str, *, field_name: str) -> str:
    """Return ``value`` normalized as a digest, or raise naming the field that carried it."""
    normalized = normalize_digest(value)
    if normalized is None:
        raise ValidationError(
            f"{field_name} must be a sha256 digest ('sha256:' + 64 hex, or bare hex); got "
            f"{value!r}. An adapter is content-addressed, so a digest that will not normalize is "
            "a refusal rather than a name_only registration (ADR-0058 rule 1).",
            details={"field": field_name, "value": value},
        )
    return normalized


@dataclass(frozen=True, slots=True)
class AdapterRegistration:
    """One adapter an application offers this package, with the claim it makes about its base.

    Built by the application from a reviewed ``model.adapter_manifest`` record — field for field,
    with the digests normalized here rather than trusted — and handed to a provider at construction
    or through :meth:`~modelrack.provider.Provider.register_adapters`. It is an *offer*: whether it
    is actually registered on a running server is decided per base, at launch, by
    :func:`baseaicore.verify_adapter_base_compatibility`, and reported as an :class:`AdapterState`.

    The base claim is deliberately two fields rather than one. A PEFT ``adapter_config.json`` names
    its base by *name*, which is not a proof, so ``base_artifact_digest`` may be absent — and an
    absent digest means the registration can only ever reach ``NAME_ONLY`` confidence, flagged
    everywhere the resulting subject surfaces. A digest that is present is the proof, and a
    mismatch against the base actually served is a refusal, never an attempt (ADR-0058 rule 5).

    Excluded on purpose: a **scale**, because there is nowhere in
    :class:`baseaicore.AdapterIdentity` for one to live and one adapter at two scales would vary
    behaviour without varying identity (ADR-0063); and any measurement, because evidence is
    measured against the subject elsewhere and never carried by the thing being measured
    (ADR-0059).

    Attributes:
        name: The manifest's pin and display name, ``^[a-z][a-z0-9_-]{1,63}$``. It is what a
            caller names in :attr:`~modelrack.types.GenerationRequest.adapter`, and what appears in
            the canonical subject string, so it is validated here by constructing the identity
            rather than by a second copy of the rule.
        artifact_path: Where the served artifact is, as an absolute or application-relative path.
            A **locator**, not an identity: renaming the file changes nothing, and changing its
            content makes a different subject (ADR-0061 rule 5). Not checked for existence here —
            a provider reports a missing artifact when it tries to launch with it, which is the
            moment a person can act on it.
        artifact_sha256: The sha256 of the served artifact. **This is the identity.** Normalized on
            construction.
        source_sha256: The sha256 of the training checkpoint, for lineage only, where the operator
            recorded one. Never part of identity, because re-converting one checkpoint can yield a
            different served artifact and the artifact is what produced the behaviour.
        base_model_name: The provider model name the manifest declares as this adapter's base.
        base_artifact_digest: The base's artifact digest where the manifest declares one, else
            ``None``. Present means the claim can be proved; absent means it can only be
            corroborated by name.
        data_classification: How sensitive the material this adapter was trained on is
            (ADR-0065). Required, with no default, because a fail-open default would let an
            unreviewed manifest claim ``public``. This package only carries it — the effective
            classification of work is ``max(caller, adapter)``, computed where classification is
            recorded — but carrying it is what lets an application record the invariant instead of
            assuming it.
        adapter_format: The artifact's format. Only ``"gguf"`` is accepted in v1; a PEFT checkpoint
            is converted once, on drop, as part of the operator's scan workflow.

    Raises:
        ValidationError: If ``name`` is not the manifest's name shape; if ``artifact_sha256`` or
            either optional digest will not normalize; or if ``adapter_format`` is anything but
            ``"gguf"``. Every one is a refusal to hold an adapter that could not be applied
            honestly, taken at construction so no provider has to re-check it.
    """

    name: str
    artifact_path: Path
    artifact_sha256: str
    base_model_name: str
    data_classification: DataClassification
    source_sha256: str | None = None
    base_artifact_digest: str | None = None
    adapter_format: str = GGUF_ADAPTER_FORMAT

    # Derived, cached on first use, and invisible to equality, hashing and repr — the same
    # treatment ModelIdentity gives its canonical ID.
    _identity_cache: AdapterIdentity | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Normalize the digests and refuse a registration that could not be applied honestly."""
        if self.adapter_format != GGUF_ADAPTER_FORMAT:
            raise ValidationError(
                f"adapter_format must be {GGUF_ADAPTER_FORMAT!r}; got {self.adapter_format!r}. "
                "v1 accepts only what llama.cpp serves (ADR-0061 rule 1); a PEFT checkpoint is "
                "converted once, on drop, by the operator's scan workflow.",
                details={"field": "adapter_format", "value": self.adapter_format},
            )
        object.__setattr__(
            self, "artifact_sha256", _normalized(self.artifact_sha256, field_name="artifact_sha256")
        )
        if self.source_sha256 is not None:
            object.__setattr__(
                self, "source_sha256", _normalized(self.source_sha256, field_name="source_sha256")
            )
        if self.base_artifact_digest is not None:
            object.__setattr__(
                self,
                "base_artifact_digest",
                _normalized(self.base_artifact_digest, field_name="base_artifact_digest"),
            )
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        # Constructing the identity here is what validates `name`: the shape lives in BaseAiCore
        # and is not restated, so the two cannot drift on an escaping detail.
        object.__setattr__(
            self,
            "_identity_cache",
            AdapterIdentity(
                name=self.name,
                artifact_digest=self.artifact_sha256,
                source_digest=self.source_sha256,
            ),
        )

    @property
    def identity(self) -> AdapterIdentity:
        """Return the adapter axis of the execution subject this registration names.

        The value object from BaseAiCore, not a copy of it: what evidence, routing, provenance and
        explanations key on is defined once, in the domain foundation, and every component reads
        the same definition (ADR-0058 §1).
        """
        cached = self._identity_cache
        # Unreachable after __post_init__, which always fills it; the check is what makes the
        # property total for mypy without an assert that could be stripped under -O.
        if cached is None:  # pragma: no cover — __post_init__ always sets it
            cached = AdapterIdentity(
                name=self.name,
                artifact_digest=self.artifact_sha256,
                source_digest=self.source_sha256,
            )
            object.__setattr__(self, "_identity_cache", cached)
        return cached


class AdapterStatus(StrEnum):
    """Why an adapter can or cannot be selected right now.

    A caller that cannot see this cannot explain why a freshly scanned adapter is not being used,
    which is the operator confusion ADR-0062's consequences name explicitly. Each member is
    actionable: two of them resolve on their own, two of them need a person.
    """

    REGISTERED = "registered"
    """Pre-registered on a running server and selectable now."""

    PENDING_RESTART = "pending_restart"
    """Compatible, but its base's server was launched before it arrived.

    It folds in at the next natural idle — never mid-work (ADR-0062 decision 3) — so this state
    resolves itself the moment nothing is in flight against that server. One restart per newly
    trained adapter is the honest floor, and it is surfaced rather than hidden.
    """

    AWAITING_BASE = "awaiting_base"
    """No server is running for the base this adapter claims; it registers when one starts.

    Not a problem, and distinct from :attr:`PENDING_RESTART`: nothing has to be restarted, because
    nothing is running.
    """

    INCOMPATIBLE = "incompatible"
    """Refused: the base actually served is not the base this adapter declares.

    Fail closed (ADR-0058 rule 5) — an adapter applied to the wrong base produces plausible,
    confident, wrong output, which is the worst failure available here. Needs a person: either the
    manifest is stale or the base file changed.
    """


@dataclass(frozen=True, slots=True)
class AdapterState:
    """What one registration's situation is, on one provider, right now.

    Returned by :meth:`~modelrack.provider.Provider.list_adapters` and never constructed by a
    caller. A snapshot, not a live view: a server restart between two calls changes every field
    below except the registration itself.

    Attributes:
        adapter: The registration this describes.
        status: Whether it is selectable, and if not, why.
        base_model_name: The base it is registered on, or would be — the served name where a
            server is running, else the name its manifest declares.
        base_confidence: How well its base claim was proved, once a base has actually been served
            under it: ``DIGEST`` when the declared digest matched, ``NAME_ONLY`` when the manifest
            declared none and only the names agreed. ``None`` before any verification has happened.
            A ``NAME_ONLY`` value is a **permanent caveat** that must be shown wherever the
            resulting subject is shown (ADR-0058 rule 5), never quietly rounded up.
        server_id: The id llama-server itself assigned this adapter, read back from
            ``GET /lora-adapters`` rather than assumed from argv order, and the id sent in a
            request's ``lora`` field. ``None`` unless :attr:`status` is
            :attr:`AdapterStatus.REGISTERED`.
        reason: Why, in a sentence a person can act on, when the status needs one — the digest
            mismatch for ``incompatible``, what has to happen for ``pending_restart``. ``None``
            for a plain ``registered``.
    """

    adapter: AdapterRegistration
    status: AdapterStatus
    base_model_name: str
    base_confidence: IdentityConfidence | None = None
    server_id: int | None = None
    reason: str | None = None
