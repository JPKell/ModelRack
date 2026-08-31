"""Domain module — the ``Provider`` protocol and the types describing what a provider *is*.

Imports :mod:`baseaicore` and this package's own types; performs no I/O. This is the interface
ADR-0007 exists to establish: one abstraction,
several adapters, one conformance suite, so that three applications contain no provider HTTP code
and cannot disagree about what a token count or a timing means.

The protocol is **structural**. An adapter satisfies it by having the right methods, not by
inheriting anything, which is what lets a downstream repository write a test double without
importing a base class from here. ``mypy --strict`` is where that satisfaction is actually
proven; :func:`typing.runtime_checkable` is enabled for convenience but checks only that names are
present, never that a signature matches.

**Capabilities are declarations, not suggestions.** A caller checks
:class:`ProviderCapabilities` and branches; it never assumes
(ADR-0007 rule 2). An adapter that cannot do
something declares ``False`` and raises
:class:`~modelrack.errors.CapabilityUnsupported` when asked anyway — it never accepts the request
and quietly ignores it, which would produce a result the caller misreads as the model's own
choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NoReturn, Protocol, runtime_checkable

from baseaicore import UNSUPPORTED, Measurement, ValidationError

from modelrack.errors import CapabilityUnsupported

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import datetime

    from baseaicore import ModelDescriptor, ModelIdentity, ProviderKind, RuntimeProfile

    from modelrack.streaming import StreamEvent
    from modelrack.types import GenerationRequest, GenerationResult

__all__ = [
    "LoadResult",
    "Provider",
    "ProviderCapabilities",
    "ProviderHealth",
    "ProviderStatus",
    "ResidentModel",
    "refuse_capability",
    "require_capability",
]


class ProviderStatus(StrEnum):
    """How well a provider is answering, in the suite's own health vocabulary.

    The values match the component statuses every application reports from
    ``GET /api/v1/health``
    (Graceful Degradation), so an application
    maps a :class:`ProviderHealth` straight into its health document without
    inventing a translation that could drift.

    ``not_configured`` is deliberately absent: it is a statement an *application* makes about a
    provider it chose not to set up, and by the time a :class:`Provider` object exists to be asked,
    it is configured. A provider that reports its own configuration as absent would be describing
    a state it cannot be in.
    """

    OK = "ok"
    """Reachable and answering normally."""

    DEGRADED = "degraded"
    """Reachable, but something is wrong or reduced — slow, or missing an optional feature."""

    UNAVAILABLE = "unavailable"
    """Not reachable at all."""


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """A point-in-time answer to "can this provider be used, and what is it?".

    Attributes:
        status: The health verdict.
        base_url: Where the provider was contacted. Present even on success so an application can
            show which endpoint answered.
        is_remote: Whether ``base_url`` points somewhere other than loopback. Carried so a caller
            can surface egress rather than discover it from a firewall log: the suite permits
            remote providers but never hides one (spec §14, and Graceful Degradation requires a
            remote failure to name the host).
        detail: A short human-readable summary, e.g. ``"ollama 0.32.13, 11 models"`` — the exact
            shape an application's health component wants.
        provider_version: The provider's own version, where it reports one.
        model_count: How many models the provider is serving.
        latency_ms: How long the health probe took.
    """

    status: ProviderStatus
    base_url: str
    is_remote: bool = False
    detail: str = ""
    provider_version: str | None = None
    model_count: Measurement = UNSUPPORTED
    latency_ms: Measurement = UNSUPPORTED

    def __post_init__(self) -> None:
        """Validate the base URL is present.

        Raises:
            ValidationError: If ``base_url`` is empty. A health result that cannot say where it
                probed cannot be acted on, and a failure that does not name its host is exactly
                the silent-egress case ``is_remote`` exists to prevent.
        """
        if not self.base_url or not self.base_url.strip():
            raise ValidationError(
                f"ProviderHealth.base_url must name where the provider was contacted; got "
                f"{self.base_url!r}.",
                details={"field": "base_url", "value": self.base_url},
            )


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What one provider can actually do.

    The field set is normative — ADR-0007 rule 2
    defers to [spec §7](../../docs/packages/modelrack/spec.md) for it, and this dataclass is that
    list. Every flag defaults to ``False``, because the honest default for "did this adapter
    declare it?" is no: a capability that appears by omission is a capability nobody tested.

    Attributes:
        streaming: Can produce incremental output.
        tool_calling: Accepts tool definitions and can request calls.
        structured_output: Can be constrained to a supplied JSON Schema.
        json_mode: Can be asked for valid JSON without a schema.
        token_counts: Reports token counts. When ``False``, every count is ``UNSUPPORTED`` and no
            throughput figure can honestly be derived.
        token_level_chunks: Each streamed delta is one model token. Gates **any** per-token
            latency claim ([spec §11.4](../../docs/packages/modelrack/spec.md)): when ``False``,
            the gap between deltas is inter-chunk latency and a caller must not relabel it.
        thinking_control: Reasoning output can be requested or suppressed.
        logprobs: Reports per-token log probabilities.
        force_unload: A model can be evicted on demand.
        residency_query: Which models are currently loaded can be queried.
        kv_metrics: Reports KV-cache metrics.
        context_configurable: The served context length can be set by the caller. **Load-bearing,
            not informational** ([spec §11.10](../../docs/packages/modelrack/spec.md),
            ADR-0023 §4): it is what tells a
            caller whether it may set a context or must record the one it got as ``assumed``. An
            adapter that cannot configure context declares ``False`` rather than accepting the
            setting and ignoring it — the latter produces a run whose recorded context never
            happened.
        embedding: Can produce embeddings.
    """

    streaming: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    json_mode: bool = False
    token_counts: bool = False
    token_level_chunks: bool = False
    thinking_control: bool = False
    logprobs: bool = False
    force_unload: bool = False
    residency_query: bool = False
    kv_metrics: bool = False
    context_configurable: bool = False
    embedding: bool = False


@dataclass(frozen=True, slots=True)
class LoadResult:
    """What happened when a model was asked to load.

    Attributes:
        identity: The model that was loaded.
        already_resident: Whether it was already in memory, so no load occurred. Distinct from a
            fast load: a benchmark measuring cold-start time must be able to tell that it
            accidentally measured a warm model, which is the difference between a real figure and
            one an order of magnitude wrong — it is what a caller sets FreeWeight's cold/warm
            marker from.
        load_ms: How long loading took. ``UNSUPPORTED`` when the provider does not report it and
            no load was observed — never ``0``, which would claim an instantaneous load.
        profile_hash: The :class:`~baseaicore.RuntimeProfile` the model was loaded under, hashed.
            Recorded because the same weights under a different profile are a different
            measurement subject (ADR-0023).
    """

    identity: ModelIdentity
    already_resident: bool = False
    load_ms: Measurement = UNSUPPORTED
    profile_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ResidentModel:
    """A model currently held in memory by the provider.

    Attributes:
        identity: Which weights are resident.
        vram_bytes: Device memory the model occupies. Per device and never summed across a
            machine (ADR-0027).
        total_bytes: Total memory the model occupies, device and host together.
        context_length: The context this instance is **actually being served at**, where the
            provider reports it. Not the model's advertised maximum, which a descriptor already
            carries and which is frequently many times larger: the two differ whenever anything
            configured the context, and the difference is the whole of ADR-0023 §4's distinction
            between a *reported* served context and an *assumed* one. ``UNSUPPORTED`` where the
            provider does not say, which is a fact about the provider and never a zero.
        expires_at: When the provider intends to evict it, where the provider says so — Ollama's
            ``keep_alive`` produces exactly this. ``None`` when the provider does not schedule
            eviction or does not report it.
    """

    identity: ModelIdentity
    vram_bytes: Measurement = UNSUPPORTED
    total_bytes: Measurement = UNSUPPORTED
    expires_at: datetime | None = None
    context_length: Measurement = UNSUPPORTED


def refuse_capability(capability: str, *, action: str) -> NoReturn:
    """Raise :class:`~modelrack.errors.CapabilityUnsupported`, in the one wording every adapter
    refuses with.

    The single spelling of ADR-0007 rule 2's refusal. Written once rather than per adapter so that
    the message, the error type and — most of all — ``details["capability"]`` are identical from
    every provider: the conformance suite asserts on that key, and a caller branching on a refusal
    should not have to know which adapter produced it.

    Typed :data:`~typing.NoReturn` so an adapter whose answer is *always* a refusal — an
    OpenAI-compatible server has no residency endpoint under any configuration — can call this as
    the whole body of a method that declares a return type, without an unreachable ``raise`` after
    it to convince a type checker.

    Args:
        capability: The flag's field name on :class:`ProviderCapabilities`, e.g. ``"streaming"``.
        action: What the caller was trying to do, in the infinitive — completing the sentence
            "this provider ... cannot <action>".

    Raises:
        CapabilityUnsupported: Always, naming ``capability`` in ``details``.
    """
    raise CapabilityUnsupported(
        f"This provider does not declare {capability!r} and cannot {action}. Check "
        "capabilities() and branch, rather than assuming.",
        details={"capability": capability},
    )


def require_capability(capabilities: ProviderCapabilities, capability: str, *, action: str) -> None:
    """Raise unless ``capability`` is declared.

    Args:
        capabilities: What the adapter declared. Read by attribute name, so a flag that does not
            exist is an :class:`AttributeError` at the call site rather than a silent ``False``
            that would refuse a capability the provider actually has.
        capability: The flag's field name on :class:`ProviderCapabilities`, e.g. ``"streaming"``.
        action: What the caller was trying to do, in the infinitive.

    Raises:
        CapabilityUnsupported: If the flag is not declared, via :func:`refuse_capability`.
    """
    if getattr(capabilities, capability):
        return
    refuse_capability(capability, action=action)


@runtime_checkable
class Provider(Protocol):
    """The one interface every model runtime is reached through.

    Synchronous by design (ADR-0003): callers own
    their own concurrency, and this package adds no queueing, no retry and no rate limiting
    ([spec §3](../../docs/packages/modelrack/spec.md)).

    Implementations raise the typed errors in :mod:`modelrack.errors` and never a raw transport
    exception ([spec §11.7](../../docs/packages/modelrack/spec.md)). Every method may raise
    :class:`~modelrack.errors.ProviderUnavailable` or
    :class:`~modelrack.errors.ProviderTimeout`; the per-method ``Raises`` sections below name only
    what is specific to that call.

    Attributes:
        kind: Which sort of provider this is. Reused from :mod:`baseaicore` rather than redefined,
            so the value that reaches a canonical model ID is the same object the adapter declares
            (ADR-0008).
    """

    kind: ProviderKind

    def health(self) -> ProviderHealth:
        """Probe the provider and report whether it can be used.

        Returns:
            The verdict. Returns :attr:`ProviderStatus.UNAVAILABLE` rather than raising when the
            provider cannot be reached: "is it up?" is a question whose negative answer is not an
            exceptional condition, and an application's health endpoint asks it precisely when it
            expects the answer might be no.
        """
        ...

    def capabilities(self) -> ProviderCapabilities:
        """Report what this provider can do.

        Returns:
            The declaration. Cheap and non-probing: it describes the adapter's knowledge of the
            provider kind, so a caller can branch before spending a request.
        """
        ...

    def list_models(self, *, refresh: bool = False) -> Sequence[ModelDescriptor]:
        """List the models the provider is serving.

        Args:
            refresh: Bypass any metadata this provider has cached and re-read from the source.
                The explicit escape hatch a TTL alone cannot provide: a tag can be repointed at
                any moment, so a caller who *knows* a model was re-pulled says so rather than
                waiting out an expiry (see :mod:`modelrack.cache`). An adapter that caches nothing
                accepts the argument and ignores it, so a caller need not ask which kind it holds.

        Returns:
            One descriptor per model, each carrying the identity and whatever metadata the
            provider exposes, with unreported fields as ``UNSUPPORTED`` rather than guessed.
        """
        ...

    def inspect_model(self, identity: ModelIdentity, *, refresh: bool = False) -> ModelDescriptor:
        """Fetch full metadata for one model.

        Args:
            identity: The model to inspect.
            refresh: Bypass any metadata this provider has cached and re-read from the source.
                The explicit escape hatch a TTL alone cannot provide: a tag can be repointed at
                any moment, so a caller who *knows* a model was re-pulled says so rather than
                waiting out an expiry (see :mod:`modelrack.cache`). An adapter that caches nothing
                accepts the argument and ignores it, so a caller need not ask which kind it holds.

        Returns:
            The descriptor, with the provider's untouched response preserved in
            :attr:`~baseaicore.ModelDescriptor.raw`.

        Raises:
            ModelNotFound: If the provider does not have it.
        """
        ...

    def resolve(self, reference: str, *, refresh: bool = False) -> ModelIdentity:
        """Resolve a user-supplied model reference to a concrete identity.

        Handles the shorthand people actually type — a bare name, an alias, a unique prefix — and
        returns what it actually names. The resolution is never allowed to hide a retag
        ([spec §11.8](../../docs/packages/modelrack/spec.md)): a tag such as ``qwen3.5:latest``
        can be repointed at any time, so an identity resolved from one carries
        :attr:`~baseaicore.IdentityConfidence.NAME_ONLY` unless the provider exposed a digest,
        and every consumer treats that as the permanent caveat it is.

        Args:
            reference: What the user typed.
            refresh: Bypass any metadata this provider has cached and re-read from the source.
                The explicit escape hatch a TTL alone cannot provide: a tag can be repointed at
                any moment, so a caller who *knows* a model was re-pulled says so rather than
                waiting out an expiry (see :mod:`modelrack.cache`). An adapter that caches nothing
                accepts the argument and ignores it, so a caller need not ask which kind it holds.

        Returns:
            The resolved identity, with its digest where the provider exposes one — normalized
            through :func:`baseaicore.normalize_digest`, so a digest that will not normalize
            yields a ``name_only`` identity rather than a malformed one
            (ADR-0024 §2).

        Raises:
            ModelNotFound: If nothing matches, or if a prefix matches more than one model — an
                ambiguous reference resolved by picking one would run a different model than the
                user meant.
        """
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one generation and return the complete result.

        A cancellation token on the request has no effect here: a blocking round trip offers no
        boundary at which it could take effect. Callers that need to cancel use :meth:`stream`.

        Args:
            request: What to generate, and how.

        Returns:
            The complete outcome.

        Raises:
            ModelNotFound: If the requested model is not available.
            CapabilityUnsupported: If the request needs something this provider has declared it
                cannot do.
            ContextLimitExceeded: If the request needs more context than the provider will serve.
            ProviderRejected: If the provider understood the request and refused it.
            ProviderProtocolError: If the response cannot be parsed.
        """
        ...

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        """Run one generation, yielding events as they arrive.

        The iterator yields zero or more deltas followed by exactly one terminal event —
        :class:`~modelrack.streaming.StreamCompleted` or
        :class:`~modelrack.streaming.StreamFailed` — and nothing after it. Failures that prevent
        the stream from starting raise instead, because there is no stream to terminate.

        Cancellation takes effect within one chunk boundary and leaves no connection open
        ([spec §11.6](../../docs/packages/modelrack/spec.md)). Abandoning the iterator without
        draining it must also close cleanly; implementations own that in their ``finally``.

        A token **already cancelled when this is called** yields one
        :class:`~modelrack.streaming.StreamFailed` carrying
        :class:`~modelrack.errors.GenerationCancelled` and opens no connection at all. Delivered
        rather than raised, so a caller has one cancellation path instead of two — but ordered
        *after* the capability checks, because a request naming something the provider never
        declared is malformed whichever way the token points, and reporting it as a cancellation
        would hide the caller's own bug.

        Args:
            request: What to generate, and how. Its ``cancel`` token is honoured here.

        Yields:
            Deltas, then one terminal event.

        Raises:
            CapabilityUnsupported: If this provider has declared it cannot stream.
            ModelNotFound: If the requested model is not available.
        """
        ...

    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult:
        """Ask the provider to load a model under a runtime profile.

        Args:
            identity: The model to load.
            profile: How it should be loaded and served. A default-constructed profile means
                "provider defaults" and is itself a legal profile.

        Returns:
            What happened, including whether the model was already resident.

        Raises:
            ModelNotFound: If the provider does not have it.
            CapabilityUnsupported: If this provider cannot be asked to load a model explicitly.
        """
        ...

    def unload(self, identity: ModelIdentity) -> bool:
        """Ask the provider to evict a model from memory.

        Args:
            identity: The model to evict.

        Returns:
            ``True`` if the model was resident and has been evicted, ``False`` if it was not
            resident to begin with. The second is not a failure — it is the state the caller
            wanted — which is why this returns a bool rather than raising.

        Raises:
            CapabilityUnsupported: If this provider declares no ``force_unload``.
        """
        ...

    def list_resident(self) -> Sequence[ResidentModel]:
        """List the models the provider currently holds in memory.

        Returns:
            One entry per resident model. Empty when none are loaded, which is different from the
            provider being unable to answer — that raises instead.

        Raises:
            CapabilityUnsupported: If this provider declares no ``residency_query``.
        """
        ...
