"""Provider adapter — a deterministic, scriptable model runtime that never leaves the process.

Imports :mod:`baseaicore`, this package's own types and the standard library. It opens no socket,
reads no environment and needs no model, which is what makes the suite's default test suite pass
"with no GPU, no Ollama, no network"
(testing standards §3, gold standard G9).

Built **before** the Ollama adapter, deliberately
(ADR-0007 rule 6): FreeWeight's runner,
LoadCoach's executor and IdeaPress's workflows are all developed against this, so it has to exist
before the thing it stands in for. It is shipped API, not a test helper — the supported import
path is :mod:`modelrack.testing`.

**The fake is honest before it is convenient.** Everything a real adapter is forbidden to do, this
one is forbidden to do too:

* A capability it has not declared is refused with
  :class:`~modelrack.errors.CapabilityUnsupported`, never accepted and quietly dropped. That
  includes ``context_configurable``, which is load-bearing rather than informational
  ([spec §11.10](../../../docs/packages/modelrack/spec.md)).
* A measurement it does not have is ``UNSUPPORTED``, never ``0``
  (ADR-0016). With ``token_counts``
  undeclared, every token count is absent — not a plausible-looking number.
* What it *observed* and what it *claims to have spent* stay in separate fields. Client timings
  come from the scripted delays because the fake really did simulate them; ``backend_*`` fields
  are ``UNSUPPORTED`` unless a script supplies them, because the fake ran no model and has no
  account of work it did not do.
* A digest that will not normalize is discarded with a recorded reason and yields a ``name_only``
  identity, never a malformed one
  (ADR-0024 §2).
* Every stream ends with exactly one terminal event. **Cancellation is delivered as**
  :class:`~modelrack.streaming.StreamFailed`, not raised through the iterator: a raise mid-drain
  ends the stream without a terminal event, which is precisely how :mod:`modelrack.streaming`
  defines a *truncated* stream. Conflating "the caller stopped it" with "the connection dropped"
  would make the one failure a consumer must handle gracefully indistinguishable from the one it
  must treat as a defect.

The named risk for this phase is a fake more forgiving than the runtime it replaces, which would
hide a real integration bug in three applications for months
(audit §11.3). The mitigations are structural
rather than aspirational: the nasty cases are first-class script members
(:class:`~modelrack.providers._fake_script.FakeFailureMode`), the weakest possible declaration is
one constant away (:data:`~modelrack.providers._fake_script.MINIMAL_CAPABILITIES`), the script
itself refuses internally dishonest combinations, and the same conformance suite that runs here
runs against the recorded Ollama and OpenAI-compatible transports in Phases 3 and 4.

**Where things live.** The development plan names one file for this phase; it is three, because
one would be the thousand-line "god module" the
coding standards §13 name as an anti-pattern, and
because each seam is real. ``_fake_script`` holds the declarative value objects a test writes
down, ``_fake_generation`` the pure functions that turn a script, a seed and a request into
content, and this module the provider that decides when — and whether — that content is delivered.
Every script type is re-exported here and from :mod:`modelrack.testing`, so no caller sees the
split.

**Determinism** is by construction, not by convention. Text, chunking, token counts, tool-call ids
and schema-shaped output all derive from SHA-256 over a canonical seed string, so the same script
and seed produce byte-identical output in another process, under another ``PYTHONHASHSEED``, on
another platform and on another Python version. :func:`random.Random` was rejected for this: its
core generator is reproducible but the derived helpers (``choice``, ``sample``, ``shuffle``) have
changed between releases, and "identical across platforms" has to survive a Python upgrade to mean
anything.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Any, Final

from baseaicore import (
    UNSUPPORTED,
    Measurement,
    ModelDescriptor,
    ModelIdentity,
    ProviderKind,
    TokenCount,
    TokenUsage,
    Unsupported,
    ValidationError,
    is_supported,
    normalize_digest,
    utc_now,
)

from modelrack.errors import (
    CapabilityUnsupported,
    ContextLimitExceeded,
    GenerationCancelled,
    ModelNotFound,
    ProviderUnavailable,
    ProviderUnavailableReason,
)
from modelrack.provider import (
    LoadResult,
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    ResidentModel,
)
from modelrack.providers._fake_errors import failure_error, timeout_error
from modelrack.providers._fake_generation import (
    MILLISECONDS_PER_SECOND,
    _Plan,
    _planned_steps,
    _planned_text,
    _planned_tool_calls,
    _render_prompt,
    _simulated_token_count,
)
from modelrack.providers._fake_script import (
    DEFAULT_MODEL,
    FULL_CAPABILITIES,
    MINIMAL_CAPABILITIES,
    SIMULATED_TOKEN_CHARACTERS,
    FakeFailure,
    FakeFailureMode,
    FakeGeneration,
    FakeModel,
    FakeScript,
    FakeToolCall,
)
from modelrack.streaming import StreamCompleted, StreamEvent, StreamFailed, TokenDelta
from modelrack.types import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    ResponseFormatKind,
    Timing,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import datetime

    from baseaicore import RuntimeProfile

__all__ = [
    "DEFAULT_MODEL",
    "FULL_CAPABILITIES",
    "MINIMAL_CAPABILITIES",
    "SIMULATED_TOKEN_CHARACTERS",
    "FakeFailure",
    "FakeFailureMode",
    "FakeGeneration",
    "FakeModel",
    "FakeProvider",
    "FakeScript",
    "FakeToolCall",
]

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: Final[float] = 60.0


class FakeProvider:
    """A complete :class:`~modelrack.provider.Provider` that runs no model and opens no socket.

    Satisfies the protocol structurally, without inheriting anything from it, which is the whole
    point of the protocol being structural: a downstream repository substitutes this wherever it
    would use :class:`~modelrack.providers.ollama.OllamaProvider` and its production code cannot
    tell the difference.

    Everything it does is described by a :class:`~modelrack.providers._fake_script.FakeScript` and
    a seed, and everything it produces is derived from them by SHA-256 — so the same pair yields
    byte-identical text, chunking, token counts and tool-call identifiers in another process, on
    another platform and under another ``PYTHONHASHSEED``.

    Simulated delays do **not** cost wall time unless a sleep function is injected. A script that
    says the first token takes 900 ms produces ``client_ttft_ms == 900.0`` in a test that runs in
    microseconds — which is what keeps thousands of downstream tests fast enough to be run
    constantly (gold standard G10). Pass ``sleep=time.sleep`` when a test genuinely needs the time
    to pass, such as one racing a background thread against a cancellation.

    Invariants:
        * Deterministic: identical ``(script, seed)`` inputs produce identical outputs, for the
          same call index, forever.
        * Honest: it refuses anything its declared
          :class:`~modelrack.provider.ProviderCapabilities` do not cover, and reports every
          measurement it does not have as ``UNSUPPORTED``.
        * Every stream ends with exactly one terminal event and yields nothing after it.

    Thread safety: the script is frozen and the mutable state — the position in the script and the
    set of resident models — is guarded by a lock, so several threads may drive one instance. Two
    threads generating at once each consume their own generation, in whatever order they arrive;
    a test that needs a fixed pairing drives one provider per thread.

    Args:
        script: What this provider serves and does. ``None`` uses a fully-populated single-model
            catalogue with :data:`~modelrack.providers._fake_script.FULL_CAPABILITIES`.
        seed: The root of every derived value. Two providers with different seeds answer the same
            prompt differently, which is what stops a consumer's test from depending on one exact
            sentence.
        sleep: Called with a duration **in seconds** to make a scripted delay real. ``None``
            simulates delays without spending time.
        clock: Where ``observed_at`` on a descriptor comes from. Injected so a test can freeze it;
            it is the only real-world reading this provider takes.
        default_timeout_seconds: The limit applied when a request names none. A default exists
            precisely so that ``timeout_seconds=None`` can mean "use the default" and never "no
            timeout" ([spec §14](../../../docs/packages/modelrack/spec.md)).

    Raises:
        ValidationError: If ``seed`` is not a whole number, or ``default_timeout_seconds`` is not
            a finite number above zero.
    """

    kind: ProviderKind = ProviderKind.FAKE

    def __init__(
        self,
        script: FakeScript | None = None,
        *,
        seed: int = 0,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], datetime] = utc_now,
        default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Validate the injected boundaries and start at the beginning of the script."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValidationError(
                f"FakeProvider.seed must be a whole number; got {seed!r}.",
                details={"field": "seed", "value": repr(seed)},
            )
        if (
            isinstance(default_timeout_seconds, bool)
            or not isinstance(default_timeout_seconds, int | float)
            or not math.isfinite(default_timeout_seconds)
            or default_timeout_seconds <= 0
        ):
            raise ValidationError(
                f"FakeProvider.default_timeout_seconds must be a finite number above 0; got "
                f"{default_timeout_seconds!r}. There is no 'no timeout': a call with no ceiling "
                "is a hung application (spec §14).",
                details={
                    "field": "default_timeout_seconds",
                    "value": repr(default_timeout_seconds),
                },
            )
        self._script = script if script is not None else FakeScript()
        self._seed = seed
        self._sleep = sleep
        self._clock = clock
        self._default_timeout_seconds = float(default_timeout_seconds)
        self._lock = threading.Lock()
        self._generation_index = 0
        self._resident: set[str] = set()

    # ------------------------------------------------------------------ inspection and control

    @property
    def script(self) -> FakeScript:
        """Return the script this provider is running. Frozen, so reading it cannot change it."""
        return self._script

    @property
    def seed(self) -> int:
        """Return the seed every derived value comes from."""
        return self._seed

    @property
    def generations_consumed(self) -> int:
        """Return how many generate or stream calls have been served.

        Counts calls, not scripted entries, so it keeps rising past the end of a script that
        repeats its final generation. A workflow test asserting "this stage made exactly two model
        calls" reads it here rather than counting log lines.
        """
        with self._lock:
            return self._generation_index

    def reset(self) -> None:
        """Rewind to the start of the script and evict everything resident.

        For a parametrized test that reuses one provider across cases. Cheaper and clearer than
        rebuilding it, and it makes "this case starts from a cold provider" a statement rather
        than an assumption.
        """
        with self._lock:
            self._generation_index = 0
            self._resident.clear()

    # ------------------------------------------------------------------------------- protocol

    def health(self) -> ProviderHealth:
        """Report whether this provider can be used, without raising when it cannot.

        Returns:
            The scripted verdict. A provider scripted ``UNAVAILABLE`` reports no version and no
            model count: something that cannot be reached cannot be asked what it is serving, and
            answering anyway would be the fake inventing knowledge a real client would not have.
        """
        script = self._script
        if script.health_status is ProviderStatus.UNAVAILABLE:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                base_url=script.base_url,
                is_remote=script.is_remote,
                detail="fake provider scripted unavailable",
                latency_ms=script.health_latency_ms,
            )
        return ProviderHealth(
            status=script.health_status,
            base_url=script.base_url,
            is_remote=script.is_remote,
            detail=(
                f"fake {script.provider_version or 'provider'}, "
                f"{len(script.models)} model{'' if len(script.models) == 1 else 's'}"
            ),
            provider_version=script.provider_version,
            model_count=len(script.models),
            latency_ms=script.health_latency_ms,
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report what this provider declares it can do.

        Returns:
            The declaration, unchanged. Cheap and non-probing, and therefore answerable even while
            the provider is scripted unavailable: a caller decides whether it *may* stream before
            it discovers whether it *can* connect.
        """
        return self._script.capabilities

    def list_models(self) -> Sequence[ModelDescriptor]:
        """List the catalogue.

        Returns:
            One descriptor per model, in catalogue order, each observed at the injected clock's
            current instant. Empty when the catalogue is empty, which is a real state — a runtime
            with nothing pulled — and not a failure.

        Raises:
            ProviderUnavailable: If this provider is scripted unavailable.
        """
        self._require_available()
        return tuple(self._describe(model) for model in self._script.models)

    def inspect_model(self, identity: ModelIdentity) -> ModelDescriptor:
        """Fetch full metadata for one model.

        The descriptor carries the identity the catalogue holds **now**, which may differ from the
        one asked for: that is what a retag looks like, and returning the requested identity back
        unchanged would hide it ([spec §11.8](../../../docs/packages/modelrack/spec.md)). The
        difference is recorded in the descriptor's ``raw`` and logged at DEBUG.

        Args:
            identity: The model to inspect. Matched on the provider's model name; an alias works
                too, because that is what a caller who resolved one will be holding.

        Returns:
            The descriptor, with the fake's synthesized provider payload preserved in ``raw``.

        Raises:
            ModelNotFound: If the catalogue has nothing under that name.
            ProviderUnavailable: If this provider is scripted unavailable.
        """
        self._require_available()
        model = self._require_model(identity.provider_model_name)
        descriptor = self._describe(model)
        requested = identity.artifact_digest
        current = descriptor.identity.artifact_digest
        if requested is not None and requested != current:
            logger.debug(
                "fake.model.retagged",
                extra={
                    "model": model.name,
                    "requested_digest": requested,
                    "current_digest": current,
                },
            )
        return descriptor

    def resolve(self, reference: str) -> ModelIdentity:
        """Resolve what a user typed to a concrete identity.

        Matches an exact name first, then an alias, then a unique prefix of either — the shorthand
        people actually type. Whichever route it took, the identity that comes back carries the
        provider's own model name and the digest the catalogue holds, so an identity resolved from
        a tag that has been repointed says ``name_only`` rather than pretending to pin weights it
        cannot prove (ADR-0024 §2).
        A resolution that changed the reference is logged at DEBUG.

        Args:
            reference: What the user typed.

        Returns:
            The resolved identity.

        Raises:
            ModelNotFound: If nothing matches, or if a prefix matches more than one model.
                Ambiguity is an error rather than a choice: picking one would run different
                weights than the user meant, and they would have no way to tell.
            ProviderUnavailable: If this provider is scripted unavailable.
        """
        self._require_available()
        model = self._resolve_model(reference)
        identity = self._identity_for(model)[0]
        if reference != model.name:
            logger.debug(
                "fake.model.resolved",
                extra={
                    "reference": reference,
                    "resolved_to": model.name,
                    "identity_confidence": identity.identity_confidence.value,
                },
            )
        return identity

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one scripted generation and return the complete result.

        A cancellation token on the request has no effect: a blocking round trip offers no
        boundary at which it could take effect, which is why LoadCoach always streams and
        assembles the response itself ([spec §13](../../../docs/packages/modelrack/spec.md)).

        Args:
            request: What to generate, and how.

        Returns:
            The complete outcome. ``client_ttft_ms`` is ``UNSUPPORTED`` — a blocking call has no
            first-token moment to observe, and reporting the total here would invent one.

        Raises:
            CapabilityUnsupported: If the request needs something this provider has not declared.
            ModelNotFound: If the catalogue has nothing under the requested name.
            ContextLimitExceeded: If the request needs more context than the profile allows.
            ProviderTimeout: If the scripted delays exceed the effective limit.
            ProviderUnavailable: If this provider is scripted unavailable.
            ProviderError: Whatever a scripted failure asks for.
        """
        plan, limit_ms = self._prepare(request, streaming=False)
        if plan.failure is not None:
            step_index = plan.failure_step if plan.failure_step is not None else 0
            elapsed_ms = plan.elapsed_ms_before(step_index)
            self._sleep_ms(elapsed_ms)
            raise failure_error(
                plan,
                request,
                elapsed_ms=elapsed_ms,
                chunks_delivered=step_index,
                limit_ms=limit_ms,
            )
        total_ms = plan.total_delay_ms
        if total_ms > limit_ms:
            self._sleep_ms(limit_ms)
            raise timeout_error(total_ms, limit_ms)
        self._sleep_ms(total_ms)
        return self._assemble(plan, wall_ms=total_ms, ttft_ms=UNSUPPORTED)

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        """Run one scripted generation, yielding events as they arrive.

        Yields zero or more deltas — reasoning first, then the answer, then tool calls — followed
        by exactly one terminal event and nothing after it.

        Everything that can fail *before* the stream starts is checked here, eagerly, and raises
        from this call rather than from the first iteration: a refused connection or an unknown
        model has no stream to terminate. Everything that fails once deltas are flowing, including
        the caller's own cancellation, arrives as
        :class:`~modelrack.streaming.StreamFailed` with the partial text beside it.

        Args:
            request: What to generate, and how. Its ``cancel`` token is honoured here, taking
                effect within one delta.

        Yields:
            Deltas, then one terminal event.

        Raises:
            CapabilityUnsupported: If streaming, or anything else the request needs, is not
                declared.
            ModelNotFound: If the catalogue has nothing under the requested name.
            ContextLimitExceeded: If the request needs more context than the profile allows.
            ProviderUnavailable: If this provider is scripted unavailable.
            ProviderError: Whatever a failure scripted with ``after_chunks=None`` asks for.
        """
        plan, limit_ms = self._prepare(request, streaming=True)
        if plan.failure is not None and plan.failure_step is None:
            raise failure_error(
                plan, request, elapsed_ms=0.0, chunks_delivered=0, limit_ms=limit_ms
            )
        return self._walk(plan, request, limit_ms)

    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult:
        """Mark a model resident, reporting whether it already was.

        Args:
            identity: The model to load.
            profile: How it should be loaded and served. Recorded as its hash on the result,
                because the same weights under a different profile are a different measurement
                subject (ADR-0023).

        Returns:
            What happened. ``load_ms`` is ``UNSUPPORTED`` when the model was already resident —
            no load happened, and ``0`` would claim an instantaneous one, which is exactly the
            reading that turns a warm run into a cold-start figure an order of magnitude wrong.

        Raises:
            CapabilityUnsupported: If ``force_unload`` is not declared. The normative
                capability set has no separate "can load" flag —
                ADR-0007 rule 2 fixes the field
                list — so ``force_unload`` is read as the single statement that residency is
                controllable at all. Inventing a fourteenth flag here would change a dataclass
                three applications code against; Phase 4 revisits it against a real provider that
                cannot control residency.
            ModelNotFound: If the catalogue has nothing under that name.
            ProviderUnavailable: If this provider is scripted unavailable.
        """
        self._require_available()
        self._require_capability("force_unload", "load a model on demand")
        model = self._require_model(identity.provider_model_name)
        with self._lock:
            already_resident = model.name in self._resident
            self._resident.add(model.name)
        return LoadResult(
            identity=self._identity_for(model)[0],
            already_resident=already_resident,
            load_ms=UNSUPPORTED if already_resident else model.load_ms,
            profile_hash=profile.profile_hash,
        )

    def unload(self, identity: ModelIdentity) -> bool:
        """Evict a model from memory.

        Args:
            identity: The model to evict.

        Returns:
            ``True`` if it was resident and has been evicted, ``False`` if it was not resident to
            begin with — the state the caller wanted, and therefore not a failure.

        Raises:
            CapabilityUnsupported: If ``force_unload`` is not declared.
            ModelNotFound: If the catalogue has nothing under that name.
            ProviderUnavailable: If this provider is scripted unavailable.
        """
        self._require_available()
        self._require_capability("force_unload", "evict a model on demand")
        model = self._require_model(identity.provider_model_name)
        with self._lock:
            was_resident = model.name in self._resident
            self._resident.discard(model.name)
        return was_resident

    def list_resident(self) -> Sequence[ResidentModel]:
        """List what is currently held in memory.

        Returns:
            One entry per resident model, ordered by name so two runs agree. Empty when nothing is
            loaded, which is different from being unable to answer — that raises instead.
            ``expires_at`` is always ``None``: the fake schedules no eviction, and inventing a
            deadline would let a consumer's expiry display pass without a provider that has one.

        Raises:
            CapabilityUnsupported: If ``residency_query`` is not declared.
            ProviderUnavailable: If this provider is scripted unavailable.
        """
        self._require_available()
        self._require_capability("residency_query", "report which models are resident")
        with self._lock:
            resident = sorted(self._resident)
        # Every resident name came from `_require_model`, so it is a catalogue name and the
        # lookup below cannot miss. No `if name in by_name` guard: defensive code for a state
        # that cannot occur is code no test can reach and no reader can evaluate.
        by_name = {model.name: model for model in self._script.models}
        return tuple(
            ResidentModel(
                identity=self._identity_for(by_name[name])[0],
                vram_bytes=by_name[name].vram_bytes,
                total_bytes=by_name[name].total_bytes,
            )
            for name in resident
        )

    # ------------------------------------------------------------------------------- internals

    def _require_available(self) -> None:
        """Raise unless the script says this provider can be reached at all."""
        if self._script.health_status is not ProviderStatus.UNAVAILABLE:
            return
        raise ProviderUnavailable(
            f"The fake provider at {self._script.base_url} is scripted unavailable.",
            details={
                "base_url": self._script.base_url,
                "reason": ProviderUnavailableReason.CONNECTION_REFUSED.value,
            },
        )

    def _require_capability(self, capability: str, action: str) -> None:
        """Raise unless the named capability flag is declared.

        Refused here rather than attempted and quietly degraded: an adapter that accepted a tool
        definition and dropped it would produce a result whose ``finish_reason`` never mentions
        tools, and the caller would read that as the model choosing not to call one
        (ADR-0007 rule 2).
        """
        if getattr(self._script.capabilities, capability):
            return
        raise CapabilityUnsupported(
            f"This provider does not declare {capability!r} and cannot {action}. Check "
            "capabilities() and branch, rather than assuming.",
            details={"capability": capability},
        )

    def _require_model(self, name: str) -> FakeModel:
        """Return the catalogue entry reachable under an exact name or alias, or raise."""
        for model in self._script.models:
            if name == model.name or name in model.aliases:
                return model
        raise ModelNotFound(
            f"No model named {name!r} is served by this provider.",
            details={"reference": name, "known_model_count": len(self._script.models)},
        )

    def _resolve_model(self, reference: str) -> FakeModel:
        """Return the single model a reference names, or raise if none or several do."""
        for model in self._script.models:
            if reference == model.name or reference in model.aliases:
                return model
        prefixed = [
            model
            for model in self._script.models
            if any(name.startswith(reference) for name in (model.name, *model.aliases))
        ]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            raise ModelNotFound(
                f"{reference!r} is a prefix of {len(prefixed)} models "
                f"({', '.join(model.name for model in prefixed)}); it names none of them. Give "
                "enough of the name to pick one — resolving an ambiguous reference by choosing "
                "would run weights you did not ask for.",
                details={
                    "reference": reference,
                    "known_model_count": len(self._script.models),
                    "matched_model_count": len(prefixed),
                },
            )
        raise ModelNotFound(
            f"No model matching {reference!r} is served by this provider.",
            details={"reference": reference, "known_model_count": len(self._script.models)},
        )

    def _identity_for(self, model: FakeModel) -> tuple[ModelIdentity, str | None]:
        """Return the model's identity and, when a digest was discarded, why.

        The digest a script writes down is whatever a provider might report — bare hex, prefixed,
        uppercase, truncated, or not hexadecimal at all. It goes through
        :func:`baseaicore.normalize_digest`, and one that will not normalize is dropped with a
        reason rather than stored malformed, yielding a ``name_only`` identity that carries the
        permanent caveat it has earned
        (ADR-0024 §2).
        """
        normalized = normalize_digest(model.digest)
        if model.digest is not None and normalized is None:
            reason = (
                f"provider reported digest {model.digest!r}, which is not 'sha256:' followed by "
                "64 hex characters; discarded, identity is name_only"
            )
            return ModelIdentity(self.kind, model.name), reason
        return ModelIdentity(self.kind, model.name, artifact_digest=normalized), None

    def _describe(self, model: FakeModel) -> ModelDescriptor:
        """Return a descriptor for one catalogue entry, observed at the injected clock's instant."""
        identity, discarded_reason = self._identity_for(model)
        raw: dict[str, Any] = (
            dict(model.raw)
            if model.raw
            else {
                "provider": "fake",
                "name": model.name,
                "digest": model.digest,
                "aliases": list(model.aliases),
            }
        )
        if discarded_reason is not None:
            # Always recorded, even over a script-supplied `raw`: spec §11.9 requires the reason a
            # digest was discarded to be recoverable, and a catalogue entry that supplied its own
            # payload is exactly the case where it would otherwise vanish.
            raw["digest_discarded_reason"] = discarded_reason
            logger.debug(
                "fake.model.digest_discarded",
                extra={"model": model.name, "reason": discarded_reason},
            )
        return ModelDescriptor(
            identity=identity,
            observed_at=self._clock(),
            family=model.family,
            architecture=model.architecture,
            parameter_count=model.parameter_count,
            active_parameter_count=model.active_parameter_count,
            expert_count=model.expert_count,
            quantization=model.quantization,
            weight_format=model.weight_format,
            size_bytes=model.size_bytes,
            max_context=model.max_context,
            embedding_dim=model.embedding_dim,
            layers=model.layers,
            attention_heads=model.attention_heads,
            kv_heads=model.kv_heads,
            head_dim=model.head_dim,
            vocab_size=model.vocab_size,
            rope_config=model.rope_config,
            sliding_window=model.sliding_window,
            declared_capabilities=model.declared_capabilities,
            license_text=model.license_text,
            raw=raw,
        )

    def _next_generation(self) -> tuple[FakeGeneration, int]:
        """Return the generation this call should run, and its call index.

        Raises:
            ValidationError: If the script has run out and does not repeat its final generation.
                Deliberately loud: a workflow that made more model calls than its script accounts
                for is the defect under test, and a fake that kept answering would hide it.
        """
        generations = self._script.generations
        with self._lock:
            index = self._generation_index
            self._generation_index = index + 1
        if index < len(generations):
            return generations[index], index
        if not self._script.repeat_final_generation:
            raise ValidationError(
                f"The script describes {len(generations)} generations and this is call "
                f"{index + 1}. Add a generation, or set repeat_final_generation=True to let the "
                "last one answer every call after it.",
                details={"generation_count": len(generations), "call_index": index},
            )
        return generations[-1], index

    def _effective_limit_ms(self, request: GenerationRequest) -> float:
        """Return the timeout for this call in milliseconds, defaulting rather than disabling."""
        seconds = (
            request.timeout_seconds
            if request.timeout_seconds is not None
            else self._default_timeout_seconds
        )
        return seconds * MILLISECONDS_PER_SECOND

    def _sleep_ms(self, duration_ms: float) -> None:
        """Spend a simulated duration, in real time only when a sleep function was injected."""
        if self._sleep is not None and duration_ms > 0:
            self._sleep(duration_ms / MILLISECONDS_PER_SECOND)

    def _check_context(self, request: GenerationRequest, prompt_tokens: int) -> None:
        """Raise if the caller's chosen context cannot hold the prompt and its answer.

        Checked only when the caller actually set a context, and therefore only on a provider that
        declared ``context_configurable`` — the flag having already been enforced. This is the
        honest half of ADR-0023 §4: a
        provider that accepts a served context is a provider that can be asked for more than it
        will serve.
        """
        context_size = request.runtime_profile.context_size
        if context_size is None:
            return
        requested = prompt_tokens + (request.sampling.max_output_tokens or 0)
        if requested <= context_size:
            return
        raise ContextLimitExceeded(
            f"The request needs about {requested} tokens of context and the runtime profile "
            f"serves {context_size}.",
            details={"requested_tokens": requested, "maximum_tokens": context_size},
        )

    def _prepare(self, request: GenerationRequest, *, streaming: bool) -> tuple[_Plan, float]:
        """Run every check a call must survive, consume a generation, and plan the whole call.

        The order is deliberate. Capability refusals come first because they cost no request at
        all — this is the adapter refusing on the provider's declared behalf. The model lookup and
        the context check follow, and only then is a generation consumed, so a call rejected
        before it could have reached a provider does not eat a step of the script.
        """
        self._require_available()
        if streaming:
            self._require_capability("streaming", "stream a generation")
        if request.tools:
            self._require_capability("tool_calling", "accept tool definitions")
        response_format = request.response_format
        if response_format is not None:
            if response_format.kind is ResponseFormatKind.JSON:
                self._require_capability("json_mode", "constrain its output to JSON")
            elif response_format.kind is ResponseFormatKind.JSON_SCHEMA:
                self._require_capability("structured_output", "enforce a JSON Schema")
        if request.runtime_profile.context_size is not None:
            self._require_capability("context_configurable", "serve a caller-chosen context")
        model = self._require_model(request.identity.provider_model_name)
        prompt_text = _render_prompt(request)
        prompt_tokens = _simulated_token_count(prompt_text)
        self._check_context(request, prompt_tokens)
        generation, generation_index = self._next_generation()
        plan = self._build_plan(
            request, model, generation, generation_index, prompt_text, prompt_tokens
        )
        return plan, self._effective_limit_ms(request)

    def _build_plan(
        self,
        request: GenerationRequest,
        model: FakeModel,
        generation: FakeGeneration,
        generation_index: int,
        prompt_text: str,
        prompt_tokens: int,
    ) -> _Plan:
        """Compute everything the call will produce, once, for both ``generate`` and ``stream``."""
        capabilities = self._script.capabilities
        identity = self._identity_for(model)[0]
        seed_material = "\x00".join(
            (
                str(self._seed),
                identity.canonical_id,
                str(request.sampling.seed),
                str(generation_index),
                prompt_text,
            )
        )
        text, chunks, truncated = _planned_text(request, generation, seed_material)
        thinking: str | Unsupported = (
            generation.thinking
            if capabilities.thinking_control and generation.thinking is not None
            else UNSUPPORTED
        )
        tool_calls, argument_texts = _planned_tool_calls(generation, generation_index)
        if generation.finish_reason is not None:
            finish_reason = generation.finish_reason
        elif tool_calls:
            finish_reason = FinishReason.TOOL_CALLS
        elif truncated:
            finish_reason = FinishReason.LENGTH
        else:
            finish_reason = FinishReason.STOP
        steps = _planned_steps(generation, chunks, thinking, tool_calls, argument_texts)
        failure = generation.failure
        failure_step = (
            None
            if failure is None or failure.after_chunks is None
            else min(failure.after_chunks, len(steps))
        )
        return _Plan(
            identity=identity,
            model=model,
            base_url=self._script.base_url,
            model_count=len(self._script.models),
            steps=steps,
            text=text,
            thinking=thinking,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=self._planned_usage(generation, text, thinking, argument_texts, prompt_tokens),
            backend_timing=generation.backend_timing
            if generation.backend_timing is not None
            else Timing(),
            prompt_tokens=prompt_tokens,
            provider_version=self._script.provider_version,
            failure=failure,
            failure_step=failure_step,
            raw={
                # No prompt, no caller metadata and no generated text. The first two are the
                # caller's to hold and are never sent to a provider (spec §7); the third would
                # mean a streamed response was accumulated twice, against the flat per-stream
                # memory budget in spec §15. What is here is what explains a surprising result.
                "provider": "fake",
                "provider_version": self._script.provider_version,
                "model": model.name,
                "generation_index": generation_index,
                "seed": self._seed,
                "finish_reason": finish_reason.value,
                "chunk_count": len(chunks),
                "tool_call_count": len(tool_calls),
                "prompt_tokens": prompt_tokens,
            },
        )

    def _planned_usage(
        self,
        generation: FakeGeneration,
        text: str,
        thinking: str | Unsupported,
        argument_texts: tuple[str, ...],
        prompt_tokens: int,
    ) -> GenerationUsage:
        """Compute what the call consumed, gated by what this provider declares it can count.

        ``output_chars``, ``output_words`` and ``output_bytes`` are reported even when
        ``token_counts`` is undeclared. They are not the provider's counts — they are observations
        of a string this process is holding, and a caller could compute them itself. Withholding
        them would be the fake pretending not to know something it plainly does.

        ``thinking_tokens`` and ``tool_tokens`` are breakdowns *of* ``output_tokens``, never a
        fifth and sixth billing class: every provider that exposes reasoning tokens bills them at
        its output rate (ADR-0030), so adding
        them to a total computed from ``tokens`` would count them twice.
        """
        capabilities = self._script.capabilities
        observed_chars = len(text)
        observed_words = len(text.split())
        observed_bytes = len(text.encode("utf-8"))
        if not capabilities.token_counts:
            return GenerationUsage(
                output_chars=observed_chars,
                output_words=observed_words,
                output_bytes=observed_bytes,
            )
        thinking_text = thinking if isinstance(thinking, str) else ""
        tool_text = "".join(argument_texts)
        cache_read: TokenCount = (
            generation.cache_read_tokens
            if generation.cache_read_tokens is not None
            else UNSUPPORTED
        )
        cache_write: TokenCount = (
            generation.cache_write_tokens
            if generation.cache_write_tokens is not None
            else UNSUPPORTED
        )
        if generation.input_tokens is not None:
            input_tokens: TokenCount = generation.input_tokens
        else:
            # The four billing classes are disjoint: a token billed at the cache-hit rate is not
            # also billed at the input rate. Reconciling a provider's overlapping figures into
            # that shape is the adapter's job (ADR-0030), so the fake does it too.
            billed = prompt_tokens - (cache_read if is_supported(cache_read) else 0)
            input_tokens = max(billed, 0)
        output_tokens: TokenCount = (
            generation.output_tokens
            if generation.output_tokens is not None
            else _simulated_token_count(text)
            + _simulated_token_count(thinking_text)
            + _simulated_token_count(tool_text)
        )
        return GenerationUsage(
            tokens=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
            ),
            # UNSUPPORTED rather than 0 when there is no reasoning content, because
            # `thinking` is UNSUPPORTED in that case too: reporting that a provider counted
            # zero tokens of something it did not report at all is a contradiction. Tool
            # tokens differ — `tool_calls` is an empty tuple, which says calls were reported
            # and there were none, so 0 there is a real count.
            thinking_tokens=(
                _simulated_token_count(thinking_text) if isinstance(thinking, str) else UNSUPPORTED
            ),
            tool_tokens=(
                _simulated_token_count(tool_text) if capabilities.tool_calling else UNSUPPORTED
            ),
            output_chars=observed_chars,
            output_words=observed_words,
            output_bytes=observed_bytes,
        )

    def _assemble(
        self, plan: _Plan, *, wall_ms: Measurement, ttft_ms: Measurement
    ) -> GenerationResult:
        """Build the result for a call that completed, joining the plan to the observed timings."""
        backend = plan.backend_timing
        return GenerationResult(
            text=plan.text,
            identity=plan.identity,
            finish_reason=plan.finish_reason,
            usage=plan.usage,
            timing=Timing(
                client_wall_ms=wall_ms,
                client_ttft_ms=ttft_ms,
                backend_load_ms=backend.backend_load_ms,
                backend_prompt_eval_ms=backend.backend_prompt_eval_ms,
                backend_decode_ms=backend.backend_decode_ms,
                backend_total_ms=backend.backend_total_ms,
            ),
            tool_calls=plan.tool_calls,
            thinking=plan.thinking,
            provider_version=plan.provider_version,
            raw=plan.raw,
        )

    def _cancelled(self, partial_text: str) -> StreamFailed:
        """Return the terminal event for a stream the caller stopped, with its output attached.

        Delivered rather than raised. A raise mid-drain ends the iterator with no terminal event,
        which is exactly how :mod:`modelrack.streaming` defines a *truncated* stream — so raising
        here would make "the caller stopped it" indistinguishable from "the connection dropped".
        The partial text is the caller's own output being handed back, which is why this is the
        one error whose ``details`` may carry generated content.
        """
        return StreamFailed(
            error=GenerationCancelled(
                "Generation was cancelled by the caller's token.",
                details={"partial_text": partial_text},
            ),
            partial_text=partial_text,
        )

    def _walk(
        self, plan: _Plan, request: GenerationRequest, limit_ms: float
    ) -> Iterator[StreamEvent]:
        """Yield the planned deltas, then exactly one terminal event, and nothing after it."""
        cancel = request.cancel
        answer: list[str] = []
        elapsed_ms = 0.0
        for index, step in enumerate(plan.steps):
            if cancel is not None and cancel.is_cancelled:
                yield self._cancelled("".join(answer))
                return
            if plan.failure is not None and plan.failure_step == index:
                yield StreamFailed(
                    error=failure_error(
                        plan,
                        request,
                        elapsed_ms=elapsed_ms,
                        chunks_delivered=index,
                        limit_ms=limit_ms,
                    ),
                    partial_text="".join(answer),
                )
                return
            elapsed_ms += step.delay_ms
            self._sleep_ms(step.delay_ms)
            if elapsed_ms > limit_ms:
                yield StreamFailed(
                    error=timeout_error(elapsed_ms, limit_ms), partial_text="".join(answer)
                )
                return
            # Checked again after the delay: a token set while the caller was waiting on a real
            # sleep has to take effect on this delta, not the next one. This is what "within one
            # chunk boundary" (spec §11.6) costs to actually deliver.
            if cancel is not None and cancel.is_cancelled:
                yield self._cancelled("".join(answer))
                return
            yield step.event
            if isinstance(step.event, TokenDelta):
                answer.append(step.event.text)
        if cancel is not None and cancel.is_cancelled:
            yield self._cancelled("".join(answer))
            return
        if plan.failure is not None:
            yield StreamFailed(
                error=failure_error(
                    plan,
                    request,
                    elapsed_ms=elapsed_ms,
                    chunks_delivered=len(plan.steps),
                    limit_ms=limit_ms,
                ),
                partial_text="".join(answer),
            )
            return
        yield StreamCompleted(
            result=self._assemble(plan, wall_ms=elapsed_ms, ttft_ms=plan.first_delay_ms)
        )
