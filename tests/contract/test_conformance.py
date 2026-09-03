"""The provider conformance suite: one set of behaviours every adapter must exhibit.

[Spec §11.5](../../docs/packages/modelrack/spec.md) — *every adapter passes the same conformance
suite* — is the sentence this file exists to make true, and
testing standards §7 states it generally: a new
provider passes the same suite or it is not a provider.

:class:`ProviderConformanceSuite` holds the behaviours; a subclass binds them to one adapter by
supplying two fixtures. Phase 2 binds the fake twice, once with everything declared and once with
nothing, because half of what this suite checks is the *refusal* path and a suite that only ever
ran against a capable provider would never execute it. Phase 3 adds a class for the recorded
Ollama transport and Phase 4 one for the recorded OpenAI-compatible transport; neither needs a
line of this file changed.

**Capability-gated behaviours are never silently skipped.** Where a capability is declared, the
behaviour is exercised; where it is not, the suite asserts the adapter *refuses* with
:class:`~modelrack.errors.CapabilityUnsupported` naming the flag. That is what Phase 4 means by
"recorded as ``unsupported``, never silently passed" — a skipped test and a passing test look the
same in a summary line, and the whole point of a capability declaration is that omitting one has
consequences.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import (
    UNSUPPORTED,
    IdentityConfidence,
    ModelIdentity,
    ModelPricing,
    Money,
    PricingSource,
    ProviderKind,
    RuntimeProfile,
    TokenRates,
    estimate_cost,
    is_supported,
)

from modelrack import (
    CapabilityUnsupported,
    GenerationCancelled,
    GenerationRequest,
    Message,
    ModelNotFound,
    Provider,
    ProviderError,
    ProviderStatus,
    ResponseFormat,
    ResponseFormatKind,
    Role,
    StreamCompleted,
    StreamFailed,
    ThinkingDelta,
    TokenDelta,
    ToolCallDelta,
    ToolDefinition,
)
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider
from modelrack.streaming import CancellationToken
from modelrack.testing import (
    FULL_CAPABILITIES,
    MINIMAL_CAPABILITIES,
    FakeGeneration,
    FakeProvider,
    FakeScript,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from baseaicore import TokenUsage

    from modelrack import ProviderCapabilities, StreamEvent

pytestmark = pytest.mark.contract

_UNKNOWN_REFERENCE = "no-such-model:v0"
_MINIMUM_DELTAS_FOR_CANCELLATION = 3

_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class UsageShape(enum.Enum):
    """The three response shapes ADR-0070 makes every adapter account for.

    Named rather than positional because a subclass declares them by name (see
    :class:`UsageShapes`), and a fourth adapter reading this file should not have to work out
    which element of a tuple meant what.
    """

    NO_CACHE_DETAIL = "no_cache_detail"
    """A usage report from a server that does no cache accounting: both cache classes are `0`."""

    CACHE_DETAIL = "cache_detail"
    """A usage report carrying cached input, reconciled into the disjoint classes."""

    NO_USAGE_OBJECT = "no_usage_object"
    """A response reporting no counts at all: every class is `UNSUPPORTED`."""


@dataclasses.dataclass(frozen=True, slots=True)
class CacheDetailShape:
    """What one adapter's :attr:`UsageShape.CACHE_DETAIL` response claims, in wire terms.

    The suite asserts the *reconciliation arithmetic* — that the disjoint classes sum back to the
    provider's own prompt figure — and it cannot read that figure off the result, because the
    result is exactly what the reconciliation has already rewritten. So the adapter declares it.

    Attributes:
        prompt_tokens: The provider's own prompt figure, the total that includes the cached
            tokens. ``input_tokens + cache_read_tokens`` must equal it.
        cached_tokens: The cached portion the provider reported inside that total.
    """

    prompt_tokens: int
    cached_tokens: int

    def __post_init__(self) -> None:
        """Raise if the declaration is not one a real response could carry."""
        if not 0 <= self.cached_tokens <= self.prompt_tokens:
            raise ValueError(
                "CacheDetailShape declares cached_tokens outside 0..prompt_tokens; a response "
                "shaped like that is the adapter's refusal case, not its reconciliation case."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class UsageShapes:
    """How one adapter is driven into each of ADR-0070's three response shapes.

    **This is the seam a new adapter binds to.** Row D3 adds ``LlamaCppProvider`` to this suite by
    writing one ``usage_shapes`` fixture — declaring which recorded response produces each shape,
    and what its cache-detail response claims — and changes nothing in the three behaviours below.
    That is the whole reason the usage rule lands before the third adapter rather than after it.

    Attributes:
        provider_for: Returns a provider whose next ``generate`` produces the named shape. For a
            recorded adapter this selects which fixture the transport serves and returns the same
            provider; for :class:`~modelrack.testing.FakeProvider` it returns a freshly scripted
            one. Either is fine — the suite only ever calls ``generate`` on what it is handed.
        cache_detail: What the ``CACHE_DETAIL`` response claims, or ``None`` when the adapter's
            wire protocol has **no way to report cached input at all**. ``None`` is a declaration,
            not a skip: Ollama's protocol has no cache-billing vocabulary, which is precisely why
            its cache classes are `0` rather than reconciled, and an adapter that could report
            cache detail but declared ``None`` here would be exempting itself from the one case
            that catches double-billed cached input.
    """

    provider_for: Callable[[UsageShape], Provider]
    cache_detail: CacheDetailShape | None


# A price list with input and output rates and no cache rates — the local-model case, and the case
# of a published list that predates a provider's cache pricing. It is the whole point of ADR-0070
# that this list can total a real response: `estimate_cost` refuses a component whose count is
# UNSUPPORTED, but a count of exactly `0` costs exactly nothing whether or not a rate exists for
# it, so cache classes reported as `0` leave the total priced instead of turning it into a floor.
_PRICED_AT = datetime(2026, 8, 22, 14, 0, tzinfo=UTC)
_UNCACHED_PRICING = ModelPricing(
    identity=ModelIdentity(ProviderKind.OPENAI_COMPATIBLE, "conformance-priced-model"),
    rates=TokenRates(
        currency="USD",
        input_per_million_tokens=Money.from_decimal("USD", "3.00"),
        output_per_million_tokens=Money.from_decimal("USD", "15.00"),
    ),
    source=PricingSource.PROVIDER_PUBLISHED,
    observed_at=datetime(2026, 8, 1, tzinfo=UTC),
)


def _deltas(events: Sequence[StreamEvent]) -> list[TokenDelta]:
    """Return only the answer-text deltas from a drained stream."""
    return [event for event in events if isinstance(event, TokenDelta)]


class ProviderConformanceSuite:
    """Behaviours every :class:`~modelrack.Provider` implementation must exhibit.

    Not collected by pytest — the name does not begin with ``Test`` — so a subclass is what runs
    it. A subclass supplies two fixtures:

    * ``provider``: a fresh implementation, function-scoped, so a test that consumes a scripted
      response or loads a model cannot influence the next one.
    * ``known_reference``: a model reference the provider can resolve.

    and must satisfy one precondition: the standard request must produce at least
    :data:`_MINIMUM_DELTAS_FOR_CANCELLATION` deltas on a provider that declares ``streaming``,
    without which "cancellation takes effect within one delta" has nothing to measure.
    """

    @pytest.fixture
    def provider(self) -> Provider:
        """Return the implementation under test. Overridden by every subclass."""
        raise NotImplementedError

    @pytest.fixture
    def known_reference(self) -> str:
        """Return a model reference this provider can resolve. Overridden by every subclass."""
        raise NotImplementedError

    @pytest.fixture
    def identity(self, provider: Provider, known_reference: str) -> ModelIdentity:
        """Return the identity the provider resolves the known reference to."""
        return provider.resolve(known_reference)

    @pytest.fixture
    def capabilities(self, provider: Provider) -> ProviderCapabilities:
        """Return what the provider declares, so a test can branch the way a caller would."""
        return provider.capabilities()

    @staticmethod
    def request(identity: ModelIdentity, **overrides: Any) -> GenerationRequest:
        """Build the standard request every behaviour below is exercised with."""
        fields: dict[str, Any] = {
            "identity": identity,
            "messages": (Message(role=Role.USER, content="Explain KV caching briefly."),),
        }
        fields.update(overrides)
        return GenerationRequest(**fields)

    # --------------------------------------------------------------------- health, capabilities

    def test_health_names_where_it_probed(self, provider: Provider) -> None:
        health = provider.health()

        assert health.base_url.strip()
        assert isinstance(health.status, ProviderStatus)

    def test_health_never_raises_however_the_probe_goes(self, provider: Provider) -> None:
        """A negative answer is not an exceptional condition. An application's health endpoint
        calls this precisely when it expects the answer might be no, and an adapter that raised
        would turn a "provider is down" line in a health document into a 500 for the whole
        endpoint.
        """
        health = provider.health()

        assert health.status in set(ProviderStatus)
        assert health.base_url

    def test_capabilities_are_a_stable_declaration_not_a_probe(self, provider: Provider) -> None:
        """Asked twice without a call in between, a declaration cannot have changed its mind."""
        assert provider.capabilities() == provider.capabilities()

    def test_token_level_chunks_implies_streaming(self, capabilities: ProviderCapabilities) -> None:
        """A provider with no incremental output cannot be emitting one token per delta."""
        if capabilities.token_level_chunks:
            assert capabilities.streaming

    # ------------------------------------------------------------------ discovery and identity

    def test_list_models_describes_what_it_serves(self, provider: Provider) -> None:
        descriptors = provider.list_models()

        assert all(descriptor.identity.provider_model_name for descriptor in descriptors)
        assert all(descriptor.observed_at.tzinfo is not None for descriptor in descriptors)

    def test_every_digest_is_normalized_or_absent(self, provider: Provider) -> None:
        """Spec §11.9: one shape reaches storage, and a name-only identity says so.

        A malformed digest cannot exist here at all — ``ModelIdentity`` rejects one — so what this
        proves is the second half: an adapter that could not normalize what a provider reported
        produced a ``name_only`` identity rather than failing the listing.
        """
        for descriptor in provider.list_models():
            digest = descriptor.identity.artifact_digest
            expected = (
                IdentityConfidence.DIGEST if digest is not None else IdentityConfidence.NAME_ONLY
            )
            assert descriptor.identity.identity_confidence is expected

    def test_descriptors_preserve_the_provider_payload(self, provider: Provider) -> None:
        """``raw`` exists so a surprising result can be explained, and is diagnostics only."""
        assert all(descriptor.raw for descriptor in provider.list_models())

    def test_resolve_returns_an_identity_the_provider_serves(
        self, provider: Provider, known_reference: str
    ) -> None:
        identity = provider.resolve(known_reference)

        served = {descriptor.identity for descriptor in provider.list_models()}
        assert identity in served

    def test_resolve_of_an_unknown_reference_names_what_it_looked_for(
        self, provider: Provider
    ) -> None:
        with pytest.raises(ModelNotFound) as raised:
            provider.resolve(_UNKNOWN_REFERENCE)

        assert raised.value.details["reference"] == _UNKNOWN_REFERENCE
        assert "known_model_count" in raised.value.details

    def test_inspect_returns_the_identity_currently_served(
        self, provider: Provider, identity: ModelIdentity
    ) -> None:
        """A retag is surfaced, not hidden: the descriptor carries what is served now."""
        descriptor = provider.inspect_model(identity)

        assert descriptor.identity.provider_model_name == identity.provider_model_name

    def test_inspect_of_an_unknown_model_raises_model_not_found(
        self, provider: Provider, identity: ModelIdentity
    ) -> None:
        unknown = dataclasses.replace(
            identity, provider_model_name=_UNKNOWN_REFERENCE, artifact_digest=None
        )

        with pytest.raises(ModelNotFound):
            provider.inspect_model(unknown)

    # ------------------------------------------------------------------------------ generation

    def test_generate_returns_the_model_that_ran(
        self, provider: Provider, identity: ModelIdentity
    ) -> None:
        result = provider.generate(self.request(identity))

        assert result.identity.provider_model_name == identity.provider_model_name

    def test_generate_reports_no_first_token_moment(
        self, provider: Provider, identity: ModelIdentity
    ) -> None:
        """A blocking round trip has no moment at which a first token could be observed."""
        result = provider.generate(self.request(identity))

        assert not is_supported(result.timing.client_ttft_ms)

    def test_token_counts_follow_the_declaration(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """Undeclared counting means ``UNSUPPORTED``, never a plausible-looking zero."""
        usage = provider.generate(self.request(identity)).usage

        if capabilities.token_counts:
            assert is_supported(usage.tokens.output_tokens)
        else:
            assert not is_supported(usage.tokens.output_tokens)
            assert not is_supported(usage.tokens.input_tokens)

    # ------------------------------------------------------------------- usage shapes (ADR-0070)

    @pytest.fixture
    def usage_shapes(self) -> UsageShapes | None:
        """Declare how this adapter reaches each of ADR-0070's three response shapes.

        Overridden by every subclass — including one that returns ``None``, which declares that
        this configuration reports no token counts at all and sends the three behaviours below to
        their refusal branch instead. There is no default: an adapter added to this suite must say
        which shapes its protocol produces, because "it was never asked" and "it answered `0`" are
        exactly the two things ADR-0070 exists to keep apart.
        """
        raise NotImplementedError

    def _tokens_for(self, provider: Provider, known_reference: str) -> TokenUsage:
        """Return the billing classes one generation from ``provider`` reported."""
        return provider.generate(self.request(provider.resolve(known_reference))).usage.tokens

    def _assert_reports_no_counts_at_all(
        self, provider: Provider, known_reference: str, capabilities: ProviderCapabilities
    ) -> None:
        """Assert the shape a provider that declares no token counting must report.

        The refusal branch of the three behaviours below, kept as an assertion rather than a skip:
        a provider that counts nothing still has to report *nothing* — four ``UNSUPPORTED``
        classes — rather than the plausible-looking zeros ADR-0016 forbids.
        """
        assert not capabilities.token_counts
        tokens = self._tokens_for(provider, known_reference)
        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.output_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)

    def test_a_response_without_cache_detail_bills_no_cache_and_still_totals(
        self,
        usage_shapes: UsageShapes | None,
        provider: Provider,
        known_reference: str,
        capabilities: ProviderCapabilities,
    ) -> None:
        """A class the protocol could not have billed is `0`, and `0` is a number that totals.

        The observable improvement ADR-0070 exists for, asserted where it can be seen rather than
        only described in a changelog: with the cache classes reported as `0`, ``total_tokens``
        returns a number and a price list carrying no cache rates still produces a **total** for
        a real response instead of the floor ADR-0069 would otherwise have to label it.
        """
        if usage_shapes is None:
            self._assert_reports_no_counts_at_all(provider, known_reference, capabilities)
            return
        shaped = usage_shapes.provider_for(UsageShape.NO_CACHE_DETAIL)
        tokens = self._tokens_for(shaped, known_reference)

        assert tokens.cache_read_tokens == 0
        assert tokens.cache_write_tokens == 0
        assert is_supported(tokens.total_tokens)
        estimate = estimate_cost(tokens, _UNCACHED_PRICING, at=_PRICED_AT)
        assert is_supported(estimate.total)
        assert estimate.unpriced_reasons == ()

    def test_a_response_with_cache_detail_reports_disjoint_classes(
        self,
        usage_shapes: UsageShapes | None,
        provider: Provider,
        known_reference: str,
        capabilities: ProviderCapabilities,
    ) -> None:
        """Cached input is reconciled out of the input class, never billed twice.

        The arithmetic is the assertion: ``input_tokens + cache_read_tokens`` must come back to
        the provider's own prompt figure. An adapter that simply copied ``prompt_tokens`` into
        ``input_tokens`` and reported the cached figure beside it would pass a type check and
        over-bill every cached call at the full input rate — the latent defect ADR-0070 names.
        """
        if usage_shapes is None:
            self._assert_reports_no_counts_at_all(provider, known_reference, capabilities)
            return
        declared = usage_shapes.cache_detail
        if declared is None:
            pytest.skip(
                "declared: this wire protocol cannot report cached input at all, so the "
                "no-cache-detail behaviour above is the only shape it has"
            )
        shaped = usage_shapes.provider_for(UsageShape.CACHE_DETAIL)
        tokens = self._tokens_for(shaped, known_reference)

        assert tokens.cache_read_tokens == declared.cached_tokens
        assert tokens.input_tokens == declared.prompt_tokens - declared.cached_tokens
        assert tokens.input_tokens + tokens.cache_read_tokens == declared.prompt_tokens
        assert tokens.cache_write_tokens == 0

    def test_a_response_with_no_usage_object_reports_every_class_unsupported(
        self,
        usage_shapes: UsageShapes | None,
        provider: Provider,
        known_reference: str,
        capabilities: ProviderCapabilities,
    ) -> None:
        """Nothing reported is not zero reported — the boundary the whole rule turns on.

        This is the case a fabricated zero would hide in: an adapter that answered `0` here would
        report a free call for a response that told it nothing, and every consumer downstream
        would total it without a murmur.
        """
        if usage_shapes is None:
            self._assert_reports_no_counts_at_all(provider, known_reference, capabilities)
            return
        shaped = usage_shapes.provider_for(UsageShape.NO_USAGE_OBJECT)
        tokens = self._tokens_for(shaped, known_reference)

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.output_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)
        assert not is_supported(tokens.total_tokens)

    def test_caller_metadata_never_reaches_the_provider_payload(
        self, provider: Provider, identity: ModelIdentity
    ) -> None:
        """Correlation IDs travel with the request and are never sent (spec §7)."""
        marker = "conformance-correlation-4a1f"
        result = provider.generate(self.request(identity, metadata={"run_id": marker}))

        assert marker not in repr(dict(result.raw))

    # ------------------------------------------------------------------------------- streaming

    def test_streaming_is_refused_when_it_is_not_declared(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        if capabilities.streaming:
            pytest.skip("declared: exercised by the streaming behaviours below")
        with pytest.raises(CapabilityUnsupported) as raised:
            list(provider.stream(self.request(identity)))

        assert raised.value.details["capability"] == "streaming"

    def test_a_stream_ends_with_exactly_one_terminal_event(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """A truncated stream is detectable as the absence of one, never as a short answer."""
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        events = list(provider.stream(self.request(identity)))

        terminal = [event for event in events if isinstance(event, StreamCompleted | StreamFailed)]
        assert len(terminal) == 1
        assert events[-1] is terminal[0]

    def test_a_stream_yields_nothing_after_its_terminal_event(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        iterator = provider.stream(self.request(identity))
        drained = list(iterator)

        assert isinstance(drained[-1], StreamCompleted | StreamFailed)
        assert list(iterator) == []

    def test_delta_indices_order_the_stream(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        events = list(provider.stream(self.request(identity)))

        indices = [
            event.index
            for event in events
            if isinstance(event, TokenDelta | ThinkingDelta | ToolCallDelta)
        ]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    def test_streamed_text_is_the_completed_result(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """A caller that streamed and one that did not must be able to record the same thing."""
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        events = list(provider.stream(self.request(identity)))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert "".join(delta.text for delta in _deltas(events)) == terminal.result.text

    def test_a_stream_reports_the_first_token_moment_it_observed(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        events = list(provider.stream(self.request(identity)))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert is_supported(terminal.result.timing.client_ttft_ms)

    def test_cancellation_takes_effect_within_one_delta(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """Spec §11.6, and the partial output is handed back rather than discarded."""
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        token = CancellationToken()
        events: list[StreamEvent] = []
        deltas_at_cancellation = 0
        for event in provider.stream(self.request(identity, cancel=token)):
            events.append(event)
            if not token.is_cancelled and len(_deltas(events)) == 2:
                token.cancel()
                deltas_at_cancellation = len(_deltas(events))

        assert deltas_at_cancellation == 2, "the provider produced too few deltas to cancel within"
        assert len(_deltas(events)) - deltas_at_cancellation <= 1
        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == "".join(delta.text for delta in _deltas(events))
        assert terminal.error.details["partial_text"] == terminal.partial_text

    def test_a_stream_produces_enough_deltas_to_cancel_within(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """The fixture precondition, asserted rather than assumed."""
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        events = list(provider.stream(self.request(identity)))

        assert len(_deltas(events)) >= _MINIMUM_DELTAS_FOR_CANCELLATION

    def test_an_abandoned_stream_leaves_the_provider_usable(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """Walking away mid-stream is ordinary; the next call must not inherit anything from it."""
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        for _ in provider.stream(self.request(identity)):
            break

        assert provider.generate(self.request(identity)).identity.provider_model_name

    def test_a_token_already_set_yields_exactly_one_terminal_event(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """Phase 5's hardening, as a contract rather than an adapter detail: a caller has one
        cancellation path, not two. Delivered rather than raised, and nothing before it.
        """
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        token = CancellationToken()
        token.cancel()

        events = list(provider.stream(self.request(identity, cancel=token)))

        assert len(events) == 1
        terminal = events[0]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == ""

    def test_a_cancelled_provider_is_still_usable_afterwards(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """Whatever the adapter released on the way out, it released cleanly."""
        if not capabilities.streaming:
            pytest.skip("not declared: refusal asserted above")
        token = CancellationToken()
        token.cancel()
        list(provider.stream(self.request(identity, cancel=token)))

        assert provider.generate(self.request(identity)).identity.provider_model_name

    # ------------------------------------------------------------------------- metadata reads

    def test_refresh_is_accepted_by_every_adapter(
        self, provider: Provider, known_reference: str
    ) -> None:
        """The ``refresh=True`` path the development plan names as the mitigation a TTL alone
        cannot provide. An adapter that caches nothing accepts it and ignores it, so a caller
        holding a :class:`~modelrack.Provider` never has to ask which kind it is holding.
        """
        cold = provider.list_models()
        warm = provider.list_models(refresh=True)

        assert [descriptor.identity for descriptor in warm] == [
            descriptor.identity for descriptor in cold
        ]
        assert provider.resolve(known_reference, refresh=True) == provider.resolve(known_reference)

    def test_a_refreshed_inspection_returns_the_same_model(
        self, provider: Provider, identity: ModelIdentity
    ) -> None:
        assert (
            provider.inspect_model(identity, refresh=True).identity.provider_model_name
            == identity.provider_model_name
        )

    # ---------------------------------------------------------------------- capability gating

    def test_tool_definitions_are_honoured_or_refused(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        request = self.request(identity, tools=(_WEATHER_TOOL,))

        if capabilities.tool_calling:
            assert provider.generate(request).identity is not None
        else:
            with pytest.raises(CapabilityUnsupported) as raised:
                provider.generate(request)
            assert raised.value.details["capability"] == "tool_calling"

    def test_json_mode_is_honoured_or_refused(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        request = self.request(
            identity, response_format=ResponseFormat(kind=ResponseFormatKind.JSON)
        )

        if capabilities.json_mode:
            assert provider.generate(request).text
        else:
            with pytest.raises(CapabilityUnsupported) as raised:
                provider.generate(request)
            assert raised.value.details["capability"] == "json_mode"

    def test_a_schema_is_honoured_or_refused(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        request = self.request(
            identity,
            response_format=ResponseFormat(
                kind=ResponseFormatKind.JSON_SCHEMA, schema=_ANSWER_SCHEMA
            ),
        )

        if capabilities.structured_output:
            assert provider.generate(request).text
        else:
            with pytest.raises(CapabilityUnsupported) as raised:
                provider.generate(request)
            assert raised.value.details["capability"] == "structured_output"

    def test_a_caller_chosen_context_is_honoured_or_refused(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        """Spec §11.10: ``context_configurable`` is load-bearing, not informational.

        An adapter that cannot configure context declares ``False`` and refuses, rather than
        accepting the setting and ignoring it — which would produce a run whose recorded context
        never happened.
        """
        request = self.request(identity, runtime_profile=RuntimeProfile(context_size=4096))

        if capabilities.context_configurable:
            assert provider.generate(request).identity is not None
        else:
            with pytest.raises(CapabilityUnsupported) as raised:
                provider.generate(request)
            assert raised.value.details["capability"] == "context_configurable"

    def test_residency_control_is_honoured_or_refused(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        if capabilities.force_unload:
            loaded = provider.load(identity, RuntimeProfile())
            assert loaded.already_resident is False
            assert provider.unload(identity) is True
            assert provider.unload(identity) is False
        else:
            with pytest.raises(CapabilityUnsupported) as raised:
                provider.unload(identity)
            assert raised.value.details["capability"] == "force_unload"

    def test_residency_query_is_honoured_or_refused(
        self, provider: Provider, identity: ModelIdentity, capabilities: ProviderCapabilities
    ) -> None:
        if capabilities.residency_query:
            assert list(provider.list_resident()) == []
        else:
            with pytest.raises(CapabilityUnsupported) as raised:
                provider.list_resident()
            assert raised.value.details["capability"] == "residency_query"

    # ----------------------------------------------------------------------------------- errors

    def test_every_failure_is_a_typed_provider_error_with_a_code(self, provider: Provider) -> None:
        """No adapter raises a raw transport exception (spec §11.7)."""
        with pytest.raises(ProviderError) as raised:
            provider.resolve(_UNKNOWN_REFERENCE)

        assert raised.value.code
        assert raised.value.code != "INTERNAL_ERROR"


def _fake_usage_shapes() -> UsageShapes:
    """Declare the three ADR-0070 shapes for :class:`~modelrack.testing.FakeProvider`.

    Shared by every fake-backed class below that declares ``token_counts``: the shapes are a
    property of the fake's scripting surface, not of the chunking or capability variations those
    classes exist to vary. Each shape is a freshly scripted provider rather than a mutated one,
    which is what :class:`FakeScript` makes cheap and a recorded transport does not.

    ``NO_USAGE_OBJECT`` is scripted by naming ``UNSUPPORTED`` on all four classes — the escape
    hatch ADR-0070 decision 5 requires the fake to keep once its unscripted cache classes became
    `0`, and the shape LoadLedger's and PromptCadence's own tests need on demand.
    """

    def provider_for(shape: UsageShape) -> Provider:
        if shape is UsageShape.CACHE_DETAIL:
            # 13 + 8 = 21, the same prompt figure the recorded OpenAI-compatible fixture claims,
            # so the two adapters are asserted against arithmetic of the same shape.
            generation = FakeGeneration(input_tokens=13, cache_read_tokens=8)
        elif shape is UsageShape.NO_USAGE_OBJECT:
            generation = FakeGeneration(
                input_tokens=UNSUPPORTED,
                output_tokens=UNSUPPORTED,
                cache_read_tokens=UNSUPPORTED,
                cache_write_tokens=UNSUPPORTED,
            )
        else:
            generation = FakeGeneration()
        return FakeProvider(FakeScript(generations=(generation,)), seed=17)

    return UsageShapes(
        provider_for=provider_for,
        cache_detail=CacheDetailShape(prompt_tokens=21, cached_tokens=8),
    )


class TestFakeProviderConformance(ProviderConformanceSuite):
    """The fake with everything it can do declared — the capable path through the suite."""

    @pytest.fixture
    def provider(self) -> Provider:
        return FakeProvider(seed=17)

    @pytest.fixture
    def known_reference(self) -> str:
        return "fake-model:8b-q8_0"

    @pytest.fixture
    def usage_shapes(self) -> UsageShapes | None:
        return _fake_usage_shapes()


class TestFakeProviderMinimalConformance(ProviderConformanceSuite):
    """The same suite against a provider that declares nothing at all.

    This is the class that stops the suite from being a capability rubber stamp: with
    :data:`~modelrack.testing.MINIMAL_CAPABILITIES` every gated behaviour takes its refusal
    branch, so the ``CapabilityUnsupported`` path — the one a real OpenAI-compatible endpoint will
    take in Phase 4 — is executed here rather than being discovered then.
    """

    @pytest.fixture
    def provider(self) -> Provider:
        return FakeProvider(
            FakeScript(
                capabilities=MINIMAL_CAPABILITIES,
                generations=(FakeGeneration(),),
            ),
            seed=17,
        )

    @pytest.fixture
    def known_reference(self) -> str:
        return "fake-model:8b-q8_0"

    @pytest.fixture
    def usage_shapes(self) -> UsageShapes | None:
        """``None``: this configuration declares no token counting, so it has no usage shapes.

        Not an exemption — the three behaviours take their refusal branch and assert that all
        four classes come back ``UNSUPPORTED``, which is the same undeclared-counter path
        ``test_token_counts_follow_the_declaration`` covers for the two classes it checks.
        """
        return None


class TestFakeProviderChunkedConformance(ProviderConformanceSuite):
    """The fake streaming fragments that are honestly *not* one token each.

    Runs the whole suite against a provider whose ``token_level_chunks`` is ``False`` while
    everything else stays declared, which is the shape a runtime that batches tokens into
    transport chunks really has — and the shape that makes
    ``test_token_level_chunks_implies_streaming`` and the streaming behaviours meet a delta that
    is not a token.
    """

    @pytest.fixture
    def provider(self) -> Provider:
        return FakeProvider(
            FakeScript(
                capabilities=dataclasses.replace(FULL_CAPABILITIES, token_level_chunks=False),
                generations=(FakeGeneration(chunk_size=17),),
            ),
            seed=17,
        )

    @pytest.fixture
    def known_reference(self) -> str:
        return "fake-model:8b-q8_0"

    @pytest.fixture
    def usage_shapes(self) -> UsageShapes | None:
        return _fake_usage_shapes()


class TestOllamaProviderConformance(ProviderConformanceSuite):
    """The same suite against :class:`~modelrack.providers.ollama.OllamaProvider`, over a
    recorded transport.

    Spec §11.5's whole point, proven the way it has to be proven: not by asserting the fake and
    the real adapter *look* similar, but by running one behaviour suite against both and letting
    it fail if they diverge. The mock router below models just enough server-side state — which
    model is currently resident — for the residency behaviours to observe a real transition,
    because a static canned response could not:
    ``test_residency_control_is_honoured_or_refused`` loads a model and then unloads it and
    expects the second call to see what the first one did.
    """

    @pytest.fixture
    def selected_chat_fixture(self) -> dict[str, str]:
        """The recorded ``/api/chat`` body the transport serves next, by filename.

        Mutable and shared with :func:`usage_shapes`, which is how one provider is driven into
        more than one response shape without rebuilding its transport. The default is the payload
        every other behaviour in the suite has always been asserted against.
        """
        return {"complete": "chat_complete.json"}

    @pytest.fixture
    def provider(
        self, load_ollama_fixture: Callable[[str], Any], selected_chat_fixture: dict[str, str]
    ) -> Iterator[Provider]:
        resident = {"is_resident": False}

        def show_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            is_llama = body.get("model") == "llama3.2:3b-instruct-q4_0"
            fixture = "show_llama.json" if is_llama else "show_qwen.json"
            return httpx.Response(200, json=load_ollama_fixture(fixture))

        def ps_handler(_request: httpx.Request) -> httpx.Response:
            fixture = "ps_resident.json" if resident["is_resident"] else "ps_empty.json"
            return httpx.Response(200, json=load_ollama_fixture(fixture))

        def generate_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            resident["is_resident"] = body.get("keep_alive") != 0
            fixture = (
                "generate_unload.json" if body.get("keep_alive") == 0 else "generate_load.json"
            )
            return httpx.Response(200, json=load_ollama_fixture(fixture))

        def chat_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body.get("stream"):
                return httpx.Response(
                    200,
                    content=load_ollama_fixture("chat_stream.ndjson"),
                    headers={"Content-Type": "application/x-ndjson"},
                )
            return httpx.Response(200, json=load_ollama_fixture(selected_chat_fixture["complete"]))

        with respx.mock(assert_all_called=False) as mock:
            mock.get("http://ollama.conformance.test/api/version").mock(
                return_value=httpx.Response(200, json=load_ollama_fixture("version.json"))
            )
            mock.get("http://ollama.conformance.test/api/tags").mock(
                return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
            )
            mock.post("http://ollama.conformance.test/api/show").mock(side_effect=show_handler)
            mock.get("http://ollama.conformance.test/api/ps").mock(side_effect=ps_handler)
            mock.post("http://ollama.conformance.test/api/generate").mock(
                side_effect=generate_handler
            )
            mock.post("http://ollama.conformance.test/api/chat").mock(side_effect=chat_handler)
            yield OllamaProvider(base_url="http://ollama.conformance.test")

    @pytest.fixture
    def known_reference(self) -> str:
        return "qwen3.5:9b-q8_0"

    @pytest.fixture
    def usage_shapes(
        self, provider: Provider, selected_chat_fixture: dict[str, str]
    ) -> UsageShapes | None:
        """Two shapes, and a declared absence of the third.

        ``cache_detail`` is ``None`` because Ollama's protocol has no cache-billing vocabulary at
        all — there is no field by which a response could report cached input, which is exactly
        why :func:`~modelrack.providers._ollama_wire.read_usage` reports both cache classes as
        `0` rather than reconciling anything (ADR-0070 decision 3). ``NO_USAGE_OBJECT`` is a
        terminal payload carrying neither ``prompt_eval_count`` nor ``eval_count``, this
        protocol's analogue of an absent ``usage`` object.
        """
        names = {
            UsageShape.NO_CACHE_DETAIL: "chat_complete.json",
            UsageShape.NO_USAGE_OBJECT: "chat_complete_no_counts.json",
        }

        def provider_for(shape: UsageShape) -> Provider:
            selected_chat_fixture["complete"] = names[shape]
            return provider

        return UsageShapes(provider_for=provider_for, cache_detail=None)


class TestOpenAICompatibleProviderConformance(ProviderConformanceSuite):
    """The same suite against
    :class:`~modelrack.providers.openai_compatible.OpenAICompatibleProvider`.

    Spec §11.5's second proof: this adapter declares a materially different capability set than
    :class:`~modelrack.providers.ollama.OllamaProvider` — no residency control, no caller-chosen
    context — which is exactly what exercises the suite's *refusal* branches
    (``test_residency_control_is_honoured_or_refused``,
    ``test_a_caller_chosen_context_is_honoured_or_refused``) against a real transport rather than
    only against :class:`~modelrack.testing.FakeProvider`'s scripted minimal capabilities.
    """

    @pytest.fixture
    def selected_chat_fixture(self) -> dict[str, str]:
        """The recorded ``/v1/chat/completions`` body the transport serves next, by filename."""
        return {"complete": "chat_complete.json"}

    @pytest.fixture
    def provider(
        self,
        load_openai_compatible_fixture: Callable[[str], Any],
        selected_chat_fixture: dict[str, str],
    ) -> Iterator[Provider]:
        def chat_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body.get("stream"):
                return httpx.Response(
                    200,
                    content=load_openai_compatible_fixture("chat_stream.sse").encode("utf-8"),
                    headers={"Content-Type": "text/event-stream"},
                )
            return httpx.Response(
                200, json=load_openai_compatible_fixture(selected_chat_fixture["complete"])
            )

        with respx.mock(assert_all_called=False) as mock:
            mock.get("http://openai-compatible.conformance.test/v1/models").mock(
                return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
            )
            mock.post("http://openai-compatible.conformance.test/v1/chat/completions").mock(
                side_effect=chat_handler
            )
            yield OpenAICompatibleProvider(base_url="http://openai-compatible.conformance.test")

    @pytest.fixture
    def known_reference(self) -> str:
        return "qwen3.5-9b-instruct-q8_0"

    @pytest.fixture
    def usage_shapes(
        self, provider: Provider, selected_chat_fixture: dict[str, str]
    ) -> UsageShapes | None:
        """All three shapes: this is the protocol that can express cached input.

        ``chat_complete_cached.json`` reports ``prompt_tokens`` 21 with
        ``prompt_tokens_details.cached_tokens`` 8, so a correct adapter reports input 13 and
        cache read 8 — and an adapter that skipped the reconciliation would report input 21 and
        bill the cached prefix twice.
        """
        names = {
            UsageShape.NO_CACHE_DETAIL: "chat_complete.json",
            UsageShape.CACHE_DETAIL: "chat_complete_cached.json",
            UsageShape.NO_USAGE_OBJECT: "chat_complete_no_usage.json",
        }

        def provider_for(shape: UsageShape) -> Provider:
            selected_chat_fixture["complete"] = names[shape]
            return provider

        return UsageShapes(
            provider_for=provider_for,
            cache_detail=CacheDetailShape(prompt_tokens=21, cached_tokens=8),
        )
