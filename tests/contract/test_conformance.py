"""The provider conformance suite: one set of behaviours every adapter must exhibit.

[Spec §11.5](../../docs/packages/modelrack/spec.md) — *every adapter passes the same conformance
suite* — is the sentence this file exists to make true, and
[testing standards §7](../../docs/standards/testing-standards.md) states it generally: a new
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
from typing import TYPE_CHECKING, Any

import pytest
from baseaicore import IdentityConfidence, RuntimeProfile, is_supported

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
from modelrack.streaming import CancellationToken
from modelrack.testing import (
    FULL_CAPABILITIES,
    MINIMAL_CAPABILITIES,
    FakeGeneration,
    FakeProvider,
    FakeScript,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from baseaicore import ModelIdentity

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


class TestFakeProviderConformance(ProviderConformanceSuite):
    """The fake with everything it can do declared — the capable path through the suite."""

    @pytest.fixture
    def provider(self) -> Provider:
        return FakeProvider(seed=17)

    @pytest.fixture
    def known_reference(self) -> str:
        return "fake-model:8b-q8_0"


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
