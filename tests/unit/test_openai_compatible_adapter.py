"""Tests for :mod:`modelrack.providers.openai_compatible` — the second real provider adapter.

Every test here runs against a recorded transport (``respx``), never a live server — the default
suite must pass with nothing running (spec §18 acceptance criterion 3). Fixtures live under
``tests/fixtures/providers/openai_compatible/`` and are version-annotated in that directory's
``manifest.json`` (spec §19).

Two properties carry their own acceptance criteria beyond what the conformance suite in
``tests/contract/test_conformance.py`` proves generically:

* **SSE parsing survives the shapes real servers send** — :class:`TestStreaming` covers multi-line
  ``data:`` blocks, ``:``-prefixed keep-alive comments, the ``[DONE]`` sentinel, and a malformed
  frame, each named explicitly in the development plan's Phase 4 test list.
* **Every identity is name-only, and an API key never reaches diagnostics** — spec §11.9 and §14,
  proven directly rather than only implied.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import (
    IdentityConfidence,
    ModelIdentity,
    ProviderKind,
    RuntimeProfile,
    ValidationError,
    is_supported,
)

from modelrack import (
    CapabilityUnsupported,
    ContextLimitExceeded,
    FinishReason,
    GenerationCancelled,
    GenerationRequest,
    Message,
    ModelNotFound,
    ProviderProtocolError,
    ProviderRejected,
    ProviderStatus,
    ProviderTimeout,
    ProviderUnavailable,
    ResponseFormat,
    ResponseFormatKind,
    Role,
    SamplingParameters,
    StreamCompleted,
    StreamFailed,
    TokenDelta,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
)
from modelrack.providers.openai_compatible import OpenAICompatibleProvider
from modelrack.streaming import CancellationToken

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from modelrack import StreamEvent

_BASE_URL = "http://127.0.0.1:8080"
_MODEL = "qwen3.5-9b-instruct-q8_0"
_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _identity(name: str = _MODEL) -> ModelIdentity:
    """Return a bare, name-only identity for ``name`` on the OpenAI-compatible provider kind."""
    return ModelIdentity(ProviderKind.OPENAI_COMPATIBLE, name)


def _request(**overrides: Any) -> GenerationRequest:
    """Build the standard chat request most tests exercise."""
    fields: dict[str, Any] = {
        "identity": _identity(),
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def _provider(**kwargs: Any) -> OpenAICompatibleProvider:
    """Build a provider pointed at the mocked base URL."""
    return OpenAICompatibleProvider(base_url=_BASE_URL, **kwargs)


def _text_deltas(events: Sequence[StreamEvent]) -> list[TokenDelta]:
    """Return only the answer-text deltas from a drained stream."""
    return [event for event in events if isinstance(event, TokenDelta)]


def _sse_response(text: str, *, status_code: int = 200) -> httpx.Response:
    """Build a streamed response whose body is exactly this server-sent-event text."""
    return httpx.Response(
        status_code, content=text.encode("utf-8"), headers={"Content-Type": "text/event-stream"}
    )


class TestHealthAndCapabilities:
    """Probing the provider, and the static declaration it makes without probing anything."""

    @respx.mock
    def test_a_healthy_provider_reports_model_count(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        health = _provider().health()

        assert health.status is ProviderStatus.OK
        assert health.model_count == 2
        assert health.provider_version is None
        assert health.is_remote is False

    @respx.mock
    def test_an_unreachable_provider_reports_rather_than_raises(self) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(side_effect=httpx.ConnectError("refused"))

        health = _provider().health()

        assert health.status is ProviderStatus.UNAVAILABLE
        assert not is_supported(health.model_count)

    def test_capabilities_declare_only_what_this_protocol_can_carry(self) -> None:
        capabilities = _provider().capabilities()

        assert capabilities.streaming
        assert capabilities.tool_calling
        assert capabilities.structured_output
        assert capabilities.json_mode
        assert capabilities.token_counts

    def test_capabilities_are_honest_about_what_this_protocol_cannot_carry(self) -> None:
        """Spec §11.10: these are refusals a caller can act on, not omissions to discover."""
        capabilities = _provider().capabilities()

        assert capabilities.token_level_chunks is False
        assert capabilities.thinking_control is False
        assert capabilities.logprobs is False
        assert capabilities.force_unload is False
        assert capabilities.residency_query is False
        assert capabilities.kv_metrics is False
        assert capabilities.context_configurable is False
        assert capabilities.embedding is False

    @respx.mock
    def test_a_remote_host_flags_health_as_remote(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get("http://gpu-box.lan:8080/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        health = OpenAICompatibleProvider(base_url="http://gpu-box.lan:8080").health()

        assert health.is_remote is True

    @pytest.mark.parametrize("bad_url", ["ftp://x", "not-a-url", ""])
    def test_a_malformed_base_url_is_refused_at_construction(self, bad_url: str) -> None:
        with pytest.raises(ValidationError):
            OpenAICompatibleProvider(base_url=bad_url)

    @respx.mock
    def test_health_reports_degraded_when_the_credential_is_rejected(self) -> None:
        """A wrong or expired ``api_key`` is a 401 from a server running perfectly well.
        Reporting that as "unreachable" would send an operator to check the wrong thing, and
        raising would turn one bad credential into a 500 for the caller's whole health endpoint.
        """
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(401, json={"error": {"message": "invalid api key"}})
        )

        health = _provider(api_key="sk-expired-4a1f").health()  # noqa: S106

        assert health.status is ProviderStatus.DEGRADED
        assert "PROVIDER_REJECTED" in health.detail

    @respx.mock
    def test_a_degraded_health_detail_leaks_neither_the_key_nor_the_servers_message(self) -> None:
        """Spec §14 names `raw`, error `details` and the DEBUG log. A health document is rendered
        into a UI, which makes it the fifth channel the same discipline has to hold on.
        """
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "key sk-expired-4a1f is not valid"}}
            )
        )

        health = _provider(api_key="sk-expired-4a1f").health()  # noqa: S106

        assert "sk-expired-4a1f" not in health.detail
        assert "not valid" not in health.detail


class TestApiKey:
    """Spec §14: sent only in the header, never logged, never in diagnostics."""

    @respx.mock
    def test_an_api_key_is_sent_as_a_bearer_header(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        _provider(api_key="sk-super-secret-4a1f").list_models()

        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-super-secret-4a1f"

    @respx.mock
    def test_no_api_key_sends_no_authorization_header(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        _provider().list_models()

        assert "Authorization" not in route.calls.last.request.headers

    @respx.mock
    def test_an_api_key_never_reaches_a_result_or_a_debug_log(
        self, load_openai_compatible_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )
        secret = "sk-super-secret-4a1f"  # noqa: S105 — a fixture value asserted absent, not a real credential

        with caplog.at_level(logging.DEBUG, logger="modelrack"):
            provider = _provider(api_key=secret)
            provider.list_models()
            result = provider.generate(_request())

        assert secret not in repr(dict(result.raw))
        assert all(secret not in record.getMessage() for record in caplog.records)


class TestDiscovery:
    """``list_models``, ``inspect_model`` and ``resolve`` — every identity is name-only."""

    @respx.mock
    def test_list_models_with_zero_models(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("models_empty.json")
            )
        )

        assert list(_provider().list_models()) == []

    @respx.mock
    def test_list_models_reports_every_entry_as_name_only(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        descriptors = _provider().list_models()

        assert len(descriptors) == 2
        assert all(
            d.identity.identity_confidence is IdentityConfidence.NAME_ONLY for d in descriptors
        )
        assert all(d.identity.artifact_digest is None for d in descriptors)
        assert all(d.raw for d in descriptors)

    @respx.mock
    def test_inspect_model_of_a_known_name(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        descriptor = _provider().inspect_model(_identity())

        assert descriptor.identity.provider_model_name == _MODEL

    @respx.mock
    def test_inspect_model_of_an_unknown_name_raises(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        with pytest.raises(ModelNotFound) as raised:
            _provider().inspect_model(_identity("nope"))

        assert raised.value.details == {"reference": "nope", "known_model_count": 2}

    @respx.mock
    def test_an_exact_reference_resolves(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        assert _provider().resolve(_MODEL).provider_model_name == _MODEL

    @respx.mock
    def test_a_unique_prefix_resolves(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        assert _provider().resolve("qwen3.5").provider_model_name == _MODEL

    @respx.mock
    def test_an_ambiguous_prefix_is_refused(self) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "qwen-7b"}, {"id": "qwen-14b"}],
                },
            )
        )

        with pytest.raises(ModelNotFound) as raised:
            _provider().resolve("qwen")

        assert raised.value.details["matched_model_count"] == 2

    @respx.mock
    def test_resolving_nothing_names_the_reference_and_count(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        with pytest.raises(ModelNotFound) as raised:
            _provider().resolve("does-not-exist")

        assert raised.value.details == {"reference": "does-not-exist", "known_model_count": 2}

    @respx.mock
    def test_the_library_logs_nothing_at_info_or_above(
        self, load_openai_compatible_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec §17: a library must not configure or spam the host's logs."""
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        with caplog.at_level(logging.INFO, logger="modelrack"):
            _provider().resolve("qwen3.5")

        assert caplog.records == []


class TestGeneration:
    """Non-streaming ``generate()``: text, usage, finish reasons, tool calls."""

    @respx.mock
    def test_generate_returns_the_answer_and_finish_reason(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        result = _provider().generate(_request())

        assert "KV caching" in result.text
        assert result.finish_reason is FinishReason.STOP

    @respx.mock
    def test_generate_reports_token_counts(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        usage = _provider().generate(_request()).usage

        assert usage.tokens.input_tokens == 21
        assert usage.tokens.output_tokens == 15

    @respx.mock
    def test_a_usage_object_without_cache_detail_reports_cache_classes_as_zero(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """No details object means the server does no cache accounting (ADR-0070 decision 2).

        Both classes are `0` for different reasons worth keeping straight: cache *read* because
        this server would have reported it had it billed any, and cache *write* because this
        protocol has no field for one at all.
        """
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        tokens = _provider().generate(_request()).usage.tokens

        assert tokens.cache_read_tokens == 0
        assert tokens.cache_write_tokens == 0
        assert tokens.total_tokens == 36

    @respx.mock
    def test_cached_input_is_reconciled_out_of_the_input_class(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """The subtraction ADR-0030 assigns to the adapter, on the shape that needs it.

        ``prompt_tokens`` 21 already contains the 8 cached tokens reported beside it, so input is
        13. An adapter that passed ``prompt_tokens`` straight through would bill those 8 tokens
        twice — at the full input rate and again at the cache-read rate.
        """
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete_cached.json")
            )
        )

        tokens = _provider().generate(_request()).usage.tokens

        assert tokens.cache_read_tokens == 8
        assert tokens.input_tokens == 13
        assert tokens.input_tokens + tokens.cache_read_tokens == 21
        assert tokens.cache_write_tokens == 0

    @respx.mock
    def test_a_response_with_no_usage_object_reports_every_class_unsupported(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """Nothing reported is not zero reported — ADR-0070's third case."""
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete_no_usage.json")
            )
        )

        tokens = _provider().generate(_request()).usage.tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.output_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)

    @respx.mock
    @pytest.mark.parametrize(
        ("label", "details"),
        [
            ("not a mapping", "not-an-object"),
            ("explicit null", None),
            ("no cached_tokens key", {"audio_tokens": 4}),
            ("a fractional figure", {"cached_tokens": 1.5}),
            ("a negative figure", {"cached_tokens": -1}),
            ("a numeric string", {"cached_tokens": "8"}),
            ("more cached than prompt", {"cached_tokens": 22}),
        ],
    )
    def test_an_unreadable_details_object_refuses_rather_than_reporting_zero(
        self, label: str, details: Any
    ) -> None:
        """A details object this adapter cannot read leaves *both* halves of the pair unknown.

        The one case where a confident `0` would be the fabricated zero rather than the honest
        one: the server sent a details object, so it does cache accounting, so an unreadable
        figure means a class that may well have been billed was not reported. Reporting the pair
        as ``UNSUPPORTED`` is ADR-0070 decision 1's second sentence; reporting cache read as `0`
        would be its first, misapplied. ``input_tokens`` goes with it because the two are the
        halves of one subtraction — ``prompt_tokens`` beside an unknown cached figure is not
        disjoint from it — and because clamping instead would report `0` input for a call that
        certainly had some.
        """
        payload = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 15,
                "prompt_tokens_details": details,
            },
        }
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )

        tokens = _provider().generate(_request()).usage.tokens

        assert not is_supported(tokens.input_tokens), label
        assert not is_supported(tokens.cache_read_tokens), label
        # The classes the unreadable details object says nothing about are unaffected.
        assert tokens.output_tokens == 15
        assert tokens.cache_write_tokens == 0

    @respx.mock
    def test_generate_reports_no_first_token_moment_or_backend_timing(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """This protocol reports no backend timing breakdown at all — unlike Ollama."""
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        timing = _provider().generate(_request()).timing

        assert not is_supported(timing.client_ttft_ms)
        assert not is_supported(timing.backend_load_ms)
        assert not is_supported(timing.backend_prompt_eval_ms)
        assert not is_supported(timing.backend_decode_ms)
        assert not is_supported(timing.backend_total_ms)
        assert is_supported(timing.client_wall_ms)

    @respx.mock
    def test_generate_parses_a_tool_call(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete_tool_calls.json")
            )
        )

        result = _provider().generate(_request(tools=(_WEATHER_TOOL,)))

        assert result.finish_reason is FinishReason.TOOL_CALLS
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.id == "call_7f1a"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Chicago"}

    @respx.mock
    def test_a_malformed_tool_call_argument_string_is_preserved_not_dropped(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=load_openai_compatible_fixture("chat_complete_tool_calls_malformed.json"),
            )
        )

        result = _provider().generate(_request(tools=(_WEATHER_TOOL,)))

        call = result.tool_calls[0]
        assert call.arguments == {}
        assert call.raw_arguments == '{"city": "Chicago"'

    @respx.mock
    def test_a_caller_chosen_context_is_refused_before_any_request_is_sent(self) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions")

        with pytest.raises(CapabilityUnsupported) as raised:
            _provider().generate(_request(runtime_profile=RuntimeProfile(context_size=4096)))

        assert raised.value.details["capability"] == "context_configurable"
        assert route.call_count == 0

    @respx.mock
    def test_caller_metadata_never_reaches_the_request_body(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        marker = "conformance-correlation-4a1f"
        _provider().generate(_request(metadata={"run_id": marker}))

        assert marker not in route.calls.last.request.content.decode("utf-8")

    @respx.mock
    def test_a_completion_style_prompt_is_sent_as_a_single_user_message(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        _provider().generate(_request(messages=(), prompt="Explain KV caching."))

        body = json.loads(route.calls.last.request.content)
        assert body["messages"] == [{"role": "user", "content": "Explain KV caching."}]

    @respx.mock
    def test_sampling_parameters_map_to_openai_fields(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        _provider().generate(
            _request(
                sampling=SamplingParameters(
                    temperature=0.2,
                    top_p=0.9,
                    top_k=40,
                    seed=7,
                    max_output_tokens=256,
                    stop=("\n\n",),
                    repeat_penalty=1.1,
                )
            )
        )

        body = json.loads(route.calls.last.request.content)
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        assert body["top_k"] == 40
        assert body["seed"] == 7
        assert body["max_tokens"] == 256
        assert body["stop"] == ["\n\n"]
        assert body["repetition_penalty"] == 1.1
        assert body["repeat_penalty"] == 1.1, "llama-server reads this spelling, vLLM the other"

    @respx.mock
    def test_provider_options_extends_and_overrides_the_body(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        _provider().generate(
            _request(
                runtime_profile=RuntimeProfile(
                    provider_options={"mirostat": 2, "model": "override-wins"}
                )
            )
        )

        body = json.loads(route.calls.last.request.content)
        assert body["mirostat"] == 2
        assert body["model"] == "override-wins"

    @respx.mock
    def test_json_mode_sets_the_response_format_field(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        _provider().generate(_request(response_format=ResponseFormat(kind=ResponseFormatKind.JSON)))

        body = json.loads(route.calls.last.request.content)
        assert body["response_format"] == {"type": "json_object"}

    @respx.mock
    def test_a_json_schema_passes_the_schema_object_through(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        _provider().generate(
            _request(
                response_format=ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema=schema)
            )
        )

        body = json.loads(route.calls.last.request.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == schema


class TestStreaming:
    """SSE parsing: multi-line data, ``[DONE]``, keep-alive comments, malformed frames."""

    @respx.mock
    def test_recorded_multi_delta_stream_reassembles_into_the_recorded_answer(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream.sse"))
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "KV caching stores values."
        assert "".join(d.text for d in _text_deltas(events)) == terminal.result.text
        assert terminal.result.finish_reason is FinishReason.STOP
        assert terminal.result.usage.tokens.output_tokens == 4

    @respx.mock
    def test_a_stream_that_carried_a_usage_chunk_reports_cache_classes_as_zero(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """The streamed path reaches the same three cases as the non-streaming one."""
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream.sse"))
        )

        terminal = list(_provider().stream(_request()))[-1]

        assert isinstance(terminal, StreamCompleted)
        tokens = terminal.result.usage.tokens
        assert tokens.input_tokens == 21
        assert tokens.cache_read_tokens == 0
        assert tokens.cache_write_tokens == 0
        assert tokens.total_tokens == 25

    @respx.mock
    def test_a_stream_that_carried_no_usage_chunk_reports_every_class_unsupported(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """The default-value trap, asserted where it would have been sprung.

        A stream that never carried a usage chunk arrives at ``_read_usage`` as
        ``{"usage": {}}`` — an *empty* mapping produced by this adapter's own accumulator, not by
        the server. Reading that as "a usage object without cache detail" would report cache
        classes of `0` for a response that reported no usage at all, which is a fabricated zero
        manufactured out of a default value rather than out of anything the server said.
        """
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(
                load_openai_compatible_fixture("chat_stream_no_finish_reason.sse")
            )
        )

        terminal = list(_provider().stream(_request()))[-1]

        assert isinstance(terminal, StreamCompleted)
        tokens = terminal.result.usage.tokens
        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.output_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)

    @respx.mock
    def test_a_stream_reports_the_first_token_moment_it_observed(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream.sse"))
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert is_supported(terminal.result.timing.client_ttft_ms)

    @respx.mock
    def test_multiline_data_and_keepalive_comments_reassemble_correctly(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream_multiline.sse"))
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "KV caching stores values."
        assert terminal.result.finish_reason is FinishReason.STOP

    @respx.mock
    def test_a_malformed_frame_is_delivered_as_a_protocol_error_not_raised(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream_malformed.sse"))
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "KV"

    @respx.mock
    def test_a_stream_ending_without_done_is_a_protocol_error(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream_truncated.sse"))
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)

    @respx.mock
    def test_streaming_tool_calls_reassemble_across_fragments(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream_tool_calls.sse"))
        )

        events = list(_provider().stream(_request(tools=(_WEATHER_TOOL,))))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.finish_reason is FinishReason.TOOL_CALLS
        call = terminal.result.tool_calls[0]
        assert call.id == "call_7f1a"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Chicago"}
        fragments = [e for e in events if isinstance(e, ToolCallDelta)]
        assert len(fragments) >= 2

    @respx.mock
    def test_cancelling_mid_stream_stops_within_one_delta(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream.sse"))
        )
        token = CancellationToken()
        events: list[StreamEvent] = []

        for event in _provider().stream(_request(cancel=token)):
            events.append(event)
            if not token.is_cancelled and len(_text_deltas(events)) == 2:
                token.cancel()

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == "".join(d.text for d in _text_deltas(events))

    @respx.mock
    def test_a_caller_chosen_context_is_refused_before_streaming_starts(self) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions")

        with pytest.raises(CapabilityUnsupported):
            list(_provider().stream(_request(runtime_profile=RuntimeProfile(context_size=4096))))

        assert route.call_count == 0

    @respx.mock
    def test_streaming_is_refused_before_any_content_when_the_model_is_missing(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                404, json=load_openai_compatible_fixture("error_model_not_found.json")
            )
        )
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        with pytest.raises(ModelNotFound):
            list(_provider().stream(_request()))


class TestErrors:
    """Spec §13: every documented failure is a typed error, never a raw ``httpx`` exception."""

    @respx.mock
    def test_connection_refused(self) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )

        with pytest.raises(ProviderUnavailable):
            _provider().generate(_request())

    @respx.mock
    def test_timeout(self) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("too slow")
        )

        with pytest.raises(ProviderTimeout):
            _provider().generate(_request())

    @respx.mock
    def test_404_model_not_found(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                404, json=load_openai_compatible_fixture("error_model_not_found.json")
            )
        )
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json=load_openai_compatible_fixture("models.json"))
        )

        with pytest.raises(ModelNotFound) as raised:
            _provider().generate(_request())

        assert raised.value.details["known_model_count"] == 2

    @respx.mock
    def test_400_bad_request_is_provider_rejected(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                400, json=load_openai_compatible_fixture("error_bad_request.json")
            )
        )

        with pytest.raises(ProviderRejected) as raised:
            _provider().generate(_request())

        assert raised.value.details["status_code"] == 400

    @respx.mock
    def test_context_overflow_is_classified_by_the_structured_code(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                400, json=load_openai_compatible_fixture("error_context_overflow.json")
            )
        )

        with pytest.raises(ContextLimitExceeded):
            _provider().generate(_request())

    @respx.mock
    def test_context_overflow_falls_back_to_the_message_when_the_code_is_absent(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                400, json=load_openai_compatible_fixture("error_context_overflow_no_code.json")
            )
        )

        with pytest.raises(ContextLimitExceeded):
            _provider().generate(_request())

    @respx.mock
    def test_a_non_json_body_is_a_protocol_error(self) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=b"not json")
        )

        with pytest.raises(ProviderProtocolError):
            _provider().generate(_request())

    @respx.mock
    def test_an_unexpected_5xx_without_a_message_is_a_protocol_error(self) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(503, json={"detail": "no error field here"})
        )

        with pytest.raises(ProviderProtocolError):
            _provider().generate(_request())

    @respx.mock
    def test_every_documented_error_is_a_provider_error_never_a_raw_httpx_exception(self) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )

        try:
            _provider().generate(_request())
        except Exception as exc:  # noqa: BLE001 — asserting the *type*, deliberately broad
            assert not isinstance(exc, httpx.HTTPError)


class TestResidency:
    """Spec §11.10-adjacent: refused honestly, with no HTTP call ever attempted."""

    def test_load_is_refused_without_a_request(self) -> None:
        with pytest.raises(CapabilityUnsupported) as raised:
            _provider().load(_identity(), RuntimeProfile())

        assert raised.value.details["capability"] == "force_unload"

    def test_unload_is_refused_without_a_request(self) -> None:
        with pytest.raises(CapabilityUnsupported) as raised:
            _provider().unload(_identity())

        assert raised.value.details["capability"] == "force_unload"

    def test_list_resident_is_refused_without_a_request(self) -> None:
        with pytest.raises(CapabilityUnsupported) as raised:
            _provider().list_resident()

        assert raised.value.details["capability"] == "residency_query"


class TestCoverageCompleting:
    """Edge cases the classes above don't happen to reach: malformed shapes, defaults, field
    combinations a normal request never produces on its own.
    """

    @respx.mock
    def test_a_non_object_generate_response_is_a_protocol_error(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete_array.json")
            )
        )

        with pytest.raises(ProviderProtocolError):
            _provider().generate(_request())

    @respx.mock
    def test_an_in_band_error_on_a_non_streaming_call_still_raises(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """A 200 status with an ``error`` body is still an error (spec §13)."""
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete_error_200.json")
            )
        )

        with pytest.raises(ProviderRejected):
            _provider().generate(_request())

    @respx.mock
    def test_a_connection_failure_before_streaming_begins_raises(self) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )

        with pytest.raises(ProviderUnavailable):
            list(_provider().stream(_request()))

    @respx.mock
    def test_a_non_object_event_mid_stream_is_a_protocol_error(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream_non_object.sse"))
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "KV"

    @respx.mock
    def test_an_in_band_error_mid_stream_is_delivered_not_raised(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(
                load_openai_compatible_fixture("chat_stream_inband_error.sse")
            )
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderRejected)
        assert terminal.partial_text == "KV"

    @respx.mock
    def test_a_stray_non_mapping_tool_call_fragment_is_skipped(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(
                load_openai_compatible_fixture("chat_stream_tool_calls_stray_fragment.sse")
            )
        )

        events = list(_provider().stream(_request(tools=(_WEATHER_TOOL,))))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert len(terminal.result.tool_calls) == 1
        assert terminal.result.tool_calls[0].id == "call_1"

    @respx.mock
    def test_a_tool_call_fragment_with_no_index_falls_back_to_arrival_order(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(
                load_openai_compatible_fixture("chat_stream_tool_call_no_index.sse")
            )
        )

        events = list(_provider().stream(_request(tools=(_WEATHER_TOOL,))))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.tool_calls[0].id == "call_x"
        assert terminal.result.tool_calls[0].arguments == {"city": "Reno"}

    @respx.mock
    def test_a_stream_that_never_reports_a_finish_reason_is_unknown(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(
                load_openai_compatible_fixture("chat_stream_no_finish_reason.sse")
            )
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.finish_reason is FinishReason.UNKNOWN

    @respx.mock
    def test_comment_lines_and_other_sse_fields_are_ignored(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """A stream ending with no trailing blank line still dispatches its last event."""
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(
                load_openai_compatible_fixture("chat_stream_field_variety.sse")
            )
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "hi"
        assert terminal.result.finish_reason is FinishReason.STOP

    @respx.mock
    def test_a_dropped_connection_mid_stream_is_a_protocol_error_not_unavailable(self) -> None:
        class FlakyStream(httpx.SyncByteStream):
            def __iter__(self) -> Any:
                yield b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
                raise httpx.ReadError("connection reset by peer")

            def close(self) -> None:
                pass

        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, stream=FlakyStream())
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "hi"

    @respx.mock
    def test_a_timeout_mid_stream_stays_a_timeout(self) -> None:
        class SlowStream(httpx.SyncByteStream):
            def __iter__(self) -> Any:
                yield b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
                raise httpx.ReadTimeout("timed out")

            def close(self) -> None:
                pass

        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, stream=SlowStream())
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderTimeout)

    @respx.mock
    def test_an_oversize_chunk_mid_stream_is_delivered_not_raised(self) -> None:
        ok_line = b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
        big_line = b'data: {"choices":[{"index":0,"delta":{"content":"' + b"y" * 200 + b'"}}]}\n\n'
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=iter([ok_line, big_line]))
        )

        events = list(_provider(max_chunk_bytes=100).stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "ok"

    @respx.mock
    def test_a_per_request_timeout_overrides_the_client_default_while_streaming(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=_sse_response(load_openai_compatible_fixture("chat_stream.sse"))
        )

        list(_provider().stream(_request(timeout_seconds=5.0)))

        assert route.calls.last.request.extensions["timeout"] == {
            "connect": 5.0,
            "read": 5.0,
            "write": 5.0,
            "pool": 5.0,
        }

    @respx.mock
    def test_an_oversize_error_body_is_reported_as_oversize_not_reclassified(self) -> None:
        big_body = json.dumps({"error": "x" * 1000}).encode()
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(400, content=big_body)
        )

        with pytest.raises(ProviderProtocolError) as raised:
            _provider(max_response_bytes=100).generate(_request())

        assert raised.value.details["limit_bytes"] == 100

    @respx.mock
    def test_an_error_body_that_is_not_json_is_a_protocol_error_not_reclassified(self) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(500, content=b"internal server error, not json")
        )

        with pytest.raises(ProviderProtocolError):
            _provider().generate(_request())

    @respx.mock
    def test_list_models_failure_is_translated(self) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(return_value=httpx.Response(500, content=b"boom"))

        with pytest.raises(ProviderProtocolError):
            _provider().list_models()

    @respx.mock
    def test_a_malformed_models_payload_yields_an_empty_catalogue(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/v1/models").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("models_malformed.json")
            )
        )

        assert list(_provider().list_models()) == []

    @respx.mock
    def test_a_prior_tool_call_and_a_named_tool_result_are_sent_through(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        """An assistant turn's own prior tool call, and the tool result answering it."""
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        _provider().generate(
            _request(
                messages=(
                    Message(role=Role.USER, content="What's the weather in Chicago?"),
                    Message(
                        role=Role.ASSISTANT,
                        content="",
                        tool_calls=(
                            ToolCall(
                                id="call_1", name="get_weather", arguments={"city": "Chicago"}
                            ),
                        ),
                    ),
                    Message(
                        role=Role.TOOL,
                        content="72F and sunny",
                        tool_call_id="call_1",
                        name="get_weather",
                    ),
                )
            )
        )

        body = json.loads(route.calls.last.request.content)
        assert body["messages"][1]["tool_calls"][0]["id"] == "call_1"
        assert body["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert body["messages"][2]["tool_call_id"] == "call_1"
        assert body["messages"][2]["name"] == "get_weather"

    @respx.mock
    def test_an_explicit_text_response_format_is_sent_as_the_text_type(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=load_openai_compatible_fixture("chat_complete.json")
            )
        )

        _provider().generate(_request(response_format=ResponseFormat()))

        body = json.loads(route.calls.last.request.content)
        assert body["response_format"] == {"type": "text"}

    @respx.mock
    def test_a_non_string_non_mapping_tool_call_entry_is_skipped_on_generate(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    None,
                                    {
                                        "id": "call_1",
                                        "function": {"name": "get_weather", "arguments": "{}"},
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        )

        result = _provider().generate(_request(tools=(_WEATHER_TOOL,)))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"

    @respx.mock
    def test_a_tool_call_with_no_arguments_at_all_yields_empty_arguments(
        self, load_openai_compatible_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{"id": "call_1", "function": {"name": "noop"}}],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        )

        result = _provider().generate(_request(tools=(_WEATHER_TOOL,)))

        assert result.tool_calls[0].arguments == {}
        assert result.tool_calls[0].raw_arguments is None
