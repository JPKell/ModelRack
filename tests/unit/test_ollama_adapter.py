"""Tests for :mod:`modelrack.providers.ollama` — the real, HTTP-backed provider adapter.

Every test here runs against a recorded transport (``respx``), never a live Ollama — the default
suite must pass with none running (spec §18 acceptance criterion 3). Fixtures live under
``tests/fixtures/providers/ollama/`` and are version-annotated in that directory's
``manifest.json`` (spec §19: a provider version bump triggers re-capture and a changelog note).

Two properties carry their own acceptance criteria and are proven directly rather than only
implied by the conformance suite:

* **NDJSON chunk boundaries never corrupt a delta** — :class:`TestNdjsonChunking` places a
  multi-byte UTF-8 character split across two raw byte chunks and asserts it reassembles exactly,
  the scenario named in the development plan's Phase 3 test list.
* **Backend and client timings never merge** — :class:`TestTiming` asserts every ``backend_*``
  field comes from Ollama's nanosecond durations and every ``client_*`` field comes from this
  process's own :func:`baseaicore.monotonic_ns` reading, never the other way around.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import (
    UNSUPPORTED,
    IdentityConfidence,
    ModelCapabilityFlag,
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
    ThinkingDelta,
    TokenDelta,
    ToolCallDelta,
    ToolDefinition,
)
from modelrack.providers.ollama import OllamaProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from modelrack import StreamEvent

_BASE_URL = "http://127.0.0.1:11434"
_MODEL = "qwen3.5:9b-q8_0"
_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _identity(name: str = _MODEL) -> ModelIdentity:
    """Return a bare, name-only identity for ``name`` on the Ollama provider kind."""
    return ModelIdentity(ProviderKind.OLLAMA, name)


def _request(**overrides: Any) -> GenerationRequest:
    """Build the standard chat request most tests exercise."""
    fields: dict[str, Any] = {
        "identity": _identity(),
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def _provider(**kwargs: Any) -> OllamaProvider:
    """Build a provider pointed at the mocked base URL."""
    return OllamaProvider(base_url=_BASE_URL, **kwargs)


def _text_deltas(events: Sequence[StreamEvent]) -> list[TokenDelta]:
    """Return only the answer-text deltas from a drained stream."""
    return [event for event in events if isinstance(event, TokenDelta)]


def _ndjson_response(*chunks: bytes, status_code: int = 200) -> httpx.Response:
    """Build a streamed response delivered as exactly these raw byte chunks, no more, no fewer."""
    return httpx.Response(status_code, content=iter(chunks))


class TestHealthAndCapabilities:
    """Probing the provider, and the static declaration it makes without probing anything."""

    @respx.mock
    def test_a_healthy_provider_reports_version_and_model_count(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/version").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("version.json"))
        )
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )

        health = _provider().health()

        assert health.status is ProviderStatus.OK
        assert health.provider_version == "0.32.13"
        assert health.model_count == 2
        assert health.is_remote is False

    @respx.mock
    def test_an_unreachable_provider_reports_rather_than_raises(self) -> None:
        respx.get(f"{_BASE_URL}/api/version").mock(side_effect=httpx.ConnectError("refused"))

        health = _provider().health()

        assert health.status is ProviderStatus.UNAVAILABLE
        assert not is_supported(health.model_count)
        assert health.provider_version is None

    def test_capabilities_are_static_and_need_no_request(self) -> None:
        capabilities = _provider().capabilities()

        assert capabilities.streaming
        assert capabilities.tool_calling
        assert capabilities.structured_output
        assert capabilities.json_mode
        assert capabilities.token_counts
        assert capabilities.token_level_chunks
        assert capabilities.thinking_control
        assert capabilities.force_unload
        assert capabilities.residency_query
        assert capabilities.context_configurable

    def test_capabilities_never_claims_what_this_package_cannot_carry(self) -> None:
        """Neither endpoint exposes log probabilities, KV-cache counters or embeddings."""
        capabilities = _provider().capabilities()

        assert capabilities.logprobs is False
        assert capabilities.kv_metrics is False
        assert capabilities.embedding is False

    def test_a_remote_host_is_flagged_not_hidden(self) -> None:
        health_url = OllamaProvider(base_url="http://gpu-box.lan:11434")._base_url  # noqa: SLF001
        assert health_url == "http://gpu-box.lan:11434"

    @respx.mock
    def test_a_remote_host_flags_health_as_remote(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get("http://gpu-box.lan:11434/api/version").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("version.json"))
        )
        respx.get("http://gpu-box.lan:11434/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags_empty.json"))
        )

        health = OllamaProvider(base_url="http://gpu-box.lan:11434").health()

        assert health.is_remote is True

    @pytest.mark.parametrize("bad_url", ["ftp://x", "not-a-url", ""])
    def test_a_malformed_base_url_is_refused_at_construction(self, bad_url: str) -> None:
        with pytest.raises(ValidationError):
            OllamaProvider(base_url=bad_url)

    @respx.mock
    def test_health_does_not_raise_when_only_the_second_probe_fails(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """`/api/version` answers and `/api/tags` does not — a server shutting down between the
        two calls, which is exactly the moment a health probe is most likely to run. The typed
        error `_get_json` raises for the second call must not escape.
        """
        respx.get(f"{_BASE_URL}/api/version").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("version.json"))
        )
        respx.get(f"{_BASE_URL}/api/tags").mock(side_effect=httpx.ConnectError("refused"))

        health = _provider().health()

        assert health.status is ProviderStatus.UNAVAILABLE
        assert health.detail == "unreachable"

    @respx.mock
    def test_health_reports_degraded_when_the_server_answers_and_refuses(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """An authenticating proxy in front of Ollama returning 401 is a running server that
        will not serve *this* caller — a different operational state from nothing listening, and
        one an operator would otherwise go and check the wrong thing for.
        """
        respx.get(f"{_BASE_URL}/api/version").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("version.json"))
        )
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )

        health = _provider().health()

        assert health.status is ProviderStatus.DEGRADED
        assert "PROVIDER_REJECTED" in health.detail

    @respx.mock
    def test_a_degraded_health_detail_does_not_repeat_the_servers_message(self) -> None:
        """A health document is rendered into a UI; it must not become a fourth channel for a
        credential or a prompt echo to escape through (spec §14).
        """
        respx.get(f"{_BASE_URL}/api/version").mock(
            return_value=httpx.Response(200, json={"version": "0.32.13"})
        )
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(403, json={"error": "token sk-leaked-value rejected"})
        )

        health = _provider().health()

        assert "sk-leaked-value" not in health.detail


class TestDiscovery:
    """``list_models``, ``inspect_model`` and ``resolve`` — 0, 1 and many models."""

    @respx.mock
    def test_list_models_with_zero_models(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags_empty.json"))
        )

        assert list(_provider().list_models()) == []

    @respx.mock
    def test_list_models_enriches_every_entry_via_show(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )
        respx.post(f"{_BASE_URL}/api/show").mock(
            side_effect=lambda request: httpx.Response(
                200,
                json=load_ollama_fixture(
                    "show_llama.json"
                    if json.loads(request.content)["model"] == "llama3.2:3b-instruct-q4_0"
                    else "show_qwen.json"
                ),
            )
        )

        descriptors = _provider().list_models()

        assert len(descriptors) == 2
        qwen = next(d for d in descriptors if d.identity.provider_model_name == _MODEL)
        assert qwen.layers == 32
        assert qwen.kv_heads == 8
        assert qwen.attention_heads == 32
        assert qwen.max_context == 32768
        assert qwen.family == "qwen3"
        assert qwen.quantization == "Q8_0"

    @respx.mock
    def test_list_models_with_twenty_models(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        entries = [
            {
                "name": f"model-{index:02d}:latest",
                "digest": f"{index:064x}",
                "size": 1_000_000,
                "details": {"family": "test", "format": "gguf"},
            }
            for index in range(20)
        ]
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": entries})
        )
        respx.post(f"{_BASE_URL}/api/show").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("show_minimal.json"))
        )

        descriptors = _provider().list_models()

        assert len(descriptors) == 20
        assert {d.identity.provider_model_name for d in descriptors} == {e["name"] for e in entries}

    @respx.mock
    def test_inspect_model_merges_tags_and_show(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )
        respx.post(f"{_BASE_URL}/api/show").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("show_qwen.json"))
        )

        descriptor = _provider().inspect_model(_identity())

        assert descriptor.identity.artifact_digest is not None
        assert descriptor.layers == 32
        assert descriptor.raw["show"]["capabilities"] == ["completion", "tools", "thinking"]

    @respx.mock
    def test_inspect_model_of_an_unknown_name_raises_without_calling_show(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )
        show_route = respx.post(f"{_BASE_URL}/api/show")

        with pytest.raises(ModelNotFound) as raised:
            _provider().inspect_model(_identity("nope:latest"))

        assert raised.value.details == {"reference": "nope:latest", "known_model_count": 2}
        assert show_route.call_count == 0

    @respx.mock
    def test_show_complete_metadata(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )
        respx.post(f"{_BASE_URL}/api/show").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("show_qwen.json"))
        )

        descriptor = _provider().inspect_model(_identity())

        assert descriptor.parameter_count == 9_030_000_000
        assert descriptor.embedding_dim == 4096
        assert descriptor.vocab_size == 151936
        assert descriptor.head_dim == 128
        assert descriptor.declared_capabilities == frozenset(
            {ModelCapabilityFlag.TOOLS, ModelCapabilityFlag.THINKING}
        )
        assert descriptor.license_text is not None

    @respx.mock
    def test_show_partial_metadata_leaves_the_rest_unsupported(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )
        respx.post(f"{_BASE_URL}/api/show").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("show_minimal.json"))
        )

        descriptor = _provider().inspect_model(_identity())

        assert not is_supported(descriptor.parameter_count)
        assert not is_supported(descriptor.layers)
        assert not is_supported(descriptor.max_context)
        assert descriptor.architecture == "unknownarch"

    @respx.mock
    def test_show_with_no_model_info_at_all_degrades_every_field(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )
        respx.post(f"{_BASE_URL}/api/show").mock(return_value=httpx.Response(200, json={}))

        descriptor = _provider().inspect_model(_identity())

        assert descriptor.architecture is None
        assert not is_supported(descriptor.layers)
        assert not is_supported(descriptor.parameter_count)
        assert descriptor.raw["show"] == {}

    @respx.mock
    def test_an_exact_reference_resolves(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )

        assert _provider().resolve(_MODEL).provider_model_name == _MODEL

    @respx.mock
    def test_a_bare_name_resolves_through_ollamas_latest_convention(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(
                200, json={"models": [{"name": "phi4:latest", "digest": "a" * 64, "size": 1}]}
            )
        )

        assert _provider().resolve("phi4").provider_model_name == "phi4:latest"

    @respx.mock
    def test_a_unique_prefix_resolves(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )

        assert _provider().resolve("qwen3.5:9b").provider_model_name == _MODEL

    @respx.mock
    def test_an_ambiguous_prefix_is_refused(self) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen:7b", "digest": "a" * 64, "size": 1},
                        {"name": "qwen:14b", "digest": "b" * 64, "size": 1},
                    ]
                },
            )
        )

        with pytest.raises(ModelNotFound) as raised:
            _provider().resolve("qwen")

        assert raised.value.details["matched_model_count"] == 2

    @respx.mock
    def test_resolving_nothing_names_the_reference_and_count(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("tags.json"))
        )

        with pytest.raises(ModelNotFound) as raised:
            _provider().resolve("does-not-exist")

        assert raised.value.details == {"reference": "does-not-exist", "known_model_count": 2}

    @respx.mock
    def test_resolving_through_an_alias_is_logged_at_debug(
        self, load_ollama_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(
                200, json={"models": [{"name": "phi4:latest", "digest": "a" * 64, "size": 1}]}
            )
        )

        with caplog.at_level(logging.DEBUG, logger="modelrack.providers.ollama"):
            _provider().resolve("phi4")

        assert any(record.message == "ollama.model.resolved" for record in caplog.records)

    @respx.mock
    def test_the_library_logs_nothing_at_info_or_above(
        self, load_ollama_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec §17: a library must not configure or spam the host's logs."""
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(
                200, json={"models": [{"name": "phi4:latest", "digest": "a" * 64, "size": 1}]}
            )
        )

        with caplog.at_level(logging.INFO, logger="modelrack"):
            _provider().resolve("phi4")

        assert caplog.records == []


class TestDigestNormalization:
    """Spec §18: bare, prefixed, uppercase, truncated, non-hex and absent digests, each to the
    documented identity confidence — driven from recorded fixtures, not synthesized values.
    """

    @pytest.mark.parametrize(
        ("digest", "expected"),
        [
            (None, IdentityConfidence.NAME_ONLY),
            ("a" * 64, IdentityConfidence.DIGEST),
            ("sha256:" + "b" * 64, IdentityConfidence.DIGEST),
            ("SHA256:" + "C" * 64, IdentityConfidence.DIGEST),
            ("e" * 12, IdentityConfidence.NAME_ONLY),
            ("z" * 64, IdentityConfidence.NAME_ONLY),
            ("md5:" + "f" * 32, IdentityConfidence.NAME_ONLY),
        ],
        ids=[
            "absent",
            "bare-hex",
            "prefixed",
            "uppercase",
            "truncated",
            "non-hex",
            "wrong-algorithm",
        ],
    )
    @respx.mock
    def test_a_reported_digest_normalizes_or_is_discarded(
        self, digest: str | None, expected: IdentityConfidence
    ) -> None:
        entry: dict[str, Any] = {"name": "m:latest", "size": 1}
        if digest is not None:
            entry["digest"] = digest
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [entry]})
        )

        identity = _provider().resolve("m:latest")

        assert identity.identity_confidence is expected
        if expected is IdentityConfidence.DIGEST:
            assert identity.artifact_digest is not None
            assert identity.artifact_digest.startswith("sha256:")
            assert identity.artifact_digest.islower()

    @respx.mock
    def test_a_non_numeric_size_degrades_to_unsupported_rather_than_raising(self) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": "m:latest", "digest": "a" * 64, "size": "not-a-number"}]},
            )
        )
        respx.post(f"{_BASE_URL}/api/show").mock(return_value=httpx.Response(200, json={}))

        descriptor = _provider().inspect_model(_identity("m:latest"))

        assert not is_supported(descriptor.size_bytes)

    @respx.mock
    def test_a_discarded_digest_leaves_a_reason_in_raw(self) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(
                200, json={"models": [{"name": "m:latest", "digest": "not-a-digest", "size": 1}]}
            )
        )
        respx.post(f"{_BASE_URL}/api/show").mock(return_value=httpx.Response(200, json={}))

        descriptor = _provider().inspect_model(_identity("m:latest"))

        assert "not-a-digest" in descriptor.raw["digest_discarded_reason"]
        assert descriptor.identity.identity_confidence is IdentityConfidence.NAME_ONLY


class TestGeneration:
    """Non-streaming ``generate()``: text, token counts, all four backend durations, finish
    reasons.
    """

    @respx.mock
    def test_generate_returns_the_answer_and_finish_reason(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        )

        result = _provider().generate(_request())

        assert "KV caching" in result.text
        assert result.finish_reason is FinishReason.STOP
        assert result.identity == _identity()

    @respx.mock
    def test_generate_reports_token_counts(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        )

        usage = _provider().generate(_request()).usage

        assert usage.tokens.input_tokens == 26
        assert usage.tokens.output_tokens == 42

    @respx.mock
    def test_a_protocol_that_cannot_bill_a_cache_reports_zero_not_unsupported(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """ADR-0070 decision 3: Ollama has no cache-billing vocabulary, so `0` is a fact.

        The consequence is the point — with both cache classes counted rather than unavailable,
        `total_tokens` is a number and a price list without cache rates still totals, which is
        what ADR-0069 could not do for any real response before this rule.
        """
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        )

        tokens = _provider().generate(_request()).usage.tokens

        assert tokens.cache_read_tokens == 0
        assert tokens.cache_write_tokens == 0
        assert tokens.total_tokens == 68

    @respx.mock
    def test_a_terminal_payload_with_no_counts_reports_every_class_unsupported(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """Ollama's analogue of a response with no `usage` object: nothing reported, nothing known.

        The boundary the whole rule turns on. A payload that carries neither ``prompt_eval_count``
        nor ``eval_count`` has told this adapter nothing, and answering `0` for the cache classes
        here — where the counts are simply absent rather than unbillable — would be the fabricated
        zero ADR-0016 forbids.
        """
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200, json=load_ollama_fixture("chat_complete_no_counts.json")
            )
        )

        tokens = _provider().generate(_request()).usage.tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.output_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)
        assert not is_supported(tokens.total_tokens)

    @respx.mock
    def test_generate_extracts_all_four_backend_durations(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        )

        timing = _provider().generate(_request()).timing

        assert timing.backend_load_ms == pytest.approx(2.154458)
        assert timing.backend_prompt_eval_ms == pytest.approx(383.809)
        assert timing.backend_decode_ms == pytest.approx(4799.921)
        assert timing.backend_total_ms == pytest.approx(5191.566416)

    @respx.mock
    def test_generate_reports_no_first_token_moment(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """A blocking call has no boundary at which a first token could be observed."""
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("chat_complete.json"))
        )

        timing = _provider().generate(_request()).timing

        assert not is_supported(timing.client_ttft_ms)
        assert is_supported(timing.client_wall_ms)

    @pytest.mark.parametrize(
        ("done_reason", "expected"),
        [
            ("stop", FinishReason.STOP),
            ("length", FinishReason.LENGTH),
            (None, FinishReason.UNKNOWN),
        ],
    )
    @respx.mock
    def test_finish_reasons_map_from_done_reason(
        self, done_reason: str | None, expected: FinishReason
    ) -> None:
        payload: dict[str, Any] = {
            "model": _MODEL,
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
        }
        if done_reason is not None:
            payload["done_reason"] = done_reason
        respx.post(f"{_BASE_URL}/api/chat").mock(return_value=httpx.Response(200, json=payload))

        assert _provider().generate(_request()).finish_reason is expected

    @respx.mock
    def test_tool_calls_force_the_tool_calls_finish_reason_over_done_reason(self) -> None:
        """Ollama's own ``done_reason`` for a tool turn is ``"stop"`` — the same as an answer."""
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}
                        ],
                    },
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        result = _provider().generate(_request(tools=(_WEATHER_TOOL,)))

        assert result.finish_reason is FinishReason.TOOL_CALLS
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Berlin"}
        assert result.tool_calls[0].id

    @respx.mock
    def test_a_completion_style_prompt_reaches_generate_not_chat(self) -> None:
        chat_route = respx.post(f"{_BASE_URL}/api/chat")
        generate_route = respx.post(f"{_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(
                200,
                json={"model": _MODEL, "response": "answer", "done": True, "done_reason": "stop"},
            )
        )

        result = _provider().generate(GenerationRequest(identity=_identity(), prompt="hi"))

        assert result.text == "answer"
        assert chat_route.call_count == 0
        assert generate_route.call_count == 1

    def test_prompt_style_with_tools_is_refused(self) -> None:
        """Ollama's completion endpoint has no concept of tools at all."""
        with pytest.raises(CapabilityUnsupported) as raised:
            _provider().generate(
                GenerationRequest(identity=_identity(), prompt="hi", tools=(_WEATHER_TOOL,))
            )

        assert raised.value.details["capability"] == "tool_calling"

    @respx.mock
    def test_reasoning_content_is_surfaced_separately_from_the_answer(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "thinking": "Let me compute this.",
                    },
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        result = _provider().generate(_request())

        assert result.text == "42"
        assert result.thinking == "Let me compute this."

    @respx.mock
    def test_no_thinking_key_is_unsupported_not_empty(self) -> None:
        """UNSUPPORTED (not reported) is a different claim from '' (reported and empty)."""
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "42"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        result = _provider().generate(_request())

        assert not is_supported(result.thinking)

    @respx.mock
    def test_json_mode_sets_the_format_field(self) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "{}"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(_request(response_format=ResponseFormat(kind=ResponseFormatKind.JSON)))

        assert json.loads(route.calls.last.request.content)["format"] == "json"

    @respx.mock
    def test_json_schema_passes_the_schema_object_through(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "{}"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(
            _request(
                response_format=ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema=schema)
            )
        )

        assert json.loads(route.calls.last.request.content)["format"] == schema

    @respx.mock
    def test_sampling_parameters_map_to_ollama_options(self) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "x"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(
            _request(
                sampling=SamplingParameters(
                    temperature=0.2,
                    top_p=0.9,
                    top_k=40,
                    seed=7,
                    max_output_tokens=128,
                    stop=("</s>",),
                    repeat_penalty=1.1,
                ),
                runtime_profile=RuntimeProfile(
                    context_size=8192, gpu_layers=20, threads=8, batch_size=512
                ),
            )
        )

        options = json.loads(route.calls.last.request.content)["options"]
        assert options == {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 7,
            "num_predict": 128,
            "stop": ["</s>"],
            "repeat_penalty": 1.1,
            "num_ctx": 8192,
            "num_gpu": 20,
            "num_thread": 8,
            "num_batch": 512,
        }

    @respx.mock
    def test_provider_options_extends_and_overrides_named_options(self) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "x"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(
            _request(
                sampling=SamplingParameters(temperature=0.2),
                runtime_profile=RuntimeProfile(
                    provider_options={"temperature": 0.9, "mirostat": 2}
                ),
            )
        )

        options = json.loads(route.calls.last.request.content)["options"]
        assert options == {"temperature": 0.9, "mirostat": 2}

    @respx.mock
    def test_flash_attention_and_kv_precision_are_not_sent_as_options(self) -> None:
        """Both are server-startup-only in Ollama; sending them would promise a lie."""
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "x"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(
            _request(
                runtime_profile=RuntimeProfile(flash_attention=True, kv_cache_precision="q8_0")
            )
        )

        body = json.loads(route.calls.last.request.content)
        assert "flash_attention" not in body.get("options", {})
        assert "kv_cache_precision" not in body.get("options", {})

    @respx.mock
    def test_keep_alive_is_sent_at_the_top_level_not_inside_options(self) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "x"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(_request(runtime_profile=RuntimeProfile(keep_alive="10m")))

        body = json.loads(route.calls.last.request.content)
        assert body["keep_alive"] == "10m"
        assert "keep_alive" not in body.get("options", {})

    @respx.mock
    def test_caller_metadata_never_reaches_the_request_body(self) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "x"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(_request(metadata={"run_id": "secret-correlation-4a1f"}))

        assert "secret-correlation-4a1f" not in route.calls.last.request.content.decode()


class TestNdjsonChunking:
    """Phase 3's named concern: NDJSON lines and multi-byte characters split across raw chunks
    must reassemble exactly, whatever raw byte boundaries the transport happened to deliver.
    """

    @respx.mock
    def test_a_multibyte_character_split_across_two_raw_chunks_reassembles(self) -> None:
        """'café' with the 'é' (0xC3 0xA9) cut in half between two byte chunks."""
        first = b'{"model":"m","message":{"role":"assistant","content":"caf\xc3'
        second = (
            b'\xa9"},"done":false}\n'
            b'{"model":"m","message":{"role":"assistant","content":""},"done":true,'
            b'"done_reason":"stop"}\n'
        )
        respx.post(f"{_BASE_URL}/api/chat").mock(return_value=_ndjson_response(first, second))

        events = list(_provider().stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "café"
        assert [d.text for d in _text_deltas(events)] == ["café"]

    @respx.mock
    def test_a_json_line_split_across_two_raw_chunks_reassembles(self) -> None:
        """The newline itself, not just a character inside the line, can land mid-chunk."""
        whole_line = (
            b'{"model":"m","message":{"role":"assistant","content":"hello"},"done":false}\n'
        )
        terminal_line = (
            b'{"model":"m","message":{"role":"assistant","content":""},"done":true,'
            b'"done_reason":"stop"}\n'
        )
        split_at = 30
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                whole_line[:split_at], whole_line[split_at:], terminal_line
            )
        )

        events = list(_provider().stream(_request()))

        assert "".join(d.text for d in _text_deltas(events)) == "hello"

    @respx.mock
    def test_one_line_can_carry_several_raw_chunks(self) -> None:
        """The inverse boundary case: many small raw reads assembling one JSON line."""
        line = (
            b'{"model":"m","message":{"role":"assistant","content":"pieced together"},'
            b'"done":false}\n'
        )
        terminal_line = (
            b'{"model":"m","message":{"role":"assistant","content":""},"done":true,'
            b'"done_reason":"stop"}\n'
        )
        pieces = [line[i : i + 5] for i in range(0, len(line), 5)]
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(*pieces, terminal_line)
        )

        events = list(_provider().stream(_request()))

        assert "".join(d.text for d in _text_deltas(events)) == "pieced together"

    @respx.mock
    def test_recorded_multi_delta_stream_reassembles_into_the_recorded_answer(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                content=load_ollama_fixture("chat_stream.ndjson"),
                headers={"Content-Type": "application/x-ndjson"},
            )
        )

        events = list(_provider().stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "KV caching stores values."
        assert len(_text_deltas(events)) == 4


class TestStreaming:
    """The rest of the streaming contract: terminal event, truncation, mid-stream failure,
    cancellation.
    """

    @respx.mock
    def test_a_stream_ends_with_exactly_one_terminal_event(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=load_ollama_fixture("chat_stream.ndjson"))
        )

        events = list(_provider().stream(_request()))

        terminal = [e for e in events if isinstance(e, StreamCompleted | StreamFailed)]
        assert len(terminal) == 1
        assert events[-1] is terminal[0]

    @respx.mock
    def test_delta_indices_are_ordered_and_unique(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=load_ollama_fixture("chat_stream.ndjson"))
        )

        events = list(_provider().stream(_request()))
        indices = [
            e.index for e in events if isinstance(e, TokenDelta | ThinkingDelta | ToolCallDelta)
        ]

        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    @respx.mock
    def test_a_stream_reports_the_first_token_moment_it_observed(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=load_ollama_fixture("chat_stream.ndjson"))
        )

        events = list(_provider().stream(_request()))
        terminal = events[-1]

        assert isinstance(terminal, StreamCompleted)
        assert is_supported(terminal.result.timing.client_ttft_ms)
        assert is_supported(terminal.result.timing.client_wall_ms)

    @respx.mock
    def test_a_truncated_stream_is_reported_as_a_protocol_error(self) -> None:
        """Connection closed cleanly, but no ``done: true`` line ever arrived."""
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"partial"},"done":false}\n'
            )
        )

        events = list(_provider().stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "partial"

    @respx.mock
    def test_an_unparseable_line_is_delivered_as_a_protocol_error(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"ok"},"done":false}\n',
                b"not json at all\n",
            )
        )

        events = list(_provider().stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "ok"

    @respx.mock
    def test_a_mid_stream_in_band_error_is_delivered_not_raised(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """Ollama can signal a mid-generation failure as an NDJSON line, HTTP 200 already sent."""
        error_line = json.dumps(load_ollama_fixture("error_runner_stopped.json")).encode() + b"\n"
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"partial"},"done":false}\n',
                error_line,
            )
        )

        events = list(_provider().stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderRejected)
        assert terminal.partial_text == "partial"

    @respx.mock
    def test_a_mid_stream_context_overflow_is_classified_distinctly(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        error_line = json.dumps(load_ollama_fixture("error_context_overflow.json")).encode() + b"\n"
        respx.post(f"{_BASE_URL}/api/chat").mock(return_value=_ndjson_response(error_line))

        events = list(
            _provider().stream(_request(runtime_profile=RuntimeProfile(context_size=4096)))
        )

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ContextLimitExceeded)
        assert terminal.error.details["maximum_tokens"] == 4096

    @respx.mock
    def test_a_dropped_connection_mid_stream_is_a_protocol_error_not_unavailable(self) -> None:
        """A connection that already delivered content was not 'unreachable' — it broke."""

        class FlakyStream(httpx.SyncByteStream):
            def __iter__(self) -> Any:
                yield b'{"model":"m","message":{"role":"assistant","content":"hi"},"done":false}\n'
                raise httpx.ReadError("connection reset by peer")

            def close(self) -> None:
                pass

        respx.post(f"{_BASE_URL}/api/chat").mock(
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
                yield b'{"model":"m","message":{"role":"assistant","content":"hi"},"done":false}\n'
                raise httpx.ReadTimeout("timed out")

            def close(self) -> None:
                pass

        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, stream=SlowStream())
        )

        events = list(_provider().stream(_request()))

        assert isinstance(events[-1], StreamFailed)
        assert isinstance(events[-1].error, ProviderTimeout)

    @respx.mock
    def test_cancelling_mid_stream_stops_within_one_delta(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        from modelrack import CancellationToken  # noqa: PLC0415 — kept local to this test

        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=load_ollama_fixture("chat_stream.ndjson"))
        )
        token = CancellationToken()
        events: list[StreamEvent] = []

        for event in _provider().stream(_request(cancel=token)):
            events.append(event)
            if len(_text_deltas(events)) == 2 and not token.is_cancelled:
                token.cancel()

        assert len(_text_deltas(events)) == 2
        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == "KV caching"

    @respx.mock
    def test_cancelling_before_the_first_delta_yields_only_the_terminal_event(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        from modelrack import CancellationToken  # noqa: PLC0415 — kept local to this test

        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=load_ollama_fixture("chat_stream.ndjson"))
        )
        token = CancellationToken()
        token.cancel()

        events = list(_provider().stream(_request(cancel=token)))

        assert len(events) == 1
        assert isinstance(events[0], StreamFailed)
        assert events[0].partial_text == ""

    @respx.mock
    def test_abandoning_a_stream_early_closes_the_connection(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        """``.close()`` runs even when the caller never drains the iterator."""
        closed = {"count": 0}

        class TrackedStream(httpx.SyncByteStream):
            def __init__(self, body: bytes) -> None:
                self._body = body

            def __iter__(self) -> Any:
                yield self._body

            def close(self) -> None:
                closed["count"] += 1

        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200, stream=TrackedStream(load_ollama_fixture("chat_stream.ndjson"))
            )
        )

        iterator = _provider().stream(_request())
        next(iterator)
        del iterator
        import gc  # noqa: PLC0415 — forces the generator's finally block to run deterministically

        gc.collect()

        assert closed["count"] == 1

    @respx.mock
    def test_draining_a_stream_fully_closes_the_connection_exactly_once(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        closed = {"count": 0}

        class TrackedStream(httpx.SyncByteStream):
            def __init__(self, body: bytes) -> None:
                self._body = body

            def __iter__(self) -> Any:
                yield self._body

            def close(self) -> None:
                closed["count"] += 1

        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200, stream=TrackedStream(load_ollama_fixture("chat_stream.ndjson"))
            )
        )

        list(_provider().stream(_request()))

        assert closed["count"] == 1

    @respx.mock
    def test_streaming_tool_calls_emit_identity_then_argument_fragments(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"","tool_calls":'
                b'[{"function":{"name":"get_weather","arguments":{"city":"Berlin"}}}]},'
                b'"done":false}\n',
                b'{"model":"m","message":{"role":"assistant","content":""},"done":true,'
                b'"done_reason":"stop"}\n',
            )
        )

        events = list(_provider().stream(_request(tools=(_WEATHER_TOOL,))))
        deltas = [e for e in events if isinstance(e, ToolCallDelta)]

        assert deltas[0].name == "get_weather"
        assert deltas[0].arguments_fragment is None
        assert json.loads(deltas[1].arguments_fragment or "") == {"city": "Berlin"}
        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.finish_reason is FinishReason.TOOL_CALLS

    @respx.mock
    def test_streaming_reasoning_content_precedes_the_answer(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"","thinking":"hm"},'
                b'"done":false}\n',
                b'{"model":"m","message":{"role":"assistant","content":"answer"},"done":false}\n',
                b'{"model":"m","message":{"role":"assistant","content":""},"done":true,'
                b'"done_reason":"stop"}\n',
            )
        )

        events = list(_provider().stream(_request()))
        kinds = [type(e).__name__ for e in events]

        assert kinds.index("ThinkingDelta") < kinds.index("TokenDelta")
        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.thinking == "hm"
        assert terminal.result.text == "answer"

    @respx.mock
    def test_streaming_is_refused_before_any_content_when_the_model_is_missing(self) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                404, json={"error": "model 'qwen3.5:9b-q8_0' not found, try pulling it first"}
            )
        )

        with pytest.raises(ModelNotFound):
            _provider().stream(_request())


class TestErrors:
    """Every row of spec §13 this adapter can produce, with the documented ``details`` keys."""

    @respx.mock
    def test_connection_refused(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(ProviderUnavailable) as raised:
            _provider().generate(_request())

        assert raised.value.details["base_url"] == _BASE_URL

    @respx.mock
    def test_timeout(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(side_effect=httpx.ConnectTimeout("timed out"))

        with pytest.raises(ProviderTimeout) as raised:
            _provider().generate(_request())

        assert raised.value.details["base_url"] == _BASE_URL

    @respx.mock
    def test_404_model_not_found(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(404, json=load_ollama_fixture("error_model_not_found.json"))
        )
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )

        with pytest.raises(ModelNotFound) as raised:
            _provider().generate(_request())

        assert raised.value.details["reference"] == _MODEL
        assert raised.value.details["known_model_count"] == 0
        assert "not found" in str(raised.value)

    @respx.mock
    def test_400_bad_options(self, load_ollama_fixture: Callable[[str], Any]) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(400, json=load_ollama_fixture("error_bad_option.json"))
        )

        with pytest.raises(ProviderRejected) as raised:
            _provider().generate(_request())

        assert raised.value.details["status_code"] == 400
        assert raised.value.details["provider_message"] == "invalid option provided"

    @respx.mock
    def test_400_context_overflow_is_classified_distinctly(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                400, json=load_ollama_fixture("error_context_overflow.json")
            )
        )

        with pytest.raises(ContextLimitExceeded) as raised:
            _provider().generate(_request(runtime_profile=RuntimeProfile(context_size=4096)))

        assert raised.value.details["maximum_tokens"] == 4096
        assert not is_supported(raised.value.details["requested_tokens"])

    @respx.mock
    def test_context_overflow_with_no_configured_context_names_no_ceiling(self) -> None:
        """A provider that refuses without a number leaves the caller to bisect (spec §13)."""
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                400, json={"error": "prompt exceeds context length available"}
            )
        )

        with pytest.raises(ContextLimitExceeded) as raised:
            _provider().generate(_request())

        assert not is_supported(raised.value.details["maximum_tokens"])

    @respx.mock
    def test_a_non_json_body_is_a_protocol_error(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=b"<html>not json</html>")
        )

        with pytest.raises(ProviderProtocolError) as raised:
            _provider().generate(_request())

        assert "body" in raised.value.details

    @respx.mock
    def test_a_non_object_json_body_is_a_protocol_error(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(return_value=httpx.Response(200, json=[1, 2, 3]))

        with pytest.raises(ProviderProtocolError):
            _provider().generate(_request())

    @respx.mock
    def test_an_oversize_response_is_rejected(self) -> None:
        body = json.dumps(
            {
                "model": _MODEL,
                "message": {"role": "assistant", "content": "x" * 1000},
                "done": True,
                "done_reason": "stop",
            }
        ).encode()
        respx.post(f"{_BASE_URL}/api/chat").mock(return_value=httpx.Response(200, content=body))

        with pytest.raises(ProviderProtocolError) as raised:
            _provider(max_response_bytes=100).generate(_request())

        assert raised.value.details["limit_bytes"] == 100

    @respx.mock
    def test_an_unexpected_5xx_without_a_message_is_a_protocol_error(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(500, content=b"internal server error")
        )

        with pytest.raises(ProviderProtocolError) as raised:
            _provider().generate(_request())

        assert raised.value.details["status_code"] == 500

    @respx.mock
    def test_a_5xx_with_a_provider_message_is_still_classified_by_message(self) -> None:
        """The documented spec §13 row names 4xx; a 5xx carrying a real message is honoured too."""
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(503, json={"error": "server overloaded"})
        )

        with pytest.raises(ProviderRejected) as raised:
            _provider().generate(_request())

        assert raised.value.details["status_code"] == 503

    @respx.mock
    def test_an_in_band_error_on_a_non_streaming_call_still_raises(self) -> None:
        """Ollama can send HTTP 200 with only ``{"error": ...}`` even for a non-streamed call."""
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json={"error": "model runner has stopped"})
        )

        with pytest.raises(ProviderRejected):
            _provider().generate(_request())

    def test_every_documented_error_is_a_provider_error_never_a_raw_httpx_exception(self) -> None:
        """Spec §11.7, asserted at the type level for the whole hierarchy this module raises."""
        for error_type in (
            ProviderUnavailable,
            ProviderTimeout,
            ProviderProtocolError,
            ModelNotFound,
            ContextLimitExceeded,
            ProviderRejected,
            CapabilityUnsupported,
        ):
            assert issubclass(error_type, Exception)
            assert not issubclass(error_type, httpx.HTTPError)


class TestResidency:
    """``load``, ``unload`` and ``list_resident`` — all three gated on ``/api/ps`` first."""

    @respx.mock
    def test_load_when_not_resident_reports_a_real_load_time(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_empty.json"))
        )
        respx.post(f"{_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_load.json"))
        )

        result = _provider().load(_identity(), RuntimeProfile(context_size=8192))

        assert result.already_resident is False
        assert result.load_ms == pytest.approx(2100.0)
        assert result.profile_hash == RuntimeProfile(context_size=8192).profile_hash

    @respx.mock
    def test_load_when_already_resident_makes_no_generate_call(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )
        generate_route = respx.post(f"{_BASE_URL}/api/generate")

        result = _provider().load(_identity(), RuntimeProfile())

        assert result.already_resident is True
        assert not is_supported(result.load_ms)
        assert generate_route.call_count == 0

    @respx.mock
    def test_load_falls_back_to_client_measured_time_when_the_provider_reports_none(self) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(return_value=httpx.Response(200, json={"models": []}))
        respx.post(f"{_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(
                200, json={"model": _MODEL, "response": "", "done": True, "done_reason": "load"}
            )
        )

        result = _provider().load(_identity(), RuntimeProfile())

        assert is_supported(result.load_ms)

    @respx.mock
    def test_load_sends_no_prompt_key_at_all(self) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(return_value=httpx.Response(200, json={"models": []}))
        route = respx.post(f"{_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(
                200, json={"model": _MODEL, "response": "", "done": True, "done_reason": "load"}
            )
        )

        _provider().load(_identity(), RuntimeProfile())

        assert "prompt" not in json.loads(route.calls.last.request.content)

    @respx.mock
    def test_unload_when_resident_evicts_and_returns_true(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )
        route = respx.post(f"{_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("generate_unload.json"))
        )

        assert _provider().unload(_identity()) is True
        assert json.loads(route.calls.last.request.content)["keep_alive"] == 0

    @respx.mock
    def test_unload_when_not_resident_makes_no_call_and_returns_false(self) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(return_value=httpx.Response(200, json={"models": []}))
        generate_route = respx.post(f"{_BASE_URL}/api/generate")

        assert _provider().unload(_identity()) is False
        assert generate_route.call_count == 0

    @respx.mock
    def test_list_resident_empty(self) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(return_value=httpx.Response(200, json={"models": []}))

        assert list(_provider().list_resident()) == []

    @respx.mock
    def test_list_resident_reports_vram_and_expiry(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(
            return_value=httpx.Response(200, json=load_ollama_fixture("ps_resident.json"))
        )

        entries = _provider().list_resident()

        assert len(entries) == 1
        assert entries[0].identity.provider_model_name == _MODEL
        assert entries[0].vram_bytes == 9_895_000_000
        assert entries[0].total_bytes == 9_895_000_000
        assert entries[0].expires_at is not None

    @respx.mock
    def test_list_resident_reports_the_context_it_is_actually_served_at(self) -> None:
        """ADR-0023 §4's *reported* served context, which is not the advertised maximum.

        A descriptor's ``max_context`` says what the weights can do; this says what the running
        instance was configured to do. They differ whenever anything set ``num_ctx``, and a
        consumer that could not tell them apart would have to assume the larger one.
        """
        respx.get(f"{_BASE_URL}/api/ps").mock(
            return_value=httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": _MODEL,
                            "digest": "a" * 64,
                            "size": 5_274_117_078,
                            "size_vram": 5_274_117_078,
                            "context_length": 2048,
                        }
                    ]
                },
            )
        )

        entries = _provider().list_resident()

        assert entries[0].context_length == 2048

    @respx.mock
    def test_a_provider_that_does_not_report_context_says_unsupported(self) -> None:
        """Never a zero, and never the advertised maximum standing in for it (ADR-0016 §4)."""
        respx.get(f"{_BASE_URL}/api/ps").mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": _MODEL, "digest": "a" * 64, "size": 1, "size_vram": 1}]},
            )
        )

        entries = _provider().list_resident()

        assert entries[0].context_length is UNSUPPORTED

    @respx.mock
    def test_list_resident_sorts_by_name(self) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(
            return_value=httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "z-model:latest", "digest": "a" * 64, "size": 1, "size_vram": 1},
                        {"name": "a-model:latest", "digest": "b" * 64, "size": 1, "size_vram": 1},
                    ]
                },
            )
        )

        entries = _provider().list_resident()

        assert [e.identity.provider_model_name for e in entries] == [
            "a-model:latest",
            "z-model:latest",
        ]


class TestCoverageCompleting:
    """Edge cases and private-helper branches not reached by the behavioural classes above."""

    @respx.mock
    def test_a_connection_failure_before_streaming_begins_raises(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(ProviderUnavailable):
            _provider().stream(_request())

    @respx.mock
    def test_blank_lines_in_the_stream_are_skipped(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"a"},"done":false}\n',
                b"\n",
                b"   \n",
                b'{"model":"m","message":{"role":"assistant","content":""},"done":true,'
                b'"done_reason":"stop"}\n',
            )
        )

        events = list(_provider().stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "a"

    @respx.mock
    def test_a_non_object_json_line_mid_stream_is_a_protocol_error(self) -> None:
        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"ok"},"done":false}\n',
                b"[1, 2, 3]\n",
            )
        )

        events = list(_provider().stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "ok"

    @respx.mock
    def test_cancelling_exactly_on_the_terminal_line_still_reports_cancelled(self) -> None:
        """The post-loop cancellation check.

        The terminal line itself carries a final content delta, so cancellation lands *after*
        that delta is yielded but *before* the generator reaches the post-loop check that
        follows the ``break`` — the one branch the in-loop, top-of-iteration check cannot reach,
        because processing the terminal line never returns to the top of the loop.
        """
        from modelrack import CancellationToken  # noqa: PLC0415 — kept local to this test

        respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=_ndjson_response(
                b'{"model":"m","message":{"role":"assistant","content":"first"},"done":false}\n',
                b'{"model":"m","message":{"role":"assistant","content":"last"},"done":true,'
                b'"done_reason":"stop"}\n',
            )
        )
        token = CancellationToken()
        events: list[StreamEvent] = []

        for event in _provider().stream(_request(cancel=token)):
            events.append(event)
            if isinstance(event, TokenDelta) and event.text == "last":
                token.cancel()

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == "firstlast"

    @respx.mock
    def test_an_oversize_chunk_mid_stream_is_delivered_not_raised(self) -> None:
        ok_line = b'{"model":"m","message":{"role":"assistant","content":"ok"},"done":false}\n'
        big_line = (
            b'{"model":"m","message":{"role":"assistant","content":"' + b"y" * 200 + b'"},'
            b'"done":false}\n'
        )
        respx.post(f"{_BASE_URL}/api/chat").mock(return_value=_ndjson_response(ok_line, big_line))

        events = list(_provider(max_chunk_bytes=100).stream(_request()))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert terminal.partial_text == "ok"

    @respx.mock
    def test_a_per_request_timeout_overrides_the_client_default(self) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "x"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(_request(timeout_seconds=5.0))

        assert route.calls.last.request.extensions.get("timeout") == {
            "connect": 5.0,
            "read": 5.0,
            "write": 5.0,
            "pool": 5.0,
        }

    @respx.mock
    def test_an_oversize_error_body_is_reported_as_oversize_not_reclassified(self) -> None:
        big_body = json.dumps({"error": "x" * 1000}).encode()
        respx.post(f"{_BASE_URL}/api/chat").mock(return_value=httpx.Response(400, content=big_body))

        with pytest.raises(ProviderProtocolError) as raised:
            _provider(max_response_bytes=50).generate(_request())

        assert raised.value.details["limit_bytes"] == 50

    @respx.mock
    def test_tags_returning_an_unexpected_status_raises_a_typed_error(self) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(return_value=httpx.Response(500, content=b"boom"))

        with pytest.raises(ProviderProtocolError):
            _provider().list_models()

    @respx.mock
    def test_tags_connection_failure_is_translated(self) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(ProviderUnavailable):
            _provider().list_models()

    @respx.mock
    def test_a_malformed_tags_payload_yields_an_empty_catalogue(self) -> None:
        respx.get(f"{_BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={"unexpected": "shape"})
        )

        assert list(_provider().list_models()) == []

    def test_first_supported_returns_the_first_real_measurement(self) -> None:
        from baseaicore import UNSUPPORTED  # noqa: PLC0415 — kept local to this test

        assert OllamaProvider._first_supported(UNSUPPORTED, 5, 10) == 5  # noqa: SLF001

    def test_first_supported_falls_back_to_unsupported_when_nothing_qualifies(self) -> None:
        from baseaicore import UNSUPPORTED, is_supported  # noqa: PLC0415 — kept local to this test

        result = OllamaProvider._first_supported(UNSUPPORTED, UNSUPPORTED)  # noqa: SLF001

        assert not is_supported(result)

    @respx.mock
    def test_load_forwards_a_configured_keep_alive(self) -> None:
        respx.get(f"{_BASE_URL}/api/ps").mock(return_value=httpx.Response(200, json={"models": []}))
        route = respx.post(f"{_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(
                200, json={"model": _MODEL, "response": "", "done": True, "done_reason": "load"}
            )
        )

        _provider().load(_identity(), RuntimeProfile(keep_alive="15m"))

        assert json.loads(route.calls.last.request.content)["keep_alive"] == "15m"

    @respx.mock
    def test_stream_honours_a_per_request_timeout_override(
        self, load_ollama_fixture: Callable[[str], Any]
    ) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, content=load_ollama_fixture("chat_stream.ndjson"))
        )

        list(_provider().stream(_request(timeout_seconds=3.0)))

        assert route.calls.last.request.extensions.get("timeout") == {
            "connect": 3.0,
            "read": 3.0,
            "write": 3.0,
            "pool": 3.0,
        }

    @respx.mock
    def test_an_explicit_text_response_format_sends_no_format_field(self) -> None:
        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "x"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )

        _provider().generate(_request(response_format=ResponseFormat(kind=ResponseFormatKind.TEXT)))

        assert "format" not in json.loads(route.calls.last.request.content)

    @respx.mock
    def test_a_prior_tool_call_and_a_named_tool_message_are_sent_through(self) -> None:
        from modelrack import ToolCall  # noqa: PLC0415 — kept local to this test

        route = respx.post(f"{_BASE_URL}/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": _MODEL,
                    "message": {"role": "assistant", "content": "17C"},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        )
        history = (
            Message(role=Role.USER, content="weather in Berlin?"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=(ToolCall(id="c1", name="get_weather", arguments={"city": "Berlin"}),),
            ),
            Message(role=Role.TOOL, content="17C", tool_call_id="c1", name="get_weather"),
        )

        _provider().generate(_request(messages=history))

        sent_messages = json.loads(route.calls.last.request.content)["messages"]
        assert sent_messages[1]["tool_calls"] == [
            {"function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}
        ]
        assert sent_messages[2]["name"] == "get_weather"
