"""Tests for :mod:`modelrack.providers._llamacpp_wire` — pure translation of llama-server's shapes.

The usage table in that module's docstring is the contract under test here, case by case, and
the ``tokens_cached`` trap has a test of its own: a fixture carries it at its real value (prompt
plus output) so that an adapter which started reading it would fail the arithmetic, not a type
check.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from baseaicore import (
    IdentityConfidence,
    ModelIdentity,
    ProviderKind,
    RuntimeProfile,
    is_supported,
)

from modelrack import (
    FinishReason,
    GenerationRequest,
    Message,
    ResponseFormat,
    ResponseFormatKind,
    Role,
    SamplingParameters,
    ToolDefinition,
)
from modelrack.providers._gguf import ArraySummary, ArtifactStamp, GgufHeader
from modelrack.providers._llamacpp_wire import (
    LlamaCppError,
    build_chat_body,
    build_completion_body,
    build_descriptor,
    build_launch_argv,
    completion_finish_reason,
    header_kind,
    identity_for,
    is_shard,
    launch_flags,
    model_name_for,
    quantization_name,
    read_backend_timing,
    read_build_info,
    read_chat_usage,
    read_completion_usage,
    read_error,
    read_served_context,
    request_options,
)

_OBSERVED_AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
_IDENTITY = ModelIdentity(ProviderKind.LLAMACPP, "qwen3.5-9b-q8_0")
_STAMP = ArtifactStamp(size_bytes=9_000_000_000, mtime_ns=1, inode=2, device=3)


def _header(metadata: dict[str, Any], *, parameter_count: int = 14_000_000_000) -> GgufHeader:
    return GgufHeader(
        path=Path("/models/qwen3-14b.gguf"),
        version=3,
        tensor_count=443,
        metadata=metadata,
        parameter_count=parameter_count,
        stamp=_STAMP,
    )


def _chat_request(**overrides: Any) -> GenerationRequest:
    fields: dict[str, Any] = {
        "identity": _IDENTITY,
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


class TestDiscoveryHelpers:
    def test_model_name_is_the_path_below_the_root_without_the_suffix(self) -> None:
        root = Path("/models")

        assert (
            model_name_for(Path("/models/qwen3-14b.Q4_K_M.gguf"), root=root) == "qwen3-14b.Q4_K_M"
        )
        assert model_name_for(Path("/models/qwen/q.gguf"), root=root) == "qwen/q"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("big-00001-of-00003.gguf", True),
            ("big-00003-of-00003.gguf", True),
            ("big.gguf", False),
            ("big-1-of-3.gguf", False),
        ],
    )
    def test_shards_are_recognised_by_their_suffix(self, name: str, expected: bool) -> None:
        assert is_shard(Path(name)) is expected

    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            ({}, "model"),
            ({"general.type": "model"}, "model"),
            ({"general.type": "adapter"}, "adapter"),
            ({"general.type": "mmproj"}, "mmproj"),
            ({"general.type": ""}, "model"),
            ({"general.type": 7}, "model"),
        ],
    )
    def test_header_kind(self, metadata: dict[str, Any], expected: str) -> None:
        assert header_kind(_header(metadata)) == expected

    @pytest.mark.parametrize(
        ("file_type", "expected"),
        [
            (15, "Q4_K_M"),
            (18, "Q6_K"),
            (7, "Q8_0"),
            (1, "F16"),
            (99, None),
            (True, None),
            ("15", None),
        ],
    )
    def test_quantization_name(self, file_type: object, expected: str | None) -> None:
        assert quantization_name(file_type) == expected

    def test_identity_normalizes_the_digest_or_falls_back_to_name_only(self) -> None:
        digest = "sha256:" + "ab" * 32

        assert identity_for("m", digest).artifact_digest == digest
        assert identity_for("m", "SHA256:" + "AB" * 32).artifact_digest == digest
        assert identity_for("m", None).identity_confidence is IdentityConfidence.NAME_ONLY
        assert identity_for("m", "not-a-digest").identity_confidence is IdentityConfidence.NAME_ONLY


class TestDescriptor:
    _METADATA = {
        "general.architecture": "qwen3",
        "general.name": "Qwen3 14B",
        "general.file_type": 15,
        "general.license": "apache-2.0",
        "qwen3.block_count": 40,
        "qwen3.context_length": 40960,
        "qwen3.embedding_length": 5120,
        "qwen3.attention.head_count": 40,
        "qwen3.attention.head_count_kv": (8, 8, 8),
        "qwen3.attention.key_length": 128,
        "qwen3.attention.sliding_window": 1024,
        "qwen3.expert_count": 0,
        "qwen3.rope.freq_base": 1000000.0,
        "qwen3.rope.dimension_sections": (11, 11, 10),
        "tokenizer.ggml.tokens": ArraySummary(element_type="string", length=151936),
    }

    def test_every_field_the_file_states_is_read(self) -> None:
        descriptor = build_descriptor(
            _header(self._METADATA),
            name="qwen3-14b",
            digest="sha256:" + "0" * 64,
            observed_at=_OBSERVED_AT,
        )

        assert descriptor.identity == ModelIdentity(
            ProviderKind.LLAMACPP, "qwen3-14b", artifact_digest="sha256:" + "0" * 64
        )
        assert descriptor.identity.identity_confidence is IdentityConfidence.DIGEST
        assert descriptor.observed_at == _OBSERVED_AT
        assert descriptor.family == "qwen3"
        assert descriptor.architecture == "qwen3"
        assert descriptor.parameter_count == 14_000_000_000
        assert descriptor.quantization == "Q4_K_M"
        assert descriptor.weight_format == "gguf"
        assert descriptor.size_bytes == 9_000_000_000
        assert descriptor.max_context == 40960
        assert descriptor.embedding_dim == 5120
        assert descriptor.layers == 40
        assert descriptor.attention_heads == 40
        assert descriptor.kv_heads == 8, "a per-layer array whose entries agree is that number"
        assert descriptor.head_dim == 128
        assert descriptor.vocab_size == 151936
        assert descriptor.sliding_window == 1024
        assert descriptor.expert_count == 0
        assert descriptor.license_text == "apache-2.0"
        assert descriptor.rope_config == {
            "qwen3.rope.freq_base": 1000000.0,
            "qwen3.rope.dimension_sections": [11, 11, 10],
        }
        assert descriptor.raw["metadata"]["tokenizer.ggml.tokens"] == {
            "array": {"element_type": "string", "length": 151936}
        }
        assert descriptor.raw["path"] == "/models/qwen3-14b.gguf"
        assert descriptor.raw["tensor_count"] == 443

    def test_a_per_layer_array_that_disagrees_is_unsupported_not_averaged(self) -> None:
        metadata = dict(self._METADATA, **{"qwen3.attention.head_count_kv": (8, 4, 8)})

        descriptor = build_descriptor(
            _header(metadata), name="m", digest=None, observed_at=_OBSERVED_AT
        )

        assert not is_supported(descriptor.kv_heads)
        assert descriptor.raw["metadata"]["qwen3.attention.head_count_kv"] == [8, 4, 8]

    def test_a_file_without_an_architecture_reports_nothing_it_cannot_locate(self) -> None:
        descriptor = build_descriptor(
            _header({"general.name": "x"}, parameter_count=0),
            name="m",
            digest=None,
            observed_at=_OBSERVED_AT,
        )

        assert descriptor.family is None
        assert descriptor.architecture is None
        assert not is_supported(descriptor.layers)
        assert not is_supported(descriptor.vocab_size)
        assert not is_supported(descriptor.parameter_count)
        assert descriptor.quantization is None
        assert descriptor.rope_config is None
        assert descriptor.license_text is None
        assert descriptor.identity.identity_confidence is IdentityConfidence.NAME_ONLY

    def test_vocab_size_falls_back_to_a_kept_array_then_to_the_architecture_key(self) -> None:
        kept = build_descriptor(
            _header({"general.architecture": "a", "tokenizer.ggml.tokens": ("x", "y")}),
            name="m",
            digest=None,
            observed_at=_OBSERVED_AT,
        )
        keyed = build_descriptor(
            _header({"general.architecture": "a", "a.vocab_size": 32000}),
            name="m",
            digest=None,
            observed_at=_OBSERVED_AT,
        )

        assert kept.vocab_size == 2
        assert keyed.vocab_size == 32000

    def test_malformed_numbers_are_unsupported(self) -> None:
        descriptor = build_descriptor(
            _header({"general.architecture": "a", "a.block_count": "40", "a.context_length": True}),
            name="m",
            digest=None,
            observed_at=_OBSERVED_AT,
        )

        assert not is_supported(descriptor.layers)
        assert not is_supported(descriptor.max_context)


class TestLaunchFlags:
    def test_every_profile_field_becomes_its_flag(self) -> None:
        profile = RuntimeProfile(
            context_size=8192,
            kv_cache_precision="q8_0",
            gpu_layers=99,
            flash_attention=True,
            threads=8,
            batch_size=512,
            keep_alive="5m",
        )

        assert launch_flags(profile) == (
            "--ctx-size",
            "8192",
            "--n-gpu-layers",
            "99",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--flash-attn",
            "on",
            "--threads",
            "8",
            "--batch-size",
            "512",
        )

    def test_a_default_profile_sends_no_flags_and_flash_attention_off_is_explicit(self) -> None:
        assert launch_flags(RuntimeProfile()) == ()
        assert launch_flags(RuntimeProfile(flash_attention=False)) == ("--flash-attn", "off")

    def test_provider_options_split_into_launch_flags_and_request_options(self) -> None:
        profile = RuntimeProfile(
            provider_options={
                "min_p": 0.05,
                "--parallel": 2,
                "--no-mmap": True,
                "--mlock": False,
                "--cache-reuse": None,
                "--chat-template-file": "/srv/templates/t.jinja",
                "cache_prompt": False,
            }
        )

        assert launch_flags(profile) == (
            "--chat-template-file",
            "/srv/templates/t.jinja",
            "--no-mmap",
            "--parallel",
            "2",
        )
        assert request_options(profile) == {"min_p": 0.05, "cache_prompt": False}

    def test_the_argv_is_loopback_only_with_the_alias_and_profile_flags_last(self) -> None:
        argv = build_launch_argv(
            server_path="/opt/llama/llama-server",
            model_path=Path("/models/q.gguf"),
            alias="q",
            port=8180,
            profile=RuntimeProfile(context_size=4096, provider_options={"--parallel": 1}),
        )

        assert argv == (
            "/opt/llama/llama-server",
            "--model",
            "/models/q.gguf",
            "--alias",
            "q",
            "--host",
            "127.0.0.1",
            "--port",
            "8180",
            "--jinja",
            "--no-webui",
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
        )


class TestRequestBodies:
    def test_completion_body_uses_llamacpp_sampling_names(self) -> None:
        request = GenerationRequest(
            identity=_IDENTITY,
            prompt="Once upon",
            sampling=SamplingParameters(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                seed=7,
                max_output_tokens=64,
                stop=("\n\n",),
                repeat_penalty=1.1,
            ),
            runtime_profile=RuntimeProfile(provider_options={"min_p": 0.05, "--parallel": 1}),
        )

        body = build_completion_body(request, stream=True)

        assert body == {
            "prompt": "Once upon",
            "stream": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 7,
            "n_predict": 64,
            "stop": ["\n\n"],
            "repeat_penalty": 1.1,
            "min_p": 0.05,
        }

    def test_completion_body_json_mode_and_schema(self) -> None:
        json_mode = GenerationRequest(
            identity=_IDENTITY,
            prompt="p",
            response_format=ResponseFormat(kind=ResponseFormatKind.JSON),
        )
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        with_schema = GenerationRequest(
            identity=_IDENTITY,
            prompt="p",
            response_format=ResponseFormat(kind=ResponseFormatKind.JSON_SCHEMA, schema=schema),
        )
        text = GenerationRequest(
            identity=_IDENTITY,
            prompt="p",
            response_format=ResponseFormat(kind=ResponseFormatKind.TEXT),
        )

        assert build_completion_body(json_mode, stream=False)["json_schema"] == {"type": "object"}
        assert build_completion_body(with_schema, stream=False)["json_schema"] == schema
        assert "json_schema" not in build_completion_body(text, stream=False)

    def test_request_options_win_on_an_overlapping_key(self) -> None:
        request = GenerationRequest(
            identity=_IDENTITY,
            prompt="p",
            sampling=SamplingParameters(temperature=0.2),
            runtime_profile=RuntimeProfile(provider_options={"temperature": 0.9}),
        )

        assert build_completion_body(request, stream=False)["temperature"] == 0.9

    def test_chat_body_is_the_openai_shape_with_llamacpp_choices(self) -> None:
        tool = ToolDefinition(name="get_weather", description="d", parameters={"type": "object"})
        request = _chat_request(
            sampling=SamplingParameters(max_output_tokens=32, repeat_penalty=1.2),
            tools=(tool,),
            response_format=ResponseFormat(kind=ResponseFormatKind.JSON),
        )

        body = build_chat_body(request, alias="qwen3.5-9b-q8_0", stream=True)

        assert body["model"] == "qwen3.5-9b-q8_0"
        assert body["messages"] == [{"role": "user", "content": "Explain KV caching."}]
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["max_tokens"] == 32
        assert body["repeat_penalty"] == 1.2
        assert "repetition_penalty" not in body
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert body["response_format"] == {"type": "json_object"}

    def test_a_non_streaming_chat_body_carries_no_stream_options(self) -> None:
        body = build_chat_body(_chat_request(), alias="a", stream=False)

        assert body["stream"] is False
        assert "stream_options" not in body
        assert "tools" not in body
        assert "response_format" not in body


class TestCompletionUsage:
    """The native shape, row by row of the module docstring's table."""

    def test_cached_input_is_reconciled_out_of_the_prompt_total(self) -> None:
        payload = {
            "tokens_evaluated": 21,
            "tokens_predicted": 12,
            "tokens_cached": 33,
            "timings": {"cache_n": 8},
        }

        tokens = read_completion_usage(payload, text="hello world").tokens

        assert tokens.input_tokens == 13
        assert tokens.cache_read_tokens == 8
        assert tokens.output_tokens == 12
        assert tokens.cache_write_tokens == 0
        assert tokens.total_tokens == 33

    def test_tokens_cached_is_never_read_as_the_cached_class(self) -> None:
        """``tokens_cached`` is the slot's whole cache after generation — prompt plus output."""
        payload = {
            "tokens_evaluated": 21,
            "tokens_predicted": 12,
            "tokens_cached": 33,
            "timings": {"cache_n": 0},
        }

        tokens = read_completion_usage(payload, text="").tokens

        assert tokens.cache_read_tokens == 0
        assert tokens.input_tokens == 21

    def test_no_cached_input_field_at_all_is_zero(self) -> None:
        """A build that predates ``timings.cache_n`` cannot bill the class: input is the prompt."""
        without_cache_n = {
            "tokens_evaluated": 21,
            "tokens_predicted": 12,
            "timings": {"prompt_n": 21},
        }
        without_timings = {"tokens_evaluated": 21, "tokens_predicted": 12}

        for payload in (without_cache_n, without_timings):
            tokens = read_completion_usage(payload, text="").tokens
            assert tokens.input_tokens == 21
            assert tokens.cache_read_tokens == 0
            assert tokens.cache_write_tokens == 0

    @pytest.mark.parametrize("cache_n", ["eight", -1, 2.5, True, None, 22])
    def test_an_unreadable_or_impossible_cached_figure_refuses_both_halves(
        self, cache_n: object
    ) -> None:
        payload = {"tokens_evaluated": 21, "tokens_predicted": 12, "timings": {"cache_n": cache_n}}

        tokens = read_completion_usage(payload, text="").tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert tokens.output_tokens == 12
        assert tokens.cache_write_tokens == 0

    def test_an_unreadable_prompt_total_refuses_both_halves(self) -> None:
        payload = {"tokens_evaluated": "21", "tokens_predicted": 12, "timings": {"cache_n": 0}}

        tokens = read_completion_usage(payload, text="").tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.cache_read_tokens)

    def test_no_counts_at_all_is_every_class_unsupported(self) -> None:
        tokens = read_completion_usage({"content": "x", "timings": {"cache_n": 5}}, text="x").tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.output_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)

    def test_the_observations_are_always_present(self) -> None:
        usage = read_completion_usage({}, text="héllo wörld")

        assert usage.output_chars == 11
        assert usage.output_words == 2
        assert usage.output_bytes == 13


class TestChatUsage:
    def test_timings_cache_n_is_preferred_and_details_are_the_fallback(self) -> None:
        with_timings = {
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
            "timings": {"cache_n": 8},
        }
        details_only = {
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 8},
            }
        }

        assert read_chat_usage(with_timings, text="").tokens.cache_read_tokens == 8
        assert read_chat_usage(with_timings, text="").tokens.input_tokens == 13
        assert read_chat_usage(details_only, text="").tokens.cache_read_tokens == 8
        assert read_chat_usage(details_only, text="").tokens.input_tokens == 13

    def test_no_cached_input_field_is_zero(self) -> None:
        tokens = read_chat_usage(
            {"usage": {"prompt_tokens": 21, "completion_tokens": 12}}, text=""
        ).tokens

        assert (tokens.input_tokens, tokens.cache_read_tokens, tokens.cache_write_tokens) == (
            21,
            0,
            0,
        )

    @pytest.mark.parametrize(
        "details", [None, "x", {}, {"cached_tokens": "eight"}, {"cached_tokens": 30}]
    )
    def test_an_unreadable_details_object_refuses_both_halves(self, details: object) -> None:
        payload = {
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 12,
                "prompt_tokens_details": details,
            }
        }

        tokens = read_chat_usage(payload, text="").tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert tokens.output_tokens == 12

    @pytest.mark.parametrize("usage", [None, {}, "usage", 3])
    def test_an_absent_or_empty_usage_object_is_every_class_unsupported(
        self, usage: object
    ) -> None:
        tokens = read_chat_usage({"usage": usage, "timings": {"cache_n": 0}}, text="").tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.output_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert not is_supported(tokens.cache_write_tokens)

    def test_a_missing_usage_key_is_every_class_unsupported(self) -> None:
        tokens = read_chat_usage({"choices": []}, text="").tokens

        assert not is_supported(tokens.total_tokens)


class TestTimingAndFinish:
    def test_backend_timings_come_from_the_timings_object_only(self) -> None:
        timing = read_backend_timing({"timings": {"prompt_ms": 84.2, "predicted_ms": 240.5}})

        assert timing.backend_prompt_eval_ms == 84.2
        assert timing.backend_decode_ms == 240.5
        assert not is_supported(timing.backend_load_ms)
        assert not is_supported(timing.backend_total_ms)
        assert not is_supported(timing.client_wall_ms)

    @pytest.mark.parametrize("value", ["84", True, -1.0, float("nan"), float("inf"), None])
    def test_a_malformed_duration_is_unsupported(self, value: object) -> None:
        timing = read_backend_timing({"timings": {"prompt_ms": value, "predicted_ms": 1.0}})

        assert not is_supported(timing.backend_prompt_eval_ms)
        assert timing.backend_decode_ms == 1.0

    def test_no_timings_object_is_all_unsupported(self) -> None:
        assert read_backend_timing({"timings": "soon"}) == read_backend_timing({})
        assert not is_supported(read_backend_timing({}).backend_decode_ms)

    @pytest.mark.parametrize(
        ("stop_type", "expected"),
        [
            ("eos", FinishReason.STOP),
            ("word", FinishReason.STOP),
            ("limit", FinishReason.LENGTH),
            ("none", FinishReason.UNKNOWN),
            ("weird", FinishReason.UNKNOWN),
            (None, FinishReason.UNKNOWN),
            (3, FinishReason.UNKNOWN),
        ],
    )
    def test_stop_type_maps_to_a_finish_reason(
        self, stop_type: object, expected: FinishReason
    ) -> None:
        assert completion_finish_reason({"stop_type": stop_type}) is expected


class TestProps:
    def test_build_info_and_served_context(self) -> None:
        props = {"build_info": "b10792-3e1f9a2c", "default_generation_settings": {"n_ctx": 8192}}

        assert read_build_info(props) == "b10792-3e1f9a2c"
        assert read_served_context(props) == 8192

    @pytest.mark.parametrize(
        "props",
        [
            {},
            {"build_info": "", "default_generation_settings": 4},
            {"build_info": 3, "default_generation_settings": {"n_ctx": "8192"}},
        ],
    )
    def test_missing_or_malformed_props_are_absent_not_guessed(self, props: dict[str, Any]) -> None:
        assert read_build_info(props) is None
        assert not is_supported(read_served_context(props))


class TestErrors:
    def test_the_structured_shape_is_read_in_full(self) -> None:
        error = read_error(
            {
                "error": {
                    "code": 400,
                    "message": "request (9000 tokens) exceeds the available context size",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": 9000,
                    "n_ctx": 8192,
                }
            }
        )

        assert error == LlamaCppError(
            message="request (9000 tokens) exceeds the available context size",
            error_type="exceed_context_size_error",
            status_code=400,
            n_prompt_tokens=9000,
            n_ctx=8192,
        )
        assert error.is_context_overflow
        assert not error.is_not_ready

    def test_the_bare_string_shape(self) -> None:
        assert read_error({"error": "boom"}) == LlamaCppError(message="boom")

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"error": ""},
            {"error": None},
            {"error": {"code": 1}},
            {"error": {"message": ""}},
            [],
            "x",
        ],
    )
    def test_no_error(self, payload: object) -> None:
        assert read_error(payload) is None

    def test_malformed_fields_are_dropped_not_misread(self) -> None:
        error = read_error(
            {
                "error": {
                    "message": "m",
                    "type": 5,
                    "code": True,
                    "n_prompt_tokens": "9",
                    "n_ctx": False,
                }
            }
        )

        assert error == LlamaCppError(message="m")

    def test_context_overflow_is_recognised_from_prose_without_the_type(self) -> None:
        assert LlamaCppError(
            message="input is larger than the max context size. skipping"
        ).is_context_overflow
        assert not LlamaCppError(
            message="Cannot use both json_schema and grammar"
        ).is_context_overflow

    def test_not_ready(self) -> None:
        assert LlamaCppError(message="Loading model", error_type="unavailable_error").is_not_ready


def test_the_launch_key_inputs_are_deterministic() -> None:
    """Two equal profiles must produce one argv, or every request would restart the server."""
    first = launch_flags(RuntimeProfile(provider_options={"--b": 1, "--a": 2}))
    second = launch_flags(RuntimeProfile(provider_options={"--a": 2, "--b": 1}))

    assert first == second
