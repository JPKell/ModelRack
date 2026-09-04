"""Tests for :mod:`modelrack.providers.llamacpp` — the supervised llama.cpp adapter.

Every test runs against a recorded transport (``respx``), a fake launcher and a fake process
table from ``conftest.py``, and a model directory of GGUF files the test writes — never a
``llama-server``, which the machine this was written on does not have (spec §18 acceptance
criterion 3, extended to the third adapter). Fixtures live under
``tests/fixtures/providers/llamacpp/`` and are version-annotated in ``manifest.json``.

Four properties carry their own acceptance criteria beyond the conformance suite:

* **A request's runtime profile is the server's launch flags**, and a profile that differs from
  the running server's restarts it — :class:`TestResidency`.
* **Usage is read to ADR-0070's per-response rule on both wire shapes**, and ``tokens_cached``
  is never mistaken for cached input — :class:`TestGenerateNative`, :class:`TestGenerateChat`.
* **A server that exits is reported once, typed, with its stderr**, then respawned —
  :class:`TestResidency`.
* **Both streams end on their own terminal rule** — ``[DONE]`` for chat, ``stop: true`` for the
  native endpoint — and anything else is a truncation — :class:`TestStreaming`.
"""

from __future__ import annotations

import gc
import json
import sys
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import (
    UNSUPPORTED,
    IdentityConfidence,
    ModelIdentity,
    ProviderKind,
    RuntimeProfile,
    ValidationError,
    is_supported,
)

from conftest import FakeLauncher, FakeMonotonic, FakeProcessTable, FakeServerProcess, FakeSleep
from modelrack import (
    CapabilityUnsupported,
    ContextLimitExceeded,
    FinishReason,
    GenerationCancelled,
    GenerationRequest,
    Message,
    ModelNotFound,
    ProviderEvent,
    ProviderEventKind,
    ProviderProtocolError,
    ProviderRejected,
    ProviderStatus,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnavailableReason,
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
from modelrack.providers.llamacpp import InMemoryDigestStore, LlamaCppProvider
from modelrack.streaming import CancellationToken

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import datetime
    from pathlib import Path

    from modelrack import StreamEvent

_PORT = 18080
_BASE = f"http://127.0.0.1:{_PORT}"
_MODEL = "qwen3.5-9b-q8_0"
_SECOND_MODEL = "gemma/gemma-3-12b-it.Q4_K_M"
_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
_QWEN_METADATA: dict[str, Any] = {
    "general.architecture": "qwen35",
    "general.name": "Qwen3.5 9B",
    "general.file_type": 7,
    "general.license": "apache-2.0",
    "qwen35.block_count": 32,
    "qwen35.context_length": 262144,
    "qwen35.embedding_length": 4096,
    "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.rope.freq_base": 10000000.0,
    "tokenizer.ggml.tokens": [f"t{i}" for i in range(70)],
}


def _identity(name: str = _MODEL, *, digest: str | None = None) -> ModelIdentity:
    return ModelIdentity(ProviderKind.LLAMACPP, name, artifact_digest=digest)


def _request(**overrides: Any) -> GenerationRequest:
    fields: dict[str, Any] = {
        "identity": _identity(),
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def _prompt_request(**overrides: Any) -> GenerationRequest:
    fields: dict[str, Any] = {"identity": _identity(), "prompt": "Explain KV caching."}
    fields.update(overrides)
    return GenerationRequest(**fields)


def _text_deltas(events: Sequence[StreamEvent]) -> list[TokenDelta]:
    return [event for event in events if isinstance(event, TokenDelta)]


def _sse(text: str, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code, content=text.encode("utf-8"), headers={"Content-Type": "text/event-stream"}
    )


@pytest.fixture
def models(tmp_path: Path, gguf_writer: Callable[..., Path]) -> Path:
    """A model directory: one Qwen base at the root, one Gemma base in a subdirectory, and the
    things discovery must skip — a shard, an adapter, a projector and a file that is not GGUF.
    """
    directory = tmp_path / "models"
    (directory / "gemma").mkdir(parents=True)
    gguf_writer(
        directory / f"{_MODEL}.gguf",
        metadata=_QWEN_METADATA,
        tensors=(("token_embd.weight", (4096, 8)), ("output.bias", (16,))),
        payload=b"\x11" * 64,
    )
    gguf_writer(
        directory / "gemma" / "gemma-3-12b-it.Q4_K_M.gguf",
        metadata={
            "general.architecture": "gemma3",
            "general.file_type": 15,
            "gemma3.block_count": 48,
        },
        payload=b"\x22" * 64,
    )
    gguf_writer(directory / "big-00001-of-00002.gguf", metadata={"general.architecture": "x"})
    gguf_writer(directory / "big-00002-of-00002.gguf", metadata={"general.architecture": "x"})
    gguf_writer(directory / "factcheck-lora.gguf", metadata={"general.type": "adapter"})
    gguf_writer(directory / "mmproj-gemma.gguf", metadata={"general.type": "mmproj"})
    (directory / "broken.gguf").write_bytes(b"GGML" + b"\0" * 40)
    (directory / "notes.txt").write_text("not a model")
    return directory


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
def table() -> FakeProcessTable:
    return FakeProcessTable()


@pytest.fixture
def events() -> list[ProviderEvent]:
    return []


@pytest.fixture
def make_provider(
    models: Path,
    tmp_path: Path,
    launcher: FakeLauncher,
    table: FakeProcessTable,
    frozen_clock: Callable[[], datetime],
    events: list[ProviderEvent],
) -> Callable[..., LlamaCppProvider]:
    def _make(**overrides: Any) -> LlamaCppProvider:
        fields: dict[str, Any] = {
            "state_dir": tmp_path / "state",
            "server_path": sys.executable,
            "port_range": (_PORT, _PORT + 3),
            "launcher": launcher,
            "process_table": table,
            "port_is_free": lambda _port: True,
            "sleep": FakeSleep(),
            "monotonic": FakeMonotonic(),
            "clock": frozen_clock,
            "on_event": events.append,
            "startup_timeout_seconds": 5.0,
        }
        fields.update(overrides)
        return LlamaCppProvider(models, **fields)

    return _make


@pytest.fixture
def provider(make_provider: Callable[..., LlamaCppProvider]) -> Iterator[LlamaCppProvider]:
    instance = make_provider()
    yield instance
    instance.close()


class _FakeServer:
    """Recorded llama-server endpoints on every port a test's provider may spawn on.

    Call ``server(chat=..., completion=..., ...)`` to choose which recorded body each endpoint
    serves from now on — a filename, an :class:`httpx.Response`, or an exception to raise as the
    transport would; a streamed request is answered from the ``*_stream`` choice. Choices are
    read at request time, so a test can change what a *running* server answers. ``calls`` is
    the router's own log, for asserting on what was sent.
    """

    _DEFAULTS: dict[str, Any] = {
        "health": "health_ok.json",
        "props": "props.json",
        "chat": "chat_complete.json",
        "chat_stream": "chat_stream.sse",
        "completion": "completion.json",
        "completion_stream": "completion_stream.sse",
    }

    def __init__(self, router: respx.MockRouter, load: Callable[[str], Any]) -> None:
        self._router = router
        self._load = load
        self._plans: dict[int, dict[str, Any]] = {}

    @property
    def calls(self) -> Any:
        return self._router.calls

    def __call__(self, *, port: int = _PORT, **choices: Any) -> None:
        plan = self._plans.get(port)
        if plan is None:
            plan = dict(self._DEFAULTS)
            self._plans[port] = plan
            self._register(port, plan)
        plan.update(choices)

    def _respond(self, plan: dict[str, Any], key: str) -> httpx.Response:
        choice = plan[key]
        if isinstance(choice, Exception):
            raise choice
        if isinstance(choice, httpx.Response):
            return choice
        if choice.endswith(".sse"):
            return _sse(self._load(choice))
        return httpx.Response(200, json=self._load(choice))

    def _register(self, port: int, plan: dict[str, Any]) -> None:
        base = f"http://127.0.0.1:{port}"

        def streamed(request: httpx.Request) -> bool:
            return bool(json.loads(request.content).get("stream"))

        self._router.get(f"{base}/health").mock(
            side_effect=lambda _request: self._respond(plan, "health")
        )
        self._router.get(f"{base}/props").mock(
            side_effect=lambda _request: self._respond(plan, "props")
        )
        self._router.post(f"{base}/v1/chat/completions").mock(
            side_effect=lambda request: self._respond(
                plan, "chat_stream" if streamed(request) else "chat"
            )
        )
        self._router.post(f"{base}/completion").mock(
            side_effect=lambda request: self._respond(
                plan, "completion_stream" if streamed(request) else "completion"
            )
        )


@pytest.fixture
def server(load_llamacpp_fixture: Callable[[str], Any]) -> Iterator[_FakeServer]:
    with respx.mock(assert_all_called=False) as router:
        yield _FakeServer(router, load_llamacpp_fixture)


class TestConstructionAndHealth:
    def test_a_missing_model_directory_is_refused_at_construction(
        self, tmp_path: Path, make_provider: Callable[..., LlamaCppProvider]
    ) -> None:
        with pytest.raises(ValidationError) as raised:
            LlamaCppProvider(tmp_path / "absent", state_dir=tmp_path / "state", sleep=FakeSleep())

        assert raised.value.details["field"] == "model_directory"

    def test_capabilities_are_static_and_honest(self, provider: LlamaCppProvider) -> None:
        capabilities = provider.capabilities()

        assert capabilities.streaming
        assert capabilities.tool_calling
        assert capabilities.structured_output
        assert capabilities.json_mode
        assert capabilities.token_counts
        assert capabilities.token_level_chunks
        assert capabilities.force_unload
        assert capabilities.residency_query
        assert capabilities.context_configurable
        assert not capabilities.thinking_control
        assert not capabilities.logprobs
        assert not capabilities.kv_metrics
        assert not capabilities.embedding
        assert provider.kind is ProviderKind.LLAMACPP

    def test_health_is_unavailable_when_the_binary_cannot_be_found(
        self, make_provider: Callable[..., LlamaCppProvider], tmp_path: Path
    ) -> None:
        by_name = make_provider(server_path="no-such-llama-server-binary")
        by_path = make_provider(server_path=tmp_path / "nowhere" / "llama-server")

        for instance in (by_name, by_path):
            health = instance.health()
            assert health.status is ProviderStatus.UNAVAILABLE
            assert "llama-server not found" in health.detail
            assert health.base_url == _BASE
            assert health.is_remote is False
            assert is_supported(health.latency_ms)

    def test_health_while_idle_counts_models_without_hashing_any(
        self, make_provider: Callable[..., LlamaCppProvider]
    ) -> None:
        store = InMemoryDigestStore()
        instance = make_provider(digest_store=store)

        health = instance.health()

        assert health.status is ProviderStatus.OK
        assert health.model_count == 2
        assert health.provider_version is None
        assert "no server running" in health.detail
        assert "2 models, 0 resident" in health.detail
        assert len(store) == 0, "a health probe must never pay for a content digest"

    def test_health_with_a_running_server_reports_its_build(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        provider.load(_identity(), RuntimeProfile())

        health = provider.health()

        assert health.status is ProviderStatus.OK
        assert health.provider_version == "b10792-3e1f9a2c"
        assert health.base_url == _BASE
        assert "1 resident" in health.detail

    def test_health_is_degraded_when_a_running_server_stops_answering(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server()
        provider.load(_identity(), RuntimeProfile())
        server(health=httpx.Response(503, json=load_llamacpp_fixture("health_loading.json")))

        health = provider.health()

        assert health.status is ProviderStatus.DEGRADED
        assert "not answering" in health.detail

    def test_health_is_unavailable_when_the_directory_vanishes(
        self, provider: LlamaCppProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(*_args: Any, **_kwargs: Any) -> Any:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr("pathlib.Path.rglob", unreadable)

        health = provider.health()

        assert health.status is ProviderStatus.UNAVAILABLE
        assert "PROVIDER_UNAVAILABLE" in health.detail

    def test_repr_names_the_directory_and_ports(
        self, provider: LlamaCppProvider, models: Path
    ) -> None:
        assert str(models) in repr(provider)
        assert f"{_PORT}-{_PORT + 3}" in repr(provider)
        assert provider.model_directory == models


class TestDiscovery:
    def test_only_base_models_are_listed_sorted_with_digests(
        self, provider: LlamaCppProvider
    ) -> None:
        descriptors = provider.list_models()

        assert [d.identity.provider_model_name for d in descriptors] == [_SECOND_MODEL, _MODEL]
        assert all(d.identity.identity_confidence is IdentityConfidence.DIGEST for d in descriptors)
        assert all(d.weight_format == "gguf" for d in descriptors)

    def test_the_descriptor_carries_the_header(
        self, provider: LlamaCppProvider, models: Path, frozen_clock: Callable[[], datetime]
    ) -> None:
        descriptor = provider.inspect_model(_identity())

        assert descriptor.observed_at == frozen_clock()
        assert descriptor.family == "qwen35"
        assert descriptor.layers == 32
        assert descriptor.max_context == 262144
        assert descriptor.embedding_dim == 4096
        assert descriptor.kv_heads == 4
        assert descriptor.head_dim == 256
        assert descriptor.vocab_size == 70
        assert descriptor.quantization == "Q8_0"
        assert descriptor.parameter_count == 4096 * 8 + 16
        assert descriptor.size_bytes == (models / f"{_MODEL}.gguf").stat().st_size
        assert descriptor.license_text == "apache-2.0"
        assert descriptor.rope_config == {"qwen35.rope.freq_base": 10000000.0}
        assert descriptor.raw["metadata"]["general.name"] == "Qwen3.5 9B"

    def test_resolve_exact_filename_and_unique_prefix(self, provider: LlamaCppProvider) -> None:
        exact = provider.resolve(_MODEL)
        with_suffix = provider.resolve(f"{_MODEL}.gguf")
        prefix = provider.resolve("gemma/")

        assert exact.provider_model_name == _MODEL
        assert exact.identity_confidence is IdentityConfidence.DIGEST
        assert with_suffix == exact
        assert prefix.provider_model_name == _SECOND_MODEL

    def test_resolve_refuses_an_ambiguous_prefix_and_an_unknown_reference(
        self, provider: LlamaCppProvider
    ) -> None:
        with pytest.raises(ModelNotFound) as ambiguous:
            provider.resolve("")
        with pytest.raises(ModelNotFound) as unknown:
            provider.resolve("llama-70b")

        assert ambiguous.value.details["matched_model_count"] == 2
        assert unknown.value.details == {"reference": "llama-70b", "known_model_count": 2}

    def test_inspect_of_an_unknown_model_names_what_it_looked_for(
        self, provider: LlamaCppProvider
    ) -> None:
        with pytest.raises(ModelNotFound) as raised:
            provider.inspect_model(_identity("absent"))

        assert raised.value.details == {"reference": "absent", "known_model_count": 2}

    def test_a_digest_is_computed_once_per_file_and_refresh_recomputes_it(
        self, make_provider: Callable[..., LlamaCppProvider]
    ) -> None:
        store = _CountingStore()
        instance = make_provider(digest_store=store)

        instance.list_models()
        instance.list_models()
        instance.resolve(_MODEL)
        assert store.puts == 2, "two files, hashed once each"

        instance.list_models(refresh=True)
        assert store.puts == 4

    def test_a_replaced_file_gets_a_new_digest_and_header_without_refresh(
        self, provider: LlamaCppProvider, models: Path, gguf_writer: Callable[..., Path]
    ) -> None:
        before = provider.inspect_model(_identity())
        gguf_writer(
            models / f"{_MODEL}.gguf",
            metadata=dict(_QWEN_METADATA, **{"qwen35.block_count": 64}),
            payload=b"\x33" * 128,
        )

        after = provider.inspect_model(_identity())

        assert after.identity.artifact_digest != before.identity.artifact_digest
        assert after.layers == 64

    def test_the_header_cache_is_inspectable_and_clearable(
        self, provider: LlamaCppProvider
    ) -> None:
        provider.list_models()
        provider.list_models()
        stats = provider.metadata_cache_stats()

        assert stats.stores == 4, (
            "two bases, an adapter and a projector; the broken file is never stored"
        )
        assert stats.hits == 4
        assert provider.metadata_cache_ttl_seconds == 300.0

        provider.clear_metadata_cache()
        assert provider.metadata_cache_stats().entries == 0

    def test_a_zero_ttl_disables_the_header_cache(
        self, make_provider: Callable[..., LlamaCppProvider]
    ) -> None:
        instance = make_provider(metadata_ttl_seconds=0)

        instance.list_models()
        instance.list_models()

        assert instance.metadata_cache_stats().hits == 0

    def test_an_unreadable_directory_is_provider_unavailable(
        self, provider: LlamaCppProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unreadable(*_args: Any, **_kwargs: Any) -> Any:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr("pathlib.Path.rglob", unreadable)

        with pytest.raises(ProviderUnavailable) as raised:
            provider.list_models()

        assert raised.value.details["reason"] == ProviderUnavailableReason.LAUNCH_FAILED.value


def test_the_in_memory_digest_store_is_inspectable_and_clearable() -> None:
    store = InMemoryDigestStore()
    store.put("k", "sha256:" + "0" * 64)

    assert len(store) == 1
    assert store.get("k") == "sha256:" + "0" * 64
    store.clear()
    assert len(store) == 0
    assert store.get("k") is None


class _CountingStore(InMemoryDigestStore):
    """An in-memory store that counts writes, so a test can count hashes."""

    def __init__(self) -> None:
        super().__init__()
        self.puts = 0

    def put(self, key: str, digest: str) -> None:
        self.puts += 1
        super().put(key, digest)


class TestResidency:
    def test_load_spawns_the_server_with_the_profile_as_flags(
        self,
        provider: LlamaCppProvider,
        launcher: FakeLauncher,
        server: _FakeServer,
        models: Path,
    ) -> None:
        server()
        profile = RuntimeProfile(context_size=8192, gpu_layers=99, flash_attention=True)

        loaded = provider.load(_identity(), profile)

        assert loaded.already_resident is False
        assert is_supported(loaded.load_ms)
        assert loaded.profile_hash == profile.profile_hash
        argv = launcher.specs[0].argv
        assert argv[0] == sys.executable
        assert argv[1:3] == ("--model", str(models / f"{_MODEL}.gguf"))
        assert argv[3:5] == ("--alias", _MODEL)
        assert argv[5:9] == ("--host", "127.0.0.1", "--port", str(_PORT))
        assert argv[-6:] == ("--ctx-size", "8192", "--n-gpu-layers", "99", "--flash-attn", "on")
        assert launcher.specs[0].stderr_path.exists()

    def test_loading_twice_under_one_profile_is_already_resident(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        provider.load(_identity(), RuntimeProfile(context_size=8192))

        again = provider.load(_identity(), RuntimeProfile(context_size=8192, keep_alive="1h"))

        assert again.already_resident is True
        assert not is_supported(again.load_ms)
        assert len(launcher.specs) == 1, "keep_alive is not a launch flag, so no restart"

    def test_a_different_profile_restarts_the_server(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        server(port=_PORT + 1)
        provider.load(_identity(), RuntimeProfile(context_size=8192))

        reloaded = provider.load(_identity(), RuntimeProfile(context_size=4096))

        assert reloaded.already_resident is False
        assert launcher.processes[0].terminated
        assert ("--ctx-size", "4096") == launcher.specs[1].argv[-2:]
        assert len(provider.list_resident()) == 1

    def test_unload_terminates_and_reports_honestly(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        provider.load(_identity(), RuntimeProfile())

        assert provider.unload(_identity()) is True
        assert launcher.processes[0].terminated
        assert provider.unload(_identity()) is False
        assert provider.unload(_identity("never-loaded")) is False
        assert not list(provider.supervisor.state_dir.glob("*.pid.json"))

    def test_list_resident_reports_the_served_context_and_the_digest(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        assert list(provider.list_resident()) == []
        provider.load(_identity(), RuntimeProfile())

        resident = provider.list_resident()

        assert len(resident) == 1
        assert resident[0].identity == provider.resolve(_MODEL)
        assert resident[0].context_length == 8192
        assert not is_supported(resident[0].vram_bytes)
        assert not is_supported(resident[0].total_bytes)
        assert resident[0].expires_at is None

    def test_an_unreadable_props_leaves_the_context_unsupported_and_the_load_succeeds(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(props=httpx.Response(500, text="boom"))

        loaded = provider.load(_identity(), RuntimeProfile())

        assert loaded.already_resident is False
        assert not is_supported(provider.list_resident()[0].context_length)
        assert provider.health().provider_version is None

    def test_a_props_without_a_context_is_unsupported(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(props="props_no_context.json")
        provider.load(_identity(), RuntimeProfile())

        assert not is_supported(provider.list_resident()[0].context_length)

    def test_two_bases_can_be_resident_on_two_ports(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        server(port=_PORT + 1)

        provider.load(_identity(), RuntimeProfile())
        provider.load(_identity(_SECOND_MODEL), RuntimeProfile())

        assert [h.port for h in provider.supervisor.handles()] == [_PORT + 1, _PORT]
        assert [r.identity.provider_model_name for r in provider.list_resident()] == [
            _SECOND_MODEL,
            _MODEL,
        ]

    def test_a_digest_pinned_identity_that_no_longer_matches_is_refused(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        stale = _identity(digest="sha256:" + "f" * 64)

        with pytest.raises(ModelNotFound) as raised:
            provider.load(stale, RuntimeProfile())

        assert raised.value.details["reason"] == "digest_mismatch"
        assert raised.value.details["expected_digest"] == stale.artifact_digest
        assert raised.value.details["actual_digest"] == provider.resolve(_MODEL).artifact_digest

    def test_a_digest_pinned_identity_that_matches_is_served(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        pinned = provider.resolve(_MODEL)

        assert provider.load(pinned, RuntimeProfile()).identity == pinned
        assert provider.list_resident()[0].identity == pinned

    def test_a_crashed_server_is_dropped_from_the_listing(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        provider.load(_identity(), RuntimeProfile())
        launcher.processes[0].crash(137)

        assert list(provider.list_resident()) == []
        assert not list(provider.supervisor.state_dir.glob("*.pid.json"))

    def test_a_crash_is_reported_once_with_its_stderr_then_the_server_is_respawned(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        launcher.stderr_text = "ggml_cuda_init: CUDA error: out of memory\n"
        provider.load(_identity(), RuntimeProfile())
        launcher.processes[0].crash(137)

        with pytest.raises(ProviderUnavailable) as raised:
            provider.generate(_request())

        assert raised.value.details["reason"] == ProviderUnavailableReason.PROCESS_EXITED.value
        assert raised.value.details["exit_code"] == 137
        assert "out of memory" in raised.value.details["stderr_tail"]
        assert provider.generate(_request()).text, "the call after the report respawns"
        assert len(launcher.specs) == 2

    def test_another_models_crash_does_not_disturb_this_one(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        server(port=_PORT + 1)
        provider.load(_identity(), RuntimeProfile())
        provider.load(_identity(_SECOND_MODEL), RuntimeProfile())
        launcher.processes[1].crash(137)

        assert provider.generate(_request()).text
        assert [r.identity.provider_model_name for r in provider.list_resident()] == [_MODEL]

    def test_generate_under_a_different_profile_restarts_the_server(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        server(port=_PORT + 1)
        provider.load(_identity(), RuntimeProfile(context_size=8192))

        provider.generate(_request(runtime_profile=RuntimeProfile(context_size=4096)))

        assert launcher.processes[0].terminated
        assert launcher.specs[1].argv[-2:] == ("--ctx-size", "4096")

    def test_a_probe_that_cannot_connect_counts_as_not_ready(
        self, make_provider: Callable[..., LlamaCppProvider], server: _FakeServer
    ) -> None:
        server(health=httpx.ConnectError("refused"))
        instance = make_provider(startup_timeout_seconds=0.5)

        with pytest.raises(ProviderTimeout):
            instance.load(_identity(), RuntimeProfile())

    def test_a_props_that_is_not_an_object_is_ignored(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(props=httpx.Response(200, json=[1, 2]))
        provider.load(_identity(), RuntimeProfile())

        assert provider.health().provider_version is None

    def test_a_startup_failure_surfaces_from_generate_as_a_load_failure_event(
        self,
        provider: LlamaCppProvider,
        launcher: FakeLauncher,
        server: _FakeServer,
        events: list[ProviderEvent],
    ) -> None:
        server()
        launcher.plan(lambda spec: FakeServerProcess(41, exit_after_polls=1, exit_code=1))

        with pytest.raises(ProviderUnavailable):
            provider.generate(_request())

        kinds = [(e.operation, e.kind) for e in events]
        assert kinds == [
            ("load", ProviderEventKind.REQUEST_STARTED),
            ("load", ProviderEventKind.REQUEST_FAILED),
        ]

    def test_a_startup_timeout_surfaces_typed(
        self, make_provider: Callable[..., LlamaCppProvider], server: _FakeServer
    ) -> None:
        server(
            health=httpx.Response(503, json={"error": {"code": 503, "message": "Loading model"}})
        )
        instance = make_provider(startup_timeout_seconds=0.5)

        with pytest.raises(ProviderTimeout) as raised:
            instance.load(_identity(), RuntimeProfile())

        assert raised.value.details["limit_seconds"] == 0.5
        assert instance.list_resident() == ()

    def test_close_terminates_everything(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        launcher: FakeLauncher,
        server: _FakeServer,
    ) -> None:
        server()
        instance = make_provider()
        instance.load(_identity(), RuntimeProfile())

        instance.close()
        instance.close()

        assert launcher.processes[0].terminated
        assert instance.supervisor.handles() == ()

    def test_a_dropped_provider_terminates_its_servers(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        launcher: FakeLauncher,
        server: _FakeServer,
    ) -> None:
        """The safety net for a caller that never unloads: no orphan survives the adapter."""
        server()
        instance = make_provider()
        instance.load(_identity(), RuntimeProfile())

        del instance
        gc.collect()

        assert launcher.processes[0].terminated

    def test_load_and_unload_emit_events_without_paths_or_prompts(
        self, provider: LlamaCppProvider, server: _FakeServer, events: list[ProviderEvent]
    ) -> None:
        server()
        provider.load(_identity(), RuntimeProfile())
        provider.unload(_identity())

        assert [(e.operation, e.kind) for e in events] == [
            ("load", ProviderEventKind.REQUEST_STARTED),
            ("load", ProviderEventKind.REQUEST_COMPLETED),
            ("unload", ProviderEventKind.REQUEST_STARTED),
            ("unload", ProviderEventKind.REQUEST_COMPLETED),
        ]
        assert all(e.model_name == _MODEL for e in events)
        assert is_supported(events[1].elapsed_ms)


class TestGenerateChat:
    def test_a_chat_completion_is_fully_translated(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()

        result = provider.generate(_request())

        assert result.text.startswith("A KV cache stores")
        assert result.identity == _identity()
        assert result.finish_reason is FinishReason.STOP
        tokens = result.usage.tokens
        assert (tokens.input_tokens, tokens.output_tokens) == (21, 12)
        assert (tokens.cache_read_tokens, tokens.cache_write_tokens) == (0, 0)
        assert tokens.total_tokens == 33
        assert result.timing.backend_prompt_eval_ms == 84.213
        assert result.timing.backend_decode_ms == 240.518
        assert not is_supported(result.timing.backend_load_ms)
        assert is_supported(result.timing.client_wall_ms)
        assert not is_supported(result.timing.client_ttft_ms)
        assert result.provider_version == "b10792-3e1f9a2c"
        assert result.thinking is UNSUPPORTED
        assert result.raw["usage"]["prompt_tokens"] == 21

    def test_the_request_body_is_the_chat_shape_with_llamacpp_names(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        request = _request(
            sampling=SamplingParameters(
                temperature=0.1,
                top_p=0.9,
                top_k=20,
                seed=3,
                max_output_tokens=50,
                stop=("END",),
                repeat_penalty=1.3,
            ),
            tools=(_WEATHER_TOOL,),
            response_format=ResponseFormat(
                kind=ResponseFormatKind.JSON_SCHEMA, schema={"type": "object"}
            ),
            runtime_profile=RuntimeProfile(provider_options={"min_p": 0.05, "--parallel": 1}),
            metadata={"run_id": "correlation-9f1e"},
        )

        provider.generate(request)

        sent = json.loads(server.calls.last.request.content)
        assert sent["model"] == _MODEL
        assert sent["messages"] == [{"role": "user", "content": "Explain KV caching."}]
        assert sent["stream"] is False
        assert "stream_options" not in sent
        assert (sent["temperature"], sent["top_p"], sent["top_k"], sent["seed"]) == (
            0.1,
            0.9,
            20,
            3,
        )
        assert sent["max_tokens"] == 50
        assert sent["stop"] == ["END"]
        assert sent["repeat_penalty"] == 1.3
        assert sent["tools"][0]["function"]["name"] == "get_weather"
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["min_p"] == 0.05
        assert "--parallel" not in sent
        assert "correlation-9f1e" not in server.calls.last.request.content.decode()

    def test_cached_input_is_reconciled(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat="chat_complete_cached.json")

        tokens = provider.generate(_request()).usage.tokens

        assert (tokens.input_tokens, tokens.cache_read_tokens) == (13, 8)
        assert tokens.input_tokens + tokens.cache_read_tokens == 21

    def test_no_usage_object_is_every_class_unsupported(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat="chat_complete_no_usage.json")

        result = provider.generate(_request())

        assert not is_supported(result.usage.tokens.total_tokens)
        assert not is_supported(result.usage.tokens.cache_write_tokens)
        assert not is_supported(result.timing.backend_decode_ms)

    def test_an_unreadable_details_object_refuses_both_halves(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat="chat_complete_details_unreadable.json")

        tokens = provider.generate(_request()).usage.tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.cache_read_tokens)
        assert tokens.output_tokens == 12

    def test_reasoning_content_reaches_thinking(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat="chat_complete_reasoning.json")

        result = provider.generate(_request())

        assert isinstance(result.thinking, str)
        assert result.thinking.startswith("The user wants")

    def test_tool_calls_are_parsed(self, provider: LlamaCppProvider, server: _FakeServer) -> None:
        server(chat="chat_complete_tool_calls.json")

        result = provider.generate(_request(tools=(_WEATHER_TOOL,)))

        assert result.finish_reason is FinishReason.TOOL_CALLS
        assert result.tool_calls[0].id == "call_8b1d2f"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Berlin"}
        assert result.text == ""

    def test_a_missing_fingerprint_falls_back_to_the_props_build(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        body = load_llamacpp_fixture("chat_complete.json")
        del body["system_fingerprint"]
        server(chat=httpx.Response(200, json=body))

        assert provider.generate(_request()).provider_version == "b10792-3e1f9a2c"

    def test_generate_emits_started_and_completed(
        self, provider: LlamaCppProvider, server: _FakeServer, events: list[ProviderEvent]
    ) -> None:
        server()
        provider.generate(_request(metadata={"run_id": "r1"}))

        generate_events = [e for e in events if e.operation == "generate"]
        assert [e.kind for e in generate_events] == [
            ProviderEventKind.REQUEST_STARTED,
            ProviderEventKind.REQUEST_COMPLETED,
        ]
        assert generate_events[1].output_tokens == 12
        assert generate_events[1].finish_reason == "stop"
        assert generate_events[1].metadata == {"run_id": "r1"}


class TestGenerateNative:
    def test_a_prompt_reaches_the_native_endpoint(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        request = _prompt_request(
            sampling=SamplingParameters(max_output_tokens=64),
            response_format=ResponseFormat(kind=ResponseFormatKind.JSON),
        )

        result = provider.generate(request)

        sent = server.calls.last.request
        assert sent.url.path == "/completion"
        body = json.loads(sent.content)
        assert body["prompt"] == "Explain KV caching."
        assert body["n_predict"] == 64
        assert body["json_schema"] == {"type": "object"}
        assert result.text.startswith("A KV cache stores")
        assert result.finish_reason is FinishReason.STOP
        assert result.thinking is UNSUPPORTED
        assert result.provider_version == "b10792-3e1f9a2c", (
            "from /props: the native shape has no fingerprint"
        )
        assert result.raw["stop_type"] == "eos"

    def test_usage_on_the_native_shape_never_reads_tokens_cached(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        """``tokens_cached`` is 33 in the fixture — prompt plus output — and must not appear."""
        assert load_llamacpp_fixture("completion.json")["tokens_cached"] == 33
        server()

        tokens = provider.generate(_prompt_request()).usage.tokens

        assert (tokens.input_tokens, tokens.output_tokens) == (21, 12)
        assert (tokens.cache_read_tokens, tokens.cache_write_tokens) == (0, 0)

    @pytest.mark.parametrize(
        ("fixture", "expected"),
        [
            ("completion_cached.json", (13, 8)),
            ("completion_no_cache_n.json", (21, 0)),
            ("completion_no_timings.json", (21, 0)),
        ],
    )
    def test_the_three_cached_input_cases(
        self,
        provider: LlamaCppProvider,
        server: _FakeServer,
        fixture: str,
        expected: tuple[int, int],
    ) -> None:
        server(completion=fixture)

        tokens = provider.generate(_prompt_request()).usage.tokens

        assert (tokens.input_tokens, tokens.cache_read_tokens) == expected

    def test_no_counts_is_every_class_unsupported(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(completion="completion_no_counts.json")

        tokens = provider.generate(_prompt_request()).usage.tokens

        assert not is_supported(tokens.input_tokens)
        assert not is_supported(tokens.cache_read_tokens)

    def test_a_limit_stop_is_length(self, provider: LlamaCppProvider, server: _FakeServer) -> None:
        server(completion="completion_limit.json")

        assert provider.generate(_prompt_request()).finish_reason is FinishReason.LENGTH

    def test_tools_on_a_prompt_are_refused_before_anything_is_spawned(
        self, provider: LlamaCppProvider, launcher: FakeLauncher
    ) -> None:
        with pytest.raises(CapabilityUnsupported) as raised:
            provider.generate(_prompt_request(tools=(_WEATHER_TOOL,)))

        assert raised.value.details["capability"] == "tool_calling"
        assert launcher.specs == []


class TestErrors:
    """Every row of spec §13 this adapter can produce, with the documented type and details."""

    def test_a_context_overflow_carries_the_servers_own_numbers(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server(chat=httpx.Response(400, json=load_llamacpp_fixture("error_context_overflow.json")))

        with pytest.raises(ContextLimitExceeded) as raised:
            provider.generate(_request())

        assert raised.value.details == {"requested_tokens": 9000, "maximum_tokens": 8192}

    def test_a_context_overflow_without_numbers_falls_back_to_the_served_context(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server(
            chat=httpx.Response(
                400, json=load_llamacpp_fixture("error_context_overflow_no_counts.json")
            )
        )

        with pytest.raises(ContextLimitExceeded) as raised:
            provider.generate(_request())

        assert not is_supported(raised.value.details["requested_tokens"])
        assert raised.value.details["maximum_tokens"] == 8192, "from /props"

    def test_a_context_overflow_with_nothing_reported_uses_the_profile_then_unsupported(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        error = load_llamacpp_fixture("error_context_overflow_no_counts.json")
        server(props="props_no_context.json", chat=httpx.Response(400, json=error))

        with pytest.raises(ContextLimitExceeded) as from_profile:
            provider.generate(_request(runtime_profile=RuntimeProfile(context_size=2048)))
        assert from_profile.value.details["maximum_tokens"] == 2048

        server(port=_PORT + 1, props="props_no_context.json", chat=httpx.Response(400, json=error))
        with pytest.raises(ContextLimitExceeded) as unknown:
            provider.generate(_request(identity=_identity(_SECOND_MODEL)))
        assert not is_supported(unknown.value.details["maximum_tokens"])

    def test_a_rejected_request_preserves_the_message_and_type(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server(chat=httpx.Response(400, json=load_llamacpp_fixture("error_bad_request.json")))

        with pytest.raises(ProviderRejected) as raised:
            provider.generate(_request())

        assert raised.value.details["status_code"] == 400
        assert raised.value.details["provider_message"] == "Cannot use both json_schema and grammar"
        assert raised.value.details["error_type"] == "invalid_request_error"

    def test_a_server_error_with_a_message_is_rejected_not_protocol(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server(chat=httpx.Response(500, json=load_llamacpp_fixture("error_server.json")))

        with pytest.raises(ProviderRejected) as raised:
            provider.generate(_request())

        assert raised.value.details["status_code"] == 500

    def test_a_loading_server_is_unavailable_with_reason_not_ready(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server(chat=httpx.Response(503, json=load_llamacpp_fixture("error_loading.json")))

        with pytest.raises(ProviderUnavailable) as raised:
            provider.generate(_request())

        assert raised.value.details["reason"] == ProviderUnavailableReason.NOT_READY.value
        assert raised.value.details["base_url"] == _BASE

    def test_a_200_carrying_an_error_object_is_classified_the_same_way(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server(completion=httpx.Response(200, json=load_llamacpp_fixture("error_bad_request.json")))

        with pytest.raises(ProviderRejected):
            provider.generate(_prompt_request())

    def test_a_non_json_error_body_is_a_protocol_error_naming_the_status(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat=httpx.Response(502, text="<html>bad gateway</html>"))

        with pytest.raises(ProviderProtocolError) as raised:
            provider.generate(_request())

        assert raised.value.details["status_code"] == 502
        assert "bad gateway" in raised.value.details["body"]

    def test_a_json_error_body_without_a_message_is_a_protocol_error(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat=httpx.Response(418, json={"teapot": True}))

        with pytest.raises(ProviderProtocolError):
            provider.generate(_request())

    def test_a_non_object_success_body_is_a_protocol_error(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat=httpx.Response(200, json=[1, 2]))

        with pytest.raises(ProviderProtocolError):
            provider.generate(_request())

    def test_an_oversize_body_is_refused_under_the_cap(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        server: _FakeServer,
        load_llamacpp_fixture: Any,
    ) -> None:
        server()
        instance = make_provider(max_response_bytes=64)

        with pytest.raises(ProviderProtocolError) as raised:
            instance.generate(_request())

        assert raised.value.details["limit_bytes"] == 64

    def test_an_oversize_error_body_is_still_refused_under_the_cap(
        self, make_provider: Callable[..., LlamaCppProvider], server: _FakeServer
    ) -> None:
        server(chat=httpx.Response(400, json={"error": {"message": "x" * 200, "type": "t"}}))
        instance = make_provider(max_response_bytes=64)

        with pytest.raises(ProviderProtocolError) as raised:
            instance.generate(_request())

        assert raised.value.details["limit_bytes"] == 64

    def test_transport_failures_are_typed(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat=httpx.ConnectError("refused"))
        with pytest.raises(ProviderUnavailable):
            provider.generate(_request())

        server(chat=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderTimeout):
            provider.generate(_request())

        server(props=httpx.ConnectError("refused"))
        provider.unload(_identity())
        assert provider.load(_identity(), RuntimeProfile()).already_resident is False

    def test_a_per_request_timeout_reaches_the_transport(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()

        provider.generate(_request(timeout_seconds=7.5))

        assert server.calls.last.request.extensions["timeout"]["read"] == 7.5

    def test_generate_emits_failed_with_the_code(
        self,
        provider: LlamaCppProvider,
        server: _FakeServer,
        events: list[ProviderEvent],
        load_llamacpp_fixture: Any,
    ) -> None:
        server(chat=httpx.Response(400, json=load_llamacpp_fixture("error_bad_request.json")))

        with pytest.raises(ProviderRejected):
            provider.generate(_request())

        failed = [
            e
            for e in events
            if e.operation == "generate" and e.kind is ProviderEventKind.REQUEST_FAILED
        ]
        assert failed[0].error_code == "PROVIDER_REJECTED"


class TestStreaming:
    def test_a_chat_stream_delivers_deltas_then_a_completed_result_with_usage(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()

        events = list(provider.stream(_request()))

        deltas = _text_deltas(events)
        assert [d.text for d in deltas] == ["A KV", " cache", " stores", " keys", " and values."]
        assert [d.index for d in deltas] == [0, 1, 2, 3, 4]
        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "A KV cache stores keys and values."
        assert terminal.result.finish_reason is FinishReason.STOP
        tokens = terminal.result.usage.tokens
        assert (tokens.input_tokens, tokens.output_tokens, tokens.cache_read_tokens) == (21, 5, 0)
        assert terminal.result.timing.backend_decode_ms == 101.204
        assert is_supported(terminal.result.timing.client_ttft_ms)
        assert terminal.result.provider_version == "b10792-3e1f9a2c"
        assert terminal.result.raw["usage"]["completion_tokens"] == 5
        sent = json.loads(server.calls.last.request.content)
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}

    def test_a_chat_stream_with_cached_input_reconciles_from_the_usage_chunk(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat_stream="chat_stream_cached.sse")

        terminal = list(provider.stream(_request()))[-1]

        assert isinstance(terminal, StreamCompleted)
        tokens = terminal.result.usage.tokens
        assert (tokens.input_tokens, tokens.cache_read_tokens) == (13, 8)

    def test_a_chat_stream_without_a_usage_chunk_reports_every_class_unsupported(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat_stream="chat_stream_no_usage.sse")

        terminal = list(provider.stream(_request()))[-1]

        assert isinstance(terminal, StreamCompleted)
        assert not is_supported(terminal.result.usage.tokens.total_tokens)
        assert not is_supported(terminal.result.usage.tokens.cache_write_tokens)
        assert terminal.result.finish_reason is FinishReason.STOP
        assert terminal.result.raw == {}

    def test_reasoning_deltas_precede_text_and_assemble_into_thinking(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat_stream="chat_stream_reasoning.sse")

        events = list(provider.stream(_request()))

        thinking = [e for e in events if isinstance(e, ThinkingDelta)]
        assert [t.text for t in thinking] == ["The user wants", " a short answer."]
        assert thinking[0].index == 0
        assert _text_deltas(events)[0].index == 2
        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.thinking == "The user wants a short answer."

    def test_streamed_tool_calls_reassemble_across_fragments(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(chat_stream="chat_stream_tool_calls.sse")

        events = list(provider.stream(_request(tools=(_WEATHER_TOOL,))))

        fragments = [e for e in events if isinstance(e, ToolCallDelta)]
        assert len(fragments) == 3
        assert fragments[0].name == "get_weather"
        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.finish_reason is FinishReason.TOOL_CALLS
        assert terminal.result.tool_calls[0].arguments == {"city": "Berlin"}
        assert terminal.result.tool_calls[0].id == "call_8b1d2f"

    def test_a_stray_non_mapping_tool_call_fragment_is_skipped(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        body = (
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[7, {"index":0,"id":"c1",'
            '"function":{"name":"get_weather","arguments":"{}"}}]}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        server(chat_stream=_sse(body))

        events = list(provider.stream(_request(tools=(_WEATHER_TOOL,))))

        assert len([e for e in events if isinstance(e, ToolCallDelta)]) == 1
        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.tool_calls[0].name == "get_weather"

    def test_a_per_request_timeout_reaches_the_stream_transport(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()

        list(provider.stream(_request(timeout_seconds=9.5)))

        assert server.calls.last.request.extensions["timeout"]["read"] == 9.5

    def test_a_keyboard_interrupt_during_the_status_check_still_closes_the_response(
        self, provider: LlamaCppProvider, server: _FakeServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``KeyboardInterrupt`` is not a ``ProviderError``, so it takes the bare
        ``except BaseException`` path — which exists so a Ctrl-C mid-connect leaks no socket.
        """
        server(chat_stream=httpx.Response(400, json={"error": {"message": "x", "type": "t"}}))

        def interrupt(*_args: Any, **_kwargs: Any) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(provider, "_raise_for_status", interrupt)

        with pytest.raises(KeyboardInterrupt):
            provider.stream(_request())

    def test_a_native_stream_ends_on_its_stop_chunk_with_no_done_sentinel(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()

        events = list(provider.stream(_prompt_request()))

        assert [d.text for d in _text_deltas(events)] == [
            "A KV",
            " cache",
            " stores",
            " keys",
            " and values.",
        ]
        terminal = events[-1]
        assert isinstance(terminal, StreamCompleted)
        assert terminal.result.text == "A KV cache stores keys and values."
        assert terminal.result.finish_reason is FinishReason.STOP
        tokens = terminal.result.usage.tokens
        assert (tokens.input_tokens, tokens.output_tokens, tokens.cache_read_tokens) == (21, 5, 0)
        assert terminal.result.raw["stop"] is True
        assert terminal.result.provider_version == "b10792-3e1f9a2c"

    def test_a_native_stream_with_cached_input(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server(completion_stream="completion_stream_cached.sse")

        terminal = list(provider.stream(_prompt_request()))[-1]

        assert isinstance(terminal, StreamCompleted)
        assert (
            terminal.result.usage.tokens.input_tokens,
            terminal.result.usage.tokens.cache_read_tokens,
        ) == (13, 8)

    @pytest.mark.parametrize(
        ("request_factory", "fixture_key", "fixture"),
        [
            (_request, "chat_stream", "chat_stream_truncated.sse"),
            (_prompt_request, "completion_stream", "completion_stream_truncated.sse"),
        ],
    )
    def test_a_truncated_stream_is_a_failure_carrying_the_partial_text(
        self,
        provider: LlamaCppProvider,
        server: _FakeServer,
        request_factory: Callable[[], GenerationRequest],
        fixture_key: str,
        fixture: str,
    ) -> None:
        choices: dict[str, Any] = {fixture_key: fixture}
        server(**choices)

        events = list(provider.stream(request_factory()))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)
        assert "terminal" in str(terminal.error)
        assert terminal.partial_text == "A KV cache stores"
        assert len(_text_deltas(events)) == 3

    @pytest.mark.parametrize(
        ("request_factory", "fixture_key", "fixture"),
        [
            (_request, "chat_stream", "chat_stream_error.sse"),
            (_prompt_request, "completion_stream", "completion_stream_error.sse"),
        ],
    )
    def test_an_in_band_error_is_a_failure_carrying_the_partial_text(
        self,
        provider: LlamaCppProvider,
        server: _FakeServer,
        request_factory: Callable[[], GenerationRequest],
        fixture_key: str,
        fixture: str,
    ) -> None:
        choices: dict[str, Any] = {fixture_key: fixture}
        server(**choices)

        events = list(provider.stream(request_factory()))

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderRejected)
        assert terminal.error.details["provider_message"] == "context shift is disabled"
        assert terminal.partial_text

    @pytest.mark.parametrize("body", ["data: not json\n\n", "data: [1, 2]\n\n"])
    def test_a_malformed_event_is_a_protocol_failure(
        self, provider: LlamaCppProvider, server: _FakeServer, body: str
    ) -> None:
        server(chat_stream=_sse(body))

        terminal = list(provider.stream(_request()))[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderProtocolError)

    def test_cancellation_takes_effect_within_one_event_and_keeps_the_partial_text(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        token = CancellationToken()
        events: list[StreamEvent] = []
        for event in provider.stream(_request(cancel=token)):
            events.append(event)
            if len(_text_deltas(events)) == 2:
                token.cancel()

        terminal = events[-1]
        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, GenerationCancelled)
        assert terminal.partial_text == "A KV cache"
        assert len(_text_deltas(events)) == 2

    def test_a_token_already_cancelled_spawns_nothing_and_opens_nothing(
        self, provider: LlamaCppProvider, launcher: FakeLauncher, server: _FakeServer
    ) -> None:
        server()
        token = CancellationToken()
        token.cancel()

        events = list(provider.stream(_request(cancel=token)))

        assert len(events) == 1
        assert isinstance(events[0], StreamFailed)
        assert isinstance(events[0].error, GenerationCancelled)
        assert launcher.specs == []
        assert len(server.calls) == 0

    def test_a_pre_stream_rejection_raises_rather_than_streams(
        self, provider: LlamaCppProvider, server: _FakeServer, load_llamacpp_fixture: Any
    ) -> None:
        server(
            chat_stream=httpx.Response(
                400, json=load_llamacpp_fixture("error_context_overflow.json")
            )
        )

        with pytest.raises(ContextLimitExceeded):
            provider.stream(_request())

    def test_a_pre_stream_transport_failure_raises_typed(
        self, provider: LlamaCppProvider, server: _FakeServer, events: list[ProviderEvent]
    ) -> None:
        server(chat_stream=httpx.ConnectError("refused"))

        with pytest.raises(ProviderUnavailable):
            provider.stream(_request())

        assert events[-1].kind is ProviderEventKind.REQUEST_FAILED
        assert events[-1].operation == "stream"

    def test_a_transport_interruption_mid_stream_is_a_failure_not_a_raise(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        def chunks() -> Iterator[bytes]:
            yield b'data: {"choices":[{"index":0,"delta":{"content":"A KV"}}]}\n\n'
            raise httpx.ReadError("reset")

        server(chat_stream=httpx.Response(200, content=chunks()))

        events = list(provider.stream(_request()))

        assert isinstance(events[-1], StreamFailed)
        assert isinstance(events[-1].error, ProviderProtocolError)
        assert events[-1].partial_text == "A KV"

    def test_a_read_timeout_mid_stream_is_a_timeout_failure(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        def chunks() -> Iterator[bytes]:
            yield b'data: {"content":"A KV","stop":false}\n\n'
            raise httpx.ReadTimeout("slow")

        server(completion_stream=httpx.Response(200, content=chunks()))

        terminal = list(provider.stream(_prompt_request()))[-1]

        assert isinstance(terminal, StreamFailed)
        assert isinstance(terminal.error, ProviderTimeout)

    def test_an_oversize_line_is_refused_under_the_chunk_cap(
        self, make_provider: Callable[..., LlamaCppProvider], server: _FakeServer
    ) -> None:
        server()
        instance = make_provider(max_chunk_bytes=32)

        terminal = list(instance.stream(_request()))[-1]

        assert isinstance(terminal, StreamFailed)
        assert terminal.error.details["limit_bytes"] == 32

    def test_stream_events_are_observed(
        self, provider: LlamaCppProvider, server: _FakeServer, events: list[ProviderEvent]
    ) -> None:
        server()
        list(provider.stream(_request()))

        stream_events = [e for e in events if e.operation == "stream"]
        assert stream_events[0].kind is ProviderEventKind.REQUEST_STARTED
        assert [
            e.chunk_index for e in stream_events if e.kind is ProviderEventKind.CHUNK_RECEIVED
        ] == [0, 1, 2, 3, 4]
        assert stream_events[-1].kind is ProviderEventKind.REQUEST_COMPLETED
        assert stream_events[-1].output_tokens == 5

    def test_an_abandoned_stream_leaves_the_provider_usable(
        self, provider: LlamaCppProvider, server: _FakeServer
    ) -> None:
        server()
        for _ in provider.stream(_request()):
            break

        assert provider.generate(_request()).text
