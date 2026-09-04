"""Live tests for the llama.cpp adapter: real GGUF files on disk, and a real ``llama-server``.

Marked ``@pytest.mark.live`` and deselected by default. The default suite proves this adapter
against recorded fixtures and a fake launcher (spec §18 acceptance criterion 3, extended: it
passes with no ``llama-server`` installed). These tests are the one place that premise is
deliberately broken, in two steps that need different things:

* :class:`TestRealArtifacts` needs only a directory of GGUF files —
  ``MODELRACK_LLAMACPP_MODELS`` — and proves that discovery, header parsing and hashing hold
  against multi-gigabyte real artifacts rather than the small files the unit tests write. It
  hashes **one** file, the smallest, and reports how long that took, because the cost of a
  content digest is a fact this adapter's handoff has to state rather than guess.
* :class:`TestRealServer` additionally needs the binary — ``MODELRACK_LLAMACPP_SERVER``, or
  ``llama-server`` on ``PATH`` — and runs the journey: health, load, resident, generate, stream,
  unload, and **no process left behind**, checked in the process table by pid.

Both skip when what they need is absent, and ``MODELRACK_REQUIRE_LLAMACPP=1`` turns that skip
into a failure the way ``MODELRACK_REQUIRE_OLLAMA`` does for the Ollama live suite: a silently
skipped provider is an untested provider, and the LA1 exit demonstration depends on this file
having actually run on the reference machine.

These assert **shape and plausibility**, never exact content (testing standards §3).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from baseaicore import IdentityConfidence, RuntimeProfile, is_supported, normalize_digest

from modelrack import (
    FinishReason,
    GenerationRequest,
    Message,
    ProviderStatus,
    Role,
    StreamCompleted,
    TokenDelta,
)
from modelrack.providers._gguf import read_gguf_header, sha256_of_file
from modelrack.providers._llamacpp_process import PosixProcessTable
from modelrack.providers._llamacpp_wire import header_kind, is_shard
from modelrack.providers.llamacpp import LlamaCppProvider

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.live

_MODELS = os.environ.get("MODELRACK_LLAMACPP_MODELS")
_SERVER = os.environ.get("MODELRACK_LLAMACPP_SERVER", "llama-server")
_REQUIRE = os.environ.get("MODELRACK_REQUIRE_LLAMACPP") == "1"
_LIVE_PROMPT = "Name one thing a KV cache stores."


def _skip_or_fail(reason: str) -> None:
    """Skip, unless ``MODELRACK_REQUIRE_LLAMACPP=1`` says an absent prerequisite must fail."""
    if _REQUIRE:
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def model_directory() -> Path:
    """Return the directory of real GGUF files, or skip the module."""
    if not _MODELS:
        _skip_or_fail(
            "Set MODELRACK_LLAMACPP_MODELS to a directory of GGUF files to run the llama.cpp "
            "live tests. Set MODELRACK_REQUIRE_LLAMACPP=1 to make this a failure instead."
        )
    directory = Path(_MODELS or "")
    if not directory.is_dir():
        _skip_or_fail(f"MODELRACK_LLAMACPP_MODELS={_MODELS!r} is not a directory.")
    return directory


@pytest.fixture(scope="module")
def base_files(model_directory: Path) -> list[Path]:
    """Every base-model GGUF under the directory, smallest first."""
    files = [
        path
        for path in sorted(model_directory.rglob("*.gguf"))
        if path.is_file() and not is_shard(path) and header_kind(read_gguf_header(path)) == "model"
    ]
    if not files:
        _skip_or_fail(f"No base-model GGUF files under {model_directory}.")
    return sorted(files, key=lambda path: path.stat().st_size)


@pytest.fixture(scope="module")
def provider(
    model_directory: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[LlamaCppProvider]:
    """Return an adapter over the real directory and the real launcher, closed at module end."""
    instance = LlamaCppProvider(
        model_directory,
        state_dir=tmp_path_factory.mktemp("llamacpp-state"),
        server_path=_SERVER,
        startup_timeout_seconds=600.0,
    )
    try:
        yield instance
    finally:
        instance.close()


class TestRealArtifacts:
    """Discovery, header parsing and hashing against the real files — no server needed."""

    def test_every_base_file_parses_and_names_an_architecture(self, base_files: list[Path]) -> None:
        for path in base_files:
            header = read_gguf_header(path)
            architecture = header.metadata.get("general.architecture")
            assert isinstance(architecture, str) and architecture, path
            assert is_supported(header.parameter_count) and header.parameter_count > 0, path
            assert header.metadata.get(f"{architecture}.block_count"), path

    def test_the_smallest_file_hashes_to_a_normalized_digest(self, base_files: list[Path]) -> None:
        """Times the one full-content digest this suite pays for, and prints it: the cost of
        identity on this machine is what the handoff reports, so it should be measured.
        """
        path = base_files[0]
        started = time.perf_counter()

        digest = sha256_of_file(path)

        elapsed_s = time.perf_counter() - started
        size_gb = path.stat().st_size / 1e9
        print(  # noqa: T201 — the measurement is this test's output
            f"\nsha256 of {path.name}: {size_gb:.2f} GB in {elapsed_s:.1f} s "
            f"({size_gb / elapsed_s:.2f} GB/s)"
        )
        assert normalize_digest(digest) == digest

    def test_resolve_pins_the_smallest_model_to_its_digest(
        self, provider: LlamaCppProvider, base_files: list[Path]
    ) -> None:
        """Resolves — and therefore hashes — one model only; ``list_models`` would hash them all."""
        name = base_files[0].relative_to(provider.model_directory).with_suffix("").as_posix()

        identity = provider.resolve(name)

        assert identity.identity_confidence is IdentityConfidence.DIGEST
        assert identity.artifact_digest == sha256_of_file(base_files[0])
        descriptor = provider.inspect_model(identity)
        assert descriptor.identity == identity
        assert is_supported(descriptor.layers)
        assert is_supported(descriptor.max_context)


@pytest.fixture(scope="module")
def llama_server_binary() -> str:
    """Return the ``llama-server`` executable, or skip the tests that need one."""
    if shutil.which(_SERVER) is None and not Path(_SERVER).is_file():
        _skip_or_fail(
            f"No llama-server at {_SERVER!r}. Install llama.cpp or set "
            "MODELRACK_LLAMACPP_SERVER to the binary."
        )
    return _SERVER


@pytest.mark.usefixtures("llama_server_binary")
class TestRealServer:
    """The journey against a real ``llama-server``: the LA1 critical path's first rung."""

    def test_health_is_ok_before_anything_is_loaded(self, provider: LlamaCppProvider) -> None:
        health = provider.health()

        assert health.status is ProviderStatus.OK
        assert health.provider_version is None, "no server yet: no build to report"

    def test_the_journey_loads_generates_streams_and_unloads_leaving_nothing(
        self, provider: LlamaCppProvider, base_files: list[Path]
    ) -> None:
        name = base_files[0].relative_to(provider.model_directory).with_suffix("").as_posix()
        identity = provider.resolve(name)
        profile = RuntimeProfile(context_size=4096)

        loaded = provider.load(identity, profile)
        assert loaded.already_resident is False
        assert is_supported(loaded.load_ms) and loaded.load_ms > 0
        handle = provider.supervisor.handle_for(name)
        assert handle is not None
        pid = handle.process.pid
        assert PosixProcessTable().is_alive(pid)

        resident = provider.list_resident()
        assert [r.identity for r in resident] == [identity]
        assert resident[0].context_length == 4096, "the served context, reported by /props"
        assert provider.health().provider_version, "the build, from /props"

        request = GenerationRequest(
            identity=identity,
            messages=(Message(role=Role.USER, content=_LIVE_PROMPT),),
            runtime_profile=profile,
        )
        result = provider.generate(request)
        assert result.text.strip()
        assert result.finish_reason in {FinishReason.STOP, FinishReason.LENGTH}
        tokens = result.usage.tokens
        assert is_supported(tokens.input_tokens) and tokens.input_tokens > 0
        assert is_supported(tokens.output_tokens) and tokens.output_tokens > 0
        assert is_supported(tokens.cache_read_tokens) and tokens.cache_read_tokens >= 0
        assert tokens.cache_write_tokens == 0
        assert is_supported(result.timing.backend_decode_ms)
        assert result.provider_version

        events = list(provider.stream(request))
        assert isinstance(events[-1], StreamCompleted)
        assert any(isinstance(event, TokenDelta) for event in events)
        assert events[-1].result.text == "".join(
            event.text for event in events if isinstance(event, TokenDelta)
        )
        assert is_supported(events[-1].result.usage.tokens.output_tokens)

        native = provider.generate(
            GenerationRequest(identity=identity, prompt=_LIVE_PROMPT, runtime_profile=profile)
        )
        assert native.text.strip()
        assert is_supported(native.usage.tokens.input_tokens)

        assert provider.unload(identity) is True
        assert provider.list_resident() == ()
        assert not PosixProcessTable().is_alive(pid), "unload must leave no process behind"
        assert not list(provider.supervisor.state_dir.glob("*.pid.json"))
