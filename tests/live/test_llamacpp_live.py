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
* :class:`TestAdapterCanary` is **I17's semantic half** and needs more again: two real LoRA
  adapter GGUFs for one base, named by ``MODELRACK_LLAMACPP_ADAPTERS`` (see the class docstring
  for exactly what to put there). It cannot be faked — the whole assertion is that two adapters
  produce *different* continuations of one prompt, which only real weights can do — so where the
  artefacts are absent it skips with them named, and never passes vacuously.

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
from baseaicore import (
    DataClassification,
    IdentityConfidence,
    RuntimeProfile,
    is_supported,
    normalize_digest,
)

from modelrack import (
    AdapterRegistration,
    AdapterStatus,
    FinishReason,
    GenerationRequest,
    Message,
    ProviderStatus,
    Role,
    SamplingParameters,
    StreamCompleted,
    ThinkingDelta,
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
_ADAPTERS = os.environ.get("MODELRACK_LLAMACPP_ADAPTERS")
_ADAPTER_BASE = os.environ.get("MODELRACK_LLAMACPP_ADAPTER_BASE")
_LIVE_PROMPT = "Name one thing a KV cache stores."
# A thinking model given no output cap spends its whole context on reasoning_content and answers
# nothing — observed on the reference machine with Qwen3.5 9B at context 4096: 57 s of decoding,
# an empty `text`. The cap keeps the journey short; the assertions accept reasoning as output.
_OUTPUT_CAP = SamplingParameters(max_output_tokens=256)
_CANARY_ADAPTER_COUNT = 2


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
            sampling=_OUTPUT_CAP,
        )
        result = provider.generate(request)
        thinking = result.thinking if isinstance(result.thinking, str) else ""
        assert result.text.strip() or thinking.strip(), "neither an answer nor reasoning"
        assert result.finish_reason in {FinishReason.STOP, FinishReason.LENGTH}
        tokens = result.usage.tokens
        print(  # noqa: T201 — the measurements are this test's output
            f"\nload_ms={loaded.load_ms:.0f} build={provider.health().provider_version} "
            f"finish={result.finish_reason.value} text_chars={len(result.text)} "
            f"thinking_chars={len(thinking)} tokens=in {tokens.input_tokens} out "
            f"{tokens.output_tokens} cache_read {tokens.cache_read_tokens} "
            f"prompt_ms={result.timing.backend_prompt_eval_ms} "
            f"decode_ms={result.timing.backend_decode_ms}"
        )
        assert is_supported(tokens.input_tokens) and tokens.input_tokens > 0
        assert is_supported(tokens.output_tokens) and tokens.output_tokens > 0
        assert is_supported(tokens.cache_read_tokens) and tokens.cache_read_tokens >= 0
        assert tokens.cache_write_tokens == 0
        assert is_supported(result.timing.backend_decode_ms)
        assert result.provider_version

        events = list(provider.stream(request))
        assert isinstance(events[-1], StreamCompleted)
        assert any(isinstance(event, TokenDelta | ThinkingDelta) for event in events)
        assert events[-1].result.text == "".join(
            event.text for event in events if isinstance(event, TokenDelta)
        )
        streamed = events[-1].result.usage.tokens
        assert is_supported(streamed.output_tokens)
        print(  # noqa: T201
            f"stream: {len(events) - 1} deltas, cache_read {streamed.cache_read_tokens} "
            f"(second request with a shared prefix), "
            f"ttft_ms={events[-1].result.timing.client_ttft_ms:.0f}"
        )

        native = provider.generate(
            GenerationRequest(
                identity=identity,
                prompt=_LIVE_PROMPT,
                runtime_profile=profile,
                sampling=SamplingParameters(max_output_tokens=64),
            )
        )
        assert native.text.strip()
        assert is_supported(native.usage.tokens.input_tokens)

        assert provider.unload(identity) is True
        assert provider.list_resident() == ()
        assert not PosixProcessTable().is_alive(pid), "unload must leave no process behind"
        assert not list(provider.supervisor.state_dir.glob("*.pid.json"))


@pytest.fixture(scope="module")
def adapter_paths() -> list[Path]:
    if not _ADAPTERS:
        _skip_or_fail(
            "I17's semantic canary needs two real LoRA adapter GGUFs for one base. Set "
            "MODELRACK_LLAMACPP_ADAPTERS to two .gguf paths separated by os.pathsep, and "
            "MODELRACK_LLAMACPP_ADAPTER_BASE to the model name they were trained on. See "
            "this class's docstring for how to produce them. No adapter GGUF exists on the "
            "machine this phase was written on, which is why this skips rather than passes."
        )
    paths = [Path(part) for part in (_ADAPTERS or "").split(os.pathsep) if part]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        _skip_or_fail(f"MODELRACK_LLAMACPP_ADAPTERS names files that do not exist: {missing}")
    if len(paths) < _CANARY_ADAPTER_COUNT:
        _skip_or_fail(
            f"The canary needs {_CANARY_ADAPTER_COUNT} adapters; "
            f"MODELRACK_LLAMACPP_ADAPTERS named {len(paths)}. One adapter cannot show that a "
            "prefix was not reused across a switch."
        )
    if not _ADAPTER_BASE:
        _skip_or_fail(
            "Set MODELRACK_LLAMACPP_ADAPTER_BASE to the model name the adapters were trained "
            "on — the compatibility check is by digest and fails closed, so it has to be the "
            "base actually served."
        )
    return paths


@pytest.fixture(scope="module")
def adapter_provider(
    model_directory: Path,
    adapter_paths: list[Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[LlamaCppProvider]:
    if shutil.which(_SERVER) is None and not Path(_SERVER).is_file():
        _skip_or_fail(f"{_SERVER!r} is not on PATH; the canary needs a real llama-server.")
    probe = LlamaCppProvider(
        model_directory,
        state_dir=tmp_path_factory.mktemp("canary-probe"),
        server_path=_SERVER,
    )
    base = probe.resolve(_ADAPTER_BASE or "")
    probe.close()
    assert base.artifact_digest is not None, "a served base is always digest-bound here"
    registrations = [
        AdapterRegistration(
            name=f"canary-{index}",
            artifact_path=path,
            artifact_sha256=sha256_of_file(path),
            base_model_name=base.provider_model_name,
            base_artifact_digest=base.artifact_digest,
            data_classification=DataClassification.CONFIDENTIAL,
        )
        for index, path in enumerate(adapter_paths[:_CANARY_ADAPTER_COUNT])
    ]
    instance = LlamaCppProvider(
        model_directory,
        state_dir=tmp_path_factory.mktemp("canary-state"),
        adapters=registrations,
        server_path=_SERVER,
        startup_timeout_seconds=600.0,
    )
    try:
        yield instance
    finally:
        instance.close()


class TestAdapterCanary:
    """I17's semantic half: one prompt, two adapters, two different continuations.

    **This is the assertion no fake can make.** The structural half —
    ``tests/contract/test_adapter_isolation.py`` — proves that ModelRack states the whole adapter
    configuration on every request and never pins a slot, which is what makes llama-server's own
    ``lora_should_clear_cache`` reachable. It cannot prove that the *server* then honours it,
    because a recorded transport has no KV cache to reuse. Only two real adapters on one real base
    can show that the second request did not continue the first one's prefix.

    **What it needs, precisely, so an operator can produce it without a conversation:**

    * One base GGUF already under ``MODELRACK_LLAMACPP_MODELS`` — any instruct model llama.cpp
      serves. Name it in ``MODELRACK_LLAMACPP_ADAPTER_BASE`` as the model name this adapter serves
      it under (its path below the models directory, without ``.gguf``).
    * **Two** LoRA adapters trained on **that exact base**, converted to GGUF with
      ``convert_lora_to_gguf.py`` from the llama.cpp tree, and behaviourally distinguishable —
      two adapters that answer alike prove nothing, so pick two whose styles differ visibly (a
      terse-answers LoRA and a verbose-explainer LoRA is enough). Any rank works.
    * ``MODELRACK_LLAMACPP_ADAPTERS`` set to the two ``.gguf`` paths, separated by ``os.pathsep``.

    The digests are computed here rather than declared, and the base is claimed **by digest**, so
    the run also exercises the fail-closed compatibility check against a real artifact.
    """

    def test_both_adapters_register_against_the_real_base(
        self, adapter_provider: LlamaCppProvider
    ) -> None:
        """Registration by digest against a real artifact, before anything is generated."""
        identity = adapter_provider.resolve(_ADAPTER_BASE or "")
        adapter_provider.generate(
            GenerationRequest(identity=identity, prompt="hi", sampling=_OUTPUT_CAP)
        )

        states = adapter_provider.list_adapters()
        assert len(states) == _CANARY_ADAPTER_COUNT
        for state in states:
            assert state.status is AdapterStatus.REGISTERED, state.reason
            assert state.base_confidence is IdentityConfidence.DIGEST
            assert state.server_id is not None

    def test_two_adapters_continue_one_prompt_differently(
        self, adapter_provider: LlamaCppProvider
    ) -> None:
        """The canary. Same prompt, same seed, two adapters — and two different answers.

        The prefix is deliberately shared and long enough to be worth caching, so a server that
        reused adapter A's KV prefix for adapter B would answer *identically*: the failure this
        test exists to catch looks like agreement, which is why it is written as a difference.
        The bare base runs last, so its answer is also compared — a bare-base request that
        inherited an adapter's prefix would match that adapter rather than differ from both.
        """
        identity = adapter_provider.resolve(_ADAPTER_BASE or "")
        sampling = SamplingParameters(max_output_tokens=64, temperature=0.0, seed=7)
        prompt = (
            "You are answering a question about caching in transformer inference. "
            "Consider the key-value cache carefully, then answer in one sentence. "
            "Question: what does a KV cache store, and why does it help?"
        )

        answers: dict[str, str] = {}
        cache_reads: dict[str, object] = {}
        for name in ("canary-0", "canary-1", None):
            result = adapter_provider.generate(
                GenerationRequest(identity=identity, prompt=prompt, adapter=name, sampling=sampling)
            )
            answers[name or "bare"] = result.text
            cache_reads[name or "bare"] = result.usage.tokens.cache_read_tokens
            if name is not None:
                assert result.adapter is not None
                assert result.adapter.name == name
                assert result.adapter_base_confidence is IdentityConfidence.DIGEST
            else:
                assert result.adapter is None

        print(f"\ncanary cache_read by subject: {cache_reads}")  # noqa: T201 — the evidence
        assert answers["canary-0"] != answers["canary-1"], (
            "Two adapters produced byte-identical continuations of one prompt at temperature 0. "
            "Either the adapters are behaviourally identical — pick two that differ — or a prefix "
            "computed under one was reused for the other, which is exactly what I17 forbids."
        )
        assert answers["bare"] not in (answers["canary-0"], answers["canary-1"]), (
            "The bare base answered exactly as one of the adapters did, which is what a request "
            "carrying no `lora` field would produce: llama-server restores the launch-time set "
            "and keeps the slot's prefix."
        )
