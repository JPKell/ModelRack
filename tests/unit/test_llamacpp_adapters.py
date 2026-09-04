"""Tests for Phase 7 — adapter registration, selection, and the restarts that never cut in.

Every test here runs against a recorded transport (``respx``), the fake launcher and the fake
process table from ``conftest.py``: no ``llama-server``, no GPU, no adapter file larger than a test
writes. What is asserted is not that adapters *work* — that is the live canary's job — but that
every path through this adapter either names the subject it ran or refuses:

* an adapter is registered only against the base it declares, **verified by digest, fail closed**;
* a refusal is recorded with both digests, never a silent drop;
* an unknown or incompatible name is :class:`~modelrack.errors.AdapterNotFound`, never a bare-base
  generation under the caller's adapter subject;
* a newly registered adapter is ``pending_restart`` and folds in **at an idle**, never mid-work;
* a bare-base request is byte-for-byte what it was before the adapter axis existed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import (
    DataClassification,
    IdentityConfidence,
    ModelIdentity,
    ProviderKind,
    RuntimeProfile,
)

from conftest import FakeLauncher, FakeMonotonic, FakeProcessTable, FakeSleep
from modelrack import (
    AdapterNotFound,
    AdapterRegistration,
    AdapterStatus,
    CapabilityUnsupported,
    GenerationRequest,
    Message,
    ProviderRejected,
    ProviderUnavailable,
    ProviderUnavailableReason,
    Role,
    StreamCompleted,
)
from modelrack.providers.llamacpp import LlamaCppProvider
from modelrack.testing import FakeProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime
    from pathlib import Path

    from modelrack import StreamEvent

_PORT = 18180
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_MODEL = "qwen3.5-9b-q8_0"
_OTHER_MODEL = "gemma/gemma-3-12b-it"
_STALE_DIGEST = "sha256:" + "de" * 32


# ------------------------------------------------------------------------------- scaffolding


@pytest.fixture
def models(tmp_path: Path, gguf_writer: Callable[..., Path]) -> Path:
    """Two bases, so "this adapter is for another base" is a state a test can reach."""
    directory = tmp_path / "models"
    (directory / "gemma").mkdir(parents=True)
    gguf_writer(
        directory / f"{_MODEL}.gguf",
        metadata={"general.architecture": "qwen35", "qwen35.block_count": 32},
        payload=b"\x11" * 64,
    )
    gguf_writer(
        directory / "gemma" / "gemma-3-12b-it.gguf",
        metadata={"general.architecture": "gemma3", "gemma3.block_count": 48},
        payload=b"\x22" * 64,
    )
    return directory


@pytest.fixture
def adapter_files(tmp_path: Path) -> dict[str, Path]:
    """Two adapter artifacts. Their bytes are never parsed here — only hashed and passed."""
    directory = tmp_path / "adapters"
    directory.mkdir()
    written: dict[str, Path] = {}
    for name, payload in (("factcheck", b"\xa1" * 96), ("house-voice", b"\xb2" * 96)):
        path = directory / f"{name}.gguf"
        path.write_bytes(payload)
        written[name] = path
    return written


def _digest_of(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def base_digest(models: Path) -> str:
    return _digest_of(models / f"{_MODEL}.gguf")


@pytest.fixture
def registration(adapter_files: dict[str, Path]) -> Callable[..., AdapterRegistration]:
    def _make(name: str = "factcheck", **overrides: Any) -> AdapterRegistration:
        fields: dict[str, Any] = {
            "name": name,
            "artifact_path": adapter_files[name],
            "artifact_sha256": _digest_of(adapter_files[name]),
            "base_model_name": _MODEL,
            "data_classification": DataClassification.INTERNAL,
        }
        fields.update(overrides)
        return AdapterRegistration(**fields)

    return _make


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture(autouse=True)
def server(
    launcher: FakeLauncher, load_llamacpp_fixture: Callable[[str], Any]
) -> Iterator[respx.MockRouter]:
    """llama-server on every port a test may spawn on, answering from its own command line.

    ``GET /lora-adapters`` is rendered from the ``--lora`` flags the launcher actually saw, the
    way the real server renders it from ``params_base.lora_adapters`` — which is what makes "the
    id is the server's, not argv order" a claim a test can break.
    """

    def lora(request: httpx.Request) -> httpx.Response:
        port = int(request.url.port or 0)
        argv: tuple[str, ...] = ()
        for spec in reversed(launcher.specs):
            if spec.port == port:
                argv = tuple(spec.argv)
                break
        paths = [argv[index + 1] for index, flag in enumerate(argv) if flag == "--lora"]
        return httpx.Response(
            200,
            json=[
                {"id": index, "path": path, "scale": 1.0, "task_name": "", "prompt_prefix": ""}
                for index, path in enumerate(paths)
            ],
        )

    def chat(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                content=load_llamacpp_fixture("chat_stream.sse").encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json=load_llamacpp_fixture("chat_complete.json"))

    with respx.mock(assert_all_called=False) as router:
        for offset in range(4):
            base = f"http://127.0.0.1:{_PORT + offset}"
            router.get(f"{base}/health").mock(
                return_value=httpx.Response(200, json=load_llamacpp_fixture("health_ok.json"))
            )
            router.get(f"{base}/props").mock(
                return_value=httpx.Response(200, json=load_llamacpp_fixture("props.json"))
            )
            router.get(f"{base}/lora-adapters").mock(side_effect=lora)
            router.post(f"{base}/v1/chat/completions").mock(side_effect=chat)
            router.post(f"{base}/completion").mock(
                return_value=httpx.Response(200, json=load_llamacpp_fixture("completion.json"))
            )
        yield router


@pytest.fixture
def make_provider(
    models: Path,
    tmp_path: Path,
    launcher: FakeLauncher,
    frozen_clock: Callable[[], datetime],
) -> Callable[..., LlamaCppProvider]:
    def _make(**overrides: Any) -> LlamaCppProvider:
        fields: dict[str, Any] = {
            "state_dir": tmp_path / "state",
            "server_path": sys.executable,
            "port_range": (_PORT, _PORT + 3),
            "launcher": launcher,
            "process_table": FakeProcessTable(),
            "port_is_free": lambda _port: True,
            "sleep": FakeSleep(),
            "monotonic": FakeMonotonic(),
            "clock": frozen_clock,
        }
        fields.update(overrides)
        return LlamaCppProvider(models, **fields)

    return _make


def _identity(name: str = _MODEL) -> ModelIdentity:
    return ModelIdentity(ProviderKind.LLAMACPP, name)


def _request(**overrides: Any) -> GenerationRequest:
    fields: dict[str, Any] = {
        "identity": _identity(),
        "messages": (Message(role=Role.USER, content="Explain KV caching."),),
    }
    fields.update(overrides)
    return GenerationRequest(**fields)


def _last_body(router: respx.MockRouter) -> dict[str, Any]:
    for call in reversed(router.calls):
        if call.request.method == "POST":
            body: dict[str, Any] = json.loads(call.request.content)
            return body
    raise AssertionError("no request was sent")


def _launch_argv(launcher: FakeLauncher) -> tuple[str, ...]:
    return tuple(launcher.specs[-1].argv)


# --------------------------------------------------------------------------------- the flag


class TestTheFlagIsLoadBearing:
    def test_only_this_adapter_declares_it(
        self, make_provider: Callable[..., LlamaCppProvider]
    ) -> None:
        provider = make_provider()
        try:
            assert provider.capabilities().adapter_hot_swap is True
        finally:
            provider.close()
        assert FakeProvider(seed=1).capabilities().adapter_hot_swap is False

    def test_a_provider_that_declares_false_refuses_an_adapter(self) -> None:
        """ADR-0062 decision 5: callers branch on the flag, never on a provider's name."""
        with pytest.raises(CapabilityUnsupported) as raised:
            FakeProvider(seed=1).generate(
                GenerationRequest(
                    identity=ModelIdentity(ProviderKind.FAKE, "fake-model:8b-q8_0"),
                    prompt="hi",
                    adapter="factcheck",
                )
            )

        assert raised.value.details["capability"] == "adapter_hot_swap"

    def test_a_provider_that_declares_false_refuses_the_registry_calls(self) -> None:
        provider = FakeProvider(seed=1)

        with pytest.raises(CapabilityUnsupported):
            provider.list_adapters()
        with pytest.raises(CapabilityUnsupported):
            provider.register_adapters(())


# ------------------------------------------------------------------------------ registration


class TestLaunchTimeRegistration:
    def test_a_matching_digest_registers_and_is_selectable(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
        base_digest: str,
        adapter_files: dict[str, Path],
    ) -> None:
        provider = make_provider(adapters=[registration(base_artifact_digest=base_digest)])
        try:
            provider.generate(_request())
            argv = _launch_argv(launcher)
            assert "--lora" in argv
            assert str(adapter_files["factcheck"]) in argv
            assert "--lora-init-without-apply" in argv
            state = provider.list_adapters()[0]
            assert state.status is AdapterStatus.REGISTERED
            assert state.base_confidence is IdentityConfidence.DIGEST
            assert state.server_id == 0
        finally:
            provider.close()

    def test_a_manifest_with_no_base_digest_registers_as_name_only(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        """Admitted, and **flagged** — a reduced-confidence outcome, not a weaker check."""
        provider = make_provider(adapters=[registration()])
        try:
            result = provider.generate(_request())
            state = provider.list_adapters()[0]
        finally:
            provider.close()

        assert state.status is AdapterStatus.REGISTERED
        assert state.base_confidence is IdentityConfidence.NAME_ONLY
        assert result.adapter is None  # this request named none; the flag is on the state

    def test_a_declared_digest_that_does_not_match_is_refused_with_both_digests(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
        base_digest: str,
    ) -> None:
        """Fail closed: it never reaches the command line, and the reason names both sides."""
        provider = make_provider(adapters=[registration(base_artifact_digest=_STALE_DIGEST)])
        try:
            provider.generate(_request())
            argv = _launch_argv(launcher)
            state = provider.list_adapters()[0]
        finally:
            provider.close()

        assert "--lora" not in argv
        assert state.status is AdapterStatus.INCOMPATIBLE
        assert state.reason is not None
        assert _STALE_DIGEST in state.reason
        assert base_digest in state.reason

    def test_one_refusal_does_not_take_the_base_or_its_siblings_down(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        base_digest: str,
    ) -> None:
        """A stale manifest is one adapter's problem, never a denial of service on the base."""
        provider = make_provider(
            adapters=[
                registration("factcheck", base_artifact_digest=_STALE_DIGEST),
                registration("house-voice", base_artifact_digest=base_digest),
            ]
        )
        try:
            assert provider.generate(_request()).text
            statuses = {state.adapter.name: state.status for state in provider.list_adapters()}
        finally:
            provider.close()

        assert statuses == {
            "factcheck": AdapterStatus.INCOMPATIBLE,
            "house-voice": AdapterStatus.REGISTERED,
        }

    def test_a_renamed_base_still_registers_when_the_digest_matches(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        base_digest: str,
    ) -> None:
        """Identity is the hash and the path is a locator (ADR-0061 rule 5)."""
        provider = make_provider(
            adapters=[
                registration(
                    base_model_name="whatever-it-used-to-be-called",
                    base_artifact_digest=base_digest,
                )
            ]
        )
        try:
            provider.generate(_request())
            state = provider.list_adapters()[0]
        finally:
            provider.close()

        assert state.status is AdapterStatus.REGISTERED
        assert state.base_confidence is IdentityConfidence.DIGEST

    def test_an_adapter_for_another_base_is_awaiting_that_base(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
    ) -> None:
        provider = make_provider(adapters=[registration(base_model_name=_OTHER_MODEL)])
        try:
            provider.generate(_request())
            argv = _launch_argv(launcher)
            state = provider.list_adapters()[0]
        finally:
            provider.close()

        assert "--lora" not in argv
        assert state.status is AdapterStatus.AWAITING_BASE
        assert state.base_model_name == _OTHER_MODEL

    def test_with_no_adapters_the_command_line_is_untouched(
        self, make_provider: Callable[..., LlamaCppProvider], launcher: FakeLauncher
    ) -> None:
        """A-1's invariant on the launch path: no adapters, no adapter flags, nothing changed."""
        provider = make_provider()
        try:
            provider.generate(_request())
            argv = _launch_argv(launcher)
        finally:
            provider.close()

        assert not any(token.startswith("--lora") for token in argv)
        assert provider.list_adapters() == ()

    def test_the_ids_come_from_the_server_not_from_argv_order(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        server: respx.MockRouter,
        adapter_files: dict[str, Path],
    ) -> None:
        """A server that assigns different ids than argv order is believed, not corrected."""
        server.get(f"{_BASE_URL}/lora-adapters").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"id": 7, "path": str(adapter_files["house-voice"]), "scale": 1.0},
                    {"id": 3, "path": str(adapter_files["factcheck"]), "scale": 1.0},
                ],
            )
        )
        provider = make_provider(adapters=[registration("factcheck"), registration("house-voice")])
        try:
            provider.generate(_request(adapter="factcheck"))
            body = _last_body(server)
            ids = {state.adapter.name: state.server_id for state in provider.list_adapters()}
        finally:
            provider.close()

        assert ids == {"factcheck": 3, "house-voice": 7}
        assert body["lora"] == [
            {"id": 3, "scale": 1.0},
            {"id": 7, "scale": 0.0},
        ]

    def test_an_adapter_the_server_does_not_report_is_not_selectable(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        server: respx.MockRouter,
    ) -> None:
        server.get(f"{_BASE_URL}/lora-adapters").mock(return_value=httpx.Response(200, json=[]))
        provider = make_provider(adapters=[registration()])
        try:
            provider.generate(_request())
            state = provider.list_adapters()[0]
        finally:
            provider.close()

        assert state.status is AdapterStatus.PENDING_RESTART
        assert state.server_id is None

    def test_an_unreadable_lora_endpoint_leaves_the_base_usable(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        server: respx.MockRouter,
    ) -> None:
        """A server that answers /health and /props but not this still serves the bare base."""
        server.get(f"{_BASE_URL}/lora-adapters").mock(return_value=httpx.Response(500, text="no"))
        provider = make_provider(adapters=[registration()])
        try:
            assert provider.generate(_request()).text
            assert _last_body(server).get("lora") is None
        finally:
            provider.close()


# ---------------------------------------------------------------------------------- selection


class TestPerRequestSelection:
    def test_the_selected_adapter_is_the_only_one_enabled(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        server: respx.MockRouter,
    ) -> None:
        """ADR-0063: one adapter, at ``1.0``; every other registered adapter explicitly off."""
        provider = make_provider(adapters=[registration("factcheck"), registration("house-voice")])
        try:
            provider.generate(_request(adapter="house-voice"))
            body = _last_body(server)
        finally:
            provider.close()

        assert body["lora"] == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]

    def test_a_bare_request_to_an_adapter_server_disables_every_adapter(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        server: respx.MockRouter,
    ) -> None:
        """The correctness decision of this phase, stated as a test.

        llama-server treats an *absent* ``lora`` field as "restore the launch-time set", and takes
        that branch without clearing the slot's prompt cache. ``--lora`` registers at scale
        ``1.0``, so a bare-base request that sent nothing would run with **every** adapter applied,
        against whatever prefix ran last.
        """
        provider = make_provider(adapters=[registration("factcheck"), registration("house-voice")])
        try:
            provider.generate(_request())
            body = _last_body(server)
        finally:
            provider.close()

        assert body["lora"] == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]

    def test_the_result_names_the_whole_subject(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        base_digest: str,
        adapter_files: dict[str, Path],
    ) -> None:
        provider = make_provider(adapters=[registration(base_artifact_digest=base_digest)])
        try:
            result = provider.generate(_request(adapter="factcheck"))
        finally:
            provider.close()

        assert result.adapter is not None
        assert result.adapter.name == "factcheck"
        assert result.adapter.artifact_digest == _digest_of(adapter_files["factcheck"])
        assert result.adapter_base_confidence is IdentityConfidence.DIGEST

    def test_a_name_only_selection_carries_its_caveat_onto_the_result(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        provider = make_provider(adapters=[registration()])
        try:
            result = provider.generate(_request(adapter="factcheck"))
        finally:
            provider.close()

        assert result.adapter_base_confidence is IdentityConfidence.NAME_ONLY

    def test_a_streamed_result_names_the_subject_too(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        server: respx.MockRouter,
        load_llamacpp_fixture: Callable[[str], Any],
    ) -> None:
        provider = make_provider(adapters=[registration()])
        try:
            events: list[StreamEvent] = list(provider.stream(_request(adapter="factcheck")))
        finally:
            provider.close()

        completed = events[-1]
        assert isinstance(completed, StreamCompleted)
        assert completed.result.adapter is not None
        assert completed.result.adapter.name == "factcheck"

    def test_a_bare_request_reports_no_adapter(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        provider = make_provider(adapters=[registration()])
        try:
            result = provider.generate(_request())
        finally:
            provider.close()

        assert result.adapter is None
        assert result.adapter_base_confidence is None

    def test_an_unknown_adapter_names_the_registered_set(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        provider = make_provider(adapters=[registration()])
        try:
            with pytest.raises(AdapterNotFound) as raised:
                provider.generate(_request(adapter="no-such-adapter"))
        finally:
            provider.close()

        assert raised.value.details["adapter"] == "no-such-adapter"
        assert raised.value.details["registered"] == ["factcheck"]
        assert raised.value.details["reason"] == "unknown"

    def test_an_incompatible_adapter_is_refused_with_both_digests(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        base_digest: str,
    ) -> None:
        """Never a silent bare-base generation (ADR-0062 decision 4)."""
        provider = make_provider(adapters=[registration(base_artifact_digest=_STALE_DIGEST)])
        try:
            with pytest.raises(AdapterNotFound) as raised:
                provider.generate(_request(adapter="factcheck"))
        finally:
            provider.close()

        assert raised.value.details["reason"] == "incompatible_base"
        assert raised.value.details["declared_base_digest"] == _STALE_DIGEST
        assert raised.value.details["served_base_digest"] == base_digest

    def test_an_adapter_for_another_base_is_refused_on_this_one(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        provider = make_provider(adapters=[registration(base_model_name=_OTHER_MODEL)])
        try:
            with pytest.raises(AdapterNotFound) as raised:
                provider.generate(_request(adapter="factcheck"))
        finally:
            provider.close()

        assert raised.value.details["reason"] == "incompatible_base"

    @pytest.mark.parametrize("key", ["lora", "id_slot", "slot_id", "--lora"])
    def test_the_escape_hatch_cannot_smuggle_an_adapter_or_a_slot_pin(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        key: str,
    ) -> None:
        """``provider_options`` is a request escape hatch, not a second selection channel.

        An adapter chosen there would change the weights without changing the recorded subject;
        a slot pin would reach past the server's own cache-clearing rule.
        """
        provider = make_provider(adapters=[registration()])
        try:
            with pytest.raises(ProviderRejected) as raised:
                provider.generate(
                    _request(
                        runtime_profile=RuntimeProfile(provider_options={key: [{"id": 0}]}),
                    )
                )
        finally:
            provider.close()

        assert key in str(raised.value)


# ------------------------------------------------------------------- pending_restart and idle


class TestPendingRestartAndTheInFlightGuard:
    def test_an_adapter_registered_after_launch_is_pending(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
    ) -> None:
        provider = make_provider()
        try:
            provider.generate(_request())
            provider.register_adapters([registration()])
            state = provider.list_adapters()[0]
            assert state.status is AdapterStatus.PENDING_RESTART
            assert state.reason is not None
            assert "idle" in state.reason
            assert len(launcher.specs) == 1  # nothing was restarted by registering
        finally:
            provider.close()

    def test_it_folds_in_at_the_next_idle_request(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
        server: respx.MockRouter,
    ) -> None:
        """The next natural idle is the boundary before a request, and it costs one restart."""
        provider = make_provider()
        try:
            provider.generate(_request())
            provider.register_adapters([registration()])
            provider.generate(_request(adapter="factcheck"))

            assert len(launcher.specs) == 2
            assert launcher.processes[0].terminated
            assert "--lora" in _launch_argv(launcher)
            assert _last_body(server)["lora"] == [{"id": 0, "scale": 1.0}]
            assert provider.list_adapters()[0].status is AdapterStatus.REGISTERED
        finally:
            provider.close()

    def test_a_second_idle_request_does_not_restart_again(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
    ) -> None:
        """One restart per newly registered adapter is the floor, and also the ceiling."""
        provider = make_provider()
        try:
            provider.generate(_request())
            provider.register_adapters([registration()])
            provider.generate(_request())
            provider.generate(_request())
        finally:
            provider.close()

        assert len(launcher.specs) == 2

    def test_a_stream_in_flight_is_never_cut_into(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
        server: respx.MockRouter,
        load_llamacpp_fixture: Callable[[str], Any],
    ) -> None:
        """The race, forced by holding a partly drained stream open — not by timing.

        A second request arriving while the first is mid-stream must not restart the server the
        first is reading from. It gets a typed, temporary refusal instead, and the stream that was
        already running finishes normally.
        """
        provider = make_provider()
        try:
            provider.generate(_request())
            events = provider.stream(_request())
            first = next(iter(events))  # the stream is now in flight
            provider.register_adapters([registration()])  # the adapter arrives mid-work

            with pytest.raises(ProviderUnavailable) as raised:
                provider.generate(_request(adapter="factcheck"))

            assert raised.value.details["reason"] == (
                ProviderUnavailableReason.RESTART_PENDING.value
            )
            assert raised.value.details["restart_reason"] == "adapter_registration"
            assert raised.value.details["in_flight"] == 1
            assert len(launcher.specs) == 1
            assert not launcher.processes[0].terminated

            rest = list(events)
            assert isinstance(rest[-1], StreamCompleted)
            assert first is not None

            # And once nothing is in flight, the same request is served.
            provider.generate(_request(adapter="factcheck"))
            assert len(launcher.specs) == 2
        finally:
            provider.close()

    def test_a_profile_change_also_waits_for_idle(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        launcher: FakeLauncher,
        server: respx.MockRouter,
        load_llamacpp_fixture: Callable[[str], Any],
    ) -> None:
        """Phase 6 restarted immediately; a stream on another thread saw its connection drop."""
        provider = make_provider()
        try:
            events = provider.stream(_request())
            next(iter(events))

            with pytest.raises(ProviderUnavailable) as raised:
                provider.generate(_request(runtime_profile=RuntimeProfile(context_size=4096)))

            assert raised.value.details["restart_reason"] == "profile_change"
            assert not launcher.processes[0].terminated
            list(events)
        finally:
            provider.close()

    def test_a_stream_that_is_never_started_still_releases_its_claim(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
    ) -> None:
        """The claim is taken when the connection opens, not when the first event is read.

        A caller can take the iterator and never start it — the response is already open, so
        ``_walk``'s ``finally`` will never run to give the claim back. Without release on
        collection, one dropped iterator would make that server un-restartable for the life of
        the process, and every later rescan would silently never take effect.
        """
        provider = make_provider()
        try:
            provider.generate(_request())
            events = provider.stream(_request())  # opened, never iterated
            provider.register_adapters([registration()])

            with pytest.raises(ProviderUnavailable):
                provider.generate(_request(adapter="factcheck"))

            del events
            gc.collect()

            provider.generate(_request(adapter="factcheck"))
            assert len(launcher.specs) == 2
        finally:
            provider.close()

    def test_unloading_forgets_every_adapter_state(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        """A registration on a server that is gone is awaiting a base, not registered on one."""
        provider = make_provider(adapters=[registration()])
        try:
            provider.generate(_request())
            assert provider.list_adapters()[0].status is AdapterStatus.REGISTERED

            provider.unload(_identity())
            assert provider.list_adapters()[0].status is AdapterStatus.AWAITING_BASE
        finally:
            provider.close()

    def test_registering_a_name_again_replaces_it(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        adapter_files: dict[str, Path],
    ) -> None:
        """New bytes under a familiar name are a new subject, and the old one is retired."""
        provider = make_provider(adapters=[registration()])
        try:
            replacement = registration(
                "factcheck", artifact_sha256=_digest_of(adapter_files["house-voice"])
            )
            provider.register_adapters([replacement])
            states = provider.list_adapters()
        finally:
            provider.close()

        assert len(states) == 1
        assert states[0].adapter.artifact_sha256 == _digest_of(adapter_files["house-voice"])

    def test_two_requests_in_flight_both_have_to_finish(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
        launcher: FakeLauncher,
    ) -> None:
        """The count is a count, not a flag: one of two finishing does not make a server idle."""
        provider = make_provider()
        try:
            provider.generate(_request())
            first = provider.stream(_request())
            second = provider.stream(_request())
            next(iter(first))
            next(iter(second))
            provider.register_adapters([registration()])

            list(first)  # one finishes; the other is still reading
            with pytest.raises(ProviderUnavailable) as raised:
                provider.generate(_request(adapter="factcheck"))
            assert raised.value.details["in_flight"] == 1

            list(second)
            provider.generate(_request(adapter="factcheck"))
            assert len(launcher.specs) == 2
        finally:
            provider.close()

    def test_an_incompatible_adapter_registered_after_launch_is_reported_as_incompatible(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        """A stale manifest handed over mid-session is refused on sight, not at the next restart."""
        provider = make_provider()
        try:
            provider.generate(_request())
            provider.register_adapters([registration(base_artifact_digest=_STALE_DIGEST)])
            state = provider.list_adapters()[0]
        finally:
            provider.close()

        assert state.status is AdapterStatus.INCOMPATIBLE
        assert state.reason is not None
        assert _STALE_DIGEST in state.reason

    def test_an_adapter_naming_another_base_by_both_handles_is_simply_not_this_ones(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        registration: Callable[..., AdapterRegistration],
    ) -> None:
        """Neither handle matches, so it is another base's adapter — silence, not a refusal."""
        provider = make_provider(
            adapters=[
                registration(base_model_name=_OTHER_MODEL, base_artifact_digest=_STALE_DIGEST)
            ]
        )
        try:
            provider.generate(_request())
            state = provider.list_adapters()[0]
        finally:
            provider.close()

        assert state.status is AdapterStatus.AWAITING_BASE
        assert state.reason is None
