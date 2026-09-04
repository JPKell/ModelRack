"""I17 — a prompt prefix computed under adapter A is never reused for adapter B.

**What this can and cannot assert.** The prefix cache in question is *llama-server's own*: a slot
holds the tokens it has already processed, and the server decides which slot answers a request.
ModelRack keeps no such cache — its :class:`~modelrack.cache.MetadataCache` holds parsed GGUF
headers and its ``DigestStore`` holds file hashes, neither of which is generation state — so there
is no ModelRack cache a test could catch reusing a prefix. Asserting "the server did not reuse the
cache" needs the server, and that is the semantic canary in ``tests/live/``.

What runs here, in the default gate with no binary, is the half that is **this package's to get
right**: every request ModelRack sends states its whole adapter configuration, and ModelRack never
sends anything that could bind a request to a prefix computed under a different adapter. Those two
properties are what make the server's own rule reachable:

* llama-server clears a slot's prompt cache when a task's adapter set differs from the slot's
  (``lora_should_clear_cache``, ``server-context.cpp``) — **but only on the branch where the
  request carries a ``lora`` field.** A request with no ``lora`` takes the other branch:
  ``slot.lora = params_base.lora_adapters``, with no cache check at all. Since ``--lora``
  registers at scale ``1.0`` and ``--lora-init-without-apply`` governs only whether the set is
  applied at *init*, a bare-base request that sent no ``lora`` field would run with **every**
  registered adapter applied, against whatever prefix ran last. Stating the configuration on every
  request is what closes that.
* Slot choice is the server's, because that is what lets it compare the slot's adapters with the
  task's. A caller-pinned ``id_slot`` reaches past the rule, so this adapter never sends one and
  refuses a caller that tries.

**No cross-adapter batching**, recorded rather than assumed (ADR-0062 decision 4): llama-server's
``server_slot::can_batch_with`` includes ``are_lora_equal``, so two slots whose adapter sets differ
are never in one decode batch. That is the server's guarantee, not this package's, and it is
written down here so it stays checked if this suite's single-user concurrency ever changes.

**The property is proved by making it fail.** :func:`assert_adapter_isolation` is run over the real
request path and then over three injected defects — the selection dropped, the previous request's
selection left in place, and a slot pin added. A conformance property that cannot fail proves
nothing, and each of those three is a real defect this phase could have shipped.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from baseaicore import DataClassification, ModelIdentity, ProviderKind, RuntimeProfile

from conftest import FakeLauncher, FakeMonotonic, FakeProcessTable, FakeSleep
from modelrack import (
    AdapterRegistration,
    GenerationRequest,
    Message,
    ResponseFormat,
    ResponseFormatKind,
    Role,
    SamplingParameters,
    ToolDefinition,
)
from modelrack.providers._llamacpp_wire import (
    SLOT_PINNING_KEYS,
    build_chat_body,
    build_completion_body,
)
from modelrack.providers.llamacpp import LlamaCppProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from datetime import datetime

pytestmark = pytest.mark.contract

_PORT = 18280
_MODEL = "qwen3.5-9b-q8_0"
_ADAPTERS = ("factcheck", "house-voice", "summarize")
_GOLDEN_MODEL = "qwen3.5-9b-q8_0"


# ------------------------------------------------------------------------------ the property


@dataclass(frozen=True, slots=True)
class Exchange:
    """One recorded request: what the caller asked for, and what actually went on the wire."""

    adapter: str | None
    body: Mapping[str, Any]


def assert_adapter_isolation(
    exchanges: Sequence[Exchange], *, registered: Mapping[str, int]
) -> None:
    """Assert I17's structural half over a sequence of recorded requests.

    Args:
        exchanges: Every request that was sent, in order, each paired with the adapter its caller
            named.
        registered: Adapter name to the server id it was registered under. Empty when the server
            has no adapters, which is the case that must look exactly like Phase 6.

    Raises:
        AssertionError: Naming the first exchange that breaks the property, and which clause.
    """
    for position, exchange in enumerate(exchanges):
        where = f"exchange {position} (adapter={exchange.adapter!r})"
        lora = exchange.body.get("lora")

        # 1. The key is present exactly when the server has adapters — so a server with none
        #    produces a body byte-for-byte identical to Phase 6's.
        if not registered:
            assert lora is None, f"{where}: a server with no adapters must send no `lora` key"
            _assert_no_slot_pin(exchange.body, where)
            continue
        assert isinstance(lora, list), f"{where}: an adapter-registered server needs a `lora` list"

        # 2. The configuration is **complete**: every registered adapter appears exactly once, so
        #    nothing is left to the server's launch-time default.
        ids = [entry["id"] for entry in lora]
        assert sorted(ids) == sorted(registered.values()), (
            f"{where}: `lora` names {sorted(ids)}, but the server has {sorted(registered.values())}"
            " registered. A partial list leaves the rest to the launch scales."
        )
        assert len(ids) == len(set(ids)), f"{where}: an adapter appears twice in `lora`"

        # 3. One adapter, at 1.0, and it is the one the caller named (ADR-0063).
        enabled = [entry["id"] for entry in lora if entry["scale"] != 0.0]
        if exchange.adapter is None:
            assert enabled == [], f"{where}: no adapter was named, but {enabled} are enabled"
        else:
            assert enabled == [registered[exchange.adapter]], (
                f"{where}: {exchange.adapter!r} is id {registered[exchange.adapter]}, "
                f"but {enabled} are enabled"
            )
        for entry in lora:
            assert entry["scale"] in (0.0, 1.0), (
                f"{where}: scale {entry['scale']} — the scale is fixed at 1.0 and is not a "
                "request parameter (ADR-0063 rule 2)"
            )

        # 4. Nothing binds this request to a slot, and so to a slot's cache.
        _assert_no_slot_pin(exchange.body, where)


def _assert_no_slot_pin(body: Mapping[str, Any], where: str) -> None:
    for key in SLOT_PINNING_KEYS:
        assert key not in body, (
            f"{where}: `{key}` pins this request to a slot, reaching past the server's own "
            "adapter-aware cache clearing"
        )


# ------------------------------------------------------------------------------- scaffolding


def _digest_of(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def models(tmp_path: Path, gguf_writer: Callable[..., Path]) -> Path:
    directory = tmp_path / "models"
    directory.mkdir()
    gguf_writer(
        directory / f"{_MODEL}.gguf",
        metadata={"general.architecture": "qwen35", "qwen35.block_count": 32},
        payload=b"\x11" * 64,
    )
    return directory


@pytest.fixture
def adapter_registrations(tmp_path: Path, models: Path) -> list[AdapterRegistration]:
    directory = tmp_path / "adapters"
    directory.mkdir()
    base_digest = _digest_of(models / f"{_MODEL}.gguf")
    registrations: list[AdapterRegistration] = []
    for index, name in enumerate(_ADAPTERS):
        path = directory / f"{name}.gguf"
        path.write_bytes(bytes([0xA0 + index]) * 96)
        registrations.append(
            AdapterRegistration(
                name=name,
                artifact_path=path,
                artifact_sha256=_digest_of(path),
                base_model_name=_MODEL,
                base_artifact_digest=base_digest,
                data_classification=DataClassification.CONFIDENTIAL,
            )
        )
    return registrations


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
def transport(
    launcher: FakeLauncher, load_llamacpp_fixture: Callable[[str], Any]
) -> Iterator[respx.MockRouter]:
    """A recorded server that reports the adapters it was launched with, per port."""

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

    def completion(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                content=load_llamacpp_fixture("completion_stream.sse").encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json=load_llamacpp_fixture("completion.json"))

    with respx.mock(assert_all_called=False) as router:
        for offset in range(2):
            base = f"http://127.0.0.1:{_PORT + offset}"
            router.get(f"{base}/health").mock(
                return_value=httpx.Response(200, json=load_llamacpp_fixture("health_ok.json"))
            )
            router.get(f"{base}/props").mock(
                return_value=httpx.Response(200, json=load_llamacpp_fixture("props.json"))
            )
            router.get(f"{base}/lora-adapters").mock(side_effect=lora)
            router.post(f"{base}/v1/chat/completions").mock(side_effect=chat)
            router.post(f"{base}/completion").mock(side_effect=completion)
        yield router


@pytest.fixture
def make_provider(
    models: Path, tmp_path: Path, launcher: FakeLauncher, frozen_clock: Callable[[], datetime]
) -> Callable[..., LlamaCppProvider]:
    def _make(adapters: Sequence[AdapterRegistration] = ()) -> LlamaCppProvider:
        return LlamaCppProvider(
            models,
            state_dir=tmp_path / "state",
            adapters=adapters,
            server_path=sys.executable,
            port_range=(_PORT, _PORT + 1),
            launcher=launcher,
            process_table=FakeProcessTable(),
            port_is_free=lambda _port: True,
            sleep=FakeSleep(),
            monotonic=FakeMonotonic(),
            clock=frozen_clock,
        )

    return _make


def _request(adapter: str | None, *, chat: bool) -> GenerationRequest:
    identity = ModelIdentity(ProviderKind.LLAMACPP, _MODEL)
    if chat:
        return GenerationRequest(
            identity=identity,
            messages=(Message(role=Role.USER, content="Explain KV caching."),),
            adapter=adapter,
        )
    return GenerationRequest(identity=identity, prompt="Explain KV caching.", adapter=adapter)


def _drive(
    provider: LlamaCppProvider,
    transport: respx.MockRouter,
    plan: Sequence[tuple[str | None, bool, bool]],
) -> list[Exchange]:
    """Run a plan of ``(adapter, chat, streaming)`` requests and record what went on the wire."""
    exchanges: list[Exchange] = []
    before = len(transport.calls)
    for adapter, chat, streaming in plan:
        request = _request(adapter, chat=chat)
        if streaming:
            list(provider.stream(request))
        else:
            provider.generate(request)
        sent = [
            call.request for call in list(transport.calls)[before:] if call.request.method == "POST"
        ]
        assert sent, "the request never reached the server"
        exchanges.append(Exchange(adapter=adapter, body=json.loads(sent[-1].content)))
        before = len(transport.calls)
    return exchanges


# ------------------------------------------------------------------------------- the tests


@pytest.mark.usefixtures("transport")
class TestTheRequestPathIsolatesAdapters:
    def test_the_property_holds_over_an_alternating_sequence(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        adapter_registrations: list[AdapterRegistration],
        transport: respx.MockRouter,
    ) -> None:
        """Twenty randomized requests over both endpoints and both streaming modes.

        The sequence is drawn from ``random``, which ``pytest-randomly`` seeds and prints, so a
        failure that appears only under some orders is reproducible from the reported seed — and
        an order-dependent failure here is exactly the defect shape this test exists for.
        """
        provider = make_provider(adapter_registrations)
        try:
            # noqa: S311 — a test fixture's request order, not a cryptographic draw; the
            # generator is `pytest-randomly`'s, whose seed is printed on failure.
            plan = [
                (
                    random.choice([*_ADAPTERS, None]),  # noqa: S311 — see above
                    random.choice([True, False]),  # noqa: S311 — see above
                    random.choice([True, False]),  # noqa: S311 — see above
                )
                for _ in range(20)
            ]
            exchanges = _drive(provider, transport, plan)
            registered = {
                state.adapter.name: state.server_id
                for state in provider.list_adapters()
                if state.server_id is not None
            }
        finally:
            provider.close()

        assert len(registered) == len(_ADAPTERS)
        assert_adapter_isolation(exchanges, registered=registered)

    def test_every_request_reaches_the_server(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        adapter_registrations: list[AdapterRegistration],
        transport: respx.MockRouter,
    ) -> None:
        """ModelRack caches no generation, so an adapter switch can never be answered locally."""
        provider = make_provider(adapter_registrations)
        try:
            plan = [("factcheck", True, False), ("house-voice", True, False), (None, True, False)]
            exchanges = _drive(provider, transport, plan)
        finally:
            provider.close()

        assert len(exchanges) == len(plan)
        assert len({id(exchange.body) for exchange in exchanges}) == len(plan)

    def test_a_server_with_no_adapters_sends_the_body_it_always_did(
        self,
        make_provider: Callable[..., LlamaCppProvider],
        adapter_registrations: list[AdapterRegistration],
        transport: respx.MockRouter,
    ) -> None:
        """A-1's invariant on the request path, asserted against the adapter-registered body.

        The two bodies must differ **only** by the ``lora`` key: if anything else moved, a
        deployment that has never heard of adapters would be sending a different request than it
        was before this phase.
        """
        bare = make_provider()
        try:
            without = _drive(bare, transport, [(None, True, False), (None, False, False)])
        finally:
            bare.close()
        with_adapters = make_provider(adapter_registrations)
        try:
            withs = _drive(with_adapters, transport, [(None, True, False), (None, False, False)])
        finally:
            with_adapters.close()

        assert_adapter_isolation(without, registered={})
        for plain, decorated in zip(without, withs, strict=True):
            assert "lora" not in plain.body
            assert {k: v for k, v in decorated.body.items() if k != "lora"} == dict(plain.body)


class TestThePropertyCanFail:
    """The three defects this phase could have shipped, each caught by the checker.

    Without these, ``assert_adapter_isolation`` would be a test that passes because nothing ever
    exercises it — the worst kind of correctness test, since its silence reads as proof.
    """

    _REGISTERED = {"factcheck": 0, "house-voice": 1}

    def _sound(self) -> list[Exchange]:
        return [
            Exchange(
                adapter="factcheck",
                body={"lora": [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]},
            ),
            Exchange(
                adapter="house-voice",
                body={"lora": [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]},
            ),
            Exchange(
                adapter=None,
                body={"lora": [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]},
            ),
        ]

    def test_the_sound_sequence_passes(self) -> None:
        """The control: the checker is not simply always failing."""
        assert_adapter_isolation(self._sound(), registered=self._REGISTERED)

    def test_a_dropped_selection_is_caught(self) -> None:
        """The silent-wrong-subject defect: the caller's adapter never reaches the server."""
        exchanges = self._sound()
        exchanges[1] = Exchange(adapter="house-voice", body={})

        with pytest.raises(AssertionError, match="needs a `lora` list"):
            assert_adapter_isolation(exchanges, registered=self._REGISTERED)

    def test_a_stale_selection_is_caught(self) -> None:
        """The cross-contamination defect: request B runs under request A's adapter."""
        exchanges = self._sound()
        exchanges[1] = Exchange(
            adapter="house-voice",
            body={"lora": [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]},
        )

        with pytest.raises(AssertionError, match="is id 1"):
            assert_adapter_isolation(exchanges, registered=self._REGISTERED)

    def test_a_partial_configuration_is_caught(self) -> None:
        """The defect that leaves the rest to llama-server's launch scales — every adapter on."""
        exchanges = self._sound()
        exchanges[0] = Exchange(adapter="factcheck", body={"lora": [{"id": 0, "scale": 1.0}]})

        with pytest.raises(AssertionError, match="A partial list"):
            assert_adapter_isolation(exchanges, registered=self._REGISTERED)

    def test_a_slot_pin_is_caught(self) -> None:
        """The prefix-reuse lever: a pinned slot reaches past the server's own cache clearing."""
        exchanges = self._sound()
        exchanges[1] = Exchange(
            adapter="house-voice",
            body={"lora": [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}], "id_slot": 0},
        )

        with pytest.raises(AssertionError, match="pins this request to a slot"):
            assert_adapter_isolation(exchanges, registered=self._REGISTERED)

    def test_a_per_request_scale_is_caught(self) -> None:
        """ADR-0063 rule 2: the scale is fixed at 1.0 and is not a request parameter."""
        exchanges = self._sound()
        exchanges[0] = Exchange(
            adapter="factcheck",
            body={"lora": [{"id": 0, "scale": 0.7}, {"id": 1, "scale": 0.0}]},
        )

        with pytest.raises(AssertionError, match="the scale is fixed"):
            assert_adapter_isolation(exchanges, registered=self._REGISTERED)

    def test_a_bare_server_that_sends_lora_is_caught(self) -> None:
        """The other direction of A-1: a deployment with no adapters must send nothing new."""
        exchanges = [Exchange(adapter=None, body={"lora": [{"id": 0, "scale": 0.0}]})]

        with pytest.raises(AssertionError, match="must send no `lora` key"):
            assert_adapter_isolation(exchanges, registered={})


class TestTheAdapterFreeBodyIsByteForByteWhatItWas:
    """A-1's invariant, asserted against a **golden captured from the Phase 6 code itself**.

    ``tests/fixtures/providers/llamacpp/phase6_request_bodies.json`` was produced by running
    ``build_chat_body`` and ``build_completion_body`` as they stood at the commit before adapters
    existed, over four requests chosen to touch every branch either function has: bare and fully
    loaded, chat and completion, streamed and not, with tools, a JSON mode, a JSON schema, every
    sampling field and both kinds of ``provider_options`` entry.

    Field assertions elsewhere check that the right values are present. This checks that **nothing
    else moved** — no key added, none dropped, none renamed, no value nudged — which is the only
    form in which "byte-for-byte what it was" is a claim rather than a hope. A deployment that has
    never heard of adapters sends exactly the bytes it sent before this phase.
    """

    @staticmethod
    def _load() -> dict[str, dict[str, Any]]:
        path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "providers"
            / "llamacpp"
            / "phase6_request_bodies.json"
        )
        loaded: dict[str, dict[str, Any]] = json.loads(path.read_text())
        return loaded

    @staticmethod
    def _rebuild(name: str) -> dict[str, Any]:
        identity = ModelIdentity(ProviderKind.LLAMACPP, _GOLDEN_MODEL)
        tool = ToolDefinition(
            name="get_weather",
            description="Weather.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        sampling = SamplingParameters(
            temperature=0.1,
            top_p=0.9,
            top_k=20,
            seed=3,
            max_output_tokens=64,
            stop=("STOP",),
            repeat_penalty=1.1,
        )
        profile = RuntimeProfile(
            context_size=4096, gpu_layers=99, provider_options={"min_p": 0.05, "--parallel": 1}
        )
        message = (Message(role=Role.USER, content="Explain KV caching."),)
        if name == "chat_plain":
            return build_chat_body(
                GenerationRequest(identity=identity, messages=message),
                alias=_GOLDEN_MODEL,
                stream=False,
            )
        if name == "chat_full":
            return build_chat_body(
                GenerationRequest(
                    identity=identity,
                    messages=message,
                    sampling=sampling,
                    tools=(tool,),
                    runtime_profile=profile,
                    response_format=ResponseFormat(kind=ResponseFormatKind.JSON),
                ),
                alias=_GOLDEN_MODEL,
                stream=True,
            )
        if name == "completion_plain":
            return build_completion_body(
                GenerationRequest(identity=identity, prompt="Explain KV caching."), stream=False
            )
        return build_completion_body(
            GenerationRequest(
                identity=identity,
                prompt="Explain KV caching.",
                sampling=sampling,
                runtime_profile=profile,
                response_format=ResponseFormat(
                    kind=ResponseFormatKind.JSON_SCHEMA,
                    schema={"type": "object", "properties": {"answer": {"type": "string"}}},
                ),
            ),
            stream=True,
        )

    @pytest.mark.parametrize(
        "name", ["chat_plain", "chat_full", "completion_plain", "completion_full"]
    )
    def test_the_body_is_identical_to_the_phase_6_golden(self, name: str) -> None:
        golden = self._load()[name]

        rebuilt = self._rebuild(name)

        assert json.dumps(rebuilt, sort_keys=True) == json.dumps(golden, sort_keys=True), (
            f"{name}: the adapter-free request body is no longer what Phase 6 sent. A deployment "
            "with no adapters must be byte-for-byte unaffected by the adapter axis (ADR-0058)."
        )

    def test_the_golden_covers_both_endpoints_and_both_streaming_modes(self) -> None:
        """A golden that only covered the bare cases would pass while the loaded ones drifted."""
        golden = self._load()

        assert {"chat_plain", "chat_full", "completion_plain", "completion_full"} == set(golden)
        assert golden["chat_full"]["stream"] is True
        assert golden["completion_full"]["stream"] is True
        assert all("lora" not in body for body in golden.values())
