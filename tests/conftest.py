"""Shared fixtures: a socket guard, a canonical identity, a deterministic clock, and recorded
Ollama fixtures.

The socket guard is the one that matters most in this repository. ModelRack is an HTTP client, so
"the default suite passes with no Ollama running" ([spec §18](../docs/packages/modelrack/spec.md),
acceptance criterion 3) is not something that stays true by good intentions: one adapter test that
quietly reaches ``127.0.0.1:11434`` passes on the maintainer's machine and fails in CI, or worse,
passes in CI against nothing and asserts on a fabricated failure path.

The guard therefore fails any test that opens a socket, and exempts only ``tests/live/`` — the
directory whose entire purpose is to talk to a real provider, and which is deselected by default
through the ``live`` marker (``addopts = "-m 'not live and not performance'"``). ``respx`` never
trips it: it replaces ``httpx``'s transport before a socket is ever asked for, which is what makes
a recorded-fixture adapter test possible at all under this guard.

``load_ollama_fixture`` lives here — not in ``tests/unit/`` or ``tests/contract/`` alone — because
both the Ollama adapter's own unit tests and the shared provider conformance suite need the same
recorded payloads, and this is the one conftest both directories inherit from
(``tests/`` has no ``__init__.py`` anywhere, so a plain cross-directory import would be fragile;
a root-level fixture is not). ``write_gguf`` lives here for the same reason: the GGUF reader's
unit tests, the llama.cpp adapter's tests and the conformance suite all need a model file on
disk, and none of them may depend on the multi-gigabyte real ones.
"""

from __future__ import annotations

import json
import socket
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest
from baseaicore import ModelIdentity, ProviderKind

from modelrack.providers._llamacpp_process import LaunchSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from baseaicore import Clock

_OLLAMA_FIXTURE_DIR: Final[Path] = Path(__file__).parent / "fixtures" / "providers" / "ollama"
_OPENAI_COMPATIBLE_FIXTURE_DIR: Final[Path] = (
    Path(__file__).parent / "fixtures" / "providers" / "openai_compatible"
)
_LLAMACPP_FIXTURE_DIR: Final[Path] = Path(__file__).parent / "fixtures" / "providers" / "llamacpp"

# GGUF value-type ids, from llama.cpp's gguf.h, for the writer below.
_GGUF_UINT32: Final[int] = 4
_GGUF_INT32: Final[int] = 5
_GGUF_FLOAT32: Final[int] = 6
_GGUF_BOOL: Final[int] = 7
_GGUF_STRING: Final[int] = 8
_GGUF_ARRAY: Final[int] = 9
_GGUF_UINT64: Final[int] = 10
_GGUF_INT64: Final[int] = 11
_GGUF_FLOAT64: Final[int] = 12

# A fixed instant with a non-zero millisecond component, so a truncation bug shows up as a changed
# value rather than hiding behind a round number.
_FIXED_INSTANT: Final[datetime] = datetime(2026, 8, 22, 14, 3, 11, 250_000, tzinfo=UTC)


@pytest.fixture
def fixed_now() -> datetime:
    """Return the instant every test in this suite treats as "now"."""
    return _FIXED_INSTANT


@pytest.fixture
def frozen_clock(fixed_now: datetime) -> Clock:
    """Return a :data:`baseaicore.Clock` that always reports :func:`fixed_now`."""
    return lambda: fixed_now


@pytest.fixture
def identity() -> ModelIdentity:
    """Return the model identity used by requests and results built in tests.

    Name-only on purpose: it is the harder of the two cases to get right — a tag can be repointed,
    so everything built on it carries a permanent caveat — and a test suite whose every fixture
    pinned a digest would never exercise it.
    """
    return ModelIdentity(ProviderKind.OLLAMA, "qwen3.5:9b-q8_0")


@pytest.fixture
def digest_identity() -> ModelIdentity:
    """Return a model identity that pins exact weights through a normalized digest."""
    return ModelIdentity(
        ProviderKind.OLLAMA,
        "qwen3.5:9b-q8_0",
        artifact_digest="sha256:" + "1f3a9c4e2b70" + "0" * 52,
    )


@pytest.fixture
def load_ollama_fixture() -> Callable[[str], Any]:
    """Return a loader for one recorded Ollama response, by filename.

    Every payload under ``tests/fixtures/providers/ollama/`` (manifest included — spec §19 wants
    the provider version recorded beside the fixtures it describes) is JSON except
    ``chat_stream.ndjson``, which is one line of JSON per streamed chunk and is returned as raw
    bytes so a test controls its own chunking rather than inheriting whatever this loader would
    have chosen.

    A fixture-returning-a-function, the same shape as :func:`frozen_clock` above: pytest fixtures
    take no arguments, so a parameterized lookup is injected as a callable instead of a fixture
    per fixture file.
    """

    def _load(name: str) -> Any:
        path = _OLLAMA_FIXTURE_DIR / name
        if path.suffix == ".ndjson":
            return path.read_bytes()
        return json.loads(path.read_text())

    return _load


@pytest.fixture
def load_openai_compatible_fixture() -> Callable[[str], Any]:
    """Return a loader for one recorded OpenAI-compatible response, by filename.

    Mirrors :func:`load_ollama_fixture` exactly, for the sibling directory Phase 4 adds. Every
    payload under ``tests/fixtures/providers/openai_compatible/`` is JSON except the ``.sse``
    files, which are raw server-sent-event text and are returned as-is so a test controls its own
    chunking, the same reason ``chat_stream.ndjson`` is returned as bytes rather than parsed.
    """

    def _load(name: str) -> Any:
        path = _OPENAI_COMPATIBLE_FIXTURE_DIR / name
        if path.suffix == ".sse":
            return path.read_text()
        return json.loads(path.read_text())

    return _load


@pytest.fixture
def load_llamacpp_fixture() -> Callable[[str], Any]:
    """Return a loader for one recorded llama-server response, by filename.

    Mirrors :func:`load_openai_compatible_fixture` for the directory Phase 6 adds. JSON files are
    parsed; ``.sse`` files are raw server-sent-event text, returned as-is so a test controls its
    own chunking. Every file is a *representative* payload for the build the directory's
    ``manifest.json`` names — there is no llama.cpp on the machine this suite was written on.
    """

    def _load(name: str) -> Any:
        path = _LLAMACPP_FIXTURE_DIR / name
        if path.suffix == ".sse":
            return path.read_text()
        return json.loads(path.read_text())

    return _load


def _gguf_string(value: str) -> bytes:
    """Encode one GGUF string: a u64 byte length, then UTF-8."""
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value: Any) -> tuple[int, bytes]:
    """Encode one Python value as ``(type_id, bytes)`` in GGUF's little-endian layout.

    ``bool`` before ``int`` because ``bool`` is an ``int``; small non-negative ints are u32 (the
    type real files use for counts), larger ones u64, negatives i32/i64; floats are f32 unless
    given as a one-element ``("f64", value)`` tuple; lists are arrays typed by their first element
    (empty arrays are string arrays).
    """
    if isinstance(value, bool):
        return _GGUF_BOOL, struct.pack("<?", value)
    if isinstance(value, int):
        if 0 <= value < 1 << 32:
            return _GGUF_UINT32, struct.pack("<I", value)
        if value >= 0:
            return _GGUF_UINT64, struct.pack("<Q", value)
        if value >= -(1 << 31):
            return _GGUF_INT32, struct.pack("<i", value)
        return _GGUF_INT64, struct.pack("<q", value)
    if isinstance(value, float):
        return _GGUF_FLOAT32, struct.pack("<f", value)
    if isinstance(value, str):
        return _GGUF_STRING, _gguf_string(value)
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "f64":
        return _GGUF_FLOAT64, struct.pack("<d", value[1])
    if isinstance(value, list):
        if not value:
            return _GGUF_ARRAY, struct.pack("<IQ", _GGUF_STRING, 0)
        element_type, _ = _gguf_value(value[0])
        body = b"".join(_gguf_value(item)[1] for item in value)
        return _GGUF_ARRAY, struct.pack("<IQ", element_type, len(value)) + body
    raise TypeError(f"cannot encode {value!r} as a GGUF value")


def write_gguf(
    path: Path,
    *,
    metadata: dict[str, Any],
    tensors: tuple[tuple[str, tuple[int, ...]], ...] = (),
    version: int = 3,
    payload: bytes = b"",
) -> Path:
    """Write a small but structurally complete GGUF file for tests.

    The header is exactly what llama.cpp's ``gguf.h`` describes — magic, version, tensor count,
    key/value count, the pairs, then one info record per tensor (name, dimensions, ggml type
    ``0`` = F32, offset) — followed by ``payload``, which stands in for the weights and is what
    makes two files with the same header hash differently.
    """
    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, value in metadata.items():
        type_id, encoded = _gguf_value(value)
        out += _gguf_string(key) + struct.pack("<I", type_id) + encoded
    offset = 0
    for name, dims in tensors:
        out += _gguf_string(name) + struct.pack("<I", len(dims))
        for dim in dims:
            out += struct.pack("<Q", dim)
        out += struct.pack("<IQ", 0, offset)
        elements = 1
        for dim in dims:
            elements *= dim
        offset += elements * 4
    out += payload
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def gguf_writer() -> Callable[..., Path]:
    """Return :func:`write_gguf`, for a test that needs a GGUF file on disk."""
    return write_gguf


class FakeServerProcess:
    """A scripted :class:`~modelrack.providers._llamacpp_process.ServerProcess`.

    Alive until told otherwise: ``exit_after_polls`` makes it exit on its own after that many
    ``poll()`` calls (a server that crashes during or after startup); ``survives_terminate``
    makes ``SIGTERM`` ineffective so only ``kill()`` ends it (a server that must be
    kill-treed). ``terminated`` and ``killed`` record what the supervisor did.
    """

    def __init__(
        self,
        pid: int,
        *,
        exit_after_polls: int | None = None,
        exit_code: int = 1,
        survives_terminate: bool = False,
    ) -> None:
        self._pid = pid
        self._exit_after_polls = exit_after_polls
        self._exit_code = exit_code
        self._survives_terminate = survives_terminate
        self._code: int | None = None
        self.polls = 0
        self.terminated = False
        self.killed = False

    @property
    def pid(self) -> int:
        return self._pid

    def poll(self) -> int | None:
        self.polls += 1
        if (
            self._code is None
            and self._exit_after_polls is not None
            and self.polls >= self._exit_after_polls
        ):
            self._code = self._exit_code
        return self._code

    def wait(self, timeout_seconds: float) -> int | None:
        return self._code

    def terminate(self) -> None:
        self.terminated = True
        if self._code is None and not self._survives_terminate:
            self._code = -15

    def kill(self) -> None:
        self.killed = True
        if self._code is None:
            self._code = -9

    def crash(self, exit_code: int) -> None:
        """Make the process exit on its own, as a server that died between calls would."""
        if self._code is None:
            self._code = exit_code


class FakeLauncher:
    """A scripted :data:`~modelrack.providers._llamacpp_process.ProcessLauncher`.

    Records every :class:`LaunchSpec` it was given and every process it returned, writes
    ``stderr_text`` to the spec's stderr path (so the supervisor's tail-reading is real), and
    plays a queue of behaviours — a process factory or an exception to raise — falling back to a
    healthy process that lives until terminated.
    """

    def __init__(self, *, stderr_text: str = "llama-server: listening\n") -> None:
        self.specs: list[LaunchSpec] = []
        self.processes: list[FakeServerProcess] = []
        self.stderr_text = stderr_text
        self._queue: list[Callable[[LaunchSpec], FakeServerProcess] | Exception] = []
        self._next_pid = 40_000

    def plan(self, *behaviours: Callable[[LaunchSpec], FakeServerProcess] | Exception) -> None:
        """Queue behaviours for the next spawns, consumed in order."""
        self._queue.extend(behaviours)

    def next_pid(self) -> int:
        """Return a fresh pid for a scripted process."""
        self._next_pid += 1
        return self._next_pid

    def __call__(self, spec: LaunchSpec) -> FakeServerProcess:
        self.specs.append(spec)
        spec.stderr_path.write_text(self.stderr_text)
        if self._queue:
            behaviour = self._queue.pop(0)
            if isinstance(behaviour, Exception):
                raise behaviour
            process = behaviour(spec)
        else:
            process = FakeServerProcess(self.next_pid())
        self.processes.append(process)
        return process

    @property
    def live(self) -> list[FakeServerProcess]:
        """Every process this launcher started that has not exited."""
        return [process for process in self.processes if process.poll() is None]


class FakeProcessTable:
    """A scripted :class:`~modelrack.providers._llamacpp_process.ProcessTable`.

    ``alive`` is the set of pids that exist; ``command_lines`` what ``/proc`` would say for
    them (absent means "cannot say"); ``ignores_term`` the pids that survive ``SIGTERM`` and die
    only on ``SIGKILL``. Every signal is recorded.
    """

    def __init__(self) -> None:
        self.alive: set[int] = set()
        self.command_lines: dict[int, tuple[str, ...]] = {}
        self.ignores_term: set[int] = set()
        self.signals: list[tuple[int, int]] = []

    def is_alive(self, pid: int) -> bool:
        return pid in self.alive

    def command_line(self, pid: int) -> tuple[str, ...] | None:
        return self.command_lines.get(pid)

    def signal_group(self, pid: int, signum: int) -> None:
        self.signals.append((pid, signum))
        if signum == 9 or (signum == 15 and pid not in self.ignores_term):
            self.alive.discard(pid)


class FakeMonotonic:
    """A monotonic nanosecond clock that advances by a fixed step on every reading."""

    def __init__(self, *, step_seconds: float = 0.1, start_ns: int = 1_000_000_000) -> None:
        self._now_ns = start_ns
        self._step_ns = int(step_seconds * 1_000_000_000)

    def __call__(self) -> int:
        self._now_ns += self._step_ns
        return self._now_ns


class FakeSleep:
    """Records every sleep instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test outside ``tests/live/`` that opens a network connection.

    Installed in Phase 1, before any adapter existed, precisely so that Phase 3's first real HTTP
    code was written against a safety net that already had its own shape — rather than one grown
    alongside the adapter it is meant to keep honest.
    """
    if "live" in request.node.keywords:
        yield
        return

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "A test tried to open a network connection. The default suite must pass with no "
            "provider running (spec §18): use a recorded fixture, a fake transport, or "
            "FakeProvider. Tests that genuinely need a live provider belong in tests/live/ and "
            "carry the `live` marker."
        )

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    yield
