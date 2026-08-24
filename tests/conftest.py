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
a root-level fixture is not).
"""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest
from baseaicore import ModelIdentity, ProviderKind

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from baseaicore import Clock

_OLLAMA_FIXTURE_DIR: Final[Path] = Path(__file__).parent / "fixtures" / "providers" / "ollama"
_OPENAI_COMPATIBLE_FIXTURE_DIR: Final[Path] = (
    Path(__file__).parent / "fixtures" / "providers" / "openai_compatible"
)

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
