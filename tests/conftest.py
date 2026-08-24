"""Shared fixtures: a socket guard, a canonical identity, and a deterministic clock.

The socket guard is the one that matters most in this repository. ModelRack is an HTTP client, so
"the default suite passes with no Ollama running" ([spec §18](../docs/packages/modelrack/spec.md),
acceptance criterion 3) is not something that stays true by good intentions: one adapter test that
quietly reaches ``127.0.0.1:11434`` passes on the maintainer's machine and fails in CI, or worse,
passes in CI against nothing and asserts on a fabricated failure path.

The guard therefore fails any test that opens a socket, and exempts only ``tests/live/`` — the
directory whose entire purpose is to talk to a real provider, and which is deselected by default
through the ``live`` marker (``addopts = "-m 'not live and not performance'"``).
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import pytest
from baseaicore import ModelIdentity, ProviderKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from baseaicore import Clock

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


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test outside ``tests/live/`` that opens a network connection.

    Phase 1 ships no adapter, so nothing here *can* reach a provider yet — which is exactly when
    to install the guard. Adding it in Phase 3 alongside the first HTTP code would mean writing
    the adapter and its safety net at the same moment, and the net would be shaped by whatever the
    adapter already did.
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
