"""Supported test doubles: a deterministic provider and the vocabulary that scripts it.

Shipped API, not a test-suite helper. FreeWeight's benchmark runner, LoadCoach's job executor and
IdeaPress's workflows are all developed and tested against what is exported here, which is why the
[testing standards](../../docs/standards/testing-standards.md) §7 name this module as the one
place a model provider is replaced from — and why it is built before the Ollama adapter rather
than after it ([ADR-0007](../../docs/adr/0007-provider-abstraction.md) rule 6).

    >>> from modelrack.testing import FakeProvider, FakeScript
    >>> provider = FakeProvider(FakeScript(), seed=7)
    >>> identity = provider.resolve("fake-model")
    >>> provider.capabilities().streaming
    True

Deliberately **not** exported from :mod:`modelrack` itself. A test double one autocomplete away
from the production namespace is a test double that eventually ships inside an application, and
the boundary is worth the extra import line. The same reason SweatMeter's doubles live in
``sweatmeter.testing`` and WeightsDB's in ``weightsdb.testing``.

Everything here is deterministic given a script and a seed, and honest about what it cannot do:
a capability the script does not declare is refused with
:class:`~modelrack.errors.CapabilityUnsupported`, and a measurement the fake does not have is
``UNSUPPORTED`` rather than a plausible number
([ADR-0016](../../docs/adr/0016-unavailable-is-not-zero.md)). Reach for
:data:`MINIMAL_CAPABILITIES` when a consumer needs testing against the weakest provider it must
survive; that swap is one line precisely so it actually gets made.
"""

from __future__ import annotations

from modelrack.providers.fake import (
    DEFAULT_MODEL,
    FULL_CAPABILITIES,
    MINIMAL_CAPABILITIES,
    SIMULATED_TOKEN_CHARACTERS,
    FakeFailure,
    FakeFailureMode,
    FakeGeneration,
    FakeModel,
    FakeProvider,
    FakeScript,
    FakeToolCall,
)

__all__ = [
    "DEFAULT_MODEL",
    "FULL_CAPABILITIES",
    "MINIMAL_CAPABILITIES",
    "SIMULATED_TOKEN_CHARACTERS",
    "FakeFailure",
    "FakeFailureMode",
    "FakeGeneration",
    "FakeModel",
    "FakeProvider",
    "FakeScript",
    "FakeToolCall",
]
