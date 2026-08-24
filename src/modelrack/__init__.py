"""modelrack — the suite's only model client.

Layer 3: a capability package over :mod:`baseaicore`'s vocabulary and ``httpx``. One
implementation of "talk to a local inference runtime", normalized into provider-neutral types, so
that FreeWeight, LoadCoach and IdeaPress never contain provider HTTP code, never parse provider
JSON, and never disagree about what a token count or a timing means
([spec §1](../../docs/packages/modelrack/spec.md)).

What is exported below is the public API as of Phase 3
(``docs/packages/modelrack/development-plan.md``): the provider-neutral request and result
vocabulary, the streamed-event union with its cancellation token, the ``Provider`` protocol and the
types describing what a provider is, and the full error hierarchy.

The first adapter that ships is the **fake** one, deliberately
([ADR-0007](../../docs/adr/0007-provider-abstraction.md) rule 6): ``FakeProvider`` is imported from
``modelrack.testing``, not from here, so that the rest of the suite can be developed and tested
without a GPU, a model or a running runtime, while a test double stays one import away from the
production namespace rather than inside it. The first *real* adapter,
:class:`~modelrack.providers.ollama.OllamaProvider`, is imported from
``modelrack.providers.ollama`` for a related but distinct reason: it is the one place in this
package that imports ``httpx``, and a process that only ever talks to the fake — most of this
suite's own test runs — has no reason to pay for that import.
``OpenAICompatibleProvider`` arrives in Phase 4.

Anything not listed in ``__all__`` is private and may change without a version bump.

    >>> from baseaicore import ModelIdentity, ProviderKind
    >>> from modelrack import GenerationRequest, Message, Role
    >>> request = GenerationRequest(
    ...     identity=ModelIdentity(ProviderKind.OLLAMA, "qwen3.5:9b-q8_0"),
    ...     messages=(Message(role=Role.USER, content="Explain KV caching."),),
    ... )
    >>> request.timeout_seconds is None      # the adapter's default, never "no timeout"
    True

Two invariants run through every type here. An unavailable measurement is ``UNSUPPORTED``, never
``0`` ([ADR-0016](../../docs/adr/0016-unavailable-is-not-zero.md)); and what a provider *reported*
about its own work is never merged with what this process *observed*, which is why
:class:`Timing` prefixes every field and offers no combined duration.
"""

from __future__ import annotations

from modelrack.__about__ import __version__
from modelrack.errors import (
    CapabilityUnsupported,
    ContextLimitExceeded,
    GenerationCancelled,
    ModelNotFound,
    ProviderError,
    ProviderProtocolError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnavailableReason,
)
from modelrack.provider import (
    LoadResult,
    Provider,
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    ResidentModel,
)
from modelrack.streaming import (
    CancellationToken,
    StreamCompleted,
    StreamEvent,
    StreamFailed,
    ThinkingDelta,
    TokenDelta,
    ToolCallDelta,
)
from modelrack.types import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    Message,
    ResponseFormat,
    ResponseFormatKind,
    Role,
    SamplingParameters,
    Timing,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "CancellationToken",
    "CapabilityUnsupported",
    "ContextLimitExceeded",
    "FinishReason",
    "GenerationCancelled",
    "GenerationRequest",
    "GenerationResult",
    "GenerationUsage",
    "LoadResult",
    "Message",
    "ModelNotFound",
    "Provider",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderHealth",
    "ProviderProtocolError",
    "ProviderRejected",
    "ProviderStatus",
    "ProviderTimeout",
    "ProviderUnavailable",
    "ProviderUnavailableReason",
    "ResidentModel",
    "ResponseFormat",
    "ResponseFormatKind",
    "Role",
    "SamplingParameters",
    "StreamCompleted",
    "StreamEvent",
    "StreamFailed",
    "ThinkingDelta",
    "Timing",
    "TokenDelta",
    "TokenUsage",
    "ToolCall",
    "ToolCallDelta",
    "ToolDefinition",
    "__version__",
]
