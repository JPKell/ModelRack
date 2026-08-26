"""modelrack — the suite's only model client.

Layer 3: a capability package over :mod:`baseaicore`'s vocabulary and ``httpx``. One
implementation of "talk to a local inference runtime", normalized into provider-neutral types, so
that FreeWeight, LoadCoach and IdeaPress never contain provider HTTP code, never parse provider
JSON, and never disagree about what a token count or a timing means
([spec §1](../../docs/packages/modelrack/spec.md)).

What is exported below is the public API as of Phase 5
(``docs/packages/modelrack/development-plan.md``): the provider-neutral request and result
vocabulary, the streamed-event union with its cancellation token, the ``Provider`` protocol and the
types describing what a provider is, the full error hierarchy, and the three operational modules
Phase 5 added — the residency vocabulary (:mod:`modelrack.residency`), the one metadata cache this
package is allowed to have (:mod:`modelrack.cache`), and the optional observability hook
(:mod:`modelrack.events`).

The first adapter that ships is the **fake** one, deliberately
(ADR-0007 rule 6): ``FakeProvider`` is imported from
``modelrack.testing``, not from here, so that the rest of the suite can be developed and tested
without a GPU, a model or a running runtime, while a test double stays one import away from the
production namespace rather than inside it. The first *real* adapter,
:class:`~modelrack.providers.ollama.OllamaProvider`, is imported from
``modelrack.providers.ollama`` for a related but distinct reason: it is the one place in this
package that imports ``httpx``, and a process that only ever talks to the fake — most of this
suite's own test runs — has no reason to pay for that import. The second real adapter,
:class:`~modelrack.providers.openai_compatible.OpenAICompatibleProvider`, is imported from
``modelrack.providers.openai_compatible`` for the same reason, and exists to prove the vocabulary
below is not secretly shaped around Ollama: nothing in this module changed to support it.

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
``0`` (ADR-0016); and what a provider *reported*
about its own work is never merged with what this process *observed*, which is why
:class:`Timing` prefixes every field and offers no combined duration.
"""

from __future__ import annotations

from modelrack.__about__ import __version__
from modelrack.cache import (
    DEFAULT_METADATA_TTL_SECONDS,
    CacheStats,
    MetadataCache,
    MetadataSnapshot,
)
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
from modelrack.events import (
    EventCallback,
    ProviderEvent,
    ProviderEventKind,
)
from modelrack.provider import (
    LoadResult,
    Provider,
    ProviderCapabilities,
    ProviderHealth,
    ProviderStatus,
    ResidentModel,
    refuse_capability,
    require_capability,
)
from modelrack.residency import (
    FORCE_UNLOAD,
    RESIDENCY_QUERY,
    ResidencySupport,
    find_resident,
    is_resident,
    residency_support,
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
    "DEFAULT_METADATA_TTL_SECONDS",
    "FORCE_UNLOAD",
    "RESIDENCY_QUERY",
    "CacheStats",
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
    "EventCallback",
    "MetadataCache",
    "MetadataSnapshot",
    "ProviderEvent",
    "ProviderEventKind",
    "ResidencySupport",
    "find_resident",
    "is_resident",
    "refuse_capability",
    "require_capability",
    "residency_support",
    "__version__",
]
