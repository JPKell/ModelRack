# ADR-0007 — Provider abstraction and Ollama first

**Status:** Accepted (2026-08-21)
**Amended 2026-08-21** (final architecture audit): the capability list below is aligned with `ProviderCapabilities` in the ModelRack specification, which is normative.

## Context

Three applications need to talk to local model runtimes. The prior projects wrote three separate
Ollama clients, each with its own error handling and its own idea of what a response looks like; one
of them additionally imported application domain types, which made it unreusable.

The suite must not couple application code to any provider's JSON. It must also be honest about
provider differences: Ollama exposes load/prompt-eval/decode durations and rich model metadata;
llama.cpp exposes different timings; vLLM exposes KV-cache and prefix-cache counters; a generic
OpenAI-compatible endpoint exposes very little. A benchmark suite that pretends these are the same
produces wrong numbers.

## Decision

**One provider abstraction — `modelrack.Provider` — with Ollama as the only fully supported provider
in the first releases.**

```python
class Provider(Protocol):
    def health(self) -> ProviderHealth: ...
    def capabilities(self) -> ProviderCapabilities: ...
    def list_models(self) -> Sequence[ModelDescriptor]: ...
    def inspect_model(self, identity: ModelIdentity) -> ModelDescriptor: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]: ...
    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult: ...
    def unload(self, identity: ModelIdentity) -> bool: ...
```

Rules:

1. Applications never construct or read provider JSON. The raw response is available on results as
   `raw` for **diagnostics only**; using it for business logic is a boundary violation.
2. Every provider declares its `ProviderCapabilities`. The normative field set is the dataclass in
   [ModelRack §7](../packages/modelrack/spec.md): `streaming`, `tool_calling`, `structured_output`,
   `json_mode`, `token_counts`, `token_level_chunks`, `thinking_control`, `logprobs`, `force_unload`,
   `residency_query`, `kv_metrics`, `context_configurable`, `embedding`. Callers check capabilities;
   they never assume. `context_configurable` in particular gates whether a caller may set a served
   context ([ADR-0023](0023-runtime-profile-resolution.md)).
3. Unsupported measurements are `Unsupported`, never zero ([ADR-0016](0016-unavailable-is-not-zero.md)).
4. Provider-specific settings live in `RuntimeProfile.provider_options` and inside the adapter.
5. Errors are translated into a typed hierarchy: `ProviderUnavailable`, `ProviderTimeout`,
   `ModelNotFound`, `ProviderProtocolError`, `ContextLimitExceeded`, `CapabilityUnsupported`.
6. ModelRack ships `FakeProvider` as a first-class, tested component — built **before** the Ollama
   adapter — so the entire suite is testable without a GPU or a model.
7. Shipped in the first releases: `OllamaProvider`, `OpenAICompatibleProvider` (reduced
   capabilities), `FakeProvider`. `llama.cpp` and `vLLM` are documented extension points with no
   promised date.

## Alternatives considered

**Depend on the `ollama` Python SDK.** Rejected: it binds the abstraction to one provider's release
cadence and object model, and the endpoints we need (`/api/tags`, `/api/show`, `/api/chat`,
`/api/generate`, `/api/ps`) are simple, stable HTTP that `httpx` handles directly with better error
control.

**Target only the OpenAI-compatible API.** Every runtime speaks it, so one adapter would cover
everything. Rejected: it is the lowest common denominator. Ollama's `/api/show` metadata (layers,
KV heads, head dimension, quantization, context) and its per-phase durations are exactly what the
KV-cache and performance benchmarks need, and the OpenAI shape exposes none of it.

**LiteLLM or a similar universal client.** Rejected: a large dependency oriented toward remote paid
providers, normalizing away precisely the local-runtime details the suite measures.

**No abstraction — each application calls Ollama directly.** Rejected: this is the documented prior
failure.

## Consequences

*Positive.* One client, one set of error translations, one place to fix a bug. Applications become
provider-agnostic for free. `FakeProvider` makes deterministic testing of the whole stack possible.
Capability declarations keep benchmarks honest about what a runtime can actually report.

*Negative.* The abstraction can only expose what the weakest supported provider offers, unless
callers branch on capabilities — which they must. Adding a provider is real work: the conformance
suite plus recorded fixtures plus a capability declaration.

*Negative.* Ollama-shaped concepts (`keep_alive`, `/api/ps` residency) risk leaking into the
interface. Mitigated by keeping them in `provider_options` and by requiring the second and third
adapters to be written before the interface is declared stable at 1.0.

## Revisit when

A second provider is implemented and the interface proves to fit it badly — that is the moment to
reshape, before 1.0 freezes the API.
