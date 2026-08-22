# ModelRack — Specification

**Type:** Python package · **Import/distribution name:** `modelrack` · **Layer:** 3 (capability package)
**Status:** Specified, not implemented. **Decision record:** [ADR-0007](../../adr/0007-provider-abstraction.md).

---

## 1. Purpose

Be the suite's only model client. One implementation of "talk to a local inference runtime",
normalized into provider-neutral types, so that FreeWeight, LoadCoach and IdeaPress never contain
provider HTTP code, never parse provider JSON, and never disagree about what a token count or a
timing means.

## 2. Scope

* The `Provider` protocol and its normalized request/result types.
* Provider adapters: `OllamaProvider`, `OpenAICompatibleProvider`, `FakeProvider`.
* Model discovery, metadata inspection and normalization into `ModelDescriptor`.
* Non-streaming generation, streaming generation, chat, tool-enabled chat.
* Capability declaration and probing per provider.
* Model load/unload and residency inspection where the provider supports it.
* Timing and token normalization, keeping backend-reported and client-observed values separate.
* Error translation into a typed hierarchy.
* Cancellation of an in-flight generation.

## 3. Explicit non-goals

* **No routing.** Choosing a model is LoadCoach's job.
* **No scoring or benchmarking.** Interpreting a result is FreeWeight's job.
* **No retry or fallback policy.** ModelRack surfaces typed errors; the caller decides.
* No persistence, no database, no caching beyond a documented in-memory metadata cache with an
  explicit TTL and a `clear()`.
* No prompt construction or templating.
* No queueing, no concurrency control, no rate limiting (callers own their concurrency policy).
* No model downloading, conversion or quantization.
* No async API ([ADR-0003](../../adr/0003-sync-vs-async-strategy.md)).

## 4. Responsibilities

| Responsibility | Detail |
|---|---|
| Provider abstraction | One protocol, several adapters, one conformance suite |
| Discovery | List models, resolve a user reference (name, alias, prefix) to a `ModelIdentity` |
| Metadata | Normalize provider metadata into `ModelDescriptor`, preserving the raw response |
| Generation | Non-streaming and streaming, chat and completion, with tools where supported |
| Normalization | Tokens, timings, finish reasons, tool calls, thinking content where the provider exposes it |
| Capabilities | Declare what each provider can do; never assume |
| Residency | `load`, `unload`, `list_resident` where supported |
| Errors | Translate transport, protocol and semantic failures into typed errors |
| Testability | `FakeProvider` as a first-class deterministic implementation |

## 5. Dependencies

`baseaicore`, `httpx>=0.27,<1`. Nothing else. (Not `setspec` — ModelRack's types are in-process;
applications serialize them.)

## 6. Consumers

FreeWeight (benchmark execution), LoadCoach (job execution), IdeaPress (direct backend), and external
tools.

## 7. Public API

```python
class Provider(Protocol):
    kind: ProviderKind
    def health(self) -> ProviderHealth: ...
    def capabilities(self) -> ProviderCapabilities: ...
    def list_models(self) -> Sequence[ModelDescriptor]: ...
    def inspect_model(self, identity: ModelIdentity) -> ModelDescriptor: ...
    def resolve(self, reference: str) -> ModelIdentity: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]: ...
    def load(self, identity: ModelIdentity, profile: RuntimeProfile) -> LoadResult: ...
    def unload(self, identity: ModelIdentity) -> bool: ...
    def list_resident(self) -> Sequence[ResidentModel]: ...

@dataclass(frozen=True, slots=True)
class GenerationRequest:
    identity: ModelIdentity
    messages: Sequence[Message]              # or prompt=… for completion-style
    runtime_profile: RuntimeProfile = RuntimeProfile()
    sampling: SamplingParameters = SamplingParameters()   # temperature, top_p, top_k, seed,
                                                          # max_output_tokens, stop, repeat_penalty
    tools: Sequence[ToolDefinition] = ()
    response_format: ResponseFormat | None = None         # TEXT | JSON | JSON_SCHEMA(schema)
    timeout_seconds: float | None = None
    cancel: CancellationToken | None = None
    metadata: Mapping[str, Any] = {}                      # caller correlation IDs, never sent

@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    identity: ModelIdentity
    finish_reason: FinishReason              # STOP, LENGTH, TOOL_CALLS, CONTENT_FILTER,
                                             # CANCELLED, ERROR, UNKNOWN
    usage: TokenUsage                        # input/output/thinking/tool tokens + chars/words/bytes
    timing: Timing                           # backend_* and client_* kept separate
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: str | Unsupported = UNSUPPORTED
    provider_version: str | None = None
    raw: Mapping[str, Any] = {}              # diagnostics only

StreamEvent = TokenDelta | ToolCallDelta | ThinkingDelta | StreamCompleted | StreamFailed

@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    streaming: bool; tool_calling: bool; structured_output: bool; json_mode: bool
    token_counts: bool; token_level_chunks: bool; thinking_control: bool; logprobs: bool
    force_unload: bool; residency_query: bool; kv_metrics: bool; context_configurable: bool
    embedding: bool

# Adapters
OllamaProvider(base_url="http://127.0.0.1:11434", *, timeout=…, client=None)
OpenAICompatibleProvider(base_url, *, api_key=None, timeout=…, client=None)
FakeProvider(script: FakeScript | None = None, *, seed: int = 0)

# Errors (all subclass baseaicore.SuiteError)
ProviderError               PROVIDER_ERROR
├── ProviderUnavailable     PROVIDER_UNAVAILABLE
├── ProviderTimeout         PROVIDER_TIMEOUT
├── ProviderProtocolError   PROVIDER_PROTOCOL_ERROR
├── ModelNotFound           MODEL_NOT_FOUND
├── ContextLimitExceeded    CONTEXT_LIMIT_EXCEEDED
├── CapabilityUnsupported   CAPABILITY_UNSUPPORTED
├── GenerationCancelled     GENERATION_CANCELLED
└── ProviderRejected        PROVIDER_REJECTED        # 4xx from the provider, e.g. bad options
```

## 8. Inputs

A provider base URL and options; `GenerationRequest` objects; user model references.

## 9. Outputs

`ModelDescriptor`, `GenerationResult`, `StreamEvent` sequences, `ProviderHealth`,
`ProviderCapabilities`, `LoadResult`, `ResidentModel`, and typed errors.

## 10. Data ownership

None persistent. An optional in-memory metadata cache (default TTL 300 s) is explicitly documented,
inspectable and clearable; it never survives the process.

## 11. Public contracts

1. Applications never see provider JSON except through `raw`, which is diagnostics only.
2. Every unavailable measurement is `UNSUPPORTED`, never zero
   ([ADR-0016](../../adr/0016-unavailable-is-not-zero.md)).
3. Backend-reported timings and client-observed timings are separate fields and are never merged.
4. `token_level_chunks` gates any per-token latency claim; when false, inter-chunk latency is exactly
   that, and callers must not relabel it.
5. Every adapter passes the same conformance suite.
6. Streaming is cancellable within one chunk boundary and leaks no connection.
7. Errors are typed; no adapter raises a raw `httpx` exception.
8. `resolve()` records the alias it resolved from, and never resolves silently in a way that hides a
   retag.
9. Every digest a provider reports is normalized to `"sha256:" + 64 lowercase hex` through
   `baseaicore.normalize_digest`. A digest that will not normalize is discarded with a recorded
   reason, yielding a `name_only` identity rather than a malformed one
   ([ADR-0024 §2](../../adr/0024-canonical-id-and-model-references.md)). ModelRack is the only place
   in the suite that sees a raw provider digest, so it is the only place this can be got right.
10. `capabilities().context_configurable` is load-bearing, not informational: it is what tells a
   caller whether it may set a served context or must record one as assumed
   ([ADR-0023 §4](../../adr/0023-runtime-profile-resolution.md)). An adapter that cannot configure
   context declares `False` rather than accepting the setting and ignoring it.

## 12. Configuration

Constructor arguments only — base URL, timeouts (connect/read/total), headers, an injected
`httpx.Client`, cache TTL, `verify` for TLS. ModelRack reads no environment variable and no file; the
application owns configuration.

## 13. Error behaviour

| Condition | Error | Notes |
|---|---|---|
| Connection refused / DNS failure | `ProviderUnavailable` | Includes the base URL |
| Read/connect timeout | `ProviderTimeout` | Includes elapsed time and the limit |
| Non-JSON or unexpected JSON | `ProviderProtocolError` | Raw body attached in `details` (truncated), for the caller to store as an artifact |
| 404 / model missing | `ModelNotFound` | Includes the reference and the known model count |
| Provider reports a context overflow | `ContextLimitExceeded` | Includes requested and maximum where known |
| Tools/schema requested but unsupported | `CapabilityUnsupported` | Names the capability |
| Cancellation token triggered | `GenerationCancelled` | Partial text preserved in `details`. Only reachable from `stream()`; `generate()` offers no boundary at which a token can take effect, which is why LoadCoach always calls `stream()` and assembles the response itself ([LoadCoach API §5](../../apps/loadcoach/api.md)) |
| 4xx with a provider message | `ProviderRejected` | Provider message preserved verbatim |
| Stream truncated without a terminal chunk | `ProviderProtocolError` | Partial result preserved |

Never retried internally. Never swallowed. Never converted into an empty result.

## 14. Security considerations

* Base URLs are validated; only `http`/`https` schemes accepted. Non-loopback URLs are permitted but
  the health result flags them as remote so callers can surface egress.
* API keys are passed by the caller, sent only in headers, and never logged, never in `raw`, never in
  error `details` (a test asserts this).
* Response size caps (configurable; default 64 MiB total, 8 MiB per chunk) prevent memory exhaustion.
* Provider responses are parsed defensively; no `eval`, no dynamic dispatch on response content.
* Tool definitions are passed through unmodified; ModelRack never executes a tool call — it returns
  the request to the caller.
* Timeouts are mandatory (a default exists; `None` means "use the default", never "no timeout").

## 15. Performance

| Measure | Target |
|---|---|
| Overhead per non-streaming request (excluding provider time) | ≤ 5 ms |
| Overhead per streamed chunk | ≤ 1 ms |
| `list_models` with metadata cached | ≤ 10 ms |
| `list_models` cold, 20 models | ≤ 3 s (dominated by per-model `show` calls) |
| Memory per active stream | ≤ 1 MiB, flat regardless of response length |

Connection pooling via a shared `httpx.Client`. Streaming never accumulates the full response more
than once.

## 16. Cross-platform

Pure Python over HTTP; fully portable. `OllamaProvider` requires only a reachable endpoint, not a
local installation.

## 17. Observability

* No logging from the library at INFO or above (a library must not configure or spam the host's
  logs). DEBUG-level logs of request shape (never content) are available under
  `modelrack.<adapter>`.
* Every result carries timing and token data so callers can log and persist it.
* An optional `on_event` callback (request started/completed/failed, chunk received) lets an
  application emit its own structured logs and events without ModelRack knowing about them.

## 18. Test strategy

| Area | Tests |
|---|---|
| Conformance suite | Runs against `FakeProvider`, `OllamaProvider` (recorded), `OpenAICompatibleProvider` (recorded): discovery, inspect, generate, stream, cancel, errors, capabilities |
| Ollama adapter | Recorded `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/ps` responses; timing extraction; digest extraction; multi-model listing |
| Metadata normalization | Every architecture field present/absent; malformed values → `UNSUPPORTED`; `raw` preserved byte-for-byte |
| Streaming | Ordered deltas, terminal event, truncated stream, mid-stream error, cancellation, chunk-size variance, unicode split across chunks |
| Errors | Every row of §13, with the documented type and details, and a test in each consumer asserting it maps that error to a documented code rather than to `INTERNAL_ERROR` |
| Digest normalization | Bare hex, prefixed, uppercase, truncated, non-hex and absent digests from recorded fixtures, each to the documented identity confidence |
| Fake provider | Deterministic output for a seed; scripted delays, tool calls, malformed responses, timeouts |
| Security | API key never appears in `raw`, `details` or DEBUG logs; oversize response rejected; non-http scheme rejected |
| Performance | Overhead budgets against a fake HTTP transport |
| Live (marked) | Real Ollama: discovery, generate, stream, unload, timings plausible |

Coverage floor: **95 %**. The default suite must pass with no Ollama running.

## 19. Compatibility and versioning

* Semantic versioning; pre-1.0 `0.x`.
* Adding a provider or a capability flag is a minor bump. Changing the `Provider` protocol is major.
* Recorded fixtures record the provider version they came from; a provider version bump triggers
  re-capture and a note in the changelog.
* The `Provider` protocol is not declared stable until at least two real adapters exist (Ollama and
  OpenAI-compatible), by design.

## 20. Acceptance criteria

1. FreeWeight and LoadCoach contain **no** provider HTTP code — verified by a grep-based boundary
   test in each repository.
2. The conformance suite passes for all three shipped adapters.
3. The full default test suite passes with no Ollama installed.
4. A live test against Ollama 0.32.13 discovers models, generates, streams and unloads.
5. Cancelling a stream stops it within one chunk and leaves no open connection (asserted by a
   connection-count test).
6. Every error row in §13 is produced by a test.
7. `mypy --strict`, `ruff`, `lint-imports` clean; coverage ≥ 95 %.

## 21. Future extensions

* `LlamaCppProvider` (HTTP server mode, plus optional `llama-bench` subprocess integration for
  FreeWeight).
* `VLLMProvider` including its metrics endpoint (KV-cache utilization, prefix-cache hits).
* Embeddings API across providers.
* An async provider API, if [ADR-0003](../../adr/0003-sync-vs-async-strategy.md)'s revisit trigger
  fires.
* Token counting/tokenization where a provider exposes it.
* Multi-modal inputs (images) once a consumer needs them.
