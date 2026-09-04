# ModelRack — Specification

**Type:** Python package · **Import/distribution name:** `modelrack` · **Layer:** 3 (capability package)
**Status:** Phases 1–5 implemented in `modelrack 0.5.0`; every §20 acceptance criterion met.
**Decision record:** [ADR-0007](../../adr/0007-provider-abstraction.md).

---

## 1. Purpose

Be the suite's only model client. One implementation of "talk to a local inference runtime",
normalized into provider-neutral types, so that FreeWeight, LoadCoach and IdeaPress never contain
provider HTTP code, never parse provider JSON, and never disagree about what a token count or a
timing means.

## 2. Scope

* The `Provider` protocol and its normalized request/result types.
* Provider adapters: `OllamaProvider`, `OpenAICompatibleProvider`, `LlamaCppProvider`,
  `FakeProvider`.
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
  explicit TTL and a `clear()`. The one exception is process-supervision state: `LlamaCppProvider`
  writes pid files and captured stderr under a `state_dir` the application names inside its own
  data root, so a server orphaned by a crashed process can be recovered by the next one
  ([ADR-0062](../../adr/0062-llamacpp-serves-adapters-through-a-supervised-process.md)
  decision 6), and one versioned `digests.json` of computed artifact digests beside them
  ([ADR-0071](../../adr/0071-modelrack-persists-artifact-digests-in-a-json-file-the-application-names.md)),
  because a content hash is invalidated by content changing and never by a process ending. Never
  model data; the digest file is clearable and safe to delete.
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
    def list_models(self, *, refresh: bool = False) -> Sequence[ModelDescriptor]: ...
    def inspect_model(
        self, identity: ModelIdentity, *, refresh: bool = False
    ) -> ModelDescriptor: ...
    def resolve(self, reference: str, *, refresh: bool = False) -> ModelIdentity: ...
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
LlamaCppProvider(model_directory, *, state_dir, server_path="llama-server",
                 port_range=(8180, 8189), launcher=None, process_table=None,
                 digest_store=None, timeout=…, client=None)   # ADR-0062
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

None persistent, except the pid files and captured stderr `LlamaCppProvider` keeps under an
application-supplied `state_dir` (ADR-0062) — supervision state, never model data or a cache. An
optional in-memory metadata cache (default TTL 300 s) is explicitly documented, inspectable and
clearable; it never survives the process. Generation results are never cached, and neither are
residency or health — both are live state whose stale answer is worse than no answer.
`LlamaCppProvider`'s content digests are keyed by path and file stamp in an injectable
`DigestStore`, by default a versioned `<state_dir>/digests.json` written atomically and pruned of
files that no longer exist (ADR-0071); `clear_digest_cache()` removes it, and so may an operator.
An in-memory store remains available for a caller that wants no persistence.

Because a tag can be repointed at any moment, a TTL alone cannot make a cached digest trustworthy.
Every metadata read therefore takes a keyword-only `refresh: bool = False`, the explicit bypass a
caller who *knows* a model was re-pulled uses instead of waiting out an expiry. It is on the
protocol rather than on the adapters so that a caller holding a `Provider` can use it without
downcasting to a concrete adapter; an adapter that caches nothing accepts it and ignores it.

## 11. Public contracts

1. Applications never see provider JSON except through `raw`, which is diagnostics only.
2. Every unavailable measurement is `UNSUPPORTED`, never zero
   ([ADR-0016](../../adr/0016-unavailable-is-not-zero.md)) — and a token class the wire protocol
   has no way to bill is zero, never unavailable. The rule is per response: cache detail the
   protocol can express but a response omits means the server does no cache accounting (both
   cache classes `0`); cache detail that is present is reconciled into the disjoint classes; a
   response with no usage object at all reports every class `UNSUPPORTED`
   ([ADR-0070](../../adr/0070-an-absent-token-class-is-zero-only-where-the-protocol-cannot-bill-it.md)).
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
`httpx.Client`, cache TTL, `verify` for TLS. ModelRack reads no environment variable and no
configuration file; the application owns configuration. `LlamaCppProvider` reads GGUF files under
an application-supplied `model_directory` and spawns `llama-server` through an injected launcher
(ADR-0062); both directories it touches are named by the application, and its digest file lives
in the second (ADR-0071).

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
local installation. `LlamaCppProvider` is the exception: it requires a local `llama-server` binary
and POSIX process groups (`start_new_session`, `killpg`) for its kill-tree guarantee, and says so
rather than degrading.

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
| Conformance suite | Runs against `FakeProvider`, `OllamaProvider` (recorded), `OpenAICompatibleProvider` (recorded), `LlamaCppProvider` (recorded transport plus a fake process launcher): discovery, inspect, generate, stream, cancel, errors, capabilities; usage per ADR-0070 — no cache detail → cache classes `0` and an estimate that totals, cache detail → disjoint classes, no usage object → every class `UNSUPPORTED` |
| Ollama adapter | Recorded `/api/tags`, `/api/show`, `/api/chat`, `/api/generate`, `/api/ps` responses; timing extraction; digest extraction; multi-model listing |
| llama.cpp adapter | Recorded `/health`, `/props`, `/completion` and `/v1/chat/completions` responses; a fake process launcher for spawn, health wait, kill-tree, pid files and orphan recovery, with the real launcher proven against shell processes; GGUF headers and content digests from files the tests write; `tokens_cached` never read as cached input |
| Metadata normalization | Every architecture field present/absent; malformed values → `UNSUPPORTED`; `raw` preserved byte-for-byte |
| Streaming | Ordered deltas, terminal event, truncated stream, mid-stream error, cancellation, chunk-size variance, unicode split across chunks |
| Errors | Every row of §13, with the documented type and details, and a test in each consumer asserting it maps that error to a documented code rather than to `INTERNAL_ERROR` |
| Digest normalization | Bare hex, prefixed, uppercase, truncated, non-hex and absent digests from recorded fixtures, each to the documented identity confidence |
| Fake provider | Deterministic output for a seed; scripted delays, tool calls, malformed responses, timeouts |
| Security | API key never appears in `raw`, `details` or DEBUG logs; oversize response rejected; non-http scheme rejected |
| Performance | Overhead budgets against a fake HTTP transport |
| Live (marked) | Real Ollama: discovery, generate, stream, unload, timings plausible |

Coverage floor: **95 %**. The default suite must pass with no Ollama running and no
`llama-server` installed.

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
2. The conformance suite passes for all shipped adapters (fake, Ollama, OpenAI-compatible,
   llama.cpp).
3. The full default test suite passes with no Ollama installed and no `llama-server` installed.
4. A live test against Ollama 0.32.13 discovers models, generates, streams and unloads.
5. Cancelling a stream stops it within one chunk and leaves no open connection (asserted by a
   connection-count test).
6. Every error row in §13 is produced by a test.
7. `mypy --strict`, `ruff`, `lint-imports` clean; coverage ≥ 95 %.

## 21. Future extensions

* `llama-bench` subprocess integration for FreeWeight, beside the `LlamaCppProvider` Phase 6
  added.
* `VLLMProvider` including its metrics endpoint (KV-cache utilization, prefix-cache hits).
* Embeddings API across providers.
* An async provider API, if [ADR-0003](../../adr/0003-sync-vs-async-strategy.md)'s revisit trigger
  fires.
* Token counting/tokenization where a provider exposes it.
* Multi-modal inputs (images) once a consumer needs them.
