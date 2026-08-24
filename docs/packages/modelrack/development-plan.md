# ModelRack — Development Plan

**Sequence position:** third component. Depends on BaseAiCore Phase 4.
**Target:** `modelrack 0.5.0` by the end of Phase 5.

The ordering is deliberate and inherited from the prior project's best decision: **the fake provider
is built before the real one**, so that every downstream component can be developed and tested
without a GPU, a model, or a running Ollama.

---

## Phase 1 — Types and the Provider protocol

**Goal:** the provider-neutral vocabulary exists and type-checks; no I/O yet.

**Prerequisites:** `baseaicore>=0.4,<0.5`.

**Work**
* Repository skeleton.
* `types.py`: `Message`, `ToolDefinition`, `ToolCall`, `SamplingParameters`, `ResponseFormat`,
  `FinishReason`, `TokenUsage`, `Timing`, `GenerationRequest`, `GenerationResult`.
* `streaming.py`: `StreamEvent` union — `TokenDelta`, `ToolCallDelta`, `ThinkingDelta`,
  `StreamCompleted`, `StreamFailed`; `CancellationToken`.
* `provider.py`: the `Provider` protocol, `ProviderHealth`, `ProviderCapabilities`, `LoadResult`,
  `ResidentModel`.
* `errors.py`: the full error hierarchy from the spec.

**Files/subsystems**
```text
src/modelrack/{__init__,__about__,types,streaming,provider,errors}.py
tests/unit/{test_types,test_streaming_types,test_errors}.py
```

**Tests**
* Request/result construction and validation; `UNSUPPORTED` accepted for every optional measurement.
* `Timing` keeps backend and client values distinct — a test asserts there is no combined field.
* Error hierarchy codes match the spec exactly.
* A structural `Protocol` conformance test compiles against a stub implementation.

**Acceptance criteria**
1. `mypy --strict` clean; the protocol is satisfiable by a stub.
2. No field named in a way that would let a caller confuse backend and client timings.

**Known risks:** designing the request/result shape around Ollama specifically. Mitigated by writing
the OpenAI-compatible adapter in Phase 4 and revisiting before 1.0.
**Likely failure modes:** an over-large request object; a result that cannot express a tool-call-only
response.
**Gold standards:** provider-neutral vocabulary; unsupported-safe measurements.
**Deferred:** all adapters.

---

## Phase 2 — FakeProvider

**Goal:** the entire suite can be developed and tested without a model. This phase is a prerequisite
for FreeWeight's runner, LoadCoach's executor and IdeaPress's workflows.

**Prerequisites:** Phase 1.

**Work**
* `providers/fake.py`: `FakeProvider` and `FakeScript` — a declarative description of what the fake
  should do.
* Scriptable behaviours: fixed or seeded-random text, per-token or per-chunk delays, configurable
  chunk sizes, token counts, tool calls (single, multiple, malformed arguments), thinking content,
  finish reasons, malformed responses, protocol errors, timeouts, mid-stream failures, slow first
  token, and a configurable model catalogue with digests and metadata.
* Deterministic given a seed; identical output across processes and platforms.

**Files/subsystems**
```text
src/modelrack/providers/fake.py
src/modelrack/testing.py            # helpers other repositories import in their tests
tests/unit/test_fake_provider.py
tests/contract/test_conformance.py  # the shared suite, first run against the fake
```

**Tests**
* Determinism: same seed ⇒ identical text, chunking and token counts, twice and across processes.
* Every scripted failure mode is reachable and produces the documented error.
* Streaming yields the terminal event exactly once; cancellation stops within one chunk.
* Conformance suite passes.

**Acceptance criteria**
1. A downstream repository can `from modelrack.testing import FakeProvider, FakeScript` and simulate
   a full benchmark or job without Ollama.
2. The conformance suite exists and passes against the fake.
3. Coverage ≥ 95 %.

**Known risks:** a fake that is too forgiving, hiding real provider behaviour. Mitigated by scripting
the *nasty* cases explicitly and by requiring the same conformance suite for real adapters.
**Likely failure modes:** nondeterminism creeping in via dict ordering or time; a fake that cannot
express a real failure the Ollama adapter later hits.
**Gold standards:** deterministic, scriptable, shipped as supported API (`modelrack.testing`).
**Deferred:** real providers.

---

## Phase 3 — OllamaProvider

**Goal:** real models are discoverable, inspectable and runnable through the abstraction, with
timings and digests captured.

**Prerequisites:** Phase 2.

**Work**
* `providers/ollama.py`: health (`/api/version`), discovery (`/api/tags`), inspection (`/api/show`),
  generation (`/api/chat`, `/api/generate`), streaming (NDJSON), residency (`/api/ps`), load/unload
  (via `keep_alive`), tool calls, structured output (`format`), and `think` handling where exposed.
* Metadata normalization: family, parameter count, quantization, context, layers, attention heads, KV
  heads, head dimension, embedding dimension, vocab size, licence — with `raw` preserved. **Digests
  normalized through `baseaicore.normalize_digest`** to `"sha256:" + 64 lowercase hex`; a value that
  will not normalize is discarded with a reason and yields a `name_only` identity, never a malformed
  one (ADR-0024 §2).
* Timing extraction: `load_duration`, `prompt_eval_duration`, `eval_duration`, `total_duration`
  mapped to `Timing.backend_*`; client-observed wall time and TTFT measured with
  `time.perf_counter_ns()` into `Timing.client_*`.
* Error translation for every row of the spec's §13 table.
* Capability declaration for Ollama, including `token_level_chunks` and `context_configurable` set
  truthfully — the latter is what tells a caller whether it may set a served context or must record
  one as assumed (ADR-0023 §4).
* Recorded fixtures captured from Ollama 0.32.13, each annotated with the provider version.

**Files/subsystems**
```text
src/modelrack/providers/ollama.py
src/modelrack/providers/_http.py         # shared client construction, size caps, timeouts
tests/unit/test_ollama_adapter.py
tests/fixtures/providers/ollama/*.json|.jsonl
tests/live/test_ollama_live.py           # marked
```

**Tests**
* Discovery with 0, 1 and 20 models; alias resolution recorded, not hidden.
* `show` parsing: complete metadata, partial metadata, unknown architecture, missing fields →
  `UNSUPPORTED`.
* Digest normalization across the recorded fixtures: bare hex, prefixed, uppercase, absent — each to
  the documented identity confidence.
* Generation: text, token counts, all four backend durations, finish reasons.
* Streaming: NDJSON chunking, unicode split across chunk boundaries, terminal chunk, truncated stream,
  mid-stream error, cancellation.
* Errors: connection refused, timeout, 404 model, 400 bad options, non-JSON body, oversize response.
* Conformance suite passes against the recorded transport.
* Live (marked): real discovery, generation, streaming, unload, plausible timings.

**Acceptance criteria**
1. All unit tests pass with **no Ollama installed**.
2. The live test passes against Ollama 0.32.13 on the reference machine.
3. Digest is captured for every model that exposes one; identity confidence is set accordingly.
4. Backend and client timings are both present and distinct.
5. Coverage ≥ 95 %.

**Known risks:** Ollama's API evolving; undocumented fields. Mitigated by name-based parsing,
`UNSUPPORTED` for anything missing, `raw` preservation, and version-annotated fixtures.
**Likely failure modes:** treating a missing token count as 0; mislabelling chunk latency as token
latency; losing the digest and silently producing `name_only` identities.
**Gold standards:** one Ollama client for the whole suite; unsupported-safe normalization; typed errors.
**Deferred:** OpenAI-compatible adapter; llama.cpp; vLLM.

---

## Phase 4 — OpenAICompatibleProvider and capability honesty

**Goal:** a second real adapter proves the abstraction is not Ollama-shaped, and capability declarations become load-bearing.

**Prerequisites:** Phase 3.

**Work**
* `providers/openai_compatible.py`: `/v1/models`, `/v1/chat/completions` (streaming and not), tools
  where advertised, `response_format` where advertised, usage extraction, SSE `data:` parsing.
* Honest capability declaration: no digest (⇒ `name_only` identity), limited metadata, no residency
  control, no KV metrics, `token_level_chunks = False`.
* Any place where the Phase-1 types fit Ollama but not this adapter is fixed here, before 1.0.
* Recorded fixtures from a representative server (llama.cpp server and/or LM Studio), version-annotated.

**Files/subsystems**
```text
src/modelrack/providers/openai_compatible.py
tests/unit/test_openai_compatible_adapter.py
tests/fixtures/providers/openai_compatible/*
```

**Tests**
* Conformance suite passes for this adapter, with capability-gated tests skipped **explicitly**
  (recorded as `unsupported`, never silently passed).
* Identity confidence is `name_only`; consumers can detect it.
* SSE parsing: multi-line data, `[DONE]`, keep-alive comments, malformed frames.
* API key sent in the header and absent from `raw`, `details` and DEBUG logs.

**Acceptance criteria**
1. Two real adapters pass one conformance suite.
2. Provider capabilities are derived from the adapters' declarations, not hand-written.
3. Any type changes required by this adapter are made and documented in the changelog.

**Known risks:** the lowest-common-denominator temptation — flattening the abstraction until Ollama's
rich data is lost. Mitigated by capability flags plus `UNSUPPORTED`, never by removing fields.
**Likely failure modes:** silently pretending a capability exists; assuming digests.
**Gold standards:** honest capability reporting; a protocol proven against two real providers.
**Deferred:** llama.cpp-specific and vLLM-specific adapters.

---

## Phase 5 — Residency, cancellation hardening, publication

**Goal:** the operational features LoadCoach depends on are solid, and the package is published.

**Prerequisites:** Phases 1–4.

**Work**
* `list_resident()`, `load()`, `unload()` with capability gating and clear behaviour when unsupported.
* Cancellation hardening: token propagation into the HTTP layer, guaranteed connection close, partial
  result preservation, no thread or socket leak.
* Optional metadata cache with an explicit TTL, `clear()`, and cache-hit reporting.
* `on_event` callback hook.
* Performance tests for the overhead budgets.
* README, provider documentation, quickstart; publish `modelrack 0.5.0`.

**Files/subsystems**
```text
src/modelrack/{residency.py,cache.py,events.py}
tests/unit/{test_residency,test_cancellation,test_cache}.py
tests/performance/test_overhead.py
docs/quickstart.md
```

**Tests**
* Cancellation mid-stream: stops within one chunk, closes the connection, preserves partial text —
  asserted with an open-connection count before and after.
* 100 sequential streams leak no connections and keep memory flat.
* Unload on a provider without `force_unload` raises `CapabilityUnsupported`, not a silent no-op.
* Cache: hit/miss behaviour, TTL expiry, `clear()`, and no caching of generation results (only
  metadata).
* Performance: per-request and per-chunk overhead within budget against a fake transport.

**Acceptance criteria**
1. Every §20 criterion in the [spec](spec.md) is met.
2. `modelrack 0.5.0` published; installable and usable in a standalone script with only
   `baseaicore` alongside it.
3. Performance budgets met.

**Known risks:** cancellation semantics differing per provider. Mitigated by the conformance suite
requiring the same observable behaviour from all adapters.
**Likely failure modes:** leaked sockets under cancellation; a metadata cache serving a stale digest
after a model is re-pulled (mitigated by TTL plus an explicit `refresh=True` path).
**Gold standards:** clean public API; one client suite-wide; cancellable streaming; no leaks;
deterministic fake; ≥ 95 % coverage.
**Deferred:** llama.cpp and vLLM adapters, embeddings, async API, tokenization, multi-modal input.
