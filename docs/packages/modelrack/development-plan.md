# ModelRack — Development Plan

**Sequence position:** third component. Depends on BaseAiCore Phase 4.
**Target:** `modelrack 0.5.0` by the end of Phase 5. **Reached 2026-08-26** — all five phases complete.
Phases 6–8 are the adapter arc's LA1 checkpoint
([adapter roadmap §4.1](../../roadmap/adapter-roadmap.md)), targeting the next minor (`0.7.0`).

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
  one ([ADR-0024 §2](../../adr/0024-canonical-id-and-model-references.md)).
* Timing extraction: `load_duration`, `prompt_eval_duration`, `eval_duration`, `total_duration`
  mapped to `Timing.backend_*`; client-observed wall time and TTFT measured with
  `time.perf_counter_ns()` into `Timing.client_*`.
* Error translation for every row of the spec's §13 table.
* Capability declaration for Ollama, including `token_level_chunks` and `context_configurable` set
  truthfully — the latter is what tells a caller whether it may set a served context or must record
  one as assumed ([ADR-0023 §4](../../adr/0023-runtime-profile-resolution.md)).
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
**Gold standards:** one Ollama client for the whole suite; unsupported-safe normalization; typed
errors.
**Deferred:** OpenAI-compatible adapter; llama.cpp; vLLM.

---

## Phase 4 — OpenAICompatibleProvider and capability honesty

**Goal:** a second real adapter proves the abstraction is not Ollama-shaped, and capability
declarations become load-bearing.

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
docs/providers.md                 # capability matrix across adapters
```

**Tests**
* Conformance suite passes for this adapter, with capability-gated tests skipped **explicitly**
  (recorded as `unsupported`, never silently passed).
* Identity confidence is `name_only`; consumers can detect it.
* SSE parsing: multi-line data, `[DONE]`, keep-alive comments, malformed frames.
* API key sent in the header and absent from `raw`, `details` and DEBUG logs.

**Acceptance criteria**
1. Two real adapters pass one conformance suite.
2. The capability matrix in `docs/providers.md` is generated from the adapters' declarations, not
   hand-written.
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
docs/{quickstart.md,providers.md}
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
**Deferred:** the llama.cpp adapter (Phase 6), the vLLM adapter, embeddings, async API,
tokenization, multi-modal input.

---

## Phase 6 — LlamaCppProvider: process supervision and basic serving

**Goal:** a third real adapter serves GGUF files directly by spawning and supervising
`llama-server`, with digest-bound identities — the start of the adapter arc's LA1
([adapter roadmap §4.1](../../roadmap/adapter-roadmap.md), P6;
[ADR-0062](../../adr/0062-llamacpp-serves-adapters-through-a-supervised-process.md)).

**Prerequisites:** Phase 5; the ADR-0070 usage rule (row C5) landed first, so the third adapter is
written to it rather than retrofitted.

**Work**
* `providers/llamacpp.py`: `LlamaCppProvider` — spawn / health-wait / terminate `llama-server`
  through `load()`, `unload()` and `list_resident()`, the `Provider` protocol unchanged.
* GGUF discovery and hashing from a configured model directory: header parsing for descriptors;
  the sha256 of the served artifact as the identity (identity confidence *bound*, better than
  tag-based); the digest keyed by path and file stamp, persisted in `<state_dir>/digests.json`
  ([ADR-0071](../../adr/0071-modelrack-persists-artifact-digests-in-a-json-file-the-application-names.md))
  with an explicit `refresh=True` path and a `clear_digest_cache()`, never substituted by a
  cheaper hash.
* Profile flags: `n_gpu_layers`, KV cache types, flash attention, context size, threads, batch
  size; a chat-template override through `provider_options`.
* Generation and streaming over the server's native API — `/completion` for prompts, the chat
  endpoint with llama.cpp's extensions (`timings`, `system_fingerprint`, `reasoning_content`) for
  messages.
* Usage read to ADR-0070's per-response rule from the start: cached input from `timings.cache_n`
  reconciled into disjoint classes; a class the native API cannot bill is `0`, never
  `UNSUPPORTED`; `tokens_cached` — the slot's whole cache — is never read as cached input.
* Error translation for every row of the spec's §13 table; a startup failure carries the captured
  stderr, a context overflow carries the server's own token counts.
* Recorded fixtures, version-annotated with the llama.cpp build they represent.
* The three named risks, each with its mitigation: orphaned processes (kill-tree on timeout, pid
  files, a finalizer on the adapter), port management (a configured range), startup-failure
  diagnosis (stderr captured into the typed error).

**Files/subsystems**
```text
src/modelrack/providers/llamacpp.py            # the adapter
src/modelrack/providers/_llamacpp_process.py   # the spawn seam: launcher, process table, supervisor
src/modelrack/providers/_llamacpp_wire.py      # pure translation of both wire shapes
src/modelrack/providers/_gguf.py               # header reading and content hashing
src/modelrack/providers/_openai_wire.py        # chat-completions helpers shared with Phase 4
tests/unit/{test_llamacpp_adapter,test_llamacpp_supervisor,test_llamacpp_wire,test_gguf}.py
tests/fixtures/providers/llamacpp/*
tests/live/test_llamacpp_live.py               # marked
```

**Tests**
* Supervision through the injected seam: spawn, health wait, exit during startup with stderr,
  startup timeout with kill-tree, port allocation and exhaustion, orphan recovery from pid files
  (foreign owner untouched, stale file removed, reused pid spared, orphan killed) — and the
  default launcher and process table proven against ordinary shell processes.
* GGUF: every value type, large arrays summarised, parameter count from tensor records, every
  refusal; the content digest against `hashlib`.
* Discovery: only base models listed; shards, adapters, projectors and malformed files skipped;
  a digest computed once per file and recomputed on `refresh=True` or a changed stamp.
* Residency: a profile becomes launch flags; a differing profile restarts; two bases on two
  ports; a crashed server reported once, typed, then respawned; a pinned digest that no longer
  matches is refused.
* Generation and streaming on both shapes: the three ADR-0070 cases, `tokens_cached` ignored,
  reasoning, tool calls, cancellation within one event, truncation, in-band errors.
* Conformance suite passes against the recorded transport and the fake launcher, with the
  cache-detail case declared.
* Live (marked): real headers and one real digest from the reference machine's directory; a real
  `llama-server` journey — discovery, load, generate, stream, unload, no process left behind.

**Acceptance criteria**
1. The default suite passes with **no `llama-server` installed** and no GGUF file larger than a
   test writes.
2. Every identity from this adapter carries the sha256 of the served file; a pinned digest that
   does not match is a typed refusal.
3. No code path leaves a process behind: a startup timeout kills, `unload()` kills, a dropped
   adapter kills, and a server orphaned by a crashed supervisor is recovered from its pid file.
4. A startup failure surfaces as a typed error carrying the exit code and the captured stderr.
5. The conformance suite passes for all four adapters (fake, Ollama, OpenAI-compatible,
   llama.cpp) with the three usage cases declared.
6. Coverage ≥ 95 %.

**Known risks:** orphaned processes, port collisions, startup failures nobody can diagnose — each
named in the adapter roadmap and each mitigated above. The live journey cannot run on a machine
without llama.cpp; it is an operator step on LA1's critical path.
**Likely failure modes:** reading `tokens_cached` as cached input (it is the whole cache);
hashing 40 GB on every discovery call (mitigated by the stamp-keyed, persisted digest file —
ADR-0071); a profile served by a server launched under
different flags (mitigated by comparing launch flags, not profile hashes).
**Gold standards:** one client for the whole suite; typed errors; unsupported-safe measurement;
no leaked processes; digest-bound identity.
**Deferred:** adapters (Phase 7 — registration from manifests, per-request selection,
`adapter_hot_swap`, `AdapterNotFound`, `pending_restart`, the cache-correctness conformance
test); cancellation under supervision, leak tests and publication (Phase 8); the vLLM adapter,
embeddings, async API, tokenization, multi-modal input.
