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
**Deferred:** adapters ([Phase 7](#phase-7--adapters-registration-selection-and-cache-correctness)
— registration from manifests, per-request selection, `adapter_hot_swap`, `AdapterNotFound`,
`pending_restart`, the cache-correctness conformance test); cancellation under supervision, leak
tests and publication ([Phase 8](#phase-8--hardening-and-publication-070)); the vLLM adapter,
embeddings, async API, tokenization, multi-modal input.

---

## Phase 7 — Adapters: registration, selection, and cache correctness

**Goal:** one warm base serves several LoRA adapters, selected per request, with **zero base loads
between them** — and with the identity, the refusals and the cache isolation that make the
resulting measurements trustworthy (adapter roadmap §4.1, P7;
[ADR-0058](../../adr/0058-the-execution-subject-gains-an-adapter-axis.md),
[ADR-0062](../../adr/0062-llamacpp-serves-adapters-through-a-supervised-process.md),
[ADR-0063](../../adr/0063-one-adapter-at-a-time.md)). The phase's centre of gravity is not the
feature — it is **I17**, the conformance test that proves a prompt prefix computed under adapter A
is never reused for a request running under B.

**Prerequisites:** Phase 6 (the supervised process and its injection seams);
`baseaicore>=0.4.1`, which is where `AdapterIdentity` and `verify_adapter_base_compatibility`
live. Adapter identity is **not** redefined here.

### The boundary decision this phase settles

**ModelRack does not import `setspec`, and does not read the adapter directory.**
[ADR-0061](../../adr/0061-the-adapter-registry-is-a-directory-and-a-manifest.md) rule 3 already
says so — FreeWeight and LoadCoach read the directory; ModelRack "receives manifests from the
application constructing it, and validates and mounts them" — and
[master architecture §2](../../architecture/master-architecture.md) permits only `mirrorwall` and
`commissioner` to depend on `setspec`. So this package defines **its own** frozen value object,
`AdapterRegistration`, carrying exactly the fields serving needs, and the **application** converts
`setspec.artifacts.AdapterManifestOut` into it. ModelRack's runtime dependency set stays
`baseaicore` + `httpx`, and `.importlinter` is not edited. The conversion LoadCoach 1.1 must
perform is field-for-field and is recorded in the handoff.

**Work**
* `adapters.py`: `AdapterRegistration` — `name`, `artifact_path`, `artifact_sha256`,
  `source_sha256`, `base_model_name`, `base_artifact_digest`, `data_classification`,
  `adapter_format`. Its `identity` property returns a `baseaicore.AdapterIdentity`; the digests are
  normalized through `baseaicore.normalize_digest` (spec §11.9) and a digest that will not
  normalize is a refusal, because an adapter with no artifact digest has no identity to be.
  `AdapterStatus` (`registered` · `pending_restart` · `incompatible` · `awaiting_base`) and
  `AdapterState`, the caller-visible answer to "why is my adapter not being used?".
* `ProviderCapabilities.adapter_hot_swap` — the fourteenth flag, defaulting `False` like every
  other. `LlamaCppProvider` declares `True`; Ollama, the OpenAI-compatible adapter and the fake
  declare `False`. **Load-bearing, exactly like `context_configurable`**: it is also what makes
  [ADR-0065](../../adr/0065-an-adapter-is-classified-and-local-only.md)'s local-only invariant hold
  by construction, since the only adapter declaring it is the one that supervises a local process.
* `Provider.list_adapters()` and `Provider.register_adapters()` on the protocol, both raising
  `CapabilityUnsupported` where the flag is `False`, in the shape `list_resident`/`unload` already
  use. On the protocol rather than on the concrete class so LoadCoach can show adapter rows and
  fold in a rescan **without importing `LlamaCppProvider`** — the same argument that put `refresh`
  on the protocol (spec §10). ADR-0062 decision 1's "the `Provider` protocol does not change" was
  about the load/unload seam, and that remains true: no lifecycle method was added.
* `GenerationRequest.adapter: str | None` — the registered **name**, `model`-pin semantics
  (adapter roadmap §4.2). `GenerationResult.adapter` and `.adapter_base_confidence` carry back the
  `AdapterIdentity` that actually ran and the confidence of its base claim, so a stored result
  names its whole subject rather than its base.
* Launch-time registration: at spawn, every registration whose declared base **digest** matches the
  served base's, or whose declared base **name** matches when it declares no digest, is a candidate;
  `verify_adapter_base_compatibility` then decides. `DIGEST` and `NAME_ONLY` are registered via
  `--lora` with `--lora-init-without-apply`; a raised `ValidationError` is a **recorded refusal**
  (`incompatible`, with both digests in the reason), never a silent drop. Matching by digest first
  is what makes a renamed base safe (ADR-0061 rule 5). The server's own `GET /lora-adapters` is
  read back after health, so the ids sent per request are the server's, not an assumption about
  argv order.
* Per-request selection: at most one adapter, at scale `1.0` (ADR-0063). The body carries the
  **complete** `lora` list — the selected adapter at `1.0` and every other registered adapter
  explicitly at `0.0` — because llama-server treats an *absent* `lora` field as "restore the
  launch-time set" and does not clear the slot's prompt cache when it does (see Tests). A request
  naming two adapters is unrepresentable in the field and refused where it is representable: the
  `lora` and `--lora*` escape hatches through `RuntimeProfile.provider_options` are refused, since
  an adapter selected there would change the weights without changing the recorded subject.
* `pending_restart` and the **in-flight guard**: a registration handed over after its base's server
  was launched is `pending_restart` and folds in at the next natural idle — defined as the next
  `_ensure_server` at which no request is in flight against that server, or an `unload()`. In
  flight is a per-server counter, held from just before the request is sent until a non-streamed
  result returns or a stream's iterator is exhausted, abandoned or closed. A request naming a
  `pending_restart` adapter while work is in flight is `ProviderUnavailable` with reason
  `restart_pending` — availability, not reliability
  ([ADR-0067](../../adr/0067-reliability-keys-on-the-subject-not-the-base.md) rule 2) — never a
  silent bare-base generation. The guard also covers Phase 6's profile-change restart, which
  today terminates a server other work may be streaming from.
* `AdapterNotFound` in `modelrack.errors`, `details` carrying `adapter`, the `registered` set and
  a `reason` (`unknown` or `incompatible_base`, the latter with both digests). A **permanent**
  fact about the request, which is what separates it from `restart_pending`.

**Files/subsystems**
```text
src/modelrack/adapters.py                      # AdapterRegistration, AdapterStatus, AdapterState
src/modelrack/provider.py                      # +adapter_hot_swap, +list_adapters/register_adapters
src/modelrack/errors.py                        # +AdapterNotFound, +RESTART_PENDING
src/modelrack/types.py                         # +GenerationRequest.adapter, +GenerationResult.adapter
src/modelrack/providers/llamacpp.py            # registration, selection, the in-flight guard
src/modelrack/providers/_llamacpp_wire.py      # --lora argv, the lora body field, /lora-adapters
tests/unit/test_adapters.py                    # the value object and its refusals
tests/unit/test_llamacpp_adapters.py           # registration, selection, pending_restart
tests/contract/test_adapter_isolation.py       # I17, structural half
tests/live/test_llamacpp_live.py               # I17, semantic canary (marked)
```

**Tests**
* **I17, structural half — the phase's reason for existing.** A reusable checker over recorded
  request bodies asserts four things for every exchange: the `lora` field is present **iff** the
  server has adapters registered; exactly one entry carries a non-zero scale **iff** the request
  named an adapter, and it is that adapter's server id at `1.0`; every other registered adapter
  appears explicitly at `0.0`; and no slot-pinning key (`id_slot`, `slot_id`) appears, since a
  pinned slot is the one lever that could present A's cached prefix to a B request past
  llama-server's own `lora_should_clear_cache`. Driven over a randomized alternating sequence of
  A / B / bare requests across both endpoints and both streaming modes. The checker is then run
  against **three injected defects** — the selection dropped, the previous request's selection
  left in place, and a slot pin added — and asserted to fail on each: a conformance property that
  cannot fail proves nothing.
* The **no-cross-adapter-batching** note, recorded rather than assumed: llama-server's
  `server_slot::can_batch_with` includes `are_lora_equal`, so two slots with different adapter
  sets are never in one decode batch (`tools/server/server-context.cpp`, build `b10792`). Restated
  in the spec so it stays true if this suite's single-user concurrency ever changes.
* **A-1's invariant, as a golden, not a claim.** A request naming no adapter against a provider
  with no registrations produces a body byte-for-byte identical to Phase 6's, and its result's
  `adapter` is `None`; the existing goldens pass untouched.
* Registration: a matching digest registers with `DIGEST`; a declared-digest mismatch on the same
  base name is refused with both digests in the reason and is **not** on the argv; a manifest with
  no digest and a matching name registers with `NAME_ONLY` and that confidence reaches
  `AdapterState` and every `GenerationResult` it produces; an adapter for another base is
  `awaiting_base` and registers when that base is launched; a base renamed with the digest intact
  still registers.
* Selection: an unknown name raises `AdapterNotFound` naming the registered set; an incompatible
  one raises it with `incompatible_base` and both digests; `provider_options["lora"]` and a
  `--lora` launch flag are refused; a provider declaring `adapter_hot_swap = False` raises
  `CapabilityUnsupported` naming the flag, for all three of the other adapters.
* `pending_restart`: a registration added while a server runs is `pending_restart` and does not
  reach the running server; the next request at idle restarts once and folds it in; a request
  naming it **while a stream is in flight** raises `restart_pending` and the stream survives —
  the race forced by holding a partially drained stream open on the injected transport, not by
  timing. Supervision survives: no orphan, pid files swept, kill-tree intact, asserted through
  the injected `ProcessTable`.
* Conformance: `test_adapter_selection_is_honoured_or_refused` joins the capability-gated block,
  so all four adapters answer for the flag they declare.
* Live (marked): the semantic canary — the same prompt under two adapters, distinct continuations
  after a shared prefix, with `timings.cache_n` showing the prefix **not** reused across the
  switch. It needs two real adapter GGUFs for one base; where the machine has none it **skips
  visibly**, naming the artefacts it needs, and never passes vacuously.

**Acceptance criteria**
1. A subject naming no adapter is byte-for-byte what it was in Phase 6 — request body, result and
   canonical form — asserted over the existing goldens.
2. `adapter_hot_swap` exists, defaults `False`, and is `True` for `LlamaCppProvider` alone; the
   generated capability matrix carries its row.
3. An incompatible base is refused **by digest**, fail closed, with both digests in the reason; a
   `name_only` match is admitted and its reduced confidence reaches every surface that names the
   subject.
4. Two adapters in one request cannot be expressed and cannot be smuggled through
   `provider_options`; an unknown adapter raises `AdapterNotFound` naming the registered set.
5. A newly registered adapter reports `pending_restart`, does not fold in mid-work, and folds in at
   the next idle; the in-flight guard is proved by a test that forces the race.
6. **I17 passes**: the structural property runs in the default gate with no binary and fails on all
   three injected defects; the semantic canary passes live or skips visibly with its artefacts
   named.
7. `lint-imports` green with `.importlinter` unedited; the runtime dependency set is still exactly
   `baseaicore` + `httpx`.
8. Coverage ≥ 95 %.

**Known risks:** the correctness this phase adds fails **silently** — an adapter applied to the
wrong base, or a prefix reused across a switch, produces plausible text and no error, which is why
every rule here is a refusal and why I17 is tested against injected defects rather than only
against the happy path. llama-server's `--slot-save-path` enables a prompt-cache restore path that
is not adapter-aware; ModelRack never sets it and never calls `/slots`, but an operator passing it
through `provider_options` would open a lever this phase's checker does not cover — stated so it is
a known hole rather than a surprise.
**Likely failure modes:** relying on llama-server to zero unnamed adapters instead of sending them
(an absent `lora` field restores the *launch* scales — `--lora` defaults to `1.0` and
`--lora-init-without-apply` does not change that — so a bare-base request to an adapter-registered
server would run with **every** adapter applied, and with the previous request's cache); putting
the adapter set into the launch key, which would make every rescan restart a running server and
defeat `pending_restart`; releasing the in-flight count on the streaming path only when the
iterator completes, so an abandoned stream leaves a server permanently un-restartable.
**Gold standards:** identity is never redefined, only used; a refusal beats a plausible answer;
capabilities are declarations callers branch on; a conformance property is proved by making it
fail.
**Deferred:** cancellation under supervision, leak tests, the four-adapter conformance run and
publication (Phase 8); composition, scale-mixing and per-request scale (ADR-0063 — a new ADR, not a
flag); reading the adapter directory (ADR-0061 rule 3 — the application's job); routing on adapter
evidence (LoadCoach 1.1, LA2) and adapter subject enumeration (FreeWeight 1.1, LA3).

---

## Phase 8 — Hardening and publication: `0.7.0`

**Goal:** the adapter arc's LA1 checkpoint closed and shipped — cancellation proved under process
supervision, no process or memory leaked across repeated cycles, the conformance suite green for
all four adapters, and `modelrack 0.7.0` on PyPI carrying Phases 6 and 7 together
(adapter roadmap §4.1, P8; §3's LA1 exit condition).

**Prerequisites:** Phase 7.

**Work**
* Cancellation under supervision: a token fired mid-stream against a spawned server closes the
  connection within one chunk boundary, leaves the server running and usable, and never leaves the
  in-flight count raised — the guard Phase 7 added is what makes the second half testable.
* Leak tests: twenty `load`/`unload` cycles leave no orphan, no pid file and no handle, asserted
  through the injected `ProcessTable`; the flat-memory half needs the real launcher and a real
  server and is an operator step, run once and recorded.
* The conformance suite green for all four adapters with every capability-gated skip explicit,
  including `adapter_hot_swap`.
* Sharded GGUFs: decide whether a sharded base is served or refused with a named reason, rather
  than skipped at discovery with a debug log (D3 finding 6).
* **[ADR-0074](../../adr/0074-adapter-enabled-serving-is-a-runtime-profile-field.md)'s mechanism,
  before publication.** Phase 7 found that ADR-0060's "serving mode lives in the runtime profile"
  had nothing behind it: `--lora` flags come from the registration set, `RuntimeProfile` is a
  separate object, and a base on an adapter-registered server hashes identically to the same base
  on a clean one. Two pieces: BaseAiCore gains
  `RuntimeProfile.adapters_registered: bool | None = None` (tri-state, because a `False` default
  would be hashed and move every stored `profile_hash`; `None` is dropped from the hash already),
  and this adapter **refuses** a request whose profile disagrees with the server it would use —
  the `context_configurable` discipline of ADR-0023 §4 one level up. It lands here because 0.7.0
  publishing without it ships a known silent-merge gap, and because evidence recorded before the
  field exists cannot be separated afterwards.
* `docs/providers.md` regenerated; the spec's §7 surface, §10 data ownership and §13 error table
  final; the tested llama.cpp build pinned in the documentation (ADR-0062's consequence).
* Version to `0.7.0`, `CHANGELOG.md` moved out of `## [Unreleased]`, tag, publish, verify the
  installed wheel imports and the demos run against it.

**Tests**
* Cancellation: mid-stream against a fake launcher and a recorded transport; the server survives,
  the next request succeeds, the in-flight count returns to zero.
* Leaks: the twenty-cycle assertion above; an abandoned stream and a dropped provider both release
  every handle.
* The full conformance suite across fake (three configurations), Ollama, OpenAI-compatible and
  llama.cpp.

**Acceptance criteria**
1. Twenty load/unload cycles leave nothing behind — no process, no pid file, no handle.
2. A cancelled stream leaves a usable server and a zeroed in-flight count.
3. The conformance suite passes for all four adapters, every skip declared.
4. A profile that misdescribes the server is refused, and a profile that does not mention
   `adapters_registered` hashes exactly as it did before the field existed — asserted over an
   existing golden, not claimed (ADR-0074).
5. **The LA1 exit demonstration** on the reference machine: one llama-server base, three registered
   adapters, twenty generations alternating adapters, **zero base loads** — asserted from the
   process table and from load timings, not from absence of complaint (I16).
6. `modelrack 0.7.0` on PyPI; the demos run against the published wheel.
7. Coverage ≥ 95 %.

**Known risks:** the LA1 exit needs a GPU session with real adapter artefacts, which is an operator
step on the critical path and cannot be faked; flat memory across twenty cycles is the one
assertion the injected launcher cannot make.
**Likely failure modes:** declaring the leak test green from the fake alone; publishing before the
live journey has been re-run against the shipped wheel.
**Gold standards:** no leaked processes; every skip explicit; a published package that installs
clean on a machine with no GPU.
**Deferred:** the vLLM adapter, embeddings, an async API, tokenization, multi-modal input.
