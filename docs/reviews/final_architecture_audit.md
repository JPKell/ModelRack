# Final Architecture Audit

**Date:** 2026-08-21 · **Scope:** the complete documentation set, audited as the specification three
separate engineering teams would implement from.
**State at audit:** architecture frozen, all nine repositories empty, no code, no persistent data, no
released contract. This is the last moment at which any of these findings is cheap.

**Outcome:** 41 findings — 4 Critical, 11 High, 21 Medium, 5 Low. All Critical and High findings and
17 of the 21 Medium findings are corrected in the documentation. Eight ADRs were added (0022–0029)
and seven amended (0003, 0006, 0007, 0008, 0009, 0011, 0017). No decision was reversed: every finding
was a boundary that had been left under-specified, not an architecture that was wrong.

---

## 1. What was audited, and how

Every document in `docs/` — 21 ADRs, 9 architecture documents, 13 standards, 3 application
specifications with their APIs, data models, plans and risk analyses, and 6 package specifications
with their plans.

The audit was run as seven passes:

1. **Deployment combinations** — each of the seven supported shapes traced end to end, looking for an
   optional integration that had become a requirement.
2. **Producer/consumer contracts** — both sides of every boundary read against each other, field by
   field.
3. **Expensive-to-change decisions** — identifiers, schemas, keys, formats and semantics that
   persistent data or a released contract would freeze.
4. **Development sequencing** — every phase checked for prerequisites it needs but nothing provides.
5. **Security and failure recovery** — untrusted inputs, exposure, interruption, recovery.
6. **Testability** — can each boundary be tested alone, and does any ordinary logic need a GPU.
7. **Operations and release** — configuration, migration, backup, versioning, distribution.

Findings were kept only where there is a concrete requirement violation, a contradiction between two
documents, a security or reliability concern, inappropriate coupling, a missing contract, an
implementation blocker, or a meaningful likelihood of expensive rework. Stylistic preferences were
discarded.

---

## 2. The pattern behind most of the findings

Nearly every serious finding is the same shape, and it is worth naming because it will recur during
implementation:

> **A documented invariant with no mechanism behind it.**

The specification asserts a property, an acceptance criterion promises it, a gold standard measures
it — and nothing in the design produces it. Ageing was policy with no sweep. Leases were renewed by a
heartbeat with no thread able to run it. The reproducibility fingerprint consumed a prompt hash from
a phase scheduled later. Freshness decayed from a timestamp neither side stored. `runtime_profile_hash`
was carried on every evidence record and every job, and read by nothing.

These survive review precisely because each document is individually coherent. They only surface when
two documents are read against each other, which is what this audit did.

The corresponding entry has been added to the risk register (A8), and the gold standards now require
each invariant to name the mechanism that provides it.

---

## 3. Critical findings

### C1 — Evidence freshness decays from a timestamp neither side agrees on

**Components:** FreeWeight (producer), LoadCoach (consumer), SetSpec (schema), ADR-0017.

FreeWeight's `capability_evidence` row carried `computed_at`. LoadCoach's carried `measured_at` and
`imported_at`. No document said which one ADR-0017's
`freshness_factor = 0.5 ** (age_days / half_life_days)` consumes, and no producer field maps to
`measured_at` at all.

Read as `computed_at` — the only value the producer actually stores — a nightly recomputation resets
the apparent age of four-month-old measurements, and confidence never decays. Read as `measured_at`,
the consumer receives a field the producer does not send. Either way the suite's headline claim,
*evidence with an expiry*, silently fails, and it fails as a plausible number rather than an error.

Compounding it: FreeWeight's uniqueness key included `policy_version` and LoadCoach's did not (but
included `source_id`), so a bundle carrying two policy versions could not be imported without
collision; `excluded_count` was mandatory on one side and absent on the other; and `capability.evidence`
had never been given a normative field list anywhere — SetSpec's specification described it in a
nine-word gloss.

**Correction.** [ADR-0022](../adr/0022-capability-evidence-record-contract.md) gives the payload a
normative field table, defines `measured_at` as the latest `completed_at` among contributing runs and
makes it the sole freshness input, puts `policy_version` in both uniqueness keys, and aligns the two
data models field for field. FreeWeight's plan gains the test that matters: *re-aggregating unchanged
runs does not raise confidence.*

### C2 — Routing consumes evidence for a subject it does not reproduce

**Components:** LoadCoach, BaseAiCore, ADR-0008, ADR-0017.

`MeasurementSubject` is `(identity, runtime_profile_hash, machine_fingerprint)`, and ADR-0017 makes a
differing runtime profile a **hard separation** — evidence measured under one profile does not
describe another. FreeWeight honours this. LoadCoach never resolved a runtime profile at all: it
selected a model, called the provider, and stored a `runtime_profile_hash` column nothing populated.

Two consequences, both silent:

* Evidence keyed by `runtime_profile_hash` could never match an execution. Either the hash is ignored
  (routing on evidence ADR-0017 says does not apply) or nothing ever matches.
* `min_context_tokens` was evaluated against `ModelDescriptor.max_context` — the **advertised**
  context. Under Ollama the served context is `num_ctx`, a runtime setting whose default is routinely
  a small fraction of the advertised maximum. LoadCoach would admit a model on an advertised
  131 072-token window, run it at the provider's default, and let the provider truncate the prompt.
  A confidently wrong answer, no error, no record.

**Correction.** [ADR-0023](../adr/0023-runtime-profile-resolution.md) makes the routing candidate an
**execution subject** — `(identity, resolved runtime profile)` — gives LoadCoach the
`runtime_profiles` table its ownership list had always claimed, and introduces `served_context` with
its source (`configured` / `reported` / `assumed`) as the input to every context constraint and every
KV estimate. A profile mismatch is named in the explanation with both hashes and the FreeWeight
invocation that would fix it, rather than silently reused or scored zero.

### C3 — The canonical model ID had three incompatible definitions

**Components:** BaseAiCore, all three applications, every stored measurement.

| Source | Format |
|---|---|
| ADR-0008 | `@{digest[:12]}` — which, since digests are `"sha256:<64 hex>"`, is `@sha256:1f3a` |
| Canonical Model Identity §2.1 | "the first 12 hex characters" — `@1f3a9c4e2b70` |
| Every worked example in the set | `@sha256:1f3a9c4e2b70` |

This string is an indexed column in three databases and a field in every cross-application payload.
Three implementers produce three incompatible lookup keys, and the mismatch surfaces as "LoadCoach
cannot find the model FreeWeight measured" only after both hold persistent data.

Separately, `GET /api/v1/models/{canonical_id}` used it as a path parameter. It contains `/`, `:` and
`@`; a percent-encoded `/` is normalized or rejected by common reverse proxies before reaching the
application, and Starlette will not match `/` in a path parameter without a converter that swallows
the rest of the route. Model names legitimately contain slashes (`hf.co/user/repo:q4`).

**Correction.** [ADR-0024](../adr/0024-canonical-id-and-model-references.md) fixes the format as
`"sha256:" + 12 hex` (matching every example, and keeping the algorithm visible), assigns digest
normalization to ModelRack via a new `baseaicore.normalize_digest`, and replaces the path parameter
with `{model_ref}` (a local ULID or prefix) plus `?canonical_id=` and exact-triple query forms.

### C4 — "Every cross-application payload is versioned" contradicted the APIs it governs

**Components:** SetSpec, all three applications, MirrorWall, API standards, observability standards.

Master Architecture §6 and gold standard G6 required a `schema_version` on every cross-application
payload. `POST /generate`'s body is a cross-application payload. So is `POST /jobs/{id}/feedback`,
for which no schema existed and none was planned — SetSpec's plan deferred `production.feedback` "to
LoadCoach's plan", where it never appears.

The documentation did not follow its own rule, and the drift showed: the audit found **three**
different on-the-wire shapes for one event (nested `payload` in the API standards, flat in the
observability standards, bare in LoadCoach's API) and **two** for errors (unwrapped in the API
standards, a listed SetSpec payload type in SetSpec).

**Correction.** [ADR-0025](../adr/0025-envelope-boundaries.md) answers the question that was never
asked — *what is the envelope for?* — and draws the line: the SetSpec envelope marks a **transferable
document** that outlives its request (exports, bundles, results, profiles, events); an API's own
request and response bodies are versioned by their path and contracted through the committed OpenAPI
snapshot. The membership test is mechanical. The event frame has one shape, with one documented
exception for `token` frames (a five-field envelope per token is ~100 bytes of overhead on the
hottest path in the suite). `production.feedback` is recorded as a deliberate rejection rather than
an oversight. Master Architecture §6 and G6 now say "transferable payload".

---

## 4. High findings

| # | Finding | Components | Correction |
|---|---|---|---|
| **H1** | Event and error envelope shapes contradicted themselves in three documents | API standards, observability standards, LoadCoach API, MirrorWall, SetSpec | [ADR-0025](../adr/0025-envelope-boundaries.md); observability §4.1 corrected to the nested form; the `token` exception documented once |
| **H2** | **Ageing never happens.** `effective_priority` is a stored column, ordered on by the claim query, and recomputed only at startup recovery. A long-running process ages nothing, so the starvation bound the gold standards promise does not hold | LoadCoach queue | [ADR-0029 §1](../adr/0029-queue-mechanics.md): an **ageing sweep** — one set-based `UPDATE` every 30 s on the scheduler thread. The stored column is kept because the claim query's index depends on it. The simulator now advances its clock with the process *up*, which is the only way this property is testable |
| **H3** | **The attempt counter has two writers.** The claim does `attempt = attempt + 1`; in-lease corrective retries also create `job_attempts` rows; `UNIQUE (job_id, attempt)` collides the first time a job retries in-lease and is later re-claimed. Separately, a claimed job failing admission had **no legal transition** (`leased → waiting_resources` did not exist), and neither did `leased → cancelling` | LoadCoach queue | [ADR-0029 §2–3](../adr/0029-queue-mechanics.md): the claim no longer touches `attempt`; the executor is the single writer; `admitted` becomes an explicit state and the transition table is completed and made normative |
| **H4** | **Blocking database I/O on the event loop.** SSE handlers are `async def` per ADR-0003; MirrorWall's `EventSource` port is synchronous and application-implemented over repositories. Every replay and every live batch was therefore a blocking `SELECT` on the event loop — the exact defect ADR-0003 forbids, in the one place ADR-0003 exempted from its own rule | ADR-0003, MirrorWall, all three applications | ADR-0003 amended with rules 7–8; `sse_response` owns the `anyio.to_thread.run_sync` dispatch so no application can get it wrong; an event-loop lag probe is now a MirrorWall test |
| **H5** | **Server-side request forgery.** `POST /evidence/import {"url": …}` made LoadCoach fetch an arbitrary URL. On the default loopback deployment that endpoint needs no credential, and `providers.allow_remote` does not govern it | LoadCoach, security standards | [ADR-0026 §3](../adr/0026-local-http-hardening.md): scheme, host-allowlist (loopback only by default), literal-IP, redirect and size checks before any parsing; `EVIDENCE_SOURCE_REFUSED` |
| **H6** | **No `Host` validation, and CSRF asserted without a mechanism.** A loopback-bound unauthenticated service is reachable from any page the user visits, via DNS rebinding. Security standards §1 claimed "CSRF protection on state-changing UI forms" with no mechanism named and no test in §14 | All three applications | [ADR-0026 §1–2](../adr/0026-local-http-hardening.md): `Host` allowlist enforced before routing and before auth (421 on mismatch), shared as MirrorWall middleware; double-submit token on form routes; the JSON API's exemption stated with its reasoning and its withdrawal condition; six new security tests |
| **H7** | **LoadCoach cannot authenticate to a FreeWeight that requires it.** `[evidence] freeweight_url` had no credential setting, so the LAN deployment the architecture documents was unreachable in practice | LoadCoach configuration | `freeweight_api_key_env` / `_file` added, resolving through the ordinary secret chain ([ADR-0026 §4](../adr/0026-local-http-hardening.md)) |
| **H8** | **Evidence for an undiscovered model had no defined behaviour.** `capability_evidence.model_id` was a non-null FK into a table only discovery writes, so importing before discovering either failed or dropped records — and importing twice would give different results | LoadCoach, FreeWeight API | [ADR-0022 §4](../adr/0022-capability-evidence-record-contract.md): `model_id` nullable, identity denormalized, `match_state` recorded, binding re-evaluated on every discovery pass, and the `name_only` ↔ digest matching rules stated in both directions |
| **H9** | **IdeaPress routed prose audits through `code.review`**, whose profile weights measured `code_review` capability at 0.45, applies `min_capability_scores = {code_review: 0.35}` as a hard constraint, and declares a code-review JSON schema with `required_fields = ["findings", "summary"]`. Separately, `fact_check` was bound in configuration and mapped to a task profile while appearing in no stage list | IdeaPress, LoadCoach | `content.review` added to LoadCoach's shipped profiles and to the canonical task-ID list; `fact_check` added as stage 10 with the pipeline renumbered; the stage table declared the single source of stage identifiers, with a startup check that bindings, `StageId` and the task map agree |
| **H10** | **Unclear whether LoadCoach injects its own prompts** into caller-supplied requests. Routing said a task profile is "not a prompt"; the development plan said the executor performs "prompt assembly from versioned prompt records". If LoadCoach rewrites the text, IdeaPress's per-attempt `prompt_sha256` provenance is a lie | LoadCoach, IdeaPress | LoadCoach spec §9 now states caller prompts pass through unmodified, names the only two records LoadCoach originates, and the plan adds a test asserting the transcript ModelRack received equals what the caller sent. `/generate` gains the `system` field the port needed |
| **H11** | **Multi-GPU claimed but not honoured.** SweatMeter reports per device; LoadCoach summed `free_vram` into one figure, `residency.vram_bytes` was one column, and FreeWeight's memory and energy metrics named no device. Summing admits a model that fits nowhere; a VRAM slope measured against the wrong device reads as zero bytes per token | SweatMeter, LoadCoach, FreeWeight | [ADR-0027](../adr/0027-multi-gpu-semantics.md): admission evaluates devices independently and never sums; `target_gpu_index` recorded; FreeWeight attributes to `execution.gpu_index` and skips memory/KV/energy with `multi_gpu_placement_unknown` when placement is unreported; the telemetry table split into host rows and per-device rows, removing a host-metric double-count |

---

## 5. Medium findings

Seventeen corrected, four deferred (§7).

| # | Finding | Correction |
|---|---|---|
| M1 | LoadCoach's `runtime_profiles` table was in its ownership list and in no schema; no `machines` table despite comparing machine fingerprints | Table defined; spec states LoadCoach knows exactly one machine, its own, from SweatMeter |
| M2 | Cancellation of a **non-streaming** execution was unspecified — "cancelled at the next stream boundary" has no boundary in `generate()` | LoadCoach always calls `stream()` and assembles internally; a non-streaming provider records `cancellation_deferred_to_completion` |
| M3 | `PROVIDER_REJECTED` and `GENERATION_CANCELLED` existed in ModelRack with no mapping in either consumer; ADR-0007 and ModelRack's spec listed **different** `ProviderCapabilities` fields | Full error-mapping table in LoadCoach §13; ADR-0007 amended to defer to ModelRack's dataclass |
| M4 | Upserts are needed in six places while ADR-0006 forbade `ON CONFLICT`; select-then-insert is a race | `weightsdb.upsert(...)` as the single sanctioned construct; ADR-0006 amended to forbid dialect-specific *variants*, not the construct |
| M5 | Collection envelope vs SetSpec envelope never composed — `GET /evidence` was both | [ADR-0025 §2](../adr/0025-envelope-boundaries.md): collections of SetSpec payloads nest envelopes in one order; a single document has no wrapper |
| M6 | `?since=` appeared in client guidance and no endpoint definition; compared two machines' clocks; could not express removals | Documented with a parameter table: filters `computed_at`, client echoes the previous bundle's `generated_at`, `complete: true\|false` governs removals |
| M7 | Token scope hierarchy unstated; `GET /version` auth status unstated, making a bad token and an incompatible API indistinguishable | Scopes declared cumulative; `/version` unauthenticated in all three applications |
| M8 | Prompt-pack machinery triplicated across three applications, with no extraction decision; the pack-level hash in the reproducibility fingerprint separated results for unrelated prompt edits, contradicting ADR-0017's per-benchmark rule | [ADR-0028](../adr/0028-prompt-pack-granularity.md): `prompt_subset_hash` per benchmark replaces the pack hash as the fingerprint input; machinery extracted to `setspec.prompts` at the second consumer |
| M9 | FreeWeight P6 assembles a fingerprint containing a prompt hash that P7 delivers; LoadCoach P3's VRAM constraint needs the estimate specified in P5 | Prompt library moved into FreeWeight P6; LoadCoach P3 gains the estimator as a pure function and P5 keeps the admission *policy* — both with sequencing notes explaining the split |
| M10 | "Migration failure restores the backup, original byte-identical" is unachievable on PostgreSQL: `pg_dump` round-trips are not byte-identical, restoring needs privileges the role deliberately lacks, and it cannot run under a live database | Guarantee scoped to SQLite; PostgreSQL behaviour specified honestly (transactional where DDL permits, otherwise refuse with the revision reached and the restore command); `MigrationOutcome` states which applied |
| M11 | Artifact files and their rows are not transactional; a crash between them orphans files with no reconciliation | Deferred — see §7 |
| M12 | `telemetry_samples` duplicated every host field per GPU row; `GpuSample.memory_utilization_percent` and `ram_total_bytes` had no column | Table split (ADR-0027 §4); missing fields added |
| M13 | The measurement value+reason convention was applied inconsistently (`residency.vram_bytes Measurement` as one column) | `measurement_columns` names fixed (`<name>` / `<name>_unavailable_reason`); a metadata test asserts the pairing |
| M14 | `jobs.source` had no population path; `idempotency_key` was globally unique with no expiry, so two callers could collide and a key was reserved forever | `source` derived from the token name or `X-Client-Name`; key scoped per caller with a TTL |
| M15 | `MeasurementSubject.is_comparable_with(other, *, metric_kind)` cannot answer two rows of its own matrix — benchmark version and dataset hash are not subject fields | Explicit optional arguments; omitting them yields `indeterminate`, never `comparable` |
| M16 | Cross-repository test artifacts had no distribution channel: consumers "test against the producer's OpenAPI document", MirrorWall "renders both applications' template suites" — with no stated mechanism that does not violate the import rules | OpenAPI snapshots and template suites ship as package data; consumers depend on the producer's distribution in their **`dev` extra** only, with `lint-imports` and the clean-venv check keeping it out of `src/` |
| M17 | FreeWeight's idle detection had no defined outcome on timeout — the unspecified third option was "proceed silently", which produces unexplained dispersion months later | `on_idle_timeout = "warn" \| "refuse"`; `warn` records `measured_while_busy` with the observed utilization |
| M18 | SetSpec's strict-outbound / preserving-inbound duality contradicted its own round-trip contract | Two model classes generated per payload type; the contract asserted per class |
| M19 | `NullHostReader` was named in the degradation path and not in the public API; `FakeTelemetrySource` was named in the testing standards and existed nowhere | Both made public; `sweatmeter.testing` specified; the three shipped doubles tabulated in the testing standards |
| M20 | Digest normalization (`sha256:` prefix, case, length) was required by ADR-0008 and assigned to nobody | Assigned to ModelRack via `baseaicore.normalize_digest`, with fixture tests |
| M21 | The lease heartbeat had no thread that could run it: a worker inside a 300 s blocking provider call cannot renew a 60 s lease | [ADR-0029 §4](../adr/0029-queue-mechanics.md): a lease keeper on the scheduler thread; the `lease_seconds > 3 × interval` relationship validated at startup |

---

## 6. Low findings

| # | Finding | Disposition |
|---|---|---|
| L1 | API resource-naming examples (`/api/v1/benchmark-runs`) contradicted the actual endpoints (`/runs`) | Corrected; the two singular exceptions documented |
| L2 | Application config examples showed `auto_migrate = true` flatly while the standard makes it dialect-dependent | Corrected; `config show` names which default applied |
| L3 | `parse_id(value) -> ULID` returned a type a zero-dependency package cannot have | `UlidParts`, a local frozen type |
| L4 | `--machine-fingerprint-override` appeared in one plan and no consumer's configuration | Recorded as a future extension with the reason it is not shipped: no consumer specifies it, so it would be a setting nothing reads |
| L5 | `SUPPORTED_SCHEMAS` as an exact-version tuple contradicts the accept-newer-minor policy | Changed to supported **majors** with the highest known minor |

---

## 7. Deliberately deferred

Each is recorded rather than fixed, with the reason and the trigger that would change it.

| Concern | Why deferred | Trigger |
|---|---|---|
| **Artifact/row transactionality (M11)** | Filesystem writes cannot join a database transaction, and the honest fix is a reconciliation sweep — real work with no user-visible benefit until artifacts actually orphan. The current design already writes the row last, so the failure mode is an orphaned *file*, not a dangling row | A `db status` reporting orphaned artifact bytes above a threshold; then a `<app> db gc-artifacts` command with the usual preview-confirm-transact treatment |
| **`fastapi>=0.115,<1` range width** | FastAPI is pre-1.0 and has broken on minor versions historically, so `<1` is optimistic. But the suite pins exact versions in CI lockfiles and runs a nightly compatibility matrix, which is the mechanism that would catch it | A FastAPI minor release breaking the matrix; then the range narrows to the tested minors |
| **Cross-device model sharding** | Estimating a runtime's layer-splitting policy is neither documented nor stable per provider, and the reference machine has one GPU, so it cannot be tested | A second GPU on a target machine, or a provider that reports placement (vLLM does) — recorded in ADR-0027's revisit trigger |
| **Application 1.0 depending on `0.x` packages** | Deliberate and stated in the roadmap; the consequence (every application needs a release to admit a package's 1.0) is now planned rather than discovered | Already scheduled into M9's checklist and the packaging standards' downgrade section |

Two things were reviewed and deliberately **not** changed, because the existing decision is sound:

* **The database-backed queue** (ADR-0010). The audit found four mechanism gaps inside it and closed
  them; none was an argument for a broker. The reasoning against Redis and Celery holds.
* **Nine repositories** (ADR-0001/0015). The audit's cross-repository test-artifact finding (M16) is
  the kind of friction ADR-0001 predicted and priced; it has a clean solution that does not require
  a monorepo.

---

## 8. ADRs added and amended

**Added**

| ADR | Title | Closes |
|---|---|---|
| [0022](../adr/0022-capability-evidence-record-contract.md) | Capability evidence record contract | C1, H8, M6 |
| [0023](../adr/0023-runtime-profile-resolution.md) | Runtime profile resolution and served context | C2 |
| [0024](../adr/0024-canonical-id-and-model-references.md) | Canonical ID format and model references in URLs | C3, M20 |
| [0025](../adr/0025-envelope-boundaries.md) | Envelope boundaries: what carries a SetSpec envelope | C4, H1, M5 |
| [0026](../adr/0026-local-http-hardening.md) | Local HTTP hardening: Host validation, CSRF and outbound fetch | H5, H6, H7, M7 |
| [0027](../adr/0027-multi-gpu-semantics.md) | Multi-GPU semantics | H11, M12 |
| [0028](../adr/0028-prompt-pack-granularity.md) | Prompt attribution granularity and shared prompt tooling | M8 |
| [0029](../adr/0029-queue-mechanics.md) | Queue mechanics: ageing, attempts, admission states and leases | H2, H3, M21 |

**Amended** (each carries an *Amended by* or *Amended* note at its head; none is superseded)

| ADR | Change |
|---|---|
| [0003](../adr/0003-sync-vs-async-strategy.md) | Rules 7–8: the SSE bridge dispatches its synchronous event-store reads to the threadpool (H4) |
| [0006](../adr/0006-sqlite-and-postgresql-roles.md) | Upserts permitted through one helper; the prohibition narrowed to dialect-specific variants (M4) |
| [0007](../adr/0007-provider-abstraction.md) | Capability list deferred to ModelRack's dataclass, which is normative (M3) |
| [0008](../adr/0008-canonical-model-identity.md) | Canonical string corrected; points to ADR-0024 (C3) |
| [0009](../adr/0009-setspec-schema-strategy.md) | Envelope scope narrowed to transferable payloads; strict/preserving model pair; supported majors (C4, M18) |
| [0011](../adr/0011-shared-package-boundaries.md) | Prompt tooling added to the extraction schedule at the same second-consumer trigger (M8) |
| [0017](../adr/0017-benchmark-confidence-and-freshness.md) | `age_days` measured from `measured_at`; prompt separation per benchmark; runtime-profile separation made operable (C1, C2, M8) |

---

## 9. Documents corrected

Architecture: `master-architecture`, `canonical-model-identity`, `machine-identity-and-reproducibility`,
`graceful-degradation`, `traceability-matrix`, `risk-register`, `executive-summary`.
Standards: `api-and-contract`, `observability`, `security`, `configuration`, `database`, `gold`,
`testing`, `prompt-management`, `cross-platform`, `packaging-and-release`.
Applications: all three specs, APIs, data models and development plans; IdeaPress's workflows;
FreeWeight's benchmark catalog; LoadCoach's routing and queue documents.
Packages: all six specifications; five development plans.
Plus `README.md`, `roadmap/master-roadmap.md` and `adr/README.md`.

---

## 10. Deployment combinations, re-verified after correction

| Combination | Verdict | What the audit checked |
|---|---|---|
| **FreeWeight alone** | Works | Six package dependencies, none of them an application; starts with no provider; the full test suite passes with no GPU, no Ollama, no network |
| **LoadCoach alone** | Works | Starts with no provider and no evidence; routes on declared capabilities and manual scores; `[evidence] freeweight_url = ""` reports `not_configured`, distinct from `unavailable` |
| **IdeaPress alone, direct provider** | Works | `sweatmeter` remains an optional extra (telemetry display only); no import of `freeweight` or `loadcoach` at any level; workflows complete against plain Ollama |
| **FreeWeight + LoadCoach** | Works, and now means something | Before correction, evidence could not be matched to an execution (C2) and its freshness was wrong (C1). Both closed; a profile mismatch is now visible and actionable rather than silent |
| **IdeaPress + LoadCoach** | Works | Audit stages no longer route through a code-review profile (H9); prompts pass through unmodified (H10); feedback is attributed per caller (M14); the task map is checked against the live profile list |
| **All three** | Works, with one honest limit | GPU contention between a benchmark and served inference is a measurement hazard, not a failure. LoadCoach defers on VRAM; FreeWeight's idle check now has a defined outcome (M17) instead of proceeding silently; `loadcoach queue pause` is the documented remedy and the limit is stated in the master architecture |
| **External application consuming packages** | Works | BaseAiCore, ModelRack and SweatMeter each install and function alone; the clean-venv install-and-import check is a blocking CI gate in every package repository |
| **External application consuming the LoadCoach API** | Works | `/version` needs no credential; the OpenAPI snapshot now ships as package data so a third party can generate a client or a mock (M16); scopes are cumulative and documented; no `LoadCoachClient` is required |

**Optional integrations verified as still optional.** No package imports an application. No
application imports another. No cross-application database access exists or is constructible. Every
peer connection is discovered at runtime and degrades to a named state. Two new failure surfaces
introduced by this audit — the evidence-source credential and the fetch allowlist — both default to
"not configured", which is the correct default for something optional.

---

## 11. Clean-room verification

The final pass: assume three teams receive only these documents, never speak to each other, and each
implements one repository. What would they still have to guess about, and where would two reasonable
guesses produce incompatible implementations?

### 11.1 Resolved by this audit

The audit's central finding is that this list used to be long, and most of it was invisible from
inside any single document. Ten items that would each have produced a genuine incompatibility are now
pinned by a normative statement and a golden or contract test:

1. The canonical ID string, byte for byte (C3).
2. The `capability.evidence` field set, both timestamps, and which one decays (C1).
3. Both sides' uniqueness keys, and what a re-import does (C1).
4. What a consumer does with evidence for a model it has not discovered (H8).
5. Whether evidence applies to an execution under a different runtime profile (C2).
6. Which context number a constraint means (C2).
7. The event frame shape, and whether an error body is enveloped (C4, H1).
8. Whether an API request body needs a SetSpec envelope (C4).
9. Whether LoadCoach may modify a caller's prompt (H10).
10. Whether a prompt hash is per pack or per benchmark (M8).

### 11.2 What a team would still have to decide

These remain open, and each is deliberately open — an implementation detail a team may settle
locally, because no other repository can observe the choice. They are listed so that nobody mistakes
silence for an omission.

| Open question | Why it is safe to leave open |
|---|---|
| Internal module layout below `domain/`, `services/`, `infrastructure/` | Not observable across a repository boundary; the layering contract and the import-linter rules bound it |
| The scoring functions' internal shape (pure functions vs a strategy class) | The `RoutingStrategy` port and the golden-decision test fix the *behaviour*; the shape is local |
| Cursor encoding for pagination | Documented as opaque base64 of a stable sort key; clients never construct one, so the encoding is private |
| The exact SQL of the ageing sweep and the claim, per dialect | The semantics, the index and the query-plan assertions are normative; the statement is not |
| Jinja macro internals and CSS class names | MirrorWall's contract is explicitly the rendered markup and the `data-` attribute API |
| Fake-provider script format | `modelrack.testing` is a supported API, but only its behaviour is contracted; the declarative format is ModelRack's to shape |
| Benchmark fixture *content* | Manifests, hashes, metric keys and scorer semantics are contracted; the corpus itself is FreeWeight's and is deliberately unpublished (contamination) |

### 11.3 Three residual risks a clean-room build should watch

Not defects — places where the documents are correct but a team could still drift, and where the
suite's own tests are the intended safety net rather than the prose.

1. **`FakeProvider` faithfulness.** Every application's default test suite runs against it, so a fake
   that is more forgiving than Ollama hides a real integration bug. The mitigation exists (the
   conformance suite runs against fake, recorded and live adapters; the nasty cases are scripted
   explicitly) but it depends on the ModelRack team scripting failures they have not yet met. The
   nightly live job is what closes the gap, and it should be treated as load-bearing rather than
   optional.

2. **Golden-value tests as the real contract.** Canonical IDs, machine fingerprints, profile hashes,
   canonical JSON and prompt hashes are now fixed by goldens. A golden test that fails is
   indistinguishable, at the terminal, from a golden test that needs updating. Every one of these
   carries a comment saying so; the discipline of never "updating a golden to make CI green" is the
   single habit on which cross-repository compatibility rests.

3. **The documentation set lives in its own repository.** ADR-0015 accepted this and mitigated it by
   generating configuration references and OpenAPI snapshots into the component repositories. That
   mitigation now carries more weight, because this audit put normative field tables into ADRs. The
   milestone-repeated consistency review (roadmap §8) is the mechanism, and it should explicitly
   include: the evidence field set against both data models, the task profile list against
   `LOADCOACH_TASK_MAP`, the stage list against `[models.stages]`, and the error-code tables against
   each application's implementation.

### 11.4 Verdict

With the corrections in this audit applied, three teams working only from these documents would
produce implementations that interoperate. The contracts that cross repositories — model identity,
capability evidence, the event and error envelopes, the LoadCoach API, prompt hashing — are each
specified to the field, versioned, and testable from one side alone using artifacts the other side
publishes.

What remains genuinely unspecified is, in every case, something no other repository can observe.
