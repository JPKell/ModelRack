# Requirement Traceability Matrix

Maps every major suite requirement to the component that **owns** it, the components that consume it,
and the contract through which it is exposed. Its purpose is to make ownership gaps, duplicated
responsibility, circular dependencies and vague contracts visible — and then to record how each was
resolved.

**Reading the "Contract" column:**
`Python API` = an importable interface · `SetSpec` = a versioned cross-application payload ·
`HTTP v1` = a versioned public HTTP API · `Internal` = owned entirely inside one component ·
`Config` = a documented configuration surface.

---

## 1. Core domain requirements

| Requirement | Owner | Consumers | Contract | Specification |
|---|---|---|---|---|
| Canonical model identity | **BaseAiCore** | All | Python API + SetSpec `model.identity` | [Canonical Model Identity](canonical-model-identity.md) · [ADR-0008](../adr/0008-canonical-model-identity.md) |
| Model descriptive metadata | **BaseAiCore** (type) / **ModelRack** (population) | All | Python API | [BaseAiCore](../packages/baseaicore/spec.md) §7 |
| Runtime profile and its hash | **BaseAiCore** | FreeWeight, LoadCoach | Python API | [Canonical Model Identity §4](canonical-model-identity.md) |
| Measurement subject and comparability | **BaseAiCore** | FreeWeight, LoadCoach | Python API | [Canonical Model Identity §5](canonical-model-identity.md) |
| Machine identity and fingerprint | **BaseAiCore** (type + hash) / **SweatMeter** (collection) | All | Python API + SetSpec `machine.profile` | [Machine Identity](machine-identity-and-reproducibility.md) |
| "Unavailable is not zero" | **BaseAiCore** | All | Python API (`Unsupported`) | [ADR-0016](../adr/0016-unavailable-is-not-zero.md) |
| Capability identifier type | **BaseAiCore** | All | Python API | [BaseAiCore §7](../packages/baseaicore/spec.md) |
| Capability **vocabulary** and its version | **SetSpec** | FreeWeight, LoadCoach, IdeaPress | SetSpec | [Master Architecture §1.4](master-architecture.md) |
| IDs (ULID) and timezone-aware timestamps | **BaseAiCore** | All | Python API | [BaseAiCore §7](../packages/baseaicore/spec.md) |
| Canonical JSON and hashing | **BaseAiCore** | All | Python API | [BaseAiCore §7](../packages/baseaicore/spec.md) |

## 2. Infrastructure requirements

| Requirement | Owner | Consumers | Contract | Specification |
|---|---|---|---|---|
| Provider abstraction | **ModelRack** | FreeWeight, LoadCoach, IdeaPress | Python API | [ADR-0007](../adr/0007-provider-abstraction.md) |
| Ollama inference | **ModelRack** | FreeWeight, LoadCoach, IdeaPress | Python API | [ModelRack §7](../packages/modelrack/spec.md) |
| OpenAI-compatible inference | **ModelRack** | IdeaPress, LoadCoach | Python API | [ModelRack §7](../packages/modelrack/spec.md) |
| Deterministic fake provider | **ModelRack** (`modelrack.testing`) | All test suites | Python API | [ModelRack Phase 2](../packages/modelrack/development-plan.md) |
| Provider capability declaration | **ModelRack** | FreeWeight, LoadCoach, IdeaPress | Python API | [ModelRack §7](../packages/modelrack/spec.md) |
| Streaming and cancellation | **ModelRack** (provider) / **MirrorWall** (transport) | All | Python API + SSE | [ADR-0004](../adr/0004-sse-vs-websockets.md) |
| Hardware telemetry | **SweatMeter** | FreeWeight, LoadCoach, IdeaPress (optional) | Python API | [SweatMeter §7](../packages/sweatmeter/spec.md) |
| Static machine profiling | **SweatMeter** | FreeWeight, LoadCoach | Python API | [SweatMeter §7](../packages/sweatmeter/spec.md) |
| Energy estimation | **SweatMeter** (integration) / **FreeWeight** (metrics) | FreeWeight | Python API | [SweatMeter §7](../packages/sweatmeter/spec.md) · [Benchmark Catalog §3.14](../apps/freeweight/benchmark-catalog.md) |
| Database engine, sessions, pragmas | **WeightsDB** | All three applications | Python API | [ADR-0005](../adr/0005-database-strategy.md) |
| Migrations, backup, restore, DB health | **WeightsDB** | All three applications | Python API | [Database Standards](../standards/database-standards.md) |
| Design tokens and UI components | **MirrorWall** | All three applications | Jinja macros + CSS/JS | [ADR-0020](../adr/0020-ui-rendering-strategy.md) |
| JSON/error envelopes, request IDs | **MirrorWall** (helpers) / **SetSpec** (shapes) | All three applications | Python API + SetSpec | [API Standards §4](../standards/api-and-contract-standards.md) |
| SSE transport with replay | **MirrorWall** | All three applications | HTTP v1 (SSE) | [MirrorWall §7](../packages/mirrorwall/spec.md) |

## 3. Contract requirements

| Requirement | Owner | Consumers | Contract | Specification |
|---|---|---|---|---|
| Schema envelope and version negotiation | **SetSpec** | All | SetSpec | [ADR-0009](../adr/0009-setspec-schema-strategy.md) |
| Benchmark result schema | **SetSpec** (shape) / **FreeWeight** (content) | LoadCoach, external tools | SetSpec `benchmark.result` | [SetSpec §7](../packages/setspec/spec.md) |
| Capability evidence schema | **SetSpec** (shape) / **FreeWeight** (content) | LoadCoach | SetSpec `capability.evidence` | [ADR-0022](../adr/0022-capability-evidence-record-contract.md) — normative field set, freshness semantics, uniqueness and identity binding |
| Runtime profile resolution and served context | **LoadCoach** (for execution) / **FreeWeight** (for measurement) | Both | Recorded on every job and every run | [ADR-0023](../adr/0023-runtime-profile-resolution.md) |
| Envelope boundaries (what carries `schema_version`) | **SetSpec** | All | SetSpec + OpenAPI | [ADR-0025](../adr/0025-envelope-boundaries.md) |
| Per-device measurement and admission | **SweatMeter** (reports) / consumers (attribute) | FreeWeight, LoadCoach | Python API + `gpu_index` on every figure | [ADR-0027](../adr/0027-multi-gpu-semantics.md) |
| Prompt record schema, rendering and hashing | **SetSpec** (`setspec.prompts`) | All three applications | Python API + `prompt.record` / `prompt.manifest` | [ADR-0028](../adr/0028-prompt-pack-granularity.md) |
| `Host` validation, CSRF, outbound-fetch allowlist | **MirrorWall** (middleware) / **Standards** (policy) | All three applications | HTTP | [ADR-0026](../adr/0026-local-http-hardening.md) |
| Evidence bundle | **SetSpec** (shape) / **FreeWeight** (content) | LoadCoach | SetSpec `benchmark.evidence_bundle` | [FreeWeight API §6](../apps/freeweight/api.md) |
| Event envelope | **SetSpec** | All | SetSpec `event.envelope` | [Observability §4.1](../standards/observability-standards.md) |
| Error envelope | **SetSpec** | All | SetSpec `error.envelope` | [API Standards §4](../standards/api-and-contract-standards.md) |
| JSON Schema and golden payloads | **SetSpec** | Producer and consumer test suites | Package data | [SetSpec Phase 4](../packages/setspec/development-plan.md) |

## 4. Application requirements

| Requirement | Owner | Consumers | Contract | Specification |
|---|---|---|---|---|
| Benchmark definitions and manifests | **FreeWeight** | User | Internal + export | [Benchmark Catalog](../apps/freeweight/benchmark-catalog.md) |
| Benchmark execution and scheduling | **FreeWeight** | User | Internal + HTTP v1 | [FreeWeight §4](../apps/freeweight/spec.md) |
| Benchmark **scoring** | **FreeWeight** | User, LoadCoach (as evidence) | SetSpec + HTTP v1 | [Benchmark Catalog §1](../apps/freeweight/benchmark-catalog.md) |
| Reproducibility fingerprint | **FreeWeight** | User, LoadCoach | SetSpec | [Machine Identity §4](machine-identity-and-reproducibility.md) |
| Capability aggregation, **confidence and freshness** | **FreeWeight** | LoadCoach | SetSpec `capability.evidence` | [ADR-0017](../adr/0017-benchmark-confidence-and-freshness.md) |
| Result comparison and comparability verdicts | **FreeWeight** (using BaseAiCore rules) | User | HTTP v1 | [FreeWeight API §5](../apps/freeweight/api.md) |
| External benchmark isolation and sandboxing | **FreeWeight** | User | Internal | [ADR-0018](../adr/0018-external-benchmark-isolation.md) |
| Task profiles | **LoadCoach** | IdeaPress (by name only), external tools | HTTP v1 | [Routing §2](../apps/loadcoach/routing.md) |
| Model routing and selection | **LoadCoach** | IdeaPress, external tools | HTTP v1 | [Routing](../apps/loadcoach/routing.md) |
| Routing explainability | **LoadCoach** | IdeaPress, user | HTTP v1 | [Routing §8](../apps/loadcoach/routing.md) |
| Job queue, priorities, scheduling | **LoadCoach** | IdeaPress, external tools | HTTP v1 | [ADR-0010](../adr/0010-queue-implementation.md) · [Queue](../apps/loadcoach/queue-and-scheduling.md) |
| Admission control and residency | **LoadCoach** | Internal | Internal | [Queue §5–6](../apps/loadcoach/queue-and-scheduling.md) |
| Execution, retry, fallback, validation | **LoadCoach** | IdeaPress, external tools | HTTP v1 | [Queue §7](../apps/loadcoach/queue-and-scheduling.md) |
| Production evidence and reliability | **LoadCoach** | Internal, user | HTTP v1 | [Routing §11](../apps/loadcoach/routing.md) |
| Content workflows and stages | **IdeaPress** | User | Internal + HTTP v1 | [Workflows](../apps/ideapress/workflows.md) |
| Requirement compilation and coverage | **IdeaPress** | User | Internal | [Workflows §3](../apps/ideapress/workflows.md) |
| Deterministic content validation | **IdeaPress** | User | Internal | [Workflows §4](../apps/ideapress/workflows.md) |
| Inference backend abstraction | **IdeaPress** | Internal | Python protocol (internal) | [Workflows §6](../apps/ideapress/workflows.md) |
| Content export | **IdeaPress** | User | Files | [IdeaPress §7](../apps/ideapress/spec.md) |

## 5. Cross-cutting requirements

| Requirement | Owner | Applies to | Contract | Specification |
|---|---|---|---|---|
| Configuration strategy and precedence | **Standards** (each app implements) | All three applications | Config | [Configuration Standards](../standards/configuration-standards.md) |
| Prompt storage and versioning | **Standards** (each component implements) | FreeWeight, LoadCoach, IdeaPress | JSON records + package data | [ADR-0012](../adr/0012-prompt-storage-format.md) |
| API conventions and versioning | **Standards** (each app implements) | All three applications | HTTP v1 | [ADR-0013](../adr/0013-api-versioning.md) |
| Authentication and exposure | **Standards** (each app implements) | All three applications | HTTP v1 | [ADR-0014](../adr/0014-authentication-strategy.md) |
| Security posture and trust boundaries | **Standards** | All | — | [Security Standards](../standards/security-standards.md) |
| Structured logging, request IDs, events | **Standards** + **MirrorWall** helpers | All three applications | Log format + SetSpec | [Observability Standards](../standards/observability-standards.md) |
| Health and status reporting | **Standards** + **MirrorWall** primitives | All three applications | HTTP v1 | [Graceful Degradation §3](graceful-degradation.md) |
| Graceful degradation behaviour | **Standards** (each component implements) | All | — | [Graceful Degradation](graceful-degradation.md) |
| Performance budgets | **Standards** (each component measures) | All | Tests | [Performance Targets](performance-targets.md) |
| Cross-platform isolation | **Standards** + **SweatMeter** interfaces | All | Python protocols | [Cross-Platform Standards](../standards/cross-platform-standards.md) |
| Packaging, versioning, release | **Standards** | All nine repositories | — | [Packaging Standards](../standards/packaging-and-release-standards.md) |
| Testing architecture | **Standards** | All | — | [Testing Standards](../standards/testing-standards.md) |
| Quality targets | **Standards** | All | Measured gates | [Gold Standards](../standards/gold-standards.md) |

---

## 6. Ownership analysis

### 6.1 Gaps found and closed

| Potential gap | Resolution |
|---|---|
| Who computes **confidence** for benchmark evidence? | **FreeWeight** computes it ([ADR-0017](../adr/0017-benchmark-confidence-and-freshness.md)); LoadCoach applies it. Stated in both specs so neither reimplements it |
| Who owns the **capability vocabulary** — BaseAiCore or SetSpec? | BaseAiCore owns the *type* (`CapabilityId`, syntax); SetSpec owns the *contents* and their version. Recorded in §1 above and in both specs |
| Who owns the **event envelope** — MirrorWall or SetSpec? | SetSpec owns the shape; MirrorWall owns the transport and helpers. MirrorWall is the one capability package permitted to import SetSpec, and only for this |
| Who owns **machine profile collection** vs its **type**? | BaseAiCore defines `MachineProfile` and the fingerprint function; SweatMeter populates it |
| Who owns **energy metrics**? | SweatMeter integrates power samples into joules; FreeWeight defines the derived benchmark metrics (joules per token, per task) |
| Who decides **retry policy** — ModelRack or its callers? | ModelRack surfaces typed errors and never retries internally; callers own policy. Stated in ModelRack's non-goals |
| Who owns the **sandbox**? | FreeWeight, as the only component that runs untrusted code ([ADR-0018](../adr/0018-external-benchmark-isolation.md)) |
| Who owns **task profiles** — LoadCoach or IdeaPress? | LoadCoach. IdeaPress knows only its own stage vocabulary and maps to task IDs in exactly one adapter module |
| Who owns **prompt records** for a shared concern (e.g. a JSON-repair instruction)? | Each component owns its own pack; a duplicated prompt is acceptable, and no prompt is ever shared. **The machinery is a different question and was previously unanswered**: the loader, renderer and hasher live in `setspec.prompts`, because prompt hashes appear in cross-application evidence and three implementations would be three chances for a determinism contract to disagree ([ADR-0028](../adr/0028-prompt-pack-granularity.md)) |
| Who decides which **runtime profile** an execution runs under, and how does evidence match it? | LoadCoach resolves it and records it; evidence contributes only when its `runtime_profile_hash` matches. Previously unowned, which meant the hash was stored and never used ([ADR-0023](../adr/0023-runtime-profile-resolution.md)) |
| Which timestamp does evidence **freshness** decay from? | `measured_at`, the latest contributing run's `completed_at`. Previously the producer stored only `computed_at` and the consumer only `measured_at`, with no mapping — so re-aggregation would have reset apparent age ([ADR-0022](../adr/0022-capability-evidence-record-contract.md)) |
| Does an HTTP request body crossing an application boundary need a SetSpec envelope? | No. Transferable documents carry the envelope; an API's own bodies are versioned by their path and contracted through OpenAPI ([ADR-0025](../adr/0025-envelope-boundaries.md)) |
| Who owns the **comparability rules**? | BaseAiCore defines the rules as data; FreeWeight applies them in comparison and evidence separation |

### 6.2 Duplicated responsibility checked

| Concern | Appears in | Verdict |
|---|---|---|
| Model identity storage | FreeWeight and LoadCoach both have a `models` table | **Not duplication.** Each owns its own database; the *identity definition* is shared through BaseAiCore and the column set is normative in both data models |
| Telemetry | FreeWeight persists during runs; LoadCoach reads for admission control | **Not duplication.** One implementation (SweatMeter), two uses |
| Validation | FreeWeight scores benchmark outputs; LoadCoach validates job outputs; IdeaPress validates content | **Not duplication.** Three different questions ("is this correct for the benchmark?", "does this satisfy the request contract?", "does this meet the author's requirements?") with three different consequences |
| Queueing | FreeWeight has a run scheduler; LoadCoach has a job queue; IdeaPress has one stage task per project | **Not duplication**, but the closest call in the suite. Resolution: FreeWeight's scheduler is deliberately minimal (one GPU workload, no priorities, no leases); LoadCoach's queue is the full implementation. If FreeWeight ever needs priorities or leases, that is the trigger to extract a shared queue — recorded here as a watch item, not as work |
| Event stores | All three persist events with sequences | **Partially shared.** The SSE transport, replay semantics and envelope live in MirrorWall + SetSpec; the tables are per-application because they reference application entities |
| Model discovery | All three discover models | **Not duplication.** One implementation (ModelRack); three callers with different persistence needs |
| Capability scoring | FreeWeight computes capability scores; LoadCoach computes task fit | **Not duplication**, and the boundary is explicit: capability score = "how good is this model at X?" (measured); task fit = "how well does that serve this task, right now?" (weighted, adjusted) |

### 6.3 Circular dependency check

Verified in [Dependency and Boundary Rules §6](dependency-and-boundary-rules.md). No cycles exist at
package, application or documentation level. The two deliberate asymmetries:

* **FreeWeight → LoadCoach** is consumer-pull: LoadCoach reads FreeWeight's public evidence. FreeWeight
  has no knowledge of LoadCoach — no code, no configuration, no schema.
* **IdeaPress → LoadCoach** is consumer-push: IdeaPress calls LoadCoach and sends feedback. LoadCoach
  never calls IdeaPress.

The one *documentation* cycle — SetSpec's benchmark schemas depend on FreeWeight's real output while
FreeWeight's export depends on the frozen schemas — is resolved by sequencing (draft → real results →
freeze → export), recorded in [SetSpec's plan](../packages/setspec/development-plan.md) and in the
[roadmap §2](../roadmap/master-roadmap.md). It is a build-order dependency, not a runtime one.

### 6.4 Contract clarity check

Every row in §1–§5 names a contract type. Contracts that are `Internal` are owned by exactly one
component and are not depended on by anything else — verified by the import-linter contracts and by
the rule that cross-application communication uses only HTTP v1 or SetSpec payloads.

The three contracts under the most pressure, and their protection:

| Contract | Pressure | Protection |
|---|---|---|
| `benchmark.evidence_bundle` | The whole FreeWeight → LoadCoach value proposition | Frozen at M3; JSON Schema + goldens; contract tests in both repositories; consumer harness that imports only `setspec`; the normative field set and matching rules in [ADR-0022](../adr/0022-capability-evidence-record-contract.md), including the freshness test that asserts re-aggregation does not raise confidence |
| LoadCoach `/api/v1/generate` | IdeaPress and any external tool | OpenAPI snapshot diff-checked in CI; additive-only within v1; version negotiation on first contact |
| `ModelIdentity` | Every component, every stored measurement | Golden-value tests; a change is a suite-wide coordinated release |
