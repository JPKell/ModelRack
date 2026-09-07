# Local AI Suite — Master Documentation

**Status:** Architecture frozen 2026-08-21, audited and corrected the same day. Implementation is
done: all fourteen components hold working software, are tagged, and are published or release-ready,
and [the roadmap's §9](roadmap/master-roadmap.md#9-current-state-and-immediate-next-steps) is the one
place that records where each of them stands. Read the
[final architecture audit](reviews/final_architecture_audit.md) before starting a phase: it added
ADR-0022 – ADR-0029 and corrected the specifications they touch.
**Purpose:** this documentation set is the **single source of truth** for the suite. An implementation
agent assigned any application, package or phase should be able to build it from these documents
without inventing architectural decisions. If a decision is missing, that is a defect here — close it
with an ADR before writing the code.

---

## 1. What this suite is

Applications and shared Python packages for operating local open-weight AI models. The nine
components of Suite 1.0:

```text
Measure AI   →   Manage AI   →   Apply AI
FreeWeight       LoadCoach       IdeaPress
```

Each application works alone. Each gains from the others without requiring them. Start with the
[Executive Summary](architecture/executive-summary.md).

**Built after 1.0** — the two post-1.0 arcs, planned on 2026-09-01/02 with their contracts accepted
as ADRs 0045–0067 before any code, and delivered between 2026-09-02 and 2026-09-06:

```text
Harness AI
PromptCadence    + CutCtx · ToolYard · LoadLedger · Commissioner
```

PromptCadence is a fourth application — a plan-approved, tier-routed agent loop over LoadCoach — and
the four packages are the capabilities it justifies extracting. In parallel, the
[Adapter arc](roadmap/adapter-roadmap.md) added hot-swappable LoRA serving, over llama.cpp, to the
existing components. Both were executed row by row through
[Outstanding Work](roadmap/outstanding-work.md), which records what each row shipped; each of the
five has its own repository, and `promptcadence`, `cutctx`, `toolyard`, `loadledger` and
`commissioner` are importable packages on PyPI.

---

## 2. Start here

| If you are… | Read, in this order |
|---|---|
| **New to the suite** | [Executive Summary](architecture/executive-summary.md) → [Master Architecture](architecture/master-architecture.md) → [Master Roadmap](roadmap/master-roadmap.md) |
| **Implementing a phase** | The master requirements → [Master Architecture](architecture/master-architecture.md) → the component's `spec.md` → that phase in its `development-plan.md` → the standards it touches |
| **Working on either post-1.0 arc** | [Outstanding Work](roadmap/outstanding-work.md) (the schedule) → the arc's roadmap ([PromptCadence](roadmap/promptcadence-roadmap.md) · [Adapter](roadmap/adapter-roadmap.md)) → the decisions it rests on (ADRs [0045–0067](adr/README.md)) → the component's `spec.md` → that phase in its `development-plan.md` |
| **Deciding something architectural** | [ADR index](adr/README.md) → [Dependency and Boundary Rules](architecture/dependency-and-boundary-rules.md) → [Traceability Matrix](architecture/traceability-matrix.md) |
| **Reviewing a change** | [Coding Standards](standards/coding-standards.md) → [Testing Standards](standards/testing-standards.md) → [Gold Standards](standards/gold-standards.md) |
| **Wondering why something is the way it is** | [Legacy Material Inventory](inventory/legacy-material-inventory.md) → the relevant [ADR](adr/README.md) |
| **Wondering what changed after the freeze** | [Final Architecture Audit](reviews/final_architecture_audit.md) |

---

## 3. Architecture

| Document | Contents |
|---|---|
| [Executive Summary](architecture/executive-summary.md) | Purpose, vision, components, dependency model, independent deployment, benefits, development order |
| [Master Architecture](architecture/master-architecture.md) | Canonical vocabulary, layering, ownership boundaries, runtime and concurrency model, communication contracts, deployment shapes, data flows, extension points, what the architecture forbids |
| [Canonical Model Identity](architecture/canonical-model-identity.md) | `ModelIdentity`, descriptor, runtime profile, measurement subject, comparability rules, persistence |
| [Adapter Identity and Serving](architecture/adapter-identity-and-serving.md) | The adapter axis on the execution subject, selection versus serving mode, the directory-and-manifest registry, measured-never-inherited evidence, two-level residency, adapter governance |
| [Machine Identity and Reproducibility](architecture/machine-identity-and-reproducibility.md) | `MachineProfile`, machine fingerprint, reproducibility fingerprint, environment drift, required provenance |
| [Dependency and Boundary Rules](architecture/dependency-and-boundary-rules.md) | Allowed and forbidden imports, cross-application communication, enforcement, circular-dependency analysis |
| [Graceful Degradation](architecture/graceful-degradation.md) | The four outcomes, the full degradation matrix, health reporting, startup validation |
| [Performance Targets](architecture/performance-targets.md) | Application-overhead budgets by area, memory budgets, what is deliberately not promised, verification |
| [Traceability Matrix](architecture/traceability-matrix.md) | Requirement → owner → consumers → contract; ownership gaps, duplication and circularity analysis |
| [Risk Register](architecture/risk-register.md) | Suite-wide risks, trade-offs, non-goals, watch items |

---

## 4. Standards

| Document | Contents |
|---|---|
| [Coding Standards](standards/coding-standards.md) | Docstring-first development, typing, naming, errors, state, comments, tooling, anti-patterns |
| [Testing Standards](standards/testing-standards.md) | Test layout and types, never using a model as an oracle, determinism, coverage, contract testing, CI execution |
| [API and Contract Standards](standards/api-and-contract-standards.md) | Versioning, resource naming, envelopes, errors, request IDs, pagination, SetSpec payloads, SSE, limits, documentation |
| [Security Standards](standards/security-standards.md) | Trust boundaries, exposure, authentication, input validation, filesystem safety, model output, sandboxing, secrets, egress, threat model |
| [Configuration Standards](standards/configuration-standards.md) | Precedence (and the separate execution-parameter chain), format, environment, validation, defaults, secrets, runtime-changeable settings |
| [Database Standards](standards/database-standards.md) | Ownership, engines, schema conventions, indexes, migrations, transactions, backup, destructive operations |
| [Observability Standards](standards/observability-standards.md) | Structured logging, correlation IDs, events vs logs, metrics, health, error reporting, log storage |
| [Prompt Management Standards](standards/prompt-management-standards.md) | JSON prompt records, packs and hashing, traceability, rendering, overrides, testing, review |
| [UI/UX Standards](standards/ui-ux-standards.md) | Design tokens, typography, shell, components, data display, states, accessibility, responsiveness, theme, acceptance checklist |
| [CLI Standards](standards/cli-standards.md) | Command shape, help, output, exit codes, non-interactive operation, local vs client mode, errors, scriptability |
| [Cross-Platform Standards](standards/cross-platform-standards.md) | Support tiers, the isolation rule, platform-dependent surfaces, degradation, testing, adding a platform |
| [Packaging and Release Standards](standards/packaging-and-release-standards.md) | Repository model, licensing, versioning, dependencies, CI, release procedure, compatibility, distribution |
| [Gold Standards](standards/gold-standards.md) | Measurable quality targets, suite-wide and per component, with their gates |

---

## 5. Architecture Decision Records

Full index with statuses: [adr/README.md](adr/README.md).

| ADR | Decision |
|---|---|
| [0001](adr/0001-application-and-package-separation.md) | Three applications, six packages, nine repositories |
| [0002](adr/0002-web-framework.md) | FastAPI on Uvicorn |
| [0003](adr/0003-sync-vs-async-strategy.md) | Async at the HTTP edge, synchronous everywhere below |
| [0004](adr/0004-sse-vs-websockets.md) | Server-Sent Events for all streaming |
| [0005](adr/0005-database-strategy.md) | SQLAlchemy 2.0 + Alembic |
| [0006](adr/0006-sqlite-and-postgresql-roles.md) | SQLite default, PostgreSQL supported, nothing else |
| [0007](adr/0007-provider-abstraction.md) | One provider abstraction, Ollama first, fake provider built first |
| [0008](adr/0008-canonical-model-identity.md) | Minimal immutable identity; descriptor and runtime profile separate |
| [0009](adr/0009-setspec-schema-strategy.md) | Pydantic models, per-payload `MAJOR.MINOR`, goldens and JSON Schema |
| [0010](adr/0010-queue-implementation.md) | Database-backed queue with leases; no broker |
| [0011](adr/0011-shared-package-boundaries.md) | Extraction at the second consumer; `LoadCoachClient` deferred |
| [0012](adr/0012-prompt-storage-format.md) | Prompts as versioned JSON records |
| [0013](adr/0013-api-versioning.md) | Path-based major versioning, additive within a major |
| [0014](adr/0014-authentication-strategy.md) | No auth on loopback; mandatory bearer tokens otherwise |
| [0015](adr/0015-repository-and-distribution-model.md) | One repository per component, `src/` layout, Trusted Publishing |
| [0016](adr/0016-unavailable-is-not-zero.md) | An `Unsupported` sentinel that refuses to behave like a number |
| [0017](adr/0017-benchmark-confidence-and-freshness.md) | FreeWeight computes confidence; LoadCoach applies it |
| [0018](adr/0018-external-benchmark-isolation.md) | Subprocess isolation; tiered sandbox ending in refusal |
| [0019](adr/0019-python-baseline-and-config-format.md) | Python ≥ 3.12; TOML configuration, JSON data |
| [0020](adr/0020-ui-rendering-strategy.md) | Server-rendered HTML with progressive enhancement |
| [0021](adr/0021-telemetry-collection-strategy.md) | `/proc` + `/sys` and `nvidia-smi`, all readers injectable |

ADRs 0022–0044 were added during implementation (the post-freeze audit, then the FreeWeight,
LoadCoach and IdeaPress builds). **ADRs 0045–0067** are the two post-1.0 arcs' contracts, written
2026-09-02 before any of their code; **ADRs 0068–0102** were added as those arcs were built, each
closing a question the code raised:

| ADR | Decision |
|---|---|
| [0045](adr/0045-promptcadence-reaches-models-only-through-loadcoach.md) – [0057](adr/0057-the-explanation-is-materialized-and-the-rows-stay-authoritative.md) | The PromptCadence arc: a fourth application that reaches models only through LoadCoach; ordered data classification; tiers as configuration; the bypass that removes planning and never governance; approval as a mode with its own scope; mountable package tables; the one payload that travels; compaction as a view; tool discipline; Commissioner's scope; multi-provider registration; the `ExecutionIntent`; the materialized explanation |
| [0058](adr/0058-the-execution-subject-gains-an-adapter-axis.md) – [0067](adr/0067-reliability-keys-on-the-subject-not-the-base.md) | The Adapter arc: the adapter axis on the execution subject; evidence measured never inherited; selection versus serving mode; the directory-and-manifest registry; llama.cpp through a supervised process; one adapter at a time; selection through the capability vocabulary; adapters classified and local-only; two-level residency; reliability keyed on the subject |

---

## 6. Applications

### [FreeWeight](apps/freeweight/spec.md) — measure

| Document | Contents |
|---|---|
| [Specification](apps/freeweight/spec.md) | Purpose, scope, non-goals, responsibilities, contracts, configuration, errors, security, performance, tests, acceptance criteria |
| [Benchmark Catalog](apps/freeweight/benchmark-catalog.md) | Scoring ladder, categories, native suites, external adapters, goal suites, manifests, capability mapping |
| [Subjective Goals](apps/freeweight/subjective-goals.md) | User-authored goals: the goal pack, rule and judged criteria, the calibration protocol, the jury, the authoring wizard, starter packs |
| [Data Model](apps/freeweight/data-model.md) | Tables, run state machine, retention, query-plan requirements |
| [API](apps/freeweight/api.md) | `/api/v1` endpoints, events, exports, the evidence integration point |
| [Development Plan](apps/freeweight/development-plan.md) | 14 phases (plus 8A, 8B, 10A) from first page to 1.0, with model assignment per step |
| [Risks](apps/freeweight/risks.md) | Technical, integration, security, portability, performance, model, maintenance risks; trade-offs; traps |

### [LoadCoach](apps/loadcoach/spec.md) — manage

| Document | Contents |
|---|---|
| [Specification](apps/loadcoach/spec.md) | Purpose, scope, non-goals, responsibilities, contracts, configuration, errors, security, performance, tests, acceptance criteria |
| [Routing](apps/loadcoach/routing.md) | Task profiles, constraint filtering, scoring, adjustment factors, ranking, explanation, context budgeting, overrides, production evidence |
| [Queue and Scheduling](apps/loadcoach/queue-and-scheduling.md) | Job classes, states, leases, ageing, admission, residency, retries, cancellation, timeouts, recovery, simulation |
| [Data Model](apps/loadcoach/data-model.md) | Tables, retention, query-plan requirements |
| [API](apps/loadcoach/api.md) | `/api/v1` endpoints, generation, jobs, evidence, feedback, scopes, client guidance |
| [Development Plan](apps/loadcoach/development-plan.md) | 9 phases, including the WeightsDB and MirrorWall extractions |
| [Risks](apps/loadcoach/risks.md) | Risks, trade-offs, traps |

### [IdeaPress](apps/ideapress/spec.md) — apply

| Document | Contents |
|---|---|
| [Specification](apps/ideapress/spec.md) | Purpose, scope, non-goals, responsibilities, contracts, configuration, errors, security, performance, tests, acceptance criteria |
| [Workflows](apps/ideapress/workflows.md) | The pipeline, stages, requirement compilation, validation, bounded loops, the inference port, context assembly, commit, what a model may never do |
| [Data Model](apps/ideapress/data-model.md) | Tables, unit state machine, retention and privacy, query-plan requirements |
| [API](apps/ideapress/api.md) | `/api/v1` endpoints, stage tasks, units, backends, errors, streaming |
| [Development Plan](apps/ideapress/development-plan.md) | 9 phases, standalone first, LoadCoach last |
| [Risks](apps/ideapress/risks.md) | Risks, trade-offs, traps |

### [PromptCadence](apps/promptcadence/spec.md) — harness

| Document | Contents |
|---|---|
| [Specification](apps/promptcadence/spec.md) | Purpose, scope, non-goals, responsibilities, contracts, configuration, errors, security, performance, tests, acceptance criteria |
| [Lifecycle](apps/promptcadence/lifecycle.md) | The two paths, classification, tiers, the plan and the `ExecutionIntent`, deviation handling, budgets, compaction, the state machine and its recovery edges, the explanation |
| [Development Plan](apps/promptcadence/development-plan.md) | 9 phases, bypass loop first, planning last |

---

## 7. Packages

| Package | Layer | Specification | Development plan | Built at |
|---|---|---|---|---|
| **BaseAiCore** | 1 — domain foundation | [spec](packages/baseaicore/spec.md) | [plan](packages/baseaicore/development-plan.md) | First, before everything |
| **SetSpec** | 2 — contracts | [spec](packages/setspec/spec.md) | [plan](packages/setspec/development-plan.md) | Second; frozen at M3 |
| **ModelRack** | 3 — capability | [spec](packages/modelrack/spec.md) | [plan](packages/modelrack/development-plan.md) | Third |
| **SweatMeter** | 3 — capability | [spec](packages/sweatmeter/spec.md) | [plan](packages/sweatmeter/development-plan.md) | Fourth (parallel with ModelRack) |
| **WeightsDB** | 3 — capability | [spec](packages/weightsdb/spec.md) | [plan](packages/weightsdb/development-plan.md) | Extracted at LoadCoach P1 |
| **MirrorWall** | 3 — capability | [spec](packages/mirrorwall/spec.md) | [plan](packages/mirrorwall/development-plan.md) | Extracted at LoadCoach P4 |
| **CutCtx** | 3 — capability | [spec](packages/cutctx/spec.md) | [plan](packages/cutctx/development-plan.md) | M10; published as `cutctx 0.1.0` |
| **ToolYard** | 3 — capability | [spec](packages/toolyard/spec.md) | [plan](packages/toolyard/development-plan.md) | M10, before any PromptCadence tool executed; `toolyard 0.1.1` |
| **LoadLedger** | 3 — capability | [spec](packages/loadledger/spec.md) | [plan](packages/loadledger/development-plan.md) | M10; published as `loadledger 0.2.0` |
| **Commissioner** | 3 — capability | [spec](packages/commissioner/spec.md) | [plan](packages/commissioner/development-plan.md) | M10, after SetSpec 0.5 published its payload; `commissioner 0.1.1` |

The last four were built at M10 of the
[PromptCadence arc](roadmap/promptcadence-roadmap.md), each with two named consumers, per
[ADR-0011](adr/0011-shared-package-boundaries.md)'s extraction rule: PromptCadence is the first
consumer of all four, and IdeaPress the second, adopting three of them at M13.

---

## 8. Roadmap and history

| Document | Contents |
|---|---|
| [Master Roadmap](roadmap/master-roadmap.md) | Milestones M1–M9, dependency graph, work streams, parallelism rules, integration milestones, stabilization phases, version trajectory, the professional-delivery checklist, immediate next steps |
| [Outstanding Work](roadmap/outstanding-work.md) | The one schedule: every remaining row of both arcs in execution order, one row per model session, with its model and effort |
| [PromptCadence Arc](roadmap/promptcadence-roadmap.md) | M10–M13: the harness and its four packages — decisions D-1…D-13 (now ADRs 0045–0057), milestones, work streams, integration verifications, risks |
| [Adapter Arc](roadmap/adapter-roadmap.md) | LA0–LA3: hot-swappable LoRA serving — decisions A-1…A-10 (now ADRs 0058–0067), checkpoints, per-component work, sequencing against the harness arc |
| [Model Assignment Guide](roadmap/model-assignment.md) | Advisory: which model and reasoning effort to point at each phase, what makes a phase hard for a model, the first-instance rule, and where never to economize |
| [Legacy Material Inventory](inventory/legacy-material-inventory.md) | Everything inspected in `planning/` and `.old_projects/`: what was adopted, what was rejected and why, conflicts and their resolutions, technical debt not inherited, the observed environment |
| [Final Architecture Audit](reviews/final_architecture_audit.md) | The post-freeze audit: 41 findings by severity, the corrections made, ADRs 0022–0029 and the seven amended, deployment combinations re-verified, deliberately deferred concerns, and the clean-room verification |

---

## 9. Organization of this documentation set

The structure follows the recommended layout, with one addition:

```text
docs/
├── README.md            this index
├── architecture/        suite-level architecture and analysis
├── standards/           suite-wide standards every component follows
├── adr/                 architecture decision records
├── roadmap/             the master development roadmap
├── inventory/           ← addition: the legacy-material inventory
├── reviews/             ← addition: architecture reviews and audits
├── apps/                one directory per application
└── packages/            one directory per shared package
```

**Why `reviews/` exists:** an audit is neither architecture nor a decision record. It is the evidence
that the architecture was checked, and the trail explaining why eight ADRs appeared after the freeze.
The ADRs carry the decisions; the audit carries the findings, their severity, and what was
deliberately left alone. A reader who wants to know *what changed after the freeze and why* has one
place to look.

**Why `inventory/` exists as its own directory:** the requirements mandate an inventory of prior
material *before* the architecture is designed, and that inventory is neither architecture nor a
standard — it is the audit trail explaining which prior decisions were kept, which were rejected and
why. Keeping it separate stops it from being mistaken for current architecture, while leaving it
citable from every document that inherited a pattern from it. It is the first place to look when
someone asks "why didn't you just do what the old spec said?"

Each application directory carries more than `spec.md` and `development-plan.md` because four
subjects are too large to nest inside a specification without burying them: the benchmark catalogue,
the routing and queue designs, the workflow pipeline, and the trajectory lifecycle. Each is
referenced from its specification and does not duplicate it.

---

## 10. Deliverables checklist

| Requirement | Deliverable | Status |
|---|---|---|
| §4 Inventory of existing material | [Legacy Material Inventory](inventory/legacy-material-inventory.md) | Complete |
| §28 Executive summary | [Executive Summary](architecture/executive-summary.md) | Complete |
| §28 Master architecture | [Master Architecture](architecture/master-architecture.md) + 8 supporting documents | Complete |
| §28 Master development roadmap | [Master Roadmap](roadmap/master-roadmap.md) | Complete |
| §28 Coding standards | [Coding Standards](standards/coding-standards.md) | Complete |
| §28 Testing standards | [Testing Standards](standards/testing-standards.md) | Complete |
| §28 API and contract standards | [API and Contract Standards](standards/api-and-contract-standards.md) | Complete |
| §28 Security standards | [Security Standards](standards/security-standards.md) | Complete |
| §28 Configuration standards | [Configuration Standards](standards/configuration-standards.md) | Complete |
| §28 Packaging and release standards | [Packaging and Release Standards](standards/packaging-and-release-standards.md) | Complete |
| §28 UI/UX standards | [UI/UX Standards](standards/ui-ux-standards.md) | Complete |
| §28 CLI standards | [CLI Standards](standards/cli-standards.md) | Complete |
| §29 ADRs | [102 ADRs](adr/README.md) — 21 at the freeze, covering every listed topic and seven more, and one per decision taken since | Complete |
| §30 Specification per component | 14 specifications, each with all 21 required sections | Complete |
| §31 Development plan per component | 14 plans, 91 phases, each with goal, prerequisites, work, files, tests, acceptance criteria, risks, failure modes, gold standards, deferred work | Complete |
| §32 Gold standards | [Gold Standards](standards/gold-standards.md), suite-wide and per component | Complete |
| §33 Risk and failure analysis | [Risk Register](architecture/risk-register.md) + three application risk documents + per-phase risks | Complete |
| §34 Traceability matrix | [Traceability Matrix](architecture/traceability-matrix.md) | Complete |
| §35 Performance planning | [Performance Targets](architecture/performance-targets.md) + per-component budgets | Complete |
| §16 Observability | [Observability Standards](standards/observability-standards.md) | Complete |
| §12 Prompt management | [Prompt Management Standards](standards/prompt-management-standards.md) | Complete |
| §15 Database standards | [Database Standards](standards/database-standards.md) | Complete |
| §20 Graceful degradation | [Graceful Degradation](architecture/graceful-degradation.md) | Complete |
| §22 Cross-platform strategy | [Cross-Platform Standards](standards/cross-platform-standards.md) | Complete |
| §37 Professional delivery target | [Master Roadmap §7](roadmap/master-roadmap.md) | Complete |
| §39 Consistency review | [§11 below](#11-consistency-review) | Complete |

---

## 11. Consistency review

Performed across the whole set on 2026-08-21, per requirement §39. Each item states how it was
verified, not merely that it was.

| Check | Verification | Result |
|---|---|---|
| Component names used consistently | Grep for every component name and its lowercase form across all documents; no `openweight_bench`, no alternative spellings | Pass |
| Public contracts agree across documents | Evidence bundle, generate response, event and error envelopes cross-checked between producer spec, consumer spec, API documents and SetSpec | Pass |
| Model identity consistent everywhere | `ModelIdentity` fields, canonical-ID form and column set compared across BaseAiCore, both data models, both APIs and the architecture document | Pass |
| Configuration precedence consistent | Defaults → file → env → CLI stated identically in the standards and all three specs; the separate execution-parameter chain named as separate in both places it appears | Pass |
| API conventions consistent | Versioning, envelopes, pagination, SSE framing and error codes cross-checked between the standards and all three API documents | Pass |
| Database ownership consistent | Each data model states exclusive ownership; no document describes an application reading another's database | Pass |
| No application accesses another's DB | Stated as forbidden in the architecture, boundary rules, database standards and all three specs; the "temporary shortcut" from old planning explicitly rejected | Pass |
| No shared package imports application code | Import-linter contracts specified per repository; clean-venv install-check specified; each package spec lists its permitted imports | Pass |
| FreeWeight runs independently | Spec §20.1, degradation matrix, and its e2e suite requirement (peers absent) | Pass |
| LoadCoach runs independently | Spec §20.1 and §20.3 (routes with no evidence and no FreeWeight); declared-capability fallback documented in Routing §5.1 | Pass |
| IdeaPress runs independently | Spec §20.1, backend-parity test, and standalone-first phase ordering (LoadCoach not touched until P7) | Pass |
| IdeaPress can optionally use LoadCoach | Spec §20.2, Workflows §6, IdeaPress P7, LoadCoach API §12 | Pass |
| LoadCoach can optionally consume FreeWeight evidence | LoadCoach P6, FreeWeight API §6, integration milestone I4 | Pass |
| Tests planned before implementation | Every one of the 74 phases lists its tests before its acceptance criteria; Testing Standards rule zero | Pass |
| All phases contain acceptance criteria | Verified per phase across all nine development plans | Pass |
| All major decisions have rationale | 21 ADRs, each with context, decision, real alternatives, consequences and a revisit trigger | Pass |
| Old planning not treated as authoritative | Inventory §3 lists 21 rejected concepts with reasons; §4 records 10 conflicts and their resolutions | Pass |
| No unnecessary infrastructure introduced | No Redis, Celery, RabbitMQ, Kafka, Kubernetes, message broker, external cache or service mesh appears anywhere; ADR-0010 records the reasoning and the revisit trigger | Pass |
| Milestone labels consistent | M1–M9 cross-checked between the executive summary, the roadmap and all three development plans | Pass |
| Cross-document links resolve | Every relative link in every document checked against the file tree | Pass |
| Phase cross-references consistent | Extraction and adoption phases (WeightsDB at LC-P1, MirrorWall at LC-P4, both adopted at FW-P12) consistent in all five documents that mention them | Pass |

Issues found and fixed during the review are listed in the review's own commit; the two substantive
ones were a milestone renumbering (FreeWeight 1.0 needed its own milestone, M6, because it lands
after LoadCoach's extractions) and phase-reference drift between the package plans and FreeWeight's
adoption phase.

---

## 12. Maintaining this set

* **This set is versioned in its own repository** (`ai-suite-docs`), per
  [ADR-0015](adr/0015-repository-and-distribution-model.md). Component-level documentation that can
  drift from code — configuration references, OpenAPI snapshots, platform-support matrices — lives in
  each component's own repository and is generated and CI-diff-checked there.
* **A change here that contradicts an ADR requires a new ADR.** ADRs are superseded, never edited to
  hide a change of mind.
* **The consistency review in §11 is repeated at every milestone**, not only at M9.
* **When implementation reveals that a document is wrong, fix the document first**, then the code. A
  documentation set that lies is worse than none.
