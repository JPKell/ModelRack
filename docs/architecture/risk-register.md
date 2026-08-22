# Suite Risk Register

Risks that span more than one component, or that threaten the suite as a whole. Component-specific
risks live in each application's risk document
([FreeWeight](../apps/freeweight/risks.md) · [LoadCoach](../apps/loadcoach/risks.md) ·
[IdeaPress](../apps/ideapress/risks.md)) and in each package's specification.

**L** = likelihood, **I** = impact, both Low/Medium/High. Every risk names a mitigation that exists in
the design (not an intention) and an early signal that tells us it is materializing.

---

## 1. Architectural risks

| # | Risk | L | I | Mitigation | Early signal |
|---|---|---|---|---|---|
| A1 | **Boundary erosion** — an application imports another, or reads its database, "temporarily" | Medium | Critical | import-linter contracts in every repository; clean-venv install-and-import check; the rejection of "shared DB as a shortcut" recorded in the [inventory](../inventory/legacy-material-inventory.md) | A PR that adds a dependency between application repositories |
| A2 | **Shared package absorbs application logic** — WeightsDB grows a benchmark table, MirrorWall grows a runs page | Medium | High | Empty-`MetaData` test; term-scan test for application vocabulary; explicit non-goals in every package spec | A package release whose changelog names an application feature |
| A3 | **Premature extraction** of WeightsDB or MirrorWall shaped by one consumer | Medium | Medium | Extraction scheduled at the *second* consumer ([ADR-0011](../adr/0011-shared-package-boundaries.md)); FreeWeight's unchanged test suite is the adoption gate | An extracted package needing a change for its second consumer within weeks |
| A4 | **Contract freeze too early** — schemas frozen before their producer exists | Medium | High | SetSpec ships draft payloads, freezes only after FreeWeight P11 produces real results | Repeated `1.0-draft` changes after freeze |
| A5 | **Nine repositories become coordination overhead** | Medium | Medium | Shared CI template; documented cross-repository change procedure; compatible ranges; nightly compatibility matrix; ADR-0001's monorepo revisit trigger | More than half of changes needing three or more coordinated releases |
| A6 | **Over-engineering creep** — a broker, a cache layer, a plugin system arriving without a need | Medium | High | "No infrastructure without an ADR" rule; ADR-0010's reasoning as precedent; per-component "premature optimizations to avoid" lists | A design discussion that starts from a technology rather than a problem |
| A7 | **Under-engineering in the queue or event store** — the two genuinely subtle components | Medium | High | Scheduling simulator (with the clock advancing while the process is *up*, which is what exercises ageing); kill-point recovery tests; gap-free sequence tests; the mechanisms in [ADR-0029](../adr/0029-queue-mechanics.md) | Flaky tests around ordering or recovery |
| A8 | **A documented invariant with no mechanism behind it** — the class the final audit found repeatedly: ageing with no sweep, a lease with no keeper, a fingerprint input with no producer | Medium | High | Every invariant in the gold standards names the mechanism that provides it and the test that proves it; "the specification says so" is not an implementation | An acceptance criterion nobody can point at code for |

## 2. Delivery risks

| # | Risk | L | I | Mitigation | Early signal |
|---|---|---|---|---|---|
| D1 | **Scope inflation** — the benchmark catalogue and workflow stages both invite endless addition | High | High | 1.0 scope fixed per component; deferred lists are explicit and final for 1.0; external adapters confined to one phase | Phases growing new work items after they start |
| D2 | **Never reaching a demonstrable state** — the failure of both prior projects | Medium | Critical | Every phase ends in something runnable; UI work is not deferred to the end; FreeWeight P1 serves a page before any benchmark exists | A phase whose acceptance criteria cannot be demonstrated to a person |
| D3 | **Single maintainer, long sequence** — motivation and context loss across a nine-milestone plan | High | High | Phases sized to end in a visible result; parallelism table for when something is blocked; documentation written before code so context is recoverable | Long gaps between commits; phases restarted rather than continued |
| D4 | **Documentation drift** — code and this set diverging | High | Medium | Generated configuration references and OpenAPI snapshots diff-checked in CI; consistency review repeated at every milestone; architecture-level docs kept free of code-level detail | A doc statement contradicted by a passing test |
| D5 | **Test suite becoming slow enough to skip** | Medium | High | 3-minute default-suite budget; live and performance tests marked and excluded; fake provider everywhere | Default suite over budget |
| D6 | **Agent-implemented phases diverging from intent** | Medium | High | Docstring-first workflow; acceptance criteria per phase; standards documents referenced from every plan; this documentation set as the single source of truth | A phase completed with tests that assert implementation rather than contract |

## 3. Technical risks (cross-component)

| # | Risk | L | I | Mitigation | Early signal |
|---|---|---|---|---|---|
| T1 | **`Unsupported` handled inconsistently**, reintroducing fabricated zeros | Medium | High | The sentinel raises on coercion; storage keeps NULL + reason; JSON emits `"unsupported"`; UI renders `—`; tests at every layer | A chart or aggregate showing 0 where a sensor is absent |
| T2 | **Model identity fragmenting** across applications | Low | Critical | One type in BaseAiCore, golden-value tests, normative column set, identity created only by ModelRack | Two applications disagreeing about a model's name |
| T3 | **Timing semantics confused** — backend vs client, duration vs timestamp | Medium | High | Separate fields that are never merged; `perf_counter_ns` for durations; `token_level_chunks` gating per-token claims | Implausible tokens/second figures |
| T4 | **GPU contention** between suite applications on one machine | High | Medium | FreeWeight runs one GPU workload; LoadCoach's admission control sees the shortfall and defers; `loadcoach queue pause` documented for benchmarking sessions | Benchmarks with unusually wide dispersion |
| T5 | **SQLite limits reached** (write contention, size, aggregate latency) | Low | Medium | PostgreSQL supported and CI-tested; indexes and query-plan assertions from the first migration; retention controls | Dashboard queries or dispatch latency exceeding budget |
| T6 | **Streaming/replay defects** — duplicated or missing events | Medium | Medium | Persist-before-publish; subscribe-before-replay; dedupe by sequence; bounded queues; tests under injected races | A user seeing a duplicated progress line |
| T7 | **Threading defects** in schedulers, samplers and event fan-out | Medium | High | Shared mutable state confined to three named lock-protected objects per application; stress and leak tests; clean shutdown paths | Test flakiness under `pytest-randomly`; threads outliving the process |
| T8 | **Evidence that does not describe the execution it is used for** — matched on identity while the runtime profile differs | Medium | High | Candidates are execution subjects; evidence matches on `runtime_profile_hash` or is absent and named ([ADR-0023](../adr/0023-runtime-profile-resolution.md)) | Routing unchanged after importing evidence that obviously should have changed it |
| T9 | **Confidence that never decays** because freshness reads the aggregation time | Medium | High | `measured_at` is the freshness input and is carried from the contributing runs; a test re-aggregates old runs and asserts confidence does not rise ([ADR-0022](../adr/0022-capability-evidence-record-contract.md)) | Four-month-old evidence reporting full freshness |
| T10 | **Advertised context mistaken for served context**, producing silent truncation | Medium | High | Served context resolved, recorded with its source, and used by every constraint and estimate; `assumed_context` flagged | A model chosen for long context returning short, truncated answers |
| T11 | **Multi-GPU assumptions** — summed VRAM admitting work that fits nowhere; a slope measured against the wrong device | Low (today) | High | Per-device admission and attribution; memory metrics skipped when placement is unknown ([ADR-0027](../adr/0027-multi-gpu-semantics.md)) | An OOM after an admission check said it fits |

## 4. Security risks (cross-component)

| # | Risk | L | I | Mitigation | Early signal |
|---|---|---|---|---|---|
| S1 | **Accidental network exposure** of an inference service | Medium | High | Loopback default; bind + token + acknowledgement flag; startup refusal; the same rule in all three applications | A deployment guide suggesting `0.0.0.0` without tokens |
| S2 | **Model output treated as trusted** — executed, pathed, or rendered raw | Medium | Critical | Never executed anywhere in the suite; containment-checked paths; autoescaping and allowlist sanitizing; hostile-payload tests | A `\|safe` filter or a `subprocess` call near model output |
| S3 | **Untrusted code execution** for coding benchmarks | Low | Critical | Tiered sandbox with **refusal** at the bottom tier; no host-execution path exists; refusal asserted by test | A request to "just run it locally for now" |
| S4 | **Secret leakage** into repositories, logs or exports | Medium | High | gitleaks in CI; secrets only from env/keyring/file-reference; redaction filter; tests asserting absence | A config example containing a real-looking key |
| S5 | **Silent data egress** through a remote provider | Low | High | Off by default; explicit opt-in; egress badge in UI and in routing explanations; no other outbound calls exist | An outbound connection in a network-isolation test |
| S6 | **Malicious import** — evidence bundle, project archive, benchmark dataset | Low | High | Schema validation, size and ratio caps, hardened extraction, hash pinning, temp-directory staging | An import path that writes before validating |
| S7 | **Vulnerable dependency** | Medium | Medium | Small dependency budget; `pip-audit` blocking; lockfiles; justification required for new dependencies | An audit finding with no upgrade path |
| S8 | **DNS rebinding against an unauthenticated loopback service** — a visited web page reaching the local API | Medium | High | `Host` allowlist enforced before routing and before auth, in all three applications ([ADR-0026](../adr/0026-local-http-hardening.md)) | A request arriving with an unexpected `Host` |
| S9 | **Server-side request forgery** through `POST /evidence/import {"url": …}` | Low | Medium | Scheme, host-allowlist (loopback only by default), literal-IP, redirect and size checks before any parsing; `EVIDENCE_SOURCE_REFUSED` | An import naming a host the user never configured |
| S10 | **CSRF against HTML form routes** on an unauthenticated loopback bind | Medium | Medium | Double-submit token on form posts; the JSON API's exemption stated with its withdrawal condition rather than assumed | A state-changing form route without the token |

## 5. Dependency and ecosystem risks

| # | Risk | L | I | Mitigation | Early signal |
|---|---|---|---|---|---|
| E1 | **Ollama API changes** | Medium | High | Isolated in one adapter; name-based parsing; `UNSUPPORTED` for missing fields; version-annotated fixtures; nightly live tests | A live test failing after a provider upgrade |
| E2 | **External benchmark projects change or disappear** | High | Medium | Subprocess isolation; pinned versions and dataset hashes; adapters fail loudly; native suites carry the product without them | An adapter failing on a fresh install |
| E3 | **Python or dependency version churn** (3.14 wheels, pydantic/SQLAlchemy majors) | Medium | Medium | Supported matrix 3.12/3.13 with 3.14 as early warning; compatible ranges; lockfiles for CI | The non-blocking 3.14 job failing persistently |
| E4 | **PyPI name unavailable** | Medium | Low | Verify before first publish; documented `aisuite-<name>` fallback with import names unchanged | Name taken at claim time |
| E5 | **FastAPI/pydantic coupling** blocking an upgrade | Low | Medium | Framework confined to `web/`; domain and services import no framework; a swap would touch one layer | A pydantic major that FastAPI lags on |

## 6. Product risks

| # | Risk | L | I | Mitigation | Early signal |
|---|---|---|---|---|---|
| P1 | **Benchmark contamination** — public benchmarks in training data | High | High | Native, unpublished suites carry the core; contamination noted per external suite; refreshed-set adapters listed as future work; users told plainly which suites are public | Implausibly high scores on public suites versus native ones |
| P2 | **Evidence never gathered** — users skip benchmarking, so LoadCoach routes blind | High | Medium | Declared/manual priors; production evidence accrues automatically; `low_evidence` flag; the value of benchmarking is visible in the explanation | Most routing decisions flagged `low_evidence` |
| P3 | **Explanations technically complete but unreadable** | Medium | Medium | The one-minute usability criterion in LoadCoach P8; readable rendering, not a JSON dump | Users asking "so why *did* it pick that?" after reading the page |
| P4 | **Local models too weak for the content pipeline**, making IdeaPress frustrating | Medium | Medium | Deterministic gates catch weak output rather than shipping it; per-stage model bindings; bounded loops with honest stop reasons; `CONTENT_REJECTED` distinguished from failure | Units routinely pausing at the same gate |
| P5 | **Three applications, one person, no users** — the suite is built but never validated by use | Medium | High | The maintainer is the first user; each 1.0 gate requires a full real journey on the reference machine; beta milestones exist to force early use | Milestones passing without anyone having used the product for real work |

---

## 7. Deliberate suite-wide trade-offs

| Trade-off | Given up | Gained |
|---|---|---|
| Nine repositories | Coordination simplicity | Independent installability, versioning and release |
| Sync core, async edge | Theoretical concurrency ceiling | One code path for web and CLI; simple debugging |
| Database-backed queue | Sub-100 ms dispatch guarantees | Zero infrastructure; durable, inspectable state |
| Server-rendered UI | Rich client-side interactivity | No build toolchain; accessible, offline, testable in Python |
| Two database dialects only | Broad database support | Honest, tested portability |
| Local-first with no telemetry | Usage insight | Privacy that needs no explanation |
| Deterministic scoring preferred | Precision on subjective qualities | Numbers that mean something |
| Explanations for every decision | Storage | The core promise of LoadCoach |
| Provenance-heavy records | Storage and required fields | Reproducible comparison |
| Extraction at the second consumer | Early reuse | Abstractions shaped by reality |

## 8. Non-goals restated (suite level)

Not a hosted service. Not a cluster scheduler. Not a training or fine-tuning tool. Not a public
leaderboard. Not a general agent framework. Not multi-tenant. Not dependent on Kubernetes, Redis,
Celery, RabbitMQ or Kafka. Each has been considered and rejected with a recorded reason; adding any
of them requires an ADR that demonstrates a concrete, present need.

## 9. Watch items

Not risks yet, but the design's most likely future pressure points. Each has a named trigger that
would turn it into work:

| Watch item | Trigger that makes it real |
|---|---|
| A shared queue package | FreeWeight needing priorities, leases or multi-worker scheduling |
| `LoadCoachClient` package | A second consumer of LoadCoach's API outside IdeaPress |
| Async ModelRack | More than ~50 concurrent executions, or remote-provider fan-out as a primary use case |
| DuckDB for FreeWeight analytics | Aggregate query budgets missed at realistic volume on SQLite |
| Monorepo consolidation | Sustained cross-repository coordination cost (ADR-0001) |
| Multi-machine execution | A second GPU host in the deployment picture |
| A client-rendered island in IdeaPress | An editing experience the server-rendered approach cannot deliver |
| Prometheus/OpenTelemetry export | An operator running the suite alongside existing monitoring |
