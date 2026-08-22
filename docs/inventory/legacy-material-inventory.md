# Legacy Material Inventory

**Status:** Complete — this inventory closed on 2026-08-21.
**Scope:** everything under `planning/` and `.old_projects/` at the time of the architecture freeze.

This document records what was inspected before the master architecture was written, what was
kept, what was rejected, and why. It exists so that later agents do not "rediscover" an old
decision and reintroduce it. **No document under `planning/` or `.old_projects/` is
authoritative.** Where this inventory and an old document disagree, this inventory wins; where
this inventory and an ADR disagree, the ADR wins.

---

## 1. Material inspected

| Path | Nature | Size / substance | Verdict |
|---|---|---|---|
| `planning/00_executive_summary.md` | Platform vision | 5 KB | Largely adopted |
| `planning/01_detailed_design_and_roadmap.md` | Architecture + roadmap | 16 KB | Adopted with corrections |
| `planning/local_ai_platform_architecture_specs/…` | Byte-identical duplicate of the two files above plus component specs | — | De-duplicated; component specs adopted as seeds |
| `planning/.../components/*.md` | Seven short package specs | 1.3–1.9 KB each | Adopted as seeds, substantially expanded |
| `planning/.../applications/*.md` | Three short app specs | 2.0–2.8 KB each | Adopted as seeds, substantially expanded |
| `planning/old_benchmark_spec.md` | FreeWeight technical spec v1.0 | 3 617 lines | Richest single source; adopted for benchmark content, rejected for platform choices |
| `planning/freeweight_bench_brand_package/` | Brand kit: logos, icons, tokens, 26-section design system | ~850 KB assets | Adopted wholesale as the FreeWeight visual identity and the seed of MirrorWall tokens |
| `.old_projects/local_model_benchmarks/` | Working partial implementation of the benchmark spec (`openweight_bench`, ~6 800 LOC incl. tests) | 6 modules + 4 adapters + 6 test modules | Best code in the corpus; patterns adopted, packaging rejected |
| `.old_projects/content_factory/` | Working content-production framework (~8 300 LOC) | engine, models, webapp, telemetry, 3 content types | IdeaPress's direct ancestor; pipeline adopted, layering rejected |
| `.old_projects/lm_ai_dev/` | ABL (Progress 4GL) code-intelligence agent (~62 000 LOC incl. a 37 000-line generated schema) | agent, benchmark harness, web UI, prompts | Out of suite scope; four patterns harvested |

Empty target directories confirmed at freeze time: `FreeWeight/`, `LoadCoach/`, `IdeaPress/`,
`py/BaseAiCore/`, `py/ModelRack/`, `py/SweatMeter/`, `py/SetSpec/`, `py/WeightsDB/`,
`py/MirrorWall/`. **The suite is a greenfield build.** There is no production data to migrate
and no user-facing behaviour to preserve. This is the single most important fact in this
inventory: it removes every backward-compatibility constraint that the old material assumed.

---

## 2. Concepts adopted

These are load-bearing ideas taken from the old material. Each names the document that now owns it.

### 2.1 From `old_benchmark_spec.md`

| Concept | Why it survives | Now owned by |
|---|---|---|
| **"Unsupported is not zero"** | A benchmark that reports `0 W` when it cannot read power is worse than one that reports nothing. The prior implementation enforced this with a sentinel that refuses arithmetic and truth-testing — a genuinely good design. | [ADR-0016](../adr/0016-unavailable-is-not-zero.md); `BaseAiCore` |
| **Reproducibility fingerprint** | A single hash over model digest + runtime config + backend version + machine fingerprint + benchmark manifest + dataset hashes + prompt versions + app version. Makes "are these two results comparable?" answerable. | [Machine Identity and Reproducibility](../architecture/machine-identity-and-reproducibility.md) |
| **Model ≠ model runtime variant** | The same weights under different context size / KV precision / offload settings are different measurement subjects. | [Canonical Model Identity](../architecture/canonical-model-identity.md) |
| **Objective scoring ladder** | executable → rule-based → reference → human → LLM-judge, in that order of preference. | [FreeWeight Benchmark Catalog](../apps/freeweight/benchmark-catalog.md) |
| **Auditing must measure false positives** | Precision/recall/F1 with a clean-code false-positive rate; a model that flags everything must not score well. | Same |
| **Judge-bias suite** | Position, verbosity, style, repetition, transitivity, self-preference. | Same |
| **Effective vs advertised context** | Depth × position × distractor sweeps, with an explicit, configurable "effective context" threshold. | Same |
| **KV-cache theory vs observation** | Theoretical bytes/token from architecture metadata, compared against an observed VRAM slope. | Same |
| **Token economy retained per sample** | Tokens *and* chars *and* words *and* bytes, because tokenizers are not comparable across models. | Same |
| **Raw samples before aggregates** | Never store only a rolled-up score; every headline number must be drillable to the response that produced it. | [FreeWeight Spec](../apps/freeweight/spec.md) |
| **Persisted run events with replay** | Events are rows; SSE is a view over them; `Last-Event-ID` replays. Survives browser refresh and server restart. | [API and Contract Standards](../standards/api-and-contract-standards.md) |
| **One GPU workload at a time** | Concurrent unrelated work contaminates every performance and memory measurement. | FreeWeight scheduler; [LoadCoach Queue](../apps/loadcoach/queue-and-scheduling.md) |
| **`perf_counter_ns` for durations, wall clock for timestamps** | Separate concerns, never mixed. | [Coding Standards](../standards/coding-standards.md) |
| **External benchmarks run as subprocesses in isolated environments** | Keeps PyTorch/transformers/CUDA dependency conflicts out of the application environment. | [ADR-0018](../adr/0018-external-benchmark-isolation.md) |
| **Never execute model-generated code on the host** | Sandbox or refuse. | [Security Standards](../standards/security-standards.md) |
| **Benchmark manifests with pinned versions and dataset hashes** | A benchmark that silently updates invalidates every historical comparison. | [FreeWeight Benchmark Catalog](../apps/freeweight/benchmark-catalog.md) |
| **Deterministic fake provider built before the real one** | Makes the whole stack testable without a GPU or a model. | [Testing Standards](../standards/testing-standards.md); `ModelRack` |
| **No universal single score** | Category scores plus user-defined weighted profiles; raw measurements always inspectable. | FreeWeight |

### 2.2 From `.old_projects/local_model_benchmarks/`

| Pattern | Where it goes |
|---|---|
| `UNSUPPORTED` singleton that raises on `__bool__`, `__int__`, arithmetic and ordering | `BaseAiCore.measurement` |
| Injectable readers (`/proc` text source, clock, `nvidia-smi` runner) so parsers are tested against captured fixture text | `SweatMeter` |
| `_safe(fn, default=UNSUPPORTED)` wrapper — one failing sensor degrades one field, never the sample | `SweatMeter` |
| Machine fingerprint that deliberately **excludes** driver/toolkit versions and attached storage, so a driver upgrade does not orphan history | `BaseAiCore.MachineProfile.fingerprint` |
| Backend-reported timings and client-observed timings stored **side by side**, never merged | `ModelRack.Timing` |
| `token_level_chunks` capability flag gating whether inter-chunk latency may be called token latency | `ModelRack.ProviderCapabilities` |
| Event store guaranteeing gap-free per-run sequences, subscribe-before-replay, bounded subscriber queues | `MirrorWall.events` + each app's event table |
| SQLite pragmas (`foreign_keys`, WAL, `busy_timeout`) applied at connect time; `BEGIN IMMEDIATE` transaction helper | `WeightsDB` |
| Static-path resolution that rejects anything escaping the static root | `MirrorWall` |
| Capability declaration per backend (`streaming`, `tool_calling`, `structured_output`, `force_unload`, `kv_metrics`, …) | `ModelRack.ProviderCapabilities` |
| Docstrings that state *why* a design constraint exists and cite the requirement | [Coding Standards](../standards/coding-standards.md) |

### 2.3 From `.old_projects/content_factory/`

| Pattern | Where it goes |
|---|---|
| **Python owns control flow; models perform bounded tasks.** Models never decide whether a workflow may terminate. | [IdeaPress Spec](../apps/ideapress/spec.md) |
| **Generator never approves its own output.** Auditors report; the writer repairs. | [IdeaPress Workflows](../apps/ideapress/workflows.md) |
| The correctness **gauntlet**: deterministic validation → bounded repair → fast audit → escalated deep audit → adversarial pass → quality critique → bounded revision → final deterministic validation → commit | Same |
| Quality judge may return **"leave it alone"**; stylistic preference alone does not trigger revision; revision stops on diminishing returns | Same |
| **Requirement compiler**: author intent → machine-checkable, ID'd, blocking/non-blocking requirements carried through every stage | Same |
| **Model catalog + role map**: models declared once, workflow *roles* point at catalog entries, projects override roles without touching code | IdeaPress `stage → model` binding; direct ancestor of LoadCoach task profiles |
| Content-type plug-ins auto-discovered behind one interface; the engine knows units and requirements, not chapters or quests | IdeaPress workflow/content-type registry |
| `contained_path(root, *parts)` proving artifact paths stay inside project roots; validated unit IDs | [Security Standards](../standards/security-standards.md) |
| Exclusive "model lane" job manager so local model work never overlaps | IdeaPress job runner; LoadCoach queue |
| Failure memory — remembering what previously failed to avoid repeating it | IdeaPress (deferred to a later phase, recorded as a future extension) |

### 2.4 From `.old_projects/lm_ai_dev/`

| Pattern | Where it goes |
|---|---|
| Per-task inference settings (model, context size, max output, timeout, temperature, retries, keep-alive) rather than one global model config | LoadCoach task profiles; IdeaPress stage bindings |
| Resumable benchmark runs — every trial checkpointed so an interrupted matrix resumes without repeating completed work | FreeWeight run scheduler |
| Paired candidate-vs-baseline comparison with explicit **gates** (a change must beat the baseline by a stated margin) | FreeWeight comparison + regression detection |
| Snapshot hashes over inputs (source, index, references) recorded on the run so a result is invalidated when its inputs change | FreeWeight provenance; benchmark freshness |
| Hard tool boundary: read-only source tree, writes confined to one directory, no shell, no traversal, symlink escapes rejected | Security Standards |
| System prompts in files, referenced from config (`system_prompt_file`) | [ADR-0012](../adr/0012-prompt-storage-format.md) — extended to full JSON prompt records |

### 2.5 From the brand package

Adopted intact: the FreeWeight logo/icon set, `brand-tokens.json` / `brand-tokens.css`, the type
scale, the 4/8/12/16/24/32/48 spacing scale, radii, the 36 px dense table row, the 48 px header
and 34 px telemetry bar, the chart palette, the light/dark token pairs, the accessibility rules,
the motion budget, and the UI acceptance checklist. These become the seed of
[UI/UX Standards](../standards/ui-ux-standards.md) and of `MirrorWall`'s token layer, with the
brand-specific navy/accent reserved to FreeWeight and per-app accents allowed for LoadCoach and
IdeaPress.

---

## 3. Concepts rejected

Each rejection names the requirement that overrides it and the document that records the replacement.

| Rejected | Source | Reason | Replacement |
|---|---|---|---|
| **"Do not use Flask, FastAPI, … no frontend build system"; hand-rolled `http.server` + manual routing** | `old_benchmark_spec.md` §2.1 | The master requirements mandate an evaluated framework choice, OpenAPI documentation, typed request/response models, and a versioned public API. Hand-rolled routing produced 220 lines of server + 132 lines of dispatch in the prior code before a single benchmark existed, with no schema validation and no generated docs. Minimal dependencies remain a goal; *zero* dependencies is not. | [ADR-0002](../adr/0002-web-framework.md) — FastAPI on Uvicorn |
| **Web-only, no CLI** | All prior projects | Every application must ship both a web UI and a CLI over one service layer. | [CLI Standards](../standards/cli-standards.md) |
| **Client-side SPA with hash routing as the only UI** | `old_benchmark_spec.md` §2.2 | Three applications would each re-implement routing, state and rendering in untyped, untested JavaScript. | [ADR-0020](../adr/0020-ui-rendering-strategy.md) — server-rendered HTML + progressive enhancement |
| **Hand-written `to_dict()` on every dataclass** | `openweight_bench/adapters/base.py`, `telemetry.py` | ~200 lines of pure boilerplate that drifts from its dataclass silently and produces no JSON Schema. | Pydantic v2 models in `SetSpec`; frozen dataclasses stay for pure in-process value types |
| **`field: Any = UNSUPPORTED` typing** | Same | Erases all type information on exactly the fields most likely to be misused. | `Measurement = int | float | Unsupported` used as the real annotation ([ADR-0016](../adr/0016-unavailable-is-not-zero.md)) |
| **Raw `sqlite3` + string-concatenated migrations as the long-term plan** | `openweight_bench/migrations.py`, `abl_benchmark.ensure_schema` | Requirement §15 mandates real migrations, indexes, FKs, migration tests and a clean upgrade path across SQLite **and** PostgreSQL. `CREATE TABLE IF NOT EXISTS` at connect time is not a migration system. | [ADR-0005](../adr/0005-database-strategy.md) — SQLAlchemy 2.0 + Alembic |
| **One monolithic package per application** (`openweight_bench/*`, flat `abl_*.py` modules) | `local_model_benchmarks`, `lm_ai_dev` | No reuse boundary; 931-line telemetry module and 2 103-line orchestrator; a 37 416-line generated `schema.py` sitting in the import path. | `src/` layout, shared packages, explicit service layers |
| **Prompts embedded in Python** (`engine/prompting.py`, 406 lines of f-string prompts; `SYSTEM_PROMPT` constants) | `content_factory`, `lm_ai_dev` | Requirement §12 forbids it. Benchmark comparability additionally requires the *exact* prompt version to be recoverable. | [ADR-0012](../adr/0012-prompt-storage-format.md) — versioned JSON prompt records |
| **Infrastructure importing domain types** (`models/backend.py` → `from engine.types import GenerationContext`) | `content_factory` | This is precisely the dependency inversion the suite forbids; it is why that provider layer could never be reused. | [Dependency and Boundary Rules](../architecture/dependency-and-boundary-rules.md), enforced by an import-linter contract in CI |
| **`except Exception: pass` around lifecycle work** (`ModelManager._notify`, `unload_local_models`) | `content_factory` | Silent swallowing hides provider failures. | Structured exceptions; narrow catches; explicit degraded states |
| **YAML for configuration** (`models.yaml`, `project.yaml`) | `content_factory` | Adds a dependency and a parser with historically surprising semantics; `tomllib` is in the standard library from 3.11. | [ADR-0019](../adr/0019-python-baseline-and-config-format.md) — TOML |
| **JSON for configuration** (`abl_config.json`) | `lm_ai_dev` | No comments, poor human editing ergonomics. | Same |
| **Remote paid providers as first-class engine backends** (`openai_backend.py`, `anthropic_backend.py` wired into the default backend map) | `content_factory` | The suite is local-first: content must stay on the machine unless the user explicitly opts out. Remote providers become explicit, off-by-default configuration. | [Security Standards](../standards/security-standards.md) §"Data egress" |
| **Application-embedded benchmark harness** (`abl_benchmark.py` inside the agent) | `lm_ai_dev` | Benchmarking is FreeWeight's responsibility; embedding it in every consumer duplicates scoring and provenance. | [Traceability Matrix](../architecture/traceability-matrix.md) |
| **`LoadCoachClient` as a seventh shared package now** | `planning/.../components/loadcoach_client_spec.md` | It has exactly one consumer (IdeaPress) and would ship an unstable API surface. The real contract is the versioned HTTP API plus `SetSpec` payload models. | [ADR-0011](../adr/0011-shared-package-boundaries.md) — deferred with an explicit trigger |
| **"Shared DB access as a temporary development shortcut"** | `planning/.../applications/freeweight_integration_spec.md` | Temporary shortcuts across an application boundary become permanent. Requirement §2 forbids it outright. | Boundary rules — export files or HTTP only |
| **Employer/ABL-specific concepts as suite concepts** (`coding.abl`, ABL routine analysis, `.abl-agent` conventions) | `planning/`, `lm_ai_dev` | Employer-specific work stays out of the public suite; the suite only guarantees that its packages are usable by such a tool. | Noted as an out-of-scope consumer in the traceability matrix |
| **"Vanilla JS/CSS must be preserved"** as a MirrorWall design rule | `planning/.../components/mirror_wall_spec.md` | Restated as *"no npm/bundler toolchain and no SPA framework"*, which is the actual intent; "vanilla" as an end in itself blocked server-side templating. | [ADR-0020](../adr/0020-ui-rendering-strategy.md) |
| **Sequential 13-phase platform roadmap with FreeWeight refactor at phase 5** | `planning/01_detailed_design_and_roadmap.md` §14 | It was written for a migration from an existing FreeWeight. There is nothing to migrate; the phases become greenfield build order, and several package phases collapse into their first consumer. | [Master Roadmap](../roadmap/master-roadmap.md) |
| **`psutil` as the telemetry dependency** | `content_factory/telemetry/system_metrics.py` | The `/proc` + `/sys` readers in `local_model_benchmarks` are better: no dependency, exact fixture-testability, and they already handle the sensors this suite needs on its primary platform. | [ADR-0021](../adr/0021-telemetry-collection-strategy.md) |
| **Two parallel telemetry implementations** | `content_factory` vs `local_model_benchmarks` | Direct evidence of the duplication `SweatMeter` exists to eliminate. | `SweatMeter` |

---

## 4. Conflicts found and how they were resolved

| # | Conflict | Resolution | Recorded in |
|---|---|---|---|
| 1 | Old benchmark spec forbids web frameworks; master requirements demand an evaluated framework with OpenAPI. | Master requirements win. Dependency minimalism is retained as a *budget*, not a prohibition. | ADR-0002 |
| 2 | Old benchmark spec's SPA vs. the requirement for three polished, accessible, keyboard-usable UIs sharing primitives. | Server-rendered templates + shared component macros + islands of vanilla JS. No build step either way. | ADR-0020 |
| 3 | Old planning says extract `WeightsDB`/`MirrorWall` "only after duplication is proven"; requirement §15 demands migrations and §25 demands shared UI primitives from the start. | Both honoured: FreeWeight builds `freeweight.storage` and its own UI layer first; the second consumer (LoadCoach) triggers extraction, and FreeWeight adopts the extracted packages in a dedicated phase. Nothing is extracted on speculation. | Master Roadmap; ADR-0011 |
| 4 | Old planning proposes `LoadCoachClient`; requirement §3 lists six shared packages. | Deferred (see rejections). The contract is HTTP + `SetSpec`. | ADR-0011 |
| 5 | `ModelIdentity(provider, provider_model_name, digest)` is too weak to distinguish runtime variants, but the old benchmark spec correctly separates model from runtime config. | Identity stays minimal and immutable; a separate `RuntimeProfile` carries context/KV/offload settings; measurements are keyed on the pair. | ADR-0008 |
| 6 | Old benchmark spec §35 defines a *five-layer* config precedence (app → suite → test → user → run override); requirement §11 defines a *four-layer* precedence (defaults → file → env → CLI). | They are different axes and both survive: §11 governs **application configuration**, the old §35 governs **benchmark execution parameters** resolved *within* a run. Documented explicitly so they are never conflated. | [Configuration Standards](../standards/configuration-standards.md) |
| 7 | Old material variously names the product "OpenWeight Bench" and "FreeWeight" (the specs are byte-identical apart from the name). | **FreeWeight** everywhere. `openweight_bench` is a dead name. | This document |
| 8 | Ollama is described both as "the model provider" and as "one backend among llama.cpp/vLLM/OpenAI-compatible". | Ollama is the only provider implemented in the first releases; the others are interface-compatible extension points with no promised delivery date. | ADR-0007 |
| 9 | `content_factory` treats OpenAI/Anthropic as peers of local providers; the suite is local-first. | Remote providers are supported but require explicit opt-in configuration and are surfaced in the UI as data egress. | Security Standards |
| 10 | Old benchmark spec assumes Docker/Podman for sandboxed code execution; neither is installed on the target machine (`bwrap` is). | Sandbox strategy becomes tiered — container → bubblewrap → **refuse** — and code-execution benchmarks are skipped with an explicit reason rather than silently run on the host. | ADR-0018 |

---

## 5. Technical debt observed (and not inherited)

Recorded so the same mistakes are not repeated:

1. **Test mass without a boundary.** `lm_ai_dev` has ~5 000 lines of tests against modules that
   cannot be reused, because there is no package boundary to test *against*. Test the public
   contract, not the module.
2. **God modules.** `engine/orchestrator.py` (2 103 lines), `webapp/services.py` (1 520 lines),
   `telemetry.py` (931 lines). Each grew because there was no service/domain seam.
3. **Generated code in the import path.** `schema.py` (37 416 lines) had to be parsed rather than
   imported, which the project solved with a bespoke `inject_schema.py`. Generated artifacts
   belong in data files.
4. **Two half-finished UIs.** `content_factory/webapp/static` and `openweight_bench/static`
   (12-line `app.js` stub) — the second was abandoned before the first screen shipped, because
   the UI was scheduled after eight infrastructure phases with nothing demonstrable in between.
   Every phase in this suite's plans must end in something a person can run and see.
5. **Duplicated Ollama clients** in `lm_ai_dev` (`OllamaClient`), `content_factory`
   (`openai_compatible_backend`), and `local_model_benchmarks` (`adapters/ollama.py`) — three
   implementations, three sets of error handling, one bug fix each.
6. **`.env` committed** in `content_factory`. Secrets never enter version control; see Security
   Standards.
7. **`__pycache__`, `.venv/` and `venv/` inside the source tree** of two projects.

---

## 6. Migration concerns

There are none in the usual sense — no user data and no shipped API exist. The only carry-over
obligations are:

1. **Brand assets** must be copied into the FreeWeight repository (`freeweight/web/static/brand/`)
   from `planning/freeweight_bench_brand_package/`, preserving the licence/attribution files
   shipped with the fonts. Nothing else in `planning/` or `.old_projects/` is copied verbatim.
2. **Ollama fixture files** (`.old_projects/local_model_benchmarks/tests/fixtures/ollama/*`) are a
   useful starting corpus for `ModelRack`'s recorded-response tests, but they must be
   re-captured against the currently installed Ollama (0.32.13) before being trusted, and each
   fixture must record the provider version it came from.
3. `.old_projects/` and `planning/` remain on disk as read-only history. They are **not** part of
   any repository created by this suite and must not be added to one.

---

## 7. Environment observed at freeze time

Facts used when setting defaults and performance targets. They describe the primary development
machine, not a requirement on users.

| Property | Value |
|---|---|
| OS | Ubuntu 26.04 LTS (Linux) |
| Python | 3.14.4 default; 3.13.15 also present |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 311 MiB VRAM, driver 580.173.02 |
| Inference provider | Ollama 0.32.13 |
| Containers | Docker **absent**, Podman **absent**, `bwrap` (bubblewrap) present |
| Git | present; `gh` CLI **absent** |
| Package tooling | `pip` 25.1.1; `uv` **absent** |

Consequences recorded in the architecture: the 16 GB VRAM ceiling makes model residency and
context budgeting first-class concerns rather than theoretical ones; the absence of a container
runtime makes the tiered sandbox mandatory; the absence of `gh` means release procedures must not
assume it interactively.
