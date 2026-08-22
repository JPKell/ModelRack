# Executive Summary

**Suite:** a local-first toolkit for operating open-weight AI models.
**Components:** three applications (FreeWeight, LoadCoach, IdeaPress) and six shared Python packages.
**Status:** architecture frozen 2026-08-21, audited and corrected the same day
([final architecture audit](../reviews/final_architecture_audit.md), ADR-0022 – ADR-0029).
Implementation has not started.

---

## 1. Purpose

Running open-weight models locally is cheap, private and fast — and almost entirely unmeasured.
Users pick models by reputation, guess at context limits until something out-of-memories, and
discover a model's weaknesses in production. This suite closes that loop with three products that
each stand alone and compose when combined:

```text
Measure AI   →   Manage AI   →   Apply AI
FreeWeight       LoadCoach       IdeaPress
```

* **FreeWeight** measures what a model can actually do *on this machine* — capability, speed,
  memory, reliability — and records enough provenance that the measurement can be reproduced and
  compared.
* **LoadCoach** turns those measurements into decisions: given a task, the installed models, the
  current state of the hardware and a queue of work, it chooses a model, executes the request, and
  can explain exactly why it chose what it chose.
* **IdeaPress** applies models to real work — turning an idea into finished content through
  configurable, validated workflows in which Python owns the control flow and models perform
  bounded tasks.

## 2. Suite vision

A user should be able to install any one of these three applications, run one command, and have a
working local AI tool that never phones home. A user who installs two should get more than the sum:
benchmark evidence makes routing intelligent, and intelligent routing makes content workflows
faster, cheaper and more reliable — without a single line of workflow code changing.

## 3. Component overview

### Applications

| Application | Answers | Owns | Default port |
|---|---|---|---:|
| **FreeWeight** | "How well does this model perform on this machine, for this capability?" | Benchmark definitions, execution, scoring, result history, provenance, evidence export | 8765 |
| **LoadCoach** | "Given this task and this machine right now, which model should run it, and how?" | Task profiles, routing, queue, execution, validation, retries/fallback, routing explanations, production feedback | 8766 |
| **IdeaPress** | "How do I turn this idea into finished content?" | Workflow definitions, projects, stages, drafts, validation gates, exports | 8767 |

### Shared packages

| Package | Provides | First consumer |
|---|---|---|
| **BaseAiCore** | Canonical model identity, machine profile, capability IDs, the `Unsupported` measurement sentinel, IDs, timestamps, domain errors | All |
| **SetSpec** | Versioned cross-application schemas: benchmark results, capability evidence, machine profiles, event and error envelopes | FreeWeight |
| **ModelRack** | Provider-neutral model discovery, metadata, generation and streaming (Ollama first) | FreeWeight |
| **SweatMeter** | CPU/RAM/GPU/VRAM/thermal/power telemetry and static machine profiling | FreeWeight |
| **WeightsDB** | SQLAlchemy engine/session setup, SQLite pragmas, migration runner, backup, health checks | LoadCoach (then adopted by FreeWeight) |
| **MirrorWall** | Design tokens, layout and component macros, SSE helpers, JSON/error envelopes, request IDs, telemetry widgets | LoadCoach (then adopted by FreeWeight) |

## 4. Dependency model

Dependencies point one way only: applications depend on packages; packages depend on
infrastructure; nothing depends on an application.

```mermaid
graph TD
    FW[FreeWeight]:::app
    LC[LoadCoach]:::app
    IP[IdeaPress]:::app

    MR[ModelRack]:::pkg
    SM[SweatMeter]:::pkg
    SS[SetSpec]:::pkg
    WD[WeightsDB]:::pkg
    MW[MirrorWall]:::pkg
    BC[BaseAiCore]:::pkg

    FW --> MR & SM & SS & WD & MW & BC
    LC --> MR & SM & SS & WD & MW & BC
    IP --> MR & SS & WD & MW & BC
    MR --> BC
    SM --> BC
    SS --> BC
    WD --> BC
    MW --> BC

    IP -. "optional, HTTP" .-> LC
    FW -. "evidence export, HTTP or file" .-> LC

    classDef app fill:#2F80ED,stroke:#1D6FDB,color:#fff
    classDef pkg fill:#EEF2F7,stroke:#D8E0E8,color:#0E1823
```

Solid arrows are Python imports. Dotted arrows are **optional** connections over versioned HTTP
contracts, never imports and never shared databases.

## 5. Independent deployment

Every application is independently installable, independently versioned, and independently useful:

| Scenario | Works? | What you get |
|---|---|---|
| FreeWeight alone | Yes | Full benchmarking, comparison, export |
| LoadCoach alone | Yes | Routing from declared model capabilities and live resources; no benchmark evidence |
| IdeaPress alone | Yes | Full content workflows against Ollama or any OpenAI-compatible endpoint |
| FreeWeight + LoadCoach | Yes | Routing backed by measured evidence, with confidence and freshness weighting — for the runtime profile the evidence was measured under, which LoadCoach names in the explanation when it differs |
| IdeaPress + LoadCoach | Yes | Automatic model selection, queueing, validation and fallback per workflow stage |
| All three | Yes | Measured → managed → applied, end to end |
| IdeaPress + FreeWeight, no LoadCoach | Yes | They simply do not interact; IdeaPress never requires FreeWeight |

Three rules make this hold, and CI enforces them:
1. No application imports another application's Python modules.
2. No application reads or writes another application's database.
3. All cross-application traffic uses a versioned public API or a versioned exported file.

## 6. Major benefits

* **Local-first.** Default binding is `127.0.0.1`; no content leaves the machine unless the user
  explicitly configures a remote provider, and the UI marks that configuration as data egress.
* **Evidence over folklore.** Model choice becomes a measurement with provenance, sample counts,
  confidence and an expiry, rather than an opinion.
* **Explainable routing.** Every LoadCoach decision retains its candidates, scores, hard-constraint
  rejections and fallback ordering.
* **Reproducible measurement.** Every benchmark result carries the model digest, runtime profile,
  provider version, machine fingerprint, benchmark version and dataset hashes needed to reproduce
  or invalidate it.
* **Honest degradation.** A missing GPU, an absent provider, an unreadable sensor or stale evidence
  produces an explicit state — never a fabricated zero and never a crash.
* **Genuine reuse.** One Ollama client, one telemetry implementation, one model identity, one set of
  UI primitives, shared through installable versioned packages — not copies and not submodules.

## 7. Development order

Packages are built when the first application actually needs them, and every phase ends in
something demonstrable.

```mermaid
graph LR
    A[BaseAiCore + SetSpec] --> B[ModelRack + SweatMeter]
    B --> C[FreeWeight 1.0]
    C --> D[WeightsDB + MirrorWall extraction]
    D --> E[LoadCoach 1.0]
    E --> F[IdeaPress 1.0]
    F --> G[Suite integration + public release]
```

Named release milestones: **M1** package foundation · **M2** FreeWeight beta · **M3** FreeWeight
1.0-rc and contract freeze · **M4** LoadCoach beta (WeightsDB and MirrorWall extracted) · **M5**
LoadCoach 1.0 · **M6** FreeWeight 1.0 · **M7** IdeaPress beta · **M8** IdeaPress 1.0 (optional
LoadCoach backend) · **M9** suite 1.0 public release. The
[Master Roadmap](../roadmap/master-roadmap.md) defines each milestone's content, its exit criteria,
and what can proceed in parallel.

## 8. What this suite deliberately is not

* Not a model training, fine-tuning or quantization tool.
* Not a multi-tenant hosted service, and not a cluster scheduler. Single-machine first; a second
  machine is an explicit future extension, not a hidden assumption.
* Not a general agent framework. IdeaPress runs *bounded* model tasks inside Python-owned control flow.
* Not a leaderboard. FreeWeight measures *your* models on *your* hardware and refuses to collapse
  that into one universal number.
* Not dependent on Kubernetes, Redis, Celery, RabbitMQ or Kafka. Nothing in the current
  requirements justifies them; see [ADR-0010](../adr/0010-queue-implementation.md).
