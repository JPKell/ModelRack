# Master Development Roadmap

**From:** empty repositories (architecture frozen 2026-08-21).
**To:** three professionally deliverable applications and six published packages.
**Corrected 2026-08-21** by the [final architecture audit](../reviews/final_architecture_audit.md):
the prompt library moves from FreeWeight P7 into P6 (the fingerprint needs it), `setspec.prompts` is
extracted at LoadCoach P4 alongside MirrorWall, and LoadCoach P3 gains the VRAM estimator its
constraint filter requires.
**Sequencing principle:** dependency order and rework risk, not calendar dates. No phase is dated,
because a single-maintainer project's calendar is a fiction; every phase instead has prerequisites,
acceptance criteria and an exit condition.

---

## 1. Milestones

| # | Milestone | Content | Exit condition |
|---|---|---|---|
| **M1** | Package foundation | BaseAiCore 0.4 · SetSpec 0.1–0.2 (draft payloads) · ModelRack 0.5 · SweatMeter 0.3 | A script using only these packages discovers a model, generates text, and prints machine telemetry |
| **M2** | FreeWeight beta | FreeWeight P1–P10 | A real model is benchmarked end to end; results are drillable, comparable and exportable |
| **M3** | FreeWeight 1.0-rc · **contract freeze** | FreeWeight P11 · SetSpec 0.3 (schemas frozen, goldens published) | An evidence bundle is consumed by a `setspec`-only harness with no FreeWeight code or DB access |
| **M4** | LoadCoach beta · **extraction complete** | LoadCoach P1–P6 · WeightsDB 0.2 · MirrorWall 0.2 · SetSpec 0.4 (`setspec.prompts`) | LoadCoach routes, executes, streams and imports FreeWeight evidence; two applications share the extracted packages |
| **M5** | LoadCoach 1.0 | LoadCoach P7–P9 | Explainable, durable, secure routing service; published to PyPI |
| **M6** | FreeWeight 1.0 | FreeWeight P12–P14 | FreeWeight on the shared packages, external adapters, hardened; published to PyPI |
| **M7** | IdeaPress beta | IdeaPress P1–P6 | A complete project is produced and exported with **only Ollama** present |
| **M8** | IdeaPress 1.0 | IdeaPress P7–P9 | Optional LoadCoach backend; hardened; published to PyPI |
| **M9** | Suite 1.0 | Integration verification, cross-repository CI, documentation set, public release | Every gold standard met; all install paths verified; release notes published |

---

## 2. Dependency graph

```mermaid
graph TD
    BC["BaseAiCore 0.4"] --> SS["SetSpec 0.1–0.2"]
    BC --> MR["ModelRack 0.5"]
    BC --> SM["SweatMeter 0.3"]
    SS --> FW1["FreeWeight P1–P10<br/>M2 beta"]
    MR --> FW1
    SM --> FW1
    FW1 --> FW2["FreeWeight P11<br/>M3 1.0-rc"]
    FW2 --> SSF["SetSpec 0.3<br/>schemas frozen"]
    SSF --> FW2
    FW2 --> LC1["LoadCoach P1–P6<br/>M4 beta"]
    LC1 --> WDB["WeightsDB 0.2<br/>(extracted at LC-P1)"]
    LC1 --> MW["MirrorWall 0.2<br/>(extracted at LC-P4)"]
    LC1 --> SSP["SetSpec 0.4<br/>setspec.prompts<br/>(extracted at LC-P4)"]
    SSP --> FW3
    SSP --> IP1
    LC1 --> LC2["LoadCoach P7–P9<br/>M5 1.0"]
    WDB --> FW3["FreeWeight P12–P14<br/>M6 1.0"]
    MW --> FW3
    WDB --> IP1["IdeaPress P1–P6<br/>M7 beta"]
    MW --> IP1
    MR --> IP1
    LC2 --> IP2["IdeaPress P7–P9<br/>M8 1.0"]
    IP1 --> IP2
    FW3 --> SUITE["M9 Suite 1.0"]
    LC2 --> SUITE
    IP2 --> SUITE
```

The one non-obvious edge is **FreeWeight P11 → SetSpec 0.3 → FreeWeight P11**: the schemas are frozen
only after FreeWeight has produced real results against the draft models, and FreeWeight's evidence
export then ships against the frozen schemas. Freezing a contract before its producer exists is a
guess; this ordering makes it an observation.

---

## 3. Work streams and what can proceed in parallel

Four streams. Within a stream, phases are strictly ordered; across streams, the table states what may
overlap.

| Stream | Contents |
|---|---|
| **A — Foundation packages** | BaseAiCore, SetSpec, ModelRack, SweatMeter |
| **B — FreeWeight** | FreeWeight P1–P14 |
| **C — LoadCoach + extractions** | LoadCoach P1–P9, WeightsDB, MirrorWall |
| **D — IdeaPress** | IdeaPress P1–P9 |

```mermaid
gantt
    dateFormat X
    axisFormat %s
    title Sequencing by dependency (units are phases, not time)

    section A Foundation
    BaseAiCore P1-P4         :a1, 0, 4
    SetSpec P1-P3            :a2, after a1, 3
    ModelRack P1-P5          :a3, after a1, 5
    SweatMeter P1-P4         :a4, after a1, 4
    SetSpec P4 freeze        :a5, 15, 1

    section B FreeWeight
    FW P1-P2 skeleton+storage :b1, after a1, 2
    FW P3-P4 models+telemetry :b2, after a3, 2
    FW P5-P9 engine+benchmarks:b3, after b2, 5
    FW P10 UI  (M2)           :b4, after b3, 1
    FW P11 evidence (M3)      :b5, after b4, 1
    FW P12-P14 adopt+ext (M6) :b6, after c2, 3

    section C LoadCoach
    LC P1-P4 + WeightsDB + MirrorWall (M4 pt1) :c1, after b5, 4
    LC P5-P6 queue+evidence (M4)               :c2, after c1, 2
    LC P7-P9 feedback+UI+harden (M5)           :c3, after c2, 3

    section D IdeaPress
    IP P1-P6 standalone (M7)  :d1, after c1, 6
    IP P7-P9 loadcoach (M8)   :d2, after c3, 3

    section Suite
    M9 Suite 1.0              :e1, after b6, 2
```

### 3.1 Explicit parallelism rules

| These may run concurrently | Because |
|---|---|
| ModelRack P1–P5 and SweatMeter P1–P4 | Both depend only on BaseAiCore; no shared surface |
| SetSpec P1–P3 and ModelRack/SweatMeter | SetSpec does not depend on either |
| FreeWeight P1–P2 and ModelRack P3–P5 | FreeWeight's skeleton and storage need no provider |
| FreeWeight P12–P14 and LoadCoach P7–P9 | Different repositories; FreeWeight P12 needs only the *published* WeightsDB/MirrorWall |
| IdeaPress P1–P6 and LoadCoach P5–P9 | IdeaPress standalone needs no LoadCoach, only the extracted packages |
| IdeaPress P1–P6 and FreeWeight P12–P14 | Entirely independent |
| Documentation and hardening within any application's final phases | Different files, same acceptance gate |

| These may **not** overlap | Because |
|---|---|
| FreeWeight P11 and SetSpec P4 (freeze) | Circular by design; sequence is draft → real results → freeze → export |
| LoadCoach P1 and FreeWeight's storage refactor | WeightsDB is extracted *from* FreeWeight; FreeWeight must be stable first |
| MirrorWall extraction and FreeWeight UI changes | The extraction is a move, not a copy; a moving target breaks it |
| IdeaPress P7 and LoadCoach P1–P9 | The LoadCoach backend requires a stable, released LoadCoach API (M5) |
| Any two GPU-bound work streams on the reference machine | One GPU; benchmark measurements are invalid when shared |

### 3.2 The single-maintainer reality

With one person, "parallel" means *unblocked*, not *simultaneous*. The practical ordering that
minimizes context switching is: finish stream A; drive stream B to M3; drive stream C to M4; then
alternate between B (P12–P14) and D (P1–P6) as each hits a natural pause; finish C to M5; finish D;
then M9. The parallelism table above matters mainly for deciding what to pick up when something is
blocked — for example, when a live benchmark run is occupying the GPU for an hour.

---

## 4. Integration milestones

Points where two components must actually work together. Each has a dedicated verification, and none
is considered complete on the basis of a code review.

| # | Integration | At | Verification |
|---|---|---|---|
| **I1** | FreeWeight ↔ ModelRack | FW P3 | Discovery through ModelRack only; no provider HTTP code in FreeWeight (asserted) |
| **I2** | FreeWeight ↔ SweatMeter | FW P4 | Telemetry bar live; machine profile persisted; no-GPU path exercised |
| **I3** | FreeWeight → SetSpec | FW P6, frozen at FW P11 | Exported results validate against schemas and goldens |
| **I4** | FreeWeight → LoadCoach (evidence) | LC P6 | A bundle produced by FreeWeight changes LoadCoach routing, verified with **no shared code and no shared database** |
| **I5** | WeightsDB ↔ two applications | LC P1, FW P12 | Two schemas, two migration histories, one package; FreeWeight's test suite passes unchanged after adoption |
| **I6** | MirrorWall ↔ two applications | LC P4, FW P12 | Both applications' template suites render against the same version in CI |
| **I7** | IdeaPress ↔ LoadCoach | IP P7 | Backend switch changes no workflow code; degradation and version mismatch handled; feedback lands in LoadCoach's reliability stats; every task ID in `LOADCOACH_TASK_MAP` exists in the running LoadCoach's `/task-profiles`; the prompt LoadCoach forwards equals the prompt IdeaPress rendered |
| **I9** | Prompt hashing across components | LC P4, FW P12 | The same prompt record hashes identically under FreeWeight's, LoadCoach's and IdeaPress's installed `setspec`, and FreeWeight's pack hashes unchanged across the adoption |
| **I8** | Full suite | M9 | All three running together on one machine; every optional link exercised on and off |

---

## 5. Stabilization phases

Stabilization is scheduled work, not what happens if there is time left.

| Phase | When | Content | Gate |
|---|---|---|---|
| **S1 — Foundation stabilization** | End of M1 | Package APIs reviewed against their first real consumer; breaking changes made now while everything is `0.x`; golden values locked | Every package installs alone, type-checks from a consumer, ≥ 95 % coverage |
| **S2 — FreeWeight stabilization** | FW P14 (M6) | Performance budgets, security checklist, accessibility audit, upgrade testing from every released version, documentation | All FreeWeight acceptance criteria and gold standards met |
| **S3 — LoadCoach stabilization** | LC P9 (M5) | Auth and LAN-exposure review, scheduling simulation at scale, security checklist, operations documentation | All LoadCoach acceptance criteria and gold standards met |
| **S4 — IdeaPress stabilization** | IP P9 (M8) | Model-output sanitization sweep, archive-import hardening, performance, documentation | All IdeaPress acceptance criteria and gold standards met |
| **S5 — Suite stabilization** | M9 | Cross-repository compatibility matrix, install-path verification, documentation consistency review, dependency audit, release notes, **and the package-1.0 range widening** (every application needs a release whose ranges admit the 1.0 packages — see [Packaging Standards §4](../standards/packaging-and-release-standards.md)) | Every item in §7 checked |

---

## 6. Version trajectory

| Component | M1 | M3 | M4 | M5 | M6 | M8 | M9 |
|---|---|---|---|---|---|---|---|
| BaseAiCore | 0.4 | 0.4–0.5 | 0.5 | 0.5 | 0.6 | 0.6 | **1.0** |
| SetSpec | 0.2 | **0.3** (frozen) | 0.3 | 0.4 | 0.4 | 0.5 | **1.0** |
| ModelRack | 0.5 | 0.5 | 0.6 | 0.6 | 0.7 | 0.7 | **1.0** |
| SweatMeter | 0.3 | 0.3 | 0.4 | 0.4 | 0.4 | 0.4 | **1.0** |
| WeightsDB | — | — | **0.2** | 0.2 | 0.3 | 0.3 | **1.0** |
| MirrorWall | — | — | **0.2** | 0.3 | 0.3 | 0.4 | **1.0** |

SetSpec's M4 column is **0.4**, not 0.3: `setspec.prompts` is extracted during LoadCoach P4
([ADR-0028](../adr/0028-prompt-pack-granularity.md)). The schema freeze at M3 is unaffected —
prompt tooling is additive and the frozen payload schemas do not change.
| FreeWeight | — | 1.0-rc | 1.0-rc | 1.0-rc | **1.0** | 1.0.x | 1.1 |
| LoadCoach | — | — | 0.9-beta | **1.0** | 1.0.x | 1.0.x | 1.1 |
| IdeaPress | — | — | — | — | 0.9-beta | **1.0** | 1.0.x |

Packages reach 1.0 only at M9, when all three applications have exercised them. Applications reach
1.0 when their own acceptance criteria pass — an application at 1.0 depending on a `0.x` package is
deliberate and honest, and the compatible-range pinning in
[Packaging Standards](../standards/packaging-and-release-standards.md) makes it safe.

---

## 7. M9 — Professional delivery checklist

Nothing here is optional; each maps to requirement §37.

**Installation and distribution**
- [ ] `pip install freeweight|loadcoach|ideapress` into a clean venv, each starting with zero configuration
- [ ] `pipx install` verified for all three
- [ ] All six packages installable and importable standalone
- [ ] `python -m <app>` works for all three
- [ ] Optional extras (`[postgres]`) install and function

**Releases**
- [ ] Every component released from a tag by CI with Trusted Publishing; no manual upload has ever occurred
- [ ] Semantic versions, changelogs and release notes for every component
- [ ] Compatibility matrix published per application (tested package ranges)
- [ ] Checksums published for application artifacts

**Documentation**
- [ ] README per repository with purpose, install, quickstart and links
- [ ] Configuration reference per application, generated and CI-diff-checked
- [ ] API documentation per application: OpenAPI snapshot plus a written guide
- [ ] `--help` complete and correct at every CLI level
- [ ] Web UI help/about page per application
- [ ] Troubleshooting guide per application, aligned with `<app> doctor`
- [ ] Security documentation: trust boundaries, exposure, egress, sandboxing
- [ ] Backup and restore procedure per application, tested
- [ ] Upgrade guide from every released version; rollback considerations documented
- [ ] Developer documentation and `CONTRIBUTING.md` per repository
- [ ] This documentation set reviewed for consistency (§8) and published

**Quality**
- [ ] Every gold standard in [Gold Standards](../standards/gold-standards.md) met and measured
- [ ] Coverage floors met in every repository
- [ ] Performance budgets measured on the reference machine and published with the machine described
- [ ] Security checklist complete; `pip-audit` and `gitleaks` clean
- [ ] Accessibility checklist complete for all three UIs
- [ ] Cross-repository compatibility matrix green

**Operations**
- [ ] Migration path tested from every released version with real data
- [ ] Downgrade procedure exercised: upgrade, write data, restore the pre-migration backup, start the
      older version — and a database ahead of the code refuses with `SchemaAhead` naming both revisions
- [ ] Every application's dependency ranges admit the 1.0 packages, verified by a clean-venv resolve
- [ ] Backup/restore tested on both dialects
- [ ] `<app> doctor` diagnoses every documented failure mode
- [ ] Degradation matrix exercised end to end for all three applications

---

## 8. Documentation consistency review (repeated before every milestone)

The review in [§9 of this roadmap](#9-current-state-and-immediate-next-steps) is run at each
milestone, not only at M9. It checks: component names, public contracts, model identity terms,
configuration precedence, API conventions, database ownership, no cross-application DB access, no
package importing an application, each application independently runnable, the optional links, tests
planned before implementation, acceptance criteria present in every phase, and rationale recorded for
every architectural decision. Any drift is fixed in the documentation before the milestone is
declared.

---

## 9. Current state and immediate next steps

**Current state:** architecture frozen; `docs/` complete; all nine repositories empty.

The first three actions, in order:

1. **Create the BaseAiCore repository** and execute
   [BaseAiCore Phase 1](../packages/baseaicore/development-plan.md#phase-1--measurement-identity-time-and-cost-primitives).
   Everything else is blocked on it.
2. **Establish the shared CI workflow template** (format, lint, types, import-linter, tests,
   coverage, security, build, install-check) once, and copy it into each repository as it is created.
3. **Claim the distribution names** on PyPI, or record the fallback naming decision in
   [ADR-0015](../adr/0015-repository-and-distribution-model.md).

Before any of the three, read the [final architecture audit](../reviews/final_architecture_audit.md):
it added ADR-0022 – ADR-0029 and corrected the specifications they touch, and BaseAiCore Phase 1 now
carries golden tests for the canonical-ID format and digest normalization that the earlier text left
ambiguous.

An implementation agent assigned any phase should read, in this order: the master requirements, the
[Master Architecture](../architecture/master-architecture.md), the relevant component
[specification](../README.md), that phase in the component's development plan, and the standards it
touches. It should not need to invent an architectural decision; if it does, that gap is a defect in
this documentation set and should be closed with an ADR before the code is written.
