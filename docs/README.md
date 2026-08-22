# ModelRack — Local Documentation

**This is a curated copy**, not the source of truth. It was generated from the suite's central
`ai-suite-docs` repository so that `modelrack` can be read, reviewed and implemented **without**
checking out the other eight repositories. If this documentation set and the central one ever
disagree, the central `ai-suite-docs` repository wins — treat a disagreement here as staleness to be
refreshed, not as a second opinion to reconcile.

The directory layout below **mirrors the master `docs/` tree exactly** (`apps/<name>/`,
`packages/<name>/`, `standards/`, `adr/`, `architecture/`, `roadmap/`), so every relative link inside
a copied document (`../../adr/...`, `../../standards/...`) resolves correctly here without rewriting.

---

## Start here

1. **This component's own specification**, in order:
- [development-plan.md](packages/modelrack/development-plan.md)
- [spec.md](packages/modelrack/spec.md)
2. **Standards it must follow** — [standards/](standards/), particularly `coding-standards.md`,
   `testing-standards.md`, `security-standards.md`, `api-and-contract-standards.md` and
   `configuration-standards.md`.
3. **Architecture Decision Records referenced from the development plan** — [adr/](adr/README.md).
4. **Cross-cutting architecture** — [architecture/](architecture/master-architecture.md), especially
   `dependency-and-boundary-rules.md` (what this component may and may not import) and
   `graceful-degradation.md` (what happens when a dependency is unavailable).

## What's included, and why

### This component
- [development-plan.md](packages/modelrack/development-plan.md)
- [spec.md](packages/modelrack/spec.md)

### Package dependencies (`spec.md` — the contract; a development plan only where this component
### originates that package as an extraction, per ADR-0011)
- [baseaicore/spec.md](packages/baseaicore/spec.md) — the contract this component depends on

### Suite-wide standards (13 files)
Every file in [standards/](standards/) — these apply to all nine repositories identically.

### Architecture Decision Records (30 files)
The complete ADR set in [adr/](adr/README.md), including the eight added by the post-freeze audit
(0022–0029) and the seven amended by it. A component only *acts on* the ADRs its own spec and
development plan reference, but the full set is included because ADRs cross-reference each other and
a partial set would have dangling links.

### Cross-cutting architecture (9 files)
The complete [architecture/](architecture/master-architecture.md) set: canonical model identity,
machine identity and reproducibility, dependency and boundary rules, graceful degradation,
performance targets, the traceability matrix, the risk register and the executive summary.

### Roadmap
[roadmap/master-roadmap.md](roadmap/master-roadmap.md) (milestones, sequencing, integration points)
and [roadmap/model-assignment.md](roadmap/model-assignment.md) (which model/effort tier fits which
phase — advisory, not normative).

### Inventory and audit history
[inventory/legacy-material-inventory.md](inventory/legacy-material-inventory.md) — what was kept and
rejected from the prior projects, and why; useful when a design choice here looks surprising.
[reviews/final_architecture_audit.md](reviews/final_architecture_audit.md) — the pre-implementation
audit that produced ADR-0022–0029; read it if a document here references one of those and you want the
finding that motivated it.

## What's deliberately *not* here

* Other applications' internal development plans, data models or risk analyses — this component does
  not need them, and per the dependency and boundary rules, must never import their code.
* **A consequence you will notice:** the suite-wide documents in `architecture/` (particularly
  `traceability-matrix.md` and `risk-register.md`), `adr/` and `roadmap/` are written to be read as
  part of the *whole* suite, so a handful of their links point at other applications' internal
  documents (another app's `data-model.md`, `risks.md`, `routing.md`, `workflows.md`,
  `queue-and-scheduling.md`) that are deliberately not copied here. Those links will not resolve from
  inside this repository — that is expected, not a copying error. If you need one, it means you are
  looking at a cross-cutting document for suite-level context, not for this component's own contract;
  the central `ai-suite-docs` repository has the full set.
* Other packages' development plans (only their `spec.md` is included, when this component depends on
  them) — the *contract* is what matters to a consumer, not how the package's own team builds it.
* Generated deliverables this component will produce later in its own hardening phase — a
  configuration reference, an OpenAPI snapshot, a quickstart guide, a troubleshooting guide. Those are
  authored *by this repository* during its own development plan and will live directly under `docs/`
  (e.g. `docs/quickstart.md`) alongside — not inside — the copied material above.

## Keeping this in sync

This snapshot can drift from the central `ai-suite-docs` repository as both evolve. Re-generate it
rather than hand-editing the copied files; hand edits to anything outside this README will be
overwritten the next time the snapshot is refreshed, and any correction that belongs in the source of
truth should go to `ai-suite-docs` directly.
