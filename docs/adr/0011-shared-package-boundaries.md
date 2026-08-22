# ADR-0011 — Shared package boundaries and extraction timing

**Status:** Accepted (2026-08-21)
**Amended by:** [ADR-0028](0028-prompt-pack-granularity.md) — adds prompt tooling to the extraction schedule, at the same second-consumer trigger.

## Context

Six shared packages are specified. Two competing failure modes exist: extracting too early (an
abstraction shaped by one consumer, frozen before the second reveals what it actually needs) and
extracting too late (three divergent copies — the documented history of the prior projects, which
had three Ollama clients and two telemetry implementations).

The requirements say both "do not extract abstractions prematurely" and "each application must have
migrations, shared UI primitives, and a database from the beginning". Resolving that tension is this
ADR's job.

## Decision

**Extraction timing is decided per package by how many consumers are already known, and by whether
the interface is discoverable without a second consumer.**

| Package | Built | Rationale |
|---|---|---|
| **BaseAiCore** | Up front, before any application | Its types cross every boundary; there is no version of this suite in which model identity is application-local |
| **SetSpec** | Up front | A contract package by definition; the whole point is to be shared |
| **ModelRack** | Up front | Three known consumers; the interface is dictated by provider APIs, not by any one application |
| **SweatMeter** | Up front | Two known consumers, an obvious interface, and the prior duplication is documented |
| **WeightsDB** | **Extracted at LoadCoach P1**, from FreeWeight's `freeweight.storage` | Storage plumbing looks generic and is not — pragmas, session policy, migration ergonomics only become clear with a second schema. FreeWeight adopts it in a dedicated later phase |
| **MirrorWall** | **Extracted at LoadCoach P4**, from FreeWeight's web layer | Same reasoning, more strongly: a component library designed against one UI is a theme, not a library |
| **Prompt tooling** (`setspec.prompts`) | **Extracted at LoadCoach P4**, from FreeWeight's `freeweight.services.prompts` | Three applications each need a record loader, a `StrictUndefined` renderer and canonical hashing; the hashes appear in cross-application evidence, so three implementations would be three chances for a determinism contract to disagree. Into SetSpec rather than a seventh package ([ADR-0028](0028-prompt-pack-granularity.md)) |

Rules that make deferred extraction safe rather than a slogan:

1. FreeWeight builds `freeweight/infrastructure/db/` and `freeweight/web/ui/` **as if they were
   packages** — no benchmark vocabulary in the plumbing, no run-specific concepts in the components.
2. The extraction phase is a named, scheduled phase in the roadmap with its own acceptance criteria,
   not an aspiration.
3. FreeWeight's adoption of the extracted packages is likewise a named phase, and the extraction is
   not "done" until both applications run on the shared package.
4. Nothing is extracted with fewer than two real consumers.

**`LoadCoachClient` is not created.** The old planning proposed it as a seventh package. It has one
consumer (IdeaPress), and the real contract is the versioned HTTP API plus SetSpec payload models.
IdeaPress implements a thin `LoadCoachBackend` adapter (~200 lines) over `httpx`. This is recorded
as a deliberate rejection, not an oversight.

## Alternatives considered

**Extract everything up front.** Rejected: WeightsDB and MirrorWall would be designed against a
single consumer and frozen. The cost of a wrong abstraction is higher than the cost of one extraction
phase.

**Extract nothing; let applications duplicate.** Rejected: this is the documented prior failure.

**Ship `LoadCoachClient` now.** Rejected: an unstable API surface published for one consumer, and a
package whose only job is to wrap eight HTTP calls.

**A single `aisuite-common` package containing everything shared.** Rejected: it forces every
consumer to install a web framework, an ORM and an HTTP client to get a model identity type, and it
destroys the layering that keeps BaseAiCore dependency-free.

## Consequences

*Positive.* Packages that ship are shaped by at least two real consumers. FreeWeight is not blocked
waiting for infrastructure it can write in place. The deferred packages arrive with real, tested
requirements. The dependency budget stays small at every layer.

*Negative.* FreeWeight is touched twice: once to build the in-app version, once to adopt the
extracted package. This is planned work with its own phase, and the second pass is mostly deletion.
Between extraction and adoption, two implementations coexist briefly — bounded by the roadmap
ordering and by the rule that extraction is not complete until both consumers run on the package.

*Negative.* If IdeaPress ever gains a sibling that also calls LoadCoach, the client adapter will be
duplicated. That is the trigger below.

## Revisit when

* A second consumer of the LoadCoach HTTP API appears outside IdeaPress → create `LoadCoachClient`,
  extracted from IdeaPress's adapter and from the OpenAPI document.
* A third application needs storage, UI or prompt plumbing before the scheduled extraction phases →
  pull those phases forward.
* An extracted package accumulates application-specific concepts → that is a boundary violation; the
  concept moves back into the application.
