# ADR-0015 — Repository and distribution model

**Status:** Accepted (2026-08-21)

## Context

[ADR-0001](0001-application-and-package-separation.md) establishes nine independently versioned
components. This ADR fixes how they are stored, built, named and shipped. The requirements are
explicit: independent Git repositories, no submodules as the dependency mechanism, editable installs
during development, PyPI for stable reusable packages, semantic versioning, compatible version
ranges rather than an unversioned `main`.

## Decision

**One Git repository per component; `src/` layout; hatchling; PyPI via Trusted Publishing; dependency
ranges only.**

1. Nine repositories, plus a tenth (`ai-suite-docs`) holding this documentation set. The workspace
   directory that contains them is not itself a repository and nothing depends on paths outside a
   repository.
2. `src/` layout everywhere, so tests exercise the installed distribution and packaging errors
   surface immediately.
3. Build backend: **hatchling** — PEP 621-native, straightforward package-data handling (MirrorWall's
   templates and static files, SetSpec's JSON Schemas, prompt packs, Alembic scripts).
4. Version in one place (`src/<pkg>/__about__.py`), asserted by a test to match the Git tag and the
   built metadata.
5. Distribution names equal import names: `baseaicore`, `setspec`, `modelrack`, `sweatmeter`,
   `weightsdb`, `mirrorwall`, `freeweight`, `loadcoach`, `ideapress`. **Availability is verified
   before first publish**; the documented fallback is `aisuite-<name>` with the import name
   unchanged.
6. Applications depend on packages by compatible range (`>=0.4,<0.5` pre-1.0; `>=1.2,<2` after),
   never on a Git URL or branch. Local development uses editable installs; CI never does.
7. Releases are produced only by CI from an annotated tag, published to PyPI with Trusted Publishing
   (OIDC). No long-lived API tokens exist anywhere.
8. Licence: Apache-2.0 for every component.

## Alternatives considered

**Monorepo.** See ADR-0001 — the strongest alternative, rejected there for independent versioning
and install ergonomics, with the revisit trigger recorded.

**Git submodules.** Explicitly rejected by the requirements and by experience: pointer rot, forgotten
`--recursive`, no version resolution.

**Vendoring shared code into each application.** Rejected: the documented prior failure.

**Private index / Git+SSH dependencies.** Rejected: users must be able to `pip install freeweight`
from PyPI.

**`setuptools` or `poetry-core` as the build backend.** Both workable. Hatchling chosen for simpler
package-data configuration and PEP 621 nativeness; the decision is low-stakes and reversible.

**Flat layout (package at the repository root).** Rejected: it lets tests import the source tree
rather than the installed distribution, which is how missing package data ships to users.

**A namespace package (`aisuite.modelrack`).** Rejected: it complicates independent installation and
tooling for no gain, and makes the import name longer at every call site.

## Consequences

*Positive.* Each component is independently versioned, released and installable. Third parties can
depend on one package without the rest. `src/` plus an install-check job catches packaging mistakes
before users do. Trusted Publishing removes the largest supply-chain risk in the release path.

*Negative.* Nine repositories, nine CI configurations, and coordinated changes cost several PRs plus
a release. Mitigated by a shared workflow template, a documented cross-repository change procedure,
editable installs locally, and a nightly compatibility matrix.

*Negative.* Distribution names may be taken on PyPI. Mitigated by verifying and by a documented
fallback that keeps import names stable.

*Negative.* This documentation set lives in its own repository and can drift from the code. Mitigated
by generating configuration references and OpenAPI snapshots into each component's own repository,
where CI diff-checks them, and by keeping this set to architecture and standards rather than to
API-level detail that code can generate.

## Revisit when

ADR-0001's monorepo trigger fires, or a distribution name conflict forces the fallback naming (which
is a README/`pyproject` change, not an architecture change).
