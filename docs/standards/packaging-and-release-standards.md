# Packaging and Release Standards

**Covers:** repository layout, Git strategy, versioning, dependency ranges, CI/CD, publishing and
release procedure for every application and package in the suite.

---

## 1. Repository model

Every application and every shared package is its **own Git repository**, tracked independently.

```text
~/ai/suite/                       # workspace directory (not itself a source repository)
├── docs/          ← this documentation set; its own repository (ai-suite-docs)
├── FreeWeight/    ← repository
├── LoadCoach/     ← repository
├── IdeaPress/     ← repository
└── py/
    ├── BaseAiCore/    ← repository
    ├── SetSpec/       ← repository
    ├── ModelRack/     ← repository
    ├── SweatMeter/    ← repository
    ├── WeightsDB/     ← repository
    └── MirrorWall/    ← repository
```

* **No Git submodules** for runtime dependencies — ever. Shared code is consumed as an installed
  Python distribution.
* No copying of shared source into an application.
* The workspace directory is a convenience for local development, not a build unit. Nothing depends
  on a relative path outside its own repository.
* `planning/` and `.old_projects/` stay outside every repository as read-only history.

### 1.1 Required repository contents

```text
<repo>/
├── pyproject.toml            # PEP 621 metadata, tool config
├── README.md                 # what it is, install, quickstart, links
├── CHANGELOG.md              # Keep a Changelog format
├── LICENSE                   # see §2
├── CONTRIBUTING.md           # dev setup, standards links, PR expectations
├── SECURITY.md               # how to report a vulnerability
├── .gitignore .editorconfig .importlinter .pre-commit-config.yaml
├── .github/workflows/{ci.yml,release.yml}
├── src/<package>/            # src layout, always
├── tests/
├── docs/                     # component docs (configuration reference, API snapshot, usage)
└── requirements/             # lockfiles for CI and releases
```

### 1.2 Layout rules

* **`src/` layout** everywhere, so tests run against the installed distribution and packaging
  mistakes surface immediately.
* Build backend: `hatchling` (simple, PEP 621-native, good package-data handling for MirrorWall's
  templates and static files).
* Package data (templates, static assets, JSON Schemas, prompt records, migration scripts) is
  declared explicitly and asserted by a test that installs a wheel and loads each resource through
  `importlib.resources`.

## 2. Licensing

| Component | Licence | Rationale |
|---|---|---|
| BaseAiCore, SetSpec, ModelRack, SweatMeter, WeightsDB, MirrorWall | **Apache-2.0** | Permissive with an explicit patent grant; the least friction for reuse, including in a corporate environment |
| FreeWeight, LoadCoach, IdeaPress | **Apache-2.0** | Consistency across the suite; keeps the applications adoptable |
| Vendored third-party assets | their own | Licence file preserved beside the asset, listed in `THIRD_PARTY_NOTICES.md` |

Every source file omits a licence header (the LICENSE file governs); every repository states the
licence in `pyproject.toml` and `README.md`. External benchmark datasets are **not** redistributed —
they are downloaded by the user, with their licence and source recorded in the benchmark manifest.

---

## 3. Versioning

Semantic versioning, with a pre-1.0 phase that says what it means:

* `0.x.y` — public API may change on a **minor** bump; every change is still in the changelog, and
  consumers pin `>=0.x,<0.(x+1)`.
* `1.0.0` — the public API is committed. From then on: **major** = break, **minor** = additive,
  **patch** = fix.
* The version lives in exactly one place (`src/<pkg>/__about__.py`, exposed as `__version__` and read
  by hatchling). A test asserts that the tag, the metadata and `__version__` agree.

### 3.1 Three independent version axes

| Axis | Example | Bumped when |
|---|---|---|
| Distribution version | `modelrack 0.4.2` | Python API changes |
| HTTP API version | `/api/v1` | Wire-breaking endpoint change (major only) |
| Schema version | `benchmark.result 1.1` | Payload structure change (major = breaking) |

They are independent. Releasing `freeweight 2.0.0` does not force `/api/v2`, and a `benchmark.result`
minor bump does not force an application major.

### 3.2 What counts as breaking

Removing or renaming any public name; changing a parameter's meaning, type or ordering; making an
optional parameter required; changing a return type; changing an error `code`'s meaning; tightening
validation; changing a default that alters behaviour; changing a serialized shape.

Not breaking: adding an optional keyword-only parameter; adding a field to a response; adding an
error `code`; performance changes; adding an extension point.

---

## 4. Dependency policy

* Applications depend on packages by **compatible range**, never a Git URL or branch:

```toml
dependencies = [
    "baseaicore>=0.4,<0.5",
    "setspec>=0.3,<0.4",
    "modelrack>=0.5,<0.6",
    "sweatmeter>=0.4,<0.5",
    "weightsdb>=0.2,<0.3",
    "mirrorwall>=0.2,<0.3",
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.30,<1",
    "pydantic>=2.9,<3",
    "sqlalchemy>=2.0.30,<3",
    "alembic>=1.13,<2",
    "typer>=0.12,<1",
    "jinja2>=3.1,<4",
    "httpx>=0.27,<1",
]
```

* Post-1.0 the ranges widen to `>=1.2,<2`.
* **An application at 1.0 may depend on `0.x` packages**, which the roadmap does deliberately. The
  consequence must be planned rather than discovered: when a package reaches 1.0, the pinned range
  `>=0.6,<0.7` excludes it, so every application needs a release that widens the range. Those
  releases are part of the M9 checklist, and the upgrade guide states plainly that an application
  release is required to move to a package's 1.0 — an application is never left unable to resolve
  its own dependencies.
* Optional features are extras, and the base install never pulls them:

```toml
[project.optional-dependencies]
postgres  = ["psycopg[binary]>=3.2,<4"]
telemetry = ["sweatmeter>=0.4,<0.5"]     # IdeaPress: optional display only
dev       = ["pytest", "pytest-cov", "mypy", "ruff", "import-linter", "respx", …]
```

An application's `dev` extra may depend on **another application's distribution**, solely to obtain
its committed OpenAPI snapshot as package data for contract tests
([Testing Standards §8](testing-standards.md)). `ideapress[dev]` therefore depends on `loadcoach`,
while `ideapress` does not; the import-linter contract forbidding `from loadcoach import …` anywhere
under `src/` is what keeps the test-time dependency from becoming a runtime one, and the
clean-venv install-check job proves the base install pulls in no application.

* During local development, editable installs are used and are the **only** sanctioned way to
  consume unreleased package code:

```bash
python -m pip install -e ./py/BaseAiCore -e ./py/SetSpec -e ./py/ModelRack \
                      -e ./py/SweatMeter -e ./FreeWeight[dev]
```

  CI never uses editable installs; it installs from built artifacts or from the index.
* Lockfiles (`requirements/ci.lock`, `requirements/release.lock`) are generated and committed, so a
  green build is reproducible.
* A new third-party dependency needs a justification in the PR description; the dependency budget is
  deliberately small.

---

## 5. CI

Every repository runs the same workflow on every push and pull request
(`.github/workflows/ci.yml`), with jobs in this order and all of them blocking:

| Job | Command | Notes |
|---|---|---|
| format | `ruff format --check .` | |
| lint | `ruff check .` | |
| types | `mypy src tests` | strict for packages |
| boundaries | `lint-imports` | import-linter contracts |
| tests | `pytest -m "not live and not performance"` | matrix: Python 3.12, 3.13 (+3.14 non-blocking) |
| db-matrix | `pytest -m integration` against SQLite **and** PostgreSQL service | applications and WeightsDB only |
| coverage | `coverage report --fail-under=85` + `diff-cover --fail-under=90` | 95 % for packages/domain |
| contracts | `pytest -m contract` incl. schema goldens and OpenAPI snapshot | fails on undocumented API change |
| security | `pip-audit`, `gitleaks detect` | blocking |
| build | `python -m build` + `twine check dist/*` | sdist and wheel |
| install-check | clean venv, install the wheel, `python -c "import <pkg>"`, run `<app> --version` | catches missing package data and accidental app dependencies |
| docs | configuration-reference regeneration diff | fails on drift |

Nightly (`schedule`): `pytest -m performance`, `pytest -m live` on a self-hosted runner with a GPU
and Ollama, plus the cross-repository compatibility matrix (§7).

---

## 6. Release procedure

```text
1. All CI green on main
2. Update CHANGELOG.md (move Unreleased → the new version, dated)
3. Bump __about__.py
4. Open a release PR; review the changelog and the public-API diff
5. Merge
6. Tag:  git tag -a v0.5.0 -m "…"   &&   git push --tags
7. Tag triggers release.yml:
       build sdist + wheel  →  twine check  →  run the test suite against the built wheel
       →  publish to PyPI via Trusted Publishing (OIDC)  →  create the GitHub release with notes
8. Verify: pip install <name>==<version> in a clean venv, import, smoke test
9. Downstream: open dependency-bump PRs in consumers
```

Rules:

* Releases are **only** produced by CI from a tag. No manual `twine upload`, ever.
* PyPI publishing uses **Trusted Publishing / OIDC**; no long-lived API tokens exist in any
  repository or organisation secret store.
* Every release is preceded by a successful publish to TestPyPI for the first release of each
  package, and by an install-check job for every release.
* Release notes are generated from the changelog, plus a "compatibility" section naming the ranges
  of suite packages this version works with.
* Applications additionally publish: the OpenAPI snapshot, the configuration reference, migration
  notes, and (post-1.0) a signed checksum file for the artifacts.
* A release is never re-tagged. Mistakes are fixed with a new patch version and, where necessary, a
  PyPI yank plus a changelog note explaining why.

### 6.1 Downgrade

Migrations are forward-only in intent, and downgrades exist where they are lossless. The honest
statement to users, tested before every application 1.0:

* **Downgrading the application without downgrading the database is refused**, not attempted: a
  database ahead of the code raises `SchemaAhead` at startup and names both revisions and the backup
  directory.
* A supported downgrade path is: stop the application, restore the automatic pre-migration backup,
  install the older version. This is the procedure in each application's `docs/upgrading.md`, and it
  is exercised by a test that upgrades, writes data, restores and starts the older code.
* Where a migration has a lossless `downgrade()`, `<app> db downgrade <rev>` is the faster path and
  is named in the same document. A migration whose downgrade would lose data raises instead of
  silently dropping, per [Database Standards §5](database-standards.md).

---

## 7. Cross-repository compatibility

* Each application declares, in its README and release notes, the tested version ranges of every
  suite package it depends on.
* A nightly **compatibility matrix** job installs the current release of each application against
  the lowest and highest supported version of each package and runs the contract and e2e suites.
* SetSpec additionally runs a **schema compatibility** job: every published schema version is
  validated against its golden payloads, and the reader in each application is tested against every
  supported major.
* Breaking a consumer is a release blocker for the package, not a problem for the consumer to
  discover later.

---

## 8. Distribution names and public publishing

* Publish to PyPI only what is genuinely reusable: the six shared packages, and the three
  applications (so users can `pip install freeweight`).
* Distribution names are claimed before first publish. If a name is taken, the documented fallback
  is `aisuite-<name>` with the import name unchanged; the choice is recorded in the repository's
  README and in [ADR-0015](../adr/0015-repository-and-distribution-model.md).
* Never publish: employer-specific code, private datasets, or an application-internal module dressed
  up as a library.
* Pre-1.0 packages are published with a clear "API may change on minor releases" note in the README
  and the PyPI description.

---

## 9. Branching and commits

* `main` is always releasable. Work happens on short-lived branches (`feat/…`, `fix/…`, `docs/…`,
  `chore/…`).
* Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`,
  `build:`, `ci:`), with `!` or a `BREAKING CHANGE:` footer for breaks. The changelog is written by
  hand from these, not generated blindly.
* Squash merge; the squashed message is the changelog-quality summary.
* Every PR: passes CI, updates the changelog when user-visible, updates docs when behaviour changes,
  and adds or extends tests.
* Tags are `vMAJOR.MINOR.PATCH`, annotated, on `main` only.

---

## 10. Installation targets

The suite must install cleanly in each of these, and each is covered by an install-check job or a
documented manual verification before a 1.0 release:

```text
pip install <app>                       # from PyPI into a fresh venv
pipx install <app>                      # isolated CLI + server
pip install -e .[dev]                   # development
pip install <app>[postgres]             # optional extra
python -m <app>                         # module execution
```

Post-1.0 optional targets, explicitly out of scope until then: OS packages, container images,
single-file executables. They are listed as future extensions so nobody assumes they exist.
