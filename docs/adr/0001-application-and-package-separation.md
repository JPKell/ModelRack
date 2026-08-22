# ADR-0001 — Application and package separation

**Status:** Accepted (2026-08-21)

## Context

The suite delivers three products (FreeWeight, LoadCoach, IdeaPress) that share substantial
infrastructure: model access, telemetry, identity, storage plumbing and UI primitives. Two obvious
structures present themselves — a single monolithic repository with three entry points, or a
monorepo of independently versioned packages — and the prior projects demonstrate the cost of
getting it wrong: three separate Ollama clients, two telemetry implementations, and a provider layer
that could not be reused because it imported application domain types.

The requirements are explicit: each application must be independently downloadable and fully
functional alone; applications must never depend on another application's internals; shared code
must be distributed as installable packages, not submodules or copies.

## Decision

Three applications and six shared packages, each in its own repository, each independently versioned
and released. Applications depend on packages by version range. Applications communicate with each
other only over versioned HTTP APIs or versioned exported files.

Dependency direction is one-way and enforced by `import-linter` in every repository:

```text
Applications → capability packages → contract package → domain foundation
```

No package may import an application. No application may import another application. No application
may access another application's database.

## Alternatives considered

**Single application with pluggable modes.** One codebase, one install, three UIs. Rejected: a user
who wants only a content tool would install a benchmark engine and a router; the "works standalone"
requirement becomes vacuous; and every change risks all three products.

**Monorepo with path-based dependencies.** One repository, several distributions, editable installs
by relative path. Simpler day-to-day, but it makes independent versioning and independent release
cadence awkward, encourages accidental cross-imports (the linter becomes the only barrier, and it is
easy to relax), and pushes users toward "clone the repo" rather than "pip install the app".
Rejected, though acknowledged as the most tempting alternative — see "Revisit when".

**Git submodules for shared code.** Explicitly rejected by the requirements and by experience:
submodule pointers rot, contributors forget `--recursive`, and there is no version resolution.

**Copy shared code into each application.** Rejected: this is exactly the failure the inventory
documents.

## Consequences

*Positive.* Each product is genuinely installable alone. Package APIs get designed as APIs because a
second consumer exists. Employer or third-party tools can consume `modelrack` or `sweatmeter` without
inheriting anything else. Releases are independent, so a FreeWeight bug fix does not gate LoadCoach.

*Negative.* Nine repositories to maintain, nine CI configurations, and a cross-repository change
(add a field to `ModelIdentity`, use it in three applications) becomes several coordinated PRs and a
release. Dependency ranges must be curated. A compatibility matrix job is required.

*Mitigations.* One shared CI workflow template; a documented cross-repository change procedure in
the packaging standards; editable installs for local development; the nightly compatibility matrix.

## Revisit when

The coordination overhead of a cross-cutting change exceeds the isolation benefit — concretely, if
three consecutive months see more than half of all changes requiring coordinated releases across
three or more repositories. In that case the migration is to a monorepo that still publishes the
same distributions, which preserves every external promise made here.
