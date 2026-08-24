# Contributing to ModelRack

This repository is one component of the Local AI Suite. Before changing anything, read
`docs/packages/modelrack/spec.md` and the current
phase in `development-plan.md` — both are in this repository's `docs/` folder, copied from the suite's
central documentation set so this repository can be worked on independently.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Required reading, in order

1. This component's spec — purpose, scope, non-goals, contracts.
2. `development-plan.md` in the same folder — the phase you are implementing, its acceptance criteria and its tests.

## Rules that apply to every change here

* Follow the architecture's dependency direction.
  This repository's `.importlinter` enforces it in CI; do not weaken that file to make an import work.
* No business logic in a route handler or CLI command body — both call one service method and render
.
* An unavailable measurement is `Unsupported`, never zero, never `None` used as a substitute
.
* Prompts are versioned JSON records, not Python string literals.
* Every phase's acceptance criteria in `development-plan.md` must be demonstrable, not merely
  test-covered — the plan states what to run and what a person should see.

## Before opening a pull request

```bash
ruff format --check .
ruff check .
mypy src tests
lint-imports
pytest -m "not live and not performance"
```

All of the above run in CI (`.github/workflows/ci.yml`); a red CI run blocks merge.

## Commit style

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`, `build:`,
`ci:`), with `!` or a `BREAKING CHANGE:` footer for breaking changes. Update `CHANGELOG.md` under
`## [Unreleased]` for any user-visible change.
