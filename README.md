# ModelRack

The suite's only model client: a provider-neutral abstraction over local inference runtimes (Ollama first), with a deterministic FakeProvider.

**Status:** specified, not yet implemented. This repository currently holds the project scaffold
(directory structure, tooling configuration, and a local copy of the relevant suite documentation) —
see [development plan](docs/packages/modelrack/development-plan.md) for what each phase adds.

Part of the **Local AI Suite** — see [docs/architecture/executive-summary.md](docs/architecture/executive-summary.md)
for how ModelRack fits with the suite's other applications and packages.

## Install

```bash
pip install modelrack
```

## Quickstart

```python
import modelrack
```

See [docs/packages/modelrack/spec.md](docs/packages/modelrack/spec.md) §20 for a runnable example.

## Documentation

This repository carries its own copy of the relevant suite documentation under [`docs/`](docs/README.md),
so it can be read and implemented independently of the other eight suite repositories. Start with
[`docs/README.md`](docs/README.md).

| Read this | For |
|---|---|
| [docs/packages/modelrack/spec.md](docs/packages/modelrack/spec.md) | Purpose, scope, non-goals, public contracts, configuration, acceptance criteria |
| [docs/packages/modelrack/development-plan.md](docs/packages/modelrack/development-plan.md) | The phased build plan: goals, work, tests, acceptance criteria per phase |
| [docs/standards/](docs/standards/) | Coding, testing, security, API, database and packaging standards every phase follows |
| [docs/adr/](docs/adr/README.md) | The architectural decisions this design rests on |

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest -m "not live and not performance"
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow and [`SECURITY.md`](SECURITY.md) for
how to report a vulnerability.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
