# ADR-0019 — Python baseline and configuration format

**Status:** Accepted (2026-08-21)

## Context

Two small decisions that touch every repository and are cheap to fix now, expensive later: which
Python versions are supported, and what format configuration files use.

Environment at freeze time: Ubuntu 26.04 with Python 3.14.4 as the default interpreter and 3.13.15
also installed. Prior projects used YAML (`models.yaml`, `project.yaml`) and JSON (`abl_config.json`).

## Decision

### Python

* `requires-python = ">=3.12"`.
* **Supported and CI-blocking: 3.12, 3.13.** 3.14 runs as a non-blocking early-warning job and is
  promoted to supported once every runtime dependency publishes wheels for it.
* Rationale for the 3.12 floor: `tomllib` (3.11), PEP 695 type parameter syntax, `StrEnum` (3.11),
  `typing.override` (3.12), better error messages, and `Self` — while remaining installable on
  current stable distributions. 3.11 buys little and costs typing ergonomics.
* Library code uses no feature newer than 3.12 unless guarded. `from __future__ import annotations`
  in every module for uniform behaviour across the range.

### Configuration format

* **TOML** for configuration files, read with stdlib `tomllib`.
* **JSON** for data interchange, prompt records and schemas.
* **No YAML anywhere.**

## Alternatives considered

**Python floor at 3.11.** Broader compatibility. Rejected: 3.12's typing and error-message
improvements are worth more than the marginal reach, and every target distribution ships ≥ 3.12.

**Python floor at 3.13.** Rejected: too aggressive for a tool users install on existing machines.

**Making 3.14 supported immediately.** Rejected: dependency wheel availability is not under our
control, and claiming support we cannot test is worse than testing it non-blockingly.

**YAML for configuration.** Familiar, and what a prior project used. Rejected: an extra dependency
(`PyYAML`) for something `tomllib` does natively, plus historically surprising type coercion, plus
significant-whitespace editing hazards. The suite's dependency budget does not spend a slot here.

**JSON for configuration.** Rejected: no comments, no trailing commas, poor hand-editing ergonomics.
It remains the right choice for machine-written data.

**INI via `configparser`.** Rejected: no nesting, everything is a string, no typed values.

**Python configuration files.** Rejected: executing configuration is an arbitrary-code-execution
surface and unvalidatable.

**Environment variables only.** Rejected: unusable for nested structures like a list of task profiles
or provider entries; environment remains an override layer, not the primary format.

## Consequences

*Positive.* Zero configuration-parsing dependencies. Comment-friendly, typed, unambiguous config
files that map cleanly onto nested pydantic settings. A Python range that is current without being
reckless, and a CI matrix that catches a version-specific regression before users do.

*Negative.* TOML's nested-array-of-tables syntax is verbose for deeply nested structures. Mitigated
by keeping configuration shallow — anything deeply nested (task profiles, benchmark suites, prompt
records) is JSON data, not configuration.

*Negative.* Users coming from YAML tools need a moment to adjust. Mitigated by `<app> config init`
writing a fully commented example.

*Negative.* Dropping 3.11 excludes some older long-term-support distributions. Documented in the
README; `pipx` with a newer interpreter is the workaround.

## Revisit when

3.12 reaches end of life (October 2028), or a required dependency drops a version in the supported
range.
