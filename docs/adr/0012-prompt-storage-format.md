# ADR-0012 — Prompts as versioned JSON records

**Status:** Accepted (2026-08-21)

## Context

Prompts determine benchmark scores, routing quality and content quality. In the prior projects they
lived in Python: a 406-line `prompting.py` of f-strings in `content_factory`, and `SYSTEM_PROMPT`
constants in `lm_ai_dev` (which had begun moving toward `system_prompt_file` configuration — the
right instinct).

Consequences of prompts-in-code: a prompt change is invisible in behavioural terms, cannot be
versioned independently of the application, cannot be attached to a benchmark result, cannot be
overridden by a user without editing installed source, and cannot be validated at startup.

Requirement §12 forbids embedding prompts in Python except for very small documented protocol
constants, and asks for structured JSON storage with IDs, versions, variables and response
expectations.

## Decision

**Prompts are JSON records in a versioned pack**, loaded and validated at startup, rendered through
Jinja2 with `StrictUndefined`.

* One file per prompt: `prompts/<area>/<prompt_id>.v<major>.json`.
* Record fields: `prompt_id`, `version` (semver), `schema_version`, `purpose`, `task`, `capability`,
  `system`, `template`, `variables` (typed, with requiredness and description), `response`
  (`format`, optional `json_schema_ref`, `expectations`), `model_requirements`, `metadata`
  (including a mandatory `change_reason`).
* A pack manifest lists every prompt with its version and SHA-256, plus a `pack_sha256`.
* Every use records `prompt_id`, `version` and `sha256`; every FreeWeight run records the pack hash
  as part of its reproducibility fingerprint.
* User overrides live in `$XDG_CONFIG_HOME/<app>/prompts/`, are marked as overrides in every record
  that used them, and require `--allow-prompt-override` for benchmark runs.
* The single exception for Python constants: a structural fragment under ~200 characters that is part
  of a request rather than an instruction, with a comment explaining why it is not a prompt record.

Full detail: [Prompt Management Standards](../standards/prompt-management-standards.md).

## Alternatives considered

**Markdown or plain text files with front matter.** More pleasant to write and diff. Rejected for
the record itself: variables, types, response contracts and model requirements are structured data,
and front matter is a second format to parse. Long prompt bodies remain readable in JSON because
they are stored as strings with real newlines and reviewed rendered.

**YAML.** Nicer multi-line strings, and it was used by the prior project. Rejected: an extra
dependency and a parser with surprising type coercion, for a format read by machines far more often
than written by hand. Consistent with [ADR-0019](0019-python-baseline-and-config-format.md), which
chooses TOML for configuration and JSON for data.

**Prompts in the database.** Editable in the UI, versioned by rows. Rejected as the *primary* store:
prompts must be reviewable in pull requests, shipped with the package, and hashable for
reproducibility. A database-backed override layer is a possible future extension for IdeaPress's
per-project prompt tuning, and would follow the same record schema.

**A prompt-management service (Langfuse, PromptLayer, …).** Rejected: an external service in a
local-first product, with data egress.

**Keep prompts in Python but hash the source.** Rejected: the hash would change on unrelated edits,
users could not override, and the requirement forbids it.

## Consequences

*Positive.* Prompt changes are reviewable diffs with a mandatory reason. Benchmark results name the
exact prompt that produced them and can re-render it. Startup validation catches a malformed prompt
before a two-hour run does. Users can override without touching installed code. LoadCoach can read
`model_requirements` as routing constraints.

*Negative.* Multi-line prompt text in JSON is awkward to author by hand. Mitigated by
`<app> prompts edit` (round-trips a record through `$EDITOR` as readable text), `prompts build`
(regenerates the manifest) and `prompts show --render` (previews with example variables).

*Negative.* Another versioning axis to manage. Justified: it is the axis benchmark comparability
depends on.

*Negative.* Rendering with `StrictUndefined` makes a missing variable an error at runtime. That is
the point — an empty string silently substituted into a judge prompt is a corrupted measurement.

## Revisit when

Per-project prompt customization in IdeaPress needs database-backed editing, at which point the
override layer becomes a store with the same schema, the same hashing and the same traceability.
