# ADR-0009 — SetSpec schema and versioning strategy

**Status:** Accepted (2026-08-21)
**Amended by:** [ADR-0025](0025-envelope-boundaries.md) — defines which bodies carry the envelope; [ADR-0022](0022-capability-evidence-record-contract.md) — normative field list for `capability.evidence`; [ADR-0028](0028-prompt-pack-granularity.md) — adds `setspec.prompts`.

## Context

FreeWeight produces benchmark evidence that LoadCoach consumes. IdeaPress exchanges requests and
results with LoadCoach. All three emit events over SSE and errors over HTTP. These applications are
released independently, installed independently, and will routinely run at different versions on the
same machine.

Requirement: every cross-application payload carries an explicit schema version; breaking changes
increment the major version; readers reject unsupported majors clearly.

## Decision

**Pydantic v2 models in a dedicated `setspec` package, versioned per payload type, published with
generated JSON Schema and golden example payloads.**

1. **Envelope.** Every **transferable** payload — a document that outlives the request that produced
   it: an export, an evidence bundle, a result, a machine profile, an event — is wrapped. An HTTP
   API's own request and response bodies are versioned by their path and documented by OpenAPI
   instead; [ADR-0025](0025-envelope-boundaries.md) draws the line and gives the membership test.

```json
{"schema": "benchmark.result", "schema_version": "1.0",
 "generated_at": "…", "generator": {"name": "freeweight", "version": "1.0.0"},
 "payload": { … }}
```

2. **Versioning.** `"MAJOR.MINOR"` per payload type. Major = breaking (removed/renamed field,
   changed type or meaning, tightened validation). Minor = additive (new optional field).
3. **Reader policy.** Accept any minor within a supported major, **including minors newer than the
   reader knows**. Reject an unsupported major with `SCHEMA_VERSION_UNSUPPORTED`, naming both
   versions. Never partially parse an unsupported major.
4. **Unknown fields are preserved.** Every payload type has **two** model classes generated from one
   definition: a strict outbound model (`extra="forbid"`) that writers use, and a preserving inbound
   model (`extra="allow"`) that readers use and that round-trips unknown keys into an `extras`
   mapping. A reader re-exporting data dumps through the *inbound* model, which re-emits the
   preserved keys — so "writers emit only known fields" (rule 5) and "an older reader does not strip
   a newer writer's fields" are both true, of different models. The round-trip contract
   (`load(dump(x)) == x`) is asserted per model class, not across the pair.
5. **Writers emit only known fields**, and never `null` where "unsupported" is meant — the string
   `"unsupported"` is the value for an unavailable measurement.
6. **Coexisting majors.** When `benchmark.result 2.0` ships, the v1 models remain importable as
   `setspec.benchmark.v1` for at least one minor release of every consumer.
7. **Publication.** Each version publishes generated JSON Schema plus at least three golden payloads
   as package data. Contract tests in producers and consumers run against them; a schema change
   without a version bump fails CI.
8. **Independence.** Schema versions are independent of package versions and of API versions.

9. **Supported majors, not supported versions.** `SUPPORTED_SCHEMAS` maps a schema name to the set
   of **majors** a build understands, plus the highest minor it knows of each. A newer minor within a
   supported major is accepted (rule 3), so exact-version matching would contradict the reader
   policy.

Initial payload types: `benchmark.result`, `benchmark.run_summary`, `benchmark.evidence_bundle`,
`capability.evidence`, `machine.profile`, `model.identity`, `event.envelope`, `error.envelope`,
`capability.vocabulary`, `prompt.record`, `prompt.manifest`
([ADR-0028](0028-prompt-pack-granularity.md)).

`error.envelope` is a shape, not a wrapped document: it models the object inside `{"error": {…}}` and
is transported unwrapped ([ADR-0025](0025-envelope-boundaries.md) §4).

## Alternatives considered

**Plain dataclasses with hand-written `to_dict`/`from_dict`.** What the prior code did — roughly 200
lines of boilerplate that drifted from its dataclasses silently and produced no schema. Rejected.

**JSON Schema as the source of truth with generated Python.** Schema-first is defensible and
tool-friendly. Rejected: code generation adds a build step and the generated models are less
pleasant to use than hand-written pydantic, which already *emits* JSON Schema. We keep the
generated schema as an artifact, not as the source.

**Protocol Buffers / Avro / MessagePack.** Strong versioning stories and compact wire formats.
Rejected: binary payloads are unreadable in a local-first tool where users inspect exports by hand,
they need a compiler in the build, and the volumes involved make compactness irrelevant.

**Single global schema version for the whole suite.** Simpler to reason about. Rejected: it forces a
breaking bump on every payload type when only one changes, which would push consumers into
unnecessary major upgrades.

**Semver triple (`MAJOR.MINOR.PATCH`) per payload.** Rejected: a patch has no meaning for a data
contract — either the shape changed or it did not.

## Consequences

*Positive.* Validation, JSON Schema, OpenAPI integration and typed models from one definition.
Mixed-version deployments work by design. A clear, testable rule for what a reader does with an
unknown version. Golden payloads make cross-repository compatibility testable without a shared
environment.

*Negative.* Pydantic becomes a dependency of every application (it already is, via FastAPI).
Maintaining coexisting majors costs work — bounded by the one-minor-release deprecation window.
`extra="allow"` slightly weakens strictness on inbound payloads; deliberate, and the reason
(preservation across a version gap) is documented on the models.

*Negative.* Discipline is required to keep SetSpec free of behaviour — it must hold schemas and
version negotiation only, never scoring or routing logic. Enforced by the boundary rules and by
review.

## Revisit when

Payload volume makes JSON a measured bottleneck (evidence bundles in the hundreds of megabytes), or
a consumer outside Python needs first-class support with generated clients.
