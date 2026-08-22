# ADR-0028 — Prompt attribution granularity and shared prompt tooling

**Status:** Accepted (2026-08-21)
**Amends:** [ADR-0012](0012-prompt-storage-format.md), [Prompt Management Standards §3–4](../standards/prompt-management-standards.md), [ADR-0011](0011-shared-package-boundaries.md) (adds an extraction trigger).

## Context

Two problems surfaced in the audit, both about *granularity*.

**Attribution.** A FreeWeight run records `prompt_pack_hash` and hashes it into the reproducibility
fingerprint. ADR-0017's hard separations, however, are stated per benchmark: evidence is separated
when "the prompt pack version used by *that benchmark* differs". These cannot both hold. A pack is
application-wide, so editing an unrelated content prompt changes `pack_sha256`, changes every run's
fingerprint, and separates results that share every prompt they actually used. In a system whose
value depends on comparing runs over months, a fingerprint that changes for unrelated reasons is
almost as damaging as one that fails to change — it makes everything incomparable and trains the user
to ignore the signal.

**Tooling.** All three applications independently build the same prompt-pack machinery — record
schema, loader, `StrictUndefined` renderer, variable validation, canonical hashing, manifest builder,
`prompts list|show|build` CLI — in FreeWeight P7, LoadCoach P4 and IdeaPress P2. The traceability
matrix answers the question it was asked ("who owns a shared *prompt*?" — nobody, duplication is
fine) but not the one that matters here: who owns the *machinery*. Three implementations of canonical
hashing that must each be correct is exactly the duplication ADR-0011 exists to prevent, and it meets
ADR-0011's own bar of "at least two real consumers".

## Decision

### 1. Prompts are attributed per benchmark, and packs are hashed per subset

* Every benchmark manifest already lists `prompt_ids` with versions. The manifest's
  `prompt_subset_hash` is the SHA-256 over the sorted `(prompt_id, version, sha256)` of **only the
  prompts that benchmark uses**.
* The reproducibility fingerprint's `benchmark` section carries `prompt_subset_hash`, not
  `prompt_pack_hash`.
* `capability.evidence` carries `prompt_subset_hashes` as a mapping of benchmark key →
  `prompt_subset_hash` ([ADR-0022](0022-capability-evidence-record-contract.md)), and ADR-0017's hard
  separation compares those per benchmark.
* `runs.prompt_pack_id` / `prompt_pack_version` / `prompt_pack_hash` are retained as *provenance*
  — they answer "which pack was installed" — but they are **not** fingerprint inputs.
* Editing a prompt no benchmark uses therefore changes no fingerprint. Editing one a benchmark uses
  changes that benchmark's subset hash, separates that benchmark's results, and forces its suite
  version bump exactly as the prompt standards require.

### 2. Prompt tooling is extracted to SetSpec at LoadCoach P4

The record schema, the manifest schema, canonical hashing, the loader/validator and the renderer move
into `setspec.prompts` when the second consumer arrives — the same "extract at the second consumer"
timing WeightsDB and MirrorWall follow.

SetSpec is the right home and not an arbitrary one: a prompt record *is* a versioned schema-bearing
document with a hash that must be byte-identical across applications, which is SetSpec's entire remit.
It adds no dependency (pydantic and canonical JSON are already there) and no behaviour — the library
loads, validates, renders and hashes; it holds no prompt.

* FreeWeight writes the first implementation in its P7 as an in-application module, built as if it
  were a package, exactly as it does for storage and UI.
* LoadCoach P4 extracts it; LoadCoach and IdeaPress consume it from the start.
* FreeWeight adopts it in its P12, alongside WeightsDB and MirrorWall, under the same acceptance
  criterion: the existing test suite passes unchanged.
* Each application keeps its **own** pack, its own prompts and its own `prompts` CLI commands. Nothing
  about prompt *content* is shared, and no prompt is ever imported from another component.

### 3. `RenderedPrompt` hashing is a cross-application contract

Because a prompt hash appears in benchmark evidence that another application reads, `sha256` over a
record and `rendered_sha256` over a rendering are contract-grade determinism, tested with goldens in
the same way as `canonical_json`.

## Alternatives considered

**Keep the pack hash in the fingerprint.** Simplest, and it is what the freeze said. Rejected: it
separates results for reasons unrelated to the measurement, and users respond to over-sensitive
provenance by ignoring it.

**Hash every prompt version into every result individually.** Maximally precise. Rejected: a result
would carry a growing list of hashes that are almost all irrelevant to it; the per-benchmark subset is
the natural unit because the manifest already declares it.

**Leave the machinery duplicated in three applications.** Rejected: it is three chances to get
canonical hashing subtly wrong, and hashes that disagree across applications would corrupt the
separation rules that depend on them — a cross-application failure caused by a deliberately
uncoordinated implementation.

**A seventh package, `promptrack`.** Rejected: a package whose only job is a loader and a hasher,
duplicating SetSpec's dependency set and its purpose, is the `LoadCoachClient` mistake in another
costume ([ADR-0011](0011-shared-package-boundaries.md)).

**Put the machinery in BaseAiCore.** Rejected: it needs Jinja2 and pydantic, and BaseAiCore's zero
dependencies are load-bearing.

## Consequences

*Positive.* Fingerprints change when the measurement changes and not otherwise. One implementation of
prompt hashing, so the separation rules that depend on it cannot disagree between applications. Two
of the three applications never write the machinery at all.

*Negative.* SetSpec gains Jinja2 as a dependency, taking it from one runtime dependency to two, and
its gold-standard dependency budget is amended accordingly. Justified by the alternative being three
implementations of a determinism contract.

*Negative.* FreeWeight is touched twice for prompts, as it already is for storage and UI. The second
pass is deletion, and it rides along with the P12 adoption phase rather than adding one.

## Revisit when

A fourth consumer of prompt machinery appears outside the suite, which would argue for a standalone
package after all; or Jinja2's presence in SetSpec blocks a consumer that wants schemas without a
template engine — at which point `setspec[prompts]` becomes an extra.
