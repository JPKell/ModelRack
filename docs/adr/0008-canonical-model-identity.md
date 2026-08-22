# ADR-0008 — Canonical model identity

**Status:** Accepted (2026-08-21)
**Amended by:** [ADR-0024](0024-canonical-id-and-model-references.md) — fixes the canonical string format and removes it from URL paths.

## Context

Three applications, several providers, and a user who types `qwen3.5` when they mean
`qwen3.5:9b-q8_0` whose digest changed last Tuesday. Without one authoritative identity, FreeWeight
measures one thing, LoadCoach routes to another, and IdeaPress reports a third — and none of the
comparisons mean anything.

The old planning proposed `ModelIdentity(provider, provider_model_name, digest)`. The old benchmark
specification separately, and correctly, insisted that results attach to *model + runtime
configuration*, because the same weights behave differently at different context sizes and KV
precisions. Both are right; neither alone is sufficient.

## Decision

Identity is minimal and immutable; everything else is a separate concept.

```python
@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider_kind: ProviderKind      # ollama | openai_compatible | llamacpp | vllm | fake
    provider_model_name: str
    artifact_digest: str | None      # "sha256:<64 hex>" when exposed
```

* Canonical string: `{provider_kind}/{provider_model_name}@{digest_short}`, where `digest_short` is
  `"sha256:" + the first 12 hex characters` of the digest, or the literal `"unknown"` — for display
  and lookup, never parsed back. Fixed by [ADR-0024](0024-canonical-id-and-model-references.md).
* `IdentityConfidence` (`digest` | `name_only`) travels with every stored measurement.
* `ModelDescriptor` holds mutable descriptive metadata, separately.
* `RuntimeProfile` (context size, KV precision, GPU layers, flash attention, threads, batch,
  keep-alive, provider options) hashes to `runtime_profile_hash`.
* `MeasurementSubject = (identity, runtime_profile_hash, machine_fingerprint)` is the unit of
  comparability.
* Sampling parameters are **request** parameters, recorded on the sample/job, not part of identity or
  runtime profile.
* Identity is created only by ModelRack from a provider response or via `ModelRack.resolve()`.
  Applications never build one from string fragments.

Full detail: [Canonical Model Identity](../architecture/canonical-model-identity.md).

## Alternatives considered

**Name only.** Simple and matches what users type. Rejected: `qwen3.5:latest` is not a stable
identity; a retag silently corrupts history.

**Digest only.** Perfectly stable. Rejected: unusable for humans, and unavailable from providers that
expose no digest — identity would become impossible for them.

**Include the endpoint URL.** Would distinguish two servers. Rejected: a port change would fragment
one machine's history for no measurement-relevant reason. Endpoints are deployment facts, recorded on
the run.

**A suite-assigned surrogate ID with a mapping table.** Rejected: every application would need the
mapping, which means either a shared database (forbidden) or a synchronization protocol (complexity
without benefit).

**Fold runtime settings into identity.** Rejected: it makes "the same model" un-nameable and
prevents the deliberate cross-runtime studies (KV precision, context size) that are core FreeWeight
features.

**Include quantization as an identity field.** Rejected: it is descriptive metadata that the model
name almost always encodes; duplicating it invites two sources of truth. The digest is the
authority on "which weights".

## Consequences

*Positive.* One name across the suite. Retagging is visible rather than silent. Comparability is a
computable property, not a judgement call. Runtime and quantization studies are first-class.

*Negative.* Three concepts where developers might want one; the temptation to attach a "score" to a
model row must be resisted, and is called out in the boundary rules. `name_only` identities carry a
permanent caveat that must be surfaced in the UI and in evidence confidence, which is extra work in
every consumer.

*Negative.* `NULL` digests do not collide under SQL `UNIQUE`, so repositories must enforce
"one `name_only` row per (kind, name)" in code, with a test proving that a later digest **upgrades**
the row rather than duplicating it.

## Revisit when

A provider appears whose model identity genuinely needs a fourth field (for example a per-adapter
LoRA identity, or a served-model alias distinct from the artifact). At that point the field is added
as optional with a documented migration, since identity is a value object stored as columns rather
than a parsed string.
