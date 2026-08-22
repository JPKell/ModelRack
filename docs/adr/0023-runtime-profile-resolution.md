# ADR-0023 — Runtime profile resolution and served context

**Status:** Accepted (2026-08-21)
**Amends:** [ADR-0008](0008-canonical-model-identity.md) (measurement subject in execution), [ADR-0017](0017-benchmark-confidence-and-freshness.md) (runtime-profile hard separation, made operable).

## Context

`MeasurementSubject = (identity, runtime_profile_hash, machine_fingerprint)` is the unit of
comparability, and ADR-0017 makes a differing `runtime_profile_hash` a **hard separation** — evidence
measured under one profile does not describe another.

FreeWeight takes this seriously: a run names its runtime profile, stores it, and hashes it into the
reproducibility fingerprint. The audit found that **LoadCoach never resolves a runtime profile at
all**. It selects a model, calls the provider, and stores a `runtime_profile_hash` column that
nothing populates. Two consequences follow, and both are silent:

1. Evidence keyed by `runtime_profile_hash` can never be matched to an execution, so either the hash
   is ignored (routing on evidence ADR-0017 says does not apply) or no evidence ever matches.
2. Routing's `min_context_tokens` hard constraint is evaluated against `ModelDescriptor.max_context`,
   which is the **advertised** context. Under Ollama the *served* context is `num_ctx`, a runtime
   setting whose default is frequently a small fraction of the advertised maximum. LoadCoach would
   admit a model on an advertised 131 072-token context, run it at the provider's default, and let
   the provider truncate the prompt — producing a confidently wrong answer with no error. This is the
   same class of defect as a fabricated zero.

The suite already has the vocabulary to fix this; it simply was not applied above FreeWeight.

## Decision

**Every execution resolves an explicit `RuntimeProfile` before the provider call, and the candidate
that routing scores is the pair `(identity, runtime_profile)` — the execution subject.**

### 1. A profile is always resolved

`RuntimeProfile()` with every field `None` is a legal profile meaning "provider defaults"; its
`profile_hash` is stable and it is stored like any other. There is no "no profile" state.

Resolution order for a LoadCoach execution (a case of the execution-parameter chain in
[Configuration Standards §1.1](../standards/configuration-standards.md)):

```text
[runtime].default        → per-model override in [runtime.models."<canonical_id>"]
                         → task-profile runtime settings
                         → request override (overrides.runtime_profile)
```

The resolved profile and its hash are stored on the job and named in the routing explanation.

### 2. LoadCoach gains a `runtime_profiles` table

Listed in LoadCoach's data ownership since the freeze but never defined. It mirrors FreeWeight's:
`id`, `profile_hash UNIQUE`, `context_size`, `kv_cache_precision`, `gpu_layers`, `flash_attention`,
`threads`, `batch_size`, `keep_alive`, `provider_options_json`, `created_at`.

### 3. Evidence matches only its own subject

Capability evidence contributes to a candidate's score only when its `runtime_profile_hash` equals
the candidate's resolved hash — ADR-0017's hard separation, applied. Evidence for the same model
under a different profile is **not** silently reused and **not** scored zero: it is absent, named in
the explanation as `evidence_profile_mismatch` with both hashes, and it counts toward the
`low_evidence` flag exactly as any other absence does.

This makes the FreeWeight ↔ LoadCoach pairing operationally meaningful: a user who benchmarks under
the profile LoadCoach runs gets evidence-driven routing, and a user who does not is told why not,
with the two hashes side by side and a suggested `freeweight run start --context-size …`.

### 4. Served context, not advertised context

A new resolved value, recorded on every decision:

```text
served_context =
    runtime_profile.context_size            when set                          → source "configured"
    provider-reported served context        when the provider exposes it      → source "reported"
    descriptor.max_context                  otherwise, flagged "assumed"      → source "assumed"
```

* `min_context_tokens` and the context budget in [Routing §9](../apps/loadcoach/routing.md) are
  evaluated against `served_context`, never against `max_context`.
* KV and VRAM estimation in [Queue §5](../apps/loadcoach/queue-and-scheduling.md) uses
  `served_context`.
* When `ProviderCapabilities.context_configurable` is true, LoadCoach **sets** `context_size`
  explicitly for any task profile that declares `min_context_tokens`, so the served context is known
  rather than assumed.
* When it is false and the source is `assumed`, the decision carries the flag `assumed_context` and
  the explanation says so. A new rejection reason `context_not_configurable` applies when a profile
  requires a context the provider cannot be asked to serve.

### 5. FreeWeight records the same value

A FreeWeight run records `served_context` and its source alongside the runtime profile, so a
benchmark taken at an assumed context is distinguishable from one taken at a configured context.

## Alternatives considered

**Ignore the runtime profile in LoadCoach and match evidence on identity alone.** The status quo.
Rejected: it contradicts ADR-0017's hard separation and reintroduces exactly the "same weights,
different behaviour" error that ADR-0008 was written to prevent.

**Fold the runtime profile into identity.** Rejected again, for ADR-0008's reasons.

**Let LoadCoach re-benchmark under its own profile.** Rejected: benchmark execution is FreeWeight's,
and duplicating it here is the trap LoadCoach's risk register already names.

**Treat a profile mismatch as a confidence penalty rather than a separation.** Tempting, because it
keeps evidence usable. Rejected: a KV-precision or context change moves memory and speed metrics by
factors, not by percentages, and there is no defensible penalty coefficient. Absence with a named
reason is honest; a guessed multiplier is not.

**Trust `max_context` and let the provider truncate.** Rejected outright — silent truncation of a
user's input is the failure mode routing exists to avoid.

## Consequences

*Positive.* Evidence and execution describe the same subject, so imported evidence changes routing
for a reason a user can verify. Context constraints become true rather than aspirational. The
KV-cache estimate has a real context number to work from, which is what makes admission control
better than a guess. FreeWeight's runtime-comparison studies become directly actionable: they tell a
user which profile to configure.

*Negative.* A user who benchmarks under default settings and runs LoadCoach under configured settings
gets `evidence_profile_mismatch` until they align the two. This is a real friction — and it is the
honest surfacing of a mismatch that previously produced wrong routing silently. The explanation names
the fix.

*Negative.* One more table, one more configuration section, and one more value on every decision.

## Revisit when

A provider exposes per-request context without a load, making `served_context` a request parameter
rather than a profile field — at which point the resolution order gains a level rather than changing
shape.
