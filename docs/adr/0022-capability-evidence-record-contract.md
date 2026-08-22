# ADR-0022 — Capability evidence record contract

**Status:** Accepted (2026-08-21)
**Amends:** [ADR-0017](0017-benchmark-confidence-and-freshness.md) (freshness input), [ADR-0009](0009-setspec-schema-strategy.md) (adds a normative field list for one payload type).

## Context

`capability.evidence` and `benchmark.evidence_bundle` are the suite's most load-bearing
cross-application contract: they are the entire FreeWeight → LoadCoach value proposition, and they
will exist as persistent rows in two independently released applications.

The final architecture audit found that the two sides did not agree, and that the payload itself had
never been given a normative field list:

* FreeWeight's `capability_evidence` row carried `computed_at` (when the aggregation ran).
  LoadCoach's carried `measured_at` and `imported_at`. Nothing said which timestamp
  ADR-0017's `freshness_factor = 0.5 ** (age_days / half_life_days)` consumes. Read as `computed_at`,
  a nightly recomputation resets the apparent age of four-month-old measurements and confidence never
  decays — a silent, plausible, wrong number, which is the exact failure class ADR-0016 exists to
  prevent.
* FreeWeight's uniqueness key included `policy_version`; LoadCoach's did not, but included
  `source_id`. A bundle containing two policy versions of one capability therefore could not be
  imported without collision or silent last-write-wins.
* `excluded_count` was mandatory on the producer and absent on the consumer.
* Nothing said what a consumer does with evidence for a model it has never discovered — and
  `capability_evidence.model_id` was a non-null foreign key into a table only discovery writes.
* `GET /evidence/export?since=…` appeared in client guidance but in no endpoint definition, with no
  statement of which timestamp it filters or whose clock supplies it.

## Decision

### 1. Normative field set

`capability.evidence` v1 carries exactly these fields. Both applications store all of them.

| Field | Type | Meaning |
|---|---|---|
| `model` | identity triple + `canonical_id` + `identity_confidence` | The measured weights ([ADR-0024](0024-canonical-id-and-model-references.md)) |
| `runtime_profile_hash` | string | The profile the measurement was taken under ([ADR-0023](0023-runtime-profile-resolution.md)) |
| `machine_fingerprint` | string | Where it was measured |
| `capability_id` | string | A term in the SetSpec vocabulary |
| `score` | number 0–1 | The capability score |
| `confidence` | number 0.05–1 | Computed by FreeWeight per ADR-0017 |
| `sample_count` | integer ≥ 0 | Supported samples that produced `score` |
| `excluded_count` | integer ≥ 0 | Samples excluded, with the exclusion visible |
| `dispersion` | number \| `"unsupported"` | Coefficient of variation or disagreement rate |
| `measured_at` | RFC 3339 UTC | **The latest `completed_at` among contributing runs** |
| `computed_at` | RFC 3339 UTC | When the aggregation ran |
| `policy_version` | string | Confidence-policy version ([ADR-0017](0017-benchmark-confidence-and-freshness.md)) |
| `vocabulary_version` | string | Capability-vocabulary version the `capability_id` came from |
| `benchmark_versions` | mapping suite key → version | Hard-separation input |
| `dataset_hashes` | mapping | Hard-separation input |
| `prompt_subset_hashes` | mapping benchmark key → hash | Hard-separation input, **per benchmark, not per pack** ([ADR-0028](0028-prompt-pack-granularity.md)) |
| `contributing_metrics` | list | Metric key, weight, sample count per contributor |
| `source_run_ids` | list | Producer-local run IDs, for the producer's own drill-down |
| `environment` | object | Provider kind + version, GPU driver, CUDA, OS version at measurement |

`source_run_ids` is the only producer-local field; consumers store it opaquely and never resolve it.

### 2. Freshness is measured from `measured_at`

`age_days` in ADR-0017's `freshness_factor` is `now − measured_at`. `computed_at` never feeds
confidence. Recomputing evidence does not make it fresher, and a test asserts exactly that.

### 3. Uniqueness

* **Producer** (FreeWeight): `UNIQUE (model_id, runtime_profile_id, machine_id, capability_id, policy_version)`.
* **Consumer** (LoadCoach): `UNIQUE (source_id, canonical_id, runtime_profile_hash, machine_fingerprint, capability_id, policy_version)`.

`policy_version` is in both keys, so two policy versions coexist on both sides and a re-import is a
row-wise upsert rather than a collision. Routing uses the highest `policy_version` present for a
subject and names it in the explanation.

### 4. Evidence for an unknown model is retained, not rejected

`capability_evidence.model_id` is **nullable**. Every evidence row additionally stores the identity
triple and `canonical_id` denormalized. Import never fails because a model is unknown.

Matching rules, applied at import and again whenever discovery adds or upgrades a model row:

| Bundle identity | Local registry row | Result |
|---|---|---|
| Exact triple match | present | Bound: `model_id` set |
| Digest present | `name_only` row for the same `(kind, name)` | Registry row **upgraded** with the digest, then bound — the same in-place upgrade rule applications already use internally |
| `name_only` | row with a digest | **Not bound.** Retained with `match_state = "ambiguous_name_only"`; never used for routing, because the weights cannot be proven to be the ones installed |
| No candidate | — | Retained with `match_state = "unmatched"` |

Unbound evidence is visible in `GET /evidence` with its `match_state`, counted in the import result,
and never contributes to a routing score.

### 5. Incremental pull

* `GET /api/v1/evidence/export` accepts `?since=<RFC 3339>`, filtering on **`computed_at`**, on
  FreeWeight's clock.
* A client never supplies its own clock. It stores the `generated_at` of the previous bundle envelope
  and sends that value back. This makes the comparison single-clock and correct across machines.
* Every bundle declares `complete: true|false`. Only a complete bundle lets a consumer observe
  removals: evidence present locally for that source and absent from a complete bundle is marked
  `superseded`, never deleted.
* A consumer that has never imported from a source performs a complete pull.

## Alternatives considered

**Leave the field list to the pydantic models.** Rejected: the models live in one repository and the
audit found the two consumers had already drifted apart on paper. A normative table is what makes the
contract reviewable before any code exists.

**Reject evidence for unknown models.** Simpler importer. Rejected: it makes the order of discovery
and import load-bearing, silently drops data the user asked to import, and produces the confusing
result that importing twice gives different outcomes.

**Use `computed_at` for freshness and re-aggregate rarely.** Rejected: it makes correctness depend on
an operational habit, and the failure is invisible.

**Bind evidence to a surrogate model ID assigned by the producer.** Rejected for the reason ADR-0008
already gives: every consumer would need the mapping.

## Consequences

*Positive.* Freshness means what ADR-0017 says it means. Re-import is idempotent. Two policy versions
can coexist during a policy change. Evidence can be imported before models are discovered, which is
the ordinary case on a fresh LoadCoach. `name_only` ambiguity is refused rather than guessed.

*Negative.* Two timestamps per evidence row and a `match_state` column that must be re-evaluated on
every discovery pass. Bounded, and covered by tests on both sides.

*Negative.* `prompt_subset_hashes` is per benchmark, which means FreeWeight must attribute prompts to
benchmarks rather than hashing the whole pack into every result. That is the correct granularity and
is specified in [ADR-0028](0028-prompt-pack-granularity.md).

## Revisit when

A second evidence producer exists (a federated import from another machine), which would make
`source_id` semantics load-bearing in ways one producer never exercises.
