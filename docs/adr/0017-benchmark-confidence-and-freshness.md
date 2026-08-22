# ADR-0017 — Benchmark confidence and freshness

**Status:** Accepted (2026-08-21)
**Amended by:** [ADR-0022](0022-capability-evidence-record-contract.md) — `age_days` is measured from `measured_at`; [ADR-0023](0023-runtime-profile-resolution.md) — makes the runtime-profile separation operable in LoadCoach; [ADR-0028](0028-prompt-pack-granularity.md) — prompt separation is per benchmark.

## Context

LoadCoach routes on benchmark evidence. Not all evidence deserves equal weight: a coding score from
40 samples measured yesterday on the current driver is worth more than one from 3 samples measured
four months ago before an Ollama upgrade. Requirement §19 demands that benchmark version, model
digest, sample count, confidence, age, runtime/provider changes, machine changes, quantization
differences, configuration changes and result consistency all be accounted for.

Two failure modes to avoid: treating a three-sample measurement as authority, and discarding old
evidence so aggressively that routing has nothing to work with.

## Decision

**FreeWeight computes confidence; LoadCoach applies it. One formula, one owner, one implementation.**

### Confidence

```text
confidence = sample_factor × consistency_factor × freshness_factor × environment_factor × identity_factor
             clamped to [0.05, 1.0]
```

| Factor | Definition | Rationale |
|---|---|---|
| `sample_factor` | `min(1.0, sqrt(n / n_target))`, `n_target = 30` by default per capability | Diminishing returns; 3 samples ≈ 0.32, 30 ≈ 1.0 |
| `consistency_factor` | `1 − min(0.5, coefficient_of_variation)` for continuous metrics; `1 − min(0.5, disagreement_rate)` for pass/fail | A wildly variable measurement is a weak one |
| `freshness_factor` | `0.5 ** (age_days / half_life_days)`, floored at 0.3; half-life 90 d for quality, 30 d for performance/memory. **`age_days` is `now − measured_at`**, where `measured_at` is the latest `completed_at` among the contributing runs — never the aggregation time, so recomputing evidence does not make it fresher ([ADR-0022](0022-capability-evidence-record-contract.md)) | Quality is stable while the weights are; speed follows the environment |
| `environment_factor` | 1.0 with no drift; ×0.7 for a provider minor change or driver/CUDA change affecting performance evidence; ×0.5 for a provider change with template/sampling implications affecting quality evidence; ×1.0 for OS patch level | Drift reduces trust without destroying it |
| `identity_factor` | 1.0 for `digest` identity; 0.6 for `name_only` | Weights may have changed under the name |

### Hard separations (not confidence reductions)

Evidence is **discarded or partitioned**, never merely discounted, when:

* the benchmark suite version differs (different measurement);
* the dataset hash differs;
* the `prompt_subset_hash` of the prompts *that benchmark uses* differs — per benchmark, not per pack
  ([ADR-0028](0028-prompt-pack-granularity.md));
* the model digest differs (different weights);
* the runtime profile hash differs (different subject) — including a difference between the profile
  the evidence was measured under and the profile a consumer resolves for execution
  ([ADR-0023](0023-runtime-profile-resolution.md));
* the machine fingerprint differs **and** the metric is performance/memory/energy.

Quality metrics from another machine are retained with a machine badge; performance metrics are not.

### Application in routing

```text
task_fit = Σ(weight_c × score_c × confidence_c) / Σ(weight_c)
```

then multiplied by reliability, availability and cost factors. When no evidence exists for a
capability, that capability contributes nothing (it is **not** scored 0), the missing capability is
named in the routing explanation, and the resulting decision is marked as low-evidence.

### Staleness surfaces

* `stale` when `freshness_factor < 0.5` (≈ one half-life) **or** environment drift is detected.
* FreeWeight badges stale results and offers a re-run.
* LoadCoach shows evidence age and confidence in every routing explanation.
* `<app> health` reports the age of the newest evidence for each capability.

## Alternatives considered

**Treat every result equally.** Simple. Rejected: a 3-sample outlier would outrank a 100-sample
measurement.

**Discard anything older than N days.** Rejected: a user who benchmarked thoroughly once and then
stopped would lose all evidence overnight, degrading routing for no measurement reason. Decay with a
floor keeps old evidence usable while newer evidence outweighs it.

**Bayesian posterior with a prior per capability.** More principled and genuinely attractive.
Rejected for now as unexplainable to a user: the requirement is that a routing decision be
*understandable*, and "0.71 with confidence 0.62 from 40 samples, 12 days old" is understandable in a
way a posterior distribution is not. Recorded as a possible future refinement behind the same
interface.

**Let LoadCoach compute confidence.** Rejected: LoadCoach does not know benchmark internals
(dispersion, sample structure, dataset identity). Duplicating that knowledge would create two
sources of truth and put benchmark logic inside the router.

**Let each consumer choose its own policy.** Rejected: identical evidence would be weighted
differently by different consumers, making routing behaviour unexplainable across the suite.

## Consequences

*Positive.* One formula, documented and testable. Old evidence degrades gracefully instead of
vanishing. Environment drift is visible rather than silent. The distinction between "measured badly"
and "not measured" is preserved. Every parameter (half-lives, `n_target`, floors) is configurable and
recorded with the evidence.

*Negative.* The parameters are judgement calls with no empirical basis yet. Mitigated by making them
configuration, by recording the policy version on every evidence record, and by revisiting once real
usage data exists.

*Negative.* Multiplying five factors can drive confidence low quickly (a name-only, drifted, 5-sample,
90-day-old result lands near the floor). Deliberate — and the floor of 0.05 keeps it usable as a
tiebreak rather than discarding it.

## Revisit when

Real routing data allows the parameters to be fitted rather than guessed; or production feedback
shows confidence-weighted routing systematically underperforming a simpler rule.
