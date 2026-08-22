# Machine Identity and Reproducibility

**Owner:** BaseAiCore (types) and SweatMeter (collection).
**Consumers:** FreeWeight (provenance), LoadCoach (admission control, evidence validity), IdeaPress (display only).

A benchmark number without its machine is a rumour. This document defines what identifies a
machine, what must be recorded to make a measurement reproducible, and when a stored measurement
stops describing reality.

---

## 1. Static identity vs. live utilization

Two different things, two different types, two different lifetimes.

| | `MachineProfile` | `TelemetrySample` |
|---|---|---|
| Contains | Hostname, OS, kernel, CPU model, core counts, RAM, GPU models/UUIDs/VRAM, driver, CUDA/ROCm, storage devices | CPU %, RAM used/available, temperatures, GPU %, VRAM used, power, clocks, fan, disk throughput |
| Changes | Rarely (hardware or OS change) | Continuously |
| Persisted | Once per fingerprint, `last_seen_at` updated | Only while a measured run or executing job is in flight |
| Used for | Provenance, comparability, capacity limits | Live display, energy integration, peak VRAM, admission control, throttle detection |

Mixing them is a defect: a machine profile that carries "current VRAM used" makes historical rows
meaningless the moment the value changes.

---

## 2. `MachineProfile`

```python
@dataclass(frozen=True, slots=True)
class MachineProfile:
    """Static identity of the machine a measurement was produced on."""

    machine_fingerprint: str                 # see §3
    hostname: str | None
    os_name: str | None                      # "Linux"
    os_version: str | None                   # "Ubuntu 26.04 LTS"
    kernel: str | None
    architecture: str | None                 # "x86_64"
    cpu_model: str | None
    physical_cores: Measurement = UNSUPPORTED
    logical_cores: Measurement = UNSUPPORTED
    ram_bytes: Measurement = UNSUPPORTED
    gpus: tuple[GpuProfile, ...] = ()
    storage: tuple[StorageDevice, ...] = ()
    python_version: str | None = None
    observed_at: datetime | None = None      # timezone-aware UTC


@dataclass(frozen=True, slots=True)
class GpuProfile:
    index: int
    name: str | None                         # "NVIDIA GeForce RTX 5060 Ti"
    uuid: str | None                         # stable across reboots; the real identity
    vram_total_bytes: Measurement = UNSUPPORTED
    driver_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    vendor: GpuVendor = GpuVendor.UNKNOWN    # nvidia | amd | intel | apple | unknown
```

---

## 3. The machine fingerprint

```text
machine_fingerprint = sha256(
    hostname | os_name | architecture | cpu_model |
    physical_cores | logical_cores | ram_bytes |
    sorted("gpu_name:gpu_uuid" for each GPU)
)  -> 64 hex chars
```

**Deliberately included:** hardware that changes measurements — CPU, memory size, GPU set.
**Deliberately excluded**, with the reason:

| Excluded | Why |
|---|---|
| GPU driver version, CUDA/ROCm version | A driver upgrade must not orphan a machine's entire history. Version *changes* are tracked separately as environment drift (§5). |
| Attached storage | Plugging in a disk is not a new machine. |
| Python version | Application environment, not machine identity; recorded on the run. |
| Live utilization | Not identity. |
| Container/VM identifiers | Not stable, and not meaningful on the primary deployment shape. |

This exclusion policy is inherited verbatim from the prior implementation, where it was the
correct call ([inventory §2.2](../inventory/legacy-material-inventory.md)).

Unreadable fields become `UNSUPPORTED` and hash as the literal string `unsupported`. A machine that
cannot report its CPU model still gets a stable fingerprint; it simply carries less identity.

---

## 4. The reproducibility fingerprint

Computed per FreeWeight run (and stored on LoadCoach jobs at reduced scope), this is the answer to
"could this measurement be repeated, and is that other result the same thing?"

```text
reproducibility_fingerprint = sha256(canonical_json({
    "model": {
        "provider_kind": …, "provider_model_name": …, "artifact_digest": …,
        "identity_confidence": …
    },
    "runtime_profile_hash": …,
    "provider": {"kind": "ollama", "version": "0.32.13"},
    "machine_fingerprint": …,
    "environment": {                    # drift-sensitive, see §5
        "gpu_driver_version": …, "cuda_version": …, "os_version": …
    },
    "benchmark": {
        "suite_key": …, "suite_version": …, "manifest_hash": …,
        "dataset_hashes": {…}, "prompt_subset_hash": …   # only the prompts this suite uses
    },
    "execution": {
        "effective_parameters": {…},    # resolved sampling/limits, post-precedence
        "repetitions": …, "seed": …, "case_selection_hash": …,
        "served_context": …, "served_context_source": …,   # configured | reported | assumed
        "gpu_index": …                                      # the device metrics are attributed to
    },
    "application": {"name": "freeweight", "version": "1.2.0", "git_commit": "…"}
}))
```

Rules:
1. Canonical JSON: UTF-8, sorted keys, no insignificant whitespace, `Unsupported` serialized as the
   string `"unsupported"`.
1a. The **prompt subset hash**, not the pack hash, is the fingerprint input: it covers only the
   prompts the benchmark declares in its manifest, so editing an unrelated prompt does not separate
   results that share every prompt they used ([ADR-0028](../adr/0028-prompt-pack-granularity.md)).
   The installed pack's identity is still recorded on the run as provenance.
2. The **full input document is stored**, not just the hash. A hash you cannot explain is useless
   during a regression hunt.
3. Two runs with different fingerprints are never silently merged or averaged. The comparison UI
   shows a field-level diff of the two documents.
4. The fingerprint is displayed on every run detail page and included in every export.

---

## 5. Environment drift and evidence validity

Some environment changes do not change machine identity but do invalidate comparisons. Each is
recorded on the run and evaluated when evidence is used:

| Change | Effect on identity | Effect on evidence |
|---|---|---|
| GPU driver / CUDA upgrade | None | Performance and memory evidence **confidence reduced**; quality evidence unaffected |
| Provider version change (e.g. Ollama minor) | None | Performance evidence confidence reduced; if the provider's template or sampling defaults changed, quality evidence is reduced too |
| OS/kernel upgrade | None | Small confidence reduction for performance evidence |
| Benchmark suite version change | None | Results are **not comparable**; hard separation, not a confidence reduction |
| Dataset hash change | None | Same — hard separation |
| Prompt version change | None | Hard separation for any benchmark whose manifest declares that prompt; no effect on benchmarks that do not use it |
| Model digest change under the same name | New identity | Old evidence no longer applies to the new weights |
| Runtime profile change | New subject | Evidence applies only to its own profile |

The numeric decay policy is specified once, in
[ADR-0017](../adr/0017-benchmark-confidence-and-freshness.md), and implemented once, in FreeWeight's
evidence aggregation. LoadCoach applies the resulting confidence; it does not invent its own.

---

## 6. What every measured result must carry

The minimum provenance set. A result missing any of these is rejected at write time by FreeWeight
and at import time by LoadCoach.

```text
model identity (3 fields) + identity_confidence
runtime profile (full) + runtime_profile_hash
provider kind + provider version
machine fingerprint + machine profile snapshot reference
benchmark suite key + version + manifest hash + dataset hashes
prompt subset hash + the (prompt_id, version, sha256) of every prompt used
served context + its source (configured | reported | assumed)
target GPU index, and whether more than one GPU was visible
effective execution parameters (post-precedence)
repetition count, sample count, seed (or "nondeterministic")
started_at / completed_at (RFC 3339, UTC, timezone-aware)
application name + version + git commit
reproducibility fingerprint
per-metric: value, unit, aggregation, higher_is_better, sample count, dispersion
```

Optional but strongly recommended, and required for any benchmark whose numbers depend on them:
telemetry summary (peak VRAM, peak/mean power, max temperature, throttle flag), cold/warm marker,
and the raw provider response reference.

---

## 7. Reproduction workflow

A user must be able to take any stored result and re-run it:

```bash
freeweight run repeat <run_id>            # re-executes with the identical effective config
freeweight run repeat <run_id> --check    # re-executes and diffs against the original
```

`repeat` refuses, with an explicit reason, when the environment can no longer satisfy the recorded
configuration: model digest absent, provider version mismatch beyond the configured tolerance,
machine fingerprint mismatch, or dataset hash mismatch. `--force` proceeds and records the
divergence in the new run's fingerprint document rather than pretending the runs match.

---

## 8. Rules

1. Static identity and live utilization never appear in the same type or the same table.
2. The fingerprint inputs are stored, not only the hash.
3. Driver and provider versions are drift signals, never identity components.
3a. A measurement is attributed to one device. On a machine where more than one GPU is visible and
   the provider does not report placement, memory, KV and energy metrics are `unsupported` with
   reason `multi_gpu_placement_unknown` rather than attributed to a guess
   ([ADR-0027](../adr/0027-multi-gpu-semantics.md)).
4. A measurement without complete provenance is not persisted.
5. Comparisons across a fingerprint boundary are shown with the diff that separates them.
6. Missing environment information is `unsupported` and lowers confidence; it is never assumed.
