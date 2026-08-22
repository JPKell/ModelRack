# ADR-0027 — Multi-GPU semantics

**Status:** Accepted (2026-08-21)
**Extends:** [ADR-0021](0021-telemetry-collection-strategy.md), [Queue and Scheduling §5–6](../apps/loadcoach/queue-and-scheduling.md).

## Context

SweatMeter reports GPUs as a tuple and its acceptance criteria require multi-GPU fixtures to parse
correctly. `MachineProfile.gpus` is a tuple, and the machine fingerprint hashes the sorted GPU set.
Multi-GPU is, on paper, supported.

Everything above SweatMeter is written for one GPU. LoadCoach's admission control reads
`fits = estimate_vram + vram_headroom_bytes <= free_vram` with a single `free_vram` and a single
scalar `vram_headroom_bytes`. `residency.vram_bytes` is one column. FreeWeight's memory and energy
metrics — `peak VRAM`, `observed_kv_bytes_per_token`, `mean GPU power` — name no device.

Two failure modes follow on a two-GPU machine, and both are silent:

* Summing free VRAM across devices admits a model that fits nowhere, because weights must land on one
  device unless the runtime is explicitly told to shard. The result is an OOM the estimate said could
  not happen — the precise outcome LoadCoach's conservative-admission trade-off exists to avoid.
* A VRAM slope measured while the provider placed the model on the *other* device reads as zero
  bytes per token, and a zero that came from looking at the wrong sensor is exactly the fabricated
  measurement ADR-0016 forbids.

The reference machine has one GPU, so neither would be caught locally.

## Decision

**No component sums or averages across GPUs. Every VRAM, power and residency figure names its device.**

### 1. SweatMeter is unchanged and remains per-device

`GpuSample` already carries `index` and `uuid`. `TelemetryWindow`'s methods already take
`gpu_index`. This ADR adds no API; it constrains the consumers.

### 2. LoadCoach admission evaluates devices independently

```text
fits = any( estimate_vram + vram_headroom_bytes <= free_vram(g)  for g in visible_gpus )
```

* `vram_headroom_bytes` is **per device**.
* The device that satisfied the check is recorded on the job and in the routing explanation as
  `target_gpu_index`, with the numbers that justified it.
* Above `max_concurrent_jobs = 1`, the aggregate check is per device: concurrent jobs targeting the
  same device sum against that device's free VRAM, not against the machine's total.
* Residency is tracked per device: `residency` gains `gpu_index`, and `max_resident_models` is
  interpreted per device.
* **Cross-device sharding is out of scope for 1.0.** When a model's estimate fits no single device,
  it does not fit; the rejection reason `insufficient_vram` reports the per-device numbers. A
  provider that shards anyway is outside what LoadCoach can estimate, and the job records the
  divergence between estimate and observed placement like any other estimate error.

### 3. FreeWeight measures one target device

* `execution.gpu_index` (default `0`) names the device whose metrics are attributed to the run, and
  is recorded on the run and in the reproducibility fingerprint's execution section.
* When more than one GPU is visible, the run records `multi_gpu_visible: true`.
* Memory, KV-cache and energy metrics are `unsupported` with reason `multi_gpu_placement_unknown`
  when more than one GPU is visible **and** the provider does not report which device holds the
  model — because a slope measured against the wrong device is not a small error, it is a wrong
  number. Quality, throughput and latency metrics are unaffected and continue normally.
* Telemetry is persisted for **every** visible GPU during a run regardless, so the placement can be
  seen after the fact.

### 4. Host metrics appear once per sample

FreeWeight's `telemetry_samples` writes one row per GPU per sample, which duplicates every host field
(`cpu_percent`, `ram_used_bytes`, …) across those rows. Any host aggregate would double-count on a
two-GPU machine.

The table is split: `telemetry_samples` holds the timestamp and the host fields, one row per sample;
`telemetry_gpu_samples` holds the per-device fields with a foreign key to it and `gpu_index`. A
machine with no GPU produces host rows and no GPU rows, which also removes the "one row per sample
with `gpu_index NULL`" special case.

### 5. Aggregates name their device

Every derived figure that came from a device — peak VRAM, mean and peak power, energy, maximum
temperature, throttle verdict — carries `gpu_index` in storage, in exports and in the UI. There is no
machine-wide "GPU power" number.

## Alternatives considered

**Sum free VRAM across devices.** The implicit status quo. Rejected: it admits work that cannot run,
and the failure is an OOM rather than a deferral.

**Support cross-device sharding at 1.0.** Rejected: the estimate would have to model the runtime's
layer-splitting policy per provider, which is neither documented nor stable, for a deployment shape
the reference machine cannot test. It is recorded as a future extension with a real trigger.

**Declare multi-GPU unsupported and refuse to start.** Rejected: the applications work fine on such a
machine; only three families of *measurement* are ambiguous, and marking those `unsupported` with a
reason is the suite's established answer.

**Keep one telemetry table and de-duplicate host fields in queries.** Rejected: it makes correctness
depend on every future query remembering to do so, which is how the fabricated-number class of bug
gets back in.

## Consequences

*Positive.* Admission stops promising what the hardware cannot deliver. Memory and energy numbers
either name their device or say they cannot. The host/GPU table split removes a double-counting trap
and makes the no-GPU case ordinary rather than special. Multi-GPU stops being a claim that no
consumer honours.

*Negative.* One more table, one join for a combined telemetry chart, and `gpu_index` threaded through
metrics, exports and the UI. The join is indexed and inside the existing budget.

*Negative.* On a multi-GPU machine without provider-reported placement, memory and KV benchmarks are
unavailable rather than approximate. Deliberate, and consistent with every other measurement decision
in the suite.

## Revisit when

A provider reports model placement per device (vLLM exposes this), which would make multi-GPU memory
measurement possible and turn §3's `unsupported` into a real measurement; or a second GPU joins the
reference machine and sharding becomes a use case rather than a hypothetical.
