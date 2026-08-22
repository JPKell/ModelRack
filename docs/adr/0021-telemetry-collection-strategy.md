# ADR-0021 — Telemetry collection strategy

**Status:** Accepted (2026-08-21)

## Context

SweatMeter must report CPU, RAM, load, GPU utilization, VRAM, temperatures, power, fan and clocks,
plus a static machine profile — on Linux with NVIDIA first, degrading gracefully everywhere else,
with unavailable values represented as unavailable rather than zero.

The prior projects took two different approaches: `local_model_benchmarks` read `/proc` and `/sys`
directly and shelled out to `nvidia-smi`, with every reader injectable so parsing was tested against
captured fixture text; `content_factory` used `psutil` plus `nvidia-smi`, best-effort, with failures
swallowed. The first is the better design and is adopted.

## Decision

**Direct `/proc` and `/sys` reads for host metrics; `nvidia-smi` CSV subprocess for NVIDIA GPUs;
every reader injectable; no third-party runtime dependency.**

1. **Host metrics (Linux):** `/proc/stat`, `/proc/meminfo`, `/proc/loadavg`, `/proc/diskstats`,
   `/proc/cpuinfo`, `/sys/class/thermal`, `/sys/class/hwmon`, `/sys/block`. No `psutil`.
2. **NVIDIA:** `nvidia-smi --query-gpu=… --format=csv,noheader,nounits` with an explicit argument
   list (never a shell), a timeout, and defensive CSV parsing. Multi-GPU supported; per-field
   availability handled individually.
3. **Injectable readers.** Every reader is a callable or small class supplied through the
   constructor: a text source for each `/proc` file, a `/sys` root path, an `nvidia-smi` runner, and
   a clock. Tests exercise the parsers against captured fixture text, so **all parsing is tested on
   any machine**, GPU or not.
4. **No sensor failure escapes.** A `_safe()` wrapper degrades exactly one field to `UNSUPPORTED`,
   logs at DEBUG, and never propagates. `snapshot()` and `machine_profile()` do not raise.
5. **Unavailable ≠ zero** ([ADR-0016](0016-unavailable-is-not-zero.md)) throughout.
6. **Platform isolation** ([Cross-Platform Standards](../standards/cross-platform-standards.md)):
   `HostReader` and `GpuReader` are Protocols selected by a factory; Windows/macOS implementations
   are documented stubs that raise a catchable, informative error.
7. **`pynvml` is an optional extra**, not a dependency: when installed it may replace the `nvidia-smi`
   subprocess for lower-overhead sampling, behind the same `GpuReader` interface, and the conformance
   suite runs against both.
8. **Sampling** is a thread with a configurable interval (default 1 s), a callback/iterator
   interface, no internal persistence, and a measured overhead budget (≤ 1 % of one core, ≤ 1 %
   effect on measured throughput).

## Alternatives considered

**`psutil`.** Mature, cross-platform, would deliver Windows and macOS host metrics immediately.
Rejected as a *required* dependency: it is a compiled package (wheel availability constrains the
supported Python matrix), it does not cover NVIDIA GPUs (the metrics that matter most here), and
direct `/proc` parsing on the primary platform is simple, fast and exactly fixture-testable. Recorded
as the natural implementation for the tier-3 Windows/macOS `HostReader` stubs, as an optional extra
— which is precisely the shape the platform isolation was designed for.

**`pynvml` / NVML bindings as the primary GPU source.** Lower overhead and richer data than parsing
CSV. Rejected as primary: it adds a dependency whose version must track the driver, and
`nvidia-smi` is present wherever a driver is. Available as an optional accelerator behind the same
interface.

**`nvidia-smi --query --xml`.** Rejected: heavier to parse, slower, and no additional field the suite
needs.

**A DCGM/exporter-based approach.** Rejected: a daemon to install and run for a local tool.

**Best-effort with swallowed exceptions (the `content_factory` approach).** Rejected: it hides
whether a metric is unavailable or the reader is broken, which is precisely the distinction this
suite must preserve.

## Consequences

*Positive.* Zero runtime dependencies for SweatMeter. Every parser is unit-tested against fixture
text on any runner. One failing sensor degrades one field. `nvidia-smi` is available wherever an
NVIDIA driver is, so there is nothing extra to install. Adding AMD (`rocm-smi`) or Apple later means
one new `GpuReader`, no changes above it.

*Negative.* Subprocess cost per GPU sample (~20–40 ms measured), which sets the practical floor for
the sampling interval and is why the optional `pynvml` path exists. Budgeted and tested.

*Negative.* `nvidia-smi` output format can change between driver versions. Mitigated by parsing by
**column name** rather than position, by tolerating missing columns as `UNSUPPORTED`, and by fixtures
captured per driver version with the version recorded.

*Negative.* Windows and macOS host telemetry does not exist yet, so the telemetry bar is empty there
and memory/KV benchmarks are skipped with a recorded reason. Documented per platform and reported by
`<app> doctor`.

## Revisit when

* Sampling overhead becomes material for a use case (then: promote `pynvml` to the default GPU
  reader where available).
* Windows or macOS moves to tier 1 (then: implement those `HostReader`s, likely on `psutil` as an
  extra).
* A non-NVIDIA GPU becomes a primary target.
