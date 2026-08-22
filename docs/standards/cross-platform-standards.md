# Cross-Platform Standards

**Primary platform:** Linux (x86-64), the only platform with a support commitment before suite 1.0.
**Design requirement:** platform-specific behaviour is isolated behind interfaces so Windows and
macOS can be added later without touching business logic.

---

## 1. Support tiers

| Tier | Platform | Meaning |
|---|---|---|
| **1 — Supported** | Linux x86-64 (Ubuntu 24.04+/Debian 13+/Fedora 41+ or equivalent) | Fully implemented, CI-tested, release-blocking |
| **2 — Best effort** | Linux ARM64 | Should work; built and unit-tested in CI; hardware-specific telemetry unverified |
| **3 — Interface only** | Windows 11, macOS 14+ (Apple Silicon and Intel) | Interfaces and stubs exist; unimplemented readers raise a documented, catchable error and the application degrades. **Not** advertised as supported |

The applications themselves (FastAPI, SQLAlchemy, Jinja2, Typer) are portable. What is not portable
is telemetry, process/GPU inspection, sandboxing and path conventions — precisely the things behind
interfaces.

---

## 2. The isolation rule

`if platform.system() == "Windows"` must **never** appear in domain, service, web or CLI code. Exactly
two places may branch on platform:

1. **Factory functions** that select an implementation:

```python
def create_host_reader(*, platform_name: str | None = None) -> HostReader:
    """Return the HostReader for this platform.

    Raises:
        UnsupportedPlatformError: with the platform name and what is missing, when no
            implementation exists. Callers degrade; they do not crash.
    """
    match platform_name or sys.platform:
        case "linux":  return LinuxHostReader()
        case "win32":  return WindowsHostReader()   # NotImplemented stub, tier 3
        case "darwin": return DarwinHostReader()    # NotImplemented stub, tier 3
        case other:    raise UnsupportedPlatformError(other, feature="host telemetry")
```

2. **The implementations themselves**, each in its own module (`sweatmeter/platforms/linux.py`,
   `.../windows.py`, `.../darwin.py`).

Everything above the factory sees only the Protocol.

---

## 3. Platform-dependent surfaces

| Surface | Interface | Linux (tier 1) | Windows (tier 3) | macOS (tier 3) |
|---|---|---|---|---|
| CPU utilization, load | `HostReader` | `/proc/stat`, `/proc/loadavg` | PDH counters / WMI | `host_processor_info` / `sysctl` |
| Memory | `HostReader` | `/proc/meminfo` | `GlobalMemoryStatusEx` | `vm_stat` / `sysctl` |
| CPU temperature | `HostReader` | `/sys/class/thermal`, `/sys/class/hwmon` | WMI (often unavailable) | SMC (usually unavailable) |
| Disk throughput | `HostReader` | `/proc/diskstats` | PDH | `iostat` |
| Static machine facts | `HostReader` | `/proc/cpuinfo`, `platform` | WMI | `sysctl` |
| NVIDIA GPU | `GpuReader` | `nvidia-smi` | `nvidia-smi.exe` | n/a |
| AMD GPU | `GpuReader` | `rocm-smi` (future) | — | — |
| Apple GPU | `GpuReader` | — | — | `powermetrics` (requires privileges) |
| Sandbox | `SandboxRunner` | podman/docker → `bwrap` → refuse | container → refuse | container → refuse |
| Config/data/state paths | `PathProvider` | XDG | `%APPDATA%`, `%LOCALAPPDATA%` | `~/Library/Application Support` |
| Service management | docs only | systemd user unit | Windows Service / Task Scheduler | launchd |
| Terminal colour | Typer/Click | ANSI | ANSI (modern terminals) | ANSI |

---

## 4. Portable-by-default coding rules

* `pathlib.Path` everywhere; never string concatenation, never `os.sep`, never hard-coded `/`
  (`ruff PTH`).
* Never hard-code `/tmp`, `/proc`, `~/.config` outside a platform module — use `PathProvider` and
  `tempfile`.
* Text I/O always specifies `encoding="utf-8"`; never rely on the locale default.
* Line endings: `newline=""` for CSV, `\n` in generated text files, `.gitattributes` normalizing to LF.
* No shell invocation. Subprocesses use an argument list, an explicit executable resolved with
  `shutil.which`, a timeout, and captured output (`shell=True` is banned by lint).
* File locking, signals and process groups are used only through a platform module — SIGTERM/SIGINT
  handling has a documented Windows equivalent in the stub.
* Case-insensitive filesystems are assumed possible: never rely on two files differing only by case.
* Long paths: keep generated artifact paths under 200 characters.

---

## 5. Degradation on unsupported platforms

A tier-3 platform must still be *usable* for everything that does not need a platform reader:

| Feature | Windows/macOS today |
|---|---|
| Web UI, CLI, database, migrations, exports | Works |
| Model discovery and inference through Ollama | Works |
| Benchmark execution and scoring (quality suites) | Works |
| Machine profile | Partial — CPU model and RAM via `platform`; specifics `unsupported` |
| Live CPU/RAM/temperature telemetry | `unsupported`; telemetry bar shows `—` |
| GPU/VRAM telemetry | Works where `nvidia-smi` is on PATH; otherwise `unsupported` |
| Memory/KV-cache benchmarks | Skipped with `telemetry_unavailable` — a VRAM slope cannot be measured without VRAM readings. The same skip applies on any platform when more than one GPU is visible and placement is unreported, with reason `multi_gpu_placement_unknown` ([ADR-0027](../adr/0027-multi-gpu-semantics.md)) |
| Energy metrics | `unsupported` |
| Code-execution benchmarks | Only with a container runtime; otherwise skipped |

The rule from [Graceful Degradation](../architecture/graceful-degradation.md) applies without
exception: an unimplemented platform reader produces `unsupported` measurements and a degraded
health component — never a crash, never a fabricated number, and never a silent skip without a
recorded reason.

---

## 6. Documenting the gap

Each package with platform-specific code ships `docs/platform-support.md` stating, per platform:
what works, what is `unsupported`, what a future implementation would require (which API, which
privileges), and which tests exist. `<app> doctor` prints the same information for the running
platform, so a user on macOS immediately sees why the telemetry bar is empty.

---

## 7. Testing

* **Every platform reader is tested on every platform** by injecting captured fixture text rather
  than reading the real system: `/proc/stat`, `/proc/meminfo`, `/proc/cpuinfo`, `/proc/diskstats`,
  `/sys` thermal trees and `nvidia-smi` CSV output are all fixtures. Linux parsing is therefore
  fully tested in CI on any runner.
* Factory selection is tested with an injected platform name for every branch, including the
  unsupported-platform error.
* Stub implementations are tested to raise the documented error type with a useful message — never
  to return zeros.
* CI runs the full suite on Linux (blocking). A Windows and a macOS job run the platform-independent
  subset (`-m "not linux_only"`) as **non-blocking early warning**, so portability regressions in
  path handling, encodings and subprocess use are caught before anyone attempts a port.
* A ruff/grep check fails the build if `platform.system()`, `sys.platform` or `os.name` appears
  outside `*/platforms/*` and factory modules.

---

## 8. Adding a platform later

The work is bounded by design:

1. Implement `HostReader`, `GpuReader`, `SandboxRunner` and `PathProvider` for the platform.
2. Add fixtures captured from that platform and run the shared conformance suites against the new
   implementations.
3. Add a CI job and promote the platform's tier in this document.
4. Update `docs/platform-support.md` and the README.

No application code changes. If a port ever requires editing `freeweight/services/` or
`loadcoach/domain/`, the isolation rule has been violated somewhere and that is the defect to fix.
