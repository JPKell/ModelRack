# ADR-0018 — External benchmark isolation and sandboxing

**Status:** Accepted (2026-08-21)

## Context

FreeWeight's value multiplies when it can run established benchmarks (lm-evaluation-harness, IFEval,
EvalPlus, BFCL, RULER, JudgeBench, CriticBench) alongside its native suites. Those projects bring
PyTorch, transformers, CUDA builds and pinned dependency sets that conflict with each other and would
destroy a lightweight application environment.

Some of them execute **model-generated code** to score it. The reference machine has no Docker and no
Podman; it does have `bwrap` (bubblewrap).

Requirements: keep external benchmark environments isolated; never execute generated code on the
host; pin benchmark versions and dataset hashes.

## Decision

**External benchmarks run as subprocesses in isolated environments and return normalized JSON.
Code execution obeys a tiered sandbox policy whose lowest tier is refusal.**

### Isolation

```text
FreeWeight  →  adapter  →  subprocess in an isolated environment  →  normalized JSON  →  scoring
```

* Each external benchmark has its own environment (a `venv`, a `uv` environment, or a container),
  created and pinned by the adapter, entirely separate from FreeWeight's environment.
* FreeWeight never imports an external benchmark package.
* The adapter owns: environment setup and verification, invocation with an explicit argument list
  (never a shell), a timeout, output parsing into the suite's metric model, and error translation.
* Every external benchmark has a **manifest** recording source repository, release tag, commit,
  licence, install command, dataset paths, dataset hashes, required executables, container
  requirement, and network requirement. Versions and dataset hashes are pinned; a benchmark never
  auto-updates between comparison runs.
* Adapter output is treated as untrusted input: schema-validated, size-limited, and never `eval`ed.

### Sandbox tiers for code execution

| Tier | Mechanism | Configuration |
|---|---|---|
| 1 | `podman` (preferred) or `docker` | `--network=none`, read-only rootfs, tmpfs workdir, memory/CPU/pids limits, wall-clock timeout, non-root user, all capabilities dropped |
| 2 | `bubblewrap` (`bwrap`) | `--unshare-all`, read-only binds of the minimal runtime, private `/tmp`, `--die-with-parent`, `--new-session`, rlimits + timeout |
| 3 | **Refuse** | Benchmark skipped, `sandbox_unavailable` recorded on the run with the reason |

* There is no host-execution tier, and no flag to create one.
* The tier used is recorded on every result that executed code. Correctness results are comparable
  across tiers; performance results are labelled by tier.
* Sandboxes receive no credentials, no home directory, no application database and no network.
* Tier selection is automatic (highest available), overridable downward, and reported by
  `<app> doctor`.

## Alternatives considered

**Import external benchmarks as libraries.** Rejected: guaranteed dependency conflicts, and a heavy
scientific stack inside an application that promises a small footprint.

**Require Docker for all external benchmarks.** Rejected: the reference machine has no container
runtime, and requiring one would disable every external benchmark including those that execute no
code. Containers are required only for the tier-1 sandbox.

**Run generated code in a restricted Python subinterpreter or with `RestrictedPython`.** Rejected:
in-process "sandboxes" for Python are not security boundaries.

**Skip code-execution benchmarks entirely.** Rejected: EvalPlus-style executable verification is the
single most objective coding signal available, and the scoring ladder puts execution first.

**Run them on the host with a timeout and hope.** Rejected outright — requirement §21 and basic
prudence.

## Consequences

*Positive.* The application environment stays small and installable. Dependency conflicts are
impossible by construction. Generated code never touches the host. The available tier is visible to
the user, and a missing runtime degrades to a recorded skip instead of a silent risk.

*Negative.* Subprocess boundaries mean serialized results, slower iteration, and more failure modes
to translate. Environment setup is real work for each adapter, and can be slow the first time.

*Negative.* On the reference machine, code-execution benchmarks run under bubblewrap, which is a
weaker boundary than a container (shared kernel, no cgroup-level resource caps by default). Mitigated
by rlimits, timeouts, no network, and by documenting the difference; a container is recommended in
the FreeWeight documentation for this class of benchmark.

*Negative.* Users must install external benchmarks themselves, since their licences and dataset terms
forbid redistribution. Mitigated by `freeweight external install <name>` doing the pinned setup and
verifying dataset hashes.

## Revisit when

A container runtime becomes a reasonable prerequisite for the audience, or a sandboxing mechanism
with stronger guarantees and comparable availability appears.
