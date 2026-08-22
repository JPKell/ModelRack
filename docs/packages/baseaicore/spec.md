# BaseAiCore — Specification

**Type:** Python package · **Import/distribution name:** `baseaicore` · **Layer:** 1 (domain foundation)
**Status:** Specified, not implemented.

---

## 1. Purpose

Provide the smallest stable set of domain types that every component of the suite agrees on: how a
model is named, how a machine is identified, how a measurement that does not exist is represented,
what a capability is called, and how IDs and timestamps are made. If two components would otherwise
invent the same concept twice, and that concept is pure domain vocabulary with no I/O, it belongs
here.

## 2. Scope

* Model identity, descriptor and runtime profile; measurement subject.
* Machine profile, GPU profile, storage device, machine fingerprint.
* The `Unsupported` sentinel and the `Measurement` type.
* Capability identifier type and validation (the *vocabulary contents* live in SetSpec).
* Provider kind enumeration and provider identity.
* Base error hierarchy with stable codes.
* ID generation (ULID) and timezone-aware time helpers.
* Canonical JSON and hashing helpers used by fingerprints.

## 3. Explicit non-goals

* No I/O of any kind: no HTTP, no filesystem, no database, no subprocess.
* No third-party dependencies. Standard library only.
* No serialization schemas for the wire — that is SetSpec.
* No provider communication — that is ModelRack.
* No telemetry collection — that is SweatMeter (SweatMeter *produces* BaseAiCore's `MachineProfile`).
* No scoring, routing, workflow or benchmark logic.
* No configuration loading, no logging configuration, no framework integration.
* No mutable global state beyond the `UNSUPPORTED` singleton.

## 4. Responsibilities

| Responsibility | Detail |
|---|---|
| Canonical model identity | `ModelIdentity`, `IdentityConfidence`, canonical ID string, equality and hashing ([Canonical Model Identity](../../architecture/canonical-model-identity.md)) |
| Model description | `ModelDescriptor` with architecture fields needed by KV-cache analysis |
| Runtime profile | `RuntimeProfile` and its stable `profile_hash` |
| Measurement subject | `MeasurementSubject` and comparability helpers |
| Machine identity | `MachineProfile`, `GpuProfile`, `StorageDevice`, `machine_fingerprint` ([Machine Identity](../../architecture/machine-identity-and-reproducibility.md)) |
| Unavailability | `Unsupported`, `UNSUPPORTED`, `Measurement`, `is_supported()` ([ADR-0016](../../adr/0016-unavailable-is-not-zero.md)) |
| Capability identifiers | `CapabilityId` value type with syntax validation and namespacing rules |
| Errors | `SuiteError` and the shared subclasses every layer reuses |
| Identity and time | `new_id()` (ULID), `utc_now()`, RFC 3339 formatting/parsing, duration helpers |
| Hashing | `canonical_json()`, `sha256_of()` used by every fingerprint in the suite |

## 5. Dependencies

Python ≥ 3.12 standard library. Nothing else, at runtime or as an optional extra.

## 6. Consumers

SetSpec, ModelRack, SweatMeter, WeightsDB, MirrorWall, FreeWeight, LoadCoach, IdeaPress, and any
external tool that wants suite-compatible identities.

## 7. Public API

```python
# Identity
ProviderKind(StrEnum)                    # OLLAMA, OPENAI_COMPATIBLE, LLAMACPP, VLLM, FAKE
IdentityConfidence(StrEnum)              # DIGEST, NAME_ONLY
ModelIdentity(provider_kind, provider_model_name, artifact_digest=None)
    .canonical_id -> str                 # "{kind}/{name}@sha256:<12 hex>" or "…@unknown" (ADR-0024)
normalize_digest(value: str | None) -> str | None
    # "sha256:" + 64 lowercase hex, or None. Accepts bare hex and mixed case; returns None for
    # anything that will not normalize, so a malformed digest yields a name_only identity rather
    # than a malformed one. ModelRack calls this on every provider response (ADR-0024 §2).
    .identity_confidence -> IdentityConfidence
    .with_digest(digest) -> ModelIdentity
ModelCapabilityFlag(StrEnum)             # TOOLS, VISION, THINKING, STRUCTURED_OUTPUT, EMBEDDING
ModelDescriptor(identity, observed_at, …)
RuntimeProfile(context_size=None, kv_cache_precision=None, …)
    .profile_hash -> str
MeasurementSubject(identity, runtime_profile_hash, machine_fingerprint)
    .is_comparable_with(other, *, metric_kind,
                       benchmark_version: str | None = None,
                       other_benchmark_version: str | None = None,
                       dataset_hashes: Mapping[str, str] | None = None,
                       other_dataset_hashes: Mapping[str, str] | None = None,
                       ) -> ComparabilityVerdict
    # Benchmark version and dataset hashes are NOT part of a measurement subject, but two rows of
    # the comparability matrix turn on them. They are therefore explicit arguments, and omitting
    # them yields verdict `indeterminate` with the reason naming what was not supplied — never
    # `comparable` by default. A helper that answered "yes" because it was not told what it
    # needed would be worse than no helper.

# Machine
GpuVendor(StrEnum)                       # NVIDIA, AMD, INTEL, APPLE, UNKNOWN
GpuProfile(index, name, uuid, vram_total_bytes, driver_version, cuda_version, …)
StorageDevice(name, size_bytes, model, rotational)
MachineProfile(machine_fingerprint, hostname, os_name, …, gpus, storage)
compute_machine_fingerprint(*, hostname, os_name, architecture, cpu_model,
                            physical_cores, logical_cores, ram_bytes, gpus) -> str

# Measurement
Unsupported;  UNSUPPORTED;  Measurement = int | float | Unsupported
is_supported(value) -> bool
supported_values(values) -> list[int | float]        # filters, for aggregation

# Capabilities
CapabilityId(value)                      # validated syntax: root[.specialization]*
    .root -> str ;  .is_specialization -> bool ;  .inherits_from(other) -> bool

# Errors
SuiteError(message, *, details=None)          code = "INTERNAL_ERROR"
├── ConfigurationError                        code = "CONFIGURATION_ERROR"
├── ValidationError                           code = "VALIDATION_ERROR"
├── NotFoundError                             code = "NOT_FOUND"
├── ConflictError                             code = "CONFLICT"
├── UnsupportedOperationError                 code = "UNSUPPORTED_OPERATION"
├── UnsupportedPlatformError                  code = "UNSUPPORTED_PLATFORM"
└── DependencyUnavailableError                code = "DEPENDENCY_UNAVAILABLE"

# Identity and time
new_id() -> str                          # 26-char Crockford base32 ULID, time-sortable
parse_id(value) -> UlidParts             # frozen (timestamp: datetime, randomness: bytes, text: str);
                                         # validates and raises ValidationError. There is no third-
                                         # party ULID type in a zero-dependency package.
utc_now() -> datetime                    # timezone-aware UTC
to_rfc3339(dt) -> str                    # millisecond precision, trailing Z
from_rfc3339(text) -> datetime           # rejects naive input
Clock = Callable[[], datetime]           # the injectable clock type used suite-wide

# Hashing
canonical_json(value) -> str             # sorted keys, UTF-8, stable float format,
                                         # UNSUPPORTED -> "unsupported"
sha256_of(value) -> str                  # canonical_json then sha256 hex
```

## 8. Inputs

Constructor arguments only. The package reads nothing from the environment.

## 9. Outputs

Immutable value objects, deterministic strings (canonical IDs, fingerprints, hashes, ULIDs) and
typed exceptions.

## 10. Data ownership

Owns no persistent data. It defines the shape of identity information that consumers persist; the
normative column names for identity storage are in
[Canonical Model Identity §6](../../architecture/canonical-model-identity.md).

## 11. Public contracts

1. `ModelIdentity` equality, hashing and `canonical_id` are stable across processes, machines and
   Python versions. Golden values are asserted in tests. The canonical-ID format is fixed by
   [ADR-0024](../../adr/0024-canonical-id-and-model-references.md) and is a persisted lookup key in
   three databases, so its golden test is the one that must never be "updated to match".
2. `compute_machine_fingerprint` excludes driver/toolkit versions and storage, and is stable across a
   driver upgrade.
3. `RuntimeProfile.profile_hash` is stable and ignores `None` fields.
4. `UNSUPPORTED` raises on `bool`, `int`, `float`, arithmetic and ordering; it is a singleton and
   survives pickling and copying as the same object.
5. `canonical_json` output is byte-identical for equal inputs.
6. Error `code` values are part of the public contract.

## 12. Configuration

None. Deliberately.

## 13. Error behaviour

* Invalid construction raises `ValidationError` naming the field and the expectation (empty model
  name, malformed digest, naive datetime, malformed capability ID, malformed ULID).
* No function returns a magic value on failure.
* No function catches broadly; there is nothing to catch.

## 14. Security considerations

* No I/O, so no traversal, injection or egress surface.
* `canonical_json` must not be used on values containing secrets — documented on the function.
* Hostname is part of the machine fingerprint; the fingerprint is a hash, so a shared benchmark
  export reveals no hostname unless the consumer also exports the profile. Documented so consumers
  can choose.

## 15. Performance

| Operation | Target |
|---|---|
| `ModelIdentity` construction + `canonical_id` | ≤ 5 µs |
| `RuntimeProfile.profile_hash` | ≤ 50 µs |
| `compute_machine_fingerprint` | ≤ 100 µs |
| `new_id()` | ≤ 2 µs |
| `canonical_json` on a 10 KB structure | ≤ 2 ms |

Value objects use `frozen=True, slots=True`. Hashes are computed lazily and cached on the instance.

## 16. Cross-platform

Fully portable — no platform-specific code. `UnsupportedPlatformError` is *defined* here for other
packages to raise, but nothing here branches on platform.

## 17. Observability

No logging (a library at this layer must not emit log records). Errors carry structured `details`
that consumers log. `__repr__` on every value object is stable and informative for log context.

## 18. Test strategy

| Area | Tests |
|---|---|
| Identity | Equality, hashing, canonical ID goldens, digest upgrade, `name_only` confidence, unicode and separator characters in names, model names containing `/`, `:` and `@` |
| Digest normalization | Bare hex, `sha256:`-prefixed, uppercase, wrong length, non-hex, empty, `None` — each to the documented result |
| Comparability arguments | Omitting benchmark version or dataset hashes yields `indeterminate`, never `comparable` |
| Runtime profile | Hash stability, `None` handling, ordering independence, nested `provider_options` |
| Comparability | Every cell of the comparability matrix |
| Fingerprint | Golden values, driver-change invariance, GPU-change sensitivity, `UNSUPPORTED` field handling |
| Unsupported | Every refused operation raises `TypeError`; singleton identity through copy/deepcopy/pickle; `supported_values` filtering |
| Capability IDs | Valid/invalid syntax, namespacing, inheritance |
| IDs and time | ULID monotonicity within a millisecond, sortability, round-trip, naive-datetime rejection, RFC 3339 formatting |
| Canonical JSON | Key ordering, float formatting, `UNSUPPORTED`, nesting, non-ASCII, byte-identical repeats |
| Errors | Code stability, `details` preservation, chaining |
| Packaging | Clean-venv install and import; zero third-party imports asserted by a test |

Coverage floor: **95 %** (it is a shared package); in practice this package should reach ~100 %.

## 19. Compatibility and versioning

* Semantic versioning; pre-1.0 `0.x` with minor bumps for breaking changes.
* Consumers pin `>=0.4,<0.5` pre-1.0.
* This is the most widely depended-upon package: a breaking change is a suite-wide event and needs
  a coordinated release plan in the PR description.
* Golden-value tests make an accidental change to identity, fingerprint or canonical JSON a CI
  failure rather than a silent data-compatibility break.

## 20. Acceptance criteria

1. `pip install baseaicore` in a clean venv; `import baseaicore` works with nothing else installed.
2. A five-line script builds a `ModelIdentity`, prints its canonical ID, and computes a machine
   fingerprint from literal values.
3. `UNSUPPORTED + 1`, `bool(UNSUPPORTED)`, `float(UNSUPPORTED)` and `UNSUPPORTED < 1` all raise.
4. `mypy --strict` clean; `ruff` clean; import-linter confirms zero suite and third-party imports.
5. Golden tests pass for canonical IDs, fingerprints, profile hashes and canonical JSON.
6. Coverage ≥ 95 %.
7. Every public symbol has a docstring stating its contract.

## 21. Future extensions

* Additional `ProviderKind` values as providers are added (additive).
* A `machine_fingerprint` override, if DHCP or container hostname churn proves to fragment history in
  practice. It is **not** provided today: the escape hatch would have to be honoured by every
  consumer that stores a fingerprint, and no consumer specifies one, so shipping it would create a
  setting nothing reads.
* Optional LoRA/adapter identity as a fourth optional identity field, if a provider requires it.
* `EmbeddingModelIdentity` if embedding models need distinct handling.
* A `Money`/cost value type if remote-provider cost tracking is added.
* Richer comparability verdicts (per-metric-class rules) as FreeWeight's metric taxonomy matures.
