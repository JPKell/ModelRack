# BaseAiCore — Specification

**Type:** Python package · **Import/distribution name:** `baseaicore` · **Layer:** 1 (domain foundation)
**Status:** Phase 1 implemented (measurement, identity, IDs, time, hashing, errors, money and
cost). Phases 2–4 specified, not implemented.

---

## 1. Purpose

Provide the smallest stable set of domain types that every component of the suite agrees on: how a
model is named, how a machine is identified, how a measurement that does not exist is represented,
what a capability is called, what a token of work costs, and how IDs and timestamps are made. If two
components would otherwise invent the same concept twice, and that concept is pure domain vocabulary
with no I/O, it belongs here.

## 2. Scope

* Model identity, descriptor and runtime profile; measurement subject.
* Machine profile, GPU profile, storage device, machine fingerprint.
* The `Unsupported` sentinel and the `Measurement` type.
* Exact money, token usage, model pricing observations and cost estimation
  ([ADR-0030](../../adr/0030-model-cost-and-pricing.md)).
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
* No price acquisition: no bundled price catalogue, no provider price lookup, no exchange rates. The
  package defines what a price *is* and how to apply one; obtaining one is ModelRack's and the
  applications' job ([ADR-0030](../../adr/0030-model-cost-and-pricing.md)).
* No currency conversion. Cross-currency arithmetic raises rather than assuming a rate.
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
| Money | `Money` as exact integer nanos in a named currency; no floats, no conversion ([ADR-0030](../../adr/0030-model-cost-and-pricing.md)) |
| Token usage | `TokenUsage` — the disjoint billable token counts of one call, each a `Measurement` |
| Model pricing | `PricingSource`, `TokenRates`, `ModelPricing` — a dated, sourced, windowed price observation with a stable `pricing_hash` |
| Cost estimation | `CostEstimate` and `estimate_cost()`; an unpriceable component makes the total `UNSUPPORTED` with a reason, never a partial sum |
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

# Money and cost (ADR-0030)
Money(currency, nanos)                   # exact billionths of one currency unit; never a float
    .zero(currency) -> Money                        # classmethod
    .from_decimal(currency, amount) -> Money        # classmethod; ROUND_HALF_EVEN to whole nanos
    .to_decimal() -> Decimal                        # display/parse boundary only, never hashed
    .as_canonical() -> dict                         # {"currency", "nanos"}; the form that is hashed
    # + and - and * int within one currency; ordering within one currency. Cross-currency
    # arithmetic or comparison raises ValidationError: converting needs an exchange rate, which is
    # time-varying data outside the user's control and is never assumed here.
normalize_currency(value) -> str         # ISO 4217 alpha-3 *shape*, uppercased; ValidationError
                                         # otherwise. The code list itself changes over time and is
                                         # deliberately not compiled into this package.

TokenCount = int | Unsupported           # the whole-number restriction of Measurement
TokenUsage(input_tokens=UNSUPPORTED, output_tokens=UNSUPPORTED,
           cache_write_tokens=UNSUPPORTED, cache_read_tokens=UNSUPPORTED)   # each a TokenCount
    # The four counts are DISJOINT: input_tokens excludes tokens billed at a cache rate. Providers
    # disagree about this (one reports cached tokens inside the prompt count, another beside it);
    # reconciling it belongs to the provider adapter in ModelRack, which is the only layer that
    # knows the convention. Getting it wrong double-counts (ADR-0030 §Consequences).
    .total_tokens -> TokenCount          # UNSUPPORTED if any component is unsupported
    .as_counts() -> dict[str, TokenCount]   # keyed by token class, in canonical class order

PricingSource(StrEnum)                   # PROVIDER_RESPONSE, PROVIDER_PUBLISHED, USER_OVERRIDE,
                                         # CATALOG, ESTIMATE
TokenRates(currency, input_per_million_tokens=UNSUPPORTED, output_per_million_tokens=UNSUPPORTED,
           cache_write_per_million_tokens=UNSUPPORTED, cache_read_per_million_tokens=UNSUPPORTED)
    # Prices as the provider quotes them — per million tokens, never pre-divided into a per-token
    # fraction. Every stated rate is in `currency` and may not be negative.
    .rate_for(token_class) -> Money | Unsupported    # "input" | "output" | "cache_write" | "cache_read"
    .as_canonical() -> dict                          # the form that goes into pricing_hash
ModelPricing(identity, rates, source, observed_at,
             effective_from=None, effective_until=None, price_tier=None, region=None)
    # A price is an observation, not a property of the model: `source` says where it came from,
    # `observed_at` when we learned it, the window when the provider says it applies, and
    # `price_tier`/`region` the dimensions the same model legitimately has several prices along.
    .pricing_hash -> str                 # canonical JSON -> sha256[:16]. Excludes `observed_at`, so
                                         # re-reading an unchanged price list yields the same hash
                                         # and a stored cost can name the exact price that produced it.
    .is_effective_at(when) -> bool       # True when no window is stated — "the provider did not
                                         # tell us" is not the same as "it expired".
CostEstimate(currency, total, input_cost, output_cost, cache_write_cost, cache_read_cost,
             pricing_hash, pricing_source, priced_at, unpriced_reasons)
    .is_complete -> bool
estimate_cost(usage, pricing, *, at) -> CostEstimate
    # `total` is UNSUPPORTED unless the pricing is effective at `at` and every token class with a
    # non-zero count has a rate; `unpriced_reasons` names each gap. A partial sum is never returned
    # as a total, and a model with no price is never costed at zero. A token class whose count is
    # zero contributes exactly zero whether or not it has a rate.

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
new_id() -> str                          # 26-char Crockford base32 ULID, time-sortable, from
                                         # one process-wide generator
UlidGenerator(*, clock=utc_now, randomness_source=None)
    .new_id() -> str                     # thread-safe, and monotonic within a millisecond: the
                                         # plain ULID spec is unordered inside one millisecond, and
                                         # the suite reads rows back in key order. Construct one
                                         # with a frozen clock and a seeded source for tests.
RandomnessSource(Protocol)               # .randbytes(n) -> bytes; random.SystemRandom by default
parse_id(value) -> UlidParts             # frozen (timestamp: datetime, randomness: bytes, text: str);
                                         # validates and raises ValidationError. There is no third-
                                         # party ULID type in a zero-dependency package.
utc_now() -> datetime                    # timezone-aware UTC
to_rfc3339(dt) -> str                    # millisecond precision, trailing Z
from_rfc3339(text) -> datetime           # rejects naive input
Clock = Callable[[], datetime]           # the injectable clock type used suite-wide
monotonic_ns() -> int                    # perf_counter_ns; a duration never comes from a clock
elapsed_ms(start_ns, end_ns=None) -> float   # milliseconds between two monotonic_ns readings

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
7. `Money` arithmetic is exact and never crosses currencies; `Money.nanos` is the stored form and
   `Decimal` appears only at the parse and display boundary.
8. `ModelPricing.pricing_hash` is stable across processes and excludes `observed_at`, so it
   identifies the price rather than the reading of it.
9. `estimate_cost` never returns a numeric total that omits an unpriced component, and never prices
   an absent rate as zero ([ADR-0030](../../adr/0030-model-cost-and-pricing.md)).

## 12. Configuration

None. Deliberately.

## 13. Error behaviour

* Invalid construction raises `ValidationError` naming the field and the expectation (empty model
  name, malformed digest, naive datetime, malformed capability ID, malformed ULID, malformed
  currency code, negative token count, negative price).
* Cross-currency `Money` arithmetic and comparison raise `ValidationError` naming both currencies.
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
| `ModelPricing.pricing_hash` | ≤ 50 µs |
| `estimate_cost` | ≤ 20 µs |

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
| Money | Exact arithmetic, cross-currency refusal on every operator, `Decimal` round-trip, rounding at the half, currency-code validation |
| Token usage | Disjointness documented and asserted, `UNSUPPORTED` propagation through `total_tokens`, negative counts rejected |
| Pricing | `pricing_hash` stability and `observed_at` exclusion, validity-window boundaries, tier/region participation in the hash |
| Cost estimation | Golden costs, unpriced rate with a non-zero count, unpriced rate with a zero count, unreported token count, pricing outside its window, total equals the sum of its components |
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
8. `estimate_cost` of a usage record against a price list that lacks one needed rate returns
   `UNSUPPORTED` with a reason naming the missing rate — not a partial sum and not zero.
9. `Money(currency="USD", nanos=1) + Money(currency="EUR", nanos=1)` raises `ValidationError`.

## 21. Future extensions

* Additional `ProviderKind` values as providers are added (additive).
* A `machine_fingerprint` override, if DHCP or container hostname churn proves to fragment history in
  practice. It is **not** provided today: the escape hatch would have to be honoured by every
  consumer that stores a fingerprint, and no consumer specifies one, so shipping it would create a
  setting nothing reads.
* Optional LoRA/adapter identity as a fourth optional identity field, if a provider requires it.
* `EmbeddingModelIdentity` if embedding models need distinct handling.
* Non-token billing units — per request, per image, per audio second, per tool call — as a sibling
  rate type alongside `TokenRates` when a supported provider needs one
  ([ADR-0030](../../adr/0030-model-cost-and-pricing.md) "revisit when").
* A `BilledAmount` type distinct from `CostEstimate`, if a provider ever returns an authoritative
  billed figure with its response.
* Richer comparability verdicts (per-metric-class rules) as FreeWeight's metric taxonomy matures.
