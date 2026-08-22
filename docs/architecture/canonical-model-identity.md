# Canonical Model Identity

**Owner:** BaseAiCore. **Consumers:** every component.
**Decision records:** [ADR-0008](../adr/0008-canonical-model-identity.md), [ADR-0024](../adr/0024-canonical-id-and-model-references.md) (string format, digest normalization, URL handling), [ADR-0023](../adr/0023-runtime-profile-resolution.md) (execution subject).

One model must have one name across the entire suite. This document defines that name, the types
that carry it, and the rules for when two measurements may be compared.

---

## 1. The five concepts, kept separate

Conflating these is the single most common way a system like this rots. They are separate types,
stored in separate tables, and never merged.

| Concept | Type | Mutability | Example content |
|---|---|---|---|
| **Identity** | `ModelIdentity` | Immutable | `ollama` / `qwen3.5:9b-q8_0` / `sha256:1f3a…` |
| **Descriptive metadata** | `ModelDescriptor` | Refreshable from the provider | family `qwen3.5`, 9.2 B params, Q8_0, 131 072 ctx |
| **Runtime profile** | `RuntimeProfile` | Chosen per use | ctx 32 768, KV f16, 999 GPU layers, flash-attn on |
| **Benchmark evidence** | SetSpec `CapabilityEvidence` | Accumulates, expires | `coding` 0.71, confidence 0.62, n=120, 9 days old |
| **Production evidence / routing score** | LoadCoach tables | Continuously updated | 97.8 % validation pass, p95 4.1 s, score 0.83 for `code.review` |

`ModelIdentity` never contains a score. `CapabilityEvidence` never contains a workflow. A routing
score is never persisted onto a model row.

---

## 2. `ModelIdentity`

```python
@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Immutable name of one set of weights as exposed by one provider kind.

    Identity answers "which weights?", never "how good?" and never "configured how?".
    Two identities are equal iff all three fields are equal.
    """

    provider_kind: ProviderKind      # "ollama" | "openai_compatible" | "llamacpp" | "vllm" | "fake"
    provider_model_name: str         # exactly as the provider names it, case preserved
    artifact_digest: str | None      # "sha256:<64 hex>" when the provider exposes one
```

**Why these three and nothing else**

* `provider_kind` — the same weights served by Ollama and by vLLM behave differently enough
  (templating, sampling defaults, KV handling) that measurements are not interchangeable. Kind, not
  endpoint: an endpoint is a deployment detail that must not fragment history when a port changes.
* `provider_model_name` — what a user types and what the provider accepts. Round-trips exactly.
* `artifact_digest` — the only field that survives a retag. Ollama's `qwen3:latest` can point at
  different weights across two weeks; the digest is what makes a comparison honest.

**Explicitly excluded**: endpoint URL, hostname, file path, load state, any measurement,
quantization (it belongs to the descriptor — the *name* usually encodes it, but the name is
authoritative for identity), and any user-assigned label.

### 2.1 Canonical ID string

```python
def canonical_id(self) -> str:
    """Stable, human-readable, URL-safe identity string.

    ollama/qwen3.5:9b-q8_0@sha256:1f3a9c4e2b70
    ollama/qwen3.5:9b-q8_0@unknown          # provider exposed no digest
    """
```

Rules ([ADR-0024](../adr/0024-canonical-id-and-model-references.md) is normative):
* Form: `{provider_kind}/{provider_model_name}@{digest_short}` where `digest_short` is
  `"sha256:"` followed by the **first 12 hex characters** of the digest, or the literal `unknown`.
  The algorithm prefix is retained so the string stays meaningful if a provider ever exposes a
  digest that is not SHA-256. A golden-value test in BaseAiCore fixes this format.
* `artifact_digest` itself is always `"sha256:" + 64 lowercase hex characters`, or `None`.
  ModelRack normalizes whatever the provider reports (bare hex, mixed case, prefixed) into that
  shape; anything that will not normalize is discarded with a recorded reason, producing a
  `name_only` identity rather than a malformed one.
* It is a **display and lookup** key, not a storage key. Databases store the three columns and a
  generated/indexed canonical column for lookup.
* It is never parsed to recover the parts — the parts are always available separately. (A model
  name may contain `/`, `:` and `@`; the string is lossy by design and documented as such.)
* **It is never a URL path segment.** Endpoints address models by the application-local ULID
  (`GET /api/v1/models/{model_ref}`, prefixes accepted) and look up by identity with
  `GET /api/v1/models?canonical_id=…` or the exact triple. Request bodies and CLI arguments continue
  to accept a canonical ID, a bare name or an unambiguous prefix, where no encoding hazard exists.

### 2.2 Identity confidence

```python
class IdentityConfidence(StrEnum):
    DIGEST = "digest"        # digest present; identity is exact
    NAME_ONLY = "name_only"  # provider exposes no digest; identity may drift under the same name
```

Anything that stores a measurement stores this alongside it. A `name_only` result carries a
permanent caveat: it can never be proven to describe the same weights later. FreeWeight shows this
in the UI; LoadCoach reduces evidence confidence for it
([ADR-0017](../adr/0017-benchmark-confidence-and-freshness.md)).

### 2.3 Aliases

Providers expose tags (`qwen3.5:latest` → `qwen3.5:9b-q8_0`). ModelRack records an
`alias → identity` observation with a timestamp; it does not resolve aliases silently. A run
requested by alias stores the **resolved** identity plus the alias it came from, so history stays
truthful when the tag moves.

---

## 3. `ModelDescriptor`

```python
@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Descriptive facts about a model, as reported by a provider at a point in time."""

    identity: ModelIdentity
    observed_at: datetime                     # timezone-aware, UTC
    family: str | None = None
    architecture: str | None = None
    parameter_count: Measurement = UNSUPPORTED          # total, in parameters
    active_parameter_count: Measurement = UNSUPPORTED   # MoE active params per token
    expert_count: Measurement = UNSUPPORTED
    quantization: str | None = None
    weight_format: str | None = None                    # gguf, safetensors, …
    size_bytes: Measurement = UNSUPPORTED
    max_context: Measurement = UNSUPPORTED              # advertised, not effective
    embedding_dim: Measurement = UNSUPPORTED
    layers: Measurement = UNSUPPORTED
    attention_heads: Measurement = UNSUPPORTED
    kv_heads: Measurement = UNSUPPORTED
    head_dim: Measurement = UNSUPPORTED
    vocab_size: Measurement = UNSUPPORTED
    rope_config: Mapping[str, Any] | None = None
    sliding_window: Measurement = UNSUPPORTED
    declared_capabilities: frozenset[ModelCapabilityFlag] = frozenset()
    license_text: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)   # untouched provider response
```

* Architecture fields (`layers`, `kv_heads`, `head_dim`) are load-bearing: FreeWeight's KV-cache
  benchmark computes a theoretical bytes-per-token from them and compares it against the observed
  VRAM slope. Missing fields make that benchmark report `unsupported`, not a wrong number.
* `max_context` is what the model *advertises*. Effective context is a measurement and lives in
  benchmark results.
* `declared_capabilities` are provider claims (`tools`, `vision`, `thinking`, `structured_output`),
  distinct from *measured* capabilities.
* `raw` is preserved for diagnostics and for extracting fields the normalizer does not yet know.
  Nothing above ModelRack may read `raw` for business logic.

---

## 4. `RuntimeProfile`

The same weights under different runtime settings are different measurement subjects. This was the
single most valuable correction inherited from the prior benchmark spec.

```python
@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """How a provider is asked to run a model. Hashes to a stable key."""

    context_size: int | None = None
    kv_cache_precision: str | None = None       # "f16" | "q8_0" | "q4_0" | …
    gpu_layers: int | None = None
    flash_attention: bool | None = None
    threads: int | None = None
    batch_size: int | None = None
    keep_alive: str | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def profile_hash(self) -> str:
        """SHA-256 over the canonical JSON of every non-None field, first 16 hex chars."""
```

Sampling parameters (`temperature`, `top_p`, `top_k`, `seed`, `max_output_tokens`, stop sequences)
are **request** parameters, not runtime profile: they change per benchmark test and per task, and
are recorded on the sample/job. The runtime profile describes how the model is *loaded and served*.

---

## 5. Measurement subject and comparability

```python
@dataclass(frozen=True, slots=True)
class MeasurementSubject:
    identity: ModelIdentity
    runtime_profile_hash: str
    machine_fingerprint: str
```

**Comparability rules** — used by FreeWeight's comparison view, its regression detection, and
LoadCoach's evidence import:

| Comparison | Allowed? | Presentation |
|---|---|---|
| Same subject, same benchmark version, same dataset hash | Yes | Direct comparison |
| Same subject, different benchmark version | No | Must be labelled and separated; never averaged |
| Same identity + runtime profile, different machine | Quality metrics: yes with a machine badge. Performance/memory metrics: **no** | Split by machine |
| Same identity, different runtime profile | Yes, as an explicit *runtime comparison* (this is the KV-precision and context-size study) | Side by side, never merged |
| Same family, different quantization | Yes, as an explicit *quantization comparison* | Side by side, never merged |
| `name_only` identity across a gap in time | Flagged | Warning: weights may have changed under this name |
| Same measurement subject, but a consumer resolved a **different runtime profile** for execution | No | Evidence does not apply to that execution; named as `evidence_profile_mismatch` with both hashes ([ADR-0023](../adr/0023-runtime-profile-resolution.md)) |

`MeasurementSubject.is_comparable_with(other, *, metric_kind)` decides only the rows above that turn
on identity, profile and machine. Benchmark version and dataset hash are **not** part of a
measurement subject, so the two rows that turn on them take those values as explicit arguments:
`is_comparable_with(other, *, metric_kind, benchmark_version=…, dataset_hashes=…)`, with the
benchmark arguments optional and the verdict marked `indeterminate` when they are omitted. A helper
that silently answers "comparable" because it was not told what it needed would be worse than no
helper.

The UI never silently averages across a boundary marked "no". Where a chart mixes subjects, the
grouping dimension is always visible.

---

## 6. Persistence

Every application that stores models uses the same column set (names are normative so that exports
and imports line up):

```text
provider_kind          TEXT NOT NULL
provider_model_name    TEXT NOT NULL
artifact_digest        TEXT NULL
canonical_id           TEXT NOT NULL      -- generated, indexed
identity_confidence    TEXT NOT NULL      -- 'digest' | 'name_only'
UNIQUE (provider_kind, provider_model_name, artifact_digest)
```

`NULL` digests do not collide under SQL `UNIQUE` semantics, so applications additionally enforce
"one `name_only` row per (kind, name)" in the repository layer, with a test proving that a model
first seen without a digest and later seen with one is **upgraded in place** rather than duplicated.

The same upgrade rule applies **across** applications, when LoadCoach imports evidence that carries a
digest for a model it knows only by name: the local row is upgraded and the evidence bound. The
reverse — `name_only` evidence arriving for a model held locally with a digest — is **not** matched,
because the weights cannot be proven to be the installed ones. Both cases, and the retention of
unmatched evidence, are specified in
[ADR-0022 §4](../adr/0022-capability-evidence-record-contract.md).

---

## 7. Rules

1. Every component that names a model uses `ModelIdentity`. No component invents a second naming scheme.
2. Identity is created **only** by ModelRack from a provider response, or by parsing a user-supplied
   reference through `ModelRack.resolve()`. Applications never construct identities from string
   fragments. The one exception is deserializing an identity that arrived in a SetSpec payload, which
   goes through `setspec.model.v1.ModelIdentityPayload` and reconstructs the three fields directly —
   never by splitting a canonical ID.
3. A measurement is never stored without its full `MeasurementSubject`.
4. Descriptor refreshes never rewrite history: results keep the descriptor snapshot they were
   produced with.
5. Retagging is an observation, not a mutation.
6. If a provider offers no digest, that fact is recorded, surfaced and propagated — never hidden.
