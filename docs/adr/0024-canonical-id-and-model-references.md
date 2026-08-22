# ADR-0024 — Canonical ID format and model references in URLs

**Status:** Accepted (2026-08-21)
**Amends:** [ADR-0008](0008-canonical-model-identity.md) §"Canonical string".

## Context

The canonical ID is a persisted, indexed column in three databases, a field in every cross-application
payload, and — until this ADR — a URL path segment. It is therefore among the most expensive strings
in the suite to change after data exists.

The audit found it specified three different ways:

* ADR-0008: `{provider_kind}/{provider_model_name}@{digest[:12] or "unknown"}`. Since
  `artifact_digest` is the string `"sha256:<64 hex>"`, `digest[:12]` is `"sha256:1f3a"`.
* [Canonical Model Identity §2.1](../architecture/canonical-model-identity.md): "`digest_short` is
  the first 12 hex characters of the digest" — `"1f3a9c4e2b70"`.
* Every worked example in LoadCoach's API and routing documents:
  `ollama/qwen3.5:9b-q8_0@sha256:1f3a9c4e2b70` — the `sha256:` prefix **plus** 12 hex characters.

Three implementers would produce three incompatible lookup keys, and the mismatch would surface as
"LoadCoach cannot find the model FreeWeight measured" only after both had persistent data.

Separately, `GET /api/v1/models/{canonical_id}` uses this string as a path parameter. It contains
`/`, `:` and `@`. A percent-encoded `/` (`%2F`) is normalized or rejected by common reverse proxies
before it reaches the application, and Starlette does not match `/` in a path parameter unless the
converter is declared `:path` — which then swallows the rest of the route. Model names legitimately
contain `/` (`hf.co/user/repo:q4`), so this is not a theoretical case.

## Decision

### 1. One format, matching the examples

```python
digest_short = f"sha256:{hex[:12]}"   # when a digest is present
digest_short = "unknown"              # when the provider exposes none

canonical_id = f"{provider_kind}/{provider_model_name}@{digest_short}"
```

```text
ollama/qwen3.5:9b-q8_0@sha256:1f3a9c4e2b70
ollama/qwen3.5:9b-q8_0@unknown
```

Retaining the `sha256:` prefix keeps the hash algorithm visible in the string, which matters the day
a provider exposes a digest that is not SHA-256. A golden-value test in BaseAiCore fixes this format.

### 2. Digests are normalized by ModelRack

Providers report digests inconsistently (bare hex, `sha256:`-prefixed, upper or lower case).
`ModelIdentity.artifact_digest` is always `"sha256:" + 64 lowercase hex characters` or `None`.
Normalization happens in the adapter, and a digest that does not match that shape after normalization
is discarded with a recorded reason, producing a `name_only` identity rather than a malformed one.

### 3. The canonical ID is never a URL path segment

* `GET /api/v1/models/{model_ref}` where `model_ref` is the application-local ULID or an unambiguous
  ULID prefix. Ambiguous prefixes return 400 listing candidates, as elsewhere in the suite.
* `GET /api/v1/models?canonical_id=<percent-encoded>` is the lookup-by-identity form, and
  `?provider_kind=&provider_model_name=&artifact_digest=` is the exact-triple form for a caller that
  has the parts and should not build the string at all.
* Any other endpoint that took a `{canonical_id}` path parameter takes `{model_ref}` instead.

Applications continue to accept a canonical ID, a bare model name, or an unambiguous prefix in
**request bodies and CLI arguments**, where no encoding hazard exists; those are resolved through
`ModelRack.resolve()` and the resolution is recorded.

### 4. Still lossy, still never parsed

`provider_model_name` may contain `/`, `:` and `@`. The canonical ID is a display and lookup key,
built from the parts and never decomposed back into them. Storage keeps the three columns; the
canonical column exists for indexed lookup and display.

## Alternatives considered

**`digest[:12]` literally (ADR-0008's text).** Rejected: `sha256:1f3a` reads as a truncation error and
loses the algorithm without gaining brevity.

**Bare 12 hex characters.** Defensible and shorter. Rejected because every worked example in the
documentation set — including the ones a clean-room implementer would copy — uses the prefixed form,
and because it drops the algorithm name.

**Full 64-character digest in the canonical ID.** Rejected: unreadable in a table, a log line or a UI
badge, which is the whole purpose of this string.

**Keep the canonical ID as a path parameter and require double encoding.** Rejected: it depends on
proxy behaviour the suite does not control, and it breaks `curl` ergonomics the CLI standards value.

**Hash the canonical ID into an opaque URL-safe key.** Rejected: it destroys greppability in logs and
makes a URL unreadable for no benefit over a ULID.

## Consequences

*Positive.* One format, fixed by a golden test before any row exists. URLs work through any proxy and
with plain `curl`. Model names containing slashes are supported, which they were not. Digest handling
is normalized in exactly one place.

*Negative.* An external tool holding a canonical ID needs one extra request to resolve it to a
`model_ref`. Acceptable, and the query form makes it a single round trip.

*Negative.* `model_ref` is application-local, so a URL to a FreeWeight model does not address the same
model in LoadCoach. That was already true and is inherent to per-application databases; the canonical
ID remains the portable identifier, in payloads where it belongs.

## Revisit when

A provider exposes a non-SHA-256 artifact digest, which the format already anticipates, or identity
gains a fourth field per ADR-0008's own revisit trigger.
