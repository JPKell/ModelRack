# ADR-0025 — Envelope boundaries: what carries a SetSpec envelope

**Status:** Accepted (2026-08-21)
**Amends:** [ADR-0009](0009-setspec-schema-strategy.md) (scope of the envelope), [Master Architecture §6](../architecture/master-architecture.md) obligation 1, [Gold Standards](../standards/gold-standards.md) G6.

## Context

Master Architecture §6 said "every payload carries `schema_version`" and G6 said "every
cross-application payload is versioned". Read literally, `POST /api/v1/generate`'s request body — a
cross-application payload — needs a SetSpec envelope, and `POST /jobs/{id}/feedback` needs a schema
that was never written.

The documentation did not follow that reading. The audit found three different on-the-wire shapes for
the same event, and two for errors:

| Location | Shape |
|---|---|
| [API Standards §8](../standards/api-and-contract-standards.md) | `data: {"schema":"event.envelope","schema_version":"1.0","payload":{…}}` — nested |
| [Observability §4.1](../standards/observability-standards.md) | `{"schema":…,"schema_version":…,"event_id":…,"sequence":…,…}` — flat |
| [LoadCoach API §4](../apps/loadcoach/api.md) | `data: {"delta": "Local ", "index": 0}` — bare |
| [API Standards §4](../standards/api-and-contract-standards.md) | error body `{"error": {…}}` — unwrapped |
| [SetSpec §7](../packages/setspec/spec.md) | `ErrorEnvelope` listed as a versioned payload type |

Three applications, one shared UI package and a schema package cannot all be right. The underlying
question was never answered: *what is the envelope for?*

## Decision

**The SetSpec envelope marks a document that outlives the request that produced it. An HTTP API's
own request and response bodies are versioned by the path (`/api/v1`) and documented by OpenAPI.**

### 1. Two envelopes, clearly separated

| | SetSpec envelope | API body |
|---|---|---|
| Carries | `schema`, `schema_version`, `generated_at`, `generator`, `payload` | The resource, or `{items, page, total}` |
| Used for | Exported files, evidence bundles, benchmark results, machine profiles, model identities, events — anything a user may keep, move between machines, or read years later | Ordinary requests and responses of an application's own `/api/v1` |
| Versioned by | `schema_version`, per payload type | The path major, plus additive-only evolution |
| Rejected on | Unsupported major → `SCHEMA_VERSION_UNSUPPORTED` | Unsupported major → `API_VERSION_UNSUPPORTED` |

A body is SetSpec-enveloped **iff** its `schema` name appears in `SUPPORTED_SCHEMAS` and the endpoint
documents it as a SetSpec payload. Everything else is an API body. There is no third case.

### 2. Collections of SetSpec payloads

An endpoint returning many SetSpec documents returns the ordinary collection envelope whose `items`
are SetSpec envelopes:

```json
{"items": [{"schema": "capability.evidence", "schema_version": "1.0", "…": "…"}],
 "page": {"limit": 50, "next_cursor": null, "has_more": false}}
```

An endpoint returning **one** document — `GET /api/v1/evidence/export` — returns the SetSpec envelope
directly, with no collection wrapper, because a bundle is a single document. This is the rule that
was missing and that made `GET /evidence` ambiguous.

### 3. Events

The SSE `data:` field carries the SetSpec envelope with the event as `payload`. The flat form in
Observability §4.1 is **incorrect** and is corrected there.

```text
id: 42
event: sample.completed
data: {"schema":"event.envelope","schema_version":"1.0",
       "generated_at":"2026-08-21T09:14:02.318Z",
       "generator":{"name":"freeweight","version":"1.0.0"},
       "payload":{"event_id":"01J9K2M…","sequence":42,"type":"sample.completed",
                  "entity":{"kind":"run","id":"01J9K2M…"},
                  "timestamp":"2026-08-21T09:14:02.318Z",
                  "message":"Sample 12/40 completed",
                  "progress":{"completed":12,"total":40},
                  "data":{"sample_id":"…","score":0.75,"duration_ms":1840}}}
```

**One documented exception:** `event: token` frames during streamed generation carry a bare payload
(`{"delta": "…", "index": 0}`). A five-field envelope on every token is roughly a hundred bytes of
overhead per token on the hottest path in the suite, for a frame whose meaning is fully determined by
its event name and its stream. The exception is named here, applies only to `token`, and the frame's
shape is documented in the endpoint's OpenAPI. Every other frame — including the terminal `result`
and `error` frames — is enveloped.

### 4. Errors

The HTTP error body is `{"error": {…}}` exactly as [API Standards §4](../standards/api-and-contract-standards.md)
shows: **not** SetSpec-enveloped, because an error is a property of one request, not a document.
`setspec.ErrorEnvelope` models the inner `error` object, so all three applications and MirrorWall
agree on its fields; it remains in `SUPPORTED_SCHEMAS` as a shape, marked "transported unwrapped".

### 5. Cross-application request bodies are API bodies

`POST /generate`, `POST /jobs`, `POST /jobs/{id}/feedback` and `POST /route` carry ordinary API
bodies. They are contracts — versioned by `/api/v1`, additive-only, diff-checked through the
committed OpenAPI snapshot, and exercised by consumer contract tests against that snapshot. They do
not need SetSpec schemas, and `production.feedback` is **not** created.

The rule in Master Architecture §6 and G6 is restated accordingly: *every **transferable** payload
carries a schema version; every HTTP API is versioned in its path.*

## Alternatives considered

**Envelope everything.** Consistent, and briefly attractive. Rejected: it doubles the size of every
small response, makes `curl` output unreadable, forces a second version axis onto endpoints that
already have one, and would require a schema and three goldens for every request body in the suite —
ceremony with no reader on the other end that is not already reading OpenAPI.

**Envelope nothing; version everything by the path.** Rejected: exported files and evidence bundles
outlive both applications' API versions and are read by tools that never saw the API. That is
precisely what SetSpec is for.

**Flat event envelope (Observability's form).** One less level of nesting. Rejected: it makes the
envelope fields indistinguishable from the payload's own fields, so a reader cannot tell which are
governed by `schema_version`, and it breaks the single `load_envelope` code path SetSpec provides.

**Envelope token frames too, for uniformity.** Rejected on measured cost against zero benefit; the
exception is documented rather than discovered.

## Consequences

*Positive.* One answer to "does this body have an envelope?" that an implementer can apply without
asking. `GET /evidence` and `GET /evidence/export` stop contradicting each other. MirrorWall's
`sse_response` has one frame shape to produce and its client one to parse. No schema is invented for
a body that already has an OpenAPI contract.

*Negative.* Two versioning mechanisms coexist, and a reader must know which applies. Mitigated by the
`SUPPORTED_SCHEMAS` membership test being mechanical, and by the table above.

*Negative.* The `token` exception is a special case, and special cases get copied. Mitigated by
scoping it to exactly one event name and asserting in a contract test that no other frame is bare.

## Revisit when

A non-Python consumer needs generated clients for the HTTP APIs, at which point the OpenAPI documents
carry that weight and this decision is unaffected; or streaming volume makes even the bare token
frame a measured cost, which would be a transport decision, not an envelope one.
