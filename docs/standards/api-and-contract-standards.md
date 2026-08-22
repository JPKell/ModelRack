# API and Contract Standards

**Applies to:** every HTTP API and every cross-application payload in the suite.
**Related:** [ADR-0013 API versioning](../adr/0013-api-versioning.md), [ADR-0009 SetSpec schema strategy](../adr/0009-setspec-schema-strategy.md), [ADR-0025 envelope boundaries](../adr/0025-envelope-boundaries.md), [ADR-0004 SSE](../adr/0004-sse-vs-websockets.md), [ADR-0014 authentication](../adr/0014-authentication-strategy.md), [ADR-0026 local HTTP hardening](../adr/0026-local-http-hardening.md).

A cross-application interface is a public contract. Once released it changes additively or it
changes its major version — there is no third option.

---

## 1. Versioning

* Path-versioned: `/api/v1/…`. The version is the **major** version only.
* Within a major version, only additive changes: new endpoints, new optional request fields, new
  response fields. Clients must ignore unknown response fields.
* Breaking changes (removing or renaming a field, changing a type, tightening validation, changing
  an error `code`'s meaning, changing default behaviour) require `/api/v2/`.
* When `v2` ships, `v1` remains available for at least one minor release of the application and is
  documented as deprecated with a removal version and a migration note.
* `GET /api/v1/version` returns the application version, the API major versions served, and the
  SetSpec schema versions understood:

```json
{
  "application": {"name": "loadcoach", "version": "1.3.0", "git_commit": "a1b2c3d"},
  "api": {"current": "v1", "supported": ["v1"], "deprecated": []},
  "schemas": {"benchmark.result": ["1.0", "1.1"], "capability.evidence": ["1.0"]}
}
```

* Clients check compatibility on first contact and fail loudly with both versions named
  (`API_VERSION_UNSUPPORTED`) rather than guessing.
* **`GET /api/v1/version` is never authenticated.** Negotiation happens before a client can know
  whether its credential is valid, and requiring a scope would make a bad token and an incompatible
  API indistinguishable. It returns version metadata only — no names, no counts, no configuration.
  `GET /api/v1/health` does require `read` when authentication is on, because its component detail is
  operational information ([ADR-0026 §5](../adr/0026-local-http-hardening.md)).

---

## 2. Resource naming

* Plural, lowercase, hyphenated nouns, named for the resource as the owning application calls it:
  `/api/v1/runs`, `/api/v1/jobs`, `/api/v1/task-profiles`. Hyphenate only where the name is genuinely
  two words; do not prefix a resource with its application's subject matter (`/runs`, not
  `/benchmark-runs` — the base path already says whose runs these are).
* Two documented singular exceptions, because they name one thing rather than a collection:
  `/api/v1/queue` (this process's queue) and `/api/v1/health`.
* Hierarchy expresses ownership: `/api/v1/runs/{run_id}/tests/{test_id}/samples`.
* Actions that are not CRUD are sub-resources with a verb, POST-only:
  `POST /api/v1/runs/{run_id}/cancel`, `POST /api/v1/evidence/import`.
* Query parameters are `snake_case`; so are all JSON keys everywhere in the suite.
* No verbs in collection paths. No `/getRun`, no `/api/v1/run_list`.

| Method | Use | Idempotent |
|---|---|---|
| GET | Read. Never mutates. | Yes |
| POST | Create, or invoke an action | No (unless an idempotency key is supplied) |
| PUT | Full replace of a known resource | Yes |
| PATCH | Partial update | No |
| DELETE | Remove | Yes |

---

## 3. Request and response bodies

* JSON only (`application/json; charset=utf-8`) for API endpoints. `text/event-stream` for streams.
  HTML is served from separate UI routes, never from `/api/`.
* Request bodies are validated by pydantic models; unknown fields are **rejected** on requests
  (`extra="forbid"`) so typos surface immediately, and **preserved** on inbound cross-application
  payloads (see §7).
* A single resource is returned as the object itself, not wrapped:

```json
{"run_id": "01J9…", "status": "completed", "created_at": "2026-08-21T09:14:02.318Z"}
```

* A body that is a **SetSpec payload** (an export, an evidence bundle, a result, a machine profile)
  is returned inside the SetSpec envelope and is never additionally wrapped. A body that is an
  ordinary API resource is returned bare. The membership test — does its `schema` name appear in
  `SUPPORTED_SCHEMAS`, and does the endpoint document it as a SetSpec payload — is in
  [ADR-0025](../adr/0025-envelope-boundaries.md).
* Collections are wrapped, always with the same envelope:

```json
{
  "items": [ … ],
  "page": {"limit": 50, "next_cursor": "eyJvZmZzZXQiOjUwfQ", "has_more": true},
  "total": 1284
}
```

`total` is optional and omitted when counting is expensive; `has_more` is always present.

* A collection **of** SetSpec payloads puts SetSpec envelopes in `items`; the two envelopes nest in
  exactly that order and never the reverse. A single SetSpec document — `GET /evidence/export` —
  is returned as the envelope alone, with no collection wrapper, because a bundle is one document.

---

## 4. Errors

One shape, everywhere, for every non-2xx response. It is **not** wrapped in a SetSpec envelope: an
error describes one request rather than a document that outlives it. `setspec.ErrorEnvelope` models
the inner `error` object so every application and MirrorWall agree on its fields, and it is
transported unwrapped ([ADR-0025 §4](../adr/0025-envelope-boundaries.md)).

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'qwen3.5:70b' is not available from provider 'ollama'.",
    "details": {"provider_kind": "ollama", "requested": "qwen3.5:70b", "available_count": 11},
    "request_id": "01J9K2M4P7Q8R9S0T1U2V3W4X5",
    "timestamp": "2026-08-21T09:14:02.318Z"
  }
}
```

* `code` is `SCREAMING_SNAKE_CASE`, stable, and documented in the component's spec. Clients branch
  on `code`, never on `message`.
* `message` is one human-readable sentence, safe to display, free of secrets, paths outside the
  data root, and stack traces.
* `details` is structured and machine-usable; it never contains credentials or prompt content.
* `request_id` appears on every response (success and failure) and in every log line for that
  request.

**Status code mapping**

| Status | Meaning | Example codes |
|---|---|---|
| 400 | Malformed or invalid request | `VALIDATION_ERROR`, `SCHEMA_VERSION_UNSUPPORTED` |
| 401 | Missing/invalid credentials | `UNAUTHENTICATED` |
| 403 | Authenticated but not permitted | `FORBIDDEN` |
| 404 | Resource does not exist | `RUN_NOT_FOUND`, `MODEL_NOT_FOUND` |
| 409 | State conflict | `RUN_ALREADY_RUNNING`, `JOB_NOT_CANCELLABLE` |
| 413 | Body too large | `PAYLOAD_TOO_LARGE` |
| 422 | Semantically invalid though well-formed | `CAPABILITY_UNSUPPORTED`, `CONTEXT_LIMIT_EXCEEDED` |
| 429 | Rate/queue limit | `QUEUE_FULL`, `RATE_LIMITED` |
| 499 | Client closed the request | `CLIENT_DISCONNECTED` (logged, not returned) |
| 500 | Unexpected server fault | `INTERNAL_ERROR` |
| 503 | Dependency unavailable | `PROVIDER_UNAVAILABLE`, `INSUFFICIENT_RESOURCES` |
| 504 | Upstream timeout | `PROVIDER_TIMEOUT` |

A validation error lists every field problem at once:

```json
{"error": {"code": "VALIDATION_ERROR", "message": "Request body failed validation.",
  "details": {"fields": [{"path": "options.temperature", "problem": "must be between 0 and 2, got 3.5"}]},
  "request_id": "…", "timestamp": "…"}}
```

---

## 5. Request IDs, headers and timestamps

* Every request gets a request ID: taken from `X-Request-ID` if the client supplied a valid one
  (ULID or UUID, ≤ 64 chars, safe charset), otherwise generated. Returned in `X-Request-ID` and
  echoed in the body's `request_id`, and bound to every log record for that request.
* Timestamps: RFC 3339 / ISO 8601, **UTC**, millisecond precision, trailing `Z` —
  `2026-08-21T09:14:02.318Z`. Never naive, never local, never epoch seconds in JSON.
* Durations: integers or floats in the unit named by the field (`duration_ms`, `ttft_ms`,
  `interval_seconds`).
* Standard headers on every response: `X-Request-ID`, `X-Api-Version` (`v1`), `Cache-Control`
  (`no-store` for API responses by default), `Content-Type`.

---

## 6. Pagination, filtering, sorting

* **Cursor pagination** for anything that grows without bound (samples, events, jobs, results):
  `?limit=50&cursor=…`. Cursors are opaque base64 of a stable sort key; clients never construct them.
* Offset pagination (`?limit=&offset=`) is permitted only for small bounded collections (task
  profiles, benchmark definitions) and is documented as such.
* `limit` default 50, maximum 500. Over-limit requests are clamped and the response says so in
  `page.limit`.
* Filtering uses explicit named parameters (`?status=failed&model=ollama%2Fqwen3.5%3A9b`), never a
  query DSL.
* Sorting: `?sort=-created_at` (leading `-` for descending). Only documented sortable fields are
  accepted; anything else is a `VALIDATION_ERROR`.
* Sort order is always total (ties broken by ID) so cursor pagination cannot skip or repeat rows.

---

## 7. Cross-application payloads (SetSpec)

Every **transferable** payload — a document that outlives the request that produced it: an export, an
evidence bundle, a benchmark result, a machine profile, a model identity, an event frame — is a
SetSpec model and carries its version. An HTTP API's own request and response bodies are versioned by
their path and contracted through the committed OpenAPI snapshot instead; `POST /generate`,
`POST /jobs` and `POST /jobs/{id}/feedback` therefore carry no envelope, and no SetSpec schema is
created for them ([ADR-0025](../adr/0025-envelope-boundaries.md)).


```json
{
  "schema": "benchmark.result",
  "schema_version": "1.0",
  "generated_at": "2026-08-21T09:14:02.318Z",
  "generator": {"name": "freeweight", "version": "1.0.0"},
  "payload": { … }
}
```

Rules:

1. `schema_version` is `"MAJOR.MINOR"`. **Major** = breaking. **Minor** = additive.
2. Readers accept any minor within a supported major, including minors newer than they know.
3. Readers reject an unsupported major with `SCHEMA_VERSION_UNSUPPORTED`, naming both versions.
   They never "try their best".
4. Readers **preserve unknown fields** on round-trip. Each payload type has a strict outbound model
   (`extra="forbid"`) for writers and a preserving inbound model (`extra="allow"`) for readers; a
   re-export goes through the inbound model, so an older reader does not silently strip a newer
   writer's fields.
5. Writers never emit unknown fields, and never emit `null` where the field means "unsupported" —
   `"unsupported"` is the value for that.
6. Every schema publishes generated JSON Schema and golden example payloads as package data, used
   by contract tests in both producer and consumer.
7. Schema versions and API versions are independent. Bumping one does not bump the other.
8. `SUPPORTED_SCHEMAS` declares supported **majors**, not an exhaustive list of versions — a newer
   minor within a supported major is accepted by rule 2, so exact-version matching would contradict
   the reader policy.

---

## 8. Streaming

Server-Sent Events, not WebSockets ([ADR-0004](../adr/0004-sse-vs-websockets.md)). One convention
across all three applications.

```text
GET /api/v1/runs/{run_id}/events        Accept: text/event-stream
GET /api/v1/jobs/{job_id}/stream
GET /api/v1/system/telemetry/stream
```

Frame format:

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

: heartbeat 2026-08-21T09:14:17.004Z
```

This is the **only** event frame shape. The envelope fields are siblings of `payload`, and the event's
own fields live inside it; a flat form in which they mix is a defect
([ADR-0025 §3](../adr/0025-envelope-boundaries.md)).

Rules:

* `id` is a **gap-free, per-stream, monotonically increasing** integer starting at 1. Reconnecting
  clients send `Last-Event-ID` and receive everything after it, with no gap and no duplicate.
* Events are persisted before they are published; the store is the source of truth and the in-memory
  fan-out is only a latency optimization. This is what makes replay and restart-survival work.
* Event names are `noun.verb` in past tense: `run.started`, `sample.completed`, `job.state_changed`,
  `telemetry.sampled`, `stage.failed`.
* `data` is always one JSON object using the event envelope schema; never bare text, never multi-line
  JSON.
* Heartbeat comment every 15 s so proxies and clients can detect a dead connection.
* Terminal event (`run.completed`, `run.failed`, `run.cancelled`, `job.completed`, …) is always sent
  before the server closes; clients treat closure without a terminal event as an interruption and
  reconnect.
* Subscriber queues are bounded; a client that cannot keep up is dropped and expected to reconnect
  and replay. A slow client never grows server memory without bound.
* Streamed **generation** (token deltas) uses the same transport with `event: token`, plus a final
  `event: result` carrying usage and timings.
* **One documented exception to the envelope**: `event: token` frames carry a bare payload
  (`{"delta": "…", "index": 0}`), because a five-field envelope per token is roughly a hundred bytes
  of overhead on the hottest path in the suite for a frame whose meaning is fully determined by its
  name and its stream. The exception applies to `token` and to nothing else; the terminal `result`
  and `error` frames are enveloped, and a contract test asserts no other frame is bare.
* **SSE handlers are `async def`, and the event store is synchronous.** Every read into the store —
  replay and each live batch — is dispatched to the worker threadpool by MirrorWall's `sse_response`.
  A handler that queries the database on the event loop is a defect
  ([ADR-0003 §6–8](../adr/0003-sync-vs-async-strategy.md)).

---

## 9. Authentication

Detail in [ADR-0014](../adr/0014-authentication-strategy.md) and
[Security Standards](security-standards.md).

* Loopback binding + no configured tokens ⇒ no authentication required (the local-first default).
* Any non-loopback binding **requires** at least one API token; startup refuses otherwise.
* Scheme: `Authorization: Bearer <token>`. Tokens are 32 random bytes, base32-encoded, shown once at
  creation, stored only as a SHA-256 hash, compared in constant time.
* Tokens carry a scope: `read`, `write`, `admin`. Reads never require `write`.
* `401` for missing/invalid credentials, `403` for insufficient scope. Both are logged with the
  request ID and the source address, never with the token.

---

## 10. Limits and safety

| Limit | Default | Configurable |
|---|---|---|
| Request body | 4 MiB (`prompt`-bearing endpoints: 16 MiB) | Yes |
| Upload (import files) | 128 MiB | Yes |
| URL length | 8 KiB | No |
| Header size | 16 KiB | No |
| Concurrent SSE connections per process | 200 | Yes |
| Request timeout (non-streaming) | 120 s | Yes |
| Queue depth (LoadCoach) | 1 000 | Yes |

* CORS is **disabled** by default. Enabling it requires an explicit origin allowlist; `*` is
  rejected when authentication is enabled. Enabling CORS also makes bearer tokens mandatory, because
  the JSON API's CSRF exemption depends on cross-origin requests failing preflight
  ([ADR-0026 §2](../adr/0026-local-http-hardening.md)).
* The `Host` header is validated against an allowlist before routing; a mismatch is **421**. On a
  loopback bind the allowlist is `localhost`, `127.0.0.1`, `[::1]` and the bound address; on any other
  bind `server.allowed_hosts` is required. This is what closes DNS rebinding against an
  unauthenticated loopback service ([ADR-0026 §1](../adr/0026-local-http-hardening.md)).
* A URL supplied **in a request body** (today only `POST /evidence/import`) is fetched only after the
  scheme, host-allowlist, literal-IP, redirect and size checks in
  [ADR-0026 §3](../adr/0026-local-http-hardening.md).
* Every state-changing endpoint validates content type and rejects unexpected media types with 415.
* Import endpoints validate schema version *before* parsing the body's payload, and enforce archive
  safety rules (see Security Standards).

---

## 11. Documentation

* FastAPI generates OpenAPI 3.1 at `/api/v1/openapi.json`, with interactive docs at `/api/v1/docs`
  (loopback only by default; disabled when bound non-loopback unless explicitly enabled).
* Every endpoint has a summary, a description, documented error codes and at least one example.
* The OpenAPI document is committed as a snapshot artifact (`docs/api/openapi-v1.json` in each app
  repo) and a CI test fails when it changes without a corresponding changelog entry — that is how
  accidental breaking changes are caught.
* SetSpec publishes JSON Schema for every payload version; a schema change without a version bump
  fails CI.

---

## 12. Client obligations

Any component calling another application's API must:

1. Check `/api/v1/version` on first contact and cache the result with a TTL.
2. Fail with `API_VERSION_UNSUPPORTED` or `SCHEMA_VERSION_UNSUPPORTED` rather than parsing
   optimistically.
3. Set a connect timeout (5 s default) and a read timeout appropriate to the operation (streams:
   idle timeout based on heartbeat, not total duration).
4. Retry only idempotent requests, with jittered exponential backoff and a bounded attempt count.
5. Propagate `X-Request-ID` when it already has one, so a trace spans both applications.
6. Treat the peer being unavailable as a **degraded** state, never a fatal one.
