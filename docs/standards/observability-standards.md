# Observability Standards

**Applies to:** all three applications; shared helpers live in MirrorWall (request IDs, envelopes)
and in each application's `observability/` module.
**Principle:** an operator must be able to answer "what happened, to what, when, and why" from the
logs and the database alone — without reproducing the problem.

---

## 1. Structured logging

* Python's `logging`, configured once at startup by the composition root.
* Two formatters: **text** for a TTY (human-friendly, aligned, coloured) and **JSON Lines** for
  everything else. Selected automatically; overridable with `logging.format`.
* One logger per module (`logging.getLogger(__name__)`). No `print` outside CLI rendering
  (`ruff T20`).
* Message strings are stable, low-cardinality **event names**; variable data goes in structured
  fields:

```python
logger.info("run.completed", extra={"run_id": run.id, "duration_ms": 41822, "samples": 640})
```

```json
{"ts":"2026-08-21T09:14:02.318Z","level":"INFO","event":"run.completed","logger":"freeweight.services.runs",
 "run_id":"01J9K2M4P7Q8R9S0T1U2V3W4X5","duration_ms":41822,"samples":640,
 "request_id":"01J9K2M4P7Q8R9S0T1U2V3W4X6","app":"freeweight","version":"1.0.0","pid":12345}
```

* Never build messages by interpolating variable data: `logger.info(f"run {id} done")` is banned —
  it destroys aggregation and invites injection into log files.

### 1.1 Levels

| Level | Use | Examples |
|---|---|---|
| DEBUG | Developer detail, off in production | Provider request bodies (hashed), query timings, cache hits |
| INFO | State transitions a user would want to see | Run started/completed, job dispatched, model loaded, migration applied, server started |
| WARNING | Degradation or an unusual but handled condition | Sensor unavailable, evidence stale, retry attempted, slow query, client dropped |
| ERROR | An operation failed | Provider error after retries, validation failure that ended a job, migration failure |
| CRITICAL | The process cannot continue | Database unopenable, configuration invalid at startup |

Default level `INFO`. A healthy idle process logs nothing at INFO — log volume tracks activity, not
time.

---

## 2. Correlation identifiers

Every log record carries every identifier in scope, propagated through a `contextvars`-based context
so no function needs to pass them explicitly.

| ID | Format | Scope |
|---|---|---|
| `request_id` | ULID | One HTTP request or one CLI invocation |
| `run_id` | ULID | One FreeWeight benchmark run |
| `run_test_id` / `sample_id` | ULID | One test / one sample within a run |
| `job_id` | ULID | One LoadCoach job |
| `attempt` | int | One execution attempt within a job |
| `project_id` / `unit_id` / `stage` | ULID / string | IdeaPress work |
| `model_canonical_id` | string | Which model was involved |
| `trace_id` | ULID | Optional; spans applications when a caller propagates `X-Request-ID` |

Rules: a request ID is generated if the client did not supply a valid one; it is returned in
`X-Request-ID` and in every error envelope; a background task started by a request inherits it; a
cross-application call forwards it.

---

## 3. What is logged, and when

### 3.1 Always (INFO)

* Process start: version, git commit, Python version, config file used, effective bind address,
  database dialect and revision, provider kind and version, degraded components.
* Process stop: reason, in-flight work drained or interrupted.
* Every state transition of a run, job, stage or model residency change, with the previous and new
  state and the reason.
* Every migration applied, backup taken, destructive operation performed (with the preview counts).
* Every routing decision at a summary level (task, chosen model, score, candidate count, evidence
  freshness) — the full explanation goes to the database, not the log.
* Every authentication failure (source address, token name if identifiable, never the token).
* Every provider error, timeout and retry, with attempt number and the delay before the next one.

### 3.2 Never

* Secrets, tokens, API keys, authorization headers, cookies.
* Full prompts or full generated content at INFO or above. With `logging.include_content = false`
  (the default) only hashes and lengths are logged, at any level.
* Absolute paths outside the data root; user home directories are abbreviated.
* Personal data beyond what the user themselves put into a project.
* Full stack traces in HTTP responses (they belong in the log, correlated by request ID).

A redaction filter removes any field whose key matches
`(?i)(token|key|secret|password|authorization|cookie)` regardless of origin, and is tested.

---

## 4. Events vs. logs

Two different mechanisms with two different audiences — do not conflate them.

| | Log records | Domain events |
|---|---|---|
| Audience | Operator, developer | The application itself, its UI, its API clients |
| Storage | Rotating files / stdout | Database table (`run_events`, `job_events`, …) |
| Lifetime | Rotation policy (default 7 files × 10 MiB) | Lifetime of the parent entity |
| Ordering | Best effort | Gap-free per-stream sequence starting at 1 |
| Delivery | None | SSE with `Last-Event-ID` replay |
| Schema | Log fields | SetSpec `event.envelope`, versioned |

Every domain event that matters to a user is also logged at INFO with the same event name, so a log
search and a UI timeline tell the same story.

### 4.1 Event envelope

```json
{
  "schema": "event.envelope",
  "schema_version": "1.0",
  "generated_at": "2026-08-21T09:14:02.318Z",
  "generator": {"name": "freeweight", "version": "1.0.0"},
  "payload": {
    "event_id": "01J9K2M…",
    "sequence": 42,
    "type": "sample.completed",
    "entity": {"kind": "run", "id": "01J9K2M…"},
    "timestamp": "2026-08-21T09:14:02.318Z",
    "message": "Sample 12/40 completed",
    "progress": {"completed": 12, "total": 40},
    "data": {"sample_id": "…", "score": 0.75, "duration_ms": 1840}
  }
}
```

The envelope fields (`schema`, `schema_version`, `generated_at`, `generator`) are **siblings of**
`payload`, never mixed into it — otherwise a reader cannot tell which fields `schema_version`
governs, and the single `load_envelope` code path in SetSpec does not apply. The producing
application identifies itself in `generator`, which is why the event payload carries no separate
`source` field. This is the shape [API Standards §8](api-and-contract-standards.md) puts on the wire,
and [ADR-0025 §3](../adr/0025-envelope-boundaries.md) is normative. The one exception — bare
`event: token` frames — is documented there.

Event type names are `noun.verb` in past tense and are part of the public contract for the
application's major API version.

---

## 5. Metrics

No metrics server, no Prometheus dependency, no external collector — the suite is a local tool.
Instead, metrics that matter are **rows**, and the API exposes them:

* Per run/job/stage: duration, attempts, retries, tokens, validation outcomes, error codes.
* Per model: request count, success rate, mean and p95 latency, mean tokens/second — computed from
  job rows on demand.
* Per queue: depth by state and priority, oldest queued age, dispatch latency, starvation counter.
* Process: uptime, in-flight requests, active SSE connections, threadpool saturation, memory RSS —
  exposed at `GET /api/v1/system/status`.

`GET /api/v1/system/status` is the machine-readable operational snapshot; the UI's System page
renders it. A future Prometheus endpoint is an extension point, deliberately not built now
([ADR-0010](../adr/0010-queue-implementation.md) records the same "no infrastructure without a
demonstrated need" reasoning).

---

## 6. Health endpoints

Defined once in [Graceful Degradation §3](../architecture/graceful-degradation.md): one shape,
component-level statuses, `not_configured` distinct from `unavailable`, HTTP 200 for `ok`/`degraded`
and 503 for `unavailable`. Every application implements it identically, and `<app> health` renders
the same payload.

`GET /api/v1/system/status` extends health with live operational numbers (queue depth, active
connections, current telemetry snapshot, model residency).

---

## 7. Error reporting

* Every error surfaced to a user carries a stable `code` and the `request_id`.
* Every ERROR log record includes the exception type, the stable code, the correlation IDs, and the
  traceback in a dedicated field (`exc_info` rendered into the JSON record).
* Errors that are expected in normal operation (provider unavailable, validation failure, cancelled
  job) are logged at WARNING or INFO with a reason — reserving ERROR for genuine faults keeps ERROR
  meaningful.
* No telemetry, crash reporting or error aggregation leaves the machine, ever.

---

## 8. Log storage

* Default destination: stdout (so `journald`, a supervisor or a terminal captures it) **and** a
  rotating file under `$XDG_STATE_HOME/<app>/logs/<app>.log`.
* Rotation: 10 MiB × 7 files by default, configurable; files are mode `0600`.
* `<app> logs path` prints the location; `<app> logs tail [--follow] [--level]` is provided so a user
  never has to know the format or the path.
* Log files are never written outside the state directory.

---

## 9. Performance observability

* Every HTTP response carries `X-Response-Time-Ms`.
* Every generation records **provider time and application overhead separately**, and both are
  returned in the response metadata (see [Performance Targets §2](../architecture/performance-targets.md)).
* Queries slower than a configurable threshold (default 200 ms) log at WARNING with the statement
  name and duration — never the parameter values.
* A run/job records its own overhead so a user can see what the suite cost them versus what the model
  cost them.

---

## 10. Testing

| Test | Asserts |
|---|---|
| Structured output | JSON formatter emits parseable lines with the required fields |
| Event frame shape | Every non-`token` SSE frame parses through `setspec.load_envelope`; a `token` frame does not and is the only frame that does not |
| Correlation | `request_id` appears on every record produced during a request, including from a background task it started |
| Redaction | A request carrying a token produces no log line containing it |
| Content policy | With `include_content=false`, no prompt or response text appears at any level |
| Levels | A degraded sensor logs WARNING, not ERROR; a cancelled job logs INFO |
| Event sequence | Gap-free, starts at 1, survives restart, replays without duplicate |
| Health shape | Matches the documented schema for ok / degraded / unavailable / not_configured |
| Rotation | Files rotate at the configured size and keep the configured count |
