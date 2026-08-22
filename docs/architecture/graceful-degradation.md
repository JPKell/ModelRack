# Graceful Degradation

**Applies to:** every component.
**Principle:** a missing dependency produces an explicit state — an error, a degraded result, an
`unsupported` measurement, or a queued job. Never a crash, never a fabricated value, never a silent
substitution.

---

## 1. The four outcomes

| Outcome | Meaning | Shape |
|---|---|---|
| **Error** | The request cannot be satisfied now and retrying immediately will not help | Typed exception → HTTP error envelope with a stable `code`; CLI exit code 2–5 |
| **Degraded** | The operation completed with reduced fidelity or coverage | Normal result plus `degradations: [{code, message, impact}]`; UI badge; log at WARNING |
| **Unsupported** | This environment cannot produce this specific measurement or feature | `Unsupported` sentinel → NULL + reason in storage, `—` in UI, `"unsupported"` in JSON |
| **Queued / deferred** | A resource is temporarily unavailable | Job state with a reason and a retry schedule |

Choosing between them is a design decision made per case in the table below — never left to a
`try/except` that returns `None`.

---

## 2. Degradation matrix

`E` = error, `D` = degraded, `U` = unsupported, `Q` = queued. Every row names the user-visible
signal and the code path that must exist.

| Condition | FreeWeight | LoadCoach | IdeaPress | Signal |
|---|---|---|---|---|
| **No GPU present** | D — GPU/VRAM/energy metrics `U`; quality benchmarks run normally; memory-slope benchmark skipped with reason | D — admission control uses RAM only; VRAM constraints not applied | D — telemetry widget hidden | Health: `gpu: unavailable`; run record notes skipped tests |
| **`nvidia-smi` missing or failing** | D — same as above, distinguished as "tool unavailable" not "no GPU" | D | D | Health component `gpu_telemetry: unavailable (nvidia-smi not found)` |
| **GPU sensor unavailable** (temp/power/fan/clock) | U per field; energy metrics become `U` when power is `U` | Ignored by admission control | Blank in widget | `—` in UI; NULL + `reason` in DB |
| **Ollama not running** | E on any run start (`PROVIDER_UNAVAILABLE`); discovery returns the last known models marked stale; UI and CLI still work | E on execute; jobs stay `queued` with `waiting_for_provider` up to their max wait, then `failed` | E on the stage; workflow pauses at the failed stage, project intact | Health: `provider: unavailable`; explicit banner |
| **Provider returns malformed JSON** | E for that sample; run continues; sample stored with `error_text` and the raw body as an artifact | E for that attempt; retry policy applies; then fallback candidate | E for the stage; retry per stage policy | `PROVIDER_PROTOCOL_ERROR` |
| **Provider timeout** | Sample marked `timeout`; never counted as a score of 0 | Attempt fails; retry/fallback; job records each attempt | Stage retry then pause | `PROVIDER_TIMEOUT` |
| **Model not found** | E at run start with the list of known models | Candidate removed from routing with rejection reason `model_absent` | E with the configured model named | `MODEL_NOT_FOUND` |
| **Insufficient VRAM for the requested context** | Context-fit benchmark records the maximum successful context — this *is* the measurement, not a failure | Candidate rejected with `insufficient_vram (needs X, free Y)`; if all candidates fail, job stays `queued` as `waiting_for_resources` | Surfaced from LoadCoach or from the direct backend error | `INSUFFICIENT_RESOURCES` |
| **Insufficient system RAM** | Run refused before start with the estimate | Same as VRAM | Same | `INSUFFICIENT_RESOURCES` |
| **Model lacks a required capability** (tools, structured output) | Test skipped with `unsupported_capability`, never scored 0 | Hard-constraint rejection before scoring | Stage requiring it errors with a clear message and a suggested model | `CAPABILITY_UNSUPPORTED` |
| **Container runtime absent** (code-execution benchmarks) | Benchmark **skipped**, reason `sandbox_unavailable`; never executed on the host | n/a | n/a | ADR-0018 tier check |
| **No benchmark evidence at all** | n/a | D — routes on declared capabilities + config; every decision states `evidence: none`; confidence factor at its floor | n/a | UI banner "routing without measured evidence" |
| **Stale benchmark evidence** | Marks results stale in the UI when environment drift is detected | D — confidence decayed per ADR-0017; explanation shows age and decay | n/a | Badge with age and reason |
| **Incompatible SetSpec major version** | Import/export refused with both versions named | Import refused; existing evidence untouched | Backend adapter refuses to start in LoadCoach mode | `SCHEMA_VERSION_UNSUPPORTED` |
| **Incompatible API major version** | n/a | Client rejects with both versions named | Same | `API_VERSION_UNSUPPORTED` |
| **Optional application unreachable** | n/a | Evidence import skipped; last import retained and marked stale | Fall back to direct backend (or error if pinned) | Health component `degraded` |
| **Evidence measured under a different runtime profile** | n/a | D — that evidence does not apply; the capability is **absent** for the candidate (not zeroed), the explanation names both hashes and the FreeWeight invocation that would fix it, and the decision counts toward `low_evidence` | n/a | `evidence_profile_mismatch` in the explanation |
| **Evidence for a model not yet discovered** | n/a | Retained with `match_state = "unmatched"`, reported in the import result, contributes nothing, binds automatically on the next discovery pass | n/a | Import counts; evidence page |
| **Served context cannot be established** | Recorded on the run with its source | D — decision flagged `assumed_context`; a profile needing a context the provider will not be asked to serve is rejected `context_not_configurable` | Surfaced from LoadCoach as a degradation on the attempt | `assumed_context` |
| **More than one GPU visible, placement unreported** | Memory, KV and energy tests **skipped** with `multi_gpu_placement_unknown`; quality and throughput unaffected | D — admission evaluates each device independently; a model fitting no single device is deferred, never admitted on a summed total | n/a | Skip reason on the run; per-device numbers in the rejection |
| **`Host` header not in the allowlist** | 421 before routing and before auth | Same | Same | `MISDIRECTED_REQUEST`, WARNING log with the presented value |
| **Evidence import URL fails the fetch allowlist** | n/a | E — `EVIDENCE_SOURCE_REFUSED` before any bytes are parsed; existing evidence untouched | n/a | Names the rule that refused it |
| **Database migration pending** | Startup refuses with the exact `alembic upgrade` command; read-only inspection commands still work | Same | Same | `MIGRATION_REQUIRED`, exit 4 |
| **Database migration fails** | Automatic restore from the pre-migration backup; original DB never left half-migrated | Same | Same | `MIGRATION_FAILED` + restore log |
| **Database locked (SQLite)** | Retry with backoff to `busy_timeout`, then E | Same | Same | `STORAGE_BUSY` |
| **Disk full** | Run aborted at the next checkpoint; completed samples preserved | Job fails; queue preserved | Draft written to a temp file and reported | `STORAGE_FULL` |
| **SSE client disconnects** | Nothing; events are rows. Client replays from `Last-Event-ID` | Same | Same | Debug log only |
| **Server restarts mid-run/job** | Run marked `interrupted` on startup recovery; completed tests retained; resumable | Leases expire; jobs return to `queued` (idempotent) or `failed` (non-idempotent, per policy) | Workflow resumes at the last committed unit | Startup recovery log + UI notice |
| **Prompt pack missing/invalid** | Startup validation fails with the file and field named | Same | Same | `PROMPT_INVALID`, exit 3 |
| **Remote provider configured but unreachable** | E, and the error names the remote host so egress is obvious | E | E | `PROVIDER_UNAVAILABLE` |

---

## 3. Health reporting

Every application exposes `GET /api/v1/health` and `<app> health [--json]`, using one shape:

```json
{
  "status": "ok",
  "version": "1.2.0",
  "checked_at": "2026-08-21T10:04:11.482Z",
  "components": [
    {"name": "database",  "status": "ok",          "detail": "sqlite 3.46, 12 migrations applied"},
    {"name": "provider",  "status": "ok",          "detail": "ollama 0.32.13, 11 models"},
    {"name": "gpu",       "status": "degraded",    "detail": "power sensor unavailable"},
    {"name": "evidence",  "status": "degraded",    "detail": "last import 14 days ago"},
    {"name": "loadcoach", "status": "unavailable", "detail": "connection refused at 127.0.0.1:8766"}
  ]
}
```

* `status` for the whole application is the worst component status that is **required** for its core
  function. Optional components never make the application `unavailable`.
* Component statuses: `ok` | `degraded` | `unavailable` | `not_configured`.
* `not_configured` is not a problem: LoadCoach with no FreeWeight configured reports
  `evidence: not_configured`, and overall `ok`.
* HTTP status is 200 for `ok`/`degraded`, 503 for `unavailable`. Machines read the body, not the code.

---

## 4. Startup validation

At startup each application, in order:

1. Loads and validates configuration (precedence resolved, types checked, unknown keys reported).
2. Refuses to start on unsafe combinations — non-loopback bind without authentication, remote
   provider without an explicit `allow_remote` acknowledgement, world-writable data directory.
3. Verifies the database exists and is at head; refuses with an actionable message otherwise, and
   refuses when the database is **ahead** of the code (`SchemaAhead`) naming both revisions and the
   backup directory — the state a downgrade without a restore leaves behind.
4. Validates the prompt pack (parse, required variables, schema compatibility).
5. Probes optional dependencies **without blocking**: provider, GPU telemetry, optional peer
   applications. Failures here produce degraded health, never a refusal to start.
6. Logs a one-line startup summary naming every degraded component.

The distinction is deliberate: **configuration errors block startup; environment gaps do not.**

---

## 5. Testing degradation

Failure paths are tested to the same standard as success paths. Each application's suite includes,
at minimum:

* Provider absent, provider timeout, provider 500, provider malformed body, provider truncated stream.
* GPU absent, `nvidia-smi` absent, `nvidia-smi` malformed, multi-GPU, missing temperature, missing power.
* Database at an older revision, database locked, migration failure with restore.
* Schema major mismatch on import and on export consumption.
* Peer application unreachable, peer returning an incompatible API version.
* Disconnect and reconnect mid-stream with `Last-Event-ID` replay.
* Process kill mid-run/mid-job followed by restart recovery.
* Sandbox tier unavailable → benchmark skipped, not executed.
* Evidence whose runtime profile does not match the execution; evidence for an undiscovered model.
* A disallowed `Host` header; a forged form post; an import URL outside the allowlist.
* More than one GPU visible with placement unreported.

A degradation path with no test is treated as an unimplemented feature.
