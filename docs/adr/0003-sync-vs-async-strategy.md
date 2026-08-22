# ADR-0003 — Sync core, async edge

**Status:** Accepted (2026-08-21)
**Amended 2026-08-21** (final architecture audit): §6 of the decision now states how the SSE bridge avoids blocking the event loop.

## Context

Choosing an ASGI framework ([ADR-0002](0002-web-framework.md)) makes async available everywhere. The
requirements warn against choosing async merely because it is newer, and ask for a concrete
assessment of whether asynchronous I/O materially improves model streaming, concurrent inference,
long-lived connections, API responsiveness, queue handling and telemetry streaming.

What the workloads actually look like:

| Workload | Nature | Concurrency |
|---|---|---|
| Benchmark execution | Long blocking provider calls, CPU-bound scoring, subprocesses | Deliberately **1** GPU workload at a time |
| LoadCoach execution | Long blocking provider calls | Bounded by GPU; default 1, typically ≤ 4 |
| IdeaPress stages | Long blocking provider calls | 1 per project, a handful of projects |
| SSE connections | Idle almost always, held for minutes to hours | **Hundreds** — dashboards, run views, telemetry |
| JSON API reads | Short database queries | Tens |
| Telemetry sampling | Subprocess + file reads, every second | 1 |

Only one row benefits materially from async: idle long-lived connections. Inference concurrency is
capped by a single 16 GB GPU, not by the I/O model.

## Decision

**Async at the HTTP edge; synchronous everywhere below it.**

1. Route handlers that touch a database, a provider or the filesystem are `def`. Starlette runs them
   in a bounded worker threadpool, so they never block the event loop.
2. Route handlers that only stream or fan out events are `async def`.
3. `services/`, `domain/` and `infrastructure/` are **entirely synchronous**. There is no async
   variant of any service method, and no `asyncio` import below `web/`.
4. ModelRack exposes a **synchronous** API only. Streaming is a synchronous generator.
5. Background work (run scheduler, job queue workers, telemetry sampler) uses `threading.Thread`
   with explicit lifecycle management and its own sessions.
6. The bridge from worker threads to SSE clients goes through the persisted event store plus a
   thread-safe notification; SSE handlers await new events without any worker needing to know about
   the event loop.
7. **The event store is synchronous and database-backed, and an SSE handler is `async def`** — so
   every call an SSE handler makes into it (`replay`, and each pull from a subscription) is dispatched
   with `anyio.to_thread.run_sync` (Starlette's `run_in_threadpool`). An SSE handler never touches a
   session directly. Without this, rule 1's exemption for "handlers that only stream" would put a
   blocking `SELECT` on the event loop once per reconnect and once per batch — the precise defect
   this ADR exists to forbid. MirrorWall's `sse_response` owns the dispatch, so no application can get
   it wrong.
8. The threadpool is therefore shared between sync route handlers and SSE replay. Replay is bounded
   (one batched read per client per wake-up, never a per-event round trip), the in-memory fan-out
   carries the steady-state stream without touching the database at all, and
   `/api/v1/system/status` exposes threadpool saturation so the interaction is observable rather than
   assumed.

Blocking inside an `async def` is a defect. A review checklist item and a runtime warning
(event-loop lag monitor logging at WARNING above a threshold) catch it.

## Alternatives considered

**Async all the way down.** Async provider clients, async SQLAlchemy, async services. Rejected: it
duplicates every service for the CLI (which is synchronous and must share the exact same code path),
makes CPU-bound scoring awkward, complicates subprocess handling, and buys concurrency that a single
GPU cannot use. The maintenance cost is permanent; the benefit is zero for the dominant workloads.

**Dual sync + async APIs in ModelRack.** Two implementations, two test suites, two sets of bugs, for
a provider that is called at most a handful of times concurrently. Rejected as unjustified
complexity.

**Fully synchronous (WSGI).** Rejected in ADR-0002 — one thread per open SSE connection is the wrong
trade for a product whose UIs stream continuously.

**Process pool for blocking work.** Rejected: serialization overhead, complicated cancellation, no
shared in-process state, and no CPU-bound hot spot that justifies it. Scoring is not the bottleneck;
inference is.

## Consequences

*Positive.* One code path shared by web and CLI. Simple, debuggable stack traces. Synchronous
SQLAlchemy (mature, better-tooled). Trivially testable services. Hundreds of cheap idle SSE
connections. Cancellation via simple flags and connection closure.

*Negative.* The threadpool is a finite resource: a burst of slow database queries — or a burst of SSE
reconnections all replaying at once — can saturate it and delay other sync handlers. Mitigations: the threadpool size is configurable (default 40), slow
queries are logged, long work goes to background workers rather than request handlers, and
`/api/v1/system/status` exposes threadpool saturation.

*Negative.* Thread-safety becomes a real concern for shared in-process state (event fan-out,
telemetry cache, model residency). Mitigated by keeping shared mutable state in exactly three named,
lock-protected, individually tested objects per application.

## Revisit when

* An application needs more than ~50 genuinely concurrent inference executions.
* Remote providers with high per-request latency become a primary use case (fan-out to many slow
  endpoints is where async wins decisively).
* Threadpool saturation appears in real usage rather than in speculation.

The migration path if that day comes: make ModelRack's provider interface async **in addition to**
sync (the adapters already isolate the HTTP client), and make only LoadCoach's executor async. The
domain layer would not change.
