# ADR-0029 — Queue mechanics: ageing, attempts, admission states and leases

**Status:** Accepted (2026-08-21)
**Extends:** [ADR-0010](0010-queue-implementation.md), [Queue and Scheduling](../apps/loadcoach/queue-and-scheduling.md).

## Context

ADR-0010 chose a database-backed queue and accepted that "we implement lease semantics, ageing and
recovery ourselves — bounded, well-understood, and covered by a scheduling simulation test suite".
The audit found four places where the specification does not actually describe a working mechanism.
Each would be found by the simulator, but only after being built wrong.

1. **Ageing never happens.** `effective_priority` is a stored `INT` column, the claim query orders by
   it, and the only place the specification recomputes it is startup recovery. A long-running process
   therefore never ages anything, and the starvation bound the gold standards promise — "a
   low-priority job's wait is bounded by the configured ageing policy" — does not hold.
2. **The attempt counter has two writers with incompatible semantics.** The claim statement does
   `attempt = attempt + 1`. Retries *within* a lease (a corrective retry after a validation failure)
   also create `job_attempts` rows. `job_attempts` is `UNIQUE (job_id, attempt)`. A job that is
   claimed (job.attempt → 1), retries in-lease (attempt row 2), then loses its lease and is
   re-claimed (job.attempt → 2) collides with the attempt row it already wrote.
3. **A claimed job that fails admission has nowhere to go.** Admission runs "before a claimed job
   executes", i.e. in state `leased`. The state machine has `queued → waiting_resources` but no
   `leased → waiting_resources`, and no `leased → cancelling`, so a cancel arriving between claim and
   provider call has no legal transition either.
4. **The lease heartbeat has no thread.** Leases are 60 s, "renewed by a heartbeat while executing",
   while an attempt may run 300 s. The worker thread is inside a blocking provider call for that
   entire time and cannot renew its own lease, so every long generation loses its lease mid-flight
   and is reclaimed by another worker — the double-execution failure the atomic claim was designed to
   prevent.

## Decision

### 1. Effective priority is computed, not stored stale

`jobs.effective_priority` remains a stored column, because the claim query's index depends on it, and
it is maintained by the **ageing sweep**: a task on the scheduler thread that runs every
`queue.ageing_interval_seconds` (default 30) and issues one set-based statement:

```sql
UPDATE jobs
   SET effective_priority = MIN(
           base_priority + CAST((julianday(:now) - julianday(queued_at)) * 1440
                                * :ageing_priority_per_minute AS INTEGER),
           :class_band_top + :overflow_allowance),
       updated_at = :now
 WHERE state IN ('queued', 'waiting_resources')
   AND effective_priority <> <the computed value>;
```

(Written portably through SQLAlchemy expressions; the dialect-specific date arithmetic lives in one
repository method.)

* One statement for the whole queue, bounded by queue depth, at a cadence far coarser than dispatch
  latency — the ageing granularity is 30 s, and the policy grants one priority point per minute.
* `queued_at` is the ageing origin, so time spent in `waiting_resources` counts as waiting, which is
  what a starved job experiences.
* The sweep is idempotent and its cost is asserted against the idle-CPU budget.
* Startup recovery runs the same sweep, so recovery and steady state share one code path.

### 2. `jobs.attempt` counts attempts; the claim does not increment it

* The claim statement sets state, lease owner and expiry. It **does not** touch `attempt`.
* `attempt` is incremented in exactly one place: when an attempt row is created, by the executor,
  inside the transaction that writes it. Every attempt — first, in-lease retry, fallback, or
  post-requeue — takes the next number.
* `job_attempts.rank` continues to record candidate rank (1 = primary, 2+ = fallback), which is a
  different axis and was already correct.
* A re-claim after a lost lease therefore continues the attempt sequence rather than restarting it,
  and `UNIQUE (job_id, attempt)` holds. `max_attempts` counts total attempts across leases, which is
  the semantics a caller expects from "retry at most three times".

### 3. Two admission states, and a complete transition set

Admission moves to its own state so a claimed job always has somewhere to go:

```text
leased --> admitted        : resources available on some device (ADR-0027)
leased --> waiting_resources : admission deferred; lease released, job re-queued for later claim
leased --> cancelling      : cancel requested between claim and execution
admitted --> executing     : provider call started
```

* `leased → waiting_resources` **releases the lease**. A job waiting on VRAM must not hold a worker
  or a lease; it is re-evaluated by the admission pass when a model unloads or telemetry shows
  headroom, and re-enters `queued`.
* `waiting_resources → cancelled` and `queued → cancelled` already existed and are unchanged.
* The full transition table is normative and lives in
  [Queue and Scheduling §2](../apps/loadcoach/queue-and-scheduling.md); every legal transition and
  every rejected illegal one is a test, as the testing standards require of any state machine.

### 4. Leases are renewed by the scheduler, not by the working thread

* A single **lease keeper** on the scheduler thread renews the leases of all jobs this process is
  executing, every `lease_seconds / 3` (default 20 s). It never blocks on a provider.
* A worker thread inside a blocking provider call therefore keeps its lease without doing anything.
* If the process dies, no keeper runs, every lease expires, and recovery proceeds exactly as ADR-0010
  describes. If the keeper itself stalls — the event a lease exists to detect — leases expire and the
  job is reclaimed, which is the correct outcome.
* `lease_seconds` (60) must exceed `3 × renewal_interval` plus scheduling slack; this relationship is
  validated at startup rather than left to a comment.

## Alternatives considered

**Compute effective priority in the claim query instead of storing it.** Removes the sweep entirely
and can never be stale. Rejected: an expression over `queued_at` in `ORDER BY` cannot use the
`(state, effective_priority DESC, created_at)` index, so the claim query — the hottest statement in
the application, with a query-plan assertion attached — degrades to a scan of the active set. Storing
the value keeps the plan and the sweep keeps it fresh.

**Age only on enqueue and on dispatch.** Cheaper still. Rejected: a queue that is busy with
high-priority work dispatches constantly but never re-evaluates the jobs it is starving, which is the
exact scenario the bound must cover.

**Keep the claim-time increment and give in-lease retries a sub-counter.** Rejected: two counters
where one suffices, and every consumer of `job_attempts` would have to know which is which.

**Let a job hold its lease while in `waiting_resources`.** Rejected: it burns a worker slot on a job
that is by definition not running, and it makes the starvation counter and the concurrency limit
disagree about what is in flight.

**Heartbeat from the worker thread between stream chunks.** Works for streamed generation, and fails
for every non-streaming call and every long prefill — the cases where the lease matters most.
Rejected for being correct only where it is not needed.

## Consequences

*Positive.* The starvation bound the gold standards assert becomes a property of a mechanism rather
than of prose. Long generations stop losing their leases, which removes the most likely source of the
double-execution defect LoadCoach's risk register ranks first. A claimed job always has a legal next
state. Attempt numbering is single-writer and monotonic.

*Negative.* One periodic write against the active queue every 30 s, and a lease keeper thread. Both
are budgeted, and both are exercised by the scheduling simulator over a fake clock at no real-time
cost.

*Negative.* `admitted` is a fifth non-terminal state to reason about. It earns its place by making the
admission deferral path expressible.

## Revisit when

ADR-0010's own trigger fires — jobs executed by workers on other machines — at which point lease
renewal crosses a process boundary and the keeper becomes a protocol rather than a thread.
