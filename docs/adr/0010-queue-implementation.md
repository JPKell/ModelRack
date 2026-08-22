# ADR-0010 — Database-backed queue, no broker

**Status:** Accepted (2026-08-21)

## Context

LoadCoach must queue inference jobs with priorities, cancellation, retries, fallback, timeouts,
maximum wait times, starvation prevention, restart recovery and model-residency awareness. FreeWeight
needs a simpler version of the same for benchmark runs. IdeaPress needs one exclusive execution lane
per project.

The requirements are unambiguous: start as simple as reasonably possible; do not introduce Redis,
Celery, RabbitMQ, Kafka or Kubernetes without a demonstrated concrete requirement; SQLite-backed
scheduling may be sufficient.

The actual workload: one machine, one 16 GB GPU, **one to four concurrent executions**, tens to
hundreds of jobs per day, each taking seconds to minutes. The queue's hard problem is not throughput
— it is *policy* (which job, on which model, with what resources, without starving anyone).

## Decision

**A queue table in the application's own database, with lease-based claiming, worked by threads in
the same process.**

* `jobs` table: state, priority, class (interactive/normal/background/batch), payload, constraints,
  attempt count, timestamps, `lease_owner`, `lease_expires_at`, cancellation flag, result reference.
* Claiming is a single atomic `UPDATE … WHERE state='queued' AND … RETURNING` (or `UPDATE` + reselect
  on SQLite under `BEGIN IMMEDIATE`), so two workers can never claim the same job.
* Leases expire; a crashed worker's job returns to `queued` (idempotent work) or `failed`
  (non-idempotent), per the job's declared policy.
* Workers poll with adaptive backoff (50 ms when busy → 1 s when idle) plus an in-process wake-up
  signal on enqueue, so dispatch latency stays inside its budget without busy-waiting.
* Cancellation is a flag the executor checks at every stream boundary; queued jobs are cancelled
  transactionally.
* Priority ageing prevents starvation: effective priority rises with wait time, with the policy and
  its parameters in [LoadCoach Queue and Scheduling](../apps/loadcoach/queue-and-scheduling.md).
* Recovery on startup is idempotent: expired leases released, orphaned `running` jobs reconciled,
  and the reconciliation logged.

No Redis. No Celery. No RabbitMQ. No Kafka. No separate worker process.

## Alternatives considered

**Celery + Redis/RabbitMQ.** The reflexive Python answer. Rejected: it adds two services to install,
configure, secure, monitor and back up, for a workload of a few concurrent jobs on one machine. It
also makes the local-first zero-configuration promise impossible, and the hard parts of this queue
(resource-aware admission, model residency, routing) are exactly the parts Celery does not solve.

**Redis alone as the queue.** Rejected for the same installation and durability reasons, plus a
second source of truth: job state would live in Redis while job history lives in the database, and
reconciling the two after a crash is worse than never splitting them.

**`multiprocessing` or a process pool.** Rejected: no shared in-process state (residency cache,
event fan-out, telemetry), harder cancellation, and no benefit for I/O-bound work.

**An in-memory queue.** Rejected: a restart would lose queued work, and requirement §17 mandates
persistent jobs and queue recovery.

**Filesystem queue directory.** Rejected: worse durability and atomicity than the database that is
already present.

**`LISTEN/NOTIFY` (PostgreSQL) for wake-ups.** Attractive on PostgreSQL, unavailable on SQLite.
Rejected as a *requirement*; recorded as an optional optimization behind the same interface.

## Consequences

*Positive.* Zero additional infrastructure. Job state and job history are the same rows, so there is
nothing to reconcile. Cancellation, priority and recovery are ordinary SQL. Everything is testable
with a temporary database and a fake clock. Backups cover the queue automatically.

*Negative.* Polling wastes a small amount of CPU (budgeted at ≤ 0.5 % of one core at idle) and adds
up to ~100 ms of dispatch latency in the worst case — acceptable against multi-second inference.
Throughput is bounded by database write rate, orders of magnitude above what this workload needs.
Single-process design means the queue stops when the application stops — correct, since the queue's
purpose is to schedule *this machine's* GPU.

*Negative.* We implement lease semantics, ageing and recovery ourselves. Bounded, well-understood,
and covered by a scheduling simulation test suite.

## Revisit when

* Jobs must be executed by workers on **other machines** (multi-machine execution is an explicit
  future extension).
* Sustained throughput exceeds what a single database writer can dispatch (thousands of jobs per
  minute) — far beyond a local GPU's capacity.
* Queue durability requirements exceed what the application database provides.

If that day comes, the `JobQueue` port is the seam: a broker-backed implementation would replace it
without touching routing, execution or the API.
