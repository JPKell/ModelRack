# ADR-0006 — SQLite default, PostgreSQL supported

**Status:** Accepted (2026-08-21)
**Amended 2026-08-21** (final architecture audit): the portability rules now state how upserts are written.

## Context

The suite is local-first and must start with zero configuration on a single workstation. It must
also survive a larger deployment: LoadCoach on a shared GPU host with several users and a long job
history, or a FreeWeight instance accumulating millions of samples.

The requirements name exactly two targets — SQLite as the default zero-configuration database and
PostgreSQL as the supported larger/multi-user database — and explicitly warn against trying to
support every database equally.

## Decision

**SQLite is the default. PostgreSQL is fully supported. Nothing else is supported.**

| | SQLite | PostgreSQL |
|---|---|---|
| When | Default; single user; one process | Multi-user, shared host, large history, existing infrastructure |
| Configuration | None — a path under the data root | `storage.database_url` |
| Connect-time settings | `foreign_keys=ON`, `journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL` | `statement_timeout`, `lock_timeout` |
| Migrations at startup | Applied automatically (`auto_migrate` default `true`) | **Not** automatic — `<app> db upgrade` is deliberate |
| Backups | SQLite backup API, `<app> db backup` | `pg_dump`, or the site's own tooling |
| Concurrency | One writer, short transactions, WAL readers | Normal MVCC |
| CI | Every integration test | Every integration test (service container) |

Portability rules that make this real: no dialect-specific SQL in application code; JSON and
timezone-aware timestamps go through WeightsDB type decorators; enumerations are `TEXT` with `CHECK`
constraints rather than native enum types; every application's integration suite runs against both.

## Alternatives considered

**SQLite only.** Simplest, and adequate for the primary use case. Rejected: the LAN-deployment shape
in the architecture (LoadCoach on a GPU host serving several clients) is a real requirement, and
SQLite's single-writer model plus network-filesystem hazards make it the wrong tool there.

**PostgreSQL only.** Rejected outright: it destroys zero-configuration local startup, which is a
headline promise of the product.

**DuckDB for analytics alongside SQLite.** Genuinely attractive for FreeWeight's aggregate queries
over millions of samples. Rejected for now as premature: the performance budgets are met by SQLite
with proper indexes at the expected data volumes. Recorded as a future extension for FreeWeight's
analytics layer only, never as the primary store.

**MySQL/MariaDB, SQL Server, Oracle.** Rejected: each adds dialect-neutrality cost and testing
burden with no user in sight.

## Consequences

*Positive.* Install and run with nothing else present. Users who need more scale have a documented
path that does not change application behaviour. Two dialects in CI keeps the abstraction honest —
a dialect-specific mistake fails immediately rather than at a user's site.

*Negative.* Every schema and query must work on both, which forbids some conveniences
(`JSONB` operators, dialect-specific `ON CONFLICT` clauses, array columns, native enums). This is a
deliberate narrowing and is documented in the database standards.

**Upserts are the one place this needed a positive answer rather than a prohibition**, because the
suite has several (model discovery, `feedback` per `(job_id, source)`, `settings`, evidence import,
`reliability_stats`). Select-then-insert is a race, so:

* Upserts go through `weightsdb.upsert(session, model, values, index_elements=…)`, a single helper
  that emits `sqlalchemy.dialects.sqlite.insert(...).on_conflict_do_update(...)` or the PostgreSQL
  equivalent. Both dialects support `ON CONFLICT` with the same semantics for the unique-index form
  the suite uses; what was forbidden is *dialect-specific variants*, not the construct.
* Application code never writes `ON CONFLICT` directly, so there is one implementation to keep
  portable and one place the both-dialects test suite has to cover.

*Negative.* Two migration behaviours (auto on SQLite, explicit on PostgreSQL). Justified: automatic
migration of a shared database during someone's startup is exactly the kind of surprise a multi-user
deployment must not have.

*Negative.* Users may put a SQLite file on NFS/SMB and lose durability. Mitigated: startup detects a
network filesystem where possible and logs a WARNING recommending PostgreSQL.

## Revisit when

* Aggregate query budgets are missed on SQLite at realistic volumes (then: DuckDB for analytics, not
  a new primary store).
* A deployment shape appears that neither dialect serves.
