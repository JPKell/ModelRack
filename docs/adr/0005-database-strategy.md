# ADR-0005 — SQLAlchemy 2.0 + Alembic

**Status:** Accepted (2026-08-21)

## Context

Every application persists structured data with real relationships: runs → tests → samples →
metrics; jobs → attempts → validations; projects → units → drafts. The requirements mandate
SQLAlchemy "unless there is a documented architectural reason not to", real migrations, indexes,
foreign keys, transaction handling, migration testing, backups and a clean upgrade path — across
both SQLite and PostgreSQL.

The prior implementations used raw `sqlite3` with hand-written migration tuples, and in one case
`CREATE TABLE IF NOT EXISTS` at connect time. The first is workable but has no PostgreSQL story and
no autogenerate safety net; the second is not a migration system at all and drifts silently.

## Decision

**SQLAlchemy 2.0 (ORM with typed `DeclarativeBase`) + Alembic**, provided to applications through
[WeightsDB](../packages/weightsdb/spec.md).

* WeightsDB owns: engine and session factories, dialect pragmas, a timezone-aware datetime type, a
  JSON type that renders `JSONB` on PostgreSQL, transaction helpers, the migration runner wrapper,
  backup/restore and health checks.
* Each application owns: its own `MetaData` and models, its own Alembic history, its own repositories.
* ORM objects never leave the repository layer; services receive domain objects or typed row DTOs.
* Bulk paths (sample and telemetry inserts) use Core `insert()` with executemany, not ORM instances.
* Alembic autogenerate runs in CI and must produce an **empty** diff against head — a model changed
  without a migration fails the build.

## Alternatives considered

**Raw `sqlite3` + hand-written migrations.** What the prior code did. Rejected: no PostgreSQL
support, hand-written SQL for every query, no autogenerate parity check, and every application would
re-implement the same plumbing — which is precisely the duplication WeightsDB exists to prevent.

**SQLAlchemy Core only (no ORM).** Tempting: explicit SQL-shaped code, no lazy-loading surprises, no
session identity map to reason about. Rejected because relationship handling and cascade semantics
across the deep run→test→sample→metric tree would be hand-maintained, and because the typed
declarative models double as schema documentation. Core remains the tool for bulk and analytical
queries — this is a "both, deliberately" decision, not an ORM-only one.

**SQLModel.** Combines pydantic and SQLAlchemy. Rejected: it blurs the wire/storage boundary the
architecture deliberately keeps separate (SetSpec models are contracts; ORM models are storage), and
it adds a layer whose release cadence has lagged SQLAlchemy's.

**Peewee / Tortoise / Piccolo.** Rejected: smaller ecosystems, weaker migration tooling, and no
capability the suite needs that SQLAlchemy lacks.

**A document store or plain JSON files.** Rejected: the data is relational, the queries are
analytical (aggregate metrics across runs, models and machines), and requirement §15 mandates FKs
and migrations.

## Consequences

*Positive.* One dialect-neutral data layer; migrations that are reviewable, testable and reversible;
autogenerate as a safety net; typed models that document the schema; a large ecosystem for
diagnosing performance; the same repositories work on SQLite and PostgreSQL.

*Negative.* Two significant dependencies. A learning curve around session lifetimes and the identity
map — mitigated by the strict rule that sessions are opened only by the service layer, that
repositories take a session, and that ORM objects never escape the repository. Alembic requires
discipline (batch mode for SQLite ALTERs, lock-awareness on PostgreSQL) — covered in
[Database Standards §5](../standards/database-standards.md).

*Negative.* Slight overhead versus raw SQL. Irrelevant against provider latency, and bulk paths use
Core anyway; the budgets in [Performance Targets §3.5](../architecture/performance-targets.md) are
set with the ORM in the loop.

## Revisit when

A profiling result shows ORM overhead is a measurable share of a user-visible operation (rather than
a theoretical concern), or a required feature is unavailable in SQLAlchemy. Neither is expected.
