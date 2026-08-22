# Architecture Decision Records

Every significant architectural decision in the suite is recorded here. An ADR is written **before**
the decision is implemented, and it is not edited to hide a change of mind — it is superseded by a
new ADR that references it.

## Format

Every ADR contains, in this order:

```text
Status          Proposed | Accepted | Superseded by ADR-XXXX | Deprecated
Context         The forces, constraints and evidence
Decision        What we will do, stated unambiguously
Alternatives considered   What else was evaluated, and why it lost
Consequences    What this costs, what it enables, what it forecloses
Revisit when    The concrete trigger that would reopen the decision
```

A decision without a "revisit when" trigger is a decision nobody can safely revisit.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-application-and-package-separation.md) | Application and package separation | Accepted |
| [0002](0002-web-framework.md) | Web framework: FastAPI | Accepted |
| [0003](0003-sync-vs-async-strategy.md) | Sync core, async edge | Accepted |
| [0004](0004-sse-vs-websockets.md) | Server-Sent Events for streaming | Accepted |
| [0005](0005-database-strategy.md) | SQLAlchemy 2.0 + Alembic | Accepted |
| [0006](0006-sqlite-and-postgresql-roles.md) | SQLite default, PostgreSQL supported | Accepted |
| [0007](0007-provider-abstraction.md) | Provider abstraction and Ollama first | Accepted |
| [0008](0008-canonical-model-identity.md) | Canonical model identity | Accepted |
| [0009](0009-setspec-schema-strategy.md) | SetSpec schema and versioning strategy | Accepted |
| [0010](0010-queue-implementation.md) | Database-backed queue, no broker | Accepted |
| [0011](0011-shared-package-boundaries.md) | Shared package boundaries and extraction timing | Accepted |
| [0012](0012-prompt-storage-format.md) | Prompts as versioned JSON records | Accepted |
| [0013](0013-api-versioning.md) | API versioning | Accepted |
| [0014](0014-authentication-strategy.md) | Authentication strategy | Accepted |
| [0015](0015-repository-and-distribution-model.md) | Repository and distribution model | Accepted |
| [0016](0016-unavailable-is-not-zero.md) | Unavailable is not zero | Accepted |
| [0017](0017-benchmark-confidence-and-freshness.md) | Benchmark confidence and freshness | Accepted |
| [0018](0018-external-benchmark-isolation.md) | External benchmark isolation and sandboxing | Accepted |
| [0019](0019-python-baseline-and-config-format.md) | Python baseline and configuration format | Accepted |
| [0020](0020-ui-rendering-strategy.md) | UI rendering strategy | Accepted |
| [0021](0021-telemetry-collection-strategy.md) | Telemetry collection strategy | Accepted |
| [0022](0022-capability-evidence-record-contract.md) | Capability evidence record contract | Accepted |
| [0023](0023-runtime-profile-resolution.md) | Runtime profile resolution and served context | Accepted |
| [0024](0024-canonical-id-and-model-references.md) | Canonical ID format and model references in URLs | Accepted |
| [0025](0025-envelope-boundaries.md) | Envelope boundaries: what carries a SetSpec envelope | Accepted |
| [0026](0026-local-http-hardening.md) | Local HTTP hardening: Host validation, CSRF and outbound fetch | Accepted |
| [0027](0027-multi-gpu-semantics.md) | Multi-GPU semantics | Accepted |
| [0028](0028-prompt-pack-granularity.md) | Prompt attribution granularity and shared prompt tooling | Accepted |
| [0029](0029-queue-mechanics.md) | Queue mechanics: ageing, attempts, admission states and leases | Accepted |

## Writing a new ADR

1. Copy the format above; number it sequentially.
2. Record real alternatives with real reasons — an ADR whose alternatives are strawmen is worthless.
3. Link it from this index and from the documents it governs.
4. Never delete an ADR. Supersede it.

## Amendments

ADRs 0022–0029 were added by the [final architecture audit](../reviews/final_architecture_audit.md)
on 2026-08-21, before implementation began. Where one of them narrows or corrects an earlier
decision, the earlier ADR carries an **Amended by** note at its head and the amending ADR states what
it changes. No earlier decision was reversed; each was found to be under-specified at a boundary
rather than wrong.
