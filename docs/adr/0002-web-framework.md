# ADR-0002 — Web framework: FastAPI

**Status:** Accepted (2026-08-21)

## Context

All three applications need an HTTP layer serving a JSON API, server-rendered HTML, and long-lived
streaming connections. The requirements demand explicit API versioning, an error envelope, request
IDs, schema validation, OpenAPI documentation and typed request/response representations.

The prior benchmark specification forbade web frameworks entirely and mandated
`http.server.ThreadingHTTPServer` with hand-written routing. The partial implementation that
followed it produced ~350 lines of server and dispatch code, with no validation, no generated
documentation and no typed request models, before a single benchmark existed. That constraint is
rejected in the [inventory](../inventory/legacy-material-inventory.md); dependency minimalism
survives as a budget, not a prohibition.

Requirements that bear directly on the choice:

* OpenAPI documentation and JSON Schema for the public API.
* Typed request/response models (the suite already uses pydantic for SetSpec).
* Streaming responses and many concurrent long-lived connections.
* Server-rendered HTML alongside the JSON API ([ADR-0020](0020-ui-rendering-strategy.md)).
* A thin HTTP layer — business logic lives in services regardless of framework.

## Decision

**FastAPI** (on Starlette, served by Uvicorn) for all three applications.

* JSON API routes use pydantic models for request and response; OpenAPI 3.1 is generated at
  `/api/v1/openapi.json`.
* HTML routes use Starlette's Jinja2 templating with MirrorWall's shared templates.
* Streaming uses `StreamingResponse` with the SSE conventions in
  [API Standards §8](../standards/api-and-contract-standards.md).
* Handlers that block (database, provider, filesystem) are declared `def` and run in Starlette's
  worker threadpool; only genuinely async work is `async def` ([ADR-0003](0003-sync-vs-async-strategy.md)).

## Alternatives considered

**Flask (+ Flask-Smorest or apispec).** Mature, simple, synchronous, and a fine fit for FreeWeight
alone. Rejected because long-lived SSE connections under a WSGI server consume a worker thread each:
LoadCoach's telemetry stream plus per-job streams plus browser dashboards across three applications
makes "one thread per open connection" a real constraint at exactly the moment the product is most
useful. OpenAPI support is bolt-on, and typed request models require an extra layer that FastAPI
provides natively.

**Starlette alone.** FastAPI's foundation without its dependency injection, validation or OpenAPI
generation. Rejected: we would rebuild request validation and schema generation — the two things the
requirements explicitly ask for.

**Litestar.** Technically excellent, comparable feature set, better-structured DI. Rejected on
ecosystem size: fewer contributors, less documentation, and less agent/tooling familiarity, against
no capability the suite needs that FastAPI lacks.

**Django / Django REST Framework.** Rejected: an ORM, admin, auth and migration stack we do not
want, and a project structure at odds with the domain-first layering.

**Hand-rolled `http.server`.** Rejected — see Context.

## Consequences

*Positive.* Typed request/response models with automatic validation and error shapes; OpenAPI and
JSON Schema for free, which the contract tests consume; async streaming without a thread per
connection; dependency-injection hooks for the composition root; a large, well-documented ecosystem;
`TestClient` makes e2e testing straightforward.

*Negative.* Four runtime dependencies where zero were once claimed (`fastapi`, `starlette`,
`uvicorn`, `pydantic` — pydantic was already required by SetSpec). Async is now available, which
invites accidental blocking in an `async def` handler — mitigated by ADR-0003's explicit rule and a
review checklist item. FastAPI's DI can bleed framework types into services if abused — mitigated by
the layering contract, which forbids `fastapi` imports below `web/`.

*Neutral.* Uvicorn is the production server; no Gunicorn/worker-process model is used, because the
applications keep in-process state (schedulers, samplers, event fan-out) that a multi-process model
would fragment. Scaling out is not a current requirement.

## Revisit when

* A deployment needs multiple worker processes for one application (in-process state would have to
  move to the database or a broker first — a larger decision than the framework).
* FastAPI's maintenance stalls or its pydantic coupling blocks a required pydantic upgrade.
* Measurements show the threadpool bridge is a bottleneck rather than the provider.
