# ADR-0013 — API versioning

**Status:** Accepted (2026-08-21)

## Context

LoadCoach's API is consumed by IdeaPress and potentially by third-party tools. FreeWeight's evidence
API is consumed by LoadCoach. Each application is released independently, so a consumer and a
producer will routinely differ in version. A cross-application API is a public contract from the
first release that another component depends on.

## Decision

**Path-based major versioning: `/api/v1/…`.**

* Only the **major** version appears in the path.
* Within a major version, changes are additive only: new endpoints, new optional request fields, new
  response fields. Clients must ignore unknown response fields.
* Breaking changes create `/api/v2/`, and `/api/v1/` continues to be served for at least one minor
  release of the application, marked deprecated with a removal version.
* `GET /api/v1/version` advertises the application version, the API versions served (current,
  supported, deprecated) and the SetSpec schema versions understood.
* Clients check compatibility on first contact and fail with `API_VERSION_UNSUPPORTED`, naming both
  versions, rather than probing endpoints.
* Every response carries `X-Api-Version`.
* The OpenAPI document is committed as a snapshot per application; a change to it without a
  changelog entry fails CI. This is the mechanism that catches accidental breaking changes.
* API versions are independent of the application's semantic version and of SetSpec schema versions.

## Alternatives considered

**Header-based versioning** (`Accept: application/vnd.suite.v1+json`). Purer REST. Rejected: harder
to use from `curl` and a browser, easy to omit, invisible in logs and in the OpenAPI URL, and it
complicates caching. The suite's users inspect and script against these endpoints by hand.

**Query-parameter versioning** (`?api_version=1`). Rejected: easy to forget, and it muddles routing.

**No versioning; never break.** Tempting for a small suite. Rejected: the requirements mandate
explicit versioning, and "never break" is a promise nobody can keep across a multi-year design.

**Semantic versioning in the path** (`/api/v1.2/`). Rejected: minor versions are additive by
definition, so putting them in the path forces clients to change URLs for changes that do not affect
them.

**Per-endpoint versioning.** Rejected: consumers would have to track a matrix of endpoint versions.

## Consequences

*Positive.* Obvious, greppable, curl-friendly, cache-friendly. FastAPI mounts `v1` and a future `v2`
as separate routers, so both can be served from one process during a deprecation window. The
committed OpenAPI snapshot turns "did we break the API?" into a CI answer.

*Negative.* A `v2` means duplicated routers and translation logic during the overlap. Bounded by the
deprecation window and by the expectation that most changes are additive.

*Negative.* Path versioning is coarse: a single breaking change to one endpoint bumps the whole
surface. Accepted deliberately — a consumer's mental model stays "I speak v1".

## Revisit when

A `v2` is actually needed. The overlap mechanics (shared services, thin v1 translation layer, a
deprecation header, and a documented sunset date) are specified then, not now.
