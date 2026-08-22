# ADR-0004 — Server-Sent Events for streaming

**Status:** Accepted (2026-08-21)

## Context

Three applications stream: benchmark progress and live telemetry (FreeWeight), job state and token
deltas (LoadCoach), stage progress and draft deltas (IdeaPress). All of it is **server → client**.
The only client → server messages in the design are ordinary requests: start, cancel, submit
feedback.

The requirements ask for the streaming technology to be evaluated with SSE favoured unless
WebSockets have a demonstrated requirement.

## Decision

**Server-Sent Events** for all streaming in all three applications, over the shared conventions in
[API Standards §8](../standards/api-and-contract-standards.md):

* `GET` endpoints returning `text/event-stream`.
* Gap-free per-stream integer `id`, starting at 1.
* Events persisted **before** publication; the store is the source of truth and the in-memory
  fan-out is a latency optimization.
* `Last-Event-ID` replay on reconnect, with no gap and no duplicate across the replay/live handoff.
* Heartbeat comment every 15 s.
* Named events (`run.started`, `sample.completed`, `token`, `job.state_changed`).
* Bounded subscriber queues: a slow client is dropped and reconnects, rather than growing server
  memory.
* A terminal event is always emitted before the server closes the stream.

WebSockets are not used anywhere in the suite.

## Alternatives considered

**WebSockets.** Bidirectional, binary-capable, lower per-message overhead. Rejected: no
bidirectional requirement exists; they need a heavier server and client implementation, custom
reconnect and replay logic (there is no `Last-Event-ID` equivalent), custom heartbeat/ping handling,
and they are harder to test, to `curl`, and to proxy. The one place they would win — a
high-frequency bidirectional control channel — does not exist in this design.

**Long polling.** Works everywhere, needs no special transport. Rejected: higher latency, more
request overhead, and it re-implements event ordering and replay by hand.

**Polling a JSON endpoint.** Simplest of all and genuinely adequate for telemetry. Rejected as the
*primary* mechanism because benchmark progress and token streaming need low latency, and because one
mechanism for all three is simpler than two. Polling remains the documented fallback for a client
that cannot use SSE, since the same events are readable via
`GET /api/v1/runs/{id}/events?after_sequence=N`.

**A message broker (Redis pub/sub, NATS).** Rejected: single-process applications on one machine.
See [ADR-0010](0010-queue-implementation.md) for the same reasoning applied to queues.

## Consequences

*Positive.* Native browser `EventSource` with automatic reconnection and `Last-Event-ID`. Plain HTTP
— proxies, `curl`, and the CLI all work without special handling. Persisted events make replay,
restart survival and the run timeline the *same* mechanism. Testing is ordinary HTTP testing.

*Negative.* One-directional: cancellation is a separate `POST`, which is fine and arguably clearer.
Browsers historically limit concurrent HTTP/1.1 connections per origin to six — a dashboard opening
several streams could hit it. Mitigated by multiplexing (one telemetry stream shared across the
page, one run stream) and by HTTP/2 when a reverse proxy provides it; documented in the UI standards.

*Negative.* Text-only frames; every payload is JSON. No binary use case exists.

*Negative.* Persisting every event costs writes. Mitigated by batching, by an event-retention policy
per entity, and by never persisting raw telemetry outside a measured run.

## Revisit when

A genuine bidirectional, low-latency requirement appears — for example an interactive chat surface
with client-side interruption semantics that a `POST /cancel` cannot express, or collaborative
editing in IdeaPress. At that point WebSockets would be added **for that feature only**, not as a
replacement for the event stream.
