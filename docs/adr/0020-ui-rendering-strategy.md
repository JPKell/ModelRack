# ADR-0020 — UI rendering strategy

**Status:** Accepted (2026-08-21)

## Context

Three applications need polished, dense, technical web interfaces with live updates, large data
tables, charts, light/dark themes, accessibility and keyboard usability — sharing primitives through
MirrorWall without becoming identical.

The prior benchmark specification mandated a hand-written single-page application (`index.html`,
`app.js`, `styles.css`, hash routing, no build system). The implementation that followed reached a
12-line `app.js` stub and stopped. The other prior project built a separate, differently structured
vanilla UI. Two half-finished UIs, no shared components.

Constraints from the requirements: polished UIs; shared primitives via MirrorWall; do not prematurely
build a large custom frontend framework; use standard technologies; keep it lightweight; offline
capable.

## Decision

**Server-rendered HTML (Jinja2) with progressive enhancement by small vanilla ES modules. No npm, no
bundler, no SPA framework.**

1. **Pages are rendered on the server.** Routing, data fetching, authorization and state live in
   Python, where they are already tested. HTML routes are thin and call the same services as the API.
2. **MirrorWall ships shared Jinja macros** (buttons, tables, badges, cards, drawers, dialogs, forms,
   the telemetry bar, chart containers), a base layout, and the token CSS. Applications supply their
   own pages and navigation.
3. **Interactivity is added as islands** — small ES modules (native `import`, no bundler) attached to
   `data-`-annotated elements: table sort/filter/column visibility, drawers, dialogs, theme toggle,
   SSE subscriptions, chart rendering.
4. **Live updates come over SSE** ([ADR-0004](0004-sse-vs-websockets.md)) and patch the DOM in place.
   Progress survives a refresh because events replay.
5. **Read-only content works without JavaScript.** Tables render server-side, forms submit normally.
   JavaScript improves the experience; it is not required to see data.
6. **Charts** use a vendored charting library (ECharts, offline, ~1 MB, cached) inside a MirrorWall
   chart container that owns theming and the accessible table alternative.
7. **No build step.** CSS and JS are served as authored. Assets are versioned by content hash in the
   URL for caching.

## Alternatives considered

**Hand-written SPA (the prior plan).** Rejected: three applications would each re-implement routing,
state, fetching and rendering in untyped, untested JavaScript, and the evidence says these never get
finished. It also duplicates authorization and data-shaping that already exist in Python.

**React/Vue/Svelte with a build pipeline.** Rejected: npm, a bundler, a lockfile, a build step in CI,
and a second language ecosystem for a local-first tool whose interfaces are forms, tables and charts.
The requirements explicitly warn against a large custom frontend framework, and this is the heavier
version of that mistake.

**HTMX.** Genuinely close to this decision and seriously considered — it fits server-rendered
partials with progressive enhancement precisely. Rejected as a *dependency*: what we need from it
(swap a fragment, subscribe to SSE) is a few dozen lines of ES module, while adopting it would put a
third-party runtime in the critical path of every interaction and pull application logic into HTML
attributes. The patterns HTMX popularized are adopted; the library is not.

**Alpine.js / Stimulus.** Same reasoning, smaller scope. Rejected for the same reason and to keep
the vendored asset budget for the charting library.

**Web Components.** Rejected as a base: more ceremony than value for this scale, and awkward
server-side rendering interaction. Individual components may use custom elements internally if it
helps.

**A desktop framework (Electron/Tauri).** Rejected: the applications are servers first, and a browser
is a fine client.

## Consequences

*Positive.* One language for logic. Fast first paint and no bundle to download. Accessibility is
easier with real HTML. Testable through ordinary HTTP tests plus template snapshot tests. Genuinely
offline. MirrorWall macros are shareable in a way React components never would be across three
independently versioned applications. The UI degrades to functional without JavaScript.

*Negative.* Highly interactive views (a live-updating dense table with client-side sorting over
thousands of rows) need real JavaScript, and we write it ourselves. Bounded by the JS budget in
[Performance Targets §3.7](../architecture/performance-targets.md) (≤ 60 KB per page excluding the
charting vendor) and by keeping components small, documented and unit-tested with a lightweight
browser-free harness.

*Negative.* Full page loads for navigation. At loopback latency this is imperceptible, and the
budgets require it to stay so.

*Negative.* No component-level type checking in JS. Mitigated by keeping modules small, by JSDoc
annotations, and by putting all real logic in Python.

## Revisit when

An application genuinely needs a highly interactive client-side workspace — a plausible future for
IdeaPress's editor. The escape hatch is deliberate: a single page may become a client-rendered
island with its own tooling **without** changing the other pages, because the JSON API already
exists and is versioned.
