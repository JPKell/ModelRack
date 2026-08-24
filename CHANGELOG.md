# Changelog

All notable changes to `modelrack` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/), pre-1.0 per
`docs/standards/packaging-and-release-standards.md` §3.

## [Unreleased]

## [0.2.0] — 2026-08-23

Phase 2 of the [development plan](docs/packages/modelrack/development-plan.md): `FakeProvider`,
the first adapter, built before the real one on purpose
([ADR-0007](docs/adr/0007-provider-abstraction.md) rule 6). From here FreeWeight's runner,
LoadCoach's executor and IdeaPress's workflows can be developed and tested with no GPU, no model
and no network.

### Added
- `modelrack.testing`: `FakeProvider`, `FakeScript`, `FakeGeneration`, `FakeModel`,
  `FakeToolCall`, `FakeFailure`, `FakeFailureMode`, `FULL_CAPABILITIES`, `MINIMAL_CAPABILITIES`,
  `DEFAULT_MODEL` and `SIMULATED_TOKEN_CHARACTERS`. This is the supported import path the
  [testing standards](docs/standards/testing-standards.md) §7 name; the fake is deliberately
  **not** re-exported from `modelrack` itself, because a test double one autocomplete away from
  the production namespace is one refactor away from inside it.
- Determinism by construction. Text, chunking, token counts, tool-call identifiers and
  schema-shaped output all derive from SHA-256 over a canonical seed string built from the seed,
  the identity, the sampling seed, the call index and the rendered prompt. `random.Random` was
  rejected: its core generator is reproducible across releases but the derived helpers
  (`choice`, `sample`, `shuffle`) are not, and "identical across processes and platforms" has to
  survive a Python upgrade to mean anything. Proven by golden values plus a test that regenerates
  in a subprocess under three different `PYTHONHASHSEED`s — the failure mode being guarded is
  nondeterminism arriving through set or dict ordering.
- Scriptable behaviour covering the cases that actually break callers, not only the happy path:
  seeded or fixed text, per-chunk and slow-first-token delays, chunk sizes and hand-placed chunk
  boundaries (including a split inside a grapheme cluster), token counts, reasoning content,
  tool calls with valid, absent and unparseable arguments, finish reasons, output-limit
  truncation, and eight failure modes covering every row of
  [spec §13](docs/packages/modelrack/spec.md) a provider can be scripted into. The two remaining
  rows — `CapabilityUnsupported` and `GenerationCancelled` — arise from the fake's own gating and
  the caller's own token, and deliberately have no script member: a provider cannot decide to be
  cancelled, and one that could script a capability refusal while declaring the capability would
  break the mechanism the refusal exists to enforce.
- A configurable model catalogue with digests, aliases and the full `ModelDescriptor` metadata
  set. Digests are written down as a provider would report them — bare hex, prefixed, uppercase,
  truncated, non-hex, absent — and normalized through `baseaicore.normalize_digest` on the way
  out; one that will not normalize is discarded **with the reason recorded in the descriptor's
  `raw`**, yielding a `name_only` identity rather than a malformed one
  ([ADR-0024 §2](docs/adr/0024-canonical-id-and-model-references.md)). `resolve()` handles exact
  names, aliases and unique prefixes, refuses an ambiguous prefix rather than choosing, and logs
  the resolution at DEBUG so a retag is never hidden (spec §11.8).
- The provider conformance suite (`tests/contract/test_conformance.py`), the artifact spec §11.5
  requires: one set of behaviours, bound to the fake three times — fully capable, fully
  incapable, and streaming fragments that are honestly not one token each. Phases 3 and 4 add a
  class per recorded transport and change nothing else. Capability-gated behaviours are never
  silently skipped: where a capability is declared the behaviour is exercised, and where it is
  not the suite asserts the adapter *refuses* with `CapabilityUnsupported` naming the flag.

### Changed
- `FakeProvider`'s constructor takes more than [spec §7](docs/packages/modelrack/spec.md)'s
  sketch: `sleep`, `clock` and `default_timeout_seconds` join `script` and `seed`. All three are
  injected boundaries the spec requires the behaviour of elsewhere — §14 mandates a default
  timeout, and a descriptor's `observed_at` must come from somewhere a test can freeze. `sleep`
  defaults to `None`, meaning a scripted 900 ms first token is reported in `Timing` without
  costing wall time; passing `time.sleep` makes it real for the tests that need it.
- Everything a script can express beyond one call — the catalogue, the capability declaration,
  the health verdict, the provider version, the base URL — lives on `FakeScript` rather than on
  the constructor, and per-call behaviour lives in `FakeScript.generations`. The development plan
  lists the catalogue among the scriptable behaviours, and one frozen object that
  `dataclasses.replace` varies is easier to share between tests than six constructor arguments.
- `providers/` gained an `__init__.py`. It was an implicit namespace package, which works until
  something in the toolchain assumes otherwise; a typed distribution should not rely on that.

### Fixed
- Nothing. Phase 1 shipped no behaviour to break.

### Security
- Caller `metadata` never reaches the provider payload, the prompt never reaches it, and neither
  does the generated text — the first two because correlation data and prompts are never sent to
  a provider (spec §7), the third because a streamed response accumulated twice would break the
  flat per-stream memory budget in spec §15. All three are asserted.
- A scripted error body is capped at 512 characters like a default one. An error object is not a
  place to move an unbounded response, and a fake that let a script past the limit would let a
  consumer build an expectation a real adapter can never meet.

### Notes for Phase 4
- `load()` is gated on `force_unload`. The normative capability set has no separate "can load"
  flag and ADR-0007 rule 2 fixes the field list, so `force_unload` is read as the single
  statement that residency is controllable at all. Phase 4 is where the interface meets a real
  provider that cannot control residency, and is the right moment to confirm or split it.

## [0.1.0] — 2026-08-23

Phase 1 of the [development plan](docs/packages/modelrack/development-plan.md): the
provider-neutral vocabulary and the `Provider` protocol. No I/O and no adapter — `FakeProvider`
arrives in Phase 2, `OllamaProvider` in Phase 3, in that order and deliberately, so the rest of the
suite can be built and tested without a GPU, a model or a running runtime
([ADR-0007](docs/adr/0007-provider-abstraction.md) rule 6).

### Added
- `types`: `Role`, `Message`, `ToolDefinition`, `ToolCall`, `SamplingParameters`,
  `ResponseFormat`/`ResponseFormatKind`, `FinishReason`, `Timing`, `GenerationUsage`,
  `GenerationRequest`, `GenerationResult`. Every count and duration defaults to `UNSUPPORTED`
  rather than `0` ([ADR-0016](docs/adr/0016-unavailable-is-not-zero.md)), and every value object
  is frozen — a result that later code mutates no longer describes the call it came from.
- `Timing` keeps the provider's account and this process's observation apart:
  `backend_load_ms`, `backend_prompt_eval_ms`, `backend_decode_ms`, `backend_total_ms`,
  `client_wall_ms`, `client_ttft_ms`. There is deliberately **no** combined or unprefixed
  duration, which is Phase 1's acceptance criterion 2 — the moment one exists, callers reach for
  it and a benchmark starts comparing one runtime's self-report against another's wall clock.
  `backend_total_ms` is the provider's own total and is never recomputed from the phases.
- `streaming`: the `StreamEvent` union (`TokenDelta`, `ThinkingDelta`, `ToolCallDelta`,
  `StreamCompleted`, `StreamFailed`) and `CancellationToken`. Every stream ends with exactly one
  terminal event, so a truncated stream is detectable as the absence of one rather than being
  indistinguishable from a short answer. `CancellationToken` is backed by `threading.Event`
  (cancelling from a request handler while a background thread streams is the ordinary case), is
  one-way and idempotent, and its `raise_if_cancelled` hands the caller back its own partial
  output rather than discarding it.
- `provider`: the structural `Provider` protocol, plus `ProviderHealth`, `ProviderStatus`,
  `ProviderCapabilities`, `LoadResult` and `ResidentModel`. All 13 capability flags default to
  `False`, because the honest default for "did this adapter declare it?" is no — a capability
  that appears by omission is one nobody tested. `ProviderStatus` reuses the suite's own health
  vocabulary (`ok`/`degraded`/`unavailable`) so an application maps a result straight into
  `GET /api/v1/health` without inventing a translation that could drift.
- `errors`: the full hierarchy from spec §7 — `ProviderError` and its eight subclasses — with the
  documented codes, so no adapter ever raises a raw `httpx` exception
  ([spec §11.7](docs/packages/modelrack/spec.md)). `ProviderUnavailableReason` distinguishes a
  refused connection from a DNS failure from a TLS failure, because those need different
  responses from a human.
- `py.typed`, without which every downstream `mypy --strict` would silently treat this package as
  untyped — the protocol would still exist, but nothing would check that FreeWeight's or
  LoadCoach's use of it is correct, and the `Typing :: Typed` classifier would be a claim the
  distribution did not honour. Verified end to end: a consumer installing only the built wheel
  type-checks against the protocol and is told when it misuses one.
- A `conftest` socket guard that fails any test outside `tests/live/` which opens a connection.
  Installed now, before any adapter exists, precisely because adding it in Phase 3 alongside the
  first HTTP code would mean writing the adapter and its safety net at the same moment — and
  "the default suite passes with no Ollama running" ([spec §18](docs/packages/modelrack/spec.md),
  acceptance criterion 3) does not stay true by good intentions.

### Changed
- `GenerationResult.usage` is a `GenerationUsage`, not a bare `baseaicore.TokenUsage` as spec §7's
  sketch shows. The two documents conflict and the ADR is the newer one:
  [ADR-0030](docs/adr/0030-model-cost-and-pricing.md) defines `TokenUsage` as **billing**
  vocabulary whose four classes are disjoint, and BaseAiCore's implementation explicitly declines
  a `thinking_tokens` field because every provider that exposes reasoning tokens bills them at its
  output rate. But FreeWeight's `samples` table, LoadCoach's job rows and IdeaPress's all persist
  `thinking_tokens`, `tool_tokens` and `output_chars`/`output_words`/`output_bytes`.
  `GenerationUsage` therefore *embeds* the billable `TokenUsage` unchanged — a consumer storing a
  cost stores `usage.tokens` directly — and carries the observation counts beside it, documented
  as breakdowns of `output_tokens` rather than a fifth disjoint class, so a total computed from
  `tokens` cannot double-count them.
- Coverage floor raised from 85 % to 95 %, the number [spec §18](docs/packages/modelrack/spec.md)
  and acceptance criterion 7 both state; the scaffold shipped 85 in `pyproject.toml` and in the CI
  job. Current coverage is 100 %.

### Fixed
- `.importlinter` checked nothing: its inline `root_packages = modelrack` was read character by
  character, so `lint-imports` failed with `Could not find package 'm'` on every run. Rewritten as
  newline-separated lists with `include_external_packages`, and `setspec` added to the forbidden
  set — [spec §5](docs/packages/modelrack/spec.md) says "Nothing else. (Not `setspec` …)", and the
  scaffold's list omitted it.
- `ruff format --check .` failed on ten vendored documents under `docs/`. ruff 0.16 formats Python
  code blocks inside markdown, and those files are byte-identical copies of the suite's master
  documents shared with nine other repositories — reformatting them here would desync every copy.
  `docs/` is now excluded from ruff instead.

### Security
- `pytest` moved from `>=8,<9` to `>=9.0.3,<10`, excluding PYSEC-2026-1845 (vulnerable
  `/tmp/pytest-of-{user}` handling, affecting pytest through 9.0.2). Matches the pin BaseAiCore
  and SetSpec already moved to.
