# Changelog

All notable changes to `modelrack` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/), pre-1.0 per
packaging and release standards §3.

## [Unreleased]

## [0.5.0] — 2026-08-26

Phase 5 of the [development plan](docs/packages/modelrack/development-plan.md), and the last one:
residency, cancellation hardening, the metadata cache and the observability hook. Every acceptance
criterion in [spec §20](docs/packages/modelrack/spec.md) is now met, which makes this release the
one where `modelrack` stops being a package under construction and starts being the thing
FreeWeight, LoadCoach and IdeaPress are built against.

The theme of the phase is that four features which each *look* like a convenience are, at this
layer, promises a scheduler builds on. A metadata cache that served a stale digest would let a run
record weights that never ran. A cancelled stream that leaked its socket would take a worker down
after a few hundred jobs rather than immediately. An `unload` that quietly did nothing would leave
LoadCoach believing it had freed VRAM it had not. Each is implemented as the guarantee rather than
the convenience, and each has a test that fails when the guarantee stops holding.

### Added
- `modelrack.residency`: the residency vocabulary every adapter and every caller shares.
  `residency_support()` projects a provider's declaration into `ResidencySupport(can_query,
  can_control)`, so LoadCoach's rule that "providers without residency control simply skip all of
  this" is a one-line branch taken *before* spending a call rather than a `try` around an
  operation the caller already knew would be refused. `find_resident()` and `is_resident()` match
  on the provider-side model name alone — the only field a provider's residency report and a
  caller's `ModelIdentity` genuinely agree on, since a provider does not re-derive a digest for
  what it has loaded (ADR-0024 §2), and comparing whole identities would report a digest-pinned
  identity as *not* resident against the very entry running it.
- `modelrack.cache`: `MetadataCache`, the one cache [spec §3](docs/packages/modelrack/spec.md)
  carves out of its own prohibition — in-memory, TTL-bounded (default 300 s), thread-safe,
  inspectable through `CacheStats` and clearable. Its clock is monotonic and injected: a TTL
  measured against the wall clock is extended or expired by an NTP correction, and a five-minute
  expiry asserted with a real `sleep` is a test nobody runs. `MetadataSnapshot` pairs a payload
  with the instant it was read, because caching the payload alone would falsify
  `ModelDescriptor.observed_at`, whose documented meaning is *when this snapshot was read from
  the provider*.
- `modelrack.events`: the optional `on_event` hook [spec §17](docs/packages/modelrack/spec.md)
  asks for — `ProviderEvent`, `ProviderEventKind`, `EventEmitter` and `emit()`. An event carries
  no prompt, no generated text, no tool arguments and no credential, and the enforcement is
  structural: there is no field one could reach. What it does carry is the caller's own
  `GenerationRequest.metadata`, the mapping that is never sent to the provider, which is how a
  host joins an event to its own run without this package knowing what a run is.
- `refresh: bool = False` on `Provider.list_models`, `inspect_model` and `resolve`, implemented by
  all three adapters. The development plan names the stale-digest failure mode and its mitigation
  as "TTL **plus** an explicit `refresh=True` path": a tag such as `qwen3.5:latest` can be
  repointed the moment after it is read, so a caller who *knows* a model was re-pulled says so
  rather than waiting out an expiry. An adapter that caches nothing accepts the argument and
  ignores it, so a caller holding a `Provider` never has to ask which kind it is holding.
- `metadata_ttl_seconds` and `on_event` constructor arguments on `OllamaProvider` and
  `OpenAICompatibleProvider`; `on_event` on `FakeProvider`. Configuration stays constructor-only
  ([spec §12](docs/packages/modelrack/spec.md)): this package still reads no environment variable
  and no file.
- `metadata_cache_ttl_seconds`, `metadata_cache_stats()` and `clear_metadata_cache()` on both real
  adapters — spec §10's requirement that the cache be documented, inspectable and clearable.
- `modelrack.provider.require_capability()` and `refuse_capability()`: one spelling of ADR-0007
  rule 2's refusal, shared by every adapter. The message, the error type and — most of all —
  `details["capability"]` are now identical from all three, which a new test asserts directly:
  a downstream test that matches on a refusal must not pass against the fake and fail against
  Ollama.
- `tests/unit/test_cancellation.py`, and with it acceptance criterion 5. A counting
  `httpx.BaseTransport` hands back response bodies that report whether they are still open, which
  makes "leaves no open connection" a genuine count rather than an inspection of somebody else's
  pool. Seven exit paths are covered — drained, cancelled, abandoned, explicitly closed, failed
  mid-flight, truncated, refused before streaming — plus a `KeyboardInterrupt` landing inside
  error classification, and a hundred sequential streams of each kind leaving nothing open.
- `tests/performance/test_overhead.py`: spec §15's budgets (≤ 5 ms per non-streaming request,
  ≤ 1 ms per streamed chunk, ≤ 10 ms for a cached twenty-model listing), measured against a
  transport that does no I/O so that what the clock sees is this package's own work. Each budget
  is asserted twice — once bare, once with an `on_event` callback attached — because an
  observability hook that quietly costs a millisecond a chunk is a hook nobody can afford to turn
  on.
- `docs/quickstart.md`: ten sections from `pip install` to the error table, all of it runnable
  without a GPU, a model or a server until the section that says otherwise.
- `docs/providers.md` and `scripts/generate_provider_matrix.py`: the capability matrix across all
  three adapters, **generated from each adapter's own `capabilities()`** as the development plan's
  Phase 4 acceptance criterion 2 requires — a hand-written matrix is a claim about an adapter, a
  generated one is the adapter's own statement, and the two diverge silently. Late rather than
  never: this was Phase 4's, and the omission was found while reconciling the repository's copy of
  the development plan against the suite's master copy, which had been softened in the local copy.
  `tests/unit/test_provider_matrix.py` fails when the committed file and the adapters disagree, so
  a stale matrix is caught by the ordinary test run rather than by review.
- `.github/workflows/nightly.yml`: the `performance` suite on a schedule, and the `live` suite
  present-but-disabled pending a self-hosted runner with an Ollama on it. Testing standards §10
  puts both in a nightly job; a marked test that runs nowhere is an untested behaviour with a
  green tick beside it.

### Changed
- Cancellation is now checked **before** a connection is opened. A token already set when
  `stream()` is called yields exactly one terminal `StreamFailed(GenerationCancelled)` and opens
  no socket at all — previously the adapter connected, read one chunk and cancelled on it. The
  event is delivered rather than raised, so a caller keeps one cancellation path instead of two,
  and it is ordered *after* the capability checks: a request naming something the provider never
  declared is malformed whichever way the token points, and reporting it as a cancellation would
  hide the caller's own bug. Both real adapters and the fake behave identically, and the
  conformance suite now asserts it for all of them.
- Streaming in both real adapters is split into a `_drain` generator and an observing `_walk`
  wrapper. `_drain` has six terminal exits, and an emitter called from each of them is one that
  will eventually be forgotten at a seventh; the wrapper sees them all, so "every stream reports
  how it ended" is structural. The wrapper closes the inner generator explicitly in `finally`,
  because without that the `response.close()` it guards would run only when the garbage collector
  reached it — prompt in CPython, unspecified elsewhere.
- `OllamaProvider.load()`, `unload()` and `list_resident()` now apply the same capability gate the
  OpenAI-compatible adapter applies, rather than assuming their own declaration. Ollama declares
  both flags, so no behaviour changes; what changes is that the refusal path is reached from one
  place in the codebase instead of two, and cannot drift between adapters.
- `OllamaProvider.load()` and `unload()` match residency through `find_resident()` rather than a
  private name comparison, so a digest-confident identity — which is what LoadCoach holds — is
  recognised as resident against Ollama's name-only `/api/ps` report.
- `ModelDescriptor.observed_at` on a cache hit is the instant the provider answered, not the
  instant the descriptor was assembled; and a descriptor built from a fresh `/api/show` and an
  older `/api/tags` entry takes the **older** of the two, because a snapshot is only as current as
  its stalest half — which matters most for the digest, the one field on it that can change under
  a caller without warning.
- `FakeProvider` emits the same event stream a real adapter does, deliberately: a downstream
  repository testing its observability against this double is testing against the shape it will
  see in production, or against nothing at all.
- README: a new section covering all four Phase 5 features, and a development section listing the
  full gate rather than only `pytest`.

### Fixed
- The fake's capability refusals were a private near-copy of the OpenAI-compatible adapter's, which
  meant three implementations of one message and three chances for it to drift. All three now
  delegate to `modelrack.provider.require_capability()`.
- **`health()` could raise, in both real adapters.** The `Provider` protocol is explicit that it
  returns `UNAVAILABLE` rather than raising, because "is it up?" is a question whose negative
  answer is not exceptional and an application's health endpoint asks it precisely when it expects
  the answer might be no. Both adapters caught only the transport-shaped errors: `OllamaProvider`
  escaped a typed `ProviderUnavailable` when `/api/version` answered and `/api/tags` then did not
  — a server shutting down between the two calls, which is the moment a health probe is most
  likely to run — and `OpenAICompatibleProvider` escaped a `ProviderRejected` on a 401, i.e. on
  every wrong or expired `api_key`. One bad credential turned into a 500 for the caller's entire
  health document. Present since Phases 3 and 4 respectively; found while reviewing this phase's
  work, and now covered by a conformance test that calls `health()` on every adapter and asserts
  only that it returned.
- Relatedly, a provider that **answers and refuses** is now reported as
  `ProviderStatus.DEGRADED` — "reachable, but something is wrong" — rather than as `UNAVAILABLE`.
  A 401 from a running server is a different operational state from nothing listening, and
  conflating them sends an operator to check the wrong thing. `DEGRADED` was in the health
  vocabulary from Phase 1 and no adapter had ever produced it, which was the smell. The detail
  names the error *code* and never the server's own message: a health document is rendered into a
  UI, which makes it one more channel a credential or a prompt echo could otherwise escape
  through (spec §14), and a test asserts both stay out of it.
- **Streaming held about 993 KB of pure overhead on a long answer, against spec §15's 1 MiB
  budget for an entire active stream.** Both real adapters assembled the response into a
  `list[str]` of per-chunk fragments and joined it at the end. A CPython `str` costs roughly 49
  bytes of object header, so a 20 000-chunk generation — a long but ordinary answer — retained
  20 000 live objects carrying 8 bytes of text each: about six times the size of the answer, in
  headers alone, held for the whole stream. Both now accumulate into an `io.StringIO`, which
  brings the same stream from ~993 KB of overhead to ~20 KB, flat. Present since Phases 3 and 4;
  it passed every existing test because nothing measured memory, and it is exactly the kind of
  defect the development plan put a performance suite in this phase to find.
- `tests/performance/test_overhead.py` asserts that budget in the form that has teeth. Two things
  had to be right for the test to mean anything, and both were wrong in the first draft: the body
  must arrive in realistic per-read chunks, because a transport that hands `httpx` a whole body at
  once makes *its* `LineDecoder` return every line as one list — nearly two megabytes of somebody
  else's memory, in a state no socket produces; and the measurement must be the **live** set, not
  `tracemalloc`'s peak, because peak counts the transient payload-delta-index churn of every
  iteration and therefore grows with length on any implementation, including one that retains
  nothing. Measured correctly, the budget is asserted twice — as an absolute, and as *flatness*
  between a 50-chunk and a 20 000-chunk stream. The old accumulator slipped under the absolute
  figure by 45 KB and fails the flatness assertion by 73×, which is why both are there.
- `docs/packages/modelrack/development-plan.md` in this repository had drifted from the suite's
  master copy — Phase 4's `docs/providers.md` deliverable and its acceptance criterion had been
  softened, and Phase 5's file list had lost the same file. The mirror is a downstream copy and is
  now re-synced with the master.

### Documentation
- [Spec §7](docs/packages/modelrack/spec.md)'s protocol listing now shows the `refresh` keyword,
  and §10 states why it is on the protocol rather than on the adapters: a caller holding a
  `Provider` must be able to force a re-read without downcasting to a concrete adapter, which is
  the thing the abstraction exists to prevent. The development plan already required "TTL **plus**
  an explicit `refresh=True` path"; §7 had simply not been updated to match, and the specification
  is the authority, so the specification is what moved.

### Notes
- Coverage is 100 % of statements and branches, against a floor of 95 %.
- No Ollama, no GPU and no network are needed for the default suite; the socket guard in
  `tests/conftest.py` fails any test outside `tests/live/` that opens one.

## [0.4.0] — 2026-08-23

Phase 4 of the [development plan](docs/packages/modelrack/development-plan.md):
`OpenAICompatibleProvider`, the second real adapter, and the phase whose whole purpose is proving
the vocabulary Phase 1 designed is not secretly shaped around Ollama
(ADR-0007 rule 1). No type in `modelrack.types`,
`modelrack.streaming` or `modelrack.provider` changed to support it — the acceptance test the
design itself was under, stated plainly rather than only implied by a green test suite.

### Added
- `modelrack.providers.openai_compatible.OpenAICompatibleProvider`: the third `Provider`
  implementation, reached over `/v1/models` and `/v1/chat/completions`, streaming and not, with
  tool calls, JSON mode and JSON-Schema structured output. Imported from its own module, the same
  reason `OllamaProvider` is: it is the second and last place in this package `httpx` is imported.
- Honest capability declaration where this protocol genuinely differs from Ollama's, each proven
  by the conformance suite's *refusal* branch rather than only asserted: no digest anywhere in
  `/v1/models` (`identity_confidence` is `NAME_ONLY` unconditionally —
  ADR-0024 §2), no residency-control
  endpoint (`force_unload` and `residency_query` both `False`; `load`, `unload` and
  `list_resident` refuse immediately, before any HTTP call), no per-request field to set a served
  context length (`context_configurable` is `False`, and a request naming one is refused in
  `_build_body` before a byte is sent — spec §11.10), no backend timing breakdown (every
  `Timing.backend_*` field is `UNSUPPORTED`; only `client_*` fields are ever set), and
  `token_level_chunks = False` on principle: nothing in this streaming format promises one delta
  per model token.
- A structured error code where Ollama has only prose. `error.code == "context_length_exceeded"`
  is checked before any message-text sniffing — the same marker-phrase fallback
  `modelrack.providers._ollama_wire` uses, kept only as the fallback here because this protocol
  usually has something better.
- A minimal SSE parser (`_iter_sse_events`) for the one subset of the format this protocol needs:
  `data:` field lines (joined with `\n` across consecutive lines, per the SSE grammar), a blank
  line dispatching what was buffered, `:`-prefixed comment lines ignored, and the `[DONE]`
  sentinel. Verified directly against multi-line data, keep-alive comments, a malformed frame and
  a stream missing its `[DONE]` sentinel — the development plan's own Phase 4 test list, by name.
- Streamed tool calls reassembled from fragments that arrive a few characters at a time across
  many chunks and are not valid JSON until the last one lands — unlike Ollama, whose `arguments`
  is already a parsed object. A fragment that never becomes valid JSON is preserved as
  `ToolCall.raw_arguments` and yields empty `arguments` rather than raising, so the
  malformed-arguments case `modelrack.testing` scripts on purpose stays diagnosable against a real
  adapter, not just the fake.
- `tests/fixtures/providers/openai_compatible/`: recorded response shapes representative of
  llama.cpp server and LM Studio, version-annotated in that directory's `manifest.json` (spec
  §19), covering discovery, non-streaming and streaming generation (plain and tool-calling),
  every documented error body, and the SSE edge cases above.
- `TestOpenAICompatibleProviderConformance` in `tests/contract/test_conformance.py`: the shared
  behaviour suite, bound to this adapter over a recorded transport. Two real adapters now pass one
  conformance suite (spec §11.5's acceptance criterion, met a second time).
### Changed
- `modelrack/__init__.py`'s module docstring now names both real adapters and states the
  no-type-changes result plainly.

### Fixed
- Nothing shipped in Phase 3 broke; this is new surface, not a repair.

### Security
- The API key a caller supplies is sent only as `Authorization: Bearer <key>` and confirmed absent
  from every result's `raw`, from error `details`, and from a DEBUG-level log capture (spec §14) —
  proven directly rather than only by omission.
- Reuses `modelrack.providers._http`'s response size caps and connection-pooled client unchanged;
  no new transport surface was introduced.

## [0.3.0] — 2026-08-23

Phase 3 of the [development plan](docs/packages/modelrack/development-plan.md): `OllamaProvider`,
the first real adapter. Discovery, generation, streaming, tool calls, structured output, and
residency control, all reached over Ollama's HTTP API and proven against the same conformance
suite `FakeProvider` already passes — and, beyond the recorded-fixture default suite, against a
real Ollama 0.32.13 server on the reference machine (`pytest -m live`), including a hybrid
architecture whose `/api/show` response reports `attention.head_count_kv` as JSON `null`: handled
correctly as `UNSUPPORTED` with no code written specifically for that case, because the type check
that degrades any non-numeric value already covered it.

### Added
- `modelrack.providers.ollama.OllamaProvider`: the second `Provider` implementation, and the
  first to make a real HTTP call. Imported from its own module, not from `modelrack` or
  `modelrack.testing` — the one place in this package `httpx` is imported, so a process that only
  ever talks to the fake never pays for that import.
- `providers/_http.py`: transport plumbing shared by every real adapter — `build_client`,
  `validate_base_url` (http/https only, non-loopback flagged remote per spec §14),
  `translate_transport_error` for a failure before a connection is established,
  `translate_stream_interruption` for one *after* a stream has already yielded content
  (deliberately a different mapping — see Fixed, below), `read_capped_json` and
  `iter_capped_lines` for the two size caps spec §14 requires (default 64 MiB total, 8 MiB per
  streamed line), and best-effort `[Errno N]` message classification into
  `ProviderUnavailableReason`, documented as exactly that: message-sniffing, because httpx's own
  exception wrapping discards the structured `errno` a raw `OSError` carried.
- `providers/_ollama_wire.py`: pure translation from Ollama's wire shapes to this package's
  types — architecture-prefixed `/api/show` metadata (`general.architecture` read first, every
  other field looked up under that prefix, which is what makes the parser work across any model
  family without a hard-coded architecture name), nanosecond-to-millisecond backend timing
  extraction, tool-call parsing with synthesized ids (Ollama's own shape carries none), and a
  best-effort context-overflow message match — Ollama gives no distinct error code for that
  condition, only prose, so it is matched conservatively and documented as exactly that.
- NDJSON streaming built on `httpx.Response.iter_lines()`'s incremental UTF-8 decoder rather than
  hand-rolled buffering, verified directly (`tests/unit/test_ollama_adapter.py::TestNdjsonChunking`)
  against a multi-byte character split across two raw byte chunks, a JSON line split across two
  raw chunks including at the newline itself, and one line assembled from many small raw reads —
  the scenario the development plan names by name for this phase.
- Every scripted failure a real Ollama can produce, translated to the typed error spec §13 names
  for it: connection refused, timeout, a non-JSON body, an oversize response or streamed chunk,
  404 model-not-found, a 4xx (or a 5xx that still carries a provider message — see Changed, below)
  bad-options rejection, a context-overflow message, and a mid-stream in-band error — Ollama can
  signal a generation failure as an ordinary NDJSON line after already sending a 200, and this
  adapter delivers it as `StreamFailed` rather than raising, the same rule
  `modelrack.providers.fake` already follows for cancellation.
- `tests/fixtures/providers/ollama/`: recorded response shapes, version-annotated in that
  directory's `manifest.json` (spec §19), covering complete metadata, partial metadata, an unknown
  architecture, and every documented error body — the artifact spec §18's test-strategy table
  names for "Ollama adapter" and "Metadata normalization".
- `tests/live/test_ollama_live.py`: real discovery, generation, streaming, cancellation and
  unload against an actual server, marked `@pytest.mark.live` and excluded from the default run.
  Skips gracefully when unreachable; `MODELRACK_REQUIRE_OLLAMA=1` turns that skip into a failure,
  the escape hatch `WEIGHTSDB_REQUIRE_POSTGRES` already gives WeightsDB's own conditionally-skipped
  dialect tests, so a broken Ollama integration cannot hide behind a silent skip in the nightly job.
- `TestOllamaProviderConformance` in `tests/contract/test_conformance.py`: the shared behaviour
  suite, bound to this adapter over a recorded transport that models real server-side state (which
  model is currently resident) rather than static canned responses, because the residency
  behaviours load a model and then unload it and expect the second call to see what the first did.

### Changed
- Error classification after a stream has already yielded content is *not* the same mapping used
  before one starts. `ProviderUnavailable`'s own contract is that the provider "could not be
  reached **at all**", which a connection that was already delivering deltas contradicts; a reset
  or malformed-framing failure there is classified as `ProviderProtocolError` instead — spec
  §13's "stream truncated without a terminal chunk" row, applied to *why* it truncated rather than
  only to a clean-but-incomplete close. A timeout stays a timeout either way.
- `ProviderRejected` is produced from *any* status carrying an extractable `{"error": "..."}`
  message, not only a 4xx. Spec §13's table names "4xx with a provider message", but Ollama is not
  rigorously HTTP-semantic about which status accompanies which failure, and trusting the exact
  status number over the message it carries would be the version-fragility
  risk register E1 names as this adapter's own biggest risk.
  An unexpected status with *no* extractable message still falls back to `ProviderProtocolError`.
- `list_models()` costs one `/api/show` call per model on top of one `/api/tags` call, which is
  exactly what [spec §15](docs/packages/modelrack/spec.md)'s performance budget already priced in
  ("cold, 20 models: ≤ 3 s, dominated by per-model `show` calls") — `/api/tags` alone carries no
  layer count, head counts or embedding width, and those are what FreeWeight's KV-cache benchmark
  needs.

### Fixed
- Nothing shipped in Phase 2 broke; this is new surface, not a repair.

### Security
- Response size caps enforced during the read itself (`read_capped_json`, `iter_capped_lines`),
  never after buffering the whole body first — the point of a cap (spec §14) is never holding more
  than the limit in memory, and checking only afterward would have already spent it.
- Every request disables redirect-following (`follow_redirects=False`): a local inference runtime
  has no legitimate redirect to follow, and one would be a silent change of which server is
  actually being talked to.

## [0.2.0] — 2026-08-23

Phase 2 of the [development plan](docs/packages/modelrack/development-plan.md): `FakeProvider`,
the first adapter, built before the real one on purpose
(ADR-0007 rule 6). From here FreeWeight's runner,
LoadCoach's executor and IdeaPress's workflows can be developed and tested with no GPU, no model
and no network.

### Added
- `modelrack.testing`: `FakeProvider`, `FakeScript`, `FakeGeneration`, `FakeModel`,
  `FakeToolCall`, `FakeFailure`, `FakeFailureMode`, `FULL_CAPABILITIES`, `MINIMAL_CAPABILITIES`,
  `DEFAULT_MODEL` and `SIMULATED_TOKEN_CHARACTERS`. This is the supported import path the
  testing standards §7 name; the fake is deliberately
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
  (ADR-0024 §2). `resolve()` handles exact
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
(ADR-0007 rule 6).

### Added
- `types`: `Role`, `Message`, `ToolDefinition`, `ToolCall`, `SamplingParameters`,
  `ResponseFormat`/`ResponseFormatKind`, `FinishReason`, `Timing`, `GenerationUsage`,
  `GenerationRequest`, `GenerationResult`. Every count and duration defaults to `UNSUPPORTED`
  rather than `0` (ADR-0016), and every value object
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
  ADR-0030 defines `TokenUsage` as **billing**
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
