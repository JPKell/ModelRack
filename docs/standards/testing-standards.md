# Testing Standards

**Applies to:** every repository in the suite.
**Rule zero:** tests are planned before the implementation phase that needs them. Every phase in
every development plan lists the tests it must add and the acceptance criteria they prove.

---

## 1. Layout

Tests live in a top-level `tests/` directory so the source tree stays clean. They are never shipped
inside the package.

```text
<repo>/
├── src/<package>/
└── tests/
    ├── conftest.py            # shared fixtures, deterministic clock, temp roots
    ├── unit/                  # pure logic, no I/O, milliseconds
    ├── contract/              # public API/schema conformance in both directions
    ├── integration/           # real DB, real files, fake provider
    ├── e2e/                   # full app through HTTP and CLI, fake provider
    ├── performance/           # marked, excluded by default
    ├── live/                  # marked, requires a real provider/GPU, never in default CI
    └── fixtures/
        ├── providers/ollama/  # recorded responses + the provider version they came from
        ├── telemetry/         # captured /proc, /sys, nvidia-smi output
        ├── schemas/           # golden SetSpec payloads per version
        └── prompts/
```

---

## 2. Test types and what each must cover

| Type | Scope | Speed | Provider | Must cover |
|---|---|---|---|---|
| **Unit** | One function/class | < 10 ms | none | Every scoring/aggregation formula, every parser, every state transition, every validator, boundary and malformed input |
| **Contract** | A published interface | < 100 ms | fake | SetSpec round-trip, version acceptance/rejection, API request/response against the OpenAPI document, error envelope shape, port conformance for every implementation |
| **Integration** | Several components + real storage | < 2 s | fake | Migrations up/down, repository behaviour, transaction rollback, event persistence and replay, service orchestration |
| **E2E** | Whole application | < 30 s | fake | The user journeys named in each app's acceptance criteria, through both HTTP and CLI |
| **Regression** | A fixed past defect | fast | any | One test per fixed bug, named for the defect, added with the fix |
| **Failure-path** | Degradation matrix | fast | fake/broken | Every row of [Graceful Degradation](../architecture/graceful-degradation.md) that applies to the component |
| **Migration** | Schema evolution | < 5 s | none | Fresh install, stepwise upgrade, idempotency, failure + restore, data preservation |
| **Performance** | Overhead budgets | seconds | fake | Every budget in [Performance Targets](../architecture/performance-targets.md) that the component owns |
| **Security** | Trust boundaries | fast | none | Path traversal, size limits, auth required/enforced, no secrets in logs, sandbox refusal, no code execution |
| **Compatibility** | Versions | CI matrix | fake | Supported Python versions; SQLite **and** PostgreSQL; minimum and maximum pinned dependency versions. `temporary_postgres` skips when no server is configured, and CI sets `WEIGHTSDB_REQUIRE_POSTGRES=1`, which turns that skip into a failure — a silently skipped dialect is an untested dialect |
| **Live** | Real provider/GPU | minutes | **real** | Smoke coverage that the fakes are still faithful |

---

## 3. Never let a model be the oracle

Real models are nondeterministic and unavailable in CI. They are therefore never the thing a test
asserts against.

* **`FakeProvider` is built before the real provider** in ModelRack, and is a first-class, tested,
  documented component. It produces configurable text, delays, stream chunk patterns, token counts,
  tool calls, malformed payloads, truncated streams, timeouts and errors — deterministically, from a
  seed.
* **Recorded responses** (`respx` transport fixtures) cover real provider wire formats. Every
  recording records the provider name and version it came from and is re-captured when that version
  is bumped.
* **Live tests** are marked `@pytest.mark.live`, deselected by default (`addopts = "-m 'not live and
  not performance'"`), and run manually or in a nightly job on a machine with the hardware. They
  assert *shape and plausibility* (a response arrived, tokens > 0, timings monotonic) — never exact
  content.
* A CI run must pass with no GPU, no Ollama, no network.

---

## 4. Determinism

* No test depends on wall-clock time. Inject a clock; `DTZ` lint plus a `conftest` guard that fails
  any test calling `datetime.now()` without a timezone.
* `pytest-randomly` runs tests in random order every run; order dependence is a defect.
* Any randomness is seeded and the seed is asserted or recorded.
* No test depends on network access. A `conftest` socket guard blocks real connections outside
  `tests/live/`.
* Temporary directories via `tmp_path`. No test writes to the developer's real config or data
  directories — a guard fixture points XDG variables at `tmp_path` for every test.
* Parallel-safe: no shared global state, no fixed ports (bind port 0), no shared database file.

---

## 5. What must be tested, by kind of code

| Code | Non-negotiable coverage |
|---|---|
| Metric/score formula | Known value, boundary values, division-by-zero guard, `UNSUPPORTED` input, empty input |
| Parser (provider JSON, `/proc`, `nvidia-smi`, CSV) | Valid fixture, truncated, malformed, empty, unexpected fields, unit conversion, missing optional fields |
| Scorer | Known-pass, known-fail, boundary, malformed model response, missing data, refusal to score an unsupported capability |
| State machine | Every legal transition, every illegal transition rejected, terminal-state immutability, recovery from each non-terminal state |
| Repository | CRUD, uniqueness, FK cascade, transaction rollback, concurrent access, the index plan for hot queries |
| Wire schema | Round-trip equality, unknown-field preservation, minor forward compatibility, major rejection, invalid values |
| HTTP route | Success, validation error shape, auth required, not-found, size limit, error envelope, request ID propagation |
| CLI command | Success, `--json` output shape, exit code per failure class, missing argument, non-interactive behaviour |
| Streaming | Ordered delivery, disconnect/reconnect with `Last-Event-ID`, replay without gap or duplicate, heartbeat, slow-consumer drop |
| Degradation path | Each applicable row of the degradation matrix |
| Security boundary | Traversal rejected, oversize rejected, unauthenticated rejected, sandbox refusal, no secret in log output |

---

## 6. Coverage

* **85 %** line coverage overall per repository; **95 %** for `domain/` and for shared packages.
* Coverage is a floor, not a goal. A phase whose new code has 100 % coverage and no failure-path
  test has not met this standard.
* Exclusions must be justified inline (`# pragma: no cover — platform-specific, see §cross-platform`).
* `diff-cover` requires 90 % coverage on changed lines in every pull request.

---

## 7. Fixtures and doubles

* **Fakes over mocks.** Prefer a real, simple implementation over `unittest.mock` patching. Mocks
  that assert call shapes couple tests to implementation. The suite ships three, as supported API,
  and they are named here so nobody invents a fourth:

  | Double | Ships in | Replaces |
  |---|---|---|
  | `FakeProvider`, `FakeScript` | `modelrack.testing` | A model provider |
  | `ScriptedHostReader`, `ScriptedGpuReader`, `NullHostReader`, `NullGpuReader` | `sweatmeter.testing` | `/proc`, `/sys` and `nvidia-smi`, driven from fixture text or a scripted sample series |
  | `temporary_sqlite`, `temporary_postgres`, `migration_harness` | `weightsdb.testing` | A database |

  An in-memory repository per application is written by that application.
* `unittest.mock` is appropriate for verifying that a side effect happened (an event was emitted, a
  backup was taken) and for simulating an exception from a boundary.
* Monkeypatching module internals is a smell that a dependency should have been injected.
* Every port has a **conformance test suite** that runs against every implementation — real, fake
  and recorded. A new provider passes the same suite or it is not a provider.

---

## 8. Contract testing across applications

Cross-application compatibility is proven without a shared environment:

1. SetSpec generates JSON Schema and a set of **golden payloads** per schema version, published as
   package data.
2. The producer's test suite asserts that what it emits validates against the schema and matches the
   golden payload structurally.
3. The consumer's test suite asserts that it can read every golden payload for every supported major
   version, and that it rejects the next major with `SCHEMA_VERSION_UNSUPPORTED`.
4. The same technique applies to HTTP, and the artifact needs a distribution channel or the technique
   is theoretical. Each application **ships its committed OpenAPI snapshot as package data**
   (`<app>/api/openapi-v1.json`, loadable through `importlib.resources`) and exposes it as
   `<app>.api_snapshot()`. A consumer's contract tests install the producer's distribution as a
   **test-only** dependency and drive a schema-driven mock from that file. This is not a runtime
   dependency and not an import of the producer's code: `ideapress[dev]` may depend on `loadcoach`,
   while `ideapress` must not, and the import-linter contract that forbids
   `from loadcoach import …` in `src/` is what keeps the two apart. The same mechanism gives
   MirrorWall the two applications' template suites for its cross-consumer render job.
5. A nightly **compatibility matrix** job installs the released versions of each pair and runs the
   contract suites, so a package release that breaks a consumer is caught before the consumer
   upgrades.

---

## 9. Test quality rules

* One behaviour per test; the name states it: `test_route_rejects_candidate_when_vram_insufficient`.
* Arrange–Act–Assert, visually separated.
* Assert on behaviour and values, not on internal call sequences.
* No conditional logic in tests. A loop over table-driven cases is fine (`pytest.mark.parametrize`);
  an `if` deciding what to assert is not.
* Failure messages must identify the case: parametrized IDs, and explicit messages on non-obvious
  assertions.
* Tests are held to the same standards as production code: typed, documented where non-obvious,
  no duplication that a fixture would remove.
* A skipped test states why and under what condition it would run.

---

## 10. CI execution

```bash
# default (every push, every PR) — no GPU, no provider, no network
pytest -m "not live and not performance"

# nightly, on hardware
pytest -m "live"
pytest -m "performance"
```

The default suite must complete in **under 3 minutes** per repository. When it does not, the
slowest tests move to a marked job — the fast suite's job is to be run constantly.

Required CI gates before merge: format, lint, type check, import-linter, unit + contract +
integration + e2e, coverage floor, `diff-cover`, `pip-audit`, `gitleaks`, package build, and a
clean-venv install-and-import check.
