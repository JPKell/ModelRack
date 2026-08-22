# Gold Standards

Quality targets for the suite. Each is measurable, has an owner, and is checked either by an
automated gate or by a named review step. A component is not "done" when it works; it is done when
it meets its gold standards.

---

## 1. Suite-wide

| # | Gold standard | Measure | Gate |
|---|---|---|---|
| G1 | No cross-application imports | `import-linter` contracts pass in every repository | CI, blocking |
| G2 | No shared package imports an application | Same, plus a clean-venv install-and-import check | CI, blocking |
| G3 | Every application starts with zero configuration | `pip install <app> && <app> serve` reaches a healthy state on a clean machine with only Ollama running | E2E test + pre-release manual check |
| G4 | Every application runs standalone | Its full e2e suite passes with every peer application absent | CI, blocking |
| G5 | Unavailable never becomes zero | `Unsupported` refuses arithmetic and truthiness; storage keeps NULL + reason; UI renders `—` | Unit + UI tests |
| G6 | Every **transferable** payload is versioned, and every HTTP API is versioned in its path | Schema-version present on exports, bundles, results, profiles and event frames; unknown major rejected, unknown minor accepted with preservation; API bodies contracted by the committed OpenAPI snapshot. The membership test is [ADR-0025](../adr/0025-envelope-boundaries.md) | Contract tests, both directions |
| G7 | Every public API is documented | OpenAPI snapshot committed; a change without a changelog entry fails | CI, blocking |
| G8 | Deterministic serialization | Canonical JSON: sorted keys, stable float formatting, identical bytes across runs and platforms | Unit tests |
| G9 | Test suite runs without a GPU, a model or a network | Default `pytest` run passes in a sandboxed container | CI, blocking |
| G10 | Fast feedback | Default suite < 3 minutes per repository | CI timing, advisory→blocking at 1.0 |
| G11 | Coverage floors | 85 % overall, 95 % on `domain/` and shared packages, 90 % on changed lines | CI, blocking |
| G12 | No secret ever leaves the process | gitleaks clean; redaction tests pass; no outbound call in the default configuration | CI + a network-isolation e2e test |
| G13 | Safe upgrades | Migration tests: fresh, stepwise, idempotent, failure+restore, both dialects | CI, blocking |
| G14 | Predictable errors | Every error has a stable code, documented in the component spec, asserted by tests | Contract tests |
| G15 | Documented behaviour | Every public function/class/module has a docstring stating its contract | `ruff D1xx`, blocking |
| G16 | Minimal dependency footprint | Runtime dependency budget respected (§2); a new dependency needs a PR justification | Review + a test asserting the declared set |
| G17 | Accessible UI | WCAG 2.1 AA checklist passes; contrast asserted over token pairs in both themes | Pre-release checklist + contrast test |
| G18 | Reproducible releases | Tag → CI → artifacts; no manual upload; install-check passes from the index | Release workflow |
| G19 | Application overhead is measured, not assumed | Every performance budget has a test; regressions > 25 % fail | Nightly performance job |
| G20 | Honest degradation | Every row of the degradation matrix has a test | CI, blocking |
| G21 | Evidence describes the execution it is used for | A routing decision uses capability evidence only when its `runtime_profile_hash` matches the resolved execution profile; a mismatch is named, never silently reused or zeroed | Unit + integration tests ([ADR-0023](../adr/0023-runtime-profile-resolution.md)) |
| G22 | No measurement is attributed to a device it did not come from | No component sums VRAM across GPUs; every VRAM, power and energy figure carries `gpu_index` | Unit tests + multi-GPU fixtures ([ADR-0027](../adr/0027-multi-gpu-semantics.md)) |
| G23 | Local services resist the attack that targets them | `Host` allowlist enforced before routing; CSRF token on form posts; body-supplied URLs pass the fetch allowlist | Security tests ([ADR-0026](../adr/0026-local-http-hardening.md)) |

### 1.1 Runtime dependency budget

| Component | Allowed runtime dependencies | Count |
|---|---|---:|
| BaseAiCore | *(none)* | 0 |
| SetSpec | `pydantic`, `jinja2` (prompt rendering, [ADR-0028](../adr/0028-prompt-pack-granularity.md)) | 2 |
| ModelRack | `httpx` | 1 |
| SweatMeter | *(none)* | 0 |
| WeightsDB | `sqlalchemy`, `alembic` | 2 |
| MirrorWall | `jinja2`, `starlette` | 2 |
| Each application | `fastapi`, `uvicorn`, `typer`, `pydantic-settings`, plus suite packages | ≤ 6 direct non-suite |

Exceeding a budget requires an ADR.

---

## 2. Per component

### BaseAiCore
* Zero third-party runtime dependencies; imports only the standard library.
* 100 % coverage of identity, fingerprint, `Unsupported` and ID/time helpers.
* `mypy --strict` clean; every public symbol documented.
* Identity equality, hashing and canonical-ID generation proven stable across processes and
  Python versions (golden values in tests).
* Installs and is usable in a script with no other suite package present.

### SetSpec
* Every schema publishes JSON Schema and at least three golden payloads per version.
* Round-trip equality for every model; unknown-field preservation proven.
* Unsupported major rejected with the documented error, tested per schema.
* No schema change ships without a version bump (CI-enforced diff against published schemas).
* Capability vocabulary is versioned; additions are minor, removals are major.

### ModelRack
* One provider client for the whole suite; no application contains provider HTTP code.
* Provider conformance suite passes for Ollama (recorded), OpenAI-compatible (recorded) and
  `FakeProvider`.
* Every error path translated to a typed error: unreachable, timeout, model-not-found, protocol
  error, capability-unsupported, context-exceeded.
* Timing: backend-reported and client-observed values stored separately, never merged.
* `token_level_chunks` correctly gates any "per-token latency" claim.
* Streaming is cancellable within one chunk boundary; no leaked connections after cancellation
  (asserted by a connection-count test).

### SweatMeter
* Works with no GPU, no `nvidia-smi`, missing sensors, and multiple GPUs — each an explicit test.
* No sensor failure can raise out of `snapshot()`; a property-based test with a fault-injecting
  reader proves it.
* Machine fingerprint stable across driver upgrades and disk changes; changes when CPU/RAM/GPU set
  changes.
* Sampling overhead ≤ 1 % of one core at 1 s and ≤ 1 % effect on measured throughput.
* Standalone: a five-line script prints CPU/RAM/GPU/VRAM with only this package installed.

### WeightsDB
* Two applications share it while keeping entirely separate schemas and files — proven by an
  integration test that runs two migration histories side by side.
* Defines no application table; a test asserts its `MetaData` is empty.
* Migration failure restores the backup; the original database is byte-identical afterwards.
* SQLite pragmas verified at connect; PostgreSQL path exercised in CI.

### MirrorWall
* A component can be upgraded without either application changing its pages — proven by rendering
  both applications' template suites against a new version in CI.
* Contains no application page, no navigation tree, no domain term.
* Accessibility: every shipped component passes keyboard and ARIA tests.
* Contrast validated for every token pair in both themes.
* SSE helper guarantees gap-free sequences, replay without duplication, and bounded subscriber
  queues under a slow-consumer test.

### FreeWeight
* **Reproducible benchmarks**: two runs of the same subject with the same fingerprint produce
  metrics within the documented tolerance, and the fingerprint document explains any difference.
* Raw samples are always retained; every headline metric drills to its samples in ≤ 2 interactions.
* No cold and warm measurements are ever mixed in one headline number (asserted by aggregation tests).
* Every scoring formula has a unit test with known values and boundary cases.
* Deterministic scoring is preferred over LLM judging wherever the task allows it; the scorer ladder
  is documented per benchmark.
* Evidence export is consumable by LoadCoach without any FreeWeight code or database access —
  proven by a contract test using only the exported file.
* Evidence freshness reflects when the measurement was taken, not when it was aggregated —
  a test recomputes evidence over old runs and asserts the confidence does not rise.
* A benchmark run survives a browser refresh and a server restart with its progress intact.

### LoadCoach
* **Every routing decision is explainable**: candidates, hard-constraint rejections with reasons,
  per-capability contributions, confidence, availability and the final ordering are persisted and
  retrievable for every job — 100 % of decisions, not a sample.
* Routes correctly with no benchmark evidence at all, and says so in the explanation.
* Queue survives restart with no lost, duplicated or stuck job; recovery is idempotent for
  idempotent work and explicit for the rest.
* No starvation: a low-priority job's wait time is bounded by the configured ageing policy, proven by
  a scheduling simulation test.
* Cancellation is honoured within one chunk boundary; a cancelled job never leaves a resident model
  or an open connection.
* Fallback ordering is deterministic and recorded; a fallback is never silent.
* A model is never admitted on a context it will not be served: `min_context_tokens` is evaluated
  against the served context, and an assumed context is flagged on the decision.
* Ageing is a running mechanism, not a startup one — the simulator advances a fake clock with the
  process alive and asserts the starvation bound.

### IdeaPress
* Runs a complete workflow with **no** LoadCoach and no FreeWeight installed.
* Switching the inference backend (Ollama ↔ LoadCoach ↔ OpenAI-compatible) changes configuration
  only — a test runs the identical workflow against all three backends and compares the produced
  structure.
* Python owns control flow: no model output can decide that a workflow may terminate, and a test
  proves a "everything is fine, stop" response does not end a gated stage.
* Every stage output passes deterministic validation before it is committed.
* A project survives an interrupted stage: committed units intact, the failed stage resumable.
* Every generated artifact records the model, prompt version and validation results that produced it.

---

## 3. Documentation gold standards

| Standard | Measure |
|---|---|
| An implementation agent can build a phase without inventing architecture | Every phase names its files, tests, acceptance criteria and deferred work |
| No contradictions between documents | Consistency review checklist executed before each documentation release |
| Every architectural decision has a rationale | Each ADR has Context, Decision, Alternatives, Consequences, Status |
| Every component has one owner | Traceability matrix has no duplicated ownership and no gaps |
| Docs stay true | Configuration reference and OpenAPI snapshots are generated and diff-checked in CI |

---

## 4. Release gold standards (suite 1.0)

- [ ] All three applications install from PyPI into a clean venv and start with zero configuration.
- [ ] All six packages install standalone and pass their own test suites.
- [ ] Compatibility matrix green across the declared version ranges.
- [ ] Every gate in §1 is enforced in CI in every repository.
- [ ] Every component meets its §2 standards.
- [ ] Migration path documented and tested from every previously released version.
- [ ] Security checklist complete; no known vulnerable dependency.
- [ ] Accessibility checklist complete for all three UIs.
- [ ] README, configuration reference, API docs, troubleshooting guide and backup/restore procedure
      published per component.
- [ ] Performance budgets measured on the reference machine and published with the machine described.
