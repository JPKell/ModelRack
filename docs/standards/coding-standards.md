# Coding Standards

**Applies to:** every repository in the suite.
**Priority order when they conflict:** correctness → clarity → consistency → concision → cleverness (last, and usually wrong).

Human readability outranks elegance. Code in this suite is read by future agents and by a
maintainer who has forgotten the context; it must state its intent rather than imply it.

---

## 1. Docstring-first development

Behaviour is defined before it is implemented. The workflow is mandatory for every public
function, method and class:

```text
1. define the behaviour            what it does, what it promises, what it refuses
2. write the docstring             the contract, in prose, before any logic exists
3. write the tests                 against the docstring, where the contract is testable
4. implement                       to satisfy the documented contract
```

The purpose is explicit: implementation follows stated intent. Nobody — human or agent — should
ever have to reverse-engineer meaning out of a function body.

### 1.1 What a docstring must contain

**Every public function/method:**

```python
def aggregate_capability_evidence(
    results: Sequence[BenchmarkResult],
    *,
    now: datetime,
    policy: ConfidencePolicy,
) -> list[CapabilityEvidence]:
    """Roll benchmark results up into per-capability evidence with confidence.

    Groups ``results`` by (capability, measurement subject), normalizes each metric to
    the capability's 0.0–1.0 scale, and combines them into one score per group. Confidence
    is derived from sample count, result dispersion, benchmark age and environment drift
    as defined in ADR-0017; it is never inferred from the score itself.

    Args:
        results: Completed results only. Interrupted or failed results must be filtered
            out by the caller — this function does not know why a result is incomplete.
        now: Timezone-aware reference time for age decay. Injected so tests are deterministic.
        policy: Decay half-lives and floors. See ``ConfidencePolicy`` defaults.

    Returns:
        One entry per (capability, subject) that had at least one scored sample, sorted by
        capability ID then subject. Capabilities with no results are omitted rather than
        returned with a zero score — absence of evidence is not evidence of incapacity.

    Raises:
        ValueError: If ``now`` is naive, or a result carries a capability ID outside the
            vocabulary version this build supports.
    """
```

Required sections: a one-line summary in the imperative mood; a paragraph of behaviour when the
summary is not self-evident; `Args` for anything non-obvious; `Returns` describing meaning, not
type; `Raises` for every exception the caller is expected to handle. Google style throughout,
checked by `ruff`'s pydocstyle rules.

**Every public class:** what it represents, its invariants, its thread-safety, its lifecycle, and
what it deliberately does not do.

**Every module:** a module docstring stating its role in its layer and any rule that constrains it
(for example: "Domain module — imports no framework").

### 1.2 Document the "why", especially for constraints

The prior implementation did this well and it is adopted as a standard: when a decision exists to
prevent a specific failure, say so.

```python
class Unsupported:
    """Sentinel for a measurement this environment cannot provide.

    Boolean and arithmetic coercion raise on purpose. The failure this guards against is
    ``value or 0`` / ``value + x`` silently converting "not measurable" into a real-looking
    number — the single most damaging class of bug in a benchmarking system (ADR-0016).
    """
```

Reference the ADR or requirement by identifier where one exists.

### 1.3 Private helpers

Private functions (`_name`) need a one-line docstring stating purpose. If a private function needs
three paragraphs, it is public API in disguise or it does too much.

---

## 2. Typing

* **Type hints are mandatory** on every public function, method, class attribute and module
  constant. Private helpers should be typed; anything crossing a module boundary must be.
* `mypy --strict` on every shared package. Applications run `mypy` in strict mode with a documented,
  time-boxed allowlist for third-party gaps only (`ignore_missing_imports` per module, never
  blanket).
* No bare `Any` at a public boundary. `Any` inside a provider's `raw` payload is fine; `Any` in a
  function signature needs a comment justifying it.
* Prefer precise types: `Sequence` over `list` for parameters, `Mapping` over `dict`, concrete
  types for returns.
* Use `Protocol` for ports. Use `ABC` only when shared implementation genuinely exists.
* Domain enums are `StrEnum` so they serialize and log readably.
* `from __future__ import annotations` at the top of every module (uniformity across 3.12–3.14).
* Never `# type: ignore` without a trailing comment naming the reason.

```python
# Good
def list_models(self, *, refresh: bool = False) -> tuple[ModelDescriptor, ...]: ...

# Bad
def list_models(self, refresh=False): ...
```

---

## 3. Naming

* Names say what a thing *is* or *does*, not how it is built: `benchmark_results`, not `br_list`.
* Functions are verb phrases (`resolve_model`, `aggregate_evidence`); predicates start with
  `is_`/`has_`/`can_`; classes are noun phrases.
* No abbreviations except a documented short list: `id`, `db`, `url`, `api`, `cpu`, `gpu`, `ram`,
  `vram`, `ttft`, `sse`, `json`, `http`, `utc`.
* Units belong in the name: `duration_ms`, `vram_used_bytes`, `interval_seconds`,
  `temperature_c`, `power_w`, `size_bytes`. A number without a unit in its name is a defect.
* Booleans read as assertions: `is_streaming`, `has_digest`, `allow_remote_providers`.
* Same concept, same name, everywhere in the suite: `model_identity`, `machine_fingerprint`,
  `runtime_profile_hash`, `capability_id`, `task_profile_id`, `run_id`, `job_id`, `request_id`.

---

## 4. Functions and classes

* One purpose per function. If the docstring needs "and", split it.
* Soft limits (a reviewer's prompt to look harder, not a hard gate): 40 lines per function,
  6 parameters, cyclomatic complexity 10 (`ruff C901` at 10), 400 lines per module.
* Keyword-only arguments for anything optional or boolean: `def run(*, dry_run: bool = False)`.
  Positional booleans are banned.
* Prefer pure functions in `domain/`. I/O lives in `infrastructure/`, orchestration in `services/`.
* Classes hold state or implement a port. A class with one method and no state is a function.
* Value objects are `@dataclass(frozen=True, slots=True)`. Wire models are pydantic. ORM models are
  SQLAlchemy `DeclarativeBase` and never leave the repository layer.
* No mutable default arguments, ever.

---

## 5. Dependencies and injection

* Explicit dependencies. A function's inputs are its parameters, not module globals.
* Inject at external boundaries: providers, telemetry readers, clocks, filesystem roots, event
  sinks, HTTP clients, random sources.

```python
class RunScheduler:
    def __init__(
        self,
        *,
        provider: Provider,                 # Protocol — real, fake, or recorded in tests
        telemetry: TelemetrySource,
        runs: RunRepository,
        events: EventSink,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
```

* Every application has one composition root (`bootstrap.py`) where concretions are built. Nothing
  else calls a constructor for infrastructure.
* Time: never call `datetime.now()` in domain or service code — take a clock. Durations use
  `time.perf_counter_ns()`; timestamps use timezone-aware `datetime` in UTC. The two are never
  interchanged.
* Randomness: injected `random.Random` instance with a recorded seed wherever results are compared.

---

## 6. Errors

* One base exception per package/application, with a stable machine-readable `code`:

```python
class SuiteError(Exception):
    """Base for every suite error. ``code`` is stable and appears in API error envelopes."""
    code: ClassVar[str] = "INTERNAL_ERROR"
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None: ...
```

* Error `code` values are part of the public contract: adding one is a minor change, changing the
  meaning of one is a major change. They are listed in each component's spec.
* **Never** swallow broadly. `except Exception: pass` is banned (`ruff BLE001`, `S110`). The one
  sanctioned pattern is a *narrow* catch around a single optional sensor read that degrades one
  field to `UNSUPPORTED` and logs at DEBUG — the `_safe()` helper in SweatMeter, which exists in
  exactly one place and is tested.
* Catch narrowly, wrap with context, re-raise as a domain error at layer boundaries. Never lose the
  cause: `raise DomainError(...) from exc`.
* No magic return values. Not `-1` for "missing", not `0` for "unknown", not `{}` for "failed".
  Return a typed result, raise, or return `UNSUPPORTED`.
* Exception messages state what failed, what was expected and what the caller can do. No bare
  `raise ValueError("invalid")`.

---

## 7. State and purity

* No module-level mutable state. Configuration, caches and registries are objects with owners.
* Singletons only for genuinely process-wide immutables (the `UNSUPPORTED` sentinel; the logger).
* Caches are explicit objects with a documented invalidation rule and a `clear()` for tests.
  `functools.lru_cache` on a function that touches I/O is banned.
* Determinism where practical: stable sort orders, canonical JSON (sorted keys) for anything hashed
  or compared, seeded randomness, no dependence on `dict` insertion order across processes.

---

## 8. Comments

* Comments explain **why**. The code already states what.
* Every non-obvious constant carries its provenance: `# Ollama's default keep-alive is 5m; we hold
  longer during a run so cold-load timings are not polluted by an eviction.`
* `TODO(owner): text` only with an owner and an issue reference. A `TODO` with neither is deleted.
* Commented-out code is deleted. Git remembers.

---

## 9. Security-adjacent rules

Full detail in [Security Standards](security-standards.md); the coding-level rules:

* Never `eval`, `exec`, `pickle.loads`, `yaml.load`, or `subprocess(shell=True)` on any input
  derived from a model, a request or a file the user did not explicitly hand us.
* Model-generated content is data. It is never executed, never used to build a path, never
  interpolated into SQL, and never rendered without escaping.
* All filesystem writes go through a `contained_path(root, *parts)` helper that proves containment
  after `resolve()`.
* Parameterized SQL only. String-built SQL is banned outside migration DDL.
* Secrets never enter logs, error messages, exports, telemetry or exception `details`.

---

## 10. Logging

Full detail in [Observability Standards](observability-standards.md). At the code level:

* One logger per module: `logger = logging.getLogger(__name__)`.
* Structured extras, never f-string interpolation of variable data into the message:

```python
logger.info("run.completed", extra={"run_id": run.id, "duration_ms": elapsed, "samples": n})
```

* Levels: DEBUG (developer detail), INFO (state transitions a user would care about), WARNING
  (degradation), ERROR (operation failed), CRITICAL (process cannot continue).
* Never log prompts or generated content at INFO or above; full-content logging is an explicit
  opt-in configuration flag, off by default.

---

## 11. Tooling

Identical configuration in every repository, kept in `pyproject.toml`.

| Concern | Tool | Configuration |
|---|---|---|
| Format | **ruff format** | line length 100, double quotes, magic trailing comma |
| Lint | **ruff** | `E,W,F,I,N,D,UP,ANN,B,A,C4,DTZ,T20,SIM,TID,PTH,ERA,PL,RUF,S,BLE,ARG,C901` |
| Types | **mypy** | `strict = true` (packages); applications strict with per-module third-party exemptions |
| Imports | **ruff isort** (`I`) + **import-linter** | layering contracts per [boundary rules](../architecture/dependency-and-boundary-rules.md) |
| Tests | **pytest** | `pytest-cov`, `pytest-randomly`, `respx` (HTTP), `freezegun` or an injected clock |
| Coverage | **coverage.py** | fail-under 85 % overall; 95 % on `domain/` and on shared packages |
| Security | **bandit** (via ruff `S`), **pip-audit**, **gitleaks** | CI-blocking |
| Docstrings | ruff `D` (Google convention) | `D1xx` enforced on public API |
| Dead code | **vulture** (advisory) | reported, not blocking |

Key enforced rules worth naming: `DTZ` (timezone-aware datetimes), `T20` (no `print` outside CLI
rendering), `PTH` (use `pathlib`), `TID252` (no relative imports beyond one level), `ERA`
(no commented-out code), `BLE001` (no bare `except Exception`), `ANN` (annotations required).

Pre-commit runs format, lint, type check and the fast unit subset. CI re-runs everything; local
hooks are a convenience, never the gate.

---

## 12. Python version

`requires-python = ">=3.12"`. CI matrix: 3.12 and 3.13 are **supported**; 3.14 runs as an
early-warning job and becomes supported once every dependency publishes wheels for it
([ADR-0019](../adr/0019-python-baseline-and-config-format.md)). No feature newer than 3.12 is used
in library code unless guarded.

---

## 13. Anti-patterns (rejected, with the evidence)

| Anti-pattern | Why it is banned | Where it bit us |
|---|---|---|
| Business logic in a route handler or CLI command | Forces duplication for the second interface | Two prior projects |
| Hand-written `to_dict()` per dataclass | ~200 lines of drift-prone boilerplate, no schema | `openweight_bench` |
| `field: Any = UNSUPPORTED` | Erases typing exactly where it matters | `openweight_bench` |
| `except Exception: pass` | Hides provider failures | `content_factory` |
| Prompts in Python source | Unversioned, uncomparable, unreviewable | `content_factory`, `lm_ai_dev` |
| God modules (>1 000 lines) | No seam, no reuse, no tests | 2 103-line orchestrator |
| Zero-instead-of-unsupported | Fabricates measurements | The bug class the whole suite is designed against |
| Speculative abstraction | Interfaces with one implementation and no second in sight | — |
| Provider JSON above ModelRack | Couples applications to Ollama's wire format | Three separate Ollama clients |
