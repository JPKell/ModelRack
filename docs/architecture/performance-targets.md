# Performance Targets

**Principle:** measure application overhead, never promise model performance.

The suite cannot make a model faster. What it can promise is that the software wrapped around the
model adds a small, bounded, *measured* cost. Every number here is an application-overhead budget,
verified by a performance test, on the reference machine.

---

## 1. Reference machine

Targets are stated against the development machine recorded at architecture freeze
([inventory §7](../inventory/legacy-material-inventory.md)):

```text
Ubuntu 26.04 LTS · Python 3.13 · NVIDIA RTX 5060 Ti (16 GB) · Ollama 0.32.13 · local SQLite (WAL)
```

Targets are ratios and absolute ceilings for *software overhead*, not for inference. On slower
hardware the ratios hold; the absolute ceilings scale. Any published number states the machine it
came from.

---

## 2. The separation that matters

```text
total_request_time  =  application_overhead  +  provider_time
```

* `provider_time` is what ModelRack observes around the provider call.
* `application_overhead` is everything else: routing, validation, persistence, serialization, HTTP.
* Every application exposes both, per request, in its API response metadata and in its logs. A
  benchmark or job whose overhead exceeds its budget is a bug with a number attached.

---

## 3. Budgets

### 3.1 HTTP and API

| Measure | Target | Ceiling | How verified |
|---|---|---|---|
| JSON GET, in-memory data (health, version, config) | p50 ≤ 5 ms | p99 ≤ 25 ms | `tests/performance/test_api_overhead.py` against the ASGI app |
| JSON GET, single-row DB read | p50 ≤ 10 ms | p99 ≤ 50 ms | Same, seeded DB |
| JSON list, 50 rows + pagination | p50 ≤ 25 ms | p99 ≤ 100 ms | Seeded 100 k-row table |
| Non-streaming generate: overhead **excluding** provider time | ≤ 15 ms | ≤ 50 ms | Fake provider with fixed latency; overhead = total − provider |
| HTML page render (server-side, warm templates) | p50 ≤ 30 ms | p99 ≤ 120 ms | Template render benchmark |
| Static asset (cached, ETag hit) | ≤ 2 ms | ≤ 10 ms | — |

### 3.2 Streaming

| Measure | Target | Ceiling |
|---|---|---|
| Added latency per streamed chunk (provider → client) | ≤ 5 ms | ≤ 20 ms |
| Time from provider's first token to client's first SSE frame | ≤ 20 ms | ≤ 60 ms |
| SSE heartbeat interval | 15 s ± 1 s | — |
| Concurrent idle SSE connections held per process | ≥ 200 | — |
| Memory per idle SSE connection | ≤ 64 KiB | ≤ 256 KiB |
| Event replay from `Last-Event-ID` (1 000 events) | ≤ 100 ms | ≤ 400 ms |

TTFT reported to users is always the **provider's** TTFT plus the measured suite overhead, with the
two shown separately.

### 3.3 Queue and scheduling (LoadCoach)

| Measure | Target | Ceiling |
|---|---|---|
| Enqueue (HTTP accepted → row committed) | ≤ 15 ms | ≤ 50 ms |
| Dispatch latency (job eligible → execution starts), idle worker | ≤ 100 ms | ≤ 500 ms |
| Routing decision (20 candidates, evidence cached) | ≤ 20 ms | ≤ 100 ms |
| Routing decision (cold evidence cache) | ≤ 150 ms | ≤ 500 ms |
| Queue poll overhead at idle | ≤ 0.5 % of one core | ≤ 2 % |
| Cancellation acknowledged (queued job) | ≤ 50 ms | ≤ 200 ms |
| Cancellation acknowledged (executing job, at the next stream boundary) | ≤ 1 s | ≤ 5 s |
| Recovery of 1 000 in-flight jobs after restart | ≤ 2 s | ≤ 10 s |

### 3.4 Telemetry (SweatMeter)

| Measure | Target | Ceiling |
|---|---|---|
| Snapshot without GPU (`/proc` + `/sys` only) | ≤ 3 ms | ≤ 10 ms |
| Snapshot with `nvidia-smi` (single GPU) | ≤ 40 ms | ≤ 120 ms |
| Sampler CPU cost at 1 s interval | ≤ 0.5 % of one core | ≤ 1.5 % |
| Persisted sample write (batched) | ≤ 2 ms/sample amortized | ≤ 10 ms |
| Effect of sampling on measured benchmark throughput | ≤ 1 % | ≤ 2 % (measured and reported per run) |

The last row is the important one: telemetry must not distort the measurement it is documenting.
FreeWeight runs a calibration test (sampling on vs off, identical workload) and records the delta.

### 3.5 Database

| Measure | Target | Ceiling |
|---|---|---|
| Single-row insert (sample) | ≤ 1 ms | ≤ 5 ms |
| Batched sample insert (100 rows, one transaction) | ≤ 15 ms | ≤ 60 ms |
| Dashboard aggregate over 100 k samples | ≤ 200 ms | ≤ 1 s |
| Run detail page query set | ≤ 100 ms | ≤ 400 ms |
| Migration on a 1 GB SQLite DB | ≤ 60 s | ≤ 300 s (with progress output) |
| Backup of a 1 GB SQLite DB | ≤ 30 s | ≤ 120 s |

Every query in a page's critical path has an index and an `EXPLAIN QUERY PLAN` assertion in tests
for the shapes that matter (no full scan on `samples`, `jobs`, `telemetry_samples`).

### 3.6 Startup and discovery

| Measure | Target | Ceiling |
|---|---|---|
| `python -m <app>` → serving (warm page cache, existing DB) | ≤ 1.5 s | ≤ 3 s |
| CLI `--help` | ≤ 250 ms | ≤ 600 ms |
| CLI simple command (health, version) end to end | ≤ 500 ms | ≤ 1.5 s |
| Model discovery, 20 models, metadata cached | ≤ 200 ms | ≤ 1 s |
| Model discovery, 20 models, cold (provider `show` per model) | ≤ 3 s | ≤ 10 s |
| First-run database creation + migration | ≤ 2 s | ≤ 5 s |

CLI startup is protected by lazy imports: `--help` must not import SQLAlchemy, FastAPI or httpx. A
test asserts the imported module set for `--help`.

### 3.7 UI responsiveness

| Measure | Target | Ceiling |
|---|---|---|
| First contentful paint, local, warm | ≤ 400 ms | ≤ 1 s |
| Interaction to visual feedback (sort, filter, tab) | ≤ 100 ms | ≤ 250 ms |
| Telemetry bar update cadence | 1 s (configurable 0.25–5 s) | — |
| Telemetry bar update: no layout shift | CLS 0 | — |
| Table with 1 000 rows × 20 columns: sort | ≤ 150 ms | ≤ 500 ms |
| Chart re-theme on light/dark switch | ≤ 200 ms | ≤ 500 ms |
| Total JS shipped per page (uncompressed, excl. charting vendor) | ≤ 60 KB | ≤ 120 KB |
| Charting vendor bundle (vendored, cached) | ≤ 1 MB | — |

### 3.8 Benchmark execution overhead (FreeWeight)

| Measure | Target | Ceiling |
|---|---|---|
| Per-sample overhead outside the provider call (scoring, persistence, events) | ≤ 10 ms | ≤ 50 ms |
| Per-sample overhead as a share of a 2 s inference | ≤ 0.5 % | ≤ 2.5 % |
| Run start (validate → persist → first provider call) | ≤ 500 ms | ≤ 2 s |
| Aggregation for a 10 000-sample run | ≤ 5 s | ≤ 20 s |
| Export of a 10 000-sample run to JSON | ≤ 10 s | ≤ 30 s |

---

## 4. Memory budgets

| Component | Idle RSS | Under load | Ceiling |
|---|---|---|---|
| FreeWeight server | ≤ 120 MB | ≤ 400 MB during a run | 800 MB |
| LoadCoach server | ≤ 120 MB | ≤ 500 MB with 4 concurrent streams | 1 GB |
| IdeaPress server | ≤ 120 MB | ≤ 600 MB with a long project loaded | 1 GB |
| CLI (non-server command) | ≤ 80 MB | — | 200 MB |

Streaming never accumulates a full response in more than one place; long documents are written to
artifacts rather than held in memory. A test drives a 1 M-token synthetic stream and asserts a flat
memory profile.

---

## 5. What is deliberately not promised

* Tokens per second, TTFT, or quality for any model — those are measurements FreeWeight *produces*.
* Any target on hardware other than the reference machine.
* Any target with a remote provider in the path (network dominates).
* Scaling beyond one machine and a handful of concurrent users.
* Performance during a benchmark run for *other* work on the same GPU — the suite explicitly
  serializes GPU work instead.

---

## 6. Verification

* Each repository has `tests/performance/`, marked `@pytest.mark.performance`, excluded from the
  default run and executed in a dedicated CI job on a fixed runner.
* Assertions are on **medians over N iterations with a documented tolerance**, never on a single
  sample, and never on wall-clock in a shared CI container where the ceiling would be noise. Where
  CI cannot be trusted for absolute timing, the test asserts the *ratio* (overhead ÷ fake-provider
  latency) instead.
* Every phase in every development plan that adds a path listed above also adds or extends its
  performance test; the phase's acceptance criteria name the budget.
* Regressions greater than 25 % against the recorded baseline fail the performance job.
