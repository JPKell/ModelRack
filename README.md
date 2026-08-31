# ModelRack

The suite's only model client: a provider-neutral abstraction over local inference runtimes (Ollama first), with a deterministic FakeProvider.

**Status:** `0.6.0` — **Phases 1–5 complete; the package is feature-complete against its
[specification](docs/packages/modelrack/spec.md).** The provider-neutral vocabulary, the
streamed-event union and the `Provider` protocol exist and type-check; a deterministic, scriptable
`FakeProvider` ships in `modelrack.testing`; two real adapters — `OllamaProvider` and
`OpenAICompatibleProvider` — talk to a real Ollama server and an OpenAI-compatible one (llama.cpp
server, LM Studio, …) over HTTP, both proven against the same conformance suite the fake proves
itself against; and Phase 5 adds the operational surface LoadCoach depends on — residency with
capability gating, hardened cancellation, an explicit metadata cache, and an optional `on_event`
observability hook. See the [development plan](docs/packages/modelrack/development-plan.md) for
what each phase adds, and the [quickstart](docs/quickstart.md) to run something in five minutes.

Part of the **Local AI Suite**.

## Install

```bash
pip install modelrack
```

## Quickstart

A fuller tour lives in [docs/quickstart.md](docs/quickstart.md). The shape of it: application code
takes a `Provider` and never names one:

```python
from baseaicore import ModelIdentity
from modelrack import GenerationRequest, Message, Provider, Role, SamplingParameters


def summarize(provider: Provider, identity: ModelIdentity, text: str) -> str:
    request = GenerationRequest(
        identity=identity,
        messages=(Message(role=Role.USER, content=f"Summarize: {text}"),),
        sampling=SamplingParameters(temperature=0.0, seed=42),
    )
    return provider.generate(request).text
```

Its tests supply the fake, which needs no GPU, no model and no network:

```python
from modelrack.testing import FakeProvider, FakeScript

provider = FakeProvider(FakeScript(), seed=42)
print(summarize(provider, provider.resolve("fake-model"), "a long document"))
```

## Testing against the fake

`FakeProvider` is shipped API, not a test helper — every default test suite in the suite runs
against it, so it is deterministic, honest about what it cannot do, and scriptable into the cases
that actually break callers. A `FakeScript` says what the provider serves and what successive
calls do:

```python
from modelrack.testing import (
    FakeFailure,
    FakeFailureMode,
    FakeGeneration,
    FakeProvider,
    FakeScript,
    FakeToolCall,
)

script = FakeScript(
    generations=(
        # A slow first token, then steady output — reported in `Timing`, but costing no wall
        # time unless you inject `sleep=time.sleep`.
        FakeGeneration(word_count=40, first_chunk_delay_ms=900, chunk_delay_ms=8),
        # A tool call whose arguments are not valid JSON, which real models do emit.
        FakeGeneration(
            word_count=0,
            tool_calls=(FakeToolCall(name="get_weather", raw_arguments='{"city": "Berlin"'),),
        ),
        # A stream that stops without its terminal chunk, four deltas in.
        FakeGeneration(
            failure=FakeFailure(mode=FakeFailureMode.TRUNCATED_STREAM, after_chunks=4),
        ),
    ),
)
provider = FakeProvider(script, seed=42)
```

Given the same script and seed, it produces byte-identical text, chunking, token counts and
tool-call identifiers in another process, on another platform and under another `PYTHONHASHSEED`.

Two ready-made declarations bracket the range a caller has to survive: `FULL_CAPABILITIES` (the
default) and `MINIMAL_CAPABILITIES`, where every flag is `False` and every optional feature is
refused with `CapabilityUnsupported`. Testing against only the first is testing against the easy
half:

```python
from modelrack.testing import MINIMAL_CAPABILITIES

weak = FakeProvider(FakeScript(capabilities=MINIMAL_CAPABILITIES))
weak.capabilities().streaming  # False — and stream() raises rather than silently degrading
```

The script refuses to describe a provider the fake could only imitate by lying: reasoning content
on a provider that declares it reports none, token counts on one that declares it counts nothing,
or `token_level_chunks` alongside deltas that are not one token each — each is a `ValidationError`
at construction rather than a wrong number in a downstream benchmark. And every adapter, fake or
real, passes one conformance suite:
[`tests/contract/test_conformance.py`](tests/contract/test_conformance.py).

Two rules run through every type here. An unavailable measurement is `UNSUPPORTED`, never `0`
(ADR-0016) — so a provider that reported no token
counts yields a result that says so rather than one that averages away real throughput. And what a
provider *reported* about its own work is never merged with what this process *observed*:

```python
from modelrack import Timing

timing = Timing(backend_decode_ms=300, client_wall_ms=412)
timing.backend_decode_ms  # what Ollama said it spent decoding
timing.client_wall_ms  # what this process measured end to end
```

There is deliberately no combined duration field. The two disagree for real reasons — queueing,
transport, scheduling — and a benchmark comparing one runtime's self-report against another's wall
clock is comparing nothing.

Capabilities are checked, never assumed
(ADR-0007 rule 2):

```python
def stream_if_possible(provider: Provider) -> bool:
    capabilities = provider.capabilities()
    # `token_level_chunks` gates any per-token latency claim: when it is False, the gap between
    # two deltas is inter-chunk latency and must not be relabelled.
    return capabilities.streaming and capabilities.token_level_chunks
```

See [docs/packages/modelrack/spec.md](docs/packages/modelrack/spec.md) §20 for a runnable example.

## Talking to a real Ollama

`OllamaProvider` implements the same `Provider` protocol over Ollama's HTTP API — swap it in where
`FakeProvider` stood in a test, and application code does not change:

```python
from modelrack.providers.ollama import OllamaProvider

provider = OllamaProvider(base_url="http://127.0.0.1:11434")  # the default, if omitted
identity = provider.resolve("qwen3.5:9b-q8_0")
print(summarize(provider, identity, "a long document"))
```

Imported from `modelrack.providers.ollama`, not from `modelrack` itself — this is the one module
in the package that imports `httpx`, and a process that only ever talks to the fake has no reason
to pay for that import. Two things this adapter is built around, both load-bearing:

* **NDJSON streaming survives a chunk boundary landing anywhere.** A streamed response is one JSON
  object per line, and neither a line break nor a multi-byte character inside one line is
  guaranteed to arrive in a single TCP read. Reassembly is `httpx`'s own incremental UTF-8 decoder
  (`Response.iter_lines()`), not a hand-rolled buffer — verified directly against this `httpx`
  version with a character split deliberately across two raw chunks.
* **Backend and client timings are read from two different places, never merged.** Ollama's
  `load_duration`, `prompt_eval_duration`, `eval_duration` and `total_duration` (nanoseconds,
  converted once) become `Timing.backend_*`; this process's own `client_wall_ms` and
  `client_ttft_ms` come from `baseaicore.monotonic_ns()` measured from outside the call.

Every unit test for this adapter runs against a recorded transport
(`tests/fixtures/providers/ollama/`, version-annotated in that directory's `manifest.json`) — the
default suite needs no Ollama installed. `tests/live/test_ollama_live.py` is the marked exception:
run `pytest -m live` against a real server to prove the fixtures are still faithful; it skips
gracefully when none is reachable (`MODELRACK_REQUIRE_OLLAMA=1` turns that skip into a failure, the
same escape hatch WeightsDB gives its own conditionally-skipped dialect tests).

## Talking to an OpenAI-compatible server

`OpenAICompatibleProvider` speaks the same `Provider` protocol over a local llama.cpp server, LM
Studio, or anything else exposing `/v1/models` and `/v1/chat/completions`:

```python
from modelrack.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(base_url="http://127.0.0.1:8080", api_key=None)
identity = provider.resolve("qwen3.5-9b-instruct-q8_0")
print(summarize(provider, identity, "a long document"))
```

Its `capabilities()` is honestly different from Ollama's, not merely a subset asserted the same
way — the full comparison is [docs/providers.md](docs/providers.md), generated from the adapters'
own declarations so it cannot drift away from them: no digest anywhere in `/v1/models` (every identity is `NAME_ONLY`), no residency-control
endpoint (`load`, `unload` and `list_resident` all refuse with `CapabilityUnsupported`), and no
per-request field to set a served context length (`context_configurable` is `False`, refused
before a request is sent rather than silently ignored). Fixtures live under
`tests/fixtures/providers/openai_compatible/`, representative of
llama.cpp server and LM Studio; there is no live-server suite for this adapter yet.

## Residency, cancellation, caching and events

The four operational features Phase 5 adds. Each is a promise a scheduler can build on rather than
a convenience.

**Residency is a branch, not a `try`.** LoadCoach asks what it may do before it does it, and a
provider that cannot manage residency refuses with `CapabilityUnsupported` naming the flag —
never a silent no-op that would leave a scheduler believing it had evicted something:

```python
from baseaicore import RuntimeProfile
from modelrack import find_resident, residency_support

support = residency_support(provider.capabilities())
if support.is_manageable:
    loaded = provider.load(identity, RuntimeProfile())
    loaded.already_resident  # a warm model measured as a cold start is an order of magnitude out
    entry = find_resident(provider.list_resident(), identity)
    provider.unload(identity)  # False when it was not loaded — the state you wanted, not a failure
```

**Cancellation stops within one chunk, preserves what was generated, and leaks nothing.** Every
exit path — drained, cancelled, abandoned, failed mid-flight, refused before it began — releases
the response body, asserted by a connection-counting transport in
[`tests/unit/test_cancellation.py`](tests/unit/test_cancellation.py) across a hundred sequential
streams. A token already set before `stream()` is called opens no connection at all.

**Metadata is cached; a generation never is.** Discovery costs one `/api/tags` plus one
`/api/show` per model, which is why spec §15 budgets a cold twenty-model listing in seconds and a
warm one in ten milliseconds. A generation is not a fact about anything — two identical requests
are two runs — so nothing puts a `GenerationResult` in the cache, and a test asserts it. Residency
and health are never cached either: both are live state whose stale answer is worse than no answer.

```python
provider = OllamaProvider(metadata_ttl_seconds=300)  # spec §10's default; 0 disables it
provider.list_models()  # cold
provider.list_models()  # warm
provider.list_models(refresh=True)  # a tag was just re-pulled: read it again now
provider.metadata_cache_stats()  # hits, misses, expirations, stores, entries
provider.clear_metadata_cache()
```

A cached descriptor keeps the instant the provider actually answered, so `observed_at` never claims
a freshness the data does not have.

**Events carry no content.** `on_event` reports requests starting, chunking, completing and
failing, so an application can emit its own structured logs without ModelRack knowing what a run
is. A `ProviderEvent` has no field a prompt, a generated token, a tool argument or an API key could
reach — that is the enforcement, not a convention — and it passes through the caller's own
`metadata` correlation identifiers, which are never sent to the provider:

```python
from modelrack import ProviderEvent, ProviderEventKind


def log(event: ProviderEvent) -> None:
    if event.kind is ProviderEventKind.REQUEST_COMPLETED:
        emit_metric(event.operation, event.elapsed_ms, run_id=event.metadata["run_id"])


provider = OllamaProvider(on_event=log)
```

A callback that raises is logged at DEBUG and does not disturb the generation — a completed result
destroyed by a bug in a metrics hook would be a far worse outcome than a missing log line.

## Documentation

Project documentation lives under [`docs/`](docs/README.md). Start with [`docs/README.md`](docs/README.md).

| Read this | For |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | Getting something running, then everything the client can do, in ten short sections |
| [docs/providers.md](docs/providers.md) | Which adapter declares which capability, and what branching on each one buys you. Generated from the adapters themselves |
| [docs/packages/modelrack/spec.md](docs/packages/modelrack/spec.md) | Purpose, scope, non-goals, public contracts, configuration, acceptance criteria |
| [docs/packages/modelrack/development-plan.md](docs/packages/modelrack/development-plan.md) | The phased build plan: goals, work, tests, acceptance criteria per phase |

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

ruff format --check . && ruff check . && mypy src tests && lint-imports
pytest -m "not live and not performance"      # the default gate; coverage floor 95%
pytest -m performance                          # spec §15's overhead budgets, nightly
pytest -m live                                 # needs a real Ollama; skips if none is reachable
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow and [`SECURITY.md`](SECURITY.md) for
how to report a vulnerability.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
