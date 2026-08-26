# ModelRack — Quickstart

Everything below runs with no GPU, no model and no server, until the section that says otherwise.

```bash
pip install modelrack
```

`modelrack` needs `baseaicore` and `httpx`, and nothing else.

---

## 1. The shape of every call

Application code takes a `Provider` and never names one. That is the whole point of the package:
FreeWeight, LoadCoach and IdeaPress contain no provider HTTP code and cannot disagree about what a
token count or a timing means.

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

Three adapters satisfy that protocol — `FakeProvider`, `OllamaProvider`,
`OpenAICompatibleProvider` — and all three pass one conformance suite.

## 2. Run it against the fake

```python
from modelrack.testing import FakeProvider

provider = FakeProvider(seed=42)
identity = provider.resolve("fake-model")
print(summarize(provider, identity, "a long document"))
```

Same script, same seed, same bytes — in another process, on another platform, under another
`PYTHONHASHSEED`.

## 3. Stream, and stop part-way

`stream()` yields zero or more deltas and then **exactly one** terminal event. A truncated stream
is detectable as the absence of one, rather than being indistinguishable from a short answer.

```python
from modelrack import StreamCompleted, StreamFailed, TokenDelta
from modelrack.streaming import CancellationToken

token = CancellationToken()
text = []

for event in provider.stream(request):
    match event:
        case TokenDelta():
            text.append(event.text)
            if len(text) == 20:
                token.cancel()          # safe from another thread, too
        case StreamCompleted():
            print("finished:", event.result.finish_reason.value)
        case StreamFailed():
            print("stopped:", event.error.code, "after", len(event.partial_text), "chars")
```

Cancellation takes effect within one chunk, preserves the partial text on the terminal event, and
leaves no connection open. A token already set *before* `stream()` is called yields that one
terminal event and opens no connection at all.

`generate()` is deliberately **not** cancellable — a single blocking round trip has no boundary at
which a token could take effect. Code that needs to cancel streams and assembles the result itself;
`StreamCompleted.result` is exactly what `generate()` would have returned.

## 4. Check capabilities; never assume

```python
capabilities = provider.capabilities()

if capabilities.tool_calling:
    result = provider.generate(request_with_tools)
else:
    ...  # a different plan, chosen before spending a request
```

Asking for something a provider has not declared raises `CapabilityUnsupported` naming the flag —
it is never accepted and quietly ignored, which would produce a result you would misread as the
model's own choice.

Two flags are load-bearing rather than informational:

* **`token_level_chunks`** gates any per-token latency claim. When it is `False`, the gap between
  two deltas is inter-chunk latency and must not be relabelled.
* **`context_configurable`** tells you whether you may set a served context, or must record the one
  you got as *assumed*.

## 5. Talk to a real Ollama

```python
from modelrack.providers.ollama import OllamaProvider

provider = OllamaProvider("http://127.0.0.1:11434")   # the default
print(provider.health())
identity = provider.resolve("qwen3.5:9b-q8_0")
print(summarize(provider, identity, "a long document"))
```

`health()` never raises: "is it up?" is a question whose negative answer is not exceptional.

An OpenAI-compatible server (llama.cpp server, LM Studio, …) is the same code with a different
import:

```python
from modelrack.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider("http://127.0.0.1:8080", api_key=None)
```

Both are imported from their own modules rather than from `modelrack`, because they are the only
two places in the package that import `httpx`.

## 6. Residency: what is loaded, and what you may do about it

```python
from modelrack import find_resident, residency_support

support = residency_support(provider.capabilities())
if support.is_manageable:
    provider.load(identity, RuntimeProfile())
    entry = find_resident(provider.list_resident(), identity)
    print(entry.vram_bytes if entry else "not loaded")
    provider.unload(identity)
else:
    ...  # degrade to load-on-demand, and record that you did
```

`load()` reports `already_resident`, which is the difference between a real cold-start figure and
one an order of magnitude wrong. `unload()` returns `False` when the model was not loaded — that
is the state you wanted, not a failure. On a provider that declares neither power, all three refuse
with `CapabilityUnsupported` rather than silently doing nothing.

Membership is matched on the provider-side model name, because that is the only field a provider's
residency report and your own `ModelIdentity` genuinely agree on.

## 7. Metadata caching

Discovery is cached for five minutes by default; generations are **never** cached, and neither is
residency or health.

```python
provider = OllamaProvider(metadata_ttl_seconds=300)   # 0 disables it entirely

provider.list_models()                 # cold: /api/tags plus one /api/show per model
provider.list_models()                 # warm
provider.list_models(refresh=True)     # a model was just re-pulled — read it again now

print(provider.metadata_cache_stats())  # hits, misses, expirations, stores, entries
provider.clear_metadata_cache()
```

A cached descriptor keeps the instant the provider actually answered, so `observed_at` never claims
a freshness the data does not have. `refresh=True` exists because a TTL alone cannot help you: a
tag such as `qwen3.5:latest` can be repointed the moment after it is read.

## 8. Watch what the client is doing

```python
from modelrack import ProviderEvent, ProviderEventKind

def log(event: ProviderEvent) -> None:
    if event.kind is ProviderEventKind.REQUEST_COMPLETED:
        print(event.operation, event.model_name, event.elapsed_ms, event.metadata["run_id"])

provider = OllamaProvider(on_event=log)
provider.generate(GenerationRequest(..., metadata={"run_id": run_id}))
```

An event carries **no content** — no prompt, no generated text, no tool arguments, no credential.
There is no field one could reach. What it does carry is `metadata`, your own correlation
identifiers, which are never sent to the provider. A callback that raises is logged at DEBUG and
does not disturb the generation.

The library logs nothing at INFO or above. DEBUG-level logs of request *shape* are available under
`modelrack.ollama` and `modelrack.openai_compatible`.

## 9. When it goes wrong

Every failure is a typed error under `ProviderError`, and no adapter ever raises a raw `httpx`
exception:

| You will see | When |
|---|---|
| `ProviderUnavailable` | Connection refused, DNS failure, TLS failure |
| `ProviderTimeout` | Connect or read timeout |
| `ProviderProtocolError` | Non-JSON, unexpected JSON, truncated stream, oversize response |
| `ModelNotFound` | 404, or an ambiguous reference that names no single model |
| `ContextLimitExceeded` | The provider reports a context overflow |
| `CapabilityUnsupported` | You asked for something the provider has not declared |
| `GenerationCancelled` | Your token fired — partial text preserved |
| `ProviderRejected` | A 4xx with the provider's own message, verbatim |

Nothing is retried internally, nothing is swallowed, and nothing is converted into an empty result.
Retry and fallback policy belong to the caller — in this suite, to LoadCoach.

## 10. Where to go next

* [Specification](packages/modelrack/spec.md) — scope, non-goals, public contracts, acceptance
  criteria.
* [Development plan](packages/modelrack/development-plan.md) — what each phase added and why.
* `tests/contract/test_conformance.py` — the behaviours every adapter must exhibit, and the
  best single description of what a `Provider` promises.
