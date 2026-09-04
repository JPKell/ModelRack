# ModelRack — provider capability matrix

Generated from each adapter's own `capabilities()` by
[`scripts/generate_provider_matrix.py`](../scripts/generate_provider_matrix.py). Do not
hand-edit — regenerate instead. `tests/unit/test_provider_matrix.py` fails when this file
and the adapters disagree.

A capability is a **declaration a caller branches on**, never an assumption
(ADR-0007 rule 2). Asking for something a provider has not declared raises
`CapabilityUnsupported` naming the flag; it is never accepted and quietly ignored, which
would produce a result you would misread as the model's own choice.

| Capability | `OllamaProvider` | `OpenAICompatibleProvider` | `LlamaCppProvider` | `FakeProvider` (full) | `FakeProvider` (minimal) | What declaring it buys a caller |
|---|---|---|---|---|---|---|
| `streaming` | yes | yes | yes | yes | no | `stream()` yields deltas; otherwise it refuses. |
| `tool_calling` | yes | yes | yes | yes | no | `tools=` is accepted and calls can be requested. |
| `structured_output` | yes | yes | yes | yes | no | `ResponseFormat.JSON_SCHEMA` is enforced. |
| `json_mode` | yes | yes | yes | yes | no | `ResponseFormat.JSON` is honoured without a schema. |
| `token_counts` | yes | yes | yes | yes | no | Token counts are real; otherwise every count is `UNSUPPORTED`. |
| `token_level_chunks` | yes | no | yes | yes | no | **Gates any per-token latency claim.** When `False`, the gap between two deltas is inter-chunk latency and must not be relabelled. |
| `thinking_control` | yes | no | no | yes | no | Reasoning output can be requested or suppressed. |
| `logprobs` | no | no | no | no | no | Per-token log probabilities are reported. |
| `force_unload` | yes | no | yes | yes | no | `load()` and `unload()` act; otherwise both refuse. |
| `residency_query` | yes | no | yes | yes | no | `list_resident()` answers; otherwise it refuses. |
| `kv_metrics` | no | no | no | no | no | KV-cache metrics are reported. |
| `context_configurable` | yes | no | yes | yes | no | **Load-bearing.** Whether you may set a served context, or must record the one you got as *assumed*. |
| `embedding` | no | no | no | no | no | Embeddings can be produced. Out of scope until spec §21. |
| `adapter_hot_swap` | no | no | yes | no | no | **Load-bearing.** Whether `GenerationRequest.adapter` may name a LoRA adapter, and whether `list_adapters()`/`register_adapters()` answer. When `False`, all three refuse. |

## Reading the columns

* **`OllamaProvider`** — the richest daemon-backed adapter: digests, residency
  control, a configurable served context, and deltas that really are one token each.
* **`OpenAICompatibleProvider`** — honestly narrower, and that narrowness is the
  point. `/v1/models` carries no digest anywhere, so every identity is `NAME_ONLY`
  (ADR-0024 §2); the protocol has no residency endpoint and no per-request served
  context, so those are declared `False` rather than accepted and ignored.
* **`LlamaCppProvider`** — the server this package spawns itself (ADR-0062), so
  residency control, a residency query and a configurable served context are
  literally what supervision is; identities are digest-bound because the adapter
  hashes the file it serves. `thinking_control` stays `False`: reasoning is *read*
  where the server reports it, but requesting or suppressing it is not exposed.
* **`FakeProvider` (full)** — the default script, matching Ollama's declaration, so a
  consumer's ordinary tests exercise the capable path.
* **`FakeProvider` (minimal)** — every flag `False`. Testing only against the full
  declaration is testing against the easy half; `MINIMAL_CAPABILITIES` is one
  constructor argument away precisely so that swap actually gets made.

```python
from modelrack.testing import MINIMAL_CAPABILITIES, FakeProvider, FakeScript

weak = FakeProvider(FakeScript(capabilities=MINIMAL_CAPABILITIES))
weak.capabilities().streaming  # False — and stream() refuses rather than degrading
```
