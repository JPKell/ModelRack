#!/usr/bin/env python3
"""Regenerate ``docs/providers.md`` from what each adapter actually declares.

The development plan's Phase 4 acceptance criterion 2 is that the capability matrix is *generated
from the adapters' declarations, not hand-written*, and the reason is the one this whole package
exists for: a hand-written matrix is a claim about an adapter, while a generated one is the
adapter's own statement. The two diverge silently — an adapter gains a capability, the table does
not, and a caller who read the table branches away from something the provider can now do. Worse
in the other direction: a table that promises a capability an adapter dropped sends a caller
straight into a ``CapabilityUnsupported`` it had no reason to expect.

Run after any change to a ``_CAPABILITIES`` declaration::

    python scripts/generate_provider_matrix.py

The output is committed, and ``tests/unit/test_provider_matrix.py`` fails when it drifts — so a
stale matrix is caught by the ordinary test run rather than by review.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

from modelrack import ProviderCapabilities
from modelrack.providers.fake import FULL_CAPABILITIES, MINIMAL_CAPABILITIES
from modelrack.providers.llamacpp import LlamaCppProvider
from modelrack.providers.ollama import OllamaProvider
from modelrack.providers.openai_compatible import OpenAICompatibleProvider

_OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "providers.md"

# The llama.cpp adapter needs an existing model directory to construct and creates its state
# directory only on the first spawn, so pointing both at the temp directory reads its static
# declaration without spawning, hashing or writing anything.
_SCRATCH = Path(tempfile.gettempdir())

_COLUMNS: tuple[tuple[str, ProviderCapabilities], ...] = (
    ("`OllamaProvider`", OllamaProvider(base_url="http://127.0.0.1:11434").capabilities()),
    (
        "`OpenAICompatibleProvider`",
        OpenAICompatibleProvider(base_url="http://127.0.0.1:8080").capabilities(),
    ),
    (
        "`LlamaCppProvider`",
        LlamaCppProvider(_SCRATCH, state_dir=_SCRATCH / "modelrack-unused-state").capabilities(),
    ),
    ("`FakeProvider` (full)", FULL_CAPABILITIES),
    ("`FakeProvider` (minimal)", MINIMAL_CAPABILITIES),
)

# What each flag means for a *caller*, not what it means internally. Keyed by field name so a flag
# added to ProviderCapabilities without a note here fails loudly rather than rendering an empty
# cell — the matrix is only useful if every row says what branching on it buys you.
_NOTES: dict[str, str] = {
    "streaming": "`stream()` yields deltas; otherwise it refuses.",
    "tool_calling": "`tools=` is accepted and calls can be requested.",
    "structured_output": "`ResponseFormat.JSON_SCHEMA` is enforced.",
    "json_mode": "`ResponseFormat.JSON` is honoured without a schema.",
    "token_counts": "Token counts are real; otherwise every count is `UNSUPPORTED`.",
    "token_level_chunks": (
        "**Gates any per-token latency claim.** When `False`, the gap between two deltas is "
        "inter-chunk latency and must not be relabelled."
    ),
    "thinking_control": "Reasoning output can be requested or suppressed.",
    "logprobs": "Per-token log probabilities are reported.",
    "force_unload": "`load()` and `unload()` act; otherwise both refuse.",
    "residency_query": "`list_resident()` answers; otherwise it refuses.",
    "kv_metrics": "KV-cache metrics are reported.",
    "context_configurable": (
        "**Load-bearing.** Whether you may set a served context, or must record the one you got "
        "as *assumed*."
    ),
    "embedding": "Embeddings can be produced. Out of scope until spec §21.",
    "adapter_hot_swap": (
        "**Load-bearing.** Whether `GenerationRequest.adapter` may name a LoRA adapter, and "
        "whether `list_adapters()`/`register_adapters()` answer. When `False`, all three refuse."
    ),
}


def _cell(value: bool) -> str:
    return "yes" if value else "no"


def render() -> str:
    """Return the full text of ``docs/providers.md`` for the current declarations.

    Returns:
        The rendered document, newline-terminated. Pure: it reads the adapters' static
        declarations and touches nothing else, which is what lets a test call it and compare.

    Raises:
        KeyError: If :class:`~modelrack.ProviderCapabilities` has a field with no caller-facing
            note in ``_NOTES``. Loud on purpose — a flag nobody explained is a flag nobody can
            branch on.
    """
    fields = [field.name for field in dataclasses.fields(ProviderCapabilities)]
    headers = [name for name, _ in _COLUMNS]
    lines = [
        "# ModelRack — provider capability matrix",
        "",
        "Generated from each adapter's own `capabilities()` by",
        "[`scripts/generate_provider_matrix.py`](../scripts/generate_provider_matrix.py). Do not",
        "hand-edit — regenerate instead. `tests/unit/test_provider_matrix.py` fails when this file",
        "and the adapters disagree.",
        "",
        "A capability is a **declaration a caller branches on**, never an assumption",
        "(ADR-0007 rule 2). Asking for something a provider has not declared raises",
        "`CapabilityUnsupported` naming the flag; it is never accepted and quietly ignored, which",
        "would produce a result you would misread as the model's own choice.",
        "",
        "| Capability | " + " | ".join(headers) + " | What declaring it buys a caller |",
        "|---|" + "---|" * len(headers) + "---|",
    ]
    for field in fields:
        cells = [_cell(getattr(capabilities, field)) for _, capabilities in _COLUMNS]
        lines.append(f"| `{field}` | " + " | ".join(cells) + f" | {_NOTES[field]} |")
    lines.extend(
        [
            "",
            "## Reading the columns",
            "",
            "* **`OllamaProvider`** — the richest daemon-backed adapter: digests, residency",
            "  control, a configurable served context, and deltas that really are one token each.",
            "* **`OpenAICompatibleProvider`** — honestly narrower, and that narrowness is the",
            "  point. `/v1/models` carries no digest anywhere, so every identity is `NAME_ONLY`",
            "  (ADR-0024 §2); the protocol has no residency endpoint and no per-request served",
            "  context, so those are declared `False` rather than accepted and ignored.",
            "* **`LlamaCppProvider`** — the server this package spawns itself (ADR-0062), so",
            "  residency control, a residency query and a configurable served context are",
            "  literally what supervision is; identities are digest-bound because the adapter",
            "  hashes the file it serves. `thinking_control` stays `False`: reasoning is *read*",
            "  where the server reports it, but requesting or suppressing it is not exposed.",
            "* **`FakeProvider` (full)** — the default script, matching Ollama's declaration, so a",
            "  consumer's ordinary tests exercise the capable path.",
            "* **`FakeProvider` (minimal)** — every flag `False`. Testing only against the full",
            "  declaration is testing against the easy half; `MINIMAL_CAPABILITIES` is one",
            "  constructor argument away precisely so that swap actually gets made.",
            "",
            "```python",
            "from modelrack.testing import MINIMAL_CAPABILITIES, FakeProvider, FakeScript",
            "",
            "weak = FakeProvider(FakeScript(capabilities=MINIMAL_CAPABILITIES))",
            "weak.capabilities().streaming  # False — and stream() refuses rather than degrading",
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Write ``docs/providers.md`` and report what it wrote."""
    text = render()
    _OUTPUT.write_text(text)
    print(f"Wrote {_OUTPUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
