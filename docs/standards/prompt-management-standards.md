# Prompt Management Standards

**Decision records:** [ADR-0012](../adr/0012-prompt-storage-format.md), [ADR-0028](../adr/0028-prompt-pack-granularity.md) (attribution granularity and shared tooling).
**Rule:** prompts are versioned data, not source code. A benchmark comparison, a routing decision
and a content workflow are all only as reproducible as the prompt that produced them.

---

## 1. What may live in Python

Almost nothing. Prompts are **not** embedded in Python source, with one narrow exception:

> A protocol constant shorter than ~200 characters that is structurally part of a request rather
> than an instruction to the model — for example the fixed JSON-only reminder appended by a
> structured-output validator — may be a module constant, **only** when accompanied by a comment
> stating why it is not a prompt record.

Everything else — system prompts, task instructions, judge rubrics, audit checklists, repair
instructions, few-shot examples — is a prompt record.

This rule exists because two prior projects embedded prompts in Python (a 406-line `prompting.py`
of f-strings, and `SYSTEM_PROMPT` constants), which made prompt changes invisible in diffs of
behaviour, impossible to version independently, and impossible to attach to a benchmark result.

---

## 2. Storage format

JSON, one file per prompt, under the owning component's `prompts/` directory:

```text
src/<app>/prompts/
├── manifest.json                     # pack identity + hash + index
├── benchmarks/
│   ├── audit.detect_defects.v3.json
│   └── judge.pairwise.v2.json
├── stages/
│   ├── draft.article.v1.json
│   └── edit.line_edit.v2.json
└── system/
    └── structured_output_guard.v1.json
```

### 2.1 Prompt record schema

```json
{
  "prompt_id": "stages.draft.article",
  "version": "2.1.0",
  "schema_version": "1.0",
  "purpose": "Produce a first draft of one article section from a compiled requirement set.",
  "task": "content.article_draft",
  "capability": "creative_writing",
  "system": "You are drafting one section of an article. Follow every hard requirement…",
  "template": "SECTION: {{ section_title }}\n\nREQUIREMENTS\n{{ requirements_json }}\n\nCONTEXT\n{{ context }}\n\nWrite the section.",
  "variables": {
    "section_title": {"type": "string", "required": true,  "description": "Human-readable section title"},
    "requirements_json": {"type": "string", "required": true,  "description": "Canonical JSON of compiled requirements"},
    "context": {"type": "string", "required": false, "default": "", "description": "Bounded prior context"},
    "word_target": {"type": "integer", "required": false, "default": 800, "min": 100, "max": 5000}
  },
  "response": {
    "format": "text",
    "json_schema_ref": null,
    "expectations": ["No headings above H3", "No meta-commentary about being an AI"]
  },
  "model_requirements": {
    "min_context_tokens": 8192,
    "requires_capabilities": [],
    "recommended_temperature": 0.7
  },
  "metadata": {
    "author": "suite",
    "created_at": "2026-08-21T00:00:00Z",
    "changed_at": "2026-08-21T00:00:00Z",
    "change_reason": "Tightened the requirement-compliance instruction after audit-pass regressions.",
    "supersedes": "2.0.0",
    "tags": ["content", "draft"]
  }
}
```

Field rules:

| Field | Rule |
|---|---|
| `prompt_id` | Dotted, stable, unique within the pack. Never renamed — a rename is a new prompt. |
| `version` | Semantic. **Patch**: typo/whitespace with no behavioural intent. **Minor**: clarification, added optional variable. **Major**: changed instruction, changed required variables, changed output contract. |
| `schema_version` | Version of *this record format*, independent of the prompt's own version. |
| `system` / `template` | Jinja2 with `StrictUndefined`. A referenced-but-unsupplied variable is an error, never an empty string. |
| `variables` | Every variable used in `system` or `template` must be declared, with type, requiredness and description. Undeclared use fails validation. |
| `response.format` | `text` \| `json` \| `json_schema` \| `enum`. When `json_schema`, `json_schema_ref` names a schema file and the caller must validate against it. |
| `model_requirements` | Hard requirements that make a model ineligible (context, capabilities), plus soft recommendations. LoadCoach reads these as routing constraints. |
| `metadata.change_reason` | Mandatory on every version bump. "Improved wording" is not a reason. |

---

## 3. Prompt packs and hashing

Every component that owns prompts ships one **pack** with a manifest:

```json
{
  "pack_id": "ideapress.core",
  "pack_version": "1.4.0",
  "schema_version": "1.0",
  "generated_at": "2026-08-21T00:00:00Z",
  "prompts": [
    {"prompt_id": "stages.draft.article", "version": "2.1.0", "sha256": "9f2c…"},
    {"prompt_id": "stages.edit.line_edit", "version": "1.0.3", "sha256": "1ab4…"}
  ],
  "pack_sha256": "c3d9…"
}
```

* `sha256` is over the record's canonical JSON (sorted keys, UTF-8, no insignificant whitespace).
* `pack_sha256` is over the sorted list of `(prompt_id, version, sha256)`.
* A **`prompt_subset_hash`** is the same computation over an arbitrary subset. Every benchmark
  manifest declares one, covering only the prompts that benchmark uses, and it is that hash — not
  `pack_sha256` — that enters the reproducibility fingerprint and the evidence separation rules.
  Editing a prompt no benchmark uses therefore separates nothing
  ([ADR-0028](../adr/0028-prompt-pack-granularity.md)).
* These hashes appear in cross-application evidence, so their determinism is a contract: they are
  golden-tested in the same way as `canonical_json`, and there is one implementation of them
  (`setspec.prompts`) rather than one per application.
* The manifest is regenerated by `<app> prompts build` and validated in CI; a record edited without
  a regenerated manifest fails the build.

---

## 4. Traceability

Every use of a prompt records what was used:

| Consumer | Records |
|---|---|
| FreeWeight sample | `prompt_id`, `version`, `sha256`, rendered-prompt hash, and the pack hash on the run |
| FreeWeight run | `prompt_pack_id`, `pack_version`, `pack_sha256` as provenance — **not** fingerprint inputs; the fingerprint takes the per-benchmark `prompt_subset_hash` |
| LoadCoach job | `prompt_id`, `version`, `sha256` for every prompt used in execution or validation |
| IdeaPress unit | `prompt_id`, `version`, `sha256` per stage attempt, stored with the draft |

Consequences that must hold:

* A benchmark result can name the exact prompt text that produced it, and re-render it.
* Two results are **not comparable** for quality metrics when the versions of the prompts *their
  benchmark uses* differ, and are separated in the comparison UI exactly like a benchmark-version
  difference. A difference elsewhere in the pack separates nothing.
* Changing a prompt used by a benchmark requires a benchmark suite version bump.
* The rendered prompt is stored as a hash by default and as full text only when
  `logging.include_content` / the run's `store_prompts` option is enabled (privacy and size).

---

## 5. Loading and rendering

The loader, validator, renderer and hasher live in **`setspec.prompts`** — one implementation for
three applications, because the hashes cross an application boundary
([ADR-0028](../adr/0028-prompt-pack-granularity.md)). Each application still owns its own pack, its
own records and its own `prompts` CLI commands; no prompt is ever shared or imported across
components.

```python
class PromptLibrary:
    """Loads, validates and renders versioned prompt records from a pack."""

    def get(self, prompt_id: str, *, version: str | None = None) -> PromptRecord:
        """Return a prompt record; latest version when ``version`` is None."""

    def render(self, prompt_id: str, variables: Mapping[str, Any], *, version: str | None = None) -> RenderedPrompt:
        """Validate variables and render system+user text.

        Raises:
            PromptNotFound: no such prompt or version.
            PromptVariableError: required variable missing, unknown variable supplied,
                or a value outside its declared type/range.
            PromptRenderError: template failure (StrictUndefined, syntax).
        """
```

* Loading happens **once at startup**; the whole pack is validated then. A malformed prompt is a
  startup failure (exit 3), never a runtime surprise mid-run.
* `RenderedPrompt` carries `system`, `user`, `prompt_id`, `version`, `sha256`, `rendered_sha256`.
* Rendering is pure and deterministic: same record + same variables ⇒ same output, byte for byte.
* Unknown variables supplied by the caller are an **error**, not ignored — that is how a renamed
  variable is caught.

---

## 6. Overrides

Users may override prompts without editing the installed package:

```text
$XDG_CONFIG_HOME/<app>/prompts/<prompt_id>.json
```

* An override must declare the same `prompt_id`, must validate against the record schema, and is
  loaded with its own version and hash.
* Overridden prompts are marked in the UI and in every record that used them
  (`prompt_source: "user_override"`), because an overridden prompt invalidates comparison with
  results produced by the shipped one.
* FreeWeight refuses to run a **benchmark** with an overridden prompt unless `--allow-prompt-override`
  is passed, and records the override in the reproducibility fingerprint when it is.

---

## 7. Testing

Every repository that ships prompts includes:

| Test | Asserts |
|---|---|
| Pack parses | Every file is valid JSON and validates against the record schema |
| Manifest is current | Recomputed hashes match the committed manifest |
| Variables declared | Every `{{ variable }}` in `system`/`template` is declared; every declared variable is used |
| Rendering | Each prompt renders with its documented example variables; `StrictUndefined` raises on a missing one |
| Type and range | Declared types/min/max are enforced |
| Schema compatibility | A prompt declaring `response.format = "json_schema"` names a schema that exists and is valid |
| Determinism | Same inputs ⇒ identical bytes, twice |
| No inline prompts | A source scan finds no multi-line string literal that looks like a prompt outside `prompts/` |
| Model requirements | Declared `min_context_tokens` and required capabilities are values the routing layer understands |
| Traceability | A simulated run records `prompt_id`, `version` and `sha256` on every sample |
| Subset hashing | A benchmark's `prompt_subset_hash` changes when one of *its* prompts changes and does not change when another prompt in the pack does |
| Cross-application determinism | The same record hashes identically under every application's installed `setspec` version in the compatibility matrix |

---

## 8. Review

A prompt change is a behaviour change and is reviewed as one:

* The PR shows the record diff, the version bump and the `change_reason`.
* Any prompt used by a benchmark requires a before/after run on at least one model, with the
  affected metrics in the PR description.
* A prompt change that alters `response.format` or required variables is a major bump and must list
  every consumer it affects.
