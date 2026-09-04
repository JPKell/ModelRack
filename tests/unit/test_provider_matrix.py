"""The capability matrix in ``docs/providers.md`` is generated, and stays that way.

The development plan's Phase 4 acceptance criterion 2 asks for a matrix *generated from the
adapters' declarations, not hand-written*. A generator alone only satisfies that on the day it is
run: the file is committed, the declarations keep moving, and a stale table is worse than no table
— it sends a caller branching away from a capability an adapter has since gained, or straight into
a ``CapabilityUnsupported`` it had no reason to expect.

So the check is here rather than in review. ``scripts/`` is not importable as a package, which is
deliberate — it holds standalone tools, not library code — so this loads the generator by path the
way a CLI would run it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from modelrack import ProviderCapabilities

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATOR = _REPO_ROOT / "scripts" / "generate_provider_matrix.py"
_MATRIX = _REPO_ROOT / "docs" / "providers.md"


@pytest.fixture(scope="module")
def generator() -> Iterator[ModuleType]:
    """Import ``scripts/generate_provider_matrix.py`` by path, as running it would."""
    spec = importlib.util.spec_from_file_location("_provider_matrix_generator", _GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


class TestTheMatrixIsGenerated:
    def test_the_generator_exists_where_the_document_says_it_does(self) -> None:
        assert _GENERATOR.is_file()

    def test_the_committed_matrix_matches_the_adapters(self, generator: ModuleType) -> None:
        """The whole point: run the generator, compare, and fail with the fix in the message."""
        assert _MATRIX.read_text() == generator.render(), (
            "docs/providers.md no longer matches what the adapters declare. Regenerate it:\n"
            "    python scripts/generate_provider_matrix.py"
        )

    def test_every_capability_flag_has_a_row(self, generator: ModuleType) -> None:
        """A flag added to ``ProviderCapabilities`` without a row would be invisible to callers."""
        rendered = generator.render()

        for field in dataclasses.fields(ProviderCapabilities):
            assert f"| `{field.name}` |" in rendered, f"{field.name} has no row in the matrix"

    def test_a_flag_with_no_caller_facing_note_fails_loudly(
        self, generator: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row that renders an empty cell is a row nobody can branch on, so the generator
        raises rather than shipping one.
        """
        notes = dict(generator._NOTES)  # noqa: SLF001 — asserting the generator's own contract
        del notes["streaming"]
        monkeypatch.setattr(generator, "_NOTES", notes)

        with pytest.raises(KeyError):
            generator.render()

    def test_every_shipped_adapter_has_a_column(self, generator: ModuleType) -> None:
        rendered = generator.render()

        for name in (
            "`OllamaProvider`",
            "`OpenAICompatibleProvider`",
            "`LlamaCppProvider`",
            "`FakeProvider`",
        ):
            assert name in rendered

    def test_the_declarations_the_matrix_reports_are_the_adapters_own(
        self, generator: ModuleType
    ) -> None:
        """Not a re-statement: the generator reads ``capabilities()``, so this asserts the source
        rather than the rendering.
        """
        from modelrack.providers.ollama import OllamaProvider

        declared = OllamaProvider(base_url="http://127.0.0.1:11434").capabilities()
        columns = dict(generator._COLUMNS)  # noqa: SLF001 — asserting the generator's own contract

        assert columns["`OllamaProvider`"] == declared

    def test_reading_the_declarations_makes_no_request(self, generator: ModuleType) -> None:
        """The conftest socket guard is armed: building the columns above already proved this,
        and stating it as its own assertion keeps it from being an accident.
        """
        assert generator.render()
