"""Gold standard G16: the declared dependency budget matches gold-standards.md §1.1.

ADR-0114 makes the dependency budget the enumerated set a component declares, not a count. Adding
a name to the set below needs an ADR; removing one needs only a release. A version-range change
is neither, and does not move this test. gold-standards.md §1.1 enumerates the non-suite
dependencies ModelRack is entitled to beyond the suite packages it imports; this reads the
component's own `pyproject.toml` and proves the two have not drifted apart.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Suite packages are unbudgeted (gold-standards.md §1.1): declaring one is an architectural
# decision recorded elsewhere (an ADR, a spec), not a spend against this component's own
# non-suite dependency budget.
SUITE_PACKAGES = frozenset(
    {
        "baseaicore",
        "setspec",
        "modelrack",
        "sweatmeter",
        "weightsdb",
        "mirrorwall",
        "loadledger",
        "cutctx",
        "toolyard",
        "commissioner",
    }
)

# gold-standards.md §1.1: the enumerated non-suite runtime dependencies approved for
# ModelRack.
APPROVED_NON_SUITE_DEPENDENCIES: frozenset[str] = frozenset({"httpx"})


def _dependency_name(requirement: str) -> str:
    """Return the bare distribution name a PEP 508 requirement string declares.

    Strips the environment marker, any extras (``pkg[extra]``) and every version specifier — the
    parts of a requirement that do not answer "which name did this spend from the budget".
    """
    name = requirement.split(";", 1)[0].strip()
    for separator in ("[", ">", "<", "=", "!", "~", " "):
        index = name.find(separator)
        if index != -1:
            name = name[:index]
    return name.strip()


def test_the_declared_dependencies_match_the_enumerated_budget() -> None:
    """G16: `[project.dependencies]`, minus suite packages, is exactly §1.1's approved set."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]
    non_suite = {
        name
        for requirement in dependencies
        if (name := _dependency_name(requirement)) not in SUITE_PACKAGES
    }
    assert non_suite == APPROVED_NON_SUITE_DEPENDENCIES
