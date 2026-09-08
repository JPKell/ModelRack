"""tests/unit/test_readme_version.py — the README's stated version cannot drift from `__about__`.

An operator reads the README's status line to know what they are about to install. Nothing bound
that line to the package's actual version, so a release could bump `__about__.__version__` and
leave the README naming an older one — which is exactly what the M9 re-audit found across the
suite (row L7). This test parses the version the README states after its `Status:` line and
asserts it against `modelrack.__about__.__version__`.
"""

from __future__ import annotations

import re
from pathlib import Path

from modelrack import __about__

REPO_ROOT = Path(__file__).resolve().parents[2]

_VERSION_RE = re.compile(r"Status:(?:.|\n)*?(\d+\.\d+\.\d+)")


def test_readme_states_the_current_version() -> None:
    """The first version named after README.md's `Status:` line equals `__about__.__version__`."""
    text = (REPO_ROOT / "README.md").read_text()
    match = _VERSION_RE.search(text)
    assert match is not None, "README.md has no version after its Status: line"
    assert match.group(1) == __about__.__version__
