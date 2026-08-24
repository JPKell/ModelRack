"""Provider adapters — the only modules in the suite that know a runtime's wire format.

Everything above this package speaks :class:`modelrack.Provider` and the provider-neutral types in
:mod:`modelrack.types`; nothing above it parses provider JSON
([ADR-0007](../../../docs/adr/0007-provider-abstraction.md) rule 1).

Adapters are imported from their own modules rather than re-exported here, so importing
:mod:`modelrack` never drags an adapter — and its transport — into a process that only wanted the
vocabulary. :class:`~modelrack.providers.fake.FakeProvider` is reached through
:mod:`modelrack.testing`, the path the suite's testing standards name.
"""

from __future__ import annotations
