# ADR-0016 — Unavailable is not zero

**Status:** Accepted (2026-08-21)

## Context

A benchmarking suite's core promise is that its numbers mean something. The most damaging bug class
in such a system is not a crash — it is a plausible number that was never measured: `0 W` because
the power sensor is unreadable, `0` prompt tokens because the provider did not report them, `0.0`
tokens/second because a timing field was missing.

These values then flow into averages, charts, capability scores and routing decisions, where they are
indistinguishable from real measurements. `value or 0`, `float(value or 0)` and
`sum(values) / len(values)` are the specific idioms that produce them.

The prior implementation solved this well, and the solution is adopted as suite policy.

## Decision

**A first-class `Unsupported` sentinel in BaseAiCore that refuses to behave like a number.**

```python
Measurement = int | float | Unsupported     # the real annotation, never `Any`

class Unsupported:
    """A measurement this environment genuinely cannot provide.

    Boolean, numeric and ordering coercion all raise. The failure this guards against is
    ``value or 0`` / ``value + x`` quietly turning "not measurable" into a real-looking
    number — the most damaging bug class in a measurement system.
    """
    __bool__ = __int__ = __float__ = _refuse
    __add__ = __radd__ = __sub__ = __rsub__ = _refuse
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _refuse
    __lt__ = __le__ = __gt__ = __ge__ = _refuse

UNSUPPORTED = Unsupported()          # module-level singleton
```

Rules across the suite:

1. Any value that might not be obtainable is typed `Measurement`, never `Any`, never `float | None`.
2. Checks are explicit: `if value is UNSUPPORTED` / `is_supported(value)`. Truthiness raises.
3. **Storage:** `NULL` **plus a reason** — a reason column or a key in the row's detail JSON. `NULL`
   alone cannot distinguish "not measured this time" from "not measurable here".
4. **JSON:** the string `"unsupported"`. Never `null`, never `0`.
5. **UI:** `—` with a tooltip naming the reason. Never `0`, never blank.
6. **Aggregation:** unsupported values are excluded from statistics, and the sample count that
   produced each statistic is reported alongside it. A metric with no supported samples is itself
   unsupported — not zero, and not omitted silently.
7. **Scoring:** a capability that could not be measured is absent from evidence, never scored zero.
   Absence of evidence is not evidence of incapacity.
8. **Skips:** a test skipped because a capability is unsupported records `skipped` with a reason and
   never contributes a zero to any average.
9. `None` keeps its ordinary Python meaning ("no value in this context", e.g. an optional digest) and
   is not a substitute for `UNSUPPORTED`.

## Alternatives considered

**`None`.** The obvious Python answer. Rejected as insufficient: `None` is already used for ordinary
optionality, it is falsy (so `value or 0` still fires), and it cannot carry the "this is a
measurement that does not exist here" meaning distinctly.

**`float("nan")`.** Propagates through arithmetic instead of raising. Rejected: it silently
contaminates aggregates, serializes badly to JSON, and `nan != nan` surprises comparisons.

**A `Maybe`/`Result` wrapper type.** Rejected: it forces unwrapping at every call site for a
codebase that is otherwise plain Python, and the sentinel achieves the same protection with far less
ceremony.

**A parallel `*_available: bool` field per measurement.** Rejected: doubles the field count and can
drift out of sync with the value it describes.

**Just document the convention.** Rejected — this is exactly the failure mode a convention cannot
prevent. The type must refuse.

## Consequences

*Positive.* The dangerous idioms raise loudly at development time instead of producing a wrong chart
in production. The distinction survives all the way into storage, JSON, exports and the UI.
Aggregations become honest about their sample counts. Tests can assert "this is unsupported, not
zero" as a first-class property.

*Negative.* Callers must handle it explicitly, which is more code at every measurement site. That is
the intended cost.

*Negative.* It cannot be used inside pydantic models without a custom serializer/validator. SetSpec
provides one, mapping `UNSUPPORTED ↔ "unsupported"`, and it is tested.

*Negative.* Sentinels are unusual in modern Python and need a clear docstring, which they have.

## Revisit when

Never, in spirit. Mechanically, if a future Python or pydantic feature makes an equivalent guarantee
more idiomatic, the sentinel's *representation* may change — the guarantee may not.
