# ADR-0030 — Model cost: prices are dated observations, not model properties

**Status:** Accepted (2026-08-22)
**Related:** [ADR-0016](0016-unavailable-is-not-zero.md) (unavailable is not zero),
[ADR-0008](0008-canonical-model-identity.md) / [ADR-0024](0024-canonical-id-and-model-references.md)
(what a model identity is and is not), [ADR-0017](0017-benchmark-confidence-and-freshness.md)
(the routing score's cost factor).

## Context

LoadCoach's routing score multiplies capability evidence by "reliability, availability and cost
factors" ([master architecture §routing](../architecture/master-architecture.md)), FreeWeight wants
to report what a benchmark run cost, and IdeaPress wants to show what a piece of content cost to
produce. All three need one answer to "what does a token cost here?", and none of them can get a
trustworthy answer from the provider at the moment they need it.

The properties that make model cost hard are not incidental — they are the whole problem:

* **The price is not ours.** A provider changes its price list on its own schedule, with no
  notification, and applies the change to traffic we have already planned. The number we hold is a
  copy of something that has since moved.
* **The price is not single-valued.** The same weights cost different amounts on the standard tier,
  the batch tier and the priority tier; in one region versus another; under a negotiated agreement
  versus the public list; during a promotional window versus after it. Cached-input tokens are
  billed at a different rate again, and the definition of a cached token differs between providers.
* **The price is often absent.** A model served by Ollama on the user's own hardware has no token
  price at all. Neither does a provider that publishes no list, nor a model behind a gateway that
  strips the usage block from the response.
* **The price arrives late.** A provider's invoice is authoritative and appears weeks after the run
  it describes. Anything we compute before then is an estimate, however precise it looks.
* **A stale price is indistinguishable from a fresh one** once it has been reduced to a number in a
  column. `$0.42` in a results table carries no evidence of the price list it came from, when that
  list was read, or whether that list was still in force on the day of the run.

A naive implementation — a `price_per_token: float` field on the model descriptor — fails on every
one of these, and fails silently. The failure mode is precisely the one
[ADR-0016](0016-unavailable-is-not-zero.md) exists to prevent: a plausible number nobody measured.

## Decision

**Cost is derived, never stored as a fact. A price is a dated, sourced observation with a validity
window; an unknown price is `UNSUPPORTED`, never zero; and money is exact integers, never floats.**

Seven rules, all enforced by BaseAiCore's types rather than by convention.

### 1. Store usage; derive cost

A run, a job and a request store **`TokenUsage`** — the token counts — plus the `pricing_hash` of the
price record that was applied at the time. They do not store a money figure as their primary record
of cost.

This is the rule that makes every other one survivable. When a provider corrects a price, when a
price list is read for the first time six months after a run, or when a costing bug is found, the
history is re-costed from the counts. Had the money figure been the stored fact, the correction
would have had nowhere to go and the history would have been quietly wrong forever.

### 2. Money is an exact integer in a named currency

```python
Money(currency="USD", nanos=3_000_000_000)     # $3.00
```

`nanos` are billionths of one currency unit. Token prices are quoted per **million** tokens, so
prices are stored per million tokens and never pre-divided into a per-token fraction — `$0.019/1M`
is `19_000_000` nanos per million tokens exactly, not `1.9e-8` dollars per token approximately.

Floats are rejected outright: they do not sum associatively, they do not compare reliably, and
`0.1 + 0.2` in a cost column is the same category of defect as a fabricated measurement.

### 3. No currency conversion, ever, in this layer

Cross-currency arithmetic and cross-currency comparison raise `ValidationError`. Converting requires
an exchange rate, which is time-varying external data outside the user's control — exactly the thing
this ADR says must not be silently assumed. A consumer that genuinely needs one currency converts at
a rate it obtained and recorded itself, and it is then that consumer's number.

### 4. A price is an observation, with provenance and a window

```python
ModelPricing(
    identity=...,                 # which weights, on which provider kind
    rates=TokenRates(...),        # per-million-token prices, per billable token class
    source=PricingSource.PROVIDER_PUBLISHED,
    observed_at=...,              # when we learned it
    effective_from=..., effective_until=...,   # the provider's stated window, when known
    price_tier="batch", region="eu-west-1",    # the dimensions the price varies on
)
```

`source` distinguishes a price the provider returned with the response from one scraped out of a
documentation page from one the user typed into a configuration file. `price_tier` and `region` name
the dimensions along which the same model legitimately has several different prices, so a catalogue
holding all of them is not a set of contradictions.

`pricing_hash` (canonical JSON → SHA-256, first 16 hex characters) identifies **the price**, not the
reading of it: `observed_at` is deliberately excluded, so re-reading an unchanged price list yields
the same hash and a stored cost can name exactly which price produced it.

### 5. Costing outside the stated window yields `UNSUPPORTED`, with a reason

If a price record states a validity window and the instant being costed falls outside it, the result
is unsupported and the reason says so. It is not extrapolated. A price record with no stated window
is treated as applicable — we know the provider did not tell us, and saying so is different from
inventing an end date.

### 6. An absent price is `UNSUPPORTED`; a local model is not free

A model with no token price is not a model that costs `$0.00`. It costs electricity, hardware
depreciation and wall-clock time — quantities SweatMeter measures and this package does not price.
Zeroing it would make it win every "cheapest model" comparison in the suite by default, which is the
single most consequential form the zero-instead-of-unsupported bug could take here.

The same applies per token class: a run that read 12 000 cached tokens under a price list that
does not state a cached-read rate has an unsupported total, and `CostEstimate.unpriced_reasons`
names the rate that was missing. A partial sum is never presented as a total. A token class with a
count of **zero** contributes exactly zero regardless of its rate, and is the one case where an
absent rate is harmless.

### 7. The result is named an estimate, because that is what it is

`estimate_cost()` returns a `CostEstimate`, which carries the pricing source, the pricing hash, the
instant it was priced at and the reasons for anything it could not price. The provider's invoice
remains authoritative. No API, export or UI in the suite presents a computed figure as a billed
amount.

### What this ADR does not decide

BaseAiCore defines the vocabulary and the arithmetic. It does not acquire prices: there is no
bundled price catalogue, no HTTP call and no configuration file. Compiling a price list into a
zero-dependency domain package would ship prices that are stale on the day of release with the
authority of code. Acquisition — provider response, published list, user override — belongs to
ModelRack and to the applications, which record a `PricingSource` when they hand a record over.

## Alternatives considered

**`price_per_token: float` on `ModelDescriptor`.** The obvious minimal change. Rejected on every
count above: floats drift, a descriptor is refreshable provider metadata with a different lifetime
and owner than a price, and a single field cannot express tier, region, window or provenance. It
also makes the price look like a property of the model, which is the misconception this ADR exists
to correct.

**`Decimal` as the stored representation.** Genuinely exact, and standard library. Rejected as the
*stored* form: `Decimal("3.0")` and `Decimal("3.00")` are equal but serialize differently, so a
canonical hash over a price record would depend on how the value was typed; precision is
context-dependent and inherited from a thread-local; and equality versus `compare_total` is a trap.
`Decimal` is used at the parse and display boundary — `Money.from_decimal()` and `Money.to_decimal()`
— where its ergonomics are worth having and its representation does not reach a hash.

**Money as minor units (cents).** The usual accounting answer. Rejected: two decimal places cannot
express a per-million-token price without either scaling the unit anyway or losing the cheapest
models entirely, and the scaling factor would then be implicit. Nanos state it in the field name.

**Zero for models with no price.** Rejected — see rule 6. It is ADR-0016's bug class, aimed at the
comparison that decides which model the suite recommends.

**A bundled price catalogue, refreshed at release.** Rejected: prices change on a weekly cadence and
releases do not. A catalogue in the package would be authoritative-looking and wrong, and it would
put network-shaped data in the one package that is defined by having no I/O.

**Automatic currency conversion with a bundled rate table.** Rejected for the same reason, more
strongly: an FX rate is stale within the hour.

**Recording the computed money figure as the primary fact.** Rejected — see rule 1. It is
unrepairable after a price correction.

## Consequences

*Positive.* A cost figure in this suite can always answer "from which price, read when, from where,
valid over what window?". A price correction re-costs history instead of corrupting it. Local models
are never mistaken for free ones. Cross-currency mistakes fail loudly at the arithmetic rather than
quietly in a chart. The arithmetic is exact, so a total and its components always agree.

*Negative.* Callers must handle `UNSUPPORTED` at every cost site, and must supply a `ModelPricing`
rather than reading a number off the model. That is the intended cost, and it is the same shape as
the cost ADR-0016 already imposes.

*Negative.* Every consumer that stores a cost must also store `TokenUsage` and a `pricing_hash`,
which is three columns where a naive design has one. The database standards' identity columns
already set this precedent.

*Negative.* The suite ships no prices, so cost is unsupported everywhere until an application
supplies a price record. This is honest and is the correct default, but it means the cost column is
empty on first run and the UI must say why rather than showing `0`.

*Negative.* Providers disagree on whether cached-input tokens are counted inside or outside the
prompt-token figure. `TokenUsage` fields are defined as **disjoint**, which pushes the reconciliation
into ModelRack's provider adapters — the only layer that knows each provider's convention. Getting
that wrong there double-counts, so it is a named test case in the provider conformance suite.

## Revisit when

* A provider the suite supports bills for a unit that is not a token — per request, per image, per
  audio second, per tool call. `TokenRates` gains a sibling rate type and `estimate_cost` gains that
  term; the surrounding provenance model is unaffected.
* A supported provider exposes an authoritative per-request billed amount in its response, at which
  point `PricingSource.PROVIDER_RESPONSE` records become authoritative rather than estimated and a
  `BilledAmount` type may be warranted alongside `CostEstimate`.
* Two consumers independently implement price-list acquisition, which is the
  [ADR-0011](0011-shared-package-boundaries.md) trigger for extracting a catalogue component.
