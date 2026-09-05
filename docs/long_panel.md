> 🇬🇧 English · 🇪🇸 [Español](es/long_panel.md)

# The long national accounts panel

`qna_panel` gives you the OECD's quarterly national accounts, which for
most of Europe begin in 1995. `qna_long_panel` extends that spine
backwards, per country, by ratio-splicing archived national vintages
onto it.

```python
from puremacro.fetch import qna_long_panel

long, seams = qna_long_panel(["ESP", "JPN"], return_seams=True)

long.loc["ESP"].index.min()   # 1970-01-01   (OECD: 1995-01-01)
long.loc["JPN"].index.min()   # 1955-04-01   (OECD: 1994-01-01)
```

The column schema is exactly `qna_panel`'s, so `qna_identity`,
`qna_rebase` and `qna_contributions` work unchanged. Alongside each
value column sits a `src_<column>` recording which vintage produced
each quarter.

| country | reaches | gain | sources |
|---|---|---|---|
| **Spain** | 1970Q1 | +100 quarters | INE base-1995 tables (JSON API) + the base-1986 workbook |
| **Japan** | 1955Q2 | +155 quarters | Cabinet Office 93SNA and 68SNA archived releases |

## What a ratio splice does, and does not, preserve

The only thing worth keeping from an abandoned vintage is its **growth
rates**. Its levels are expressed in a base and a methodology that were
later replaced, so carrying them over unchanged would put a step in the
series at the seam. The older segment is therefore rescaled by the
ratio between the two vintages over their overlap, and only then used
to extend the newer one backwards.

A useful consequence: any **constant** factor is absorbed exactly and
silently. Japan's ESRI files are in billions of yen at annual rates
while the OECD spine is in millions per quarter — a factor of 250 — and
nothing has to special-case it. Rebasings and currency redenominations
behave the same way.

## The ratio's stability is the test

What a rescaling *cannot* absorb is a ratio that drifts across the
overlap. That means the two vintages disagree about growth itself, not
merely about level, and the spliced level then depends on which quarter
you happened to anchor to — a result about your arbitrary choice rather
than about the economy.

So drift is measured and reported per column:

```python
seams[~seams.stable][["code", "column", "older", "overlap_n", "ratio_drift"]]
```

The answer differs by column, and only the seam table will tell you:

- **Spain** — GDP and household consumption hold steady across both
  seams (0.9–1.3% drift). Capital formation does not: 5.1% drift, with
  the ratio ranging 1.04–1.39 across 76 overlapping quarters. Spanish
  GDP back to 1970 is defensible; Spanish investment back to 1970 is
  indicative at best.
- **Japan** — everything holds except government consumption at the
  68SNA seam (5.3%), which is unsurprising: 68SNA defined the
  government sector differently.

Nine of twenty-eight seams are unstable. `qna_long_panel` warns and
names the worst; `drop_unstable=True` blanks those quarters instead if
you would rather have a gap than a number you might not notice is shaky.

## The identity does not close, on purpose

Columns are spliced independently, each keeping its own source's growth
rates, so `C + G + I + X − M` no longer equals spliced GDP:

```python
from puremacro.fetch import long_panel_residual
(long_panel_residual(long) / long["gdp"]).abs().mean()   # ~0.33%
```

Forcing it to zero — by splicing GDP and component *shares* and
rebuilding levels — would produce a tidier panel that no agency ever
published, and would have buried the investment problem above inside
whichever series was designated the residual. Measuring the gap is the
more useful answer.

Within each source vintage the identity closes **exactly**, which is
what validated the column mappings in the first place — including the
discovery that Spanish `cons_hh` must be households **plus NPISH** to
match the OECD's S1M sector.

## What is refused

Three situations raise rather than returning a number:

- **no overlap** — there is nothing to estimate a ratio from, and
  concatenating two segments puts an unmeasured step in the series;
- **fewer than four overlapping quarters** — the ratio cannot be told
  apart from a one-off revision;
- **a definitional break** — a change in what is being measured rather
  than a revision of it.

Germany is the standing example. Its pre-1991 nominal accounts cover
**West Germany**; rescaling those onto unified Germany would fabricate
an East German economy back to 1970. `ratio_splice` refuses unless the
caller passes `allow_definitional_break=True` and takes responsibility.
Germany *can* reach 1970 in volumes, because the Bundesbank has already
done that linking officially over the 1991 annual average and publishes
the result (`BBNZ1 Q.DE.Y.H.*.L`).

## Why only two countries

Because only two archived sources were measured to reach further back
than the OECD already does. Seven candidates buy **zero** quarters, and
the reasons are kept in the code rather than lost:

```python
from puremacro.fetch import LONG_PANEL_KNOWN_GAPS
LONG_PANEL_KNOWN_GAPS["MEX"]
```

- **Eurostat** publishes the same numbers as the OECD — the splice
  ratio is 1.0 — beats it for no country, is *shorter* for Denmark
  (1995 vs 1990) and Switzerland (1995 vs 1980), and has no United
  Kingdom row at all, where the OECD has the UK from 1960.
- **IMF** ties the OECD on seven of eight economies checked and is 12
  quarters shorter on the United States.
- **INEGI** serves nothing before 1993 and is identical to the OECD
  after dividing by four. **IBGE** ties the OECD at 1996Q1 — a hard
  floor across all seven candidate tables.
- **CEPALSTAT**'s Mexican 1980–92 join has a ratio drifting 0.754–0.837
  across the overlap, which is exactly the condition this module
  refuses to call a splice.

"We checked and it buys nothing" is a different statement from "we did
not check", and the difference is worth keeping.
