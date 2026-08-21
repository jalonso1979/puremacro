# Quarterly National Accounts

`puremacro.fetch` builds a quarterly national accounts panel from the OECD's
SDMX service and — the part that usually gets hand-rolled in every notebook
that touches one — does the work that comes *after* the fetch: putting
countries on a common price reference year, scoring the accounting identities,
and decomposing growth into contributions.

Nothing on this page except `qna_panel` and `qna_countries` touches the
network. A live panel and a frozen CSV of one behave identically.

```python
from puremacro.fetch import (
    qna_panel, qna_countries, qna_rebase, qna_identity, qna_contributions,
)

panel = qna_panel(["USA", "JPN", "DEU"], start="1995",
                  output=True, income=True, real=True)
```

Notebook 40 (`notebooks/40_quarterly_national_accounts.py`) works through all
of it on a frozen six-country panel.

## The three products of a build

`qna_panel` returns current-price levels, implicit deflators and — with
`real=True` — volume measures, tied by

```
nominal = real x deflator / 100
```

component by component. Every transform below preserves that identity exactly.

## Three approaches to GDP

The OECD publishes three separate measurements of the same quantity, from three
separate source systems. `qna_panel` reaches all of them:

| flag | approach | identity | registry |
|---|---|---|---|
| *(always)* | expenditure | `Y = C + G + I + X - M` | `QNA_COMPONENTS` |
| `output=True` | output (production) | `Y = Σ VA_j + (D21 - D31) + YA1` | `QNA_ACTIVITIES` |
| `income=True` | income | `Y = D1 + B2A3G + (D2 - D3)` | `QNA_INCOME` |

Two caveats the registries encode:

- **Two of the fifteen value-added columns are memo items, not addends.**
  `va_mfg` sits inside `va_ind`, and `va_services` aggregates seven columns
  already listed. `QNA_VA_ADDITIVE` names the ten that sum to the total;
  `QNA_VA_MEMO` names the two that do not. Adding every `va_*` column
  double-counts about a third of the economy.
- **Income columns exist only in current prices.** There is no volume measure
  of compensation of employees, so they carry no deflator and no `_real`
  column even under `real=True`.

Coverage is not uniform. Of the OECD's 49 reference areas, 46 appear in the
by-activity output flow — the United States is absent from it entirely, since
the US industry accounts are a separate BEA release — and 40 in the income
flow. A country that does not publish an approach reads `NaN`.

## `labor=True` — the labour input behind those flows

The same accounts measure the labour that produced the output: employment and
hours worked, each split into employees and the self-employed, on the
**domestic** concept — labour in resident production units, which is the
concept GDP is measured on, so `gdp_real / hours` is a productivity measure
rather than two different populations divided by each other.

| column | what it is |
|---|---|
| `emp`, `emp_employees`, `emp_selfemp` | persons, thousands |
| `hours`, `hours_employees`, `hours_selfemp` | hours worked, millions |

Registered in `QNA_LABOR`, with the scales in `QNA_LABOR_UNITS`. The block
carries no price, so it gets no deflator and no `_real` column — `real=True`
does not change that.

Two ratios are the point of the split:

- **`emp_selfemp / emp`** — the share of the workforce whose labour income the
  accounts book inside `surplus_mixed` rather than `comp_emp`. This is the
  missing input for the labour-share correction at the end of this page.
- **`hours / emp`** — average hours per worker, the margin that carries a
  German recession (2008–10: employment −0.3%, hours −3.7%) where a Spanish
  one is carried by heads instead (−9.9% against −9.8%).

**Coverage is the raggedest of any block on this page**, and worth checking
before dividing one of these columns by another:

- 38 of the 49 reference areas publish heads and 34 publish hours. The United
  States, Japan, Argentina, Brazil, China, Colombia, Indonesia, India, Saudi
  Arabia and Turkey publish nothing in this flow at all — at any level of
  activity, so no choice of key recovers them. Canada publishes hours without
  heads; Australia, Switzerland, Korea, Russia and South Africa publish heads
  without hours.
- **Chile reports hours per week and Costa Rica at an annual rate**, both
  labelled exactly like everyone else's quarterly figure. A Chilean quarter
  reads ~41 hours per worker and a Costa Rican one ~2,157, against ~540 for a
  normal one. Put a plausibility band on `hours / emp` before using it.
- Ten reference areas publish the block with no adjusted variant at all,
  several of them while publishing adjusted *aggregates*. `sa="x13"` adjusts
  them here, and the meta column `sa_labor` reports who did the adjusting,
  separately from `sa` (headline aggregates) and `sa_detail` (asset and
  durability splits).
- Korea adjusts the employment total at source but not its two components.
  Heads and hours are each resolved as one family, so Korea falls back to the
  raw series for all three and the decomposition still adds up; the price is
  that `sa_labor` reads `none` for Korea under the default. Taking the
  adjusted total with raw parts would put a 1.1pp seasonal artefact straight
  into `emp_selfemp / emp`.

## `qna_countries()` — ask the source what it carries

Queries the SDMX **availability** endpoint for a dataflow's reference areas, so
a panel covers what the source actually supports rather than a hand-typed list
that goes stale the next time the OECD onboards a country.

Country groupings (`OECD`, `EA20`, `G7`, …) are dropped by default — see
`QNA_AGGREGATES` — because a panel that silently mixes `OECD` in with `USA`
double-counts every aggregate it touches. Falls back to a frozen list when the
endpoint is unreachable: it never raises and never returns empty.

## `qna_rebase(panel, year)` — one price reference year

The OECD references each country's volumes to *that country's* base year: 2017
for the United States, 2018 for Mexico, 2020 for Spain. Raw deflator columns
are therefore indices against different years, and comparing them across
countries as levels compares nothing.

`qna_rebase` rescales each country's deflators to the requested year and its
volumes by the same factor, so `nominal = real x deflator / 100` still holds
exactly. This is a re-**referencing** — one scalar per country, leaving every
growth rate and every chain link untouched — not a re-basing, which a published
chain-linked volume does not permit at all.

## `qna_identity(panel)` — score the identities, separately

Returns one row per country. The residuals are reported apart from each other
because they are different things:

| column | what it is |
|---|---|
| `nominal_*` | **statistical discrepancy** in current prices. Often forced to zero — a presentation choice. Where it is not zero, independent seasonal adjustment of each series is a large part of the reason, visible as a sign-alternating residual. |
| `real_*` | **chain-linking gap** in volume terms. Not a data error: chain-linked volumes are not additive away from the reference year, and the gap widens with distance from it. |
| `output_*`, `income_*` | each approach scored against **its own flow's GDP** |
| `crossflow_output`, `crossflow_income` | the disagreement *between* flows |

That last row matters. The OECD publishes GDP separately in each flow and the
figures do not always agree — Japan's output-flow GDP differs from its
expenditure-flow GDP by up to 0.61%, Germany's income-flow GDP by up to 1.77%.
Scoring an approach against a *different* flow's GDP would charge that
disagreement to the approach's own components, so `APPROACH_GDP` carries
`gdp_output` and `gdp_income` as their own columns and the disagreement gets
its own report. Across the full 49-country panel the output identity closes to
a median 0.001% of GDP and the income identity to 0.003%.

The income residual is the one with a name and a literature: the **GDP–GDI
statistical discrepancy**, which for the United States runs to ±2% of GDP and
is informative about the business cycle in its own right (Nalewaik 2010). Most
European offices force it to zero.

## `qna_contributions(panel)` — decompose growth

Real GDP growth, split into what each component contributed, with
previous-period **nominal** weights:

```
g_t = Σ_i ω_{i,t-1} g_{i,t},    ω_{i,t-1} = (P_i Q_i)_{t-1} / (P Q)_{t-1}
```

This is the calculation chain-linking requires, and one that needs all three
products of a build at once: volumes for the growth rate, current prices for
the weight. Imports enter negatively. `annualise=True` rescales the parts along
with the aggregate. What the weights do not span is returned as an explicit
`residual` column rather than spread over the components.

## A note on the labour share

`comp_emp / gdp` is the **unadjusted** labour share, and the adjective is
load-bearing: the income of the self-employed is not in `D1` at all but inside
`surplus_mixed`. A country with a large self-employed sector reads low for
reasons that have nothing to do with how its employees are paid. See Gollin
(2002).

The correction needs a split of the workforce, which is what `labor=True`
supplies. With all three blocks on, the panel carries every column
`puremacro.labor_share.gollin_adjusted_ls` asks for:

```python
panel = qna_panel(["ESP", "ITA", "DEU"], start="1995",
                  output=True, income=True, labor=True)
```

| `gollin_adjusted_ls` wants | panel column | from |
|---|---|---|
| `compensation_employees` | `comp_emp` | `income=True` |
| `mixed_income` | `surplus_mixed` | `income=True` |
| `value_added` | `va_total` | `output=True` |
| `employment_employees` | `emp_employees` | `labor=True` |
| `employment_self` | `emp_selfemp` | `labor=True` |

Before this, no single source in `puremacro` carried all five.

## Offline use

The four transforms never touch the network, so the usual pattern is to fetch
once, freeze, and work from the snapshot:

```python
panel.reset_index().to_csv("qna.csv", index=False)          # freeze
flat = pd.read_csv("qna.csv")
panel = (flat.assign(date=pd.to_datetime(flat["date"]))
             .set_index(["code", "date"]).sort_index())     # thaw
```

One catch: `qna_meta()` reads `panel.attrs`, which no CSV round trip preserves.
Freeze `qna_meta(panel)` as its own file if you need the provenance — which you
do if you plan to rebase, since it is what records each country's base year.
`tools/gen_notebook_data_qna40.py` is a worked example of the whole pattern.
