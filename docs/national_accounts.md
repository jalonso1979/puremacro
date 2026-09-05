> 🇬🇧 English · 🇪🇸 [Español](es/national_accounts.md)

# Quarterly National Accounts

`puremacro.fetch` builds a quarterly national accounts panel from the OECD's
SDMX service and — the part that usually gets hand-rolled in every notebook
that touches one — does the work that comes *after* the fetch: putting
countries on a common price reference year, scoring the accounting identities,
and decomposing growth into contributions.

Nothing on this page except `qna_panel`, `qna_labor` and `qna_countries`
touches the network. A live panel and a frozen CSV of one behave identically.

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

## `assets=True` and `durability=True` — two splits of the same totals

Neither adds an approach; each opens up a total the expenditure block already
carries, and both are joined with their own deflators.

| flag | splits | into | registry |
|---|---|---|---|
| `assets=True` | `inv` (gross fixed capital formation) | `inv_equip`, `inv_struct`, `inv_dwell`, `inv_ipp` | `QNA_ASSETS` |
| `durability=True` | household consumption | `cons_dur`, `cons_semidur`, `cons_nondur`, `cons_serv` | `QNA_DURABILITY` |

Two things to know before using them:

- **The durability split is a different institutional sector.** The headline
  `cons_hh` is `S1M` — households *plus* NPISH. The durability columns are
  `S14`, households only. They will not sum to `cons_hh`, and the gap is
  NPISH, not an error.
- **Several countries publish these splits unadjusted while their headline
  aggregates are adjusted** — Mexico, Japan and Turkey among them. `qna_meta`
  reports this in `sa_detail`, separately from `sa`, and `sa="x13"` adjusts
  them here. This is the reason `sa_detail` exists as its own column.

Durables are the part the national accounts book as consumption but that macro
theory treats as household capital (Cooley & Prescott 1995; Gomme & Rupert
2007), which is why the split is worth having rather than a curiosity.

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

### The ISIC breakdown: `labor_activities=True`

`qna_panel(..., labor=True, labor_activities=True)` — or `qna_labor(...,
activities=True)` — returns every stem again per ISIC activity, so `hours`
gains `hours_agri` and `hours_public` and the block goes from six columns to
eighteen. The two activities are registered in `QNA_LABOR_ACTIVITIES`:
agriculture (`A`), where most of the labour is self-employed and most of the
year-to-year output is the weather, and public administration, education and
health (`OTQ`), whose value added the SNA *defines* as compensation plus
consumption of fixed capital, so its measured productivity growth is near zero
by construction rather than by finding.

Subtracting the two is the whole point:

```python
market_hours = p["hours"] - p["hours_agri"] - p["hours_public"]
```

That turns a whole-economy `Y/H` into the **market sector** — the concept the
United States publishes as its nonfarm business sector, and the only basis on
which it can be compared with anyone else. The whole economy (`_T`) is always
requested alongside the parts, because a part without its whole is not usable,
and every activity of one unit of measure is resolved as a single seasonal
family, since a source-adjusted total minus a raw part is not a subtraction of
anything. It costs nothing extra — the same request, three activities instead
of one — and 34 reference areas publish hours for all three.

`labor_activities` requires `labor=True`, and raises if it is off rather than
returning a panel without the columns you asked for.

For the **annual** counterpart — which is the only place the United States and
Japan have a national-accounts denominator at all, because the quarterly flow
returns zero rows for both — see `ana_by_activity` in
`puremacro.fetch.oecd_ana_activity`.

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
  labelled exactly like everyone else's quarterly figure — a Chilean quarter
  reads ~41 hours per worker and a Costa Rican one ~2,157, against ~430 for a
  normal one. Both are put back on a quarterly basis, and `qna_meta`'s
  `hours_scale` records the factor (`13.0`, `0.25`, or `1.0` for untouched).
  Detection is by the level: a country is only rescaled if its median
  `hours / emp` is outside 150–1000 *and* a candidate factor lands it inside
  250–700. Every observation the 31 same-basis countries have ever published
  sits in 304–572, so the band cannot fire on a country that merely works
  short weeks. Canada publishes hours but no heads, so nothing can be checked
  and its hours are left alone. `hours_rescale=False` returns the numbers
  exactly as the OECD sends them.
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

## `qna_labor()` — the labour block without the accounts

`qna_panel(..., labor=True)` joins the labour block onto the national
accounts, which means it also downloads the expenditure block and, under
`sa="x13"`, seasonally adjusts it. When the labour series are all you want,
that is three extra SDMX round-trips per chunk of ten countries and a lot of
X-13 you did not ask for:

```python
from puremacro.fetch import qna_labor

lab = qna_labor(["DEU", "MEX", "KOR"], start="1995", sa="x13")
```

It returns a **long** frame — `code`, `date`, `variable`, `value`,
`sa_source` — carrying the same columns (six by default, eighteen with
`activities=True`), in the same units, with the same hours correction and the
same family-level seasonal adjustment. Two
differences from the joined route are deliberate:

- **`sa_source` is per series, not per country.** A country adjusted at source
  for heads but not for hours gets an honest label on each, where `qna_meta`'s
  `sa_labor` can only report `mixed` for the block as a whole.
- **It does not depend on the expenditure block at all.** In the joined route
  the labour rows are filtered to the countries the expenditure flow returned,
  so a country publishing labour but no expenditure is dropped and a failed
  expenditure request takes the labour rows with it. That is the right
  behaviour when you asked for a national-accounts panel and the wrong one
  when you asked for employment.

This is what `build_panel.build_all` uses to fill the labour gaps its local
workbook does not cover.

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

## `qna_capital()` and `qna_tfp()` — growth accounting

`puremacro.capital` turns the panel into the two inputs a production function needs
and the residual they leave over. The package could already *estimate* production
functions — `korv_gmm` fits a capital-skill-complementarity CES — but it could not
*build* the capital series they take as given.

```python
from puremacro.fetch import qna_panel
from puremacro.capital import qna_capital, qna_tfp

panel = qna_panel(["DEU", "ESP", "FRA", "ITA"], start="1995",
                  assets=True, labor=True, income=True, real=True)
cap = qna_capital(panel)          # perpetual inventory, one stock per asset
tfp = qna_tfp(panel, cap)         # Solow residual against hours
```

Two choices are load-bearing:

- **Depreciation converts geometrically**, `1-(1-δ)^(1/4)`, never `δ/4`. The linear
  shortcut compounds to less than `δ` over four quarters, so it under-depreciates and
  biases the steady-state stock **up** — by 5.3% for equipment and 8.5% for IPP.
- **The aggregate is a Törnqvist index of capital services, not a sum of stocks.**
  Chain-linked volumes are not additive away from their reference year, so `Σ K_i`
  depends on which year the OECD chose; `qna_rebase` the same panel and it moves by up
  to 1%. Aggregating *growth rates* with rental-cost weights is invariant to machine
  precision. Use `aggregate="sum"` only for the fixed-base publishers (ARG, IDN, MEX,
  ZAF), where additivity holds by construction.

**Read `k0_sensitivity` before quoting a level.** The initial stock has to be assumed
and the assumption decays at the asset's own depreciation rate — fast for equipment
(δ=0.13) and IPP (0.20), not for structures (0.03) or dwellings (0.011). Measured on
this data, a ±50% error in `K₀` is still worth 7–15% of the aggregate *level* at the end
of a 30-year panel, while the same error moves four-quarter *growth* by under 0.06pp a
year. Growth rates are usable; levels are an assumption wearing a number's clothes.

Coverage: 34 of 49 reference areas publish all four asset classes as volumes, 33 also in
current prices (Colombia does not, so it has no deflators and is refused a services
index), and 29 additionally publish the hours `qna_tfp` needs.

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
