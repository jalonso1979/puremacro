> 🇬🇧 English · 🇪🇸 [Español](es/did.md)

# Modern Difference-in-Differences (DiD)

Classical Two-Way Fixed Effects (TWFE) regressions of the form:

$$y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$

fail when treatment timing is **staggered** (units are treated at different points in time) and treatment effects are **heterogeneous** across cohorts or dynamic over time (Goodman-Bacon 2021, de Chaisemartin & D'Haultfœuille 2020). Under heterogeneous effects, TWFE implicitly subtracts already-treated units as controls for later-treated units, generating negative weights that can invert the sign of the true treatment effect.

`puremacro.did` implements the modern suite of heterogeneity-robust DiD estimators in pure Python (numpy / scipy / pandas only), providing panel-bootstrap inference, dynamic event-study aggregations, and publication-ready table exports.

---

## Overview of Estimators

| Estimator | Function | Key Reference | Strategy |
|---|---|---|---|
| **Callaway & Sant'Anna** | `callaway_santanna` | Callaway & Sant'Anna (2021, *J. Econometrics*) | Group-time $ATT(g, t)$ with clean control cohorts (never-treated or not-yet-treated) |
| **Sun & Abraham** | `sun_abraham` | Sun & Abraham (2021, *J. Econometrics*) | Interaction-weighted event study using cohort shares |
| **Borusyak, Jaravel & Spiess** | `borusyak_jaravel_spiess` | Borusyak, Jaravel & Spiess (2024, *Rev. Econ. Stud.*) | Imputation estimator: fits FE on untreated cells and projects counterfactuals |
| **de Chaisemartin & D'Haultfœuille** | `cdh_did` | de Chaisemartin & D'Haultfœuille (2020, *AER*) | $DID_M$ / $DID_M^\ell$ switchers estimator with a placebo test |
| **Synthetic DiD** | `synthetic_did` | Arkhangelsky, Athey, et al. (2021, *AER*) | Double-weighting: unit weights $\omega$ match pre-trends + time weights $\lambda$ |
| **Multi-Cohort SDID** | `sdid_multi_cohort` | Arkhangelsky et al. (2021); Roth et al. (2023, *J. Econometrics*) | Per-cohort `synthetic_did` on an untreated donor window, cohort-size weighted |

Two input conventions coexist. `callaway_santanna`, `sun_abraham`, `borusyak_jaravel_spiess` and `synthetic_did` take a **long-format DataFrame** with column names passed as `unit=`, `time=`, `outcome=`, `treat_time=` (`treat_time` is the per-unit first-treatment period, `NaN` for never-treated units). `cdh_did` and `sdid_multi_cohort` take **four aligned 1-D arrays** `(y, treatment, panel_id, time_id)` where `treatment` is the 0/1 treatment status of each row.

---

## 1. Callaway & Sant'Anna (2021)

Estimates average treatment effects for each cohort $g$ (adoption year) at each calendar period $t$, denoted $ATT(g, t)$, against the universal base period $g - 1$. It then aggregates these into an event-study profile tracking dynamic impacts $e = t - g$. The block below builds a synthetic staggered panel that every later block reuses:

```python
import numpy as np
import pandas as pd
from puremacro.did import callaway_santanna

# Synthetic staggered panel: 60 counties x 12 years. Cohorts adopt in 2004
# and 2007, a third group never adopts. True effect = 1.0 + 0.2 * (years
# since adoption); outcomes carry county and year fixed effects.
rng = np.random.default_rng(0)
cohorts = np.array([2004.0, 2007.0, np.nan])
rows = []
for i in range(60):
    g = cohorts[i % 3]
    alpha_i = rng.normal()
    for year in range(2000, 2012):
        e = year - g if not np.isnan(g) else -1.0
        tau = 1.0 + 0.2 * e if e >= 0 else 0.0
        rows.append({
            "county_id": i, "year": year, "first_treated_year": g,
            "employment": alpha_i + 0.1 * (year - 2000) + tau + rng.normal(scale=0.3),
        })
df = pd.DataFrame(rows)

res_cs = callaway_santanna(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control="never_treated",   # or "not_yet_treated" (control_group= is accepted as an alias)
    n_boot=500,
    ci=0.95,
)

# 1. View summary of results
print(res_cs.summary())

# 2. Inspect event study aggregation: columns [event_time, att, se, lo, hi, n_cohorts]
print(res_cs.att_event_study.head())

# 3. Export table directly to LaTeX or Typst
print(res_cs.to_latex())
print(res_cs.to_typst())
```

### Key Attributes of `CallawaySantannaResult`

- `att_gt`: DataFrame of cohort-time estimates $ATT(g, t)$ with bootstrap standard errors (columns `g, t, event_time, att, se, lo, hi`).
- `att_event_study`: Aggregated dynamic effects relative to treatment timing ($e = -K \dots +L$); each event time is the unweighted mean over the cohorts that identify it (`n_cohorts`).
- `att_overall`: Simple mean of the post-treatment $ATT(g, t)$ cells (every identified cohort-period cell counts once; it is *not* weighted by cohort size — use `sun_abraham` for a unit-share-weighted overall effect).
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Publication table renderers of the event study (no index column); `.plot()` draws it with its confidence band.

---

## 2. Sun & Abraham (2021)

Sun & Abraham explicitly model cohort-specific paths and weight dynamic coefficients by each cohort's sample share. This ensures that dynamic estimates at horizon $e$ are not distorted by compositional changes in which cohorts identify the effect. `att_overall` is the cohort-size-share-weighted mean of the post-treatment $ATT(g, t)$:

```python
from puremacro.did import sun_abraham

res_sa = sun_abraham(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    ci=0.90,
)

print(res_sa.summary())
print(res_sa.to_markdown())
```

---

## 3. Borusyak, Jaravel & Spiess (2024) Imputation

The BJS imputation estimator is asymptotically efficient under parallel trends. It works in three intuitive steps:
1. **Fit**: Estimates unit and time fixed effects using *only* untreated observations ($D_{it} = 0$: never-treated units and the pre-treatment rows of eventually-treated units).
2. **Impute**: Projects counterfactual outcomes $\hat{y}_{it}(0)$ for treated observations.
3. **Average**: Computes treatment effects as $\hat{\tau}_{it} = y_{it} - \hat{y}_{it}(0)$ and aggregates across event times ($e \ge 0$ only — BJS evaluates $\hat\tau$ on treated cells, so there are no pre-trend rows). `att_overall` weights every treated cell equally.

A treated cell is identified only if its period and its unit each have at least one untreated observation. In a panel **without never-treated units** every period from the last cohort's adoption onwards has no untreated observation, so its time fixed effect cannot be estimated: `borusyak_jaravel_spiess` raises a `ValueError` naming those periods by default, and `unidentified="drop"` warns and excludes those cells from every aggregate instead.

```python
from puremacro.did import borusyak_jaravel_spiess

res_bjs = borusyak_jaravel_spiess(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    n_boot=500,
)

print(res_bjs.summary())
print("Overall ATT:", res_bjs.att_overall)
```

---

## 4. de Chaisemartin & D'Haultfœuille (2020) $DID_M$

`cdh_did` compares units that **switch** into treatment between $t-1$ and $t$ with units whose status is stable at 0 over the same window, avoiding the negative TWFE weights. It reports the instantaneous $DID_M$, long-run $DID_M^\ell$ for horizons $\ell$, and a switchers placebo $p$-value (pre-switch trend of switchers vs. stable units). This estimator uses the four-array convention:

```python
from puremacro.did import cdh_did

# 0/1 treatment status per row (NaN first_treated_year compares as False -> 0)
treated = (df["year"] >= df["first_treated_year"]).astype(int).to_numpy()

res_cdh = cdh_did(
    df["employment"].to_numpy(), treated,
    df["county_id"].to_numpy(), df["year"].to_numpy(),
    horizons=(1, 2, 3), n_boot=200, seed=0,
)
print(res_cdh.summary())
print(res_cdh.to_markdown())   # columns [estimand, horizon, att, se]
```

---

## 5. Synthetic Difference-in-Differences (SDID)

Arkhangelsky et al. (2021) unify synthetic control methods and difference-in-differences:
- Unlike Synthetic Control, SDID is invariant to additive unit and time level shifts: both weight problems include the intercepts $\omega_0$, $\lambda_0$ of the paper, so adding a constant to any unit's path (or a common constant to any period) leaves $\hat\tau$ unchanged.
- Unlike classical DiD, SDID does not require parallel trends across the entire donor pool; instead, it finds unit weights $\omega_i \ge 0$ that align the pre-treatment trends between treated and control groups, and time weights $\lambda_t \ge 0$ that prioritize more relevant pre-treatment periods:

$$\hat{\tau}^{\text{SDID}} = \arg\min_{\tau, \mu, \alpha, \beta} \sum_{i=1}^N \sum_{t=1}^T \left( y_{it} - \mu - \alpha_i - \beta_t - \tau W_{it} \right)^2 \hat{\omega}_i \hat{\lambda}_t$$

`synthetic_did` handles a **single treatment cohort** (one common adoption period, `treat_time` equal for every treated unit and `NaN` for donors) and requires a **balanced panel** — a missing `(unit, time)` cell raises a `ValueError` naming it. Standard errors come from a donor bootstrap.

```python
from puremacro.did import synthetic_did

# Single reform cohort: 8 reform states adopt in quarter 12, 32 states never do.
rng = np.random.default_rng(1)
rows = []
for s in range(40):
    reform = s < 8
    alpha_s = rng.normal(scale=2.0)
    for q in range(24):
        effect = 0.8 if (reform and q >= 12) else 0.0
        rows.append({
            "state": f"S{s:02d}", "quarter": q,
            "reform_quarter": 12.0 if reform else np.nan,
            "gdp_growth": alpha_s + 0.05 * q + effect + rng.normal(scale=0.3),
        })
panel_sdid = pd.DataFrame(rows)

res_sdid = synthetic_did(
    panel_sdid,
    unit="state",
    time="quarter",
    outcome="gdp_growth",
    treat_time="reform_quarter",   # per-unit adoption period, NaN for donors
    n_boot=200,
    seed=0,
)

print(res_sdid.summary())
# Inspect optimal donor weights omega and time weights lambda
print("Top donor units:\n", res_sdid.omega[res_sdid.omega > 0.05])
print(res_sdid.lambda_w.round(3))
fig = res_sdid.plot()   # treated-mean vs omega-weighted synthetic path
```

For staggered adoption across multiple cohorts, use `sdid_multi_cohort`. It runs `synthetic_did` once per adoption cohort and averages the cohort estimates with cohort-size weights. Each cohort's donor pool is untreated throughout its SDID window: `control="never_treated"` uses never-treated units over the full panel, `control="not_yet_treated"` also admits later-treated units but truncates the window at their earliest adoption date, and the default `"auto"` picks never-treated donors when at least two exist. Like `cdh_did`, it takes the four-array form:

```python
from puremacro.did import sdid_multi_cohort

res_multi = sdid_multi_cohort(
    df["employment"].to_numpy(),        # outcome
    treated,                            # 0/1 treatment status per row
    df["county_id"].to_numpy(),         # unit id
    df["year"].to_numpy(),              # time id
    aggregation="att_g_t",
    control="auto",
    n_boot=100,
    seed=0,
)
print(res_multi.summary())
print(res_multi.att_g_t)                # one row per cohort
print(res_multi.to_markdown())          # per-cohort table plus the aggregate
```

---

## 6. Publication output

Every result object in `puremacro.did` (and `puremacro.synthetic_control.SyntheticControlResult`) exposes `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()` and `plot()`; the exporters never emit a positional index column. The Honest-DiD sensitivity tools that consume these results are documented in [honest_did.md](honest_did.md).

```python
print(res_sa.to_latex())
print(res_multi.to_typst())
fig_cs = res_cs.plot()
fig_cdh = res_cdh.plot()
fig_multi = res_multi.plot()
```

---

## Summary of Diagnostic Guidelines

1. **Pre-treatment Parallel Trends**: Always inspect event-study coefficients for $e < 0$ in `callaway_santanna` / `sun_abraham`. They should be statistically indistinguishable from zero. `borusyak_jaravel_spiess` reports post-treatment rows only; use `cdh_did`'s `placebo_p` (switchers placebo) as its pre-trend check.
2. **Never-Treated vs. Not-Yet-Treated**:
   - If a genuine never-treated group exists, set `control="never_treated"`.
   - If eventually all units receive treatment, use `control="not_yet_treated"` to avoid dropping the latest adopters (`control_group=` is accepted as an alias of `control=`). The BJS imputation estimator cannot identify the periods after the last adoption in such panels — it raises unless `unidentified="drop"`.
3. **Publication Tables**: Export any result object directly to LaTeX or Typst via `.to_latex()` and `.to_typst()`.
