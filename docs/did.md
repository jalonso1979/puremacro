> 🇬🇧 English · 🇪🇸 [Español](es/did.md)

# Modern Difference-in-Differences (DiD)

Classical Two-Way Fixed Effects (TWFE) regressions of the form:

$$y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$

fail when treatment timing is **staggered** (units are treated at different points in time) and treatment effects are **heterogeneous** across cohorts or dynamic over time (Goodman-Bacon 2021, de Chaisemartin & D'Haultfœuille 2020). Under heterogeneous effects, TWFE implicitly subtracts already-treated units as controls for later-treated units, generating negative weights that can invert the sign of the true treatment effect.

`puremacro.did` implements the modern suite of robust, heterogeneity-robust DiD estimators in pure Python, providing exact bootstrap inference, dynamic event-study aggregations, and publication-ready table exports.

---

## Overview of Estimators

| Estimator | Function | Key Reference | Strategy |
|---|---|---|---|
| **Callaway & Sant'Anna** | `callaway_santanna` | Callaway & Sant'Anna (2021, *J. Econometrics*) | Group-time $ATT(g, t)$ with clean control cohorts (never-treated or not-yet-treated) |
| **Sun & Abraham** | `sun_abraham` | Sun & Abraham (2021, *J. Econometrics*) | Interaction-weighted event study using cohort shares |
| **Borusyak, Jaravel & Spiess** | `borusyak_jaravel_spiess` | Borusyak, Jaravel & Spiess (2024, *Rev. Econ. Stud.*) | Imputation estimator: fits FE on untreated cells and projects counterfactuals |
| **Synthetic DiD** | `synthetic_did` | Arkhangelsky, Athey, et al. (2021, *AER*) | Double-weighting: unit weights $\omega$ match pre-trends + time weights $\lambda$ |
| **Multi-Cohort SDID** | `sdid_multi_cohort` | Roth et al. (2023, *J. Econometrics*) | Synthetic DiD generalized across multiple adoption cohorts |

---

## 1. Callaway & Sant'Anna (2021)

Estimates average treatment effects for each cohort $g$ (adoption year) at each calendar period $t$, denoted $ATT(g, t)$. It then aggregates these into an event-study profile tracking dynamic impacts $e = t - g$:

```python
import numpy as np
import pandas as pd
from puremacro.did import callaway_santanna

# Example panel: 'unit', 'year', 'treated_year' (0 if never treated), 'outcome'
# df = ...

res_cs = callaway_santanna(
    df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control_group="never_treated",  # or "not_yet_treated"
    n_boot=1000,
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

- `att_gt`: DataFrame of cohort-time estimates $ATT(g, t)$ with bootstrap standard errors.
- `att_event_study`: Aggregated dynamic effects relative to treatment timing ($e = -K \dots +L$).
- `att_overall`: Summary post-treatment average effect across all treated units.
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Publication table renderers.

---

## 2. Sun & Abraham (2021)

Sun & Abraham explicitly model cohort-specific paths and weight dynamic coefficients by each cohort's sample share. This ensures that dynamic estimates at horizon $e$ are not distorted by compositional changes in which cohorts identify the effect:

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
1. **Fit**: Estimates unit and time fixed effects using *only* non-treated observations ($D_{it} = 0$).
2. **Impute**: Projects counterfactual outcomes $\hat{y}_{it}(0)$ for treated observations.
3. **Average**: Computes treatment effects as $\hat{\tau}_{it} = y_{it} - \hat{y}_{it}(0)$ and aggregates across event times.

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

## 4. Synthetic Difference-in-Differences (SDID)

Arkhangelsky et al. (2021) unify synthetic control methods and difference-in-differences:
- Unlike Synthetic Control, SDID is invariant to additive unit and time level shifts.
- Unlike classical DiD, SDID does not require parallel trends across the entire donor pool; instead, it finds unit weights $\omega_i \ge 0$ that align the pre-treatment trends between treated and control groups, and time weights $\lambda_t \ge 0$ that prioritize more relevant pre-treatment periods:

$$\hat{\tau}^{\text{SDID}} = \arg\min_{\tau, \mu, \alpha, \beta} \sum_{i=1}^N \sum_{t=1}^T \left( y_{it} - \mu - \alpha_i - \beta_t - \tau W_{it} \right)^2 \hat{\omega}_i \hat{\lambda}_t$$

```python
from puremacro.did import synthetic_did

# Single treated cohort
res_sdid = synthetic_did(
    df,
    unit="state",
    time="quarter",
    outcome="gdp_growth",
    treatment="has_reform",
)

print(res_sdid.summary())
# Inspect optimal donor weights omega and time weights lambda
print("Top donor units:\n", res_sdid.omega[res_sdid.omega > 0.05])
```

For staggered adoption across multiple cohorts, use `sdid_multi_cohort`:

```python
from puremacro.did import sdid_multi_cohort

res_multi = sdid_multi_cohort(
    df,
    unit="state",
    time="quarter",
    outcome="gdp_growth",
    treat_time="reform_quarter",
    aggregation="att",
)
print(res_multi.summary())
```

---

## Summary of Diagnostic Guidelines

1. **Pre-treatment Parallel Trends**: Always inspect event-study coefficients for $e < 0$. They should be statistically indistinguishable from zero.
2. **Never-Treated vs. Not-Yet-Treated**:
   - If a genuine never-treated group exists, set `control_group="never_treated"`.
   - If eventually all units receive treatment, use `control_group="not_yet_treated"` to avoid dropping the latest adopters.
3. **Publication Tables**: Export any result object directly to LaTeX or Typst via `.to_latex()` and `.to_typst()`.
