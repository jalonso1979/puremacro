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
| **Spillover-robust DiD** | `spatial_did` | Clarke (2017); Berg, Reisinger & Streitz (2021); Butts (2023) | One coefficient per distance ring around the treated units, with the beyond-ring units as the only controls |

Two input conventions coexist. `callaway_santanna`, `sun_abraham`, `borusyak_jaravel_spiess`, `synthetic_did` and `spatial_did` take a **long-format DataFrame** with column names passed as `unit=`, `time=`, `outcome=`, `treat_time=` (`treat_time` is the per-unit first-treatment period, `NaN` for never-treated units). `spatial_did` needs the geography on top of those four column names: `coords=` (a DataFrame indexed by unit id whose first two columns are `[lat, lon]`, or a mapping `id -> (lat, lon)`) or a ring assignment built beforehand and passed as `assignment=`. `cdh_did` and `sdid_multi_cohort` take **four aligned 1-D arrays** `(y, treatment, panel_id, time_id)` where `treatment` is the 0/1 treatment status of each row.

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

## 6. Spillover-robust DiD: exposure rings

A plant opens in one county and hires in the three next to it; a state raises its minimum wage and shoppers cross the border. In both cases the treatment reaches units the design calls controls, and SUTVA fails twice over: the comparison group is partly treated, so the direct effect is understated, and the spillover — usually the interesting object — is invisible to a regression that has no coefficient for it.

`spatial_did` implements the standard fix (Clarke 2017; Berg, Reisinger & Streitz 2021; Butts 2023): partition the **untreated** units into distance rings around the treated ones, give each ring its own coefficient, and keep only the units beyond the outermost ring as the comparison group. With common timing and $\text{Post}_t = \mathbf{1}\{t \ge t_0\}$:

$$y_{it} = \mu_i + \tau_t + \sum_{r=0}^{R} \delta_r\, \text{Ring}^r_{it} + \varepsilon_{it}$$

$\text{Ring}^0_{it} = 1$ when unit $i$ is itself treated at $t$; $\text{Ring}^r_{it} = 1$ when $i$ is untreated at $t$ and its distance to the nearest treated unit falls in band $r$. The omitted category is "untreated and beyond `rings[-1]`", so every $\delta_r$ is read against those units. Ring 0 is **absorbing**: a treated unit next to another treated unit stays in ring 0, so $\delta_0$ is the *total* effect on the treated — its own direct effect plus whatever spillover it receives from other treated units — and equals the pure direct ATT only when no two treated units lie within `rings[-1]` of each other. $\delta_1 \dots \delta_R$ are the spillovers.

The naive DiD that pools every untreated unit into the control group is exactly the short regression, so with $M$ the two-way within projection, Frisch-Waugh (1933) gives

$$\hat\delta_{\text{naive}} = \delta_0 + \sum_{r=1}^{R} \theta_r \delta_r, \qquad \theta_r = \frac{(M\,\text{Ring}^0)'(M\,\text{Ring}^r)}{(M\,\text{Ring}^0)'(M\,\text{Ring}^0)}.$$

Ring indicators are mutually exclusive and switch on together, so $\theta_r < 0$ in every realistic design: a *positive* spillover biases the naive estimate *downward*. The result reports `naive_att`, `direct_effect` and `contamination = naive_att - direct_effect = Σ θ_r δ_r`, and the identity holds to machine precision — but only inside the one within projection `spatial_did` runs. It says nothing about a naive TWFE regression fitted on a different sample, which matters as soon as rows have been dropped (`n_obs_dropped`, printed by `summary()`).

### 6.1 Building the rings

Band 1 is the closed interval $[0, \texttt{rings[1]}]$ and band $r \ge 2$ is the half-open $(\texttt{rings[r-1]}, \texttt{rings[r]}]$. Anything past `rings[-1]` — including the $+\infty$ of "nothing nearby has been treated yet" — lands in the control code $R+1$.

Under staggered adoption the ring indicator is time-varying, and the default `ring_timing="already_treated"` measures the distance to the nearest *already*-treated unit at $t$: every ring indicator is then identically zero before any nearby treatment happens, and the pre-period stays a clean baseline. `ring_timing="ever_treated"` instead switches the ring on at the nearest treated unit's own adoption date. It is the right choice when anticipation is the mechanism, and it mechanically contaminates the pre-period otherwise; it also ignores exposure to a nearer-in-time but farther-away treated unit. Ties never change a ring code — the distance is a minimum — they only affect the reported `nearest_treated` identity, where the earliest cohort wins and, within a cohort, the first unit in `ids` order.

`exposure_rings` returns a `RingAssignment` that you can inspect before estimating anything. The example uses the shipped Census county internal points:

```python
import numpy as np
import pandas as pd
from puremacro.datasets import load_us_county_centroids
from puremacro.did import exposure_rings, spatial_did

# Real geography: Census internal points for four Corn Belt states.
counties = load_us_county_centroids()
region = counties[counties["state"].isin(["IA", "IL", "MO", "NE"])]
coords = region[["lat", "lon"]]          # latitude first: the haversine default

# 20 counties get a plant in 2015; the other 389 never do.
rng_sp = np.random.default_rng(7)
plants = rng_sp.choice(coords.index.to_numpy(), size=20, replace=False)
plant_year = pd.Series(np.where(coords.index.isin(plants), 2015.0, np.nan),
                       index=coords.index)
years = list(range(2011, 2021))

ra = exposure_rings(coords, treat_time=plant_year, times=years,
                    rings=(0.0, 40.0, 80.0, 150.0))
print(ra.summary())
print(ra.to_frame("counts"))       # ring, label, lo_edge, hi_edge, n_units, n_unit_periods
print(ra.unit_frame().head())      # one row per unit: distance, ring, entry ring, entry time
```

Read `counts` carefully. `n_unit_periods` partitions all $N \times T$ cells and therefore sums to `len(ra.frame)`; `n_units` counts units *ever* in the ring and does **not** sum to $N$ when the rings are time-varying — under `already_treated` every unit sits in the control ring during the pre-period, which is why the control row reports all 409. `unit_frame()` collapses to one row per unit and raises when any unit walks inward through two rings (`is_static=False`, `n_switchers > 0`); use `.frame` then. `ra.plot()` draws the nearest-treated distance histogram with the cut points on it — cut where the distribution is thin, never through a mode.

### 6.2 Estimating

`cutoff_km` is **required** under the default `cov_type="conley"` and deliberately has no default: the Conley radius is an assumption about how far the error field reaches, and resolving it silently would bury that choice. `2 * rings[-1]` is the natural starting point, because two untreated units can each sit within `rings[-1]` of the same treated unit and still be `2 * rings[-1]` apart.

```python
# A synthetic outcome on that real geography: an effect of 1.0 on the treated
# county and a spillover decaying 0.5 / 0.25 / 0.10 across the three rings.
unit_ring = ra.unit_frame().set_index("unit")["ring"]
true_effect = unit_ring.map({0: 1.0, 1: 0.5, 2: 0.25, 3: 0.10, 4: 0.0})
mu = pd.Series(rng_sp.normal(size=len(coords)), index=coords.index)
tau = pd.Series(rng_sp.normal(scale=0.2, size=len(years)), index=years)
panel_sp = pd.DataFrame([
    {"county": c, "year": y, "plant_year": plant_year[c],
     "log_emp": mu[c] + tau[y] + true_effect[c] * (y >= 2015)
                + 0.25 * rng_sp.normal()}
    for c in coords.index for y in years
])

res_sp = spatial_did(
    panel_sp, unit="county", time="year", outcome="log_emp",
    treat_time="plant_year", coords=coords,
    rings=(0.0, 40.0, 80.0, 150.0),
    cutoff_km=300.0,        # = 2 * rings[-1]; there is no default
    placebo_periods=2,
)
print(res_sp.summary())
```

Inference is the Conley (1999) space-time HAC by default: a distance kernel across units within each period and a Bartlett kernel across periods up to `time_lags` (the Driscoll-Kraay bandwidth rule when left at `None`). `cov_type="cluster"` is available and **under-covers here**: rings are a deterministic function of geography, so regressor and error are both spatially correlated, and clustering by unit throws away exactly the off-diagonal $K(d_{ij}) u_i u_j$ terms the design induces. The Conley error is itself valid only under increasing-domain asymptotics — the region expands, the mixing field decays, the cutoff grows slowly — and says nothing under infill. The honest sample size is `n_treated_clusters`, the number of connected components of the treated-unit graph at radius `cutoff_km`; in the run above 20 treated counties in one contiguous region make just **two** spatially separated experiments at a 300 km cutoff, `spatial_did` warns, and `summary()` prints it. Read the p-values as indicative.

### 6.3 Reading the result

```python
print(res_sp.ring_table[["ring", "label", "n_units", "n_obs",
                         "effect", "se", "p", "dropped"]].round(3))
print(f"naive {res_sp.naive_att:+.3f} = direct {res_sp.direct_effect:+.3f} "
      f"+ contamination {res_sp.contamination:+.3f};  theta {res_sp.theta.round(3)}")
print(bool(abs(res_sp.naive_att
               - (res_sp.direct_effect + res_sp.theta @ res_sp.coef[1:])) < 1e-12))

print(res_sp.att_by_ring_es.round(3))                          # event study, one number per ring
print(res_sp.event_study.head().round(3))                      # the full dynamic path
print(res_sp.pretrend.round(3))                                # per-ring, no all-rings row
print(res_sp.placebo[["ring", "label", "effect", "se", "p"]].round(3))
print(res_sp.tests[["test", "stat", "df", "p", "underpowered"]].round(4))
print(round(res_sp.outer_ring_contrast_bound, 4),
      round(res_sp.outer_ring_contrast_share, 3))
fig_sp = res_sp.plot()      # ring effects with the naive DiD as a dashed line, plus the event study
```

- **`ring_table`** is the headline: one row per ring plus a final row for the omitted control group, with `n_units` (units ever in the ring), `n_obs` (unit-periods carrying the dummy) and `dropped`. Its final row counts the units *never* exposed at any period (72 here) — not the 409 that the assignment's `counts` table reports for the same ring, which counts units *ever* in it. It always comes from the **static two-way-FE fit**; `static_estimator` is fixed at `'twfe'` and `event_study_estimator` governs only the event study. With more than one cohort that static fit carries the Goodman-Bacon (2021) / de Chaisemartin-D'Haultfœuille (2020) negative-weight problem once per ring, and `summary()` says so — read `att_by_ring_es` beside it.
- **`total_effect`** is $\delta_0 + \sum_{r\ge1} (N_r/N_0)\,\delta_r$: the aggregate effect per treated unit, including its spillover footprint. $N_0$ counts ever-treated units and $N_r$ the never-treated units whose *entry* ring is $r$, so each unit is counted exactly once. With 20 plants and 317 exposed neighbours it is much larger than $\delta_0$; that is arithmetic, not a bigger treatment effect.
- **`event_study`** indexes each cell by the unit's **current** ring with the clock anchored at entry, so a unit migrating inward does not drag its old band's dynamic path with it. `att_by_ring_es` aggregates it back to one number per ring over $e \ge 0$ with post-cell-count weights; it is not equal to the static `effect` in general, because the static coefficient is a GLS-type weighted average of the dynamic path rather than a count-weighted one.
- **`pretrend`** reports one Wald row per ring for "every $\delta_{r,e}$ with $e \le -2$ is zero". **`placebo`** refits the whole static specification on the pre-exposure subsample with the entry date shifted back by `placebo_periods`, and its `n_units` / `n_obs` count the units carrying that ring's fake dummy, not the placebo sample's size.
- **`tests`** holds `outer_ring`, `no_spillover`, `equal_rings` and one `dynamics_ring*` row per ring, each with an `underpowered` flag.

### 6.4 Contiguity rings, and what happens when a ring is thin

When "how far" means "how many borders", `contiguity_rings` measures rings in hops on an adjacency graph instead of kilometres — the Berg-Reisinger-Streitz / Delgado-Florax version of the design. The shipped state centroids carry a `neighbors` column of land-border neighbours, which is exactly the input `contiguity_weights` wants. A hop count has no kilometre scale for a Conley kernel, so this route forces `cov_type="cluster"`:

```python
from puremacro.datasets import load_us_state_centroids
from puremacro.spatial import contiguity_weights
from puremacro.did import contiguity_rings

states = load_us_state_centroids()
mainland = states[states["neighbors"].str.strip() != ""]     # drops AK, HI, PR
W = contiguity_weights({s: nb.split() for s, nb in mainland["neighbors"].items()})

adopt = {"CA": 2016.0, "OR": 2016.0, "WA": 2016.0,
         "NY": 2019.0, "MA": 2019.0, "IL": 2019.0}
adopt_year = pd.Series({s: adopt.get(s, np.nan) for s in W.ids})
yrs = list(range(2012, 2023))
ra_hop = contiguity_rings(W, treat_time=adopt_year, times=yrs, orders=2)
print(ra_hop.summary())

rng_hop = np.random.default_rng(3)
eff = {0: 0.8, 1: 0.3, 2: 0.1, 3: 0.0}
ring_at = ra_hop.frame.set_index(["unit", "time"])["ring"]
mu_s = {s: rng_hop.normal() for s in W.ids}
tau_s = {y: 0.2 * rng_hop.normal() for y in yrs}
panel_hop = pd.DataFrame([
    {"state": s, "year": y, "adopt_year": adopt_year[s],
     "log_wage": mu_s[s] + tau_s[y] + eff[int(ring_at[(s, y)])]
                 + 0.2 * rng_hop.normal()}
    for s in W.ids for y in yrs
])

res_hop = spatial_did(panel_hop, unit="state", time="year", outcome="log_wage",
                      treat_time="adopt_year", assignment=ra_hop,
                      cov_type="cluster")
print(res_hop.summary())
print(res_hop.tests[["test", "stat", "df", "p", "underpowered"]])
```

Passing `assignment=` fixes the bands, the timing and the metric, so `rings=`, `ring_timing=` and `metric=` all raise if you also set them — the package refuses to echo back a keyword that did nothing.

Six treated states leave ring 0 with six units, and that is where the size gate bites: the `dynamics_ring0` row of `tests` and the ring-0 row of `pretrend` both come back with `stat = p = NaN` and `underpowered=True`, and `summary()` prints `NOT REPORTED` instead of a p-value. Every *joint* Wald statistic here is a quadratic form in a robust covariance; when a ring is carried by a handful of units that covariance behaves like a sample covariance with that many degrees of freedom and the chi-square reference is badly wrong. Measured on seeded null designs at a nominal 0.10, the per-ring pre-trend rejects 0.285 with fewer than ten units in the ring against 0.085–0.144 above it, and `no_spillover` 0.231 against 0.079–0.110; ten units is where the distortion goes away, which is why the threshold sits there. Rather than print a confident p-value from a reference distribution it knows is wrong, `spatial_did` blanks the row and warns naming the rings. The **per-coefficient** ring table is unaffected: a single-coefficient Wald stays correctly sized however small the ring is, which is why `outer_ring` is never gated.

### 6.5 What this design cannot tell you

The caveats below are not hedges; each is a place where the estimator is silent or wrong, and each is stated in the module's own docstring.

- **The outermost-ring test bounds a difference, not a level.** $\delta_R$ is identified as (outermost ring) minus (the beyond-ring controls). A spillover *common* to both shifts them by the same amount and leaves $\delta_R$ exactly zero. So `outer_ring_contrast_bound` = $|\delta_R| + z\,\text{se}_R$ rules out differences larger than itself; it says nothing about the level of the spillover out there. The field name, the docstring and `summary()` all say "contrast" for that reason, and the only real defence is sensitivity to widening `rings[-1]`.
- **There is no all-rings joint pre-trend row.** Twelve restrictions read off a robust covariance carried by a few dozen treated units is not a chi-square: on seeded nulls at a nominal 0.10 it rejected 0.515 (90 units / 9 treated), 0.20 and 0.18 (200 / 22, with every ring clearing the size gate), 0.135 (200 / 20), 0.095 (300 / 30) and 0.087 (400 / 45). It is over-sized wherever the design is small enough for the question to matter, so it does not ship. `pretrend` carries the same information three restrictions at a time; a Bonferroni or Holm adjustment across those rows is the honest joint statement. Of the joint rows that do ship, the per-ring pre-trend is the least reliable (0.10–0.17 at a nominal 0.10 even past the gate) — read it as indicative.
- **There is no `estimator="cs"` per-ring Callaway-Sant'Anna loop.** It would need one scalar treat time per unit, so it could not represent a unit in ring 2 at $t=5$ and ring 1 at $t=8$; it produces no cross-ring covariance, which leaves the contamination decomposition and every joint test undefined; and its panel bootstrap ignores exactly the spatial correlation this module exists to handle. Run `callaway_santanna` yourself on the per-ring sub-panel if you want those point estimates.
- **Ring cut points are a researcher degree of freedom.** There is no anticipation window, no doughnut design, no continuous exposure measure and no automatic boundary selection. Fix the cut points ex ante on substantive grounds and report sensitivity; `RingAssignment.plot()` exists to help you cut where the distance distribution is thin.
- **Critical values are standard normal, never $t$.** Under spatial dependence the effective degrees of freedom are not $n - k$, so no $t$ distribution is offered.
- **No maps.** The module is pure numpy / scipy / pandas so that it runs under Pyodide; choropleths and any geometry stack stay out of scope, and `matplotlib` is imported inside `plot()`.
- **Small rings are dropped, never merged.** `min_units_per_ring` drops the *rows* of an under-populated ring; it never reassigns them to the control group, which would contaminate exactly the comparison the design depends on. Dropped rings are flagged in `ring_table["dropped"]` and listed in `dropped_rings`.
- **An indefinite HAC meat is possible.** The two-dimensional Bartlett kernel is not positive definite, so a sandwich variance can come out negative; that coefficient's SE, t, p and CI are then `NaN` rather than the square root of a negative number. `psd_adjust="clip"` projects the meat onto the PSD cone, which is conservative — every variance weakly increases.

---

## 7. Publication output

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
3. **Spillovers**: If treatment can reach a nearby untreated unit, the control group is contaminated and every estimator above is estimating the direct effect *minus* a contamination term. Use `spatial_did` and read `contamination` beside `direct_effect`; the outermost ring is the only evidence that the remaining comparison group is clean, and it bounds a difference, not a level.
4. **Publication Tables**: Export any result object directly to LaTeX or Typst via `.to_latex()` and `.to_typst()`.

## References

- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. and Wager, S. (2021). Synthetic difference-in-differences. *American Economic Review* 111(12), 4088–4118.
- Berg, T., Reisinger, M. and Streitz, D. (2021). Spillover effects in empirical corporate finance. *Journal of Financial Economics* 142(3), 1109–1127.
- Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-study designs: robust and efficient estimation. *Review of Economic Studies* 91(6), 3253–3285.
- Butts, K. (2023). Difference-in-differences estimation with spatial spillovers. arXiv:2105.03737.
- Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics* 225(2), 200–230.
- Clarke, D. (2017). Estimating difference-in-differences in the presence of spillovers. MPRA Paper 81604.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics* 92(1), 1–45.
- de Chaisemartin, C. and D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review* 110(9), 2964–2996.
- Delgado, M. S. and Florax, R. J. G. M. (2015). Difference-in-differences techniques for spatial data: local autocorrelation and spatial interaction. *Economics Letters* 137, 123–126.
- Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics* 80(4), 549–560.
- Frisch, R. and Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica* 1(4), 387–401.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics* 225(2), 254–277.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic production in the Caribbean and Central America. *PNAS* 107(35), 15367–15372.
- Huber, M. and Steinmayr, A. (2021). A framework for separating individual-level treatment effects from spillover effects. *Journal of Business & Economic Statistics* 39(2), 422–436.
- Roth, J., Sant'Anna, P. H. C., Bilinski, A. and Poe, J. (2023). What's trending in difference-in-differences? A synthesis of the recent econometrics literature. *Journal of Econometrics* 235(2), 2218–2244.
- Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics* 225(2), 175–199.
- Verbitsky-Savitz, N. and Raudenbush, S. W. (2012). Causal inference under interference in spatial settings. *Epidemiologic Methods* 1(1), 107–130.
