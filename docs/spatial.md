> 🇬🇧 English · 🇪🇸 [Español](es/spatial.md)

# Spatial Econometrics for Regional Macro

`puremacro.spatial` brings the spatial toolkit that regional macro and applied trade work need most: spatial weights matrices, Moran's I / Geary's C autocorrelation diagnostics, and Conley spatial HAC standard errors for cross-sections and for the two-way-FE panel local projections in `puremacro.lp`. Its companion `puremacro.bartik.shift_share_iv` estimates shift-share (Bartik) instrumental-variable regressions with the Adão-Kolesár-Morales (2019) shock-level standard errors.

Everything runs on the four-package Pyodide core (numpy, scipy, pandas, matplotlib). No GIS stack is needed: coordinates are plain latitude/longitude columns, neighbours are plain dictionaries, and economic proximity is a plain flow matrix.

---

## 1. Spatial weights

A `SpatialWeights` object holds a sparse `n × n` matrix `W` with a zero diagonal and the unit labels `ids`. Four builders cover the usual cases:

| Builder | Input | Weight |
| --- | --- | --- |
| `contiguity_weights(neighbours)` | `{unit: [neighbour, ...]}` | 1 for shared borders (symmetrised by default) |
| `knn_weights(coords, k)` | latitude/longitude (or planar) coordinates | 1 for the `k` nearest units |
| `distance_weights(coords, cutoff, decay=...)` | coordinates and a cutoff in km | inverse-distance, uniform or Gaussian decay inside the cutoff |
| `economic_weights(flows)` | origin × destination flow matrix (trade, migration, input-output) | row share of the flow, self-flows dropped |

All builders row-standardise by default so that `W.lag(x)` is the neighbour average of `x`. Units without neighbours are reported as islands (`W.n_islands`, `W.islands`) and `distance_weights` warns when the cutoff leaves some.

```python
import pandas as pd
from puremacro.spatial import distance_weights, knn_weights

capitals = pd.DataFrame(
    {
        "lat": [40.4168, 41.3874, 39.4699, 37.3891, 41.6488, 36.7213, 43.2630, 43.3623],
        "lon": [-3.7038, 2.1686, -0.3763, -5.9845, -0.8891, -4.4214, -2.9350, -8.4115],
    },
    index=["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Malaga", "Bilbao", "A Coruna"],
)
W = distance_weights(capitals, cutoff=450.0, decay="inverse")   # haversine km by default
print(W.summary())
print(W.neighbors("Madrid"))                                     # {label: weight}
Wk = knn_weights(capitals, k=3)
print(Wk.to_frame().head())                                      # edge list: source, target, weight
```

`W.lag(series)` aligns a pandas Series or DataFrame by label, so the order of your data frame never has to match the order of the weights. `W.to_dense()` returns the numpy matrix when you need it in a solver.

---

## 2. Spatial autocorrelation diagnostics

Moran's I and Geary's C summarise whether a variable is clustered (similar neighbours), dispersed (dissimilar neighbours) or random over the map. With `z = x − mean(x)` and `S₀ = Σᵢⱼ wᵢⱼ`:

$$I = \frac{n}{S_0}\,\frac{z' W z}{z' z}, \qquad C = \frac{(n-1)\sum_{ij} w_{ij}(x_i - x_j)^2}{2 S_0\, z'z}.$$

Under no spatial autocorrelation `E[I] = −1/(n−1)` and `E[C] = 1`. Positive autocorrelation pushes `I` above its expectation and `C` below one. Both functions report the Cliff-Ord normality and randomisation variances with their z-scores and p-values, plus a permutation p-value that shuffles `x` across units (`n_perm`, default 999).

```python
import numpy as np
from puremacro.spatial import contiguity_weights, gearys_c, morans_i

rng = np.random.default_rng(0)
side = 8
neighbours = {}
for i in range(side):
    for j in range(side):
        u = i * side + j
        neighbours[u] = [v for v in (u - side, u + side, u - 1, u + 1)
                         if 0 <= v < side * side and abs((v % side) - j) <= 1]
W = contiguity_weights(neighbours)              # rook contiguity on an 8 x 8 lattice
rho = 0.6
x = np.linalg.solve(np.eye(W.n) - rho * W.to_dense(), rng.standard_normal(W.n))   # SAR field
print(morans_i(x, W, n_perm=499).summary())
print(gearys_c(x, W, n_perm=499).summary())
```

`MoranResult.plot()` draws the Moran scatterplot (`z` against its spatial lag); the slope of the fitted line is Moran's I. The statistics are golden-tested against `esda` (PySAL) to 1e-10.

---

## 3. Conley spatial HAC standard errors

Regional shocks are correlated across nearby units. Clustering by administrative region assumes the correlation stops at the border; Conley (1999) instead lets the covariance of the scores decay with distance:

$$\hat V = (X'X)^{-1}\Big[\sum_i \sum_j K(d_{ij})\, u_i u_j\, x_i x_j'\Big](X'X)^{-1}, \qquad K(d) = \max\{0,\, 1 - d/\text{cutoff}\}\ \text{(Bartlett) or } \mathbf{1}\{d \le \text{cutoff}\}\ \text{(uniform)}.$$

The cutoff is the bandwidth: at `cutoff=0` the estimator is exactly HC0, and with a uniform kernel and units grouped in far-apart blocks it is exactly the cluster-robust covariance. Report several cutoffs; the errors should stabilise once the cutoff exceeds the range of the spatial correlation.

```python
import numpy as np
from puremacro.spatial import conley_se, pairwise_distances

rng = np.random.default_rng(1)
n = 200
coords = rng.uniform([36.0, -9.0], [43.5, 3.0], size=(n, 2))      # lat, lon over Spain
D = pairwise_distances(coords, "haversine")
common = np.exp(-D / 150.0) @ rng.standard_normal(n)              # shock correlated within ~150 km
x = rng.standard_normal(n) + 0.5 * common
y = 1.0 + 0.5 * x + 0.5 * rng.standard_normal(n) + common
X = np.column_stack([np.ones(n), x])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
for cutoff in (0.0, 100.0, 300.0):
    print(f"cutoff {cutoff:5.0f} km  se(beta) = {conley_se(X, resid, coords, cutoff)[1]:.4f}")
```

### 3.1 Panel local projections with spatial HAC

`panel_lp` accepts `cov_type="conley"`. The covariance is the Hsiang (2010) space-time HAC: a Conley kernel across units within each period, and a Bartlett kernel across periods up to `time_lags` (by default the Driscoll-Kraay bandwidth rule already used by `panel_lp_dk`). With a cutoff larger than every pairwise distance it collapses exactly to Driscoll-Kraay; with `time_lags=0` it is a per-period Conley covariance.

```python
import numpy as np
import pandas as pd
from puremacro.lp import panel_lp

rng = np.random.default_rng(2)
regions = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Malaga", "Bilbao", "A Coruna"]
coords = pd.DataFrame(
    {
        "lat": [40.4168, 41.3874, 39.4699, 37.3891, 41.6488, 36.7213, 43.2630, 43.3623],
        "lon": [-3.7038, 2.1686, -0.3763, -5.9845, -0.8891, -4.4214, -2.9350, -8.4115],
    },
    index=regions,
)
T = 80
common = rng.standard_normal(T)
rows = []
for r in regions:
    shock = rng.standard_normal(T)
    y = np.cumsum(0.3 * shock + 0.5 * common + rng.standard_normal(T))
    rows += [{"code": r, "date": t, "y": y[t], "shock": shock[t]} for t in range(T)]
panel = pd.DataFrame(rows).set_index(["code", "date"])
irf = panel_lp(panel, "y", "shock", horizons=range(0, 9), n_lags=2,
               cov_type="conley", coords=coords, cutoff_km=400.0)
print(irf.round(3))
```

`coords` must be indexed by the entity labels of the panel (missing entities raise `KeyError`). Use `kernel="uniform"` for a hard cutoff and `metric="euclidean"` when the coordinates are already planar (kilometres on a projected grid).

---

## 4. Shift-share IV with shock-level standard errors

A shift-share instrument combines pre-period exposure shares `sᵢₖ` with sector-level shocks `gₖ`: `zᵢ = Σₖ sᵢₖ gₖ`. Adão, Kolesár and Morales (2019) show that units with similar share vectors have correlated residuals even when they are far apart, so heteroskedasticity-robust or geographically clustered errors under-cover. Their estimator aggregates the residuals to the sector level:

$$\widehat{\text{se}}_{\text{AKM}}(\hat\beta) = \frac{\sqrt{\sum_k \tilde g_k^2 \Big(\sum_i w_i s_{ik} \hat\varepsilon_i\Big)^2}}{\big|\sum_i w_i \tilde z_i \tilde x_i\big|},$$

where tildes denote residuals from the controls (and, for the shocks, from the share-weighted `shock_controls`). `shift_share_iv` returns the 2SLS estimate, both standard errors, the robust first-stage F and the Rotemberg weights of Goldsmith-Pinkham, Sorkin and Swift (2020), which tell you which sectors drive the estimate.

```python
import numpy as np
import pandas as pd
from puremacro.bartik import shift_share_iv

rng = np.random.default_rng(3)
n_regions, n_industries = 300, 25
shares = pd.DataFrame(rng.dirichlet(np.full(n_industries, 0.5), size=n_regions),
                      columns=[f"ind{k:02d}" for k in range(n_industries)])
shocks = pd.Series(rng.standard_normal(n_industries), index=shares.columns)   # national industry shocks
exposure = shares.to_numpy() @ shocks.to_numpy()
employment_growth = exposure + rng.standard_normal(n_regions)
industry_confounder = shares.to_numpy() @ rng.standard_normal(n_industries)    # what breaks robust SEs
wage_growth = 0.8 * employment_growth + industry_confounder + 0.5 * rng.standard_normal(n_regions)
df = pd.DataFrame({"wage_growth": wage_growth, "employment_growth": employment_growth})
res = shift_share_iv(df, "wage_growth", "employment_growth", shares, shocks)
print(res.summary())
print(res.rotemberg_weights.sort_values(ascending=False).head())
```

Pass `se="robust"` to report the conventional error as the headline, `weights=` for population weights, and `controls=` for unit-level covariates. Shares must be non-negative and are aligned to `df` by index when given as a DataFrame.

---

## 5. Choosing a specification

Four specifications sit on the same weights matrix, and they are not nested in a way that lets a single regression pick between them:

| model | equation | estimator |
| --- | --- | --- |
| SLX | `y = X β + W X_d θ + ε` | OLS |
| SAR | `y = ρ W y + X β + ε` | concentrated ML or Kelejian-Prucha GMM |
| SEM | `y = X β + u`, `u = λ W u + ε` | concentrated ML or Kelejian-Prucha GMM |
| SDM | `y = ρ W y + X β + W X_d θ + ε` | concentrated ML or Kelejian-Prucha GMM |

`ols_spatial` is the entry point to call first. It is ordinary least squares with the Anselin-Bera-Florax-Yoon (1996) Lagrange-multiplier battery attached: five statistics computed entirely **under the null of no spatial dependence**, so nothing spatial has to be estimated to run them. With `e` the OLS residual, `σ̂² = e'e/n`, `T = tr(WW + W'W)`, `M = I − X(X'X)⁻¹X'` and `nJ = [(WXb)'M(WXb) + T σ̂²]/σ̂²`:

$$\text{LM}_{\text{lag}} = \frac{(e'Wy/\hat\sigma^2)^2}{nJ}, \qquad \text{LM}_{\text{err}} = \frac{(e'We/\hat\sigma^2)^2}{T},$$

$$\text{RLM}_{\text{lag}} = \frac{(d_\rho - d_\lambda)^2}{nJ - T}, \qquad \text{RLM}_{\text{err}} = \frac{(d_\lambda - T d_\rho/nJ)^2}{T\,(1 - T/nJ)},$$

with `d_ρ = e'Wy/σ̂²` and `d_λ = e'We/σ̂²`. The robust versions are the point: on a genuine SAR the *error* statistic also rejects, and vice versa, so the plain pair almost always rejects twice. `RLM_lag` is robust to a locally misspecified error process and `RLM_err` to a locally misspecified lag, and the pair is what separates them.

```python
import numpy as np
import pandas as pd
from puremacro.datasets import load_us_state_centroids
from puremacro.spatial import contiguity_weights, ols_spatial

states = load_us_state_centroids()
lower48 = states.drop(index=["AK", "HI", "PR"])                  # 48 states + DC
neighbours = {s: [nb for nb in row.split() if nb in lower48.index]
              for s, row in lower48["neighbors"].items()}
W = contiguity_weights(neighbours)                               # the real land-border graph
ids = list(W.ids)
coords = lower48.loc[ids, ["lat", "lon"]]
print(W.summary())

rng = np.random.default_rng(11)
n = W.n
log_area = np.log(lower48.loc[ids, "land_sqmi"].to_numpy())
log_area = (log_area - log_area.mean()) / log_area.std()
X = pd.DataFrame({"log_area": log_area, "coastal": rng.standard_normal(n)}, index=ids)
A_inv = np.linalg.inv(np.eye(n) - 0.5 * W.to_dense())            # a true SAR field, rho = 0.5
y = pd.Series(A_inv @ (1.0 + 0.8 * X["log_area"] - 0.4 * X["coastal"]
                       + 0.5 * rng.standard_normal(n)), index=ids)

fit_ols = ols_spatial(y, X, W)
print(fit_ols.summary())
print(fit_ols.lm.recommendation)
```

`recommendation` is a short string, and the whole decision rule is in it. Neither statistic rejects → `ols`. Only one rejects → that one's model. Both reject → the robust pair decides, and if *both* robust statistics reject the string is `sdm or sarar (both robust LM reject)`, because two rejections point at a model with two spatial processes, not at either simple one. If neither robust statistic rejects after both plain ones did, the answer is `inconclusive`, and that is the honest report.

Three caveats travel with the battery. `summary()` prints the first one; all three are spelled out here:

- It is a **local** test. It says which alternative the OLS residuals point at, not that the indicated model is correct.
- Every statistic uses the ML variance `e'e/n`, as the Gaussian likelihood derivation requires. Substituting the degrees-of-freedom-corrected `e'e/(n−k)` would rescale `RLM_lag` by `(n−k)/n` and `LM_err` by `((n−k)/n)²`, and would rescale `LM_lag`, `RLM_err` and `LM_SARMA` by no constant at all, because `nJ` is **affine**, not linear, in `1/σ̂²`. (Measured on `n = 60`, `k = 3`, where `(n−k)/n = 0.95`: the five ratios are 0.902812, 0.902500, 0.950000, 0.934909 and 0.909873.) The `sigma2` that `ols_spatial` reports for its own standard errors *is* dof-corrected. The two scalings live side by side on purpose.
- The residual Moran's I here is the Cliff-Ord (1972) statistic with the `X`-dependent moments, and it differs from `morans_i` from §2, which uses the raw-variable moments. OLS residuals are not the raw variable.

`ols_spatial` takes `cov_type='nonrobust'` or `'hc1'` and deliberately offers **no** Conley option: the battery is derived under homoskedastic Gaussian OLS residuals, and pairing it with a spatial-HAC covariance would print two inconsistent assumptions in one table. Call `conley_cov` from §3 directly if that is what you want. The robust LM-lag is also undefined when `nJ − T` vanishes — `W X b` lying in the column space of `X`, for instance a single regressor that is an eigenvector of `W` — and `lm_spatial_tests` raises with that diagnosis rather than dividing by zero.

---

## 6. Cross-section spatial models

The default estimator is Gaussian maximum likelihood through the *concentrated* log-likelihood of Ord (1975): `β` and `σ²` are profiled out, leaving a smooth one-dimensional problem in the spatial parameter,

$$\ln L_c(\rho) = -\tfrac{n}{2}\left(\ln 2\pi + 1\right) + \ln\lvert I - \rho W\rvert - \tfrac{n}{2}\ln\frac{\text{SSR}(\rho)}{n},$$

maximised by a bounded scalar search over the admissible interval and cross-checked against the root of the analytic score. The Jacobian `ln|I − ρW|` is exact — dense eigenvalues for `n ≤ 1000`, a sparse LU factorisation above — and there is no stochastic log-determinant anywhere in the module, so the same data always give the same `ρ`.

```python
from puremacro.spatial import sar, sdm, sem, slx

fit_sar = sar(y, X, W)          # concentrated Gaussian ML
fit_sem = sem(y, X, W)
fit_sdm = sdm(y, X, W)          # SAR augmented with W X, plus the common-factor test
fit_slx = slx(y, X, W)          # plain OLS on [X, W X]

print(pd.DataFrame(
    {"spatial": [fit_sar.rho, fit_sem.lam, fit_sdm.rho],
     "log_area": [fit_sar.params["log_area"], fit_sem.params["log_area"],
                  fit_sdm.params["log_area"]],
     "loglik": [fit_sar.loglik, fit_sem.loglik, fit_sdm.loglik],
     "aic": [fit_sar.aic, fit_sem.aic, fit_sdm.aic]},
    index=["sar", "sem", "sdm"]).round(3))
print(fit_sar.summary())
```

**SDM against SEM: the common-factor test.** An SDM collapses to an SEM exactly when `θ = −ρ β_d` — write the SEM in reduced form and the constraint falls out. `sdm` reports the Wald test of that restriction as `common_factor = (statistic, df, p-value)`; a non-rejection says the extra Durbin terms are the spatial error process in disguise, a rejection says they are not.

```python
stat, dof, pval = fit_sdm.common_factor
print(f"common factor: W = {stat:.4f}, df = {dof:.0f}, p = {pval:.4f}")
print("SEM is rejected" if pval < 0.05 else "SEM is not rejected")
print(fit_slx.params.round(3))
```

The test needs the **full** Durbin block: with `durbin=` selecting a strict subset the restriction does not nest SEM at all, and `common_factor=True` then raises rather than printing a claim about a model outside the parameter space being tested. The constant is always excluded from the Durbin block — for a row-standardised island-free `W`, `W1 = 1` makes the lagged constant exactly collinear with the constant. And the statistic is only *interpretable* where `sign(θ_r) = −sign(ρ β_r)` for every `r`; where that fails `summary()` appends the caveat rather than suppressing the number.

**When to prefer `method='gmm'`.** ML is the default and is efficient under its assumptions, but those assumptions bind hard. Under heteroskedasticity of unknown form the Gaussian SAR ML estimator of `ρ` is itself **inconsistent** (Lin & Lee 2010) — no sandwich rescues a standard error for the wrong number, which is why `vcov='robust'` with `method='ml'` raises instead of quietly computing one. Kelejian-Prucha GMM is spatial two-stage least squares with `[X, WX, …, W^q X]` as instruments; it admits heteroskedasticity-robust errors, needs no log-determinant, and is the route to take when the residuals are plainly heteroskedastic or `n` is large enough that the Jacobian dominates.

```python
fit_gmm = sar(y, X, W, method="gmm", vcov="robust")
print(fit_gmm.params.round(4))
print("rho admissible:", fit_gmm.rho_admissible, " interval:",
      tuple(round(b, 4) for b in fit_gmm.rho_bounds))
```

Four things about the GMM path are deliberate limits rather than oversights:

- `vcov='robust'` is a White sandwich around the **homoskedastic** Kelejian-Prucha moment conditions. It corrects the standard errors, not the moments, and it does not make the GMM `λ` heteroskedasticity-consistent. The Kelejian-Prucha (2010) heteroskedasticity-robust GMM is not implemented.
- `sem(method='gmm')` reports `λ` with `bse['lambda'] = NaN`, because an honest standard error for it needs those same `Ψ` matrices. The `β` standard errors stay asymptotically valid: the feasible generalised spatial estimator has the limiting distribution of the one with `λ` known.
- Kelejian-Prucha 2SLS is unconstrained, so on a weakly identified design it can return a `ρ` **outside** the admissible interval. When it does, a `RuntimeWarning` fires, `rho_admissible` is `False`, `fitted` / `resid_reduced` / `pseudo_r2` are `NaN` rather than pushed through an explosive multiplier, and `.effects()` raises.
- There is no **SARAR/SAC** anywhere in the module — no `error_weights=` argument exists. Use `sdm`, which nests the SEM common-factor restriction and is testable against it.

Two more warnings worth keeping in the front of your mind. A `ρ` estimated on a binary `W` is not comparable with one estimated on the row-standardised version: the normalisation changes the parameter's scale (on a 6×6 rook lattice a binary `W` restricts `ρ` to about `(−0.277, 0.277)`). And `logdet='chebyshev'` is an *approximation*, not an exact route: on a 20×20 rook lattice at order 5 its error is ~2e-4 at `ρ = 0.3` but ~0.66 at `ρ = 0.9`. Reach for it only on very large symmetrisable `W`, and never for a headline number.

---

## 7. Impacts, not coefficients

This is the single most misread quantity in applied spatial work. **A SAR `β` is not a marginal effect.** In `y = ρWy + Xβ + ε` the outcome enters its own right-hand side, so raising `x_r` in one unit raises that unit's `y`, which raises its neighbours' `y`, which feeds back. The derivative matrix of the `r`-th regressor is

$$S_r(W) = (I - \rho W)^{-1}\left(I \beta_r + W \theta_r\right),$$

and `β_r` is neither its diagonal nor its row sum. LeSage and Pace (2009, sec. 2.7) summarise it in three scalars: the **direct** impact `tr(S_r)/n` (own-unit, feedback loops included), the **total** impact `1'S_r 1/n` (the average row sum), and the **indirect** impact as their difference — the part that leaks to everyone else.

```python
imp = fit_sar.effects(n_sim=1000)
print(imp.summary())
print(imp.to_frame()[["variable", "direct", "indirect", "total",
                      "total_lo", "total_hi"]].round(3))
print("true beta 0.800 | OLS", round(float(fit_ols.params["log_area"]), 3),
      "| SAR beta", round(float(fit_sar.params["log_area"]), 3),
      "| direct", round(float(imp.direct[0]), 3),
      "| total", round(float(imp.total[0]), 3))
```

Read that line. The truth is `0.8`; OLS, which absorbs the whole spatial multiplier into the slope, reports about `1.31`; the SAR `β` is back near `0.84`; the *direct* impact is a little above it because the feedback loops through the neighbours and back are part of a unit's own response; and the *total* impact — what a policy applied everywhere would move on average — is roughly `1.93`. Report all three, or report the total and say so. Reporting `β` alone as "the effect" is the error.

`n_sim` draws `[β, θ, ρ]` from its estimated normal distribution and pushes each draw through the impact formula. The standard errors are therefore a normal approximation to the **parameter** distribution propagated through a non-linear map — not a bootstrap of the data — and the reported p-values are simulation quantile p-values, `2·min(share of draws > 0, share < 0)`, deliberately not `2Φ(−|effect|/se)`: with `ρ` near a bound the draw distribution is skewed and a normal p-value would contradict the interval printed beside it. Draws whose `ρ` falls outside the admissible interval are **discarded, not clipped** (above 5% discarded warns, above 50% raises). `n_sim=0`, the default, returns point estimates with `NaN` standard errors, so the plainest call always works.

Four caveats:

- **An indirect effect is the *sum* of all `n(n−1)` off-diagonal cross-partials divided by `n`** — the average total spillover a unit sends, equivalently receives. It is not the effect on any one neighbour, not the effect on the first ring, and not the mean individual cross-partial — that mean is smaller by a factor of `n−1` (48 on the lower-48 graph, where an indirect impact of `0.737456` corresponds to a mean cross-partial of `0.015364`). Do not narrate it as "the spillover onto adjacent states".
- **The average total impact is not `1/(1−ρ)`.** That shortcut equals `1'A⁻¹1/n` only for a row-standardised, island-free `W` — the first pair of numbers below, where the two agree to the last digit. Drop either condition and it breaks. Take the *binary* version of the very same land-border graph: its largest eigenvalue is `5.4154`, so `ρ` must stay below `0.1847`, and at `ρ = 0.15` the exact `1'A⁻¹1/n` is about `4.13` against the shortcut's `1.18` — the shortcut is off by a factor of three and a half, and understates rather than overstates. A single island breaks it too. This module always computes `1'A⁻¹1`.

```python
print("row-standardised, island-free W -- the shortcut does hold:")
print("  1'A^-1 1/n =", round(imp.s / imp.n, 6),
      " vs the 1/(1-rho) shortcut =", round(1.0 / (1.0 - imp.rho), 6))

Wb = W.binary()                                   # same graph, no row standardisation
rho_b = 0.15                                      # admissible: 1 / max eig(Wb) = 0.1847
Ab = np.eye(Wb.n) - rho_b * Wb.to_dense()
exact_b = float(np.ones(Wb.n) @ np.linalg.solve(Ab, np.ones(Wb.n)) / Wb.n)
print("binary W at rho = 0.15 -- the shortcut does not:")
print("  1'A^-1 1/n =", round(exact_b, 6),
      " vs the 1/(1-rho) shortcut =", round(1.0 / (1.0 - rho_b), 6))
print("SLX impacts are exact and local:")
print(fit_slx.effects(n_sim=500).to_frame()[["variable", "direct", "indirect", "total"]].round(3))
```

- **SEM has no impact decomposition at all**, and `.effects()` raises for it: in `y = Xβ + u` the marginal effect of `x_r` is `β_r` on the diagonal and zero off it, whatever `λ` is. That is a feature of the model, not a gap in the software.
- **SLX impacts are exact and local**: `direct = β`, `indirect = θ·S₀/n`, no spatial multiplier and no simulation needed for the point estimates. This is a real argument for SLX when the theory says spillovers are one ring deep.

One divergence from `spreg`, deliberate and documented: `spat_impacts='full'` assigns every SLX term to the *indirect* column, dropping `θ_r tr(A⁻¹W)/n` from the direct effect; this module follows the textbook and keeps it there. The **total** is identical either way, and for a SAR (`θ = 0`) the two agree exactly.

---

## 8. Spatial panels

`spatial_panel` fits SAR, SDM and SEM on a **balanced** `(entity, time)` panel by quasi-maximum likelihood, with any combination of individual and time fixed effects:

$$y_{it} = \rho \sum_j w_{ij} y_{jt} + x_{it}'\beta + (Wx)_{it}'\theta + \mu_i + \xi_t + \varepsilon_{it}.$$

The fixed effects are the problem. The `μ_i` are `n` incidental parameters each estimated from `T` observations and the `ξ_t` are `T` of them each estimated from `n`; concentrating them out of the Gaussian likelihood leaves a score that is not centred at the truth. With `G = W(I − ρW)⁻¹` and two-way effects, `E[∂L/∂ρ] = −tr(G) − (T−1)/(1−ρ)` and `E[∂L/∂σ²] = −(n+T−1)/(2σ²)`, which over the `O(nT)` information is an `O(1/T)` bias from the individual effects plus an `O(1/n)` bias from the time effects. `β` is unaffected; `ρ` and `σ²` are not.

Lee and Yu (2010) remove both exactly. `method='transformation'` (the default) runs the likelihood on an orthonormal transformation whose score is centred **by construction**, so `bias_correction` reports `'none'` because the correction is provably zero, not because it was skipped. `method='direct'` concentrates the effects out and applies the analytic correction `θ̂ − I⁻¹a` instead; it is kept for comparability with software that does it that way, and because its Jacobian needs no assumption on the row sums of `W`.

```python
from puremacro.spatial import spatial_panel

rng = np.random.default_rng(4)
T = 24
S_inv = np.linalg.inv(np.eye(n) - 0.4 * W.to_dense())            # true rho = 0.4
xs = rng.standard_normal((n, T))
ys = S_inv @ (0.9 * xs + rng.standard_normal((n, 1))             # mu_i
              + rng.standard_normal((1, T))                      # xi_t
              + 0.4 * rng.standard_normal((n, T)))
idx = pd.MultiIndex.from_product([ids, range(T)], names=["code", "date"])
panel = pd.DataFrame({"gsp": ys.ravel(), "spend": xs.ravel()}, index=idx)

fit_tr = spatial_panel(panel, "gsp", "spend", W, model="sar", effects="two-way")
print(fit_tr.summary())
```

The impacts table `spatial_panel` prints is the same LeSage-Pace decomposition as §7 (its point estimates are pinned against `spatial_effects` to 1e-12), with one convention difference: the panel reports `point ± z·se` intervals rather than empirical simulation quantiles. `impacts=None` — the default — means "yes for SAR/SDM, no for SEM"; asking for `impacts=True` with `model='sem'` raises rather than silently returning `None`.

Running the same data through `method='direct'` shows the correction that the transformation approach never needs:

```python
fit_dir = spatial_panel(panel, "gsp", "spend", W, model="sar", effects="two-way",
                        method="direct", impacts=False)
print(fit_dir.bias.round(4))                    # what Lee-Yu subtracted
print("transformation rho:", round(fit_tr.rho, 4),
      " direct+Lee-Yu rho:", round(fit_dir.rho, 4))
```

**What the standard errors are valid for.** `vcov='oim'` (the default) is the inverse expected information: valid under iid Gaussian `ε`, extended by Lee (2004) to non-normal but still homoskedastic errors. `vcov='dk'` (Driscoll-Kraay) and `vcov='conley'` replace the **`β` block only** — the `ρ`/`λ` and `σ²` rows and columns always come from the OIM.

```python
fit_conley = spatial_panel(panel, "gsp", "spend", W, model="sar", effects="two-way",
                           vcov="conley", coords=coords, cutoff_km=800.0, impacts=False)
print(fit_conley.vcov_note)
print(pd.DataFrame({"oim": fit_tr.se, "conley": fit_conley.se}).round(4))
```

That restriction is deliberate, and there is no `vcov='qml'` or `vcov='cluster'` here at all — not merely unavailable, but *wrong* for this score. A per-cell outer product misstates the spatial parameter badly: `s_it^ρ` carries `(Wy)_it e_it`, and `Wy` loads on every neighbour's `ε`, so the omitted cross-`i` terms are large. On a 6×6 row-standardised rook lattice at `ρ = 0.5` with `T = 8` the outer-product meat is 163.9 against a true score variance of 303.5 — 54% of the truth, understating the `ρ` standard error by 27%.

And no covariance of any kind is robust to a mis-specified `W`: if `W` is wrong, `ρ̂` is inconsistent. Under heteroskedastic `ε` the SAR QMLE **point estimate** is inconsistent (Lin & Lee 2010), so no sandwich rescues it there either.

Restrictions to plan around:

- **Balanced panels only.** The transformation needs a common `T`; an unbalanced panel raises and names `puremacro.inference.balanced_panel.balanced_subpanel`.
- **Static only.** Dynamic spatial panels (a time-lagged `y` or `W y_{t−1}`) are not implemented; Yu, de Jong and Lee (2008) is cited for the `n/T → c` asymptotics behind the bias term, not implemented.
- **No random effects**, no Hausman test, **no SARAR/SAC**, and **no SDEM** (`model='sem'` with `durbin=` raises rather than fitting an equation the module does not define).
- **Dense spectrum.** The information matrix, the bias vector and the impacts all need `tr(G)`, `tr(G²)`, `tr(G'G)` and `diag(G)`, which an LU factorisation does not provide and a stochastic probe would make irreproducible — so `logdet='lu'` raises, a `RuntimeWarning` fires above 2000 units and `dense_max = 5000` is a hard ceiling. A 3,143-county panel is near the ceiling; census tracts are past it.
- With a **time-effects** component and `method='transformation'`, every row of `W` must sum to one — that is what makes `1` an eigenvector and lets the time effects drop out of an exact `(n−1)`-unit SAR. The error message names `standardize()`.

---

## 9. Spatial local projections

`spatial_lp` is a two-way fixed-effects panel local projection (Jordà 2005) augmented with the spatially lagged shock, so a regional impulse response splits into a response to the unit's **own** shock and a response to the shock its **neighbours** receive. For every horizon `h`,

$$y_{i,t+h} - y_{i,t-1} = \mu_i^h + \tau_t^h + \beta_h x_{it} + \sum_p \gamma_{p,h}\,(W^{(p)}x)_{it} + (\text{lags}) + \varepsilon_{i,t+h},$$

with `W^(p)` the `p`-th order neighbour matrix in the Anselin (1988) sense — the walks of exactly `p` steps, minus the diagonal and minus every lower order, so `γ₂` measures a second ring instead of double-counting the first.

```python
from puremacro.spatial import spatial_lp

rng = np.random.default_rng(7)
T = 70
shock = rng.standard_normal((n, T))
dy = 0.6 * shock + 0.35 * (W.to_dense() @ shock) + 0.3 * rng.standard_normal((n, T))
idx = pd.MultiIndex.from_product([ids, range(T)], names=["code", "date"])
lp_panel = pd.DataFrame({"emp": np.cumsum(dy, axis=1).ravel(),
                         "spend": shock.ravel()}, index=idx)

irf = spatial_lp(lp_panel, "emp", "spend", W, horizons=range(0, 7), n_lags=2,
                 cov_type="conley", coords=coords, cutoff_km=800.0)
print(irf.summary())
```

### What `total` is, and what it is not

`β_h` and `γ_{p,h}` are estimated **after** a two-way within transformation, so they are responses *relative to the period mean*. The time effect `τ_t^h` absorbs, by construction, every component of the response that is common to all units in a period — including the response to the aggregate part of the shock. So:

- `direct` = `β_h`: the extra response of a unit whose own shock is one unit above the period average, holding its neighbourhood fixed;
- `indirect` = `c₁ γ_{1,h}`: the extra response of a unit whose first-ring neighbours' shocks are `c₁` above the period average (one such column per order — `indirect2`, `indirect3`, … — plus `indirect_all` when there is more than one);
- `total` = `direct + Σ_p c_p γ_{p,h}`: a **cross-sectional contrast**, the response of a unit whose own and neighbour shocks are both above the period average, relative to a unit at the period average.

`c_p` is the size of the neighbourhood contrast, and `total_scale=` sets it. Under the default `total_scale='row_sum'` it is the mean row sum of `W^(p)` — exactly `1` for a row-standardised island-free `W`, so `indirect = γ_{1,h}` in that case; `total_scale='unit'` forces every `c_p = 1`, and a float or one float per order sets them by hand. The vector actually used, `c = (1, c₁, …, c_P)`, is on `.total_scale`.

**`total` is not the response to a uniform unit shock hitting every unit, and this module never claims that it is.** A uniform shock is exactly the variation `τ_t` removes: within-transformed, the predicted response to a uniform `+1` is `γ(s_i − s̄)`, whose cross-unit mean is zero. Adding an arbitrary aggregate component `κ·x̄_t` to the data-generating process leaves the estimates unchanged to 1e-16, so no aggregate multiplier is identified here at all. `total` equals the aggregate response only under the additional, untestable assumption that the common response is zero — which is the whole point of Chodorow-Reich (2019). Write it down that way in the paper.

These are also reduced-form coefficients on own and neighbour **shocks**. They are *not* the LeSage-Pace partial derivatives of §7; there is no `(I − ρW)⁻¹` multiplier anywhere in this estimator.

### Inference, honestly

```python
print(irf.spillover[["h", "order", "gamma", "se", "corr_own"]].round(4).to_string(index=False))
print(irf.cumulative[["h", "cum_direct", "cum_indirect", "cum_total", "se_total"]].round(3).to_string(index=False))
```

The cross-horizon covariance is retained in full (`.vcov`), so the standard error of `direct + indirect` carries the off-diagonal term and the cumulative bands are honest rather than a sum of per-horizon variances.

**The joint cross-horizon spillover test is deliberately not shipped.** A χ² reference for the stacked restriction `γ_{p,h} = 0 at every h` is grossly oversized in exactly the panels this module targets: a 150-replication Monte Carlo with `N = 20`, `T = 50`, `H = 6` and a true `γ = 0` rejected at **0.313** against a nominal 0.10 (0.187 at nominal 0.05). A many-restriction Wald with an estimated HAC covariance on `T ≈ 50` periods does not have its nominal size, so it does not ship. Test one horizon at a time, or build your own fixed-*b* reference from `.vcov`.

The per-horizon z-test that *does* ship is not exact either, and the module reports its size rather than one favourable number. On the same design (`N = 20`, `T = 50`, `h = 0…5`, true `γ = 0`, 400 replications, Monte Carlo standard error 0.011 at nominal 0.05):

```text
cov_type            nominal 0.05              nominal 0.10
                  h = 0   worst over h      h = 0   worst over h
driscoll-kraay    0.083   0.110 (h = 3)     0.138   0.172 (h = 5)
conley 1000 km    0.068   0.100 (h = 3)     0.138   0.152 (h = 2)
cluster           0.090   0.090 (h = 0)     0.160   0.160 (h = 0)
```

So the honest contrast with the joint test's 0.313 is 0.08–0.11 at a nominal 0.05, and the size drifts up with the horizon as the overlapping-window MA(h) dependence grows. Doubling the cross-section to `N = 40` does not fix Driscoll-Kraay (it is consistent as `T → ∞`, not in `N`); Conley does improve. No finite-sample correction is applied anywhere — which is what makes `spillover_orders=()` reproduce `panel_lp` bit for bit — so every band here is asymptotic and measurably too narrow on a regional panel. **Read a `γ` whose p-value sits near the threshold as undecided.**

The default `cov_type` is `'driscoll-kraay'` rather than `panel_lp`'s `'cluster'`, deliberately: cluster-by-entity assumes independence across entities, which this regression denies by construction, since `(Wx)_i` is a function of other units' shocks. For the `T ≈ 40–100` regional panels this module targets, `'conley'` is the recommended choice.

Two more identification notes. `γ` is identified purely off the cross-sectional variation in `Wx` given `x`, so a near-uniform shock, a dense `W`, or a `W` chosen after looking at the results all give weakly identified or specification-searched estimates; `spillover['corr_own']` reports the within-sample correlation of the two-way-demeaned own and spillover columns, and above 0.999 a `RuntimeWarning` fires. And `x` enters as **exogenous** conditional on the fixed effects and lags — no IV path ships, even though the designs this module supports (Chodorow-Reich 2019; Auerbach, Gorodnichenko and Murphy 2020; Dupor et al. 2023) instrument regional spending with military procurement or shift-share exposure. Build that instrument with `shift_share_iv` from §4 and treat it as the shock.

---

## 10. Practical checklist

- **Bandwidth.** Report Conley errors at two or three cutoffs. If they keep growing with the cutoff, the spatial correlation is not local and Driscoll-Kraay (`panel_lp_dk`) or a larger cutoff is the honest choice.
- **Islands.** A unit without neighbours has a zero spatial lag; check `W.n_islands` before Moran's I and use `distance_weights` with a larger cutoff or `knn_weights` if islands appear. An island also breaks the `1/(1−ρ)` shortcut for the average total impact — which is why §7 never uses it.
- **Coordinates.** Latitude first, longitude second, in degrees. `metric="euclidean"` treats the columns as planar distances in the same unit as the cutoff.
- **Shift-share.** The AKM error is valid when the shocks are as-good-as-random across sectors; when the identification comes from the shares instead, follow Goldsmith-Pinkham, Sorkin and Swift and inspect the Rotemberg weights.
- **Specify before you fit.** Run `ols_spatial` first and read `lm.recommendation`. It is a local test under the null of no dependence: it says which alternative the residuals point at, not that the indicated model is right. Two robust rejections mean SDM or SARAR, not "pick the bigger statistic".
- **Never report a spatial-lag `β` as an effect.** Report `.effects()` — direct, indirect and total — and say which one you are talking about. The indirect effect is the *sum* of all `n(n−1)` off-diagonal cross-partials divided by `n` — the average total spillover a unit sends, equivalently receives — not the effect on one neighbour, and not the mean individual cross-partial (which is smaller by a factor of `n−1`). SEM has no such decomposition, by construction.
- **Say which `W`, and show a second one.** `ρ` on a binary `W` is not comparable with `ρ` on its row-standardised version, and no standard error of any kind is robust to a mis-specified `W`. Report contiguity, k-nearest-neighbour and economic weights side by side.
- **Heteroskedasticity is not a standard-error problem here.** Under heteroskedasticity of unknown form the Gaussian SAR ML estimate of `ρ` is itself inconsistent, which is why `vcov='robust'` with `method='ml'` raises. Refit with `method='gmm'`.
- **Panels: check what you actually corrected.** `spatial_panel` defaults to the Lee-Yu transformation, whose bias correction is provably zero — `bias_correction='none'` there is a statement, not a skip. Panels must be balanced, must be static, and must fit under `dense_max = 5000` units. `vcov='dk'`/`'conley'` replace the `β` block only.
- **Spatial local projections: `total` is a contrast, not a multiplier.** The two-way time effect absorbs exactly the uniform-shock variation, so no aggregate multiplier is identified. Quote `total` as a cross-sectional contrast, use the per-horizon tests one at a time (the joint cross-horizon test is not shipped because its measured size was 0.31 against a nominal 0.10), and treat a marginal `γ` as undecided.

## References

- Adão, R., Kolesár, M. and Morales, E. (2019). Shift-share designs: theory and inference. *Quarterly Journal of Economics* 134(4), 1949–2010.
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
- Anselin, L., Bera, A. K., Florax, R. and Yoon, M. J. (1996). Simple diagnostic tests for spatial dependence. *Regional Science and Urban Economics* 26(1), 77–104.
- Auerbach, A. J., Gorodnichenko, Y. and Murphy, D. (2020). Local fiscal multipliers and fiscal spillovers in the USA. *IMF Economic Review* 68, 195–229.
- Chodorow-Reich, G. (2019). Geographic cross-sectional fiscal spending multipliers: what have we learned? *American Economic Journal: Economic Policy* 11(2), 1–34.
- Cliff, A. D. and Ord, J. K. (1972). Testing for spatial autocorrelation among regression residuals. *Geographical Analysis* 4(3), 267–284.
- Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics* 92(1), 1–45.
- Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics* 80(4), 549–560.
- Dupor, B., Karabarbounis, M., Kudlyak, M. and Mehkari, M. S. (2023). Regional consumption responses and the aggregate fiscal multiplier. *Review of Economic Studies* 90(6), 2982–3021.
- Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
- Goldsmith-Pinkham, P., Sorkin, I. and Swift, H. (2020). Bartik instruments: what, when, why, and how. *American Economic Review* 110(8), 2586–2624.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic production in the Caribbean and Central America. *PNAS* 107(35), 15367–15372.
- Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review* 95(1), 161–182.
- Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial two-stage least squares procedure for estimating a spatial autoregressive model with autoregressive disturbances. *Journal of Real Estate Finance and Economics* 17(1), 99–121.
- Kelejian, H. H. and Prucha, I. R. (1999). A generalized moments estimator for the autoregressive parameter in a spatial model. *International Economic Review* 40(2), 509–533.
- Kelejian, H. H. and Prucha, I. R. (2010). Specification and estimation of spatial autoregressive models with autoregressive and heteroskedastic disturbances. *Journal of Econometrics* 157(1), 53–67.
- Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood estimators for spatial autoregressive models. *Econometrica* 72(6), 1899–1925.
- Lee, L.-F. and Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics* 154(2), 165–185.
- LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Lin, X. and Lee, L.-F. (2010). GMM estimation of spatial autoregressive models with unknown heteroskedasticity. *Journal of Econometrics* 157(1), 34–52.
- Ord, J. K. (1975). Estimation methods for models of spatial interaction. *Journal of the American Statistical Association* 70(349), 120–126.
- Pace, R. K. and LeSage, J. P. (2004). Chebyshev approximation of log-determinants of spatial weight matrices. *Computational Statistics & Data Analysis* 45(2), 179–196.
- Yu, J., de Jong, R. and Lee, L.-F. (2008). Quasi-maximum likelihood estimators for spatial dynamic panel data models with fixed effects when both n and T are large. *Journal of Econometrics* 146(1), 118–134.
