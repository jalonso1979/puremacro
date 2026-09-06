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

## 5. Practical checklist

- **Bandwidth.** Report Conley errors at two or three cutoffs. If they keep growing with the cutoff, the spatial correlation is not local and Driscoll-Kraay (`panel_lp_dk`) or a larger cutoff is the honest choice.
- **Islands.** A unit without neighbours has a zero spatial lag; check `W.n_islands` before Moran's I and use `distance_weights` with a larger cutoff or `knn_weights` if islands appear.
- **Coordinates.** Latitude first, longitude second, in degrees. `metric="euclidean"` treats the columns as planar distances in the same unit as the cutoff.
- **Shift-share.** The AKM error is valid when the shocks are as-good-as-random across sectors; when the identification comes from the shares instead, follow Goldsmith-Pinkham, Sorkin and Swift and inspect the Rotemberg weights.

## References

- Adão, R., Kolesár, M. and Morales, E. (2019). Shift-share designs: theory and inference. *Quarterly Journal of Economics* 134(4), 1949–2010.
- Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics* 92(1), 1–45.
- Goldsmith-Pinkham, P., Sorkin, I. and Swift, H. (2020). Bartik instruments: what, when, why, and how. *American Economic Review* 110(8), 2586–2624.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic production in the Caribbean and Central America. *PNAS* 107(35), 15367–15372.
