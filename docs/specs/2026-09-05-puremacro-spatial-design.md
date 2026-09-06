# `puremacro.spatial` — spatial econometrics for regional macro

**Status:** Drafted 2026-09-05. Architectural spec for a spatial layer that sits between the existing regional-panel estimators (`lp.panel_lp`, `lp.panel_lp_dk`, `lp.cce_panel_lp`, `var.panel.mean_group_svar`, `bartik.*`, `connectedness.*`) and the data builders (`build_subnational_panel`, `fetch.state_industry_panel`, `bartik.county_epu`). Implementation in four phases; phase 1 ships with 2.4.0.
**Target releases:** 2.4.0 (phase 1 — weights, diagnostics, spatial HAC, AKM), 2.5.0 (phase 2 — spatial LP spillovers, GVAR), 2.6.0 (phase 3 — SAR/SEM/SDM, spatial panels), 2.7.0 (phase 4 — spatial DiD, county examples).
**Driving lenses:** regional fiscal-multiplier and exposure designs need inference that respects distance; everything must stay inside the four-package Pyodide core; every estimator ships with a reference golden.

## Motivation

puremacro already estimates regional panels (two-way FE local projections with cluster or Driscoll-Kraay errors, CCE, mean-group SVARs), builds shift-share exposures (`bartik.build_exposure_iv`, `rotemberg_weights`) and measures spillovers in the time domain (`connectedness.spillover_index`, `bk2018`). What it lacks is any representation of *where* the units are:

1. **Inference ignores spatial correlation.** Cluster-by-entity errors treat neighbouring states as independent; Driscoll-Kraay is robust to arbitrary cross-sectional dependence only at the cost of a time-series bandwidth that regional panels with T ≈ 40-100 cannot support. Conley (1999) HAC, and its space-time version, is the standard fix and is absent.
2. **Shift-share inference uses the wrong clustering.** Adão, Kolesár and Morales (2019) show that regional clustering of shift-share regressions under-covers badly when shocks are the random element; the shock-level standard error is a one-line formula that puremacro does not offer.
3. **No spatial lags.** A regional LP cannot enter `W @ shock` to estimate spillovers; a SAR/SDM cross-section cannot be estimated; there is no Moran's I to even diagnose dependence.
4. **No global VAR.** The GVAR of Pesaran, Schuermann and Weiner (2004) is the macro-spatial workhorse (country VARX* models linked by trade weights) and maps directly onto the existing VAR stack.

## Non-goals

- **No** geometry stack. No shapely, geopandas, pyproj or libpysal at runtime: they do not run under Pyodide and the estimators need only neighbour lists or centroids. PySAL (`esda`, `spreg`) is used exactly the way statsmodels is: as a reference package in the `dev` extra for goldens.
- **No** breaking change to `panel_lp` / `panel_lp_dk` / `lp_iv` / `bartik.*`. New covariance types and functions are additive.
- **No** maps in phase 1. `plot()` on the weights object draws the sparsity pattern and neighbour-count histogram; choropleths need geometry and are out of scope.
- **No** Bayesian spatial models.

## Architecture

### Module map (phase 1 in full; later phases as stubs)

```
puremacro/spatial/
├── __init__.py        public surface (curated __all__)
├── weights.py         SpatialWeights (scipy.sparse CSR), builders:
│                      contiguity_weights, knn_weights, distance_weights,
│                      economic_weights; haversine_km
├── diagnostics.py     morans_i, gearys_c  -> MoranResult / GearyResult
├── hac.py             conley_cov, conley_se, spatial_hac_panel_cov
│                      (space-time Conley; Bartlett or uniform kernels)
└── _results.py        frozen dataclasses with the presentation contract

puremacro/bartik/
└── akm.py             [new]  shift_share_iv -> ShiftShareIVResult
                              (robust and AKM shock-level SEs, first-stage F,
                              Rotemberg weights reused from exposure_iv)

puremacro/lp/
├── panel.py           [extended] cov_type='conley' (+ coords=, cutoff_km=,
│                       time_lags=, kernel=)
└── _panel_helpers.py  [extended] _focal_conley_se factory

docs/spatial.md, docs/es/spatial.md      user guide (runnable blocks)
tests/test_spatial/                      unit + reference tests
tests/test_dgp_adversarial.py            [extended] spatial DGPs
```

Phase 2 adds `spatial/lp.py` (`spatial_lp`: own and `W @ shock` responses with direct / indirect / total effects) and `var/gvar.py` (`gvar`: country VARX* with foreign-variable weights, weak-exogeneity tests, generalised IRFs). Phase 3 adds `spatial/models.py` (`sar`, `sem`, `sdm` by ML with eigenvalue or Chebyshev log-determinant, Kelejian-Prucha GMM, LeSage-Pace effect decompositions) and `spatial/panel.py` (spatial FE panels with the Lee-Yu correction). Phase 4 adds `did/spatial_did.py` (exposure rings) and county-level examples.

### Weights

`SpatialWeights` is a frozen dataclass wrapping a `scipy.sparse.csr_matrix` `W` of shape `(n, n)` with unit `ids`, a `kind` tag and a `row_standardized` flag. Methods: `lag(x)` (`W @ x` for arrays, Series or DataFrames aligned on `ids`), `standardize()`, `neighbors(id)`, `n_islands`, `to_dense()`, `to_frame()` (edge list), `summary()`, `to_markdown/to_latex/to_typst` (neighbour-count table), `plot()`.

Builders return row-standardised weights by default:

| builder | input | weight |
|---|---|---|
| `contiguity_weights(neighbors)` | `{id: [neighbour ids]}` | 1 for each listed pair (symmetrised) |
| `knn_weights(coords, k)` | `(n, 2)` lat/lon or planar | 1 for the k nearest |
| `distance_weights(coords, cutoff)` | as above plus a cutoff | `1/d^power`, uniform, or Gaussian inside the cutoff |
| `economic_weights(flows)` | `(n, n)` flow matrix | row shares, zero diagonal |

Distances use `haversine_km` for `metric="haversine"` (lat/lon in degrees) and Euclidean otherwise.

### Diagnostics

`morans_i(x, W, n_perm=999, seed=0)` returns `I`, its expectation `-1/(n-1)`, the variance under the normality and randomisation assumptions, the analytic z and p, and the permutation p-value. `gearys_c` mirrors it. Both return frozen result objects with the five presentation methods; `MoranResult.plot()` is the Moran scatterplot.

### Spatial HAC

`conley_cov(X, resid, coords, cutoff_km, kernel="bartlett", metric="haversine")` returns `(X'X)^{-1} S (X'X)^{-1}` with `S = Σ_i Σ_j K(d_ij) u_i u_j'`, `u_i = X_i e_i`, `K` the Bartlett (`1 - d/cutoff`) or uniform kernel inside the cutoff. `conley_se` is its diagonal square root. The distance matrix is formed once in dense form; the documented cost is O(n²) memory, adequate for regional panels (n ≤ a few thousand).

`spatial_hac_panel_cov(X, resid, coords, entity_keys, time_keys, cutoff_km, time_lags, kernel)` is the space-time version: `S = Σ_{t,s} w(|t-s|) Σ_{i,j} K(d_ij) u_it u_js'` with a Bartlett kernel in time of bandwidth `time_lags`. With `time_lags=0` it reduces to Conley period by period; with `cutoff_km=0` it reduces to a heteroskedasticity-robust estimator; with a cutoff larger than every distance and `time_lags=0` it reduces to clustering by period.

`panel_lp(..., cov_type="conley", coords=<DataFrame indexed by entity with lat/lon or x/y>, cutoff_km=500, time_lags=None)` wires the panel version into the two-way FE local projections; `time_lags=None` uses the Driscoll-Kraay bandwidth rule on the number of periods.

### Shift-share inference

`bartik.shift_share_iv(df, y, x, shares, shocks, controls=(), weights=None, se="akm")` estimates the just-identified 2SLS with instrument `z_i = Σ_k s_ik g_k` after partialling out the controls. Standard errors: `"robust"` (HC1) and `"akm"` (Adão-Kolesár-Morales 2019, eq. 24): with `x̃_i` the residualised regressor and `ε̂_i` the 2SLS residuals,

```
SE_AKM = sqrt( Σ_k ĝ_k² ( Σ_i s_ik x̃_i ε̂_i )² ) / | Σ_i z_i x̃_i |
```

where `ĝ_k` are the shocks residualised on the share-weighted constant (and on sector-level controls when given). The result carries both standard errors, the first-stage F, and the Rotemberg weights.

### Result objects

All new results are frozen dataclasses implementing `summary()`, `to_frame()`, `to_markdown()`, `to_latex()`, `to_typst()` and `plot()`, rendered through the `puremacro.reports` helpers, and are added to the public API snapshot.

## Validation

| what | reference | tolerance |
|---|---|---|
| `morans_i`, `gearys_c` on synthetic lattices | `esda.Moran`, `esda.Geary` (dev extra, skipped when absent) | 1e-10 on the statistic, 1e-8 on the normality z |
| `conley_cov` | brute-force double loop in the test; cutoff → 0 equals HC0; cutoff → ∞ with far-apart clusters equals cluster-robust without small-sample correction | 1e-10 |
| `spatial_hac_panel_cov` | `time_lags=0` equals per-period Conley; cutoff → ∞ equals Driscoll-Kraay meat | 1e-10 |
| `shift_share_iv` AKM | Monte Carlo coverage ≥ 0.93 at nominal 0.95 when shocks are iid and shares fixed; robust SE demonstrably under-covers in the same design | — |
| weights builders | symmetry, row sums, island detection, haversine against known city distances | exact / 1 km |

Adversarial DGPs: an island unit, a fully disconnected graph, duplicate coordinates, a cutoff smaller than the minimum distance, a single-period panel, sectors with zero total share.

## Rollout

1. Phase 1 (this spec's implementation): weights, diagnostics, spatial HAC, `panel_lp(cov_type="conley")`, `shift_share_iv`, docs (EN/ES), tests, goldens, CHANGELOG. No behaviour change for existing calls.
2. Phase 2: `spatial_lp`, `gvar`; the `connectedness` page cross-links the structural counterpart.
3. Phase 3: cross-section and panel spatial models with `spreg` goldens.
4. Phase 4: spatial DiD and county-level worked examples using `bartik.county_epu` and the shipped crosswalks.
