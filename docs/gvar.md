> 🇬🇧 English · 🇪🇸 [Español](es/gvar.md)

# Global VAR (GVAR)

Thirty countries and four variables each make a 120-variable system. A VAR(2) on it has 28,800 autoregressive coefficients and perhaps 150 quarterly observations to estimate them from: the stacked global VAR is not badly estimated, it is not estimable at all. The GVAR of Pesaran, Schuermann and Weiner (2004) buys the interdependence back by never estimating that system. Each country is a small VARX\* in its own variables plus a *foreign* aggregate built from trade weights, every country is estimated separately, and the global model is then **assembled** from the country blocks by exact matrix algebra rather than fitted.

`puremacro.var.gvar` implements that pipeline: `star_variables` builds the foreign block, `gvar` estimates the country models and solves the link system in one call, and `solve_gvar` is the pure linear-algebra step exposed on its own so it can be read and tested without any estimation. Pure numpy/scipy/pandas, so it runs under Pyodide.

```text
from puremacro.var import gvar

res = gvar(panel, W, p=2, q=1)          # panel: (country, date) frame or {country: frame}
res.summary()                           # country blocks, cond(G), stability, weak exogeneity
res.girf("US", "r", horizon=24, n_boot=500)     # generalised responses, pointwise bands
res.gfevd(horizon=24); res.pp(b); res.forecast(8)
```

---

## 1. The country model

Country `i` carries `k_i` variables `x_it` and the star aggregate `x*_it = Σ_j w_ij x_jt`, where `w_ij` is `i`'s trade share with `j` (zero diagonal, rows summing to one). Its VARX\*(p_i, q_i) is

$$x_{it} = a_{i0} + a_{i1}t + \sum_{l=1}^{p_i}\Phi_{il}x_{i,t-l} + \Lambda_{i0}x^{*}_{it} + \sum_{l=1}^{q_i}\Lambda_{il}x^{*}_{i,t-l} + \sum_{l=0}^{q_d}\Psi_{il}d_{t-l} + u_{it},$$

with `d_t` an optional strictly exogenous global block (oil prices are the canonical case). Every equation of country `i` shares one regressor matrix, so seemingly-unrelated regression collapses to equation-by-equation OLS and `gvar` runs it that way.

Two separate assumptions license that OLS, and the module keeps them separate because they fail separately:

- **Weak exogeneity** of `x*_it` for the parameters of country `i` (Pesaran, Shin & Smith 2000) — the country's own innovations must not feed back into its foreign block within the period. `gvar` reports an approximate test for it (§6).
- **Granularity**, `max_j w_ij → 0` as `N` grows. With few countries a single partner carries a large weight, `x*_it` is contemporaneously correlated with `u_it`, and the country OLS is *inconsistent*. Nothing in the code can repair this, and every small example below — six units, weights up to 0.69 — is econometrically wrong on exactly that count. It is pedagogy, not an estimate.

### 1.1 Building the weight matrix and the star variables

Trade weights are just a row-standardised flow matrix with a zero diagonal, which is what `puremacro.spatial.economic_weights` produces; `gvar` accepts a `SpatialWeights`, a labelled `DataFrame` or a bare `(N, N)` array (with `country_order`). Here the "countries" are six US states and the flows are a gravity approximation built from the shipped Census geography — land area over great-circle distance between the state internal points.

```python
import numpy as np
import pandas as pd
from puremacro.datasets import load_us_state_centroids
from puremacro.spatial import economic_weights, pairwise_distances

states = ["CA", "TX", "NY", "FL", "IL", "PA"]
geo = load_us_state_centroids().loc[states]
D = pairwise_distances(geo[["lat", "lon"]].to_numpy(), "haversine")     # km
size = geo["land_sqmi"].to_numpy()
flows = np.outer(size, size) / np.where(D > 0.0, D, 1.0)                # gravity flows
np.fill_diagonal(flows, 0.0)
W = economic_weights(pd.DataFrame(flows, index=states, columns=states)) # row-standardised
Wf = pd.DataFrame(W.to_dense(), index=states, columns=states)
print(Wf.round(3))
print("largest single weight per row:", Wf.max(axis=1).round(2).to_dict())
```

Those largest weights run from 0.38 to 0.69. In a real GVAR they would be under 0.2; print them every time, because they are the granularity condition made visible.

`star_variables` builds the foreign panel on its own, so it can be plotted or fed to another estimator without fitting anything. Country `i`'s star weight for variable `v` is renormalised over the countries that actually carry `v`,

$$\tilde w_{ij}^{(v)} = \frac{w_{ij}\,\mathbf 1\{v \in K_j\}}{\sum_{m \neq i,\ v \in K_m} w_{im}},$$

so a row still sums to one when some partners lack the variable.

```python
from puremacro.var.gvar import star_variables

rng = np.random.default_rng(11)
T, N = 160, len(states)
Wd = W.to_dense()
Y, R = np.zeros((T, N)), np.zeros((T, N))
for t in range(1, T):                       # a state-level VARX* system, by construction
    ystar, rstar = Wd @ Y[t - 1], Wd @ R[t - 1]
    common = rng.standard_normal()          # one national factor in the innovations
    Y[t] = 0.55 * Y[t-1] + 0.25 * ystar - 0.08 * R[t-1] + 0.5 * common + rng.standard_normal(N)
    R[t] = 0.50 * R[t-1] + 0.20 * rstar + 0.10 * Y[t-1] + 0.3 * common + 0.7 * rng.standard_normal(N)
dates = pd.RangeIndex(T, name="date")
panel = {s: pd.DataFrame({"y": Y[:, i], "r": R[:, i]}, index=dates) for i, s in enumerate(states)}

xstar = star_variables(panel, W)             # (country, date) frame, columns y_star, r_star
print(xstar.loc["CA"].head(3).round(3))
```

The panel may be a `(country, date)` MultiIndexed frame or a mapping `country -> frame`. It must be balanced, gap-free and numeric: an interior NaN always raises, and so does an entirely NaN `(country, variable)` cell unless you opt in (§8).

---

## 2. Estimating the country blocks

```python
from puremacro.var import gvar

res = gvar(panel, W, p=1, q=1)
print(res.summary())
print(res.country_models["CA"].summary())
```

`p` and `q` accept an int or a per-country mapping; `ic="aic"`/`"bic"` selects `(p_i, q_i)` per country on a common grid sample instead. Every country is fitted over the *same* effective sample `t = t0 … T-1` with `t0 = max(s, q_d)` and `s = max_i max(p_i, q_i)`. That is deliberately not configurable: it is what makes the residual vectors align by date, `Sigma_eps` a genuine contemporaneous covariance, and the link identity below exact at every date.

`res.country_models[c]` is a `CountryVARX` with `coef`, `se`, `tstat`, `pvalue` (indexed by regressor, columns the country's own variables), the sliced `Phi`, `Lambda`, `Psi` matrices, `resid`, `fitted`, `r2`, `aic`/`bic` and `condition_number`. `res.coef_frame()` flattens every country into one long table.

Two warnings the summary prints and you should not read past. The `t` statistics are conditional on weak exogeneity, and on levels data they are **not asymptotically normal** (Sims, Stock & Watson 1990) — with I(1) variables a coefficient on a lagged level has a non-standard limit. And `trend='ct'` puts an unrestricted linear trend in a levels VARX\*, which implies a *quadratic* trend in the levels of the solved global system; VECX\* GVARs restrict the trend to the cointegrating space, and this module has no cointegrating space to restrict it to.

---

## 3. Solving the global system

This is the step that makes it a *global* model, and it is exact algebra rather than estimation. Order the global vector `x_t = (x'_1t, …, x'_Nt)'`, `k = Σ_i k_i`. Let `S_i` (`k_i × k`) select country `i`'s own variables and `W*_i` (`k*_i × k`) build its star variables, so that

$$z_{it} = \begin{pmatrix}x_{it}\\ x^{*}_{it}\end{pmatrix} = W_i x_t, \qquad W_i = \begin{bmatrix}S_i\\ W^{*}_i\end{bmatrix}.$$

Write `A_i0 = [I, −Λ_i0]` and `A_il = [Φ_il, Λ_il]`, zero-padded to `s`. Each country equation is then `A_i0 W_i x_t = a_i0 + a_i1 t + Σ_l A_il W_i x_{t−l} + u_it`, and stacking over `i` gives one square system

$$G x_t = a_0 + a_1 t + \sum_{l=1}^{s} H_l x_{t-l} + \sum_l \Upsilon_l d_{t-l} + \varepsilon_t,$$

whose row block `i` is `G_i = S_i − Λ_i0 W*_i` and `H_{l,i} = Φ_il S_i + Λ_il W*_i`. When `G` is non-singular, `x_t = G^{-1}(...)` is an ordinary VAR(s) with coefficient matrices `F_l = G^{-1} H_l` and reduced-form innovations `G^{-1} ε_t`. `res.G`, `res.H`, `res.F`, `res.G_inv`, `res.eigenvalues`, `res.max_eigenvalue` and `res.stable` are all on the result.

`solve_gvar` exposes the step by itself. Two one-variable countries make the whole mechanism visible — and make visible how it fails:

```python
from puremacro.var.gvar import solve_gvar

S = {"A": np.array([[1.0, 0.0]]), "B": np.array([[0.0, 1.0]])}
Wstar = {"A": np.array([[0.0, 1.0]]), "B": np.array([[1.0, 0.0]])}
link = {c: np.vstack([S[c], Wstar[c]]) for c in ("A", "B")}
A0 = {"A": np.array([[1.0, -0.3]]), "B": np.array([[1.0, -0.2]])}       # [I, -Lambda_0]
A_lags = {"A": [np.array([[0.5, 0.1]])], "B": [np.array([[0.4, 0.0]])]} # [Phi_1, Lambda_1]
sol = solve_gvar(link, A0, A_lags, country_order=["A", "B"], block_sizes={"A": 1, "B": 1})
print(sol.G.round(3), sol.F[0].round(3), sep="\n")
print("max |eig| =", round(float(np.abs(np.linalg.eigvals(sol.F[0])).max()), 4))

A0_bad = {"A": np.array([[1.0, -2.0]]), "B": np.array([[1.0, -0.5]])}   # l1 * l2 = 1
try:
    solve_gvar(link, A0_bad, A_lags, country_order=["A", "B"], block_sizes={"A": 1, "B": 1})
except np.linalg.LinAlgError as exc:
    print(str(exc)[:118], "...")
```

With one variable each, `G = [[1, −λ₁], [−λ₂, 1]]` is singular exactly when `λ₁λ₂ = 1`: the contemporaneous cross-country system `x_t = Λ_0 W* x_t + …` then has no unique solution. The error names the country row and column blocks loading on the null space, which is how you find the culprit in a 30-country model.

**Read `condition_number`, never `condition_number_raw`.** Under a diagonal rescaling of the data `G → D G D^{-1}`, a similarity transform that leaves the eigenvalues and the determinant alone but can move the raw condition number by eleven orders of magnitude — quote one series in millions and a well-posed model looks singular. `G` is therefore equilibrated (LAPACK balancing plus Ruiz sweeps) before the conditioning gate, `res.condition_number` is that unit-invariant number, and `condition_number_raw` is kept for reference only. Country design matrices are column-equilibrated before their own singularity check for the same reason, with the scaling undone exactly in the coefficients, standard errors and `(Z'Z)^{-1}`.

**Instability is a flag, not an error.** A GVAR on I(1) levels carries unit roots by construction, so `max |eigenvalue| ≥ 1` is expected and `gvar` only warns (`stability_warn=False` silences it). Bootstrap bands drawn from an explosive solved system, however, should not be trusted.

---

## 4. Generalised impulse responses

With `Ψ_h` the MA coefficients of the solved VAR(s) and `R_h = Ψ_h G^{-1}`, the response to a one-standard-deviation innovation in the combination `d` of the country equations is

$$\mathrm{GIRF}_h(d) = \frac{R_h \Sigma_\varepsilon d}{\sqrt{d'\Sigma_\varepsilon d}}.$$

*Generalised*, not orthogonalised, and that is a decision rather than a convenience. A Cholesky factorisation of `Σ_ε` needs an ordering of all `k` country-variables; nobody can defend one, and the answer changes with it. The generalised response is invariant to the ordering because it never imposes one — but the price is that it is a conditional expectation given one innovation with the others at their conditional means, **not** the effect of an independent structural shock. Responses to different `(country, variable)` pairs do not decompose anything. This module offers no other route: no Cholesky, sign or narrative identification of the global system.

```python
g = res.girf("TX", "y", horizon=16, n_boot=200, ci=0.90, seed=0)
print(g.summary())

gc = res.girf_combination({("TX", "r"): 1.0, ("CA", "r"): 1.0},
                          label="joint TX + CA rate innovation", horizon=12)
print(gc.to_frame().query("h == 0").head(4).round(4))
```

`girf` shocks one country-variable; `girf_combination` takes a mapping of `(country, variable)` (or `"country:variable"`) to weights. Because of the `sqrt(d' Σ_ε d)` normalisation the path is invariant to the positive scale of `d`, so `1.0` and `3.0` give identical responses. `GVARGIRFResult` carries `irf` `(H+1, k)`, `lower`/`upper`/`median`, `names`, `shock_sd`, and `plot(by_variable=True)` draws the standard GVAR figure — one panel per variable, one line per country.

The bands are **pointwise percentile bands from a recursive-residual bootstrap**: never joint over horizons, and with no bias correction, which is why `median` (the bootstrap median) differs from `irf` (the point estimate) by the OLS bias. Each replication resamples whole rows of the stacked residual matrix (`bootstrap='iid'`) or flips their signs (`'wild'`), simulates the solved system forward, rebuilds the star panel from the *same* weights, re-estimates every country block at **fixed** lag orders and re-solves. Replications that explode or fail to estimate are dropped and counted in `n_boot_failed`; more than 10% dropped triggers a warning and the surviving band is biased inward.

One quiet scale trap: `sigma_ddof` (default 0, the MLE divisor the GVAR literature uses) is not cosmetic. The GIRF is homogeneous of degree ½ in `Σ_ε`, so changing the divisor scales every response by `sqrt(T_eff / (T_eff − sigma_ddof))`.

---

## 5. GFEVD, persistence profiles, forecasts

```python
theta = res.gfevd(horizon=8)                       # (H+1, k, k), rows sum to 1
raw = res.gfevd(horizon=8, normalize=False)
print("unnormalised row sums, h = 0:", raw[0].sum(axis=1).round(3))
print("unnormalised row sums, h = 8:", raw[8].sum(axis=1).round(3))

pp = res.pp({("TX", "y"): 1.0, ("CA", "y"): -1.0}, horizon=20)
print("persistence profile:", pp[:5].round(3), "->", round(float(pp[-1]), 4))

print(res.forecast(4).round(3).iloc[:, :4])
```

**The generalised FEVD.** `gfevd` is Pesaran-Shin (1998) with `Ψ_l` replaced by `R_l` and `Σ` by `Σ_ε`. The shares are non-negative and, with the default `normalize=True` (Diebold-Yılmaz), each row sums to exactly one. The unnormalised rows are a different matter, and the block above prints values below one at impact:

> **The Pesaran-Shin `≥ 1` bound on the unnormalised row sum does not hold for a GVAR, not even at `h = 0`.** Their proof needs `Ψ_0 = I`, so that row `i` at impact is `Σ_j r_ij² ≥ r_ii² = 1`. A GVAR has `R_0 = G^{-1}` instead, and the impact row sum is a Rayleigh quotient of `C C'` with `C` the column-normalised `Σ_ε^{1/2}`, so all that can be said is
>
> $$\lambda_{\min}(\mathrm{corr}(\Sigma_\varepsilon)) \le \mathrm{rowsum}_i(h=0) \le \lambda_{\max}(\mathrm{corr}(\Sigma_\varepsilon)).$$
>
> The `≥ 1` bound returns only when `G = I` — no contemporaneous star block anywhere. This is not a rare edge case: on the module's own three-country test DGP, sweeping only the seed, 50 of 59 fits have an impact row sum below one, and for `h ≥ 1` values near 0.28 occur on ordinary three-variable systems. **Never read an unnormalised row sum as a completeness check.**

**Persistence profiles.** `pp(b, horizon=…)` is Pesaran & Shin (1996),

$$\mathrm{PP}(b, h) = \frac{b' R_h \Sigma_\varepsilon R_h' b}{b' R_0 \Sigma_\varepsilon R_0' b},$$

exactly 1 at `h = 0` by construction. A profile that decays to zero marks an asymptotically stationary — cointegrating — combination; one converging to a positive constant marks an I(1) combination. Pass `b` as an array of length `k` or a mapping `(country, variable) -> weight`.

**Forecast.** `forecast(steps)` iterates the solved system's conditional mean forward from the end of the sample; it is a point path, not a density, and returns a frame indexed `1..steps`. If the model was fitted with `exog`, a matching `exog_future` is required — the solved system *conditions* on `d_t` and never shocks it.

**Watch `T_eff` against `k`.** `Σ_ε` is a `k × k` matrix estimated from `T_eff` observations, and the canonical GVAR has `k ≫ T_eff`, in which case it is singular. `gvar` warns at `T_eff < 2k` and again at `T_eff < k` rather than silently regularising: there is no shrinkage estimator here. GIRFs still evaluate on a singular `Σ_ε`, but the GFEVD and the bootstrap bands do not mean much. The six-state example above has `k = 12` and `T_eff = 159`, comfortably clear.

---

## 6. The weak-exogeneity test — what it is and is not

```python
we = res.weak_exogeneity
print(we.summary())
print(we.table.sort_values("p_value").head(4).round(4))
```

For each star variable of country `i` the auxiliary regression (in first differences by default, since canonical GVAR inputs are I(1) and the F statistic needs stationary regressors)

$$\Delta x^{*}_{il,t} = \mu + \sum_a \psi_a' \Delta x^{*}_{i,t-a} + \sum_a \phi_a' \Delta x_{i,t-a} + \gamma' \hat u_{i,t-1} + \eta_t$$

is run and `γ = 0` tested by `F(k_i, T_e − m)`. The whole lagged star block enters, as in Dees et al. (2007, eq. 15). Now the caveats, which are the point of the section:

- It is an **approximate reduced-form** test, **not** the error-correction test of Dees et al. (2007, eq. 15). That test uses the estimated cointegrating terms of a VECX\*; this module has no cointegration layer, so the country model's own lagged residual `û_{i,t−1}` stands in for them. The null tested here is the necessary condition that country `i`'s model innovations do not Granger-cause its own foreign block.
- `û_{i,t−1}` is a **generated regressor and no generated-regressor correction is applied**, so the F distribution is approximate in a second, independent sense.
- **Measured size, disclosed rather than omitted.** Under the null (three independent countries, `p = q = 1`, 1000 replications, 6000 tests per cell) the rejection rate is not the nominal one. On I(1) levels — the canonical GVAR input — it is 0.027/0.109/0.178 at nominal 0.01/0.05/0.10 with `T = 200`, and 0.035/0.114/0.182 with `T = 300`. The distortion is *stable in T*, which identifies it as the unaddressed generated-regressor problem rather than a small-sample artefact. On stationary data (AR(1), ρ = 0.5, `T = 200`) it is conservative instead: 0.002 at nominal 0.05.

So the result ships as a **diagnostic ordering of p-values, not a calibrated decision rule**. There is no `expected_rejections` attribute, and the boolean column is named `reject_nominal` to say plainly that `alpha` is a nominal threshold and not the test's size. Rank by `p_value`, or by `p_holm` for a family-wise ordering, and treat a small p-value as a reason to look at that country's star block. Do not count rejections and do not compare `n_reject_nominal` with `alpha × n_tests`.

---

## 7. Practical checklist

- **Build the weights from flows, not from geography, when you can.** `economic_weights(trade_matrix)` row-standardises and zeroes the diagonal, which is exactly what `gvar` validates for: non-negative, zero diagonal, rows summing to one within 1e-8. Fixed weights only — a mapping of frames or a 3-d array raises rather than silently averaging.
- **Print the largest weight in each row.** That number *is* the granularity condition. Above roughly 0.2 the country OLS is materially inconsistent and the standard errors understate the damage.
- **How much `T`?** Country `i` needs `T_eff − m_i ≥ k_i + 1` just to have a non-singular `Σ_i` (below that `gvar` raises, naming the country); the summary flags fewer than `k_i + 5` residual degrees of freedom. For the *global* objects the binding constraint is `Σ_ε`: aim for `T_eff > 2k`, and know that the canonical `k ≫ T_eff` GVAR reports GFEVDs and bands from a singular covariance.
- **When `G` is ill-conditioned.** First check `condition_number` (equilibrated), not the raw one. Then restrict the contemporaneous star block — `contemporaneous_star={"US": ("po",)}` is how the canonical model excludes contemporaneous foreign `y*` from the US equations — and check the trade weights for a row that concentrates on one partner. `cond_tol` (default 1e12) raises rather than returning a solved system you cannot use.
- **Lag orders.** The default `p = 2, q = 1` is the canonical GVAR choice. `ic=` selects per country, but the criteria are comparable **within one grid only**: changing `max_p` changes the grid sample and hence every candidate's log-likelihood.
- **Nothing is silently ignored.** `max_p`/`max_q` without `ic`, `exog_lags` without `exog`, `we_lags`/`alpha` with `weak_exogeneity=False`, `ci`/`seed`/`bootstrap` with `n_boot=0` — each raises rather than doing nothing.

```python
res_ic = gvar(panel, W, ic="bic", max_p=2, max_q=1,
              contemporaneous_star={"CA": ("r",), "TX": ("r",)})
print(res_ic.lag_orders)
print(res_ic.country_models["CA"].contemporaneous_star,
      res_ic.country_models["NY"].contemporaneous_star)
print(f"cond(G) = {res_ic.condition_number:.3f}, max |eig| = {res_ic.max_eigenvalue:.3f}")
```

---

## 8. Deliberately out of scope

Everything below is a decision, not a gap. Reach for another tool rather than expecting a future release.

- **No VECX\*/cointegration layer.** The country models are unrestricted VARX\* in levels: no rank test, no restricted trend, no error-correction terms — and consequently the weak-exogeneity test of §6 is not the Dees et al. (2007, eq. 15) test.
- **No per-country variable sets by default.** Every country must carry every column of `data`; an entirely NaN `(country, variable)` raises, naming both, exactly like an interior NaN, because the commoner cause is a misnamed or accidentally dropped series that would otherwise estimate a different model in silence. The oil-producer / US asymmetry is an explicit opt-in: `allow_missing_variables=True` restores the permissive reading and emits a `RuntimeWarning` naming every dropped cell. A *star* block may name any variable some other country carries, even one country `i` does not hold itself. Genuinely per-country variable *definitions* are out of scope.
- **A variable endogenous in one country cannot be exogenous in another.** A name in both `data` and `exog` raises: the solved system would otherwise condition on a variable endogenous to one of its own blocks.
- **No structural identification of the global system** beyond the generalised route — no Cholesky, sign or narrative restrictions on the GVAR.
- **No time-varying (rolling) trade weights and no rolling-window estimation.**
- **No shocks to the global exogenous block `d_t`;** the solved system conditions on it.
- **No dominant-unit / factor-augmented GVAR** (Chudik & Pesaran 2011).
- **No structural-break testing** of the country equations.
- **No shrinkage for `Σ_ε`.** The module warns at `T_eff < 2k` and `T_eff < k` instead of silently regularising a singular covariance.
- **Bootstrap bands are pointwise percentile bands**, never joint over horizons, and carry no bias correction.

## References

- Chudik, A. and Pesaran, M. H. (2011). Infinite-dimensional VARs and factor models. *Journal of Econometrics* 163(1), 4–22.
- Chudik, A. and Pesaran, M. H. (2016). Theory and practice of GVAR modelling. *Journal of Economic Surveys* 30(1), 165–197.
- Dees, S., di Mauro, F., Pesaran, M. H. and Smith, L. V. (2007). Exploring the international linkages of the euro area: a global VAR analysis. *Journal of Applied Econometrics* 22(1), 1–38.
- Pesaran, M. H., Schuermann, T. and Weiner, S. M. (2004). Modeling regional interdependencies using a global error-correcting macroeconometric model. *Journal of Business & Economic Statistics* 22(2), 129–162.
- Pesaran, M. H. and Shin, Y. (1996). Cointegration and speed of convergence to equilibrium. *Journal of Econometrics* 71(1–2), 117–143.
- Pesaran, M. H. and Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters* 58(1), 17–29.
- Pesaran, M. H., Shin, Y. and Smith, R. J. (2000). Structural analysis of vector error correction models with exogenous I(1) variables. *Journal of Econometrics* 97(2), 293–343.
- Sims, C. A., Stock, J. H. and Watson, M. W. (1990). Inference in linear time series models with some unit roots. *Econometrica* 58(1), 113–144.
