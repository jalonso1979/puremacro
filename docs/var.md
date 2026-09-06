> 🇬🇧 English · 🇪🇸 [Español](es/var.md)

# VAR & FAVAR

`puremacro.var` fits a reduced-form VAR and then answers the only question a
reduced form cannot: *which* of the correlated innovations was the shock. Every
identification scheme in `puremacro.var.identify` takes the same reduced form
and differs only in what it is willing to assume — a recursive ordering, a
long-run neutrality, a sign pattern, an external instrument, a variance break.
Picking between them is the modelling decision; the code deliberately makes them
interchangeable so the choice can be made on economics rather than on which one
was easiest to wire up.

Nothing on this page touches the network. Everything takes an ndarray; `favar`
also accepts a DataFrame and keeps its column names.

```python
from puremacro.var import lag_select
from puremacro.var.identify import cholesky

Y = ...                                   # (T, n) ndarray, rows time-ordered
p = lag_select(Y, maxlags=8, ic="bic")
res = cholesky(Y, p=p, horizon=20, n_boot=500, ci=0.9, seed=0)
print(res.summary())
res.irf_point[4, 1, 0]                    # variable 1 at h=4 to shock 0
```

## The reduced form

`fit_var(Y, p)` is OLS with a constant. It returns a frozen
`VarEstimateResult`:

| attribute | shape | what it is |
|---|---|---|
| `A_list` | list of `p` arrays, each `(n, n)` | `A_1 … A_p`, lag 1 first |
| `c` | `(n,)` | intercept |
| `Sigma` | `(n, n)` | residual covariance, denominator `T − p − 1 − np` |
| `resid` | `(T − p, n)` | reduced-form residuals |
| `X` | `(T − p, 1 + np)` | design matrix, constant in column 0 |

`Sigma` carries the full degrees-of-freedom correction, not `T − p`. The
dataclass is iterable and indexable, so the five-tuple unpack every internal
caller uses still works:

```python
from puremacro.var import fit_var

A_list, c, Sigma, resid, X = fit_var(Y, p=2)     # equivalent
```

`lag_select(Y, maxlags=8, ic="bic")` returns an int; `ic` accepts `"aic"`,
`"hq"` or `"bic"`, and anything else falls through to BIC rather than raising.
It minimises `log|Σ_p|` plus the usual penalty in the coefficient count `n²p`
(intercepts excluded), each candidate evaluated on its own effective sample
`T − p`, and skips any `p` that fails to estimate.
`companion(A_list)` returns the `(np, np)` companion matrix and
`is_stable(A_list)` tests `max |eig| < 1.0` on it.

`irf`, `fevd` and `historical_decomp` in `puremacro.var.irf` work off
`(A_list, B0)` directly — `historical_decomp` wants the residuals as well — for
when you have an impact matrix from somewhere else. `gfevd` takes
`(A_list, Sigma)` instead and needs no impact matrix at all: it is the
Pesaran–Shin generalised decomposition — order-invariant, rows that do not
naturally sum to one, rescaled to sum to one under the default `normalize=True`
(the Diebold–Yılmaz convention).

## `(H+1, n, n)` — say it out loud once

**Every IRF, FEVD and band array in this package is `(horizon, response,
shock)`.** `irf_point[h, i, j]` is the response of variable `i` at horizon `h`
to a one-standard-deviation shock in structural equation `j`. Horizon `0` is
impact, so an array with `horizon=20` has 21 rows.

This is worth stating because the two axes that are not the horizon are both
length `n` and a transposition is silent: it produces plausible-looking impulse
responses with the roles of response and shock swapped. `ProxySVARResult`
shipped as `(n, n, H+1)` until 0.47.0 flipped it to match the other result
dataclasses. The inference layer still uses the other convention —
`inference.wild_bootstrap.wild_bootstrap_var` returns `(n, n, H+1)` and
`proxy_svar` transposes it back on the way out — so if you write your own
`impact_fn` and call that layer directly, you must transpose yourself.

The exceptions are the results that have no `n` shocks to report.
`GKRobustBandsResult`'s arrays are `(H+1, n)` — it describes a single named
shock, so the shock axis is gone rather than degenerate — `FAVARResult`'s panel
frames are `(H+1, N)` for the same reason, and the regime GIRFs in
`puremacro.var.regime` carry a regime axis instead (last section).

## The identification schemes

Everything in the table is importable from `puremacro.var.identify`. The
"needs" column names the argument that carries the assumption — what you decide
beyond `Y`, `p` and `horizon`. Several of them have a default
(`permanent_var_idx=0`, `target_idx=0`, `max_fev_at=1`, `shock_idx=0`);
defaulting is still choosing.

| call | assumption | needs |
|---|---|---|
| `cholesky` | recursive contemporaneous ordering | `ordering` (optional) |
| `bq` | one shock has all the long-run effect on one variable | `permanent_var_idx` |
| `sign_restrictions` | IRF signs at named horizons | `restrictions` |
| `sign_zero` | signs **and** exact zeros on the impact matrix | `zero_constraints`, `sign_constraints` |
| `narrative_sign_svar` | signs plus the sign or dominance of a shock on named dates | `sign_matrix`, `restrictions` |
| `proxy` | an observable correlated with one shock, orthogonal to the rest | `instrument_series` |
| `identify_maxshare` | the shock is the one that explains most of a variable's FEV | `target_idx`, `max_fev_at` |
| `hetero` | shock variances shift across two known regimes | `regime_indicator` |
| `magmav_svar` | shock variances shift at dates the estimator finds itself | — |
| `non_gaussian_svar` | at most one shock is Gaussian | — |
| `gk_robust_bands` | signs, but reported as an identified *set* | `restrictions`, `shock_idx` |

Two naming traps. `puremacro.var.identify.cholesky` is the **function**
`cholesky_svar`, not the submodule — the package `__init__` shadows it, as it
does for `bq`, `proxy`, `hetero`, `maxshare` and `sign_zero`. And **the
identified shock is column 0** of `B` in `proxy` (whatever `shock_target_idx`
you passed), in `identify_maxshare` and in `sign_restrictions`, and column 0 is
the most non-Gaussian shock in `non_gaussian_svar`. The two schemes that let you
name the shock instead are `narrative_sign_svar` (per restriction, and per
column of `sign_matrix`) and `gk_robust_bands` (`shock_idx`).

### Cholesky

```python
from puremacro.var.identify import cholesky, compute_chol_shocks

res = cholesky(Y, p=4, horizon=20, ordering=[2, 0, 1], n_boot=500, ci=0.9, seed=0)
t_index, eps = compute_chol_shocks(Y, p=4, ordering=[2, 0, 1])   # (T-p,), (T-p, n)
```

`ordering` permutes the variables before the factorisation and then **un-permutes
both axes on the way out**, so `irf_point[h, i, j]` always means "variable `i`
responding to the shock attached to variable `j`" in your original column order,
no matter what ordering you passed. Reorder the variables and the numbers change;
the axis labels do not. `compute_chol_shocks` follows the same rule and returns
the shocks themselves — `ε_t = L⁻¹ u_t` — with `t_index` giving the row offsets
into `Y` so you can map them back to dates for an LP-IV or a narrative check.

### Blanchard–Quah

`bq(Y, p=..., horizon=..., permanent_var_idx=0)` imposes that only the shock in
column `permanent_var_idx` has a non-zero long-run effect on that variable.
**The IRFs come back cumulated** along the horizon axis, so `irf_point[h]` is a
*level* response and not a difference — the whole point, since the VAR is
normally run on growth rates. Do not `cumsum` them again. If `(I − ΣA_i)` is
singular the call raises a `LinAlgError` naming the near-unit-root, rather than
returning an inverse of a singular matrix.

### Sign restrictions, and the four different formats

The sign-restriction family does not share one argument format, because the
schemes restrict different objects. This is the table to check before typing:

| call | argument | format | restricts |
|---|---|---|---|
| `sign_restrictions` | `restrictions` | `{h: [s₀ … s_{n−1}]}` | responses to **shock column 0** at horizon `h` |
| `gk_robust_bands` | `restrictions` | `{h: [s₀ … s_{n−1}]}` | responses to shock `shock_idx` |
| `narrative_sign_svar` | `sign_matrix` | `{h: S}`, `S` shape `(n, n)` or `(n,)`; a bare array means `{0: S}` | `S[i, j]` restricts variable `i` to shock `j`; an `(n,)` vector restricts shock 0 |
| `sign_zero` | `sign_constraints`, `zero_constraints` | `{(i, j): ±1}`, `[(i, j)]` | entries of `B0` — impact only, horizon 0 |

`0` always means unrestricted. `sign_restrictions` draws `n_draws=2000`
Haar-uniform rotations by default and raises `RuntimeError` if none is
admissible; `n_accepted / n_draws` is in the `summary()` and is the number to
look at before believing the bands, since a 1% acceptance rate means the median
is a median over twenty draws.

`sign_zero` is the odd one out in the whole subpackage: it takes `A_list` and
`Sigma` rather than `Y`, it returns only `B0` and `Q` (no IRFs), and a failed
search does not raise — inspect `success`, which is `False` with `B0=None` when
no draw satisfied everything within `n_draws=1000`.

### Narrative sign restrictions

Antolín-Díaz & Rubio-Ramírez (2018), with Ludvigson–Ma–Ng magnitude bounds
attached. Three kinds of `NarrativeRestriction`:

| `kind` | says | extra fields |
|---|---|---|
| `"shock_sign"` | shock `j` had this sign on this date (AD-RR Type I) | `sign` |
| `"hd_dominance"` | shock `j` was the most (`"most"`, Type II) or overwhelmingly the most (`"overwhelming"`, Type III) important driver of variable `i` over a window | `variable`, `window`, `dominance` |
| `"shock_bound"` | the absolute magnitude of shock `j` on this date lies in a stated range (LMN 2021) | `min_magnitude`, `max_magnitude` |

```python
from puremacro.var.identify import narrative_sign_svar, NarrativeRestriction

res = narrative_sign_svar(
    Y, p=4, horizon=20,
    sign_matrix={0: [+1, +1, 0]},
    restrictions=[NarrativeRestriction(kind="shock_sign", date=100, shock=0, sign=+1)],
    n_draws=2000, ci=0.9, seed=0,
)
res.ess, res.n_narrative_accepted, res.restriction_fail_counts
```

`date` is a row index into `Y` when `dates=None`, and a timestamp located in
`dates` otherwise. A bare `(date, shock, sign)` tuple is accepted as Type I
shorthand, and a `puremacro.narrative.NarrativeEvent` is adapted to one using its
announcement date and sign, targeting shock 0.

Two diagnostics carry the weight of the method. `restriction_fail_counts` gives,
per restriction, how many of the traditionally-accepted draws it killed — a
count of zero means the restriction did nothing and a count equal to
`n_traditional_accepted` means it killed everything. And `ess` is the Kish
effective sample size of the AD-RR importance weights: when `ess` is far below
`n_narrative_accepted`, a handful of draws are carrying the bands and the
percentiles are not describing what they appear to describe. Reduced-form
parameters are fixed at OLS unless `bayes_draws=True`, which samples them from
the conjugate Normal-Inverse-Wishart posterior instead.

### Proxy / external instrument

```python
from puremacro.var.identify import proxy

res = proxy(Y, p=4, horizon=20, instrument_series=z, n_boot=500, ci=0.9, seed=0)
res.first_stage_F        # Olea-Pflueger effective F
res.B[:, 0]              # the identified impact vector
```

The instrument is aligned to the residuals by its **last** `T − p` observations
(`z[-T_eff:]`) and demeaned. Under the Mertens–Ravn / Stock–Watson assumptions
`Π = Cov(u, z)/Var(z)` **is** the impact column up to scale, and the
normalisation that pins the scale is `b₁′ Σ⁻¹ b₁ = 1`, which follows from
`Σ = BB′`. So the impact vector is

```
b₁ = Π / sqrt(Π′ Σ⁻¹ Π)
```

computed through a Cholesky solve rather than an explicit inverse. This is worth
spelling out because the obvious-looking alternative, `(Σ Π)/sqrt(Π′ Σ Π)`, is
what this package shipped from 0.92.0 to 1.8.0 and is wrong: it is proportional
to `Σ b₁` rather than to `b₁`, correct only when `Σ` is a multiple of the
identity. It still satisfies `b′ Σ⁻¹ b = 1` exactly and the SVD completion still
returned `BB′ = Σ` to machine precision, so no internal consistency check could
see it. On a DGP with true `b₁ = [1, 0.8, −0.5]` it converged, at T = 400,000, to
`[0.919, 1.049, −0.608]`. **If you published a proxy-SVAR IRF from a release
before this fix, re-run it.**

`shock_target_idx` does *not* change the identification — it only selects which
residual column the first-stage F is computed on. The identified shock is
column 0 of `B` regardless.

`first_stage_F` is the Olea–Pflueger effective F, not a Wald F. `summary()`
labels it `WEAK` at or below 23.0 — the published 5%-worst-case-bias cutoff for
a single instrument is 23.1, rounded down in the flag — and below it the
bootstrap bands understate the uncertainty and weak-IV-robust inference is the
honest answer.

### Max-share

```python
from puremacro.var.identify import identify_maxshare, maxshare, news_maxshare

res = identify_maxshare(Y, p=4, target_idx=0, max_fev_at=40,
                        horizon=40, n_bootstrap=500, ci=0.68, seed=0)
```

The optimisation is closed form: the impact vector is `chol(Σ) q` where `q` is
the top eigenvector of a symmetric PSD matrix, so there is no search and no
starting value. `max_fev_at` accumulates horizons `0 … max_fev_at − 1`, so
`max_fev_at=1` is impact and a large value is the Barsky–Sims news shock.
`news_maxshare(A_list, Sigma, target_var=..., horizon=40)` is the same engine
with that horizon baked in.

Note the two argument-name breaks from the rest of the subpackage:
`identify_maxshare` uses `n_bootstrap` (not `n_boot`) and defaults to `ci=0.68`
(not `0.9`).

**`MaxShareResult.B` is not a valid structural impact matrix beyond column 0.**
`_complete_B` orthonormalises the remaining columns in the Euclidean metric, so
`B B′ ≠ Σ`. Column 0 is exactly right — it is `chol(Σ) q` — but `fevd` and
`fev_share_at_target` are normalised by `B B′` rather than by `Σ` and are
therefore not the forecast-error variance decomposition of the estimated system.
At `max_fev_at=1` the max-share shock is by construction the whole impact
forecast error of the target variable, so the share is analytically exactly 1.0
— and `fev_share_at_target` reports visibly less. The low-level helper does not
have the problem: it returns `chol(Σ) Q` with a genuinely orthogonal `Q`, so
`B B′ = Σ` to machine precision and the impact share comes back at 1.0. For
shares, go through it:

```python
from puremacro.var import fit_var, fevd
from puremacro.var.identify import maxshare

r = fit_var(Y, p=4)
B = maxshare(r.A_list, r.Sigma, target_var=0, horizon=7)   # == max_fev_at=8
share = fevd(r.A_list, B, 24)[7, 0, 0]
```

`maxshare(..., horizon=h)` and `identify_maxshare(..., max_fev_at=h+1)` return
the same impact column up to its sign — one calls `numpy.linalg.eigh` and the
other `scipy.linalg.eigh`, and the sign of an eigenvector is arbitrary — but the
off-by-one is real.

### Heteroskedasticity: Rigobon and Magnusson–Mavroeidis

`hetero` (Rigobon 2003) takes a binary `regime_indicator` of length `T` or
`T − p` — it slices off the first `p` itself when you give it `T` — and
eigendecomposes `L₀⁻¹ Σ₁ L₀⁻ᵀ`. `variance_ratios` are those eigenvalues, the
across-regime variance ratio of each shock; a ratio near 1 is a shock the regime
break does not separate, and the column is then only as identified as the
neighbouring eigenvalues are distinct. Each regime needs at least `n + 1`
observations or the call raises. Bands come from a moving-block bootstrap with
`block_len` defaulting to `round((T − p)^(1/3))`; `n_boot=0` returns
`lower=upper=None`.

`magmav_svar` (Magnusson–Mavroeidis 2014) finds the break dates itself by
sup-Wald plus BIC over `{0, 1, 2, 3, 4}` breaks. When BIC picks zero breaks
there is no heteroskedasticity to identify from, and the function **warns and
falls back to a plain Cholesky** rather than failing — so check `k_breaks` and
`eu` before reading the IRFs. `eu == (1, 1)` means the impact-matrix optimiser
converged; `(0, 0)` means either zero breaks or non-convergence. It is a
convergence proxy, not the paper's existence-and-uniqueness test, and the
dataclass docstring says so.

### Non-Gaussian

`non_gaussian_svar(Y, p=..., horizon=..., seed=0)` runs FastICA on the whitened
residuals (Lanne–Meitz–Saikkonen 2017) and orders the columns by descending
absolute excess kurtosis, so the most non-Gaussian shock is column 0. It ships
its own falsification: `lr_test` is a likelihood-ratio test against the Gaussian
baseline (keys `stat`, `df`, `p_value`) and `consistency_check` reports
`max_abs_diff` of `B0 B0′` against `Σ_u`. A `p_value` that does not reject
Gaussianity means the identifying assumption is not in the data and the column
ordering is noise. It takes no bootstrap arguments and returns no bands.

### Giacomini–Kitagawa robust bands

Standard sign-restriction output is a posterior under a uniform prior on `Q`,
not the identified set. `gk_robust_bands` separates the two: for each of
`n_var_draws` bootstrap VAR estimates it maps the set with `n_q_per_draw` Haar
draws and takes the min, max and within-set median, then aggregates across VAR
draws. `irf_lower` and `irf_upper` are the *set* endpoints, not a confidence
interval, and are `(H+1, n)` — one named shock. `n_accepted_per_draw` is the
per-VAR-draw acceptance count; VAR draws where nothing was admissible become
NaN rows and drop out of the `nanpercentile`.

`n_var_draws=1` skips VAR uncertainty and gives the point identified set.
`gk_robust_bands_from_gibbs` takes posterior draws instead of the bootstrap, and
the shapes line up with `minnesota_gibbs` exactly:

```python
from puremacro.var import minnesota_gibbs
from puremacro.var.identify import gk_robust_bands_from_gibbs

post = minnesota_gibbs(df, 4, n_draws=1000, burn=500)     # A_draws (D, p, n, n)
res = gk_robust_bands_from_gibbs(post["A_draws"], post["Sigma_draws"],
                                 horizon=20, restrictions={0: [1, 1, 0]},
                                 shock_idx=0, n_q_per_draw=300, ci=0.9)
```

## The result dataclasses

All frozen; all but `HeteroResult` carry a `summary()`. The central estimate is
not called the same thing everywhere, because it is not the same thing
everywhere — a median over rotation draws is not a point estimate:

| function | class | central | bands | notes |
|---|---|---|---|---|
| `cholesky` | `CholeskySVARResult` | `irf_point` | `irf_lower`, `irf_upper` | `n_boot`, `n_fail`, `ci` |
| `bq` | `BQSVARResult` | `irf_point` | `irf_lower`, `irf_upper` | **cumulated** |
| `sign_restrictions` | `SignRestrictionResult` | `irf_median` | `irf_lower`, `irf_upper` | `n_draws`, `n_accepted` |
| `narrative_sign_svar` | `NarrativeSignSVARResult` | `irf_median` (weighted) | `irf_lower`, `irf_upper` | `weights`, `ess`, `restriction_fail_counts` |
| `proxy` | `ProxySVARResult` | `irf_point` | `irf_lower`, `irf_upper` | `B`, `first_stage_F` |
| `hetero` | `HeteroResult` | `irfs` (= `point`) | `lower`, `upper` (or `None`) | `B`, `variance_ratios`, `fevd` |
| `identify_maxshare` | `MaxShareResult` | `irfs` | `irf_lower`, `irf_upper` (or `None`) | `B`, `q`, `fevd`, `fev_share_at_target` |
| `magmav_svar` | `MagMavSVARResult` | `irf_point` | `irf_lower`, `irf_upper` | `variance_change_dates`, `k_breaks`, `eu` |
| `non_gaussian_svar` | `NonGaussianSVARResult` | `irf` | — | `B0`, `Q`, `kurtosis`, `lr_test` |
| `sign_zero` | `SignZeroResult` | — | — | `success`, `B0`, `Q`, `n_draws_used` |
| `gk_robust_bands` | `GKRobustBandsResult` | `irf_median` | `irf_lower`, `irf_upper` | all `(H+1, n)` |
| `mean_group_svar` | `PanelSVARResult` | `irf_mean` | `irf_lower`, `irf_upper` | `country_irfs` is `(N, H+1, n, n)` |
| `favar` | `FAVARResult` | `irf_panel` | `irf_lower_panel`, `irf_upper_panel` | DataFrames, `(H+1, N)` |

Everything not marked otherwise is `(H+1, n, n)`. `HeteroResult` lives in
`identify/hetero.py`, not `identify/_results.py`, and carries `irfs` and `point`
as two names for the same array.

## Bootstrap bands, and what happens to a draw that fails

A bootstrap draw of a VAR can produce a `Σ_b` that is not positive definite, or
a long-run matrix that is singular, or an optimiser that will not converge.
There is no correct answer to what to do with such a draw, and the schemes here
do not all do the same thing. Three conventions are in use, and only one of them
tells you it happened.

| scheme | resampling | a failed draw |
|---|---|---|
| `cholesky`, `bq`, `identify_maxshare` | recursive residual | **dropped**; warns above 5%; raises if all fail |
| `magmav_svar` | regime-preserving residual | dropped; warns above 5%; NaN bands if all fail |
| `proxy` | Rademacher wild (residuals *and* proxy re-signed together) | dropped; warning above 5% failures; `n_fail` reported |
| `hetero` | moving block | replaced by the point IRF, silently |
| `favar` | recursive residual | replaced by the point IRF, silently |
| `gk_robust_bands` | recursive residual | NaN row, excluded by `nanpercentile` |
| `bootstrap_bands` | your choice | NaN draw, excluded by `nanpercentile` |

The drop-and-warn path is the one to trust, and it is stricter than it looks. A
bootstrap `Σ_b` is rejected when the smallest Cholesky pivot falls below `1e-4`
times the largest — roughly `cond(Σ_b) > 1e8` — which is far stricter than
`puremacro._linalg.safe_cholesky`'s general threshold of `1e-7` (`cond > 1e14`).
The reason is stated in `cholesky.py` and is worth repeating: an estimator that
returns wide bands is honest, one that returns narrow bands built on
ill-conditioned draws is silent garbage. LAPACK's `potrf` does not raise on an
ultra-degenerate `Σ_b`; it returns NaN diagonals, so those are caught explicitly
too. `n_fail` is on the result, and `summary()` prints it as a rate.

Where the failed draw is replaced by the point estimate instead, `n_fail` does
not exist and cannot: the bands look fine and are too narrow by however many
draws failed.

### `bootstrap_bands` — bands for an impact matrix of your own

```python
from puremacro.var import bootstrap_bands
from puremacro.var.identify.cholesky import cholesky_factor

out = bootstrap_bands(Y, 4, lambda A_list, Sigma: cholesky_factor(Sigma), 20,
                      n_boot=500, alpha=0.10, method="recursive",
                      band="pointwise")
out["point"], out["lower"], out["upper"], out["draws"]
```

`identify_fn(A_list, Sigma, **id_kwargs) -> B0`. `method` is `"recursive"`,
`"wild"` (Rademacher) or `"block"` (block length `round(T^(1/3))`).
`bias_correct=True` runs the Kilian (1998) pilot stage — `n_pilot=100` draws to
estimate the finite-sample bias of the coefficients, subtracted and then shrunk
in steps of 0.01 until the corrected companion has `max |eig| < 0.999`.

Two things about this function differ from the rest of the page and will bite:

- **It takes `alpha`, not `ci`.** `alpha=0.10` is a 90% band. Everything else in
  `puremacro.var` takes `ci=0.9` for the same thing.
- **`out["point"]` is the bootstrap median, not the OLS estimate.** It is
  `nanpercentile(draws, 50)`. If you want the point estimate, compute it
  yourself from `fit_var` and your `identify_fn`.

`band="sup-t"` returns simultaneous bands instead (Montiel Olea &
Plagborg-Møller 2019), computed per `(response, shock)` pair across horizons,
and adds `band` and `crit_value` (an `(n, n)` array of per-pair critical values,
NaN where every horizon was degenerate) to the returned dict. Horizons whose
draws are exactly degenerate — the structural zeros a Cholesky impact matrix has
in every draw — collapse to the point rather than studentising `0/0`. It raises
if fewer than 20 draws came back fully finite, since a sup-t critical value
estimated from a handful of draws is not a critical value. The band is centred
on the bootstrap median, a documented simplification of the paper's Algorithm 2,
which centres on the original-sample estimate.

## FAVAR

Bernanke, Boivin & Eliasz (2005), for when the three variables that fit in a VAR
are not the three variables you care about. Principal components summarise a wide
informational panel, the VAR is run on `[policy, factors]`, and the responses are
projected back onto every series in the panel through the loadings.

```python
from puremacro.var import favar

res = favar(panel_df, policy_series, n_factors=3, p=2, horizon=20,
            n_boot=200, ci=0.90, seed=42)
res.irf_panel["unemployment"]        # (H+1,) response in the series' own units
res.explained_variance_ratio         # (K,) share of panel variance per factor
print(res.summary())
```

`panel_data` is `(T, N)` — a DataFrame keeps its column names, an ndarray gets
`Var_1 … Var_N`. The panel is standardised, factors are extracted by SVD and
scaled by `sqrt(T)`, then orthogonalised against the policy series; the state
vector is `Z = [Y, F̃]` of dimension `1 + K` and the policy shock is shock 0 of a
Cholesky on `Z`, which is what makes ordering the policy variable first the
identifying assumption. `irf_panel` is returned in the **original units** of each
series — the normalised responses are multiplied back by each column's standard
deviation — so the columns are directly readable and directly plottable.

Three limits worth knowing before quoting a FAVAR number:

- **Demean the policy series before passing it.** Both the factor
  orthogonalisation and the loading regression run through the origin with no
  intercept, while the panel has been standardised to mean zero, so the mean you
  leave in changes both. The VAR on `Z` does carry a constant, so the policy
  variable's *own* response is unaffected to machine precision — which is
  exactly why this is easy to miss. On a synthetic 20-series panel, adding a
  constant 4.0 to a unit-standard-deviation policy series moved panel impulse
  responses by as much as the largest response itself and left `irf_policy`
  unchanged.
- **Failed bootstrap draws are replaced by the point IRF**, so `irf_lower_panel`
  and `irf_upper_panel` are too narrow by however many draws failed, with
  nothing recording how many.
- The factors are principal components of the **whole** panel, not of the
  slow-moving block. BBE's identification uses the slow/fast split; this is the
  simpler variant, and the orthogonalisation against `Y` is doing all the work of
  separating the policy instrument from the factors.

## Panel VAR

Canova–Ciccarelli mean-group: one SVAR per country, IRFs averaged across
countries, bands from the **cross-country** distribution rather than from any
bootstrap. The bands therefore describe heterogeneity across countries, not
sampling uncertainty within one — which is the right object for "does this shock
do the same thing everywhere" and the wrong one for "is this response
significant".

```python
from puremacro.var.identify import mean_group_svar

res = mean_group_svar({"USA": Y_us, "DEU": Y_de, "ESP": Y_es},
                      p=4, horizon=20, identification="cholesky", ci=0.9)
res.irf_mean                 # (H+1, n, n)
res.country_irfs             # (N, H+1, n, n)
res.country_ids              # ('DEU', 'ESP', 'USA') — SORTED, not insertion order
```

**`country_ids` is `tuple(sorted(panel_data))`.** `country_irfs[i]` is the
country at `country_ids[i]`, which is not the order you passed the dict in.
Always index through `country_ids`.

The canonical version supports `identification="cholesky"` and `"bq"` only, and
raises `ValueError` naming the alternatives for anything else — schemes that
need per-country bootstrap kwargs are better run per country against their own
module. All countries must share `n`; `T_i` may differ.

There is a second, older `mean_group_svar` in `puremacro.var.panel`. It accepts
`"proxy"` and `"rigobon"` as well, substitutes a zero IRF with a warning when a
country fails instead of raising, returns `country_ids` as a **list in insertion
order**, and names its band fields `irf_lo` / `irf_hi` rather than
`irf_lower` / `irf_upper`. Two things its docstrings claim and its code does not:
the dataclass says `(n, n, H+1)` where the code returns `(H+1, n, n)` like
everything else, and the module header advertises a `"maxshare"` scheme the
dispatch table has no entry for (it raises `KeyError`). Prefer the `identify`
version unless you need `proxy` or `rigobon` across a panel.

`peak_summary` and `peak_distribution` turn a dict of per-country
`{"point", "lo", "hi", "n_obs"}` into one row per country — peak horizon and
value, `irf_4` / `irf_8` / `irf_16`, and the accumulated response at h=16 — for
forest plots across countries. `peak_lo` and `peak_hi` are the band bounds *at*
the peak horizon and are explicitly **not** a confidence interval on the argmax,
which has a non-standard sampling distribution.

## Not on this page

`puremacro.var` also ships `bvar` (Minnesota prior as dummy observations, NIW
Gibbs), `vecm`, `tvp` (time-varying parameters), `regime` (Markov-switching,
threshold, TVECM, generalised IRFs) and `diagnostics` (`granger_causality`,
`block_exogeneity`). Check their shapes rather than assuming: `regime.girf`
shocks one equation at a time, so `girf_by_regime` is `(K, S, H+1, n)` — `K`
starting regimes by `S` shock sizes — and `girf_pooled` is `(S, H+1, n)`, not
`(H+1, n, n)`.
