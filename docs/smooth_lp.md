> 🇬🇧 English · 🇪🇸 [Español](es/smooth_lp.md)

# Smooth Local Projections

`puremacro.lp.smooth_lp` implements the penalized B-spline Local Projection methodology developed by **Barnichon and Brownlees (2019, *The Review of Economics and Statistics*)**.

Standard Local Projections (**Jordà 2005**) estimate impulse response functions (IRFs) via separate Ordinary Least Squares (OLS) regressions for each horizon $h = 0, \dots, H$. While Jordà's estimator is robust to dynamic misspecification, estimating independent horizon-by-horizon regressions ignores the intrinsic smoothness of macroeconomic propagation mechanisms, frequently producing noisy, jagged IRF trajectories and excessively wide confidence bands at longer horizons.

Smooth Local Projections overcome these inefficiencies by **jointly estimating impulse responses across all horizons** using a continuous B-spline basis regularized by a roughness difference penalty, achieving substantial variance reduction while remaining asymptotically unbiased.

---

## 1. Econometric Methodology

### 1.1 Model Formulation

Let $y_t$ denote the dependent response variable, $x_t$ the structural shock or policy intervention, and $z_t$ a vector of control variables (a constant, `n_lags` lags of $y_t$, $x_t$ and the controls, and the contemporaneous controls). In standard local projections:

$$y_{t+h} = \alpha_h + \beta_h x_t + \gamma_h' z_t + \varepsilon_{t+h}, \quad h = 0, \dots, H$$

Each horizon $h$ is estimated on its own sample $S_h = \{t : t + h \le T\}$ (the last $h$ observations are lost at horizon $h$), exactly as `lp_hac` does. Using the Frisch-Waugh-Lovell (FWL) theorem, the controls $z_t$ are partialled out horizon by horizon to obtain the residualized lead $\tilde{y}_{h,t}$ and the residualized shock $\tilde{x}_{h,t}$.

Barnichon and Brownlees approximate the impulse response function $\beta(h)$ as a linear combination of $K$ B-spline basis functions evaluated on the horizon grid $h \in \{0, 1, \dots, H\}$:

$$\beta(h) = \sum_{k=1}^K B_k(h) \theta_k = B_h \theta$$

where $B$ is the $(H+1) \times K$ clamped cubic B-spline basis matrix and $\theta \in \mathbb{R}^K$ is the spline parameter vector.

The stacked system across all horizons is estimated via **Penalized Least Squares (PLS)**:

$$\min_\theta \; \sum_{h=0}^{H} \sum_{t \in S_h} \big( \tilde{y}_{h,t} - \tilde{x}_{h,t} B_h \theta \big)^2 + \lambda \, \theta' P \theta
\;=\; \min_\theta \; \| \tilde{Y} - X \theta \|^2 + \lambda \, \theta' P \theta$$

where:
- $X = B \otimes \tilde{x}$ is the stacked (Kronecker) design matrix over the horizon-specific samples.
- $P = D_d' D_d$ is the roughness penalty matrix formed by the $d$-th difference operator matrix $D_d$ (default second-order difference $d=2$).
- $\lambda \ge 0$ is the regularization parameter governing the bias-variance trade-off. **`lam` and `res.optimal_lambda` are the $\lambda$ of exactly this objective**, i.e. the penalty is $\lambda\,\theta'P\theta$ against the stacked sum of squared residuals (which is $O(T)$, so $\lambda$ is not scale-free: the automatic grid below adapts to the sample).
  - As $\lambda \to 0$, the estimates converge to the unpenalized projection of the horizon-by-horizon OLS local projections onto the spline basis; with a saturated basis (`n_knots = H - degree`) they coincide *exactly* with `lp_hac` on each horizon's sample.
  - As $\lambda \to \infty$, the impulse response is regularized into a polynomial of degree $d-1$ (a straight line for $d=2$).

With `gls=True` the sum of squares is replaced by the **Penalized Generalized Least Squares (PGLS)** quadratic form $(\tilde{Y} - X\theta)'\,\Omega^{-1}\,(\tilde{Y} - X\theta)$, where $\Omega$ is the $(H+1)\times(H+1)$ cross-horizon covariance of the horizon-by-horizon OLS residuals. Estimating $\Omega$ needs a balanced residual panel, so with `gls=True` every horizon uses the common sample $t + H \le T$ (`res.sample == "balanced"`); with the default `gls=False` each horizon uses its own sample (`res.sample == "per-horizon"`, `res.n_obs` gives $|S_h|$).

**Basis.** `n_knots` is the number of *interior* knots, equally spaced between the first and the last horizon, so the basis has $K = $ `n_knots + degree + 1` functions. By default about one knot per three horizons is used. $K$ is capped at the number of horizons (so that $\lambda \to 0$ recovers the unpenalized local projection); an explicit `n_knots` above the cap is reduced with a `UserWarning`, and the effective value is stored in `res.n_knots`.

### 1.2 Data-Driven $\lambda$ Selection

With `lam="auto"` (the default; case-insensitive, `None` is equivalent) `smooth_lp` searches the grid `res.lambda_grid = logspace(-5, 5, 50) * mean_h(x̃_h' x̃_h)`, which spans from essentially unpenalized to essentially polynomial impulse responses for any sample size, and minimizes:

1. **Akaike Information Criterion (`selection='aic'`)**: $\log(\text{RSS}/N) + 2\,\text{df}(\lambda)/N$ with effective degrees of freedom $\text{df}(\lambda) = \text{tr}\big((X' \Omega^{-1} X + \lambda P)^{-1} X' \Omega^{-1} X\big)$ and $N = \sum_h |S_h|$.
2. **Bayesian Information Criterion (`selection='bic'`)**: $\log(\text{RSS}/N) + \log(N)\,\text{df}(\lambda)/N$.
3. **Generalized Cross-Validation (`selection='gcv'`)**: $(\text{RSS}/N) / (1 - \text{df}(\lambda)/N)^2$.
4. **K-Fold Cross-Validation (`selection='cv'`)**: out-of-sample squared error over $K = \min(5, T/10)$ consecutive time blocks; the fold estimator is the same penalized (G)LS estimator as the final fit.

The selected value is reported by `res.optimal_lambda`, `res.df_lambda`, in the `lambda` column, and in `res.summary()`. A user-supplied number fixes $\lambda$ (`res.selection_criterion == "fixed"`); `selection` is validated in either case.

### 1.3 Statistical Inference

Two robust inference schemes are provided:

1. **Analytical Sandwich HAC (`ci_type='analytic'`)**:  
   Newey-West / Bartlett kernel standard errors for the spline coefficients $\theta$ (bandwidth `hac_lags`, default $H$), projected onto the horizon basis:
   $$\widehat{\text{Var}}(\hat{\beta}) = B \left( X' X + \lambda P \right)^{-1} X' \hat{\Sigma}_{HAC} X \left( X' X + \lambda P \right)^{-1} B'$$
2. **Moving Block Bootstrap (`ci_type='bootstrap'`)**:  
   Resamples overlapping temporal blocks of length $\lceil T^{1/3} \rceil$ to preserve serial dependence and heteroskedasticity non-parametrically across horizons (`n_boot` replications, reproducible with `seed`).

---

## 2. Empirical Variance Reduction

On calibrated macroeconomic DGPs (e.g., standard monetary VARs or AR(2) series):
- Unpenalized Local Projections suffer from high sample variance, with empirical standard deviations that grow rapidly with horizon $h$.
- Smooth Local Projections deliver a markedly lower Monte Carlo variance and mean squared error at intermediate and distant horizons ($h \ge 4$); Barnichon and Brownlees report RMSE gains of the order of 30-50% in their simulations, and the package test-suite checks a variance reduction of at least 20% on an AR(2) DGP. The gain is DGP-dependent.

---

## 3. Basic & Advanced Usage

### Quickstart Example

```python
import numpy as np
import pandas as pd
from puremacro.lp import smooth_lp

# 1. Generate synthetic macroeconomic time series
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.7 * y[t-1] + 0.5 * shock[t] + 0.2 * rng.standard_normal()

df = pd.DataFrame({"y": y, "shock": shock})

# 2. Fit Smooth Local Projection with Automated AIC Tuning
res_smooth = smooth_lp(
    df=df,
    y="y",
    x="shock",
    horizons=20,
    n_lags=4,
    lam="auto",
    selection="aic",
    degree=3,
    penalty_order=2,
    ci_type="analytic",
    alpha=0.05,
)

# 3. Print Results & Summary (the summary reports the selected lambda)
print(res_smooth.summary())
print(f"Optimal Lambda: {res_smooth.optimal_lambda:.4g}  (effective df = {res_smooth.df_lambda:.2f})")

# 4. Visualize Smooth IRF with Confidence Bands
fig = res_smooth.plot()

# 5. Access Numerical Results
point_estimates = res_smooth.point       # (H+1,) array of IRF values
standard_errors = res_smooth.se          # (H+1,) array of HAC SEs
ci_lower = res_smooth.ci_lower           # 95% lower bounds
ci_upper = res_smooth.ci_upper           # 95% upper bounds
```

### Comparing Unpenalized vs Smooth LP

```python
# lambda = 0 with a saturated basis (H - degree interior knots) is exactly
# the horizon-by-horizon OLS local projection (same estimates as lp_hac)
res_ols = smooth_lp(df=df, y="y", x="shock", horizons=20, n_knots=17, lam=0.0)

# Data-driven smooth LP
res_opt = smooth_lp(df=df, y="y", x="shock", horizons=20, lam="auto", selection="bic")

# Comparison: optimal smoothing yields tighter, non-oscillating bands
print("OLS average SE   :", np.mean(res_ols.se))
print("Smooth average SE:", np.mean(res_opt.se))
print("BIC lambda       :", res_opt.optimal_lambda, "on the grid", res_opt.lambda_grid[[0, -1]])
```

### Calling conventions: arrays, mixed inputs and array controls

`smooth_lp` accepts the same two conventions as `lp_hac`:

```python
y_arr, x_arr = df["y"].to_numpy(), df["shock"].to_numpy()
controls = rng.standard_normal((T, 2))

# Arrays: response first, shock second (exactly like lp_hac(y, shock, ...))
res_arr = smooth_lp(y_arr, x_arr, horizons=12, n_lags=2, controls=controls)

# DataFrame with column names; y/x may also be arrays of length len(df),
# and controls may be column names, a (T,) / (T, k) array, or a DataFrame
res_mix = smooth_lp(df, "y", x_arr, horizons=12, n_lags=2, controls=controls)

assert np.allclose(res_arr.point, res_mix.point)
print(res_arr.y_name, res_arr.x_name)        # 'y' 'x'  (Series names are used when available)
print(res_arr.n_obs)                          # per-horizon sample sizes T - n_lags - h
```

---

## 4. Full API Specification

### `smooth_lp`

```text
smooth_lp(
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None = None,
    x: str | np.ndarray | None = None,
    horizons: int | Iterable[int] = 20,
    n_lags: int = 4,
    controls: Sequence[str] | np.ndarray | pd.DataFrame | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    penalty_order: int = 2,
    lam: float | str | None = "auto",
    selection: str = "aic",
    alpha: float = 0.05,
    ci_type: str = "analytic",
    *,
    lambda_: float | None = None,     # alias for lam
    lags: int | None = None,          # alias for n_lags
    horizon: int | None = None,       # alias for horizons (max horizon)
    ci: float | None = None,          # confidence level, sets alpha = 1 - ci
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    hac_lags: int | None = None,
    gls: bool = False,
) -> SmoothLPResult
```

`lp_smooth` is an alias of `smooth_lp`.

#### Parameters:
- `df`: Input `pandas.DataFrame` containing target, shock, and controls; or, with array input, the 1-D response series.
- `y`: Column name (or array of length `len(df)`) of the response; with array input, the 1-D shock series (`smooth_lp(y_arr, x_arr)`).
- `x`: Column name (or array of length `len(df)`) of the shock variable; with array input it may be given as `x=` instead of the second positional argument (passing both raises).
- `horizons`: Maximum horizon integer $H$ (at least 1) or iterable of at least two horizons.
- `n_lags`: Number of autoregressive lags included for target, shock, and controls.
- `controls`: Optional exogenous controls: column names, a `(T,)` / `(T, k)` array, or a DataFrame.
- `n_knots`: Number of *interior* spline knots (basis size `n_knots + degree + 1`, adaptive default if `None`, capped at the number of horizons with a warning).
- `degree`: Polynomial degree of the B-splines (default `3` for cubic splines).
- `penalty_order`: Difference order of the roughness penalty matrix $P = D_d' D_d$ (default `2`).
- `lam`: Regularization parameter $\lambda$ of the stacked objective. `'auto'` (case-insensitive) or `None` for data-driven selection, or a non-negative float; negative, NaN or other strings raise `ValueError`.
- `selection`: Criterion for $\lambda$ optimization: `'aic'`, `'bic'`, `'gcv'`, or `'cv'` (validated even when `lam` is fixed).
- `alpha`: Significance level for confidence intervals (default `0.05` for 95% coverage).
- `ci_type`: Inference method: `'analytic'` (sandwich HAC) or `'bootstrap'` (moving block bootstrap); anything else raises `ValueError`.
- `n_boot`: Number of bootstrap replications when `ci_type='bootstrap'`.
- `seed`: Random number generator seed for bootstrap reproducibility.
- `hac_lags`: Bartlett bandwidth of the HAC estimator (default: the maximum horizon).
- `gls`: If `True`, performs feasible Generalized Least Squares weighting across horizons on the common (balanced) sample.

---

## 5. Result Interface

`smooth_lp` returns a `SmoothLPResult`, a subclass of `LPResult` (itself a `pandas.DataFrame`) with the same layout as every other `puremacro.lp` estimator:

- **Data columns** (indexed by `h`): `h`, `beta` (point estimates $\hat\beta(h)$), `se`, `lo`, `hi` (pointwise $1-\alpha$ bands), `lambda` (the $\lambda$ used, repeated on every row), `t` ($t$-statistic for $H_0: \beta(h)=0$).
- **Array properties**: `.point`, `.se`, `.ci_lower`, `.ci_upper`, `.t_stat`, `.horizons` (aliases of the columns above). There is no `p_value` column; use `2 * (1 - scipy.stats.norm.cdf(abs(res.t_stat)))` if needed.
- **Estimation attributes** (they survive `.copy()`, slicing and column selection): `optimal_lambda`, `df_lambda`, `lambda_grid`, `selection_criterion` (`'aic'`, `'bic'`, `'gcv'`, `'cv'` or `'fixed'`), `ci_type`, `theta`, `vcov` (of $\hat\beta$), `vcov_theta`, `B`, `P`, `n_knots` (effective interior knots), `n_basis`, `degree`, `penalty_order`, `gls`, `n_obs` (per-horizon sample sizes), `sample` (`'per-horizon'` or `'balanced'`), `y_name`, `x_name`, `method == 'LP-smooth'`, and `.metadata` (a dict of the above).
- **Methods**:
  - `.plot()`: Matplotlib figure of the estimated impulse response function with shaded confidence bands.
  - `.summary()`: the LP table followed by the smoothing diagnostics (selected $\lambda$ and how it was chosen, effective degrees of freedom, basis, inference method, samples).
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Formatted tables ready for publication.
