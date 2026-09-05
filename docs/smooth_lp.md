> 🇬🇧 English · 🇪🇸 [Español](es/smooth_lp.md)

# Smooth Local Projections

`puremacro.lp.smooth_lp` implements the penalized B-spline Local Projection methodology developed by **Barnichon and Brownlees (2019, *The Review of Economics and Statistics*)**.

Standard Local Projections (**Jordà 2005**) estimate impulse response functions (IRFs) via separate Ordinary Least Squares (OLS) regressions for each horizon $h = 0, \dots, H$. While Jordà's estimator is robust to dynamic misspecification, estimating independent horizon-by-horizon regressions ignores the intrinsic smoothness of macroeconomic propagation mechanisms, frequently producing noisy, jagged IRF trajectories and excessively wide confidence bands at longer horizons.

Smooth Local Projections overcome these inefficiencies by **jointly estimating impulse responses across all horizons** using a continuous B-spline basis regularized by a roughness difference penalty, achieving dramatic variance reduction while remaining asymptotically unbiased.

---

## 1. Econometric Methodology

### 1.1 Model Formulation

Let $y_t$ denote the dependent response variable, $x_t$ the structural shock or policy intervention, and $w_t$ a vector of control variables (such as lags of $y_t$, $x_t$, and other macro aggregates). In standard local projections:

$$y_{t+h} = \alpha_h + \beta_h x_t + \gamma_h' w_t + \varepsilon_{t+h}, \quad h = 0, \dots, H$$

Using the Frisch-Waugh-Lovell (FWL) theorem, we partial out the controls $w_t$ to obtain residualized response vectors $\tilde{y}$ and residualized shock vectors $\tilde{x}$.

Barnichon and Brownlees approximate the continuous impulse response function $\beta(h)$ as a linear combination of $K$ B-spline basis functions evaluated across the horizon grid $h \in \{0, 1, \dots, H\}$:

$$\beta(h) = \sum_{k=1}^K B_k(h) \theta_k = B_h \theta$$

where $B$ is an $(H+1) \times K$ clamped cubic B-spline basis matrix and $\theta \in \mathbb{R}^K$ is the spline parameter vector.

The full stacked system across all horizons is estimated via **Penalized Least Squares (PLS)** or **Penalized Generalized Least Squares (PGLS)**:

$$\min_\theta \; (\tilde{Y} - X \theta)' \Omega^{-1} (\tilde{Y} - X \theta) + \lambda \theta' P \theta$$

where:
- $X = B \otimes \tilde{x}$ is the Kronecker design matrix.
- $\Omega$ is the error covariance matrix across horizons (identity for PLS; estimated horizon covariance for PGLS).
- $P = D_d' D_d$ is the roughness penalty matrix formed by the $d$-th difference operator matrix $D_d$ (typically second-order difference $d=2$).
- $\lambda \ge 0$ is the regularization parameter governing the bias-variance trade-off:
  - As $\lambda \to 0$, the estimates converge to unpenalized OLS local projections.
  - As $\lambda \to \infty$, the impulse response is regularized into a smooth low-order polynomial curve.

### 1.2 Data-Driven $\lambda$ Selection

`puremacro.lp.smooth_lp` provides automated, data-driven optimization of the smoothing penalty $\lambda$:

1. **Akaike Information Criterion (`selection='aic'`)**: Minimizes penalized in-sample prediction error with an effective degrees of freedom penalty $\text{df}(\lambda) = \text{tr}((X' \Omega^{-1} X + \lambda P)^{-1} X' \Omega^{-1} X)$.
2. **Bayesian Information Criterion (`selection='bic'`)**: Applies a stronger sample-size penalty $\log(N) \cdot \text{df}(\lambda)$.
3. **Generalized Cross-Validation (`selection='gcv'`)**: Rotation-invariant leave-one-out cross-validation approximation.
4. **K-Fold Cross-Validation (`selection='cv'`)**: Out-of-sample block validation across time segments.

### 1.3 Statistical Inference

Two robust inference schemes are provided:

1. **Analytical Sandwich HAC (`ci_type='analytic'`)**:  
   Computes Newey-West / Bartlett kernel standard errors for the spline coefficients $\theta$, projected onto the horizon basis:
   $$\widehat{\text{Var}}(\hat{\beta}) = B \left( X' X + \lambda P \right)^{-1} X' \hat{\Sigma}_{HAC} X \left( X' X + \lambda P \right)^{-1} B'$$
2. **Moving Block Bootstrap (`ci_type='bootstrap'`)**:  
   Resamples overlapping temporal blocks of observations to preserve serial dependence and heteroskedasticity non-parametrically across horizons.

---

## 2. Empirical Variance Reduction

On calibrated macroeconomic DGPs (e.g., standard monetary VARs or AR(2) investment series):
- Unpenalized Local Projections suffer from high sample variance, with empirical standard deviations that grow rapidly with horizon $h$.
- Smooth Local Projections achieve **30% to 50% lower root mean squared error (RMSE)** across intermediate and distant horizons ($h \ge 4$) while maintaining nominal coverage rates.

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

# 3. Print Results & Summary
print(res_smooth.summary())
print(f"Optimal Lambda: {res_smooth.attrs.get('lambda_optimal', 'N/A')}")

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
# Fixed small lambda approximating unpenalized OLS LP
res_ols = smooth_lp(df=df, y="y", x="shock", horizons=20, lam=1e-4)

# Data-driven smooth LP
res_opt = smooth_lp(df=df, y="y", x="shock", horizons=20, lam="auto", selection="bic")

# Comparison: optimal smoothing yields tighter, non-oscillating bands
print("OLS average SE   :", np.mean(res_ols.se))
print("Smooth average SE:", np.mean(res_opt.se))
```

---

## 4. Full API Specification

### `smooth_lp`

```python
smooth_lp(
    df: pd.DataFrame | np.ndarray,
    y: str | np.ndarray | None = None,
    x: str | np.ndarray | None = None,
    horizons: int | Iterable[int] = 20,
    n_lags: int = 4,
    controls: Sequence[str] | np.ndarray | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    penalty_order: int = 2,
    lam: float | str | None = "auto",
    selection: str = "aic",
    alpha: float = 0.05,
    ci_type: str = "analytic",
    *,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = None,
    hac_lags: int | None = None,
    gls: bool = False,
) -> LPResult
```

#### Parameters:
- `df`: Input `pandas.DataFrame` or `numpy.ndarray` containing target, shock, and controls.
- `y`: Column name or array of the response variable.
- `x`: Column name or array of the shock variable.
- `horizons`: Maximum horizon integer $H$ or iterable sequence of horizons.
- `n_lags`: Number of autoregressive lags included for target, shock, and controls.
- `controls`: Optional list of exogenous control variable names.
- `n_knots`: Number of internal spline knots (adaptive default if `None`).
- `degree`: Polynomial degree of the B-splines (default `3` for cubic splines).
- `penalty_order`: Difference order of the roughness penalty matrix $P = D_d' D_d$ (default `2`).
- `lam`: Regularization parameter $\lambda$. Set to `'auto'` for data-driven selection or provide a positive float.
- `selection`: Information criterion for $\lambda$ optimization: `'aic'`, `'bic'`, `'gcv'`, or `'cv'`.
- `alpha`: Significance level for confidence intervals (default `0.05` for 95% coverage).
- `ci_type`: Inference method: `'analytic'` (sandwich HAC) or `'bootstrap'` (moving block bootstrap).
- `n_boot`: Number of bootstrap replications when `ci_type='bootstrap'`.
- `seed`: Random number generator seed for bootstrap reproducibility.
- `gls`: If `True`, performs feasible Generalized Least Squares weighting across horizons.

---

## 5. Result Interface

`smooth_lp` returns a specialized `LPResult` object (subclassing `pandas.DataFrame`):

- **Data Columns**:
  - `point`: Point estimates $\hat{\beta}(h)$.
  - `se`: Standard errors.
  - `ci_lower`, `ci_upper`: Pointwise confidence intervals.
  - `t_stat`: $t$-statistics testing $H_0: \beta(h) = 0$.
  - `p_value`: Asymptotic $p$-values.
- **Methods**:
  - `.plot()`: Matplotlib plot depicting the estimated impulse response function with shaded confidence bands.
  - `.summary()`: Diagnostic text summary with regression diagnostics and optimal smoothing parameters.
  - `.to_markdown()`, `.to_latex()`, `.to_typst()`: Formatted tables ready for publication.
