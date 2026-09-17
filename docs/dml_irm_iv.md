> 🇬🇧 English · 🇪🇸 [Español](es/dml_irm_iv.md)

# Double / Debiased Machine Learning for Treatment Effects & Instrumental Variables (IRM & DML-IV)

While the Partially Linear Regression (PLR) model assumes a constant additive treatment effect ($Y = D \theta_0 + g_0(X) + U$), empirical economic applications frequently involve:
1. **Treatment Effect Heterogeneity**: The causal effect of a policy $D \in \{0, 1\}$ varies arbitrarily across individual or regional characteristics $X$. This is governed by the **Interactive Regression Model (IRM)** for the **Average Treatment Effect (ATE)** and **Average Treatment Effect on the Treated (ATT)**.
2. **Endogenous Policies & Excluded Instruments**: The treatment $D$ is endogenous ($\mathbb{E}[U \mid D, X] \neq 0$), but researchers observe exogenous policy instruments $Z$ along with high-dimensional confounders $X$. This is governed by **Double ML Instrumental Variables (DML-IV)**.

`puremacro.causal` provides institutional implementations of both frameworks:

- **`DoubleMLIRM` & `dml_irm`**: Neyman-orthogonal doubly robust scores with pure-NumPy coordinate descent for regularized logistic propensity classification, automatic overlap trimming, and sensitivity diagnostic plots (`.plot_overlap()`, `.plot_coefficients()`, `.plot_tuning()`).
- **`DoubleMLIV` & `dml_iv`**: Cross-fitted 2SLS orthogonal residualization with Montiel Olea & Pflueger (2013) effective $F$-statistic weak-instrument diagnostics.
- **`LogisticCoordinateDescent`**: Pure-NumPy $\ell_1$ (Lasso), $\ell_2$ (Ridge), and ElasticNet logistic regression utilizing upper-bound quadratic surrogates and AIC/BIC path selection.

Zero C++ compiler toolchains or external machine learning dependencies (`scikit-learn`, `torch`) are required. All estimators adhere strictly to the Pyodide 4-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`).

---

## 1. Theoretical & Algorithmic Framework

### 1.1 The Interactive Regression Model (IRM)

Let $Y \in \mathbb{R}$ be the outcome, $D \in \{0, 1\}$ a binary policy intervention, and $X \in \mathbb{R}^p$ a vector of high-dimensional confounding covariates. Under unconfoundedness (conditional ignorability) and overlap:

$$(Y(1), Y(0)) \perp D \mid X, \qquad \varepsilon \le m_0(X) \le 1 - \varepsilon$$

The structural relationship satisfies:

$$Y = g_0(D, X) + U, \qquad \mathbb{E}[U \mid D, X] = 0$$

$$D = m_0(X) + V, \qquad \mathbb{E}[V \mid X] = 0, \quad m_0(X) \equiv \mathbb{P}(D = 1 \mid X)$$

Here, $g_0(d, X) \equiv \mathbb{E}[Y \mid D = d, X]$ models the conditional expectation of outcomes under treatment state $d \in \{0, 1\}$, and $m_0(X)$ is the propensity score.

#### Neyman-Orthogonal Scores for ATE & ATT

The Neyman-orthogonal score for the **Average Treatment Effect (ATE)** $\theta_0 = \mathbb{E}[Y(1) - Y(0)]$ is the doubly robust moment function (Chernozhukov et al. 2018):

$$\psi_{\text{ATE}}(W; \theta, \eta) = g(1, X) - g(0, X) + \frac{D \big(Y - g(1, X)\big)}{m(X)} - \frac{(1 - D) \big(Y - g(0, X)\big)}{1 - m(X)} - \theta$$

The score for the **Average Treatment Effect on the Treated (ATT)** $\theta_0 = \mathbb{E}[Y(1) - Y(0) \mid D = 1]$ is:

$$\psi_{\text{ATT}}(W; \theta, \eta) = \frac{D \big(Y - g(0, X)\big)}{\mathbb{P}(D = 1)} - \frac{m(X)(1 - D)\big(Y - g(0, X)\big)}{\mathbb{P}(D = 1)\big(1 - m(X)\big)} - \theta$$

Both scores satisfy $\mathbb{E}[\psi(W; \theta_0, \eta_0)] = 0$ and have zero Gateaux derivatives with respect to the nuisance functions $\eta = (g(0, \cdot), g(1, \cdot), m(\cdot))$ at the truth. Consequently, first-order approximation errors from regularized machine learners for $g$ and $m$ do not bias the causal estimate $\hat{\theta}$.

### 1.2 Pure-NumPy Regularized Logistic Coordinate Descent

To estimate the propensity score $m_0(X) = \sigma(X \beta) = \frac{1}{1 + e^{-X \beta}}$ without external dependencies, `puremacro` implements `LogisticCoordinateDescent`. The negative log-likelihood with ElasticNet penalty is:

$$\min_{\beta_0, \beta} -\frac{1}{N} \sum_{i=1}^N \left[ D_i \ln p_i + (1 - D_i) \ln(1 - p_i) \right] + \lambda \left[ \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right]$$

Because the logistic Hessian $p_i(1 - p_i)$ is upper-bounded by $1/4$, coordinate updates can be computed using the global quadratic surrogate curvature:

$$c_j = \frac{1}{4N} \sum_{i=1}^N X_{i, j}^2, \qquad c_0 = \frac{1}{4}$$

At each coordinate sweep, the gradient is $g_j = \frac{1}{N} X_j^\top (p - D)$, and the partial residual update is:

$$\rho_j = c_j \beta_j - g_j$$

$$\beta_j^{\text{new}} = \frac{S\big(\rho_j, \, \lambda \alpha\big)}{c_j + \lambda(1 - \alpha)}$$

where $S(z, \tau) \equiv \operatorname{sign}(z) \max(0, |z| - \tau)$ is the soft-thresholding operator. The optimal penalty $\lambda^\star$ is selected along a geometric path via the Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC):

$$\text{BIC}(\lambda) = \text{Deviance}(\lambda) + \operatorname{df}(\lambda) \ln N$$

### 1.3 Overlap Trimming

When propensity scores approach $0$ or $1$, the inverse probability weights $1/m(X)$ and $1/(1 - m(X))$ explode, destabilizing finite-sample variances. `DoubleMLIRM` enforces strict overlap trimming:

- **`trimming_rule="clip"`**: Propensity scores are projected into the safe compact interval:
  $$\hat{m}(X) \leftarrow \operatorname{clip}\big(\hat{m}(X), \, \varepsilon, \, 1 - \varepsilon\big), \qquad \varepsilon = \text{trimming\_threshold}$$
- **`trimming_rule="drop"`**: Observations where $\hat{m}(X_i) < \varepsilon$ or $\hat{m}(X_i) > 1 - \varepsilon$ are discarded from the score evaluation.

### 1.4 Double ML Instrumental Variables (DML-IV)

When treatment $D$ is endogenous, let $Z \in \mathbb{R}^{k_z}$ denote excluded instrumental variables:

$$Y = D^\top \theta_0 + g_0(X) + U, \qquad \mathbb{E}[U \mid X, Z] = 0$$

$$D = r_0(X, Z) + V$$

The DML-IV estimator solves this by residualizing three conditional expectations via $K$-fold cross-fitting:
1. $\ell_0(X) \equiv \mathbb{E}[Y \mid X] \implies \tilde{Y} = Y - \hat{\ell}^{(-k)}(X)$
2. $m_0(X) \equiv \mathbb{E}[D \mid X] \implies \tilde{D} = D - \hat{m}^{(-k)}(X)$
3. $r_0(X) \equiv \mathbb{E}[Z \mid X] \implies \tilde{Z} = Z - \hat{r}^{(-k)}(X)$

The structural parameter $\theta_0$ is then estimated by **Two-Stage Least Squares (2SLS)** on the orthogonal out-of-fold residuals:

$$\hat{\theta} = \left( \tilde{Z}^\top \tilde{D} \right)^{-1} \tilde{Z}^\top \tilde{Y}$$

#### Montiel Olea & Pflueger (2013) Weak-IV Diagnostics

To guard against weak instruments in high dimensions, `DoubleMLIV` computes both the conventional first-stage $F$-statistic and the **Montiel Olea & Pflueger (2013) effective $F$-statistic**, which is robust to arbitrary heteroskedasticity and clustering:

$$F_{\text{eff}} = \frac{\tilde{D}^\top \tilde{Z} (\tilde{Z}^\top \tilde{Z})^{-1} \tilde{Z}^\top \tilde{D}}{\operatorname{tr}(\hat{W})}$$

`DMLIVResult` reports `first_stage_f`, `first_stage_effective_f`, and flags `weak_instrument = True` whenever $F_{\text{eff}} < 10.0$ (or below the critical threshold for a 10% worst-case Nagar bias).

---

## 2. API Architecture & Public Objects

| Object | Type | Signature | Description |
|---|---|---|---|
| `DoubleMLIRM` | Class | `DoubleMLIRM(ml_g, ml_m, n_folds=5, score='ATE', trimming_threshold=0.01, trimming_rule='clip')` | Interactive Regression Model estimator for ATE and ATT. |
| `dml_irm` | Function | `dml_irm(Y, D, X, ...)` | Functional one-call interface returning `DMLIRMResult`. |
| `DMLIRMResult` | Frozen Dataclass | Attributes: `theta`, `se`, `t_stat`, `p_value`, `propensity_scores`, `n_trimmed`, etc. | Holds ATE/ATT estimates, standard errors, and diagnostics. |
| `DoubleMLIV` | Class | `DoubleMLIV(ml_l, ml_m, ml_r, n_folds=5)` | Double ML Instrumental Variables estimator. |
| `dml_iv` | Function | `dml_iv(Y, D, Z, X, ...)` | Functional one-call interface returning `DMLIVResult`. |
| `DMLIVResult` | Frozen Dataclass | Attributes: `theta`, `se`, `first_stage_f`, `first_stage_effective_f`, `weak_instrument`, etc. | Holds IV estimates and weak-instrument diagnostic metrics. |
| `LogisticCoordinateDescent` | Class | `LogisticCoordinateDescent(penalty='l1', C=1.0, max_iter=1000, tol=1e-5)` | Pure-NumPy regularized logistic classifier with BIC path tuning. |

---

## 3. End-to-End Walkthrough

### 3.1 Estimating Heterogeneous Treatment Effects (DML-IRM)

```python
import numpy as np
import pandas as pd
from puremacro.causal import DoubleMLIRM, LogisticCoordinateDescent

# 1. Generate synthetic DGP with confounding and heterogeneous effects
rng = np.random.default_rng(2026)
N, p = 1000, 20
X = rng.normal(size=(N, p))

# True propensity score depends on first 3 controls
logit_m = 0.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2]
m_prob = 1.0 / (1.0 + np.exp(-logit_m))
D = rng.binomial(1, m_prob)

# True outcome: baseline g0 + heterogeneous treatment effect theta(X)
true_ate = 2.5
hetero_effect = true_ate + 0.5 * X[:, 0]
Y = 1.2 * X[:, 0] + 0.8 * X[:, 1]**2 + D * hetero_effect + rng.normal(scale=1.0, size=N)

# 2. Configure learners: Ridge/Lasso for regression, LogisticCoordinateDescent for propensity
irm = DoubleMLIRM(
    ml_g="ridge",
    ml_m="logistic",
    n_folds=5,
    score="ATE",
    trimming_threshold=0.02,
    trimming_rule="clip",
    random_state=42,
)
res_irm = irm.fit(Y, D, X)

print(res_irm.summary())
print(f"Estimated ATE: {res_irm.theta:.4f} ± {1.96 * res_irm.se:.4f} (True: {true_ate:.2f})")
print(f"Observations trimmed: {res_irm.n_trimmed}")
```

### 3.2 Visualizing Propensity Overlap & Regularization Diagnostics

`DMLIRMResult` provides three built-in diagnostic plotting methods:

```python
# 1. Overlap plot: checks common support between treatment and control groups
fig1 = res_irm.plot_overlap()

# 2. Coefficients plot: inspects regularized nuisance weights
fig2 = res_irm.plot_coefficients()

# 3. Tuning plot: examines BIC/AIC loss along the regularization path
fig3 = res_irm.plot_tuning()
```

### 3.3 Instrumental Variables with High-Dimensional Controls (DML-IV)

```python
from puremacro.causal import DoubleMLIV

# 1. Generate endogenous treatment DGP
# Z: excluded instrument, U: unobserved confounder affecting D and Y
U = rng.normal(size=N)
Z = rng.normal(size=N)
D_endo = 1.5 * Z + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 1.2 * U + rng.normal(scale=0.5, size=N)

true_iv_theta = 1.75
Y_iv = D_endo * true_iv_theta + 2.0 * X[:, 0] + 1.5 * U + rng.normal(scale=1.0, size=N)

# 2. Fit Double ML IV
iv_model = DoubleMLIV(
    ml_l="lasso",
    ml_m="lasso",
    ml_r="lasso",
    n_folds=5,
    random_state=42,
)
res_iv = iv_model.fit(Y_iv, D_endo, Z, X)

print(res_iv.summary())
print(f"Estimated IV Theta : {res_iv.theta:.4f} (True: {true_iv_theta:.2f})")
print(f"First-Stage F-stat : {res_iv.first_stage_f:.2f}")
print(f"Effective F-stat   : {res_iv.first_stage_effective_f:.2f}")
print(f"Weak Instrument?   : {res_iv.weak_instrument}")
```

---

## 4. Publication Tables: Markdown, $\LaTeX$ & Typst

All DML result objects include one-call publication formatters:

```python
# GitHub-Flavored Markdown table
print(res_irm.to_markdown())

# Academic LaTeX table with booktabs
print(res_irm.to_latex())

# Modern Typst table
print(res_irm.to_typst())
```

---

## References

1. Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1–C68.
2. Montiel Olea, J. L. and Pflueger, C. (2013). "A robust test for weak instruments." *Journal of Business & Economic Statistics*, 31(3), 358–369.
3. Robinson, P. M. (1988). "Root-N-consistent semiparametric regression." *Econometrica*, 56(4), 931–954.
