> 🇬🇧 English · 🇪🇸 [Español](es/causal_dml.md)

# Double / Debiased Machine Learning (DML-PLR)

`puremacro.causal.dml` (re-exported from `puremacro.causal`) implements the **Double / Debiased Machine Learning** estimator of **Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins (2018, *The Econometrics Journal*)** for the **partially linear regression (PLR)** model, built on the partialling-out idea of **Robinson (1988, *Econometrica*)** and the high-dimensional-controls program of **Belloni, Chernozhukov and Hansen (2014, *Review of Economic Studies*)**. The module exposes five public objects:

1. `DoubleMLPLR` — the estimator class, `DoubleMLPLR(...).fit(Y, D, X) -> DMLResult`.
2. `dml_plr` — a one-call functional interface with the same arguments.
3. `DMLResult` — a frozen dataclass holding the estimates, the out-of-fold residuals and the `summary` / `plot` / `to_markdown` / `to_latex` / `to_typst` presentation suite.
4. `LassoCoordinateDescent` — an $\ell_1$ learner (cyclical coordinate descent along a regularization path, model selected by BIC or AIC).
5. `RidgeGCV` — an $\ell_2$ learner (SVD closed form, penalty selected by generalized cross-validation).

For heterogeneous treatment effects (ATE/ATT) via the Interactive Regression Model (`DoubleMLIRM`), regularized logistic classification (`LogisticCoordinateDescent`), overlap trimming, and instrumental variables (`DoubleMLIV`), see the dedicated guide: [Double ML for Treatment Effects & IV (IRM & DML-IV)](dml_irm_iv.md).

Everything is pure NumPy / SciPy / pandas / Matplotlib in float64, needs no network access and runs unchanged in Pyodide. It complements the other causal pages, [Modern Difference-in-Differences](did.md) and [Honest DiD sensitivity analysis](honest_did.md), and is showcased end to end in notebook 57 (`notebooks/57_multiconstraint_occbin_and_dml.py`), where a policy multiplier is estimated with high-dimensional macroeconomic controls and compared with naive OLS and naive lasso.

---

## 1. Theoretical & Algorithmic Framework

### 1.1 The partially linear model

Let $Y$ be the outcome, $D \in \mathbb{R}^{k_d}$ the treatment or policy variable(s) whose effect is of interest, and $X \in \mathbb{R}^p$ a (possibly high-dimensional) vector of controls. The PLR model is

$$Y = D^\top \theta_0 + g_0(X) + U, \qquad \mathbb{E}[U \mid X, D] = 0,$$

$$D = m_0(X) + V, \qquad \mathbb{E}[V \mid X] = 0,$$

where $\theta_0$ is the low-dimensional structural parameter and $g_0$ and $m_0$ are unknown, potentially non-linear nuisance functions. Taking conditional expectations of the first equation defines a third nuisance, $\ell_0(X) \equiv \mathbb{E}[Y \mid X] = m_0(X)^\top \theta_0 + g_0(X)$.

### 1.2 Robinson partialling-out and the Neyman-orthogonal score

Subtracting $\ell_0(X)$ from $Y$ and $m_0(X)$ from $D$ removes the nuisance entirely (Robinson 1988):

$$Y - \ell_0(X) = \big(D - m_0(X)\big)^\top \theta_0 + U .$$

The implied moment condition is the **Robinson / Neyman-orthogonal score**

$$\psi(W; \theta, \eta) = \Big(Y - \ell(X) - \big(D - m(X)\big)^\top \theta\Big)\big(D - m(X)\big), \qquad \eta = (\ell, m),$$

with $\mathbb{E}[\psi(W; \theta_0, \eta_0)] = 0$. The score is orthogonal because its Gateaux derivative with respect to the nuisances vanishes at the truth: perturbing $\ell$ in the direction $h(X)$ gives $-\mathbb{E}[h(X) V] = 0$, and perturbing $m$ in the direction $h(X)$ gives $\mathbb{E}[h(X)(\theta_0 V - U)] = 0$, both by $\mathbb{E}[V \mid X] = 0$ and $\mathbb{E}[U \mid X, D] = 0$. First-order estimation errors in $\hat\ell$ and $\hat m$ therefore do not transmit to $\hat\theta$; only their product matters, and it is $o_p(N^{-1/2})$ whenever each learner converges at rate $o_p(N^{-1/4})$. This is what removes the regularization bias of a single-equation lasso of $Y$ on $(D, X)$.

Writing $\tilde Y_i = Y_i - \hat\ell(X_i)$ and $\tilde D_i = D_i - \hat m(X_i)$, the solution of the empirical score is the OLS coefficient of $\tilde Y$ on $\tilde D$ without intercept:

$$\hat\theta = \Big( \sum_{i=1}^N \tilde D_i \tilde D_i^\top \Big)^{-1} \sum_{i=1}^N \tilde D_i \tilde Y_i .$$

`puremacro` solves this $k_d \times k_d$ system with `np.linalg.solve` and falls back to the Moore-Penrose pseudo-inverse if the residualized treatments are collinear.

### 1.3 $K$-fold cross-fitting (DML2: one pooled solve)

Using the same observations to fit $\hat\ell, \hat m$ and to evaluate the score reintroduces an overfitting bias. Cross-fitting removes it:

1. Draw a random permutation of the $N$ indices with `np.random.default_rng(random_state)` and split it into $K$ folds $I_1, \dots, I_K$ of (almost) equal size with `np.array_split`.
2. For each fold $k$, fit one learner for $\ell$ and one learner **per treatment column** for $m$ on the complement $I_k^c$, then residualize the held-out fold: $\tilde Y_i = Y_i - \hat\ell^{(-k)}(X_i)$ and $\tilde D_i = D_i - \hat m^{(-k)}(X_i)$ for $i \in I_k$. This costs $K (1 + k_d)$ learner fits in total.
3. Stack the out-of-fold residuals of all $N$ observations and solve the normal equations of Section 1.2 **once**.

Step 3 is the pooled "DML2" variant of Chernozhukov et al. (2018, Definition 3.2): the point estimate is the solution of the score summed over all folds, not the average of $K$ fold-specific estimates (DML1). The `summary()` header reports this as `Score: Robinson Orthogonal (DML2)`, and the estimate can be reproduced exactly from the residuals stored on the result, $\hat\theta = (\tilde D^\top \tilde D)^{-1} \tilde D^\top \tilde Y$. Because the split is a single seeded permutation, the estimate depends on `random_state` (see Section 6).

### 1.4 Sandwich variance and inference

With $\hat U_i = \tilde Y_i - \tilde D_i^\top \hat\theta$, the plug-in sandwich estimator is

$$\hat J = \frac{1}{N} \sum_i \tilde D_i \tilde D_i^\top, \qquad \hat\Omega = \frac{1}{N} \sum_i \hat U_i^2 \, \tilde D_i \tilde D_i^\top, \qquad \hat V = \frac{1}{N} \hat J^{-1} \hat\Omega \hat J^{-1},$$

i.e. a heteroskedasticity-robust (HC0) variance for the residual-on-residual regression, which is the $\sqrt{N}$-asymptotic variance of Theorem 3.1 in Chernozhukov et al. (2018). Standard errors are $\sqrt{\operatorname{diag} \hat V}$, `t_stat` is $\hat\theta / \widehat{se}$ (`nan` if a standard error is zero), `p_value` is two-sided under the standard normal, and the confidence interval is $\hat\theta \pm z_{1 - \alpha / 2} \, \widehat{se}$ with the normal quantile from `scipy.stats.norm`. No degrees-of-freedom correction, clustering or HAC adjustment is applied.

### 1.5 Nuisance learners

**Lasso (`LassoCoordinateDescent`).** With `fit_intercept=True` (the default) the columns of $X$ are standardized to zero mean and unit variance and $y$ is centered; with `fit_intercept=False` the penalty acts on the raw columns, and the coordinate update divides by the per-column second moment $c_j = \tfrac{1}{N} x_j^\top x_j$ so that it stays the exact coordinate minimizer whatever the column scale. The learner minimizes

$$\frac{1}{2N} \lVert y - X\beta \rVert_2^2 + \alpha \lVert \beta \rVert_1$$

by cyclical coordinate descent with the soft-thresholding update $\beta_j \leftarrow S(\rho_j, \alpha) / c_j$, $\rho_j = \tfrac{1}{N} x_j^\top r + c_j \beta_j$, stopping when the largest coefficient change in a sweep is below `tol` or after `max_iter` sweeps. When `alpha=None` (the default) the learner walks a geometric path of `n_alphas` penalties from $\alpha_{\max} = \max_j |x_j^\top y_c| / N$ (all coefficients zero; replaced by $10^{-3}$ if it falls below $10^{-12}$) down to $\alpha_{\max} \cdot$ `eps` (floored at $10^{-7}$), warm-starting each step from the previous solution, and keeps the penalty that minimizes

$$\text{BIC}(\alpha) = N \log \hat\sigma^2_\alpha + \mathrm{df}_\alpha \log N \qquad \text{or} \qquad \text{AIC}(\alpha) = N \log \hat\sigma^2_\alpha + 2 \, \mathrm{df}_\alpha ,$$

where $\hat\sigma^2_\alpha$ is the in-sample mean squared residual and $\mathrm{df}_\alpha$ the number of non-zero coefficients (`criterion="bic"` or `"aic"`). The chosen penalty is stored in `alpha_`, and `coef_` / `intercept_` are reported on the original scale. This is an information-criterion route rather than cross-validation or the theoretically tuned penalty of Belloni, Chernozhukov and Hansen (2014); the inner loop over the $p$ coordinates is Python-level, so cost grows linearly in $p$ per sweep.

**Ridge (`RidgeGCV`).** After the same standardization, the learner computes one economy SVD $X_s = U S V^\top$ and evaluates, for every penalty $a$ on the grid (`alphas`, default `np.logspace(-4, 6, 100)`), the generalized cross-validation criterion

$$\text{GCV}(a) = \frac{\operatorname{RSS}(a) / N}{\big(1 - \operatorname{tr} H(a) / N\big)^2}, \qquad \operatorname{tr} H(a) = \sum_i \frac{s_i^2}{s_i^2 + a},$$

in $O(p)$ per grid point, then sets $\hat\beta = V \operatorname{diag}\!\big(s_i / (s_i^2 + a^\star)\big) U^\top y_c$ at the GCV minimizer $a^\star$ (stored in `alpha_`). The penalty acts on the un-normalized objective $\lVert y_c - X_s \beta \rVert_2^2 + a \lVert \beta \rVert_2^2$ with standardized columns (so $s_i^2 \approx N$ for orthogonal regressors), which is the scale to use when supplying a custom `alphas` grid.

**User-supplied learners.** Any object exposing `fit(X, y)` and `predict(X)` can be passed as `learner`. If it is a class (or any callable), it is called with `learner_kwargs` to build a fresh learner for every nuisance regression; if it is an instance, it is `copy.deepcopy`-ed for every nuisance regression, so each of the $K(1 + k_d)$ fits gets its own object and the instance you passed is never fitted (its `coef_` is still `None` afterwards). Combining an instance with `learner_kwargs` raises `ValueError` rather than dropping the keywords silently: configure the instance itself, or pass the class.

---

## 2. Methodological & Model Options

| Dimension | `learner="lasso"` (`"l1"`) | `learner="ridge"` (`"l2"`) | Custom `fit`/`predict` object |
|---|---|---|---|
| **Nuisance model** | Sparse linear in the supplied features | Dense linear, shrinks all coefficients | Anything (trees, kernels, your own basis) |
| **Tuning** | Path of `n_alphas` penalties, BIC (default) or AIC | GCV over `alphas` grid (100 points by default) | Inside the object |
| **Fixed penalty** | `learner_kwargs={"alpha": a}` or `LassoCoordinateDescent(alpha=a)` | `RidgeGCV(alphas=[a])` | n/a |
| **Cost per fit** | Python loop over $p$ coordinates $\times$ sweeps $\times$ path length | One SVD of the $(N_{\text{train}} \times p)$ training block | Yours |
| **Best suited to** | Many candidate controls, few relevant ($p$ up to a few hundred) | Correlated controls, $p \ll N$, speed | Non-linear nuisances without hand-built dictionaries |
| **Reported `learner` field** | The alias string you passed | The alias string you passed | The class name (`'RidgeGCV'`) for both a class and an instance; the function's `__name__` for a factory function |

Common to every option: `n_folds` $K$ with $2 \le K \le N$ (default 5), `alpha` as the CI significance level (default 0.05, i.e. 95% intervals), `random_state` for the fold permutation (default 42), and NumPy arrays, pandas `Series` or `DataFrame` inputs. Both built-in learners are linear in the columns of `X`; capturing non-linear confounding is the user's job through a dictionary of transformations (squares, interactions, splines), as in Example 1.

---

## 3. Runnable Worked Examples

Every script below runs offline in well under a second of compute (plus package import) and is seeded.

### 3.1 Planted effect with a non-linear confounder: naive OLS versus DML

$X_1$ enters the treatment equation through $X_1 + X_1^2 - 1$ and the outcome through $X_1 + 1.5 (X_1^2 - 1)$. OLS of $Y$ on $D$ alone absorbs the confounder; adding $X$ linearly does not help because $X_1^2 - 1$ is uncorrelated with $X_1$. DML with a quadratic dictionary lets the lasso pick $X_1^2$ in both nuisance regressions.

```python
import numpy as np
from puremacro.causal import dml_plr

rng = np.random.default_rng(0)
N, p, theta_0 = 1000, 10, 0.5
X = rng.standard_normal((N, p))
q = X[:, 0] ** 2 - 1.0                                   # nonlinear confounder
D = X[:, 0] + q + rng.standard_normal(N)                 # m_0(X) = X_1 + X_1^2 - 1
Y = theta_0 * D + X[:, 0] + 1.5 * q + 0.5 * X[:, 1] + rng.standard_normal(N)

def ols_slope(y, regressors):
    Z = np.column_stack([np.ones(len(y)), regressors])
    return np.linalg.lstsq(Z, y, rcond=None)[0][1]

theta_ols_d = ols_slope(Y, D[:, None])                   # Y on D only
theta_ols_lin = ols_slope(Y, np.column_stack([D, X]))     # Y on D and linear X

dictionary = np.column_stack([X, X ** 2])                # 2p = 20 controls
res = dml_plr(Y, D, dictionary, n_folds=5, learner="lasso", random_state=0)

print(f"true theta            : {theta_0:.3f}")
print(f"OLS, D only           : {theta_ols_d:.3f}")
print(f"OLS, D + linear X     : {theta_ols_lin:.3f}")
print(f"DML-PLR (lasso, K=5)  : {res.theta:.3f}  se={res.se:.3f}  95% CI=[{res.ci_lower:.3f}, {res.ci_upper:.3f}]")
print(res.summary())
```

Output:

```text
true theta            : 0.500
OLS, D only           : 1.535
OLS, D + linear X     : 1.546
DML-PLR (lasso, K=5)  : 0.533  se=0.033  95% CI=[0.469, 0.597]
==============================================================================
Double / Debiased Machine Learning (DML-PLR)
Model: Partially Linear Regression (Robinson 1988 / Chernozhukov et al. 2018)
==============================================================================
Outcome: Y                  Observations: 1000         Folds: 5
Learner: lasso              Score: Robinson Orthogonal (DML2)
------------------------------------------------------------------------------
Variable              Coef.   Std.Err.        z    P>|z| [95% Conf. Interval]
------------------------------------------------------------------------------
D                    0.5331     0.0328   16.259   <0.001     0.4688     0.5973
==============================================================================
```

Both OLS specifications triple the true effect (1.535 and 1.546 against 0.5); DML returns 0.533 with a 95% interval $[0.469, 0.597]$ that covers the truth. In this deliberately low-dimensional case OLS on the full 20-column dictionary would also be unbiased (0.524 on these data); the point of DML is that the dictionary can grow with $N$ (all pairwise interactions, spline bases) while inference on $\theta$ remains $\sqrt{N}$-valid.

### 3.2 Lasso versus ridge, learner keyword arguments and a user-supplied learner

The data-generating process is deliberately sparse: only $X_1$ drives $D$ and only $X_2$ drives $Y$ beyond $D$, with true $\theta_0 = 0.5$.

```python
import numpy as np
from puremacro.causal import dml_plr, LassoCoordinateDescent, RidgeGCV

rng = np.random.default_rng(0)
N, p = 300, 20
X = rng.standard_normal((N, p))
D = X[:, 0] + rng.standard_normal(N)
Y = 0.5 * D + X[:, 1] + rng.standard_normal(N)

res_lasso = dml_plr(Y, D, X, n_folds=3, learner="lasso")                  # BIC path (default)
res_aic = dml_plr(Y, D, X, n_folds=3, learner="lasso", criterion="aic")   # learner kwarg
res_ridge = dml_plr(Y, D, X, n_folds=3, learner="ridge")                  # SVD-GCV

class OLSLearner:
    """Any object with fit(X, y) and predict(X) is a valid nuisance learner."""
    def fit(self, X, y):
        Z = np.column_stack([np.ones(len(y)), X])
        self.beta_ = np.linalg.lstsq(Z, y, rcond=None)[0]
        return self
    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.beta_

res_ols = dml_plr(Y, D, X, n_folds=3, learner=OLSLearner())

for r in (res_lasso, res_aic, res_ridge, res_ols):
    print(f"{r.learner:<12} theta={r.theta:.4f}  se={r.se:.4f}  z={r.t_stat:.2f}")

# The standalone learners are usable on their own
lasso = LassoCoordinateDescent(criterion="bic").fit(X, D)
ridge = RidgeGCV().fit(X, D)
print("lasso: alpha_=%.4f  nonzero coef=%d" % (lasso.alpha_, np.count_nonzero(lasso.coef_)))
print("ridge: alpha_=%.4f  coef_[0]=%.3f" % (ridge.alpha_, ridge.coef_[0]))
```

Output:

```text
lasso        theta=0.4770  se=0.0522  z=9.13
lasso        theta=0.4624  se=0.0516  z=8.96
ridge        theta=0.4645  se=0.0514  z=9.04
OLSLearner   theta=0.4660  se=0.0518  z=8.99
lasso: alpha_=0.1102  nonzero coef=1
ridge: alpha_=17.8865  coef_[0]=1.004
```

All four learners land within one standard error of each other (0.462 to 0.477) because $N = 300$ comfortably exceeds $p = 20$; the standalone BIC lasso selects exactly the one true regressor of $D$, and ridge-GCV shrinks the same coefficient to 1.004. The `criterion="aic"` keyword is forwarded to `LassoCoordinateDescent` through `**learner_kwargs`; the `learner` field of the result stays `"lasso"` because it records the alias, not the settings.

### 3.3 Multiple treatments, pandas inputs and manuscript export

With a `DataFrame` of treatments the estimates come back as arrays, one nuisance regression is fitted per treatment column, and the column and `Series` names flow into the tables. `alpha=0.10` requests 90% intervals.

```python
import numpy as np
import pandas as pd
from puremacro.causal import DoubleMLPLR

rng = np.random.default_rng(3)
N, p = 400, 8
X = pd.DataFrame(rng.standard_normal((N, p)), columns=[f"x{j}" for j in range(p)])
D = pd.DataFrame({
    "policy_rate": 0.8 * X["x0"] + rng.standard_normal(N),
    "credit_spread": -0.6 * X["x1"] + rng.standard_normal(N),
})
Y = pd.Series(
    -0.4 * D["policy_rate"] + 0.9 * D["credit_spread"] + X["x0"] + X["x1"] + rng.standard_normal(N),
    name="gdp_growth",
)

res = DoubleMLPLR(n_folds=4, learner="ridge", alpha=0.10, random_state=7).fit(Y, D, X)
print(res.treatment_names, res.outcome_name)
print("theta:", np.round(res.theta, 4), " se:", np.round(res.se, 4))
print("residuals_d shape:", res.residuals_d.shape, " ci_level:", res.ci_level)
print(res.to_markdown())
print(res.to_typst())

# No to_frame(): assemble a DataFrame from the fields when you need one
table = pd.DataFrame(
    {"theta": res.theta, "se": res.se, "ci_lower": res.ci_lower, "ci_upper": res.ci_upper},
    index=res.treatment_names,
)
print(table.round(4))
```

Output:

```text
('policy_rate', 'credit_spread') gdp_growth
theta: [-0.394   0.8464]  se: [0.0479 0.0488]
residuals_d shape: (400, 2)  ci_level: 0.9
### DML-PLR: gdp_growth
*Learner: ridge | N: 400 | Folds: 4*

| Variable | Coef. | Std.Err. | z | P>\|z\| | [90% Conf. Interval] |
|:---|---:|---:|---:|---:|:---:|
| policy_rate | -0.3940 | 0.0479 | -8.229 | <0.001 | [-0.4728, -0.3152] |
| credit_spread | 0.8464 | 0.0488 | 17.342 | <0.001 | [0.7662, 0.9267] |
#figure(
  table(
    columns: (2fr, 1.2fr, 1.2fr, 1fr, 1fr, 2fr),
    align: (left, right, right, right, right, center),
    stroke: none,
    table.hline(),
    [*Variable*], [*Coef.*], [*Std.Err.*], [*z*], [*P>|z|*], [*90% CI*],
    table.hline(stroke: 0.5pt),
    [policy\_rate], [-0.3940], [0.0479], [-8.229], [\<0.001], [[-0.4728, -0.3152]],
    [credit\_spread], [0.8464], [0.0488], [17.342], [\<0.001], [[0.7662, 0.9267]],
    table.hline(),
  ),
  caption: [Double Machine Learning Estimates (ridge, N=400)],
)
                theta      se  ci_lower  ci_upper
policy_rate   -0.3940  0.0479   -0.4728   -0.3152
credit_spread  0.8464  0.0488    0.7662    0.9267
```

The planted effects were $-0.4$ and $0.9$; the 90% intervals $[-0.473, -0.315]$ and $[0.766, 0.927]$ cover both. `to_markdown()` escapes the pipes of the `P>|z|` header as `P>\|z\|`, so the header and delimiter rows agree on six cells and the table renders in strict GFM and in mkdocs; `to_typst()` escapes the underscores of the treatment names and the `<` of the p-value cell. `to_latex()` produces the matching booktabs table, escaping `gdp_growth` in `\caption{}` as well (see Section 5).

---

## 4. Full API Specification

```text
DoubleMLPLR(
    n_folds: int = 5,
    learner: str | Any = "lasso",
    alpha: float = 0.05,
    random_state: int | None = 42,
    learner_kwargs: dict[str, Any] | None = None,
)

DoubleMLPLR.fit(
    Y: np.ndarray | pd.Series,
    D: np.ndarray | pd.Series | pd.DataFrame,
    X: np.ndarray | pd.DataFrame,
) -> DMLResult

dml_plr(
    Y: np.ndarray | pd.Series,
    D: np.ndarray | pd.Series | pd.DataFrame,
    X: np.ndarray | pd.DataFrame,
    n_folds: int = 5,
    learner: str | Any = "lasso",
    alpha: float = 0.05,
    random_state: int | None = 42,
    **learner_kwargs: Any,
) -> DMLResult

LassoCoordinateDescent(
    alpha: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-5,
    n_alphas: int = 50,
    eps: float = 1e-3,
    criterion: str = "bic",
    fit_intercept: bool = True,
)
LassoCoordinateDescent.fit(X, y) -> LassoCoordinateDescent
LassoCoordinateDescent.predict(X) -> np.ndarray

RidgeGCV(
    alphas: Sequence[float] | np.ndarray | None = None,
    fit_intercept: bool = True,
)
RidgeGCV.fit(X, y) -> RidgeGCV
RidgeGCV.predict(X) -> np.ndarray
```

#### `DoubleMLPLR` / `dml_plr` parameters

| Parameter | Meaning |
|---|---|
| `Y` | Outcome, shape $(N,)$. A pandas `Series` name becomes `outcome_name` (default `"Y"`). |
| `D` | Treatment(s), shape $(N,)$ or $(N, k_d)$. A `Series` name or `DataFrame` columns become `treatment_names`; unnamed arrays get `"D"` or `"D_1", "D_2", ...`. |
| `X` | Controls, shape $(N, p)$, array or `DataFrame` (converted with `to_numpy(dtype=float)`). A 1-D `X` is read as a single column; three or more dimensions raise `ValueError`, and row counts must agree or a `ValueError` is raised. |
| `n_folds` | Number of cross-fitting folds $K$; must be an integer $\ge 2$ and at most $N$ (`ValueError` otherwise). |
| `learner` | `"lasso"` / `"l1"`, `"ridge"` / `"l2"`, a learner class or callable (instantiated per nuisance fit with `learner_kwargs`), or a learner instance (deep-copied per nuisance fit). Any other string raises `ValueError`; a non-callable object without `fit`/`predict` raises `TypeError`. |
| `alpha` | Significance level of the confidence intervals; `ci_level = 1 - alpha`. Must lie strictly between 0 and 1 (`ValueError` otherwise). **Not** the lasso penalty. |
| `random_state` | Seed of the fold permutation. `None` draws a fresh permutation on every call, so results are not reproducible. |
| `learner_kwargs` / `**learner_kwargs` | Keyword arguments forwarded to the learner constructor, e.g. `criterion="aic"`, `n_alphas=100`, `alphas=np.logspace(-2, 4, 50)`. Because `alpha` is taken by the CI level, a fixed lasso penalty must go through `DoubleMLPLR(learner_kwargs={"alpha": ...})` or a `LassoCoordinateDescent(alpha=...)` instance. Passing keywords together with a learner *instance* raises `ValueError`. |

`fit` validates its inputs before any learner is built: `Y`, `D` or `X` containing `NaN` or `inf` raises `ValueError` naming the offending array, `D` with three or more dimensions or zero columns raises `ValueError`, and a training fold with $n_{\text{train}} \le p + 1$ observations emits a `UserWarning` (Section 6).

#### Learner parameters

| Parameter | Learner | Meaning |
|---|---|---|
| `alpha` | lasso | Fixed $\ell_1$ penalty; `None` selects it on the path. |
| `n_alphas`, `eps` | lasso | Length of the geometric path and ratio $\alpha_{\min} / \alpha_{\max}$. |
| `max_iter`, `tol` | lasso | Maximum coordinate-descent sweeps per penalty and convergence threshold on the largest coefficient change. |
| `criterion` | lasso | `"bic"` (default) or `"aic"`, case-insensitive. |
| `alphas` | ridge | Grid of $\ell_2$ penalties for GCV; default `np.logspace(-4, 6, 100)`. |
| `fit_intercept` | both | Center $y$, standardize the columns of $X$ and report an intercept (default `True`). |

Fitted attributes on both learners: `coef_` (original scale), `intercept_`, `alpha_` (selected penalty). Calling `predict` before `fit` raises `RuntimeError`; `fit` raises `ValueError` on a non-finite, empty or length-mismatched design, and both constructors reject a negative or non-finite penalty.

---

## 5. Result Interface & Manuscript Export

`DMLResult` is a frozen dataclass; attributes are read-only. For a single treatment the statistics are Python floats and `residuals_d` is one-dimensional; for $k_d > 1$ they are arrays of length $k_d$ and `residuals_d` has shape $(N, k_d)$.

| Field | Content |
|---|---|
| `theta` | Cross-fitted DML estimate(s) $\hat\theta$. |
| `se` | Sandwich standard error(s). |
| `t_stat`, `p_value` | $\hat\theta / \widehat{se}$ and the two-sided normal p-value. |
| `ci_lower`, `ci_upper` | Bounds of the $(1 - \alpha)$ interval. |
| `n_obs`, `n_folds` | $N$ and $K$. |
| `learner` | Alias string passed, or the class name of a learner class or instance (the `__name__` of a factory function). |
| `residuals_y`, `residuals_d` | Out-of-fold $\tilde Y$ and $\tilde D$; enough to recompute $\hat\theta$ and the variance. |
| `ci_level` | $1 - \alpha$. |
| `treatment_names`, `outcome_name` | Labels used by every export. |

Methods:

- `res.summary() -> str`: fixed-width text report (header with outcome, $N$, $K$, learner and score type, one row per treatment).
- `res.plot(kind="forest", ax=None, **kwargs) -> matplotlib.axes.Axes`: `"forest"` draws the point estimates with their confidence intervals and a zero line; `"residuals"` scatters $\tilde D$ against $\tilde Y$ with the fitted DML slope (first treatment only when $k_d > 1$). Accepts `figsize` and `alpha` (marker transparency) through `kwargs`; returns the axes, not the figure.
- `res.to_markdown() -> str`: level-3 heading, an italic line with learner / $N$ / $K$, and a six-column table. Pipes inside cells are escaped as `\|` — the `P>|z|` header and any `|` in a treatment name — so the header and delimiter rows have the same cell count and strict GFM renderers (mkdocs included) recognize the table.
- `res.to_latex() -> str`: a `table` environment with a `booktabs` tabular and nothing else; `booktabs` is the only package the document must load. The sample-size note is a `\multicolumn` row inside the tabular rather than a `\subcaption{}`, the p-value cell is set in math mode (`$<0.001$`) because a bare `<` in OT1 text mode prints as an inverted exclamation mark, and `puremacro.reports.latex_escape` is applied to the treatment names, to `outcome_name` in `\caption{}` and to the learner name.
- `res.to_typst() -> str`: a `#figure(table(...))` block with `hline` rules and a caption naming the learner and $N$; treatment names, the learner name and the `<0.001` cell pass through `puremacro.reports.typst_escape`.
- There is **no** `to_frame()`; Example 3.3 shows the two-line `pd.DataFrame` construction from the fields.

---

## 6. Caveats & Limitations

- **PLR only.** The interactive (IRM / ATE), instrumental (PLIV) and panel variants of Chernozhukov et al. (2018) are not implemented; the parameter is the partially linear coefficient, which equals the ATE only under constant effects.
- **Single split, no repetition.** One seeded permutation defines the folds; there is no repeated sample splitting with median or mean aggregation. Estimates move with `random_state` (the Example 3.2 ridge fit gives 0.524, 0.506 and 0.465 for `random_state=0`, `1` and `42`), so report the seed or average over several yourself.
- **Linear learners.** Lasso and ridge are linear in the columns you pass; non-linear confounding must be captured by an explicit dictionary of transformations (Example 3.1) or a custom learner. The lasso coordinate loop is pure Python over $p$, so dictionaries with thousands of columns are slow; ridge is one SVD per fit.
- **Penalty selection.** The lasso path uses BIC / AIC with the non-zero count as degrees of freedom, not cross-validation; ridge uses GCV on a fixed grid. Neither is the theoretically tuned penalty of Belloni, Chernozhukov and Hansen (2014).
- **Inference is i.i.d. and marginal.** The sandwich variance is HC0 with normal critical values; no clustering, no HAC correction for serially correlated $U$ (relevant for macro time series), no small-sample adjustment. With several treatments the intervals are marginal, and the full covariance matrix is not returned.
- **`alpha` is the CI level**, both in `DoubleMLPLR` and in `dml_plr`; the lasso penalty travels through `learner_kwargs` or a learner instance.
- **Missing values must be removed by you.** `fit` refuses non-finite input rather than propagating it: a `NaN` or `inf` anywhere in `Y`, `D` or `X` raises `ValueError: X contains NaN or inf; drop or impute missing observations before calling fit()` (with the offending array named) before any learner is built. Drop or impute first; the estimator does neither.
- **Thin training folds are only warned about.** When the smallest training fold has $n_{\text{train}} \le p + 1$ observations, `fit` emits a `UserWarning` and continues. In that regime `RidgeGCV` interpolates the training fold and both built-in learners return biased estimates whose nominal 95% intervals under-cover; raise $N$, lower `n_folds` or shrink the dictionary.
- **Runtime contract.** No network access, no torch / MLX / CuPy code paths, float64 throughout; the same NumPy / SciPy code path runs unchanged in Pyodide.

---

## References

- Belloni, A., Chernozhukov, V., & Hansen, C. (2014). "Inference on Treatment Effects after Selection among High-Dimensional Controls." *The Review of Economic Studies*, 81(2), 608–650.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1–C68.
- Robinson, P. M. (1988). "Root-N-Consistent Semiparametric Regression." *Econometrica*, 56(4), 931–954.
