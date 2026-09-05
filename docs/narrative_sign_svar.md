> 🇬🇧 English · 🇪🇸 [Español](es/narrative_sign_svar.md)

# Narrative Sign Restrictions in SVAR

`puremacro.var.identify.narrative_sign` implements the structural vector autoregression (SVAR) identification methodology of **Antolín-Díaz and Rubio-Ramírez (2018, *American Economic Review*)** and the magnitude inequality extensions of **Ludvigson, Ma, and Ng (2021, *Journal of Political Economy*)**.

Traditional sign-restricted SVARs (e.g., Faust 1998; Uhlig 2005; Rubio-Ramírez, Waggoner, and Zha 2010) frequently suffer from wide set-identification regions that yield economically inconclusive policy analysis. Narrative sign restrictions overcome this weakness by conditioning structural rotation matrices on historical narrative information—such as key central bank announcements, geopolitical oil shocks, or specific historical crisis dates—to rule out structural draws that contradict established historical consensus.

---

## 1. Econometric Methodology

### 1.1 The Reduced-Form VAR and Structural Rotations

Consider an $n$-variable reduced-form $\text{VAR}(p)$ model:

$$y_t = c + \sum_{l=1}^p A_l y_{t-l} + u_t, \quad u_t \sim \mathcal{N}(0, \Sigma)$$

where $\Sigma$ is the positive-definite reduced-form covariance matrix. Let $P = \text{chol}(\Sigma)$ denote the lower-triangular Cholesky factor such that $P P' = \Sigma$.

The structural shocks $\varepsilon_t \sim \mathcal{N}(0, I_n)$ are related to the reduced-form innovations $u_t$ via:

$$u_t = B \varepsilon_t = P Q \varepsilon_t$$

where $Q \in \mathcal{O}(n)$ is an orthonormal rotation matrix satisfying $Q Q' = Q' Q = I_n$. Structural impulse responses at horizon $h$ are given by:

$$\Psi_h = \Phi_h B = \Phi_h P Q$$

where $\Phi_h$ represents the $h$-step reduced-form moving-average coefficient matrix ($\Phi_0 = I_n$).

### 1.2 Narrative Restriction Typology

Antolín-Díaz and Rubio-Ramírez (2018) formalize narrative restrictions on structural shock realizations $\varepsilon_t = B^{-1} u_t = Q' P^{-1} u_t$ and their historical decompositions into distinct structural classes:

1. **Type I: Structural Shock Sign Restrictions (`'shock_sign'`)**  
   Restricts the sign of structural shock $j$ on a specific historical date $t^*$:
   $$\text{sign}(\varepsilon_{j, t^*}) = s, \quad s \in \{-1, +1\}$$
   *Example*: "The monetary policy shock in 1979Q4 (Volcker monetary tightening) was strictly positive."

2. **Type II: Historical Decomposition Dominance (`'hd_dominance'`, `dominance='most'`)**  
   Restricts structural shock $j$ to be the single most important contributor to the $(L+1)$-period cumulative unexpected change in variable $i$ ending on date $t_1^*$:
   $$|H_{i, j}(t_1^*, L)| \ge \max_{k \neq j} |H_{i, k}(t_1^*, L)|$$
   where the historical contribution of shock $k$ over a window of length $L$ is:
   $$H_{i, k}(t_1^*, L) = \sum_{l=0}^L (\Phi_l B)_{i, k} \, \varepsilon_{k, t_1^* - l}$$

3. **Type III: Historical Decomposition Overwhelming Dominance (`'hd_dominance'`, `dominance='overwhelming'`)**  
   Restricts structural shock $j$ to be larger in absolute contribution than all other shocks combined:
   $$|H_{i, j}(t_1^*, L)| \ge \sum_{k \neq j} |H_{i, k}(t_1^*, L)|$$
   *Example*: "The monetary shock was the overwhelming contributor to the unanticipated rise in the federal funds rate in 1979Q4."

4. **Type IV: Shock Magnitude Bounds (`'shock_bound'`)**  
   Imposes absolute inequality bounds on structural shock realizations per Ludvigson, Ma, and Ng (2021):
   $$\underline{m} \le |\varepsilon_{j, t^*}| \le \bar{m}$$
   *Example*: "The monetary shock during the Volcker regime change was at least 2 standard deviations in magnitude ($|\varepsilon_{j, t^*}| \ge 2.0$)."

### 1.3 Sampling and Importance Weighting (AD-RR Algorithm 1)

1. **Haar Rotation Draws**: Draw random Gaussian matrices $Z \sim \mathcal{N}(0, I_n)$ and compute $Q$ via QR decomposition with positive diagonal normalization ($R_{ii} > 0$).
2. **Traditional Sign Check**: Check if the resulting structural impulse responses $\Psi_h = \Phi_h P Q$ satisfy traditional sign restrictions specified in `sign_matrix`.
3. **Narrative Validation**: Compute the structural shock series $\varepsilon = u (B^{-1})'$ and evaluate each narrative restriction on its designated date(s).
4. **Importance Weighting**: To prevent conditioning from biasing the posterior toward parameter values for which the narrative events are unsurprising, surviving draws are weighted by:
   $$w = \frac{1}{\omega}$$
   where $\omega$ is the probability that the narrative restrictions hold when shocks on the restricted dates are redrawn i.i.d. standard normal $\mathcal{N}(0, I_n)$:
   - When all narrative restrictions are pure Type I (`'shock_sign'`) on $m$ distinct (date, shock) pairs, $\omega = 0.5^m$ is known in exact closed form ($w = 2^m$).
   - For Type II, III, or IV restrictions, $\omega$ is estimated via Monte Carlo simulation with $S = \text{n\_weight\_sims}$ draws.
5. **Weighted Inference**: Credible interval bands and medians are computed as pointwise weighted quantiles across surviving draws.

---

## 2. Acceptance Diagnostics & Diagnostic Metrics

The returned `NarrativeSignResult` provides rich diagnostic fields:

| Attribute | Type | Economic Meaning |
|---|---|---|
| `n_draws` | `int` | Total Haar rotation draws evaluated. |
| `n_traditional_accepted` | `int` | Number of draws satisfying traditional sign restrictions. |
| `n_narrative_accepted` | `int` | Number of draws satisfying both traditional and narrative restrictions. |
| `acceptance_rate` | `float` | Overall acceptance fraction (`n_narrative_accepted / n_draws`). |
| `traditional_acceptance_rate`| `float` | Fraction satisfying traditional sign restrictions (`n_traditional_accepted / n_draws`). |
| `narrative_acceptance_rate`  | `float` | Conditional narrative acceptance rate (`n_narrative_accepted / n_traditional_accepted`). |
| `weights` | `np.ndarray` | Raw AD-RR importance weights $1/\hat{\omega}$ for surviving draws. |
| `ess` / `effective_draws` | `float` | Kish Effective Sample Size: $(\sum w_i)^2 / \sum w_i^2$. Warns if weight concentration is severe. |
| `restriction_labels` | `tuple[str]` | Human-readable string representation of each restriction. |
| `restriction_fail_counts` | `tuple[int]` | Binding-ness diagnostic: count of traditionally accepted draws rejected by each restriction. |

---

## 3. Replication: The Volcker 1979Q4 Monetary Policy Shock

In October 1979, Federal Reserve Chairman Paul Volcker announced a dramatic shift in monetary policy operating procedures, targeting non-borrowed reserves and causing an unprecedented spike in the federal funds rate to break runaway inflation.

In a 3-variable VAR with Federal Funds Rate (FFR), Inflation, and Output Growth, Antolín-Díaz and Rubio-Ramírez (2018) impose:
- **Traditional signs**: +FFR, -Inflation, -Output Growth at horizon $h=0$.
- **Narrative Restriction 1 (Type I)**: Monetary shock in 1979Q4 is positive ($\varepsilon_{MP, 1979Q4} > 0$).
- **Narrative Restriction 2 (Type III)**: Monetary shock is the overwhelming contributor to the unexpected surge in the federal funds rate in 1979Q4.

Conditioning on these narrative events tightens the 68% credible band width by **more than 20%** across all variables compared to traditional sign restrictions alone, eliminating counter-intuitive persistent output growth expansions.

### Runnable Replication Code

```python
import numpy as np
import pandas as pd
from puremacro.var.identify import (
    NarrativeRestriction,
    identify_narrative_sign,
)

# 1. Load or prepare macroeconomic time series panel (FFR, Inflation, GDP Growth)
# Here we demonstrate on synthetic quarterly data calibrated to 1965Q1-2007Q4
rng = np.random.default_rng(42)
T = 172
dates = pd.date_range("1965-01-01", periods=T, freq="QE")
volcker_idx = dates.get_loc("1979-10-01")

# Calibrated stationary VAR data
Y = np.zeros((T, 3))
for t in range(1, T):
    Y[t] = 0.5 * Y[t-1] + rng.standard_normal(3)
# Plant Volcker tightening in 1979Q4
Y[volcker_idx, 0] += 3.5

# 2. Define Traditional Sign Restrictions at horizon 0
# Shock 0 = Monetary policy shock (+FFR, -Inflation, -Output Growth)
sign_matrix = {0: np.array([+1, -1, -1])}

# 3. Define Narrative Restrictions
restrictions = [
    # Type I: Monetary shock in 1979Q4 was positive (contractionary)
    (volcker_idx, 0, +1),
    # Type III: Monetary shock was overwhelming contributor to unexpected FFR change
    NarrativeRestriction(
        kind="hd_dominance",
        date=volcker_idx,
        shock=0,
        variable=0,  # FFR
        window=0,
        dominance="overwhelming",
    ),
]

# 4. Estimate SVAR with Narrative Restrictions
res_narr = identify_narrative_sign(
    Y=Y,
    p=2,
    horizon=16,
    sign_matrix=sign_matrix,
    restrictions=restrictions,
    n_draws=3000,
    n_weight_sims=300,
    ci=0.68,
    seed=123,
)

# 5. Inspect Diagnostics and Summary
print(res_narr.summary())
print(f"Traditional Acceptance: {res_narr.traditional_acceptance_rate:.2%}")
print(f"Narrative Acceptance  : {res_narr.narrative_acceptance_rate:.2%}")
print(f"Kish ESS              : {res_narr.effective_draws:.1f}")

# 6. Generate Impulse Response Figures and Export Tables
fig = res_narr.plot(shock_idx=0)
md_table = res_narr.to_markdown()
tex_table = res_narr.to_latex()
```

---

## 4. Full API Specification

### `identify_narrative_sign`

```python
identify_narrative_sign(
    Y: np.ndarray | pd.DataFrame | VarEstimateResult,
    restrictions: list | None = None,
    *,
    p: int | None = None,
    horizon: int | None = 20,
    sign_matrix: dict | np.ndarray | None = None,
    dates: Sequence[Any] | None = None,
    bayes_draws: bool = False,
    n_draws: int = 2000,
    n_weight_sims: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> NarrativeSignResult
```

#### Parameters:
- `Y`: $(T, n)$ array-like time series panel or fitted `VarEstimateResult`.
- `restrictions`: List of narrative restrictions. Elements may be:
  - `NarrativeRestriction` instances.
  - Short-hand tuples `(date, shock_idx, sign)`.
  - `puremacro.narrative.NarrativeEvent` objects (automatically mapped to Type I shock 0).
- `p`: VAR lag order (inferred automatically if `Y` is a `VarEstimateResult`).
- `horizon`: Impulse response horizon $H$ (default `20`).
- `sign_matrix`: Traditional sign restrictions dict `{h: S}` with $S \in \{-1, 0, 1\}$.
- `dates`: Calendar date labels of length $T$ for row resolution when date strings or timestamps are provided.
- `bayes_draws`: When `True`, samples $(A, \Sigma)$ from conjugate Normal-Inverse-Wishart posterior for each draw.
- `n_draws`: Total candidate Haar rotation draws (default `2000`).
- `n_weight_sims`: Monte Carlo draws used to evaluate $\omega$ for non-Type I restrictions (default `500`).
- `ci`: Pointwise credible interval coverage level (default `0.90`).
- `seed`: Reproducibility seed for rotation generator and simulation sampler.

---

## 5. Result Interface & Downstream Capabilities

`NarrativeSignResult` inherits from `_IRFPlotMixin` and provides:

- `.plot(shock_idx=0, target_idx=None)`: Matplotlib multi-panel figure displaying median impulse response and shaded credible bands.
- `.summary()`: Comprehensive plain-text summary reporting acceptance metrics, effective sample sizes, and binding restriction counts.
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Clean publication-ready tables of structural responses.
- `.fevd(horizon=20)`: Forecast Error Variance Decomposition across structural shocks.
- `.historical_decomposition(variable=0, shock=0)`: Historical decomposition series of target variable explained by structural shocks over time.
