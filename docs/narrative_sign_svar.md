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
   *Example*: "The monetary policy shock in 1979Q4 (Volcker monetary tightening) was strictly positive." The sign is a required field of a Type I restriction (`(date, shock, sign)` tuples are shorthand).

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
   *Example*: "The monetary shock during the Volcker regime change was at least 2 standard deviations in magnitude ($|\varepsilon_{j, t^*}| \ge 2.0$)." The bound is **unsigned by default** (`sign=None`); pass `sign=+1` or `sign=-1` to additionally restrict the sign of the shock.

### 1.3 Restriction Dates

`date` may be an **integer** or a **calendar date**:

- An integer is always the 0-based **row index into `Y`** (it must be $\ge p$ so that a residual exists), whether or not `dates` is supplied.
- A `pd.Timestamp`, ISO string (`"1979-10-06"`), `datetime.date` or `np.datetime64` is located in `dates`. A `DataFrame` indexed by a `DatetimeIndex` or `PeriodIndex` supplies `dates` automatically (an explicit `dates=` overrides it). An exact stamp match wins; otherwise the date is matched to the observation whose **period contains it**, where the period length is inferred from the spacing of the index — same quarter on a quarterly index (so the announcement date 1979-10-06 hits 1979Q4 whether the index is stamped 1979-10-01 or 1979-12-31), same month on a monthly index, exact match on anything finer. A date whose period is absent from the index raises `ValueError` rather than being remapped to a neighbouring period.

### 1.4 Sampling and Importance Weighting (AD-RR Algorithm 1)

1. **Haar Rotation Draws**: Draw random Gaussian matrices $Z \sim \mathcal{N}(0, I_n)$ and compute $Q$ via QR decomposition with positive diagonal normalization ($R_{ii} > 0$). With `bayes_draws=True`, a fresh $(A, c, \Sigma)$ is drawn from the conjugate Normal-Inverse-Wishart posterior for every rotation draw (redrawn until the VAR is stable, up to 50 attempts; a draw with no stable candidate is skipped and counted in `n_unstable_draws`).
2. **Traditional Sign Check**: Check if the resulting structural impulse responses $\Psi_h = \Phi_h P Q$ satisfy traditional sign restrictions specified in `sign_matrix`.
3. **Narrative Validation**: Compute the structural shock series $\varepsilon = u (B^{-1})'$ and evaluate each narrative restriction on its designated date(s).
4. **Importance Weighting**: To prevent conditioning from biasing the posterior toward parameter values for which the narrative events are unsurprising, surviving draws are weighted by:
   $$w = \frac{1}{\omega}$$
   where $\omega$ is the probability that the narrative restrictions hold when shocks on the restricted dates are redrawn i.i.d. standard normal $\mathcal{N}(0, I_n)$:
   - When all narrative restrictions are pure Type I (`'shock_sign'`) on $m$ distinct (date, shock) pairs, $\omega = 0.5^m$ is known in exact closed form ($w = 2^m$).
   - For Type II, III, or IV restrictions, $\omega$ is estimated via Monte Carlo simulation with $S = \text{n\_weight\_sims}$ draws. An estimate $\hat\omega = 0$ is floored at $1/S$ (the weight is capped at $S$); the number of draws on which the floor binds is reported as `n_weight_floor`.
5. **Weighted Inference**: Credible interval bands and medians are computed as pointwise weighted quantiles across surviving draws. The impact matrix of every surviving draw (and, in Bayesian mode, its autoregressive coefficients) is kept on the result, so `.irf(h)` and `.fevd(h)` for horizons beyond the estimated $H$ are weighted medians of the *extended* draws — never a single representative draw.

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
| `ess` / `effective_draws` | `float` | Kish Effective Sample Size: $(\sum w_i)^2 / \sum w_i^2$. |
| `n_weight_floor` | `int` | Surviving draws whose $\hat\omega$ was 0 and was floored at $1/\text{n\_weight\_sims}$ (their weights are capped, so `ess` overstates efficiency). |
| `n_unstable_draws` | `int` | Bayesian mode only: posterior draws skipped because no stable VAR was found in 50 attempts. |
| `restriction_labels` | `tuple[str]` | Human-readable string representation of each restriction. |
| `restriction_fail_counts` | `tuple[int]` | Binding-ness diagnostic: count of traditionally accepted draws rejected by each restriction. |
| `accepted_B`, `accepted_A` | `np.ndarray` | Impact matrices $(m, n, n)$ of the $m$ surviving draws and, in Bayesian mode, their autoregressive matrices $(m, p, n, n)$ (`None` in OLS mode). |
| `A_list`, `intercept`, `residuals`, `Sigma`, `B` | — | Reduced-form objects and impact matrix of the representative (median-target) draw; `B B' = Sigma` in both modes. |
| `init_y` | `np.ndarray` | The first $p$ observations, the default pre-sample condition of the historical decomposition. |

### Warnings

`identify_narrative_sign` emits a `RuntimeWarning` (never silently degrades) when:

- **too few draws survive** to resolve the requested bands — fewer than $\max(10, \lceil 2/(1-\text{ci}) \rceil)$ accepted draws, in which case the pointwise bands may collapse onto the median;
- **the importance weights are concentrated** — Kish `ess` below 10% of `n_narrative_accepted`, so a handful of draws drive the weighted bands;
- **the $\omega$ floor binds** for at least one surviving draw (`n_weight_floor > 0`; increase `n_weight_sims`);
- **unstable posterior draws were skipped** in Bayesian mode (`n_unstable_draws > 0`). If *every* posterior draw is unstable, a `RuntimeError` is raised instead.

`summary()` prints the reduced-form mode, the unstable-draw count and the floor count alongside the acceptance counts.

---

## 3. Application: The Volcker 1979Q4 Monetary Policy Shock

In October 1979, Federal Reserve Chairman Paul Volcker announced a dramatic shift in monetary policy operating procedures, targeting non-borrowed reserves and causing an unprecedented spike in the federal funds rate to break runaway inflation.

Antolín-Díaz and Rubio-Ramírez (2018) revisit Uhlig's (2005) sign-identified monetary SVAR — a six-variable **monthly** US VAR on Uhlig's dataset — and sharpen it with narrative restrictions on the Volcker episode, chiefly:

- **Narrative Restriction 1 (Type I)**: the monetary policy shock in October 1979 is positive (contractionary).
- **Narrative Restriction 2 (Type III)**: the monetary shock is the overwhelming contributor to the unexpected movement in the federal funds rate in October 1979.

Their headline finding is that these restrictions substantially tighten the sign-identified set and remove the counter-intuitive responses that traditional sign restrictions alone admit.

The runnable example below applies the **same two Volcker restrictions** to a small **synthetic** 3-variable quarterly VAR (FFR, inflation, output growth) with a +3.5 s.d. shock planted in 1979Q4. It illustrates the mechanism and the API; it does **not** reproduce the paper's numbers (Uhlig's dataset is not fetched offline, and AD-RR report their IRFs graphically). `puremacro.examples.narrative_sign_adrr.run_demo()` is a fuller version of the same synthetic exercise, and `tests/test_replication_adrr2018.py` pins a regression snapshot of *that demo's own output* — it is a regression test, not a check against published values.

### Runnable Example

```python
import numpy as np
import pandas as pd
from puremacro.var.identify import (
    NarrativeRestriction,
    identify_narrative_sign,
)

# 1. Synthetic quarterly panel (FFR, Inflation, Output growth), 1965Q1-2007Q4
rng = np.random.default_rng(42)
T = 172
dates = pd.date_range("1965-01-01", periods=T, freq="QS")  # quarter-start stamps
volcker_idx = dates.get_loc("1979-10-01")                   # row of 1979Q4

# Calibrated stationary VAR data
Y = np.zeros((T, 3))
for t in range(1, T):
    Y[t] = 0.5 * Y[t-1] + rng.standard_normal(3)
# Plant the Volcker tightening in 1979Q4
Y[volcker_idx, 0] += 3.5

# 2. Traditional sign restrictions at horizon 0
# Shock 0 = monetary policy shock (+FFR, -Inflation, -Output growth)
sign_matrix = {0: np.array([+1, -1, -1])}

# 3. Narrative restrictions. Calendar dates are resolved through `dates`
#    (the FOMC announcement of 1979-10-06 maps to the 1979Q4 observation);
#    an integer such as `volcker_idx` would be read as a row index instead.
restrictions = [
    # Type I: the monetary shock in 1979Q4 was positive (contractionary)
    ("1979-10-06", 0, +1),
    # Type III: overwhelming contributor to the unexpected FFR change
    NarrativeRestriction(
        kind="hd_dominance",
        date="1979-10-06",
        shock=0,
        variable=0,  # FFR
        window=0,
        dominance="overwhelming",
    ),
]

# 4. Estimate the SVAR with narrative restrictions
res_narr = identify_narrative_sign(
    Y,
    restrictions,
    p=2,
    horizon=16,
    sign_matrix=sign_matrix,
    dates=dates,
    n_draws=3000,
    n_weight_sims=300,
    ci=0.68,
    seed=123,
)

# 5. Diagnostics and summary
print(res_narr.summary())
print(f"Traditional acceptance: {res_narr.traditional_acceptance_rate:.2%}")
print(f"Narrative acceptance  : {res_narr.narrative_acceptance_rate:.2%}")
print(f"Kish ESS              : {res_narr.effective_draws:.1f}")

# 6. Figures, tables and downstream objects
fig = res_narr.plot(shock_idx=0, target_idx=None)   # one panel per variable
md_table = res_narr.to_markdown(target_idx=0, shock_idx=0)
tex_table = res_narr.to_latex(target_idx=0, shock_idx=0)
fevd_20 = res_narr.fevd(horizon=20)                 # weighted median beyond H=16
hd_ffr = res_narr.historical_decomposition(variable=0)
```

---

## 4. Full API Specification

### `identify_narrative_sign` (alias `narrative_sign_svar`)

```text
identify_narrative_sign(
    Y: np.ndarray | pd.DataFrame | VarEstimateResult,
    restrictions: list | None = None,
    *,
    p: int | None = None,
    lags: int | None = None,          # alias for p
    horizon: int | None = None,       # defaults to 20
    horizons: int | None = None,      # alias for horizon
    sign_matrix: dict | np.ndarray | None = None,
    dates: Sequence[datetime-like] | None = None,
    bayes_draws: bool = False,
    n_draws: int = 2000,
    n_weight_sims: int = 500,
    ci: float = 0.9,
    seed: int | None = 0,
) -> NarrativeSignResult
```

#### Parameters:
- `Y`: $(T, n)$ array-like time series panel or fitted `VarEstimateResult`. A `DataFrame` with a `DatetimeIndex`/`PeriodIndex` supplies `dates` automatically and its column names become `result.names`.
- `restrictions`: List of narrative restrictions. Elements may be:
  - `NarrativeRestriction` instances.
  - Short-hand tuples `(date, shock_idx, sign)` (Type I).
  - `puremacro.narrative.NarrativeEvent` objects (automatically mapped to Type I on shock 0, using the announcement date and sign).
- `p` / `lags`: VAR lag order (inferred automatically if `Y` is a `VarEstimateResult`; otherwise required). Passing both with different values raises `ValueError`.
- `horizon` / `horizons`: Impulse response horizon $H$ (default `20`). Passing both with different values raises `ValueError`.
- `sign_matrix`: Traditional sign restrictions dict `{h: S}` with $S \in \{-1, 0, 1\}$ of shape $(n, n)$, or $(n,)$ applied to shock column 0; a bare array means `{0: S}`. `None` (the default) imposes no traditional sign restrictions.
- `dates`: Calendar date labels of length $T$ for resolving calendar restriction dates (see §1.3). Overrides a DataFrame index when given.
- `bayes_draws`: When `True`, samples $(A, \Sigma)$ from the conjugate Normal-Inverse-Wishart posterior for each draw (see §1.4).
- `n_draws`: Total candidate Haar rotation draws (default `2000`, must be $\ge 1$).
- `n_weight_sims`: Monte Carlo draws used to evaluate $\omega$ for non-Type I restrictions (default `500`, must be $\ge 1$).
- `ci`: Pointwise credible interval coverage level (default `0.90`); must lie strictly between 0 and 1.
- `seed`: Reproducibility seed for the rotation generator and the weight simulator (default `0`; `None` draws fresh entropy).

Unknown keyword arguments raise `TypeError` — a typo such as `n_draw=` never runs silently with the default.

---

## 5. Result Interface & Downstream Capabilities

`NarrativeSignResult` inherits from `_IRFPlotMixin` and provides:

- `.plot(*, target_idx=0, shock_idx=0, title="", ylabel="Response", scale=1.0, ax=None)`: with integer indices, a single panel (median response and shaded credible band) for that (variable, shock) pair, optionally drawn on `ax`. `target_idx=None` draws one panel per response variable for the given shock; `shock_idx=None` one panel per shock; both `None` the full $n \times n$ grid. Returns the matplotlib `Figure`.
- `.summary()`: Plain-text summary reporting the reduced-form mode, acceptance metrics, effective sample size, floor/unstable counts, and binding restriction counts.
- `.to_frame()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`: Tidy tables of the median response and bands (`target_idx`/`shock_idx` select the pair).
- `.irf(horizon=None)`: Weighted-median impulse responses of shape $(h+1, n, n)$. For $h \le H$ a slice of `irf_median`; for $h > H$ the weighted median of every accepted draw's IRF extended to $h$ (the first $H+1$ rows coincide with `irf_median`).
- `.fevd(horizon=None)`: Weighted-median forecast error variance decomposition, rows renormalised to sum to 1; extended beyond $H$ in the same way.
- `.historical_decomposition(variable=None, shock=None, init_y=None)`: Historical decomposition using the representative draw's $B$ together with that draw's own reduced-form objects (OLS in OLS mode; the posterior draw's own $(A, c)$ and residuals in Bayesian mode). `init_y` defaults to the stored first $p$ observations, so `deterministic + shocks.sum(axis=2)` reproduces $y_t$ exactly for $t \ge p$. Returns a dict (`'shocks'` of shape $(T-p, n, n)$, `'deterministic'` of shape $(T-p, n)$) or a tidy `DataFrame` when `variable` and/or `shock` are given.
