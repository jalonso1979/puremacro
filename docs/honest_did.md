> 🇬🇧 English · 🇪🇸 [Español](es/honest_did.md)

# Honest Difference-in-Differences Sensitivity Analysis

`puremacro.did.honest_did` implements the sensitivity analysis framework for staggered Difference-in-Differences (DiD) and event-study estimators developed by **Rambachan and Roth (2023, *Review of Economic Studies*)**.

In empirical policy evaluations, the core identification assumption is **parallel trends**: in the absence of treatment, average outcomes in treated and comparison groups would have evolved along parallel trajectories. Researchers conventionally test this assumption by checking whether pre-treatment coefficients in event studies are statistically indistinguishable from zero. However, Rambachan and Roth demonstrate that:
1. **Pre-testing lacks power**: Real violations of parallel trends frequently fail to be rejected.
2. **Pre-test bias**: Conditioning analysis on passing a pre-test introduces substantial statistical distortion.

The "Honest DiD" approach replaces binary pre-trend tests with **formal sensitivity analysis**, deriving robust confidence intervals and identified sets under bounded potential post-treatment violations of parallel trends.

---

## 1. Econometric Framework

Let $\hat{\beta} = (\hat{\beta}_{pre}', \hat{\beta}_{post}')'$ denote the vector of estimated event-study coefficients with asymptotic covariance matrix $\hat{\Sigma}$. Let $\delta = (\delta_{pre}', \delta_{post}')'$ represent the true underlying deviation from parallel trends, so that:

$$\hat{\beta} \sim \mathcal{N}(\theta + \delta, \Sigma)$$

where $\theta_{pre} = 0$ by definition, and $\theta_{post}$ represents the true causal treatment effect vector.

When parallel trends fails ($\delta_{post} \neq 0$), the treatment effect is only **partially identified**. Rambachan and Roth restrict the allowable set of post-treatment counterfactual violations $\delta \in \Delta$ using two canonical restrictions:

### 1.1 Smoothness / Bounded Second Differences ($\Delta^{SD}(M)$)

Imposes that the slope of the counterfactual trend cannot change too rapidly from one period to the next. The second differences of $\delta$ are bounded by $M \ge 0$:

$$\Delta^{SD}(M) = \left\{ \delta \in \mathbb{R}^T : \left| (\delta_{t+1} - \delta_t) - (\delta_t - \delta_{t-1}) \right| \le M, \quad \forall t \right\}$$

- $M = 0$: Enforces a strictly linear differential trend. Any linear pre-treatment trend is extrapolated forward into the post-treatment period.
- $M > 0$: Permits deviations from linear trends, where larger values of $M$ accommodate more volatile slope accelerations.

### 1.2 Relative Magnitudes ($\Delta^{RM}(\bar{M})$)

Bounds post-treatment violations by a multiple $\bar{M} \ge 0$ of the maximum observed pre-treatment deviation from parallel trends:

$$\Delta^{RM}(\bar{M}) = \left\{ \delta \in \mathbb{R}^T : |\delta_l| \le \bar{M} \cdot \max_{s < 0} |\delta_s|, \quad \forall l \ge 0 \right\}$$

Alternatively, in first differences:
$$|\delta_t - \delta_{t-1}| \le \bar{M} \cdot \max_{s \le 0} |\delta_s - \delta_{s-1}|$$

- $\bar{M} = 0$: Imposes exact parallel trends post-treatment ($\delta_{post} = 0$).
- $\bar{M} = 1$: Post-treatment violations can be no larger than the worst pre-treatment divergence observed in the historical baseline.

---

## 2. Optimization and Inference

For a target parameter $\theta = l' \tau_{post}$ (e.g., the effect at horizon $h$ or the cumulative effect):

1. **Identified Set Computation**:  
   The identified set $[\theta^{lo}(M), \theta^{hi}(M)]$ is solved via exact convex linear programming using the high-performance HiGHS interior-point / simplex solver (`scipy.optimize.linprog(method='highs')`).
2. **Robust Confidence Intervals**:  
   Constructed following **Imbens and Manski (2004)** and **Stoye (2009)**:
   $$CI_{1-\alpha}(M) = \left[ \hat{\theta}^{lo} - c_\alpha \cdot \text{se}(\hat{\theta}^{lo}), \; \hat{\theta}^{hi} + c_\alpha \cdot \text{se}(\hat{\theta}^{hi}) \right]$$
   where the critical value $c_\alpha$ satisfies $\Phi(c_\alpha + \frac{\hat{\theta}^{hi} - \hat{\theta}^{lo}}{\max(\text{se})}) - \Phi(-c_\alpha) = 1 - \alpha$.
3. **Breakdown Value $M^*$**:  
   The breakdown value $M^*$ is the smallest violation magnitude at which the robust $(1-\alpha)$ confidence interval first includes zero:
   $$M^* = \inf \{ M \ge 0 : 0 \in CI_{1-\alpha}(M) \}$$
   Solved to machine precision via Brent's method (`scipy.optimize.brentq`). If the baseline estimate is already statistically insignificant at $M=0$, $M^* = 0.0$.

---

## 3. Replication: Benzarti & Carloni (2019) French Restaurant VAT Cut

In July 2009, France reduced the value-added tax (VAT) on sit-down restaurants from 19.6% to 5.5%. Benzarti and Carloni (2019) investigate whether the incidence of this tax reduction benefited restaurant owners via increased firm profit margins.

Rambachan and Roth (2023, Section 6.1) analyze the sensitivity of the post-treatment profit surge using pre-treatment data from 2004 to 2007 (reference year 2008):
- **Baseline OLS (2009)**: Estimated profit response $\hat{\beta}_{2009} = 0.1960$ ($\text{se} = 0.0190$, $t > 10$).
- **Pre-treatment maximum deviation**: $\max_{s \le 2007} |\hat{\beta}_s| = 0.0730$.
- **Sensitivity under Relative Magnitudes ($\Delta^{RM}$)**: The breakdown value is $M^* \approx 2.06$. The tax cut's positive impact on firm profits remains statistically significant even if post-treatment parallel trends violations are more than **twice as large** as the largest pre-treatment divergence.

### Runnable Replication Code

```python
import numpy as np
from puremacro.did import honest_did

# 1. Benzarti & Carloni (2019) Event Study Coefficients & Covariance
years = [2004, 2005, 2006, 2007, 2009, 2010, 2011, 2012]
ref_year = 2008

beta_hat = np.array([
    0.006696, 0.029345, -0.006473, 0.073015,
    0.195961, 0.312064,  0.239542, 0.126043,
])

sigma_hat = np.array([
    [0.000843, 0.000477, 0.000262, 0.000235, 0.000168, 0.000113, 0.000020, -0.000137],
    [0.000477, 0.000643, 0.000399, 0.000244, 0.000220, 0.000180, 0.000038, -0.000030],
    [0.000262, 0.000399, 0.000523, 0.000212, 0.000184, 0.000146, 0.000070,  0.000060],
    [0.000235, 0.000244, 0.000212, 0.000309, 0.000120, 0.000133, 0.000102,  0.000108],
    [0.000168, 0.000220, 0.000184, 0.000120, 0.000361, 0.000295, 0.000163,  0.000085],
    [0.000113, 0.000180, 0.000146, 0.000133, 0.000295, 0.000472, 0.000248,  0.000142],
    [0.000020, 0.000038, 0.000070, 0.000102, 0.000163, 0.000248, 0.000412,  0.000221],
    [-0.000137,-0.000030, 0.000060, 0.000108, 0.000085, 0.000142, 0.000221,  0.000489],
])

# 2. Run Relative Magnitude Sensitivity for the Initial Post-Treatment Year (2009)
res_rm = honest_did(
    b_hat=beta_hat,
    sigma=sigma_hat,
    event_time=years,
    base_period=ref_year,
    target_horizon=2009,
    method="relative_magnitude",
    m_vec=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    alpha=0.05,
)

# 3. Print Sensitivity Summary & Breakdown Value
print(res_rm.summary())
print(f"Breakdown Value M*: {res_rm.breakdown_value:.4f}")

# 4. Generate Sensitivity Visualization
fig = res_rm.plot()

# 5. Export Output Tables
df_bounds = res_rm.to_frame()
latex_code = res_rm.to_latex()
```

---

## 4. Full API Specification

### `honest_did`

```python
honest_did(
    b_hat: Any = None,
    sigma: np.ndarray | Sequence[Sequence[float]] | None = None,
    se: Sequence[float] | None = None,
    method: str = "smoothness",
    m_vec: Sequence[float] | None = None,
    base_period: int = -1,
    alpha: float = 0.05,
    l_vec: Sequence[float] | np.ndarray | None = None,
    pre_periods: int | Sequence[int | float] | None = None,
    post_periods: int | Sequence[int | float] | None = None,
    *,
    result: Any = None,
    event_time: Sequence[int | float] | None = None,
    target_horizon: int | Sequence[int] | None = None,
    **kwargs: Any,
) -> HonestDiDResult
```

#### Parameters:
- `b_hat` / `result`: Vector of event-study coefficients, or a result object from `puremacro.did.callaway_santanna` or `puremacro.did.sun_abraham`.
- `sigma`: Full asymptotic covariance matrix $(T, T)$ of event study coefficients.
- `se`: Standard error vector (used if `sigma` is diagonal or unavailable).
- `method`: Sensitivity restriction class:
  - `'smoothness'`: Bounded second differences ($\Delta^{SD}(M)$).
  - `'relative_magnitude'`: Bounds proportional to maximum pre-trend ($\Delta^{RM}(\bar{M})$).
- `m_vec`: Grid of sensitivity violation values $M \ge 0$ to evaluate.
- `base_period`: Omitted reference event time normalized to zero (default `-1`).
- `alpha`: Significance level (default `0.05` for 95% confidence intervals).
- `target_horizon`: Specific event time horizon to evaluate (e.g., `0` or `2009`).
- `l_vec`: Custom contrast vector $l$ defining the linear combination $\theta = l' \tau_{post}$.

---

## 5. Result Interface

The `HonestDiDResult` container provides:

- `.table` / `.to_frame()`: `pd.DataFrame` containing the sensitivity curve:
  - `M`: Violation magnitude.
  - `id_lo`, `id_hi`: Lower and upper bounds of the identified set.
  - `ci_lo`, `ci_hi`: Imbens-Manski robust confidence intervals.
  - `significant`: Boolean indicator ($0 \notin [ci_{lo}, ci_{hi}]$).
- `.breakdown_value`: The exact critical threshold $M^*$.
- `.plot()`: Matplotlib plot illustrating the identified set and robust confidence bands as a function of $M$, marking the breakdown value $M^*$.
- `.plot_ascii()`: Terminal-friendly ASCII rendering of the sensitivity bounds.
- `.summary()`: Publication summary displaying method, pre-treatment baseline metrics, and breakdown values.
- `.to_markdown()`, `.to_latex()`, `.to_typst()`: Formatted tables for manuscripts.
