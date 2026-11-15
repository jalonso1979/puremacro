# Empirical Distributional LP ↔ One-Asset HANK Household Incidence Bridge

> 🇬🇧 English · 🇪🇸 [Español](es/hank_distributional_bridge.md)

The module `puremacro.models.hank_distributional_bridge` (`puremacro/models/hank_distributional_bridge.py`) connects cross-sectional household impulse responses from `puremacro`'s one-asset HANK solvers (`puremacro.models.hank_fiscal` and `puremacro.models.hank_sequence_space`) to empirical distributional and quantile local projections (`puremacro.lp.distributional`).

---

## 1. Purpose and Accounting Architecture

Evaluating heterogeneous-agent models solely on aggregate consumption $dC_t$ leaves household incidence and distributional transmission unverified. This bridge:

1. **Extracts cross-sectional group trajectories** ($T \times G$) across wealth quantiles, borrowing-constrained hand-to-mouth (HtM) vs. unconstrained households, productivity states, or custom $(a, s)$ state partitions under `balanced_budget`, `debt_rule`, `proportional`, `lump_sum`, monetary, or cash-on-hand transfer shocks.
2. **Audits exact accounting conservation (fail-closed)**: Verifies that population-weighted group IRFs reproduce aggregate model paths to `< 1e-10` without silent rescaling:
   $$\sum_{g=1}^{G} p_g \, dc_{t,g} = dC_t, \qquad \sum_{g=1}^{G} ds_{t,g}^C = 0, \qquad \sum_{g=1}^{G} p_g \, da_{t,g} = dA_t, \qquad \sum_{g=1}^{G} ds_{t,g}^A = 0.$$
   Unconverged models or conservation violations raise `ValueError` unless explicitly inspected via `allow_unconverged=True` or `strict_conservation=False`.
3. **Aligns & scores against empirical LPs**: Compares HANK group responses against `DistributionalLPResult` (from `estimate_distributional_lp` or `estimate_quantile_lp`) or user-supplied `EmpiricalGroupTargets` / `DataFrame` / mapping inputs, reporting:
   - Overall and per-group **RMSE**, **MAE**, **bias**, **sign concordance**, pointwise **$t$-statistics** / diagonal **Wald $\chi^2$**, and **confidence-band coverage**.
   - **Inequality-gradient & dispersion contrasts** comparing the low-wealth vs. high-wealth response gap (`low_group - high_group`, impact and cumulative ratios, gradient RMSE, and dispersion ratio).
   - **Steady-state calibration metadata** (`aggregate_mpc`, `htm_constrained_share`, `wealth_gini`, group MPCs, `top_grid_mass`) and a standardized `SolverDiagnosticRecord` (`PASS`, `CONVERGED_INACCURATE`, `UNSUPPORTED_REGIME`, `FAILED`).
4. **Bilingual reporting**: Supports `lang="en" | "es"` across `.summary()`, `.to_markdown()`, and `.to_dict()`, plus `.to_frame()`, `.to_latex()`, `.to_typst()`, and `.plot()`.

---

## 2. Partition Schemes & Dynamic Tracking Modes

| Partition `groups=` | Groups Generated |
| :--- | :--- |
| `"quintiles"`, `"deciles"`, `"quartiles"`, `"terciles"` | Equal steady-state liquid-wealth bins (`Q1..Q5`, `D1..D10`, etc.) |
| `"htm_vs_unconstrained"` | `"HtM"` vs. `"Unconstrained"` (`htm_definition="constrained"` at $a_{ss} \le a_{\min}$ or `"bottom_quartile"`) |
| `"income"`, `"wealth_x_income"` | Productivity states or wealth-by-productivity strata |
| Integer `G >= 2` or edge sequence `[0.0, ..., 1.0]` | Custom quantile partitions |
| `Mapping[str, np.ndarray]` | User-supplied partition weights over `(n_a,)` or `(n_a, n_s)` summing to 1 |

Dynamic tracking across $t = 0, \dots, T-1$ is controlled by `tracking`:
- `"cohort"` *(default)*: Tracks the initial steady-state group cohorts forward with exact population weights $p_g$, defined consistently across both nonlinear transitions (`FiscalHANKResult`, `NonlinearHANKResult`) and first-order Fake News linear perturbations (`SequenceSpaceHANKResult`).
- `"cross_section"`: Re-computes cross-sectional quantile boundaries each period along $D_t$ (nonlinear quantile partitions only).
- `"fixed_bins"`: Holds initial grid membership weights fixed across $D_t$ (bin population shares evolve with wealth mobility).

---

## 3. Copy-Pasteable Example

```python
import numpy as np
import pandas as pd

from puremacro.lp.distributional import estimate_distributional_lp
from puremacro.models.hank_distributional_bridge import (
    bridge_hank_distributional,
    solve_hank_group_irf,
)

# 1. Solve a financed HANK lump-sum tax cut (negative G shock under balanced budget)
T = 60
shock_seq = -0.02 * (0.7 ** np.arange(T))
hank_girf = solve_hank_group_irf(
    shock_seq=shock_seq,
    closure="balanced_budget",
    tax_type="lump_sum",
    horizon=T,
    groups="quintiles",
    n_a=100,
    a_max=100.0,
)
print(hank_girf.summary(lang="en"))

# 2. Construct an illustrative synthetic macro sample and estimate Distributional LPs
rng = np.random.default_rng(7)
n_obs, H_dgp = 300, 10
eps = rng.normal(0.0, 1.0, size=n_obs)
noise = rng.normal(0.0, 0.00025, size=(n_obs, 5))
noise -= noise.mean(axis=1, keepdims=True)
shares = np.empty((n_obs, 5), dtype=np.float64)
for t in range(n_obs):
    sig = np.zeros(5)
    for k in range(H_dgp):
        if t - k >= 0:
            sig += hank_girf.consumption_share_irf[k] * eps[t - k]
    shares[t] = hank_girf.steady_state_group_consumption_shares + sig + noise[t]

df_emp = pd.DataFrame(shares, columns=["Q1", "Q2", "Q3", "Q4", "Q5"])
df_emp["eps"] = eps
lp_res = estimate_distributional_lp(
    df_emp,
    shock="eps",
    bin_cols=["Q1", "Q2", "Q3", "Q4", "Q5"],
    horizons=range(0, 10),
    n_lags=1,
    distribution_type="shares",
)

# 3. Align & score the HANK model against empirical Distributional LP responses
bridge = bridge_hank_distributional(hank_girf, lp_res)
print(bridge.summary(lang="en"))
print(bridge.summary(lang="es"))
print(bridge.to_frame("per_group")[["rmse", "sign_concordance", "band_coverage", "steady_state_mpc"]])
```

---

## 4. Explicit Validation Scope & Methodological Limits

- **Stationary-distribution MIT / first-order perturbation scope**: Both `FiscalHANKResult` / `NonlinearHANKResult` (`deterministic_mit_transition`) and `SequenceSpaceHANKResult` (`linearized_sequence_space`) linearize or solve perfect-foresight deterministic aggregate paths around the stationary one-asset household distribution. They do **not** compute higher-order stochastic aggregate-risk equilibria with time-varying risk premia or state-contingent precautionary saving against future aggregate shocks.
- **One-asset liquidity margin**: Households hold a single liquid asset ($a \ge 0$). The bridge does not model two-asset illiquid-liquid portfolio rebalancing (wealthy hand-to-mouth dynamics driven by adjustment costs).
- **Scale compatibility with Quantile LPs**: Because `estimate_quantile_lp` (`distribution_type="quantiles"`) estimates conditional quantile shifts $dQ_\tau(y_{t+h} - y_{t-1})$ rather than cross-bin shares, `response_type="auto"` aligns it against HANK group consumption levels (`"level"`) and emits an explicit warning; pass `response_type="percent"` when empirical quantile outcomes are specified in log points.
- **Upper asset grid support**: Always maintain sufficient upper grid range (`a_max >= 100.0`). Tight upper bounds (`top_grid_mass > 1e-6`) distort high-wealth group consumption/saving and trigger `UNSUPPORTED_REGIME` in `solver_diagnostics`.
