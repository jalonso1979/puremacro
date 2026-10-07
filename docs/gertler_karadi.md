> 🇬🇧 English · 🇪🇸 [Español](es/gertler_karadi.md)

# Gertler-Karadi (2011) DSGE with Financial Frictions

`puremacro.dsge.gertler_karadi` implements the canonical macro-finance DSGE model of banking frictions, balance-sheet amplification, and unconventional credit policy developed by **Gertler and Karadi (2011, *Journal of Monetary Economics*)**, with support for occasionally binding regimes via **OccBin (Guerrieri and Iacoviello 2015)**.

Following the 2007–2008 Global Financial Crisis, macroeconomic modeling recognized that disruptions in financial intermediation are not merely passive reflections of real downturns, but powerful primary sources of macroeconomic amplification. The Gertler-Karadi (GK 2011) framework models financial intermediaries (banks) that fund productive capital investments by issuing household deposits subject to an endogenous agency problem.

---

## 1. Theoretical Framework

### 1.1 Balance Sheets and the Agency Friction

Banks intermediate funds between household savers and non-financial firms. The aggregate bank balance sheet equates total asset holdings $S_t = Q_t K_t$ to bank net worth $N_t$ plus household deposits $B_t$:

$$Q_t K_t = N_t + B_t$$

Bankers maximize their expected terminal equity transferred to households upon retirement. To introduce financial frictions, GK (2011) introduce a **moral hazard agency problem**:
- At the end of each period, a banker can divert a fraction $\lambda_b \in (0, 1)$ of bank assets $Q_t K_t$ back to their household.
- If a banker diverts assets, the bank defaults, depositors seize the remaining fraction $(1 - \lambda_b)$, and the bank is closed.

For depositors to voluntarily provide funds, the bank's continuation value $V_t$ must satisfy the **incentive compatibility constraint**:

$$V_t \ge \lambda_b Q_t K_t$$

The continuation value can be decomposed as:

$$V_t = \nu_t Q_t K_t + \eta_t N_t$$

where $\nu_t$ represents the marginal value of expanding assets (holding net worth fixed) and $\eta_t$ represents the marginal value of expanding net worth (holding assets fixed).

When the incentive constraint binds, it establishes an **endogenous bank leverage ratio** $\phi_t$:

$$\phi_t \equiv \frac{Q_t K_t}{N_t} = \frac{\eta_t}{\lambda_b - \nu_t}$$

This constraint endogenously pins down the **credit spread** (or external finance premium):

$$\text{Spread}_t \equiv \mathbb{E}_t \left[ R_{k, t+1} - R_{t+1} \right]$$

### 1.2 Macroeconomic Amplification & Capital Quality Shocks

The primary driving experiment in Gertler and Karadi (2011) is a **capital quality shock** $\xi_t$.

The effective capital stock evolves according to $K_{t+1} = \xi_{t+1} [(1 - \delta) K_t + I_t]$. A negative innovation $\varepsilon_\xi < 0$:
1. Directly reduces the productive capability of physical capital.
2. Depresses the asset price $Q_t$.
3. Induces massive capital losses on intermediary balance sheets. Because banks are leveraged at $\phi \approx 4$, a 1% decline in asset value contracts bank net worth by approximately **4%**:
   $$\frac{d N_t}{N_{ss}} \approx \phi \cdot \frac{d Q_t}{Q_{ss}}$$
4. Forces banks to shed assets or spike credit spreads $\mathbb{E}_t[R_{k,t+1} - R_{t+1}]$ by hundreds of basis points to satisfy depositor incentive compatibility.
5. Induces a severe, persistent collapse in private investment and aggregate GDP.

---

## 2. Dual Solution Solvers

`puremacro.dsge.gertler_karadi` provides dual solution engines:

1. **Klein (1998) Generalized Schur (QZ) Linear Solver (`method='klein'`)**:  
   Computes exact first-order rational expectations perturbations around the deterministic steady state using `puremacro.dsge.klein`. Fast and ideal for standard linear impulse response analysis.
2. **OccBin Piecewise-Linear Backward Recursion (`method='occbin'`)**:  
   Solves dynamic models with occasionally binding regime switches per Guerrieri and Iacoviello (2015):
   - **Unconventional Credit Policy (`constraint_type='credit_policy'`)**: The central bank directly intermediates credit when private credit spreads spike above a target threshold (e.g., 100 bps):
     $$\psi_t = \nu_g \cdot \max \left( 0, \, \text{Spread}_t - \text{threshold} \right)$$
   - **Macroprudential Leverage Caps (`constraint_type='leverage_cap'`)**: Imposes a regulatory ceiling on bank leverage, $\phi_t \le \phi_{max}$, **with a public backstop**. While the cap binds, the constrained regime pegs $\phi_t = \phi_{max}$ and frees the public credit share, so the balance sheet $(1 - \psi_t) Q_t K_t = \phi_t N_t$ determines how much intermediation the public sector absorbs, under complementary slackness:
     $$\phi_t \le \phi_{max}, \qquad \psi_t \ge 0, \qquad \psi_t = 0 \ \text{where the cap is slack}.$$
     The moral-hazard incentive constraint is *kept*, so a binding cap compresses intermediation rents rather than rationing private credit. Enforcing the cap by deleting the incentive constraint instead — the pre-2.5.0 formulation — makes the constrained regime violate Blanchard-Kahn and admits a continuum of spurious solutions; see the `solve_gertler_karadi` docstring.

---

## 3. Canonical Calibration (GK 2011 Table 1)

| Parameter | Value | Economic Description |
|---|---|---|
| `beta` | $0.99$ | Quarterly subjective discount factor ($R_{ss} \approx 4.0\%$ annualized) |
| `sigma` | $1.0$ | Intertemporal elasticity of substitution (log utility) |
| `h` | $0.815$ | Consumption habit persistence |
| `varphi` | $0.276$ | Inverse Frisch labor supply elasticity |
| `alpha` | $0.33$ | Capital share in production |
| `delta` | $0.025$ | Physical quarterly depreciation rate (10% annualized) |
| `eta_i` | $1.728$ | Investment adjustment cost curvature |
| `theta_b` | $0.972$ | Banker survival probability (average tenure $\approx 9$ years) |
| `lambda_b`| $0.381$ | Divertable fraction of assets (moral hazard intensity) |
| `omega_b` | $0.002$ | Start-up wealth endowment for entering bankers |
| `gamma` | $0.779$ | Calvo price stickiness probability |
| `rho_xi` | $0.66$ | Capital quality shock autoregressive persistence |

---

## 4. Basic & Advanced Usage

### Simulating a Capital Quality Shock under Klein and OccBin

```python
import numpy as np
from puremacro.dsge.gertler_karadi import (
    solve_gertler_karadi,
    solve_steady_state,
)

# 1. Inspect Deterministic Steady State
ss = solve_steady_state()
print(f"Steady-state bank leverage phi : {ss['phi']:.2f}x")
print(f"Annualized credit spread (bps) : {ss['spread_ann']:.1f}")

# 2. Simulate Capital Quality Shock (-5% shock to xi) under Klein linear solver
res_klein = solve_gertler_karadi(
    shock_type="capital_quality",
    shock_size=-0.05,
    horizon=40,
    method="klein",
)

# 3. Simulate under OccBin with Central Bank Credit Policy Intervention
res_occbin = solve_gertler_karadi(
    shock_type="capital_quality",
    shock_size=-0.05,
    horizon=40,
    method="occbin",
    constraint_type="credit_policy",
    threshold=0.0025,  # Intervene if spread exceeds 100 bps
)

# 4. Compare Responses. to_frame() holds level deviations x_t - x_ss (the model
#    is linearised in levels); to_frame(units="pct") gives 100 * (x_t - x_ss) / x_ss.
df_klein = res_klein.to_frame()
df_occbin = res_occbin.to_frame()
pct_klein = res_klein.to_frame(units="pct")
pct_occbin = res_occbin.to_frame(units="pct")

print("Peak Net Worth Contraction (Klein, %) :", pct_klein["N"].min())
print("Peak Net Worth Contraction (OccBin, %):", pct_occbin["N"].min())
print("Peak Spread Surge (Klein, bps)        :", df_klein["prem"].max() * 40000)
print("Peak Spread Surge (OccBin, bps)       :", df_occbin["prem"].max() * 40000)

# 5. Diagnostic Summary & Visualizations
print(res_occbin.summary())
fig = res_occbin.plot(units="pct")
```

Bank net worth falls 61.7% on impact under Klein and 49.2% under OccBin, where the credit policy absorbs part of the loss; the annualised spread rises by up to about 604 bps and 256 bps above its steady state of 100 bps. Read the percentages from `to_frame(units="pct")`: in levels net worth falls by 0.852 and 0.679 against $N_{ss} = 1.381$, and multiplying a level deviation by 100 gives a percentage only for variables whose steady state is 1.

---

## 5. Full API Specification

### `solve_gertler_karadi`

```text
solve_gertler_karadi(
    params: Mapping[str, float] | None = None,
    shock_type: str = "capital_quality",
    shock_size: float = -0.05,
    horizon: int = 40,
    method: str = "occbin",
    constraint_type: str = "credit_policy",
    threshold: float | None = None,
    max_iter: int = 50,
) -> GertlerKaradiResult
```

#### Parameters:
- `params`: Optional overrides for canonical calibration `GK2011_PARAMS`.
- `shock_type`: Innovated shock: `'capital_quality'` ($\varepsilon_\xi$), `'tfp'` ($\varepsilon_a$), or `'monetary'` ($\varepsilon_r$).
- `shock_size`: Initial innovation magnitude at $t=0$ (default `-0.05` for -5% capital quality shock).
- `horizon`: Simulation horizon in quarters (default `40`).
- `method`: Solution engine: `'occbin'` (piecewise-linear) or `'klein'` (linear QZ).
- `constraint_type`: For OccBin: `'credit_policy'` or `'leverage_cap'`.
- `threshold`: Activation threshold for the regime switch, as a deviation from steady state. Default `0.0025` (100 bps spread) for `credit_policy`; `0.20 * phi_ss` (= `0.8197` at this calibration) for `leverage_cap`. A tighter leverage cap binds for longer and needs a longer `horizon` (10% above $\phi_{ss}$ needs `horizon >= 80`, 5% needs `>= 200`); the solver reports `converged=False` rather than guessing.
- `max_iter`: Maximum backward iterations for OccBin convergence.

---

## 6. Result Interface

`GertlerKaradiResult` provides structured model simulations and outputs:

- **Attributes**:
  - `irf`: `pandas.DataFrame` of impulse responses, one column per model variable and one row per quarter (row 0 is the impact quarter). The entries are **level deviations from the steady state**, $x_t - x_{ss}$, in model units: the equilibrium conditions are linearised in levels, so they are neither percent nor log deviations. `res["N"]` returns one column.
  - `variables`: List of the model's variable names, the columns of `irf`: `['Y', 'C', 'I', 'K', 'L', 'w', 'rho_c', 'Q', 'R', 'Rk', 'prem', 'N', 'Ne', 'Nn', 'phi', 'nu', 'eta', 'Omega', 'Pm', 'Z', 'U', 'Pi', 'Rn', 'xi', 'a', 'psi']`.
  - `steady_state`: Dictionary of calculated steady-state values, keyed by the same names (plus calibration targets such as `spread_ann`).
  - `regimes`: for OccBin, the regime indicator per quarter (`0` = reference, `1` = constrained); `binding_periods` counts the constrained quarters.
  - `converged`: `False` when the OccBin regime iteration hit `max_iter` (a `RuntimeWarning` is emitted and `summary()` flags it).
- **Units of a level deviation**:
  - `Y`, `C`, `I`, `K`, `N`, `Ne`, `Nn`, `L`, `w`: model units ($L_{ss} = 1/3$).
  - `R`, `Rk`, `Rn`: gross *quarterly* returns, so the deviation is the change in the net quarterly rate as a fraction (0.0025 = 25 bps per quarter, 100 bps annualised). `Pi` is gross quarterly inflation.
  - `prem`: the expected quarterly spread $\mathbb{E}_t[R_{k,t+1} - R_{t+1}]$ as a fraction; `40000 * prem` is annualised basis points.
  - `phi`: private bank leverage $(1 - \psi_t) Q_t K_t / N_t$, a ratio; `psi`: the public share of intermediated assets, whose steady state is 0.
  - `100 * (x_t - x_ss)` is a percentage only for variables whose steady state is 1 (`Q`, `U`, `xi`, `a`, `Pi`). At the default solve capital falls by 0.2866 in level on impact against $K_{ss} = 5.6601$: that is −5.06%, not −28.66%.
- **Methods**:
  - `to_frame(units="level")`: Returns a `pandas.DataFrame` indexed by simulation quarters $t = 0, \dots, T-1$. `units="level"` (the default) returns `irf` unchanged; `units="pct"` returns percent deviations $100\,(x_t - x_{ss})/x_{ss}$. `psi`, whose steady state is 0, has no percent deviation: it is NaN and listed in `df.attrs["pct_undefined"]`. `df.attrs["units"]` records the choice.
  - `.plot(variables=None, style="publication", figsize=None, units="level")`: Matplotlib multi-panel figure; by default Output, Investment, Bank Net Worth, Credit Spread, Policy Rate `Rn` and Leverage $\phi$, with the constrained OccBin quarters shaded. `units` is passed to `to_frame`, and the y-axis label states it.
  - `.summary()`: Plain-text report of the steady state, the shock and the impact, minimum, maximum, mean and final values of the key variables, in level deviations (the header says so and points to `units="pct"`).
  - `.to_markdown(units="level")`, `.to_latex(units="level")`, `.to_typst(units="level")`: Formatted tables for academic manuscripts; `units` is passed to `to_frame`.
