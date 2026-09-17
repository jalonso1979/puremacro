> 🇬🇧 English · 🇪🇸 [Español](es/policy_simulators.md)

# Macroeconomic Policy Simulators & Interactive Browser Labs

`puremacro` delivers a unified, dual-delivery simulation architecture for quantitative macroeconomic policy counterfactuals:

1. **Production Python API (`puremacro.models`)**:
   - **`TradePolicySimulator`**: General equilibrium quantitative trade model based on **Caliendo and Parro (2015, *Review of Economic Studies*)**, solving for bilateral tariff counterfactuals, terms of trade, input-output linkages, and welfare decomposition with rigorous goods market clearing ($\max_i |X_i - Y_i| < 10^{-6}$).
   - **`MonetaryTransmissionSimulator`**: Side-by-side comparison of **Heterogeneous-Agent New Keynesian (HANK)** and **Representative-Agent New Keynesian (RANK)** models, featuring the **Kaplan, Moll, and Violante (2018, *American Economic Review*)** direct versus indirect transmission decomposition across Marginal Propensity to Consume (MPC) wealth deciles.
2. **Static WebAssembly / Browser-Native Interactive Labs (`curso/site/labs/`)**:
   - **`comercio-aranceles.html` / `.js`**: Real-time trade war counterfactual simulator (MEX-USA-CHN) with live bilateral tariff sliders, general equilibrium solver in client-side JavaScript, terms of trade, and trade diversion visualizers.
   - **`politica-monetaria-hank.html` / `.js`**: Interactive HANK versus RANK monetary transmission laboratory with live MPC decile ladders and Kaplan-Moll-Violante direct/indirect channel curves.
   - Designed for offline operation, zero-build deployment, and full compatibility with Pyodide and iPad tablets.

Both interfaces adhere to frozen dataclass specifications and include full export suites (`.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`).

---

## 1. Quantitative Trade Policy Simulator (`TradePolicySimulator`)

### 1.1 Theoretical Framework: Caliendo & Parro (2015) GE Model

Consider a global economy with $N$ countries ($n, i \in \{1, \dots, N\}$) and $J$ sectors ($j, k \in \{1, \dots, J\}$). Production combines labor $L_{n, j}$ with intermediate input bundles from all sectors under constant returns to scale.

Using **Exact Hat Algebra** (Dekle, Eaton, and Kortum 2007; Caliendo and Parro 2015), counterfactual changes in prices, wages, and expenditures are expressed in proportional terms $\hat{x} \equiv x' / x$, bypassing the need to estimate unobserved productivity levels or baseline iceberg trade costs.

Given counterfactual bilateral gross tariff changes $\hat{\tau}_{ni}^j = \frac{1 + t_{ni}'^j}{1 + t_{ni}^j}$:

1. **Unit Cost Changes**:
   $$\hat{c}_n^j = \hat{w}_n^{\gamma_n^j} \prod_{k=1}^J \left( \hat{P}_n^k \right)^{\gamma_n^{j, k}}$$
   where $\gamma_n^j > 0$ is the value-added share and $\gamma_n^{j, k} \ge 0$ are intermediate cost shares ($\gamma_n^j + \sum_k \gamma_n^{j, k} = 1$).
2. **Bilateral Trade Shares (Eaton-Kortum Gravity)**:
   $$\hat{\pi}_{ni}^j = \left( \frac{\hat{c}_i^j \hat{\tau}_{ni}^j}{\hat{P}_n^j} \right)^{-\theta_j}$$
   where $\theta_j > 0$ is the sectoral trade elasticity.
3. **Sectoral Price Indices**:
   $$\hat{P}_n^j = \left( \sum_{i=1}^N \pi_{ni}^j \left( \hat{c}_i^j \hat{\tau}_{ni}^j \right)^{-\theta_j} \right)^{-\frac{1}{\theta_j}}$$
4. **Goods & Factor Market Clearing**:
   $$\max_n \left| \sum_{j=1}^J \gamma_n^j Y_n'^j - w_n' L_n \right| < 10^{-6}$$
   Equilibrium wages $\hat{w}_n$ are solved iteratively via tatonnement until excess demand satisfies the tolerance threshold.

### 1.2 Welfare & Real Income Decomposition

Real income and welfare changes $\widehat{\mathcal{W}}_n = \frac{\hat{I}_n}{\hat{P}_n}$ (where $\hat{P}_n = \prod_j (\hat{P}_n^j)^{\alpha_n^j}$) are decomposed into three distinct economic mechanisms:
1. **Terms-of-Trade Effect**: Changes in export prices relative to import prices.
2. **Input-Output Efficiency**: Intermediate cost savings across global supply chains.
3. **Tariff Revenue Effect**: Changes in domestic tariff receipts redistributed to households.

### 1.3 Python Usage Example

```python
import numpy as np
from puremacro.models import TradePolicySimulator

# 1. Instantiate Trade Policy Simulator using the calibrated NAFTA-China 3-country preset
sim = TradePolicySimulator.from_preset("nafta_china")

# 2. Simulate a unilateral 20% tariff increase by the USA on Mexican manufactures
counterfactual = sim.simulate_tariff_counterfactual(
    tariff_shocks={("USA", "MEX", "Manufactures"): 0.20}
)

# Verify general equilibrium convergence
print(f"Converged: {counterfactual.converged} (Iterations: {counterfactual.iterations})")
print(f"Max Market Clearing Error: {counterfactual.market_clearing_residual:.2e} (< 1e-6)")

# Inspect country-level results
print(counterfactual.summary())

# Export welfare impacts to academic LaTeX
print(counterfactual.to_latex())
```

---

## 2. Monetary & Macroprudential Transmission Simulator (`MonetaryTransmissionSimulator`)

### 2.1 HANK vs. RANK Transmission Mechanisms

Standard Representative-Agent New Keynesian (RANK) models predict that monetary policy affects aggregate consumption almost entirely through the **direct intertemporal substitution channel**: higher interest rates induce households to postpone consumption along a single Euler equation.

In contrast, Heterogeneous-Agent New Keynesian (HANK) models (Kaplan, Moll, and Violante 2018; Auclert et al. 2021) incorporate uninsurable idiosyncratic earnings risk and borrowing constraints. A substantial share of households are **wealthy hand-to-mouth** or **poor hand-to-mouth**, exhibiting high Marginal Propensities to Consume (MPC). Consequently, the **indirect general equilibrium channel**—operating through labor demand, hours worked, and wage income—dominates aggregate dynamics.

### 2.2 Kaplan-Moll-Violante (2018) Sequence-Space Decomposition

Using the sequence-space Jacobian methodology (Auclert, Bardóczy, Rognlie, and Straub 2021), the aggregate consumption response $d\mathbf{C} \in \mathbb{R}^T$ over horizon $T$ is decomposed into:

$$d\mathbf{C} = \underbrace{\mathbf{J}^{C, r} \, d\mathbf{r}}_{\text{Direct Channel (Substitution)}} + \underbrace{\mathbf{J}^{C, Y} \, d\mathbf{Y}}_{\text{Indirect Channel (GE Income)}}$$

where $\mathbf{J}^{C, r} \equiv \frac{\partial \mathbf{C}}{\partial \mathbf{r}}$ and $\mathbf{J}^{C, Y} \equiv \frac{\partial \mathbf{C}}{\partial \mathbf{Y}}$ are $T \times T$ sequence-space Jacobian matrices.

| Feature | RANK Economy | HANK Economy |
|---|---|---|
| **Market Completeness** | Complete asset markets | Incomplete markets, borrowing constraints |
| **MPC Distribution** | Uniform: $1 - \beta \approx 0.015$ | Heterogeneous across deciles: $0.02$ to $0.45+$ |
| **Direct Channel Share** | $\approx 100\%$ | $20\% - 40\%$ |
| **Indirect Channel Share** | $\approx 0\%$ | $60\% - 80\%$ |
| **Transmission Driver** | Intertemporal Euler equation | General equilibrium labor income feedback |

### 2.3 Python Usage Example

```python
from puremacro.models import MonetaryTransmissionSimulator

# 1. Instantiate Monetary Transmission Simulator
sim_mon = MonetaryTransmissionSimulator(
    n_a=50,
    beta=0.985,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
)

# 2. Simulate a 25 bps contractionary interest rate hike with persistence rho=0.7 and horizon T=40
res = sim_mon.simulate_transmission(
    shock_type="rate",
    magnitude=0.0025,
    rho=0.7,
    T=40,
)

print(res.summary())
print(f"HANK Aggregate MPC : {res.aggregate_mpc_hank:.4f}")
print(f"RANK Aggregate MPC : {res.aggregate_mpc_rank:.4f}")
print(f"HANK Indirect GE Channel Share : {res.indirect_share_hank:.1f}%")
print(f"RANK Indirect GE Channel Share : {res.indirect_share_rank:.1f}%")

# Inspect MPC across wealth deciles
print(res.mpc_deciles_hank)
```

---

## 3. Interactive WebAssembly / Browser Labs (`curso/site/labs/`)

In addition to the Python API, `puremacro` provides two static interactive laboratories built in vanilla HTML5, CSS, and Canvas with zero build step, zero npm dependencies, and offline support.

### 3.1 `comercio-aranceles.html` & `.js` (Tariff War Simulator)

- **Location**: `curso/site/labs/comercio-aranceles.html` and `curso/site/labs/comercio-aranceles.js`.
- **Functionality**:
  * Simulates a tri-lateral trade war between Mexico (MEX), the United States (USA), and China (CHN).
  * Interactive sliders for bilateral tariffs: USA on MEX, USA on CHN, Mexican retaliation, Chinese retaliation.
  * Instantaneous client-side general equilibrium solution in JavaScript.
  * Live visualization of real wage impacts, terms of trade shifts, tariff revenue, and trade diversion flows.

### 3.2 `politica-monetaria-hank.html` & `.js` (HANK vs. RANK Laboratory)

- **Location**: `curso/site/labs/politica-monetaria-hank.html` and `curso/site/labs/politica-monetaria-hank.js`.
- **Functionality**:
  * Side-by-side impulse response curves for output, inflation, and consumption.
  * Interactive sliders for policy rate shocks, persistence, and the share of hand-to-mouth households.
  * Live bar charts displaying the MPC distribution across wealth deciles.
  * Dynamic visualization of the Kaplan-Moll-Violante direct versus indirect transmission channel shares.

Both labs can be served locally (`python -m http.server`) or embedded directly within Pyodide-powered educational notebooks.

---

## References

1. Auclert, A., Bardóczy, B., Rognlie, M., and Straub, L. (2021). "Using the sequence-space Jacobian to solve and estimate heterogeneous-agent models." *Econometrica*, 89(6), 3115–3148.
2. Caliendo, L. and Parro, F. (2015). "Estimates of the trade and welfare effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
3. Dekle, R., Eaton, J., and Kortum, S. (2007). "Unbalanced trade." *American Economic Review*, 97(2), 351–355.
4. Kaplan, G., Moll, B., and Violante, G. L. (2018). "Monetary policy according to HANK." *American Economic Review*, 108(3), 697–743.
