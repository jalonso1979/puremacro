> 🇬🇧 English · 🇪🇸 [Español](es/spatial_and_trade_ge.md)

# Quantitative Spatial Economics & Gravity Trade GE

`puremacro.trade` and `puremacro.spatial` implement state-of-the-art general equilibrium frameworks for international trade and economic geography:
1. The multi-country, multi-sector Ricardian trade model with input-output linkages, intermediate goods, and tariffs developed by **Caliendo and Parro (2015, *Review of Economic Studies*)**, solved via **Exact Hat Algebra**.
2. The continuous geographic spatial equilibrium model with labor mobility, agglomeration externalities, and congestion forces developed by **Allen and Arkolakis (2014, *Quarterly Journal of Economics*)**.

Both models are symmetrically exposed across `puremacro.trade` and `puremacro.spatial`. They conform strictly to the pure Python / Pyodide runtime contract (zero C++ dependencies, pure NumPy/SciPy/Pandas/Matplotlib), enabling rapid counterfactual policy evaluations—such as trade wars, retaliatory tariff escalations, regional transport infrastructure investments, and localized environmental shocks.

---

## 1. Theoretical & Algorithmic Framework

### 1.1 Caliendo-Parro (2015) Multi-Sector Ricardian Trade with I-O Linkages

Consider a global economy with $N$ countries ($n, i = 1, \dots, N$) and $J$ sectors ($j, k = 1, \dots, J$). Production in country $n$ and sector $j$ combines labor $L_{n, j}$ with intermediate inputs from all sectors $k = 1, \dots, J$ under constant returns to scale:

$$Y_n^j = z_n^j \left( L_n^j \right)^{\gamma_n^j} \prod_{k=1}^J \left( M_n^{j, k} \right)^{\gamma_n^{j, k}}$$

where $\gamma_n^j > 0$ is the value-added share, $\gamma_n^{j, k} \ge 0$ is the cost share of intermediate input $k$ in producing good $j$, and cost shares sum to unity:

$$\gamma_n^j + \sum_{k=1}^J \gamma_n^{j, k} = 1, \quad \forall n, j$$

Cost minimization implies unit cost of the input bundle:

$$c_n^j = \Upsilon_n^j w_n^{\gamma_n^j} \prod_{k=1}^J \left( P_n^k \right)^{\gamma_n^{j, k}}$$

where $w_n$ is the competitive wage in country $n$, $P_n^k$ is the composite price index of sector $k$, and $\Upsilon_n^j$ is a constant.

### 1.2 Exact Hat Algebra Formulation

Following the **Exact Hat Algebra** methodology of Dekle, Eaton, and Kortum (2007) extended by Caliendo and Parro (2015), counterfactual changes are expressed in proportional changes $\hat{x} \equiv x' / x$. This allows researchers to solve for counterfactual equilibria directly from observed baseline trade shares $\pi_{ni}^j$ and input-output matrices without estimating unobserved fundamentals (such as technology levels $T_n^j$ or bilateral baseline iceberg frictions $d_{ni}^j$).

Given a counterfactual gross tariff change $\hat{\tau}_{ni}^j \equiv (1 + t_{ni}'^j) / (1 + t_{ni}^j)$:

1. **Unit Cost Changes**:
   $$\hat{c}_n^j = \hat{w}_n^{\gamma_n^j} \prod_{k=1}^J \left( \hat{P}_n^k \right)^{\gamma_n^{j, k}}$$
2. **Bilateral Trade Share Changes (Eaton-Kortum Gravity)**:
   $$\hat{\pi}_{ni}^j = \left( \frac{\hat{c}_i^j \hat{\tau}_{ni}^j}{\hat{P}_n^j} \right)^{-\theta_j}$$
   where $\theta_j > 0$ is the sector-specific trade elasticity.
3. **Sectoral Price Index Changes**:
   $$\hat{P}_n^j = \left( \sum_{i=1}^N \pi_{ni}^j \left( \hat{c}_i^j \hat{\tau}_{ni}^j \right)^{-\theta_j} \right)^{-\frac{1}{\theta_j}}$$
4. **Expenditures & Goods Market Clearing**:
   Counterfactual sectoral expenditures $X_n'^j$ and gross output $Y_n'^j$ solve the global Leontief input-output balance:
   $$X_n'^j = \alpha_n^j I_n' + \sum_{k=1}^J \gamma_n^{k, j} Y_n'^k, \quad Y_n'^j = \sum_{i=1}^N \frac{\pi_{in}'^j}{\tau_{in}'^j} X_i'^j$$
   where $I_n' = w_n' L_n + R_n' + D_n'$ is national income comprising labor income, tariff revenues $R_n'$, and trade deficits $D_n'$.
5. **Labor Market Clearing & Wage Tatonnement**:
   Wages $\hat{w}_n$ are updated iteratively until excess labor demand satisfies:
   $$\left| \sum_{j=1}^J \gamma_n^j Y_n'^j - w_n' L_n \right| < \text{tol}$$
6. **Real Income & Welfare Changes**:
   $$\widehat{\mathcal{W}}_n = \frac{\hat{I}_n}{\hat{P}_n}, \quad \text{where} \quad \hat{P}_n = \prod_{j=1}^J \left( \hat{P}_n^j \right)^{\alpha_n^j}$$

---

### 1.3 Allen-Arkolakis (2014) Geographic Gravity Model with Labor Mobility

Allen and Arkolakis (2014) formulate a continuous spatial general equilibrium model over $N$ geographical regions where workers freely migrate until real utility is equalized everywhere.

#### Trade Flows and Market Access
Bilateral trade between origin $i$ and destination $j$ satisfies gravity:

$$\pi_{ij} = \frac{\left( \frac{w_i \tau_{ij}}{A_i} \right)^{-\theta}}{\sum_{k=1}^N \left( \frac{w_k \tau_{kj}}{A_k} \right)^{-\theta}} = \frac{\left( \frac{w_i \tau_{ij}}{A_i} \right)^{-\theta}}{P_j^{-\theta}}$$

where $\tau_{ij} \ge 1$ are iceberg trade costs ($\tau_{ii} = 1$), $\theta = \sigma - 1$ is the trade elasticity, and $P_j$ is the Consumer Market Access index:

$$P_j^{-\theta} = \text{CMA}_j = \sum_{i=1}^N \tau_{ij}^{-\theta} \left( \frac{w_i}{A_i} \right)^{-\theta}$$

Firm Market Access ($\text{FMA}_i$) captures export demand:

$$\text{FMA}_i = \sum_{j=1}^N \tau_{ij}^{-\theta} P_j^\theta w_j L_j$$

Goods market clearing equates regional labor income to total sales:

$$w_i L_i = \sum_{j=1}^N \pi_{ij} w_j L_j = \left( \frac{w_i}{A_i} \right)^{-\theta} \text{FMA}_i$$

### 1.4 Agglomeration, Congestion, and Spatial Equilibrium

Worker utility in location $i$ depends on real wages, local amenities $a_i$, and endogenous population $L_i$:

$$u_i = a_i \frac{w_i}{P_i}$$

Endogenous productivity and amenities incorporate spillover forces:
- **Productivity Agglomeration**: $A_i = \bar{A}_i L_i^\alpha$, where $\alpha > 0$ reflects Marshallian external scale economies.
- **Amenity Congestion**: $a_i = \bar{a}_i L_i^\beta$, where $\beta < 0$ captures housing supply inelasticity, commuting congestion, and pollution.

#### Spatial Utility Equalization
Free labor mobility equalizes utility across all populated regions:

$$u_i = \bar{u}, \quad \forall i \quad \implies \quad L_i = \bar{L} \frac{\left( \frac{w_i \bar{a}_i}{P_i} \right)^{-\frac{1}{\beta}}}{\sum_k \left( \frac{w_k \bar{a}_k}{P_k} \right)^{-\frac{1}{\beta}}}$$

#### Uniqueness Theorem
Allen and Arkolakis (2014, Theorem 2) prove that the spatial general equilibrium is **unique and strictly stable** if and only if congestion forces dominate agglomeration spillovers:

$$\alpha + \beta \le \frac{\theta}{1 + \theta} \quad \text{and} \quad \alpha \le \frac{1}{\theta}$$

`AllenArkolakisModel.is_unique` verifies this condition automatically during initialization.

---

## 2. Methodological & Model Options

| Dimension | Caliendo-Parro (2015) | Allen-Arkolakis (2014) |
|---|---|---|
| **Core Economic Mechanism** | Multi-sector trade with input-output linkages & tariffs | Spatial labor mobility with agglomeration & congestion |
| **Solving Approach** | Exact Hat Algebra on observable trade shares | Contraction mapping fixed point on $(w_i, L_i)$ |
| **Endogenous Margins** | Sectoral prices, wages, expenditure shares | Population distribution, wages, market access |
| **Labor Mobility** | Immobile across countries; mobile across sectors | Free continuous spatial mobility across regions |
| **Primary Use Cases** | Free trade agreements, tariff wars, NAFTA/USMCA | High-speed rail, highway corridors, climate shocks |
| **Convergence Rate** | 10–30 iterations ($< 0.01$ s) | 20–50 iterations ($< 0.005$ s) |

---

## 3. Canonical Calibration & Spatial Specifications

### 1. Caliendo-Parro 3-Country, 2-Sector Economy
- Countries: United States (`USA`), China (`CHN`), Germany (`DEU`).
- Sectors: Manufacturing (tradable, $\theta = 5.0$) and Services (non-tradable, $\theta = 4.0$).
- Value added share: $\gamma_n^j = 0.50$; input-output share: $\gamma_n^{j, k} = 0.25$.
- Initial tariffs: baseline free trade $\tau_{ni}^j = 1.0$.

### 2. Allen-Arkolakis 4-City US Geography
- Cities: New York City, Los Angeles, Chicago, Houston (with actual latitude/longitude coordinates).
- Trade elasticity: $\theta = 4.0$.
- Agglomeration elasticity: $\alpha = 0.08$; congestion elasticity: $\beta = -0.35$.
- Iceberg costs: parametrized via Haversine great-circle distances: $\tau_{ij} = 1.0 + \delta \cdot d_{ij}^\gamma$.

---

## 4. Runnable Worked Examples

The following script simulates a $25\%$ bilateral tariff shock in Caliendo-Parro and evaluates a transport infrastructure investment in Allen-Arkolakis:

```python
import numpy as np
from puremacro.trade import CaliendoParroModel
from puremacro.spatial import AllenArkolakisModel

# 1. 3-Country, 2-Sector Caliendo-Parro Trade Economy with Tariffs
N, J = 3, 2
trade_shares = np.zeros((J, N, N))
trade_shares[0] = np.array([
    [0.60, 0.20, 0.20],
    [0.20, 0.60, 0.20],
    [0.20, 0.20, 0.60],
])
trade_shares[1] = np.eye(N)  # Non-tradables

gamma_va = np.full((N, J), 0.50)
gamma_io = np.full((N, J, J), 0.25)
alpha = np.full((N, J), 0.50)
theta = np.array([5.0, 4.0])
labor_income = np.array([100.0, 100.0, 100.0])

cp_model = CaliendoParroModel(
    trade_shares=trade_shares,
    gamma_va=gamma_va,
    gamma_io=gamma_io,
    alpha=alpha,
    theta=theta,
    labor_income=labor_income,
    nontradables=[1],
    country_codes=["USA", "CHN", "DEU"],
    sector_codes=["Manufactures", "Services"],
)

# Simulate 25% Bilateral Tariff Shock on Manufacturing
cp_res = cp_model.simulate_tariff_shock("USA", "CHN", sector="Manufactures", tariff_rate=0.25)
assert cp_res.converged
assert len(cp_res.welfare_pct) == 3

# 2. 4-City Allen-Arkolakis Spatial GE Model with Labor Mobility
coords = np.array([
    [40.7128, -74.0060],  # NYC
    [34.0522, -118.2437], # LA
    [41.8781, -87.6298],  # Chicago
    [29.7604, -95.3698],  # Houston
])
names = ["NYC", "LA", "CHI", "HOU"]

aa_model = AllenArkolakisModel.from_coordinates(
    coords,
    region_names=names,
    theta=4.0,
    alpha=0.08,
    beta=-0.35,
    total_population=100.0,
)
aa_res = aa_model.solve_equilibrium()
assert aa_res.converged
assert np.isclose(np.sum(aa_res.population), 100.0)

# Simulate Regional Transport Infrastructure Shock (20% bilateral cost reduction between NYC and Chicago)
infra_res = aa_model.simulate_infrastructure_shock("NYC", "CHI", cost_reduction=0.20)
assert infra_res.converged

print(f"Caliendo-Parro US welfare change: {cp_res.welfare_pct[0]:+.2f}%")
print(f"Allen-Arkolakis population in NYC: {aa_res.population[0]:.2f}, LA: {aa_res.population[1]:.2f}")
print(f"Infrastructure welfare change: {infra_res.welfare_pct:+.2f}%")
```

---

## 5. Full API Specification

```text
CaliendoParroModel(
    trade_shares: np.ndarray,
    gamma_va: np.ndarray,
    gamma_io: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    labor_income: np.ndarray,
    deficits: np.ndarray | None = None,
    tariffs: np.ndarray | None = None,
    country_codes: Sequence[str] | None = None,
    sector_codes: Sequence[str] | None = None,
    nontradables: Sequence[int | str] | None = None,
)

CaliendoParroModel.solve_counterfactual(
    tariffs_new: np.ndarray | None = None,
    tau_hat: np.ndarray | None = None,
    d_hat: np.ndarray | None = None,
    deficits_new: np.ndarray | None = None,
    deficit_rule: str = "fixed",
    tol: float = 1e-8,
    max_iter: int = 1500,
    damping: float = 0.35,
) -> CaliendoParroResult

CaliendoParroModel.simulate_tariff_shock(
    importer: str | int,
    exporter: str | int,
    sector: str | int | None = None,
    tariff_rate: float = 0.10,
    **kwargs: Any,
) -> CaliendoParroResult

CaliendoParroModel.simulate_trade_war(
    coalition_a: Sequence[str | int],
    coalition_b: Sequence[str | int],
    tariff_rate: float = 0.25,
    **kwargs: Any,
) -> CaliendoParroResult

AllenArkolakisModel(
    trade_costs: np.ndarray,
    fundamental_productivity: np.ndarray,
    fundamental_amenity: np.ndarray,
    theta: float = 4.0,
    alpha: float = 0.10,
    beta: float = -0.30,
    total_population: float = 1.0,
    region_names: Sequence[str] | None = None,
    coordinates: np.ndarray | None = None,
)

AllenArkolakisModel.from_coordinates(
    coords: Any,
    fundamental_productivity: np.ndarray | None = None,
    fundamental_amenity: np.ndarray | None = None,
    distance_cost_param: float = 0.001,
    distance_exponent: float = 0.5,
    metric: str = "haversine",
    theta: float = 4.0,
    alpha: float = 0.10,
    beta: float = -0.30,
    total_population: float = 1.0,
    region_names: Sequence[str] | None = None,
) -> AllenArkolakisModel

AllenArkolakisModel.solve_equilibrium(
    tol: float = 1e-8,
    max_iter: int = 2500,
    damping: float = 0.35,
) -> AllenArkolakisResult

AllenArkolakisModel.simulate_infrastructure_shock(
    origin: str | int,
    destination: str | int,
    cost_reduction: float = 0.20,
    **kwargs: Any,
) -> AllenArkolakisResult

AllenArkolakisModel.simulate_climate_shock(
    productivity_shocks: Mapping[str | int, float] | np.ndarray | None = None,
    amenity_shocks: Mapping[str | int, float] | np.ndarray | None = None,
    **kwargs: Any,
) -> AllenArkolakisResult
```

---

## 6. Result Interface & Policy Counterfactuals

### `CaliendoParroResult`
- `res.w_hat`: Gross change in nominal wages $\hat{w}_n$.
- `res.P_index_hat`: Gross change in consumer price indices $\hat{P}_n$.
- `res.real_wage_hat`: Gross change in real wages $\hat{w}_n / \hat{P}_n$.
- `res.welfare_pct`: Percentage change in national welfare $(\widehat{\mathcal{W}}_n - 1) \times 100$.
- `res.tariff_revenue_prime`: Counterfactual tariff revenues by country.
- `res.summary() -> pd.DataFrame`: Country-level summary of welfare, wages, prices, and revenues.
- `res.sector_summary() -> pd.DataFrame`: Country-by-sector breakdown of gross outputs and prices.
- `res.to_frame()`, `res.to_markdown()`, `res.to_latex()`, `res.to_typst()`: Publication serialization formats.
- `res.plot(kind="welfare") -> matplotlib.figure.Figure`: Bar chart of country welfare impacts.

### `AllenArkolakisResult`
- `res.wages`: Equilibrium regional nominal wages $w_i$.
- `res.population`: Spatial population distribution $L_i$, strictly conserving $\sum L_i = \bar{L}$.
- `res.price_index`: Spatial CES price index $P_i$.
- `res.real_wages`: Spatial real wages $w_i / P_i$.
- `res.welfare`: Equalized spatial utility level $\bar{u}$.
- `res.consumer_market_access`: Regional consumer market access index $\text{CMA}_i$.
- `res.firm_market_access`: Regional firm export market access index $\text{FMA}_i$.
- `res.welfare_pct`: Counterfactual percentage utility change $(\bar{u}' / \bar{u} - 1) \times 100$.
- `res.summary() -> pd.DataFrame`: Regional summary table.
- `res.plot(kind="spatial") -> matplotlib.figure.Figure`: Geographic scatter plot sized by population and colored by real wages.

---

## References

- Allen, T., & Arkolakis, C. (2014). "Trade and the Topography of the Spatial Economy." *The Quarterly Journal of Economics*, 129(3), 1085–1140.
- Caliendo, L., & Parro, F. (2015). "Estimates of the Trade and Welfare Effects of NAFTA." *The Review of Economic Studies*, 82(1), 1–44.
- Dekle, R., Eaton, J., & Kortum, S. (2007). "Unbalanced Trade." *American Economic Review*, 97(2), 351–355.
- Eaton, J., & Kortum, S. (2002). "Technology, Geography, and Trade." *Econometrica*, 70(5), 1741–1779.
