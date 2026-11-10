> 🇬🇧 English · 🇪🇸 [Español](es/trade_reallocation.md)

# Joint sourcing, capital and labor transitions

`puremacro.trade.reallocation` solves an **experimental two-country,
two-sector** competitive equilibrium with endogenous origin sourcing, capital
accumulation and costly sector labor reallocation. Goods prices, consumption,
factor prices and all three allocation margins are solved jointly.
The defaults are synthetic assumptions, with pooled national household income.

This small model has its own explicit closure. The existing
[dynamic MRIO model](trade_dynamic.md) has fixed sourcing and nationally mobile
labor; [conditional sourcing](trade_adjustment.md) takes output and prices as
given. Their returned paths cannot simply be combined into this equilibrium.
There are no intermediate inputs, international migration, traded bonds,
uncertainty or heterogeneous workers here.

## Run the temporary tariff

AAA announces at date zero a 10% import tariff on sector S01 for dates 2–7.
All other rates, including the terminal rates, remain zero.

```python
import numpy as np
from puremacro.trade import (
    ReallocationEconomy, audit_reallocation, reallocation_stability,
    run_reallocation_horizon_ladder, solve_reallocation_steady_state,
)

economy = ReallocationEconomy()
initial = solve_reallocation_steady_state(economy)
assert reallocation_stability(initial)["stable_graph"]

def policy(horizon):
    tariffs = np.zeros((horizon, 2, 2))
    tariffs[2:8, 0, 0] = 0.10
    return tariffs

ladder = run_reallocation_horizon_ladder(economy, policy, initial=initial)
assert ladder.converged, ladder.to_frame()
run = ladder.accepted
assert run is not None and audit_reallocation(run)["accepted"]
print(ladder.to_frame())
print(run.welfare)
print(run.sector_frame().head())
```

The full example compares the same policy with each adjustment cost set to zero:

```bash
python -m puremacro.examples.trade_reallocation --output trade-example --lang en
```

It exports complete arrays (`path-*.npz`), national and sector accounts, period
audit errors, welfare, horizon attempts, PNG/PDF figures, parameters, environment
versions and source hashes. A failed horizon check remains in the record; an
unresolved case makes the command exit unsuccessfully. The example tries
40, 80, 160 and, if needed, 320 dates. The API defaults stop at 160.

## Economic quantities and timing

Capital $K_{cs,t}$ is inherited. The household chooses current sector labor
$L_{cs,t}$, a current supplier recipe $q_{cso,t}$, and next capital $K_{cs,t+1}$.
Initial labor and supplier ratios are inherited from an audited steady state.
National labor supply $\bar L_c$ is fixed. All dates are known initially.

Competitive firms produce $Y_{cs}=Z_{cs}K_{cs}^aL_{cs}^{1-a}$, with
$w_{cs}=(1-a)p_{cs}Y_{cs}/L_{cs}$ and $R_{cs}=ap_{cs}Y_{cs}/K_{cs}$.
Each country's household owns its capital and pools both sectors' earnings.

The final composite is $A_c=\prod_s(X_{cs}/b_{cs})^{b_{cs}}$, with positive
weights summing to one. Sector goods combine origins using

$$
\left[\sum_o\alpha_{cso}(q_{cso}/\alpha_{cso})^\rho\right]^{1/\rho}=1,
\qquad \rho=(\sigma-1)/\sigma.
$$

Home and foreign weights are `home_shares` and `1-home_shares`. These are
preferences/technology weights, not fixed purchase shares. The code includes
the Cobb–Douglas limit $\sigma=1$. The choice coordinate is
$r_{cs}=\log[(q_{cs,foreign}/\alpha_{cs,foreign})/(q_{cs,home}/\alpha_{cs,home})]$.
With importer tariffs applying only to foreign purchases,

$$
P_{cs}=\sum_o(1+\tau_{cso})p_{os}q_{cso},\quad
P_c=\prod_sP_{cs}^{b_{cs}},\quad
X_{cs}=b_{cs}P_cA_c/P_{cs},\quad Q_{cso}=X_{cs}q_{cso}.
$$

Goods clear at every origin: $Y_{os}=\sum_cQ_{cso}$.
The producer price in BBB, S02 is the numeraire at every date.

## Real adjustment costs and accounts

Let $g_{cs}=K'_{cs}/K_{cs}-1$ and $I_{cs}=K'_{cs}-(1-\delta)K_{cs}$.
All adjustment costs consume **national final goods**:

$$
\Phi^K_{cs}=\frac{\kappa_K}{2}K_{cs}g_{cs}^2,\quad
\Phi^L_c=\frac{\kappa_L}{2\bar L_c}\sum_s(L_{cs}-L_{cs,prev})^2,\quad
\Phi^Q_{cs}=\frac{\kappa_Q}{2}\bar L_cb_{cs}(r_{cs}-r_{cs,prev})^2.
$$

Absorption is $A_c=C_c+\sum_s(I_{cs}+\Phi^K_{cs}+\Phi^Q_{cs})+\Phi^L_c$.
Tariff revenue $TR_c=\sum_{so}\tau_{cso}p_{os}Q_{cso}$ is rebated lump sum.
The household budget is
$P_cA_c=\sum_s(w_{cs}L_{cs}+R_{cs}K_{cs})+TR_c$.
Households take prices and rebates as given. Financial autarky implies balanced
trade at producer prices each date. Investment is reversible and can be
negative; capital, consumption, quantities, prices and marginal capital values
must be positive.

With $m_t=\beta(C_{t+1}/C_t)^{-\gamma}$, $q^K=1+\kappa_Kg$ and
$l_t=L_{1,t}/\bar L$, the interior conditions are

$$
q^K_t=m_t[R_{t+1}/P_{t+1}+1-\delta+\kappa_K(g_{t+1}+g_{t+1}^2/2)],
$$

$$
(w_{1,t}-w_{2,t})/P_t-2\kappa_L(l_t-l_{t-1})
+2m_t\kappa_L(l_{t+1}-l_t)=0,
$$

$$
(A_t/\bar L)\partial_{r_{s,t}}\log P_{s,t}
+\kappa_Q(r_{s,t}-r_{s,t-1})-m_t\kappa_Q(r_{s,t+1}-r_{s,t})=0.
$$

The sourcing derivative multiplies gross absorption, including investment and
all costs. Setting each cost to zero gives, respectively, the frictionless
capital Euler equation, equal sector wages, or static CES sourcing. General
equilibrium welfare need not improve monotonically when a cost is removed.

## Terminal approximation and acceptance

A separate audited steady state uses the final tariff. The continuation value
is linear in terminal capital, with derivative $u'(C^*)/\beta$; derivatives
with respect to inherited labor and sourcing are zero at stationarity. Thus
the last capital condition is $q^K_{T-1}=(C^*/C_{T-1})^{-\gamma}$ and the last
forward labor/sourcing cost terms vanish. This is a first-order tail approximation.

`run.converged` means the **finite** equations and independent audit passed.
It does not establish tail accuracy. The default horizon ladder requires:

| Quantity | Limit |
| --- | --- |
| Solver and independent accounting error | $10^{-8}$ |
| Maximum change in the first 20 log/logit/sourcing coordinates | $10^{-6}$ |
| Change in consumption equivalents | $10^{-4}$ percentage points |
| Maximum terminal coordinate gap from the steady state | $10^{-5}$ |

The last three compare successive accepted finite solves; all policy prefixes
must match and extensions must keep the terminal policy. Failed solves break
the comparison chain. These are observed refinement checks, not global
uniqueness or rigorous infinite-tail bounds.

The solver uses 16 unknowns per date and a sparse three-diagonal block Jacobian
with complex-step derivatives. One redundant budget is replaced by the
numeraire. `audit_reallocation` rebuilds both budgets, bilateral goods,
production, factor payments, fiscal and external balances, adjustment resources
and every optimality condition from the economic arrays. It does not use the
retained residual. Numerical log/logit bounds are ±30; solutions near them are
rejected. The evaluation budget counts SciPy `nfev`, excluding Jacobian work.
See [SciPy least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

`reallocation_stability` checks the local stationary companion pencil using
[ordered generalized Schur decomposition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ordqz.html).
A stable graph requires 16 stable roots, including zero modes for variables
without lags, a well-conditioned projection and a small matrix-equation error.
The default calibration has 10 active inherited directions. A failed terminal
graph rejects a transition by default. `require_stable_terminal=False` permits
an explicitly finite boundary experiment; the horizon ladder always keeps the
guard. This diagnostic is local and does not prove household global optimality.

## Welfare and the illustrative calibration

`run.welfare` reports national consumption equivalents relative to initial
steady consumption. It includes the recorded discounted consumption stream,
constant terminal consumption, and the linear terminal-capital correction.
In normalized utility the correction is
$\beta^{T-1}(C^*/C_0)^{-\gamma}\sum_s(K_{s,T}-K_s^*)/C_0$.
The log-utility and near-log limits are supported. The result marks
`tail_is_approximation=True`, even after refinement passes.

The initially considered sourcing cost **1.0** failed the local terminal graph
(15 stable roots instead of 16) and the horizon pilot. It was retained as
rejected, and the illustrative default was explicitly changed to **0.1**.
No tolerance was relaxed, and no universal upper bound on sourcing costs is
inferred. Other defaults are $\beta=.96$, $\delta=.08$, $a=.3$, $\gamma=2$,
$\sigma=4$, $\kappa_K=4$, $\kappa_L=5$, home weight .75, equal sector weights,
and unit productivity and labor supply.

For the example, all four cases pass at 160 dates. The full-model 80-to-160
comparison has coordinate gap $3.78\times10^{-8}$, welfare gap
$6.23\times10^{-7}$ percentage points and terminal gap $1.45\times10^{-7}$.
The illustrative national consumption equivalents are about **+0.1233% for
AAA** and **−0.1469% for BBB**. These synthetic values are not empirical tariff
recommendations. Sector frames report wages and earnings, but pooled national
income does not identify worker-specific welfare.

Labor-adjustment research such as
[Artuc, Chaudhuri and McLaren](https://www.nber.org/system/files/working_papers/w13465/w13465.pdf)
motivates studying adjustment explicitly. This continuous pooled-household
model is not a replication of their worker-choice model. Larger dimensions,
MRIO intermediates, borrowing, worker heterogeneity, empirical estimation and
global optimality remain outside this first implementation.
