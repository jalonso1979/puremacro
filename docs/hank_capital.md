> 🇬🇧 English · 🇪🇸 [Español](es/hank_capital.md)

# Productive capital in two-asset HANK

`calibrate_capital_hank` builds a two-asset economy in which households own
nominal government bonds and installed productive capital. Investment changes
the capital supply, its price clears the illiquid market, and the return includes
both operating rental income and the value of remaining installed capital.
`solve_hank_reduced_policy` solves its local stochastic equilibrium with reduced
distribution and household marginal-value coordinates. Full-grid household
backward steps and distribution transport remain available for nonlinear audits
and state-conditioned stochastic continuation corrections.

```python
import numpy as np
from puremacro.models import calibrate_capital_hank, solve_hank_reduced_policy

# Small demonstration grid, not a household-grid convergence certificate.
ss = calibrate_capital_hank(n_a=10, n_b=10, a_max=100., b_max=15.)
policy = solve_hank_reduced_policy(
    ss, distribution_rank=12, marginal_rank=48, rho=.7, shock_std=1e-4,
)
audit = policy.audit(np.r_[-1e-5, np.zeros(7)], tolerance=1e-5)
print(ss.diagnostics, policy.diagnostics, audit.diagnostics)
print(audit.paths[["Y", "K", "I", "q", "r_a", "goods_mkt"]])
```

Linear policy paths contain log control deviations. Nonlinear audit paths
contain level deviations and residuals. In those paths, `K` is inherited
physical capital and `Knext` is next period's inherited stock. `I` is physical
investment; `investment_goods` adds installation costs. The exact nominal rate
and quoted capital price are carried between audited dates.

## Production and investment timing

Write \(K_t\) for capital inherited at date \(t\), \(q_{t-1}\) for its actual
purchase price, and \(q_t\) for the price of capital installed for \(t+1\).
Production, marginal cost and the capital rental rate satisfy

\[
Y_t=Z K_t^\alpha N_t^{1-\alpha},\qquad
mc_t=\frac{w_tN_t}{(1-\alpha)Y_t},\qquad
R^K_t=\frac{\alpha mc_tY_t}{K_t}.
\]

Physical investment and its resource cost are

\[
\Phi(I_t,K_t)=\frac{\psi}{2}\left(\frac{I_t}{K_t}-\delta\right)^2K_t,
\qquad \psi=\frac1{\delta\epsilon_I},\qquad
q_t=1+\psi\left(\frac{I_t}{K_t}-\delta\right),
\]
\[
I_t=K_t[\delta+(q_t-1)/\psi],\qquad
K_{t+1}=(1-\delta)K_t+I_t.
\]

The technology is reversible: negative investment is allowed while
\(K_{t+1}>0\). Irreversible investment would require an additional complementarity
condition and is not imposed here.

The return on installed capital includes the installation rent
\(\Pi^I_t=(q_t-1)I_t-\Phi_t=-K_t\Phi_K(I_t,K_t)\):

\[
1+r^a_t=\frac{R^K_t+(1-\delta)q_t+\Pi^I_t/K_t}{q_{t-1}}
=\frac{R^K_t-(I_t+\Phi_t)/K_t+q_tK_{t+1}/K_t}{q_{t-1}}.
\]

Leaving out this rent changes the marginal capital valuation equation when
investment moves. The tests independently differentiate the installation
technology with respect to investment and inherited capital to check both
the investment FOC and the realized return.

## Household and aggregate closure

The existing convex-cost two-asset household solver chooses consumption, liquid
bonds, and illiquid wealth on the full grid. Both asset markets clear:

\[
\int a'_t\,dD_t=q_t K_{t+1},\qquad \int b'_t\,dD_t=\bar B.
\]

Government purchases are fixed in levels after calibration. Taxes finance
government purchases and liquid interest only:

\[
1+r^b_t=\frac{1+i_{t-1}}{1+\pi_t},\quad
T_t=r^b_t\bar B+G,\quad
i_t=\bar r^b+\phi\pi_t+\varepsilon_t.
\]

Operating profits are
\(\Pi^P_t=Y_t-w_tN_t-R^K_tK_t-\mathrm{pricecost}_t\).
Nonfinancial income \(w_tN_t+\Pi^P_t-T_t\) is distributed proportional to
household productivity, whose stationary mean is one. Installation rents are
already included in capital returns, so are not rebated again. Flexible common
hours satisfy

\[
w_t\int e c_t^{-\gamma}\,dD_t=v_\phi N_t^{1/\mathrm{Frisch}}.
\]

Rotemberg price adjustment uses
\(\mathrm{pricecost}_t=\mu\,[\log(1+\pi_t)]^2Y_t/[2(\mu-1)\kappa]\).
The price Phillips curve is

\[
\log(1+\pi_t)=\kappa(mc_t-1/\mu)
+E_t\left[\frac{Y_{t+1}}{Y_t}
\frac{\log(1+\pi_{t+1})}{1+r^b_{t+1}}\right].
\]

Household budgets and the two markets imply the separately audited resource
equation

\[
Y_t=C_t+\mathrm{CHI}_t+I_t+\Phi_t+\mathrm{pricecost}_t+G.
\]

`resource_accounting_res` also checks the identity **off equilibrium**. It
relates the goods residual to current asset-market residuals and inherited
ownership gaps. In particular an actual transported distribution can have
\(\int a_t\,dD_t\ne q_{t-1}K_t\). Audits preserve the actual previous quote;
they never change that return to absorb an illiquid-market error.

The recursive state is the distribution coordinates plus
`lag_nominal`, `monetary_shock`, and `log_capital`. The last coordinate is
\(\log(K_t/\bar K)\). In linear equilibrium the inherited price can be
identified by \(q_{t-1}=\int a_t\,dD_t/K_t\); off-equilibrium nonlinear paths
carry that price explicitly.

## Calibration and reference scope

The default physical parameters use the rounded ABRS values
\(\alpha=.33,\delta=.02,\mu=1.015,\epsilon_I=4\); inverse IES is 2 and Frisch
elasticity is 1. Given the steady liquid and illiquid returns, calibration solves
for **beta**, keeping those production parameters fixed:

\[
(\bar r^a+\delta)\bar K=\alpha\bar Y/\mu,\qquad
\bar K=\bar A,\qquad
\bar Y=\frac{\bar C+\overline{\mathrm{CHI}}+\delta\bar K}{1-g}.
\]

Debt supply equals stationary liquid demand. TFP and labor disutility normalize
steady hours to one. Changing the household grid can change beta and the
normalizations; it does not silently change alpha, depreciation, or installation
cost parameters. Failed household solves and a beta interval that does not
bracket the capital condition raise. The calibration history retains warnings
from intermediate household solves, and final diagnostics retain upper-grid
mass, household policy errors, and resource residuals.

Production, investment, price adjustment, and the capital valuation identity
are matched to the authors' [SSJ two-asset example at pinned commit
eb4ac4c](https://github.com/shade-econ/sequence-jacobian/blob/eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9/src/sequence_jacobian/examples/two_asset.py).
See also their [two-asset tutorial](https://github.com/shade-econ/sequence-jacobian/blob/eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9/notebooks/two_asset.ipynb)
and [paper](https://doi.org/10.3982/ECTA17434).
The independent aggregate-block reference test evaluates the authors' equations
at off-steady paths, not just their stationary parameter values.

This is **not a replication of the full published HANK economy**. The reference
has sticky wages, an intermediary combining equity and bonds, a liquidity
premium, a larger earnings process, and a different distribution of profits and
taxes. Its exact calibrated alpha is inferred from targets, whereas this model
holds the rounded .33 fixed. Its portfolio cost denominator is
\((1+r^a)a+\chi_0\); the existing PureMacro household solver uses \(a+\bar a\).
PureMacro retains that solver and reports the difference. A published-model
replication would require those household, wage, intermediary, equity, and
calibration blocks as well.

Projected equilibrium convergence is distinct from household-grid accuracy and
omitted marginal-value equations. Compare ranks, inspect full-equation and
Euler/KKT diagnostics, and use held-out nonlinear histories. A solved local
stochastic tail does not establish global aggregate-risk accuracy or
second-order welfare effects.
