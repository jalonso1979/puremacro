> 🇬🇧 English · 🇪🇸 [Español](es/portfolio_and_macro_allocation.md)

# Practitioner & Hobbyist Portfolio Analysis & Macro Allocation (`puremacro.portfolio`)

`puremacro.portfolio` provides a pure-NumPy/SciPy quantitative portfolio optimization,
macro-quadrant tactical asset allocation, multi-factor attribution, realistic
rebalancing backtester, and Monte Carlo retirement/wealth glidepath engine:

1. **Covariance Denoising & Portfolio Optimization (`puremacro.portfolio.allocation`)**:
   - `shrink_covariance`: Analytical Ledoit-Wolf shrinkage and Random Matrix Theory (Marčenko-Pastur) eigenvalue denoising.
   - `optimize_portfolio`: Unified entry point supporting `"max_sharpe"`, `"min_variance"`, `"risk_parity"` (Equal Risk Contribution), `"hrp"` (López de Prado 2016 Hierarchical Risk Parity), `"min_cvar"` (Rockafellar-Uryasev Minimum Conditional Value-at-Risk), and `"kelly"` (fractional Kelly criterion).
   - `markowitz_frontier`: Traces the efficient frontier and identifies the Minimum-Variance, Tangency (Max-Sharpe), and Risk-Parity portfolios.
   - `black_litterman`: Reverse-optimized market equilibrium prior blended with absolute or relative macro/investor views (including Idzorek 0–100% view confidences).
2. **Macro-Regime Tactical Allocation & Factor Attribution (`puremacro.portfolio.macro_regime`)**:
   - `classify_macro_quadrants`: Classifies growth and inflation dynamics into the 4 classic macro quadrants (`Goldilocks`, `Reflation`, `Stagflation`, `Deflationary Bust`).
   - `all_weather_portfolio`: Bridgewater-style All-Weather regime-balanced allocation or probability-tilted macro allocation across the 4 macro quadrants.
   - `factor_attribution`: Multi-factor OLS + Newey-West HAC alpha/beta decomposition with rolling factor exposures and variance attribution.
3. **Realistic Rebalancing Backtester & Wealth Glidepath (`puremacro.portfolio.backtest`)**:
   - `backtest_portfolio`: Tracks drifting weights, calendar or corridor (`drift_band`) rebalancing, proportional transaction costs in basis points, underwater drawdowns, and a 4-panel Tear Sheet (`.plot()`).
   - `simulate_wealth_glidepath`: Block-bootstrap / parametric Monte Carlo simulator for accumulation (DCA) and retirement decumulation (Safe Withdrawal Rate / 4% rule ruin probability and fan charts).

---

## 1. Covariance Shrinkage, HRP, Risk Parity & Black-Litterman

```python
import numpy as np
import pandas as pd
from puremacro.portfolio import (
    shrink_covariance,
    optimize_portfolio,
    markowitz_frontier,
    hierarchical_risk_parity,
    black_litterman,
)

rng = np.random.default_rng(42)
dates = pd.date_range("2015-01-31", periods=120, freq="ME")
returns = pd.DataFrame(
    rng.normal([0.008, 0.004, 0.005, 0.003], [0.045, 0.020, 0.035, 0.015], size=(120, 4)),
    index=dates,
    columns=["Equities", "Bonds", "Gold", "TIPS"],
)

# 1a. Random Matrix Theory (Marčenko-Pastur) or Ledoit-Wolf covariance shrinkage
cov_lw = shrink_covariance(returns, method="ledoit_wolf")

# 1b. Hierarchical Risk Parity (HRP) & Equal Risk Contribution (Risk Parity)
hrp_res = hierarchical_risk_parity(returns)
rp_res = optimize_portfolio(returns, method="risk_parity")
print(hrp_res.summary())
print(rp_res.summary())

# 1c. Black-Litterman with investor/macro views
P = np.array([[1.0, -1.0, 0.0, 0.0]])  # Equities outperform Bonds by 4% annualized
Q = np.array([0.04 / 12.0])
bl_res = black_litterman(
    cov_lw,
    market_weights=[0.50, 0.30, 0.10, 0.10],
    P=P,
    Q=Q,
    confidences=[0.70],
)
print(bl_res.summary())
```

---

## 2. Macro-Quadrant Classification & All-Weather Portfolio

```python
from puremacro.portfolio import (
    classify_macro_quadrants,
    all_weather_portfolio,
    factor_attribution,
)

growth = pd.Series(rng.normal(2.2, 1.0, size=120), index=dates, name="gdp_growth")
inflation = pd.Series(rng.normal(2.5, 0.8, size=120), index=dates, name="cpi_inflation")

quadrants = classify_macro_quadrants(growth, inflation, method="change")
aw_res = all_weather_portfolio(returns, growth, inflation, method="risk_parity")
print(aw_res.summary())
```

---

## 3. Rebalancing Backtest Tear Sheet & Retirement Wealth Glidepath

```python
from puremacro.portfolio import backtest_portfolio, simulate_wealth_glidepath

bt = backtest_portfolio(
    returns,
    weights=rp_res.weights,
    rebalance="quarterly",
    tc_bps=10.0,
)
print(bt.summary())

glide = simulate_wealth_glidepath(
    returns,
    weights=rp_res.weights,
    initial_wealth=500_000.0,
    horizon_years=30,
    annual_withdrawal=20_000.0,
    n_sims=500,
    seed=42,
)
print(glide.summary())
```
