# Quickstart Guide

Get up and running with `puremacro` in less than 2 minutes.

---

## 1. Sequence-Space HANK Model

Solve an incomplete-markets New Keynesian model with uninsurable idiosyncratic income risk:

```python
from puremacro.models import solve_hank_sequence_space

# Solve GE equilibrium transition path to a 25 bps rate hike
res = solve_hank_sequence_space(T=40, beta=0.985, phi_pi=1.5, kappa=0.1)
print(res.summary())

# Inspect output and inflation impulse responses
print("Peak output drop:", res.irf_output.min())
print("Bottom-decile MPC:", res.mpc_distribution["Decile 1"])
```

---

## 2. Factor-Augmented VAR (FAVAR)

Extract latent factors from a 50-variable macroeconomic panel:

```python
import pandas as pd
from puremacro.var import favar

# Load data
panel = pd.read_csv("macro_panel.csv")
policy_rate = panel["FEDFUNDS"]

# Estimate FAVAR with 3 factors and 90% bootstrap bands
favar_res = favar(
    panel.drop(columns=["FEDFUNDS"]),
    policy_rate,
    n_factors=3,
    p=2,
    horizon=20,
)
print(favar_res.summary())
```

---

## 3. Mixed-Frequency GDP Nowcasting

Track quarterly GDP growth in real time from monthly indicators with missing releases:

```python
from puremacro.nowcast import nowcast_gdp

res = nowcast_gdp(monthly_indicators_df, historical_gdp_series, n_factors=2)
print(res.summary())
# View news contribution of the latest industrial production release
print(res.news_decomposition)
```

---

## 4. DICE Climate-Macro Simulation

Simulate global warming trajectories and the Social Cost of Carbon:

```python
from puremacro.climate import simulate_dice_model

dice_res = simulate_dice_model(
    n_periods=30,
    time_step_years=5,
    carbon_tax_initial=50.0,
    carbon_tax_growth=0.03,
)
print(dice_res.summary())
```
