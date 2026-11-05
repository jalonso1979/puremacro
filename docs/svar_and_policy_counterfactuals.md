> 🇬🇧 English · 🇪🇸 [Español](es/svar_and_policy_counterfactuals.md)

# SVAR Identification, Semi-Structural Policy Counterfactuals & Structural Bridges

PureMacro provides a unified toolkit for modern macroeconometric identification,
Lucas-critique-robust policy counterfactuals, granular and distributional
estimation, and structural-empirical sensitivity bridges:

1. **First-Class SVAR Identification Suite (`puremacro.var.svar`)**:
   - Volatility-regime identification (`svar_heteroskedastic` — Rigobon 2003; Lanne & Lütkepohl 2008)
   - Non-Gaussian / higher-moment identification (`svar_non_gaussian` — Lanne, Meitz & Saikkonen 2017; Keweloh 2021)
   - External-instrument Proxy-SVAR with Olea-Pflueger effective $F$ and weak-IV Anderson-Rubin bands (`svar_proxy` — Mertens & Ravn 2013; Stock & Watson 2018)
   - Sign + Narrative restrictions with Giacomini & Kitagawa (2021) prior-robust identified-set bounds (`svar_sign_narrative`)
2. **Semi-Structural Policy Counterfactuals (`puremacro.lp.counterfactual`)**:
   - Lucas-critique-robust IRF counterfactuals (`mckay_wolf_counterfactual` — McKay & Wolf 2023)
   - Optimal Policy Perturbations (`barnichon_mesters_opp` — Barnichon & Mesters 2020, 2023)
3. **Granular IV & Functional/Distributional VARs (`puremacro.inference.giv`, `puremacro.var.functional`)**:
   - Granular Instrumental Variables (`estimate_giv` — Gabaix & Koijen 2024)
   - Functional / Distributional VARs (`estimate_functional_var` — Chang, Kim & Schorfheide 2024)
4. **Structural–Empirical Bridges (`puremacro.structural.bridge`, `puremacro.sovereign`)**:
   - Andrews, Gentzkow & Shapiro (2017) sensitivity matrix $\Lambda$ (`ags_sensitivity`)
   - Empirical Bohn (1998) / Ghosh et al. (2013) fiscal reaction function estimation (`estimate_fiscal_reaction_function`, `fit_stochastic_dsa_from_data`)
   - Sovereign Default SMM / minimum-distance calibration (`calibrate_fiscal_sovereign_smm`)

---

## 1. SVAR Identification Suite (`puremacro.var.svar`)

```python
import numpy as np
import pandas as pd
from puremacro.var.svar import (
    svar_heteroskedastic,
    svar_non_gaussian,
    svar_proxy,
    svar_sign_narrative,
)

rng = np.random.default_rng(42)
T = 240
B0_true = np.array([[1.0, 0.2], [-0.4, 1.0]])
eps = rng.standard_t(df=5, size=(T, 2))
eps[T // 2 :, 0] *= 2.2  # High-volatility regime in second half
u = eps @ B0_true.T
Y = pd.DataFrame(u, columns=["output", "inflation"])

# 1a. Identification via heteroskedasticity
het_res = svar_heteroskedastic(Y, regime_split=T // 2, p=1, horizon=12)
print(het_res.summary())

# 1b. Non-Gaussian / higher-moment identification
ng_res = svar_non_gaussian(Y, p=1, horizon=12, method="gmm")
print(ng_res.summary())

# 1c. Sign + Narrative restrictions with Giacomini-Kitagawa (2021) bounds
sn_res = svar_sign_narrative(
    Y,
    p=1,
    horizon=12,
    sign_restrictions={(0, 0, 0): 1, (1, 0, 0): -1},
    n_draws=200,
)
print(sn_res.summary())
```

---

## 2. Semi-Structural Policy Counterfactuals (`puremacro.lp.counterfactual`)

McKay & Wolf (2023, *REStud*) show that any linear counterfactual policy rule
can be implemented without solving a full structural model by combining baseline
non-policy IRFs with a matrix of contemporaneous and news policy-shock IRFs.
Barnichon & Mesters (2020, *AER*; 2023, *JPE*) use those same policy IRFs to
test whether the baseline policy response was optimal (`opp_gradient == 0`).

```python
import numpy as np
from puremacro.lp.counterfactual import (
    mckay_wolf_counterfactual,
    barnichon_mesters_opp,
)

H = 12
h_grid = np.arange(H + 1)
base_y = -0.8 * (0.75 ** h_grid)
base_i = -0.3 * (0.70 ** h_grid)

# Policy Jacobian matrices Theta_y and Theta_i (H+1 x k)
Theta_y = -0.45 * np.eye(H + 1)
Theta_i = np.eye(H + 1)

cf = mckay_wolf_counterfactual(
    baseline_target_irf=base_y,
    baseline_instrument_irf=base_i,
    policy_target_irfs=Theta_y,
    policy_instrument_irfs=Theta_i,
    rule_target_weights=1.5,
)
print(cf.summary())

opp = barnichon_mesters_opp(
    baseline_target_irf=base_y,
    policy_target_irfs=Theta_y,
    loss_weights=1.0,
    discount_beta=0.99,
)
print(opp.summary())
```

---

## 3. Granular IV & Functional VAR (`puremacro.inference.giv`, `puremacro.var.functional`)

```python
import numpy as np
from puremacro.inference.giv import estimate_giv
from puremacro.var.functional import estimate_functional_var

# Granular IV (Gabaix & Koijen 2024)
rng = np.random.default_rng(7)
T, N = 120, 15
shares = (np.arange(1, N + 1, dtype=float) ** -1.1)
shares /= shares.sum()
shocks = rng.normal(0.0, 1.0, size=(T, N))
X_t = shocks @ shares
Y_t = 1.5 * X_t + rng.normal(0.0, 0.1, size=T)

giv = estimate_giv(shocks, shares, Y_t, endog_macro=X_t, n_factors=1)
print(giv.summary())
```

---

## 4. Structural Sensitivity ($\Lambda$) & Empirical Sovereign Bridges

```python
import numpy as np
from puremacro.structural.bridge import MomentTargets, fit_structural, ags_sensitivity

G = np.array([[2.0, 0.5], [0.2, 1.5], [1.0, 1.0]])
targets = MomentTargets(
    values=np.array([2.0, -1.0, 0.4]),
    covariance=np.diag([0.04, 0.09, 0.16]),
    labels=("debt_pct", "spread_mean_pct", "spending_pct"),
)
fit = fit_structural(lambda th: G @ th, targets, theta0=[0.5, -0.5], parameter_names=("beta", "d0"))
sens = ags_sensitivity(fit)
print(sens.summary())
```
