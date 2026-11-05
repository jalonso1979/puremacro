> 🇬🇧 [English](../svar_and_policy_counterfactuals.md) · 🇪🇸 Español

# Identificación SVAR, Contrafactuales Semi-Estructurales y Puentes Estructurales

PureMacro ofrece un conjunto integrado de herramientas para la identificación
macroeconométrica moderna, contrafactuales de política robustos a la crítica de
Lucas, estimación granular y distribucional, y puentes de sensibilidad entre
modelos estructurales y momentos empíricos:

1. **Suite de Identificación SVAR (`puremacro.var.svar`)**:
   - Identificación por regímenes de volatilidad (`svar_heteroskedastic` — Rigobon 2003; Lanne y Lütkepohl 2008)
   - Identificación no gaussiana / momentos superiores (`svar_non_gaussian` — Lanne, Meitz y Saikkonen 2017; Keweloh 2021)
   - Proxy-SVAR con instrumentos externos, estadístico $F$ efectivo de Olea-Pflueger y bandas Anderson-Rubin robustas a instrumentos débiles (`svar_proxy` — Mertens y Ravn 2013; Stock y Watson 2018)
   - Restricciones de signo y narrativas con cotas robustas a la distribución a priori de Giacomini y Kitagawa (2021) (`svar_sign_narrative`)
2. **Contrafactuales de Política Semi-Estructurales (`puremacro.lp.counterfactual`)**:
   - Contrafactuales de IRF robustos a la crítica de Lucas (`mckay_wolf_counterfactual` — McKay y Wolf 2023)
   - Perturbaciones Óptimas de Política (`barnichon_mesters_opp` — Barnichon y Mesters 2020, 2023)
3. **Variables Instrumentales Granulares y VAR Funcional (`puremacro.inference.giv`, `puremacro.var.functional`)**:
   - Variables Instrumentales Granulares (`estimate_giv` — Gabaix y Koijen 2024)
   - VAR Funcional / Distribucional (`estimate_functional_var` — Chang, Kim y Schorfheide 2024)
4. **Puentes Estructural–Empíricos (`puremacro.structural.bridge`, `puremacro.sovereign`)**:
   - Matriz de sensibilidad $\Lambda$ de Andrews, Gentzkow y Shapiro (2017) (`ags_sensitivity`)
   - Estimación empírica de funciones de reacción fiscal de Bohn (1998) y Ghosh et al. (2013) (`estimate_fiscal_reaction_function`, `fit_stochastic_dsa_from_data`)
   - Calibración SMM / distancia mínima de incumplimiento soberano (`calibrate_fiscal_sovereign_smm`)

---

## 1. Suite de Identificación SVAR (`puremacro.var.svar`)

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
eps[T // 2 :, 0] *= 2.2  # Régimen de alta volatilidad en la segunda mitad
u = eps @ B0_true.T
Y = pd.DataFrame(u, columns=["output", "inflation"])

# 1a. Identificación por heterocedasticidad
het_res = svar_heteroskedastic(Y, regime_split=T // 2, p=1, horizon=12)
print(het_res.summary())

# 1b. Identificación no gaussiana por momentos superiores
ng_res = svar_non_gaussian(Y, p=1, horizon=12, method="gmm")
print(ng_res.summary())

# 1c. Restricciones de signo y narrativas con cotas Giacomini-Kitagawa (2021)
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

## 2. Contrafactuales de Política Semi-Estructurales (`puremacro.lp.counterfactual`)

McKay y Wolf (2023, *REStud*) demuestran que cualquier regla lineal contrafactual
puede construirse combinando las funciones de impulso-respuesta (IRF) base con
una matriz de choques de política contemporáneos y de noticias. Barnichon y
Mesters (2020, *AER*; 2023, *JPE*) utilizan esas mismas respuestas para evaluar
si la regla de política observada minimizó la pérdida cuadrática (`opp_gradient == 0`).

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

## 3. Variables Instrumentales Granulares y VAR Funcional (`puremacro.inference.giv`, `puremacro.var.functional`)

```python
import numpy as np
from puremacro.inference.giv import estimate_giv

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

## 4. Sensibilidad Estructural ($\Lambda$) y Puentes Soberanos Empíricos

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
