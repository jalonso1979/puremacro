# Puente Empírico LP Distribucional ↔ Incidencia de Hogares HANK de un Activo

> 🇬🇧 [English](../hank_distributional_bridge.md) · 🇪🇸 Español

El módulo `puremacro.models.hank_distributional_bridge` (`puremacro/models/hank_distributional_bridge.py`) conecta las respuestas al impulso transversales por grupo de hogares de los solucionadores HANK de un activo (`puremacro.models.hank_fiscal` y `puremacro.models.hank_sequence_space`) con las proyecciones locales distribucionales y cuantílicas empíricas (`puremacro.lp.distributional`).

---

## 1. Propósito y arquitectura contable

Evaluar modelos de agentes heterogéneos únicamente con el consumo agregado $dC_t$ deja sin verificar la transmisión distributiva entre hogares. Este puente:

1. **Extrae trayectorias transversales por grupo** ($T \times G$) por cuantiles de riqueza líquida, hogares restringidos *hand-to-mouth* (HtM) frente a no restringidos, estados de productividad o particiones personalizadas $(a, s)$ bajo cierres `balanced_budget`, `debt_rule`, impuestos `proportional` o `lump_sum`, choques monetarios o transferencias directas de liquidez.
2. **Audita la conservación contable exacta (*fail-closed*)**: Comprueba que la suma ponderada por población de las IRF grupales reproduce con exactitud las sendas agregadas del modelo (`< 1e-10`) sin reescalamiento silencioso:
   $$\sum_{g=1}^{G} p_g \, dc_{t,g} = dC_t, \qquad \sum_{g=1}^{G} ds_{t,g}^C = 0, \qquad \sum_{g=1}^{G} p_g \, da_{t,g} = dA_t, \qquad \sum_{g=1}^{G} ds_{t,g}^A = 0.$$
   Los modelos no convergidos o con violaciones de conservación lanzan `ValueError`, salvo inspección explícita con `allow_unconverged=True` o `strict_conservation=False`.
3. **Alinea y califica frente a LP empíricas**: Contrasta las respuestas grupales del modelo HANK contra `DistributionalLPResult` (procedente de `estimate_distributional_lp` o `estimate_quantile_lp`) u objetivos `EmpiricalGroupTargets` / `DataFrame` / diccionarios suministrados por el usuario, reportando:
   - **RMSE**, **MAE**, sesgo, **concordancia de signo**, **estadísticos $t$** puntuales, prueba diagonal **$\chi^2$ de Wald** y **cobertura dentro de bandas de confianza** globales y por grupo.
   - **Contraste de gradiente de desigualdad y dispersión**: diferencia y cocientes entre el grupo de menor y mayor riqueza (`low_group - high_group` en impacto y acumulado, RMSE del gradiente y razón de dispersión transversal).
   - **Metadatos de calibración estacionaria** (`aggregate_mpc`, `htm_constrained_share`, `wealth_gini`, MPC por grupo, `top_grid_mass`) y registro estándar `SolverDiagnosticRecord` (`PASS`, `CONVERGED_INACCURATE`, `UNSUPPORTED_REGIME`, `FAILED`).
4. **Salida bilingüe**: Soporta `lang="en" | "es"` en `.summary()`, `.to_markdown()` y `.to_dict()`, además de `.to_frame()`, `.to_latex()`, `.to_typst()` y `.plot()`.

---

## 2. Esquemas de partición y modos de seguimiento dinámico

| Partición `groups=` | Grupos generados |
| :--- | :--- |
| `"quintiles"`, `"deciles"`, `"quartiles"`, `"terciles"` | Quintiles, deciles, cuartiles o terciles estacionarios de riqueza líquida (`Q1..Q5`, `D1..D10`, etc.) |
| `"htm_vs_unconstrained"` | `"HtM"` frente a `"Unconstrained"` (`htm_definition="constrained"` en $a_{ss} \le a_{\min}$ o `"bottom_quartile"`) |
| `"income"`, `"wealth_x_income"` | Estados de productividad o cruce riqueza × ingreso |
| Entero `G >= 2` o bordes `[0.0, ..., 1.0]` | Partición cuantílica arbitraria definida por el usuario |
| `Mapping[str, np.ndarray]` | Pesos de partición personalizados sobre `(n_a,)` o `(n_a, n_s)` con suma unitaria |

El seguimiento dinámico sobre $t = 0, \dots, T-1$ se controla mediante `tracking`:
- `"cohort"` *(por defecto)*: Sigue hacia adelante las cohortes iniciales del estado estacionario con masas fijas $p_g$; consistente tanto en transiciones no lineales (`FiscalHANKResult`, `NonlinearHANKResult`) como en perturbaciones lineales *Fake News* (`SequenceSpaceHANKResult`).
- `"cross_section"`: Recalcula los umbrales cuantílicos transversales en cada periodo sobre $D_t$ (solo modelos no lineales con partición por cuantiles).
- `"fixed_bins"`: Mantiene fijos los intervalos iniciales sobre la malla mientras la masa poblacional evoluciona con $D_t$.

---

## 3. Ejemplo ejecutable

```python
import numpy as np
import pandas as pd

from puremacro.lp.distributional import estimate_distributional_lp
from puremacro.models.hank_distributional_bridge import (
    bridge_hank_distributional,
    solve_hank_group_irf,
)

# 1. Resolver un recorte tributario de suma alzada financiado en HANK (G negativo)
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
print(hank_girf.summary(lang="es"))

# 2. Construir muestra sintética ilustrativa y estimar LP distribucionales
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

# 3. Alinear y calificar el modelo HANK frente a la evidencia empírica
bridge = bridge_hank_distributional(hank_girf, lp_res)
print(bridge.summary(lang="es"))
print(bridge.to_frame("per_group")[["rmse", "sign_concordance", "band_coverage", "steady_state_mpc"]])
```

---

## 4. Alcance explícito de validación y límites metodológicos

- **Alcance estacionario / transición MIT determinista y perturbación lineal**: Tanto `FiscalHANKResult` / `NonlinearHANKResult` (`deterministic_mit_transition`) como `SequenceSpaceHANKResult` (`linearized_sequence_space`) evalúan sendas deterministas con previsión perfecta o perturbaciones de primer orden alrededor de la distribución estacionaria de un activo. **No** resuelven equilibrios no lineales con riesgo agregado estocástico, primas de riesgo endógenas ni ahorro precautorio dependiente de volatilidad agregada.
- **Un solo activo líquido**: Los hogares acumulan únicamente un activo líquido ($a \ge 0$); el puente no incorpora recomposición entre activos líquidos e ilíquidos ni hogares *wealthy hand-to-mouth* de dos activos.
- **Compatibilidad de unidades con LP cuantílicas**: Dado que `estimate_quantile_lp` (`distribution_type="quantiles"`) estima respuestas condicionales por cuantil $dQ_\tau(y_{t+h} - y_{t-1})$ y no cuotas entre estratos, `response_type="auto"` alinea frente a niveles de consumo grupal (`"level"`) y emite una advertencia explícita; utilice `response_type="percent"` si los cuantiles empíricos están expresados en puntos logarítmicos/porcentuales.
- **Soporte superior de la malla de activos**: Mantenga un límite superior amplio (`a_max >= 100.0`). Un tope estrecho (`top_grid_mass > 1e-6`) trunca a los hogares más ricos y activa el estado `UNSUPPORTED_REGIME` en `solver_diagnostics`.
