> 🇬🇧 [English](../quickstart.md) · 🇪🇸 Español

# Guía de inicio rápido

Comience a utilizar `puremacro 2.0` en menos de 2 minutos. Todos los estimadores centrales operan sobre Python puro, NumPy, SciPy, Pandas y Matplotlib — sin compiladores de C, sin entornos de ejecución de Fortran y 100% compatibles con Pyodide y el navegador web.

---

## 1. Proyecciones Locales (LP) con visualización en 1 línea

Estime funciones de respuesta al impulso (FRI) directamente mediante regresiones predictivas de Jordà (2005):

```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# 1. Generación de datos sintéticos: respuesta del PIB a un choque de política
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
gdp = np.cumsum(0.7 * shock + 0.3 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": gdp, "shock": shock})

# 2. Ajuste de proyección local hasta el horizonte 12 con 4 retardos de control
res = lp_hac(df, y="gdp", x="shock", horizon=12, lags=4, ci=0.90)

# 3. Inspeccionar resumen y graficar la FRI con bandas de confianza
print(res.summary())
res.plot(title="Respuesta del PIB ante Choque Estructural")

# 4. Exportar la tabla directamente a LaTeX o Typst para su artículo
print(res.to_latex())
print(res.to_typst())
```

---

## 2. VAR Estructural (SVAR) con bandas bootstrap

Estime un VAR e identifique choques estructurales mediante Cholesky, restricciones de signo o variables instrumentales proxy:

```python
from puremacro.var.identify import cholesky_svar

# Sistema macroeconómico (T, 3): [PIB, Inflación, Tasa de Interés]
Y = rng.standard_normal((200, 3)).cumsum(axis=0)

# Estimar VAR(2) y calcular respuestas al impulso con bandas bootstrap al 90%
res_svar = cholesky_svar(Y, p=2, horizon=16, n_boot=500, ci=0.90, seed=42)

# Inspeccionar resumen y graficar la respuesta de la variable 0 al choque 0
print(res_svar.summary())
res_svar.plot(target_idx=0, shock_idx=0, title="Respuesta del Producto al Choque Monetario")

# Exportar a un DataFrame ordenado
df_irf = res_svar.to_frame(target_idx=0, shock_idx=0)
```

---

## 3. Diferencias en diferencias escalonadas modernas

Estime efectos de tratamiento robustos a la heterogeneidad bajo adopción escalonada, evitando los problemas de ponderación negativa del estimador clásico TWFE:

```python
from puremacro.did import callaway_santanna

# Efectos medios de tratamiento por grupo y tiempo de Callaway y Sant'Anna (2021)
res_did = callaway_santanna(
    panel_df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control_group="never_treated",
    n_boot=500,
    ci=0.95,
)

print(res_did.summary())
# Perfil dinámico de estudio de eventos relativo al momento de adopción
print(res_did.att_event_study.head())
print(res_did.to_markdown())
```

---

## 4. Modelo HANK en el espacio de secuencias

Resuelva un modelo nuevo keynesiano con mercados incompletos y riesgo de ingreso idiosincrásico no asegurable utilizando jacobianos en el espacio de secuencias (Auclert et al. 2021):

```python
from puremacro.models import solve_hank_sequence_space

# Resolver la trayectoria de transición de equilibrio general ante un alza de tasa de 25 pb
res_hank = solve_hank_sequence_space(T=40, beta=0.985, phi_pi=1.5, kappa=0.1)
print(res_hank.summary())

# Inspeccionar respuestas del producto y la inflación
print("Caída máxima del producto:", res_hank.irf_output.min())
print("Propensión marginal a consumir (PMC) del primer decil:", res_hank.mpc_distribution["Decile 1"])
```

---

## 5. Nowcasting del PIB con frecuencias mixtas

Monitoree el crecimiento trimestral del PIB en tiempo real a partir de indicadores mensuales con bordes irregulares y retrasos de publicación:

```python
from puremacro.nowcast import nowcast_gdp

res_nowcast = nowcast_gdp(monthly_indicators_df, historical_gdp_series, n_factors=2)
print(res_nowcast.summary())

# Ver la contribución de noticias de la última publicación
print(res_nowcast.news_decomposition)
```

---

## 6. VAR aumentado con factores (FAVAR)

Extraiga factores latentes de paneles informacionales de alta dimensión y proyecte las respuestas de política a series macroeconómicas individuales (Bernanke, Boivin y Eliasz 2005):

```python
from puremacro.var import favar

favar_res = favar(
    panel_macro_df,
    policy_rate_series,
    n_factors=3,
    p=2,
    horizon=20,
    ci=0.90,
)
print(favar_res.summary())

# Graficar respuestas para variables macroeconómicas específicas
favar_res.plot(variables=["Industrial_Production", "CPI", "Employment"])
```

---

## 7. Aproximación de orden superior en DSGE y paridad con Dynare

Resuelva modelos DSGE no lineales hasta primer o segundo orden con poda (*pruning*) de Kim et al. (2008), derivadas cruzadas y compatibilidad con `oo_.dr` de Dynare:

```python
from puremacro.dsge import build_dynare, load_mod

# 1. Cargar archivo .mod nativo de Dynare con choques y opciones de stoch_simul
model = load_mod("rbc.mod")

# 2. Resolver perturbación de segundo orden con poda
sol = model.solve(order=2)

# 3. Inspeccionar reglas de política con la misma estructura oo_.dr de Dynare
print(sol.oo_dr["ghx"])   # derivadas de estado de primer orden
print(sol.oo_dr["ghxx"])  # derivadas de estado de segundo orden
print(sol.oo_dr.summary())

# 4. Momentos teóricos analíticos y descomposición de varianza
mom = sol.theoretical_moments()
print(mom.summary())
print(mom.to_latex())
```

---

## 8. Descarga de cómputo desde iPad / Juno / Pyodide a Google Colab

Cuando trabaje en una tableta o sesión de Pyodide con limitaciones de memoria o tiempo de CPU, genere un cuaderno ejecutable de Google Colab con sincronización automática de resultados a Google Drive:

```python
from puremacro.runtime.colab import (
    generate_colab_notebook,
    show_colab_offload_dialog,
    load_colab_result,
)

# 1. Generar cuaderno con autenticación y montaje de Google Drive
nb = generate_colab_notebook(
    task_code="""
import puremacro as pm
res = pm.dsge.estimate_sw07(n_draws=10000, n_chains=4)
pm.runtime.store.save_frame(res.summary(), "sw07_posterior.pmz")
""",
    mount_drive=True,
    export_result_file="sw07_posterior.pmz",
)

# 2. Abrir en Colab con 1 clic
show_colab_offload_dialog(nb, filename="sw07_offload.ipynb")

# 3. Recuperar el resultado en la sesión local mediante el cartucho puro .pmz
res = load_colab_result("sw07_posterior.pmz")
```
