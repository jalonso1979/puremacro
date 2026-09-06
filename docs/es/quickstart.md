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
import numpy as np
import pandas as pd
from puremacro.did import callaway_santanna

# Adopción escalonada: 60 condados observados 2000-2011, cohortes tratadas por
# primera vez en 2005 y 2008 más un grupo nunca tratado (sustitúyalo por su panel)
rng = np.random.default_rng(0)
rows = []
for county in range(60):
    g = {0: 2005, 1: 2008, 2: np.nan}[county % 3]
    for year in range(2000, 2012):
        effect = 3.0 if (not np.isnan(g) and year >= g) else 0.0
        rows.append({"county_id": county, "year": year, "first_treated_year": g,
                     "employment": 100 + 0.5 * (year - 2000) + effect + rng.standard_normal()})
panel_df = pd.DataFrame(rows)

# ATT grupo-tiempo de Callaway y Sant'Anna (2021)
res_did = callaway_santanna(
    panel_df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control="never_treated",
    n_boot=200,
    ci=0.95,
)

print(res_did.summary())
# Perfil dinámico del estudio de eventos relativo al momento de adopción
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
import numpy as np
import pandas as pd
from puremacro.nowcast import nowcast_gdp

# Diez años de seis indicadores mensuales movidos por un factor común y la
# historia trimestral del PIB; el último mes del trimestre en curso está
# publicado solo en parte (borde irregular)
rng = np.random.default_rng(1)
months = pd.date_range("2016-01-31", periods=120, freq="ME")
factor = np.cumsum(rng.standard_normal(120)) * 0.3
monthly_indicators_df = pd.DataFrame(
    np.outer(factor, rng.uniform(0.5, 1.5, 6)) + 0.5 * rng.standard_normal((120, 6)),
    index=months, columns=["ip", "retail", "orders", "hours", "pmi", "exports"],
)
monthly_indicators_df.iloc[-1, 3:] = np.nan            # aún no publicado
quarters = pd.period_range("2016Q1", periods=39, freq="Q")
historical_gdp_series = pd.Series(
    factor.reshape(-1, 3).mean(axis=1)[:39] + 0.2 * rng.standard_normal(39), index=quarters, name="gdp",
)

res_nowcast = nowcast_gdp(monthly_indicators_df, historical_gdp_series, n_factors=2)
print(res_nowcast.summary())
print(res_nowcast.to_frame().tail())
```

---

## 6. VAR aumentado con factores (FAVAR)

Extraiga factores latentes de paneles informacionales de alta dimensión y proyecte las respuestas de política a series macroeconómicas individuales (Bernanke, Boivin y Eliasz 2005):

```python
import numpy as np
import pandas as pd
from puremacro.var import favar

# Un panel informativo (T x N) y la tasa de política, simulados para el ejemplo
rng = np.random.default_rng(2)
T = 240
f = np.cumsum(rng.standard_normal(T)) * 0.1
names = ["Industrial_Production", "CPI", "Employment"] + [f"x{i}" for i in range(9)]
panel_macro_df = pd.DataFrame(
    np.outer(f, rng.uniform(0.5, 1.5, len(names))) + 0.3 * rng.standard_normal((T, len(names))),
    columns=names,
)
policy_rate_series = pd.Series(0.5 * f + 0.2 * rng.standard_normal(T), name="policy_rate")

favar_res = favar(
    panel_macro_df,
    policy_rate_series,
    n_factors=3,
    p=2,
    horizon=20,
    ci=0.90,
    n_boot=50,
)
print(favar_res.summary())

# Graficar las respuestas de variables macroeconómicas concretas
favar_res.plot(variables=["Industrial_Production", "CPI", "Employment"])
```

---

## 7. Aproximación de orden superior en DSGE y paridad con Dynare

Resuelva modelos DSGE no lineales hasta primer o segundo orden con poda (*pruning*) de Kim et al. (2008), derivadas cruzadas y compatibilidad con `oo_.dr` de Dynare:

```python
from puremacro.dsge import load_mod

# 1. Un archivo .mod de Dynare: una ruta o, como aquí, el propio texto
rbc_mod = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;

alpha = 0.30;
beta  = 0.99;
delta = 0.025;
gamma = 1.0;
rho   = 0.80;

model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
end;

initval;
  k = 38.0;
  a = 0.0;
  c = 2.0;
end;

shocks;
  var eps; stderr 0.01;
end;
"""
model = load_mod(rbc_mod)   # LinearModel de primer orden (load_mod(rbc_mod, order=2) va directo al segundo orden)

# 2. Resolver la perturbación de segundo orden con poda
sol = model.solve(order=2)

# 3. Reglas de decisión al estilo Dynare (oo_.dr)
print(sol.oo_dr["ghx"])   # transición de estados de primer orden
print(sol.oo_dr["ghxx"])  # curvatura de segundo orden
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

# 1. Empaqueta la tarea pesada en un cuaderno autocontenido (con celdas de autenticación
#    y montaje de Drive). La variable `result` de la tarea se exporta como cartucho .pmz.
nb = generate_colab_notebook(
    """
import puremacro as pm
result = pm.dsge.estimate_sw07(n_draws=10000, n_chains=4)
""",
    mount_drive=True,
    save_path="sw07_offload.ipynb",
    output_filename="sw07_posterior.pmz",
)

# 2. Muestra las instrucciones (tarjeta HTML en Juno / Jupyter, texto en la terminal)
show_colab_offload_dialog("sw07_offload.ipynb")

# 3. Cuando el cartucho vuelva desde Google Drive, cárgalo en la sesión local:
#    posterior = load_colab_result("sw07_posterior.pmz")
```
