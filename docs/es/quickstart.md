> 🇬🇧 [English](../quickstart.md) · 🇪🇸 Español

# Guía de inicio rápido

Comience con el ejemplo de proyecciones locales y luego elija un modelo o estimador.
El núcleo numérico utiliza NumPy, SciPy, pandas y Matplotlib, sin extensiones
compiladas propias. El destino soportado es una instalación local en una estación
de trabajo:

```bash
python -m pip install puremacro
```

La ejecución en el navegador bajo Pyodide tiene soporte limitado. La
[guía de ejecución](tablet.md) explica los límites de formatos de archivo, red y
cómputo; los [itinerarios de cuadernos](notebooks.md) ofrecen ejemplos más amplios.

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
from puremacro.dsge import estimate_sw07
result = estimate_sw07(n_draws=10000, n_chains=4)
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

---

## 9. Diferenciación automática directa y protocolo unificado de modelos

Evalúe gradientes nativos de Kalman con datos explícitos y simule modelos de espacio de estados mediante el protocolo `DifferentiableModel`:

```python
import numpy as np
from puremacro.core.protocol import wrap_differentiable_model, DifferentiableModel
from puremacro.state_space import StateSpaceModel

# 1. Declare state-space system and wrap into universal protocol adapter
ssm = StateSpaceModel(
    T=np.array([[0.75]]),
    Z=np.array([[1.0]]),
    Q=np.array([[0.04]]),
    H=np.array([[0.0025]]),
)
model_diff = wrap_differentiable_model(ssm)

# 2. Evaluate the native likelihood and Kalman sensitivity score on explicit data
params = np.array([0.75, 0.20, 0.05])
data = model_diff.simulate(periods=100, seed=42)
loglik, score = model_diff.loglik_and_grad(params=params, data=data)
print(f"Log-likelihood: {loglik:.4f}")
print("Analytical score gradient:", score)

# 3. Simulate forward path and compute structural impulse responses
sim_df = model_diff.simulate(periods=30, seed=42)
irf_df = model_diff.impulse_response(shock="eps", horizon=12)
print(f"IRF length: {len(irf_df)}, Simulation length: {len(sim_df)}")
```

---

## 10. Monte Carlo secuencial (SMC) templado y adaptativo

Estime distribuciones posteriores macroeconómicas y la densidad marginal de datos ($\ln p(Y)$) utilizando mutaciones MALA guiadas por gradientes:

```python
import numpy as np
from puremacro.inference.smc import smc_sample

# 1. Target log-likelihood function and analytical score
def loglik_and_grad(theta):
    diff = theta[0] - 0.45
    val = -0.5 * (diff / 0.04) ** 2
    grad = np.array([-diff / (0.04 ** 2)])
    return float(val), grad

# 2. Prior specification and adaptive tempered SMC sampling
priors = {"rho": {"dist": "uniform", "lb": -0.5, "ub": 0.95}}
res_smc = smc_sample(loglik_and_grad, priors, n_particles=200, mutation="mala", seed=42)

# 3. Log marginal data density and summary table
print(f"Log MDD: {res_smc.log_mdd:.4f} +/- {res_smc.mdd_se:.4f}")
print(res_smc.summary())
print(res_smc.to_latex())
```

---

## 11. Demostraciones HANK reducidas no lineales y de riesgo agregado

Ejecute un DAG reducido con PMC y fracciones restringidas prescritas, y un ejemplo algebraico de riesgo agregado. No resuelven políticas ni distribución de hogares; consulte [el alcance HANK](hank_ssj_and_aggregate_risk.md):

```python
import numpy as np
from puremacro.dsge import (
    solve_hank_nonlinear_transition,
    solve_hank_aggregate_risk_order2,
)

# 1. Solve nonlinear HANK transition path to an interest rate hike
T_hank = 30
shock_path = 0.02 * (0.65 ** np.arange(T_hank))
res_trans = solve_hank_nonlinear_transition(
    shocks={"eps_m": shock_path},
    horizon=T_hank,
    tol=1e-8,
    max_iter=30,
)
print("Transition converged:", res_trans.converged)
print(res_trans.summary())

# 2. Compute 2nd-order aggregate risk with precautionary shifts
res_risk = solve_hank_aggregate_risk_order2(
    sigma_agg=0.015,
    rho_agg=0.85,
    n_distribution_moments=4,
)
print("Precautionary wealth shift:", round(float(res_risk.precautionary_shift_wealth), 6))
print("Precautionary rate discount:", round(float(res_risk.precautionary_shift_rate), 6))
```

---

## 12. Default soberano cuantitativo y reestructuración de deuda

Calcule el riesgo de default endógeno y la curva de precios de bonos en el marco de Arellano (2008) y evalúe el alivio por reestructuración:

```python
import numpy as np
from puremacro.sovereign import (
    solve_sovereign_default,
    evaluate_debt_restructuring,
)

# 1. Solve sovereign default equilibrium with endogenous borrowing limit
res_sov = solve_sovereign_default(
    delta=1.0,
    kappa=0.0,
    r_star=0.01,
    beta=0.95,
    gamma=2.0,
    n_b=25,
    n_y=9,
    max_iter=150,
)
print("Sovereign solve converged:", res_sov.converged)
print(f"Mean spread: {res_sov.spread_mean * 10000:.1f} bps")

# 2. Evaluate a haircut at an explicit state; keep the short-term maturity
res_restr = evaluate_debt_restructuring(
    res_sov,
    haircut=0.20,
    maturity_extension=0.0,
    b_distress=0.20,
    y_distress=1.0,
    horizon=20,
    seed=42,
)
print(f"Spread reduction: {res_restr.spread_reduction_bps:.1f} bps")
print(f"Debt-to-GDP relief: {res_restr.debt_gdp_relief_pct:.2f}%")
```

---

## 13. Proyecciones locales distribucionales con suma exacta

Estime respuestas dinámicas de participaciones de deciles ante choques macroeconómicos respetando estrictamente la restricción de suma cero ($\sum \beta_b = 0$):

```python
import numpy as np
import pandas as pd
from puremacro.lp.distributional import (
    estimate_distributional_lp,
    estimate_quantile_lp,
)

# 1. Synthetic cross-sectional shares (5 quintiles summing to 1.0)
rng_dist = np.random.default_rng(42)
T_dist = 120
n_bins = 5
bin_cols = [f"Q{i+1}" for i in range(n_bins)]
raw_shares = rng_dist.dirichlet(np.ones(n_bins) * 2.0, size=T_dist)
df_dist = pd.DataFrame(raw_shares, columns=bin_cols)
df_dist["dist_shock"] = rng_dist.standard_normal(T_dist)
df_dist["gdp_growth"] = rng_dist.standard_normal(T_dist) * 0.5
df_dist["income"] = 10.0 + 2.0 * rng_dist.standard_normal(T_dist)

# 2. Fit Distributional Local Projection and verify exact adding-up
res_dlp = estimate_distributional_lp(
    df_dist,
    shock="dist_shock",
    bin_cols=bin_cols,
    horizons=range(0, 8),
    n_lags=2,
    controls=["gdp_growth"],
)
print("Adding-up maximum error:", res_dlp.adding_up_max_error)
print(res_dlp.summary())

# 3. Fit Quantile Local Projection (10th, 50th, 90th percentiles)
res_qlp = estimate_quantile_lp(
    df_dist,
    shock="dist_shock",
    y="income",
    quantiles=[0.10, 0.50, 0.90],
    horizons=range(0, 8),
    n_lags=1,
)
print("Quantile IRF matrix shape:", res_qlp.irf_matrix.shape)
```

---

## 14. IAM climático-macroeconómico espacial y PolicyStudio sin servidor

Analice cuellos de botella en redes de producción, migración climática multirregional bajo impuestos al carbono y exporte paneles interactivos sin servidor:

```python
import numpy as np
from puremacro.trade import baqaee_farhi_decomposition
from puremacro.climate.spatial_iam import SpatialClimateIAM, solve_spatial_climate_iam
from puremacro.reports import result_to_interactive_html

# 1. Baqaee-Farhi 2nd-order production network bottleneck
Omega = np.array([[0.10, 0.05], [0.30, 0.10]], dtype=float)
b = np.array([0.40, 0.60], dtype=float)
res_net = baqaee_farhi_decomposition(
    Omega,
    b,
    elasticities=0.25,
    tfp_shocks=np.array([-0.05, 0.0]),
)
print("1st order effect:", round(float(res_net.first_order_dlog_y), 5))
print("2nd order curvature:", round(float(res_net.second_order_dlog_y), 5))

# 2. Multi-region Spatial Climate IAM with endogenous migration
iam_model = SpatialClimateIAM(
    initial_population=np.array([100.0, 150.0]),
    initial_trade_shares=np.array([[0.8, 0.2], [0.2, 0.8]]),
    initial_migration_shares=np.array([[0.95, 0.05], [0.05, 0.95]]),
    initial_temperatures=np.array([12.0, 26.0]),
    carbon_intensity=np.array([0.3, 0.7]),
    damage_coefficients=np.array([0.005, 0.02]),
    region_names=("North", "South"),
)
res_iam = solve_spatial_climate_iam(iam_model, horizon=10, policy_scenario="laissez_faire")
print("Population conservation error:", res_iam.population_conservation_error)

# 3. Zero-server self-contained HTML5 dashboard export
html_doc = result_to_interactive_html(res_net, title="Supply Chain Bottleneck Analysis")
print("HTML dashboard generated bytes:", len(html_doc))
```
