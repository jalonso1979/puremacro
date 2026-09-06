# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Herramientas de Frontera Dynare — puremacro 2.0 y 2.1
#
# **¿Podemos resolver, simular y estimar modelos macroeconómicos DSGE avanzados completamente en Python puro, sin MATLAB ni compilación C++?**
#
# Dynare ha sido durante dos décadas el estándar para la modelización macroeconómica DSGE. Sin embargo, su flujo de trabajo tradicional requiere licencias comerciales de MATLAB o instalaciones complejas de C++ MEX.
#
# Con **puremacro 2.0 y 2.1**, investigadores y estudiantes pueden cargar archivos `.mod` de Dynare directamente, calcular reglas de decisión lineales y de segundo orden con poda (*pruning*), descomposición de varianza de errores de pronóstico (FEVD), descomposición histórica de shocks mediante suavizador de Kalman, restricciones ocasionalmente activas (OccBin / Zero Lower Bound), simulaciones no lineales deterministas (*perfect foresight*) y estimación Bayesiana MCMC completa—todo en **Python puro**, compatible con navegadores WebAssembly (JupyterLite) y tablets sin instalación previa.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.dsge import (
    load_mod,
    build_dynare,
    compute_fevd,
    compute_shock_decomposition,
    solve_occbin,
    OccBinConstraint,
    solve_perfect_foresight,
    estimate_dsge_bayesian,
    BetaPrior,
    InvGammaPrior,
)

# %% [markdown]
# ## 1. Carga y Resolución del Modelo de Smets y Wouters (2007) (.mod)
#
# Cargamos directamente el archivo canonical de Johannes Pfeifer para el modelo de Smets & Wouters (2007, *AER*) (`sw07_pfeifer.mod`).
# `puremacro.dsge.load_mod` procesa todas las ecuaciones, variables predeterminadas, estados estacionarios y bloques de shocks, identificando automáticamente los 15 estados y 25 variables de salto.

# %%
import puremacro.dsge
# Resolve the reference .mod from the installed package so the notebook runs
# from any working directory (tools/build_notebooks.py uses notebooks/ as cwd).
mod_path = Path(puremacro.dsge.__file__).parent / "_references" / "sw07_pfeifer.mod"
m = load_mod(mod_path, order=1)

print(f"Variables endógenas  : {len(m.variables)}")
print(f"Shocks estructurales : {len(m.shocks)}")
print(f"Estados predeterminados: {m.n_states}")
print(f"Variables de salto   : {m.n_controls}")

# Ejecutamos stoch_simul para obtener reglas de decisión, momentos teóricos e IRFs
sim_res = m.stoch_simul(irf=24)
fig_irfs = sim_res.plot(variables=["labobs", "robs", "pinfobs", "dy"], shocks=["ea", "em"])
plt.show()

# %% [markdown]
# ## 2. Descomposición de Varianza del Error de Pronóstico (FEVD)
#
# La descomposición FEVD cuantifica el porcentaje de la varianza del error de predicción atribuible a cada perturbación estructural en horizontes $h \in \{1, 4, 8, 16, \dots, \infty\}$.
#
# Mediante la representación de medias móviles ortogonalizadas:
# $$ y_{t+h} - \mathbb{E}_t y_{t+h} = \sum_{k=0}^{h-1} \Psi_k u_{t+h-k} $$
# Puremacro garantiza numéricamente con precisión de máquina que la suma de participaciones es exactamente igual a 1.0 (100%) para cada variable y horizonte.

# %%
fevd_res = compute_fevd(m, horizons=[1, 4, 8, 16, 32, None])
print(fevd_res.summary())

fig_fevd = fevd_res.plot(variables=["labobs", "robs", "pinfobs", "dy"])
plt.show()

# %% [markdown]
# ## 3. Descomposición Histórica de Shocks
#
# ¿Qué perturbaciones estructurales causaron las fluctuaciones observadas del ciclo económico?
# Mediante el suavizador de Kalman (*Kalman smoother*), puremacro reconstruye la trayectoria histórica de cada serie como la suma de:
# 1. Estado estacionario $\bar{y}$
# 2. Decaimiento de la condición inicial $C A^t s_0$
# 3. Contribución acumulada de cada shock estructural $\sum_j \text{Shock}_j(t)$

# %%
np.random.seed(42)
T_hist = 40
data_hist = pd.DataFrame({
    "labobs": np.sin(np.linspace(0, 3 * np.pi, T_hist)) * 1.5 + np.random.randn(T_hist) * 0.2,
    "robs": np.cos(np.linspace(0, 2 * np.pi, T_hist)) * 0.8 + np.random.randn(T_hist) * 0.1,
    "pinfobs": np.sin(np.linspace(0, 2.5 * np.pi, T_hist)) * 0.5 + np.random.randn(T_hist) * 0.15,
    "dy": np.random.randn(T_hist) * 0.6,
})

decomp_res = compute_shock_decomposition(m, data_hist)
print(f"Variables descompuestas: {decomp_res.variable_names}")

fig_decomp = decomp_res.plot(variable="labobs")
plt.show()

# %% [markdown]
# ## 4. Restricciones Ocasionalmente Activas y Trampa de Liquidez (OccBin)
#
# Cuando el tipo de interés nominal alcanza el límite inferior cero (*Zero Lower Bound*, $r_t \ge -r_{ss}$), la aproximación lineal habitual no es válida.
# Siguiendo el algoritmo de Guerrieri & Iacoviello (2015, *JME*), `puremacro.dsge.solve_occbin` resuelve el modelo lineal a trozos mediante iteración regresiva de regímenes.

# %%
params_nk = {
    "beta": 0.99,
    "sigma": 1.0,
    "kappa": 0.1,
    "phi_pi": 1.5,
    "phi_y": 0.125,
    "rho_g": 0.8,
    "r_ss": 0.01,
}
variables_nk = ["y", "pi", "r", "g"]
shocks_nk = ["eps_r", "eps_g"]
ss_nk = {v: 0.0 for v in variables_nk}

def nk_ref(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]

def nk_cons(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (-p.r_ss),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]

ref_mod = build_dynare(nk_ref, variables=variables_nk, shocks=shocks_nk, params=params_nk, steady_state=ss_nk)
cons_mod = build_dynare(nk_cons, variables=variables_nk, shocks=shocks_nk, params=params_nk, steady_state=ss_nk, check_steady_state=False)

constraint = OccBinConstraint(variable="r", threshold=-params_nk["r_ss"], operator="<")
shock_seq = np.array([0.0, -0.020])

occ_res = solve_occbin(ref_mod, cons_mod, constraint, shock_sequence=shock_seq, horizon=40)
print(occ_res.summary())

fig_occ = occ_res.plot()
plt.show()

# %% [markdown]
# ## 5. Simulación No Lineal Determinista (Previsión Perfecta / Perfect Foresight)
#
# Para grandes transiciones lejos del estado estacionario (por ejemplo, convergencia económica, reformas fiscales o transiciones energéticas), las perturbaciones locales pierden validez.
# `puremacro.dsge.solve_perfect_foresight` aplica el método de Newton-Raphson apilado (*stacked solver*, Boucekkine 1995, Juillard 1996) con inversión dispersa por bloques.

# %%
alpha, beta, delta, sigma = 0.33, 0.96, 0.08, 1.0
k_ss = ((1.0 / beta - (1.0 - delta)) / alpha) ** (1.0 / (alpha - 1.0))
c_ss = k_ss ** alpha - delta * k_ss
y_ss = np.array([c_ss, k_ss])

def ramsey_eqs(lead, curr, lag, exo):
    c_t, k_t = curr[0], curr[1]
    c_p, k_p = lead[0], lead[1]
    k_m = lag[1]
    A_t = exo[0]
    
    euler = c_t ** (-sigma) - beta * (c_p ** (-sigma)) * (alpha * A_t * (k_t ** (alpha - 1.0)) + 1.0 - delta)
    resource = k_t - (A_t * (k_m ** alpha) + (1.0 - delta) * k_m - c_t)
    return np.array([euler, resource])

y_init = np.array([c_ss * 0.7, 0.5 * k_ss])
exo_path = np.ones((60, 1))

pf_res = solve_perfect_foresight(ramsey_eqs, y_init=y_init, y_ss=y_ss, exogenous_path=exo_path, n_periods=60)
print(pf_res.summary())

fig_pf = pf_res.plot()
plt.show()

# %% [markdown]
# ## 6. Estimación Bayesiana DSGE vía Metropolis-Hastings
#
# `puremacro.dsge.estimate_dsge_bayesian` ejecuta el protocolo Bayesiano estándar de estimación:
# 1. Búsqueda numérica de la moda posterior.
# 2. Aproximación de Laplace para la covarianza de la propuesta $\Sigma = (-H)^{-1}$.
# 3. Muestreo MCMC Random Walk Metropolis-Hastings con adaptación automática de escala y diagnóstico de convergencia Gelman-Rubin.

# %%
true_rho, true_sigma = 0.85, 0.02
np.random.seed(42)
y_obs = np.zeros(100)
for t in range(1, 100):
    y_obs[t] = true_rho * y_obs[t-1] + np.random.randn() * true_sigma

def log_lik_fn(theta):
    rho, sig = theta[0], theta[1]
    if not (0.01 < rho < 0.99) or sig <= 0.001:
        return -1e10
    resids = y_obs[1:] - rho * y_obs[:-1]
    return float(-0.5 * len(resids) * np.log(2 * np.pi * sig**2) - 0.5 * np.sum(resids**2) / (sig**2))

priors = {
    "rho": BetaPrior(mean=0.8, std=0.1),
    "sigma": InvGammaPrior(s=0.02, nu=4.0),
}

bayes_res = estimate_dsge_bayesian(
    log_lik_fn,
    priors=priors,
    initial_params=np.array([0.7, 0.04]),
    n_draws=400,
    n_burn=100,
    n_chains=2,
    seed=42,
)

print(bayes_res.summary())
fig_bayes = bayes_res.plot_priors_posteriors()
plt.show()

# %% [markdown]
# ## Conclusiones
#
# Con `puremacro`:
# 1. **Cero dependencias de MATLAB / C++**: Resolución y estimación de modelos DSGE directamente en Python.
# 2. **Paridad total con Dynare**: Carga de archivos `.mod`, reglas de decisión, momentos teóricos, FEVD, descomposición de shocks, OccBin y previsión perfecta.
# 3. **Ejecución universal**: Funciona sin modificaciones en Apple Silicon, Linux, Windows y navegadores con JupyterLite.
