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
# # Estimación de Modelos DSGE mediante el Muestreador No-U-Turn (NUTS) y Gradientes Analíticos Exactos de Verosimilitud
#
# **¿Cómo pueden los macroeconomistas computacionales escalar la estimación bayesiana MCMC de modelos DSGE a altas dimensiones sin sucumbir a la maldición de la dimensionalidad ni depender de diferencias finitas lentas e inestables?**
#
# Durante más de dos décadas, la estimación empírica de modelos de Equilibrio General Dinámico y Estocástico (DSGE)—desde Lubik y Schorfheide (2004) hasta Smets y Wouters (2007)—ha dependido de manera casi exclusiva del algoritmo de paseo aleatorio Metropolis-Hastings (RWMH). Aunque conceptualmente simple, RWMH es inherentemente difusivo: las propuestas gaussianas isotrópicas saltan a ciegas por el espacio de parámetros, obligando a reducir la escala del salto a $O(D^{-1/2})$ para evitar el colapso de la tasa de aceptación. En consecuencia, recorrer la distribución posterior requiere $O(D^2)$ pasos, generando una autocorrelación extrema, cadenas bloqueadas y fallos graves de exploración en modelos de media y gran escala ($D \ge 20$) con valles estrechos y geometrías en forma de embudo.
#
# El algoritmo Hamiltonian Monte Carlo (HMC) supera esta limitación fundamental incorporando momentos auxiliares y simulando la física de una partícula sin fricción que recorre los contornos de energía de la distribución posterior. Guiado por el campo vectorial del gradiente $\nabla_\theta \log p(\theta \mid Y)$, HMC atraviesa masas de probabilidad de alta dimensión con una eficiencia que escala como $O(D^{1/4})$, generando propuestas lejanas e incorrelacionadas con una tasa de aceptación cercana al 100%.
#
# No obstante, aplicar HMC a modelos macroeconómicos DSGE enfrentaba históricamente dos barreras insuperables:
# 1. **Cálculo de Gradientes**: Evaluar el gradiente de la verosimilitud a través del filtro de Kalman exigía re-resolver el modelo y re-filtrar las observaciones reiteradamente mediante diferencias finitas numéricas ($2K$ o $4K$ evaluaciones completas del filtro por gradiente), un proceso computacionalmente prohibitivo y susceptible a errores de truncamiento y cancelación numérica.
# 2. **Calibración de la Longitud de Trayectoria**: El HMC estándar requiere sintonizar manualmente el número de pasos de integración leapfrog $L$ y el tamaño de paso $\epsilon$. Un $L$ insuficiente produce paseos aleatorios difusivos, mientras que un $L$ excesivo hace que la partícula retroceda sobre sus propios pasos ("U-turns"), desperdiciando cálculo.
#
# Con **puremacro 3.0.0**, todo el flujo bayesiano basado en gradientes queda unificado en **Python 100% puro** bajo el contrato de cuatro paquetes de Pyodide (`numpy`, `scipy`, `pandas`, `matplotlib`), sin requerir MATLAB ni compiladores C++:
# - **Recursión hacia adelante del Score de Kalman en un solo paso**: Gradientes analíticos exactos de log-verosimilitud $\nabla_\theta \log L(Y \mid \theta)$ mediante diferenciación implícita de las reglas de decisión (solucionador de Sylvester generalizado con descomposición de Schur compleja) y diferenciación de la ecuación discreta de Lyapunov.
# - **El Muestreador No-U-Turn (NUTS)**: Finalización autónoma de trayectorias mediante el criterio de giro en U generalizado de Betancourt (2017), adaptación del paso por promedio dual de Hoffman-Gelman (2014) con objetivo $\delta^* = 0.80$ y adaptación diagonal de la matriz de masa mediante acumuladores en línea de Welford.
# - **Conjunto Completo de Diagnósticos MCMC**: Split-$\hat{R}$, tamaño muestral efectivo (ESS) central y de colas, fracción bayesiana de información perdida de energía (E-BFMI), conteo de divergencias hamiltonianas y exportación de reportes a Markdown, LaTeX y Typst.

# %% [markdown]
# ## El método en matemáticas — Geometría del Conjunto Típico, Flujo Simpléctico y Score de Kalman
#
# ### 1. La Maldición de la Dimensionalidad y el Conjunto Típico
# En espacios probabilísticos de alta dimensión $\mathbb{R}^D$, la geometría del volumen desafía la intuición geométrica habitual. Considérese una distribución objetivo $p(\theta)$. La masa de probabilidad en un cascarón esférico diferencial de radio $r = \|\theta\|$ es el producto de la densidad y el volumen de la superficie esférica:
# $$ dP(r) = p(r) \cdot S_{D-1}(r) \, dr \propto p(r) \cdot r^{D-1} \, dr $$
# Para una distribución normal estándar $\mathcal{N}(0, I_D)$, $p(r) \propto \exp(-r^2 / 2)$, de modo que la distribución radial de masa es:
# $$ \frac{dP(r)}{dr} \propto r^{D-1} \exp\left(-\frac{r^2}{2}\right) $$
# Derivando con respecto a $r$, se comprueba que la masa radial alcanza su máximo en:
# $$ r^* = \sqrt{D - 1} \approx \sqrt{D} $$
# Esto engendra una paradoja crucial:
# - En el modo ($r = 0$), la densidad $p(\theta)$ es máxima, pero el elemento de volumen diferencial $r^{D-1}$ es cero: hay prácticamente **cero masa de probabilidad** en el modo.
# - En las colas lejanas ($r \gg \sqrt{D}$), el volumen es inmenso ($r^{D-1} \to \infty$), pero la densidad cae exponencialmente a cero.
# - Casi toda la masa de probabilidad se concentra en una cáscara hiper-esférica delgada de radio $\sqrt{D}$ y espesor $O(1)$, denominada el **conjunto típico** (*typical set*).
#
# **Por qué falla Metropolis-Hastings de paseo aleatorio**: Un salto aleatorio $\theta^* = \theta + \sigma \xi$ con $\xi \sim \mathcal{N}(0, I_D)$ apunta en una dirección arbitraria que casi con certeza abandona esta cáscara delgada. Hacia el interior, el volumen es despreciable; hacia el exterior, la verosimilitud se anula. Para mantener una tasa de aceptación razonable ($\approx 23.4\%$), la dispersión debe reducirse a $\sigma = O(D^{-1/2})$, transformando la exploración en una difusión lenta que requiere $O(D^2)$ iteraciones para explorar la distribución posterior.
#
# ### 2. Dinámica Hamiltoniana en el Espacio de Fases
# Hamiltonian Monte Carlo elimina la difusión pasiva transformando la inferencia posterior en un sistema dinámico físico. Ampliamos el espacio de parámetros $\theta \in \mathbb{R}^D$ (posición) con momentos auxiliares $p \sim \mathcal{N}(0, M)$. El Hamiltoniano total (energía conservada) es:
# $$ H(\theta, p) = U(\theta) + K(p) = -\log p(\theta \mid Y) + \frac{1}{2} p' M^{-1} p $$
# El estado conjunto evoluciona según las ecuaciones de movimiento de Hamilton:
# $$ \frac{d\theta}{dt} = \frac{\partial H}{\partial p} = M^{-1} p, \qquad \frac{dp}{dt} = -\frac{\partial H}{\partial \theta} = \nabla_\theta \log p(\theta \mid Y) $$
# Dado que la energía total se conserva a lo largo del flujo:
# $$ \frac{dH}{dt} = \frac{\partial H}{\partial \theta}' \frac{d\theta}{dt} + \frac{\partial H}{\partial p}' \frac{dp}{dt} = (-\dot{p})' (M^{-1} p) + (\dot{\theta})' (\nabla_\theta \log p) = 0 $$
# y el volumen en el espacio de fases se preserva en virtud del teorema de Liouville ($|\det \mathcal{J}| = 1$), las trayectorias se deslizan suavemente a lo largo de las curvas de iso-probabilidad del conjunto típico. Una propuesta tras una longitud de trayectoria $L$ tiene una probabilidad teórica de aceptación $\min(1, \exp(-\Delta H)) = 1.0$, escalando como $O(D^{1/4})$.
#
# ### 3. Integrador Simpléctico Leapfrog (Salto de Rana)
# La integración numérica debe preservar la 2-forma simpléctica $d\theta \wedge dp$ y el volumen del espacio de fases. El esquema clásico *leapfrog* actualiza el momento medio paso, avanza la posición un paso completo y finaliza el momento:
# $$ p\left(t + \frac{\epsilon}{2}\right) = p(t) + \frac{\epsilon}{2} \nabla_\theta \log p(\theta(t) \mid Y) $$
# $$ \theta(t + \epsilon) = \theta(t) + \epsilon M^{-1} p\left(t + \frac{\epsilon}{2}\right) $$
# $$ p(t + \epsilon) = p\left(t + \frac{\epsilon}{2}\right) + \frac{\epsilon}{2} \nabla_\theta \log p(\theta(t + \epsilon) \mid Y) $$
#
# ### 4. Criterio de Parada No-U-Turn (NUTS)
# Para evitar la calibración manual de la longitud $L$, NUTS construye recursivamente un árbol binario de pasos leapfrog hacia adelante y hacia atrás en el tiempo ($v \in \{-1, +1\}$). En cada profundidad $j$, la longitud se duplica ($2^j$ pasos). La expansión del árbol se detiene en cuanto los extremos $(\theta^-, \theta^+)$ y momentos $(p^-, p^+)$ comienzan a acercarse mutuamente (un giro en U):
# $$ (\theta^+ - \theta^-)' M^{-1} p^+ < 0 \qquad \text{o} \qquad (\theta^+ - \theta^-)' M^{-1} p^- < 0 $$
#
# ### 5. Promedio Dual de Hoffman-Gelman y Adaptación de Matriz de Masa
# - **Promedio Dual**: El tamaño de paso $\epsilon$ se adapta durante el calentamiento (*warmup*) mediante el método de Nesterov y Hoffman-Gelman buscando la tasa objetivo $\delta^* = 0.80$:
#   $$ \bar{H}_m = (1 - \eta_m) \bar{H}_{m-1} + \eta_m (\delta^* - \alpha_m), \qquad \log \epsilon_m = \mu - \frac{\sqrt{m}}{\gamma} \bar{H}_m $$
# - **Matriz de Masa Diagonal**: Las curvaturas posteriores anisótropas (por ejemplo $\sigma \approx 1.0$ frente a $\kappa \approx 0.10$) exigen escalas individuales. El acumulador en línea de Welford calcula la covarianza diagonal $M^{-1} \approx \operatorname{diag}(\operatorname{Var}(\theta))$ en ventanas escalonadas de calentamiento con regularización previa.
#
# ### 6. Score Analítico Exacto de Kalman en un Solo Paso
# En modelos de espacio de estados lineales:
# $$ s_t = T(\theta) s_{t-1} + R(\theta) \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, Q), \qquad y_t = d(\theta) + Z(\theta) s_t + v_t, \quad v_t \sim \mathcal{N}(0, H) $$
# La descomposición del error de predicción de la log-verosimilitud es:
# $$ \log L(Y_{1:T} \mid \theta) = -\frac{1}{2} \sum_{t=1}^T \left[ n_y \log(2\pi) + \log \det F_t + v_t' F_t^{-1} v_t \right] $$
# Differentiating respecto al parámetro estructural $\theta_j$ resulta en:
# $$ \frac{\partial \log L}{\partial \theta_j} = -\frac{1}{2} \sum_{t=1}^T \left[ \operatorname{tr}\left(F_t^{-1} \frac{\partial F_t}{\partial \theta_j}\right) + 2 v_t' F_t^{-1} \frac{\partial v_t}{\partial \theta_j} - (F_t^{-1} v_t)' \frac{\partial F_t}{\partial \theta_j} (F_t^{-1} v_t) \right] $$
# Puremacro calcula este score exacto en una única pasada temporal propagando conjuntamente las sensibilidades del estado y la covarianza $(\frac{\partial a_{t|t-1}}{\partial \theta_j}, \frac{\partial P_{t|t-1}}{\partial \theta_j})$ mediante la formulación estabilizada de Joseph, resolviendo las derivadas de las reglas de decisión $\frac{\partial G}{\partial \theta_j}$ mediante el sistema de Sylvester y la descomposición de Schur compleja.

# %% [markdown]
# ## Intuición
#
# **La Bola en el Cuenco.** Imagine que busca acumulación de probabilidad en un relieve montañoso cubierto por una densa niebla.
# - **Random-Walk Metropolis-Hastings** equivale a un excursionista con los ojos vendados que da pasos al azar. En 2 dimensiones puede encontrar cumbres, pero en 30 dimensiones casi cualquier paso conduce a un abismo o a un desierto vacío. El excursionista se ve obligado a dar pasitos microscópicos, sufriendo una autocorrelación severa y una convergencia exasperantemente lenta.
# - **Hamiltonian Monte Carlo** es como lanzar una canica sin fricción sobre ese relieve. Al dotarla de momento aleatorio, la canica transforma energía potencial ($-\log p(\theta \mid Y)$) en energía cinética, trepando pendientes escarpadas, surcando valles estrechos y recorriendo todo el conjunto típico en órbitas amplias y armónicas.
# - **El Muestreador No-U-Turn (NUTS)** actúa como un freno inteligente: permite que la canica ruede lo más lejos posible y la detiene exactamente cuando empieza a dar la vuelta hacia el punto de origen.
# - **Los Gradientes Analíticos Exactos** proporcionan la fuerza de gravedad física que orienta la canica. En lugar de estimar la gravedad sacudiendo la montaña con diferencias finitas, puremacro calcula el gradiente exacto mediante la recursión del score de Kalman con precisión de máquina en milisegundos.

# %%
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as sp_gamma

# Editorial styling and palette contract
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

import puremacro.dsge as dsge
from puremacro.dsge import load_mod
from puremacro.dsge._gradients import (
    ScoreDiagnosticsResult,
    build_state_space_sensitivities,
    kalman_score,
)
from puremacro.dsge.nuts import (
    NUTSResult,
    compute_bulk_ess,
    compute_ebfmi,
    compute_split_rhat,
    compute_tail_ess,
)
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.state_space import StateSpaceModel

# ---------------------------------------------------------------------------
# Sección 1: Definición del Modelo DSGE Neokeynesiano Canónico de 3 Ecuaciones
# ---------------------------------------------------------------------------
# Especificación canónica de Clarida, Galí y Gertler (1999 JEL), Woodford (2003)
NK_3EQ_MOD = """
// Modelo Neokeynesiano DSGE Canónico de 3 Ecuaciones
var y pi r g;
varexo eps_r eps_g;

parameters beta sigma kappa phi_pi rho_g;
beta   = 0.99;    // Factor de descuento trimestral
sigma  = 1.00;    // Elasticidad intertemporal de sustitución (EIS)
kappa  = 0.10;    // Pendiente de la Curva de Phillips Neokeynesiana
phi_pi = 1.50;    // Coeficiente de respuesta de la Regla de Taylor a inflación
rho_g  = 0.80;    // Persistencia del choque de demanda agregada

model;
  // 1. Curva IS dinámica
  y = y(+1) - (1/sigma)*(r - pi(+1)) + g;

  // 2. Curva de Phillips Neokeynesiana
  pi = beta*pi(+1) + kappa*y;

  // 3. Regla de política monetaria tipo Taylor
  r = phi_pi*pi + eps_r;

  // 4. Choque de demanda exógeno autorregresivo
  g = rho_g*g(-1) + eps_g;
end;

steady_state_model;
  y  = 0;
  pi = 0;
  r  = 0;
  g  = 0;
end;

varobs y pi;

estimated_params;
  sigma, gamma_pdf, 1.0, 0.20;
  kappa, gamma_pdf, 0.1, 0.03;
end;
"""

print("=" * 75)
print("puremacro 3.0.0 -- Demostración de NUTS y Gradientes Analíticos de Verosimilitud")
print("=" * 75)

model = load_mod(NK_3EQ_MOD)
print(f"Variables del modelo:   {model.variables}")
print(f"Estados predeterminados:{model.states}")
print(f"Variables de salto:     {model.controls}")
print(f"Choques estructurales:  {model.shocks}")
print(f"Series observables:     {model._varobs}")

# Validaciones estructurales del modelo
assert len(model.variables) == 4
assert model.states == ("g",)
assert model.shocks == ("eps_r", "eps_g")
assert model._varobs == ("y", "pi")

# %% [markdown]
# ## Sección 2: Simulación de la Economía Observable
#
# Generamos $T = 120$ trimestres (30 años) de series de tiempo sintéticas para la brecha de producto $y_t$ y la inflación $\pi_t$ bajo los verdaderos parámetros estructurales: $\sigma^* = 1.00$ y $\kappa^* = 0.10$.

# %%
# Simular series de tiempo macroeconómicas trimestrales
n_periods = 120
seed_sim = 101
sim_df = model.simulate(periods=n_periods, seed=seed_sim)
data = sim_df[["y", "pi"]].copy()

print(f"\nSimulación de {n_periods} trimestres de datos observables agregados (semilla={seed_sim}):")
print(data.head(5).to_string())

assert len(data) == n_periods
assert list(data.columns) == ["y", "pi"]
assert not data.isna().any().any()

# %% [markdown]
# ## Sección 3: Score Analítico Exacto de Kalman vs Diferencias Finitas Numéricas
#
# Evaluamos el vector de score exacto $\nabla_\theta \log L(Y \mid \theta)$ mediante la recursión analítica del filtro de Kalman de puremacro y lo contrastamos frente a un esquema de diferencias finitas centrales de 5 puntos:
# $$ \left.\frac{\partial \log L}{\partial \theta_j}\right|_{\text{FD}} = \frac{-\log L(\theta + 2h e_j) + 8\log L(\theta + h e_j) - 8\log L(\theta - h e_j) + \log L(\theta - 2h e_j)}{12 h} $$

# %%
obs = list(model._varobs)
p_names = ("sigma", "kappa")

# 1. Score Analítico Exacto vía Filtro de Kalman
t_score_0 = time.perf_counter()
ssm = make_state_space_from_varobs(model, obs)
sens = build_state_space_sensitivities(model, obs, p_names)
ll_exact, score_exact = kalman_score(data.to_numpy(), ssm, sens)
t_score_sec = time.perf_counter() - t_score_0

score_diag = ScoreDiagnosticsResult(
    loglik=ll_exact,
    gradient=score_exact,
    param_names=p_names,
    elapsed_sec=t_score_sec,
)

# 2. Diferencias Finitas Centrales de 5 Puntos
def ssm_evaluator(p: dict) -> StateSpaceModel:
    from puremacro.dsge.dynare import build_dynare
    m_perturbed = build_dynare(
        model._dynare_equations,
        variables=model.variables,
        shocks=model.shocks,
        params=p,
        steady_state=model.steady_state,
        check_steady_state=False,
        strict=False,
    )
    return make_state_space_from_varobs(m_perturbed, obs)

base_params = {
    "beta": 0.99,
    "sigma": 1.00,
    "kappa": 0.10,
    "phi_pi": 1.50,
    "rho_g": 0.80,
}

t_fd_0 = time.perf_counter()
verif_df = score_diag.verify_numerical_gradient(data.to_numpy(), ssm_evaluator, base_params, h=1e-5)
t_fd_sec = time.perf_counter() - t_fd_0

print(f"\nScore analítico exacto evaluado en:        {t_score_sec * 1000:.2f} ms")
print(f"Diferencias finitas numéricas evaluadas en: {t_fd_sec * 1000:.2f} ms")
print(f"Log-verosimilitud en parámetros base:       {ll_exact:.4f}")

print("\n--- Tabla de Verificación del Score de Kalman ---")
print(score_diag.to_markdown())

print("\n--- Exportación a Tabla LaTeX ---")
print(score_diag.to_latex())

assert np.isfinite(ll_exact)
assert len(score_exact) == 2
assert np.all(np.isfinite(score_exact))

# %% [markdown]
# ## Sección 4: Estimación Estructural DSGE de Extremo a Extremo con NUTS
#
# Estimamos los parámetros estructurales $\sigma$ y $\kappa$ simultáneamente con el algoritmo `method="nuts"` guiado por los gradientes analíticos exactos de puremacro.
#
# Configuración del muestreador:
# - Extracciones por cadena: `n_draws = 150`
# - Calentamiento (*burn-in*): `burn_in = 75`
# - Cadenas paralelas: `n_chains = 2`
# - Adaptación del paso: Promedio dual con tasa objetivo $\delta^* = 0.80$
# - Métrica de masa: Matriz diagonal adaptada en línea con el algoritmo de Welford

# %%
n_draws = 150
n_chains = 2
burn_in = 75
seed_nuts = 42

print(f"\nEjecutando Estimación MCMC mediante NUTS:")
print(f"  Cadenas: {n_chains} | Extracciones posteriores por cadena: {n_draws} | Calentamiento: {burn_in}")
print(f"  Gradientes Analíticos: Recursión de Kalman hacia adelante + Sylvester complejo")

t_nuts_0 = time.time()
res = model.estimate(
    data,
    method="nuts",
    n_draws=n_draws,
    n_chains=n_chains,
    burn_in=burn_in,
    seed=seed_nuts,
)
t_nuts_sec = time.time() - t_nuts_0

print(f"\nEstimación NUTS completada en {t_nuts_sec:.2f} segundos.")
assert isinstance(res, NUTSResult)
assert res.draws.shape == (n_chains, n_draws, 2)
assert res.param_names == ("sigma", "kappa")

# %% [markdown]
# ## Sección 5: Diagnósticos MCMC y Reportes de Publicación
#
# Examinamos los diagnósticos de convergencia:
# 1. **Split-$\hat{R}$ de Gelman-Rubin**: Divide cada cadena en dos mitades para contrastar estacionariedad interna y mezcla entre cadenas (umbral $\hat{R} < 1.05$).
# 2. **Tamaño Muestral Efectivo (ESS) Central y de Cola**: Evalúa las extracciones independientes efectivas para la media y los cuantiles exteriores (5% y 95%).
# 3. **E-BFMI**: Fracción bayesiana de información perdida de energía de Betancourt (umbral $> 0.30$) evaluando la eficiencia de la distribución de momento.
# 4. **Divergencias Hamiltonianas**: Detecta violaciones en la conservación de la energía o curvaturas patológicas ($\Delta H > 1000$).

# %%
summary_df = res.summary()
print("\n--- Resumen Posterior de la Estimación NUTS ---")
print(summary_df.to_string())

diag = res.diagnostics
print("\n--- Diagnósticos del Muestreador NUTS ---")
print(f"  Divergencias totales:           {diag['n_divergences']} (Tasa: {diag['divergence_rate']:.2%})")
print(f"  Profundidad media del árbol:    {diag['mean_tree_depth']:.2f}")
print(f"  Tasa de alcance de prof. máx.:  {diag['max_tree_depth_hit_rate']:.1%}")
print(f"  Tasa de aceptación media:       {diag['mean_accept_rate']:.3f} (Objetivo: 0.800)")
print(f"  Tamaños de paso adaptados:      {[f'{s:.4f}' for s in diag['step_sizes']]}")
print(f"  E-BFMI por cadena:              {[f'{e:.3f}' for e in diag['ebfmi']]}")

max_rhat = float(summary_df["r_hat"].max())
min_bulk_ess = float(summary_df["ess_bulk"].min())
min_tail_ess = float(summary_df["ess_tail"].min())
min_ebfmi = min(diag["ebfmi"])

print(f"\nVerificación de Convergencia:")
print(f"  Máximo Split-R_hat: {max_rhat:.4f}  (< 1.05: {'CORRECTO' if max_rhat < 1.05 else 'REVISAR'})")
print(f"  Mínimo ESS central: {min_bulk_ess:.1f}")
print(f"  Mínimo ESS de cola: {min_tail_ess:.1f}")
print(f"  Mínimo E-BFMI:      {min_ebfmi:.3f}  (> 0.30: {'CORRECTO' if min_ebfmi > 0.30 else 'REVISAR'})")

# Formatos de exportación
print("\n--- Tabla de Reporte en Markdown ---")
print(res.to_markdown())

print("\n--- Exportación de Tabla a LaTeX ---")
print(res.to_latex())

print("\n--- Exportación de Tabla a Typst ---")
print(res.to_typst())

# Validaciones cuantitativas
assert max_rhat < 1.05, f"Split R_hat superó 1.05: {max_rhat}"
assert diag["n_divergences"] == 0, f"Se registraron {diag['n_divergences']} divergencias"
assert min_ebfmi > 0.30, f"E-BFMI inferior al umbral 0.30: {min_ebfmi}"
assert np.all(np.isfinite(res.draws))

# %% [markdown]
# ## Sección 6: Galería de Visualizaciones Diagnósticas para Publicación
#
# Construimos una galería diagnóstica de 4 paneles para publicación:
# - **Panel A: Trazas MCMC**: Estacionariedad multi-cadena y mezcla rápida sin la adherencia típica del paseo aleatorio.
# - **Panel B: Densidades Posteriores**: Comparación visual de la previa frente a la posterior junto a los modos y valores verdaderos.
# - **Panel C: Autocorrelación**: Caída casi instantánea de la función de autocorrelación a través de los rezagos.
# - **Panel D: Diagnóstico de Energía**: Histogramas superpuestos de la energía marginal $E$ y la transición de energía $\Delta E$.

# %%
fig = plt.figure(figsize=(13, 9.5))

colors = ["0.15", "0.55"]
true_vals = {"sigma": 1.00, "kappa": 0.10}

# Panel 1: Traza de sigma
ax1 = plt.subplot2grid((2, 2), (0, 0))
for c in range(n_chains):
    ax1.plot(res.draws[c, :, 0], color=colors[c], lw=1.2, alpha=0.85, label=f"Cadena {c+1}")
ax1.axhline(true_vals["sigma"], color="black", linestyle="--", lw=1.2, label=r"Verdadero $\sigma^* = 1.00$")
if res.mode is not None and "sigma" in res.mode:
    ax1.axhline(res.mode["sigma"], color="0.40", linestyle=":", lw=1.2, label=f"Modo ({res.mode['sigma']:.3f})")
ax1.set_title(r"(a) Trazas Multi-Cadena: Elasticidad Intertemporal $\sigma$", fontweight="bold")
ax1.set_xlabel("Iteración MCMC (Post-Calentamiento)")
ax1.set_ylabel(r"$\sigma$")
ax1.legend(loc="upper right", fontsize=8)

# Panel 2: Traza de kappa
ax2 = plt.subplot2grid((2, 2), (0, 1))
for c in range(n_chains):
    ax2.plot(res.draws[c, :, 1], color=colors[c], lw=1.2, alpha=0.85, label=f"Cadena {c+1}")
ax2.axhline(true_vals["kappa"], color="black", linestyle="--", lw=1.2, label=r"Verdadero $\kappa^* = 0.10$")
if res.mode is not None and "kappa" in res.mode:
    ax2.axhline(res.mode["kappa"], color="0.40", linestyle=":", lw=1.2, label=f"Modo ({res.mode['kappa']:.3f})")
ax2.set_title(r"(b) Trazas Multi-Cadena: Pendiente Phillips $\kappa$", fontweight="bold")
ax2.set_xlabel("Iteración MCMC (Post-Calentamiento)")
ax2.set_ylabel(r"$\kappa$")
ax2.legend(loc="upper right", fontsize=8)

# Panel 3: Densidad Posterior con Curva Previa
ax3 = plt.subplot2grid((2, 2), (1, 0))
sigma_flat = res.draws[:, :, 0].ravel()
ax3.hist(sigma_flat, bins=22, density=True, alpha=0.55, color="0.45", edgecolor="0.2", label="Posterior MCMC")
# Curva previa: Gamma(media=1.0, std=0.20) => k = 25, escala = 0.04
x_sig = np.linspace(0.65, 1.35, 200)
prior_sig = sp_gamma.pdf(x_sig, a=25.0, scale=0.04)
ax3.plot(x_sig, prior_sig, color="black", linestyle="--", lw=1.5, label="Previa (Gamma)")
ax3.axvline(true_vals["sigma"], color="black", linestyle="-", lw=1.5, label=r"Verdadero $\sigma^*$")
if res.mode is not None and "sigma" in res.mode:
    ax3.axvline(res.mode["sigma"], color="0.3", linestyle=":", lw=1.5, label="Modo")
ax3.set_title(r"(c) Distribución Posterior vs Previa: $\sigma$", fontweight="bold")
ax3.set_xlabel(r"$\sigma$")
ax3.set_ylabel("Densidad")
ax3.legend(loc="upper right", fontsize=8)

# Panel 4: Diagnóstico de Energía de Betancourt
ax4 = plt.subplot2grid((2, 2), (1, 1))
E_flat = res.energy_trace.ravel()
dE_flat = np.concatenate([np.diff(res.energy_trace[c]) for c in range(n_chains)])
ax4.hist(E_flat - np.mean(E_flat), bins=25, density=True, alpha=0.55, color="0.30", edgecolor="0.1", label=r"Energía marginal $E - \bar{E}$")
ax4.hist(dE_flat, bins=25, density=True, alpha=0.45, color="0.70", edgecolor="0.3", label=r"Transición de energía $\Delta E$")
ax4.set_title(rf"(d) Diagnóstico de Energía de Betancourt (E-BFMI = {min_ebfmi:.3f})", fontweight="bold")
ax4.set_xlabel("Desviación de Energía")
ax4.set_ylabel("Densidad")
ax4.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Lea la salida
#
# **Lea la salida.**
# 1. **Score Analítico Exacto vs Diferencias Finitas**: El cálculo del score de Kalman en un solo paso de puremacro evalúa las derivadas exactas de la verosimilitud en menos de $25\text{ ms}$, coincidiendo estrechamente con las diferencias finitas centrales de 5 puntos sin necesidad de re-resolver el modelo $2K$ veces ni sufrir por cancelación numérica.
# 2. **Exploración Hamiltoniana sin Calibración Manual**: A diferencia del HMC tradicional donde el paso y la longitud de trayectoria debían afinarse por ensayo y error, NUTS selecciona automáticamente la trayectoria óptima en cada iteración (profundidades de árbol medias de $\approx 2\text{--}3$ duplicaciones) adaptando el paso a la tasa de aceptación fijada ($\approx 80\%$).
# 3. **Convergencia y Diagnósticos MCMC**:
#    - **Split-$\hat{R}$**: Ambos parámetros registran un split-$\hat{R} < 1.02$, muy por debajo del umbral conservador de $1.05$, confirmando la estacionariedad intra-cadena y la mezcla óptima entre cadenas.
#    - **Tamaño Muestral Efectivo**: Tanto el ESS central como el ESS de colas superan 80 extracciones efectivas sobre una muestra moderada de 300 extracciones post-calentamiento, reflejando baja autocorrelación.
#    - **Cero Divergencias**: No se registran divergencias hamiltonianas ($\Delta_{\max} = 1000$), lo que corrobora que el integrador leapfrog sigue con fidelidad la variedad posterior sin caer en trampas numéricas de energía.
#    - **E-BFMI**: Los valores de la fracción bayesiana de información perdida de energía ($\approx 1.0\text{--}1.2$) superan holgadamente el umbral crítico de $0.30$, garantizando un muestreo adecuado de la energía.
# 4. **Identificación Estructural de Parámetros**: Las distribuciones posteriores engloban con precisión los parámetros que generaron los datos sintéticos ($\sigma^* = 1.00$, $\kappa^* = 0.10$). La información muestral actualiza efectivamente las distribuciones previas, reduciendo la incertidumbre y centrando los modos en torno a los valores reales.

# %%
# Su turno: personalice el análisis posterior e intervalos de credibilidad
# ← cambie esto: configure el nivel de credibilidad deseado (ej. 0.90, 0.95, 0.99)
cred_level_yt = 0.95
alpha_yt = (1.0 - cred_level_yt) / 2.0

flat_draws_yt = res.draws.reshape(-1, 2)
lower_ci = np.quantile(flat_draws_yt, alpha_yt, axis=0)
upper_ci = np.quantile(flat_draws_yt, 1.0 - alpha_yt, axis=0)
posterior_corr = np.corrcoef(flat_draws_yt[:, 0], flat_draws_yt[:, 1])[0, 1]

print(f"Análisis Posterior Personalizado (Bandas de Credibilidad al {int(cred_level_yt * 100)}%):")
for idx, p in enumerate(res.param_names):
    p_mean = float(flat_draws_yt[:, idx].mean())
    p_std = float(flat_draws_yt[:, idx].std())
    print(f"  {p:6s}: Media = {p_mean:.4f}, Desv = {p_std:.4f}, Intervalo = [{lower_ci[idx]:.4f}, {upper_ci[idx]:.4f}]")
print(f"Correlación Posterior corr(sigma, kappa): {posterior_corr:.4f}")

assert len(lower_ci) == 2
assert len(upper_ci) == 2
assert (upper_ci > lower_ci).all()

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro.dsge.nuts` incorpora el estado del arte en muestreo Monte Carlo hamiltoniano guiado por gradientes para modelos DSGE en Python 100% puro:
# - **Score Analítico de Kalman**: Recursión de sensibilidad del espacio de estados con solucionador de Sylvester complejo para obtener gradientes exactos sin ruido de diferencias finitas ni sobrecostos de re-resolución.
# - **Muestreador No-U-Turn**: Generación recursiva de árboles binarios con criterio de giro en U de Betancourt (2017) que suprime la calibración manual de trayectorias.
# - **Promedio Dual y Matriz de Masa**: Adaptación de paso de Nesterov y estimación de métrica diagonal en línea de Welford para balancear escalas paramétricas anisótropas.
# - **Cero Dependencias Externas**: Ejecución completa en Python estándar y en navegadores web vía WebAssembly / Pyodide usando únicamente `numpy`, `scipy`, `pandas` y `matplotlib`.
