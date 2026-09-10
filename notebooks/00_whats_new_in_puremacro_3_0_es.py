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
# # Novedades en puremacro 3.0: Guía de Hitos Interactiva
#
# `puremacro 3.0.0` representa un salto generacional transformador para la computación macroeconómica
# en Python. Mientras que el ciclo de versiones 2.x llevó a `puremacro` a la **paridad operativa completa con Dynare**
# (abarcando el analizador completo de archivos `.mod`, perturbación podada de 2º y 3º orden, simulación por sendero
# extendido de Fair-Taylor, filtrado por tramos multirégimen de OccBin y análisis automatizado de identificación),
# la versión 3.0.0 pasa de *emular las herramientas heredadas* a *superarlas*.
#
# La versión 3.0.0 incorpora tres pilares pioneros directamente en los flujos de investigación macroeconómica:
# 1. **Gradientes analíticos exactos de verosimilitud ($\nabla_\theta \ln L$)** — Evaluación del vector estructural
#    del score mediante diferenciación de parámetros en el AST, solución de la ecuación matricial generalizada de
#    Sylvester vía descomposición de Schur compleja y sustitución regresiva triangular, y recursión forward del score
#    de Kalman en un solo paso.
# 2. **Hamiltonian Monte Carlo y No-U-Turn Sampler (NUTS) en Python puro** — Muestreo posterior basado en gradientes
#    para modelos macroeconómicos de alta dimensión con criterio de parada de giro en U generalizado de Betancourt (2017),
#    promedio dual de Hoffman-Gelman (2014), adaptación online de covarianza de Welford con contracción (*shrinkage*)
#    de Stan y diagnósticos MCMC rigurosos ($\hat{R}$ dividida normalizada por rangos, ESS bulk/tail, E-BFMI).
# 3. **Puente de agentes heterogéneos (HANK) en el espacio de secuencias en archivos `.mod`** — Introducción de la
#    sintaxis `hetagent_block; ... end;` en especificaciones estilo Dynare, acoplando limpiamente las decisiones
#    microeconómicas de los hogares (distribuciones estacionarias de riqueza $\mathcal{D}^*(a)$, curvas de $MPC(a)$)
#    con las condiciones de vaciado de mercado del modelo DSGE agregado mediante los jacobianos de *Fake-News*
#    de Auclert, Bardóczy, Rognlie y Straub (2021) ($\mathcal{J}_{C, r}, \mathcal{J}_{C, Y}$) y solvers de transición
#    de equilibrio general lineales y no lineales de Broyden.
#
# Los tres pilares operan en **100% Python puro** bajo el estricto **contrato de cuatro paquetes de Pyodide**
# (`numpy`, `scipy`, `pandas`, `matplotlib`), sin necesidad de compiladores en C/Fortran y permitiendo su ejecución
# fluida en navegadores web, JupyterLite, iPads y notebooks en la nube.

# %%
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Renderizado sin interfaz gráfica al ejecutarse desde la terminal
if "ipykernel" not in sys.modules and not hasattr(sys, "ps1"):
    import matplotlib
    matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt

# Aplicar estilo de publicación del repositorio si está disponible
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
try:
    import _nbstyle
    _nbstyle.apply_style()
except ImportError:
    pass

import puremacro
print(f"Versión de puremacro cargada: {puremacro.__version__}")

# %% [markdown]
# ---
# ## Sección 1: Gradientes analíticos exactos de verosimilitud ($\nabla_\theta \ln L$)
#
# ### 1.1 El cuello de botella del gradiente en la estimación macroeconométrica
#
# En las cadenas de herramientas heredadas (como Dynare o rutinas de MATLAB), la estimación de modelos DSGE
# mediante optimizadores basados en gradientes (L-BFGS-B, SQP) o Hamiltonian Monte Carlo depende de aproximaciones
# numéricas por diferencias finitas centrales:
# $$
# \frac{\partial \ln L}{\partial \theta_j} \approx \frac{\ln L(\theta + h e_j) - \ln L(\theta - h e_j)}{2h}
# $$
#
# Para $K$ parámetros estimados, cada evaluación del gradiente exige $2K$ pasadas completas del filtro de Kalman
# sobre toda la muestra histórica. En modelos de mediana y gran escala ($K \ge 35$), las diferencias finitas sufren
# de un coste computacional prohibitivo e inestabilidad numérica: un paso $h$ excesivamente grande introduce sesgo de
# truncamiento, mientras que un paso $h$ demasiado pequeño provoca cancelaciones catastróficas en coma flotante cerca
# de las fronteras de raíz unitaria.
#
# ### 1.2 El solver generalizado de Sylvester
#
# `puremacro 3.0` reemplaza las diferencias finitas con un **motor de score analítico exacto en una sola pasada forward**.
#
# Alrededor del estado estacionario determinista, la perturbación de primer orden produce la ecuación matricial cuadrática
# de Riccati para la matriz de transición de estados $G(\theta)$:
# $$
# \mathcal{F}(G; \theta) \equiv A_+(\theta) G(\theta)^2 + A_0(\theta) G(\theta) + A_-(\theta) = 0
# $$
#
# Diferenciando con respecto a un parámetro estructural escalar $\theta_j$, se obtiene la **ecuación matricial generalizada de Sylvester**:
# $$
# \hat{A} \frac{\partial G}{\partial \theta_j} + B \frac{\partial G}{\partial \theta_j} C = D_j
# $$
# donde:
# - $\hat{A} = A_0 + A_+ G \in \mathbb{R}^{N \times N}$
# - $B = A_+ \in \mathbb{R}^{N \times N}$
# - $C = G \in \mathbb{R}^{N \times N}$
# - $D_j = - \left( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} \right)$
#
# Dado que la estabilidad de Blanchard-Kahn garantiza que todos los autovalores de $C = G$ se sitúan estrictamente dentro
# del círculo unitario abierto, la función `solve_sylvester_generalized` calcula la descomposición de Schur compleja
# $C = U_c T_c U_c^H$ **una sola vez** y resuelve el sistema triangular desacoplado columna por columna simultáneamente
# para los $K$ parámetros en $\mathcal{O}(K \cdot N^3)$ operaciones en coma flotante en $< 1$ ms.

# %%
from puremacro.dsge._gradients import solve_sylvester_generalized

# Demostración del solver matricial generalizado de Sylvester con precisión de máquina: A_hat * X + B * X * C = D
N, M = 4, 3
rng = np.random.default_rng(2026)

A_hat = rng.standard_normal((N, N)) + 4.0 * np.eye(N)
B = rng.standard_normal((N, N)) * 0.2
# Matriz de transición estable (autovalores estrictamente dentro del círculo unitario)
C = rng.standard_normal((M, M)) * 0.25
D = rng.standard_normal((N, M))

X_sol = solve_sylvester_generalized(A_hat, B, C, D)
residual = A_hat @ X_sol + B @ X_sol @ C - D
max_error = np.max(np.abs(residual))

print(f"Dimensiones de la solución X de Sylvester: {X_sol.shape}")
print(f"Residuo máximo ||A_hat @ X + B @ X @ C - D||_inf: {max_error:.2e}")
assert max_error < 1e-12, "¡El solver de Sylvester no alcanzó precisión de máquina!"

# %% [markdown]
# ### 1.3 Recursión forward del score de Kalman en un solo paso
#
# A partir de las derivadas de las reglas de decisión $\frac{\partial G}{\partial \theta_j}$ y $\frac{\partial N}{\partial \theta_j}$,
# la función `build_state_space_sensitivities` construye las sensibilidades del modelo de espacio de estados
# $(\frac{\partial T}{\partial \theta_j}, \frac{\partial Z}{\partial \theta_j}, \frac{\partial R}{\partial \theta_j}, \frac{\partial Q}{\partial \theta_j}, \frac{\partial H}{\partial \theta_j})$.
#
# La sensibilidad de la covarianza estacionaria no condicional inicial $\frac{\partial P_0}{\partial \theta_j}$ se resuelve
# mediante la ecuación diferenciada discreta de Lyapunov.
#
# La log-verosimilitud gaussiana y el vector de score analítico exacto $\nabla_\theta \ln L$ se evalúan conjuntamente en una sola pasada forward:
# $$
# \frac{\partial \ln L}{\partial \theta_j} = -\frac{1}{2} \sum_{t=1}^{T_{obs}} \left[ \operatorname{tr}\left( F_t^{-1} \frac{\partial F_t}{\partial \theta_j} \right) + 2 v_t^\top F_t^{-1} \frac{\partial v_t}{\partial \theta_j} - (F_t^{-1} v_t)^\top \frac{\partial F_t}{\partial \theta_j} (F_t^{-1} v_t) \right]
# $$
# propagando las sensibilidades del estado y de las covarianzas $(\frac{\partial a_{t|t-1}}{\partial \theta_j}, \frac{\partial P_{t|t-1}}{\partial \theta_j})$
# hacia adelante junto con las ecuaciones recursivas de Kalman, sin requerir grafos de retropropagación en memoria.

# %%
from puremacro.dsge import load_mod
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge._gradients import (
    build_state_space_sensitivities,
    compute_decision_rule_derivatives,
    kalman_score,
    ScoreDiagnosticsResult,
)

MOD_CODE_GRAD = """
var y c a;
varexo e_a;
parameters beta sigma rho_a;
beta = 0.99;
sigma = 1.0;
rho_a = 0.8;

model;
  c = c(+1) - (1/sigma)*y;
  y = c + a;
  a = rho_a * a(-1) + e_a;
end;

steady_state_model;
  y = 0; c = 0; a = 0;
end;

shocks;
  var e_a; stderr 0.2;
end;

varobs y;
"""

# 1. Analizar y compilar el modelo DSGE
model_grad = load_mod(MOD_CODE_GRAD)
varobs = ["y"]
param_names = ["sigma", "rho_a"]

# 2. Simular muestra de datos observados
sim_df = model_grad.simulate(periods=40, seed=42)
y_sim = sim_df[["y"]].to_numpy()

# 3. Construir el modelo de espacio de estados y las sensibilidades analíticas
ssm = make_state_space_from_varobs(model_grad, varobs)
sensitivities = build_state_space_sensitivities(model_grad, varobs, param_names)

# 4. Evaluar la log-verosimilitud exacta y el score en una sola pasada forward
t0 = time.perf_counter()
loglik, score = kalman_score(y_sim, ssm, sensitivities)
elapsed = time.perf_counter() - t0

# 5. Empaquetar en ScoreDiagnosticsResult
res_score = ScoreDiagnosticsResult(
    loglik=loglik,
    gradient=score,
    param_names=tuple(param_names),
    elapsed_sec=elapsed,
)

print(f"Log-Verosimilitud exacta : {res_score.loglik:.4f}")
print(f"Tiempo transcurrido      : {res_score.elapsed_sec * 1000:.3f} ms")
print("\nTabla de diagnósticos del score analítico:")
print(res_score.to_markdown())

# Inspeccionar sensibilidades analíticas de las reglas de decisión: dghx / d_rho_a
dghx, dghu = compute_decision_rule_derivatives(model_grad, "rho_a")
print("\nSensibilidades analíticas de la transición de estados (dghx / d_rho_a):")
print(pd.DataFrame(dghx, index=model_grad.variables, columns=model_grad.states))

# %% [markdown]
# ### 1.4 Visualización del score analítico exacto y sensibilidades de transición
#
# Graficamos el vector de score analítico y las sensibilidades de las reglas de decisión respecto a los parámetros estructurales.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Gráfico de barras del vector de score analítico
x_pos = np.arange(len(param_names))
bars = ax1.bar(x_pos, score, color=["#1f77b4", "#2ca02c"], edgecolor="black", width=0.5, alpha=0.85)
ax1.axhline(0, color="black", lw=0.8, linestyle="--")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([r"$\sigma$ (Aversión al riesgo)", r"$\rho_a$ (Persistencia tecnológica)"], fontsize=10)
ax1.set_title(r"Vector de score exacto $\nabla_\theta \ln L$", fontsize=11, fontweight="bold")
ax1.set_ylabel("Score de log-verosimilitud")
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f"{height:.3f}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3 if height >= 0 else -12),
                 textcoords="offset points", ha="center", va="bottom", fontsize=9)

# Gráfico de barras de sensibilidad de reglas de decisión ante persistencia
vars_plot = model_grad.variables
dghx_vals = dghx[:, 0]
ax2.bar(vars_plot, dghx_vals, color="#d62728", edgecolor="black", width=0.5, alpha=0.85)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title(r"Sensibilidad de estado $\partial ghx / \partial \rho_a$", fontsize=11, fontweight="bold")
ax2.set_ylabel(r"$\partial y / \partial \rho_a$")
ax2.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Sección 2: Hamiltonian Monte Carlo y NUTS en Python puro
#
# ### 2.1 La necesidad de MCMC basado en gradientes en macroeconomía
#
# El algoritmo clásico de Random-Walk Metropolis-Hastings (RWMH) explora distribuciones posteriores de alta dimensión
# mediante difusión browniana: la distancia recorrida esperada escala como $\mathcal{O}(\sqrt{N})$. En modelos DSGE
# de mediana y gran escala con curvaturas complejas y fuertes correlaciones posteriores, RWMH difunde con lentitud,
# presenta alta autocorrelación y produce tamaños muestrales efectivos deficientes ($ESS < 2\%$).
#
# Hamiltonian Monte Carlo (HMC) resuelve esto transformando el muestreo posterior en dinámica hamiltoniana.
# Introduce variables continuas de momento $p \sim \mathcal{N}(0, M)$ conjugadas con los parámetros $\theta$,
# definiendo la energía hamiltoniana:
# $$
# H(\theta, p) = V(\theta) + K(p) = - \ln p(\theta \mid Y) + \frac{1}{2} p^\top M^{-1} p
# $$
#
# El algoritmo **No-U-Turn Sampler (NUTS)** (Hoffman y Gelman, 2014; Betancourt, 2017) construye árboles binarios
# recursivos de integración leapfrog hacia adelante y hacia atrás en el tiempo, deteniéndose de manera automática
# en el momento exacto en que la trayectoria comienza a replegarse sobre sí misma:
# $$
# (\theta^+ - \theta^-)^\top M^{-1} p^+ < 0 \quad \text{o} \quad (\theta^+ - \theta^-)^\top M^{-1} p^- < 0
# $$
#
# ### 2.2 Características del motor NUTS en Python puro de puremacro
#
# 1. **Integrador simpléctico Leapfrog (Verlet)**: Conserva el volumen en el espacio de fases y acota el error del hamiltoniano sombra a $\mathcal{O}(\epsilon^2)$ sin deriva de energía.
# 2. **Promedio dual escalonado de Stan**: Adapta automáticamente el tamaño de paso $\epsilon$ hacia la probabilidad objetivo $\delta^* = 0.80$.
# 3. **Adaptación de covarianza en línea de Welford**: Estima la matriz de masa diagonal $M^{-1}$ con contracción regularizada hacia la previa.
# 4. **Diagnósticos MCMC rigurosos**: $\hat{R}$ dividida normalizada por rangos, $ESS_{bulk}$, $ESS_{tail}$ y fracción bayesiana de energía de información faltante ($E\text{-}BFMI$).

# %%
MOD_CODE_NUTS = """
var y c a;
varexo e_a;
parameters beta sigma rho_a;
beta = 0.99;
sigma = 1.0;
rho_a = 0.8;

model;
  c = c(+1) - (1/sigma)*y;
  y = c + a;
  a = rho_a * a(-1) + e_a;
end;

steady_state_model;
  y = 0; c = 0; a = 0;
end;

varobs y;
estimated_params;
  sigma, gamma_pdf, 1.0, 0.25;
end;
"""

# 1. Cargar el modelo con especificación de estimación
model_nuts = load_mod(MOD_CODE_NUTS)

# 2. Simular observaciones
rng_nuts = np.random.default_rng(42)
df_data = pd.DataFrame({"y": rng_nuts.normal(0, 0.5, size=25)})

# 3. Estimar mediante NUTS en Python puro con gradientes analíticos exactos de verosimilitud
# Tamaño de muestra compacto para ejecución ultrarrápida (< 3 segundos)
t0 = time.perf_counter()
res_nuts = model_nuts.estimate(
    data=df_data,
    method="nuts",
    n_draws=30,
    n_chains=2,
    burn_in=15,
    seed=42,
)
t_nuts = time.perf_counter() - t0

print(f"Estimación con NUTS completada en {t_nuts:.2f} s")
print("\nTabla de resumen posterior de parámetros:")
print(res_nuts.summary())
print("\nDiagnósticos de NUTS:")
print(f"  Transiciones divergentes : {res_nuts.diagnostics.get('n_divergences', 0)}")
print(f"  Tamaños de paso adaptados: {np.round(res_nuts.diagnostics.get('step_sizes', []), 4)}")

# %% [markdown]
# ### 2.3 Diagnósticos posteriores de MCMC y distribución de energía
#
# El objeto `NUTSResult` proporciona métodos nativos de visualización diagnóstica:
# - `.plot_trace()` — Trayectorias de las cadenas múltiples y dispersión posterior.
# - `.plot_posterior()` — Densidad marginal posterior con intervalos de credibilidad.
# - `.energy_diagnostics()` — Transiciones de energía y estadístico $E\text{-}BFMI$.

# %%
# 1. Gráfico de trazas a través de las cadenas
fig_trace, axes_trace = res_nuts.plot_trace()
plt.suptitle("Trazas posteriores de NUTS", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# 2. Densidad marginal posterior
fig_post, axes_post = res_nuts.plot_posterior()
plt.suptitle("Distribución marginal posterior (NUTS)", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# 3. Diagnósticos de energía (E-BFMI)
energy_stats, fig_energy, ax_energy = res_nuts.energy_diagnostics()
print(f"Reporte de diagnósticos de energía:")
print(f"  E-BFMI promedio entre cadenas : {energy_stats['mean_ebfmi']:.3f}")
print(f"  E-BFMI superado (>= 0.3)      : {energy_stats['passed']}")
plt.show()

# %% [markdown]
# ---
# ## Sección 3: Puente de agentes heterogéneos (HANK) en el espacio de secuencias en archivos `.mod`
#
# ### 3.1 Superando la barrera del agente representativo
#
# Los modelos nuevo-keynesianos de agente representativo (RANK) colapsan las decisiones familiares en una sola ecuación
# de Euler, ignorando el riesgo de ingresos, la desigualdad de riqueza y a los hogares con restricciones de liquidez
# y alta propensión marginal a consumir (*hand-to-mouth*).
#
# Los modelos con agentes heterogéneos (HANK) sustituyen al agente ficticio con un continuo de hogares sujetos a choques
# laborales idiosincrásicos y restricciones de endeudamiento. Sin embargo, en la representación de espacio de estados,
# resolver modelos HANK requiere acarrear la distribución infinita de riqueza $\mathcal{D}_t(a, s)$ como variable de estado,
# provocando la maldición de la dimensionalidad.
#
# El método del **Jacobiano en el espacio de secuencias (SSJ)** (**Auclert, Bardóczy, Rognlie y Straub, 2021, *Econometrica*)**
# resuelve modelos HANK directamente en el espacio de trayectorias y choques sobre un horizonte temporal $T$.
#
# ### 3.2 La sintaxis `hetagent_block` en archivos `.mod`
#
# `puremacro 3.0` incorpora el marco del espacio de secuencias de manera nativa dentro de archivos de sintaxis `.mod`
# mediante el nuevo bloque `hetagent_block; ... end;`:
#
# ```dynare
# hetagent_block;
#   model = one_asset_hank;
#   n_a = 40;
#   a_max = 25.0;
#   borrowing_limit = 0.0;
#   grid = hyperbolic;
# end;
# ```
#
# El compilador de modelos acopla de manera automática el bloque microeconómico de hogares (resuelto por el Método de
# Grilla Endógena o EGM) con las condiciones de vaciado de mercado del DSGE agregado a través del **algoritmo de Fake-News**,
# evaluando los jacobianos intertemporales de consumo $\mathcal{J}_{C, r} = \frac{\partial \mathbf{C}}{\partial \mathbf{r}}$ y
# $\mathcal{J}_{C, Y} = \frac{\partial \mathbf{C}}{\partial \mathbf{Y}}$ en tiempo $\mathcal{O}(T^2)$.

# %%
from puremacro.dsge.hank import load_hank_mod, solve_hank_bridge

SAMPLE_HANK_MOD = """
var Y C r pi i;
varexo eps_m;

parameters beta gamma r_ss phi_pi kappa;
beta = 0.985;
gamma = 1.0;
r_ss = 0.01;
phi_pi = 1.5;
kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 40;
  a_max = 25.0;
  borrowing_limit = 0.0;
  grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
"""

# 1. Cargar y compilar el modelo HANK desde especificación .mod
model_hank = load_hank_mod(SAMPLE_HANK_MOD)

print("Resumen del estado estacionario en HANK:")
for k, v in model_hank.steady_state.items():
    print(f"  {k:5s} = {v:8.4f}")

# 2. Calcular los jacobianos en el espacio de secuencias mediante Fake-News
T_horizon = 20
jacobians = model_hank.compute_jacobians(T=T_horizon)
J_C_r = jacobians["J_C_r"]
J_C_Y = jacobians["J_C_Y"]

print(f"\nDimensiones del jacobiano J_C_r: {J_C_r.shape}")
print(f"Elasticidad de interés contemporánea (dC_0 / dr_0): {J_C_r[0, 0]:.4f}")
print(f"Propensión marginal al consumo contemporánea (dC_0 / dY_0): {J_C_Y[0, 0]:.4f}")

# %% [markdown]
# ### 3.3 Simulación de la transición de equilibrio general
#
# Simulamos la respuesta de equilibrio general de la economía HANK ante un choque expansivo de política monetaria
# ($\epsilon_m = -25$ pb, $\rho = 0.5$) a lo largo de 15 trimestres.
#
# La relajación monetaria reduce las tasas de interés reales, estimulando el crédito y el gasto, a la vez que
# redistribuye ingresos hacia hogares con alta propensión marginal al consumo (PMC), amplificando la demanda agregada.

# %%
# Simular sendero de transición de equilibrio general
res_hank = model_hank.simulate(
    shock="eps_m",
    magnitude=-0.0025,
    rho=0.5,
    horizon=15,
    nonlinear=False,
)

print(f"Estado de la simulación HANK: Convergencia = {res_hank.converged}")
print("\nTabla de resumen del sendero de transición:")
print(res_hank.summary())

# Graficar transiciones de equilibrio general (IRF)
fig_tr, axes_tr = res_hank.plot_transition()
plt.suptitle("Transición de equilibrio general HANK (bajada de 25 pb en la tasa)", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# Graficar distribución estacionaria de riqueza D*(a) y curva de PMC
fig_dist, axes_dist = res_hank.plot_distribution()
plt.suptitle("Distribuciones microeconómicas HANK", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# %% [markdown]
# ### 3.4 Atajo en una sola línea: `solve_hank_bridge`
#
# Para flujos de trabajo ágiles, `solve_hank_bridge` analiza la especificación `.mod`, resuelve el equilibrio
# estacionario, calcula los jacobianos de Fake-News y obtiene la trayectoria de equilibrio general en una sola instrucción.

# %%
# Resolver el modelo HANK directamente desde la cadena .mod en 1 línea
res_bridge = solve_hank_bridge(
    SAMPLE_HANK_MOD,
    shock="eps_m",
    magnitude=-0.0025,
    horizon=15,
)

print(f"solve_hank_bridge ejecutado exitosamente: {res_bridge.converged}")
print("\nPrimeros 5 períodos del producto (Y) y consumo (C):")
print(res_bridge.transition_paths[["Y", "C", "pi", "r"]].head())

# %% [markdown]
# ---
# ## Conclusión y próximos pasos
#
# `puremacro 3.0.0` aporta tres capacidades de referencia para la investigación macroeconómica:
# 1. **Gradientes analíticos exactos de verosimilitud ($\nabla_\theta \ln L$)**: Elimina las lentas y ruidosas diferencias finitas mediante el solver generalizado de Sylvester y la recursión forward de Kalman.
# 2. **HMC y NUTS en Python puro**: Exploración posterior de alta eficiencia con adaptación automática del paso y diagnósticos MCMC integrales.
# 3. **Modelos HANK en el espacio de secuencias en `.mod`**: Integración fluida de distribuciones de riqueza microeconómicas y vaciado de mercado macroeconómico mediante jacobianos de Fake-News.
#
# Todas las herramientas se ajustan estrictamente al **contrato de cuatro paquetes de Pyodide** (`numpy`, `scipy`, `pandas`, `matplotlib`) con **cero dependencias compiladas**.
#
# - **Documentación**: [https://jalonso1979.github.io/puremacro/](https://jalonso1979.github.io/puremacro/)
# - **Código fuente**: [https://github.com/jalonso1979/puremacro](https://github.com/jalonso1979/puremacro)
# - **Guía bilingüe**: [Español (dsge_v3.md)](../docs/es/dsge_v3.md) · [English (dsge_v3.md)](../docs/dsge_v3.md)
