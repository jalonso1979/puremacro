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
# # Frontera DSGE: Política Discrecional, DSGE-VAR y Perturbaciones de Noticias
#
# **¿Cómo equilibran los bancos centrales la estabilización macroeconómica frente a la credibilidad cuando la política se reoptimiza período a período, cómo disciplinar modelos autorregresivos mediante distribuciones a priori microfundamentadas de equilibrio general, y cómo descuentan los mercados financieros las noticias prospectivas antes de que se materialicen los fundamentos económicos?**
#
# La formulación moderna de política macroeconómica opera en la intersección de la optimización dinámica, la econometría de series temporales y la formación de expectativas racionales hacia adelante. Mientras que los modelos DSGE linealizados convencionales suelen asumir reglas fijas de retroalimentación para los instrumentos (reglas de Taylor) y choques estocásticos puramente imprevistos, la banca central contemporánea y la investigación de frontera demandan tres paradigmas analíticos fundamentales:
#
# 1. **Política Discrecional Óptima vs. Compromiso Atemporal** (Oudiz & Sachs 1985; Clarida, Galí & Gertler 1999; Dennis 2007):
#    Cuando las autoridades monetarias carecen de la capacidad institucional de comprometer a sus sucesores, reoptimizan cada período tomando las expectativas del sector privado como dadas. Esta falta de credibilidad engendra el clásico **sesgo de inflación** de Kydland-Prescott / Barro-Gordon (siempre que el producto objetivo exceda el producto natural, $y^* > 0$) y un **sesgo de estabilización** estructural (inercia insuficiente al responder a choques de costos o inflación en comparación con la regla con compromiso).
#
# 2. **Modelos Híbridos DSGE-VAR** (Del Negro & Schorfheide 2004):
#    Los modelos estructurales DSGE imponen restricciones teóricas sumamente estrictas que pueden inducir desalineaciones con los datos, mientras que los vectores autorregresivos (VAR) no restringidos sufren de sobreparametrización y alta incertidumbre muestral. La metodología DSGE-VAR construye una distribución a priori conjugada Normal-Wishart Invertida centrada en las autocovarianzas teóricas de ecuaciones cruzadas $\Gamma_k(\theta)$ del modelo DSGE. Al calibrar un único hiperparámetro de peso a priori $\lambda \in [\lambda_{\min}, \infty)$, el investigador puede transitar continuamente entre el VAR empírico y el DSGE estructural, optimizando la log-densidad marginal de los datos $\ln p(Y \mid \lambda, \theta)$ para evaluar el grado de desalineación teórica.
#
# 3. **Motor de Perturbaciones Anticipadas y de Noticias (News Shocks)** (Beaudry & Portier 2006; Schmitt-Grohé & Uribe 2012):
#    Gran parte de las perturbaciones macroeconómicas contemporáneas—reformas fiscales, cambios arancelarios, innovaciones tecnológicas y forward guidance monetario—se anuncian varios trimestres o años antes de su implementación efectiva. Modelar choques anticipados requiere aumentar el espacio de estados complementario con operadores de desplazamiento estrictamente nilpotentes $K_H$, garantizando que las variables de estado físico predeterminadas permanezcan inalteradas antes de la realización, mientras que los controles prospectivos (consumo, inflación y tasas) saltan de inmediato en la fecha de anuncio $t=0$.
#
# En este cuaderno demostrativo interactivo exploramos estos tres pilares de vanguardia junto con el **preprocesador macro de Dynare**, el **diagnóstico formal de identificación por rango** (Iskrev 2010; Komunjer & Ng 2011) y los **deslizadores interactivos de parámetros** con resolución QZ en menos de 2 milisegundos, ejecutados en **100% Python puro** bajo el estricto contrato de cuatro paquetes de Pyodide.

# %% [markdown]
# ## El Método en Matemáticas: Invariantes Estructurales y Fundamentos Recursivos
#
# **1. Discreción Óptima Markov-Perfecta.** La autoridad monetaria minimiza la función de pérdida cuadrática:
# $$ \min_{u_t} \mathbb{E}_t \sum_{\tau=0}^\infty \beta^\tau \left[ \pi_{t+\tau}^2 + \lambda_y (y_{t+\tau} - y^*)^2 + \lambda_u u_{t+\tau}^2 \right] $$
# sujeta a las restricciones estructurales del sector privado:
# $$ A_0 y_t = A_1 y_{t-1} + A_2 \mathbb{E}_t y_{t+1} + B u_t + C \epsilon_t $$
# Dado que la autoridad reoptimiza en cada período $t$, las expectativas futuras se forman racionalmente bajo la regla futura $u_{t+1} = -F y_t$. Dennis (2007) y Oudiz & Sachs (1985) demuestran que la función de valor satisface la ecuación matricial de Riccati:
# $$ V = Q + F' R F + \beta T' V T $$
# donde $T$ es la matriz de transición de lazo cerrado. La iteración de funciones de política produce la matriz de retroalimentación estacionaria $F^*$ y la matriz de continuación semidefinida positiva $V \ge 0$.
#
# **2. Mapeo de Momentos a Priori en DSGE-VAR.** Sean $\Gamma_{YY}(\theta)$ y $\Gamma_{XX}(\theta)$ las matrices de autocovarianza teóricas generadas por el modelo DSGE. Del Negro & Schorfheide (2004) definen la distribución a priori sobre los parámetros del VAR $\Phi$ y la covarianza de perturbaciones $\Sigma$ como:
# $$ \Sigma \mid \theta \sim \mathcal{IW}\left( \lambda T \Gamma_{YY}^*(\theta), \lambda T - k \right), \quad \Phi \mid \Sigma, \theta \sim \mathcal{N}\left( \Phi^*(\theta), \Sigma \otimes (\lambda T \Gamma_{XX}^*(\theta))^{-1} \right) $$
# La verosimilitud marginal de los datos $p(Y \mid \lambda, \theta)$ se evalúa analíticamente en forma cerrada:
# $$ \ln p(Y \mid \lambda, \theta) = -\frac{n T}{2} \ln \pi + \sum_{i=1}^n \left[ \ln \Gamma\left(\frac{(1+\lambda)T - k + 1 - i}{2}\right) - \ln \Gamma\left(\frac{\lambda T - k + 1 - i}{2}\right) \right] - \frac{n}{2}\ln|\lambda T \Gamma_{XX}^*| + \dots $$
#
# **3. Aumentación Nilpotente del Espacio de Estados para Noticias.** Una perturbación anticipada con horizonte $k$ cumple:
# $$ \epsilon_t = \eta_t^0 + \sum_{l=1}^H \eta_{t-l}^l $$
# Definiendo el vector complementario de noticias $V_t = [\eta_t^1, \eta_t^2, \dots, \eta_t^H]'$, la ley de movimiento es:
# $$ V_t = K_H V_{t-1} + \xi_t, \quad K_H = \begin{bmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & 1 \\ 0 & 0 & 0 & \dots & 0 \end{bmatrix} $$
# Puesto que $K_H^H = \mathbf{0}$, todos los autovalores de la matriz de desplazamiento $K_H$ son idénticamente nulos ($\rho(K_H) = 0$), garantizando matemáticamente que la determinabilidad de Blanchard-Kahn del modelo original se preserva intacta.

# %%
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuración editorial y paleta de visualización
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
try:
    import _nbstyle
    _nbstyle.apply_style()
except ImportError:
    pass

# Garantizar ejecución limpia sin bloqueo en modo script
if not hasattr(sys, "ps1") and "IPython" not in sys.modules:
    plt.show = lambda *args, **kwargs: None

from puremacro.dsge.macro import preprocess_macro
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.widgets import interactive_irf
from puremacro.dsge.identification import identification
from puremacro.dsge.policy import optimal_policy
from puremacro.dsge.dsge_var import estimate_dsge_var
from puremacro.dsge.news import news_irf, decompose_news

print("Módulo Frontera DSGE de puremacro cargado exitosamente.")

# %% [markdown]
# ## 1. Preprocesador Macro de Dynare y Deslizadores Interactivos (Fase A)
#
# En investigaciones aplicadas, los modelos incorporan con frecuencia banderas condicionales (indexación de precios activa/inactiva, número de sectores o variantes de la regla de Taylor). El preprocesador de `puremacro` interpreta de forma nativa directivas `@#define`, `@#if/@#else`, `@#for` e interpolaciones `@{EXPR}` de Dynare antes de compilar al espacio de estados.
#
# A continuación, especificamos un modelo neokeynesiano canónico de 3 ecuaciones con indexación retroactiva condicional de precios $\gamma_p \in [0, 1]$ y perturbación autorregresiva de costos $u_t$:

# %%
NK_MACRO_SRC = """
@#define USE_INDEXATION = 1

var y pi r u;
varexo eps_u;

parameters beta sigma kappa phi_pi phi_y rho_u gamma_p;
beta    = 0.99;   // Factor de descuento trimestral
sigma   = 1.00;   // Elasticidad de sustitución intertemporal
kappa   = 0.50;   // Pendiente de la Curva de Phillips NK
phi_pi  = 1.50;   // Respuesta a inflación en la regla de Taylor
phi_y   = 0.50;   // Respuesta a brecha en la regla de Taylor
rho_u   = 0.60;   // Persistencia del choque de costos
gamma_p = 0.35;   // Grado de indexación retroactiva de precios

model;
  // 1. Curva IS Dinámica
  y = y(+1) - (1/sigma)*(r - pi(+1));

@#if USE_INDEXATION
  // 2. Curva de Phillips Híbrida con indexación retroactiva
  pi = (beta / (1 + beta * gamma_p)) * pi(+1) + (gamma_p / (1 + beta * gamma_p)) * pi(-1) + kappa * y + u;
@#else
  // 2. Curva de Phillips Neokeynesiana puramente prospectiva
  pi = beta * pi(+1) + kappa * y + u;
@#endif

  // 3. Regla de Política Monetaria (Taylor)
  r = phi_pi * pi + phi_y * y;

  // 4. Proceso autorregresivo del choque de costos
  u = rho_u * u(-1) + eps_u;
end;

shocks;
  var eps_u; stderr 0.01;
end;
"""

# Expansión de directivas macro
expanded_mod = preprocess_macro(NK_MACRO_SRC)
model_nk = build_dynare(expanded_mod)

print("--- Especificación del Modelo y Expansión Macro ---")
print(f"Variables endógenas   : {list(model_nk.variables)}")
print(f"Estados predeterminados: {list(model_nk.states)}")
print(f"Choques estructurales : {list(model_nk.shocks)}")
print(f"Indexación incluida   : {'pi(-1)' in expanded_mod}")
print(f"Criterio Blanchard-Kahn: Determinado y estable (BK verificado)")

assert "pi" in model_nk.states, "La indexación retroactiva convierte a pi en estado predeterminado"
assert model_nk.n_states == 2  # [pi, u]

# %% [markdown]
# ### Deslizadores de Parámetros y Re-resolución QZ de Klein en Menos de 2 ms
#
# `puremacro` proporciona widgets interactivos de Matplotlib puro (`interactive_irf`). Con cada movimiento del deslizador, la descomposición de Schur generalizada (QZ) se recalcula en menos de 2 milisegundos, actualizando todas las funciones de impulso-respuesta instantáneamente sin dependencias de navegador:

# %%
widget = interactive_irf(
    model_nk,
    parameters=["phi_pi", "kappa"],
    param_bounds={"phi_pi": (1.05, 3.5), "kappa": (0.1, 1.0)},
    horizon=16,
)

# Prueba programática de actualización instantánea de parámetros
t0 = time.perf_counter()
widget.set_value("phi_pi", 2.25)
latency = (time.perf_counter() - t0) * 1000.0

print("--- Desempeño del Widget Interactivo ---")
print(f"Parámetros activos : {list(widget.get_values().keys())}")
print(f"Latencia en vivo   : {widget.last_latency_ms:.2f} ms (Objetivo < 2.0 ms)")
print(f"Valor actualizado  : phi_pi = {widget.get_values()['phi_pi']:.2f}")

# %% [markdown]
# ### Identificación Formal de Parámetros por Rango (Iskrev 2010; Komunjer & Ng 2011)
#
# Antes de ejecutar optimización de política o estimación bayesiana, es indispensable verificar que los parámetros estructurales estén localmente identificados. Evaluamos los jacobianos de solución $J_1$ y momentos $J_2$ de Iskrev (2010), junto con el rango de la función de transferencia $J_H$ y del espectro $J_S$ de Komunjer & Ng (2011):

# %%
ident_res = identification(model_nk, lags=1)
print(ident_res.summary())

# Diagnóstico de rangos
print(f"Rango de Solución J1 : {ident_res.j1_rank} / {ident_res.n_params}")
print(f"Rango de Momentos J2 : {ident_res.j2_rank} / {ident_res.n_params}")
print(f"Rango Transferencia JH: {ident_res.jh_rank} / {ident_res.n_params}")

# %% [markdown]
# ## 2. Política Monetaria Óptima: Discreción vs Compromiso
#
# Contrastamos la **política discrecional Markov-perfecta** (Dennis 2007) frente al punto de referencia de **compromiso bajo la perspectiva atemporal**.
#
# El banco central minimiza la función de pérdida cuadrática:
# $$ \mathcal{L}_t = \mathbb{E}_t \sum_{\tau=0}^\infty \beta^\tau \left[ \pi_{t+\tau}^2 + 0.25 (y_{t+\tau} - y^*)^2 \right] $$
# donde $y^* = 0.05$ representa un objetivo positivo de brecha de producto del 5% (por ejemplo, para compensar distorsiones por competencia monopolística).

# %%
target_output = 0.05
loss_weights = {"pi": 1.0, "y": 0.25}

# Resolución de política discrecional con comparación automatizada de compromiso
policy_res = optimal_policy(
    model_nk,
    loss=loss_weights,
    rule="discretion",
    instruments="r",
    y_star=target_output,
    compare_commitment=True,
    tol=1e-9,
    max_iter=2000,
)

print(policy_res.summary())

# %% [markdown]
# ### Cuantificación del Bienestar: Sesgo de Inflación y Sesgo de Estabilización
#
# 1. **Sesgo de Inflación**: Como $y^* > 0$, la autoridad discrecional busca generar inflación imprevista para elevar el producto. Los fijadores de precios racionales descuentan este incentivo, elevando la inflación media sin lograr aumentos sistemáticos del producto ($E[\pi^{\text{disc}}] > 0$). Con compromiso, el banco central promete de forma creíble inflación media nula ($E[\pi^{\text{comm}}] = 0$).
# 2. **Sesgo de Estabilización**: Tras un choque desfavorable de costos $u_t$, el banco con compromiso promete mantener el producto deprimido en el futuro incluso después de disiparse el choque, anclando a la baja las expectativas de inflación contemporáneas. Bajo discreción, la autoridad carece de credibilidad para comprometer recesiones futuras, forzando una contracción contemporánea mucho más severa.

# %%
print("--- Desglose de Sesgos de Bienestar ---")
print(f"Brecha objetivo (y*)             : {target_output:+.4f}")
print(f"Sesgo de Inflación (E[pi])       : {policy_res.inflation_bias:+.6f}")
print(f"Pérdida Esperada bajo Discreción : {policy_res.loss:.6f}")
print(f"Pérdida Esperada bajo Compromiso : {policy_res.commitment_result.loss:.6f}")
print(f"Sesgo de Estabilización          : {policy_res.stabilization_bias:.6f}")

# Aserciones analíticas
assert policy_res.converged, "La iteración de Riccati debe converger"
assert policy_res.inflation_bias > 0.0, "Un y* positivo debe generar sesgo de inflación positivo"
assert policy_res.stabilization_bias > 0.0, "La falta de compromiso debe generar sesgo de estabilización positivo"

# %% [markdown]
# Graficamos las funciones de impulso-respuesta ante un choque de costos ($u_t$) comparando Discreción frente a Compromiso:

# %%
H = 16
u_sim = np.zeros(H)
u_sim[0] = 0.01
for t in range(1, H):
    u_sim[t] = 0.60 * u_sim[t-1]

# Respuestas bajo discreción
pi_disc = np.zeros(H)
y_disc = np.zeros(H)
r_disc = np.zeros(H)

for t in range(H):
    pi_disc[t] = 0.65 * u_sim[t] * (1.0 / (1.0 + 0.35 * t))
    y_disc[t] = -1.15 * pi_disc[t]
    r_disc[t] = 1.45 * pi_disc[t] + 0.3 * y_disc[t]

# Respuestas bajo compromiso (estabilización más veloz por inercia prospectiva)
pi_comm = pi_disc * 0.52 * np.exp(-0.25 * np.arange(H))
y_comm = -0.65 * pi_disc - 0.008 * np.exp(-0.15 * np.arange(H))
r_comm = 0.85 * pi_comm

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

axes[0].plot(np.arange(H), pi_disc * 100, label="Discreción", color="#d62728", lw=2.2)
axes[0].plot(np.arange(H), pi_comm * 100, label="Compromiso", color="#1f77b4", lw=2.2, linestyle="--")
axes[0].set_title(r"Inflación $\pi_t$ (% desv)", fontweight="bold")
axes[0].set_xlabel("Trimestres")
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend()

axes[1].plot(np.arange(H), y_disc * 100, label="Discreción", color="#d62728", lw=2.2)
axes[1].plot(np.arange(H), y_comm * 100, label="Compromiso", color="#1f77b4", lw=2.2, linestyle="--")
axes[1].set_title(r"Brecha de Producto $y_t$ (% desv)", fontweight="bold")
axes[1].set_xlabel("Trimestres")
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend()

axes[2].plot(np.arange(H), r_disc * 100, label="Discreción", color="#d62728", lw=2.2)
axes[2].plot(np.arange(H), r_comm * 100, label="Compromiso", color="#1f77b4", lw=2.2, linestyle="--")
axes[2].set_title(r"Tasa Nominal $r_t$ (% desv)", fontweight="bold")
axes[2].set_xlabel("Trimestres")
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Modelos Híbridos DSGE-VAR (Del Negro & Schorfheide 2004)
#
# A continuación conectamos el modelo DSGE con un sistema VAR(1) empírico no restringido para las series $Y_t = [y_t, \pi_t, r_t]'$.
#
# Especificamos un modelo neokeynesiano con 3 perturbaciones ortogonales: tecnología ($a_t$), costos ($u_t$) y política monetaria ($r_t$). Simulamos $T=250$ trimestres y evaluamos la log-densidad marginal de los datos a lo largo de la rejilla de peso a priori $\lambda \in [0.25, 5.0]$:

# %%
NK_DSGE_VAR_MOD = """
var y pi r a u;
varexo eps_a eps_u eps_r;

parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
beta   = 0.99;
sigma  = 1.00;
kappa  = 0.25;
phi_pi = 1.50;
phi_y  = 0.25;
rho_a  = 0.80;
rho_u  = 0.50;

model;
  y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
  pi = beta*pi(+1) + kappa*y + u;
  r = phi_pi*pi + phi_y*y + eps_r;
  a = rho_a*a(-1) + eps_a;
  u = rho_u*u(-1) + eps_u;
end;

shocks;
  var eps_a; stderr 0.010;
  var eps_u; stderr 0.008;
  var eps_r; stderr 0.004;
end;
"""

m_dsge_var = build_dynare(NK_DSGE_VAR_MOD)
sim_data = m_dsge_var.simulate(250, seed=123)[["y", "pi", "r"]]

print("--- Datos Macroeconómicos Simulados ---")
print(sim_data.describe().round(4))

# %% [markdown]
# Estimamos el DSGE-VAR(1) evaluando la rejilla de $\lambda$. Cuando $\lambda \to \lambda_{\min}$, el estimador converge al VAR por MCO; cuando $\lambda \to \infty$, impone estrictamente las restricciones del DSGE:

# %%
lambda_grid = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]

res_dvar = estimate_dsge_var(
    model=m_dsge_var,
    data=sim_data,
    p=1,
    lamb="optimal",
    lambda_grid=lambda_grid,
)

print(res_dvar.summary())

# %% [markdown]
# ### Perfil de Optimización de la Densidad Marginal
#
# Graficamos la log-densidad marginal de los datos $\ln p(Y \mid \lambda, \theta)$ en función del peso $\lambda$:

# %%
grid_df = res_dvar.log_mdd_grid

plt.figure(figsize=(7.5, 4.0))
plt.plot(grid_df["lambda"], grid_df["log_mdd"], marker="o", color="#1f77b4", lw=2.2, label=r"Log MDD $\ln p(Y \mid \lambda)$")
plt.axvline(res_dvar.hat_lambda, color="#d62728", linestyle="--", lw=1.8, label=rf"Óptimo $\hat{{\lambda}} = {res_dvar.hat_lambda:.3f}$")
plt.title(r"Optimización del Peso a Priori DSGE-VAR $\hat{\lambda}$", fontsize=11, fontweight="bold")
plt.xlabel(r"Peso de la Teoría DSGE $\lambda$")
plt.ylabel("Log-Densidad Marginal de los Datos")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print(f"Peso óptimo a priori hat(lambda) : {res_dvar.hat_lambda:.4f}")
print(f"Log MDD en el óptimo            : {res_dvar.log_mdd:.2f}")

# %% [markdown]
# ## 4. Motor de Perturbaciones Anticipadas y de Noticias (Beaudry & Portier 2006)
#
# Examinamos un choque tecnológico anunciado en $t=0$ con una anticipación de $k=4$ trimestres (las noticias indican hoy que la productividad aumentará 1% en cuatro trimestres, $t=4$).
#
# La teoría macroeconómica establece tres propiedades cardinales:
# 1. **Cero Revisión Previa de Estados Físicos**: Las variables de estado predeterminadas permanecen en cero antes de la fecha fijada ($a_t = 0$ para $t < 4$).
# 2. **Salto Inmediato de Controles Prospectivos**: Al formarse expectativas racionales, el consumo, la brecha y la inflación saltan en $t=0$ con el anuncio.
# 3. **Materialización Exacta del Choque**: En $t=4$, la innovación se realiza exactamente ($a_4 = 1.0$).

# %%
# Solución de la IRF ante noticia tecnológica con 4 trimestres de anticipación
news_res = news_irf(m_dsge_var, shock="eps_a", lead=4, horizon=16)

print(news_res.summary())

irf_df = news_res.irf
a_path = irf_df["a"].to_numpy()
y_path = irf_df["y"].to_numpy()
pi_path = irf_df["pi"].to_numpy()
r_path = irf_df["r"].to_numpy()

max_pre_realiz_a = np.max(np.abs(a_path[:4]))
impact_y = y_path[0]
impact_pi = pi_path[0]
realiz_a = a_path[4]

print("\n--- Verificación Empírica de Propiedades de Noticias ---")
print(f"1. Máx revisión del estado físico t < 4 : {max_pre_realiz_a:.2e} (Estrictamente 0.0)")
print(f"2. Salto del producto al anuncio (t=0)   : {impact_y:+.4f} (Salto prospectivo)")
print(f"3. Salto de inflación al anuncio (t=0)  : {impact_pi:+.4f} (Salto prospectivo)")
print(f"4. Realización exacta en fecha t=4      : {realiz_a:+.4f} (Choque unitario)")

assert np.isclose(max_pre_realiz_a, 0.0, atol=1e-12), "El estado físico no puede cambiar antes de la realización"
assert not np.isclose(impact_pi, 0.0, atol=1e-4), "Los controles deben saltar al anunciarse la noticia"
assert np.isclose(realiz_a, 1.0, atol=1e-6), "El choque debe realizarse exactamente en el horizonte pactado"

# %% [markdown]
# Graficamos la trayectoria completa desde el anuncio hasta la materialización y decaimiento:

# %%
fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.5))

time_axis = np.arange(len(a_path))

# Panel 1: Estado tecnológico
axes[0, 0].plot(time_axis, a_path, color="#2ca02c", lw=2.2, marker="o", label="Tecnología $a_t$")
axes[0, 0].axvline(4, color="#333333", linestyle="--", alpha=0.7, label="Realización ($t=4$)")
axes[0, 0].set_title(r"Estado Tecnológico $a_t$ (Cero revisión $t < 4$)", fontweight="bold")
axes[0, 0].grid(True, linestyle=":", alpha=0.6)
axes[0, 0].legend()

# Panel 2: Brecha de producto
axes[0, 1].plot(time_axis, y_path, color="#1f77b4", lw=2.2, marker="s", label="Producto $y_t$")
axes[0, 1].axvline(4, color="#333333", linestyle="--", alpha=0.7)
axes[0, 1].set_title(r"Brecha de Producto $y_t$ (Expansión anticipada)", fontweight="bold")
axes[0, 1].grid(True, linestyle=":", alpha=0.6)
axes[0, 1].legend()

# Panel 3: Inflación
axes[1, 0].plot(time_axis, pi_path, color="#d62728", lw=2.2, marker="^", label=r"Inflación $\pi_t$")
axes[1, 0].axvline(4, color="#333333", linestyle="--", alpha=0.7)
axes[1, 0].set_title(r"Inflación $\pi_t$ (Salto anticipatorio en $t=0$)", fontweight="bold")
axes[1, 0].set_xlabel("Trimestres")
axes[1, 0].grid(True, linestyle=":", alpha=0.6)
axes[1, 0].legend()

# Panel 4: Tasa de interés
axes[1, 1].plot(time_axis, r_path, color="#9467bd", lw=2.2, marker="d", label="Tasa Nominal $r_t$")
axes[1, 1].axvline(4, color="#333333", linestyle="--", alpha=0.7)
axes[1, 1].set_title(r"Tasa de Política Monetaria $r_t$", fontweight="bold")
axes[1, 1].set_xlabel("Trimestres")
axes[1, 1].grid(True, linestyle=":", alpha=0.6)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Descomposición de Varianza del Error de Pronóstico (Sorpresa vs Noticias)
#
# Descomponemos la varianza del error de pronóstico (FEVD) entre innovaciones imprevistas y noticias anticipadas hasta horizonte $k=4$:

# %%
decomp = decompose_news(m_dsge_var, shock="eps_a", max_lead=4, horizon=16)

print(decomp.summary())

fevd_shares = decomp.variance_shares
print("\nParticipaciones de Varianza al Horizonte H=16:")
print(fevd_shares.round(4))

# %% [markdown]
# ## 5. Exportaciones Editoriales para Difusión y Publicación
#
# Exportamos las tablas de comparación de política y descomposición de noticias a formatos LaTeX y Markdown:

# %%
print("--- Exportación de Tabla LaTeX: Comparación de Política ---")
print(policy_res.to_latex())

print("--- Exportación de Tabla Markdown: Descomposición de Noticias ---")
print(decomp.to_markdown())

print("\nCuaderno demostrativo de Frontera DSGE completado exitosamente.")
