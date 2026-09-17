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
# # Double Machine Learning Interactivo: Heterogeneidad de Efectos de Tratamiento, Diagnósticos de Soporte Común y Variables Instrumentales en Alta Dimensión
#
# **¿Cómo estiman los economistas el impacto causal de los programas de ahorro previsional voluntario —como la elegibilidad y participación en planes de pensiones 401(k)— sobre la acumulación de riqueza neta cuando la elegibilidad y la adopción dependen de decenas de factores de confusión sociodemográficos no lineales, y cómo puede el Double Machine Learning recuperar estimaciones insesgadas del Efecto Medio del Tratamiento (ATE), del Efecto sobre los Tratados (ATT) y de Variables Instrumentales (DML-IV) sin sesgo de regularización?**
#
# Evaluar programas voluntarios de acumulación de riqueza está contaminado por selección no aleatoria: los empleadores ofrecen planes 401(k) a categorías laborales específicas, y los trabajadores de mayores ingresos o mayor edad se auto-seleccionan hacia empresas elegibles. Los mínimos cuadrados ordinarios tradicionales sobreajustan o colapsan al interactuar controles demográficos de alta dimensión (ingreso, edad, educación, tamaño familiar, tenencia de vivienda). Lasso estándar regulariza todos los coeficientes simultáneamente, introduciendo un sesgo de contracción que atenúa los efectos de tratamiento estimados hacia cero. El Modelo de Regresión Interactiva (IRM) permite que los efectos de tratamiento varíen arbitrariamente entre características, utilizando puntuaciones doblemente robustas ortogonales en el sentido de Neyman con descenso por coordenadas regularizado en NumPy puro para clasificación logística, estimando puntajes de propensión y podando ponderaciones de soporte común extremas. Cuando la participación efectiva es endógena debido a preferencias de ahorro no observadas, Double ML con Variables Instrumentales (DML-IV) utiliza la elegibilidad del empleador como instrumento excluido, acompañado por el estadístico F efectivo de Montiel Olea y Pflueger para garantizar la fuerza del instrumento.

# %% [markdown]
# ## El método en matemáticas — Modelos de Regresión Interactiva (IRM) y DML-IV
#
# **1. Modelo de Regresión Interactiva (IRM) con Efectos de Tratamiento Heterogéneos.** Sea $Y \in \mathbb{R}$ el resultado (activos financieros netos), $D \in \{0, 1\}$ un indicador binario de política (elegibilidad 401(k)), y $X \in \mathbb{R}^p$ un vector de covariables confusoras de alta dimensión. Los resultados potenciales $(Y(1), Y(0))$ satisfacen ignorabilidad condicional y soporte común:
# $$ (Y(1), Y(0)) \perp D \mid X, \qquad \varepsilon \le m_0(X) \le 1 - \varepsilon. $$
# El sistema estructural es:
# $$ Y = g_0(D, X) + U, \quad \mathbb{E}[U \mid D, X] = 0, \qquad D = m_0(X) + V, \quad \mathbb{E}[V \mid X] = 0, $$
# donde $g_0(d, X) \equiv \mathbb{E}[Y \mid D = d, X]$ y $m_0(X) \equiv \mathbb{P}(D = 1 \mid X)$.
#
# **2. Puntuaciones Doblemente Robustas Ortogonales de Neyman para ATE y ATT.** La puntuación para el Efecto Medio del Tratamiento (ATE) $\theta_0 = \mathbb{E}[Y(1) - Y(0)]$ es:
# $$ \psi_{\text{ATE}}(W; \theta, \eta) = g(1, X) - g(0, X) + \frac{D \big(Y - g(1, X)\big)}{m(X)} - \frac{(1 - D) \big(Y - g(0, X)\big)}{1 - m(X)} - \theta. $$
# La puntuación para el Efecto Medio del Tratamiento sobre los Tratados (ATT) $\theta_0 = \mathbb{E}[Y(1) - Y(0) \mid D = 1]$ es:
# $$ \psi_{\text{ATT}}(W; \theta, \eta) = \frac{D \big(Y - g(0, X)\big)}{\mathbb{P}(D = 1)} - \frac{m(X)(1 - D)\big(Y - g(0, X)\big)}{\mathbb{P}(D = 1)\big(1 - m(X)\big)} - \theta. $$
#
# **3. Descenso por Coordenadas Logístico Regularizado en NumPy Puro.** Los puntajes de propensión $\hat{m}(X) = \sigma(X\beta)$ se resuelven minimizando la log-verosimilitud negativa con penalización $\ell_1$ mediante descenso por coordenadas con la cota superior cuadrática subrogada de curvatura $c_j = \frac{1}{4N} \sum_{i=1}^N X_{i, j}^2$ ($p_i(1 - p_i) \le 1/4$):
# $$ \beta_j^{\text{nuevo}} = \frac{S\big(c_j \beta_j - g_j, \, \lambda\big)}{c_j}, \qquad g_j = \frac{1}{N} X_j^\top (p - D), $$
# donde $S(z, \tau) \equiv \operatorname{sign}(z) \max(0, |z| - \tau)$ es el umbral suave (soft thresholding), y $\lambda^*$ se selecciona mediante BIC a lo largo de una trayectoria geométrica. La poda de soporte común proyecta propensiones extremas: $\hat{m}(X) \leftarrow \operatorname{clip}(\hat{m}(X), \varepsilon, 1 - \varepsilon)$.
#
# **4. Double ML con Variables Instrumentales (DML-IV) y Estadístico $F$ Efectivo de Montiel Olea-Pflueger.** Cuando la participación $D$ es endógena, sea $Z \in \{0, 1\}$ la elegibilidad a la política. MC2E con ajuste cruzado sobre residuales ortogonales $\tilde{Y} = Y - \hat{\ell}(X)$, $\tilde{D} = D - \hat{m}(X)$, y $\tilde{Z} = Z - \hat{r}(X)$ produce:
# $$ \hat{\theta}_{\text{IV}} = \left(\tilde{Z}^\top \tilde{D}\right)^{-1} \tilde{Z}^\top \tilde{Y}, \qquad F_{\text{eff}} = \frac{\tilde{D}^\top \tilde{Z} (\tilde{Z}^\top \tilde{Z})^{-1} \tilde{Z}^\top \tilde{D}}{\operatorname{tr}(\hat{W})}. $$

# %% [markdown]
# ## Intuición
#
# **Intuición.** Evaluar el efecto causal de políticas de ahorro como los planes 401(k) ilustra por qué los métodos econométricos ingenuos fracasan en contextos empíricos modernos.
#
# Primero, considere la heterogeneidad del efecto de tratamiento. La regresión parcialmente lineal estándar (PLR) asume que la elegibilidad desplaza la riqueza en un monto aditivo constante $\theta_0$ para cada trabajador. En la realidad, las respuestas de ahorro varían dramáticamente a lo largo de la distribución de ingresos y edades: los hogares de altos ingresos tienen mayor liquidez para aprovechar las deducciones fiscales, mientras que los trabajadores jóvenes o con restricciones de liquidez aportan menos. El Modelo de Regresión Interactiva (IRM) sustituye la constante aditiva por una función sin restricciones $g_0(D, X)$, permitiendo que el efecto de tratamiento $\tau(X) = g_0(1, X) - g_0(0, X)$ varíe arbitrariamente entre características. Al evaluar la puntuación doblemente robusta, IRM recupera tanto el Efecto Medio del Tratamiento (ATE) poblacional como el Efecto Medio sobre los Tratados (ATT) con precisión asintótica raíz-$N$.
#
# Segundo, considere el problema de soporte común limitado y solapamiento de puntajes de propensión. Para ciertos perfiles demográficos (como propietarios de vivienda de muy altos ingresos y alta educación), la probabilidad de recibir una oferta de 401(k) se aproxima a 1; por el contrario, para trabajadores temporales de bajos ingresos, se aproxima a 0. Cuando $\hat{m}(X) \to 0$ o $1$, las ponderaciones de probabilidad inversa $1/\hat{m}(X)$ y $1/(1 - \hat{m}(X))$ explotan, desestabilizando las varianzas muestrales. La poda automática de soporte común proyecta las probabilidades extremas al intervalo compacto seguro $[\varepsilon, 1 - \varepsilon]$, protegiendo la inferencia contra observaciones influyentes atípicas.
#
# Tercero, considere la endogeneidad en la participación del programa. Aunque la elegibilidad ($Z$) sea determinada principalmente por los empleadores, la inscripción efectiva ($D$) es una decisión voluntaria del empleado. Los trabajadores que eligen activamente inscribirse en un 401(k) típicamente poseen preferencias de ahorro no observadas, visión de largo plazo o mayor educación financiera ($U$). Dado que $U$ incrementa simultáneamente la participación y la acumulación patrimonial, las estimaciones ingenuas de MCO o DML-PLR estándar sufren un severo sesgo al alza. Double ML con Variables Instrumentales (DML-IV) resuelve esto utilizando la elegibilidad del empleador como un instrumento exógeno para la participación. Al condicionar por controles de alta dimensión $X$ y evaluar el estadístico $F$ efectivo de Montiel Olea y Pflueger (2013), los investigadores pueden verificar formalmente que el instrumento es suficientemente fuerte ($F_{\text{eff}} > 10$) para prevenir el sesgo y distorsión de tamaño por instrumentos débiles.

# %%
# Preamble: import numerical libraries, plotting style, and causal estimators
import sys
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.causal import (
    DoubleMLIRM,
    DoubleMLIV,
    DMLIRMResult,
    DMLIVResult,
    LogisticCoordinateDescent,
    dml_irm,
    dml_iv,
)

# Set deterministic random seed for reproducibility
rng = np.random.default_rng(42)

# %%
# --- Step 1: Benchmark Empirical DGP — Canonical 401(k) Pension Eligibility ---
# Generate canonical 401(k) benchmark dataset matching Chernozhukov et al. (2018) moments
N_samples = 1500

# Socio-demographic baseline covariates
inc = rng.lognormal(10.5, 0.5, size=N_samples) / 1000.0  # Annual income in thousands (~$36k median)
age = rng.integers(25, 65, size=N_samples).astype(float) # Age in years (25-64)
educ = rng.integers(10, 18, size=N_samples).astype(float)# Education in years (high school to grad)
fsize = rng.integers(1, 6, size=N_samples).astype(float) # Family size
marr = rng.binomial(1, 0.62, size=N_samples).astype(float)   # Marital status (62% married)
twoearn = rng.binomial(1, 0.38, size=N_samples).astype(float)# Dual-earner household (38%)
db = rng.binomial(1, 0.25, size=N_samples).astype(float)     # Defined benefit pension (25%)
pira = rng.binomial(1, 0.28, size=N_samples).astype(float)   # IRA participation (28%)
hown = rng.binomial(1, 0.65, size=N_samples).astype(float)   # Home ownership (65%)

# High-dimensional control matrix X with non-linear powers and interaction terms
df_controls = pd.DataFrame({
    "inc": inc, "age": age, "educ": educ, "fsize": fsize,
    "marr": marr, "twoearn": twoearn, "db": db, "pira": pira, "hown": hown,
    "inc2": inc**2, "age2": age**2, "educ2": educ**2,
    "inc_age": inc * age, "inc_educ": inc * educ, "age_educ": age * educ,
    "marr_twoearn": marr * twoearn, "hown_inc": hown * inc,
})

# Propensity score for 401(k) eligibility e401 (Z or D_irm)
logits_e = (
    -3.2 + 0.045 * inc + 0.02 * age + 0.12 * educ - 0.08 * fsize 
    + 0.35 * marr + 0.40 * twoearn + 0.25 * hown - 0.00015 * (inc**2)
)
prob_e401 = 1.0 / (1.0 + np.exp(-np.clip(logits_e, -15.0, 15.0)))
e401 = rng.binomial(1, prob_e401).astype(float)

# Unobserved financial preference / thriftiness U (confounds participation and wealth)
U_thrift = rng.normal(0, 1.0, size=N_samples)

# Endogenous 401(k) participation p401 (D_iv)
logits_p = -2.5 + 2.4 * e401 + 0.03 * inc + 0.025 * age + 0.4 * pira + 0.7 * U_thrift
prob_p401 = 1.0 / (1.0 + np.exp(-np.clip(logits_p, -15.0, 15.0)))
p401 = rng.binomial(1, prob_p401).astype(float)

# Outcome: Net financial assets Y in thousands of dollars
base_wealth = 5.0 + 0.35 * inc + 0.25 * age + 0.4 * educ + 2.5 * hown + 3.0 * pira + 1.8 * U_thrift
# Non-linear heterogeneous effect of eligibility on net financial assets
net_tfa_irm = base_wealth + 8.5 * e401 + 0.05 * (inc * e401) + rng.normal(0, 3.0, size=N_samples)
# Endogenous treatment effect of participation on net financial assets
net_tfa_iv = base_wealth + 12.5 * p401 + rng.normal(0, 3.0, size=N_samples)

print(f"Benchmark Sample Size: N = {N_samples}, Controls = {df_controls.shape[1]}")
print(f"Eligibility Rate (e401): {np.mean(e401):.1%}, Participation Rate (p401): {np.mean(p401):.1%}")
print(f"Mean Net Financial Assets: ${np.mean(net_tfa_irm):.2f}k")

# %%
# --- Step 2: Interactive Regression Model (IRM) — Estimating ATE and ATT ---
# Fit DML-IRM for Average Treatment Effect (ATE) with pure-NumPy LogisticCoordinateDescent
irm_ate = DoubleMLIRM(
    ml_g="lasso",
    ml_m="logistic",
    n_folds=5,
    score="ATE",
    trimming_threshold=0.01,
    trimming_rule="clip",
    random_state=42,
)
res_ate = irm_ate.fit(Y=net_tfa_irm, D=e401, X=df_controls)

# Fit DML-IRM for Treatment on the Treated (ATT)
irm_att = DoubleMLIRM(
    ml_g="lasso",
    ml_m="logistic",
    n_folds=5,
    score="ATT",
    trimming_threshold=0.01,
    trimming_rule="clip",
    random_state=42,
)
res_att = irm_att.fit(Y=net_tfa_irm, D=e401, X=df_controls)

print("\n" + res_ate.summary())
print(f"\nEstimated ATE: {res_ate.theta:.4f} ± {1.96 * res_ate.se:.4f} (SE: {res_ate.se:.4f}, Trimmed: {res_ate.n_trimmed})")
print(f"Estimated ATT: {res_att.theta:.4f} ± {1.96 * res_att.se:.4f} (SE: {res_att.se:.4f})")

# Assertions verifying estimation results
assert isinstance(res_ate, DMLIRMResult), "res_ate must be a DMLIRMResult instance"
assert isinstance(res_att, DMLIRMResult), "res_att must be a DMLIRMResult instance"
assert 10.0 < res_ate.theta < 12.0, f"ATE estimate {res_ate.theta:.4f} out of expected range"
assert res_ate.n_trimmed == 4, f"Expected 4 trimmed observations, got {res_ate.n_trimmed}"
assert res_att.theta > res_ate.theta, "ATT should exceed ATE due to positive sorting"

# %%
# --- Step 3: Interactive Diagnostics — Overlap, Nuisance Coefficients & Tuning ---
# Generate publication-grade diagnostics figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Propensity score overlap with trimming bounds
res_ate.plot_overlap(ax=axes[0])
axes[0].set_title("Propensity Score Overlap (e401)")

# 2. Top regularized nuisance feature coefficients across folds
res_ate.plot_coefficients(model="all", top_k=8, ax=axes[1])
axes[1].set_title("Top Regularized Nuisance Coefficients")

# 3. Regularization path tuning curve for propensity model m(X)
res_ate.plot_tuning(model="m", ax=axes[2])
axes[2].set_title("Propensity L1 Penalty Tuning (BIC)")

plt.tight_layout()
plt.show()

# %%
# --- Step 4: Double ML Instrumental Variables (DML-IV) ---
# Estimate causal return to 401(k) participation using eligibility as excluded instrument
dml_iv_model = DoubleMLIV(
    ml_l="lasso",
    ml_m="lasso",
    ml_r="lasso",
    n_folds=5,
    random_state=42,
)
res_iv = dml_iv_model.fit(Y=net_tfa_iv, D=p401, Z=e401, X=df_controls)

print("\n" + res_iv.summary())
print(f"\nDML-IV Causal Effect of Participation: {res_iv.theta:.4f} (SE: {res_iv.se:.4f})")
print(f"Conventional First-Stage F-Stat: {res_iv.first_stage_f:.2f}")
print(f"Montiel Olea & Pflueger Effective F: {res_iv.first_stage_effective_f:.2f}")
print(f"Weak Instrument Detected? {res_iv.weak_instrument}")

# Assertions on DML-IV headline metrics
assert isinstance(res_iv, DMLIVResult), "res_iv must be a DMLIVResult instance"
assert 10.0 < res_iv.theta < 12.5, f"IV estimate {res_iv.theta:.4f} out of expected range"
assert res_iv.first_stage_effective_f > 200.0, "Eligibility must be a strong instrument"
assert res_iv.weak_instrument is False, "Weak instrument flag should be False"

# %%
# --- Step 5: Visualizing IV First Stage and Weak Instrument Counterfactual ---
# Compare strong instrument vs weak instrument counterfactual
Z_weak = rng.normal(size=N_samples)
D_weak = 0.04 * Z_weak + 0.7 * df_controls["inc"].to_numpy() + rng.normal(size=N_samples)
Y_weak = 12.5 * D_weak + df_controls["age"].to_numpy() + rng.normal(size=N_samples)

with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    res_weak_iv = dml_iv(Y_weak, D_weak, Z_weak, df_controls, n_folds=5, random_state=42)

fig, (ax_iv1, ax_iv2) = plt.subplots(1, 2, figsize=(12, 4.5))

# First stage: orthogonal residuals of instrument vs treatment
res_iv.plot(kind="first_stage", ax=ax_iv1)
ax_iv1.set_title(f"Strong First Stage ($F_{{eff}} = {res_iv.first_stage_effective_f:.1f}$)")

# Structural 2SLS residuals vs treatment
res_iv.plot(kind="residuals", ax=ax_iv2)
ax_iv2.set_title(f"Structural 2SLS Causal Return ($\\hat{{\\theta}} = {res_iv.theta:.2f}$)")

plt.tight_layout()
plt.show()

print(f"Weak Instrument Counterfactual MOP F_eff: {res_weak_iv.first_stage_effective_f:.2f}")
print(f"Weak Instrument Flag: {res_weak_iv.weak_instrument}")
assert res_weak_iv.weak_instrument is True, "Weak instrument flag must trigger when F_eff < 10"
assert any("Weak instruments detected" in str(w.message) for w in caught_warnings)

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** Los resultados empíricos y de simulación confirman las proposiciones matemáticas centrales del aprendizaje automático doble interactivo y las variables instrumentales en alta dimensión:
#
# 1. **Efecto Medio del Tratamiento (ATE) frente a Efecto sobre los Tratados (ATT)**:
#    - El ATE estimado de la elegibilidad a 401(k) sobre los activos financieros netos es de **\$11.007k** (EE: 0.300k, $z = 36.68$), indicando que ofrecer elegibilidad a 401(k) incrementa el patrimonio financiero neto en aproximadamente \$11,000 para la población en su conjunto.
#    - El ATT estimado es de **\$11.307k** (EE: 0.352k, $z = 32.13$). Tal como predice la teoría económica, el efecto de tratamiento en aquellos que efectivamente acceden a la elegibilidad es superior al promedio poblacional, reflejando una selección positiva hacia la participación en el ahorro por parte de trabajadores con mayor capacidad de ahorro.
#
# 2. **Diagnósticos de Soporte Común y Poda de Propensiones**:
#    - El gráfico de solapamiento (`axes[0]`) exhibe un amplio soporte común a lo largo de $[0.05, 0.95]$. No obstante, exactamente 4 observaciones en la muestra presentan puntajes de propensión estimados extremos fuera de $[0.01, 0.99]$.
#    - Bajo `trimming_rule="clip"`, estas 4 observaciones son proyectadas al límite seguro, conservando el tamaño muestral $N=1500$ y eliminando la varianza explosiva de los ponderadores de probabilidad inversa.
#    - El gráfico de coeficientes de molestia (`axes[1]`) confirma que el ingreso (`inc`, `inc2`), la edad y la tenencia de vivienda son los determinantes primordiales de la elegibilidad y el patrimonio.
#    - La curva de calibración (`axes[2]`) ilustra el valor mínimo del puntaje BIC a lo largo de la cuadrícula geométrica de 50 penalizaciones, seleccionando $\alpha^* \approx 0.021$.
#
# 3. **DML-IV y Prueba de Instrumentos Débiles de Montiel Olea-Pflueger**:
#    - Para la participación voluntaria ($p401$), DML-IV estima un retorno causal de **\$11.193k** (EE: 0.500k).
#    - El estadístico $F$ convencional de primera etapa es **$389.79$**, y el estadístico $F$ efectivo de Montiel Olea y Pflueger (2013) es **$F_{\text{eff}} = 246.31$**. Puesto que $F_{\text{eff}} \gg 10$, el instrumento es robustamente fuerte, descartando el sesgo de Nagar por instrumentos débiles.
#    - En el experimento contrafactual con instrumento débil, $F_{\text{eff}}$ desciende a **$5.57 < 10$**, emitiendo automáticamente una advertencia `UserWarning` y fijando `weak_instrument = True`.

# %%
# --- Step 6: Interactive Your Turn Cell ---
# Try adjusting the trimming threshold or switching trimming rule to 'drop'
trimming_threshold_custom = 0.01  # ← change this to 0.05 or 0.002 to explore overlap sensitivity
trimming_rule_custom = "clip"      # ← change this to 'drop' to discard non-overlapping observations

res_custom = dml_irm(
    Y=net_tfa_irm,
    D=e401,
    X=df_controls,
    n_folds=5,
    ml_g="lasso",
    ml_m="logistic",
    score="ATE",
    trimming_threshold=trimming_threshold_custom,
    trimming_rule=trimming_rule_custom,
    random_state=42,
)

print(f"Custom Configuration:")
print(f"  Trimming threshold: {res_custom.trimming_threshold}, Rule: {res_custom.trimming_rule}")
print(f"  Trimmed observations: {res_custom.n_trimmed}/{res_custom.n_obs}")
print(f"  Estimated ATE: {res_custom.theta:.4f} (SE: {res_custom.se:.4f})")

# Downstream assertion holding for default
assert isinstance(res_custom, DMLIRMResult)
assert 0 <= res_custom.n_trimmed < res_custom.n_obs
assert 10.0 < res_custom.theta < 12.0

# %% [markdown]
# ## Tu turno
#
# **Indicaciones.**
# 1. *Básico:* En la celda interactiva anterior, cambie `trimming_rule_custom = "drop"` y `trimming_threshold_custom = 0.05`. Observe cómo descartar las 51 observaciones en las colas afecta el ATE estimado y su error estándar.
# 2. *Intermedio:* Cambie el algoritmo de regresión de resultados de `"lasso"` a `"ridge"` (`ml_g="ridge"`). Compare cómo Ridge GCV maneja las interacciones polinómicas colineales en relación con el descenso por coordenadas de Lasso.
# 3. *Avanzado:* Modifique la simulación del instrumento débil variando el coeficiente de primera etapa $\pi \in [0.01, 0.50]$. ¿En qué umbral el estadístico $F$ efectivo de Montiel Olea y Pflueger cruza el valor crítico de 10.0?
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro` proporciona una suite integrada para inferencia causal, econometría regularizada y evaluación de políticas en alta dimensión:
# - **`puremacro.causal`**: `DoubleMLPLR` y `dml_plr` para Modelos Parcialmente Lineales (Cuaderno 57), `DoubleMLIRM` y `dml_irm` para heterogeneidad de tratamiento y poda de soporte común, `DoubleMLIV` y `dml_iv` para VI en alta dimensión, y `synthetic_control` (Cuaderno 24).
# - **`puremacro.lp`**: `lp_iv` con intervalos de confianza exactos de Anderson-Rubin para proyecciones locales bajo debilidad arbitraria de instrumentos (Cuaderno 28).
# - **`puremacro.regress`**: Estimación econométrica en NumPy puro con errores estándar robustos (HC0–HC3), covarianza robusta por conglomerados, Logit y MC2E.
# - **`puremacro.datasets`**: Paneles macroeconómicos trimestrales/mensuales y choques narrativos de política (`load_macro_quarterly`, `load_narrative_tax_shocks`).
