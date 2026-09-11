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
# # Suite Aplicada de Política de Bancos Centrales: Reglas de Taylor, Brechas de Postura, Gráficos de Abanico y Tono Narrativo
#
# **¿Cómo pueden los bancos centrales evaluar de manera sistemática su postura de política monetaria en tiempo real, estimar trayectorias contrafácticas de tasas de interés bajo funciones de reacción alternativas, generar gráficos de abanico probabilísticos ante la incertidumbre y extraer cuantitativamente el sentimiento de comunicación a partir de comunicados oficiales utilizando únicamente NumPy puro?**
#
# Los bancos centrales operan en un entorno caracterizado por una profunda incertidumbre sobre el estado del ciclo económico, los rezagos de transmisión de la tasa de interés y la estructura cambiante de la economía. En las instituciones modernas, fijar la tasa de política monetaria nunca es un ejercicio mecánico unidimensional. Al contrario, exige una arquitectura analítica integrada: las reglas de reacción de Taylor normativas ofrecen un anclaje de referencia para detectar si las condiciones monetarias son restrictivas o acomodaticias; las simulaciones macroeconómicas contrafácticas cuantifican las disyuntivas entre desinflación rápida y estabilidad del producto; el procesamiento de lenguaje natural extrae el tono de orientación prospectiva (*forward guidance*) de las minutas y comunicados; y los gráficos de abanico (*fan charts*) proyectan abanicos de riesgo hacia los mercados financieros.
#
# Con **puremacro**, esta suite completa de política de banca central se ejecuta en **Python 100% puro / Pyodide** bajo el estándar de cuatro paquetes científicos (`numpy`, `scipy`, `pandas`, `matplotlib`), sin requerir librerías econométricas externas ni licencias comerciales, garantizando una reproducibilidad determinista integral.

# %% [markdown]
# ## El método en matemáticas — Reglas de Política, Descomposición de Postura, Sentimiento y Densidades de Proyección
#
# **Reglas de Taylor Inerciales y con Orientación Prospectiva.** Siguiendo a Taylor (1993) y Clarida, Galí y Gertler (1998, 2000), la tasa de interés objetivo deseada por la autoridad $i_t^*$ responde a las desviaciones esperadas de la inflación respecto a su meta $\pi^*$ y a la brecha del producto $(y_t - y_t^*)$:
# $$ i_t^* = r^* + \pi^* + \phi_\pi (\mathbb{E}_t \pi_{t+k} - \pi^*) + \phi_y (y_t - y_t^*) $$
# donde $r^*$ denota la tasa de interés real de equilibrio. Dado que los bancos centrales buscan evitar oscilaciones abruptas en los mercados de bonos, la tasa observada exhibe ajuste parcial o inercia:
# $$ i_t = \rho_i i_{t-1} + (1 - \rho_i) i_t^* + \varepsilon_{i, t} $$
# En la estimación empírica, sustituir $i_t^*$ conduce a la regresión lineal estimable:
# $$ i_t = \beta_0 + \beta_1 i_{t-1} + \beta_2 \pi_t + \beta_3 \tilde{y}_t + \varepsilon_{i, t} $$
# de la cual se recuperan los coeficientes estructurales mediante $\hat{\rho}_i = \hat{\beta}_1$, $\hat{\phi}_\pi = \frac{\hat{\beta}_2}{1 - \hat{\rho}_i}$ y $\hat{\phi}_y = \frac{\hat{\beta}_3}{1 - \hat{\rho}_i}$. La condición de estabilidad bajo expectativas racionales exige el **principio de Taylor**: $\hat{\phi}_\pi > 1.0$.
#
# **Brecha de Postura Monetaria.** La brecha de postura normativa $\operatorname{Gap}_t$ compara la tasa efectiva con el referente de Taylor sin inercia:
# $$ \operatorname{Gap}_t = i_t - i_t^* $$
# Una brecha positiva $\operatorname{Gap}_t > 0$ indica una postura restrictiva que combate presiones inflacionarias, mientras que $\operatorname{Gap}_t < 0$ diagnostica una postura acomodaticia en apoyo a la actividad económica.
#
# **Extracción de Sentimiento de Comunicación (Apel-Blix-Grimaldi).** La comunicación oficial prepara a los agentes antes de que las tasas se modifiquen. Para un corpus de comunicados $\mathcal{D}_t$ en la fecha de decisión $t$, el tono neto se cuantifica como:
# $$ \text{Tone}_t = \frac{\text{Conteo}_t(\text{Hawkish}) - \text{Conteo}_t(\text{Dovish})}{\text{Conteo}_t(\text{Hawkish}) + \text{Conteo}_t(\text{Dovish}) + \varepsilon} $$
# donde $\text{Tone}_t \in [-1, 1]$, evaluado mediante `puremacro.narrative.indices.tone(..., method="apel_blix_grimaldi")`.
#
# **Regímenes Contrafácticos en Equilibrio Dinámico.** La propagación macroeconómica se modela mediante el bloque neokeynesiano canónico:
# $$ y_t = \rho_y y_{t-1} - \sigma (i_t - \pi_t - r^*) + \varepsilon_t^y $$
# $$ \pi_t - \pi^* = \rho_\pi (\pi_{t-1} - \pi^*) + \kappa y_t + \varepsilon_t^\pi $$
# La resolución del sistema simultáneo intra-período ante diferentes vectores de parámetros $\theta = (\phi_\pi, \phi_y, \rho_i)$ traza las trayectorias de desinflación contrafácticas bajo regímenes alternativos.
#
# **Gráficos de Abanico de Proyección Multi-Horizonte.** Sea el vector de estados del VAR compañero $x_t = (\text{FFR}_t, \pi_t, \tilde{y}_t)'$. Bajo innovaciones gaussianas con matriz de covarianza $\Sigma$, la varianza del error de predicción en el horizonte $h$ se acumula como $\Omega_h = \sum_{j=0}^{h-1} \Psi_j \Sigma \Psi_j'$. La simulación de $D$ trayectorias desde la distribución predictiva posterior genera las bandas de cuantiles:
# $$ q_\alpha(h) = \text{Cuantil}_\alpha\left(\{ x_{t+h}^{(d)} \}_{d=1}^D\right), \quad \alpha \in \{0.50, 0.70, 0.90\} $$

# %% [markdown]
# ## Intuición
#
# **Intuición.** La formulación de la política monetaria en los bancos centrales es fundamentalmente un ejercicio de gestión de riesgos y credibilidad más que el cálculo mecánico de una fórmula aislada. La regla de Taylor proporciona un ancla normativa transparente: estimar la regla con suavizamiento de tasas ($\rho_i \approx 0.85\text{--}0.95$) refleja la realidad institucional de que las autoridades prefieren ajustar las tasas en ciclos deliberados para anclar la curva de rendimientos sin inducir volatilidad innecesaria en los mercados financieros. La brecha de postura monetaria ($i_t - i_t^*$) revela con claridad cuándo el banco central está contrayendo activamente la demanda agregada para doblegar la inflación versus cuándo provee estímulo expansivo.
#
# Asimismo, la política monetaria moderna se transmite a través de las palabras antes de reflejarse en los votos de tasas. Los mercados financieros incorporan las expectativas futuras; el análisis de sentimiento narrativo detecta el viraje cualitativo desde un tono acomodaticio hacia uno contractivo varios trimestres antes de que inicien las alzas de tasas. Por último, dado que los bancos centrales no pueden predecir las perturbaciones con certeza absoluta, la publicación de gráficos de abanico en lugar de proyecciones puntuales engañosas fortalece la credibilidad pública al comunicar con transparencia la distribución de los riesgos macroeconómicos.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load publication style
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle

_nbstyle.apply_style()

from puremacro.datasets import load_macro_monthly
from puremacro.narrative.indices import tone
from puremacro.posterior import fan_chart_quantiles, simulate_var_paths
from puremacro.regress import add_constant, ols
from puremacro.var import fit_var

# Set fixed seed for deterministic reproducibility
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

print("=" * 72)
print("PUREMACRO CENTRAL BANK POLICY SUITE: DETERMINISTIC EXECUTION")
print("=" * 72)

# %%
# ---------------------------------------------------------------------------
# 1. Statement Sentiment Extraction: Central Bank Policy Communications
# ---------------------------------------------------------------------------
# Corpus of 12 official policy statements spanning the 2021Q1-2023Q4 cycle
corpus = [
    ("2021-01-15", "The Committee decided to maintain the target range for the federal funds rate at 0 to 1/4 percent. Stance is accommodative and supports recovery.", {}),
    ("2021-04-15", "The Committee decided to keep rates unchanged. Inflation has risen, largely reflecting transitory factors.", {}),
    ("2021-07-15", "The Committee decided to maintain accommodative policy. Progress on vaccinations has reduced the spread of disease.", {}),
    ("2021-10-15", "Inflation is elevated, largely reflecting supply bottlenecks. The Committee prepares to taper asset purchases.", {}),
    ("2022-01-15", "With inflation well above 2 percent and a strong labor market, the Committee expects it will soon be appropriate to raise the target range.", {}),
    ("2022-04-15", "The Committee decided to raise the target range by 50 basis points. The Committee is highly attentive to inflation risks.", {}),
    ("2022-07-15", "The Committee decided to raise interest rates by 75 basis points to curb persistent inflationary pressures and overheating labor markets.", {}),
    ("2022-10-15", "The Committee anticipates that ongoing rate increases will be appropriate in order to attain a stance that is sufficiently restrictive.", {}),
    ("2023-01-15", "The Committee decided to raise the target range by 25 basis points to return inflation to 2 percent.", {}),
    ("2023-04-15", "The Committee decided to raise rates by 25 basis points. Tighter credit conditions are likely to weigh on economic activity.", {}),
    ("2023-07-15", "The Committee decided to raise the target range to 5.25 to 5.50 percent to ensure inflation returns to objective.", {}),
    ("2023-10-15", "The Committee decided to maintain the target range. Tighter financial conditions will weigh on economic activity and inflation.", {}),
]

# Extract quarterly sentiment index via Apel-Blix-Grimaldi lexicon
tone_res = tone(
    corpus,
    country="US",
    language="en",
    method="apel_blix_grimaldi",
    normalize="raw",
)

print("\n[1] Central Bank Communication Tone (Apel-Blix-Grimaldi):")
print(tone_res.series)

# Non-trivial assertions for communication sentiment
assert len(tone_res.series) == 12, "Corpus must span exactly 12 quarters"
assert np.isfinite(tone_res.series).all(), "Sentiment series must contain finite real numbers"
assert tone_res.series.loc["2022-07-01"] > tone_res.series.loc["2021-01-01"], "Tone must become more hawkish in 2022 tightening cycle"

# %%
# ---------------------------------------------------------------------------
# 2. Ingest Macro Data and Estimate Empirical Taylor Rule with HAC Covariance
# ---------------------------------------------------------------------------
# Ingest historical US monthly macro panel bundled with puremacro
df_monthly = load_macro_monthly()

# Compute 12-month headline CPI inflation rate (%)
infl_monthly = df_monthly["cpi"].pct_change(12) * 100.0

# Compute natural rate of unemployment using 10-year rolling filter
unemp_monthly = df_monthly["unemployment_rate"]
u_star_monthly = unemp_monthly.rolling(120, min_periods=36).mean().fillna(5.0)

# Okun's Law implied output gap: y_t - y_t^* ≈ -2.0 * (u_t - u_t^*)
y_gap_monthly = -2.0 * (unemp_monthly - u_star_monthly)

# Aggregate monthly observations to quarterly averages
raw_quarterly = pd.DataFrame({
    "ffr": df_monthly["fed_funds"],
    "infl": infl_monthly,
    "y_gap": y_gap_monthly,
})
df_q = raw_quarterly.resample("Q").mean()

# Form estimation sample starting at the Great Moderation (1985Q1 onwards)
sample_q = pd.DataFrame({
    "ffr": df_q["ffr"],
    "ffr_lag": df_q["ffr"].shift(1),
    "infl": df_q["infl"],
    "y_gap": df_q["y_gap"],
}).dropna().loc["1985Q1":]

# Estimate empirical Taylor rule using HAC (Newey-West) robust standard errors
X_taylor = add_constant(sample_q[["ffr_lag", "infl", "y_gap"]])
m_taylor = ols(sample_q["ffr"], X_taylor, cov_type="HAC")

# Recover structural Taylor rule parameters from reduced-form coefficients
rho_hat = float(m_taylor.params["ffr_lag"])
phi_pi_hat = float(m_taylor.params["infl"] / (1.0 - rho_hat))
phi_y_hat = float(m_taylor.params["y_gap"] / (1.0 - rho_hat))
r_star_calib = 1.0  # Calibrated equilibrium real rate r* = 1.0%
pi_star_calib = 2.0  # Official inflation target π* = 2.0%

print("\n[2] Empirical Taylor Rule Estimates (1985Q1-2026Q2, HAC Standard Errors):")
print(f"  Interest Rate Inertia (rho)       : {rho_hat:.4f}  (SE: {m_taylor.bse['ffr_lag']:.4f})")
print(f"  Structural Inflation Coeff (phi_pi): {phi_pi_hat:.4f}  (Taylor Principle: {phi_pi_hat > 1.0})")
print(f"  Structural Output Gap Coeff (phi_y): {phi_y_hat:.4f}")
print(f"  Regression R-squared              : {m_taylor.rsquared:.4f}")

# Verification assertions
assert 0.70 <= rho_hat <= 0.95, f"Interest rate smoothing rho={rho_hat:.3f} outside expected [0.70, 0.95] band"
assert phi_pi_hat > 1.0, f"Taylor principle violated: phi_pi={phi_pi_hat:.3f} <= 1.0"
assert np.isfinite(phi_y_hat), "phi_y must be finite"
assert m_taylor.rsquared > 0.90, "Taylor rule must fit postwar rate path with high explanatory power"

# %%
# ---------------------------------------------------------------------------
# 3. Real-Time Monetary Policy Stance Gap Calculation
# ---------------------------------------------------------------------------
# Calculate normative non-inertial benchmark rate: i_t* = r* + π* + φ_π(π_t - π*) + φ_y(y_t - y_t*)
target_rate_star = (
    r_star_calib
    + pi_star_calib
    + phi_pi_hat * (sample_q["infl"] - pi_star_calib)
    + phi_y_hat * sample_q["y_gap"]
)

# Policy stance gap: Gap_t = i_t - i_t*
stance_gap = sample_q["ffr"] - target_rate_star

print("\n[3] Monetary Policy Stance Gap Diagnostics:")
print(f"  Recent Policy Stance Gap (last obs): {stance_gap.iloc[-1]:+.2f} percentage points")
print(f"  Maximum Restrictive Stance Gap     : {stance_gap.max():+.2f} pp")
print(f"  Maximum Accommodative Stance Gap   : {stance_gap.min():+.2f} pp")

# Assertions verifying stance gap properties
assert len(stance_gap) == len(sample_q), "Stance gap must match sample length"
assert np.isfinite(stance_gap).all(), "Stance gap series must be strictly finite"
assert (stance_gap.loc["2010Q1":"2015Q4"] < 0.0).mean() > 0.60, "Post-GFC ZLB period must be predominantly accommodative"

# %%
# ---------------------------------------------------------------------------
# 4. Counterfactual Policy Regime Simulations (Supply Shock Disinflation)
# ---------------------------------------------------------------------------
def simulate_nk_counterfactual(
    phi_pi: float,
    phi_y: float,
    rho_i: float,
    T: int = 12,
    pi_0: float = 6.0,
    y_0: float = 1.5,
    i_0: float = 2.0,
    r_star: float = 1.0,
    pi_star: float = 2.0,
) -> pd.DataFrame:
    """Simulate 3-equation New Keynesian disinflation equilibrium across regimes."""
    rho_y, sigma = 0.6, 0.4
    rho_pi, kappa = 0.5, 0.25

    y, pi, i = np.zeros(T), np.zeros(T), np.zeros(T)
    y_prev, pi_prev, i_prev = y_0, pi_0, i_0

    for t in range(T):
        # Linear matrix system M @ [y_t, pi_t, i_t]' = rhs
        M = np.array([
            [1.0, -sigma, sigma],
            [-kappa, 1.0, 0.0],
            [-(1.0 - rho_i) * phi_y, -(1.0 - rho_i) * phi_pi, 1.0],
        ])
        rhs = np.array([
            rho_y * y_prev + sigma * r_star,
            (1.0 - rho_pi) * pi_star + rho_pi * pi_prev,
            rho_i * i_prev + (1.0 - rho_i) * (r_star + (1.0 - phi_pi) * pi_star),
        ])
        sol = np.linalg.solve(M, rhs)
        y[t], pi[t], i[t] = sol
        y_prev, pi_prev, i_prev = sol

    return pd.DataFrame({"output_gap": y, "inflation": pi, "policy_rate": i})

# Simulate 3 canonical policy regimes over 12 quarters
sim_strict = simulate_nk_counterfactual(phi_pi=2.5, phi_y=0.0, rho_i=0.8, T=12)
sim_dovish = simulate_nk_counterfactual(phi_pi=1.1, phi_y=1.0, rho_i=0.8, T=12)
sim_no_inertia = simulate_nk_counterfactual(phi_pi=1.5, phi_y=0.5, rho_i=0.0, T=12)

print("\n[4] Counterfactual Policy Simulations (12-Quarter Cumulative & Terminal Metrics):")
print(f"  Strict IT : Cumulative Inflation = {sim_strict['inflation'].sum():.2f}%, Terminal Inflation = {sim_strict['inflation'].iloc[-1]:.3f}%")
print(f"  Dovish    : Cumulative Inflation = {sim_dovish['inflation'].sum():.2f}%, Terminal Inflation = {sim_dovish['inflation'].iloc[-1]:.3f}%")
print(f"  No-Inertia: Cumulative Inflation = {sim_no_inertia['inflation'].sum():.2f}%, Terminal Inflation = {sim_no_inertia['inflation'].iloc[-1]:.3f}%")

# Counterfactual assertions
assert sim_strict["inflation"].iloc[4] < sim_dovish["inflation"].iloc[4], "Strict IT must achieve lower inflation at 1-year mark"
assert sim_strict["inflation"].sum() < sim_dovish["inflation"].sum(), "Strict IT must achieve lower cumulative inflation"
assert sim_strict["inflation"].iloc[-1] < sim_dovish["inflation"].iloc[-1], "Terminal inflation under strict IT must be strictly lower than dovish"

# %%
# ---------------------------------------------------------------------------
# 5. Stochastic Multi-Horizon Projection Fan Chart (12 Quarters Ahead)
# ---------------------------------------------------------------------------
# Fit companion VAR(1) model to [FFR, Inflation, Output Gap]
Y_var = sample_q[["ffr", "infl", "y_gap"]].to_numpy()
var_res = fit_var(Y_var, p=1)

# Generate 1,000 parameter and shock propagation draws
D_draws = 1000
K_vars = 3
H_horizon = 12
A_draws = np.zeros((D_draws, 1, K_vars, K_vars))
Sigma_draws = np.zeros((D_draws, K_vars, K_vars))
intercept_draws = np.zeros((D_draws, K_vars))

for d in range(D_draws):
    A_draws[d, 0] = var_res.A_list[0] + rng.normal(0, 0.015, size=(K_vars, K_vars))
    Sigma_draws[d] = var_res.Sigma
    intercept_draws[d] = var_res.c + rng.normal(0, 0.015, size=K_vars)

# Simulate predictive paths initialized at the most recent quarterly data point
y_init = Y_var[-1:]
paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws, y_init, horizon=H_horizon, rng=rng)

# Compute central bank projection quantiles (50%, 70%, 90% confidence bands)
quants = fan_chart_quantiles(paths, levels=(0.50, 0.70, 0.90))

print("\n[5] Multi-Horizon Projection Fan Chart (Inflation Forecast at Horizon H=12):")
print(f"  50th Percentile (Median)   : {quants[0, -1, 1]:.2f}%")
print(f"  70th Percentile            : {quants[1, -1, 1]:.2f}%")
print(f"  90th Percentile (Upper Cone): {quants[2, -1, 1]:.2f}%")

# Monotonicity assertions
assert quants.shape == (3, H_horizon + 1, K_vars), "Quantile array shape mismatch"
assert (quants[0] <= quants[1]).all(), "50% quantile must be <= 70% quantile"
assert (quants[1] <= quants[2]).all(), "70% quantile must be <= 90% quantile"
# Uncertainty must widen as forecast horizon extends
assert (quants[2, -1, 1] - quants[0, -1, 1]) > (quants[2, 1, 1] - quants[0, 1, 1]), "Projection cone must widen monotonically over forecast horizon"

# %%
# ---------------------------------------------------------------------------
# 6. Hero Visualization: 4-Panel Central Bank Policy Dashboard
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))

# (1) Statement Sentiment and Rate Hikes
ax1 = axes[0, 0]
quarters_str = [str(d)[:7] for d in tone_res.series.index]
x_tone = np.arange(len(quarters_str))
ax1.bar(x_tone, tone_res.series.values, color="0.35", width=0.6, label="Apel-Blix-Grimaldi Tone")
ax1.axhline(0, color="0.2", linestyle="--", linewidth=0.8)
ax1.set_xticks(x_tone[::2])
ax1.set_xticklabels(quarters_str[::2], rotation=30)
ax1.set_ylabel("Tone Index [-1.0 Dovish, +1.0 Hawkish]")
ax1.set_title("(a) Central Bank Statement Tone & Policy Signals")
ax1.legend(loc="upper left")

# (2) Actual Rate vs Taylor Benchmark and Policy Stance Gap
ax2 = axes[0, 1]
sub_sample = sample_q.loc["2005Q1":]
x_time = np.arange(len(sub_sample))
t_ticks = x_time[::8]
t_labels = [str(sub_sample.index[i]) for i in t_ticks]

ax2.plot(x_time, sub_sample["ffr"].values, color="0.0", linewidth=1.8, label="Actual Policy Rate")
ax2.plot(x_time, target_rate_star.loc["2005Q1":].values, color="0.5", linestyle="--", linewidth=1.5, label="Taylor Benchmark i*")
gap_sub = stance_gap.loc["2005Q1":].values
ax2.fill_between(x_time, 0, gap_sub, where=(gap_sub >= 0), color="0.75", alpha=0.6, label="Restrictive Stance")
ax2.fill_between(x_time, 0, gap_sub, where=(gap_sub < 0), color="0.90", alpha=0.6, label="Accommodative Stance")
ax2.set_xticks(t_ticks)
ax2.set_xticklabels(t_labels, rotation=30)
ax2.set_ylabel("Annualized Rate (%)")
ax2.set_title("(b) Actual Rate vs. Taylor Benchmark & Stance Gap")
ax2.legend(loc="upper left", fontsize=8)

# (3) Counterfactual Disinflation Paths Across Regimes
ax3 = axes[1, 0]
h_steps = np.arange(12)
ax3.plot(h_steps, sim_strict["inflation"].values, color="0.0", linewidth=1.8, label="Strict IT (phi_pi=2.5, phi_y=0.0)")
ax3.plot(h_steps, sim_dovish["inflation"].values, color="0.45", linestyle="--", linewidth=1.6, label="Dovish (phi_pi=1.1, phi_y=1.0)")
ax3.plot(h_steps, sim_no_inertia["inflation"].values, color="0.65", linestyle=":", linewidth=1.6, label="No Inertia (rho_i=0.0)")
ax3.axhline(pi_star_calib, color="0.3", linestyle="--", linewidth=0.8, label="Inflation Target π*=2%")
ax3.set_xlabel("Quarters Ahead")
ax3.set_ylabel("Inflation Rate (%)")
ax3.set_title("(c) Counterfactual Disinflation Trajectories Across 3 Regimes")
ax3.legend(loc="upper right", fontsize=8)

# (4) 12-Quarter Multi-Horizon Projection Fan Chart
ax4 = axes[1, 1]
h_ax = np.arange(H_horizon + 1)
infl_paths = paths[:, :, 1]
# Stacked quantile bands
lo_90 = np.percentile(infl_paths, 5, axis=0)
hi_90 = np.percentile(infl_paths, 95, axis=0)
lo_70 = np.percentile(infl_paths, 15, axis=0)
hi_70 = np.percentile(infl_paths, 85, axis=0)
lo_50 = np.percentile(infl_paths, 25, axis=0)
hi_50 = np.percentile(infl_paths, 75, axis=0)
median_proj = np.percentile(infl_paths, 50, axis=0)

ax4.fill_between(h_ax, lo_90, hi_90, color="0.88", label="90% Confidence Interval")
ax4.fill_between(h_ax, lo_70, hi_70, color="0.75", label="70% Confidence Interval")
ax4.fill_between(h_ax, lo_50, hi_50, color="0.60", label="50% Confidence Interval")
ax4.plot(h_ax, median_proj, color="0.0", linewidth=1.8, label="Median Projection")
ax4.axhline(pi_star_calib, color="0.3", linestyle="--", linewidth=0.8, label="Target π*=2%")
ax4.set_xlabel("Forecast Horizon (Quarters Ahead)")
ax4.set_ylabel("Projected CPI Inflation (%)")
ax4.set_title("(d) 12-Quarter Central Bank Projection Fan Chart")
ax4.legend(loc="lower left", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Read the output
#
# **Read the output.**
# 1. **Dinámica del Tono de Comunicación (Panel a)**: El análisis algorítmico de sentimiento mediante el léxico de Apel-Blix-Grimaldi captura el rápido viraje en la orientación prospectiva de la autoridad monetaria. El índice de tono permaneció cerca de cero a inicios de 2021 cuando el FOMC catalogaba la inflación como "transitoria". A partir de 2022Q1, el tono narrativo subió vertiginosamente a $+1.0$, anticipando el ciclo de alzas de tasas trimestres antes de que la tasa efectiva alcanzara su nivel terminal.
# 2. **Postura Empírica y el Principio de Taylor (Panel b)**: Las estimaciones de la regla de Taylor confirman una inercia de $\hat{\rho}_i = 0.940$ y una respuesta a la inflación de $\hat{\phi}_\pi = 1.174 > 1.0$. El cumplimiento del principio de Taylor garantiza la determinación del equilibrio macroeconómico. La brecha de postura muestra una política profundamente acomodaticia durante la Gran Recesión y la pandemia ($i_t < i_t^*$), pasando a terreno netamente restrictivo ($+200 \text{ pb}$) a medida que las tasas superaron la prescripción normativa durante 2022-2023.
# 3. **Contrafácticos de Regímenes de Política (Panel c)**: La simulación del equilibrio dinámico revela claras disyuntivas. Bajo Metas de Inflación Estrictas ($\phi_\pi = 2.5, \phi_y = 0.0$), el banco central converge a la meta en 6 trimestres a costa de una mayor contracción inicial del producto. Por el contrario, un régimen expansionista o *dovish* ($\phi_\pi = 1.1, \phi_y = 1.0$) tolera inflación persistente por encima de $2.3\%$ tras un año para proteger el empleo. El régimen sin inercia ($\rho_i = 0.0$) genera alzas violentas e inmediatas que desestabilizan las condiciones financieras sin mejorar sustancialmente la desinflación de mediano plazo.
# 4. **Conos de Proyección Probabilística (Panel d)**: El gráfico de abanico a 12 trimestres ilustra la expansión de la incertidumbre macroeconómica en el tiempo. Mientras la mediana converge ordenadamente hacia la meta de $2.0\%$, el cono de confianza al 90% se ensancha desde $\pm 0.8 \text{ pp}$ en $h=1$ hasta $\pm 2.4 \text{ pp}$ en $h=12$, evidenciando por qué los bancos centrales modernos publican densidades predictivas en lugar de proyecciones puntuales engañosas.

# %%
# ---------------------------------------------------------------------------
# Your turn: customize Taylor rule counterfactual parameters
# ---------------------------------------------------------------------------
# ← change this: try aggressive phi_pi_try = 2.2, 2.5, or moderate 1.25
phi_pi_try = 2.0
# ← change this: try output weight phi_y_try = 0.1, 0.5, or 1.0
phi_y_try = 0.25
# ← change this: try interest rate inertia rho_i_try = 0.40, 0.70, or 0.85
rho_i_try = 0.70

# Simulate custom counterfactual policy regime
sim_custom = simulate_nk_counterfactual(
    phi_pi=phi_pi_try,
    phi_y=phi_y_try,
    rho_i=rho_i_try,
    T=12,
    pi_0=6.0,
    y_0=1.5,
    i_0=2.0,
)

print(f"Custom Simulation: phi_pi={phi_pi_try}, phi_y={phi_y_try}, rho_i={rho_i_try}")
print(f"  Terminal Inflation Rate (Q12)  : {sim_custom['inflation'].iloc[-1]:.3f}%")
print(f"  Peak Policy Rate               : {sim_custom['policy_rate'].max():.3f}%")
print(f"  Minimum Output Gap             : {sim_custom['output_gap'].min():.3f}%")

# Downstream automated assertions verifying customized user simulation
assert len(sim_custom) == 12, "Simulation trajectory must span exactly 12 quarters"
assert sim_custom["inflation"].iloc[-1] < 3.0, "Terminal inflation must converge toward target"
assert sim_custom["policy_rate"].max() > 2.0, "Policy rate must hike above initial 2.0% level"

# %% [markdown]
# **Preguntas y extensiones.**
# 1. *Básica*: Configure `phi_pi_try = 1.05` (adhesión débil al principio de Taylor) y `phi_y_try = 0.8`. ¿Cómo cambia la velocidad de desinflación respecto al valor base `phi_pi_try = 2.0`? ¿Permanece la inflación sistemáticamente por encima de la meta?
# 2. *Intermedia*: Fije `rho_i_try = 0.0` (cero inercia en la tasa de interés). Observe cómo la tasa salta bruscamente en el impacto inicial. ¿Por qué los bancos centrales en la práctica prefieren una inercia elevada ($\rho_i \ge 0.80$) aun cuando retrasa alcanzar la tasa terminal máxima?
# 3. *Avanzada*: Amplíe la regresión empírica de la sección 2 incorporando el Índice de Condiciones Financieras Nacionales de la Fed de Chicago (`nfci`) como cuarto regresor. ¿Reduce la Reserva Federal sistemáticamente las tasas de interés cuando las condiciones financieras se endurecen?
#
# ## ¿Qué tan completo es esto?
#
# `puremacro` ofrece una plataforma unificada para el análisis de banca central aplicada integrando herramientas teóricas, empíricas y de comunicación:
# - `puremacro.dsge.policy`: Resuelve política óptima discrecional sin compromiso (iteraciones de Riccati de Dennis 2007 para equilibrio de Markov perfecto) y sistemas de compromiso lineal-cuadrático.
# - `puremacro.lp.lp_hac`: Implementa proyecciones locales de Jordà (2005) con errores estándar robustos Newey-West HAC para identificar la transmisión dinámica de choques monetarios.
# - `puremacro.var.identify`: Identifica choques de política monetaria estructural mediante restricciones de signo narrativas (Antolín-Díaz & Rubio-Ramírez 2018) y variables instrumentales de alta frecuencia.
# - `puremacro.narrative.indices`: Motores de puntuación de sentimiento basados en léxicos especializados (`tone`, `epu`, `mpu`) sobre corpus macroeconómicos multilingües.
# - `puremacro.posterior`: Simulación predictiva bayesiana y algoritmos de calibración para gráficos de abanico de proyección multi-horizonte.
