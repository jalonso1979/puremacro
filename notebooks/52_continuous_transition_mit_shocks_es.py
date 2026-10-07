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
# # Dinámica de Transición Continua y Choques MIT: Espacio de Secuencias Distribucionales No Lineales
#
# **¿Cómo se mueven la distribución de la riqueza $\mu_t(k, z)$ y los precios de los factores $\{r_t, w_t\}$ de una economía de mercados incompletos después de un choque de productividad imprevisto y transitorio, y cuánto dura la respuesta más allá del propio choque?**
#
# En una economía de Bewley-Huggett-Aiyagari la distribución de la riqueza es una variable de estado. Cuando llega un choque agregado imprevisto (un "choque MIT": un evento de probabilidad cero que, una vez ocurrido, todos prevén con perfecta certeza), la distribución $\mu_0(k, z)$ está predeterminada en el impacto, de modo que el capital agregado no puede saltar. Los mercados de factores se vacían a través de los precios: el salario real y la tasa de interés real se mueven en el impacto, y el capital solo se ajusta a medida que los hogares ahorran.
#
# Este cuaderno calcula la trayectoria de transición no lineal de toda la distribución. Un paso hacia atrás con el Método de Malla Endógena (EGM) da las políticas de los hogares a lo largo de una trayectoria de precios, un paso hacia adelante con la lotería de Young (2010) mueve la distribución, $\mu_{t+1} = \mathcal{T}_t^* \mu_t$, y una iteración cuasi-Newton de Broyden sobre toda la trayectoria de la tasa de interés vacía el mercado de capital en cada fecha. La calibración es anual ($\beta = 0.96$, $\delta = 0.08$), así que un periodo es un año.

# %% [markdown]
# ## El método en matemáticas — Dinámica de Transición Continua y Choques MIT
#
# **1. Método de Malla Endógena (EGM) hacia atrás.** Dada una trayectoria de precios de los factores $\{r_t, w_t\}_{t=0}^{T-1}$ y la política de consumo del estado estacionario terminal en la fecha $T$, las políticas de los hogares se calculan hacia atrás desde $t = T-1$ hasta $t = 0$. Con utilidad CRRA $u(c) = \frac{c^{1-\gamma} - 1}{1-\gamma}$, un hogar con productividad $z_i$ que elige un ahorro $a'$ satisface la ecuación de Euler
# $$ u'\left(c_t^{\text{endo}}(a', z_i)\right) = \beta (1 + r_{t+1}) \sum_{j=1}^{n_z} P_z(z_i, z_j) \left[c_{t+1}(a', z_j)\right]^{-\gamma}, $$
# de modo que el consumo endógeno se obtiene invirtiendo la utilidad marginal, $c_t^{\text{endo}}(a', z_i) = \left(\beta (1 + r_{t+1}) \sum_{j} P_z(z_i, z_j) \left[c_{t+1}(a', z_j)\right]^{-\gamma}\right)^{-1/\gamma}$. De la restricción presupuestaria $(1 + r_t) a_t + w_t z_i = c_t + a'$, los activos al inicio del periodo que llevan a elegir $a'$ son:
# $$ a_t^{\text{endo}}(a', z_i) = \frac{c_t^{\text{endo}}(a', z_i) + a' - w_t z_i}{1 + r_t}. $$
# Interpolar linealmente los pares $(a_t^{\text{endo}}, a')$ sobre la malla fija de activos $\mathcal{K} = \{k_1, \dots, k_{N_k}\}$ e imponer la restricción de endeudamiento $a' \ge 0$ da las políticas $a'_t(k, z_i) = \max\left\{0, \text{interp}\left(k; a_t^{\text{endo}}(\cdot, z_i), a'\right)\right\}$ y $c_t(k, z_i) = (1 + r_t) k + w_t z_i - a'_t(k, z_i)$.
#
# **2. Evolución de la densidad hacia adelante (lotería de Young 2010).** A partir de la distribución estacionaria $\mu_0(k, z)$, la densidad avanza con la secuencia de políticas $\{a'_t\}_{t=0}^{T-1}$. Una elección de ahorro $a' = a'_t(k_i, z_j)$ en el intervalo de la malla $[k_m, k_{m+1}]$ se reparte entre los dos nodos de modo que se preserva el valor esperado de $a'$:
# $$ w_{\text{lo}}(a') = \frac{k_{m+1} - a'}{k_{m+1} - k_m}, \qquad w_{\text{hi}}(a') = 1 - w_{\text{lo}}(a'). $$
# El operador hacia adelante $\mathcal{T}_t^*$ reparte la masa sobre la malla de activos y después aplica la transición de productividad:
# $$ \mu_{t+1}(k_m, z_l) = \sum_{j=1}^{n_z} P_z(z_j, z_l) \sum_{i=1}^{N_k} \mu_t(k_i, z_j) \left[ w_{\text{lo}}\left(a'_t(k_i, z_j)\right) \mathbf{1}_{\{m = j_{\text{lo}}\}} + w_{\text{hi}}\left(a'_t(k_i, z_j)\right) \mathbf{1}_{\{m = j_{\text{hi}}\}} \right]. $$
# Los pesos suman uno, así que el operador conserva la masa total hasta el error de redondeo; la corrida de abajo imprime la mayor desviación de $\sum_{k, z} \mu_t(k, z)$ respecto de uno.
#
# **3. Vaciado de mercado en el espacio de secuencias.** La oferta agregada de capital es $K_t^s(\mathbf{r}) = \sum_{i=1}^{N_k} \sum_{j=1}^{n_z} k_i \, \mu_t(k_i, z_j)$. Una empresa competitiva con tecnología $Y_t = Z_t (K_t^d)^\alpha L^{1-\alpha}$ y depreciación $\delta$ fija $r_t = \alpha Z_t (K_t^d / L)^{\alpha - 1} - \delta$ y $w_t = (1 - \alpha) Z_t (K_t^d / L)^\alpha$, así que la demanda de capital es
# $$ K_t^d(r_t; Z_t) = L \left( \frac{r_t + \delta}{\alpha Z_t} \right)^{\frac{1}{\alpha - 1}}. $$
# El equilibrio exige vaciar el mercado en cada fecha $t = 0, \dots, T-1$:
# $$ H_t(\mathbf{r}) \equiv K_t^s(\mathbf{r}) - K_t^d(r_t; Z_t) = 0, \qquad \mathbf{H}(\mathbf{r}) = \mathbf{0} \in \mathbb{R}^T. $$
# La condición terminal en la fecha $T$ es el estado estacionario inicial para todo choque transitorio ($\rho < 1$), y el choque se anula a partir de $T$. `continuous_mit_shock` registra el choque que queda en $T-1$, $s\rho^{T-1}$, en `metadata["mit_shock"]` y emite un aviso cuando su fracción del impacto, $\rho^{T-1}$, supera `truncation_tol` $= 10^{-3}$; con $T = 40$ eso ocurre para cualquier $\rho$ mayor que $0.838$ (impreso en el Experimento 2). Un choque permanente hay que pedirlo con `persistence=1.0`, que resuelve un estado estacionario terminal en $Z = 1 + s$. Hasta la versión 4.3.0 un choque transitorio cuyo $Z_{T-1}$ difería de uno en más de unos $10^{-5}$ se resolvía en silencio como permanente en $Z_{T-1}$; con $T = 40$ y $s = 0.05$ eso ocurría para todo $\rho$ mayor que $0.806$.
#
# **4. Cuasi-Newton de Broyden.** El sistema $\mathbf{H}(\mathbf{r}) = \mathbf{0}$ se resuelve con el método de Broyden, que actualiza una inversa aproximada del jacobiano mediante correcciones de rango uno (Sherman-Morrison):
# $$ B_{k+1} = B_k + \frac{(\Delta \mathbf{r}_k - B_k \Delta \mathbf{H}_k)(\Delta \mathbf{r}_k^\top B_k)}{\Delta \mathbf{r}_k^\top B_k \Delta \mathbf{H}_k}, \qquad \mathbf{r}_{k+1} = \mathbf{r}_k - \theta \, B_k \mathbf{H}(\mathbf{r}_k), $$
# donde $\theta \in (0, 1]$ es el paso que acepta una búsqueda lineal con retroceso y $B_0 = \text{diag}\left( \frac{1 - \alpha}{K_t^d} (r_t + \delta) \right)$ invierte la diagonal de la derivada de la demanda de la empresa. La palabra clave `damping` no interviene en esta iteración. Broyden solo la usa en un paso de reserva, que se da cuando falla la búsqueda lineal, y eso no ocurre nunca en este cuaderno. Es el peso de relajación $\omega$ del solucionador de shooting del Experimento 4, $\mathbf{r}_{k+1} = (1 - \omega)\,\mathbf{r}_k + \omega\,\mathbf{r}_k^{\text{implied}}$.
#
# **5. Una forma cerrada para la respuesta en el impacto.** Como $K_0 = K^*$, las condiciones de la empresa dan directamente los precios de la fecha 0. Con $Z_0 = 1 + s$,
# $$ r_0 - r^* = s\,(r^* + \delta), \qquad \frac{w_0}{w^*} - 1 = s. $$
# Esta relación se deriva a mano de las condiciones de primer orden de la empresa y no usa el solucionador, así que es una verificación independiente de los precios de la fecha 0 que este devuelve.

# %% [markdown]
# ## Intuición
#
# **Intuición.** En un modelo de agente representativo, un choque de productividad mueve el capital a lo largo de una trayectoria de punto de silla. Con riesgo de ingreso no asegurable, la distribución de la riqueza añade inercia: el capital es la suma de los ahorros de muchos hogares, y esos ahorros tardan en acumularse.
#
# Cuando un choque de PTF de $+5\%$ llega por sorpresa, los productos marginales del capital y del trabajo suben de inmediato. El capital está predeterminado en $t = 0$, así que la tasa de interés y el salario absorben todo el impacto, exactamente como dice la forma cerrada de arriba. El mayor rendimiento y el mayor salario elevan el ahorro. Los hogares cerca de la restricción de endeudamiento, cuyo ingreso es sobre todo laboral, pueden reconstruir sus colchones de ahorro, y los hogares más ricos ganan más con su capital.
#
# Por eso el capital sube durante varios años aunque el choque se desvanezca. A medida que el capital se acumula, su producto marginal cae, y la tasa de interés baja de su valor de estado estacionario mientras el capital sigue por encima de $K^*$. El capital sigue creciendo mientras el ahorro neto sea positivo, así que su máximo llega después de que la tasa de interés ya dio la vuelta. Luego la distribución de la riqueza regresa hacia la estacionaria.
#
# Encontrar el equilibrio significa encontrar toda la trayectoria $\mathbf{r} = \{r_t\}_{t=0}^{T-1}$ en la que el ahorro de los hogares iguala la demanda de las empresas en cada fecha. El método de Broyden trata esa trayectoria como un solo vector y actualiza un jacobiano aproximado, lo que típicamente converge de forma superlineal. La iteración de punto fijo amortiguada ("shooting") sobre el mismo sistema también converge aquí, solo que en más iteraciones.

# %%
# Preamble: import numerical libraries, plotting style, and continuous solvers
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.vfi import (
    solve_aiyagari_continuous,
    continuous_mit_shock,
    ContinuousStationaryDistribution,
)

# Nothing below is random; the seed is set only by convention.
rng = np.random.default_rng(42)

# Calibration (annual: one period is one year)
beta = 0.96      # household discount factor
gamma = 2.0      # coefficient of relative risk aversion (CRRA)
alpha = 0.36     # capital share (Cobb-Douglas)
delta = 0.08     # depreciation rate per year
rho_z = 0.90     # persistence of idiosyncratic labour productivity
sigma_z = 0.20   # standard deviation of productivity innovations
N_k = 150        # asset grid nodes
n_z = 5          # productivity states

print(f"Calibration: beta={beta}, gamma={gamma}, alpha={alpha}, delta={delta}, "
      f"rho_z={rho_z}, sigma_z={sigma_z}, N_k={N_k}, n_z={n_z}")

# %%
# --- Experiment 1: Baseline stationary incomplete-markets equilibrium ---
# The solver's controls are written out explicitly: the EGM stops when the consumption
# policy moves by less than egm_tol (or after egm_max_iter steps, which it reports),
# Brent's method on r stops at xtol, and the equilibrium counts as converged only if
# |K^s - K^d| < tol_ge at r* and the stationary distribution converged too.
ss_base = solve_aiyagari_continuous(
    beta=beta,
    gamma=gamma,
    alpha=alpha,
    delta=delta,
    rho_z=rho_z,
    sigma_z=sigma_z,
    N_k=N_k,
    n_z=n_z,
    max_evals=60,
    egm_tol=1e-8,
    egm_max_iter=10_000,
    xtol=1e-8,
    tol_ge=1e-4,
)
meta = ss_base.metadata

# Baseline wealth inequality statistics
K_hist = ss_base.distribution.asset_grid
gini_base = ss_base.distribution.gini()
p_lorenz, L_base = ss_base.distribution.lorenz(100)
p50_base = ss_base.distribution.percentile(50.0)
p90_base = ss_base.distribution.percentile(90.0)
Y_base = float((ss_base.K ** alpha) * (ss_base.L ** (1.0 - alpha)))

print("Baseline stationary equilibrium:")
print(f"  Interest rate r*          = {ss_base.r:.6f} ({ss_base.r * 100:.3f}%), 1/beta - 1 = {(1 / beta - 1) * 100:.3f}%")
print(f"  Real wage w*              = {ss_base.w:.6f}")
print(f"  Capital K*                = {ss_base.K:.4f}")
print(f"  Output Y*                 = {Y_base:.4f}")
print(f"  Capital-output ratio K/Y  = {ss_base.K / Y_base:.3f}")
print(f"  Wealth Gini G*            = {gini_base:.4f}")
print(f"  Median wealth (P50)       = {p50_base:.3f}")
print(f"  90th percentile (P90)     = {p90_base:.3f}")
print("Convergence diagnostics:")
print(f"  converged                 = {ss_base.converged}")
print(f"  EGM at r*                 : converged={meta['egm_converged']}, {meta['egm_iterations']} iterations, "
      f"step {meta['egm_residual']:.1e} (egm_tol {meta['egm_tol']:.0e}); cap hits in the search: {meta['egm_cap_hits']}")
print(f"  Stationary distribution   : converged={meta['dist_converged']}, mass error {ss_base.distribution.mass_error:.1e}")
print(f"  Market clearing |K^s-K^d| = {ss_base.capital_market_clearing_error:.1e} (tol_ge {meta['tol_ge']:.0e}) "
      f"after {ss_base.n_evals} evaluations of r")

# Each flag is a real check since the 30 September fixes: through 4.3.0 `converged` was always True
# and the household EGM stopped silently at 500 iterations.
assert ss_base.converged and meta["egm_converged"] and meta["dist_converged"] and not meta["nonconvergence_reasons"]
assert ss_base.capital_market_clearing_error < meta["tol_ge"]
assert ss_base.distribution.mass_error < 1e-12
assert 0.0 < ss_base.r < 1.0 / beta - 1.0, "precautionary saving must push r* below the rate of time preference"
assert 0.40 < gini_base < 0.65, "sanity range for this calibration, not an empirical target"

# %%
# --- Experiment 2: Non-linear transition after an unexpected TFP shock ---
# A 40-year transition after a +5% TFP shock that decays at rate 0.8 per year.
# damping keeps its default: Broyden uses it only if the line search fails (math, section 4).
horizon = 40
shock_size = 0.05
persistence = 0.80

trans_res = continuous_mit_shock(
    ss_base,
    shock_type="tfp",
    shock_size=shock_size,
    persistence=persistence,
    horizon=horizon,
    solver="broyden",
    tol=1e-4,
)
mit = trans_res.metadata["mit_shock"]

# Headline transition metrics
impact_r = trans_res.r_path[0]
impact_w = trans_res.w_path[0]
impact_K_s = trans_res.K_s_path[0]
peak_K_idx = int(np.argmax(trans_res.K_s_path))
peak_K = trans_res.K_s_path[peak_K_idx]
cross_idx = int(np.argmax(trans_res.r_path < ss_base.r))  # first year with r_t below r*
r_jump_bps = (impact_r - ss_base.r) * 1e4
w_jump_pct = (impact_w / ss_base.w - 1.0) * 100.0

# Independent check: the closed form of section 5 of the math
pred_r_jump_bps = shock_size * (ss_base.r + delta) * 1e4
pred_w_jump_pct = shock_size * 100.0

print(f"Transition: {horizon} years, TFP shock {shock_size * 100:+.1f}% with persistence {persistence}")
print(f"  Broyden iterations                 = {trans_res.iterations}; converged = {trans_res.converged} "
      f"(relaxation {trans_res.metadata['relaxation_converged']})")
print(f"  Max market-clearing residual       = {trans_res.max_residual:.2e} (tol 1e-04)")
print(f"  Max mass conservation error        = {trans_res.mass_conservation_error:.1e}")
print(f"  Terminal condition                 = {mit['terminal_condition']}")
print(f"  Shock left at T-1                  = {mit['shock_at_last_date']:.1e} ({mit['remaining_share'] * 100:.3f}% of the impact; "
      f"truncation_tol {mit['truncation_tol'] * 100:.1f}%, truncated = {mit['truncated']})")
print(f"  Warns for persistence above        = {mit['truncation_tol'] ** (1 / (horizon - 1)):.3f} (rho^(T-1) > truncation_tol at T = {horizon})")
print(f"  |K_0^s - K*|                     = {abs(impact_K_s - ss_base.K):.1e}")
print(f"  Impact interest rate r_0           = {impact_r * 100:.3f}% (jump {r_jump_bps:+.2f} bps; closed form {pred_r_jump_bps:+.2f} bps)")
print(f"  Impact real wage w_0               = {impact_w:.4f} (jump {w_jump_pct:+.4f}%; closed form {pred_w_jump_pct:+.4f}%)")
print(f"  First year with r_t < r*           = {cross_idx}")
print(f"  Peak capital K_peak                = {peak_K:.4f} in year {peak_K_idx} (TFP shock left: {shock_size * persistence ** peak_K_idx * 100:.2f}%)")
print(f"  Capital in the last year, t = {horizon - 1}   = {trans_res.K_s_path[-1]:.4f} (K* = {ss_base.K:.4f})")

assert trans_res.converged and trans_res.max_residual < 1e-4
assert mit["terminal_condition"] == "initial_steady_state" and not mit["truncated"]
assert trans_res.mass_conservation_error < 1e-12
assert abs(impact_K_s - ss_base.K) < 1e-10, "capital is predetermined at t = 0"
assert abs(r_jump_bps - pred_r_jump_bps) < 0.1, "impact rate must match the closed form"
assert abs(w_jump_pct - pred_w_jump_pct) < 1e-3, "impact wage must match the closed form"
assert 0 < cross_idx < peak_K_idx, "capital keeps rising after r_t falls below r*"
assert np.isclose(trans_res.K_s_path[-1], ss_base.K, atol=0.05), "capital must be back near K* by the last year"

# %%
# --- Experiment 3: Wealth inequality along the transition ---
ginis = np.array([
    ContinuousStationaryDistribution(pdf=d, asset_grid=K_hist).gini()
    for d in trans_res.distributions
])
time_grid = np.arange(horizon + 1)
min_gini_idx = int(np.argmin(ginis))
min_gini = ginis[min_gini_idx]
# Share of households at the borrowing constraint k = 0 (first grid node)
at_constraint = np.array([d[0].sum() for d in trans_res.distributions])
# Change since year 0 in the mass of each asset grid node (summed over productivity)
density_mat = np.array([np.sum(d, axis=1) if d.ndim == 2 else d for d in trans_res.distributions])
d_mass = density_mat - density_mat[0]

dist_peak = ContinuousStationaryDistribution(pdf=trans_res.distributions[peak_K_idx], asset_grid=K_hist)
dist_term = ContinuousStationaryDistribution(pdf=trans_res.distributions[-1], asset_grid=K_hist)
_, L_peak = dist_peak.lorenz(100)
_, L_term = dist_term.lorenz(100)

print("Wealth inequality along the transition:")
print(f"  Gini in year 0        = {ginis[0]:.4f}")
print(f"  Minimum Gini (year {min_gini_idx}) = {min_gini:.4f} (change {min_gini - ginis[0]:+.4f})")
print(f"  Gini in year {horizon}       = {ginis[-1]:.4f}")
print(f"  Share at k = 0        : {at_constraint[0]:.4f} in year 0, minimum {at_constraint.min():.4f} in year {int(np.argmin(at_constraint))}")
print(f"  Change in mass by year {peak_K_idx}: largest loss {d_mass[peak_K_idx].min() * 100:+.2f} pp at k = "
      f"{K_hist[np.argmin(d_mass[peak_K_idx])]:.2f}, largest gain {d_mass[peak_K_idx].max() * 100:+.2f} pp at k = "
      f"{K_hist[np.argmax(d_mass[peak_K_idx])]:.2f}")

assert len(trans_res.distributions) == horizon + 1
assert min_gini < ginis[0], "wealth inequality must compress during the expansion"
assert np.isclose(ginis[-1], ginis[0], atol=0.01), "the Gini must return close to its initial level"

# Figure 1: factor prices, market clearing, Gini path and Lorenz curves
fig1, axes1 = _nbstyle.figura(2, 2, figsize=(11, 8))
t_years = np.arange(horizon)

ax1 = axes1[0, 0]
ax1.plot(t_years, trans_res.r_path * 100, label="Real rate $r_t$ (%)", lw=2, color=_nbstyle.S1["color"])
ax1.axhline(ss_base.r * 100, ls="--", color=_nbstyle.SPINE, label=f"Initial $r^*={ss_base.r * 100:.2f}\\%$")
ax1.set_title("Factor prices: real interest rate path")
ax1.set_xlabel("Year $t$")
ax1.set_ylabel("Percent per year (%)")
ax1.legend(loc="upper right")

ax2 = axes1[0, 1]
ax2.plot(t_years, trans_res.K_s_path, label="Capital supply $K_t^s$", lw=2, color=_nbstyle.S1["color"])
ax2.plot(t_years, trans_res.K_d_path, label="Capital demand $K_t^d$", ls="--", lw=1.8, color=_nbstyle.S2["color"])
ax2.axhline(ss_base.K, ls=":", color=_nbstyle.SPINE, label=f"Initial $K^*={ss_base.K:.2f}$")
ax2.set_title("Capital market clearing ($K_t^s$ vs $K_t^d$)")
ax2.set_xlabel("Year $t$")
ax2.set_ylabel("Aggregate capital (model units)")
ax2.legend(loc="upper right")

ax3 = axes1[1, 0]
ax3.plot(time_grid, ginis, label="Wealth Gini $G_t$", lw=2, color=_nbstyle.S1["color"])
ax3.axhline(ginis[0], ls="--", color=_nbstyle.SPINE, label=f"Initial Gini $G_0={ginis[0]:.4f}$")
ax3.scatter([min_gini_idx], [min_gini], color=_nbstyle.S1["color"], s=40, zorder=5, label=f"Minimum Gini ({min_gini:.4f})")
ax3.set_title("Wealth inequality: Gini coefficient")
ax3.set_xlabel("Year $t$")
ax3.set_ylabel("Gini coefficient")
ax3.legend(loc="upper right")

ax4 = axes1[1, 1]
ax4.plot(p_lorenz * 100, p_lorenz * 100, linestyle=":", color=_nbstyle.SPINE, label="45° equality line", alpha=0.5)
ax4.plot(p_lorenz * 100, L_base * 100, label=f"Lorenz $t=0$ ($G={ginis[0]:.3f}$)", lw=2, color=_nbstyle.S1["color"])
ax4.plot(p_lorenz * 100, L_peak * 100, label=f"Lorenz $t={peak_K_idx}$ ($G={ginis[peak_K_idx]:.3f}$)", lw=1.8, ls="--", color=_nbstyle.S2["color"])
ax4.plot(p_lorenz * 100, L_term * 100, label=f"Lorenz $t={horizon}$ ($G={ginis[-1]:.3f}$)", lw=1.5, ls="-.", color=_nbstyle.S3["color"])
ax4.set_title("Wealth Lorenz curves")
ax4.set_xlabel("Cumulative population (%)")
ax4.set_ylabel("Cumulative wealth (%)")
ax4.legend(loc="upper left")

# Figure 2: how the wealth distribution moves. The level is dominated by the mass at k = 0,
# so plot the change since year 0 in the mass of each asset grid node.
k_mask = K_hist <= 15.0
k_sub = K_hist[k_mask]

fig2, (ax_lines, ax_heat) = _nbstyle.figura(1, 2, figsize=(12.5, 4.6))
line_styles = [_nbstyle.S1, _nbstyle.S2, _nbstyle.S3, _nbstyle.S4]
for style, t_show in zip(line_styles, sorted({1, peak_K_idx, 20, horizon})):
    ax_lines.plot(k_sub, d_mass[t_show, k_mask] * 100, **style, label=f"Year {t_show}")
ax_lines.axhline(0, color=_nbstyle.SPINE, lw=0.8)
ax_lines.set_title("Change in wealth mass since year 0", fontsize=11)
ax_lines.set_xlabel("Assets $k$ (grid nodes up to 15)")
ax_lines.set_ylabel("Change in mass per grid node (pp)")
ax_lines.legend(loc="lower right")

im = ax_heat.imshow(d_mass[:, k_mask] * 100, aspect="auto", origin="lower",
                    extent=[k_sub[0], k_sub[-1], 0, horizon], cmap=_nbstyle.CMAP_SEQ)
cbar = fig2.colorbar(im, ax=ax_heat)
cbar.set_label("Change in mass per grid node (pp)")
ax_heat.set_title("Change in wealth mass since year 0, all years", fontsize=11)
ax_heat.set_xlabel("Assets $k$ (grid nodes up to 15)")
ax_heat.set_ylabel("Year $t$")

# %%
# --- Experiment 4: Broyden quasi-Newton against damped shooting ---
# Both solve the same system H(r) = 0, so their paths must agree to the tolerance.
# This is an internal consistency check: both routes run through puremacro's code.
shoot_res = continuous_mit_shock(
    ss_base,
    shock_type="tfp",
    shock_size=shock_size,
    persistence=persistence,
    horizon=horizon,
    solver="shooting",
    damping=0.5,
    tol=1e-4,
    max_iter=30,
)
gap_r_bps = np.max(np.abs(shoot_res.r_path - trans_res.r_path)) * 1e4
gap_K = np.max(np.abs(shoot_res.K_s_path - trans_res.K_s_path))

print("Algorithm comparison (same system, same tolerance):")
print(f"  Broyden quasi-Newton : {trans_res.iterations:2d} iterations, max residual {trans_res.max_residual:.2e}")
print(f"  Damped shooting      : {shoot_res.iterations:2d} iterations, max residual {shoot_res.max_residual:.2e}")
print(f"  Largest gap between the two paths: r {gap_r_bps:.1e} bps, K {gap_K:.1e}")

assert trans_res.converged and shoot_res.converged
assert gap_K < 1e-4, "two solvers of the same system must agree to the tolerance"

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.**
#
# 1. **Estado estacionario (Experimento 1).** La tasa de interés de equilibrio es $r^* = 1.976\%$ anual, muy por debajo de la tasa de preferencia temporal $1/\beta - 1 = 4.167\%$. Los hogares mantienen más capital del que tendrían con seguro completo, porque ahorran contra el riesgo de ingreso no asegurable y el límite de endeudamiento. El capital es $K^* = 8.7924$, es decir, 3.609 años de producto. La riqueza está concentrada: el Gini es $0.5269$, y el hogar mediano tiene $5.800$ frente a $22.702$ en el percentil 90. Los diagnósticos ahora informan: en $r^*$ el EGM de los hogares cumplió su tolerancia tras 308 iteraciones, la distribución estacionaria convergió y el mercado de capital se vacía hasta $8.0 \times 10^{-7}$ frente a `tol_ge` $= 10^{-4}$. Hasta la versión 4.3.0 `converged` era siempre True y el EGM se detenía en silencio tras 500 iteraciones.
# 2. **Impacto (Experimento 2).** El capital está predeterminado ($|K_0^s - K^*| = 1.8 \times 10^{-15}$), de modo que los precios absorben el choque de PTF de $+5\%$. La tasa de interés salta $+49.88$ puntos base hasta $2.475\%$, y el salario $+5.0000\%$. Ambos coinciden con la forma cerrada $s(r^* + \delta)$ y $s$, una verificación independiente derivada a mano de las condiciones de primer orden de la empresa. La celda de Tu turno la repite para cualquier choque.
# 3. **Propagación (Experimento 2).** La tasa de interés cae por debajo de $r^*$ en el año 5, pero el capital sigue subiendo hasta el año 7, cuando alcanza su máximo $K_{\text{peak}} = 9.0388$ y solo queda $1.05\%$ del choque de PTF. El capital es un acervo: crece mientras el ahorro neto sea positivo, incluso después de que el rendimiento cayó por debajo de su valor de estado estacionario. En el último año del horizonte ($t = 39$) el capital todavía es $8.8324$ frente a $K^* = 8.7924$, así que un horizonte de 40 años apenas contiene esta respuesta. La indicación 3 mide el error de truncamiento, para este choque y para uno más persistente.
# 4. **Desigualdad (Experimento 3).** El Gini baja de $0.5269$ a $0.5212$ en el año 7, un cambio de $-0.0057$, y es $0.5251$ en el año 40. La proporción de hogares en la restricción de endeudamiento baja de $0.0707$ a $0.0679$ en el año 6. La figura 2 muestra a dónde va esa masa: para el año 7 el nodo en $k = 0$ perdió $0.28$ puntos porcentuales de la población y el nodo en $k = 2.62$ ganó $0.29$ puntos, mientras que la distribución por encima de $k = 4$ apenas se mueve. El perfil dentado proviene de la lotería, que coloca la masa en los dos nodos de la malla que rodean cada elección de ahorro. Esto es congruente con que los hogares restringidos, que dependen del ingreso laboral, reconstruyan sus colchones durante el auge, pero ambos efectos son pequeños.
# 5. **Solucionadores (Experimentos 2 y 4).** Broyden necesita 5 iteraciones y el shooting amortiguado 11 para llegar a la misma tolerancia. Las dos trayectorias coinciden hasta $6.6 \times 10^{-6}$ en el capital y $3.9 \times 10^{-3}$ puntos base en la tasa de interés. Esa coincidencia es una verificación interna, pues ambos evalúan el mismo sistema $\mathbf{H}(\mathbf{r})$ de puremacro. La masa se conserva hasta $4.4 \times 10^{-16}$. El choque que queda en $T-1$ es $8.3 \times 10^{-6}$, un $0.017\%$ del impacto y menos que `truncation_tol` $= 0.1\%$, así que no hay aviso de truncamiento. La condición terminal es el estado estacionario inicial, como para todo choque transitorio.

# %% [markdown]
# ## Tu turno
#
# Prediga la respuesta en el impacto antes de ejecutar la celda. Anote $r_0 - r^*$ en puntos base y $w_0 / w^* - 1$ en porcentaje con la forma cerrada de la sección 5, usando el $r^*$ impreso y $\delta = 0.08$. La celda recalcula la transición, contrasta el solucionador con esa forma cerrada y verifica que el capital se mueva en la dirección del choque. Ambas verificaciones se cumplen para todo choque y toda persistencia dentro del rango anunciado. Con una persistencia mayor que 0.838 la celda imprime además un aviso de truncamiento, porque el choque no se ha extinguido en el año 39 (indicación 3).

# %%
# Your turn: change the shock, predict the impact, and let the solver check you
user_shock_size = 0.05    # ← change this: TFP shock on impact, -0.08 to 0.08 (not 0)
user_persistence = 0.80   # ← change this: yearly persistence of the shock, 0.5 to 0.92
assert user_shock_size != 0.0 and -0.08 <= user_shock_size <= 0.08
assert 0.5 <= user_persistence <= 0.92

user_res = continuous_mit_shock(
    ss_base,
    shock_type="tfp",
    shock_size=user_shock_size,
    persistence=user_persistence,
    horizon=40,
    solver="broyden",
    tol=1e-4,
)
user_dr_bps = (user_res.r_path[0] - ss_base.r) * 1e4
user_dw_pct = (user_res.w_path[0] / ss_base.w - 1.0) * 100.0
my_dr_bps = user_shock_size * (ss_base.r + delta) * 1e4   # the closed form; replace it with your own number
my_dw_pct = user_shock_size * 100.0
user_mit = user_res.metadata["mit_shock"]
K_gap = user_res.K_s_path[1:] - ss_base.K
term_text = f"{user_mit['terminal_condition']}; shock left at T-1 = {user_mit['remaining_share'] * 100:.3f}% of the impact"
if user_mit["truncated"]:
    term_text += f" (truncated; horizon={user_mit['horizon_needed']} would be long enough)"

print(f"Shock {user_shock_size * 100:+.1f}%, persistence {user_persistence:.2f}:")
print(f"  Impact rate jump : solver {user_dr_bps:+.3f} bps, prediction {my_dr_bps:+.3f} bps")
print(f"  Impact wage jump : solver {user_dw_pct:+.4f}%, prediction {my_dw_pct:+.4f}%")
print(f"  Capital peak/trough in year {int(np.argmax(np.abs(user_res.K_s_path - ss_base.K)))}, "
      f"largest gap K_t - K* = {K_gap[np.argmax(np.abs(K_gap))]:+.4f}")
print(f"  Terminal condition: {term_text}")

assert user_res.converged, "the transition must converge"
assert user_mit["terminal_condition"] == "initial_steady_state", "a transitory shock keeps the initial steady state"
assert abs(user_dr_bps - my_dr_bps) < 0.1, "impact rate differs from the prediction by more than 0.1 bps"
assert abs(user_dw_pct - my_dw_pct) < 1e-3, "impact wage differs from the prediction"
assert np.sign(K_gap.mean()) == np.sign(user_shock_size), "capital must move in the direction of the shock"

# %% [markdown]
# **Indicaciones.**
# 1. *Básico.* Tome $s \in \{0.02, 0.05, 0.08, -0.05\}$. Prediga el salto de la tasa para cada uno antes de ejecutar. ¿El salto es "casi lineal" en $s$ o exactamente lineal, y por qué cambiar `user_persistence` no lo altera?
# 2. *Intermedio.* Para $\rho \in \{0.6, 0.8, 0.92\}$ con $s = 0.05$, registre `t_peak = int(np.argmax(res.K_s_path))` y `t_cross = int(np.argmax(res.r_path < ss_base.r))`. Prediga primero: ¿el máximo del capital llega más tarde cuando $\rho$ sube, y deja de crecer el capital en cuanto $r_t$ cae por debajo de $r^*$? Verifíquelo con `assert t_peak[0] < t_peak[1] < t_peak[2]` y `assert all(0 < c < p for c, p in zip(t_cross, t_peak))`. Explique el orden con $r_t = \alpha Z_t (K_t/L)^{\alpha-1} - \delta$, que es igual a $r^*$ cuando $K_t / K^* = Z_t^{1/(1-\alpha)}$.
# 3. *Avanzado.* Con $\rho = 0.92$ y un horizonte de 40 años el choque no se ha extinguido en $T-1$. El solucionador sigue usando el estado estacionario inicial como condición terminal, anula el choque a partir del año 40 y emite un aviso con un horizonte que bastaría; imprima `user_res.metadata["mit_shock"]`. Resuelva la misma transición con `horizon=150` y calcule el error de truncamiento `e = np.abs(r40.K_s_path - r150.K_s_path[:40])`. ¿Dónde es mayor, cómo se compara con la brecha terminal `r40.K_s_path[-1] - ss_base.K` y qué tan pequeño es en los primeros 10 años? Repítalo con $\rho = 0.8$, donde no hay aviso, y enuncie una regla para elegir $T$ que mire el capital además del choque.
#
# ## ¿Qué tan exhaustivo es esto?
#
# - `puremacro.vfi.continuous_transition`: los solucionadores de transición de Broyden y de shooting (`solve_continuous_transition`, `continuous_mit_shock`, `TransitionShock`); `shock_type` también admite `"rate"` y `"beta"`.
# - `puremacro.vfi.continuous_distribution`: el equilibrio estacionario y la distribución de Young (2010) (`solve_aiyagari_continuous`, `continuous_stationary_distribution`, `ContinuousStationaryDistribution`); `docs/es/vfi_continuous_equilibrium.md` documenta los controles de convergencia usados arriba.
# - `puremacro.models.hank_sequence_space`: jacobianos lineales en el espacio de secuencias para economías HANK (cuaderno 31), la contraparte lineal de la transición no lineal que se resuelve aquí.
# - `puremacro.vfi.collocation` y `puremacro.vfi.fem`: solucionadores de proyección para el problema del hogar (cuaderno 51).
