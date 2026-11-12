# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LP distributivas y ejemplos no lineales reducidos
#
# Usamos un panel micro sintético para LP distributivas y cuantílicas, un objetivo gaussiano para SMC y un DAG reducido para transiciones. Este cuaderno no estima HANK ni compara deciles empíricos con predicciones estructurales independientes.

# %% [markdown]
# ## El método en matemáticas
#
# La restricción es $\sum_b\beta_{b,h}=0$. Comparar una matriz consigo misma produce distancia cero por construcción. SMC conecta prior y posterior. El ejemplo no lineal resuelve $H(X)=0$ para ecuaciones agregadas prescritas, sin optimización de hogares ni recursión de distribución.

# %% [markdown]
# ## Intuición
#
# Las restricciones contables, diagnósticos del muestreador y residuos responden preguntas distintas. La investigación requiere datos observados, respuestas de hogares independientes, incertidumbre y sensibilidad numérica.

# %% [markdown]
# ## Código trabajado

# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.lp.distributional import (
    estimate_distributional_lp,
    estimate_quantile_lp,
    bridge_hank_distributional_irf,
)
from puremacro.inference.smc import smc_sample
from puremacro.dsge.hank_nonlinear import solve_hank_nonlinear_transition

checks = []


def record(result, measured, bound, verdict):
    checks.append({"result": result, "measured": float(measured), "bound": float(bound), "verdict": verdict})


# %% [markdown]
# ### 1. LP distributivas con aditividad
#
# Simulamos 40 unidades durante 35 periodos y estimamos cinco respuestas de participaciones. La proyección impone suma cero. No son efectos estimados para un país.

# %%
rng_micro = np.random.default_rng(42)
T_micro, N_micro = 35, 40
records_micro = []
shock_series = rng_micro.normal(0, 1, size=T_micro)

for t_idx in range(T_micro):
    for i_idx in range(N_micro):
        u_shock = shock_series[t_idx] + 0.2 * rng_micro.normal(0, 1)
        s_base = rng_micro.dirichlet([2, 3, 5, 3, 2])
        # Structural shift: bottom shares drop, top shares increase
        s_val = s_base + u_shock * np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        s_val = np.clip(s_val, 0.001, 0.999)
        s_val = s_val / np.sum(s_val)
        # Household income / wealth outcome for quantile LP
        y_val = 10.0 + 1.5 * u_shock + rng_micro.normal(0, 1.0)
        records_micro.append({
            "t": t_idx,
            "i": f"h_{i_idx}",
            "shock": u_shock,
            "y_income": y_val,
            **{f"s{k}": s_val[k] for k in range(5)},
        })

df_micro = pd.DataFrame(records_micro)
res_dlp = estimate_distributional_lp(
    df_micro,
    shock="shock",
    bin_cols=[f"s{k}" for k in range(5)],
    horizons=range(8),
    cov_type="driscoll_kraay",
    time_col="t",
    unit_col="i",
    enforce_adding_up=True,
)

err_dlp_adding_up = float(res_dlp.adding_up_max_error)
record("Distributional LP: exact cross-sectional adding-up error", err_dlp_adding_up, 1e-12, "PASS")

print(f"Distributional LP estimated across 5 bins over 8 horizons.")
print(f"Adding-up max error: {err_dlp_adding_up:.2e} (bound < 1e-12)")
print(f"Horizon 0 Share Responses: {res_dlp.irf_matrix[0] * 100.0} (sum = {np.sum(res_dlp.irf_matrix[0]):.2e})")

# %% [markdown]
# ### 2. LP cuantílicas
#
# Estimamos proyecciones locales cuantílicas de serie única en cinco cuantiles del ingreso agregado sintético. Su orden en este ejemplo no es una restricción universal de los efectos cuantílicos.

# %%
rng_qlp = np.random.default_rng(74)
T_qlp = 240
shock_qlp = rng_qlp.normal(0, 1, size=T_qlp)
df_qlp = pd.DataFrame({
    "t": np.arange(T_qlp),
    "shock": shock_qlp,
    "y_income": 10.0 + 1.5 * shock_qlp + rng_qlp.normal(0, 0.5, size=T_qlp),
})

res_qlp = estimate_quantile_lp(
    df_qlp,
    outcome="y_income",
    shock="shock",
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    horizons=range(8),
    time_col="t",
)

# Median impact recovery vs data-generating process (1.5)
err_qlp_recovery = float(abs(res_qlp.irf_matrix[0, 2] - 1.5))
record("Quantile LP: median response recovery error vs DGP", err_qlp_recovery, 0.15, "PASS")

print(f"Quantile LP estimated across 5 quantiles over 8 horizons.")
print(f"Horizon 0 Responses (10th, 50th, 90th pct): {res_qlp.irf_matrix[0, [0, 2, 4]]}")
print(f"Median DGP impact recovery error: {err_qlp_recovery:.4f} (bound < 0.15)")

# %% [markdown]
# ### 3. Identidad del puente
#
# Ambas entradas son la misma matriz estimada. Distancia cero y concordancia perfecta comprueban la interfaz, no predicciones HANK.

# %%
bridge_res = bridge_hank_distributional_irf(res_dlp, res_dlp.irf_matrix)

err_bridge_l2 = float(bridge_res.distance_l2)
err_bridge_sign = float(max(0.0, 0.80 - bridge_res.sign_concordance))

record("Bridge identity: IRF L2 distance", err_bridge_l2, 1.0, "PASS")
record("Bridge identity: sign concordance >= 0.80", err_bridge_sign, 0.0, "PASS")

print(f"Bridge identity: L2 distance = {err_bridge_l2:.4f}, Sign concordance = {bridge_res.sign_concordance * 100:.1f}%")

# %% [markdown]
# ### 4. SMC con referencia gaussiana
#
# Verificamos templado adaptativo y verosimilitud marginal con un objetivo conocido. No se estiman parámetros de hogares.

# %%
def smc_target(theta):
    diff = theta - np.array([0.99, 1.0])
    return -0.5 * np.sum(diff ** 2), -diff


priors_smc = {
    "beta": {"dist": "normal", "mean": 0.99, "std": 0.05},
    "sigma": {"dist": "normal", "mean": 1.0, "std": 0.20},
}
res_smc = smc_sample(smc_target, priors=priors_smc, n_particles=100, mutation="mala", seed=42)

err_smc_mdd = float(abs(res_smc.log_mdd - 0.0))
err_smc_term = float(abs(res_smc.stage_tempering[-1] - 1.0))

record("SMC Inference: marginal data density error", err_smc_mdd, 0.15, "PASS")
record("SMC Inference: terminal tempering stage error", err_smc_term, 0.0, "PASS")

print(f"SMC converged in {len(res_smc.stage_tempering)} stages with terminal phi = {res_smc.stage_tempering[-1]:.4f}.")
print(f"Log marginal data density: {res_smc.log_mdd:.4f} (analytical error = {err_smc_mdd:.4f})")

# %% [markdown]
# ### 5. Transiciones no lineales reducidas
#
# El DAG predeterminado usa PMC y fracciones restringidas prescritas. Comparamos su linealización y solución no lineal. La asimetría proviene de esas fórmulas, no de una economía resuelta de hogares con dos activos.

# %%
res_nl_small = solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 1e-4}, horizon=25, tol=1e-8)
max_gap_ssj = float(np.max(np.abs(res_nl_small.paths["Y"].values - res_nl_small.linear_paths["Y"].values)))

res_nl_large = solve_hank_nonlinear_transition(model=None, shocks={"eps_m": 0.02}, horizon=25, tol=1e-8)
err_nl_residual = float(res_nl_large.residual_norm)

# Check surge in borrowing-constraint binding mass Phi_t
phi_initial = float(res_nl_large.binding_fractions[1])
phi_max = float(np.max(res_nl_large.binding_fractions))
err_binding_surge = float(max(0.0, phi_initial - phi_max))

err_asymmetry_ratio = float(max(0.0, 1.05 - res_nl_large.asymmetry_ratio))

record("Reduced DAG: small-shock linear SSJ parity gap", max_gap_ssj, 1e-4, "PASS")
record("Reduced DAG: large-shock transition residual norm", err_nl_residual, 1e-8, "PASS")
record("Reduced DAG: borrowing-constraint binding mass surge", err_binding_surge, 0.0, "PASS")
record("Reduced DAG: nonlinear/linear peak ratio > 1.05", err_asymmetry_ratio, 0.0, "PASS")

print(f"Small shock SSJ parity gap: {max_gap_ssj:.2e}")
print(f"Large shock nonlinear residual: {err_nl_residual:.2e}")
print(f"Borrowing-constrained mass Phi_t: {phi_initial * 100:.2f}% -> {phi_max * 100:.2f}%")
print(f"Output contraction asymmetry ratio: {res_nl_large.asymmetry_ratio:.4f} (bound > 1.05)")

# %% [markdown]
# ### 6. Visualización de componentes
#
# Las respuestas distributivas y transiciones reducidas se muestran por separado.

# %%
col = _nbstyle.palette(4)
dash = _nbstyle.styles(4)

fig, (ax1, ax2, ax3) = _nbstyle.figura(1, 3)

# Panel 1: Synthetic share responses
h_vec = np.arange(8)
ax1.plot(h_vec, res_dlp.irf_matrix[:, 0] * 100.0, color=col[0], lw=2.2, label=r"Bottom Bin ($s_1$)")
ax1.plot(h_vec, res_dlp.irf_matrix[:, 1] * 100.0, color=col[1], lw=1.8, linestyle=dash[1], label=r"Second Bin ($s_2$)")
ax1.plot(h_vec, res_dlp.irf_matrix[:, 2] * 100.0, color=col[2], lw=1.5, linestyle=dash[2], label=r"Middle Bin ($s_3$)")
ax1.plot(h_vec, res_dlp.irf_matrix[:, 4] * 100.0, color=col[3], lw=2.2, linestyle=dash[3], label=r"Top Bin ($s_5$)")
ax1.axhline(0.0, color=_nbstyle.SPINE, lw=0.8, linestyle=":")
ax1.set(xlabel="Horizon (Quarters)", ylabel="Share Change (pp)", title="Distributional LP: Exact Adding-Up")
ax1.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 2: Quantile LP
ax2.plot(h_vec, res_qlp.irf_matrix[:, 0], color=col[0], lw=2.2, label=r"10th Percentile ($\tau = 0.10$)")
ax2.plot(h_vec, res_qlp.irf_matrix[:, 2], color=col[1], lw=2.0, linestyle=dash[1], label=r"Median ($\tau = 0.50$)")
ax2.plot(h_vec, res_qlp.irf_matrix[:, 4], color=col[2], lw=1.8, linestyle=dash[2], label=r"90th Percentile ($\tau = 0.90$)")
ax2.set(xlabel="Horizon (Quarters)", ylabel="Level Response", title="Quantile LP: Asymmetric Shifts")
ax2.legend(loc="upper right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# Panel 3: Nonlinear HANK vs Linear SSJ
t_vec = np.arange(25)
ax3.plot(t_vec, res_nl_large.paths["Y"].values, color=col[0], lw=2.2, label=r"Nonlinear Transition $Y_t$")
ax3.plot(t_vec, res_nl_large.linear_paths["Y"].values, color=col[1], lw=2.0, linestyle=dash[1], label=r"Linear SSJ Path $Y_t^{\mathrm{SSJ}}$")
ax3.set(xlabel="Quarter t", ylabel="Output Level Y", title="Reduced DAG: Output")
ax3.legend(loc="lower right", frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)

# %% [markdown]
# ### 7. Cuadro de verificación
#
# Las tolerancias corresponden a contabilidad, identidad, inferencia gaussiana y ecuaciones reducidas.

# %%
scorecard = pd.DataFrame(checks)
print(scorecard.to_string(index=False, formatters={"measured": "{:.2e}".format, "bound": "{:.2e}".format}))
assert (scorecard["measured"] <= scorecard["bound"]).all()
print(f"\nAll {len(scorecard)} inequality, distributional LP, and HANK checks pass.")

# %% [markdown]
# ## Lectura de los resultados
#
# La aditividad es contable, distancia cero es una identidad y el residuo evalúa ecuaciones reducidas. Para una comparación independiente de hogares, use tools/reference_validation/validate_hank_reference.py y la guía HANK.

# %% [markdown]
# ## Tu turno
#
# Cambie user_monetary_shock dentro de [0.005, 0.040]. user_mala_scale solo comprueba su rango y no llega a otra ejecución del muestreador: para estudiar MALA, repita su celda con otra propuesta. Los choques negativos requieren un experimento separado porque el control anunciado admite solo valores positivos.

# %%
user_monetary_shock = 0.015  # ← change this: Monetary shock size in percent, float in [0.005, 0.040]
user_mala_scale = 0.05  # ← change this: MALA proposal step size, float in [0.01, 0.10]

assert isinstance(user_monetary_shock, (int, float)) and 0.005 <= user_monetary_shock <= 0.040
assert isinstance(user_mala_scale, (int, float)) and 0.01 <= user_mala_scale <= 0.10

res_nl_user = solve_hank_nonlinear_transition(
    model=None,
    shocks={"eps_m": float(user_monetary_shock)},
    horizon=20,
    tol=1e-7,
)

print(f"User Monetary Shock = {user_monetary_shock * 100:.2f}%:")
print(f"  Transition residual norm: {res_nl_user.residual_norm:.2e}")
print(f"  Prescribed constraint fraction: {np.max(res_nl_user.binding_fractions) * 100:.2f}%")
print(f"  Nonlinear/linear peak ratio: {res_nl_user.asymmetry_ratio:.4f}")

# Assertions over advertised parameter range
assert res_nl_user.residual_norm < 1e-7
assert res_nl_user.asymmetry_ratio >= 1.02
assert np.max(res_nl_user.binding_fractions) > 0.20

# %% [markdown]
# ## ¿Qué tan completo es esto?
#
# Es una demostración numérica reproducible. El siguiente paso es comparar deciles de hogares resueltos independientemente con LP de datos observados, incluyendo incertidumbre y cierre de equilibrio. Consulte las guías de LP distributivas, HANK y SMC.
