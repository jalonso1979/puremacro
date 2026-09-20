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
# # Juegos arancelarios: bienestar hicksiano, represalias y comprobaciones numéricas
#
# **¿Cómo distinguimos un candidato de juego arancelario de una mejor respuesta verificada numéricamente?**
#
# Esta economía de dos países, balanceada a mano, es sintética. Usamos la misma función de gasto de consumo y la misma contabilidad auditada en cada comparación. Los países A y B son ilustrativos; no se predice ninguna política nacional real.

# %% [markdown]
# ## El método en matemáticas
#
# El objetivo es la variación equivalente de consumo $EV_i=e_i(P_0,U_i)-e_i(P_0,U_{i0})$ respecto a una única referencia fija sin aranceles. EV es cero en esa referencia, por lo que las ganancias porcentuales se dividen entre el gasto inicial en el consumo seleccionado $m_{i0}$. La ganancia por desviarse unilateralmente es $r_i=\max_t EV_i(t,\tau_{-i})-EV_i(\tau)$. Un candidato numérico de Nash debe satisfacer tanto $\max_i|BR_i(\tau_{-i})-\tau_i|\leq\epsilon_\tau$ como $\max_i r_i/m_{i0}\leq\epsilon_r$ en el intervalo declarado.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Los pasos pequeños por amortiguación pueden ocultar grandes incentivos para desviarse. Un arancel óptimo en la frontera depende del límite impuesto. El dilema del prisionero debe comprobarse en la matriz de pagos con acciones fijas; las represalias no lo implican. Un equilibrio general fallido no proporciona un pago.

# %% [markdown]
# ## Código resuelto
#
# Construimos una tabla balanceada con un sector productor por país, dos canastas de consumo (categorías 0 y 2), inversión separada (1), impuestos domésticos y saldos externos no nulos. Las remuneraciones fijas del trabajo y el capital cierran las cuentas de producción.

# %%
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()
from puremacro.trade import (
    calibrate_trade_model, solve_policy_equilibrium,
    compute_unilateral_optimal_tariff, compute_welfare_payoff_matrix,
    solve_multilateral_nash_tariffs,
)

# Columns: intermediate A/B, then C1/I/C2 for A and C1/I/C2 for B.
flows = np.array([[10., 12., 24., 8., 6., 4., 20., 16.],
                  [8., 14., 4.5, 15., 10.5, 35., 18., 15.]])
taxes = np.array([4., 6., 1.2, 1., .8, 1.8, 2., 1.2])
value_added = flows.sum(1) - flows[:, :2].sum(0) - taxes[:2]
data = np.vstack([flows, taxes, np.r_[2*value_added/3, np.zeros(6)],
                  np.r_[value_added/3, np.zeros(6)]])
calib = calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
calib = replace(calib, metadata={**calib.metadata, "is_synthetic": True,
    "source": "hand-balanced teaching table", "unit": "illustrative value units"})
players = tuple(calib.country_codes)
print({k: calib.metadata[k] for k in ("source", "is_synthetic", "unit")})
assert calib.metadata["is_synthetic"]
base = solve_policy_equilibrium(calib, sigma=2., tol=1e-9)
options = dict(metric="hicksian_ev", sigma=2., consumption_categories=(0, 2),
               base_equilibrium=base, ge_tol=1e-9)

# %% [markdown]
# ### Una búsqueda unilateral acotada
#
# Seleccionamos explícitamente `metric="hicksian_ev"`. El modelo usa abastecimiento intermedio CES con sigma 2, canastas finales Leontief y devolución fiscal de suma fija. El objetivo es bienestar de consumo condicional; se excluye la inversión. Todas las búsquedas comparten una referencia auditada. Los porcentajes dividen EV entre el gasto inicial en consumo.

# %%
tariff_ceiling = 0.30
opt = compute_unilateral_optimal_tariff(calib, country_idx=players[0],
    tariff_max=tariff_ceiling, num_grid=9, method="bounded", **options)
assert opt.equilibrium.converged
assert opt.equilibrium.max_residual <= options["ge_tol"]
assert opt.baseline_welfare == 0.
print("Optimal tariff on the declared interval:", opt.optimal_tariff_rate)
print("EV / baseline consumption (%):", opt.welfare_gain_pct)
print("Boundary:", opt.metadata["boundary"])

# Fixed actions in every payoff cell: 0 or 10%, with no implicit Nash label.
payoffs = compute_welfare_payoff_matrix(calib, player_a=players[0], player_b=players[1],
    optimal_a=.10, optimal_b=.10, **options)
print(payoffs.summary())
print("Is this finite game a Prisoner's Dilemma?", payoffs.is_prisoners_dilemma)
assert all(eq.converged for eq in payoffs.scenarios.values())

# %% [markdown]
# ### Diagnósticos numéricos de mejores respuestas
#
# Cada intento de equilibrio se audita. La recuperación automática prueba Newton, el método híbrido y luego continuación de Keller, sin relajar la tolerancia; si todos fallan, se produce un error. La convergencia exige mejores respuestas simultáneas en el vector final y ganancias por desviación normalizadas por consumo. El resultado puede ser inconcluso; debe comunicarse sin afirmar un teorema de Nash.

# %%
nash = solve_multilateral_nash_tariffs(calib, player_countries=players,
    tariff_max=tariff_ceiling, best_response_grid_size=7, max_iter=20,
    relaxation=.8, tol=1e-4, regret_tol=1e-6, **options)
print("Candidate tariffs:", nash.nash_tariffs)
print("Numerical convergence:", nash.converged)
print("Final simultaneous best-response gap:", nash.outer_error)
print("Maximum regret / baseline consumption:", nash.metadata["relative_max_regret"])
print("Boundary status:", nash.metadata["best_response_boundaries"])
print("Final GE attempts:", nash.equilibrium.metadata["policy_solver_attempts"])
if nash.converged:
    assert nash.equilibrium.converged and nash.outer_error <= 1e-4
    assert nash.metadata["relative_max_regret"] <= 1e-6
    assert not nash.metadata["inner_solver_failures"]

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
gain = 100 * opt.welfare_curve / opt.metadata["baseline_consumption_by_country"][0]
axes[0].plot(100 * opt.tariff_grid, gain, marker="o", color=_nbstyle.TINTA)
axes[0].set(title="Synthetic unilateral search", xlabel="Tariff (%)",
            ylabel="EV / baseline consumption (%)")
iterations = [h["iteration"] for h in nash.iteration_history]
gaps = [h["undamped_update_gap"] for h in nash.iteration_history]
axes[1].semilogy(iterations, np.maximum(gaps, 1e-16), marker="o", color=_nbstyle.TINTA)
axes[1].set(title="Undamped iteration gaps", xlabel="Iteration", ylabel="Policy gap")

# %% [markdown]
# ## Lectura de los resultados
#
# La curva monetaria de bienestar tiene una referencia igual a cero. Su representación porcentual usa consumo inicial fijo; no divide entre EV inicial. Cada celda de pagos usa las mismas dos acciones por jugador. La convergencia corresponde a la búsqueda numérica declarada: revisa el límite arancelario, las fronteras, la resolución y la ganancia final por desviarse. El ejercicio no demuestra aranceles óptimos únicos ni interiores.

# %% [markdown]
# ## Tu turno
#
# Cambia el límite y compara el candidato y sus diagnósticos. Una respuesta en el límite superior puede cambiar con la restricción institucional.

# %%
custom_ceiling = 0.20  # ← change this
assert 0 < custom_ceiling <= .50
custom = solve_multilateral_nash_tariffs(calib, player_countries=players,
    tariff_max=custom_ceiling, best_response_grid_size=7, max_iter=20,
    relaxation=.8, tol=1e-4, regret_tol=1e-6, **options)
assert all(0 <= t <= custom_ceiling for t in custom.nash_tariffs.values())
print("Alternative candidate:", custom.nash_tariffs)
print("Converged / gap / relative regret:", custom.converged, custom.outer_error,
      custom.metadata["relative_max_regret"])

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# Es un ejercicio docente reproducible, no un pronóstico empírico de una guerra comercial. Un equilibrio CES escalar derivado por separado y una minimización primal de gasto validan una cuadrícula completa de 41 por 41 perfiles; consulta `reviews/2026-09-20-hicksian-policy/REPORT.md`. Las búsquedas locales pueden omitir máximos estrechos. EV hicksiana está disponible para las preferencias y la contabilidad declaradas; la descomposición causal TOT/Alloc/TariffRec y la certificación de teoremas siguen sin estar disponibles. Consulta `docs/trade_policy.md`.
