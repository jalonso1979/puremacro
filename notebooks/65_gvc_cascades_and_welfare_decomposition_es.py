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
# # Propagación de costos en CGV, procedencia y límites de validación del bienestar
#
# **¿Cómo distinguir un contrafactual comercial resuelto de una afirmación de bienestar sin sustento?**
#
# Este tutorial utiliza tablas pequeñas generadas con estructuras de OECD ICIO, FIGARO y EXIOBASE. Son datos sintéticos para enseñanza, no observaciones de esas bases.

# %% [markdown]
# ## El método en matemáticas
#
# Con coeficientes de insumos fijos $A$, la propagación de costos satisface $dp=(I-A^T)^{-1}dv$, manteniendo fijos los precios de factores. El equilibrio general también modifica precios, cantidades, transferencias e impuestos. Un residuo pequeño solo verifica las ecuaciones especificadas.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Un aumento de costos llega a los clientes mediante sus compras de insumos. Tablas sintéticas similares pueden generar cascadas similares por construcción; esto no es evidencia empírica entre bases. La variación equivalente (equivalent variation) requiere una función de gasto y utilidad inicial independientes.

# %% [markdown]
# ## Código resuelto
#
# Generamos los datos de forma explícita y mostramos su procedencia antes de reportar resultados.

# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

repo = Path.cwd() if (Path.cwd() / "puremacro").is_dir() else Path.cwd().parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "notebooks"))
import _nbstyle
_nbstyle.apply_style()
from puremacro.trade.data import generate_synthetic_mrio, package_mrio_to_calibration_result
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
from puremacro.trade.regularize import compute_spectral_radius

# %% [markdown]
# ### Procedencia
#
# Los lectores de datos reales requieren un archivo por defecto. `fallback_to_synthetic=True` permite datos generados solo con autorización explícita. Las calibraciones conservan procedencia, conversiones y ajustes. La división entre trabajo y capital es un supuesto del modelo.

# %%
calibrations = {}
for source in ("oecd", "figaro", "exiobase"):
    raw = generate_synthetic_mrio(source, custom_c=3, custom_s=3, seed=42)
    calibrations[source] = package_mrio_to_calibration_result(raw)
provenance = pd.DataFrame([
    {"layout": name, "source": c.metadata["source"], "synthetic": c.metadata["is_synthetic"],
     "seed": c.metadata["seed"], "unit": c.metadata["unit"], "year": c.metadata["year"]}
    for name, c in calibrations.items()
])
print(provenance.to_string(index=False))
assert provenance["synthetic"].all()

# %% [markdown]
# ### Propagación de costos con precios fijos
#
# La cota espectral superior verifica productividad de estas matrices no negativas. El residuo del sistema lineal verifica este cálculo, no un teorema de bienestar.

# %%
network_rows = []
shock_size = 0.01  # ← change this
assert 0 < shock_size < 1
for name, c in calibrations.items():
    size = c.n_countries * c.n_sectors
    A = c.a.reshape(size, size, order="F")
    rho, lower, upper = compute_spectral_radius(A)
    assert upper < 1.0
    shock = np.zeros(size); shock[0] = shock_size
    # Fixed-coefficient unit-cost propagation, with factor prices held fixed.
    price_change = np.linalg.solve(np.eye(size) - A.T, shock)
    residual = np.max(np.abs((np.eye(size) - A.T) @ price_change - shock))
    assert residual < 1e-10
    network_rows.append({"layout": name, "rho": rho, "rho_upper": upper,
                         "max_cost_change": price_change.max(), "residual": residual})
network = pd.DataFrame(network_rows).set_index("layout")
print(network.to_string())

# %% [markdown]
# ### Un experimento arancelario resuelto
#
# Usamos una tabla didáctica separada, balanceada a mano, con dos países y categorías explícitas de consumo e inversión. Las tablas sintéticas anteriores ilustran redes y no son la fuente de este ejemplo de bienestar. Primero reportamos PIB nominal e ingresos arancelarios; después evaluamos el consumo.

# %%
# Separate hand-balanced teaching economy: two countries, C and investment.
Z = np.array([[10., 12.], [8., 14.]])
F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])
production_tax = np.array([4., 6.])
va = Z.sum(1) + F.sum(1) - Z.sum(0) - production_tax
table = np.vstack([np.hstack([Z, F]), np.r_[production_tax, [2., 1., 3., 2.]],
                   np.r_[2*va/3, np.zeros(4)], np.r_[va/3, np.zeros(4)]])
c = calibrate_trade_model(table, ns=1, nc=2, nfd=2, country_codes=["A", "B"])
base = solve_trade_equilibrium(c, method="condensed", accounting="consistent", tol=1e-10)
rates = np.zeros(c.n_countries); rates[0] = 0.10
counterfactual = solve_trade_equilibrium(c, tau=rates, tau_fd=rates,
                                        method="condensed", accounting="consistent", tol=1e-10, base_result=base)
assert base.converged and counterfactual.converged
outcomes = pd.DataFrame({
    "country": c.country_codes,
    "nominal_GDP_change_pct": 100 * (counterfactual.gdp / base.gdp - 1),
    "tariff_revenue": counterfactual.tariffs,
})
print(outcomes.to_string(index=False))
print("Maximum equilibrium residual:", counterfactual.max_residual)
# Exact Hicksian equivalent variation is not inferred from GDP or revenue.
# The historical decompose_hicksian_ev_3way TOT/Alloc labels remain quarantined.

# %% [markdown]
# ### Bienestar hicksiano del consumo
#
# La categoría 0 representa consumo; se excluye inversión. Con una canasta Leontief, la utilidad es su cantidad y la función de gasto es $e(P,U)=PU$. Por tanto, $EV=P_0(C_1-C_0)$, positiva para ganancias. La interfaz general también admite preferencias Cobb–Douglas entre categorías de consumo seleccionadas explícitamente. Las transferencias incluyen los aranceles una sola vez. La atribución a precios, ingreso factorial y transferencias compara estados finales; no es un teorema causal de términos de intercambio y eficiencia.

# %%
welfare_results = [compute_hicksian_welfare(c, counterfactual, base_result=base,
                                           target_country=i) for i in range(c.n_countries)]
outcomes["EV_pct_consumption"] = [r.ev_pct_consumption for r in welfare_results]
welfare_table = pd.DataFrame([
    {"country": r.country_code, "EV": r.ev, "CV": r.cv,
     "prices": r.price_effect, "factor_income": r.factor_income_effect,
     "fiscal_transfers": r.fiscal_transfer_effect, "residual": r.decomposition_residual}
    for r in welfare_results
])
for i, r in enumerate(welfare_results):
    direct_ev = base.Pfd_final[0, 0, i] * (counterfactual.c_fd[0, 0, i] - base.c_fd[0, 0, i])
    np.testing.assert_allclose(r.ev, direct_ev, atol=1e-8)
print(welfare_table.to_string(index=False))

# %% [markdown]
# ### Presentación de resultados numéricos

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
network["rho_upper"].plot.bar(ax=axes[0], color=_nbstyle.TINTA, rot=0)
axes[0].set(title="Synthetic networks: spectral upper bound", ylabel="Upper bound")
outcomes.set_index("country")["EV_pct_consumption"].plot.bar(ax=axes[1], color=_nbstyle.NOTA, rot=0)
axes[1].set(title="Synthetic tariff experiment", ylabel="EV / baseline consumption (%)")
axes[1].axhline(0, color=_nbstyle.SPINE, linewidth=.6)

# %% [markdown]
# ## Lectura de los resultados
#
# Las tablas identifican los datos como sintéticos. El residuo y las cotas espectrales son diagnósticos diferentes. EV y CV del consumo se obtienen de una función de gasto explícita y se contrastan con la valoración directa de cantidades. La nueva atribución utiliza precios de comprador, ingreso factorial y transferencias. La descomposición histórica TOT/Alloc y la certificación de teoremas siguen sin estar disponibles.

# %% [markdown]
# ## Tu turno
#
# Repite el experimento con otra semilla u otro proveedor. Compara CES y Leontief usando los mismos aranceles, cierre fiscal y tolerancia. Verifica los flujos reportados contra cada tecnología antes de interpretar diferencias.

# %% [markdown]
# ## ¿Qué tan exhaustivo es esto?
#
# Este ejemplo enseña procedencia, propagación de coeficientes fijos y una solución GE pequeña. No valida archivos nativos MRIO, comparaciones empíricas de bienestar, equilibrio de Nash o una descomposición causal de términos de intercambio y eficiencia. EV depende de las preferencias y del cierre fiscal declarados. Consulta `docs/STRUCTURAL_VALIDATION_STATUS.md` para los límites actuales.
