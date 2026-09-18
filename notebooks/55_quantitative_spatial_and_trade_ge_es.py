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
# # Economía Espacial Cuantitativa y Equilibrio General de Comercio: Álgebra de Variaciones de Caliendo-Parro y Geografía Económica de Allen-Arkolakis
#
# **¿Cómo se propagan los choques arancelarios internacionales, las perturbaciones en las cadenas de suministro y las inversiones regionales en infraestructura de transporte a través de los encadenamientos insumo-producto y la movilidad laboral espacial para remodelar los flujos comerciales, la distribución regional de la población y el bienestar económico agregado?**
#
# En la economía espacial y del comercio internacional contemporánea, las intervenciones de política localizada y las fricciones geográficas no operan de forma aislada. En el ámbito del comercio global, los procesos de producción se encuentran profundamente fragmentados a través de las fronteras nacionales: los insumos intermedios representan más de la mitad del comercio mundial total, lo que significa que los aranceles aplicados a componentes intermedios aguas arriba se propagan en cascada hacia los sectores aguas abajo mediante redes de insumo-producto, distorsionando precios relativos, desviando flujos comerciales hacia terceros países y generando complejas repercusiones de equilibrio general sobre el bienestar. Simultáneamente, en el espacio geográfico, los trabajadores y las empresas reaccionan endógenamente ante reducciones en los costos de transporte: menores fricciones de flete expanden el acceso al mercado, induciendo flujos migratorios laborales hacia regiones de alto acceso hasta que las externalidades centrípetas de aglomeración son contrarrestadas por la congestión centrífuga en vivienda y amenidades.
#
# Este cuaderno de exhibición implementa, resuelve y analiza los dos marcos analíticos de equilibrio general fundacionales en la economía espacial y del comercio cuantitativo:
# 1. El modelo de comercio ricardiano multipaís y multisectorial de **Caliendo y Parro (2015)** con encadenamientos insumo-producto, elasticidades comerciales sectoriales, aranceles y desequilibrios comerciales, resuelto mediante Álgebra de Variaciones Proporcionales Exactas (Exact Hat Algebra).
# 2. El modelo de equilibrio general espacial geográfico continuo de **Allen y Arkolakis (2014)** con costos de transporte tipo iceberg, aglomeración de productividad, congestión de amenidades y movilidad laboral espacial perfecta.

# %% [markdown]
# ## El método en matemáticas — Comercio Cuantitativo y Equilibrio General Espacial
#
# **1. Álgebra de Variaciones Proporcionales Exactas con Encadenamientos Insumo-Producto de Caliendo y Parro (2015).**
# Considérese una economía internacional con $N$ países ($i, n = 1, \dots, N$) y $J$ sectores ($j, k = 1, \dots, J$). Los hogares representativos del país $n$ maximizan una utilidad Cobb-Douglas sobre bienes compuestos sectoriales con participaciones de gasto $\alpha_n^j$ ($\sum_{j=1}^J \alpha_n^j = 1$):
# $$ U_n = \prod_{j=1}^J (C_n^j)^{\alpha_n^j}. $$
# La producción sectorial $Y_n^j$ se genera combinando trabajo (con participación de valor agregado $\gamma_n^j > 0$) e insumos intermedios de todos los sectores $k=1, \dots, J$ (con participaciones de costo $\gamma_n^{j, k} \ge 0$), bajo rendimientos constantes a escala:
# $$ \gamma_n^j + \sum_{k=1}^J \gamma_n^{j, k} = 1, \qquad c_n^j = \Upsilon_n^j w_n^{\gamma_n^j} \prod_{k=1}^J (P_n^k)^{\gamma_n^{j, k}}, $$
# donde $c_n^j$ denota el costo unitario de la canasta de insumos y $\Upsilon_n^j$ es una constante tecnológica. En el Álgebra de Variaciones Proporcionales Exactas (donde $\hat{x} \equiv x' / x$ representa el cambio relativo entre el equilibrio contrafactual y el de referencia), la variación en el costo unitario satisface:
# $$ \ln \hat{c}_i^j = \gamma_i^j \ln \hat{w}_i + \sum_{k=1}^J \gamma_i^{j, k} \ln \hat{P}_i^k. $$
# Los bienes del sector $j$ se comercializan internacionalmente sujetos a costos de transporte tipo iceberg $d_{ni}^j \ge 1$ y aranceles ad-valorem brutos $\tau_{ni}^j = 1 + t_{ni}^j \ge 1$, definiendo costos comerciales totales $\kappa_{ni}^j = \tau_{ni}^j d_{ni}^j$. Bajo la dispersión de productividad de Fréchet de Eaton y Kortum (2002) con elasticidad comercial sectorial $\theta_j > 0$, la variación en el índice de precios sectorial $\hat{P}_n^j$ y las cuotas de gasto bilateral contrafactuales $\pi_{ni}'^j$ satisfacen:
# $$ \hat{P}_n^j = \left[ \sum_{i=1}^N \pi_{ni}^j \left( \hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j \right)^{-\theta_j} \right]^{-1/\theta_j}, \qquad \pi_{ni}'^j = \pi_{ni}^j \left( \frac{\hat{c}_i^j \hat{\tau}_{ni}^j \hat{d}_{ni}^j}{\hat{P}_n^j} \right)^{-\theta_j}. $$
# El gasto nacional total $I_n'$ agrega el ingreso laboral contrafactual $w_n' L_n$, los ingresos arancelarios $R_n'$ y el déficit comercial contrafactual $D_n'$ ($I_n' = w_n' L_n + R_n' + D_n'$). El gasto sectorial $X_n'^j$ y el producto bruto $Y_i'^j$ satisfacen las identidades de vaciado de mercado:
# $$ X_n'^j = \alpha_n^j I_n' + \sum_{k=1}^J \gamma_n^{k, j} Y_n'^k, \qquad Y_i'^j = \sum_{n=1}^N \frac{\pi_{ni}'^j}{\tau_{ni}'^j} X_n'^j, \qquad w_n' L_n = \sum_{j=1}^J \gamma_n^j Y_n'^j. $$
# Las variaciones en el salario real y el cambio porcentual en el bienestar nacional están dados por:
# $$ \widehat{\left(\frac{w_n}{P_n}\right)} = \frac{\hat{w}_n}{\prod_{j=1}^J (\hat{P}_n^j)^{\alpha_n^j}}, \qquad \hat{W}_n = \frac{\hat{I}_n}{\prod_{j=1}^J (\hat{P}_n^j)^{\alpha_n^j}}, \qquad \Delta W_n (\%) = (\hat{W}_n - 1) \times 100. $$
#
# **2. Equilibrio General Espacial Cuantitativo de Allen y Arkolakis (2014).**
# Considérese una economía espacial continua con $N$ regiones, una dotación total de trabajo $\bar{L} = \sum_{i=1}^N L_i$ y costos de transporte tipo iceberg $\tau_{ij} \ge 1$ ($\tau_{ii} = 1$). La utilidad indirecta en la región $i$ depende de los salarios reales y de las amenidades locales $a_i$:
# $$ u_i = a_i \frac{w_i}{P_i}, \qquad a_i = \bar{a}_i L_i^\beta, $$
# donde $\bar{a}_i > 0$ representa las amenidades fundamentales y $\beta < 0$ captura la congestión de amenidades (escasez de suelo, vivienda o costos de traslado urbano). La libre movilidad del trabajo iguala la utilidad real en todas las regiones pobladas ($u_i = \bar{u}$), determinando la distribución espacial de la población:
# $$ L_i = \bar{L} \frac{\left(\bar{a}_i w_i / P_i\right)^{-1/\beta}}{\sum_{k=1}^N \left(\bar{a}_k w_k / P_k\right)^{-1/\beta}}. $$
# El índice de precios CES $P_i$ y el acceso al mercado de firmas $\text{FMA}_i$ dependen de la elasticidad comercial $\theta$ y de la productividad local $A_j = \bar{A}_j L_j^\alpha$, donde $\alpha > 0$ modela derrames de aglomeración marshallianos:
# $$ P_i^{-\theta} = \sum_{j=1}^N \tau_{ji}^{-\theta} \left( \frac{w_j}{\bar{A}_j L_j^\alpha} \right)^{-\theta}, \qquad \text{FMA}_i = \sum_{j=1}^N \tau_{ij}^{-\theta} P_j^\theta w_j L_j. $$
# El vaciado del mercado de bienes ($w_i L_i = \sum_j \pi_{ji} w_j L_j$) determina los salarios de equilibrio:
# $$ w_i^{1 + \theta} = (\bar{A}_i L_i^\alpha)^\theta L_i^{-1} \text{FMA}_i. $$
# La existencia y unicidad del equilibrio general espacial está garantizada siempre que los parámetros satisfagan el Teorema 2 de Allen y Arkolakis (2014): $\alpha + \beta \le \frac{\theta}{1 + \theta}$ y $\alpha \le \frac{1}{\theta}$.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Los modelos cuantitativos de comercio y economía espacial revelan cómo los choques localizados se propagan a través de redes macroeconómicas complejas, transformando señales de precios locales en reasignaciones sistémicas de la producción, el empleo y el bienestar económico.
#
# En el comercio internacional, el marco de **Caliendo y Parro (2015)** resalta dos canales de transmisión determinantes:
# 1. **Desvío de comercio y manipulación de términos de intercambio:** Cuando el País A eleva aranceles sobre las importaciones del País B, el precio doméstico de los bienes de B se incrementa. Dado que las variedades son sustitutos imperfectos según la elasticidad comercial $\theta_j$, los consumidores y productores aguas abajo sustituyen compras hacia socios comerciales no gravados (el Resto del Mundo) y productores nacionales. Si bien el País A puede obtener temporalmente ingresos arancelarios y mejorar sus términos de intercambio en un choque unilateral, una guerra comercial recíproca desencadena represalias comerciales simétricas, eliminando los beneficios en términos de intercambio y dejando a ambas economías con pérdidas netas de eficiencia derivadas de la distorsión productiva y la caída del ingreso real.
# 2. **Multiplicadores en cadenas de suministro:** Los modelos comerciales clásicos suponen que los bienes se producen únicamente a partir de factores primarios (trabajo y capital). En la práctica, las manufacturas dependen intensamente de insumos intermedios. Cuando los aranceles encarecen el acero o los componentes electrónicos importados, las industrias automotriz, aeroespacial y de maquinaria enfrentan mayores costos de producción $\hat{c}_i^j$. Estos sobrecostos se propagan a través de las matrices insumo-producto nacionales ($\gamma_i^{j, k}$) y vuelven a cruzar fronteras en las exportaciones de bienes intermedios, acumulando distorsiones arancelarias y multiplicando las pérdidas de bienestar global. De manera fundamental, el **Álgebra de Variaciones Proporcionales Exactas** resuelve este sistema de equilibrio general utilizando exclusivamente las cuotas de comercio de referencia observables $\pi_{ni}^j$, los coeficientes insumo-producto $\gamma_n^{j, k}$ y las elasticidades comerciales $\theta_j$, obviando por completo la estimación de niveles absolutos no observables de productividad fundamental o barreras físicas de transporte.
#
# En el espacio geográfico, el modelo de **Allen y Arkolakis (2014)** formaliza la interacción entre fuerzas centrípetas y centrífugas:
# - **Expansión del acceso al mercado:** La construcción o mejora de infraestructura de transporte (como corredores de carga de alta velocidad, autopistas o modernizaciones portuarias) reduce las fricciones bilaterales de flete $\tau_{ij}$. Esto expande de inmediato el acceso al mercado de consumidores ($\text{CMA}_i = P_i^{-\theta}$), abaratando los índices locales de precios de importación, al tiempo que expande el acceso al mercado de firmas ($\text{FMA}_i$), permitiendo a los productores locales abastecer mercados más distantes con mayor rentabilidad.
# - **Equilibrio entre aglomeración y congestión:** El impulso inicial a los salarios reales ($w_i / P_i$) atrae trabajadores desde regiones periféricas. A medida que la fuerza de trabajo se concentra en los nodos conectados, las **economías de aglomeración** ($\alpha > 0$) potencian la productividad mediante mercados laborales más densos, mayor especialización de insumos y derrames de conocimiento. No obstante, la concentración ilimitada se ve frenada por la **congestión de amenidades** ($\beta < 0$): al acumularse la población, el suelo y la vivienda se vuelven escasos, los alquileres suben y la infraestructura pública se satura. El equilibrio espacial general se alcanza cuando la utilidad real de los trabajadores se iguala exactamente en todas las regiones pobladas ($u_i = \bar{u}$), preservando la masa total de población nacional y generando ganancias positivas en el bienestar agregado.

# %%
# Preámbulo: importar librerías numéricas, estilo de gráficos y solucionadores de equilibrio general
import sys
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.trade.caliendo_parro import CaliendoParroModel
from puremacro.spatial.allen_arkolakis import AllenArkolakisModel

# Fijar semilla pseudoaleatoria determinista para reproducibilidad
rng = np.random.default_rng(42)

# --- Calibración de economía global de Caliendo-Parro (2015): 3 países y 2 sectores ---
N_cp, J_cp = 3, 2
country_codes = ["USA", "CHN", "ROW"]
sector_codes = ["Manufactures", "Services"]

# Cuotas de comercio bilateral: forma (J, N, N) donde pi[j, n, i] = cuota de n en i
trade_shares = np.array([
    [[0.55, 0.25, 0.20], [0.15, 0.70, 0.15], [0.25, 0.25, 0.50]],
    [[1.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 0.00, 1.00]],
], dtype=float)

# Participaciones de valor agregado gamma_va: forma (N, J)
gamma_va = np.array([[0.40, 0.60], [0.35, 0.65], [0.45, 0.55]])

# Coeficientes de insumo-producto gamma_io: forma (N, J, J)
gamma_io = np.zeros((N_cp, J_cp, J_cp))
for n in range(N_cp):
    for j in range(J_cp):
        rem = 1.0 - gamma_va[n, j]
        gamma_io[n, j, 0] = rem * 0.60
        gamma_io[n, j, 1] = rem * 0.40

# Participaciones de gasto en consumo final alpha: forma (N, J)
alpha = np.array([[0.30, 0.70], [0.45, 0.55], [0.35, 0.65]])

# Elasticidades comerciales sectoriales theta: forma (J,)
theta_cp = np.array([5.0, 4.0])

# Ingreso laboral de referencia (PIB) y déficit comercial inicial: forma (N,)
labor_income = np.array([120.0, 90.0, 100.0])
deficits = np.array([10.0, -10.0, 0.0])

# Construir modelo base de Caliendo-Parro
cp_model = CaliendoParroModel(
    trade_shares=trade_shares,
    gamma_va=gamma_va,
    gamma_io=gamma_io,
    alpha=alpha,
    theta=theta_cp,
    labor_income=labor_income,
    deficits=deficits,
    nontradables=[1],
    country_codes=country_codes,
    sector_codes=sector_codes,
)

# --- Configuración geográfica espacial de 5 regiones de Allen-Arkolakis (2014) ---
region_names = ["North", "South", "East", "West", "Central"]
coords = np.array([
    [45.0, -93.0],
    [30.0, -90.0],
    [40.7, -74.0],
    [37.7, -122.4],
    [38.6, -90.2],
])

aa_model = AllenArkolakisModel.from_coordinates(
    coords,
    region_names=region_names,
    theta=4.0,
    alpha=0.08,
    beta=-0.35,
    total_population=100.0,
)

print(f"Caliendo-Parro Model: {N_cp} countries, {J_cp} sectors ({sector_codes[0]}: Tradable, {sector_codes[1]}: Non-Tradable)")
print(f"Allen-Arkolakis Model: {len(region_names)} regions, Total Population = {aa_model.total_population:.1f}")
print(f"Spatial Uniqueness Condition Satisfied: {aa_model.is_unique}")

# %%
# --- Experimento 1: Aranceles, desvío de comercio y guerras comerciales (Caliendo-Parro 2015) ---
# 1. Choque arancelario unilateral: EE. UU. impone un arancel del 15% sobre manufacturas chinas
uni_res = cp_model.simulate_tariff_shock(
    importer="USA", exporter="CHN", sector="Manufactures", tariff_rate=0.15
)

# 2. Guerra comercial bilateral recíproca: EE. UU. y China imponen aranceles del 25% mutuos en todos los bienes transables
war_res = cp_model.simulate_trade_war(
    coalition_a=["USA"], coalition_b=["CHN"], tariff_rate=0.25
)

# Imprimir diagnósticos principales
print("--- Experiment 1: Caliendo-Parro Simulation Diagnostics ---")
print(f"Unilateral Tariff: Converged = {uni_res.converged} | Residual = {uni_res.market_clearing_residual:.2e}")
print(f"  USA Welfare Change: {uni_res.welfare_pct[0]:+.2f}% | Tariff Revenue = {uni_res.tariff_revenue_prime[0]:.2f}")
print(f"  CHN Welfare Change: {uni_res.welfare_pct[1]:+.2f}% | Tariff Revenue = {uni_res.tariff_revenue_prime[1]:.2f}")
print(f"  ROW Welfare Change: {uni_res.welfare_pct[2]:+.2f}% | Tariff Revenue = {uni_res.tariff_revenue_prime[2]:.2f}")

print(f"\nReciprocal Trade War (25%): Converged = {war_res.converged} | Residual = {war_res.market_clearing_residual:.2e}")
print(f"  USA Welfare Change: {war_res.welfare_pct[0]:+.2f}%")
print(f"  CHN Welfare Change: {war_res.welfare_pct[1]:+.2f}%")
print(f"  ROW Welfare Change: {war_res.welfare_pct[2]:+.2f}%")

# Aserciones en línea: verificar convergencia, vaciado de mercado y reorientación de flujos comerciales
assert uni_res.converged and war_res.converged, "Both Caliendo-Parro solves must converge"
assert uni_res.market_clearing_residual < 1e-6, f"Unilateral residual {uni_res.market_clearing_residual:.2e} exceeds 1e-6"
assert war_res.market_clearing_residual < 1e-6, f"Trade war residual {war_res.market_clearing_residual:.2e} exceeds 1e-6"
assert war_res.welfare_pct[0] < 0.0, "USA must suffer net welfare contraction in trade war"
assert war_res.welfare_pct[1] < 0.0, "China must suffer net welfare contraction in trade war"
assert war_res.pi_prime[0, 0, 1] < trade_shares[0, 0, 1], "US import share from China must fall"
assert war_res.pi_prime[0, 0, 2] > trade_shares[0, 0, 2], "Trade diversion: US import share from ROW must rise"

# Figura 1: Reasignación de cuotas de comercio y comparación de bienestar
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.2))

# Subgráfico A: Cuotas de comercio base en manufacturas
im_base = axes1[0].imshow(trade_shares[0], cmap=_nbstyle.CMAP_SEQ, vmin=0.0, vmax=1.0)
axes1[0].set_title("(A) Baseline Trade Shares $\\pi_{ni}^0$ (Mfg)")
axes1[0].set_xticks(range(N_cp))
axes1[0].set_yticks(range(N_cp))
axes1[0].set_xticklabels(country_codes)
axes1[0].set_yticklabels(country_codes)
axes1[0].set_xlabel("Exporter ($i$)")
axes1[0].set_ylabel("Importer ($n$)")
for n in range(N_cp):
    for i in range(N_cp):
        val = trade_shares[0, n, i]
        col = _nbstyle.FONDO if val > 0.55 else _nbstyle.TINTA
        axes1[0].text(i, n, f"{val:.2f}", ha="center", va="center", color=col, fontsize=9.5, fontweight="bold")
plt.colorbar(im_base, ax=axes1[0], fraction=0.046, pad=0.04)

# Subgráfico B: Cuotas de comercio contrafactuales en manufacturas (Guerra del 25%)
im_war = axes1[1].imshow(war_res.pi_prime[0], cmap=_nbstyle.CMAP_SEQ, vmin=0.0, vmax=1.0)
axes1[1].set_title("(B) Counterfactual Trade Shares $\\pi_{ni}'^0$ (25% War)")
axes1[1].set_xticks(range(N_cp))
axes1[1].set_yticks(range(N_cp))
axes1[1].set_xticklabels(country_codes)
axes1[1].set_yticklabels(country_codes)
axes1[1].set_xlabel("Exporter ($i$)")
axes1[1].set_ylabel("Importer ($n$)")
for n in range(N_cp):
    for i in range(N_cp):
        val = war_res.pi_prime[0, n, i]
        col = _nbstyle.FONDO if val > 0.55 else _nbstyle.TINTA
        axes1[1].text(i, n, f"{val:.2f}", ha="center", va="center", color=col, fontsize=9.5, fontweight="bold")
plt.colorbar(im_war, ax=axes1[1], fraction=0.046, pad=0.04)

# Subgráfico C: Impacto en el bienestar real por país
x_bar = np.arange(N_cp)
bar_w = 0.35
axes1[2].bar(x_bar - bar_w / 2, uni_res.welfare_pct, width=bar_w, label="Unilateral Tariff (USA 15% on CHN)", color=_nbstyle.S2["color"], edgecolor=_nbstyle.FONDO, lw=0.8)
axes1[2].bar(x_bar + bar_w / 2, war_res.welfare_pct, width=bar_w, label="Bilateral Trade War (USA & CHN 25%)", color=_nbstyle.S1["color"], edgecolor=_nbstyle.FONDO, lw=0.8)
axes1[2].axhline(0, color=_nbstyle.SPINE, linestyle="--", linewidth=0.8)
axes1[2].set_xticks(x_bar)
axes1[2].set_xticklabels(country_codes)
axes1[2].set_ylabel("Welfare Change $\\Delta W / W$ (%)")
axes1[2].set_title("(C) Real Welfare Impact by Country")
axes1[2].legend(loc="lower left", fontsize=8.5)

# %%
# --- Experimento 2: Geografía espacial y corredor de infraestructura (Allen-Arkolakis 2014) ---
# 1. Equilibrio general espacial de referencia
base_res = aa_model.solve_equilibrium(tol=1e-8)

# 2. Contrafactual: Corredor de transporte Norte-Sur (reducción del 20% en costos de transporte tipo iceberg)
rail_res = aa_model.simulate_infrastructure_shock("North", "South", cost_reduction=0.20)

print("--- Experiment 2: Allen-Arkolakis Spatial Simulation Diagnostics ---")
print(f"Baseline Spatial Solve: Converged = {base_res.converged} | Max Residual = {base_res.max_residual:.2e}")
print(f"  Labor Conservation Residual: {base_res.labor_conservation_residual:.2e}")
print(f"  Spatial Utility Variance: {base_res.spatial_utility_variance:.2e} (Equalized Utility u_bar = {base_res.welfare:.4f})")

print(f"\nInfrastructure Shock (North-South -20% Cost): Converged = {rail_res.converged}")
print(f"  Aggregate Spatial Welfare Change: {rail_res.welfare_pct:+.4f}%")
print(f"  Labor Conservation Residual: {rail_res.labor_conservation_residual:.2e}")
print(f"  Spatial Utility Dispersion: {rail_res.spatial_utility_variance:.2e}")

# Aserciones en línea: verificar conservación de trabajo, convergencia, bienestar positivo y reasignación laboral
assert base_res.converged and rail_res.converged, "Both spatial equilibrium solves must converge"
assert base_res.labor_conservation_residual < 1e-12, "Baseline labor conservation failed"
assert rail_res.labor_conservation_residual < 1e-12, "Counterfactual labor conservation failed"
assert np.isclose(np.sum(rail_res.population), aa_model.total_population, atol=1e-10), "Total population mass not conserved"
assert rail_res.welfare_pct > 0.0, f"Infrastructure investment must yield positive aggregate welfare, got {rail_res.welfare_pct}"
assert rail_res.L_hat[0] > 1.0 and rail_res.L_hat[1] > 1.0, "Connected regions (North, South) must attract population"
assert rail_res.spatial_utility_variance < 1e-6, "Spatial price and utility equalization violated"

# Figura 2: Red geográfica y reasignación de equilibrio espacial
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.6))

# Subgráfico A: Mapa de red espacial geográfica y corredor de transporte
ax_geo = axes2[0]
lats, lons = coords[:, 0], coords[:, 1]
# Trazar enlaces comerciales de fondo
for i in range(len(region_names)):
    for j in range(i + 1, len(region_names)):
        ax_geo.plot([lons[i], lons[j]], [lats[i], lats[j]], color=_nbstyle.REJILLA, linestyle=":", linewidth=1.0, zorder=1)

# Resaltar corredor de transporte mejorado Norte-Sur
n_idx, s_idx = region_names.index("North"), region_names.index("South")
ax_geo.plot([lons[n_idx], lons[s_idx]], [lats[n_idx], lats[s_idx]], color=_nbstyle.TINTA, linestyle="-", linewidth=2.6, label="Upgraded North-South Corridor (-20% $\\tau$)", zorder=2)

# Gráfico de dispersión de regiones con tamaño proporcional a la población base
pop_sizes = base_res.population * 25.0
scatter = ax_geo.scatter(lons, lats, s=pop_sizes, c=_nbstyle.S2["color"], edgecolors=_nbstyle.FONDO, linewidths=1.2, zorder=3)

# Etiquetar regiones con variaciones de población y salarios
for i, name in enumerate(region_names):
    dx, dy = 1.2, 0.6
    if name == "West":
        dx = -8.0
    ax_geo.text(lons[i] + dx, lats[i] + dy, f"{name}\n($\\hat{{L}}={rail_res.L_hat[i]:.3f}$)", fontsize=8.5, fontweight="bold", zorder=4)

ax_geo.set_title("(A) Geographic Spatial Network & Corridor")
ax_geo.set_xlabel("Longitude ($^\\circ$W)")
ax_geo.set_ylabel("Latitude ($^\\circ$N)")
ax_geo.legend(loc="lower left", fontsize=8.5)

# Subgráfico B: Variaciones regionales en población y salarios
x_loc = np.arange(len(region_names))
w_bar2 = 0.35
pop_pct = (rail_res.L_hat - 1.0) * 100.0
wage_pct = (rail_res.w_hat - 1.0) * 100.0

axes2[1].bar(x_loc - w_bar2 / 2, pop_pct, width=w_bar2, label="Population Change $\\hat{L}_i - 1$ (%)", color=_nbstyle.S1["color"], edgecolor=_nbstyle.FONDO, lw=0.8)
axes2[1].bar(x_loc + w_bar2 / 2, wage_pct, width=w_bar2, label="Nominal Wage Change $\\hat{w}_i - 1$ (%)", color=_nbstyle.S2["color"], edgecolor=_nbstyle.FONDO, lw=0.8)
axes2[1].axhline(0, color=_nbstyle.SPINE, linestyle="--", linewidth=0.8)
axes2[1].set_xticks(x_loc)
axes2[1].set_xticklabels(region_names)
axes2[1].set_ylabel("Percentage Change (%)")
axes2[1].set_title("(B) Regional Labor Reallocation & Wage Responses")
axes2[1].legend(loc="upper right", fontsize=8.5)

# %%
# --- Experimento 3: Cascadas en cadenas de suministro — Amplificación insumo-producto ---
# Calibrar modelo contrafactual sin insumos intermedios (gamma_va = 1.0, ricardiano puro con solo trabajo)
gamma_va_noio = np.ones((N_cp, J_cp))
gamma_io_noio = np.zeros((N_cp, J_cp, J_cp))

cp_model_noio = CaliendoParroModel(
    trade_shares=trade_shares,
    gamma_va=gamma_va_noio,
    gamma_io=gamma_io_noio,
    alpha=alpha,
    theta=theta_cp,
    labor_income=labor_income,
    deficits=deficits,
    nontradables=[1],
    country_codes=country_codes,
    sector_codes=sector_codes,
)

# Simular guerra comercial idéntica del 25% en modelo sin encadenamientos insumo-producto
war_res_noio = cp_model_noio.simulate_trade_war(
    coalition_a=["USA"], coalition_b=["CHN"], tariff_rate=0.25
)

print("--- Experiment 3: Input-Output Linkage Comparison ---")
print(f"Trade War with Full I-O Linkages:   USA Welfare = {war_res.welfare_pct[0]:+.2f}% | CHN Welfare = {war_res.welfare_pct[1]:+.2f}%")
print(f"Trade War without I-O Linkages:      USA Welfare = {war_res_noio.welfare_pct[0]:+.2f}% | CHN Welfare = {war_res_noio.welfare_pct[1]:+.2f}%")

# Aserciones en línea: verificar convergencia del solucionador e impacto de insumos intermedios
assert war_res_noio.converged, "No-I-O trade war solve must converge"
assert war_res_noio.market_clearing_residual < 1e-6, "Market clearing residual must be < 1e-6"
assert not np.allclose(war_res.welfare_pct, war_res_noio.welfare_pct), "I-O linkages must alter quantitative welfare responses"

# Figura 3: Comparación de bienestar con vs. sin encadenamientos insumo-producto
fig3, ax3 = plt.subplots(figsize=(7.5, 4.0))
x_io = np.arange(N_cp)
w_io = 0.35

ax3.bar(x_io - w_io / 2, war_res.welfare_pct, width=w_io, label="With Full Input-Output Linkages (Caliendo-Parro)", color=_nbstyle.S1["color"], edgecolor=_nbstyle.FONDO, lw=0.8)
ax3.bar(x_io + w_io / 2, war_res_noio.welfare_pct, width=w_io, label="Without Input-Output Linkages (Pure Ricardian)", color=_nbstyle.S2["color"], edgecolor=_nbstyle.FONDO, lw=0.8)
ax3.axhline(0, color=_nbstyle.SPINE, linestyle="--", linewidth=0.8)
ax3.set_xticks(x_io)
ax3.set_xticklabels(country_codes)
ax3.set_ylabel("Welfare Change $\\Delta W / W$ (%)")
ax3.set_title("Input-Output Supply Chain Amplification of Trade War Losses")
ax3.legend(loc="lower left", fontsize=8.5)

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** Los tres experimentos numéricos ilustran los mecanismos determinantes que rigen las barreras al comercio internacional y la geografía económica espacial:
#
# 1. **Guerras comerciales y desvío de comercio con insumo-producto (Experimento 1):** En la simulación de equilibrio general de Caliendo-Parro, la imposición de un arancel unilateral del 15% por parte de Estados Unidos sobre las manufacturas chinas genera inicialmente una modesta ganancia en el ingreso real (+0.33%) para EE. UU. mediante una mejora en los términos de intercambio, provocando una contracción del bienestar en China (-0.22%) y un efecto derrame adverso para terceros países en el Resto del Mundo (-1.16%). No obstante, cuando China responde de forma recíproca con aranceles del 25% sobre todos los bienes transables en una guerra comercial bilateral, la ventaja en los términos de intercambio se disipa por completo. Ambas economías sufren pérdidas netas de bienestar significativas (-1.11% en EE. UU. y -0.69% en China). Asimismo, la matriz de cuotas de comercio bilateral revela un marcado desvío comercial: el gasto estadounidense en importaciones manufactureras chinas colapsa del 25.0% al 9.8%, mientras que las compras al Resto del Mundo aumentan del 20.0% al 26.0% y la absorción doméstica se incrementa del 55.0% al 64.2%.
# 2. **Corredores de infraestructura y reasignación espacial del empleo (Experimento 2):** En el marco geográfico de Allen-Arkolakis, la modernización de la infraestructura de transporte en el corredor Norte-Sur (reduciendo los costos de transporte tipo iceberg en un 20%) induce un incremento del bienestar espacial agregado de +0.0627%. La reducción de los costos bilaterales de flete expande el acceso al mercado de consumidores ($\text{CMA}_i$) y el acceso al mercado de firmas ($\text{FMA}_i$), desencadenando una migración laboral endógena. La población en las regiones directamente beneficiadas (Norte y Sur) crece un +0.34% y +0.35%, respectivamente, absorbiendo mano de obra de las regiones Este, Oeste y Central (-0.23%). La respuesta salarial espacial demuestra cómo las fuerzas de aglomeración ($\alpha = 0.08$) aumentan la productividad local, mientras que la congestión de amenidades ($\beta = -0.35$) equilibra la afluencia de trabajadores para preservar la igualación espacial de precios y utilidad ($\text{Var}(u_i) / \bar{u} < 10^{-6}$) con conservación estricta de la población total ($\sum L_i = 100.000$).
# 3. **Cascadas en cadenas de suministro y amplificación insumo-producto (Experimento 3):** Al comparar el contrafactual de guerra comercial bajo el modelo completo de Caliendo-Parro frente a una economía ricardiana pura sin bienes intermedios ($\gamma_{va} = 1.0$), se comprueba el papel crucial de las cadenas globales de valor. Los insumos intermedios generan un multiplicador en la red productiva: los aranceles a componentes aguas arriba elevan los costos de producción para las industrias aguas abajo, propagando las distorsiones de precios a través de múltiples etapas de producción. En consecuencia, las pérdidas de bienestar en una economía moderna interconectada difieren sustancialmente de los modelos clásicos que ignoran los encadenamientos insumo-producto.

# %%
# Tu turno: calibrar aranceles comerciales e inversiones en infraestructura de transporte
# Ajuste los parámetros a continuación para ejecutar sus propios experimentos contrafactuales de política.
# La celda ejecutable reevalúa ambos modelos de equilibrio general y verifica las aserciones subsiguientes.

# ← change this: Tasa arancelaria unilateral impuesta por EE. UU. sobre manufacturas chinas (p. ej., 0.05, 0.15, 0.30)
tariff_rate_custom = 0.15

# ← change this: Reducción en costos de transporte entre Norte y Sur (p. ej., 0.10, 0.20, 0.35)
cost_reduction_custom = 0.20

# 1. Re-simular choque arancelario unilateral personalizado en el modelo de Caliendo-Parro
res_cp_custom = cp_model.simulate_tariff_shock(
    importer="USA", exporter="CHN", sector="Manufactures", tariff_rate=tariff_rate_custom
)

# 2. Re-simular reducción de costos de transporte personalizada en el modelo de Allen-Arkolakis
res_aa_custom = aa_model.simulate_infrastructure_shock(
    "North", "South", cost_reduction=cost_reduction_custom
)

print(f"Custom Run Results (Tariff Rate = {tariff_rate_custom:.2f}, Cost Reduction = {cost_reduction_custom:.2f}):")
print(f"  Caliendo-Parro USA Welfare Change : {res_cp_custom.welfare_pct[0]:+.2f}% (Converged: {res_cp_custom.converged})")
print(f"  Caliendo-Parro USA Tariff Revenue  : {res_cp_custom.tariff_revenue_prime[0]:.2f}")
print(f"  Allen-Arkolakis Spatial Welfare    : {res_aa_custom.welfare_pct:+.4f}% (Converged: {res_aa_custom.converged})")
print(f"  Allen-Arkolakis North Population   : {res_aa_custom.population[0]:.2f} (Hat: {res_aa_custom.L_hat[0]:.4f})")

# Aserciones subsiguientes que validan los parámetros personalizados y la integridad de la solución
assert tariff_rate_custom >= 0.0, "Tariff rate must be non-negative"
assert 0.0 < cost_reduction_custom < 1.0, "Cost reduction must be in (0, 1)"
assert res_cp_custom.converged, "Custom Caliendo-Parro solver failed to converge"
assert res_aa_custom.converged, "Custom Allen-Arkolakis solver failed to converge"
assert res_cp_custom.market_clearing_residual < 1e-6, "Caliendo-Parro market clearing residual exceeds 1e-6"
assert res_aa_custom.labor_conservation_residual < 1e-12, "Allen-Arkolakis labor conservation failed"
assert res_aa_custom.welfare_pct > 0.0, "Infrastructure improvement must yield positive welfare gain"

# %% [markdown]
# **Preguntas y desafíos.**
# 1. *Básico:* Aumente la tasa arancelaria unilateral `tariff_rate_custom` de 0.15 a 0.35. Observe cómo barreras arancelarias más elevadas generan inicialmente mayor recaudación pero contraen progresivamente los volúmenes de comercio y acentúan el desvío comercial hacia socios comerciales no gravados.
# 2. *Intermedio:* Modifique el parámetro de reducción de costos de transporte `cost_reduction_custom` entre 0.05 y 0.40. Verifique que la afluencia de población hacia las regiones del corredor conectado escala de forma continua con la magnitud del subsidio al transporte, mientras que la población total se conserva con precisión de máquina ($|\sum L_i - \bar{L}| < 10^{-12}$).
# 3. *Avanzado:* Explore el impacto de la elasticidad de congestión $\beta$. En `aa_model`, disminuya el valor absoluto de $\beta$ hacia cero (por ejemplo, $\beta = -0.15$). Note cómo una menor congestión en vivienda/amenidades permite que las fuerzas de aglomeración ($\alpha = 0.08$) predominen, induciendo una marcada concentración geográfica de la población en los nodos centrales con mayor acceso al mercado.
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro` proporciona una suite unificada para el comercio internacional, la economía espacial cuantitativa y el modelado de equilibrio general:
# - `puremacro.trade.caliendo_parro`: Equilibrio general de comercio ricardiano multipaís y multisectorial con encadenamientos insumo-producto, elasticidades comerciales sectoriales, aranceles, desequilibrios comerciales y contrafactuales mediante álgebra de variaciones proporcionales exactas (Exact Hat Algebra) (`CaliendoParroModel`, `simulate_tariff_shock`, `simulate_trade_war`).
# - `puremacro.spatial.allen_arkolakis`: Equilibrio general espacial geográfico continuo con costos de transporte, amenidades, productividades y movilidad laboral (`AllenArkolakisModel`, `from_coordinates`, `simulate_infrastructure_shock`, `simulate_climate_shock`).
# - `puremacro.spatial.weights`: Matrices de pesos espaciales (reina, torre, decaimiento por distancia, distancia inversa y núcleo gaussiano) para econometría regional y modelos espaciales de corte transversal.
# - `puremacro.spatial.panel`: Estimadores para paneles espaciales autorregresivos (SAR) y de error espacial (SEM) con efectos fijos y cuasi-máxima verosimilitud.
# - `puremacro.spatial.hac`: Errores estándar espaciales HAC de Conley (1999) robustos a correlación espacial y temporal arbitraria en datos de panel y corte transversal.
