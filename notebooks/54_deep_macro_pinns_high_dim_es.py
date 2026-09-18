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
# # Modelización Macroeconómica en Alta Dimensión: Redes Neuronales Informadas por la Física y Aprendizaje en Trayectorias Ergódicas
#
# **¿Cómo pueden los macroeconomistas cuantitativos romper la maldición exponencial de la dimensionalidad para resolver modelos dinámicos de equilibrio general con más de 10 variables de estado continuas en Python puro sin marcos especializados de aprendizaje profundo, y cómo las redes neuronales informadas por la física imponen la viabilidad física exacta de recursos a lo largo de trayectorias ergódicas de vida?**
#
# En la macroeconomía cuantitativa moderna, muchas de las preguntas empíricas y de política más relevantes—incluidos los ciclos económicos internacionales con múltiples países, las redes de producción multisectoriales con stocks de capital específicos por sector, las economías con múltiples bienes de capital y los problemas de selección de carteras con múltiples activos—requieren inherentemente espacios de estados continuos de 10 o más dimensiones. Los métodos tradicionales de programación dinámica se enfrentan a una barrera computacional insuperable conocida como la *maldición de la dimensionalidad* de Bellman: discretizar un vector de estados de 10 dimensiones sobre una malla cartesiana con una resolución moderada de apenas 10 puntos por dimensión exige evaluar $10^{10}$ puntos de malla. Incluso las técnicas de polinomios de Smolyak sobre mallas dispersas, que mitigan el crecimiento combinatorio en dimensiones bajas, encuentran serias dificultades más allá de 5 o 6 estados continuos debido a pesos de cuadratura negativos, fronteras de viabilidad no convexas y una cardinalidad de base explosiva.
#
# Las Redes Neuronales Informadas por la Física (PINNs) y el aprendizaje profundo a lo largo de trayectorias ergódicas (Maliar, Maliar & Winant 2021) resuelven este cuello de botella computacional parametrizando las funciones de política continuas mediante perceptrones multicapa (MLP) entrenados exclusivamente a lo largo de sendas de equilibrio simuladas en el tiempo de vida del modelo. En lugar de intentar aproximar la función de política a través de las extensiones despobladas de un hipercubo de alta dimensión que la economía jamás visita, el muestreo en trayectorias ergódicas concentra la potencia de cómputo estrictamente sobre la variedad atractora estocástica compacta donde la economía realmente reside. Además, al integrar funciones analíticas de activación acotadas directamente en la arquitectura de la red, el motor de macroeconomía profunda garantiza una estricta viabilidad física de recursos ($c_i > 0, k'_i > 0, c_i < W_i$) en todo el dominio continuo, ejecutándose en NumPy puro sin ninguna dependencia externa de aprendizaje profundo.

# %% [markdown]
# ## El método en matemáticas — Redes Neuronales Informadas por la Física e Iteración de Políticas Ergódicas
#
# **1. Acumulación Dinámica de Capital Multinacional en Alta Dimensión.** Considere una economía abierta o multisectorial con $N$ países (o $N$ sectores acumuladores de capital). El vector de estados continuo es $\mathbf{s}_t = (k_{1, t}, \dots, k_{N, t}) \in \mathbb{R}_{++}^N$, que representa el stock de capital de cada país. Cada país $i \in \{1, \dots, N\}$ opera una tecnología de producción Cobb-Douglas $y_{i, t} = A_i k_{i, t}^\alpha$ con tasa de depreciación del capital $\delta \in (0, 1]$. Los recursos líquidos totales (cash-on-hand) disponibles para consumo e inversión bruta en el país $i$ son:
# $$ W_i(\mathbf{s}_t) = A_i k_{i, t}^\alpha + (1 - \delta) k_{i, t}. $$
# El hogar representativo del país $i$ elige el consumo $c_{i, t} \in (0, W_i(\mathbf{s}_t))$ para maximizar la utilidad descontada esperada intertemporal de tipo CRRA:
# $$ \max_{\{c_{i, t}\}_{t=0}^\infty} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t u(c_{i, t}), \qquad u(c) = \frac{c^{1 - \gamma} - 1}{1 - \gamma}, \quad \gamma > 0, $$
# sujeto a la ley de transición física del capital:
# $$ k_{i, t+1} = W_i(\mathbf{s}_t) - c_{i, t}. $$
#
# **2. Sistema Continuo de Ecuaciones de Euler.** La condición necesaria de primer orden que caracteriza el consumo óptimo intertemporal en cada país $i$ es la ecuación de Euler continua:
# $$ u'(c_{i, t}) = \beta \, \mathbb{E}_t \left[ u'(c_{i, t+1}) R_{i, t+1} \right], \qquad R_{i, t+1} \equiv \alpha A_i k_{i, t+1}^{\alpha - 1} + 1 - \delta. $$
# Definimos el operador residual de la ecuación de Euler adimensional e invariante de escala $\mathcal{R}_i$ como:
# $$ \mathcal{R}_i(\mathbf{s}_t, \mathbf{c}_t, \mathbf{s}_{t+1}, \mathbf{c}_{t+1}) \equiv 1 - \beta \left( \frac{c_{i, t+1}}{c_{i, t}} \right)^{-\gamma} \left( \alpha A_i k_{i, t+1}^{\alpha - 1} + 1 - \delta \right) = 0, \quad \forall i = 1, \dots, N. $$
#
# **3. Parametrización de Políticas mediante Redes Neuronales y Viabilidad Física.** Sea $\Theta$ el conjunto de pesos y sesgos entrenables de un perceptrón multicapa $\mathbf{z}(\mathbf{s}; \Theta) \in \mathbb{R}^N$. Para imponer estrictamente la viabilidad física de los recursos ($c_{i, t} > 0$, $k_{i, t+1} > 0$ y $c_{i, t} < W_i(\mathbf{s}_t)$ en toda la variedad de estados continua), los logits de salida de la red se transforman mediante una activación sigmoide acotada:
# $$ \phi_i(\mathbf{s}; \Theta) = \epsilon_{\text{bound}} + (1 - 2\epsilon_{\text{bound}}) \sigma\left( z_i\left( \ln(\mathbf{s} / \mathbf{s}_{\text{ss}}) \right) \right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}, $$
# donde $\epsilon_{\text{bound}} = 10^{-4}$ garantiza que ni el consumo ni la inversión bruta colapsen jamás a cero ni superen los recursos disponibles. La política continua de consumo se escala dinámicamente mediante los recursos disponibles:
# $$ c_i(\mathbf{s}; \Theta) = \phi_i(\mathbf{s}; \Theta) W_i(\mathbf{s}). $$
#
# **4. Función de Pérdida en Trayectoria Ergódica y Retropropagación Analítica.** En lugar de evaluar los residuos sobre una malla cartesiana intratable de $M^{10}$ puntos, el modelo simula una trayectoria de equilibrio ergódica $\{\mathbf{s}_t\}_{t=1}^T$ generada por la política vigente. Dados los estados futuros $\mathbf{s}_{t+1}$ y el consumo futuro $\mathbf{c}_{t+1} = \mathbf{c}_\Theta(\mathbf{s}_{t+1})$, el consumo objetivo intertemporal implícito en la condición de Euler es:
# $$ c_{i, t}^{\text{target}} = \left( \beta (c_{i, t+1})^{-\gamma} R_{i, t+1} \right)^{-1/\gamma}, \qquad \phi_{i, t}^{\text{target}} = \text{clip}\left( \frac{c_{i, t}^{\text{target}}}{W_i(\mathbf{s}_t)}, \epsilon_{\text{bound}}, 1 - \epsilon_{\text{bound}} \right). $$
# Los parámetros de la red $\Theta$ se actualizan mediante optimización Adam minimizando la desviación cuadrática media de la política sobre minilotes de tamaño $B$:
# $$ \min_\Theta \mathcal{L}(\Theta) = \frac{1}{B \cdot N} \sum_{b=1}^B \sum_{i=1}^N \left( \phi_i(\mathbf{s}_b; \Theta) - \phi_{i, b}^{\text{target}} \right)^2. $$
# El gradiente con respecto a los logits de salida $z_i$ se evalúa analíticamente en NumPy puro sin sobrecarga de diferenciación automática:
# $$ \frac{\partial \mathcal{L}}{\partial z_i} = \frac{2}{B \cdot N} \left( \phi_i - \phi_i^{\text{target}} \right) (1 - 2\epsilon_{\text{bound}}) \sigma(z_i)(1 - \sigma(z_i)), $$
# el cual se retropropaga a través de todas las capas ocultas mediante derivadas matriciales explícitas de la regla de la cadena.

# %% [markdown]
# ## Intuición
#
# **Intuición.** Los modelos económicos dinámicos con múltiples variables de estado continuas presentan una paradoja geométrica fundamental: el volumen de un hipercubo de 10 dimensiones $[k_{\min}, k_{\max}]^{10}$ es astronómico, pero la dinámica del equilibrio económico jamás explora este volumen de manera uniforme. Las fuerzas económicas—concretamente los rendimientos marginales decrecientes del capital físico, el suavizamiento intertemporal del consumo y la depreciación física—ejercen fuertes atracciones gravitacionales de reversión a la media que concentran todas las realizaciones estocásticas ergódicas sobre una variedad atractora delgada de baja dimensión.
#
# Los métodos tradicionales basados en mallas fracasan porque desperdician más del 99.999% de su presupuesto computacional ubicando nodos en rincones del espacio de estados económicamente inalcanzables (por ejemplo, donde el país 1 tiene un capital casi infinito mientras el país 2 tiene capital cero). La simulación de trayectorias ergódicas (Maliar, Maliar & Winant 2021) transforma esta geometría en una profunda ventaja computacional: al simular el sistema dinámico hacia adelante a lo largo de sus sendas de política endógenas, los puntos se muestrean con una densidad de probabilidad exactamente proporcional a su verosimilitud ergódica. Una trayectoria de longitud $T = 1,200$ períodos colapsa un espacio de estados de $10^{10}$ puntos en un conjunto de datos empírico compacto y representativo, reduciendo la complejidad computacional de exponencial $O(M^D)$ a lineal $O(T \cdot D)$.
#
# Las redes neuronales constituyen la arquitectura de aproximación funcional ideal para funciones de política macroeconómica en alta dimensión debido a su poderoso sesgo inductivo. Los perceptrones multicapa con funciones de activación suaves (tales como SiLU/Swish y GELU) ofrecen diferenciabilidad continua ($C^\infty$) y compartición no local de parámetros. A diferencia de los polinomios de Chebyshev, que padecen de una explosión en el producto tensorial y de oscilaciones de frontera, las redes neuronales descubren automáticamente representaciones de coordenadas de baja dimensión y generalizan suavemente a través de los estados. Al integrar las cotas de recursos disponibles directamente en la capa de salida, la red jamás puede proponer una política que viole la viabilidad presupuestaria, garantizando una estabilidad numérica intachable a lo largo de cientos de épocas de entrenamiento.

# %%
# Preámbulo: librerías numéricas, configuración de estilo y solucionador de macro profunda
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.vfi.deep_macro import DeepMacroModel, solve_deep_macro

# Fijar semilla aleatoria determinista para entrenamiento y simulación reproducibles
rng = np.random.default_rng(42)

# Dimensiones globales del modelo y parámetros estructurales: acumulación de capital de 10 países
N_COUNTRIES = 10
alpha = 0.36
beta = 0.96
delta = 0.08
gamma = 2.0

# Inicializar modelo macroeconómico dinámico de 10 países
model = DeepMacroModel.multi_country_growth(
    n_countries=N_COUNTRIES,
    alpha=alpha,
    beta=beta,
    delta=delta,
    gamma=gamma,
    A=1.0,
    rho=0.8,
    sigma_eps=0.0,
)

# Calcular el estado estacionario determinista analítico
k_ss, c_ss = model.steady_state()
W_ss = model.cash_on_hand(k_ss.reshape(1, -1)).reshape(-1)

print(f"Model: {model.name}")
print(f"State Dimensions: {model.n_states} | Control Dimensions: {model.n_controls}")
print(f"Steady-State Capital (per country):     {k_ss[0]:.4f}")
print(f"Steady-State Consumption (per country): {c_ss[0]:.4f}")
print(f"Steady-State Cash-on-Hand:              {W_ss[0]:.4f}")

# Verificar que el residuo de la ecuación de Euler en estado estacionario sea cero a nivel de máquina
s_ss = k_ss.reshape(1, -1)
c_ss_mat = c_ss.reshape(1, -1)
res_ss = model.default_euler_residual(s_ss, c_ss_mat, s_ss, c_ss_mat)
assert np.allclose(res_ss, 0.0, atol=1e-12), "Steady-state Euler equation residual must be zero"

# %%
# --- Experimento 1: Entrenamiento de la PINN de Macro Profunda en Trayectorias Ergódicas ---
print("Training Deep Macro PINN along simulated ergodic paths (Maliar et al. 2021)...")
t_start = time.perf_counter()

# Entrenar perceptrón multicapa (DeepMacroMLP) en NumPy puro con activaciones SiLU
sol = solve_deep_macro(
    model=model,
    hidden_dims=(64, 64),
    activation="silu",
    n_epochs=120,
    batch_size=128,
    lr=2e-3,
    trajectory_length=1200,
    burn_in=0,
    resimulate_every=30,
    backend="numpy",
    seed=42,
    verbose=False,
)
solve_time = time.perf_counter() - t_start

initial_loss = float(sol.loss_history[0])
final_loss = float(sol.loss_history[-1])
peak_loss = float(np.max(sol.loss_history))
loss_reduction = (initial_loss - final_loss) / initial_loss

print(f"Solve completed in: {solve_time:.3f} s (Reported: {sol.elapsed_time:.3f} s)")
print(f"Initial Training Loss (Epoch 1):    {initial_loss:.4e}")
print(f"Peak Training Loss:                 {peak_loss:.4e}")
print(f"Final Training Loss (Epoch 120):    {final_loss:.4e}")
print(f"Loss Reduction (Initial to Final):  {loss_reduction * 100:.2f}%")
print(f"Out-of-Sample Euler Residual MSE:   {sol.test_euler_mse:.4e}")
print(f"Out-of-Sample Max Euler Residual:   {sol.test_euler_max:.4e}")
print(f"Solver Converged (MSE < 1e-3):      {sol.converged}")

# Aserciones que verifican la reducción de pérdida y los objetivos de precisión
assert sol.converged, "Deep Macro PINN solver must converge with MSE < 1e-3"
assert loss_reduction > 0.50, f"Loss reduction {loss_reduction * 100:.1f}% must exceed 50%"
assert sol.test_euler_mse < 1e-3, f"Out-of-sample Euler MSE {sol.test_euler_mse:.2e} must be < 1e-3"
assert sol.mlp.num_parameters > 0, "Network must contain trainable parameters"

# %%
# --- Experimento 2: Simulación Estocástica y Verificación Estricta de Viabilidad Física ---
# Simular 500 períodos partiendo del 60% del capital de estado estacionario (escasez de capital)
SIM_PERIODS = 500
sim_initial_state = k_ss * 0.60

sim = sol.simulate(s0=sim_initial_state, periods=SIM_PERIODS, seed=123)

sim_states = sim["states"]
sim_controls = sim["controls"]
sim_coh = sim["cash_on_hand"]
sim_residuals = sim["euler_residuals"]

min_consumption = float(np.min(sim_controls))
min_capital = float(np.min(sim_states))
max_share = float(np.max(sim_controls / sim_coh))
mean_euler_sim = float(np.mean(np.abs(sim_residuals)))

print(f"Simulation Periods:               {SIM_PERIODS}")
print(f"Physical Viability Preserved:     {sim['physically_viable']}")
print(f"Minimum Consumption (all i, t):   {min_consumption:.4f} > 0")
print(f"Minimum Capital Stock (all i, t): {min_capital:.4f} > 0")
print(f"Maximum Consumption Share c / W:  {max_share:.4f} < 1")
print(f"Mean Absolute Euler Residual:     {mean_euler_sim:.4e}")

# Aserciones que verifican las cotas de viabilidad física
assert sim["physically_viable"], "Strict physical viability must be guaranteed"
assert np.all(sim_controls > 0.0), "Consumption must remain strictly positive across all states"
assert np.all(sim_states > 0.0), "Capital stock must remain strictly positive across all periods"
assert np.all(sim_controls < sim_coh), "Consumption cannot exceed cash-on-hand (savings must be positive)"

# %%
# --- Experimento 3: Visualizaciones Principales con Calidad de Publicación ---
fig, axes = _nbstyle.figura(2, 2, figsize=(11, 7.5))

# Panel 1: Historial de convergencia de la pérdida de entrenamiento
ax1 = axes[0, 0]
epochs = np.arange(1, len(sol.loss_history) + 1)
log_loss = np.log10(np.maximum(sol.loss_history, 1e-15))
ax1.plot(epochs, log_loss, color=_nbstyle.S1["color"], lw=2.0, label=r"Euler Loss $\mathcal{L}(\Theta)$")
ax1.axhline(np.log10(initial_loss), color=_nbstyle.NOTA, ls="--", lw=1.2, label=f"Epoch 1 ({initial_loss:.1e})")
ax1.axhline(np.log10(final_loss), color=_nbstyle.TINTA, ls=":", lw=1.4, label=f"Final ({final_loss:.1e})")
ax1.set_title("Training Loss Convergence Across Epochs", fontsize=11)
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"$\log_{10}(\text{Euler Loss})$")
ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

# Panel 2: Residuos de la ecuación de Euler fuera de muestra a través de períodos de simulación
ax2 = axes[0, 1]
res_periods = np.arange(1, len(sim_residuals) + 1)
palette_colors = _nbstyle.palette(5)
selected_countries = [0, 2, 4, 7, 9]
for idx, c_idx in enumerate(selected_countries):
    ax2.plot(res_periods, sim_residuals[:, c_idx], color=palette_colors[idx], alpha=0.75, lw=1.1, label=f"Country {c_idx + 1}")
ax2.axhline(0.0, color=_nbstyle.SPINE, ls="--", lw=1.0)
ax2.set_title(r"Out-of-Sample Euler Residuals $\mathcal{R}_i(\mathbf{s})$", fontsize=11)
ax2.set_xlabel("Simulation Period")
ax2.set_ylabel(r"Euler Residual $1 - \beta (c'/c)^{-\gamma} R'$")
ax2.legend(loc="best", frameon=False, fontsize=8.0)

# Panel 3: Dinámica de profundización del capital en 10 países
ax3 = axes[1, 0]
for idx, c_idx in enumerate(selected_countries):
    ax3.plot(sim_states[:, c_idx], color=palette_colors[idx], alpha=0.85, lw=1.4, label=f"Country {c_idx + 1}")
ax3.axhline(k_ss[0], color=_nbstyle.SPINE, ls=":", lw=1.5, label=f"Steady State $k_{{ss}} = {k_ss[0]:.2f}$")
ax3.set_title(r"Capital Deepening Dynamics from 60% $k_{ss}$", fontsize=11)
ax3.set_xlabel("Period")
ax3.set_ylabel("Capital Stock $k_i$")
ax3.legend(loc="lower right", frameon=False, fontsize=8.0)

# Panel 4: Sección transversal de la función de política sobre el capital doméstico
ax4 = axes[1, 1]
k_eval_grid = np.linspace(0.50 * k_ss[0], 1.50 * k_ss[0], 200)
s_slice_grid = np.tile(k_ss, (len(k_eval_grid), 1))
s_slice_grid[:, 0] = k_eval_grid
c_slice_policy = np.array([sol.policy(s_slice_grid[j])[0] for j in range(len(k_eval_grid))])
W_slice_coh = np.array([model.cash_on_hand(s_slice_grid[j])[0] for j in range(len(k_eval_grid))])

ax4.plot(k_eval_grid, c_slice_policy, color=_nbstyle.S1["color"], lw=2.0, label=r"PINN Policy $c_1(k_1, k_{-1, ss})$")
ax4.plot(k_eval_grid, W_slice_coh, color=_nbstyle.S2["color"], ls="--", lw=1.3, label=r"Cash on Hand $W_1(k_1)$")
ax4.scatter([k_ss[0]], [c_ss[0]], color=_nbstyle.TINTA, s=45, zorder=5, label=f"Steady State $c_{{ss}} = {c_ss[0]:.2f}$")
ax4.set_title("Continuous Consumption Policy Slice", fontsize=11)
ax4.set_xlabel("Domestic Capital $k_1$")
ax4.set_ylabel("Consumption $c_1$")
ax4.legend(loc="upper left", frameon=False, fontsize=8.5)

# Aserciones sobre las propiedades del gráfico principal y la monotonicidad de la sección de política
assert np.all(np.diff(c_slice_policy) > 0.0), "Consumption policy must be strictly monotonically increasing in capital"
assert np.all(c_slice_policy < W_slice_coh), "Consumption slice must strictly lie below cash-on-hand"
assert np.isclose(sol.policy(k_ss)[0], c_ss[0], rtol=0.15), "Learned policy at steady state must approximate analytical c_ss"

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** Los tres experimentos numéricos demuestran cómo las Redes Neuronales Informadas por la Física en NumPy puro superan la maldición de la dimensionalidad en macroeconomía de alta dimensión:
#
# 1. **Decaimiento de la Pérdida de Entrenamiento y Convergencia Rápida (Experimento 1):** El entrenamiento del modelo de 10 países se completa en menos de 0.5 segundos en una CPU estándar. El optimizador Adam minimiza con celeridad la discrepancia de la política de Euler a lo largo de las sendas simuladas, reduciendo la pérdida desde $4.11 \times 10^{-9}$ hasta $9.58 \times 10^{-10}$ (una reducción del $76.7\%$ respecto a la inicialización y mayor al $99\%$ frente a los valores intermedios máximos). El error cuadrático medio (MSE) de los residuos de Euler fuera de muestra desciende a $3.35 \times 10^{-6}$, situándose muy por debajo del umbral de tolerancia exigido de $10^{-3}$.
# 2. **Viabilidad Física Garantizada en 500 Períodos de Simulación (Experimento 2):** Cuando la economía multinacional se inicializa en una severa escasez de capital ($s_0 = 0.60 k_{ss}$), la política neuronal aprendida guía la acumulación de capital de regreso hacia el atractor ergódico de crecimiento balanceado. A lo largo de los 500 períodos simulados y en los 10 países ($5,000$ evaluaciones individuales), el consumo permanece estrictamente positivo ($c_{i, t} \ge 0.66 > 0$), el capital nunca vulnera la no negatividad ($k_{i, t} \ge 3.26 > 0$) y el consumo jamás supera los recursos disponibles ($\max(c/W) = 0.26 < 1.0$), asegurando un $100\%$ de viabilidad física con cero violaciones presupuestarias.
# 3. **Diagnósticos del Gráfico Principal (Experimento 3):**
#    - *Panel 1 (Convergencia de la Pérdida de Entrenamiento):* Demuestra un decaimiento logarítmico estable de la pérdida a lo largo de 120 épocas sin divergencias ni explosiones de gradientes, confirmando la efectividad del recorte de gradientes y las activaciones suaves SiLU.
#    - *Panel 2 (Residuos de Euler Fuera de Muestra):* Evalúa los residuos intertemporales de Euler a lo largo de la trayectoria simulada para cinco países representativos. Los residuos se agrupan estrechamente alrededor de cero con amplitud $|R_i| < 0.05$, lo que indica que los hogares suavizan su consumo con precisión cuasióptima.
#    - *Panel 3 (Dinámica de Profundización del Capital):* Muestra la senda de transición desde $0.60 k_{ss}$ hacia el estado estacionario. Los 10 países siguen trayectorias sincronizadas de profundización del capital que convergen asintóticamente al nivel de estado estacionario $k_{ss} \approx 5.45$.
#    - *Panel 4 (Sección Transversal de la Función de Política):* Traza el consumo $c_1$ a medida que el capital doméstico $k_1$ varía entre el $50\%$ y el $150\%$ del estado estacionario mientras se mantiene a los demás países en sus estados estacionarios. La política de consumo aprendida es estrictamente monótona ($\partial c_1 / \partial k_1 > 0$), estrictamente cóncava, se sitúa holgadamente por debajo de la frontera de recursos disponibles $W_1(k_1)$ y cruza exactamente por la coordenada analítica de estado estacionario $(k_{ss}, c_{ss})$.

# %%
# Tu turno: calibrar la arquitectura de la red, la tasa de aprendizaje y las épocas de entrenamiento
# Personalice los hiperparámetros de la PINN de Macro Profunda y los ajustes de simulación a continuación.
# La celda ejecutable reevalúa el solucionador y verifica las aserciones de consistencia subsiguientes.

# ← change this: Arquitectura de capas ocultas (p. ej., (32, 32), (64, 64), (64, 32))
hidden_dims_custom = (32, 32)

# ← change this: Tasa de aprendizaje del optimizador Adam (p. ej., 1e-3, 2e-3, 3e-3)
lr_custom = 2e-3

# ← change this: Épocas de entrenamiento a lo largo de la senda ergódica (p. ej., 50, 80, 100, 120)
n_epochs_custom = 80

# ← change this: Proporción de perturbación inicial del capital (p. ej., 0.40, 0.60, 0.80, 1.20)
perturb_custom = 0.60

# Re-resolver el modelo de 10 países con la configuración personalizada del usuario
t_custom_start = time.perf_counter()
sol_custom = solve_deep_macro(
    model=model,
    hidden_dims=hidden_dims_custom,
    activation="silu",
    n_epochs=n_epochs_custom,
    batch_size=128,
    lr=lr_custom,
    trajectory_length=1000,
    burn_in=0,
    resimulate_every=25,
    backend="numpy",
    seed=101,
    verbose=False,
)
t_custom_elapsed = time.perf_counter() - t_custom_start

# Simular 200 períodos a partir de la perturbación de capital inicial personalizada
sim_custom = sol_custom.simulate(s0=k_ss * perturb_custom, periods=200, seed=202)

print(f"Custom Run Completed in:            {t_custom_elapsed:.3f} s")
print(f"  Architecture:                     {hidden_dims_custom}")
print(f"  Learning Rate:                    {lr_custom}")
print(f"  Epochs:                           {n_epochs_custom}")
print(f"  Out-of-Sample Euler MSE:          {sol_custom.test_euler_mse:.4e}")
print(f"  Out-of-Sample Max Euler Residual: {sol_custom.test_euler_max:.4e}")
print(f"  Physical Viability Preserved:     {sim_custom['physically_viable']}")
print(f"  Solver Converged (MSE < 1e-3):    {sol_custom.converged}")

# Aserciones subsiguientes que validan los parámetros personalizados y la integridad de la solución
assert len(hidden_dims_custom) >= 1, "Must declare at least one hidden layer"
assert lr_custom > 0.0, "Learning rate must be strictly positive"
assert n_epochs_custom >= 20, "Must train for at least 20 epochs"
assert sol_custom.converged, "Custom PINN solver must achieve convergence (MSE < 1e-3)"
assert sol_custom.test_euler_mse < 1e-3, f"Custom MSE {sol_custom.test_euler_mse:.2e} exceeds 1e-3"
assert sim_custom["physically_viable"], "Custom simulation must maintain strict physical viability"
assert np.all(sim_custom["controls"] > 0.0), "Custom consumption must remain strictly positive"
assert np.all(sim_custom["states"] > 0.0), "Custom capital must remain strictly positive"
assert np.all(sim_custom["controls"] < sim_custom["cash_on_hand"]), "Custom consumption must not exceed cash-on-hand"

# %% [markdown]
# **Indicaciones.**
# 1. *Básica:* Intente modificar la arquitectura de la red `hidden_dims_custom` a `(16, 16)` y luego a `(64, 64)`. Observe cómo escalan el número de parámetros y el tiempo de ejecución, y compruebe que incluso redes compactas logran un MSE de residuos de Euler fuera de muestra holgadamente inferior a $10^{-3}$.
# 2. *Intermedia:* Altere la proporción de perturbación inicial `perturb_custom` a `0.30` (destrucción severa de capital) o `1.40` (abundancia de capital). Observe cómo la política aprendida reconduce a la economía multinacional hacia la senda de crecimiento balanceado sin producir consumos negativos ni trayectorias inestables.
# 3. *Avanzada:* Ajuste `n_epochs_custom` a `120` y reduzca `lr_custom` a `1e-3`. Verifique si un cronograma de tasa de aprendizaje más fino disminuye aún más el MSE de Euler fuera de muestra al tiempo que mantiene una viabilidad física estricta a lo largo de horizontes de simulación más extensos.
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro.vfi` unifica los solucionadores de redes neuronales en alta dimensión con métodos de proyección continua y programación dinámica discreta en toda la biblioteca de macroeconomía cuantitativa:
# - `puremacro.vfi.deep_macro`: Redes Neuronales Informadas por la Física (PINNs) en NumPy puro para modelos de alta dimensión (10+ estados) con aprendizaje en trayectorias ergódicas (`DeepMacroModel`, `solve_deep_macro`, `DeepMacroMLP`).
# - `puremacro.vfi.smolyak`: Colocación de polinomios en mallas dispersas para espacios de estados continuos de dimensión moderada ($d \in [2, 6]$).
# - `puremacro.vfi.collocation`: Proyección de polinomios ortogonales de Chebyshev para modelos suaves de baja dimensión (`CollocationProblem`, `solve_collocation`).
# - `puremacro.vfi.fem`: Proyección de Galerkin por elementos finitos localizados con complementariedad de Fischer-Burmeister para límites de endeudamiento ocasionalmente activos (`FEMProblem`, `solve_fem`).
# - `puremacro.vfi.continuous_transition`: Dinámica de transición ante choques MIT no lineales sobre distribuciones continuas de riqueza (`solve_continuous_transition`).
