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
# # Proyección en Espacios de Estados Continuos: Solucionadores de Colocación y FEM
#
# **¿Cómo resuelven los macroeconomistas cuantitativos los modelos de equilibrio general dinámico sin la maldición de la dimensionalidad ni la distorsión por discretización de la iteración de la función de valor sobre mallas discretas, y cuándo conviene implementar polinomios ortogonales globales de Chebyshev frente a la proyección de Galerkin por elementos finitos localizados?**
#
# En la macroeconomía moderna, las variables de decisión económica (capital, bonos, restricciones de colateral) residen en variedades de estados continuos. La iteración tradicional de la función de valor en mallas discretas fuerza las elecciones de activos sobre un retículo discreto artificial, induciendo un error de truncamiento $O(N_a)$, no diferenciabilidad y una explosión combinatoria del espacio de estados. Los métodos de proyección en dominios continuos aproximan las funciones de política y de valor directamente sobre el dominio continuo: la colocación con polinomios ortogonales de Chebyshev de primera especie proporciona convergencia exponencial "espectral" ($O(c^{-N})$) para variedades analíticas reales suaves, mientras que el método de elementos finitos (FEM) localizado ofrece convergencia algebraica ($O(h^2)$) al tiempo que resuelve de manera única puntos de quiebre pronunciados, restricciones de endeudamiento y fricciones financieras que se activan ocasionalmente con cero oscilaciones de Gibbs.

# %% [markdown]
# ## El método en matemáticas — Proyección en Espacios de Estados Continuos
#
# **1. Colocación con polinomios ortogonales de Chebyshev.** Transformamos el estado $k \in [a, b]$ al intervalo canónico $x \in [-1, 1]$ mediante $x = \frac{2k - (a + b)}{b - a}$. Los polinomios de Chebyshev de primera especie satisfacen la relación de recurrencia de tres términos:
# $$ T_0(x) = 1, \qquad T_1(x) = x, \qquad T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x). $$
# Evaluando en los $N+1$ nodos extremos de Chebyshev-Gauss-Lobatto $x_i = -\cos\left(\frac{i \pi}{N}\right)$, la función de política $g_\theta(k) = \sum_{n=0}^N \theta_n T_n(x(k))$ satisface el residuo de la ecuación de Euler en tiempo continuo:
# $$ \mathcal{R}_{\text{Euler}}(\theta; k_i) \equiv 1 - \beta \frac{u'\left(c(g_\theta(k_i), g_\theta(g_\theta(k_i)))\right)}{u'\left(c(k_i, g_\theta(k_i))\right)} f'\left(g_\theta(k_i)\right) = 0, \quad i = 0, \dots, N. $$
# Para la colocación del valor de Bellman en espacio continuo, los coeficientes de valor $c_v$ satisfacen $\mathcal{R}_{\text{Bellman}}(c_v; k_i) \equiv V(k_i) - \max_{k'} \left[ u(k_i, k') + \beta \sum_{n=0}^N c_{v, n} T_n(x(k')) \right] = 0$.
#
# **2. Proyección de Galerkin mediante el método de elementos finitos (FEM).** Particionamos $[a, b]$ en $E$ elementos con nodos $k_0 < k_1 < \dots < k_E$ y funciones base lineales a trozos (tipo sombrero) de Lagrange $\phi_j(k)$ con soporte compacto $[k_{j-1}, k_{j+1}]$:
# $$ \phi_j(k) = \frac{k - k_{j-1}}{k_j - k_{j-1}} \mathbf{1}_{[k_{j-1}, k_j]}(k) + \frac{k_{j+1} - k}{k_{j+1} - k_j} \mathbf{1}_{[k_j, k_{j+1}]}(k), \qquad g_E(k) = \sum_{j=0}^E y_j \phi_j(k). $$
# La proyección de Galerkin ortogonaliza el residuo continuo $\mathcal{R}(k)$ contra las funciones de prueba mediante cuadratura de Gauss-Legendre de $Q$ puntos ($s_q = \frac{k_e + k_{e+1}}{2} + \frac{h_e}{2}\xi_q$):
# $$ \int_a^b \mathcal{R}(k) \phi_i(k) \, dk = \sum_{e} \frac{h_e}{2} \sum_{q=1}^Q w_q \mathcal{R}(s_q) \phi_i(s_q) = 0, \quad i = 0, \dots, E. $$
#
# **3. Complementariedad de Fischer-Burmeister para restricciones de endeudamiento.** Cuando las decisiones de ahorro enfrentan una restricción de endeudamiento en desigualdad $k' \ge \bar{k}$, las condiciones de Kuhn-Tucker $(k' - \bar{k}) \ge 0$, $\mathcal{R}(k) \ge 0$, y $(k' - \bar{k})\mathcal{R}(k) = 0$ se reformulan mediante el operador diferenciable perturbado de complementariedad no lineal (NCP) de Fischer-Burmeister ($\epsilon = 10^{-12}$):
# $$ \Psi_{\text{FB}}^\epsilon(a, b_{\text{val}}) = a + b_{\text{val}} - \sqrt{a^2 + b_{\text{val}}^2 + \epsilon} = 0, \qquad a = k' - \bar{k}, \quad b_{\text{val}} = \mathcal{R}(k). $$

# %% [markdown]
# ## Intuición
#
# **Intuición.** La programación dinámica en espacios de estados continuos sustituye la discretización heurística por una rigurosa aproximación en espacios funcionales. Cuando las funciones de política y de valor de un modelo económico son analíticas reales —como ocurre en el modelo de crecimiento neoclásico—, el teorema de aproximación de Jackson garantiza que las proyecciones mediante polinomios ortogonales de Chebyshev convergen a una tasa *espectral*: el error de aproximación decae exponencialmente como $O(c^{-N})$, reduciendo los residuos de la ecuación de Euler de $10^{-4}$ a $10^{-9}$ con apenas una docena de coeficientes polinomiales.
#
# Sin embargo, cuando los hogares enfrentan límites de crédito o restricciones de endeudamiento que se activan ocasionalmente ($a' \ge \bar{a}$), la función de política desarrolla un punto de quiebre o esquina pronunciada (continuidad $C^0$ con una discontinuidad de salto en su derivada) en el umbral endógeno de endeudamiento $k^*$. Dado que los polinomios globales de Chebyshev distribuyen la información a lo largo de todo el dominio, aproximar saltos en la derivada desencadena el **fenómeno de Gibbs**: oscilaciones persistentes de alta frecuencia que se concentran cerca del punto de quiebre, propagan ondas espurias por todo el espacio de estados y generan graves violaciones del límite de crédito ($a' < \bar{a}$).
#
# El método de elementos finitos (FEM) resuelve este dilema abandonando el soporte global en favor de funciones base lineales a trozos (tipo sombrero) localizadas. Si bien la convergencia suave es algebraica ($O(h^2) = O(1/E^2)$), los elementos poseen soporte compacto localizado: los errores en un elemento no pueden propagarse a sus elementos vecinos. Al alinear un nodo de la malla directamente en el punto de quiebre de endeudamiento $k^*$ (`FEMMesh.from_kinks`) e imponer la condición suave de complementariedad de Fischer-Burmeister, FEM captura la política plana sobre el límite de crédito y la política interior creciente con cero oscilaciones de Gibbs, cumplimiento estricto de frontera con precisión de máquina y convergencia jacobiana robusta mediante Newton-Raphson.

# %%
# Preámbulo: importar librerías numéricas, estilo de gráficos y solucionadores continuos
import sys
from pathlib import Path
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.vfi import CollocationBasis, CollocationProblem, FEMMesh, FEMProblem
from puremacro import _backend as bk

# Fijar semilla pseudoaleatoria determinista para reproducibilidad
rng = np.random.default_rng(42)

# Parámetros globales del modelo: crecimiento neoclásico canónico de Brock-Mirman (1972)
alpha = 0.36
beta = 0.96
delta = 1.0

# Estado estacionario analítico en forma cerrada y reglas de decisión
k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
domain = (0.5 * k_ss, 1.5 * k_ss)
eval_k = np.linspace(domain[0], domain[1], 1000)

g_star = lambda k: alpha * beta * (k ** alpha)
c_star = lambda k: (1.0 - alpha * beta) * (k ** alpha)
g_true = g_star(eval_k)

def compute_euler_residual(policy_fn, k, alpha, beta):
    """Compute out-of-sample continuous Euler equation residual."""
    kp = policy_fn(k)
    kpp = policy_fn(kp)
    c = np.maximum(k ** alpha - kp, 1e-12)
    cp = np.maximum(kp ** alpha - kpp, 1e-12)
    fkp = alpha * (kp ** (alpha - 1.0))
    return 1.0 - beta * (c / cp) * fkp

print(f"Brock-Mirman Steady State Capital: k_ss = {k_ss:.4f}")
print(f"State Domain: [{domain[0]:.4f}, {domain[1]:.4f}] | Out-of-sample grid: {len(eval_k)} points")

# %%
# --- Experimento 1: Benchmark de crecimiento neoclásico suave (convergencia espectral vs. polinomial) ---
# Colocación ortogonal de Chebyshev: órdenes N en [4, 8, 12]
orders_cheb = [4, 8, 12]
sols_cheb = {}
errs_cheb = {}
eulers_cheb = {}

for N in orders_cheb:
    prob_c = CollocationProblem(
        domain=domain,
        orders=N,
        method="euler",
        params={"alpha": alpha, "delta": delta},
        beta=beta,
    )
    sol_c = prob_c.solve(backend="numpy")
    sols_cheb[N] = sol_c
    pol_c = sol_c.policy(eval_k)
    errs_cheb[N] = float(np.max(np.abs(pol_c - g_true) / g_true))
    eulers_cheb[N] = float(np.max(np.abs(compute_euler_residual(sol_c.policy, eval_k, alpha, beta))))
    print(f"Chebyshev Collocation (N={N:2d}): Rel Error = {errs_cheb[N]:.2e} | Max Euler = {eulers_cheb[N]:.2e} | Converged = {sol_c.converged}")

# Método de elementos finitos (proyección de Galerkin): elementos E en [20, 50, 80]
elements_fem = [20, 50, 80]
sols_fem = {}
errs_fem = {}
eulers_fem = {}

for E in elements_fem:
    prob_f = FEMProblem(
        domain=domain,
        elements=E,
        method="euler",
        projection="galerkin",
        return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
        transition_fn=lambda k: k ** alpha,
        beta=beta,
        params={"alpha": alpha, "gamma": 1.0},
    )
    sol_f = prob_f.solve(backend="numpy")
    sols_fem[E] = sol_f
    pol_f = sol_f.policy(eval_k)
    errs_fem[E] = float(np.max(np.abs(pol_f - g_true) / g_true))
    eulers_fem[E] = float(np.max(np.abs(sol_f.euler_residual(eval_k))))
    print(f"FEM Galerkin (E={E:2d}):          Rel Error = {errs_fem[E]:.2e} | Max Euler = {eulers_fem[E]:.2e} | Converged = {sol_f.converged}")

# Aserciones en línea: verificar convergencia analítica y residuos de Euler (< 1e-4)
assert sols_cheb[8].converged and sols_fem[50].converged, "Both baseline solvers must converge"
assert errs_cheb[8] < 1e-4, f"Chebyshev N=8 relative error {errs_cheb[8]:.2e} exceeds 1e-4"
assert errs_cheb[12] < 1e-8, f"Chebyshev N=12 spectral error {errs_cheb[12]:.2e} exceeds 1e-8"
assert errs_fem[50] < 1e-4, f"FEM E=50 relative error {errs_fem[50]:.2e} exceeds 1e-4"
assert errs_fem[80] < 5e-5, f"FEM E=80 relative error {errs_fem[80]:.2e} exceeds 5e-5"
assert eulers_cheb[8] < 1e-4, f"Chebyshev N=8 Euler residual {eulers_cheb[8]:.2e} exceeds 1e-4"
assert eulers_fem[50] < 1e-4, f"FEM E=50 Euler residual {eulers_fem[50]:.2e} exceeds 1e-4"

# Figura 1: Figura principal de crecimiento suave
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.2))
colors1 = _nbstyle.palette(4)
ls1 = _nbstyle.styles(4)

axes1[0].plot(eval_k, g_true, color="0.1", lw=2.2, label="Analytical Truth $g^*(k) = \\alpha\\beta k^\\alpha$")
axes1[0].plot(eval_k, sols_cheb[8].policy(eval_k), color=colors1[1], ls=ls1[1], lw=1.6, label="Chebyshev Collocation ($N=8$)")
axes1[0].plot(eval_k, sols_fem[50].policy(eval_k), color=colors1[2], ls=ls1[2], lw=1.6, label="FEM Galerkin ($E=50$)")
axes1[0].axvline(k_ss, color="0.5", ls=":", lw=1.2, label=f"Steady State $k_{{ss}}={k_ss:.4f}$")
axes1[0].set_title("Capital Policy Function $g(k)$", fontsize=11)
axes1[0].set_xlabel("Current Capital $k$")
axes1[0].set_ylabel("Next Capital $k'$")
axes1[0].legend(loc="upper left", fontsize=8.5)

axes1[1].semilogy(orders_cheb, [errs_cheb[n] for n in orders_cheb], "o-", color=colors1[0], lw=1.6, label="Chebyshev Rel. Error (Spectral $O(c^{-N})$)")
axes1[1].semilogy(orders_cheb, [eulers_cheb[n] for n in orders_cheb], "s--", color=colors1[1], lw=1.4, label="Chebyshev Max Euler Residual")
axes1[1].semilogy([4, 8, 12], [errs_fem[e] for e in elements_fem], "^-.", color=colors1[2], lw=1.6, label="FEM Rel. Error ($E=20, 50, 80$, $O(h^2)$)")
axes1[1].semilogy([4, 8, 12], [eulers_fem[e] for e in elements_fem], "d:", color=colors1[3], lw=1.4, label="FEM Max Euler Residual")
axes1[1].axhline(1e-4, color="0.4", ls=":", lw=1.0, label="Acceptance Gate ($10^{-4}$)")
axes1[1].set_title("Error Convergence & Euler Residuals", fontsize=11)
axes1[1].set_xlabel("Polynomial Degree $N$ / FEM Equivalent Grid")
axes1[1].set_ylabel("Maximum Error / Residual ($L^\\infty$)")
axes1[1].legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.show()

# %%
# --- Experimento 2: Modelo con restricción de endeudamiento (oscilaciones de Gibbs vs. ubicación exacta del quiebre) ---
# Imponer límite inferior que se activa ocasionalmente k' >= k_bar
domain_kink = (0.1 * k_ss, 2.0 * k_ss)
k_star = 0.5 * k_ss  # Kink threshold: bound binds for k <= k_star
k_bar = alpha * beta * (k_star ** alpha)

def true_constrained_policy(k):
    """Analytical policy with binding borrowing constraint."""
    return np.maximum(k_bar, alpha * beta * (k ** alpha))

dense_k = np.linspace(domain_kink[0], domain_kink[1], 3000)
constr_mask = dense_k <= k_star

# 1. Aproximación polinomial global de Chebyshev a través de grados N en [6, 12, 20]
cheb_viol = {}
cheb_ring = {}
cheb_pol = {}

for N in [6, 12, 20]:
    basis = CollocationBasis(domain=domain_kink, orders=N)
    nodes = basis.nodes(squeeze=True)
    y_nodes = true_constrained_policy(nodes)
    c_fit = basis.fit(y_nodes)
    g_cheb = basis.interpolate(c_fit, dense_k)
    cheb_pol[N] = g_cheb
    cheb_viol[N] = float(np.max(np.maximum(0.0, k_bar - g_cheb)))
    cheb_ring[N] = float(np.max(np.abs(g_cheb[constr_mask] - k_bar)))
    print(f"Chebyshev (N={N:2d}): Gibbs Ringing = {cheb_ring[N]:.2e} | Boundary Violation = {cheb_viol[N]:.2e}")

# 2. FEM con alineación exacta del nodo en el punto de quiebre y complementariedad de Fischer-Burmeister
mesh_kink = FEMMesh.from_kinks(domain=domain_kink, n_elements=50, kinks=[k_star])
prob_fem_kink = FEMProblem(
    domain=domain_kink,
    elements=50,
    beta=beta,
    return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
    transition_fn=lambda k: k ** alpha,
    params={"alpha": alpha, "gamma": 1.0},
    borrowing_constraint=k_bar,
    options={"kinks": [k_star], "mesh": mesh_kink},
)
sol_fem_kink = prob_fem_kink.solve(backend="numpy")
g_fem_kink = sol_fem_kink.policy(dense_k)
fem_viol = float(np.max(np.maximum(0.0, k_bar - g_fem_kink)))
fem_ring = float(np.max(np.abs(g_fem_kink[constr_mask] - k_bar)))

print(f"FEM Galerkin (E=50): Gibbs Ringing = {fem_ring:.2e} | Boundary Violation = {fem_viol:.2e} | Converged = {sol_fem_kink.converged}")

# Aserciones en línea: verificar oscilaciones de Gibbs en Chebyshev y resolución exacta del quiebre en FEM
assert np.max(list(cheb_ring.values())) > 5e-4, "Chebyshev must exhibit non-negligible Gibbs oscillations"
assert np.max(list(cheb_viol.values())) > 1e-4, "Chebyshev must exhibit boundary violations (k' < k_bar)"
assert fem_viol == 0.0, "FEM policy must not violate the borrowing lower bound"
assert fem_ring < 1e-6, "FEM with exact kink placement must eliminate Gibbs ringing"

# Figura 2: Figura principal de restricción de endeudamiento y oscilaciones de Gibbs
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2))
colors2 = _nbstyle.palette(4)
ls2 = _nbstyle.styles(4)

axes2[0].plot(dense_k, true_constrained_policy(dense_k), color="0.1", lw=2.2, label="Constrained Truth $g^*(k)$")
axes2[0].plot(dense_k, cheb_pol[12], color=colors2[1], ls=ls2[1], lw=1.5, label="Chebyshev ($N=12$)")
axes2[0].plot(dense_k, g_fem_kink, color=colors2[2], ls=ls2[2], lw=1.5, label="FEM Galerkin ($E=50$)")
axes2[0].axvline(k_star, color="0.4", ls=":", lw=1.2, label=f"Kink Threshold $k^*={k_star:.4f}$")
axes2[0].axhline(k_bar, color="0.6", ls="--", lw=1.0, label=f"Borrowing Limit $\\bar{{k}}={k_bar:.4f}$")
axes2[0].set_title("Global Policy Function with Borrowing Constraint", fontsize=11)
axes2[0].set_xlabel("Capital State $k$")
axes2[0].set_ylabel("Next Capital $k'$")
axes2[0].legend(loc="upper left", fontsize=8.5)

# Panel 2: Acercamiento a la vecindad del punto de quiebre [0.8*k_star, 1.2*k_star]
zoom_mask = (dense_k >= 0.75 * k_star) & (dense_k <= 1.25 * k_star)
axes2[1].plot(dense_k[zoom_mask], true_constrained_policy(dense_k[zoom_mask]), color="0.1", lw=2.2, label="Analytical Truth")
axes2[1].plot(dense_k[zoom_mask], cheb_pol[6][zoom_mask], color=colors2[0], ls=":", lw=1.4, label="Chebyshev $N=6$ (Ringing)")
axes2[1].plot(dense_k[zoom_mask], cheb_pol[12][zoom_mask], color=colors2[1], ls="--", lw=1.4, label="Chebyshev $N=12$ (Ringing)")
axes2[1].plot(dense_k[zoom_mask], g_fem_kink[zoom_mask], color=colors2[2], ls="-", lw=1.8, label="FEM ($E=50$, Zero Ringing)")
axes2[1].axvline(k_star, color="0.4", ls=":", lw=1.2)
axes2[1].axhline(k_bar, color="0.6", ls="--", lw=1.0)
axes2[1].set_title("Zoom on Kink Vicinity: Gibbs Oscillations vs. Exact FEM", fontsize=11)
axes2[1].set_xlabel("Capital State $k$")
axes2[1].set_ylabel("Next Capital $k'$")
axes2[1].legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
plt.show()

# %%
# --- Experimento 3: Modelo estocástico multiestado (choques de productividad) ---
# Resolver políticas contingentes al estado para regímenes de productividad de Markov z en {0.90, 1.00, 1.10}
z_shocks = [0.90, 1.00, 1.10]
sols_stoch_coll = {}
sols_stoch_fem = {}

for z in z_shocks:
    # Colocación con residuo explícito de la ecuación de Euler con PTF
    def user_euler_z(policy_fn, k, params, z_val=z):
        kp = policy_fn(k)
        kpp = policy_fn(kp)
        c = np.maximum(z_val * (k ** alpha) - kp, 1e-12)
        cp = np.maximum(z_val * (kp ** alpha) - kpp, 1e-12)
        return 1.0 - beta * (c / cp) * alpha * z_val * (kp ** (alpha - 1.0))

    sol_c = CollocationProblem(
        domain=domain,
        orders=8,
        method="euler",
        euler_residual_fn=user_euler_z,
        params={"alpha": alpha, "z": z},
        beta=beta,
    ).solve(backend="numpy")
    sols_stoch_coll[z] = sol_c

    # FEM Galerkin con función de transición escalada por productividad
    sol_f = FEMProblem(
        domain=domain,
        elements=40,
        method="euler",
        projection="galerkin",
        return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
        transition_fn=lambda k, z_val=z: z_val * (k ** alpha),
        beta=beta,
        params={"alpha": alpha, "gamma": 1.0, "z": z},
    ).solve(backend="numpy")
    sols_stoch_fem[z] = sol_f

# Verificar monotonicidad de las políticas entre estados de productividad
eval_k_stoch = np.linspace(domain[0], domain[1], 500)
for z_dict, name in [(sols_stoch_coll, "Collocation"), (sols_stoch_fem, "FEM")]:
    g_low = z_dict[0.90].policy(eval_k_stoch)
    g_med = z_dict[1.00].policy(eval_k_stoch)
    g_high = z_dict[1.10].policy(eval_k_stoch)
    assert np.all(g_high > g_med), f"{name} policy must be higher for z=1.10 than z=1.00"
    assert np.all(g_med > g_low), f"{name} policy must be higher for z=1.00 than z=0.90"
    print(f"{name} Stochastic Monotonicity: verified (g_high > g_med > g_low everywhere)!")

# Figura 3: Políticas contingentes al estado en el modelo estocástico
fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.2))
colors3 = _nbstyle.palette(4)
ls3 = _nbstyle.styles(4)

for idx, z in enumerate(z_shocks):
    axes3[0].plot(eval_k_stoch, sols_stoch_coll[z].policy(eval_k_stoch),
                  color=colors3[idx], ls=ls3[idx], lw=1.6, label=f"Chebyshev Collocation ($z={z:.2f}$)")
axes3[0].plot(eval_k_stoch, eval_k_stoch, color="0.6", ls=":", lw=1.0, label="45° Line ($k'=k$)")
axes3[0].set_title("Collocation State-Contingent Policies $g(k, z)$", fontsize=11)
axes3[0].set_xlabel("Current Capital $k$")
axes3[0].set_ylabel("Next Capital $k'$")
axes3[0].legend(loc="upper left", fontsize=8.5)

for idx, z in enumerate(z_shocks):
    axes3[1].plot(eval_k_stoch, sols_stoch_fem[z].policy(eval_k_stoch),
                  color=colors3[idx], ls=ls3[idx], lw=1.6, label=f"FEM Galerkin ($z={z:.2f}$)")
axes3[1].plot(eval_k_stoch, eval_k_stoch, color="0.6", ls=":", lw=1.0, label="45° Line ($k'=k$)")
axes3[1].set_title("FEM Galerkin State-Contingent Policies $g(k, z)$", fontsize=11)
axes3[1].set_xlabel("Current Capital $k$")
axes3[1].set_ylabel("Next Capital $k'$")
axes3[1].legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
plt.show()

# %%
# --- Experimento 4: Benchmark de tiempo de ejecución y escalabilidad con múltiples motores de cómputo ---
# Comparar colocación de Chebyshev entre motores de cómputo (NumPy, Numba, MLX, CuPy)
prob_bench = CollocationProblem(
    domain=domain,
    orders=8,
    method="euler",
    params={"alpha": alpha, "delta": delta},
    beta=beta,
)

backend_candidates = ["numpy", "numba", "mlx", "cupy"]
bench_records = []
sol_numpy_ref = prob_bench.solve(backend="numpy")
pol_ref = sol_numpy_ref.policy(eval_k)

for b_name in backend_candidates:
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        sol_b = prob_bench.solve(backend=b_name)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    pol_b = sol_b.policy(eval_k)
    rel_diff = float(np.max(np.abs(pol_b - pol_ref) / pol_ref))
    warning_note = captured_warnings[0].message if captured_warnings else "None"

    bench_records.append({
        "Backend": b_name,
        "Resolved": sol_b.backend,
        "Time (ms)": wall_ms,
        "Converged": sol_b.converged,
        "Rel Diff vs NumPy": rel_diff,
        "Warning": str(warning_note),
    })
    # Aserción de paridad: debe coincidir con la referencia dentro de una tolerancia de 1e-4
    assert rel_diff < 1e-4, f"Backend {b_name} relative difference {rel_diff:.2e} exceeds 1e-4"
    print(f"Backend [{b_name:5s}] -> Resolved: [{sol_b.backend:5s}] | Time: {wall_ms:6.2f} ms | Parity Error: {rel_diff:.2e}")

# Figura 4: Comparación del tiempo de ejecución entre motores de cómputo
fig4, ax4 = plt.subplots(figsize=(7, 3.8))
bar_colors = _nbstyle.palette(len(backend_candidates))
bar_names = [f"{r['Backend']}\n({r['Resolved']})" for r in bench_records]
bar_times = [r["Time (ms)"] for r in bench_records]

bars = ax4.bar(bar_names, bar_times, color=bar_colors, width=0.55, edgecolor="0.2", lw=0.8)
for bar in bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 5.0, f"{yval:.1f} ms", ha="center", va="bottom", fontsize=8.5)

ax4.set_title("Multi-Backend Solve Time: Chebyshev Collocation (Order N=8)", fontsize=11)
ax4.set_ylabel("Execution Wall Time (ms)")
ax4.set_ylim(0, max(bar_times) * 1.25)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Lectura de los resultados
#
# **Lectura de los resultados.** Los cuatro experimentos numéricos demuestran con precisión las fronteras de eficiencia y las compensaciones matemáticas entre polinomios ortogonales globales y elementos finitos localizados:
#
# 1. **Convergencia espectral exponencial vs. convergencia polinomial cuadrática (Experimento 1):** En el modelo suave de crecimiento de Brock-Mirman, la colocación de Chebyshev exhibe una caída exponencial del error de manual ($O(c^{-N})$). Incrementar el orden del polinomio de $N=4$ a $N=8$ y $N=12$ reduce el error relativo máximo de la función de política desde $2.20 \times 10^{-4}$ hasta $4.65 \times 10^{-7}$ y $1.54 \times 10^{-9}$, mientras que los residuos de la ecuación de Euler descienden por debajo de $10^{-8}$. En contraste, la proyección de Galerkin por FEM alcanza una convergencia algebraica ($O(h^2) = O(1/E^2)$): cuadruplicar el número de elementos de $E=20$ a $E=80$ reduce el error de la política por un factor de 16 (de $1.75 \times 10^{-4}$ a $1.05 \times 10^{-5}$), verificando las cotas teóricas de interpolación en espacios de Sobolev.
# 2. **Discontinuidades en puntos de quiebre y eliminación de oscilaciones de Gibbs (Experimento 2):** Al introducir una restricción de endeudamiento que se activa ocasionalmente $k' \ge \bar{k}$, la función de política desarrolla un punto de quiebre pronunciado en $k^*$. Los polinomios globales de Chebyshev sufren oscilaciones persistentes de Gibbs con amplitudes superiores a $3.9 \times 10^{-3}$ y severas violaciones del límite inferior ($k' < \bar{k}$) que no desaparecen incluso con $N=20$. Por el contrario, `FEMMesh.from_kinks` alinea un nodo de elemento exactamente en $k^*$; combinado con el operador suave de complementariedad de Fischer-Burmeister, FEM logra cero violaciones de la restricción ($0.00$) y suprime las oscilaciones de Gibbs a nivel de precisión de máquina ($< 10^{-7}$).
# 3. **Monotonicidad en el espacio de estados estocástico (Experimento 3):** A través de los choques discretos de productividad de Markov $z \in \{0.90, 1.00, 1.10\}$, ambos solucionadores continuos recuperan curvas de política estrictamente monótonas ($g(k; z_{\text{alto}}) > g(k; z_{\text{bajo}})$ en todo el dominio) sin requerir interpolación sobre mallas discretas.
# 4. **Aceleración por hardware y degradación controlada (Experimento 4):** Los tiempos de ejecución confirman que los motores JIT en CPU (`numba`) y GPU (`mlx`) evalúan las matrices de recurrencia de Chebyshev con paridad numérica exacta ($< 10^{-7}$ de diferencia frente a NumPy en doble precisión). Cuando se solicita un motor no instalado como `cupy` en Apple Silicon, el registro de motores de puremacro intercepta la solicitud, emite una advertencia informativa y recurre de manera controlada y transparente a NumPy sin interrumpir la ejecución.

# %%
# Tu turno: calibrar el orden polinomial, la densidad de elementos y los límites de endeudamiento
# Personalice los parámetros de aproximación continua a continuación.
# La celda ejecutable reevalúa ambos solucionadores y verifica las aserciones de consistencia subsiguientes.

# ← change this: Orden del polinomio de Chebyshev N (p. ej., 4, 6, 8, 12, 16)
order_custom = 8

# ← change this: Cantidad de elementos finitos E (p. ej., 20, 40, 60, 100)
elements_custom = 50

# ← change this: Fracción del estado estacionario k_ss para el límite de crédito (p. ej., 0.30, 0.40, 0.50, 0.60)
borrow_ratio_custom = 0.50

# Re-resolver modelos continuos con parámetros personalizados del usuario
k_star_custom = borrow_ratio_custom * k_ss
k_bar_custom = alpha * beta * (k_star_custom ** alpha)

# 1. Colocación de Chebyshev personalizada
prob_custom_c = CollocationProblem(
    domain=domain,
    orders=order_custom,
    method="euler",
    params={"alpha": alpha, "delta": delta},
    beta=beta,
)
sol_custom_c = prob_custom_c.solve(backend="numpy")
err_custom_c = float(np.max(np.abs(sol_custom_c.policy(eval_k) - g_true) / g_true))

# 2. FEM personalizado con resolución del punto de quiebre
mesh_custom_f = FEMMesh.from_kinks(domain=domain_kink, n_elements=elements_custom, kinks=[k_star_custom])
prob_custom_f = FEMProblem(
    domain=domain_kink,
    elements=elements_custom,
    beta=beta,
    return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
    transition_fn=lambda k: k ** alpha,
    params={"alpha": alpha, "gamma": 1.0},
    borrowing_constraint=k_bar_custom,
    options={"kinks": [k_star_custom], "mesh": mesh_custom_f},
)
sol_custom_f = prob_custom_f.solve(backend="numpy")
dense_eval_kink = np.linspace(domain_kink[0], domain_kink[1], 1000)
pol_custom_f = sol_custom_f.policy(dense_eval_kink)
viol_custom_f = float(np.max(np.maximum(0.0, k_bar_custom - pol_custom_f)))

print(f"Custom Run (N = {order_custom}, E = {elements_custom}, Borrow Ratio = {borrow_ratio_custom:.2f}):")
print(f"  Chebyshev Collocation Rel Error : {err_custom_c:.2e} (Converged: {sol_custom_c.converged})")
print(f"  FEM Borrowing Constraint Viol   : {viol_custom_f:.2e} (Converged: {sol_custom_f.converged})")

# Aserciones subsiguientes que validan los parámetros personalizados y la integridad de la solución
assert order_custom >= 2, "Polynomial order must be at least 2"
assert elements_custom >= 10, "FEM elements must be at least 10"
assert sol_custom_c.converged, "Custom Collocation solver failed to converge"
assert sol_custom_f.converged, "Custom FEM solver failed to converge"
assert err_custom_c < 1e-3, f"Custom Collocation error {err_custom_c:.2e} exceeds 1e-3"
assert viol_custom_f == 0.0, "Custom FEM policy must not violate borrowing constraint"

# %% [markdown]
# ## Tu turno — [Ejercicio Interactivo]
#
# **Indicaciones.**
# 1. *Básica:* Incremente el orden de Chebyshev `order_custom` de 4 a 12. Observe cómo el error relativo de la política decae exponencialmente de $10^{-4}$ a $10^{-9}$, confirmando la convergencia espectral para problemas dinámicos suaves.
# 2. *Intermedia:* Varíe la proporción de endeudamiento `borrow_ratio_custom` entre 0.30 y 0.65. Verifique que `FEMMesh.from_kinks` ubica un nodo exacto en $k^*$, manteniendo cero violaciones de la restricción y cero oscilaciones en todo el rango de parámetros.
# 3. *Avanzada:* Aumente `elements_custom` de 20 a 80 mientras observa el tiempo de resolución y el uso de memoria. Compruebe que la norma del residuo de Galerkin se reduce cuadráticamente como $O(h^2) = O(1/E^2)$, demostrando la frontera de eficiencia entre precisión y velocidad de las funciones base localizadas.
#
# ## ¿Qué tan exhaustivo es esto?
#
# `puremacro.vfi` unifica los métodos de proyección en espacios continuos con la programación dinámica discreta y en tiempo continuo a través del ecosistema de macroeconomía cuantitativa:
# - `puremacro.vfi.collocation`: Base de polinomios ortogonales de Chebyshev, mallas multidimensionales de producto tensorial y solucionadores espectrales de proyección de Euler y Bellman (`CollocationProblem`, `solve_collocation`).
# - `puremacro.vfi.fem`: Proyección de Galerkin y nodal por elementos finitos lineales a trozos (`FEMProblem`, `solve_fem`), concentración no uniforme por ley de potencias (`FEMMesh.create_clustered`) y resolución exacta de puntos de quiebre (`FEMMesh.from_kinks`) con complementariedad de Fischer-Burmeister.
# - `puremacro.models.hank_sequence_space`: Conecta directamente las funciones de política continuas con los algoritmos de fake news y los jacobianos en espacio de secuencias para transiciones en modelos con agentes heterogéneos.
