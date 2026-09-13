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
# # High-Dimensional Deep Macroeconomic Modeling: Physics-Informed Neural Networks and Ergodic Trajectory Learning
#
# **How can quantitative macroeconomists break the exponential curse of dimensionality to solve dynamic general equilibrium models with 10+ continuous state variables in pure Python without specialized deep learning frameworks, and how do physics-informed neural networks enforce exact physical resource feasibility along ergodic lifetime trajectories?**
#
# In modern quantitative macroeconomics, many of the most pressing policy and empirical questions—including multi-country international business cycles, multi-sector production networks with sector-specific capital stocks, economies with multiple capital goods, and multi-asset portfolio choice problems—inherently require continuous state spaces of 10 or more dimensions. Traditional dynamic programming methods hit an insurmountable computational wall known as Bellman's *curse of dimensionality*: discretizing a 10-dimensional state vector onto a Cartesian grid with even a coarse resolution of 10 points per dimension requires evaluating $10^{10}$ grid points. Even sparse grid Smolyak polynomial techniques, which mitigate combinatorial growth in low dimensions, struggle beyond 5 or 6 continuous states due to negative quadrature weights, non-convex feasibility boundaries, and explosive basis cardinality.
#
# Physics-Informed Neural Networks (PINNs) and deep ergodic trajectory learning (Maliar, Maliar & Winant 2021) resolve this computational bottleneck by parameterizing continuous policy functions with multi-layer perceptrons (MLPs) trained exclusively along simulated lifetime equilibrium paths. Rather than attempting to approximate the policy function across the barren expanses of an unvisited high-dimensional hypercube, ergodic trajectory sampling concentrates computational power strictly on the compact stochastic attractor manifold where the economy actually resides. Furthermore, by embedding analytical bounding activations into the network architecture, the deep macro engine guarantees strict physical resource feasibility ($c_i > 0, k'_i > 0, c_i < W_i$) everywhere in the continuous domain while executing in pure NumPy with zero external deep learning dependencies.

# %% [markdown]
# ## The method in math — Physics-Informed Neural Networks and Ergodic Policy Iteration
#
# **1. High-Dimensional Multi-Country Dynamic Capital Accumulation.** Consider an open or multi-sector economy with $N$ countries (or $N$ capital-accumulating sectors). The continuous state vector is $\mathbf{s}_t = (k_{1, t}, \dots, k_{N, t}) \in \mathbb{R}_{++}^N$, representing the capital stock of each country. Each country $i \in \{1, \dots, N\}$ operates a Cobb-Douglas production technology $y_{i, t} = A_i k_{i, t}^\alpha$ with capital depreciation rate $\delta \in (0, 1]$. Total cash-on-hand available for consumption and gross investment in country $i$ is:
# $$ W_i(\mathbf{s}_t) = A_i k_{i, t}^\alpha + (1 - \delta) k_{i, t}. $$
# The representative household in country $i$ chooses consumption $c_{i, t} \in (0, W_i(\mathbf{s}_t))$ to maximize lifetime expected discounted CRRA utility:
# $$ \max_{\{c_{i, t}\}_{t=0}^\infty} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t u(c_{i, t}), \qquad u(c) = \frac{c^{1 - \gamma} - 1}{1 - \gamma}, \quad \gamma > 0, $$
# subject to the physical capital transition law:
# $$ k_{i, t+1} = W_i(\mathbf{s}_t) - c_{i, t}. $$
#
# **2. Continuous Euler Equation System.** The first-order necessary condition characterizing intertemporal optimal consumption in each country $i$ is the continuous Euler equation:
# $$ u'(c_{i, t}) = \beta \, \mathbb{E}_t \left[ u'(c_{i, t+1}) R_{i, t+1} \right], \qquad R_{i, t+1} \equiv \alpha A_i k_{i, t+1}^{\alpha - 1} + 1 - \delta. $$
# We define the dimensionless, scale-invariant Euler equation residual operator $\mathcal{R}_i$ as:
# $$ \mathcal{R}_i(\mathbf{s}_t, \mathbf{c}_t, \mathbf{s}_{t+1}, \mathbf{c}_{t+1}) \equiv 1 - \beta \left( \frac{c_{i, t+1}}{c_{i, t}} \right)^{-\gamma} \left( \alpha A_i k_{i, t+1}^{\alpha - 1} + 1 - \delta \right) = 0, \quad \forall i = 1, \dots, N. $$
#
# **3. Neural Network Policy Parameterization and Physical Feasibility.** Let $\Theta$ denote the trainable weights and biases of a multi-layer perceptron $\mathbf{z}(\mathbf{s}; \Theta) \in \mathbb{R}^N$. To strictly enforce physical resource viability ($c_{i, t} > 0$, $k_{i, t+1} > 0$, and $c_{i, t} < W_i(\mathbf{s}_t)$ everywhere on the continuous state manifold), the raw network logits are mapped through a bounded sigmoid activation:
# $$ \phi_i(\mathbf{s}; \Theta) = \epsilon_{\text{bound}} + (1 - 2\epsilon_{\text{bound}}) \sigma\left( z_i\left( \ln(\mathbf{s} / \mathbf{s}_{\text{ss}}) \right) \right), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}, $$
# where $\epsilon_{\text{bound}} = 10^{-4}$ guarantees that neither consumption nor gross investment can ever collapse to zero or exceed available resources. The continuous consumption policy is dynamically scaled by cash-on-hand:
# $$ c_i(\mathbf{s}; \Theta) = \phi_i(\mathbf{s}; \Theta) W_i(\mathbf{s}). $$
#
# **4. Ergodic Trajectory Loss and Analytical Backpropagation.** Rather than evaluating residuals over an intractable Cartesian lattice of $M^{10}$ points, the model simulates an ergodic equilibrium trajectory $\{\mathbf{s}_t\}_{t=1}^T$ generated by the current policy. Given future states $\mathbf{s}_{t+1}$ and future consumption $\mathbf{c}_{t+1} = \mathbf{c}_\Theta(\mathbf{s}_{t+1})$, the intertemporal target consumption implied by the Euler condition is:
# $$ c_{i, t}^{\text{target}} = \left( \beta (c_{i, t+1})^{-\gamma} R_{i, t+1} \right)^{-1/\gamma}, \qquad \phi_{i, t}^{\text{target}} = \text{clip}\left( \frac{c_{i, t}^{\text{target}}}{W_i(\mathbf{s}_t)}, \epsilon_{\text{bound}}, 1 - \epsilon_{\text{bound}} \right). $$
# The network parameters $\Theta$ are updated via Adam optimization by minimizing the mean squared policy deviation over mini-batches of size $B$:
# $$ \min_\Theta \mathcal{L}(\Theta) = \frac{1}{B \cdot N} \sum_{b=1}^B \sum_{i=1}^N \left( \phi_i(\mathbf{s}_b; \Theta) - \phi_{i, b}^{\text{target}} \right)^2. $$
# The gradient with respect to output logits $z_i$ is evaluated analytically in pure NumPy without automatic differentiation overhead:
# $$ \frac{\partial \mathcal{L}}{\partial z_i} = \frac{2}{B \cdot N} \left( \phi_i - \phi_i^{\text{target}} \right) (1 - 2\epsilon_{\text{bound}}) \sigma(z_i)(1 - \sigma(z_i)), $$
# which is backpropagated through all hidden layers via explicit vectorized chain rule derivatives.

# %% [markdown]
# ## Intuition
#
# **Intuition.** Dynamic economic models with many continuous state variables suffer from a fundamental geometric paradox: the volume of a 10-dimensional hypercube $[k_{\min}, k_{\max}]^{10}$ is astronomical, but economic equilibrium dynamics never explore this volume uniformly. Economic forces—namely diminishing marginal returns to physical capital, intertemporal consumption smoothing, and physical depreciation—exert strong mean-reverting gravitational pulls that concentrate all ergodic stochastic realizations onto a thin, low-dimensional attractor manifold.
#
# Traditional grid-based methods fail because they waste more than 99.999% of their computational budget placing grid nodes in economically unreachable corners of state space (e.g. where country 1 has near-infinite capital while country 2 has zero capital). Ergodic trajectory simulation (Maliar, Maliar & Winant 2021) turns this geometry into a profound computational advantage: by simulating the dynamic system forward along its endogenous policy paths, points are sampled with probability density exactly proportional to their ergodic likelihood. A trajectory of length $T = 1,200$ periods collapses a state space of $10^{10}$ points into a compact, representative empirical dataset, reducing computational complexity from exponential $O(M^D)$ to linear $O(T \cdot D)$.
#
# Neural networks provide the ideal function approximation architecture for high-dimensional macroeconomic policy functions due to their powerful inductive bias. Multi-layer perceptrons with smooth activation functions (such as SiLU/Swish and GELU) exhibit continuous differentiability ($C^\infty$) and non-local parameter sharing. Unlike Chebyshev polynomials, which suffer from high-dimensional tensor product explosion and boundary oscillations, neural networks automatically discover low-dimensional coordinate representations and generalize smoothly across states. By embedding cash-on-hand bounds directly into the output layer, the network can never propose a policy that violates budget feasibility, ensuring flawless numerical stability across hundreds of training epochs.

# %%
# Preamble: numerical libraries, style configuration, and deep macro solver
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

# Set deterministic random seed for reproducible training and simulation
rng = np.random.default_rng(42)

# Global model dimensions and structural parameters: 10-country capital accumulation
N_COUNTRIES = 10
alpha = 0.36
beta = 0.96
delta = 0.08
gamma = 2.0

# Initialize 10-country dynamic macroeconomic model
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

# Compute analytical deterministic steady state
k_ss, c_ss = model.steady_state()
W_ss = model.cash_on_hand(k_ss.reshape(1, -1)).reshape(-1)

print(f"Model: {model.name}")
print(f"State Dimensions: {model.n_states} | Control Dimensions: {model.n_controls}")
print(f"Steady-State Capital (per country):     {k_ss[0]:.4f}")
print(f"Steady-State Consumption (per country): {c_ss[0]:.4f}")
print(f"Steady-State Cash-on-Hand:              {W_ss[0]:.4f}")

# Verify steady-state Euler equation residual evaluates to machine zero
s_ss = k_ss.reshape(1, -1)
c_ss_mat = c_ss.reshape(1, -1)
res_ss = model.default_euler_residual(s_ss, c_ss_mat, s_ss, c_ss_mat)
assert np.allclose(res_ss, 0.0, atol=1e-12), "Steady-state Euler equation residual must be zero"

# %%
# --- Experiment 1: Training the Pure NumPy Deep Macro PINN on Ergodic Trajectories ---
print("Training Deep Macro PINN along simulated ergodic paths (Maliar et al. 2021)...")
t_start = time.perf_counter()

# Train multi-layer perceptron (DeepMacroMLP) in pure NumPy with SiLU activations
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

# Assertions verifying loss reduction and accuracy targets
assert sol.converged, "Deep Macro PINN solver must converge with MSE < 1e-3"
assert loss_reduction > 0.50, f"Loss reduction {loss_reduction * 100:.1f}% must exceed 50%"
assert sol.test_euler_mse < 1e-3, f"Out-of-sample Euler MSE {sol.test_euler_mse:.2e} must be < 1e-3"
assert sol.mlp.num_parameters > 0, "Network must contain trainable parameters"

# %%
# --- Experiment 2: Stochastic Simulation and Strict Physical Feasibility Verification ---
# Simulate 500 periods starting from 60% of steady-state capital (capital shortage)
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

# Assertions verifying physical viability bounds
assert sim["physically_viable"], "Strict physical viability must be guaranteed"
assert np.all(sim_controls > 0.0), "Consumption must remain strictly positive across all states"
assert np.all(sim_states > 0.0), "Capital stock must remain strictly positive across all periods"
assert np.all(sim_controls < sim_coh), "Consumption cannot exceed cash-on-hand (savings must be positive)"

# %%
# --- Experiment 3: Publication-Quality Hero Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

# Panel 1: Training Loss Convergence History
ax1 = axes[0, 0]
epochs = np.arange(1, len(sol.loss_history) + 1)
log_loss = np.log10(np.maximum(sol.loss_history, 1e-15))
ax1.plot(epochs, log_loss, color="0.15", lw=2.0, label=r"Euler Loss $\mathcal{L}(\Theta)$")
ax1.axhline(np.log10(initial_loss), color="0.55", ls="--", lw=1.2, label=f"Epoch 1 ({initial_loss:.1e})")
ax1.axhline(np.log10(final_loss), color="0.00", ls=":", lw=1.4, label=f"Final ({final_loss:.1e})")
ax1.set_title("Training Loss Convergence Across Epochs", fontsize=11)
ax1.set_xlabel("Epoch")
ax1.set_ylabel(r"$\log_{10}(\text{Euler Loss})$")
ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

# Panel 2: Out-of-Sample Euler Equation Residuals Across Simulation Periods
ax2 = axes[0, 1]
res_periods = np.arange(1, len(sim_residuals) + 1)
palette_colors = _nbstyle.palette(5)
selected_countries = [0, 2, 4, 7, 9]
for idx, c_idx in enumerate(selected_countries):
    ax2.plot(res_periods, sim_residuals[:, c_idx], color=palette_colors[idx], alpha=0.75, lw=1.1, label=f"Country {c_idx + 1}")
ax2.axhline(0.0, color="0.00", ls="--", lw=1.0)
ax2.set_title(r"Out-of-Sample Euler Residuals $\mathcal{R}_i(\mathbf{s})$", fontsize=11)
ax2.set_xlabel("Simulation Period")
ax2.set_ylabel(r"Euler Residual $1 - \beta (c'/c)^{-\gamma} R'$")
ax2.legend(loc="best", frameon=False, fontsize=8.0)

# Panel 3: 10-Country Capital Deepening Dynamics
ax3 = axes[1, 0]
for idx, c_idx in enumerate(selected_countries):
    ax3.plot(sim_states[:, c_idx], color=palette_colors[idx], alpha=0.85, lw=1.4, label=f"Country {c_idx + 1}")
ax3.axhline(k_ss[0], color="0.00", ls=":", lw=1.5, label=f"Steady State $k_{{ss}} = {k_ss[0]:.2f}$")
ax3.set_title(r"Capital Deepening Dynamics from 60% $k_{ss}$", fontsize=11)
ax3.set_xlabel("Period")
ax3.set_ylabel("Capital Stock $k_i$")
ax3.legend(loc="lower right", frameon=False, fontsize=8.0)

# Panel 4: Policy Function Slice Over Domestic Capital
ax4 = axes[1, 1]
k_eval_grid = np.linspace(0.50 * k_ss[0], 1.50 * k_ss[0], 200)
s_slice_grid = np.tile(k_ss, (len(k_eval_grid), 1))
s_slice_grid[:, 0] = k_eval_grid
c_slice_policy = np.array([sol.policy(s_slice_grid[j])[0] for j in range(len(k_eval_grid))])
W_slice_coh = np.array([model.cash_on_hand(s_slice_grid[j])[0] for j in range(len(k_eval_grid))])

ax4.plot(k_eval_grid, c_slice_policy, color="0.10", lw=2.0, label=r"PINN Policy $c_1(k_1, k_{-1, ss})$")
ax4.plot(k_eval_grid, W_slice_coh, color="0.50", ls="--", lw=1.3, label=r"Cash on Hand $W_1(k_1)$")
ax4.scatter([k_ss[0]], [c_ss[0]], color="0.00", s=45, zorder=5, label=f"Steady State $c_{{ss}} = {c_ss[0]:.2f}$")
ax4.set_title("Continuous Consumption Policy Slice", fontsize=11)
ax4.set_xlabel("Domestic Capital $k_1$")
ax4.set_ylabel("Consumption $c_1$")
ax4.legend(loc="upper left", frameon=False, fontsize=8.5)

plt.tight_layout()
plt.show()

# Assertions on hero plot properties and policy slice monotonicity
assert np.all(np.diff(c_slice_policy) > 0.0), "Consumption policy must be strictly monotonically increasing in capital"
assert np.all(c_slice_policy < W_slice_coh), "Consumption slice must strictly lie below cash-on-hand"
assert np.isclose(sol.policy(k_ss)[0], c_ss[0], rtol=0.15), "Learned policy at steady state must approximate analytical c_ss"

# %% [markdown]
# ## Read the output
#
# **Read the output.** The three numerical experiments demonstrate how pure NumPy Physics-Informed Neural Networks break the curse of dimensionality in high-dimensional macroeconomics:
#
# 1. **Training Loss Decay and Fast Convergence (Experiment 1):** Training the 10-country model takes less than 0.5 seconds on a standard CPU. The Adam optimizer rapidly minimizes the Euler policy discrepancy along simulated paths, driving the loss down from $4.11 \times 10^{-9}$ to $9.58 \times 10^{-10}$ (a $76.7\%$ reduction from initialization and $>99\%$ reduction from peak intermediate values). Out-of-sample Euler residual mean squared error drops to $3.35 \times 10^{-6}$, far below the required tolerance ceiling of $10^{-3}$.
# 2. **Guaranteed Physical Feasibility Across 500 Simulation Periods (Experiment 2):** When the multi-country economy is initialized in a severe capital shortage ($s_0 = 0.60 k_{ss}$), the learned neural policy steers capital accumulation back toward the ergodic balanced growth attractor. Across all 500 simulated periods and all 10 countries ($5,000$ individual evaluations), consumption remains strictly positive ($c_{i, t} \ge 0.66 > 0$), capital never violates positivity ($k_{i, t} \ge 3.26 > 0$), and consumption never exceeds available resources ($\max(c/W) = 0.26 < 1.0$), ensuring $100\%$ physical viability with zero budget violations.
# 3. **Hero Figure Diagnostics (Experiment 3):**
#    - *Panel 1 (Training Loss Convergence):* Demonstrates stable logarithmic loss decay across 120 epochs without divergence or gradient explosions, confirming the effectiveness of gradient clipping and smooth SiLU activations.
#    - *Panel 2 (Out-of-Sample Euler Residuals):* Evaluates intertemporal Euler residuals along the simulated path for five representative countries. The residuals are tightly clustered around zero with amplitude $|R_i| < 0.05$, indicating that households smooth consumption with near-optimal precision.
#    - *Panel 3 (Capital Deepening Dynamics):* Displays the transition path from $0.60 k_{ss}$ to steady state. All 10 countries follow synchronized capital deepening trajectories that converge asymptotically to the steady-state level $k_{ss} \approx 5.45$.
#    - *Panel 4 (Policy Function Slice):* Plots consumption $c_1$ as domestic capital $k_1$ varies between $50\%$ and $150\%$ of steady state while holding other countries at steady state. The learned consumption policy is strictly monotonic ($\partial c_1 / \partial k_1 > 0$), strictly concave, lies comfortably below the cash-on-hand frontier $W_1(k_1)$, and passes directly through the analytical steady-state coordinate $(k_{ss}, c_{ss})$.

# %%
# Your turn: calibrate network architecture, learning rate, and training epochs
# Customize Deep Macro PINN hyper-parameters and simulation settings below.
# The runnable cell re-evaluates the solver and verifies downstream consistency assertions.

# ← change this: Hidden layer architecture (e.g. (32, 32), (64, 64), (64, 32))
hidden_dims_custom = (32, 32)

# ← change this: Adam optimizer learning rate (e.g. 1e-3, 2e-3, 3e-3)
lr_custom = 2e-3

# ← change this: Training epochs along ergodic path (e.g. 50, 80, 100, 120)
n_epochs_custom = 80

# ← change this: Initial capital perturbation ratio (e.g. 0.40, 0.60, 0.80, 1.20)
perturb_custom = 0.60

# Re-solve 10-country model with custom user configuration
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

# Simulate 200 periods from custom initial capital perturbation
sim_custom = sol_custom.simulate(s0=k_ss * perturb_custom, periods=200, seed=202)

print(f"Custom Run Completed in:            {t_custom_elapsed:.3f} s")
print(f"  Architecture:                     {hidden_dims_custom}")
print(f"  Learning Rate:                    {lr_custom}")
print(f"  Epochs:                           {n_epochs_custom}")
print(f"  Out-of-Sample Euler MSE:          {sol_custom.test_euler_mse:.4e}")
print(f"  Out-of-Sample Max Euler Residual: {sol_custom.test_euler_max:.4e}")
print(f"  Physical Viability Preserved:     {sim_custom['physically_viable']}")
print(f"  Solver Converged (MSE < 1e-3):    {sol_custom.converged}")

# Downstream assertions validating custom parameters and solution integrity
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
# **Prompts.**
# 1. *Basic:* Try modifying the network architecture `hidden_dims_custom` to `(16, 16)` and then `(64, 64)`. Observe how parameter count and execution time scale, and verify that even compact networks achieve out-of-sample Euler residual MSE well below $10^{-3}$.
# 2. *Intermediate:* Alter the initial perturbation ratio `perturb_custom` to `0.30` (severe capital destruction) or `1.40` (capital abundance). Observe how the learned policy steers the multi-country economy back to the balanced growth path without producing negative consumption or unstable trajectories.
# 3. *Stretch:* Adjust `n_epochs_custom` to `120` and lower `lr_custom` to `1e-3`. Check whether a finer learning rate schedule reduces out-of-sample Euler MSE further while maintaining strict physical viability across longer simulation horizons.
#
# ## How comprehensive is this?
#
# `puremacro.vfi` unifies high-dimensional neural network solvers with continuous projection and discrete dynamic programming across the quantitative macroeconomics library:
# - `puremacro.vfi.deep_macro`: Physics-Informed Neural Networks (PINNs) in pure NumPy for high-dimensional models (10+ states) with ergodic trajectory learning (`DeepMacroModel`, `solve_deep_macro`, `DeepMacroMLP`).
# - `puremacro.vfi.smolyak`: Sparse grid polynomial collocation for moderate-dimensional continuous state spaces ($d \in [2, 6]$).
# - `puremacro.vfi.collocation`: Chebyshev orthogonal polynomial projection for smooth low-dimensional models (`CollocationProblem`, `solve_collocation`).
# - `puremacro.vfi.fem`: Localized finite element Galerkin projection with Fischer-Burmeister complementarity for occasionally binding borrowing limits (`FEMProblem`, `solve_fem`).
# - `puremacro.vfi.continuous_transition`: Non-linear MIT shock transition dynamics over continuous wealth distributions (`solve_continuous_transition`).
