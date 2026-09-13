> 🇬🇧 English · 🇪🇸 [Español](es/deep_macro.md)

# Deep Macro & Physics-Informed Neural Networks

`puremacro.vfi.deep_macro` implements a high-dimensional dynamic macroeconomic solver based on **Physics-Informed Neural Networks (PINNs)** and ergodic trajectory sampling, following the groundbreaking methodology of **Maliar, Maliar, and Winant (2021, *Journal of Monetary Economics*)**. The engine solves dynamic economic models with $10+$ continuous state variables where traditional tensor-product and sparse grid collocation methods encounter the curse of dimensionality.

Crucially, the entire deep learning pipeline—including multi-layer perceptron forward propagation, analytical backpropagation, and the Adam optimization algorithm—is implemented in **vectorized pure NumPy**. This design guarantees 100% compliance with the Pyodide, WebAssembly, and browser runtime environments (zero imports of PyTorch, TensorFlow, or JAX), while supporting optional hardware acceleration hooks for Apple Silicon MLX and NVIDIA CuPy.

---

## 1. Theoretical & Algorithmic Framework

### 1.1 The Curse of Dimensionality in Dynamic Economics

In dynamic economic models featuring multi-sector production, multi-country trade linkages, or rich asset structures, the continuous state vector $s_t \in \mathbb{R}^D$ frequently spans $D \ge 10$ dimensions (such as 10 national capital stocks $k_t = (k_{1, t}, \dots, k_{10, t})'$).

Traditional global projection methods discretize each state dimension into $N$ grid points:
- **Full Tensor Product Grids**: Require $N^D$ evaluation points. For $N = 10$ and $D = 10$, this entails $10^{10}$ nodes—an intractable computational barrier.
- **Smolyak Sparse Grids**: Significantly reduce grid complexity to $O(D^\mu / \mu!)$, but remain constrained to $D \le 6$ dimensions for higher polynomial orders.

Physics-Informed Neural Networks bypass spatial grid discretization entirely. By parameterizing policy functions with deep neural networks $c(s; \mathbf{w})$ and training exclusively on the economically relevant ergodic attractor set, deep learning scales efficiently to ultra-high-dimensional state spaces.

### 1.2 Physics-Informed Neural Network (PINN) Formulation

Consider an economy governed by the continuous Euler residual system:

$$\mathcal{R}(s, c(s); s', c(s')) \equiv u'(c(s)) - \beta \mathbb{E} \left[ u'(c(s')) \cdot \left( \mathbf{f}'(k', z') + 1 - \delta \right) \right] = \mathbf{0}$$

A Multi-Layer Perceptron (MLP) approximates the continuous policy mapping $c(s; \mathbf{w}): \mathbb{R}^D \to \mathbb{R}^P$, where $\mathbf{w} = \{W^{(l)}, b^{(l)}\}_{l=0}^{L-1}$ denotes network weights and biases.

The PINN objective minimizes the Mean Squared Euler Equation Error over a batch of sampled states:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{B} \sum_{b=1}^B \left\| \mathcal{R}\left( s_b, c(s_b; \mathbf{w}); s'_b, c(s'_b; \mathbf{w}) \right) \right\|_2^2$$

Unlike standard supervised learning that requires pre-computed labels or training data, PINNs train purely on physical equilibrium conditions: the loss is zero if and only if the intertemporal Euler equation is satisfied.

### 1.3 Vectorized Pure NumPy MLP Architecture & Pyodide Safety

To ensure zero external dependencies and seamless operation inside browser-based environments (such as Juno or Google Colab with Pyodide), `DeepMacroMLP` implements feedforward and backpropagation passes using vectorized NumPy linear algebra:

#### Forward Pass
For layer $l = 0, \dots, L-1$:

$$z^{(l+1)} = a^{(l)} W^{(l)} + b^{(l)}, \quad a^{(l+1)} = \sigma\left( z^{(l+1)} \right)$$

where $a^{(0)} = s \in \mathbb{R}^{B \times D}$, and $\sigma(\cdot)$ is a smooth activation function.

#### Analytical Backward Pass
Using the exact matrix chain rule, output gradients $\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \odot \sigma'(z^{(L)})$ propagate backward:

$$\delta^{(l)} = \left( \delta^{(l+1)} (W^{(l)})^\top \right) \odot \sigma'\left( z^{(l)} \right)$$

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{B} (a^{(l)})^\top \delta^{(l+1)}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{B} \sum_{b=1}^B \delta_b^{(l+1)}$$

### 1.4 Ergodic Trajectory Sampling (Maliar, Maliar & Winant 2021)

Uniform grid sampling wastes computational effort exploring state space regions that have zero probability of being visited in equilibrium. Maliar et al. (2021) introduce **ergodic trajectory sampling**:

1. **Simulation**: Starting from an initial state $s_0$, simulate lifetime equilibrium paths using the current neural network policy:
   $$s_{t+1} = \mathcal{T}\left( s_t, c(s_t; \mathbf{w}), \epsilon_{t+1} \right), \quad t = 0, \dots, T_{\text{sim}}$$
2. **Burn-in Discard**: Discard the initial $T_{\text{burn}}$ transition periods to eliminate dependence on $s_0$.
3. **Mini-Batch Sampling**: Randomly draw training batches $s_b \sim \{s_t\}_{t=T_{\text{burn}}}^{T_{\text{sim}}}$ exclusively from the high-density ergodic set.
4. **Periodic Re-simulation**: Re-simulate the trajectory every $K_{\text{resim}}$ epochs as the network policy evolves.

### 1.5 Bounded Output Activation & Physical Viability Guarantees

Economic controls are bounded by physical resource constraints. For example, consumption must be strictly positive ($c > 0$) and cannot exceed cash-on-hand ($c < W(s)$).

`DeepMacroModel` enforces physical viability by applying an element-wise sigmoid bounding function to the raw network outputs $\tilde{c} = \text{MLP}(s)$:

$$c_i(s) = \epsilon_{\min} + (W_i(s) - 2\epsilon_{\min}) \cdot \sigma\left( \tilde{c}_i \right)$$

where $W_i(s) = A_i k_i^\alpha + (1 - \delta)k_i$ is total available wealth. This structural guarantee prevents the optimizer from wandering into economically unviable regions (negative consumption or negative capital) during training.

---

## 2. Methodological & Model Options

### Smooth Activation Functions
`DeepMacroMLP` supports smooth, infinitely differentiable activation functions equipped with analytical exact derivatives:

- **SiLU / Swish** ($\sigma(x) = x \cdot \text{sigmoid}(x)$): Default smooth non-linearity avoiding vanishing gradients.
- **GELU** ($\sigma(x) = 0.5 x (1 + \text{erf}(x / \sqrt{2}))$): Gaussian error linear unit standard in modern transformer architectures.
- **Tanh** ($\sigma(x) = \tanh(x)$): Symmetric bounded zero-centered activation.
- **Softplus** ($\sigma(x) = \ln(1 + e^x)$): Smooth positive activation.

### Pure NumPy Adam Optimizer
The built-in `AdamOptimizer` implements moment updates with bias correction:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_{t-1}$$

with global gradient norm clipping ($\|g\|_2 \le \bar{G}$) to ensure numerical stability.

---

## 3. Canonical Calibration & High-Dimensional Specifications

The multi-country capital accumulation benchmark models $N = 10$ interdependent economies:

| Parameter | Symbol | Benchmark Value | Economic Meaning |
|---|---|---|---|
| Number of countries (states & controls) | $N$ | $10$ | 10 capital stocks $k_i$, 10 consumption choices $c_i$ |
| Capital output elasticity | $\alpha$ | $0.360$ | Sectoral Cobb-Douglas production share |
| Subjective discount factor | $\beta$ | $0.960$ | Household annual discount rate |
| Capital depreciation rate | $\delta$ | $0.080$ | Annual physical capital depreciation rate |
| MLP hidden architecture | — | `(64, 64)` | 2 hidden layers with 64 units each (4,874 parameters) |
| Hidden activation | — | `"silu"` | Smooth SiLU activation function |
| Training epochs | — | $50$–$400$ | Epoch budget (converges in $< 0.5$ seconds for 50 epochs) |
| Ergodic trajectory length | $T_{\text{sim}}$ | $1500$ | Length of simulated stochastic equilibrium path |
| Mini-batch size | $B$ | $64$–$128$ | Number of sampled state vectors per gradient step |

---

## 4. Runnable Worked Examples

The following script instantiates a 10-country dynamic growth model ($D = 10$), trains the physics-informed neural network along simulated ergodic paths, and verifies out-of-sample Euler residual accuracy:

```python
import numpy as np
from puremacro.vfi.deep_macro import (
    DeepMacroModel,
    solve_deep_macro,
    DeepMacroSolution,
)

# 1. Instantiate High-Dimensional Multi-Country Growth Model (10 States, 10 Controls)
model = DeepMacroModel.multi_country_growth(
    n_countries=10,
    alpha=0.36,
    beta=0.96,
    delta=0.08,
)
k_ss, c_ss = model.steady_state()
assert len(k_ss) == 10 and len(c_ss) == 10

# 2. Train Pure NumPy Physics-Informed Neural Network along Ergodic Trajectories
solution = solve_deep_macro(
    model,
    hidden_dims=(64, 64),
    activation="silu",
    n_epochs=50,
    batch_size=64,
    lr=2e-3,
    trajectory_length=1000,
    burn_in=100,
    seed=42,
    verbose=False,
)

assert isinstance(solution, DeepMacroSolution)
assert solution.converged
assert solution.test_euler_mse < 1e-3

# 3. Simulate Long Trajectory and Evaluate Physical Viability
sim_data = solution.simulate(periods=300, seed=123)
assert sim_data["physically_viable"]
summary_df = solution.summary()

print(f"Deep Macro solved 10-country model in {len(solution.loss_history)} epochs")
print(f"Final Loss: {solution.loss_history[-1]:.2e}, Out-of-sample MSE: {solution.test_euler_mse:.2e}")
```

---

## 5. Full API Specification

```text
DeepMacroModel(
    n_states: int,
    n_controls: int,
    beta: float = 0.96,
    params: dict[str, Any] = field(default_factory=dict),
    reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    transition_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray] | None = None,
    euler_residual_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    name: str = "Multi-Country Capital Accumulation",
)

DeepMacroModel.multi_country_growth(
    n_countries: int = 10,
    alpha: float = 0.36,
    beta: float = 0.96,
    delta: float = 0.08,
    A: float | np.ndarray = 1.0,
    gamma: float = 1.0,
    rho: float = 0.90,
    sigma_eps: float = 0.02,
) -> DeepMacroModel

DeepMacroMLP(
    input_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    output_dim: int = 1,
    activation: str = "silu",
    output_activation: str = "linear",
    seed: int = 42,
    backend: str = "numpy",
)

AdamOptimizer(
    params: list[np.ndarray],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
)

solve_deep_macro(
    model: DeepMacroModel,
    hidden_dims: Sequence[int] = (64, 64),
    activation: str = "silu",
    n_epochs: int = 400,
    batch_size: int = 128,
    lr: float = 2e-3,
    trajectory_length: int = 2000,
    burn_in: int = 200,
    resimulate_every: int = 50,
    weight_decay: float = 1e-6,
    grad_clip: float = 1.0,
    backend: str = "numpy",
    seed: int = 42,
    verbose: bool = False,
    **kwargs: Any,
) -> DeepMacroSolution
```

#### Parameters:
- `model`: Problem definition specifying state/control dimensions, discount factor $\beta$, and economic parameters.
- `hidden_dims`: Layer dimensions of the neural network (e.g. `(64, 64)`).
- `activation`: Smooth activation function (`"silu"`, `"gelu"`, `"tanh"`, `"sigmoid"`, `"relu"`, `"softplus"`).
- `n_epochs`: Total number of training epochs.
- `batch_size`: Mini-batch size sampled from simulated ergodic state paths.
- `lr`: Adam optimizer learning rate.
- `trajectory_length`: Simulation horizon for ergodic path sampling.
- `resimulate_every`: Frequency (in epochs) at which state trajectories are re-simulated.
- `grad_clip`: Maximum gradient norm threshold for Adam optimization.

---

## 6. Result Interface & Diagnostic Validation

`DeepMacroSolution` encapsulates the trained neural network, training convergence history, and out-of-sample diagnostic validation:

### Dataclass Attributes
- `model`: Reference to the underlying `DeepMacroModel`.
- `mlp`: Trained `DeepMacroMLP` policy network.
- `loss_history`: NumPy array recording the training loss per epoch.
- `test_euler_mse`: Out-of-sample Mean Squared Euler Error on independent test trajectories ($\text{MSE} < 10^{-3}$).
- `test_euler_max`: Maximum absolute Euler equation residual across all test coordinates.
- `test_trajectory`: Array of shape $(T_{\text{test}}, D)$ storing the out-of-sample simulated path.
- `elapsed_time`: Wall-clock training runtime in seconds.
- `converged`: Boolean flag indicating whether $\text{MSE} < 10^{-3}$.

### Policy Evaluation & Simulation Methods
- `solution.policy(s) -> np.ndarray`: Evaluates the continuous policy function $c(s)$ at scalar state vector or batched state matrix $s$, guaranteeing physical viability ($0 < c_i < W_i$).
- `solution.simulate(s0=None, periods=300, seed=123) -> dict`: Simulates forward economic trajectories, returning `{"states", "controls", "euler_residuals", "physically_viable"}`.
- `solution.summary() -> pd.DataFrame`: Summary table displaying network architecture, training epochs, final loss, out-of-sample MSE, and execution time.
- `solution.to_frame() -> pd.DataFrame`: Alias returning the summary table.
- `solution.to_markdown(digits=4) -> str`: Formatted Markdown table.
- `solution.to_latex(digits=4) -> str`: Publication-ready LaTeX tabular environment.
- `solution.to_typst(digits=4) -> str`: Formatted Typst table.
- `solution.plot(figsize=(14, 4.5)) -> matplotlib.figure.Figure`: Multi-panel diagnostic figure visualizing training loss convergence, out-of-sample Euler residuals across countries, and simulated capital accumulation paths.

---

## References

- Coleman, W. J. (1990). "Solving the stochastic growth model by policy-function iteration." *Journal of Business & Economic Statistics*, 8(1), 27–29.
- Judd, K. L., Maliar, L., Maliar, S., & Valero, R. (2014). "Smolyak method for solving dynamic economic models: Lagrange interpolation, anisotropic grid and adaptive domain." *Journal of Economic Dynamics and Control*, 44, 92–123.
- Maliar, L., Maliar, S., & Winant, P. (2021). "Deep learning for solving dynamic economic models." *Journal of Monetary Economics*, 122, 76–101.
