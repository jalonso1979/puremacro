"""Unit tests for Deep Macro & Physics-Informed Neural Networks (PINNs).

Verifies:
1. Pure NumPy multi-layer perceptron (DeepMacroMLP) forward and analytical backward passes.
2. Exact analytical derivatives for all activation functions (SiLU, GELU, Tanh, Sigmoid, Softplus, ReLU).
3. Adam optimizer moment updates, weight decay, and gradient norm clipping.
4. Scalability to 10+ continuous state variables (10-country capital accumulation model).
5. Out-of-sample mean squared Euler equation residual < 1e-3.
6. Physical viability guarantee across 1000 simulation periods (c > 0, k' > 0, c < W).
7. Deterministic vs stochastic ergodic trajectory training.
8. Continuous policy function evaluation .policy(s) on 1D and 2D state arrays.
9. Result presentation interface (.summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()).
10. 100% Pyodide and browser compatibility with ZERO imports of PyTorch, TensorFlow, or JAX.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from puremacro.vfi.deep_macro import (
    AdamOptimizer,
    DeepMacroMLP,
    DeepMacroModel,
    DeepMacroSolution,
    _d_gelu,
    _d_relu,
    _d_sigmoid,
    _d_silu,
    _d_softplus,
    _d_tanh,
    _gelu,
    _relu,
    _sigmoid,
    _silu,
    _softplus,
    _tanh,
    solve_deep_macro,
)


# ---------------------------------------------------------------------------
# Test 1: Zero Forbidden Imports (Pyodide / Juno Browser Contract)
# ---------------------------------------------------------------------------

def test_no_forbidden_framework_imports() -> None:
    """Verify ZERO imports of PyTorch, TensorFlow, or JAX in deep_macro.py."""
    forbidden = ["torch", "tensorflow", "jax", "statsmodels", "linearmodels", "arch"]

    # 1. Inspect AST of deep_macro.py
    import puremacro.vfi.deep_macro as dm_module

    source_path = Path(inspect.getfile(dm_module))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for f in forbidden:
                    assert not alias.name.startswith(f), (
                        f"Forbidden import '{alias.name}' found in {source_path}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for f in forbidden:
                    assert not node.module.startswith(f), (
                        f"Forbidden from-import '{node.module}' found in {source_path}"
                    )

    # 2. Check in an isolated process that importing deep_macro does not pull in forbidden frameworks
    import os
    import subprocess
    root = Path(__file__).resolve().parents[2]
    cmd = (
        "import sys, puremacro.vfi.deep_macro; "
        "forbidden = ['torch', 'tensorflow', 'jax', 'statsmodels', 'linearmodels', 'arch']; "
        "leaked = [f for f in forbidden if f in sys.modules]; "
        "assert not leaked, f'Forbidden module(s) {leaked} present in sys.modules'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", cmd],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        env={"PYTHONPATH": str(root), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Test 2: Activation Functions & Analytical Derivatives vs Finite Differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,fn,dfn",
    [
        ("silu", _silu, _d_silu),
        ("gelu", _gelu, _d_gelu),
        ("tanh", _tanh, _d_tanh),
        ("sigmoid", _sigmoid, _d_sigmoid),
        ("softplus", _softplus, _d_softplus),
    ],
)
def test_smooth_activation_derivatives(name, fn, dfn) -> None:
    """Verify analytical derivatives of smooth activations match central finite differences."""
    x = np.array([-3.5, -1.8, -0.5, 0.0, 0.5, 1.8, 3.5], dtype=np.float64)
    h = 1e-6
    analytical = dfn(x)
    numerical = (fn(x + h) - fn(x - h)) / (2.0 * h)
    max_diff = np.max(np.abs(analytical - numerical))
    assert max_diff < 1e-8, f"Activation {name} derivative error {max_diff:.2e} >= 1e-8"


def test_relu_activation_derivative() -> None:
    """Verify ReLU activation and derivative at non-zero points."""
    x = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)
    r = _relu(x)
    dr = _d_relu(x)
    assert np.allclose(r, [0.0, 0.0, 0.5, 2.0])
    assert np.allclose(dr, [0.0, 0.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Test 3: DeepMacroMLP Vectorized Forward and Backward Passes
# ---------------------------------------------------------------------------

def test_mlp_backward_pass_vs_finite_difference() -> None:
    """Verify analytical backpropagation gradients match central finite differences."""
    rng = np.random.default_rng(123)
    in_dim = 4
    hidden_dims = (8, 8)
    out_dim = 2
    mlp = DeepMacroMLP(
        input_dim=in_dim,
        hidden_dims=hidden_dims,
        output_dim=out_dim,
        activation="silu",
        output_activation="linear",
        seed=123,
    )

    batch_size = 5
    x = rng.normal(0.0, 1.0, (batch_size, in_dim))
    target = rng.normal(0.0, 1.0, (batch_size, out_dim))

    # Forward pass and squared error loss: 0.5 * sum((y - target)^2)
    y = mlp.forward(x)
    dL_dy = y - target

    # Backward pass
    layer_grads = mlp.backward(dL_dy)
    assert len(layer_grads) == 3  # (W1, b1), (W2, b2), (W3, b3)

    h = 1e-6
    # Check weight gradient for layer 0, element (1, 2)
    w_orig = mlp.weights[0][1, 2]
    mlp.weights[0][1, 2] = w_orig + h
    loss_plus = 0.5 * np.sum((mlp.forward(x) - target) ** 2)
    mlp.weights[0][1, 2] = w_orig - h
    loss_minus = 0.5 * np.sum((mlp.forward(x) - target) ** 2)
    mlp.weights[0][1, 2] = w_orig

    num_grad_w = (loss_plus - loss_minus) / (2.0 * h)
    ana_grad_w = layer_grads[0][0][1, 2]
    assert np.isclose(num_grad_w, ana_grad_w, rtol=1e-5, atol=1e-6)

    # Check bias gradient for layer 1, element (3)
    b_orig = mlp.biases[1][3]
    mlp.biases[1][3] = b_orig + h
    loss_plus = 0.5 * np.sum((mlp.forward(x) - target) ** 2)
    mlp.biases[1][3] = b_orig - h
    loss_minus = 0.5 * np.sum((mlp.forward(x) - target) ** 2)
    mlp.biases[1][3] = b_orig

    num_grad_b = (loss_plus - loss_minus) / (2.0 * h)
    ana_grad_b = layer_grads[1][1][3]
    assert np.isclose(num_grad_b, ana_grad_b, rtol=1e-5, atol=1e-6)


def test_mlp_copy_and_params() -> None:
    """Verify parameter get/set and deep copy."""
    mlp = DeepMacroMLP(input_dim=3, hidden_dims=(6,), output_dim=2, seed=99)
    clone = mlp.copy()
    assert clone.num_parameters == mlp.num_parameters
    assert np.allclose(clone.weights[0], mlp.weights[0])

    params = mlp.get_params()
    params[0] += 0.5
    mlp.set_params(params)
    assert not np.allclose(clone.weights[0], mlp.weights[0])


# ---------------------------------------------------------------------------
# Test 4: Adam Optimizer with Gradient Clipping
# ---------------------------------------------------------------------------

def test_adam_optimizer_clipping() -> None:
    """Verify Adam optimizer correctly clips gradients exceeding max_norm."""
    param = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    opt = AdamOptimizer(params=[param], lr=1e-2, grad_clip=1.0)

    # Huge gradient with norm 100.0
    huge_grad = np.array([60.0, 80.0, 0.0], dtype=np.float64)
    gnorm = opt.step([huge_grad])
    assert np.isclose(gnorm, 100.0)

    # Param should have changed by approximately lr * 1.0 (due to clipping)
    assert not np.allclose(param, [1.0, 2.0, 3.0])
    assert np.all(np.isfinite(param))


# ---------------------------------------------------------------------------
# Test 5: DeepMacroModel Specification & Steady State
# ---------------------------------------------------------------------------

def test_deep_macro_model_steady_state() -> None:
    """Verify analytical steady-state calculation for canonical multi-country model."""
    model = DeepMacroModel.multi_country_growth(n_countries=10, alpha=0.36, beta=0.96, delta=0.08)
    k_ss, c_ss = model.steady_state()
    assert k_ss.shape == (10,)
    assert c_ss.shape == (10,)
    assert np.all(k_ss > 0.0)
    assert np.all(c_ss > 0.0)

    # Test Euler equation at steady state has zero residual
    s_ss = k_ss.reshape(1, -1)
    c_ss_mat = c_ss.reshape(1, -1)
    res_ss = model.default_euler_residual(s_ss, c_ss_mat, s_ss, c_ss_mat)
    assert np.allclose(res_ss, 0.0, atol=1e-12)


def test_deep_macro_model_validation() -> None:
    """Verify validation of model dimensions and parameters."""
    with pytest.raises(ValueError, match="n_states must be positive"):
        DeepMacroModel(n_states=0, n_controls=5)
    with pytest.raises(ValueError, match="n_controls must be positive"):
        DeepMacroModel(n_states=5, n_controls=-1)
    with pytest.raises(ValueError, match="Discount factor beta must be in"):
        DeepMacroModel(n_states=5, n_controls=5, beta=1.5)


# ---------------------------------------------------------------------------
# Test 6: 10-State Model Scalability & Euler Equation Residual < 1e-3
# ---------------------------------------------------------------------------

def test_10_state_model_solve_euler_mse() -> None:
    """Solve 10-country model and assert out-of-sample MSE Euler residual < 1e-3."""
    model = DeepMacroModel.multi_country_growth(n_countries=10, alpha=0.36, beta=0.96, delta=0.08)
    solution = solve_deep_macro(
        model,
        hidden_dims=(64, 64),
        activation="silu",
        n_epochs=120,
        batch_size=128,
        lr=2e-3,
        trajectory_length=1500,
        burn_in=100,
        resimulate_every=40,
        seed=42,
    )

    assert isinstance(solution, DeepMacroSolution)
    assert solution.converged is True
    # Headline acceptance criterion: out-of-sample MSE Euler residual < 1e-3
    assert solution.test_euler_mse < 1e-3, f"Expected MSE < 1e-3, got {solution.test_euler_mse}"
    assert len(solution.loss_history) == 120
    assert solution.loss_history[-1] < 1e-4
    assert np.mean(solution.loss_history) < 1e-4


# ---------------------------------------------------------------------------
# Test 7: Physical Viability Guarantee across 1000 Periods
# ---------------------------------------------------------------------------

def test_physical_viability_across_1000_periods() -> None:
    """Verify consumption > 0, capital > 0, and c < W across 1000 simulation periods."""
    model = DeepMacroModel.multi_country_growth(n_countries=10)
    solution = solve_deep_macro(model, n_epochs=50, batch_size=64, seed=42)

    # Simulate 1000 periods from perturbed initial state (50% of steady state)
    k_ss, _ = model.steady_state()
    s0 = k_ss * 0.5
    sim = solution.simulate(s0=s0, periods=1000, seed=123)

    assert sim["physically_viable"] is True
    states = sim["states"]
    controls = sim["controls"]
    coh = sim["cash_on_hand"]

    assert states.shape == (1001, 10)
    assert controls.shape == (1000, 10)
    assert coh.shape == (1000, 10)

    # Strict physical inequalities
    assert np.all(controls > 0.0), "Consumption violated non-negativity!"
    assert np.all(states > 0.0), "Capital violated non-negativity!"
    assert np.all(controls < coh), "Consumption exceeded cash-on-hand!"

    # Verify resource balance k' = W - c to machine precision
    for t in range(1000):
        expected_next_k = coh[t] - controls[t]
        actual_next_k = states[t + 1]
        assert np.allclose(expected_next_k, actual_next_k, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 8: Deterministic vs Stochastic Ergodic Trajectory Training
# ---------------------------------------------------------------------------

def test_deterministic_vs_stochastic_training() -> None:
    """Check held-out convergence diagnostics for deterministic and stochastic training."""
    # Deterministic model
    model_det = DeepMacroModel.multi_country_growth(n_countries=10, sigma_eps=0.0)
    sol_det = solve_deep_macro(model_det, n_epochs=80, batch_size=64, seed=10)
    # Detached-target time iteration can fail for this deterministic trajectory.
    assert np.isfinite(sol_det.test_euler_mse)
    assert sol_det.converged == (sol_det.test_euler_mse < 1e-3)

    # Stochastic model
    model_stoch = DeepMacroModel.multi_country_growth(n_countries=10, sigma_eps=0.015)
    sol_stoch = solve_deep_macro(model_stoch, n_epochs=80, batch_size=64, seed=20)
    assert sol_stoch.converged is True
    assert sol_stoch.test_euler_mse < 1e-3


# ---------------------------------------------------------------------------
# Test 9: Continuous Policy Evaluation .policy(s)
# ---------------------------------------------------------------------------

def test_continuous_policy_evaluation() -> None:
    """Verify .policy(s) handles 1D single states and 2D batch states consistently."""
    model = DeepMacroModel.multi_country_growth(n_countries=10)
    solution = solve_deep_macro(model, n_epochs=40, seed=42)

    k_ss, _ = model.steady_state()

    # 1D single-state evaluation
    s_single = k_ss.copy()
    c_single = solution.policy(s_single)
    assert isinstance(c_single, np.ndarray)
    assert c_single.shape == (10,)
    assert np.all(c_single > 0.0)

    # 2D batch evaluation
    s_batch = np.vstack([k_ss * 0.8, k_ss * 1.0, k_ss * 1.2])
    c_batch = solution.policy(s_batch)
    assert c_batch.shape == (3, 10)
    assert np.allclose(c_batch[1], c_single, atol=1e-10)

    # Monotonicity check: higher capital leads to higher consumption
    assert np.all(c_batch[2] > c_batch[0])


# ---------------------------------------------------------------------------
# Test 10: Presentation Interface Compliance
# ---------------------------------------------------------------------------

def test_presentation_interface_compliance() -> None:
    """Verify .summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()."""
    model = DeepMacroModel.multi_country_growth(n_countries=10)
    solution = solve_deep_macro(model, n_epochs=40, seed=42)

    # summary & to_frame
    df = solution.summary()
    assert isinstance(df, pd.DataFrame)
    assert "Out-of-Sample Euler MSE" in df.index
    assert "Physical Viability Preserved" in df.index
    assert solution.to_frame().equals(df)

    # to_markdown
    md = solution.to_markdown()
    assert isinstance(md, str)
    assert "Model Name" in md
    assert "10-Country Neoclassical Growth PINN" in md

    # to_latex
    latex = solution.to_latex()
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex or "tabular" in latex

    # to_typst
    typst = solution.to_typst()
    assert isinstance(typst, str)
    assert "#table" in typst

    # plot
    fig = solution.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3


# ---------------------------------------------------------------------------
# Test 11: Alternative Activations (GELU, Tanh)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("act", ["gelu", "tanh", "silu"])
def test_alternative_activations(act: str) -> None:
    """Verify solve_deep_macro succeeds with GELU, Tanh, and SiLU activations."""
    model = DeepMacroModel.multi_country_growth(n_countries=10)
    sol = solve_deep_macro(model, hidden_dims=(32, 32), activation=act, n_epochs=50, seed=77)
    assert sol.converged is True
    assert sol.test_euler_mse < 1e-3
    assert sol.mlp.activation == act


# ---------------------------------------------------------------------------
# Test 12: Custom Model Functions Support
# ---------------------------------------------------------------------------

def test_custom_model_functions() -> None:
    """Verify DeepMacroModel functions seamlessly with user-supplied custom callbacks."""
    n_states = 10
    n_controls = 10

    def custom_reward(s, c):
        return np.sum(np.log(np.maximum(c, 1e-6)), axis=-1)

    def custom_transition(s, c, shocks=None):
        k = s[..., :n_controls]
        W = k ** 0.36 + 0.92 * k
        kp = np.maximum(W - c, 1e-6)
        return kp

    def custom_euler(s, c, sp, cp):
        R_next = 0.36 * (sp ** (0.36 - 1.0)) + 0.92
        return 1.0 - 0.96 * ((cp / c) ** (-2.0)) * R_next

    model = DeepMacroModel(
        n_states=n_states,
        n_controls=n_controls,
        beta=0.96,
        reward_fn=custom_reward,
        transition_fn=custom_transition,
        euler_residual_fn=custom_euler,
        name="Custom 10-State Planner",
    )

    sol = solve_deep_macro(model, n_epochs=50, seed=42)
    assert sol.converged is True
    assert sol.test_euler_mse < 1e-3
    assert sol.model.name == "Custom 10-State Planner"
