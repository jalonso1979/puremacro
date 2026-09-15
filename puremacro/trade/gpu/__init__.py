"""GPU and Apple Silicon MLX Acceleration Engine for puremacro.trade.

Provides hardware-accelerated general equilibrium modeling for multi-country multi-sector
computable general equilibrium (CGE) trade models:

- Dual-device PyTorch (NVIDIA CUDA / Apple Silicon MPS), Apple MLX and NumPy execution.
- Batched parallel Jacobian evaluation across the macro dimension B = 3*nc - 1 (reduced
  layout using factor return equivalence r_c == w_c, verified against the calibration)
  or B = 4*nc - 1 (full layout); 230 / 307 for the canonical 77 countries.
- One-time pre-inversion of Leontief operators (I - B^T) and (I - A) converting serial
  column perturbations into vectorized multi-RHS GEMM.
- Vectorized tensor contractions (einsum) for bilateral trade flows.
- Forward-mode (JVP) and reverse-mode (VJP) automatic differentiation for small-economy
  numerical stability.
- Levenberg-Marquardt regularized damping with two-sided equilibration and Armijo line search.
- Adaptive homotopy continuation along tariff parameter lambda in [0, 1].

Jacobians are evaluated in float64 wherever the device supports it; float32-only devices
(Apple MPS, MLX GPU stream) are used only for the coarse phase when requested explicitly.
This subpackage is outside the four-package Pyodide contract: torch and mlx are optional
and imported lazily on first use, so ``import puremacro.trade`` never loads them.
"""
from __future__ import annotations

from puremacro.trade.gpu.backend import (
    DeviceInfo,
    detect_device,
    device_context,
    get_memory_usage,
    has_mlx,
    has_torch,
    reset_peak_memory,
    select_compute_device,
    to_numpy,
    to_tensor,
)
from puremacro.trade.gpu.batched_jacobian import BatchedJacobianEvaluator
from puremacro.trade.gpu.homotopy import solve_homotopy_continuation
from puremacro.trade.gpu.mlx_solver import solve_trade_equilibrium_mlx
from puremacro.trade.gpu.solver_gpu import solve_trade_equilibrium_gpu

__all__ = [
    # Hardware & backend utilities
    "DeviceInfo",
    "detect_device",
    "select_compute_device",
    "device_context",
    "to_tensor",
    "to_numpy",
    "get_memory_usage",
    "reset_peak_memory",
    "has_torch",
    "has_mlx",
    # Evaluators
    "BatchedJacobianEvaluator",
    # Solvers
    "solve_trade_equilibrium_gpu",
    "solve_trade_equilibrium_mlx",
    # Homotopy continuation
    "solve_homotopy_continuation",
]
