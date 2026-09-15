"""GPU and Apple Silicon MLX Acceleration Engine for puremacro.trade.

Provides hardware-accelerated general equilibrium modeling for multi-country multi-sector
computable general equilibrium (CGE) trade models:

- Dual-device PyTorch (NVIDIA CUDA / Apple Silicon MPS) and Apple MLX acceleration.
- Batched parallel Jacobian evaluation across batch dimension B = 230 (using factor return
  equivalence r_c == w_c) and B = 307.
- One-time pre-inversion of Leontief operators (I - B^T) and (I - A) converting serial
  column perturbations into vectorized multi-RHS GEMM.
- Vectorized tensor contractions (einsum) for bilateral trade flows.
- Forward-mode automatic differentiation / VJP support for small-economy numerical stability.
- Levenberg-Marquardt regularized damping with two-sided equilibration and Armijo line search.
- Adaptive homotopy continuation along tariff parameter lambda in [0, 1].
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
