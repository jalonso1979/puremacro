"""Shared pluggable compute backends for puremacro's compute-heavy modules.

Four tiers:
  - "numpy" : reference implementation and correctness ORACLE. Always
    available; keeps puremacro's Pyodide-compatible core intact.
  - "numba" : CPU JIT (+ prange). Compiled kernels, NOT an array namespace.
  - "mlx"   : Apple-Silicon GPU. numpy-like namespace (mlx.core); xp-generic
    kernels run on it unchanged. float32-only (Metal has no float64).
  - "cupy"  : NVIDIA GPU. numpy-like namespace (cupy); xp-generic kernels run
    unchanged. Optional [cuda] extra; cannot be installed/run without CUDA.

numba/mlx/cupy are optional extras detected via importlib.util.find_spec and
skipped, never hard-failed. (Promoted from models/nested_dmp/backend.py so the
nested DMP model and the general vfi engine share one source of truth.)
"""
from __future__ import annotations

import importlib.util

import numpy as np

SUPPORTED = ("numpy", "numba", "mlx", "cupy")


def backend_available(name: str) -> bool:
    """True if backend ``name`` can be used in this environment."""
    if name == "numpy":
        return True
    if name in ("numba", "mlx", "cupy"):
        return importlib.util.find_spec(name) is not None
    raise ValueError(f"Unknown backend {name!r}; supported: {SUPPORTED}")


def available_backends() -> tuple[str, ...]:
    """Tuple of installed backends, numpy first."""
    return tuple(b for b in SUPPORTED if backend_available(b))


def get_array_namespace(name: str):
    """Return a numpy-like array module for namespace-style backends.

    "numpy"->numpy; "mlx"->mlx.core; "cupy"->cupy. "numba" is not a namespace.
    """
    if name == "numpy":
        return np
    if name == "mlx":
        if not backend_available("mlx"):
            raise ImportError(
                "mlx not installed; on Apple Silicon run "
                "`pip install puremacro[backend]`."
            )
        import mlx.core as mx

        return mx
    if name == "cupy":
        if not backend_available("cupy"):
            raise ImportError(
                "cupy not installed; on an NVIDIA host run "
                "`pip install puremacro[cuda]` (choose the wheel matching your "
                "CUDA toolkit, e.g. cupy-cuda12x)."
            )
        import cupy as cp

        return cp
    if name == "numba":
        raise ValueError(
            "numba backend uses compiled kernels, not an array namespace; "
            "call the kernels_numba.* functions directly."
        )
    raise ValueError(f"Unknown backend {name!r}; supported: {SUPPORTED}")


def to_numpy(x) -> np.ndarray:
    """Bring an array from any backend back to numpy.

    cupy blocks implicit conversion, so route it through cp.asnumpy; numpy and
    mlx (which forces evaluation) go through np.asarray.
    """
    if type(x).__module__.split(".")[0] == "cupy":
        import cupy as cp

        return cp.asnumpy(x)
    return np.asarray(x)


def njit_fallback(*args, **kwargs):
    """Decorator compiling with numba.njit if numba is available, or no-op fallback."""
    if backend_available("numba"):
        try:
            import numba

            return numba.njit(*args, **kwargs)
        except Exception:
            pass

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(fn):
        return fn

    return decorator


__all__ = [
    "SUPPORTED",
    "backend_available",
    "available_backends",
    "get_array_namespace",
    "to_numpy",
]
