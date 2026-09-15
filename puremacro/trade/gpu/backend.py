"""Unified GPU & Apple Silicon Acceleration Backend for CGE Trade Models.

This module provides device detection, hardware memory tracking, device context
management, and seamless tensor conversion helpers across NVIDIA CUDA, Apple
Silicon Metal Performance Shaders (MPS), Apple Silicon MLX (Unified Memory Architecture),
and CPU fallbacks.

PyTorch and MLX are optional, heavy dependencies (importing torch costs roughly a
second and 175 MB of RSS). They are therefore never imported at module import
time: availability is probed with :func:`importlib.util.find_spec` and the
packages are loaded lazily, on first use, through :func:`_load_torch` and
:func:`_load_mlx`. ``import puremacro.trade`` stays a NumPy/SciPy-only import.
"""
from __future__ import annotations

import contextlib
import functools
import importlib.util
import platform
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Generator, Literal

import numpy as np

#: Device / backend strings accepted by :func:`select_compute_device`.
DEVICE_STRINGS: tuple[str, ...] = (
    "auto", "cuda", "cuda:N", "mps", "mlx", "gpu", "cpu", "torch", "numpy",
)


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware device specification and capabilities.

    Notes
    -----
    On Apple Silicon Unified Memory Architecture (UMA):
    - Metal GPU ALUs strictly execute in float32/bfloat16.
    - Double-precision (float64) operations in Apple MLX execute via the CPU
      stream (`stream=mx.cpu`) using Apple Accelerate LAPACK routines with zero
      memory copy overhead, sharing the same physical LPDDR5 unified RAM pool.
    """
    backend: Literal["torch", "mlx", "numpy"]
    device: str
    device_name: str
    total_memory_gb: float | None = None
    supports_float64: bool = True
    is_uma: bool = False

    def __repr__(self) -> str:
        uma_str = ", UMA" if self.is_uma else ""
        mem_str = f", {self.total_memory_gb:.1f} GB" if self.total_memory_gb else ""
        if self.backend == "mlx":
            f64_str = "f32-gpu/f64-cpu-stream" if self.device == "gpu" else "f64"
        else:
            f64_str = "f64" if self.supports_float64 else "f32-only"
        return f"<DeviceInfo: {self.backend}:{self.device} ({self.device_name}{mem_str}, {f64_str}{uma_str})>"


# ---------------------------------------------------------------------------
# Lazy accelerator loading
# ---------------------------------------------------------------------------

def has_torch() -> bool:
    """Return True if PyTorch is installed (cheap: does not import it)."""
    return importlib.util.find_spec("torch") is not None


def has_mlx() -> bool:
    """Return True if Apple MLX is installed (cheap: does not import it)."""
    return importlib.util.find_spec("mlx") is not None


@functools.lru_cache(maxsize=None)
def _load_torch() -> Any | None:
    """Import and return ``torch`` on first use; None when absent or broken.

    A broken installation (e.g. an ``OSError`` from a missing shared library)
    is reported once as a RuntimeWarning and then treated as unavailable.
    """
    if not has_torch():
        return None
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the local install
        warnings.warn(
            f"PyTorch is installed but failed to import ({exc!r}); treating it as unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return torch


@functools.lru_cache(maxsize=None)
def _load_mlx() -> Any | None:
    """Import and return ``mlx.core`` on first use; None when absent or broken."""
    if not has_mlx():
        return None
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - depends on the local install
        warnings.warn(
            f"Apple MLX is installed but failed to import ({exc!r}); treating it as unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return mx


def _require_torch(what: str) -> Any:
    torch = _load_torch()
    if torch is None:
        raise RuntimeError(f"Device '{what}' requested but PyTorch is not installed.")
    return torch


def _require_mlx() -> Any:
    mx = _load_mlx()
    if mx is None:
        raise RuntimeError("Apple MLX requested but 'mlx' package is not installed.")
    return mx


def _mps_available(torch: Any) -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _is_torch_tensor(obj: Any) -> bool:
    """True if ``obj`` is a torch.Tensor, without importing torch to find out."""
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, torch.Tensor)


def _is_mlx_array(obj: Any) -> bool:
    """True if ``obj`` is an mlx.core.array, without importing mlx to find out."""
    mx = sys.modules.get("mlx.core")
    return mx is not None and isinstance(obj, mx.array)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_compute_device(preferred: str | None = None) -> tuple[str, str]:
    """Select compute backend and device string.

    Parameters
    ----------
    preferred : str | None
        Preferred device or backend: 'auto', 'cuda', 'cuda:N', 'mps', 'mlx',
        'gpu', 'cpu', 'torch' or 'numpy'.
        If 'auto' or None, automatically selects the best available hardware:
        CUDA -> MPS -> MLX -> PyTorch CPU -> NumPy.
        'cpu' resolves to PyTorch on the CPU when PyTorch is installed and to
        the NumPy serial evaluator otherwise; 'gpu' is the best available GPU
        of any backend; 'numpy' always selects the NumPy evaluator.

    Returns
    -------
    tuple[str, str]
        (backend, device_str) where backend is 'torch', 'mlx', or 'numpy',
        and device_str is 'cuda', 'cuda:N', 'mps', 'gpu', or 'cpu'.

    Raises
    ------
    ValueError
        If ``preferred`` is not one of the documented strings.
    RuntimeError
        If an explicitly requested device or backend is not available.
    """
    pref = (preferred or "auto").lower().strip()

    # Explicit MLX request
    if pref in ("mlx", "apple_mlx"):
        _require_mlx()
        return ("mlx", "gpu")

    if pref == "numpy":
        return ("numpy", "cpu")

    if pref == "cpu":
        return ("torch", "cpu") if _load_torch() is not None else ("numpy", "cpu")

    if pref == "cuda" or pref.startswith("cuda:"):
        torch = _require_torch(pref)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        if pref.startswith("cuda:"):
            idx_str = pref[len("cuda:"):]
            if not idx_str.isdigit() or int(idx_str) >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device '{pref}' requested but only {torch.cuda.device_count()} device(s) are available."
                )
        return ("torch", pref)

    if pref == "mps":
        torch = _require_torch(pref)
        if not _mps_available(torch):
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")
        return ("torch", "mps")

    if pref in ("torch", "pytorch"):
        torch = _load_torch()
        if torch is None:
            raise RuntimeError("PyTorch requested but not installed.")
        if torch.cuda.is_available():
            return ("torch", "cuda")
        if _mps_available(torch):
            return ("torch", "mps")
        return ("torch", "cpu")

    if pref == "gpu":
        torch = _load_torch()
        if torch is not None and torch.cuda.is_available():
            return ("torch", "cuda")
        if torch is not None and _mps_available(torch):
            return ("torch", "mps")
        if has_mlx():
            return ("mlx", "gpu")
        raise RuntimeError("A GPU was requested but neither CUDA, MPS nor MLX is available.")

    if pref != "auto":
        raise ValueError(
            f"Unknown device/backend {preferred!r}; expected one of {DEVICE_STRINGS}."
        )

    # 'auto' mode: CUDA -> MPS -> MLX -> torch CPU -> NumPy
    torch = _load_torch()
    if torch is not None and torch.cuda.is_available():
        return ("torch", "cuda")

    if torch is not None and _mps_available(torch):
        return ("torch", "mps")

    if has_mlx():
        return ("mlx", "gpu")

    if torch is not None:
        return ("torch", "cpu")

    return ("numpy", "cpu")


def _resolve_backend_device(device: str | None, backend: str | None) -> tuple[str, str]:
    """Resolve a validated ``(backend, device)`` pair from a device/backend request.

    This is the single place where the solvers and the batched evaluator turn
    the user's ``device`` and ``backend`` arguments into a concrete pair, so the
    device string is never re-parsed (``'gpu'`` is the MLX device, not a request
    for "any GPU").
    """
    dev = None if device is None else str(device).lower().strip()
    be = None if backend is None else str(backend).lower().strip()
    if dev == "auto":
        dev = None
    if be == "auto":
        be = None

    if be is None:
        return select_compute_device(dev)

    if be in ("torch", "pytorch"):
        if dev is None:
            return select_compute_device("torch")
        sel_backend, sel_device = select_compute_device(dev)
        if sel_backend != "torch":
            raise ValueError(f"device={device!r} is not a PyTorch device (expected 'cuda', 'cuda:N', 'mps' or 'cpu').")
        return (sel_backend, sel_device)

    if be in ("mlx", "apple_mlx"):
        _require_mlx()
        if dev in (None, "mlx", "gpu"):
            return ("mlx", "gpu")
        if dev == "cpu":
            return ("mlx", "cpu")
        raise ValueError(f"device={device!r} is not an MLX device (expected 'gpu' or 'cpu').")

    if be == "numpy":
        if dev not in (None, "cpu"):
            raise ValueError(f"device={device!r} is not available for the NumPy backend (expected 'cpu').")
        return ("numpy", "cpu")

    raise ValueError(f"Unknown backend {backend!r}; expected 'torch', 'mlx', 'numpy' or None.")


def _device_info(backend: str, dev: str) -> DeviceInfo:
    """Build the :class:`DeviceInfo` for an already-resolved ``(backend, device)`` pair."""
    if backend == "torch":
        torch = _require_torch(dev)
        if dev == "cuda" or dev.startswith("cuda:"):
            dev_idx = int(dev[len("cuda:"):]) if dev.startswith("cuda:") else torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_mem = props.total_memory / (1024**3)
            return DeviceInfo(
                backend="torch",
                device=dev,
                device_name=dev_name,
                total_memory_gb=total_mem,
                supports_float64=True,
                is_uma=False,
            )
        elif dev == "mps":
            # Apple Silicon Metal Performance Shaders
            # Metal GPUs share unified memory with host CPU
            soc_name = platform.processor() or "Apple Silicon"
            return DeviceInfo(
                backend="torch",
                device="mps",
                device_name=f"Apple Metal ({soc_name})",
                total_memory_gb=_host_memory_gb(),
                supports_float64=False,  # MPS framework does not support float64
                is_uma=True,
            )
        else:
            return DeviceInfo(
                backend="torch",
                device="cpu",
                device_name=platform.processor() or "Host CPU",
                total_memory_gb=None,
                supports_float64=True,
                is_uma=False,
            )

    elif backend == "mlx":
        # Apple MLX Native
        soc_name = platform.processor() or "Apple Silicon"
        return DeviceInfo(
            backend="mlx",
            device=dev,
            device_name=f"Apple MLX ({soc_name})",
            total_memory_gb=_host_memory_gb(),
            supports_float64=(dev != "gpu"),  # Metal GPU lacks hardware float64; MLX CPU supports float64
            is_uma=True,
        )

    return DeviceInfo(
        backend="numpy",
        device="cpu",
        device_name=platform.processor() or "Host CPU",
        total_memory_gb=None,
        supports_float64=True,
        is_uma=False,
    )


def _host_memory_gb() -> float | None:
    """Total physical memory of an Apple Silicon host in GB (None if unknown)."""
    if platform.system() != "Darwin":
        return None
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return float(out) / (1024**3)
    except Exception:
        return None


def detect_device(preferred: str | None = None) -> DeviceInfo:
    """Inspect and return detailed capabilities of the selected hardware accelerator.

    Parameters
    ----------
    preferred : str | None
        Optional device preference; see :func:`select_compute_device`.

    Returns
    -------
    DeviceInfo
        Structured hardware capabilities dataclass.
    """
    backend, dev = select_compute_device(preferred)
    return _device_info(backend, dev)


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def get_memory_usage(
    device: str | None = None,
    backend: str | None = None,
) -> dict[str, float | str]:
    """Report current allocated and peak memory usage across hardware backends.

    Returns
    -------
    dict[str, float | str]
        Dictionary with 'allocated_mb', 'peak_mb', 'cache_mb' (if applicable),
        and 'device'.
    """
    if backend is None or device is None:
        sel_backend, sel_device = select_compute_device(device)
        backend = backend or sel_backend
        device = device or sel_device

    info: dict[str, float | str] = {"backend": backend, "device": device}

    # Only report accelerator counters for a library that is already loaded:
    # nothing has been allocated on a device whose library was never imported.
    torch = sys.modules.get("torch") if backend == "torch" else None
    mx = sys.modules.get("mlx.core") if backend == "mlx" else None

    if torch is not None:
        if (device == "cuda" or device.startswith("cuda:")) and torch.cuda.is_available():
            info["allocated_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
            info["peak_mb"] = float(torch.cuda.max_memory_allocated() / (1024**2))
            info["cache_mb"] = float(torch.cuda.memory_reserved() / (1024**2))
            return info
        elif device == "mps" and hasattr(torch, "mps"):
            try:
                info["allocated_mb"] = float(torch.mps.current_allocated_memory() / (1024**2))
                info["driver_mb"] = float(torch.mps.driver_allocated_memory() / (1024**2))
                info["peak_mb"] = info["allocated_mb"]
                return info
            except Exception:
                pass

    elif mx is not None:
        try:
            get_act = getattr(mx, "get_active_memory", getattr(getattr(mx, "metal", None), "get_active_memory", lambda: 0))
            get_peak = getattr(mx, "get_peak_memory", getattr(getattr(mx, "metal", None), "get_peak_memory", lambda: 0))
            get_cache = getattr(mx, "get_cache_memory", getattr(getattr(mx, "metal", None), "get_cache_memory", lambda: 0))
            info["allocated_mb"] = float(get_act() / (1024**2))
            info["peak_mb"] = float(get_peak() / (1024**2))
            info["cache_mb"] = float(get_cache() / (1024**2))
            return info
        except Exception:
            pass

    # CPU / OS level fallback
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes
        scale = (1024**2) if platform.system() == "Darwin" else 1024.0
        info["allocated_mb"] = float(usage.ru_maxrss / scale)
        info["peak_mb"] = info["allocated_mb"]
    except Exception:
        info["allocated_mb"] = 0.0
        info["peak_mb"] = 0.0

    return info


def reset_peak_memory(
    device: str | None = None,
    backend: str | None = None,
) -> None:
    """Reset peak memory tracking counters on the active device.

    A library that has not been imported yet has nothing to reset, so this never
    triggers the (slow) first import of torch or mlx.
    """
    if backend is None or device is None:
        sel_backend, sel_device = select_compute_device(device)
        backend = backend or sel_backend
        device = device or sel_device

    torch = sys.modules.get("torch") if backend == "torch" else None
    mx = sys.modules.get("mlx.core") if backend == "mlx" else None

    if torch is not None:
        if (device == "cuda" or device.startswith("cuda:")) and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    elif mx is not None:
        try:
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
                mx.metal.reset_peak_memory()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass


@contextlib.contextmanager
def device_context(
    device: str | None = None,
    backend: str | None = None,
) -> Generator[tuple[str, str], None, None]:
    """Context manager setting active device and cleaning up caches on exit.

    Usage
    -----
    with device_context("mps") as (backend, dev):
        # Work with device
        pass
    """
    sel_backend, sel_device = select_compute_device(device)
    curr_backend = backend or sel_backend
    curr_device = device or sel_device

    try:
        yield (curr_backend, curr_device)
    finally:
        reset_peak_memory(curr_device, curr_backend)


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def to_tensor(
    array: Any,
    device: str | None = None,
    dtype: Any = None,
    backend: str = "torch",
) -> Any:
    """Convert numpy array, scalar, or tensor to the target backend and device.

    Parameters
    ----------
    array : Any
        Input array or tensor.
    device : str | None
        Target device ('cuda', 'mps', 'cpu', 'gpu').
    dtype : Any
        Target dtype (e.g. torch.float32, torch.float64, mx.float32).
    backend : str
        Target backend ('torch' or 'mlx').

    Returns
    -------
    Any
        Converted tensor or array.
    """
    if backend == "torch":
        torch = _load_torch()
        if torch is None:
            raise RuntimeError("PyTorch is not available.")
        if _is_torch_tensor(array):
            t = array
        elif _is_mlx_array(array):
            t = torch.from_numpy(np.asarray(array))
        elif isinstance(array, np.ndarray):
            t = torch.from_numpy(array)
        else:
            t = torch.tensor(array)

        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t

    elif backend == "mlx":
        mx = _load_mlx()
        if mx is None:
            raise RuntimeError("Apple MLX is not available.")
        if _is_mlx_array(array):
            m = array
        elif _is_torch_tensor(array):
            m = mx.array(array.detach().cpu().numpy())
        elif isinstance(array, np.ndarray):
            m = mx.array(array)
        else:
            m = mx.array(np.asarray(array))

        if dtype is not None:
            m = m.astype(dtype)
        return m

    else:
        # numpy fallback
        return to_numpy(array)


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor from PyTorch, Apple MLX, or NumPy to a NumPy array.

    Parameters
    ----------
    tensor : Any
        PyTorch Tensor, MLX array, or NumPy array.

    Returns
    -------
    np.ndarray
        Standard NumPy ndarray.
    """
    if isinstance(tensor, np.ndarray):
        return tensor

    if _is_torch_tensor(tensor):
        return tensor.detach().cpu().numpy()

    # MLX arrays implement __array__ (forcing evaluation); np.asarray copies only
    # when it must, unlike np.array(..., copy=False) which raises under NumPy 2.
    return np.asarray(tensor)
