"""Unified GPU & Apple Silicon Acceleration Backend for CGE Trade Models.

This module provides device detection, hardware memory tracking, device context
management, and seamless tensor conversion helpers across NVIDIA CUDA, Apple
Silicon Metal Performance Shaders (MPS), Apple Silicon MLX (Unified Memory Architecture),
and CPU fallbacks.
"""
from __future__ import annotations

import contextlib
import os
import platform
from dataclasses import dataclass
from typing import Any, Generator, Literal

import numpy as np

# Optional imports for hardware acceleration
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    _HAS_MLX = False


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


def has_torch() -> bool:
    """Return True if PyTorch is available."""
    return _HAS_TORCH


def has_mlx() -> bool:
    """Return True if Apple MLX is available."""
    return _HAS_MLX


def select_compute_device(preferred: str | None = None) -> tuple[str, str]:
    """Select compute backend and device string.

    Parameters
    ----------
    preferred : str | None
        Preferred device or backend: 'auto', 'cuda', 'mps', 'mlx', 'cpu', 'torch'.
        If 'auto' or None, automatically selects best available hardware:
        CUDA -> MPS -> MLX -> CPU.

    Returns
    -------
    tuple[str, str]
        (backend, device_str) where backend is 'torch', 'mlx', or 'numpy',
        and device_str is 'cuda', 'mps', 'gpu', or 'cpu'.
    """
    pref = (preferred or "auto").lower().strip()

    # Explicit MLX request
    if pref in ("mlx", "apple_mlx"):
        if _HAS_MLX:
            return ("mlx", "gpu")
        raise RuntimeError("Apple MLX requested but 'mlx' package is not installed.")

    # Explicit PyTorch device requests
    if pref in ("cuda", "mps", "cpu"):
        if not _HAS_TORCH:
            raise RuntimeError(f"Device '{pref}' requested but PyTorch is not installed.")
        if pref == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        if pref == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")
        return ("torch", pref)

    if pref in ("torch", "pytorch"):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch requested but not installed.")
        if torch.cuda.is_available():
            return ("torch", "cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return ("torch", "mps")
        return ("torch", "cpu")

    # 'auto' mode: CUDA -> MPS -> MLX -> CPU
    if _HAS_TORCH and torch.cuda.is_available():
        return ("torch", "cuda")

    if _HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ("torch", "mps")

    if _HAS_MLX:
        return ("mlx", "gpu")

    if _HAS_TORCH:
        return ("torch", "cpu")

    return ("numpy", "cpu")


def detect_device(preferred: str | None = None) -> DeviceInfo:
    """Inspect and return detailed capabilities of the selected hardware accelerator.

    Parameters
    ----------
    preferred : str | None
        Optional device preference ('auto', 'cuda', 'mps', 'mlx', 'cpu').

    Returns
    -------
    DeviceInfo
        Structured hardware capabilities dataclass.
    """
    backend, dev = select_compute_device(preferred)

    if backend == "torch":
        if dev == "cuda":
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_mem = props.total_memory / (1024**3)
            return DeviceInfo(
                backend="torch",
                device="cuda",
                device_name=dev_name,
                total_memory_gb=total_mem,
                supports_float64=True,
                is_uma=False,
            )
        elif dev == "mps":
            # Apple Silicon Metal Performance Shaders
            # Metal GPUs share unified memory with host CPU
            import subprocess
            soc_name = platform.processor() or "Apple Silicon"
            total_mem_gb = None
            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                total_mem_gb = float(out) / (1024**3)
            except Exception:
                pass
            return DeviceInfo(
                backend="torch",
                device="mps",
                device_name=f"Apple Metal ({soc_name})",
                total_memory_gb=total_mem_gb,
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
        total_mem_gb = None
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            total_mem_gb = float(out) / (1024**3)
        except Exception:
            pass
        return DeviceInfo(
            backend="mlx",
            device=dev,
            device_name=f"Apple MLX ({soc_name})",
            total_memory_gb=total_mem_gb,
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

    if backend == "torch" and _HAS_TORCH:
        if device == "cuda" and torch.cuda.is_available():
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

    elif backend == "mlx" and _HAS_MLX:
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
    """Reset peak memory tracking counters on the active device."""
    if backend is None or device is None:
        sel_backend, sel_device = select_compute_device(device)
        backend = backend or sel_backend
        device = device or sel_device

    if backend == "torch" and _HAS_TORCH:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    elif backend == "mlx" and _HAS_MLX:
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
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not available.")
        if isinstance(array, torch.Tensor):
            t = array
        elif _HAS_MLX and isinstance(array, mx.array):
            t = torch.from_numpy(np.array(array, copy=False))
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
        if not _HAS_MLX:
            raise RuntimeError("Apple MLX is not available.")
        if isinstance(array, mx.array):
            m = array
        elif _HAS_TORCH and isinstance(array, torch.Tensor):
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

    if _HAS_TORCH and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    if _HAS_MLX and isinstance(tensor, mx.array):
        return np.array(tensor, copy=False)

    return np.asarray(tensor)
