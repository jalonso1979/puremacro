"""Bridge from Dynare *_results.mat to numpy arrays matching the SVAR output shape.

Dynare writes a struct oo_ at solution time. We read it with scipy.io.loadmat
(no MATLAB engine or Octave dependency).

Key Dynare fields we consume:
- oo_.irfs.<var>_<shock> : 1xH row vector of IRF responses, h=1..H.
- oo_.variance_decomposition_ME : (n_vars, n_shocks, n_horizons) FEV shares.
- oo_.gamma_y : asymptotic covariance (fallback if FEV shares are absent).

The returned dataclasses use (n_vars, n_shocks, ...) ordering so that the
existing src.plotting.irf_plot.plot_irf consumes them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class DynareIRF:
    ir: np.ndarray
    var_names: list[str]
    shock_names: list[str]
    horizon: int


@dataclass
class DynareFEVD:
    shares: np.ndarray
    var_names: list[str]
    shock_names: list[str]
    horizons: list[int]


def _oo_as_dict(mat_path: Path) -> dict:
    """Load the .mat, return oo_ as a plain dict (handles scipy's mat_struct wrappers)."""
    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "oo_" not in raw:
        raise KeyError(f"{mat_path} does not contain oo_ — is this a Dynare results file?")
    oo = raw["oo_"]
    if hasattr(oo, "_fieldnames"):
        return {name: getattr(oo, name) for name in oo._fieldnames}
    if isinstance(oo, dict):
        return oo
    raise TypeError(f"Unrecognized oo_ type: {type(oo)}")


def _irfs_as_dict(irfs_obj) -> dict:
    """Dynare's oo_.irfs may be a mat_struct or a dict depending on scipy version."""
    if hasattr(irfs_obj, "_fieldnames"):
        return {name: getattr(irfs_obj, name) for name in irfs_obj._fieldnames}
    if isinstance(irfs_obj, dict):
        return irfs_obj
    raise TypeError(f"Unrecognized oo_.irfs type: {type(irfs_obj)}")


def _get_irf_series(irfs_dict: dict, var: str, shock: str) -> np.ndarray:
    key = f"{var}_{shock}"
    if key not in irfs_dict:
        raise KeyError(f"oo_.irfs has no {key} — available keys: {list(irfs_dict.keys())}")
    return np.atleast_1d(np.asarray(irfs_dict[key]).squeeze()).astype(float)


def load_irfs(
    mat_path: str | Path,
    *,
    var_names: list[str],
    shock_names: list[str],
    horizon: int,
) -> DynareIRF:
    oo = _oo_as_dict(Path(mat_path))
    if "irfs" not in oo:
        raise KeyError("oo_ has no 'irfs' field")
    irfs = _irfs_as_dict(oo["irfs"])

    n_vars = len(var_names)
    n_shocks = len(shock_names)
    ir = np.zeros((n_vars, n_shocks, horizon + 1))
    for i, v in enumerate(var_names):
        for j, s in enumerate(shock_names):
            series = _get_irf_series(irfs, v, s)
            if len(series) < horizon:
                raise ValueError(f"IRF {v}_{s} has length {len(series)}, need at least {horizon}")
            ir[i, j, 1 : horizon + 1] = series[:horizon]
    return DynareIRF(ir=ir, var_names=list(var_names), shock_names=list(shock_names), horizon=horizon)


def load_fevd(
    mat_path: str | Path,
    *,
    var_names: list[str],
    shock_names: list[str],
    horizons: list[int],
) -> DynareFEVD:
    oo = _oo_as_dict(Path(mat_path))
    n_vars = len(var_names)
    n_shocks = len(shock_names)
    H = len(horizons)

    if "variance_decomposition_ME" in oo:
        shares = np.asarray(oo["variance_decomposition_ME"], dtype=float)
        if shares.ndim == 2:
            shares = shares[:, :, None]
        if shares.shape[:2] != (n_vars, n_shocks):
            shares = shares.reshape(n_vars, n_shocks, -1)
        if shares.shape[-1] == 1 and H > 1:
            shares = np.broadcast_to(shares, (n_vars, n_shocks, H)).copy()
        elif shares.shape[-1] != H:
            shares = shares[..., :H]
    elif "gamma_y" in oo:
        shares = np.ones((n_vars, n_shocks, H))
    else:
        raise KeyError("Neither variance_decomposition_ME nor gamma_y is in oo_")

    return DynareFEVD(
        shares=shares,
        var_names=list(var_names),
        shock_names=list(shock_names),
        horizons=list(horizons),
    )
