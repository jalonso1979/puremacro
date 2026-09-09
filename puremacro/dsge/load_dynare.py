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
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ._results import Dynare2ndDR, DynareDR


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


def _to_plain_dict(obj: Any) -> Any:
    """Recursively convert scipy mat_struct or nested objects to dict."""
    if hasattr(obj, "_fieldnames"):
        return {name: _to_plain_dict(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    return obj


def _extract_name_list(raw_names: Any) -> list[str]:
    """Extract list of clean strings from MATLAB 2D char matrix, object array, or list."""
    if raw_names is None:
        return []
    if isinstance(raw_names, (list, tuple)):
        return [str(x).strip() for x in raw_names]
    arr = np.asarray(raw_names)
    if arr.ndim == 2 and arr.dtype.kind in ("U", "S", "a"):
        return ["".join(row).strip() for row in arr]
    if arr.ndim == 1:
        out = []
        for item in arr:
            if isinstance(item, (np.ndarray, list)):
                out.append("".join(str(c) for c in item).strip())
            else:
                out.append(str(item).strip())
        return out
    return [str(arr).strip()]


def _unfold_ghxx(ghxx_raw: np.ndarray, n_v: int, n_x: int) -> np.ndarray:
    """Unfold second-order tensor from folded symmetric format into (n_v, n_x^2)."""
    ghxx_raw = np.atleast_2d(ghxx_raw)
    if ghxx_raw.shape[1] == n_x * n_x:
        return ghxx_raw
    n_folded = n_x * (n_x + 1) // 2
    if ghxx_raw.shape[1] != n_folded:
        raise ValueError(
            f"ghxx has {ghxx_raw.shape[1]} columns, expected {n_x**2} (unfolded) "
            f"or {n_folded} (folded) for {n_x} states"
        )
    unfolded = np.zeros((n_v, n_x * n_x), dtype=float)
    col = 0
    for i in range(n_x):
        for j in range(i, n_x):
            val = ghxx_raw[:, col]
            unfolded[:, i * n_x + j] = val
            unfolded[:, j * n_x + i] = val
            col += 1
    return unfolded


def load_dynare_dr(
    mat_path_or_dict: str | Path | dict,
    *,
    order: int = 1,
    var_names: Sequence[str] | None = None,
    state_names: Sequence[str] | None = None,
    shock_names: Sequence[str] | None = None,
) -> DynareDR | Dynare2ndDR:
    """Load decision rules from Dynare oo_.dr structure.

    Parameters
    ----------
    mat_path_or_dict : str | Path | dict
        Path to Dynare results .mat file or raw results dictionary.
    order : int, default 1
        Perturbation order: 1 (returns DynareDR) or 2 (returns Dynare2ndDR).
    var_names : sequence of str, optional
        Names of endogenous variables in declaration order.
    state_names : sequence of str, optional
        Names of predetermined state variables.
    shock_names : sequence of str, optional
        Names of exogenous shocks.
    """
    if isinstance(mat_path_or_dict, (str, Path)):
        p = Path(mat_path_or_dict)
        if not p.exists():
            raise FileNotFoundError(f"Dynare results file not found: {p}")
        raw = loadmat(str(p), squeeze_me=True, struct_as_record=False)
    elif isinstance(mat_path_or_dict, dict):
        raw = mat_path_or_dict
    else:
        raise TypeError(f"Expected str, Path, or dict, got {type(mat_path_or_dict)}")

    if "oo_" not in raw:
        raise KeyError(f"'{mat_path_or_dict}' does not contain 'oo_' structure (is this a Dynare results file?)")

    oo = _to_plain_dict(raw["oo_"])
    if "dr" not in oo or oo["dr"] is None:
        raise KeyError(f"'{mat_path_or_dict}' does not contain 'oo_.dr' structure")
    dr = _to_plain_dict(oo["dr"])

    # Extract M_ metadata if available
    M = _to_plain_dict(raw.get("M_", {})) if isinstance(raw, dict) and "M_" in raw else {}

    # Extract ghx and ghu
    if "ghx" not in dr or "ghu" not in dr:
        raise KeyError("oo_.dr missing required decision rule fields 'ghx' or 'ghu'")
    ghx = np.asarray(dr["ghx"], dtype=float)
    if ghx.ndim == 0:
        ghx = ghx.reshape(1, 1)
    elif ghx.ndim == 1:
        ghx = ghx[:, None]
    n_v, n_x = ghx.shape

    ghu = np.asarray(dr["ghu"], dtype=float)
    if ghu.ndim == 0:
        ghu = ghu.reshape(1, 1)
    elif ghu.ndim == 1:
        if ghu.shape[0] == n_v:
            ghu = ghu[:, None]
        else:
            ghu = ghu[None, :]
    _, n_u = ghu.shape

    # Extract order_var and unpermute from DR order to declaration order
    order_var = dr.get("order_var", None)
    if order_var is not None:
        ov = np.asarray(order_var, dtype=int).ravel()
        if len(ov) == n_v:
            p_idx = ov - 1 if np.min(ov) >= 1 else ov
            inv_p = np.argsort(p_idx)
            ghx = ghx[inv_p, :]
            ghu = ghu[inv_p, :]

    # Steady state (ys) - in official Dynare, ys is already stored in declaration order
    ys_raw = dr.get("ys", None)
    if ys_raw is None:
        ys_raw = oo.get("steady_state", np.zeros(n_v))
    ys = np.asarray(ys_raw, dtype=float).ravel()
    if len(ys) != n_v:
        ys = np.resize(ys, n_v)

    # Name resolution hierarchy
    if var_names is not None:
        v_names = tuple(var_names)
    elif "endo_names" in M and M["endo_names"] is not None and np.size(M["endo_names"]) > 0:
        v_names = tuple(_extract_name_list(M["endo_names"]))
    else:
        v_names = tuple(f"y{i+1}" for i in range(n_v))

    if state_names is not None:
        s_names = tuple(state_names)
    elif "state_var" in M and M["state_var"] is not None and np.size(M["state_var"]) > 0:
        s_idx = np.asarray(M["state_var"]).ravel().astype(int) - 1
        s_names = tuple(v_names[i] for i in s_idx if 0 <= i < len(v_names))
    else:
        s_names = tuple(f"y{i+1}" for i in range(n_x))

    if shock_names is not None:
        e_names = tuple(shock_names)
    elif "exo_names" in M and M["exo_names"] is not None and np.size(M["exo_names"]) > 0:
        e_names = tuple(_extract_name_list(M["exo_names"]))
    else:
        e_names = ("e",) if n_u == 1 else tuple(f"e{i+1}" for i in range(n_u))

    df_ghx = pd.DataFrame(ghx, index=list(v_names), columns=list(s_names))
    df_ghu = pd.DataFrame(ghu, index=list(v_names), columns=list(e_names))
    s_ys = pd.Series(ys, index=list(v_names))

    if order == 1:
        return DynareDR(
            ghx=df_ghx,
            ghu=df_ghu,
            ys=s_ys,
            state_variables=s_names,
            variable_names=v_names,
            shock_names=e_names,
        )

    # Order 2 extraction
    if "ghxx" not in dr or dr["ghxx"] is None:
        raise KeyError(f"oo_.dr has no 'ghxx' field required for order={order}")

    ghxx_raw = np.asarray(dr["ghxx"], dtype=float)
    if ghxx_raw.ndim == 0:
        ghxx_raw = ghxx_raw.reshape(1, 1)
    elif ghxx_raw.ndim == 1:
        if ghxx_raw.shape[0] == n_v:
            ghxx_raw = ghxx_raw[:, None]
        else:
            ghxx_raw = ghxx_raw[None, :]
    if ghxx_raw.shape[1] == n_x * (n_x + 1) // 2 and ghxx_raw.shape[1] != n_x**2:
        ghxx_raw = _unfold_ghxx(ghxx_raw, n_v, n_x)
    if order_var is not None and len(ov) == n_v:
        ghxx_raw = ghxx_raw[inv_p, :]

    # ghs2
    if "ghs2" in dr and dr["ghs2"] is not None:
        ghs2_raw = np.asarray(dr["ghs2"], dtype=float).ravel()
        if len(ghs2_raw) != n_v:
            ghs2_raw = np.resize(ghs2_raw, n_v)
        if order_var is not None and len(ov) == n_v:
            ghs2_raw = ghs2_raw[inv_p]
    else:
        ghs2_raw = np.zeros(n_v)
    s_ghs2 = pd.Series(ghs2_raw, index=list(v_names))

    # ghxu and ghuu
    if "ghxu" in dr and dr["ghxu"] is not None:
        ghxu_raw = np.asarray(dr["ghxu"], dtype=float)
        if ghxu_raw.ndim == 0:
            ghxu_raw = ghxu_raw.reshape(1, 1)
        elif ghxu_raw.ndim == 1:
            if ghxu_raw.shape[0] == n_v:
                ghxu_raw = ghxu_raw[:, None]
            else:
                ghxu_raw = ghxu_raw[None, :]
        if order_var is not None and len(ov) == n_v:
            ghxu_raw = ghxu_raw[inv_p, :]
    else:
        ghxu_raw = np.zeros((n_v, n_x * n_u))

    if "ghuu" in dr and dr["ghuu"] is not None:
        ghuu_raw = np.asarray(dr["ghuu"], dtype=float)
        if ghuu_raw.ndim == 0:
            ghuu_raw = ghuu_raw.reshape(1, 1)
        elif ghuu_raw.ndim == 1:
            if ghuu_raw.shape[0] == n_v:
                ghuu_raw = ghuu_raw[:, None]
            else:
                ghuu_raw = ghuu_raw[None, :]
        if ghuu_raw.shape[1] == n_u * (n_u + 1) // 2 and ghuu_raw.shape[1] != n_u**2:
            ghuu_raw = _unfold_ghxx(ghuu_raw, n_v, n_u)
        if order_var is not None and len(ov) == n_v:
            ghuu_raw = ghuu_raw[inv_p, :]
    else:
        ghuu_raw = np.zeros((n_v, n_u * n_u))

    cols_xx = [f"{s1}_{s2}" for s1 in s_names for s2 in s_names]
    cols_xu = [f"{s}_{e}" for s in s_names for e in e_names]
    cols_uu = [f"{e1}_{e2}" for e1 in e_names for e2 in e_names]

    df_ghxx = pd.DataFrame(ghxx_raw, index=list(v_names), columns=cols_xx)
    df_ghxu = pd.DataFrame(ghxu_raw, index=list(v_names), columns=cols_xu)
    df_ghuu = pd.DataFrame(ghuu_raw, index=list(v_names), columns=cols_uu)

    return Dynare2ndDR(
        ghx=df_ghx,
        ghu=df_ghu,
        ghxx=df_ghxx,
        ghxu=df_ghxu,
        ghuu=df_ghuu,
        ghs2=s_ghs2,
        ys=s_ys,
        state_variables=s_names,
        variable_names=v_names,
        shock_names=e_names,
    )


def load_dynare_moments(mat_path_or_dict: str | Path | dict) -> dict[str, np.ndarray]:
    """Parse theoretical moments and autocorrelations from Dynare results."""
    if isinstance(mat_path_or_dict, (str, Path)):
        p = Path(mat_path_or_dict)
        if not p.exists():
            raise FileNotFoundError(f"Dynare results file not found: {p}")
        raw = loadmat(str(p), squeeze_me=True, struct_as_record=False)
    elif isinstance(mat_path_or_dict, dict):
        raw = mat_path_or_dict
    else:
        raise TypeError(f"Expected str, Path, or dict, got {type(mat_path_or_dict)}")

    if "oo_" not in raw:
        raise KeyError(f"'{mat_path_or_dict}' does not contain 'oo_' structure (is this a Dynare results file?)")

    oo = _to_plain_dict(raw["oo_"])
    out: dict[str, np.ndarray] = {}

    if "mean" in oo and oo["mean"] is not None:
        out["mean"] = np.asarray(oo["mean"], dtype=float).squeeze()
    elif "steady_state" in oo and oo["steady_state"] is not None:
        out["mean"] = np.asarray(oo["steady_state"], dtype=float).squeeze()
    else:
        out["mean"] = np.array([], dtype=float)

    if "var" in oo and oo["var"] is not None:
        out["var"] = np.asarray(oo["var"], dtype=float)
    elif "gamma_y" in oo and oo["gamma_y"] is not None:
        out["var"] = np.asarray(oo["gamma_y"], dtype=float)
    else:
        out["var"] = np.array([], dtype=float)

    if "autocorr" in oo and oo["autocorr"] is not None:
        autocorr = oo["autocorr"]
        if isinstance(autocorr, (list, tuple)):
            out["autocorr"] = np.stack([np.asarray(m, dtype=float) for m in autocorr], axis=-1)
        else:
            out["autocorr"] = np.asarray(autocorr, dtype=float)
    else:
        out["autocorr"] = np.array([], dtype=float)

    return out

