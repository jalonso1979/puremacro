"""Piece-6 helpers for integration-order pre-tests + diff-VAR FEVD.

This module is a *thin* shim used by the notebook sweep. It exists so the
notebook cells stay short. It does NOT add anything new to ``var_sm.py``
(which Piece 5 owns); instead it wraps:

* ``puremacro.tests.unit_root.adf_test``, ``kpss_test`` (per-series ADF/KPSS).
* ``puremacro.var.vecm.johansen`` (Johansen trace test for cointegrating rank).
* ``puremacro.var.irf.fevd`` (forecast-error variance decomposition).

It also provides a ``fevd_from_array`` helper that takes a level/diff Y matrix
and an OLS VAR(p), and returns the Cholesky FEVD as an (H+1, n, n) array — the
"local wrapper" referred to in the Piece 6 brief. We do not patch this into
``src.teaching.var_sm`` because Piece 5 owns that file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Make sure puremacro is importable when notebooks call us.
# ---------------------------------------------------------------------------
def _ensure_puremacro() -> None:
    # Walk up from __file__ looking for a directory whose *parent* contains a
    # puremacro/__init__.py so that `import puremacro` resolves to the real
    # package.  Works whether this file lives in src/teaching/ or in
    # puremacro/puremacro/teaching/.
    p = Path(__file__).resolve()
    for ancestor in p.parents:
        candidate = ancestor / "puremacro" / "__init__.py"
        if candidate.exists():
            if str(ancestor) not in sys.path:
                sys.path.insert(0, str(ancestor))
            return


_ensure_puremacro()

# Load adf_test / kpss_test via puremacro._tests_unit_root to avoid
# namespace-package shadowing (puremacro.tests resolves to the pytest-tests
# directory when puremacro is installed as an editable namespace package).
def _import_unit_root():
    import importlib.util as _ilu
    import types as _types
    _alias = "puremacro._tests_unit_root"
    if _alias in sys.modules:
        return sys.modules[_alias]
    # Ensure parent package exists in sys.modules with correct __path__
    # so that relative imports inside unit_root.py (from .._linalg) resolve.
    _pkg_name = "puremacro._tests"
    if _pkg_name not in sys.modules:
        _pkg = _types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "tests")]
        _pkg.__package__ = "puremacro"
        sys.modules[_pkg_name] = _pkg
    _ur_path = Path(__file__).resolve().parent.parent / "tests" / "unit_root.py"
    _spec = _ilu.spec_from_file_location(f"{_pkg_name}.unit_root", _ur_path)
    _mod = _ilu.module_from_spec(_spec)
    _mod.__package__ = _pkg_name
    sys.modules[f"{_pkg_name}.unit_root"] = _mod
    sys.modules[_alias] = _mod
    _spec.loader.exec_module(_mod)
    return _mod


_ur = _import_unit_root()
adf_test = _ur.adf_test
kpss_test = _ur.kpss_test
del _ur, _import_unit_root
from puremacro.var.vecm import johansen as _pm_johansen  # noqa: E402
from puremacro.var.irf import fevd as _pm_fevd  # noqa: E402


def unit_root_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF/KPSS p-values in level and first difference, one row per series."""
    rows = []
    for name, s in series.items():
        s = s.dropna()
        if len(s) < 20:
            continue
        try:
            a = adf_test(s.values, regression="c")
        except Exception:
            a = {"p_value": float("nan")}
        try:
            k = kpss_test(s.values, regression="c")
        except Exception:
            k = {"p_value": float("nan")}
        sd = s.diff().dropna()
        try:
            a_d = adf_test(sd.values, regression="c") if len(sd) >= 20 else {"p_value": float("nan")}
        except Exception:
            a_d = {"p_value": float("nan")}
        try:
            k_d = kpss_test(sd.values, regression="c") if len(sd) >= 20 else {"p_value": float("nan")}
        except Exception:
            k_d = {"p_value": float("nan")}
        rows.append({
            "variable": name,
            "n_obs": int(len(s)),
            "adf_p_lvl": a.get("p_value"),
            "kpss_p_lvl": k.get("p_value"),
            "adf_p_diff": a_d.get("p_value"),
            "kpss_p_diff": k_d.get("p_value"),
        })
    return pd.DataFrame(rows).set_index("variable") if rows else pd.DataFrame()


def johansen_rank(df: pd.DataFrame, p: int = 2) -> int | None:
    """Estimated cointegrating rank from the Johansen trace test (5% level).

    Returns ``None`` if the test cannot be run (insufficient observations
    or numerical failure).
    """
    df = df.dropna()
    if df.shape[0] < 30 or df.shape[1] < 2:
        return None
    try:
        out = _pm_johansen(df, p=p)
    except Exception:
        return None
    # Different versions of puremacro have used 'r_hat' or 'rank_5pct'.
    if isinstance(out, dict):
        if "r_hat" in out:
            return int(out["r_hat"])
        if "rank_5pct" in out:
            return int(out["rank_5pct"])
    return None


def fevd_from_var(
    A_list: list[np.ndarray],
    Sigma: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Cholesky FEVD: (H+1, n, n) where [h, i, j] is share of Var(y_i) from shock j.

    Wraps ``puremacro.var.irf.fevd`` after computing the Cholesky factor of Sigma.
    """
    B0 = np.linalg.cholesky(Sigma)
    return _pm_fevd(A_list, B0, horizon=horizon)


def fevd_table(
    fev: np.ndarray,
    var_names: list[str],
    shock: str,
    targets: list[str] | None = None,
    horizons: tuple[int, ...] = (4, 8, 20),
) -> pd.DataFrame:
    """Compact FEVD table: rows = target variables, cols = horizons."""
    j = var_names.index(shock)
    targets = targets or [v for v in var_names if v != shock]
    rows = {}
    for t in targets:
        i = var_names.index(t)
        rows[t] = {f"h={h}": fev[h, i, j] for h in horizons if h < fev.shape[0]}
    return pd.DataFrame(rows).T


__all__ = [
    "adf_test",
    "kpss_test",
    "unit_root_table",
    "johansen_rank",
    "fevd_from_var",
    "fevd_table",
]
