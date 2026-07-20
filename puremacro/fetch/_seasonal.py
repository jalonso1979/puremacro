"""X13-ARIMA seasonal-adjustment helper.

Wraps ``statsmodels.tsa.x13.x13_arima_analysis`` so callers (currently
the ILOSTAT fetchers) can SA NSA quarterly series before merging into
the SA-by-default labor panel.

Hard system dependency: the X13ASOSS binary must be on PATH. On macOS,
``brew install x13as``. On Linux, download from the US Census Bureau
(https://www.census.gov/data/software/x13as.html).
"""
from __future__ import annotations

import os
import shutil

import pandas as pd


def _x13_arima_analysis(*args, **kwargs):
    """Module-level shim: lazy-imports statsmodels' x13 and delegates.

    Why a wrapper, not a top-level import: ARCHITECTURE.md's Pyodide
    contract forbids importing ``statsmodels`` at module load. Tests still
    monkeypatch this name via ``monkeypatch.setattr(_seasonal,
    "_x13_arima_analysis", _raise)`` — that contract is preserved because
    this is a real module-level attribute, just lazily backed.
    """
    from statsmodels.tsa.x13 import x13_arima_analysis as _impl
    return _impl(*args, **kwargs)


_INSTALL_HINT = (
    "X13-ARIMA-SEATS binary (x13as) not found on PATH. "
    "Install via `brew install x13as` (macOS) or download from the "
    "US Census Bureau (https://www.census.gov/data/software/x13as.html) "
    "and add the binary to PATH."
)


def _find_x13_path() -> str | None:
    """Return the directory containing the x13as binary, or None if not found.

    statsmodels' internal path-finder uses environment variables (X13PATH /
    X12PATH) rather than shutil.which, so we bridge the gap here by locating
    the binary via shutil.which and returning its parent directory.
    """
    binary = shutil.which("x13as") or shutil.which("x13as_ascii")
    if binary:
        return os.path.dirname(binary)
    return None


def x13_seasonal_adjust(
    series: pd.Series,
    *,
    freq: str = "Q",
) -> pd.Series:
    """Run X13-ARIMA-SEATS and return the seasonally adjusted series.

    Parameters
    ----------
    series : pd.Series
        Input series. Must have at least 12 observations (X13 minimum
        for quarterly) or 36 (monthly). Index is preserved.
    freq : str, default "Q"
        ``"Q"`` for quarterly or ``"M"`` for monthly.

    Returns
    -------
    pd.Series
        Seasonally adjusted series, same index and name as input.

    Raises
    ------
    RuntimeError
        If the X13 binary is not installed (with install hint), if the
        series is too short, or if ARIMA estimation fails to converge.
    """
    if freq not in {"Q", "M"}:
        raise ValueError(f"freq must be 'Q' or 'M'; got {freq!r}")
    min_len = 12 if freq == "Q" else 36
    if len(series.dropna()) < min_len:
        raise RuntimeError(
            f"series too short for X13 SA at freq={freq!r}: "
            f"got {len(series.dropna())} non-NaN observations, "
            f"need at least {min_len} (insufficient length)."
        )
    x12path = _find_x13_path()
    try:
        result = _x13_arima_analysis(series, freq=freq, x12path=x12path)
    except FileNotFoundError as exc:
        # Raised by the monkeypatch in tests (and potentially by subprocess).
        raise RuntimeError(_INSTALL_HINT) from exc
    except Exception as exc:
        # X13NotFoundError (statsmodels) is not a FileNotFoundError subclass;
        # detect it by name. All other exceptions are wrapped for clarity.
        exc_type_name = type(exc).__name__
        if "NotFound" in exc_type_name:
            raise RuntimeError(_INSTALL_HINT) from exc
        raise RuntimeError(f"X13 ARIMA estimation failed: {exc}") from exc
    sa = result.seasadj.copy()
    sa.name = series.name
    return sa


__all__ = ["x13_seasonal_adjust"]
