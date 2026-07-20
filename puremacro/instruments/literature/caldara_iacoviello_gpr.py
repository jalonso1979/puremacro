"""Caldara-Iacoviello (2022) Geopolitical Risk (GPR) index (monthly).

A news-based monthly index of geopolitical risk constructed from
counts of articles in major newspapers discussing adverse geopolitical
events. Published at matteoiacoviello.com/gpr.htm; we fetch the CSV
mirror.

Reference
---------
Caldara, D. and Iacoviello, M. (2022). Measuring geopolitical risk.
American Economic Review 112(4), 1194-1225.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes
from .._helpers import _csv_to_instrument


_MIRROR = "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.csv"

_REFERENCE = (
    "Caldara, D. and Iacoviello, M. (2022). Measuring geopolitical risk. "
    "American Economic Review 112(4), 1194-1225."
)


def load(*, csv_path: str | Path | None = None) -> Instrument:
    """Load the Caldara-Iacoviello GPR monthly index.

    Parameters
    ----------
    csv_path : optional local path to the CSV. When None, attempt the
        canonical matteoiacoviello.com download.

    Returns
    -------
    Instrument
        Monthly GPR series, name ``"caldara_iacoviello_gpr"``,
        category ``"literature"``, frequency ``"M"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Caldara-Iacoviello GPR index. Download "
                "the CSV from https://www.matteoiacoviello.com/gpr.htm "
                "and pass csv_path=."
            ) from e
    expected_cols = {"month", "GPR"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"GPR CSV missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}."
        )
    df = df.dropna(subset=["month", "GPR"]).copy()
    return _csv_to_instrument(
        df,
        name="caldara_iacoviello_gpr",
        source="Caldara-Iacoviello (2022) Geopolitical Risk index",
        frequency="M",
        value_col="GPR",
        date_col="month",
        metadata={"reference": _REFERENCE},
    )


__all__ = ["load"]
