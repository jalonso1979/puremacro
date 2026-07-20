"""Romer-Romer (2004) narrative monetary policy shocks.

The Romer-Romer measure is the residual from regressing the FOMC's
intended federal-funds-rate change (from internal Greenbook records)
on the Greenbook's own GDP-growth and inflation forecasts. The
residual identifies the fraction of intended policy that is NOT
explained by the staff's outlook — the "exogenous" monetary shock.

Original paper covers 1969Q1-1996Q4. Several extensions circulate
(Coibion 2012, Wieland-Yang 2020); pass ``csv_path=`` to load any
extended version with the ``date,RR_shock`` schema.

Reference
---------
Romer, C.D. and Romer, D.H. (2004). A new measure of monetary shocks:
derivation and implications. AER 94(4), 1055-1084.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes
from .._helpers import _csv_to_instrument


_MIRROR = (
    "https://eml.berkeley.edu/~dromer/papers/RomerandRomerDataAppendix.csv"
)

_REFERENCE = (
    "Romer, C.D. and Romer, D.H. (2004). A new measure of monetary "
    "shocks: derivation and implications. AER 94(4), 1055-1084."
)


def load(
    *,
    csv_path: str | Path | None = None,
    value_col: str = "RR_shock",
) -> Instrument:
    """Load the Romer-Romer 2004 monetary shock series.

    Parameters
    ----------
    csv_path : optional local path to the CSV. When None, attempt the
        mirror download; raises RuntimeError if both fail.
    value_col : str, default ``"RR_shock"``
        Name of the shock column in the CSV. Common alternatives:
        ``"intended_residual"``, ``"MP_shock"``.

    Returns
    -------
    Instrument
        Quarterly series, name ``"rr_2004_monetary"``, category
        ``"literature"``, frequency ``"Q"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Romer-Romer 2004 monetary shock series. "
                "Download the CSV from David Romer's website "
                "(eml.berkeley.edu/~dromer/) and pass csv_path=."
            ) from e
    expected_cols = {"date", value_col}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"RR2004 CSV missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}. "
            f"Pass value_col= if your shock column has a different name."
        )
    df = df.dropna(subset=["date", value_col]).copy()
    return _csv_to_instrument(
        df,
        name="rr_2004_monetary",
        source="Romer-Romer 2004 narrative monetary shocks",
        frequency="Q",
        value_col=value_col,
        date_col="date",
        metadata={"reference": _REFERENCE},
    )


__all__ = ["load"]
