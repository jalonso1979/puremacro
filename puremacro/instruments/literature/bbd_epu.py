"""Baker-Bloom-Davis Economic Policy Uncertainty index (US, monthly).

The index is a free monthly publication from policyuncertainty.com,
constructed from a news-coverage component, tax-code-expiration counts,
and disagreement among economic forecasters. We fetch the news-based
US headline series, which is the most-cited variant in macro VARs.

Reference
---------
Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic policy
uncertainty. QJE 131(4), 1593-1636.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes
from .._helpers import _csv_to_instrument


_MIRROR = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"

_REFERENCE = (
    "Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic "
    "policy uncertainty. QJE 131(4), 1593-1636."
)


def load(*, csv_path: str | Path | None = None) -> Instrument:
    """Load the BBD EPU US monthly index.

    Parameters
    ----------
    csv_path : optional local path to the CSV. When None, attempt the
        canonical policyuncertainty.com download.

    Returns
    -------
    Instrument
        Monthly series, name ``"bbd_epu_us"``, category ``"literature"``,
        frequency ``"M"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch BBD EPU index. Download "
                "US_Policy_Uncertainty_Data.csv from "
                "https://www.policyuncertainty.com/ and pass csv_path=."
            ) from e
    expected_cols = {"Year", "Month", "News_Based_Policy_Uncert_Index"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"BBD EPU CSV missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}."
        )
    df = df.dropna(subset=["Year", "Month"]).copy()
    return _csv_to_instrument(
        df,
        name="bbd_epu_us",
        source="Baker-Bloom-Davis Economic Policy Uncertainty (US, monthly)",
        frequency="M",
        value_col="News_Based_Policy_Uncert_Index",
        year_col="Year",
        month_col="Month",
        metadata={"reference": _REFERENCE},
    )


__all__ = ["load"]
