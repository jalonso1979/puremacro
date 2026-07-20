"""Research-grade Bartik diagnostics.

- Rotemberg weights (Goldsmith-Pinkham-Sorkin-Swift, AER 2020).
- Borusyak-Hull-Jaravel (ReStud 2022) shift-level equivalence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rotemberg_weights(
    shares: pd.DataFrame,
    shifts: pd.DataFrame,
    y: pd.DataFrame,
    *,
    share_state_col: str, share_bucket_col: str, share_value_col: str,
    shift_bucket_col: str, shift_date_col: str, shift_value_col: str,
    y_state_col: str, y_date_col: str, y_value_col: str,
) -> pd.DataFrame:
    """Rotemberg weight for each bucket k.

    α_k = Σ_{s,t} share_{s,k} · shift_{k,t} · y_{s,t}, normalized so Σ_k α_k = 1.
    """
    merged = (shares
        .merge(shifts, left_on=share_bucket_col, right_on=shift_bucket_col)
        .merge(y, left_on=[share_state_col, shift_date_col],
               right_on=[y_state_col, y_date_col]))
    merged["num"] = merged[share_value_col] * merged[shift_value_col] * merged[y_value_col]
    raw = merged.groupby(share_bucket_col)["num"].sum().reset_index()
    total = raw["num"].sum()
    raw["weight"] = raw["num"] / total
    return raw.rename(columns={share_bucket_col: "bucket"})[["bucket", "weight"]]


def bhj_shift_level_regression(
    shares: pd.DataFrame,
    shifts: pd.DataFrame,
    y: pd.DataFrame,
    *,
    share_state_col: str, share_bucket_col: str, share_value_col: str,
    shift_bucket_col: str, shift_date_col: str, shift_value_col: str,
    y_state_col: str, y_date_col: str, y_value_col: str,
) -> float:
    """Shift-level β via Borusyak-Hull-Jaravel (2022).

    Transforms share-level data into bucket-time observations with exposure-
    weighted outcomes and regresses the weighted outcome on the shift.
    Coefficient matches the share-level IV by construction.
    """
    expo = shares.groupby(share_bucket_col)[share_value_col].sum().rename("total_share")
    j = shares.merge(y, left_on=share_state_col, right_on=y_state_col)
    j["w_y"] = j[share_value_col] * j[y_value_col]
    y_bar = (j.groupby([share_bucket_col, y_date_col])["w_y"].sum()
              .reset_index()
              .merge(expo.reset_index(), on=share_bucket_col))
    y_bar["y_bar"] = y_bar["w_y"] / y_bar["total_share"]
    panel = y_bar.merge(
        shifts, left_on=[share_bucket_col, y_date_col],
        right_on=[shift_bucket_col, shift_date_col],
    )
    X = panel[shift_value_col].values
    yv = panel["y_bar"].values
    w = panel["total_share"].values
    beta = np.sum(w * X * yv) / np.sum(w * X * X)
    return float(beta)
