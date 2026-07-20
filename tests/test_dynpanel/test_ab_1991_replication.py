"""Replication of Arellano-Bond (1991) Table 4, column 2.

Requires ``tests/fixtures/abdata.csv`` (see fixture README for how to
obtain). Skipped automatically if the CSV is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dynpanel import ab_gmm

ABDATA_CSV = Path(__file__).parent.parent / "fixtures" / "abdata.csv"


@pytest.fixture(scope="module")
def abdata():
    if not ABDATA_CSV.exists():
        pytest.skip(
            f"Replication fixture {ABDATA_CSV.name} not present. "
            "See tests/fixtures/abdata.README.md for how to obtain."
        )
    df = pd.read_csv(ABDATA_CSV)
    needed = {"id", "year", "n", "w", "k", "ys"}
    missing = needed - set(df.columns)
    if missing:
        pytest.skip(f"abdata.csv missing required columns: {sorted(missing)}")
    return df.sort_values(["id", "year"]).reset_index(drop=True)


def test_ab_1991_table4_col2_recovers_published_lag_coefficients(abdata):
    """AB 1991 Table 4 col. 2: two-step diff-GMM with Windmeijer SE.

    Published: L1.n = 0.474 (s.e. 0.085), L2.n = -0.053 (s.e. 0.027).
    We assert recovery within 0.05 absolute on coefficients (generous
    tolerance — any reasonable port should clear it).
    """
    df = abdata
    res = ab_gmm(
        y=df["n"].to_numpy(),
        panel_id=df["id"].to_numpy(),
        time_id=df["year"].to_numpy(),
        lag_dep_var=2,
        X_pred=df[["w", "k", "ys"]].to_numpy(),
        two_step=True,
        windmeijer=True,
        collapse=True,
    )
    coef_L1 = res.coefs[0]
    coef_L2 = res.coefs[1]
    assert abs(coef_L1 - 0.474) < 0.05, f"L1.n = {coef_L1:.3f}"
    assert abs(coef_L2 - (-0.053)) < 0.05, f"L2.n = {coef_L2:.3f}"


def test_ab_1991_hansen_j_does_not_reject(abdata):
    """Hansen J should not reject overid restrictions on the AB DGP."""
    df = abdata
    res = ab_gmm(
        y=df["n"].to_numpy(),
        panel_id=df["id"].to_numpy(),
        time_id=df["year"].to_numpy(),
        lag_dep_var=2,
        X_pred=df[["w", "k", "ys"]].to_numpy(),
        two_step=True,
        windmeijer=True,
        collapse=True,
    )
    assert res.hansen_j_p > 0.05, f"Hansen J p = {res.hansen_j_p:.3f}"
