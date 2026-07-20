"""Tests for puremacro.lp.iv_lewbel."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _build_panel(T_per_entity: int, n_entities: int, beta_true: float, seed: int) -> pd.DataFrame:
    """Build a panel satisfying Lewbel (2012) identification conditions.

    DGP:
        Z ~ N(0, 1)                         — heteroskedasticity source
        eps_common ~ N(0, 1)                 — shared error driving endogeneity
        eps_x ~ N(0, exp(0.4*Z))             — het shock in x (het source = Z)
        x = 0.8 * eps_common + eps_x         — endogenous regressor
        u = 0.8 * eps_common + 0.3 * eps_u  — structural error (homoskedastic in Z)
        y = beta_true * x + u

    Lewbel validity: Cov(Z * x_res, u) ≈ 0 because u is homoskedastic in Z.
    Lewbel relevance: Var(x | Z) is heteroskedastic in Z via eps_x.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_entities):
        for t in range(T_per_entity):
            z = rng.standard_normal()
            eps_common = rng.standard_normal()
            eps_x_std = rng.standard_normal()
            eps_u = rng.standard_normal()
            sigma_z = np.sqrt(np.exp(0.4 * z))
            x = 0.8 * eps_common + sigma_z * eps_x_std
            u = 0.8 * eps_common + 0.3 * eps_u
            rows.append({
                "code": f"E{i}",
                "date": pd.Timestamp("2000-01-01") + pd.DateOffset(months=3 * t),
                "y": beta_true * x + u,
                "x": x,
                "z": z,
            })
    return pd.DataFrame(rows)


def test_lp_iv_lewbel_returns_long_form_dataframe():
    df = _build_panel(T_per_entity=80, n_entities=10, beta_true=1.0, seed=0)
    from puremacro.lp.iv_lewbel import lp_iv_lewbel
    out = lp_iv_lewbel(
        df, y="y", x_endog="x", heterosk_source="z",
        horizons=range(0, 5), n_lags=2,
    )
    expected_cols = {"h", "beta", "se", "t", "lo", "hi", "first_stage_F", "lewbel_p"}
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == 5
    # Coefficient at h=0 should be near β_true on average.
    assert abs(float(out.loc[out["h"] == 0, "beta"].iloc[0]) - 1.0) < 0.5
