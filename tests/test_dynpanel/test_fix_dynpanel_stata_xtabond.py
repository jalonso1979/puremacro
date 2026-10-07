"""Stata [XT] xtabond Example 4 on abdata: two-step GMM with the Windmeijer
bias-corrected (WC) robust VCE, and the uncollapsed instrument layout.

Source: https://www.stata.com/manuals/xtxtabond.pdf, Example 4 (PDF p.9):

    . xtabond n l(0/1).w l(0/2).(k ys) yr1980-yr1984 year, lags(2) twostep
    > vce(robust) noconstant

Number of obs = 611, Number of groups = 140, Number of instruments = 41,
"Two-step results", "WC-robust std. err.". GMM-type instruments L(2/.).n
(uncollapsed), standard instruments the first differences of the 14
exogenous regressors.

Before the fix, ``ab_gmm(..., collapse=False, lag_dep_var=2)`` built 42
columns, one of them identically zero, and raised LinAlgError; with that
column removed by hand the WC standard errors were 1.01-1.68 times Stata's
because dS/dbeta was evaluated at the step-2 residuals.

The data are not bundled; the tests skip when abdata is unavailable (see
``conftest.load_abdata_xtabond_design``).
"""
from __future__ import annotations

import numpy as np

from puremacro.dynpanel import ab_gmm
from puremacro.dynpanel.instruments import build_instruments

EX4_COEF = np.array([
    .6287089, -.0651882, -.5257597, .3112899, .2783619, .0140994, -.0402484,
    .5919243, -.5659863, .1005433, .0006378, -.0550044, -.075978, -.0740708,
    -.0906606, .0112155,
])
EX4_WC_SE = np.array([
    .1934138, .0450501, .1546107, .2030006, .0728019, .0924575, .0432745,
    .1730916, .2611008, .1610987, .0168042, .0313389, .0419276, .0528381,
    .0642615, .0116783,
])
ATOL = 1e-7  # Stata prints 7 decimals


def test_uncollapsed_layout_has_41_live_instruments(abdata_xtabond):
    d = abdata_xtabond
    b = build_instruments(
        d["y"], d["panel_id"], d["time_id"],
        lag_dep_var=2, X_exog=d["X_exog"], collapse=False,
    )
    Z = b["Z"]
    assert Z.shape == (611, 41)
    assert np.all(np.any(Z != 0.0, axis=0)), "an all-zero instrument column survived"
    assert b["dropped_instr_labels"] == ["_yL0_L2@t1978"]
    # 27 GMM-type columns (L(2/.).n, t = 1979..1984) + 14 standard ones
    assert sum(lab.startswith("_yL0_") for lab in b["instr_labels"]) == 27


def test_xtabond_example4_two_step_wc_robust(abdata_xtabond):
    d = abdata_xtabond
    res = ab_gmm(
        d["y"], d["panel_id"], d["time_id"],
        lag_dep_var=2, X_exog=d["X_exog"], collapse=False,
        two_step=True, windmeijer=True, names=d["names"],
    )
    assert (res.n_obs, res.n_panels, res.n_instruments) == (611, 140, 41)
    assert res.step == 2 and res.windmeijer
    assert res.hansen_j_df == 41 - 16
    assert any("_yL0_L2@t1978" in note for note in res.notes)
    np.testing.assert_allclose(res.coefs, EX4_COEF, rtol=0, atol=ATOL)
    np.testing.assert_allclose(res.se, EX4_WC_SE, rtol=0, atol=ATOL)
    # the headline pair: two-step L1.n .6287089 with WC-robust s.e. .1934138
    assert abs(res.coefs[0] - .6287089) < ATOL
    assert abs(res.se[0] - .1934138) < ATOL
