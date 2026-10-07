"""Replication of Arellano-Bond (1991) Table 4, column a1, via Stata's
[XT] xtabond manual (https://www.stata.com/manuals/xtxtabond.pdf).

The manual reproduces AB (1991) Table 4 column a1 on ``abdata``:

- Example 1 (PDF p.5), ``xtabond n l(0/1).w l(0/2).(k ys) yr1980-yr1984
  year, lags(2) noconstant``: one-step coefficients and homoskedastic
  standard errors; "The coefficients are identical to those reported in
  column a1 of table 4".
- Example 2 (PDF p.7), the same command with ``vce(robust)``: "now the
  standard errors match that reported in Arellano and Bond (1991, table 4,
  column a1)".

All three runs report 611 observations, 140 groups and 41 instruments.
The Stata values are printed to 7 decimals; puremacro reproduces every
printed digit (max |difference| < 5e-8), so the tolerance is 1e-7.

The data are not bundled (see ``conftest.load_abdata_xtabond_design``);
without a copy of abdata these tests are skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dynpanel import ab_gmm

# Stata [XT] xtabond, Example 1 (p.5) = Example 2 (p.7) coefficients,
# in Stata's order: L1.n L2.n w L1.w k L1.k L2.k ys L1.ys L2.ys
# yr1980..yr1984 year.
EX1_COEF = np.array([
    .6862261, -.0853582, -.6078208, .3926237, .3568456, -.0580012, -.0199475,
    .6085073, -.7111651, .1057969, .0029062, -.0404378, -.0652767, -.0690928,
    -.0650302, .0095545,
])
# Example 1 (p.5): one-step, homoskedastic ("Std. err.").
EX1_SE = np.array([
    .1486163, .0444365, .0657694, .1092374, .0370314, .0583051, .0416274,
    .1345412, .1844599, .1428568, .0212705, .0354707, .048209, .0627354,
    .0781322, .0142073,
])
# Example 2 (p.7): one-step, vce(robust) ("Robust std. err.").
EX2_SE = np.array([
    .1445943, .0560155, .1782055, .1679931, .0590203, .0731797, .0327126,
    .1725313, .2317163, .1412021, .0158028, .0280582, .0365451, .047413,
    .0576305, .0102896,
])
ATOL = 1e-7  # printed to 7 decimals


def _fit(d, **kw):
    return ab_gmm(
        d["y"], d["panel_id"], d["time_id"],
        lag_dep_var=2, X_exog=d["X_exog"], collapse=False,
        two_step=False, names=d["names"], **kw,
    )


def test_ab_1991_table4_col_a1_one_step_homoskedastic(abdata_xtabond):
    """xtabond Example 1: one-step coefficients and homoskedastic SEs."""
    res = _fit(abdata_xtabond, robust=False)
    assert (res.n_obs, res.n_panels, res.n_instruments) == (611, 140, 41)
    np.testing.assert_allclose(res.coefs, EX1_COEF, rtol=0, atol=ATOL)
    np.testing.assert_allclose(res.se, EX1_SE, rtol=0, atol=ATOL)


def test_ab_1991_table4_col_a1_one_step_robust(abdata_xtabond):
    """xtabond Example 2: one-step coefficients and robust SEs (AB 1991 a1)."""
    res = _fit(abdata_xtabond)
    assert (res.n_obs, res.n_panels, res.n_instruments) == (611, 140, 41)
    assert res.step == 1 and not res.windmeijer
    np.testing.assert_allclose(res.coefs, EX1_COEF, rtol=0, atol=ATOL)
    np.testing.assert_allclose(res.se, EX2_SE, rtol=0, atol=ATOL)
    assert res.names[0] == "L1.n"
    assert res.coefs[0] == pytest.approx(.6862261, abs=ATOL)
    assert res.se[0] == pytest.approx(.1445943, abs=ATOL)
