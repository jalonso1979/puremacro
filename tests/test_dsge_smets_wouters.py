"""Unit tests for the SW07 linearised system + Klein/Sylvester solution.

Tests organised by equation block. Each block-test verifies that the
relevant row of A (G0), B (G1), C (Psi) has the correct linearised
coefficients from Pfeifer's sw07.mod up to a tight tolerance.

Variable layout in the 44-vector z_t:
  States  [0-19]: kpf_lag, invef_lag, cf_lag, kp_lag, inve_lag, c_lag,
                  pinf_lag, w_lag, y_lag, yf_lag, r_lag,
                  a, b, g, qs, ms, spinf, sw, epinfma, ewma
  Controls [20-43]: zcapf, rkf, kf, pkf, cf, invef, yf, labf, wf, rrf, kpf,
                    mc, zcap, rk, k, pk, c, inve, y, lab, pinf, w, r, kp

Matrices A=G0 (forward coefficients), B=G1 (contemporaneous), C=Psi (shocks).
Note: 'b' (risk-premium process) is a STATE variable (shock process); the
innovation 'eb' enters only the AR-process row, not the consumption Euler
row directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dsge.smets_wouters import (
    _build_sw07_matrices,
    _compute_derived,
    SW07_POSTERIOR_MODE,
    STATE_NAMES,
    CONTROL_NAMES,
    SHOCK_NAMES,
)

# Backward-compat alias used in fixture
SW07_PRIOR_MEAN = SW07_POSTERIOR_MODE


@pytest.fixture
def matrices():
    A, B, C, sn, cn, shk = _build_sw07_matrices(SW07_POSTERIOR_MODE)
    return {"A": A, "B": B, "C": C}


def _row_for_control(name: str) -> int:
    n_pre = len(STATE_NAMES)
    return n_pre + CONTROL_NAMES.index(name)


def _col_for_state(name: str) -> int:
    return STATE_NAMES.index(name)


def _col_for_control(name: str) -> int:
    n_pre = len(STATE_NAMES)
    return n_pre + CONTROL_NAMES.index(name)


def _col_for_shock(name: str) -> int:
    return SHOCK_NAMES.index(name)


# ---------------------------------------------------------------------------
# E7 — Consumption Euler (Pfeifer eq. S7 / SW07 eq. 2)
#
# Pfeifer mod (line 270-273):
#   c = (chabb/cgamma)/(1+chabb/cgamma)*c(-1)
#       + (1/(1+chabb/cgamma))*c(+1)
#       + ((csigma-1)*cwhlc/(csigma*(1+chabb/cgamma)))*(lab - lab(+1))
#       - (1-chabb/cgamma)/(csigma*(1+chabb/cgamma))*(r - pinf(+1)) + b
#
# Klein form A E_t[z(t+1)] = B z_t (G0=A, G1=B):
#   Let hg = chabb/cgamma, inv1hg = 1/(1+hg), hg_inv1hg = hg/(1+hg),
#       rr_coef = (1-hg)/(csigma*(1+hg)), lab_coef = (csigma-1)*cwhlc/(csigma*(1+hg))
#
#   A[c, c]    =  inv1hg                   (forward c(+1))
#   A[c, lab]  = -lab_coef                 (forward lab(+1) with negative sign)
#   A[c, pinf] =  rr_coef                  (forward pinf(+1))
#   B[c, c]    =  1.0                      (contemporaneous c_t on LHS)
#   B[c, c_lag]= -hg_inv1hg               (c(-1) via lag state)
#   B[c, r]    =  rr_coef                  (r_t)
#   B[c, lab]  = -lab_coef                 (contemporaneous lab_t)
#   B[c, b]    = -1.0                      (risk-premium state b_t; b is a STATE)
#   C[c, *]    =  0  for all shocks        (b enters via state, not Psi)
# ---------------------------------------------------------------------------


def test_consumption_euler_B_coefficients(matrices):
    """B row 'c': LHS coefficients on (c_t, c_lag, r_t) from the rearranged Euler."""
    p = SW07_POSTERIOR_MODE
    d = _compute_derived(p)
    chabb, csigma, cgamma = p["chabb"], p["csigma"], d["cgamma"]
    hg = chabb / cgamma
    hg_inv1hg = hg / (1.0 + hg)
    ihg = 1.0 - hg
    denom_c = csigma * (1.0 + hg)
    rr_coef = ihg / denom_c
    row_c = _row_for_control("c")

    B = matrices["B"]
    np.testing.assert_allclose(
        B[row_c, _col_for_control("c")], 1.0, atol=1e-12,
        err_msg="B[c, c] should be 1.0",
    )
    np.testing.assert_allclose(
        B[row_c, _col_for_state("c_lag")], -hg_inv1hg, atol=1e-12,
        err_msg="B[c, c_lag] should be -hg/(1+hg) where hg=chabb/cgamma",
    )
    np.testing.assert_allclose(
        B[row_c, _col_for_control("r")], rr_coef, atol=1e-12,
        err_msg="B[c, r] should be (1-hg)/(csigma*(1+hg)) where hg=chabb/cgamma",
    )


def test_consumption_euler_A_coefficients(matrices):
    """A row 'c': forward coefficients on (E_t c_{t+1}, E_t pinf_{t+1})."""
    p = SW07_POSTERIOR_MODE
    d = _compute_derived(p)
    chabb, csigma, cgamma = p["chabb"], p["csigma"], d["cgamma"]
    hg = chabb / cgamma
    inv1hg = 1.0 / (1.0 + hg)
    ihg = 1.0 - hg
    denom_c = csigma * (1.0 + hg)
    rr_coef = ihg / denom_c
    row_c = _row_for_control("c")

    A = matrices["A"]
    np.testing.assert_allclose(
        A[row_c, _col_for_control("c")], inv1hg, atol=1e-12,
        err_msg="A[c, c] should be 1/(1+hg) where hg=chabb/cgamma",
    )
    np.testing.assert_allclose(
        A[row_c, _col_for_control("pinf")], rr_coef, atol=1e-12,
        err_msg="A[c, pinf] should be (1-hg)/(csigma*(1+hg)) where hg=chabb/cgamma",
    )


def test_consumption_euler_b_state_coefficient(matrices):
    """B row 'c': risk-premium STATE b enters with coefficient -1.

    'b' is an AR state variable (shock process), not a Psi shock column.
    The innovation 'eb' enters only the AR-process equation for b.
    So C[c, eb] == 0 and B[c, b] == -1.
    """
    row_c = _row_for_control("c")
    col_b = _col_for_state("b")

    B = matrices["B"]
    C = matrices["C"]
    np.testing.assert_allclose(
        B[row_c, col_b], -1.0, atol=1e-12,
        err_msg="B[c, b] should be -1.0 (risk-premium state enters Euler directly)",
    )
    # The 'eb' column of Psi/C should be zero for the consumption row.
    np.testing.assert_allclose(
        C[row_c, _col_for_shock("eb")], 0.0, atol=1e-12,
        err_msg="C[c, eb] should be 0 — b enters via STATE, not Psi",
    )


# ---------------------------------------------------------------------------
# E9 — BK and max-eigenvalue smoke test
# ---------------------------------------------------------------------------


def test_bk_and_max_eigenvalue():
    """BK is satisfied (eu=(1,1)) and max|eig(G)| < 1."""
    from puremacro.dsge.smets_wouters import solve_sw07

    sol = solve_sw07()
    assert sol.eu == (1, 1), f"BK failed: eu={sol.eu}"
    G_x = sol.G[:len(STATE_NAMES), :len(STATE_NAMES)]
    max_eig = float(np.abs(np.linalg.eigvals(G_x)).max())
    assert max_eig < 1.0, f"G not stable: max|eig|={max_eig:.6f}"


# ---------------------------------------------------------------------------
# E10 — Qualitative IRF smoke tests
# ---------------------------------------------------------------------------


def test_tech_shock_irf_positive_output():
    """A 1-unit technology shock raises output on impact (y[0] > 0)."""
    from puremacro.dsge.smets_wouters import tech_shock_irf

    irf = tech_shock_irf(range(0, 41))
    y0 = float(irf.iloc[0]["y"])
    assert y0 > 0.0, f"Tech shock: y[0]={y0:.4f} should be positive"


def test_tech_shock_irf_hump_shape():
    """Tech shock IRF has hump shape: y rises before falling."""
    from puremacro.dsge.smets_wouters import tech_shock_irf

    irf = tech_shock_irf(range(0, 41))
    y0 = float(irf.iloc[0]["y"])
    y4 = float(irf.iloc[4]["y"])
    y40 = float(irf.iloc[40]["y"])
    assert y4 > y0, f"Tech shock: y[4]={y4:.4f} should exceed y[0]={y0:.4f} (hump)"
    assert y40 < y4, f"Tech shock: y[40]={y40:.4f} should be below y[4]={y4:.4f} (decay)"


def test_monetary_shock_irf_output_negative():
    """A 1-unit monetary tightening shock lowers output (y[4] < 0)."""
    from puremacro.dsge.smets_wouters import monetary_shock_irf

    irf = monetary_shock_irf(range(0, 41))
    y4 = float(irf.iloc[4]["y"])
    r0 = float(irf.iloc[0]["r"])
    assert y4 < 0.0, f"Monetary shock: y[4]={y4:.4f} should be negative"
    assert r0 > 0.0, f"Monetary shock: r[0]={r0:.4f} should be positive"


# ---------------------------------------------------------------------------
# E11 — Growth-rate IRF quantitative range tests
# ---------------------------------------------------------------------------
# The published SW07 Fig 1 shows GROWTH-RATE IRFs (dy = y(h) - y(h-1)).
# Level IRFs y(h) accumulate due to habit persistence and crhoa=0.9977 ≈ 1;
# comparing level y[4] with growth-rate dy[4] is an apples-to-oranges mistake.
# These tests confirm that growth-rate magnitudes are in the correct range.
# ---------------------------------------------------------------------------


def test_tech_shock_growth_irf_impact():
    """Growth-rate output at h=0 (=impact level) matches published ~+0.30 (unit_sd)."""
    from puremacro.dsge.smets_wouters import tech_shock_growth_irf

    gr = tech_shock_growth_irf(range(0, 21), unit_sd=True)
    dy0 = float(gr.set_index("h")["dy"].iloc[0])
    # Published SW07 Fig 1: dy[0] ≈ +0.30; accept [0.20, 0.50] (covers 20% model error).
    assert 0.20 <= dy0 <= 0.50, (
        f"dy[0]={dy0:.3f} outside expected range [0.20, 0.50] for tech shock (unit_sd)"
    )


def test_tech_shock_growth_irf_h4_range():
    """Growth-rate output at h=4 matches published ~+0.05 (unit_sd)."""
    from puremacro.dsge.smets_wouters import tech_shock_growth_irf

    gr = tech_shock_growth_irf(range(0, 21), unit_sd=True)
    dy4 = float(gr.set_index("h")["dy"].iloc[4])
    # Published SW07 Fig 1: dy[4] ≈ +0.05; accept [0.01, 0.20].
    assert 0.01 <= dy4 <= 0.20, (
        f"dy[4]={dy4:.3f} outside expected range [0.01, 0.20] for tech shock (unit_sd)"
    )


def test_tech_shock_growth_irf_decays():
    """Growth-rate output declines from impact: dy[0] > dy[4] > 0 (unit_sd)."""
    from puremacro.dsge.smets_wouters import tech_shock_growth_irf

    gr = tech_shock_growth_irf(range(0, 21), unit_sd=True).set_index("h")
    dy0 = float(gr["dy"].iloc[0])
    dy4 = float(gr["dy"].iloc[4])
    assert dy0 > dy4, f"Growth IRF should decay: dy[0]={dy0:.3f} > dy[4]={dy4:.3f}"
    assert dy4 > 0.0, f"dy[4]={dy4:.3f} should still be positive at h=4"
