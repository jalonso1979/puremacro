"""Synthetic tests for puremacro.labor_flows_enoe — no ENOE microdata needed.

Covers the pure (browser-clean) half of the module: person-ID construction,
labor-status assignment, and the logm/3 + expm + IRW monthly-matrix recovery.
The .dta loaders (out-of-browser side-channel) are exercised only through
their input-validation paths.
"""
import numpy as np
import pandas as pd
import pytest

from puremacro.labor_flows_enoe import (
    STATES,
    assign_labor_status,
    make_person_id,
    quarterly_to_monthly_matrix,
)


def test_states_contract():
    """README contract: 4 states F/I/U/N, in this order."""
    assert STATES == ("F", "I", "U", "N")


# --------------------------------------------------------------------------- #
# make_person_id                                                              #
# --------------------------------------------------------------------------- #

def test_make_person_id_composes_key():
    """ENT(2) + CON(5) + V_SEL(2) + N_HOG(1) + H_MUD(1) + N_REN(2) = 13 chars."""
    df = pd.DataFrame({
        "ENT": [9.0, 15.0],
        "CON": [123.0, 4.0],
        "V_SEL": [2.0, 11.0],
        "N_HOG": [1.0, 2.0],
        "H_MUD": [0.0, 1.0],
        "N_REN": [3.0, 12.0],
    })
    pid = make_person_id(df)
    assert list(pid) == ["0900123021003", "1500004112112"]
    assert (pid.str.len() == 13).all()


def test_make_person_id_missing_column_raises():
    df = pd.DataFrame({
        "ENT": [9.0], "CON": [123.0], "V_SEL": [2.0],
        "N_HOG": [1.0], "H_MUD": [0.0],   # N_REN missing
    })
    with pytest.raises(ValueError, match="N_REN"):
        make_person_id(df)


def test_make_person_id_alphanumeric_fallback():
    """Alphanumeric codes (e.g. 'A0' dwelling selections) are preserved,
    not coerced to int; numeric strings in the same column still zero-pad."""
    df = pd.DataFrame({
        "ENT": [1.0, 1.0],
        "CON": [1.0, 1.0],
        "V_SEL": ["A0", "7"],
        "N_HOG": [1.0, 1.0],
        "H_MUD": [0.0, 0.0],
        "N_REN": [1.0, 1.0],
    })
    pid = make_person_id(df)
    assert list(pid) == ["0100001A01001", "0100001071001"]


# --------------------------------------------------------------------------- #
# assign_labor_status                                                         #
# --------------------------------------------------------------------------- #

def test_assign_labor_status_maps_fiun():
    """Formal (P3K1==1), informal (P3K1==2), unemployed, and the two
    PEI codes (3 disponible, 4 no disponible) both -> N."""
    df = pd.DataFrame({
        "CLASE2": [1.0, 1.0, 2.0, 3.0, 4.0],
        "P3K1": [1.0, 2.0, np.nan, np.nan, np.nan],
    })
    status = assign_labor_status(df)
    assert list(status) == ["F", "I", "U", "N", "N"]


def test_assign_labor_status_formality_union():
    """Formal iff ANY of A_SEG_SOC / P3J / P3K1 == 1 (survey-format union)."""
    df = pd.DataFrame({
        "CLASE2": [1.0, 1.0, 1.0, 1.0],
        "A_SEG_SOC": [1.0, np.nan, np.nan, 0.0],
        "P3J": [np.nan, 1.0, 2.0, 2.0],
        "P3K1": [np.nan, np.nan, 1.0, 2.0],
    })
    status = assign_labor_status(df)
    assert list(status) == ["F", "F", "F", "I"]


def test_assign_labor_status_string_codes():
    """Object-dtype codes ('1', '2 ', ...) map identically to float codes."""
    df = pd.DataFrame({
        "CLASE2": ["1", "2 ", "3", "4"],
        "P3K1": ["1", "", "", ""],
    })
    status = assign_labor_status(df)
    assert list(status) == ["F", "U", "N", "N"]


def test_assign_labor_status_no_formality_cols_all_informal():
    df = pd.DataFrame({"CLASE2": [1.0, 1.0]})
    status = assign_labor_status(df)
    assert list(status) == ["I", "I"]


def test_assign_labor_status_requires_clase2():
    with pytest.raises(ValueError, match="CLASE2"):
        assign_labor_status(pd.DataFrame({"P3K1": [1.0]}))


def test_assign_labor_status_ineligible_stays_nan():
    """CLASE2==0 (children under 12) maps to no state."""
    df = pd.DataFrame({"CLASE2": [0.0, 2.0], "P3K1": [np.nan, np.nan]})
    status = assign_labor_status(df)
    assert pd.isna(status.iloc[0])
    assert status.iloc[1] == "U"


# --------------------------------------------------------------------------- #
# quarterly_to_monthly_matrix                                                 #
# --------------------------------------------------------------------------- #

def _dense_monthly() -> np.ndarray:
    """A valid, diagonally-dominant monthly stochastic matrix (embeddable)."""
    M = np.array([
        [0.92, 0.03, 0.03, 0.02],
        [0.05, 0.88, 0.04, 0.03],
        [0.10, 0.15, 0.65, 0.10],
        [0.03, 0.05, 0.04, 0.88],
    ])
    np.testing.assert_allclose(M.sum(axis=1), 1.0)
    return M


def test_roundtrip_recovers_monthly_matrix():
    """P_Q = M^3, then logm(P_Q)/3 + expm recovers M (embeddable case)."""
    M = _dense_monthly()
    P_Q = M @ M @ M
    M_hat = quarterly_to_monthly_matrix(P_Q)
    np.testing.assert_allclose(M_hat, M, atol=1e-10)


def test_roundtrip_identity():
    I4 = np.eye(4)
    np.testing.assert_allclose(quarterly_to_monthly_matrix(I4), I4, atol=1e-12)


def test_recovered_matrix_is_stochastic_after_regularization():
    """A sparse P_Q whose principal log has negative off-diagonals exercises
    the IRW clipping path; the output must still be row-stochastic and >= 0."""
    from scipy import linalg as spla

    P_Q = np.array([
        [0.90, 0.10, 0.00, 0.00],
        [0.05, 0.90, 0.05, 0.00],
        [0.00, 0.05, 0.90, 0.05],
        [0.00, 0.00, 0.10, 0.90],
    ])
    # Sanity: this input really triggers the regularization branch.
    Q = spla.logm(P_Q).real / 3.0
    off = ~np.eye(4, dtype=bool)
    assert Q[off].min() < 0.0

    M_hat = quarterly_to_monthly_matrix(P_Q)
    assert M_hat.shape == (4, 4)
    assert (M_hat >= 0.0).all()
    np.testing.assert_allclose(M_hat.sum(axis=1), 1.0, atol=1e-12)


def test_zero_row_masked_not_fabricated():
    """An origin state with no observations (all-zero row) must come back as
    an all-zero row, while the remaining rows stay stochastic."""
    M = _dense_monthly()
    P_Q = M @ M @ M
    P_Q[2, :] = 0.0
    M_hat = quarterly_to_monthly_matrix(P_Q)
    np.testing.assert_allclose(M_hat[2, :], 0.0)
    others = np.delete(M_hat, 2, axis=0)
    assert (others >= 0.0).all()
    np.testing.assert_allclose(others.sum(axis=1), 1.0, atol=1e-12)


def test_non_square_raises():
    with pytest.raises(ValueError, match="square"):
        quarterly_to_monthly_matrix(np.ones((4, 3)))

def test_complex_logm_fallback():
    """A matrix with negative eigenvalues (like a permutation-heavy matrix)
    results in a complex logm. The function should take the real part and
    produce a valid row-stochastic monthly matrix."""
    # A block permutation matrix has eigenvalues including -1, so logm is complex
    P_Q = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    M_hat = quarterly_to_monthly_matrix(P_Q)
    assert M_hat.shape == (4, 4)
    assert (M_hat >= 0.0).all()
    np.testing.assert_allclose(M_hat.sum(axis=1), 1.0, atol=1e-12)
