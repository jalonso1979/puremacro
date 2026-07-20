"""Tests for SigmaObject — Python port of MAV/SigmaObject.m.

The MATLAB class is the reference; these tests pin down the same
identities using small hand-computed examples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.volatility import (
    SigmaObject, clean_correlation, project_psd,
)


@pytest.mark.pyodide_smoke
def test_sigma_object_constructs_covariance_from_sig_and_R():
    sig = np.array([2.0, 1.0])
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    S = SigmaObject(sig=sig, R=R)
    expected = np.array([[4.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(S.Sigma, expected, rtol=1e-12)


def test_sigma_object_var_equal_weights():
    sig = np.array([1.0, 1.0, 1.0])
    R = np.eye(3)
    S = SigmaObject(sig=sig, R=R)
    w = np.ones(3) / 3
    assert abs(S.var(w) - 1.0 / 3.0) < 1e-12
    assert abs(S.std(w) - np.sqrt(1.0 / 3.0)) < 1e-12


def test_sigma_object_diagonal_only_when_R_is_identity():
    sig = np.array([0.02, 0.01, 0.03])
    R = np.eye(3)
    S = SigmaObject(sig=sig, R=R)
    w = np.array([0.5, 0.3, 0.2])
    # With R = I the off-diagonal contribution is zero.
    assert abs(S.cov_premium_var(w)) < 1e-15
    assert abs(S.cov_share(w)) < 1e-12


def test_sigma_object_cov_premium_positive_when_correlated():
    sig = np.array([1.0, 1.0])
    R = np.array([[1.0, 0.6], [0.6, 1.0]])
    S = SigmaObject(sig=sig, R=R)
    w = np.array([0.5, 0.5])
    assert S.cov_premium_var(w) > 0
    assert 0.0 < S.cov_share(w) < 1.0


def test_sigma_object_diag_contrib_decomposes_var_no_cov():
    sig = np.array([0.02, 0.01, 0.03])
    R = np.eye(3)
    S = SigmaObject(sig=sig, R=R)
    w = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(
        S.diag_contrib(w).sum(), S.var_no_cov(w), rtol=1e-12,
    )


def test_sigma_object_pair_contrib_zero_diagonal():
    sig = np.array([1.0, 1.0])
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    S = SigmaObject(sig=sig, R=R)
    M = S.pair_contrib(np.array([0.7, 0.3]))
    assert M[0, 0] == 0 and M[1, 1] == 0


def test_sigma_object_pair_contrib_recovers_off_diagonal_total():
    """The MATLAB convention: off-diagonal variance contribution
    equals 2 · Σ_{i<j} pair_contrib_{ij}."""
    sig = np.array([1.0, 0.5, 1.5])
    R = np.array([[1.0, 0.4, 0.2],
                   [0.4, 1.0, 0.5],
                   [0.2, 0.5, 1.0]])
    S = SigmaObject(sig=sig, R=R)
    w = np.array([0.4, 0.3, 0.3])
    M = S.pair_contrib(w)
    iu = np.triu_indices(3, k=1)
    twice_off = 2.0 * float(M[iu].sum())
    assert abs(twice_off - S.cov_premium_var(w)) < 1e-12


def test_sigma_object_mean_corr():
    R = np.array([[1.0, 0.4, 0.2],
                   [0.4, 1.0, 0.6],
                   [0.2, 0.6, 1.0]])
    S = SigmaObject(sig=np.ones(3), R=R)
    expected = (0.4 + 0.2 + 0.6) / 3.0
    assert abs(S.mean_corr() - expected) < 1e-12


def test_sigma_object_pc1_share_reasonable():
    rng = np.random.default_rng(0)
    n, T = 5, 200
    f = rng.standard_normal(T)
    X = f[:, None] + 0.3 * rng.standard_normal((T, n))
    R = np.corrcoef(X, rowvar=False)
    sig = X.std(axis=0)
    S = SigmaObject(sig=sig, R=R)
    pc1 = S.pc1_share("R")
    # A dominant common factor should explain >50% of correlation matrix variance.
    assert 0.5 < pc1 < 1.0


def test_sigma_object_summary_columns():
    sig = np.array([0.02, 0.01])
    R = np.array([[1.0, 0.3], [0.3, 1.0]])
    S = SigmaObject(sig=sig, R=R, code="USA", period="2020Q1")
    out = S.summary(np.array([0.5, 0.5]))
    assert isinstance(out, pd.DataFrame)
    expected_cols = {"code", "period", "var_total", "var_diag", "var_cov",
                       "sigma_total", "sigma_diag", "cov_share",
                       "mean_corr", "pc1_share_R"}
    assert expected_cols.issubset(out.columns)
    assert out.iloc[0]["code"] == "USA"


def test_sigma_object_handles_nans_in_R():
    sig = np.array([1.0, 1.0, 1.0])
    R = np.array([[1.0, np.nan, 0.2],
                   [np.nan, 1.0, 0.5],
                   [0.2, 0.5, 1.0]])
    S = SigmaObject(sig=sig, R=R)
    assert not np.isnan(S.Sigma).any()
    assert np.isfinite(S.var(np.ones(3) / 3))


def test_project_psd_clips_negative_eigenvalues():
    # Symmetric but indefinite (eigenvalues 3 and -1).
    M = np.array([[1.0, 2.0], [2.0, 1.0]])
    P = project_psd(M)
    eigs = np.linalg.eigvalsh(P)
    assert eigs.min() >= -1e-10


def test_clean_correlation_enforces_diag_one_and_psd():
    R = np.array([[0.95, 0.6, 0.3],   # diag != 1, off-diag asymmetric below
                   [0.6,  1.05, np.nan],
                   [0.3,  np.nan, 1.0]])
    Rc = clean_correlation(R)
    np.testing.assert_allclose(np.diag(Rc), 1.0, atol=1e-12)
    eigs = np.linalg.eigvalsh(Rc)
    assert eigs.min() >= -1e-9
