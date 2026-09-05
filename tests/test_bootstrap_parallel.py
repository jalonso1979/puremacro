"""Tests for multi-core parallel dispatch (n_jobs) and vectorized bootstrap draws."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference.block_bootstrap import block_bootstrap
from puremacro.inference.wild_bootstrap import wild_bootstrap
from puremacro.var.bootstrap import bootstrap_bands


def test_wild_bootstrap_parallel_equivalence():
    rng_seq = np.random.default_rng(101)
    rng_par = np.random.default_rng(101)
    residuals = np.array([0.5, -0.3, 1.2, -0.8, 0.1, -0.4, 0.9, -0.2])

    def refit(e):
        return np.array([e.mean(), e.std()])

    draws_seq = wild_bootstrap(residuals, refit, n_boot=50, rng=rng_seq, n_jobs=1)
    draws_par = wild_bootstrap(residuals, refit, n_boot=50, rng=rng_par, n_jobs=2)

    assert draws_seq.shape == (50, 2)
    assert draws_par.shape == (50, 2)
    np.testing.assert_allclose(draws_seq, draws_par)


def test_block_bootstrap_parallel_equivalence():
    rng_seq = np.random.default_rng(202)
    rng_par = np.random.default_rng(202)
    residuals = np.random.default_rng(1).standard_normal(40)

    def refit(e):
        return np.array([e.mean(), np.median(e)])

    draws_seq = block_bootstrap(residuals, refit_fn=refit, B=30, block_length=3, rng=rng_seq, n_jobs=1)
    draws_par = block_bootstrap(residuals, refit_fn=refit, B=30, block_length=3, rng=rng_par, n_jobs=2)

    assert draws_seq.shape == (30, 2)
    assert draws_par.shape == (30, 2)
    np.testing.assert_allclose(draws_seq, draws_par)


def test_var_bootstrap_bands_parallel_equivalence():
    rng = np.random.default_rng(303)
    Y = rng.standard_normal((60, 2))

    def id_fn(A_list, Sigma):
        return np.linalg.cholesky(Sigma)

    rng_seq = np.random.default_rng(42)
    rng_par = np.random.default_rng(42)

    res_seq = bootstrap_bands(Y, 1, id_fn, horizon=3, n_boot=20, rng=rng_seq, n_jobs=1)
    res_par = bootstrap_bands(Y, 1, id_fn, horizon=3, n_boot=20, rng=rng_par, n_jobs=2)

    assert res_seq["point"].shape == (4, 2, 2)
    assert res_par["point"].shape == (4, 2, 2)
    np.testing.assert_allclose(res_seq["draws"], res_par["draws"])
    np.testing.assert_allclose(res_seq["lower"], res_par["lower"])
    np.testing.assert_allclose(res_seq["upper"], res_par["upper"])


def test_wild_bootstrap_var_parallel_equivalence():
    from puremacro.inference.wild_bootstrap import wild_bootstrap_var

    rng = np.random.default_rng(404)
    Y = rng.standard_normal((80, 2))

    def id_fn(A_list, Sigma, resid):
        return np.linalg.cholesky(Sigma)

    pt_seq, lo_seq, hi_seq = wild_bootstrap_var(Y, p=1, horizon=3, impact_fn=id_fn, n_boot=20, seed=42, n_jobs=1)
    pt_par, lo_par, hi_par = wild_bootstrap_var(Y, p=1, horizon=3, impact_fn=id_fn, n_boot=20, seed=42, n_jobs=2)

    np.testing.assert_allclose(pt_seq, pt_par)
    np.testing.assert_allclose(lo_seq, lo_par)
    np.testing.assert_allclose(hi_seq, hi_par)


def test_cum_irf_block_bootstrap_parallel():
    import pandas as pd
    from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap

    rng = np.random.default_rng(505)
    entities = ["US", "DE", "FR", "JP"]
    times = pd.date_range("2000-01-01", periods=30, freq="QE")
    idx = pd.MultiIndex.from_product([entities, times], names=["code", "date"])
    df = pd.DataFrame({
        "y": rng.standard_normal(len(idx)),
        "x": rng.standard_normal(len(idx)),
    }, index=idx)

    res_seq = cum_irf_block_bootstrap(df, y="y", x="x", horizons=[0, 1, 2], B=10, seed=42, n_jobs=1)
    res_par = cum_irf_block_bootstrap(df, y="y", x="x", horizons=[0, 1, 2], B=10, seed=42, n_jobs=2)

    assert len(res_seq) == 3
    assert len(res_par) == 3
    np.testing.assert_allclose(res_seq["cum_beta"].values, res_par["cum_beta"].values)
    np.testing.assert_allclose(res_seq["cum_lo"].values, res_par["cum_lo"].values)
    np.testing.assert_allclose(res_seq["cum_hi"].values, res_par["cum_hi"].values)

