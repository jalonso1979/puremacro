"""Tests for WASM-aware bootstrap fallback when threading is disabled or restricted."""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from puremacro.runtime import capabilities, refresh
from puremacro.inference.wild_bootstrap import wild_bootstrap, wild_bootstrap_var
from puremacro.inference.block_bootstrap import block_bootstrap
from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
from puremacro.var.bootstrap import bootstrap_bands


def test_capabilities_threads_override():
    """Verify PUREMACRO_THREADS env var correctly overrides capabilities().threads."""
    os.environ["PUREMACRO_THREADS"] = "0"
    try:
        caps = refresh()
        assert caps.threads is False
        assert "threads" in caps.overridden
    finally:
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()


def test_wild_bootstrap_wasm_fallback():
    """Verify wild_bootstrap runs cleanly and deterministically when threads=False."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(size=50)

    def refit_fn(e):
        return np.array([np.mean(e), np.std(e)])

    # Reference with n_jobs=1
    ref_draws = wild_bootstrap(residuals, refit_fn, n_boot=20, rng=np.random.default_rng(123), n_jobs=1)

    # Simulated WASM environment: PUREMACRO_THREADS=0
    os.environ["PUREMACRO_THREADS"] = "0"
    try:
        refresh()
        draws_multi = wild_bootstrap(
            residuals, refit_fn, n_boot=20, rng=np.random.default_rng(123), n_jobs=4
        )
        np.testing.assert_allclose(draws_multi, ref_draws, rtol=1e-12)
    finally:
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()


def test_block_bootstrap_wasm_fallback():
    """Verify block_bootstrap runs cleanly when threads=False and when ThreadPoolExecutor fails."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(size=60)

    def refit_fn(boot):
        return np.array([np.mean(boot)])

    # Reference with n_jobs=1
    ref_draws = block_bootstrap(residuals, refit_fn=refit_fn, B=25, rng=np.random.default_rng(99), n_jobs=1)

    # Simulated WASM: threads=False
    os.environ["PUREMACRO_THREADS"] = "0"
    try:
        refresh()
        draws_multi = block_bootstrap(
            residuals, refit_fn=refit_fn, B=25, rng=np.random.default_rng(99), n_jobs=2
        )
        np.testing.assert_allclose(draws_multi, ref_draws, rtol=1e-12)
    finally:
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()


def test_bootstrap_bands_wasm_fallback():
    """Verify var/bootstrap.py bootstrap_bands runs cleanly under simulated WASM."""
    rng = np.random.default_rng(42)
    T, n = 40, 2
    Y = rng.normal(size=(T, n))

    def identify_fn(A_list, Sigma, **kwargs):
        return np.linalg.cholesky(Sigma)

    # Test under PUREMACRO_THREADS=0 with n_jobs=2
    os.environ["PUREMACRO_THREADS"] = "0"
    try:
        refresh()
        res = bootstrap_bands(
            Y, p=1, identify_fn=identify_fn, horizon=5, n_boot=20, rng=np.random.default_rng(7), n_jobs=2
        )
        assert "point" in res
        assert "lower" in res
        assert "upper" in res
        assert res["draws"].shape == (20, 6, n, n)
    finally:
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()


def test_wild_bootstrap_var_wasm_fallback():
    """Verify wild_bootstrap_var runs cleanly and serially when threads=False."""
    rng = np.random.default_rng(404)
    Y = rng.standard_normal((80, 2))

    def id_fn(A_list, Sigma, resid):
        return np.linalg.cholesky(Sigma)

    # Reference serial execution with n_jobs=1
    pt_seq, lo_seq, hi_seq = wild_bootstrap_var(
        Y, p=1, horizon=3, impact_fn=id_fn, n_boot=20, seed=42, n_jobs=1
    )

    # Simulated WASM environment: PUREMACRO_THREADS=0
    os.environ["PUREMACRO_THREADS"] = "0"
    try:
        refresh()
        assert capabilities().threads is False
        pt_wasm, lo_wasm, hi_wasm = wild_bootstrap_var(
            Y, p=1, horizon=3, impact_fn=id_fn, n_boot=20, seed=42, n_jobs=4
        )
        np.testing.assert_allclose(pt_seq, pt_wasm, rtol=1e-12)
        np.testing.assert_allclose(lo_seq, lo_wasm, rtol=1e-12)
        np.testing.assert_allclose(hi_seq, hi_wasm, rtol=1e-12)
    finally:
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()


def test_cum_irf_block_bootstrap_wasm_fallback():
    """Verify cum_irf_block_bootstrap runs cleanly and serially when threads=False."""
    import pandas as pd

    rng = np.random.default_rng(505)
    entities = ["US", "DE", "FR", "JP"]
    times = pd.date_range("2000-01-01", periods=30, freq="QE")
    idx = pd.MultiIndex.from_product([entities, times], names=["code", "date"])
    df = pd.DataFrame({
        "y": rng.standard_normal(len(idx)),
        "x": rng.standard_normal(len(idx)),
    }, index=idx)

    # Reference serial execution with n_jobs=1
    res_seq = cum_irf_block_bootstrap(
        df, y="y", x="x", horizons=[0, 1, 2], B=10, seed=42, n_jobs=1
    )

    # Simulated WASM environment: PUREMACRO_THREADS=0
    os.environ["PUREMACRO_THREADS"] = "0"
    try:
        refresh()
        assert capabilities().threads is False
        res_wasm = cum_irf_block_bootstrap(
            df, y="y", x="x", horizons=[0, 1, 2], B=10, seed=42, n_jobs=4
        )
        assert len(res_wasm) == 3
        np.testing.assert_allclose(res_seq["cum_beta"].values, res_wasm["cum_beta"].values, rtol=1e-12)
        np.testing.assert_allclose(res_seq["cum_lo"].values, res_wasm["cum_lo"].values, rtol=1e-12)
        np.testing.assert_allclose(res_seq["cum_hi"].values, res_wasm["cum_hi"].values, rtol=1e-12)
    finally:
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()


def test_threadpool_exception_defensive_fallback():
    """Verify defensive fallback to serial when ThreadPoolExecutor raises RuntimeError."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(size=30)

    def refit_fn(e):
        return np.array([np.sum(e)])

    with patch("concurrent.futures.ThreadPoolExecutor", side_effect=RuntimeError("thread creation failed")):
        # Even with capabilities().threads True, if ThreadPoolExecutor raises, it should fall back to serial
        draws = wild_bootstrap(residuals, refit_fn, n_boot=10, rng=np.random.default_rng(1), n_jobs=2)
        assert len(draws) == 10
