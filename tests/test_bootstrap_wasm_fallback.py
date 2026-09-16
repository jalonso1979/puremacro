"""Bootstrap engines under a single-threaded runtime and thread-pool failure.

All five engines dispatch through :mod:`puremacro.inference._parallel`. Each
test here drives a public engine end to end and checks the *mechanism*, not
only the numbers, because the numbers cannot tell the paths apart: every
engine pre-draws its random material, so the thread pool and the serial loop
agree bit for bit by construction. Hence

* under ``PUREMACRO_THREADS=0`` the executor must never be constructed
  (a spy raises if it is, so a bypassed gate fails the test);
* under a live runtime with ``n_jobs=2`` it must be constructed exactly once
  (positive control), and the draws must still equal the ``n_jobs=1`` run;
* an exception raised inside ``refit_fn`` must propagate unchanged;
* ``n_jobs=0`` must raise before any replication runs;
* a thread pool that cannot start must warn and fall back once.
"""
from __future__ import annotations

import concurrent.futures
import os
import threading
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from puremacro.inference.block_bootstrap import block_bootstrap
from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
from puremacro.inference.wild_bootstrap import wild_bootstrap, wild_bootstrap_var
from puremacro.runtime import refresh
from puremacro.var.bootstrap import bootstrap_bands

_REAL_POOL = concurrent.futures.ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Runtime fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def no_threads(monkeypatch):
    """Simulate a WASM kernel: ``capabilities().threads`` is False for the test."""
    monkeypatch.setenv("PUREMACRO_THREADS", "0")
    caps = refresh()
    assert caps.threads is False and "threads" in caps.overridden
    yield
    # Restore the caller's environment first (monkeypatch would only do so
    # after this teardown) and re-detect against it; otherwise the cached
    # snapshot would keep threads=False, or disagree with a PUREMACRO_THREADS
    # that was already set in the shell.
    monkeypatch.undo()
    refresh()


@pytest.fixture
def live_threads(monkeypatch):
    """A runtime whose thread probe succeeds (skips on a host without threads)."""
    monkeypatch.delenv("PUREMACRO_THREADS", raising=False)
    if not refresh().threads:
        pytest.skip("host has no working OS threads")
    yield
    monkeypatch.undo()
    refresh()


class _PoolSpy:
    """Stand-in for ``ThreadPoolExecutor`` that counts constructions.

    With ``forbid=True`` a construction is an immediate test failure: the
    engines only swallow RuntimeError/OSError/ValueError from the pool, so
    the AssertionError escapes.
    """

    def __init__(self, *, forbid: bool = False):
        self.calls = 0
        self.max_workers: list[int | None] = []
        self.forbid = forbid

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.max_workers.append(kwargs.get("max_workers", args[0] if args else None))
        if self.forbid:
            raise AssertionError(
                "ThreadPoolExecutor must not be constructed when "
                "capabilities().threads is False"
            )
        return _REAL_POOL(*args, **kwargs)


# ---------------------------------------------------------------------------
# One runner per engine: fixed data, fixed seed, returns the bootstrap output
# ---------------------------------------------------------------------------

def _run_wild(n_jobs: int) -> np.ndarray:
    residuals = np.random.default_rng(42).normal(size=50)
    return wild_bootstrap(
        residuals, lambda e: np.array([e.mean(), e.std()]),
        n_boot=20, rng=np.random.default_rng(123), n_jobs=n_jobs,
    )


def _run_block(n_jobs: int) -> np.ndarray:
    residuals = np.random.default_rng(42).normal(size=60)
    return block_bootstrap(
        residuals, refit_fn=lambda b: np.array([b.mean(), b[0]]),
        B=25, rng=np.random.default_rng(99), n_jobs=n_jobs,
    )


def _run_bands(n_jobs: int) -> np.ndarray:
    Y = np.random.default_rng(42).normal(size=(40, 2))
    res = bootstrap_bands(
        Y, p=1, identify_fn=lambda A_list, Sigma, **kw: np.linalg.cholesky(Sigma),
        horizon=5, n_boot=20, rng=np.random.default_rng(7), n_jobs=n_jobs,
    )
    assert set(res) >= {"point", "lower", "upper", "draws"}
    assert res["draws"].shape == (20, 6, 2, 2)
    return res["draws"]


def _run_wild_var(n_jobs: int) -> np.ndarray:
    Y = np.random.default_rng(404).standard_normal((80, 2))
    point, lo, hi = wild_bootstrap_var(
        Y, p=1, horizon=3,
        impact_fn=lambda A_list, Sigma, resid: np.linalg.cholesky(Sigma),
        n_boot=20, seed=42, n_jobs=n_jobs,
    )
    return np.stack([point, lo, hi])


def _run_cum_irf(n_jobs: int) -> np.ndarray:
    rng = np.random.default_rng(505)
    entities = ["US", "DE", "FR", "JP"]
    times = pd.date_range("2000-01-01", periods=30, freq="QE")
    idx = pd.MultiIndex.from_product([entities, times], names=["code", "date"])
    df = pd.DataFrame({"y": rng.standard_normal(len(idx)),
                       "x": rng.standard_normal(len(idx))}, index=idx)
    res = cum_irf_block_bootstrap(
        df, y="y", x="x", horizons=[0, 1, 2], B=10, seed=42, n_jobs=n_jobs,
    )
    assert len(res) == 3
    return res[["cum_beta", "cum_lo", "cum_hi"]].to_numpy()


_ENGINES = {
    "wild_bootstrap": _run_wild,
    "block_bootstrap": _run_block,
    "bootstrap_bands": _run_bands,
    "wild_bootstrap_var": _run_wild_var,
    "cum_irf_block_bootstrap": _run_cum_irf,
}

#: Engines whose per-draw work is plain numpy arithmetic: bit-exact equality.
_EXACT = {"wild_bootstrap", "block_bootstrap"}


def _assert_same_draws(engine: str, got: np.ndarray, ref: np.ndarray) -> None:
    if engine in _EXACT:
        np.testing.assert_array_equal(got, ref)
    else:
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


# Engines whose refit_fn is the caller's (the VAR engines catch their own
# identification failures by design), driven with a caller-supplied refit.

def _wild_with(refit, n_jobs: int) -> np.ndarray:
    return wild_bootstrap(np.ones(10), refit, n_boot=8,
                          rng=np.random.default_rng(0), n_jobs=n_jobs)


def _block_with(refit, n_jobs: int) -> np.ndarray:
    return block_bootstrap(np.arange(10.0), refit_fn=refit, B=8,
                           rng=np.random.default_rng(0), n_jobs=n_jobs)


_REFIT_ENGINES = {"wild_bootstrap": _wild_with, "block_bootstrap": _block_with}


class _RefitBoom(Exception):
    """Distinct type, so only the user's own error can satisfy the assertion."""


# ---------------------------------------------------------------------------
# The runtime override itself
# ---------------------------------------------------------------------------

def test_capabilities_threads_override(monkeypatch):
    """PUREMACRO_THREADS=0 overrides the probe and is recorded as an override."""
    monkeypatch.setenv("PUREMACRO_THREADS", "0")
    caps = refresh()
    assert caps.threads is False
    assert "threads" in caps.overridden
    monkeypatch.delenv("PUREMACRO_THREADS")
    assert "threads" not in refresh().overridden


# ---------------------------------------------------------------------------
# The gate: serial under threads=False, pooled under threads=True
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", sorted(_ENGINES))
def test_no_executor_is_constructed_when_threads_are_disabled(engine, no_threads):
    """threads=False: n_jobs>1 runs serially and never touches ThreadPoolExecutor."""
    run = _ENGINES[engine]
    ref = run(1)
    spy = _PoolSpy(forbid=True)
    with patch("concurrent.futures.ThreadPoolExecutor", new=spy):
        got = run(4)
    assert spy.calls == 0
    _assert_same_draws(engine, got, ref)


@pytest.mark.parametrize("engine", sorted(_ENGINES))
def test_threaded_path_uses_the_pool_and_matches_serial(engine, live_threads):
    """Positive control: with threads available the pool IS used, once, and the
    draws equal the n_jobs=1 run because all randomness is drawn before dispatch."""
    run = _ENGINES[engine]
    ref = run(1)
    spy = _PoolSpy()
    with patch("concurrent.futures.ThreadPoolExecutor", new=spy):
        got = run(2)
    assert spy.calls == 1
    assert spy.max_workers == [2]
    _assert_same_draws(engine, got, ref)


def test_negative_n_jobs_means_every_core(live_threads):
    n_cpu = os.cpu_count() or 1
    if n_cpu == 1:
        pytest.skip("single-CPU host: n_jobs=-1 is the serial loop")
    spy = _PoolSpy()
    with patch("concurrent.futures.ThreadPoolExecutor", new=spy):
        _run_wild(-1)
    assert spy.max_workers == [n_cpu]


# ---------------------------------------------------------------------------
# n_jobs validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", sorted(_ENGINES))
def test_n_jobs_zero_raises_value_error(engine):
    """n_jobs=0 raised ValueError in 3.3.0 (from ThreadPoolExecutor) and must not
    silently become a serial run."""
    with pytest.raises(ValueError, match="n_jobs must be a positive integer"):
        _ENGINES[engine](0)


def test_n_jobs_zero_raises_before_any_replication_runs():
    calls: list[int] = []

    def refit(e):
        calls.append(1)
        return np.array([e.mean()])

    for run in _REFIT_ENGINES.values():
        with pytest.raises(ValueError, match="n_jobs"):
            run(refit, 0)
    assert calls == []


# ---------------------------------------------------------------------------
# Errors inside refit_fn are the caller's, on every path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", sorted(_REFIT_ENGINES))
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_refit_fn_exception_propagates_unchanged(engine, n_jobs, live_threads):
    """A failing refit_fn is reported, not swallowed by a silent serial re-run."""
    calls: list[int] = []
    lock = threading.Lock()

    def refit(e):
        with lock:
            calls.append(1)
            n = len(calls)
        if n == 3:
            raise _RefitBoom("replication 3 failed")
        return np.array([e.mean()])

    with pytest.raises(_RefitBoom, match="replication 3 failed"):
        _REFIT_ENGINES[engine](refit, n_jobs)
    # Never more evaluations than replications: no threaded-then-serial rerun.
    assert len(calls) <= 8


def test_refit_fn_failing_only_on_worker_threads_is_not_masked(live_threads):
    """The finding's case: a refit_fn that is not thread-safe raised off the main
    thread, the 3.4.0 fallback re-ran it serially and the caller never knew."""
    main = threading.get_ident()

    def refit(e):
        if threading.get_ident() != main:
            raise _RefitBoom("not thread-safe")
        return np.array([e.mean()])

    with pytest.raises(_RefitBoom, match="not thread-safe"):
        _wild_with(refit, 2)


# ---------------------------------------------------------------------------
# A pool that cannot start: warn once, run serially once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", sorted(_REFIT_ENGINES))
@pytest.mark.parametrize("exc", [
    RuntimeError("thread creation failed"),
    PermissionError("WASM thread creation blocked"),
    ValueError("bad executor argument"),
], ids=["RuntimeError", "PermissionError", "ValueError"])
def test_pool_that_cannot_start_warns_and_runs_serially_once(engine, exc, live_threads):
    calls: list[int] = []

    def refit(e):
        calls.append(1)
        return np.array([e.mean()])

    run = _REFIT_ENGINES[engine]
    ref = run(refit, 1)
    calls.clear()
    with patch("concurrent.futures.ThreadPoolExecutor", side_effect=exc):
        with pytest.warns(RuntimeWarning, match=f"{engine}: could not start a thread pool") as rec:
            got = run(refit, 4)
    np.testing.assert_array_equal(got, ref)
    assert len(calls) == 8, "replications were evaluated more than once"
    # Exactly one fallback warning (pytest.warns records every warning, so an
    # unrelated one must not be mistaken for it), and it points at the code
    # that called the engine, not at the helper.
    fallback = [w for w in rec if issubclass(w.category, RuntimeWarning)
                and "could not start a thread pool" in str(w.message)]
    assert len(fallback) == 1
    assert fallback[0].filename == __file__


def test_pool_construction_errors_of_other_types_propagate(live_threads):
    """Only the documented executor failures are absorbed; anything else is a bug
    the caller must see."""
    with patch("concurrent.futures.ThreadPoolExecutor", side_effect=TypeError("bogus")):
        with pytest.raises(TypeError, match="bogus"):
            _run_wild(2)


# ---------------------------------------------------------------------------
# wild_bootstrap input coercion (3.3.0 contract)
# ---------------------------------------------------------------------------

def test_wild_bootstrap_accepts_array_likes():
    """Lists, tuples, 2-D lists and pandas objects are coerced like in 3.3.0."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=12)

    def refit(e):
        return np.array([e.mean(), e.std()])

    ref = wild_bootstrap(base, refit, n_boot=6, rng=np.random.default_rng(1))
    for alike in (list(base), tuple(base), pd.Series(base)):
        got = wild_bootstrap(alike, refit, n_boot=6, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(got, ref)

    base2 = rng.normal(size=(12, 3))
    ref2 = wild_bootstrap(base2, lambda e: e.mean(axis=0), n_boot=6,
                          rng=np.random.default_rng(2))
    for alike in (base2.tolist(), pd.DataFrame(base2)):
        got2 = wild_bootstrap(alike, lambda e: e.mean(axis=0), n_boot=6,
                              rng=np.random.default_rng(2))
        # A DataFrame coerces to a Fortran-ordered array, so the axis-0 mean
        # sums in a different order: same weights and values, 1-ulp rounding.
        np.testing.assert_allclose(got2, ref2, rtol=1e-13, atol=0)


def test_wild_bootstrap_hands_refit_fn_a_float_ndarray():
    """Integer and pandas inputs reach refit_fn as float64 ndarrays, so ndarray
    idioms such as ``e[:, None]`` keep working."""
    seen: list[tuple[type, np.dtype]] = []

    def refit(e):
        seen.append((type(e), e.dtype))
        return np.linalg.lstsq(np.ones((4, 1)), e[:, None], rcond=None)[0].ravel()

    wild_bootstrap(pd.Series([1, -2, 3, -4]), refit, n_boot=3, rng=np.random.default_rng(0))
    wild_bootstrap([1, -2, 3, -4], refit, n_boot=3, rng=np.random.default_rng(0))
    assert len(seen) == 6
    assert all(t is np.ndarray and dt == np.float64 for t, dt in seen)
