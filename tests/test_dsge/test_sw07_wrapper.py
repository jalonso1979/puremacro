"""Tests for the SW07 thin-wrapper layer over the generic Bayesian DSGE engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


FIXTURE = Path(__file__).parent.parent / "fixtures" / "sw07_parity_seed0_200draws.npz"


@pytest.mark.slow
def test_sw07_parity_short_chain():
    """estimate_sw07(seed=0, n_chains=1, n_draws=200, burn_in=50) against the
    frozen pre-0.53.0 reference: posterior-function parity deterministically,
    structural parity exactly, sampler invariants on a fresh short chain.

    The original bit-for-bit check (atol=1e-10 on every draw) only holds on
    the software stack that generated the fixture: last-bit float
    differences in the L-BFGS-B mode refinement (which does not even take
    the same branch on every platform — it converges on some stacks and
    aborts to the initial-params fallback on others), SVD, and Cholesky
    are amplified chaotically by MH accept/reject, so identical seeds
    produce entirely different draw trajectories elsewhere.

    What IS cross-platform stable is the log-posterior FUNCTION: the
    fixture's ``log_posterior_trace`` entries are plain evaluations at the
    fixture's own draws, so recomputing them today involves no optimizer
    branch and no accept/reject amplification — only last-bit Kalman noise
    (observed ~1e-3 log points across numpy/scipy/BLAS drift). Matching
    them pins the entire wrapper wiring against the pre-0.53.0 reference:
    bundled-data loading, observation equation, priors, and fixed
    calibrated params. Any real regression in those moves the
    log-posterior by orders of magnitude more than the 0.05 tolerance.
    """
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import (
        _FIXED_PARAMS, _load_bundled_data, estimate_sw07,
    )
    from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names

    ref = np.load(FIXTURE)
    names = param_names()
    lb = np.array([PRIORS[n]["lb"] for n in names])
    ub = np.array([PRIORS[n]["ub"] for n in names])

    # --- Posterior-function parity vs the frozen reference --------------
    # Recompute the log-posterior at a spread of the fixture's own draws
    # and match the fixture's recorded values (deterministic on every
    # platform, unlike the chain trajectory).
    y = _load_bundled_data()[list(OBSERVED_VARS)].to_numpy()
    neg_log_post = _make_neg_log_posterior(
        y, make_state_space, PRIORS, names, _FIXED_PARAMS,
    )
    probe_idx = list(range(0, 200, 20)) + [199]
    lp_recomputed = np.array(
        [-neg_log_post(ref["draws"][0, k]) for k in probe_idx]
    )
    np.testing.assert_allclose(
        lp_recomputed, ref["log_posterior_trace"][0, probe_idx],
        rtol=0, atol=0.05,
        err_msg="log-posterior at frozen reference draws no longer matches "
                "the pre-0.53.0 values: wrapper wiring (data / observation "
                "equation / priors / fixed params) has changed",
    )

    # --- Fresh short chain: structure + sampler invariants --------------
    res = estimate_sw07(seed=0, n_chains=1, n_draws=200, burn_in=50)

    assert tuple(res.param_names) == tuple(str(n) for n in ref["param_names"])
    assert res.param_names == names
    assert res.draws.shape == ref["draws"].shape
    assert res.log_posterior_trace.shape == ref["log_posterior_trace"].shape
    assert res.mode_hessian_inv.shape == ref["mode_hessian_inv"].shape
    assert len(res.accept_rates) == len(ref["accept_rates"])

    assert np.isfinite(res.draws).all()
    assert np.isfinite(res.log_posterior_trace).all()
    assert (res.draws >= lb).all() and (res.draws <= ub).all()
    assert 0.05 <= res.accept_rates[0] <= 0.65
    mode_vec = np.array([res.mode[n] for n in names])
    assert ((mode_vec >= lb) & (mode_vec <= ub)).all()
    assert np.isfinite(res.mode_hessian_inv).all()
    assert np.allclose(res.mode_hessian_inv, res.mode_hessian_inv.T, atol=1e-8)


def test_sw07posteriorresult_is_alias_for_dsge():
    from puremacro.dsge._results import DSGEPosteriorResult, SW07PosteriorResult
    assert SW07PosteriorResult is DSGEPosteriorResult


def test_dsge_posterior_result_default_model_name_is_unknown():
    import numpy as np
    from puremacro.dsge._results import DSGEPosteriorResult
    res = DSGEPosteriorResult(
        draws=np.zeros((1, 5, 3)),
        param_names=("a", "b", "c"),
        log_posterior_trace=np.zeros((1, 5)),
        accept_rates=(0.25,),
        mode={"a": 0.0, "b": 0.0, "c": 0.0},
        mode_hessian_inv=np.eye(3),
        n_burn_in=0,
        data_n_obs=10,
        seed=0,
    )
    assert res.model_name == "unknown"
    res2 = DSGEPosteriorResult(
        draws=np.zeros((1, 5, 3)),
        param_names=("a", "b", "c"),
        log_posterior_trace=np.zeros((1, 5)),
        accept_rates=(0.25,),
        mode={"a": 0.0, "b": 0.0, "c": 0.0},
        mode_hessian_inv=np.eye(3),
        n_burn_in=0,
        data_n_obs=10,
        seed=0,
        model_name="MyModel",
    )
    assert res2.model_name == "MyModel"


def test_fertility_solution_dataclass_is_frozen_with_expected_fields():
    import dataclasses
    import numpy as np
    import pytest
    from puremacro.dsge._results import FertilitySolution

    res = FertilitySolution(
        ss={"c": 1.0, "k": 5.0},
        params={"alpha": 0.4},
        G=np.eye(5),
        N=np.zeros((5, 3)),
        F=np.zeros((7, 5)),
        L=np.zeros((7, 3)),
        klein_solution=None,
        var_names=("a", "mun", "ph", "k", "n", "c", "y", "l_w", "u", "i", "b", "l_o"),
        shock_names=("ea", "ep", "en"),
    )
    assert res.ss["c"] == 1.0
    assert res.G.shape == (5, 5)
    assert res.shock_names == ("ea", "ep", "en")
    assert dataclasses.is_dataclass(res)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.ss = {}
