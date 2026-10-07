"""Tests for the SW07 thin-wrapper layer over the generic Bayesian DSGE engine."""
from __future__ import annotations

from importlib import resources

import numpy as np
import pytest


# The same file the dsge_estimation replication cases read: shipped as package
# data and rebuilt by ``python tools/build_sw07_data.py fixture``.
FIXTURE = resources.files("puremacro.replication.data") / "sw07_parity_seed0_200draws.npz"


def _load_reference() -> dict[str, np.ndarray]:
    """Arrays of the package-data SW07 fixture (works from an installed wheel)."""
    with FIXTURE.open("rb") as fh, np.load(fh) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.slow
def test_sw07_parity_short_chain():
    """estimate_sw07(seed=0, n_chains=1, n_draws=200, burn_in=500) against the
    SW07 fixture: posterior-function parity deterministically, structural
    parity exactly, sampler invariants on a fresh short chain.

    The fixture (``puremacro/replication/data/sw07_parity_seed0_200draws.npz``,
    written by ``python tools/build_sw07_data.py fixture``) holds the optimised
    posterior mode of the SW07 model on the bundled data, the inverse Hessian
    there, and 200 random-walk Metropolis draws around the mode (every 50th of
    10,000 after a 2,000-draw burn-in), with ``log_posterior_trace`` the log
    posterior at each kept draw. It also records the SHA-256 of the
    ``_sw07_data.csv`` it was built on. Through 4.3.0 the draws were a frozen
    pre-0.53.0 chain on the old data; the 2026-09-30 SW07 fix (markup MA terms,
    rebuilt data) replaced them.

    Comparing draw trajectories bit for bit only holds on the software stack
    that generated them: last-bit float differences in the mode refinement,
    SVD and Cholesky are amplified chaotically by MH accept/reject, so the
    same seed gives different trajectories elsewhere. What IS cross-platform
    stable is the log-posterior FUNCTION: the stored trace entries are plain
    evaluations at the stored draws, so recomputing them involves no optimizer
    branch and no accept/reject, only last-bit Kalman noise (about 1e-3 log
    points across numpy/scipy/BLAS versions). Matching them pins the whole
    wrapper wiring: bundled-data loading, observation equation, priors and
    fixed calibrated params. A real regression in any of those moves the log
    posterior by far more than the 0.05 tolerance.

    ``burn_in`` is 500 because ``random_walk_metropolis`` adapts its scalar
    proposal scale only on ``(it + 1) % 100 == 0``. With ``burn_in < 100``
    adaptation never fires, the proposal keeps the oversized
    ``diag(prior_stds**2)`` fallback scale, and the chain can reject every
    draw (measured at ``burn_in=50`` on the pre-fix model: acceptance 0.000).
    That is a known sampler limitation, documented in docs/dsge_estimation.md.
    """
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import (
        _FIXED_PARAMS, _load_bundled_data, estimate_sw07,
    )
    from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names

    ref = _load_reference()
    names = param_names()
    lb = np.array([PRIORS[n]["lb"] for n in names])
    ub = np.array([PRIORS[n]["ub"] for n in names])

    # --- Posterior-function parity vs the fixture -------------------------
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
        err_msg="log-posterior at the fixture's draws no longer matches the "
                "values stored by tools/build_sw07_data.py: wrapper wiring "
                "(data / observation equation / priors / fixed params) has "
                "changed, or the fixture needs rebuilding",
    )

    # --- Fresh short chain: structure + sampler invariants --------------
    res = estimate_sw07(seed=0, n_chains=1, n_draws=200, burn_in=500)

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
