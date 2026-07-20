"""Tests for puremacro.var.identify.narrative_sign — AD-RR (2018) narrative
sign restrictions.

Strategy
--------
- Planted-truth DGP: a stable 3-variable VAR(1) with known impact matrix
  B0 whose column 0 has the sign pattern (+, +, -), driven by i.i.d.
  N(0, 1) structural shocks with one huge planted realization
  eps[t*, 0] = 5.0. Traditional sign restrictions pin the pattern at
  h = 0, so the "target shock" label is meaningful.

    * CORRECT narrative restrictions (Type I shock sign + Type III
      overwhelming HD dominance on t*) must (a) strictly tighten the
      average pointwise band width for the target-shock column relative
      to plain sign restrictions, and (b) cover the true impact
      responses at h = 0 (plus a majority of cells at longer horizons —
      set identification does not guarantee pointwise coverage
      everywhere).
    * FALSE-but-satisfiable narrative restriction (shock 1 claimed most
      important for variable 0 on the huge-shock date) must cut
      acceptance and shift the weighted median impact away from truth.
    * IMPOSSIBLE narrative restriction (wrong shock sign on the huge
      realization; also a contradictory +/- pair on the same date) must
      raise the diagnostic RuntimeError naming the binding restriction.

- Diagnostics contract: weights (length, positivity, closed-form
  2**m constancy in the pure-Type-I case), Kish ESS bounds,
  per-restriction fail counts, frozen-dataclass result, summary().

- Adapter: a NarrativeEvent (monetary, sign +1, mid-quarter announcement
  date) must reproduce exactly the run with the equivalent
  (row_index, 0, +1) tuple under the same seed.

- Reproducibility: identical results under the same seed; the Haar
  stream is spawn-separated from the weight simulator, so
  n_traditional_accepted is invariant to the restriction set.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.var.identify import (
    NarrativeRestriction,
    NarrativeSignSVARResult,
    narrative_sign_svar,
)

# ---------------------------------------------------------------------------
# Planted-truth DGP
# ---------------------------------------------------------------------------

T, N = 400, 3
TSTAR = 250  # row of the huge planted shock (eps[TSTAR, 0] = 5.0)
A_TRUE = np.array([[0.5, 0.1, 0.0],
                   [0.0, 0.4, 0.1],
                   [0.1, 0.0, 0.45]])
B0_TRUE = np.array([[1.0, 0.0, 0.3],
                    [0.6, 1.0, -0.2],
                    [-0.4, 0.3, 1.0]])


def _simulate():
    rng = np.random.default_rng(42)
    eps = rng.standard_normal((T, N))
    eps[TSTAR, 0] = 5.0
    Y = np.zeros((T, N))
    for t in range(1, T):
        Y[t] = A_TRUE @ Y[t - 1] + B0_TRUE @ eps[t]
    return Y, eps


Y_SIM, EPS_SIM = _simulate()

SM = np.zeros((N, N))
SM[:, 0] = [+1, +1, -1]
SIGN_MATRIX = {0: SM}

CORRECT_RESTRICTIONS = [
    (TSTAR, 0, +1),
    NarrativeRestriction(kind="hd_dominance", date=TSTAR, shock=0,
                         variable=0, dominance="overwhelming"),
]


def _true_irf_col0(horizon: int) -> np.ndarray:
    Phi = [np.eye(N)]
    for _ in range(horizon):
        Phi.append(Phi[-1] @ A_TRUE)
    return np.stack([Ph @ B0_TRUE for Ph in Phi])[:, :, 0]  # (H+1, n)


def _run(restrictions, *, n_draws=2000, seed=0, **kw):
    return narrative_sign_svar(
        Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
        restrictions=restrictions, n_draws=n_draws,
        n_weight_sims=200, seed=seed, **kw,
    )


# ---------------------------------------------------------------------------
# Planted truth: tighten + cover
# ---------------------------------------------------------------------------

def test_correct_narrative_restriction_tightens_bands():
    plain = _run([])
    narr = _run(CORRECT_RESTRICTIONS)
    # Haar stream is invariant to the restriction set for a given seed.
    assert narr.n_traditional_accepted == plain.n_traditional_accepted
    assert 0 < narr.n_narrative_accepted < narr.n_traditional_accepted
    w_plain = (plain.irf_upper - plain.irf_lower)[:, :, 0].mean()
    w_narr = (narr.irf_upper - narr.irf_lower)[:, :, 0].mean()
    assert w_narr < w_plain  # the AD-RR headline: the identified set tightens


def test_correct_narrative_restriction_covers_truth():
    narr = _run(CORRECT_RESTRICTIONS)
    truth = _true_irf_col0(8)
    lo, hi = narr.irf_lower[:, :, 0], narr.irf_upper[:, :, 0]
    inside = (lo <= truth) & (truth <= hi)
    assert inside[0].all()          # impact responses covered
    assert inside.mean() >= 0.6     # majority of (h, i) cells covered


def test_empty_restrictions_reproduce_traditional_with_unit_weights():
    plain = _run([])
    assert plain.n_narrative_accepted == plain.n_traditional_accepted
    assert np.all(plain.weights == 1.0)
    assert plain.restriction_labels == ()
    np.testing.assert_allclose(plain.ess, plain.n_narrative_accepted)


# ---------------------------------------------------------------------------
# False narrative restrictions
# ---------------------------------------------------------------------------

def test_false_hd_restriction_cuts_acceptance_and_shifts_bands():
    """Claiming shock 1 was the most important driver of variable 0 on
    the huge-shock date is false but satisfiable by misaligned rotations:
    acceptance drops and the weighted median impact moves off the truth."""
    correct = _run(CORRECT_RESTRICTIONS, n_draws=4000)
    false = _run([NarrativeRestriction(kind="hd_dominance", date=TSTAR,
                                       shock=1, variable=0,
                                       dominance="most")], n_draws=4000)
    assert false.n_narrative_accepted < correct.n_narrative_accepted
    truth_00 = B0_TRUE[0, 0]
    err_false = abs(false.irf_median[0, 0, 0] - truth_00)
    err_correct = abs(correct.irf_median[0, 0, 0] - truth_00)
    assert err_false > err_correct


def test_false_shock_sign_collapses_to_diagnostic_error():
    with pytest.raises(RuntimeError, match=r"narrative_sign_svar.*Most binding.*shock_sign"):
        _run([(TSTAR, 0, -1)])


def test_contradictory_pair_zero_acceptance_error():
    with pytest.raises(RuntimeError, match="Most binding"):
        _run([(TSTAR, 0, +1), (TSTAR, 0, -1)])


# ---------------------------------------------------------------------------
# Weights / ESS diagnostics
# ---------------------------------------------------------------------------

def test_pure_type1_weights_are_closed_form_constant():
    res = _run([(TSTAR, 0, +1), (100, 1, int(np.sign(EPS_SIM[100, 1]) or 1))])
    # 2 distinct (date, shock) pairs -> omega = 0.25 -> weight = 4 for all.
    assert res.weights.shape == (res.n_narrative_accepted,)
    np.testing.assert_allclose(res.weights, 4.0)
    np.testing.assert_allclose(res.ess, res.n_narrative_accepted)


def test_mixed_restriction_ess_bounds_and_fail_counts():
    res = _run(CORRECT_RESTRICTIONS)
    assert res.weights.shape == (res.n_narrative_accepted,)
    assert np.all(res.weights > 0)
    assert 0 < res.ess <= res.n_narrative_accepted + 1e-9
    assert len(res.restriction_labels) == 2
    assert len(res.restriction_fail_counts) == 2
    assert all(0 <= f <= res.n_traditional_accepted
               for f in res.restriction_fail_counts)


# ---------------------------------------------------------------------------
# NarrativeEvent adapter
# ---------------------------------------------------------------------------

def _dates():
    return pd.period_range("1950Q1", periods=T, freq="Q").to_timestamp()


def test_narrative_event_adapter_matches_tuple_run():
    from puremacro.narrative import NarrativeEvent

    dates = _dates()
    event = NarrativeEvent(
        date=dates[TSTAR] + pd.Timedelta(days=5),  # mid-quarter announcement
        country="USA", magnitude=1.0, magnitude_unit="z",
        target="policy_rate", subtarget=None, sign=+1, confidence=1.0,
        source_text="Volcker-style episode", source_url="",
        scoring_method="manual", kind="monetary",
    )
    via_event = narrative_sign_svar(
        Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
        restrictions=[event], dates=dates, n_draws=1000, seed=3,
    )
    via_tuple = narrative_sign_svar(
        Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
        restrictions=[(TSTAR, 0, +1)], n_draws=1000, seed=3,
    )
    np.testing.assert_array_equal(via_event.irf_median, via_tuple.irf_median)
    np.testing.assert_array_equal(via_event.weights, via_tuple.weights)
    assert via_event.n_narrative_accepted == via_tuple.n_narrative_accepted


def test_narrative_event_with_ambiguous_sign_rejected():
    from puremacro.narrative import NarrativeEvent

    event = NarrativeEvent(
        date="1979-10-06", country="USA", magnitude=1.0, magnitude_unit="z",
        target="policy_rate", subtarget=None, sign=0, confidence=1.0,
        source_text="", source_url="", scoring_method="manual",
        kind="monetary",
    )
    with pytest.raises(ValueError, match="sign=0"):
        narrative_sign_svar(Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
                            restrictions=[event], dates=_dates(),
                            n_draws=100, seed=0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducible_under_seed():
    a = _run(CORRECT_RESTRICTIONS, seed=7)
    b = _run(CORRECT_RESTRICTIONS, seed=7)
    np.testing.assert_array_equal(a.irf_median, b.irf_median)
    np.testing.assert_array_equal(a.irf_lower, b.irf_lower)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.n_narrative_accepted == b.n_narrative_accepted

    c = _run(CORRECT_RESTRICTIONS, seed=8)
    assert not np.array_equal(a.irf_median, c.irf_median)


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

def test_result_is_frozen_dataclass_with_summary():
    res = _run(CORRECT_RESTRICTIONS, n_draws=1000)
    assert isinstance(res, NarrativeSignSVARResult)
    assert dataclasses.is_dataclass(res)
    assert res.irf_median.shape == (9, N, N)
    assert res.irf_lower.shape == (9, N, N)
    assert res.irf_upper.shape == (9, N, N)
    with pytest.raises(Exception):
        res.ess = 0.0  # frozen
    s = res.summary()
    assert "Narrative-sign SVAR" in s
    assert "hd_dominance" in s
    assert "ESS" in s


def test_sign_matrix_vector_form_equivalent_to_column0_matrix():
    vec = np.array([+1.0, +1.0, -1.0])
    res_vec = narrative_sign_svar(Y_SIM, p=1, horizon=8, sign_matrix={0: vec},
                                  restrictions=[(TSTAR, 0, +1)],
                                  n_draws=800, seed=1)
    res_mat = narrative_sign_svar(Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
                                  restrictions=[(TSTAR, 0, +1)],
                                  n_draws=800, seed=1)
    np.testing.assert_array_equal(res_vec.irf_median, res_mat.irf_median)


# ---------------------------------------------------------------------------
# Input validation — diagnostic errors name the calling function
# ---------------------------------------------------------------------------

def test_bad_restriction_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        NarrativeRestriction(kind="hd_maximal", date=10, shock=0)


def test_hd_dominance_requires_variable():
    with pytest.raises(ValueError, match="variable"):
        NarrativeRestriction(kind="hd_dominance", date=10, shock=0)


def test_bad_dominance_and_negative_window():
    with pytest.raises(ValueError, match="dominance"):
        NarrativeRestriction(kind="hd_dominance", date=10, shock=0,
                             variable=0, dominance="huge")
    with pytest.raises(ValueError, match="window"):
        NarrativeRestriction(kind="hd_dominance", date=10, shock=0,
                             variable=0, window=-1)


def test_unknown_restriction_object_raises_typeerror():
    with pytest.raises(TypeError, match="narrative_sign_svar"):
        _run(["definitely not a restriction"])


def test_date_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        _run([(T + 5, 0, +1)])
    with pytest.raises(ValueError, match="out of range"):
        _run([(0, 0, +1)])  # inside the first p rows


def test_non_integer_date_without_dates_raises():
    with pytest.raises(ValueError, match="integer row index"):
        _run([("1979-10-06", 0, +1)])


def test_date_not_found_in_dates_raises():
    with pytest.raises(ValueError, match="not found"):
        narrative_sign_svar(Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
                            restrictions=[("1900-01-01", 0, +1)],
                            dates=_dates(), n_draws=100, seed=0)


def test_dates_length_mismatch_raises():
    with pytest.raises(ValueError, match="len\\(dates\\)"):
        narrative_sign_svar(Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
                            restrictions=[(TSTAR, 0, +1)],
                            dates=_dates()[:-3], n_draws=100, seed=0)


def test_shock_and_variable_index_validation():
    with pytest.raises(ValueError, match="shock index"):
        _run([(TSTAR, 5, +1)])
    with pytest.raises(ValueError, match="variable index"):
        _run([NarrativeRestriction(kind="hd_dominance", date=TSTAR,
                                   shock=0, variable=9)])


def test_hd_window_beyond_sample_raises():
    with pytest.raises(ValueError, match="window"):
        _run([NarrativeRestriction(kind="hd_dominance", date=T - 2,
                                   shock=0, variable=0, window=10)])


def test_sign_matrix_validation():
    with pytest.raises(ValueError, match="sign_matrix"):
        narrative_sign_svar(Y_SIM, p=1, horizon=8, sign_matrix=None,
                            restrictions=[], n_draws=100, seed=0)
    with pytest.raises(ValueError, match="entries"):
        narrative_sign_svar(Y_SIM, p=1, horizon=8,
                            sign_matrix={0: np.full((N, N), 2.0)},
                            restrictions=[], n_draws=100, seed=0)
    with pytest.raises(ValueError, match="shape"):
        narrative_sign_svar(Y_SIM, p=1, horizon=8,
                            sign_matrix={0: np.ones((N, N + 1))},
                            restrictions=[], n_draws=100, seed=0)
    with pytest.raises(ValueError, match="horizon"):
        narrative_sign_svar(Y_SIM, p=1, horizon=8,
                            sign_matrix={99: SM},
                            restrictions=[], n_draws=100, seed=0)


# ---------------------------------------------------------------------------
# hd_dominance semantics
# ---------------------------------------------------------------------------

def test_hd_dominance_most_accepts_at_planted_date():
    res = _run([NarrativeRestriction(kind="hd_dominance", date=TSTAR,
                                     shock=0, variable=0,
                                     dominance="most")])
    assert res.n_narrative_accepted > 0
    # 'overwhelming' is strictly stronger than 'most': acceptance can
    # only shrink (weakly) on the identical Haar stream.
    over = _run([NarrativeRestriction(kind="hd_dominance", date=TSTAR,
                                      shock=0, variable=0,
                                      dominance="overwhelming")])
    assert over.n_narrative_accepted <= res.n_narrative_accepted


def test_hd_dominance_window_runs():
    res = _run([NarrativeRestriction(kind="hd_dominance", date=TSTAR,
                                     shock=0, variable=0, window=2,
                                     dominance="most")])
    assert res.n_narrative_accepted > 0
    assert res.irf_median.shape == (9, N, N)
