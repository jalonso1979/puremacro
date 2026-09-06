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
    NarrativeSignResult,
    NarrativeSignSVARResult,
    identify_narrative_sign,
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
    # NOTE (audit M1/M106): this test used to assert that sign_matrix=None
    # raises 'sign_matrix is required' — enshrining a bug, since None is the
    # documented default and omitting the argument was already allowed. See
    # test_sign_matrix_none_behaves_like_omitted below for the fixed contract.
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


# ---------------------------------------------------------------------------
# Public API Standardization & Result Capabilities (v2.3.0)
# ---------------------------------------------------------------------------

def test_identify_narrative_sign_api_and_class_name():
    """Verify primary identify_narrative_sign function and NarrativeSignResult class."""
    res = identify_narrative_sign(
        Y_SIM, CORRECT_RESTRICTIONS, p=1, horizons=8, sign_matrix=SIGN_MATRIX,
        n_draws=1000, n_weight_sims=200, seed=0,
    )
    assert isinstance(res, NarrativeSignResult)
    assert isinstance(res, NarrativeSignSVARResult)
    assert NarrativeSignSVARResult is NarrativeSignResult
    assert identify_narrative_sign is narrative_sign_svar


def test_acceptance_diagnostics_properties():
    """Verify acceptance diagnostics properties on NarrativeSignResult."""
    res = _run(CORRECT_RESTRICTIONS, n_draws=1000)
    assert hasattr(res, "acceptance_rate")
    assert hasattr(res, "traditional_acceptance_rate")
    assert hasattr(res, "narrative_acceptance_rate")
    assert hasattr(res, "effective_draws")

    assert res.acceptance_rate == float(res.n_narrative_accepted / res.n_draws)
    assert res.traditional_acceptance_rate == float(res.n_traditional_accepted / res.n_draws)
    assert res.narrative_acceptance_rate == float(res.n_narrative_accepted / res.n_traditional_accepted)
    assert res.effective_draws == float(res.ess)
    assert 0.0 < res.acceptance_rate <= 1.0


def test_svar_irf_fevd_and_historical_decomposition():
    """Verify .irf(), .fevd(), and .historical_decomposition() capabilities."""
    res = _run(CORRECT_RESTRICTIONS, n_draws=1000)

    # 1. .irf()
    irf_all = res.irf()
    assert irf_all.shape == (9, N, N)
    np.testing.assert_array_equal(irf_all, res.irf_median)

    irf_h4 = res.irf(horizon=4)
    assert irf_h4.shape == (5, N, N)
    np.testing.assert_array_equal(irf_h4, res.irf_median[:5])

    with pytest.raises(ValueError, match="horizon"):
        res.irf(horizon=-1)

    # 2. .fevd()
    fevd_all = res.fevd()
    assert fevd_all.shape == (9, N, N)
    for h in range(9):
        for i in range(N):
            np.testing.assert_allclose(fevd_all[h, i, :].sum(), 1.0, atol=1e-6)

    fevd_h4 = res.fevd(horizon=4)
    assert fevd_h4.shape == (5, N, N)

    # 3. .historical_decomposition()
    hd_all = res.historical_decomposition()
    assert isinstance(hd_all, dict)
    assert "shocks" in hd_all and "deterministic" in hd_all
    assert hd_all["shocks"].shape == (T - 1, N, N)
    assert hd_all["deterministic"].shape == (T - 1, N)

    # Variable decomposition into DataFrame
    hd_var0 = res.historical_decomposition(variable=0)
    assert isinstance(hd_var0, pd.DataFrame)
    assert set(hd_var0.columns) == {"shock_0", "shock_1", "shock_2", "deterministic"}
    assert len(hd_var0) == T - 1

    # Shock decomposition across variables
    hd_shock0 = res.historical_decomposition(shock=0)
    assert isinstance(hd_shock0, pd.DataFrame)
    assert len(hd_shock0.columns) == N
    assert len(hd_shock0) == T - 1

    # Specific variable and shock pair
    hd_v0_s0 = res.historical_decomposition(variable=0, shock=0)
    assert isinstance(hd_v0_s0, pd.DataFrame)
    assert list(hd_v0_s0.columns) == ["shock_0"]

    # Bounds validation
    with pytest.raises(ValueError, match="variable index"):
        res.historical_decomposition(variable=99)
    with pytest.raises(ValueError, match="shock index"):
        res.historical_decomposition(shock=99)


def test_identify_narrative_sign_from_var_estimate_result():
    """Verify identify_narrative_sign accepts VarEstimateResult directly."""
    from puremacro.var.estimate import estimate_var

    var_res = estimate_var(Y_SIM, p=1)
    res_direct = identify_narrative_sign(
        var_res, CORRECT_RESTRICTIONS, horizons=8, sign_matrix=SIGN_MATRIX,
        n_draws=1000, n_weight_sims=200, seed=0,
    )
    res_raw = identify_narrative_sign(
        Y_SIM, CORRECT_RESTRICTIONS, p=1, horizons=8, sign_matrix=SIGN_MATRIX,
        n_draws=1000, n_weight_sims=200, seed=0,
    )
    np.testing.assert_array_equal(res_direct.irf_median, res_raw.irf_median)
    assert res_direct.acceptance_rate == res_raw.acceptance_rate
    assert res_direct.n_narrative_accepted == res_raw.n_narrative_accepted


def test_result_presentation_methods():
    """Verify result presentation contract (.summary, .plot, .to_latex, .to_typst, .to_markdown, .to_frame)."""
    res = _run(CORRECT_RESTRICTIONS, n_draws=1000)

    # .to_frame()
    df = res.to_frame(target_idx=0, shock_idx=0)
    assert isinstance(df, pd.DataFrame)
    assert {"h", "point", "lower", "upper"} <= set(df.columns)

    # .to_markdown()
    md = res.to_markdown(target_idx=0, shock_idx=0)
    assert isinstance(md, str)
    assert "|  h |" in md or "| h |" in md

    # .to_latex()
    latex = res.to_latex(target_idx=0, shock_idx=0)
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex

    # .to_typst()
    typst = res.to_typst(target_idx=0, shock_idx=0)
    assert isinstance(typst, str)
    assert "#table" in typst

    # .plot()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    res_ax = res.plot(target_idx=0, shock_idx=0, ax=ax)
    assert res_ax is not None
    plt.close(fig)


def test_bayesian_draws_mode():
    """Verify bayes_draws=True runs and yields valid NarrativeSignResult with diagnostics."""
    res = narrative_sign_svar(
        Y_SIM, p=1, horizon=4, sign_matrix=SIGN_MATRIX,
        restrictions=[(TSTAR, 0, +1)], bayes_draws=True,
        n_draws=200, seed=1,
    )
    assert isinstance(res, NarrativeSignResult)
    assert res.acceptance_rate > 0.0
    assert res.effective_draws > 0.0
    assert res.irf().shape == (5, N, N)



# ---------------------------------------------------------------------------
# Regression tests for the v2.3.0 audit (r1-narrative-svar / review-r1-narrative)
# Each test documents the behaviour of the old code in its docstring.
# ---------------------------------------------------------------------------

import warnings


def _qe_dates():
    """Quarter-END stamps (1950Q1 .. ), the pandas `freq="QE"` convention."""
    return pd.date_range("1950-01-01", periods=T, freq="QE")


def test_sign_matrix_none_behaves_like_omitted():
    """Old code: sign_matrix=None raised 'sign_matrix is required' while
    omitting the argument silently imposed nothing (M1/M106). None is the
    documented default and must mean 'no traditional sign restrictions'."""
    a = narrative_sign_svar(Y_SIM, p=1, horizon=4, sign_matrix=None,
                            restrictions=[(TSTAR, 0, +1)], n_draws=100, seed=0)
    b = narrative_sign_svar(Y_SIM, p=1, horizon=4,
                            restrictions=[(TSTAR, 0, +1)], n_draws=100, seed=0)
    assert a.n_traditional_accepted == 100 == b.n_traditional_accepted
    np.testing.assert_array_equal(a.irf_median, b.irf_median)


def test_integer_date_is_row_index_even_with_dates():
    """C4: with dates= given, an integer date was re-interpreted as
    pd.Timestamp(int) (epoch nanoseconds) and the quarter fallback silently
    remapped it to 1970Q1 — a different restricted date, no error. An integer
    must always be the 0-based row index into Y, and a Timestamp / ISO string
    must map to the same shock row."""
    from puremacro.var.identify.narrative_sign import _map_date_to_eps_row

    dates = _qe_dates()
    assert _map_date_to_eps_row(TSTAR, dates, p=1, T=T) == TSTAR - 1
    assert _map_date_to_eps_row(TSTAR, None, p=1, T=T) == TSTAR - 1
    assert _map_date_to_eps_row(np.int64(TSTAR), dates, p=2, T=T) == TSTAR - 2

    res_int_dates = _run([(TSTAR, 0, +1)], n_draws=600, dates=dates)
    res_ts_dates = _run([(dates[TSTAR], 0, +1)], n_draws=600, dates=dates)
    res_iso_dates = _run([(str(dates[TSTAR].date()), 0, +1)], n_draws=600, dates=dates)
    res_int = _run([(TSTAR, 0, +1)], n_draws=600)
    for r in (res_ts_dates, res_iso_dates, res_int):
        assert r.n_narrative_accepted == res_int_dates.n_narrative_accepted
        assert r.restriction_fail_counts == res_int_dates.restriction_fail_counts
        np.testing.assert_array_equal(r.irf_median, res_int_dates.irf_median)


def test_dataframe_datetimeindex_supplies_dates_automatically():
    """Old code discarded a DataFrame's DatetimeIndex and raised 'restriction
    date must be an integer row index ... when `dates` is not supplied' for a
    timestamp restriction. The index must be used as `dates` automatically
    (PeriodIndex too); an explicit dates= still wins."""
    dates = _qe_dates()
    df = pd.DataFrame(Y_SIM, index=dates, columns=["a", "b", "c"])
    announcement = dates[TSTAR] - pd.Timedelta(days=40)  # mid-quarter date
    auto = narrative_sign_svar(df, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
                               restrictions=[(announcement, 0, +1)],
                               n_draws=600, seed=0)
    explicit = _run([(TSTAR, 0, +1)], n_draws=600)
    np.testing.assert_array_equal(auto.irf_median, explicit.irf_median)
    assert auto.names == ("a", "b", "c")

    periods = pd.period_range("1950Q1", periods=T, freq="Q")
    dfp = pd.DataFrame(Y_SIM, index=periods)
    via_period = narrative_sign_svar(
        dfp, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
        restrictions=[(periods[TSTAR].to_timestamp(), 0, +1)], n_draws=600, seed=0,
    )
    np.testing.assert_array_equal(via_period.irf_median, explicit.irf_median)


def test_period_fallback_does_not_remap_across_months():
    """Review finding 11: the year-month / quarter fallback silently mapped a
    date whose month was missing from a *monthly* index (2002-07 deleted) to
    the neighbouring month. The fallback is now the period implied by the
    index spacing: same month on a monthly index, same quarter on a quarterly
    index, exact match on anything finer."""
    from puremacro.var.identify.narrative_sign import _map_date_to_eps_row

    mdates = pd.date_range("1990-01-01", periods=T, freq="MS")
    md2 = mdates.delete(150)
    with pytest.raises(ValueError, match="not found"):
        _map_date_to_eps_row(mdates[150], md2, p=1, T=T - 1)
    # within-month day on a monthly index -> that month
    assert _map_date_to_eps_row(mdates[150] + pd.Timedelta(days=10), mdates, p=1, T=T) == 149
    # within-quarter announcement on a quarter-end index -> that quarter's stamp
    qe = _qe_dates()
    assert _map_date_to_eps_row(qe[TSTAR] - pd.Timedelta(days=40), qe, p=1, T=T) == TSTAR - 1
    # daily index: exact matches only
    ddates = pd.date_range("2000-01-01", periods=T, freq="D")
    with pytest.raises(ValueError, match="not found"):
        _map_date_to_eps_row(ddates[-1] + pd.Timedelta(days=1), ddates, p=1, T=T)
    # a float is neither a row index nor a date: reject instead of epoch-ns
    with pytest.raises(ValueError, match="integer row index"):
        _map_date_to_eps_row(59.0, qe, p=1, T=T)


def test_unknown_keyword_arguments_raise_typeerror():
    """M3/M102: a **kwargs sink swallowed typos such as n_draw=, ci_level=
    and bogus=, silently running with the defaults."""
    for bad in (dict(n_draw=5), dict(ci_level=0.5), dict(bogus=1), dict(bootstrap=True)):
        with pytest.raises(TypeError):
            _run([], n_draws=50, **bad)


def test_horizon_and_lag_aliases_conflict_loudly():
    """M3: horizon=3 with horizons=7 silently used 7. Aliases must agree;
    `lags` mirrors estimate_var's alias for p."""
    with pytest.raises(ValueError, match="conflicting horizon"):
        narrative_sign_svar(Y_SIM, p=1, horizon=3, horizons=7,
                            sign_matrix=SIGN_MATRIX, n_draws=50, seed=0)
    a = narrative_sign_svar(Y_SIM, p=1, horizons=5, sign_matrix=SIGN_MATRIX,
                            n_draws=300, seed=0)
    b = narrative_sign_svar(Y_SIM, p=1, horizon=5, sign_matrix=SIGN_MATRIX,
                            n_draws=300, seed=0)
    assert a.irf_median.shape == (6, N, N)
    np.testing.assert_array_equal(a.irf_median, b.irf_median)
    with pytest.raises(ValueError, match="conflicting p"):
        narrative_sign_svar(Y_SIM, p=1, lags=2, horizon=3,
                            sign_matrix=SIGN_MATRIX, n_draws=50, seed=0)
    c = narrative_sign_svar(Y_SIM, lags=1, horizon=5, sign_matrix=SIGN_MATRIX,
                            n_draws=300, seed=0)
    np.testing.assert_array_equal(c.irf_median, b.irf_median)


def test_ci_and_draw_counts_are_validated():
    """Audit (e): ci=1.5, 0.0, -0.2 and 90 were accepted silently (ci=-0.2
    produced lower > upper); n_weight_sims=0 raised ZeroDivisionError."""
    for ci in (1.5, 0.0, -0.2, 90, 1.0):
        with pytest.raises(ValueError, match="ci must be"):
            _run([], n_draws=50, ci=ci)
    with pytest.raises(ValueError, match="n_weight_sims"):
        narrative_sign_svar(Y_SIM, p=1, horizon=2, sign_matrix=SIGN_MATRIX,
                            restrictions=[], n_draws=50, n_weight_sims=0, seed=0)
    with pytest.raises(ValueError, match="n_draws"):
        _run([], n_draws=0)
    with pytest.raises(ValueError, match="horizon must be >= 0"):
        narrative_sign_svar(Y_SIM, p=1, horizons=-1, sign_matrix=SIGN_MATRIX,
                            restrictions=[], n_draws=50, seed=0)


def test_shock_sign_restriction_requires_explicit_sign():
    """The dataclass default sign=+1 let `NarrativeRestriction(kind='shock_sign', ...)`
    silently restrict the shock to be positive; a Type I restriction without a
    sign is now an error, and float signs are normalised to int."""
    with pytest.raises(ValueError, match="sign"):
        NarrativeRestriction(kind="shock_sign", date=10, shock=0)
    r = NarrativeRestriction(kind="shock_sign", date=10, shock=0, sign=1.0)
    assert r.sign == 1 and isinstance(r.sign, int)
    with pytest.raises(ValueError, match="sign must be"):
        NarrativeRestriction(kind="shock_sign", date=10, shock=0, sign=2)


def test_irf_and_fevd_beyond_horizon_are_weighted_medians():
    """M4/M104: irf(h > H) returned compute_irf(A_list, B_star) — the IRF of
    a single representative draw — and fevd(h > H) that draw's FEVD, so the
    first H+1 rows changed meaning (max diff 0.68 vs irf_median). Both must
    be the weighted median across the accepted draws extended to h."""
    from puremacro.var.irf import irf as compute_irf

    res = _run(CORRECT_RESTRICTIONS, n_draws=1500)  # H = 8
    ext = res.irf(20)
    assert ext.shape == (21, N, N)
    np.testing.assert_allclose(ext[:9], res.irf_median, atol=1e-12)
    single = compute_irf(list(res.A_list), res.B, 20)
    assert not np.allclose(ext, single)
    fe = res.fevd(20)
    assert fe.shape == (21, N, N)
    np.testing.assert_allclose(fe[:9], res.fevd_median, atol=1e-12)
    np.testing.assert_allclose(fe.sum(axis=2), 1.0, atol=1e-9)
    # the per-draw objects behind the extension are exposed
    assert res.accepted_B.shape == (res.n_narrative_accepted, N, N)
    assert res.accepted_A is None  # OLS mode: every draw shares A_list
    np.testing.assert_allclose(res.B @ res.B.T, res.Sigma, atol=1e-10)
    # without stored draws the extension refuses rather than returning one draw
    bare = dataclasses.replace(res, accepted_B=None)
    with pytest.raises(ValueError, match="per-draw"):
        bare.irf(20)
    with pytest.raises(ValueError, match="per-draw"):
        bare.fevd(20)


def test_historical_decomposition_identity_holds_by_default():
    """Review check: the default init_y=0 broke y_t = det_t + sum_j shocks_t[:, j]
    on data with non-zero initial conditions (max error ~7). The first p
    observations are now stored and used by default, also when Y is a
    VarEstimateResult (recovered from the design matrix)."""
    from puremacro.var.estimate import estimate_var

    Y = Y_SIM + np.array([10.0, 5.0, -3.0])
    Y[0] += np.array([3.0, -2.0, 1.0])
    res = narrative_sign_svar(Y, p=1, horizon=4, sign_matrix=SIGN_MATRIX,
                              restrictions=[(TSTAR, 0, +1)], n_draws=300, seed=0)
    hd = res.historical_decomposition()
    np.testing.assert_allclose(hd["deterministic"] + hd["shocks"].sum(axis=2),
                               Y[1:], atol=1e-9)
    np.testing.assert_array_equal(res.init_y, Y[:1])

    vr = estimate_var(Y, p=2)
    res2 = narrative_sign_svar(vr, horizon=4, sign_matrix=SIGN_MATRIX,
                               restrictions=[(TSTAR, 0, +1)], n_draws=300, seed=0)
    np.testing.assert_allclose(res2.init_y, Y[:2])
    hd2 = res2.historical_decomposition()
    np.testing.assert_allclose(hd2["deterministic"] + hd2["shocks"].sum(axis=2),
                               Y[2:], atol=1e-9)
    # explicit zeros still give the pure-intercept counterfactual
    hd0 = res.historical_decomposition(init_y=np.zeros((1, N)))
    assert not np.allclose(hd0["deterministic"], hd["deterministic"])


def test_few_survivors_warning():
    """Audit (d): n_draws=50 left a single surviving draw whose bands
    collapsed onto the median (ESS 1.0) with no warning."""
    rng = np.random.default_rng(0)
    Yr = rng.standard_normal((120, 3))
    with pytest.warns(RuntimeWarning, match="bands may collapse"):
        r = narrative_sign_svar(Yr, p=1, horizon=4,
                                sign_matrix={0: np.array([+1, -1, -1])},
                                restrictions=[(10, 0, +1)], n_draws=50, seed=0)
    assert r.n_narrative_accepted < 10
    # a healthy run emits nothing
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(CORRECT_RESTRICTIONS, n_draws=1000)


def test_omega_floor_is_counted_and_warned():
    """M5: with an almost-impossible narrative pattern every omega_hat was 0,
    every weight was floored to exactly n_weight_sims and the ESS read as
    fully efficient — with no warning and no trace in summary()."""
    R = [NarrativeRestriction(kind="hd_dominance", date=TSTAR, shock=0,
                              variable=0, window=2, dominance="overwhelming"),
         NarrativeRestriction(kind="shock_bound", date=TSTAR, shock=0,
                              min_magnitude=3.0)]
    with pytest.warns(RuntimeWarning, match="floored"):
        r = narrative_sign_svar(Y_SIM, p=1, horizon=8, sign_matrix=SIGN_MATRIX,
                                restrictions=R, n_draws=1500, n_weight_sims=25,
                                seed=11)
    assert r.n_weight_floor > 0
    assert np.all(r.weights <= 25.0 + 1e-12)
    assert "omega floor" in r.summary()


def test_weight_concentration_warning():
    """Docs §2 promised an ESS warning 'if weight concentration is severe';
    grep found no warning in the module. _warn_diagnostics fires when the
    Kish ESS is below 10% of the accepted draws and stays silent otherwise."""
    from puremacro.var.identify.narrative_sign import _warn_diagnostics

    common = dict(n_trad=400, ci=0.9, n_weight_floor=0, n_weight_sims=500,
                  n_unstable=0, n_draws=1000)
    with pytest.warns(RuntimeWarning, match="concentrated"):
        _warn_diagnostics(n_accepted=200, ess=12.0, **common)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_diagnostics(n_accepted=200, ess=150.0, **common)


def test_plot_multi_panel_when_index_is_none():
    """M2: the documented res.plot(shock_idx=0, target_idx=None) raised
    'Per-column arrays must each be 1-dimensional'; it now draws one panel
    per response variable (and an n x n grid when both are None)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = _run(CORRECT_RESTRICTIONS, n_draws=500)
    fig = res.plot(shock_idx=0, target_idx=None)
    assert len(fig.axes) == N
    plt.close(fig)
    fig2 = res.plot(shock_idx=None, target_idx=None)
    assert len(fig2.axes) == N * N
    plt.close(fig2)
    fig3 = res.plot(target_idx=1, shock_idx=None, title="row")
    assert len(fig3.axes) == N
    plt.close(fig3)
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="ax="):
        res.plot(target_idx=None, ax=ax)
    plt.close("all")


def test_summary_reports_reduced_form_mode():
    res = _run(CORRECT_RESTRICTIONS, n_draws=500)
    s = res.summary()
    assert "reduced form      : OLS point estimate" in s
    assert not res.bayes_draws and res.n_unstable_draws == 0
