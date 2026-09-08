"""statsmodels parity for the four small inference extras.

One class per module:

* ``TestMultipleTests``  -> ``puremacro.inference.multiple``
* ``TestCollinearity``   -> ``puremacro.inference.collinearity``
* ``TestProportions``    -> ``puremacro.inference.proportions``
* ``TestDiagnostics``    -> ``puremacro.inference.diagnostics``

Every parity test is guarded with ``pytest.importorskip("statsmodels")``;
the behaviour, type and failure-path tests run everywhere, which is the
point — statsmodels is a dev-only dependency and the shipped modules must
not need it.

On "making sure a test can fail" (CONTRIBUTING.md): three tests here are
explicit **controls**, named ``..._is_load_bearing``. Each one exhibits a
plausible wrong implementation and asserts it gives a *different* answer on
the fixture actually used, so the parity sweep above it cannot be green by
accident:

* ``test_step_down_accumulate_is_load_bearing`` — drop the running maximum
  from Holm.
* ``test_step_up_reversal_is_load_bearing`` — reverse the BH accumulation
  only once instead of twice.
* ``test_implicit_constant_detection_is_load_bearing`` — treat "has a
  constant" as "has a column of ones".
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.inference.collinearity import vif
from puremacro.inference.diagnostics import durbin_watson
from puremacro.inference.multiple import fdrcorrection, multipletests
from puremacro.inference.proportions import proportion_confint


# ---------------------------------------------------------------------------
# Fixtures shared by the multiple-testing sweep
# ---------------------------------------------------------------------------

# Every method name and alias implemented, including the mixed case that
# statsmodels accepts because it lowercases before dispatching.
ALL_METHODS = [
    "b", "bonf", "bonferroni",
    "s", "sidak",
    "h", "holm",
    "hs", "holm-sidak",
    "sh", "simes-hochberg",
    "ho", "hommel",
    "fdr_bh", "fdr_i", "fdr_p", "fdri", "fdrp",
    "fdr_by", "fdr_n", "fdr_c", "fdrn", "fdrcorr",
    "fdr_gbs",
    "HOLM", "Fdr_BH",
]

# Adversarial p-value vectors. The names are used in test ids.
_RNG = np.random.default_rng(20260906)
PVAL_CASES = {
    "all_ones": np.ones(6),
    "all_zeros": np.zeros(6),
    "ties": np.array([0.02, 0.02, 0.02, 0.5, 0.5, 0.9]),
    "all_tied": np.full(7, 0.037),
    "length_one_small": np.array([0.003]),
    "length_one_large": np.array([0.87]),
    "length_two": np.array([0.6, 0.01]),
    "unsorted": _RNG.random(11),
    "already_sorted": np.sort(_RNG.random(9)),
    "above_alpha_only": np.array([0.2, 0.4, 0.6, 0.8, 0.99]),
    "below_alpha_only": np.array([1e-6, 2e-5, 3e-4, 1e-3]),
    "subnormal_and_one": np.array([1e-300, 1e-12, 0.5, 1.0]),
    "just_under_one": np.array([1.0, 1.0, 0.999999999]),
    "cubed_uniform_200": _RNG.random(200) ** 3,
    "descending": np.linspace(0.9, 0.01, 8),
}


def _sm_multitest():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests as sm_multipletests

    return sm_multipletests


def _assert_same_tuple(got, want):
    """Assert two ``multipletests`` 4-tuples agree, element by element.

    Bit-exact, not ``allclose``. The whole sweep was measured at max abs
    diff 0.0, so anything looser would stop discriminating between
    algebraically identical but numerically different formulas — Sidak's
    ``-expm1(n·log1p(-p))`` against ``1 - (1-p)**n`` differs only at
    ``p ~ 1e-300``, which any absolute tolerance swallows whole.
    """
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(np.asarray(got[1], dtype=float),
                                  np.asarray(want[1], dtype=float))
    assert got[2] == want[2]
    assert got[3] == want[3]


class TestMultipleTests:
    """``puremacro.inference.multiple`` vs ``statsmodels.stats.multitest``."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    @pytest.mark.parametrize("case", list(PVAL_CASES))
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.5])
    def test_parity_full_tuple(self, method, case, alpha):
        sm_multipletests = _sm_multitest()
        pvals = PVAL_CASES[case]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = sm_multipletests(pvals, alpha=alpha, method=method)
            got = multipletests(pvals, alpha=alpha, method=method)
        _assert_same_tuple(got, want)

    @pytest.mark.parametrize("method", ["holm", "hs", "fdr_bh", "fdr_by",
                                        "bonferroni", "sidak",
                                        "simes-hochberg", "hommel"])
    def test_parity_is_sorted_true(self, method):
        """``is_sorted=True`` keeps results in ascending-p order."""
        sm_multipletests = _sm_multitest()
        pvals = np.sort(PVAL_CASES["unsorted"])
        want = sm_multipletests(pvals, alpha=0.05, method=method,
                                is_sorted=True)
        got = multipletests(pvals, alpha=0.05, method=method, is_sorted=True)
        _assert_same_tuple(got, want)

    @pytest.mark.parametrize("method", ["holm", "hs", "sh", "ho",
                                        "fdr_bh", "fdr_by"])
    def test_is_sorted_flag_is_load_bearing(self, method):
        """Control: ``is_sorted=True`` on unsorted input must change things.

        Every order-dependent method reads the vector positionally once the
        flag is set, so lying about the order has to move the answer. If it
        did not, the sorting branch would be dead code and the parity test
        above would prove nothing about it.

        ``bonferroni`` and ``sidak`` are deliberately excluded: they are
        elementwise, so the flag genuinely cannot change their values, and
        asserting otherwise would be asserting a falsehood.
        """
        unsorted = np.array([0.5, 0.01, 0.2, 0.04, 0.9, 0.03])
        assert not np.all(np.diff(unsorted) >= 0)
        lied = multipletests(unsorted, alpha=0.05, method=method,
                             is_sorted=True)[1]
        honest = multipletests(unsorted, alpha=0.05, method=method,
                               is_sorted=False)[1]
        assert not np.allclose(lied, honest)

    @pytest.mark.parametrize("method", ["holm", "hs", "fdr_bh", "fdr_by"])
    def test_parity_returnsorted(self, method):
        sm_multipletests = _sm_multitest()
        pvals = PVAL_CASES["unsorted"]
        want = sm_multipletests(pvals, alpha=0.05, method=method,
                                returnsorted=True)
        got = multipletests(pvals, alpha=0.05, method=method,
                            returnsorted=True)
        _assert_same_tuple(got, want)
        # returnsorted really does reorder: the corrected vector comes back
        # non-decreasing, which the original order is not.
        assert np.all(np.diff(got[1]) >= 0)

    @pytest.mark.parametrize("case", list(PVAL_CASES))
    def test_corpus_methods_are_bit_exact(self, case):
        """The two methods the corpus actually calls must match to the bit.

        ``holm`` at N15:267, N15:969, N15:1033 and N21:449; ``fdr_bh``
        alongside it. ``assert_array_equal`` on floats is deliberate: these
        were measured at max abs diff 0.0, and a regression to "close enough"
        is exactly what this test exists to catch.
        """
        sm_multipletests = _sm_multitest()
        pvals = PVAL_CASES[case]
        for method in ("holm", "fdr_bh"):
            want = sm_multipletests(pvals, alpha=0.05, method=method)
            got = multipletests(pvals, alpha=0.05, method=method)
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_array_equal(got[1], want[1])

    def test_corpus_call_site_shape(self):
        """N21:449 passes ``Series.values`` and keeps index ``[1]`` only."""
        res = pd.DataFrame({"p_adj": [0.001, 0.04, 0.2, 0.6, 0.9]})
        res["p_holm"] = multipletests(res["p_adj"].values, method="holm")[1]
        assert res["p_holm"].notna().all()
        assert np.all(np.diff(np.sort(res["p_holm"].values)) >= 0)
        # N15:969 passes the Series itself, not .values.
        out = multipletests(res["p_adj"], alpha=0.05, method="holm")[1]
        np.testing.assert_array_equal(out, res["p_holm"].values)

    @pytest.mark.parametrize("smethod,pmethod", [
        ("indep", "indep"), ("i", "i"), ("p", "p"), ("poscorr", "poscorr"),
        ("n", "n"), ("negcorr", "negcorr"),
    ])
    @pytest.mark.parametrize("is_sorted", [False, True])
    def test_fdrcorrection_parity(self, smethod, pmethod, is_sorted):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.multitest import fdrcorrection as sm_fdr

        pvals = np.sort(PVAL_CASES["unsorted"]) if is_sorted \
            else PVAL_CASES["unsorted"]
        want = sm_fdr(pvals, alpha=0.05, method=smethod, is_sorted=is_sorted)
        got = fdrcorrection(pvals, alpha=0.05, method=pmethod,
                            is_sorted=is_sorted)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])

    def test_aliases_agree_with_canonical_names(self):
        pvals = PVAL_CASES["unsorted"]
        groups = [
            ("bonferroni", ["b", "bonf"]),
            ("sidak", ["s"]),
            ("holm", ["h", "HOLM"]),
            ("holm-sidak", ["hs"]),
            ("simes-hochberg", ["sh"]),
            ("hommel", ["ho"]),
            ("fdr_bh", ["fdr_i", "fdr_p", "fdri", "fdrp", "Fdr_BH"]),
            ("fdr_by", ["fdr_n", "fdr_c", "fdrn", "fdrcorr"]),
        ]
        for canonical, aliases in groups:
            ref = multipletests(pvals, method=canonical)
            for alias in aliases:
                _assert_same_tuple(multipletests(pvals, method=alias), ref)

    # -- monotonicity controls ---------------------------------------------

    def test_step_down_accumulate_is_load_bearing(self):
        """Holm without the running maximum gives a different answer here.

        The fixture is chosen so the raw step-down products are *not*
        monotone (0.2*4 = 0.8 exceeds 0.3*3 = 0.9? no — 0.25*4 = 1.0 exceeds
        0.26*3 = 0.78), which is the only situation in which
        ``maximum.accumulate`` changes anything. If this control ever starts
        passing trivially, the sweep above has stopped testing the step.
        """
        pvals = np.array([0.01, 0.25, 0.26, 0.27])
        n = pvals.size
        order = np.argsort(pvals)
        naive = np.empty(n)
        naive[order] = np.minimum(pvals[order] * np.arange(n, 0, -1), 1.0)
        real = multipletests(pvals, method="holm")[1]
        assert not np.allclose(naive, real), (
            "fixture no longer exercises the running maximum"
        )
        # And the real one is the monotone envelope of the naive one.
        assert np.all(np.diff(real[order]) >= 0)
        assert np.all(real >= naive - 1e-15)

    def test_step_up_reversal_is_load_bearing(self):
        """BH with only one of the two reversals is systematically wrong."""
        pvals = np.array([0.005, 0.02, 0.03, 0.5, 0.9])
        n = pvals.size
        order = np.argsort(pvals)
        raw = pvals[order] / (np.arange(1, n + 1) / n)
        # The classic bug: accumulate on the reversed array and forget to
        # reverse the result back.
        buggy = np.empty(n)
        buggy[order] = np.minimum(np.minimum.accumulate(raw[::-1]), 1.0)
        real = multipletests(pvals, method="fdr_bh")[1]
        assert not np.allclose(buggy, real), (
            "fixture no longer distinguishes the double reversal"
        )
        assert np.all(np.diff(real[order]) >= 0)

    def test_corrected_pvalues_are_monotone_in_sorted_order(self):
        """Every implemented method returns a non-decreasing sorted vector."""
        pvals = PVAL_CASES["cubed_uniform_200"]
        order = np.argsort(pvals)
        for method in ("bonferroni", "sidak", "holm", "holm-sidak",
                       "simes-hochberg", "hommel", "fdr_bh", "fdr_by"):
            adj = multipletests(pvals, method=method)[1]
            assert np.all(np.diff(adj[order]) >= -1e-15), method
            assert np.all(adj <= 1.0 + 1e-15), method

    # -- failure paths ------------------------------------------------------

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="method not recognized"):
            multipletests(np.array([0.1, 0.2]), method="holms")

    @pytest.mark.parametrize("method", ["fdr_tsbh", "fdr_2sbh", "fdr_tsbky",
                                        "fdr_twostage", "fdr_2sbky"])
    def test_two_stage_methods_raise_not_implemented(self, method):
        """Known to statsmodels, not implemented here — say so precisely.

        The list is exactly ``fdrcorrection_twostage``'s two methods and
        their aliases. ``fdr_gbs`` used to be in it and is not two-stage:
        it is a single closed-form pass, it is implemented, and
        ``test_fdr_gbs_is_implemented_and_matches_statsmodels`` pins it.
        """
        with pytest.raises(NotImplementedError, match="does not implement"):
            multipletests(np.array([0.1, 0.2]), method=method)

    @pytest.mark.parametrize("case", list(PVAL_CASES))
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.5])
    def test_fdr_gbs_is_implemented_and_matches_statsmodels(self, case, alpha):
        """``fdr_gbs`` must return statsmodels' numbers, not an exception.

        Regression test. Before the fix this method raised
        ``NotImplementedError`` describing itself as part of "the two-stage
        FDR family", which it is not: statsmodels implements it in eight
        lines at ``multitest.py:249-262`` and it never iterates.
        """
        sm_multipletests = _sm_multitest()
        pvals = PVAL_CASES[case]
        with warnings.catch_warnings():
            # p == 1 divides by zero in the odds ratio, in both libraries.
            warnings.simplefilter("ignore")
            want = sm_multipletests(pvals, alpha=alpha, method="fdr_gbs")
            got = multipletests(pvals, alpha=alpha, method="fdr_gbs")
        _assert_same_tuple(got, want)

    # -- signature ----------------------------------------------------------

    def test_signature_matches_statsmodels(self):
        """Every parameter, in statsmodels' order, with the same defaults.

        Regression test for the missing ``maxiter``. This is a stronger
        assertion than "the keywords we use work": a drop-in replacement is
        only positionally safe if the *order* matches, and statsmodels 0.14
        inserted ``maxiter`` in the fourth slot, ahead of ``is_sorted``.
        """
        import inspect

        sm_multipletests = _sm_multitest()
        assert (inspect.signature(multipletests)
                == inspect.signature(sm_multipletests))

    @pytest.mark.parametrize("method", ["holm", "hs", "fdr_bh", "fdr_by",
                                        "bonferroni", "simes-hochberg"])
    def test_fourth_positional_argument_is_maxiter(self, method):
        """``multipletests(p, 0.05, 'holm', 1)`` must not mean is_sorted=1.

        Regression test. With ``maxiter`` missing from the signature, the
        fourth positional argument landed on ``is_sorted``, so the function
        believed an unsorted vector was ascending: for the fixture below,
        Holm returned ``[1, 1, 1, 1]`` instead of ``[1, 0.04, 1, 0.6]`` and
        the one significant hypothesis stopped being significant. No
        exception, no warning — which is why this is asserted against
        statsmodels rather than against a hand-written expectation.
        """
        sm_multipletests = _sm_multitest()
        pvals = np.array([0.5, 0.01, 0.9, 0.2])
        assert not np.all(np.diff(pvals) >= 0), "fixture must be unsorted"
        want = sm_multipletests(pvals, 0.05, method, 1)
        got = multipletests(pvals, 0.05, method, 1)
        _assert_same_tuple(got, want)
        # ...and the fourth argument really is inert for these methods, so
        # the test above cannot pass by accidentally honouring it.
        _assert_same_tuple(multipletests(pvals, 0.05, method, 99), want)

    def test_maxiter_is_accepted_as_a_keyword(self):
        """statsmodels accepts ``maxiter=`` for any method; so must this."""
        pvals = np.array([0.5, 0.01, 0.9, 0.2])
        got = multipletests(pvals, method="holm", maxiter=1)
        want = multipletests(pvals, method="holm")
        _assert_same_tuple(got, want)
        # For the unimplemented two-stage family the keyword must reach a
        # NotImplementedError, not a TypeError about an unexpected keyword.
        with pytest.raises(NotImplementedError):
            multipletests(pvals, method="fdr_tsbh", maxiter=1)

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            multipletests(np.array([[0.1, 0.2], [0.3, 0.4]]), method="holm")
        with pytest.raises(ValueError, match="must be 1-D"):
            fdrcorrection(np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            multipletests(np.array([]), method="holm")
        with pytest.raises(ValueError, match="empty"):
            fdrcorrection(np.array([]))

    def test_fdrcorrection_unknown_method_raises(self):
        with pytest.raises(ValueError, match="indep and negcorr"):
            fdrcorrection(np.array([0.1, 0.2]), method="bh")


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

def _collinear_designs():
    """Designs that exercise every branch of the constant detection."""
    rng = np.random.default_rng(1234)
    n = 180
    x1 = rng.normal(size=n)
    x2 = x1 + 0.1 * rng.normal(size=n)      # highly but not perfectly collinear
    x3 = rng.normal(size=n)
    ones = np.ones(n)

    d1 = np.zeros(n); d1[:60] = 1.0
    d2 = np.zeros(n); d2[60:120] = 1.0
    d3 = np.zeros(n); d3[120:] = 1.0        # d1+d2+d3 == 1: implicit constant

    return {
        # explicit constant -> centered R^2 in every auxiliary regression
        "with_constant": np.column_stack([ones, x1, x2, x3]),
        # constant stripped -> uncentered R^2 (the documented trap)
        "constant_stripped": np.column_stack([x1, x2, x3]),
        # same, with means far from zero, where the trap bites hardest
        "no_constant_shifted": np.column_stack([x1 + 10, x2 + 10, x3 + 5]),
        # a constant column that is not a column of ones
        "constant_of_fives": np.column_stack([np.full(n, 5.0), x1, x3]),
        # implicit constant: saturated dummies span the intercept
        "implicit_constant": np.column_stack([d1, d2, d3, x3]),
        # wildly different units; the gate must not mistake scale for rank
        "badly_scaled": np.column_stack([ones, 1e-8 * x1, 1e6 * x3]),
        # high but finite collinearity: VIF ~ 1e8
        "very_high_vif": np.column_stack(
            [ones, x1, x1 + 1e-4 * rng.normal(size=n), x3]
        ),
        # two columns only
        "two_columns": np.column_stack([ones, x1]),
    }


COLLINEAR_DESIGNS = _collinear_designs()


class TestCollinearity:
    """``puremacro.inference.collinearity.vif`` vs statsmodels."""

    @pytest.mark.parametrize("name", list(COLLINEAR_DESIGNS))
    def test_parity_all_columns(self, name):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        X = COLLINEAR_DESIGNS[name]
        want = np.array([sm_vif(X, i) for i in range(X.shape[1])])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            got = np.asarray(vif(X), dtype=float)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-10)

    @pytest.mark.parametrize("name", list(COLLINEAR_DESIGNS))
    def test_parity_single_index(self, name):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        X = COLLINEAR_DESIGNS[name]
        for i in range(X.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                got = vif(X, i)
            assert isinstance(got, float)
            assert got == pytest.approx(sm_vif(X, i), rel=0.0, abs=1e-10)

    def test_corpus_call_site_strips_the_constant(self):
        """N15:562 does ``variance_inflation_factor(X_full[:, 1:], i)``.

        The constant has been sliced off, so every auxiliary regression is
        constant-free and the R-squared is uncentered. Reproduce the exact
        call shape, including the dict comprehension over names.
        """
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        racers = ["a", "b", "c"]
        X_full = COLLINEAR_DESIGNS["with_constant"]
        with pytest.warns(UserWarning, match="no constant column"):
            got = {c: float(vif(X_full[:, 1:], i))
                   for i, c in enumerate(racers)}
        want = {c: float(sm_vif(X_full[:, 1:], i))
                for i, c in enumerate(racers)}
        for c in racers:
            assert got[c] == pytest.approx(want[c], rel=0.0, abs=1e-10)

    def test_constant_in_or_out_changes_the_answer(self):
        """The documented trap has to be a real difference, not a nuance."""
        X = COLLINEAR_DESIGNS["no_constant_shifted"]
        with pytest.warns(UserWarning):
            without = np.asarray(vif(X), dtype=float)
        with_const = np.asarray(
            vif(np.column_stack([np.ones(X.shape[0]), X])), dtype=float
        )[1:]
        # Uncentered R^2 on mean-shifted data inflates VIF by orders of
        # magnitude. If this ever stopped being true the warning would be
        # pointless and the centered/uncentered switch untested.
        assert np.all(without > 10 * with_const)

    def test_implicit_constant_detection_is_load_bearing(self):
        """A saturated dummy set has no column of ones but is centered.

        Control: the naive "is there a column of ones" test says no, so a
        naive implementation uses the uncentered total sum of squares and
        gets a *different* number. Ours matches statsmodels' rank-based
        detection instead.
        """
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        X = COLLINEAR_DESIGNS["implicit_constant"]
        assert not np.any(np.all(X == 1.0, axis=0)), "fixture has a ones column"

        # What a naive uncentered implementation would produce for column 3.
        idx = 3
        x_i = X[:, idx]
        x_noti = np.delete(X, idx, axis=1)
        beta = np.linalg.pinv(x_noti) @ x_i
        resid = x_i - x_noti @ beta
        naive_uncentered = 1.0 / (1.0 - (1 - (resid @ resid) / (x_i @ x_i)))

        got = vif(X, idx)
        want = sm_vif(X, idx)
        assert got == pytest.approx(want, rel=0.0, abs=1e-10)
        assert not np.isclose(naive_uncentered, want), (
            "fixture no longer distinguishes centered from uncentered"
        )

    # -- magnitude, conditioning and the bit-level contract -----------------

    def test_large_magnitude_column_returns_statsmodels_numbers(self):
        """A nominal level series in yen scale must not kill the call.

        Regression test. ``2e15``-scale GDP makes numpy's rank tolerance
        (``max(M, N) * eps * s_max``) swallow the intercept direction, so
        the auxiliary design for the constant column is judged to have an
        *implicit* constant, its centered total sum of squares is exactly
        ``0.0``, and ``ssr / tss`` is a division by zero. In numpy that is
        ``inf`` and the VIF comes back as statsmodels' ``0.0``; in Python
        floats it was an uncaught ``ZeroDivisionError`` that took the two
        VIFs a user actually reads down with it.
        """
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        from puremacro.inference.collinearity import _k_constant

        yrs = np.arange(1990, 2024, dtype=float)
        gdp = 2.0e15 * np.exp(0.08 * (yrs - 1990))
        rate = 5.0 + np.sin(yrs)
        X = np.column_stack([np.ones(yrs.size), gdp, rate])

        # The fixture only exercises the defect if the auxiliary design for
        # column 0 really does trip the implicit-constant branch. Assert it,
        # so a future numpy tolerance change turns this into a red test
        # rather than a silently vacuous one.
        assert _k_constant(X[:, 1:]) == 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            want = np.array([sm_vif(X, i) for i in range(X.shape[1])])
        with pytest.warns(UserWarning, match="are constant"):
            got = np.asarray(vif(X), dtype=float)
        np.testing.assert_array_equal(got, want)
        assert got[0] == 0.0                      # the degenerate column
        assert got[1] == pytest.approx(1.0049732504759898, abs=1e-12)

    def test_bit_identity_on_near_collinear_designs(self):
        """The module docstring claims bit-identity. Assert it, bit for bit.

        ``assert_array_equal``, not ``assert_allclose(atol=1e-10)``: the
        defect this guards against — using ``np.dot(centered, centered)``
        for the centered total sum of squares where statsmodels evaluates
        ``np.sum(weights * (endog - np.average(endog, weights=weights))**2)``
        — moved the answer by 6.7e-16 absolute at worst on this very sweep,
        which is a million times below the tolerance the sweeps above use.
        It differed on 139 of the 633 column-VIFs compared here; the
        statsmodels expression differs on none of them.
        """
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        checked = 0
        largest = 0.0
        for seed in range(120):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(30, 300))
            k = int(rng.integers(3, 8))
            X = rng.standard_normal((n, k - 1))
            # Make one column nearly a copy of another, at a random distance
            # from exact collinearity, so the VIFs reach up to ~1e12.
            j = int(rng.integers(0, k - 1))
            X[:, j] = (X[:, (j + 1) % (k - 1)]
                       + 10.0 ** rng.uniform(-6, -1) * rng.standard_normal(n))
            X = np.column_stack([np.ones(n), X])
            X = X * 10.0 ** rng.uniform(-3, 3, size=k)   # wildly mixed units
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    got = np.asarray(vif(X), dtype=float)
            except np.linalg.LinAlgError:
                continue          # past the documented gate; not this test's job
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                want = np.array([sm_vif(X, i) for i in range(k)])
            np.testing.assert_array_equal(got, want)
            checked += k
            largest = max(largest, float(got.max()))

        # Positive control on the fixture. Every column but the constant runs
        # the centered branch, so a healthy count means the branch under test
        # was actually exercised; and the sweep has to reach VIFs large
        # enough for a one-ULP perturbation of tss to be amplified into the
        # answer, which is the whole reason the assertion is exact. Measured:
        # 633 columns across 120 designs, largest VIF 9.0e11.
        assert checked > 400, checked
        assert largest > 1e8, largest

    def test_columns_sharing_a_large_mean_are_not_rejected(self):
        """Levels data with an intercept: healthy VIFs, no LinAlgError.

        Regression test for the singularity gate. With a constant in the
        design, VIF_j is exactly invariant to adding a constant to any
        column — the intercept absorbs the shift — so a gate that rejects
        on the *raw* condition number rejects for a location problem that
        changes no VIF. This design is ill-conditioned in exactly that way
        and every real regressor's VIF is about 1.0.
        """
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        rng = np.random.default_rng(5)
        n = 100
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 3)) + 1e7])

        # Fixture control: the *uncentered* normalised design really is past
        # the gate's threshold, so this test would have been red before the
        # centering went in and is not vacuous now.
        raw = X / np.sqrt((X ** 2).sum(axis=0))
        assert np.linalg.cond(raw.T @ raw) > 1e14

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            want = np.array([sm_vif(X, i) for i in range(X.shape[1])])
        with pytest.warns(UserWarning, match="are constant"):
            got = np.asarray(vif(X), dtype=float)
        np.testing.assert_array_equal(got, want)
        assert np.all(got[1:] < 1.05) and np.all(got[1:] > 1.0)

    def test_constant_free_shifted_design_still_raises_with_the_right_cure(self):
        """The same shift *without* an intercept is a declared divergence.

        With no constant column the R-squared is uncentered, so the VIFs are
        genuinely location-dependent and the gate must not center. This
        design's statsmodels VIFs are ~6e12 at best and below 1 (which is
        arithmetically impossible) at larger shifts, i.e. noise. puremacro
        raises — and the message must name the cure that actually works
        (add a constant), not the one the shared ``inv_xtx`` text suggests
        (rescale), which the gate has already done.
        """
        X = np.random.default_rng(1).standard_normal((100, 3)) + 3e6
        with pytest.raises(np.linalg.LinAlgError) as excinfo:
            vif(X)
        message = str(excinfo.value)
        assert "add_constant" in message
        assert "rescal" not in message.lower()

    # -- return types -------------------------------------------------------

    def test_dataframe_returns_named_series(self):
        X = COLLINEAR_DESIGNS["with_constant"]
        df = pd.DataFrame(X, columns=["const", "x1", "x2", "x3"])
        out = vif(df)
        assert isinstance(out, pd.Series)
        assert list(out.index) == ["const", "x1", "x2", "x3"]
        np.testing.assert_allclose(out.values, np.asarray(vif(X), dtype=float))
        subset = vif(df, [1, 3])
        assert isinstance(subset, pd.Series)
        assert list(subset.index) == ["x1", "x3"]

    def test_ndarray_returns_ndarray_and_int_returns_float(self):
        X = COLLINEAR_DESIGNS["with_constant"]
        assert isinstance(vif(X), np.ndarray)
        assert isinstance(vif(X, [0, 2]), np.ndarray)
        assert isinstance(vif(X, 2), float)
        assert isinstance(vif(pd.DataFrame(X), 2), float)

    def test_negative_index_wraps_and_statsmodels_disagrees(self):
        """Negative indices wrap; statsmodels returns ``inf``. Measure both.

        This is a declared divergence, not an accident, and the assertion
        below says so out loud rather than pinning puremacro's convention
        on its own. statsmodels drops the target column with
        ``mask = np.arange(k) != exog_idx``, which masks nothing for a
        negative index: the column stays in its own auxiliary design and the
        auxiliary R-squared is 1 by construction, so the answer is ``inf``
        for every negative index (or ``0.0``, when the target column is the
        constant and the degenerate centered-tss branch fires first). Either
        way it is not that column's VIF. Reproducing it would mean
        reproducing a bug, so ``vif`` wraps Python-style instead — and the
        second half of this test is what makes the divergence visible if
        statsmodels ever fixes it.
        """
        pytest.importorskip("statsmodels")
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as sm_vif,
        )

        X = COLLINEAR_DESIGNS["with_constant"]
        k = X.shape[1]
        for j in range(1, k + 1):
            assert vif(X, -j) == vif(X, k - j)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sm_value = sm_vif(X, -j)
            assert sm_value != vif(X, -j)
            # Column ``k - j == 0`` is the constant, whose degenerate branch
            # returns 0.0 before the never-masked column can make it inf.
            assert np.isinf(sm_value) if k - j else sm_value == 0.0

    def test_boolean_mask_index_raises(self):
        """A boolean mask must be refused, not read as ones and zeros.

        Regression test. ``int(True) == 1`` and ``int(False) == 0``, so the
        mask below used to return the VIFs of columns 1, 0, 1, 0 — right
        shape, right dtype, wrong columns, no error.
        """
        X = COLLINEAR_DESIGNS["with_constant"]
        mask = np.array([True, False, True, False])
        with pytest.raises(ValueError, match="boolean mask"):
            vif(X, mask)
        with pytest.raises(ValueError, match="boolean mask"):
            vif(X, [True, False, True, False])
        with pytest.raises(ValueError, match="boolean mask"):
            vif(X, True)
        # The documented alternative works and selects the right columns.
        np.testing.assert_array_equal(vif(X, np.flatnonzero(mask)),
                                      vif(X, [0, 2]))

    # -- failure paths ------------------------------------------------------

    def test_exact_collinearity_raises(self):
        n = 60
        rng = np.random.default_rng(0)
        x1 = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x1, 2.0 * x1, rng.normal(size=n)])
        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            vif(X)

    def test_zero_column_raises(self):
        n = 40
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(n), np.zeros(n), rng.normal(size=n)])
        with pytest.raises(np.linalg.LinAlgError, match="identically"):
            vif(X)

    def test_dummy_trap_raises(self):
        """add_constant on a saturated dummy set is exactly singular.

        The message must be ``inv_xtx``'s, not the conditioning one: an
        exactly rank-deficient design is passed straight through so the
        caller gets the null-space column list, which is the answer they
        came for. Matching only on "singular" would not tell the two
        branches apart — both say it.
        """
        n = 90
        d = np.zeros((n, 3))
        d[:30, 0] = 1.0
        d[30:60, 1] = 1.0
        d[60:, 2] = 1.0
        X = np.column_stack([np.ones(n), d])
        with pytest.raises(np.linalg.LinAlgError,
                           match="most aligned with the null space"):
            vif(X)

    def test_vif_one_raises_on_an_exactly_redundant_column(self):
        """The backstop inside ``_vif_one``, which ``vif``'s gate pre-empts.

        Exercised directly because it is unreachable through ``vif``:
        every exactly-collinear design is caught on the whole matrix before
        an auxiliary regression runs, and a zero column — the one input for
        which ``pinv``'s residual sum of squares is *exactly* 0.0 — is
        rejected earlier still, with its own message. Measured on this
        fixture size, near-exact collinearity does not reach the branch:
        a duplicated column leaves ssr = 2.9e-29 and a doubled one
        1.2e-28, both non-zero. Without this test the branch, and the
        Raises clause documenting it, would be an untested claim.
        """
        from puremacro.inference.collinearity import _vif_one

        n = 60
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(n), rng.normal(size=n), np.zeros(n)])
        with pytest.raises(np.linalg.LinAlgError,
                           match="exact linear combination"):
            _vif_one(X, 2)
        # ...and the near-exact case slips past the branch, which is why the
        # whole-design gate exists. It also lands on the second reason the
        # arithmetic is done in np.float64: ssr/tss underflows, R2 rounds to
        # exactly 1, and 1/(1-R2) is numpy's inf — statsmodels' answer —
        # where a Python float would have raised ZeroDivisionError.
        near = np.column_stack([np.ones(n), rng.normal(size=n)])
        near = np.column_stack([near, 2.0 * near[:, 1]])
        assert np.isinf(_vif_one(near, 2))
        sm = pytest.importorskip("statsmodels.stats.outliers_influence")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert np.isinf(sm.variance_inflation_factor(near, 2))

    def test_one_column_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            vif(np.ones((10, 1)))

    def test_one_dimensional_raises(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            vif(np.ones(10))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_raises(self, bad):
        """``vif``'s own guard, matched on its own words.

        ``_k_constant`` carries a second finiteness check with a similar
        message, so a looser regex passes with either guard deleted. This
        one matches the sentence only the outer guard has.
        """
        X = COLLINEAR_DESIGNS["with_constant"].copy()
        X[3, 2] = bad
        with pytest.raises(ValueError, match="Drop or impute"):
            vif(X)

    def test_out_of_range_index_raises(self):
        X = COLLINEAR_DESIGNS["with_constant"]
        with pytest.raises(IndexError, match="out of range"):
            vif(X, 99)

    def test_warns_only_when_the_constant_is_missing(self):
        with pytest.warns(UserWarning, match="no constant column"):
            vif(COLLINEAR_DESIGNS["constant_stripped"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vif(COLLINEAR_DESIGNS["with_constant"])
            vif(COLLINEAR_DESIGNS["constant_of_fives"])
            vif(COLLINEAR_DESIGNS["implicit_constant"])


# ---------------------------------------------------------------------------
# Binomial proportion intervals
# ---------------------------------------------------------------------------

PROPORTION_METHODS = ["normal", "agresti_coull", "beta", "wilson",
                      "jeffreys", "jeff"]


class TestProportions:
    """``puremacro.inference.proportions`` vs ``statsmodels.stats.proportion``."""

    @pytest.mark.parametrize("method", PROPORTION_METHODS)
    @pytest.mark.parametrize("nobs", [1, 2, 5, 25, 100, 1000, 10 ** 6])
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.5])
    def test_parity_scalar_including_boundaries(self, method, nobs, alpha):
        """Sweeps count = 0, 1, n/2, n-1, n — the boundaries included."""
        pytest.importorskip("statsmodels")
        from statsmodels.stats.proportion import (
            proportion_confint as sm_confint,
        )

        counts = sorted({0, 1, nobs // 2, nobs - 1, nobs})
        for count in counts:
            if not 0 <= count <= nobs:
                continue
            want = sm_confint(count, nobs, alpha=alpha, method=method)
            got = proportion_confint(count, nobs, alpha=alpha, method=method)
            assert isinstance(got[0], float) and isinstance(got[1], float)
            for g, w in zip(got, want):
                if np.isnan(w):
                    assert np.isnan(g)
                else:
                    assert g == pytest.approx(float(w), rel=0.0, abs=1e-12)

    @pytest.mark.parametrize("method", PROPORTION_METHODS)
    def test_parity_array_input(self, method):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.proportion import (
            proportion_confint as sm_confint,
        )

        count = np.array([0, 1, 7, 24, 25])
        want = sm_confint(count, 25, alpha=0.05, method=method)
        got = proportion_confint(count, 25, alpha=0.05, method=method)
        assert isinstance(got[0], np.ndarray)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(np.isnan(g), np.isnan(w))
            np.testing.assert_allclose(g, w, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("method", PROPORTION_METHODS)
    def test_parity_broadcast_nobs(self, method):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.proportion import (
            proportion_confint as sm_confint,
        )

        count = np.array([0, 3, 40, 500])
        nobs = np.array([4, 9, 40, 1000])
        want = sm_confint(count, nobs, alpha=0.05, method=method)
        got = proportion_confint(count, nobs, alpha=0.05, method=method)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("method", PROPORTION_METHODS)
    def test_parity_pandas_series_keeps_index(self, method):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.proportion import (
            proportion_confint as sm_confint,
        )

        count = pd.Series([0, 7, 25], index=["none", "some", "all"])
        want = sm_confint(count, 25, alpha=0.05, method=method)
        got = proportion_confint(count, 25, alpha=0.05, method=method)
        for g, w in zip(got, want):
            assert isinstance(g, pd.Series)
            assert list(g.index) == ["none", "some", "all"]
            np.testing.assert_allclose(g.values, np.asarray(w),
                                       rtol=0.0, atol=1e-12)

    def test_corpus_wilson_call_site(self):
        """N04:126-127 — ``wilson_ci`` returns percentages.

        ``lo, hi = proportion_confint(successes, total, alpha=0.05,
        method="wilson"); return 100 * asarray(lo), 100 * asarray(hi)``.
        """
        pytest.importorskip("statsmodels")
        from statsmodels.stats.proportion import (
            proportion_confint as sm_confint,
        )

        for successes, total in [(0, 12), (3, 12), (12, 12), (17, 43)]:
            lo, hi = proportion_confint(successes, total, alpha=0.05,
                                        method="wilson")
            wlo, whi = sm_confint(successes, total, alpha=0.05,
                                  method="wilson")
            got = (100.0 * np.asarray(lo), 100.0 * np.asarray(hi))
            want = (100.0 * np.asarray(wlo), 100.0 * np.asarray(whi))
            np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-10)

    def test_wilson_boundary_behaviour(self):
        """Wilson stays an interval where Wald collapses to a point."""
        lo0, hi0 = proportion_confint(0, 20, method="wilson")
        assert lo0 == pytest.approx(0.0, abs=1e-15)
        assert hi0 > 0.15
        lon, hin = proportion_confint(20, 20, method="wilson")
        assert hin == pytest.approx(1.0, abs=1e-15)
        assert lon < 0.85
        # Symmetry of the score interval about 1/2.
        assert lo0 == pytest.approx(1.0 - hin, abs=1e-15)
        assert hi0 == pytest.approx(1.0 - lon, abs=1e-15)
        # Wald, for contrast, is degenerate at both ends.
        assert proportion_confint(0, 20, method="normal") == (0.0, 0.0)
        assert proportion_confint(20, 20, method="normal") == (1.0, 1.0)

    def test_beta_boundary_substitution(self):
        """Clopper-Pearson is one-sided at the boundaries, not nan."""
        lo, hi = proportion_confint(0, 15, method="beta")
        assert lo == 0.0 and 0.0 < hi < 1.0
        lo, hi = proportion_confint(15, 15, method="beta")
        assert hi == 1.0 and 0.0 < lo < 1.0
        # Array path takes a different branch in statsmodels; check it too.
        alo, ahi = proportion_confint(np.array([0, 15]), 15, method="beta")
        assert alo[0] == 0.0 and ahi[1] == 1.0
        assert np.isfinite(alo).all() and np.isfinite(ahi).all()

    def test_jeffreys_is_not_clipped_at_zero(self):
        """Unlike beta, the Jeffreys posterior gives a positive lower bound."""
        lo, hi = proportion_confint(0, 15, method="jeffreys")
        assert lo > 0.0
        assert hi < 1.0

    # -- failure paths ------------------------------------------------------

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_bad_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            proportion_confint(3, 10, alpha=alpha, method="wilson")

    def test_zero_nobs_raises(self):
        with pytest.raises(ValueError, match="nobs must be strictly"):
            proportion_confint(0, 0, method="wilson")

    def test_count_out_of_range_raises(self):
        with pytest.raises(ValueError, match="0 <= count <= nobs"):
            proportion_confint(11, 10, method="wilson")
        with pytest.raises(ValueError, match="0 <= count <= nobs"):
            proportion_confint(-1, 10, method="wilson")
        with pytest.raises(ValueError, match="0 <= count <= nobs"):
            proportion_confint(np.array([1, 20]), 10, method="wilson")

    def test_unknown_method_raises(self):
        with pytest.raises(NotImplementedError, match="not available"):
            proportion_confint(3, 10, method="wilsonn")

    def test_binom_test_is_declared_not_silently_wrong(self):
        with pytest.raises(NotImplementedError, match="binom_test"):
            proportion_confint(3, 10, method="binom_test")


# ---------------------------------------------------------------------------
# Durbin-Watson
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """``puremacro.inference.diagnostics`` vs ``statsmodels.stats.stattools``."""

    @pytest.mark.parametrize("shape,axis", [
        ((50,), 0), ((200,), 0), ((3,), 0),
        ((50, 3), 0), ((50, 3), 1), ((4, 10), 1),
        ((10, 4, 2), 0), ((10, 4, 2), 1),
    ])
    def test_parity(self, shape, axis):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.stattools import durbin_watson as sm_dw

        rng = np.random.default_rng(hash(shape) % 2 ** 31 + axis)
        resids = rng.normal(size=shape)
        want = np.asarray(sm_dw(resids, axis=axis))
        got = np.asarray(durbin_watson(resids, axis=axis))
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)

    def test_parity_pandas_and_lists(self):
        pytest.importorskip("statsmodels")
        from statsmodels.stats.stattools import durbin_watson as sm_dw

        rng = np.random.default_rng(9)
        x = np.cumsum(rng.normal(size=250))
        for obj in (x, pd.Series(x), list(x)):
            assert durbin_watson(obj) == pytest.approx(
                float(sm_dw(np.asarray(obj))), rel=0.0, abs=1e-12
            )
        frame = pd.DataFrame({"a": x, "b": np.roll(x, 3)})
        np.testing.assert_allclose(
            np.asarray(durbin_watson(frame)),
            np.asarray(sm_dw(frame)),
            rtol=0.0, atol=1e-12,
        )

    def test_corpus_call_site(self):
        """N04:1808 formats ``durbin_watson(m.resid)`` with ``:.2f``."""
        sm = pytest.importorskip("statsmodels.api")
        from statsmodels.stats.stattools import durbin_watson as sm_dw

        rng = np.random.default_rng(4)
        n = 400
        X = sm.add_constant(rng.normal(size=(n, 3)))
        eps = np.empty(n)
        eps[0] = rng.normal()
        for t in range(1, n):
            eps[t] = 0.6 * eps[t - 1] + rng.normal()
        y = X @ np.array([1.0, 0.5, -0.2, 0.1]) + eps
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        assert f"{durbin_watson(m.resid):.2f}" == f"{sm_dw(m.resid):.2f}"
        assert durbin_watson(m.resid) == pytest.approx(
            float(sm_dw(m.resid)), rel=0.0, abs=1e-12
        )

    def test_known_limits(self):
        """DW ~ 2(1 - r1): 2 for white noise, ~0 and ~4 at the extremes."""
        rng = np.random.default_rng(0)
        white = rng.normal(size=20000)
        assert durbin_watson(white) == pytest.approx(2.0, abs=0.05)

        ar1 = np.empty(20000)
        ar1[0] = white[0]
        for t in range(1, ar1.size):
            ar1[t] = 0.9 * ar1[t - 1] + white[t]
        assert durbin_watson(ar1) < 0.5

        alternating = np.array([1.0, -1.0] * 10000)
        assert durbin_watson(alternating) > 3.9

    def test_returns_float_for_one_dimensional_input(self):
        """``type(...) is float``, not ``isinstance``.

        ``np.float64`` subclasses ``float``, so ``isinstance(x, float)``
        cannot tell the narrowing conversion from its absence: deleting the
        conversion left this assertion green. Only an exact type check makes
        it load-bearing.
        """
        out = durbin_watson(np.array([1.0, 2.0, 1.5]))
        assert type(out) is float
        assert type(out) is not np.float64
        arr = durbin_watson(np.ones((5, 2)) * np.array([1.0, -1.0]))
        assert isinstance(arr, np.ndarray) and arr.shape == (2,)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2 observations"):
            durbin_watson(np.array([1.0]))
        with pytest.raises(ValueError, match="at least 2 observations"):
            durbin_watson(np.ones((1, 4)), axis=0)

    @pytest.mark.parametrize("axis", [1, 3, -2])
    def test_bad_axis_raises_the_same_error_statsmodels_raises(self, axis):
        """An out-of-range axis is numpy's ``AxisError``, not ``IndexError``.

        Regression test. Reading ``resids.shape[axis]`` before differencing
        produced a bare ``IndexError('tuple index out of range')`` that named
        neither the axis nor the array; statsmodels reaches ``np.diff``
        first and raises ``AxisError('axis 3 is out of bounds for array of
        dimension 1')``. The fix is ordering, not a new check, so the test
        asserts the statsmodels type and message rather than a puremacro one.
        """
        sm_dw = pytest.importorskip("statsmodels.stats.stattools")
        resids = np.random.default_rng(0).normal(size=20)

        with pytest.raises(Exception) as sm_exc:
            sm_dw.durbin_watson(resids, axis)
        with pytest.raises(type(sm_exc.value)) as pm_exc:
            durbin_watson(resids, axis)
        assert type(pm_exc.value).__name__ == "AxisError"
        assert str(pm_exc.value) == str(sm_exc.value)


# ---------------------------------------------------------------------------
# Pyodide contract
# ---------------------------------------------------------------------------

def test_modules_do_not_import_statsmodels():
    """Release gate 2 in miniature: the shipped modules stay import-clean.

    Imports the four modules in a subprocess with a fresh interpreter and
    asserts statsmodels never lands in ``sys.modules``. Done out of process
    because this test file itself imports statsmodels, so an in-process
    check would be inert — the positive control is that the subprocess
    reports the four modules as actually imported.
    """
    import os
    import pathlib
    import subprocess
    import sys

    import puremacro

    # Point the child at the same puremacro this session imported, rather
    # than whatever a bare cwd would resolve to.
    root = str(pathlib.Path(puremacro.__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=root + os.pathsep
               + os.environ.get("PYTHONPATH", ""))

    code = (
        "import sys\n"
        "import puremacro.inference.multiple as a\n"
        "import puremacro.inference.collinearity as b\n"
        "import puremacro.inference.proportions as c\n"
        "import puremacro.inference.diagnostics as d\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in "
        "{'statsmodels', 'linearmodels', 'arch'}]\n"
        "print(len([a, b, c, d]), loaded)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True, cwd=root, env=env)
    assert out.stdout.strip() == "4 []", out.stdout + out.stderr
