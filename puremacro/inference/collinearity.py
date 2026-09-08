"""Variance inflation factors — the multicollinearity diagnostic.

A pure-numpy drop-in for
``statsmodels.stats.outliers_influence.variance_inflation_factor``, extended
to compute the whole vector in one call and to label it with the DataFrame's
column names.

The trap this module exists to document
---------------------------------------
``variance_inflation_factor`` expects **the constant to be IN the design**.
It regresses column *j* on the *other columns of whatever you passed*, and
its R-squared is centered only when statsmodels detects a constant among
those other columns. Strip the intercept first — the natural-looking
``X[:, 1:]`` after ``add_constant`` — and every auxiliary regression becomes
constant-free, the R-squared silently switches to the **uncentered** form,
and the VIFs come out larger, sometimes by an order of magnitude, for
variables whose mean is far from zero.

Neither convention is a bug; they answer different questions. But they are
different numbers under the same name, so this module reproduces
statsmodels' detection rule exactly (including its *implicit*-constant
branch, where a saturated set of dummies spans the intercept without any
single column being constant) rather than approximating it, and
:func:`vif` warns when it is handed a design with no constant at all.

Notes
-----
Verified against statsmodels 0.14.6
(``statsmodels/stats/outliers_influence.py:152`` for the estimator,
``statsmodels/base/data.py:130`` for the constant detection,
``statsmodels/regression/linear_model.py:1773`` for the centered /
uncentered R-squared switch, and ``:1725-1743`` for ``centered_tss``, whose
three branches are the reason the sum below is written with an explicit
weight vector).

References
----------
Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). Regression Diagnostics:
    Identifying Influential Data and Sources of Collinearity. Wiley.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .._linalg import inv_xtx

__all__ = ["vif"]


def _k_constant(exog):
    """Return statsmodels' ``k_constant`` for a design matrix.

    Reproduces ``statsmodels.base.data.ModelData._handle_constant`` with
    ``hasconst=None`` (the default path taken by ``OLS(y, X)``). The rule is
    not "is there a column of ones": it is a four-branch search that also
    catches a constant column of any non-zero value, prefers a literal
    column of ones when several constants are present, and — when no
    explicit constant is found — falls back to a *rank* test for an
    implicit one.

    Parameters
    ----------
    exog : ndarray, shape (n, k)
        Design matrix. Must be float and finite.

    Returns
    -------
    int
        1 if statsmodels would treat the design as containing a constant
        (and therefore center its total sum of squares), else 0.

    Raises
    ------
    ValueError
        If ``exog`` contains ``inf`` or ``nan``, mirroring statsmodels'
        ``MissingDataError``.
    """
    exog_max = np.max(exog, axis=0)
    if not np.isfinite(exog_max).all():
        raise ValueError("vif: exog contains inf or nans")
    exog_min = np.min(exog, axis=0)
    const_idx = np.where(exog_max == exog_min)[0].squeeze()
    k_constant = const_idx.size
    check_implicit = False

    if k_constant == 1:
        # A single constant column only counts if it is not the zero column.
        if exog[:, const_idx].mean() == 0:
            check_implicit = True
    elif k_constant > 1:
        # Several constant columns: a literal column of ones wins; failing
        # that, the first non-zero constant; failing that, nothing.
        values = []
        for idx in const_idx:
            value = exog[:, idx].mean()
            if value == 1:
                k_constant = 1
                break
            values.append(value)
        else:
            if np.any(np.array(values) != 0):
                k_constant = 1
            else:
                check_implicit = True
    else:  # k_constant == 0
        check_implicit = True

    if check_implicit:
        # Implicit constant: adding a column of ones does not raise the
        # rank, i.e. the intercept already lies in the column space (a
        # saturated dummy set is the usual case).
        augmented = np.column_stack((np.ones(exog.shape[0]), exog))
        k_constant = int(
            np.linalg.matrix_rank(exog) == np.linalg.matrix_rank(augmented)
        )
    return int(k_constant)


def _vif_one(exog, idx):
    """Variance inflation factor for a single column index.

    Parameters
    ----------
    exog : ndarray, shape (n, k)
        Full design matrix, float.
    idx : int
        Column whose VIF is wanted.

    Returns
    -------
    float
        ``1 / (1 - R2_idx)`` from the auxiliary regression of column ``idx``
        on all the others. ``0.0`` when column ``idx`` is constant and the
        auxiliary design carries a constant, which is statsmodels' answer
        for that degenerate case; see Notes.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the residual sum of squares is exactly zero, i.e. column ``idx``
        lies in the span of the others and its VIF is infinite. ``vif``
        normally catches this earlier, on the whole design; this is the
        backstop for a design that slips under the condition-number gate.

    Notes
    -----
    Every division here is evaluated in ``np.float64``, never in Python
    ``float``. That is not a stylistic preference: the centered total sum of
    squares is exactly ``0.0`` whenever column ``idx`` is constant, and
    statsmodels — which is doing numpy arithmetic — gets ``ssr / 0.0 =
    inf``, hence ``R2 = -inf`` and ``VIF = 1 / inf = 0.0``. Python's float
    raises ``ZeroDivisionError`` for the same expression, which would kill
    the whole call (including the healthy VIFs of every other column) on
    any design where the rank test finds an implicit constant among the
    others — which numpy's rank tolerance
    (``max(M, N) * eps * s_max``) makes true for an ``add_constant`` design
    containing a column of magnitude 1e13 or more, measured at n=120 on a
    column varying by 50%, and from 1e11 on one varying by 0.1%. That is any
    nominal series in rupiah, lira or yen, which is not an exotic corner.
    The same reasoning covers ``R2`` rounding to exactly 1 from below: numpy
    gives ``inf``, Python raises.
    """
    x_i = exog[:, idx]
    x_noti = np.delete(exog, idx, axis=1)

    # ``pinv``, not ``lstsq`` and not a normal-equations inverse, because
    # ``pinv`` is literally what statsmodels' default
    # ``OLS.fit(method="pinv")`` calls — same SVD, same singular-value
    # cutoff, so the residual sum of squares comes out bit-identical.
    # ``lstsq(rcond=None)`` looks equivalent and is not: its cutoff is
    # ``max(m, n) * eps`` against numpy's ``pinv`` default of
    # ``1e-15 * max(m, n)``, which on a design whose columns differ by many
    # orders of magnitude truncates a different set of singular values and
    # moves the VIF in the 8th digit. A normal-equations inverse would be
    # worse still: it squares the condition number, exactly where VIF is
    # large and precision matters most.
    #
    # Rank deficiency *among the other columns* is harmless here and is
    # deliberately not gated: the residual is a projection onto their span,
    # and a redundant column does not change that span. The singularity that
    # does matter — column ``idx`` lying inside that span — is caught below
    # and gated once, on the whole design, in ``vif``.
    beta = np.linalg.pinv(x_noti) @ x_i
    resid = x_i - x_noti @ beta
    # ``RegressionResults.ssr`` (linear_model.py:1720) is ``np.dot(wresid,
    # wresid)``, so this is a dot product and not ``np.sum(resid ** 2)``.
    ssr = np.float64(resid @ resid)

    if _k_constant(x_noti):
        # ``RegressionResults.centered_tss`` (linear_model.py:1725-1743) has
        # three branches and OLS does *not* take the ``np.dot`` one at the
        # bottom. OLS inherits WLS, whose ``__init__`` expands ``weights=1``
        # into an all-ones ndarray, so ``getattr(model, 'weights', None)``
        # is not None and the *weights* branch at :1731-1733 runs instead:
        # ``np.sum(weights * (endog - np.average(endog, weights=weights))**2)``.
        # ``np.sum`` and ``np.dot`` sum in different orders, so the two forms
        # differ in the last bit of tss for about two vectors in three, and
        # the VIFs they produce differ for about one column in five: 139 of
        # 633 on the sweep in ``test_bit_identity_on_near_collinear_designs``
        # with the dot product, 0 of 633 with the expression below. The
        # propagated size is small there (6.7e-16 absolute at worst), but
        # VIF = 1/(1-R2) amplifies a *relative* perturbation of tss by a
        # factor of VIF, so the same flipped bit is worth of order 1 in
        # absolute terms once the VIF reaches 1e8. The ones vector is
        # materialised rather than short-circuited precisely so this stays
        # the expression statsmodels evaluates, not one that is merely equal
        # to it in exact arithmetic.
        weights = np.ones_like(x_i)
        mean = np.average(x_i, weights=weights)
        tss = np.float64(np.sum(weights * (x_i - mean) ** 2))
    else:
        # ``uncentered_tss`` (linear_model.py:1745-1754) *is* a dot product.
        tss = np.float64(x_i @ x_i)

    if ssr == 0.0:
        # R-squared is exactly 1: the column is an exact linear combination
        # of the others and its VIF is infinite. statsmodels evaluates
        # ``1. / 0.`` here and returns ``inf`` behind a RuntimeWarning.
        raise np.linalg.LinAlgError(
            f"vif: column {idx} is an exact linear combination of the other "
            "columns, so its variance inflation factor is infinite and the "
            "design is not estimable. Drop the redundant regressor."
        )

    # Written as statsmodels writes it (form the R-squared, then invert its
    # complement) rather than the algebraically equal ``tss / ssr``: the two
    # differ in the last bits, and parity is measured against statsmodels.
    #
    # ``errstate`` only silences numpy's generic "divide by zero encountered"
    # text. The condition it fires on is not swallowed: ``vif`` reports it
    # by name, with the offending column index, after the values come back.
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared_i = 1 - ssr / tss
        return float(1.0 / (1.0 - r_squared_i))


def vif(exog, exog_idx=None):
    """Variance inflation factor(s) for a design matrix.

    ``VIF_j = 1 / (1 - R2_j)``, where ``R2_j`` comes from an OLS of column
    ``j`` of ``exog`` on every other column. It is the factor by which the
    sampling variance of coefficient ``j`` is inflated relative to an
    orthogonal design; the usual rules of thumb are 5 (worth a look) and 10
    (severe).

    Numerically identical to
    ``statsmodels.stats.outliers_influence.variance_inflation_factor``, which
    it can replace positionally: ``vif(X, j)`` for
    ``variance_inflation_factor(X, j)``. The two documented exceptions are a
    negative ``j``, which wraps here and returns ``inf`` there, and a
    singular design, which raises here; both are in Notes.

    Parameters
    ----------
    exog : ndarray or DataFrame, shape (n, k)
        Design matrix with **all** explanatory variables, *including the
        constant* if the model has one. See Notes — this is the single
        decision that moves the numbers.
    exog_idx : int, sequence of int, or None, default None
        Which column(s) to compute. ``None`` computes every column.
        Negative indices count from the end, Python-style — which is *not*
        what statsmodels does with them; see Notes. A boolean mask is
        rejected rather than reinterpreted as integers.

    Returns
    -------
    float, ndarray, or pandas.Series
        A float when ``exog_idx`` is a single integer. Otherwise a
        pandas Series indexed by the selected column names when ``exog`` is
        a DataFrame, and a float ndarray when it is not.

    Raises
    ------
    ValueError
        If ``exog`` is not 2-D, has fewer than two columns (there is nothing
        to regress on), contains ``inf`` / ``nan``, or if ``exog_idx`` is a
        boolean mask.
    IndexError
        If ``exog_idx`` is out of range.
    numpy.linalg.LinAlgError
        If ``exog`` is singular or numerically indistinguishable from
        singular — i.e. at least one VIF is infinite. The message names the
        columns most aligned with the null space. See Notes for the
        threshold and for what statsmodels does instead.

    Warns
    -----
    UserWarning
        When ``exog`` contains no constant column, explicit or implicit —
        legal, but almost always a mistake, because it silently switches
        every auxiliary R-squared to the uncentered form; and when a
        requested column is constant while the rest of the design carries a
        constant, whose VIF is then ``0.0`` in statsmodels and here alike —
        a degenerate value that reads like "perfectly orthogonal" and is
        not. See Notes for both.

    Notes
    -----
    **Put the constant in.** statsmodels' R-squared is centered when it
    detects a constant among the *regressors of the auxiliary regression*
    and uncentered when it does not. Passing ``add_constant(X)`` leaves a
    constant in every auxiliary design, so every R-squared is centered — the
    textbook VIF. Passing ``X`` without the constant, or slicing it off with
    ``X[:, 1:]``, makes every auxiliary design constant-free and every
    R-squared uncentered, which inflates the VIF of any variable with a
    non-zero mean. Both are reproduced here, faithfully; the warning fires
    only for the second.

    **Constant detection is not "a column of ones".** A design with a
    saturated set of dummies has an implicit constant and is treated as
    centered even with no constant column present. This function
    reproduces that rank-based branch, so a dummy-saturated design gives the
    same answer as statsmodels rather than the (larger) uncentered one.

    **A degenerate design raises here and does not in statsmodels.** This is
    the one deliberate behavioural divergence, and it has a measured
    boundary. ``exog`` is passed through ``puremacro._linalg.inv_xtx`` after
    each column is scaled to unit length — and, when the design has an
    explicit constant column, after each non-constant column is also
    centered, because a VIF computed against an intercept is exactly
    invariant to shifting a column and the gate has to be invariant to
    whatever the VIF is invariant to. The gate is therefore a test of
    collinearity and not of units *or* of location; it rejects at
    ``cond(X'X) ≳ 1e14``, which on the reference sweep in
    ``tests/test_inference_extras_parity.py`` corresponds to a VIF of order
    ``1e13``. Below that — up to VIFs of ``1e12``, which is already far past
    any interpretable range — this function is bit-identical to statsmodels
    (``np.testing.assert_array_equal``, not ``assert_allclose``; see
    ``test_bit_identity_on_near_collinear_designs``). Above it, statsmodels
    reports whatever the pseudo-inverse produced (``8.3e13``, ``9.0e15``, or
    ``inf`` behind a RuntimeWarning); those numbers are floating-point
    noise, not measurements, and this function raises
    ``numpy.linalg.LinAlgError`` naming the culprit columns instead.

    Centering is skipped when the constant is only *implicit* — a saturated
    dummy set — and that is not an oversight. Deleting one dummy destroys
    the implicit constant, so the auxiliary regressions are constant-free,
    their R-squareds uncentered and their VIFs location-dependent. Centering
    such a design would make its columns sum to zero exactly and the gate
    would reject a design statsmodels evaluates without complaint.

    **A constant column's own VIF is ``0.0``, not an error.** If the target
    column is constant and the auxiliary design has a constant, the centered
    total sum of squares is exactly zero, ``R2`` is ``-inf`` and
    ``1 / (1 - R2)`` is ``0``. statsmodels returns that 0.0 behind numpy's
    anonymous "divide by zero" RuntimeWarning; this function returns the
    same number and names the column in a ``UserWarning``, because a VIF of
    zero otherwise reads as "perfectly orthogonal" when it means "no
    variance to inflate". Note that numpy's rank tolerance makes the
    auxiliary design's implicit-constant test true for an ``add_constant``
    design containing a column of magnitude 1e13 or more — from 1e11 when
    that column barely varies — so this is the normal outcome for a nominal
    series in rupiah, lira or yen, not an exotic corner.

    **Negative indices wrap here and do not in statsmodels.**
    ``vif(X, -1)`` returns the VIF of the last column; statsmodels'
    ``variance_inflation_factor(X, -1)`` returns ``inf``, because it drops
    the target column with ``mask = np.arange(k) != exog_idx``, which masks
    nothing at all for a negative index — the column is left in its own
    auxiliary design and the auxiliary R-squared is 1 by construction. The
    result is infinite for every negative index on every design, except
    where the target is a constant column and the degenerate branch above
    returns ``0.0`` first; neither is that column's VIF. That is a
    statsmodels bug rather than a convention, so it is not reproduced;
    ``vif(X, -j)`` means ``vif(X, k - j)``, as everywhere else in Python.
    Positive indices, which is what the corpus passes, are unaffected.

    The gate is on the whole design, so an exactly-redundant regressor
    suppresses *every* VIF rather than only its own. That is deliberate:
    when a design is not estimable, the VIFs of its other columns are
    answers to a question nobody asked. Drop the redundant column and
    re-run.

    There is deliberately **no** per-column gate on the auxiliary design.
    The auxiliary regression only needs the *span* of the remaining columns,
    and near-redundancy among them does not change that span, so a column
    whose own VIF is a healthy 1.05 still gets its exact value even when the
    other regressors are badly conditioned between themselves.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1 = rng.normal(size=n)
    >>> x2 = x1 + 0.1 * rng.normal(size=n)   # nearly a copy of x1
    >>> x3 = rng.normal(size=n)
    >>> X = np.column_stack([np.ones(n), x1, x2, x3])
    >>> v = vif(X)
    >>> bool(v[1] > 10 and v[2] > 10)        # x1 and x2 inflate each other
    True
    >>> bool(v[3] < 1.1)                     # x3 is orthogonal to both
    True

    With a DataFrame the result is labelled:

    >>> import pandas as pd
    >>> df = pd.DataFrame(X, columns=["const", "x1", "x2", "x3"])
    >>> list(vif(df).index)
    ['const', 'x1', 'x2', 'x3']
    >>> round(float(vif(df, 3)), 3) == round(float(v[3]), 3)
    True
    """
    is_frame = isinstance(exog, pd.DataFrame)
    names = list(exog.columns) if is_frame else None
    arr = np.asarray(exog, dtype=float)

    if arr.ndim != 2:
        raise ValueError(
            f"vif: exog must be 2-D (n, k), got shape {arr.shape}."
        )
    n, k = arr.shape
    if k < 2:
        raise ValueError(
            f"vif: exog has {k} column(s); a variance inflation factor needs "
            "at least two so that one column can be regressed on the rest."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            "vif: exog contains inf or nan. Drop or impute the offending "
            "rows before computing variance inflation factors."
        )

    # Singularity gate, per the diagnostic-error contract in CONTRIBUTING.md.
    # It sits on the *whole* design rather than on each auxiliary regression
    # because that is where it means something: a singular ``exog`` is
    # exactly the condition under which some VIF is infinite, and
    # ``inv_xtx`` reports which columns are aligned with the null space —
    # which is the answer the caller came for. (An auxiliary design that is
    # merely rank-deficient among the *other* columns is harmless; see
    # ``_vif_one``.)
    #
    # The gate runs on *column-normalised* regressors. ``inv_xtx`` rejects on
    # the condition number of X'X, which for raw data is dominated by units:
    # a design holding a constant alongside a variable measured in millionths
    # is perfectly well identified but has cond(X'X) ~ 1e24. Scaling each
    # column to unit length makes the gate a test of collinearity rather than
    # of units, which is the only thing a VIF is about.
    norms = np.sqrt((arr ** 2).sum(axis=0))
    zero_cols = np.nonzero(norms == 0)[0]
    if zero_cols.size:
        raise np.linalg.LinAlgError(
            f"vif: column(s) {zero_cols.tolist()} of exog are identically "
            "zero, so X'X is singular. Drop the empty regressor(s) before "
            "computing variance inflation factors."
        )

    # ...and, when an *explicit* constant column is present, on *centered*
    # regressors as well. Deleting any other column then leaves that constant
    # behind, so every auxiliary regression has an intercept, and VIF_j is
    # exactly invariant to adding a constant to any column: shifting column j
    # changes neither its centered tss nor the residual, and shifting any
    # other column leaves the span of {1, others} untouched. A gate that is
    # not equally invariant rejects for a location problem that changes no
    # VIF — which is what used to happen to levels data (columns sharing a
    # common mean of 1e7), where statsmodels returns VIFs of 1.00-1.08 and
    # this function refused the whole call.
    #
    # The test is "an explicit constant column", not ``_k_constant``, and the
    # difference is load-bearing. A saturated dummy set has an *implicit*
    # constant, but deleting one dummy destroys it: the auxiliary regressions
    # are then constant-free, their R-squareds uncentered, and the VIFs are
    # finite and location-dependent. Centering that design would make its
    # columns sum to zero exactly and the gate would reject a design
    # statsmodels evaluates perfectly happily.
    #
    # Constant columns are left uncentered rather than zeroed, so a second
    # constant column still shows up as the exact duplicate of the first that
    # it is (both VIFs really are infinite in that design).
    centered = arr - arr.mean(axis=0)
    is_constant_col = ~np.any(centered != 0.0, axis=0)
    has_explicit_constant = bool(is_constant_col.any())
    if has_explicit_constant:
        gate = np.where(is_constant_col, arr, centered)
    else:
        # No constant column: the VIFs are uncentered R-squareds, which
        # genuinely do depend on the location of the data, so neither may the
        # gate.
        gate = arr
    gate = gate / np.sqrt((gate ** 2).sum(axis=0))

    try:
        inv_xtx(gate, name="vif")
    except np.linalg.LinAlgError as exc:
        if int(np.linalg.matrix_rank(gate)) < k:
            # Exact rank deficiency: ``inv_xtx`` has already named the columns
            # most aligned with the null space and told the caller to drop a
            # redundant regressor, which is the right advice. Pass it through.
            raise
        # Full rank but past the conditioning threshold. ``inv_xtx``'s stock
        # advice ("Try rescaling regressors") is wrong here, because the gate
        # has already normalised — and, above, centered — every column, so
        # there is no rescaling left to do.
        cure = (
            "Drop the near-redundant regressor: no rescaling or recentering "
            "is left to try, since the gate already normalises every column "
            "to unit length and centers it."
            if has_explicit_constant else
            "The gate could not center, because exog has no constant column "
            "and the uncentered R-squared it implies is not location-"
            "invariant. If the columns share a large common mean, pass "
            "add_constant(X): that is both the textbook VIF and the fix for "
            "this conditioning. Otherwise drop the near-redundant regressor."
        )
        raise np.linalg.LinAlgError(
            f"vif: exog is numerically singular at full rank {k} "
            "(condition number ≳ 1e14 after column normalisation), so at "
            "least one variance inflation factor is enormous — of order "
            "1e12 or more on the reference sweep — and is floating-point "
            "noise rather than a measurement. " + cure
        ) from exc

    # This one *is* ``_k_constant``: it is about the centered/uncentered
    # convention, which an implicit constant does change.
    if not _k_constant(arr):
        warnings.warn(
            "vif: exog has no constant column, so every auxiliary R-squared "
            "is uncentered and the VIFs are larger than the textbook ones. "
            "statsmodels behaves identically; pass add_constant(X) if you "
            "wanted the centered convention.",
            UserWarning,
            stacklevel=2,
        )

    if exog_idx is None:
        idxs = list(range(k))
        scalar = False
    else:
        # A boolean mask is the one plausible spelling that must not be
        # accepted: ``int(True) == 1`` and ``int(False) == 0``, so a mask
        # selecting columns 0 and 2 would silently return the VIFs of
        # columns 1 and 0 — right shape, right dtype, wrong columns, no
        # error. Checked on the array's dtype so it catches a bare ``True``,
        # a list of bools and a numpy bool array alike.
        idx_arr = np.asarray(exog_idx)
        if idx_arr.dtype == bool:
            raise ValueError(
                "vif: exog_idx must be integer column indices, not a boolean "
                "mask — True and False would be read as columns 1 and 0. "
                "Pass np.flatnonzero(mask) instead."
            )
        if idx_arr.ndim == 0:
            idxs = [int(idx_arr)]
            scalar = True
        else:
            idxs = [int(i) for i in idx_arr.ravel()]
            scalar = False

    for i in idxs:
        if not -k <= i < k:
            raise IndexError(
                f"vif: exog_idx {i} is out of range for a design with "
                f"{k} columns."
            )
    # Python's own wrap-around, and deliberately not statsmodels' behaviour;
    # see the "Negative indices" paragraph in the Notes above.
    idxs = [i % k for i in idxs]

    out = np.array([_vif_one(arr, i) for i in idxs], dtype=float)

    # A VIF of exactly zero can arise from one condition only: R2 = -inf,
    # i.e. a centered total sum of squares of exactly 0, i.e. a constant
    # target column measured against an auxiliary design that carries a
    # constant. The value matches statsmodels (which gets there through
    # ``ssr / 0.0`` behind numpy's anonymous "divide by zero" RuntimeWarning);
    # naming the column is the part statsmodels does not do, and 0.0 reads
    # far too much like "no collinearity at all" to leave unexplained.
    degenerate = [names[i] if is_frame else i
                  for i, v in zip(idxs, out) if v == 0.0]
    if degenerate:
        warnings.warn(
            f"vif: column(s) {degenerate} of exog are constant, so their "
            "centered total sum of squares is zero and their variance "
            "inflation factor is reported as 0.0 — a degenerate value, not "
            "an absence of collinearity. statsmodels returns the same 0.0. "
            "The VIFs of the other columns are unaffected.",
            UserWarning,
            stacklevel=2,
        )

    if scalar:
        return float(out[0])
    if is_frame:
        return pd.Series(out, index=[names[i] for i in idxs])
    return out
