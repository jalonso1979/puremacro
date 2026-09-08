"""General-purpose linear regression with statsmodels-compatible inference.

This module is the pure-numpy replacement for ``statsmodels.api.OLS`` /
``WLS`` / ``add_constant`` in a notebook that has to run under Pyodide.
The design goal is narrow and testable: a call site should change only
its constructor line, and **no number should move**. Everything below —
the covariance formulas, the small-sample factors, the ``use_t``
defaults, the degrees of freedom used for p-values — reproduces
statsmodels 0.14.6 on a well-conditioned full-rank design. How far that
holds as the design degrades is measured, not assumed: see
:func:`ols`'s Notes on conditioning.

The traps this module exists to get right, all of them silent if you get
them wrong:

* ``cov_type='HAC'`` defaults to ``use_correction=False`` while
  ``cov_type='cluster'`` defaults to ``use_correction=True`` and
  ``'hac-groupsum'`` to ``use_correction='cluster'``. A wrapper that
  hardcodes one convention moves half of any real corpus.
* ``use_t`` is ``True`` for ``'nonrobust'`` and ``False`` for every
  robust covariance, so HC1/HAC p-values are normal-tail unless the
  caller asks otherwise.
* ``cluster``, ``hac-panel`` and ``hac-groupsum`` set
  ``df_resid_inference = G - 1``; that, not ``n - k``, is the t
  denominator and the ``f_test`` denominator df.
* ``hac-groupsum`` uses **two different** group counts: the variance
  correction uses the number of time periods, while
  ``df_resid_inference`` uses the number of *panel units* (the number of
  times the time index resets). This is statsmodels' behaviour and it is
  reproduced here rather than tidied up.
* WLS runs every sandwich on the **weighted** arrays (``wexog``,
  ``wresid``), including the HC family. Using raw residuals with a
  weighted bread is a plausible-looking mistake that produces plausible-
  looking numbers.
* HC2 and HC3 divide by ``1 - h_ii``, which is exactly zero for a row
  that is the only observation with a non-zero value on some regressor.
  An unguarded division there turns the *whole* covariance matrix into
  NaN, and statsmodels' finite answer on the same design is the ratio of
  two rounding errors. This module raises instead; see :func:`ols`'s
  Notes, departure 6.

Differences from statsmodels that are deliberate, and their consequences,
are listed in :func:`ols`'s Notes.

Examples
--------
>>> import numpy as np
>>> from puremacro.regress.ols import add_constant, ols
>>> rng = np.random.default_rng(0)
>>> X = add_constant(rng.standard_normal((50, 2)))
>>> y = X @ np.array([1.0, 0.5, -0.2]) + rng.standard_normal(50)
>>> res = ols(y, X, cov_type="HC1")
>>> res.params.shape, res.use_t
((3,), False)
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .._linalg import inv_xtx
from ..inference._ols_helpers import ols_hac
from ..inference.dk import driscoll_kraay

__all__ = [
    "add_constant",
    "ols",
    "OLSResult",
    "TTestResult",
    "FTestResult",
    "PredictionResult",
]


# ---------------------------------------------------------------------------
# add_constant
# ---------------------------------------------------------------------------

def add_constant(data, prepend: bool = True, has_constant: str = "skip"):
    """Prepend (or append) a column of ones, as ``statsmodels.add_constant``.

    Parameters
    ----------
    data : array_like or pandas.DataFrame or pandas.Series
        Column-ordered design matrix. A 1-D ndarray is treated as a single
        column; a Series is promoted to a one-column DataFrame first, both
        exactly as statsmodels does.
    prepend : bool, default True
        Put the constant in the first column. If False it is appended last.
    has_constant : {'skip', 'add', 'raise'}, default 'skip'
        What to do when ``data`` already contains a constant column.
        ``'skip'`` returns the data unchanged (the statsmodels default,
        and the reason 61 call sites can be ported blind), ``'add'``
        adds a second constant anyway, ``'raise'`` raises ``ValueError``.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Same container type as the input. For a DataFrame the new column
        is named ``'const'``.

    Raises
    ------
    ValueError
        If ``has_constant='raise'`` and a constant column is present, if
        ``has_constant`` is not one of the three accepted strings, or if
        the input has more than two dimensions.

    Notes
    -----
    A column counts as constant when its maximum equals its minimum *and*
    it is not identically zero — an all-zero column is not treated as a
    constant, matching ``statsmodels.tools.tools.add_constant`` and
    ``statsmodels.tsa.tsatools.add_trend``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> add_constant(np.arange(3.0))
    array([[1., 0.],
           [1., 1.],
           [1., 2.]])
    >>> list(add_constant(pd.DataFrame({"x": [1.0, 2.0]})).columns)
    ['const', 'x']
    >>> add_constant(np.ones((2, 1)))          # already constant: skipped
    array([[1.],
           [1.]])
    """
    if has_constant not in ("skip", "add", "raise"):
        raise ValueError(
            f"has_constant must be one of 'skip', 'add', 'raise'; got {has_constant!r}"
        )

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return _add_constant_pandas(data, prepend, has_constant)

    x = np.asarray(data)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError("Only implemented for 2-dimensional arrays")

    is_const = _is_constant_column(x)
    if is_const.any():
        if has_constant == "skip":
            # statsmodels returns the input untouched here — not a copy,
            # and not up-cast to float. Reproduced so that a caller who
            # relies on the dtype sees the same thing.
            return x
        if has_constant == "raise":
            cols = ",".join(str(c) for c in np.arange(x.shape[1])[is_const])
            raise ValueError(f"Column(s) {cols} are constant.")

    cols = [np.ones(x.shape[0]), x]
    return np.column_stack(cols if prepend else cols[::-1])


def _add_constant_pandas(data, prepend: bool, has_constant: str):
    """The DataFrame/Series branch of :func:`add_constant`."""
    x = pd.DataFrame(data) if isinstance(data, pd.Series) else data.copy()

    def _safe_is_const(s) -> bool:
        # Mixed-dtype protection, as in statsmodels' add_trend: a column
        # of strings must answer "not constant", not raise.
        try:
            arr = np.asarray(s)
            return bool(arr.max() == arr.min()) and bool(np.any(arr != 0.0))
        except Exception:
            return False

    col_const = [_safe_is_const(x[c]) for c in x.columns]
    if any(col_const):
        if has_constant == "raise":
            const_cols = ", ".join(
                str(c) for c, flag in zip(x.columns, col_const) if flag
            )
            raise ValueError(
                f"x contains one or more constant columns. Column(s) "
                f"{const_cols} are constant. Adding a constant with "
                "trend='c' is not allowed."
            )
        if has_constant == "skip":
            return x

    const = pd.DataFrame(
        np.ones((len(x), 1)), index=x.index, columns=["const"]
    )
    frames = [const, x] if prepend else [x, const]
    return pd.concat(frames, axis=1)


def _is_constant_column(x: np.ndarray) -> np.ndarray:
    """Boolean mask of non-zero constant columns of a 2-D array.

    Raises ``ValueError`` on a zero-row array. statsmodels reduces
    ``max``/``min`` over the rows here and lets numpy raise the empty-
    reduction error; answering "no constant column" instead would hand
    back a ``(0, k + 1)`` design that can never be fitted, which is the
    silent-garbage outcome the package exists to avoid.
    """
    if x.shape[0] == 0:
        raise ValueError(
            "add_constant: cannot tell whether a zero-row array of shape "
            f"{x.shape} already has a constant column. statsmodels raises "
            "the equivalent ValueError on the same input."
        )
    return (np.ptp(x, axis=0) == 0) & np.all(x != 0.0, axis=0)


# ---------------------------------------------------------------------------
# covariance kernels
# ---------------------------------------------------------------------------

# HC2 and HC3 divide by ``1 - h_ii``. Below this gap the quotient is
# dominated by the rounding error in the residual (which is O(eps.||y||)),
# so ``u_i / (1 - h_ii)`` carries fewer than four correct digits and at
# ``1 - h_ii == 0`` it is 0/0. 1e-12 is roughly 4500 machine epsilons: it
# admits a genuine near-singleton (h = 1 - 1e-9 keeps ~7 digits) and
# refuses the arithmetic that has none.
_LEVERAGE_TOL = 1e-12


def _default_maxlags(n_periods: int) -> int:
    """statsmodels' fallback bandwidth ``floor(4 (T/100)^(2/9))``."""
    return int(np.floor(4 * (n_periods / 100.0) ** (2.0 / 9.0)))


def _check_maxlags(value, cov_type: str) -> int:
    """Validate a Bartlett bandwidth: a non-negative whole number.

    A bare ``int(value)`` is not enough. ``int(-3)`` leaves the lag loop
    empty, which turns HAC into HC0 *and keeps calling it HAC*; ``int(3.7)``
    silently moves the bandwidth. Both are the kind of quiet degradation
    that a horizon loop (`maxlags = h - 1` at `h = 0`) produces by
    accident, so both are refused here. statsmodels raises ``IndexError``
    on the first and ``TypeError`` on the second.
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"cov_type={cov_type!r}: cov_kwds['maxlags'] must be a "
            f"non-negative integer; got {value!r}"
        ) from None
    if not np.isfinite(as_float) or as_float != int(as_float):
        raise ValueError(
            f"cov_type={cov_type!r}: cov_kwds['maxlags'] must be a whole "
            f"number of lags; got {value!r}. Truncating it here would move "
            "the bandwidth without saying so."
        )
    lags = int(as_float)
    if lags < 0:
        raise ValueError(
            f"cov_type={cov_type!r}: cov_kwds['maxlags'] must be "
            f"non-negative; got {lags}. A negative bandwidth leaves the lag "
            "sum empty, which silently publishes HC0 numbers under the "
            f"label {cov_type!r}."
        )
    return lags


def _hac_weights(kwds: dict, nlags: int, cov_type: str) -> np.ndarray | None:
    """Resolve ``cov_kwds['kernel']`` / ``['weights_func']`` to lag weights.

    Returns ``None`` when the rule is statsmodels' Bartlett default, so
    the caller can stay on the verified default code path
    (:func:`~puremacro.inference._ols_helpers.ols_hac` /
    :func:`~puremacro.inference.dk.driscoll_kraay`) instead of
    re-deriving numbers that are already measured.

    ``kernel`` overwrites ``weights_func`` when both are given, exactly as
    ``base/covtype.py`` does (`kwds['weights_func'] = kwds.pop('kernel')`).
    """
    spec = kwds["kernel"] if "kernel" in kwds else kwds.get("weights_func")
    bartlett = 1.0 - np.arange(nlags + 1) / (nlags + 1.0)
    if spec is None:
        return None
    if callable(spec):
        w = np.asarray(spec(nlags), dtype=float).ravel()
    elif isinstance(spec, str):
        if spec == "bartlett":
            w = bartlett
        elif spec == "uniform":
            w = np.ones(nlags + 1)
        else:
            raise ValueError(
                f"cov_type={cov_type!r}: kernel {spec!r} is not available. "
                "statsmodels' kernel_dict holds 'bartlett' and 'uniform'."
            )
    else:
        raise ValueError(
            f"cov_type={cov_type!r}: cov_kwds['kernel'] must be 'bartlett', "
            f"'uniform' or a callable weights function; got {spec!r}"
        )
    if w.shape[0] != nlags + 1:
        raise ValueError(
            f"cov_type={cov_type!r}: the kernel returned {w.shape[0]} "
            f"weights for maxlags={nlags}; statsmodels expects maxlags + 1."
        )
    return None if np.array_equal(w, bartlett) else w


def _hac_meat(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``S = w0 x'x + sum_l w_l (Gamma_l + Gamma_l')`` over consecutive rows.

    The generic form of ``sandwich_covariance.S_hac_simple``, used only
    when a non-default kernel is requested; the Bartlett path stays on
    the already-measured helpers.
    """
    S = weights[0] * (x.T @ x)
    for lag in range(1, len(weights)):
        if lag >= x.shape[0]:
            break
        g = x[lag:].T @ x[:-lag]
        S = S + weights[lag] * (g + g.T)
    return S


def _panel_key(values, n: int, cov_type: str, label: str) -> np.ndarray:
    """Coerce a ``groups`` / ``time`` vector and check its length.

    Without the length check a short vector reaches ``np.add.at`` or a
    boolean mask and comes back as ``"array is not broadcastable to
    correct shape"`` — an error that names neither the keyword nor the
    two lengths.
    """
    arr = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    arr = np.ravel(arr)
    if arr.shape[0] != n:
        raise ValueError(
            f"cov_type={cov_type!r}: cov_kwds[{label!r}] has {arr.shape[0]} "
            f"entries but the regression has {n} observations. Subset the "
            "labels the same way the design was subset."
        )
    return arr


def _hc_cov(wexog: np.ndarray, wresid: np.ndarray, bread: np.ndarray,
            kind: str, nobs: float, df_resid: float) -> np.ndarray:
    """Eicker-Huber-White covariance, HC0 through HC3.

    ``bread`` is ``(X'WX)^-1``; ``wexog`` / ``wresid`` are the weighted
    design and residuals, which for unweighted OLS are the raw ones.
    HC2/HC3 leverages are the diagonal of the weighted hat matrix, as in
    ``RegressionResults.cov_HC2`` — not the raw-exog hat matrix used by
    the (unused) module-level ``sandwich_covariance.cov_hc2``.
    """
    if kind in ("HC0", "HC1"):
        het_scale = wresid ** 2
        if kind == "HC1":
            het_scale = het_scale * (nobs / df_resid)
    else:
        h = np.einsum("ij,jk,ik->i", wexog, bread, wexog)
        one_minus_h = 1.0 - h
        singular = np.nonzero(one_minus_h <= _LEVERAGE_TOL)[0]
        if singular.size:
            raise np.linalg.LinAlgError(_leverage_message(kind, singular, h))
        if kind == "HC2":
            het_scale = wresid ** 2 / one_minus_h
        else:
            het_scale = (wresid / one_minus_h) ** 2
    xs = wexog * np.sqrt(het_scale)[:, None]
    return bread @ (xs.T @ xs) @ bread


def _leverage_message(kind: str, rows: np.ndarray, h: np.ndarray) -> str:
    """The diagnostic for a leverage-1 row under HC2 / HC3."""
    shown = rows[:8].tolist()
    tail = "" if rows.size <= 8 else f" (and {rows.size - 8} more)"
    return (
        f"{kind}: leverage is 1 at observation(s) {shown}{tail}, so the "
        "leave-one-out scaling u_i / (1 - h_ii) is 0 / 0 and every entry "
        "of the covariance matrix would be NaN. A row reaches leverage 1 "
        "when it is the only observation with a non-zero value on some "
        "regressor - an outlier dummy, a singleton panel unit, an "
        "event-study cell of size one - which is exactly the case where "
        "deleting it makes the design rank-deficient, so HC2 and HC3 are "
        "undefined there. statsmodels returns a finite number instead, but "
        "it is the ratio of two quantities of order 1e-16 and moves by "
        "whole factors between algebraically identical designs, so it is "
        "not reproduced. Use cov_type='HC0' or 'HC1' (neither divides by "
        "1 - h_ii), or drop the singleton row."
    )


def _cluster_meat(xu: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """``S = sum_g (sum_{i in g} x_i u_i)(...)'`` for integer-coded groups.

    Written here rather than borrowed: the three cluster meats already in
    puremacro (``lp/_panel_helpers.two_way_fe_within``,
    ``did/spatial_did._cluster_meat``, ``cross_nest._fit_ols``) are each
    welded to a different caller — a MultiIndex panel frame, an
    absorbed-FE parameter count, a univariate design — and none of them
    is reachable with a plain ``(y, X, groups)`` triple.
    """
    sums = np.zeros((n_groups, xu.shape[1]))
    np.add.at(sums, codes, xu)
    return sums.T @ sums


def _cluster_correction(n_groups: int, nobs: int, k_params: int) -> float:
    """``G/(G-1) * (n-1)/(n-k)`` — statsmodels' ``use_correction=True``.

    Raises ``ValueError`` for a single group rather than letting
    ``G/(G-1)`` come back as a bare ``ZeroDivisionError`` that names
    nothing. statsmodels divides by zero here too; the contract in
    CONTRIBUTING is stricter than that.
    """
    if n_groups < 2:
        raise ValueError(
            f"the cluster small-sample factor G/(G-1) is undefined with "
            f"{n_groups} cluster(s). Pass use_correction=False, or cluster "
            "on a variable that takes at least two values."
        )
    return (n_groups / (n_groups - 1.0)) * ((nobs - 1.0) / float(nobs - k_params))


def _cluster_cov(xu: np.ndarray, bread: np.ndarray, groups: np.ndarray,
                 use_correction: bool) -> tuple[np.ndarray, int]:
    """One-way cluster-robust covariance and the number of clusters."""
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    codes = np.ravel(codes)
    n_groups = int(codes.max()) + 1 if codes.size else 0
    cov = bread @ _cluster_meat(xu, codes, n_groups) @ bread
    if use_correction:
        cov = cov * _cluster_correction(n_groups, xu.shape[0], xu.shape[1])
    return cov, n_groups


def _cluster_cov_2way(xu: np.ndarray, bread: np.ndarray, groups: np.ndarray,
                      use_correction: bool) -> tuple[np.ndarray, tuple[int, int]]:
    """Two-way cluster: ``C_a + C_b - C_ab`` (Cameron-Gelbach-Miller).

    ``C_ab`` clusters on the intersection of the two group variables.
    Each of the three terms carries its own small-sample factor, exactly
    as ``sandwich_covariance.cov_cluster_2groups`` does.
    """
    g0 = np.asarray(groups)[:, 0]
    g1 = np.asarray(groups)[:, 1]
    cov0, n0 = _cluster_cov(xu, bread, g0, use_correction)
    cov1, n1 = _cluster_cov(xu, bread, g1, use_correction)
    # Intersection cluster: pair the two codes and re-factorise.
    _, c0 = np.unique(g0, return_inverse=True)
    _, c1 = np.unique(g1, return_inverse=True)
    joint = np.ravel(c0) * (int(np.ravel(c1).max()) + 1) + np.ravel(c1)
    cov01, _ = _cluster_cov(xu, bread, joint, use_correction)
    return cov0 + cov1 - cov01, (n0, n1)


def _groupidx_from_runs(keys: np.ndarray) -> list[tuple[int, int]]:
    """Start/end row indices of each contiguous run of equal ``keys``."""
    keys = np.asarray(keys)
    breaks = (np.nonzero(keys[:-1] != keys[1:])[0] + 1).tolist()
    return list(zip([0] + breaks, breaks + [len(keys)]))


def _groupidx_from_time(time: np.ndarray) -> list[tuple[int, int]]:
    """Panel-unit boundaries inferred from a resetting time index."""
    time = np.asarray(time)
    breaks = (np.nonzero(time[1:] < time[:-1])[0] + 1).tolist()
    return list(zip([0] + breaks, breaks + [len(time)]))


def _hac_panel_meat(xu: np.ndarray, weights: np.ndarray,
                    groupidx: list[tuple[int, int]]) -> np.ndarray:
    """HAC applied *within* each panel unit, then summed.

    Mirrors ``sandwich_covariance.S_nw_panel``: cross-products at lag
    ``l`` are taken only between rows of the same unit, so the estimator
    never straddles a unit boundary. ``weights`` is the lag-weight vector,
    Bartlett unless the caller asked for another kernel.
    """
    S = weights[0] * (xu.T @ xu)
    for lag in range(1, len(weights)):
        lead, lagged = [], []
        for lo, hi in groupidx:
            if lo + lag < hi:
                lead.append(xu[lo + lag:hi])
                lagged.append(xu[lo:hi - lag])
        if not lead:
            raise ValueError(
                f"hac-panel: every unit is shorter than lag {lag}; "
                "reduce maxlags or check the group ordering."
            )
        s = np.vstack(lead).T @ np.vstack(lagged)
        S = S + weights[lag] * (s + s.T)
    return S


def _groupsum_meat(xu: np.ndarray, time: np.ndarray, maxlags: int,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """Driscoll-Kraay meat: sum within period, then Newey-West over periods.

    Delegates to :func:`puremacro.inference.dk.driscoll_kraay`, which was
    measured against ``cov_nw_groupsum`` at 9.7e-17. Two wrinkles of
    statsmodels' ``group_sums`` are reproduced here rather than there:
    it bincounts over the raw integer codes, so *empty* periods enter the
    HAC series as rows of zeros; and it re-factorises by order of first
    appearance when the codes are very sparse.
    """
    time = np.asarray(time)
    if not np.issubdtype(time.dtype, np.integer):
        raise ValueError(
            "hac-groupsum: 'time' must be an integer period code "
            "(statsmodels bincounts it); pass pd.factorize(t)[0]."
        )
    if time.size and time.min() < 0:
        raise ValueError("hac-groupsum: 'time' must be non-negative.")
    if time.size and time.max() > 2 * xu.shape[0]:
        # statsmodels' group_sums relabels here to bound the bincount.
        time = pd.factorize(time)[0]
    codes = time
    present = np.unique(codes)
    n_periods = int(codes.max()) + 1 if codes.size else 0
    if present.size != n_periods:
        # Pad the empty periods with explicit zero rows so the Bartlett
        # lag alignment matches statsmodels' zero-filled bincount.
        missing = np.setdiff1d(np.arange(n_periods), present)
        xu = np.vstack([xu, np.zeros((missing.size, xu.shape[1]))])
        codes = np.concatenate([codes, missing])
    if weights is None:
        return driscoll_kraay(xu, codes, lags=maxlags)
    # Non-Bartlett kernel: driscoll_kraay hardcodes Bartlett, so sum
    # within period here and run the generic kernel over the period
    # series. Same grouping (sorted unique codes), same lag alignment.
    uniq, inv = np.unique(codes, return_inverse=True)
    by_t = np.zeros((uniq.size, xu.shape[1]))
    np.add.at(by_t, np.ravel(inv), xu)
    return _hac_meat(by_t, weights)


# ---------------------------------------------------------------------------
# restriction parsing
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _parse_restriction(r_matrix, names: tuple[str, ...] | None,
                       k: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a restriction specification to ``(R, q)``.

    Accepts an ``(q, k)`` array, a length-``k`` 1-D array, a ``(R, q)``
    tuple, or a small subset of statsmodels' string syntax (see
    :meth:`OLSResult.f_test`).
    """
    if isinstance(r_matrix, str):
        # Column labels keep their own dtype (they can be integers), but
        # a restriction string can only name them as text.
        labels = None if names is None else tuple(str(c) for c in names)
        return _parse_restriction_string(r_matrix, labels, k)
    if isinstance(r_matrix, tuple):
        if len(r_matrix) != 2:
            raise ValueError("a tuple restriction must be (R, q)")
        R = np.atleast_2d(np.asarray(r_matrix[0], dtype=float))
        q = np.atleast_1d(np.asarray(r_matrix[1], dtype=float)).ravel()
        if q.size == 1 and R.shape[0] > 1:
            q = np.repeat(q, R.shape[0])
        return _check_restriction(R, q, k)
    R = np.atleast_2d(np.asarray(r_matrix, dtype=float))
    return _check_restriction(R, np.zeros(R.shape[0]), k)


def _check_restriction(R: np.ndarray, q: np.ndarray, k: int):
    if R.ndim != 2 or R.shape[1] != k:
        raise ValueError(
            f"restriction matrix has shape {R.shape}, expected (q, {k})"
        )
    if q.shape[0] != R.shape[0]:
        raise ValueError("R and q must have the same number of rows")
    return R, q


def _parse_restriction_string(spec: str, names: tuple[str, ...] | None, k: int):
    """Parse ``'x1 = 0, x1 - 2*x2 = 3'`` into ``(R, q)``.

    Deliberately small: patsy's constraint grammar is much larger, so
    anything this does not recognise raises ``NotImplementedError``
    naming the array form rather than guessing.
    """
    if names is None:
        raise NotImplementedError(
            "string restrictions need named regressors (pass X as a "
            "DataFrame), or give the (q, k) restriction array instead."
        )
    rows, qs = [], []
    for part in spec.replace("==", "=").split(","):
        if not part.strip():
            continue
        if "=" in part:
            lhs, rhs = part.split("=", 1)
        else:
            lhs, rhs = part, "0"
        lrow, lconst = _parse_restriction_side(lhs, names, spec)
        rrow, rconst = _parse_restriction_side(rhs, names, spec)
        rows.append(lrow - rrow)
        qs.append(rconst - lconst)
    if not rows:
        raise NotImplementedError(f"could not read any restriction from {spec!r}")
    return _check_restriction(np.array(rows, dtype=float),
                              np.array(qs, dtype=float), k)


def _reject_numeric_label(token: str, spec: str) -> None:
    """Refuse a token that is both a number and a column label.

    Integer column labels (``pd.DataFrame(arr)``) survive into ``names``
    as ``'0'``, ``'1'``, ..., so ``"1 = 0"`` could mean "coefficient 1 is
    zero" or "coefficient 1 equals coefficient 0". statsmodels raises a
    ``TypeError`` on the same call; guessing would publish a plausible
    wrong F.
    """
    if _NUMBER.match(token):
        raise NotImplementedError(
            f"restriction {spec!r} is ambiguous: {token!r} is both a number "
            "and a column label of this design. Pass the (q, k) restriction "
            "array instead."
        )


def _parse_restriction_side(side: str, names: tuple[str, ...], spec: str):
    """One side of a constraint -> (coefficient row, constant)."""
    row = np.zeros(len(names))
    const = 0.0
    # Whole-side name match first: real designs carry column names with
    # operators inside them ("I(sigma_s ** 2)", "region_North-East"), and
    # tokenising those would either mangle them or refuse them.
    stripped = side.strip()
    if stripped in names:
        _reject_numeric_label(stripped, spec)
        row[names.index(stripped)] = 1.0
        return row, const
    if _NUMBER.match(stripped):
        return row, float(stripped)
    for term in re.findall(r"[+-]?[^+-]+", side):
        term = term.strip()
        if not term:
            continue
        sign = -1.0 if term.startswith("-") else 1.0
        term = term.lstrip("+-").strip()
        coef, var = 1.0, None
        for factor in term.split("*"):
            num, _, den = factor.partition("/")
            num, den = num.strip(), den.strip()
            for piece, invert in ((num, False), (den, True)):
                if not piece:
                    continue
                if _NUMBER.match(piece):
                    coef = coef / float(piece) if invert else coef * float(piece)
                elif piece in names and not invert:
                    _reject_numeric_label(piece, spec)
                    if var is not None:
                        raise NotImplementedError(
                            f"restriction {spec!r} multiplies two regressors; "
                            "pass the (q, k) restriction array instead."
                        )
                    var = piece
                else:
                    raise NotImplementedError(
                        f"cannot parse {piece!r} in restriction {spec!r}; "
                        "pass the (q, k) restriction array instead."
                    )
        if var is None:
            const += sign * coef
        else:
            row[names.index(var)] += sign * coef
    return row, const


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTestResult:
    """Result of :meth:`OLSResult.t_test` — one row per restriction.

    Attributes
    ----------
    effect : np.ndarray
        ``R b``, the estimated value of each linear combination.
    sd : np.ndarray
        ``sqrt(diag(R V R'))``.
    tvalue : np.ndarray
        ``(R b - q) / sd``. Also available as ``statistic``.
    pvalue : np.ndarray
        Two-sided, from ``t(df_denom)`` when ``use_t`` else the normal.
    df_denom : float
        ``df_resid_inference`` — ``G - 1`` under a cluster-family
        covariance, ``n - k`` otherwise.
    use_t : bool
        Which reference distribution was used.
    """

    effect: np.ndarray
    sd: np.ndarray
    tvalue: np.ndarray
    pvalue: np.ndarray
    df_denom: float
    use_t: bool

    @property
    def statistic(self) -> np.ndarray:
        """Alias of :attr:`tvalue`, as in statsmodels' ``ContrastResults``."""
        return self.tvalue

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Confidence interval for each ``effect``, shape ``(q, 2)``."""
        if self.use_t:
            crit = stats.t.ppf(1 - alpha / 2.0, self.df_denom)
        else:
            crit = stats.norm.ppf(1 - alpha / 2.0)
        return np.column_stack(
            (self.effect - crit * self.sd, self.effect + crit * self.sd)
        )

    def summary(self) -> str:
        lines = ["Test for constraints", f"{'':>4}{'coef':>12}{'std err':>12}"
                 f"{'t' if self.use_t else 'z':>10}{'P>|t|':>10}"]
        for i, (e, s, t, p) in enumerate(
            zip(np.atleast_1d(self.effect), np.atleast_1d(self.sd),
                np.atleast_1d(self.tvalue), np.atleast_1d(self.pvalue))
        ):
            lines.append(f"c{i:<3}{e:12.4f}{s:12.4f}{t:10.3f}{p:10.3f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FTestResult:
    """Result of :meth:`OLSResult.f_test` (a Wald test on the F scale).

    Attributes
    ----------
    fvalue : float
        ``(R b - q)' [R V R']^-1 (R b - q) / J``. Also ``statistic``.
    pvalue : float
        ``F(J, df_denom)`` upper tail.
    df_num : int
        Number of restrictions actually used, ``J``.
    df_denom : float
        ``df_resid_inference``.
    """

    fvalue: float
    pvalue: float
    df_num: int
    df_denom: float

    @property
    def statistic(self) -> float:
        """Alias of :attr:`fvalue`."""
        return self.fvalue

    def summary(self) -> str:
        return (
            f"F test: F={self.fvalue:.6g}, p={self.pvalue:.6g}, "
            f"df_denom={self.df_denom:g}, df_num={self.df_num:d}"
        )


@dataclass(frozen=True)
class PredictionResult:
    """Result of :meth:`OLSResult.get_prediction`.

    Attributes
    ----------
    predicted_mean : np.ndarray
        ``X_new b``.
    se_mean : np.ndarray
        ``sqrt(diag(X_new V X_new'))`` with ``V`` the *fitted* covariance,
        so a robust ``cov_type`` propagates into the prediction SE.
    se_obs : np.ndarray
        Adds the residual variance ``scale`` (divided by the prediction
        weights when given).
    df : float
        ``df_resid`` — note statsmodels uses this here, not
        ``df_resid_inference``.
    use_t : bool
    """

    predicted_mean: np.ndarray
    se_mean: np.ndarray
    se_obs: np.ndarray
    df: float
    use_t: bool

    def conf_int(self, alpha: float = 0.05, obs: bool = False) -> np.ndarray:
        """Interval for the mean (``obs=False``) or a new observation."""
        se = self.se_obs if obs else self.se_mean
        if self.use_t:
            crit = stats.t.ppf(1 - alpha / 2.0, self.df)
        else:
            crit = stats.norm.ppf(1 - alpha / 2.0)
        return np.column_stack(
            (self.predicted_mean - crit * se, self.predicted_mean + crit * se)
        )


@dataclass(frozen=True)
class OLSResult:
    """Fitted linear model, with statsmodels' attribute names.

    Constructed by :func:`ols`; never instantiated directly. Attribute
    names and types mirror ``statsmodels.regression.linear_model``'s
    ``RegressionResults`` so that a call site changes only its
    constructor line: ``params``, ``bse``, ``tvalues`` and ``pvalues``
    are pandas Series indexed by column name when ``X`` is a DataFrame
    and plain ndarrays otherwise.

    Attributes
    ----------
    params : np.ndarray or pd.Series
        Coefficients.
    cov : np.ndarray
        The ``(k, k)`` covariance of ``params`` actually used for
        inference; :meth:`cov_params` is the labelled accessor.
    resid : np.ndarray or pd.Series
        ``y - X b`` on the *unweighted* scale, as statsmodels' ``resid``.
    wresid : np.ndarray
        ``sqrt(w) (y - X b)``; equal to ``resid`` without weights. This is
        the residual that enters every sandwich and ``scale``.
    fittedvalues : np.ndarray or pd.Series
        ``X b``.
    nobs : float
        Sample size after any ``missing='drop'``.
    df_resid, df_model : float
        ``n - rank`` and ``rank - k_constant``.
    df_resid_inference : float
        Denominator df for t/F inference: ``G - 1`` for the cluster
        family, else ``df_resid``.
    scale : float
        ``wresid'wresid / df_resid``.
    llf, aic, bic : float
        Gaussian log-likelihood and the two information criteria, with
        statsmodels' parameter count (``rank``, scale not counted).
    rsquared, rsquared_adj : float
        Centered when a constant is present, uncentered otherwise.
    cov_type : str
        The normalised covariance name.
    use_t : bool
        Whether t or normal tails were used.
    k_params, k_constant : int
        Rank of the design, and 1 if it contains (or implies) a constant.
    n_groups : int, tuple or None
        Number of clusters / panel units, when the covariance has them.
    names : tuple or None
        Column labels when ``X`` was a DataFrame, in their original
        dtype (integer labels stay integers, so ``params[1]`` works).
    """

    params: np.ndarray | pd.Series
    cov: np.ndarray
    resid: np.ndarray | pd.Series
    wresid: np.ndarray
    fittedvalues: np.ndarray | pd.Series
    nobs: float
    df_resid: float
    df_model: float
    df_resid_inference: float
    scale: float
    llf: float
    rsquared: float
    rsquared_adj: float
    cov_type: str
    use_t: bool
    k_params: int
    k_constant: int
    n_groups: int | tuple[int, int] | None
    names: tuple | None
    exog: np.ndarray = field(repr=False)
    weights: np.ndarray | None = field(repr=False)

    # -- internals ---------------------------------------------------------

    def _wrap(self, arr: np.ndarray) -> np.ndarray | pd.Series:
        """Label a length-k vector when the design carried column labels."""
        if self.names is None:
            return arr
        return pd.Series(arr, index=list(self.names))

    @property
    def _params_arr(self) -> np.ndarray:
        return np.asarray(self.params, dtype=float)

    # -- inference ---------------------------------------------------------

    @property
    def bse(self) -> np.ndarray | pd.Series:
        """Standard errors, ``sqrt(diag(cov_params()))``."""
        return self._wrap(np.sqrt(np.diag(self.cov)))

    @property
    def tvalues(self) -> np.ndarray | pd.Series:
        """``params / bse``."""
        return self._wrap(self._params_arr / np.sqrt(np.diag(self.cov)))

    @property
    def pvalues(self) -> np.ndarray | pd.Series:
        """Two-sided p-values, on ``t(df_resid_inference)`` if ``use_t``."""
        t = self._params_arr / np.sqrt(np.diag(self.cov))
        if self.use_t:
            p = stats.t.sf(np.abs(t), self.df_resid_inference) * 2
        else:
            p = stats.norm.sf(np.abs(t)) * 2
        return self._wrap(p)

    @property
    def aic(self) -> float:
        """``-2 llf + 2 k``, with ``k = rank`` (scale not counted)."""
        return -2 * self.llf + 2 * self.k_params

    @property
    def bic(self) -> float:
        """``-2 llf + log(n) k``."""
        return -2 * self.llf + np.log(self.nobs) * self.k_params

    def cov_params(self) -> np.ndarray | pd.DataFrame:
        """Covariance of ``params``; a labelled DataFrame in the pandas case."""
        if self.names is None:
            return self.cov
        return pd.DataFrame(self.cov, index=list(self.names),
                            columns=list(self.names))

    def conf_int(self, alpha: float = 0.05) -> np.ndarray | pd.DataFrame:
        """Confidence intervals at level ``1 - alpha``.

        Uses the same distribution and the same ``df_resid_inference`` as
        :attr:`pvalues`.

        Returns
        -------
        np.ndarray of shape (k, 2), or a DataFrame with columns ``[0, 1]``
        when the design carried column names — matching what statsmodels'
        pandas wrapper returns.
        """
        se = np.sqrt(np.diag(self.cov))
        if self.use_t:
            crit = stats.t.ppf(1 - alpha / 2.0, self.df_resid_inference)
        else:
            crit = stats.norm.ppf(1 - alpha / 2.0)
        b = self._params_arr
        out = np.column_stack((b - crit * se, b + crit * se))
        if self.names is None:
            return out
        return pd.DataFrame(out, index=list(self.names), columns=[0, 1])

    def t_test(self, r_matrix) -> TTestResult:
        """Test ``R b = q`` restriction by restriction.

        Parameters
        ----------
        r_matrix : array_like, tuple or str
            An ``(q, k)`` array (or length-``k`` vector) testing ``R b = 0``,
            a ``(R, q)`` tuple for a non-zero null, or a restriction string
            such as ``"d_ln_pop = 1"`` / ``"x1 = x2, x3 = 0"`` when the
            design has column names.

        Returns
        -------
        TTestResult

        Raises
        ------
        NotImplementedError
            If a string restriction uses syntax this parser does not
            cover; the message names the ``(q, k)`` array form.

        Notes
        -----
        Values match ``RegressionResults.t_test``; shapes are tidier.
        For a single restriction statsmodels returns ``sd`` and
        ``tvalue`` as ``(1, 1)`` matrices and ``pvalue`` as a 0-d array,
        whereas every field here is a length-``q`` vector.
        """
        R, q = _parse_restriction(r_matrix, self.names, self.k_params)
        effect = R @ self._params_arr
        sd = np.sqrt(np.diag(R @ self.cov @ R.T))
        t = (effect - q) / sd
        if self.use_t:
            p = stats.t.sf(np.abs(t), self.df_resid_inference) * 2
        else:
            p = stats.norm.sf(np.abs(t)) * 2
        return TTestResult(effect=effect, sd=sd, tvalue=t, pvalue=p,
                           df_denom=float(self.df_resid_inference),
                           use_t=self.use_t)

    def f_test(self, r_matrix) -> FTestResult:
        """Joint Wald test of ``R b = q`` on the F scale.

        The denominator degrees of freedom are ``df_resid_inference``, so
        under ``cov_type='cluster'`` they are ``G - 1`` and not ``n - k``.
        Accepts the same restriction forms as :meth:`t_test`.

        Returns
        -------
        FTestResult

        Notes
        -----
        Matches ``RegressionResults.f_test``, including its use of a
        pseudo-inverse for ``R V R'`` and its reduction of ``J`` to the
        rank of that matrix when the restrictions are linearly dependent.
        """
        R, q = _parse_restriction(r_matrix, self.names, self.k_params)
        Rbq = R @ self._params_arr - q
        cov_c = R @ self.cov @ R.T
        J = R.shape[0]
        rank = int(np.linalg.matrix_rank(cov_c))
        if rank < J:
            # Linearly dependent restrictions: statsmodels reduces the
            # numerator df to the rank and warns. Silently keeping J
            # would deflate F by a factor of J/rank.
            warnings.warn(
                "covariance of constraints does not have full rank. The "
                f"number of constraints is {J}, but rank is {rank}",
                UserWarning,
                stacklevel=2,
            )
            J = rank
        F = float(Rbq @ np.linalg.pinv(cov_c) @ Rbq) / J
        p = float(stats.f.sf(F, J, self.df_resid_inference))
        return FTestResult(fvalue=F, pvalue=p, df_num=J,
                           df_denom=float(self.df_resid_inference))

    def get_prediction(self, exog=None, weights=None) -> PredictionResult:
        """Predicted mean and its standard error at ``exog``.

        Parameters
        ----------
        exog : array_like or DataFrame, optional
            New design rows. Defaults to the estimation design.
        weights : array_like, optional
            Prediction weights, used only for :attr:`PredictionResult.se_obs`
            (the mean prediction does not depend on them). Defaults to the
            estimation weights when ``exog`` is omitted.

        Returns
        -------
        PredictionResult
        """
        if exog is None:
            X = self.exog
            if weights is None:
                weights = self.weights
        else:
            X = np.asarray(exog, dtype=float)
            if X.ndim == 1:
                X = X[None, :] if self.k_params > 1 else X[:, None]
        mean = X @ self._params_arr
        var_mean = np.einsum("ij,jk,ik->i", X, self.cov, X)
        var_resid = np.full(X.shape[0], self.scale)
        if weights is not None:
            var_resid = var_resid / np.asarray(weights, dtype=float)
        return PredictionResult(
            predicted_mean=mean,
            se_mean=np.sqrt(var_mean),
            se_obs=np.sqrt(var_mean + var_resid),
            df=float(self.df_resid),
            use_t=self.use_t,
        )

    def summary(self) -> str:
        """Plain-text coefficient table.

        Deliberately not a clone of statsmodels' ``Summary`` object: it is
        a string, so ``print(res.summary())`` works but
        ``res.summary().tables[1]`` does not.
        """
        names = tuple(
            str(c) for c in
            (self.names or tuple(f"x{i}" for i in range(self.k_params)))
        )
        b = self._params_arr
        se = np.sqrt(np.diag(self.cov))
        t = b / se
        p = np.asarray(self.pvalues, dtype=float)
        ci = np.asarray(self.conf_int(), dtype=float)
        width = max(8, max(len(n) for n in names))
        head = (
            f"OLS regression ({self.cov_type}, "
            f"{'t' if self.use_t else 'z'} inference)\n"
            f"  n = {self.nobs:g}, df_resid = {self.df_resid:g}, "
            f"R^2 = {self.rsquared:.4f}, adj R^2 = {self.rsquared_adj:.4f}\n"
            f"  llf = {self.llf:.4f}, AIC = {self.aic:.4f}, BIC = {self.bic:.4f}\n"
        )
        cols = (f"{'':<{width}}{'coef':>12}{'std err':>12}"
                f"{'t' if self.use_t else 'z':>9}{'P>|t|':>9}"
                f"{'[0.025':>12}{'0.975]':>12}")
        rows = [
            f"{n:<{width}}{b[i]:12.4f}{se[i]:12.4f}{t[i]:9.3f}{p[i]:9.3f}"
            f"{ci[i, 0]:12.4f}{ci[i, 1]:12.4f}"
            for i, n in enumerate(names)
        ]
        return head + cols + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------

_HC_KINDS = ("HC0", "HC1", "HC2", "HC3")
_COV_KWDS_ALLOWED = {
    "nonrobust": set(),
    "HC0": set(), "HC1": set(), "HC2": set(), "HC3": set(),
    "HAC": {"maxlags", "use_correction", "kernel", "weights_func"},
    "cluster": {"groups", "use_correction", "df_correction"},
    "hac-panel": {"maxlags", "groups", "time", "use_correction",
                  "df_correction", "kernel", "weights_func"},
    "hac-groupsum": {"maxlags", "time", "use_correction", "df_correction",
                     "kernel", "weights_func"},
}


def _normalise_cov_type(cov_type: str) -> str:
    """Map a user string onto the canonical covariance name."""
    if not isinstance(cov_type, str):
        raise TypeError(f"cov_type must be a string, got {type(cov_type)!r}")
    lowered = cov_type.lower()
    if lowered == "nonrobust":
        return "nonrobust"
    if cov_type.upper() in _HC_KINDS:
        return cov_type.upper()
    if lowered == "hac":
        return "HAC"
    if lowered == "cluster":
        return "cluster"
    if lowered in ("hac-panel", "nw-panel"):
        return "hac-panel"
    if lowered in ("hac-groupsum", "nw-groupsum"):
        return "hac-groupsum"
    raise ValueError(
        f"cov_type {cov_type!r} not recognised. Available: 'nonrobust', "
        "'HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'cluster', 'hac-panel', "
        "'hac-groupsum'."
    )


def _panel_use_correction(value, cov_type: str) -> str | bool:
    """Validate the tri-state ``use_correction`` of the panel estimators."""
    if value is True:
        # statsmodels silently applies *no* correction here: True matches
        # neither the 'hac' nor the 'cluster' branch of cov_nw_panel /
        # cov_nw_groupsum. Refusing is the diagnostic version of that.
        raise ValueError(
            f"{cov_type}: use_correction must be one of False, 'hac', "
            "'cluster' — it is not a bool. statsmodels applies no "
            "correction at all when given True; pass False if that is "
            "what you meant."
        )
    if value not in (False, "hac", "cluster", "c"):
        raise ValueError(
            f"{cov_type}: use_correction must be False, 'hac' or 'cluster'; "
            f"got {value!r}"
        )
    return value


def _detect_k_constant(X: np.ndarray) -> int:
    """1 if the design contains, or implies, a constant column.

    Reproduces ``statsmodels.base.data._handle_constant``, including the
    *implicit* constant case (a full set of group dummies has no column
    of ones but spans one), because ``rsquared`` and ``df_model`` hang
    off this flag.
    """
    if X.size == 0:
        return 0
    col_max, col_min = X.max(axis=0), X.min(axis=0)
    const_idx = np.where(col_max == col_min)[0]
    if const_idx.size == 1:
        if X[:, const_idx[0]].mean() != 0:
            return 1
    elif const_idx.size > 1:
        means = [X[:, i].mean() for i in const_idx]
        if any(m == 1 for m in means):
            return 1
        if any(m != 0 for m in means):
            return 1
    # No usable constant column: look for one spanned by the design.
    augmented = np.column_stack((np.ones(X.shape[0]), X))
    return int(np.linalg.matrix_rank(X) == np.linalg.matrix_rank(augmented))


def _prepare(y, X, weights, missing):
    """Coerce inputs, apply ``missing``, and remember pandas labels."""
    names = None
    row_index = None
    x_index = None
    if isinstance(X, pd.DataFrame):
        # Column labels are kept in their own dtype. Stringifying them
        # would make ``m.params[1]`` a KeyError on a design built as
        # ``pd.DataFrame(design_matrix)``, where statsmodels indexes by
        # the integer label.
        names = tuple(X.columns)
        x_index = row_index = X.index
        X_arr = X.to_numpy(dtype=float)
    elif isinstance(X, pd.Series):
        names = (X.name,) if X.name is not None else None
        x_index = row_index = X.index
        X_arr = X.to_numpy(dtype=float)[:, None]
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got {X_arr.ndim} dimensions")

    if isinstance(y, (pd.Series, pd.DataFrame)):
        # Both sides labelled: the rows must line up. Everything after
        # this point is positional, so an unchecked mismatch pairs row i
        # of y with row i of X and returns a plausible wrong number
        # instead of an error - the failure mode that `.dropna()` /
        # `.loc[...]` subsetting produces. Message and test (both
        # indexes present, compared with `Index.equals`) are
        # statsmodels' `PandasData._check_integrity`.
        if x_index is not None and not y.index.equals(x_index):
            raise ValueError("The indices for endog and exog are not aligned")
        if row_index is None:
            row_index = y.index
        y_arr = np.asarray(y, dtype=float).squeeze()
    else:
        y_arr = np.asarray(y, dtype=float).squeeze()
    y_arr = np.atleast_1d(y_arr)
    if y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows but X has {X_arr.shape[0]}"
        )

    w_arr = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float).squeeze()
        if w_arr.ndim == 0:
            # statsmodels' WLS broadcasts a scalar weight over the sample
            # (`np.repeat(weights, len(endog))`). It leaves params and bse
            # exactly where OLS puts them and moves llf by 0.5 n log w.
            w_arr = np.repeat(float(w_arr), y_arr.shape[0])
        w_arr = np.atleast_1d(w_arr)
        if w_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"weights must have one entry per observation: got "
                f"{w_arr.shape[0]} for {y_arr.shape[0]} observations"
            )
        if np.any(w_arr[np.isfinite(w_arr)] < 0):
            raise ValueError("weights must be non-negative")

    if missing not in ("none", "drop", "raise"):
        raise ValueError("missing must be 'none', 'drop' or 'raise'")
    if missing != "none":
        bad = np.isnan(y_arr) | np.isnan(X_arr).any(axis=1)
        if w_arr is not None:
            bad = bad | np.isnan(w_arr)
        if bad.any():
            if missing == "raise":
                raise ValueError("NaNs were encountered in the data")
            keep = ~bad
            y_arr, X_arr = y_arr[keep], X_arr[keep]
            if w_arr is not None:
                w_arr = w_arr[keep]
            if row_index is not None:
                row_index = row_index[keep]

    # statsmodels checks the design for inf/NaN in `_handle_constant` and
    # raises MissingDataError even under missing='none' - it only lets a
    # NaN slide when it is in *endog*. Its own check is
    # `isfinite(exog.max(axis=0)).all()`, which a -inf slips past, so
    # there the SVD fails a moment later instead; either way statsmodels
    # refuses. Without this check inv_xtx does not save us:
    # np.linalg.cholesky accepts a NaN matrix and the pivot-ratio guard
    # is False on NaN, so the all-NaN "result" comes back with nobs and
    # df_resid attached and no exception anywhere.
    if X_arr.size and not np.isfinite(X_arr).all():
        bad = np.nonzero(~np.isfinite(X_arr).all(axis=1))[0]
        raise ValueError(
            f"ols: exog contains inf or nans - {bad.size} row(s) of X are "
            f"not finite (first: {bad[:5].tolist()}). Pass missing='drop' "
            "to delete them. statsmodels refuses the same input too, "
            "including under missing='none'."
        )
    return y_arr, X_arr, w_arr, names, row_index


def ols(y, X, *, weights=None, cov_type: str = "nonrobust",
        cov_kwds: dict | None = None, use_t: bool | None = None,
        missing: str = "none") -> OLSResult:
    """Fit ``y = X b + u`` with statsmodels-compatible standard errors.

    A drop-in for ``sm.OLS(y, X).fit(...)`` and ``sm.WLS(y, X,
    weights=w).fit(...)``: same covariance formulas, same small-sample
    factors, same ``use_t`` defaults, same degrees of freedom.

    Parameters
    ----------
    y : array_like or pandas.Series
        Dependent variable, length ``n``.
    X : array_like or pandas.DataFrame
        Design matrix ``(n, k)``. **The constant is not added for you** —
        call :func:`add_constant` first, exactly as with statsmodels.
        When ``X`` is a DataFrame, ``params`` / ``bse`` / ``tvalues`` /
        ``pvalues`` come back as Series indexed by column name, with the
        labels in their original dtype — integer columns stay integers,
        so ``m.params[1]`` works.
        When both ``y`` and ``X`` carry an index the two must be equal;
        an unchecked mismatch would pair row ``i`` of ``y`` with row
        ``i`` of ``X`` positionally.
    weights : array_like or float, optional
        Precision weights (variance proportional to ``1/w``), i.e. WLS.
        The fit and *every* covariance run on the whitened arrays
        ``sqrt(w) y`` and ``sqrt(w) X``. A scalar is broadcast over the
        sample, as ``sm.WLS`` does.
    cov_type : str, default 'nonrobust'
        One of ``'nonrobust'``, ``'HC0'``, ``'HC1'``, ``'HC2'``, ``'HC3'``,
        ``'HAC'``, ``'cluster'``, ``'hac-panel'``, ``'hac-groupsum'``
        (Driscoll-Kraay). Case-insensitive; ``'nw-panel'`` and
        ``'nw-groupsum'`` are accepted as statsmodels' old aliases.
    cov_kwds : dict, optional
        Options for the chosen covariance:

        ``maxlags`` : int
            Bartlett bandwidth for ``'HAC'``, ``'hac-panel'``,
            ``'hac-groupsum'``. Defaults to ``floor(4 (T/100)^(2/9))``.
            Must be a non-negative whole number: a negative bandwidth
            leaves the lag sum empty (HC0 wearing a HAC label) and a
            fractional one would be truncated, so both raise.
        ``kernel`` : {'bartlett', 'uniform'} or callable
            Lag weights for the HAC family, as ``weights_func`` under its
            other statsmodels spelling; ``kernel`` wins when both are
            given. Defaults to Bartlett. Ignored by ``'hac-panel'`` at
            ``maxlags=0``, where statsmodels hardcodes ``[1, 0]``.
        ``groups`` : array_like
            Cluster labels for ``'cluster'`` (one column, or two for
            two-way clustering ``C_a + C_b - C_ab``); unit labels for
            ``'hac-panel'``.
        ``time`` : array_like
            Integer period codes for ``'hac-groupsum'``; for
            ``'hac-panel'`` an alternative to ``groups``, where a *fall*
            in the time index marks a new unit.
        ``use_correction`` : bool or {False, 'hac', 'cluster'}
            Small-sample factor. Defaults: ``False`` for ``'HAC'``,
            ``True`` for ``'cluster'``, ``'hac'`` for ``'hac-panel'``,
            ``'cluster'`` for ``'hac-groupsum'``. For the two panel types
            it is *not* a bool.
        ``df_correction`` : bool
            Pass ``False`` to keep ``df_resid_inference = n - k`` under a
            cluster-family covariance instead of ``G - 1``.
        ``use_t`` : bool
            Accepted for every robust ``cov_type`` because
            ``RegressionResults.__init__`` pops it out of ``cov_kwds``;
            the ``use_t`` argument below wins when both are given.
    use_t : bool, optional
        Reference distribution for p-values and confidence intervals.
        Default ``True`` for ``'nonrobust'`` and ``False`` for every
        robust covariance — statsmodels' behaviour, and a common source
        of silently moving p-values.
    missing : {'none', 'drop', 'raise'}, default 'none'
        ``'drop'`` removes rows with a NaN in ``y``, ``X`` or ``weights``
        and reports ``nobs`` after the drop.

    Returns
    -------
    OLSResult

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``X'X`` is singular or ill-conditioned *after* the columns are
        scaled to unit norm, so the message is about collinearity and
        never about units. It names the columns most aligned with the
        null space (via :func:`puremacro._linalg.inv_xtx`). statsmodels
        instead returns a minimum-norm pinv solution here, so a
        rank-deficient design that statsmodels tolerated will now fail
        loudly. Also raised by ``'HC2'`` / ``'HC3'`` when some row has
        leverage 1 (departure 6 below).
    ValueError
        For an unknown ``cov_type``, a missing or misspelled ``cov_kwds``
        key, a non-integer ``time`` under ``'hac-groupsum'``, or
        ``use_correction=True`` on a panel covariance where statsmodels
        would silently apply nothing. Also for the inputs that would
        otherwise produce a number nobody could act on: a ``y`` and an
        ``X`` whose pandas indexes disagree, an ``X`` holding inf or NaN
        under ``missing='none'``, a negative or fractional ``maxlags``, a
        ``groups`` / ``time`` vector of the wrong length, a single
        cluster, and a saturated design (``n == rank``).

    See Also
    --------
    puremacro.inference._ols_helpers.ols_hac : the HAC kernel reused here.
    puremacro.inference.dk.driscoll_kraay : the Driscoll-Kraay meat.

    Notes
    -----
    Parity, measured against statsmodels 0.14.6 in
    ``tests/test_regress/test_ols_parity.py``: on the designs in that
    file every covariance type agrees on ``params``, ``bse``,
    ``tvalues``, ``pvalues``, ``conf_int``, ``rsquared``, ``aic`` and
    ``bic`` to ``atol=1e-10``, and on ``f_test`` / ``t_test`` to
    ``atol=1e-8``.

    **That 1e-10 is a measurement on those designs, not a guarantee.**
    statsmodels pseudo-inverts ``X`` directly, while this module forms
    and factors ``X'X`` (through ``inv_xtx``, which is what buys the
    collinearity diagnostic), and squaring the matrix squares its
    condition number. The disagreement therefore tracks
    ``cond(X'X) x eps``. Measured on ``[1, x1, x2]`` with
    ``corr(x1, x2) = rho``, ``n = 200``, ``cov_type='HC1'``, worst of six
    seeds, as the largest absolute ``bse`` difference:

    ====================  ==============  ==========
    ``corr(x1, x2)``      ``cond(X'X)``   max |dbse|
    ====================  ==============  ==========
    0.9                   2.1e+01         1.0e-15
    0.99                  2.2e+02         4.9e-14
    0.999                 2.2e+03         2.0e-12
    0.9999                2.2e+04         5.3e-11
    0.99999               2.2e+05         1.9e-09
    ====================  ==============  ==========

    So the 1e-10 figure holds to roughly ``cond(X'X) = 1e4`` and decays
    by a decade or two per decade of conditioning after that; ``params``
    and ``conf_int`` degrade on the same schedule. Column scaling is
    *not* part of this: the design is equilibrated to unit column norms
    before the factorisation, so putting a regressor in persons rather
    than millions changes nothing (neither the numbers nor whether the
    fit is accepted).

    Six deliberate departures. The first five change no number on a
    well-posed problem; the sixth is a case where no correct number
    exists:

    1. A singular design raises instead of being min-normed (above).
    2. Unknown ``cov_kwds`` keys raise. statsmodels documents that it
       does not check them, so ``{'maxlag': 4}`` there silently becomes
       the default bandwidth.
    3. ``use_correction=True`` is refused for ``'hac-panel'`` /
       ``'hac-groupsum'``, where statsmodels quietly applies no
       correction at all.
    4. ``summary()`` returns a string, not a ``Summary`` object.
    5. ``cov_kwds['maxlags']`` may be omitted for ``'HAC'``, where it
       falls back to ``floor(4 (T/100)^(2/9))``. statsmodels raises
       ``KeyError`` when the key is absent but uses that same rule when
       it is present and ``None``, so no working call changes.
    6. ``'HC2'`` / ``'HC3'`` raise on a row with leverage exactly 1
       instead of returning a number. Such a row is perfectly fitted, so
       the leave-one-out scaling ``u_i / (1 - h_ii)`` is ``0 / 0``:
       statsmodels' value there is the quotient of two quantities of
       order 1e-16 and moves by whole factors — 0.34 against 11.0 on one
       draw of an unbalanced panel with a singleton unit, 4.60 against
       0.54 on the next — so there is nothing to reproduce. HC0 and HC1
       are unaffected and agree with statsmodels to 1.6e-15 on the same
       design.

    Pandas inputs are aligned by *label* only in the sense that a
    disagreement is refused: when both ``y`` and ``X`` carry an index
    the two must be equal (``Index.equals``, statsmodels'
    ``PandasData._check_integrity``), and everything after that check is
    positional. Nothing is silently reindexed.

    One container difference is worth naming because it is visible
    rather than numerical: statsmodels returns ``params`` as a Series
    whenever *any* input is pandas, inventing the names ``const``,
    ``x1``, ``x2`` when ``X`` is a bare ndarray. Here the rule is
    ``X``-driven — Series if and only if ``X`` carried column names —
    so a pandas ``y`` with an ndarray ``X`` gives ndarrays rather than a
    Series indexed by invented labels. ``resid`` and ``fittedvalues``
    still take their index from whichever input had one.

    ``'hac-groupsum'`` keeps statsmodels' two-group-count quirk: the
    variance correction divides by the number of *time periods* while
    ``df_resid_inference`` counts *panel units* (the number of times the
    time index falls). It only bites when ``use_t=True``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({"x": rng.standard_normal(80)})
    >>> X = add_constant(df)
    >>> y = 2.0 + 0.5 * df["x"] + rng.standard_normal(80)
    >>> res = ols(y, X, cov_type="HAC", cov_kwds={"maxlags": 3})
    >>> round(float(res.params["x"]), 3)
    0.489
    >>> res.cov_type, res.use_t
    ('HAC', False)
    """
    y_arr, X_arr, w_arr, names, row_index = _prepare(y, X, weights, missing)
    n, k = X_arr.shape
    cov_type = _normalise_cov_type(cov_type)
    kwds = dict(cov_kwds or {})
    if cov_type != "nonrobust" and "use_t" in kwds:
        # Not a misspelled key: RegressionResults.__init__ pops 'use_t'
        # out of cov_kwds and lets an explicit keyword win over it. The
        # 'nonrobust' branch never reads cov_kwds at all, so there the
        # key stays refused rather than being honoured where statsmodels
        # would ignore it.
        use_t_kwd = kwds.pop("use_t")
        if use_t is None:
            use_t = use_t_kwd
    unknown = set(kwds) - _COV_KWDS_ALLOWED[cov_type]
    if unknown:
        raise ValueError(
            f"cov_type {cov_type!r} does not use cov_kwds "
            f"{sorted(unknown)}; accepted keys: "
            f"{sorted(_COV_KWDS_ALLOWED[cov_type]) or 'none'}"
        )

    # WLS is OLS on the whitened data; with no weights the whitened and
    # raw arrays are the same object-value, so one code path serves both.
    if w_arr is None:
        wexog, wendog = X_arr, y_arr
    else:
        sqrt_w = np.sqrt(w_arr)
        wexog, wendog = X_arr * sqrt_w[:, None], y_arr * sqrt_w

    # Column equilibration, and it is load-bearing rather than cosmetic.
    # inv_xtx gates on the ratio of smallest to largest Cholesky pivot of
    # X'X, and those pivots carry the column norms, so an un-equilibrated
    # design is refused for having a regressor in levels (population in
    # persons beside an intercept) rather than for being collinear:
    # cond(X'X) = 1.1e12 was refused while a year polynomial at 1.9e20 was
    # accepted. Scaling each column to unit norm makes the gate
    # scale-free - multiplying a regressor by 1e6 can no longer decide
    # whether the fit is accepted - and it lowers the condition number
    # that forming the normal equations then squares. Everything below
    # runs in the scaled space; the covariance is scaled back once, at the
    # end, and beta_s / col_norm is the same estimator to machine
    # precision.
    col_norm = np.sqrt(np.einsum("ij,ij->j", wexog, wexog))
    col_norm = np.where(col_norm > 0.0, col_norm, 1.0)
    sexog = wexog / col_norm
    try:
        bread = inv_xtx(sexog, name="ols")
    except np.linalg.LinAlgError as exc:
        if "rescaling" in str(exc):
            # inv_xtx's advice ("try rescaling regressors") is already
            # spent: the columns it saw were unit-norm. Say what is
            # actually wrong instead.
            raise np.linalg.LinAlgError(
                f"ols: X'X is numerically singular at full rank {k} even "
                "after the columns were scaled to unit norm "
                f"(cond(X'X) ~ {np.linalg.cond(sexog.T @ sexog):.2e}). The "
                "regressors are near-collinear on a scale-free measure, so "
                "rescaling will not help: drop one, or orthogonalise them "
                "(centre a time trend before taking its powers)."
            ) from exc
        raise
    # Left-to-right, matching inference._ols_helpers.ols_hac, so that the
    # HAC branch's reused vcov belongs to exactly these coefficients.
    beta_s = bread @ sexog.T @ wendog
    wresid = wendog - sexog @ beta_s
    beta = beta_s / col_norm
    fitted = X_arr @ beta
    resid = y_arr - fitted

    k_constant = _detect_k_constant(X_arr)
    rank = k  # inv_xtx already refused anything rank-deficient
    df_resid = float(n - rank)
    if df_resid <= 0:
        raise ValueError(
            f"ols: the design is saturated (n = {n}, rank = {rank}), so "
            "df_resid is 0 and scale, bse, every p-value and R^2_adj are "
            "undefined. statsmodels returns scale=inf and bse=[inf, ...] "
            "here rather than saying so."
        )
    df_model = float(rank - k_constant)
    ssr = float(wresid @ wresid)
    scale = ssr / df_resid

    cov, n_groups, df_resid_inference = _covariance(
        cov_type, kwds, sexog, wendog, wresid, bread, scale, n, rank, df_resid
    )
    # Undo the equilibration: with X = X_s diag(c), (X'X)^-1 = diag(1/c)
    # (X_s'X_s)^-1 diag(1/c), and every sandwich inherits the same
    # two-sided scaling because its meat is built from X_s.
    cov = cov / np.outer(col_norm, col_norm)

    if use_t is None:
        use_t = cov_type == "nonrobust"
    use_t = bool(use_t)

    # R^2 is centered only when a constant is present; for WLS the
    # centering is weighted (statsmodels' centered_tss).
    if k_constant:
        if w_arr is None:
            centered = y_arr - y_arr.mean()
            tss = float(centered @ centered)
        else:
            mean = float(np.average(y_arr, weights=w_arr))
            tss = float(np.sum(w_arr * (y_arr - mean) ** 2))
    else:
        tss = float(wendog @ wendog)
    rsquared = 1.0 - ssr / tss
    rsquared_adj = 1.0 - (n - k_constant) / df_resid * (1.0 - rsquared)

    # The two branches are algebraically identical, but they are written
    # here exactly as OLS.loglike and WLS.loglike group their terms, so
    # llf (and hence aic / bic) is bit-identical rather than merely equal
    # to 13 digits.
    nobs2 = n / 2.0
    if w_arr is None:
        llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / n) - nobs2
    else:
        llf = -np.log(ssr) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        llf += 0.5 * float(np.sum(np.log(w_arr)))

    if names is not None:
        params_out: np.ndarray | pd.Series = pd.Series(beta, index=list(names))
    else:
        params_out = beta
    if row_index is not None:
        resid_out: np.ndarray | pd.Series = pd.Series(resid, index=row_index)
        fitted_out: np.ndarray | pd.Series = pd.Series(fitted, index=row_index)
    else:
        resid_out, fitted_out = resid, fitted

    return OLSResult(
        params=params_out,
        cov=cov,
        resid=resid_out,
        wresid=wresid,
        fittedvalues=fitted_out,
        nobs=float(n),
        df_resid=df_resid,
        df_model=df_model,
        df_resid_inference=df_resid_inference,
        scale=scale,
        llf=float(llf),
        rsquared=float(rsquared),
        rsquared_adj=float(rsquared_adj),
        cov_type=cov_type,
        use_t=use_t,
        k_params=int(rank),
        k_constant=int(k_constant),
        n_groups=n_groups,
        names=names,
        exog=X_arr,
        weights=w_arr,
    )


def _covariance(cov_type, kwds, wexog, wendog, wresid, bread, scale, n, k,
                df_resid):
    """Dispatch to the requested covariance.

    Returns ``(cov, n_groups, df_resid_inference)``. ``n_groups`` is
    ``None`` whenever the estimator has no group dimension.

    ``wexog`` and ``bread`` arrive column-equilibrated, so the covariance
    returned here lives in the scaled space; :func:`ols` undoes that once.
    """
    if cov_type == "nonrobust":
        return scale * bread, None, df_resid

    if cov_type in _HC_KINDS:
        cov = _hc_cov(wexog, wresid, bread, cov_type, float(n), df_resid)
        return cov, None, df_resid

    # xu is the score: weighted exog times weighted residual. Every
    # sandwich below is (X'X)^-1 S (X'X)^-1 with S built from it.
    xu = wexog * wresid[:, None]

    if cov_type == "HAC":
        maxlags = kwds.get("maxlags", None)
        if maxlags is None:
            maxlags = _default_maxlags(n)
        maxlags = _check_maxlags(maxlags, cov_type)
        weights = _hac_weights(kwds, maxlags, cov_type)
        if weights is None:
            # Reuse the package's verified HAC kernel rather than
            # re-deriving it (measured against statsmodels at 1.1e-15,
            # gapprobe §3.3). ols_hac refits on (wendog, wexog) with the
            # same expression used for `beta` above, so its residuals are
            # bit-identical to wresid and the vcov it returns belongs to
            # the coefficients we report.
            cov = ols_hac(wendog, wexog, lags=maxlags)["vcov"]
        else:
            cov = bread @ _hac_meat(xu, weights) @ bread
        if kwds.get("use_correction", False):
            cov = cov * (n / float(n - k))
        return cov, None, df_resid

    if cov_type == "cluster":
        if "groups" not in kwds:
            raise ValueError("cov_type='cluster' requires cov_kwds['groups']")
        groups = kwds["groups"]
        if not hasattr(groups, "shape"):
            # statsmodels transposes a bare sequence here, so
            # groups=[unit, time] is a *two-way* cluster rather than a
            # (2, n) design (base/covtype.py, the 'cluster' branch).
            groups = np.asarray(groups).T
        if hasattr(groups, "to_numpy"):
            groups = groups.to_numpy()
        groups = np.asarray(groups)
        if groups.ndim == 2 and groups.shape[1] == 1:
            groups = groups[:, 0]
        if groups.shape[0] != n:
            raise ValueError(
                "cov_type='cluster': cov_kwds['groups'] has "
                f"{groups.shape[0]} entries but the regression has {n} "
                "observations. Subset the labels the same way the design "
                "was subset."
            )
        use_correction = kwds.get("use_correction", True)
        if groups.ndim == 1:
            cov, n_groups = _cluster_cov(xu, bread, groups, use_correction)
            g_for_df = n_groups
        elif groups.ndim == 2 and groups.shape[1] == 2:
            cov, n_groups = _cluster_cov_2way(xu, bread, groups, use_correction)
            g_for_df = min(n_groups)
        else:
            raise ValueError(
                "cov_type='cluster' accepts groups with one or two columns; "
                f"got shape {groups.shape}"
            )
        df_inf = df_resid
        if kwds.get("df_correction", None) is not False:
            if g_for_df < 2:
                raise ValueError(
                    "cov_type='cluster': df_resid_inference is G - 1 = 0 "
                    f"with {g_for_df} cluster(s), so every t statistic and "
                    "confidence interval is undefined. Pass "
                    "df_correction=False to keep n - k, or cluster on a "
                    "variable that takes at least two values."
                )
            df_inf = float(g_for_df - 1)
        return cov, n_groups, df_inf

    if cov_type == "hac-panel":
        if "maxlags" not in kwds:
            raise ValueError("cov_type='hac-panel' requires cov_kwds['maxlags']")
        maxlags = _check_maxlags(kwds["maxlags"], cov_type)
        groups, time = kwds.get("groups", None), kwds.get("time", None)
        if groups is not None:
            groupidx = _groupidx_from_runs(
                _panel_key(groups, n, cov_type, "groups")
            )
        elif time is not None:
            groupidx = _groupidx_from_time(
                _panel_key(time, n, cov_type, "time")
            )
        else:
            raise ValueError(
                "cov_type='hac-panel' needs either cov_kwds['groups'] or "
                "cov_kwds['time']"
            )
        use_correction = _panel_use_correction(
            kwds.get("use_correction", "hac"), "hac-panel"
        )
        if maxlags == 0:
            # cov_nw_panel hardcodes [1, 0] at nlags=0 "to avoid the
            # scalar check in hac_nw", which also means the kernel choice
            # is ignored there. Reproduced rather than tidied.
            weights = np.array([1.0, 0.0])
        else:
            weights = _hac_weights(kwds, maxlags, cov_type)
            if weights is None:
                weights = 1.0 - np.arange(maxlags + 1) / (maxlags + 1.0)
        cov = bread @ _hac_panel_meat(xu, weights, groupidx) @ bread
        n_groups = len(groupidx)
        if use_correction == "hac":
            cov = cov * (n / float(n - k))
        elif use_correction in ("cluster", "c"):
            cov = cov * _cluster_correction(n_groups, n, k)
        df_inf = df_resid
        if kwds.get("df_correction", None) is not False:
            df_inf = float(n_groups - 1)
        return cov, n_groups, df_inf

    # hac-groupsum (Driscoll-Kraay)
    if "maxlags" not in kwds:
        raise ValueError("cov_type='hac-groupsum' requires cov_kwds['maxlags']")
    if "time" not in kwds:
        raise ValueError("cov_type='hac-groupsum' requires cov_kwds['time']")
    maxlags = _check_maxlags(kwds["maxlags"], cov_type)
    time = _panel_key(kwds["time"], n, cov_type, "time")
    use_correction = _panel_use_correction(
        kwds.get("use_correction", "cluster"), "hac-groupsum"
    )
    weights = _hac_weights(kwds, maxlags, cov_type)
    cov = bread @ _groupsum_meat(xu, time, maxlags, weights) @ bread
    if use_correction == "hac":
        cov = cov * (n / float(n - k))
    elif use_correction in ("cluster", "c"):
        # Note: the correction counts PERIODS ...
        cov = cov * _cluster_correction(len(np.unique(time)), n, k)
    # ... while the inference df counts UNITS. Two group counts, one
    # estimator; this is statsmodels' behaviour, verified in gapprobe §3.5.
    n_units = int(len(np.nonzero(time[1:] < time[:-1])[0]) + 1)
    df_inf = df_resid
    if kwds.get("df_correction", None) is not False:
        df_inf = float(n_units - 1)
    return cov, n_units, df_inf
