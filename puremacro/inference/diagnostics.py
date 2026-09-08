"""Residual diagnostics for a fitted regression.

Currently the Durbin-Watson statistic, a pure-numpy drop-in for
``statsmodels.stats.stattools.durbin_watson``.

References
----------
Durbin, J. and Watson, G. S. (1950, 1951). Testing for serial correlation in
    least squares regression, I and II. Biometrika 37(3-4), 409-428 and
    38(1-2), 159-177.
"""
from __future__ import annotations

import numpy as np

__all__ = ["durbin_watson"]


def durbin_watson(resids, axis=0):
    r"""Durbin-Watson statistic for first-order residual autocorrelation.

    .. math::

       DW = \frac{\sum_{t=2}^{T} (e_t - e_{t-1})^2}{\sum_{t=1}^{T} e_t^2}

    Numerically identical to ``statsmodels.stats.stattools.durbin_watson``
    (statsmodels 0.14.6, ``statsmodels/stats/stattools.py:14``).

    Parameters
    ----------
    resids : array_like
        Residuals, usually from a fitted regression. May be multi-
        dimensional; the statistic is then computed along ``axis``. A pandas
        Series or DataFrame is accepted and converted with ``np.asarray``,
        exactly as statsmodels does, so the return value is never a pandas
        object.
    axis : int, default 0
        Axis along which the residuals are ordered in time.

    Returns
    -------
    float or ndarray
        The statistic. A float when the reduction leaves a scalar (the usual
        1-D case), otherwise an ndarray with ``axis`` removed.

    Raises
    ------
    ValueError
        If ``resids`` has fewer than two observations along ``axis``. With
        one observation the differenced series is empty and statsmodels
        returns ``0.0`` — a value indistinguishable from "extreme positive
        autocorrelation". Raising instead is the one deliberate divergence
        from statsmodels here, and it concerns only a degenerate call.
    numpy.exceptions.AxisError
        If ``axis`` is out of bounds for ``resids``. Raised by ``np.diff``,
        which is where statsmodels raises it too, so the type and the
        message match.

    Notes
    -----
    ``DW ≈ 2(1 - r₁)`` where ``r₁`` is the first-order residual
    autocorrelation, so it lives in ``[0, 4]``: 2 is no serial correlation,
    near 0 is strong positive, near 4 is strong negative. It has no
    denominator adjustment and no p-value; the null distribution depends on
    the design matrix, which is why statsmodels does not return one and
    neither does this function. For a test with a p-value use a
    Breusch-Godfrey or Ljung-Box statistic.

    There is **no** small-sample correction and no demeaning: the numerator
    starts at ``t = 2`` and the denominator at ``t = 1``, so the statistic is
    biased towards 2 in short samples by construction.

    A residual vector that is identically zero gives ``0 / 0``. statsmodels
    returns ``nan`` with a numpy warning and so does this function — the
    condition means the regression fit exactly, which is a message about the
    design, not about serial correlation.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> white = rng.normal(size=5000)
    >>> round(durbin_watson(white), 1)          # no serial correlation
    2.0
    >>> ar1 = np.empty(5000)
    >>> ar1[0] = white[0]
    >>> for t in range(1, 5000):
    ...     ar1[t] = 0.9 * ar1[t - 1] + white[t]
    >>> bool(durbin_watson(ar1) < 0.5)          # strong positive
    True
    >>> alt = np.array([1.0, -1.0] * 2500)
    >>> bool(durbin_watson(alt) > 3.5)          # strong negative
    True

    Column-wise on a 2-D array of residuals:

    >>> panel = np.column_stack([white, ar1])
    >>> durbin_watson(panel).shape
    (2,)
    """
    resids = np.asarray(resids)

    # Difference first, length-check second. Reversing the two would mean
    # ``resids.shape[axis]`` validates the axis, and a tuple index raises a
    # bare ``IndexError("tuple index out of range")`` that names neither the
    # axis nor the array. ``np.diff`` normalises the axis itself and raises
    # numpy's ``AxisError``, which is both informative and exactly what
    # statsmodels raises here (it reaches ``np.diff`` before touching the
    # shape). Doing it this way rather than importing
    # ``normalize_axis_index`` also survives that helper's move between
    # ``numpy.core.multiarray`` and ``numpy.lib.array_utils`` in numpy 2.
    diff_resids = np.diff(resids, 1, axis=axis)

    n = resids.shape[axis]
    if n < 2:
        raise ValueError(
            f"durbin_watson: need at least 2 observations along axis "
            f"{axis}, got {n}. With one observation the differenced series "
            "is empty and the statistic would come back as a spurious 0.0."
        )

    dw = np.sum(diff_resids ** 2, axis=axis) / np.sum(resids ** 2, axis=axis)
    if np.ndim(dw) == 0:
        # statsmodels returns the ``np.float64`` that ``np.sum`` produced;
        # this narrows it to a builtin float, which is JSON-serialisable and
        # reprs as ``1.97`` rather than ``np.float64(1.97)`` under numpy 2.
        # The narrowing is invisible to ``isinstance(x, float)`` — np.float64
        # subclasses float — so only ``type(x) is float`` can test it, which
        # is what ``test_returns_float_for_one_dimensional_input`` does.
        return float(dw)
    return dw
