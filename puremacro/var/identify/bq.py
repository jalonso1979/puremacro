"""Blanchard-Quah (1989) long-run identification extended to n-variable VAR.

Imposes that only the 'permanent' shock (indicated by `permanent_var_idx`) has
non-zero long-run effect on the variable at that index; all other shocks are
restricted to have zero long-run effect on that variable.

The VAR is run on a stationary vector. Variables that enter in first
differences (output growth in Blanchard-Quah, productivity and hours growth
in Gali 1999) are reported as level responses by cumulating their IRFs over
the horizon; variables that enter in levels (the unemployment rate in
Blanchard-Quah) must not be cumulated. ``bq_svar(..., cumulate=...)`` says
which rows to cumulate.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.random import default_rng

from ..._linalg import safe_cholesky
from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import BQSVARResult

# Same thresholds as cholesky_svar — see var/identify/cholesky.py for
# rationale on the stricter bootstrap conditioning bar.
_BOOT_FAIL_WARN_THRESHOLD = 0.05
_BOOT_COND_RATIO_FLOOR = 1e-4


def _bq_impact(A_list, Sigma, permanent_var_idx=0):
    n = Sigma.shape[0]
    try:
        Psi_1 = np.linalg.inv(np.eye(n) - sum(A_list))
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "bq_svar: long-run matrix (I - A(1)) is singular — the VAR has "
            "a (near-)unit root or collinear variables, so the long-run "
            "restriction cannot be imposed on this sample"
        ) from exc
    # Permute so that permanent variable comes first.
    perm = [permanent_var_idx] + [i for i in range(n) if i != permanent_var_idx]
    P = np.eye(n)[perm]
    Psi_1_p = P @ Psi_1 @ P.T
    Sigma_p = P @ Sigma @ P.T
    Omega = Psi_1_p @ Sigma_p @ Psi_1_p.T
    L = safe_cholesky(Omega, name="bq_svar Ω")
    B_p = np.linalg.solve(Psi_1_p, L)
    # Un-permute back to original variable order.
    return P.T @ B_p @ P


def _cumulate_mask(cumulate, n: int) -> np.ndarray:
    """Boolean mask of the response variables (axis 1) to cumulate.

    ``True``/``False`` select all or none; a length-``n`` sequence of bools
    is a mask; any other sequence lists 0-based variable indices.
    """
    if isinstance(cumulate, (bool, np.bool_)):
        return np.full(n, bool(cumulate))
    if isinstance(cumulate, (str, bytes)) or not isinstance(cumulate, (Sequence, np.ndarray)):
        raise TypeError(
            "bq_svar: cumulate must be True, False, a sequence of variable "
            f"indices or a boolean mask of length n={n}; got {cumulate!r}"
        )
    arr = np.asarray(cumulate)
    if arr.ndim != 1:
        raise ValueError(f"bq_svar: cumulate must be one-dimensional; got shape {arr.shape}")
    if arr.size == 0:
        return np.zeros(n, dtype=bool)
    if arr.dtype == bool:
        if arr.size != n:
            raise ValueError(
                f"bq_svar: boolean cumulate mask has length {arr.size}, expected n={n}"
            )
        return arr.copy()
    if arr.dtype.kind not in "iu":
        raise TypeError(
            f"bq_svar: cumulate indices must be integers; got dtype {arr.dtype}"
        )
    if arr.min() < 0 or arr.max() >= n:
        raise ValueError(
            f"bq_svar: cumulate indices {arr.tolist()} out of range for n={n} variables"
        )
    mask = np.zeros(n, dtype=bool)
    mask[arr] = True
    return mask


def _cumulate_rows(irf: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cumulate ``irf[:, i, :]`` over the horizon for the variables ``i`` in
    ``mask``; leave the other rows as raw responses."""
    if mask.all():
        return np.cumsum(irf, axis=0)
    out = np.array(irf, dtype=float, copy=True)
    if mask.any():
        out[:, mask, :] = np.cumsum(irf[:, mask, :], axis=0)
    return out


def bq_svar(
    Y: np.ndarray,
    *,
    p: int | None = None,
    horizon: int = 20,
    permanent_var_idx: int = 0,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
    lags: int | None = None,
    cumulate: bool | Sequence[int] | Sequence[bool] = True,
) -> BQSVARResult:
    """Blanchard-Quah (1989) SVAR identified by a long-run restriction.

    Fits a VAR(p) with constant to the stationary vector ``Y`` (columns in
    the order you pass them), computes the long-run multiplier
    ``Psi(1) = (I - A_1 - ... - A_p)^-1`` and chooses the impact matrix
    ``B`` (``B B' = Sigma``) so that ``Psi(1) B`` is lower triangular after
    moving ``permanent_var_idx`` to the front: only the shock in column
    ``permanent_var_idx`` has a long-run effect on the *level* of that
    variable (Blanchard-Quah 1989, section 1; the long-run matrix is
    ``C(1)`` there).

    Parameters
    ----------
    Y : ndarray, shape (T, n)
        Stationary data, rows ordered in time. A variable with a unit root
        enters in first differences (e.g. ``100 * diff(log GDP)``); a
        stationary one may enter in levels (e.g. the unemployment rate).
    p, lags : int, optional
        Lag order (``lags`` is an alias); default 2.
    horizon : int
        Last horizon ``H``; IRFs have ``H + 1`` rows (h = 0..H).
    permanent_var_idx : int
        Variable (0-based column) on which only its own shock has a
        long-run effect.
    n_boot : int
        Residual-bootstrap replications for the bands; 0 returns NaN bands.
    ci : float
        Central coverage of the percentile bands.
    seed : int
        Seed of the bootstrap generator.
    cumulate : bool, sequence of int, or boolean mask; default ``True``
        Which response variables (axis 1 of the IRF arrays) are cumulated
        over the horizon before they are returned. Cumulating the response
        of a first-differenced variable gives the response of its level:
        with ``Y[:, i] = 100 * diff(log y)`` the cumulated row is the percent
        deviation of ``y``. A variable that enters in levels must not be
        cumulated, or its response becomes the running sum of the true one
        and converges to ``((I - A(1))^-1 B)[i, :]`` instead of 0.

        * ``True`` (default, the behaviour of every earlier release):
          cumulate every variable. Right when all columns are differenced,
          as in Gali's (1999) difference specification
          ``Y = [dlog productivity, dlog hours]``.
        * a list of indices, e.g. ``[0]``: cumulate only those variables.
          The canonical Blanchard-Quah specification
          ``Y = [dlog GNP, unemployment rate]`` needs ``cumulate=[0]``:
          output is reported as a log level and unemployment as the rate.
        * a boolean mask of length ``n``, same meaning.
        * ``False`` or ``[]``: return the raw (non-cumulated) responses.

        The same transformation is applied to the point estimate and to
        every bootstrap draw before the percentiles are taken, so the bands
        are quantiles of the reported object (``np.diff`` of cumulated bands
        is not a band of the raw response).

    Returns
    -------
    BQSVARResult
        ``irf_point[h, i, j]`` is the response of variable ``i`` at horizon
        ``h`` to a one-standard-deviation structural shock ``j``, in the
        units of ``Y[:, i]`` — cumulated over ``0..h`` for the variables
        selected by ``cumulate``. ``irf_lower`` / ``irf_upper`` are the
        percentile bands of the same object.

    References
    ----------
    Blanchard, O.J. and Quah, D. (1989). The dynamic effects of aggregate
        demand and supply disturbances. AER 79(4), 655-673 (NBER WP 2737,
        1988: ``X = (dY, U)'`` with Y log GNP and U the level of the
        unemployment rate; the output level response is the partial sum of
        the dY responses).
    Gali, J. (1999). Technology, employment, and the business cycle. AER
        89(1), 249-271.
    """
    if lags is not None:
        p = lags
    if p is None:
        p = 2
    rng = default_rng(seed)
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    mask = _cumulate_mask(cumulate, Sigma.shape[0])
    B = _bq_impact(A_list, Sigma, permanent_var_idx)
    # compute_irf returns (H+1, n, n) indexed [h, response i, shock j];
    # cumulate the selected response rows along the horizon axis (0).
    point = _cumulate_rows(compute_irf(A_list, B, horizon), mask)

    T, n = Y.shape
    accepted = []
    n_fail = 0
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q

    if n_boot <= 0:
        nan_bands = np.full_like(point, np.nan)
        return BQSVARResult(
            irf_point=point, irf_lower=nan_bands, irf_upper=nan_bands,
            n_boot=0, n_fail=0, ci=ci,
        )

    for b in range(n_boot):
        idx = rng.integers(0, len(resid), size=len(resid))
        eps_b = resid[idx]
        Yb = np.zeros_like(Y)
        Yb[:p] = Y[:p]
        for t in range(p, T):
            yt = c.copy()
            for l in range(p):
                yt += A_list[l] @ Yb[t - l - 1]
            yt += eps_b[t - p]
            Yb[t] = yt
        A_b, _, Sigma_b, _, _ = estimate_var(Yb, p)
        # Reject ill-conditioned reduced-form Σ_b before even attempting
        # the long-run impact factorisation: bands built on bootstrap
        # draws with cond(Σ_b) > 1e8 are statistically meaningless. Also
        # guard against LAPACK's silent NaN-factor case on ultra-
        # degenerate Σ_b (see cholesky.py for the same pattern).
        try:
            L_test = safe_cholesky(Sigma_b, name="bq_svar bootstrap Σ_b")
        except np.linalg.LinAlgError:
            n_fail += 1
            continue
        diagL = np.abs(np.diag(L_test))
        if (not np.all(np.isfinite(diagL))
                or (diagL.size
                    and diagL.min() < _BOOT_COND_RATIO_FLOOR * diagL.max())):
            n_fail += 1
            continue
        try:
            B_b = _bq_impact(A_b, Sigma_b, permanent_var_idx)
        except np.linalg.LinAlgError:
            n_fail += 1
            continue
        accepted.append(_cumulate_rows(compute_irf(A_b, B_b, horizon), mask))

    if not accepted:
        raise np.linalg.LinAlgError(
            f"bq_svar: all {n_boot} bootstrap draws produced a non-PD "
            "long-run covariance Ω. The data may be too short or the "
            "VAR may be near non-stationary."
        )
    fail_rate = n_fail / n_boot
    if fail_rate > _BOOT_FAIL_WARN_THRESHOLD:
        warnings.warn(
            f"bq_svar: {n_fail}/{n_boot} bootstrap draws "
            f"({fail_rate:.1%}) produced a non-PD long-run Ω and were "
            "dropped. Bands are computed from the surviving draws and "
            "may be unreliable.",
            stacklevel=2,
        )
    draws = np.stack(accepted, axis=0)
    return BQSVARResult(
        irf_point=point,
        irf_lower=np.quantile(draws, lo_q, axis=0),
        irf_upper=np.quantile(draws, hi_q, axis=0),
        n_boot=n_boot,
        n_fail=n_fail,
        ci=ci,
    )
