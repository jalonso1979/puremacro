"""Hansen-Lunde-Nason (2011) Model Confidence Set.

The Model Confidence Set (MCS) takes a collection of M forecasting models
and returns the subset that contains the best model(s) with a prescribed
level of confidence 1 - alpha. Unlike a pairwise Diebold-Mariano test — which
only ever compares two forecasts and leaves a multi-model ranking ambiguous
— the MCS controls the family-wise error of the whole comparison through a
sequential *elimination* procedure:

  1. Form the loss-differential matrix and, via a block bootstrap that
     preserves serial and cross-model dependence, estimate the distribution
     of an equivalence-test statistic (T_max, the range T_R, or the
     semi-quadratic T_SQ) under the null of equal predictive ability.
  2. Test the null on the current set. If it is rejected, eliminate the
     single worst model (largest standardized excess loss) and repeat.
  3. Record, for every model, its MCS p-value as the running maximum of the
     equivalence-test p-values along the elimination path.

The set retained at confidence 1 - alpha is ``{i : p_MCS(i) >= alpha}``.
Because the p-values are a running maximum they are monotone along the
elimination order, so a larger alpha yields a (weakly) smaller set — the
standard MCS nesting.

This is the genuine bootstrap elimination inference procedure. It is NOT to
be confused with :func:`puremacro.nowcast.combine.model_confidence_set`,
which is a deliberately lightweight *thresholded* stand-in (keep models
within ``(1+alpha)`` of the best MSE and average them) used only to build a
combination weight, with no bootstrap and no size control.

Reference
---------
Hansen, P.R., Lunde, A. and Nason, J.M. (2011). The Model Confidence Set.
    Econometrica 79(2), 453-497.

Politis, D.N. and Romano, J.P. (1994). The stationary bootstrap. Journal of
    the American Statistical Association 89(428), 1303-1313. (Default resampler.)
"""
from __future__ import annotations

import numpy as np

__all__ = ["model_confidence_set", "losses_from_forecasts"]


# --------------------------------------------------------------------------- #
# Loss construction (mirrors forecast.compare's loss vocabulary).             #
# --------------------------------------------------------------------------- #
def losses_from_forecasts(
    forecasts: np.ndarray, realized: np.ndarray, *,
    loss: str = "mse", power: float = 2.0,
) -> np.ndarray:
    """Build a (T, M) per-period loss matrix from a forecast panel.

    ``forecasts`` is (T, M), ``realized`` is (T,). Loss options match
    :mod:`puremacro.forecast.compare`: ``"mse"`` (squared error), ``"mae"``
    (absolute error), ``"lp"`` (|error|**power).
    """
    F = np.asarray(forecasts, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    y = np.asarray(realized, dtype=float).ravel()
    e = F - y[:, None]
    if loss == "mse":
        return e ** 2
    if loss == "mae":
        return np.abs(e)
    if loss == "lp":
        return np.abs(e) ** power
    raise ValueError(f"unknown loss {loss!r}")


# --------------------------------------------------------------------------- #
# Block bootstrap of time indices (shared across models by construction, so   #
# cross-model dependence is preserved).                                       #
# --------------------------------------------------------------------------- #
def _bootstrap_indices(
    T: int, n_boot: int, block: float, kind: str, rng: np.random.Generator,
) -> np.ndarray:
    """Return an (n_boot, T) integer index array for the chosen block scheme."""
    if kind == "stationary":
        # Politis-Romano: restart a new random block with prob 1/block each
        # step, otherwise advance by one (wrapping). Expected block length = block.
        p = 1.0 / max(block, 1.0)
        rand_start = rng.integers(0, T, size=(n_boot, T))
        new_block = rng.random((n_boot, T)) < p
        idx = np.empty((n_boot, T), dtype=np.intp)
        idx[:, 0] = rand_start[:, 0]
        for t in range(1, T):
            cont = (idx[:, t - 1] + 1) % T
            idx[:, t] = np.where(new_block[:, t], rand_start[:, t], cont)
        return idx
    if kind == "moving":
        # Fixed-length overlapping blocks, concatenated and trimmed to T.
        L = max(1, int(round(block)))
        n_blocks = int(np.ceil(T / L))
        starts = rng.integers(0, T, size=(n_boot, n_blocks))
        offsets = np.arange(L)
        # (n_boot, n_blocks, L) -> flatten -> trim to T, wrap with modulo.
        blocks = (starts[:, :, None] + offsets[None, None, :]) % T
        return blocks.reshape(n_boot, n_blocks * L)[:, :T].astype(np.intp)
    raise ValueError(f"unknown bootstrap {kind!r}")


# --------------------------------------------------------------------------- #
# One equivalence test on the currently-active model set.                     #
# --------------------------------------------------------------------------- #
def _equivalence_test(
    active: list[int], loss_bar: np.ndarray, loss_star: np.ndarray,
    statistic: str,
) -> tuple[float, int]:
    """Return ``(p_value, worst_model_index)`` for the active set.

    ``loss_bar`` is the (M,) full-sample mean loss; ``loss_star`` is the
    (n_boot, M) matrix of block-bootstrap mean losses. All quantities for a
    subset and for every pair reduce to differences of these precomputed
    means, so each step is O(n_boot * |active|).
    """
    a = np.asarray(active, dtype=np.intp)
    lb = loss_bar[a]                      # (k,)
    ls = loss_star[:, a]                  # (B, k)
    k = a.size

    # Standardized excess loss of each model vs the set average: t_{i.}.
    # d_{i.} = Lbar_i - mean_j Lbar_j  (mean over the active set including i;
    # the constant 1/k shift is immaterial to the argmax and cancels in t).
    d_dot = lb - lb.mean()                                   # (k,)
    d_dot_star = ls - ls.mean(axis=1, keepdims=True)         # (B, k)
    dev_dot = d_dot_star - d_dot                             # recentered
    scale_dot = np.sqrt((dev_dot ** 2).mean(axis=0))        # bootstrap sd
    good = scale_dot > 1e-12
    t_dot = np.where(good, d_dot / np.where(good, scale_dot, 1.0), 0.0)
    tstar_dot = np.where(good, dev_dot / np.where(good, scale_dot, 1.0), 0.0)

    # Elimination rule (shared by all statistics): the model with the largest
    # standardized loss relative to the set is the one to remove.
    worst = int(a[int(np.argmax(t_dot))])

    if statistic == "tmax":
        t_obs = float(np.max(t_dot))
        t_boot = tstar_dot.max(axis=1)
    else:
        # Pairwise standardized differences t_{ij} for the range / semi-quad.
        d_ij = lb[:, None] - lb[None, :]                    # (k, k)
        d_ij_star = ls[:, :, None] - ls[:, None, :]         # (B, k, k)
        dev_ij = d_ij_star - d_ij
        scale_ij = np.sqrt((dev_ij ** 2).mean(axis=0))      # (k, k)
        gij = scale_ij > 1e-12
        t_ij = np.where(gij, d_ij / np.where(gij, scale_ij, 1.0), 0.0)
        tstar_ij = np.where(gij, dev_ij / np.where(gij, scale_ij, 1.0), 0.0)
        iu = np.triu_indices(k, k=1)
        if statistic == "tr":
            t_obs = float(np.max(np.abs(t_ij[iu])))
            t_boot = np.abs(tstar_ij[:, iu[0], iu[1]]).max(axis=1)
        elif statistic == "tsq":
            t_obs = float(np.sum(t_ij[iu] ** 2))
            t_boot = (tstar_ij[:, iu[0], iu[1]] ** 2).sum(axis=1)
        else:
            raise ValueError(f"unknown statistic {statistic!r}")

    p_value = float(np.mean(t_boot >= t_obs))
    return p_value, worst


# --------------------------------------------------------------------------- #
# Public entry point.                                                          #
# --------------------------------------------------------------------------- #
def model_confidence_set(
    losses: np.ndarray | None = None, *,
    forecasts: np.ndarray | None = None, realized: np.ndarray | None = None,
    loss: str = "mse", power: float = 2.0,
    alpha: float = 0.10, statistic: str = "tmax",
    n_boot: int = 1000, block: float | None = None,
    bootstrap: str = "stationary", seed: int | None = None,
) -> dict:
    """Hansen-Lunde-Nason (2011) Model Confidence Set via block-bootstrap elimination.

    Parameters
    ----------
    losses : (T, M) ndarray, optional
        Per-period losses for M models. If omitted, built from ``forecasts``
        (T, M) and ``realized`` (T,) with :func:`losses_from_forecasts`.
    forecasts, realized, loss, power
        Alternative input path (see ``losses_from_forecasts``). Ignored when
        ``losses`` is given.
    alpha : float, default 0.10
        Significance level; the returned set has confidence ``1 - alpha``.
    statistic : {"tmax", "tr", "tsq"}, default "tmax"
        Equivalence-test statistic. ``"tmax"`` = max standardized excess loss
        (the common default), ``"tr"`` = range of standardized pairwise
        differences, ``"tsq"`` = semi-quadratic (sum of squares). The
        elimination rule is the same for all three.
    n_boot : int, default 1000
        Number of block-bootstrap resamples.
    block : float, optional
        Expected (stationary) / fixed (moving) block length. Defaults to
        ``max(2, round(T ** (1/3)))``.
    bootstrap : {"stationary", "moving"}, default "stationary"
        Block-bootstrap scheme (Politis-Romano stationary, or moving block).
    seed : int, optional
        Seed for the RNG. Same seed ⇒ identical p-values (reproducible).

    Returns
    -------
    dict
        ``included`` — sorted indices of models in the MCS at level ``alpha``;
        ``excluded`` — the complement; ``pvalues`` — (M,) MCS p-values (running
        max along the elimination path; the surviving model is 1.0);
        ``elimination_order`` — indices in the order they were removed (worst
        first); plus echoed ``statistic``, ``alpha``, ``n_boot``, ``block``.

    Notes
    -----
    This is the genuine bootstrap elimination test. For the lightweight
    thresholded stand-in used to form *combination weights* see
    :func:`puremacro.nowcast.combine.model_confidence_set`.
    """
    if losses is None:
        if forecasts is None or realized is None:
            raise ValueError("provide `losses`, or both `forecasts` and `realized`")
        losses = losses_from_forecasts(forecasts, realized, loss=loss, power=power)
    L = np.asarray(losses, dtype=float)
    if L.ndim != 2:
        raise ValueError("`losses` must be a (T, M) matrix")
    T, M = L.shape
    if M < 2:
        raise ValueError("need at least 2 models to form a confidence set")

    if block is None:
        block = max(2.0, round(T ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)

    # Draw the bootstrap once and reuse across every elimination step, so the
    # whole sequence is coherent (same resamples) and cheap.
    idx = _bootstrap_indices(T, n_boot, float(block), bootstrap, rng)
    loss_bar = L.mean(axis=0)                       # (M,)
    loss_star = L[idx].mean(axis=1)                 # (n_boot, M)

    pvalues = np.zeros(M, dtype=float)
    active = list(range(M))
    elimination_order: list[int] = []
    running = 0.0
    # Run the elimination all the way down to a single model; alpha enters only
    # at the final thresholding, which keeps the p-values (and hence the nested
    # sets across alpha) well defined.
    while len(active) > 1:
        p, worst = _equivalence_test(active, loss_bar, loss_star, statistic)
        running = max(running, p)
        pvalues[worst] = running
        elimination_order.append(worst)
        active.remove(worst)
    # The last surviving model is never eliminated: MCS p-value 1 by convention.
    pvalues[active[0]] = 1.0

    included = np.array(sorted(i for i in range(M) if pvalues[i] >= alpha), dtype=int)
    excluded = np.array([i for i in range(M) if i not in set(included.tolist())],
                        dtype=int)
    return {
        "included": included,
        "excluded": excluded,
        "pvalues": pvalues,
        "elimination_order": np.array(elimination_order, dtype=int),
        "statistic": statistic,
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "block": float(block),
    }
