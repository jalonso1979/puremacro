"""MCMC convergence diagnostics for puremacro Gibbs samplers
(``minnesota_gibbs``, ``ms_var_fit``, ``tvp_var_fit``, ``tvp_var_sv_fit``).

Currently exposes:
    - ``geweke_z``                  : Geweke (1992) z-score on
                                      first-vs-last segment of one chain,
                                      with an AR-fitted spectral density
                                      at frequency zero (R ``coda``).
    - ``gelman_rubin``              : classic R̂ across chains, or split-R̂
                                      with ``split=True``.
    - ``effective_sample_size``     : ESS via Geyer's initial monotone
                                      sequence (Vehtari et al. 2021).
    - ``autocorrelations``          : full ACF for diagnostic plotting.
    - ``trace_summary``             : summary statistics across chains.

Conventions shared by the diagnostics
-------------------------------------
* The spectral density at frequency zero, ``S(0)``, is the *long-run
  variance* ``sum_{k=-inf}^{inf} gamma_k`` of the chain (no ``1/(2 pi)``
  factor), so that ``Var(mean of n draws) ~= S(0) / n``.  For a stationary
  AR(1) with unit marginal variance and coefficient ``phi``,
  ``S(0) = (1 + phi) / (1 - phi)``.
* The effective sample size is ``n / tau`` with the integrated
  autocorrelation time ``tau = S(0) / gamma_0 = 1 + 2 sum_{k>=1} rho_k``.
  For the same AR(1), ``ESS = n (1 - phi) / (1 + phi)``.
* Sample autocovariances use divisor ``n`` (the positive semi-definite
  estimator used by R ``acf``/``ar`` and by arviz).
"""
from __future__ import annotations

import numpy as np


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Sample autocovariances ``gamma_0 .. gamma_{n-1}`` of a 1-D series.

    Demeaned, divisor ``n`` for every lag (the positive semi-definite
    estimator of R ``acf(type="covariance")`` and arviz ``autocov``),
    computed by zero-padded FFT so the cost is ``O(n log n)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n == 0:
        return np.zeros(0)
    xc = x - x.mean()
    m = 1 << int(np.ceil(np.log2(max(2 * n, 2))))  # >= 2n - 1: no wrap-around
    f = np.fft.rfft(xc, n=m)
    return np.fft.irfft(f * np.conj(f), n=m)[:n] / n


def _spectrum0_ar(x: np.ndarray, order_max: int | None = None) -> dict:
    """AR-fitted spectral density at frequency zero (R ``coda::spectrum0.ar``).

    Fits an autoregression ``x_t - mu = sum_{j=1}^{p} a_j (x_{t-j} - mu) + e_t``
    by Yule-Walker (Levinson-Durbin recursion on the divisor-``n`` sample
    autocovariances), chooses the order ``p`` in ``0..order_max`` by AIC
    ``n log(sigma2_p) + 2 p``, rescales the innovation variance as R
    ``ar.yw`` does, ``var_pred = sigma2_p * n / (n - (p + 1))``, and returns

        S(0) = var_pred / (1 - sum_j a_j)**2,

    the long-run variance of the chain (``Var(mean) ~= S(0) / n``; same
    units as ``Var(x)``).  ``order_max`` defaults to R's
    ``min(n - 1, floor(10 log10 n))`` (capped at ``n - 2`` so the rescaling
    is defined).  As in coda, a series that is constant or exactly linear
    in time has ``S(0) = 0``: constant means a range
    ``max(x) - min(x) <= 1e-15 max|x|``; exactly linear means a residual
    s.d. of a regression on a time trend ``<= 1.5e-8`` times the s.d. of
    the series (coda uses an absolute 1.5e-8, which misclassifies series
    measured in very small units).  Any two-draw series is exactly linear.

    Matches ``coda::spectrum0.ar`` (coda 0.19-4.1, R/heidel.R) and
    ``stats::ar.yw`` (R 4.6.0) to about 1e-11 relative on test series;
    the error is largest when ``1 - sum(a)`` is small (near-unit-root
    series such as a random walk).

    Returns
    -------
    dict with ``spec`` (S(0)), ``order`` (chosen p), ``ar`` (coefficients
    ``a_1..a_p``) and ``var_pred`` (rescaled innovation variance).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2 or not np.all(np.isfinite(x)):
        return {"spec": float("nan"), "order": 0, "ar": np.zeros(0),
                "var_pred": float("nan")}
    degenerate = {"spec": 0.0, "order": 0, "ar": np.zeros(0), "var_pred": 0.0}
    # Constant up to rounding (the rule effective_sample_size uses).  Needed
    # because for, e.g., np.full(n, 0.3) the computed mean is not exactly
    # 0.3, x - mean is a constant ~1e-17 and the relative test below
    # compares rounding noise with itself.
    if float(x.max() - x.min()) <= np.finfo(float).resolution * float(np.abs(x).max()):
        return degenerate
    # coda: lm(x ~ time); a (near-)zero residual s.d. means S(0) := 0.
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    xc = x - x.mean()
    slope = float(tc @ xc) / float(tc @ tc)
    resid = xc - slope * tc
    resid_sd = float(np.sqrt(resid @ resid / max(n - 1, 1)))
    scale = float(np.sqrt(xc @ xc / max(n - 1, 1)))
    if resid_sd == 0.0 or resid_sd <= 1.5e-8 * scale:
        return degenerate

    r = _autocovariance(x)
    if order_max is None:
        order_max = min(n - 1, int(np.floor(10.0 * np.log10(n))))
    order_max = int(max(0, min(order_max, n - 2)))

    # Levinson-Durbin: coefs[m] are the Yule-Walker AR(m) coefficients and
    # v[m] the corresponding innovation variance (R's Fortran ``eureka``).
    v = np.empty(order_max + 1)
    v[0] = r[0]
    phi = np.zeros(0)
    coefs = [phi]
    last = 0
    for m in range(1, order_max + 1):
        if not v[m - 1] > 0.0:
            break
        k = (r[m] - float(phi @ r[m - 1:0:-1])) / v[m - 1]
        phi = np.concatenate([phi - k * phi[::-1], [k]])
        v[m] = v[m - 1] * (1.0 - k * k)
        coefs.append(phi)
        last = m
    v = v[: last + 1]
    with np.errstate(divide="ignore"):
        aic = n * np.log(v) + 2.0 * np.arange(last + 1)
    p = int(np.argmin(aic))
    var_pred = float(v[p]) * n / (n - (p + 1))
    a = coefs[p]
    spec = var_pred / (1.0 - float(a.sum())) ** 2
    return {"spec": float(spec), "order": p, "ar": a.copy(), "var_pred": var_pred}


def _spectral_density_zero(x: np.ndarray, max_lag: int | None = None) -> float:
    """Legacy Bartlett-kernel estimate of the spectral density at frequency 0.

    Kept only for ``geweke_z(..., spectrum="bartlett")`` (the behaviour of
    puremacro 4.3.0 and earlier).  The default bandwidth ``int(4 (n/100)**(2/9))`` is a
    Newey-West rule of thumb that is far too short for persistent MCMC
    chains (5-7 lags for 500-2500 draws), so it underestimates S(0) badly
    when the lag-1 autocorrelation is high; ``geweke_z`` now uses
    :func:`_spectrum0_ar` instead.
    """
    n = len(x)
    if max_lag is None:
        max_lag = int(4 * (n / 100) ** 0.2)
    x_dem = x - x.mean()
    g0 = float((x_dem ** 2).mean())
    s = g0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = float((x_dem[lag:] * x_dem[:-lag]).mean())
        s += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    return max(s, 1e-12)


def geweke_z(
    chain: np.ndarray,
    first: float = 0.1,
    last: float = 0.5,
    *,
    spectrum: str = "ar",
) -> float:
    """Geweke (1992) convergence z-score for one chain.

    Compares the mean of the first ``first`` fraction of the draws
    (``n_A = int(first * n)``) with the mean of the last ``last`` fraction
    (``n_B = int(last * n)``); defaults are Geweke's 10% and 50%:

        z = (mean_A - mean_B) / sqrt(S_A(0) / n_A + S_B(0) / n_B),

    where ``S_A(0)``, ``S_B(0)`` are the spectral densities at frequency
    zero (long-run variances, see the module docstring) estimated
    separately on each segment.  Under the null of stationarity ``z`` is
    asymptotically N(0, 1); ``|z| > 1.96`` suggests the early draws have
    not converged.

    Parameters
    ----------
    chain : array-like, shape (n,)
        One scalar chain.
    first, last : float
        Segment fractions; ``0 < first, last < 1`` and
        ``first + last <= 1`` (the segments must not overlap).
    spectrum : {"ar", "bartlett"}, default "ar"
        Estimator of ``S(0)``.  ``"ar"`` fits an AR(p) model with the order
        chosen by AIC, exactly as R ``coda::geweke.diag`` does through
        ``spectrum0.ar`` (see :func:`_spectrum0_ar`).  On stationary AR(1)
        chains of 5,000 draws its rejection rate at |z| > 1.96 is about
        0.057, 0.072 and 0.088 for ``phi`` = 0.5, 0.9 and 0.95 (4,000
        seeded replications); at ``phi = 0.99`` it is 0.19 with 5,000
        draws, 0.087 with 20,000 and 0.074 with 50,000.  The size is close
        to nominal once the first segment holds a few dozen effective
        draws (``n_A (1 - phi) / (1 + phi)``); with fewer, the AR fit is
        biased towards zero and the test over-rejects.  ``"bartlett"``
        reproduces the estimator of puremacro 4.3.0 and earlier (Bartlett
        kernel with ``int(4 (m/100)**(2/9))`` lags), which underestimates
        ``S(0)`` for persistent chains: at 5,000 draws it rejects about 33%
        of the time at ``phi = 0.9``, 48% at 0.95 and 76% at 0.99.

    Notes
    -----
    The segments are taken as ``chain[:n_A]`` and ``chain[-n_B:]``, with
    at least 2 draws each.  ``coda::geweke.diag`` uses
    ``ceiling(1 + first (n - 1))`` and ``n - floor(n - last (n - 1)) + 1``
    draws (``window`` is inclusive), i.e. one or two draws more at each
    inner boundary (501 and 2501 draws for ``n = 5000``; 101 and 500
    against 99 and 499 for ``n = 999``); on identical segments the two
    agree to rounding.

    Degenerate cases.  As in coda, a segment that is constant or exactly
    linear in time has ``S(0) = 0`` (see :func:`_spectrum0_ar`); a stuck
    first segment, common with random-walk Metropolis, therefore
    contributes no variance.  When both spectra are 0 the result is
    ``nan`` if the segment means agree to rounding (``|diff| <= 1e-12
    max|x|``, e.g. a constant chain) and ``+-inf`` otherwise (e.g. two
    different constants, or a linear trend).  Any two-draw segment is
    exactly linear: with the default fractions the first segment has two
    draws, and contributes no variance, for ``n < 30``, and for
    ``n = 4, 5`` both segments do, so ``z`` is ``+-inf``.  ``n < 4``
    leaves no room for two disjoint two-draw segments and returns
    ``nan``.  The test is informative only when each segment holds a few
    dozen effective draws.
    """
    if spectrum not in ("ar", "bartlett"):
        raise ValueError(f"spectrum must be 'ar' or 'bartlett', got {spectrum!r}")
    if not (0.0 < first < 1.0) or not (0.0 < last < 1.0):
        raise ValueError("first and last must lie strictly between 0 and 1.")
    if first + last > 1.0 + 1e-12:
        raise ValueError(
            "first + last must not exceed 1: the two segments would overlap."
        )
    chain = np.asarray(chain, dtype=float).ravel()
    n = len(chain)
    n_first = max(2, int(first * n))
    n_last = max(2, int(last * n))
    if n_first + n_last > n:
        return float("nan")           # n < 4: no two disjoint 2-draw segments
    first_seg = chain[:n_first]
    last_seg = chain[-n_last:]
    diff = float(first_seg.mean() - last_seg.mean())
    if spectrum == "bartlett":
        s_first = _spectral_density_zero(first_seg)
        s_last = _spectral_density_zero(last_seg)
        se = float(np.sqrt(s_first / n_first + s_last / n_last))
        return diff / se if se > 0 else float("nan")
    s_first = _spectrum0_ar(first_seg)["spec"]
    s_last = _spectrum0_ar(last_seg)["spec"]
    var = s_first / n_first + s_last / n_last
    if not np.isfinite(var) or not np.isfinite(diff):
        return float("nan")
    if var <= 0.0:
        # Both segments constant or exactly linear.  A difference of means
        # at rounding level (np.full(n, 0.3) gives ~1e-17) counts as 0.
        scale = max(float(np.abs(first_seg).max()), float(np.abs(last_seg).max()))
        if abs(diff) <= 1e-12 * scale:
            return float("nan")
        return float(np.copysign(np.inf, diff))
    return diff / float(np.sqrt(var))


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split each of M chains into halves -> (2M, n // 2).

    When ``n`` is odd the middle draw is dropped (arviz 0.21
    ``_split_chains``: ``ary[:, :half]`` and ``ary[:, -half:]``).
    """
    chains = np.asarray(chains, dtype=float)
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, chains.shape[1] - half:]], axis=0)


def gelman_rubin(chains: np.ndarray, *, split: bool = False) -> dict:
    """Gelman-Rubin potential scale reduction factor R̂.

    With ``W`` the mean within-chain variance (divisor ``n - 1``), ``B``
    ``n`` times the variance of the chain means (divisor ``M - 1``) and
    ``var_plus = (n - 1)/n W + B/n``, returns ``R_hat = sqrt(var_plus / W)``
    (Gelman and Rubin 1992; Vehtari et al. 2021, eqs. 1-4).  R̂ near 1
    indicates the between-chain variance has shrunk to the within-chain
    variance.

    Parameters
    ----------
    chains : ndarray, shape (M, n)
        M chains of n draws each.
    split : bool, default False
        ``False`` computes the classic (unsplit) statistic on the M chains
        as given; it matches arviz ``rhat(method="identity")``.  ``True``
        first splits every chain in half (dropping the middle draw when
        ``n`` is odd) and computes R̂ on the resulting 2M half-chains:
        the split-R̂ of Gelman et al. (BDA3) and Vehtari et al. (2021, p.6),
        which also detects a trend *within* chains (the unsplit statistic
        is about 1 when every chain drifts the same way) and is defined for
        a single chain.  It matches arviz ``rhat(method="split")``.  The
        ``1.01`` threshold of Vehtari et al. refers to their rank-normalised
        split-R̂, not to the classic statistic.

    Returns
    -------
    dict with ``R_hat``, ``W``, ``B`` and ``var_plus`` (computed on the
    half-chains when ``split=True``).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must be 2-D (M, n).")
    if split:
        if chains.shape[1] < 4:
            raise ValueError("split R-hat needs at least 4 draws per chain.")
        chains = _split_chains(chains)
    M, n = chains.shape
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n * ((chain_means - grand_mean) ** 2).sum() / max(M - 1, 1)
    W = ((chains - chain_means[:, None]) ** 2).sum() / (M * max(n - 1, 1))
    var_plus = ((n - 1) / n) * W + B / n
    R_hat = float(np.sqrt(var_plus / W)) if W > 0 else float("nan")
    return {"R_hat": R_hat, "W": float(W), "B": float(B), "var_plus": float(var_plus)}


def autocorrelations(chain: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Sample ACF up to ``max_lag``."""
    chain = np.asarray(chain, dtype=float).ravel()
    x = chain - chain.mean()
    var = float((x ** 2).mean())
    if var == 0:
        return np.zeros(max_lag + 1)
    rho = np.zeros(max_lag + 1)
    rho[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            break
        rho[lag] = float((x[lag:] * x[:-lag]).mean()) / var
    return rho


def effective_sample_size(chain: np.ndarray, max_lag: int | None = None) -> float:
    """Effective sample size of one chain via Geyer's initial monotone sequence.

    ``ESS = n / tau`` with the integrated autocorrelation time

        tau = -1 + 2 sum_{t'=0}^{k} P_t',   P_t' = rho_{2t'} + rho_{2t'+1},

    i.e. ``1 + 2 sum_{t=1}^{2k+1} rho_t`` (Vehtari, Gelman, Simpson,
    Carpenter and Bürkner 2021, "Rank-normalization, folding, and
    localization", arXiv:1903.08008, eqs. 12-13; Geyer 1992).  The pairs
    start at lag 0; ``k`` is the last index before the first negative pair
    (initial positive sequence); the pairs are then replaced by their
    running minimum (initial monotone sequence).  As in Stan and arviz
    0.21 (``arviz.stats.diagnostics._ess``, which this function matches for
    a single chain), the truncated sums ending at the odd lag ``2k+1`` and
    at the next even lag are averaged, and ``ESS`` is capped at
    ``n log10(n)``.  Autocorrelations are
    ``rho_t = 1 - (s^2 - gamma_t) / gamma_0``, the single-chain case of
    Vehtari et al.'s eq. 10 (``gamma_t`` the divisor-``n`` autocovariance,
    ``s^2`` the divisor-``n-1`` variance), i.e. ``gamma_t/gamma_0 - 1/(n-1)``.

    ``ESS`` measures efficiency for the posterior *mean*: ``Var(mean) =
    Var(x) / ESS``.  For a stationary AR(1) with coefficient ``phi`` the
    true value is ``n (1 - phi) / (1 + phi)``, which exceeds ``n`` for
    antithetic chains (``phi < 0``, as NUTS often produces).

    Parameters
    ----------
    chain : array-like
        One chain (flattened if multi-dimensional).
    max_lag : int, optional
        Largest autocorrelation lag the estimator may read.  Default
        ``None``: lags up to ``n - 2``, so only the truncation rule decides
        where the sum stops.  With ``max_lag >= 3`` the pair loop stops at
        the last pair ``(rho_{2j}, rho_{2j+1})`` with ``2j + 1 <= max_lag``,
        and the sum uses lags up to ``max_lag - 1`` at most.  Small values:
        ``max_lag <= 0`` applies no correction (``tau = 1``, so
        ``ESS = n`` for ``n >= 10``; below that the cap applies);
        ``max_lag = 1`` gives ``tau = 1 + 2 rho_1``; ``max_lag = 2`` gives
        ``tau = 1 + 2 rho_1 + max(rho_2, 0)``, the same as ``max_lag = 3``.
        A ``max_lag`` shorter than the autocorrelation time drops positive
        autocorrelations and overstates the ESS of a persistent chain
        (AR(1) with ``phi = 0.9``, ``n = 2000``: true ESS 105; ``max_lag =
        2`` gives about 550).  In puremacro 4.3.0 and earlier the default
        was ``min(n - 1, 1000)``, the pairs started at lag 1 and
        ``max_lag <= 1`` returned ``n``.

    Returns
    -------
    float
        ``n`` for a constant chain or ``n < 4``; ``nan`` if the chain has
        non-finite values.  Otherwise at most the cap ``n log10(n)``
        (which is below ``n`` when ``n < 10``).  For ``n == 4`` (and
        ``max_lag`` None or ``>= 3``) the pair loop cannot run and, as in
        arviz, the result is that cap, 2.41.
    """
    chain = np.asarray(chain, dtype=float).ravel()
    n = len(chain)
    if n == 0 or not np.all(np.isfinite(chain)):
        return float("nan")
    ptp = float(chain.max() - chain.min())
    if n < 4 or ptp <= np.finfo(float).resolution * float(np.abs(chain).max()):
        return float(n)                           # constant up to rounding
    acov = _autocovariance(chain)
    var_plus = float(acov[0])                     # (n-1)/n * s^2
    mean_var = var_plus * n / (n - 1.0)           # s^2
    if not var_plus > 0.0:
        return float(n)

    def rho_at(t: int) -> float:
        return 1.0 - (mean_var - float(acov[t])) / var_plus

    tau_floor = 1.0 / np.log10(n)                 # ESS <= n log10(n)
    # The pair loop below reads the pair (rho_{t+1}, rho_{t+2}) while
    # t < limit, i.e. lags up to limit + 1 = n - 2 (arviz) or max_lag.
    limit = n - 3
    if max_lag is not None:
        max_lag = int(max_lag)
        if max_lag < 3:
            # Too few lags for the pair loop, whose first pair is lags 2-3.
            # max_lag <= 0: no autocorrelation correction, tau = 1.
            # max_lag == 1: the lag-0 pair only, tau = -1 + 2 (rho_0 + rho_1).
            # max_lag == 2: that sum averaged with the one truncated at lag 2,
            # tau = 1 + 2 rho_1 + max(rho_2, 0), which is what the loop gives
            # for max_lag == 3.
            if max_lag <= 0:
                tau = 1.0
            else:
                tau = 1.0 + 2.0 * rho_at(1)
                if max_lag == 2:
                    tau += max(rho_at(2), 0.0)
            return float(n / max(tau, tau_floor))
        limit = min(limit, max_lag - 1)
    rho = np.zeros(n)
    rho_even = 1.0
    rho_odd = rho_at(1)
    rho[0] = rho_even
    rho[1] = rho_odd
    # Geyer's initial positive sequence (pairs from lag 0).  Here limit >= 2
    # unless n == 4, so the loop runs at least once unless n == 4 or
    # rho_0 + rho_1 <= 0; in those two cases max_t == -1, tau == 0 and,
    # exactly as in arviz, ESS is the cap n log10(n) (2.41 for n == 4).
    t = 1
    while t < limit and (rho_even + rho_odd) > 0.0:
        rho_even = rho_at(t + 1)
        rho_odd = rho_at(t + 2)
        if (rho_even + rho_odd) >= 0.0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2
    max_t = t - 2
    # Average the sums truncated at the odd lag and at the next even lag.
    if rho_even > 0.0 and max_t + 1 < n:
        rho[max_t + 1] = rho_even
    # Geyer's initial monotone sequence: running minimum of the pairs.
    t = 1
    while t <= max_t - 2:
        if (rho[t + 1] + rho[t + 2]) > (rho[t - 1] + rho[t]):
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2
    tau = -1.0 + 2.0 * float(np.sum(rho[: max_t + 1])) + float(
        np.sum(rho[max_t + 1: max_t + 2])
    )
    tau = max(tau, tau_floor)
    return float(n / tau)


def trace_summary(chains: np.ndarray) -> dict:
    """Summary stats across an (M, n) bundle of chains.

    Keys: ``n_chains``, ``n_iter``, ``mean``, ``std`` (pooled, ddof=1),
    ``geweke_z`` (per chain, AR spectral estimate), ``ess_per_chain``
    (Geyer initial monotone sequence), ``R_hat_split`` (split-R̂, defined
    for any M when ``n >= 4``; the statistic to judge convergence) and,
    when ``M >= 2``, ``R_hat`` (the classic unsplit statistic, kept for
    backward compatibility; it cannot see a drift shared by all chains).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        chains = chains[None, :]
    M, n = chains.shape
    out = {
        "n_chains": M,
        "n_iter": n,
        "mean": float(chains.mean()),
        "std": float(chains.std(ddof=1)),
        "geweke_z": [float(geweke_z(c)) for c in chains],
        "ess_per_chain": [float(effective_sample_size(c)) for c in chains],
    }
    if M >= 2:
        out["R_hat"] = gelman_rubin(chains)["R_hat"]
    if n >= 4:
        out["R_hat_split"] = gelman_rubin(chains, split=True)["R_hat"]
    return out


def _rw_metropolis_step(
    x: np.ndarray,
    lp: float,
    c: float,
    L: np.ndarray,
    rng: np.random.Generator,
    log_posterior_fn,
) -> tuple[np.ndarray, float, bool]:
    """Perform a single Random-Walk Metropolis step.

    Returns (new_x, new_lp, accepted)
    """
    n_params = len(x)
    z = rng.standard_normal(n_params)
    x_new = x + c * (L @ z)
    lp_new = log_posterior_fn(x_new)
    log_alpha = lp_new - lp
    if np.log(rng.uniform()) < log_alpha:
        return x_new, lp_new, True
    return x, lp, False


def random_walk_metropolis(
    log_posterior_fn,
    init,
    proposal_cov,
    n_draws: int,
    *,
    seed: int = 0,
    accept_target: float = 0.25,
    adapt_burnin: int = 0,
) -> dict:
    """Random-Walk Metropolis-Hastings with optional scalar-c proposal adaptation.

    Parameters
    ----------
    log_posterior_fn : callable(np.ndarray) -> float
        Target log-density (may return -np.inf).
    init : np.ndarray, shape (n_params,)
        Initial parameter vector.
    proposal_cov : np.ndarray, shape (n_params, n_params)
        Base proposal covariance. Actual proposal at iteration t is
        N(0, c**2 * proposal_cov) where c is adapted during adapt_burnin
        (no covariance adaptation; only scalar c).
    n_draws : int
        Post-burn-in iterations to retain.
    seed : int
        RNG seed.
    accept_target : float, default 0.25
        Adaptation target acceptance rate (only used if adapt_burnin > 0).
    adapt_burnin : int, default 0
        Number of additional iterations BEFORE n_draws used for scalar-c
        adaptation. Every 100 iterations, scale c by 1.1 if recent
        accept-rate > accept_target * 1.2, divide by 1.1 if < accept_target * 0.8.
        c is frozen at the end of burn-in.

    Returns
    -------
    dict with keys:
        chain       : (n_draws, n_params) — post-burn-in samples
        log_post    : (n_draws,)          — log-density at each retained sample
        accept_rate : float               — over the n_draws iterations only
        final_scale : float               — scalar c after adaptation (1.0 if adapt_burnin=0)
    """
    init = np.asarray(init, dtype=float).ravel()
    n_params = len(init)
    cov = np.asarray(proposal_cov, dtype=float)
    if cov.shape != (n_params, n_params):
        raise ValueError(
            f"proposal_cov shape {cov.shape} doesn't match init length {n_params}"
        )
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-10 * np.eye(n_params)
        L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(seed)

    x = init.copy()
    lp = log_posterior_fn(x)
    c = 1.0
    if not np.isfinite(lp):
        raise RuntimeError(
            "log_posterior_fn(init) is not finite; pick a better starting point"
        )

    # Adaptation phase.
    adapt_window = 100
    adapt_accepts = 0
    adapt_count = 0
    for it in range(adapt_burnin):
        x, lp, accepted = _rw_metropolis_step(x, lp, c, L, rng, log_posterior_fn)
        if accepted:
            adapt_accepts += 1
        adapt_count += 1
        if (it + 1) % adapt_window == 0:
            rate = adapt_accepts / adapt_count
            if rate > accept_target * 1.2:
                c *= 1.1
            elif rate < accept_target * 0.8:
                c /= 1.1
            adapt_accepts = 0
            adapt_count = 0

    # Retained chain.
    chain = np.empty((n_draws, n_params))
    log_post = np.empty(n_draws)
    accepts = 0
    for it in range(n_draws):
        x, lp, accepted = _rw_metropolis_step(x, lp, c, L, rng, log_posterior_fn)
        if accepted:
            accepts += 1
        chain[it] = x
        log_post[it] = lp

    return {
        "chain": chain,
        "log_post": log_post,
        "accept_rate": accepts / n_draws,
        "final_scale": c,
    }


__all__ = [
    "geweke_z", "gelman_rubin",
    "autocorrelations", "effective_sample_size",
    "trace_summary",
    "random_walk_metropolis",
]
