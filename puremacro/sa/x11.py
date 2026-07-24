"""Native X-11/ARIMA seasonal adjustment — pure Python, Pyodide-safe.

The X-11 decomposition engine (Shiskin-Young-Musgrave, as documented
table-by-table in Ladiray & Quenneville 2001, *Seasonal Adjustment with
the X-11 Method*) with an airline-model regARIMA pre-adjustment
(:mod:`puremacro.sa._airline`) — the core of X-13ARIMA-SEATS' X-11
mode, reimplemented natively so the Pyodide build gets a genuine
X-11-class adjuster instead of the STL fallback. Validated against the
real X-13ARIMA-SEATS binary via frozen goldens
(``tools/gen_validation_goldens_sa.py``; see docs/VALIDATION.md).

Honest naming: this is "X-11/ARIMA-style, X-13-validated" — the v1
scope is the airline model (X-13's default fit for most series), the
automatic log/level decision, and the full B/C/D filter iteration with
MSR seasonal-filter and I/C Henderson-length selection. NOT implemented
(v1): TRAMO automdl beyond the airline model, trading-day / Easter
regressors, outlier regressors (flagged for v2), SEATS.

End handling (pre-registered v1 decision, batch-6 plan): the series is
extended with 60 periods of airline fore/backcasts so every ORIGINAL
observation receives fully symmetric filters (2x12 needs 6, Henderson-23
needs 11, the 3x9 seasonal MA needs 5 years). The binary's default is a
1-year extension plus asymmetric end filters, so agreement with the
binary is tight in the interior and looser in the first/last two years
— the golden tolerances encode exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._airline import AirlineFit, choose_log, extend, fit_airline

__all__ = ["X11Result", "x11_arima", "deseasonalize_x11",
           "henderson_weights"]

_SEASONAL_MA = {
    "3x3": np.convolve(np.ones(3) / 3, np.ones(3) / 3),
    "3x5": np.convolve(np.ones(3) / 3, np.ones(5) / 5),
    "3x9": np.convolve(np.ones(3) / 3, np.ones(9) / 9),
}
_EXTENSION = 60          # periods of ARIMA fore/backcast on each side


@dataclass(frozen=True)
class X11Result:
    """Final D-table outputs of the native X-11/ARIMA decomposition,
    aligned 1:1 with the input series (extension trimmed)."""

    sa: np.ndarray            # D11 seasonally adjusted
    trend: np.ndarray         # D12 Henderson trend
    seasonal: np.ndarray      # D10 seasonal factors
    irregular: np.ndarray     # D13
    mode: str                 # 'mult' | 'add'
    period: int
    seasonal_filter: str      # '3x3' | '3x5' | '3x9' (MSR-selected)
    henderson: int            # final trend-filter length
    msr: float
    ic_ratio: float
    arima: AirlineFit
    transform_diag: dict


def henderson_weights(n: int) -> np.ndarray:
    """Henderson filter weights for odd length n (closed form)."""
    if n % 2 != 1 or n < 3:
        raise ValueError(f"henderson_weights: n must be odd >= 3, got {n}")
    m = (n - 1) // 2
    p = m + 2
    j = np.arange(-m, m + 1, dtype=float)
    num = (315.0 * ((p - 1) ** 2 - j ** 2) * (p ** 2 - j ** 2)
           * ((p + 1) ** 2 - j ** 2) * (3 * p ** 2 - 16 - 11 * j ** 2))
    den = (8.0 * p * (p ** 2 - 1) * (4 * p ** 2 - 1) * (4 * p ** 2 - 9)
           * (4 * p ** 2 - 25))
    return num / den


def _ma_sym(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Symmetric weighted MA; NaN where the window is incomplete."""
    half = (len(w) - 1) // 2
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) >= len(w):
        out[half:len(x) - half] = np.convolve(x, w[::-1], mode="valid")
    return out


def _centered_2xp(x: np.ndarray, period: int) -> np.ndarray:
    w = np.ones(period + 1)
    w[0] = w[-1] = 0.5
    return _ma_sym(x, w / period)


def _detrend(y: np.ndarray, trend: np.ndarray, mode: str) -> np.ndarray:
    return y / trend if mode == "mult" else y - trend


def _deseas(y: np.ndarray, seasonal: np.ndarray, mode: str) -> np.ndarray:
    return y / seasonal if mode == "mult" else y - seasonal


def _seasonal_ma(si: np.ndarray, period: int, name: str) -> np.ndarray:
    """Composite seasonal MA applied per calendar position across years,
    with same-value end padding (ends are covered by the ARIMA extension
    for the original span, so this only affects the extension's rim)."""
    w = _SEASONAL_MA[name]
    half = (len(w) - 1) // 2
    out = np.full_like(si, np.nan, dtype=float)
    for k in range(period):
        col = si[k::period]
        padded = np.concatenate([np.full(half, col[0]), col,
                                 np.full(half, col[-1])])
        out[k::period] = np.convolve(padded, w[::-1], mode="valid")
    return out


def _normalize(factors: np.ndarray, period: int, mode: str) -> np.ndarray:
    """Center the factors so they average out over any year: divide by
    (mult) / subtract (add) their centered 2xperiod MA, edge-filled."""
    c = _centered_2xp(factors, period)
    # fill the rim by nearest valid value (extension rim only)
    idx = np.where(np.isfinite(c))[0]
    c[: idx[0]] = c[idx[0]]
    c[idx[-1] + 1:] = c[idx[-1]]
    return factors / c if mode == "mult" else factors - c


def _replace_extremes(si: np.ndarray, prelim_seasonal: np.ndarray,
                      period: int, mode: str, orig: np.ndarray,
                      lo: float = 1.5, hi: float = 2.5) -> np.ndarray:
    """Sigma-limit extreme-value replacement of the SI ratios.

    Irregular deviations are measured against the preliminary seasonal;
    the sigma is a per-year five-year-span standard deviation; points
    get weight 1 below lo*sigma, 0 above hi*sigma, linear between; low-
    weight SIs are replaced by the weighted mean of themselves and up to
    two nearest full-weight same-calendar-position neighbours per side.

    ``orig`` marks the ORIGINAL data span: the ARIMA-extension periods
    are model-smooth, so letting them into the sigma windows collapses
    the sigma toward zero and the whole sample gets 'replaced' toward
    the model's seasonal pattern (a bug this argument exists to
    prevent). Extension SIs are excluded from the sigma, are never
    flagged, and are not used as replacement donors."""
    irr = si / prelim_seasonal if mode == "mult" else si - prelim_seasonal
    wgt = _irregular_weights(irr, period, mode, orig, lo=lo, hi=hi)
    return _replace_si(si, wgt, period, orig)


def _year_block_sigma(d: np.ndarray, period: int) -> np.ndarray:
    """X-11 sigma convention: one sigma per YEAR, computed over the
    centered five-year span (edge years borrow the nearest interior
    span). NaNs in d are ignored."""
    n = len(d)
    n_years = int(np.ceil(n / period))
    sig_y = np.full(n_years, np.nan)
    for i in range(n_years):
        lo_y, hi_y = max(0, i - 2), min(n_years, i + 3)
        block = d[lo_y * period: hi_y * period]
        block = block[np.isfinite(block)]
        if len(block) >= 2 * period:
            sig_y[i] = float(np.sqrt(np.mean(block ** 2)))
    ok = np.where(np.isfinite(sig_y))[0]
    if len(ok) == 0:
        fallback = float(np.sqrt(np.nanmean(d ** 2)))
        sig_y[:] = fallback if np.isfinite(fallback) and fallback > 0 \
            else np.inf
    else:
        sig_y[: ok[0]] = sig_y[ok[0]]
        sig_y[ok[-1] + 1:] = sig_y[ok[-1]]
        for i in range(len(sig_y)):
            if not np.isfinite(sig_y[i]):
                sig_y[i] = sig_y[i - 1]
    return np.repeat(sig_y, period)[:n]


def _irregular_weights(irr: np.ndarray, period: int, mode: str,
                       orig: np.ndarray, lo: float = 1.5,
                       hi: float = 2.5) -> np.ndarray:
    """Sigma-limit weights on an irregular series (the B17/C17 tables).

    Two rounds, as X-11 does: provisional sigma -> drop >hi*sigma
    points -> final sigma -> weights (1 below lo*sigma, 0 above
    hi*sigma, linear between). ARIMA-extension periods are excluded
    from the sigma and always get weight 1."""
    dev = irr - 1.0 if mode == "mult" else irr.copy()
    dev = np.where(orig, dev, np.nan)
    sig1 = _year_block_sigma(dev, period)
    dev_clean = dev.copy()
    dev_clean[np.abs(dev) > hi * sig1] = np.nan
    sig = _year_block_sigma(dev_clean, period)
    with np.errstate(invalid="ignore"):
        z = np.abs(dev) / np.where(sig > 0, sig, np.inf)
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip((hi - z) / (hi - lo), 0.0, 1.0)


def _replace_si(si: np.ndarray, wgt: np.ndarray, period: int,
                orig: np.ndarray) -> np.ndarray:
    """Donor-average replacement of low-weight SIs: the point at its own
    weight plus up to two nearest full-weight same-calendar-position
    neighbours per side (original span only, on both sides)."""
    n = len(si)
    out = si.copy()
    low = np.where((wgt < 1.0) & orig)[0]
    for t in low:
        same = np.arange(t % period, n, period)
        full = same[(wgt[same] >= 1.0) & orig[same]]
        before = full[full < t][-2:]
        after = full[full > t][:2]
        neigh = np.concatenate([before, after])
        vals = np.concatenate([[si[t]], si[neigh]])
        ws = np.concatenate([[wgt[t]], np.ones(len(neigh))])
        if ws.sum() > 0:
            out[t] = float(vals @ ws / ws.sum())
    return out


def _weight_adjustment(irr: np.ndarray, wgt: np.ndarray,
                       mode: str) -> np.ndarray:
    """The B20/C20 adjustment: the extreme part of the irregular.

    The modified irregular shrinks toward neutral by its weight,
    I* = 1 + w (I - 1) (mult) / w I (add); the adjustment is what
    division (subtraction) by it removes from the original series:
    adj = I / I* (mult), I - I* (add). Weight-1 points get a neutral
    adjustment exactly."""
    if mode == "mult":
        i_star = 1.0 + wgt * (irr - 1.0)
        i_star = np.where(i_star > 0, i_star, np.nan)
        adj = irr / i_star
        return np.where(np.isfinite(adj), adj, 1.0)
    adj = irr - wgt * irr
    return np.where(np.isfinite(adj), adj, 0.0)


def _seasonal_pass(y_ext: np.ndarray, trend: np.ndarray, period: int,
                   mode: str, filter_name: str, orig: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """One SI -> (replaced-extremes) -> seasonal-factor pass.
    Returns (normalized seasonal factors, replaced SI)."""
    si = _detrend(y_ext, trend, mode)
    prelim = _normalize(_seasonal_ma(si, period, "3x3"), period, mode)
    si_rep = _replace_extremes(si, prelim, period, mode, orig)
    seas = _normalize(_seasonal_ma(si_rep, period, filter_name),
                      period, mode)
    return seas, si_rep


def _pct_changes(x: np.ndarray, lag: int, mode: str) -> np.ndarray:
    a, b = x[lag:], x[:-lag]
    return np.abs(a / b - 1.0) if mode == "mult" else np.abs(a - b)


def _ic_ratio(sa: np.ndarray, period: int, mode: str) -> float:
    trend13 = _ma_sym(sa, henderson_weights(13 if period == 12 else 5))
    m = np.isfinite(trend13)
    irr = _deseas(sa[m], trend13[m], mode)
    ibar = float(np.mean(_pct_changes(irr, 1, mode)))
    cbar = float(np.mean(_pct_changes(trend13[m], 1, mode)))
    return ibar / cbar if cbar > 0 else np.inf


def _henderson_length(ic: float, period: int) -> int:
    if period == 12:
        return 9 if ic < 1.0 else (13 if ic < 3.5 else 23)
    return 5 if ic < 1.0 else 7


def _msr(si_rep: np.ndarray, seasonal: np.ndarray, period: int,
         mode: str) -> float:
    """Global moving-seasonality ratio, X-11 D9.A convention: per-month
    mean absolute year-on-year changes of the (extreme-REPLACED)
    irregular over those of the seasonal, averaged across months.
    Callers must pass ORIGINAL-span arrays only — the ARIMA extension is
    model-smooth in both I and S and corrupts the ratio."""
    irr = si_rep / seasonal if mode == "mult" else si_rep - seasonal
    ibars, sbars = [], []
    for k in range(period):
        ik, sk = irr[k::period], seasonal[k::period]
        if len(ik) < 2:
            continue
        ibars.append(np.mean(_pct_changes(ik, 1, mode)))
        sbars.append(np.mean(_pct_changes(sk, 1, mode)))
    sbar = float(np.mean(sbars))
    return float(np.mean(ibars)) / sbar if sbar > 0 else np.inf


def _select_seasonal_filter(si_rep: np.ndarray, seasonal: np.ndarray,
                            period: int, mode: str) -> tuple[str, float]:
    """Global MSR rule with the classic cut-a-year retry (<= 5 cuts)."""
    si_c, se_c = si_rep.copy(), seasonal.copy()
    msr = _msr(si_c, se_c, period, mode)
    for _ in range(5):
        if msr < 2.5:
            return "3x3", msr
        if 3.5 <= msr < 5.5:
            return "3x5", msr
        if msr >= 6.5:
            return "3x9", msr
        if len(si_c) <= 7 * period:
            break
        si_c, se_c = si_c[:-period], se_c[:-period]
        msr = _msr(si_c, se_c, period, mode)
    return "3x5", msr


def x11_arima(values, dates=None, *, period: int, mode: str = "auto",
              extension: int = _EXTENSION) -> X11Result:
    """Native X-11 decomposition with airline-ARIMA extension.

    Parameters
    ----------
    values : array-like, gap-free (interpolate upstream).
    period : 12 (monthly) or 4 (quarterly).
    mode : 'auto' (X-13-style log/level AICC decision -> mult/add),
           'mult' (requires strictly positive values) or 'add'.
    extension : fore/backcast periods per side (default 60 so every
           original point gets fully symmetric filters).
    """
    y = np.asarray(values, dtype=float)
    if np.isnan(y).any():
        raise ValueError("x11_arima: series must be gap-free")
    if period not in (4, 12):
        raise ValueError(f"x11_arima: period must be 4 or 12, got {period}")

    tdiag: dict = {}
    if mode == "auto":
        use_log, tdiag = choose_log(y, period)
        mode = "mult" if use_log else "add"
    elif mode == "mult" and (y <= 0).any():
        raise ValueError("x11_arima: mode='mult' needs positive values")
    elif mode not in ("mult", "add"):
        raise ValueError(f"x11_arima: unknown mode {mode!r}")

    z = np.log(y) if mode == "mult" else y
    fit = fit_airline(z, period)
    z_ext = extend(z, fit, extension, extension)
    y_ext = np.exp(z_ext) if mode == "mult" else z_ext
    sl = slice(extension, extension + len(y))
    orig_ext = np.zeros(len(y_ext), dtype=bool)
    orig_ext[sl] = True

    # --- the B -> C -> D weight cascade ---------------------------------------
    # X-11's defining structure (Ladiray-Quenneville): each stage
    # computes extreme-value weights from ITS irregular (B17/C17), forms
    # the adjustment table (B20/C20) and hands the next stage a
    # weight-MODIFIED ORIGINAL (C1/D1) whose trends no longer chase the
    # extremes. SA and the irregular are always measured against the
    # true original B1. Batch-7 measurements showed this cascade — not
    # end filters — is what separated v1 from the binary on wild series.
    n_ext = len(y_ext)
    neutral = 1.0 if mode == "mult" else 0.0

    def bstage(y_mod: np.ndarray, seasonal_filter: str):
        """B/C-type pass: trends from the (modified) series, SA and
        irregular against the original. Returns full-grid indices of
        the stage core plus the per-core tables."""
        t2 = _centered_2xp(y_mod, period)
        m2 = np.isfinite(t2)
        idx2 = np.where(m2)[0]
        yb, tb, ob = y_mod[m2], t2[m2], orig_ext[m2]
        seas5, _ = _seasonal_pass(yb, tb, period, mode, "3x3", ob)
        sa6 = _deseas(yb, seas5, mode)
        ic = _ic_ratio(sa6, period, mode)
        h_len = _henderson_length(ic, period)
        t7 = _ma_sym(sa6, henderson_weights(h_len))
        m7 = np.isfinite(t7)
        idx = idx2[m7]
        seas10, _ = _seasonal_pass(yb[m7], t7[m7], period, mode,
                                   seasonal_filter, ob[m7])
        sa11 = _deseas(y_ext[idx], seas10, mode)
        irr13 = _deseas(sa11, t7[m7], mode)
        return idx, ic, h_len, irr13

    def modified_original(idx: np.ndarray, irr13: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
        wgt = _irregular_weights(irr13, period, mode, orig_ext[idx])
        adj = _weight_adjustment(irr13, wgt, mode)
        adj_full = np.full(n_ext, neutral)
        adj_full[idx] = adj
        y_mod = y_ext / adj_full if mode == "mult" else y_ext - adj_full
        return y_mod, wgt

    # B stage on the original; C1 = B1 tempered by B20.
    idxB, _, _, irr13B = bstage(y_ext, "3x5")
    c1, _ = modified_original(idxB, irr13B)
    # C stage on C1; final weights C17 gate the D-stage replacement.
    idxC, _, _, irr13C = bstage(c1, "3x5")
    d1, wC = modified_original(idxC, irr13C)
    w17_full = np.ones(n_ext)
    w17_full[idxC] = wC

    # --- D stage: trends from D1, final SI from the ORIGINAL ------------------
    t2D = _centered_2xp(d1, period)
    m2D = np.isfinite(t2D)
    idx2D = np.where(m2D)[0]
    yD, tD, oD = d1[m2D], t2D[m2D], orig_ext[m2D]
    seas5D, _ = _seasonal_pass(yD, tD, period, mode, "3x3", oD)
    sa6D = _deseas(yD, seas5D, mode)
    ic_d = _ic_ratio(sa6D, period, mode)
    h_d = _henderson_length(ic_d, period)
    t7D = _ma_sym(sa6D, henderson_weights(h_d))
    m7D = np.isfinite(t7D)
    idxD = idx2D[m7D]
    orig_d = orig_ext[idxD]
    d8 = _detrend(y_ext[idxD], t7D[m7D], mode)     # final unmodified SI
    d9 = _replace_si(d8, w17_full[idxD], period, orig_d)
    seas_d5 = _normalize(_seasonal_ma(d9, period, "3x5"), period, mode)
    # MSR (D9.A): REPLACED irregular vs the 3x5 preliminary seasonal,
    # original span only.
    fname, msr = _select_seasonal_filter(d9[orig_d], seas_d5[orig_d],
                                         period, mode)
    seas_d = (seas_d5 if fname == "3x5"
              else _normalize(_seasonal_ma(d9, period, fname),
                              period, mode))
    sa_d = _deseas(y_ext[idxD], seas_d, mode)
    trend_fin = _ma_sym(sa_d, henderson_weights(h_d))
    irr_fin = _deseas(sa_d, trend_fin, mode)

    # --- trim back to the original span --------------------------------------
    off3 = int(idxD[0])

    def cut(arr: np.ndarray) -> np.ndarray:
        lo = sl.start - off3
        return arr[lo: lo + len(y)]

    return X11Result(
        sa=cut(sa_d), trend=cut(trend_fin),
        seasonal=cut(seas_d), irregular=cut(irr_fin),
        mode=mode, period=period, seasonal_filter=fname,
        henderson=h_d, msr=float(msr), ic_ratio=float(ic_d),
        arima=fit, transform_diag=tdiag,
    )


def deseasonalize_x11(df: pd.DataFrame, value_col: str, *, by: str,
                      date_col: str = "date", freq: str = "Q",
                      min_obs: int = 24) -> pd.Series:
    """Group-wise native X-11: same contract as
    :func:`puremacro.sa.stl.deseasonalize` / ``deseasonalize_x13``.
    Units below ``min_obs`` (or that fail, e.g. too short for the
    airline fit) are returned as NaN."""
    period = {"Q": 4, "M": 12}.get(freq)
    if period is None:
        raise ValueError(f"freq must be 'Q' or 'M', got {freq!r}")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    work = df[[by, date_col, value_col]].copy()
    work["_orig_idx"] = df.index
    work = work.sort_values([by, date_col])
    for _, sub in work.groupby(by, sort=False):
        v = sub[value_col].to_numpy(dtype=float)
        if np.isfinite(v).sum() < max(min_obs, 3 * period + period + 1):
            continue
        s = pd.Series(v).interpolate(limit_direction="both")
        try:
            res = x11_arima(s.to_numpy(), period=period)
        except (ValueError, np.linalg.LinAlgError):
            continue
        sa = np.where(np.isfinite(v), res.sa, np.nan)
        out.loc[sub["_orig_idx"].to_numpy()] = sa
    return out
