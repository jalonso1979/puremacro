"""Causal price indicators and selection-aware comparison of supplied strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._research import ResearchReport, dated_frame, finite, integer, readonly

__all__ = ["technical_indicators", "momentum_weights", "StrategyComparisonResult", "compare_strategies"]


def _seeded_average(values: np.ndarray, period: int, alpha: float) -> np.ndarray:
    """SMA initialization followed by a recursive exponential mean."""
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < period:
        return out
    start = int(valid[0])
    if not np.isfinite(values[start:]).all():
        raise ValueError("internal indicator input has gaps after initialization")
    first = start + period - 1
    out[first] = values[start:first+1].mean()
    for i in range(first+1, len(values)):
        out[i] = alpha*values[i] + (1-alpha)*out[i-1]
    return out


def technical_indicators(
    prices: pd.DataFrame | pd.Series, *, fast: int = 12, slow: int = 26,
    signal: int = 9, rsi_period: int = 14, lookback: int = 20, band_width: float = 2,
) -> pd.DataFrame:
    """Return (asset, indicator) columns on the original price dates.

    Positive finite prices are required; adjustments are the caller's choice.
    SMA/EMA use the current close and earlier observations. EMA uses an initial
    arithmetic mean and alpha=2/(span+1). MACD is the difference of independently
    initialized fast/slow EMAs; its signal uses the same initialization rule on
    the first valid MACD values (other vendors may seed differently).
    RSI uses Wilder smoothing (alpha=1/period); a flat series is defined as 50.
    Bollinger bands use a population standard deviation (ddof=0). Breakout
    thresholds exclude the current close. Warm-up outputs stay NaN. These are
    observations at the close, not executable same-close trading recommendations.
    """
    p = dated_frame(prices, "prices")
    if np.any(p.to_numpy() <= 0):
        raise ValueError("prices must be positive")
    fast, slow = integer(fast,"fast"), integer(slow,"slow")
    signal, rsi_period = integer(signal,"signal"), integer(rsi_period,"rsi_period")
    lookback = integer(lookback,"lookback")
    width = finite(band_width,"band_width",0)
    if fast >= slow:
        raise ValueError("fast must be below slow")
    columns: dict[tuple[str, str], Any] = {}
    for name in p:
        s = p[name]; x = s.to_numpy()
        ef = _seeded_average(x, fast, 2/(fast+1))
        es = _seeded_average(x, slow, 2/(slow+1))
        macd = ef-es
        sig = _seeded_average(macd, signal, 2/(signal+1))
        delta = np.r_[np.nan, np.diff(x)]
        gain = _seeded_average(np.maximum(delta,0), rsi_period, 1/rsi_period)
        loss = _seeded_average(np.maximum(-delta,0), rsi_period, 1/rsi_period)
        total = gain+loss
        rsi = np.divide(100*gain, total, out=np.full(len(x),50.), where=total!=0)
        mid, std = s.rolling(lookback).mean(), s.rolling(lookback).std(ddof=0)
        values = {"sma_fast": s.rolling(fast).mean(), "sma_slow": s.rolling(slow).mean(),
                  "ema_fast": ef, "ema_slow": es, "macd": macd, "macd_signal": sig,
                  "macd_histogram": macd-sig, "rsi": rsi, "momentum": s/s.shift(lookback)-1,
                  "band_mid": mid, "band_upper": mid+width*std, "band_lower": mid-width*std,
                  "breakout_high": s.shift(1).rolling(lookback).max(),
                  "breakout_low": s.shift(1).rolling(lookback).min()}
        columns.update({(name,key): value for key,value in values.items()})
    out = pd.DataFrame(columns,index=p.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["asset","indicator"])
    return out


def momentum_weights(prices: pd.DataFrame, *, lookback: int = 20,
                     top_n: int | None = None, positive_only: bool = True) -> pd.Series:
    """Equal-weight eligible momentum winners at the final observed close.

    Eligibility requires positive, complete prices throughout the trailing
    window. No winner means all cash. Ties are resolved by asset name so column
    ordering does not change a top-N selection. Returns unshifted weights for
    use inside walk_forward_weights; the backtester supplies execution timing.
    """
    p = dated_frame(prices, "prices", missing=True)
    period = integer(lookback,"lookback")
    if top_n is not None:
        top_n = integer(top_n,"top_n")
    out = pd.Series(0., index=p.columns, name="weight")
    if len(p) <= period:
        return out
    sample = p.iloc[-period-1:]
    valid = sample.notna().all() & (sample > 0).all()
    scores = sample.iloc[-1]/sample.iloc[0]-1
    candidates = [name for name in p if valid[name] and (not positive_only or scores[name] > 0)]
    candidates.sort(key=lambda name: (-scores[name], name))
    selected = candidates if top_n is None else candidates[:top_n]
    if selected:
        out.loc[selected] = 1/len(selected)
    return out


@dataclass(frozen=True)
class StrategyComparisonResult(ResearchReport):
    names: tuple[str, ...]
    mean_difference: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    adjusted_pvalue: np.ndarray
    global_pvalue: float
    n_obs: int
    n_boot: int
    block_length: int
    confidence: float
    seed: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"strategy": self.names, "mean_difference": self.mean_difference,
                             "ci_lower_pointwise": self.ci_lower, "ci_upper_pointwise": self.ci_upper,
                             "pvalue_max_test": self.adjusted_pvalue})


def compare_strategies(
    strategy_returns: pd.DataFrame, benchmark: pd.Series, *, block_length: int = 10,
    n_boot: int = 999, confidence: float = .95, seed: int = 0,
) -> StrategyComparisonResult:
    """Compare per-period net returns with a centered circular block max test.

    Paired resampling preserves cross-strategy dependence and within-block time
    dependence. Every candidate is centered at its sample mean, giving the
    least-favorable joint null of zero mean advantage (a nonstudentized White
    Reality-Check-style statistic). The global one-sided null is that no supplied
    strategy beats the benchmark in arithmetic mean net return. Max-test p-values
    account for the supplied candidate family; confidence intervals are pointwise.
    Finite Monte Carlo p-values use (exceedances+1)/(B+1).

    This is asymptotic bootstrap inference under suitable stationary, weakly
    dependent returns. It does not account for omitted tuning attempts, structural
    breaks, or costs absent from the inputs. Supply a predeclared, common held-out
    evaluation sample and include all configurations searched on that sample.
    """
    r = dated_frame(strategy_returns, "strategy_returns")
    if not isinstance(benchmark, pd.Series) or not benchmark.index.equals(r.index):
        raise ValueError("benchmark must be a Series on exactly the strategy dates")
    ref = benchmark.to_numpy(dtype=float)
    if not np.isfinite(ref).all():
        raise ValueError("benchmark must be finite")
    length = integer(block_length,"block_length")
    boot = integer(n_boot,"n_boot", 19)
    seed = integer(seed,"seed",0)
    level = finite(confidence,"confidence")
    t = len(r)
    if t < 3 or length >= t or not 0 < level < 1:
        raise ValueError("require T>=3, block_length<T and confidence in (0,1)")
    d = r.to_numpy()-ref[:,None]
    means = d.mean(0)
    centered = d-means
    rng = np.random.default_rng(seed)
    draws = np.empty((boot,d.shape[1]))
    for b in range(boot):
        starts = rng.integers(0,t,size=(t+length-1)//length)
        indices = ((starts[:,None]+np.arange(length)) % t).ravel()[:t]
        draws[b] = centered[indices].mean(0)
    maxima = np.maximum(0,draws.max(1))
    observed = np.maximum(0,means)
    pvalues = (1+(maxima[:,None] >= observed).sum(0))/(boot+1)
    alpha = (1-level)/2
    lower = means-np.quantile(draws,1-alpha,axis=0)
    upper = means-np.quantile(draws,alpha,axis=0)
    return StrategyComparisonResult(tuple(r.columns), readonly(means), readonly(lower),
        readonly(upper), readonly(pvalues), float((1+(maxima>=observed.max()).sum())/(boot+1)),
        t, boot, length, level, seed)
