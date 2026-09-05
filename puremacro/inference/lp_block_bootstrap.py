"""Cumulative-IRF bands for panel LP via the entity (cluster) bootstrap.

Per-horizon `panel_lp` returns Driscoll-Kraay HAC SE on `β_h`. The
cumulative IRF $C_H = \\sum_{h=0}^H β_h$ has analytic SE that depends on
cross-horizon covariances, which `PanelOLS` does not surface; a naive
`cumsum(lo_h)` gives only an upper bound on the true CI.

We use the **entity bootstrap** (Cameron-Gelbach-Miller 2008): sample
entities with replacement, refit the panel LP at each horizon, read
percentile bands off `cumsum(β_h^{(b)})`. The cluster (entity) bootstrap
is asymptotically valid for panel LPs under cross-sectional independence
of within-entity errors and is robust to unbalanced panels — every
sampled entity contributes its full time series, so lag structure is
preserved automatically.

For applications with strong common time shocks (e.g. global oil prices)
we recommend re-running with ``time_effects=True`` in the underlying LP;
the cluster bootstrap remains valid because the time-FE residuals are
still uncorrelated across entities under the maintained assumption.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# puremacro.teaching.panel_lm.panel_lp uses linearmodels.PanelOLS — lazy-imported
# inside cum_irf_block_bootstrap below (Pyodide contract; see ARCHITECTURE.md).


def cum_irf_block_bootstrap(
    df_wide: pd.DataFrame,
    *,
    y: str,
    x: str,
    horizons: Iterable[int],
    n_lags: int = 2,
    controls: Sequence[str] = (),
    B: int = 200,
    alpha: float = 0.10,
    seed: int = 42,
    time_effects: bool = False,
    band: str = "pointwise",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Cumulative-IRF bands via the entity (cluster) bootstrap.

    Parameters
    ----------
    df_wide
        Wide panel indexed by (code, date) with columns `y`, `x`, controls.
    y, x, horizons, n_lags, controls, time_effects
        Forwarded to :func:`src.teaching.panel_lm.panel_lp`.
    B
        Bootstrap replications. Default 200; cut to 100 for speed.
    alpha
        Two-sided level (``alpha=0.10`` → 90% CI).
    seed
        RNG seed.
    band
        'pointwise' (default; per-horizon percentile bands, the historical
        behavior) or 'sup-t' (Montiel Olea & Plagborg-Møller 2019, Journal
        of Applied Econometrics 34(1)): a *simultaneous* band across all
        requested horizons, computed by the quantile-of-max-studentized-
        deviation of the bootstrap draws of ``cumsum(β_h^{(b)})`` around
        the original-sample ``cum_beta`` (the paper's Algorithm 2). The
        sup-t band covers the whole cumulative-IRF path jointly with
        probability 1−α; the pointwise band only covers each horizon
        separately.
    n_jobs
        Number of parallel worker threads. Default 1. Set to -1 to use all CPUs.

    Returns
    -------
    DataFrame with columns ``h, beta, cum_beta, cum_lo, cum_hi``.
    `beta`/`cum_beta` are point estimates from the original sample.
    With ``band='pointwise'``, `cum_lo`/`cum_hi` are α/2 and 1−α/2
    percentiles of the bootstrap distribution of `cumsum(β_h^{(b)})`.
    With ``band='sup-t'``, `cum_lo`/`cum_hi` are ``cum_beta ± c·sd_boot``
    with the sup-t critical value ``c`` stored (additively) in
    ``result.attrs['supt_crit']``; ``result.attrs['band']`` records the
    band type in both cases.
    """
    if band not in ("pointwise", "sup-t"):
        raise ValueError(f"unknown band {band!r}; use 'pointwise' or 'sup-t'")
    from puremacro.teaching.panel_lm import panel_lp  # lazy: Pyodide contract

    horizons = list(horizons)
    rng = np.random.default_rng(seed)

    point = panel_lp(df_wide, y=y, x=x, horizons=horizons,
                     n_lags=n_lags, controls=list(controls),
                     time_effects=time_effects)
    point_betas = point["beta"].to_numpy()

    entities = list(df_wide.index.get_level_values(0).unique())
    n_e = len(entities)
    if n_e < 2:
        raise ValueError(f"cluster bootstrap needs ≥2 entities; got {n_e}")

    def _fit_draw(sampled) -> np.ndarray:
        pieces: list[pd.DataFrame] = []
        for k, e in enumerate(sampled):
            piece = df_wide.xs(e, level=0).copy()
            piece.index = pd.MultiIndex.from_tuples(
                [(f"_b{k}_{e}", t) for t in piece.index],
                names=df_wide.index.names,
            )
            pieces.append(piece)
        boot_panel = pd.concat(pieces).sort_index()

        try:
            res_b = panel_lp(boot_panel, y=y, x=x, horizons=horizons,
                             n_lags=n_lags, controls=list(controls),
                             time_effects=time_effects)
            return res_b["beta"].to_numpy()
        except Exception:
            return np.full(len(horizons), np.nan)

    all_sampled = [rng.choice(entities, size=n_e, replace=True) for _ in range(B)]
    if n_jobs == 1:
        boot = np.array([_fit_draw(s) for s in all_sampled])
    else:
        import concurrent.futures
        import os

        workers = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            boot = np.array(list(ex.map(_fit_draw, all_sampled)))

    cum_boot = np.nancumsum(boot, axis=1)
    cum_point = np.cumsum(point_betas)
    supt_crit: float | None = None
    if band == "pointwise":
        lo = np.nanpercentile(cum_boot, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(cum_boot, 100 * (1 - alpha / 2), axis=0)
    else:  # 'sup-t': joint band across horizons (MOP 2019, Algorithm 2)
        from .supt import supt_band

        good = cum_boot[np.isfinite(cum_boot).all(axis=1)]
        if good.shape[0] < 20:
            raise ValueError(
                f"sup-t band needs >= 20 fully-finite bootstrap draws; got "
                f"{good.shape[0]} of B={B} (failed refits leave NaN rows — "
                f"increase B or inspect the panel)"
            )
        res = supt_band(theta=cum_point, draws=good,
                        method="bootstrap", alpha=alpha)
        lo, hi = res.lower, res.upper
        supt_crit = res.crit_value

    out = pd.DataFrame({
        "h": horizons,
        "beta": point_betas,
        "cum_beta": cum_point,
        "cum_lo": lo,
        "cum_hi": hi,
    })
    out.attrs["band"] = band
    if supt_crit is not None:
        out.attrs["supt_crit"] = supt_crit
    return out
