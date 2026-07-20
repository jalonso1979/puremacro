"""LP-GARCH-IV: estimate the response of macro outcomes to the
*forward-looking* component of GARCH-fitted volatility, using
news-based uncertainty indices as instruments.

Three building blocks:

1. ``fit_argarch`` -- AR(1)-GARCH(1,1) on Δlog of a target series; returns
   the conditional volatility ``sigma`` (level) and the standardized
   innovation ``z`` aligned to the input index.
2. ``fit_argarchx`` -- same model with exogenous covariates entering the
   variance equation: σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} + γ' X_{t-1}.
   Returns the augmented σ plus the γ table with t-stats.
3. ``lp_sigma_iv`` -- per-horizon LP-IV: outcome on the GARCH σ_t,
   instrumented by one or more news-based proxies. Reports OLS-LP and
   IV-LP side-by-side with first-stage F.

Why this exists: σ_t from a univariate GARCH is mechanically endogenous
to past realized shocks (σ²_t = α ε²_{t-1} + β σ²_{t-1} + ω). News
indices capture *expected/forward-looking* uncertainty from a different
information set, so they are candidate instruments for the
news-anticipated portion of σ_t. The IV-LP isolates the response of
outcomes to that news-anticipated component.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class GarchFit:
    """Container for AR(1)-GARCH(1,1) outputs."""
    sigma: pd.Series   # conditional volatility (level), aligned to input index
    z: pd.Series       # standardized innovation ε_t / σ_t
    eps: pd.Series     # raw mean residual
    params: pd.Series  # all model parameters
    summary: str       # full text summary


def _fit_arch_model(y: pd.Series, X: pd.DataFrame | None = None) -> GarchFit:
    """Internal: fit AR(1)-GARCH(1,1) (with optional vol covariates)."""
    from arch import arch_model
    y = pd.Series(y).dropna().astype(float)
    if X is not None:
        X = X.reindex(y.index)
        # drop rows where any covariate is NaN
        common = X.dropna().index.intersection(y.index)
        y = y.loc[common]
        X = X.loc[common]
    am = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1,
                    x=(X.values if X is not None else None),
                    rescale=False)
    res = am.fit(disp="off")
    sigma: pd.Series = pd.Series(res.conditional_volatility).rename("sigma")
    eps: pd.Series = pd.Series(res.resid).rename("eps")
    z = (eps / sigma).rename("z")
    return GarchFit(sigma=sigma, z=z, eps=eps,
                    params=res.params, summary=str(res.summary()))


def fit_argarch(y: pd.Series) -> GarchFit:
    """AR(1)-GARCH(1,1) — baseline univariate model."""
    return _fit_arch_model(y, X=None)


def fit_argarchx(y: pd.Series, X: pd.DataFrame) -> GarchFit:
    """AR(1)-GARCH-X(1,1) — variance equation augmented with X."""
    return _fit_arch_model(y, X=X)


def lp_sigma_iv(
    df: pd.DataFrame,
    *,
    y: str,
    sigma: str,
    instruments: Sequence[str],
    horizons: Iterable[int] = range(0, 21),
    controls: Sequence[str] = (),
    n_lags: int = 2,
    cov_type: str = "robust",
    hac_bandwidth_func=lambda h: h + 4,
) -> pd.DataFrame:
    """LP per horizon: y_{t+h} - y_{t-1} on σ_t (endogenous), instrumented.

    Lags of *controls* (typically log y itself) are added as exogenous
    regressors. Lags of σ are NOT added — they would absorb the very
    variation the instrument is trying to identify.
    """
    from linearmodels.iv import IV2SLS

    df = df.copy().sort_index()
    rows = []
    for h in horizons:
        d = df.copy()
        d["__y_h"] = d[y].shift(-h) - d[y].shift(1)
        ctrl_cols = []
        for v in list(controls):
            for L in range(1, n_lags + 1):
                col = f"{v}_L{L}"
                d[col] = d[v].shift(L)
                ctrl_cols.append(col)
        keep = ["__y_h", sigma] + list(instruments) + ctrl_cols
        sub = d[keep].dropna()
        if len(sub) < 25:
            rows.append({"h": h, "beta_iv": np.nan, "se_iv": np.nan,
                          "beta_ols": np.nan, "se_ols": np.nan,
                          "f_first_stage": np.nan, "j_pvalue": np.nan,
                          "n_obs": len(sub)})
            continue
        sub = sub.assign(const=1.0)
        exog = sub[["const"] + ctrl_cols] if ctrl_cols else sub[["const"]]
        endog = sub[[sigma]]
        instr = sub[list(instruments)]
        beta_iv = se_iv = f_fs = j_p = np.nan
        try:
            kwargs: dict[str, object] = {"cov_type": cov_type}
            if cov_type == "kernel":
                kwargs["kernel"] = "bartlett"
                kwargs["bandwidth"] = int(hac_bandwidth_func(h))
            iv = IV2SLS(sub["__y_h"], exog, endog, instr).fit(**kwargs)  # type: ignore[arg-type]  # linearmodels stubs don't model **dict expansion
            beta_iv = float(iv.params[sigma])
            se_iv = float(iv.std_errors[sigma])
            f_fs = float(iv.first_stage.diagnostics.loc[sigma, "f.stat"])  # type: ignore[attr-defined]  # IV2SLSResults not in stubs
            try:
                j_p = float(iv.j_stat.pval)  # type: ignore[attr-defined]  # IV2SLSResults not in stubs
            except (AttributeError, TypeError):
                j_p = np.nan  # exactly identified or unsupported
        except Exception as e:
            print(f"[lp_sigma_iv h={h}] {e}")
        beta_ols = se_ols = np.nan
        try:
            import statsmodels.api as sm
            X_ols = sm.add_constant(sub[[sigma] + ctrl_cols])
            ols = sm.OLS(sub["__y_h"], X_ols).fit(
                cov_type="HAC", cov_kwds={"maxlags": int(hac_bandwidth_func(h))})
            beta_ols = float(ols.params[sigma])
            se_ols = float(ols.bse[sigma])
        except Exception:
            pass
        rows.append({"h": h, "beta_iv": beta_iv, "se_iv": se_iv,
                      "beta_ols": beta_ols, "se_ols": se_ols,
                      "f_first_stage": f_fs, "j_pvalue": j_p,
                      "n_obs": len(sub)})
    return pd.DataFrame(rows).set_index("h")


def panel_lp_sigma_iv(
    df: pd.DataFrame,
    *,
    y: str,
    sigma: str,
    instruments: Sequence[str],
    horizons: Iterable[int] = range(0, 21),
    controls: Sequence[str] = (),
    unit_col: str = "code",
    date_col: str | None = None,
    n_lags: int = 2,
    fe: str = "entity",
) -> pd.DataFrame:
    """Panel LP-σ-IV via within transform.

    Parameters
    ----------
    fe : "entity" (default) — subtract country means only. Keeps
         common-time variation as identifying signal (best when the
         instrument is largely global, e.g., GPR).
       : "two-way" — subtract entity *and* time means. Use when the
         instrument has meaningful country-specific variation (e.g.,
         country-level WUI) that survives time-demeaning.
    """
    from linearmodels.iv import IV2SLS
    work = df.copy()
    if date_col is None:
        if isinstance(work.index, pd.MultiIndex):
            work = work.reset_index()
        date_col = "date"
    work[date_col] = pd.to_datetime(work[date_col])

    def _demean(s: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = s.copy()
        for c in cols:
            unit_m = s.groupby(unit_col)[c].transform("mean")
            if fe == "two-way":
                grand = s[c].mean()
                time_m = s.groupby(date_col)[c].transform("mean")
                out[c] = s[c] - unit_m - time_m + grand
            else:  # entity only
                out[c] = s[c] - unit_m
        return out

    rows = []
    for h in horizons:
        d = work.copy().sort_values([unit_col, date_col])
        d["__y_h"] = (d.groupby(unit_col)[y].shift(-h)
                       - d.groupby(unit_col)[y].shift(1))
        ctrl_cols = []
        for v in list(controls):
            for L in range(1, n_lags + 1):
                col = f"{v}_L{L}"
                d[col] = d.groupby(unit_col)[v].shift(L)
                ctrl_cols.append(col)
        keep = [unit_col, date_col, "__y_h", sigma] + list(instruments) + ctrl_cols
        sub = d[keep].dropna()
        if len(sub) < 100:
            rows.append({"h": h, "beta_iv": np.nan, "se_iv": np.nan,
                          "f_first_stage": np.nan, "j_pvalue": np.nan,
                          "n_obs": len(sub)})
            continue
        cols_to_dm = ["__y_h", sigma] + list(instruments) + ctrl_cols
        d2 = _demean(sub, cols_to_dm)
        try:
            exog = (d2[ctrl_cols].astype(float)
                    if ctrl_cols else pd.DataFrame(index=d2.index))
            iv = IV2SLS(d2["__y_h"].astype(float), exog,
                        d2[[sigma]].astype(float),
                        d2[list(instruments)].astype(float)).fit(cov_type="robust")
            beta_iv = float(iv.params[sigma])
            se_iv = float(iv.std_errors[sigma])
            try:
                f_fs = float(iv.first_stage.diagnostics.loc[sigma, "f.stat"])  # type: ignore[attr-defined]  # IV2SLSResults not in stubs
            except Exception:
                f_fs = np.nan
            try:
                j_p = float(iv.j_stat.pval)  # type: ignore[attr-defined]  # IV2SLSResults not in stubs
            except (AttributeError, TypeError):
                j_p = np.nan
        except Exception as e:
            print(f"[panel_lp_sigma_iv h={h}] {e}")
            beta_iv = se_iv = f_fs = j_p = np.nan
        rows.append({"h": h, "beta_iv": beta_iv, "se_iv": se_iv,
                      "f_first_stage": f_fs, "j_pvalue": j_p,
                      "n_obs": len(sub)})
    return pd.DataFrame(rows).set_index("h")


def simulate_garch_irf(
    *,
    mu: float, phi: float,
    omega: float, alpha: float, beta: float,
    horizon: int = 24,
    n_paths: int = 5000,
    seed: int = 0,
    level_shock_size: float = 1.0,    # σ-shock in units of unconditional σ
    innovation_shock_size: float = 3.0,  # z-shock in standard normals
):
    """Monte-Carlo IRFs from an AR(1)-GARCH(1,1) model.

    Two impulse experiments share the same draws of future ``z_t`` so that
    differences are attributable purely to the initial condition:

    1. Level shock: at ``t=0`` we *force* σ² to be ``(1+level_shock_size)
       × σ²_uncond``; future σ evolves under the GARCH dynamics.
    2. Innovation shock: at ``t=0`` we *force* ``z_0`` to be
       ``innovation_shock_size`` (in σ units); σ at ``t≥1`` reacts via
       the α·ε² term.

    A baseline path (no shock) is also simulated to compute IRFs as
    deviations from baseline.

    Returns dict with arrays of length ``horizon+1``:
        ``sigma_baseline``, ``sigma_level``, ``sigma_innov``,
        ``g_baseline``,     ``g_level``,     ``g_innov``
        (all averaged over ``n_paths`` Monte-Carlo replications).

    Notes
    -----
    Uses the unconditional variance σ² = ω/(1-α-β) (assumed stable,
    α+β<1) as the pre-shock baseline.
    """
    rng = np.random.default_rng(seed)
    if alpha + beta >= 1:
        raise ValueError("GARCH(1,1) instability: α+β ≥ 1")
    sig2_uncond = omega / (1 - alpha - beta)
    sig_uncond = float(np.sqrt(sig2_uncond))

    # Common z draws across the three paths so the difference is the IC.
    Z = rng.standard_normal((n_paths, horizon + 1))

    def _simulate(sig2_0, z0_override=None):
        sig2 = np.full((n_paths, horizon + 1), np.nan)
        eps = np.zeros((n_paths, horizon + 1))
        g = np.zeros((n_paths, horizon + 1))
        sig2[:, 0] = sig2_0
        z = Z.copy()
        if z0_override is not None:
            z[:, 0] = z0_override
        # iterate
        g_prev = np.full(n_paths, mu / max(1 - phi, 1e-9))  # steady state mean
        for t in range(horizon + 1):
            sig_t = np.sqrt(sig2[:, t])
            eps[:, t] = sig_t * z[:, t]
            g[:, t] = mu + phi * g_prev + eps[:, t]
            g_prev = g[:, t]
            if t + 1 <= horizon:
                sig2[:, t + 1] = omega + alpha * eps[:, t] ** 2 + beta * sig2[:, t]
        return np.sqrt(sig2).mean(0), g.mean(0)

    sig_b, g_b = _simulate(sig2_uncond)
    sig_l, g_l = _simulate(sig2_uncond * (1 + level_shock_size))
    sig_i, g_i = _simulate(sig2_uncond, z0_override=innovation_shock_size)

    return {
        "sigma_baseline": sig_b, "sigma_level": sig_l, "sigma_innov": sig_i,
        "g_baseline": g_b,       "g_level": g_l,       "g_innov": g_i,
        "sigma_uncond": sig_uncond,
    }


def lp_two_shocks(
    df: pd.DataFrame,
    *,
    y: str,
    shock_a: str,
    shock_b: str,
    horizons: Iterable[int] = range(0, 21),
    controls: Sequence[str] = (),
    n_lags: int = 2,
    hac_bandwidth_func=lambda h: h + 4,
) -> pd.DataFrame:
    """Two LPs side-by-side: y_{t+h}-y_{t-1} on shock_a_t and on shock_b_t.

    Caller chooses the transformation of each regressor (level, log,
    diff, ...). Returns RAW coefficients plus the pooled SD of each
    regressor so the caller can convert to "% per 1-SD shock" via
    ``β · SD · 100``.

    Columns: beta_a, se_a, beta_b, se_b, sd_a, sd_b, n_obs.
    HAC (Newey-West) errors with bandwidth ``h+4``.
    """
    import statsmodels.api as sm

    df = df.copy().sort_index()
    rows = []
    for h in horizons:
        d = df.copy()
        d["__y_h"] = d[y].shift(-h) - d[y].shift(1)
        ctrl_cols = []
        for v in list(controls):
            for L in range(1, n_lags + 1):
                col = f"{v}_L{L}"
                d[col] = d[v].shift(L)
                ctrl_cols.append(col)
        keep = ["__y_h", shock_a, shock_b] + ctrl_cols
        sub = d[keep].dropna()
        sd_a = float(sub[shock_a].std(ddof=0)) if len(sub) else np.nan
        sd_b = float(sub[shock_b].std(ddof=0)) if len(sub) else np.nan
        if len(sub) < 25:
            rows.append({"h": h, "beta_a": np.nan, "se_a": np.nan,
                          "beta_b": np.nan, "se_b": np.nan,
                          "sd_a": sd_a, "sd_b": sd_b, "n_obs": len(sub)})
            continue
        bw = int(hac_bandwidth_func(h))
        out = {"h": h, "n_obs": len(sub), "sd_a": sd_a, "sd_b": sd_b}
        for tag, raw_col in (("a", shock_a), ("b", shock_b)):
            X = sm.add_constant(sub[[raw_col] + ctrl_cols])
            try:
                res = sm.OLS(sub["__y_h"], X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": bw})
                out[f"beta_{tag}"] = float(res.params[raw_col])
                out[f"se_{tag}"] = float(res.bse[raw_col])
            except Exception:
                out[f"beta_{tag}"] = np.nan
                out[f"se_{tag}"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows).set_index("h")


# Back-compat alias kept for any callers that still use Δlevel / Δσ form.
def lp_level_vs_sigma(df, *, y, level, sigma, horizons=range(0, 21),
                      controls=(), n_lags=2,
                      hac_bandwidth_func=lambda h: h + 4):
    df = df.copy()
    df["__d_lev"] = df[level].diff()
    df["__d_sig"] = df[sigma].diff()
    out = lp_two_shocks(df, y=y, shock_a="__d_lev", shock_b="__d_sig",
                        horizons=horizons, controls=controls, n_lags=n_lags,
                        hac_bandwidth_func=hac_bandwidth_func)
    return out.rename(columns={"beta_a": "beta_level", "se_a": "se_level",
                                "beta_b": "beta_sigma", "se_b": "se_sigma",
                                "sd_a": "sd_level", "sd_b": "sd_sigma"})


def panel_lp_two_shocks(
    df: pd.DataFrame,
    *,
    y: str,
    shock_a: str,
    shock_b: str,
    horizons: Iterable[int] = range(0, 21),
    controls: Sequence[str] = (),
    unit_col: str = "code",
    date_col: str | None = None,
    n_lags: int = 2,
    fe: str = "entity",
) -> pd.DataFrame:
    """Panel version of lp_two_shocks via within transform; clustered SE.

    fe='entity' (default): subtract unit means. 'two-way': also subtract
    time means. Errors clustered on the unit (entity) dimension.
    Returns columns: beta_a, se_a, beta_b, se_b, sd_a, sd_b (within), n_obs.
    """
    from linearmodels.iv import IV2SLS  # used only for OLS fit
    work = df.copy()
    if date_col is None:
        if isinstance(work.index, pd.MultiIndex):
            work = work.reset_index()
        date_col = "date"
    work[date_col] = pd.to_datetime(work[date_col])

    def _demean(s: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = s.copy()
        for c in cols:
            unit_m = s.groupby(unit_col)[c].transform("mean")
            if fe == "two-way":
                grand = s[c].mean()
                time_m = s.groupby(date_col)[c].transform("mean")
                out[c] = s[c] - unit_m - time_m + grand
            else:
                out[c] = s[c] - unit_m
        return out

    rows = []
    for h in horizons:
        d = work.copy().sort_values([unit_col, date_col])
        d["__y_h"] = (d.groupby(unit_col)[y].shift(-h)
                       - d.groupby(unit_col)[y].shift(1))
        ctrl_cols = []
        for v in list(controls):
            for L in range(1, n_lags + 1):
                col = f"{v}_L{L}"
                d[col] = d.groupby(unit_col)[v].shift(L)
                ctrl_cols.append(col)
        keep = [unit_col, date_col, "__y_h", shock_a, shock_b] + ctrl_cols
        sub = d[keep].dropna()
        if len(sub) < 100:
            rows.append({"h": h, "beta_a": np.nan, "se_a": np.nan,
                          "beta_b": np.nan, "se_b": np.nan,
                          "sd_a": np.nan, "sd_b": np.nan,
                          "n_obs": len(sub)})
            continue
        d2 = _demean(sub, ["__y_h", shock_a, shock_b] + ctrl_cols)
        sd_a = float(d2[shock_a].std(ddof=0))
        sd_b = float(d2[shock_b].std(ddof=0))
        out = {"h": h, "n_obs": len(sub), "sd_a": sd_a, "sd_b": sd_b}
        for tag, raw_col in (("a", shock_a), ("b", shock_b)):
            try:
                exog = (d2[[raw_col] + ctrl_cols].astype(float)
                        if ctrl_cols else d2[[raw_col]].astype(float))
                empty_endog = pd.DataFrame(index=d2.index)
                empty_instr = pd.DataFrame(index=d2.index)
                res = IV2SLS(d2["__y_h"].astype(float), exog,
                             empty_endog, empty_instr).fit(
                                 cov_type="clustered",
                                 clusters=sub[unit_col].values)
                out[f"beta_{tag}"] = float(res.params[raw_col])
                out[f"se_{tag}"] = float(res.std_errors[raw_col])
            except Exception as e:
                print(f"[panel_lp_two_shocks h={h} {tag}] {e}")
                out[f"beta_{tag}"] = np.nan
                out[f"se_{tag}"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows).set_index("h")


def panel_lp_level_vs_sigma(df, *, y, level, sigma, horizons=range(0, 21),
                            controls=(), unit_col="code", date_col=None,
                            n_lags=2, fe="entity"):
    """Back-compat: feeds Δ-form regressors into panel_lp_two_shocks."""
    work = df.copy()
    if date_col is None:
        if isinstance(work.index, pd.MultiIndex):
            work = work.reset_index()
        date_col = "date"
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values([unit_col, date_col])
    work["__d_lev"] = work.groupby(unit_col)[level].diff()
    work["__d_sig"] = work.groupby(unit_col)[sigma].diff()
    out = panel_lp_two_shocks(work, y=y, shock_a="__d_lev", shock_b="__d_sig",
                              horizons=horizons, controls=controls,
                              unit_col=unit_col, date_col=date_col,
                              n_lags=n_lags, fe=fe)
    return out.rename(columns={"beta_a": "beta_level", "se_a": "se_level",
                                "beta_b": "beta_sigma", "se_b": "se_sigma",
                                "sd_a": "sd_level", "sd_b": "sd_sigma"})


__all__ = ["fit_argarch", "fit_argarchx", "lp_sigma_iv",
           "panel_lp_sigma_iv", "lp_two_shocks", "panel_lp_two_shocks",
           "lp_level_vs_sigma", "panel_lp_level_vs_sigma",
           "simulate_garch_irf", "GarchFit"]
