"""Shift-share (Bartik) IV with shock-level standard errors.

``shift_share_iv`` estimates the just-identified 2SLS regression of ``y`` on
``x`` using the shift-share instrument ``z_i = sum_k s_ik g_k`` (shares ``s``
by unit and sector, shocks ``g`` by sector), after partialling out the
controls and a constant. Two standard errors are reported:

* ``robust``: heteroskedasticity-robust (HC1) unit-level errors, the usual
  practice, which Adão, Kolesár and Morales (2019) show under-cover when the
  identifying variation comes from the sectoral shocks.
* ``akm``: the shock-level standard error of AKM (2019, eq. 24). With
  ``eps_i`` the 2SLS residuals and ``g~_k`` the shocks residualised on the
  share-weighted constant (and on ``shock_controls`` when given),

      SE_AKM = sqrt( sum_k g~_k^2 (sum_i s_ik eps_i)^2 ) / | sum_i z_i x~_i |

  which equals the heteroskedasticity-robust error of the equivalent
  shock-level IV regression of Borusyak, Hull and Jaravel (2022).

References
----------
Adão, R., Kolesár, M. and Morales, E. (2019). Shift-share designs: theory and
    inference. Quarterly Journal of Economics 134(4), 1949-2010.
Borusyak, K., Hull, P. and Jaravel, X. (2022). Quasi-experimental shift-share
    research designs. Review of Economic Studies 89(1), 181-213.
Goldsmith-Pinkham, P., Sorkin, I. and Swift, H. (2020). Bartik instruments:
    what, when, why, and how. American Economic Review 110(8), 2586-2624.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["shift_share_iv", "ShiftShareIVResult"]


def _render(df: pd.DataFrame, fmt: str, **kwargs: Any) -> str:
    from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst
    if fmt == "markdown":
        return _df_to_markdown(df, index=False, **kwargs)
    if fmt == "latex":
        return _df_to_latex(df, index=False, **kwargs)
    return _df_to_typst(df, index=False, **kwargs)


@dataclass(frozen=True, eq=False)
class ShiftShareIVResult:
    """Result of :func:`shift_share_iv`.

    Attributes
    ----------
    beta : float
        2SLS coefficient on ``x``.
    se : float
        The standard error selected by ``se=`` (``se_akm`` or ``se_robust``).
    se_akm, se_robust : float
        Shock-level (AKM) and unit-level HC1 standard errors.
    t, p_value, ci_lower, ci_upper : float
        Computed from ``se`` with the normal approximation.
    first_stage_F : float
        HC1-robust first-stage F (single instrument: squared robust t).
    n_units, n_sectors : int
    rotemberg_weights : pandas.Series
        Goldsmith-Pinkham-Sorkin-Swift weights by sector (sum to one).
    shocks_residualized : pandas.Series
        The shocks ``g~_k`` entering the AKM formula.
    se_type : str
        ``'akm'`` or ``'robust'``.
    """

    beta: float
    se: float
    se_akm: float
    se_robust: float
    t: float
    p_value: float
    ci_lower: float
    ci_upper: float
    first_stage_F: float
    n_units: int
    n_sectors: int
    rotemberg_weights: pd.Series
    shocks_residualized: pd.Series
    se_type: str
    alpha: float

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("beta", self.beta), (f"se ({self.se_type})", self.se), ("se (AKM)", self.se_akm),
            ("se (robust HC1)", self.se_robust), ("t", self.t), ("p-value", self.p_value),
            (f"CI lower ({1 - self.alpha:.0%})", self.ci_lower), (f"CI upper ({1 - self.alpha:.0%})", self.ci_upper),
            ("first-stage F", self.first_stage_F), ("units", self.n_units), ("sectors", self.n_sectors),
        ]
        return pd.DataFrame({"statistic": [r[0] for r in rows], "value": [r[1] for r in rows]})

    def summary(self) -> str:
        ratio = self.se_akm / self.se_robust if self.se_robust > 0 else float("nan")
        top = self.rotemberg_weights.abs().sort_values(ascending=False).head(3)
        return "\n".join([
            "Shift-share IV (Bartik) with shock-level inference",
            f"  beta = {self.beta:+.6f}   se[{self.se_type}] = {self.se:.6f}   t = {self.t:+.3f}   p = {self.p_value:.4f}",
            f"  {1 - self.alpha:.0%} CI: [{self.ci_lower:+.6f}, {self.ci_upper:+.6f}]",
            f"  se AKM = {self.se_akm:.6f}   se robust (HC1) = {self.se_robust:.6f}   ratio AKM/robust = {ratio:.2f}",
            f"  first-stage F (robust) = {self.first_stage_F:.2f}   units = {self.n_units}   sectors = {self.n_sectors}",
            "  largest |Rotemberg weights|: " + ", ".join(f"{k}: {v:+.3f}" for k, v in top.items()),
        ])

    def to_markdown(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, ax=None, top: int = 15, figsize: tuple[float, float] = (7.0, 4.0)):
        """Bar chart of the largest Rotemberg weights."""
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        w = self.rotemberg_weights.reindex(self.rotemberg_weights.abs().sort_values(ascending=False).index).head(top)
        ax.barh([str(k) for k in w.index][::-1], w.values[::-1], color=np.where(w.values[::-1] >= 0, "steelblue", "firebrick"))
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Rotemberg weight")
        ax.set_title(f"Shift-share IV: beta = {self.beta:+.3f} (AKM se {self.se_akm:.3f}, robust se {self.se_robust:.3f})")
        if fig is not None:
            fig.tight_layout()
        return fig if fig is not None else ax.get_figure()


def _partial_out(M: np.ndarray, Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Residualise the columns of ``M`` on ``Z`` by weighted least squares."""
    sw = np.sqrt(w)[:, None]
    Zw = Z * sw
    coef, *_ = np.linalg.lstsq(Zw, M * sw, rcond=None)
    return M - Z @ coef


def shift_share_iv(
    df: pd.DataFrame,
    y: str,
    x: str,
    shares: pd.DataFrame | np.ndarray,
    shocks: pd.Series | np.ndarray,
    *,
    controls: Sequence[str] = (),
    weights: str | np.ndarray | None = None,
    se: str = "akm",
    shock_controls: pd.DataFrame | np.ndarray | None = None,
    alpha: float = 0.05,
) -> ShiftShareIVResult:
    """Shift-share (Bartik) IV with robust and AKM standard errors.

    Parameters
    ----------
    df : DataFrame with one row per unit holding ``y``, ``x`` and ``controls``.
    y, x : column names of the outcome and the endogenous regressor.
    shares : (n_units, n_sectors) exposure shares ``s_ik``; a DataFrame is
        aligned on ``df``'s index and its columns name the sectors.
    shocks : (n_sectors,) sectoral shocks ``g_k``; a Series is aligned on the
        share columns.
    controls : unit-level control columns (a constant is always included).
        With incomplete shares (rows not summing to one) add the sum of
        shares as a control, as Borusyak, Hull and Jaravel recommend.
    weights : optional unit weights (column name or array).
    se : {'akm', 'robust'}
        Which standard error fills ``se`` / ``t`` / ``p_value`` / the CI; both
        are always reported.
    shock_controls : optional (n_sectors, q) sector-level controls the shocks
        are residualised on before entering the AKM formula (a share-weighted
        constant is always included).
    alpha : confidence level is ``1 - alpha``.
    """
    if se not in ("akm", "robust"):
        raise ValueError(f"se must be 'akm' or 'robust', got {se!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    n = len(df)
    if isinstance(shares, pd.DataFrame):
        missing = [u for u in df.index if u not in shares.index]
        if missing:
            raise KeyError(f"shares are missing {len(missing)} unit(s) of df, e.g. {missing[:3]}")
        S = shares.loc[df.index].to_numpy(dtype=float)
        sector_ids = list(shares.columns)
    else:
        S = np.asarray(shares, dtype=float)
        if S.shape[0] != n:
            raise ValueError(f"shares has {S.shape[0]} rows, df has {n}")
        sector_ids = list(range(S.shape[1]))
    K = S.shape[1]
    if isinstance(shocks, pd.Series):
        missing = [k for k in sector_ids if k not in shocks.index]
        if missing:
            raise KeyError(f"shocks are missing {len(missing)} sector(s), e.g. {missing[:3]}")
        g = shocks.loc[sector_ids].to_numpy(dtype=float)
    else:
        g = np.asarray(shocks, dtype=float).ravel()
        if g.shape[0] != K:
            raise ValueError(f"shocks has {g.shape[0]} entries, shares has {K} sectors")
    if np.any(S < 0):
        raise ValueError("shares must be non-negative")
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(g))):
        raise ValueError("shares and shocks must be finite")
    if K < 2:
        raise ValueError("shift_share_iv needs at least two sectors")
    if weights is None:
        w = np.ones(n)
    else:
        w = df[weights].to_numpy(dtype=float) if isinstance(weights, str) else np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != n or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be non-negative, finite and have one entry per unit")
        w = w * n / w.sum()

    yv = df[y].to_numpy(dtype=float)
    xv = df[x].to_numpy(dtype=float)
    z = S @ g
    C = np.column_stack([np.ones(n)] + [df[c].to_numpy(dtype=float) for c in controls])
    if not np.all(np.isfinite(np.column_stack([yv, xv, C]))):
        raise ValueError("y, x and controls must be finite")
    yt, xt, zt = (_partial_out(v[:, None], C, w).ravel() for v in (yv, xv, z))
    if float(np.sum(w * zt * zt)) <= 1e-12 * max(float(np.sum(w * z * z)), 1e-300):
        raise ValueError(
            "the shift-share instrument has no variation after partialling out the controls "
            "(constant shocks across sectors, or shares collinear with the controls)"
        )
    zx = float(np.sum(w * zt * xt))
    if abs(zx) < 1e-14:
        raise ValueError("the shift-share instrument is uncorrelated with x (sum_i z_i x_i = 0)")
    beta = float(np.sum(w * zt * yt) / zx)
    eps = yt - beta * xt
    k_par = C.shape[1] + 1

    # unit-level HC1
    se_robust = float(np.sqrt(np.sum((w * zt * eps) ** 2) * n / max(n - k_par, 1)) / abs(zx))

    # shock-level AKM: residualise shocks on the share-weighted constant (+ shock controls)
    s_k = (w[:, None] * S).sum(axis=0)
    Zs = np.ones((K, 1))
    if shock_controls is not None:
        Q = shock_controls.loc[sector_ids].to_numpy(dtype=float) if isinstance(shock_controls, pd.DataFrame) else np.asarray(shock_controls, dtype=float)
        if Q.shape[0] != K:
            raise ValueError("shock_controls must have one row per sector")
        Zs = np.column_stack([Zs, Q])
    sk_pos = np.where(s_k > 0, s_k, 0.0)
    g_res = _partial_out(g[:, None], Zs, sk_pos if sk_pos.sum() > 0 else np.ones(K)).ravel()
    R_k = (w[:, None] * S * eps[:, None]).sum(axis=0)          # sum_i w_i s_ik eps_i
    se_akm = float(np.sqrt(np.sum(g_res ** 2 * R_k ** 2)) / abs(zx))

    # robust first stage
    pi = zx / float(np.sum(w * zt * zt))
    v1 = xt - pi * zt
    se_pi = float(np.sqrt(np.sum((w * zt * v1) ** 2) * n / max(n - k_par, 1)) / np.sum(w * zt * zt))
    first_stage_F = float((pi / se_pi) ** 2) if se_pi > 0 else float("inf")

    rot = pd.Series(g * (w[:, None] * S * xt[:, None]).sum(axis=0) / zx, index=sector_ids, name="rotemberg_weight")
    se_sel = se_akm if se == "akm" else se_robust
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    t = beta / se_sel if se_sel > 0 else float("inf")
    return ShiftShareIVResult(
        beta=beta, se=se_sel, se_akm=se_akm, se_robust=se_robust, t=float(t),
        p_value=float(2.0 * stats.norm.sf(abs(t))), ci_lower=beta - z_crit * se_sel, ci_upper=beta + z_crit * se_sel,
        first_stage_F=first_stage_F, n_units=int(n), n_sectors=int(K),
        rotemberg_weights=rot, shocks_residualized=pd.Series(g_res, index=sector_ids, name="shock_residualized"),
        se_type=se, alpha=float(alpha),
    )
