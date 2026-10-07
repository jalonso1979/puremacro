"""Shift-share (Bartik) IV with shock-level standard errors.

``shift_share_iv`` estimates the just-identified 2SLS regression of ``y`` on
``x`` using the shift-share instrument ``z_i = sum_k s_ik g_k`` (shares ``s``
by unit and sector, shocks ``g`` by sector), after partialling out the
controls and a constant. Two standard errors are reported:

* ``robust``: heteroskedasticity-robust (HC1) unit-level errors, the usual
  practice, which Adão, Kolesár and Morales (2019) show under-cover when the
  identifying variation comes from the sectoral shocks.
* ``akm``: the shock-level standard error of Adão, Kolesár and Morales
  (2019), IV version, eq. (39), with the sector-cluster extension of
  eq. (40) (equation numbers of arXiv:1806.07928v5, Aug 2019; Remark 5
  gives the construction). With ``w_i`` the unit weights, ``z''`` and
  ``x''`` the instrument and the regressor residualised on the controls by
  weighted least squares, and ``eps_i = y''_i - beta x''_i`` the 2SLS
  structural residuals,

      Xhat   = (S' diag(w) S)^{-1} S' diag(w) z''              (eq. 28)
      R_k    = sum_i w_i s_ik eps_i
      SE_AKM = sqrt( sum_c ( sum_{k in c} Xhat_k R_k )^2 ) / | sum_i w_i z''_i x''_i |

  ``Xhat`` is AKM's estimate of the shocks net of the controls: the
  regression of the control-partialled instrument on the shares (Remark 5,
  step 2). Without ``sector_clusters`` every sector is its own cluster and
  the formula is eq. (39); with them the products ``Xhat_k R_k`` are summed
  within clusters before squaring (eq. 40; AKM's ADH application, Table 5
  of arXiv v5, clusters the 4-digit SIC sectors by 3-digit SIC industry,
  which also makes the error robust to serial correlation of the shocks
  across the two periods). The standard error is in units of
  ``beta`` and does not depend on the scale of the weights. This is the
  ``AKM`` row of ``ShiftShareSE::ivreg_ss`` (Kolesár's R package, version
  1.1.0), which it reproduces on the package's ADH data: estimate
  -0.7742267, AKM standard error 0.2403730 with 3-digit SIC clusters.

  Computing ``Xhat`` needs a full-column-rank share matrix. Sectors whose
  share column is (numerically) a linear combination of the columns before
  it are dropped from the projection and from the sum, with a warning, as
  ``ShiftShareSE`` does (its vignette, Section 3.1); the result then depends
  on the column order. Sectors nobody is exposed to (an all-zero column)
  contribute nothing and are dropped silently. With more sectors than units
  at most ``n`` sectors can be kept.

  ``akm_shocks='residualized'`` replaces ``Xhat`` with the shocks
  residualised at the sector level on a constant and ``shock_controls``,
  weighted by the sector's total share ``sum_i w_i s_ik``. That is the
  alternative in Section 3.2 (item 3) of the ``ShiftShareSE`` vignette, valid
  when every control has shift-share structure. It coincides with ``Xhat``
  when the only control is the intercept and each unit's shares sum to one,
  and differs otherwise. Up to puremacro 4.3.0 it was the only formula, so
  ``se_akm`` was not AKM's error whenever there were unit-level controls or
  incomplete shares.

References
----------
Adão, R., Kolesár, M. and Morales, E. (2019). Shift-share designs: theory and
    inference. Quarterly Journal of Economics 134(4), 1949-2010.
    arXiv:1806.07928v5: Remark 5 and eqs. (28)-(29), eq. (39), eq. (40).
Kolesár, M. (2022). ShiftShareSE: Inference in regressions with shift-share
    structure, R package 1.1.0, and its vignette "Standard errors in
    shift-share regressions" (23 April 2022).
Borusyak, K., Hull, P. and Jaravel, X. (2022). Quasi-experimental shift-share
    research designs. Review of Economic Studies 89(1), 181-213.
Goldsmith-Pinkham, P., Sorkin, I. and Swift, H. (2020). Bartik instruments:
    what, when, why, and how. American Economic Review 110(8), 2586-2624.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_triangular

__all__ = ["shift_share_iv", "ShiftShareIVResult"]

#: Relative tolerance of the collinearity check on the share columns: a column
#: is dropped when the norm of its residual on the columns kept before it falls
#: below this fraction of its own norm (the rule and default of R's ``qr``,
#: which ShiftShareSE uses to drop collinear sectors).
_COLLINEAR_TOL = 1e-7


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
        Shock-level (AKM 2019, eq. 39, or eq. 40 with sector clusters) and
        unit-level HC1 standard errors, both in units of ``beta``.
    t, p_value, ci_lower, ci_upper : float
        Computed from ``se`` with the normal approximation.
    first_stage_F : float
        HC1-robust first-stage F (single instrument: squared robust t).
    n_units, n_sectors : int
    rotemberg_weights : pandas.Series
        Goldsmith-Pinkham-Sorkin-Swift weights by sector (sum to one).
    shocks_residualized : pandas.Series
        The sector-level shocks entering the AKM formula. With
        ``akm_shocks='projection'`` (the default) these are AKM's ``Xhat_k``
        (eq. 28), the coefficients of the control-partialled instrument on
        the shares, NaN for sectors dropped as collinear or never exposed.
        With ``akm_shocks='residualized'``, the shocks residualised at the
        sector level on a constant and ``shock_controls``.
    se_type : str
        ``'akm'`` or ``'robust'``.
    akm_shocks : str
        ``'projection'`` or ``'residualized'``: how ``shocks_residualized``
        was built.
    n_sector_clusters : int or None
        Number of sector clusters summed over in ``se_akm`` (eq. 40); None
        when each sector is its own cluster (eq. 39).
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
    akm_shocks: str = "projection"
    n_sector_clusters: int | None = None

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
        akm_how = "hat-X projection" if self.akm_shocks == "projection" else "sector-level residualised shocks"
        if self.n_sector_clusters is not None:
            akm_how += f", {self.n_sector_clusters} sector clusters"
        return "\n".join([
            "Shift-share IV (Bartik) with shock-level inference",
            f"  beta = {self.beta:+.6f}   se[{self.se_type}] = {self.se:.6f}   t = {self.t:+.3f}   p = {self.p_value:.4f}",
            f"  {1 - self.alpha:.0%} CI: [{self.ci_lower:+.6f}, {self.ci_upper:+.6f}]",
            f"  se AKM = {self.se_akm:.6f} ({akm_how})   se robust (HC1) = {self.se_robust:.6f}   ratio AKM/robust = {ratio:.2f}",
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


def _independent_columns(A: np.ndarray, tol: float = _COLLINEAR_TOL) -> np.ndarray:
    """Boolean mask of the columns of ``A`` kept by a left-to-right rank scan.

    Column ``j`` is kept when the norm of its residual on the columns kept
    before it exceeds ``tol`` times its own norm. This is the limited-pivoting
    rule of LINPACK ``dqrdc2`` behind R's ``qr()``, which ShiftShareSE's
    ``drop_collinear`` uses, so the same sectors are dropped (the scan here
    runs on the weighted shares ``sqrt(w_i) s_ik``, which can only matter for
    zero weights or columns at the tolerance). All-zero columns are never kept.
    """
    n, K = A.shape
    norms = np.linalg.norm(A, axis=0)
    keep = np.zeros(K, dtype=bool)
    Q = np.empty((n, min(n, K)))
    r = 0
    for j in range(K):
        if norms[j] == 0.0 or r >= n:
            continue
        v = A[:, j].copy()
        for _ in range(2):                      # Gram-Schmidt with re-orthogonalisation
            v -= Q[:, :r] @ (Q[:, :r].T @ v)
        nv = float(np.linalg.norm(v))
        if nv > tol * norms[j]:
            Q[:, r] = v / nv
            r += 1
            keep[j] = True
    return keep


def _share_projection(A: np.ndarray, b: np.ndarray, tol: float = _COLLINEAR_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares coefficients of ``b`` on the independent columns of ``A``.

    Returns ``(keep, coef)``: the mask of :func:`_independent_columns` and the
    coefficients on those columns, from one Householder QR of ``[A_keep, b]``.
    When ``A`` has full column rank (the usual case) that same QR also proves
    it, since then every diagonal entry of ``R`` clears the tolerance.
    """
    n, K = A.shape
    norms = np.linalg.norm(A, axis=0)
    if n >= K and np.all(norms > 0):
        R = np.linalg.qr(np.column_stack([A, b]), mode="r")
        if np.all(np.abs(np.diag(R)[:K]) > tol * norms):
            return np.ones(K, dtype=bool), solve_triangular(R[:K, :K], R[:K, K])
    keep = _independent_columns(A, tol)
    k = int(keep.sum())
    R = np.linalg.qr(np.column_stack([A[:, keep], b]), mode="r")
    return keep, solve_triangular(R[:k, :k], R[:k, k])


def _cluster_sum(c: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
    """Sum ``c`` within the clusters given by ``labels`` (identity when None)."""
    if labels is None:
        return c
    codes = pd.factorize(pd.Series(labels, dtype=object))[0]
    return np.bincount(codes, weights=c)


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
    sector_clusters: pd.Series | np.ndarray | Sequence[Any] | None = None,
    akm_shocks: str | None = None,
    instrument: str | np.ndarray | None = None,
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
        shares as a control, as Borusyak, Hull and Jaravel and AKM
        (Section 4.2) recommend.
    weights : optional unit weights (column name or array). They weight
        every regression, including the AKM projection of eq. (28), as
        ``ShiftShareSE`` does; their scale does not matter.
    se : {'akm', 'robust'}
        Which standard error fills ``se`` / ``t`` / ``p_value`` / the CI; both
        are always reported.
    shock_controls : optional (n_sectors, q) sector-level controls. They are
        used only by ``akm_shocks='residualized'``, which residualises the
        shocks on a constant and these controls (weighted by the sectors'
        total shares) before the AKM formula; passing them without
        ``akm_shocks`` selects that method. For the default projection put
        the shift-share controls ``shares @ shock_controls`` among the
        unit-level ``controls`` instead.
    alpha : confidence level is ``1 - alpha``.
    sector_clusters : optional (n_sectors,) cluster labels of the sectors,
        e.g. 3-digit codes for 4-digit sectors (``ShiftShareSE``'s
        ``sector_cvar``). ``se_akm`` then allows the shocks to be correlated
        within a cluster (AKM eq. 40). A Series is aligned on the share
        columns. In panels whose "sectors" are sector-period pairs, cluster
        by sector to allow serial correlation of the shocks (AKM Section 5.2).
    akm_shocks : {'projection', 'residualized'}, optional
        How the AKM formula nets the controls out of the shocks.
        ``'projection'`` (the default unless ``shock_controls`` is given) is
        AKM's ``Xhat`` of eq. (28); ``'residualized'`` is the sector-level
        residualisation described in the module docstring.
    instrument : optional column name or (n_units,) array holding the
        shift-share instrument ``z_i`` when it was built outside and is not
        exactly ``shares @ shocks`` (e.g. published at limited precision).
        ``z`` then comes from it for the estimate and both standard errors,
        while ``shocks`` still define the Rotemberg weights and the
        ``'residualized'`` shocks. Default: ``z = shares @ shocks``.
    """
    if se not in ("akm", "robust"):
        raise ValueError(f"se must be 'akm' or 'robust', got {se!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if akm_shocks is None:
        akm_shocks = "projection" if shock_controls is None else "residualized"
    if akm_shocks not in ("projection", "residualized"):
        raise ValueError(f"akm_shocks must be 'projection' or 'residualized', got {akm_shocks!r}")
    if akm_shocks == "projection" and shock_controls is not None:
        raise ValueError(
            "shock_controls are only used by akm_shocks='residualized'; with the AKM projection "
            "add the shift-share controls shares @ shock_controls to the unit-level controls"
        )
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
    labels: np.ndarray | None = None
    if sector_clusters is not None:
        if isinstance(sector_clusters, pd.Series):
            missing = [k for k in sector_ids if k not in sector_clusters.index]
            if missing:
                raise KeyError(f"sector_clusters are missing {len(missing)} sector(s), e.g. {missing[:3]}")
            labels = sector_clusters.loc[sector_ids].to_numpy(dtype=object)
        else:
            labels = np.asarray(sector_clusters, dtype=object).ravel()
            if labels.shape[0] != K:
                raise ValueError(f"sector_clusters has {labels.shape[0]} entries, shares has {K} sectors")
        if pd.isna(pd.Series(labels, dtype=object)).any():
            raise ValueError("sector_clusters must not contain missing labels")
    if weights is None:
        w = np.ones(n)
    else:
        w = df[weights].to_numpy(dtype=float) if isinstance(weights, str) else np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != n or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be non-negative, finite and have one entry per unit")
        w = w * n / w.sum()

    yv = df[y].to_numpy(dtype=float)
    xv = df[x].to_numpy(dtype=float)
    z_sg = S @ g
    if instrument is None:
        z = z_sg
    else:
        z = df[instrument].to_numpy(dtype=float) if isinstance(instrument, str) else np.asarray(instrument, dtype=float).ravel()
        if z.shape[0] != n or not np.all(np.isfinite(z)):
            raise ValueError("instrument must be finite and have one entry per unit")
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

    # shock-level AKM (eqs. 28, 39, 40)
    R_k = (w[:, None] * S * eps[:, None]).sum(axis=0)          # sum_i w_i s_ik eps_i
    if akm_shocks == "projection":
        sw = np.sqrt(w)
        Sw = S * sw[:, None]
        keep, hX = _share_projection(Sw, sw * zt)                   # eq. (28)
        dropped = [sector_ids[k] for k in np.flatnonzero(~keep) if np.any(Sw[:, k] != 0.0)]
        if dropped:
            warnings.warn(
                f"shift_share_iv: the share matrix is collinear (rank {int(keep.sum())} < {K} sectors); "
                f"{len(dropped)} sector(s) dropped from the AKM projection, e.g. {dropped[:3]}. As in "
                "ShiftShareSE, the AKM standard error then depends on the order of the sectors.",
                RuntimeWarning, stacklevel=2,
            )
        shock_net = np.full(K, np.nan)
        shock_net[keep] = hX
        contrib = hX * R_k[keep]
        clusters = labels[keep] if labels is not None else None
    else:
        s_k = (w[:, None] * S).sum(axis=0)
        Zs = np.ones((K, 1))
        if shock_controls is not None:
            Q = shock_controls.loc[sector_ids].to_numpy(dtype=float) if isinstance(shock_controls, pd.DataFrame) else np.asarray(shock_controls, dtype=float)
            if Q.ndim == 1:
                Q = Q[:, None]
            if Q.shape[0] != K:
                raise ValueError("shock_controls must have one row per sector")
            Zs = np.column_stack([Zs, Q])
        sk_pos = np.where(s_k > 0, s_k, 0.0)
        shock_net = _partial_out(g[:, None], Zs, sk_pos if sk_pos.sum() > 0 else np.ones(K)).ravel()
        contrib = shock_net * R_k
        clusters = labels
    cl_sums = _cluster_sum(contrib, clusters)
    se_akm = float(np.sqrt(np.sum(cl_sums ** 2)) / abs(zx))
    n_clusters = int(len(cl_sums)) if labels is not None else None

    # robust first stage
    pi = zx / float(np.sum(w * zt * zt))
    v1 = xt - pi * zt
    se_pi = float(np.sqrt(np.sum((w * zt * v1) ** 2) * n / max(n - k_par, 1)) / np.sum(w * zt * zt))
    first_stage_F = float((pi / se_pi) ** 2) if se_pi > 0 else float("inf")

    # Rotemberg weights of the instrument shares @ shocks (GPSS 2020)
    zx_sg = zx if instrument is None else float(np.sum(w * z_sg * xt))
    rot = pd.Series(g * (w[:, None] * S * xt[:, None]).sum(axis=0) / zx_sg, index=sector_ids, name="rotemberg_weight")
    se_sel = se_akm if se == "akm" else se_robust
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    t = beta / se_sel if se_sel > 0 else float("inf")
    return ShiftShareIVResult(
        beta=beta, se=se_sel, se_akm=se_akm, se_robust=se_robust, t=float(t),
        p_value=float(2.0 * stats.norm.sf(abs(t))), ci_lower=beta - z_crit * se_sel, ci_upper=beta + z_crit * se_sel,
        first_stage_F=first_stage_F, n_units=int(n), n_sectors=int(K),
        rotemberg_weights=rot, shocks_residualized=pd.Series(shock_net, index=sector_ids, name="shock_residualized"),
        se_type=se, alpha=float(alpha), akm_shocks=akm_shocks, n_sector_clusters=n_clusters,
    )
