"""KORV (2000) two-elasticity system-GMM estimator.

Stacks three first-order conditions of the nested CES production function
in log-differences and estimates (σ_su, σ_eu) jointly. The third elasticity
σ_es (equipment-skilled) is a local Allen elasticity computed at sample-
mean factor shares — not a primitive — and is reported alongside.

Per-country: identification from time variation in factor ratios and
prices. Pooled: country-clustered weighting matrix, two-step GMM, with
Hansen J test on the over-identifying restriction.

Diagnostic variants
-------------------
fit_korv_pooled_usercost
    Replaces the asset-price P_K in m2 with a Hall-Jorgenson user-cost
    r_e = P_K · (r + δ_e − E[Δlog P_K]), where δ_e = 0.13 (BEA equipment
    depreciation) and E[Δlog P_K] uses perfect-foresight one-step-ahead.
    All three moments still present; over-identification unchanged (df=1).

fit_korv_pooled_2moments
    Drops m3 (labor-share FOC). Just-identified (2 moments, 2 params);
    Hansen J undefined. Isolates whether m3 mis-specification drives the
    rejection.

References
----------
Krusell, Ohanian, Ríos-Rull, Violante (2000), Capital-Skill Complementarity
    and Inequality, Econometrica.
León-Ledesma, McAdam, Willman (2010), Identifying the Elasticity of
    Substitution with Biased Technical Change, AER.
Hall, R., Jorgenson, D. (1967), Tax Policy and Investment Behavior, AER.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class KorvFit:
    sigma_su: float
    sigma_eu: float
    sigma_es: float            # local Allen elasticity at sample-mean shares
    se_sigma_su: float
    se_sigma_eu: float
    obj_value: float           # GMM objective at minimum
    hansen_J: Optional[float]  # over-id test stat (None for just-id)
    hansen_p:  Optional[float]
    n_obs:    int
    n_country: int
    converged: bool


@dataclass(frozen=True)
class SymCESFit:
    """Result of the symmetric four-factor nested CES estimator.

    Three structural elasticities of substitution:
      - sigma_kl: between-nest, capital vs labor (the headline σ_KL).
      - sigma_ie: within-capital, intangibles vs equipment.
      - sigma_su: within-labor, skilled vs unskilled.

    Standard errors are GMM sandwich SEs from the numerical Jacobian of the
    moment conditions evaluated at the point estimates. Hansen J is the
    over-identification statistic on the four-moment joint system (df=1);
    None for sequential block estimates.
    """
    sigma_kl: float
    sigma_ie: float
    sigma_su: float
    se_sigma_kl: float
    se_sigma_ie: float
    se_sigma_su: float
    obj_value: float
    hansen_J: Optional[float]
    hansen_p: Optional[float]
    n_obs: int
    n_country: int
    converged: bool


def log_share_ratio(labor_share) -> np.ndarray:
    """log(s_K/s_L) from a labor share, for the m_4 moment.

    The share-ratio form dlog(s_K/s_L) = (1-sigma) dlog(P_K/P_C) is exact and
    needs no capital-share weight, unlike the labor-share form. Values outside
    (0, 1) are rejected rather than silently propagated.
    """
    ls = np.asarray(labor_share, dtype=float)
    if np.any(~np.isfinite(ls)) or np.any((ls <= 0.0) | (ls >= 1.0)):
        raise ValueError("labor_share must be finite and strictly inside (0, 1)")
    return np.log1p(-ls) - np.log(ls)


def _moments(theta: np.ndarray, dat: dict) -> np.ndarray:
    """Stack three IV-moment conditions at parameters (sigma_su, sigma_eu).

    Returns a length-3 vector of sample means of z_i · r_i(θ), where z_i
    is the instrument and r_i is the residual from each FOC. Using the
    instrument equal to the right-hand-side variable gives the standard
    IV/GMM 'score' moment condition:

      g1 = E[Δlog(w_s/w_u) · (Δlog(L_s/L_u) + σ_su · Δlog(w_s/w_u))]
      g2 = E[Δlog(w_u/P_K) · (Δlog(K_e/L_u) - σ_eu · Δlog(w_u/P_K))]
      g3 = E[Δlog(P_K/P_C) · (Δlog(LS_u) - (1-σ_eu) · Δlog(P_K/P_C))]

    Each is zero at the true parameter value; the gradient is -Var(z) ≠ 0,
    giving clean GMM identification unlike raw residual means.

    Returns an (n_obs, 3) matrix of per-observation contributions
    (z_t · r_t) so the weighting matrix operates on the sample mean.
    """
    sigma_su, sigma_eu = theta
    # CES relative demand is dlog(L_s/L_u) = -sigma_su * dlog(w_s/w_u): the
    # quantity and price ratios run in the SAME direction here, unlike r2/r3
    # where the price ratio is inverted, so the residual carries a plus sign.
    r1 = dat['dlog_ls_lu']    + sigma_su        * dat['dlog_ws_wu']
    r2 = dat['dlog_ke_lu']    - sigma_eu        * dat['dlog_wu_pk']
    r3 = dat['dlog_lsushare'] - (1 - sigma_eu)  * dat['dlog_pk_pc']
    g1 = dat['dlog_ws_wu']  * r1
    g2 = dat['dlog_wu_pk']  * r2
    g3 = dat['dlog_pk_pc']  * r3
    return np.column_stack([g1, g2, g3])


def _gmm_objective(theta: np.ndarray, dat: dict, W: np.ndarray) -> float:
    """Quadratic GMM objective: ḡ' W ḡ where ḡ = mean of IV moment contribs."""
    g = _moments(theta, dat).mean(axis=0)
    return float(g @ W @ g)


def _allen_sigma_es(sigma_su: float, sigma_eu: float, share_e: float) -> float:
    """Local Allen-Uzawa elasticity for equipment vs skilled labor at
    given equipment-share-of-inner-aggregate. Standard nested-CES result.

    The Allen-Uzawa elasticity between equipment (e) and skilled labor (s)
    in the KORV nest is bounded by [min(σ_su, σ_eu), max(σ_su, σ_eu)].
    We approximate it as the share-weighted harmonic mean of σ_su and σ_eu,
    which interpolates between the two primitive elasticities as the
    equipment share rises.
    """
    # Guard against degenerate shares.
    share_e = float(np.clip(share_e, 1e-6, 1 - 1e-6))
    return 1.0 / (share_e / sigma_eu + (1 - share_e) / sigma_su)


def _two_step_se(res_x: np.ndarray, dat: dict, W1: np.ndarray, n: int
                 ) -> np.ndarray:
    """GMM sandwich standard errors via numerical Jacobian."""
    eps = 1e-5
    G = np.zeros((3, 2))
    for j, dx in enumerate([np.array([eps, 0.0]), np.array([0.0, eps])]):
        m_plus  = _moments(res_x + dx, dat).mean(axis=0)
        m_minus = _moments(res_x - dx, dat).mean(axis=0)
        G[:, j] = (m_plus - m_minus) / (2 * eps)
    try:
        var = np.linalg.inv(G.T @ W1 @ G) / n
        se = np.sqrt(np.diag(var))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    return se


def fit_korv_country(panel: pd.DataFrame) -> KorvFit:
    """Per-country GMM fit. `panel` must have columns
    dlog_ls_lu, dlog_ws_wu, dlog_ke_lu, dlog_wu_pk, dlog_lsushare, dlog_pk_pc
    plus a 'k_equip_share_inner' column for σ_es computation.

    Drop rows with any NaN before calling.
    """
    _moment_cols = ('dlog_ls_lu', 'dlog_ws_wu', 'dlog_ke_lu',
                    'dlog_wu_pk', 'dlog_lsushare', 'dlog_pk_pc')
    dat = {c: panel[c].to_numpy() for c in _moment_cols}
    n = len(panel)
    if n < 8:
        return KorvFit(
            sigma_su=np.nan, sigma_eu=np.nan, sigma_es=np.nan,
            se_sigma_su=np.nan, se_sigma_eu=np.nan,
            obj_value=np.nan, hansen_J=None, hansen_p=None,
            n_obs=int(n), n_country=1, converged=False,
        )
    # First-step: identity W.
    W0 = np.eye(3)
    res = minimize(
        _gmm_objective, x0=np.array([0.7, 1.5]),
        args=(dat, W0), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000},
    )
    # Two-step (efficient) GMM: re-estimate W as the inverse covariance of
    # moments at the first-step estimates.
    m_mat = _moments(res.x, dat)
    Omega = np.cov(m_mat.T) + 1e-8 * np.eye(3)
    W1 = np.linalg.inv(Omega)
    res2 = minimize(
        _gmm_objective, x0=res.x,
        args=(dat, W1), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000},
    )
    sigma_su, sigma_eu = res2.x
    se = _two_step_se(res2.x, dat, W1, n)
    # Allen σ_es at sample-mean equipment share of inner aggregate.
    if 'k_equip_share_inner' in panel.columns:
        sh = float(panel['k_equip_share_inner'].mean())
    else:
        sh = 0.5  # neutral default
    sigma_es = _allen_sigma_es(sigma_su, sigma_eu, sh)
    return KorvFit(
        sigma_su=float(sigma_su), sigma_eu=float(sigma_eu),
        sigma_es=float(sigma_es),
        se_sigma_su=float(se[0]), se_sigma_eu=float(se[1]),
        obj_value=float(res2.fun),
        hansen_J=None, hansen_p=None,
        n_obs=int(n), n_country=1, converged=bool(res2.success),
    )


def fit_korv_pooled(panel: pd.DataFrame) -> KorvFit:
    """Panel-pooled GMM with country-clustered weighting.

    `panel` must have a 'code' column plus the moment-input columns from
    fit_korv_country.
    """
    _moment_cols = ('dlog_ls_lu', 'dlog_ws_wu', 'dlog_ke_lu',
                    'dlog_wu_pk', 'dlog_lsushare', 'dlog_pk_pc')
    dat = {c: panel[c].to_numpy() for c in _moment_cols}
    n = len(panel)
    n_c = panel['code'].nunique()
    W0 = np.eye(3)
    res = minimize(
        _gmm_objective, x0=np.array([0.7, 1.5]),
        args=(dat, W0), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000},
    )
    # Two-step with country-clustered Omega.
    # Each country's contribution is the mean of its moment rows.
    m_mat = _moments(res.x, dat)
    cluster_df = pd.DataFrame(m_mat, columns=['m1', 'm2', 'm3'])
    cluster_df['code'] = panel['code'].values
    cluster_means = cluster_df.groupby('code')[['m1', 'm2', 'm3']].mean()
    # Cluster-robust covariance of moments.
    Omega_cluster = (cluster_means.values.T @ cluster_means.values) / n_c
    Omega = Omega_cluster + 1e-8 * np.eye(3)
    W1 = np.linalg.inv(Omega)
    res2 = minimize(
        _gmm_objective, x0=res.x,
        args=(dat, W1), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000},
    )
    sigma_su, sigma_eu = res2.x
    # Hansen J at the efficient estimate.
    # Over-identifying restrictions: 3 moments - 2 params = 1 restriction.
    m_bar = _moments(res2.x, dat).mean(axis=0)
    J = float(n * (m_bar @ W1 @ m_bar))
    from scipy.stats import chi2
    p_J = float(chi2.sf(J, df=1))
    se = _two_step_se(res2.x, dat, W1, n)
    if 'k_equip_share_inner' in panel.columns:
        sh = float(panel['k_equip_share_inner'].mean())
    else:
        sh = 0.5
    sigma_es = _allen_sigma_es(sigma_su, sigma_eu, sh)
    return KorvFit(
        sigma_su=float(sigma_su), sigma_eu=float(sigma_eu),
        sigma_es=float(sigma_es),
        se_sigma_su=float(se[0]), se_sigma_eu=float(se[1]),
        obj_value=float(res2.fun),
        hansen_J=J, hansen_p=p_J,
        n_obs=int(n), n_country=int(n_c), converged=bool(res2.success),
    )


def _moments_m1m2(theta: np.ndarray, dat: dict) -> np.ndarray:
    """Two-moment variant of _moments: only m1 (skill ratio) and m2 (equipment ratio).

    Drops m3 (labor-share FOC).  Returns (n_obs, 2) matrix.

    g1 = Δlog(w_s/w_u) · (Δlog(L_s/L_u) + σ_su · Δlog(w_s/w_u))
    g2 = Δlog(w_u/P_K) · (Δlog(K_e/L_u) − σ_eu · Δlog(w_u/P_K))
    """
    sigma_su, sigma_eu = theta
    r1 = dat['dlog_ls_lu'] + sigma_su * dat['dlog_ws_wu']
    r2 = dat['dlog_ke_lu'] - sigma_eu * dat['dlog_wu_pk']
    return np.column_stack([dat['dlog_ws_wu'] * r1, dat['dlog_wu_pk'] * r2])


def _gmm_objective_2mom(theta: np.ndarray, dat: dict, W: np.ndarray) -> float:
    """Quadratic GMM objective for the 2-moment (m1, m2) system."""
    g = _moments_m1m2(theta, dat).mean(axis=0)
    return float(g @ W @ g)


def _two_step_se_2mom(res_x: np.ndarray, dat: dict, W1: np.ndarray, n: int
                      ) -> np.ndarray:
    """GMM sandwich SEs for the 2-moment system via numerical Jacobian."""
    eps = 1e-5
    G = np.zeros((2, 2))
    for j, dx in enumerate([np.array([eps, 0.0]), np.array([0.0, eps])]):
        m_plus  = _moments_m1m2(res_x + dx, dat).mean(axis=0)
        m_minus = _moments_m1m2(res_x - dx, dat).mean(axis=0)
        G[:, j] = (m_plus - m_minus) / (2 * eps)
    try:
        var = np.linalg.inv(G.T @ W1 @ G) / n
        se = np.sqrt(np.diag(var))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    return se


def fit_korv_pooled_2moments(panel: pd.DataFrame) -> KorvFit:
    """Panel-pooled GMM using only m1 and m2 (drop labor-share FOC m3).

    Just-identified: 2 moments, 2 parameters → Hansen J is undefined (None).
    `panel` must have a 'code' column plus columns:
    dlog_ls_lu, dlog_ws_wu, dlog_ke_lu, dlog_wu_pk.
    The columns dlog_lsushare and dlog_pk_pc are not required.
    """
    _moment_cols = ('dlog_ls_lu', 'dlog_ws_wu', 'dlog_ke_lu', 'dlog_wu_pk')
    dat = {c: panel[c].to_numpy() for c in _moment_cols}
    n = len(panel)
    n_c = panel['code'].nunique()
    W0 = np.eye(2)
    res = minimize(
        _gmm_objective_2mom, x0=np.array([0.7, 1.5]),
        args=(dat, W0), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000},
    )
    # Two-step with country-clustered Omega.
    m_mat = _moments_m1m2(res.x, dat)
    cluster_df = pd.DataFrame(m_mat, columns=['m1', 'm2'])
    cluster_df['code'] = panel['code'].values
    cluster_means = cluster_df.groupby('code')[['m1', 'm2']].mean()
    Omega_cluster = (cluster_means.values.T @ cluster_means.values) / n_c
    Omega = Omega_cluster + 1e-8 * np.eye(2)
    W1 = np.linalg.inv(Omega)
    res2 = minimize(
        _gmm_objective_2mom, x0=res.x,
        args=(dat, W1), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000},
    )
    sigma_su, sigma_eu = res2.x
    se = _two_step_se_2mom(res2.x, dat, W1, n)
    if 'k_equip_share_inner' in panel.columns:
        sh = float(panel['k_equip_share_inner'].mean())
    else:
        sh = 0.5
    sigma_es = _allen_sigma_es(sigma_su, sigma_eu, sh)
    return KorvFit(
        sigma_su=float(sigma_su), sigma_eu=float(sigma_eu),
        sigma_es=float(sigma_es),
        se_sigma_su=float(se[0]), se_sigma_eu=float(se[1]),
        obj_value=float(res2.fun),
        hansen_J=None, hansen_p=None,
        n_obs=int(n), n_country=int(n_c), converged=bool(res2.success),
    )


def build_usercost_column(
    panel: pd.DataFrame,
    *,
    pk_col: str = 'p_k',
    r_col: str | None = None,
    r_bar: float = 0.04,
    delta_e: float = 0.13,
) -> pd.Series:
    """Compute Hall-Jorgenson user-cost of equipment for each row.

    r_e_t = P_K_t · (r_t + δ_e − E_t[Δlog P_K_{t+1}])

    where:
    - ``r_col`` (optional): per-row real interest rate column in `panel`.
      Falls back to ``r_bar`` when ``r_col`` is None or for rows where it
      is NaN.
    - ``delta_e``: depreciation rate (default 0.13, BEA equipment).
    - E_t[Δlog P_K] approximated by perfect-foresight: Δlog P_K_{t+1}
      (one-step-ahead, sorted within each country by year).

    The panel must contain a 'code' column for within-country sorting and
    differencing.  Returns the ``log r_e`` series (log of user cost) aligned
    to `panel`'s index.  Rows where cap_factor = r_t + δ_e − E[ΔlogP_K] ≤ 0
    are set to NaN (drop them before passing to GMM).

    Parameters
    ----------
    panel : pd.DataFrame
        Must have 'code' and ``pk_col`` columns, sorted by (code, year).
    pk_col : str
        Column name for the equipment price index (in level, not log).
    r_col : str or None
        Column name for per-row annualized real interest rate (fraction, e.g.
        0.04 not 4.0).  If None, uses ``r_bar`` everywhere.
    r_bar : float
        Fallback interest rate (default 0.04 = 4 % p.a.).
    delta_e : float
        Equipment depreciation rate (default 0.13).

    Returns
    -------
    pd.Series
        ``log_r_e`` series indexed as `panel`.  NaN where cap_factor ≤ 0.
    """
    df = panel.copy()
    df['_log_pk'] = np.log(df[pk_col].clip(lower=1e-9))
    df['_dlog_pk_forward'] = df.groupby('code')['_log_pk'].transform(
        lambda s: s.diff().shift(-1)
    )
    if r_col is not None and r_col in df.columns:
        r_t = df[r_col].fillna(r_bar)
    else:
        r_t = r_bar
    cap_factor = r_t + delta_e - df['_dlog_pk_forward']
    log_r_e = df['_log_pk'] + np.log(cap_factor.where(cap_factor > 0))
    return log_r_e.rename('log_r_e')


def fit_korv_pooled_usercost(
    panel: pd.DataFrame,
    *,
    pk_col: str = 'p_k',
    wu_col: str = 'w_u',
    r_col: str | None = None,
    r_bar: float = 0.04,
    delta_e: float = 0.13,
) -> KorvFit:
    """Panel-pooled GMM replacing asset price P_K with Hall-Jorgenson user cost.

    Constructs ``dlog_wu_re = Δlog(w_u / r_e)`` where r_e is the equipment
    user cost, and uses it in place of ``dlog_wu_pk`` for m2.  m1 and m3 are
    unchanged (m3 still uses P_K via dlog_pk_pc).  The system retains 3
    moments and 2 parameters (1 over-identifying restriction; J ~ χ²(1)).

    Parameters
    ----------
    panel : pd.DataFrame
        Must have a 'code' column and all standard moment-input columns, PLUS
        the level columns ``pk_col`` and ``wu_col`` in order to build the
        user-cost series.  Rows must be sorted by (code, year) before passing.
    pk_col, wu_col : str
        Level column names for the equipment price index and unskilled wage.
    r_col : str or None
        Per-row real interest rate; falls back to ``r_bar`` when None or NaN.
    r_bar : float
        Fallback annualized real interest rate (default 0.04).
    delta_e : float
        Equipment depreciation rate (default 0.13).

    Returns
    -------
    KorvFit
        Same structure as fit_korv_pooled.  If the user-cost construction
        removes too many rows (< 8 per country on average), the estimator
        may return poorly-identified results.
    """
    df = panel.copy()
    log_r_e = build_usercost_column(
        df, pk_col=pk_col, r_col=r_col, r_bar=r_bar, delta_e=delta_e,
    )
    df['_log_w_u'] = np.log(df[wu_col].clip(lower=1e-9))
    log_ratio = df['_log_w_u'] - log_r_e
    df['dlog_wu_pk'] = log_ratio.groupby(df['code']).diff()

    _moment_cols = ('dlog_ls_lu', 'dlog_ws_wu', 'dlog_ke_lu',
                    'dlog_wu_pk', 'dlog_lsushare', 'dlog_pk_pc')
    df_clean = df.dropna(subset=list(_moment_cols))
    return fit_korv_pooled(df_clean)


def fit_sigma_su_pooled(
    panel: pd.DataFrame,
    *,
    iv_col: str = 'dlog_ns_nu',
    lhs_col: str = 'dlog_ls_lu',
    rhs_col: str = 'dlog_ws_wu',
) -> SymCESFit:
    """Block A: pooled 2SLS for σ_su from m_1 alone.

    m_1: Δlog(L_s/L_u) = -σ_su · Δlog(w_s/w_u) + ε

    CES relative demand slopes DOWN: relative skilled employment falls when the
    skill premium rises. The 2SLS slope beta therefore estimates -σ_su, and
    σ_su = -beta. A positive beta -- relative quantity rising with relative
    price -- traces a supply relation rather than the demand curve, and should
    be read as a failure of the supply-shift instrument, not as a large σ_su.

    Identification: instrument Δlog(w_s/w_u) with Δlog(N_s/N_u), the
    cohort-population skilled-share change (Katz–Murphy IV).

    Returns a SymCESFit with sigma_ie = sigma_kl = NaN; only sigma_su and
    its SE are populated. hansen_J = None (just-identified single eqn).
    """
    needed = ('code', iv_col, lhs_col, rhs_col)
    df = panel.dropna(subset=list(needed)).copy()
    # Within-country demeaning to absorb country FE.
    for col in (iv_col, lhs_col, rhs_col):
        df[col + '_demean'] = (
            df.groupby('code')[col].transform(lambda s: s - s.mean())
        )
    y  = df[lhs_col + '_demean'].to_numpy()
    x  = df[rhs_col + '_demean'].to_numpy()
    z  = df[iv_col + '_demean'].to_numpy()
    # 2SLS: β̂ = cov(z, y) / cov(z, x)
    sx = float(np.dot(z, x) / len(z))
    sy = float(np.dot(z, y) / len(z))
    if abs(sx) < 1e-12:
        return SymCESFit(
            sigma_kl=np.nan, sigma_ie=np.nan, sigma_su=np.nan,
            se_sigma_kl=np.nan, se_sigma_ie=np.nan, se_sigma_su=np.nan,
            obj_value=np.nan, hansen_J=None, hansen_p=None,
            n_obs=len(df), n_country=df['code'].nunique(), converged=False,
        )
    beta = sy / sx
    resid = y - beta * x
    # Country-clustered SE.
    df['_resid_z'] = resid * z
    cluster_sum = df.groupby('code')['_resid_z'].sum().to_numpy()
    var_num = float(np.dot(cluster_sum, cluster_sum)) / len(df) ** 2
    var_den = (np.mean(z * x)) ** 2
    se = float(np.sqrt(var_num / max(var_den, 1e-12)))
    return SymCESFit(
        sigma_kl=np.nan, sigma_ie=np.nan, sigma_su=float(-beta),
        se_sigma_kl=np.nan, se_sigma_ie=np.nan, se_sigma_su=se,
        obj_value=0.0, hansen_J=None, hansen_p=None,
        n_obs=len(df), n_country=df['code'].nunique(), converged=True,
    )


def fit_sigma_ie_pooled(
    panel: pd.DataFrame,
    *,
    lhs_col: str = 'dlog_ki_ke',
    rhs_col: str = 'dlog_re_ri',
) -> SymCESFit:
    """Block B: panel-FE OLS for σ_ie from the within-capital FOC m_2.

    m_2: Δlog(K_i/K_e) = σ_ie · Δlog(r_e/r_i) + ε

    Identification: relative-price variation in P_e/P_i after country-FE
    demeaning. This block uses OLS at quarterly frequency on
    investment-flow proxies.

    Returns SymCESFit with only sigma_ie and its SE populated.
    """
    needed = ('code', lhs_col, rhs_col)
    df = panel.dropna(subset=list(needed)).copy()
    for col in (lhs_col, rhs_col):
        df[col + '_demean'] = (
            df.groupby('code')[col].transform(lambda s: s - s.mean())
        )
    y = df[lhs_col + '_demean'].to_numpy()
    x = df[rhs_col + '_demean'].to_numpy()
    var_x = float(np.dot(x, x))
    if var_x < 1e-12:
        return SymCESFit(
            sigma_kl=np.nan, sigma_ie=np.nan, sigma_su=np.nan,
            se_sigma_kl=np.nan, se_sigma_ie=np.nan, se_sigma_su=np.nan,
            obj_value=np.nan, hansen_J=None, hansen_p=None,
            n_obs=len(df), n_country=df['code'].nunique(), converged=False,
        )
    beta = float(np.dot(x, y) / var_x)
    resid = y - beta * x
    df['_xr'] = x * resid
    cluster_sum = df.groupby('code')['_xr'].sum().to_numpy()
    var_num = float(np.dot(cluster_sum, cluster_sum))
    se = float(np.sqrt(var_num) / max(var_x, 1e-12))
    return SymCESFit(
        sigma_kl=np.nan, sigma_ie=float(beta), sigma_su=np.nan,
        se_sigma_kl=np.nan, se_sigma_ie=se, se_sigma_su=np.nan,
        obj_value=0.0, hansen_J=None, hansen_p=None,
        n_obs=len(df), n_country=df['code'].nunique(), converged=True,
    )


def calibrate_nest_weights(
    panel: pd.DataFrame,
    *,
    sigma: float,
    share_col: str,
    p_num_col: str,
    p_den_col: str,
    base_year: int,
    weight_col: str = 'alpha_c',
) -> pd.DataFrame:
    """Calibrate per-country CES inner-nest weight from a base-year cost share.

    The CES cost-share identity for the equipment share of the capital
    nest at year t is

        s_e = α · (P_e / P_K)^{1-σ_ie},

    where P_K is the CES dual unit cost of the inner aggregator. Inverting
    for α at base-year shares and prices gives

        α = s_e · (P_e / P_K)^{σ_ie - 1}.

    For the calibration we approximate P_K by the *other* inner-nest
    input price P_den (so that P_e / P_K ≈ P_num / P_den), giving the
    operational formula implemented here:

        α ≈ s_e · (P_num / P_den)^{σ - 1}.

    At σ = 1 (Cobb–Douglas) the relative-price correction vanishes and
    α = s_e exactly. The same routine is used to calibrate the labor-nest
    weight γ by passing the unskilled-share/wage columns instead.

    Returns the input DataFrame with a new column `weight_col` attached;
    the weight is constant per country (taken from `base_year`).

    Note: if `base_year` is missing for a country in the input panel,
    that country's `weight_col` will be NaN after the merge. Filter or
    impute before passing the result to `build_inner_nest_aggregators`.
    """
    base = panel[panel['year'] == base_year].copy()
    base['_ratio'] = base[p_num_col] / base[p_den_col]
    if abs(sigma - 1.0) < 1e-9:
        base[weight_col] = base[share_col]
    else:
        base[weight_col] = base[share_col] * base['_ratio'] ** (sigma - 1.0)
    keep = base[['code', weight_col]].drop_duplicates(subset='code')
    return panel.merge(keep, on='code', how='left')


def build_inner_nest_aggregators(
    panel: pd.DataFrame,
    *,
    sigma_ie: float,
    sigma_su: float,
) -> pd.DataFrame:
    """Construct CES inner-nest aggregators K, L, P_K, w from level series.

    Quantity aggregators (with ρ = 1 - 1/σ_ie, η = 1 - 1/σ_su):

        K = [α K_e^ρ + (1-α) K_i^ρ]^{1/ρ}
        L = [γ L_s^η + (1-γ) L_u^η]^{1/η}

    Price duals:

        P_K = [α P_e^{1-σ_ie} + (1-α) P_i^{1-σ_ie}]^{1/(1-σ_ie)}
        w   = [γ w_s^{1-σ_su} + (1-γ) w_u^{1-σ_su}]^{1/(1-σ_su)}

    At σ → 1 the aggregators reduce to Cobb-Douglas geometric means.

    Required input columns: code, K_e, K_i, L_s, L_u, P_e, P_i, w_s, w_u,
    alpha_c, gamma_c. Returns a copy with added columns K_agg, L_agg, P_K, w.
    """
    df = panel.copy()
    a = df['alpha_c'].to_numpy()
    g = df['gamma_c'].to_numpy()

    def _ces_q(x1, x2, s, sigma):
        if abs(sigma - 1.0) < 1e-9:
            return (x1 ** s) * (x2 ** (1 - s))
        rho = 1.0 - 1.0 / sigma
        return (s * x1 ** rho + (1 - s) * x2 ** rho) ** (1.0 / rho)

    def _ces_p(p1, p2, s, sigma):
        if abs(sigma - 1.0) < 1e-9:
            return (p1 ** s) * (p2 ** (1 - s))
        return (s * p1 ** (1 - sigma) + (1 - s) * p2 ** (1 - sigma)) ** (
            1.0 / (1 - sigma)
        )

    df['K_agg'] = _ces_q(df['K_e'].to_numpy(), df['K_i'].to_numpy(), a, sigma_ie)
    df['L_agg'] = _ces_q(df['L_s'].to_numpy(), df['L_u'].to_numpy(), g, sigma_su)
    df['P_K']   = _ces_p(df['P_e'].to_numpy(), df['P_i'].to_numpy(), a, sigma_ie)
    df['w']     = _ces_p(df['w_s'].to_numpy(), df['w_u'].to_numpy(), g, sigma_su)
    return df


def fit_sigma_kl_pooled(
    panel: pd.DataFrame,
    *,
    iv_col: str,
    m3_lhs: str = 'dlog_k_l',
    m3_rhs: str = 'dlog_w_r',
    m4_lhs: str = 'dlog_ls_ratio',
    m4_rhs: str = 'dlog_pk_pc',
) -> SymCESFit:
    """Block C: joint pooled GMM for σ_KL from m_3 and m_4.

    m_3: Δlog(K/L)       = σ_KL · Δlog(w/r) + ε_3
    m_4: Δlog(s_K/s_L)   = (1 - σ_KL) · Δlog(P_K/P_C) + ε_4

    m_4 is stated on the CAPITAL-TO-LABOR SHARE RATIO. The labor share obeys
    Δlog s_L = s_K (σ_KL - 1) Δlog(P_K/P_C) instead -- opposite in sign and
    scaled by the capital share -- so passing a labor-share column here
    inverts the implied departure from Cobb-Douglas. Build the ratio with
    :func:`log_share_ratio`.

    Instruments:
      - For m_3, demeaned RHS (no separate IV; the quantity-FOC variation
        is treated as exogenous after country-FE demeaning).
      - For m_4, ``iv_col`` (the cumulated structural ISTC shock, supplied
        externally).

    Two moments, one parameter → over-identified by 1 (Hansen J on df=1).
    """
    if m4_lhs == 'dlog_ls':
        raise ValueError(
            "m4_lhs='dlog_ls' passes a LABOR SHARE to a moment stated on the "
            "capital-to-labor share ratio, which inverts the sign of the "
            "implied sigma_KL. Pass 'dlog_ls_ratio' (see log_share_ratio), or "
            "rescale by the capital share explicitly."
        )
    needed = ('code', m3_lhs, m3_rhs, m4_lhs, m4_rhs, iv_col)
    df = panel.dropna(subset=list(needed)).copy()
    for col in (m3_lhs, m3_rhs, m4_lhs, m4_rhs, iv_col):
        df[col + '_d'] = (
            df.groupby('code')[col].transform(lambda s: s - s.mean())
        )
    n = len(df)

    def _gbar(sigma):
        r3 = df[m3_lhs + '_d'] - sigma * df[m3_rhs + '_d']
        r4 = df[m4_lhs + '_d'] - (1 - sigma) * df[m4_rhs + '_d']
        g3 = (df[m3_rhs + '_d'] * r3).mean()
        g4 = (df[iv_col + '_d']  * r4).mean()
        return np.array([g3, g4])

    res1 = minimize(
        lambda s: float(_gbar(s[0]) @ np.eye(2) @ _gbar(s[0])),
        x0=np.array([1.0]), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 2000},
    )
    sigma1 = float(res1.x[0])
    # Two-step: cluster-robust Omega.
    r3 = (df[m3_lhs + '_d'] - sigma1 * df[m3_rhs + '_d']).to_numpy()
    r4 = (df[m4_lhs + '_d'] - (1 - sigma1) * df[m4_rhs + '_d']).to_numpy()
    z3 = df[m3_rhs + '_d'].to_numpy()
    z4 = df[iv_col  + '_d'].to_numpy()
    m_mat = np.column_stack([z3 * r3, z4 * r4])
    cluster_df = pd.DataFrame(m_mat, columns=['m3', 'm4'])
    cluster_df['code'] = df['code'].values
    cluster_means = cluster_df.groupby('code')[['m3', 'm4']].mean()
    Omega = (cluster_means.values.T @ cluster_means.values) / cluster_means.shape[0]
    W1 = np.linalg.inv(Omega + 1e-8 * np.eye(2))
    res2 = minimize(
        lambda s: float(_gbar(s[0]) @ W1 @ _gbar(s[0])),
        x0=np.array([sigma1]), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 2000},
    )
    sigma_kl = float(res2.x[0])
    m_bar = _gbar(sigma_kl)
    J = float(n * (m_bar @ W1 @ m_bar))
    from scipy.stats import chi2
    p_J = float(chi2.sf(J, df=1))
    # Numerical Jacobian for SE.
    eps = 1e-5
    G = (_gbar(sigma_kl + eps) - _gbar(sigma_kl - eps)) / (2 * eps)
    var = 1.0 / (G @ W1 @ G) / n
    se = float(np.sqrt(max(var, 0.0)))
    return SymCESFit(
        sigma_kl=sigma_kl, sigma_ie=np.nan, sigma_su=np.nan,
        se_sigma_kl=se, se_sigma_ie=np.nan, se_sigma_su=np.nan,
        obj_value=float(res2.fun), hansen_J=J, hansen_p=p_J,
        n_obs=n, n_country=df['code'].nunique(), converged=bool(res2.success),
    )


def fit_symmetric_nested_ces_pooled(
    panel: pd.DataFrame,
    *,
    iv_su: str = 'dlog_ns_nu',
    iv_kl: str,
) -> SymCESFit:
    """Sequential Block A → B → C estimator for the symmetric nested CES.

    Returns a single SymCESFit with all three structural elasticities
    populated. Each block is run on the subset of `panel` rows for which
    that block's required columns are non-NaN; the three blocks may
    therefore have different effective sample sizes. n_obs and n_country
    in the returned fit reflect the Block C (σ_KL) sample, which is the
    headline.

    No joint Hansen J is computed here; for that use
    fit_symmetric_nested_ces_joint.
    """
    fit_su = fit_sigma_su_pooled(panel, iv_col=iv_su)
    fit_ie = fit_sigma_ie_pooled(panel)
    fit_kl = fit_sigma_kl_pooled(panel, iv_col=iv_kl)
    converged = all([fit_su.converged, fit_ie.converged, fit_kl.converged])
    return SymCESFit(
        sigma_kl=fit_kl.sigma_kl,
        sigma_ie=fit_ie.sigma_ie,
        sigma_su=fit_su.sigma_su,
        se_sigma_kl=fit_kl.se_sigma_kl,
        se_sigma_ie=fit_ie.se_sigma_ie,
        se_sigma_su=fit_su.se_sigma_su,
        obj_value=fit_kl.obj_value,
        hansen_J=fit_kl.hansen_J,
        hansen_p=fit_kl.hansen_p,
        n_obs=fit_kl.n_obs,
        n_country=fit_kl.n_country,
        converged=converged,
    )


def _moments_symmetric(
    theta: np.ndarray, dat: dict,
) -> np.ndarray:
    """Stack four IV-moment conditions for the symmetric nested CES.

    theta = (sigma_kl, sigma_ie, sigma_su)
    Returns an (n_obs, 4) matrix of per-observation moment contributions.
    Identification:
      g1 (m_1, σ_su): IV = Δlog(N_s/N_u)
      g2 (m_2, σ_ie): self-instrumented (FE-demeaned dlog(r_e/r_i))
      g3 (m_3, σ_KL): self-instrumented (FE-demeaned dlog(w/r))
      g4 (m_4, σ_KL): IV = cumulated ISTC shock
    """
    sigma_kl, sigma_ie, sigma_su = theta
    r1 = dat['dlog_ls_lu'] + sigma_su * dat['dlog_ws_wu']
    r2 = dat['dlog_ki_ke'] - sigma_ie * dat['dlog_re_ri']
    r3 = dat['dlog_k_l']   - sigma_kl * dat['dlog_w_r']
    # m_4 in SHARE-RATIO form: dlog(s_K/s_L) = (1-sigma_KL) dlog(P_K/P_C).
    # The labor share does NOT satisfy this relation -- it satisfies
    # dlog s_L = s_K (sigma_KL - 1) dlog(P_K/P_C), which differs in sign and
    # carries a capital-share weight. Passing a labor share here inverts the
    # implied departure from Cobb-Douglas.
    r4 = dat['dlog_ls_ratio'] - (1 - sigma_kl) * dat['dlog_pk_pc']
    g1 = dat['z_su']      * r1
    g2 = dat['dlog_re_ri'] * r2
    g3 = dat['dlog_w_r']   * r3
    g4 = dat['z_kl']      * r4
    return np.column_stack([g1, g2, g3, g4])


def fit_symmetric_nested_ces_joint(
    panel: pd.DataFrame,
    *,
    iv_su: str,
    iv_kl: str,
) -> SymCESFit:
    """Block D: joint GMM over all four moments with Hansen J on df=1.

    Inputs must have columns dlog_ls_lu, dlog_ws_wu, dlog_ki_ke,
    dlog_re_ri, dlog_k_l, dlog_w_r, dlog_ls_ratio, dlog_pk_pc plus iv_su,
    iv_kl and a 'code' column. NaN rows are dropped.

    ``dlog_ls_ratio`` is the change in log(s_K/s_L), NOT the change in the
    labor share; build it with :func:`log_share_ratio`.
    """
    cols = ('code', 'dlog_ls_lu', 'dlog_ws_wu', 'dlog_ki_ke', 'dlog_re_ri',
            'dlog_k_l', 'dlog_w_r', 'dlog_ls_ratio', 'dlog_pk_pc',
            iv_su, iv_kl)
    df = panel.dropna(subset=list(cols)).copy()
    # Country-demean.
    for c in cols[1:]:
        df[c + '_d'] = df.groupby('code')[c].transform(lambda s: s - s.mean())
    dat = {
        'dlog_ls_lu':  df['dlog_ls_lu_d'].to_numpy(),
        'dlog_ws_wu':  df['dlog_ws_wu_d'].to_numpy(),
        'dlog_ki_ke':  df['dlog_ki_ke_d'].to_numpy(),
        'dlog_re_ri':  df['dlog_re_ri_d'].to_numpy(),
        'dlog_k_l':    df['dlog_k_l_d'].to_numpy(),
        'dlog_w_r':    df['dlog_w_r_d'].to_numpy(),
        'dlog_ls_ratio': df['dlog_ls_ratio_d'].to_numpy(),
        'dlog_pk_pc':  df['dlog_pk_pc_d'].to_numpy(),
        'z_su':        df[iv_su + '_d'].to_numpy(),
        'z_kl':        df[iv_kl + '_d'].to_numpy(),
    }
    n = len(df)
    n_c = df['code'].nunique()
    W0 = np.eye(4)
    res = minimize(
        lambda th: float(_moments_symmetric(th, dat).mean(axis=0)
                         @ W0 @ _moments_symmetric(th, dat).mean(axis=0)),
        x0=np.array([1.0, 0.6, 2.0]), method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 8000},
    )
    # Two-step with country-clustered Omega.
    m_mat = _moments_symmetric(res.x, dat)
    cluster_df = pd.DataFrame(m_mat, columns=['m1', 'm2', 'm3', 'm4'])
    cluster_df['code'] = df['code'].values
    cluster_means = cluster_df.groupby('code')[['m1', 'm2', 'm3', 'm4']].mean()
    Omega = (cluster_means.values.T @ cluster_means.values) / n_c
    W1 = np.linalg.inv(Omega + 1e-8 * np.eye(4))
    res2 = minimize(
        lambda th: float(_moments_symmetric(th, dat).mean(axis=0)
                         @ W1 @ _moments_symmetric(th, dat).mean(axis=0)),
        x0=res.x, method='Nelder-Mead',
        options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 8000},
    )
    th = res2.x
    m_bar = _moments_symmetric(th, dat).mean(axis=0)
    J = float(n * (m_bar @ W1 @ m_bar))
    from scipy.stats import chi2
    p_J = float(chi2.sf(J, df=1))
    # Sandwich SEs.
    eps = 1e-5
    G = np.zeros((4, 3))
    for j, dxv in enumerate([
        np.array([eps, 0, 0]), np.array([0, eps, 0]), np.array([0, 0, eps]),
    ]):
        G[:, j] = (
            _moments_symmetric(th + dxv, dat).mean(axis=0)
            - _moments_symmetric(th - dxv, dat).mean(axis=0)
        ) / (2 * eps)
    try:
        var = np.linalg.inv(G.T @ W1 @ G) / n
        se = np.sqrt(np.diag(var))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    return SymCESFit(
        sigma_kl=float(th[0]), sigma_ie=float(th[1]), sigma_su=float(th[2]),
        se_sigma_kl=float(se[0]), se_sigma_ie=float(se[1]),
        se_sigma_su=float(se[2]),
        obj_value=float(res2.fun), hansen_J=J, hansen_p=p_J,
        n_obs=n, n_country=n_c, converged=bool(res2.success),
    )


__all__ = [
    'KorvFit',
    'SymCESFit',
    'fit_korv_country',
    'fit_korv_pooled',
    'fit_korv_pooled_2moments',
    'fit_korv_pooled_usercost',
    'fit_sigma_su_pooled',
    'fit_sigma_ie_pooled',
    'fit_sigma_kl_pooled',
    'fit_symmetric_nested_ces_pooled',
    'fit_symmetric_nested_ces_joint',
    'build_usercost_column',
    'calibrate_nest_weights',
    'build_inner_nest_aggregators',
    '_allen_sigma_es',
    '_moments',
    '_moments_m1m2',
    '_moments_symmetric',
]
