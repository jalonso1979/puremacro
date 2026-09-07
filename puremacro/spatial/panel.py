"""Spatial panel models with fixed effects and the Lee-Yu bias correction.

:func:`spatial_panel` estimates, by (quasi-)maximum likelihood on a
**balanced** ``(entity, time)`` panel,

===== =========================================================================
SAR   ``y_it = rho * sum_j w_ij y_jt + x_it' beta + mu_i + xi_t + eps_it``
SDM   ``y_it = rho * sum_j w_ij y_jt + x_it' beta + (W x)_it' theta
      + mu_i + xi_t + eps_it``
SEM   ``y_it = x_it' beta + mu_i + xi_t + u_it``,
      ``u_it = lam * sum_j w_ij u_jt + eps_it``
===== =========================================================================

with ``effects`` selecting which of the individual effects ``mu_i`` and the
time effects ``xi_t`` are present.

Why the fixed effects bias maximum likelihood
---------------------------------------------
The individual effects are ``n`` incidental parameters each estimated from
``T`` observations, and the time effects are ``T`` incidental parameters each
estimated from ``n`` observations. Concentrating them out of the Gaussian
likelihood (the ``method="direct"`` route) leaves a score that is not centred
at the truth: at ``theta_0``, with ``G = W (I_n - rho W)^-1``,

===================== =========================================================
``E[dL/dbeta]``       ``0``
``E[dL/dsigma2]``     ``-(n + T - 1) / (2 sigma^2)``
``E[dL/drho]``        ``-tr(G) - (T - 1) / (1 - rho)``
===================== =========================================================

for two-way effects; with individual effects only the pair is
``(-n/(2 sigma^2), -tr(G))`` and with time effects only it is
``(-T/(2 sigma^2), -T/(1 - rho))``. Divided by the ``O(nT)`` information this
is an ``O(1/T)`` bias from the individual effects plus an ``O(1/n)`` bias from
the time effects.

The two concentrated objectives, side by side
---------------------------------------------
Both routes run on the same doubly-demeaned arrays --- the spatial lag is
taken **before** any demeaning, because ``J_n S(rho) != S(rho) J_n`` --- and
differ only in the pair (Jacobian factor, variance divisor). Writing
``SSR(rho)`` for the concentrated sum of squares of the demeaned data,
``jT`` for the Jacobian factor and ``Nd`` for the divisor,

    ``loglik_c(rho) = -(Nd/2) (ln 2pi + 1) - (Nd/2) ln(SSR(rho)/Nd)
                      + jT * J(rho)``

============================ ============ ============= ======================
route                        ``jT``       ``Nd``        ``J(rho)``
============================ ============ ============= ======================
``method="direct"``          ``T``        ``n T``       ``ln|I_n - rho W|``
``"transformation"``, ent    ``T - 1``    ``n (T-1)``   ``ln|I_n - rho W|``
``"transformation"``, tim    ``T``        ``(n-1) T``   ``ln|I_n - rho W|
                                                        - ln(1 - rho)``
``"transformation"``, both   ``T - 1``    ``(n-1)(T-1)``  ``ln|I_n - rho W|
                                                          - ln(1 - rho)``
============================ ============ ============= ======================

(``ent`` = the individual effects are removed, ``tim`` = the time effects
are.) Getting the pair out of step --- demeaned data with divisor ``n(T-1)``
but Jacobian factor ``T``, say --- yields a *different and wrong* ``rho_hat``;
the two must move together, which is what
``test_direct_and_transformation_agree_on_beta_and_rho_under_individual_effects``
pins.

Lee & Yu (2010) remove both incidental-parameter biases exactly with an
orthonormal transformation. Let ``[F_{T,T-1}, l_T/sqrt(T)]`` be the eigenvector
matrix of ``J_T = I_T - l_T l_T'/T`` (``F'F = I_{T-1}``, ``FF' = J_T``): the
``T-1`` transformed periods ``Y F_{T,T-1}`` carry no ``mu_i`` and still have
iid ``N(0, sigma^2 I_n)`` errors. When every row of ``W`` sums to **one**,
``l_n`` is an eigenvector of ``W`` and ``[F_{n,n-1}, l_n/sqrt(n)]``
block-triangularises it, so ``|I_n - rho W| = (1 - rho) |I_{n-1} - rho F'WF|``
and the time effects disappear from an exact ``(n-1)``-unit SAR. Because
``FF' = J`` and ``F'F = I``, ``||F'v||^2 = ||J v||^2``: ``F`` is never
materialised, the estimator runs on demeaned data with the *effective* counts,
and the transformed score is centred by construction --- so
``bias_correction`` reports ``"none"`` because the correction is *provably
zero*, not because it was skipped.

For ``method="direct"`` (kept for comparability with software that does it,
and because its Jacobian needs no assumption on the row sums) the analytic
correction ``theta_corrected = theta_hat - I^-1 a`` is applied instead, with
``a`` the score bias above and ``I`` the expected information built with the
**effective** counts. With individual effects only that correction is provably
``(0_k, 0, -sigma^2/(T-1))``: it rescales ``sigma_hat^2 = SSR/(nT)`` by
``T/(T-1)`` --- exactly the transformation divisor --- and leaves ``beta_hat``
and ``rho_hat`` untouched, because the two concentrated objectives differ by
the positive affine map ``loglik* = ((T-1)/T)(loglik_direct + const)``. (The
magnitude matters: ``a/I`` carries the units of the parameter, so the
correction is ``O(sigma^2/T)``, never ``O(sigma^2)``.)

Standard errors
---------------
``vcov="oim"`` (the default) is the inverse expected information, valid under
iid Gaussian ``eps``. ``vcov="dk"`` (Driscoll-Kraay) and ``vcov="conley"``
replace **the beta block only**; the ``rho``/``lambda`` and ``sigma^2`` rows
and columns always come from the OIM. Two reasons, both deliberate:

* a per-cell outer product (a White or entity-clustered meat) is *wrong* for
  the spatial parameter. ``s_it^rho`` carries ``(Wy)_it e_it``, and
  ``Wy = G X beta + G eps`` loads on every neighbour's ``eps``; the omitted
  cross-``i`` terms are large. On a 6x6 row-standardised rook lattice at
  ``rho = 0.5`` the outer-product meat ``T(tr(G'G) + sum_i g_ii^2)`` is
  **54.0%** of the true score variance ``T(tr(G^2) + tr(G'G))`` --- both are
  linear in ``T``, so the ratio holds at every ``T`` (at ``T = 8``, 163.9
  against 303.5) --- understating the ``rho`` standard error by 26.5%.
  ``vcov="qml"`` and ``vcov="cluster"`` therefore do not exist here.
* the per-cell allocation of the Jacobian term ``-jT tr(G*)`` across units
  (as ``-(jT/T) diag(G*)_i``) sums to the exact score, but its cross-sectional
  split is a modelling choice, not a theorem. Driscoll-Kraay sums the scores
  across units *within a period* before squaring, so it sees only the period
  total and the split cancels for it **exactly**. Conley does **not**: its
  kernel-weighted within-period quadratic form ``U_t' K U_t`` is not a
  function of the period sum, so a different allocation of the same total is a
  different Conley meat. Measured on a 6x6 row-standardised rook lattice with
  ``T = 14`` at ``rho = 0.5``, re-allocating ``diag(G*)`` to its own mean (the
  same period total) moves the Driscoll-Kraay meat by ``2.8e-13`` against a
  largest entry of ``353`` and the Conley meat (cutoff 2.0, one time lag,
  euclidean) by ``8.2`` against a largest entry of ``513`` --- 1.6%. The split
  therefore *does* enter the Conley covariance; it reaches the reported
  ``beta`` standard errors only through the small ``beta``-``rho`` block of
  the bread, which moved them by 0.08% and 0.005% in that experiment.
  ``test_the_jacobian_split_cancels_for_dk_but_not_for_conley`` pins both
  halves of that sentence. Either way the contract is the same: report the
  robust covariance where it is unambiguous (the ``beta`` block) and the OIM
  elsewhere.

Under heteroskedastic ``eps`` the SAR QMLE **point estimate** is inconsistent
(Lin & Lee 2010), so no sandwich rescues it; Lee (2004)'s QML robustness is to
non-normality only. And no covariance of any kind is robust to a mis-specified
``W``: if ``W`` is wrong, ``rho_hat`` is inconsistent.

Deliberately omitted
--------------------
* **Dynamic spatial panels** (a time-lagged ``y`` and/or ``W y_{t-1}``). Yu,
  de Jong & Lee (2008) is cited for the ``n/T -> c`` asymptotics behind the
  ``O(1/T)`` bias term, *not* implemented.
* **Random effects** (Baltagi-Egger-Pfaffermayr) and Hausman-type tests.
* **SARAR / SAC** (a lag *and* a spatially correlated error) and GMM /
  spatial-2SLS estimation. This module is quasi-ML only.
* **SDEM** (``model="sem"`` with ``durbin=``): the signature would reach it,
  so it raises rather than fitting an equation this docstring does not define.
* **Unbalanced panels.** The transformation approach needs a common ``T`` for
  ``F_{T,T-1}``; an unbalanced panel raises and names
  :func:`puremacro.inference.balanced_panel.balanced_subpanel`.
* **``logdet="lu"`` and every stochastic trace estimator.** The information
  matrix, the bias vector and the impacts all need ``tr(G)``, ``tr(G^2)``,
  ``tr(G'G)`` and ``diag(G)``, which an LU factorisation does not provide; a
  Hutchinson probe would put unseeded randomness into a point estimate. The
  panel path therefore uses the exact dense spectrum: ``O(n^2)`` memory and
  ``O(n^3)`` time, the same budget :mod:`puremacro.spatial.hac` already
  documents. A :class:`RuntimeWarning` fires above 2000 units and
  ``dense_max`` (5000) is a hard ceiling. This is a deliberate limit for
  *regional* panels; a 3143-county panel is near the ceiling and census tracts
  are past it.
* **``vcov="qml"`` / ``vcov="cluster"``** --- see above; they are not merely
  unavailable, they are wrong for this score.
* **Sharing :func:`puremacro.spatial.models.spatial_effects`.** The impact
  decomposition below is a private, vectorised closed form rather than a call
  into ``spatial_effects``. Deliberate, and named here so the two do not drift
  silently: ``spatial_effects`` evaluates its scalar terms *per draw* in a
  Python loop and reaches ``1'A^-1 1`` through a truncated ``rho^q`` power
  series (falling back to a dense solve per draw when the series does not
  converge), whereas the panel needs those terms for every one of ``n_draws``
  draws and gets them from the already-computed spectrum in one vectorised
  expression. The reported *intervals* also follow different conventions:
  ``spatial_effects`` reports empirical simulation quantiles and simulation
  p-values, this module reports ``point +/- z_{1-alpha/2} se``. The **point
  estimates are the same function of ``(beta, theta, rho, W)``**, and
  ``test_panel_impacts_agree_with_models_spatial_effects`` pins them against
  ``spatial_effects`` to 1e-12, so a future consolidation is a mechanical
  integration step rather than a re-derivation.

Everything runs on numpy / scipy / pandas: no geometry or econometrics stack.

References
----------
Lee, L.-F. and Yu, J. (2010). Estimation of spatial autoregressive panel data
    models with fixed effects. Journal of Econometrics 154(2), 165-185.
Lee, L.-F. and Yu, J. (2010). Some recent developments in spatial panel data
    models. Regional Science and Urban Economics 40(5), 255-271.
Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood
    estimators for spatial autoregressive models. Econometrica 72(6), 1899-1925.
Lin, X. and Lee, L.-F. (2010). GMM estimation of spatial autoregressive models
    with unknown heteroskedasticity. Journal of Econometrics 157(1), 34-52.
Yu, J., de Jong, R. and Lee, L.-F. (2008). Quasi-maximum likelihood estimators
    for spatial dynamic panel data models with fixed effects when both n and T
    are large. Journal of Econometrics 146(1), 118-134.
Elhorst, J.P. (2014). Spatial Econometrics: From Cross-Sectional Data to
    Spatial Panels. Springer, chapter 3.
Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.
LeSage, J. and Pace, R.K. (2009). Introduction to Spatial Econometrics. CRC,
    chapters 2 and 4.
Pace, R.K. and Barry, R. (1997). Quick computation of spatial autoregressive
    estimators. Geographical Analysis 29(3), 232-247.
Driscoll, J.C. and Kraay, A.C. (1998). Consistent covariance matrix estimation
    with spatially dependent panel data. REStat 80(4), 549-560.
Conley, T.G. (1999). GMM estimation with cross sectional dependence.
    Journal of Econometrics 92(1), 1-45.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy import optimize, stats

from .._linalg import inv_xtx
from ..lp._panel_helpers import as_panel_index
from .hac import _canonical_cov_token
from .weights import SpatialWeights

__all__ = ["spatial_panel", "SpatialPanelResult"]

_MODELS = ("sar", "sem", "sdm")
_EFFECTS = ("none", "individual", "time", "two-way")
_METHODS = ("transformation", "direct")
_BIAS = ("auto", "lee-yu", "none")
_VCOV = ("oim", "dk", "conley")
_LOGDET = ("auto", "eigen")

_VCOV_NOTES = {
    "oim": (
        "oim: inverse expected information; valid under iid Gaussian errors "
        "(Lee 2004 extends it to non-normal but homoskedastic errors)."
    ),
    "dk": (
        "dk: Driscoll-Kraay meat on the BETA block only (arbitrary "
        "cross-sectional dependence, needs a long T); the rho/lambda and "
        "sigma^2 rows are the OIM."
    ),
    "conley": (
        "conley: Conley (1999) space-time HAC on the BETA block only "
        "(distance-decaying dependence inside the cutoff); the rho/lambda and "
        "sigma^2 rows are the OIM."
    ),
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _render(df: pd.DataFrame, fmt: str, **kwargs: Any) -> str:
    from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst
    if fmt == "markdown":
        return _df_to_markdown(df, index=False, **kwargs)
    if fmt == "latex":
        return _df_to_latex(df, index=False, **kwargs)
    return _df_to_typst(df, index=False, **kwargs)


def _normalise_vcov(value: Any) -> Any:
    """Fold the shared covariance-token aliases onto this module's spellings.

    ``spatial_panel`` spells Driscoll-Kraay ``'dk'`` and Conley ``'conley'``;
    the projection estimators (:func:`puremacro.spatial.lp.spatial_lp`,
    :func:`puremacro.lp.panel_lp`) accept ``'driscoll-kraay'`` /
    ``'driscollkraay'`` and ``'spatial'`` / ``'spatial-hac'`` for the same two
    estimators.  Accept those spellings here too, so the vocabulary is the same
    everywhere even though the keyword name is not (``vcov=`` for the
    likelihood-based estimators, ``cov_type=`` for the projection ones).

    Anything else --- including the ``'cluster'`` family, which this module
    deliberately does not offer --- is returned untouched, so
    :func:`_check_enum` still reports the token the caller actually passed.
    """
    if not isinstance(value, str):
        return value
    canon = _canonical_cov_token(value)
    if canon == "driscoll-kraay":
        return "dk"
    if canon == "conley":
        return "conley"
    return value


def _check_enum(value: Any, allowed: tuple, argname: str) -> str:
    if not isinstance(value, str) or value.lower() not in allowed:
        raise ValueError(
            f"spatial_panel: {argname}={value!r} is not recognised; "
            f"valid values are {list(allowed)}"
        )
    return value.lower()


def _double_centre(A: np.ndarray) -> np.ndarray:
    """``J_n A J_n`` with ``J_n = I - l l'/n`` (rows and columns centred)."""
    return A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()


def _dk_bandwidth(n_periods: int) -> int:
    """``floor(4 (T/100)^(2/9))`` with a floor of 1 --- the same rule and the
    same ``floor`` as :func:`puremacro.lp._panel_helpers._focal_dk_se`."""
    return max(1, int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0))))


def _hac_meat_from_scores(
    scores: np.ndarray,
    entity_keys: np.ndarray,
    time_keys: np.ndarray,
    coords: Any,
    cutoff_km: float,
    time_lags: int,
    *,
    kernel: str = "bartlett",
    metric: str = "haversine",
) -> np.ndarray:
    """Conley (1999) / Hsiang (2010) space-time HAC meat for an arbitrary
    ``(N, p)`` score matrix.

    :func:`puremacro.spatial.hac.spatial_hac_panel_meat` hardcodes
    ``U = X * resid[:, None]``, which is the score of an *OLS* moment
    condition; a spatial ML score is not of that form (the ``rho`` block
    carries a Jacobian term and the ``sigma^2`` block a quadratic one), so the
    body is repeated here with the score passed in directly. Identical
    numerics --- see ``notes_for_integrator``: this is the natural candidate
    for a shared ``hac_meat_from_scores`` in ``spatial/hac.py``.
    """
    from .hac import _entity_coords, kernel_matrix
    from .weights import pairwise_distances

    U_flat = np.asarray(scores, dtype=float)
    ent = np.asarray(entity_keys)
    tim = np.asarray(time_keys)
    entities, ent_idx = np.unique(ent, return_inverse=True)
    periods, t_idx = np.unique(tim, return_inverse=True)
    n_e, n_t, p = len(entities), len(periods), U_flat.shape[1]
    C = _entity_coords(coords, entities)
    K = kernel_matrix(pairwise_distances(C, metric), float(cutoff_km), kernel)
    U = np.zeros((n_t, n_e, p))
    U[t_idx, ent_idx, :] = U_flat
    KU = np.einsum("ij,tjk->tik", K, U)
    S = np.zeros((p, p))
    for t in range(n_t):
        S += U[t].T @ KU[t]
    L = int(time_lags)
    for ell in range(1, L + 1):
        if ell >= n_t:
            break
        w = 1.0 - ell / (L + 1.0)
        Gam = np.zeros((p, p))
        for t in range(ell, n_t):
            Gam += U[t].T @ KU[t - ell]
        S += w * (Gam + Gam.T)
    return S


def _psd_draws(
    mean: np.ndarray,
    cov: np.ndarray,
    size: int,
    rng: np.random.Generator,
    func: str,
) -> np.ndarray:
    """Multivariate normal draws through an eigenvalue-clipped square root.

    ``np.random.Generator.multivariate_normal`` accepts a non-PSD covariance
    silently; a sandwich covariance can have a tiny negative eigenvalue. Clip
    to zero (which is conservative: it shrinks the drawn variance) and warn.
    """
    Sym = 0.5 * (cov + cov.T)
    w, Q = np.linalg.eigh(Sym)
    if np.any(w < -1e-10 * max(1.0, float(np.max(np.abs(w))))):
        warnings.warn(
            f"{func}: the covariance block used for the impact simulation is not "
            "positive semi-definite; negative eigenvalues were clipped to zero, "
            "so the simulated impact standard errors are conservative.",
            RuntimeWarning,
            stacklevel=3,
        )
    w = np.clip(w, 0.0, None)
    root = Q * np.sqrt(w)
    z = rng.standard_normal((size, len(mean)))
    return mean[None, :] + z @ root.T


# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SpatialPanelResult:
    """Result of :func:`spatial_panel`.

    Attributes
    ----------
    model, effects, method, bias_correction, vcov_type : str
        ``model`` in ``{'sar','sem','sdm'}``; ``effects`` in
        ``{'none','individual','time','two-way'}``; ``method`` in
        ``{'transformation','direct'}``; ``bias_correction`` is what was
        actually applied (``'lee-yu'`` or ``'none'``); ``vcov_type`` in
        ``{'oim','dk','conley'}``.
    vcov_note : str
        One line stating what the reported covariance is valid for.
    params : pandas.Series
        All estimates in the canonical order ``[beta..., rho|lambda, sigma2]``.
        Durbin terms are named ``'W.<col>'``.
    beta : pandas.Series
        The slope block of ``params``.
    rho, lam : float
        Spatial autoregressive parameter (``nan`` for ``model='sem'``) and
        spatial error parameter (``nan`` for ``'sar'``/``'sdm'``).
    sigma2, sigma2_dfc : float
        ML variance ``SSR / Nd`` (after any bias correction) and the
        degrees-of-freedom-corrected ``SSR / (N_eff - k)``, reported for
        reference only --- it is never used in the information matrix.
    vcov : numpy.ndarray
        ``(k+2, k+2)`` covariance of ``params``, same order.
    se, z, p_values : pandas.Series
    conf_int : pandas.DataFrame
        Columns ``['lower', 'upper']`` at ``1 - alpha``.
    bias : pandas.Series or None
        The Lee-Yu bias vector actually subtracted, in ``params`` order;
        ``None`` when no correction was applied.
    impacts : pandas.DataFrame or None
        LeSage-Pace decomposition; ``None`` for ``model='sem'`` (there is no
        spatial multiplier: the marginal effect *is* ``beta``) and when
        ``impacts=False``.
    loglik, loglik_null, lr_stat, lr_pvalue : float
        Concentrated log-likelihood at the optimum and at ``rho = 0``, the LR
        statistic ``2 (loglik - loglik_null)`` and its 1-df p-value.
    r2_within : float
        ``1 - SSR / (demeaned TSS)``.
    profile : pandas.DataFrame
        Columns ``['rho', 'loglik']`` on the 51-point bracketing grid (the
        ``'rho'`` column holds ``lambda`` for ``model='sem'``).
    resid : pandas.Series
        MultiIndex ``(entity, time)`` transformed residuals ``e_it``.
    entity_effects, time_effects : pandas.Series or None
        Recovered ``mu_i`` / ``xi_t``, each normalised to sum to zero (the
        two-way split is identified only up to a constant, which is
        ``intercept``).
    intercept : float
        The level the zero-mean ``entity_effects`` / ``time_effects`` are
        measured from. For ``effects='none'`` there are no such series and the
        level is a fitted coefficient, so this is exactly
        ``params['const']`` --- not the (machine-zero) mean of the residual.
    n_entities, n_periods, n_obs : int
    n_eff, n_eff_entities, n_eff_periods, df_resid : int
        Effective counts ``N* = n* T*`` used by the transformation likelihood
        and by the information matrix.
    n_nonpsd : int
        Number of ``params`` entries whose sandwich variance came out
        non-positive (their ``se``/``z``/``p``/CI are ``nan``).
    ids : tuple
    rho_bounds : tuple
        The interval the spatial parameter was actually searched over: the
        user's ``rho_bounds`` when given, otherwise ``stability_bounds``
        shrunk by ``1e-6`` at each end.
    stability_bounds, invertible_bounds : tuple
        Both intervals, as the package's spectrum rule requires.
        ``invertible_bounds`` is ``(1/lam_min, 1/lam_max)`` over the **real**
        eigenvalues (a complex eigenvalue never makes ``I - rho W`` singular
        for a real ``rho``), intersected with ``(-1, 1)`` when ``W`` is
        row-standardised: outside it ``S(rho)`` is singular, and a user
        ``rho_bounds`` reaching outside it raises. ``stability_bounds`` is
        that interval further intersected with ``(-1/max|lam_i|,
        1/max|lam_i|)`` over the **full** (possibly complex) spectrum, on
        which the spatial multiplier series converges; it is the default
        search interval, and a user ``rho_bounds`` reaching outside it only
        warns.
    logdet_method : str
    weights_kind : str
    converged : bool
        ``True`` only when the bounded optimiser reported success *and* its
        optimum beat the 51-point profile grid. A run whose reported spatial
        parameter had to be snapped back to a grid point, or that stopped
        within ``1e-4`` of a search bound, reports ``False``.
    n_evals, n_draws, n_rejected, seed : int
    alpha : float
    """

    model: str
    effects: str
    method: str
    bias_correction: str
    vcov_type: str
    vcov_note: str
    params: pd.Series
    beta: pd.Series
    rho: float
    lam: float
    sigma2: float
    sigma2_dfc: float
    vcov: np.ndarray
    se: pd.Series
    z: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    bias: pd.Series | None
    impacts: pd.DataFrame | None
    loglik: float
    loglik_null: float
    lr_stat: float
    lr_pvalue: float
    r2_within: float
    profile: pd.DataFrame
    resid: pd.Series
    entity_effects: pd.Series | None
    time_effects: pd.Series | None
    intercept: float
    n_entities: int
    n_periods: int
    n_obs: int
    n_eff: int
    n_eff_entities: int
    n_eff_periods: int
    df_resid: int
    n_nonpsd: int
    ids: tuple
    rho_bounds: tuple
    stability_bounds: tuple
    invertible_bounds: tuple
    logdet_method: str
    weights_kind: str
    converged: bool
    n_evals: int
    n_draws: int
    n_rejected: int
    seed: int
    alpha: float

    # -- presentation -------------------------------------------------------
    @property
    def spatial_param_name(self) -> str:
        """``'rho'`` for SAR/SDM, ``'lambda'`` for SEM."""
        return "lambda" if self.model == "sem" else "rho"

    def to_frame(self) -> pd.DataFrame:
        """Coefficient table: ``['term','coef','se','z','p_value','ci_lower','ci_upper']``."""
        return pd.DataFrame({
            "term": list(self.params.index),
            "coef": self.params.to_numpy(dtype=float),
            "se": self.se.to_numpy(dtype=float),
            "z": self.z.to_numpy(dtype=float),
            "p_value": self.p_values.to_numpy(dtype=float),
            "ci_lower": self.conf_int["lower"].to_numpy(dtype=float),
            "ci_upper": self.conf_int["upper"].to_numpy(dtype=float),
        })

    def impacts_frame(self) -> pd.DataFrame:
        """Long form of the LeSage-Pace decomposition.

        Columns ``['variable','direct','direct_se','indirect','indirect_se',
        'total','total_se','total_p']``. Empty (with those columns) when there
        are no impacts --- ``model='sem'`` or ``impacts=False``.
        """
        cols = ["variable", "direct", "direct_se", "indirect", "indirect_se",
                "total", "total_se", "total_p"]
        if self.impacts is None:
            return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
        imp = self.impacts
        with np.errstate(divide="ignore", invalid="ignore"):
            zt = imp["total"].to_numpy(float) / imp["total_se"].to_numpy(float)
        return pd.DataFrame({
            "variable": list(imp.index),
            "direct": imp["direct"].to_numpy(float),
            "direct_se": imp["direct_se"].to_numpy(float),
            "indirect": imp["indirect"].to_numpy(float),
            "indirect_se": imp["indirect_se"].to_numpy(float),
            "total": imp["total"].to_numpy(float),
            "total_se": imp["total_se"].to_numpy(float),
            "total_p": 2.0 * stats.norm.sf(np.abs(zt)),
        })

    def summary(self) -> str:
        """Multi-line text report: header, coefficients, impacts, caveats."""
        removal = ("Lee-Yu transformation approach"
                   if self.method == "transformation" else "direct approach")
        head = (f"Spatial panel: {self.model.upper()} with {self.effects} effects "
                f"({removal})")
        sp_name = self.spatial_param_name
        sp_val = self.lam if self.model == "sem" else self.rho
        se_sp = float(self.se.get(sp_name, np.nan))
        z_sp = float(self.z.get(sp_name, np.nan))
        p_sp = float(self.p_values.get(sp_name, np.nan))
        lines = [
            head,
            f"  units {self.n_entities} (weights: {self.weights_kind})"
            f"   periods {self.n_periods}   observations {self.n_obs}"
            f"   effective {self.n_eff} = {self.n_eff_entities} x {self.n_eff_periods}",
            f"  {sp_name} = {sp_val:+.4f} (se {se_sp:.4f}, z {z_sp:+.2f}, p {p_sp:.4f})"
            f"   sigma^2 = {self.sigma2:.4f}   within R^2 = {self.r2_within:.3f}",
            f"  log-likelihood {self.loglik:.2f}"
            f"   LR({sp_name}=0) = {self.lr_stat:.2f} (p {self.lr_pvalue:.4f})"
            f"   converged: {self.converged}",
            "",
            _render(self.to_frame(), "markdown"),
        ]
        if self.impacts is not None:
            lines += [
                "",
                f"LeSage-Pace effects ({self.n_draws} draws, seed {self.seed}"
                + (f", {self.n_rejected} rejected" if self.n_rejected else "")
                + ")",
                _render(self.impacts_frame(), "markdown"),
            ]
        elif self.model == "sem":
            lines += ["", "No impact decomposition: SEM has no spatial multiplier, "
                          "so the marginal effect of x is beta itself."]
        lines.append("")
        if self.bias_correction == "lee-yu" and self.bias is not None:
            terms = ", ".join(
                f"{k} {v:+.4f}" for k, v in self.bias.items() if abs(float(v)) > 1e-12
            ) or "nothing (all components are numerically zero)"
            lines.append(f"Lee-Yu analytic correction applied; subtracted: {terms}.")
        elif self.method == "transformation":
            lines.append(
                "No bias correction: the transformation approach has an exactly "
                "centred score, so the Lee-Yu correction is provably zero (it was "
                "not skipped)."
            )
        elif self.effects == "none":
            lines.append(
                "No bias correction: effects='none' leaves no incidental parameters, "
                "so there is no Lee-Yu bias to correct."
            )
        else:
            lines.append(
                "No bias correction applied, and the direct approach needs one: "
                "concentrating out the incidental parameters leaves "
                f"{self.spatial_param_name} and sigma^2 with an O(1/T) + O(1/n) bias "
                "(on a 3x3 lattice with T = 40 and time effects the uncorrected rho "
                "averages 0.29 against a true 0.50). Use bias_correction='lee-yu' or "
                "method='transformation'."
            )
        lines.append(f"Covariance -- {self.vcov_note}")
        lines.append(
            "No covariance is robust to a mis-specified W: if W is wrong the "
            "spatial parameter is inconsistent and no sandwich repairs it. Under "
            "heteroskedastic errors the SAR QMLE point estimate itself is "
            "inconsistent (Lin & Lee 2010)."
        )
        if self.n_nonpsd:
            lines.append(
                f"{self.n_nonpsd} parameter(s) had a non-positive sandwich variance; "
                "their se/z/p/CI are nan."
            )
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Coefficient table as markdown (via ``reports._df_to_markdown``)."""
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Coefficient table as a LaTeX ``tabular``."""
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Coefficient table as a Typst ``#table``."""
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, axes=None, figsize: tuple[float, float] = (10.0, 4.0)):
        """Two panels: the concentrated profile and the effects decomposition.

        Left: the concentrated log-likelihood against the spatial parameter,
        with the optimum, a grey line at zero and the 95% LR interval shaded
        (the dashed line is ``loglik - chi2(0.95, 1)/2``). Right: grouped bars
        of direct / indirect / total per regressor with the simulated
        confidence interval as error bars; for ``model='sem'`` (no multiplier)
        it becomes a coefficient plot of ``beta``.

        Returns the :class:`matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt

        fig = None
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax0, ax1 = axes[0], axes[1]
        prof = self.profile
        ax0.plot(prof["rho"], prof["loglik"], color="steelblue", linewidth=1.4)
        sp_name = self.spatial_param_name
        sp_val = self.lam if self.model == "sem" else self.rho
        ax0.axvline(0.0, color="grey", linewidth=0.8)
        ax0.plot([sp_val], [self.loglik], marker="o", color="firebrick",
                 label=f"{sp_name} = {sp_val:+.3f}")
        crit = self.loglik - 0.5 * stats.chi2.ppf(0.95, 1)
        ax0.axhline(crit, color="firebrick", linestyle="--", linewidth=0.9)
        inside = prof["loglik"].to_numpy(float) >= crit
        if inside.any():
            g = prof["rho"].to_numpy(float)[inside]
            ax0.axvspan(g.min(), g.max(), color="firebrick", alpha=0.10)
        ax0.set_xlabel(sp_name)
        ax0.set_ylabel("concentrated log-likelihood")
        ax0.set_title("profile likelihood")
        ax0.legend(frameon=False)

        if self.impacts is None:
            names = list(self.beta.index)
            pos = np.arange(len(names))
            vals = self.beta.to_numpy(float)
            errs = self.se.reindex(names).to_numpy(float) * stats.norm.ppf(1 - self.alpha / 2)
            ax1.bar(pos, vals, yerr=errs, color="steelblue", capsize=3, width=0.6)
            ax1.set_xticks(pos)
            ax1.set_xticklabels(names, rotation=45, ha="right")
            ax1.set_title("coefficients (no spatial multiplier)")
        else:
            imp = self.impacts
            names = list(imp.index)
            pos = np.arange(len(names))
            width = 0.26
            for off, key, colour in ((-width, "direct", "steelblue"),
                                     (0.0, "indirect", "darkorange"),
                                     (width, "total", "seagreen")):
                v = imp[key].to_numpy(float)
                e = imp[f"{key}_se"].to_numpy(float) * stats.norm.ppf(1 - self.alpha / 2)
                ax1.bar(pos + off, v, width=width, yerr=e, capsize=2,
                        color=colour, label=key)
            ax1.set_xticks(pos)
            ax1.set_xticklabels(names, rotation=45, ha="right")
            ax1.legend(frameon=False)
            ax1.set_title(f"LeSage-Pace effects ({self.n_draws} draws)")
        ax1.axhline(0.0, color="grey", linewidth=0.8)
        if fig is not None:
            fig.tight_layout()
            return fig
        return ax0.get_figure()


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------
def spatial_panel(
    df: pd.DataFrame,
    y: str,
    x: Sequence[str] | str,
    weights: SpatialWeights,
    *,
    model: str = "sar",
    effects: str = "two-way",
    durbin: bool | Sequence[str] = False,
    method: str = "transformation",
    bias_correction: str = "auto",
    vcov: str = "oim",
    coords: Any | None = None,
    cutoff_km: float | None = None,
    time_lags: int | None = None,
    kernel: str | None = None,
    metric: str | None = None,
    logdet: str = "auto",
    rho_bounds: tuple[float, float] | None = None,
    impacts: bool | None = None,
    n_draws: int | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
    entity_level: str = "code",
    time_level: str = "date",
    unit_col: str | None = None,
    time_col: str | None = None,
    dense_max: int = 5000,
    tol: float = 1e-9,
) -> SpatialPanelResult:
    """Fit a fixed-effects spatial panel (SAR / SDM / SEM) by quasi-ML.

    The individual and time effects are removed exactly --- by Lee & Yu
    (2010)'s orthonormal transformation (``method='transformation'``, the
    default, whose score is centred by construction) or by concentrating them
    out and applying the analytic Lee-Yu bias correction
    (``method='direct'``). See the module docstring for the two concentrated
    objectives written side by side, and for what the standard errors are and
    are not valid for.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel with ``y`` and every ``x`` as columns, indexed by
        ``(entity, time)`` in either order, or long-form with ``unit_col`` /
        ``time_col``. Normalised by
        :func:`puremacro.lp._panel_helpers.as_panel_index`.
    y : str
        Outcome column.
    x : sequence of str or str
        Regressor columns. A single string is accepted.
    weights : SpatialWeights
        Row order is irrelevant: the panel is reindexed onto ``weights.ids``.
    model : {'sar', 'sem', 'sdm'}, default 'sar'
        ``'sdm'`` is ``'sar'`` with ``durbin=True``.
    effects : {'none', 'individual', 'time', 'two-way'}, default 'two-way'
        Which fixed effects to remove. ``'none'`` prepends a constant.
    durbin : bool or sequence of str, default False
        Append the spatial lags ``W x`` as extra regressors, named
        ``'W.<col>'``. A sequence lags only the named subset. Raises for
        ``model='sem'`` (an SDEM is out of scope --- see the module docstring).
    method : {'transformation', 'direct'}, default 'transformation'
        How the effects are removed; see the module docstring's table.
    bias_correction : {'auto', 'lee-yu', 'none'}, default 'auto'
        ``'auto'`` is ``'lee-yu'`` for ``method='direct'`` with at least one
        set of effects, and ``'none'`` otherwise. ``'lee-yu'`` with
        ``method='transformation'`` raises: it would double-correct an
        already-centred estimator.
    vcov : {'oim', 'dk', 'conley'}, default 'oim'
        ``'dk'`` and ``'conley'`` replace the ``beta`` block only. The aliases
        the projection estimators accept for the same two are accepted here as
        well (``'driscoll-kraay'`` / ``'driscoll_kraay'`` / ``'driscollkraay'``
        for ``'dk'``, ``'spatial'`` / ``'spatial-hac'`` for ``'conley'``);
        only the keyword differs (``vcov=`` here, ``cov_type=`` there). There
        is no ``'qml'`` or ``'cluster'``; see the module docstring for why they
        are wrong here rather than merely absent.
    coords, cutoff_km : optional
        Required by ``vcov='conley'``; rejected otherwise. ``coords`` is one
        ``[lat, lon]`` pair per entity (DataFrame indexed by entity id, or a
        mapping).
    time_lags : int, optional
        Bartlett bandwidth in time for ``'dk'`` / ``'conley'``; defaults to
        ``floor(4 (T/100)^(2/9))`` with a floor of 1. Rejected for ``'oim'``.
    kernel : {'bartlett', 'uniform'}, optional
        Spatial kernel for ``vcov='conley'`` (default ``'bartlett'``).
    metric : {'haversine', 'euclidean'}, optional
        Distance metric for ``vcov='conley'`` (default ``'haversine'``).
    logdet : {'auto', 'eigen'}, default 'auto'
        Only the exact dense spectrum is available; ``'lu'`` raises, because
        the information matrix, the bias vector and the impacts all need
        traces an LU factorisation does not provide.
    rho_bounds : (float, float), optional
        Search interval for the spatial parameter. Defaults to the stability
        interval implied by the spectrum; a user interval outside it raises.
    impacts : bool, optional
        Compute the LeSage-Pace direct / indirect / total decomposition.
        ``None`` (the default) means ``True`` for ``model='sar'``/``'sdm'``
        and ``False`` for ``model='sem'``, which has no spatial multiplier.
        An explicit ``impacts=True`` with ``model='sem'`` raises rather than
        silently returning ``impacts=None``.
    n_draws, seed : int, optional
        Draws (and their seed) for the simulated impact standard errors;
        ``None`` (the default) means ``1000`` and ``0``. Both configure the
        impact simulation only, so passing either while no simulation will run
        (``impacts=False``, or ``model='sem'``) raises rather than storing an
        argument that did nothing. Deterministic: the same seed gives
        bit-identical impacts, and there is no unseeded mode.
    alpha : float, default 0.05
        Two-sided level for every confidence interval.
    entity_level, time_level, unit_col, time_col : str
        Panel index plumbing, passed straight to ``as_panel_index``.
    dense_max : int, default 5000
        Hard ceiling on the number of units (the spectrum and ``G`` are dense).
    tol : float, default 1e-9
        Tolerance for the symmetry, row-sum and identification checks.

    Returns
    -------
    SpatialPanelResult

    Raises
    ------
    ValueError
        An unrecognised ``model`` / ``effects`` / ``method`` /
        ``bias_correction`` / ``vcov`` / ``logdet``; ``T < 2`` with individual
        or two-way effects; ``n < 2`` with time or two-way effects; ``n < 3``;
        ``N_eff <= k + 1``; an unbalanced panel or duplicate ``(entity, time)``
        rows (the message names ``balanced_subpanel``); row sums that are not
        all one when a time-effects component meets
        ``method='transformation'`` (the message names ``standardize()``); a
        regressor absorbed by the effects, named explicitly; a user-supplied
        constant with ``effects != 'none'``; inverted ``rho_bounds`` or bounds
        reaching outside the *invertibility* interval (reaching outside the
        narrower *stability* interval only warns); ``vcov='conley'`` without ``coords``
        or ``cutoff_km``; ``vcov='dk'`` with ``T < 3``; a keyword that is
        meaningless for the chosen ``vcov``; ``durbin`` with ``model='sem'``;
        ``impacts=True`` with ``model='sem'``; ``n_draws=``/``seed=`` when no
        impact simulation will run; ``bias_correction='lee-yu'`` with
        ``effects='none'`` or with ``method='transformation'``;
        non-finite data; ``n > dense_max``; a spatial parameter that is not
        identified (the demeaned spatial lag is explained exactly by the
        demeaned regressors).
    KeyError
        ``y``, an ``x`` column or a ``durbin`` column is absent from ``df``;
        panel entities absent from ``weights.ids`` or vice versa (both
        directions are listed, first three shown).
    TypeError
        ``weights`` is not a :class:`SpatialWeights`.
    numpy.linalg.LinAlgError
        From ``inv_xtx``, naming ``'spatial_panel'`` and the collinear columns.

    Notes
    -----
    Complexity: ``O(n^3)`` once for the spectrum and for ``G``, ``O(n^2)``
    memory, ``O(nTk^2)`` for the two concentrating OLS passes and ``O(n)`` per
    likelihood evaluation. A :class:`RuntimeWarning` fires above 2000 units.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.panel import spatial_panel
    >>> side, T = 4, 8
    >>> nb = {}
    >>> for r in range(side):
    ...     for c in range(side):
    ...         nb[f"u{r * side + c:02d}"] = [
    ...             f"u{rr * side + cc:02d}"
    ...             for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    ...             if 0 <= rr < side and 0 <= cc < side]
    >>> W = contiguity_weights(nb)
    >>> rng = np.random.default_rng(0)
    >>> n = W.n
    >>> S_inv = np.linalg.inv(np.eye(n) - 0.4 * W.to_dense())
    >>> xs = rng.standard_normal((n, T))
    >>> mu = rng.standard_normal((n, 1))
    >>> ys = S_inv @ (1.5 * xs + mu + 0.3 * rng.standard_normal((n, T)))
    >>> idx = pd.MultiIndex.from_product([list(W.ids), range(T)],
    ...                                  names=["code", "date"])
    >>> panel = pd.DataFrame({"y": ys.ravel(), "x": xs.ravel()}, index=idx)
    >>> res = spatial_panel(panel, "y", "x", W, effects="individual",
    ...                     impacts=False)
    >>> res.model, res.bias_correction
    ('sar', 'none')
    >>> bool(0.2 < res.rho < 0.6), bool(1.2 < res.beta["x"] < 1.8)
    (True, True)
    """
    func = "spatial_panel"
    # ---------------- enum / type validation ------------------------------
    model = _check_enum(model, _MODELS, "model")
    effects = _check_enum(effects, _EFFECTS, "effects")
    method = _check_enum(method, _METHODS, "method")
    bias_correction = _check_enum(bias_correction, _BIAS, "bias_correction")
    vcov = _check_enum(_normalise_vcov(vcov), _VCOV, "vcov")
    if isinstance(logdet, str) and logdet.lower() == "lu":
        raise ValueError(
            f"{func}: logdet='lu' is not available for the panel. The information "
            "matrix, the Lee-Yu bias vector and the LeSage-Pace impacts all need "
            "tr(G), tr(G^2), tr(G'G) and diag(G), which an LU factorisation does "
            "not provide, and a stochastic trace probe would put unseeded "
            "randomness into a point estimate. Use logdet='eigen' (the default) "
            "and keep n <= dense_max."
        )
    logdet = _check_enum(logdet, _LOGDET, "logdet")
    logdet_method = "eigen"
    if not isinstance(weights, SpatialWeights):
        raise TypeError(
            f"{func}: weights must be a puremacro.spatial.SpatialWeights, got "
            f"{type(weights).__name__}"
        )
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{func}: df must be a pandas DataFrame, got {type(df).__name__}")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"{func}: alpha must lie in (0, 1), got {alpha!r}")

    # keywords that are meaningless for the chosen vcov (no silent ignoring)
    if vcov != "conley":
        for name, val in (("coords", coords), ("cutoff_km", cutoff_km),
                          ("kernel", kernel), ("metric", metric)):
            if val is not None:
                raise ValueError(
                    f"{func}: {name}= is only used by vcov='conley', but vcov={vcov!r}"
                )
    if vcov not in ("dk", "conley") and time_lags is not None:
        raise ValueError(
            f"{func}: time_lags= is only used by vcov='dk' or 'conley', but "
            f"vcov={vcov!r}"
        )
    if vcov == "conley":
        if coords is None:
            raise ValueError(f"{func}: vcov='conley' needs coords= (one [lat, lon] per entity)")
        if cutoff_km is None:
            raise ValueError(f"{func}: vcov='conley' needs cutoff_km=")
    kernel = "bartlett" if kernel is None else str(kernel)
    metric = "haversine" if metric is None else str(metric)

    # model / durbin / bias-correction interaction
    if model == "sem" and durbin is not False:
        raise ValueError(
            f"{func}: durbin= with model='sem' would be an SDEM, which is out of "
            "scope in this release (the module docstring lists it under "
            "'Deliberately omitted'). Use model='sdm' for a Durbin lag model, or "
            "drop durbin=."
        )
    if model == "sdm" and durbin is False:
        durbin = True
    if method == "transformation" and bias_correction == "lee-yu":
        raise ValueError(
            f"{func}: bias_correction='lee-yu' with method='transformation' would "
            "double-correct. The transformation score is centred by construction, "
            "so the correction is provably zero. Use method='direct' to get the "
            "analytic correction, or bias_correction='auto'/'none'."
        )

    ent = effects in ("individual", "two-way")
    tim = effects in ("time", "two-way")
    if bias_correction == "lee-yu" and not (ent or tim):
        raise ValueError(
            f"{func}: bias_correction='lee-yu' is meaningless with effects='none'. "
            "The Lee-Yu bias is the incidental-parameter bias of the fixed effects; "
            "with no effects removed the score is already centred and the correction "
            "is identically zero. Pass bias_correction='none' (or 'auto'), or remove "
            "some effects."
        )
    if bias_correction == "auto":
        bias_correction = "lee-yu" if (method == "direct" and (ent or tim)) else "none"

    # the impact simulation: what runs, and which of its knobs are meaningful
    given = [nm for nm, val in (("n_draws", n_draws), ("seed", seed)) if val is not None]
    if model == "sem":
        if impacts is True:
            raise ValueError(
                f"{func}: impacts=True is meaningless for model='sem'. The SEM puts "
                "the spatial multiplier on the errors, not on y, so there is no "
                "direct/indirect decomposition: the marginal effect of x IS beta. "
                "Drop impacts= (or pass impacts=False) and read the coefficient table."
            )
        do_impacts = False
    else:
        do_impacts = True if impacts is None else bool(impacts)
    if given and not do_impacts:
        why = ("model='sem' runs no impact simulation"
               if model == "sem" else "impacts=False runs no impact simulation")
        raise ValueError(
            f"{func}: {'/'.join(given)}= configure(s) the LeSage-Pace impact "
            f"simulation, but {why}. Drop {'/'.join(given)}=, or ask for the impacts."
        )
    n_draws = 1000 if n_draws is None else int(n_draws)
    seed = 0 if seed is None else int(seed)
    if do_impacts and n_draws < 2:
        raise ValueError(f"{func}: n_draws must be at least 2, got {n_draws!r}")

    # ---------------- panel contract --------------------------------------
    frame, entity_level, time_level = as_panel_index(
        df, entity_level=entity_level, time_level=time_level,
        unit_col=unit_col, time_col=time_col, func=func,
    )
    x_cols = [x] if isinstance(x, str) else [str(c) for c in x]
    if len(set(x_cols)) != len(x_cols):
        raise ValueError(f"{func}: x contains duplicate column names: {x_cols}")
    if y in x_cols:
        raise ValueError(f"{func}: y={y!r} also appears in x")
    missing = [c for c in [y] + x_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"{func}: column(s) {missing} are not in the frame; "
                       f"columns are {list(frame.columns)[:10]}")
    if durbin is True:
        durbin_cols = list(x_cols)
    elif durbin is False:
        durbin_cols = []
    else:
        durbin_cols = [str(c) for c in durbin]
        bad = [c for c in durbin_cols if c not in x_cols]
        if bad:
            raise KeyError(f"{func}: durbin column(s) {bad} are not among x={x_cols}")

    sub = frame[[y] + x_cols]
    if sub.index.duplicated().any():
        dup = sub.index[sub.index.duplicated()][:3].tolist()
        raise ValueError(
            f"{func}: duplicate (entity, time) rows, e.g. {dup}. The transformation "
            "approach is defined only for a rectangular panel; de-duplicate first "
            "(see puremacro.inference.balanced_panel.balanced_subpanel)."
        )
    panel_ents = set(sub.index.get_level_values(entity_level))
    w_ids = list(weights.ids)
    w_set = set(w_ids)
    only_panel = sorted(panel_ents - w_set, key=str)
    only_w = sorted(w_set - panel_ents, key=str)
    if only_panel or only_w:
        raise KeyError(
            f"{func}: the panel entities and weights.ids differ. In the panel but "
            f"not in the weights: {only_panel[:3]} ({len(only_panel)} total); in "
            f"the weights but not in the panel: {only_w[:3]} ({len(only_w)} total)."
        )
    periods = sorted(set(sub.index.get_level_values(time_level)))
    n = weights.n
    T = len(periods)
    if n > int(dense_max):
        raise ValueError(
            f"{func}: {n} units exceeds dense_max={dense_max}. The spectrum of W and "
            "the dense G are O(n^2) memory / O(n^3) time; raise dense_max "
            "deliberately if you can afford it. This is the same budget documented "
            "in puremacro.spatial.hac."
        )
    if n > 2000:
        warnings.warn(
            f"{func}: {n} units means a dense {n}x{n} eigendecomposition and a dense "
            "G; expect O(n^3) time and O(n^2) memory.",
            RuntimeWarning, stacklevel=2,
        )
    if tim and n < 2:
        raise ValueError(
            f"{func}: effects={effects!r} removes the time effects with a "
            f"cross-sectional transformation that leaves n-1 = {n - 1} units; got n={n}."
        )
    if n < 3:
        raise ValueError(f"{func}: needs at least 3 units to identify a spatial "
                         f"parameter, got n={n}")
    if ent and T < 2:
        raise ValueError(
            f"{func}: effects={effects!r} removes the individual effects with a "
            f"within transformation that leaves T-1 = {T - 1} periods; got T={T}. "
            "Use effects='time' or effects='none', or a cross-sectional estimator."
        )

    # (n, T) blocks, aligned to weights.ids -> panel row order is never assumed
    def _block(col: str) -> np.ndarray:
        wide = sub[col].unstack(time_level).reindex(index=w_ids, columns=periods)
        return wide.to_numpy(dtype=float)

    full_index = pd.MultiIndex.from_product([w_ids, periods],
                                            names=[entity_level, time_level])
    if len(sub) != n * T:
        miss = full_index.difference(sub.index)
        by_ent = pd.Series(list(miss.get_level_values(entity_level))).value_counts()
        by_per = pd.Series(list(miss.get_level_values(time_level))).value_counts()
        raise ValueError(
            f"{func}: the panel is not balanced -- {n} entities x {T} periods = "
            f"{n * T} cells but the frame has {len(sub)} rows, {len(miss)} of them "
            f"missing. Entities short of periods (count): "
            f"{dict(list(by_ent.items())[:3])}; periods short of entities (count): "
            f"{dict(list(by_per.items())[:3])}. Balanced panels only in this release: "
            "the transformation approach needs a common T for F_{T,T-1}. Trim with "
            "puremacro.inference.balanced_panel.balanced_subpanel first."
        )

    blocks = {c: _block(c) for c in [y] + x_cols}
    for c, arr in blocks.items():
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(
                f"{func}: column {c!r} has {n_bad} non-finite value(s) among the "
                f"{n * T} (entity, period) cells; drop or impute them first "
                "(see puremacro.inference.balanced_panel.balanced_subpanel)."
            )
    if weights.W.nnz and not np.all(np.isfinite(weights.W.data)):
        raise ValueError(f"{func}: W contains non-finite weights")

    # ---------------- W's row sums and the space transformation -----------
    row_sums = weights.row_sums()
    row_std = bool(np.max(np.abs(row_sums - 1.0)) < tol) if n else False
    if tim and method == "transformation" and not row_std:
        worst = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"{func}: effects={effects!r} with method='transformation' needs every "
            "row of W to sum to exactly one, so that l_n is an eigenvector of W and "
            "|I - rho W| factors as (1 - rho)|I_{n-1} - rho F'WF|. Unit "
            f"{weights.ids[worst]!r} has row sum {row_sums[worst]:.6g}. Fix it with "
            "weights.standardize() (an island has row sum 0 and cannot be fixed that "
            "way), or use effects='individual', or method='direct' (whose Jacobian "
            "is the plain T ln|S| and needs no assumption on the row sums)."
        )
    if bias_correction == "lee-yu" and tim and not row_std:
        raise ValueError(
            f"{func}: the Lee-Yu bias term for the time effects is -(T or T-1)/(1-rho), "
            "which assumes W l = l. Row-standardise with weights.standardize(), or "
            "pass bias_correction='none'."
        )

    # ---------------- spatial lags, BEFORE any demeaning ------------------
    W = weights.W
    WY = W @ blocks[y]
    lag_of = {c: (W @ blocks[c]) for c in x_cols} if (model == "sem" or durbin_cols) else {}

    reg_names: list[str] = []
    reg_raw: list[np.ndarray] = []
    if effects == "none":
        reg_names.append("const")
        reg_raw.append(np.ones((n, T)))
    for c in x_cols:
        reg_names.append(c)
        reg_raw.append(blocks[c])
    for c in durbin_cols:
        reg_names.append(f"W.{c}")
        reg_raw.append(lag_of[c])
    if len(set(reg_names)) != len(reg_names):
        dup = [c for c in reg_names if reg_names.count(c) > 1]
        raise ValueError(f"{func}: generated regressor name(s) {sorted(set(dup))} collide "
                         "with a column you supplied; rename the column.")
    k = len(reg_names)

    # for SEM the spatial filter needs W X for EVERY regressor
    if model == "sem":
        reg_lag_raw = []
        for name, arr in zip(reg_names, reg_raw):
            if name == "const":
                reg_lag_raw.append(np.tile(row_sums[:, None], (1, T)))
            else:
                base = name[2:] if name.startswith("W.") else name
                reg_lag_raw.append(W @ arr if name.startswith("W.") else lag_of[base])
        reg_lag_raw = [np.asarray(a, dtype=float) for a in reg_lag_raw]
    else:
        reg_lag_raw = None

    # ---------------- demeaning and the effective counts ------------------
    def _demean(A: np.ndarray) -> np.ndarray:
        B = np.asarray(A, dtype=float).copy()
        if ent:
            B -= B.mean(axis=1, keepdims=True)
        if tim:
            B -= B.mean(axis=0, keepdims=True)
        return B

    _absorbed = {
        "individual": "time-invariant",
        "time": "cross-sectionally constant",
        "two-way": "annihilated by the two-way within transformation",
    }
    for name, arr in zip(reg_names, reg_raw):
        if name == "const":
            continue
        scale = max(1.0, float(np.max(np.abs(arr))))
        if float(np.max(np.abs(arr - arr.mean()))) < 1e-10 * scale:
            if effects == "none":
                raise ValueError(
                    f"{func}: regressor {name!r} is constant; effects='none' already "
                    "prepends a constant column named 'const'."
                )
            raise ValueError(
                f"{func}: regressor {name!r} is constant, so effects={effects!r} "
                "absorbs it entirely; drop it. Only effects='none' takes a constant, "
                "and it supplies its own."
            )
        if effects != "none" and float(np.max(np.abs(_demean(arr)))) < 1e-10 * scale:
            raise ValueError(
                f"{func}: regressor {name!r} is {_absorbed[effects]}, so "
                f"effects={effects!r} absorbs it; drop it or change effects."
            )

    y_dd = _demean(blocks[y]).ravel(order="F")
    wy_dd = _demean(WY).ravel(order="F")
    X_dd = np.column_stack([_demean(a).ravel(order="F") for a in reg_raw])
    WX_dd = (np.column_stack([_demean(a).ravel(order="F") for a in reg_lag_raw])
             if model == "sem" else None)

    n_star = n - 1 if tim else n
    T_star = T - 1 if ent else T
    N_star = n_star * T_star
    if N_star <= k + 1:
        raise ValueError(
            f"{func}: the effective sample N* = {n_star} x {T_star} = {N_star} is too "
            f"small for {k} regressor(s) plus the spatial parameter and sigma^2."
        )
    if method == "direct":
        jac_factor = float(T)
        n_div = float(n * T)
    else:
        jac_factor = float(T_star)
        n_div = float(N_star)
    if vcov == "dk" and T < 3:
        raise ValueError(f"{func}: vcov='dk' needs at least 3 periods, got T={T}")

    entity_keys = np.tile(np.asarray(w_ids, dtype=object), T)
    time_keys = np.repeat(np.asarray(periods, dtype=object), n)

    # ---------------- spectrum, Jacobian and the parameter space ----------
    Wd = weights.to_dense()
    symmetric = bool(np.max(np.abs(Wd - Wd.T)) < tol) if n else True
    evals = sla.eigvalsh(Wd) if symmetric else sla.eigvals(Wd)
    evals = np.asarray(evals)

    drop_unit = tim and method == "transformation"
    if drop_unit:
        j = int(np.argmin(np.abs(evals - 1.0)))
        if abs(evals[j] - 1.0) > 1e-7:
            raise ValueError(
                f"{func}: W should have an eigenvalue at 1 when its rows sum to one; "
                f"the closest is {evals[j]!r}. Check the weights."
            )
        ev_used = np.delete(evals, j)
    else:
        ev_used = evals

    def _jacobian(r: float) -> float:
        # dropping the computed unit eigenvalue beats subtracting ln(1 - rho):
        # the eigenvalue is 1 +/- 1e-15 and ln(1 - rho*omega_hat) amplifies that
        # rounding error near the singularity (about 1e-6 at rho = 1 - 1e-9).
        return float(np.sum(np.log(np.abs(1.0 - r * ev_used))))

    # Two intervals, both reported (see SpatialPanelResult.invertible_bounds).
    # (a) invertibility: I - rho W is singular only at rho = 1/lam for a REAL
    #     eigenvalue lam --- a complex lam never annihilates a real rho --- so
    #     the exact non-singularity interval comes from the real spectrum alone.
    # (b) stability: (a) further intersected with (-1/max|lam_i|, 1/max|lam_i|)
    #     over the FULL complex spectrum, on which the multiplier series
    #     converges. This is the default search interval, and it can be strictly
    #     tighter than (a): a user interval between the two only warns.
    real_mask = np.abs(np.imag(evals)) < 1e-10 if np.iscomplexobj(evals) else np.ones(n, bool)
    real_parts = np.real(evals[real_mask]) if real_mask.any() else np.array([])
    lo = -np.inf
    hi = np.inf
    neg = real_parts[real_parts < -1e-12]
    pos = real_parts[real_parts > 1e-12]
    if neg.size:
        lo = 1.0 / float(neg.min())
    if pos.size:
        hi = 1.0 / float(pos.max())
    if row_std:
        lo = max(lo, -1.0)
        hi = min(hi, 1.0)
    lo_i = -0.999 if not np.isfinite(lo) else lo
    hi_i = 0.999 if not np.isfinite(hi) else hi
    mod_max = float(np.max(np.abs(evals))) if n else 1.0
    if mod_max > 1e-12:
        lo = max(lo, -1.0 / mod_max)
        hi = min(hi, 1.0 / mod_max)
    if not np.isfinite(lo):
        lo = -0.999
    if not np.isfinite(hi):
        hi = 0.999
    stab = (lo + 1e-6, hi - 1e-6)
    invertible = (lo_i + 1e-6, hi_i - 1e-6)
    if rho_bounds is not None:
        b = (float(rho_bounds[0]), float(rho_bounds[1]))
        if b[0] >= b[1]:
            raise ValueError(f"{func}: rho_bounds={rho_bounds!r} is inverted or empty")
        if b[0] < invertible[0] - 1e-12 or b[1] > invertible[1] + 1e-12:
            raise ValueError(
                f"{func}: rho_bounds={b} reaches outside the invertibility interval "
                f"{invertible} implied by the real spectrum of W; S(rho) would be "
                "singular there."
            )
        if b[0] < stab[0] - 1e-12 or b[1] > stab[1] + 1e-12:
            warnings.warn(
                f"{func}: rho_bounds={b} reaches outside the stability interval "
                f"{stab} (though S(rho) stays invertible inside {invertible}). The "
                "spatial multiplier series does not converge out there, so the "
                "impacts and the usual asymptotics are not to be trusted at such a "
                "rho.",
                RuntimeWarning, stacklevel=2,
            )
        search = b
    else:
        search = stab

    # ---------------- concentrated likelihood -----------------------------
    n_evals = 0
    if model in ("sar", "sdm"):
        XtX_inv = inv_xtx(X_dd, name=func)
        b_O = XtX_inv @ (X_dd.T @ y_dd)
        b_L = XtX_inv @ (X_dd.T @ wy_dd)
        e_O = y_dd - X_dd @ b_O
        e_L = wy_dd - X_dd @ b_L
        a_c = float(e_O @ e_O)
        b_c = float(e_O @ e_L)
        c_c = float(e_L @ e_L)
        if c_c <= tol * max(1.0, float(wy_dd @ wy_dd)):
            raise ValueError(
                f"{func}: the demeaned spatial lag Wy is explained exactly by the "
                "demeaned regressors, so SSR(rho) is constant and rho is not "
                "identified. This happens with one unit per period, or when a Durbin "
                "term reproduces Wy."
            )

        def _ssr(r: float) -> float:
            return a_c - 2.0 * r * b_c + r * r * c_c

        def _loglik(r: float) -> float:
            nonlocal n_evals
            n_evals += 1
            s = _ssr(float(r))
            if s <= 0:
                return -np.inf
            return (-0.5 * n_div * (np.log(2.0 * np.pi) + 1.0)
                    - 0.5 * n_div * np.log(s / n_div) + jac_factor * _jacobian(float(r)))
    else:
        # the named collinearity error (G7) must reach the user BEFORE the
        # profile loop's bare solve does; at lam = 0 the filtered design IS X_dd.
        inv_xtx(X_dd, name=func)

        def _sem_fit(lam: float):
            ys = y_dd - lam * wy_dd
            Xs = X_dd - lam * WX_dd
            XtX = Xs.T @ Xs
            bb = np.linalg.solve(XtX, Xs.T @ ys)
            e = ys - Xs @ bb
            return float(e @ e), bb, e, Xs, ys

        a_c = float(_sem_fit(0.0)[0])

        def _ssr(r: float) -> float:
            return _sem_fit(float(r))[0]

        def _loglik(r: float) -> float:
            nonlocal n_evals
            n_evals += 1
            s = _ssr(float(r))
            if s <= 0:
                return -np.inf
            return (-0.5 * n_div * (np.log(2.0 * np.pi) + 1.0)
                    - 0.5 * n_div * np.log(s / n_div) + jac_factor * _jacobian(float(r)))

    grid = np.linspace(search[0], search[1], 51)
    grid_ll = np.array([_loglik(g) for g in grid])
    if not np.any(np.isfinite(grid_ll)):
        raise ValueError(f"{func}: the concentrated likelihood is not finite anywhere "
                         f"on the search interval {search}")
    if (search[1] - search[0]) > 1e-3 and float(np.nanmax(grid_ll) - np.nanmin(grid_ll)) < 1e-12:
        # (skipped for a deliberately pinned, near-degenerate rho_bounds interval)
        raise ValueError(
            f"{func}: the concentrated likelihood is flat in the spatial parameter, "
            "which is therefore not identified."
        )
    j_star = int(np.nanargmax(grid_ll))
    step = grid[1] - grid[0]
    br_lo = max(search[0], grid[j_star] - 2.0 * step)
    br_hi = min(search[1], grid[j_star] + 2.0 * step)
    opt = optimize.minimize_scalar(
        lambda r: -_loglik(r), bounds=(br_lo, br_hi), method="bounded",
        options={"xatol": 1e-8},
    )
    rho_hat = float(opt.x)
    grid_fallback = bool(-float(opt.fun) < grid_ll[j_star])
    if grid_fallback:                              # never lose to the grid
        rho_hat = float(grid[j_star])
    loglik = float(_loglik(rho_hat))
    # a reported rho that is a grid point rather than an optimiser solution is
    # not a converged optimum, whatever opt.success says.
    converged = bool(getattr(opt, "success", True)) and not grid_fallback
    at_bound = min(abs(rho_hat - search[0]), abs(rho_hat - search[1])) < 1e-4
    if at_bound:
        converged = False
        warnings.warn(
            f"{func}: the spatial parameter converged to {rho_hat:.6f}, within 1e-4 of "
            f"the search bound {search}; the usual asymptotics do not hold at a "
            "boundary optimum and the standard errors are unreliable.",
            RuntimeWarning, stacklevel=2,
        )

    if model in ("sar", "sdm"):
        beta_hat = b_O - rho_hat * b_L
        ssr = _ssr(rho_hat)
        resid_vec = y_dd - rho_hat * wy_dd - X_dd @ beta_hat
        Xeff_dd = X_dd
    else:
        ssr, beta_hat, resid_vec, Xeff_dd, _ys = _sem_fit(rho_hat)
        inv_xtx(Xeff_dd, name=func)          # rank diagnostic on the filtered design
    sigma2_hat = float(ssr) / n_div
    loglik_null = float(-0.5 * n_div * (np.log(2.0 * np.pi) + 1.0 + np.log(a_c / n_div)))
    lr_stat = float(2.0 * (loglik - loglik_null))
    lr_pvalue = float(stats.chi2.sf(max(lr_stat, 0.0), 1))
    tss = float(y_dd @ y_dd)
    r2_within = float(1.0 - ssr / tss) if tss > 0 else float("nan")

    # ---------------- pieces of the information matrix --------------------
    eye_n = np.eye(n)

    def _pieces(r: float) -> dict:
        S_dense = eye_n - r * Wd
        S_inv = sla.solve(S_dense, eye_n)
        Gd = Wd @ S_inv
        A = _double_centre(Gd) if tim else Gd
        gam = np.diag(A) if (tim and method == "transformation") else np.diag(Gd)
        return {
            "S_inv": S_inv, "G": Gd, "A": A,
            "tr_A": float(np.trace(A)),
            "tr_A2": float(np.sum(A * A.T)),
            "tr_AtA": float(np.sum(A * A)),
            "tr_G": float(np.trace(Gd)),
            "gamma": np.asarray(gam, dtype=float).copy(),
        }

    def _systematic_lag(pc: dict, bet: np.ndarray) -> np.ndarray:
        M = np.zeros((n, T))
        for coef, arr in zip(bet, reg_raw):
            M = M + coef * arr
        Z = pc["S_inv"] @ M
        return _demean(Wd @ Z).ravel(order="F")

    def _information(r: float, bet: np.ndarray, s2: float, pc: dict) -> np.ndarray:
        I = np.zeros((k + 2, k + 2))
        if model == "sem":
            Xf = X_dd - r * WX_dd
            I[:k, :k] = Xf.T @ Xf / s2
            I[k, k] = T_star * (pc["tr_A2"] + pc["tr_AtA"])
        else:
            I[:k, :k] = X_dd.T @ X_dd / s2
            m_dd = _systematic_lag(pc, bet)
            I[:k, k] = X_dd.T @ m_dd / s2
            I[k, :k] = I[:k, k]
            I[k, k] = T_star * (pc["tr_A2"] + pc["tr_AtA"]) + float(m_dd @ m_dd) / s2
        I[k, k + 1] = I[k + 1, k] = T_star * pc["tr_A"] / s2
        I[k + 1, k + 1] = N_star / (2.0 * s2 * s2)
        return I

    pieces_hat = _pieces(rho_hat)
    I_hat = _information(rho_hat, beta_hat, sigma2_hat, pieces_hat)
    theta_hat = np.concatenate([beta_hat, [rho_hat, sigma2_hat]])

    # ---------------- Lee-Yu bias correction ------------------------------
    bias_vec = None
    if bias_correction == "lee-yu":
        a_vec = np.zeros(k + 2)
        rho_term = 0.0
        if ent:
            rho_term -= pieces_hat["tr_G"]
        if tim:
            rho_term -= ((T - 1.0) if ent else float(T)) / (1.0 - rho_hat)
        a_vec[k] = rho_term
        if ent and tim:
            cnt = float(n + T - 1)
        elif ent:
            cnt = float(n)
        else:
            cnt = float(T)
        a_vec[k + 1] = -cnt / (2.0 * sigma2_hat)
        bias_vec = np.linalg.solve(I_hat, a_vec)
        if np.linalg.norm(bias_vec) > 0.5 * np.linalg.norm(theta_hat):
            warnings.warn(
                f"{func}: the Lee-Yu bias correction is larger than half the parameter "
                "vector; the information matrix is near-singular or the spatial "
                "parameter is weakly identified.",
                RuntimeWarning, stacklevel=2,
            )
        theta_final = theta_hat - bias_vec
    else:
        theta_final = theta_hat

    beta_f = theta_final[:k]
    rho_f = float(theta_final[k])
    sigma2_f = float(theta_final[k + 1])
    if sigma2_f <= 0:
        raise ValueError(
            f"{func}: the Lee-Yu correction drove sigma^2 to {sigma2_f:.3e}; the "
            "sample is too small for the correction. Use method='transformation'."
        )
    pieces_f = pieces_hat if bias_vec is None else _pieces(rho_f)
    I_final = I_hat if bias_vec is None else _information(rho_f, beta_f, sigma2_f, pieces_f)
    V = np.linalg.solve(I_final, np.eye(k + 2))
    V = 0.5 * (V + V.T)

    # ---------------- robust covariance (beta block only) -----------------
    if vcov != "oim":
        if model == "sem":
            e_f = (y_dd - rho_f * wy_dd) - (X_dd - rho_f * WX_dd) @ beta_f
            Xscore = X_dd - rho_f * WX_dd
            lag_score = (wy_dd - WX_dd @ beta_f)
        else:
            e_f = y_dd - rho_f * wy_dd - X_dd @ beta_f
            Xscore = X_dd
            lag_score = wy_dd
        scores = np.empty((n * T, k + 2))
        scores[:, :k] = Xscore * (e_f[:, None] / sigma2_f)
        gamma_full = np.tile(pieces_f["gamma"], T)
        scores[:, k] = lag_score * e_f / sigma2_f - (jac_factor / T) * gamma_full
        scores[:, k + 1] = -(n_div / (n * T)) / (2.0 * sigma2_f) + e_f ** 2 / (2.0 * sigma2_f ** 2)
        L = _dk_bandwidth(T) if time_lags is None else int(time_lags)
        if L < 0:
            raise ValueError(f"{func}: time_lags must be non-negative, got {time_lags!r}")
        if vcov == "dk":
            from ..inference.dk import driscoll_kraay
            S_meat = driscoll_kraay(scores, time_keys, lags=L)
        else:
            S_meat = _hac_meat_from_scores(
                scores, entity_keys, time_keys, coords, float(cutoff_km), L,
                kernel=kernel, metric=metric,
            )
        V_sand = V @ S_meat @ V
        V = V.copy()
        V[:k, :k] = 0.5 * (V_sand[:k, :k] + V_sand[:k, :k].T)

    diag = np.diag(V).copy()
    n_nonpsd = int(np.sum(diag <= 0))
    if n_nonpsd:
        warnings.warn(
            f"{func}: {n_nonpsd} parameter(s) have a non-positive variance on the "
            "covariance diagonal; their se/z/p/CI are nan.",
            RuntimeWarning, stacklevel=2,
        )
    se_vec = np.where(diag > 0, np.sqrt(np.where(diag > 0, diag, 1.0)), np.nan)

    sp_name = "lambda" if model == "sem" else "rho"
    index = pd.Index(list(reg_names) + [sp_name, "sigma2"], name="term")
    params = pd.Series(theta_final, index=index)
    se = pd.Series(se_vec, index=index)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = params / se
    p_values = pd.Series(2.0 * stats.norm.sf(np.abs(z.to_numpy(float))), index=index)
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    conf_int = pd.DataFrame(
        {"lower": params - z_crit * se, "upper": params + z_crit * se}, index=index
    )
    beta_series = params.iloc[:k].copy()
    bias_series = pd.Series(bias_vec, index=index) if bias_vec is not None else None

    # ---------------- impacts ---------------------------------------------
    impacts_df = None
    n_rejected = 0
    if do_impacts:
        impacts_df, n_rejected = _lesage_pace_impacts(
            reg_names=reg_names, x_cols=x_cols, durbin_cols=durbin_cols,
            params=params, V=V, k=k, evals=evals, Wd=Wd, W=W, n=n,
            row_std=row_std, stab=stab, n_draws=int(n_draws), seed=int(seed),
            alpha=float(alpha), func=func, symmetric=symmetric,
        )

    # ---------------- recovered effects and residuals ---------------------
    if model == "sem":
        R = blocks[y].copy()
    else:
        R = blocks[y] - rho_f * WY
    for coef, arr in zip(beta_f, reg_raw):
        R = R - coef * arr
    # R is already net of every regressor, the 'const' column included, so its
    # mean is machine zero when effects='none': the level is the fitted
    # coefficient, not a leftover. Report that instead of an inert 1e-16.
    intercept = float(params["const"]) if effects == "none" else float(R.mean())
    entity_effects = None
    time_effects = None
    if ent:
        mu = R.mean(axis=1) - intercept
        entity_effects = pd.Series(mu - mu.mean(), index=pd.Index(w_ids, name=entity_level))
    if tim:
        xi = R.mean(axis=0) - intercept
        time_effects = pd.Series(xi - xi.mean(), index=pd.Index(periods, name=time_level))

    resid = pd.Series(
        resid_vec,
        index=pd.MultiIndex.from_arrays([entity_keys, time_keys],
                                        names=[entity_level, time_level]),
        name="resid",
    )
    profile = pd.DataFrame({"rho": grid, "loglik": grid_ll})

    return SpatialPanelResult(
        model=model, effects=effects, method=method,
        bias_correction=bias_correction, vcov_type=vcov,
        vcov_note=_VCOV_NOTES[vcov],
        params=params, beta=beta_series,
        rho=float("nan") if model == "sem" else rho_f,
        lam=rho_f if model == "sem" else float("nan"),
        sigma2=sigma2_f,
        sigma2_dfc=float(ssr) / max(N_star - k, 1),
        vcov=V, se=se, z=z, p_values=p_values, conf_int=conf_int,
        bias=bias_series, impacts=impacts_df,
        loglik=loglik, loglik_null=loglik_null, lr_stat=lr_stat, lr_pvalue=lr_pvalue,
        r2_within=r2_within, profile=profile, resid=resid,
        entity_effects=entity_effects, time_effects=time_effects, intercept=intercept,
        n_entities=int(n), n_periods=int(T), n_obs=int(n * T), n_eff=int(N_star),
        n_eff_entities=int(n_star), n_eff_periods=int(T_star),
        df_resid=int(N_star - k - 1), n_nonpsd=n_nonpsd,
        ids=tuple(w_ids), rho_bounds=(float(search[0]), float(search[1])),
        stability_bounds=(float(stab[0]), float(stab[1])),
        invertible_bounds=(float(invertible[0]), float(invertible[1])),
        logdet_method=logdet_method, weights_kind=weights.kind,
        converged=converged, n_evals=int(n_evals), n_draws=int(n_draws),
        n_rejected=int(n_rejected), seed=int(seed), alpha=float(alpha),
    )


# ---------------------------------------------------------------------------
# LeSage-Pace effects
# ---------------------------------------------------------------------------
def _partial_fractions(Wd: np.ndarray, n: int, symmetric: bool) -> np.ndarray | None:
    """Coefficients ``c_i`` with ``l'(I - rho W)^-1 l / n = sum_i c_i/(1 - rho w_i)``.

    From the eigendecomposition ``W = V diag(w) V^-1``: ``c = (V'l) * (V^-1 l)/n``.
    Returns ``None`` when ``W`` is not diagonalisable to working precision.
    """
    ell = np.ones(n)
    try:
        if symmetric:
            _w, V = sla.eigh(Wd)
            a = V.T @ ell
            return (a * a) / n
        _w, V = sla.eig(Wd)
        u = np.linalg.solve(V, ell.astype(complex))
        return (V.T @ ell.astype(complex)) * u / n
    except (np.linalg.LinAlgError, ValueError):
        return None


def _lesage_pace_impacts(
    *, reg_names, x_cols, durbin_cols, params, V, k, evals, Wd, W, n,
    row_std, stab, n_draws, seed, alpha, func, symmetric,
):
    """Direct / indirect / total effects with simulated standard errors.

    Closed forms, no truncated power series (a series in ``rho^q tr(W^q)``
    overflows for a non-standardised ``W`` and needs 263 terms at ``rho=0.9``):

    ``direct_r = [beta_r tr(S^-1) + theta_r tr(G)] / n``
    ``total_r  = [beta_r l'S^-1 l + theta_r l'S^-1 W l] / n``

    with ``tr(S^-1)/n = mean_i 1/(1 - rho w_i)`` and ``tr(G)/n =
    mean_i w_i/(1 - rho w_i)`` from the spectrum, and ``l'S^-1 l/n =
    1/(1 - rho)`` exactly when every row of ``W`` sums to one (then also
    ``l'S^-1 W l = l'S^-1 l``, so ``total_r = (beta_r + theta_r)/(1 - rho)``).
    """
    ev = np.asarray(evals)
    idx = {name: i for i, name in enumerate(reg_names)}
    variables = [c for c in x_cols]
    rho_idx = k

    cvec = None
    if not row_std:
        cvec = _partial_fractions(Wd, n, symmetric)
        if cvec is not None:                      # one exact cross-check
            rho0 = float(params.iloc[rho_idx])
            ell = np.ones(n)
            direct_solve = float(ell @ sla.solve(np.eye(n) - rho0 * Wd, ell)) / n
            approx = float(np.real(np.sum(cvec / (1.0 - rho0 * ev))))
            if not np.isfinite(approx) or abs(approx - direct_solve) > 1e-7 * max(1.0, abs(direct_solve)):
                cvec = None
        if cvec is None:
            warnings.warn(
                f"{func}: W is neither row-standardised nor diagonalisable to working "
                "precision, so the aggregate multiplier l'(I - rho W)^-1 l is solved "
                "densely once per simulation draw; this is O(n^3) per draw. "
                "weights.standardize() makes it a closed form.",
                RuntimeWarning, stacklevel=3,
            )

    def _kernels(rvals: np.ndarray):
        """(d0, d1, s0, s1) for an array of rho values."""
        Z = 1.0 - rvals[:, None] * ev[None, :]
        d0 = np.real(np.sum(1.0 / Z, axis=1)) / n
        d1 = np.real(np.sum(ev[None, :] / Z, axis=1)) / n
        if row_std:
            s0 = 1.0 / (1.0 - rvals)
            s1 = s0.copy()
        elif cvec is not None:
            s0 = np.real(np.sum(cvec[None, :] / Z, axis=1))
            s1 = np.real(np.sum((cvec * ev)[None, :] / Z, axis=1))
        else:
            ell = np.ones(n)
            s0 = np.empty(len(rvals))
            s1 = np.empty(len(rvals))
            for i, r in enumerate(rvals):
                v = sla.solve(np.eye(n) - r * Wd, ell)
                s0[i] = float(ell @ v) / n
                s1[i] = float(ell @ (Wd @ v)) / n
        return d0, d1, s0, s1

    def _effects(bet: np.ndarray, the: np.ndarray, rvals: np.ndarray):
        d0, d1, s0, s1 = _kernels(rvals)
        direct = bet * d0[:, None] + the * d1[:, None]
        total = bet * s0[:, None] + the * s1[:, None]
        return direct, total - direct, total

    b_pos = np.array([idx[c] for c in variables])
    t_pos = np.array([idx[f"W.{c}"] if f"W.{c}" in idx else -1 for c in variables])
    theta_hat = np.array([
        float(params.iloc[t_pos[i]]) if t_pos[i] >= 0 else 0.0 for i in range(len(variables))
    ])
    beta_hat = np.array([float(params.iloc[p]) for p in b_pos])
    rho_hat = float(params.iloc[rho_idx])
    d_pt, i_pt, t_pt = _effects(beta_hat[None, :], theta_hat[None, :], np.array([rho_hat]))

    draw_cols = list(b_pos) + [p for p in t_pos if p >= 0] + [rho_idx]
    Vb = V[np.ix_(draw_cols, draw_cols)]
    mean = np.array([float(params.iloc[p]) for p in draw_cols])
    n_rejected = 0
    if not np.all(np.isfinite(Vb)):
        se_d = np.full(len(variables), np.nan)
        se_i = se_d.copy()
        se_t = se_d.copy()
    else:
        rng = np.random.default_rng(seed)
        kept = []
        attempts = 0
        max_attempts = 10 * n_draws
        while len(kept) < n_draws and attempts < max_attempts:
            batch = _psd_draws(mean, Vb, n_draws, rng, func)
            attempts += n_draws
            ok = (batch[:, -1] > stab[0]) & (batch[:, -1] < stab[1])
            n_rejected += int(np.sum(~ok))
            kept.append(batch[ok])
            if sum(len(a) for a in kept) >= n_draws:
                break
        D = np.vstack(kept) if kept else np.empty((0, len(draw_cols)))
        if len(D) < n_draws:
            raise ValueError(
                f"{func}: only {len(D)} of {n_draws} impact draws stayed inside the "
                f"stability interval {stab} after {attempts} attempts; the spatial "
                "parameter is too close to the boundary for simulation-based impact "
                "inference. Pass impacts=False."
            )
        D = D[:n_draws]
        bet_d = D[:, :len(variables)]
        the_d = np.zeros((n_draws, len(variables)))
        col = len(variables)
        for i in range(len(variables)):
            if t_pos[i] >= 0:
                the_d[:, i] = D[:, col]
                col += 1
        rho_d = D[:, -1]
        dd, ii, tt = _effects(bet_d, the_d, rho_d)
        se_d = dd.std(axis=0, ddof=1)
        se_i = ii.std(axis=0, ddof=1)
        se_t = tt.std(axis=0, ddof=1)

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    out = pd.DataFrame(
        {
            "direct": d_pt[0], "direct_se": se_d,
            "direct_lo": d_pt[0] - z_crit * se_d, "direct_hi": d_pt[0] + z_crit * se_d,
            "indirect": i_pt[0], "indirect_se": se_i,
            "indirect_lo": i_pt[0] - z_crit * se_i, "indirect_hi": i_pt[0] + z_crit * se_i,
            "total": t_pt[0], "total_se": se_t,
            "total_lo": t_pt[0] - z_crit * se_t, "total_hi": t_pt[0] + z_crit * se_t,
        },
        index=pd.Index(variables, name="variable"),
    )
    return out, n_rejected
