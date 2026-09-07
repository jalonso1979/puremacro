"""Global VAR (GVAR): country VARX*(p, q) models linked by trade weights.

The GVAR of Pesaran, Schuermann & Weiner (2004), in the exposition of Dees,
di Mauro, Pesaran & Smith (2007). Each country ``i`` is a small VARX* in its
own variables ``x_it`` (``k_i x 1``) and a *foreign* (star) aggregate
``x*_it = sum_j w_ij x_jt`` built from a trade-flow weight matrix with zero
diagonal and unit row sums::

    x_it = a_i0 + a_i1 t + sum_{l=1..p_i} Phi_il x_{i,t-l}
           + Lambda_i0 x*_it + sum_{l=1..q_i} Lambda_il x*_{i,t-l}
           + sum_{l=0..q_d} Psi_il d_{t-l} + u_it

with ``d_t`` optional strictly exogenous global variables. Every equation of
country ``i`` shares one regressor matrix, so seemingly-unrelated regression
collapses to equation-by-equation OLS; *weak exogeneity* of ``x*_it``
(Pesaran, Shin & Smith 2000) together with the granularity condition
``max_j w_ij -> 0`` is what makes that OLS consistent. The two claims are
separate and this module keeps them separate.

The link step is exact matrix algebra. Order the global vector
``x_t = (x_1t', ..., x_Nt')'`` (``k = sum_i k_i``). Let ``S_i`` (``k_i x k``)
select country ``i``'s variables and ``W*_i`` (``k*_i x k``) build its star
variables, so ``z_it = (x_it', x*_it')' = W_i x_t`` with ``W_i = [S_i; W*_i]``.
Writing ``A_i0 = [I, -Lambda_i0]`` and ``A_il = [Phi_il, Lambda_il]``
(zero-padded to ``s = max_i max(p_i, q_i)``), stacking
``A_i0 W_i x_t = a_i0 + a_i1 t + sum_l A_il W_i x_{t-l} + u_it`` over ``i``
gives the global system::

    G x_t = a_0 + a_1 t + sum_{l=1..s} H_l x_{t-l} + sum_l Upsilon_l d_{t-l} + eps_t

with row block ``i`` of ``G`` equal to ``S_i - Lambda_i0 W*_i`` and of ``H_l``
equal to ``Phi_il S_i + Lambda_il W*_i``. ``G`` is square and, when it is
non-singular, ``x_t = G^-1(...)`` is an ordinary VAR(s) whose reduced-form
innovations are ``G^-1 eps_t``.

Inference is generalised (Pesaran & Shin 1998). With ``Psi_h`` the MA
coefficients of the solved VAR(s) and ``R_h = Psi_h G^-1``,

* ``GIRF_h(d) = R_h Sigma_eps d / sqrt(d' Sigma_eps d)`` -- the response to a
  one-standard-deviation innovation in the combination ``d`` of the country
  equations;
* the generalised FEVD replaces ``Psi_l`` by ``R_l`` and ``Sigma`` by
  ``Sigma_eps`` in the Pesaran-Shin formula;
* the persistence profile of Pesaran & Shin (1996) is
  ``PP(b, h) = b' R_h Sigma_eps R_h' b / (b' R_0 Sigma_eps R_0' b)``,
  which equals 1 at ``h = 0`` by construction.

Units, conditioning and honest defaults
---------------------------------------
Under a diagonal rescaling ``x -> D x`` the link matrix transforms as
``G -> D G D^-1``: a similarity transform that leaves ``det(G)`` and the
eigenvalues alone but moves ``cond(G)`` by orders of magnitude. Reporting or
gating on the raw condition number would reject a well-posed model merely
because one series is quoted in millions. ``G`` is therefore *equilibrated*
(LAPACK balancing followed by Ruiz max-norm sweeps) before the conditioning
gate, which leaves the reported number invariant to a diagonal rescaling up
to a small factor rather than eleven orders of magnitude, and
:attr:`GVARResult.condition_number` is the equilibrated number;
:attr:`GVARResult.condition_number_raw` is kept alongside it for reference
only. Country design matrices are column-equilibrated (each column divided by
its root-mean-square) before the ``inv_xtx`` singularity gate for the same
reason, and the scaling is undone exactly in the coefficients, standard
errors and ``(Z'Z)^-1``.

Generalised FEVD row sums
-------------------------
The Pesaran-Shin ``>= 1`` bound on the unnormalised generalised FEVD row sum
**does not hold for a GVAR**, not even at ``h = 0``. Their proof needs
``Psi_0 = I``, so that row ``i`` at impact is ``sum_j r_ij^2 >= r_ii^2 = 1``
for the residual correlation matrix ``r``. A GVAR substitutes ``R_0 = G^-1``,
and with ``a = R_0' e_i``, ``b = Sigma_eps^(1/2) a`` and ``C`` the
column-normalised ``Sigma_eps^(1/2)`` the impact row sum is ``|C'b|^2/|b|^2``,
a Rayleigh quotient of ``C C'``. Its eigenvalues are those of
``C'C = corr(Sigma_eps)``, so::

    lambda_min(corr(Sigma_eps)) <= rowsum_i(h=0) <= lambda_max(corr(Sigma_eps))

and nothing more. Only when ``G = I`` -- no contemporaneous star block
anywhere, hence ``R_0 = I`` and ``b`` proportional to a column of ``C`` -- does
the ``>= 1`` bound come back. It is not a rare failure: on the module's own
three-country x two-variable test DGP, sweeping only the seed, 50 of 59 fits
have an impact row sum below one (smallest observed 0.866). For ``h >= 1`` the
row sum can be far below one (values near 0.28 occur on ordinary
three-variable systems). What *is* true at every horizon is that the shares
are non-negative and that ``normalize=True`` (the default) sums each row to
exactly one. Never read an unnormalised row sum as a completeness check.

What this module deliberately does **not** do (see ``docs/gvar.md``):

* **No VECX*/cointegration layer.** The country models are unrestricted VARX*
  in levels: no rank test, no restricted trend, no error-correction terms.
  Consequently :func:`gvar`'s weak-exogeneity test is the *approximate
  reduced-form* residual-augmented F-test described in
  :class:`WeakExogeneityResult`, **not** the error-correction test of Dees et
  al. (2007, eq. 15). The lagged VARX* residual is a generated regressor and
  no generated-regressor correction is applied, so the test is approximate in
  a second, independent sense.

  **Measured size (disclosed rather than omitted).** 1000 replications per
  cell, three independent countries x two variables each (weak exogeneity
  holds by construction), ``p = q = 1``, ``we_transform='difference'``, 6000
  tests per cell. On I(1) levels -- the canonical GVAR input -- the test is
  **over-sized and the distortion does not shrink in T**::

      random walks, T = 200:  0.027 at nominal 0.01, 0.109 at 0.05, 0.178 at 0.10
      random walks, T = 300:  0.035 at nominal 0.01, 0.114 at 0.05, 0.182 at 0.10
      AR(1) rho = 0.5, T=200: 0.000 at nominal 0.01, 0.002 at 0.05, 0.011 at 0.10

  That is roughly a factor of two too many rejections on levels and an order
  of magnitude too few on stationary data. The stability in ``T`` identifies
  it as the unaddressed generated-regressor problem, not a small-sample
  artefact. The test therefore ships as a **diagnostic ordering of p-values,
  not a calibrated decision rule**: there is no ``expected_rejections``
  headline, and the boolean column is named ``reject_nominal`` to say plainly
  that ``alpha`` is a nominal threshold and not the test's true size. Read the
  ``p_value`` / ``p_holm`` ranking; do not count rejections.
* **No per-country variable sets.** Every country must carry every column of
  ``data``: an entirely NaN ``(country, variable)`` cell raises a
  ``ValueError`` naming the country and the variable, exactly like an interior
  NaN. Accidentally dropping a column for one country would otherwise estimate
  a silently different model. The oil-producer / US asymmetry, in which one
  country genuinely has no counterpart for a variable, is available only as an
  explicit opt-in: ``gvar(..., allow_missing_variables=True)`` restores the
  permissive reading of an all-NaN cell and emits a ``RuntimeWarning`` naming
  each dropped ``(country, variable)``. A star block may name any variable some
  *other* country carries, even one country ``i`` does not hold itself.
  Genuinely per-country variable *definitions* are out of scope.
* **A variable that is endogenous in one country cannot be exogenous in
  another.** A name appearing in both ``data`` and ``exog`` raises: the solved
  global system would otherwise condition on a variable that is endogenous to
  one of its own blocks, which is wrong rather than merely restrictive.
* No structural identification of the global system beyond the generalised
  route: no Cholesky, sign or narrative restrictions on the GVAR.
* No time-varying (rolling) trade weights, no rolling-window estimation.
* No shocks to the global exogenous block ``d_t``; the solved system
  conditions on it.
* No dominant-unit / factor-augmented GVAR (Chudik & Pesaran 2011).
* No structural-break testing of the country equations.
* No shrinkage for ``Sigma_eps``. The canonical GVAR has ``k >> T_eff``, in
  which case ``Sigma_eps`` is singular; the module warns at ``T_eff < 2k`` and
  again at ``T_eff < k`` rather than silently regularising.
* Bootstrap bands are pointwise percentile bands, never joint over horizons,
  and carry no bias correction.

Pure numpy / scipy / pandas (matplotlib only inside ``plot``), so the module
runs under Pyodide.

References
----------
Pesaran, M.H., Schuermann, T. and Weiner, S.M. (2004). Modeling regional
    interdependencies using a global error-correcting macroeconometric model.
    Journal of Business & Economic Statistics 22(2), 129-162.
Dees, S., di Mauro, F., Pesaran, M.H. and Smith, L.V. (2007). Exploring the
    international linkages of the euro area: a global VAR analysis.
    Journal of Applied Econometrics 22(1), 1-38.
Pesaran, M.H., Shin, Y. and Smith, R.J. (2000). Structural analysis of vector
    error correction models with exogenous I(1) variables.
    Journal of Econometrics 97(2), 293-343.
Pesaran, M.H. and Shin, Y. (1998). Generalized impulse response analysis in
    linear multivariate models. Economics Letters 58(1), 17-29.
Pesaran, M.H. and Shin, Y. (1996). Cointegration and speed of convergence to
    equilibrium. Journal of Econometrics 71(1-2), 117-143.
Chudik, A. and Pesaran, M.H. (2016). Theory and practice of GVAR modelling.
    Journal of Economic Surveys 30(1), 165-197.
Sims, C.A., Stock, J.H. and Watson, M.W. (1990). Inference in linear time
    series models with some unit roots. Econometrica 58(1), 113-144.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy import stats

from .._linalg import inv_xtx
from .estimate import companion

__all__ = [
    "gvar",
    "solve_gvar",
    "star_variables",
    "GVARResult",
    "CountryVARX",
    "GVARGIRFResult",
    "WeakExogeneityResult",
]

_WE_TEST_NAME = (
    "approximate reduced-form residual-augmented F-test "
    "(NOT the VECX* test of Dees et al. 2007, eq. 15)"
)


# ---------------------------------------------------------------------------
# Small shared numerics -- candidates for hoisting (see the module report)
# ---------------------------------------------------------------------------
def _render(df: pd.DataFrame, fmt: str, **kwargs: Any) -> str:
    """Dispatch a frame to the package's markdown / LaTeX / Typst renderers."""
    from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst

    if fmt == "markdown":
        return _df_to_markdown(df, index=False, **kwargs)
    if fmt == "latex":
        return _df_to_latex(df, index=False, **kwargs)
    return _df_to_typst(df, index=False, **kwargs)


def _ma_coefficients(A_list: Sequence[np.ndarray], horizon: int) -> np.ndarray:
    """MA coefficients of a VAR(p): ``Psi_0 = I``, ``Psi_h = sum_l Psi_{h-l} A_l``.

    Parameters
    ----------
    A_list : sequence of (n, n) ndarray
        Autoregressive coefficient matrices ``A_1 .. A_p``.
    horizon : int
        Largest horizon ``H``; ``H + 1`` matrices are returned.

    Returns
    -------
    ndarray, shape (H + 1, n, n)
    """
    A_list = [np.asarray(A, dtype=float) for A in A_list]
    n = A_list[0].shape[0]
    p = len(A_list)
    Phi = [np.eye(n)]
    for h in range(1, horizon + 1):
        Ph = np.zeros((n, n))
        for l in range(1, min(h, p) + 1):
            Ph = Ph + Phi[h - l] @ A_list[l - 1]
        Phi.append(Ph)
    return np.stack(Phi)


def _gfevd_from_ma(
    Psi: np.ndarray,
    Sigma: np.ndarray,
    *,
    normalize: bool = True,
    name: str = "gfevd",
) -> np.ndarray:
    """Pesaran-Shin (1998) generalised FEVD from MA coefficients.

    Identical estimand to :func:`puremacro.var.irf.gfevd`, but takes the MA
    coefficients directly so the GVAR can pass ``R_h = Psi_h G^-1``.

    Parameters
    ----------
    Psi : ndarray, shape (H + 1, n, n)
        MA coefficients (``Psi[0]`` need not be the identity: for the GVAR it
        is ``G^-1``).
    Sigma : ndarray, shape (n, n)
        Innovation covariance.
    normalize : bool
        Rescale each row to sum to one (the Diebold-Yilmaz convention).
    name : str
        Caller label used in the error message.

    Returns
    -------
    ndarray, shape (H + 1, n, n)

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``Sigma`` has a non-positive diagonal entry: the estimand divides
        by ``sigma_jj``, so such a shock has no defined contribution. No
        absolute floor is applied, because a floor would break the exact
        invariance of the estimand to a change of units.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    sigma_jj = np.diag(Sigma)
    bad = np.flatnonzero(~(sigma_jj > 0.0))
    if bad.size:
        raise np.linalg.LinAlgError(
            f"{name}: Sigma has a non-positive diagonal entry at variable(s) "
            f"{bad.tolist()} (values {sigma_jj[bad].tolist()}). The "
            "Pesaran-Shin GFEVD divides by sigma_jj, so those shocks have no "
            "defined contribution. Drop the degenerate variable(s) or check "
            "the residual covariance."
        )
    horizon = Psi.shape[0] - 1
    out = np.zeros((horizon + 1, n, n))
    num_cum = np.zeros((n, n))
    den_cum = np.zeros(n)
    for h in range(horizon + 1):
        Ph = Psi[h]
        PS = Ph @ Sigma
        num_cum = num_cum + (PS ** 2) / sigma_jj[None, :]
        var_i = np.einsum("ij,jk,ik->i", Ph, Sigma, Ph)
        den_cum = den_cum + np.maximum(var_i, 0.0)
        theta = num_cum / den_cum[:, None]
        if normalize:
            row_sum = theta.sum(axis=1, keepdims=True)
            theta = theta / np.where(row_sum == 0, 1.0, row_sum)
        out[h] = theta
    return out


def _collinear_names(Z: np.ndarray, names: Sequence[str]) -> list[str]:
    """Names of the columns most aligned with ``Z``'s null space."""
    n_cols = Z.shape[1]
    rank = int(np.linalg.matrix_rank(Z))
    deficit = max(n_cols - rank, 1)
    _, _, vh = np.linalg.svd(Z, full_matrices=False)
    loadings = np.abs(vh[-deficit:]).sum(axis=0)
    order = np.argsort(loadings)[::-1][:deficit]
    return [str(names[j]) for j in order]


def _multivariate_ols(
    Z: np.ndarray,
    Y: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    name: str = "OLS",
) -> dict:
    """Multivariate OLS with a shared regressor matrix (SUR collapses to OLS).

    Because every equation shares ``Z``, ``Var(vec(B)) = Sigma (x) (Z'Z)^-1``
    and ``se[r, c] = sqrt(Sigma[c, c] * (Z'Z)^-1[r, r])``.

    The design matrix is column-equilibrated (each column divided by its
    root-mean-square) before the ``inv_xtx`` singularity gate, so a regressor
    quoted in millions is not mistaken for a collinear one; the scaling is
    undone exactly afterwards.

    Parameters
    ----------
    Z : ndarray, shape (T, m)
        Regressors.
    Y : ndarray, shape (T, k)
        Dependent variables.
    names : sequence of str, optional
        Regressor names, used only in error messages.
    name : str
        Caller label for the diagnostic error.

    Returns
    -------
    dict
        ``coef`` (m, k), ``resid``, ``fitted``, ``Sigma`` (dof-corrected),
        ``Sigma_ml`` (MLE divisor), ``XtX_inv``, ``se``, ``tstat``,
        ``pvalue``, ``dof``, ``loglik``, ``r2``, ``condition_number``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the design matrix is rank deficient; the message names the
        offending regressors.
    """
    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T, m = Z.shape
    k = Y.shape[1]
    if names is None:
        names = [f"x{j}" for j in range(m)]
    scale = np.sqrt(np.mean(Z ** 2, axis=0))
    scale = np.where(scale > 0.0, scale, 1.0)
    Zs = Z / scale[None, :]
    try:
        XtX_inv_s = inv_xtx(Zs, name=name)
    except np.linalg.LinAlgError as exc:
        culprits = _collinear_names(Zs, names)
        raise np.linalg.LinAlgError(
            f"{name}: the design matrix is rank deficient "
            f"({T} rows, {m} columns). Regressors most aligned with the null "
            f"space: {culprits}. Drop a redundant regressor, shorten the lag "
            f"order, or restrict the star block. (underlying: {exc})"
        ) from exc
    XtX_inv = XtX_inv_s / np.outer(scale, scale)
    coef_s = np.linalg.lstsq(Zs, Y, rcond=None)[0]
    coef = coef_s / scale[:, None]
    fitted = Z @ coef
    resid = Y - fitted
    dof = T - m
    Sigma = resid.T @ resid / dof
    Sigma_ml = resid.T @ resid / T
    se = np.sqrt(np.outer(np.diag(XtX_inv), np.diag(Sigma)))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, coef / se, np.nan)
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), dof)
    sign, logdet = np.linalg.slogdet(Sigma_ml)
    if sign > 0:
        loglik = -0.5 * T * (k * np.log(2.0 * np.pi) + logdet + k)
    else:
        loglik = float("nan")
    ss_res = np.sum(resid ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    sv = np.linalg.svd(Zs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return {
        "coef": coef,
        "resid": resid,
        "fitted": fitted,
        "Sigma": Sigma,
        "Sigma_ml": Sigma_ml,
        "XtX_inv": XtX_inv,
        "se": se,
        "tstat": tstat,
        "pvalue": pvalue,
        "dof": dof,
        "loglik": float(loglik),
        "r2": r2,
        "condition_number": cond,
    }


def _equilibrate(A: np.ndarray, n_iter: int = 12) -> np.ndarray:
    """Return a scaled copy of ``A`` whose conditioning is (nearly) unit free.

    ``G`` transforms as ``G -> D G D^-1`` under a diagonal rescaling of the
    data, so ``cond(G)`` on raw units carries no information (a single series
    quoted in millions moves it by eleven orders of magnitude). The matrix is
    first passed through LAPACK's balancing (``scipy.linalg.matrix_balance``,
    the transform the eigen-solvers themselves use, whose scalings are exact
    powers of two and therefore reproducible) and then through ``n_iter``
    Ruiz max-norm sweeps. The result is invariant to a diagonal similarity up
    to a small factor -- measured at 2-4x on random 6x6 systems rescaled by
    1e-3 .. 1e-9 -- rather than exactly invariant, which no cheap scaling
    achieves.
    """
    M = np.array(A, dtype=float, copy=True)
    if M.size and np.all(np.isfinite(M)):
        try:
            M, _ = sla.matrix_balance(M, permute=False)
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - defensive
            M = np.array(A, dtype=float, copy=True)
    for _ in range(n_iter):
        rm = np.max(np.abs(M), axis=1)
        rm = np.where(rm > 0, rm, 1.0)
        M = (1.0 / np.sqrt(rm))[:, None] * M
        cm = np.max(np.abs(M), axis=0)
        cm = np.where(cm > 0, cm, 1.0)
        M = M * (1.0 / np.sqrt(cm))[None, :]
    return M


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------------------
# The link/solution step
# ---------------------------------------------------------------------------
class GVARSolution(NamedTuple):
    """Return value of :func:`solve_gvar`.

    A ``NamedTuple`` rather than a result dataclass: it is a container for the
    solved matrices, not a presentation object. Its first four entries are
    ``(G, H, F, G_inv)``, so ``G, H, F, G_inv = solve_gvar(...)[:4]`` works.

    It is deliberately **absent from** ``__all__``: it carries none of the six
    presentation methods (G2) that every exported result of this package
    provides, so exporting it would advertise a contract it does not meet.
    Reach it as :func:`solve_gvar`'s return value, or unpack it.

    Attributes
    ----------
    G : ndarray, shape (k, k)
    H : ndarray, shape (s, k, k)
    F : ndarray, shape (s, k, k)
        ``F_l = G^-1 H_l``.
    G_inv : ndarray, shape (k, k)
        ``R_0``, the impact of ``eps`` on ``x``.
    a0 : ndarray, shape (k,) or None
        ``G^-1 a_0`` when ``a0`` was supplied.
    a1 : ndarray, shape (k,) or None
        ``G^-1 a_1`` when ``a1`` was supplied.
    Upsilon : ndarray, shape (q_d + 1, k, k_d) or None
        ``G^-1 Upsilon_l`` when ``Upsilon`` was supplied.
    condition_number : float
        Condition number of the *equilibrated* ``G``.
    condition_number_raw : float
        Condition number of ``G`` in the caller's units (not unit invariant;
        reported for reference only).
    """

    G: np.ndarray
    H: np.ndarray
    F: np.ndarray
    G_inv: np.ndarray
    a0: np.ndarray | None
    a1: np.ndarray | None
    Upsilon: np.ndarray | None
    condition_number: float
    condition_number_raw: float


def _singular_G_message(
    G: np.ndarray,
    countries: Sequence[str],
    block_sizes: Sequence[int],
    cond_eq: float,
    cond_tol: float,
) -> str:
    """Build the diagnostic for a singular / ill-conditioned ``G``.

    Recomputes a *full* SVD -- only on the failure path, so the happy path
    keeps the cheap ``compute_uv=False`` call.
    """
    M = _equilibrate(G)
    u, sv, vh = np.linalg.svd(M)
    left = np.abs(u[:, -1])
    right = np.abs(vh[-1, :])
    offs = np.concatenate([[0], np.cumsum(block_sizes)])
    row_load = {
        countries[i]: float(np.sum(left[offs[i]:offs[i + 1]] ** 2))
        for i in range(len(countries))
    }
    col_load = {
        countries[i]: float(np.sum(right[offs[i]:offs[i + 1]] ** 2))
        for i in range(len(countries))
    }
    top_rows = sorted(row_load, key=row_load.get, reverse=True)[:3]
    top_cols = sorted(col_load, key=col_load.get, reverse=True)[:3]
    return (
        f"solve_gvar: the contemporaneous link matrix G is singular or "
        f"ill-conditioned (equilibrated cond(G) = {cond_eq:.3e}, tolerance "
        f"{cond_tol:.1e}; smallest singular value {sv[-1]:.3e}). The country "
        f"row blocks loading most on the null space are {top_rows} and the "
        f"column blocks {top_cols}. A singular G means the contemporaneous "
        "cross-country system x_t = Lambda_0 W* x_t + ... has no unique "
        "solution -- the textbook case is two one-variable countries with "
        "G = [[1, -l1], [-l2, 1]], singular exactly when l1*l2 = 1. Restrict "
        "the contemporaneous star block (contemporaneous_star=) or check the "
        "trade weights."
    )


def solve_gvar(
    link_matrices: Mapping[str, np.ndarray],
    A0: Mapping[str, np.ndarray],
    A_lags: Mapping[str, Sequence[np.ndarray]],
    *,
    country_order: Sequence[str],
    block_sizes: Mapping[str, int],
    a0: Mapping[str, np.ndarray] | None = None,
    a1: Mapping[str, np.ndarray] | None = None,
    Upsilon: Mapping[str, Sequence[np.ndarray]] | None = None,
    cond_tol: float = 1e12,
) -> GVARSolution:
    """Build and solve the global link system from per-country blocks.

    The pure link/solution step, exposed so it can be tested and taught
    independently of estimation. Row block ``i`` of ``G`` is
    ``A0[i] @ link_matrices[i]`` and of ``H_l`` is
    ``A_lags[i][l] @ link_matrices[i]``, zero-padded to
    ``s = max_i len(A_lags[i])``. One LU factorisation of ``G`` then serves
    ``G_inv``, every ``F_l = G^-1 H_l`` and the solved deterministics.

    Parameters
    ----------
    link_matrices : mapping str -> ndarray, shape (k_i + k*_i, k)
        ``W_i = [S_i; W*_i]`` for each country.
    A0 : mapping str -> ndarray, shape (k_i, k_i + k*_i)
        ``[I, -Lambda_i0]``.
    A_lags : mapping str -> sequence of ndarray, shape (k_i, k_i + k*_i)
        ``[Phi_il, Lambda_il]`` for ``l = 1..s_i``.
    country_order : sequence of str
        Row-block order; must match the global variable ordering.
    block_sizes : mapping str -> int
        ``k_i``; the sum must equal the common column count ``k``.
    a0, a1 : mapping str -> ndarray, shape (k_i,), optional
        Country intercepts / trend coefficients. When given, the solved
        ``G^-1 a_0`` and ``G^-1 a_1`` are returned.
    Upsilon : mapping str -> sequence of ndarray, shape (k_i, k_d), optional
        Global-exogenous coefficient blocks ``Psi_i0 .. Psi_i,q_d``.
    cond_tol : float
        Reject ``G`` when the *equilibrated* condition number exceeds this.

    Returns
    -------
    GVARSolution

    Raises
    ------
    ValueError
        On any shape mismatch, naming the country and both shapes.
    numpy.linalg.LinAlgError
        When ``G`` is singular or its equilibrated condition number exceeds
        ``cond_tol``; the message names the country blocks aligned with the
        null space.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.var.gvar import solve_gvar
    >>> S = {"A": np.array([[1.0, 0.0]]), "B": np.array([[0.0, 1.0]])}
    >>> Wst = {"A": np.array([[0.0, 1.0]]), "B": np.array([[1.0, 0.0]])}
    >>> link = {c: np.vstack([S[c], Wst[c]]) for c in ("A", "B")}
    >>> A0 = {"A": np.array([[1.0, -0.3]]), "B": np.array([[1.0, -0.2]])}
    >>> Al = {"A": [np.array([[0.5, 0.1]])], "B": [np.array([[0.4, 0.0]])]}
    >>> sol = solve_gvar(link, A0, Al, country_order=["A", "B"],
    ...                  block_sizes={"A": 1, "B": 1})
    >>> np.round(sol.G, 3)
    array([[ 1. , -0.3],
           [-0.2,  1. ]])
    >>> bool(np.allclose(sol.F[0], np.linalg.solve(sol.G, sol.H[0])))
    True
    """
    countries = list(country_order)
    missing = [c for c in countries if c not in link_matrices]
    if missing:
        raise ValueError(
            f"solve_gvar: no link matrix for country/countries {missing}; "
            f"link_matrices has keys {sorted(link_matrices)}."
        )
    k = None
    for c in countries:
        Wi = np.asarray(link_matrices[c], dtype=float)
        if Wi.ndim != 2:
            raise ValueError(
                f"solve_gvar: link_matrices[{c!r}] must be 2-d, got shape {Wi.shape}."
            )
        if k is None:
            k = Wi.shape[1]
        elif Wi.shape[1] != k:
            raise ValueError(
                f"solve_gvar: link_matrices[{c!r}] has {Wi.shape[1]} columns but "
                f"the first country has {k}; all link matrices must map the same "
                "global vector."
            )
    sizes = [int(block_sizes[c]) for c in countries]
    if sum(sizes) != k:
        raise ValueError(
            f"solve_gvar: block_sizes sum to {sum(sizes)} but the link matrices "
            f"have {k} columns. block_sizes must partition the global vector."
        )
    s = max((len(A_lags[c]) for c in countries), default=0)
    if s == 0:
        raise ValueError("solve_gvar: A_lags must supply at least one lag matrix.")

    G = np.zeros((k, k))
    H = np.zeros((s, k, k))
    a0_stack = np.zeros(k) if a0 is not None else None
    a1_stack = np.zeros(k) if a1 is not None else None
    q_d = None
    if Upsilon is not None:
        q_d = max(len(Upsilon[c]) for c in countries)
        k_d = None
        for c in countries:
            for M in Upsilon[c]:
                k_d = np.asarray(M).shape[1]
                break
            if k_d is not None:
                break
        Ups_stack = np.zeros((q_d, k, k_d if k_d else 0))
    else:
        Ups_stack = None

    off = 0
    for c, ki in zip(countries, sizes):
        Wi = np.asarray(link_matrices[c], dtype=float)
        A0i = np.asarray(A0[c], dtype=float)
        if A0i.shape != (ki, Wi.shape[0]):
            raise ValueError(
                f"solve_gvar: A0[{c!r}] has shape {A0i.shape}, expected "
                f"({ki}, {Wi.shape[0]}) = (k_i, k_i + k*_i)."
            )
        G[off:off + ki] = A0i @ Wi
        for l, Ail in enumerate(A_lags[c]):
            Ail = np.asarray(Ail, dtype=float)
            if Ail.shape != (ki, Wi.shape[0]):
                raise ValueError(
                    f"solve_gvar: A_lags[{c!r}][{l}] has shape {Ail.shape}, "
                    f"expected ({ki}, {Wi.shape[0]})."
                )
            H[l, off:off + ki] = Ail @ Wi
        if a0_stack is not None:
            a0_stack[off:off + ki] = np.asarray(a0[c], dtype=float).ravel()
        if a1_stack is not None:
            a1_stack[off:off + ki] = np.asarray(a1[c], dtype=float).ravel()
        if Ups_stack is not None:
            for l, M in enumerate(Upsilon[c]):
                Ups_stack[l, off:off + ki] = np.asarray(M, dtype=float)
        off += ki

    M_eq = _equilibrate(G)
    sv_eq = np.linalg.svd(M_eq, compute_uv=False)
    cond_eq = float(sv_eq[0] / sv_eq[-1]) if sv_eq[-1] > 0 else float("inf")
    sv_raw = np.linalg.svd(G, compute_uv=False)
    cond_raw = float(sv_raw[0] / sv_raw[-1]) if sv_raw[-1] > 0 else float("inf")
    if not np.isfinite(cond_eq) or cond_eq > cond_tol:
        raise np.linalg.LinAlgError(
            _singular_G_message(G, countries, sizes, cond_eq, cond_tol)
        )

    lu, piv = sla.lu_factor(G)
    G_inv = sla.lu_solve((lu, piv), np.eye(k))
    F = np.stack([sla.lu_solve((lu, piv), H[l]) for l in range(s)])
    a0_solved = sla.lu_solve((lu, piv), a0_stack) if a0_stack is not None else None
    a1_solved = sla.lu_solve((lu, piv), a1_stack) if a1_stack is not None else None
    if Ups_stack is not None:
        Ups_solved = np.stack([sla.lu_solve((lu, piv), Ups_stack[l]) for l in range(q_d)])
    else:
        Ups_solved = None
    return GVARSolution(
        G=G,
        H=H,
        F=F,
        G_inv=G_inv,
        a0=a0_solved,
        a1=a1_solved,
        Upsilon=Ups_solved,
        condition_number=cond_eq,
        condition_number_raw=cond_raw,
    )


# ---------------------------------------------------------------------------
# Panel coercion and star weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _GVARSpec:
    """Internal, immutable description of a fitted GVAR's design.

    Carried on :class:`GVARResult` so ``girf``'s bootstrap can re-estimate the
    whole system on simulated data at fixed lag orders. Not part of the public
    API and deliberately undocumented in the user-facing docs.
    """

    countries: tuple
    country_vars: dict
    names: tuple
    name_pairs: tuple
    own_idx: dict
    Sel: dict
    Wstar: dict
    star_names: dict
    contemp_idx: dict
    lag_orders: dict
    trend: str
    s: int
    t0: int
    T: int
    k: int
    exog: np.ndarray | None
    exog_names: tuple
    q_d: int
    reg_names: dict
    reg_blocks: dict
    reg_lags: dict
    dates_all: pd.Index
    sigma_ddof: int
    cond_tol: float


def _coerce_panel(
    data: Any,
    *,
    country_level: str,
    date_level: str,
    country_order: Sequence[str] | None,
    caller: str,
    allow_missing_variables: bool = False,
) -> tuple[pd.DataFrame, tuple, pd.Index, dict]:
    """Coerce ``data`` to a validated ``(country, date)`` panel.

    Returns ``(frame, countries, dates, country_vars)``. Every country must
    carry every column unless ``allow_missing_variables`` is set, in which case
    an entirely NaN ``(country, variable)`` is read as "this country does not
    hold that variable" and reported with a ``RuntimeWarning``.
    """
    if isinstance(data, pd.DataFrame):
        frame = data
    elif isinstance(data, Mapping):
        objs = list(data.values())
        keys = list(data.keys())
        if not objs:
            raise ValueError(f"{caller}: `data` mapping is empty.")
        for key, obj in zip(keys, objs):
            if not isinstance(obj, pd.DataFrame):
                raise TypeError(
                    f"{caller}: data[{key!r}] is {type(obj).__name__}, expected a "
                    "pandas DataFrame indexed by date."
                )
        frame = pd.concat(objs, keys=keys, names=[country_level, date_level])
    else:
        raise TypeError(
            f"{caller}: `data` must be a pandas DataFrame with a "
            f"({country_level}, {date_level}) MultiIndex, or a mapping "
            f"country -> DataFrame indexed by date; got {type(data).__name__}."
        )

    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError(
            f"{caller}: `data` must have a 2-level MultiIndex "
            f"({country_level}, {date_level}); found index "
            f"{type(frame.index).__name__} with {frame.index.nlevels} level(s) "
            f"named {list(frame.index.names)}."
        )
    if frame.index.names[0] != country_level or frame.index.names[1] != date_level:
        raise ValueError(
            f"{caller}: expected index level names "
            f"({country_level!r}, {date_level!r}) but found "
            f"{tuple(frame.index.names)}. Rename the levels or pass "
            "country_level= / date_level=."
        )
    if frame.index.has_duplicates:
        dups = frame.index[frame.index.duplicated()][:5].tolist()
        raise ValueError(
            f"{caller}: duplicated (country, date) rows, first five {dups}. "
            "Aggregate or drop the duplicates first."
        )
    if frame.shape[1] == 0:
        raise ValueError(f"{caller}: `data` has no columns.")

    bad_dtype = [
        (str(c), str(frame[c].dtype))
        for c in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if bad_dtype:
        raise ValueError(
            f"{caller}: non-numeric column(s) {bad_dtype}. Convert them to a "
            "float dtype first."
        )

    obs_order = list(dict.fromkeys(frame.index.get_level_values(0)))
    if country_order is not None:
        countries = list(country_order)
        missing = [c for c in countries if c not in obs_order]
        extra = [c for c in obs_order if c not in countries]
        if missing or extra:
            raise KeyError(
                f"{caller}: country_order does not match the data. Missing from "
                f"the data: {missing}; present in the data but not listed: {extra}."
            )
    else:
        countries = list(obs_order)
    if len(set(countries)) != len(countries):
        seen, dup = set(), []
        for c in countries:
            if c in seen:
                dup.append(c)
            seen.add(c)
        raise ValueError(f"{caller}: duplicated country id(s) {sorted(set(dup))}.")

    ref_dates = frame.loc[countries[0]].index
    if not ref_dates.is_monotonic_increasing:
        ref_dates = ref_dates.sort_values()
    offenders = []
    for c in countries[1:]:
        idx = frame.loc[c].index
        if len(idx) != len(ref_dates) or not idx.sort_values().equals(ref_dates):
            diff = sorted(set(ref_dates).symmetric_difference(set(idx)), key=str)[:5]
            offenders.append((c, diff))
    if offenders:
        raise ValueError(
            f"{caller}: the date index differs across countries. Offending "
            f"countries and the first five dates in the symmetric difference: "
            f"{offenders}. Reindex every country to the common sample first."
        )
    if len(ref_dates) < 2:
        raise ValueError(
            f"{caller}: only {len(ref_dates)} date(s) in the panel. A GVAR needs "
            "a time series per country; supply at least s + m_i + k_i + 1 periods."
        )

    frame = frame.sort_index()
    country_vars: dict[str, tuple] = {}
    dropped: list[tuple] = []
    for c in countries:
        sub = frame.loc[c].reindex(ref_dates)
        keep = []
        for v in frame.columns:
            col = sub[v]
            if col.isna().all():
                if not allow_missing_variables:
                    raise ValueError(
                        f"{caller}: country {c!r} does not carry variable "
                        f"{str(v)!r} -- the whole column is NaN. The default is "
                        "a COMMON variable set: every country must supply every "
                        "column of `data`, so that an accidentally dropped or "
                        "misnamed column cannot silently estimate a smaller "
                        "global model. If the omission is deliberate (the "
                        "oil-producer / US asymmetry), pass "
                        "allow_missing_variables=True."
                    )
                dropped.append((c, str(v)))
                continue
            if col.isna().any():
                first_bad = col.index[col.isna()][0]
                raise ValueError(
                    f"{caller}: interior missing value in country {c!r}, variable "
                    f"{str(v)!r}, first at date {first_bad!r}. A partial gap is "
                    "never allowed; an entirely NaN (country, variable) is "
                    "allowed only under allow_missing_variables=True."
                )
            if not np.isfinite(col.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"{caller}: country {c!r}, variable {str(v)!r} contains a "
                    "non-finite value (inf)."
                )
            keep.append(v)
        if not keep:
            raise ValueError(
                f"{caller}: country {c!r} has no usable variable (every column is "
                "entirely NaN)."
            )
        country_vars[c] = tuple(keep)
    if dropped:
        warnings.warn(
            f"{caller}: allow_missing_variables=True dropped {len(dropped)} "
            f"(country, variable) cell(s) that are entirely NaN: {dropped}. The "
            "global vector is smaller than the common variable set implies.",
            RuntimeWarning,
            stacklevel=3,
        )
    return frame, tuple(countries), ref_dates, country_vars


def _coerce_weights(weights: Any, countries: Sequence[str], caller: str) -> pd.DataFrame:
    """Coerce and validate the trade-weight matrix; return a labelled frame."""
    if isinstance(weights, Mapping):
        raise TypeError(
            f"{caller}: time-varying trade weights are not supported; pass one "
            "(N, N) weight matrix. A mapping of frames was given."
        )
    if isinstance(weights, pd.DataFrame):
        missing = [c for c in countries if c not in weights.index]
        missing_c = [c for c in countries if c not in weights.columns]
        if missing or missing_c:
            raise KeyError(
                f"{caller}: weight ids do not cover the data countries. Missing "
                f"rows {missing}, missing columns {missing_c}; weights carry "
                f"{list(weights.index)[:10]}."
            )
        Wc = weights.loc[list(countries), list(countries)].to_numpy(dtype=float)
    elif hasattr(weights, "ids") and hasattr(weights, "to_dense"):
        ids = list(weights.ids)
        missing = [c for c in countries if c not in ids]
        if missing:
            raise KeyError(
                f"{caller}: SpatialWeights ids do not cover the data countries; "
                f"missing {missing} (weights carry {ids[:10]})."
            )
        dense = weights.to_dense()
        pos = [ids.index(c) for c in countries]
        Wc = dense[np.ix_(pos, pos)].astype(float)
    else:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim == 3:
            raise TypeError(
                f"{caller}: time-varying trade weights are not supported; a 3-d "
                f"array of shape {arr.shape} was given."
            )
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"{caller}: `weights` must be a square (N, N) matrix, got shape "
                f"{arr.shape}."
            )
        if arr.shape[0] != len(countries):
            raise ValueError(
                f"{caller}: `weights` is {arr.shape[0]}x{arr.shape[0]} but the "
                f"panel has {len(countries)} countries."
            )
        Wc = arr

    if not np.isfinite(Wc).all():
        raise ValueError(f"{caller}: `weights` contains a non-finite entry.")
    if (Wc < 0).any():
        bad = [countries[i] for i in np.unique(np.argwhere(Wc < 0)[:, 0])]
        raise ValueError(
            f"{caller}: negative trade weights in row(s) {bad}. Trade shares must "
            "be non-negative."
        )
    if np.abs(np.diag(Wc)).max() > 0:
        bad = [countries[i] for i in np.flatnonzero(np.abs(np.diag(Wc)) > 0)]
        raise ValueError(
            f"{caller}: the weight matrix has a non-zero diagonal at {bad}. Set "
            "the diagonal of the trade matrix to zero; "
            "puremacro.spatial.economic_weights does this."
        )
    rs = Wc.sum(axis=1)
    bad_rows = [
        (countries[i], float(rs[i]))
        for i in range(len(countries))
        if rs[i] > 0 and abs(rs[i] - 1.0) > 1e-8
    ]
    if bad_rows:
        raise ValueError(
            f"{caller}: weight rows must sum to 1 within 1e-8. Offending "
            f"(country, row sum): {bad_rows}. Row-standardise the trade matrix "
            "(SpatialWeights.standardize() or economic_weights(...))."
        )
    return pd.DataFrame(Wc, index=list(countries), columns=list(countries))


def _default_star_sets(countries, country_vars) -> dict:
    """Country ``i``'s own variables that at least one other country carries."""
    out = {}
    for c in countries:
        out[c] = tuple(
            v for v in country_vars[c]
            if any(v in country_vars[o] for o in countries if o != c)
        )
    return out


def _resolve_star_sets(countries, country_vars, star_vars, caller) -> dict:
    if star_vars is None:
        return _default_star_sets(countries, country_vars)
    if not isinstance(star_vars, Mapping):
        raise TypeError(
            f"{caller}: `star_vars` must be a mapping country -> sequence of "
            f"variable names, got {type(star_vars).__name__}."
        )
    unknown = [c for c in star_vars if c not in countries]
    if unknown:
        raise KeyError(
            f"{caller}: star_vars names unknown country/countries {unknown}; "
            f"valid countries are {list(countries)}."
        )
    default = _default_star_sets(countries, country_vars)
    out = {}
    for c in countries:
        if c not in star_vars:
            out[c] = default[c]
            continue
        req = tuple(star_vars[c])
        for v in req:
            donors = [o for o in countries if o != c and v in country_vars[o]]
            if not donors:
                raise ValueError(
                    f"{caller}: star variable {v!r} requested for country {c!r} but "
                    "no other country carries it, so the renormalised weights would "
                    "have a zero denominator. Drop it from star_vars or add a donor "
                    "country."
                )
        if len(set(req)) != len(req):
            raise ValueError(
                f"{caller}: star_vars[{c!r}] repeats a variable: {req}."
            )
        out[c] = req
    return out


def _star_weight_frames(countries, country_vars, star_sets, Wc, caller) -> dict:
    """Per-country renormalised star weights, index = star var, columns = country."""
    W = Wc.to_numpy(dtype=float)
    idx = {c: i for i, c in enumerate(countries)}
    out = {}
    for c in countries:
        svars = star_sets[c]
        if not svars:
            out[c] = pd.DataFrame(
                np.zeros((0, len(countries))), index=[], columns=list(countries)
            )
            continue
        i = idx[c]
        if W[i].sum() <= 0:
            raise ValueError(
                f"{caller}: country {c!r} has an all-zero row in the trade weight "
                "matrix but a non-empty star block, so every foreign variable "
                f"would be identically zero and the design matrix singular. Pass "
                f"star_vars={{{c!r}: ()}} to give it a plain VAR, or supply trade "
                "partners."
            )
        rows = np.zeros((len(svars), len(countries)))
        for b, v in enumerate(svars):
            donors = [o for o in countries if o != c and v in country_vars[o]]
            denom = float(sum(W[i, idx[o]] for o in donors))
            if denom <= 0:
                raise ValueError(
                    f"{caller}: country {c!r}, star variable {v!r}: the countries "
                    f"carrying {v!r} ({donors}) receive zero trade weight from "
                    f"{c!r}, so the renormalised star weight is undefined. Drop "
                    f"{v!r} from star_vars[{c!r}] or adjust the weights."
                )
            for o in donors:
                rows[b, idx[o]] = W[i, idx[o]] / denom
        out[c] = pd.DataFrame(rows, index=list(svars), columns=list(countries))
    return out


def _global_index(countries, country_vars):
    names, pairs, own_idx = [], [], {}
    for c in countries:
        start = len(names)
        for v in country_vars[c]:
            names.append(f"{c}:{v}")
            pairs.append((c, v))
        own_idx[c] = np.arange(start, len(names))
    return tuple(names), tuple(pairs), own_idx


def _selection_matrices(countries, country_vars, star_sets, star_w, own_idx, k):
    Sel, Wstar = {}, {}
    col_of = {}
    for c in countries:
        for j, v in enumerate(country_vars[c]):
            col_of[(c, v)] = int(own_idx[c][j])
    for c in countries:
        ki = len(country_vars[c])
        S = np.zeros((ki, k))
        S[np.arange(ki), own_idx[c]] = 1.0
        Sel[c] = S
        svars = star_sets[c]
        Ws = np.zeros((len(svars), k))
        if svars:
            frame = star_w[c]
            for b, v in enumerate(svars):
                for o in countries:
                    w = float(frame.loc[v, o])
                    if w != 0.0:
                        Ws[b, col_of[(o, v)]] = w
        Wstar[c] = Ws
    return Sel, Wstar


def star_variables(
    data: pd.DataFrame | Mapping[str, pd.DataFrame],
    weights: Any,
    *,
    star_vars: Mapping[str, Sequence[str]] | None = None,
    country_order: Sequence[str] | None = None,
    country_level: str = "country",
    date_level: str = "date",
    suffix: str = "_star",
    allow_missing_variables: bool = False,
) -> pd.DataFrame:
    """Build the foreign (star) variable panel ``x*_it = sum_j w~_ij x_jt``.

    Exposed on its own so the star panel can be inspected, plotted or fed to
    another estimator without fitting a GVAR. The renormalised weight for
    country ``i`` and variable ``v`` is
    ``w~_ij = w_ij 1{v in K_j} / sum_{m != i, v in K_m} w_im``, so the row sums
    to one over the countries that actually carry ``v``.

    Parameters
    ----------
    data : DataFrame or mapping
        A ``(country, date)`` MultiIndexed frame, or a mapping
        ``country -> DataFrame`` indexed by date.
    weights : DataFrame, ndarray or SpatialWeights
        Trade weights: non-negative, zero diagonal, rows summing to one.
    star_vars : mapping str -> sequence of str, optional
        Star block per country. Default: country ``i``'s own variables that at
        least one other country also carries. A requested variable only needs a
        *donor* country, so a country may carry a foreign variable it does not
        hold itself.
    country_order : sequence of str, optional
        Country ordering; required when ``weights`` is a bare ndarray.
    country_level, date_level : str
        Index level names.
    suffix : str
        Appended to each variable name to form the output column.
    allow_missing_variables : bool
        Opt out of the common-variable-set rule; see :func:`gvar`. By default
        an entirely NaN ``(country, variable)`` raises.

    Returns
    -------
    pandas.DataFrame
        Same ``(country, date)`` MultiIndex as ``data``; one column per star
        variable, named ``f"{variable}{suffix}"``. Cells for a
        ``(country, variable)`` outside that country's star block are NaN.

    Raises
    ------
    TypeError, ValueError, KeyError
        The same data and weight validation errors as :func:`gvar`.
    ValueError
        If a generated column name collides with an existing data column.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.var.gvar import star_variables
    >>> rng = np.random.default_rng(0)
    >>> dates = pd.RangeIndex(6, name="date")
    >>> frames = {c: pd.DataFrame(rng.standard_normal((6, 1)), columns=["y"],
    ...                           index=dates) for c in ("A", "B", "C")}
    >>> W = pd.DataFrame([[0, .5, .5], [.5, 0, .5], [.5, .5, 0.]],
    ...                  index=list("ABC"), columns=list("ABC"))
    >>> xs = star_variables(frames, W)
    >>> list(xs.columns)
    ['y_star']
    >>> bool(np.isclose(xs.loc[("A", 0), "y_star"],
    ...                 0.5 * frames["B"].iloc[0, 0] + 0.5 * frames["C"].iloc[0, 0]))
    True
    """
    caller = "star_variables"
    frame, countries, dates, country_vars = _coerce_panel(
        data,
        country_level=country_level,
        date_level=date_level,
        country_order=country_order,
        caller=caller,
        allow_missing_variables=allow_missing_variables,
    )
    Wc = _coerce_weights(weights, countries, caller)
    star_sets = _resolve_star_sets(countries, country_vars, star_vars, caller)
    star_w = _star_weight_frames(countries, country_vars, star_sets, Wc, caller)
    names, pairs, own_idx = _global_index(countries, country_vars)
    k = len(names)
    _, Wstar = _selection_matrices(countries, country_vars, star_sets, star_w, own_idx, k)

    X = np.zeros((len(dates), k))
    for c in countries:
        sub = frame.loc[c].reindex(dates)
        X[:, own_idx[c]] = sub[list(country_vars[c])].to_numpy(dtype=float)

    all_star = list(dict.fromkeys(v for c in countries for v in star_sets[c]))
    cols = [f"{v}{suffix}" for v in all_star]
    clash = [c for c in cols if c in frame.columns]
    if clash:
        raise ValueError(
            f"{caller}: generated star column(s) {clash} collide with existing "
            f"data columns. Pass a different suffix= (currently {suffix!r})."
        )
    out = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_product(
            [list(countries), list(dates)], names=[country_level, date_level]
        ),
        columns=cols,
    )
    for c in countries:
        if not star_sets[c]:
            continue
        Xs = X @ Wstar[c].T
        for b, v in enumerate(star_sets[c]):
            out.loc[(c, slice(None)), f"{v}{suffix}"] = Xs[:, b]
    return out


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CountryVARX:
    """One country's estimated VARX*(p, q) block.

    Attributes
    ----------
    country : str
    variables : tuple of str
        The country's own ``k_i`` variables.
    star_variables : tuple of str
        The ``k*_i`` foreign variables.
    contemporaneous_star : tuple of str
        The subset entering ``Lambda_i0``.
    p, q : int
        Own and foreign lag orders.
    coef, se, tstat, pvalue : pandas.DataFrame, shape (m_i, k_i)
        Index = regressor names, columns = own variables. ``pvalue`` comes
        from ``t(T_eff - m_i)``, not the normal: ``T_eff`` is routinely 30-120.
    a0, a1 : ndarray, shape (k_i,)
        Intercept and trend coefficient (``a1`` is zero unless ``trend='ct'``).
    Phi : tuple of ndarray
        ``p`` matrices ``(k_i, k_i)``.
    Lambda : tuple of ndarray
        ``q + 1`` matrices ``(k_i, k*_i)``; ``Lambda[0]`` has zero columns for
        star variables excluded contemporaneously.
    Psi : tuple of ndarray
        ``q_d + 1`` matrices ``(k_i, k_d)``; empty when there is no ``exog``.
    Sigma : ndarray
        ``U'U / (T_eff - m_i)`` -- the dof-corrected covariance behind the
        standard errors.
    Sigma_ml : ndarray
        ``U'U / T_eff`` -- the MLE divisor behind ``loglik``/``aic``/``bic``,
        chosen so those numbers are comparable with the lag-selection criteria.
    resid, fitted : pandas.DataFrame, shape (T_eff, k_i)
    regressor_names, regressor_blocks, regressor_lags : tuple
        Column labels, their block (``'det'``, ``'star'``, ``'own'``,
        ``'exog'``) and lag.
    n_obs, n_params, dof : int
    r2 : pandas.Series
    loglik, aic, bic : float
    condition_number : float
        Condition number of the column-equilibrated design matrix.
    """

    country: str
    variables: tuple
    star_variables: tuple
    contemporaneous_star: tuple
    p: int
    q: int
    coef: pd.DataFrame
    se: pd.DataFrame
    tstat: pd.DataFrame
    pvalue: pd.DataFrame
    a0: np.ndarray
    a1: np.ndarray
    Phi: tuple
    Lambda: tuple
    Psi: tuple
    Sigma: np.ndarray
    Sigma_ml: np.ndarray
    resid: pd.DataFrame
    fitted: pd.DataFrame
    regressor_names: tuple
    regressor_blocks: tuple
    regressor_lags: tuple
    n_obs: int
    n_params: int
    dof: int
    r2: pd.Series
    loglik: float
    aic: float
    bic: float
    condition_number: float

    # -- presentation -------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Long coefficient table for this country."""
        rows = []
        for r, reg in enumerate(self.regressor_names):
            for c, eq in enumerate(self.variables):
                rows.append(
                    {
                        "equation": eq,
                        "regressor": reg,
                        "block": self.regressor_blocks[r],
                        "lag": self.regressor_lags[r],
                        "coef": float(self.coef.iloc[r, c]),
                        "se": float(self.se.iloc[r, c]),
                        "t": float(self.tstat.iloc[r, c]),
                        "p_value": float(self.pvalue.iloc[r, c]),
                    }
                )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary of the country block."""
        lines = [
            f"VARX*({self.p}, {self.q}) -- {self.country}: "
            f"{len(self.variables)} endogenous {list(self.variables)}, "
            f"{len(self.star_variables)} foreign {list(self.star_variables)}",
            f"  T_eff = {self.n_obs}, params/equation = {self.n_params}, "
            f"residual dof = {self.dof}",
        ]
        if self.dof < len(self.variables) + 5:
            lines.append(
                f"  !! only {self.dof} residual degrees of freedom for "
                f"{len(self.variables)} equations -- Sigma is barely identified; "
                "shorten the lags or restrict the star block."
            )
        lines.append(
            f"  log L = {self.loglik:.3f}   AIC = {self.aic:.3f}   "
            f"BIC = {self.bic:.3f}   (MLE divisor T_eff)"
        )
        lines.append(f"  cond(Z, equilibrated) = {self.condition_number:.3e}")
        if self.condition_number > 1e8:
            lines.append(
                "  !! the design matrix is badly conditioned; coefficients are "
                "fragile."
            )
        lines.append("")
        lines.append("  equation      R2     sigma   top |t| on contemporaneous x*")
        contemp = [
            r for r, b in enumerate(self.regressor_blocks)
            if b == "star" and self.regressor_lags[r] == 0
        ]
        for c, eq in enumerate(self.variables):
            sig = float(np.sqrt(self.Sigma[c, c]))
            if contemp:
                ts = np.abs([self.tstat.iloc[r, c] for r in contemp])
                best = int(np.nanargmax(ts)) if np.isfinite(ts).any() else 0
                lab = self.regressor_names[contemp[best]]
                tval = float(self.tstat.iloc[contemp[best], c])
                extra = f"{lab} (t = {tval:+.2f})"
            else:
                extra = "(none)"
            lines.append(
                f"  {eq:<12s} {float(self.r2[eq]):+.3f} {sig:9.4f}   {extra}"
            )
        lines.append("")
        lines.append(
            "  t statistics use t(dof), are conditional on the weak exogeneity of "
            "x*, and assume granularity (max_j w_ij -> 0). They are not "
            "trustworthy for a two- or three-country toy system."
        )
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, *, figsize: tuple[float, float] | None = None):
        """Residual series per equation with a +/- 2 sigma band.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n = len(self.variables)
        if figsize is None:
            figsize = (4.0 * n, 3.0)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        x = np.arange(self.n_obs)
        for j, v in enumerate(self.variables):
            ax = axes[0, j]
            u = self.resid[v].to_numpy(dtype=float)
            band = 2.0 * float(np.sqrt(self.Sigma[j, j]))
            ax.axhspan(-band, band, color="lightgrey", alpha=0.6)
            ax.plot(x, u, color="steelblue", linewidth=1.0)
            ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(f"{self.country}:{v}  R2 = {float(self.r2[v]):.2f}")
            ax.set_xlabel("observation")
        fig.tight_layout()
        return fig


@dataclass(frozen=True, eq=False)
class WeakExogeneityResult:
    """Approximate weak-exogeneity test for the foreign variables.

    **What is tested.** For every star variable ``x*_il`` of country ``i`` the
    auxiliary regression (in first differences by default, because canonical
    GVAR inputs are I(1) and the F statistic needs stationary regressors)

    ``D x*_{il,t} = mu + sum_a psi_a' D x*_{i,t-a} + sum_a phi_a' D x_{i,t-a}
    + gamma' u^_{i,t-1} + eta_t``

    is run and ``gamma = 0`` tested by ``F(k_i, T_e - m)``, where ``u^_i`` is
    country ``i``'s own VARX* residual vector. The whole lagged star block
    enters, as in Dees et al. (2007, eq. 15), not just lags of the tested
    variable.

    **What is not tested.** This is *not* the error-correction test of Dees et
    al. (2007, eq. 15): that regression uses the estimated cointegrating
    (error-correction) terms of a VECX*, and this module has no cointegration
    layer, so the model's own lagged errors stand in for them. The null here is
    the necessary condition that country ``i``'s model innovations do not
    Granger-cause its own foreign block. ``u^_{i,t-1}`` is a *generated*
    regressor and no generated-regressor correction is applied, so the
    distribution is approximate.

    **Measured size, and what it means for reading this table.** Under the
    null (three independent countries, ``p = q = 1``, 1000 replications, 6000
    tests per cell) the rejection rate is **not** the nominal one. On I(1)
    levels, the canonical GVAR input, it is 0.109 at nominal 0.05 and 0.178 at
    nominal 0.10 with ``T = 200``, and 0.114 / 0.182 with ``T = 300`` -- the
    distortion is stable in ``T``, which identifies it as the unaddressed
    generated-regressor problem rather than a small-sample artefact. On
    stationary data (AR(1) with rho = 0.5, ``T = 200``) it is conservative
    instead, 0.002 at nominal 0.05.

    Consequently this result deliberately reports **no calibrated rejection
    count**: there is no ``expected_rejections`` attribute, and the boolean
    column is ``reject_nominal``, named so it cannot be read as a decision at
    a true level ``alpha``. Rank the star variables by ``p_value`` (or
    ``p_holm`` for a family-wise ordering) and treat a small p-value as a
    reason to look at that country's star block, not as a test outcome.

    Attributes
    ----------
    table : pandas.DataFrame
        Columns ``country, star_variable, F, df_num, df_den, p_value, p_holm,
        reject_nominal, n_obs, lags``.
    transform : str
        ``'levels'`` or ``'difference'``.
    lags : dict
        Auxiliary-regression lag length per country.
    alpha : float
        The **nominal** threshold behind ``reject_nominal``. It is not the
        test's size; see above.
    n_tests : int
    n_reject_nominal : int
        How many p-values fall below the nominal ``alpha``. Uncalibrated: do
        not compare it with ``alpha * n_tests``.
    test_name : str
    skipped : tuple of str
        Countries with no star block, or too few observations for the
        auxiliary regression.
    """

    table: pd.DataFrame
    transform: str
    lags: dict
    alpha: float
    n_tests: int
    n_reject_nominal: int
    test_name: str
    skipped: tuple

    def to_frame(self) -> pd.DataFrame:
        """The test table itself."""
        return self.table.copy()

    def summary(self) -> str:
        """Human-readable summary, including the full disclaimer."""
        lines = [
            "Weak exogeneity of the foreign variables -- "
            "approximate reduced-form residual-augmented F-test",
            "",
            "  The null is that the LAGGED RESIDUALS of country i's VARX* enter",
            "  with zero coefficients in an auxiliary regression for each of that",
            "  country's foreign variables, i.e. that country i's model",
            "  innovations do not Granger-cause its own foreign block. This is a",
            "  NECESSARY condition for weak exogeneity. It is the reduced-form",
            "  counterpart of Dees, di Mauro, Pesaran & Smith (2007, eq. 15) with",
            "  the error-correction terms replaced by the model's own errors,",
            "  because this module does not estimate a VECX* and has no",
            "  cointegrating vectors to form them from. It is NOT that test. The",
            "  lagged residual is a generated regressor and no generated-regressor",
            "  correction is applied, so the F distribution is approximate.",
            "",
            "  MEASURED SIZE: on I(1) levels the rejection rate under the null",
            "  is 0.11 at nominal 0.05 and 0.18 at nominal 0.10 (1000 reps, T =",
            "  200 and 300 alike); on stationary data it is 0.002 at nominal",
            "  0.05. The nominal level is NOT the size, so the count below is a",
            "  descriptive tally, not a calibrated number of rejections. Rank by",
            "  p_value / p_holm instead of counting.",
            "",
            f"  {self.n_reject_nominal} of {self.n_tests} star variables fall "
            f"below the NOMINAL {self.alpha:.0%} threshold",
            f"  transform = {self.transform}; auxiliary lags = {self.lags}",
        ]
        if self.skipped:
            lines.append(f"  skipped (no star block or too few obs): {list(self.skipped)}")
        if self.n_tests:
            top = self.table.sort_values("p_value").head(15)
            lines.append("")
            lines.append(
                "  country      star_var       F     df    p       p_holm  "
                "p<alpha(nominal)"
            )
            for _, r in top.iterrows():
                lines.append(
                    f"  {str(r['country']):<12s} {str(r['star_variable']):<12s} "
                    f"{r['F']:7.3f} {int(r['df_num']):3d},{int(r['df_den']):<4d} "
                    f"{r['p_value']:.4f} {r['p_holm']:.4f}  "
                    f"{bool(r['reject_nominal'])}"
                )
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, *, max_bars: int = 25, figsize: tuple[float, float] | None = None):
        """Horizontal bar chart of the p-values, smallest first.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        tab = self.table.sort_values("p_value")
        shown = tab.head(max_bars)
        labels = [
            f"{r['country']}:{r['star_variable']}" for _, r in shown.iterrows()
        ]
        vals = shown["p_value"].to_numpy(dtype=float)
        if figsize is None:
            figsize = (6.0, max(2.0, 0.28 * max(len(labels), 1) + 1.0))
        fig, ax = plt.subplots(figsize=figsize)
        colors = ["firebrick" if v < self.alpha else "steelblue" for v in vals]
        ax.barh(np.arange(len(vals)), vals, color=colors)
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(self.alpha, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("p-value")
        ax.set_title("Weak exogeneity (approximate F-test)")
        if len(tab) > len(shown):
            ax.annotate(
                f"+{len(tab) - len(shown)} more",
                xy=(0.98, 0.02),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
            )
        fig.tight_layout()
        return fig


@dataclass(frozen=True, eq=False)
class GVARGIRFResult:
    """Generalised impulse responses of a solved GVAR.

    ``GIRF_h = R_h Sigma_eps d / sqrt(d' Sigma_eps d)`` with
    ``R_h = Psi_h G^-1``. The responses are *generalised*, not orthogonalised.

    Attributes
    ----------
    irf : ndarray, shape (H + 1, k)
    lower, upper, median : ndarray or None
        Pointwise bootstrap percentile band and median (``None`` when
        ``n_boot == 0``). The median differs from ``irf`` by the OLS bias.
    boot_irf : ndarray or None
        ``(n_boot_ok, H + 1, k)``; only when ``store_draws=True``.
    names, countries, variables : tuple of str
        Length ``k`` column labels and their decomposition.
    shock_label : str
    shock_vector : ndarray, shape (k,)
        The combination ``d`` on ``eps``, before normalisation.
    shock_sd : float
        ``sqrt(d' Sigma_eps d)``.
    horizon : int
    cumulative : bool
    ci : float
    n_boot, n_boot_ok, n_boot_failed : int
    bootstrap : str
        ``'iid'`` or ``'wild'``.
    seed : int or None
    """

    irf: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    median: np.ndarray | None
    boot_irf: np.ndarray | None
    names: tuple
    countries: tuple
    variables: tuple
    shock_label: str
    shock_vector: np.ndarray
    shock_sd: float
    horizon: int
    cumulative: bool
    ci: float
    n_boot: int
    n_boot_ok: int
    n_boot_failed: int
    bootstrap: str
    seed: int | None

    def to_frame(self) -> pd.DataFrame:
        """Long response table: ``h, country, variable, name, response, band``."""
        H1, k = self.irf.shape
        rows = []
        for h in range(H1):
            for j in range(k):
                rows.append(
                    {
                        "h": h,
                        "country": self.countries[j],
                        "variable": self.variables[j],
                        "name": self.names[j],
                        "response": float(self.irf[h, j]),
                        "lower": float(self.lower[h, j]) if self.lower is not None else np.nan,
                        "upper": float(self.upper[h, j]) if self.upper is not None else np.nan,
                        "median": float(self.median[h, j]) if self.median is not None else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def _peaks(self) -> pd.DataFrame:
        absirf = np.abs(self.irf)
        peak_h = absirf.argmax(axis=0)
        rows = []
        for j, nm in enumerate(self.names):
            h = int(peak_h[j])
            excl = np.nan
            if self.lower is not None:
                excl = bool(self.lower[h, j] > 0.0 or self.upper[h, j] < 0.0)
            rows.append(
                {
                    "country": self.countries[j],
                    "variable": self.variables[j],
                    "name": nm,
                    "impact": float(self.irf[0, j]),
                    "peak": float(self.irf[h, j]),
                    "peak_h": h,
                    "final": float(self.irf[-1, j]),
                    "band_excludes_zero": excl,
                }
            )
        out = pd.DataFrame(rows)
        return out.reindex(out["peak"].abs().sort_values(ascending=False).index)

    def summary(self) -> str:
        """Human-readable summary; always repeats the generalised-IRF caveat."""
        cum = " (cumulative)" if self.cumulative else ""
        lines = [
            f"Generalised impulse responses to a one-s.d. {self.shock_label} shock "
            f"(Pesaran-Shin 1998), horizon {self.horizon}{cum}",
            f"  shock size sqrt(d' Sigma_eps d) = {self.shock_sd:.6g}",
        ]
        if self.n_boot > 0:
            lines.append(
                f"  {self.ci:.0%} pointwise percentile band from {self.n_boot} "
                f"{self.bootstrap} recursive-residual bootstrap replications, "
                f"{self.n_boot_ok} usable ({self.n_boot_failed} dropped), seed = "
                f"{self.seed}"
            )
        else:
            lines.append("  point estimate only (n_boot = 0)")
        lines.append("")
        lines.append(
            "  country      variable     impact      peak  h*    final  band!=0"
        )
        for _, r in self._peaks().head(10).iterrows():
            lines.append(
                f"  {str(r['country']):<12s} {str(r['variable']):<10s} "
                f"{r['impact']:+9.4f} {r['peak']:+9.4f} {int(r['peak_h']):3d} "
                f"{r['final']:+8.4f}  {r['band_excludes_zero']}"
            )
        lines.append("")
        lines.append(
            "  These responses are generalised, not orthogonalised: each is the "
            "conditional expectation given one country-variable innovation with "
            "the others at their conditional means. They do not decompose the "
            "forecast error into independent shocks and they do not identify a "
            "structural shock."
        )
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(
        self,
        *,
        select: Sequence[Any] | None = None,
        by_variable: bool = False,
        ncols: int = 3,
        figsize: tuple[float, float] | None = None,
    ):
        """Small multiples of the responses.

        Parameters
        ----------
        select : sequence, optional
            Names (``'US:y'``) or ``(country, variable)`` pairs. Default: the
            shocked column first, then up to eight more by ``|peak|``.
        by_variable : bool
            One panel per variable with one line per country -- the standard
            GVAR figure. ``select`` is then a sequence of variable names.
        ncols : int
        figsize : tuple, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        h = np.arange(self.horizon + 1)
        if by_variable:
            allv = list(dict.fromkeys(self.variables))
            panels = [str(v) for v in (select if select is not None else allv)]
            unknown = [v for v in panels if v not in allv]
            if unknown:
                raise KeyError(
                    f"GVARGIRFResult.plot: unknown variable(s) {unknown}; the "
                    f"system carries {allv}."
                )
            n = len(panels)
            nrows = int(np.ceil(n / ncols))
            if figsize is None:
                figsize = (4.0 * min(ncols, n), 2.8 * nrows)
            fig, axes = plt.subplots(nrows, min(ncols, n), figsize=figsize, squeeze=False)
            for a, v in enumerate(panels):
                ax = axes[a // ncols, a % ncols]
                for j in range(len(self.names)):
                    if self.variables[j] != v:
                        continue
                    ax.plot(h, self.irf[:, j], linewidth=1.1, label=self.countries[j])
                ax.axhline(0.0, color="black", linewidth=0.7)
                ax.set_title(v)
                ax.set_xlabel("horizon")
                if a == 0:
                    ax.legend(fontsize=7, frameon=False, ncol=2)
            for a in range(len(panels), nrows * min(ncols, n)):
                axes[a // ncols, a % ncols].axis("off")
            fig.tight_layout()
            return fig

        name_to_j = {nm: j for j, nm in enumerate(self.names)}
        if select is None:
            order = list(self._peaks()["name"])
            shocked = [
                nm for nm, v in zip(self.names, self.shock_vector) if v != 0.0
            ]
            picks = list(dict.fromkeys(shocked + order))[:9]
        else:
            picks = []
            for item in select:
                nm = item if isinstance(item, str) else f"{item[0]}:{item[1]}"
                if nm not in name_to_j:
                    raise KeyError(
                        f"GVARGIRFResult.plot: unknown response {nm!r}; valid "
                        f"names are {list(self.names)[:20]}."
                    )
                picks.append(nm)
        n = len(picks)
        nrows = int(np.ceil(n / ncols))
        if figsize is None:
            figsize = (4.0 * min(ncols, n), 2.8 * nrows)
        fig, axes = plt.subplots(nrows, min(ncols, n), figsize=figsize, squeeze=False)
        for a, nm in enumerate(picks):
            j = name_to_j[nm]
            ax = axes[a // ncols, a % ncols]
            if self.lower is not None:
                ax.fill_between(h, self.lower[:, j], self.upper[:, j],
                                color="lightsteelblue", alpha=0.7)
            ax.plot(h, self.irf[:, j], color="darkblue", linewidth=1.3)
            ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(nm)
            ax.set_xlabel("horizon")
        for a in range(n, nrows * min(ncols, n)):
            axes[a // ncols, a % ncols].axis("off")
        fig.tight_layout()
        return fig


@dataclass(frozen=True, eq=False)
class GVARResult:
    """A fitted and solved Global VAR.

    Attributes
    ----------
    countries : tuple of str
        Country ids in global order.
    variables : tuple of str
        Union of variable names, first-appearance order.
    names : tuple of str
        The ``k`` labels ``'country:variable'`` in global order.
    name_pairs : tuple of (str, str)
    dates : pandas.Index
        The ``T_eff`` common estimation dates.
    panel : pandas.DataFrame
        The full global panel, ``(T, k)``, index = all dates, columns =
        ``names``.
    country_models : dict of str -> CountryVARX
    G : ndarray, shape (k, k)
    H : ndarray, shape (s, k, k)
    F : ndarray, shape (s, k, k)
        ``F_l = G^-1 H_l``.
    G_inv : ndarray, shape (k, k)
        ``R_0``.
    intercept, trend_coef : ndarray, shape (k,)
        ``G^-1 a_0`` and ``G^-1 a_1`` (the latter zero unless ``trend='ct'``).
    Upsilon : ndarray or None
        ``(q_d + 1, k, k_d)`` solved global-exogenous coefficients.
    Sigma_eps : ndarray, shape (k, k)
        ``E'E / (T_eff - sigma_ddof)``. Uncentred: with ``trend='n'`` there is
        no constant to remove the mean, so this is a second-moment matrix.
    resid : pandas.DataFrame, shape (T_eff, k)
    weights : pandas.DataFrame
    star_weights : dict of str -> DataFrame
        Renormalised star weights per country.
    star_data : pandas.DataFrame
    link_matrices : dict of str -> ndarray
        ``W_i``, shape ``(k_i + k*_i, k)``.
    lag_orders : dict of str -> (int, int)
    ic : str or None
    s : int
    stable : bool
    max_eigenvalue : float
    eigenvalues : ndarray
    condition_number : float
        Condition number of the **equilibrated** ``G`` -- the unit-invariant
        one. ``cond(G)`` on raw units is meaningless because ``G -> D G D^-1``
        under a rescaling.
    condition_number_raw : float
    weak_exogeneity : WeakExogeneityResult or None
    n_obs, n_countries, n_variables : int
    trend : str
    exog_names : tuple of str
    sigma_ddof : int

    The internal design object that ``girf``'s bootstrap re-estimates from is
    carried as the plain attribute ``_spec``, deliberately **not** as a
    dataclass field: it is an implementation detail and must not appear in the
    package's public-API surface. Read it through the read-only :attr:`spec`
    property if you are debugging; it carries no compatibility promise.
    """

    countries: tuple
    variables: tuple
    names: tuple
    name_pairs: tuple
    dates: pd.Index
    panel: pd.DataFrame
    country_models: dict
    G: np.ndarray
    H: np.ndarray
    F: np.ndarray
    G_inv: np.ndarray
    intercept: np.ndarray
    trend_coef: np.ndarray
    Upsilon: np.ndarray | None
    Sigma_eps: np.ndarray
    resid: pd.DataFrame
    weights: pd.DataFrame
    star_weights: dict
    star_data: pd.DataFrame
    link_matrices: dict
    lag_orders: dict
    ic: str | None
    s: int
    stable: bool
    max_eigenvalue: float
    eigenvalues: np.ndarray
    condition_number: float
    condition_number_raw: float
    weak_exogeneity: WeakExogeneityResult | None
    n_obs: int
    n_countries: int
    n_variables: int
    trend: str
    exog_names: tuple
    sigma_ddof: int

    #: Internal design bundle. A class attribute, not an annotated dataclass
    #: field, so ``dataclasses.fields(GVARResult)`` -- and hence the package's
    #: public-API snapshot -- never sees it. :func:`gvar` overwrites it per
    #: instance with ``object.__setattr__`` (the dataclass is frozen).
    #: Unannotated on purpose -- an annotation here would make it a field.
    _spec = None

    @property
    def spec(self) -> Any:
        """Internal design object. **Not part of the public API.**

        Carried so :meth:`girf`'s bootstrap can re-estimate the system at the
        fitted lag orders. A property rather than a field precisely so that it
        stays out of ``dataclasses.fields`` and the API snapshot.
        """
        return self._spec

    # -- helpers ------------------------------------------------------------
    def _index_of(self, country: str, variable: str) -> int:
        key = (str(country), str(variable))
        try:
            return self.name_pairs.index(key)
        except ValueError:
            raise KeyError(
                f"GVARResult: unknown (country, variable) {key}; valid names are "
                f"{list(self.names)[:20]}"
                + (" ..." if len(self.names) > 20 else "")
            ) from None

    def _R(self, horizon: int) -> np.ndarray:
        """``R_h = Psi_h G^-1`` for ``h = 0..horizon``."""
        Psi = _ma_coefficients([self.F[l] for l in range(self.s)], horizon)
        return Psi @ self.G_inv

    # -- estimands ----------------------------------------------------------
    def gfevd(self, horizon: int = 24, *, normalize: bool = True) -> np.ndarray:
        """Generalised forecast-error variance decomposition.

        Pesaran-Shin (1998) with ``Psi_l`` replaced by ``R_l = Psi_l G^-1`` and
        ``Sigma`` by ``Sigma_eps``.

        Parameters
        ----------
        horizon : int
            Largest horizon ``H``.
        normalize : bool
            Rescale each row to sum to one (Diebold-Yilmaz). The unnormalised
            row sums are **not** bounded below by one at any horizon, ``h = 0``
            included: the Pesaran-Shin argument needs ``Psi_0 = I`` and a GVAR
            has ``R_0 = G^-1``. At impact the row sum is a Rayleigh quotient
            lying in ``[lambda_min(corr(Sigma_eps)),
            lambda_max(corr(Sigma_eps))]``, and it recovers the ``>= 1`` bound
            only when ``G = I`` (no contemporaneous star block anywhere). See
            the module docstring, "Generalised FEVD row sums".

        Returns
        -------
        ndarray, shape (H + 1, k, k)

        Raises
        ------
        ValueError
            If ``horizon < 0``.
        numpy.linalg.LinAlgError
            If ``Sigma_eps`` has a non-positive diagonal entry.
        """
        if horizon < 0:
            raise ValueError(f"GVARResult.gfevd: horizon must be >= 0, got {horizon}.")
        return _gfevd_from_ma(
            self._R(horizon), self.Sigma_eps, normalize=normalize,
            name="GVARResult.gfevd",
        )

    def pp(self, vector: Any, *, horizon: int = 48) -> np.ndarray:
        """Persistence profile of a linear combination (Pesaran & Shin 1996).

        ``PP(b, h) = b' R_h Sigma_eps R_h' b / (b' R_0 Sigma_eps R_0' b)``,
        which is exactly 1 at ``h = 0`` by construction. It measures how the
        variance of the h-step forecast error of ``b'x_t`` compares with its
        impact variance: a profile that decays to zero marks an asymptotically
        stationary (cointegrating) combination; one that converges to a
        positive constant marks an I(1) combination.

        Parameters
        ----------
        vector : array-like of length k, or mapping (country, variable) -> float
            The combination ``b``.
        horizon : int

        Returns
        -------
        ndarray, shape (horizon + 1,)

        Raises
        ------
        ValueError
            If ``horizon < 0`` or the vector has the wrong length.
        numpy.linalg.LinAlgError
            If ``b' R_0 Sigma_eps R_0' b <= 0``.
        """
        if horizon < 0:
            raise ValueError(f"GVARResult.pp: horizon must be >= 0, got {horizon}.")
        b = self._as_vector(vector, "GVARResult.pp")
        R = self._R(horizon)
        num = np.einsum("i,hij,jk,hlk,l->h", b, R, self.Sigma_eps, R, b)
        den = float(num[0])
        if not np.isfinite(den) or den <= 0.0:
            raise np.linalg.LinAlgError(
                f"GVARResult.pp: the impact variance b' R_0 Sigma_eps R_0' b is "
                f"{den:.3e}, which is not positive, so the persistence profile is "
                "undefined. Check that Sigma_eps is positive definite and that b "
                "is not the zero vector."
            )
        return num / den

    def _as_vector(self, vector: Any, caller: str) -> np.ndarray:
        if isinstance(vector, Mapping):
            b = np.zeros(self.n_variables)
            for key, val in vector.items():
                if isinstance(key, str):
                    if key not in self.names:
                        raise KeyError(
                            f"{caller}: unknown name {key!r}; valid names are "
                            f"{list(self.names)[:20]}."
                        )
                    b[self.names.index(key)] = float(val)
                else:
                    b[self._index_of(*key)] = float(val)
            return b
        b = np.asarray(vector, dtype=float).ravel()
        if b.size != self.n_variables:
            raise ValueError(
                f"{caller}: the combination vector has {b.size} entries but the "
                f"global system has {self.n_variables} variables."
            )
        return b

    # -- GIRFs --------------------------------------------------------------
    def girf(
        self,
        country: str,
        variable: str,
        *,
        horizon: int = 24,
        n_boot: int = 0,
        ci: float = 0.90,
        bootstrap: str = "iid",
        cumulative: bool = False,
        store_draws: bool = False,
        seed: int | None = 0,
    ) -> GVARGIRFResult:
        """Generalised impulse responses to one country-variable innovation.

        ``GIRF_h = R_h Sigma_eps e_j / sqrt(Sigma_eps[j, j])``.

        Parameters
        ----------
        country, variable : str
            The shocked equation.
        horizon : int
            Largest horizon.
        n_boot : int
            Bootstrap replications; ``0`` gives point estimates only.
        ci : float
            Coverage of the pointwise percentile band, in ``(0, 1)``.
        bootstrap : {'iid', 'wild'}
            ``'iid'`` resamples whole rows of ``E`` with replacement;
            ``'wild'`` multiplies each row by an independent Rademacher draw.
        cumulative : bool
            Return cumulative responses.
        store_draws : bool
            Keep every replication in ``boot_irf``.
        seed : int or None
            RNG seed (``numpy.random.default_rng``). Default 0, so bands are
            reproducible.

        Returns
        -------
        GVARGIRFResult

        Raises
        ------
        KeyError
            Unknown ``(country, variable)``.
        ValueError
            Bad ``horizon``, ``n_boot``, ``ci`` or ``bootstrap``; or
            ``store_draws=True`` with ``n_boot == 0``.
        numpy.linalg.LinAlgError
            If the shocked equation has non-positive residual variance.
        """
        j = self._index_of(country, variable)
        d = np.zeros(self.n_variables)
        d[j] = 1.0
        return self._girf(
            d,
            label=f"{country}:{variable}",
            horizon=horizon,
            n_boot=n_boot,
            ci=ci,
            bootstrap=bootstrap,
            cumulative=cumulative,
            store_draws=store_draws,
            seed=seed,
        )

    def girf_combination(
        self,
        shock: Mapping[Any, float],
        *,
        label: str | None = None,
        horizon: int = 24,
        n_boot: int = 0,
        ci: float = 0.90,
        bootstrap: str = "iid",
        cumulative: bool = False,
        store_draws: bool = False,
        seed: int | None = 0,
    ) -> GVARGIRFResult:
        """Generalised responses to a *combination* of country innovations.

        ``GIRF_h = R_h Sigma_eps d / sqrt(d' Sigma_eps d)``. Because of the
        normalisation the path is invariant to the positive scale of ``d``:
        ``{('US', 'r'): 1.0}`` and ``{('US', 'r'): 3.0}`` give identical
        responses.

        Parameters
        ----------
        shock : mapping
            Keys are ``(country, variable)`` pairs or ``'country:variable'``
            names; values are the weights of ``d``.
        label : str, optional
            Shock label for ``summary()``; default built from ``shock``.
        horizon, n_boot, ci, bootstrap, cumulative, store_draws, seed
            As for :meth:`girf`.

        Returns
        -------
        GVARGIRFResult
        """
        d = self._as_vector(shock, "GVARResult.girf_combination")
        if label is None:
            parts = [
                f"{w:+.3g}*{nm}" for nm, w in zip(self.names, d) if w != 0.0
            ]
            label = " ".join(parts) if parts else "zero"
        return self._girf(
            d,
            label=label,
            horizon=horizon,
            n_boot=n_boot,
            ci=ci,
            bootstrap=bootstrap,
            cumulative=cumulative,
            store_draws=store_draws,
            seed=seed,
        )

    @staticmethod
    def _girf_path(R: np.ndarray, Sigma: np.ndarray, d: np.ndarray,
                   cumulative: bool, caller: str) -> tuple[np.ndarray, float]:
        var = float(d @ Sigma @ d)
        if not np.isfinite(var) or var <= 0.0:
            raise np.linalg.LinAlgError(
                f"{caller}: d' Sigma_eps d = {var:.3e} is not positive, so the "
                "one-standard-deviation normalisation is undefined. The shocked "
                "equation has (numerically) zero residual variance, or d is the "
                "zero vector. This mirrors the sigma_jj guard in "
                "puremacro.var.irf.gfevd."
            )
        sd = float(np.sqrt(var))
        path = (R @ (Sigma @ d)) / sd
        if cumulative:
            path = np.cumsum(path, axis=0)
        return path, sd

    def _girf(
        self,
        d: np.ndarray,
        *,
        label: str,
        horizon: int,
        n_boot: int,
        ci: float,
        bootstrap: str,
        cumulative: bool,
        store_draws: bool,
        seed: int | None,
    ) -> GVARGIRFResult:
        caller = "GVARResult.girf"
        if horizon < 0:
            raise ValueError(f"{caller}: horizon must be >= 0, got {horizon}.")
        if n_boot < 0:
            raise ValueError(f"{caller}: n_boot must be >= 0, got {n_boot}.")
        if not (0.0 < ci < 1.0):
            raise ValueError(f"{caller}: ci must lie in (0, 1), got {ci}.")
        if bootstrap not in ("iid", "wild"):
            raise ValueError(
                f"{caller}: bootstrap must be 'iid' or 'wild', got {bootstrap!r}."
            )
        if n_boot == 0 and (
            store_draws or bootstrap != "iid" or ci != 0.90 or seed != 0
        ):
            raise ValueError(
                f"{caller}: store_draws / bootstrap / ci / seed configure the "
                "bootstrap band, which is not being computed (n_boot=0). Set "
                "n_boot > 0 or drop them."
            )
        R = self._R(horizon)
        path, sd = self._girf_path(R, self.Sigma_eps, d, cumulative, caller)

        lower = upper = median = boot_stack = None
        n_ok = n_failed = 0
        if n_boot > 0:
            if n_boot < 50:
                warnings.warn(
                    f"{caller}: percentile bands from {n_boot} < 50 bootstrap "
                    "replications are unreliable.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            draws, n_failed, cause = self._bootstrap_girf(
                d, horizon=horizon, n_boot=n_boot, bootstrap=bootstrap,
                cumulative=cumulative, seed=seed,
            )
            n_ok = len(draws)
            if n_ok == 0:
                raise RuntimeError(
                    f"{caller}: all {n_boot} bootstrap replications failed "
                    f"(dominant cause: {cause}). The solved system is probably "
                    "explosive, or a country model is not identified on simulated "
                    "data. Report the point estimate with n_boot=0."
                )
            if n_failed > 0.10 * n_boot:
                warnings.warn(
                    f"{caller}: {n_failed} of {n_boot} bootstrap replications "
                    f"failed (dominant cause: {cause}); the band is conditioned "
                    "on the surviving draws and is biased inward.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            arr = np.stack(draws)
            alpha = (1.0 - ci) / 2.0
            lower = np.percentile(arr, 100.0 * alpha, axis=0)
            upper = np.percentile(arr, 100.0 * (1.0 - alpha), axis=0)
            median = np.percentile(arr, 50.0, axis=0)
            boot_stack = arr if store_draws else None

        return GVARGIRFResult(
            irf=path,
            lower=lower,
            upper=upper,
            median=median,
            boot_irf=boot_stack,
            names=self.names,
            countries=tuple(c for c, _ in self.name_pairs),
            variables=tuple(v for _, v in self.name_pairs),
            shock_label=label,
            shock_vector=d.copy(),
            shock_sd=sd,
            horizon=horizon,
            cumulative=cumulative,
            ci=ci,
            n_boot=n_boot,
            n_boot_ok=n_ok,
            n_boot_failed=n_failed,
            bootstrap=bootstrap,
            seed=seed,
        )

    def _bootstrap_girf(
        self,
        d: np.ndarray,
        *,
        horizon: int,
        n_boot: int,
        bootstrap: str,
        cumulative: bool,
        seed: int | None,
    ) -> tuple[list, int, str]:
        """Recursive-residual bootstrap of the whole GVAR.

        Each replication (i) draws ``eps^(b)`` by resampling whole rows of the
        stacked residual matrix ``E`` (``'iid'``) or multiplying them by
        independent Rademacher variables (``'wild'``); (ii) simulates
        ``x^(b)_t = intercept + trend_coef * t + sum_l F_l x^(b)_{t-l}
        + G^-1 eps^(b)_t + sum_j Upsilon_j d_{t-j}`` forward from the fixed
        first ``t0`` observations of the real data; (iii) rebuilds the star
        panel from ``x^(b)`` and the *same* weights; (iv) re-estimates every
        country VARX* at the **fixed** ``(p_i, q_i)`` -- lag orders are not
        re-selected; (v) re-solves the link system and recomputes the GIRF.
        A replication whose simulated path exceeds ``1e6 * (1 + max|x|)``, or
        whose re-estimation fails, is dropped and counted.
        """
        spec = self.spec
        rng = np.random.default_rng(seed)
        X = self.panel.to_numpy(dtype=float)
        E = self.resid.to_numpy(dtype=float)
        T_eff, k = E.shape
        t0, T = spec.t0, spec.T
        bound = 1e6 * (1.0 + float(np.max(np.abs(X))))
        F = self.F
        s = self.s
        draws: list = []
        causes: dict[str, int] = {}
        n_failed = 0
        for _ in range(n_boot):
            if bootstrap == "iid":
                idx = rng.integers(0, T_eff, size=T_eff)
                Eb = E[idx]
            else:
                eta = rng.integers(0, 2, size=T_eff) * 2.0 - 1.0
                Eb = E * eta[:, None]
            Xb = X.copy()
            ok = True
            for t in range(t0, T):
                val = self.intercept + self.trend_coef * (t + 1.0)
                for l in range(s):
                    val = val + F[l] @ Xb[t - l - 1]
                val = val + self.G_inv @ Eb[t - t0]
                if spec.exog is not None and self.Upsilon is not None:
                    for jj in range(self.Upsilon.shape[0]):
                        val = val + self.Upsilon[jj] @ spec.exog[t - jj]
                if not np.all(np.isfinite(val)) or np.max(np.abs(val)) > bound:
                    ok = False
                    causes["explosive simulated path"] = (
                        causes.get("explosive simulated path", 0) + 1
                    )
                    break
                Xb[t] = val
            if not ok:
                n_failed += 1
                continue
            try:
                blocks, Eb_fit = _fit_blocks(Xb, spec)
                sol = _assemble(blocks, spec)
                Sigma_b = Eb_fit.T @ Eb_fit / (T_eff - spec.sigma_ddof)
                Rb = _ma_coefficients([sol.F[l] for l in range(s)], horizon) @ sol.G_inv
                pathb, _ = self._girf_path(
                    Rb, Sigma_b, d, cumulative, "GVARResult.girf"
                )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                n_failed += 1
                key = type(exc).__name__
                causes[key] = causes.get(key, 0) + 1
                continue
            if not np.all(np.isfinite(pathb)):
                n_failed += 1
                causes["non-finite GIRF"] = causes.get("non-finite GIRF", 0) + 1
                continue
            draws.append(pathb)
        cause = max(causes, key=causes.get) if causes else "none"
        return draws, n_failed, cause

    # -- forecast -----------------------------------------------------------
    def forecast(self, steps: int, *, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        """Deterministic (conditional-mean) forecast from the solved system.

        ``x_{T+h} = intercept + trend_coef * (T + h) + sum_l F_l x_{T+h-l}
        + sum_j Upsilon_j d_{T+h-j}``, with the trend index continuing the
        1-based position in the estimation sample (``t = 1`` at the first row
        of ``panel``).

        Parameters
        ----------
        steps : int
            Forecast horizon, ``>= 1``.
        exog_future : DataFrame, optional
            Future path of the global exogenous block, ``steps`` rows and the
            same columns as the ``exog`` used at fit time. Required whenever
            ``exog`` was used.

        Returns
        -------
        pandas.DataFrame
            ``steps`` rows indexed ``1..steps`` (``name='step'``), columns
            ``names``.

        Raises
        ------
        ValueError
            If ``steps < 1``, if ``exog`` was used and ``exog_future`` is
            missing or wrongly shaped, or if ``exog_future`` is given when no
            ``exog`` was used.
        """
        if steps < 1:
            raise ValueError(f"GVARResult.forecast: steps must be >= 1, got {steps}.")
        spec = self.spec
        has_exog = spec.exog is not None
        if has_exog and exog_future is None:
            raise ValueError(
                "GVARResult.forecast: this GVAR was fitted with a global "
                "exogenous block, so a future path is required. Pass "
                f"exog_future= with {steps} row(s) and columns "
                f"{list(self.exog_names)}."
            )
        if not has_exog and exog_future is not None:
            raise ValueError(
                "GVARResult.forecast: `exog_future` was given but this GVAR was "
                "fitted without an exogenous block."
            )
        D_future = None
        if has_exog:
            if list(exog_future.columns) != list(self.exog_names):
                raise ValueError(
                    f"GVARResult.forecast: exog_future has columns "
                    f"{list(exog_future.columns)}, expected {list(self.exog_names)}."
                )
            if len(exog_future) < steps:
                raise ValueError(
                    f"GVARResult.forecast: exog_future has {len(exog_future)} rows "
                    f"but {steps} step(s) were requested."
                )
            D_future = exog_future.to_numpy(dtype=float)[:steps]

        X = self.panel.to_numpy(dtype=float)
        T = spec.T
        hist = list(X[-self.s:]) if self.s > 0 else []
        out = np.zeros((steps, self.n_variables))
        for h in range(1, steps + 1):
            val = self.intercept + self.trend_coef * (T + h)
            for l in range(self.s):
                lagpos = h - l - 1
                prev = out[lagpos - 1] if lagpos >= 1 else hist[self.s + lagpos - 1]
                val = val + self.F[l] @ prev
            if has_exog:
                for jj in range(self.Upsilon.shape[0]):
                    pos = h - jj
                    if pos >= 1:
                        dvec = D_future[pos - 1]
                    else:
                        dvec = spec.exog[T + pos - 1]
                    val = val + self.Upsilon[jj] @ dvec
            out[h - 1] = val
        return pd.DataFrame(
            out, index=pd.RangeIndex(1, steps + 1, name="step"), columns=list(self.names)
        )

    # -- presentation -------------------------------------------------------
    def coef_frame(self, country: str | None = None) -> pd.DataFrame:
        """Long coefficient table, optionally restricted to one country.

        Parameters
        ----------
        country : str, optional

        Returns
        -------
        pandas.DataFrame
            Columns ``country, equation, regressor, block, lag, coef, se, t,
            p_value, dof``.

        Raises
        ------
        KeyError
            Unknown country.
        """
        if country is not None:
            if country not in self.country_models:
                raise KeyError(
                    f"GVARResult.coef_frame: unknown country {country!r}; valid "
                    f"countries are {list(self.countries)}."
                )
            keys = [country]
        else:
            keys = list(self.countries)
        parts = []
        for c in keys:
            m = self.country_models[c]
            df = m.to_frame()
            df.insert(0, "country", c)
            df["dof"] = m.dof
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def to_frame(self) -> pd.DataFrame:
        """Long coefficient table over every country."""
        return self.coef_frame()

    def summary(self) -> str:
        """Human-readable summary of the fitted and solved GVAR."""
        first, last = self.dates[0], self.dates[-1]
        k = self.n_variables
        lines = [
            f"Global VAR (GVAR) -- {self.n_countries} countries, {k} variables, "
            f"{self.n_obs} periods ({first} .. {last})",
        ]
        ps = sorted({p for p, _ in self.lag_orders.values()})
        qs = sorted({q for _, q in self.lag_orders.values()})
        if self.ic is None:
            if len(ps) == 1 and len(qs) == 1:
                lines.append(f"  lag orders: p = {ps[0]}, q = {qs[0]} (fixed)")
            else:
                lines.append(
                    f"  lag orders (fixed): p in {ps}, q in {qs} -- per country "
                    f"{self.lag_orders}"
                )
        else:
            lines.append(
                f"  lag orders selected by {self.ic.upper()}: p in {ps}, q in {qs}"
            )
        lines.append(
            f"  trend = {self.trend!r}; global exogenous block: "
            + (", ".join(self.exog_names) if self.exog_names else "none")
        )
        if self.trend == "ct":
            lines.append(
                "  !! an unrestricted linear trend in a levels VARX* implies a "
                "quadratic trend in the levels of the solved system."
            )
        n_feedback = sum(
            1 for c in self.countries
            if np.abs(self.G[self._rows(c)] - self._sel_rows(c)).max() > 1e-12
        )
        lines.append(
            f"  link matrix G: equilibrated cond = {self.condition_number:.4g} "
            f"(raw {self.condition_number_raw:.4g}); {n_feedback} of "
            f"{self.n_countries} countries carry contemporaneous foreign feedback"
        )
        verdict = "stable" if self.stable else "NOT stable"
        note = "" if self.stable else (
            " (unit roots are expected in levels; see docs/gvar.md)"
        )
        lines.append(
            f"  solved global VAR({self.s}): max |eigenvalue| = "
            f"{self.max_eigenvalue:.4f} -> {verdict}{note}"
        )
        ratio = self.n_obs / k if k else float("inf")
        lines.append(
            f"  Sigma_eps from T_eff = {self.n_obs} observations for k = {k} "
            f"({ratio:.1f} per variable), divisor T_eff - {self.sigma_ddof}"
        )
        if self.n_obs < k:
            lines.append(
                "  !! T_eff < k: Sigma_eps is singular. GIRFs still evaluate, but "
                "the GFEVD and the bootstrap bands are unreliable."
            )
        elif self.n_obs < 2 * k:
            lines.append(
                "  !  T_eff < 2k: Sigma_eps is poorly estimated; read the GFEVD "
                "and the bands with care."
            )
        if self.weak_exogeneity is not None:
            we = self.weak_exogeneity
            lines.append(
                f"  weak exogeneity: {we.n_reject_nominal} of {we.n_tests} star "
                f"variables below the NOMINAL {we.alpha:.0%} threshold "
                f"-- {we.test_name}; the nominal level is not the measured "
                "size (0.11 at 0.05 on I(1) levels), so read the p-value "
                "ranking, not this count"
            )
        lines.append("")
        lines.append(
            "  country      k_i  k*_i   p   q  obs/par        R2 range   max|t| x*_0"
        )
        for c in self.countries:
            m = self.country_models[c]
            contemp = [
                r for r, b in enumerate(m.regressor_blocks)
                if b == "star" and m.regressor_lags[r] == 0
            ]
            if contemp:
                tmax = float(np.nanmax(np.abs(m.tstat.iloc[contemp].to_numpy())))
                tstr = f"{tmax:8.2f}"
            else:
                tstr = "       -"
            r2 = m.r2.to_numpy(dtype=float)
            lines.append(
                f"  {c:<12s} {len(m.variables):3d} {len(m.star_variables):5d} "
                f"{m.p:3d} {m.q:3d} {m.n_obs / max(m.n_params, 1):8.2f}  "
                f"[{np.nanmin(r2):+.2f}, {np.nanmax(r2):+.2f}] {tstr}"
            )
        lines.append("")
        lines.append(
            "  Coefficient t statistics are conditional on the weak exogeneity of "
            "x* and, with levels data, are not asymptotically normal (Sims, Stock "
            "& Watson 1990)."
        )
        return "\n".join(lines)

    def _rows(self, country: str) -> np.ndarray:
        return self.spec.own_idx[country]

    def _sel_rows(self, country: str) -> np.ndarray:
        return self.spec.Sel[country]

    def to_markdown(self, **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_frame`."""
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, *, figsize: tuple[float, float] = (11.0, 4.6)):
        """Companion eigenvalues and the country-block map of ``|G - I|``.

        Left: the eigenvalues of the solved companion in the complex plane with
        the unit circle drawn -- the single picture that says whether the solved
        GVAR is stationary. Right: ``|G - I|`` aggregated to country blocks,
        showing which countries carry contemporaneous foreign feedback.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ev = self.eigenvalues
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        ax1.plot(np.cos(theta), np.sin(theta), color="grey", linewidth=0.9)
        ax1.scatter(ev.real, ev.imag, s=16, color="darkblue", alpha=0.8)
        ax1.axhline(0.0, color="black", linewidth=0.6)
        ax1.axvline(0.0, color="black", linewidth=0.6)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title(f"companion eigenvalues (max |.| = {self.max_eigenvalue:.4f})")
        ax1.set_xlabel("Re")
        ax1.set_ylabel("Im")

        N = self.n_countries
        M = np.abs(self.G - np.eye(self.n_variables))
        blk = np.zeros((N, N))
        for a, ca in enumerate(self.countries):
            ra = self.spec.own_idx[ca]
            for b, cb in enumerate(self.countries):
                cbi = self.spec.own_idx[cb]
                blk[a, b] = float(M[np.ix_(ra, cbi)].max()) if ra.size and cbi.size else 0.0
        im = ax2.imshow(blk, cmap="magma_r")
        ax2.set_xticks(np.arange(N))
        ax2.set_xticklabels(self.countries, rotation=90, fontsize=8)
        ax2.set_yticks(np.arange(N))
        ax2.set_yticklabels(self.countries, fontsize=8)
        ax2.set_title("max |G - I| by country block")
        fig.colorbar(im, ax=ax2, fraction=0.046)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Estimation internals
# ---------------------------------------------------------------------------
def _design(X: np.ndarray, spec: _GVARSpec, c: str):
    """Build ``(Z, Y)`` for one country from the global panel ``X``."""
    p, q = spec.lag_orders[c]
    own = spec.own_idx[c]
    t0, T = spec.t0, spec.T
    Xi = X[:, own]
    Ws = spec.Wstar[c]
    Xsi = X @ Ws.T if Ws.shape[0] else np.zeros((T, 0))
    cols = []
    if spec.trend in ("c", "ct"):
        cols.append(np.ones((T - t0, 1)))
    if spec.trend == "ct":
        cols.append((np.arange(t0, T, dtype=float) + 1.0)[:, None])
    ci = spec.contemp_idx[c]
    if ci.size:
        cols.append(Xsi[t0:T][:, ci])
    for l in range(1, p + 1):
        cols.append(Xi[t0 - l:T - l])
    for l in range(1, q + 1):
        cols.append(Xsi[t0 - l:T - l])
    if spec.exog is not None:
        for jj in range(spec.q_d + 1):
            cols.append(spec.exog[t0 - jj:T - jj])
    Z = np.column_stack(cols) if cols else np.zeros((T - t0, 0))
    Y = Xi[t0:T]
    return Z, Y, Xsi


def _reg_layout(trend: str, star_names, contemp, p, q, own_vars, exog_names, q_d):
    """Regressor names / blocks / lags for one country."""
    names, blocks, lags = [], [], []
    if trend in ("c", "ct"):
        names.append("const")
        blocks.append("det")
        lags.append(0)
    if trend == "ct":
        names.append("trend")
        blocks.append("det")
        lags.append(0)
    for v in contemp:
        names.append(f"{v}_star")
        blocks.append("star")
        lags.append(0)
    for l in range(1, p + 1):
        for v in own_vars:
            names.append(f"{v}.L{l}")
            blocks.append("own")
            lags.append(l)
    for l in range(1, q + 1):
        for v in star_names:
            names.append(f"{v}_star.L{l}")
            blocks.append("star")
            lags.append(l)
    if exog_names:
        for jj in range(q_d + 1):
            for v in exog_names:
                names.append(f"{v}.L{jj}")
                blocks.append("exog")
                lags.append(jj)
    return tuple(names), tuple(blocks), tuple(lags)


def _fit_blocks(X: np.ndarray, spec: _GVARSpec):
    """Fit every country VARX* on the global panel ``X``; return blocks and ``E``."""
    T_eff = spec.T - spec.t0
    E = np.zeros((T_eff, spec.k))
    blocks = {}
    for c in spec.countries:
        Z, Y, _ = _design(X, spec, c)
        ki = Y.shape[1]
        m_i = Z.shape[1]
        if T_eff - m_i < ki + 1:
            raise ValueError(
                f"gvar: country {c!r} has T_eff = {T_eff} observations and "
                f"m_i = {m_i} regressors, leaving {T_eff - m_i} residual degrees "
                f"of freedom for k_i = {ki} equations. At least k_i + 1 = "
                f"{ki + 1} are required for a non-singular Sigma_i. Use fewer "
                "lags, restrict the star block, or supply a longer sample."
            )
        star_cols = [
            j for j, b in enumerate(spec.reg_blocks[c]) if b == "star"
        ]
        for j in star_cols:
            col = Z[:, j]
            if np.ptp(col) <= 1e-12 * max(1.0, float(np.abs(col).max())):
                raise ValueError(
                    f"gvar: country {c!r} has a numerically constant star "
                    f"regressor {spec.reg_names[c][j]!r}. It cannot be "
                    f"distinguished from the intercept. Pass star_vars={{{c!r}: "
                    "()}} or drop that variable from the star block."
                )
        fit = _multivariate_ols(
            Z, Y, names=spec.reg_names[c], name=f"gvar country {c!r}"
        )
        blocks[c] = {"Z": Z, "Y": Y, **fit}
        E[:, spec.own_idx[c]] = fit["resid"]
    return blocks, E


def _slice_blocks(fit: dict, spec: _GVARSpec, c: str):
    """Split a country's coefficient matrix into a0, a1, Lambda, Phi, Psi."""
    B = fit["coef"]
    p, q = spec.lag_orders[c]
    ki = len(spec.country_vars[c])
    kstar = spec.Wstar[c].shape[0]
    kd = spec.exog.shape[1] if spec.exog is not None else 0
    pos = 0
    a0 = np.zeros(ki)
    a1 = np.zeros(ki)
    if spec.trend in ("c", "ct"):
        a0 = B[pos].copy()
        pos += 1
    if spec.trend == "ct":
        a1 = B[pos].copy()
        pos += 1
    Lam0 = np.zeros((ki, kstar))
    ci = spec.contemp_idx[c]
    if ci.size:
        Lam0[:, ci] = B[pos:pos + ci.size].T
        pos += ci.size
    Phi = []
    for _ in range(p):
        Phi.append(B[pos:pos + ki].T.copy())
        pos += ki
    Lam = [Lam0]
    for _ in range(q):
        Lam.append(B[pos:pos + kstar].T.copy())
        pos += kstar
    Psi = []
    if spec.exog is not None:
        for _ in range(spec.q_d + 1):
            Psi.append(B[pos:pos + kd].T.copy())
            pos += kd
    return a0, a1, tuple(Phi), tuple(Lam), tuple(Psi)


def _assemble(blocks: dict, spec: _GVARSpec) -> GVARSolution:
    """Build the link matrices from fitted blocks and solve."""
    link, A0, A_lags, a0s, a1s, Ups = {}, {}, {}, {}, {}, {}
    for c in spec.countries:
        a0, a1, Phi, Lam, Psi = _slice_blocks(blocks[c], spec, c)
        ki = len(spec.country_vars[c])
        kstar = spec.Wstar[c].shape[0]
        link[c] = np.vstack([spec.Sel[c], spec.Wstar[c]])
        A0[c] = np.hstack([np.eye(ki), -Lam[0]])
        rows = []
        for l in range(1, spec.s + 1):
            Ph = Phi[l - 1] if l <= len(Phi) else np.zeros((ki, ki))
            La = Lam[l] if l < len(Lam) else np.zeros((ki, kstar))
            rows.append(np.hstack([Ph, La]))
        A_lags[c] = rows
        a0s[c] = a0
        a1s[c] = a1
        if spec.exog is not None:
            Ups[c] = Psi
    return solve_gvar(
        link,
        A0,
        A_lags,
        country_order=spec.countries,
        block_sizes={c: len(spec.country_vars[c]) for c in spec.countries},
        a0=a0s,
        a1=a1s,
        Upsilon=Ups if spec.exog is not None else None,
        cond_tol=spec.cond_tol,
    )


def _select_orders(X, spec_base, c, max_p, max_q, ic, exog, q_d_fixed, trend, caller):
    """Grid-search ``(p_i, q_i)`` for one country on the common grid sample."""
    own = spec_base["own_idx"][c]
    Ws = spec_base["Wstar"][c]
    T = X.shape[0]
    Xi = X[:, own]
    Xsi = X @ Ws.T if Ws.shape[0] else np.zeros((T, 0))
    ki = Xi.shape[1]
    ci = spec_base["contemp_idx"][c]
    s_grid = max(max_p, max_q, q_d_fixed)
    T_g = T - s_grid
    best = None
    best_feasible = None
    for p in range(1, max_p + 1):
        for q in range(0, max_q + 1):
            if Ws.shape[0] == 0 and q > 0:
                continue
            cols = []
            if trend in ("c", "ct"):
                cols.append(np.ones((T_g, 1)))
            if trend == "ct":
                cols.append((np.arange(s_grid, T, dtype=float) + 1.0)[:, None])
            if ci.size:
                cols.append(Xsi[s_grid:T][:, ci])
            for l in range(1, p + 1):
                cols.append(Xi[s_grid - l:T - l])
            for l in range(1, q + 1):
                cols.append(Xsi[s_grid - l:T - l])
            if exog is not None:
                for jj in range(q_d_fixed + 1):
                    cols.append(exog[s_grid - jj:T - jj])
            Z = np.column_stack(cols)
            m_i = Z.shape[1]
            if T_g - m_i < ki + 1:
                continue
            # The LARGEST feasible order, not the last one visited: the loop
            # order is (p outer, q inner), so an unconditional assignment would
            # name whatever candidate happened to come last.
            if best_feasible is None or (p, q) > best_feasible:
                best_feasible = (p, q)
            try:
                coef = np.linalg.lstsq(Z, Xi[s_grid:T], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            U = Xi[s_grid:T] - Z @ coef
            Sig = U.T @ U / T_g
            sign, logdet = np.linalg.slogdet(Sig)
            # Exact singularity only. A NUMERICALLY near-singular Sigma still
            # returns sign = +1 with a very negative logdet, which would inflate
            # the likelihood and win the grid -- but every construction that
            # produces one (a duplicated variable, or two variables driven by
            # the same innovation) also makes the country design rank deficient,
            # so `inv_xtx` in the final fit raises first, naming the collinear
            # regressor. A condition-number floor here would additionally reject
            # candidates whose residuals are merely strongly correlated, which is
            # ordinary in macro data, so the cheap test is the right one.
            if sign <= 0 or not np.isfinite(logdet):
                continue
            ll = -0.5 * T_g * (ki * np.log(2.0 * np.pi) + logdet + ki)
            n_par = ki * m_i + ki * (ki + 1) / 2.0
            crit = -2.0 * ll + (2.0 if ic == "aic" else np.log(T_g)) * n_par
            cand = (crit, p, q)
            if best is None or cand[0] < best[0] - 1e-12:
                best = cand
    if best is None:
        raise ValueError(
            f"{caller}: the whole (p, q) grid is infeasible for country {c!r} "
            f"(T = {T}, grid sample T_g = {T_g}, k_i = {ki}). Every candidate "
            f"leaves fewer than k_i + 1 residual degrees of freedom, or a "
            f"singular residual covariance."
            + (f" The largest order with enough degrees of freedom was "
               f"{best_feasible}, and even that gave a singular Sigma_i."
               if best_feasible else "")
            + " Lower max_p / max_q, or supply a longer sample."
        )
    return best[1], best[2]


def _as_country_map(value, countries, argname, default, caller, *, require_all=False):
    """Broadcast an int / mapping / None argument over the countries."""
    if value is None:
        return {c: default for c in countries}
    if isinstance(value, Mapping):
        unknown = [c for c in value if c not in countries]
        if unknown:
            raise KeyError(
                f"{caller}: `{argname}` names unknown country/countries {unknown}; "
                f"valid countries are {list(countries)}."
            )
        missing = [c for c in countries if c not in value]
        if missing and require_all:
            raise KeyError(
                f"{caller}: `{argname}` is a mapping but has no entry for "
                f"{missing}. Give every country a value or pass a single int."
            )
        return {
            c: (int(value[c]) if c in value else default) for c in countries
        }
    return {c: int(value) for c in countries}


def _weak_exogeneity_test(result_blocks, spec, X, we_lags, transform, alpha, caller):
    """Approximate residual-augmented F test for every star variable."""
    rows = []
    skipped = []
    lags_used = {}
    for c in spec.countries:
        kstar = spec.Wstar[c].shape[0]
        if kstar == 0:
            skipped.append(c)
            continue
        r = int(we_lags[c])
        lags_used[c] = r
        own = spec.own_idx[c]
        t0, T = spec.t0, spec.T
        Xi = X[t0:T, own]
        Xsi = (X @ spec.Wstar[c].T)[t0:T]
        U = result_blocks[c]["resid"]
        ki = Xi.shape[1]
        T_eff = Xi.shape[0]
        if transform == "difference":
            dXi = np.vstack([np.full((1, ki), np.nan), np.diff(Xi, axis=0)])
            dXs = np.vstack([np.full((1, kstar), np.nan), np.diff(Xsi, axis=0)])
            t_start = max(r + 1, 1)
        else:
            dXi, dXs = Xi, Xsi
            t_start = max(r, 1)
        tt = np.arange(t_start, T_eff)
        T_e = tt.size
        m = 1 + r * kstar + r * ki + ki
        if T_e - m < 1:
            skipped.append(c)
            continue
        base = [np.ones((T_e, 1))]
        for a in range(1, r + 1):
            base.append(dXs[tt - a])
            base.append(dXi[tt - a])
        base.append(U[tt - 1])
        Zu = np.column_stack(base)
        Zr = np.column_stack(base[:-1]) if len(base) > 1 else np.ones((T_e, 1))
        for b in range(kstar):
            y = dXs[tt, b]
            resu = y - Zu @ np.linalg.lstsq(Zu, y, rcond=None)[0]
            resr = y - Zr @ np.linalg.lstsq(Zr, y, rcond=None)[0]
            rss_u = float(resu @ resu)
            rss_r = float(resr @ resr)
            df_den = T_e - m
            if rss_u <= 0:
                F = np.inf
                pval = 0.0
            else:
                F = ((rss_r - rss_u) / ki) / (rss_u / df_den)
                pval = float(stats.f.sf(F, ki, df_den))
            rows.append(
                {
                    "country": c,
                    "star_variable": spec.star_names[c][b],
                    "F": float(F),
                    "df_num": int(ki),
                    "df_den": int(df_den),
                    "p_value": pval,
                    "n_obs": int(T_e),
                    "lags": r,
                }
            )
    table = pd.DataFrame(
        rows,
        columns=["country", "star_variable", "F", "df_num", "df_den", "p_value",
                 "n_obs", "lags"],
    )
    if len(table):
        table["p_holm"] = _holm(table["p_value"].to_numpy())
        table["reject_nominal"] = table["p_value"] < alpha
    else:
        table["p_holm"] = pd.Series(dtype=float)
        table["reject_nominal"] = pd.Series(dtype=bool)
    table = table[
        ["country", "star_variable", "F", "df_num", "df_den", "p_value", "p_holm",
         "reject_nominal", "n_obs", "lags"]
    ]
    return WeakExogeneityResult(
        table=table,
        transform=transform,
        lags=lags_used,
        alpha=alpha,
        n_tests=int(len(table)),
        n_reject_nominal=int(table["reject_nominal"].sum()) if len(table) else 0,
        test_name=_WE_TEST_NAME,
        skipped=tuple(dict.fromkeys(skipped)),
    )


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------
def gvar(
    data: pd.DataFrame | Mapping[str, pd.DataFrame],
    weights: Any,
    *,
    p: int | Mapping[str, int] | None = None,
    q: int | Mapping[str, int] | None = None,
    ic: str | None = None,
    max_p: int = 2,
    max_q: int = 1,
    star_vars: Mapping[str, Sequence[str]] | None = None,
    contemporaneous_star: Mapping[str, Sequence[str]] | None = None,
    exog: pd.DataFrame | None = None,
    exog_lags: int | None = None,
    trend: str = "c",
    country_order: Sequence[str] | None = None,
    country_level: str = "country",
    date_level: str = "date",
    sigma_ddof: int = 0,
    weak_exogeneity: bool = True,
    we_lags: int | Mapping[str, int] | None = None,
    we_transform: str = "difference",
    alpha: float = 0.05,
    stability_warn: bool = True,
    cond_tol: float = 1e12,
    allow_missing_variables: bool = False,
) -> GVARResult:
    """Estimate a Global VAR: country VARX* models linked by trade weights.

    Every country ``i`` is fitted by OLS over the **same** effective sample
    ``t = t0 .. T-1`` with ``t0 = max(s, q_d)`` and ``s = max_i max(p_i, q_i)``,
    so the residual vectors are aligned by date, ``Sigma_eps`` is a genuine
    contemporaneous covariance and the link identity
    ``G x_t = a_0 + a_1 t + sum_l H_l x_{t-l} + sum_l Upsilon_l d_{t-l} + eps_t``
    holds exactly at every date. The common sample is deliberately not
    configurable: one code path keeps that identity true.

    Parameters
    ----------
    data : DataFrame or mapping
        A ``(country, date)`` MultiIndexed frame with one column per variable,
        or a mapping ``country -> DataFrame`` indexed by date. **Every country
        must carry every column**: an entirely NaN ``(country, variable)``
        raises, naming the country and the variable, unless
        ``allow_missing_variables=True``. An interior NaN always raises.
    weights : DataFrame, ndarray or SpatialWeights
        Trade weights: non-negative, zero diagonal, rows summing to one. A bare
        ndarray requires ``country_order``. Time-varying weights are not
        supported.
    p, q : int or mapping, optional
        Own and foreign lag orders. Default ``p = 2``, ``q = 1`` (the canonical
        GVAR choice). Cannot be combined with ``ic``.
    ic : {'aic', 'bic'}, optional
        Select ``(p_i, q_i)`` per country on the common grid sample
        ``t = max(max_p, max_q, q_d) ..``. The criteria are comparable **within
        one grid only** -- changing ``max_p`` changes the grid sample, so a
        candidate's log-likelihood is not comparable across grids.
    max_p, max_q : int
        Grid bounds for ``ic``.
    star_vars : mapping, optional
        Star block per country. Default: country ``i``'s own variables that at
        least one other country carries. A requested variable only needs a donor
        country, so a country may carry a foreign variable it does not hold
        itself. ``()`` gives that country a plain VAR.
    contemporaneous_star : mapping, optional
        The subset of each star block entering ``Lambda_i0``. Restricting it is
        how the canonical US model excludes contemporaneous foreign ``y*``.
    exog : DataFrame, optional
        Strictly exogenous global variables ``d_t``, indexed by the data's
        dates. A column name that also appears in ``data`` raises: a variable
        endogenous in one country cannot be exogenous elsewhere.
    exog_lags : int, optional
        Common ``q_d``. Per-country ``q_d`` is out of scope. The effective
        sample starts at ``max(s, q_d)``, so a long ``exog_lags`` costs
        observations. The default depends on how the lag orders are fixed:

        * with explicit ``p`` / ``q``, it is ``max_i q_i``;
        * with ``ic=``, it is ``max_q`` -- the grid **bound**, not the selected
          orders. ``q_d`` has to be pinned before the grid search runs, because
          it is part of the common selection sample, and the ``q_i`` are not
          known yet. So an ``ic`` fit with ``exog`` can lose more observations
          than the finally selected ``q_i`` would suggest. Pass ``exog_lags``
          explicitly if that matters.
    trend : {'c', 'ct', 'n'}
        Deterministics. ``'ct'`` adds an unrestricted linear trend, which on
        I(1) levels implies a **quadratic** trend in the solved system; that is
        why VECX* GVARs restrict the trend to the cointegrating space, which
        this module does not do.
    country_order : sequence of str, optional
    country_level, date_level : str
    sigma_ddof : int
        ``Sigma_eps = E'E / (T_eff - sigma_ddof)``; default 0, the MLE divisor
        the GVAR literature uses. This is **not** cosmetic: the GIRF is
        homogeneous of degree 1/2 in ``Sigma_eps``, so the divisor scales every
        response by ``sqrt(T_eff / (T_eff - sigma_ddof))``.
    weak_exogeneity : bool
        Run the approximate weak-exogeneity test.
    we_lags : int or mapping, optional
        Auxiliary-regression lag length; default ``p_i``.
    we_transform : {'difference', 'levels'}
    alpha : float
        The **nominal** threshold behind the ``reject_nominal`` column of the
        weak-exogeneity table. It is not the test's measured size (0.109 at
        alpha = 0.05 on I(1) levels; see :class:`WeakExogeneityResult`).
        Meaningless -- and therefore an error -- with
        ``weak_exogeneity=False``.
    stability_warn : bool
        Emit a ``RuntimeWarning`` when the solved companion has a root above
        ``1 + 1e-6``. Instability is a flag, never an error: a GVAR on I(1)
        levels carries unit roots by construction.
    cond_tol : float
        Reject ``G`` when its *equilibrated* condition number exceeds this.
    allow_missing_variables : bool
        Opt out of the common-variable-set rule. With ``True`` an entirely NaN
        ``(country, variable)`` is read as "this country does not carry that
        variable", the global vector shrinks accordingly, and a
        ``RuntimeWarning`` names every dropped cell. This is how the
        oil-producer / US asymmetry is expressed. Off by default, because the
        far commoner cause of an all-NaN column is a misnamed or accidentally
        dropped series, which would otherwise estimate a different model in
        silence.

    Returns
    -------
    GVARResult

    Raises
    ------
    TypeError
        For a ``data`` that is neither a DataFrame nor a mapping, for
        time-varying weights, and (no ``**kwargs`` sink anywhere) for any
        unknown keyword.
    ValueError
        For a non-2-level index or missing level names; unbalanced or
        duplicated ``(country, date)``; an interior NaN; an entirely NaN
        ``(country, variable)`` without ``allow_missing_variables=True``; a
        country left with no usable variable; non-numeric columns; negative weights; a non-zero
        weight diagonal; rows not summing to one; an all-zero weight row for a
        country with a star block; a star variable with no donor country; a
        constant star column; unknown ``trend`` / ``ic`` / ``we_transform``;
        ``p < 1`` or ``q < 0``; both ``p``/``q`` and ``ic``; too few residual
        degrees of freedom; an infeasible lag grid; an ``exog`` index that is
        not the data's date index; an ``exog`` name that collides with an
        endogenous variable. Also -- no keyword is silently ignored -- for
        ``max_p`` / ``max_q`` without ``ic``, ``exog_lags`` without ``exog``,
        ``we_lags`` / ``we_transform`` / ``alpha`` with
        ``weak_exogeneity=False``, and ``q > 0`` for a country with an empty
        star block.
    KeyError
        Weight ids that do not match the data countries; unknown countries or
        variables in ``star_vars`` / ``contemporaneous_star`` / ``p`` / ``q`` /
        ``we_lags``.
    numpy.linalg.LinAlgError
        A rank-deficient country design (the message names the collinear
        regressors and the country) or a singular / ill-conditioned ``G``.

    Notes
    -----
    With small ``N`` the granularity condition ``max_j w_ij -> 0`` fails,
    ``x*_it`` is contemporaneously correlated with ``u_it``, and the country
    OLS is inconsistent. Two- and three-country systems -- including every
    example below -- are pedagogically useful and econometrically wrong.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.var.gvar import gvar
    >>> rng = np.random.default_rng(7)
    >>> T, dates = 120, pd.RangeIndex(120, name="date")
    >>> A = np.array([[0.5, 0.1, 0.0, 0.0], [0.0, 0.4, 0.1, 0.0],
    ...               [0.1, 0.0, 0.5, 0.1], [0.0, 0.0, 0.0, 0.4]])
    >>> X = np.zeros((T, 4))
    >>> for t in range(1, T):
    ...     X[t] = A @ X[t - 1] + rng.standard_normal(4)
    >>> frames = {"A": pd.DataFrame(X[:, :2], columns=["y", "r"], index=dates),
    ...           "B": pd.DataFrame(X[:, 2:], columns=["y", "r"], index=dates)}
    >>> W = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["A", "B"],
    ...                  columns=["A", "B"])
    >>> res = gvar(frames, W, p=1, q=1)
    >>> res.n_countries, res.n_variables, res.s
    (2, 4, 1)
    >>> g = res.girf("A", "y", horizon=8)
    >>> g.irf.shape
    (9, 4)
    >>> bool(np.isclose(res.pp({("A", "y"): 1.0}, horizon=4)[0], 1.0))
    True
    """
    caller = "gvar"
    if trend not in ("c", "ct", "n"):
        raise ValueError(
            f"{caller}: trend must be one of 'c', 'ct', 'n'; got {trend!r}."
        )
    if we_transform not in ("levels", "difference"):
        raise ValueError(
            f"{caller}: we_transform must be 'levels' or 'difference'; got "
            f"{we_transform!r}."
        )
    if ic is not None and ic not in ("aic", "bic"):
        raise ValueError(
            f"{caller}: ic must be 'aic', 'bic' or None; got {ic!r}."
        )
    if ic is not None and (p is not None or q is not None):
        raise ValueError(
            f"{caller}: pass either explicit p/q or ic=, not both."
        )
    if max_p < 1 or max_q < 0:
        raise ValueError(
            f"{caller}: max_p must be >= 1 and max_q >= 0; got max_p={max_p}, "
            f"max_q={max_q}."
        )
    if sigma_ddof < 0:
        raise ValueError(f"{caller}: sigma_ddof must be >= 0, got {sigma_ddof}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"{caller}: alpha must lie in (0, 1), got {alpha}.")
    # No silently ignored keywords: an argument that is meaningless for the
    # chosen mode is an error, not a no-op.
    if ic is None and (max_p != 2 or max_q != 1):
        raise ValueError(
            f"{caller}: max_p / max_q only bound the `ic` search grid and are "
            f"meaningless with explicit lag orders (ic is None). Pass ic='aic' "
            "or ic='bic', or drop max_p / max_q."
        )
    if exog is None and exog_lags is not None:
        raise ValueError(
            f"{caller}: exog_lags={exog_lags} was given but there is no `exog` "
            "block for it to lag. Pass exog=, or drop exog_lags."
        )
    if exog_lags is not None and exog_lags < 0:
        raise ValueError(f"{caller}: exog_lags must be >= 0, got {exog_lags}.")
    if not weak_exogeneity and (
        we_lags is not None or we_transform != "difference" or alpha != 0.05
    ):
        raise ValueError(
            f"{caller}: we_lags / we_transform / alpha configure the "
            "weak-exogeneity test, which is switched off "
            "(weak_exogeneity=False). Set weak_exogeneity=True or drop them."
        )

    frame, countries, dates, country_vars = _coerce_panel(
        data,
        country_level=country_level,
        date_level=date_level,
        country_order=country_order,
        caller=caller,
        allow_missing_variables=allow_missing_variables,
    )
    Wc = _coerce_weights(weights, countries, caller)
    star_sets = _resolve_star_sets(countries, country_vars, star_vars, caller)
    star_w = _star_weight_frames(countries, country_vars, star_sets, Wc, caller)
    names, pairs, own_idx = _global_index(countries, country_vars)
    k = len(names)
    Sel, Wstar = _selection_matrices(
        countries, country_vars, star_sets, star_w, own_idx, k
    )

    T = len(dates)
    X = np.zeros((T, k))
    for c in countries:
        sub = frame.loc[c].reindex(dates)
        X[:, own_idx[c]] = sub[list(country_vars[c])].to_numpy(dtype=float)

    # contemporaneous star selection
    contemp_idx, contemp_names = {}, {}
    if contemporaneous_star is not None:
        unknown = [c for c in contemporaneous_star if c not in countries]
        if unknown:
            raise KeyError(
                f"{caller}: contemporaneous_star names unknown country/countries "
                f"{unknown}; valid countries are {list(countries)}."
            )
    for c in countries:
        svars = star_sets[c]
        if contemporaneous_star is None or c not in contemporaneous_star:
            sel = list(svars)
        else:
            sel = list(contemporaneous_star[c])
            bad = [v for v in sel if v not in svars]
            if bad:
                raise KeyError(
                    f"{caller}: contemporaneous_star[{c!r}] lists {bad}, which is "
                    f"not in that country's star block {list(svars)}."
                )
        contemp_idx[c] = np.array([svars.index(v) for v in sel], dtype=int)
        contemp_names[c] = tuple(sel)

    # exog
    exog_arr = None
    exog_names: tuple = ()
    if exog is not None:
        if not isinstance(exog, pd.DataFrame):
            raise TypeError(
                f"{caller}: `exog` must be a pandas DataFrame indexed by date, got "
                f"{type(exog).__name__}."
            )
        clash = [str(c) for c in exog.columns if str(c) in set(frame.columns.astype(str))]
        if clash:
            raise ValueError(
                f"{caller}: exog column(s) {clash} also appear in `data`. A "
                "variable that is endogenous in one country cannot be treated as "
                "exogenous for the system: the solved global model would condition "
                "on a variable endogenous to one of its own blocks."
            )
        if not exog.index.equals(pd.Index(dates)):
            raise ValueError(
                f"{caller}: `exog` must be indexed by the data's dates "
                f"({len(dates)} rows from {dates[0]!r} to {dates[-1]!r}); got "
                f"{len(exog)} rows."
            )
        for col in exog.columns:
            if not pd.api.types.is_numeric_dtype(exog[col]):
                raise ValueError(
                    f"{caller}: exog column {str(col)!r} is non-numeric "
                    f"({exog[col].dtype})."
                )
            if exog[col].isna().any():
                raise ValueError(
                    f"{caller}: exog column {str(col)!r} contains missing values."
                )
        exog_arr = exog.to_numpy(dtype=float)
        exog_names = tuple(exog.columns)

    # lag orders
    spec_base = {"own_idx": own_idx, "Wstar": Wstar, "contemp_idx": contemp_idx}
    if ic is None:
        p_map = _as_country_map(p, countries, "p", 2, caller, require_all=True)
        q_map = _as_country_map(q, countries, "q", 1, caller, require_all=True)
        for c in countries:
            if p_map[c] < 1:
                raise ValueError(
                    f"{caller}: p must be >= 1; country {c!r} got {p_map[c]}."
                )
            if q_map[c] < 0:
                raise ValueError(
                    f"{caller}: q must be >= 0; country {c!r} got {q_map[c]}."
                )
        for c in countries:
            if Wstar[c].shape[0] == 0:
                if q is not None and q_map[c] > 0:
                    raise ValueError(
                        f"{caller}: q={q_map[c]} was requested for country {c!r}, "
                        "but that country has an empty star block (an island "
                        "row in `weights`, or an empty `star_vars` entry), so "
                        "there are no foreign variables to lag and q is "
                        "meaningless for it. "
                        f"Pass q as a mapping with q[{c!r}] = 0, or give {c!r} a "
                        "star block."
                    )
                q_map[c] = 0
        q_d = int(exog_lags) if exog_lags is not None else max(q_map.values())
    else:
        q_map_guess = {c: max_q if Wstar[c].shape[0] else 0 for c in countries}
        q_d = int(exog_lags) if exog_lags is not None else max(q_map_guess.values())
        p_map, q_map = {}, {}
        for c in countries:
            pi, qi = _select_orders(
                X, spec_base, c, max_p, max_q, ic, exog_arr, q_d, trend, caller
            )
            p_map[c] = pi
            q_map[c] = qi

    lag_orders = {c: (int(p_map[c]), int(q_map[c])) for c in countries}
    s = max(max(pi, qi) for pi, qi in lag_orders.values())
    t0 = max(s, q_d if exog_arr is not None else 0)
    if T - t0 < 2:
        raise ValueError(
            f"{caller}: the effective sample has {T - t0} observation(s) "
            f"(T = {T}, sample start {t0}). Supply a longer panel or shorten the "
            "lag orders."
        )

    reg_names, reg_blocks, reg_lags = {}, {}, {}
    for c in countries:
        pi, qi = lag_orders[c]
        n_, b_, l_ = _reg_layout(
            trend, star_sets[c], contemp_names[c], pi, qi,
            country_vars[c], exog_names, q_d,
        )
        reg_names[c], reg_blocks[c], reg_lags[c] = n_, b_, l_

    spec = _GVARSpec(
        countries=countries,
        country_vars=country_vars,
        names=names,
        name_pairs=pairs,
        own_idx=own_idx,
        Sel=Sel,
        Wstar=Wstar,
        star_names={c: tuple(star_sets[c]) for c in countries},
        contemp_idx=contemp_idx,
        lag_orders=lag_orders,
        trend=trend,
        s=s,
        t0=t0,
        T=T,
        k=k,
        exog=exog_arr,
        exog_names=exog_names,
        q_d=q_d,
        reg_names=reg_names,
        reg_blocks=reg_blocks,
        reg_lags=reg_lags,
        dates_all=pd.Index(dates),
        sigma_ddof=sigma_ddof,
        cond_tol=cond_tol,
    )

    blocks, E = _fit_blocks(X, spec)
    T_eff = T - t0
    est_dates = pd.Index(dates[t0:])

    for c in countries:
        ki = len(country_vars[c])
        dof = blocks[c]["dof"]
        if dof < ki + 5:
            warnings.warn(
                f"{caller}: country {c!r} has only {dof} residual degrees of "
                f"freedom for k_i = {ki} equations; Sigma_i and every standard "
                "error built on it are barely identified.",
                RuntimeWarning,
                stacklevel=2,
            )

    sol = _assemble(blocks, spec)
    Sigma_eps = E.T @ E / (T_eff - sigma_ddof)
    if T_eff < k:
        warnings.warn(
            f"{caller}: T_eff = {T_eff} < k = {k}, so Sigma_eps is singular. "
            "GIRFs still evaluate, but the GFEVD and the bootstrap bands are "
            "unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif T_eff < 2 * k:
        warnings.warn(
            f"{caller}: T_eff = {T_eff} < 2k = {2 * k}; Sigma_eps is poorly "
            "estimated.",
            RuntimeWarning,
            stacklevel=2,
        )

    eig = np.linalg.eigvals(companion([sol.F[l] for l in range(s)]))
    max_eig = float(np.max(np.abs(eig)))
    stable = bool(max_eig < 1.0)
    if stability_warn and max_eig > 1.0 + 1e-6:
        warnings.warn(
            f"{caller}: the solved global companion has max |eigenvalue| = "
            f"{max_eig:.4f} > 1. This is expected for a GVAR on I(1) levels "
            "(the unit roots are the cointegration the model carries), but "
            "bootstrap bands from an explosive system should not be trusted.",
            RuntimeWarning,
            stacklevel=2,
        )

    country_models = {}
    for c in countries:
        fit = blocks[c]
        a0, a1, Phi, Lam, Psi = _slice_blocks(fit, spec, c)
        ki = len(country_vars[c])
        idx = list(reg_names[c])
        cols = list(country_vars[c])
        m_i = fit["coef"].shape[0]
        n_par = ki * m_i + ki * (ki + 1) / 2.0
        ll = fit["loglik"]
        country_models[c] = CountryVARX(
            country=c,
            variables=tuple(country_vars[c]),
            star_variables=tuple(star_sets[c]),
            contemporaneous_star=contemp_names[c],
            p=lag_orders[c][0],
            q=lag_orders[c][1],
            coef=pd.DataFrame(fit["coef"], index=idx, columns=cols),
            se=pd.DataFrame(fit["se"], index=idx, columns=cols),
            tstat=pd.DataFrame(fit["tstat"], index=idx, columns=cols),
            pvalue=pd.DataFrame(fit["pvalue"], index=idx, columns=cols),
            a0=a0,
            a1=a1,
            Phi=Phi,
            Lambda=Lam,
            Psi=Psi,
            Sigma=fit["Sigma"],
            Sigma_ml=fit["Sigma_ml"],
            resid=pd.DataFrame(fit["resid"], index=est_dates, columns=cols),
            fitted=pd.DataFrame(fit["fitted"], index=est_dates, columns=cols),
            regressor_names=reg_names[c],
            regressor_blocks=reg_blocks[c],
            regressor_lags=reg_lags[c],
            n_obs=int(T_eff),
            n_params=int(m_i),
            dof=int(fit["dof"]),
            r2=pd.Series(fit["r2"], index=cols),
            loglik=float(ll),
            aic=float(-2.0 * ll + 2.0 * n_par),
            bic=float(-2.0 * ll + np.log(T_eff) * n_par),
            condition_number=float(fit["condition_number"]),
        )

    we = None
    if weak_exogeneity:
        we_map = _as_country_map(we_lags, countries, "we_lags", None, caller)
        for c in countries:
            if we_map[c] is None:
                we_map[c] = lag_orders[c][0]
            if we_map[c] < 1:
                raise ValueError(
                    f"{caller}: we_lags must be >= 1; country {c!r} got {we_map[c]}."
                )
        we = _weak_exogeneity_test(
            blocks, spec, X, we_map, we_transform, alpha, caller
        )

    star_panel = pd.DataFrame(
        np.nan,
        index=pd.MultiIndex.from_product(
            [list(countries), list(dates)], names=[country_level, date_level]
        ),
        columns=[f"{v}_star" for v in dict.fromkeys(
            v for c in countries for v in star_sets[c]
        )],
    )
    for c in countries:
        if not star_sets[c]:
            continue
        Xs = X @ Wstar[c].T
        for b, v in enumerate(star_sets[c]):
            star_panel.loc[(c, slice(None)), f"{v}_star"] = Xs[:, b]

    all_vars = tuple(dict.fromkeys(v for c in countries for v in country_vars[c]))
    result = GVARResult(
        countries=countries,
        variables=all_vars,
        names=names,
        name_pairs=pairs,
        dates=est_dates,
        panel=pd.DataFrame(X, index=pd.Index(dates), columns=list(names)),
        country_models=country_models,
        G=sol.G,
        H=sol.H,
        F=sol.F,
        G_inv=sol.G_inv,
        intercept=sol.a0 if sol.a0 is not None else np.zeros(k),
        trend_coef=sol.a1 if sol.a1 is not None else np.zeros(k),
        Upsilon=sol.Upsilon,
        Sigma_eps=Sigma_eps,
        resid=pd.DataFrame(E, index=est_dates, columns=list(names)),
        weights=Wc,
        star_weights=star_w,
        star_data=star_panel,
        link_matrices={
            c: np.vstack([Sel[c], Wstar[c]]) for c in countries
        },
        lag_orders=lag_orders,
        ic=ic,
        s=s,
        stable=stable,
        max_eigenvalue=max_eig,
        eigenvalues=eig,
        condition_number=sol.condition_number,
        condition_number_raw=sol.condition_number_raw,
        weak_exogeneity=we,
        n_obs=int(T_eff),
        n_countries=len(countries),
        n_variables=k,
        trend=trend,
        exog_names=exog_names,
        sigma_ddof=int(sigma_ddof),
    )
    # ``_spec`` is not a dataclass field (see GVARResult), so it is attached
    # here; the dataclass is frozen, hence ``object.__setattr__``.
    object.__setattr__(result, "_spec", spec)
    return result
