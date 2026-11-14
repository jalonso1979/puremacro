"""Nonlinear general equilibrium solvers for :mod:`puremacro.trade`.

Solver architectures for the ``2*ns*nc + 4*nc - 1`` equilibrium equations:
1. 'newton': Dense damped Newton-Raphson with backtracking line search
   ('condensed' is accepted as a name for it).
2. 'equilibrated_newton': Dense Newton on the Ruiz-equilibrated system.
3. 'sparse_lu': Newton with a sparse LU factorisation (scipy.sparse.linalg.splu)
   of the finite-difference Jacobian.
4. 'krylov': Inexact Newton-Krylov (matrix-free JFNK via GMRES/BiCGSTAB) with
   physical two-sided equilibration (D_L @ J @ D_R) eliminating coordinate scale disparities.
5. 'broyden', 'hybr', 'lm': quasi-Newton and SciPy MINPACK wrappers.
6. 'keller_pac': Keller's Bordered Pseudo-Arclength Continuation (PAC) in the
   tariff schedule, lambda in [0, 1] from the starting to the target schedule. It records
   steps where the oriented tangent has tau_lambda <= 0 (merged into events in
   ``metadata["fold_points"]``) but does not demonstrate that the model has a fold: no
   traversal of an economic turning point has been exhibited on a CGE calibration, and on
   the tested calibrations runs that record such events stop without converging.

Also provides:
- Hawkins-Simon Spectral Viability Filter: shifted Collatz-Wielandt power iteration
  (O(M^2) per step) that certifies rho(B_tau) < 1 with two-sided bounds.
- Ill-Conditioned Network Stabilization: SVD modal projection component clamping (|c_k| <= 20.0),
  depth-m Anderson acceleration with least-squares mixing weights, displacement wage clamping
  (|Delta omega_c| <= 0.30), and a block-elimination / 1D secant step for one stiff
  country coordinate (:func:`solve_fold_continuation_step`; its residual and convergence
  flag describe the step it returns, after any displacement clamp).

Every solve uses consistent producer/purchaser accounting
(:mod:`puremacro.trade._accounting`). Pure NumPy/SciPy, vectorised.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
import time
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Sequence
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize._numdiff import approx_derivative

from puremacro.trade.equilibrium import (
    check_accounting,
    check_policy_options,
    compute_equilibrium_residuals,
    unpack_equilibrium_vector,
)
from puremacro.trade.postprocessing import postprocess_trade_equilibrium

if TYPE_CHECKING:
    from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult


def build_initial_guess(calib: TradeCalibrationResult) -> np.ndarray:
    """Construct initial state vector x0 from calibrated model parameters.

    In the calibrated baseline:
    - Sector gross output prices p = 1.0 -> log(p) = 0.0
    - Gross output quantities y = ytot -> log(y) = log(ytot)
    - Factor rental rates r = 1.0, w = 1.0 -> log(r) = 0.0, log(w) = 0.0
    - Government transfers T = TT + TTfd (or calib.T)
    - Net foreign transfers XN = invforT[:nc-1]

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.

    Returns
    -------
    np.ndarray
        1D state vector of shape (2*ns*nc + 3*nc + nc - 1,).
    """
    nc, ns = calib.n_countries, calib.n_sectors
    log_p0 = np.zeros(ns * nc, dtype=float)
    log_y0 = np.log(np.asarray(calib.ytot, dtype=float).flatten(order="F"))
    log_r0 = np.zeros(nc, dtype=float)
    log_w0 = np.zeros(nc, dtype=float)

    if calib.T is not None:
        T0 = np.asarray(calib.T, dtype=float).flatten(order="F")
    elif calib.TT is not None and calib.TTfd is not None:
        T0 = (np.asarray(calib.TT, dtype=float) + np.asarray(calib.TTfd, dtype=float)).flatten(order="F")
    else:
        T0 = np.zeros(nc, dtype=float)

    XN0 = np.asarray(calib.invforT, dtype=float).ravel()[:nc - 1]
    return np.concatenate([log_p0, log_y0, log_r0, log_w0, T0, XN0])


def _resolve_tariffs(
    calib: TradeCalibrationResult,
    tau: np.ndarray | float | None,
    tau_fd: np.ndarray | float | None,
    tauf: np.ndarray | float | None = None,
    tauf_fd: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve and validate tariff multipliers and national tariff vectors."""
    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand

    tauf_vec = np.zeros(nc, dtype=float) if tauf is None else np.asarray(tauf, dtype=float).ravel()
    tauf_fd_vec = np.zeros(nc, dtype=float) if tauf_fd is None else np.asarray(tauf_fd, dtype=float).ravel()

    if tau is None and tauf is not None:
        tau = tauf_vec
    if tau_fd is None and tauf_fd is not None:
        tau_fd = tauf_fd_vec

    if tau is not None:
        tau_arr = np.asarray(tau, dtype=float)
        if tau_arr.ndim == 0:
            tauf_vec = np.full(nc, float(tau_arr))
            tau = tauf_vec
        else:
            tau = tau_arr

    if tau_fd is not None:
        tau_fd_arr = np.asarray(tau_fd, dtype=float)
        if tau_fd_arr.ndim == 0:
            tauf_fd_vec = np.full(nc, float(tau_fd_arr))
            tau_fd = tauf_fd_vec
        else:
            tau_fd = tau_fd_arr

    if tau is None:
        tau_a = np.ones((ns * nc, ns, nc), dtype=float)
    elif tau.shape == (ns * nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float)
    elif tau.ndim == 4 and tau.shape == (ns, nc, ns, nc):
        tau_a = np.asarray(tau, dtype=float).transpose(1, 0, 2, 3).reshape(ns * nc, ns, nc)
    elif tau.ndim == 2 and tau.shape == (ns * nc, ns * nc):
        tau_a = np.asarray(tau, dtype=float).reshape((ns * nc, ns, nc), order="F")
    elif tau.size == nc:
        tauf_vec = np.asarray(tau, dtype=float).ravel()
        tau_a = np.ones((ns * nc, ns, nc), dtype=float)
        for ik in range(nc):
            block = np.ones((ns, ns), dtype=float) * (1.0 + tauf_vec[ik])
            for orig_k in range(nc):
                if orig_k != ik:
                    tau_a[orig_k * ns : (orig_k + 1) * ns, :, ik] = block
    else:
        tau_a = np.asarray(tau, dtype=float)

    if tau_fd is None:
        taufd_a = np.ones((ns * nc, nfd, nc), dtype=float)
    elif tau_fd.shape == (ns * nc, nfd, nc):
        taufd_a = np.asarray(tau_fd, dtype=float)
    elif tau_fd.ndim == 4 and tau_fd.shape == (ns, nc, nfd, nc):
        taufd_a = np.asarray(tau_fd, dtype=float).transpose(1, 0, 2, 3).reshape(ns * nc, nfd, nc)
    elif tau_fd.ndim == 2 and tau_fd.shape == (ns * nc, nfd * nc):
        taufd_a = np.asarray(tau_fd, dtype=float).reshape((ns * nc, nfd, nc), order="F")
    elif tau_fd.size == nc:
        tauf_fd_vec = np.asarray(tau_fd, dtype=float).ravel()
        taufd_a = np.ones((ns * nc, nfd, nc), dtype=float)
        for ik in range(nc):
            block_fd = np.ones((ns, nfd), dtype=float) * (1.0 + tauf_fd_vec[ik])
            for orig_k in range(nc):
                if orig_k != ik:
                    taufd_a[orig_k * ns : (orig_k + 1) * ns, :, ik] = block_fd
    else:
        taufd_a = np.asarray(tau_fd, dtype=float)

    return tau_a, taufd_a, tauf_vec, tauf_fd_vec


# ===========================================================================
# 0. Hawkins-Simon Spectral Viability Filter
# ===========================================================================

class ViabilityResult(NamedTuple):
    """Result of the Hawkins-Simon viability check for a tariff schedule.

    A plain four-field named tuple ``(rho, cw_lower, cw_upper, is_viable)``.
    It always iterates, unpacks and converts (``tuple``, ``list``,
    ``np.asarray``) as exactly four elements; read single fields by name
    (``result.rho``). In 4.3.0 the class guessed a three- or four-element
    unpacking from the caller's bytecode; that guess was removed because the
    length of ``list(result)`` depended on the statement that followed it.
    """

    rho: float
    """Rayleigh-quotient estimate of the dominant eigenvalue rho(B_tau)."""
    cw_lower: float
    """Collatz-Wielandt lower bound min_i (B_tau x)_i / x_i <= rho(B_tau)."""
    cw_upper: float
    """Collatz-Wielandt upper bound max_i (B_tau x)_i / x_i >= rho(B_tau)."""
    is_viable: bool
    """True when the upper bound certifies rho(B_tau) < 1 - 1e-6."""


_VIABILITY_THRESHOLD = 1.0 - 1e-6


def _viability_cost_matrix(
    calib: TradeCalibrationResult,
    tau: np.ndarray | float | None,
    *,
    exact_tax: bool = False,
) -> np.ndarray:
    """Tariff-augmented cost matrix B_tau = a * multiplier / (1 - t_j).

    A Python/NumPy scalar (any 0-d input) is a uniform import tariff RATE
    (multiplier ``1 + rate`` on every cross-border flow, one on domestic flows).
    Every array input, including lists and other array-likes, is interpreted
    exactly as :func:`solve_trade_equilibrium` interprets it, through
    :func:`_resolve_tariffs`: ``(nc,)`` vectors are national import RATES, while
    ``(M, M)``, ``(M, ns, nc)`` and ``(ns, nc, ns, nc)`` arrays are MULTIPLIERS
    (``1 + rate``) used as given.

    ``exact_tax=False`` clips production tax rates to ``[-0.9, 0.999]`` (the
    4.3.0 rule of the public check). ``exact_tax=True`` uses the model's exact
    ``1 - t_j`` (the model prices output as cost / (1 - t_j) and rejects
    ``t_j >= 1``), so the bounds describe the price system the solver
    actually solves; the policy pre-screen uses it.
    """
    nc, ns = calib.n_countries, calib.n_sectors
    M = ns * nc
    tax_flat = np.asarray(calib.tax, dtype=float).flatten(order="F")
    if not exact_tax:
        tax_flat = np.clip(tax_flat, -0.9, 0.999)
    one_minus_t = np.maximum(1.0 - tax_flat, 1e-12)
    if hasattr(calib.a, "ndim") and calib.a.ndim == 3:
        a_2d = calib.a.reshape((M, M), order="F")
    else:
        a_2d = np.asarray(calib.a, dtype=float)

    if tau is None:
        tau_mult = np.ones((M, M), dtype=float)
    elif isinstance(tau, (int, float, np.floating, np.integer)) and not isinstance(tau, bool):
        tau_a, _, _, _ = _resolve_tariffs(calib, tau=float(tau), tau_fd=None)
        tau_mult = tau_a.reshape((M, M), order="F")
    else:
        tau_arr = np.asarray(tau, dtype=float)
        tau_a, _, _, _ = _resolve_tariffs(calib, tau=tau_arr, tau_fd=None)
        if tau_a.shape != (M, ns, nc):
            raise ValueError(
                f"Tariff array of shape {tau_arr.shape} is not a supported schedule: expected "
                f"(nc,)={(nc,)} rates or multipliers of shape {(M, M)}, {(M, ns, nc)} or {(ns, nc, ns, nc)}"
            )
        tau_mult = tau_a.reshape((M, M), order="F")

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        B_tau = (a_2d * tau_mult) / one_minus_t[None, :]
        B_tau = np.nan_to_num(B_tau, nan=0.0, posinf=1e12, neginf=0.0)
        B_tau = np.maximum(B_tau, 0.0)
    return B_tau


def _shifted_cw_bounds(
    B: np.ndarray, shift: float, max_iter: int, tol: float
) -> tuple[float, float, float, bool]:
    """Collatz-Wielandt bounds from power iteration on ``B + shift * I``.

    Local fallback used only while :func:`puremacro.trade.regularize.compute_spectral_radius`
    does not accept a ``shift`` keyword. For any nonnegative ``B`` and positive
    ``x``, ``min_i (Bx)_i/x_i <= rho(B) <= max_i (Bx)_i/x_i``, so every iterate
    yields valid bounds; the small shift only removes periodicity. Returns
    ``(rho, lower, upper, resolved)``.
    """
    n = B.shape[0]
    if n == 0 or not np.any(B):
        return 0.0, 0.0, 0.0, True
    x = np.ones(n)
    lower, upper = 0.0, float(np.max(np.sum(B, axis=1)))
    resolved = False
    for _ in range(max_iter):
        Bx = B @ x
        ratios = Bx / x
        lower = max(lower, float(np.min(ratios)))
        upper = min(upper, float(np.max(ratios)))
        if upper - lower <= tol * max(1.0, upper):
            resolved = True
            break
        nxt = Bx + shift * x
        x = np.maximum(nxt / np.max(nxt), np.finfo(float).tiny)
    rho = float(np.dot(x, B @ x) / np.dot(x, x))
    return min(max(rho, lower), upper), lower, upper, resolved


def _viability_bounds(
    B: np.ndarray, eps: float, max_iter: int, tol: float
) -> tuple[float, float, float]:
    """Spectral bounds of ``B`` with the shift ``eps`` (see check_hawkins_simon_viability)."""
    from puremacro.trade.regularize import compute_spectral_radius

    try:
        accepts_shift = "shift" in inspect.signature(compute_spectral_radius).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        accepts_shift = False
    if accepts_shift:
        rho, lower, upper = compute_spectral_radius(B, max_iter=max_iter, tol=tol, shift=eps)
        return float(rho), float(lower), float(upper)
    # Thin local shifted iteration until regularize.compute_spectral_radius
    # grows the ``shift`` keyword; unresolved straddles fall through to the full
    # routine (strongly connected blocks, eigenvalues and resolvent test vector)
    # and the two sets of valid bounds are intersected.
    rho, lower, upper, resolved = _shifted_cw_bounds(B, eps, max_iter, tol)
    if not resolved and lower < _VIABILITY_THRESHOLD <= upper:
        rho2, lower2, upper2 = compute_spectral_radius(B, max_iter=max_iter, tol=tol)
        rho, lower, upper = float(rho2), max(lower, float(lower2)), min(upper, float(upper2))
    return rho, lower, upper


def check_hawkins_simon_viability(
    calib: TradeCalibrationResult,
    tau: np.ndarray | float | None = None,
    max_iter: int = 50,
    eps: float = 1e-3,
    tol: float = 1e-12,
) -> ViabilityResult:
    """Check the Hawkins-Simon viability condition for a tariff schedule.

    Bounds the spectral radius rho(B_tau) of the tariff-augmented input-output
    cost matrix B_{tau, ij} = a_{ij} * m_{ij} / (1 - t_j), where m is the tariff
    multiplier, with a shifted Collatz-Wielandt power iteration on
    ``B_tau + eps * I``. For any positive vector x the ratios (B_tau x)_i / x_i
    bracket rho(B_tau), so the returned bounds are certificates, not estimates.
    With Leontief intermediates (``sigma=0``) a positive price system exists
    only if rho(B_tau) < 1.

    Cost: one dense matrix-vector product per iteration (O(M^2)); bounds that
    still straddle ``1 - 1e-6`` after ``max_iter`` steps are refined by a second
    pass of ``max(1000, max_iter)`` iterations. No wall-time bound is
    guaranteed; it depends on the matrix, the machine and
    :func:`puremacro.trade.regularize.compute_spectral_radius`.

    Production tax rates enter as ``1 / (1 - t_j)`` with ``t_j`` clipped to
    ``[-0.9, 0.999]`` (the 4.3.0 rule of this public check). The pre-screen of
    :func:`puremacro.trade.policy_solver.solve_policy_equilibrium` uses the
    model's exact ``1 - t_j`` instead.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau : np.ndarray, float, or None, default None
        Tariff schedule, with the conventions of :func:`solve_trade_equilibrium`:

        - None: baseline (all multipliers one).
        - float (any 0-d input: Python or NumPy scalar, 0-d array): uniform
          import tariff RATE on every cross-border flow (multiplier
          ``1 + rate``; domestic flows keep multiplier one).
          ``solve_trade_equilibrium(tau=rate)`` resolves a scalar the same way.
        - ``(nc,)`` array: national import tariff RATES by importing country.
          Lists and other array-likes are converted with ``np.asarray`` and read
          like arrays (4.3.0 silently treated any non-``ndarray`` input as the
          baseline).
        - ``(M, M)``, ``(M, ns, nc)`` or ``(ns, nc, ns, nc)`` array: tariff
          MULTIPLIERS (``1 + rate``), used as given, exactly as the solver's
          ``_resolve_tariffs`` interprets them. A zero-diagonal rate matrix is
          therefore read as multipliers (domestic multiplier zero), as the
          solver would read it; pass ``1 + rates``.
    max_iter : int, default 50
        Power-iteration steps in the first pass.
    eps : float, default 1e-3
        Positive diagonal shift of the power iteration, ``B_tau + eps * I``. It removes
        periodicity without changing eigenvectors or the bounds' validity; a
        small shift keeps the convergence ratio close to |lambda_2|/rho. It is
        forwarded as ``shift=eps`` to
        :func:`puremacro.trade.regularize.compute_spectral_radius` when that
        function accepts the keyword, and otherwise used by a local shifted
        iteration.
    tol : float, default 1e-12
        Relative gap between the Collatz-Wielandt bounds that stops the iteration.

    Returns
    -------
    ViabilityResult
        Named tuple ``(rho, cw_lower, cw_upper, is_viable)``. The function
        raises instead of returning a non-viable result, so ``is_viable`` is
        ``True`` on every return.

    Raises
    ------
    ValueError
        If ``tau`` contains NaN, has an unsupported shape, or if the upper bound
        does not certify rho(B_tau) < 1 - 1e-6. The message distinguishes a
        certified violation (lower bound >= 1 - 1e-6) from an unresolved
        certificate (bounds straddle the threshold).
    """
    rho, cw_lower, cw_upper, status = _hawkins_simon_certificate(
        calib, tau, max_iter=max_iter, eps=eps, tol=tol)
    if status != "viable":
        if status == "violated":
            reason = "violates Hawkins-Simon viability condition"
        else:
            reason = "has unresolved Hawkins-Simon viability"
        raise ValueError(f"Tariff schedule {reason}: spectral bounds [{cw_lower:.8g}, {cw_upper:.8g}]")
    return ViabilityResult(float(rho), float(cw_lower), float(cw_upper), True)


def _hawkins_simon_certificate(
    calib: TradeCalibrationResult,
    tau: np.ndarray | float | None,
    *,
    max_iter: int = 50,
    eps: float = 1e-3,
    tol: float = 1e-12,
    exact_tax: bool = False,
) -> tuple[float, float, float, str]:
    """Non-raising core of :func:`check_hawkins_simon_viability`.

    Returns ``(rho, cw_lower, cw_upper, status)`` with ``status`` one of
    ``"viable"`` (``cw_upper < 1 - 1e-6``), ``"violated"`` (``cw_lower >= 1 - 1e-6``)
    or ``"unresolved"`` (the bounds straddle the threshold after the refinement
    pass). ``"violated"`` includes the band ``[1 - 1e-6, 1)`` in which a positive
    price system still exists; callers that need a certificate of non-existence
    must test ``cw_lower >= 1`` themselves (the policy pre-screen does).
    ``exact_tax`` is passed to :func:`_viability_cost_matrix`.
    """
    if tau is not None and np.any(np.isnan(np.asarray(tau, dtype=float))):
        raise ValueError("Tariff vector contains NaN values")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    B_tau = _viability_cost_matrix(calib, tau, exact_tax=exact_tax)
    rho, cw_lower, cw_upper = _viability_bounds(B_tau, eps, max_iter, tol)
    threshold = _VIABILITY_THRESHOLD
    if cw_lower < threshold <= cw_upper:
        rho, cw_lower, cw_upper = _viability_bounds(B_tau, eps, max(1000, max_iter), tol)
    if cw_upper < threshold:
        status = "viable"
    elif cw_lower >= threshold:
        status = "violated"
    else:
        status = "unresolved"
    return float(rho), float(cw_lower), float(cw_upper), status


# ===========================================================================
# 1. Architecture A: Dense Damped Newton-Raphson (Baseline)
# ===========================================================================

def _fd_steps(x: np.ndarray, eps: float, scale: np.ndarray) -> np.ndarray:
    """Per-unknown forward-difference steps ``h_j = eps * max(|x_j|, s_j)``.

    ``s_j`` is the natural scale of unknown ``j`` (one for the log price,
    output and factor-price blocks; a monetary scale of the calibration for the
    value-unit transfer ``T`` and foreign-balance ``XN`` blocks, see
    :func:`_consistent_fd_scale`). The returned steps are the exactly
    representable increments ``(x_j + h_j) - x_j``, so dividing by them does not
    add rounding error. An absolute step on a value-unit unknown is lost to
    rounding once ``|x_j|`` is large (``ulp(2.2e10) = 3.8e-6``), which made the
    transfer and foreign-balance Jacobian columns meaningless at native scale.
    """
    x = np.asarray(x, dtype=float)
    h = eps * np.maximum(np.abs(x), np.asarray(scale, dtype=float))
    return (x + h) - x


def _consistent_fd_scale(calib: TradeCalibrationResult, n: int) -> np.ndarray:
    """Natural scales ``s_j`` of the equilibrium unknowns for :func:`_fd_steps`.

    One for ``log p``, ``log y``, ``log r`` and ``log w``; for ``T`` and ``XN``
    the median absolute baseline factor income ``median_c |L_c + K_c|``
    (calibration value units, 1 if that is zero or not finite).
    """
    nc, ns = calib.n_countries, calib.n_sectors
    M = ns * nc
    scale = np.ones(n, dtype=float)
    income = np.abs(np.asarray(calib.l_endow, dtype=float).ravel()
                    + np.asarray(calib.k_endow, dtype=float).ravel())
    money = float(np.median(income)) if income.size else 1.0
    if not np.isfinite(money) or money <= 0.0:
        money = 1.0
    scale[2 * M + 2 * nc:] = money
    return scale


def _fd_jacobian_dense(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    f_val: np.ndarray,
    eps_fd: float,
    fd_scale: np.ndarray | None,
) -> np.ndarray:
    """Dense forward-difference Jacobian.

    ``fd_scale=None`` uses the absolute step ``eps_fd`` on every unknown;
    otherwise the scaled steps of :func:`_fd_steps` are used.
    """
    n = len(x)
    J = np.empty((n, n), dtype=float)
    if fd_scale is None:
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps_fd
            J[:, j] = (f(x_pert) - f_val) / eps_fd
        return J
    h = _fd_steps(x, eps_fd, fd_scale)
    for j in range(n):
        x_pert = x.copy()
        x_pert[j] += h[j]
        J[:, j] = (f(x_pert) - f_val) / h[j]
    return J


def _consistent_acceptor(
    calib: TradeCalibrationResult,
    tau_a: np.ndarray,
    taufd_a: np.ndarray,
    tauf_vec: np.ndarray | None,
    tauf_fd_vec: np.ndarray | None,
    tol: float,
    *,
    sigma: float = 0.0,
    foreign_saving_units: str = "numeraire",
) -> Callable[[np.ndarray], bool]:
    """Stopping test identical to the postprocessing acceptance audit.

    ``accept(x)`` is true when the solved residuals AND the physical residuals
    (all goods equations, including the one dropped for Walras' law, and the
    realized net exports minus foreign saving) are finite and at most ``tol`` in
    absolute value: the residual set that
    :func:`puremacro.trade.postprocessing.postprocess_trade_equilibrium` audits
    before reporting ``converged``. (Demand feasibility is a property of the
    root, not of the iterate's precision; postprocessing still checks it.)
    """
    from puremacro.trade._accounting import evaluate

    def accept(x: np.ndarray) -> bool:
        block = evaluate(
            x, calib, tau_a, taufd_a, tauf_vec, tauf_fd_vec, sigma=sigma,
            foreign_saving_units=foreign_saving_units,
        )
        audit = np.concatenate([np.ravel(block["residuals"]), np.ravel(block["physical_residuals"])])
        return bool(np.isfinite(audit).all() and float(np.max(np.abs(audit))) <= tol)

    return accept


def _record(info: dict[str, Any] | None, key: str, value: Any) -> None:
    if info is not None:
        info[key] = value


def _damped_newton_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-2,
    *,
    fd_scale: np.ndarray | None = None,
    accept: Callable[[np.ndarray], bool] | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Pure NumPy Damped Newton-Raphson solver with forward Jacobian and line search.

    Stops when ``max|f| <= tol`` (or ``sum|f| <= tol``) and, when ``accept`` is
    given, ``accept(x)`` is also true; the solvers pass the full
    acceptance audit there (the solved residuals plus the omitted goods equation
    and the realized foreign balances), so the solver never stops at a state
    that postprocessing then rejects at the same tolerance. ``fd_scale`` selects
    scaled finite-difference steps (see :func:`_fd_steps`).

    Non-finite residuals or Jacobians and a failed least-squares fallback end
    the iteration with ``converged=False`` at the last finite iterate instead of
    raising; ``info`` (when given) receives ``residual_history`` (max residual
    per accepted iterate) and ``termination``.
    """
    x = np.asarray(x_init, dtype=float).copy()
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))
    history = [max_res]
    _record(info, "residual_history", history)

    def done(xx: np.ndarray, mr: float, df: float) -> bool:
        return (mr <= tol or df <= tol) and (accept is None or bool(accept(xx)))

    if not np.isfinite(f_val).all():
        _record(info, "termination", "non_finite_residual")
        return x, False, 0, max_res, diff, f_val
    if done(x, max_res, diff):
        _record(info, "termination", "tolerance")
        return x, True, 0, max_res, diff, f_val

    for it in range(max_iter):
        # Numerical forward finite-difference Jacobian
        J = _fd_jacobian_dense(f, x, f_val, eps_fd, fd_scale)
        if not np.isfinite(J).all():
            _record(info, "termination", "non_finite_jacobian")
            return x, False, it, max_res, diff, f_val

        # Newton direction
        try:
            delta = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            try:
                delta = np.linalg.lstsq(J, -f_val, rcond=1e-10)[0]
            except (np.linalg.LinAlgError, ValueError):
                _record(info, "termination", "linear_solve_failed")
                return x, False, it, max_res, diff, f_val

        # Half steps for the first three iterations, full steps afterwards
        xscal = 0.5 if it < 3 else 1.0

        # Backtracking line search
        alpha_step = xscal
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        if not (np.isfinite(x_trial).all() and np.isfinite(f_trial).all()):
            # Never step onto a non-finite state: its Jacobian is undefined.
            _record(info, "termination", "non_finite_residual")
            return x, False, it + 1, max_res, diff, f_val

        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))
        history.append(max_res)

        if done(x, max_res, diff):
            _record(info, "termination", "tolerance")
            return x, True, it + 1, max_res, diff, f_val

    _record(info, "termination", "max_iter")
    return x, False, max_iter, max_res, diff, f_val


def _ruiz_equilibrate(
    J: np.ndarray, sweeps: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided Ruiz scaling ``A = diag(dr) @ J @ diag(dc)``.

    Each sweep divides every row and every column by the square root of its
    largest absolute entry, so after a few sweeps all rows and columns of ``A``
    have infinity norm close to one. Rows or columns that are entirely zero keep
    a unit scale. Returns ``(A, dr, dc)``.
    """
    A = np.array(J, dtype=float, copy=True)
    n_rows, n_cols = A.shape
    dr = np.ones(n_rows)
    dc = np.ones(n_cols)
    for _ in range(sweeps):
        r = np.sqrt(np.abs(A).max(axis=1))
        r[~(r > 0.0)] = 1.0
        A /= r[:, None]
        dr /= r
        c = np.sqrt(np.abs(A).max(axis=0))
        c[~(c > 0.0)] = 1.0
        A /= c[None, :]
        dc /= c
    return A, dr, dc


def _equilibrated_newton_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-6,
    *,
    fd_scale: np.ndarray | None = None,
    accept: Callable[[np.ndarray], bool] | None = None,
    info: dict[str, Any] | None = None,
    max_backtracks: int = 30,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Newton's method on the row/column-equilibrated system, with backtracking.

    The equilibrium unknowns mix log prices with levels in value units, and the
    equations mix both, so on a large table the raw Jacobian can span many
    orders of magnitude although it is regular. Each iteration builds a
    forward-difference Jacobian with relative steps ``h_j = eps_fd *
    max(|x_j|, s_j)`` (``s_j = fd_scale[j]``, one by default), scales it with
    :func:`_ruiz_equilibrate` (which brings every row and column to unit
    infinity norm), solves
    the scaled Newton system and backtracks (Armijo, at most ``max_backtracks``
    halvings) on the 2-norm of the row-scaled residual ``dr * f``, which weighs
    every equation by its own sensitivity instead of letting the value-unit
    equations dominate.

    Stops as :func:`_damped_newton_solve` does: ``max|f| <= tol`` or ``sum|f|
    <= tol``, and ``accept(x)`` when given. When no step decreases the scaled
    residual the solve ends with termination ``"no_descent"``; then, and at
    ``"max_iter"``, ``info`` also receives ``equilibrated_condition`` (largest
    over smallest singular value of the last equilibrated Jacobian) and
    ``weakest_direction`` (the unit right singular vector of its smallest
    singular value, in the original unknowns), which locate a near-singular
    Jacobian.
    """
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    scale = np.ones(n) if fd_scale is None else np.asarray(fd_scale, dtype=float)
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))
    history = [max_res]
    _record(info, "residual_history", history)

    def done(xx: np.ndarray, mr: float, df: float) -> bool:
        return (mr <= tol or df <= tol) and (accept is None or bool(accept(xx)))

    def diagnose(A: np.ndarray, dc: np.ndarray) -> None:
        if info is None:
            return
        try:
            _, s, vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return
        v = dc * vt[-1]
        norm = float(np.linalg.norm(v))
        info["equilibrated_condition"] = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        info["weakest_direction"] = v / norm if norm > 0 else v

    if not np.isfinite(f_val).all():
        _record(info, "termination", "non_finite_residual")
        return x, False, 0, max_res, diff, f_val
    if done(x, max_res, diff):
        _record(info, "termination", "tolerance")
        return x, True, 0, max_res, diff, f_val

    A = None
    dc = np.ones(n)
    for it in range(max_iter):
        J = _fd_jacobian_dense(f, x, f_val, eps_fd, scale)
        if not np.isfinite(J).all():
            _record(info, "termination", "non_finite_jacobian")
            return x, False, it, max_res, diff, f_val
        A, dr, dc = _ruiz_equilibrate(J)
        g = dr * f_val
        try:
            dz = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            try:
                dz = np.linalg.lstsq(A, -g, rcond=1e-12)[0]
            except (np.linalg.LinAlgError, ValueError):
                _record(info, "termination", "linear_solve_failed")
                return x, False, it, max_res, diff, f_val
        dx = dc * dz

        g_norm = float(np.linalg.norm(g))
        lam = 1.0
        accepted = False
        for _ in range(max_backtracks):
            x_trial = x + lam * dx
            f_trial = f(x_trial)
            if (np.isfinite(x_trial).all() and np.isfinite(f_trial).all()
                    and float(np.linalg.norm(dr * f_trial)) < (1.0 - 1e-4 * lam) * g_norm):
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            diagnose(A, dc)
            _record(info, "termination", "no_descent")
            return x, False, it, max_res, diff, f_val

        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))
        history.append(max_res)
        if done(x, max_res, diff):
            _record(info, "termination", "tolerance")
            return x, True, it + 1, max_res, diff, f_val

    if A is not None:
        diagnose(A, dc)
    _record(info, "termination", "max_iter")
    return x, False, max_iter, max_res, diff, f_val


def _near_singular_metadata(
    calib: TradeCalibrationResult, info: dict[str, Any]
) -> dict[str, Any]:
    """Summarise ``info["weakest_direction"]`` by country for result metadata.

    The squared entries of the unit direction are summed by the country each
    unknown belongs to (sector prices and outputs, ``r``, ``w``, ``T`` and
    ``XN``); the country with the largest share is reported with that share.
    Returns an empty dict when the solver recorded no diagnostic.
    """
    v = info.get("weakest_direction")
    if v is None:
        return {}
    ns, nc = calib.n_sectors, calib.n_countries
    v = np.asarray(v, dtype=float).ravel()
    owner = np.concatenate([
        np.tile(np.repeat(np.arange(nc), ns), 2),
        np.tile(np.arange(nc), 3),
        np.arange(nc - 1),
    ])
    out: dict[str, Any] = {"equilibrated_condition": float(info.get("equilibrated_condition", np.nan))}
    if owner.size != v.size:
        return out
    weight = np.bincount(owner, weights=v ** 2, minlength=nc)
    j = int(np.argmax(weight))
    codes = list(getattr(calib, "country_codes", ()) or ())
    out["near_singular_country"] = codes[j] if j < len(codes) else j
    out["near_singular_country_weight"] = float(weight[j] / max(float(weight.sum()), 1e-300))
    return out


# ===========================================================================
# 2. Architecture B: Sparse Jacobian Newton via splu and Graph Coloring
# ===========================================================================

def _build_cge_sparsity_pattern(
    calib: TradeCalibrationResult,
    tau_a: np.ndarray | None = None,
    sigma_y: float = 0.0,
    variable_markups: bool = False,
    config: Any | None = None,
    full_block: bool = False,
    **kwargs: Any,
) -> sp.csc_matrix:
    """Construct structural Jacobian sparsity pattern for the CGE equilibrium system."""
    if config is not None:
        tech = getattr(config, "technology", None)
        if tech is not None:
            sigma_y = max(float(sigma_y), float(getattr(tech, "sigma_y", 0.0)))
        mkt = getattr(config, "market_structure", None)
        if mkt is not None:
            variable_markups = variable_markups or bool(getattr(mkt, "variable_markups", False))

    tech_arg = kwargs.get("technology") or kwargs.get("tech_cfg")
    if tech_arg is not None:
        sigma_y = max(float(sigma_y), float(getattr(tech_arg, "sigma_y", 0.0)))

    mkt_arg = kwargs.get("market_structure") or kwargs.get("market_cfg")
    if mkt_arg is not None:
        variable_markups = variable_markups or bool(getattr(mkt_arg, "variable_markups", False))

    if "sigma_y" in kwargs:
        sigma_y = max(float(sigma_y), float(kwargs["sigma_y"]))
    if "variable_markups" in kwargs:
        variable_markups = variable_markups or bool(kwargs["variable_markups"])
    if "full_block" in kwargs:
        full_block = full_block or bool(kwargs["full_block"])

    ns, nc = calib.n_sectors, calib.n_countries
    M = ns * nc
    n = 2 * M + 4 * nc - 1
    S = sp.lil_matrix((n, n), dtype=bool)

    # Intermediate linkage matrix
    if tau_a is not None:
        a_eff = (calib.a * tau_a).reshape((M, M), order="F") != 0
    else:
        a_eff = calib.a.reshape((M, M), order="F") != 0

    # Block ff0: Goods market clearing
    # ff0 w.r.t p: country-level final demand price linkages
    S[:M, :M] = True
    # ff0 w.r.t y: (I - A) structure
    S[:M, M:2 * M] = sp.eye(M, dtype=bool) + sp.csc_matrix(a_eff)
    # ff0 w.r.t macro: factor returns and transfers enter through consumer budget
    S[:M, 2 * M:] = True

    # Block ff1: Zero-profit condition
    # ff1 w.r.t p: (I - B^T)
    S[M:2 * M, :M] = sp.eye(M, dtype=bool) + sp.csc_matrix(a_eff.T)

    # Dynamic activation of ff1 w.r.t y (cross-derivative block S[M:2M, M:2M])
    # Activated when outer CES nest is non-Leontief (sigma_y >= 1e-6) or variable markups are active
    if (float(sigma_y) >= 1e-6) or bool(variable_markups):
        if full_block:
            S[M:2 * M, M:2 * M] = True
        else:
            S[M:2 * M, M:2 * M] = sp.eye(M, dtype=bool)

    # ff1 w.r.t r, w: only for national factor returns of country i
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[s_start:s_end, 2 * M + i] = True
        S[s_start:s_end, 2 * M + nc + i] = True

    # Block ff2: Labor market clearing
    # ff2 w.r.t y: only sectors in country i
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[2 * M + i, s_start:s_end] = True
        S[2 * M + i, 2 * M + i] = True
        S[2 * M + i, 2 * M + nc + i] = True

    # Block ff3: Capital market clearing
    # ff3 w.r.t y: only sectors in country i
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[2 * M + nc + i, s_start:s_end] = True
        S[2 * M + nc + i, 2 * M + i] = True
        S[2 * M + nc + i, 2 * M + nc + i] = True

    # Block ff4: Trade balance (current accounts)
    S[2 * M + 2 * nc : 2 * M + 3 * nc - 1, :] = True

    # Block ff5: Fiscal budget consistency
    for i in range(nc):
        s_start, s_end = M + i * ns, M + (i + 1) * ns
        S[2 * M + 3 * nc - 1 + i, s_start:s_end] = True
        S[2 * M + 3 * nc - 1 + i, 2 * M + 2 * nc + i] = True
    S[2 * M + 3 * nc - 1 :, :M] = True
    S[2 * M + 3 * nc - 1 :, 2 * M + 3 * nc - 1 :] = True

    return S.tocsc()


def _sparse_lu_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
    sparsity: sp.spmatrix | None = None,
    *,
    fd_scale: np.ndarray | None = None,
    accept: Callable[[np.ndarray], bool] | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Sparse Jacobian Newton solver with splu and graph coloring finite differences.

    ``fd_scale`` (dense fallback only) and ``accept`` have the meaning described
    in :func:`_damped_newton_solve`; ``None`` disables them.
    """
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))
    history = [max_res]
    _record(info, "residual_history", history)

    def done(xx: np.ndarray, mr: float, df: float) -> bool:
        return (mr <= tol or df <= tol) and (accept is None or bool(accept(xx)))

    if done(x, max_res, diff):
        _record(info, "termination", "tolerance")
        return x, True, 0, max_res, diff, f_val

    for it in range(max_iter):
        # Evaluate sparse Jacobian via column grouping / CPR graph coloring
        if sparsity is not None:
            J_sparse = approx_derivative(
                f,
                x,
                method="2-point",
                sparsity=sparsity,
                f0=f_val,
                abs_step=eps_fd,
            )
            J_csc = J_sparse.tocsc() if sp.issparse(J_sparse) else sp.csc_matrix(J_sparse)
        else:
            # Fallback dense finite differences converted to CSC
            J_dense = _fd_jacobian_dense(f, x, f_val, eps_fd, fd_scale)
            J_csc = sp.csc_matrix(J_dense)

        # Direct sparse LU factorization via SuperLU
        try:
            lu = spla.splu(J_csc)
            delta = lu.solve(-f_val)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
            try:
                delta = spla.spsolve(J_csc, -f_val)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
                delta = np.linalg.lstsq(J_csc.toarray(), -f_val, rcond=1e-10)[0]

        # Half steps for the first three iterations, full steps afterwards
        xscal = 0.5 if it < 3 else 1.0

        # Backtracking line search
        alpha_step = xscal
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))
        history.append(max_res)

        if done(x, max_res, diff):
            _record(info, "termination", "tolerance")
            return x, True, it + 1, max_res, diff, f_val

    _record(info, "termination", "max_iter")
    return x, False, max_iter, max_res, diff, f_val


# ===========================================================================
# 3. Architecture C: Inexact Newton-Krylov with Two-Sided Physical Equilibration
# ===========================================================================

def _newton_krylov_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    calib: TradeCalibrationResult,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-6,
    krylov_method: str = "gmres",
    restart: int = 30,
    max_krylov_iter: int = 50,
    atol: float = 1e-5,
    *,
    accept: Callable[[np.ndarray], bool] | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Inexact Newton-Krylov (JFNK) solver with physical two-sided equilibration.

    The directional derivatives already use steps scaled by the transfer and
    foreign-balance magnitudes (``D_R``). ``accept`` has the meaning described
    in :func:`_damped_newton_solve`.
    """
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    ns, nc = calib.n_sectors, calib.n_countries
    M = ns * nc

    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))
    history = [max_res]
    _record(info, "residual_history", history)

    def done(xx: np.ndarray, mr: float, df: float) -> bool:
        return (mr <= tol or df <= tol) and (accept is None or bool(accept(xx)))

    if done(x, max_res, diff):
        _record(info, "termination", "tolerance")
        return x, True, 0, max_res, diff, f_val

    # Construct physical two-sided diagonal equilibration scalers (D_L, D_R)
    vars0 = unpack_equilibrium_vector(x, ns=ns, nc=nc)
    y_flat = vars0.y.flatten(order="F")
    L_flat = calib.l_endow.ravel()
    K_flat = calib.k_endow.ravel()
    GDP_flat = L_flat + K_flat

    # Right column scaling matrix D_R: maps state variables to O(1) units
    d_r = np.ones(n, dtype=float)
    i_T_start = 2 * M + 2 * nc
    i_XN_start = 2 * M + 3 * nc
    d_r[i_T_start:i_XN_start] = np.maximum(np.abs(vars0.T.ravel()), 1.0)
    d_r[i_XN_start:] = np.maximum(np.abs(vars0.XN.ravel()), 1.0)

    # Left row scaling matrix D_L: normalizes residual equations to relative dimensionless errors
    d_l = np.ones(n, dtype=float)
    d_l[:M] = 1.0 / np.maximum(y_flat, 1.0)
    # ff1 (zero profit) is already in O(1) price level units
    d_l[2 * M : 2 * M + nc] = 1.0 / np.maximum(L_flat, 1.0)
    d_l[2 * M + nc : 2 * M + 2 * nc] = 1.0 / np.maximum(K_flat, 1.0)
    d_l[2 * M + 2 * nc : 2 * M + 3 * nc - 1] = 1.0 / np.maximum(np.abs(GDP_flat[: nc - 1]), 1.0)
    d_l[2 * M + 3 * nc - 1 :] = 1.0 / np.maximum(np.abs(GDP_flat), 1.0)

    for it in range(max_iter):
        # Matrix-free directional derivative LinearOperator on equilibrated coordinates
        def matvec(v: np.ndarray) -> np.ndarray:
            norm_v = float(np.linalg.norm(v))
            if norm_v < 1e-14:
                return np.zeros_like(v)
            v_dir = v / norm_v
            # Scale perturbation back to physical space via D_R
            x_pert = x + eps_fd * (d_r * v_dir)
            f_pert = f(x_pert)
            return (d_l * ((f_pert - f_val) / eps_fd)) * norm_v

        A = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
        rhs = -d_l * f_val

        # Solve inexact Newton direction in equilibrated space
        if krylov_method == "bicgstab":
            sol_u, k_info = spla.bicgstab(A, rhs, maxiter=max_krylov_iter, atol=atol)
            if k_info != 0:
                sol_u, _ = spla.gmres(A, rhs, restart=restart, maxiter=max_krylov_iter, atol=atol)
        else:
            sol_u, k_info = spla.gmres(A, rhs, restart=restart, maxiter=max_krylov_iter, atol=atol)
            if k_info != 0:
                sol_u, _ = spla.bicgstab(A, rhs, maxiter=max_krylov_iter, atol=atol)

        # Un-equilibrate step direction: delta = D_R @ sol_u
        delta = d_r * sol_u

        # Damping schedule
        xscal = 0.5 if it < 3 else 1.0

        # Backtracking line search
        alpha_step = xscal
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))
        history.append(max_res)

        if done(x, max_res, diff):
            _record(info, "termination", "tolerance")
            return x, True, it + 1, max_res, diff, f_val

    _record(info, "termination", "max_iter")
    return x, False, max_iter, max_res, diff, f_val


# ===========================================================================
# 4. Broyden Quasi-Newton
# ===========================================================================

def _broyden_solve(
    f: Callable[[np.ndarray], np.ndarray],
    x_init: np.ndarray,
    tol: float = 2.5e-3,
    max_iter: int = 50,
    eps_fd: float = 1e-4,
    *,
    fd_scale: np.ndarray | None = None,
    accept: Callable[[np.ndarray], bool] | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool, int, float, float, np.ndarray]:
    """Pure NumPy Broyden quasi-Newton solver with Sherman-Morrison rank-1 inverse updates.

    ``fd_scale`` (initial Jacobian) and ``accept`` have the meaning described in
    :func:`_damped_newton_solve`; ``None`` disables them.
    """
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))
    history = [max_res]
    _record(info, "residual_history", history)

    def done(xx: np.ndarray, mr: float, df: float) -> bool:
        return (mr <= tol or df <= tol) and (accept is None or bool(accept(xx)))

    if done(x, max_res, diff):
        _record(info, "termination", "tolerance")
        return x, True, 0, max_res, diff, f_val

    # Initial Jacobian via finite differences
    B = _fd_jacobian_dense(f, x, f_val, eps_fd, fd_scale)

    try:
        H = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        H = np.linalg.pinv(B)

    for it in range(max_iter):
        delta = -H @ f_val
        alpha_step = 1.0
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)

        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)

        x_next = x_trial
        f_next = f_trial
        max_res = float(np.max(np.abs(f_next)))
        diff = float(np.sum(np.abs(f_next)))
        history.append(max_res)

        if done(x_next, max_res, diff):
            _record(info, "termination", "tolerance")
            return x_next, True, it + 1, max_res, diff, f_next

        # Sherman-Morrison rank-1 update
        s = x_next - x
        y = f_next - f_val
        Hy = H @ y
        sTH = s.T @ H
        denom = float(np.dot(sTH, y))
        if abs(denom) > 1e-12:
            H = H + np.outer(s - Hy, sTH) / denom

        x = x_next
        f_val = f_next

    _record(info, "termination", "max_iter")
    return x, False, max_iter, max_res, diff, f_val


# ===========================================================================
# 5. Ill-Conditioned Network Stabilization
# ===========================================================================

def svd_clamped_newton_step(
    J: np.ndarray,
    f_val: np.ndarray,
    D_L: np.ndarray,
    D_R: np.ndarray,
    max_comp: float = 20.0,
    max_disp: float = 0.30,
) -> np.ndarray:
    """Compute Newton step with SVD modal component clamping on ill-conditioned directions.

    Equilibrates the Jacobian J using two-sided scalers (D_L @ J @ D_R), projects the
    equilibrated residual onto the left singular vectors of J_eq, clamps the projection
    modal coefficients to [-max_comp, max_comp], and backprojects onto physical space.

    Parameters
    ----------
    J : np.ndarray
        Jacobian matrix of shape (n, n).
    f_val : np.ndarray
        Residual vector of shape (n,).
    D_L : np.ndarray
        Left row-equilibration scaling factors of shape (n,) or (n, n).
    D_R : np.ndarray
        Right column-equilibration scaling factors of shape (n,) or (n, n).
    max_comp : float, default 20.0
        Maximum allowable modal projection coefficient magnitude |c_k| <= max_comp.
    max_disp : float, default 0.30
        Factor wage displacement clamping bound |Delta w| <= max_disp. If positive,
        scales the step uniformly if max(|delta|) > max_disp.

    Returns
    -------
    np.ndarray
        Regularized Newton step delta = D_R @ sol_u of shape (n,).
    """
    dl = D_L if D_L.ndim == 1 else np.diag(D_L)
    dr = D_R if D_R.ndim == 1 else np.diag(D_R)

    J_eq = dl[:, np.newaxis] * (J * dr[np.newaxis, :])
    rhs_eq = -dl * f_val

    U, S, Vt = la.svd(J_eq)
    proj = U.T @ rhs_eq
    safe_S = np.where(S > 1e-15, S, 1e-15)
    coeffs = proj / safe_S

    clamped = np.abs(coeffs) > max_comp
    if np.any(clamped):
        coeffs = np.clip(coeffs, -max_comp, max_comp)

    sol_u = Vt.T @ coeffs
    delta = dr * sol_u

    if max_disp is not None and max_disp > 0:
        max_step = float(np.max(np.abs(delta)))
        if max_step > max_disp:
            delta = delta * (max_disp / max_step)

    return delta


def anderson_accelerate(
    x_hist: list[np.ndarray],
    g_hist: list[np.ndarray],
    f_hist: list[np.ndarray],
    m: int = 4,
) -> np.ndarray:
    """Compute depth-m Anderson accelerated iterate with least-squares mixing weights.

    Accelerates fixed-point or Newton iterations by finding optimal affine combination
    weights gamma in R^k minimizing ||sum gamma_i f_i||_2 subject to sum gamma_i = 1.

    Parameters
    ----------
    x_hist : list[np.ndarray]
        History of past state vectors [x_0, ..., x_{k-1}].
    g_hist : list[np.ndarray]
        History of proposal vectors [g_0, ..., g_{k-1}] where g_i = x_i + step_i.
    f_hist : list[np.ndarray]
        History of residual vectors [f_0, ..., f_{k-1}].
    m : int, default 4
        Maximum history depth for Anderson mixing.

    Returns
    -------
    np.ndarray
        Accelerated iterate x_{next} = sum_{i=0}^{k-1} gamma_i g_i.
    """
    k = min(len(x_hist), len(g_hist), len(f_hist), m)
    if k <= 0:
        raise ValueError("Anderson acceleration history buffers must be non-empty.")
    if k == 1:
        return np.asarray(g_hist[-1], dtype=float).copy()

    # Residual matrix R = [f_{-k}, ..., f_{-1}]
    R = np.column_stack([f_hist[-k + i] for i in range(k)])

    # Augmented KKT system for min ||R @ gamma||_2 s.t. 1^T gamma = 1:
    # [R^T R, 1; 1^T, 0] [gamma; mu] = [0; 1]
    KKT = np.empty((k + 1, k + 1), dtype=float)
    KKT[:k, :k] = R.T @ R
    KKT[:k, k] = 1.0
    KKT[k, :k] = 1.0
    KKT[k, k] = 0.0
    rhs_kkt = np.zeros(k + 1, dtype=float)
    rhs_kkt[k] = 1.0
    try:
        sol_kkt = la.lstsq(KKT, rhs_kkt)[0]
        gamma = sol_kkt[:k]
        sum_g = float(np.sum(gamma))
        if abs(sum_g) > 1e-14 and np.all(np.isfinite(gamma)):
            gamma = gamma / sum_g
        else:
            gamma = np.zeros(k, dtype=float)
            gamma[-1] = 1.0
    except Exception:
        gamma = np.zeros(k, dtype=float)
        gamma[-1] = 1.0

    x_next = np.zeros_like(g_hist[-1], dtype=float)
    for i in range(k):
        x_next += gamma[i] * np.asarray(g_hist[-k + i], dtype=float)

    return x_next


def solve_fold_continuation_step(
    S_ww: np.ndarray,
    rhs_w: np.ndarray,
    eval_stiff_fn: Callable[[float], tuple[float, float, Any]] | None = None,
    idx_stiff: int | str | None = None,
    c0: float = 0.0,
    c1: float | None = None,
    tol: float = 2.5e-3,
    max_secant_iter: int = 8,
    max_disp: float = 0.30,
    *,
    country_codes: Sequence[str] | None = None,
    return_info: bool = False,
) -> tuple[np.ndarray, float, bool] | tuple[np.ndarray, float, bool, dict[str, Any]]:
    """Block-elimination step for one stiff country coordinate, with a displacement clamp.

    Solves the linearized factor-price system ``S_ww @ dw = rhs_w`` (``nc``
    equations) by eliminating the other ``nc - 1`` coordinates conditionally on
    the stiff coordinate ``c = idx_stiff`` (nothing in the computation is
    country-specific):

        dw_{-c}(dw_c) = S_{-c,-c}^{-1} (rhs_{-c} - S_{-c,c} dw_c),

    which leaves the scalar equation
    ``f(dw_c) = rhs_c - S_{c,-c} dw_{-c}(dw_c) - S_{c,c} dw_c = 0``.

    * ``eval_stiff_fn=None`` (method ``"block_elimination"``): the scalar
      equation is solved exactly (Schur complement), so before any clamp the
      step equals ``scipy.linalg.solve(S_ww, rhs_w)`` up to rounding (when
      ``S_{-c,-c}`` is singular, least squares replaces the inner solves). No
      secant iteration runs.
    * ``eval_stiff_fn`` given (method ``"secant"``): a secant search on ``dw_c``
      calls ``eval_stiff_fn(dw_c) -> (f_c, max_residual, context)`` at points of
      the manifold ``dw_{-c}(dw_c)``. A candidate is converged when
      ``max_residual <= tol`` or ``|f_c| <= tol``; the first converged
      candidate is used, otherwise the one with the smallest ``max_residual``.

    Finally, if ``max|dw| > max_disp`` the whole step is scaled uniformly by
    ``s = max_disp / max|dw|``: the same direction, a shorter step.

    The returned ``residual`` and ``converged`` describe the step that is
    returned, after the clamp. (Up to 4.3.0 they described the unclamped step,
    so a clamped step was reported with a machine-zero residual and
    ``converged=True``.)

    * ``eval_stiff_fn=None``: ``residual = max_i |(S_ww @ dw - rhs_w)_i|`` over
      all ``nc`` rows and ``converged = residual <= tol``. A clamp by ``s < 1``
      leaves ``(1 - s) * rhs_w`` unexplained, so a clamped step is normally not
      converged.
    * ``eval_stiff_fn`` given and no clamp: ``residual`` is the ``max_residual``
      that ``eval_stiff_fn`` returned for the returned candidate, and
      ``converged`` applies the candidate rule above to it.
    * ``eval_stiff_fn`` given and a clamp applied: the scaled step lies off the
      manifold on which ``eval_stiff_fn`` evaluates, so no residual is known for
      it: ``residual`` is NaN and ``converged`` is False.

    The unclamped quantities are available with ``return_info=True``.

    Parameters
    ----------
    S_ww : np.ndarray
        Factor-price Schur complement (linearized system matrix), shape (nc, nc).
    rhs_w : np.ndarray
        Right-hand side, shape (nc,).
    eval_stiff_fn : Callable[[float], tuple[float, float, Any]], optional
        Callable returning ``(f_c, max_residual, context)`` for a candidate
        ``dw_c``. If None, the scalar equation is solved exactly.
    idx_stiff : int or str
        Position of the stiff country in ``S_ww`` and ``rhs_w``, or its code
        (resolved in ``country_codes``). Required; there is no default. An
        index outside ``[0, nc)`` raises ``ValueError``; no country is chosen
        for you.
    c0 : float, default 0.0
        First secant guess for ``dw_c``.
    c1 : float, optional
        Second secant guess (default: 10% of the exact linear ``dw_c``, or
        1e-4 when that is smaller than 1e-4 in absolute value).
    tol : float, default 2.5e-3
        Absolute convergence tolerance, in the units of ``rhs_w`` (or of
        ``eval_stiff_fn``'s residuals).
    max_secant_iter : int, default 8
        Maximum secant iterations (``eval_stiff_fn`` only).
    max_disp : float, default 0.30
        Uniform displacement clamp ``max|dw| <= max_disp`` (0.30 = 30% for log
        factor prices). ``None`` or a non-positive value disables it.
    country_codes : sequence of str, optional (keyword-only)
        Country roster of ``S_ww``. Needed when ``idx_stiff`` is a code; the
        code at the index used is recorded in ``info["country"]``.
    return_info : bool, default False (keyword-only)
        If True, also return a dict with ``idx_stiff``, ``country`` (or None),
        ``method``, ``clamped``, ``clamp_scale`` (``s``, 1.0 without a clamp),
        ``residual_kind`` (``"linear_system"``, ``"eval_stiff_fn"`` or
        ``"unavailable_after_clamp"``), ``linear_residual`` (``max|S_ww dw -
        rhs_w|`` of the returned step, always computed), ``direction_residual``
        and ``direction_converged`` (the residual and flag of the unclamped
        step, measured as above), ``unclamped_step`` and ``evaluations`` (the
        number of ``eval_stiff_fn`` calls).

    Returns
    -------
    tuple
        ``(dw, residual, converged)``, or ``(dw, residual, converged, info)``
        with ``return_info=True``. ``dw`` has shape (nc,).
    """
    S_ww = np.asarray(S_ww, dtype=float)
    rhs_w = np.asarray(rhs_w, dtype=float).ravel()
    nc = len(rhs_w)
    if S_ww.shape != (nc, nc):
        raise ValueError(f"S_ww must have shape ({nc}, {nc}) to match rhs_w; got {S_ww.shape}")
    codes = None if country_codes is None else [str(code) for code in country_codes]
    if codes is not None and len(codes) != nc:
        raise ValueError(f"country_codes has {len(codes)} entries; S_ww has {nc} rows")
    if idx_stiff is None:
        raise ValueError("idx_stiff is required: pass the stiff country's position in S_ww or its code")
    if isinstance(idx_stiff, str):
        if codes is None or idx_stiff not in codes:
            raise ValueError(f"idx_stiff={idx_stiff!r} must be one of country_codes; got {codes}")
        idx_stiff = codes.index(idx_stiff)
    idx_requested = int(idx_stiff)
    if not 0 <= idx_requested < nc:
        raise ValueError(f"idx_stiff={idx_requested} is outside [0, {nc}); pass the stiff country's "
                         f"position in S_ww or its code")
    idx_safe = idx_requested
    idx_rest = [c for c in range(nc) if c != idx_safe]

    S_rest = S_ww[np.ix_(idx_rest, idx_rest)]
    S_stiff_col = S_ww[idx_rest, idx_safe]
    rhs_rest = rhs_w[idx_rest]

    try:
        dw_rest_0 = la.solve(S_rest, rhs_rest)
        v_rest = la.solve(S_rest, -S_stiff_col)
    except la.LinAlgError:
        dw_rest_0 = la.lstsq(S_rest, rhs_rest)[0]
        v_rest = la.lstsq(S_rest, -S_stiff_col)[0]

    s_scalar = float(S_ww[idx_safe, idx_safe] + S_ww[idx_safe, idx_rest] @ v_rest)
    dw_stiff_lin = float((rhs_w[idx_safe] - S_ww[idx_safe, idx_rest] @ dw_rest_0) / (s_scalar if abs(s_scalar) > 1e-12 else 1.0))

    def on_manifold(c: float) -> np.ndarray:
        dw = np.empty(nc, dtype=float)
        dw[idx_rest] = dw_rest_0 + v_rest * c
        dw[idx_safe] = c
        return dw

    def linear_residual(dw: np.ndarray) -> float:
        return float(np.max(np.abs(S_ww @ dw - rhs_w))) if nc else 0.0

    evaluations = 0
    if eval_stiff_fn is not None:
        method = "secant"

        def accepted(f_val: float, res_val: float) -> bool:
            return bool(res_val <= tol or abs(f_val) <= tol)

        c_curr0 = float(c0)
        fc0, res0, _ = eval_stiff_fn(c_curr0)
        c_curr1 = float(c1) if c1 is not None else (dw_stiff_lin * 0.1 if abs(dw_stiff_lin) > 1e-4 else 1e-4)
        fc1, res1, _ = eval_stiff_fn(c_curr1)
        evaluations = 2

        # Candidate returned: the first converged one, else the smallest residual.
        best = (c_curr1, fc1, res1) if res1 < res0 else (c_curr0, fc0, res0)
        done = accepted(best[1], best[2])
        if not done:
            for c_try, f_try, r_try in ((c_curr0, fc0, res0), (c_curr1, fc1, res1)):
                if accepted(f_try, r_try):
                    best, done = (c_try, f_try, r_try), True
                    break

        for _ in range(max_secant_iter):
            if done or abs(fc1 - fc0) < 1e-14:
                break
            c_next = c_curr1 - fc1 * (c_curr1 - c_curr0) / (fc1 - fc0)
            fc_next, res_next, _ = eval_stiff_fn(c_next)
            evaluations += 1
            if accepted(fc_next, res_next):
                best, done = (c_next, fc_next, res_next), True
                break
            if res_next < best[2]:
                best = (c_next, fc_next, res_next)
            c_curr0, fc0 = c_curr1, fc1
            c_curr1, fc1 = c_next, fc_next

        best_c, best_f, best_res = float(best[0]), float(best[1]), float(best[2])
        dw_direction = on_manifold(best_c)
        direction_residual = best_res
        direction_converged = accepted(best_f, best_res)
    else:
        method = "block_elimination"
        dw_direction = on_manifold(dw_stiff_lin)
        direction_residual = linear_residual(dw_direction)
        direction_converged = bool(direction_residual <= tol)

    dw_full = dw_direction.copy()
    clamp_scale = 1.0
    if max_disp is not None and max_disp > 0 and nc:
        max_step = float(np.max(np.abs(dw_full)))
        if max_step > max_disp:
            clamp_scale = max_disp / max_step
            dw_full *= clamp_scale
    clamped = clamp_scale != 1.0

    step_linear_residual = linear_residual(dw_full)
    if eval_stiff_fn is None:
        residual_kind = "linear_system"
        residual, converged = step_linear_residual, bool(step_linear_residual <= tol)
    elif not clamped:
        residual_kind = "eval_stiff_fn"
        residual, converged = direction_residual, direction_converged
    else:
        residual_kind = "unavailable_after_clamp"
        residual, converged = float("nan"), False

    if not return_info:
        return dw_full, residual, converged
    info = {
        "idx_stiff": idx_safe,
        "country": None if codes is None else codes[idx_safe],
        "method": method,
        "clamped": clamped,
        "clamp_scale": float(clamp_scale),
        "residual_kind": residual_kind,
        "linear_residual": step_linear_residual,
        "direction_residual": float(direction_residual),
        "direction_converged": bool(direction_converged),
        "unclamped_step": dw_direction,
        "evaluations": evaluations,
    }
    return dw_full, residual, converged, info


solve_fold_continuation = solve_fold_continuation_step


def clamp_wage_displacement(
    delta_w: np.ndarray | float,
    max_disp: float = 0.30,
    mode: str = "coordinate",
) -> np.ndarray | float:
    """Clamp factor wage displacement to bound |Delta omega_c| <= max_disp.

    Parameters
    ----------
    delta_w : np.ndarray or float
        Factor wage displacement step(s).
    max_disp : float, default 0.30
        Maximum displacement threshold (+/- 30%).
    mode : {"coordinate", "uniform"}, default "coordinate"
        Clamping mode:
        - "coordinate": per-coordinate elementwise clipping via np.clip.
        - "uniform": uniform scaling when maximum norm exceeds max_disp.

    Returns
    -------
    np.ndarray or float
        Clamped displacement step(s).
    """
    if isinstance(delta_w, (int, float, np.floating, np.integer)):
        return float(np.clip(delta_w, -max_disp, max_disp))
    arr = np.asarray(delta_w, dtype=float)
    if mode == "uniform":
        max_step = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        if max_step > max_disp and max_disp > 0:
            return arr * (max_disp / max_step)
        return arr.copy()
    return np.clip(arr, -max_disp, max_disp)


# ===========================================================================
# 6. Keller's Bordered Pseudo-Arclength Continuation (PAC)
# ===========================================================================

_FACTOR_PRICE_BOUNDARY_LOG = float(np.log(1e-6))
"""Log factor price below which a no-reversal fold event is labelled a boundary."""


def _merge_fold_detections(
    detections: list[dict[str, Any]],
    lambda_trace: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    """Merge PAC ``tau_lambda <= 0`` detections at consecutive steps into events.

    ``detections`` holds one dict per detecting step (keys ``step``, ``lambda``,
    ``tau_lambda``, ``sigma_min``, ``kappa_2``, ``min_log_factor_price``);
    ``lambda_trace`` lists ``(step, accepted lambda at the start of the step)``
    for every step, followed by the final lambda. See :func:`solve_keller_pac`
    for the event fields and the ``kind`` rule.
    """
    groups: list[list[dict[str, Any]]] = []
    for det in detections:
        if groups and det["step"] == groups[-1][-1]["step"] + 1:
            groups[-1].append(det)
        else:
            groups.append([det])

    position = {step: i for i, (step, _) in enumerate(lambda_trace)}
    events: list[dict[str, Any]] = []
    for group in groups:
        first, last = group[0], group[-1]
        lams = [float(d["lambda"]) for d in group]
        after = position.get(last["step"])
        if after is not None and after + 1 < len(lambda_trace):
            lams.append(float(lambda_trace[after + 1][1]))
        reversed_ = any(b < a - 1e-12 * max(1.0, abs(a)) for a, b in zip(lams[:-1], lams[1:]))
        log_prices = [float(d.get("min_log_factor_price", float("nan"))) for d in group]
        finite_logs = [v for v in log_prices if np.isfinite(v)]
        min_log = min(finite_logs) if finite_logs else float("nan")
        if reversed_:
            kind = "lambda_reversal"
        elif np.isfinite(min_log) and min_log < _FACTOR_PRICE_BOUNDARY_LOG:
            kind = "factor_price_boundary"
        else:
            kind = "no_lambda_reversal"
        events.append({
            "step": int(first["step"]),
            "lambda": float(first["lambda"]),
            "tau_lambda": float(first["tau_lambda"]),
            "sigma_min": float(first["sigma_min"]),
            "kappa_2": float(first["kappa_2"]),
            "first_step": int(first["step"]),
            "last_step": int(last["step"]),
            "n_detections": len(group),
            "lambda_min": float(min(d["lambda"] for d in group)),
            "lambda_max": float(max(d["lambda"] for d in group)),
            "tau_lambda_min": float(min(d["tau_lambda"] for d in group)),
            "sigma_min_min": float(min(d["sigma_min"] for d in group)),
            "kappa_2_max": float(max(d["kappa_2"] for d in group)),
            "min_log_factor_price": min_log,
            "lambda_reversed": bool(reversed_),
            "kind": kind,
        })
    return events


def solve_keller_pac(
    calib: TradeCalibrationResult,
    tau_target: Any,
    tau_start: Any = None,
    ds_init: float = 0.05,
    ds_min: float = 1e-4,
    ds_max: float = 0.5,
    tol: float = 2.5e-3,
    max_steps: int = 100,
    base_result: TradeEquilibriumResult | None = None,
    *,
    tau_fd_target: np.ndarray | float | None = None,
    tauf_target: np.ndarray | None = None,
    tauf_fd_target: np.ndarray | None = None,
    tau_fd_start: np.ndarray | float | None = None,
    tauf_start: np.ndarray | None = None,
    tauf_fd_start: np.ndarray | None = None,
    sigma: float = 0.0,
    accounting: str = "consistent",
    fiscal_closure: str = "lump_sum",
    recycle_country: str | None = None,
    capacity_margins: dict[str, float] | np.ndarray | None = None,
    capacity_country: str | None = None,
    capacity_sector: str | None = None,
    x0: np.ndarray | None = None,
    condensed: bool | None = None,
    eps_fd: float = 1e-5,
    eps_lam: float = 1e-5,
    scalar_applies_to_final_demand: bool = True,
    foreign_saving_units: str = "numeraire",
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve the CGE trade equilibrium via Keller's Bordered Pseudo-Arclength Continuation (PAC).

    Embeds the tariff shock along a continuation path lambda in [0, 1] (the schedule is
    ``(1 - lambda) * start + lambda * target``) and solves the augmented bordered Jacobian
    system of dimension (K+1) x (K+1) with two-sided equilibration, a tangent predictor and an
    arclength corrector, followed by a Newton polish at lambda = 1.

    Fold monitor. A step whose oriented unit tangent has ``tau_lambda <= 0`` (the path
    no longer advances in lambda) is a detection. Consecutive detections form one event in
    ``metadata["fold_points"]``; ``metadata["fold_detections"]`` counts the raw detections.
    Each event keeps the keys of the first detection (``step``, ``lambda``, ``tau_lambda``,
    ``sigma_min``, ``kappa_2``) and adds ``first_step``, ``last_step``, ``n_detections``,
    ``lambda_min``, ``lambda_max``, ``tau_lambda_min``, ``sigma_min_min``,
    ``kappa_2_max``, ``min_log_factor_price`` (smallest log r or log w; the benchmark is 0),
    ``lambda_reversed`` (whether the accepted lambda fell during the event or at the next
    step, by more than ``1e-12 * max(1, |lambda|)``) and ``kind``:

    - ``"lambda_reversal"``: lambda fell, consistent with a turning point;
    - ``"factor_price_boundary"``: no reversal, and a factor price fell below 1e-6 of its
      benchmark level (a corner where a log variable diverges, not a fold);
    - ``"no_lambda_reversal"``: neither; typically finite-difference noise in
      ``tau_lambda`` near a singular Jacobian.

    ``tau_lambda <= 0`` alone is not evidence of an economic fold. No traversal of a fold of
    this CGE model has been demonstrated: on small two-country calibrations with moderate
    tariffs PAC records no event and plain Newton also converges, and runs that do record
    events (for example prohibitive tariffs that drive a factor price to zero) stop without
    converging. Consecutive detections are merged, so a single stall produces one
    ``fold_points`` entry.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau_target : TariffScenario, np.ndarray or float
        Target tariff schedule: a :class:`~puremacro.trade.scenarios.TariffScenario`,
        a scalar import tariff RATE, or an intermediate-tariff array with the
        conventions of :func:`solve_trade_equilibrium` (``(nc,)`` rates or
        multiplier arrays). A scenario sets intermediate and final-demand
        schedules; a string raises ``TypeError`` (named scenarios were
        removed). A scalar
        rate applies to intermediate imports and, by default
        (``scalar_applies_to_final_demand=True``), ALSO to final-demand imports;
        this differs from ``solve_trade_equilibrium(tau=rate)``, which leaves
        final-demand multipliers at one. "Scalar" means any 0-d real input:
        Python ``int``/``float``/``bool``, every NumPy integer, floating or bool
        scalar, and 0-d arrays (in 4.3.0 only Python ``int``/``float`` and
        ``np.float64`` did; ``np.float32``, ``np.int64`` and 0-d arrays tariffed
        intermediates only). An array target (ndim >= 1) leaves final-demand
        multipliers at one unless ``tau_fd_target``/``tauf_fd_target`` is given,
        as in ``solve_trade_equilibrium``.
    tau_start : TariffScenario, np.ndarray, float, or None, default None
        Starting tariff schedule at lambda = 0 (defaults to baseline tariffs),
        with the same conventions as ``tau_target``.
    ds_init : float, default 0.05
        Initial pseudo-arclength step size.
    ds_min : float, default 1e-4
        Minimum allowed step size before termination.
    ds_max : float, default 0.5
        Maximum allowed pseudo-arclength step size.
    tol : float, default 2.5e-3
        Maximum absolute residual tolerance ||F(x)||_inf <= tol.
    max_steps : int, default 100
        Maximum continuation steps along the manifold.
    base_result : TradeEquilibriumResult, optional
        Benchmark baseline result for post-processing.
    sigma : float, default 0.0
        Intermediate substitution elasticity.
    accounting, fiscal_closure, recycle_country, capacity_margins, capacity_country, capacity_sector
        As in :func:`solve_trade_equilibrium`.
    x0 : np.ndarray, optional
        Initial state guess. If None, uses calibrated baseline.
    condensed : bool, optional
        Only False/None is supported. True raises NotImplementedError; PAC uses
        the dense full system and is intended for small calibrations.
    eps_fd, eps_lam : float, default 1e-5
        Forward-difference steps for the state and the continuation parameter.
        The state step is scaled per unknown,
        ``h_j = eps_fd * max(|x_j|, s_j)`` (``s_j`` = 1 for log variables and the
        median baseline factor income for the value-unit transfer and foreign
        balance unknowns).
    tau_fd_target, tauf_target, tauf_fd_target : optional
        Explicit final-demand multipliers and national rate vectors for the
        target; when any is given the target is resolved exactly as
        ``solve_trade_equilibrium(tau=tau_target, tau_fd=tau_fd_target, ...)``.
    tau_fd_start, tauf_start, tauf_fd_start : optional
        The same for the starting schedule.
    scalar_applies_to_final_demand : bool, default True
        Scalar ``tau_target``/``tau_start`` only (keyword-only). True keeps the
        historical convenience convention (the rate also applies to
        final-demand imports). False mirrors ``solve_trade_equilibrium(tau=rate)``
        (final-demand multipliers one). The convention used is recorded in
        ``metadata["scalar_tariff_convention"]``.
    foreign_saving_units : {"numeraire", "world_income"}, default "numeraire"
        See ``solve_trade_equilibrium``.
    **kwargs : Any
        Accepted and ignored only for the Krylov options that
        ``solve_trade_equilibrium`` also takes; any other keyword raises
        ``TypeError``.

    Convergence requires the solved residuals, the omitted goods equation and
    the realized foreign balances to meet ``tol`` (the postprocessing audit);
    the terminal Newton polish continues until they do.

    Returns
    -------
    TradeEquilibriumResult
        Converged equilibrium solution container with post-processed trade flows and PAC metadata.
    """
    t_pac_start = time.perf_counter()

    check_accounting(accounting)
    _reject_unknown_options("solve_keller_pac", kwargs, frozenset({"krylov_method", "restart", "max_krylov_iter"}))
    check_policy_options(fiscal_closure, recycle_country, capacity_margins, capacity_country, capacity_sector)

    if condensed:
        raise NotImplementedError("Keller PAC currently supports only the full system; use condensed=False.")
    if not (0 < ds_min <= ds_init <= ds_max) or max_steps < 1 or tol <= 0:
        raise ValueError("Require 0 < ds_min <= ds_init <= ds_max, max_steps >= 1 and tol > 0.")
    if foreign_saving_units not in ("numeraire", "world_income"):
        raise ValueError("foreign_saving_units must be 'numeraire' or 'world_income'")

    def _is_scalar_rate(value: Any) -> bool:
        # One rule for every 0-d real input: Python int/float/bool, any NumPy
        # integer/floating/bool scalar and 0-d arrays. 4.3.0 tested
        # isinstance(value, (int, float)), so np.float64 (a float subclass) took
        # the scalar convention while np.float32, np.int64 and 0-d arrays fell
        # through to the intermediates-only array path.
        if value is None or isinstance(value, str):
            return False
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            return False
        return arr.ndim == 0 and arr.dtype.kind in "biuf"

    scalar_fd = bool(scalar_applies_to_final_demand)

    def _scenario_schedule(scenario: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from puremacro.trade.scenarios import build_tariff_matrices
        tau_s, tau_fd_s = build_tariff_matrices(calib, scenario)
        return _resolve_tariffs(calib, tau=tau_s, tau_fd=tau_fd_s)

    def _is_scenario(value: Any) -> bool:
        from puremacro.trade.scenarios import TariffScenario
        if isinstance(value, str):
            from puremacro.trade.scenarios import NAMED_SCENARIOS_REMOVED
            raise TypeError(NAMED_SCENARIOS_REMOVED)
        return isinstance(value, TariffScenario)

    # 2. Resolve target and start tariff structures
    if _is_scenario(tau_target):
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = _scenario_schedule(tau_target)
    elif _is_scalar_rate(tau_target):
        rate = float(tau_target)
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = _resolve_tariffs(
            calib, tau=rate, tau_fd=rate if scalar_fd else None)
    else:
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = _resolve_tariffs(calib, tau=tau_target, tau_fd=None)

    if tau_start is None:
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(calib, tau=None, tau_fd=None)
    elif _is_scenario(tau_start):
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _scenario_schedule(tau_start)
    elif _is_scalar_rate(tau_start):
        rate_s = float(tau_start)
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(
            calib, tau=rate_s, tau_fd=rate_s if scalar_fd else None)
    else:
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(calib, tau=tau_start, tau_fd=None)

    explicit_fd = tau_fd_target is not None or tauf_target is not None or tauf_fd_target is not None
    if _is_scalar_rate(tau_target) and not explicit_fd:
        scalar_convention = ("scalar rate applied to intermediate and final-demand imports"
                             if scalar_fd else
                             "scalar rate applied to intermediate imports only (final-demand multipliers one)")
    else:
        scalar_convention = "not applicable (non-scalar target or explicit final-demand schedule)"

    # Explicit schedules override the scalar convenience convention.
    if explicit_fd:
        tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v = _resolve_tariffs(
            calib, tau=tau_target, tau_fd=tau_fd_target,
            tauf=tauf_target, tauf_fd=tauf_fd_target,
        )
    if tau_fd_start is not None or tauf_start is not None or tauf_fd_start is not None:
        tau_s_a, taufd_s_a, tauf_s_v, tauf_fd_s_v = _resolve_tariffs(
            calib, tau=tau_start, tau_fd=tau_fd_start,
            tauf=tauf_start, tauf_fd=tauf_fd_start,
        )
    check_hawkins_simon_viability(calib, tau=tau_t_a)

    nc = calib.n_countries
    ns = calib.n_sectors
    M = ns * nc

    # Note: Continuation operates on the full (K+1) x (K+1) augmented bordered system
    # to preserve global equilibrium consistency across singular turning points.
    # The `condensed` parameter is retained for interface parity with solve_trade_equilibrium.
    _ = condensed

    if x0 is None:
        x_init = build_initial_guess(calib)
    else:
        x_init = np.asarray(x0, dtype=float).ravel()

    # Helper to interpolate tariffs along lambda in [0, 1]
    def get_tariffs(lam: float):
        l_c = float(np.clip(lam, -0.5, 2.0))
        t_a = (1.0 - l_c) * tau_s_a + l_c * tau_t_a
        tfd_a = (1.0 - l_c) * taufd_s_a + l_c * taufd_t_a
        tf_v = (1.0 - l_c) * tauf_s_v + l_c * tauf_t_v
        tffd_v = (1.0 - l_c) * tauf_fd_s_v + l_c * tauf_fd_t_v
        return t_a, tfd_a, tf_v, tffd_v

    model_options: dict[str, Any] = dict(
        sigma=sigma, accounting=accounting, fiscal_closure=fiscal_closure,
        recycle_country=recycle_country, capacity_margins=capacity_margins,
        capacity_country=capacity_country, capacity_sector=capacity_sector,
        foreign_saving_units=foreign_saving_units,
    )

    # Evaluators for full state
    def eval_full(x_vec: np.ndarray, lam: float) -> np.ndarray:
        t_a, tfd_a, tf_v, tffd_v = get_tariffs(lam)
        return compute_equilibrium_residuals(
            x=x_vec, calib=calib, tau=t_a, tau_fd=tfd_a, tauf=tf_v, tauf_fd=tffd_v, **model_options,
        )

    # State vector and scaling vector setup
    curr_x = x_init.copy()
    curr_lam = 0.0
    K = len(curr_x)
    # Scaled finite-difference steps and the full audit at the target.
    fd_scale = _consistent_fd_scale(calib, K)
    accept_target = _consistent_acceptor(
        calib, tau_t_a, taufd_t_a, tauf_t_v, tauf_fd_t_v, tol,
        sigma=sigma, foreign_saving_units=foreign_saving_units,
    )

    def jac_at(x_vec: np.ndarray, lam: float, f_ref: np.ndarray) -> np.ndarray:
        return _fd_jacobian_dense(lambda xx: eval_full(xx, lam), x_vec, f_ref, eps_fd, fd_scale)

    w_scale = np.ones(K, dtype=float)
    if K >= 2 * M + 4 * nc - 1:
        i_T = 2 * M + 2 * nc
        w_scale[i_T:] = 1e-6

    # Initial solve at lambda = 0.0
    f_0 = eval_full(curr_x, curr_lam)
    if float(np.max(np.abs(f_0))) > tol:
        for _ in range(10):
            f_val = eval_full(curr_x, curr_lam)
            if float(np.max(np.abs(f_val))) <= tol:
                break
            J_init = jac_at(curr_x, curr_lam, f_val)
            c_n = np.linalg.norm(J_init, axis=0); c_n[c_n == 0] = 1.0; DR = 1.0 / c_n
            r_n = np.linalg.norm(J_init * DR[np.newaxis, :], axis=1); r_n[r_n == 0] = 1.0; DL = 1.0 / r_n
            try:
                curr_x += DR * la.solve(DL[:, np.newaxis] * (J_init * DR[np.newaxis, :]), -DL * f_val)
            except la.LinAlgError:
                curr_x += DR * la.lstsq(DL[:, np.newaxis] * (J_init * DR[np.newaxis, :]), -DL * f_val)[0]

    ds = ds_init
    prev_tau_x = None
    prev_tau_lam = None
    fold_detections: list[dict[str, Any]] = []
    lambda_trace: list[tuple[int, float]] = []
    total_pac_steps = 0

    # Main Pseudo-Arclength Continuation loop
    for step_idx in range(max_steps):
        if curr_lam >= 1.0 - 1e-6:
            break

        total_pac_steps += 1
        f_curr = eval_full(curr_x, curr_lam)

        # Evaluate Jacobian J_x
        J_x = jac_at(curr_x, curr_lam, f_curr)

        # Evaluate dF/dlam
        f_pert_lam = eval_full(curr_x, curr_lam + eps_lam)
        dF_dlam = (f_pert_lam - f_curr) / eps_lam

        # Spectral diagnostic on J_x
        try:
            s_vals = la.svd(J_x, compute_uv=False)
            sigma_min = float(s_vals[-1])
            kappa_2 = float(s_vals[0] / max(sigma_min, 1e-15))
        except Exception:
            sigma_min = 1.0
            kappa_2 = 1.0

        # Tangent vector calculation via Bordered System:
        # [J_x, dF/dlam; prev_tau_x^T W^2, prev_tau_lam] [v_x; v_lam] = [0; 1]
        c_n = np.linalg.norm(J_x, axis=0); c_n[c_n == 0] = 1.0; DR = 1.0 / c_n
        r_n = np.linalg.norm(J_x * DR[np.newaxis, :], axis=1); r_n[r_n == 0] = 1.0; DL = 1.0 / r_n

        if prev_tau_x is not None and prev_tau_lam is not None:
            J_t_aug = np.empty((K + 1, K + 1), dtype=float)
            J_t_aug[:K, :K] = J_x
            J_t_aug[:K, K] = dF_dlam
            J_t_aug[K, :K] = prev_tau_x * (w_scale ** 2)
            J_t_aug[K, K] = prev_tau_lam
            rhs_t = np.zeros(K + 1, dtype=float)
            rhs_t[K] = 1.0
            try:
                c_ta = np.linalg.norm(J_t_aug, axis=0); c_ta[c_ta == 0] = 1.0; DR_t = 1.0 / c_ta
                r_ta = np.linalg.norm(J_t_aug * DR_t[np.newaxis, :], axis=1); r_ta[r_ta == 0] = 1.0; DL_t = 1.0 / r_ta
                v_sol = DR_t * la.solve(DL_t[:, np.newaxis] * (J_t_aug * DR_t[np.newaxis, :]), DL_t * rhs_t)
            except la.LinAlgError:
                v_sol = la.lstsq(J_t_aug, rhs_t)[0]
            v_x = v_sol[:K]
            v_lam = float(v_sol[K])
        else:
            try:
                v_x = DR * la.solve(DL[:, np.newaxis] * (J_x * DR[np.newaxis, :]), -DL * dF_dlam)
            except la.LinAlgError:
                v_x = DR * la.lstsq(DL[:, np.newaxis] * (J_x * DR[np.newaxis, :]), -DL * dF_dlam)[0]
            v_lam = 1.0

        # Normalize tangent with metric W
        tangent_norm = float(np.sqrt(np.sum((v_x * w_scale) ** 2) + v_lam ** 2))
        if tangent_norm < 1e-15:
            tangent_norm = 1.0
        tau_x = v_x / tangent_norm
        tau_lam = v_lam / tangent_norm

        # Orientation check: maintain forward path progression
        if prev_tau_x is not None and prev_tau_lam is not None:
            dot_prod = float(np.sum(tau_x * prev_tau_x * (w_scale ** 2)) + tau_lam * prev_tau_lam)
            if dot_prod < 0:
                tau_x = -tau_x
                tau_lam = -tau_lam
        else:
            if tau_lam < 0:
                tau_x = -tau_x
                tau_lam = -tau_lam

        # Fold monitor: record tau_lambda <= 0 detections; consecutive ones are
        # merged into single events after the loop (_merge_fold_detections).
        lambda_trace.append((step_idx, float(curr_lam)))
        if tau_lam <= 0.0:
            fold_detections.append({
                "step": step_idx,
                "lambda": float(curr_lam),
                "tau_lambda": float(tau_lam),
                "sigma_min": float(sigma_min),
                "kappa_2": float(kappa_2),
                "min_log_factor_price": (float(np.min(curr_x[2 * M:2 * M + 2 * nc]))
                                         if K >= 2 * M + 2 * nc else float("nan")),
            })

        prev_tau_x = tau_x.copy()
        prev_tau_lam = float(tau_lam)

        # Adaptive arclength step size: target delta_lambda ~ 0.02
        # Keep reductions after rejected correctors; adapt only on acceptance.

        # Clip predictor to lambda = 1.0 if approaching target
        if curr_lam < 1.0 and tau_lam > 0 and curr_lam + ds * tau_lam > 1.0:
            ds = min(ds, max((1.0 - curr_lam) / max(tau_lam, 1e-4), ds_min))

        # Secant tangent predictor
        x_pred = curr_x + ds * tau_x
        lam_pred = curr_lam + ds * tau_lam
        if lam_pred > 1.0 and tau_lam > 0:
            lam_pred = 1.0

        # Arclength corrector Newton iterations
        x_k = x_pred.copy()
        lam_k = float(lam_pred)
        conv_corr = False

        for it_corr in range(12):
            f_k = eval_full(x_k, lam_k)
            g_k = float(np.sum(tau_x * (w_scale ** 2) * (x_k - curr_x)) + tau_lam * (lam_k - curr_lam) - ds)
            err_k = max(float(np.max(np.abs(f_k))), abs(g_k))

            if err_k <= tol:
                conv_corr = True
                curr_x = x_k
                curr_lam = lam_k
                break

            # Evaluate augmented Jacobian
            J_xk = jac_at(x_k, lam_k, f_k)

            f_pert_k = eval_full(x_k, lam_k + eps_lam)
            dF_dlam_k = (f_pert_k - f_k) / eps_lam

            # Bordered (K+1) x (K+1) system
            J_aug = np.empty((K + 1, K + 1), dtype=float)
            J_aug[:K, :K] = J_xk
            J_aug[:K, K] = dF_dlam_k
            J_aug[K, :K] = tau_x * (w_scale ** 2)
            J_aug[K, K] = tau_lam

            rhs_aug = np.empty(K + 1, dtype=float)
            rhs_aug[:K] = -f_k
            rhs_aug[K] = -g_k

            # Two-sided equilibration of J_aug
            c_aug = np.linalg.norm(J_aug, axis=0); c_aug[c_aug == 0] = 1.0; DR_a = 1.0 / c_aug
            r_aug = np.linalg.norm(J_aug * DR_a[np.newaxis, :], axis=1); r_aug[r_aug == 0] = 1.0; DL_a = 1.0 / r_aug

            try:
                d_step = DR_a * la.solve(DL_a[:, np.newaxis] * (J_aug * DR_a[np.newaxis, :]), DL_a * rhs_aug)
            except la.LinAlgError:
                d_step = DR_a * la.lstsq(DL_a[:, np.newaxis] * (J_aug * DR_a[np.newaxis, :]), DL_a * rhs_aug)[0]

            # Backtracking on augmented step
            alpha_corr = 1.0
            x_trial = x_k + alpha_corr * d_step[:K]
            lam_trial = lam_k + alpha_corr * d_step[K]
            for _ in range(5):
                f_tr = eval_full(x_trial, lam_trial)
                g_tr = float(np.sum(tau_x * (w_scale ** 2) * (x_trial - curr_x)) + tau_lam * (lam_trial - curr_lam) - ds)
                err_tr = max(float(np.max(np.abs(f_tr))), abs(g_tr))
                if err_tr < err_k or err_tr <= tol:
                    break
                alpha_corr *= 0.5
                x_trial = x_k + alpha_corr * d_step[:K]
                lam_trial = lam_k + alpha_corr * d_step[K]

            x_k = x_trial
            lam_k = lam_trial

        if not conv_corr:
            # Halve arclength step and retry
            ds *= 0.5
            if ds < ds_min:
                break
        else:
            curr_x = x_k
            curr_lam = lam_k
            ds = min(ds_max, ds * (1.25 if it_corr < 5 else 1.0))

    t_pac_end = time.perf_counter()
    solve_duration = t_pac_end - t_pac_start
    lambda_trace.append((max_steps, float(curr_lam)))
    fold_points = _merge_fold_detections(fold_detections, lambda_trace)

    # Final terminal polish at target lambda = 1.0
    status = "converged"
    x_polish = curr_x.copy()
    try:
        f_final = eval_full(x_polish, 1.0)
        res_final = float(np.max(np.abs(f_final)))

        for _ in range(15):
            if (res_final <= tol and (tol > 1e-6 or res_final < 1e-10)
                    and (accept_target is None or accept_target(x_polish))):
                break
            J_fin = jac_at(x_polish, 1.0, f_final)
            c_f = np.linalg.norm(J_fin, axis=0); c_f[c_f == 0] = 1.0; DR_f = 1.0 / c_f
            r_f = np.linalg.norm(J_fin * DR_f[np.newaxis, :], axis=1); r_f[r_f == 0] = 1.0; DL_f = 1.0 / r_f
            try:
                step_f = DR_f * la.solve(DL_f[:, np.newaxis] * (J_fin * DR_f[np.newaxis, :]), -DL_f * f_final)
            except (la.LinAlgError, ValueError):
                step_f = DR_f * la.lstsq(DL_f[:, np.newaxis] * (J_fin * DR_f[np.newaxis, :]), -DL_f * f_final)[0]

            if np.any(np.isnan(step_f)) or np.any(np.isinf(step_f)):
                raise ValueError("Terminal polish step contains NaNs or infs")

            x_polish += step_f
            f_final = eval_full(x_polish, 1.0)
            res_final = float(np.max(np.abs(f_final)))

        conv_total = bool(np.isfinite(res_final) and res_final <= tol
                          and (accept_target is None or accept_target(x_polish)))
        if conv_total:
            curr_x = x_polish
        else:
            status = "step_limit_exhausted"
    except (la.LinAlgError, ValueError):
        conv_total = False
        res_final = float("nan")
        f_final = np.full(K, np.nan, dtype=float)
        status = "step_limit_exhausted"

    # Diagnostics always describe the iterate that is actually returned.
    f_final = eval_full(curr_x, 1.0)
    res_final = float(np.max(np.abs(f_final)))
    conv_total = bool(np.isfinite(res_final) and res_final <= tol
                      and (accept_target is None or accept_target(curr_x)))
    status = "converged" if conv_total else "step_limit_exhausted"

    res_dataclass = postprocess_trade_equilibrium(
        x_sol=curr_x,
        calib=calib,
        tau=tau_t_a,
        tau_fd=taufd_t_a,
        tauf=tauf_t_v,
        tauf_fd=tauf_fd_t_v,
        converged=conv_total,
        iterations=total_pac_steps,
        base_result=base_result,
        **model_options,
        metadata={
            "method": "keller_pac",
            "status": status,
            "tol": tol,
            "max_steps": max_steps,
            "pac_steps": total_pac_steps,
            "fold_points": fold_points,
            "fold_detections": len(fold_detections),
            "fold_monitor": ("tau_lambda <= 0 detections; consecutive detections merged into "
                             "one event; not by itself evidence of an economic fold"),
            "final_lambda": float(curr_lam),
            "solve_duration_seconds": solve_duration,
            "sigma": sigma,
            "fiscal_closure": fiscal_closure,
            "residual_norm": float(np.sum(np.abs(f_final))),
            "max_residual": res_final,
            "scalar_tariff_convention": scalar_convention,
            "fd_step": "scaled",
        },
    )

    return replace(
        res_dataclass,
        residuals=f_final,
        residual_norm=float(np.sum(np.abs(f_final))),
        diff=float(np.sum(np.abs(f_final))),
        max_residual=res_final,
    )


# ===========================================================================
# 7. Master Solver Interfaces
# ===========================================================================

_KELLER_OPTIONS = frozenset({
    "tau_start", "tau_fd_start", "tauf_start", "tauf_fd_start", "ds_init", "ds_min", "ds_max",
    "max_steps", "condensed", "eps_fd", "eps_lam", "scalar_applies_to_final_demand",
})
_SOLVER_OPTIONS = _KELLER_OPTIONS | {"krylov_method", "restart", "max_krylov_iter"}


def _reject_unknown_options(function: str, kwargs: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise TypeError(f"{function}() got unexpected keyword argument(s): {unknown}")


def solve_trade_equilibrium(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    tauf: np.ndarray | None = None,
    tauf_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    method: str = "newton",
    tol: float = 2.5e-3,
    max_iter: int = 50,
    base_result: TradeEquilibriumResult | None = None,
    *,
    backend: str = "numpy",
    sigma: float = 0.0,
    accounting: str = "consistent",
    fiscal_closure: str = "lump_sum",
    recycle_country: str | None = None,
    capacity_margins: dict[str, float] | np.ndarray | None = None,
    capacity_country: str | None = None,
    capacity_sector: str | None = None,
    foreign_saving_units: str = "numeraire",
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve the multi-country multi-sector general equilibrium nonlinear system.

    The model uses consistent producer/purchaser accounting: producer-price
    trade and duty bases, homogeneous Cobb-Douglas factor costs, output-revenue
    taxes, actual final-use tax bases, lump-sum rebates of all tariff revenue
    from the full bilateral schedule, fixed baseline foreign balances and a
    first-producer-price numeraire (:mod:`puremacro.trade._accounting`).

    Numerics. The Newton, sparse-LU, Broyden, Newton-Krylov and Keller solvers
    stop only when the solved residuals AND the audited physical residuals (the
    goods equation dropped for Walras' law and the realized net exports minus
    foreign saving) are at most ``tol``, so ``converged`` is not flipped
    afterwards by the postprocessing audit for a state the solver accepted.
    Finite-difference Jacobians use per-unknown steps
    ``h_j = eps * max(|x_j|, s_j)`` (``s_j`` = 1 for the log blocks, the median
    baseline factor income for the value-unit ``T`` and ``XN`` blocks), so the
    transfer and foreign-balance columns stay accurate at any currency unit.
    ``tol`` is an ABSOLUTE bound in the calibration's value units: the
    attainable floor is roughly ``1e-16 * max(output)``, so choose it for the
    units. ``hybr``/``lm`` use SciPy's own termination test; their state is
    audited afterwards.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters from :func:`calibrate_trade_model`.
    tau : float or np.ndarray, optional
        Intermediate-input tariffs. The interpretation depends on the shape:

        - scalar (any 0-d input) or shape ``(nc,)``: ad valorem tariff RATES by
          importing country (0.10 = 10%), NOT multipliers. Each is applied as
          the multiplier ``1 + rate`` to every foreign-origin intermediate flow
          into that country; domestic flows keep multiplier 1. Passing
          ``[1.10, 1.0]`` therefore imposes 110% and 100% tariffs.
        - shape ``(ns*nc, ns, nc)`` (origin country-sector, destination sector,
          destination country), ``(ns, nc, ns, nc)`` or ``(ns*nc, ns*nc)``:
          MULTIPLIERS ``1 + rate`` used as given.
        - None: no tariffs (all multipliers 1), unless ``tauf`` is given.
    tau_fd : float or np.ndarray, optional
        Final-demand tariffs with the same rules: a scalar or ``(nc,)`` vector
        holds RATES by importing country; ``(ns*nc, nfd, nc)``,
        ``(ns, nc, nfd, nc)`` or ``(ns*nc, nfd*nc)`` arrays are MULTIPLIERS.
        None: multipliers 1, unless ``tauf_fd`` is given.
    tauf : np.ndarray, optional
        National intermediate tariff rates by importing country, shape (nc,).
    tauf_fd : np.ndarray, optional
        National final demand tariff rates by importing country, shape (nc,).
    x0 : np.ndarray, optional
        Initial state vector. If None, constructed via :func:`build_initial_guess`.
    method : {"newton", "equilibrated_newton", "sparse_lu", "krylov", "condensed", "broyden", "hybr", "lm", "keller_pac"}, default "newton"
        Nonlinear root-finding algorithm:
        - 'newton': Dense Damped Newton-Raphson with backtracking line search.
        - 'equilibrated_newton': Dense Newton on the Ruiz row/column-equilibrated
          system with relative finite-difference steps (``eps_fd`` keyword, default
          1e-6) and backtracking on the scaled residual; suited to large tables
          whose raw Jacobian is badly scaled. When it stops without converging,
          ``metadata`` reports ``equilibrated_condition`` and the country carrying
          the weakest Jacobian direction (``near_singular_country`` and its share
          ``near_singular_country_weight``).
        - 'sparse_lu': Newton with a sparse LU (splu) factorisation of the Jacobian.
        - 'krylov': Inexact Newton-Krylov (JFNK via GMRES/BiCGSTAB) with two-sided equilibration.
        - 'condensed': the 'newton' solver (``metadata["effective_method"]`` says so).
        - 'broyden': Broyden quasi-Newton with Sherman-Morrison rank-1 updates.
        - 'hybr': SciPy MINPACK Powell hybrid root wrapper.
        - 'lm': SciPy Levenberg-Marquardt root wrapper.
        - 'keller_pac': Keller's Bordered Pseudo-Arclength Continuation in the tariff
          schedule (see :func:`solve_keller_pac`; its fold monitor is a diagnostic, and no
          traversal of an economic fold has been demonstrated).
    tol : float, default 2.5e-3
        Maximum absolute residual tolerance: ||F(x)||_inf <= tol.
    max_iter : int, default 50
        Maximum solver iterations.
    base_result : TradeEquilibriumResult, optional
        Benchmark baseline result used for relative index evaluation.
    backend : {"numpy"}, default "numpy"
        Array backend (keyword-only). Only NumPy is implemented; any other value
        raises ``NotImplementedError``.
    sigma : float, default 0.0
        Intermediate sourcing substitution elasticity (0 = Leontief).
    accounting : {"consistent"}, default "consistent"
        Any other value raises ``ValueError``: legacy accounting was removed.
    fiscal_closure : str, default "lump_sum"
        Use of tariff revenue. Any value other than ``"lump_sum"`` recycles one
        country's revenue and requires ``recycle_country`` (``TypeError``
        without it); the consistent accounting implements lump-sum rebates
        only, so such a request then raises ``NotImplementedError``.
    recycle_country : str, optional
        Country code whose revenue a non-lump-sum ``fiscal_closure`` recycles.
        No default.
    capacity_margins : dict or np.ndarray, optional
        Spare capacity margins of a capacity barrier; requires
        ``capacity_country`` and ``capacity_sector`` (``TypeError`` without
        them). Not implemented by the consistent accounting
        (``NotImplementedError``).
    capacity_country, capacity_sector : str, optional
        Country and sector codes of the capacity barrier. No defaults.
    foreign_saving_units : {"numeraire", "world_income"}, default "numeraire"
        Keyword-only. ``"numeraire"`` holds baseline foreign saving fixed in
        units of the numeraire (first country/sector producer price = 1); with
        nonzero foreign saving the real allocation, and Hicksian EV/CV, then
        depend on which country is listed first. ``"world_income"`` holds each
        country's baseline saving fixed as a share of world factor income,
        which makes the system homogeneous of degree zero in nominal variables
        and the results invariant to the country order.
    **kwargs
        Solver options: ``eps_fd`` (finite-difference step), ``krylov_method``,
        ``restart`` and ``max_krylov_iter`` ('krylov'), and the continuation
        options of :func:`solve_keller_pac` ('keller_pac'). Any other keyword
        raises ``TypeError``.

    Returns
    -------
    TradeEquilibriumResult
        Result container with solution arrays, reconstructed flows, and diagnostics.
        ``metadata["solver_termination"]`` and ``metadata["solver_residual_history"]``
        record why the finite-difference solvers stopped (``"tolerance"``,
        ``"max_iter"``, ``"non_finite_residual"``, ``"non_finite_jacobian"``,
        ``"linear_solve_failed"``, or ``"non_finite_iterate"`` when sparse LU,
        Krylov or Broyden reach ``max_iter`` at a non-finite state); a
        non-finite Newton state ends the solve with ``converged=False`` instead
        of raising.
    """
    check_accounting(accounting)
    _reject_unknown_options("solve_trade_equilibrium", kwargs, _SOLVER_OPTIONS)
    check_policy_options(fiscal_closure, recycle_country, capacity_margins, capacity_country, capacity_sector)
    if foreign_saving_units not in ("numeraire", "world_income"):
        raise ValueError("foreign_saving_units must be 'numeraire' or 'world_income'")
    if backend != "numpy":
        raise NotImplementedError("solve_trade_equilibrium supports the NumPy backend only")
    if method == "quasi_condensed":
        raise NotImplementedError(
            "method='quasi_condensed' belongs to the flexible model; "
            "use solve_flexible_trade_equilibrium(..., method='quasi_condensed')"
        )

    tau_a, taufd_a, tauf_vec, tauf_fd_vec = _resolve_tariffs(
        calib, tau=tau, tau_fd=tau_fd, tauf=tauf, tauf_fd=tauf_fd
    )

    if x0 is None:
        x_init = build_initial_guess(calib)
    else:
        x_init = np.asarray(x0, dtype=float).ravel()

    model_options: dict[str, Any] = dict(
        sigma=sigma, accounting=accounting, fiscal_closure=fiscal_closure,
        recycle_country=recycle_country, capacity_margins=capacity_margins,
        capacity_country=capacity_country, capacity_sector=capacity_sector,
        foreign_saving_units=foreign_saving_units,
    )

    # Scaled finite-difference steps and the postprocessing audit as the stopping test.
    fd_scale = _consistent_fd_scale(calib, len(x_init))
    accept = _consistent_acceptor(
        calib, tau_a, taufd_a, tauf_vec, tauf_fd_vec, tol,
        sigma=sigma, foreign_saving_units=foreign_saving_units,
    )
    solver_info: dict[str, Any] = {}

    def obj_fun(xx: np.ndarray) -> np.ndarray:
        return compute_equilibrium_residuals(
            x=xx, calib=calib, tau=tau_a, tau_fd=taufd_a, tauf=tauf_vec, tauf_fd=tauf_fd_vec,
            **model_options,
        )

    # Evaluate once before dispatch, so an unsupported option or an invalid
    # model raises here instead of inside a solver's error handling.
    obj_fun(x_init)

    t_solve_start = time.perf_counter()

    # Method dispatch
    if method in ("newton", "condensed"):
        x_sol, conv, iters, max_res, diff, res = _damped_newton_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter, eps_fd=1e-5,
            fd_scale=fd_scale, accept=accept, info=solver_info,
        )
    elif method == "equilibrated_newton":
        x_sol, conv, iters, max_res, diff, res = _equilibrated_newton_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter,
            eps_fd=float(kwargs.get("eps_fd", 1e-6)),
            fd_scale=fd_scale, accept=accept, info=solver_info,
        )
    elif method == "sparse_lu":
        x_sol, conv, iters, max_res, diff, res = _sparse_lu_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter, sparsity=None,
            fd_scale=fd_scale, accept=accept, info=solver_info,
        )
    elif method == "krylov":
        x_sol, conv, iters, max_res, diff, res = _newton_krylov_solve(
            obj_fun,
            x_init,
            calib=calib,
            tol=tol,
            max_iter=max_iter,
            krylov_method=kwargs.get("krylov_method", "gmres"),
            restart=kwargs.get("restart", 30),
            max_krylov_iter=kwargs.get("max_krylov_iter", 50),
            atol=0.0,
            accept=accept, info=solver_info,
        )
    elif method == "broyden":
        x_sol, conv, iters, max_res, diff, res = _broyden_solve(
            obj_fun, x_init, tol=tol, max_iter=max_iter,
            fd_scale=fd_scale, accept=accept, info=solver_info,
        )
    elif method in ("hybr", "lm"):
        from scipy.optimize import root

        n = len(x_init)
        # MINPACK stops on a *relative* step (xtol) and, for lm, a relative
        # reduction of the sum of squares (ftol); ``tol`` is an absolute bound
        # on the residual, judged below. Up to 4.4.0 ``tol`` was passed as
        # xtol, so hybr could stop with max|F| above tol and report
        # converged=False.
        opts: dict[str, Any] = {"xtol": 1e-12}
        if method == "hybr":
            opts["maxfev"] = max(2000, max_iter * n)
        elif method == "lm":
            opts["maxiter"] = max(2000, max_iter * n)
            opts["ftol"] = 1e-12

        res_scipy = root(obj_fun, x_init, method=method, options=opts)
        x_sol = np.asarray(res_scipy.x, dtype=float).ravel()
        res = obj_fun(x_sol)
        max_res = float(np.max(np.abs(res)))
        diff = float(np.sum(np.abs(res)))
        conv = bool(np.isfinite(max_res) and max_res <= tol)
        iters = int(getattr(res_scipy, "nfev", 0))
    elif method == "keller_pac":
        keller_options = {k: v for k, v in kwargs.items() if k in _KELLER_OPTIONS}
        keller_options.setdefault("max_steps", max_iter if max_iter > 50 else 100)
        return solve_keller_pac(
            calib=calib,
            tau_target=tau_a,
            tau_fd_target=taufd_a,
            tauf_target=tauf_vec,
            tauf_fd_target=tauf_fd_vec,
            tol=tol,
            base_result=base_result,
            x0=x0,
            **model_options,
            **keller_options,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid options are: 'newton', 'equilibrated_newton', 'sparse_lu', "
            "'krylov', 'condensed', 'broyden', 'hybr', 'lm', 'keller_pac'."
        )

    t_solve_end = time.perf_counter()
    solve_duration = t_solve_end - t_solve_start
    if ("termination" in solver_info and solver_info["termination"] == "max_iter"
            and not (np.isfinite(x_sol).all() and np.isfinite(res).all())):
        # Sparse LU, Krylov and Broyden can exhaust max_iter at a non-finite
        # iterate; say so instead of recording a plain iteration limit.
        solver_info["termination"] = "non_finite_iterate"

    res_dataclass = postprocess_trade_equilibrium(
        x_sol=x_sol,
        calib=calib,
        tau=tau_a,
        tau_fd=taufd_a,
        tauf=tauf_vec,
        tauf_fd=tauf_fd_vec,
        converged=conv,
        iterations=iters,
        base_result=base_result,
        **model_options,
        metadata={
            **({"solver_termination": solver_info["termination"],
                "solver_residual_history": list(solver_info.get("residual_history", []))}
               if "termination" in solver_info else {}),
            **_near_singular_metadata(calib, solver_info),
            "fd_step": "scaled",
            "method": method,
            "effective_method": "newton" if method == "condensed" else method,
            "backend": backend,
            "tol": tol,
            "max_iter": max_iter,
            "solve_duration_seconds": solve_duration,
        },
    )

    return res_dataclass


def solve_equilibrium(
    calib: TradeCalibrationResult,
    tau: np.ndarray | None = None,
    tau_fd: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    method: str = "newton",
    tol: float = 2.5e-3,
    max_iter: int = 50,
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve the trade equilibrium; a short form of :func:`solve_trade_equilibrium`.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibrated model structural parameters.
    tau, tau_fd : np.ndarray, optional
        Intermediate and final-demand tariffs (conventions of
        :func:`solve_trade_equilibrium`).
    x0 : np.ndarray, optional
        Initial state vector.
    method : str, default "newton"
        Solver architecture (see :func:`solve_trade_equilibrium`).
    tol : float, default 2.5e-3
        Maximum absolute residual tolerance.
    max_iter : int, default 50
        Maximum solver iterations.
    **kwargs : Any
        Forwarded to :func:`solve_trade_equilibrium` (``accounting``,
        ``fiscal_closure``/``recycle_country``,
        ``capacity_margins``/``capacity_country``/``capacity_sector``, ...).

    Returns
    -------
    TradeEquilibriumResult
        Equilibrium solution container.
    """
    return solve_trade_equilibrium(
        calib=calib,
        tau=tau,
        tau_fd=tau_fd,
        x0=x0,
        method=method,
        tol=tol,
        max_iter=max_iter,
        **kwargs,
    )


build_cge_sparsity_pattern = _build_cge_sparsity_pattern

__all__ = [
    "build_initial_guess",
    "build_cge_sparsity_pattern",
    "solve_trade_equilibrium",
    "solve_equilibrium",
    "check_hawkins_simon_viability",
    "ViabilityResult",
    "solve_keller_pac",
    "svd_clamped_newton_step",
    "anderson_accelerate",
    "solve_fold_continuation_step",
    "solve_fold_continuation",
    "clamp_wage_displacement",
]
