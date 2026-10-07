"""Solvers of the condensed model: damped Newton, continuation, pseudo-arclength, multistart.

* :func:`newton`: dense-LU Newton on the ``2N`` system with a forward-difference
  Jacobian (prices cached across the ``Y_hat`` columns), Armijo backtracking on
  ``||r||_2`` (``c = 1e-4``, halving, ``alpha >= 1e-6``), a clamp
  ``max |d omega| <= 0.3`` on log-wage displacements, chord reuse of the
  Jacobian while ``||r_new|| / ||r|| <= 0.1``, rejection of non-finite trials or
  ``Y_hat <= 0``, and the stopping rule ``max |r| <= 1e-12`` with the dropped
  Walras residual ``<= 1e-10`` within 40 iterations. Best-trial acceptance
  without descent and clipping of the unknowns are never used.
* :func:`continuation`: natural-parameter continuation in the tariff scale,
  ``tau(lam) = 1 + lam * delta`` for every wedge family. Tries ``lam = 1`` first,
  then halves the step on failure (floor ``1/256``), doubles it after a
  success, and predicts with a secant through the last two accepted points. It
  never returns a partial ``lam``.
* :func:`arclength`: pseudo-arclength continuation in ``(z, lam)`` with the
  tangent from ``J v = -dr/dlam``, monitors ``dlam/ds``, ``sign det J_z`` and
  ``sigma_min / sigma_max``, raises :class:`FoldDetected` on a sign change, and
  checks that the landed root equals the direct root to ``1e-10``.
* :func:`multistart_near`: near-start uniqueness evidence (``omega_0 ~ N(0, 0.15^2)``,
  ``Y_hat_0 = 1 + N(0, 0.05^2)``); a converged root farther than ``1e-9`` raises
  :class:`MultipleEquilibria`. This is evidence, not a uniqueness proof.
  Eight starts up to eleven sectors and six beyond (the IO engine used eight
  only at exactly eleven sectors; a documented deviation).
* :func:`nested_fallback`: given wages, the ``N`` income equations are affine
  in ``Y_hat``; solve them and polish with Newton.
* :func:`existence_gate`: Collatz-Wielandt certificate ``rho(B_tau) < 1 - 1e-3``
  (fail closed; bounds that certify ``rho < 1`` but not the margin, or nothing at
  all, raise :class:`ProductivityUncertified`; wedges of another table raise
  :class:`CalibrationError`).
* :func:`solve_condensed`: gate, solve, optional arclength and multistart, then
  the independent raw-flow certificate; returns
  :class:`~puremacro.trade.condensed._results.CondensedEquilibriumResult` or
  raises. A failed solve never returns ``converged=True``; a result solved with
  ``certify=False`` (diagnostics only) is ``converged`` but neither ``passed``
  nor ``certified``.

No turning point (fold) has been exhibited on an economic example. The tests
exercise the fold monitor with a synthetic determinant sign change and with an
existence boundary: on the synthetic 3x4 economy a 500% USA goods duty passes
the existence gate, but the equilibrium with a positive U.S. wage ceases to
exist near a 259% duty (``lambda* = 0.51773`` of 500%). Along that branch
``lambda`` rises monotonically towards ``lambda*`` while the U.S. wage tends to
zero, natural-parameter continuation stalls at ``lambda = 0.515625``, and
:func:`arclength` raises :class:`FoldDetected` at ``lambda = 0.5177`` (U.S.
wage ``1.6e-8``) because the Jacobian is numerically singular there
(``sigma_min / sigma_max = 3e-11``, ``det J`` of order ``1e-13``) and the
determinant-sign monitor flips, not because the branch turns back.
:class:`FoldDetected` therefore means "the path cannot be continued
regularly": a turning point or an existence boundary.

Quoting the IO documentation: "No global uniqueness, universal fold threshold,
or global Newton guarantee is asserted." and "Numerical residual checks do not
establish global equilibrium existence, uniqueness, or optimality."
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.linalg as la

from ._results import CondensedEquilibriumResult
from .calibration import CondensedCalibration, ProductivityBounds, certify_productivity
from .certify import CERTIFICATE_TOL, certify_raw_flows
from .errors import (
    CalibrationError,
    CondensedSolveError,
    ContinuationFailure,
    EquilibriumNotFound,
    FoldDetected,
    MultipleEquilibria,
)
from .model import CondensedLeontiefModel
from .tariffs import TariffWedges

SOLVER_TOL = 1e-12
WALRAS_TOL = 1e-10
MAX_DISPLACEMENT = 0.3
ARMIJO_C = 1e-4
MIN_ALPHA = 1e-6
EXISTENCE_MARGIN = 1e-3
MIN_STEP = 1.0 / 256.0


def existence_gate(calib: CondensedCalibration, wedges: TariffWedges, *,
                   margin: float = EXISTENCE_MARGIN) -> ProductivityBounds:
    """Certify ``rho(B_tau) < 1 - margin`` for ``B_tau = a tau / (1 - t)`` (fail closed).

    :meth:`TariffWedges.from_arrays` rejects multipliers below one, so
    ``B_tau(lam) = a (1 + lam (tau - 1)) / (1 - t)`` is entrywise nondecreasing
    in the tariff scale on ``[0, 1]`` and a certificate at ``lam = 1`` covers
    every continuation stage.

    Raises
    ------
    CalibrationError
        The wedge arrays do not have the shapes of this calibration (they were
        built for another table).
    ExistenceViolation
        A certified lower bound of at least one.
    ProductivityUncertified
        The bounds certify ``rho < 1`` but not ``rho < 1 - margin`` ("margin not
        met"), or they certify nothing ("inconclusive").
    """
    M, N = int(calib.n_cells), int(calib.n_countries)
    for name, shape in (("tau", (M, M)), ("tau_fd", (M, N, 3)), ("tau_V", (M, N))):
        got = tuple(np.shape(getattr(wedges, name)))
        if got != shape:
            raise CalibrationError(
                f"wedges.{name} has shape {got} but the calibration has {M} cells and {N} economies "
                f"(expected {shape}); the wedges were built for another table"
            )
    B_tau = (calib.a * wedges.tau) / (1.0 - calib.t)[None, :]
    return certify_productivity(B_tau, margin=margin)


def _stage_record(lam: float, info: dict[str, Any], **extra: Any) -> dict[str, Any]:
    rec = {"lam": float(lam), "ok": True, "iterations": info["iterations"], "jacobians": info["jacobians"],
           "max_scaled_residual": info["max_scaled_residual"], "walras_residual": info["walras_residual"],
           "seconds": info["seconds"]}
    rec.update(extra)
    return rec


def newton(
    model: CondensedLeontiefModel,
    z0: np.ndarray | None = None,
    *,
    tol: float = SOLVER_TOL,
    walras_tol: float = WALRAS_TOL,
    max_iter: int = 40,
    max_displacement: float = MAX_DISPLACEMENT,
    tag: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Damped Newton with Armijo line search and a log-wage displacement clamp.

    Parameters
    ----------
    model : CondensedLeontiefModel
    z0 : np.ndarray, optional
        Starting point; the benchmark ``(0, 1)`` by default.
    tol : float, default 1e-12
        Stopping tolerance on ``max |r|``.
    walras_tol : float, default 1e-10
        Tolerance on the dropped Walras residual.
    max_iter : int, default 40
    max_displacement : float, default 0.3
        Cap on ``max_k |d omega_k|`` per step.
    tag : str
        Diagnostic tag for exception messages.

    Returns
    -------
    (z, info) : the root and ``{"iterations", "jacobians", "max_scaled_residual",
    "walras_residual", "hist", "seconds"}``.

    Raises
    ------
    EquilibriumNotFound
        Singular Jacobian, no descent in the line search, or the stopping rule
        not met within ``max_iter``; ``z_last`` and ``residual_last`` are attached.
    """
    t0 = time.time()
    n = model.n_countries
    z = model.z0.copy() if z0 is None else np.array(z0, dtype=float, copy=True)
    if z.shape != (2 * n,):
        raise ValueError(f"z0 must have shape {(2 * n,)}")
    st = model.state(z)
    r = st.residual
    p = st.p
    walras = float(abs(st.walras_residual))
    norm_r = float(la.norm(r))
    hist = [float(np.max(np.abs(r)))]
    if hist[0] < tol and walras < walras_tol:
        return z, {"iterations": 0, "jacobians": 0, "max_scaled_residual": hist[0],
                   "walras_residual": walras, "hist": hist, "seconds": time.time() - t0}
    n_jac = 0
    J = None
    lu_J = None
    norm_r_prev = norm_r
    for it in range(max_iter):
        max_r = float(np.max(np.abs(r)))
        if max_r <= tol and walras <= walras_tol:
            break
        recompute = True
        if J is not None and it > 0:
            if norm_r / max(norm_r_prev, 1e-300) <= 0.1:
                recompute = False
        if recompute:
            J = model.jacobian(z, r0=r, p0=p)
            n_jac += 1
            try:
                lu_J = la.lu_factor(J)
            except la.LinAlgError as exc:
                raise EquilibriumNotFound(f"{tag}: singular Jacobian at iteration {it}",
                                          z_last=z.copy(), residual_last=max_r) from exc
        dz = la.lu_solve(lu_J, -r)
        mx = float(np.max(np.abs(dz[:n])))
        if mx > max_displacement:
            dz = dz * (max_displacement / mx)
        alpha = 1.0
        accepted = False
        norm_r_prev = norm_r
        while alpha >= MIN_ALPHA:
            z_trial = z + alpha * dz
            if np.all(z_trial[n:] > 0) and np.all(np.isfinite(z_trial)):
                st_trial = model.state(z_trial)
                r_trial = st_trial.residual
                norm_trial = float(la.norm(r_trial))
                if np.all(np.isfinite(r_trial)) and norm_trial <= (1.0 - ARMIJO_C * alpha) * norm_r:
                    z = z_trial
                    r = r_trial
                    p = st_trial.p
                    walras = float(abs(st_trial.walras_residual))
                    norm_r = norm_trial
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            raise EquilibriumNotFound(
                f"{tag}: no descent in line search at iteration {it} (alpha < {MIN_ALPHA:.1e}, max|r|={hist[-1]:.3e})",
                z_last=z.copy(), residual_last=hist[-1],
            )
        hist.append(float(np.max(np.abs(r))))
    max_r = float(np.max(np.abs(r)))
    if max_r > tol or walras > walras_tol:
        raise EquilibriumNotFound(
            f"{tag}: Newton did not reach the stopping rule in {max_iter} iterations: "
            f"max|r| = {max_r:.3e} (tol={tol:.0e}), walras = {walras:.3e} (tol={walras_tol:.0e})",
            z_last=z.copy(), residual_last=max_r,
        )
    return z, {"iterations": len(hist) - 1, "jacobians": n_jac, "max_scaled_residual": max_r,
               "walras_residual": walras, "hist": hist, "seconds": time.time() - t0}


def continuation(
    calib: CondensedCalibration,
    wedges: TariffWedges,
    *,
    closure: str = "factor",
    numeraire: str = "factor",
    fd_rule: str = "absorption",
    inventories: str = "tariffed",
    tol: float = SOLVER_TOL,
    walras_tol: float = WALRAS_TOL,
    max_iter: int = 40,
    min_step: float = MIN_STEP,
    tag: str = "",
) -> tuple[np.ndarray, list[dict[str, Any]], CondensedLeontiefModel]:
    """Natural-parameter continuation in the tariff scale ``lam in [0, 1]``.

    Returns ``(z_final, stage_history, model_at_lam_1)``. Every stage record has
    ``lam``, ``ok``, iterations, Jacobians, residuals and seconds (failed
    stages carry ``err``).

    Raises
    ------
    ContinuationFailure
        The step fell below ``min_step``; ``lam_reached`` and ``z_last`` are attached.
    """
    model_full = CondensedLeontiefModel(calib, wedges, closure=closure, numeraire=numeraire,
                                        fd_rule=fd_rule, inventories=inventories)
    try:
        z_direct, info = newton(model_full, z0=model_full.z0, tol=tol, walras_tol=walras_tol,
                                max_iter=max_iter, tag=f"{tag} [direct lambda=1]")
        return z_direct, [_stage_record(1.0, info)], model_full
    except EquilibriumNotFound:
        pass
    n = calib.n_countries
    z = model_full.z0.copy()
    z_prev = None
    lam = 0.0
    lam_prev = None
    step = 0.5
    path: list[dict[str, Any]] = []
    while lam < 1.0 - 1e-14:
        lam_target = min(1.0, lam + step)
        model_lam = model_full.with_wedges(wedges.scaled(lam_target))
        if z_prev is not None and lam_prev is not None and abs(lam - lam_prev) > 1e-12:
            z_pred = z + (z - z_prev) * ((lam_target - lam) / (lam - lam_prev))
            z_pred[n:] = np.maximum(z_pred[n:], 0.2)
        else:
            z_pred = z.copy()
        try:
            z_corr, info = newton(model_lam, z0=z_pred, tol=tol, walras_tol=walras_tol, max_iter=max_iter,
                                  tag=f"{tag} [lam={lam_target:.4f}]")
            path.append(_stage_record(lam_target, info))
            z_prev, lam_prev = z, lam
            z, lam = z_corr, lam_target
            step = min(1.0, 2.0 * step)
        except EquilibriumNotFound as exc:
            path.append({"lam": lam_target, "ok": False, "err": str(exc)})
            step *= 0.5
            if step < min_step:
                raise ContinuationFailure(
                    f"{tag}: continuation stalled with step {step:.6f} < min_step {min_step:.6f}",
                    lam_reached=lam, z_last=z.copy(),
                ) from exc
    return z, path, model_full


def arclength(
    model: CondensedLeontiefModel,
    z_direct: np.ndarray,
    *,
    max_steps: int = 200,
    tol: float = SOLVER_TOL,
    landing_tol: float = 1e-10,
    tag: str = "",
) -> list[dict[str, Any]]:
    """Pseudo-arclength continuation from the benchmark to ``lam = 1`` with fold monitors.

    Parameters
    ----------
    model : CondensedLeontiefModel
        The model at ``lam = 1`` (its wedges are scaled internally).
    z_direct : np.ndarray
        The directly converged root at ``lam = 1``.
    max_steps : int, default 200
    tol : float, default 1e-12
        Corrector tolerance on the augmented residual.
    landing_tol : float, default 1e-10
        Required ``|z_arc - z_direct|_inf`` after the Newton landing at ``lam = 1``.

    Returns
    -------
    list of dict
        Per accepted step: ``step``, ``lam``, ``ds``, ``dlam_ds``, ``sign_det``, ``sigma_min_rel``.

    Raises
    ------
    FoldDetected
        ``dlam/ds`` turned negative or ``det J_z`` changed sign.
    ContinuationFailure
        The step fell below ``1e-6`` or ``max_steps`` was exhausted.
    CondensedSolveError
        The landed root differs from ``z_direct`` by more than ``landing_tol``.
    """
    wedges = model.wedges
    dim = 2 * model.n_countries
    n = model.n_countries
    z_direct = np.asarray(z_direct, dtype=float)

    def build(lam_val: float) -> CondensedLeontiefModel:
        return model.with_wedges(wedges.scaled(lam_val))

    z = model.z0.copy()
    lam = 0.0
    ds = 0.1
    m0 = build(0.0)
    J0 = m0.jacobian(z)
    sign_det_prev = int(np.sign(la.det(J0)))
    dr_dlam = (build(1e-6).residual(z) - m0.residual(z)) / 1e-6
    v0 = la.solve(J0, -dr_dlam)
    tangent = np.concatenate([v0, [1.0]])
    tangent = tangent / float(la.norm(tangent))
    history: list[dict[str, Any]] = []
    it = 0
    for step_idx in range(max_steps):
        if lam >= 1.0 - 1e-12:
            break
        if tangent[dim] > 0 and lam + ds * tangent[dim] > 1.0:
            ds = (1.0 - lam) / tangent[dim]  # aim the last predictor at lambda = 1 instead of overshooting
        z_curr = z + ds * tangent[:dim]
        lam_curr = float(lam + ds * tangent[dim])
        converged = False
        for it in range(15):
            if not np.all(np.isfinite(z_curr)) or np.any(z_curr[n:] <= 0):
                break
            m_curr = build(lam_curr)
            r_c = m_curr.residual(z_curr)
            g_c = float(tangent[:dim] @ (z_curr - z) + tangent[dim] * (lam_curr - lam) - ds)
            res_aug = np.concatenate([r_c, [g_c]])
            if not np.all(np.isfinite(res_aug)):
                break
            if float(np.max(np.abs(res_aug))) <= tol:
                converged = True
                break
            J_z = m_curr.jacobian(z_curr)
            dr_dl = (build(lam_curr + 1e-6).residual(z_curr) - r_c) / 1e-6
            J_aug = np.empty((dim + 1, dim + 1), dtype=float)
            J_aug[:dim, :dim] = J_z
            J_aug[:dim, dim] = dr_dl
            J_aug[dim, :dim] = tangent[:dim]
            J_aug[dim, dim] = tangent[dim]
            try:
                d_aug = la.solve(J_aug, -res_aug)
            except la.LinAlgError:
                break
            z_curr = z_curr + d_aug[:dim]
            lam_curr += float(d_aug[dim])
        if not converged:
            ds *= 0.5
            if ds < 1e-6:
                raise ContinuationFailure(f"{tag}: arclength continuation stalled (ds < 1e-6)",
                                          lam_reached=lam, z_last=z.copy())
            continue
        m_acc = build(lam_curr)
        J_acc = m_acc.jacobian(z_curr)
        sign_det = int(np.sign(la.det(J_acc)))
        dlam_ds = float(tangent[dim])
        svals = la.svdvals(J_acc)
        sigma_ratio = float(svals[-1] / svals[0]) if svals[0] > 0 else 0.0
        if dlam_ds < -1e-6 or (sign_det != 0 and sign_det_prev != 0 and sign_det != sign_det_prev):
            raise FoldDetected(lam=lam_curr, sigma_min=sigma_ratio, z_last=z_curr.copy())
        dr_dl_acc = (build(lam_curr + 1e-6).residual(z_curr) - m_acc.residual(z_curr)) / 1e-6
        v_next = la.solve(J_acc, -dr_dl_acc)
        t_next = np.concatenate([v_next, [1.0]])
        t_next = t_next / float(la.norm(t_next))
        if float(t_next @ tangent) < 0:
            t_next = -t_next
        tangent = t_next
        z = z_curr
        lam = lam_curr
        sign_det_prev = sign_det
        history.append({"step": step_idx, "lam": lam, "ds": ds, "dlam_ds": dlam_ds,
                        "sign_det": sign_det, "sigma_min_rel": sigma_ratio})
        if it <= 3:
            ds = min(1.5 * ds, 0.5)
        elif it > 7:
            ds = max(0.7 * ds, 1e-6)
    else:
        if lam < 1.0 - 1e-12:
            raise ContinuationFailure(f"{tag}: arclength continuation exceeded {max_steps} steps",
                                      lam_reached=lam, z_last=z.copy())
    z_land, _ = newton(build(1.0), z0=z, tol=tol, tag=f"{tag} [arclength landing]")
    landing_error = float(np.max(np.abs(z_land - z_direct)))
    if landing_error > landing_tol:
        raise CondensedSolveError(
            f"{tag}: arclength landing root differs from the direct root: {landing_error:.2e} > {landing_tol:.0e}",
            z_last=z_land, residual_last=landing_error,
        )
    return history


def multistart_near(
    model: CondensedLeontiefModel,
    z_target: np.ndarray,
    *,
    n_starts: int | None = None,
    tol: float = SOLVER_TOL,
    uniqueness_tol: float = 1e-9,
    seed: int = 42,
    tag: str = "",
) -> list[dict[str, Any]]:
    """Near-start multistart: every converged start must land within ``uniqueness_tol`` of ``z_target``.

    Starts: ``omega_0 ~ N(0, 0.15^2)``, ``Y_hat_0 = 1 + N(0, 0.05^2)`` floored at
    0.2; eight starts up to 11 sectors and six beyond (the IO engine used
    eight only at exactly 11 sectors and six otherwise; a documented deviation).
    Starts where Newton fails are recorded, not counted as evidence.

    Raises
    ------
    MultipleEquilibria
        A converged start landed farther than ``uniqueness_tol`` (``roots`` attached).
    """
    n = model.n_countries
    if n_starts is None:
        n_starts = 8 if model.n_sectors <= 11 else 6
    rng = np.random.default_rng(seed)
    z_target = np.asarray(z_target, dtype=float)
    results = []
    for i in range(n_starts):
        omega0 = rng.normal(0.0, 0.15, size=n)
        yhat0 = np.maximum(1.0 + rng.normal(0.0, 0.05, size=n), 0.2)
        z0 = np.concatenate([omega0, yhat0])
        try:
            z_root, info = newton(model, z0=z0, tol=tol, max_iter=50,
                                  tag=f"{tag} [multistart start {i + 1}/{n_starts}]")
            dist = float(np.max(np.abs(z_root - z_target)))
            if dist > uniqueness_tol:
                raise MultipleEquilibria(
                    f"{tag}: start {i + 1} found a distinct equilibrium (distance {dist:.2e} > {uniqueness_tol:.0e})",
                    roots=[z_target, z_root],
                )
            results.append({"start": i + 1, "converged": True, "dist": dist, "iterations": info["iterations"]})
        except EquilibriumNotFound as exc:
            results.append({"start": i + 1, "converged": False, "err": str(exc)})
    return results


def nested_fallback(
    model: CondensedLeontiefModel,
    z0: np.ndarray | None = None,
    *,
    tol: float = SOLVER_TOL,
    max_iter: int = 50,
    tag: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the affine income system for ``Y_hat`` given wages, then polish with Newton.

    Given ``w``, the ``N`` income residuals are affine in ``Y_hat``; ``N + 1``
    state evaluations at the cached prices recover the matrix and the right-hand
    side. Newton then starts from ``(omega, Y_hat(w))``.
    """
    n = model.n_countries
    omega = np.zeros(n) if z0 is None else np.array(np.asarray(z0, dtype=float)[:n], copy=True)
    w_cur = np.exp(omega)
    p_cur = model.prices(w_cur)
    zh0 = np.concatenate([omega, np.zeros(n, dtype=float)])
    st0 = model.state(zh0, p=p_cur)
    b_vec = -st0.r_Y * model.Y0
    M_mat = np.empty((n, n), dtype=float)
    for j in range(n):
        zh_j = zh0.copy()
        zh_j[n + j] = 1.0
        st_j = model.state(zh_j, p=p_cur)
        M_mat[:, j] = (st_j.r_Y * model.Y0) + b_vec
    try:
        yhat = la.solve(M_mat, b_vec)
    except la.LinAlgError as exc:
        raise EquilibriumNotFound(f"{tag}: singular affine income system in the nested fallback",
                                  z_last=zh0) from exc
    return newton(model, z0=np.concatenate([omega, yhat]), tol=tol, max_iter=max_iter, tag=f"{tag} [nested]")


def solve_condensed(
    calib: CondensedCalibration,
    wedges: TariffWedges,
    *,
    closure: str = "factor",
    numeraire: str = "factor",
    fd_rule: str = "absorption",
    inventories: str = "tariffed",
    method: str = "continuation",
    tol: float = SOLVER_TOL,
    walras_tol: float = WALRAS_TOL,
    max_iter: int = 40,
    run_arclength: bool = False,
    run_multistart: bool = False,
    certify: bool = True,
    certificate_tol: float = CERTIFICATE_TOL,
    existence_margin: float = EXISTENCE_MARGIN,
    z0: np.ndarray | None = None,
    tag: str = "",
) -> CondensedEquilibriumResult:
    """Solve one tariff experiment and certify it.

    Steps: (1) existence gate on ``rho(B_tau)``; (2) solve by ``method``
    (``"continuation"``: direct Newton, then continuation, then the nested
    fallback on :class:`ContinuationFailure`; ``"newton"``: plain Newton from
    ``z0``); (3) optional pseudo-arclength fold monitoring with a landing check
    and near-start multistart; (4) the independent ten-block raw-flow
    certificate at ``certificate_tol``.

    Parameters
    ----------
    calib : CondensedCalibration
    wedges : TariffWedges
    closure, numeraire, fd_rule, inventories : str
        Model options; see :class:`~puremacro.trade.condensed.model.CondensedLeontiefModel`.
    method : {"continuation", "newton"}
    tol, walras_tol, max_iter : float, float, int
        Newton stopping rule (``1e-12``, ``1e-10``, 40).
    run_arclength, run_multistart : bool
        Extra evidence (fold monitors, local uniqueness); both are evidence, not proofs.
    certify : bool
        Evaluate the raw-flow certificate (strict, ``cell_floor = 1`` in the
        table's units). ``certify=False`` is a diagnostic shortcut: the result
        then has ``certificate=None`` and reports ``passed=False`` and
        ``certified=False`` although Newton converged. Call
        :func:`~puremacro.trade.condensed.certify.certify_raw_flows` directly
        for another ``cell_floor``.
    z0 : np.ndarray, optional
        Starting point for ``method="newton"``.

    Returns
    -------
    CondensedEquilibriumResult
        Always with ``converged=True`` and, when ``certify`` is True, a passing
        certificate (``passed=True``): every failure raises a
        :class:`~puremacro.trade.condensed.errors.CondensedSolveError` subclass.
        When continuation stalls, the nested fallback's diagnostic tag and its
        stage record (``continuation_lam_reached``) name the stall.
    """
    t_start = time.time()
    bounds = existence_gate(calib, wedges, margin=existence_margin)
    model = CondensedLeontiefModel(calib, wedges, closure=closure, numeraire=numeraire, fd_rule=fd_rule,
                                   inventories=inventories)
    path: list[dict[str, Any]] = []
    if method == "newton":
        z_start = model.z0 if z0 is None else np.asarray(z0, dtype=float)
        z_conv, info = newton(model, z0=z_start, tol=tol, walras_tol=walras_tol, max_iter=max_iter, tag=tag)
        path.append(_stage_record(1.0, info))
    elif method == "continuation":
        try:
            z_conv, path, model = continuation(calib, wedges, closure=closure, numeraire=numeraire,
                                               fd_rule=fd_rule, inventories=inventories, tol=tol,
                                               walras_tol=walras_tol, max_iter=max_iter, tag=tag)
        except ContinuationFailure as exc:
            stall_tag = f"{tag} [continuation stalled at lambda={exc.lam_reached:.6g}]"
            z_conv, info = nested_fallback(model, tol=tol, max_iter=max_iter, tag=stall_tag)
            path.append(_stage_record(1.0, info, fallback="nested", continuation_lam_reached=exc.lam_reached))
    else:
        raise ValueError(f"unknown solve method {method!r}; use 'continuation' or 'newton'")
    state = model.state(z_conv)
    arc: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    if run_arclength:
        arc = arclength(model, z_conv, tol=tol, tag=tag)
    if run_multistart:
        starts = multistart_near(model, z_conv, tol=tol, tag=tag)
    certificate = None
    if certify:
        certificate = certify_raw_flows(calib, wedges, state, closure=closure, numeraire=numeraire,
                                        inventories=inventories, tol=certificate_tol, strict=True)
    final = path[-1] if path else {}
    return CondensedEquilibriumResult(
        state=state, certificate=certificate, converged=True,
        iterations=int(final.get("iterations", 0)), jacobians=int(final.get("jacobians", 0)),
        max_scaled_residual=float(np.max(np.abs(state.residual))), walras_residual=float(state.walras_residual),
        path=tuple(path), seconds=time.time() - t_start, calib=calib, wedges=wedges,
        options={"closure": closure, "numeraire": numeraire, "fd_rule": fd_rule, "inventories": inventories,
                 "method": method, "tol": tol, "walras_tol": walras_tol, "max_iter": max_iter,
                 "certificate_tol": certificate_tol, "existence_margin": existence_margin},
        arclength_path=tuple(arc), multistart=tuple(starts), existence=bounds.to_dict(),
        metadata={"model": "condensed_leontief_one_factor", "wedges_sha256": wedges.sha256,
                  "n_unknowns": model.n_vars},
    )


__all__ = [
    "arclength",
    "continuation",
    "existence_gate",
    "multistart_near",
    "nested_fallback",
    "newton",
    "solve_condensed",
]
