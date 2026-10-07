"""Economic Regularization and Accounting Balancing Algorithms for MRIO Tables.

This module provides production-grade matrix regularization and balancing routines
for Multi-Regional Input-Output (MRIO) datasets conforming to the puremacro
Pyodide runtime contract (NumPy and SciPy only).

Core capabilities:
1. Economic Regularization (``regularize_mrio_table``):
   - Phantom output injection (1e-6 M USD) for inactive/zero-output sector nodes.
   - Value-added flooring with dual TLS debit on active nodes:
     ``VA >= max(1e-3 * Y, min(1.0, 0.5 * Y))``. The absolute 1 M USD floor is
     capped at half of the node's output, so flooring never lifts value added
     above gross output (4.3.0 applied ``max(1e-3 * Y, 1.0)``, which pushed VA
     above Y on nodes with Y < 1 M USD).
   - Residual TLS reconciliation ensuring column outlays equal gross output Y
     (< 1e-11 * max(Y)). Without a supplied ``Y``, Y is the row sales, so outlays
     also equal row sales; a supplied ``Y`` that differs from row sales is
     reported (warning by default), because the sales identity then fails.
   - Collatz-Wielandt spectral radius verification (certified rho(B) < 0.999).
2. Biproportional Matrix Balancing:
   - ``balance_ras``: Classical iterative Bregman row/column scaling.
   - ``balance_gras``: Generalized RAS for matrices containing negative entries.
   - ``balance_quadratic``: Constrained weighted least-squares quadratic balancing.
   RAS and GRAS count a margin deviation up to ``max(tol, 1e-12 * largest
   target)`` as converged, warn when they stop above it and can return a
   convergence record (``return_info=True``).
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple
import warnings

import numpy as np
import scipy.linalg


_OUTPUT_MISMATCH_POLICIES = ("warn", "raise", "row_sales", "ignore")


_EIGVALS_MAX_N = 256
"""Largest irreducible block whose unresolved bounds are completed with a dense eigensolve."""

_DENSE_RESOLVENT_MAX_N = 1024
"""Largest irreducible block whose unresolved bounds are refined by dense inverse iteration."""

_BALANCE_RELATIVE_FLOOR = 1e-12
"""Margin deviation, relative to the largest target, that RAS/GRAS treat as met.

Row and column sums of floating-point matrices cannot match targets of size
``T`` more closely than a few ``eps * T``; an absolute ``tol`` below that is
unattainable, so the convergence flag and the warning use
``max(tol, 1e-12 * max(1, max|u|, max|v|))``. The stopping rule keeps the
absolute ``tol`` (as in 4.3.0), so the returned matrices are unchanged.
"""


def _cw_power_iteration(
    B: np.ndarray, x: np.ndarray, shift: float, max_iter: int, tol: float,
    lower: float, upper: float,
) -> tuple[np.ndarray, float, float]:
    """Shifted power iteration ``x <- (B + shift I) x`` accumulating Collatz-Wielandt bounds.

    ``lower``/``upper`` enter as already certified bounds and are only
    tightened (every positive iterate certifies its own ratio bounds).
    """
    for _ in range(max_iter):
        Bx = B @ x
        ratios = Bx / x
        lower = max(lower, float(np.min(ratios)))
        upper = min(upper, float(np.max(ratios)))
        if upper - lower <= tol * max(1.0, upper):
            break
        nxt = Bx + shift * x
        x = np.maximum(nxt / np.max(nxt), np.finfo(float).tiny)
    return x, lower, upper


def _resolvent_refine(
    B: np.ndarray, lam: float, lower: float, upper: float, tol: float,
    n_rounds: int = 2, n_solves: int = 3,
) -> tuple[float, float, np.ndarray | None]:
    """Tighten Collatz-Wielandt bounds with the positive vector ``(lam I - B)^{-1} 1``.

    For ``lam > rho(B)`` and irreducible nonnegative ``B`` the resolvent
    ``(lam I - B)^{-1} = sum_k B^k / lam^{k+1}`` is strictly positive, so every
    solve yields a positive test vector; repeated solves are inverse iteration
    and damp each non-Perron component by ``(lam - rho) / |lam - lambda_i|``.
    After the first round ``lam`` is moved just above the improved upper bound
    (which is >= rho) and the matrix refactored. The bounds are certified by the
    ratios ``(B v)_i / v_i`` at a positive vector, never by the eigenvalue used
    to choose ``lam``; a solve that loses positivity is discarded.
    """
    n = B.shape[0]
    best = None
    for round_index in range(n_rounds):
        if round_index:
            if upper - lower <= tol * max(1.0, upper):
                break
            lam = upper * (1.0 + 1e-9) + 1e-15
        try:
            factor = scipy.linalg.lu_factor(lam * np.eye(n) - B, check_finite=False)
        except (scipy.linalg.LinAlgError, ValueError):
            break
        vec = np.ones(n) if best is None else best
        for _ in range(n_solves):
            try:
                vec = scipy.linalg.lu_solve(factor, vec, check_finite=False)
            except (scipy.linalg.LinAlgError, ValueError):
                break
            if not (np.all(np.isfinite(vec)) and np.all(vec > 0)):
                break
            vec = vec / np.max(vec)
            ratios = (B @ vec) / vec
            lower = max(lower, float(np.min(ratios)))
            upper = min(upper, float(np.max(ratios)))
            best = vec
            if upper - lower <= tol * max(1.0, upper):
                break
    return lower, upper, best


def compute_spectral_radius(
    B: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-12,
    *,
    shift: float = 1e-3,
) -> tuple[float, float, float]:
    """Spectral radius rho(B) of a nonnegative matrix with Collatz-Wielandt bounds.

    By the Collatz-Wielandt theorem, for any nonnegative matrix ``B`` and any
    strictly positive vector ``x``::

        min_i (B x)_i / x_i  <=  rho(B)  <=  max_i (B x)_i / x_i.

    The test vector comes from the shifted power iteration
    ``x <- (B + shift * I) x / max(...)`` started at ``x = 1``. The shift adds a
    positive diagonal, which makes every irreducible block primitive (removes
    periodicity) without changing eigenvectors. Each non-Perron component
    contracts by ``|lambda_i + shift| / (rho + shift)`` per step: for primitive
    blocks a small shift keeps this close to ``|lambda_2| / rho`` (up to 4.3.0
    the shift was the maximum row sum of ``B``, which on real input-output
    tables pushed the ratio to ~0.99 and left the bounds unresolved after
    hundreds of iterations), whereas periodic or nearly periodic blocks
    (``lambda_2`` near ``-rho``) contract only at ``(rho - shift) / (rho +
    shift)`` and need a shift comparable to ``rho``.

    Reducible matrices are split into strongly connected diagonal blocks, whose
    spectra make up the spectrum of ``B``; each block is bounded separately and
    the maxima are returned, so isolated or inactive nodes cannot pin the lower
    bound to zero. When a block's bounds are still wider than ``tol`` after
    ``max_iter`` iterations they are completed as follows. Blocks of at most
    1024 rows get inverse iteration with ``(lam I - B)`` (at most two
    factorizations with three solves each): ``lam`` is just above the dense
    eigenvalue estimate for blocks of at most 256 rows, and just above the
    current upper bound for blocks of at most 1024 rows. Larger blocks get a
    second power pass of at most ``max_iter`` steps with the shift equal to the
    current upper bound (which is >= rho), warm-started at the first pass's
    vector, and the two passes' bounds are intersected. The returned bounds are
    always certified by ratios at a positive vector.

    Parameters
    ----------
    B : np.ndarray
        Finite nonnegative square matrix of shape (M, M).
    max_iter : int, default=300
        Maximum number of power iterations per strongly connected block (per
        pass: a block of more than 1024 rows can take a second pass).
    tol : float, default=1e-12
        Stop when ``upper - lower <= tol * max(1, upper)``.
    shift : float, default=1e-3
        Positive diagonal shift of the power iteration (keyword-only). The
        default matches the ``eps`` of ``check_hawkins_simon_viability`` and of
        the research code's ``collatz_wielandt_bound``.

    Returns
    -------
    tuple[float, float, float]
        ``(rho, collatz_wielandt_lower, collatz_wielandt_upper)``. ``rho`` is the
        dense eigenvalue for small unresolved blocks and otherwise the Rayleigh
        quotient at the final test vector clipped into ``[lower, upper]``; use
        the bounds, not ``rho``, for any certification.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or not np.all(np.isfinite(B)) or np.any(B < 0):
        raise ValueError("B must be a finite nonnegative square matrix")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    shift = float(shift)
    if not np.isfinite(shift) or shift <= 0.0:
        raise ValueError("shift must be a finite positive number")
    n = B.shape[0]
    if n == 0 or not np.any(B):
        return 0.0, 0.0, 0.0
    if n == 1:
        value = float(B[0, 0])
        return value, value, value
    # Reducible matrices have the spectra of their strongly connected diagonal
    # blocks. Isolated/inactive nodes must not pin the lower bound to zero.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    if np.all(B > 0):
        # A strictly positive dense matrix is already irreducible; avoid a
        # second, sparse copy of potentially millions of nonzero entries.
        count, labels = 1, None
    else:
        count, labels = connected_components(csr_matrix(B), directed=True, connection="strong")
    if count > 1:
        blocks = []
        for label in range(count):
            idx = np.flatnonzero(labels == label)
            blocks.append(compute_spectral_radius(B[np.ix_(idx, idx)], max_iter=max_iter, tol=tol, shift=shift))
        return tuple(float(max(v[i] for v in blocks)) for i in range(3))
    x, lower, upper = _cw_power_iteration(B, np.ones(n), shift, max_iter, tol, 0.0, float("inf"))
    if upper - lower > tol * max(1.0, upper) and n > _DENSE_RESOLVENT_MAX_N:
        # Periodic or nearly periodic block (lambda_2 near -rho): the small
        # shift contracts that component at (rho - shift) / (rho + shift) only.
        # A shift equal to the upper bound (>= rho) contracts it at
        # (upper - rho) / (upper + rho); the bounds of both passes are valid,
        # so they are intersected.
        x, lower, upper = _cw_power_iteration(B, x, max(upper, shift), max_iter, tol, lower, upper)
    Bx = B @ x
    rho = float(np.dot(x, Bx) / np.dot(x, x))
    if upper - lower > tol * max(1.0, upper):
        if n <= _EIGVALS_MAX_N:
            rho = float(np.max(np.abs(scipy.linalg.eigvals(B))))
            lam = rho + max(1e-10, abs(rho) * 1e-6)
            lower, upper, _ = _resolvent_refine(B, lam, lower, upper, tol)
        elif n <= _DENSE_RESOLVENT_MAX_N:
            # upper >= rho(B) for any positive x, so lam > rho and the resolvent is positive.
            lam = upper + max(1e-12, abs(upper) * 1e-6)
            lower, upper, vec = _resolvent_refine(B, lam, lower, upper, tol)
            if vec is not None:
                rho = float(np.dot(vec, B @ vec) / np.dot(vec, vec))
    rho = float(min(max(rho, lower), upper))
    return rho, lower, upper


def spectral_radius(B: np.ndarray) -> float:
    """Return a numerical spectral estimate; use compute_spectral_radius for bounds."""
    rho, _, cw_upper = compute_spectral_radius(B)
    if abs(cw_upper - rho) < 1e-4:
        return max(rho, cw_upper)
    return rho


def regularize_mrio_table(
    Z: np.ndarray,
    F: np.ndarray,
    VA: np.ndarray,
    TLS: np.ndarray,
    Y: np.ndarray | None = None,
    floor_output: float = 1e-6,
    floor_va_ratio: float = 1e-3,
    floor_va_abs: float = 1.0,
    *,
    tau: np.ndarray | None = None,
    spectral_tol: float = 1e-3,
    n_countries: int | None = None,
    n_sectors: int | None = None,
    floor_va_max_share: float = 0.5,
    output_mismatch: str = "warn",
    return_report: bool = False,
) -> (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
      | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]):
    """Applies phantom output injection, VA flooring with dual TLS debit, and residual TLS reconciliation.

    Steps, in order:

    1. Output. Without ``Y``, gross output is the row sales
       ``Y_i = sum_j Z_ij + sum_f F_if``. A supplied ``Y`` is compared with the
       row sales; a gap larger than ``1e-9 * max(1, max|Y|)`` is handled by
       ``output_mismatch``.
    2. Phantom output. Nodes with ``Y_i < floor_output`` receive
       ``floor_output`` in their own country's first final-use column, in value
       added and in ``Y``.
    3. Value-added floor with dual TLS debit. For active nodes the floor is
       ``max(floor_va_ratio * Y, min(floor_va_abs, floor_va_max_share * Y))``;
       for phantom nodes it is ``floor_output``. The shortfall is added to
       value added (first row of a 2-D ``VA``) and debited from TLS. With the
       defaults a floored node ends with ``VA <= 0.5 * Y``, so flooring never
       makes value added exceed gross output.
    4. Residual TLS. ``TLS_j = Y_j - sum_i Z_ij - VA_j`` so column outlays
       equal ``Y`` to ``1e-11 * max(Y)``.
    5. Productivity. The Collatz-Wielandt bounds of ``B = tau * Z / Y`` must
       certify ``rho(B) < 1 - spectral_tol``: a certified violation raises
       "non-productive", bounds that straddle the threshold raise "unresolved".

    Parameters
    ----------
    Z : np.ndarray
        Intermediate transaction matrix of shape (M, M).
    F : np.ndarray
        Final demand matrix of shape (M, C * K_F) or (M, C, K_F).
    VA : np.ndarray
        Primary factor payments of shape (M,) or (K_VA, M).
    TLS : np.ndarray
        Net taxes less subsidies on production of shape (M,).
    Y : np.ndarray, optional
        Gross sectoral output of shape (M,). If None, evaluated from row sales.
        A supplied ``Y`` is used as given for the TLS residual and for
        ``A = Z / Y``; if it differs from row sales the returned table satisfies
        the outlays identity but not the sales identity (see ``output_mismatch``).
    floor_output : float, default=1e-6
        Phantom output injection threshold in Million USD.
    floor_va_ratio : float, default=1e-3
        Minimum ratio of value added to gross output (always applied).
    floor_va_abs : float, default=1.0
        Minimum absolute value added in Million USD, applied only up to
        ``floor_va_max_share * Y`` of the node.
    tau : np.ndarray, optional
        Gross tariff wedge matrix for spectral radius evaluation (B_tau = tau * A).
    spectral_tol : float, default=1e-3
        Required margin below 1.0 for spectral radius (rho < 1.0 - spectral_tol = 0.999).
    n_countries : int, optional
        Number of economies C.
    n_sectors : int, optional
        Number of industrial sectors S.
    floor_va_max_share : float, default=0.5
        Cap, as a share of the node's output, on the absolute floor
        ``floor_va_abs`` (keyword-only). ``1.0`` allows the absolute floor up
        to the full output; values above 1 are rejected.
    output_mismatch : {"warn", "raise", "row_sales", "ignore"}, default "warn"
        What to do when a supplied ``Y`` differs from row sales (keyword-only):
        ``"warn"`` keeps ``Y`` and emits a ``RuntimeWarning``; ``"raise"``
        raises ``ValueError``; ``"row_sales"`` replaces ``Y`` by row sales and
        records the change in the report; ``"ignore"`` keeps ``Y`` silently.
    return_report : bool, default False
        If True (keyword-only), also return a dict recording every adjustment:
        ``output_mismatch`` (policy, node indices, maximum gap),
        ``phantom_nodes`` (indices), ``va_floor`` (node indices, value added
        before and after, gross output, floor and which term bound), and
        ``spectral`` (rho estimate, certified bounds, threshold).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (Z_clean, F_clean, VA_clean, TLS_clean, Y_clean), followed by the report
        dict when ``return_report=True``.
    """
    if output_mismatch not in _OUTPUT_MISMATCH_POLICIES:
        raise ValueError(f"output_mismatch must be one of {_OUTPUT_MISMATCH_POLICIES}, got {output_mismatch!r}")
    for name, value, upper_limit in (("floor_va_ratio", floor_va_ratio, 1.0),
                                     ("floor_va_max_share", floor_va_max_share, 1.0)):
        if not (np.isfinite(value) and 0.0 <= value <= upper_limit):
            raise ValueError(f"{name} must lie in [0, {upper_limit}], got {value!r}")
    if not (np.isfinite(floor_va_abs) and floor_va_abs >= 0.0):
        raise ValueError(f"floor_va_abs must be a finite nonnegative number, got {floor_va_abs!r}")
    Z_clean = np.array(Z, dtype=np.float64, copy=True)
    F_clean = np.array(F, dtype=np.float64, copy=True)
    VA_clean = np.array(VA, dtype=np.float64, copy=True)
    TLS_clean = np.array(TLS, dtype=np.float64, copy=True)

    M = Z_clean.shape[0]
    if Z_clean.shape != (M, M):
        raise ValueError(f"Z must be square (M, M), got shape {Z_clean.shape}")
    if F_clean.shape[0] != M:
        raise ValueError(f"F rows ({F_clean.shape[0]}) do not match Z rows ({M})")

    is_va_2d = (VA_clean.ndim == 2)
    if is_va_2d:
        if VA_clean.shape[1] != M:
            raise ValueError(f"VA columns ({VA_clean.shape[1]}) do not match Z ({M})")
    else:
        if VA_clean.shape[0] != M:
            raise ValueError(f"VA length ({VA_clean.shape[0]}) does not match Z ({M})")

    if TLS_clean.shape != (M,):
        raise ValueError(f"TLS shape ({TLS_clean.shape}) must be ({M},)")

    # Deduce dimensions (C, S, K_F)
    is_f_3d = (F_clean.ndim == 3)
    if is_f_3d:
        C = F_clean.shape[1]
        K_F = F_clean.shape[2]
        S = M // C if C > 0 else M
    elif n_countries is not None and n_sectors is not None and n_countries * n_sectors == M:
        C = n_countries
        S = n_sectors
        K_F = F_clean.shape[1] // C if C > 0 else F_clean.shape[1]
    else:
        # Check standard known configurations
        known_dims = [
            (77, 45), (77, 11), (46, 64), (44, 56), (189, 26), (49, 163), (49, 200)
        ]
        matched = False
        for kc, ks in known_dims:
            if kc * ks == M:
                C = kc
                S = ks
                K_F = F_clean.shape[1] // C if (F_clean.shape[1] % C == 0) else 1
                matched = True
                break
        if not matched:
            C = 1
            S = M
            K_F = F_clean.shape[1]

    # Gross output baseline from row sales
    if is_f_3d:
        f_row_sum = np.sum(F_clean, axis=(1, 2))
    else:
        f_row_sum = np.sum(F_clean, axis=1)

    row_sales = np.sum(Z_clean, axis=1) + f_row_sum
    mismatch_record: dict[str, Any] = {"policy": output_mismatch, "nodes": [], "max_abs": 0.0,
                                       "tolerance": 0.0, "replaced_by_row_sales": False}
    y_from_sales = Y is None
    if Y is not None:
        Y_clean = np.array(Y, dtype=np.float64, copy=True)
        if Y_clean.shape != (M,):
            raise ValueError(f"Y shape ({Y_clean.shape}) must be ({M},)")
        gap = np.abs(row_sales - Y_clean)
        tolerance = 1e-9 * max(1.0, float(np.max(np.abs(Y_clean)))) if M else 0.0
        bad = np.flatnonzero(gap > tolerance)
        mismatch_record.update(nodes=[int(i) for i in bad], tolerance=tolerance,
                               max_abs=float(np.max(gap)) if M else 0.0)
        if bad.size and output_mismatch == "raise":
            raise ValueError(
                f"Supplied Y differs from row sales on {bad.size} node(s) (max gap "
                f"{mismatch_record['max_abs']:.6g} > {tolerance:.3g}); pass Y=None or "
                "output_mismatch='row_sales' to calibrate to transaction row sales."
            )
        if bad.size and output_mismatch == "row_sales":
            Y_clean = row_sales.copy()
            y_from_sales = True
            mismatch_record["replaced_by_row_sales"] = True
        elif bad.size and output_mismatch == "warn":
            warnings.warn(
                f"regularize_mrio_table: supplied Y differs from row sales on {bad.size} node(s) "
                f"(max gap {mismatch_record['max_abs']:.6g}); TLS is reconciled to the supplied Y, "
                "so the returned table balances outlays but not sales. Pass Y=None or "
                "output_mismatch='row_sales' to use row sales, or output_mismatch='ignore'.",
                RuntimeWarning, stacklevel=2,
            )
    else:
        Y_clean = row_sales.copy()

    # -------------------------------------------------------------------------
    # 1. Phantom Output Injection (1e-6 M USD)
    # -------------------------------------------------------------------------
    inactive_mask = (Y_clean < floor_output)
    if np.any(inactive_mask):
        for idx in np.where(inactive_mask)[0]:
            c_idx = idx // S if S > 0 else 0
            # Inject into domestic household final demand
            if is_f_3d:
                c_idx = min(c_idx, C - 1)
                F_clean[idx, c_idx, 0] += floor_output
            else:
                fd_col = min(c_idx * K_F, F_clean.shape[1] - 1)
                F_clean[idx, fd_col] += floor_output

            # Inject into value added
            if is_va_2d:
                VA_clean[0, idx] += floor_output
            else:
                VA_clean[idx] += floor_output

            Y_clean[idx] += floor_output

    # Update gross output to reflect row sales after phantom injection if Y was not provided
    if y_from_sales:
        if is_f_3d:
            f_row_sum = np.sum(F_clean, axis=(1, 2))
        else:
            f_row_sum = np.sum(F_clean, axis=1)
        Y_clean = np.sum(Z_clean, axis=1) + f_row_sum

    # -------------------------------------------------------------------------
    # 2. Value-Added Flooring & Dual TLS Debit
    #    active nodes: VA >= max(ratio * Y, min(abs, max_share * Y))
    # -------------------------------------------------------------------------
    va_totals = np.array(np.sum(VA_clean, axis=0) if is_va_2d else VA_clean, copy=True)
    ratio_floor = floor_va_ratio * Y_clean
    capped_abs_floor = np.minimum(floor_va_abs, floor_va_max_share * np.maximum(Y_clean, 0.0))
    va_floors = np.where(
        inactive_mask,
        floor_output,
        np.maximum(ratio_floor, capped_abs_floor),
    )
    delta_va = np.maximum(0.0, va_floors - va_totals)
    floored = np.flatnonzero(delta_va > 0.0)
    binding = np.where(inactive_mask, "phantom",
                       np.where(ratio_floor >= capped_abs_floor, "ratio",
                                np.where(capped_abs_floor < floor_va_abs, "absolute_capped", "absolute")))
    va_floor_record = {
        "rule": (f"active: VA >= max({floor_va_ratio:g} * Y, min({floor_va_abs:g}, "
                 f"{floor_va_max_share:g} * Y)); phantom: VA >= {floor_output:g}"),
        "nodes": [int(i) for i in floored],
        "value_added_before": [float(va_totals[i]) for i in floored],
        "value_added_after": [float(va_totals[i] + delta_va[i]) for i in floored],
        "gross_output": [float(Y_clean[i]) for i in floored],
        "floor": [float(va_floors[i]) for i in floored],
        "binding": [str(binding[i]) for i in floored],
    }
    if floored.size:
        if is_va_2d:
            VA_clean[0, :] += delta_va
        else:
            VA_clean += delta_va

        # Dual TLS Debit: debit the adjustment from net production taxes (TLS)
        # preserving the total sector outlays identity
        TLS_clean -= delta_va

    # -------------------------------------------------------------------------
    # 3. Residual TLS Reconciliation
    # -------------------------------------------------------------------------
    va_current = np.sum(VA_clean, axis=0) if is_va_2d else VA_clean
    TLS_clean = Y_clean - np.sum(Z_clean, axis=0) - va_current

    # Verify column outlays match gross output Y to < 1e-11 * max(Y)
    max_y = float(np.max(Y_clean)) if Y_clean.size > 0 else 1.0
    outlays = np.sum(Z_clean, axis=0) + va_current + TLS_clean
    reconciliation_err = float(np.max(np.abs(outlays - Y_clean)))
    if reconciliation_err > 1e-11 * max_y:
        raise ValueError(
            f"Residual TLS reconciliation error {reconciliation_err:.2e} "
            f"exceeds tolerance 1e-11 * max(Y)."
        )

    # -------------------------------------------------------------------------
    # 4. Spectral Radius Verification (rho(B) < 0.999)
    # -------------------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(
            Z_clean,
            Y_clean.reshape(1, M),
            out=np.zeros_like(Z_clean),
            where=(Y_clean.reshape(1, M) > 0),
        )

    B = (tau * A) if tau is not None else A
    rho, lower, upper = compute_spectral_radius(B, max_iter=300, tol=1e-10)
    threshold = 1.0 - spectral_tol
    if lower < threshold <= upper:
        raise ValueError(
            f"Spectral viability unresolved: certified bounds [{lower:.8g}, {upper:.8g}] "
            f"straddle threshold {threshold:.8g}; refine the calibration or spectral calculation."
        )
    if lower >= threshold:
        raise ValueError(
            f"Spectral radius rho(B) = {rho:.6f} >= threshold {threshold:.6f}. "
            "Leontief cost system is non-productive."
        )

    if return_report:
        report = {
            "output_source": "row_sales" if y_from_sales else "supplied",
            "output_mismatch": mismatch_record,
            "phantom_nodes": [int(i) for i in np.flatnonzero(inactive_mask)],
            "va_floor": va_floor_record,
            "spectral": {"rho": float(rho), "collatz_wielandt_lower": float(lower),
                         "collatz_wielandt_upper": float(upper), "threshold": float(threshold)},
        }
        return Z_clean, F_clean, VA_clean, TLS_clean, Y_clean, report
    return Z_clean, F_clean, VA_clean, TLS_clean, Y_clean


def _prepare_targets(
    name: str, u: np.ndarray, v: np.ndarray, rescale_targets: bool
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Validate row/column targets and reconcile their totals.

    Row and column targets of a balanced matrix must have equal totals. When
    they differ and ``rescale_targets`` is True the column targets are scaled
    by ``sum(u) / sum(v)``; a relative gap above ``1e-9`` also emits a
    ``RuntimeWarning`` because the caller's column targets are then not met.
    """
    u_target = np.array(u, dtype=np.float64, copy=True)
    v_target = np.array(v, dtype=np.float64, copy=True)
    sum_u = float(np.sum(u_target))
    sum_v = float(np.sum(v_target))
    if abs(sum_v) <= 1e-12 and abs(sum_u) > 1e-12:
        raise ValueError("Column target sums to zero while row target is non-zero")
    scale = 1.0
    rescaled = False
    # The 4.3.0 rule (absolute gap above 1e-12) decides whether to rescale, so
    # the balanced matrices are unchanged; only gaps above 1e-9 relative warn.
    if abs(sum_u - sum_v) > 1e-12 and abs(sum_v) > 1e-12:
        if rescale_targets:
            scale = sum_u / sum_v
            v_target = v_target * scale
            rescaled = True
            if abs(sum_u - sum_v) > 1e-9 * max(abs(sum_u), abs(sum_v)):
                warnings.warn(
                    f"{name}: row targets sum to {sum_u:.12g} but column targets sum to {sum_v:.12g}; "
                    f"column targets were rescaled by {scale:.12g} (pass rescale_targets=False to keep them).",
                    RuntimeWarning, stacklevel=3,
                )
    return u_target, v_target, scale, rescaled


def _balance_tolerance(tol: float, u_target: np.ndarray, v_target: np.ndarray) -> tuple[float, float]:
    """Effective margin tolerance ``max(tol, 1e-12 * scale)`` and the margin scale.

    ``scale = max(1, max|u|, max|v|)``. Below ``1e-12 * scale`` the margins
    are at floating-point resolution, so a smaller absolute ``tol`` cannot be
    met and must not be reported as non-convergence.
    """
    scale = max(1.0, float(np.max(np.abs(u_target), initial=0.0)),
                float(np.max(np.abs(v_target), initial=0.0)))
    return max(float(tol), _BALANCE_RELATIVE_FLOOR * scale), scale


def _balance_info(
    Z: np.ndarray, u_target: np.ndarray, v_target: np.ndarray, iterations: int, tol: float,
    scale: float, rescaled: bool,
) -> dict[str, Any]:
    row_dev = float(np.max(np.abs(np.sum(Z, axis=1) - u_target), initial=0.0))
    col_dev = float(np.max(np.abs(np.sum(Z, axis=0) - v_target), initial=0.0))
    max_dev = max(row_dev, col_dev)
    effective_tol, margin_scale = _balance_tolerance(tol, u_target, v_target)
    return {
        "converged": bool(np.isfinite(max_dev) and max_dev <= effective_tol),
        "iterations": int(iterations),
        "max_row_deviation": row_dev,
        "max_col_deviation": col_dev,
        "max_deviation": max_dev,
        "relative_deviation": max_dev / margin_scale,
        "tol": float(tol),
        "effective_tol": effective_tol,
        "margin_scale": margin_scale,
        "column_target_scale": float(scale),
        "targets_rescaled": bool(rescaled),
    }


def _warn_unconverged(name: str, info: dict[str, Any]) -> None:
    if not info["converged"]:
        warnings.warn(
            f"{name} stopped after {info['iterations']} iterations with margin deviation "
            f"{info['max_deviation']:.3e} ({info['relative_deviation']:.1e} of the largest target) > "
            f"tolerance {info['effective_tol']:.1e}; the targets may be infeasible for the prior's "
            "sign/zero pattern. The returned matrix does not match the margins.",
            RuntimeWarning, stacklevel=3,
        )


def balance_ras(
    Z0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
    *,
    rescale_targets: bool = True,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Classical biproportional matrix balancing (RAS algorithm).

    Scales non-negative matrix Z0 to match target row sums u and column sums v
    via iterative Bregman projections:
        Z_{ij}^{(k+1)} = r_i Z_{ij}^0 s_j

    Rows (columns) with a zero target are scaled to zero. Targets must be
    nonnegative. If ``sum(u) != sum(v)`` the column targets are rescaled to
    ``sum(u)`` (``rescale_targets=True``, the default; a relative gap above
    1e-9 warns). Iteration stops when the largest margin deviation is at most
    ``tol`` or after ``max_iter`` sweeps. The result counts as converged when
    that deviation is at most ``max(tol, 1e-12 * max(1, max|u|, max|v|))``:
    the relative term is the floating-point resolution of the margins, which
    an absolute ``tol`` cannot beat on large tables. When the margins are off
    by more than that after ``max_iter`` sweeps (for example because the zero
    pattern of ``Z0`` makes the targets infeasible) a ``RuntimeWarning`` is
    emitted.

    Parameters
    ----------
    Z0 : np.ndarray
        Prior non-negative matrix of shape (m, n).
    u : np.ndarray
        Target row marginals of shape (m,).
    v : np.ndarray
        Target column marginals of shape (n,).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-10
        Absolute stopping tolerance on the largest margin deviation; the
        convergence flag accepts ``max(tol, 1e-12 * max(1, max|u|, max|v|))``.
    rescale_targets : bool, default=True
        Rescale ``v`` to the total of ``u`` when the totals differ (keyword-only).
    return_info : bool, default=False
        If True (keyword-only) return ``(Z, info)`` where ``info`` has
        ``converged``, ``iterations``, ``max_row_deviation``,
        ``max_col_deviation``, ``max_deviation``, ``relative_deviation``
        (``max_deviation / margin_scale``), ``tol``, ``effective_tol``,
        ``margin_scale``, ``column_target_scale`` and ``targets_rescaled``.

    Returns
    -------
    np.ndarray
        Balanced non-negative matrix (and the info dict when ``return_info``).
    """
    Z = np.array(Z0, dtype=np.float64, copy=True)

    if np.any(Z < 0):
        raise ValueError("balance_ras requires non-negative entries in Z0. Use balance_gras for negative entries.")
    if np.any(np.asarray(u, dtype=float) < 0) or np.any(np.asarray(v, dtype=float) < 0):
        raise ValueError("balance_ras requires non-negative target margins. Use balance_gras for signed targets.")

    u_target, v_target, scale, rescaled = _prepare_targets("balance_ras", u, v, rescale_targets)
    if abs(float(np.sum(u_target))) <= 1e-12 and abs(float(np.sum(v_target))) <= 1e-12:
        Z = np.zeros_like(Z)
        info = _balance_info(Z, u_target, v_target, 0, tol, scale, rescaled)
        return (Z, info) if return_info else Z

    iterations = 0
    for iterations in range(1, max_iter + 1):
        # 1. Row scaling (zero targets zero the row)
        row_sums = np.sum(Z, axis=1)
        r_step = np.ones_like(u_target)
        mask_r = row_sums > 0
        r_step[mask_r] = u_target[mask_r] / row_sums[mask_r]
        Z *= r_step[:, None]

        # 2. Column scaling (zero targets zero the column)
        col_sums = np.sum(Z, axis=0)
        s_step = np.ones_like(v_target)
        mask_s = col_sums > 0
        s_step[mask_s] = v_target[mask_s] / col_sums[mask_s]
        Z *= s_step[None, :]

        # Convergence test
        curr_row = np.sum(Z, axis=1)
        curr_col = np.sum(Z, axis=0)
        max_dev = max(
            float(np.max(np.abs(curr_row - u_target))),
            float(np.max(np.abs(curr_col - v_target))),
        )
        if max_dev <= tol:
            break

    info = _balance_info(Z, u_target, v_target, iterations, tol, scale, rescaled)
    _warn_unconverged("balance_ras", info)
    return (Z, info) if return_info else Z


def balance_gras(
    Z0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
    *,
    rescale_targets: bool = True,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Generalized RAS (GRAS) balancing for matrices with positive and negative entries.

    Decomposes Z0 = P0 - N0 with P0 >= 0 and N0 >= 0, and updates multipliers (r, s)
    via closed-form quadratic roots:
        Z_{ij} = r_i P0_{ij} s_j - r_i^{-1} N0_{ij} s_j^{-1}

    If ``sum(u) != sum(v)`` the column targets are rescaled to ``sum(u)``
    (``rescale_targets=True``, the default; a relative gap above 1e-9 warns).
    Iteration stops when the largest margin deviation is at most ``tol`` or
    after ``max_iter`` sweeps; the result counts as converged when that
    deviation is at most ``max(tol, 1e-12 * max(1, max|u|, max|v|))`` (see
    :func:`balance_ras`). A ``RuntimeWarning`` is emitted when the margins are
    off by more than that after ``max_iter`` sweeps, which happens when a row
    or column cannot reach its target with its sign pattern (e.g. a positive
    target on a row whose prior entries are all negative).

    Parameters
    ----------
    Z0 : np.ndarray
        Prior real matrix of shape (m, n).
    u : np.ndarray
        Target row marginals of shape (m,).
    v : np.ndarray
        Target column marginals of shape (n,).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-10
        Absolute stopping tolerance on the largest margin deviation; the
        convergence flag accepts ``max(tol, 1e-12 * max(1, max|u|, max|v|))``.
    rescale_targets : bool, default=True
        Rescale ``v`` to the total of ``u`` when the totals differ (keyword-only).
    return_info : bool, default=False
        If True (keyword-only) return ``(Z, info)`` with the same keys as
        :func:`balance_ras`.

    Returns
    -------
    np.ndarray
        Balanced real matrix (and the info dict when ``return_info``).
    """
    Z_orig = np.array(Z0, dtype=np.float64, copy=True)
    u_target, v_target, scale, rescaled = _prepare_targets("balance_gras", u, v, rescale_targets)

    m, n = Z_orig.shape

    P0 = np.maximum(Z_orig, 0.0)
    N0 = np.maximum(-Z_orig, 0.0)

    r = np.ones(m, dtype=np.float64)
    s = np.ones(n, dtype=np.float64)
    Z = Z_orig.copy()

    iterations = 0
    for iterations in range(1, max_iter + 1):
        # 1. Update r given current s
        p_row = P0 @ s
        n_row = N0 @ (1.0 / s)

        rad_r = np.sqrt(u_target ** 2 + 4.0 * p_row * n_row)
        mask_p = p_row > 1e-15
        mask_only_n = (~mask_p) & (n_row > 1e-15)

        r_new = np.ones_like(r)
        r_new[mask_p] = (u_target[mask_p] + rad_r[mask_p]) / (2.0 * p_row[mask_p])
        r_new[mask_only_n] = -n_row[mask_only_n] / np.where(u_target[mask_only_n] != 0, u_target[mask_only_n], -1.0)
        r = np.maximum(r_new, 1e-14)

        # 2. Update s given current r
        p_col = r @ P0
        n_col = (1.0 / r) @ N0

        rad_s = np.sqrt(v_target ** 2 + 4.0 * p_col * n_col)
        mask_p_col = p_col > 1e-15
        mask_only_n_col = (~mask_p_col) & (n_col > 1e-15)

        s_new = np.ones_like(s)
        s_new[mask_p_col] = (v_target[mask_p_col] + rad_s[mask_p_col]) / (2.0 * p_col[mask_p_col])
        s_new[mask_only_n_col] = -n_col[mask_only_n_col] / np.where(v_target[mask_only_n_col] != 0, v_target[mask_only_n_col], -1.0)
        s = np.maximum(s_new, 1e-14)

        # Evaluate current matrix and check margin convergence
        Z = r[:, None] * P0 * s[None, :] - (1.0 / r)[:, None] * N0 * (1.0 / s)[None, :]
        curr_row = np.sum(Z, axis=1)
        curr_col = np.sum(Z, axis=0)

        max_dev = max(
            float(np.max(np.abs(curr_row - u_target))),
            float(np.max(np.abs(curr_col - v_target))),
        )
        if max_dev <= tol:
            break

    info = _balance_info(Z, u_target, v_target, iterations, tol, scale, rescaled)
    _warn_unconverged("balance_gras", info)
    return (Z, info) if return_info else Z


def balance_quadratic(
    Z0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Constrained weighted least-squares quadratic matrix balancing.

    Solves:
        min_Z 1/2 sum_{i,j} w_{ij} (Z_{ij} - Z_{ij}^0)^2
        s.t.  sum_j Z_{ij} = u_i  for all i,
              sum_i Z_{ij} = v_j  for all j.

    Parameters
    ----------
    Z0 : np.ndarray
        Prior matrix of shape (m, n).
    u : np.ndarray
        Target row marginals of shape (m,).
    v : np.ndarray
        Target column marginals of shape (n,).
    weights : np.ndarray, optional
        Positive weight matrix of shape (m, n). If None, uniform unit weights
        are used, yielding the closed-form projection.

    Returns
    -------
    np.ndarray
        Balanced matrix satisfying row and column marginals.
    """
    Z = np.array(Z0, dtype=np.float64, copy=True)
    u_target = np.array(u, dtype=np.float64, copy=True)
    v_target = np.array(v, dtype=np.float64, copy=True)

    m, n = Z.shape
    sum_u = float(np.sum(u_target))
    sum_v = float(np.sum(v_target))
    if abs(sum_v) <= 1e-12 and abs(sum_u) > 1e-12:
        raise ValueError("Column target sums to zero while row target is non-zero")
    if abs(sum_u - sum_v) > 1e-12:
        v_target = v_target * (sum_u / sum_v)

    u0 = np.sum(Z, axis=1)
    v0 = np.sum(Z, axis=0)
    delta_u = u_target - u0
    delta_v = v_target - v0

    if weights is None:
        # Exact closed-form projection under uniform weights
        tot = float(np.sum(delta_u))
        return Z + delta_u[:, None] / n + delta_v[None, :] / m - tot / (m * n)

    # General weighted least squares via normal equations
    W = np.array(weights, dtype=np.float64)
    if W.shape != (m, n):
        raise ValueError(f"Weights shape {W.shape} does not match Z0 shape {(m, n)}")
    if np.any(W <= 0):
        raise ValueError("Weights must be strictly positive.")

    C = 1.0 / W  # Variances
    dr = np.sum(C, axis=1)
    dc = np.sum(C, axis=0)

    # Saddle point system: [diag(dr) C; C^T diag(dc)] [lambda; mu] = [delta_u; delta_v]
    K = np.block([
        [np.diag(dr), C],
        [C.T, np.diag(dc)],
    ])
    rhs = np.concatenate([delta_u, delta_v])
    sol, _, _, _ = scipy.linalg.lstsq(K, rhs)
    lam = sol[:m]
    mu = sol[m:]

    return Z + C * (lam[:, None] + mu[None, :])


def validate_accounting_identities(
    Z: np.ndarray,
    F: np.ndarray,
    VA: np.ndarray,
    TLS: np.ndarray,
    Y: np.ndarray,
    tol: float = 1e-9,
    tau: np.ndarray | None = None,
    spectral_tol: float = 1e-3,
    *,
    floor_output: float = 1e-6,
    floor_va_ratio: float = 1e-3,
    floor_va_abs: float = 1.0,
    floor_va_max_share: float = 0.5,
) -> dict[str, Any]:
    """Validate full Walrasian accounting identities across MRIO transaction matrices.

    Checks (all enter ``valid``; ``abs_tol = tol * max(Y)``):

    1. Sales balance: ``sum_j Z_ij + sum_{c, f} F_{ic}^f == Y_i`` within ``abs_tol``.
    2. Outlays balance: ``sum_i Z_ij + VA_j + TLS_j == Y_j`` within ``abs_tol``.
    3. Global balance: ``|Sales_j - Outlays_j| <= abs_tol``.
    4. Output and value-added floors, the rule :func:`regularize_mrio_table`
       enforces: ``Y_j >= floor_output`` (1e-6) and
       ``VA_j >= max(floor_va_ratio * Y_j, min(floor_va_abs, floor_va_max_share * Y_j))
       - abs_tol`` on every node (phantom nodes, whose ``VA = Y = floor_output``,
       satisfy it). With the defaults this is ``VA >= max(1e-3 * Y, min(1, 0.5 * Y))``.
    5. Productivity: the Collatz-Wielandt upper bound certifies
       ``rho(B_tau) < 1 - spectral_tol`` (the point estimate alone is not enough).

    Returns a dict with ``valid`` and the measured errors, including
    ``min_value_added``, ``value_added_floor_violations`` (count),
    ``value_added_exceeds_output`` (count, informational), the spectral
    estimate and bounds, and ``spectral_certified``.
    """
    Z_arr = np.asarray(Z, dtype=np.float64)
    F_arr = np.asarray(F, dtype=np.float64)
    VA_arr = np.asarray(VA, dtype=np.float64)
    TLS_arr = np.asarray(TLS, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)

    M = Z_arr.shape[0]
    max_y = float(np.max(Y_arr)) if Y_arr.size > 0 else 1.0
    abs_tol = tol * max_y

    if F_arr.ndim == 3:
        f_row_sum = np.sum(F_arr, axis=(1, 2))
    else:
        f_row_sum = np.sum(F_arr, axis=1)

    va_col_sum = np.sum(VA_arr, axis=0) if VA_arr.ndim == 2 else VA_arr

    sales = np.sum(Z_arr, axis=1) + f_row_sum
    outlays = np.sum(Z_arr, axis=0) + va_col_sum + TLS_arr

    sales_err = float(np.max(np.abs(sales - Y_arr)))
    outlays_err = float(np.max(np.abs(outlays - Y_arr)))
    balance_err = float(np.max(np.abs(sales - outlays)))

    va_floor = np.maximum(floor_va_ratio * Y_arr,
                          np.minimum(floor_va_abs, floor_va_max_share * np.maximum(Y_arr, 0.0)))
    va_violations = int(np.count_nonzero(va_col_sum < va_floor - abs_tol))
    output_floor_ok = bool(np.all(Y_arr >= floor_output - 1e-12))

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.divide(
            Z_arr, Y_arr.reshape(1, M), out=np.zeros_like(Z_arr), where=(Y_arr.reshape(1, M) > 0)
        )

    B_tau = (tau * A) if tau is not None else A
    rho, cw_lower, cw_upper = compute_spectral_radius(B_tau)

    bound_threshold = 1.0 - spectral_tol
    spectral_certified = bool(cw_upper < bound_threshold)
    valid = bool(
        sales_err <= abs_tol
        and outlays_err <= abs_tol
        and balance_err <= abs_tol
        and output_floor_ok
        and va_violations == 0
        and spectral_certified
    )

    return {
        "valid": valid,
        "max_sales_error": sales_err,
        "max_outlays_error": outlays_err,
        "max_sales_outlays_error": balance_err,
        "min_gross_output": float(np.min(Y_arr)),
        "min_value_added": float(np.min(va_col_sum)),
        "value_added_floor_violations": va_violations,
        "value_added_exceeds_output": int(np.count_nonzero(va_col_sum > Y_arr + abs_tol)),
        "spectral_radius": float(rho),
        "collatz_wielandt_lower": float(cw_lower),
        "collatz_wielandt_upper": float(cw_upper),
        "spectral_certified": spectral_certified,
    }


__all__ = [
    "compute_spectral_radius",
    "spectral_radius",
    "regularize_mrio_table",
    "balance_ras",
    "balance_gras",
    "balance_quadratic",
    "validate_accounting_identities",
]
