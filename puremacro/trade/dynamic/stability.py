"""Blanchard-Kahn determinacy diagnostic for the dynamic sector-capital model.

Not part of the IO engine: added because the assessment of puremacro's
bundled 77x11 table found that, under the native closure (financial autarky,
wage-indexed transfers) with ``beta = .96``, ``delta = .08`` and ``phi = 2``,
the linearised dynamics have 846 stable roots for 847 predetermined stocks,
with the unstable root ``1.00185`` loading on Ireland. This is a local
property of the linearised closure on that table, not a claim about its
cause; other tables can be indeterminate as well. On the bundled table the
finite-horizon paths then carry a terminal gap that grows with the horizon
(1 percent USA merchandise duty: about 0.02 at 40 dates, 0.07 at 640), and
the admissible shock shrinks as the horizon grows: USA merchandise duties of
1 to 4 percent solve at 40 dates, 3 percent also at 160 dates, while the line
search stalls for 4 percent at 160 dates and for 5 percent at 40 dates, as
installation at IRL:GOV approaches zero on the terminal boundary date. A
line-search failure is evidence, not a proof, that no interior path exists,
and the boundary was not searched exhaustively. This diagnostic is evaluated
before a transition is attempted (``require_determinacy=True``).

Construction. Linearise the date equations ``F(z_{t+1}, z_t, z_{t-1}) = 0``
at a stationary state: ``A_plus dz_{t+1} + A_0 dz_t + A_minus dz_{t-1} = 0``.
Only the capital block of ``z_{t-1}`` enters (it supplies ``K_t``), so with
``s_t = (dz_t, dK_t)`` where ``dK_t`` is the capital block of ``dz_{t-1}``:

.. code-block:: text

    [ A_plus  0 ] s_{t+1} = [ -A_0   -A_minus_K ] s_t
    [ 0       I ]           [  E_K    0         ]

``E_K`` selects the capital block. The generalized eigenvalues of this pencil
(``scipy.linalg.eig(rhs, lhs)``) are classified: stable ``|lambda| < 1 - tol``,
unit ``| |lambda| - 1 | <= tol``, unstable ``|lambda| > 1 + tol`` (infinite
ones included, from the static rows of ``A_plus``). Determinacy requires
``n_stable == N`` (the predetermined stocks) and ``n_unit == 0``. The blocks
come from two batched Jacobian-vector product sweeps (``n_vars`` directions
each); the eigenvalue problem is dense of order ``n_vars + N``.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import linalg

from ._results import DynamicStabilityResult, DynamicSteadyStateResult
from .economy import DynamicEconomy, DynamicTariff

__all__ = ["stability_report"]


def jacobian_blocks(economy: DynamicEconomy, z: np.ndarray,
                    policy: DynamicTariff | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense ``(A_plus, A_0, A_minus)`` of the date equations at ``z`` (internal helper).

    Built from the model's analytic JVPs: with ``D1[j] = dstate/dz_t[e_j]``
    and ``D2[j] = dstate/dz_{t-1}[e_j]`` (two batched sweeps),
    ``A_plus[:, j] = jvp(0, D1[j])``, ``A_0[:, j] = jvp(D1[j], D2[j])`` and
    ``A_minus[:, j] = jvp(D2[j], 0)``. The blocks are the linearisation of
    ``F(z_{t+1}, z_t, z_{t-1})`` at ``z_{t+1} = z_t = z_{t-1} = z``; they
    describe the local dynamics only when ``z`` is stationary, which this
    helper does not check (:func:`stability_report` does). Not exported.
    """
    policy = economy.zero_policy() if policy is None else policy
    z = np.asarray(z, dtype=float)
    if z.shape != (economy.n_vars,):
        raise ValueError("z must be a retained state vector of length n_vars")
    n = economy.n_vars
    state = economy.state(z, z, policy)
    eye = np.eye(n)
    zeros = np.zeros((n, n))
    zs = np.tile(z, (n, 1))
    policies = (policy,) * n
    states = [state] * n
    current = economy.states_jvp(zs, zs, eye, zeros, policies, states)
    lagged = economy.states_jvp(zs, zs, zeros, eye, policies, states)
    a_plus = np.empty((n, n))
    a_zero = np.empty((n, n))
    a_minus = np.empty((n, n))
    for j in range(n):
        d1, d2 = current[j], lagged[j]
        zero_d = {key: np.zeros_like(value) for key, value in d1.items()}
        a_plus[:, j] = economy.jvp_from_states(state, state, zero_d, d1)
        a_zero[:, j] = economy.jvp_from_states(state, state, d1, d2)
        a_minus[:, j] = economy.jvp_from_states(state, state, d2, zero_d)
    return a_plus, a_zero, a_minus


def stability_report(economy: DynamicEconomy, steady: Any, policy: DynamicTariff | None = None, *,
                     tolerance: float = 1e-8, stationarity_tol: float = 1e-8,
                     max_order: int = 6000) -> DynamicStabilityResult:
    """Blanchard-Kahn count at a stationary state.

    Parameters
    ----------
    economy : DynamicEconomy
    steady : DynamicSteadyStateResult or array
        The stationary retained vector (its own policy is used when ``policy``
        is ``None`` and ``steady`` is a result). An array must be passed
        together with the policy it was solved under (the baseline is assumed
        when ``policy`` is ``None``).
    policy : DynamicTariff, optional
    tolerance : float
        Unit-circle band for classifying roots.
    stationarity_tol : float
        The point must be stationary under ``policy``: the maximum absolute
        date residual ``F(z, z, z)`` must not exceed this value (default
        ``1e-8``, the certificate floor), otherwise ``ValueError``. A
        determinacy count at a nonstationary point would be meaningless.
    max_order : int
        Refuse (``ValueError``) dense eigenvalue problems above this order
        (``n_vars + N``); the 77x11 table has order 1848 and takes tens of
        seconds (24 to 36 s at 2 BLAS threads on the reference machine),
        EXIOBASE-scale tables would need tens of gigabytes.

    Raises
    ------
    TypeError
        ``economy`` is not a :class:`DynamicEconomy`.
    ValueError
        Invalid ``tolerance``/``stationarity_tol``, wrong state length, an
        order above ``max_order``, or a point that is not stationary under
        ``policy``.

    Returns
    -------
    DynamicStabilityResult
        Counts, the slowest stable and smallest unstable root, all
        eigenvalues, and the country loading shares of the unstable
        eigenvector with the smallest modulus above one.
    """
    tic = time.perf_counter()
    if not isinstance(economy, DynamicEconomy):
        raise TypeError("economy must be a DynamicEconomy")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(stationarity_tol) or stationarity_tol <= 0:
        raise ValueError("stationarity_tol must be finite and positive")
    if isinstance(steady, DynamicSteadyStateResult):
        z = np.asarray(steady.z, dtype=float)
        if policy is None:
            policy = steady.policy
    else:
        z = np.asarray(steady, dtype=float)
    policy = economy.zero_policy() if policy is None else policy
    if z.shape != (economy.n_vars,):
        raise ValueError("steady must be a retained state vector of length n_vars")
    n, N, nc = economy.n_vars, economy.n, economy.nc
    if n + N > max_order:
        raise ValueError(f"Dense stability problem of order {n + N} exceeds max_order={max_order}; "
                         "raise max_order explicitly or skip the determinacy gate")
    residual = float(np.max(np.abs(economy.equations(z, z, z, policy, policy))))
    if not np.isfinite(residual) or residual > stationarity_tol:
        raise ValueError(f"steady is not stationary under policy {policy.label!r}: max |F(z, z, z)| = "
                         f"{residual:.3g} exceeds stationarity_tol={stationarity_tol:.3g}. Pass the policy the "
                         "state was solved under (or a DynamicSteadyStateResult, which carries it).")
    a_plus, a_zero, a_minus = jacobian_blocks(economy, z, policy)
    k_slice = economy.k_slice
    lhs = np.zeros((n + N, n + N))
    rhs = np.zeros((n + N, n + N))
    lhs[:n, :n] = a_plus
    lhs[n:, n:] = np.eye(N)
    rhs[:n, :n] = -a_zero
    rhs[:n, n:] = -a_minus[:, k_slice]
    rhs[n:, k_slice] = np.eye(N)
    homogeneous, vectors = linalg.eig(rhs, lhs, right=True, homogeneous_eigvals=True, check_finite=False)
    alpha, beta_ = homogeneous[0], homogeneous[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        eigenvalues = np.where(beta_ != 0, alpha / np.where(beta_ != 0, beta_, 1.0), np.inf + 0j)
    modulus = np.abs(eigenvalues)
    infinite = ~np.isfinite(modulus)
    stable = np.isfinite(modulus) & (modulus < 1.0 - tolerance)
    unit = np.isfinite(modulus) & (np.abs(modulus - 1.0) <= tolerance)
    unstable = infinite | (np.isfinite(modulus) & (modulus > 1.0 + tolerance))
    n_stable, n_unit, n_unstable, n_infinite = int(stable.sum()), int(unit.sum()), int(unstable.sum()), int(infinite.sum())
    determinate = bool(n_stable == N and n_unit == 0)
    slowest = float(np.max(modulus[stable])) if n_stable else float("nan")
    finite_unstable = np.isfinite(modulus) & (modulus > 1.0 + tolerance)
    if finite_unstable.any():
        idx = int(np.argmin(np.where(finite_unstable, modulus, np.inf)))
        smallest = float(modulus[idx])
        unstable_eigenvalue = complex(eigenvalues[idx])
        vector = vectors[:, idx]
        energy = np.abs(vector) ** 2
        # Country of every pencil coordinate: w (C), C (C), K_next (N), K_t (N).
        owner = np.r_[np.arange(nc), np.arange(nc), economy.p.country, economy.p.country]
        share = np.bincount(owner, weights=energy, minlength=nc) / max(float(energy.sum()), 1e-300)
        labels = (tuple(f"w:{c}" for c in economy.p.countries) + tuple(f"C:{c}" for c in economy.p.countries)
                  + tuple(f"K:{label}" for label in economy.p.cell_labels)
                  + tuple(f"Kt:{label}" for label in economy.p.cell_labels))
        magnitude = np.abs(vector) / max(float(np.max(np.abs(vector))), 1e-300)
        order = np.argsort(-magnitude)[:12]
        top = tuple((labels[i], float(magnitude[i])) for i in order)
    else:
        smallest, unstable_eigenvalue = float("inf") if n_infinite else float("nan"), None
        share = np.zeros(nc)
        top = ()
    share = np.array(share, dtype=float)
    share.flags.writeable = False
    eigen = np.array(eigenvalues, dtype=complex)
    eigen.flags.writeable = False
    return DynamicStabilityResult(
        n_variables=n + N, n_predetermined=N, n_stable=n_stable, n_unit=n_unit, n_unstable=n_unstable,
        n_infinite=n_infinite, determinate=determinate, slowest_stable_root=slowest,
        smallest_unstable_root=smallest, eigenvalues=eigen, unstable_eigenvalue=unstable_eigenvalue,
        loading_countries=tuple(economy.p.countries), loading_share=share, top_loadings=top,
        tolerance=float(tolerance), seconds=time.perf_counter() - tic,
        metadata={"policy": policy.label, "stationary_residual": residual,
                  "stationarity_tol": float(stationarity_tol),
                  "pencil": "[[A_plus,0],[0,I]] s_{t+1} = [[-A_0,-A_minus_K],[E_K,0]] s_t"})
