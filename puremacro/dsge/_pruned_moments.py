"""Exact unconditional moments of pruned second- and third-order solutions.

Kim, Kim, Schaumburg and Sims (2008) and Andreasen, Fernández-Villaverde and
Rubio-Ramírez (2018) write a pruned perturbation solution as a linear system in
an augmented state. At second order

    z_t = [xf; xs; xf (x) xf],

and at third order

    z_t = [xf; xs; xf (x) xf; xrd; xf (x) xs; xf (x) xf (x) xf],

where xf, xs and xrd are the first-, second- and third-order parts of the
predetermined states. This module builds those systems with every innovation
centred on its conditional mean, so the innovations eps_t form a martingale
difference sequence (uncorrelated with the past, though heteroskedastic):

    z_t = c + A z_{t-1} + B eps_t,        y_t - ybar = d + C z_{t-1} + D eps_t.

That gives closed forms for the mean, the covariance and every
autocovariance of the variables y (states and controls alike):

    Var(z)  = A Var(z) A' + B Sigma_eps B',
    Gamma_0 = C Var(z) C' + D Sigma_eps D',
    Gamma_k = C A^k Var(z) C' + C A^(k-1) B Sigma_eps D'      (k >= 1).

At second order the innovations [u; u (x) u - vec(Sigma); xf (x) u] need no
centring, and the system is the one Dynare's ``pruned_state_space_system``
(Mutschler 2018) builds for ``stoch_simul(order=2, pruning)``. At third order
Dynare keeps the uncentred innovation xf (x) u (x) u, whose conditional mean
xf (x) vec(Sigma_u) makes it correlated with z_{t-1}. Its mean and contemporaneous
covariance equal the ones here. Its autocovariance recursion (Dynare 8) drops the
correlation of that innovation with earlier innovations, so its order-3
autocorrelations differ from these; long simulations, Dynare's own included,
agree with the values here (``tests/test_dsge_order3_moments.py``).

Shocks are Gaussian, so odd moments vanish and even moments follow Isserlis'
theorem. The augmented state has 2 n + n^2 entries at second order and
3 n + 2 n^2 + n^3 at third order for n predetermined states, so the dense
Lyapunov solve limits the method to small and medium models, as it does in
Dynare.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.linalg

MAX_PRUNED_STATE_DIM = 3000
"""Largest augmented state solved by default: 13 states at third order (2574
entries), 53 at second order (2915 entries)."""


def pruned_state_dim(n_x: int, order: int = 3) -> int:
    """Size of the augmented pruned state for ``n_x`` predetermined states."""
    if order == 2:
        return 2 * n_x + n_x**2
    if order == 3:
        return 3 * n_x + 2 * n_x**2 + n_x**3
    raise ValueError(f"pruned state spaces exist for orders 2 and 3, got {order}")


def _check_state_dim(n_x: int, order: int, max_state_dim: int | None) -> None:
    z_dim = pruned_state_dim(n_x, order)
    if max_state_dim is not None and z_dim > max_state_dim:
        raise ValueError(
            f"exact pruned order-{order} moments need an augmented state of size {z_dim} "
            f"for {n_x} predetermined states, above max_state_dim={max_state_dim}. "
            "Raise max_state_dim (the cost grows with its cube) or use simulated moments."
        )


def _kron_perm(dims: Sequence[int], order: Sequence[int]) -> np.ndarray:
    """Index P with kron(v[order[0]], v[order[1]], ...) == kron(v[0], v[1], ...)[P]."""
    return np.arange(int(np.prod(dims))).reshape(tuple(dims)).transpose(tuple(order)).ravel()


def _perfect_matchings(items: list[int]):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for i, partner in enumerate(rest):
        for tail in _perfect_matchings(rest[:i] + rest[i + 1:]):
            yield [(first, partner)] + tail


def gaussian_moment_tensor(cov: np.ndarray, k: int) -> np.ndarray:
    """E[v_{i1} ... v_{ik}] for v ~ N(0, cov), by Isserlis' theorem (k even)."""
    cov = np.asarray(cov, dtype=float)
    letters = "abcdefghijkl"[:k]
    out = np.zeros((cov.shape[0],) * k)
    for matching in _perfect_matchings(list(range(k))):
        subscripts = ",".join(letters[p] + letters[q] for p, q in matching)
        out += np.einsum(f"{subscripts}->{letters}", *([cov] * len(matching)))
    return out


def _lyapunov(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0:
        return np.zeros((0, 0))
    method = "direct" if a.shape[0] < 10 else "bilinear"
    x = scipy.linalg.solve_discrete_lyapunov(a, q, method=method)
    return 0.5 * (x + x.T)


def _blocks(sizes: Sequence[int]) -> list[slice]:
    edges = np.cumsum([0, *sizes])
    return [slice(int(lo), int(hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def _coefficients(coef: dict[str, np.ndarray], n_x: int) -> tuple[dict, dict]:
    """All-variable coefficients ``g`` and their predetermined-state rows ``h``."""
    g = {k: np.asarray(v, dtype=float) for k, v in coef.items()}
    g["gss"] = g["gss"].reshape(g["gx"].shape[0])
    return g, {k: v[:n_x] for k, v in g.items()}


def pruned_state_space_order2(
    coef: dict[str, np.ndarray],
    n_x: int,
    sigma_u: np.ndarray,
    *,
    max_state_dim: int | None = MAX_PRUNED_STATE_DIM,
) -> dict[str, Any]:
    """Pruned state space of a second-order perturbation solution.

    Parameters
    ----------
    coef : dict
        Dynare-convention coefficients ``gx, gu, gxx, gxu, guu, gss`` for every
        reported variable, stacked with the ``n_x`` predetermined states first.
        The decision rule is y = ybar + gx x + gu u + 1/2 gxx (x (x) x)
        + gxu (x (x) u) + 1/2 guu (u (x) u) + 1/2 gss, with x the lagged states;
        ``gss`` must already be scaled to ``sigma_u``. Further keys are ignored.
    n_x : int
        Number of predetermined states (the first ``n_x`` rows of ``coef``).
    sigma_u : np.ndarray
        Covariance of the Gaussian innovations u.
    max_state_dim : int or None
        Refuse augmented states larger than this (None disables the check).

    Returns
    -------
    dict
        ``A, B, c, C, D, d``, ``Sigma_eps``, ``Var_z``, ``E_z``, the block slices
        ``z_blocks`` / ``eps_blocks``, and ``E_xfxf`` (the covariance of xf) and
        ``E_xs`` (the mean of xs).
    """
    n = int(n_x)
    _check_state_dim(n, 2, max_state_dim)
    sigma_u = np.asarray(sigma_u, dtype=float)
    e = sigma_u.shape[0]
    g, h = _coefficients(coef, n)
    hx, hu = h["gx"], h["gu"]
    n2, e2 = n * n, e * e
    vs = sigma_u.reshape(-1)                      # E[u (x) u]
    swap2 = _kron_perm((n, n), (1, 0))            # a (x) b -> b (x) a

    e_xfxf = _lyapunov(hx, hu @ sigma_u @ hu.T) if n else np.zeros((0, 0))
    vec_s1 = e_xfxf.reshape(-1)
    var_uu = gaussian_moment_tensor(sigma_u, 4).reshape(e2, e2) - np.outer(vs, vs)

    zb = _blocks([n, n, n2])
    eb = _blocks([e, e2, n * e])
    z_dim, n_eps = zb[-1].stop, eb[-1].stop
    hx_hu = np.kron(hx, hu)
    a = np.zeros((z_dim, z_dim))
    a[zb[0], zb[0]] = hx
    a[zb[1], zb[1]] = hx
    a[zb[1], zb[2]] = 0.5 * h["gxx"]
    a[zb[2], zb[2]] = np.kron(hx, hx)
    b = np.zeros((z_dim, n_eps))
    b[zb[0], eb[0]] = hu
    b[zb[1], eb[1]] = 0.5 * h["guu"]
    b[zb[1], eb[2]] = h["gxu"]
    b[zb[2], eb[1]] = np.kron(hu, hu)
    b[zb[2], eb[2]] = hx_hu + hx_hu[swap2, :]
    c_xs = 0.5 * h["gss"] + 0.5 * h["guu"] @ vs
    c = np.zeros(z_dim)
    c[zb[1]] = c_xs
    c[zb[2]] = np.kron(hu, hu) @ vs
    s_eps = scipy.linalg.block_diag(sigma_u, var_uu, np.kron(e_xfxf, sigma_u))
    var_z = _lyapunov(a, b @ s_eps @ b.T)

    e_xs = np.linalg.solve(np.eye(n) - hx, 0.5 * h["gxx"] @ vec_s1 + c_xs) if n else np.zeros(0)
    e_z = np.zeros(z_dim)
    e_z[zb[1]] = e_xs
    e_z[zb[2]] = vec_s1

    cm = np.zeros((g["gx"].shape[0], z_dim))
    cm[:, zb[0]] = g["gx"]
    cm[:, zb[1]] = g["gx"]
    cm[:, zb[2]] = 0.5 * g["gxx"]
    dm = np.zeros((g["gx"].shape[0], n_eps))
    dm[:, eb[0]] = g["gu"]
    dm[:, eb[1]] = 0.5 * g["guu"]
    dm[:, eb[2]] = g["gxu"]
    d = 0.5 * g["gss"] + 0.5 * g["guu"] @ vs

    return {
        "A": a, "B": b, "c": c, "C": cm, "D": dm, "d": d,
        "Sigma_eps": s_eps, "Var_z": var_z, "E_z": e_z,
        "z_blocks": zb, "eps_blocks": eb, "E_xfxf": e_xfxf, "E_xs": e_xs,
    }


def pruned_state_space_order3(
    coef: dict[str, np.ndarray],
    n_x: int,
    sigma_u: np.ndarray,
    *,
    max_state_dim: int | None = MAX_PRUNED_STATE_DIM,
) -> dict[str, Any]:
    """Centred pruned state space of a third-order perturbation solution.

    Parameters
    ----------
    coef : dict
        Dynare-convention coefficients ``gx, gu, gxx, gxu, guu, gss, gxxx, gxxu,
        gxuu, guuu, gxss, guss`` for every reported variable, stacked with the
        ``n_x`` predetermined states first. The decision rule is
        y = ybar + gx x + gu u + 1/2 gxx (x (x) x) + gxu (x (x) u) + 1/2 guu (u (x) u)
        + 1/2 gss + 1/6 gxxx x^3 + 1/2 gxxu x^2 u + 1/2 gxuu x u^2 + 1/6 guuu u^3
        + 1/2 gxss x + 1/2 guss u, with x the lagged states. The risk terms
        gss, gxss and guss must already be scaled to ``sigma_u``.
    n_x : int
        Number of predetermined states (the first ``n_x`` rows of ``coef``).
    sigma_u : np.ndarray
        Covariance of the Gaussian innovations u.
    max_state_dim : int or None
        Refuse augmented states larger than this (None disables the check).

    Returns
    -------
    dict
        ``A, B, c, C, D, d`` of the centred system, ``Sigma_eps``, ``Var_z``,
        ``E_z``, the block slices ``z_blocks`` / ``eps_blocks``, the lower-order
        moments ``E_xfxf`` and ``E_xs``, and ``Var_z2`` (the covariance of the
        second-order state [xf; xs; xf (x) xf]).
    """
    n = int(n_x)
    _check_state_dim(n, 3, max_state_dim)
    sigma_u = np.asarray(sigma_u, dtype=float)
    e = sigma_u.shape[0]
    g, h = _coefficients(coef, n)
    hx, hu = h["gx"], h["gu"]
    n2, n3, e2, e3 = n * n, n**3, e * e, e**3
    z_dim = pruned_state_dim(n, 3)

    vs = sigma_u.reshape(-1)                      # E[u (x) u]
    i_vs = np.kron(np.eye(n), vs.reshape(-1, 1))  # xf (x) vec(Sigma): xf -> n e^2
    swap2 = _kron_perm((n, n), (1, 0))            # a (x) b -> b (x) a

    # ---- moments of the second-order state [xf; xs; xf (x) xf] -------------
    second = pruned_state_space_order2(coef, n, sigma_u, max_state_dim=None)
    z2 = second["z_blocks"]
    e_xfxf, e_xs, var_z2 = second["E_xfxf"], second["E_xs"], second["Var_z"]
    vec_s1 = e_xfxf.reshape(-1)
    m4u = gaussian_moment_tensor(sigma_u, 4)
    m4u_13 = m4u.reshape(e, e3)                   # E[u (u (x) u (x) u)']
    var_uu = m4u.reshape(e2, e2) - np.outer(vs, vs)
    m6u = gaussian_moment_tensor(sigma_u, 6).reshape(e3, e3)
    m4x = gaussian_moment_tensor(e_xfxf, 4).reshape(n2, n2)
    e_xsxs = var_z2[z2[1], z2[1]] + np.outer(e_xs, e_xs)
    e_xs_xfxf = var_z2[z2[1], z2[2]] + np.outer(e_xs, vec_s1)

    # ---- third-order system ----------------------------------------------
    zb = _blocks([n, n, n2, n, n2, n3])
    eb = _blocks([e, e2, n * e, n * e, n2 * e, n * e2, e3])
    n_eps = eb[-1].stop
    hx_hu = np.kron(hx, hu)

    b = np.zeros((z_dim, n_eps))
    b[zb[0], eb[0]] = hu
    b[zb[1], eb[1]] = 0.5 * h["guu"]
    b[zb[1], eb[2]] = h["gxu"]
    b[zb[2], eb[1]] = np.kron(hu, hu)
    b[zb[2], eb[2]] = hx_hu + hx_hu[swap2, :]
    b[zb[3], eb[0]] = 0.5 * h["guss"]
    b[zb[3], eb[3]] = h["gxu"]
    b[zb[3], eb[4]] = 0.5 * h["gxxu"]
    b[zb[3], eb[5]] = 0.5 * h["gxuu"]
    b[zb[3], eb[6]] = h["guuu"] / 6.0
    b[zb[4], eb[0]] = np.kron(hu, 0.5 * h["gss"].reshape(-1, 1))
    b[zb[4], eb[3]] = hx_hu[swap2, :]
    b[zb[4], eb[4]] = np.kron(hx, h["gxu"]) + np.kron(0.5 * h["gxx"], hu)[swap2, :]
    b[zb[4], eb[5]] = np.kron(hx, 0.5 * h["guu"]) + np.kron(h["gxu"], hu)[swap2, :]
    b[zb[4], eb[6]] = np.kron(hu, 0.5 * h["guu"])
    base5 = np.kron(np.kron(hx, hx), hu)          # a (x) a (x) b
    base6 = np.kron(np.kron(hx, hu), hu)          # a (x) b (x) b
    b[zb[5], eb[4]] = (base5 + base5[_kron_perm((n, n, n), (0, 2, 1)), :]
                       + base5[_kron_perm((n, n, n), (2, 0, 1)), :])
    b[zb[5], eb[5]] = (base6 + base6[_kron_perm((n, n, n), (1, 0, 2)), :]
                       + base6[_kron_perm((n, n, n), (1, 2, 0)), :])
    b[zb[5], eb[6]] = np.kron(np.kron(hu, hu), hu)

    a = np.zeros((z_dim, z_dim))
    a[zb[0], zb[0]] = hx
    a[zb[1], zb[1]] = hx
    a[zb[1], zb[2]] = 0.5 * h["gxx"]
    a[zb[2], zb[2]] = np.kron(hx, hx)
    a[zb[3], zb[0]] = 0.5 * h["gxss"]
    a[zb[3], zb[3]] = hx
    a[zb[3], zb[4]] = h["gxx"]
    a[zb[3], zb[5]] = h["gxxx"] / 6.0
    a[zb[4], zb[0]] = np.kron(hx, 0.5 * h["gss"].reshape(-1, 1))
    a[zb[4], zb[4]] = np.kron(hx, hx)
    a[zb[4], zb[5]] = np.kron(hx, 0.5 * h["gxx"])
    a[zb[5], zb[5]] = np.kron(np.kron(hx, hx), hx)
    # Centring xf (x) u (x) u on its conditional mean xf (x) vec(Sigma) moves that mean into A.
    a[:, zb[0]] += b[:, eb[5]] @ i_vs

    c = np.zeros(z_dim)
    c[zb[1]] = second["c"][z2[1]]
    c[zb[2]] = second["c"][z2[2]]

    s_eps = np.zeros((n_eps, n_eps))

    def put(i: int, j: int, block: np.ndarray) -> None:
        s_eps[eb[i], eb[j]] = block
        if i != j:
            s_eps[eb[j], eb[i]] = block.T

    put(0, 0, sigma_u)
    put(0, 3, np.kron(e_xs.reshape(1, -1), sigma_u))
    put(0, 4, np.kron(vec_s1.reshape(1, -1), sigma_u))
    put(0, 6, m4u_13)
    put(1, 1, var_uu)
    put(2, 2, np.kron(e_xfxf, sigma_u))
    put(3, 3, np.kron(e_xsxs, sigma_u))
    put(3, 4, np.kron(e_xs_xfxf, sigma_u))
    put(3, 6, np.kron(e_xs.reshape(-1, 1), m4u_13))
    put(4, 4, np.kron(m4x, sigma_u))
    put(4, 6, np.kron(vec_s1.reshape(-1, 1), m4u_13))
    put(5, 5, np.kron(e_xfxf, var_uu))
    put(6, 6, m6u)

    var_z = _lyapunov(a, b @ s_eps @ b.T)

    e_z = np.zeros(z_dim)
    e_z[zb[1]] = e_xs
    e_z[zb[2]] = vec_s1

    cm = np.zeros((g["gx"].shape[0], z_dim))
    cm[:, zb[0]] = g["gx"] + 0.5 * g["gxss"] + 0.5 * g["gxuu"] @ i_vs
    cm[:, zb[1]] = g["gx"]
    cm[:, zb[2]] = 0.5 * g["gxx"]
    cm[:, zb[3]] = g["gx"]
    cm[:, zb[4]] = g["gxx"]
    cm[:, zb[5]] = g["gxxx"] / 6.0
    dm = np.zeros((g["gx"].shape[0], n_eps))
    dm[:, eb[0]] = g["gu"] + 0.5 * g["guss"]
    dm[:, eb[1]] = 0.5 * g["guu"]
    dm[:, eb[2]] = g["gxu"]
    dm[:, eb[3]] = g["gxu"]
    dm[:, eb[4]] = 0.5 * g["gxxu"]
    dm[:, eb[5]] = 0.5 * g["gxuu"]
    dm[:, eb[6]] = g["guuu"] / 6.0

    return {
        "A": a, "B": b, "c": c, "C": cm, "D": dm, "d": second["d"],
        "Sigma_eps": s_eps, "Var_z": var_z, "E_z": e_z,
        "z_blocks": zb, "eps_blocks": eb,
        "E_xfxf": e_xfxf, "E_xs": e_xs, "Var_z2": var_z2,
    }


def _output_moments(ss: dict[str, Any], lags: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Mean deviation, Gamma_0 and Gamma_1..Gamma_lags of y from a centred pruned system."""
    a, b, cm, dm = ss["A"], ss["B"], ss["C"], ss["D"]
    s_eps, var_z = ss["Sigma_eps"], ss["Var_z"]
    mean_dev = cm @ ss["E_z"] + ss["d"]
    gamma_0 = cm @ var_z @ cm.T + dm @ s_eps @ dm.T
    gamma_0 = 0.5 * (gamma_0 + gamma_0.T)
    gammas: list[np.ndarray] = []
    lagged_state = var_z @ cm.T                   # A^k Var(z) C'
    lagged_shock = b @ s_eps @ dm.T               # A^(k-1) B Sigma_eps D'
    for _ in range(int(lags)):
        lagged_state = a @ lagged_state
        gammas.append(cm @ lagged_state + cm @ lagged_shock)
        lagged_shock = a @ lagged_shock
    return mean_dev, gamma_0, gammas


def pruned_order2_moments(
    coef: dict[str, np.ndarray],
    n_x: int,
    sigma_u: np.ndarray,
    lags: int,
    *,
    max_state_dim: int | None = MAX_PRUNED_STATE_DIM,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Mean deviation, covariance and autocovariances of a pruned second-order solution.

    Returns ``(mean_dev, Gamma_0, [Gamma_1, ..., Gamma_lags])`` for the stacked
    variables of ``coef``, with ``mean_dev`` measured from the deterministic
    steady state and ``Gamma_k = cov(y_t, y_{t-k})``.
    """
    return _output_moments(
        pruned_state_space_order2(coef, n_x, sigma_u, max_state_dim=max_state_dim), lags
    )


def pruned_order3_moments(
    coef: dict[str, np.ndarray],
    n_x: int,
    sigma_u: np.ndarray,
    lags: int,
    *,
    max_state_dim: int | None = MAX_PRUNED_STATE_DIM,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Mean deviation, covariance and autocovariances of a pruned third-order solution.

    Returns ``(mean_dev, Gamma_0, [Gamma_1, ..., Gamma_lags])`` for the stacked
    variables of ``coef``, with ``mean_dev`` measured from the deterministic
    steady state and ``Gamma_k = cov(y_t, y_{t-k})``.
    """
    return _output_moments(
        pruned_state_space_order3(coef, n_x, sigma_u, max_state_dim=max_state_dim), lags
    )
