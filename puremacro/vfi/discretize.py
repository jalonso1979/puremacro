"""AR(1) discretizers for the vfi engine: Tauchen (1986) and Rouwenhorst (1995).

Both take the INNOVATION sd ``sigma`` of x' = rho*x + eps, eps~N(0,sigma^2),
and return ``(grid, P)`` with ``grid`` shape (n,) symmetric about 0 and ``P`` the
(n,n) row-stochastic transition. numpy-only (one-time setup, never a GPU/hot
path). ``rouwenhorst`` is a verbatim copy of the nested DMP model's kernel; a
test asserts the two stay identical.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


def tauchen(n: int, rho: float, sigma: float, m: float = 3.0):
    """Tauchen (1986) discretization of x' = rho*x + eps, eps~N(0,sigma^2).

    ``m`` sets the grid half-width in unconditional standard deviations.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    if not (0.0 <= abs(rho) < 1.0):
        raise ValueError(f"|rho| must be < 1; got {rho}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    sigma_z = sigma / np.sqrt(1.0 - rho ** 2)
    z_max = m * sigma_z
    grid = np.linspace(-z_max, z_max, n)
    step = grid[1] - grid[0]
    P = np.zeros((n, n))
    for i in range(n):
        mu_i = rho * grid[i]
        P[i, 0] = norm.cdf((grid[0] - mu_i + step / 2.0) / sigma)
        P[i, n - 1] = 1.0 - norm.cdf((grid[n - 1] - mu_i - step / 2.0) / sigma)
        for j in range(1, n - 1):
            hi = norm.cdf((grid[j] - mu_i + step / 2.0) / sigma)
            lo = norm.cdf((grid[j] - mu_i - step / 2.0) / sigma)
            P[i, j] = hi - lo
    return grid, P


def rouwenhorst(n: int, rho: float, sigma_eps: float):
    """Rouwenhorst (1995) discretization of x' = rho*x + eps, eps~N(0,sigma_eps^2).

    Returns ``(grid, P)`` with ``grid`` shape (n,) symmetric about 0 and ``P``
    the (n,n) row-stochastic transition matrix.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1.0 - p], [1.0 - p, p]])
    for k in range(3, n + 1):
        Pk = np.zeros((k, k))
        Pk[:-1, :-1] += p * P
        Pk[:-1, 1:] += (1.0 - p) * P
        Pk[1:, :-1] += (1.0 - p) * P
        Pk[1:, 1:] += p * P
        Pk[1:-1, :] /= 2.0
        P = Pk
    sigma_uncond = sigma_eps / np.sqrt(1.0 - rho ** 2)
    psi = np.sqrt(n - 1) * sigma_uncond
    grid = np.linspace(-psi, psi, n)
    return grid, P


def _maxent_weights(T, log_q):
    """Maximum-entropy weights p ∝ exp(log_q + T@lambda) with E_p[T]=0.

    Solves the convex dual min_lambda logsumexp(log_q + T·lambda) (gradient
    E_p[T]) by BFGS. ``T`` is (n, k) centered moment functions; ``log_q`` is the
    (n,) LOG base distribution. Everything is in log space (logsumexp) so the
    far-tail nodes -- whose base density underflows -- contribute cleanly as
    log_q = -inf (probability 0) with no log(0)/0-div warnings or NaN gradients.
    Returns (p, ok); ok is judged by the actual moment residual (max|E_p[T]|<1e-7),
    not res.success (BFGS sometimes reports precision-loss with a negligible
    gradient).
    """
    def fun(lam):
        return float(logsumexp(T @ lam + log_q))

    def jac(lam):
        s = T @ lam + log_q
        p = np.exp(s - logsumexp(s))
        return T.T @ p

    # At infeasible edge states the dual is unbounded, so BFGS explores large
    # lambda where T@lambda overflows (-> inf -> NaN). Those intermediate events
    # are benign: the result is validated by the moment residual below, and the
    # caller falls back when ok is False. Suppress them so they don't leak as
    # warnings/FP errors to callers.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        res = minimize(fun, np.zeros(T.shape[1]), jac=jac, method="BFGS",
                       options={"gtol": 1e-10})
        s = T @ res.x + log_q
        p = np.exp(s - logsumexp(s))
        ok = bool(np.max(np.abs(T.T @ p)) < 1e-7)
    return p, ok


def farmer_toda(n: int, rho: float, sigma: float, m: float = 3.0):
    """Farmer-Toda (2017) maximum-entropy discretization of x'=rho*x+eps, eps~N(0,sigma^2).

    Even-spaced grid over +/- m unconditional sd. For each from-state the
    transition row is the maximum-entropy tilt of the discretized Gaussian that
    matches the conditional mean and variance exactly (falling back to mean-only,
    then the base, at grid edges where the variance match is infeasible).
    Returns (grid, P). More accurate than Tauchen at high persistence.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    if not (0.0 <= abs(rho) < 1.0):
        raise ValueError(f"|rho| must be < 1; got {rho}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    sigma_z = sigma / np.sqrt(1.0 - rho ** 2)
    grid = np.linspace(-m * sigma_z, m * sigma_z, n)
    P = np.zeros((n, n))
    for i in range(n):
        mu_i = rho * grid[i]
        dev = grid - mu_i
        log_kernel = -0.5 * (dev / sigma) ** 2
        log_q = log_kernel - logsumexp(log_kernel)        # normalized log base (no underflow)
        # match mean + variance; fall back to mean-only, then the base, at edges
        p, ok = _maxent_weights(np.column_stack([dev, dev ** 2 - sigma ** 2]), log_q)
        if not ok:
            p1, ok1 = _maxent_weights(dev[:, None], log_q)
            p = p1 if ok1 else np.exp(log_q)
        P[i] = p
    return grid, P


def combine_markov_chains(*chains):
    """Combine INDEPENDENT Markov chains into their product chain (Kronecker).

    Each argument is a ``(grid, P)`` pair (as returned by ``tauchen`` /
    ``rouwenhorst`` / ``farmer_toda``): ``grid`` shape (n_i,), ``P`` the
    (n_i, n_i) row-stochastic transition. Returns ``(values, P_combined)``:

      * ``values`` -- (N, k) array, N = prod(n_i), k = number of chains;
        ``values[s]`` is the tuple of component values at combined state ``s``,
        enumerated in C-order (the last chain varies fastest -- the same order
        ``numpy.kron`` produces). ``values[:, j]`` is chain ``j``'s value path.
      * ``P_combined`` -- (N, N) = ``kron(P_1, ..., P_k)``, row-stochastic.

    The chains are assumed independent, so the joint transition factorizes. Use
    with ``VFIProblem`` by passing ``values[:, j]`` for shock ``j`` inside the
    return function, or -- for additive log-income components --
    ``values.sum(axis=1)`` as a scalar ``z_grid`` alongside ``P_combined``.
    """
    if len(chains) == 0:
        raise ValueError("combine_markov_chains needs at least one chain")
    grids, Ps = [], []
    for idx, ch in enumerate(chains):
        g = np.asarray(ch[0], dtype=float)
        P = np.asarray(ch[1], dtype=float)
        n = g.size
        if g.ndim != 1:
            raise ValueError(f"chain {idx}: grid must be 1-D; got shape {g.shape}")
        if P.shape != (n, n):
            raise ValueError(
                f"chain {idx}: P must have shape ({n},{n}) to match its grid; "
                f"got {P.shape}"
            )
        if np.any(P < -1e-12) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError(f"chain {idx}: P must be nonnegative and row-stochastic")
        grids.append(g)
        Ps.append(P)
    P_combined = Ps[0]
    for P in Ps[1:]:
        P_combined = np.kron(P_combined, P)
    mesh = np.meshgrid(*grids, indexing="ij")            # C-order matches kron
    values = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return values, P_combined


def markov_stationary(P):
    """Stationary distribution ``pi`` of a row-stochastic Markov matrix ``P``.

    Returns the (n,) probability vector with ``pi @ P == pi`` -- the normalized
    left eigenvector of ``P`` for eigenvalue 1 (the ergodic distribution of an
    irreducible chain; e.g. the long-run population share of each discretized
    shock state).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be a square matrix; got shape {P.shape}")
    if np.any(P < -1e-12) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P must be nonnegative and row-stochastic")
    w, V = np.linalg.eig(P.T)
    pi = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    return pi / pi.sum()


__all__ = ["tauchen", "rouwenhorst", "farmer_toda", "combine_markov_chains",
           "markov_stationary"]
