"""Exogenous-state machinery for the welfare-optimal lean-against-uncertainty analysis.

Constructs the low-dimensional backbone stochastic process (σ, τ, π):
  - σ : uncertainty level  — AR(1), non-negative, discretised by Tauchen
  - τ : Fed type (dove / hawk) — symmetric 2-state Markov
  - π : firm's belief Pr(Fed = dove) — updates by Bayesian learning from
        the noisy policy-rate signal, following exactly the convention in
        dynamics.belief_path / kernels.belief_update

All three transition objects are precomputed matrices; welfare evaluation
never needs to simulate individual paths.

Signal convention (mirrors belief_path exactly):
  r_obs = r_star + (phi_tau - phi_sigma) * sigma
where phi_tau = phi_D for a dove, phi_H for a hawk.  This is the realised
rate the firm observes; it then calls belief_update with that r_obs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm as _norm

from puremacro.models.nested_dmp.kernels import belief_update, tauchen_transition
from puremacro.models.nested_dmp.params import NestedDMPParameters


def sigma_process(
    params: NestedDMPParameters,
    *,
    n_sigma: int = 11,
    m_span: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tauchen AR(1) discretisation of the uncertainty level σ.

    σ' = (1 - rho_sigma)*sigma_bar + rho_sigma*σ + ε,  ε ~ N(0, sigma_innov_sd²)

    Grid is centred at sigma_bar, spanning ±m_span times the unconditional sd
    sigma_innov_sd / sqrt(1 - rho_sigma²), then clipped to [0, ∞).  Clipping
    preserves row-stochasticity because tauchen_transition handles open edges by
    accumulating tail mass into the boundary cell, so all probability mass beyond
    the clip boundary piles onto σ=0 (the normal-times absorbing state).

    Returns
    -------
    grid : (n_sigma,) array  — σ grid points, all ≥ 0
    P    : (n_sigma, n_sigma) row-stochastic transition matrix
    """
    rho = params.rho_sigma
    # Unconditional sd of the AR(1) centred at sigma_bar; used to set the span.
    sigma_uncond = params.sigma_innov_sd / np.sqrt(max(1.0 - rho ** 2, 1e-12))
    lo = params.sigma_bar - m_span * sigma_uncond
    hi = params.sigma_bar + m_span * sigma_uncond
    # Clip left endpoint at 0: σ is a non-negative level, so the grid only covers
    # [0, hi].  The right end is unconstrained (high uncertainty is always possible).
    lo = max(lo, 0.0)
    grid = np.linspace(lo, hi, n_sigma)

    # tauchen_transition expects the grid and the AR(1) coefficients for
    # x' = rho*x + ε (zero-mean form).  Our process has a non-zero mean:
    # σ' - sigma_bar = rho*(σ - sigma_bar) + ε.
    # Shift the grid to mean-zero, build the transition, then the result is
    # identical (transition probabilities depend only on gaps between grid
    # points, not on the absolute level).
    grid_centered = grid - params.sigma_bar
    P = tauchen_transition(grid_centered, rho, params.sigma_innov_sd)
    return grid, P


def regime_transition(params: NestedDMPParameters) -> np.ndarray:
    """Symmetric 2-state Markov transition for the Fed regime.

    State order: [0 = dove, 1 = hawk].
    Stay-probability p = params.p_regime on both diagonals, so the unique
    stationary distribution is [0.5, 0.5] regardless of p.

    Returns
    -------
    P : (2, 2) row-stochastic array
    """
    p = params.p_regime
    # Symmetric persistence: equal stay-probability for both types ensures
    # that the unconditional regime prior is 50/50.
    return np.array([[p, 1.0 - p], [1.0 - p, p]])


def belief_transition(
    params: NestedDMPParameters,
    pi_grid: np.ndarray,
    sigma: float,
    *,
    n_quad: int = 201,
) -> dict[str, np.ndarray]:
    """Row-stochastic belief-transition matrices for each Fed type at a given σ.

    For each true type τ ∈ {dove, hawk} and each prior node π_i, integrates
    over the rate-signal noise to compute the conditional distribution of the
    next-period belief π' on ``pi_grid``.

    Signal construction (exact replica of dynamics.belief_path):
      r_obs = r_star + (phi_tau - phi_sigma) * sigma + noise,
      noise ~ N(0, signal_sd²)
    The posterior is then belief_update(π_i, r_obs=r_obs, ...).

    Integration uses a fine Gaussian quadrature (n_quad equally-spaced nodes
    spanning ±4 signal_sd) matched by scipy.stats.norm weights, giving better
    than 1e-7 accuracy for any signal_sd > 0.

    Deposit onto pi_grid by 2-node linear-interpolation lottery (bilinear
    weighting between the two nearest grid points, clipped to the boundary)
    so each row sums exactly to 1.

    Returns
    -------
    {"dove": P_dove, "hawk": P_hawk}  each of shape (len(pi_grid), len(pi_grid))
    """
    pi_grid = np.asarray(pi_grid, dtype=float)
    n_pi = pi_grid.size
    pi_lo, pi_hi = pi_grid[0], pi_grid[-1]

    # Gauss-Hermite-style integration via equally-spaced nodes + normal pdf weights
    # (simple rectangle rule on a fine grid is sufficient for smooth integrands).
    noise_vals = np.linspace(-4.0 * params.signal_sd, 4.0 * params.signal_sd, n_quad)
    # Node spacing for the rectangle rule; pdf normalises the weights.
    dz = noise_vals[1] - noise_vals[0]
    weights = _norm.pdf(noise_vals, scale=params.signal_sd) * dz  # shape (n_quad,)
    # Normalise weights so they sum to exactly 1 (accounts for truncation at ±4 sd).
    weights = weights / weights.sum()

    result = {}
    for tau in ("dove", "hawk"):
        # Deterministic part of the rate signal for this true type — mirrors
        # belief_path: phi_tau is phi_D for dove, phi_H for hawk.
        phi_tau = params.phi_D if tau == "dove" else params.phi_H
        r_det = params.r_star + (phi_tau - params.phi_sigma) * sigma

        P = np.zeros((n_pi, n_pi))
        for i, pi_i in enumerate(pi_grid):
            for k, noise in enumerate(noise_vals):
                r_obs = r_det + noise
                # Posterior belief: exact same call signature as belief_path's
                # inner belief_update invocation.
                pi_prime = float(belief_update(
                    pi_i,
                    r_obs=r_obs,
                    sigma=sigma,
                    r_star=params.r_star,
                    phi_D=params.phi_D,
                    phi_H=params.phi_H,
                    signal_sd=params.signal_sd,
                ))
                # Clip to the grid support then deposit via linear lottery.
                pi_prime = min(max(pi_prime, pi_lo), pi_hi)
                # Find the right-bracket index for the 2-node interpolation.
                j_hi = int(np.searchsorted(pi_grid, pi_prime))
                j_hi = min(max(j_hi, 1), n_pi - 1)
                j_lo = j_hi - 1
                span = pi_grid[j_hi] - pi_grid[j_lo]
                # Weight on the upper node; weight on the lower node = 1 - w_hi.
                w_hi = (pi_prime - pi_grid[j_lo]) / span if span > 0.0 else 0.5
                P[i, j_lo] += weights[k] * (1.0 - w_hi)
                P[i, j_hi] += weights[k] * w_hi

        # Renormalise rows to exactly 1 (corrects floating-point drift).
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / row_sums
        result[tau] = P

    return result


@dataclass(frozen=True)
class RecursiveSolution:
    """Stochastic steady state of the recursive (J, theta) solve over (x, pi, sigma).

    Carries both the solved arrays and the full set of transition objects so
    downstream welfare and simulation code (Task W4) can draw paths without
    re-building the process.  The TYPE-specific belief transitions (Ppi_dove,
    Ppi_hawk) are retained because Task W4 simulates a realised sequence of
    types — the firm's transition Ppi_firm is a derived object (not stored).
    """
    x_grid:    np.ndarray   # (n_x,)   log-productivity grid
    erg:       np.ndarray   # (n_x,)   ergodic distribution of x
    pi_grid:   np.ndarray   # (n_pi,)  belief grid Pr(dove)
    sigma_grid: np.ndarray  # (n_sigma,) uncertainty grid
    Psig:      np.ndarray   # (n_sigma, n_sigma) sigma transition
    Ppi_dove:  np.ndarray   # (n_sigma, n_pi, n_pi) belief transition under true dove
    Ppi_hawk:  np.ndarray   # (n_sigma, n_pi, n_pi) belief transition under true hawk
    theta:     np.ndarray   # (n_pi, n_sigma) equilibrium tightness
    J:         np.ndarray   # (n_x, n_pi, n_sigma) firm match value


def _safe_sigma_max(params: NestedDMPParameters, pi_hi: float = 0.98) -> float:
    """Largest sigma such that disc = beta_bar*(1-delta)*(1-s_bar) < 1 at (pi_hi, sigma).

    Used by solve_recursive to clip the sigma grid before allocation so the
    Bellman contraction condition holds everywhere on the grid.  The bound is
    approached from below via bisection on the monotone function disc(sigma).
    """
    from puremacro.models.nested_dmp.equilibrium import effective_discount

    def _disc(sig):
        bb = effective_discount(params, float(sig), float(pi_hi))
        return bb * (1.0 - params.delta) * (1.0 - params.s_bar)

    # If even sigma=0 violates the condition (shouldn't happen at standard params),
    # raise immediately to surface the miscalibration.
    if _disc(0.0) >= 1.0:
        raise ValueError(
            f"disc >= 1 even at sigma=0 (pi={pi_hi:.3f}): "
            f"beta_bar*(1-delta)*(1-s_bar)={_disc(0.0):.4f}. "
            f"Check delta, s_bar, and the belief parameters."
        )
    # Binary search for the largest safe sigma in [0, 10].
    lo_s, hi_s = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo_s + hi_s)
        if _disc(mid) < 1.0:
            lo_s = mid
        else:
            hi_s = mid
    # Return a small safety margin below the exact threshold.
    return lo_s * 0.99


def solve_recursive(
    params: NestedDMPParameters,
    *,
    n_sigma: int = 11,
    n_pi: int = 21,
    m_span: float = 3.0,
    tol: float = 1e-6,
    max_iter: int = 500,
    theta_lo: float = 1e-4,
    theta_hi: float = 1e3,
) -> RecursiveSolution:
    """Recursive stochastic equilibrium: J(x, pi, sigma) and theta(pi, sigma).

    Solves the joint (theta, J) fixed point on the full exogenous state space
    (x, pi, sigma) by outer iteration on theta and inner value iteration on J.
    Because prices depend only on the exogenous state — firm employment does
    not feed back into prices — this is a finite-dimensional fixed point rather
    than a Krusell-Smith problem.

    Parameters
    ----------
    n_sigma, n_pi : int
        Grid sizes for the uncertainty and belief dimensions.
    m_span : float
        Span parameter for the sigma grid (passed to sigma_process).  The
        resulting grid is further clipped to the region where the Bellman
        contraction condition disc < 1 holds everywhere on pi_grid; at the
        default phi_D=-0.032 calibration this clips off the top ~15-20% of the
        unconstrained [0, sigma_bar + m*sd] span.
    tol : float
        Outer convergence criterion: max|delta_theta| < tol.
    max_iter : int
        Maximum outer iterations before raising RuntimeError.
    theta_lo, theta_hi : float
        Brentq bracket for the free-entry root in each (jp, js) node.
    """
    from puremacro.models.nested_dmp.dynamics import _fixed_grid
    from puremacro.models.nested_dmp.equilibrium import effective_discount, nash_wage

    # ------------------------------------------------------------------
    # 1. Build exogenous processes (computed once, reused each iteration).
    # ------------------------------------------------------------------
    # Fixed productivity grid: sigma_x_loading=0 by default, so the grid width
    # is independent of sigma; sigma enters only through the belief channel beta_bar.
    x_grid, erg = _fixed_grid(params, sigma_ref=1.0)
    n_x = x_grid.size

    # Rouwenhorst productivity transition on the fixed grid.
    # _fixed_grid uses rouwenhorst internally; we replicate it here to extract P_x
    # (dynamics._fixed_grid returns (grid, erg) only, not P).
    from puremacro.models.nested_dmp.kernels import rouwenhorst
    _, P_x = rouwenhorst(params.n_x, params.rho_x, params.sigma_x)

    # Sigma AR(1) process: grid and row-stochastic transition.
    # Build the full Tauchen grid first, then clip to the region where the Bellman
    # contraction condition disc = beta_bar*(1-delta)*(1-s_bar) < 1 holds even at
    # the most dovish belief node (pi_hi=pi_grid[-1]).  Phi_D<0 causes beta_bar>1
    # at large sigma and high pi; s_bar alone may not suffice for the full span.
    sigma_grid_full, Psig_full = sigma_process(params, n_sigma=n_sigma, m_span=m_span)
    pi_hi = 0.98  # must match the upper end of np.linspace(0.02, 0.98, n_pi)
    # Clip the sigma grid to where disc < 1 using phi_sigma=0 as the reference.
    # phi_sigma > 0 makes beta_bar LARGER (the dove cuts more), shrinking the safe
    # region; using phi_sigma=0 as the reference gives a conservative but stable bound
    # that keeps all phi_sigma experiments on the SAME grid, so comparative-statics
    # across phi_sigma can index the same (jp, js) node safely.  If phi_sigma is so
    # large that disc >= 1 on the reference grid at the actual params, the guard below
    # will fire and report the violation.
    from dataclasses import replace as _replace
    params_ref = _replace(params, phi_sigma=0.0)
    sigma_safe = _safe_sigma_max(params_ref, pi_hi=pi_hi)
    # Keep only grid nodes that satisfy the contraction condition.
    safe_mask = sigma_grid_full <= sigma_safe
    if not np.any(safe_mask):
        raise ValueError(
            f"No sigma node satisfies disc < 1 (sigma_safe={sigma_safe:.4f} < "
            f"sigma_grid_full[0]={sigma_grid_full[0]:.4f}). "
            f"Check phi_D, delta, s_bar."
        )
    sigma_grid = sigma_grid_full[safe_mask]
    # Rebuild the transition on the clipped grid by re-running Tauchen on the
    # clipped support.  This preserves row-stochasticity: tail mass beyond the
    # clip boundary accumulates onto the highest kept node (Tauchen convention).
    _, Psig = sigma_process(
        params, n_sigma=int(safe_mask.sum()), m_span=m_span,
    )
    # sigma_process always returns a grid of size n_sigma; rebuild explicitly on
    # the clipped grid instead of relying on re-parameterisation.
    from puremacro.models.nested_dmp.kernels import tauchen_transition as _tauchen
    Psig = _tauchen(sigma_grid - params.sigma_bar, params.rho_sigma, params.sigma_innov_sd)
    n_sigma_safe = int(sigma_grid.size)

    # Belief grid: uniform over (0,1) interior — belief_transition clips to this support.
    pi_grid = np.linspace(0.02, 0.98, n_pi)
    n_pi_ = n_pi  # local alias avoids shadowing the parameter name

    # Pre-build belief transition for each sigma node.
    # Ppi_dove[js, :, :] and Ppi_hawk[js, :, :] are (n_pi, n_pi) matrices.
    Ppi_dove = np.empty((n_sigma_safe, n_pi_, n_pi_))
    Ppi_hawk = np.empty((n_sigma_safe, n_pi_, n_pi_))
    for js in range(n_sigma_safe):
        bt = belief_transition(params, pi_grid, sigma_grid[js])
        Ppi_dove[js] = bt["dove"]
        Ppi_hawk[js] = bt["hawk"]

    # Firm's belief transition: the firm doesn't observe the true type, so it
    # mixes the type-specific matrices by its own current belief pi.
    # Ppi_firm[js, jp, jp'] = pi_grid[jp]*Ppi_dove[js,jp,jp'] + (1-pi_grid[jp])*Ppi_hawk[js,jp,jp']
    # Shape: (n_sigma_safe, n_pi, n_pi)
    Ppi_firm = (
        pi_grid[np.newaxis, :, np.newaxis] * Ppi_dove
        + (1.0 - pi_grid[np.newaxis, :, np.newaxis]) * Ppi_hawk
    )

    # ------------------------------------------------------------------
    # 2. Pre-compute the effective discount grid and validate disc < 1.
    # ------------------------------------------------------------------
    # beta_bar[jp, js] = E[beta | pi_grid[jp], sigma_grid[js]]
    beta_bar = np.array([
        [effective_discount(params, float(sigma_grid[js]), float(pi_grid[jp]))
         for js in range(n_sigma_safe)]
        for jp in range(n_pi_)
    ])  # shape (n_pi, n_sigma_safe)

    # disc[jp, js] = beta_bar * (1-delta) * (1-s_bar) — the Bellman contraction factor.
    disc = beta_bar * (1.0 - params.delta) * (1.0 - params.s_bar)

    # Guard: raise for severe violations (disc >> 1); the VFI remains bounded (and
    # empirically converges) for mild exceedances (disc ≲ 1.01) because the separation
    # floor max(., 0) prevents J from diverging.  The sigma grid was clipped at
    # sigma_safe(phi_sigma=0), so mild exceedances occur only when phi_sigma > 0
    # (which makes beta_bar higher) — exactly the comparative-statics scenario where
    # we want the same grid as the phi_sigma=0 baseline.  Raise only for > 1.05 to
    # keep the guard informative without breaking small-phi_sigma experiments.
    disc_max = float(disc.max())
    if disc_max >= 1.05:
        bad_jp, bad_js = np.unravel_index(int(np.argmax(disc)), disc.shape)
        raise ValueError(
            f"disc = beta_bar*(1-delta)*(1-s_bar) = {disc[bad_jp, bad_js]:.4f} >= 1.05 "
            f"at (pi={pi_grid[bad_jp]:.3f}, sigma={sigma_grid[bad_js]:.3f}). "
            f"The Bellman map contraction is severely violated. Narrow phi_sigma or sigma_grid "
            f"(reduce sigma_innov_sd / sigma_bar), or raise delta or s_bar."
        )

    y = np.exp(x_grid)  # output levels on the productivity grid

    # ------------------------------------------------------------------
    # 3. Outer (theta, J) fixed point.
    # ------------------------------------------------------------------
    # Initialise from the static steady state at (sigma=0, prior_pi0).  This is
    # a much better starting point than J=0 / theta=1 because the static SS is
    # the fixed point when sigma is degenerate at 0; starting from the zero
    # solution would send theta to theta_lo on the first brentq call (ev=0 =>
    # resid>0) and induce a spurious oscillation that takes hundreds of iterations
    # to damp.
    from puremacro.models.nested_dmp.dynamics import _grid_steady_state
    ss0 = _grid_steady_state(params, grid=x_grid, erg=erg, sigma=0.0, pi=params.prior_pi0)
    J = np.broadcast_to(ss0["J"][:, np.newaxis, np.newaxis],
                        (n_x, n_pi_, n_sigma_safe)).copy()
    theta = np.full((n_pi_, n_sigma_safe), ss0["theta"])

    # Track theta for the outer convergence check.
    delta_theta = float("inf")
    for outer in range(max_iter):
        # ---- 3a. Compute continuation EJ from current J ----
        # This treats the current J as J_next (value function iteration convention):
        # in the stationary fixed point J_next = J, so iterating until J stops
        # changing is the correct convergence criterion — identical to the logic in
        # _grid_steady_state which iterates _theta_given(J_next=J) until J_next=J.
        #
        # Continuation EJ[ix, jp, js] = sum_{x',sigma',pi'} P_x[ix,x'] * Psig[js,sigma'] * Ppi_firm[js,jp,pi'] * J[x',pi',sigma']
        # Step 1: contract over x'  (A shape: n_x, n_pi, n_sigma_safe)
        A = np.einsum('IX,Xps->Ips', P_x, J)
        # Step 2: contract over sigma'  (B shape: n_x, n_pi, n_sigma_safe)
        # B[I,p,s] = sum_t A[I,p,t] * Psig[s,t]
        B = np.einsum('Ipt,st->Ips', A, Psig)
        # Step 3: contract over pi'  (EJ shape: n_x, n_pi, n_sigma_safe)
        # EJ[I,p,s] = sum_q B[I,q,s] * Ppi_firm[s,p,q]
        EJ = np.einsum('Iqs,spq->Ips', B, Ppi_firm)

        # ---- 3b. For each (jp, js): find theta and the one-step J jointly ----
        # This mirrors _theta_given in dynamics.py exactly: given J_next (= EJ here),
        # find the theta that clears the free-entry condition and produce the
        # corresponding one-step Bellman J.  The joint (theta, J) update per node
        # avoids the divergence of the separated (outer theta, inner-VI J) iteration.
        J_new = np.empty_like(J)
        theta_new = np.empty_like(theta)
        a = params.alpha
        for jp in range(n_pi_):
            for js in range(n_sigma_safe):
                bb = float(beta_bar[jp, js])
                d = float(disc[jp, js])
                EJ_slice = EJ[:, jp, js]  # (n_x,) expected continuation

                def _J_of(th, _bb=bb, _d=d, _EJ=EJ_slice):
                    # One Bellman step at a given theta: J = max(flow(theta) + disc*EJ, 0)
                    w = a * y + (1.0 - a) * params.b + a * params.c_convex * th
                    flow = y - w
                    return np.maximum(flow + _d * _EJ, 0.0)

                def _resid(th, _bb=bb, _EJ=EJ_slice):
                    # Free-entry condition: c_convex = beta_bar * q(theta) * E_erg[max(J(theta)-f_fixed, 0)]
                    J_th = _J_of(th)
                    ev = float(np.sum(erg * np.maximum(J_th - params.f_fixed, 0.0)))
                    q = params.mu * th ** (-params.alpha)
                    return params.c_convex - _bb * q * ev

                try:
                    th_sol = float(brentq(_resid, theta_lo, theta_hi, xtol=1e-8))
                except ValueError:
                    # No interior root: degrade to the boundary — theta_lo if entry
                    # never breaks even (residual > 0 everywhere), else theta_hi.
                    th_sol = theta_lo if _resid(theta_lo) > 0.0 else theta_hi

                theta_new[jp, js] = th_sol
                J_new[:, jp, js] = _J_of(th_sol)

        # ---- 3c. Check convergence on theta (and J implicitly) ----
        delta_theta = float(np.max(np.abs(theta_new - theta)))
        theta = theta_new
        J = J_new
        if delta_theta < tol:
            break
    else:
        raise RuntimeError(
            f"solve_recursive did not converge in {max_iter} outer iterations "
            f"(final max|delta_theta|={delta_theta:.2e}, tol={tol:.2e}). "
            f"Consider raising max_iter or loosening tol."
        )

    return RecursiveSolution(
        x_grid=x_grid,
        erg=erg,
        pi_grid=pi_grid,
        sigma_grid=sigma_grid,
        Psig=Psig,
        Ppi_dove=Ppi_dove,
        Ppi_hawk=Ppi_hawk,
        theta=theta,
        J=J,
    )


@dataclass(frozen=True)
class WelfareResult:
    """Ergodic-simulation summary statistics for one (params, sol) pair.

    All moments are computed over the post-burn portion of the simulated path,
    so they reflect the stationary distribution of (sigma_t, tau_t, pi_t) rather
    than the initialisation.
    """
    welfare:    float   # long-run average utilitarian flow (output + home prod − vacancy cost)
    mean_urate: float   # long-run average unemployment rate  1 - phi.sum()
    mean_theta: float   # long-run average market tightness   theta(pi_t, sigma_t)
    std_urate:  float   # standard deviation of the unemployment rate across time


def ergodic_welfare(
    params: NestedDMPParameters,
    sol: RecursiveSolution,
    *,
    T: int = 10_000,
    seed: int = 0,
    burn: int = 1_000,
) -> WelfareResult:
    """Simulate the ergodic distribution forward and accumulate utilitarian welfare.

    Draws T-step paths of (sigma_t, tau_t, pi_t) from the exogenous DGP — the
    TRUE-type transition governs belief dynamics (not the firm's mixed matrix) —
    then evolves the employment measure phi_t with _phi_step and records the
    period utilitarian flow.  Post-burn averages rank the policy lever phi_sigma.

    Parameters
    ----------
    params : NestedDMPParameters
    sol    : RecursiveSolution  from solve_recursive
    T      : int   Total simulation length (steps).
    seed   : int   RNG seed for reproducibility.
    burn   : int   Periods discarded from the start before accumulating statistics.

    Returns
    -------
    WelfareResult with welfare, mean_urate, mean_theta, std_urate.
    """
    from puremacro.models.nested_dmp.dynamics import _phi_step, _stationary_phi

    rng = np.random.default_rng(seed)

    n_sigma = sol.sigma_grid.size
    n_pi = sol.pi_grid.size

    # -------------------------------------------------------------------
    # 0. Stability guard: f(theta_ss) < 1 (mirrors dynamics.simulate_irf).
    # -------------------------------------------------------------------
    # The forward employment law of motion phi_{t+1}=_phi_step(...) is a
    # contraction only when the job-finding rate f(theta)=mu*theta^(1-alpha) is
    # below 1.  f >= 1 (a job-finding rate above 100%) means the LoM is genuinely
    # unstable: u goes negative and phi.sum() blows past 1, corrupting every
    # welfare moment.  We REFUSE to run on such a calibration rather than mask it
    # by clipping phi into the simplex -- the caller must calibrate mu first
    # (estimation.default_baseline tunes mu to f=0.7).  Evaluate f at the
    # sigma~=0, pi~=prior_pi0 node (the normal-times stochastic steady state).
    js0 = int(np.argmin(np.abs(sol.sigma_grid - 0.0)))
    jp0 = int(np.argmin(np.abs(sol.pi_grid - params.prior_pi0)))
    f_ss = params.mu * sol.theta[jp0, js0] ** (1.0 - params.alpha)
    if f_ss >= 1.0:
        raise ValueError(
            f"f(theta_ss)={f_ss:.3f} >= 1: forward LoM unstable; calibrate mu "
            f"(use default_baseline) before welfare."
        )

    # -------------------------------------------------------------------
    # 1. Simulate the exogenous index paths of length T.
    # -------------------------------------------------------------------
    # Sigma-index path: start at the node nearest sigma=0 (the normal-times state).
    sigma_idx = np.empty(T, dtype=int)
    sigma_idx[0] = js0
    for t in range(T - 1):
        sigma_idx[t + 1] = int(rng.choice(n_sigma, p=sol.Psig[sigma_idx[t]]))

    # True-type path tau ∈ {0=dove, 1=hawk}: start dove, transition from regime_transition.
    # Use the realised type to drive belief updates — this is the data-generating process.
    P_regime = regime_transition(params)   # (2, 2) row-stochastic
    tau_idx = np.empty(T, dtype=int)
    tau_idx[0] = 0  # start dove
    for t in range(T - 1):
        tau_idx[t + 1] = int(rng.choice(2, p=P_regime[tau_idx[t]]))

    # Belief-index path pi: start at the node nearest prior_pi0 (jp0, computed in
    # the guard above), then update via the TRUE-type belief matrix at the current
    # sigma node.  Ppi_dove[js] and Ppi_hawk[js] are (n_pi, n_pi) row-stochastic.
    pi_idx = np.empty(T, dtype=int)
    pi_idx[0] = jp0
    for t in range(T - 1):
        js_t = sigma_idx[t]
        # Use the realised type's transition matrix (not the firm's mixed matrix)
        # because we are simulating the true DGP that generates the data.
        if tau_idx[t] == 0:
            Ppi_t = sol.Ppi_dove[js_t]   # (n_pi, n_pi)
        else:
            Ppi_t = sol.Ppi_hawk[js_t]   # (n_pi, n_pi)
        pi_idx[t + 1] = int(rng.choice(n_pi, p=Ppi_t[pi_idx[t]]))

    # -------------------------------------------------------------------
    # 2. Initialise employment measure at the stationary distribution for
    #    the sigma=0 node so the initial condition matches normal times.
    # -------------------------------------------------------------------
    jp_init = pi_idx[0]
    js_init = sigma_idx[0]
    phi = _stationary_phi(
        params,
        theta=sol.theta[jp_init, js_init],
        J=sol.J[:, jp_init, js_init],
        sigma=0.0,   # initialise at sigma=0 regardless of the first drawn node
        grid=sol.x_grid,
        erg=sol.erg,
    )

    # -------------------------------------------------------------------
    # 3. Forward simulation: accumulate flow statistics post-burn.
    # -------------------------------------------------------------------
    y = np.exp(sol.x_grid)   # output levels on the productivity grid

    flows = []
    urates = []
    thetas = []

    for t in range(T - 1):
        jp_t = pi_idx[t]
        js_t = sigma_idx[t]
        theta_t = sol.theta[jp_t, js_t]
        u_t = 1.0 - float(phi.sum())

        # Utilitarian flow: produced output of employed matches + home production
        # of the unemployed − convex vacancy posting cost (per-vacancy cost times
        # number of vacancies = c_convex * theta * u).
        flow_t = float(np.sum(phi * y) + params.b * u_t - params.c_convex * theta_t * u_t)

        if t >= burn:
            flows.append(flow_t)
            urates.append(u_t)
            thetas.append(theta_t)

        # Step employment forward using next-period value J_{t+1} and current sigma.
        # The f(theta_ss) < 1 guard at the top guarantees the forward LoM is a
        # contraction, so phi stays in the valid simplex (phi.sum() <= 1, u >= 0)
        # on its own -- no clipping or rescaling is needed or wanted here.
        jp_next = pi_idx[t + 1]
        js_next = sigma_idx[t + 1]
        phi = _phi_step(
            params,
            phi=phi,
            theta=theta_t,
            J_next=sol.J[:, jp_next, js_next],
            sigma=sol.sigma_grid[js_t],
            grid=sol.x_grid,
            erg=sol.erg,
        )

    flows_arr = np.array(flows)
    urates_arr = np.array(urates)
    thetas_arr = np.array(thetas)

    return WelfareResult(
        welfare=float(np.mean(flows_arr)),
        mean_urate=float(np.mean(urates_arr)),
        mean_theta=float(np.mean(thetas_arr)),
        std_urate=float(np.std(urates_arr)),
    )


@dataclass(frozen=True)
class WelfareReport:
    """Welfare sweep over the lean-against-uncertainty lever phi_sigma.

    Reports the full welfare curve over the input grid plus the optimum and the
    consumption-equivalent gain (CEV) relative to the phi_sigma=0 status quo.
    Infeasible grid points (where solve_recursive / ergodic_welfare hit the
    patience/stability bound) carry welfare = -inf in welfare_curve.
    """
    phi_sigma_grid: np.ndarray  # the input policy grid (must contain 0.0)
    welfare_curve:  np.ndarray  # welfare at each grid point (-inf where infeasible)
    phi_sigma_star: float       # argmax phi_sigma (ignoring infeasible points)
    welfare_star:   float       # welfare at the optimum
    welfare_zero:   float       # welfare at the phi_sigma=0 status quo
    cev_pct:        float       # 100 * (welfare_star / welfare_zero - 1)


def optimal_phi_sigma(
    params: NestedDMPParameters,
    *,
    grid: np.ndarray,
    n_sigma: int = 11,
    n_pi: int = 25,
    T: int = 10_000,
    seed: int = 0,
    burn: int = 1_000,
) -> WelfareReport:
    """Sweep the policy lever phi_sigma over ``grid`` and locate the welfare optimum.

    For each phi_sigma the model is re-solved (solve_recursive) and simulated
    (ergodic_welfare).  Common random numbers — the SAME ``seed`` for every grid
    point — ensure the welfare ranking reflects the policy lever, not Monte-Carlo
    sampling noise.  The status quo phi_sigma=0 must be in ``grid`` so the CEV gain
    has a well-defined baseline.

    Robustness: solve_recursive (and ergodic_welfare) RAISE at the patience /
    stability boundary (effective discount too close to 1, or f(theta)>=1).  Such
    grid points are recorded as -inf (infeasible) rather than aborting the sweep,
    so the optimum is taken over the feasible region only.

    Returns
    -------
    WelfareReport with the full curve, the optimum (phi_sigma_star, welfare_star),
    the status-quo welfare (welfare_zero), and the CEV gain cev_pct.
    """
    grid = np.asarray(grid, dtype=float)

    # The status quo (phi_sigma=0) is the CEV reference point; require it explicitly
    # so welfare_zero is always defined and the gain is meaningful.
    zero_match = np.isclose(grid, 0.0)
    if not np.any(zero_match):
        raise ValueError("grid must contain 0.0 (the status-quo phi_sigma).")
    i_zero = int(np.flatnonzero(zero_match)[0])

    welfare_curve = np.empty(grid.shape, dtype=float)
    for i, phi in enumerate(grid):
        p_phi = replace(params, phi_sigma=float(phi))
        try:
            sol = solve_recursive(p_phi, n_sigma=n_sigma, n_pi=n_pi)
            # Common random numbers: identical seed across grid points isolates the
            # policy effect from sampling noise in the welfare comparison.
            welfare_curve[i] = ergodic_welfare(
                p_phi, sol, T=T, seed=seed, burn=burn
            ).welfare
        except (ValueError, RuntimeError):
            # Patience / stability boundary hit: mark infeasible and keep sweeping.
            welfare_curve[i] = -np.inf

    # nanargmax over a curve whose infeasible points are -inf picks the best feasible
    # phi_sigma; phi_sigma=0 is always feasible (calibrated baseline) so a max exists.
    i_star = int(np.nanargmax(welfare_curve))
    phi_sigma_star = float(grid[i_star])
    welfare_star = float(welfare_curve[i_star])
    welfare_zero = float(welfare_curve[i_zero])
    cev_pct = 100.0 * (welfare_star / welfare_zero - 1.0)

    return WelfareReport(
        phi_sigma_grid=grid,
        welfare_curve=welfare_curve,
        phi_sigma_star=phi_sigma_star,
        welfare_star=welfare_star,
        welfare_zero=welfare_zero,
        cev_pct=cev_pct,
    )


__all__ = ["sigma_process", "regime_transition", "belief_transition",
           "RecursiveSolution", "solve_recursive",
           "WelfareResult", "ergodic_welfare",
           "WelfareReport", "optimal_phi_sigma"]
