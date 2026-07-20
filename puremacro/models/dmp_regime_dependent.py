"""DMP search-and-matching with regime-dependent precautionary multiplier
and regime-dependent productivity bump.

Setup:
  - Cobb-Douglas matching m(u, v) = mu * u^alpha * v^(1-alpha)
  - Tightness theta = v/u; job-filling q(theta) = mu * theta^(-alpha)
  - Free entry: c_eff = beta_eff * q(theta) * J
  - Tight regime: c_eff = c * (1 + eta * sigma * 1[R=H])  (precautionary vacancy cost)
  - Loose regime: y_eff = y_bar + kappa * sigma * 1[R=L]  (TFP bump from AI-narrative
    shock — easier financing of AI capex signals expected productivity gains)
  - Discount factor: beta_eff = beta_bar  (constant; no longer carries the loose channel)
  - Wage: Hall (2005) sticky-wage rule, w_t = psi * w_{t-1} + (1 - psi) * w_t^Nash
    with w^Nash = (1 - eta_b) z + eta_b (y_eff + c_eff theta). Hosios: eta_b = alpha.
    psi=0 collapses to the flexible-wage Nash bargain.
  - Separation s constant; leisure z constant

The TFP-bump channel replaces an earlier formulation that pushed the loose-regime
response through beta. The discount-factor channel was numerically too weak under
the admissibility constraint beta_eff*(1-s) < 1; the productivity channel has no
such constraint and is far better identified.

Dynamic IRF (perfect-foresight shooting + fixed-point on the wage path):
``dmp_irf`` constructs a deterministic transition path. State variables:
unemployment u_t (predetermined) and lagged wage w_{t-1} (predetermined). Forward-
looking: tightness theta_t (equivalently J_t, v_t) pinned by free entry against
the future stream of profits. The shock sigma_t follows an AR(1) decay
sigma_shock * rho^t and reverts to zero at horizon T_simul, where the economy is
at the sigma=0 steady state.

Solution algorithm (damped fixed-point iteration on the wage path):
  1. Initialize w_t = w_ss for all t.
  2. Backward-shoot given the wage path. With w_t fixed, the period Bellman
     J_t = (y_eff(t) - w_t) + beta (1-s) J_{t+1} is linear in J_{t+1}; the
     free-entry equation c_eff = beta * mu * theta^{-alpha} * J_t admits the
     closed form theta_t = (beta * mu * J_t / c_eff)^{1/alpha}.
  3. Compute candidate wage path
     w_tilde_t = psi * w_tilde_{t-1} + (1 - psi) * w_t^Nash(theta_t),
     with w_tilde_0 = w_ss.
  4. Damped update: w_{k+1} = (1 - lam) w_k + lam * w_tilde, with lam in (0, 1].
     Default lam = 0.3.
  5. Iterate until max_t |delta w_t| < 1e-6 or 50 iterations.
  6. Forward-iterate u_t.

The damping is needed because the composed map (w -> backward J -> theta -> Nash
-> w) has Jacobian-magnitude > 1 for the H-regime channel (c_eff bumped by sigma),
producing oscillating overshoot at lam = 1. The damping never alters the fixed
point; it only ensures stable approach.

Note that even at psi = 0 the boundary w_0 = w_ss imposes a one-period rigidity
of the wage with respect to the t=0 shock — the wage that was set at t = -1
before the shock realization. This is the explicit lagged-wage state of the Hall
(2005) construction. The earlier brentq-based solver did not impose this
boundary and so produced larger same-period theta responses; the fixed-point
solver is the structurally correct one given the lagged-wage state.

IRF time-index alignment: u_t is predetermined (u_0 = u_ss), so the model's log u
response at t=0 is mechanically zero. The empirical LP measures the within-quarter
urate response, which is non-zero at h=0. Standard DMP-vs-LP alignment shifts the
model index by one: model t = h + 1 corresponds to empirical horizon h. The "first
quarter of measurable response" in the model is t=1, after one period of matching
has actually happened. Returned IRFs are therefore sliced at t=1..horizon+1 from
the simulated paths, in log deviations from the sigma=0 steady state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class DMPParameters:
    beta_bar: float
    alpha: float
    eta_b: float
    s: float
    z: float
    y_bar: float
    c: float
    mu: float
    eta: float
    kappa: float
    rho_sigma: float = 0.7   # AR(1) decay of the uncertainty shock; default for backwards compat
    psi: float = 0.0         # Hall (2005) wage stickiness; psi=0 ⇒ flexible Nash
    alpha_react: float = 1.0 # Active share of unemployment in matching (lagged-u friction)
                             # 1.0 = current model (no friction); 0.5 = half of u_t is
                             # reallocating, only previous half is actively searching
                             # Combined with Hall stickiness this delivers hump-shaped IRFs.
    phi_AD: float = 0.0      # GE feedback: y_eff_t decreases by phi_AD * log(u_t/u_ss).
                             # 0 = no AD multiplier; positive = unemployment drag on output
                             # creates u↑ → y↓ → J↓ → v↓ → u↑ amplification loop.
    alpha_k: float = 0.0     # Capital share in production: y_eff_t += alpha_k * log(K_t/K_ss).
                             # 0 = no capital channel; standard macro ≈ 1/3.
    lambda_k: float = 1.0    # Capital adjustment speed: K_t = (1-lambda)K_{t-1} + lambda K*(n_t).
                             # 1.0 = instant adjustment (K = K*); small = sluggish K, larger
                             # persistence in y_eff and stronger labor-side propagation.
    alpha_N: float = 0.0     # Firm-mass effect on y_eff: y_eff_t += alpha_N * log(N_t/N_ss).
                             # 0 = no firm-extensive-margin channel. N_t partial-adjusts
                             # toward a profit-driven target so under H profits drop, marginal
                             # firms exit, aggregate productive capacity shrinks, amplifying u.
    lambda_N: float = 1.0    # Firm-mass adjustment speed: N_t = (1-lambda_N)N_{t-1} + lambda_N N*(J_t).
                             # Small lambda_N gives sluggish firm dynamics matching BDS rates
                             # (~25%/yr aggregate entry+exit ↔ lambda_N ~ 0.25-0.50).
    nu_lfpr: float = 0.0     # Auxiliary participation channel: log_lfpr_t responds to sigma
                             # by -nu_lfpr * sigma * rho^h under H and +nu_lfpr * sigma * rho^h
                             # under L. Used only in the explicit-N robustness specification of
                             # the matching-margin mechanism (Appendix); the IRFs on (u, v) are
                             # absorbed by the matching channel lambda_mu in the main model.
    lambda_mu: float = 0.0   # Regime-asymmetric matching-efficiency shock: mu_t = mu * (1 -
                             # lambda_mu * sigma_t * 1[R=H]). Under H, matching technology
                             # itself contracts, directly raising u for any (u,v); this is the
                             # structural counterpart of the empirical Beveridge-curve
                             # sign-reversal documented in the LP section. 0 = no matching shock.
    phi_v: float = 0.0       # Quadratic vacancy-adjustment cost (Pissarides-Wasmer): the effective
                             # cost firms face is c + phi_v * (v - v_lag). Steady state is
                             # invariant to phi_v (adjustment terms vanish at v == v_lag); the
                             # IRF develops a hump because v cannot jump on impact. 0 = standard
                             # DMP no-adjustment-cost case.


@dataclass
class DMPState:
    theta: float
    u: float
    v: float
    J: float
    w: float
    free_entry_residual: float


def _regime_effects(
    p: DMPParameters, sigma: float, regime: Literal["H", "L"]
) -> tuple[float, float]:
    """Return (c_eff, y_eff) for given sigma and regime."""
    c_eff = p.c * (1.0 + p.eta * sigma * (1.0 if regime == "H" else 0.0))
    y_eff = p.y_bar + p.kappa * sigma * (1.0 if regime == "L" else 0.0)
    return c_eff, y_eff


def _free_entry_residual(
    theta: float, p: DMPParameters, sigma: float, regime: Literal["H", "L"],
) -> float:
    c_eff, y_eff = _regime_effects(p, sigma, regime)
    beta_eff = p.beta_bar
    q_theta = p.mu * theta ** (-p.alpha)
    w = (1.0 - p.eta_b) * p.z + p.eta_b * (y_eff + c_eff * theta)
    denom = 1.0 - beta_eff * (1.0 - p.s)
    if denom <= 0:
        raise ValueError(
            f"Model inadmissible: beta_eff(1-s) = {beta_eff * (1.0 - p.s):.4f} >= 1 "
            f"(sigma={sigma}, regime={regime})."
        )
    J = (y_eff - w) / denom
    return c_eff - beta_eff * q_theta * J


def dmp_steady_state(
    p: DMPParameters, sigma: float = 0.0, regime: Literal["H", "L"] = "H",
) -> DMPState:
    """Solve the DMP steady state at a given regime and sigma.

    The steady state is independent of psi: when w is constant the sticky
    rule w = psi*w + (1-psi)*w^Nash collapses to w = w^Nash regardless of psi.

    Raises
    ------
    ValueError
        If beta_eff*(1-s) >= 1. With the new spec, beta_eff = beta_bar is constant,
        so this only fires for misconfigured fixed parameters. The check is kept
        for robustness.
    """
    try:
        theta = brentq(_free_entry_residual, 1e-4, 1e3, args=(p, sigma, regime),
                       xtol=1e-10, rtol=1e-10)
    except ValueError as exc:
        raise ValueError(
            f"dmp_steady_state: failed to bracket free-entry root "
            f"(sigma={sigma}, regime={regime}). Underlying: {exc}"
        ) from exc
    c_eff, y_eff = _regime_effects(p, sigma, regime)
    beta_eff = p.beta_bar
    f_theta = p.mu * theta ** (1.0 - p.alpha)
    u = p.s / (p.s + f_theta)
    v = theta * u
    w = (1.0 - p.eta_b) * p.z + p.eta_b * (y_eff + c_eff * theta)
    J = (y_eff - w) / (1.0 - beta_eff * (1.0 - p.s))
    res = _free_entry_residual(theta, p, sigma, regime)
    return DMPState(theta=theta, u=u, v=v, J=J, w=w, free_entry_residual=res)


def _theta_given_J(p: DMPParameters, J: float, c_eff: float) -> float:
    """Closed-form theta from free entry given J and c_eff.

    Free entry: c_eff = beta * mu * theta^{-alpha} * J
    => theta = (beta * mu * J / c_eff)^{1/alpha}.

    Returns a small positive value if J<=0 (job-creation collapsed); the
    forward-iteration on u still respects f(theta) = mu * theta^{1-alpha}
    going to ~0.
    """
    if J <= 0.0:
        return 1e-12
    return (p.beta_bar * p.mu * J / c_eff) ** (1.0 / p.alpha)


def _theta_given_J_phi_v(
    p: DMPParameters,
    J: float,
    c_eff: float,
    u_t: float,
    mu_t: float,
    v_lag: float,
    v_next: float,
) -> float:
    """Solve free-entry for theta given J, u, and the phi_v adjustment terms.

    Equation:
        c_eff + phi_v * (theta * u - v_lag)
              - beta * phi_v * (v_next - theta * u)
              = beta * mu_t * theta^(-alpha) * J

    Reduces to the closed form when phi_v == 0.

    v_lag and v_next come from the previous fixed-point iteration's v_path.
    brentq finds the root over [1e-6, 1e2] like dmp_steady_state.
    """
    if J <= 0.0:
        return 1e-12
    if p.phi_v == 0.0:
        return _theta_given_J(p, J, c_eff)

    def residual(theta: float) -> float:
        v = theta * u_t
        lhs = c_eff + p.phi_v * (v - v_lag) - p.beta_bar * p.phi_v * (v_next - v)
        rhs = p.beta_bar * mu_t * (theta ** (-p.alpha)) * J
        return lhs - rhs

    try:
        # Bracket: standard theta range
        f_lo = residual(1e-6)
        f_hi = residual(1e2)
        if f_lo * f_hi > 0:
            # No sign change in the bracket — fall back to closed form.
            return _theta_given_J(p, J, c_eff)
        return brentq(residual, 1e-6, 1e2, xtol=1e-10, maxiter=200)
    except Exception:
        return _theta_given_J(p, J, c_eff)


@dataclass
class DMPIRF:
    horizons: np.ndarray
    log_u: np.ndarray
    log_v: np.ndarray
    log_theta: np.ndarray
    log_lfpr: Optional[np.ndarray] = None  # only populated when nu_lfpr > 0 (explicit-N variant)


def dmp_irf(
    p: DMPParameters,
    *,
    sigma_shock: float,
    regime: Literal["H", "L"],
    horizon: int = 8,
    rho_sigma: float | None = None,
    T_simul: int = 60,
    fp_tol: float = 1e-6,
    fp_max_iter: int = 50,
    fp_damping: float = 0.3,
) -> DMPIRF:
    """Deterministic perfect-foresight IRF of the DMP model with sticky wages.

    The shock sigma_t = sigma_shock * rho_sigma^t for t < T_simul and 0 for
    t >= T_simul. At t = T_simul the economy is anchored at the sigma=0 steady
    state (boundary condition).

    Under sticky wages the wage path w_t depends on the lagged wage w_{t-1} and
    on tightness theta_t. We solve by fixed-point iteration on the wage path:
    given a candidate w_t^(k), a backward sweep pins theta_t and J_t in closed
    form, and a forward sweep updates the wage path with w_t^{k+1} = psi *
    w_{t-1}^{k+1} + (1-psi) * w_t^Nash(theta_t^(k)). The boundary w_0 = w_ss.

    Parameters
    ----------
    p : DMPParameters
    sigma_shock : float
        Initial shock size at t=0.
    regime : {"H", "L"}
    horizon : int, default 8
        Number of empirical horizons to return (output length = horizon+1).
        Under the +1 model-data alignment, this corresponds to simulating
        through model t = horizon + 1, so T_simul must exceed horizon + 1.
    rho_sigma : float or None, default None
        AR(1) decay of the sigma shock. If None, uses ``p.rho_sigma``.
    T_simul : int, default 60
        Length of the simulation. Must exceed horizon + 1.
    fp_tol : float, default 1e-6
        Convergence tolerance for the wage-path fixed point.
    fp_max_iter : int, default 50
        Maximum fixed-point iterations.
    fp_damping : float, default 0.3
        Relaxation parameter in (0, 1]: w_{k+1} = (1 - lam) w_k + lam * R(w_k),
        where R is the backward-sweep+Nash update map. lam=1 is the undamped
        iteration; smaller values trade extra iterations for stability when
        the composed map's Jacobian has modulus >= 1 (e.g., at psi=0 with the
        c_eff bumping channel).

    Raises
    ------
    ValueError
        If the model is inadmissible at any point along the path.
    """
    if rho_sigma is None:
        rho_sigma = p.rho_sigma
    if T_simul <= horizon + 1:
        raise ValueError(
            f"dmp_irf: T_simul (={T_simul}) must exceed horizon + 1 "
            f"(={horizon + 1}) under the +1 model-data alignment."
        )

    # Boundary: sigma=0 steady state.
    ss = dmp_steady_state(p, sigma=0.0, regime=regime)
    u_ss = ss.u
    v_ss = ss.v
    theta_ss = ss.theta
    J_ss = ss.J
    w_ss = ss.w

    beta_eff = p.beta_bar
    denom_const = 1.0 - beta_eff * (1.0 - p.s)
    if denom_const <= 0:
        raise ValueError(
            f"dmp_irf: beta_bar*(1-s)={beta_eff*(1.0-p.s):.4f} >= 1; inadmissible."
        )

    # Shock path: t = 0, ..., T_simul. sigma_{T_simul} = 0 so the boundary is
    # consistent with the sigma=0 steady state.
    sigma_path = np.zeros(T_simul + 1)
    sigma_path[: T_simul] = sigma_shock * (rho_sigma ** np.arange(T_simul))
    # sigma_path[T_simul] stays at 0 (boundary).

    # Precompute (c_eff, y_eff_base) along the path. y_eff_base is the part
    # that does NOT depend on u or K (it only depends on sigma and regime);
    # the full y_eff_path = y_eff_base - phi_AD * log(u/u_ss) + alpha_k * log(K/K_ss)
    # is recomputed inside the iteration as u and K co-evolve.
    c_eff_path = np.empty(T_simul + 1)
    y_eff_base = np.empty(T_simul + 1)
    for t in range(T_simul + 1):
        c_eff_path[t], y_eff_base[t] = _regime_effects(
            p, float(sigma_path[t]), regime
        )

    # Matching-efficiency path: mu_t = mu * (1 - lambda_mu * sigma_t * 1[R=H]).
    # Under H, matching technology itself contracts, directly producing
    # the Beveridge-curve shift documented empirically. Floored at mu*0.01
    # to keep the dynamics well-defined under large shocks.
    matching_shock_fires = 1.0 if regime == "H" else 0.0
    mu_path = np.maximum(
        p.mu * (1.0 - p.lambda_mu * matching_shock_fires * sigma_path),
        p.mu * 0.01,
    )

    # Initialize wage, unemployment, capital, and firm-mass paths at their
    # steady states. Capital and firm mass are tracked in log deviation
    # (log_k_path = log(K/K_ss), log_N_path = log(N/N_ss)); both are zero at ss.
    w_path = np.full(T_simul + 1, w_ss)
    u_path = np.full(T_simul + 1, u_ss)
    log_k_path = np.zeros(T_simul + 1)
    log_N_path = np.zeros(T_simul + 1)
    n_ss = 1.0 - u_ss

    theta_path = np.zeros(T_simul + 1)
    J_path = np.zeros(T_simul + 1)
    v_path = np.full(T_simul + 1, v_ss)

    # GE feedback (phi_AD on log(u/u_ss)) fires ONLY under L regime.
    # Rationale: the AD-multiplier loop u↑ → y↓ → J↓ → v↓ → u↑ is unstable
    # under H (compounds with the eta-driven c_eff bump and hits a numerical
    # cliff), but the L-loop u↓ → y↑ → J↑ → v↑ → u↓ converges. Suppressing
    # the H-side GE feedback lets eta operate without the cliff, while the
    # L-side amplification still propagates the kappa-driven expansion.
    # Capital channel (alpha_k on log(K/K_ss)) remains regime-symmetric.
    # Firm-mass channel (alpha_N on log(N/N_ss)) fires ONLY under H regime,
    # mirroring GE's asymmetry on the opposite side: under H, profits drop,
    # marginal firms exit, the extensive-margin contraction amplifies u;
    # under L, no firm-mass channel (the J↑ spiral would otherwise saturate).
    ge_fires = 1.0 if regime == "L" else 0.0
    firm_fires = 1.0 if regime == "H" else 0.0

    for it in range(fp_max_iter):
        # ---- Update y_eff_path given current (u_path, log_k_path, log_N_path). ----
        log_u_dev = np.log(np.maximum(u_path, 1e-12) / u_ss)
        y_eff_path = (
            y_eff_base
            - ge_fires * p.phi_AD * log_u_dev
            + p.alpha_k * log_k_path
            + firm_fires * p.alpha_N * log_N_path
        )

        # ---- Backward sweep: given w_path, y_eff_path, get theta_path and J_path. ----
        # Free entry uses regime-asymmetric mu_t: c = beta * mu_t * theta^{-alpha} * J,
        # so theta_t = (beta * mu_t * J / c_eff)^(1/alpha). Lower mu_t under H => lower
        # theta_t at given J => lower vacancy posting at given J.
        theta_path[T_simul] = theta_ss
        J_path[T_simul] = J_ss
        for t in range(T_simul - 1, -1, -1):
            J_t = (y_eff_path[t] - w_path[t]) + beta_eff * (1.0 - p.s) * J_path[t + 1]
            J_path[t] = J_t
            if J_t <= 0.0:
                theta_path[t] = 1e-12
            else:
                # u_t for the phi_v adjustment uses u_path from prev iter (or u_ss at t=0)
                u_t = u_path[t] if t > 0 else u_ss
                v_lag = v_path[t - 1] if t > 0 else v_ss
                v_next = v_path[t + 1] if t < T_simul else v_ss
                theta_path[t] = _theta_given_J_phi_v(
                    p, J_t, c_eff_path[t], u_t, mu_path[t], v_lag, v_next,
                )

        # ---- Forward sweep: candidate (u_path, w_path). ----
        # u is predetermined at t=0 (u_0 = u_ss); we re-derive u_t from theta_{t-1}
        # via the matching law of motion, with the reallocation friction.
        # Aggregate matching uses regime-asymmetric mu_t directly: m_t = mu_t * u^a * v^(1-a).
        u_new = np.zeros(T_simul + 1)
        u_new[0] = u_ss
        for t in range(T_simul):
            if t == 0:
                u_active = u_new[t]
            else:
                u_active = (
                    p.alpha_react * u_new[t]
                    + (1.0 - p.alpha_react) * u_new[t - 1]
                )
            v_t = theta_path[t] * u_new[t]
            m_t = mu_path[t] * u_active ** p.alpha * v_t ** (1.0 - p.alpha)
            m_t = min(m_t, u_active)
            u_new[t + 1] = max(1e-10, u_new[t] - m_t + p.s * (1.0 - u_new[t]))

        # Capital partial-adjustment: log_k_t = (1-lambda_k) log_k_{t-1} + lambda_k * log(n_t/n_ss).
        # K target scales 1:1 with employment in Cobb-Douglas (K* ∝ n in steady state).
        log_n_new = np.log(np.maximum(1.0 - u_new, 1e-12) / n_ss)
        log_k_new = np.zeros(T_simul + 1)
        for t in range(1, T_simul + 1):
            log_k_new[t] = (
                (1.0 - p.lambda_k) * log_k_new[t - 1]
                + p.lambda_k * log_n_new[t]
            )

        # Firm-mass partial-adjustment: log_N_t = (1-lambda_N) log_N_{t-1} + lambda_N * log(J_t/J_ss).
        # Firms enter/exit toward a profit-driven target N*(J_t). Under H, J drops →
        # marginal firms exit → log_N falls → y_eff drops by alpha_N*log_N → J drops
        # more, the extensive-margin amplification loop. Slow lambda_N matches the
        # ~25%/yr aggregate firm entry+exit rate documented in BDS.
        log_J_dev = np.log(np.maximum(J_path, 1e-12) / J_ss)
        log_N_new = np.zeros(T_simul + 1)
        for t in range(1, T_simul + 1):
            log_N_new[t] = (
                (1.0 - p.lambda_N) * log_N_new[t - 1]
                + p.lambda_N * log_J_dev[t]
            )

        # Wage update: Nash bargaining + Hall stickiness.
        w_tilde = np.empty(T_simul + 1)
        w_tilde[0] = w_ss
        for t in range(1, T_simul + 1):
            w_nash_t = (
                (1.0 - p.eta_b) * p.z
                + p.eta_b * (y_eff_path[t] + c_eff_path[t] * theta_path[t])
            )
            w_tilde[t] = p.psi * w_tilde[t - 1] + (1.0 - p.psi) * w_nash_t

        # Damped updates on all four paths.
        w_path_new = (1.0 - fp_damping) * w_path + fp_damping * w_tilde
        u_path_new = (1.0 - fp_damping) * u_path + fp_damping * u_new
        log_k_path_new = (1.0 - fp_damping) * log_k_path + fp_damping * log_k_new
        log_N_path_new = (1.0 - fp_damping) * log_N_path + fp_damping * log_N_new

        # Update v_path for next iteration's phi_v term (damped like u_path).
        v_new = theta_path * u_path_new
        v_path = (1.0 - fp_damping) * v_path + fp_damping * v_new

        diff_w = float(np.max(np.abs(w_path_new - w_path)))
        diff_u = float(np.max(np.abs(u_path_new - u_path)))
        diff_k = float(np.max(np.abs(log_k_path_new - log_k_path)))
        diff_N = float(np.max(np.abs(log_N_path_new - log_N_path)))
        w_path = w_path_new
        u_path = u_path_new
        log_k_path = log_k_path_new
        log_N_path = log_N_path_new
        if max(diff_w, diff_u, diff_k, diff_N) < fp_tol:
            break

    # Compute v_path from final theta_path and u_path.
    for t in range(T_simul + 1):
        v_path[t] = theta_path[t] * u_path[t]

    # Slice to requested horizon and convert to log deviations.
    # Model t = h + 1 ↔ empirical h: model t=0 is mechanically zero in u
    # because u is predetermined; the empirical LP at h=0 measures the
    # within-quarter response, which corresponds to model t=1.
    h_idx = np.arange(horizon + 1)
    log_u = np.log(np.maximum(u_path[1 : horizon + 2], 1e-12)) - np.log(u_ss)
    log_v = np.log(np.maximum(v_path[1 : horizon + 2], 1e-12)) - np.log(v_ss)
    log_theta = (
        np.log(np.maximum(theta_path[1 : horizon + 2], 1e-12)) - np.log(theta_ss)
    )

    # Auxiliary participation channel (only used in the explicit-N variant).
    # log_lfpr_t = -nu_lfpr * sigma * rho^h under H, +nu_lfpr * sigma * rho^h under L.
    # Reduces to None at nu_lfpr=0 to preserve backwards compatibility.
    if p.nu_lfpr != 0.0:
        sign = -1.0 if regime == "H" else +1.0
        log_lfpr_path = sign * p.nu_lfpr * sigma_path
        log_lfpr = log_lfpr_path[1 : horizon + 2]
    else:
        log_lfpr = None

    return DMPIRF(
        horizons=h_idx,
        log_u=log_u,
        log_v=log_v,
        log_theta=log_theta,
        log_lfpr=log_lfpr,
    )
