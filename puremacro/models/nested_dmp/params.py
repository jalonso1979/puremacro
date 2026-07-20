"""Parameter object for the nested heterogeneous-firm DMP model.

Six layers:
  1. Matching / DMP core           beta, alpha, mu, s_bar, b
  2. Heterogeneous-firm selection  rho_x, sigma_x, n_x
  3. Non-convex hiring              f_fixed, c_convex
  4. Asymmetric wage rigidity       psi_down
  5. Information / beliefs          r_star, phi_D, phi_H, signal_sd, prior_pi0
  6. Policy lever + shock           phi_sigma, rho_sigma, delta
  7. Exogenous σ-process + regime   sigma_bar, sigma_innov_sd, p_regime

Hosios efficiency is the baseline (worker bargaining share = matching
elasticity = alpha). phi_D < 0 < phi_H encodes "dove cuts, hawk hikes"
on AI-narrative uncertainty; phi_sigma is the augmented-rule response the
welfare experiment will optimise (default 0 = current "no-response" rule).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NestedDMPParameters:
    # Layer 1: matching / DMP core
    beta: float = 0.99       # quarterly discount factor
    alpha: float = 0.5       # matching elasticity (Hosios: bargaining share = alpha)
    mu: float = 1.0          # matching efficiency
    s_bar: float = 0.03      # baseline separation rate
    b: float = 0.4           # UI replacement ratio (b/y)
    # Layer 2: heterogeneous-firm selection
    rho_x: float = 0.95      # AR(1) persistence of match productivity x
    sigma_x: float = 0.10    # innovation sd of x at sigma=0
    n_x: int = 25            # productivity grid size
    sigma_x_loading: float = 0.0   # uncertainty sigma's loading on the idiosyncratic innovation sd (MPS selection); 0 = OFF (sigma -> beliefs only)
    # Layer 3: non-convex hiring
    f_fixed: float = 0.05    # fixed vacancy-creation cost (inaction-region source)
    c_convex: float = 0.20   # convex per-vacancy cost / output
    # Layer 4: asymmetric wage rigidity
    psi_down: float = 0.0    # downward wage rigidity (wages fall sluggishly); 0 = flexible (opt-in)
    # Layer 5: information / beliefs
    r_star: float = 0.01     # steady-state real rate
    phi_D: float = -0.0320   # dove cut per sigma-LTUI, DECIMAL (-320 bps; NB43 h=2)
    phi_H: float = 0.015     # hawk hike per sigma-LTUI, DECIMAL (+150 bps)
    signal_sd: float = 1.0   # rate-signal noise sd for the belief update
    prior_pi0: float = 0.5   # prior Pr(Fed = dove)
    # Layer 6: policy lever + shock + firm exit
    phi_sigma: float = 0.0   # augmented-rule uncertainty response (welfare lever)
    rho_sigma: float = 0.7   # AR(1) persistence of the LTUI shock
    delta: float = 0.0       # exogenous firm-slot exit hazard (extensive margin); 0 = off
    h_max: float = 0.0       # home-value dispersion U[0,h_max] for the N<->U margin; 0 = participation off
    # Layer 7: exogenous σ-process + regime (welfare phase-5 backbone)
    # sigma_bar=0.0 is the normal-times mean; the Tauchen grid is centred there
    # but spans ±m*sigma_innov_sd/sqrt(1-rho_sigma^2) (unconditional sd) and
    # is then clipped to [0,∞), so the grid always covers strictly positive σ
    # even at the zero-mean calibration (the right half of the symmetric span).
    sigma_bar: float = 0.0   # long-run mean uncertainty level (>= 0)
    sigma_innov_sd: float = 0.5  # innovation sd of the σ AR(1) process (> 0)
    p_regime: float = 0.9    # regime stay-probability (symmetric 2-state Markov; in (0,1))

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 1.0):
            raise ValueError(f"beta must be in (0,1); got {self.beta}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1); got {self.alpha}")
        if not (0.0 < self.s_bar < 1.0):
            raise ValueError(f"s_bar must be in (0,1); got {self.s_bar}")
        if self.mu <= 0.0:
            raise ValueError(f"mu must be > 0; got {self.mu}")
        if not (0.0 <= self.rho_x < 1.0):
            raise ValueError(f"rho_x must be in [0,1); got {self.rho_x}")
        if self.sigma_x <= 0.0:
            raise ValueError(f"sigma_x must be > 0; got {self.sigma_x}")
        if self.n_x < 2:
            raise ValueError(f"n_x must be >= 2; got {self.n_x}")
        if not (0.0 <= self.psi_down < 1.0):
            raise ValueError(f"psi_down must be in [0,1); got {self.psi_down}")
        if self.signal_sd <= 0.0:
            raise ValueError(f"signal_sd must be > 0; got {self.signal_sd}")
        if not (0.0 <= self.prior_pi0 <= 1.0):
            raise ValueError(f"prior_pi0 must be in [0,1]; got {self.prior_pi0}")
        if self.phi_D >= 0.0:
            raise ValueError(f"phi_D must be < 0 (dove cuts); got {self.phi_D}")
        if self.phi_H <= 0.0:
            raise ValueError(f"phi_H must be > 0 (hawk hikes); got {self.phi_H}")
        if not (0.0 <= self.delta < 1.0):
            raise ValueError(f"delta must be in [0,1); got {self.delta}")
        if self.h_max < 0.0:
            raise ValueError(f"h_max must be >= 0 (0 = participation off); got {self.h_max}")
        if self.sigma_x_loading < 0.0:
            raise ValueError(f"sigma_x_loading must be >= 0; got {self.sigma_x_loading}")
        # beta(r)=1/(1+r) needs r = r_star + (phi_tau - phi_sigma)*sigma > -1 on the
        # intended |sigma|<=2 support, or the discount map crosses the r=-1 pole.
        worst_r = self.r_star + (self.phi_D - self.phi_sigma) * 2.0  # dove, sigma=2
        if worst_r <= -1.0:
            raise ValueError(
                f"phi_D too large: perceived rate {worst_r:.3f} <= -1 crosses the "
                f"beta(r)=1/(1+r) pole at |sigma|=2 (admissibility)."
            )
        if self.sigma_bar < 0.0:
            raise ValueError(f"sigma_bar must be >= 0; got {self.sigma_bar}")
        if self.sigma_innov_sd <= 0.0:
            raise ValueError(f"sigma_innov_sd must be > 0; got {self.sigma_innov_sd}")
        if not (0.0 < self.p_regime < 1.0):
            raise ValueError(f"p_regime must be in (0,1); got {self.p_regime}")


__all__ = ["NestedDMPParameters"]
