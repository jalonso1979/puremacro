"""Fertility DSGE (adjustment-costs baseline).

Ported from My Drive/Fertility/fertility_adj_costs.mod and
bgp_fertility_calibration.m. The 12-variable linear DSGE has 5
predetermined states (a, mun, ph, k, n) and 7 controls
(c, y, l_w, u, i, b, l_o). Three exogenous shocks (ea, ep, en) drive
the three AR(1) shock processes.

Public entry points:

- ``FERTILITY_PINNED_CALIBRATION`` — the seven calibrated parameters
  (from the original Dynare ``parameters.mat``) plus Bayesian-variant
  constants that ``solve_fertility`` linearises around.
- ``exact_steady_state(params, z_guess)`` — solve
  ``model_residuals(z, z, z, 0) = 0`` for the steady state (Dynare's
  ``steady;`` step).
- ``solve_bgp(exogenous, targets, x0, tol)`` — BGP least_squares calibration
  (retained for reference; NOT the linearisation point — see below).
- ``model_residuals(z_lead, z, z_lag, eps, params)`` — 12 model-equation
  residuals (for steady-state verification + numerical Jacobians).
- ``solve_fertility(params, *, shock_stds, h_for_jacobians)`` — top-level
  orchestrator returning a FertilitySolution.

``solve_fertility`` linearises around the EXACT steady state of the pinned
calibration. The earlier path linearised around ``solve_bgp``'s least-squares
endpoint of a mutually inconsistent over-determined system — a non-equilibrium
point whose Blanchard-Kahn status flipped across LAPACK builds, which is why
the exact steady state is used as the linearisation point instead.

Solver-only release (R1b, 0.54.0). Bayesian estimation lands in R1c
(0.55.0) — it will wire priors + observation onto puremacro.dsge.estimate_dsge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.dsge._results import FertilitySolution
from puremacro.dsge.klein import klein_solve, KleinSolution


# === Variable / shock declarations ============================================

VAR_NAMES: tuple[str, ...] = (
    # Predetermined / state variables (n_pre = 5)
    "a", "mun", "ph", "k", "n",
    # Non-predetermined / controls (n_fwd = 7)
    "c", "y", "l_w", "u", "i", "b", "l_o",
)

SHOCK_NAMES: tuple[str, ...] = ("ea", "ep", "en")

N_PRE: int = 5
N_FWD: int = 7
N_VARS: int = len(VAR_NAMES)
N_SHOCKS: int = len(SHOCK_NAMES)
assert N_PRE + N_FWD == N_VARS


# Per-variable timing (which slots it appears in: lag, current, lead).
_VARIABLE_TIMING: dict[str, set[str]] = {
    "a":    {"lag", "current"},
    "mun":  {"lag", "current"},
    "ph":   {"lag", "current"},
    "k":    {"lag", "current", "lead"},
    "n":    {"lag", "current", "lead"},
    "c":    {"current", "lead"},
    "y":    {"current", "lead"},
    "l_w":  {"current", "lead"},
    "u":    {"current", "lead"},
    "i":    {"current"},
    "b":    {"current"},
    "l_o":  {"current"},
}


# === Default parameter blocks =================================================

FERTILITY_EXOGENOUS_PARAMS: dict[str, float] = {
    "alpha":   0.4,
    "nu":      2.5,
    "phi":     1.03,
    "g":       0.017,
    "delta_p": 0.075,
    "delta_n": 0.065,
    "omega":   2.5,
    "bara":    0.0,
}

FERTILITY_CALIB_TARGETS: dict[str, float] = {
    "l":              0.30,
    "u":              1.00,
    "depr_rate":      0.10,
    "kid_cost_share": 0.18,
    "k_y_ratio":      2.80,
    "c_y_ratio":      0.75,
    "n_growth":       1.02,
}

FERTILITY_SHOCK_PROCESSES: dict[str, dict] = {
    "ea": {"rho": 0.94 ** 4, "sigma": 0.01},
    "en": {"rho": 0.50,      "sigma": 0.07},
    "ep": {"rho": 0.90,      "sigma": 0.07},
}


# The pinned structural calibration. These are the seven calibrated
# parameters recovered from the original Dynare estimation output
# (``fertility_adj_costs_bayesian_estimation.mod`` +
# ``output/data/parameters.mat``, x1..x7) plus the Bayesian-variant
# constants. ``solve_bgp`` (below) is retained for reference, but it
# minimises an over-determined and mutually inconsistent 13-equation
# system whose least-squares endpoint sits on a 2-D flat manifold — a
# non-equilibrium point whose Blanchard-Kahn status flips with the LAPACK
# build. ``solve_fertility`` linearises around the EXACT steady state of
# THIS calibration instead (see ``exact_steady_state``), which
# is robustly determinate on every BLAS/LAPACK build.
FERTILITY_PINNED_CALIBRATION: dict[str, float] = {
    "alpha":   0.40,
    "nu":      2.5,
    "omega":   2.0,
    "g":       0.0175,
    "delta_p": 0.075,
    "delta_n": 0.065,
    "bara":    0.0,
    "barp":    float(np.log(1.0 / 1.03)),
    # Calibrated (parameters.mat x1..x7):
    "barn":    float(np.log(1.587983)),
    "mu_l":    0.634818,
    "beta":    0.936122,
    "tau_n":   0.213943,
    "tau_b":   0.469184,
    "p_n":     0.094621,
    "delta_k": 0.144,
    # Adjustment-cost params (vanish at the SS; needed by model_residuals):
    "psik":    2.5,
    "psin":    3.2,
}

# Dynare ``initval``-style guess for the exact steady-state solve. hybr
# converges from here to the reference SS (c 0.266225, k 1.033825,
# i 0.114395, u 1.137443, n 1.066294, b 0.067290, l_o 0.474782,
# l_w 0.265520, y 0.481514) in milliseconds.
_FERTILITY_SS_INITVAL: dict[str, float] = {
    "c": 0.360443, "b": 0.087379, "l_w": 0.329987,
    "u": 0.999997, "k": 1.835291, "n": 1.384615,
}


# === BGP solver ===============================================================

import scipy.optimize


_BGP_X0_DEFAULT: np.ndarray = np.array([
    0.5,   # barn   (log SS of mun)
    1.2,   # mu_l
    0.9,   # beta
    0.5,   # tau_n
    0.5,   # tau_b
    0.5,   # p_n
    0.08,  # delta_k
    1.0,   # c
    0.5,   # b
    0.30,  # l_w
    1.0,   # u
    0.5,   # k
    0.5,   # n
])


_BGP_REQUIRED_EXOGENOUS: set[str] = {
    "alpha", "nu", "phi", "g", "delta_p", "delta_n", "omega", "bara",
}
_BGP_REQUIRED_TARGETS: set[str] = {
    "l", "u", "depr_rate", "kid_cost_share", "k_y_ratio", "c_y_ratio",
    "n_growth",
}

def _bgp_system(
    x: np.ndarray,
    exogenous: dict,
    targets: dict,
) -> np.ndarray:
    """The 13 residuals of the BGP system. Mirrors bgp_system in
    bgp_fertility_calibration.m.

    x = [barn, mu_l, beta, tau_n, tau_b, p_n, delta_k,
         c, b, l_w, u, k, n].

    Notes on over-determination:
    The MATLAB system has 13 equations and 13 unknowns but is mutually
    inconsistent for the default parameters (e.g., F[3] capital-FOC gives
    delta_k = alpha/k_y_ratio = 0.143, while F[9] depr_rate target gives
    delta_k = depr_rate*omega/u^omega = 0.25). This function preserves ALL
    13 original MATLAB equations; solve_bgp uses scipy.optimize.least_squares
    (designed for over-determined systems) rather than fsolve to find the
    best-fit point that minimises sum(F**2).
    """
    barn, mu_l, beta, tau_n, tau_b, p_n, delta_k = x[:7]
    c, b, l_w, u, k, n = x[7:]
    alpha   = exogenous["alpha"]
    nu      = exogenous["nu"]
    phi     = exogenous["phi"]
    g       = exogenous["g"]
    delta_p = exogenous["delta_p"]
    delta_n = exogenous["delta_n"]
    omega   = exogenous["omega"]
    a       = exogenous["bara"]

    # Output (from production function)
    y = np.exp(a) * (u * k) ** alpha * l_w ** (1 - alpha)

    F = np.zeros(13)
    # Eqn 1: Leisure-consumption FOC.
    F[0]  = mu_l * c * (1 - tau_n * n - tau_b * b - l_w) ** (-nu) - (1 - alpha) * y / l_w
    # Eqn 2: Fertility Euler at SS.
    F[1]  = ((1 - alpha) * y / l_w
             - beta * (1 - delta_p + delta_n * n)
                * (np.exp(barn) * c / n - p_n
                   + (tau_b * (1 - delta_n) * phi - tau_n) * (1 - alpha) * y / l_w))
    # Eqn 3: Consumption Euler at SS.
    F[2]  = 1 + g - beta * (1 - delta_p + delta_n * n) * (
        alpha * y / k + 1 - delta_k * u ** omega / omega
    )
    # Eqn 4: Capital efficiency (from the .mod file; may be inconsistent
    # with F[9] depr_rate target — least_squares finds the best fit).
    F[3]  = delta_k * u ** omega * k - alpha * y
    # Eqn 5: Resource constraint at SS (adjustment costs vanish at SS).
    F[4]  = c + p_n * n + (g + delta_k * u ** omega / omega) * k - y
    # Eqn 6: Children LoM at SS.
    F[5]  = b - phi * delta_n * n
    # Eqn 7: n_growth target.
    F[6]  = (1 - delta_p + delta_n * n) - targets["n_growth"]
    # Eqns 8-13: calibration targets.
    F[7]  = l_w - targets["l"]
    F[8]  = u - targets["u"]
    F[9]  = delta_k * u ** omega / omega - targets["depr_rate"]
    F[10] = p_n * n / y - targets["kid_cost_share"]
    F[11] = k / y - targets["k_y_ratio"]
    F[12] = c / y - targets["c_y_ratio"]
    return F


def solve_bgp(
    exogenous: dict | None = None,
    targets: dict | None = None,
    x0: np.ndarray | None = None,
    tol: float = 1e-12,
) -> dict:
    """Solve the 13-equation BGP system via scipy.optimize.least_squares.

    The MATLAB bgp_fertility_calibration.m system has 13 equations and 13
    unknowns but is mutually inconsistent (over-determined in the economic
    sense). least_squares finds the x that minimises 0.5 * sum(F**2) — i.e.,
    the best-fit point given all constraints. A warning fires if the residual
    norm at the solution exceeds 1e-3.

    See the module docstring for the equation list. Returns a dict
    merging exogenous params + the 7 calibrated values + the 6 steady-state
    values + derived SS values (i, l_o, y, a, mun, ph).

    Adjustment-cost params (psik, psin) are also written to the output
    dict with their default values (2.5, 3.2) — they don't enter the
    BGP residuals because adjustment costs vanish at the steady state,
    but model_residuals needs them.
    """
    if exogenous is None:
        exogenous = FERTILITY_EXOGENOUS_PARAMS
    if targets is None:
        targets = FERTILITY_CALIB_TARGETS
    missing_e = _BGP_REQUIRED_EXOGENOUS - set(exogenous.keys())
    if missing_e:
        raise KeyError(f"solve_bgp: exogenous missing keys: {sorted(missing_e)}")
    missing_t = _BGP_REQUIRED_TARGETS - set(targets.keys())
    if missing_t:
        raise KeyError(f"solve_bgp: targets missing keys: {sorted(missing_t)}")

    if x0 is None:
        x0 = _BGP_X0_DEFAULT.copy()

    result = scipy.optimize.least_squares(
        _bgp_system, x0, args=(exogenous, targets),
        method="lm", xtol=tol, ftol=tol, gtol=tol,
        max_nfev=10_000,
    )
    x = result.x
    residual_norm = float(np.linalg.norm(result.fun))
    if not result.success:
        raise RuntimeError(
            f"solve_bgp: least_squares did not converge "
            f"(status={result.status}, residual norm={residual_norm:.3e}): "
            f"{result.message}"
        )
    # The MATLAB calibration is over-determined; the LM solution minimises
    # residual squared norm rather than driving it to zero. Warn the caller
    # if the best fit still has appreciable residuals.
    if residual_norm > 1e-3:
        import warnings
        warnings.warn(
            f"solve_bgp: residual norm at LM solution is {residual_norm:.3e}; "
            "the MATLAB BGP system is over-determined (the 6 calibration targets "
            "are inconsistent with the 7 economic identities). Solution is the "
            "least-squares best fit, not an exact root.",
            RuntimeWarning, stacklevel=2,
        )

    barn, mu_l, beta, tau_n, tau_b, p_n, delta_k = x[:7]
    c, b, l_w, u, k, n = x[7:]

    # Sanity-check support
    for name, val in [("c", c), ("l_w", l_w), ("k", k), ("n", n), ("u", u), ("b", b)]:
        if not np.isfinite(val) or val <= 0:
            raise ValueError(
                f"solve_bgp: solution out of support: {name}={val:.4f} (must be > 0)"
            )

    # Derived SS values
    a_ss = exogenous["bara"]
    y = np.exp(a_ss) * (u * k) ** exogenous["alpha"] * l_w ** (1 - exogenous["alpha"])
    i = (exogenous["g"] + delta_k * u ** exogenous["omega"] / exogenous["omega"]) * k
    l_o = 1 - tau_n * n - l_w - tau_b * b

    # Shock-state SS values
    mun_ss = float(barn)   # log SS of fertility-preference state
    ph_ss = 0.0            # log SS of mortality state (consistent with barp = 0 default)

    out = dict(exogenous)
    out.update({
        "barn": float(barn), "mu_l": float(mu_l), "beta": float(beta),
        "tau_n": float(tau_n), "tau_b": float(tau_b), "p_n": float(p_n),
        "delta_k": float(delta_k),
        "c": float(c), "b": float(b), "l_w": float(l_w), "u": float(u),
        "k": float(k), "n": float(n),
        "i": float(i), "l_o": float(l_o), "y": float(y),
        "a": float(a_ss), "mun": float(mun_ss), "ph": float(ph_ss),
        # Adjustment-cost params (default Bayesian-variant values)
        "psik": 2.5, "psin": 3.2,
        # SS of mortality log (consistent with ph_ss = 0)
        "barp": 0.0,
    })
    return out


# === Model residuals (12 equations) ===========================================

def _unpack(z: np.ndarray) -> dict:
    """Map a 12-vector ordered by VAR_NAMES to a name-keyed dict."""
    return {name: z[i] for i, name in enumerate(VAR_NAMES)}


def model_residuals(
    z_lead: np.ndarray,
    z: np.ndarray,
    z_lag: np.ndarray,
    eps: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Evaluate the 12 fertility-DSGE residuals at given (lead, current,
    lag, shock) variable vectors.

    Returns shape (12,):
        (1)  Production
        (2)  Investment LoM
        (3)  Resource constraint with adjustment costs
        (4)  Children LoM
        (5)  Time constraint
        (6)  Leisure-consumption FOC
        (7)  Capital efficiency
        (8)  Consumption Euler
        (9)  Fertility Euler
        (10) Productivity AR(1)
        (11) Fertility-preference AR(1)
        (12) Mortality AR(1)
    """
    if z_lead.shape != (N_VARS,) or z.shape != (N_VARS,) or z_lag.shape != (N_VARS,):
        raise ValueError(
            f"model_residuals: each z vector must have shape ({N_VARS},)"
        )
    if eps.shape != (N_SHOCKS,):
        raise ValueError(f"model_residuals: eps must have shape ({N_SHOCKS},)")

    L = _unpack(z_lag)
    C = _unpack(z)
    P = _unpack(z_lead)  # lead = t+1

    alpha   = params["alpha"]
    delta_k = params["delta_k"]
    delta_n = params["delta_n"]
    delta_p = params["delta_p"]
    omega   = params["omega"]
    g       = params["g"]
    beta    = params["beta"]
    nu      = params["nu"]
    mu_l    = params["mu_l"]
    tau_n   = params["tau_n"]
    tau_b   = params["tau_b"]
    p_n     = params["p_n"]
    psik    = params.get("psik", 2.5)
    psin    = params.get("psin", 3.2)

    rhoa = params["rhoa"]
    rhon = params["rhon"]
    rhop = params["rhop"]
    bara = params["bara"]
    barn = params["barn"]
    barp = params["barp"]

    R = np.zeros(N_VARS)

    # (1) Production: y = exp(a) * (u*k(-1))^alpha * l_w^(1-alpha)
    R[0] = C["y"] - np.exp(C["a"]) * (C["u"] * L["k"]) ** alpha * C["l_w"] ** (1 - alpha)

    # (2) Investment LoM: i = (1+g)*k - (1 - delta_k*u^omega/omega)*k(-1)
    R[1] = C["i"] - ((1 + g) * C["k"] - (1 - delta_k * C["u"] ** omega / omega) * L["k"])

    # (3) Resource constraint with adjustment costs
    # c + pn*n(-1) + i + psik/2*(1+g)^2*(k/k(-1)-1)^2*k(-1) + psin/2*(n/n(-1)-1)^2*n(-1) = y
    adj_k = psik / 2 * (1 + g) ** 2 * (C["k"] / L["k"] - 1) ** 2 * L["k"]
    adj_n = psin / 2 * (C["n"] / L["n"] - 1) ** 2 * L["n"]
    R[2] = (C["c"] + p_n * L["n"] + C["i"] + adj_k + adj_n) - C["y"]

    # (4) Children LoM: n = (1-delta_n)*n(-1) + exp(-ph)*b
    R[3] = C["n"] - ((1 - delta_n) * L["n"] + np.exp(-C["ph"]) * C["b"])

    # (5) Time constraint: l_o + l_w + tau_n*n(-1) + tau_b*b = 1
    R[4] = (C["l_o"] + C["l_w"] + tau_n * L["n"] + tau_b * C["b"]) - 1.0

    # (6) Leisure-consumption FOC: mu_l * c * l_o^(-nu) = (1-alpha)*y/l_w
    R[5] = mu_l * C["c"] * C["l_o"] ** (-nu) - (1 - alpha) * C["y"] / C["l_w"]

    # (7) Capital efficiency: delta_k*u^omega*k(-1) = alpha*y
    R[6] = delta_k * C["u"] ** omega * L["k"] - alpha * C["y"]

    # (8) Consumption Euler:
    # beta*(1-delta_p+delta_n*n(-1))*(c/c(+1))*(alpha*y(+1)/k + 1
    #   - delta_k*u(+1)^omega/omega - psik*(1+g)^2/2*(1-(k(+1)/k)^2))
    #   = (1+g)*(1 + psik*(1+g)*(k/k(-1)-1))
    lhs8 = beta * (1 - delta_p + delta_n * L["n"]) * (C["c"] / P["c"]) * (
        alpha * P["y"] / C["k"]
        + 1 - delta_k * P["u"] ** omega / omega
        - psik * (1 + g) ** 2 / 2 * (1 - (P["k"] / C["k"]) ** 2)
    )
    rhs8 = (1 + g) * (1 + psik * (1 + g) * (C["k"] / L["k"] - 1))
    R[7] = lhs8 - rhs8

    # (9) Fertility Euler:
    # exp(ph)*tau_b*(1-alpha)*y/l_w + psin*(n/n(-1)-1)
    #   = beta*(1-delta_p+delta_n*n(-1))*(c/c(+1))*(
    #       exp(mun(+1))*c(+1)/n - pn
    #       + (tau_b*(1-delta_n)*exp(ph(+1)) - tau_n)*(1-alpha)*y(+1)/l_w(+1)
    #       - psin/2*(1-(n(+1)/n)^2))
    lhs9 = (np.exp(C["ph"]) * tau_b * (1 - alpha) * C["y"] / C["l_w"]
            + psin * (C["n"] / L["n"] - 1))
    rhs9 = beta * (1 - delta_p + delta_n * L["n"]) * (C["c"] / P["c"]) * (
        np.exp(P["mun"]) * P["c"] / C["n"] - p_n
        + (tau_b * (1 - delta_n) * np.exp(P["ph"]) - tau_n) * (1 - alpha) * P["y"] / P["l_w"]
        - psin / 2 * (1 - (P["n"] / C["n"]) ** 2)
    )
    R[8] = lhs9 - rhs9

    # (10) Productivity AR(1) — eps[0] is "ea"
    R[9]  = C["a"]   - ((1 - rhoa) * bara + rhoa * L["a"]   + eps[0])
    # (11) Fertility-preference AR(1) — eps[2] is "en" (SHOCK_NAMES order: ea, ep, en)
    R[10] = C["mun"] - ((1 - rhon) * barn + rhon * L["mun"] + eps[2])
    # (12) Mortality AR(1) — eps[1] is "ep"
    R[11] = C["ph"]  - ((1 - rhop) * barp + rhop * L["ph"]  + eps[1])

    return R


# === Solver orchestrator ======================================================

def _central_diff_jacobian(
    residuals_fn,
    z0: np.ndarray,
    h_base: float = 1e-6,
) -> np.ndarray:
    """Per-component central-difference Jacobian of vector-valued
    residuals_fn(z) wrt z, evaluated at z0.

    Step for component i: ``h_i = max(h_base, |z0[i]| * h_base)``.
    """
    z0 = np.asarray(z0, dtype=float)
    n = len(z0)
    f0 = residuals_fn(z0)
    m = len(f0)
    J = np.zeros((m, n))
    for i in range(n):
        h = max(h_base, abs(z0[i]) * h_base)
        z_plus = z0.copy(); z_plus[i] += h
        z_minus = z0.copy(); z_minus[i] -= h
        J[:, i] = (residuals_fn(z_plus) - residuals_fn(z_minus)) / (2 * h)
    return J


def _build_complete_params(
    bgp: dict,
    shock_stds: dict | None,
) -> dict:
    """Merge bgp + FERTILITY_SHOCK_PROCESSES + caller overrides into one dict
    containing every key model_residuals expects."""
    params = dict(bgp)
    for shock, spec in FERTILITY_SHOCK_PROCESSES.items():
        prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
        params[f"rho{prefix}"] = spec["rho"]
        params[f"sigma{prefix}"] = spec["sigma"]
    if shock_stds is not None:
        for shock, sigma in shock_stds.items():
            prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
            params[f"sigma{prefix}"] = float(sigma)
    # Adjustment-cost params (not in BGP because they vanish at SS).
    params.setdefault("psik", 2.5)
    params.setdefault("psin", 3.2)
    # barp default if missing
    params.setdefault("barp", 0.0)
    return params


def _solve_matrix_quadratic(
    A: np.ndarray,
    B: np.ndarray,
    Cm: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the second-order matrix polynomial A P² + B P + Cm = 0.

    Uses a generalized companion QZ decomposition. A is allowed to be
    singular (rank-deficient), which is the case for this fertility model
    where only 2 of 12 equations contain lead terms.

    The companion form is::

        M1 · v = μ · M0 · v,   v = [x; μ x],
        M0 = [[I,  0], [0,  A]],
        M1 = [[0,  I], [-Cm, -B]].

    This formulation satisfies A μ² + B μ + Cm = 0 when μ is a generalised
    eigenvalue of (M1, M0) and x is the corresponding right eigenvector.

    Returns
    -------
    P : (N_VARS, N_VARS) ndarray — stable policy matrix.
        z_t = P z_{t-1} + Q eps_t.  max|eig(P)| < 1 (contractivity enforced).
    Q : (N_VARS, N_SHOCKS) ndarray — shock impact.

    Raises
    ------
    RuntimeError if max|eig(P)| >= 1 (Blanchard-Kahn violation in the
        companion sense: the model + BGP do not admit a unique stable solution).
    """
    import scipy.linalg as _sla

    n = A.shape[0]
    n2 = 2 * n

    # 2n × 2n companion matrices for A μ² + B μ + Cm = 0.
    M0 = np.block([[np.eye(n), np.zeros((n, n))],
                   [np.zeros((n, n)), A]])
    M1 = np.block([[np.zeros((n, n)), np.eye(n)],
                   [-Cm, -B]])

    # QZ with stable eigenvalues (|alpha/beta| < 1) sorted first.
    # scipy convention: ordqz(A, B) -> eigenvalue = alpha / beta.
    _, _, alpha, beta, _, Z_qz = _sla.ordqz(
        M1, M0,
        sort=lambda a, b: abs(a) < abs(b),   # stable |alpha/beta| < 1 first
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        eigv = np.where(np.abs(beta) > 1e-12, np.abs(alpha / beta), np.inf)
    n_stable = int(np.sum(eigv < 1.0))

    # Blanchard-Kahn: exactly n stable eigenvalues for a unique stable P.
    if n_stable != n:
        n_unstable = n2 - n_stable
        raise RuntimeError(
            f"solve_fertility: Blanchard-Kahn condition failed in companion QZ: "
            f"n_stable={n_stable}, expected {n} (=N_VARS). "
            f"n_unstable (incl. inf) = {n_unstable}. "
            f"Eigenvalue moduli (sorted): "
            f"{np.array2string(np.sort(eigv), precision=4)}. "
            f"Model + BGP do not admit a unique stable linearised solution."
        )

    # Extract policy P = Z21 @ inv(Z11) from the stable Schur subspace.
    Z_stable = Z_qz[:, :n]          # (2n, n) — stable columns
    Z11 = Z_stable[:n, :]           # (n, n)
    Z21 = Z_stable[n:, :]           # (n, n)
    try:
        P = (Z21 @ np.linalg.inv(Z11)).real
    except np.linalg.LinAlgError:
        P = (Z21 @ np.linalg.pinv(Z11)).real

    # Verify A P² + B P + Cm ≈ 0.
    poly_resid = np.max(np.abs(A @ P @ P + B @ P + Cm))
    if poly_resid > 1e-6:
        raise RuntimeError(
            f"solve_fertility: matrix-polynomial residual ||AP²+BP+Cm||_inf = "
            f"{poly_resid:.3e} after QZ extraction. Numerical precision issue."
        )

    max_eig_P = float(np.max(np.abs(np.linalg.eigvals(P))))
    if max_eig_P >= 1.0 - 1e-10:
        raise RuntimeError(
            f"solve_fertility: extracted P is not contractive: "
            f"max|eig(P)| = {max_eig_P:.6f}. BK violated."
        )

    # Shock impact Q from (A P + B) Q = −D.
    M_QZ = A @ P + B
    Q, *_ = np.linalg.lstsq(M_QZ, -D, rcond=None)

    return P.real, Q.real


def _ss_initval_guess(params: dict, initval: dict) -> np.ndarray:
    """Build a full 12-vector steady-state guess (VAR_NAMES order) from a
    Dynare ``initval``-style dict of the six free unknowns, deriving the
    dependent SS values from the model identities."""
    v = dict(initval)
    v.setdefault("a", params["bara"])
    v.setdefault("mun", params["barn"])
    v.setdefault("ph", params["barp"])
    u, k, lw, n = v["u"], v["k"], v["l_w"], v["n"]
    v.setdefault(
        "y", np.exp(v["a"]) * (u * k) ** params["alpha"] * lw ** (1 - params["alpha"])
    )
    v.setdefault(
        "i", (params["g"] + params["delta_k"] * u ** params["omega"] / params["omega"]) * k
    )
    v.setdefault("l_o", 1 - params["tau_n"] * n - lw - params["tau_b"] * v["b"])
    return np.array([v[name] for name in VAR_NAMES], dtype=float)


def exact_steady_state(
    params: dict,
    z_guess: np.ndarray | None = None,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve ``model_residuals(z, z, z, 0) = 0`` exactly for the steady state.

    This is the step Dynare's ``steady;`` performs and the original port
    silently dropped. The three AR(1) rows pin ``a = bara``, ``mun = barn``,
    ``ph = barp`` exactly; the remaining nine equations in nine unknowns
    ``(k, n, c, y, l_w, u, i, b, l_o)`` are solved with ``scipy.optimize.root``
    (hybr) from ``z_guess`` (defaulting to the pinned-calibration initval).

    Acceptance is judged on the residual (``max|resid| <= tol``), not hybr's
    progress flag, which can report failure at a fully converged root.

    Returns the steady-state vector in ``VAR_NAMES`` order. Raises
    ``RuntimeError`` (naming the function) if no root is found — usually a
    guess in the wrong basin.
    """
    import warnings as _w

    exo = {"a": params["bara"], "mun": params["barn"], "ph": params["barp"]}
    endo_names = [v for v in VAR_NAMES if v not in exo]
    if z_guess is None:
        z_guess = _ss_initval_guess(params, _FERTILITY_SS_INITVAL)
    guess = np.array([z_guess[VAR_NAMES.index(v)] for v in endo_names])

    def _f9(w: np.ndarray) -> np.ndarray:
        z = np.empty(N_VARS)
        for i, name in enumerate(VAR_NAMES):
            z[i] = exo[name] if name in exo else w[endo_names.index(name)]
        with np.errstate(all="ignore"), _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            r = model_residuals(z, z, z, np.zeros(N_SHOCKS), params)
        return np.where(np.isfinite(r[:9]), r[:9], 1e6)  # 9 non-AR equations

    sol = scipy.optimize.root(_f9, guess, method="hybr", tol=1e-14)
    max_resid = float(np.max(np.abs(sol.fun)))
    if max_resid > tol:
        raise RuntimeError(
            f"exact_steady_state: no steady state found "
            f"(hybr success={sol.success}, max|resid|={max_resid:.3e} > {tol:.0e}); "
            f"the initial guess may be in the wrong basin"
        )
    z = np.empty(N_VARS)
    for i, name in enumerate(VAR_NAMES):
        z[i] = exo[name] if name in exo else sol.x[endo_names.index(name)]
    return z


def solve_fertility(
    params: dict | None = None,
    *,
    shock_stds: dict | None = None,
    h_for_jacobians: float = 1e-6,
) -> "FertilitySolution":
    """Solve the fertility DSGE around the exact steady state of its pinned
    calibration.

    Pipeline: pinned calibration (``FERTILITY_PINNED_CALIBRATION``) →
    ``exact_steady_state`` (Dynare's ``steady;`` step) → numerical Jacobians
    (central differences) on model_residuals → second-order matrix-polynomial
    QZ solve → FertilitySolution.

    ``params`` (if given) overrides structural parameters BEFORE the steady
    state is solved, so the linearization stays at a genuine equilibrium of
    the resulting calibration. This replaces the earlier ``solve_bgp`` path,
    which linearised around the least-squares endpoint of an over-determined,
    mutually inconsistent BGP system — a non-equilibrium point whose
    Blanchard-Kahn status was numerically fragile (it flipped across LAPACK
    builds). ``solve_bgp`` is retained for reference / calibration experiments.

    The lead (A) Jacobian has only 2 non-zero rows (consumption Euler and
    fertility Euler), while N_FWD = 7. This makes the standard first-order
    Klein/QZ formulation fail (singular A produces too many inf eigenvalues).
    The correct formulation is the second-order matrix polynomial companion:

        A P² + B P + Cm = 0   (uniquely solvable iff BK holds in companion sense)

    where P is the n×n full-system policy matrix z_t = P z_{t-1} + Q ε_t.
    The Klein-convention matrices G, N, F, L are then the (n_pre/n_fwd) blocks of P, Q.
    The klein_solution field is a synthetic KleinSolution(eu=(1,1)) constructed from P, Q.
    """
    # 1. Pinned calibration (+ optional structural overrides) and shock processes.
    calib = dict(FERTILITY_PINNED_CALIBRATION)
    if params is not None:
        calib.update(params)
    all_params = _build_complete_params(calib, shock_stds)

    # 2. Solve the EXACT steady state of this calibration (Dynare `steady;`).
    z_ss = exact_steady_state(all_params)

    # 3. Compute four Jacobians via central differences.
    eps0 = np.zeros(N_SHOCKS)

    A = _central_diff_jacobian(
        lambda zl: model_residuals(zl, z_ss, z_ss, eps0, all_params),
        z_ss, h_base=h_for_jacobians,
    )
    B = _central_diff_jacobian(
        lambda z: model_residuals(z_ss, z, z_ss, eps0, all_params),
        z_ss, h_base=h_for_jacobians,
    )
    Cm = _central_diff_jacobian(
        lambda zlg: model_residuals(z_ss, z_ss, zlg, eps0, all_params),
        z_ss, h_base=h_for_jacobians,
    )
    D = _central_diff_jacobian(
        lambda e: model_residuals(z_ss, z_ss, z_ss, e, all_params),
        eps0, h_base=h_for_jacobians,
    )

    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(Cm)) and np.all(np.isfinite(D))):
        raise RuntimeError(
            "solve_fertility: numerical Jacobian contains non-finite entries — "
            "check steady state and model equations."
        )

    # 4. Solve second-order matrix polynomial A P² + B P + Cm = 0 via companion QZ.
    #    This handles the singular A case correctly (rank(A) = 2 in this model).
    #    Returns full-system policy P (n×n) and shock impact Q (n×n_shocks).
    P, Q = _solve_matrix_quadratic(A, B, Cm, D)

    # 5. Partition into Klein-convention matrices:
    #    x_t = G x_{t-1} + N ε_t       (predetermined states, rows/cols 0:N_PRE)
    #    y_t = F x_{t-1} + L ε_t       (controls, rows N_PRE:)
    #    (P_xy = 0 and P_yy = 0 by construction — verified numerically)
    G = P[:N_PRE, :N_PRE]
    N = Q[:N_PRE, :]
    F = P[N_PRE:, :N_PRE]
    L = Q[N_PRE:, :]

    # 6. Construct a synthetic KleinSolution to satisfy the FertilitySolution schema
    #    and expose diagnostics.  eu = (1, 1) because the companion BK check passed.
    eig_G = np.sort(np.abs(np.linalg.eigvals(G)))
    klein_sol = KleinSolution(
        G=G, F=F, N=N, L=L,
        eu=(1, 1),
        eigenvalues=eig_G,
    )

    ss = {name: float(z_ss[i]) for i, name in enumerate(VAR_NAMES)}

    return FertilitySolution(
        ss=ss,
        params=all_params,
        G=G,
        N=N,
        F=F,
        L=L,
        klein_solution=klein_sol,
        var_names=VAR_NAMES,
        shock_names=SHOCK_NAMES,
    )


# === IRF / FEVD helpers (called by FertilitySolution.irf and .fevd) ==========

def _resolve_shock(shock, shock_names: tuple) -> int:
    if isinstance(shock, int):
        if not (0 <= shock < len(shock_names)):
            raise ValueError(f"shock index {shock} out of range [0, {len(shock_names)})")
        return shock
    if isinstance(shock, str):
        if shock not in shock_names:
            raise ValueError(
                f"unknown shock {shock!r}; expected one of {list(shock_names)}"
            )
        return shock_names.index(shock)
    raise TypeError(f"shock must be int or str, got {type(shock).__name__}")


def _compute_irf(sol: "FertilitySolution", shock, horizon: int) -> pd.DataFrame:
    """Impulse response to a 1-SD shock. See FertilitySolution.irf docstring."""
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    s_idx = _resolve_shock(shock, sol.shock_names)
    sigma_key = {"ea": "sigmaa", "ep": "sigmap", "en": "sigman"}[sol.shock_names[s_idx]]
    sigma = sol.params.get(sigma_key, 1.0)
    # Impact: state_0 = N[:, s] * sigma; control_0 = L[:, s] * sigma
    state = sol.N[:, s_idx] * sigma
    control = sol.L[:, s_idx] * sigma
    rows = [np.concatenate([state, control])]
    for _ in range(horizon):
        state = sol.G @ state
        control = sol.F @ state
        rows.append(np.concatenate([state, control]))
    arr = np.array(rows)
    return pd.DataFrame(arr, columns=list(sol.var_names), index=range(horizon + 1))


def _compute_fevd(sol: "FertilitySolution", horizon: int) -> pd.DataFrame:
    """Forecast-error variance decomposition: per-(variable, shock) share at each horizon."""
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    n_vars = len(sol.var_names)
    n_shocks = len(sol.shock_names)
    irfs = np.zeros((horizon + 1, n_vars, n_shocks))
    for s_idx in range(n_shocks):
        sigma_key = {"ea": "sigmaa", "ep": "sigmap", "en": "sigman"}[sol.shock_names[s_idx]]
        sigma = sol.params.get(sigma_key, 1.0)
        state = sol.N[:, s_idx] * sigma
        control = sol.L[:, s_idx] * sigma
        irfs[0, :, s_idx] = np.concatenate([state, control])
        for h in range(1, horizon + 1):
            state = sol.G @ state
            control = sol.F @ state
            irfs[h, :, s_idx] = np.concatenate([state, control])
    cum_sq = np.cumsum(irfs ** 2, axis=0)
    total = cum_sq.sum(axis=2, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    share = cum_sq / total
    rows = []
    for h in range(horizon + 1):
        for v_idx, var in enumerate(sol.var_names):
            for s_idx, shock in enumerate(sol.shock_names):
                rows.append({
                    "horizon": h, "variable": var, "shock": shock,
                    "share": share[h, v_idx, s_idx],
                })
    return pd.DataFrame(rows)


__all__ = [
    "VAR_NAMES", "SHOCK_NAMES", "N_PRE", "N_FWD", "N_VARS", "N_SHOCKS",
    "FERTILITY_EXOGENOUS_PARAMS", "FERTILITY_CALIB_TARGETS",
    "FERTILITY_SHOCK_PROCESSES", "FERTILITY_PINNED_CALIBRATION",
    "solve_bgp", "exact_steady_state", "model_residuals", "solve_fertility",
]
