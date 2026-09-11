"""Shared machinery for the fertility Blanchard-Kahn diagnosis.

Analysis-only: imports the package read-only, never modifies it.
Everything here mirrors puremacro/dsge/fertility_adj_costs.py where
fidelity to the shipped pipeline matters (companion construction,
classification rule), and deliberately deviates where the point is to
get an *independent*, better-conditioned answer (complex-step
Jacobians, determinant-interpolation root finding, exact steady-state
solving).

Model recap (from the package):
    A z_{t+1} + B z_t + C z_{t-1} (+ D eps_t) = 0   after linearisation,
    solved via the 24x24 companion pencil  M1 v = mu M0 v,
    M0 = [[I,0],[0,A]],  M1 = [[0,I],[-C,-B]].
Structural counts: n = 12, rank(A) = 2 (only the two Euler rows have
leads), lag columns only on (a, mun, ph, k, n) so rank(C) = 5.
Hence det(A mu^2 + B mu + C) has degree n + rank(A) = 14, with
n - rank(C) = 7 roots at zero, and the companion pencil carries
2n - 14 = 10 infinite eigenvalues.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla
import scipy.optimize

from puremacro.dsge.fertility_adj_costs import (  # noqa: F401
    VAR_NAMES,
    SHOCK_NAMES,
    N_VARS,
    N_SHOCKS,
    N_PRE,
    FERTILITY_SHOCK_PROCESSES,
    solve_bgp,
    model_residuals,
)

# ---------------------------------------------------------------------------
# Parameter assembly (mirrors solve_fertility's merge order, but honours
# rho overrides, which solve_fertility clobbers from FERTILITY_SHOCK_PROCESSES)
# ---------------------------------------------------------------------------

_BGP_CACHE: dict | None = None


def port_bgp() -> dict:
    """The package's least-squares BGP dict (cached; RuntimeWarning silenced)."""
    global _BGP_CACHE
    if _BGP_CACHE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _BGP_CACHE = solve_bgp()
    return dict(_BGP_CACHE)


def build_params(overrides: dict | None = None) -> dict:
    """Port-default parameter dict (BGP + shock processes), then overrides.

    Unlike solve_fertility, overrides are applied LAST, so rhoa/rhon/rhop
    can be varied.
    """
    params = port_bgp()
    for shock, spec in FERTILITY_SHOCK_PROCESSES.items():
        prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
        params[f"rho{prefix}"] = spec["rho"]
        params[f"sigma{prefix}"] = spec["sigma"]
    params.setdefault("psik", 2.5)
    params.setdefault("psin", 3.2)
    params.setdefault("barp", 0.0)
    if overrides:
        params.update(overrides)
    return params


def z_from_params(params: dict) -> np.ndarray:
    """Linearisation point in VAR_NAMES order (what solve_fertility uses)."""
    return np.array([params[name] for name in VAR_NAMES], dtype=float)


# ---------------------------------------------------------------------------
# Complex-safe residuals (verbatim copy of the package equations, complex dtype)
# ---------------------------------------------------------------------------

def model_residuals_cs(z_lead, z, z_lag, eps, params):
    """Complex-safe re-statement of the 12 package residuals.

    The package function allocates a float64 output array, so complex
    inputs raise on assignment (verified in 02_jacobian_noise.py). This
    copy is used ONLY to enable complex-step differentiation; equality
    with the package on real inputs is asserted by
    ``assert_residuals_match``.
    """
    L = {name: z_lag[i] for i, name in enumerate(VAR_NAMES)}
    Cc = {name: z[i] for i, name in enumerate(VAR_NAMES)}
    P = {name: z_lead[i] for i, name in enumerate(VAR_NAMES)}

    alpha = params["alpha"]; delta_k = params["delta_k"]
    delta_n = params["delta_n"]; delta_p = params["delta_p"]
    omega = params["omega"]; g = params["g"]; beta = params["beta"]
    nu = params["nu"]; mu_l = params["mu_l"]
    tau_n = params["tau_n"]; tau_b = params["tau_b"]; p_n = params["p_n"]
    psik = params.get("psik", 2.5); psin = params.get("psin", 3.2)
    rhoa = params["rhoa"]; rhon = params["rhon"]; rhop = params["rhop"]
    bara = params["bara"]; barn = params["barn"]; barp = params["barp"]

    R = np.zeros(N_VARS, dtype=complex)
    R[0] = Cc["y"] - np.exp(Cc["a"]) * (Cc["u"] * L["k"]) ** alpha * Cc["l_w"] ** (1 - alpha)
    R[1] = Cc["i"] - ((1 + g) * Cc["k"] - (1 - delta_k * Cc["u"] ** omega / omega) * L["k"])
    adj_k = psik / 2 * (1 + g) ** 2 * (Cc["k"] / L["k"] - 1) ** 2 * L["k"]
    adj_n = psin / 2 * (Cc["n"] / L["n"] - 1) ** 2 * L["n"]
    R[2] = (Cc["c"] + p_n * L["n"] + Cc["i"] + adj_k + adj_n) - Cc["y"]
    R[3] = Cc["n"] - ((1 - delta_n) * L["n"] + np.exp(-Cc["ph"]) * Cc["b"])
    R[4] = (Cc["l_o"] + Cc["l_w"] + tau_n * L["n"] + tau_b * Cc["b"]) - 1.0
    R[5] = mu_l * Cc["c"] * Cc["l_o"] ** (-nu) - (1 - alpha) * Cc["y"] / Cc["l_w"]
    R[6] = delta_k * Cc["u"] ** omega * L["k"] - alpha * Cc["y"]
    lhs8 = beta * (1 - delta_p + delta_n * L["n"]) * (Cc["c"] / P["c"]) * (
        alpha * P["y"] / Cc["k"]
        + 1 - delta_k * P["u"] ** omega / omega
        - psik * (1 + g) ** 2 / 2 * (1 - (P["k"] / Cc["k"]) ** 2)
    )
    rhs8 = (1 + g) * (1 + psik * (1 + g) * (Cc["k"] / L["k"] - 1))
    R[7] = lhs8 - rhs8
    lhs9 = (np.exp(Cc["ph"]) * tau_b * (1 - alpha) * Cc["y"] / Cc["l_w"]
            + psin * (Cc["n"] / L["n"] - 1))
    rhs9 = beta * (1 - delta_p + delta_n * L["n"]) * (Cc["c"] / P["c"]) * (
        np.exp(P["mun"]) * P["c"] / Cc["n"] - p_n
        + (tau_b * (1 - delta_n) * np.exp(P["ph"]) - tau_n) * (1 - alpha) * P["y"] / P["l_w"]
        - psin / 2 * (1 - (P["n"] / Cc["n"]) ** 2)
    )
    R[8] = lhs9 - rhs9
    R[9] = Cc["a"] - ((1 - rhoa) * bara + rhoa * L["a"] + eps[0])
    R[10] = Cc["mun"] - ((1 - rhon) * barn + rhon * L["mun"] + eps[2])
    R[11] = Cc["ph"] - ((1 - rhop) * barp + rhop * L["ph"] + eps[1])
    return R


def assert_residuals_match(params: dict, z0: np.ndarray, n_draws: int = 20,
                           seed: int = 0, atol: float = 1e-12) -> float:
    """Verify model_residuals_cs == package model_residuals on real inputs."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_draws):
        zl = z0 * (1 + 0.05 * rng.standard_normal(N_VARS))
        zc = z0 * (1 + 0.05 * rng.standard_normal(N_VARS))
        zg = z0 * (1 + 0.05 * rng.standard_normal(N_VARS))
        ep = 0.01 * rng.standard_normal(N_SHOCKS)
        r_pkg = model_residuals(zl, zc, zg, ep, params)
        r_cs = model_residuals_cs(zl, zc, zg, ep, params).real
        worst = max(worst, float(np.max(np.abs(r_pkg - r_cs))))
    if worst > atol:
        raise AssertionError(
            f"complex-safe residual copy deviates from package: max abs diff {worst:.3e}"
        )
    return worst


# ---------------------------------------------------------------------------
# Jacobians: central (package mirror), Richardson, complex-step
# ---------------------------------------------------------------------------

def jac_central(fn, z0: np.ndarray, h_base: float) -> np.ndarray:
    """Mirror of the package's _central_diff_jacobian (same step rule)."""
    z0 = np.asarray(z0, dtype=float)
    f0 = fn(z0)
    J = np.zeros((len(f0), len(z0)))
    for i in range(len(z0)):
        h = max(h_base, abs(z0[i]) * h_base)
        zp = z0.copy(); zp[i] += h
        zm = z0.copy(); zm[i] -= h
        J[:, i] = (fn(zp) - fn(zm)) / (2 * h)
    return J


def jac_richardson(fn, z0: np.ndarray, h_base: float) -> np.ndarray:
    """Richardson-extrapolated central differences: (4 J(h/2) - J(h)) / 3."""
    return (4.0 * jac_central(fn, z0, h_base / 2) - jac_central(fn, z0, h_base)) / 3.0


def jac_complex_step(fn_cs, z0: np.ndarray, h: float = 1e-30) -> np.ndarray:
    """Complex-step Jacobian: J[:, i] = Im f(z0 + i*h*e_i) / h (no cancellation)."""
    z0 = np.asarray(z0, dtype=float)
    f0 = fn_cs(z0.astype(complex))
    J = np.zeros((len(f0), len(z0)))
    for i in range(len(z0)):
        zp = z0.astype(complex)
        zp[i] += 1j * h
        J[:, i] = fn_cs(zp).imag / h
    return J


def build_ABCD(params: dict, z_ss: np.ndarray, method: str = "central",
               h: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four Jacobians of the residuals at z_ss, by the chosen method."""
    eps0 = np.zeros(N_SHOCKS)
    if method in ("central", "richardson"):
        jac = jac_central if method == "central" else jac_richardson
        A = jac(lambda zl: model_residuals(zl, z_ss, z_ss, eps0, params), z_ss, h)
        B = jac(lambda zc: model_residuals(z_ss, zc, z_ss, eps0, params), z_ss, h)
        C = jac(lambda zg: model_residuals(z_ss, z_ss, zg, eps0, params), z_ss, h)
        D = jac(lambda ee: model_residuals(z_ss, z_ss, z_ss, ee, params), eps0, h)
    elif method == "cs":
        eps0c = np.zeros(N_SHOCKS, dtype=complex)
        A = jac_complex_step(lambda zl: model_residuals_cs(zl, z_ss, z_ss, eps0c, params), z_ss)
        B = jac_complex_step(lambda zc: model_residuals_cs(z_ss, zc, z_ss, eps0c, params), z_ss)
        C = jac_complex_step(lambda zg: model_residuals_cs(z_ss, z_ss, zg, eps0c, params), z_ss)
        D = jac_complex_step(lambda ee: model_residuals_cs(z_ss, z_ss, z_ss, ee, params), np.zeros(N_SHOCKS))
    else:
        raise ValueError(f"unknown method {method!r}")
    return A, B, C, D


# ---------------------------------------------------------------------------
# Companion QZ (exact mirror of the package's classification rule)
# ---------------------------------------------------------------------------

BETA_THRESHOLD = 1e-12  # the package's |beta| cut for "finite"


def companion_qz(A: np.ndarray, B: np.ndarray, Cm: np.ndarray) -> dict:
    """Package-faithful companion QZ; returns full diagnostics."""
    n = A.shape[0]
    M0 = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), A]])
    M1 = np.block([[np.zeros((n, n)), np.eye(n)], [-Cm, -B]])
    _, _, alpha, beta, _, Z = sla.ordqz(M1, M0, sort=lambda a, b: abs(a) < abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        eigv = np.where(np.abs(beta) > BETA_THRESHOLD, np.abs(alpha / beta), np.inf)
    n_stable = int(np.sum(eigv < 1.0))
    return {
        "alpha": alpha, "beta": beta, "moduli": eigv,
        "n_stable": n_stable, "Z": Z, "M0": M0, "M1": M1,
    }


def pencil_eig_conditions(M1: np.ndarray, M0: np.ndarray) -> list[dict]:
    """Per-eigenvalue sensitivity of the companion pencil.

    For a simple finite eigenvalue lambda with right/left vectors x, y:
        |d lambda| <= eps * (||M1|| + |lambda| ||M0||) ||x|| ||y|| / |y^H M0 x|.
    Returns one record per finite eigenvalue (nan lambda entries skipped).
    """
    lam, vl, vr = sla.eig(M1, M0, left=True, right=True)
    nrm1, nrm0 = np.linalg.norm(M1, 2), np.linalg.norm(M0, 2)
    out = []
    for j in range(len(lam)):
        lj = lam[j]
        if not np.isfinite(lj):
            continue
        x = vr[:, j]; y = vl[:, j]
        denom = abs(y.conj() @ (M0 @ x))
        sens = np.inf if denom == 0 else (nrm1 + abs(lj) * nrm0) / denom  # ||x||=||y||=1
        out.append({"eig": lj, "modulus": abs(lj), "sensitivity": float(sens)})
    return out


# ---------------------------------------------------------------------------
# Determinant-interpolation finite spectrum (QZ-independent cross-check)
# ---------------------------------------------------------------------------

def det_poly_coeffs(A: np.ndarray, B: np.ndarray, Cm: np.ndarray,
                    radius: float = 3.0, m: int = 64) -> np.ndarray:
    """Coefficients of q(mu) = det(A mu^2 + B mu + Cm) via FFT on a circle.

    q is a polynomial of degree <= 2n; with rank(A) = 2 its effective
    degree is n + rank(A) = 14. Coefficients are exact up to rounding for
    m > 2n sample points. Returned low-to-high, normalised by max |coeff|.
    """
    n = A.shape[0]
    if m <= 2 * n:
        raise ValueError("need m > 2n sample points")
    mus = radius * np.exp(2j * np.pi * np.arange(m) / m)
    with np.errstate(all="ignore"):
        vals = np.array([np.linalg.det(A * mu ** 2 + B * mu + Cm) for mu in mus])
    coeffs = np.fft.fft(vals) / m
    coeffs = coeffs / radius ** np.arange(m)  # undo the radius scaling
    coeffs = coeffs[: 2 * n + 1]
    scale = np.max(np.abs(coeffs))
    return coeffs / scale


def det_roots(A: np.ndarray, B: np.ndarray, Cm: np.ndarray,
              radius: float = 3.0, m: int = 64,
              trim_tol: float = 1e-9) -> tuple[np.ndarray, int]:
    """Finite spectrum of the quadratic pencil via determinant interpolation.

    Returns (roots ascending by modulus, effective polynomial degree).
    """
    c = det_poly_coeffs(A, B, Cm, radius=radius, m=m)
    nz = np.where(np.abs(c) > trim_tol)[0]
    if len(nz) == 0:
        raise RuntimeError("determinant identically ~0 on the contour: singular pencil")
    deg, low = int(nz[-1]), int(nz[0])
    # coefficients below trim_tol at the low end are exact zero roots
    # (n - rank(C) of them structurally); deflate instead of letting noise
    # scatter them onto a ~|noise|^(1/low) circle.
    poly = c[low: deg + 1][::-1]  # np.roots wants high-to-low
    roots = np.concatenate([np.zeros(low, dtype=complex), np.roots(poly)])
    return roots[np.argsort(np.abs(roots))], deg


def split_roots(roots: np.ndarray, params: dict,
                zero_tol: float = 1e-7, rho_tol: float = 1e-6) -> dict:
    """Partition finite roots into zeros / shock rhos / endogenous quartet."""
    rhos = np.array([params["rhoa"], params["rhon"], params["rhop"]])
    used = np.zeros(len(roots), dtype=bool)
    zeros_idx = [i for i, r in enumerate(roots) if abs(r) < zero_tol]
    for i in zeros_idx:
        used[i] = True
    rho_idx = []
    for rho in rhos:
        cand = [i for i in range(len(roots)) if not used[i]]
        if not cand:
            break
        j = min(cand, key=lambda i: abs(roots[i] - rho))
        if abs(roots[j] - rho) < max(rho_tol, 1e-6 * abs(rho)):
            rho_idx.append(j)
            used[j] = True
    endo = roots[~used]
    return {
        "zeros": roots[zeros_idx],
        "rho_roots": roots[rho_idx],
        "endogenous": endo[np.argsort(np.abs(endo))],
    }


# ---------------------------------------------------------------------------
# Quadratic eigenvectors (economic loadings of a root)
# ---------------------------------------------------------------------------

def quadratic_eigvec(A: np.ndarray, B: np.ndarray, Cm: np.ndarray,
                     mu: complex) -> tuple[np.ndarray, float]:
    """Right vector x with (A mu^2 + B mu + Cm) x ~ 0, via SVD.

    Returns (|x| normalised to max 1, smallest singular value)."""
    Q = A * mu ** 2 + B * mu + Cm
    _, s, Vh = np.linalg.svd(Q)
    x = Vh[-1].conj()
    ax = np.abs(x)
    return ax / ax.max(), float(s[-1])


# ---------------------------------------------------------------------------
# Exact steady state (what Dynare's `steady;` does and the port skips)
# ---------------------------------------------------------------------------

def exact_steady_state(params: dict, z_guess: np.ndarray | None = None) -> np.ndarray:
    """Solve model_residuals(z, z, z, 0) = 0 exactly for z, params fixed.

    The AR(1) rows pin a = bara, mun = barn, ph = barp exactly; the
    remaining 9 equations are solved for (k, n, c, y, l_w, u, i, b, l_o)
    with scipy hybr. Raises with diagnostics if no root is found.
    """
    exo = {"a": params["bara"], "mun": params["barn"], "ph": params["barp"]}
    endo_names = [v for v in VAR_NAMES if v not in exo]
    if z_guess is None:
        z_guess = z_from_params(params)
    guess = np.array([z_guess[VAR_NAMES.index(v)] for v in endo_names])

    def f9(w):
        z = np.empty(N_VARS)
        for i, name in enumerate(VAR_NAMES):
            z[i] = exo[name] if name in exo else w[endo_names.index(name)]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = model_residuals(z, z, z, np.zeros(N_SHOCKS), params)
        return np.where(np.isfinite(r[:9]), r[:9], 1e6)  # AR rows exactly 0

    sol = scipy.optimize.root(f9, guess, method="hybr", tol=1e-14)
    # judge by the residual, not hybr's progress flag (which can be False
    # at a fully converged root)
    if np.max(np.abs(sol.fun)) > 1e-10:
        raise RuntimeError(
            f"exact_steady_state: no root (success={sol.success}, "
            f"max|resid|={np.max(np.abs(sol.fun)):.3e}); guess may be in the wrong basin"
        )
    z = np.empty(N_VARS)
    for i, name in enumerate(VAR_NAMES):
        z[i] = exo[name] if name in exo else sol.x[endo_names.index(name)]
    return z


# ---------------------------------------------------------------------------
# One-call verdict
# ---------------------------------------------------------------------------

def bk_summary(params: dict, z_ss: np.ndarray, method: str = "cs",
               h: float = 1e-6) -> dict:
    """Jacobians -> det roots -> split -> counts + margins + QZ n_stable."""
    A, B, C, D = build_ABCD(params, z_ss, method=method, h=h)
    roots, deg = det_roots(A, B, C)
    parts = split_roots(roots, params)
    qz = companion_qz(A, B, C)
    mods = np.abs(roots)
    margin = float(np.min(np.abs(mods - 1.0)))
    n_stable_det = int(np.sum(mods < 1.0))  # infinite eigenvalues are unstable
    return {
        "roots": roots, "deg": deg, "parts": parts,
        "endo_moduli": np.abs(parts["endogenous"]),
        "n_endo_stable": int(np.sum(np.abs(parts["endogenous"]) < 1.0)),
        "n_stable_det": n_stable_det,
        "n_stable_qz": qz["n_stable"],
        "margin": margin,
        "qz": qz,
        "A": A, "B": B, "C": C, "D": D,
    }


# Original Dynare calibrations, for pipeline validation (03) ------------------

# fertility_adj_costs.mod (calibrated variant; Dynare log: finite eigs
# 0.1, 0.5844, 0.8889, 0.9, 0.96, 1.254, 1.856; BK verified)
MOD1_PARAMS = {
    "alpha": 0.40, "delta_k": 0.14, "omega": 1.75, "g": 0.016,
    "delta_n": 0.065, "tau_n": 0.18, "tau_b": 0.48, "mu_l": 0.95,
    "nu": 2.5, "beta": 0.95, "p_n": 0.35, "delta_p": 0.075,
    "bara": 0.0, "rhoa": 0.96, "barn": 0.30, "rhon": 0.9,
    "barp": float(np.log(1.02)), "rhop": 0.1, "psik": 1.5, "psin": 2.2,
}
MOD1_DYNARE_FINITE = np.array([0.1, 0.5844, 0.8889, 0.9, 0.96, 1.254, 1.856])
MOD1_SS_GUESS = {"c": 1.3, "i": 0.5, "u": 1.0, "k": 5.10, "n": 0.01,
                 "b": 0.03, "l_w": 0.3}

# fertility_adj_costs_bayesian_estimation.mod + output/data/parameters.mat
# (Dynare log: finite eigs 0.5, 0.6838, 0.7807, 0.9, 0.9197, 1.192, 1.549;
# BK verified)
MOD2_PARAMS = {
    "alpha": 0.40, "nu": 2.5, "barp": float(np.log(1 / 1.03)),
    "g": 0.0175, "delta_p": 0.075, "delta_n": 0.065,
    "bara": 0.0, "omega": 2.0,
    "rhoa": 0.94 ** 4, "rhon": 0.5, "rhop": 0.9,
    "psik": 2.5, "psin": 3.2,
    # loaded from parameters.mat (x1..x7)
    "barn": float(np.log(1.587983)), "mu_l": 0.634818, "beta": 0.936122,
    "tau_n": 0.213943, "tau_b": 0.469184, "p_n": 0.094621, "delta_k": 0.144,
}
MOD2_DYNARE_FINITE = np.array([0.5, 0.6838, 0.7807, 0.9, 0.9197, 1.192, 1.549])
MOD2_SS_GUESS = {"c": 0.360443, "b": 0.087379, "l_w": 0.329987, "u": 0.999997,
                 "k": 1.835291, "n": 1.384615}


def dynare_ss_guess(params: dict, extra: dict) -> np.ndarray:
    """Build a z-guess for exact_steady_state from .mod initval-style values."""
    z = np.empty(N_VARS)
    vals = dict(extra)
    vals.setdefault("a", params["bara"])
    vals.setdefault("mun", params["barn"])
    vals.setdefault("ph", params["barp"])
    u, k, lw = vals.get("u", 1.0), vals["k"], vals["l_w"]
    vals.setdefault("y", np.exp(vals["a"]) * (u * k) ** params["alpha"] * lw ** (1 - params["alpha"]))
    vals.setdefault("i", (params["g"] + params["delta_k"] * u ** params["omega"] / params["omega"]) * k)
    if "b" not in vals:
        vals["b"] = params["delta_n"] * vals["n"] * np.exp(vals["ph"])
    vals.setdefault("n", vals["b"] / (params["delta_n"] * np.exp(vals["ph"])) if "b" in vals else 1.0)
    vals.setdefault("l_o", 1 - params["tau_n"] * vals["n"] - lw - params["tau_b"] * vals["b"])
    for i, name in enumerate(VAR_NAMES):
        z[i] = vals[name]
    return z
