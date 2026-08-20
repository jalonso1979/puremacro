"""Calibration of the signal-extraction DMP model for regime_uncertainty paper §4.

Solves the two-regime equilibrium (tight prior pi_H vs loose prior pi_L) by
perturbation around steady state, computes IRFs to a unit-sigma LTUI shock,
and emits a 3-panel figure overlaying model and empirical IRFs.

The Fed has a hidden type tau in {dove, hawk} with reaction function
r_t = r* + phi_tau * sigma_t. The dove CUTS rates on uncertainty (phi_D < 0)
while the hawk HIKES (phi_H > 0). Private agents form a posterior pi_t over
the Fed type; the expected continuation discount factor

    E_t[beta_{t+1} | pi_t, sigma_t]
        = pi_t / (1 + r* + phi_D sigma_t) + (1 - pi_t) / (1 + r* + phi_H sigma_t)

is increasing in sigma under a dovish prior (high pi) and decreasing under a
hawkish prior (low pi). This produces the regime sign-flip in d(theta)/d(sigma):
a critical posterior pi* = phi_H / (phi_H - phi_D) separates contraction
(pi < pi*) from expansion (pi > pi*).

Discount-factor functional form. The continuation discount factor is the
gross-rate inverse beta(r) = 1/(1 + r), matching assumption (iii) of the
sign-flip proposition (Appendix B): beta(.) = (1 + .)^{-1} is twice
continuously differentiable, strictly decreasing, and strictly convex on the
realized support of the perceived rate. Units matter here: phi_D and phi_H are
reported in the paper in *percentage points* (NB43's -347 bps cut = phi_D, the
hawk's +150 bps = phi_H), but the model's r* = 0.01 is a *decimal*, so the
coefficients enter in decimal units (phi_D = -0.0347, phi_H = 0.015). The
dove's worst-case perceived rate at a unit-sigma shock is then
r* + phi_D = 0.01 - 0.0347 = -0.0247 — a mild ZLB cut, comfortably above the
pole of 1/(1 + r) at r = -1 — so the discount map is well-defined and positive
on the entire realized support. (An earlier draft entered phi_D ~ -3.47 as a
decimal, implying a -347-percentage-point move that pushed the dove's rate past
the r = -1 pole; that units error, not the functional form, was the problem.)
The sign-flip is derived from phi_D, phi_H, not imposed.

Run as a standalone script:

    python calibrate_dmp_signal_extraction.py

Outputs:
- ../output_figures/95_model_vs_data_irf.pdf
- ../output_tables/95_calibration_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# Hardcoded baseline parameters (per-line sources noted below).
BASELINE_PARAMS: Dict[str, float] = {
    "beta":      0.99,   # quarterly discount factor
    "r_star":    0.01,   # steady-state real rate
    "eta":       0.50,   # matching elasticity (Petrongolo-Pissarides midpoint)
    "gamma":     0.50,   # worker bargaining share (Hosios: gamma = eta)
    "s":         0.03,   # exogenous separation rate
    "b_over_y":  0.40,   # UI replacement rate (Shimer 2005)
    "c_over_y":  0.20,   # vacancy cost / output (Hagedorn-Manovskii 2008)
    "phi_D":    -0.0320,  # -320 bps per sigma-LTUI (NB43 h=2 counterfactual); decimal
    "phi_H":     0.015,   # +150 bps per sigma-LTUI; decimal (hawk HIKE, ~half the dove's magnitude)
    "pi_H":      0.10,   # posterior P(dove | tight stance)
    "pi_L":      0.90,   # posterior P(dove | loose stance)
    "rho_sigma": 0.70,   # LTUI AR(1) persistence
    "mu_normalize_theta_ss": 1.0,  # calibrate mu so that theta_ss = 1
}


def _beta_of_r(r: float, params: Dict[str, float]) -> float:
    """Discount factor as a function of the perceived per-period rate r.

    This is the gross-rate inverse used throughout the paper's theory section,

        beta(r) = 1 / (1 + r),

    which is exactly assumption (iii) of the sign-flip proposition (Appendix B):
    beta(.) = (1 + .)^{-1} is twice continuously differentiable, strictly
    decreasing, and strictly convex on the support of r* + phi_tau*sigma. With
    phi in *decimal* units (phi_D = -0.0347, phi_H = 0.015 per standardized
    sigma-LTUI), the worst-case perceived rate at a unit-sigma shock is the
    dove's r* + phi_D = 0.01 - 0.0347 = -0.0247 (a mild ZLB cut) — comfortably
    above the pole at r = -1, so 1/(1 + r) is well-defined and positive on the
    entire realized support. The `params` argument is retained for signature
    stability (no parameter is needed by the closed form). See module docstring.
    """
    return 1.0 / (1.0 + r)


def expected_discount_factor(pi: float, sigma: float, params: Dict[str, float]) -> float:
    """E_t[beta_{t+1} | pi_t, sigma_t] under the two-type prior.

    E_t[beta_{t+1} | pi, sigma]
        = pi / (1 + r* + phi_D*sigma) + (1 - pi) / (1 + r* + phi_H*sigma),

    i.e. beta(.) = 1/(1 + .) is the paper's gross-rate inverse (`_beta_of_r`).
    With phi_D < 0 (dove cuts) and phi_H > 0 (hawk hikes), the dove branch rises
    and the hawk branch falls with sigma; the net sign of d E[beta]/d sigma is
    positive iff pi > pi* = phi_H / (phi_H - phi_D).
    """
    r_dove = params["r_star"] + params["phi_D"] * sigma
    r_hawk = params["r_star"] + params["phi_H"] * sigma
    beta_dove = _beta_of_r(r_dove, params)
    beta_hawk = _beta_of_r(r_hawk, params)
    return pi * beta_dove + (1.0 - pi) * beta_hawk


def steady_state(pi: float, params: Dict[str, float]) -> Dict[str, float]:
    """Solve the steady-state DMP equations for a given prior pi and sigma = 0.

    Steady-state equations (Pissarides 2000 ch. 1):
      - Matching: q(theta) = mu theta^(-eta), p(theta) = theta q(theta) = mu theta^(1-eta)
      - Bellman: J = (y - w) + (1-s) beta J  =>  J = (y - w) / (1 - (1-s) beta)
      - Nash wage: w = gamma y + (1 - gamma) b + gamma c theta
      - Free entry: c = q(theta) . J
      - Steady-state urate: u = s / (s + p(theta))
    With y normalized to 1 and theta normalized to 1 (the calibration choice
    mu = q(1)), the system is recursive and closed-form: the Nash wage and the
    Bellman J are pinned directly at theta = 1, and mu is read off the free-entry
    condition c = q(1) J = mu J. (Solving the three equations jointly with a
    floating mu and then forcing theta = 1 ex post would mislabel the off-theta
    (J, w) as the theta = 1 steady state and break free entry / Nash at theta = 1;
    imposing theta = 1 up front keeps every steady-state equation exact.)
    beta = 1/(1 + r*) at sigma = 0.
    """
    y = 1.0
    b = params["b_over_y"] * y
    c = params["c_over_y"] * y
    eta = params["eta"]
    gamma = params["gamma"]
    s = params["s"]
    r_star = params["r_star"]
    beta = 1.0 / (1.0 + r_star)

    # Normalization: theta_ss = 1. The remaining steady-state objects follow
    # recursively (no joint root-find needed):
    theta = 1.0
    # Nash wage at theta = 1:
    w = gamma * y + (1.0 - gamma) * b + gamma * c * theta
    # Bellman job value: J = (y - w) / (1 - (1-s) beta):
    J = (y - w) / (1.0 - (1.0 - s) * beta)
    # Free entry c = q(theta) J with q(1) = mu  =>  mu = c / J:
    mu = c / J
    q_theta = mu * theta ** (-eta)          # = mu at theta = 1
    p_theta = mu * theta ** (1.0 - eta)      # = mu at theta = 1
    u_rate = s / (s + p_theta)
    v = u_rate * theta                       # since theta = V/U

    return {
        "theta":   theta,
        "q_theta": q_theta,
        "p_theta": p_theta,
        "J":       J,
        "w":       w,
        "u_rate":  u_rate,
        "v":       v,
        "mu_implied": mu,
        "y":       y,
        "b":       b,
        "c":       c,
        "beta_ss": beta,
    }


def _d_expected_discount_factor_d_sigma(pi: float, params: Dict[str, float]) -> float:
    """Marginal effect d E[beta]/d sigma evaluated at the steady state (sigma = 0).

    With beta(r) = 1/(1 + r) and r = r* + phi_tau * sigma, the chain rule gives
    beta'(r) = -1/(1 + r)^2, and at sigma = 0 both branches sit at r = r*:
        d E[beta]/d sigma |_{sigma=0}
            = pi phi_D beta'(r*) + (1-pi) phi_H beta'(r*)
            = -[ pi phi_D + (1-pi) phi_H ] / (1 + r*)^2.
    This is positive iff pi > pi* = phi_H / (phi_H - phi_D) — the sign-flip
    threshold of Proposition (regime sign-flip). The prefactor 1/(1+r*)^2 is
    scale-positive, so pi* (a ratio of the phi's) is unchanged by it.
    """
    beta_prime_rstar = -1.0 / (1.0 + params["r_star"]) ** 2
    return beta_prime_rstar * (pi * params["phi_D"] + (1.0 - pi) * params["phi_H"])


def solve_irf(pi: float, sigma0: float, horizons: int, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """First-order (log-)linearized IRF to a unit-sigma LTUI shock under prior pi.

    The prior pi is treated as fixed within a regime (the regime indicator is
    persistent on the time scale of the IRF). The LTUI shock follows an AR(1),
    sigma_h = sigma0 * rho_sigma^h, and the model is solved by *first-order
    perturbation around the regime-specific steady state* (as named in the
    method: a genuine linearization, not a nonlinear perfect-foresight solve).
    The discount-factor map enters only through its steady-state slope dEb =
    d E[beta]/d sigma|_0, evaluated at sigma = 0 where both type-branches sit at
    r = r* and beta = 1/(1 + r*) < 1 — a well-behaved, in-domain point — so the
    linearization is robust. (This is a modeling choice matching the spec's
    "first-order perturbation around steady state" language; it does not depend
    on the discount-factor pole, which the decimal-unit phi's do not approach.)

    Nonlinear system:
        w_t   = gamma y + (1-gamma) b + gamma c theta_t          (Nash wage)
        J_t   = (y - w_t) + (1-s) E[beta | pi, sigma_t] J_{t+1}   (Bellman)
        c     = q(theta_t) J_t = mu theta_t^{-eta} J_t            (free entry)
        u_{t+1} = u_t + s(1 - u_t) - p(theta_t) u_t               (urate LoM)
        V_t   = theta_t u_t                                       (tightness defn)

    Level deviations (d-hat) around the steady state, with E[beta] entering only
    through its steady-state slope dEb = d E[beta]/d sigma|_0:
        free entry:  dJ_h = eta (J_ss / theta_ss) dtheta_h
        Bellman:     dJ_h = -gamma c dtheta_h + (1-s)(beta_ss dJ_{h+1} + dEb sigma_h J_ss)
        urate LoM:   du_{h+1} = (1 - s - p_ss) du_h - p'(theta_ss) u_ss dtheta_h
    Substituting free entry into Bellman gives a backward scalar recursion for
    dJ_h under a terminal condition dJ -> 0 (shock dies out); dtheta_h, du_h and
    dV_h then follow. Returns log-deviations log_urate, log_v, log_theta, each of
    length horizons + 1 (h = 0, ..., horizons). Linear in sigma0, so a zero shock
    returns exactly zeros.
    """
    ss = steady_state(pi=pi, params=params)
    eta = params["eta"]
    s = params["s"]
    gamma = params["gamma"]
    rho_sigma = params["rho_sigma"]
    beta_ss = ss["beta_ss"]
    J_ss = ss["J"]
    theta_ss = ss["theta"]
    u_ss = ss["u_rate"]
    p_ss = ss["p_theta"]
    mu = ss["mu_implied"]
    c = ss["c"]

    dEb = _d_expected_discount_factor_d_sigma(pi=pi, params=params)

    def sigma_at(h: int) -> float:
        return sigma0 * (rho_sigma ** h)

    # Backward recursion for the level deviation of the job value dJ_h.
    # Combine free entry  dtheta_h = (theta_ss / (eta J_ss)) dJ_h  into Bellman:
    #   dJ_h [1 + gamma c theta_ss / (eta J_ss)] = (1-s)(beta_ss dJ_{h+1} + dEb sigma_h J_ss)
    k = 1.0 + gamma * c * theta_ss / (eta * J_ss)
    H_fwd = 200  # long horizon so the shock has fully decayed; dJ terminal = 0
    dJ = np.zeros(H_fwd + 2)
    for h in range(H_fwd, -1, -1):
        dJ[h] = (1.0 - s) * (beta_ss * dJ[h + 1] + dEb * sigma_at(h) * J_ss) / k

    # Recover dtheta over the requested horizons.
    dtheta = np.zeros(horizons + 1)
    for h in range(horizons + 1):
        dtheta[h] = (theta_ss / (eta * J_ss)) * dJ[h]

    # Forward recursion for the urate level deviation du_h (du_0 = 0: the urate is
    # predetermined and only reacts from h = 1 onward via lagged tightness).
    # p(theta) = mu theta^{1-eta}  =>  p'(theta_ss) = mu (1-eta) theta_ss^{-eta}.
    p_prime = mu * (1.0 - eta) * theta_ss ** (-eta)
    du = np.zeros(horizons + 1)
    for h in range(horizons):
        du[h + 1] = (1.0 - s - p_ss) * du[h] - p_prime * u_ss * dtheta[h]

    # Convert to log-deviations. log x_h ~ dx_h / x_ss.
    log_theta = dtheta / theta_ss
    log_urate = du / u_ss
    # V = theta * u  =>  log V = log theta + log u.
    log_v = log_theta + log_urate

    return {
        "log_urate": log_urate,
        "log_v":     log_v,
        "log_theta": log_theta,
    }


def sign_flip_test(params: Dict[str, float]) -> bool:
    """True iff d_theta_h2/d_sigma < 0 under pi_H and > 0 under pi_L at horizon h = 2."""
    irf_tight = solve_irf(pi=params["pi_H"], sigma0=1.0, horizons=2, params=params)
    irf_loose = solve_irf(pi=params["pi_L"], sigma0=1.0, horizons=2, params=params)
    return bool((irf_tight["log_theta"][2] < 0) and (irf_loose["log_theta"][2] > 0))


def sensitivity_table(params: Dict[str, float]) -> pd.DataFrame:
    """Comparative-statics: vary each of (phi_D, pi_H, pi_L, rho_sigma) by +/-50%,
    report whether the sign-flip proposition holds.

    Returns a DataFrame with columns:
        parameter (str), perturbation (str), value (float), sign_flip_holds (bool)
    """
    rows = [{
        "parameter":       "baseline",
        "perturbation":    "none",
        "value":           float("nan"),
        "sign_flip_holds": sign_flip_test(params),
    }]
    for key in ("phi_D", "pi_H", "pi_L", "rho_sigma"):
        for sign, label in [(-0.5, "-50%"), (+0.5, "+50%")]:
            p = params.copy()
            p[key] = params[key] * (1.0 + sign)
            # Keep pi in (0, 1)
            if key in ("pi_H", "pi_L"):
                p[key] = max(0.001, min(0.999, p[key]))
            rows.append({
                "parameter":       key,
                "perturbation":    label,
                "value":           p[key],
                "sign_flip_holds": sign_flip_test(p),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 3: empirical IRF loader, figure, meta JSON, main entrypoint.
# ---------------------------------------------------------------------------

# Resolve output dirs relative to this script (notebooks/scripts/...).
_SCRIPT_DIR = Path(__file__).resolve().parent
_TABLES_DIR = _SCRIPT_DIR.parent / "output_tables"
_FIGURES_DIR = _SCRIPT_DIR.parent / "output_figures"

# Sample-mean unemployment rate (percentage points) used to convert the
# empirical urate IRF from percentage-point units into log-deviation units
# (d log U ~= d_urate_pp / mean_urate_pp). 5.8 pp is the mean civilian
# unemployment rate over the 2007Q4-2026Q2 estimation window of the
# state-dependent LP (matches 42_meta.json sample bounds).
MEAN_URATE_PP: float = 5.8


def load_empirical_irfs(tables_dir: Path | str | None = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the empirical state-dependent LP IRFs for the model overlay.

    Reads the headline FFR (`rate_state`) state-dependent local-projection
    coefficients from:
      - ``36_lp_state_dep_rate.csv``           (unemployment rate, *pp* units)
      - ``36_lp_state_dep_log_vacancies.csv``  (log vacancies, *log* units)

    In both files the high/tight regime is the ``beta_H``/``se_H`` pair and the
    loose regime is ``beta_L``/``se_L`` (so tight -> H, loose -> L). Horizons
    h = 0..8 (length-9 arrays).

    The unemployment-rate coefficients are reported in percentage points; we
    additionally return them converted to log-deviations (d log U ~=
    beta_pp / MEAN_URATE_PP) so that they share units with the model IRFs and
    with vacancies. Tightness theta = V/U has no direct empirical IRF file, so
    we derive it as d log theta = d log V - d log U using the converted urate,
    propagating the standard error by adding the V and (converted) U standard
    errors in quadrature (treating them as independent -- a deliberate, stated
    approximation, since the LP coefficients are estimated on the same sample
    and are not literally independent).

    Returns a dict keyed by outcome (``log_urate``, ``log_v``, ``log_theta``),
    each mapping to a sub-dict with length-9 arrays:
        ``tight_beta``, ``tight_se``, ``loose_beta``, ``loose_se``
    All in log-deviation units. The raw percentage-point urate coefficients are
    additionally exposed under ``log_urate``'s ``tight_beta_pp`` / ``loose_beta_pp``
    / ``tight_se_pp`` / ``loose_se_pp`` keys for transparency in the meta JSON.
    """
    tables = Path(tables_dir) if tables_dir is not None else _TABLES_DIR
    rate = pd.read_csv(tables / "36_lp_state_dep_rate.csv").sort_values("h").reset_index(drop=True)
    vac = pd.read_csv(tables / "36_lp_state_dep_log_vacancies.csv").sort_values("h").reset_index(drop=True)

    # Unemployment rate: pp -> log-deviation via mean urate.
    u_tight_pp = rate["beta_H"].to_numpy(float)
    u_loose_pp = rate["beta_L"].to_numpy(float)
    u_tight_se_pp = rate["se_H"].to_numpy(float)
    u_loose_se_pp = rate["se_L"].to_numpy(float)
    u_tight = u_tight_pp / MEAN_URATE_PP
    u_loose = u_loose_pp / MEAN_URATE_PP
    u_tight_se = u_tight_se_pp / MEAN_URATE_PP
    u_loose_se = u_loose_se_pp / MEAN_URATE_PP

    # Vacancies: already in log units.
    v_tight = vac["beta_H"].to_numpy(float)
    v_loose = vac["beta_L"].to_numpy(float)
    v_tight_se = vac["se_H"].to_numpy(float)
    v_loose_se = vac["se_L"].to_numpy(float)

    # Tightness theta = V/U  =>  d log theta = d log V - d log U.
    th_tight = v_tight - u_tight
    th_loose = v_loose - u_loose
    th_tight_se = np.sqrt(v_tight_se ** 2 + u_tight_se ** 2)
    th_loose_se = np.sqrt(v_loose_se ** 2 + u_loose_se ** 2)

    return {
        "log_urate": {
            "tight_beta": u_tight,
            "tight_se":   u_tight_se,
            "loose_beta": u_loose,
            "loose_se":   u_loose_se,
            # Raw percentage-point coefficients (pre-conversion) for the record.
            "tight_beta_pp": u_tight_pp,
            "loose_beta_pp": u_loose_pp,
            "tight_se_pp":   u_tight_se_pp,
            "loose_se_pp":   u_loose_se_pp,
        },
        "log_v": {
            "tight_beta": v_tight,
            "tight_se":   v_tight_se,
            "loose_beta": v_loose,
            "loose_se":   v_loose_se,
        },
        "log_theta": {
            "tight_beta": th_tight,
            "tight_se":   th_tight_se,
            "loose_beta": th_loose,
            "loose_se":   th_loose_se,
            "derived":    True,  # not a direct LP; d log V - d log U
        },
    }


def _norm_to_h2(arr: np.ndarray, h2_abs: float | None = None) -> Tuple[np.ndarray, float]:
    """Normalize a series by the absolute value of its own h=2 entry.

    Returns (normalized_series, scale) where scale = |arr[2]| (or the supplied
    `h2_abs`). Sign is preserved; the h=2 magnitude becomes +/-1. Used for the
    *shape* overlay in the figure (model and empirical curves are unit-free).
    Magnitudes are reported separately in the meta JSON, not via this transform.
    """
    a = np.asarray(arr, dtype=float)
    scale = float(abs(a[2])) if h2_abs is None else float(h2_abs)
    if scale == 0.0:
        return a.copy(), 0.0
    return a / scale, scale


# Outcomes to plot, in panel order, with display titles. Tightness is derived
# (d log V - d log U), flagged in the caption / meta.
_PANELS = [
    ("log_urate", r"(a) $\log(\mathrm{urate})$"),
    ("log_v",     r"(b) $\log V$"),
    ("log_theta", r"(c) $\log \theta = \log V - \log U$"),
]


def make_figure(
    model_tight: Dict[str, np.ndarray],
    model_loose: Dict[str, np.ndarray],
    empirical: Dict[str, Dict[str, np.ndarray]],
    out_pdf: Path | str | None = None,
) -> Path:
    """Render the 3-panel model-vs-data IRF figure (shape overlay, normalized).

    Each panel shows one outcome over horizons h = 0..8. Empirical IRFs (with
    90% HAC-style +/-1.645 SE bands) and model IRFs are each normalized to their
    own |value at h=2| so the panels compare *shape and sign* on a common,
    unit-free vertical scale -- the empirical urate IRF is in percentage points
    while the model is in log-deviations, so a level overlay would be
    misleading. Tight forward stance is black; loose is grey. Model curves are
    heavy solid lines; empirical are thinner dashed lines with shaded bands.
    Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_pdf) if out_pdf is not None else (_FIGURES_DIR / "95_model_vs_data_irf.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    H = len(model_tight["log_urate"]) - 1
    h = np.arange(H + 1)
    Z = 1.645  # 90% normal band, matching the empirical LP figures in the paper

    TIGHT_C = "black"
    LOOSE_C = "0.55"  # medium grey

    fig, axes = plt.subplots(1, len(_PANELS), figsize=(11.0, 3.4), sharex=True)
    for ax, (outcome, title) in zip(axes, _PANELS):
        emp = empirical[outcome]
        # Normalize each curve to its own |h=2|.
        m_t, _ = _norm_to_h2(model_tight[outcome])
        m_l, _ = _norm_to_h2(model_loose[outcome])
        e_t, s_t = _norm_to_h2(emp["tight_beta"])
        e_l, s_l = _norm_to_h2(emp["loose_beta"])
        # Bands: empirical SE scaled by the same per-series factor.
        et_se = np.asarray(emp["tight_se"], float) / s_t if s_t else np.zeros(H + 1)
        el_se = np.asarray(emp["loose_se"], float) / s_l if s_l else np.zeros(H + 1)

        # Empirical bands.
        ax.fill_between(h, e_t - Z * et_se, e_t + Z * et_se, color=TIGHT_C, alpha=0.12, linewidth=0)
        ax.fill_between(h, e_l - Z * el_se, e_l + Z * el_se, color=LOOSE_C, alpha=0.18, linewidth=0)
        # Empirical means (dashed).
        ax.plot(h, e_t, color=TIGHT_C, lw=1.3, ls="--", label="Empirical, tight")
        ax.plot(h, e_l, color=LOOSE_C, lw=1.3, ls="--", label="Empirical, loose")
        # Model means (heavy solid).
        ax.plot(h, m_t, color=TIGHT_C, lw=2.4, ls="-", label="Model, tight")
        ax.plot(h, m_l, color=LOOSE_C, lw=2.4, ls="-", label="Model, loose")

        ax.axhline(0.0, color="0.3", lw=0.6, ls=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("horizon $h$ (quarters)")
        ax.set_xticks(range(0, H + 1, 2))
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("normalized response\n(each curve / $|$value at $h{=}2|$)")
    # Single shared legend below the panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _to_list(x) -> list:
    """JSON-serializable list from a numpy array / scalar."""
    return np.asarray(x, dtype=float).tolist()


def build_meta(
    params: Dict[str, float],
    model_tight: Dict[str, np.ndarray],
    model_loose: Dict[str, np.ndarray],
    empirical: Dict[str, Dict[str, np.ndarray]],
    sensitivity: pd.DataFrame,
    horizons: int,
) -> dict:
    """Assemble the meta dict written to 95_calibration_meta.json.

    Records the calibration parameters, the model IRFs under both priors, the
    empirical IRFs (log-deviation units, plus the raw urate percentage points),
    and the comparative-statics sensitivity records used to fill Appendix B.
    Also stores, per outcome, the within-one-SE verdict at h=2 under a single
    shared scalar that preserves the model's internal tight/loose ratio (the
    honest magnitude test reported in the paper).
    """
    def emp_block(outcome: str) -> dict:
        e = empirical[outcome]
        blk = {k: _to_list(e[k]) for k in ("tight_beta", "tight_se", "loose_beta", "loose_se")}
        if outcome == "log_urate":
            blk.update({k: _to_list(e[k]) for k in
                        ("tight_beta_pp", "loose_beta_pp", "tight_se_pp", "loose_se_pp")})
        if e.get("derived"):
            blk["derived"] = True
            blk["derivation"] = "d log theta = d log V - d log U (SE in quadrature)"
        return blk

    # Within-1-SE at h=2 under one shared scalar per outcome (model keeps its
    # tight/loose ratio; scalar best-fits the empirical h=2 pair in least squares).
    within = {}
    for outcome in ("log_urate", "log_v", "log_theta"):
        mt = float(model_tight[outcome][2])
        ml = float(model_loose[outcome][2])
        e = empirical[outcome]
        bt, bl = float(e["tight_beta"][2]), float(e["loose_beta"][2])
        st, sl = float(e["tight_se"][2]), float(e["loose_se"][2])
        denom = mt * mt + ml * ml
        k = (mt * bt + ml * bl) / denom if denom > 0 else float("nan")
        pt, pl = k * mt, k * ml
        within[outcome] = {
            "shared_scalar_k": k,
            "model_h2_tight_scaled": pt,
            "model_h2_loose_scaled": pl,
            "emp_h2_tight": bt, "emp_h2_tight_se": st,
            "emp_h2_loose": bl, "emp_h2_loose_se": sl,
            "within_1se_tight": bool(bt - st <= pt <= bt + st),
            "within_1se_loose": bool(bl - sl <= pl <= bl + sl),
        }
    # Vacancies are the one outcome that shares units with the model directly;
    # also record the raw (k=1) verdict there.
    mt, ml = float(model_tight["log_v"][2]), float(model_loose["log_v"][2])
    ev = empirical["log_v"]
    within["log_v"]["raw_no_rescale"] = {
        "model_h2_tight": mt, "model_h2_loose": ml,
        "within_1se_tight": bool(float(ev["tight_beta"][2]) - float(ev["tight_se"][2]) <= mt
                                 <= float(ev["tight_beta"][2]) + float(ev["tight_se"][2])),
        "within_1se_loose": bool(float(ev["loose_beta"][2]) - float(ev["loose_se"][2]) <= ml
                                 <= float(ev["loose_beta"][2]) + float(ev["loose_se"][2])),
    }

    # Enriched comparative-statics block for Appendix B: per perturbation row,
    # the implied threshold pi* = phi_H/(phi_H - phi_D), whether it is interior,
    # the sign of d theta_{h=2}/d sigma under each prior, and the sign-flip
    # verdict. Mirrors sensitivity_table's perturbation grid exactly.
    def _row_stats(p: Dict[str, float]) -> dict:
        pistar = p["phi_H"] / (p["phi_H"] - p["phi_D"])
        th_tight = float(solve_irf(pi=p["pi_H"], sigma0=1.0, horizons=2, params=p)["log_theta"][2])
        th_loose = float(solve_irf(pi=p["pi_L"], sigma0=1.0, horizons=2, params=p)["log_theta"][2])
        holds = (th_tight < 0) and (th_loose > 0)
        return {
            "pistar": float(pistar),
            "pistar_interior": bool(0.0 < pistar < 1.0),
            "sgn_dtheta_h2_tight": "-" if th_tight < 0 else "+",
            "sgn_dtheta_h2_loose": "+" if th_loose > 0 else "-",
            "sign_flip_holds": bool(holds),
        }

    comparative_statics = [{
        "parameter": "baseline", "perturbation": "none", "value": float("nan"),
        **_row_stats(params),
    }]
    for key in ("phi_D", "pi_H", "pi_L", "rho_sigma"):
        for frac, label in [(-0.5, "-50%"), (+0.5, "+50%")]:
            p = params.copy()
            p[key] = params[key] * (1.0 + frac)
            if key in ("pi_H", "pi_L"):
                p[key] = max(0.001, min(0.999, p[key]))
            comparative_statics.append({
                "parameter": key, "perturbation": label, "value": float(p[key]),
                **_row_stats(p),
            })

    return {
        "script": "calibrate_dmp_signal_extraction.py",
        "horizons": int(horizons),
        "mean_urate_pp_used_for_log_conversion": MEAN_URATE_PP,
        "empirical_source": {
            "urate": "36_lp_state_dep_rate.csv (FFR rate_state LP; beta in pp)",
            "vacancies": "36_lp_state_dep_log_vacancies.csv (FFR rate_state LP; beta in log)",
            "tightness": "derived: d log V - d log U",
            "note": ("Bands are from the headline state-dependent LP operationalized via "
                     "the federal funds rate (full horizon). The paper's preferred "
                     "shadow-state spec (42_meta.json) reports only the h=2 summary."),
        },
        "params": dict(params),
        "model_irf": {
            "tight_prior_pi_H": {k: _to_list(v) for k, v in model_tight.items()},
            "loose_prior_pi_L": {k: _to_list(v) for k, v in model_loose.items()},
        },
        "empirical_irf": {outcome: emp_block(outcome)
                          for outcome in ("log_urate", "log_v", "log_theta")},
        "within_one_se_at_h2": within,
        "sensitivity_table": sensitivity.to_dict(orient="records"),
        "comparative_statics": comparative_statics,
    }


def main(horizons: int = 8) -> None:
    """Solve model IRFs (both priors), load empirical IRFs, run the sensitivity
    table, and write the figure (95_model_vs_data_irf.pdf) and meta JSON
    (95_calibration_meta.json)."""
    params = BASELINE_PARAMS
    model_tight = solve_irf(pi=params["pi_H"], sigma0=1.0, horizons=horizons, params=params)
    model_loose = solve_irf(pi=params["pi_L"], sigma0=1.0, horizons=horizons, params=params)
    empirical = load_empirical_irfs()
    sensitivity = sensitivity_table(params)

    fig_path = make_figure(model_tight, model_loose, empirical)

    meta = build_meta(params, model_tight, model_loose, empirical, sensitivity, horizons)
    meta_path = _TABLES_DIR / "95_calibration_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[calibrate_dmp] sign_flip_test(baseline) = {sign_flip_test(params)}")
    print(f"[calibrate_dmp] wrote figure -> {fig_path}")
    print(f"[calibrate_dmp] wrote meta   -> {meta_path}")
    print("[calibrate_dmp] within-1SE @ h=2 (shared-scalar, model keeps tight/loose ratio):")
    for outcome, blk in meta["within_one_se_at_h2"].items():
        print(f"    {outcome:10s}: tight={blk['within_1se_tight']} loose={blk['within_1se_loose']} (k={blk['shared_scalar_k']:.3g})")


if __name__ == "__main__":
    main()
