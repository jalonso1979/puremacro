"""IRF-matching estimation of the nested-DMP model against empirical LPs.

The model IRFs (puremacro...dynamics.simulate_irf) are matched to empirical
state-dependent local projections in NORMALIZED-shape space (sign + timing),
with the magnitude gap reported as a diagnostic -- the lean free-entry model
reproduces the regime sign-flip but under-responds in magnitude (the H-side
amplification gap the model's dropped frictions would close). An exact
minimum-distance fit: simulate_irf is deterministic, so no panel + LP inner loop.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from puremacro.models.nested_dmp import dynamics as _dyn
from puremacro.models.nested_dmp.params import NestedDMPParameters

MEAN_URATE_PP = 5.8  # sample-mean civilian urate; converts pp LP coeffs to log-dev


@dataclass(frozen=True)
class EmpiricalIRFTargets:
    """Empirical state-dependent IRF targets (log-dev), loose/tight x urate/v/theta."""

    horizons: np.ndarray
    loose: dict        # {"urate","v","theta"} -> length-9 log-dev arrays
    tight: dict
    loose_se: dict     # standard errors (log-dev)
    tight_se: dict
    loose_lo: dict     # 90% band lower (urate has its own; V/theta use +/-1.645 se)
    loose_hi: dict
    tight_lo: dict
    tight_hi: dict


def load_empirical_irf_targets(tables_dir) -> EmpiricalIRFTargets:
    """Load the state-dependent LP IRFs (urate, V; derive theta) as fit targets.

    urate: state-dependent-LP urate targets (pp -> log via /MEAN_URATE_PP; has
           lo/hi bands).
    V:     state-dependent-LP vacancy targets (already log; band = beta +/- 1.645 se).
    theta = log V - log U. H columns = tight (hawk); L = loose (dove). h = 0..8.
    """
    tables = Path(tables_dir)
    rate = (
        pd.read_csv(tables / "urate_lp_targets.csv")
        .sort_values("h")
        .reset_index(drop=True)
    )
    vac = (
        pd.read_csv(tables / "vacancies_lp_targets.csv")
        .sort_values("h")
        .reset_index(drop=True)
    )
    h = rate["h"].to_numpy(int)
    Z = 1.645  # 90% CI half-width multiplier

    # Convert unemployment-rate pp coefficients to log deviations.
    u_loose = rate["beta_L"].to_numpy(float) / MEAN_URATE_PP
    u_tight = rate["beta_H"].to_numpy(float) / MEAN_URATE_PP
    u_loose_se = rate["se_L"].to_numpy(float) / MEAN_URATE_PP
    u_tight_se = rate["se_H"].to_numpy(float) / MEAN_URATE_PP

    # Vacancy log-responses are already in log units.
    v_loose = vac["beta_L"].to_numpy(float)
    v_tight = vac["beta_H"].to_numpy(float)
    v_loose_se = vac["se_L"].to_numpy(float)
    v_tight_se = vac["se_H"].to_numpy(float)

    # Market tightness theta = log V - log U.
    th_loose = v_loose - u_loose
    th_tight = v_tight - u_tight
    # SE of theta via error propagation (V and U are separate regressions -> uncorrelated).
    th_loose_se = np.sqrt(v_loose_se ** 2 + u_loose_se ** 2)
    th_tight_se = np.sqrt(v_tight_se ** 2 + u_tight_se ** 2)

    loose = {"urate": u_loose, "v": v_loose, "theta": th_loose}
    tight = {"urate": u_tight, "v": v_tight, "theta": th_tight}
    loose_se = {"urate": u_loose_se, "v": v_loose_se, "theta": th_loose_se}
    tight_se = {"urate": u_tight_se, "v": v_tight_se, "theta": th_tight_se}

    return EmpiricalIRFTargets(
        horizons=h,
        loose=loose,
        tight=tight,
        loose_se=loose_se,
        tight_se=tight_se,
        loose_lo={k: loose[k] - Z * loose_se[k] for k in loose},
        loose_hi={k: loose[k] + Z * loose_se[k] for k in loose},
        tight_lo={k: tight[k] - Z * tight_se[k] for k in tight},
        tight_hi={k: tight[k] + Z * tight_se[k] for k in tight},
    )


def calibrate_mu_to_f(params, *, f_target: float = 0.5, sigma_ref: float = 1.0,
                      mu_lo: float = 1e-4, mu_hi: float = 1e3) -> float:
    """Matching efficiency mu giving job-finding rate f(theta_ss)=f_target.

    f(theta_ss)=mu*theta_ss^(1-alpha) and theta_ss rises with mu (more entry), so
    f is increasing in mu -- a unique root. f_target<1 keeps the forward dynamics
    stable (a contraction) and the matching probability valid; 0.5 is a standard
    quarterly job-finding magnitude. Replaces a theta_ss=1 anchor (which leaves f
    tiny ~0.03 at the low-b surplus) with a realistic f.

    The inner theta-solve (_theta_given) requires the free-entry residual to
    bracket zero, which fails when mu is so small that vacancy posting is
    unprofitable at any theta in [theta_lo, theta_hi]. The bracket-scan below
    marches upward by powers of 10 until it finds a valid (a, b) with
    f(a)*f(b) < 0, then hands off to brentq.
    """
    from dataclasses import replace as _replace

    def _safe_f(mu: float):
        """Returns f(theta_ss) - f_target, or None if inner solve fails."""
        try:
            pf = _replace(params, mu=mu)
            grid, erg = _dyn._fixed_grid(pf, sigma_ref=sigma_ref)
            ss = _dyn._grid_steady_state(pf, grid=grid, erg=erg, sigma=0.0, pi=pf.prior_pi0)
            return pf.mu * ss["theta"] ** (1.0 - pf.alpha) - f_target
        except (ValueError, RuntimeError):
            # Inner theta brentq has no bracket at this mu (too small to post vacancies).
            return None

    # Scan powers of 10 from mu_lo to mu_hi to find a valid bracket [a, b].
    import math
    candidates = [mu_lo * (10.0 ** k)
                  for k in range(int(math.log10(mu_hi / mu_lo)) + 2)
                  if mu_lo * (10.0 ** k) <= mu_hi * 1.001]
    candidates.append(mu_hi)

    fa_val = fb_val = None
    a = b = None
    for mu in candidates:
        v = _safe_f(mu)
        if v is None:
            continue  # inner solve not feasible here; skip
        if fa_val is None:
            a, fa_val = mu, v
            continue
        b, fb_val = mu, v
        if fa_val * fb_val < 0:
            break  # valid bracket found
        # Both same sign: advance lower end
        a, fa_val = b, fb_val

    if a is None or b is None or fa_val is None or fb_val is None or fa_val * fb_val >= 0:
        raise RuntimeError(
            f"calibrate_mu_to_f: could not bracket f_target={f_target} in "
            f"[{mu_lo}, {mu_hi}]. Check params or widen the search range."
        )

    return float(brentq(lambda mu: _safe_f(mu), a, b, xtol=1e-10))


def model_regime_irfs(params, *, sigma0: float = 1.0, horizon: int = 8):
    """(loose, tight) model IRF dicts -- dove and hawk simulate_irf, keyed
    urate/v/theta to match the empirical targets."""
    dove = _dyn.simulate_irf(params, sigma0=sigma0, fed_type="dove", horizon=horizon)
    hawk = _dyn.simulate_irf(params, sigma0=sigma0, fed_type="hawk", horizon=horizon)
    loose = {"urate": np.asarray(dove.log_urate), "v": np.asarray(dove.log_v),
             "theta": np.asarray(dove.log_theta)}
    tight = {"urate": np.asarray(hawk.log_urate), "v": np.asarray(hawk.log_v),
             "theta": np.asarray(hawk.log_theta)}
    return loose, tight


@dataclass(frozen=True)
class FitReport:
    """Model-vs-empirical IRF fit: sign-flip match, shape distance, magnitude gap."""
    sign_match: dict          # outcome -> bool (model & data agree in sign at the peak horizon)
    shape_distance: float     # inverse-variance-weighted normalized-shape SSE
    magnitude_ratio: dict     # {"loose"/"tight"} -> {outcome -> emp_peak/model_peak}
    within_band: dict         # {"loose"/"tight"} -> {outcome -> frac of h within emp 90% band}


def _peak_norm(x):
    """x divided by |x at its max-|.| horizon|; sign preserved, peak -> +/-1."""
    x = np.asarray(x, float)
    k = int(np.argmax(np.abs(x)))
    s = abs(x[k])
    return (x / s, s) if s > 0 else (x.copy(), 0.0)


def fit_report(params, targets, *, sigma0: float = 1.0) -> FitReport:
    """Compare model (loose=dove, tight=hawk) IRFs to the empirical targets.

    Shape distance: each curve normalized to its own peak, then inverse-LP-
    variance-weighted SSE across urate/v/theta x {loose,tight} x h. Magnitude
    ratio = |empirical peak| / |model peak| (the under-response gap -- the
    headline). within_band = fraction of horizons where the model (rescaled to
    the empirical peak) lies in the empirical 90% band.
    """
    horizon = int(targets.horizons[-1])
    m_loose, m_tight = model_regime_irfs(params, sigma0=sigma0, horizon=horizon)
    outcomes = ("urate", "v", "theta")
    sign_match: dict[str, bool] = {}
    shape_sse: float = 0.0
    mag: dict[str, dict[str, float]] = {"loose": {}, "tight": {}}
    band: dict[str, dict[str, float]] = {"loose": {}, "tight": {}}
    for reg, m, e, se, lo, hi in (
        ("loose", m_loose, targets.loose, targets.loose_se, targets.loose_lo, targets.loose_hi),
        ("tight", m_tight, targets.tight, targets.tight_se, targets.tight_lo, targets.tight_hi),
    ):
        for o in outcomes:
            mn, m_pk = _peak_norm(m[o])
            en, e_pk = _peak_norm(e[o])
            w = 1.0 / np.maximum(se[o] ** 2, 1e-8)
            shape_sse += float(np.sum(w * (mn - en) ** 2))
            mag[reg][o] = float(e_pk / m_pk) if m_pk > 0 else float("inf")
            scaled = m[o] * (e_pk / m_pk) if m_pk > 0 else m[o]
            band[reg][o] = float(np.mean((scaled >= lo[o]) & (scaled <= hi[o])))
    for o in outcomes:
        kp = int(np.argmax(np.abs(targets.tight[o])))
        sign_match[o] = bool(
            np.sign(m_loose[o][kp]) == np.sign(targets.loose[o][kp])
            and np.sign(m_tight[o][kp]) == np.sign(targets.tight[o][kp])
        )
    return FitReport(sign_match=sign_match, shape_distance=shape_sse,
                     magnitude_ratio=mag, within_band=band)


@dataclass(frozen=True)
class EstimationResult:
    params: object            # fitted NestedDMPParameters
    shape_distance: float
    report: FitReport
    estimated: tuple          # names estimated


def estimate_shape_match(params, targets, *, sigma0: float = 1.0,
                         estimated=("signal_sd", "rho_sigma"),
                         maxiter: int = 200) -> EstimationResult:
    """Fit the response params to the normalized IRF shapes (sign + timing).

    Minimizes fit_report(...).shape_distance over ``estimated`` (the belief-update
    precision signal_sd and shock persistence rho_sigma, which shape the IRF
    buildup/decay). The magnitude gap is NOT a target -- it is reported. Bounded
    Nelder-Mead keeps params admissible (signal_sd>0, 0<=rho_sigma<1); inadmissible
    draws get a heavy penalty so the search stays in-region.
    """
    names = tuple(estimated)
    x0 = np.array([getattr(params, n) for n in names], float)

    def unpack(x):
        # Clamp each parameter to its admissibility region before evaluating.
        kw = {}
        for n, v in zip(names, x):
            if n == "rho_sigma":
                v = min(max(v, 1e-4), 0.999)
            elif n == "signal_sd":
                v = max(v, 1e-3)
            kw[n] = float(v)
        return replace(params, **kw)

    def obj(x):
        try:
            return fit_report(unpack(x), targets, sigma0=sigma0).shape_distance
        except Exception:
            return 1e12  # inadmissible / non-converging draw -> heavy penalty

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": maxiter})
    p_hat = unpack(res.x)
    rep = fit_report(p_hat, targets, sigma0=sigma0)
    return EstimationResult(params=p_hat, shape_distance=rep.shape_distance,
                            report=rep, estimated=names)


def ablate_beliefs(params, *, sigma0: float = 1.0, horizon: int = 8, pi_common: float = 0.5):
    """Belief ablation: run both regimes at ONE common belief pi_common.

    The sign-flip comes from dove (pi>pi*) and hawk (pi<pi*) sitting on opposite
    sides of pi*=phi_H/(phi_H-phi_D); forcing a single belief in both removes it,
    so urate moves the same direction in 'loose' and 'tight'. Confirms the belief
    channel generates the regime asymmetry. Returns (loose, tight) dicts.
    """
    dove = _dyn.simulate_irf(params, sigma0=sigma0, fed_type="dove", horizon=horizon,
                             pi_loose=pi_common, pi_hawk=pi_common)
    hawk = _dyn.simulate_irf(params, sigma0=sigma0, fed_type="hawk", horizon=horizon,
                             pi_loose=pi_common, pi_hawk=pi_common)
    loose = {"urate": np.asarray(dove.log_urate), "v": np.asarray(dove.log_v),
             "theta": np.asarray(dove.log_theta)}
    tight = {"urate": np.asarray(hawk.log_urate), "v": np.asarray(hawk.log_v),
             "theta": np.asarray(hawk.log_theta)}
    return loose, tight


def baseline_calibration(params, *, u_target: float = 0.058, f_target: float = 0.7,
                         sigma_ref: float = 1.0, s_lo: float = 5e-3, s_hi: float = 0.4):
    """Recalibrate (s_bar, mu) to US labor-market flow targets.

    mu pins the job-finding rate f(theta_ss)=f_target (calibrate_mu_to_f); s_bar
    pins steady-state unemployment u=u_target via a brentq root-find, with mu
    recomputed at each s_bar (changing s_bar shifts theta_ss and thus f, so the
    two are coupled). b (the surplus/amplification lever) is left as-is; tune it
    separately. NOTE: (u, f) over-determine the separation rate, so the implied
    s_eff (~0.044 at u=0.058, f=0.7) follows -- it is not an independent target.
    Returns a recalibrated NestedDMPParameters.
    """
    def u_resid(s_bar: float) -> float:
        p = replace(params, s_bar=s_bar)
        mu = calibrate_mu_to_f(p, f_target=f_target, sigma_ref=sigma_ref)
        p = replace(p, mu=mu)
        grid, erg = _dyn._fixed_grid(p, sigma_ref=sigma_ref)
        ss = _dyn._grid_steady_state(p, grid=grid, erg=erg, sigma=0.0, pi=p.prior_pi0)
        phi = _dyn._stationary_phi(p, theta=ss["theta"], J=ss["J"], sigma=0.0, grid=grid, erg=erg)
        return float(1.0 - phi.sum()) - u_target

    s_bar = float(brentq(u_resid, s_lo, s_hi, xtol=1e-6))
    p = replace(params, s_bar=s_bar)
    mu = calibrate_mu_to_f(p, f_target=f_target, sigma_ref=sigma_ref)
    return replace(p, mu=mu)


def default_baseline():
    """Baseline calibration at the model's default parameters.

    Tunes (mu, s_bar) to US labor-market flow targets (u~0.058, f~0.7) at the
    frozen defaults rho_sigma=0.7 and psi_down=0.0 (flexible wage). Returns a
    recalibrated NestedDMPParameters.
    """
    return baseline_calibration(NestedDMPParameters())


__all__ = ["EmpiricalIRFTargets", "load_empirical_irf_targets", "MEAN_URATE_PP",
           "calibrate_mu_to_f", "model_regime_irfs", "FitReport", "fit_report",
           "EstimationResult", "estimate_shape_match", "ablate_beliefs",
           "baseline_calibration", "default_baseline"]
