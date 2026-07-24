"""Airline-model (SARIMA (0,1,1)(0,1,1)_s) regARIMA core for native X-11.

Exact Gaussian MLE of the "airline" model on the twice-differenced
series, plus fore/backcast extension — the pre-adjustment step of the
native X-11 engine (:mod:`puremacro.sa.x11`). Pure numpy + the house
Kalman (:mod:`puremacro.state_space`); no external binary.

Model. For y (possibly log-transformed), let

    w_t = (1 - B)(1 - B^s) y_t ,
    w_t = (1 - theta B)(1 - Theta B^s) eps_t ,   eps ~ N(0, sigma2).

w is a pure MA(q) with q = s + 1 and coefficients c_1 = -theta,
c_s = -Theta, c_{s+1} = theta * Theta (all other c_j = 0).

State space. state = (eps_t, eps_{t-1}, ..., eps_{t-q})' (dim q + 1),
transition = down-shift, shock enters the first element, observation
row Z = (1, c_1, ..., c_q). All states are white noises, so the EXACT
stationary initialization is P0 = sigma2 * I — no diffuse machinery
needed. The filtered terminal state *is* (eps_T, ..., eps_{T-q})-hat,
which makes MA forecasting a direct read-off:

    E[w_{T+h}] = sum_{j >= h} c_j eps_hat_{T+h-j}   (h <= q; 0 beyond).

Forecasts of y integrate w through the double difference; backcasts run
the identical machinery on the time-reversed series (an ARMA process
and its time reversal share the autocovariance structure, hence the
same (theta, Theta) — the convention X-13 itself uses for backcasts).

Estimation: L-BFGS on (atanh-scaled theta, Theta, log sigma2) — 3 free
parameters, invertibility enforced by the 0.98 * tanh reparametrization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from puremacro.state_space import StateSpaceModel, kalman_filter

_INV_BOUND = 0.98
_H_RIDGE = 1e-10


@dataclass(frozen=True)
class AirlineFit:
    """MLE of the airline model on a (transformed) series."""

    theta: float
    Theta: float
    sigma2: float
    loglik: float
    period: int
    converged: bool
    n_obs: int          # number of w observations entering the likelihood


def _ma_coeffs(theta: float, Theta: float, s: int) -> np.ndarray:
    c = np.zeros(s + 1)
    c[0] = -theta
    c[s - 1] = -Theta
    c[s] = theta * Theta
    return c


def _airline_ss(theta: float, Theta: float, sigma2: float,
                s: int) -> StateSpaceModel:
    q = s + 1
    m = q + 1
    T = np.zeros((m, m))
    T[1:, :-1] = np.eye(m - 1)
    R = np.zeros((m, 1))
    R[0, 0] = 1.0
    Z = np.concatenate([[1.0], _ma_coeffs(theta, Theta, s)])[None, :]
    return StateSpaceModel(T=T, Z=Z, R=R, Q=np.array([[sigma2]]),
                           H=np.array([[_H_RIDGE]]),
                           c=np.zeros(m), d=np.zeros(1))


def _filter(w: np.ndarray, theta: float, Theta: float, sigma2: float,
            s: int) -> dict:
    model = _airline_ss(theta, Theta, sigma2, s)
    m = model.T.shape[0]
    return kalman_filter(w, model, a0=np.zeros(m),
                         P0=sigma2 * np.eye(m))


def double_difference(y: np.ndarray, s: int) -> np.ndarray:
    """(1 - B)(1 - B^s) y — drops the first s + 1 observations."""
    d1 = np.diff(y)
    return d1[s:] - d1[:-s]


def fit_airline(y: np.ndarray, s: int) -> AirlineFit:
    """Exact MLE of the airline model for the (transformed) series y."""
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        raise ValueError("fit_airline: y must be gap-free (interpolate "
                         "upstream; X-11 needs a complete calendar).")
    w = double_difference(y, s)
    if len(w) < 3 * s:
        raise ValueError(f"fit_airline: too short (T={len(y)}, s={s})")
    # NO demeaning: the X-13 airline default carries no constant, so
    # E[w] = 0 by assumption. Estimating and re-adding a mean would
    # inject a drift into the fore/backcasts that the binary's
    # extension does not have (it distorts small noisy series badly).
    s2_0 = float(np.var(w)) or 1.0

    def unpack(x):
        return (_INV_BOUND * np.tanh(x[0]), _INV_BOUND * np.tanh(x[1]),
                float(np.exp(x[2])))

    def neg_ll(x):
        th, Th, s2 = unpack(x)
        try:
            return -_filter(w, th, Th, s2, s)["loglik"]
        except np.linalg.LinAlgError:
            return 1e12

    x0 = np.array([np.arctanh(0.3 / _INV_BOUND),
                   np.arctanh(0.3 / _INV_BOUND), np.log(s2_0)])
    res = optimize.minimize(neg_ll, x0, method="L-BFGS-B")
    th, Th, s2 = unpack(res.x)
    return AirlineFit(theta=float(th), Theta=float(Th), sigma2=float(s2),
                      loglik=float(-res.fun), period=s,
                      converged=bool(res.success), n_obs=int(len(w)))


def _forecast_w(w: np.ndarray, fit: AirlineFit, h_max: int) -> np.ndarray:
    """E[w_{T+1..T+h_max}] from the filtered terminal epsilon stack."""
    s = fit.period
    q = s + 1
    kf = _filter(w, fit.theta, fit.Theta, fit.sigma2, s)
    eps = kf["a_filt"][-1]              # (eps_T, ..., eps_{T-q})-hat
    c = _ma_coeffs(fit.theta, fit.Theta, s)
    out = np.zeros(h_max)
    for h in range(1, min(h_max, q) + 1):
        # E[w_{T+h}] = sum_{j=h..q} c_j * eps_{T+h-j}; eps index j-h.
        out[h - 1] = float(sum(c[j - 1] * eps[j - h]
                               for j in range(h, q + 1)))
    return out


def extend(y: np.ndarray, fit: AirlineFit, n_back: int,
           n_fore: int) -> np.ndarray:
    """Return y extended with n_back backcasts and n_fore forecasts.

    Forecasts integrate the w-forecasts through the double difference;
    backcasts apply the identical procedure to the reversed series with
    the same parameters."""
    y = np.asarray(y, dtype=float)
    s = fit.period

    def _fore(z: np.ndarray, h_max: int) -> np.ndarray:
        w = double_difference(z, s)
        wf = _forecast_w(w, fit, h_max)
        zx = list(z)
        for h in range(h_max):
            t = len(zx)
            zx.append(wf[h] + zx[t - 1] + zx[t - s] - zx[t - s - 1])
        return np.asarray(zx[len(z):])

    fore = _fore(y, n_fore) if n_fore else np.empty(0)
    back = _fore(y[::-1], n_back)[::-1] if n_back else np.empty(0)
    return np.concatenate([back, y, fore])


def choose_log(y: np.ndarray, s: int) -> tuple[bool, dict]:
    """X-13-style transform{function=auto}: log vs level by AICC with
    the log-transform Jacobian correction (sum of log y over the
    effective sample). Returns (use_log, diagnostics). Series with
    non-positive values are level by construction."""
    y = np.asarray(y, dtype=float)
    diag: dict = {}
    if (y <= 0).any():
        return False, {"reason": "non-positive values"}
    k = 3.0

    def aicc(ll: float, n: int) -> float:
        return -2.0 * ll + 2.0 * k * n / max(n - k - 1.0, 1.0)

    f_lvl = fit_airline(y, s)
    f_log = fit_airline(np.log(y), s)
    n = f_lvl.n_obs
    jac = float(np.sum(np.log(y[s + 1:])))
    a_lvl = aicc(f_lvl.loglik, n)
    a_log = aicc(f_log.loglik - jac, n)
    diag = {"aicc_level": round(a_lvl, 3), "aicc_log": round(a_log, 3)}
    return bool(a_log < a_lvl), diag
