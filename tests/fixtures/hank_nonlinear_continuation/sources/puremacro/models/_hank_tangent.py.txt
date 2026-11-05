"""Piecewise analytic tangents of the published one-asset household block.

Interpolation brackets and the binding set are held at the evaluation point.
At a knot this selects a branch derivative, not a classical derivative across
the kink. No numerical perturbation or differentiation package is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .one_asset_hank import _lottery


@dataclass
class _HouseholdTape:
    uc: np.ndarray
    ce: np.ndarray
    ne: np.ndarray
    endog: np.ndarray
    lo: np.ndarray
    weight: np.ndarray
    constrained: np.ndarray
    slope_c: np.ndarray
    slope_n: np.ndarray


def _household_tape(expected, a, e, transition, p, r, w, transfer):
    uc = p["beta"] * (expected @ transition.T)
    ce = uc ** -p["eis"]
    ne = (w * e * uc / p["vphi"]) ** p["frisch"]
    endog = (ce - w * e * ne + a[:, None] - transfer * e) / (1 + r)
    lo = np.empty(ce.shape, dtype=np.intp)
    for z in range(len(e)):
        lo[:, z] = np.clip(np.searchsorted(endog[:, z], a, side="right") - 1, 0, len(a) - 2)
    left = np.take_along_axis(endog, lo, axis=0)
    step = np.take_along_axis(endog, lo + 1, axis=0) - left
    weight = (a[:, None] - left) / step

    def interpolate(y):
        yl = np.take_along_axis(y, lo, axis=0)
        return yl + weight * (np.take_along_axis(y, lo + 1, axis=0) - yl)

    unconstrained_saving = ((1 + r) * a[:, None] + transfer * e
                            + w * e * interpolate(ne) - interpolate(ce))
    slope_c = (np.take_along_axis(ce, lo + 1, axis=0) - np.take_along_axis(ce, lo, axis=0)) / step
    slope_n = (np.take_along_axis(ne, lo + 1, axis=0) - np.take_along_axis(ne, lo, axis=0)) / step
    return _HouseholdTape(uc, ce, ne, endog, lo, weight, unconstrained_saving < a[0], slope_c, slope_n)


def _household_jvp(tape, expected_dot, a, e, transition, p, r, w, c, n,
                   r_dot, w_dot, transfer_dot):
    """Tangent of marginal value, saving, consumption and hours, in that order."""
    q = p["beta"] * (expected_dot @ transition.T) / tape.uc
    ce_dot = -p["eis"] * tape.ce * q
    ne_dot = p["frisch"] * tape.ne * (w_dot / w + q)
    endog_dot = (ce_dot - w_dot * e * tape.ne - w * e * ne_dot
                 - transfer_dot * e - tape.endog * r_dot) / (1 + r)
    lo, weight = tape.lo, tape.weight

    grid_dl = np.take_along_axis(endog_dot, lo, axis=0)
    grid_dr = np.take_along_axis(endog_dot, lo + 1, axis=0)
    grid_shift = grid_dl + weight * (grid_dr - grid_dl)

    def interpolate_dot(dy, slope):
        dl, dr = np.take_along_axis(dy, lo, axis=0), np.take_along_axis(dy, lo + 1, axis=0)
        return dl + weight * (dr - dl) - slope * grid_shift

    c_dot, n_dot = interpolate_dot(ce_dot, tape.slope_c), interpolate_dot(ne_dot, tape.slope_n)
    cash_dot = r_dot * a[:, None] + transfer_dot * e
    bound = tape.constrained
    if np.any(bound):
        # Differentiate c - w*e*n = cash-a_min and n=(w*e*c^-1/eis/vphi)^frisch.
        # This includes labor's wage response at the borrowing constraint.
        we = w * e
        log_uc_dot = -(cash_dot + (1 + p["frisch"]) * w_dot * e * n) / (
            p["eis"] * c + p["frisch"] * we * n)
        c_dot[bound] = (-p["eis"] * c * log_uc_dot)[bound]
        n_dot[bound] = (p["frisch"] * n * (w_dot / w + log_uc_dot))[bound]
    saving_dot = cash_dot + w_dot * e * n + w * e * n_dot - c_dot
    saving_dot[bound] = 0.
    marginal_dot = c ** (-1 / p["eis"]) * (r_dot - (1 + r) / p["eis"] * c_dot / c)
    return marginal_dot, saving_dot, c_dot, n_dot


def _forward_tape(policy, a):
    lo, weight = _lottery(policy, a)
    return lo, weight, 1 / (a[lo + 1] - a[lo])


def _forward_jvp(distribution, distribution_dot, policy_dot, tape, transition):
    lo, weight, inverse_step = tape
    moving_mass = distribution * policy_dot * inverse_step
    low = distribution_dot * (1 - weight) - moving_mass
    high = distribution_dot * weight + moving_mass
    out = np.zeros_like(distribution)
    for z in range(distribution.shape[1]):
        out[:, z] = (np.bincount(lo[:, z], weights=low[:, z], minlength=len(out))
                     + np.bincount(lo[:, z] + 1, weights=high[:, z], minlength=len(out)))
    return out @ transition
