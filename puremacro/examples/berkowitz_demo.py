"""Berkowitz (2001) joint LR test: a more powerful PIT diagnostic.

The KS-uniformity test on PIT (already in Tier 1's ``forecast.density``)
catches gross calibration failures but has weak power against time-
dependent miscalibration. Berkowitz transforms PITs through Phi^{-1}
so the null becomes "iid N(0, 1)", then runs a joint LR on
(μ = 0, σ² = 1, ρ = 0) in an AR(1).

We compare three forecast settings:
  (a) correctly calibrated  — should fail to reject.
  (b) shifted mean          — KS may catch it; Berkowitz definitely will.
  (c) AR(1)-correlated forecast errors (calibrated marginally,
      misspecified dynamically) — KS misses; Berkowitz catches via ρ.

Run:
    python -m puremacro.examples.berkowitz_demo
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..forecast.density import (
    pit, pit_uniformity_test, berkowitz_test,
)


def main() -> None:
    rng = np.random.default_rng(0)
    T = 500

    # (a) Calibrated.
    y_a = rng.standard_normal(T)
    pit_a = norm.cdf(y_a)
    # (b) Mean shift.
    y_b = rng.standard_normal(T) + 0.4
    pit_b = norm.cdf(y_b)
    # (c) AR(1) errors but marginally calibrated.
    z_c = np.zeros(T)
    eta = rng.standard_normal(T) * np.sqrt(1 - 0.6 ** 2)
    for t in range(1, T):
        z_c[t] = 0.6 * z_c[t - 1] + eta[t]
    z_c = (z_c - z_c.mean()) / z_c.std()  # standardise marginally
    pit_c = norm.cdf(z_c)

    rows = []
    for label, p in [("calibrated  ", pit_a),
                      ("mean shift +0.4", pit_b),
                      ("AR(1) errors (marg. calibrated)", pit_c)]:
        ks = pit_uniformity_test(p)
        bk = berkowitz_test(p)
        rows.append((label, ks, bk))

    print("Berkowitz LR vs KS uniformity on PIT")
    print()
    print(f"  {'case':<32}  {'KS p-val':>10}   {'Berkowitz LR':>14}  {'p':>8}  ρ̂")
    for label, ks, bk in rows:
        print(f"  {label:<32}  {ks['p_value']:>10.3f}   "
              f"{bk['stat']:>14.2f}  {bk['p_value']:>8.4f}  "
              f"{bk['rho_hat']:+.3f}")
    print()
    print("  → KS misses (c); Berkowitz catches it via the AR(1) coefficient.")


if __name__ == "__main__":
    main()
