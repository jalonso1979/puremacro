"""TVP-VAR with stochastic volatility (Primiceri 2005, KSC sampler).

Simulate a 2-variable VAR whose lag coefficients shift at the midpoint
of the sample AND whose innovation volatility doubles in the second
half. The Gibbs sampler should recover both the time-varying
coefficient drift and the SV jump in log h_t.

This is the "complete" Primiceri picture: time-varying β AND
time-varying volatility. Compared to the constant-Σ ``tvp_var_fit``,
the SV path lets the model fit volatile windows (e.g. 2008, 2020)
without inflating the coefficient-uncertainty bands.

Run:
    python -m puremacro.examples.tvp_var_sv_demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.tvp import tvp_var_sv_fit


def _simulate(T: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    A_pre = np.array([[0.5, 0.0], [0.05, 0.4]])
    A_post = np.array([[0.2, 0.1], [0.0, 0.7]])
    sd_pre = np.array([0.3, 0.3])
    sd_post = np.array([0.8, 0.5])
    Y = np.zeros((T, 2))
    cut = T // 2
    for t in range(1, T):
        A = A_pre if t < cut else A_post
        sd = sd_pre if t < cut else sd_post
        Y[t] = A @ Y[t - 1] + sd * rng.standard_normal(2)
    return Y, sd_pre, sd_post


def run_demo() -> dict:
    Y, sd_pre, sd_post = _simulate()
    res = tvp_var_sv_fit(Y, p=1, n_draws=150, burn=300,
                          rng=np.random.default_rng(0))
    return {"Y": Y, "sd_pre": sd_pre, "sd_post": sd_post, "res": res}


def main() -> None:
    out = run_demo()
    res = out["res"]
    log_h_mean = res.log_h_draws.mean(axis=0)        # (T_eff, n)
    sigma_mean = np.exp(0.5 * log_h_mean)
    log_h_lo = np.percentile(res.log_h_draws, 5, axis=0)
    log_h_hi = np.percentile(res.log_h_draws, 95, axis=0)
    print("TVP-VAR-SV (Kim-Shephard-Chib mixture sampler)")
    print(f"  T_eff = {res.T_eff}, n_draws = {res.n_draws}, n = {res.n}")
    print()
    print("Posterior-mean σ_t (= exp(0.5 log h_t)) at sample midpoints:")
    cut = res.T_eff // 2
    for i in range(res.n):
        sigma_pre_hat = sigma_mean[cut // 2, i]
        sigma_post_hat = sigma_mean[cut + cut // 2, i]
        print(f"  variable {i}:  early σ̂ = {sigma_pre_hat:.3f}  "
              f"(true {out['sd_pre'][i]:.3f}),  "
              f"late σ̂ = {sigma_post_hat:.3f}  (true {out['sd_post'][i]:.3f})")
    print()
    A_mean = res.A_draws.mean(axis=0)
    print(f"  Posterior-mean contemporaneous A (lower-triangular):\n"
          f"{np.round(A_mean, 3)}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
        idx = np.arange(res.T_eff)
        for i in range(res.n):
            axes[i].plot(idx, sigma_mean[:, i],
                         color="0.0", lw=1.0, label=f"posterior σ̂_t (var {i})")
            axes[i].fill_between(
                idx,
                np.exp(0.5 * log_h_lo[:, i]),
                np.exp(0.5 * log_h_hi[:, i]),
                color="0.0", alpha=0.15,
            )
            axes[i].axhline(out["sd_pre"][i], color="0.5", lw=0.6, ls="--",
                            label=f"true σ pre/post = {out['sd_pre'][i]:.2f} → "
                                   f"{out['sd_post'][i]:.2f}")
            axes[i].axhline(out["sd_post"][i], color="0.5", lw=0.6, ls="--")
            axes[i].axvline(res.T_eff // 2, color="0.7", lw=0.5)
            axes[i].set_ylabel(f"σ_t (var {i})")
            axes[i].legend(frameon=False, fontsize=8)
        axes[1].set_xlabel("t")
        fig.suptitle("TVP-VAR-SV: posterior σ_t with KSC mixture sampler")
        fig.tight_layout()
        fig.savefig(out_dir / "tvp_var_sv.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'tvp_var_sv.png'}")


if __name__ == "__main__":
    main()
