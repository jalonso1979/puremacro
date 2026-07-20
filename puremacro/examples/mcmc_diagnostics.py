"""MCMC convergence diagnostics on a real Gibbs sampler.

We run two independent ``minnesota_gibbs`` chains from different
starting seeds, then check convergence with three diagnostics:

  - Geweke (1992) z-score per chain.
  - Gelman-Rubin R̂ across chains.
  - Effective sample size per chain.

A small ``n_draws`` deliberately produces a marginal R̂ to show what a
not-yet-converged sampler looks like; bumping ``n_draws`` should drive
R̂ → 1 and ESS up.

Run:
    python -m puremacro.examples.mcmc_diagnostics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..var.bvar import minnesota_gibbs
from ..mcmc import (
    geweke_z, gelman_rubin, autocorrelations,
    effective_sample_size, trace_summary,
)


def _simulate(T: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.10, 0.0],
                   [0.05, 0.5, 0.10],
                   [0.0, 0.05, 0.55]])
    Sigma = np.array([[1.0, 0.3, 0.1],
                       [0.3, 1.0, 0.2],
                       [0.1, 0.2, 1.0]]) * 0.5 ** 2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(3)
    return Y


def run_demo(n_draws: int = 600, n_chains: int = 3) -> dict:
    Y = _simulate()
    chains = []
    for c in range(n_chains):
        g = minnesota_gibbs(Y, p=1, n_draws=n_draws, burn=200,
                             rng=np.random.default_rng(100 + c))
        # Track the (1, 0, 0) element of A_1 across the chain.
        chains.append(g["A_draws"][:, 0, 0, 0])
    chains_arr = np.stack(chains)
    return {"Y": Y, "chains": chains_arr}


def main() -> None:
    out = run_demo()
    chains = out["chains"]
    print("MCMC diagnostics on minnesota_gibbs (parameter A_1[0,0])")
    print(f"  Chains: {chains.shape[0]} × {chains.shape[1]} draws")
    print()
    for c, chain in enumerate(chains):
        z = geweke_z(chain)
        ess = effective_sample_size(chain)
        print(f"  Chain {c}: mean={chain.mean():.4f}  std={chain.std():.4f}  "
              f"Geweke z={z:+.2f}  ESS={ess:.0f}")
    print()
    gr = gelman_rubin(chains)
    print(f"  Gelman-Rubin R̂ = {gr['R_hat']:.4f}  (want ≤ 1.01 for convergence)")
    print(f"  Within-chain var W = {gr['W']:.5f}, "
          f"between-chain var B = {gr['B']:.5f}")
    print()
    print("Trace summary across all chains:")
    summ = trace_summary(chains)
    print(f"  Posterior mean = {summ['mean']:.4f},  std = {summ['std']:.4f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5))
        for c, chain in enumerate(chains):
            axes[0].plot(chain, lw=0.6, color=f"{0.6 - c*0.2:.2f}",
                         label=f"chain {c}")
        axes[0].set_title(f"Trace of A_1[0,0]   (R̂={gr['R_hat']:.3f})")
        axes[0].legend(frameon=False)
        for c, chain in enumerate(chains):
            acf = autocorrelations(chain, max_lag=60)
            axes[1].plot(acf, lw=0.7, color=f"{0.6 - c*0.2:.2f}",
                         label=f"chain {c}")
        axes[1].axhline(0, color="0.5", lw=0.4)
        axes[1].set_xlabel("lag"); axes[1].set_ylabel("ACF")
        axes[1].set_title("Sample autocorrelation per chain")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "mcmc_diagnostics.png", dpi=120,
                    bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'mcmc_diagnostics.png'}")


if __name__ == "__main__":
    main()
