"""Mertens-Ravn (2013)-style proxy-SVAR replication on synthetic data.

We do not have Mertens-Ravn's narrative tax-shock series, so this script
runs proxy-SVAR identification on a controlled DGP: a stable VAR(2) in
three variables with a known structural impact matrix and a noisy proxy
correlated only with the first structural shock.

The DGP is wired so the recovered impulse response on the second
variable peaks around horizon 6-7. We then check (i) Cragg-Donald F is
strong, (ii) recovered peak is within 10% of the population truth, and
(iii) Montiel-Olea / Stock / Watson weak-IV-robust bands are well-formed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..inference.weak_iv import cragg_donald_f, msw_bands
from ..var.estimate import estimate_var
from ..var.identify.proxy import proxy_svar


# DGP constants. Holding these at module scope so tests + main() see the
# exact same population truth.
_A1 = np.array([
    [0.7, 0.0, 0.0],
    [0.0, 0.6, 0.0],
    [0.0, 0.0, 0.5],
])
_A2 = np.array([
    [0.1, 0.0, 0.0],
    [0.5, 0.2, 0.0],
    [0.0, 0.3, 0.1],
])
_B0 = np.array([
    [-1.0, 0.0, 0.0],
    [ 0.0, 0.6, 0.0],
    [ 0.0, 0.0, 0.5],
])
_PROXY_NOISE_SD = 0.5
_VAR_NAMES = ("y0", "y1", "y2")


def _population_proxy_irf(horizon: int = 20) -> np.ndarray:
    """Compute the population proxy-SVAR impact column and IRF.

    Returns an array of shape ``(horizon + 1, n)`` giving the response of
    each variable to the (Stock-Watson-normalized) structural shock 1.
    """
    Sigma = _B0 @ _B0.T
    var_z = 1.0 + _PROXY_NOISE_SD ** 2  # var(eps_0) + var(noise)
    Pi = _B0[:, 0] / var_z
    SP = Sigma @ Pi
    norm = float(np.sqrt(Pi @ SP))
    b_col1 = SP / norm

    n = _B0.shape[0]
    A_list = [_A1, _A2]
    p = len(A_list)
    Phi = [np.eye(n)]
    for h in range(1, horizon + 1):
        s = np.zeros((n, n))
        for k in range(min(h, p)):
            s += A_list[k] @ Phi[h - k - 1]
        Phi.append(s)
    return np.array([P @ b_col1 for P in Phi])


def _make_synthetic_svariv_dgp(
    T: int = 300,
    seed: int = 0,
    burn: int = 200,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a 3-variable VAR(2) with a known structural shock.

    The first structural shock (the 'tax shock') has direct impact -1.0
    on the first variable, no contemporaneous effect on the others, and
    a delayed peak indirect effect on the second variable around
    horizon 6-7. The proxy is ``eps_1 + 0.5 * noise``: structurally
    exogenous and strongly correlated with shock 1.

    Returns ``(Y, proxy)`` where ``Y`` is a ``(T, 3)`` DataFrame with
    columns ``[y0, y1, y2]`` and ``proxy`` is a ``(T,)`` Series.
    """
    rng = np.random.default_rng(seed)
    n = _B0.shape[0]
    p = 2
    eps_full = rng.standard_normal((T + burn, n))
    Y_full = np.zeros((T + burn, n))
    for t in range(p, T + burn):
        Y_full[t] = _A1 @ Y_full[t - 1] + _A2 @ Y_full[t - 2] + _B0 @ eps_full[t]
    Y = Y_full[burn:]
    eps = eps_full[burn:]
    proxy = eps[:, 0] + _PROXY_NOISE_SD * rng.standard_normal(T)

    Y_df = pd.DataFrame(Y, columns=list(_VAR_NAMES))
    proxy_s = pd.Series(proxy, name="proxy")
    return Y_df, proxy_s


def run_mertens_ravn_replication(
    seed: int = 0,
    T: int = 300,
    horizon: int = 20,
    n_boot: int = 200,
) -> dict:
    """Run proxy-SVAR identification on the synthetic DGP.

    Returns a dict with keys:

    - ``irf_point`` : ndarray ``(H+1, n, n)`` from ``proxy_svar``.
    - ``irf_lower`` / ``irf_upper`` : same shape, wild-bootstrap bands.
    - ``cd_f`` : Cragg-Donald first-stage F.
    - ``msw_lo`` / ``msw_hi`` : Anderson-Rubin (MSW) band on the scalar
      impact ratio of var1's residual on var0's residual.
    - ``msw_grid`` : grid used for MSW inversion.
    - ``population_irf`` : population truth, shape ``(H+1, n)``.
    - ``true_peak`` / ``true_peak_h`` : population peak |response| of var1.
    - ``recovered_peak`` / ``recovered_peak_h``.
    - ``Y`` / ``proxy`` : the simulated data.
    """
    Y_df, proxy_s = _make_synthetic_svariv_dgp(T=T, seed=seed)
    Y = Y_df.values
    proxy = proxy_s.values
    p = 2

    # Proxy-SVAR with wild-bootstrap bands.
    res = proxy_svar(
        Y, p=p, horizon=horizon,
        instrument_series=proxy, n_boot=n_boot, ci=0.9, seed=seed,
    )

    # Cragg-Donald F on the reduced-form residuals.
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    T_eff = resid.shape[0]
    z = (proxy[-T_eff:] - proxy[-T_eff:].mean()).reshape(-1, 1)
    X_endog = resid[:, [0]]
    cd_f = float(cragg_donald_f(np.zeros(T_eff), X_endog, z))

    # MSW bands: invert AR test for the scalar impact ratio
    # b1 = E[u1 z] / E[u0 z]  (response of var1 per unit response of var0).
    y_resp = resid[:, 1]
    x_endog = resid[:, 0]
    beta_2sls = float((z[:, 0] @ y_resp) / (z[:, 0] @ x_endog))
    grid = np.linspace(beta_2sls - 1.0, beta_2sls + 1.0, 81)
    msw_lo, msw_hi = msw_bands(grid, y_resp, x_endog, z, alpha=0.10)

    # Population truth + recovered peak (variable 1).
    pop_irf = _population_proxy_irf(horizon=horizon)
    true_peak_h = int(np.argmax(np.abs(pop_irf[:, 1])))
    true_peak = float(pop_irf[true_peak_h, 1])
    recovered_path = res.irf_point[:, 1, 0]
    recovered_peak_h = int(np.argmax(np.abs(recovered_path)))
    recovered_peak = float(recovered_path[recovered_peak_h])

    return {
        "irf_point": res.irf_point,
        "irf_lower": res.irf_lower,
        "irf_upper": res.irf_upper,
        "first_stage_F": res.first_stage_F,
        "cd_f": cd_f,
        "msw_lo": float(msw_lo),
        "msw_hi": float(msw_hi),
        "msw_grid": grid,
        "population_irf": pop_irf,
        "true_peak": true_peak,
        "true_peak_h": true_peak_h,
        "recovered_peak": recovered_peak,
        "recovered_peak_h": recovered_peak_h,
        "Y": Y_df,
        "proxy": proxy_s,
        "horizon": horizon,
        "var_names": list(_VAR_NAMES),
    }


def main() -> None:
    """Execute the synthetic replication and print a short summary."""
    result = run_mertens_ravn_replication()
    print("Mertens-Ravn (2013)-style proxy-SVAR on synthetic DGP")
    print(f"  Cragg-Donald F:    {result['cd_f']:.1f}  (Stock-Yogo 10% cutoff = 16.4)")
    print(f"  Population peak:   {result['true_peak']:+.3f} at h={result['true_peak_h']}")
    print(f"  Recovered peak:    {result['recovered_peak']:+.3f} at "
          f"h={result['recovered_peak_h']}")
    print(f"  MSW 90% band on impact ratio: [{result['msw_lo']:+.3f}, "
          f"{result['msw_hi']:+.3f}]")
    print(f"  Olea-Pflueger F:   {result['first_stage_F']:.2f}")

    out_dir = Path(__file__).parent / "output"
    if out_dir.is_dir():
        from ..plot import plot_irf_single
        H = result["horizon"]
        irf_df = pd.DataFrame({
            "h": np.arange(H + 1),
            "beta": result["irf_point"][:, 1, 0],
            "lo": result["irf_lower"][:, 1, 0],
            "hi": result["irf_upper"][:, 1, 0],
        })
        fig = plot_irf_single(
            irf_df,
            title="Synthetic proxy-SVAR: response of y1 to shock 1",
            ylabel="Response",
        )
        fig.savefig(out_dir / "svariv_mertens_ravn_irf.png",
                    dpi=120, bbox_inches="tight")
        print(f"  Figure saved: {out_dir / 'svariv_mertens_ravn_irf.png'}")


if __name__ == "__main__":
    main()
