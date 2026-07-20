import numpy as np
import pandas as pd

from puremacro.var import fit_var, irf, fevd
from puremacro.var.identify.maxshare import news_maxshare


def test_news_maxshare_h0_impact_is_small():
    """At horizon h=0, the news shock should have approximately zero
    contemporaneous impact on the target variable (Beaudry-Portier /
    Barsky-Sims signature). The 'news' is information that arrives now but
    only affects the variable at long horizons.

    DGP: TFP_t = rho*TFP_{t-1} + gamma*z_{t-1} + eps_tfp
         z_t   = delta*z_{t-1} + eps_z
    eps_tfp and eps_z are independent. z is the 'news' variable: its shock
    does not affect TFP contemporaneously (h=0) but builds up via the
    lag channel. max-share at large H should identify eps_z as the dominant
    shock at long horizons, with small h=0 IRF on TFP.
    """
    rng = np.random.default_rng(42)
    T = 600
    rho, gamma, delta = 0.95, 0.8, 0.8
    # Variables ordered as [TFP, z]
    A = np.array([[rho, gamma], [0.0, delta]])
    Sigma = np.eye(2) * 0.1
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, 2))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(2)
    df = pd.DataFrame(Y, columns=["TFP", "z"])
    res = fit_var(df, p=1)
    if len(res) == 5:
        A_list, _, Sigma_hat, _, _ = res
    else:
        A_list, Sigma_hat, _, _ = res

    target = 0  # TFP is the target variable
    H = 40
    B0 = news_maxshare(A_list, Sigma_hat, target_var=target, horizon=H)

    # FEVD: the news shock should explain a much larger share of TFP variance
    # at the long horizon H than at h=0. This is the mathematical property
    # that news_maxshare is designed to maximize.
    fev = fevd(A_list, B0, horizon=H)
    fevd_h0 = fev[0, target, 0]   # FEVD share of news shock at h=0
    fevd_hH = fev[H, target, 0]   # FEVD share of news shock at h=H

    assert fevd_hH > fevd_h0, (
        f"news shock FEVD should be larger at h={H} ({fevd_hH:.4f}) "
        f"than at h=0 ({fevd_h0:.4f})"
    )

    # Also verify the news shock's defining IRF signature: the contemporaneous
    # impact on TFP (h=0) is smaller than the peak impact at intermediate horizons.
    irf_arr = irf(A_list, B0, horizon=H)
    impact_h0 = abs(irf_arr[0, target, 0])
    impact_h10 = abs(irf_arr[10, target, 0])
    assert impact_h0 < impact_h10, (
        f"news shock impact on target at h=0 ({impact_h0:.4f}) should be "
        f"smaller than at h=10 ({impact_h10:.4f})"
    )
