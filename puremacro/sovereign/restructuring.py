"""Sovereign Spread Term Structure Decomposition & Debt Restructuring Counterfactuals.

Implements:
1. `decompose_spread_term_structure`: Multi-horizon forward zero-coupon survival
   pricing, decomposing sovereign yields and spreads into physical expected default
   log-survival loss vs. the discrete-compounding adjustment across horizons
   (e.g. 1Q, 1Y, 2Y, 3Y, 5Y, 10Y).
2. `evaluate_debt_restructuring`: Evaluates debt restructuring counterfactuals
   (face-value haircuts and maturity extensions), computing immediate sovereign
   spread compression, default hazard mitigation, and post-restructuring
   recovery paths.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from puremacro.sovereign._results import RestructuringResult, SovereignDefaultResult

__all__ = [
    "decompose_spread_term_structure",
    "evaluate_debt_restructuring",
]


def decompose_spread_term_structure(
    result: SovereignDefaultResult,
    horizons: Sequence[int] | None = None,
    *,
    maturities: Sequence[int] | None = None,
    b_eval: float | None = None,
    y_eval: float | None = None,
) -> pd.DataFrame:
    """Decompose sovereign yield and spread term structure across maturities.

    Recursively prices m-period zero-coupon defaultable claims conditional on
    no default occurring prior to maturity, decomposing the credit spread into:
        Spread(m) = Log-survival Loss(m) + Compounding Adjustment(m)

    Lenders here are risk neutral: q_m = survival_m / (1+r_star)**m.
    The legacy column ``risk_premium_component`` is the difference between
    the discrete annual spread and the continuously compounded loss rate.
    It does not identify a structural risk, liquidity or dilution premium.

    Parameters
    ----------
    result : SovereignDefaultResult
        Solved sovereign default model result.
    horizons : Sequence[int], optional
        Maturity horizons in quarters (e.g. 1Q, 1Y, 2Y, 3Y, 5Y, 10Y).
    maturities : Sequence[int], optional
        Synonym for ``horizons`` (e.g. (1, 4, 12, 20, 40)).
    b_eval : float, optional
        Debt level at which to evaluate term structure. Defaults to median debt.
    y_eval : float, optional
        Endowment at which to evaluate term structure. Defaults to median income.

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - ``horizon_quarters``: Maturity in quarters m.
        - ``maturity_years``: Maturity in years m / 4.
        - ``bond_price``: Price of m-period zero-coupon survival bond q_m.
        - ``yield_to_maturity``: Annualized yield to maturity R_m.
        - ``spread_annualized``: Annualized sovereign spread over risk-free rate.
        - ``spread_bps``: Spread in basis points.
        - ``default_loss_component``: Physical expected default loss spread (bps).
        - ``risk_premium_component``: Discrete-compounding adjustment (bps),
          retained under its legacy column name for compatibility.
    """
    n_b = len(result.b_grid)
    n_y = len(result.y_grid)
    r_star = float(result.metadata.get("r_star", 0.01))

    if b_eval is None:
        ib = n_b // 2
    else:
        ib = int(np.argmin(np.abs(result.b_grid - b_eval)))

    if y_eval is None:
        iy = n_y // 2
    else:
        iy = int(np.argmin(np.abs(result.y_grid - y_eval)))

    # Policy indices for next-period debt
    policy_idx = np.zeros((n_b, n_y), dtype=int)
    for i in range(n_b):
        for j in range(n_y):
            policy_idx[i, j] = int(np.argmin(np.abs(result.b_grid - result.policy_b[i, j])))

    repay_prob = 1.0 - result.default_prob  # (n_b, n_y)
    debt_probabilities = result.metadata.get('debt_choice_probabilities')

    if maturities is not None:
        h_list = list(maturities)
    elif horizons is not None:
        h_list = list(horizons)
    else:
        h_list = [1, 4, 8, 12, 20, 40]

    if not h_list or any(isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1 for h in h_list):
        raise ValueError('horizons must contain positive integer quarters')

    max_h = max(h_list)
    q_tensor = np.zeros((max_h + 1, n_b, n_y), dtype=np.float64)
    s_tensor = np.zeros((max_h + 1, n_b, n_y), dtype=np.float64)

    # 1-period claim
    q_tensor[1] = (repay_prob @ result.p_trans.T) / (1.0 + r_star)
    s_tensor[1] = repay_prob @ result.p_trans.T

    # Recursive forward pricing
    for h in range(2, max_h + 1):
        q_tomorrow = np.take_along_axis(q_tensor[h - 1], policy_idx, axis=0)
        s_tomorrow = np.take_along_axis(s_tensor[h - 1], policy_idx, axis=0)
        if debt_probabilities is not None:
            q_tomorrow = np.einsum('imj,mj->ij', debt_probabilities, q_tensor[h - 1])
            s_tomorrow = np.einsum('imj,mj->ij', debt_probabilities, s_tensor[h - 1])

        payoff_q = repay_prob * q_tomorrow
        payoff_s = repay_prob * s_tomorrow

        q_tensor[h] = (payoff_q @ result.p_trans.T) / (1.0 + r_star)
        s_tensor[h] = payoff_s @ result.p_trans.T

    r_rf_ann = (1.0 + r_star) ** 4 - 1.0

    records = []
    for h in h_list:
        qh = float(q_tensor[h, ib, iy])
        sh = float(s_tensor[h, ib, iy])

        # No artificial positive price/survival floor: a worthless claim has
        # infinite yield, with an undefined infinity-minus-infinity adjustment.
        # Use log survival to avoid separately flooring discounted prices.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            loss_rate = max(0.0, -4.0 / float(h) * float(np.log(sh)))
            spread_ann = float((1.0 + r_rf_ann) * np.expm1(loss_rate))
            r_ann = r_rf_ann + spread_ann
            spread_bps = spread_ann * 10000.0
            loss_spread_bps = loss_rate * 10000.0
            risk_prem_bps = spread_bps - loss_spread_bps

        records.append({
            "maturity": int(h),
            "horizon_quarters": int(h),
            "maturity_years": float(h) / 4.0,
            "bond_price": qh,
            "yield": r_ann,
            "yield_to_maturity": r_ann,
            "spread": spread_ann,
            "spread_annualized": spread_ann,
            "spread_bps": spread_bps,
            "default_loss_component": loss_spread_bps,
            "risk_premium_component": risk_prem_bps,
        })

    frame = pd.DataFrame(records)
    frame.attrs['pricing'] = 'risk-neutral zero-recovery survival claims'
    frame.attrs['risk_premium_component'] = 'discrete-compounding adjustment; no identified risk premium'
    return frame


def evaluate_debt_restructuring(
    result: SovereignDefaultResult,
    haircut: float = 0.20,
    maturity_extension: float = 0.50,
    *,
    b_distress: float | None = None,
    y_distress: float | None = None,
    horizon: int = 40,
    seed: int = 42,
) -> RestructuringResult:
    """Evaluate sovereign debt restructuring counterfactuals.

    Evaluates a restructuring plan combining:
    1. Face-value debt haircut h in [0, 1): b_new = (1 - h) * b_distress.
    2. Maturity extension: increases duration by reducing delta to
       delta_new = delta / (1 + maturity_extension).

    Computes immediate sovereign spread relief, default hazard reduction, and
    simulates the recovery trajectory.

    Parameters
    ----------
    result : SovereignDefaultResult
        Baseline solved sovereign default model.
    haircut : float, default 0.20
        Debt face-value write-down fraction (e.g. 0.20 for a 20% haircut).
    maturity_extension : float, default 0.50
        Fractional maturity extension (e.g. 0.50 for a 50% duration increase).
    b_distress : float, optional
        Pre-restructuring debt level in distress. Defaults to 85th percentile of b_grid.
    y_distress : float, optional
        Endowment in distress. Defaults to recession state (2nd lowest income point).
    horizon : int, default 40
        Quarters to simulate post-restructuring transition.
    seed : int, default 42
        Random seed for simulation.

    Returns
    -------
    RestructuringResult
        Result containing pre- and post-restructuring spreads, default hazards,
        and post-restructuring recovery path.
    """
    from puremacro.sovereign.arellano import solve_sovereign_default

    if not np.isfinite(haircut) or haircut < 0.0 or haircut >= 1.0:
        raise ValueError(f"haircut must be in [0, 1); got {haircut}")
    if not np.isfinite(maturity_extension) or maturity_extension < 0.0:
        raise ValueError(f"maturity_extension must be >= 0; got {maturity_extension}")
    if not result.converged:
        raise ValueError('restructuring requires a converged baseline equilibrium')
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    if any(value is not None and not np.isfinite(value) for value in (b_distress, y_distress)):
        raise ValueError('distress debt and income must be finite')

    n_b = len(result.b_grid)
    n_y = len(result.y_grid)

    if b_distress is None:
        ib_distress = int(0.85 * (n_b - 1))
        b_distress = float(result.b_grid[ib_distress])
    else:
        ib_distress = int(np.argmin(np.abs(result.b_grid - b_distress)))
        b_distress = float(result.b_grid[ib_distress])

    if y_distress is None:
        iy_distress = min(2, n_y - 1)
        y_distress = float(result.y_grid[iy_distress])
    else:
        iy_distress = int(np.argmin(np.abs(result.y_grid - y_distress)))
        y_distress = float(result.y_grid[iy_distress])

    base_spread = float(result.spread[ib_distress, iy_distress])
    pre_def_prob = float(result.default_prob[ib_distress, iy_distress])

    # Restructured debt and maturity
    b_new = (1.0 - haircut) * b_distress
    delta_orig = result.maturity_param_delta
    delta_new = delta_orig / (1.0 + maturity_extension)

    # Extension in years: duration increases from 1/(4*delta_orig) to 1/(4*delta_new)
    duration_orig_yrs = 1.0 / (4.0 * delta_orig)
    duration_new_yrs = 1.0 / (4.0 * delta_new)
    maturity_extension_years = duration_new_yrs - duration_orig_yrs

    # A face-value haircut changes the inherited state only. Reuse the solved
    # equilibrium exactly; rerunning a fine monotone grid with the dense default
    # could otherwise allocate an enormous action tensor unnecessarily.
    res_restructured = result if maturity_extension == 0 else solve_sovereign_default(
        beta=float(result.metadata.get("beta", 0.953)),
        gamma=float(result.metadata.get("gamma", 2.0)),
        r_star=float(result.metadata.get("r_star", 0.01)),
        delta=delta_new,
        kappa=result.coupon_rate_kappa,
        sigma_eps=result.taste_shock_scale,
        rho_y=float(result.metadata.get("rho_y", 0.945)),
        sigma_y=float(result.metadata.get("sigma_y", 0.025)),
        theta=float(result.metadata.get("theta", 0.282)),
        d0=float(result.metadata.get("d0", -0.188)),
        d1=float(result.metadata.get("d1", 0.245)),
        y_penalty_type=str(result.metadata.get("y_penalty_type", "chatterjee_eyigungor")),
        m_tauchen=float(result.metadata.get("m_tauchen", 3.0)),
        recovery_rate=float(result.metadata.get("recovery_rate", 0.0)),
        damping=float(result.metadata.get("damping", 0.4)),
        howard_steps=int(result.metadata.get("howard_steps", 15)),
        enforce_monotonicity=bool(result.metadata.get("enforce_monotonicity", True)),
        n_b=n_b,
        b_max=float(result.b_grid[-1]),
        b_min=float(result.b_grid[0]),
        debt_grid=result.b_grid if result.metadata.get("debt_grid_supplied", False) else None,
        n_y=n_y,
        tol=float(result.metadata.get("tol", 1e-7)),
        max_iter=int(result.metadata.get("max_iter", 2000)),
    )
    if not res_restructured.converged:
        raise RuntimeError('the maturity counterfactual did not converge')

    ib_new = int(np.argmin(np.abs(res_restructured.b_grid - b_new)))
    counterfactual_spread = float(res_restructured.spread[ib_new, iy_distress])
    post_def_prob = float(res_restructured.default_prob[ib_new, iy_distress])

    spread_reduction_bps = (counterfactual_spread - base_spread) * 10000.0
    mean_y = float(np.dot(result.metadata.get("stationary_dist", np.full(n_y, 1/n_y)), result.y_grid))
    debt_relief_pct = (b_distress - b_new) / mean_y * 100.0

    # Post-restructuring recovery simulation starting from (b_new, y_distress)
    rng = np.random.default_rng(seed)
    y_idx = iy_distress
    b_idx = ib_new
    in_def = False
    theta = float(result.metadata.get("theta", 0.282))

    path_records = []
    for t in range(horizon):
        if t > 0:
            y_idx = int(rng.choice(n_y, p=res_restructured.p_trans[y_idx]))
        y_val = float(res_restructured.y_grid[y_idx])

        if in_def:
            if rng.random() < theta:
                in_def = False
                b_idx = int(np.argmin(abs(res_restructured.b_grid)))
            else:
                path_records.append({
                    "period": t,
                    "debt": 0.0,
                    "debt_gdp_pct": 0.0,
                    "output": float(res_restructured.metadata["y_default_vec"][y_idx]),
                    "consumption": float(res_restructured.metadata["y_default_vec"][y_idx]),
                    "spread_bps": float(np.nan),
                    "default_prob": 1.0,
                })
                continue

        p_d = float(res_restructured.default_prob[b_idx, y_idx])
        if rng.random() < p_d:
            in_def = True
            path_records.append({
                "period": t,
                "debt": float(res_restructured.b_grid[b_idx]),
                "debt_gdp_pct": float(res_restructured.b_grid[b_idx]) / y_val * 100.0,
                "output": float(res_restructured.metadata["y_default_vec"][y_idx]),
                "consumption": float(res_restructured.metadata["y_default_vec"][y_idx]),
                "spread_bps": float(np.nan),
                "default_prob": p_d,
            })
            continue

        b_cur = float(res_restructured.b_grid[b_idx])
        b_next = float(res_restructured.policy_b[b_idx, y_idx])
        b_next_idx = int(np.argmin(np.abs(res_restructured.b_grid - b_next)))
        debt_probabilities = res_restructured.metadata.get('debt_choice_probabilities')
        if debt_probabilities is not None:
            b_next_idx = int(rng.choice(n_b, p=debt_probabilities[b_idx, :, y_idx]))
            b_next = float(res_restructured.b_grid[b_next_idx])
        q_val = float(res_restructured.bond_price[b_next_idx, y_idx])
        sp_bps = float(res_restructured.spread[b_next_idx, y_idx] * 10000.0)

        c_val = y_val - (delta_new + (1.0 - delta_new) * res_restructured.coupon_rate_kappa) * b_cur + q_val * (b_next - (1.0 - delta_new) * b_cur)

        path_records.append({
            "period": t,
            "debt": b_cur,
            "debt_gdp_pct": b_cur / y_val * 100.0,
            "output": y_val,
            "consumption": c_val,
            "spread_bps": sp_bps,
            "default_prob": p_d,
        })
        b_idx = b_next_idx

    transition_path = pd.DataFrame(path_records)

    metadata: dict[str, Any] = {
        "b_distress": b_distress,
        "b_new": b_new,
        "delta_orig": delta_orig,
        "delta_new": delta_new,
        "res_restructured": res_restructured,
        "income_timing": "period_zero_is_selected_distress_state",
        "debt_grid_state": float(res_restructured.b_grid[ib_new]),
        "debt_grid_rounding_error": float(res_restructured.b_grid[ib_new] - b_new),
    }

    return RestructuringResult(
        baseline_spread=base_spread,
        counterfactual_spread=counterfactual_spread,
        spread_reduction_bps=spread_reduction_bps,
        haircut=haircut,
        maturity_extension_years=maturity_extension_years,
        pre_default_prob=pre_def_prob,
        post_default_prob=post_def_prob,
        debt_gdp_relief_pct=debt_relief_pct,
        transition_path=transition_path,
        spread_compression=spread_reduction_bps,
        restructured_debt=b_new,
        initial_debt=b_distress,
        default_prob_post=post_def_prob,
        default_prob_pre=pre_def_prob,
        metadata=metadata,
    )
