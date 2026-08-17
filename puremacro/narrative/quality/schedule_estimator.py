"""Implementation profile and realization lag estimator for narrative policy shocks.

Estimates realistic quarterly realization schedules based on empirical policy lag profiles:
- Infrastructure / capital expenditure: multi-quarter construction S-curve
- Transfer payments / direct stimulus: front-loaded disbursement
- General tax reform: uniform annual rollout
- Monetary / FX decisions: immediate realization
"""
from __future__ import annotations

import pandas as pd


def estimate_implementation_profile(
    announcement_date: str | pd.Timestamp,
    kind: str = "fiscal",
    target: str = "both",
    subtarget: str | None = None,
    *,
    horizon_quarters: int = 8,
) -> list[tuple[pd.Timestamp, float]]:
    """Estimate a realistic quarterly implementation schedule for a policy announcement.

    Parameters
    ----------
    announcement_date : date of the announcement.
    kind : ``"fiscal"``, ``"monetary"``, ``"macropru"``, ``"fx"``, ``"structural"``.
    target : policy target (``"investment"``, ``"consumption"``, ``"policy_rate"``, etc.).
    subtarget : specific subtype (``"infra"``, ``"defense"``, ``"transfers"``, ``"tax"``, etc.).
    horizon_quarters : maximum quarters to distribute over.

    Returns
    -------
    list of (quarter_start_timestamp, weight) tuples summing to 1.0.
    """
    dt = pd.Timestamp(announcement_date)
    q0 = pd.Period(dt, freq="Q").to_timestamp()

    # 1. Monetary and FX actions take effect immediately
    if kind in ("monetary", "fx"):
        return [(dt, 1.0)]

    # 2. Fiscal policy profiles based on target/subtarget
    sub = (subtarget or "").lower()
    tgt = target.lower()

    if "infra" in sub or tgt == "investment":
        # Multi-quarter S-curve rollout for public investment / infrastructure projects
        # Lags: Q0 (permitting/design 10%), Q1 (groundbreaking 25%), Q2 (peak construction 35%),
        # Q3 (finishing 20%), Q4 (commissioning 10%)
        weights = [0.10, 0.25, 0.35, 0.20, 0.10]
    elif "transfer" in sub or "aid" in sub or "relief" in sub or (tgt == "consumption" and "tax" not in sub):
        # Frontloaded direct transfers / subsidies / cash relief
        weights = [0.75, 0.25]
    elif "defense" in sub or "procurement" in sub:
        # Multi-year defense procurement contracts
        weights = [0.15, 0.30, 0.30, 0.25]
    elif "tax" in sub:
        # Tax changes usually take effect evenly across the fiscal year
        weights = [0.25, 0.25, 0.25, 0.25]
    elif kind == "macropru":
        # Macroprudential phase-in (e.g. capital buffers, reserve requirements)
        weights = [0.20, 0.40, 0.40]
    elif kind == "structural":
        # Structural reforms with gradual phase-in over 4 quarters
        weights = [0.10, 0.30, 0.30, 0.30]
    else:
        # Default 2-quarter general fiscal action
        weights = [0.60, 0.40]

    # Generate quarterly dates
    profile: list[tuple[pd.Timestamp, float]] = []
    base_period = pd.Period(dt, freq="Q")

    # Normalize weights so they sum strictly to 1.0
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    slack = pd.Timedelta(days=7)
    for idx, w in enumerate(norm_weights):
        q_date = (base_period + idx).to_timestamp()
        # If Q0 predates dt by more than slack, use dt for Q0
        if idx == 0 and q_date < dt - slack:
            q_date = dt
        profile.append((q_date, float(w)))

    return profile


__all__ = ["estimate_implementation_profile"]
