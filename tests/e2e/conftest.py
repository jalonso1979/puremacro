"""Pytest fixtures and synthetic data generators for the puremacro E2E test suite.

Provides standardized, reproducible macroeconomic datasets and model configurations
for testing R1 (Narrative SVAR), R2 (Honest DiD), R3 (Smooth LP), R4 (Non-linear HANK),
R5 (Gertler-Karadi DSGE), and R6 (BVAR-SV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def macro_3var_series() -> tuple[pd.DataFrame, np.ndarray, int]:
    """Simulate a canonical 3-variable macro system (Output, Inflation, Interest Rate).
    
    Planted-truth DGP matching Antolín-Díaz & Rubio-Ramírez (2018):
    Column 0 is the structural monetary contraction shock with sign pattern (+1, +1, -1),
    driven by i.i.d. N(0, 1) innovations with a planted shock eps[t*, 0] = 5.0.
    """
    rng = np.random.default_rng(42)
    T = 250
    t_star = 150
    
    A_TRUE = np.array([
        [0.50, 0.10, 0.00],
        [0.00, 0.40, 0.10],
        [0.10, 0.00, 0.45]
    ])
    B0_TRUE = np.array([
        [ 1.0,  0.0,  0.3],
        [ 0.6,  1.0, -0.2],
        [-0.4,  0.3,  1.0]
    ])
    
    eps = rng.standard_normal((T, 3))
    eps[t_star, 0] = 5.0
    
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A_TRUE @ Y[t - 1] + B0_TRUE @ eps[t]
        
    dates = pd.date_range("1975-01-01", periods=T, freq="QS")
    df = pd.DataFrame(Y, index=dates, columns=["output", "inflation", "interest_rate"])
    return df, eps, t_star


@pytest.fixture(scope="session")
def event_study_did_data() -> dict:
    """Synthetic event study estimates replicating Rambachan & Roth (2023) benchmark.
    
    Pre-periods: -3, -2, -1 (base period = -1)
    Post-periods: 0, 1, 2, 3
    Pre-trend deviations with max absolute deviation = 0.20
    Post-treatment effect = 1.50, declining to 1.10
    """
    event_time = [-3, -2, -1, 0, 1, 2, 3]
    beta = np.array([0.15, -0.20, 0.0, 1.50, 1.40, 1.25, 1.10])
    se = np.array([0.10, 0.12, 0.0, 0.15, 0.18, 0.20, 0.22])
    
    n_periods = len(beta)
    cov = np.zeros((n_periods, n_periods))
    for i in range(n_periods):
        for j in range(n_periods):
            cov[i, j] = se[i] * se[j] * (0.4 ** abs(i - j))
            
    return {
        "event_time": event_time,
        "beta": beta,
        "se": se,
        "cov": cov,
        "base_period": -1,
        "pre_periods": [-3, -2, -1],
        "post_periods": [0, 1, 2, 3],
        "pre_trend_max": 0.20,
    }


@pytest.fixture(scope="session")
def staggered_panel_did_df() -> pd.DataFrame:
    """Staggered difference-in-differences panel dataset with 20 units over 10 periods."""
    rng = np.random.default_rng(42)
    n_units = 20
    n_periods = 10
    rows = []
    
    for u in range(n_units):
        unit_id = f"unit_{u}"
        if u < 7:
            treat_time = 4.0
        elif u < 14:
            treat_time = 7.0
        else:
            treat_time = np.nan
            
        unit_fe = rng.normal(0, 1)
        for t in range(n_periods):
            time_fe = 0.2 * t
            is_treated = 1.0 if (not np.isnan(treat_time) and t >= treat_time) else 0.0
            treat_effect = 2.5 * is_treated
            y = unit_fe + time_fe + treat_effect + rng.normal(0, 0.5)
            rows.append({
                "unit": unit_id,
                "time": t,
                "outcome": y,
                "treat_time": treat_time,
            })
            
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def smooth_lp_ar2_data() -> pd.DataFrame:
    """Simulate an AR(2) process with a monetary shock for testing Smooth Local Projections.
    
    DGP: y_t = 0.7 y_{t-1} - 0.2 y_{t-2} + 0.5 x_{t-1} + u_t
    x_t is an exogenous policy instrument.
    """
    rng = np.random.default_rng(101)
    T = 250
    x = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = 0.7 * y[t - 1] - 0.2 * y[t - 2] + 0.5 * x[t - 1] + rng.standard_normal() * 0.4
        
    dates = pd.date_range("1980-01-01", periods=T, freq="MS")
    return pd.DataFrame({"y": y, "x": x}, index=dates)


@pytest.fixture(scope="session")
def canonical_gk_params() -> dict:
    """Gertler & Karadi (2011) Table 1 canonical calibration parameters."""
    return {
        "beta": 0.99,            # Discount factor
        "sigma": 1.0,            # Intertemporal elasticity of substitution
        "varphi": 0.276,         # Inverse Frisch elasticity
        "theta_b": 0.972,        # Banker survival probability
        "lambda_b": 0.381,       # Fraction of divertable assets
        "omega_b": 0.002,        # Starting wealth transfer for new bankers
        "rho_xi": 0.66,          # Capital quality shock persistence
        "sigma_xi": 0.05,        # Capital quality shock standard deviation
        "target_leverage": 4.0,  # Steady-state leverage ratio
        "target_spread_bps": 100 # Steady-state credit spread (bps annualized)
    }
