"""Macroeconomic nowcasting, dynamic factor models, and mixed-frequency VARs.

Contains:
- Dynamic Factor Models with Kalman filter and smoother (kalman_dfm, nowcast_gdp).
- Mixed-frequency VAR with state-space representation (mf_var).
- Forecast combination helpers (equal_weight, inverse_mse, bates_granger, rank_weight, model_confidence_set).
- Probabilistic scoring rules (crps_gaussian, crps_ensemble, log_score_gaussian, brier_score, pit_histogram).
"""
from puremacro.nowcast.combine import (
    equal_weight,
    inverse_mse,
    bates_granger,
    rank_weight,
    model_confidence_set,
)
from puremacro.nowcast.scoring import (
    crps_gaussian,
    crps_ensemble,
    log_score_gaussian,
    brier_score,
    pit_histogram,
)
from puremacro.nowcast.dfm import kalman_dfm
from puremacro.nowcast.mfvar import mf_var
from puremacro.nowcast.dfm_nowcast import NowcastResult, nowcast_gdp

__all__ = [
    "equal_weight",
    "inverse_mse",
    "bates_granger",
    "rank_weight",
    "model_confidence_set",
    "crps_gaussian",
    "crps_ensemble",
    "log_score_gaussian",
    "brier_score",
    "pit_histogram",
    "kalman_dfm",
    "mf_var",
    "NowcastResult",
    "nowcast_gdp",
]
