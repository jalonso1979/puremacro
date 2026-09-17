"""Macroeconomic nowcasting, dynamic factor models, and mixed-frequency VARs.

Contains:
- Dynamic Factor Models with Kalman filter and RTS smoother (:class:`DynamicFactorModel`, :func:`kalman_dfm`, :func:`nowcast_gdp`).
- Analytical Bańbura & Modugno (2014) news decomposition (:func:`banbura_modugno_news`).
- High-level Latin America real-time nowcast orchestrator (:func:`realtime_nowcast`).
- Mixed-frequency VAR with state-space representation (:func:`mf_var`).
- Professional forecast evaluation suite (:func:`pit_uniformity_test`, :func:`fan_chart`).
- Probabilistic scoring rules (:func:`crps_gaussian`, :func:`crps_ensemble`, :func:`log_score_gaussian`, :func:`brier_score`, :func:`pit_histogram`).
- Forecast combination helpers (:func:`equal_weight`, :func:`inverse_mse`, :func:`bates_granger`, :func:`rank_weight`, :func:`model_confidence_set`).
"""
from __future__ import annotations

from puremacro.nowcast.combine import (
    bates_granger,
    equal_weight,
    inverse_mse,
    model_confidence_set,
    rank_weight,
)
from puremacro.nowcast.dfm import (
    DynamicFactorModel,
    DynamicFactorModelResult,
    KalmanDFMResult,
    kalman_dfm,
)
from puremacro.nowcast.dfm_nowcast import NowcastResult, nowcast_gdp
from puremacro.nowcast.evaluation import (
    FanChartResult,
    PITUniformityResult,
    fan_chart,
    pit_uniformity_test,
)
from puremacro.nowcast.mfvar import mf_var
from puremacro.nowcast.news import (
    NewsDecompositionResult,
    banbura_modugno_news,
)
from puremacro.nowcast.realtime_nowcast import (
    RealtimeNowcastResult,
    realtime_nowcast,
)
from puremacro.nowcast.scoring import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    log_score_gaussian,
    pit_histogram,
)

__all__ = [
    # Factor models & nowcasting
    "DynamicFactorModel",
    "DynamicFactorModelResult",
    "KalmanDFMResult",
    "kalman_dfm",
    "NowcastResult",
    "nowcast_gdp",
    "mf_var",
    # Real-time orchestrator & News decomposition
    "realtime_nowcast",
    "RealtimeNowcastResult",
    "banbura_modugno_news",
    "NewsDecompositionResult",
    # Evaluation suite & Fan charts
    "pit_uniformity_test",
    "PITUniformityResult",
    "fan_chart",
    "FanChartResult",
    # Probabilistic scoring
    "crps_gaussian",
    "crps_ensemble",
    "log_score_gaussian",
    "brier_score",
    "pit_histogram",
    # Forecast combination
    "equal_weight",
    "inverse_mse",
    "bates_granger",
    "rank_weight",
    "model_confidence_set",
]
