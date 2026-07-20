"""Nowcasting / mixed-frequency / probabilistic-forecast toolkit.

Public API
----------
Dynamic factor models:
    kalman_dfm                    (Doz-Giannone-Reichlin two-step)
    pca_factors, bai_ng_ic        (re-exported from :mod:`puremacro.factor`)
    static_dfm_fit                (re-exported from :mod:`puremacro.factor`)

Mixed-frequency:
    mf_var                        (Mariano-Murasawa state-space MF-VAR)
    u_midas, beta_midas           (re-exported from :mod:`puremacro.midas`)

Forecast combinations:
    equal_weight, inverse_mse, bates_granger,
    rank_weight, model_confidence_set

Probabilistic-forecast scoring:
    crps_gaussian, crps_ensemble,
    log_score_gaussian, brier_score, pit_histogram
"""
from .dfm import kalman_dfm
from .mfvar import mf_var
from .combine import (
    equal_weight, inverse_mse, bates_granger,
    rank_weight, model_confidence_set,
)
from .scoring import (
    crps_gaussian, crps_ensemble,
    log_score_gaussian, brier_score, pit_histogram,
)

# Re-exports from existing modules — no behaviour change, single
# import point for the nowcasting pipeline.
from ..factor import pca_factors, bai_ng_ic, static_dfm_fit
from ..midas import u_midas, beta_midas

__all__ = [
    # DFM
    "kalman_dfm", "pca_factors", "bai_ng_ic", "static_dfm_fit",
    # Mixed-frequency
    "mf_var", "u_midas", "beta_midas",
    # Combinations
    "equal_weight", "inverse_mse", "bates_granger",
    "rank_weight", "model_confidence_set",
    # Scoring
    "crps_gaussian", "crps_ensemble",
    "log_score_gaussian", "brier_score", "pit_histogram",
]
