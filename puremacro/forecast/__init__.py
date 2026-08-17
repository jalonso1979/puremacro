"""Macroeconomic forecasting and high-dimensional penalized regression.

Contains:
- Elastic Net and Adaptive Lasso macroeconomic forecasting (Zou 2006, Hastie et al. 2015).
- Direct multi-horizon projection with BIC regularisation tuning.
"""
from puremacro.forecast.penalized import (
    PenalizedForecastResult,
    forecast_penalized,
)
from puremacro.forecast.mcs import (
    model_confidence_set,
    losses_from_forecasts,
)
from puremacro.forecast.compare import (
    diebold_mariano,
    giacomini_white,
)
from puremacro.forecast.density import (
    pit,
    pit_uniformity_test,
    crps_gaussian,
    crps_ensemble,
    klic_amisano_giacomini,
    combine_forecasts,
    berkowitz_test,
)

__all__ = [
    "PenalizedForecastResult",
    "forecast_penalized",
    "model_confidence_set",
    "losses_from_forecasts",
    "diebold_mariano",
    "giacomini_white",
    "pit",
    "pit_uniformity_test",
    "crps_gaussian",
    "crps_ensemble",
    "klic_amisano_giacomini",
    "combine_forecasts",
    "berkowitz_test",
]
