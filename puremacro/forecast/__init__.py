"""Macroeconomic forecasting and high-dimensional penalized regression.

Contains:
- Elastic Net and Adaptive Lasso macroeconomic forecasting (Zou 2006, Hastie et al. 2015).
- Direct multi-horizon projection with BIC regularisation tuning.
"""
from puremacro.forecast.penalized import (
    PenalizedForecastResult,
    forecast_penalized,
)

__all__ = [
    "PenalizedForecastResult",
    "forecast_penalized",
]
