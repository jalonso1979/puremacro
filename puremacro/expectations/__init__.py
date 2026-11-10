"""Explicit dated information sets for linear Gaussian forecasts."""
from .gaussian import (
    GaussianSignal, GaussianInformation, build_gaussian_information, audit_gaussian_information,
)

__all__ = ["GaussianSignal", "GaussianInformation", "build_gaussian_information", "audit_gaussian_information"]
