"""Tests for DICE Integrated Climate-Macro model."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.climate.dice import DICEResult, simulate_dice_model


def test_simulate_dice_model_basic():
    res = simulate_dice_model(n_periods=20, time_step_years=5, start_year=2020)
    
    assert isinstance(res, DICEResult)
    assert len(res.trajectories) == 20
    assert "temperature_anomaly" in res.trajectories.columns
    assert "social_cost_of_carbon" in res.trajectories.columns
    assert "climate_damages" in res.trajectories.columns
    
    # Assert warming is positive and within historical / projection bounds (1.0°C to 5.0°C)
    assert 1.0 <= res.peak_temperature <= 6.0
    assert res.scc_initial > 0.0
    assert 0.0 <= res.end_century_damages <= 0.20
    
    summary_text = res.summary()
    assert "DICE-2016R forward simulation" in summary_text
    assert "Peak Global Surface Warming" in summary_text
