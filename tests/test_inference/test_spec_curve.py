"""Tests for the specification-curve module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.inference.spec_curve import (
    enumerate_specs,
    run_spec_curve,
    bootstrap_pvalue_median,
)


def test_enumerate_specs_cartesian():
    grid = {
        "id_method": ["A", "B"],
        "ls_def": ["raw", "gollin"],
        "denom": ["pc", "pc_nds"],
    }
    specs = enumerate_specs(grid)
    assert len(specs) == 8
    keys = {tuple(sorted(s.items())) for s in specs}
    assert len(keys) == 8


def test_run_spec_curve_dummy_estimator():
    grid = {"id_method": ["A", "B", "C"]}
    specs = enumerate_specs(grid)

    def est(_data, spec):
        return {"sigma_hat": {"A": 0.9, "B": 1.0, "C": 1.1}[spec["id_method"]],
                "se": 0.05}

    out = run_spec_curve(data=None, specs=specs, estimator=est)
    assert set(out.columns) >= {"id_method", "sigma_hat", "se", "ci_lo", "ci_hi"}
    assert sorted(out["sigma_hat"].tolist()) == [0.9, 1.0, 1.1]


def test_bootstrap_pvalue_size():
    rng = np.random.default_rng(20260503)
    rejections = 0
    n_rep = 100
    for _ in range(n_rep):
        curve = pd.DataFrame({"sigma_hat": rng.normal(1.25, 0.05, size=50)})
        p = bootstrap_pvalue_median(curve, h0=1.25, B=400, rng=rng)
        if p < 0.05:
            rejections += 1
    rate = rejections / n_rep
    assert rate <= 0.15, f"empirical size = {rate} (expect ≤ 0.15)"


def test_bootstrap_pvalue_power():
    rng = np.random.default_rng(20260503)
    rejections = 0
    n_rep = 50
    for _ in range(n_rep):
        curve = pd.DataFrame({"sigma_hat": rng.normal(1.0, 0.05, size=50)})
        p = bootstrap_pvalue_median(curve, h0=1.25, B=400, rng=rng)
        if p < 0.05:
            rejections += 1
    rate = rejections / n_rep
    assert rate >= 0.90, f"empirical power = {rate}"
