"""Tests for the Quandt-Andrews supF parameter-constancy test."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference.quandt_andrews import quandt_andrews_supF


RNG = np.random.default_rng(20260503)


def test_supF_no_break_size():
    """Under H0 (constant DGP), 5% rejection rate should be ≈5% over 200 reps."""
    rejections = 0
    n_rep = 200
    T = 100
    for _ in range(n_rep):
        y = RNG.standard_normal(T)
        X = np.column_stack([np.ones(T), RNG.standard_normal(T)])
        result = quandt_andrews_supF(y, X, trim=0.15)
        if result["pvalue"] < 0.05:
            rejections += 1
    rate = rejections / n_rep
    assert 0.02 <= rate <= 0.10, f"empirical size = {rate}"


def test_supF_break_power():
    """Under H1 (mid-sample mean shift of 2 SD), power ≥ 0.80."""
    rejections = 0
    n_rep = 100
    T = 100
    for _ in range(n_rep):
        eps = RNG.standard_normal(T)
        y = np.concatenate([eps[:T // 2], eps[T // 2:] + 2.0])
        X = np.ones((T, 1))
        result = quandt_andrews_supF(y, X, trim=0.15)
        if result["pvalue"] < 0.05:
            rejections += 1
    rate = rejections / n_rep
    assert rate >= 0.80, f"empirical power = {rate}"


def test_supF_returns_breakdate():
    """supF return dict has the breakdate (relative index) field populated."""
    T = 80
    eps = RNG.standard_normal(T)
    y = np.concatenate([eps[:40], eps[40:] + 1.5])
    X = np.ones((T, 1))
    result = quandt_andrews_supF(y, X, trim=0.15)
    assert "breakdate" in result
    assert 35 <= result["breakdate"] <= 45
