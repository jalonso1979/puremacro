"""Tests for Sequence-Space HANK solver (Auclert et al. 2021)."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.hank_sequence_space import (
    SequenceSpaceHANKResult,
    solve_hank_sequence_space,
)


def test_solve_hank_sequence_space_basic():
    res = solve_hank_sequence_space(T=30, n_a=30)
    
    assert isinstance(res, SequenceSpaceHANKResult)
    assert len(res.irf_output) == 30
    assert len(res.irf_consumption) == 30
    assert len(res.irf_inflation) == 30
    assert len(res.irf_rate) == 30
    assert res.jacobian_c_r.shape == (30, 30)
    assert res.jacobian_c_y.shape == (30, 30)
    
    # Check that MPC is positive and reasonable (e.g. 0.05 to 0.6)
    assert 0.01 <= res.steady_state_mpc <= 0.8
    assert len(res.mpc_distribution) == 10
    
    # Check monetary contraction leads to negative output response initially
    assert res.irf_output[0] < 0.0
    assert res.irf_consumption[0] < 0.0
    
    summary_text = res.summary()
    assert "Sequence-Space HANK" in summary_text
    assert "MPC by Wealth Decile" in summary_text
