"""Structural models for puremacro.

Currently:
  - dmp_regime_dependent: DMP search-and-matching with a regime-dependent
    vacancy-cost multiplier (precautionary) and a regime-dependent
    discount factor (reaction-function expectation).
"""
from puremacro.models.dmp_regime_dependent import DMPParameters, DMPState, dmp_steady_state, dmp_irf
from puremacro.models.hank_sequence_space import (
    SequenceSpaceHANKResult,
    FakeNewsResult,
    FiscalTransferResult,
    NonlinearHANKResult,
    solve_hank_sequence_space,
    fake_news_algorithm,
    simulate_targeted_transfer,
    solve_nonlinear_transition,
)

__all__ = [
    "DMPParameters", "DMPState", "dmp_steady_state", "dmp_irf",
    "SequenceSpaceHANKResult", "FakeNewsResult", "FiscalTransferResult",
    "NonlinearHANKResult",
    "solve_hank_sequence_space", "fake_news_algorithm", "simulate_targeted_transfer",
    "solve_nonlinear_transition",
]
