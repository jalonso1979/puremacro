"""Fiscal rules: validation, compiled penalties and their use in the fiscal solver."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.sovereign.rules import _RULE_TOL, FiscalRule, _compile_rule, _infeasible_message

B = np.array([-0.03, -0.01, 0.0, 0.01, 0.020000000000000004, 0.03, 0.05])
Y = np.array([0.85, 0.95, 1.05, 1.2])


@pytest.mark.parametrize("kwargs, match", [
    (dict(), "debt_ceiling or deficit_limit"),
    (dict(escape_below=0.9), "debt_ceiling or deficit_limit"),
    (dict(debt_ceiling=float("nan")), "finite"),
    (dict(debt_ceiling=0.02, escape_below=float("inf")), "finite"),
    (dict(deficit_limit=float("-inf")), "finite"),
    (dict(debt_ceiling=0.02, penalty=0.0), "penalty"),
    (dict(debt_ceiling=0.02, penalty=-1.0), "penalty"),
    (dict(debt_ceiling=0.02, penalty=float("inf")), "finite"),
    (dict(debt_ceiling=0.02, infeasible="ignore"), "infeasible"),
])
def test_validation_errors(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        FiscalRule(**kwargs)


def test_rule_is_frozen() -> None:
    rule = FiscalRule(debt_ceiling=0.02)
    with pytest.raises(AttributeError):
        rule.debt_ceiling = 0.03  # type: ignore[misc]
    assert _RULE_TOL == 1e-12


def test_node_at_cap_is_compliant() -> None:
    compiled = _compile_rule(FiscalRule(debt_ceiling=0.02), B, Y, 1.0)
    assert np.all(compiled.column_penalty[4] == 0.0)
    assert np.all(np.isinf(compiled.column_penalty[5]))


def test_hard_ceiling_columns() -> None:
    compiled = _compile_rule(FiscalRule(debt_ceiling=0.02, escape_below=1.0), B, Y, 1.0)
    pen = compiled.column_penalty
    assert pen.shape == (len(B), len(Y))
    active = Y >= 1.0
    assert np.all(pen[:, ~active] == 0.0)  # escaped incomes: no rule at all
    assert np.all(np.isposinf(pen[B > 0.02 + 1e-12][:, active]))
    assert np.all(pen[B <= 0.02 + 1e-12][:, active] == 0.0)
    assert compiled.transition_penalty is None
    assert not compiled.infeasible.any()


def test_mean_output_scales_thresholds() -> None:
    compiled = _compile_rule(FiscalRule(debt_ceiling=0.02), B, Y, 1.5)  # cap 0.03
    assert np.all(compiled.column_penalty[5] == 0.0)
    assert np.all(np.isinf(compiled.column_penalty[6]))


def test_soft_penalties_add_up() -> None:
    rule = FiscalRule(debt_ceiling=0.02, deficit_limit=0.01, penalty=0.05)
    compiled = _compile_rule(rule, B, Y, 1.0)
    m_opt = np.zeros((len(B), len(Y)), dtype=int)
    m_opt[1, :] = 6          # b' = 0.05 from b = -0.01: above cap and a deficit of 0.06
    m_opt[3, :] = 3          # b' = b = 0.01: compliant
    m_opt[5, :] = 5          # b' = b = 0.03: above cap, no deficit
    chosen = compiled.chosen_penalty(m_opt)
    assert np.allclose(chosen[1], 2 * 0.05) and np.all(chosen[3] == 0.0) and np.allclose(chosen[5], 0.05)
    assert np.all(np.isfinite(compiled.column_penalty))
    assert not compiled.infeasible.any()


def test_transition_penalty_blocks() -> None:
    rule = FiscalRule(deficit_limit=0.01, escape_below=1.0)
    compiled = _compile_rule(rule, B, Y, 1.0)
    assert compiled.column_penalty is None
    for j, y in enumerate(Y):
        full = np.vstack([compiled.transition_penalty(f, min(f + 3, len(B)), j) for f in range(0, len(B), 3)])
        assert full.shape == (len(B), len(B))
        if y < 1.0:
            assert np.all(full == 0.0)
        else:
            violation = (B[None, :] - B[:, None]) > 0.01 + _RULE_TOL
            assert np.array_equal(np.isposinf(full), violation)
            assert np.all(full[~violation] == 0.0)


def test_static_infeasibility() -> None:
    compiled = _compile_rule(FiscalRule(deficit_limit=-0.01, escape_below=1.0), B, Y, 1.0)
    active = Y >= 1.0
    expected_rows = B < B[0] + 0.01 - _RULE_TOL
    assert np.array_equal(compiled.infeasible[:, active], np.repeat(expected_rows[:, None], active.sum(), 1))
    assert not compiled.infeasible[:, ~active].any()
    for limit in (0.0, 0.01):
        assert not _compile_rule(FiscalRule(deficit_limit=limit), B, Y, 1.0).infeasible.any()
    # Soft rules never make a state infeasible.
    assert not _compile_rule(FiscalRule(deficit_limit=-0.01, penalty=1.0), B, Y, 1.0).infeasible.any()
    # A ceiling below the lowest node is infeasible everywhere it applies.
    assert _compile_rule(FiscalRule(debt_ceiling=-0.05), B, Y, 1.0).infeasible.all()


def test_infeasible_message_names_states() -> None:
    compiled = _compile_rule(FiscalRule(deficit_limit=-0.01), B, Y, 1.0)
    message = _infeasible_message(compiled, B, Y)
    count = int(compiled.infeasible.sum())
    assert f"{count} " in message and "b=-0.03" in message
    big = _compile_rule(FiscalRule(debt_ceiling=-0.05), B, Y, 1.0)
    assert _infeasible_message(big, B, Y).count("(b=") == 10
