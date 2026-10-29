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


# --------------------------------------------------------------------------- rules in the solver
import dataclasses  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from puremacro.sovereign import solve_fiscal_sovereign_default  # noqa: E402
from puremacro.sovereign._fiscal_paths import simulate_paths  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sovereign_fiscal" / "illustrative_101x11.json"
SMALL = dict(tau=0.2, psi=1.0, n_b=61, n_y=7)
ARRAYS = ("v_repay", "v_default", "v_expected", "bond_price", "default_prob", "policy_index", "policy_b")


def _assert_exactly_equal(a, b):
    for name in ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.iterations == b.iterations


def test_rule_none_matches_phase_one_fixture() -> None:
    frozen = json.loads(FIXTURE.read_text())
    res = solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=101, n_y=11, rule=None)
    assert res.iterations == frozen["iterations"]
    assert np.array_equal(res.policy_index, np.array(frozen["policy_index"]))
    for name in ("bond_price", "v_repay", "v_default"):
        assert np.max(np.abs(getattr(res, name) - np.array(frozen[name]))) <= 1e-10, name
    assert res.rule is None and res.metadata["rule"] is None and res.metadata["rule_infeasible_states"] == 0


def test_nonbinding_ceiling_equals_no_rule() -> None:
    _assert_exactly_equal(solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(debt_ceiling=1.0)),
                          solve_fiscal_sovereign_default(**SMALL))


def test_hard_ceiling_monotone_equals_dense_masked() -> None:
    rule = FiscalRule(debt_ceiling=0.02)
    fast = solve_fiscal_sovereign_default(**SMALL, rule=rule)
    dense = solve_fiscal_sovereign_default(**SMALL, rule=rule, policy_search="dense")
    assert np.array_equal(fast.policy_index, dense.policy_index)
    for name in ("v_repay", "v_default", "bond_price", "default_prob"):
        assert np.max(np.abs(getattr(fast, name) - getattr(dense, name))) <= 1e-12, name
    repay = fast.default_prob < 0.5
    assert np.all(fast.policy_b[repay] <= 0.02 + 1e-12)
    assert fast.metadata["monotone_certification_gap"] <= 1e-11


def test_escape_everywhere_equals_no_rule() -> None:
    _assert_exactly_equal(solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(debt_ceiling=0.02, escape_below=10.0)),
                          solve_fiscal_sovereign_default(**SMALL))


def test_escape_nowhere_equals_plain_ceiling() -> None:
    _assert_exactly_equal(solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(debt_ceiling=0.02, escape_below=0.0)),
                          solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(debt_ceiling=0.02)))


def test_large_penalty_equals_hard_ceiling() -> None:
    _assert_exactly_equal(solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(debt_ceiling=0.02, penalty=1e6)),
                          solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(debt_ceiling=0.02)))


def test_lenders_price_the_rule() -> None:
    free = solve_fiscal_sovereign_default(**SMALL)
    ruled = solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(debt_ceiling=0.02))
    below = free.b_grid <= 0.02
    assert np.max(np.abs(ruled.bond_price[below] - free.bond_price[below])) > 1e-6


@pytest.mark.parametrize("penalty", [None, 0.05])
def test_deficit_exhaustive_matches_brute_force(penalty) -> None:
    rule = FiscalRule(deficit_limit=0.002, penalty=penalty)
    kwargs = dict(tau=0.2, psi=1.0, n_b=21, n_y=5, rule=rule)
    exhaustive = solve_fiscal_sovereign_default(**kwargs, policy_search="exhaustive")
    dense = solve_fiscal_sovereign_default(**kwargs, policy_search="dense")
    assert np.array_equal(exhaustive.policy_index, dense.policy_index)
    for name in ("v_repay", "v_default", "bond_price", "default_prob"):
        assert np.max(np.abs(getattr(exhaustive, name) - getattr(dense, name))) <= 1e-12, name
    assert exhaustive.metadata["policy_search"] == "exhaustive"
    assert exhaustive.metadata["monotone_certification_gap"] is None
    if penalty is None:
        repay = exhaustive.default_prob < 0.5
        step = exhaustive.policy_b - exhaustive.b_grid[:, None]
        assert np.all(step[repay] <= 0.002 + 1e-12)


def test_deficit_rule_rejects_monotone() -> None:
    with pytest.raises(ValueError, match="exhaustive"):
        solve_fiscal_sovereign_default(**SMALL, rule=FiscalRule(deficit_limit=0.005))


def test_infeasible_rule_raises_naming_states() -> None:
    with pytest.raises(ValueError, match=r"b=-0\.03") as info:
        solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=21, n_y=5, policy_search="exhaustive",
                                       rule=FiscalRule(deficit_limit=-0.01))
    assert " states" in str(info.value)


def _forced_default_result():
    rule = FiscalRule(deficit_limit=-0.01, infeasible="default")
    return solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=21, n_y=5, policy_search="exhaustive", rule=rule)


def test_infeasible_default_mode_forces_default() -> None:
    res = _forced_default_result()
    forced = res.b_grid < res.b_grid[0] + 0.01 - 1e-12
    assert forced.any()
    assert res.converged and np.isfinite(res.residual)
    assert res.metadata["rule_infeasible_states"] == int(forced.sum()) * len(res.y_grid)
    assert np.all(res.default_prob[forced] == 1.0)
    assert np.all(np.isneginf(res.v_repay[forced]))
    assert np.all(np.isfinite(res.v_repay[~forced]))
    assert np.all(res.v_expected[forced] == res.v_default[None, :])


@pytest.mark.parametrize("sigma_eps", [0.0, 0.01])
def test_infeasible_default_mode_with_taste_shocks(sigma_eps) -> None:
    rule = FiscalRule(deficit_limit=-0.01, infeasible="default")
    res = solve_fiscal_sovereign_default(tau=0.2, psi=1.0, n_b=21, n_y=5, policy_search="dense", rule=rule,
                                         sigma_eps=sigma_eps)
    forced = res.b_grid < res.b_grid[0] + 0.01 - 1e-12
    assert res.converged and np.all(res.default_prob[forced] == 1.0)
    assert not np.isnan(res.v_expected).any() and not np.isnan(res.bond_price).any()


def test_paths_never_repay_in_forced_default_states() -> None:
    res = _forced_default_result()
    forced = np.zeros(res.default_prob.shape, dtype=bool)
    forced[res.b_grid < res.b_grid[0] + 0.01 - 1e-12] = True
    uniforms = np.random.default_rng(5).random((3, 4, 2000))
    paths = simulate_paths(res, 0, len(res.y_grid) // 2, uniforms)
    assert paths.default_entry[:, 0].all()
    assert not (paths.repaid & forced[paths.b_index, paths.y_index]).any()


def test_rule_metadata_and_field() -> None:
    rule = FiscalRule(debt_ceiling=0.02, escape_below=0.92, penalty=0.05)
    res = solve_fiscal_sovereign_default(**SMALL, rule=rule)
    assert res.rule is rule
    assert res.metadata["rule"] == dataclasses.asdict(rule)
    assert json.loads(json.dumps(res.metadata["rule"])) == dataclasses.asdict(rule)
    assert res.metadata["rule_infeasible_states"] == 0


def test_monotone_certification_ignores_forced_default_states() -> None:
    # A ceiling below the lowest node, active only at y >= 1: those states default for sure.
    rule = FiscalRule(debt_ceiling=-0.05, escape_below=1.0, infeasible="default")
    fast = solve_fiscal_sovereign_default(**SMALL, rule=rule)
    dense = solve_fiscal_sovereign_default(**SMALL, rule=rule, policy_search="dense")
    forced = np.broadcast_to(fast.y_grid[None, :] >= 1.0, fast.default_prob.shape)
    assert fast.converged and np.isfinite(fast.residual)
    assert np.isfinite(fast.metadata["monotone_certification_gap"])
    assert np.isfinite(fast.metadata["exhaustive_bellman_residual"])
    assert np.all(fast.default_prob[forced] == 1.0) and np.all(np.isneginf(fast.v_repay[forced]))
    assert np.array_equal(fast.policy_index[~forced], dense.policy_index[~forced])
    assert np.max(np.abs(fast.bond_price - dense.bond_price)) <= 1e-12
