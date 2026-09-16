"""Unit tests for Multi-Constraint OccBin ($M \\ge 2$).

Verifies:
1. Direct export of `solve_multiconstraint_occbin` from `puremacro.dsge`.
2. Polymorphic dispatch from `solve_occbin` with Mapping and Sequence of
   models/constraints, bit-for-bit equal to the direct call, and M = 1 parity
   with the single-constraint solver.
3. 2 simultaneous occasionally binding constraints (ZLB + borrowing limit),
   with the economics checked: r sits at the floor while the ZLB binds, b at
   the cap while the cap binds, both bounds hold along the whole path and
   every regime bit agrees with the shadow value that drives it.
4. Every row in which a constrained model differs from the reference is
   honoured (3.4.0 regression: a model that pegs r AND rewrites another
   equation used to lose the second row, or the peg itself).
5. A path on which no constraint binds equals the linear reference path.
6. Non-convergence (max_iter exhausted, a regime cycle, a violated bound, a
   constraint binding at T) is reported with `converged == False` AND a
   UserWarning naming the reason.
7. Constraint definitions must pair one-to-one with the constrained models.
8. The vacuous 'pegged value vs its own bound' relax test that solve_occbin
   refuses is refused by the multi-constraint solver too.
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest

from puremacro.dsge import (
    build_dynare,
    LinearModel,
    OccBinConstraint,
    OccBinResult,
    solve_occbin,
    solve_multiconstraint_occbin,
)


PARAMS = {
    "beta": 0.99,
    "sigma": 1.0,
    "kappa": 0.15,
    "phi_pi": 1.5,
    "phi_y": 0.25,
    "rho_r": 0.6,
    "rho_b": 0.5,
    "rho_g": 0.7,
    "gamma_y": 0.2,
    "chi": 0.1,
    "r_ss": 0.015,
    "b_bar": 0.02,
}
VARIABLES = ["y", "pi", "r", "b", "g"]
SHOCKS = ["eps_g", "eps_r", "eps_b"]
STEADY_STATE = {v: 0.0 for v in VARIABLES}


# Regime 0: unconstrained
def ref_eqs(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
        curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]


# Regime 1: ZLB (row 2 pegs r)
def zlb_eqs(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (-p.r_ss),
        curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]


# Regime 2: Borrowing cap (row 3 pegs b)
def borr_eqs(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
        curr.b - p.b_bar,
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]


# ZLB regime that ALSO rewrites the borrowing equation (rows 2 and 3 differ;
# the peg is the first differing row).
def zlb_two_row_eqs(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (-p.r_ss),
        curr.b - (p.rho_b * lag.b + 3.0 * p.gamma_y * curr.y + shocks_v.eps_b),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]


# ZLB regime that ALSO rewrites the IS curve (rows 0 and 2 differ and both
# contain r; the peg is NOT the first differing row).
def zlb_is_curve_eqs(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + 2.0 * p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (-p.r_ss),
        curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]


def _build_reference() -> LinearModel:
    return build_dynare(ref_eqs, variables=VARIABLES, shocks=SHOCKS, params=PARAMS, steady_state=STEADY_STATE)


def _build_constrained(eqs) -> LinearModel:
    return build_dynare(
        eqs, variables=VARIABLES, shocks=SHOCKS, params=PARAMS, steady_state=STEADY_STATE,
        check_steady_state=False, strict=False,
    )


def _joint_shock(horizon: int = 40, eps_g: float = -0.06, eps_b: float = 0.05) -> np.ndarray:
    """Demand contraction (triggers the ZLB) plus a credit boom (triggers the cap) at t = 1."""
    shocks = np.zeros((horizon, 3))
    shocks[0, 0] = eps_g
    shocks[0, 2] = eps_b
    return shocks


@pytest.fixture
def dual_constraint_models():
    """Setup a canonical New Keynesian model with dual constraints:
    - Constraint 1 (ZLB): r_t >= -r_ss (r_t pegged to -r_ss)
    - Constraint 2 (Borrowing cap): b_t <= b_bar (b_t pegged to b_bar)
    """
    m_ref = _build_reference()
    m_zlb = _build_constrained(zlb_eqs)
    m_borr = _build_constrained(borr_eqs)

    c_zlb = OccBinConstraint(variable="r", threshold=-PARAMS["r_ss"], operator="<")
    c_borr = OccBinConstraint(variable="b", threshold=PARAMS["b_bar"], operator=">")

    return m_ref, m_zlb, m_borr, c_zlb, c_borr


def _assert_same_solution(res_a: OccBinResult, res_b: OccBinResult, atol: float = 1e-12) -> None:
    assert res_a.regimes == res_b.regimes
    assert res_a.converged == res_b.converged
    np.testing.assert_allclose(res_a.simulated_path.values, res_b.simulated_path.values, atol=atol, rtol=0.0)
    for col in res_a.shadow_path.columns:
        np.testing.assert_allclose(
            res_a.shadow_path[col].values, res_b.shadow_path[col].values, atol=atol, rtol=0.0
        )


def test_public_export_multiconstraint():
    """Verify solve_multiconstraint_occbin is exported from puremacro.dsge."""
    import puremacro.dsge as dsge
    assert hasattr(dsge, "solve_multiconstraint_occbin")
    assert "solve_multiconstraint_occbin" in dsge.__all__
    assert callable(dsge.solve_multiconstraint_occbin)


def test_dual_constraint_solving(dual_constraint_models):
    """Two simultaneous constraints (ZLB + borrowing cap): regimes AND economics."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models
    r_floor, b_cap = -PARAMS["r_ss"], PARAMS["b_bar"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a verified solution must not warn
        res = solve_multiconstraint_occbin(
            m_unconstrained=m_ref,
            m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
            shock_seq=_joint_shock(),
            constraints={"zlb": c_zlb, "borrowing": c_borr},
            horizon=40,
        )

    assert isinstance(res, OccBinResult)
    assert res.converged
    # Verify both constraints bind at least once, jointly at impact, and the
    # terminal period is slack
    regimes = np.asarray(res.regimes)
    assert np.any((regimes & 1) == 1), "ZLB constraint should bind in some period"
    assert np.any((regimes & 2) == 2), "Borrowing cap constraint should bind in some period"
    assert regimes[0] == 3, f"both constraints should bind at impact, got regime {regimes[0]}"
    assert regimes[-1] == 0
    assert res.binding_periods == int(np.sum(regimes > 0))

    # Economics: the pegged variable sits AT its bound while its constraint
    # binds, the notional value lies beyond the bound there, and the bound
    # holds along the whole path.
    r = res.simulated_path["r"].values
    b = res.simulated_path["b"].values
    zlb_t = np.flatnonzero(regimes & 1)
    cap_t = np.flatnonzero(regimes & 2)
    np.testing.assert_allclose(r[zlb_t], r_floor, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(b[cap_t], b_cap, atol=1e-12, rtol=0.0)
    assert np.all(res.shadow_path["r_shadow"].values[zlb_t] < r_floor)
    assert np.all(res.shadow_path["b_shadow"].values[cap_t] > b_cap)
    assert r.min() >= r_floor - 1e-12
    assert b.max() <= b_cap + 1e-12
    # Where a constraint is slack the variable is strictly inside its bound
    # (the linear equation holds and the binding test rejected it).
    assert np.all(r[(regimes & 1) == 0] > r_floor)
    assert np.all(b[(regimes & 2) == 0] < b_cap)


def test_regime_bits_agree_with_shadow_values(dual_constraint_models):
    """Each regime bit is the fixed point of the Guerrieri-Iacoviello binding test.

    Bit k is set exactly in the periods where constraint k's shadow (notional)
    value violates the threshold; where the bit is clear the simulated value
    respects the bound. This is what `converged=True` certifies.
    """
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models
    res = solve_multiconstraint_occbin(
        m_ref, {"zlb": m_zlb, "borrowing": m_borr}, _joint_shock(),
        constraints={"zlb": c_zlb, "borrowing": c_borr}, horizon=40,
    )
    assert res.converged
    regimes = np.asarray(res.regimes)
    for k, (c_obj, var) in enumerate([(c_zlb, "r"), (c_borr, "b")]):
        bit = ((regimes >> k) & 1) == 1
        shadow = res.shadow_path[f"{var}_shadow"].values
        sim = res.simulated_path[var].values
        expected_binding = np.asarray(c_obj.evaluate(shadow), dtype=bool)
        np.testing.assert_array_equal(bit, expected_binding)
        assert not np.any(np.asarray(c_obj.evaluate(sim[~bit]), dtype=bool))
        # In slack periods the reference equation holds, so the shadow value
        # IS the simulated value.
        np.testing.assert_allclose(shadow[~bit], sim[~bit], atol=1e-12, rtol=0.0)


def test_terminal_slack_check(dual_constraint_models):
    """Test that if a constraint binds in the terminal period, converged is False and warning is emitted."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models

    # Large shock with very short horizon (horizon=3) so economy cannot return to steady state
    shocks = np.zeros((3, 3))
    shocks[0, 0] = -0.08

    with pytest.warns(UserWarning, match="constraint still binds at terminal period"):
        res = solve_multiconstraint_occbin(
            m_unconstrained=m_ref,
            m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
            shock_seq=shocks,
            constraints={"zlb": c_zlb, "borrowing": c_borr},
            horizon=3,
        )

    assert not res.converged
    assert res.regimes[-1] != 0


def test_polymorphic_dispatch_solve_occbin_mapping(dual_constraint_models):
    """solve_occbin with mappings of models/constraints equals the direct multi-constraint call."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models
    shocks = _joint_shock(35)

    res = solve_occbin(
        reference_model=m_ref,
        constrained_model={"zlb": m_zlb, "borr": m_borr},
        constraint={"zlb": c_zlb, "borr": c_borr},
        shock_sequence=shocks,
        horizon=35,
    )
    direct = solve_multiconstraint_occbin(
        m_ref, {"zlb": m_zlb, "borr": m_borr}, shocks,
        constraints={"zlb": c_zlb, "borr": c_borr}, horizon=35,
    )

    assert isinstance(res, OccBinResult)
    assert res.converged
    assert set(res.constraints) == {"zlb", "borr"}
    assert res.constraints["zlb"] is c_zlb and res.constraints["borr"] is c_borr
    assert 3 in res.regimes
    _assert_same_solution(res, direct)


def test_polymorphic_dispatch_solve_occbin_sequence(dual_constraint_models):
    """solve_occbin with sequences of models/constraints equals the direct multi-constraint call."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models
    shocks = _joint_shock(35)

    res = solve_occbin(
        reference_model=m_ref,
        constrained_model=[m_zlb, m_borr],
        constraint=[c_zlb, c_borr],
        shock_sequence=shocks,
        horizon=35,
    )
    direct = solve_multiconstraint_occbin(
        m_ref, [m_zlb, m_borr], shocks, constraints=[c_zlb, c_borr], horizon=35
    )

    assert isinstance(res, OccBinResult)
    assert res.converged
    assert len(res.regimes) == 35
    assert list(res.constraints) == ["constraint_0", "constraint_1"]
    assert 3 in res.regimes
    _assert_same_solution(res, direct)


def test_single_model_list_dispatch_matches_direct_call(dual_constraint_models):
    """M = 1: `solve_occbin(ref, [m], [c])` must return the numbers of `solve_occbin(ref, m, c)`."""
    m_ref, m_zlb, _, c_zlb, _ = dual_constraint_models
    shocks = np.zeros((40, 3))
    shocks[0, 0] = -0.06

    direct = solve_occbin(m_ref, m_zlb, c_zlb, shocks, horizon=40)
    via_list = solve_occbin(m_ref, [m_zlb], [c_zlb], shocks, horizon=40)
    via_mapping = solve_occbin(m_ref, {"zlb": m_zlb}, {"zlb": c_zlb}, shocks, horizon=40)
    via_single_constraint = solve_occbin(m_ref, {"zlb": m_zlb}, c_zlb, shocks, horizon=40)

    assert direct.converged and direct.binding_periods > 0
    for other in (via_list, via_mapping, via_single_constraint):
        _assert_same_solution(direct, other)


@pytest.mark.parametrize("eqs", [zlb_two_row_eqs, zlb_is_curve_eqs], ids=["peg+borrowing-row", "IS-row+peg"])
def test_two_row_constrained_model_honours_every_row(dual_constraint_models, eqs):
    """A constrained model that rewrites two equations must have BOTH rows spliced in.

    3.4.0 regression: only one row per constrained model was copied into the
    regime matrices. With the borrowing row also rewritten the multi path was
    bit-identical to the one-row ZLB model (row dropped); with the IS curve
    also rewritten the IS row was spliced and the Taylor-rule peg itself was
    dropped, so r fell below the floor in periods reported as binding.
    """
    m_ref, m_zlb_one_row, _, c_zlb, _ = dual_constraint_models
    m_two_rows = _build_constrained(eqs)
    r_floor = -PARAMS["r_ss"]
    shocks = np.zeros((40, 3))
    shocks[0, 0] = -0.06

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        single = solve_occbin(m_ref, m_two_rows, c_zlb, shocks, horizon=40)
        multi = solve_multiconstraint_occbin(
            m_ref, {"zlb": m_two_rows}, shocks, constraints={"zlb": c_zlb}, horizon=40
        )
    one_row = solve_occbin(m_ref, m_zlb_one_row, c_zlb, shocks, horizon=40)

    assert multi.converged and single.converged
    _assert_same_solution(single, multi)

    regimes = np.asarray(multi.regimes)
    zlb_t = np.flatnonzero(regimes & 1)
    assert zlb_t.size > 0
    r = multi.simulated_path["r"].values
    np.testing.assert_allclose(r[zlb_t], r_floor, atol=1e-12, rtol=0.0)
    assert r.min() >= r_floor - 1e-12
    assert np.all(multi.shadow_path["r_shadow"].values[zlb_t] < r_floor)
    # The notional rate is solved out of the equation the peg replaced -- the
    # reference Taylor rule (row 2) -- and not out of another differing row
    # that merely contains r (the IS curve in the IS-row+peg model, which
    # comes first): in every period, binding or slack, r_shadow must satisfy
    # the reference Taylor rule evaluated on the simulated path. Both solvers
    # share the rule, so parity alone would not detect the wrong row.
    y = multi.simulated_path["y"].values
    pi = multi.simulated_path["pi"].values
    r_lag = np.concatenate([[0.0], r[:-1]])
    taylor_rate = (
        PARAMS["rho_r"] * r_lag
        + (1.0 - PARAMS["rho_r"]) * (PARAMS["phi_pi"] * pi + PARAMS["phi_y"] * y)
        + shocks[:, SHOCKS.index("eps_r")]
    )
    np.testing.assert_allclose(multi.shadow_path["r_shadow"].values, taylor_rate, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(single.shadow_path["r_shadow"].values, taylor_rate, atol=1e-10, rtol=0.0)
    # The second differing row changes the economics: the path must differ
    # from the one-row ZLB model's path.
    assert np.max(np.abs(multi.simulated_path.values - one_row.simulated_path.values)) > 1e-3


def test_auto_detection_reads_the_peg_row_not_the_first_differing_row():
    """With constraints=None the constraint is read off the row that pegs a variable."""
    m_ref = _build_reference()
    m_two_rows = _build_constrained(zlb_is_curve_eqs)  # row 0 (IS) differs before row 2 (peg)
    shocks = np.zeros((40, 3))
    shocks[0, 0] = -0.06

    res = solve_multiconstraint_occbin(m_ref, {"zlb": m_two_rows}, shocks, horizon=40)
    c_auto = res.constraints["zlb"]
    assert c_auto.variable == "r"
    assert c_auto.operator == "<"
    assert c_auto.threshold == pytest.approx(-PARAMS["r_ss"], abs=1e-12)

    explicit = solve_multiconstraint_occbin(
        m_ref, {"zlb": m_two_rows}, shocks,
        constraints={"zlb": OccBinConstraint("r", -PARAMS["r_ss"], "<")}, horizon=40,
    )
    _assert_same_solution(res, explicit)


def test_slack_path_equals_linear_reference_path(dual_constraint_models):
    """A shock too small to trip either constraint reproduces the linear decision-rule path."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models
    horizon = 40
    shocks = np.zeros((horizon, 3))
    shocks[0, 0] = -0.001

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = solve_multiconstraint_occbin(
            m_ref, {"zlb": m_zlb, "borrowing": m_borr}, shocks,
            constraints={"zlb": c_zlb, "borrowing": c_borr}, horizon=horizon,
        )
    assert res.converged
    assert res.binding_periods == 0
    assert all(v == 0 for v in res.regimes)

    # Linear path from the reference decision rule y_t = ghx x_{t-1} + ghu u_t
    dr = m_ref.decision_rules()
    n = len(VARIABLES)
    P0 = np.zeros((n, n))
    for s in m_ref.states:
        P0[:, VARIABLES.index(s)] = dr.ghx[s].values
    R0 = dr.ghu.loc[VARIABLES, SHOCKS].values
    X = np.zeros((horizon + 1, n))
    for t in range(1, horizon + 1):
        X[t] = P0 @ X[t - 1] + R0 @ shocks[t - 1]
    np.testing.assert_allclose(res.simulated_path[VARIABLES].values, X[1:], atol=1e-10, rtol=0.0)
    # No constraint ever binds, so every shadow value is the simulated value.
    np.testing.assert_allclose(res.shadow_path["r_shadow"].values, res.simulated_path["r"].values, atol=1e-12)
    np.testing.assert_allclose(res.shadow_path["b_shadow"].values, res.simulated_path["b"].values, atol=1e-12)


def test_max_iter_exhaustion_is_reported(dual_constraint_models):
    """max_iter too small for the fixed point: converged=False AND a warning naming the reason."""
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models

    with pytest.warns(UserWarning, match=r"did not reach a fixed point within max_iter=1"):
        res = solve_multiconstraint_occbin(
            m_ref, {"zlb": m_zlb, "borrowing": m_borr}, _joint_shock(),
            constraints={"zlb": c_zlb, "borrowing": c_borr}, horizon=40, max_iter=1,
        )
    assert res.converged is False
    assert res.iterations == 1
    # The returned path is the one solved under the reported regimes (a
    # diagnostic), and the same call with enough iterations converges.
    assert len(res.regimes) == 40
    ok = solve_multiconstraint_occbin(
        m_ref, {"zlb": m_zlb, "borrowing": m_borr}, _joint_shock(),
        constraints={"zlb": c_zlb, "borrowing": c_borr}, horizon=40,
    )
    assert ok.converged and ok.iterations > 1

    # The solve_occbin dispatch must not lose the diagnostics.
    with pytest.warns(UserWarning, match="did not reach a fixed point"):
        via_dispatch = solve_occbin(
            m_ref, {"zlb": m_zlb, "borrowing": m_borr}, {"zlb": c_zlb, "borrowing": c_borr},
            _joint_shock(), horizon=40, max_iter=1,
        )
    assert via_dispatch.converged is False


def test_nonconvergence_warning_is_attributed_to_the_caller(dual_constraint_models):
    """The warning points at the user's line, also when solve_occbin forwards to the multi solver.

    With a fixed ``stacklevel=2`` a mapping/sequence call through solve_occbin
    attributed the warning to the dispatch line inside occbin.py, so the
    default 'once per location' filter and the printed location were wrong.
    """
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models
    models = {"zlb": m_zlb, "borrowing": m_borr}
    cons = {"zlb": c_zlb, "borrowing": c_borr}
    calls = {
        "direct": lambda: solve_multiconstraint_occbin(m_ref, models, _joint_shock(), constraints=cons, horizon=40, max_iter=1),
        "mapping via solve_occbin": lambda: solve_occbin(m_ref, models, cons, _joint_shock(), horizon=40, max_iter=1),
        "sequence via solve_occbin": lambda: solve_occbin(
            m_ref, [m_zlb, m_borr], [c_zlb, c_borr], _joint_shock(), horizon=40, max_iter=1
        ),
        "single via solve_occbin": lambda: solve_occbin(m_ref, m_zlb, c_zlb, _joint_shock(), horizon=40, max_iter=1),
    }
    for label, call in calls.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = call()
        assert res.converged is False
        user = [w for w in caught if issubclass(w.category, UserWarning) and "converged=False" in str(w.message)]
        assert len(user) == 1, (label, [str(w.message) for w in caught])
        assert user[0].filename == __file__, f"{label}: warning attributed to {user[0].filename}:{user[0].lineno}"


def test_regime_cycle_and_bound_violation_are_reported(dual_constraint_models):
    """A threshold inconsistent with the peg makes the regime guess cycle: reported, not hidden.

    The ZLB model pegs r at -0.015 but the constraint says r < -0.01, so the
    pegged path violates its own bound and the binding test keeps flipping
    period 5. The damping step is exercised (several iterations), no fixed
    point exists, and the result must say so: converged=False plus a warning
    naming the cycle AND the violated bound. Before 3.4.0's fix the multi
    solver returned converged=False silently.
    """
    m_ref, m_zlb, _, _, _ = dual_constraint_models
    shocks = np.zeros((40, 3))
    shocks[0, 0] = -0.06
    c_wrong = OccBinConstraint("r", -0.01, "<")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_multiconstraint_occbin(m_ref, {"zlb": m_zlb}, shocks, constraints={"zlb": c_wrong}, horizon=40)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == 1, messages
    assert "cycled at iteration" in messages[0]
    assert "violates OccBinConstraint(r < -0.01)" in messages[0]
    assert res.converged is False
    assert res.iterations > 1  # the damping branch ran before the cycle was declared
    # The diagnostic path is the one solved under the reported regimes: r is
    # pegged at -0.015 wherever the regime says the constraint binds.
    regimes = np.asarray(res.regimes)
    zlb_t = np.flatnonzero(regimes & 1)
    assert zlb_t.size > 0
    np.testing.assert_allclose(res.simulated_path["r"].values[zlb_t], -PARAMS["r_ss"], atol=1e-12, rtol=0.0)

    # The single-constraint solver reports the same kind of failure.
    with pytest.warns(UserWarning, match="cycled at iteration"):
        single = solve_occbin(m_ref, m_zlb, c_wrong, shocks, horizon=40)
    assert single.converged is False


def trigger_eqs(lead, curr, lag, shocks_v, p):
    """Alternative regime that rewrites the Taylor rule (row 2) when the demand shock is deep.

    The constrained variable g does not appear in the rewritten row, so this is
    a trigger-style constraint: g stays endogenous in both regimes and its
    simulated value is its own notional value.
    """
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + 2.0 * p.phi_y * curr.y) + shocks_v.eps_r),
        curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]


def test_trigger_style_constraint_parity_and_pkf():
    """A trigger-style constraint resolves to the same solution in both solvers and runs in the PKF."""
    import pandas as pd
    from puremacro.dsge.occbin import piecewise_kalman_filter

    m_ref = _build_reference()
    m_trig = _build_constrained(trigger_eqs)
    c_trig = OccBinConstraint("g", -0.03, "<")
    shocks = np.zeros((40, 3))
    shocks[0, 0] = -0.06  # g = -0.06, -0.042, -0.0294, ... trips the trigger for two periods

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        single = solve_occbin(m_ref, m_trig, c_trig, shocks, horizon=40)
        multi = solve_multiconstraint_occbin(m_ref, {"deep": m_trig}, shocks, constraints={"deep": c_trig}, horizon=40)
    assert single.converged and multi.converged
    assert multi.regimes[:3] == [1, 1, 0]
    _assert_same_solution(single, multi)
    g = multi.simulated_path["g"].values
    regimes = np.asarray(multi.regimes)
    # Trigger semantics: g is beyond the threshold exactly in the binding periods.
    np.testing.assert_array_equal(g < -0.03, (regimes & 1) == 1)
    np.testing.assert_allclose(multi.shadow_path["g_shadow"].values, g, atol=1e-12)

    data = pd.DataFrame(np.random.default_rng(0).normal(0.0, 0.01, size=(12, 3)), columns=["y", "pi", "r"])
    pkf = piecewise_kalman_filter(
        m_ref, {"deep": m_trig}, data, varobs=["y", "pi", "r"], constraints={"deep": c_trig}, horizon=10
    )
    assert np.isfinite(pkf.log_likelihood)
    assert pkf.constraints == {"deep": c_trig}


def test_constraints_must_pair_one_to_one_with_models(dual_constraint_models):
    """Mismatched keys, wrong sequence length or one constraint for several models raise.

    3.4.0 regression: such input silently fell through to auto-detection, which
    replaced the user's thresholds with guessed ones and never said so.
    """
    m_ref, m_zlb, m_borr, c_zlb, c_borr = dual_constraint_models
    shocks = _joint_shock()
    models = {"zlb": m_zlb, "borrowing": m_borr}

    # Mapping keys that do not match the model keys
    wrong_keys = {"ZLB": OccBinConstraint("r", -0.03, "<"), "borr": OccBinConstraint("b", 0.05, ">")}
    with pytest.raises(ValueError, match="do not match the constrained-model keys"):
        solve_multiconstraint_occbin(m_ref, models, shocks, constraints=wrong_keys, horizon=40)
    # ... also when only one key is missing
    with pytest.raises(ValueError, match="models without a constraint: \\['borrowing'\\]"):
        solve_multiconstraint_occbin(m_ref, models, shocks, constraints={"zlb": c_zlb}, horizon=40)

    # A sequence of the wrong length (through the solve_occbin dispatch too)
    with pytest.raises(ValueError, match="2 constraint\\(s\\) were given for 1 constrained model"):
        solve_occbin(m_ref, m_zlb, [c_zlb, c_borr], shocks, horizon=40)
    with pytest.raises(ValueError, match="1 constraint\\(s\\) were given for 2 constrained model"):
        solve_multiconstraint_occbin(m_ref, [m_zlb, m_borr], shocks, constraints=[c_zlb], horizon=40)

    # A single OccBinConstraint paired with several models
    with pytest.raises(ValueError, match="a single OccBinConstraint was given for 2 constrained models"):
        solve_occbin(m_ref, models, c_zlb, shocks, horizon=40)

    # A constraint naming an unknown variable, or an unknown relax variable
    with pytest.raises(ValueError, match="constraint variable 'nope' not found in model variables"):
        solve_multiconstraint_occbin(
            m_ref, {"zlb": m_zlb}, shocks, constraints={"zlb": OccBinConstraint("nope", 0.0, "<")}, horizon=40
        )
    with pytest.raises(ValueError, match="relax_variable 'nope'"):
        solve_multiconstraint_occbin(
            m_ref, {"zlb": m_zlb}, shocks,
            constraints={"zlb": OccBinConstraint("r", -PARAMS["r_ss"], "<", relax_variable="nope")}, horizon=40,
        )

    # Correctly keyed input is honoured: the user's thresholds are used verbatim.
    res = solve_multiconstraint_occbin(m_ref, models, shocks, constraints={"zlb": c_zlb, "borrowing": c_borr}, horizon=40)
    assert res.constraints == {"zlb": c_zlb, "borrowing": c_borr}


# ---------------------------------------------------------------------------
# Multiplier-style constraint: the peg lives in a row that does not determine
# the variable in the reference regime (Guerrieri-Iacoviello collateral layout)
# ---------------------------------------------------------------------------

VARIABLES_LAM = ["y", "pi", "r", "b", "g", "lam"]


def ref_lam_eqs(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
        curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b) + curr.lam,  # multiplier enters here
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        curr.lam,  # slack: lam = 0
    ]


def cap_lam_eqs(lead, curr, lag, shocks_v, p):
    eqs = ref_lam_eqs(lead, curr, lag, shocks_v, p)
    eqs[5] = curr.b - p.b_bar  # binding: b = b_bar, lam determined by row 3
    return eqs


@pytest.fixture
def multiplier_models():
    ss = {v: 0.0 for v in VARIABLES_LAM}
    m_ref = build_dynare(ref_lam_eqs, variables=VARIABLES_LAM, shocks=SHOCKS, params=PARAMS, steady_state=ss)
    m_cap = build_dynare(
        cap_lam_eqs, variables=VARIABLES_LAM, shocks=SHOCKS, params=PARAMS, steady_state=ss,
        check_steady_state=False, strict=False,
    )
    shocks = np.zeros((40, 3))
    shocks[0:4, 2] = 0.05  # four-period credit boom
    return m_ref, m_cap, shocks


def test_vacuous_relax_test_is_refused_like_solve_occbin(multiplier_models):
    """Without relax_variable the multi solver refuses the pegged-in-a-non-determining-row constraint.

    3.4.0 regression: the multi-constraint solver ran the vacuous test that
    solve_occbin refuses, chattered, and returned a path with b above the cap
    in a period labelled slack, without any warning.
    """
    m_ref, m_cap, shocks = multiplier_models
    c_bad = OccBinConstraint("b", PARAMS["b_bar"], ">")

    with pytest.raises(ValueError, match="pegs 'b' to a constant"):
        solve_occbin(m_ref, m_cap, c_bad, shocks, horizon=40)
    with pytest.raises(ValueError, match="for constrained model 'cap' pegs 'b' to a constant"):
        solve_multiconstraint_occbin(m_ref, {"cap": m_cap}, shocks, constraints={"cap": c_bad}, horizon=40)
    with pytest.raises(ValueError, match="pegs 'b' to a constant"):
        solve_occbin(m_ref, {"cap": m_cap}, {"cap": c_bad}, shocks, horizon=40)
    # Auto-detection reads the same peg off the model and cannot name the
    # multiplier either, so it refuses and asks for an explicit constraint.
    with pytest.raises(ValueError, match="pass an explicit OccBinConstraint"):
        solve_multiconstraint_occbin(m_ref, {"cap": m_cap}, shocks, horizon=40)


def test_multiplier_relax_variable_gives_parity_with_solve_occbin(multiplier_models):
    """With the multiplier as relax variable both solvers agree and b sits at the cap while binding."""
    m_ref, m_cap, shocks = multiplier_models
    c_ok = OccBinConstraint(
        "b", PARAMS["b_bar"], ">", relax_variable="lam", relax_threshold=0.0, relax_operator="<="
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        single = solve_occbin(m_ref, m_cap, c_ok, shocks, horizon=40)
        multi = solve_multiconstraint_occbin(m_ref, {"cap": m_cap}, shocks, constraints={"cap": c_ok}, horizon=40)

    assert single.converged and multi.converged
    _assert_same_solution(single, multi)
    regimes = np.asarray(multi.regimes)
    cap_t = np.flatnonzero(regimes & 1)
    assert cap_t.size >= 4, f"the four-period credit boom should keep the cap binding, got {multi.regimes[:8]}"
    b = multi.simulated_path["b"].values
    lam = multi.simulated_path["lam"].values
    np.testing.assert_allclose(b[cap_t], PARAMS["b_bar"], atol=1e-12, rtol=0.0)
    assert b.max() <= PARAMS["b_bar"] + 1e-12
    assert np.all(lam[cap_t] > 0.0), "the multiplier is positive while the cap binds"
    np.testing.assert_allclose(lam[(regimes & 1) == 0], 0.0, atol=1e-12)
