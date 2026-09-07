"""Unit tests for Gertler-Karadi (2011) DSGE with financial frictions and OccBin."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    GK2011_PARAMS,
    GertlerKaradiResult,
    build_gertler_karadi_model,
    solve_gertler_karadi,
    solve_steady_state,
)


# ---------------------------------------------------------------------------
# Test 1: Steady-State Solving & Calibration Checks
# ---------------------------------------------------------------------------

def test_gk_steady_state_calibration():
    """Verify steady-state values match canonical GK (2011) Table 1 targets."""
    ss = solve_steady_state()

    # 1. Steady-state leverage phi = Q*S / N ~ 4.0
    phi_ss = ss["phi"]
    assert 3.8 <= phi_ss <= 4.3, f"Steady-state leverage {phi_ss:.4f} outside target [3.8, 4.3]"

    # 2. Steady-state credit spread R_k - R ~ 100 bps annualized
    spread_ann = ss["spread_ann"]
    assert 95.0 <= spread_ann <= 110.0, f"Annualized spread {spread_ann:.2f} bps outside target [95, 110] bps"

    # 3. Steady-state asset price Q = 1.0, capacity utilization U = 1.0, capital quality xi = 1.0
    assert np.isclose(ss["Q"], 1.0, atol=1e-12)
    assert np.isclose(ss["U"], 1.0, atol=1e-12)
    assert np.isclose(ss["xi"], 1.0, atol=1e-12)
    assert np.isclose(ss["Pi"], 1.0, atol=1e-12)
    assert np.isclose(ss["psi"], 0.0, atol=1e-12)

    # 4. Resource constraint clearing: Y = C + I + G
    res_gap = ss["Y"] - ss["C"] - ss["I"] - ss["G"]
    assert np.isclose(res_gap, 0.0, atol=1e-10), f"Resource constraint gap {res_gap:.2e} != 0"

    # 5. Bank balance sheet clearing: Q*S = B + N
    bs_gap = ss["Q"] * ss["K"] - (ss["B"] + ss["N"])
    assert np.isclose(bs_gap, 0.0, atol=1e-10), f"Bank balance sheet gap {bs_gap:.2e} != 0"

    # 6. Bank net worth renewal: N = N_e + N_n
    nw_gap = ss["N"] - (ss["Ne"] + ss["Nn"])
    assert np.isclose(nw_gap, 0.0, atol=1e-10), f"Net worth renewal gap {nw_gap:.2e} != 0"


# ---------------------------------------------------------------------------
# Test 2: Klein Linear Solver & Capital Quality Shock
# ---------------------------------------------------------------------------

def test_gk_klein_solver_capital_quality_shock():
    """Verify Klein QZ linear solution under capital quality shock.

    Asserts:
    - Credit spread R_k - R spikes significantly (> 100 bps annualized).
    - Bank net worth N contracts sharply (> 15% drop).
    - Output Y and investment I decline.
    - Tobin's Q declines.
    """
    res = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=-0.05,
        horizon=40,
        method="klein",
    )

    assert isinstance(res, GertlerKaradiResult)
    assert res.solver_method == "klein"
    assert res.binding_periods == 0
    assert len(res.irf) == 40

    # 1. Credit spread spikes on impact
    spread_spike_ann = res.irf["prem"].iloc[0] * 40000.0
    assert spread_spike_ann > 100.0, f"Spread spike {spread_spike_ann:.1f} bps < 100 bps"

    # 2. Bank net worth contracts sharply
    ss_N = res.steady_state["N"]
    n_drop_pct = (res.irf["N"].iloc[0] / ss_N) * 100.0
    assert n_drop_pct < -15.0, f"Net worth drop {n_drop_pct:.2f}% not sharp enough (< -15%)"

    # 3. Output and investment decline
    assert res.irf["Y"].iloc[0] < 0.0, "Output did not decline on impact"
    assert res.irf["I"].min() < 0.0, "Investment did not decline"

    # 4. Asset price Q declines
    assert res.irf["Q"].iloc[0] < 0.0, "Asset price Q did not drop on impact"


# ---------------------------------------------------------------------------
# Test 3: OccBin Solver & Credit Policy Regime Switching
# ---------------------------------------------------------------------------

def test_gk_occbin_credit_policy_is_a_verified_fixed_point():
    """The shipped credit-policy configuration has a genuine piecewise-linear
    solution, because the policy rule is continuous at the trigger.

    Until 2.5.0 the constrained regime set ``psi = nu_g * (prem - prem_ss)``.
    That is *discontinuous* where the regimes meet: the reference regime says
    ``psi = 0``, so at the switch point the constrained regime jumped straight
    to ``nu_g * threshold = 0.025``. Since switching policy on compresses the
    very spread that triggered it, the binding indicator flipped and no regime
    sequence was a fixed point -- forcing a contiguous spell of length k gave
    ``k=0 -> 1..11``, ``k=1 -> 2..11``, ``k=9 -> [1..5, 10..14]``,
    ``k=11 -> [1..6, 12..16]``, with no k mapping to itself for any ``nu_g`` in
    [0.1, 10] crossed with any threshold in [0.0025, 0.012].

    The rule now responds to the spread *in excess of* the trigger,
    ``psi = nu_g * (prem - prem_ss - prem_trigger)`` with ``prem_trigger`` set
    to the OccBin threshold, exactly as a ZLB is written. ``psi`` then tends to
    zero as the spread returns to the trigger from above, the two regimes agree
    where they meet, and the iteration has a unique fixed point.
    """
    thresh = 0.0025
    with warnings.catch_warnings():
        # Any RuntimeWarning here means the solver did not certify the path.
        warnings.simplefilter("error", RuntimeWarning)
        res_occ = solve_gertler_karadi(
            shock_type="capital_quality",
            shock_size=-0.05,
            horizon=40,
            method="occbin",
            constraint_type="credit_policy",
            threshold=thresh,
        )

    assert res_occ.solver_method == "occbin"
    assert res_occ.occbin_result is not None
    assert res_occ.occbin_result.converged is True
    assert res_occ.converged is True

    regimes = np.asarray(res_occ.occbin_result.regimes, dtype=int)
    prem = res_occ.irf["prem"].to_numpy()
    psi = res_occ.irf["psi"].to_numpy()

    # -- Fixed point, re-derived here rather than read off the result. The
    #    constraint is trigger-style ("policy is on iff the spread is above the
    #    trigger"), so the binding test on the returned path is just
    #    prem > threshold. It must reproduce the regime sequence exactly.
    implied = (prem > thresh).astype(int)
    np.testing.assert_array_equal(
        implied, regimes,
        err_msg="the returned path does not re-imply its own regime sequence, "
                "so it is not a fixed point of the OccBin iteration")

    # A single contiguous spell that starts on impact and ends inside the
    # horizon, so the terminal condition is verified rather than assumed.
    assert regimes[0] == 1 and regimes[-1] == 0
    assert res_occ.binding_periods == int(regimes.sum()) == 14
    assert np.array_equal(regimes, np.r_[np.ones(14, int), np.zeros(26, int)])

    # -- The central bank injects credit, it never withdraws it.
    assert np.all(psi[regimes == 1] > 0.0), "psi must be positive while policy is on"
    assert np.all(np.abs(psi[regimes == 0]) < 1e-12), "psi must be 0 when policy is off"

    # -- Continuity at the trigger: psi is proportional to the spread in EXCESS
    #    of the trigger, so it decays to ~0 by the end of the spell instead of
    #    falling off a cliff of nu_g * threshold (= 0.025) as the old rule did.
    nu_g = GK2011_PARAMS["nu_g"]
    np.testing.assert_allclose(
        psi[regimes == 1], nu_g * (prem[regimes == 1] - thresh), atol=1e-10)
    last = int(np.flatnonzero(regimes == 1)[-1])
    assert psi[last] < 0.01 * nu_g * thresh, (
        f"psi at the last binding period is {psi[last]:.6g}; it should be near "
        f"zero, not near the old discontinuous jump {nu_g * thresh:.6g}")
    assert psi[last] < psi[0], "psi should decay over the spell, not jump at its end"

    # -- Credit policy cushions the spread and the slump.
    res_klein = solve_gertler_karadi(
        shock_type="capital_quality", shock_size=-0.05, horizon=40, method="klein")
    assert prem.max() < res_klein.irf["prem"].to_numpy().max(), (
        "credit policy must compress the peak credit spread relative to the "
        "unconstrained linear path")
    assert res_occ.irf["Y"].min() > res_klein.irf["Y"].min(), (
        "credit policy must cushion the output trough")


def test_gk_credit_policy_trigger_defaults_to_zero_in_the_calibration():
    """``prem_trigger`` defaults to 0.0, so a caller who builds the constrained
    model by hand reproduces the pre-2.5.0 algebra; only
    ``solve_gertler_karadi`` aligns it with the OccBin threshold. Adding the
    parameter cannot move the steady state, which is regime-independent."""
    assert GK2011_PARAMS["prem_trigger"] == 0.0

    ss_base = solve_steady_state()
    ss_trig = solve_steady_state({"prem_trigger": 0.0025})
    assert ss_base.keys() == ss_trig.keys()
    for key in ss_base:
        assert ss_base[key] == pytest.approx(ss_trig[key], rel=0, abs=0.0), key


def test_gk_occbin_credit_policy_converges_when_the_threshold_is_never_reached():
    """The same configuration with a threshold the spread never reaches is a
    genuine, verified OccBin solution: the constraint is slack throughout and
    the path is the linear one."""
    res_occ = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=-0.05,
        horizon=40,
        method="occbin",
        constraint_type="credit_policy",
        threshold=0.05,          # prem peaks at ~0.015, so this never binds
    )
    assert res_occ.occbin_result.converged is True
    assert res_occ.binding_periods == 0

    res_klein = solve_gertler_karadi(
        shock_type="capital_quality", shock_size=-0.05, horizon=40, method="klein")
    np.testing.assert_allclose(res_occ.irf.values, res_klein.irf.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: Small Shock Matches Linear Klein to Machine Precision
# ---------------------------------------------------------------------------

def test_gk_occbin_small_shock_matches_klein():
    """Verify that under a small shock that never triggers the threshold,
    OccBin matches the pure linear Klein solution to numerical precision.
    """
    small_shock = -0.001
    res_klein = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=small_shock,
        horizon=40,
        method="klein",
    )
    res_occ = solve_gertler_karadi(
        shock_type="capital_quality",
        shock_size=small_shock,
        horizon=40,
        method="occbin",
        constraint_type="credit_policy",
        threshold=0.0025,
    )

    assert res_occ.binding_periods == 0, "Small shock unexpectedly triggered binding constraint"
    assert res_occ.regimes == [0] * 40

    # Test numerical equivalence across all variables
    np.testing.assert_allclose(res_occ.irf.values, res_klein.irf.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 5: OccBin Leverage Cap Constraint
# ---------------------------------------------------------------------------

def _gk_leverage_cap_iteration(threshold, horizon, shock_size=-0.05):
    """Rebuild the OccBin recursion for the shipped leverage cap from scratch.

    Returns ``(simulate, update, variables)``. ``simulate(regime)`` runs the
    Guerrieri-Iacoviello backward recursion under an arbitrary boolean regime
    vector; ``update(regime, path)`` applies the solver's own binding test to
    the resulting path. A regime with ``update(r, simulate(r)) == r`` is a
    fixed point. Written independently of :func:`solve_occbin` so that the
    tests below verify the shipped answer rather than restate it.
    """
    from puremacro.dsge.gertler_karadi import GK_SHOCKS
    from puremacro.dsge.occbin import _extract_model_matrices

    ss = solve_steady_state()
    ref_model = build_gertler_karadi_model(GK2011_PARAMS, regime="reference")
    p_cons = dict(GK2011_PARAMS)
    p_cons["phi_max"] = ss["phi_ss"] + threshold
    cons_model = build_gertler_karadi_model(
        p_cons, regime="constrained", constraint_type="leverage_cap",
        check_steady_state=False)

    reg_mats = (
        _extract_model_matrices(ref_model)[:5],
        _extract_model_matrices(cons_model, ref_model=ref_model)[:5],
    )
    variables = list(ref_model.variables)
    n_vars = len(variables)

    dr = ref_model.decision_rules()
    P_ref = np.zeros((n_vars, n_vars))
    for state in ref_model.states:
        P_ref[:, variables.index(state)] = dr.ghx[state].values

    u = np.zeros((horizon, len(GK_SHOCKS)))
    u[0, GK_SHOCKS.index("eps_xi")] = shock_size
    i_phi, i_psi = variables.index("phi"), variables.index("psi")

    def simulate(regime):
        P_seq = [None] * (horizon + 1)
        D_seq = [None] * (horizon + 1)
        P_next, D_next = P_ref, np.zeros(n_vars)
        for t in range(horizon, 0, -1):
            A_p, A_0, A_m, B_u, c = reg_mats[int(regime[t - 1])]
            M_t = A_0 + A_p @ P_next
            P_seq[t] = np.linalg.solve(M_t, -A_m)
            D_seq[t] = np.linalg.solve(M_t, -(A_p @ D_next + c + B_u @ u[t - 1]))
            P_next, D_next = P_seq[t], D_seq[t]
        X = np.zeros((horizon + 1, n_vars))
        for t in range(1, horizon + 1):
            X[t] = P_seq[t] @ X[t - 1] + D_seq[t]
        return X[1:]

    def update(regime, path):
        # Complementary slackness, period by period: while the cap is on it
        # stays on for as long as the public credit share it needs is
        # non-negative; while it is off it switches on if private leverage
        # would breach the cap.
        return np.array(
            [1 if (path[t, i_psi] >= 0.0 if regime[t] else path[t, i_phi] > threshold)
             else 0 for t in range(horizon)],
            dtype=int,
        )

    return simulate, update, variables


def test_gk_occbin_leverage_cap_default_is_a_verified_fixed_point():
    """The shipped leverage cap converges at its defaults and the regime
    sequence is a genuine fixed point, re-derived here from the returned path.

    The cap is enforced through the public credit share ``psi``: the
    constrained regime replaces equation 26 (``psi = 0``) with
    ``phi = phi_max`` and equation 17, ``(1 - psi) Q K = phi N``, says how much
    intermediation the public balance sheet has to absorb. The moral-hazard
    incentive constraint (equation 16) is kept, so the complementarity is
    ``phi <= phi_max``, ``psi >= 0``, ``psi = 0`` wherever the cap is slack --
    and the two regimes agree exactly where they meet.

    Until 2.5.0 the constrained regime replaced equation **16** instead, which
    is what anchors the banker's value block ``(nu, eta, Omega)``. That
    constrained regime violates Blanchard-Kahn with no stable solution, every
    binding spell excites the explosive direction, notional leverage rises
    through the spell instead of falling back to the cap, and the exit test
    becomes self-fulfilling: at a cap 5% above ``phi_ss`` every leading spell of
    length k >= 16 was a fixed point at every horizon (16..40 at H=40, 16..80 at
    H=80, and likewise at H=120 and H=200). The path the solver returned then
    put bank leverage at -28.4 against ``phi_ss = 4.098``, net worth at -2.54
    against ``N_ss = +1.38``, and output 68% below ``Y_ss``.
    """
    ss = solve_steady_state()
    thresh = 0.20 * ss["phi_ss"]

    with warnings.catch_warnings():
        # Any warning here means the solver did not certify the path.
        warnings.simplefilter("error")
        res_cap = solve_gertler_karadi(
            shock_type="capital_quality", shock_size=-0.05, horizon=40,
            method="occbin", constraint_type="leverage_cap")

    assert res_cap.solver_method == "occbin"
    assert res_cap.occbin_result is not None
    assert res_cap.occbin_result.converged is True
    assert res_cap.converged is True

    regimes = np.asarray(res_cap.occbin_result.regimes, dtype=int)
    phi = res_cap.irf["phi"].to_numpy()
    psi = res_cap.irf["psi"].to_numpy()

    # -- The default is a cap 20% above steady-state leverage, not the
    #    degenerate cap AT the steady state that used to be shipped.
    assert thresh == pytest.approx(0.8196525715908, rel=1e-9)
    assert res_cap.occbin_result.constraint.threshold == pytest.approx(thresh)

    # -- A single contiguous spell that starts on impact and ends well inside
    #    the horizon, so the terminal condition is tested, not assumed.
    assert np.array_equal(regimes, np.r_[np.ones(13, int), np.zeros(27, int)])
    assert res_cap.binding_periods == 13
    assert regimes[0] == 1 and regimes[-1] == 0

    # -- Fixed point, re-derived from the returned path rather than read off
    #    the result: while the cap is on it stays on iff the public credit
    #    share it requires is non-negative; while it is off it switches on iff
    #    private leverage would breach the cap.
    implied = np.where(regimes == 1, psi >= 0.0, phi > thresh).astype(int)
    np.testing.assert_array_equal(
        implied, regimes,
        err_msg="the returned path does not re-imply its own regime sequence, "
                "so it is not a fixed point of the OccBin iteration")

    # -- Complementary slackness in levels.
    np.testing.assert_allclose(phi[regimes == 1], thresh, atol=1e-10)
    assert np.all(phi[regimes == 0] < thresh)
    assert np.all(psi[regimes == 1] > 0.0), "the cap must need a positive intervention"
    assert np.all(np.abs(psi[regimes == 0]) < 1e-12), "psi must be 0 once the cap is slack"

    # -- Continuity at the switch point: psi decays to ~0 inside the spell
    #    instead of falling off a cliff, which is what makes the exit condition
    #    well posed rather than self-fulfilling.
    last = int(np.flatnonzero(regimes == 1)[-1])
    assert psi[last] < 0.06 * psi[0], (
        f"psi at the last binding period is {psi[last]:.6g} against {psi[0]:.6g} on "
        "impact; it should decay to the reference regime's zero, not jump to it")
    assert psi[last] < psi[0]
    assert phi[last + 1] < thresh and phi[last + 1] > 0.9 * thresh, (
        "leverage should leave the cap continuously, not leap away from it")

    # -- The path is sane in levels, which the pre-2.5.0 formulation was not.
    assert ss["N"] + res_cap.irf["N"].min() > 0.0, "bank net worth must stay positive"
    assert res_cap.irf["Y"].min() > -0.10 * ss["Y_ss"], "output trough must be plausible"
    assert psi.max() < 0.25, "the public balance sheet must stay a minority of credit"


def test_gk_occbin_leverage_cap_is_unique_and_horizon_invariant():
    """The default cap has ONE fixed point, and it does not move with the
    truncation horizon.

    Both properties are checked against a from-scratch reimplementation of the
    backward recursion, not against the solver's own report. Uniqueness is
    established over every one of the 861 ``(start, end)`` contiguous windows at
    ``horizon=40`` -- spells that start after impact included -- and from six
    different initial guesses.

    This is exactly what the pre-2.5.0 formulation failed: its fixed-point set
    was the interval ``[16, horizon]`` -- an equilibrium set that is a function
    of the arbitrary truncation -- and which member the solver returned was an
    artefact of the all-slack initial guess.
    """
    ss = solve_steady_state()
    thresh = 0.20 * ss["phi_ss"]
    horizon = 40
    expected = np.r_[np.ones(13, int), np.zeros(horizon - 13, int)]

    simulate, update, _ = _gk_leverage_cap_iteration(thresh, horizon)

    # 1. Uniqueness over every contiguous window (861 candidates).
    fixed_points = []
    n_candidates = 0
    for start in range(horizon + 1):
        for end in range(start, horizon + 1):
            regime = np.zeros(horizon, dtype=int)
            regime[start:end] = 1
            n_candidates += 1
            if np.array_equal(update(regime, simulate(regime)), regime):
                fixed_points.append((start + 1, end))
    assert n_candidates == 861
    assert fixed_points == [(1, 13)], (
        f"expected a unique contiguous fixed point 1..13, got {fixed_points}")

    # 2. The iteration lands there from every start, not just from all-slack.
    starts = {
        "all-slack": np.zeros(horizon, int),
        "all-binding": np.ones(horizon, int),
        "leading-10": np.r_[np.ones(10, int), np.zeros(30, int)],
        "leading-20": np.r_[np.ones(20, int), np.zeros(20, int)],
        "leading-30": np.r_[np.ones(30, int), np.zeros(10, int)],
        "alternating": np.arange(horizon) % 2,
    }
    for name, guess in starts.items():
        regime = np.asarray(guess, dtype=int)
        seen = {tuple(regime.tolist())}
        for _ in range(50):
            nxt = update(regime, simulate(regime))
            if np.array_equal(nxt, regime):
                break
            assert tuple(nxt.tolist()) not in seen, f"the iteration cycled from {name}"
            seen.add(tuple(nxt.tolist()))
            regime = nxt
        else:  # pragma: no cover - would mean a non-convergent iteration
            raise AssertionError(f"no fixed point reached from {name}")
        np.testing.assert_array_equal(
            regime, expected,
            err_msg=f"start {name} converged to a different regime sequence")

    # 3. Horizon invariance of both the spell and the path.
    base = None
    for h in (40, 60, 80, 120, 200):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = solve_gertler_karadi(
                shock_type="capital_quality", shock_size=-0.05, horizon=h,
                method="occbin", constraint_type="leverage_cap")
        regimes = np.asarray(res.occbin_result.regimes, dtype=int)
        assert res.converged is True
        assert np.array_equal(regimes, np.r_[np.ones(13, int), np.zeros(h - 13, int)]), (
            f"horizon {h} gave binding periods {np.flatnonzero(regimes) + 1}")
        head = res.irf.to_numpy()[:40]
        if base is None:
            base = head
        else:
            np.testing.assert_allclose(
                head, base, atol=1e-9,
                err_msg=f"the first 40 periods moved when the horizon grew to {h}")


def test_gk_occbin_leverage_cap_at_the_steady_state_binds_forever():
    """A cap set AT the steady state binds for the whole horizon, at any horizon.

    ``threshold=0.0`` puts ``phi_max`` exactly at ``phi_ss``. After a -5%
    capital-quality shock leverage jumps to +1.877 and decays monotonically
    without ever returning below zero, so the cap binds in every simulated
    period at H = 40, 60, 80, 120 and 200 alike. The reference regime therefore
    never resumes inside the window, the terminal condition is assumed rather
    than verified, and the honest flag is ``converged=False`` -- a degenerate
    calibration, deliberately left non-convergent rather than papered over.

    This test used to assert ``converged is True`` on exactly that unverified
    path.
    """
    for h in (40, 200):
        with pytest.warns(RuntimeWarning, match="did not converge to a verified"):
            res_cap = solve_gertler_karadi(
                shock_type="capital_quality", shock_size=-0.05, horizon=h,
                method="occbin", constraint_type="leverage_cap", threshold=0.0)

        assert res_cap.solver_method == "occbin"
        assert res_cap.occbin_result.converged is False
        assert res_cap.converged is False
        assert res_cap.binding_periods == h, "the cap should bind in every period"


def test_gk_leverage_cap_reports_a_spell_that_does_not_fit_the_horizon():
    """A spell longer than the horizon is a horizon problem, and says so.

    A -10% capital-quality shock keeps the default cap binding for 63 quarters.
    At ``horizon=40`` the constraint still binds in the last simulated period,
    so the terminal condition is assumed rather than tested and the honest flag
    is ``converged=False``; at horizons that fit, the answer is the same
    verified 63-quarter spell whichever horizon is used.
    """
    with pytest.warns(RuntimeWarning, match="did not converge to a verified"):
        short = solve_gertler_karadi(
            shock_type="capital_quality", shock_size=-0.10, horizon=40,
            method="occbin", constraint_type="leverage_cap")
    assert short.converged is False
    assert short.binding_periods == 40

    for h in (80, 200):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = solve_gertler_karadi(
                shock_type="capital_quality", shock_size=-0.10, horizon=h,
                method="occbin", constraint_type="leverage_cap")
        regimes = np.asarray(res.occbin_result.regimes, dtype=int)
        assert res.converged is True
        assert np.array_equal(regimes, np.r_[np.ones(63, int), np.zeros(h - 63, int)])


def test_gk_occbin_leverage_cap_never_binds_above_the_unconstrained_peak():
    """A cap above the peak of the unconstrained path is verified and inert.

    Leverage peaks at +1.877 on impact, so a cap 50% above ``phi_ss``
    (threshold 2.049) is never reached and the OccBin path must be the linear
    Klein path to machine precision."""
    ss = solve_steady_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res_cap = solve_gertler_karadi(
            shock_type="capital_quality", shock_size=-0.05, horizon=40,
            method="occbin", constraint_type="leverage_cap",
            threshold=0.50 * ss["phi_ss"])
    assert res_cap.occbin_result.converged is True
    assert res_cap.binding_periods == 0

    res_klein = solve_gertler_karadi(
        shock_type="capital_quality", shock_size=-0.05, horizon=40, method="klein")
    np.testing.assert_allclose(res_cap.irf.values, res_klein.irf.values, atol=1e-10)


def test_gk_leverage_cap_spell_lengthens_monotonically_as_the_cap_tightens():
    """Tighter caps bind for longer, and every cell is a verified fixed point.

    A cap 15/20/25/30/35/40/45 percent above ``phi_ss`` binds for
    32/13/7/4/3/2/1 periods. Monotonicity is the economically meaningful comparative static and
    it is what the pre-2.5.0 formulation could not deliver: there the spell was
    whichever member of the continuum ``[16, horizon]`` the initial guess
    happened to reach."""
    ss = solve_steady_state()
    lengths = []
    for pct in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = solve_gertler_karadi(
                shock_type="capital_quality", shock_size=-0.05, horizon=40,
                method="occbin", constraint_type="leverage_cap",
                threshold=pct * ss["phi_ss"])
        regimes = np.asarray(res.occbin_result.regimes, dtype=int)
        assert res.converged is True
        assert regimes[-1] == 0
        k = int(regimes.sum())
        assert np.array_equal(regimes, np.r_[np.ones(k, int), np.zeros(40 - k, int)])
        lengths.append(k)
    assert lengths == [32, 13, 7, 4, 3, 2, 1], lengths
    assert lengths == sorted(lengths, reverse=True)


def test_gk_leverage_cap_constrained_regime_has_a_stable_solution():
    """The constrained regime must be a well-posed linear model in its own right.

    OccBin does not require it -- Guerrieri & Iacoviello explicitly allow an
    indeterminate alternative regime, and a ZLB regime typically has too FEW
    unstable roots. What is fatal is the other direction: *no* stable solution,
    so that a finite binding spell excites an explosive direction instead of
    having an indeterminate one pinned down by the terminal condition. The
    pre-2.5.0 leverage cap (equation 16 replaced) failed exactly that way: the
    reference's largest stable root 0.95420 disappeared and the capped regime's
    smallest unstable root was 1.01845, the banker net-worth accumulation root
    ``1 / (theta_b * (1 + prem_ss * phi_ss))`` pushed outside the unit circle.
    """
    ss = solve_steady_state()
    ref_model = build_gertler_karadi_model(GK2011_PARAMS, regime="reference")
    assert ref_model.is_determinate is True

    p_cons = dict(GK2011_PARAMS)
    p_cons["phi_max"] = ss["phi_ss"] + 0.20 * ss["phi_ss"]
    cons_model = build_gertler_karadi_model(
        p_cons, regime="constrained", constraint_type="leverage_cap",
        check_steady_state=False)
    assert cons_model.is_determinate is True, (
        "the leverage-cap regime has no stable solution; a finite binding spell "
        "would then explode instead of returning to the reference regime")


def test_occbin_refuses_a_pegged_variable_it_cannot_test_for_release():
    """A peg written into an equation that does not determine the variable in
    the reference regime has no notional value to test, so ``solve_occbin``
    refuses instead of running a vacuous relax test.

    Without ``relax_variable`` the release test compares the simulated ``phi``
    -- pinned AT the cap by the peg itself -- against the very cap it is pinned
    to, and answers "already relaxed" in every binding period.
    """
    from puremacro.dsge import OccBinConstraint, solve_occbin

    ss = solve_steady_state()
    thresh = 0.20 * ss["phi_ss"]
    ref_model = build_gertler_karadi_model(GK2011_PARAMS, regime="reference")
    p_cons = dict(GK2011_PARAMS)
    p_cons["phi_max"] = ss["phi_ss"] + thresh
    cons_model = build_gertler_karadi_model(
        p_cons, regime="constrained", constraint_type="leverage_cap",
        check_steady_state=False)
    shocks = np.zeros((40, 3))
    shocks[0, 0] = -0.05

    with pytest.raises(ValueError, match="pegs 'phi' to a constant"):
        solve_occbin(
            ref_model, cons_model,
            OccBinConstraint(variable="phi", threshold=thresh, operator=">"),
            shock_sequence=shocks, horizon=40)

    with pytest.raises(ValueError, match="relax_variable 'not_a_variable'"):
        solve_occbin(
            ref_model, cons_model,
            OccBinConstraint(variable="phi", threshold=thresh, operator=">",
                             relax_variable="not_a_variable"),
            shock_sequence=shocks, horizon=40)


# ---------------------------------------------------------------------------
# Test 6: Alternative Shocks (TFP & Monetary Policy)
# ---------------------------------------------------------------------------

def test_gk_alternative_shocks():
    """Verify model response to technology (TFP) and monetary policy shocks."""
    # 1. Technology shock (positive 1% TFP innovation)
    # Under Calvo price stickiness (Galí 1999), technology shock lowers inflation
    # and hours worked while increasing consumption.
    res_tfp = solve_gertler_karadi(shock_type="tfp", shock_size=0.01, method="klein")
    assert res_tfp.irf["a"].iloc[0] > 0.0, "Technology level did not increase"
    assert res_tfp.irf["Pi"].iloc[0] < 0.0, "Positive TFP shock did not lower inflation"
    assert res_tfp.irf["C"].iloc[0] > 0.0, "Positive TFP shock did not increase consumption"
    assert res_tfp.irf["L"].iloc[0] < 0.0, "Hours worked did not contract under sticky prices"

    # 2. Contractionary monetary policy shock (positive 25 bps nominal rate shock)
    res_mon = solve_gertler_karadi(shock_type="monetary", shock_size=0.0025, method="klein")
    assert res_mon.irf["Rn"].iloc[0] > 0.0, "Monetary shock did not increase nominal rate"
    assert res_mon.irf["Y"].iloc[0] < 0.0, "Contractionary monetary shock did not depress output"
    assert res_mon.irf["Pi"].iloc[0] < 0.0, "Contractionary monetary shock did not reduce inflation"


# ---------------------------------------------------------------------------
# Test 7: Result Presentation Suite & Subscript Access
# ---------------------------------------------------------------------------

def test_gk_result_presentation_suite():
    """Verify GertlerKaradiResult implements the puremacro presentation contract."""
    res = solve_gertler_karadi(method="occbin", shock_size=-0.05)

    # 1. Subscript access
    y_series = res["Y"]
    assert isinstance(y_series, pd.Series)
    assert len(y_series) == 40
    assert res["solver_method"] == "occbin"

    with pytest.raises(KeyError):
        _ = res["non_existent_variable"]

    # 2. .to_frame()
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40

    # 3. .summary()
    summary_text = res.summary()
    assert isinstance(summary_text, str)
    assert "GERTLER-KARADI" in summary_text
    assert "CALIBRATION" in summary_text
    assert "TRAJECTORY" in summary_text

    # 4. .to_markdown()
    md_text = res.to_markdown()
    assert isinstance(md_text, str)
    assert "|" in md_text

    # 5. .to_latex()
    latex_text = res.to_latex()
    assert isinstance(latex_text, str)
    assert "\\begin{tabular}" in latex_text

    # 6. .to_typst()
    typst_text = res.to_typst()
    assert isinstance(typst_text, str)
    assert "#table(" in typst_text

    # 7. .plot()
    fig = res.plot(style="publication")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig_default = res.plot(variables=["Y", "prem", "N"], style="default")
    assert isinstance(fig_default, plt.Figure)
    plt.close(fig_default)


# ---------------------------------------------------------------------------
# Test 8: Custom Parameters & Sensitivity
# ---------------------------------------------------------------------------

def test_gk_custom_parameters():
    """Verify that solve_gertler_karadi correctly accepts custom parameter overrides."""
    custom_params = {
        "beta": 0.985,
        "theta_b": 0.965,
        "lambda_b": 0.35,
    }
    res = solve_gertler_karadi(params=custom_params, method="klein")
    assert res.steady_state["R"] == pytest.approx(1.0 / 0.985, rel=1e-5)
    assert res.params["theta_b"] == 0.965
    assert res.params["lambda_b"] == 0.35


# ---------------------------------------------------------------------------
# Test 9: Adversarial Input Validation
# ---------------------------------------------------------------------------

def test_gk_input_validation():
    """Verify helpful error messages on invalid shock types, methods, or regimes."""
    with pytest.raises(ValueError, match="unknown shock_type"):
        solve_gertler_karadi(shock_type="invalid_shock")

    with pytest.raises(ValueError, match="unknown method"):
        solve_gertler_karadi(method="invalid_solver")

    with pytest.raises(ValueError, match="unknown constraint_type"):
        solve_gertler_karadi(method="occbin", constraint_type="unknown_constraint")


# ---------------------------------------------------------------------------
# Test 10: Shock Aliases and Variable Horizons
# ---------------------------------------------------------------------------

def test_gk_aliases_and_horizons():
    """Verify shock aliases ('xi', 'technology', 'policy') and varying horizons."""
    # Test xi alias
    res_xi = solve_gertler_karadi(shock_type="xi", shock_size=-0.03, horizon=20, method="klein")
    assert len(res_xi.irf) == 20
    assert res_xi.irf["prem"].iloc[0] > 0.0

    # Test technology alias
    res_tech = solve_gertler_karadi(shock_type="technology", shock_size=0.01, horizon=30, method="klein")
    assert len(res_tech.irf) == 30
    assert res_tech.irf["a"].iloc[0] == pytest.approx(0.01, rel=1e-5)

    # Test policy alias
    res_pol = solve_gertler_karadi(shock_type="policy", shock_size=0.005, horizon=15, method="klein")
    assert len(res_pol.irf) == 15
    assert res_pol.irf["Rn"].iloc[0] > 0.0


# ---------------------------------------------------------------------------
# Test 11: LinearModel Architecture & Blanchard-Kahn
# ---------------------------------------------------------------------------

def test_gk_linear_model_properties():
    """Verify build_gertler_karadi_model satisfies Blanchard-Kahn conditions."""
    model = build_gertler_karadi_model()
    assert len(model.variables) == 26
    assert len(model.shocks) == 3
    assert len(model.states) > 0
    assert len(model.controls) > 0

    # Check that generalized eigenvalues correctly satisfy Blanchard-Kahn condition:
    # number of generalized eigenvalues with modulus > 1 equals number of forward-looking controls
    # The engine solves the stacked lead/lag system (lagged copies of the
    # states are the predetermined block, every current variable is
    # non-predetermined), so there are n_states + n_variables generalised
    # eigenvalues and Blanchard-Kahn requires exactly n_variables unstable ones.
    eigs = model.eigenvalues
    assert len(eigs) == model.n_states + len(model.variables)
    n_unstable = np.sum(np.abs(eigs) > 1.0 + 1e-6)
    assert n_unstable == len(model.variables)
    assert model.is_determinate
    assert model.solution.G.shape == (model.n_states, model.n_states)
    assert model.solution.F.shape == (model.n_controls, model.n_states)

