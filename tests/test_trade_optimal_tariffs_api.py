"""The tariff-policy entry points name their players; nothing defaults to a country.

Also covers the removed objective (a welfare metric that lived in a module that
did not return) and the code lookups on a calibration of any size.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import (
    build_strategic_tariffs,
    calibrate_trade_model,
    compute_unilateral_optimal_tariff,
    compute_welfare_payoff_matrix,
    evaluate_national_welfare,
    get_country_code,
    resolve_country_indices,
    solve_multilateral_nash_tariffs,
    solve_trade_equilibrium,
)
from puremacro.trade import game, optimal_tariffs
from puremacro.trade.data import synthetic_codes, synthetic_icio_table


@pytest.fixture(scope="module")
def calib():
    countries, sectors = synthetic_codes(5, 4)
    return calibrate_trade_model(synthetic_icio_table(5, 4, seed=0),
                                 country_codes=countries, sector_codes=sectors)


@pytest.fixture(scope="module")
def base(calib):
    res = solve_trade_equilibrium(calib, tol=1e-8)
    assert res.converged
    return res


def test_players_are_required(calib):
    with pytest.raises(TypeError, match="player_countries"):
        solve_multilateral_nash_tariffs(calib)
    with pytest.raises(TypeError, match="once"):
        solve_multilateral_nash_tariffs(calib, ["C01"], players=["C02"])
    with pytest.raises(TypeError):
        compute_unilateral_optimal_tariff(calib)
    with pytest.raises(TypeError):
        compute_welfare_payoff_matrix(calib)


def test_welfare_needs_a_country(calib, base):
    with pytest.raises(TypeError, match="country_idx"):
        evaluate_national_welfare(calib, base)


def test_removed_metric_is_refused(calib, base):
    with pytest.raises(ValueError, match="Unknown welfare metric 'harberger'"):
        evaluate_national_welfare(calib, base, country_idx="C01", metric="harberger")


def test_codes_resolve_against_the_calibration(calib):
    assert resolve_country_indices(calib, "c03") == [2]
    assert resolve_country_indices(calib, ["C05", 0]) == [0, 4]
    assert get_country_code(calib, 4) == "C05"
    with pytest.raises(IndexError):
        get_country_code(calib, 5)
    with pytest.raises(ValueError, match="XYZ"):
        resolve_country_indices(calib, "XYZ")
    # No EU member in the calibration: the bloc is empty, never a stand-in country.
    assert resolve_country_indices(calib, "EUR") == []


def test_strategic_tariffs_follow_codes(calib):
    tau, tau_fd = build_strategic_tariffs(calib, {"C03": {"C01": 0.2}})
    ns, nc = calib.n_sectors, calib.n_countries
    expected = np.ones((ns, nc, ns, nc))
    expected[:, 0, :, 2] = 1.2
    np.testing.assert_array_equal(tau, expected)
    assert np.all(tau_fd[:, 0, :, 2] == 1.2)
    assert np.count_nonzero(tau_fd != 1.0) == ns * calib.n_final_demand


@pytest.mark.parametrize("metric", ["geary_khamis", "hicksian_ev"])
def test_multi_country_player_is_not_silently_truncated(calib, metric):
    # Otherwise welfare belongs to C01+C02 while the tariff applies only to C01.
    with pytest.raises(ValueError, match="one country or a named bloc"):
        compute_unilateral_optimal_tariff(calib, ["C01", "C02"], metric=metric,
                                          tariff_max=.1, num_grid=2, method="grid")


def test_unilateral_geary_khamis_search_runs_on_a_small_table(calib):
    res = compute_unilateral_optimal_tariff(calib, "C02", tariff_max=0.2, num_grid=3, method="grid")
    assert res.country_code == "C02"
    assert res.tariff_grid.shape == (3,)
    assert np.all(np.isfinite(res.welfare_curve))
    assert 0.0 <= res.optimal_tariff_rate <= 0.2
    assert res.metadata["boundary"] in ("lower", "upper", "interior")


def test_game_module_reexports():
    assert set(game.__all__) <= set(optimal_tariffs.__all__)
    assert set(game.__all__) == {
        "resolve_country_indices", "get_country_code", "evaluate_national_welfare",
        "build_strategic_tariffs", "compute_unilateral_optimal_tariff",
        "solve_multilateral_nash_tariffs", "compute_welfare_payoff_matrix",
    }


def test_geary_khamis_welfare_raw_state_parity_and_nominal_scale_invariance():
    """evaluate_national_welfare(metric='geary_khamis') matches TradeEquilibriumResult
    on raw state arrays and is homogeneous of degree zero in nominal prices/incomes
    when anchored to base_equilibrium.
    """
    countries, sectors = synthetic_codes(3, 2)
    calib3 = calibrate_trade_model(
        synthetic_icio_table(3, 2, seed=7), country_codes=countries, sector_codes=sectors
    )
    base_eq = solve_trade_equilibrium(calib3, tol=1e-10)
    tau, tau_fd = build_strategic_tariffs(calib3, {"C02": {"C01": 0.30, "C03": 0.30}})
    cf_eq = solve_trade_equilibrium(calib3, tau=tau, tau_fd=tau_fd, tol=1e-10)

    for code in countries:
        w0 = evaluate_national_welfare(calib3, base_eq, country_idx=code, metric="geary_khamis")
        w0_anchored = evaluate_national_welfare(
            calib3, base_eq, country_idx=code, metric="geary_khamis", base_equilibrium=base_eq
        )
        assert w0_anchored == pytest.approx(w0, rel=1e-13, abs=1e-12)

        w_cf = evaluate_national_welfare(
            calib3, cf_eq, country_idx=code, metric="geary_khamis", base_equilibrium=base_eq
        )
        for scale in (1.0, 0.5, 1.05, 3.7):
            w_raw_scaled = evaluate_national_welfare(
                calib3,
                p_sol=cf_eq.p_sol * scale,
                y_sol=cf_eq.y_sol,
                r_sol=cf_eq.r_sol * scale,
                w_sol=cf_eq.w_sol * scale,
                T_sol=cf_eq.T_sol * scale,
                XN_sol=cf_eq.XN_sol * scale,
                tau=tau,
                tau_fd=tau_fd,
                country_idx=code,
                metric="geary_khamis",
                base_equilibrium=base_eq,
            )
            assert w_raw_scaled == pytest.approx(w_cf, rel=1e-12, abs=1e-12)

    # Distortionary 120% tariff by C02 strictly reduces C02's restored real national
    # absorption across all categories, producing a negative GK welfare change even
    # under arbitrarily inflated nominal prices.
    from puremacro.trade.geary_khamis import restore_capital_formation

    tau_high, tau_fd_high = build_strategic_tariffs(calib3, {"C02": {"C01": 1.20, "C03": 1.20}})
    cf_high = solve_trade_equilibrium(calib3, tau=tau_high, tau_fd=tau_fd_high, tol=1e-10)
    q_base = restore_capital_formation(base_eq.c_fd, base_eq.XN_sol, p_fd=base_eq.Pfd_final)
    q_high = restore_capital_formation(cf_high.c_fd, cf_high.XN_sol, p_fd=cf_high.Pfd_final)
    assert np.all(q_high[:, :, 1] < q_base[:, :, 1])
    w0_c2 = evaluate_national_welfare(calib3, base_eq, country_idx="C02", metric="geary_khamis")
    w_high_c2 = evaluate_national_welfare(
        calib3,
        p_sol=cf_high.p_sol * 2.5,
        y_sol=cf_high.y_sol,
        r_sol=cf_high.r_sol * 2.5,
        w_sol=cf_high.w_sol * 2.5,
        T_sol=cf_high.T_sol * 2.5,
        XN_sol=cf_high.XN_sol * 2.5,
        tau=tau_high,
        tau_fd=tau_fd_high,
        country_idx="C02",
        metric="geary_khamis",
        base_equilibrium=base_eq,
    )
    assert w_high_c2 < w0_c2
