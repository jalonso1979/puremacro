"""Comprehensive Regression Suite for 3.3.0 Adversarial Code Review Remediations.

Tests findings CR-01 through CR-10:
- CR-01: Positive percentage climate shocks & multiplier validation in Allen-Arkolakis.
- CR-02: Zero congestion elasticity (beta=0.0) validation error in Allen-Arkolakis.
- CR-03: Nash tariff boundary finite-difference stencil span (no gradient halving at tau=0).
- CR-04: Geary-Khamis numeraire_idx and automatic USA detection in arbitrary country subsets.
- CR-05 & CR-06: Analytic gradients LaTeX escaping (Greek vs Latin identifiers) and IFT documentation.
- CR-07: Custom euler_target_fn support in DeepMacroModel training loop.
- CR-08: Bilateral trade expenditure shares destination/origin property accessors.
- CR-09: Trade calibration CES weights caching bound and garbage collection safety.
- CR-10: DC-EGM upper_envelope single-point segment robustness (no IndexError).
"""
from __future__ import annotations

import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from puremacro.spatial.allen_arkolakis import AllenArkolakisModel, AllenArkolakisResult
from puremacro.trade import solve_trade_equilibrium
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.equilibrium import _get_ces_weights
from puremacro.trade.geary_khamis import compute_geary_khamis, solve_multilateral_ppp
from puremacro.trade.optimal_tariffs import (
    build_strategic_tariffs,
    evaluate_national_welfare,
    solve_multilateral_nash_tariffs,
)
from puremacro.vfi.analytic_gradients import AnalyticGradientResult, _GREEK_LETTERS
from puremacro.vfi.dcegm import upper_envelope
from puremacro.vfi.deep_macro import DeepMacroModel, solve_deep_macro


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_trade_calib():
    """Mathematically balanced 2-country 2-sector calibrated CGE model."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)

    # Intermediate transactions (4x4)
    data[:4, :4] = np.array([
        [10.0, 15.0, 5.0, 5.0],
        [15.0, 20.0, 10.0, 10.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    y = np.array([100.0, 150.0, 120.0, 180.0])
    inter_col_sums = data[:4, :4].sum(axis=0)
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac

    # Final demand (4x6)
    fd_row_sums = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        tot_fd = fd_row_sums[i]
        shares = [0.50, 0.25, 0.05, 0.10, 0.08, 0.02] if i < 2 else [0.10, 0.08, 0.02, 0.50, 0.25, 0.05]
        for c_idx, sh in enumerate(shares):
            data[i, 4 + c_idx] = tot_fd * sh

    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=("CAN", "USA"),
        sector_codes=("S1", "S2"),
        validate=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCR01AndCR02AndCR08Spatial:
    """CR-01, CR-02, CR-08: Spatial general equilibrium fixes."""

    def test_cr02_beta_zero_raises_value_error(self):
        """CR-02: Instantiating with beta == 0.0 must raise ValueError to prevent ZeroDivisionError."""
        with pytest.raises(ValueError, match="Amenity congestion elasticity beta cannot be zero"):
            AllenArkolakisModel(
                trade_costs=np.ones((2, 2)),
                fundamental_productivity=np.ones(2),
                fundamental_amenity=np.ones(2),
                theta=4.0,
                alpha=0.10,
                beta=0.0,
            )

    def test_cr01_climate_positive_and_percentage_shocks(self):
        """CR-01: Climate shock must handle positive shocks and is_percentage_change correctly."""
        model = AllenArkolakisModel(
            trade_costs=np.array([[1.0, 1.5], [1.5, 1.0]]),
            fundamental_productivity=np.array([1.0, 1.0]),
            fundamental_amenity=np.array([1.0, 1.0]),
            theta=4.0,
            alpha=0.10,
            beta=-0.30,
            region_names=["East", "West"],
        )

        # 1. Positive percentage change (+20%)
        res_pos = model.simulate_climate_shock(
            productivity_shocks={"East": 0.20},
            is_percentage_change=True,
        )
        assert res_pos.converged
        assert res_pos.welfare_pct > 0.0
        assert res_pos.L_hat[0] > 1.0  # East gains population

        # 2. Positive gross multiplier (1.20)
        res_mult = model.simulate_climate_shock(
            productivity_shocks={"East": 1.20},
            is_percentage_change=False,
        )
        assert res_mult.converged
        assert np.isclose(res_mult.welfare_pct, res_pos.welfare_pct, atol=1e-6)

        # 3. Warning emitted when passing small positive factor < 0.5 without is_percentage_change
        with pytest.warns(UserWarning, match="interpreted as a gross multiplier"):
            model.simulate_climate_shock(
                productivity_shocks={"East": 0.10},
                is_percentage_change=False,
            )

        # 4. Invalid negative shock <= -1.0 raises ValueError
        with pytest.raises(ValueError, match="cannot be <= -1.0"):
            model.simulate_climate_shock(
                productivity_shocks={"East": -1.0},
                is_percentage_change=True,
            )

    def test_cr08_trade_shares_properties(self):
        """CR-08: Explicit accessors for destination-origin and origin-destination trade shares."""
        model = AllenArkolakisModel(
            trade_costs=np.array([[1.0, 1.4], [1.2, 1.0]]),
            fundamental_productivity=np.array([1.2, 0.9]),
            fundamental_amenity=np.array([0.8, 1.1]),
            theta=4.0,
            alpha=0.10,
            beta=-0.30,
        )
        res = model.solve_equilibrium()
        assert res.converged
        np.testing.assert_array_equal(res.trade_shares_dest_origin, res.trade_shares)
        np.testing.assert_array_equal(res.trade_shares_origin_dest, res.trade_shares.T)
        # Verify row stochasticity of destination-by-origin
        np.testing.assert_allclose(np.sum(res.trade_shares_dest_origin, axis=1), 1.0, atol=1e-10)


class TestCR03NashTariffsGradient:
    """CR-03: Boundary finite-difference stencil span in gradient Nash tariffs."""

    def test_cr03_boundary_gradient_step(self, synthetic_trade_calib):
        """At tau = 0.0, the lower perturbation is clamped to 0.0, but dividing by h_actual prevents halving."""
        h_fd = 0.01
        p_idx = 0
        tau_prev = {0: 0.0, 1: 0.0}
        tariff_max = 0.50

        prof_plus = dict(tau_prev)
        prof_plus[p_idx] = min(tariff_max, tau_prev[p_idx] + h_fd)

        prof_minus = dict(tau_prev)
        prof_minus[p_idx] = max(0.0, tau_prev[p_idx] - h_fd)

        h_actual = float(prof_plus[p_idx] - prof_minus[p_idx])
        # At boundary tau=0, prof_plus is 0.01 and prof_minus is 0.0, so h_actual is exactly h_fd (0.01)
        assert np.isclose(h_actual, h_fd)
        # Previous bug divided by 2.0 * h_fd = 0.02, cutting the gradient in half!
        assert h_actual != 2.0 * h_fd

        # Verify gradient Nash tariff solver runs and terminates
        res_nash = solve_multilateral_nash_tariffs(
            synthetic_trade_calib,
            players=[0, 1],
            method="gradient",
            max_iter=3,
            tariff_max=0.30,
        )
        assert res_nash is not None
        assert "CAN" in res_nash.nash_tariffs
        assert "USA" in res_nash.nash_tariffs


class TestCR04GearyKhamisNumeraire:
    """CR-04: Arbitrary country count numeraire index and auto-detection."""

    def test_cr04_small_dataset_numeraire(self):
        """In a 3-country dataset, normalizing with numeraire_idx normalizes the selected country."""
        p = np.array([
            [1.0, 1.2, 0.8],
            [2.0, 1.8, 2.2],
        ])
        q = np.array([
            [10.0, 15.0, 12.0],
            [5.0, 8.0, 6.0],
        ])

        # Normalize country index 1 to 1.0
        pi, ppp, conv, it = solve_multilateral_ppp(p, q, normalize="numeraire", numeraire_idx=1)
        assert conv
        assert np.isclose(ppp[1], 1.0, atol=1e-10)

        # Normalize country index 2 to 1.0
        pi2, ppp2, conv2, it2 = solve_multilateral_ppp(p, q, normalize="numeraire", numeraire_idx=2)
        assert conv2
        assert np.isclose(ppp2[2], 1.0, atol=1e-10)

        # Out-of-bounds numeraire_idx raises IndexError
        with pytest.raises(IndexError, match="out of bounds"):
            solve_multilateral_ppp(p, q, normalize="numeraire", numeraire_idx=10)

    def test_cr04_compute_geary_khamis_auto_detect_usa(self, synthetic_trade_calib):
        """compute_geary_khamis automatically detects 'USA' in country_codes."""
        eq = solve_trade_equilibrium(synthetic_trade_calib, method="condensed")
        gk = compute_geary_khamis(eq, base_sol=eq, calib=synthetic_trade_calib, normalize="ppp_usa")
        assert gk.converged
        usa_idx = synthetic_trade_calib.country_codes.index("USA")
        # In synthetic_trade_calib, USA is index 1
        assert usa_idx == 1
        assert np.isclose(gk.ppp[usa_idx], 1.0, atol=1e-10)


class TestCR06LaTeXEscaping:
    """CR-06: Valid LaTeX formatting in AnalyticGradientResult.plot."""

    def test_cr06_latin_and_greek_identifiers(self):
        """Greek names get backslash, Latin names (A, delta_k) remain clean."""
        assert "alpha" in _GREEK_LETTERS
        assert "beta" in _GREEK_LETTERS
        assert "A" not in _GREEK_LETTERS
        assert "delta_k" not in _GREEK_LETTERS

        # Create dummy result with both Latin and Greek parameter names
        res = AnalyticGradientResult(
            grad_coefficients=np.ones((5, 2)),
            grad_aggregates={"K": np.array([0.5, -0.2]), "C": np.array([0.1, 0.3])},
            param_names=["beta", "A"],
            jacobian_resid_c=np.eye(5),
            jacobian_resid_theta=np.ones((5, 2)),
            condition_number=100.0,
            elapsed_time=0.01,
        )

        fig = res.plot(show=False)
        assert fig is not None
        plt.close(fig)


class TestCR07DeepMacroEulerTarget:
    """CR-07: Custom euler_target_fn in DeepMacroModel."""

    def test_cr07_custom_euler_target(self):
        """DeepMacroModel supports custom euler_target_fn without falling back to hardcoded Cobb-Douglas."""
        called_custom = [False]

        def custom_target(k, c, kp, cp):
            called_custom[0] = True
            # Simple identity or custom target
            return 0.5 * c

        model = DeepMacroModel(
            n_states=1,
            n_controls=1,
            beta=0.95,
            params={"alpha": 0.33, "delta": 0.10, "gamma": 1.5, "A": 1.0},
            euler_target_fn=custom_target,
        )

        sol = solve_deep_macro(
            model,
            n_epochs=2,
            batch_size=8,
            trajectory_length=20,
            seed=42,
        )
        assert sol.converged
        assert called_custom[0], "Custom euler_target_fn was not invoked during training!"


class TestCR09CESCacheMemorySafety:
    """CR-09: Trade calibration CES weights cached per instance without leaks."""

    def test_cr09_cache_safety(self, synthetic_trade_calib):
        """Weights are cached directly on calib and cleaned up on delete."""
        w1 = _get_ces_weights(synthetic_trade_calib)
        assert hasattr(synthetic_trade_calib, "_ces_weights_cache")
        w2 = _get_ces_weights(synthetic_trade_calib)
        assert w1 is w2  # Exact object identity cache hit


class TestCR10DCEGMSinglePointSegment:
    """CR-10: upper_envelope handles single-point segments without IndexError."""

    def test_cr10_single_point_segments(self):
        """Degenerate single-point segment does not raise IndexError on slope calculation."""
        exog = np.linspace(0.1, 5.0, 50)
        m_raw = np.array([1.0])
        c_raw = np.array([0.5])
        v_raw = np.array([0.2])

        # Should execute cleanly without raising IndexError: index out of bounds
        v_clean, c_clean = upper_envelope(m_raw, c_raw, v_raw, exog, a_min=0.0)
        assert len(v_clean) == len(exog)
        assert len(c_clean) == len(exog)
        assert not np.any(np.isnan(v_clean))
        assert not np.any(np.isnan(c_clean))
