"""Adversarial Stress Test Suite: Multilateral Axioms & Economic Properties (M4).

Authored by m4_challenger_2 to independently and adversarially stress-test:
1. Additive Consistency: World expenditure invariant
     sum_k Nominal GDP_k = sum_m pi_m Q_m = sum_k Real GDP_k
   under natural scale across all benchmark scenarios and synthetic models.
2. Solver Equivalence: Iterative fixed-point vs direct linear solve producing
   identical pi and PPP vectors to within floating-point precision (< 10^-12).
3. Domestic Tariff Invariance: Domestic diagonal elements of tau and tau_fd
   strictly 1.0 across canonical scenarios, adversarial overrides, and sector combinations.
4. Boundary & Inactive Category Handling: Zero consumption, zero investment,
   sparse final demand, global category zero, and negative capital formation.

Conforms strictly to the puremacro Pyodide runtime contract.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from puremacro.trade.data import (
    CANONICAL_COUNTRY_CODES,
    CANONICAL_SECTOR_CODES,
    available_reference_scenarios,
    load_reference_solution,
)
from puremacro.trade.geary_khamis import (
    compute_geary_khamis,
    restore_capital_formation,
    solve_multilateral_ppp,
)
from puremacro.trade.scenarios import (
    CANONICAL_SCENARIO_NAMES,
    SCENARIOS,
    TariffScenario,
    build_tariff_matrices,
)
from puremacro.trade._results import TradeCalibrationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def reference_or_skip(scenario: str) -> dict[str, np.ndarray]:
    """Bundled MATLAB reference arrays for ``scenario``, or a precise skip.

    The arrays ship inside ``puremacro.trade``, so these checks need neither
    MATLAB nor any file outside the installation.  ``t10_54`` is the one
    scenario with no bundled reference (its MATLAB source is a dataless
    placeholder), and only that parametrisation skips.
    """
    try:
        return load_reference_solution(scenario)
    except KeyError:
        pytest.skip(
            f"No bundled reference solution for scenario {scenario!r}; "
            f"bundled scenarios: {available_reference_scenarios()}"
        )


@pytest.fixture(scope="module")
def synthetic_calib_77c() -> TradeCalibrationResult:
    """Minimal calibration result container for 77 countries and 11 sectors."""
    nc, ns, nfd = 77, 11, 3
    return TradeCalibrationResult(
        a=np.zeros((ns, nc, ns, nc)),
        afd=np.zeros((ns, nc, nfd, nc)),
        alpha=np.full((1, ns, nc), 1.0 / 3.0),
        beta=np.ones((1, ns, nc)),
        k_endow=np.ones((1, 1, nc)),
        l_endow=np.ones((1, 1, nc)),
        invforT=np.zeros((1, 1, nc)),
        tax=np.zeros((1, ns, nc)),
        ytot=np.ones((1, ns, nc)),
        country_codes=CANONICAL_COUNTRY_CODES,
        sector_codes=CANONICAL_SECTOR_CODES,
    )


# ---------------------------------------------------------------------------
# 1. Additive Consistency Suite
# ---------------------------------------------------------------------------

class TestAdditiveConsistency:
    """Adversarially verify the world expenditure invariant:
    sum_k Nominal GDP_k = sum_m pi_m Q_m = sum_k Real GDP_k
    under natural scale (normalize='none').
    """

    @pytest.mark.parametrize("scen_name", [
        "base", "t10", "t10_25", "t10_54", "t10_75", "t10_125", "t10_145"
    ])
    @pytest.mark.parametrize("matlab_compat", [True, False])
    @pytest.mark.parametrize("method", ["linear", "iterative"])
    def test_additive_consistency_benchmark_scenarios(
        self, scen_name, matlab_compat, method
    ):
        """Verify additive consistency invariant on all 7 benchmark scenarios."""
        mat = reference_or_skip(scen_name)
        p = mat["pfd_sol"].reshape(3, 77)
        c_raw = mat["c_sol"].reshape(3, 77)
        xn = mat["XN_sol"].ravel()

        q = restore_capital_formation(c_raw, xn, matlab_compat=matlab_compat)

        # Solve Geary-Khamis under natural scale
        tol = 1e-12 if method == "iterative" else 1e-10
        res = compute_geary_khamis(
            equilibrium_sol=p,
            base_sol=q,
            method=method,
            normalize="none",
            tol=tol,
            max_iter=3000,
        )

        nom_gdp = np.sum(p * q, axis=0)
        real_gdp = res.real_gdp
        q_world = np.sum(q, axis=1)

        sum_nom = float(np.sum(nom_gdp))
        sum_real = float(np.sum(real_gdp))
        sum_pi_q = float(np.dot(res.pi, q_world))

        # 1. sum_k Real GDP_k == sum_m pi_m Q_m identically
        assert math.isclose(sum_real, sum_pi_q, rel_tol=1e-12, abs_tol=1e-10), (
            f"Violation: World Real GDP ({sum_real}) != pi @ Q ({sum_pi_q})"
        )

        # 2. sum_k Nominal GDP_k == sum_m pi_m Q_m under natural scale
        assert math.isclose(sum_nom, sum_pi_q, rel_tol=1e-12, abs_tol=1e-10), (
            f"Violation: World Nominal GDP ({sum_nom}) != pi @ Q ({sum_pi_q})"
        )

        # 3. sum_k Nominal GDP_k == sum_k Real GDP_k
        assert math.isclose(sum_nom, sum_real, rel_tol=1e-12, abs_tol=1e-10), (
            f"Violation: World Nominal GDP ({sum_nom}) != World Real GDP ({sum_real})"
        )

    def test_additive_consistency_random_synthetic_stress(self):
        """Stress-test additive consistency across 50 random synthetic economies."""
        rng = np.random.default_rng(42)

        for trial in range(50):
            M = rng.integers(2, 12)
            K = rng.integers(2, 60)

            # Wide dynamic range of prices and quantities
            p = rng.uniform(0.01, 100.0, (M, K))
            q = rng.uniform(0.1, 5000.0, (M, K))

            q_world = np.sum(q, axis=1)
            nom_gdp = np.sum(p * q, axis=0)
            world_nom = float(np.sum(nom_gdp))

            for method in ("linear", "iterative"):
                pi, ppp, conv, _ = solve_multilateral_ppp(
                    p, q, method=method, tol=1e-14, max_iter=3000, normalize="none"
                )
                assert conv, f"Method {method} failed to converge on trial {trial}"

                real_gdp = np.sum(q * pi[:, np.newaxis], axis=0)
                world_real = float(np.sum(real_gdp))
                pi_q = float(np.dot(pi, q_world))

                rel_err_nom_piq = abs(world_nom - pi_q) / world_nom
                rel_err_real_piq = abs(world_real - pi_q) / world_real
                rel_err_nom_real = abs(world_nom - world_real) / world_nom

                assert rel_err_nom_piq < 1e-12, (
                    f"Trial {trial} ({method}): rel err nom vs pi@Q = {rel_err_nom_piq:.2e}"
                )
                assert rel_err_real_piq < 1e-12, (
                    f"Trial {trial} ({method}): rel err real vs pi@Q = {rel_err_real_piq:.2e}"
                )
                assert rel_err_nom_real < 1e-12, (
                    f"Trial {trial} ({method}): rel err nom vs real = {rel_err_nom_real:.2e}"
                )

    def test_algebraic_identity_sum_real_equals_pi_q_under_any_normalization(self):
        """Verify that sum_k Real GDP_k == sum_m pi_m Q_m holds identically for ANY normalization."""
        rng = np.random.default_rng(999)
        p = rng.uniform(0.5, 2.0, (4, 10))
        q = rng.uniform(10.0, 100.0, (4, 10))
        q_world = np.sum(q, axis=1)

        for norm in ("none", "pi1", "ppp_usa"):
            pi, _, _, _ = solve_multilateral_ppp(p, q, method="linear", normalize=norm)
            real_gdp = np.sum(q * pi[:, np.newaxis], axis=0)
            sum_real = float(np.sum(real_gdp))
            sum_pi_q = float(np.dot(pi, q_world))
            assert math.isclose(sum_real, sum_pi_q, rel_tol=1e-14)


# ---------------------------------------------------------------------------
# 2. Solver Equivalence Suite
# ---------------------------------------------------------------------------

class TestSolverEquivalence:
    """Verify that iterative fixed-point and direct linear solve produce
    identical pi and PPP vectors to within floating-point precision (< 10^-12).
    """

    @pytest.mark.parametrize("scen_name", [
        "base", "t10", "t10_25", "t10_54", "t10_75", "t10_125", "t10_145"
    ])
    def test_solver_equivalence_benchmark_scenarios(self, scen_name):
        """Assert < 10^-12 discrepancy between iterative and linear on benchmark data."""
        mat = reference_or_skip(scen_name)
        p = mat["pfd_sol"].reshape(3, 77)
        c_raw = mat["c_sol"].reshape(3, 77)
        xn = mat["XN_sol"].ravel()
        q = restore_capital_formation(c_raw, xn, matlab_compat=True)

        # 1. Linear solve
        pi_lin, ppp_lin, conv_lin, _ = solve_multilateral_ppp(p, q, method="linear")
        assert conv_lin is True

        # 2. Iterative solve with tight precision
        pi_it, ppp_it, conv_it, _ = solve_multilateral_ppp(
            p, q, method="iterative", tol=1e-14, max_iter=3000
        )
        assert conv_it is True

        # Assert absolute and relative error < 10^-12
        max_abs_pi = float(np.max(np.abs(pi_it - pi_lin)))
        max_abs_ppp = float(np.max(np.abs(ppp_it - ppp_lin)))
        max_rel_pi = float(np.max(np.abs(pi_it - pi_lin) / pi_lin))
        max_rel_ppp = float(np.max(np.abs(ppp_it - ppp_lin) / ppp_lin))

        assert max_abs_pi < 1e-12, f"Scenario {scen_name}: max |pi_it - pi_lin| = {max_abs_pi:.2e}"
        assert max_abs_ppp < 1e-12, f"Scenario {scen_name}: max |ppp_it - ppp_lin| = {max_abs_ppp:.2e}"
        assert max_rel_pi < 1e-12, f"Scenario {scen_name}: max rel diff pi = {max_rel_pi:.2e}"
        assert max_rel_ppp < 1e-12, f"Scenario {scen_name}: max rel diff ppp = {max_rel_ppp:.2e}"

    def test_solver_equivalence_random_synthetic_stress(self):
        """Assert < 10^-12 solver equivalence across 50 random synthetic matrices."""
        rng = np.random.default_rng(2026)

        max_diffs_pi = []
        max_diffs_ppp = []

        for trial in range(50):
            M = rng.integers(2, 15)
            K = rng.integers(2, 80)
            p = rng.uniform(0.1, 10.0, (M, K))
            q = rng.uniform(1.0, 500.0, (M, K))

            pi_lin, ppp_lin, _, _ = solve_multilateral_ppp(p, q, method="linear")
            pi_it, ppp_it, conv, _ = solve_multilateral_ppp(
                p, q, method="iterative", tol=1e-14, max_iter=4000
            )
            assert conv, f"Iterative failed to converge on synthetic trial {trial}"

            max_diffs_pi.append(float(np.max(np.abs(pi_it - pi_lin))))
            max_diffs_ppp.append(float(np.max(np.abs(ppp_it - ppp_lin))))

        assert np.max(max_diffs_pi) < 1e-12, (
            f"Max diff pi across all synthetic trials: {np.max(max_diffs_pi):.2e} >= 1e-12"
        )
        assert np.max(max_diffs_ppp) < 1e-12, (
            f"Max diff ppp across all synthetic trials: {np.max(max_diffs_ppp):.2e} >= 1e-12"
        )

    def test_operator_spectral_radius_and_nullspace_structure(self):
        """Verify the Geary-Khamis transition matrix A satisfies Perron-Frobenius properties."""
        rng = np.random.default_rng(777)
        p = rng.uniform(0.5, 2.5, (5, 20))
        q = rng.uniform(10.0, 100.0, (5, 20))

        q_world = np.sum(q, axis=1)
        expenditure = np.sum(p * q, axis=0)

        term1 = (p * q) / q_world[:, np.newaxis]
        term2 = q / expenditure[np.newaxis, :]
        A = term1 @ term2.T  # (M, M)

        eigvals = np.linalg.eigvals(A)
        # Sort by magnitude descending
        sorted_eigs = sorted(eigvals, key=lambda x: abs(x), reverse=True)

        # Dominant eigenvalue must be strictly 1.0 (Perron-Frobenius root)
        assert math.isclose(abs(sorted_eigs[0]), 1.0, rel_tol=1e-12), (
            f"Dominant eigenvalue {sorted_eigs[0]} != 1.0"
        )
        # Second eigenvalue must be strictly < 1.0 (strictly contractive)
        if len(sorted_eigs) > 1:
            assert abs(sorted_eigs[1]) < 1.0 - 1e-4, (
                f"Second eigenvalue {sorted_eigs[1]} not strictly subdominant"
            )


# ---------------------------------------------------------------------------
# 3. Domestic Tariff Invariance Suite
# ---------------------------------------------------------------------------

class TestDomesticTariffInvariance:
    """Verify that domestic diagonal elements of tau and tau_fd remain strictly
    1.0 across all scenarios, adversarial inputs, and sector combinations.
    """

    def test_canonical_scenarios_domestic_diagonal(
        self, synthetic_calib_77c: TradeCalibrationResult
    ):
        """Verify strict 1.0 domestic invariance for all pre-defined canonical scenarios."""
        calib = synthetic_calib_77c
        nc = calib.n_countries

        for scen_name in SCENARIOS:
            tau, tau_fd, _, _ = build_tariff_matrices(scen_name, calib)

            for k in range(nc):
                # Intermediate domestic sales: tau[s, k, s', k] == 1.0
                assert np.all(tau[:, k, :, k] == 1.0), (
                    f"Scenario '{scen_name}': Domestic intermediate diagonal corrupted for country {k}!"
                )
                # Final demand domestic sales: tau_fd[s, k, ifd, k] == 1.0
                assert np.all(tau_fd[:, k, :, k] == 1.0), (
                    f"Scenario '{scen_name}': Domestic final demand diagonal corrupted for country {k}!"
                )

    def test_adversarial_domestic_overrides(
        self, synthetic_calib_77c: TradeCalibrationResult
    ):
        """Adversarial attempt to force tariffs on domestic transactions."""
        calib = synthetic_calib_77c
        nc = calib.n_countries

        adversarial_scenario = TariffScenario(
            name="hostile_domestic_attack",
            description="Intentionally injecting domestic tariffs across all layers.",
            default_us_tariff=0.45,
            us_import_tariffs={"USA": 0.99, "CAN": 0.25},
            bilateral_tariffs={
                ("USA", "USA"): 0.75,
                ("CAN", "CAN"): 0.60,
                ("CHN", "CHN"): 0.85,
                ("DEU", "DEU"): 0.50,
            },
            sectoral_tariffs={
                ("USA", "USA", "MANU"): 1.50,
                ("CAN", "CAN", "AGRI"): 2.00,
                ("CHN", "CHN", "CONS"): 3.00,
            },
        )

        tau, tau_fd, _, _ = build_tariff_matrices(adversarial_scenario, calib)

        for k in range(nc):
            assert np.all(tau[:, k, :, k] == 1.0), (
                f"Adversarial breach: Domestic intermediate diagonal corrupted for country {k}!"
            )
            assert np.all(tau_fd[:, k, :, k] == 1.0), (
                f"Adversarial breach: Domestic final demand diagonal corrupted for country {k}!"
            )

    def test_invalid_negative_and_nonfinite_tariffs_rejected(self):
        """Verify negative, NaN, and infinite tariff rates are rejected defensively."""
        with pytest.raises(ValueError, match="finite non-negative"):
            TariffScenario(name="neg_default", default_us_tariff=-0.10)

        with pytest.raises(ValueError, match="finite non-negative"):
            TariffScenario(name="nan_default", default_us_tariff=float("nan"))

        with pytest.raises(ValueError, match="finite non-negative"):
            TariffScenario(name="inf_default", default_us_tariff=float("inf"))

        with pytest.raises(ValueError, match="finite non-negative"):
            TariffScenario(name="neg_bilateral", bilateral_tariffs={("CAN", "USA"): -0.25})


# ---------------------------------------------------------------------------
# 4. Boundary & Inactive Category Handling Suite
# ---------------------------------------------------------------------------

class TestBoundaryAndInactiveHandling:
    """Stress-test Geary-Khamis solver with synthetic data containing zero
    consumption, zero investment, sparse final demand, and real-world boundary cases.
    """

    def test_zero_investment_single_country(self):
        """Verify graceful handling and solver equivalence when country has zero investment."""
        p = np.array([
            [1.0, 1.1, 0.95],
            [1.2, 0.9, 1.05],
            [0.8, 1.0, 1.15],
        ])
        # Country 0 has ZERO investment (q[1, 0] = 0.0)
        q = np.array([
            [100.0, 120.0, 150.0],
            [0.0,    60.0,  80.0],
            [20.0,   25.0,  30.0],
        ])

        pi_lin, ppp_lin, conv_lin, _ = solve_multilateral_ppp(p, q, method="linear")
        pi_it, ppp_it, conv_it, _ = solve_multilateral_ppp(p, q, method="iterative", tol=1e-14)

        assert conv_lin and conv_it
        assert np.max(np.abs(pi_it - pi_lin)) < 1e-12
        assert np.max(np.abs(ppp_it - ppp_lin)) < 1e-12

        # Verify additive consistency
        nom_gdp = np.sum(p * q, axis=0)
        real_gdp = np.sum(q * pi_lin[:, np.newaxis], axis=0)
        assert math.isclose(float(np.sum(nom_gdp)), float(np.sum(real_gdp)), rel_tol=1e-12)

    def test_zero_consumption_single_country(self):
        """Verify graceful handling and solver equivalence when country has zero consumption."""
        p = np.array([
            [1.0, 1.1, 0.95],
            [1.2, 0.9, 1.05],
            [0.8, 1.0, 1.15],
        ])
        # Country 1 has ZERO consumption (q[0, 1] = 0.0)
        q = np.array([
            [100.0,   0.0, 150.0],
            [50.0,   60.0,  80.0],
            [20.0,   25.0,  30.0],
        ])

        pi_lin, ppp_lin, conv_lin, _ = solve_multilateral_ppp(p, q, method="linear")
        pi_it, ppp_it, conv_it, _ = solve_multilateral_ppp(p, q, method="iterative", tol=1e-14)

        assert conv_lin and conv_it
        assert np.max(np.abs(pi_it - pi_lin)) < 1e-12
        assert np.max(np.abs(ppp_it - ppp_lin)) < 1e-12

    def test_sparse_final_demand_multiple_zeros(self):
        """Verify solver handles sparse final demand with multiple zero entries gracefully."""
        p = np.array([
            [1.0, 1.1, 0.9, 1.05],
            [1.2, 0.9, 1.1, 0.95],
            [0.8, 1.0, 1.2, 1.00],
        ])
        # Diagonal-like sparsity in final demand
        q = np.array([
            [100.0,   0.0, 150.0,  80.0],  # Country 1 zero in Cat 0
            [ 50.0,  60.0,   0.0,  70.0],  # Country 2 zero in Cat 1
            [  0.0,  25.0,  30.0,   0.0],  # Countries 0 and 3 zero in Cat 2
        ])

        pi_lin, ppp_lin, conv_lin, _ = solve_multilateral_ppp(p, q, method="linear")
        pi_it, ppp_it, conv_it, _ = solve_multilateral_ppp(p, q, method="iterative", tol=1e-14)

        assert conv_lin and conv_it
        assert np.max(np.abs(pi_it - pi_lin)) < 1e-12
        assert np.max(np.abs(ppp_it - ppp_lin)) < 1e-12

    def test_entire_category_zero_world_consumption_raises(self):
        """Verify that an inactive category with Q_m == 0 raises a clean ValueError."""
        p = np.array([[1.0, 1.1], [1.2, 0.9], [0.8, 1.0]])
        # Category 1 has ZERO across ALL countries
        q = np.array([[100.0, 120.0], [0.0, 0.0], [20.0, 30.0]])

        with pytest.raises(ValueError, match="strictly positive for all categories"):
            solve_multilateral_ppp(p, q, method="linear")

        with pytest.raises(ValueError, match="strictly positive for all categories"):
            solve_multilateral_ppp(p, q, method="iterative")

    def test_negative_capital_formation_real_world_bgr(self):
        """Verify stability on real-world negative gross capital formation in Bulgaria (BGR)."""
        mat = reference_or_skip("base")
        p = mat["pfd_sol"].reshape(3, 77)
        c_raw = mat["c_sol"].reshape(3, 77)
        xn = mat["XN_sol"].ravel()

        q = restore_capital_formation(c_raw, xn, matlab_compat=True)
        # Verify Bulgaria has negative restored investment:
        bgr_idx = 5  # CANONICAL_COUNTRY_CODES[5] == "BGR"
        assert q[1, bgr_idx] < 0.0, "BGR capital formation was expected to be negative"

        # Solve and verify convergence and equivalence
        pi_lin, ppp_lin, conv_lin, _ = solve_multilateral_ppp(p, q, method="linear")
        pi_it, ppp_it, conv_it, _ = solve_multilateral_ppp(p, q, method="iterative", tol=1e-14)

        assert conv_lin and conv_it
        assert np.max(np.abs(pi_it - pi_lin)) < 1e-12
        assert np.max(np.abs(ppp_it - ppp_lin)) < 1e-12
        assert ppp_lin[bgr_idx] > 0.0, "BGR PPP exchange rate must be strictly positive"

    def test_near_zero_quantities_and_connectedness_limit(self):
        """Stress-test numerical stability under high dynamic range and inspect connectedness limit."""
        p = np.array([[1.0, 1.0], [1.0, 1.0]])
        # Test moderate dynamic range (10^4:1)
        q_mod = np.array([[1000.0, 0.1], [0.1, 1000.0]])
        pi_lin, ppp_lin, conv_lin, _ = solve_multilateral_ppp(p, q_mod, method="linear")
        pi_it, ppp_it, conv_it, _ = solve_multilateral_ppp(p, q_mod, method="iterative", tol=1e-14)

        assert conv_lin and conv_it
        assert np.max(np.abs(pi_it - pi_lin)) < 1e-12
        assert np.max(np.abs(ppp_it - ppp_lin)) < 1e-12

        # In extreme disconnect (off-diagonal under machine eps), iterative solves correctly
        q_ext = np.array([[1e6, 1e-15], [1e-15, 1e6]])
        pi_it_ext, ppp_it_ext, conv_it_ext, _ = solve_multilateral_ppp(p, q_ext, method="iterative")
        assert conv_it_ext
        assert np.allclose(pi_it_ext, [1.0, 1.0], atol=1e-10)
        assert np.allclose(ppp_it_ext, [1.0, 1.0], atol=1e-10)
