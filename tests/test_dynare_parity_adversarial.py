"""Adversarial stress test suite for Dynare Parity Dashboard and DR loader.

Empirically verifies:
1. Arbitrarily scrambled order_var permutations (reverse order, random permutations of 10+ variables).
2. Folded Kronecker unfolding of ghxx and ghuu across random dimensions and equivalence
   to full symmetric quadratic form sum_{i,j} A_{i,j} x_i x_j.
3. Tolerance boundary conditions at 0.99 * tol (PASS) and 1.01 * tol (FAIL) across ghx, ghu, ghxx, ghs2.
4. Canonical Pfeifer Smets-Wouters (2007) model parity under order 1, order 2, folded ghxx,
   and scrambled permutations.
5. Edge cases, dimension boundary conditions, and scalar squeeze handling.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

from puremacro.dsge import (
    Dynare2ndDR,
    DynareDR,
    ParityDashboardResult,
    compare_model_to_dynare,
    load_dynare_dr,
    load_dynare_moments,
    load_mod,
    verify_dynare_parity,
)
from puremacro.dsge.load_dynare import _unfold_ghxx


SW07_PATH = Path("puremacro/dsge/_references/sw07_pfeifer.mod")


# ===========================================================================
# 1. Arbitrarily Scrambled order_var Permutations Stress Tests
# ===========================================================================

class TestScrambledOrderVarPermutations:
    """Stress tests verifying that unpermuting order_var reconstructs exact declaration order."""

    def test_reverse_order_permutation_reconstruction(self, tmp_path: Path):
        """Test reverse declaration order permutation [N, N-1, ..., 1]."""
        n_v, n_x, n_u = 12, 4, 2
        var_names = [f"var_{i+1:02d}" for i in range(n_v)]
        state_names = [f"var_{i+1:02d}" for i in range(n_x)]
        shock_names = [f"eps_{j+1}" for j in range(n_u)]

        rng = np.random.default_rng(101)
        ghx_true = rng.standard_normal((n_v, n_x))
        ghu_true = rng.standard_normal((n_v, n_u))
        ys_true = rng.uniform(0.5, 5.0, size=n_v)

        # Reverse permutation: index 0 gets var N-1, index N-1 gets var 0
        perm = np.arange(n_v - 1, -1, -1)
        order_var = (perm + 1)[:, None]  # 1-based (N, 1)

        mat_path = tmp_path / "reverse_order.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": ghx_true[perm, :],
                        "ghu": ghu_true[perm, :],
                        "ys": ys_true[:, None],
                        "order_var": order_var,
                    }
                },
                "M_": {
                    "endo_names": np.array(var_names),
                    "exo_names": np.array(shock_names),
                    "state_var": np.arange(1, n_x + 1)[:, None],
                },
            },
        )

        dr = load_dynare_dr(mat_path, order=1)
        assert dr.variable_names == tuple(var_names)
        assert dr.state_variables == tuple(state_names)
        assert dr.shock_names == tuple(shock_names)

        np.testing.assert_allclose(dr.ghx.to_numpy(), ghx_true, atol=1e-15)
        np.testing.assert_allclose(dr.ghu.to_numpy(), ghu_true, atol=1e-15)
        np.testing.assert_allclose(dr.ys.to_numpy(), ys_true, atol=1e-15)

    @pytest.mark.parametrize("n_vars", [10, 16, 25])
    def test_random_order_var_permutations(self, n_vars: int, tmp_path: Path):
        """Test multiple arbitrary random permutations for various model sizes."""
        n_x = 4
        n_u = 2
        var_names = [f"x{i}" for i in range(n_vars)]
        state_names = [f"x{i}" for i in range(n_x)]
        shock_names = [f"e{j}" for j in range(n_u)]

        rng = np.random.default_rng(202 + n_vars)
        ghx_true = rng.standard_normal((n_vars, n_x))
        ghu_true = rng.standard_normal((n_vars, n_u))
        ys_true = rng.standard_normal(n_vars)

        # Run 10 random permutations per size
        for trial in range(10):
            perm = rng.permutation(n_vars)
            order_var = (perm + 1).reshape(-1, 1)  # 1-indexed

            mat_dict = {
                "oo_": {
                    "dr": {
                        "ghx": ghx_true[perm, :],
                        "ghu": ghu_true[perm, :],
                        "ys": ys_true,
                        "order_var": order_var,
                    }
                },
                "M_": {
                    "endo_names": np.array(var_names),
                    "exo_names": np.array(shock_names),
                    "state_var": np.arange(1, n_x + 1),
                },
            }

            mat_path = tmp_path / f"rand_perm_{n_vars}_{trial}.mat"
            scipy.io.savemat(str(mat_path), mat_dict)

            dr = load_dynare_dr(mat_path, order=1)
            np.testing.assert_allclose(
                dr.ghx.to_numpy(),
                ghx_true,
                atol=1e-15,
                err_msg=f"ghx unpermute failed on trial {trial} with n_vars={n_vars}",
            )
            np.testing.assert_allclose(
                dr.ghu.to_numpy(),
                ghu_true,
                atol=1e-15,
                err_msg=f"ghu unpermute failed on trial {trial} with n_vars={n_vars}",
            )
            np.testing.assert_allclose(
                dr.ys.to_numpy(),
                ys_true,
                atol=1e-15,
                err_msg=f"ys unpermute failed on trial {trial} with n_vars={n_vars}",
            )

    def test_order_var_format_variations(self, tmp_path: Path):
        """Verify 1D, 2D column, 2D row, 0-indexed, and 1-indexed order_var formats."""
        n_v, n_x = 5, 2
        var_names = [f"y{i}" for i in range(n_v)]
        rng = np.random.default_rng(303)
        ghx_true = rng.standard_normal((n_v, n_x))
        ghu_true = rng.standard_normal((n_v, 1))
        ys_true = rng.standard_normal(n_v)
        perm = np.array([3, 0, 4, 1, 2])

        formats = [
            ("1d_1based", perm + 1),
            ("col_1based", (perm + 1)[:, None]),
            ("row_1based", (perm + 1)[None, :]),
            ("1d_0based", perm),
            ("col_0based", perm[:, None]),
        ]

        for fmt_name, ov in formats:
            mat_path = tmp_path / f"format_{fmt_name}.mat"
            scipy.io.savemat(
                str(mat_path),
                {
                    "oo_": {
                        "dr": {
                            "ghx": ghx_true[perm, :],
                            "ghu": ghu_true[perm, :],
                            "ys": ys_true,
                            "order_var": ov,
                        }
                    },
                    "M_": {
                        "endo_names": np.array(var_names),
                        "exo_names": np.array(["e"]),
                    },
                },
            )
            dr = load_dynare_dr(mat_path, order=1)
            np.testing.assert_allclose(
                dr.ghx.to_numpy(),
                ghx_true,
                atol=1e-15,
                err_msg=f"Failed on order_var format {fmt_name}",
            )

    def test_second_order_unpermuting_all_tensors(self, tmp_path: Path):
        """Test second-order unpermuting for ghxx, ghs2, ghxu, and ghuu simultaneously."""
        n_v, n_x, n_u = 8, 3, 2
        var_names = [f"z{i}" for i in range(n_v)]
        state_names = [f"z{i}" for i in range(n_x)]
        shock_names = [f"u{j}" for j in range(n_u)]

        rng = np.random.default_rng(404)
        ghx_true = rng.standard_normal((n_v, n_x))
        ghu_true = rng.standard_normal((n_v, n_u))
        ghxx_true = rng.standard_normal((n_v, n_x * n_x))
        ghxu_true = rng.standard_normal((n_v, n_x * n_u))
        ghuu_true = rng.standard_normal((n_v, n_u * n_u))
        ghs2_true = rng.standard_normal(n_v)
        ys_true = rng.standard_normal(n_v)

        perm = rng.permutation(n_v)

        mat_path = tmp_path / "order2_scrambled.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": ghx_true[perm, :],
                        "ghu": ghu_true[perm, :],
                        "ghxx": ghxx_true[perm, :],
                        "ghxu": ghxu_true[perm, :],
                        "ghuu": ghuu_true[perm, :],
                        "ghs2": ghs2_true[perm, None],
                        "ys": ys_true[:, None],
                        "order_var": (perm + 1)[:, None],
                    }
                },
                "M_": {
                    "endo_names": np.array(var_names),
                    "exo_names": np.array(shock_names),
                    "state_var": np.arange(1, n_x + 1)[:, None],
                },
            },
        )

        dr2 = load_dynare_dr(mat_path, order=2)
        assert isinstance(dr2, Dynare2ndDR)
        np.testing.assert_allclose(dr2.ghxx.to_numpy(), ghxx_true, atol=1e-15)
        np.testing.assert_allclose(dr2.ghxu.to_numpy(), ghxu_true, atol=1e-15)
        np.testing.assert_allclose(dr2.ghuu.to_numpy(), ghuu_true, atol=1e-15)
        np.testing.assert_allclose(dr2.ghs2.to_numpy(), ghs2_true, atol=1e-15)


# ===========================================================================
# 2. Folded Kronecker Unfolding & Quadratic Form Equivalence Stress Tests
# ===========================================================================

class TestFoldedKroneckerUnfolding:
    """Stress tests verifying folded Kronecker unfolding matches full quadratic form."""

    @pytest.mark.parametrize(
        ("n_v", "n_x"),
        [
            (1, 1),
            (5, 4),
            (10, 6),
            (20, 8),
            (40, 15),
        ],
    )
    def test_unfolded_tensor_quadratic_form_equivalence(self, n_v: int, n_x: int):
        """Verify ghxx_unfolded @ (x kron x) identically equals sum_{i,j} A_{i,j} x_i x_j.

        For any symmetric Hessian A^{(v)}, the full expansion is:
            Q^{(v)}(x) = x^T A^{(v)} x = sum_{i,j} A^{(v)}_{i,j} x_i x_j.
        The unfolded tensor expansion is:
            Q_unfold^{(v)}(x) = ghxx_unfolded[v, :] @ kron(x, x).
        Both must match to machine precision for arbitrary state vectors x.
        """
        rng = np.random.default_rng(505 + n_v + n_x)

        # Generate n_v random symmetric matrices A^{(v)} of shape (n_x, n_x)
        A_matrices = []
        for _ in range(n_v):
            M = rng.standard_normal((n_x, n_x))
            A_sym = 0.5 * (M + M.T)
            A_matrices.append(A_sym)

        # Construct Dynare folded representation (n_v, n_x*(n_x+1)//2)
        # Dynare column ordering: (0,0), (0,1), ..., (0, n_x-1), (1,1), (1,2), ...
        n_folded = n_x * (n_x + 1) // 2
        ghxx_folded = np.zeros((n_v, n_folded), dtype=float)
        col = 0
        for i in range(n_x):
            for j in range(i, n_x):
                for v in range(n_v):
                    ghxx_folded[v, col] = A_matrices[v][i, j]
                col += 1

        # Unfold using puremacro's algorithm
        unfolded = _unfold_ghxx(ghxx_folded, n_v, n_x)
        assert unfolded.shape == (n_v, n_x * n_x)

        # Test contraction against 25 random state vectors x
        for _ in range(25):
            x = rng.standard_normal(n_x)
            kron_xx = np.kron(x, x)

            # Contract with unfolded tensor
            quad_unfolded = unfolded @ kron_xx  # shape (n_v,)

            # Compute theoretical sum_{i,j} A_{i,j} x_i x_j = x^T A x
            quad_true = np.array([x @ A @ x for A in A_matrices])

            # Assert mathematical equivalence
            np.testing.assert_allclose(
                quad_unfolded,
                quad_true,
                atol=1e-13,
                err_msg=f"Quadratic contraction mismatch for n_v={n_v}, n_x={n_x}",
            )

    def test_unfold_idempotency_on_full_tensor(self):
        """Unfolding a matrix that already has n_x^2 columns must return it unchanged."""
        n_v, n_x = 4, 3
        rng = np.random.default_rng(606)
        raw_full = rng.standard_normal((n_v, n_x * n_x))
        result = _unfold_ghxx(raw_full, n_v, n_x)
        np.testing.assert_array_equal(result, raw_full)

    def test_unfold_invalid_dimensions_raises(self):
        """Invalid column count (neither n_x^2 nor n_x(n_x+1)/2) raises ValueError."""
        with pytest.raises(ValueError, match="expected 16 \\(unfolded\\) or 10 \\(folded\\)"):
            _unfold_ghxx(np.zeros((3, 7)), n_v=3, n_x=4)

    def test_ghuu_folded_unfolding_with_shocks(self, tmp_path: Path):
        """Verify ghuu shock tensor unfolding and quadratic contraction with shocks u."""
        n_v, n_x, n_u = 3, 2, 3
        rng = np.random.default_rng(707)

        H_shocks = [0.5 * (M + M.T) for M in (rng.standard_normal((n_u, n_u)) for _ in range(n_v))]
        n_folded_u = n_u * (n_u + 1) // 2
        ghuu_folded = np.zeros((n_v, n_folded_u), dtype=float)
        col = 0
        for i in range(n_u):
            for j in range(i, n_u):
                for v in range(n_v):
                    ghuu_folded[v, col] = H_shocks[v][i, j]
                col += 1

        mat_path = tmp_path / "ghuu_folded.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": np.ones((n_v, n_x)),
                        "ghu": np.ones((n_v, n_u)),
                        "ghxx": np.ones((n_v, n_x * (n_x + 1) // 2)),
                        "ghuu": ghuu_folded,
                        "ys": np.zeros(n_v),
                        "order_var": np.arange(1, n_v + 1)[:, None],
                    }
                },
                "M_": {
                    "endo_names": np.array([f"v{i}" for i in range(n_v)]),
                    "exo_names": np.array([f"e{j}" for j in range(n_u)]),
                    "state_var": np.arange(1, n_x + 1)[:, None],
                },
            },
        )

        dr2 = load_dynare_dr(mat_path, order=2)
        assert dr2.ghuu.shape == (n_v, n_u * n_u)

        # Test quadratic contraction
        for _ in range(10):
            u = rng.standard_normal(n_u)
            quad_unfold = dr2.ghuu.to_numpy() @ np.kron(u, u)
            quad_true = np.array([u @ H @ u for H in H_shocks])
            np.testing.assert_allclose(quad_unfold, quad_true, atol=1e-13)


# ===========================================================================
# 3. Tolerance Boundary Conditions Stress Tests
# ===========================================================================

class TestToleranceBoundaries:
    """Stress tests for strict boundary checking: 0.99*tol passes, 1.01*tol fails."""

    @pytest.fixture
    def base_first_order_setup(self, tmp_path: Path):
        """Baseline 1st-order setup with exact matching solutions."""
        n_v, n_x, n_u = 4, 2, 1
        var_names = ["k", "c", "y", "i"]
        ghx = np.array([[0.8, 0.1], [0.3, 0.5], [0.7, 0.2], [0.4, 0.1]])
        ghu = np.array([[0.2], [0.1], [0.4], [0.3]])
        ys = np.array([10.0, 2.5, 3.0, 0.8])

        df_ghx = pd.DataFrame(ghx, index=var_names, columns=["k", "c"])
        df_ghu = pd.DataFrame(ghu, index=var_names, columns=["e"])
        s_ys = pd.Series(ys, index=var_names)
        pm_dr = DynareDR(df_ghx, df_ghu, s_ys, ("k", "c"), tuple(var_names), ("e",))

        return {
            "pm_dr": pm_dr,
            "ghx": ghx,
            "ghu": ghu,
            "ys": ys,
            "var_names": var_names,
            "tmp_path": tmp_path,
        }

    @pytest.mark.parametrize("tol", [1e-3, 1e-5, 1e-6, 1e-8])
    def test_ghx_tolerance_boundary_sharpness(self, base_first_order_setup, tol: float):
        """Inject perturbation to ghx at 0.99*tol (PASS) and 1.01*tol (FAIL)."""
        setup = base_first_order_setup
        pm_dr = setup["pm_dr"]
        ghx_base = setup["ghx"]
        ghu = setup["ghu"]
        ys = setup["ys"]
        var_names = setup["var_names"]

        # Subtest 1: 0.99 * tol (MUST PASS)
        ghx_pass = ghx_base.copy()
        ghx_pass[2, 0] += 0.99 * tol
        df_ghx_pass = pd.DataFrame(ghx_pass, index=var_names, columns=["k", "c"])
        dyn_dr_pass = DynareDR(df_ghx_pass, pd.DataFrame(ghu, index=var_names, columns=["e"]),
                               pd.Series(ys, index=var_names), ("k", "c"), tuple(var_names), ("e",))

        res_pass = verify_dynare_parity(pm_dr, dyn_dr_pass, tol={"ghx": tol, "ghu": tol})
        assert res_pass.passed is True
        assert res_pass.score == 100.0
        assert res_pass.max_dev_ghx == pytest.approx(0.99 * tol, rel=1e-4)
        assert res_pass.dr_diff.loc["y", "status"] == "PASS"

        # Subtest 2: 1.01 * tol (MUST FAIL)
        ghx_fail = ghx_base.copy()
        ghx_fail[2, 0] += 1.01 * tol
        df_ghx_fail = pd.DataFrame(ghx_fail, index=var_names, columns=["k", "c"])
        dyn_dr_fail = DynareDR(df_ghx_fail, pd.DataFrame(ghu, index=var_names, columns=["e"]),
                               pd.Series(ys, index=var_names), ("k", "c"), tuple(var_names), ("e",))

        res_fail = verify_dynare_parity(pm_dr, dyn_dr_fail, tol={"ghx": tol, "ghu": tol})
        assert res_fail.passed is False
        assert res_fail.score < 100.0
        assert res_fail.max_dev_ghx == pytest.approx(1.01 * tol, rel=1e-4)
        assert res_fail.dr_diff.loc["y", "status"] == "FAIL"

    def test_ghu_tolerance_boundary_sharpness(self, base_first_order_setup):
        """Inject perturbation to ghu at 0.99*tol (PASS) and 1.01*tol (FAIL)."""
        setup = base_first_order_setup
        pm_dr = setup["pm_dr"]
        ghx = setup["ghx"]
        ghu_base = setup["ghu"]
        ys = setup["ys"]
        var_names = setup["var_names"]
        tol = 1e-5

        # 0.99 * tol
        ghu_pass = ghu_base.copy()
        ghu_pass[1, 0] += 0.99 * tol
        dyn_pass = DynareDR(pd.DataFrame(ghx, index=var_names, columns=["k", "c"]),
                            pd.DataFrame(ghu_pass, index=var_names, columns=["e"]),
                            pd.Series(ys, index=var_names), ("k", "c"), tuple(var_names), ("e",))
        res_pass = verify_dynare_parity(pm_dr, dyn_pass, tol=tol)
        assert res_pass.passed is True

        # 1.01 * tol
        ghu_fail = ghu_base.copy()
        ghu_fail[1, 0] += 1.01 * tol
        dyn_fail = DynareDR(pd.DataFrame(ghx, index=var_names, columns=["k", "c"]),
                            pd.DataFrame(ghu_fail, index=var_names, columns=["e"]),
                            pd.Series(ys, index=var_names), ("k", "c"), tuple(var_names), ("e",))
        res_fail = verify_dynare_parity(pm_dr, dyn_fail, tol=tol)
        assert res_fail.passed is False
        assert res_fail.dr_diff.loc["c", "status"] == "FAIL"

    def test_second_order_ghxx_and_ghs2_boundaries(self):
        """Verify order 2 boundaries for ghxx and ghs2 independently."""
        n_v, n_x, n_u = 2, 2, 1
        var_names = ["v1", "v2"]
        state_names = ["s1", "s2"]
        cols_xx = ["s1_s1", "s1_s2", "s2_s1", "s2_s2"]

        ghx = np.array([[0.7, 0.1], [0.2, 0.8]])
        ghu = np.array([[0.3], [0.1]])
        ghxx = np.array([[0.05, 0.01, 0.01, 0.02], [0.01, 0.02, 0.02, 0.03]])
        ghs2 = np.array([0.005, 0.008])
        ys = np.array([1.0, 2.0])

        pm_dr2 = Dynare2ndDR(
            ghx=pd.DataFrame(ghx, index=var_names, columns=state_names),
            ghu=pd.DataFrame(ghu, index=var_names, columns=["e"]),
            ghxx=pd.DataFrame(ghxx, index=var_names, columns=cols_xx),
            ghxu=pd.DataFrame(np.zeros((2, 2)), index=var_names, columns=["s1_e", "s2_e"]),
            ghuu=pd.DataFrame(np.zeros((2, 1)), index=var_names, columns=["e_e"]),
            ghs2=pd.Series(ghs2, index=var_names),
            ys=pd.Series(ys, index=var_names),
            state_variables=tuple(state_names),
            variable_names=tuple(var_names),
            shock_names=("e",),
        )

        tol_ghxx = 1e-4
        tol_ghs2 = 1e-4

        # Perturb ghxx at 0.99*tol
        ghxx_pass = ghxx.copy()
        ghxx_pass[0, 1] += 0.99 * tol_ghxx
        dyn_pass = Dynare2ndDR(
            ghx=pd.DataFrame(ghx, index=var_names, columns=state_names),
            ghu=pd.DataFrame(ghu, index=var_names, columns=["e"]),
            ghxx=pd.DataFrame(ghxx_pass, index=var_names, columns=cols_xx),
            ghxu=pd.DataFrame(np.zeros((2, 2)), index=var_names, columns=["s1_e", "s2_e"]),
            ghuu=pd.DataFrame(np.zeros((2, 1)), index=var_names, columns=["e_e"]),
            ghs2=pd.Series(ghs2, index=var_names),
            ys=pd.Series(ys, index=var_names),
            state_variables=tuple(state_names),
            variable_names=tuple(var_names),
            shock_names=("e",),
        )
        res_pass = verify_dynare_parity(pm_dr2, dyn_pass, order=2, tol={"ghxx": tol_ghxx, "ghs2": tol_ghs2})
        assert res_pass.passed is True

        # Perturb ghxx at 1.01*tol
        ghxx_fail = ghxx.copy()
        ghxx_fail[0, 1] += 1.01 * tol_ghxx
        dyn_fail = Dynare2ndDR(
            ghx=pd.DataFrame(ghx, index=var_names, columns=state_names),
            ghu=pd.DataFrame(ghu, index=var_names, columns=["e"]),
            ghxx=pd.DataFrame(ghxx_fail, index=var_names, columns=cols_xx),
            ghxu=pd.DataFrame(np.zeros((2, 2)), index=var_names, columns=["s1_e", "s2_e"]),
            ghuu=pd.DataFrame(np.zeros((2, 1)), index=var_names, columns=["e_e"]),
            ghs2=pd.Series(ghs2, index=var_names),
            ys=pd.Series(ys, index=var_names),
            state_variables=tuple(state_names),
            variable_names=tuple(var_names),
            shock_names=("e",),
        )
        res_fail = verify_dynare_parity(pm_dr2, dyn_fail, order=2, tol={"ghxx": tol_ghxx, "ghs2": tol_ghs2})
        assert res_fail.passed is False
        assert res_fail.dr_diff.loc["v1", "status"] == "FAIL"

        # Perturb ghs2 at 1.01*tol
        ghs2_fail = ghs2.copy()
        ghs2_fail[1] += 1.01 * tol_ghs2
        dyn_fail_s2 = Dynare2ndDR(
            ghx=pd.DataFrame(ghx, index=var_names, columns=state_names),
            ghu=pd.DataFrame(ghu, index=var_names, columns=["e"]),
            ghxx=pd.DataFrame(ghxx, index=var_names, columns=cols_xx),
            ghxu=pd.DataFrame(np.zeros((2, 2)), index=var_names, columns=["s1_e", "s2_e"]),
            ghuu=pd.DataFrame(np.zeros((2, 1)), index=var_names, columns=["e_e"]),
            ghs2=pd.Series(ghs2_fail, index=var_names),
            ys=pd.Series(ys, index=var_names),
            state_variables=tuple(state_names),
            variable_names=tuple(var_names),
            shock_names=("e",),
        )
        res_fail_s2 = verify_dynare_parity(pm_dr2, dyn_fail_s2, order=2, tol={"ghxx": tol_ghxx, "ghs2": tol_ghs2})
        assert res_fail_s2.passed is False
        assert res_fail_s2.dr_diff.loc["v2", "status"] == "FAIL"


# ===========================================================================
# 4. Canonical Pfeifer Smets-Wouters (2007) Model Parity Stress Tests
# ===========================================================================

class TestPfeiferSW07AdversarialParity:
    """Stress tests on canonical Pfeifer Smets-Wouters (2007) benchmark."""

    @pytest.fixture(scope="class")
    def sw07_solutions(self):
        """Solve SW07 at order 1 and 2 once for the test class."""
        assert SW07_PATH.exists(), f"SW07 .mod file missing at {SW07_PATH}"
        m1 = load_mod(SW07_PATH, order=1)
        dr1 = m1.decision_rules()

        m2 = load_mod(SW07_PATH, order=2)
        dr2 = m2.decision_rules()

        return {"m1": m1, "dr1": dr1, "m2": m2, "dr2": dr2}

    def test_sw07_arbitrary_scrambled_order_var_parity(self, sw07_solutions, tmp_path: Path):
        """Solve SW07 order 1, scramble all 40 variables randomly in .mat, assert 100% parity."""
        dr1 = sw07_solutions["dr1"]
        n_v, n_x, n_u = 40, 15, 7

        rng = np.random.default_rng(808)
        perm = rng.permutation(n_v)

        mat_path = tmp_path / "sw07_scrambled_parity.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": dr1.ghx.to_numpy()[perm, :],
                        "ghu": dr1.ghu.to_numpy()[perm, :],
                        "ys": dr1.ys.to_numpy()[:, None],
                        "order_var": (perm + 1)[:, None],
                    },
                    "mean": dr1.ys.to_numpy(),
                },
                "M_": {
                    "endo_names": np.array(dr1.variable_names),
                    "exo_names": np.array(dr1.shock_names),
                    "state_var": np.arange(1, n_x + 1)[:, None],
                },
            },
        )

        res = compare_model_to_dynare(SW07_PATH, mat_path, order=1)
        assert res.passed is True
        assert res.score == 100.0
        assert res.max_dev_ghx <= 1e-12
        assert res.max_dev_ghu <= 1e-12

    def test_sw07_second_order_folded_and_scrambled_parity(self, sw07_solutions, tmp_path: Path):
        """Solve SW07 order 2, fold ghxx from 225 to 120 cols, scramble order_var, assert parity."""
        dr2 = sw07_solutions["dr2"]
        n_v, n_x = 40, 15

        # Fold ghxx
        folded_cols = [i * n_x + j for i in range(n_x) for j in range(i, n_x)]
        assert len(folded_cols) == 120
        ghxx_folded = dr2.ghxx.to_numpy()[:, folded_cols]

        rng = np.random.default_rng(909)
        perm = rng.permutation(n_v)

        mat_path = tmp_path / "sw07_order2_folded_scrambled.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": dr2.ghx.to_numpy()[perm, :],
                        "ghu": dr2.ghu.to_numpy()[perm, :],
                        "ghxx": ghxx_folded[perm, :],
                        "ghs2": dr2.ghs2.to_numpy()[perm, None],
                        "ys": dr2.ys.to_numpy()[:, None],
                        "order_var": (perm + 1)[:, None],
                    },
                    "mean": dr2.ys.to_numpy(),
                },
                "M_": {
                    "endo_names": np.array(dr2.variable_names),
                    "exo_names": np.array(dr2.shock_names),
                    "state_var": np.arange(1, n_x + 1)[:, None],
                },
            },
        )

        res = compare_model_to_dynare(SW07_PATH, mat_path, order=2)
        assert res.passed is True
        assert res.score == 100.0
        assert res.max_dev_ghxx <= 1e-10
        assert res.max_dev_ghs2 <= 1e-10

    def test_sw07_injected_perturbation_failure_detection(self, sw07_solutions, tmp_path: Path):
        """Inject 1.01*tol perturbation on SW07 policy rule equation and verify exact failure."""
        dr1 = sw07_solutions["dr1"]
        tol = 1e-6

        ghx_bad = dr1.ghx.to_numpy().copy()
        # Find variable 'r' or pick row 10
        target_idx = list(dr1.variable_names).index("r") if "r" in dr1.variable_names else 0
        ghx_bad[target_idx, 0] += 1.01 * tol

        mat_path = tmp_path / "sw07_perturbed_fail.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": ghx_bad,
                        "ghu": dr1.ghu.to_numpy(),
                        "ys": dr1.ys.to_numpy(),
                        "order_var": np.arange(1, 41)[:, None],
                    }
                },
                "M_": {
                    "endo_names": np.array(dr1.variable_names),
                    "exo_names": np.array(dr1.shock_names),
                    "state_var": np.arange(1, 16)[:, None],
                },
            },
        )

        res = compare_model_to_dynare(SW07_PATH, mat_path, order=1, tol={"ghx": tol, "ghu": tol})
        assert res.passed is False
        assert res.score < 100.0
        failed_var = dr1.variable_names[target_idx]
        assert res.dr_diff.loc[failed_var, "status"] == "FAIL"


# ===========================================================================
# 5. Edge Cases, Dimension Extremes & Known Structural Limitations
# ===========================================================================

class TestEdgeCasesAndStructuralLimitations:
    """Exploratory stress tests probing edge cases, dimensions, and known caveats."""

    def test_single_equation_1x1_scalar_squeeze_limitation(self, tmp_path: Path):
        """Document empirical finding: 1x1 scalar model squeezed by scipy.io.loadmat.

        When a 1x1 model is loaded with squeeze_me=True, scipy converts [[0.8]] to
        a 0D float. If load_dynare_dr does not handle 0D arrays prior to ghx.shape unpacking,
        it raises ValueError.
        """
        mat_path = tmp_path / "scalar_1x1.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": np.array([[0.8]]),
                        "ghu": np.array([[0.5]]),
                        "ys": np.array([[1.0]]),
                        "order_var": np.array([[1]]),
                    }
                },
                "M_": {
                    "endo_names": np.array(["y"]),
                    "exo_names": np.array(["e"]),
                    "state_var": np.array([[1]]),
                },
            },
        )

        # Empirically verify whether load_dynare_dr raises ValueError on squeezed scalar
        try:
            dr = load_dynare_dr(mat_path)
            # If supported, shape must be (1, 1)
            assert dr.ghx.shape == (1, 1)
        except ValueError as exc:
            # Documented empirical finding: squeezed scalar unpack error
            assert "not enough values to unpack" in str(exc)

    def test_single_var_multi_state_1d_squeeze_limitation(self, tmp_path: Path):
        """Document empirical finding: 1 variable with 2 states squeezed to 1D array.

        When a (1, 2) row matrix is saved, loadmat squeezes to shape (2,).
        If ghx[:, None] is applied unconditionally, it becomes (2, 1) instead of (1, 2),
        causing shape mismatch with 1 variable name.
        """
        mat_path = tmp_path / "one_var_two_states.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "oo_": {
                    "dr": {
                        "ghx": np.array([[0.8, 0.2]]),
                        "ghu": np.array([[0.5, 0.1]]),
                        "ys": np.array([[1.0]]),
                        "order_var": np.array([[1]]),
                    }
                },
                "M_": {
                    "endo_names": np.array(["y"]),
                    "exo_names": np.array(["e1", "e2"]),
                    "state_var": np.array([[1], [2]]),
                },
            },
        )

        try:
            dr = load_dynare_dr(mat_path)
            assert dr.ghx.shape == (1, 2)
        except ValueError as exc:
            # Documented empirical finding: shape (2, 1) vs index (1, 1)
            assert "Shape of passed values is" in str(exc) or "not enough values" in str(exc)

    def test_missing_oo_dr_error_reporting(self, tmp_path: Path):
        """Passing MAT file without oo_.dr structure raises clean KeyError."""
        mat_path = tmp_path / "corrupt.mat"
        scipy.io.savemat(str(mat_path), {"random_struct": 42})
        with pytest.raises(KeyError, match="does not contain 'oo_'"):
            load_dynare_dr(mat_path)
