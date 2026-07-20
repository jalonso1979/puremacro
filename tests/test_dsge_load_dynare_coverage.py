"""Coverage tests for puremacro.dsge.load_dynare.

The module bridges Dynare *_results.mat files (scipy.io.loadmat output) to
numpy arrays matching the SVAR output shape.  Because the real .mat files
require a Dynare run, every test monkeypatches scipy.io.loadmat to inject
synthetic mat_struct / dict objects and then validates the mathematical
contract of each public function.

Tests cover:
  - _oo_as_dict: mat_struct path, dict path, missing oo_, unrecognised type
  - _irfs_as_dict: mat_struct path, dict path, unrecognised type
  - _get_irf_series: known key, missing key, squeeze/atleast_1d cast
  - load_irfs: shape (n_vars, n_shocks, horizon+1), zero-impact period,
               series values copied correctly, too-short series error,
               missing irfs field, missing key in irfs
  - load_fevd: variance_decomposition_ME path (3-D, 2-D, single-horizon
               broadcast, slice to H), gamma_y fallback (ones), missing both
  - DynareIRF and DynareFEVD: dataclass field round-trip
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dsge.load_dynare import (
    DynareFEVD,
    DynareIRF,
    _get_irf_series,
    _irfs_as_dict,
    _oo_as_dict,
    load_fevd,
    load_irfs,
)

# ---------------------------------------------------------------------------
# Helpers: fake mat_struct objects (replicate scipy.io.matlab.mat_struct API)
# ---------------------------------------------------------------------------

from scipy.io.matlab import mat_struct as _MatStruct


def _make_struct(**kwargs) -> _MatStruct:
    """Return a mat_struct-like object with given fields."""
    s = _MatStruct()
    s._fieldnames = list(kwargs.keys())
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _make_irf_data(n_vars: int, n_shocks: int, horizon: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Return IRF arrays for all var x shock combinations (keys = 'var_shock')."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for i in range(n_vars):
        for j in range(n_shocks):
            out[f"v{i}_s{j}"] = rng.normal(size=horizon)
    return out


def _make_fake_mat(irfs_obj, extra_oo: dict | None = None) -> dict:
    """Return a fake loadmat result dict containing oo_ with given irfs."""
    oo = _make_struct(irfs=irfs_obj, **(extra_oo or {}))
    return {"oo_": oo}


# ---------------------------------------------------------------------------
# Tests for _oo_as_dict
# ---------------------------------------------------------------------------


class TestOoAsDict:
    def test_mat_struct_path(self, monkeypatch, tmp_path):
        """_oo_as_dict returns a plain dict when oo_ is a mat_struct."""
        fake_result = _make_fake_mat(irfs_obj={})
        # Monkeypatch loadmat to return our fake result
        import puremacro.dsge.load_dynare as mod
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: fake_result)

        result = _oo_as_dict(tmp_path / "fake.mat")
        assert isinstance(result, dict)
        assert "irfs" in result

    def test_dict_path(self, monkeypatch, tmp_path):
        """_oo_as_dict handles oo_ that is already a plain dict."""
        fake_result = {"oo_": {"irfs": {}, "gamma_y": np.eye(3)}}
        import puremacro.dsge.load_dynare as mod
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: fake_result)

        result = _oo_as_dict(tmp_path / "fake.mat")
        assert isinstance(result, dict)
        assert "gamma_y" in result

    def test_missing_oo_raises(self, monkeypatch, tmp_path):
        """_oo_as_dict raises KeyError when oo_ is absent."""
        import puremacro.dsge.load_dynare as mod
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: {"not_oo": 1})

        with pytest.raises(KeyError, match="oo_"):
            _oo_as_dict(tmp_path / "fake.mat")

    def test_unrecognised_oo_type_raises(self, monkeypatch, tmp_path):
        """_oo_as_dict raises TypeError for an oo_ that is neither mat_struct nor dict."""
        import puremacro.dsge.load_dynare as mod
        # Use a plain list — not a mat_struct, not a dict
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: {"oo_": [1, 2, 3]})

        with pytest.raises(TypeError, match="Unrecognized oo_ type"):
            _oo_as_dict(tmp_path / "fake.mat")

    def test_mat_struct_field_values_preserved(self, monkeypatch, tmp_path):
        """Values returned by the mat_struct path match what was set."""
        irfs_data = {"y_ea": np.array([0.1, 0.2, 0.3])}
        fake = _make_fake_mat(irfs_obj=irfs_data)
        import puremacro.dsge.load_dynare as mod
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: fake)

        result = _oo_as_dict(tmp_path / "fake.mat")
        np.testing.assert_array_equal(result["irfs"]["y_ea"], irfs_data["y_ea"])


# ---------------------------------------------------------------------------
# Tests for _irfs_as_dict
# ---------------------------------------------------------------------------


class TestIrfsAsDict:
    def test_mat_struct_path(self):
        """_irfs_as_dict extracts fields from a mat_struct."""
        irfs_struct = _make_struct(y_shock1=np.array([1.0, 2.0]), p_shock1=np.array([3.0, 4.0]))
        result = _irfs_as_dict(irfs_struct)
        assert isinstance(result, dict)
        np.testing.assert_array_equal(result["y_shock1"], [1.0, 2.0])
        np.testing.assert_array_equal(result["p_shock1"], [3.0, 4.0])

    def test_dict_path(self):
        """_irfs_as_dict passes through a plain dict unchanged."""
        d = {"y_ea": np.array([0.5, 0.6])}
        result = _irfs_as_dict(d)
        assert result is d

    def test_unrecognised_type_raises(self):
        """_irfs_as_dict raises TypeError for an unexpected type."""
        with pytest.raises(TypeError, match="Unrecognized oo_.irfs type"):
            _irfs_as_dict(42)

    def test_empty_struct(self):
        """_irfs_as_dict handles a mat_struct with no fields."""
        empty = _make_struct()
        result = _irfs_as_dict(empty)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests for _get_irf_series
# ---------------------------------------------------------------------------


class TestGetIrfSeries:
    def test_known_key_returns_float_array(self):
        d = {"y_ea": np.array([0.1, 0.2, 0.3])}
        out = _get_irf_series(d, "y", "ea")
        assert out.dtype == float
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3])

    def test_missing_key_raises_keyerror(self):
        d = {"y_ea": np.array([0.1])}
        with pytest.raises(KeyError, match="y_shock2"):
            _get_irf_series(d, "y", "shock2")

    def test_squeeze_row_vector(self):
        """1xH row vectors are squeezed to 1-D."""
        d = {"y_s": np.array([[1.0, 2.0, 3.0]])}  # shape (1, 3)
        out = _get_irf_series(d, "y", "s")
        assert out.ndim == 1
        assert len(out) == 3

    def test_squeeze_column_vector(self):
        """Hx1 column vectors are squeezed to 1-D."""
        d = {"y_s": np.array([[1.0], [2.0], [3.0]])}  # shape (3, 1)
        out = _get_irf_series(d, "y", "s")
        assert out.ndim == 1
        assert len(out) == 3

    def test_scalar_returns_length_one(self):
        """A scalar-valued IRF becomes a length-1 array (atleast_1d)."""
        d = {"y_s": np.float64(0.5)}
        out = _get_irf_series(d, "y", "s")
        assert out.ndim == 1
        assert len(out) == 1
        assert out[0] == pytest.approx(0.5)

    def test_key_error_lists_available_keys(self):
        """The error message should mention available keys."""
        d = {"y_ea": np.ones(3), "p_ea": np.ones(3)}
        with pytest.raises(KeyError, match="y_ea"):
            _get_irf_series(d, "missing", "key")


# ---------------------------------------------------------------------------
# Tests for load_irfs
# ---------------------------------------------------------------------------


class TestLoadIrfs:
    def _patch_loadmat(self, monkeypatch, irfs_data: dict):
        """Patch loadmat so oo_.irfs is a plain dict with irfs_data."""
        import puremacro.dsge.load_dynare as mod
        fake = {"oo_": {"irfs": irfs_data}}
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: fake)

    def test_output_shape(self, monkeypatch, tmp_path):
        """ir array has shape (n_vars, n_shocks, horizon+1)."""
        n_vars, n_shocks, H = 2, 2, 8
        var_names = [f"v{i}" for i in range(n_vars)]
        shock_names = [f"s{j}" for j in range(n_shocks)]
        # _make_irf_data uses "v{i}_s{j}" keys which match var_names/shock_names
        irfs_data = _make_irf_data(n_vars, n_shocks, H)
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=var_names, shock_names=shock_names, horizon=H)
        assert result.ir.shape == (n_vars, n_shocks, H + 1)

    def test_impact_period_is_zero(self, monkeypatch, tmp_path):
        """Index 0 of the horizon axis is zero (period 0 = pre-shock)."""
        H = 6
        var_names = ["y", "p"]
        shock_names = ["ea"]
        # Keys must match the "<var>_<shock>" pattern load_irfs expects
        rng = np.random.default_rng(1)
        irfs_data = {
            "y_ea": rng.normal(size=H),
            "p_ea": rng.normal(size=H),
        }
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=var_names, shock_names=shock_names, horizon=H)
        np.testing.assert_array_equal(result.ir[:, :, 0], 0.0)

    def test_series_values_copied_correctly(self, monkeypatch, tmp_path):
        """ir[i, j, 1:H+1] matches the first H elements of the IRF series."""
        H = 4
        series = np.array([0.1, 0.2, 0.3, 0.4])
        irfs_data = {"y_s": series}
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=H)
        np.testing.assert_allclose(result.ir[0, 0, 1:], series)

    def test_series_longer_than_horizon_truncated(self, monkeypatch, tmp_path):
        """Series longer than horizon: only first H elements are used."""
        H = 3
        long_series = np.arange(10.0)  # length 10, use only first 3
        irfs_data = {"y_s": long_series}
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=H)
        np.testing.assert_allclose(result.ir[0, 0, 1:], long_series[:H])

    def test_series_too_short_raises(self, monkeypatch, tmp_path):
        """IRF series shorter than horizon raises ValueError."""
        H = 8
        short_series = np.ones(3)  # only 3 < 8
        irfs_data = {"y_s": short_series}
        self._patch_loadmat(monkeypatch, irfs_data)

        with pytest.raises(ValueError, match="length"):
            load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=H)

    def test_missing_irfs_field_raises(self, monkeypatch, tmp_path):
        """oo_ without 'irfs' raises KeyError."""
        import puremacro.dsge.load_dynare as mod
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: {"oo_": {"gamma_y": np.eye(2)}})

        with pytest.raises(KeyError, match="irfs"):
            load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=4)

    def test_missing_key_in_irfs_raises(self, monkeypatch, tmp_path):
        """KeyError when a var_shock key is absent from oo_.irfs."""
        irfs_data = {"y_ea": np.ones(5)}  # only y_ea, not p_ea
        self._patch_loadmat(monkeypatch, irfs_data)

        with pytest.raises(KeyError, match="p_ea"):
            load_irfs(tmp_path / "fake.mat", var_names=["y", "p"], shock_names=["ea"], horizon=4)

    def test_returned_dataclass_fields(self, monkeypatch, tmp_path):
        """DynareIRF has correct var_names, shock_names, horizon fields."""
        var_names = ["y", "p"]
        shock_names = ["ea", "mp"]
        H = 5
        rng = np.random.default_rng(2)
        # Keys must match "<var>_<shock>" pattern
        irfs_data = {
            "y_ea": rng.normal(size=H),
            "y_mp": rng.normal(size=H),
            "p_ea": rng.normal(size=H),
            "p_mp": rng.normal(size=H),
        }
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=var_names, shock_names=shock_names, horizon=H)
        assert isinstance(result, DynareIRF)
        assert result.var_names == var_names
        assert result.shock_names == shock_names
        assert result.horizon == H

    def test_single_var_single_shock(self, monkeypatch, tmp_path):
        """Works for the minimal 1x1 case."""
        H = 4
        irfs_data = {"y_s": np.array([1.0, 2.0, 3.0, 4.0])}
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=H)
        assert result.ir.shape == (1, 1, H + 1)

    def test_irfs_via_mat_struct(self, monkeypatch, tmp_path):
        """load_irfs works when oo_.irfs is a mat_struct (not a dict)."""
        H = 4
        irfs_struct = _make_struct(y_s=np.array([1.0, 2.0, 3.0, 4.0]))
        oo_struct = _make_struct(irfs=irfs_struct)
        import puremacro.dsge.load_dynare as mod
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: {"oo_": oo_struct})

        result = load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=H)
        assert result.ir.shape == (1, 1, H + 1)
        np.testing.assert_allclose(result.ir[0, 0, 1:], [1.0, 2.0, 3.0, 4.0])

    def test_ir_dtype_is_float(self, monkeypatch, tmp_path):
        """The ir array must be float (not integer)."""
        irfs_data = {"y_s": np.array([1, 2, 3, 4], dtype=int)}
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=4)
        assert np.issubdtype(result.ir.dtype, np.floating)

    def test_zero_irf_stays_zero(self, monkeypatch, tmp_path):
        """A zero IRF series produces all-zero ir output."""
        H = 6
        irfs_data = {"y_s": np.zeros(H)}
        self._patch_loadmat(monkeypatch, irfs_data)

        result = load_irfs(tmp_path / "fake.mat", var_names=["y"], shock_names=["s"], horizon=H)
        np.testing.assert_array_equal(result.ir, 0.0)


# ---------------------------------------------------------------------------
# Tests for load_fevd
# ---------------------------------------------------------------------------


class TestLoadFevd:
    def _patch_oo(self, monkeypatch, oo_fields: dict):
        """Patch loadmat with oo_ dict directly."""
        import puremacro.dsge.load_dynare as mod
        monkeypatch.setattr(mod, "loadmat", lambda *a, **kw: {"oo_": oo_fields})

    def test_variance_decomposition_3d(self, monkeypatch, tmp_path):
        """3-D variance_decomposition_ME sliced to requested horizons."""
        n_vars, n_shocks = 2, 2
        H_available = 20
        rng = np.random.default_rng(7)
        shares_raw = rng.random((n_vars, n_shocks, H_available))
        self._patch_oo(monkeypatch, {"variance_decomposition_ME": shares_raw})

        horizons = [1, 4, 8]
        result = load_fevd(
            tmp_path / "fake.mat",
            var_names=["y", "p"],
            shock_names=["ea", "mp"],
            horizons=horizons,
        )
        assert result.shares.shape == (n_vars, n_shocks, len(horizons))

    def test_variance_decomposition_2d_gets_extra_dim(self, monkeypatch, tmp_path):
        """2-D variance_decomposition_ME is promoted to 3-D."""
        n_vars, n_shocks = 2, 3
        rng = np.random.default_rng(11)
        shares_2d = rng.random((n_vars, n_shocks))
        self._patch_oo(monkeypatch, {"variance_decomposition_ME": shares_2d})

        result = load_fevd(
            tmp_path / "fake.mat",
            var_names=["y", "p"],
            shock_names=["s1", "s2", "s3"],
            horizons=[4],
        )
        assert result.shares.shape == (n_vars, n_shocks, 1)

    def test_single_horizon_broadcast_to_multiple(self, monkeypatch, tmp_path):
        """If shares has last dim=1 and H>1, the single value is broadcast."""
        n_vars, n_shocks = 2, 2
        shares_one = np.ones((n_vars, n_shocks, 1)) * 0.5
        self._patch_oo(monkeypatch, {"variance_decomposition_ME": shares_one})

        horizons = [1, 4, 8, 12]
        result = load_fevd(
            tmp_path / "fake.mat",
            var_names=["y", "p"],
            shock_names=["ea", "mp"],
            horizons=horizons,
        )
        assert result.shares.shape == (n_vars, n_shocks, len(horizons))
        np.testing.assert_allclose(result.shares, 0.5)

    def test_gamma_y_fallback_gives_ones(self, monkeypatch, tmp_path):
        """When variance_decomposition_ME absent, gamma_y fallback returns ones array."""
        n_vars, n_shocks = 3, 2
        self._patch_oo(monkeypatch, {"gamma_y": np.eye(5)})

        horizons = [1, 4, 8]
        result = load_fevd(
            tmp_path / "fake.mat",
            var_names=["y", "p", "u"],
            shock_names=["ea", "mp"],
            horizons=horizons,
        )
        assert result.shares.shape == (n_vars, n_shocks, len(horizons))
        np.testing.assert_allclose(result.shares, 1.0)

    def test_missing_both_fields_raises(self, monkeypatch, tmp_path):
        """Raises KeyError when neither variance_decomposition_ME nor gamma_y is present."""
        self._patch_oo(monkeypatch, {"irfs": {}})

        with pytest.raises(KeyError, match="variance_decomposition_ME"):
            load_fevd(
                tmp_path / "fake.mat",
                var_names=["y"],
                shock_names=["s"],
                horizons=[4],
            )

    def test_returned_dataclass_fields(self, monkeypatch, tmp_path):
        """DynareFEVD has correct var_names, shock_names, horizons."""
        n_vars, n_shocks = 2, 1
        shares = np.ones((n_vars, n_shocks, 3))
        self._patch_oo(monkeypatch, {"variance_decomposition_ME": shares})
        horizons = [1, 4, 8]
        var_names = ["y", "p"]
        shock_names = ["ea"]

        result = load_fevd(
            tmp_path / "fake.mat",
            var_names=var_names,
            shock_names=shock_names,
            horizons=horizons,
        )
        assert isinstance(result, DynareFEVD)
        assert result.var_names == var_names
        assert result.shock_names == shock_names
        assert result.horizons == horizons

    def test_shares_shape_consistent_with_inputs(self, monkeypatch, tmp_path):
        """Output shares.shape always matches (n_vars, n_shocks, len(horizons))."""
        for n_vars, n_shocks, H in [(1, 1, 1), (3, 2, 5), (4, 4, 10)]:
            shares_raw = np.ones((n_vars, n_shocks, H))
            self._patch_oo(monkeypatch, {"variance_decomposition_ME": shares_raw})
            result = load_fevd(
                tmp_path / "fake.mat",
                var_names=[f"v{i}" for i in range(n_vars)],
                shock_names=[f"s{j}" for j in range(n_shocks)],
                horizons=list(range(1, H + 1)),
            )
            assert result.shares.shape == (n_vars, n_shocks, H)

    def test_fevd_shares_dtype_float(self, monkeypatch, tmp_path):
        """shares array must be floating-point."""
        shares = np.ones((2, 2, 4), dtype=int)
        self._patch_oo(monkeypatch, {"variance_decomposition_ME": shares})

        result = load_fevd(
            tmp_path / "fake.mat",
            var_names=["y", "p"],
            shock_names=["ea", "mp"],
            horizons=[1, 4, 8, 12],
        )
        assert np.issubdtype(result.shares.dtype, np.floating)

    def test_variance_decomposition_reshaped_when_shape_mismatch(self, monkeypatch, tmp_path):
        """3-D array with wrong leading dims gets reshaped to (n_vars, n_shocks, H)."""
        n_vars, n_shocks, H = 2, 2, 3
        # Provide a flat array of size n_vars*n_shocks*H — scipy.io can return this
        flat_shares = np.ones(n_vars * n_shocks * H)
        self._patch_oo(monkeypatch, {"variance_decomposition_ME": flat_shares.reshape(1, -1)})

        result = load_fevd(
            tmp_path / "fake.mat",
            var_names=["y", "p"],
            shock_names=["ea", "mp"],
            horizons=[1, 4, 8],
        )
        assert result.shares.shape == (n_vars, n_shocks, H)


# ---------------------------------------------------------------------------
# Tests for DynareIRF and DynareFEVD dataclasses (round-trip construction)
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_dynare_irf_stores_all_fields(self):
        ir = np.zeros((2, 3, 9))
        obj = DynareIRF(ir=ir, var_names=["y", "p"], shock_names=["ea", "mp", "tech"], horizon=8)
        assert obj.ir is ir
        assert obj.var_names == ["y", "p"]
        assert obj.shock_names == ["ea", "mp", "tech"]
        assert obj.horizon == 8

    def test_dynare_fevd_stores_all_fields(self):
        shares = np.ones((2, 2, 3))
        obj = DynareFEVD(shares=shares, var_names=["y", "p"], shock_names=["ea", "mp"], horizons=[1, 4, 8])
        assert obj.shares is shares
        assert obj.var_names == ["y", "p"]
        assert obj.shock_names == ["ea", "mp"]
        assert obj.horizons == [1, 4, 8]

    def test_dynare_irf_shape_property(self):
        ir = np.zeros((3, 2, 13))
        obj = DynareIRF(ir=ir, var_names=["a", "b", "c"], shock_names=["s1", "s2"], horizon=12)
        assert obj.ir.shape == (3, 2, 13)  # (n_vars, n_shocks, horizon+1)
