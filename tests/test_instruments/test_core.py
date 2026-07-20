"""Tests for puremacro.instruments._core: Instrument + InstrumentLike."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument, InstrumentLike


def _make_quarterly(n=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.Series(rng.standard_normal(n), index=idx, name="z")


# --------------------------------------------------------------------------
# Instrument: structural
# --------------------------------------------------------------------------
def test_instrument_is_frozen_dataclass():
    inst = Instrument(
        series=_make_quarterly(),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    assert dataclasses.is_dataclass(inst)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.name = "other"


def test_instrument_metadata_defaults_to_empty_dict():
    inst = Instrument(
        series=_make_quarterly(),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    assert inst.metadata == {}


def test_instrument_carries_documented_fields():
    s = _make_quarterly()
    inst = Instrument(
        series=s, name="test", source="synthetic",
        category="literature", frequency="Q",
        metadata={"foo": "bar"},
    )
    assert inst.series is s
    assert inst.name == "test"
    assert inst.source == "synthetic"
    assert inst.category == "literature"
    assert inst.frequency == "Q"
    assert inst.metadata == {"foo": "bar"}


def test_instrument_rejects_invalid_category():
    with pytest.raises(ValueError, match="category"):
        Instrument(
            series=_make_quarterly(),
            name="test", source="synthetic",
            category="not_a_real_category", frequency="Q",
        )


# --------------------------------------------------------------------------
# InstrumentLike: protocol
# --------------------------------------------------------------------------
def test_instrument_like_is_runtime_checkable():
    class _MyShock:
        def as_instrument(self) -> Instrument:
            return Instrument(
                series=_make_quarterly(),
                name="mine", source="synthetic",
                category="literature", frequency="Q",
            )
    assert isinstance(_MyShock(), InstrumentLike)


def test_instrument_like_rejects_non_conforming():
    class _NotAShock:
        pass
    assert not isinstance(_NotAShock(), InstrumentLike)


# --------------------------------------------------------------------------
# Instrument convenience methods
# --------------------------------------------------------------------------
def test_instrument_diagnostics_shape():
    inst = Instrument(
        series=_make_quarterly(n=24, seed=1),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    d = inst.diagnostics()
    assert set(d.keys()) >= {"n_obs", "mean", "std", "first_date", "last_date"}
    assert d["n_obs"] == 24
    assert d["first_date"] is not None
    assert d["last_date"] is not None


def test_instrument_summary_returns_string_with_name_and_source():
    inst = Instrument(
        series=_make_quarterly(),
        name="ramey_2011_defense", source="Ramey 2011 buildup events",
        category="narrative_replication", frequency="Q",
    )
    s = inst.summary()
    assert isinstance(s, str)
    assert "ramey_2011_defense" in s
    assert "Ramey 2011" in s
    assert "Q" in s


def test_instrument_validate_against_returns_correlation():
    rng = np.random.default_rng(42)
    n = 40
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    base = rng.standard_normal(n)
    z = pd.Series(base, index=idx)
    bench = pd.Series(base + 0.1 * rng.standard_normal(n), index=idx)
    inst = Instrument(
        series=z, name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    result = inst.validate_against(bench)
    assert "correlation" in result
    assert result["correlation"] > 0.9


def test_instrument_diagnostics_empty_series():
    import math
    inst = Instrument(
        series=pd.Series(dtype=float),
        name="empty", source="synthetic",
        category="literature", frequency="Q",
    )
    d = inst.diagnostics()
    assert d["n_obs"] == 0
    assert d["first_date"] is None
    assert d["last_date"] is None
    assert math.isnan(d["mean"])


def test_instrument_validate_against_non_overlapping():
    import math
    idx_a = pd.date_range("2000-01-01", periods=10, freq="QS")
    idx_b = pd.date_range("2010-01-01", periods=10, freq="QS")
    inst = Instrument(
        series=pd.Series(np.ones(10), index=idx_a),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    result = inst.validate_against(pd.Series(np.ones(10), index=idx_b))
    assert result["n_overlap"] == 0
    assert math.isnan(result["correlation"])


def test_instrument_to_proxy_svar_matches_raw_call():
    """instrument.to_proxy_svar(Y, p=, horizon=) must produce identical output
    to proxy_svar(Y, p=, horizon=, instrument_series=instrument.series.values)."""
    from puremacro.var.identify.proxy import proxy_svar

    rng = np.random.default_rng(0)
    T, n = 60, 2
    Y = rng.standard_normal((T, n)).cumsum(axis=0)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    z_arr = rng.standard_normal(T)
    z_series = pd.Series(z_arr, index=idx)
    inst = Instrument(
        series=z_series, name="test", source="synthetic",
        category="literature", frequency="Q",
    )

    res_via_inst = inst.to_proxy_svar(Y, p=2, horizon=5, n_boot=50, seed=7)
    res_raw = proxy_svar(Y, p=2, horizon=5, instrument_series=z_arr,
                         n_boot=50, seed=7)
    np.testing.assert_array_equal(res_via_inst.irf_point, res_raw.irf_point)


def test_instrument_to_lp_iv_runs_end_to_end():
    rng = np.random.default_rng(1)
    T = 120
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    z_arr = rng.standard_normal(T)
    x_arr = 0.6 * z_arr + 0.5 * rng.standard_normal(T)
    y_arr = (0.4 * x_arr + 0.3 * rng.standard_normal(T)).cumsum()

    df = pd.DataFrame({"y": y_arr, "x": x_arr}, index=idx)
    z_series = pd.Series(z_arr, index=idx)
    inst = Instrument(
        series=z_series, name="test", source="synthetic",
        category="literature", frequency="Q",
    )

    out = inst.to_lp_iv(df, y="y", x="x", horizons=range(0, 5), n_lags=2)
    assert isinstance(out, pd.DataFrame)
    assert "h" in out.columns
    assert "beta" in out.columns
    # The DGP has y growing with x, x growing with z. Sign of beta@h=0 should be positive.
    assert out.iloc[0]["beta"] > 0
    assert len(out) == 5


def test_instrument_to_lp_iv_rejects_reserved_column():
    rng = np.random.default_rng(2)
    T = 30
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    df = pd.DataFrame({
        "y": rng.standard_normal(T),
        "x": rng.standard_normal(T),
        "_instrument_z": rng.standard_normal(T),  # collision
    }, index=idx)
    inst = Instrument(
        series=pd.Series(rng.standard_normal(T), index=idx),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    with pytest.raises(ValueError, match="_instrument_z"):
        inst.to_lp_iv(df, y="y", x="x")
