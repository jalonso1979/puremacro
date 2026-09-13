"""The `io` extra (pyarrow, openpyxl) is optional since 3.3.0.

Code that cannot work without those engines has to say so up front and name the
extra, rather than failing inside a builder whose `except Exception: print(...)`
turns the missing engine into silently missing series -- the failure that once
made openpyxl a base dependency (CHANGELOG 3.3.0, "Changed").
"""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro import _optional

_ENGINE_MODULES = ("pyarrow", "fastparquet", "openpyxl")


@pytest.fixture
def no_engines(monkeypatch):
    """Make every file-format engine look uninstalled to `puremacro._optional`."""
    real = _optional._find_spec

    def fake(name, *args, **kwargs):
        return None if name in _ENGINE_MODULES else real(name, *args, **kwargs)

    monkeypatch.setattr(_optional, "_find_spec", fake)


def test_dev_environment_has_both_engines():
    # `dev` includes the `io` engines, so the suite itself runs with them.
    assert _optional.has_engine("parquet") and _optional.has_engine("excel")
    _optional.require_engines("parquet", "excel", feature="anything")


def test_missing_engines_raise_an_import_error_naming_the_extra(no_engines):
    with pytest.raises(_optional.MissingEngineError) as exc:
        _optional.require_engines("parquet", "excel", feature="build_all")
    msg = str(exc.value)
    assert isinstance(exc.value, ImportError)
    assert "build_all" in msg and "pyarrow" in msg and "openpyxl" in msg
    assert 'pip install "puremacro[io]"' in msg


def test_only_the_missing_engine_is_reported(monkeypatch):
    real = _optional._find_spec
    monkeypatch.setattr(
        _optional, "_find_spec",
        lambda name, *a, **k: None if name == "openpyxl" else real(name, *a, **k),
    )
    _optional.require_engines("parquet", feature="x")  # pyarrow is still there
    with pytest.raises(_optional.MissingEngineError, match="openpyxl") as exc:
        _optional.require_engines("parquet", "excel", feature="x")
    assert "parquet" not in str(exc.value)


def test_unknown_engine_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown engine kind"):
        _optional.require_engines("feather", feature="x")


def test_build_all_checks_before_touching_disk_or_network(no_engines, monkeypatch, tmp_path):
    from puremacro import build_panel

    data_dir = tmp_path / "processed"
    monkeypatch.setattr(build_panel, "DATA_DIR", data_dir)
    with pytest.raises(_optional.MissingEngineError, match=r"puremacro\[io\]"):
        build_panel.build_all()
    assert not data_dir.exists()


def test_shock_atlas_checks_before_loading(no_engines, tmp_path):
    from puremacro import shock_atlas

    with pytest.raises(_optional.MissingEngineError, match="load_all_shocks"):
        shock_atlas.load_all_shocks(tmp_path)


def test_climate_panel_does_not_drop_an_unreadable_macro_panel(no_engines, monkeypatch, tmp_path):
    from puremacro import build_panel, climate_panel
    from puremacro.fetch import emissions, energy_transition

    monkeypatch.setattr(emissions, "fetch_emissions_panel", lambda **k: pd.DataFrame())
    monkeypatch.setattr(energy_transition, "fetch_energy_transition", lambda **k: pd.DataFrame())
    panel = tmp_path / "panel_Q.parquet"
    panel.write_bytes(b"")  # exists; its content is never read
    monkeypatch.setattr(build_panel, "PANEL_Q_PATH", panel)
    with pytest.raises(_optional.MissingEngineError, match="build_climate_panel"):
        climate_panel.build_climate_panel(["USA"])
