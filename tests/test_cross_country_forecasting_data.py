"""The frozen WDI conversion executes the fetcher and refuses changed sources."""
import importlib.util
from pathlib import Path
import shutil

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cross_country_data_builder", ROOT / "tools/build_cross_country_forecasting_data.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
SOURCE = ROOT / "reviews/2026-10-10-cross-country-forecasting/source"


def test_reviewed_raw_responses_rebuild_the_packaged_observations():
    panel, metadata = builder.extract(SOURCE)
    raw = panel.reset_index().to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n").encode()
    packaged = (ROOT / "puremacro/datasets/data/cross_country_gdp.csv").read_bytes()
    assert raw == packaged
    assert len(panel) == 195 and not panel.attrs["missing"]
    assert metadata["source_lastupdated"] == "2026-10-08"
    assert metadata["source"]["units"].startswith("millions of constant LCU")
    assert set(panel.index.get_level_values("code")) == {"USA", "MEX", "BRA"}
    assert panel.loc["MEX"].index.equals(pd.date_range("1960-01-01", "2024-01-01", freq="YS", name="date"))


@pytest.mark.parametrize("filename", ["wdi_countries.json", "wdi_gdp.json"])
def test_source_revision_is_not_silently_accepted(tmp_path, filename):
    for source in builder.SOURCES:
        shutil.copyfile(SOURCE / source["file"], tmp_path / source["file"])
    changed = tmp_path / filename
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Unreviewed source SHA256"):
        builder.extract(tmp_path)
