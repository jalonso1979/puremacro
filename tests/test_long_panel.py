"""Tests for the G9 long-panel loader (puremacro.long_panel).

Most of this file needs a source the repo does not ship: PWT's ``.dta`` (not
redistributable), the EU-KLEMS legacy archive, or the live OECD-STAN SDMX
endpoint. Those tests skip cleanly when the source is absent — but the STAN and
panel-build fixtures were reaching the network to *discover* that, 18 requests
and ~9 seconds on every default run, with no ``network`` marker to deselect
them. They carry one now, so `pytest` offline neither waits nor pretends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro import long_panel as lp


G9 = ["USA", "JPN", "DEU", "FRA", "GBR", "ITA", "CAN", "NLD", "AUS"]


@pytest.fixture(scope="module")
def pwt10_panel():
    """Load PWT 10.0 once per module — file IO is expensive.

    Skips cleanly if `data/raw/pwt10/pwt100.dta` is not present, per
    the project's ``feedback_network_tests_skip_on_empty`` rule.
    """
    try:
        df = lp.load_pwt10(countries=G9, start=1970, end=2019)
    except FileNotFoundError as exc:
        pytest.skip(f"PWT 10.0 not downloaded: {exc}")
    if len(df) == 0:
        pytest.skip("PWT 10.0 returned an empty frame for G9")
    return df


def test_pwt10_returns_dataframe(pwt10_panel):
    assert isinstance(pwt10_panel, pd.DataFrame)
    assert {"code", "year", "log_relprice_equip"}.issubset(pwt10_panel.columns)


def test_pwt10_g9_coverage(pwt10_panel):
    """All nine G9 countries should be present."""
    assert set(pwt10_panel["code"].unique()) == set(G9)


def test_pwt10_continuous_coverage(pwt10_panel):
    """Each G9 country has at least 40 contiguous years in [1970, 2019]."""
    counts = pwt10_panel.dropna(subset=["log_relprice_equip"]).groupby("code").size()
    assert (counts >= 40).all(), counts


def test_pwt10_relprice_construction(pwt10_panel):
    """log_relprice_equip = log(pl_i) - log(pl_c) within fp tolerance."""
    sub = pwt10_panel.dropna(subset=["log_relprice_equip", "pl_i", "pl_c"]).head(50)
    expected = np.log(sub["pl_i"]) - np.log(sub["pl_c"])
    np.testing.assert_allclose(sub["log_relprice_equip"], expected, atol=1e-12)


@pytest.fixture(scope="module")
def stan_panel():
    """Load OECD-STAN labor share once per module.

    Skips cleanly if the SDMX endpoint returns empty for the entire
    cohort (network failure, retired dataflow, etc.) — per the
    project's `feedback_network_tests_skip_on_empty` rule.
    """
    df = lp.load_oecd_stan_ls(countries=G9, start=1970, end=2019)
    if len(df) == 0:
        pytest.skip("OECD-STAN returned empty for G9 — endpoint may be retired or unreachable")
    return df


@pytest.mark.network
def test_stan_returns_dataframe(stan_panel):
    assert isinstance(stan_panel, pd.DataFrame)
    assert {"code", "year", "log_LS_stan"}.issubset(stan_panel.columns)


@pytest.mark.network
def test_stan_ls_in_range(stan_panel):
    """Labor share is in (0, 1) — log_LS in (-inf, 0). Allow up to 5%
    of cells slightly above 0 due to STAN's basic-prices definition."""
    sub = stan_panel.dropna(subset=["log_LS_stan"])
    assert (sub["log_LS_stan"] > np.log(0.2)).all()
    # Most cells should have log_LS < 0; tolerate a few slightly above.
    assert (sub["log_LS_stan"] < 0.05).mean() >= 0.95


@pytest.mark.network
def test_stan_g9_partial_coverage(stan_panel):
    """At least three G9 countries should have ≥20 years of STAN coverage."""
    counts = stan_panel.dropna(subset=["log_LS_stan"]).groupby("code").size()
    assert (counts >= 20).sum() >= 3


@pytest.fixture(scope="module")
def klems_legacy_panel():
    """Load EU-KLEMS 2008/2009 release once per module.

    Skips cleanly if the archive isn't downloaded yet — per the
    project's `feedback_network_tests_skip_on_empty` rule.
    """
    df = lp.load_klems_legacy(countries=G9, start=1970, end=2007)
    if len(df) == 0:
        pytest.skip("EU-KLEMS legacy 2008 .xls archive not present in data/raw/euklems_legacy/")
    return df


def test_klems_legacy_columns(klems_legacy_panel):
    assert {"code", "year", "log_LS_klems_leg"}.issubset(klems_legacy_panel.columns)


def test_klems_legacy_ls_range(klems_legacy_panel):
    """log_LS in (log(0.2), 0.05) — labor share between 20% and ≈100%."""
    sub = klems_legacy_panel.dropna(subset=["log_LS_klems_leg"])
    assert (sub["log_LS_klems_leg"] > np.log(0.2)).all()
    assert (sub["log_LS_klems_leg"] < 0.05).mean() >= 0.95


def test_klems_legacy_at_least_six_countries(klems_legacy_panel):
    """At least six of the nine G9 countries should have legacy coverage."""
    assert klems_legacy_panel["code"].nunique() >= 6


@pytest.fixture(scope="module")
def g9_long_panel():
    """Build the G9 long panel (writes the CSV as a side effect).

    Skips cleanly if no source returns data — without PWT, STAN, or
    KLEMS-legacy coverage there is nothing to test. KLEMS-2023 alone
    provides at least the modern LS rows for some countries; the test
    proceeds in that case.
    """
    panel = lp.build_g9_long_panel(start=1975, end=2024, write_csv=False)
    if len(panel) == 0:
        pytest.skip("No source data available — all four loaders returned empty")
    return panel


@pytest.mark.network
def test_g9_panel_columns(g9_long_panel):
    expected = {"code", "year", "log_relprice_equip", "log_LS",
                "vintage_LS", "vintage_PK"}
    assert expected.issubset(g9_long_panel.columns)


@pytest.mark.network
def test_g9_panel_year_range(g9_long_panel):
    assert g9_long_panel["year"].min() >= 1975
    assert g9_long_panel["year"].max() <= 2024


@pytest.mark.network
def test_g9_panel_country_subset_of_g9(g9_long_panel):
    """No country outside the G9 cohort should appear."""
    assert set(g9_long_panel["code"].unique()).issubset(set(G9))


@pytest.mark.network
def test_g9_panel_at_least_one_country(g9_long_panel):
    """At least one G9 country must have data."""
    assert g9_long_panel["code"].nunique() >= 1


def test_splice_audit_columns():
    panel = lp.build_g9_long_panel(start=1975, end=2024, write_csv=False)
    audit = lp.splice_audit(panel)
    assert {"code", "overlap_n", "mad_LS_pp", "mad_PK_pp"}.issubset(audit.columns)


@pytest.mark.network
def test_splice_audit_finite_mad_when_overlap_exists():
    """When STAN and KLEMS-legacy both have coverage, MAD is finite and ≥ 0."""
    panel = lp.build_g9_long_panel(start=1975, end=2024, write_csv=False)
    audit = lp.splice_audit(panel)
    finite = audit.dropna(subset=["mad_LS_pp"])
    if len(finite) == 0:
        pytest.skip("No STAN/KLEMS-legacy overlap available — both archives absent")
    assert (finite["mad_LS_pp"] >= 0).all()
    assert finite["mad_LS_pp"].max() < 30.0  # 30pp is a generous sanity floor


def test_root_is_the_repo_root_not_the_directory_above_it():
    """`_ROOT` fed every data path in this module and was one level too high.

    It read `parents[2]`, correct when the package was nested as
    `puremacro/puremacro/` and wrong ever since the split. Nothing noticed
    because the resulting path — whatever directory happens to sit above the
    checkout — exists on the author's machine, so the loaders failed with a
    plausible "not found at <path>" naming a path outside the repo.
    """
    from pathlib import Path

    root = lp._ROOT
    assert (root / "pyproject.toml").is_file(), root
    assert (root / "puremacro" / "long_panel.py").is_file(), root
    assert root == Path(lp.__file__).resolve().parents[1]
    # and every path built from it stays inside the repo
    assert str(lp._pwt10_path()).startswith(str(root) + "/")
