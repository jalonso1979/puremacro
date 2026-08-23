"""VintagePanel container and the vintage_panel provider loop.

These are all offline: panels are built by hand and providers are
registered as plain callables, so nothing here touches the network.

Every test targets a way the plumbing can produce a *well-formed wrong
answer* — a slice that silently interleaves countries, a country
reported "missing" when a fetch merely failed, an override that looks
empty because its keys were typed in lower case.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.realtime import _base as B
from puremacro.fetch.realtime._base import VintagePanel
from puremacro.fetch.realtime.catalog import resolve_series
from puremacro.fetch.realtime.panel import vintage_panel


def _rows(country, variable="gdp_real", provider="oecd_stes", n=6,
          start="2000-01-01", base=100.0, units="level", seed=0):
    """Rows for one series, with every edition republishing the history.

    Two properties matter and both were absent from an earlier version
    of this helper, which made several tests below unable to fail:

    * every edition carries the whole back-series, so a triangle column
      is longer than one cell and a growth rate can actually be formed;
    * growth varies period to period, so the news/noise regressor is
      not a constant column (which is singular, not merely uninformative).
    """
    rng = np.random.default_rng(seed + abs(hash(country)) % 1000)
    dates = pd.date_range(start, periods=n, freq="QS")
    growth = 0.004 + 0.004 * rng.standard_normal(n)
    truth = base * np.exp(np.cumsum(growth))
    provisional = 0.3 * rng.standard_normal(n)
    editions = pd.date_range(dates[0] + pd.DateOffset(months=4),
                             dates[-1] + pd.DateOffset(months=10), freq="QS")
    out = []
    for v in editions:
        for i, d in enumerate(dates):
            if d + pd.DateOffset(months=4) > v:
                continue
            age = len(pd.date_range(d + pd.DateOffset(months=4), v, freq="QS"))
            out.append({
                "country": country, "variable": variable, "date": d,
                "vintage": v,
                # Provisional at first, settling afterwards, so
                # successive editions genuinely differ.
                "value": truth[i] + (provisional[i] if age == 1 else 0.0),
                "provider": provider,
                "series_id": f"{country}-{variable}", "units": units,
            })
    return out


def _panel(*groups):
    rows = []
    for g in groups:
        rows += g
    return VintagePanel(df=pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# slicing
# ---------------------------------------------------------------------------
def test_long_with_a_variable_and_no_country_refuses_to_interleave():
    """Concatenating two countries into one [date, vintage, value] frame
    yields a perfectly well-formed triangle of nonsense."""
    panel = _panel(_rows("DEU"), _rows("ESP", base=500.0))
    with pytest.raises(ValueError, match="spans several countries"):
        panel.long(variable="gdp_real")


def test_long_with_a_variable_and_one_country_is_fine():
    panel = _panel(_rows("DEU"))
    out = panel.long(variable="gdp_real")
    assert list(out.columns) == ["date", "vintage", "value"]
    assert len(out) == len(_rows("DEU"))


def test_long_with_an_unknown_variable_raises():
    panel = _panel(_rows("DEU"))
    with pytest.raises(KeyError):
        panel.long(variable="cpi")


def test_long_accepts_an_alias_spelling():
    """The panel stores canonical names; a caller passing the SDMX code
    would otherwise match nothing and get an empty slice."""
    panel = _panel(_rows("DEU"))
    n = len(_rows("DEU"))
    assert len(panel.long("DEU", "B1GQ")) == n
    assert len(panel.long("DEU", "gdp_real")) == n


def test_country_without_variable_refuses_when_ambiguous():
    panel = _panel(_rows("DEU", "gdp_real"), _rows("DEU", "con_real"))
    with pytest.raises(ValueError, match="several variables"):
        panel.long("DEU")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def test_same_key_from_two_providers_warns():
    """Two providers for one series is never a correct merge: they
    differ in units and in what a vintage date means."""
    rows = _rows("DEU", provider="oecd_stes") + _rows("DEU", provider="alfred")
    with pytest.warns(UserWarning, match="several providers"):
        panel = VintagePanel(df=pd.DataFrame(rows))
    # and only one row per key survives
    assert not panel.df.duplicated(
        subset=["country", "variable", "date", "vintage"]).any()


def test_one_provider_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _panel(_rows("DEU"), _rows("ESP"))


def test_units_default_when_the_column_is_absent():
    df = pd.DataFrame(_rows("DEU")).drop(columns=["units"])
    panel = VintagePanel(df=df)
    assert set(panel.df["units"]) == {"level"}


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------
def test_concat_merges_failure_reasons_from_every_panel():
    """The panel that returned nothing is precisely the one whose
    reasons matter; dict.update dropped them."""
    a = VintagePanel(df=pd.DataFrame(_rows("DEU")),
                     metadata={"failed": {"DEU:con_real": "404"}})
    b = VintagePanel(df=pd.DataFrame([]),
                     metadata={"failed": {"ESP:gdp_real": "429"}})
    out = VintagePanel.concat([a, b])
    assert out.metadata["failed"] == {"DEU:con_real": "404",
                                      "ESP:gdp_real": "429"}
    assert out.countries == ["DEU"]


def test_concat_of_nothing_is_empty_not_an_error():
    out = VintagePanel.concat([])
    assert out.is_empty()
    assert out.countries == []


# ---------------------------------------------------------------------------
# news_or_noise_panel
# ---------------------------------------------------------------------------
def test_failed_rows_carry_empty_strings_not_nan_verdicts():
    """A NaN in `verdict` reads as a verdict. Rows that were not
    estimated must say so."""
    panel = _panel(_rows("DEU", n=40), _rows("ESP", n=3, base=500.0))
    res = panel.news_or_noise_panel(min_obs=12)
    assert set(res["country"]) == {"DEU", "ESP"}
    bad = res[~res["ok"]]
    assert len(bad) == 1
    assert bad["verdict"].iloc[0] == ""
    assert bad["note"].iloc[0] != ""
    assert res["ok"].dtype == bool


def test_variable_filter_selects(monkeypatch):
    panel = _panel(_rows("DEU", "gdp_real", n=40),
                   _rows("DEU", "con_real", n=40))
    res = panel.news_or_noise_panel(variable="gdp_real", min_obs=5)
    assert set(res["variable"]) == {"gdp_real"}


# ---------------------------------------------------------------------------
# vintage_panel provider loop
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_providers(monkeypatch):
    """Two registered providers with disjoint variable coverage."""
    from puremacro.fetch.realtime import catalog as C

    calls = []

    def make(provider, served):
        def _fetch(countries, variables, **kw):
            calls.append((provider, tuple(countries), tuple(variables)))
            rows = []
            for c in countries:
                for v in variables:
                    if (c, v) in served:
                        rows += _rows(c, v, provider=provider, n=8)
            return VintagePanel(
                df=pd.DataFrame(rows),
                metadata={"provider": provider,
                          "failed": {f"{c}:{v}": "not served"
                                     for c in countries for v in variables
                                     if (c, v) not in served}},
            )
        return _fetch

    saved_reg = dict(B._PROVIDER_REGISTRY)
    saved_cov = dict(B._PROVIDER_COVERAGE)
    saved_cat = dict(C._CATALOGS)

    B.register_provider("p_one", make("p_one", {("DEU", "gdp_real")}), ["DEU"])
    B.register_provider("p_two", make("p_two", {("DEU", "deflator")}), ["DEU"])
    C.register_catalog("p_one", {"DEU": {"gdp_real": "A", "deflator": "B"}})
    C.register_catalog("p_two", {"DEU": {"gdp_real": "C", "deflator": "D"}})
    yield calls
    B._PROVIDER_REGISTRY.clear(); B._PROVIDER_REGISTRY.update(saved_reg)
    B._PROVIDER_COVERAGE.clear(); B._PROVIDER_COVERAGE.update(saved_cov)
    C._CATALOGS.clear(); C._CATALOGS.update(saved_cat)


def test_a_variable_missed_by_the_first_provider_falls_through(fake_providers):
    """Keying the work queue on country alone retired DEU as soon as GDP
    came back, so the deflator was never looked for anywhere else."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = vintage_panel(["DEU"], variables=["gdp_real", "deflator"],
                              providers=["p_one", "p_two"])
    assert set(panel.variables) == {"gdp_real", "deflator"}
    assert panel.metadata["missing"] == []
    # and the second provider was actually consulted
    assert any(p == "p_two" for p, _c, _v in fake_providers)


def test_failure_reasons_reach_the_caller(fake_providers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = vintage_panel(["DEU"], variables=["gdp_real", "deflator"],
                              providers=["p_one"])
    assert "deflator" not in panel.variables
    assert panel.metadata["missing_pairs"] == [("DEU", "deflator")]
    assert any("deflator" in k for k in panel.metadata["failed"])


def test_missing_countries_warn_with_reasons(fake_providers):
    with pytest.warns(UserWarning, match="returned nothing"):
        vintage_panel(["DEU"], variables=["deflator"], providers=["p_one"])


def test_unsupported_frequency_raises_rather_than_returning_empty():
    with pytest.raises(ValueError, match="freq='A' is not supported"):
        vintage_panel(["DEU"], freq="A")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        vintage_panel(["DEU"], providers="nope")


def test_unknown_variable_raises_rather_than_returning_empty():
    with pytest.raises(ValueError, match="unknown variable"):
        vintage_panel(["DEU"], series="not_a_variable")


# ---------------------------------------------------------------------------
# catalog override
# ---------------------------------------------------------------------------
def test_a_hand_written_override_is_key_normalised():
    """Overrides are typed by hand, so their keys are whatever the user
    wrote. An unnormalised lookup reports the country missing, which
    reads as 'no vintages exist'."""
    assert resolve_series("alfred", "DEU", "gdp_real",
                          catalog={"alfred": {"deu": {"B1GQ": "XYZ"}}}) == "XYZ"
    assert resolve_series("alfred", "DEU", "B1GQ",
                          catalog={"alfred": {"DEU": {"gdp_real": "XYZ"}}}) == "XYZ"


def test_override_replaces_rather_than_extends_the_builtin_table():
    assert resolve_series("alfred", "ESP", "gdp_real",
                          catalog={"alfred": {"DEU": {"gdp_real": "X"}}}) is None
