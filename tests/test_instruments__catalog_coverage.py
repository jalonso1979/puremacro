"""Coverage tests for puremacro/instruments/_catalog.py.

Targets the following previously uncovered lines (69% -> higher):

  Line 48   — _wrap_narrative: single NarrativeInstrument (non-dict) branch.
  Lines 205-210 — _wrap_connector: happy path when country is resolved.
  Lines 357-366 — _load_gk2015_ffr_surprise: GK2015 HFI loader body.
  Line 529    — _make_fred_loader inner _load closure.
  Line 609    — _load_bis_credit_gap_us body.
  Line 633    — _make_weo_loader inner _load closure.
  Lines 681-689 — _fred_csv_loader: series_id in df.columns branch AND fallback.
  Lines 725-737 — _bis_neer_us_loader: 'value' branch, numeric fallback, iloc fallback.
  Lines 770-771 — _make_oecd_stan_loader inner _load closure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import puremacro.instruments._catalog as cat
from puremacro.instruments._catalog import (
    _load_gk2015_ffr_surprise,
    _wrap_connector,
    _wrap_narrative,
    _fred_csv_loader,
    _bis_neer_us_loader,
    _make_fred_loader,
    _make_weo_loader,
    _load_bis_credit_gap_us,
    _make_oecd_stan_loader,
)
from puremacro.instruments._core import Instrument


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_narrative_event(country: str = "USA") -> "puremacro.narrative.NarrativeEvent":
    from puremacro.narrative import NarrativeEvent
    return NarrativeEvent(
        date=pd.Timestamp("2001-01-01"),
        country=country,
        magnitude=1.0,
        magnitude_unit="USD_bn",
        target="both",
        subtarget=None,
        sign=1,
        confidence=1.0,
        source_text="tax cut policy",
        source_url="http://example.com",
        scoring_method="manual",
        metadata={},
    )


def _make_narrative_instrument(country: str = "USA"):
    from puremacro.narrative import NarrativeInstrument
    ev = _make_narrative_event(country)
    return NarrativeInstrument.from_events([ev])


# ---------------------------------------------------------------------------
# _wrap_narrative: single (non-dict) NarrativeInstrument path  (line 48)
# ---------------------------------------------------------------------------

class TestWrapNarrativeSinglePath:
    """_wrap_narrative must handle a loader that returns a single
    NarrativeInstrument (not a dict) and attach registry_key + source
    metadata before calling as_instrument()."""

    def test_single_narr_returns_instrument(self):
        ni = _make_narrative_instrument()
        wrapped = _wrap_narrative(lambda **_kw: ni, "reg_key_single", "src_single")
        result = wrapped()
        assert isinstance(result, Instrument)

    def test_single_narr_metadata_registry_key_set(self):
        ni = _make_narrative_instrument()
        wrapped = _wrap_narrative(lambda **_kw: ni, "my_key", "my_source")
        result = wrapped()
        assert result.metadata.get("registry_key") == "my_key"

    def test_single_narr_metadata_source_set(self):
        ni = _make_narrative_instrument()
        wrapped = _wrap_narrative(lambda **_kw: ni, "k2", "SomeSource")
        result = wrapped()
        assert result.metadata.get("source") == "SomeSource"

    def test_single_narr_kwargs_forwarded_to_loader(self):
        """kwargs passed to the wrapper are forwarded to loader_fn."""
        received = {}

        def _loader(**kw):
            received.update(kw)
            return _make_narrative_instrument()

        wrapped = _wrap_narrative(_loader, "k3", "s")
        wrapped(my_param=42)
        assert received.get("my_param") == 42

    def test_single_narr_select_country_consumed_not_forwarded(self):
        """select_country is for dict-return path; on a single-return loader
        it is silently popped (does not break the call)."""
        ni = _make_narrative_instrument()
        wrapped = _wrap_narrative(lambda **_kw: ni, "k4", "s")
        # Should not raise even though we provide select_country
        result = wrapped(select_country="USA")
        assert isinstance(result, Instrument)


# ---------------------------------------------------------------------------
# _wrap_connector: happy path when country resolves  (lines 205-210)
# ---------------------------------------------------------------------------

class TestWrapConnectorHappyPath:
    """_wrap_connector happy path: records from iter_fn are scored and
    assembled into an Instrument when country is provided."""

    def _make_records(self):
        return [
            (pd.Timestamp("2001-01-15"), "government spending increase", "http://a.com"),
            (pd.Timestamp("2001-03-10"), "tax policy reform", "http://b.com"),
        ]

    def test_single_country_connector_returns_instrument(self):
        records = self._make_records()
        wrapped = _wrap_connector(
            lambda **_kw: iter(records),
            registry_key="conn_test",
            source="test_source",
            country="USA",
        )
        result = wrapped()
        assert isinstance(result, Instrument)

    def test_single_country_connector_metadata_registry_key(self):
        records = self._make_records()
        wrapped = _wrap_connector(
            lambda **_kw: iter(records),
            registry_key="conn_reg",
            source="conn_src",
            country="USA",
        )
        result = wrapped()
        assert result.metadata.get("registry_key") == "conn_reg"

    def test_single_country_connector_metadata_source(self):
        records = self._make_records()
        wrapped = _wrap_connector(
            lambda **_kw: iter(records),
            registry_key="x",
            source="MySrc",
            country="GBR",
        )
        result = wrapped()
        assert result.metadata.get("source") == "MySrc"

    def test_cross_country_connector_with_country_kwarg_returns_instrument(self):
        """For cross-country connectors (country=None at registration),
        passing country= at load time must succeed."""
        records = self._make_records()
        wrapped = _wrap_connector(
            lambda **_kw: iter(records),
            registry_key="cross_conn",
            source="cross_src",
            country=None,
        )
        result = wrapped(country="USA")
        assert isinstance(result, Instrument)

    def test_empty_record_set_still_returns_instrument(self):
        """An iter_fn that yields nothing must still produce an Instrument
        (with an empty/zero series)."""
        wrapped = _wrap_connector(
            lambda **_kw: iter([]),
            registry_key="empty_conn",
            source="empty_src",
            country="USA",
        )
        result = wrapped()
        assert isinstance(result, Instrument)

    def test_target_kwarg_respected(self):
        """target='investment' must be stored in metadata."""
        wrapped = _wrap_connector(
            lambda **_kw: iter([]),
            registry_key="inv_conn",
            source="s",
            country="USA",
            target="investment",
        )
        result = wrapped()
        # target is written into NarrativeInstrument.from_events then
        # surfaced in as_instrument() metadata
        assert result.metadata.get("target") == "investment"


# ---------------------------------------------------------------------------
# _load_gk2015_ffr_surprise  (lines 357-366)
# ---------------------------------------------------------------------------

class TestLoadGK2015FFRSurprise:
    """GK2015 catalog loader: known-value arithmetic + contract checks."""

    def _fomc_inputs(self):
        # Three FOMC meetings in Jan, Feb, March 2001
        dates = pd.to_datetime(["2001-01-30", "2001-02-14", "2001-03-20"])
        ff_pre = np.array([4.50, 4.50, 4.25])
        ff_post = np.array([4.50, 4.25, 4.25])
        days_rem = np.array([1.0, 14.0, 11.0])
        return ff_pre, ff_post, days_rem, dates

    def test_returns_instrument_type(self):
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
        )
        assert isinstance(result, Instrument)

    def test_series_is_monthly_period_indexed(self):
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
        )
        assert isinstance(result.series.index, pd.PeriodIndex)
        # "period[M]" or "period[ME]" depending on pandas version
        assert "M" in str(result.series.index.dtype)

    def test_category_is_monetary_hfi(self):
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
        )
        assert result.category == "monetary_hfi"

    def test_name_field(self):
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
        )
        assert result.name == "gk2015_ffr_surprise"

    def test_reference_metadata_present(self):
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
        )
        assert "reference" in result.metadata
        assert "Gertler-Karadi" in result.metadata["reference"]

    def test_known_value_feb_surprise(self):
        """Feb meeting: post - pre = -0.25, days_remaining=14, days_in_month=30.
        GK formula: (post - pre) * M / days_remaining = -0.25 * 30/14 ≈ -0.535714.
        The aggregate_to_period sum for Feb 2001 must equal that value."""
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
            days_in_month=30,
        )
        feb_val = result.series[pd.Period("2001-02", "M")]
        expected = -0.25 * 30.0 / 14.0
        assert abs(feb_val - expected) < 1e-10

    def test_no_announcement_period_is_zero(self):
        """Jan 2001: pre == post so surprise = 0; period must appear with 0."""
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
        )
        jan_val = result.series[pd.Period("2001-01", "M")]
        assert jan_val == 0.0

    def test_freq_quarterly_aggregation(self):
        """freq='Q' should produce a quarterly PeriodIndex."""
        ff_pre, ff_post, days_rem, dates = self._fomc_inputs()
        result = _load_gk2015_ffr_surprise(
            ff_futures_pre=ff_pre,
            ff_futures_post=ff_post,
            days_remaining_in_month=days_rem,
            dates=dates,
            freq="Q",
        )
        # PeriodIndex dtype is "period[Q-DEC]" or "period[QE-DEC]" depending on pandas version
        assert "Q" in str(result.series.index.dtype)


# ---------------------------------------------------------------------------
# _make_fred_loader closure  (line 529)
# ---------------------------------------------------------------------------

class TestMakeFredLoaderClosure:
    """_make_fred_loader returns a closure that calls load_fred with the
    preset series_id and frequency."""

    def test_closure_calls_load_fred_with_correct_args(self):
        mock_instr = MagicMock(spec=Instrument)
        loader = _make_fred_loader("NFCI", "W")
        with patch.object(cat, "load_fred", return_value=mock_instr) as m:
            result = loader()
        m.assert_called_once_with(series_id="NFCI", frequency="W")
        assert result is mock_instr

    def test_closure_forwards_extra_kwargs(self):
        mock_instr = MagicMock(spec=Instrument)
        loader = _make_fred_loader("VIXCLS", "D")
        with patch.object(cat, "load_fred", return_value=mock_instr) as m:
            loader(extra="arg")
        call_kwargs = m.call_args.kwargs
        assert call_kwargs["extra"] == "arg"

    def test_different_series_ids_produce_separate_closures(self):
        """Each closure is independent; they do not share state."""
        loader_nfci = _make_fred_loader("NFCI", "W")
        loader_vix = _make_fred_loader("VIXCLS", "D")
        calls = []

        def _mock_load_fred(**kwargs):
            calls.append(kwargs)
            return MagicMock(spec=Instrument)

        with patch.object(cat, "load_fred", side_effect=_mock_load_fred):
            loader_nfci()
            loader_vix()

        assert calls[0]["series_id"] == "NFCI"
        assert calls[1]["series_id"] == "VIXCLS"


# ---------------------------------------------------------------------------
# _load_bis_credit_gap_us  (line 609)
# ---------------------------------------------------------------------------

class TestLoadBisCreditGapUS:
    """_load_bis_credit_gap_us must forward fixed series_id and country to load_bis."""

    def test_calls_load_bis_with_fixed_args(self):
        mock_instr = MagicMock(spec=Instrument)
        with patch.object(cat, "load_bis", return_value=mock_instr) as m:
            result = _load_bis_credit_gap_us()
        m.assert_called_once_with(series_id="credit_to_gdp_gap", country="US")
        assert result is mock_instr

    def test_extra_kwargs_forwarded(self):
        mock_instr = MagicMock(spec=Instrument)
        with patch.object(cat, "load_bis", return_value=mock_instr) as m:
            _load_bis_credit_gap_us(csv_path="/tmp/fake.csv")
        call_kw = m.call_args.kwargs
        assert call_kw.get("csv_path") == "/tmp/fake.csv"
        assert call_kw["series_id"] == "credit_to_gdp_gap"
        assert call_kw["country"] == "US"


# ---------------------------------------------------------------------------
# _make_weo_loader closure  (line 633)
# ---------------------------------------------------------------------------

class TestMakeWeoLoaderClosure:
    """_make_weo_loader returns a closure that calls load_imf_weo with the
    preset indicator and country."""

    def test_closure_calls_load_imf_weo_with_correct_args(self):
        mock_instr = MagicMock(spec=Instrument)
        loader = _make_weo_loader("GGXWDG_NGDP", "USA")
        with patch.object(cat, "load_imf_weo", return_value=mock_instr) as m:
            result = loader()
        m.assert_called_once_with(indicator="GGXWDG_NGDP", country="USA")
        assert result is mock_instr

    def test_closure_forwards_extra_kwargs(self):
        mock_instr = MagicMock(spec=Instrument)
        loader = _make_weo_loader("GGXONLB_NGDP", "USA")
        with patch.object(cat, "load_imf_weo", return_value=mock_instr) as m:
            loader(csv_path="/tmp/weo.csv")
        call_kw = m.call_args.kwargs
        assert call_kw.get("csv_path") == "/tmp/weo.csv"

    def test_different_indicators_produce_independent_closures(self):
        calls = []

        def _mock(**kw):
            calls.append(kw)
            return MagicMock(spec=Instrument)

        loader_debt = _make_weo_loader("GGXWDG_NGDP", "USA")
        loader_bal = _make_weo_loader("GGXONLB_NGDP", "USA")

        with patch.object(cat, "load_imf_weo", side_effect=_mock):
            loader_debt()
            loader_bal()

        assert calls[0]["indicator"] == "GGXWDG_NGDP"
        assert calls[1]["indicator"] == "GGXONLB_NGDP"


# ---------------------------------------------------------------------------
# _fred_csv_loader  (lines 681-689)
# ---------------------------------------------------------------------------

class TestFredCSVLoader:
    """_fred_csv_loader: two branches — (a) series_id in df.columns, and
    (b) series_id not in df.columns (fallback to first non-DATE column)."""

    def _idx(self, n=3):
        return pd.date_range("2020-01-01", periods=n, freq="QS")

    def test_series_id_in_columns_path(self):
        """Branch A: column matching series_id exists."""
        df = pd.DataFrame({"GDPC1": [100.0, 101.0, 102.0]}, index=self._idx())
        with patch("puremacro.fetch._classic.fetch_fred", return_value=df):
            result = _fred_csv_loader(series_id="GDPC1")
        assert isinstance(result, Instrument)
        assert result.name == "fetch_fred_GDPC1"
        np.testing.assert_allclose(result.series.values, [100.0, 101.0, 102.0])

    def test_series_id_not_in_columns_fallback(self):
        """Branch B: series_id absent from columns; fallback picks first
        non-DATE numeric column."""
        df = pd.DataFrame({"VALUE": [10.0, 20.0, 30.0]}, index=self._idx())
        with patch("puremacro.fetch._classic.fetch_fred", return_value=df):
            result = _fred_csv_loader(series_id="NOSUCHCOL")
        assert isinstance(result, Instrument)
        assert result.name == "fetch_fred_NOSUCHCOL"
        np.testing.assert_allclose(result.series.values, [10.0, 20.0, 30.0])

    def test_series_name_set_correctly(self):
        df = pd.DataFrame({"FEDFUNDS": [0.5, 0.5, 0.5]}, index=self._idx())
        with patch("puremacro.fetch._classic.fetch_fred", return_value=df):
            result = _fred_csv_loader(series_id="FEDFUNDS")
        assert result.series.name == "fetch_fred_FEDFUNDS"

    def test_category_is_external_csv(self):
        df = pd.DataFrame({"GDPC1": [100.0]}, index=self._idx(1))
        with patch("puremacro.fetch._classic.fetch_fred", return_value=df):
            result = _fred_csv_loader(series_id="GDPC1")
        assert result.category == "external_csv"

    def test_metadata_series_id_stored(self):
        df = pd.DataFrame({"GDPC1": [100.0]}, index=self._idx(1))
        with patch("puremacro.fetch._classic.fetch_fred", return_value=df):
            result = _fred_csv_loader(series_id="GDPC1")
        assert result.metadata.get("series_id") == "GDPC1"

    def test_metadata_reference_contains_series_id(self):
        df = pd.DataFrame({"GDPC1": [100.0]}, index=self._idx(1))
        with patch("puremacro.fetch._classic.fetch_fred", return_value=df):
            result = _fred_csv_loader(series_id="GDPC1")
        assert "GDPC1" in result.metadata.get("reference", "")

    def test_default_series_id_is_gdpc1(self):
        """Calling without series_id should default to GDPC1."""
        df = pd.DataFrame({"GDPC1": [100.0]}, index=self._idx(1))
        with patch("puremacro.fetch._classic.fetch_fred", return_value=df):
            result = _fred_csv_loader()
        assert "GDPC1" in result.name


# ---------------------------------------------------------------------------
# _bis_neer_us_loader  (lines 725-737)
# ---------------------------------------------------------------------------

class TestBisNeerUSLoader:
    """Three internal branches: 'value' column present; numeric fallback; iloc fallback."""

    def _idx(self, n=3):
        return pd.date_range("2020-01-01", periods=n, freq="ME")

    def test_value_column_branch(self):
        """Branch: df has 'value' column."""
        import puremacro.bis_neer
        df = pd.DataFrame({"value": [100.0, 101.0, 102.0]}, index=self._idx())
        with patch.object(puremacro.bis_neer, "fetch_bis_neer", return_value=df):
            result = _bis_neer_us_loader()
        assert isinstance(result, Instrument)
        np.testing.assert_allclose(result.series.values, [100.0, 101.0, 102.0])

    def test_numeric_fallback_branch(self):
        """Branch: no 'value' column, but there are numeric columns — picks first."""
        import puremacro.bis_neer
        df = pd.DataFrame({"idx": [95.0, 96.0, 97.0]}, index=self._idx())
        with patch.object(puremacro.bis_neer, "fetch_bis_neer", return_value=df):
            result = _bis_neer_us_loader()
        assert isinstance(result, Instrument)
        np.testing.assert_allclose(result.series.values, [95.0, 96.0, 97.0])

    def test_iloc_fallback_branch(self):
        """Branch: no 'value' column, no numeric dtype columns; falls back to iloc[:,0]."""
        import puremacro.bis_neer
        df = pd.DataFrame({"label": ["95.0", "96.0", "97.0"]}, index=self._idx())
        with patch.object(puremacro.bis_neer, "fetch_bis_neer", return_value=df):
            result = _bis_neer_us_loader()
        assert isinstance(result, Instrument)
        np.testing.assert_allclose(result.series.values, [95.0, 96.0, 97.0])

    def test_series_name_is_bis_neer_us(self):
        import puremacro.bis_neer
        df = pd.DataFrame({"value": [100.0]}, index=self._idx(1))
        with patch.object(puremacro.bis_neer, "fetch_bis_neer", return_value=df):
            result = _bis_neer_us_loader()
        assert result.series.name == "bis_neer_us"
        assert result.name == "bis_neer_us"

    def test_category_is_external_csv(self):
        import puremacro.bis_neer
        df = pd.DataFrame({"value": [100.0]}, index=self._idx(1))
        with patch.object(puremacro.bis_neer, "fetch_bis_neer", return_value=df):
            result = _bis_neer_us_loader()
        assert result.category == "external_csv"

    def test_frequency_is_monthly(self):
        import puremacro.bis_neer
        df = pd.DataFrame({"value": [100.0]}, index=self._idx(1))
        with patch.object(puremacro.bis_neer, "fetch_bis_neer", return_value=df):
            result = _bis_neer_us_loader()
        assert result.frequency == "M"

    def test_fetch_called_with_us_country_code(self):
        """The loader hardcodes 'US' when calling fetch_bis_neer."""
        import puremacro.bis_neer
        df = pd.DataFrame({"value": [100.0]}, index=self._idx(1))
        with patch.object(puremacro.bis_neer, "fetch_bis_neer", return_value=df) as m:
            _bis_neer_us_loader()
        args = m.call_args
        assert args.args[0] == "US"


# ---------------------------------------------------------------------------
# _make_oecd_stan_loader closure  (lines 770-771)
# ---------------------------------------------------------------------------

class TestMakeOECDStanLoaderClosure:
    """_make_oecd_stan_loader returns a closure that calls oecd_sdmx_instrument
    with the preset dataset, country, and indicator."""

    def test_closure_calls_oecd_sdmx_with_correct_args(self):
        mock_instr = MagicMock(spec=Instrument)
        loader = _make_oecd_stan_loader("USA", "VALADD")
        import puremacro.fetch.sdmx
        with patch.object(puremacro.fetch.sdmx, "oecd_sdmx_instrument",
                          return_value=mock_instr) as m:
            result = loader()
        m.assert_called_once_with(dataset="DSD_STAN", country="USA", indicator="VALADD")
        assert result is mock_instr

    def test_closure_forwards_extra_kwargs(self):
        mock_instr = MagicMock(spec=Instrument)
        loader = _make_oecd_stan_loader("USA", "EMPN")
        import puremacro.fetch.sdmx
        with patch.object(puremacro.fetch.sdmx, "oecd_sdmx_instrument",
                          return_value=mock_instr) as m:
            loader(csv_path="/tmp/stan.csv")
        call_kw = m.call_args.kwargs
        assert call_kw.get("csv_path") == "/tmp/stan.csv"
        assert call_kw["country"] == "USA"
        assert call_kw["indicator"] == "EMPN"

    def test_different_countries_produce_independent_closures(self):
        calls = []

        def _mock(**kw):
            calls.append(kw)
            return MagicMock(spec=Instrument)

        loader_usa = _make_oecd_stan_loader("USA", "VALADD")
        loader_deu = _make_oecd_stan_loader("DEU", "VALADD")

        import puremacro.fetch.sdmx
        with patch.object(puremacro.fetch.sdmx, "oecd_sdmx_instrument",
                          side_effect=_mock):
            loader_usa()
            loader_deu()

        assert calls[0]["country"] == "USA"
        assert calls[1]["country"] == "DEU"
