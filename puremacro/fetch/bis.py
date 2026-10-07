r"""One-call cross-country panels from the BIS statistics API.

``bis_panel(freq="M")`` returns every monthly series the course needs from the
Bank for International Settlements in one wide frame indexed by
``(code, date)``: effective exchange rates (nominal and real, broad and narrow
baskets), central-bank policy rates, long consumer-price indices and US-dollar
exchange rates. ``freq="Q"`` adds the quarterly macro-financial block — total
credit to the private sector by lender and borrower, the credit-to-GDP gap and
its trend, residential property prices and debt-service ratios — and
``freq="A"`` the annual long CPI back to the seventeenth century. Each
variable is one request for all economies at once, so the whole monthly block
is eight round-trips and the quarterly one about twenty.

Why a dedicated fetcher
-----------------------
The BIS publishes the longest cross-country series that exist for several of
these concepts (policy rates from 1945, CPI from 1913, credit-to-GDP from
1947), but the older readers in :mod:`puremacro.fetch.financial` asked for
the key ``ALL``, which the BIS SDMX API does not read as a wildcard: every
one of those URLs answers 404 and the functions return an empty frame or a
FRED proxy. This module speaks the API's actual grammar — version ``+``
(latest; ``WS_TC`` is at 2.0), an empty key slot as the wildcard, ``+`` as OR
within a slot — pins every non-area dimension so a series is never mixed with
its siblings (bank vs. all lenders, market vs. nominal valuation, index vs.
year-on-year), and keeps a single BIS-to-ISO3 table.

Mechanics
---------
Requests go through :func:`puremacro._http.safe_get_bytes_cached` (stdlib
``urllib``, on-disk cache outside the repository, per-host pause) with
``Accept-Encoding: gzip``, which shrinks a response ten- to eighty-fold.
The CSV is read with ``keep_default_na=False`` because ``NA`` is Namibia,
and the literal ``NaN`` the BIS writes for missing observations becomes a
float NaN. Dates sit at the start of the period. Provider aggregates
(the euro area, emerging, advanced, all reporting, G20, world, WAEMU) are
dropped unless ``aggregates=True``, so a default panel holds economies only
and a cross-country mean or count never sees the euro members twice. The
euro area (BIS ``XM``) is then the code ``"EA"``; naming it in ``codes``
(``codes=["DEU", "EA"]``) keeps it without ``aggregates=True``.

The BIS publishes the effective exchange rates and policy rates only daily
and monthly, the CPI only monthly and annually, and the credit, property and
debt-service series only quarterly. A coarser ``freq`` is built here from the
native series by the rule in each variable's ``agg`` field of
:data:`BIS_SERIES`: ``"last"`` takes the value of the final sub-period (policy
rate, end-of-period exchange rate, credit stocks) and ``"mean"`` averages the
sub-periods (indices, average exchange rates, debt-service ratios). Both
require the period to be complete — ``"mean"`` needs every sub-period,
``"last"`` needs the final one — so the current, half-finished quarter never
shows up as a quarterly average of two months.

Euro-area policy rates. The national series of the founding members end in
1998-12 (Greece 2000-12) and the euro-area rate starts 1999-01, so a member's
complete history is the national series up to its euro entry followed by the
``"EA"`` rows, requested by name. That splice is left to the caller (see
:func:`bis_panel`).

Source: BIS statistics SDMX REST API v2, ``https://stats.bis.org/api/v2``,
dataflows ``WS_EER``, ``WS_CBPOL``, ``WS_LONG_CPI``, ``WS_XRU``, ``WS_TC``,
``WS_CREDIT_GAP``, ``WS_SPP`` and ``WS_DSR``.
"""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import warnings
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

_BASE = "https://stats.bis.org/api/v2"
_SOURCE = "BIS statistics API (stats.bis.org/api/v2)"

#: BIS two-letter area code -> ISO3. The BIS uses ISO 3166 alpha-2 for
#: economies plus a few codes of its own: ``TW`` Chinese Taipei, ``AN`` the
#: former Netherlands Antilles. The euro area ``XM`` is an aggregate and lives
#: in :data:`BIS_AGGREGATES`.
#: Built from the ``WS_XRU`` availability list (192 areas, the widest flow)
#: and the World Bank country list on 7 Oct 2026.
BIS_TO_ISO3: dict[str, str] = {
    "AE": "ARE", "AF": "AFG", "AG": "ATG", "AI": "AIA", "AL": "ALB", "AM": "ARM", "AN": "ANT", "AO": "AGO",
    "AR": "ARG", "AT": "AUT", "AU": "AUS", "AW": "ABW", "AZ": "AZE", "BA": "BIH", "BB": "BRB", "BD": "BGD",
    "BE": "BEL", "BF": "BFA", "BG": "BGR", "BH": "BHR", "BI": "BDI", "BJ": "BEN", "BN": "BRN", "BO": "BOL",
    "BR": "BRA", "BS": "BHS", "BT": "BTN", "BW": "BWA", "BY": "BLR", "BZ": "BLZ", "CA": "CAN", "CD": "COD",
    "CF": "CAF", "CG": "COG", "CH": "CHE", "CI": "CIV", "CL": "CHL", "CM": "CMR", "CN": "CHN", "CO": "COL",
    "CR": "CRI", "CV": "CPV", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DJ": "DJI", "DK": "DNK", "DM": "DMA",
    "DO": "DOM", "DZ": "DZA", "EE": "EST", "EG": "EGY", "ER": "ERI", "ES": "ESP", "ET": "ETH", "FI": "FIN",
    "FJ": "FJI", "FM": "FSM", "FR": "FRA", "GA": "GAB", "GB": "GBR", "GD": "GRD", "GE": "GEO", "GH": "GHA",
    "GM": "GMB", "GN": "GIN", "GQ": "GNQ", "GR": "GRC", "GT": "GTM", "GW": "GNB", "GY": "GUY", "HK": "HKG",
    "HN": "HND", "HR": "HRV", "HT": "HTI", "HU": "HUN", "ID": "IDN", "IE": "IRL", "IL": "ISR", "IN": "IND",
    "IQ": "IRQ", "IR": "IRN", "IS": "ISL", "IT": "ITA", "JM": "JAM", "JO": "JOR", "JP": "JPN", "KE": "KEN",
    "KG": "KGZ", "KH": "KHM", "KI": "KIR", "KM": "COM", "KN": "KNA", "KR": "KOR", "KW": "KWT", "KY": "CYM",
    "KZ": "KAZ", "LA": "LAO", "LB": "LBN", "LC": "LCA", "LK": "LKA", "LR": "LBR", "LS": "LSO", "LT": "LTU",
    "LU": "LUX", "LV": "LVA", "LY": "LBY", "MA": "MAR", "MD": "MDA", "ME": "MNE", "MG": "MDG", "MK": "MKD",
    "ML": "MLI", "MM": "MMR", "MN": "MNG", "MO": "MAC", "MR": "MRT", "MS": "MSR", "MT": "MLT", "MU": "MUS",
    "MV": "MDV", "MW": "MWI", "MX": "MEX", "MY": "MYS", "MZ": "MOZ", "NA": "NAM", "NE": "NER", "NG": "NGA",
    "NI": "NIC", "NL": "NLD", "NO": "NOR", "NP": "NPL", "NZ": "NZL", "OM": "OMN", "PA": "PAN", "PE": "PER",
    "PG": "PNG", "PH": "PHL", "PK": "PAK", "PL": "POL", "PT": "PRT", "PY": "PRY", "QA": "QAT", "RO": "ROU",
    "RS": "SRB", "RU": "RUS", "RW": "RWA", "SA": "SAU", "SB": "SLB", "SC": "SYC", "SD": "SDN", "SE": "SWE",
    "SG": "SGP", "SI": "SVN", "SK": "SVK", "SL": "SLE", "SM": "SMR", "SN": "SEN", "SR": "SUR", "SS": "SSD",
    "ST": "STP", "SV": "SLV", "SY": "SYR", "SZ": "SWZ", "TD": "TCD", "TG": "TGO", "TH": "THA", "TJ": "TJK",
    "TM": "TKM", "TN": "TUN", "TO": "TON", "TR": "TUR", "TT": "TTO", "TW": "TWN", "TZ": "TZA", "UA": "UKR",
    "UG": "UGA", "US": "USA", "UY": "URY", "UZ": "UZB", "VC": "VCT", "VE": "VEN", "VN": "VNM", "VU": "VUT",
    "WS": "WSM", "YE": "YEM", "ZA": "ZAF", "ZM": "ZMB", "ZW": "ZWE",
}

#: BIS aggregate area code -> code used when ``aggregates=True`` or when
#: the code is named in ``codes``. Dropped by default. ``XW`` is the world aggregate in ``WS_SPP`` and the SDR in
#: ``WS_XRU``; ``WA`` is the West African Economic and Monetary Union.
BIS_AGGREGATES: dict[str, str] = {
    "XM": "EA",    # euro area (changing composition)
    "4T": "EME",   # emerging market economies
    "5A": "ALL",   # all reporting economies
    "5R": "ADV",   # advanced economies
    "G2": "G20",   # G20 economies
    "XW": "WLD",   # world (SDR in WS_XRU)
    "WA": "WAEMU",  # West African Economic and Monetary Union (WS_XRU)
}

_ISO3_TO_BIS: dict[str, str] = {v: k for k, v in BIS_TO_ISO3.items()}
_ISO3_TO_BIS.update({v: k for k, v in BIS_AGGREGATES.items()})


def _s(flow, key, area, freq, agg, units, desc, *, unit_mult=0, scale=1.0,
       dataonly=True):
    return {"flow": flow, "key": key, "area": area, "freq": freq, "agg": agg,
            "units": units, "unit_mult": unit_mult, "scale": scale, "sa": False,
            "dataonly": dataonly, "description": desc}


#: variable -> BIS series spec. ``key`` is the SDMX key in the flow's DSD order
#: with ``{a}`` in the area slot; ``freq`` the native frequency; ``agg`` the
#: rule used to build a coarser frequency (``"mean"`` of complete periods or
#: ``"last"`` sub-period); ``scale`` converts BIS units to the house units
#: (credit levels: BIS billions -> millions). DSD orders:
#: WS_EER FREQ.EER_TYPE(N/R).EER_BASKET(B/N).REF_AREA;
#: WS_CBPOL FREQ.REF_AREA; WS_LONG_CPI FREQ.REF_AREA.UNIT_MEASURE(628/771);
#: WS_XRU FREQ.REF_AREA.CURRENCY.COLLECTION(A avg/E eop);
#: WS_TC FREQ.BORROWERS_CTY.TC_BORROWERS(C/G/H/N/P).TC_LENDERS(A/B)
#: .VALUATION(M/N).UNIT_TYPE(770/799/USD/XDC).TC_ADJUST(A/U);
#: WS_CREDIT_GAP FREQ.BORROWERS_CTY.TC_BORROWERS.TC_LENDERS.CG_DTYPE(A/B/C);
#: WS_SPP FREQ.REF_AREA.VALUE(N/R).UNIT_MEASURE(628/771);
#: WS_DSR FREQ.BORROWERS_CTY.DSR_BORROWERS(P/H/N).
BIS_SERIES: dict[str, dict] = {
    # monthly
    "neer": _s("WS_EER", "M.N.B.{a}", "REF_AREA", "M", "mean", "index, 2020=100",
               "Nominal effective exchange rate, broad basket (64 economies, from 1994)"),
    "reer": _s("WS_EER", "M.R.B.{a}", "REF_AREA", "M", "mean", "index, 2020=100",
               "Real (CPI-based) effective exchange rate, broad basket (64 economies, from 1994)"),
    "neer_narrow": _s("WS_EER", "M.N.N.{a}", "REF_AREA", "M", "mean", "index, 2020=100",
                      "Nominal effective exchange rate, narrow basket (from 1964)"),
    "reer_narrow": _s("WS_EER", "M.R.N.{a}", "REF_AREA", "M", "mean", "index, 2020=100",
                      "Real effective exchange rate, narrow basket (from 1964)"),
    "policy_rate": _s("WS_CBPOL", "M.{a}", "REF_AREA", "M", "last",
                      "percent per annum, end of period",
                      "Central bank policy rate (rate used is in the COMPILATION attribute)"),
    "cpi": _s("WS_LONG_CPI", "M.{a}.628", "REF_AREA", "M", "mean", "index, 2010=100",
              "Consumer prices, long series (AU and NZ: quarterly index repeated)"),
    "xr_usd": _s("WS_XRU", "M.{a}..A", "REF_AREA", "M", "mean",
                 "national currency per USD, period average",
                 "US dollar exchange rate, average of period (euro members in EUR before 1999)"),
    "xr_usd_eop": _s("WS_XRU", "M.{a}..E", "REF_AREA", "M", "last",
                     "national currency per USD, end of period",
                     "US dollar exchange rate, end of period"),
    # quarterly
    "credit_gdp": _s("WS_TC", "Q.{a}.P.A.M.770.A", "BORROWERS_CTY", "Q", "last", "percent of GDP",
                     "Credit to the private non-financial sector from all lenders, market value",
                     dataonly=False),
    "credit": _s("WS_TC", "Q.{a}.P.A.M.XDC.A", "BORROWERS_CTY", "Q", "last",
                 "millions of national currency",
                 "Credit to the private non-financial sector from all lenders, break-adjusted",
                 unit_mult=6, scale=1e3, dataonly=False),
    "credit_usd": _s("WS_TC", "Q.{a}.P.A.M.USD.A", "BORROWERS_CTY", "Q", "last", "millions of USD",
                     "Credit to the private non-financial sector from all lenders, in US dollars",
                     unit_mult=6, scale=1e3, dataonly=False),
    "credit_bank_gdp": _s("WS_TC", "Q.{a}.P.B.M.770.A", "BORROWERS_CTY", "Q", "last", "percent of GDP",
                          "Bank credit to the private non-financial sector", dataonly=False),
    "credit_hh_gdp": _s("WS_TC", "Q.{a}.H.A.M.770.A", "BORROWERS_CTY", "Q", "last", "percent of GDP",
                        "Credit to households and NPISH from all lenders", dataonly=False),
    "credit_nfc_gdp": _s("WS_TC", "Q.{a}.N.A.M.770.A", "BORROWERS_CTY", "Q", "last", "percent of GDP",
                         "Credit to non-financial corporations from all lenders", dataonly=False),
    "credit_gov_gdp": _s("WS_TC", "Q.{a}.G.A.N.770.A", "BORROWERS_CTY", "Q", "last", "percent of GDP",
                         "Credit to general government, nominal value", dataonly=False),
    "credit_gap": _s("WS_CREDIT_GAP", "Q.{a}.P.A.C", "BORROWERS_CTY", "Q", "last",
                     "percentage points of GDP",
                     "Credit-to-GDP gap (actual minus one-sided HP trend, lambda 400,000)",
                     dataonly=False),
    "credit_trend": _s("WS_CREDIT_GAP", "Q.{a}.P.A.B", "BORROWERS_CTY", "Q", "last", "percent of GDP",
                       "Credit-to-GDP trend (one-sided HP filter)", dataonly=False),
    "house_price": _s("WS_SPP", "Q.{a}.N.628", "REF_AREA", "Q", "mean", "index, 2010=100",
                      "Residential property prices, nominal", dataonly=False),
    "house_price_real": _s("WS_SPP", "Q.{a}.R.628", "REF_AREA", "Q", "mean", "index, 2010=100",
                           "Residential property prices, deflated by CPI", dataonly=False),
    "dsr": _s("WS_DSR", "Q.{a}.P", "BORROWERS_CTY", "Q", "mean", "percent of income",
              "Debt service ratio, private non-financial sector", dataonly=False),
    "dsr_hh": _s("WS_DSR", "Q.{a}.H", "BORROWERS_CTY", "Q", "mean", "percent of income",
                 "Debt service ratio, households and NPISH", dataonly=False),
    "dsr_nfc": _s("WS_DSR", "Q.{a}.N", "BORROWERS_CTY", "Q", "mean", "percent of income",
                  "Debt service ratio, non-financial corporations", dataonly=False),
    # annual
    "cpi_a": _s("WS_LONG_CPI", "A.{a}.628", "REF_AREA", "A", "mean", "index, 2010=100",
                "Consumer prices, long annual series (GB from 1661)", dataonly=False),
}

#: Frequency letter -> how many native sub-periods make one period.
_N_SUB = {("M", "Q"): 3, ("M", "A"): 12, ("Q", "A"): 4}
_ORDER = {"M": 0, "Q": 1, "A": 2}
_FREQS = ("M", "Q", "A")


def _get(url: str, **kw) -> bytes:
    """The one HTTP call of this module (tests patch this name)."""
    from puremacro._http import safe_get_bytes_cached
    return safe_get_bytes_cached(url, **kw)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fmt_period(ts: pd.Timestamp, freq: str) -> str:
    if freq == "A":
        return f"{ts.year:04d}"
    if freq == "Q":
        return f"{ts.year:04d}-Q{(ts.month - 1) // 3 + 1}"
    return f"{ts.year:04d}-{ts.month:02d}"


def _parse_period(s: pd.Series) -> pd.Series:
    """BIS TIME_PERIOD (``2026``, ``2026-Q1``, ``2026-08``, ``2026-08-31``) ->
    period-start Timestamps."""
    def norm(x: str) -> str:
        x = x.strip()
        if len(x) == 4 and x.isdigit():
            return x + "-01-01"
        if len(x) == 7 and x[4:6] == "-Q" and x[6] in "1234":
            return f"{x[:4]}-{(int(x[6]) - 1) * 3 + 1:02d}-01"
        if len(x) == 7 and x[4] == "-":
            return x + "-01"
        return x[:10]
    vals = [norm(str(x)) for x in s.tolist()]
    arr = np.empty(len(vals), dtype="datetime64[D]")
    for i, v in enumerate(vals):
        try:
            arr[i] = np.datetime64(v, "D")
        except ValueError:
            arr[i] = np.datetime64("NaT")
    # nanosecond timestamps stop at 1677-09; the annual CPI starts in 1661
    good = arr[~np.isnat(arr)]
    unit = "ns" if good.size == 0 or good.min() >= np.datetime64("1678-01-01") else "us"
    return pd.Series(arr.astype(f"datetime64[{unit}]"), index=s.index)


def _status_of(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
        return "timeout"
    return f"{type(exc).__name__}: {exc}"[:200]


def _data_url(flow: str, key: str, *, start=None, end=None, last_n=None,
              first_n=None, dataonly=False) -> str:
    params = [("format", "csv")]
    if start:
        params.append(("startPeriod", str(start)))
    if end:
        params.append(("endPeriod", str(end)))
    if last_n:
        params.append(("lastNObservations", str(int(last_n))))
    if first_n:
        params.append(("firstNObservations", str(int(first_n))))
    if dataonly:
        params.append(("detail", "dataonly"))
    q = urllib.parse.urlencode(params, safe=":")
    return f"{_BASE}/data/dataflow/BIS/{flow}/+/{key}?{q}"


def _read(flow, key, *, start=None, end=None, last_n=None, dataonly=False,
          refresh=False, timeout=120, pause=3.0) -> pd.DataFrame:
    """Fetch and parse one SDMX-CSV response; never raises on provider failure."""
    url = _data_url(flow, key, start=start, end=end, last_n=last_n, dataonly=dataonly)
    try:
        raw = _get(url, timeout=timeout, headers={"Accept-Encoding": "gzip"},
                   rate_limit_seconds=pause, refresh=refresh)
    except Exception as exc:  # any transport failure (HTTPError, timeout, IncompleteRead)
        out = pd.DataFrame()
        out.attrs.update(url=url, status=_status_of(exc))
        return out
    try:
        # NA is Namibia: no default NA strings; "NaN" is handled by to_numeric.
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False,
                         na_values=[])
    except Exception as exc:  # malformed body (HTML error page, truncation)
        out = pd.DataFrame()
        out.attrs.update(url=url, status=f"parse error: {type(exc).__name__}")
        return out
    if "OBS_VALUE" not in df.columns or "TIME_PERIOD" not in df.columns:
        out = pd.DataFrame()
        out.attrs.update(url=url, status="unexpected response (no OBS_VALUE/TIME_PERIOD)")
        return out
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"].replace("", np.nan), errors="coerce")
    df.attrs.update(url=url, status="ok")
    return df


def bis_get(flow: str, key: str = "", *, start=None, end=None, last_n=None,
            dataonly: bool = False, refresh: bool = False, timeout: float = 120,
            pause: float = 3.0) -> pd.DataFrame:
    """Raw SDMX-CSV rows of one BIS dataflow query.

    Parameters
    ----------
    flow : str
        BIS dataflow id, e.g. ``"WS_EER"``, ``"WS_TC"``. The latest version
        (``+``) is always requested.
    key : str
        SDMX key in the flow's dimension order. An empty slot is a wildcard,
        ``+`` is OR within a slot (``"M.R.B.US+MX"``); ``"ALL"`` is *not* a
        wildcard at the BIS. ``""`` asks for the whole flow — avoid it for
        the large flows (``WS_XRU``, ``WS_EER``, ``WS_CBPOL`` carry daily data).
    start, end : str, optional
        ``startPeriod`` / ``endPeriod`` in the flow's format (``"1990"``,
        ``"1990-01"``, ``"1990-Q1"``).
    last_n : int, optional
        ``lastNObservations`` per series.
    dataonly : bool
        ``detail=dataonly``: keep only the dimensions, ``TIME_PERIOD`` and
        ``OBS_VALUE`` (drops long attribute columns; much smaller).
    refresh : bool
        Re-download instead of using the on-disk cache.
    timeout, pause : float
        Seconds for the HTTP timeout and the minimum spacing between requests
        to ``stats.bis.org``.

    Returns
    -------
    pandas.DataFrame
        One row per observation, every column a string except ``OBS_VALUE``
        (float, NaN where the BIS writes ``NaN``). ``attrs["url"]``,
        ``attrs["status"]`` (``"ok"`` or e.g. ``"HTTP 404"``) and
        ``attrs["fetched_at"]``. A provider failure returns an empty frame
        and warns; it never raises.

    Examples
    --------
    >>> bis_get("WS_EER", "M.R.B.MX", start="2026")  # doctest: +SKIP
    """
    df = _read(flow, key, start=start, end=end, last_n=last_n, dataonly=dataonly,
               refresh=refresh, timeout=timeout, pause=pause)
    df.attrs["fetched_at"] = _now()
    if df.attrs.get("status") != "ok":
        warnings.warn(f"BIS {flow}/{key or '*'}: {df.attrs.get('status')} — empty frame "
                      f"({df.attrs.get('url')})", RuntimeWarning, stacklevel=2)
    return df


def _area_code(bis: str, aggregates: bool) -> str | None:
    if bis in BIS_TO_ISO3:
        return BIS_TO_ISO3[bis]
    if aggregates and bis in BIS_AGGREGATES:
        return BIS_AGGREGATES[bis]
    return None


def bis_countries(flow: str, *, refresh: bool = False, aggregates: bool = False,
                  timeout: float = 120) -> list[str]:
    """Economies (ISO3) that a BIS dataflow covers.

    Reads the flow's availability constraint
    (``/availability/dataflow/BIS/{flow}/+/*?mode=exact``), so the list is the
    union over every series of the flow — e.g. ``WS_EER`` gives the 64 broad
    economies although only 27 have a narrow index. Aggregates, the euro
    area ``"EA"`` among them, are dropped unless ``aggregates=True``. A provider failure warns and returns ``[]``.

    Examples
    --------
    >>> bis_countries("WS_CBPOL")[:3]  # doctest: +SKIP
    ['ARG', 'AUS', 'AUT']
    """
    url = f"{_BASE}/availability/dataflow/BIS/{flow}/+/*?mode=exact"
    try:
        raw = _get(url, timeout=timeout, headers={"Accept-Encoding": "gzip"},
                   rate_limit_seconds=3.0, refresh=refresh)
        doc = json.loads(raw.decode("utf-8"))
        regions = doc["data"]["dataConstraints"][0]["cubeRegions"]
    except Exception as exc:  # transport failure or an unexpected JSON shape
        warnings.warn(f"BIS availability for {flow}: {_status_of(exc)}; returning []",
                      RuntimeWarning, stacklevel=2)
        return []
    areas: set[str] = set()
    for reg in regions:
        for kv in reg.get("keyValues", []):
            if kv.get("id") in ("REF_AREA", "BORROWERS_CTY"):
                areas.update(v["value"] if isinstance(v, dict) else v for v in kv["values"])
    out = {c for c in (_area_code(a, aggregates) for a in areas) if c}
    return sorted(out)


def _empty_panel(columns: Iterable[str], missing=(), source=_SOURCE) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype=object), pd.DatetimeIndex([], dtype="datetime64[ns]")],
        names=["code", "date"])
    out = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns}, index=idx)
    out.attrs.update(meta=(), source=source, fetched_at=_now(), missing=tuple(missing))
    return out


def _resolve_codes(codes) -> list[str] | None:
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    bad = [c for c in codes if str(c).upper() not in _ISO3_TO_BIS]
    if bad:
        raise ValueError(
            f"unknown code(s) {bad}: use ISO3 codes (euro area 'EA'); valid codes are "
            f"the values of puremacro.fetch.bis.BIS_TO_ISO3 ({len(BIS_TO_ISO3)} areas) "
            f"and of BIS_AGGREGATES {sorted(BIS_AGGREGATES.values())}")
    return [str(c).upper() for c in codes]


def _variables_for(freq: str) -> list[str]:
    return [v for v, s in BIS_SERIES.items() if _ORDER[s["freq"]] <= _ORDER[freq]
            and not (s["freq"] == "A" and freq != "A")]


def _aggregate(long: pd.DataFrame, native: str, freq: str, rule: str) -> pd.DataFrame:
    """long [code, date, value] at ``native`` -> ``freq`` by ``rule``,
    complete periods only."""
    if native == freq or long.empty:
        return long
    pf = "Y" if freq == "A" else freq
    per = long["date"].dt.to_period(pf).dt.start_time
    if rule == "last":
        # the final sub-period of the target period must be present
        nxt = long["date"] + (pd.offsets.MonthBegin(1) if native == "M" else pd.offsets.QuarterBegin(1, startingMonth=1))
        is_last = nxt.dt.to_period(pf).dt.start_time != per
        out = long.loc[is_last & long["value"].notna(), ["code", "value"]].copy()
        out["date"] = per[out.index]
        return out[["code", "date", "value"]]
    g = long.assign(date=per).dropna(subset=["value"]).groupby(["code", "date"])["value"]
    res = g.agg(["mean", "count"]).reset_index()
    res = res[res["count"] == _N_SUB[(native, freq)]]
    return res.rename(columns={"mean": "value"})[["code", "date", "value"]]


def bis_panel(codes=None, *, variables=None, freq: str = "M", start=None,
              aggregates: bool = False, refresh: bool = False) -> pd.DataFrame:
    """Wide cross-country BIS panel indexed by ``(code, date)``.

    Parameters
    ----------
    codes : list of str, optional
        ISO3 codes, or a code of :data:`BIS_AGGREGATES` such as ``"EA"`` (the
        euro area), which is then kept. Default: every economy the flow
        publishes, without aggregates.
    variables : list of str, optional
        Keys of :data:`BIS_SERIES`. Default: every variable available at
        ``freq`` — monthly ``neer reer neer_narrow reer_narrow policy_rate
        cpi xr_usd xr_usd_eop``; quarterly adds ``credit_gdp credit
        credit_usd credit_bank_gdp credit_hh_gdp credit_nfc_gdp
        credit_gov_gdp credit_gap credit_trend house_price house_price_real
        dsr dsr_hh dsr_nfc``; annual adds ``cpi_a`` (the native annual CPI,
        much longer than the annual mean of the monthly ``cpi``).
    freq : {"M", "Q", "A"}
        Output frequency. Variables published at a finer frequency are
        aggregated by their ``agg`` rule (``"last"``: final sub-period, e.g.
        the policy rate at the end of the quarter; ``"mean"``: average of a
        complete period). A variable published only at a coarser frequency
        than ``freq`` is a caller error.
    start : str, optional
        First period, e.g. ``"1990"`` or ``"1990-01"``. Default: the full history.
    aggregates : bool
        Keep BIS aggregates (``EME``, ``ADV``, ``ALL``, ``G20``, ``WLD``,
        ``WAEMU`` and the euro area ``EA``; see :data:`BIS_AGGREGATES`) when
        ``codes`` is None. An aggregate named in ``codes`` is always kept.
    refresh : bool
        Re-download instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        One float column per variable, ``MultiIndex (code, date)`` with
        period-start dates; rows where every column is missing are dropped.
        The date level is ``datetime64[ns]`` except when an observation
        precedes 1678 (only ``cpi_a``: the UK from 1661), which nanoseconds
        cannot hold; the level is then ``datetime64[us]``. Cast with
        ``.astype("datetime64[ns]")`` after a ``start`` past 1677, or join on
        the date level as is (pandas aligns the resolutions).
        ``attrs["meta"]`` is a tuple of dicts, one per (code, variable):
        ``code, variable, flow, key, units, unit_mult, sa, freq_native, agg,
        first, last, n``. ``attrs["source"]``, ``attrs["fetched_at"]`` (UTC)
        and ``attrs["missing"]``: dicts ``{variable, code, reason}`` for every
        variable that failed (``code`` None) and every requested code it does
        not cover. A provider failure warns and yields an empty frame with
        the same index names and attrs keys; it never raises.

    Notes
    -----
    Euro-area policy rate, full history for a member::

        p = bis_panel(["DEU", "EA"], variables=["policy_rate"])["policy_rate"]
        de = pd.concat([p.loc["DEU"][:"1998-12"], p.loc["EA"]["1999-01":]])

    (Greece's national series runs to 2000-12, later members' to their
    euro-entry date.)

    Examples
    --------
    >>> bis_panel(["MEX", "USA"], variables=["reer", "policy_rate"])  # doctest: +SKIP
    >>> bis_panel(freq="Q", start="1960")                             # doctest: +SKIP
    """
    freq = str(freq).upper()
    if freq not in _FREQS:
        raise ValueError(f"freq must be one of {_FREQS}, got {freq!r}")
    avail = _variables_for(freq)
    if variables is None:
        variables = avail
    elif isinstance(variables, str):
        variables = [variables]
    variables = list(variables)
    unknown = [v for v in variables if v not in BIS_SERIES]
    if unknown:
        raise ValueError(f"unknown BIS variable(s) {unknown}; valid: {list(BIS_SERIES)}")
    too_coarse = [v for v in variables if v not in avail]
    if too_coarse:
        raise ValueError(
            f"variable(s) {too_coarse} are not available at freq={freq!r} "
            f"(native frequency coarser than requested); valid at {freq!r}: {avail}")
    as_list = [codes] if isinstance(codes, str) else list(codes or [])
    if any(c in BIS_SERIES for c in as_list):
        raise ValueError(
            f"codes={as_list} holds variable names; the first argument of "
            f"bis_panel is codes (ISO3). Pass variables=[...]; valid: {list(BIS_SERIES)}")
    want = _resolve_codes(codes)
    if want is not None and any(c in BIS_AGGREGATES.values() for c in want):
        aggregates = True  # an aggregate asked for by name is kept
    start_ts = pd.Timestamp(str(start)) if start is not None else None

    frames: list[pd.Series] = []
    meta: list[dict] = []
    missing: list[dict] = []
    flows_used: list[str] = []
    for var in variables:
        spec = BIS_SERIES[var]
        area = "" if want is None else "+".join(_ISO3_TO_BIS[c] for c in want)
        key = spec["key"].format(a=area)
        sp = _fmt_period(start_ts, spec["freq"]) if start_ts is not None else None
        raw = _read(spec["flow"], key, start=sp, dataonly=spec["dataonly"],
                    refresh=refresh)
        status = raw.attrs.get("status")
        if status != "ok" or raw.empty:
            reason = status if status != "ok" else "no rows"
            missing.append({"variable": var, "code": None, "reason": reason})
            continue
        if spec["flow"] not in flows_used:
            flows_used.append(spec["flow"])
        acol = spec["area"]
        raw = raw.copy()
        raw["code"] = raw[acol].map(lambda a: _area_code(a, aggregates))
        raw["date"] = _parse_period(raw["TIME_PERIOD"])
        raw = raw.dropna(subset=["code", "date"])
        if want is not None:  # naming "EA" must not let other aggregates in
            raw = raw[raw["code"].isin(want)]
        scale = spec["scale"]
        if "UNIT_MULT" in raw.columns and spec["unit_mult"]:
            # rescale by the reported multiplier so millions are exact even if
            # the BIS changes UNIT_MULT for a series
            um = pd.to_numeric(raw["UNIT_MULT"], errors="coerce").fillna(9)
            raw["value"] = raw["OBS_VALUE"] * 10.0 ** (um - spec["unit_mult"])
        else:
            raw["value"] = raw["OBS_VALUE"] * scale
        cur = None
        if "UNIT_MEASURE" in raw.columns and var == "credit":
            cur = raw.groupby("code")["UNIT_MEASURE"].first().to_dict()
        long = raw[["code", "date", "value"]]
        long = _aggregate(long, spec["freq"], freq, spec["agg"])
        if start_ts is not None:
            long = long[long["date"] >= start_ts]
        long = long.dropna(subset=["value"])
        if long.duplicated(["code", "date"]).any():
            # a key that matched more than one series per area would be a bug
            warnings.warn(f"BIS {var}: duplicate (code, date) rows; keeping the first",
                          RuntimeWarning, stacklevel=2)
            long = long.drop_duplicates(["code", "date"])
        if long.empty:
            missing.append({"variable": var, "code": None, "reason": "no observations"})
            continue
        s = long.set_index(["code", "date"])["value"].rename(var)
        frames.append(s)
        serieskeys = raw.groupby("code")[acol].first().to_dict()
        for code, grp in long.groupby("code"):
            d = {"code": code, "variable": var, "flow": spec["flow"],
                 "key": spec["key"].format(a=serieskeys.get(code, "")),
                 "units": spec["units"], "unit_mult": spec["unit_mult"], "sa": spec["sa"],
                 "freq_native": spec["freq"], "agg": spec["agg"] if spec["freq"] != freq else None,
                 "first": grp["date"].min().strftime("%Y-%m-%d"),
                 "last": grp["date"].max().strftime("%Y-%m-%d"), "n": int(len(grp))}
            if cur is not None:
                d["currency"] = cur.get(code)
            meta.append(d)
        if want is not None:
            got = set(long["code"])
            missing.extend({"variable": var, "code": c, "reason": "not published"}
                           for c in want if c not in got)

    source = _SOURCE + (": " + ", ".join(flows_used) if flows_used else "")
    if missing:
        failed = sorted({m["variable"] for m in missing if m["code"] is None})
        if failed:
            warnings.warn(f"BIS: no data for {failed} "
                          f"({', '.join(sorted({m['reason'] for m in missing if m['code'] is None}))}); "
                          "see attrs['missing']", RuntimeWarning, stacklevel=2)
    if not frames:
        return _empty_panel(variables, missing, source)
    out = pd.concat(frames, axis=1)
    out = out.reindex(columns=[v for v in variables if v in out.columns])
    for v in variables:
        if v not in out.columns:
            out[v] = np.nan
    out = out[variables].astype("float64")
    out = out.dropna(how="all").sort_index()
    out.index = out.index.set_names(["code", "date"])
    out.attrs.update(meta=tuple(meta), source=source, fetched_at=_now(),
                     missing=tuple(missing))
    return out


def bis_eer(kind: str = "real", basket: str = "broad", *, freq: str = "M", codes=None,
            start=None, refresh: bool = False) -> pd.DataFrame:
    """BIS effective exchange rate as a one-column :func:`bis_panel`.

    ``kind`` is ``"real"`` or ``"nominal"``, ``basket`` is ``"broad"``
    (64 economies, from 1994) or ``"narrow"`` (27 economies, from 1964; no
    nominal narrow index for Mexico). Index 2020=100; an increase is an
    appreciation. The column is named as in :data:`BIS_SERIES`
    (``reer``, ``neer``, ``reer_narrow``, ``neer_narrow``).
    """
    kinds = {"real": "reer", "nominal": "neer"}
    baskets = {"broad": "", "narrow": "_narrow"}
    if kind not in kinds:
        raise ValueError(f"kind must be one of {list(kinds)}, got {kind!r}")
    if basket not in baskets:
        raise ValueError(f"basket must be one of {list(baskets)}, got {basket!r}")
    return bis_panel(codes, variables=[kinds[kind] + baskets[basket]], freq=freq,
                     start=start, refresh=refresh)


def bis_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """``panel.attrs["meta"]`` of a :func:`bis_panel` frame as a DataFrame."""
    return pd.DataFrame(list(panel.attrs.get("meta", ())))


__all__ = [
    "BIS_SERIES",
    "BIS_TO_ISO3",
    "BIS_AGGREGATES",
    "bis_get",
    "bis_countries",
    "bis_panel",
    "bis_eer",
    "bis_meta",
]
