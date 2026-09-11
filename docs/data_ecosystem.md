> 🇬🇧 English · 🇪🇸 [Español](es/data_ecosystem.md)

# Global Macro Data Ecosystem

`puremacro` provides a unified, keyless, and cached data ecosystem for international macroeconomic research, policy analysis, and forecasting. All collectors operate through unified caching (`puremacro.fetch._http.cached_get`), adhere to strict Pyodide safety (zero module-scope `requests` imports), and return standardized long-form DataFrames (`code, date, variable, value, source`).

In addition to individual series fetchers, high-level **modular panel builders** (`puremacro.build_climate_panel` and `puremacro.build_financial_panel`) assemble multi-country, multi-frequency datasets with automated frequency rollups and missing-data tracking.

---

## 1. Greenhouse Gas & Emissions (`puremacro.fetch.emissions`)

Collects international emissions data from the World Bank World Development Indicators (WDI) and the OECD SDMX greenhouse gas database.

```python
from puremacro.fetch.emissions import (
    fetch_wdi_emissions,
    fetch_oecd_ghg_emissions,
    fetch_emissions_panel,
)

# Fetch World Bank WDI CO2 emissions per capita and total GHG
df_wdi = fetch_wdi_emissions(
    codes=["USA", "DEU", "JPN", "GBR", "MEX"],
    indicators=["EN.GHG.CO2.PC.CE.AR5", "EN.GHG.ALL.MT.CE.AR5"],
    start_year=2000,
)

# Fetch OECD sectoral greenhouse gas accounts (energy, transport, manufacturing)
df_oecd = fetch_oecd_ghg_emissions(
    codes=["USA", "DEU"],
    sectors=["1A1", "1A2", "1A3"],  # Energy, Industry, Transport
    start_period="2010",
)

# One-call cross-source emissions panel
df_emissions = fetch_emissions_panel(
    codes=["USA", "DEU", "JPN"],
    start_year=2000,
)
```

### Supported Indicators

| Module / Function | Indicator Code / Flow | Description |
|---|---|---|
| `fetch_wdi_emissions` | `EN.GHG.CO2.MT.CE.AR5` | Total CO2 emissions (million metric tons CO2 equivalent) |
| `fetch_wdi_emissions` | `EN.GHG.CO2.PC.CE.AR5` | CO2 emissions per capita (metric tons per capita) |
| `fetch_wdi_emissions` | `EN.GHG.CH4.MT.CE.AR5` | Methane (CH4) emissions |
| `fetch_wdi_emissions` | `EN.GHG.N2O.MT.CE.AR5` | Nitrous oxide (N2O) emissions |
| `fetch_oecd_ghg_emissions` | `OECD.ENV.EPI,DSD_AIR_GHG@DF_AIR_GHG,` | Sectoral GHG accounts (IPCC sectors: `1A1` Energy, `1A2` Industry, `1A3` Transport, `1A4b` Residential) |

---

## 2. Energy Balances & Transition (`puremacro.fetch.energy_transition`)

Collects electricity generation by source, clean energy transition shares, and primary energy consumption balances from open international repositories (OECD, Ember, Energy Institute).

```python
from puremacro.fetch.energy_transition import (
    fetch_electricity_generation_mix,
    fetch_energy_transition_panel,
)

# Fetch electricity generation mix (renewables, fossil, hydro, nuclear)
df_mix = fetch_electricity_generation_mix(
    codes=["USA", "DEU", "FRA", "ESP"],
    start_year=2010,
)

# Full energy transition panel with clean energy share
df_energy = fetch_energy_transition_panel(
    codes=["USA", "DEU"],
    start_year=2000,
)
```

### Key Metrics Provided
- `clean_elec_share`: Percentage of electricity generated from zero-carbon sources (renewables, hydro, nuclear).
- `fossil_elec_share`: Percentage of electricity generated from fossil fuels (coal, gas, oil).
- `elec_gen_twh`: Total annual electricity generation in terawatt-hours (TWh).
- `energy_intensity`: Energy consumption per unit of real GDP.

---

## 3. Commodity Benchmark Price Suites (`puremacro.fetch.commodities`)

Standardized benchmark spot and futures commodity price indices derived from the World Bank Pink Sheet and IMF Primary Commodity Price System (PCPS).

```python
from puremacro.fetch.commodities import (
    fetch_commodity_prices,
    fetch_commodity_indices,
    COMMODITY_CATEGORIES,
)

# Fetch raw price levels (in native units: USD/bbl, USD/mmbtu, USD/mt)
df_prices = fetch_commodity_prices(
    commodities=["oil_brent", "oil_wti", "gas_eu", "copper", "gold"],
    frequency="M",
    start_date="2000-01-01",
)

# Fetch composite group indices (2010=100)
df_indices = fetch_commodity_indices(
    categories=["energy", "metals", "agriculture", "fertilizers"],
    frequency="Q",
)
```

### Available Commodity Groups

| Category | Benchmarks Covered | Units |
|---|---|---|
| **Energy** | Brent Crude (`oil_brent`), WTI Crude (`oil_wti`), European Natural Gas (`gas_eu`), US Henry Hub (`gas_us`), Coal Australia (`coal_au`) | $/bbl, $/mmbtu, $/mt |
| **Industrial Metals** | Copper (`copper`), Aluminum (`aluminum`), Iron Ore (`iron_ore`), Nickel (`nickel`), Zinc (`zinc`) | $/mt, $/dmt |
| **Precious Metals** | Gold (`gold`), Silver (`silver`), Platinum (`platinum`) | $/troy oz |
| **Agriculture & Food** | Wheat (`wheat`), Maize/Corn (`maize`), Soybeans (`soybeans`), Coffee (`coffee`), Cocoa (`cocoa`) | $/mt, $/kg |
| **Fertilizers** | Urea (`urea`), Phosphate Rock (`phosphate_rock`), DAP (`dap`) | $/mt |

---

## 4. International Financial Stability & Macroprudential Indicators (`puremacro.fetch.financial`)

Fetches cross-country financial stability, sovereign risk, and credit market indicators from the Bank for International Settlements (BIS), FRED, and international central banks.

```python
from puremacro.fetch.financial import (
    fetch_sovereign_yields,
    fetch_policy_rates,
    fetch_credit_gap_panel,
    fetch_property_prices,
    fetch_financial_conditions,
)

# Benchmark 10Y and 2Y sovereign yields + 10Y-2Y term spread
df_yields = fetch_sovereign_yields(
    codes=["USA", "DEU", "GBR", "JPN"],
    maturities=["10Y", "2Y", "spread_10y2y"],
    frequency="M",
)

# Central bank policy interest rates (FED, ECB, BOE, BOJ)
df_cb = fetch_policy_rates(
    codes=["USA", "EA20", "GBR", "JPN"],
    frequency="M",
)

# BIS Credit-to-GDP gap (Basel III countercyclical capital buffer indicator)
df_credit = fetch_credit_gap_panel(
    codes=["USA", "DEU", "GBR"],
    start_year=2000,
)

# BIS Real Residential Property Price Index
df_house = fetch_property_prices(
    codes=["USA", "DEU", "GBR"],
    start_year=2000,
)

# Financial stress and spread indicators
df_fci = fetch_financial_conditions(
    indicators=["NFCI", "TEDRATE", "BAMLH0A0HYM2"],
    start_date="2000-01-01",
)
```

---

## 5. Modular Panel Builders (`puremacro.build_climate_panel` & `build_financial_panel`)

Top-level panel constructors assemble balanced or unbalanced multi-country panels ready for estimation in VAR, local projections, or panel regressions.

### `build_climate_panel`

Combines emissions, energy balances, and economic indicators into a unified panel.

```python
from puremacro import build_climate_panel

panel = build_climate_panel(
    codes=["USA", "DEU", "JPN", "GBR", "MEX"],
    start_year=2000,
    frequency="A",  # "A" for Annual, "Q" for interpolated/aggregated Quarterly
    include_gdp=True,
    include_energy=True,
)

print(panel.head())
```

**Key Columns**:
- `code`, `date`: Country ISO-3 code and Datetime index.
- `co2_total`, `co2_per_capita`, `ghg_total`: Emissions metrics.
- `clean_elec_share`, `fossil_elec_share`: Energy transition metrics.
- `gdp`, `population`: Normalization denominators.
- `imputed_*`: Boolean indicator flags marking interpolated series.

### `build_financial_panel`

Combines sovereign yield curves, central bank policy rates, BIS credit gaps, and global commodity price benchmarks aligned to macroeconomic forecasting horizons.

```python
from puremacro import build_financial_panel

panel = build_financial_panel(
    codes=["USA", "DEU", "GBR", "JPN"],
    start_date="2000-01-01",
    frequency="Q",  # "M" for Monthly, "Q" for Quarterly aggregation
    include_commodities=True,
    include_credit=True,
)

print(panel.head())
```

**Key Columns**:
- `yield_10y`, `yield_2y`, `term_spread`: Sovereign debt yield curve.
- `policy_rate`: Central bank target policy rate.
- `credit_gap`: BIS credit-to-GDP gap.
- `oil_brent`, `copper`, `gold`: Benchmark commodity prices.

---

## 6. Architectural Guarantees & Offline Testing

1. **Pyodide & Zero-Network Safety**:
   Importing `puremacro`, `puremacro.fetch`, or any panel builder never triggers network calls or requires `requests` at module scope.
2. **Deterministic Offline Fixtures**:
   Every collector includes recorded mock fixtures in `tests/data/`, allowing unit tests and CI suites to execute 100% offline without live network dependencies.
3. **Unified Caching**:
   Live calls store timestamped cache entries under `data/raw/` with automatic TTL management and stale-data warnings.
