> 🇬🇧 English · 🇪🇸 [Español](es/cross_country_panels.md)

# Cross-country macro panels

Ten modules under `puremacro.fetch` turn a provider's whole cross-section into one DataFrame with one call. They share a contract:

- **Shape.** A wide frame indexed by `(code, date)`: ISO3 codes, `date` at the start of the period (1 January, the first day of the quarter or month). One column per variable, with the names used elsewhere in puremacro (`gdp`, `cons_hh`, `inv`, `urate`, `cpi`, ...).
- **Metadata in `attrs`.** `meta` (per column or per country and variable: flow, units, coverage), `source`, `fetched_at` (UTC) and `missing` (what was asked for and not obtained, with the reason). Each module has a `*_meta(panel)` helper that shows `meta` as a table.
- **Never raises on a provider failure.** An HTTP 429, a timeout, a truncated body or a dead API gives a warning and an empty or partial frame; the reason goes to `attrs["missing"]`.
- **urllib only.** No module imports `requests`, so they work where it is absent and import safely on an iPad or under Pyodide.
- **Cached on disk** under `~/.cache/puremacro` (override with `PUREMACRO_HTTP_CACHE_DIR`), never inside the repository. A re-run only re-sends what failed. Pass `refresh=True` to get the latest release.
- **OECD pacing.** Every OECD request goes through `puremacro.fetch._oecd_sdmx.oecd_csv`, which spaces calls (5 s by default), retries on HTTP 429 and caches the `csvfile` response.

Coverage counts below were measured live on 7 October 2026.

## World Bank WDI: `wdi_panel` (`puremacro.fetch.wdi`)

`wdi_panel(indicators=None, codes=None, *, start=1960, end=None, real=True, long=False, refresh=False, pause=1.0, timeout=120.0)` is the widest and longest free annual panel: the World Development Indicators for the 217 economies the World Bank lists (territories included), from 1960.

- **Units.** Currency levels in millions of current LCU, persons in thousands, rates and indices as published.
- **Real and deflators.** `real=True` adds `<name>_real` (constant LCU at each country's own base year, see `wdi_meta`) and `<name>_defl = 100 * nominal / real`, kept only where both levels are positive (masked cells are counted in `n_masked`). Sign-changing items (`inventories`, `stat_disc`, `taxes_prod`) get no deflator. The GDP deflator reproduces WDI's own to 1e-15. Chain-linked `_real` components do not add up.
- **Indicators.** `None` pulls `WDI_CORE` (national accounts, population, the ILO labour block, PPP, exchange rate, CPI, private credit: 35 names, 51 requests, about 2.5 minutes cold). `"all"` pulls the whole `WDI_INDICATORS` registry (52 names). A mapping `{"name": "WDI.CODE"}` adds any raw code, unscaled.
- **attrs.** `meta` has one dict per column (indicator, units, unit_mult, first, last, n, n_codes, `lastupdated`). Error bodies sent with HTTP 200 are re-fetched once, so a bad answer is not served from the 30-day cache.
- **Helpers.** `wdi_countries()` (region, income group) and `wdi_meta()` (currency, base year, SNA vintage, fiscal-year notes).
- **Not in WDI.** Durables, public GFCF (only `inv_priv`), capital stocks, hours, an employment level (use `lf * (1 - urate/100)`), economy-wide compensation, anything sub-annual.

```python
from puremacro.fetch.wdi import wdi_panel
p = wdi_panel(["gdp", "pop"], ["MEX", "USA"], start=1960)
```

Source: World Bank WDI, API v2 source 2 (CC BY 4.0).

## OECD annual national accounts: `ana_panel` (`puremacro.fetch.oecd_ana_panel`)

`ana_panel(codes=None, *, start="1950", assets=False, durability=False, income=False, output=False, labor=False, sectors=False, stocks=False, real=True, long=False, refresh=False)` is the annual twin of `qna_panel`.

```python
from puremacro.fetch import ana_panel, ana_countries, ana_panel_meta
p = ana_panel(["USA", "FRA", "MEX"], start="1950", income=True, labor=True)
p.loc["FRA", ["gdp", "gdp_defl", "gdp_real", "mixed_income", "hours"]]
```

- **Money** in millions of national currency at current prices.
- **`<x>_defl`** is `100 * V / LR` (2020 = 100), exactly the OECD's published deflator. **`<x>_real`** is the chain-linked volume at 2020 prices: `LR` where published, otherwise `L` rebased by `V/L` in 2020. Not additive outside 2020.
- **One table per series.** Each (country, variable, price base) is read whole from T0102, else T0101, else T0103; `meta["tables_not_t0102"]` lists the exceptions (today only Russia's GDP volume).

With `codes=None` the panel has 64 countries: 6 start before 1960, 12 before 1970, 34 before 1980 and 46 before 1995.

| Switch | Columns | Source |
|---|---|---|
| (always) | gdp, cons_hh, cons_gov, inv, capform, exports, imports, discrepancy_exp | DSD_NAMAIN10 DF_TABLE1 (T0102) |
| `income` | comp_emp, surplus_mixed, taxes_prod_imp, subsidies, taxes_prod_imp_net, discrepancy_inc, gdp_income; surplus_gross, mixed_income, cfc, vat | DF_TABLE1 (T0103); DSD_NASEC10 DF_TABLE14 |
| `output` | gdp_output, va_total, va_agri, va_ind, va_mfg (memo), va_constr, va_trade, va_ict, va_fin, va_realest, va_prof, va_public, va_other, va_services, taxes_prod, discrepancy_out | DF_TABLE1 (T0101) |
| `durability` | cons_dur, cons_semidur, cons_nondur, cons_serv (households, S14) | DF_TABLE5A_T117, else DF_TABLE5_T117 |
| `assets` | inv_dwell, inv_struct, inv_equip, inv_transp, inv_ict, inv_othmach, inv_bio, inv_ipp | DF_TABLE1 (T0102) |
| `sectors` | inv_gov, inv_corp, inv_fin, inv_hh (S14, else S1M) | DSD_NASEC10 DF_TABLE14_GFCF |
| `stocks` | k_net, k_gross (with volumes), k_net_gov | DF_TABLE9A; DSD_NASEC10 DF_TABLE9B |
| `labor` | emp, emp_employees, emp_selfemp, hours, hours_employees, hours_selfemp (each also `_agri`, `_public`); pop | DF_TABLE3_EMPDC; DF_TABLE3_POP_EMPNC |

Income, sectors, `k_net_gov`, labour and population are current-price or count blocks and get no deflator.

`attrs`: `meta` (one dict per country, read with `ana_panel_meta`: currency, years, volume base, durability table, `inv_hh` sector, hours scale, absent columns), `variables` (one dict per column: flow, units, description, coverage, SDMX keys), `missing` (failures a re-run can fix: 429s, timeouts, non-SDMX bodies, countries without GDP), `chunks_empty` (404s or empty answers a re-run cannot change) and `requests` (every request with its status and row count). Requests go in chunks of at most 10 countries; main, assets, income and output share one DF_TABLE1 request.

Source: OECD SDMX, agency OECD.SDD.NAD.

## OECD quarterly blocks beside `qna_panel` (`puremacro.fetch.oecd_qna_extras`)

Four functions, each one call, each joining with `qna_panel` on the index.

| function | columns | units | coverage |
|---|---|---|---|
| `qna_sector_gfcf(codes=None, *, start="1947", sa="x13")` | `inv_gov`, `inv_priv`, `inv_gov_real`, `inv_priv_real` | millions of national currency | 34 countries, USA from 1947Q1; volumes for 11 |
| `qna_population(codes=None, *, start="1947")` | `pop`, `emp_nc`, `emp_employees_nc`, `emp_selfemp_nc` | thousands of persons | pop 43 (USA from 1947), employment 40 (from 1980) |
| `oecd_lfs_panel(codes=None, *, freq="Q", start="1950")` | `urate`, `urate_1564`, `emp_lfs`, `lf`, `wap`, `wap_1564`, `prate_1564`, `epop_1564` | thousands / percent | 45 countries from 1955; `freq="M"` for monthly |
| `oecd_vacancies(codes=None, *, freq="Q", start="1950")` | `vacancies`, `vacancies_new`, `reg_unemp` | thousands of persons | 18 with vacancies (DEU from 1955), 29 with registered unemployment |

- **Sources.** Government investment comes from `DF_QNA_EXPENDITURE_GFCF_SECTOR` (S13, S1W) where published, otherwise from the quarterly sector accounts `DF_QSA` (private = S1 less S13); `attrs["meta"]["source"]` says which. `urate` is the harmonised rate (`UNE_LF_M`), else the survey rate (`UNE_LF`) for BRA, RUS, ZAF and a few others. Vacancies are levels only (`TRANSFORMATION=_Z`).
- **Seasonal adjustment.** The OECD's adjusted series where published. `qna_sector_gfcf(sa="x13")` adjusts the rest with `qna_panel`'s X-13 path (`sa="prefer"` leaves them raw); the other blocks fall back to unadjusted. `attrs["meta"]["sa"]` records `oecd`, `puremacro`, `none` or `mixed` per country and variable.
- **Failures.** `attrs["missing"]` lists every (country, variable) pair not obtained, with `"not published"` or e.g. `"HTTP 429 on <flow>"`; `attrs["requests"]` logs each request, including the availability call. `qna_extras_meta(panel)` tabulates `meta`. Every call accepts `refresh=True`.

```python
from puremacro.fetch import qna_panel, qna_countries
from puremacro.fetch.oecd_qna_extras import (qna_sector_gfcf, qna_population,
                                             oecd_lfs_panel, oecd_vacancies)
codes = qna_countries()
quarterly = qna_panel(codes).join([qna_sector_gfcf(codes), qna_population(codes),
                                   oecd_lfs_panel(codes), oecd_vacancies(codes)], how="outer")
```

## OECD monthly short-term indicators: `stes_panel` (`puremacro.fetch.oecd_stes_panel`)

`stes_panel(codes=None, *, concepts=None, start="1950", refresh=False, pause=5.0, retries=3, retry_sleep=90.0)` returns 31 monthly concepts in levels, as the OECD publishes them: indices (2015 = 100; surveys and the CLI long-run average = 100), percentages, LCU per USD, persons in thousands. Logs, growth rates and seasonal adjustment of the NSA series (CPI, money, `urate_nsa`) are left to the user. The default `start="1950"` cuts US industrial production (from 1919) and Canadian CPI (from 1914); pass `start="1900"` for the full history.

```python
from puremacro.fetch import stes_panel, stes_meta
m = stes_panel(start="1900", pause=8.0)            # every economy, all 31 concepts, full history
m = stes_panel(["USA", "MEX"], concepts=["ip", "cpi", "rate_3m", "urate"])
stes_meta(m)[["concept", "unit", "n_countries", "first", "last", "stale_countries"]]
```

| group | concepts | flow | coverage |
|---|---|---|---|
| production | ip, ip_mfg, ip_constr, retail_vol (SA) | STES `DF_INDSERV` | ip 41 economies, USA from 1919 |
| markets | rate_on, rate_3m, rate_10y (% p.a.), share_price, reer_cpi | STES `DF_FINMARK` | 43-47 |
| exchange rate | fx_usd (LCU/USD, euro members spliced to EUR) | STES `DF_KEI` | 46, no USA |
| surveys | bci, cci, cli (amplitude adjusted) | STES `DF_CLI` | 46 / 41 / 17 |
| money | m1, m3 (NSA index) | STES `DF_MONAGG` | 29; no individual euro member (ask for `"EA20"`) |
| labour | urate, urate_nsa; emp, lf, prate, epop | TPS `DF_IALFS_UNE_M`, `DF_IALFS_INDIC` | 39; about 10 for monthly LFS levels |
| registers | vacancies, reg_unemp (thousands, SA) | TPS `DF_OIALAB_INDIC` | 17 / 27 |
| prices | cpi, cpi_food, cpi_energy, cpi_core; hicp, hicp_food, hicp_energy, hicp_core (NSA) | TPS COICOP 2018, then COICOP 1999 | 46/45/39/40; 30 each |

- **Consumer prices.** EU/EEA countries moved to COICOP 2018 in January 2026 and their COICOP-1999 series stop at 2025-12. Each price column takes COICOP 2018 first and fills from COICOP 1999 month by month, ratio-splicing earlier months onto the 2018 level. Gaps are large only where an aggregate was redrawn (BGR energy 13%, NLD core 5%, BGR core 4%). HICP takes the 1999 flow only for EU, CHE, GBR, ISL, NOR and TUR. `stes_meta(m)` lists `countries_fallback`, `splice_factors` and `stale_countries` (series ending more than six months early: national CPI for AUT, POL, SVK at 2025-12 and RUS at 2022-03, ZAF core at 2024-12, ISL HICP).
- **Failures.** Failed concepts go to `attrs["missing"]`; a requested country without data appears as `"concept:CODE"`. If a price column's COICOP-2018 request fails the column is dropped, not built from 1999 alone; if only the fallback fails it is kept and flagged (`"cpi@DF_PRICES_ALL"`). While the OECD throttles, a full call can block 30-45 minutes; `retries=1` fails fast.

Source: OECD SDMX API (`sdmx.oecd.org`).

## Eurostat: `eurostat_na_panel` and `eurostat_monthly_panel` (`puremacro.fetch.eurostat`)

The longest European national accounts (France and Norway from 1975, French sector accounts from 1949), the Balkan and Eastern-partnership reporters, and monthly indicators the OECD flows lack.

- `eurostat_na_panel(codes=None, *, freq="A", start=None, durability=False, assets=False, output=False, income=False, labor=False, sectors=False, capital=False, real=True, sa="prefer", refresh=False, timeout=120.0)`
- `eurostat_monthly_panel(codes=None, *, start=None, variables=None, refresh=False, timeout=120.0)`
- Building blocks: `eurostat_get(flow, key, ...)`, `eurostat_codes(flow)`, `eurostat_meta(panel)`.

```python
from puremacro.fetch.eurostat import (eurostat_na_panel, eurostat_monthly_panel,
                                      eurostat_meta, eurostat_get, eurostat_codes)

a = eurostat_na_panel()                     # 41 countries, 1975-2025, gdp ... imports + _real + _defl
q = eurostat_na_panel(freq="Q")             # 39 countries, SCA > SA > NSA per country
full = eurostat_na_panel(["DEU", "FRA"], durability=True, assets=True, output=True,
                         income=True, labor=True, sectors=True, capital=True)
m = eurostat_monthly_panel(variables=["cpi", "cpi_core", "urate"])
eurostat_meta(full)                         # one row per country and variable
```

| flag | columns | flow (annual / quarterly) |
|---|---|---|
| (always) | gdp cons_hh cons_gov inv capform exports imports | nama_10_gdp / namq_10_gdp |
| income | comp_emp surplus_mixed taxes_prod_imp subsidies taxes_products subsidies_prod taxes_prod; annual: mixed_income surplus_gross cfc | nama_10_gdp + nasa_10_nf_tr / namq_10_gdp |
| durability | cons_dur cons_semidur cons_nondur cons_serv | nama_10_fcs / namq_10_fcs |
| assets | inv_dwell inv_struct inv_constr inv_equip inv_ipp | nama_10_an6 / namq_10_an6 |
| output | va_total va_services va_agri va_ind va_mfg va_constr va_trade va_info va_fin va_realestate va_business va_public va_other | nama_10_a10 / namq_10_a10 |
| labor | emp emp_employees emp_selfemp (thousands) hours (millions) pop | nama_10_a10_e + nama_10_pe / namq_ twins |
| sectors | inv_corp inv_fin inv_hh inv_gov | nasa_10_nf_tr + gov_10a_main / nasq_10_nf_tr + gov_10q_ggnfa |
| capital (annual) | k_net k_dwell k_struct k_equip k_ipp k_gross | nama_10_nfa_st |

- **Units.** Millions of national currency; for euro members that is euro over the whole history, so there is no break at adoption. `_real` is chain-linked volumes with reference year 2020 (2015 where 2020 is not published, e.g. the United Kingdom); `_defl` is 100 in the reference year.
- **Columns always present.** The frame carries every requested column, all-NaN when nothing came back, so `df["gdp"]` never raises.
- **Failures** (timeout, 5xx, dropped connection, truncated download, expired async job) give an empty frame, a warning and the reason in `missing`. Each request leaves the country dimension open, so any subset of countries reuses the same cached download.

## IMF: `puremacro.fetch.imf`

The IMF serves its data at `api.imf.org/external/sdmx/2.1`. The old IFS host (`dataservices.imf.org`) is gone and `sdmxcentral.imf.org` answers data queries with 501. Every function below also takes `refresh=False`.

| function | returns | coverage |
|---|---|---|
| `imf_nea_panel(codes=None, *, freq="A"\|"Q", sa="SA", real=True, start=None)` | `gdp, cons, cons_gov, cons_hh, cons_priv, inv, capform, inventories, exports, imports` + `_real`, `_defl` | A: 188 economies, 1950-2026; Q SA: 66 |
| `imf_monthly_panel(codes=None, *, variables=None, start=None)` | `cpi, cpi_food, cpi_housing, cpi_transport, ip, ip_sa, ip_mfg, urate, emp, lf, policy_rate, mm_rate, tbill_rate, bond_yield, discount_rate, deposit_rate, lending_rate, neer, reer, fx_usd, fx_usd_eop` | CPI: 191 economies from 1914; fx: 222 from 1940 |
| `imf_labour_panel(codes=None, *, freq="A"\|"Q"\|"M", start=None)` | `urate, emp, lf, unemp` | A: emp 190, lf 189 |
| `imf_pcps(*, freq="M", start=None)` | `pcom_all, pcom_energy, poil, pcopper, ...` + `<code>_idx` / `<code>_usd` | 188 series from 1992, code `WLD` |
| `imf_weo(codes=None, *, indicators=None, start=None, vintage=None, history_only=False)` | 17 named WEO series + `forecast` flag | 197 economies, 1980-2031 |
| `imf_icsd(codes=None, *, start=None)` | `inv_gov, inv_priv, k_gov, k_priv, k_pubpriv, gdp` (current prices) + `_ppp`, `_gdp` variants | 194 economies, 1960-2019 |
| `imf_get(flow, key="all", ...)`, `imf_dataflows()`, `imf_meta(panel)` | raw SDMX-CSV; flow catalogue; metadata table | |

- **Units.** National-currency levels in millions (ICSD's billions are multiplied by 1000), persons in thousands, rates, indices and prices as published. Each `attrs["meta"]` row gives `unit_mult` and `scale` (the factor applied).
- **Placeholder zeros.** Some flows publish exact zeros where there are no data (Burkina Faso's expenditure side for 1952-1998, capital stocks of economies ICSD does not cover). They are set to missing and counted in `attrs["zeros_dropped"]`. Zeros that can be true (inventory changes, the PPP capital stock, unemployment, interest rates) are kept.
- **Prices and volumes.** `_real` is at constant prices of each country's own reference year, mostly chain-linked, so components do not add up. `_defl = 100 * nominal / real` has that base, which is not always in the sample (BRA, ECU and AUS never come near 100): rebase before comparing levels across countries. `cons_hh` is households only (67 economies); for households plus NPISH use `cons_priv = cons - cons_gov`.
- **Codes.** IMF group codes (`G001`, `G163`, ...) are dropped via `IMF_AGGREGATES`; `KOS` becomes `XKX`, `WBG` becomes `PSE`; historical economies (`SUN`, `YUG`, ...) are kept.
- **WEO.** History ends separately for each country and indicator (`LATEST_ACTUAL_ANNUAL_DATA`); Argentina's population is projected after 2010 while its GDP is actual to 2025. `forecast` follows real GDP. `history_only=True` cuts every column at its own boundary, which is what you want to freeze history. Fiscal years follow the IMF's mapping (India FY2024/25 is 2024, Pakistan FY2024/25 is 2025).
- **Time budget.** Each request is retried 3 times (up to 180 s each). `imf_monthly_panel` stops after two flows fail in a row, so a dead API costs about 25 minutes at worst.
- **Not in the IMF flows.** The income side, durables, GFCF by asset, hours, core or energy CPI, money-market tenors: use the OECD, Eurostat or ILO fetchers.

```python
from puremacro.fetch.imf import imf_nea_panel, imf_monthly_panel, imf_weo
a = imf_nea_panel()                                   # 188 economies, annual
q = imf_nea_panel(["USA", "MEX", "KOR"], freq="Q")
m = imf_monthly_panel(start=1990)
hist = imf_weo(history_only=True)                     # WEO history only, per-column boundaries
```

## BIS: `bis_panel` (`puremacro.fetch.bis`)

`bis_panel(codes=None, *, variables=None, freq="M", start=None, aggregates=False, refresh=False)`. Each variable is one gzip request covering every economy. Also: `bis_eer(kind="real", basket="broad", *, freq="M", ...)`, `bis_get(flow, key, ...)`, `bis_countries(flow)`, `bis_meta(panel)`.

```python
from puremacro.fetch.bis import bis_panel, bis_eer, bis_countries, bis_meta
m = bis_panel()                                    # monthly, every economy, full history
q = bis_panel(freq="Q", start="1960")
a = bis_panel(["GBR", "USA"], variables=["cpi_a", "policy_rate"], freq="A")
```

| variable | BIS flow / key | native freq | units | economies | from |
|---|---|---|---|---|---|
| neer, reer | WS_EER `M.N.B` / `M.R.B` | M | index 2020=100 | 63 | 1994-01 |
| neer_narrow, reer_narrow | WS_EER `M.N.N` / `M.R.N` | M | index 2020=100 | 25 / 26 | 1964-01 |
| policy_rate | WS_CBPOL `M` | M | % p.a., end of period | 48 | 1945-01 |
| cpi | WS_LONG_CPI `M.*.628` | M | index 2010=100 | 62 | 1913-01 |
| xr_usd, xr_usd_eop | WS_XRU `M.*..A` / `..E` | M | national currency per USD | 189 | 1791 / 1900 |
| credit_gdp, credit_bank_gdp, credit_hh_gdp, credit_nfc_gdp, credit_gov_gdp | WS_TC `Q.*.{P,P,H,N,G}.{A,B,A,A,A}.{M,M,M,M,N}.770.A` | Q | % of GDP | 43 (gov 42) | 1947Q4 |
| credit, credit_usd | WS_TC `Q.*.P.A.M.XDC.A` / `USD` | Q | millions of national currency / USD | 43 | 1940Q2 |
| credit_gap, credit_trend | WS_CREDIT_GAP `Q.*.P.A.C` / `.B` | Q | pp of GDP / % of GDP | 43 | 1957Q4 |
| house_price, house_price_real | WS_SPP `Q.*.N.628` / `R.628` | Q | index 2010=100 | about 57 | 1927Q1 |
| dsr, dsr_hh, dsr_nfc | WS_DSR `Q.*.P` / `H` / `N` | Q | % of income | about 31 / 17 | 1999Q1 |
| cpi_a | WS_LONG_CPI `A.*.628` | A | index 2010=100 | 62 | 1661 |

- **Frequency.** The BIS publishes only the native frequencies above; a coarser `freq` is built here. Policy rate, end-of-period exchange rate and credit stocks and ratios take the last sub-period; indices, average exchange rate, house prices and DSR take the mean of a complete period. An incomplete current quarter or year is left out, not averaged over the months available.
- **Aggregates.** `EA` and the other BIS aggregates (EME, ADV, ALL, G20, WLD, WAEMU) are dropped by default so cross-country means never count euro members twice; one named in `codes` is kept, and `aggregates=True` keeps all. Economy counts above exclude `EA`.
- **Euro-area policy rate.** National series end in 1998-12 (Greece 2000-12) and `EA` starts in 1999-01; splice them yourself, e.g. `pd.concat([p.loc["DEU"][:"1998-12"], p.loc["EA"]["1999-01":]])`.
- **Dates.** The date level is `datetime64[ns]`, except in panels holding `cpi_a` before 1678 (the UK from 1661), which are `datetime64[us]`.
- **Metadata.** `attrs["meta"]` gives flow, key, units, unit_mult, sa, first, last and n per code and variable. `attrs["missing"]` also lists requested codes a flow does not cover (Mexico has no narrow nominal NEER).

Source: BIS statistics SDMX API v2, https://stats.bis.org/api/v2.

## Rates, yields, stocks and commodities (`puremacro.fetch.rates`)

- `rates_panel(codes=None, *, start="1950", variables=("rate_on", "rate_3m", "yield_10y", "yield_corp_aaa", "yield_corp_baa"), refresh=False)`
- `stock_index_monthly(codes=None, *, start=None, refresh=False)`
- `commodity_prices_monthly(*, start=None, refresh=False)`
- Building blocks: `fetch_fred_many(ids)` (one request per id) and `ecb_get(flow, key, start=, end=, last_n=)` (one ECB Data Portal query).

```python
from puremacro.fetch.rates import rates_panel, stock_index_monthly, commodity_prices_monthly

r   = rates_panel(start="1900")                      # 47 countries, (code, date) monthly
r10 = rates_panel(None, variables=("yield_10y",))    # 42 countries
px  = stock_index_monthly()                          # 34 countries, stock_idx + stock_idx_avg
cm  = commodity_prices_monthly()                     # WLD, 48 commodity_* columns, 1960-01..
```

`rates_panel` columns are percent per annum, monthly averages, NSA. The names match `stes_panel` and `eurostat` (`rate_on`, `rate_3m`), so the panels splice by column.

| column | source (in order) | coverage |
|---|---|---|
| `rate_on` | FRED `IRSTCI01{CC}M156N`; euro members from entry: EONIA, then €STR from 2019-10 | 46 countries; FRA, SWE from 1955 |
| `rate_3m` | FRED `IR3TIB01{CC}M156N`; euro members from entry: 3-month EURIBOR; CZE, HUN, POL, ROU completed from ECB IRS | 45; CAN from 1956 |
| `yield_10y` | FRED `IRLTLT01{CC}M156N`; ECB IRS 10-year (28 EU countries) fills gaps; ECB FM back-histories for USA (1900) and JPN (1972) | 42 |
| `yield_corp_aaa`, `yield_corp_baa` | FRED `AAA`, `BAA` (Moody's) | USA from 1919 |
| `yield_corp_de` (opt-in) | Bundesbank BBSIS X2000 | DEU from 1957 |

- **Splices are visible.** Every source leg used appears in `attrs["meta"]`. `attrs["complete"]` is `False` when a series some provider serves failed on this call (429, timeout); the ECB leg may then have stood in with a shorter history, so check `complete` before freezing a CSV.
- **Stocks.** `stock_idx` is the Yahoo month-end close in local currency (from 1985 at the earliest); `stock_idx_avg` is the ECB monthly average (USA from 1964, JPN from 1972, `"EMU"` on request from 1986). Neither is STES `share_price`, a 2015 = 100 index of averages.
- **Commodities.** The World Bank Pink Sheet, read with the standard library only; columns are prefixed `commodity_`, indices are `commodity_index_*` (2010 = 100).
- **Cache.** 30-day TTL; a cached body that no longer parses is re-downloaded once automatically.

## ILOSTAT: `ilostat_panel` (`puremacro.fetch.ilostat`)

`ilostat_panel(codes=None, *, freq="A", variables=None, start=None, end=None, modelled=False, refresh=False, timeout=300.0, pause=3.0)` returns ILOSTAT's harmonised labour-force-survey series for every reporting economy.

| column | ILOSTAT flow / MEASURE | units | freq |
|---|---|---|---|
| emp, une, lf, wap | DF_EMP_TEMP / DF_UNE_TUNE / DF_EAP_TEAP / DF_POP_XWAP `_SEX_AGE_NB` | thousands of persons | A Q M |
| urate, prate, epop | DF_UNE_DEAP / DF_EAP_DWAP / DF_EMP_DWAP `_SEX_AGE_RT` | percent | A Q M |
| emp_ees, emp_self | DF_EMP_TEMP_SEX_STE_NB, STE_AGGREGATE_EES / _SLF | thousands of persons | A Q M |
| hours | DF_HOW_TEMP_SEX_NB | hours per week per employed person | A Q M |
| informal_rate | DF_EMP_NIFL_SEX_RT | percent of employment | A Q (M sparse) |
| labour_share | DF_LAP_2GDP_NOC_RT (ILO modelled) | percent of GDP | A |
| *_model (`modelled=True`) | DF_EMP_2EMP, DF_UNE_2EAP, DF_EAP_2EAP, DF_POP_2WAP, DF_EMP_2EMP_SEX_STE, DF_HOW_2EMP | as above | A |

- **Definitions.** Every series is the 15+, both-sexes total, NSA except the EU monthly rates ILOSTAT relays from Eurostat SA (`meta[i]["sa"]`). Breaks (`OBS_STATUS == "B"`) stay in the data and are listed in `meta["breaks"]`.
- **Modelled projections are cut.** The modelled flows publish unflagged projected years (two for emp/urate/lf/hours_model, five for wap_model, one for labour_share). Unless `end` is given they are cut back from each flow's last year, so a frozen panel does not depend on the build date; the cut is in `meta["trimmed_after"]`. `modelled=True` is annual only and raises `ValueError` at `freq="Q"`/`"M"` (the one deliberate raise).
- **Coverage.** Annual emp: 226 economies (17 from before 1970, 165 from before 1995); urate 224; employees/self-employed 214; labour_share 189 (2004-2025). Quarterly urate: 122 economies from 1948Q1. USA annual from 1947; MEX annual from 1988 (lf/wap from 1960), quarterly from 1995Q2; ESP annual from 1969, quarterly from 1986Q2.

```python
from puremacro.fetch.ilostat import ilostat_panel, ilostat_meta
a = ilostat_panel()                                   # 12 survey variables, ~12 requests
q = ilostat_panel(freq="Q")
m = ilostat_panel(freq="M", variables=("emp", "une", "urate", "prate"))
x = ilostat_panel(["MEX", "USA", "ESP"], freq="A", modelled=True)
self_share = a["emp_self"] / (a["emp_ees"] + a["emp_self"])
ilostat_meta(a).loc["urate"]["breaks"]
```

## Penn World Table 11.0 and Maddison 2023 (`puremacro.fetch.pwt`)

`fetch_pwt(table="main", *, version="11.0", codes=None, start=None, refresh=False, timeout=180.0)`, `fetch_maddison(codes=None, *, start=1950, refresh=False, timeout=180.0)` and `pwt_variables(table="main")` download the Stata files from DataverseNL. Each datafile id is an immutable release, so files are cached for ten years.

```python
from puremacro.fetch.pwt import fetch_pwt, fetch_maddison, pwt_variables

main = fetch_pwt("main")      # 185 economies, 1950-2023: rgdpe, rgdpna, pop, emp, hc, cn, rnna, labsh, ctfp, csh_*, pl_*
na   = fetch_pwt("na")        # 212 economies with v_gdp: v_c v_i v_g v_x v_m v_gdp v_gfcf (millions of current national currency), q_* (constant 2021 prices)
cap  = fetch_pwt("capital")   # Ic_/Ip_/Nc_/Np_/Dc_/Kc_/Kp_/Ksh_ by asset (Struc, Mach, TraEq, Other), 180 economies
lab  = fetch_pwt("labor")     # comp_sh, lab_sh1..4 (Gollin adjustments), labsh, yr_sch, i_* source flags
trd  = fetch_pwt("trade")     # pl_x1..6, pl_m1..6, csh_x1..6, csh_m1..6 (BEC categories)
mpd  = fetch_maddison()       # 169 economies, gdppc (2011 int. $), pop (thousands), annual from 1950; start=None goes back to 1 AD
pwt_variables("na")           # published label | true units | note on stale labels
```

| table | file (datafile id) | economies | years |
|---|---|---|---|
| main | pwt110.dta (554030) | 185 | 1950-2023 |
| na | pwt110_na_data.dta (554024) | 212 with v_gdp (216 codes) | 1950-2023 (avh to 2025) |
| capital | pwt110_capital_detail.dta (554026) | 180 | 1950-2023 |
| labor | pwt110_labor_detail.dta (554028) | 211 codes | 1950-2023 (avh to 2025) |
| trade | pwt110_trade_detail.dta (554023) | 185 | 1950-2023 |
| Maddison | maddison2023_web.dta (421303) | 169 | 1-2022 (annual from 1950) |

- **Published column names**, float64 values; the `i_*` flags are integer codes explained in `attrs["value_labels"]`.
- **The price base is 2021 everywhere**, although some labels still say "2017 prices", "2017=1" or "USA GDPo in 2011=1"; `attrs["units"]` states the true base. Maddison `gdppc` is in 2011 international dollars.
- **C + I + G + X - M is not GDP** in about a third of the national-accounts rows: use `v_gdp` as the denominator and never compute a component as a residual. `v_gfcf` is effectively a series from 1970.
- **Capital detail adds up to GFCF** (`Ic_*` sum to `v_gfcf`) for 179 of 180 economies; for Taiwan the sum is 6-8% higher.
- **Employment is in millions in every table** (the labor file's persons are rescaled).
- **Codes.** `CH2` (PWT's alternative China) is dropped unless asked for. Former economies without GDP (`ANT`, `CSK`, `SUN`, `YUG` in `na`; `ANT` in `labor`) stay and are listed in `attrs["historical_entities"]`, so count economies with `v_gdp`, not with codes. Kosovo is `RKS` in PWT (`XKX` at the World Bank and IMF), flagged in `attrs["nonstandard_codes"]`: rename before merging.
- **Licence.** Both are CC BY 4.0, so frozen CSVs may be shared; cite `attrs["citation"]`, and for Maddison follow `attrs["citation_policy"]` (cite the original sources when graphing or using fewer than 12 countries).

Sources: Feenstra, Inklaar & Timmer (2015), PWT 11.0, doi:10.34894/FABVLR; Bolt & van Zanden (2024), MPD 2023, doi:10.34894/INZBF2.

## Which call for which concept

| frequency | concept | call |
|---|---|---|
| annual | GDP and expenditure components | `wdi_panel`, `ana_panel`, `imf_nea_panel`, `eurostat_na_panel`, `fetch_pwt("na")` |
| annual | long-run output, productivity, human capital | `fetch_pwt("main")`, `fetch_maddison` |
| annual | public vs private investment, capital stocks | `ana_panel(sectors=True, stocks=True)`, `imf_icsd`, `eurostat_na_panel(sectors=True, capital=True)`, `fetch_pwt("capital")` |
| annual | employment, hours, unemployment, labour share | `ana_panel(labor=True)`, `ilostat_panel`, `imf_labour_panel`, `fetch_pwt("labor")` |
| annual | long CPI, credit, house prices | `bis_panel(freq="A")` |
| annual | forecasts and frozen WEO history | `imf_weo` |
| quarterly | GDP and expenditure components | `qna_panel`, `imf_nea_panel(freq="Q")`, `eurostat_na_panel(freq="Q")` |
| quarterly | government vs private investment | `qna_sector_gfcf`, `eurostat_na_panel(freq="Q", sectors=True)` |
| quarterly | population, employment, unemployment, vacancies | `qna_population`, `oecd_lfs_panel`, `oecd_vacancies`, `ilostat_panel(freq="Q")` |
| quarterly | credit, credit gap, house prices, debt service | `bis_panel(freq="Q")` |
| monthly | production, prices, confidence, money, labour | `stes_panel`, `imf_monthly_panel`, `eurostat_monthly_panel`, `ilostat_panel(freq="M")` |
| monthly | policy and short rates, long and corporate yields | `rates_panel`, `bis_panel`, `stes_panel` |
| monthly | exchange rates (bilateral, effective) | `bis_panel`, `bis_eer`, `imf_monthly_panel` |
| monthly | stock and commodity prices | `stock_index_monthly`, `commodity_prices_monthly`, `imf_pcps` |
