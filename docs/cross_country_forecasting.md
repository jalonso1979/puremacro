> English · [Español](es/cross_country_forecasting.md)

# From a frozen panel to a forecast comparison

**Available in puremacro 4.8.0.** Install the release to run this workflow.
It uses a frozen World Bank WDI snapshot for the United States,
Mexico and Brazil. The full application runs offline and exports its inputs,
forecasts, accuracy tables, figure, metadata and a portable `.pmz` cartridge.

```bash
python -m pip install "puremacro==4.8.0"
python -m puremacro.examples.cross_country_forecasting --output research_output/cross_country_forecasting
```

Read `report.md` and `accuracy.csv` in the output directory. `manifest.json`
records the study design, input and source-code hashes, runtime versions and
each artifact's checksum. The installed-wheel workflow needs only the base
package dependencies; fetching the World Bank API is not part of the run.

## The question and the fixed design

How do three simple methods compare when forecasting annual real GDP growth
one year ahead? The input has 65 annual GDP levels per country, 1960–2024,
from WDI indicator `NY.GDP.MKTP.KN`. Levels are in millions of constant local
currency at each country's base/reference year. Only within-country growth
is compared: `100 * (log(GDP_t) - log(GDP_(t-1)))`.

The target years are 2000–2024. At each origin, the expanding training window
starts in 1961 and ends in the preceding observation year. Methods are:

- **Zero growth:** predict unchanged GDP level.
- **Historical mean:** predict the mean of the growth observations in training.
- **AR(1):** regress growth on an intercept and its first lag using the public
  `fit_var` estimator, then forecast the next observation.

The countries, transformation, sample, lag order and methods were fixed before
comparing errors. There is no tuning, imputation or full-sample standardization.
All methods use the same target years, including 2020 and the recovery. Missing
years, nonpositive levels, insufficient training and unidentified AR regressions
raise errors instead of quietly changing the sample.

`forecasts.csv` contains the origin, target, training range, observation count,
actual, forecast and error for all 225 country/model/year combinations.
`accuracy.csv` reports RMSE, MAE and mean error by country and model. Errors are
actual minus forecast, in annual log-growth percentage points. The pooled table
weights countries equally: its RMSE is the square root of mean country MSE.
Coverage and exclusions are separate tables; 1960 has no growth observation,
and 1961–1999 are initial training years rather than forecast targets.

## What the result means

This is a **latest-vintage historical evaluation**. The snapshot was captured
on 11 October 2026 UTC (10 October in Mexico City); WDI's upstream
`lastupdated` field says 8 October 2026. Neither field reconstructs the data
available in 1999 or at another forecast origin. Publication lags are not
modeled, and the history includes subsequent revisions. The study therefore
does not establish real-time forecasting performance.

The losses describe three countries over 25 target years. They do not establish
statistical significance, general model superiority, causal policy effects or
calibrated prediction intervals. The two simple baselines are substantive
comparisons: AR(1) is not assumed to improve on them.

## Preserve and inspect the evidence

```python
from puremacro import pocket

cartridge = pocket.load("research_output/cross_country_forecasting/cross_country_forecasting.pmz")
cartridge.verify()
levels = cartridge["levels"]
print(levels.attrs["source"])
print(levels.attrs["panel_attrs"]["missing"])
print(cartridge["accuracy"])
```

The portable frame codec now preserves supported `DataFrame.attrs` recursively,
including mappings with tuple keys, lists/tuples, numeric scalars and date types.
No pickle is used. Unsupported objects and cycles raise `StoreError` with a
metadata path. The v2 frame schema is read alongside existing v1 frames; older
readers need puremacro 4.8.0 or later to read v2. Existing v1 files need no
conversion, and v1 cartridges remain verifiable,
but their original format did not retain frame attributes. New cartridge
checksums include metadata, so changing provenance after loading fails
verification. Checksums detect changes; they are not signatures of authenticity.

The CSV tables are deterministic. Cartridge creation timestamps vary between
runs, so byte equality of whole cartridges is not a reproducibility criterion.
Source and derived-data hashes are preserved separately from runtime artifacts.

## Rebuild the frozen data

The repository includes the exact country-catalog and GDP API responses in
`reviews/2026-10-10-cross-country-forecasting/source/`. From the repository root:

```bash
PYTHONPATH=. python tools/build_cross_country_forecasting_data.py --output-dir /tmp/puremacro-gdp-rebuild
```

The builder refuses unreviewed source hashes. It replays the recorded bytes at
the HTTP boundary through `wdi_panel`, then independently checks every scaled
observation against a direct JSON extraction. It neither contacts a provider
nor accepts a later revision implicitly. The frozen design and
validation records are in the same review directory. Independent external
user reproduction remains a separate adoption milestone.

Data attribution: World Bank, *World Development Indicators*, GDP (constant
LCU), [indicator NY.GDP.MKTP.KN](https://data.worldbank.org/indicator/NY.GDP.MKTP.KN),
[CC BY 4.0](https://datacatalog.worldbank.org/public-licenses#cc-by).
