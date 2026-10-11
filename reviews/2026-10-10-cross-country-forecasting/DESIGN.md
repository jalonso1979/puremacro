# Frozen study design

Chosen before fitting or comparing forecast errors on 10 October 2026.

- Countries: United States, Mexico and Brazil (USA, MEX, BRA).
- Data: World Bank WDI GDP, constant local currency, indicator NY.GDP.MKTP.KN,
  annual levels 1960–2024. Preserve the full source responses and their hashes.
- Outcome: `100 * diff(log(GDP))`, in percent log growth; country price bases
  cancel in this within-country transformation. No level comparisons across countries.
- Forecast target years: 2000–2024; one-year horizon, expanding training windows
  ending at the preceding observation year; minimum 20 training growth observations.
- Methods: zero growth (unchanged GDP level), historical mean growth, AR(1)
  growth with intercept. No model selection or tuning after evaluation.
- Metrics: RMSE, MAE and mean error by country and method; equal-country pooled
  results separately. Report all methods, including failures and unfavorable results.
- No gap filling or silent exclusion: reject incomplete or nonpositive input.
  Record the first level lost to differencing and the initial training years.
- Latest-vintage evaluation only: lagging observations is not proof of their
  historical release availability, and revised data are not historical vintages.
- No causal interpretation, significance claim, or general forecasting superiority.

Independent verification will compare the scaled input with the raw JSON,
AR estimates with a separate least-squares implementation, and exported results
from an installed wheel with the source-tree study. This is technical independent
reproduction within development; independent external adoption remains to be obtained.
