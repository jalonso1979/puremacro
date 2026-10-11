# Cross-country GDP-growth forecasting

A frozen latest-vintage World Bank WDI snapshot for USA, Mexico and Brazil. This backtest does not reconstruct information available in real time.

The fixed target is annual GDP log growth, 100 × Δlog(real GDP). Expanding training windows begin in 1961; one-year-ahead forecasts target 2000–2024. At each origin only growth through the preceding year enters estimation. Zero growth means unchanged GDP level; the historical mean and AR(1) with intercept are fitted independently for each country. No lag, model or sample tuning is performed.

## Accuracy

| code |           model | n_forecasts |     rmse |      mae | mean_error |
|------|-----------------|-------------|----------|----------|------------|
|  BRA |             ar1 |          25 | 2.916791 | 2.266982 |  -0.781841 |
|  BRA | historical_mean |          25 | 3.294677 | 2.466129 |  -1.899963 |
|  BRA |     zero_growth |          25 | 3.580671 | 3.148364 |   2.316487 |
|  MEX |             ar1 |          25 | 3.900021 | 2.691987 |  -1.399941 |
|  MEX | historical_mean |          25 | 3.966212 |  2.68571 |  -2.325353 |
|  MEX |     zero_growth |          25 | 3.568995 | 2.914979 |   1.610335 |
|  USA |             ar1 |          25 | 2.000933 |  1.28839 |  -0.718212 |
|  USA | historical_mean |          25 | 1.993801 | 1.309852 |  -0.982361 |
|  USA |     zero_growth |          25 | 2.768986 | 2.557214 |   2.180125 |

Errors are actual minus forecast, in annual log-growth percentage points. Lower RMSE and MAE indicate smaller losses in this fixed sample. The equal-country summary takes the square root of mean country MSE for RMSE.

|           model | n_countries | n_forecasts |     rmse |      mae | mean_error |
|-----------------|-------------|-------------|----------|----------|------------|
|             ar1 |           3 |          75 | 3.039823 | 2.082453 |  -0.966665 |
| historical_mean |           3 |          75 | 3.191707 | 2.153897 |  -1.735893 |
|     zero_growth |           3 |          75 | 3.327973 | 2.873519 |   2.035649 |

## Coverage and interpretation

| code | first_level_year | last_level_year | level_observations | growth_observations | first_training_growth_year | initial_training_growth_observations | evaluation_start_year | evaluation_end_year | evaluation_observations_per_model | excluded_evaluation_observations |
|------|------------------|-----------------|--------------------|---------------------|----------------------------|--------------------------------------|-----------------------|---------------------|-----------------------------------|----------------------------------|
|  BRA |             1960 |            2024 |                 65 |                  64 |                       1961 |                                   39 |                  2000 |                2024 |                                25 |                                0 |
|  MEX |             1960 |            2024 |                 65 |                  64 |                       1961 |                                   39 |                  2000 |                2024 |                                25 |                                0 |
|  USA |             1960 |            2024 |                 65 |                  64 |                       1961 |                                   39 |                  2000 |                2024 |                                25 |                                0 |

All models use the same target years, including 2020 and the recovery. The exclusions file records years outside the fixed target window and the first level without a growth observation. Invalid levels, annual gaps and rank-deficient AR fits raise.

- Historical data are revised latest-vintage observations, not observations available at each origin.
- Calendar-year origins assume the preceding year's GDP is observed; publication lags are not modeled.
- Three selected countries and 25 target years do not establish general forecasting superiority.
- RMSE, MAE and mean error are descriptive; no significance test, predictive intervals or causal claims are made.
- The 2020 contraction and recovery remain in the predeclared evaluation sample.
- Constant-LCU GDP levels are not compared across countries; only within-country log growth is forecast.

## Provenance and replay

Snapshot fetched at: `2026-10-11T02:25:05+00:00`. Upstream last-updated field: `2026-10-08`. Neither identifies historical vintages available at forecast origins.

`input_metadata.json` retains source URLs, units, transformations and checksums. The portable `.pmz` retains the input metadata in the levels frame's attrs and every result table. `manifest.json` authenticates exported artifacts and numerical sources.
