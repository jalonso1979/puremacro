# Romer–Romer (2010): original baseline replication

Independent-software and published-target comparisons: **FAIL**.

- `rr2010_baseline_software`: **PASS**
- `rr2010_published_peak`: **FAIL**

For a tax increase of one percent of GDP, the horizon-ten GDP response is **-3.08081276 log percent**, conventional SE 0.87399041, t statistic **-3.52499608**. The trough occurs at horizon 10. The printed paper reports -3.08 and -3.53 at horizon ten.

The t-statistic rounding comparison has absolute error 0.0050039250 against tolerance 0.005. The bundled original-data result misses that interval by approximately 0.000003925. This discrepancy remains unresolved; it is not removed by rounding intermediate values, widening tolerance or omitting the selected statistic. The full software comparison and each published metric remain separately inspectable.

The original-vintage regression has 232 quarterly observations, 1950Q1–2007Q4, and 218 residual degrees of freedom. It regresses 100 times the change in log real GDP on an intercept and lags zero through twelve of 100*(DEFIC+LONGR)/NOMGDP. Cumulative tax coefficients give the level response; the full conventional residual-df coefficient covariance is cumulated with them. Positive tax shocks mean tax increases.

The software comparison checks all coefficients, both complete covariance matrices, responses, standard errors, t statistics and sample sizes against a frozen independent run. The separate published comparison allows only two-decimal rounding. The offline application does not execute the authors' RATS program. Source hashes, reference software and regeneration details are in manifest.json; every numerical comparison and tolerance is in benchmark_report.json.

Scope: the original Figure 4 baseline specification, not all paper robustness exercises or independent validation of narrative tax-shock exogeneity. The displayed band is plus/minus one conventional OLS standard error, not simultaneous coverage. The earlier modified local-projection gallery case uses a different sample and estimator and is not the reference for this replication.

Sources: [paper and citation](https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763), [author manuscript, printed p. 781](https://emlab.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf), [original data and RATS archive](https://eml.berkeley.edu/~dromer/papers/DataSet.zip).
