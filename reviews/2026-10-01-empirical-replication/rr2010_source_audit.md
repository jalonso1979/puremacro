# Independent RR2010 source and inference audit

The Figure 4 point response and trough reproduce the paper. Its printed
horizon-ten t statistic does not quite meet the predeclared two-decimal
comparison. This discrepancy remains visible; it is not converted into a pass.

The original `EXOGERNR.RAT` no-controls block uses 1950Q1–2007Q4, a constant
and tax lags 0–12. Its `SUMMARIZE` command cumulates coefficients and their
joint uncertainty. Positive shocks increase taxes. The original workbook's
nonretroactive `DEFICNR` and `LONGRNR` columns match this specification. The
`GDP` quantity index and `NOMGDP` denominator come from the separate output
sheet. The [original author archive](https://eml.berkeley.edu/~dromer/papers/DataSet.zip)
has SHA256 `c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c`.

There are 232 observations, 14 coefficients and 218 residual degrees of
freedom. Conventional covariance uses SSR/218. This reproduces the original
OLS convention; it is not HAC or simultaneous inference. The published
Figure 4 uses one-standard-error bands over horizons 0–12. The implementation
and application preserve these distinctions. See the [paper, printed pages
780–782](https://eml.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf)
and [RATS regression-output documentation](https://estima.com/webhelp/topics/regressionoutput.html).

An independent audit reads the XLSX XML directly, avoiding the production
converter and openpyxl, then decodes the original RATS binary data records.
The [official Gretl reader](https://github.com/gretl-project/gretl/blob/master/lib/src/foreign_db.c)
documents the binary layout. Independent NumPy SVD and QR regressions use
both sources and the complete cumulative covariance.

| Original source | Horizon-ten response | Standard error | t statistic |
|---|---:|---:|---:|
| XLSX, SVD | -3.0808127630878372 | 0.8739904095035419 | -3.5249960750002396 |
| RATS databanks, SVD | -3.0808127630878400 | 0.8739904095035419 | -3.5249960750002427 |
| Printed paper | -3.08 | — | -3.53 |

The response minimum occurs at horizon 10. GDP values are identical across
sources. Maximum discrepancies in nominal GDP, deficit tax changes, and
long-run tax changes are respectively 1.819e-12, 3.553e-15 and 7.105e-15.
Source and solver differences move the t statistic by about 3e-15, whereas
the failure exceeds the rounding interval by 3.925e-6. They do not explain
the printed discrepancy. No live RATS execution was performed, so the
reason for the paper's printed value remains unresolved.

[Reproducible audit script](audit_rr2010_sources.py) and
[full numerical evidence](rr2010_source_audit.json) retain the source hashes,
source comparisons, all-array comparisons with the frozen statsmodels
reference, and every audited result. No puremacro estimator is imported by
the script. The dedicated test `test_real_published_t_discrepancy_cannot_be_hidden_by_software_parity`
requires the software comparison to pass while the unchanged published
t-statistic comparison and aggregate report fail.
