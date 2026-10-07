# SW07 data and model mapping audit — 2026-10-01

The declared calibration's large empirical mismatch is dominated by its risk-premium shock scale. Correcting the expectation of sample-demeaned covariance moments for the finite sample does not remove that mismatch. The published-mode sensitivity has smaller discrepancies, but its inflation and interest-rate variances move further below the observed moments after this correction. These calculations are descriptive expectations, not Monte Carlo rejection probabilities or a refitted model.

## Data units and mapping

The inspected builder applies quarterly `100*dlog` transformations to real per-capita GDP, GDP-deflated consumption and fixed investment, and real hourly compensation. Hours use average nonfarm-business hours times employment, divided by population, then `100*log` and sample demeaning. Inflation is `100*dlog(GDPDEF)`; the federal funds annual percentage rate is divided by four. These definitions agree with the paper's data appendix and equation (15). The observation matrix correctly differences GDP level deviations while leaving quarterly inflation and the nominal interest rate in levels. No factor-of-100 or factor-of-four error was found. [SW07, data appendix p.47 and measurement equation p.17](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf).

The frozen CSV contains 156 observations, 1966Q1–2004Q4. It is a revised FRED snapshot built 2026-09-30, not the original estimation vintage. Log growth is invariant to dollar or index rebasing; revisions to historical growth can still matter. The raw nine-series downloads were not found locally, so this audit verifies the builder and packaged units rather than independently reconstructing the CSV from raw source observations.

FRED identifies the selected [hours series](https://fred.stlouisfed.org/series/PRS85006023) and [GDP deflator](https://fred.stlouisfed.org/series/GDPDEF) as quarterly seasonally adjusted indexes. The [population denominator](https://fred.stlouisfed.org/series/CNP16OV) is monthly and not seasonally adjusted; FRED explicitly notes population is not seasonally adjusted even in employment tables labelled seasonally adjusted. This qualification should accompany a blanket statement that all inputs are seasonally adjusted.

## Fixed calibration is consequential

The historical constant `SW07_POSTERIOR_MODE` contains the bundled Pfeifer initialization, not the paper's estimated posterior mode. Its risk-premium standard deviation is **1.8513**. Both the bundled source and current upstream source specify that value in the `shocks` block, while the estimation starting value is **0.1818513**. The paper's mode is **0.24**. This is not evidence that an arbitrary decimal correction should be applied; it is evidence that the initialization must not be interpreted as an estimated calibration. The source also explains the risk-premium process's scaling convention and why the reported 0.24 belongs to that implementation. [Pfeifer source and explanatory notes](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod), [SW07 Table 1b](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf).

At the declared calibration, the risk-premium innovation contributes **47.1712 of 48.2591** to GDP-growth variance (97.75%), **2.7111 of 2.9090** to inflation variance (93.20%), and **16.7777 of 17.0858** to interest-rate variance (98.20%). Technology contributes only **0.1848**, **0.0236**, and **0.0250**, respectively. The government equation's loading on the technology innovation is retained in this decomposition; contributions partition the seven independent innovations, not seven artificially independent shock processes.

The technology persistence is 0.9977 (half-life about 301 quarters), versus 0.95 in the published mode. However, changing only technology persistence to 0.95 leaves predicted GDP-growth variance at **48.1883**. A highly persistent technology *level* does not imply an equally dominant contribution to differenced output-growth variance.

## Exact finite-sample expectation

For each innovation, solve the model's Lyapunov covariance and construct all temporal covariances for the 156-quarter observation sequence. With `D=I-11'/156`, the exact centered covariance matrix is `D C_ab D`. Average its entries `(t,t-h)` over the study's common window `t=4,...,155`. This treats the estimated full-sample means exactly and includes the retained first four observations in those means. No asymptotic approximation, burn-in choice, or simulation is involved. A scalar AR(1) control matches the independently known identity `E[sample variance] = population variance - Var(sample mean)` to 1e-14.

| Calibration and moment | Observed | Population | Expected sample centered |
|---|---:|---:|---:|
| Declared: GDP-growth variance | 0.716890 | 48.259068 | 48.239088 |
| Declared: inflation variance | 0.370395 | 2.908966 | 2.694628 |
| Declared: interest-rate variance | 0.698755 | 17.085808 | 16.139774 |
| Published mode: GDP-growth variance | 0.716890 | 0.871872 | 0.869648 |
| Published mode: inflation variance | 0.370395 | 0.290129 | 0.243446 |
| Published mode: interest-rate variance | 0.698755 | 0.364379 | 0.300163 |
| Published mode, fitted crr/em: GDP-growth variance | 0.716890 | 0.886040 | 0.883832 |
| Published mode, fitted crr/em: inflation variance | 0.370395 | 0.306950 | 0.262479 |
| Published mode, fitted crr/em: interest-rate variance | 0.698755 | 0.343874 | 0.286180 |

The sample means are GDP growth **0.41893**, inflation **1.00341**, and the interest rate **1.66415**. Central-moment estimation removes them; substituting fixed model intercepts would define a different empirical target. The deterministic observation intercepts therefore cannot repair this covariance mismatch. Finite-sample demeaning can materially affect persistent inflation and interest-rate moments even though it does not explain the declared calibration's orders-of-magnitude failure. Sampling variation around these expectations, parameter reestimation, calibration uncertainty, and historical regime changes require separate investigation.

Reproduce with `.venv/bin/python reviews/2026-10-01-sw07-finite-sample/audit_data_mapping.py`. The script writes [all shock/moment decompositions](mapping_moment_decomposition.csv) and [source hashes, parameters, numerical checks, and sample summaries](mapping_audit.json). It shares the package's DSGE solution and observation matrices, so it is an independent data/moment mapping audit, not an independent model-solver benchmark. No production source or calibration was changed.
