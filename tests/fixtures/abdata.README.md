# Arellano-Bond (1991) employment data

`abdata.csv` is the canonical 140-firm × 1976-1984 UK manufacturing panel
used in Arellano and Bond (1991, RES 58, Table 4) and reproduced in many
econometrics packages.

## How to obtain

The dataset is redistributed under permissive terms by several
mainstream packages:

- **Stata**: `webuse abdata`
- **R (plm)**: `data("EmplUK", package = "plm")`, then `write.csv(EmplUK, "abdata.csv", row.names = FALSE)`
- **Stata Press download**: `https://www.stata-press.com/data/r17/abdata.dta`

## Required schema

The replication test (`tests/test_dynpanel/test_ab_1991_replication.py`)
expects the following columns (others are ignored):

| Column | Description |
|--------|-------------|
| `id`   | Firm identifier (integer) |
| `year` | Year (integer, 1976–1984) |
| `n`    | log(employment) |
| `w`    | log(wage) |
| `k`    | log(capital) |
| `ys`   | log(industry output) |

Column names from `plm::EmplUK` (`firm`, `year`, `emp`, `wage`,
`capital`, `output`, `sector`) need to be transformed:
`emp→n`, `wage→w`, `capital→k`, `output→ys`, `firm→id`,
and the variables logged. See the `plm` documentation for details.

## Canonical estimates (AB 1991 Table 4, col. 2)

Two-step difference GMM with Windmeijer SE on the dynamic n equation:

| Variable | Coef    | s.e.  |
|----------|---------|-------|
| L1.n     |  0.474  | 0.085 |
| L2.n     | -0.053  | 0.027 |

Without the CSV present, the replication test is **skipped**. With the
CSV present, the test asserts coefficients within 0.05 of the published
values.
