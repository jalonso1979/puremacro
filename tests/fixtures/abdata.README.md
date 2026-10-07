# Arellano-Bond (1991) employment data (`abdata`)

`abdata` is the unbalanced 140-firm, 1976-1984 panel of UK companies
used by Arellano and Bond (1991, *Review of Economic Studies* 58) and by
the examples of Stata's `xtabond` manual. It is **not bundled** with
puremacro. The tests that need it skip unless a copy is found.

## How to obtain

The tests were verified with the Stata Press copy:

- **Stata Press download (verified):** `https://www.stata-press.com/data/r19/abdata.dta`
  (1,031 observations, 140 firms; md5 `734bf07e03600f544857933eae385787`
  for the copy used).
- **Stata:** `webuse abdata`, then `save abdata.dta`.

Other redistributions, such as R's `plm::EmplUK` with the variables
renamed and logged, have not been checked against the pinned values
below. Stata stores the variables as 4-byte floats, and the tests compare
7 printed decimals, so a copy rebuilt from other sources may not
reproduce them.

## Where the tests look

`tests/test_dynpanel/conftest.py` (`load_abdata_xtabond_design`) uses the
first file it finds:

1. the path in the environment variable `PUREMACRO_ABDATA`;
2. `tests/fixtures/abdata.dta`;
3. `tests/fixtures/abdata.csv`.

Both `.dta` (read with `pandas.read_stata`) and `.csv` files are
accepted. For example:

```bash
PUREMACRO_ABDATA=/path/to/abdata.dta python -m pytest tests/test_dynpanel -q
```

## Required schema

The loader needs these columns; any others are ignored:

| Column | Description |
|--------|-------------|
| `id`   | Firm identifier |
| `year` | Year (1976-1984) |
| `n`    | log(employment) |
| `w`    | log(real product wage) |
| `k`    | log(gross capital) |
| `ys`   | log(industry output) |

The loader builds the lags within firm, using only consecutive years, and
the year dummies `yr1980`-`yr1984` itself.

## Pinned values

The tests pin the output of Stata's `xtabond` manual
(<https://www.stata.com/manuals/xtxtabond.pdf>; the variables are defined
in its Example 1, PDF p. 4). Each example estimates

```
xtabond n l(0/1).w l(0/2).(k ys) yr1980-yr1984 year, lags(2) noconstant
```

with the options below, and each reports 611 observations, 140 groups
and 41 instruments:

| Example | Options | Output on | L1.n | s.e. | Test file |
|---------|---------|-----------|------|------|-----------|
| 1 | one-step, homoskedastic VCE | PDF p. 5 | .6862261 | .1486163 | `test_ab_1991_replication.py` |
| 2 | one-step, `vce(robust)` | PDF p. 7 | .6862261 | .1445943 | `test_ab_1991_replication.py` |
| 4 | `twostep vce(robust)` (Windmeijer-corrected) | PDF p. 9 | .6287089 | .1934138 | `test_fix_dynpanel_stata_xtabond.py` |

All 16 coefficients and standard errors of each example are compared at
an absolute tolerance of 1e-7, since Stata prints 7 decimals. The manual
notes that the coefficients of Example 1 and the robust standard errors of
Example 2 are those of Arellano and Bond (1991), Table 4, column a1.
