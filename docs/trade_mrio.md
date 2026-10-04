> 🇬🇧 English · 🇪🇸 [Español](es/trade_mrio.md)

# Native MRIO tables: provenance, readers, aggregation and coarse tariffs

`puremacro.trade.mrio` reads published multi-regional input-output databases
at their native resolution, records where every number came from, balances
them under a stated contract, aggregates them exactly, and coarsens tariff
schedules with the two rules used in the IO research workspace. It is the
ingestion layer that the trade solvers sit on: a table becomes a legacy
calibration matrix through `to_calibration_matrix`, and a calibration comes
back as a table through `MRIOTable.from_trade_calibration`.

```python
import numpy as np
from puremacro.trade import calibrate_trade_model, load_icio_data
from puremacro.trade.mrio import (
    Concordance, MRIOTable, aggregate_mrio, coarse_tariff_rates,
    regularize_table, to_calibration_matrix,
)

calib = calibrate_trade_model(load_icio_data(source="legacy"))
table = MRIOTable.from_trade_calibration(calib, negative_investment="to_inventory")
print(table.summary())
print(table.accounting_report().to_markdown())

blocs = Concordance.from_mapping(
    "blocs", {c: c if c in ("USA", "CHN") else "REST" for c in table.country_codes})
coarse = aggregate_mrio(table, Concordance.identity(table.sector_codes), region_concordance=blocs)
print(coarse.to_markdown("countries"))

three = Concordance.from_mapping(
    "three", {c: "GOODS" if k < 2 else "SERVICES" for k, c in enumerate(table.sector_codes)})
rates = np.zeros((77, 11))
rates[:, 1] = 0.25
rates[table.country_index("USA")] = 0.0
wedges = coarse_tariff_rates(rates, table, three, "bilateral")
print(wedges.summary())

matrix = to_calibration_matrix(table)          # (850, 1078): calibrate_trade_model(matrix) accepts it
```

The bundled table has three negative investment cells (inventory drawdowns
that the legacy layout merged into GFCF), which is why the bridge asks for an
explicit `negative_investment` policy. Native files are read the same way:

```python
# requires: the native OECD ICIO 2023, Eurostat FIGARO 2026 and EXIOBASE 3.8.2 files (not shipped)
from puremacro.trade.mrio import read_exiobase_native, read_figaro_native, read_oecd_native

oecd = read_oecd_native("2019_SML.csv", 2019)          # refuses the known-corrupted digest
figaro = read_figaro_native("matrix_eu-ic-io_ind-by-ind_26ed_2019.csv")
exio = read_exiobase_native("IOT_2019_ixi.zip")         # sparse Z, members checked against the archive CRCs
core = read_exiobase_native("exiobase_core", archive="IOT_2019_ixi.zip")  # extracted members checked too
balanced, report = regularize_table(oecd)
eleven = aggregate_mrio(balanced, "agregar")
print(report.to_markdown("gates"))
```

A two-country FIGARO file and a two-region EXIOBASE directory in the official
layouts ship under `tests/fixtures/trade/mrio/`, so
`read_figaro_native("tests/fixtures/trade/mrio/figaro_2c2s_26ed.csv")` runs
from a checkout.

## The table contract

An `MRIOTable` is a frozen dataclass whose arrays are read-only copies. Cells
are country-major, `i = country_index * S + sector_index`.

| attribute | shape | meaning |
|---|---|---|
| `Z` | `(M, M)` dense or `scipy.sparse.csr_matrix` | basic-price intermediate deliveries; rows sell, columns buy |
| `F` | `(M, N, K)` | final deliveries by seller cell, destination country, category |
| `fd_codes` | `K` strings | subset of `("C", "G", "X", "V", "VAL")` in that order |
| `VA`, `TLS`, `output` | `(M,)` | value added as recorded, net taxes less subsidies on the buyer's intermediate purchases, gross output |
| `TFD` | `(N, K)` | net taxes less subsidies on final purchases |
| `production_taxes`, `labor_compensation`, `operating_surplus` | `(M,)` or `None` | factor detail where the source has it |
| `merchandise_mask` | `(S,)` bool or `None` | goods sectors (ISIC A-C) |
| `metadata` | dict | `sources` (tuple of `SourceRecord`), `transformations` (tuple of strings), conventions |

C is household, NPISH and government consumption; G is gross fixed capital
formation; X is residents' purchases abroad (the OECD DPABR basket); V is the
signed change in inventories; VAL is valuables. The validating constructor is
`MRIOTable.from_arrays`: `Z`, G and X must be nonnegative, everything must be
finite, and labour and operating surplus are supplied together. Negative
consumption, inventories, valuables, value added and taxes are real source
data and are kept. `accounting_report()` recomputes the row and column
identities independently and counts the negative entries; nothing is
rebalanced there. `net_exports()` measures fob balances over all uses and
assigns the floating-point round-off of their world sum to
`reference_country`. `to_raw("cix")` gives a `RawIOData` with the legacy C, I
= G + V + VAL, Cx = X columns.

## Provenance

`identify_source(path, expected_sha256=..., expected_md5=...)` streams the
file in 4 MiB blocks and refuses an absent or changed file.
`verify_zip_member(archive, member, extracted_path=None)` checks a member's
size and CRC-32 against the archive directory. Every reader stores the
resulting `SourceRecord`s in `metadata["sources"]`, with a `status` of
`"verified"`, `"whitelisted"`, `"hashed"` or `"unknown_edition"`.

`check_oecd_source(path, year=...)` compares an OECD CSV (or the year member
of the 2016-2020 ZIP) with two registries:

| registry | digest | meaning |
|---|---|---|
| `OECD_ICIO_MD5[2019]` | `28cba31491177955445051d459053744` | accepted `2019_SML.csv`, OECD ICIO 2023 edition |
| `OECD_ICIO_MD5[2020]` | `d3e0f4979d85d6c0bb7cf4c43e324287` | accepted `2020_SML.csv` |
| `OECD_KNOWN_CORRUPTED_MD5` | `d1b887aaafa54ab3f28fde78fcd21cdf` | `data_2020_SML.csv` with lost decimal points: tokens with 3-4 decimals lost their decimal point and tokens below 0.001 became 0; refused |

A corrupted digest raises `MRIOIntegrityError`; a whitelisted digest is
recorded; anything else is read with a `RuntimeWarning` naming it an
unauthenticated edition. The FIGARO reader compares the CSV with
`FIGARO_2026ED_2019_SHA256` and the EXIOBASE reader compares the archive with
`EXIOBASE_382_2019_IXI_SHA256` in the same way; both enforce an
`expected_sha256` when one is passed, and a registered file read for a year
other than 2019 raises. EXIOBASE members read from the archive are checked
against its CRC-32 entries. An extracted EXIOBASE directory is authenticated
only when its archive is passed (`archive=`): the archive is hashed and every
extracted member is checked against the archive entry's size and CRC-32, as
the IO loader does, so one changed byte raises. Read alone, a directory's
members are hashed, recorded as `"unknown_edition"` and flagged with a
`RuntimeWarning`; `metadata["source_authenticated"]` says which case applied.
Nothing is downloaded and no substitute fixture is generated.

## Readers and source conventions

| | OECD ICIO 2023 edition | Eurostat FIGARO 2026 edition | EXIOBASE 3.8.2 ixi |
|---|---|---|---|
| reader | `read_oecd_native(path, year)` | `read_figaro_native(path, year=2019)` | `read_exiobase_native(path_or_zip, year=2019, archive=None)` |
| layout | 77 x 45 SML CSV with TLS/VA/OUT rows (parsed by `puremacro.trade._oecd_icio.read_native`) | `rowLabels` header, `{region}_{industry}` cells, five final-use columns per region, six trailing `W2_*` rows; dimensions derived from the labels | `Z.txt`, `Y.txt`, `x.txt`, `industries.txt`, `satellite/F.txt`, `satellite/F_Y.txt` with two-level region/sector headers |
| units | USD million, current prices | EUR million, current basic prices | EUR million, current basic prices |
| final uses | C = HFCE + NPISH + GGFC, G = GFCF, X = DPABR, V = INVNT | C = P3_S13 + P3_S14 + P3_S15, G = P51G, X = 0, V = P5M | C = y01 + y02.a + y02.b, G = y04, X = 0, V = y05.a, VAL = y05.b |
| consumption | resident consumption with a separate DPABR basket; includes government | territorial consumption; includes government; `W2_OP_RES` and `W2_OP_NRES` totals kept in metadata | published consumption; includes government; no DPABR basket |
| inventories | INVNT is the signed change; no stock is observed | P5M merges inventory changes and valuables; the source does not separate them | inventories and valuables are separate flows |
| factor detail | none: the VA row has no observed labour/capital split | `W2_D1` labour, `W2_D29X39` other net production taxes, `W2_B2A3G` gross operating surplus and mixed income | three skill rows for labour, four operating-surplus rows (including land and resource rents), other net taxes on production |
| output | provider OUT column | row total of the published transactions | `x.txt` |
| year | 1995-2020 | 2019 authenticated | 2019, a nowcast |
| storage | dense | dense (`sparse_matrix=True` optional) | CSR streamed in 128-row blocks |

Region codes are mapped to ISO-3 through `FIGARO_ISO3` (`FIGW1` becomes
`ROW`) and `EXIOBASE_ISO3` (the five rest-of-world regions keep distinct
names). Merchandise masks follow `goods_mask`: ISIC sections A-C for the OECD
industries, NACE A-C except C33 (repair and installation, which "is not a
merchandise shipment") for FIGARO, industries i01-i37 except i01.w for
EXIOBASE.

The FIGARO reader applies, and records under
`metadata["rounding_adjustments"]`, the two bounded repairs of the IO reader:
an empty cell whose output is one rounding unit negative (`-0.001 <= y < 0`
with absolute input mass at most `0.002`) is zeroed, and a consumption entry
in `[-0.001, 0)` is moved to V so that row totals and country expenditure
are unchanged. Anything larger raises. `rounding_repair=False` keeps the
published rounding. The EXIOBASE reader refuses a nonzero `Exports: Total
(fob)` column and nonzero final-demand factor rows, because those would need
an explicit destination model.

## Regularization contract

`regularize_table(table)` ports the eight-step build of the IO
`table_from_arrays` for any table with named categories. With `out` the
output column, `row` the row totals and `active = out > 0`:

1. Identity guard: `median_{active} |row_i - out_i| / out_i <= 1e-5`, no
   negative output, and every zero-output cell is completely empty.
2. Phantoms: each zero-output cell receives `F[i, own country, C] = VA_i = 1e-6`
   so its row and column agree exactly (`inactive="mask"` keeps them empty).
3. Value-added floor: `VA_i = 1e-3 * out_i` only where active and `VA_i <= 0`.
4. Net product taxes are recomputed as the residual that closes every column,
   `tls_j = row_j - sum_i Z_ij - VA_j`. At cells that were not floored the
   change must satisfy `sum_j |tls_j - TLS_j| <= 1e-4 * sum_j |TLS_j|` and
   `max_j |tls_j - TLS_j| / max(out_j, 1) <= 1e-2` (with the IO's `1e-9`
   relative slack; the 2019 OECD table sits exactly at 0.0100).
5. Trade balances are measured fob over all uses; `|sum_o xn_o| <= 1e-9 * world GDP`
   with `world GDP = sum VA + sum tls + sum TFD`.

The returned table has `output` equal to the row totals after phantoms, the
residual `TLS`, and the step appended to `metadata["transformations"]`; the
`MRIOBuildReport` carries the phantoms, floors, both residual-tax totals, the
world trade-balance round-off, world GDP and every gate with its value,
tolerance and pass flag, and names the offending cells where a gate is about
cells (`"cells"`, `"n_cells"`, and `"worst_cell"` for the per-cell gate).
Every tolerance is a keyword; `strict=False` returns the report with
`passed=False` instead of raising (the whitelisted 2020 OECD table fails the
per-cell gate at 1.06%). Such a table is marked `metadata["regularized"] =
False` with the failed gates in `metadata["failed_gates"]` and in its
transformation string, and `aggregate_mrio` and `to_calibration_matrix` warn
when they receive it. Under `strict=False` only empty cells with output
exactly zero receive a phantom; cells with negative output or stray entries
keep their data and are listed in `report.skipped_phantoms`. The phantom
`1e-6` and the `max(out, 1)` scale of the per-cell gate are absolute
amounts: like the IO code they assume a table in millions of currency units,
so rescale `phantom` for a table in other units. `spectral=True` adds the
Collatz-Wielandt bounds of `Z / output` from
`puremacro.trade.regularize.compute_spectral_radius`. This is a different
contract from `regularize_mrio_table`, which floors every active cell's value
added at `max(1e-3 Y, min(1, 0.5 Y))` (4.3.0: `max(1e-3 Y, 1)`, which lifted
value added above output on cells below 1 M USD) and re-derives output from row
sales before the residual unless `Y` is supplied; the two are not
interchangeable.

## Aggregation

`aggregate_mrio(table, concordance)` computes the exact aggregation with
`P = I_N kron C^T`, where `C` is the `(S, G)` 0/1 matrix of a `Concordance`:

```
Z_G = P Z P^T,   F_G[:, n, k] = P F[:, n, k],   VA_G = P VA,   TLS_G = P TLS,   output_G = P output
```

`TFD` is unchanged, factor detail is summed the same way, and a
`region_concordance` merges origins and destinations as well. Both accounting
identities are linear, so aggregation commutes with the residual-tax closure
whenever no floor applies, and every total is conserved to floating-point
rounding. When the fine table is balanced the coarse table must be too
(`max_j |row_j - col_j| / max(|col_j|, 1) <= 1e-8`); the residual is recorded
in `metadata["aggregation"]` either way. A coarse group is merchandise only
when every member is; mixed groups are listed.

Two named maps of the 45 OECD industries are registered in `CONCORDANCES`:

| name | groups | goods groups |
|---|---|---|
| `"agregar"` | `AGG11_SECTOR_CODES`: AGRI_MIN_FOOD (A01_02 to C10T12), MANUF_EXFOOD (C13T15 to C31T33), UTILITIES (D, E), CONSTRUCTION, TRADE, TRANSPORT (H49-H53), ACCOM_ICT (I, J58T60-J62_63), FIN_REALESTATE (K, L), PROF_ADMIN_PUBADM (M, N, O), EDU_HEALTH (P, Q), ARTS_OTHER (R, S, T) | AGRI_MIN_FOOD, MANUF_EXFOOD |
| `"isic_section"` | `ISIC_SECTION_11_CODES`: A, B, C, DE, F, G, H, J, KL, OPQ, REST (I, M, N, R, S, T) | A, B, C |

The Agregar map is the one the bundled 77 x 11 table was built with. Its
first group contains food manufacturing and its third group is utilities, so
puremacro's `CANONICAL_SECTOR_CODES` (AGRI, MINQ, MANU, ENEG, ...) label those
positions but do not describe their content; `goods_mask` refuses those
labels unless the classification is named (`"agregar11"` for the true
groups, `"canonical11"` to read the labels literally). The mask is decided
label by label: under `"agregar11"` each canonical label stands for the
Agregar group at its position in `CANONICAL_SECTOR_CODES`, so any order or
subset gives the same answer per code, and a code outside the named registry
raises `ValueError`.

## Coarse tariff rules

`coarse_tariff_rates(fine_rates, table, concordance, rule)` takes an `(N, S)`
schedule of ad-valorem duties `r_{os} >= 0` (multiplier `tau = 1 + r`) that
the importer levies on origin `o` and sector `s`, and the importer's benchmark
purchases at basic prices `m_{os} = sum_{j in importer} Z_{(o,s),j} + sum_{f in fd_tariffed} F_{(o,s),importer,f}`
with `fd_tariffed = ("C", "G", "V", "VAL")` by default, of which the
categories the table carries are used. Valuables are tariffed with
inventories, as in the IO EXIOBASE build (which merged them into V) and in
FIGARO (whose P5M contains them). Rates must be nonnegative (`ValueError`
otherwise), X is never tariffed and the importer's own row is zero.

* `rule="output"` (the IO `coarse_11o`): `r_{oG} = sum_{s in G} m_{os} r_{os} / sum_{s in G} m_{os}`,
  zero when the weight sum is not positive. With nonnegative weights the
  coarse rate lies between the smallest and largest fine rate of the group.
* `rule="bilateral"` (the IO `coarse_11b`): buyer-specific duty equivalents
  `tau_{(o,G),(imp,J)} = 1 + sum_{s in G, j in J} Z_{(o,s),j} r_{os} / sum_{s in G, j in J} Z_{(o,s),j}`
  and `tau^f_{(o,G),imp} = 1 + sum_{s in G} F^f_{(o,s),imp} r_{os} / sum_{s in G} F^f_{(o,s),imp}`
  per tariffed category, ones where the denominator is not positive. These
  preserve benchmark duty revenue block by block wherever the block's
  benchmark purchases are positive; a block with nonpositive purchases (an
  inventory drawdown under `negative_weights="keep"`) keeps a unit multiplier
  and drops its fine-schedule duty, exactly as the IO rule's `tot > 0` guard
  does. The output rule preserves each origin-group total as well, because its
  weights are exactly the tariffed purchases; what it loses is the allocation
  across buyer groups and categories.

Signed inventories can make a weight negative. The IO rules use the raw sums
(`negative_weights="keep"`, the default) and only guard the denominators;
`negative_weights="clip"` clips each final-use contribution at zero, since a
drawdown is not an import. The `CoarseTariffResult` carries the rates, the
weights, the wedges `tau` `(N G, N G)` and `tau_fd` `(N G, N, K)`, and the
benchmark revenue under the fine and the coarse schedules.

## Bridges to the legacy calibration

`to_calibration_matrix(table)` returns the `(M + 3, M + 3N)` matrix for
`calibrate_trade_model`: rows `0..M-1` are `[Z | F]` with C, I = G + V + VAL
and Cx = X per destination, row `M` is TLS with the final taxes condensed the
same way, rows `M + 1` and `M + 2` are labour and capital.
`calibrate_trade_model` reconstructs each column from a Cobb-Douglas split
and accepts it only when both factors are strictly positive (a capital share
strictly between 0 and 1) or both are zero, so the factor rows are built to
that standard:

* Other net production taxes move to the TLS row by default
  (`production_taxes="to_tls"`), which the legacy calibration models as an
  output tax, except at cells where they are at least as large as value
  added. There they stay in value added and a `RuntimeWarning` names the
  cells: 3 cells of the regularized FIGARO 2019 table (ARG_A01, CYP_H51,
  JPN_M73) and 30 of EXIOBASE 2019, 9 of which a plain subtraction would have
  left with negative factor income. `"to_factors"` keeps every production
  tax inside value added.
* Labour and capital split the remaining factor value added in the observed
  proportions where both are strictly positive, and 2/3-1/3 elsewhere:
  phantom cells, cells with a negative factor, and cells with one factor
  exactly zero (the FIGARO 2019 table has 30 active cells with zero operating
  surplus and 7 with zero labour). Without detail every cell uses 2/3-1/3.
* A cell with negative factor value added raises `MRIOIntegrityError`;
  `regularize_table` floors value added where it is not positive.

`return_details=True` also returns the cells on each fallback. A raw table
triggers a `RuntimeWarning` recommending `regularize_table` first, and so does
a table whose regularization gates failed. `MRIOTable.from_trade_calibration(calib)`
is the inverse bridge from `calib.data_calibra`. It reads the calibration's Cx
column as X (tariff-exempt residents' purchases abroad) only for the bundled
layout and the OECD `DPABR` mapping (`calib.metadata["final_use_mapping"]`);
the FIGARO, WIOD and Eora loaders' Cx (signed inventories and valuables)
becomes V, and EXIOBASE's Cx, which includes its export column, raises
`ValueError` unless `cx_category` names the category. A negative investment
cell moved by `negative_investment="to_inventory"` is added to V. A table
bridged from a FIGARO, WIOD or Eora calibration therefore has no X, and
`to_calibration_matrix` returns it with an empty Cx column: solve that
calibration with `accounting="consistent"`, because the legacy demand
`theta * Y / ppfd` is 0/0 for an empty category (the legacy solver returns
NaN with `converged=False`).

## Provenance of puremacro's bundled 77 x 11 table

The first fact was identified in the MATLAB pipeline directory
`computation/7_TIO_77c_vf` from the file and its MD5 (this module refuses to
read that file, so it cannot be re-derived here); the comparisons after it
were measured on 2026-09-22 with this module.

* The unaggregated file that the original MATLAB pipeline aggregated to
  `puremacro/trade/_datafiles/icio_77c_11s.npz` is `data_2020_SML.csv` with
  MD5 `d1b887aaafa54ab3f28fde78fcd21cdf`, the digest registered in
  `OECD_KNOWN_CORRUPTED_MD5`: tokens with three or four decimals lost their
  decimal point and tokens below 0.001 became zero. `load_raw_45sector_icio`
  now checks the MD5 of the file it resolves and refuses this digest with
  `MRIOIntegrityError`; its default search tries the clean
  `ICIOextended/2020_SML.csv` and `2019_SML.csv` before that legacy path.
* The bundled table's world value added is 7.046e11 USD million; the
  whitelisted OECD 2020 release aggregated with the same Agregar map gives
  7.971e7, a factor of 8,840. Its largest intermediate cell is 5.83e10 (CHN
  manufacturing to itself) against 6.79e6 in the clean release, and 962
  cells exceed 1e8 against none. Of the 717,409 intermediate cells, 35.5%
  are equal in both tables (25.5% of the 621,488 cells that are nonzero in
  at least one: tokens with at most two decimals survive the corruption
  unchanged); the value-added row agrees in 1 of 847 cells (0.12%) and the
  final-demand block in 37.7% of its cells.
* The MATLAB reference solutions in `trade_reference_solutions.npz` were
  computed from the same corrupted table, so the parity suites that compare
  puremacro with them remain internally consistent software regressions: they
  check that the solver reproduces a fixed computation, not that the
  computation describes the 2020 world economy.
* New empirical work should read the clean native files through
  `read_oecd_native`, `regularize_table` and `aggregate_mrio`. Regenerating
  the bundled table is an author decision that would break the external
  parity contract, and this module does not do it.

## What is and is not validated

`tests/test_trade_mrio.py` checks, on hand-designed fixtures with
hand-derived expected arrays: the provenance helpers against `hashlib` and
corrupted or mismatched members; concordance algebra (projection equals the
explicit Kronecker product, composition, coarse masks); the table contract
ported from the IO unit fixture; the FIGARO and EXIOBASE readers, including
the rounding repairs, the zip, archive-checked directory and chunked paths, a
changed byte in an extracted member, and every rejection; the OECD reader on
a tiny SML-layout file; every regularization gate in strict and reporting
mode, and that a failed gate never yields a table marked regularized;
conservation, composition, region aggregation and the 45/11 goods
consistency; convexity and block-by-block revenue preservation of the coarse
rules; calibration of the aggregated OECD fixture with a baseline solve at
prices of one; calibration of tables with zero-factor cells and with
production taxes above value added; and the bundled bridge round trip.

Parity with the IO implementation is exercised in a subprocess that imports
the vendored `puremacro.trade.corrected` package (skipped when the volume is
absent): `regularize_table` against `table_from_arrays`, `aggregate_mrio`
against `aggregate_icio` for both maps, and both coarse rules against
`coarse_11o` and `coarse_11b`, with observed differences of 0 (2.8e-17 for the
11o rates). The slow tests read the real 2019 files and reproduce the
recorded IO audits: OECD 3,440 active cells, 25 phantoms, no floors, median
row identity 3.99e-7, world GDP 87,854,087; FIGARO 50 x 64, 3,133 active
cells, 20,342 negative inventory entries, the single AL_T rounding repair, 67
phantoms, 9 floors, world GDP 79,054,914; EXIOBASE 49 x 163, 1,084
zero-output cells, 73 nonpositive factor cells, inventories 1,100,405.96,
valuables 62,203.98, 1,084 phantoms, 89 floors, world GDP 78,110,997, with
the extracted core authenticated against the registered archive. The
readers agree with `dynamic_model.native_data` and
`calibrate_databases_2019.figaro` to floating-point summation order: 0 on
every OECD and EXIOBASE array and against `calibrate_databases_2019.figaro`,
at most 9.3e-10 (FIGARO output against `native_data.load_figaro`). The
regularized FIGARO 2019 table (50 x 64) passes `calibrate_trade_model`.

Not validated, and not claimed:

* Full-size general equilibrium on FIGARO or EXIOBASE. The regularized
  FIGARO 2019 table calibrates, but no equilibrium is solved on it here;
  EXIOBASE is not calibrated at all, because `to_calibration_matrix`
  densifies `Z` (7,987 square). This module promises native-resolution
  tables, not native-resolution equilibria.
* Comparability of welfare across databases. The IO calibrations record
  `residence_welfare_comparable: False`; FIGARO: "Territorial consumption is
  retained; a residence bridge has not been validated"; "This 2019 release is
  a nowcast; no stronger measurement claim is implied" for EXIOBASE; and
  "active_mask means positive reported output; it is not an economic
  feasibility certificate."
* Other editions and years. The digests identify the 2019 files that were
  audited; a different edition is read with a warning and no claim.
* A GTAP reader. The only parser that decodes real GTAP HAR files in the IO
  workspace is the vendored HARPY package, licensed GPL-3 and therefore not
  redistributable under puremacro's MIT license; the pure-Python reader in
  `headlinePaper/load_gtap11.py` only round-trips its own writer's output; and
  the GTAP data themselves are licensed and cannot ship even as fixtures.
* The IO per-cell residual-tax gate is brittle by construction: 2019 sits
  exactly at the 1e-2 threshold and the clean 2020 file fails it at 1.06%.
  The tolerances are keywords and the report is returned in either case.

## Related documents

* [Consistent trade accounting](trade_accounting.md) for the solver the
  calibration matrix feeds.
* [Hicksian consumption welfare](trade_welfare.md): native OECD C includes
  government consumption.
* [Quantitative spatial and trade general equilibrium](spatial_and_trade_ge.md)
  for the legacy loaders and the harmonized adapters.
* [Structural validation status](STRUCTURAL_VALIDATION_STATUS.md).
