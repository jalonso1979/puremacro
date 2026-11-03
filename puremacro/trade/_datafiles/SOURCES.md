# Third-party data bundled with `puremacro.trade`

puremacro's own code is MIT-licensed (see `LICENSE` at the repository root).
**That licence does not cover the data in this directory.** The input-output
table is adapted from OECD data; the household sample comes from INEGI.
Retain the relevant publisher's attribution and terms when sharing results.

## `enigh2022_household_policy.npz` and `.json`

Fuente: INEGI. Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH)
2022 nueva serie, tabla `concentradohogar`.
[Survey](https://www.inegi.org.mx/programas/enigh/nc/2022/),
[source archive](https://www.inegi.org.mx/contenidos/programas/enigh/nc/2022/microdatos/enigh2022_ns_concentradohogar_csv.zip),
[INEGI free-use terms](https://www.inegi.org.mx/inegi/terminos.html).

PureMacro retains `factor`, `ing_cor`, `gasto_mon`, `alimentos`, `tot_integ`,
`est_dis`, and `upm` for all 90,102 source rows, in CSV order. No household
identifier, imputation, sampling or monetary rescaling is added. Source archive,
source CSV and packaged payload SHA-256 hashes are in the adjacent manifest.
Money is quarterly MXN at August 2022 prices. `alimentos` includes food,
beverages and tobacco. The policy example declares a positive-monetary-spending
domain: 69 zero-spending rows remain in the source payload and exclusion record.

These transformations and policy analyses are PureMacro's work, not INEGI's.
INEGI does not endorse the analysis. Its free-use terms permit copying,
extraction and adaptation with source attribution and retained metadata.
Regenerate with `tools/reference_validation/build_enigh_policy_inputs.py` and
the original archive. The example authenticates the payload before use.

---

## `icio_oecd2023_2019_isic_g20.npz`

| | |
|---|---|
| Arrays | `data` (float64, shape `(403, 460)`), `country_codes`, `sector_codes`, `fd_codes`, `metadata_json` (the build record) |
| Origin | OECD Inter-Country Input-Output (ICIO) tables, 2023 edition, reference year 2019 (`2019_SML.csv`, member of `2016-2020_SML.zip`) |
| Source archive | <https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2023/2016-2020_SML.zip> (85,359,213 bytes, MD5 `707f2a506eec51b07622ae6efa0177ab`; downloaded 2026-10-16) |
| Source file MD5 | `28cba31491177955445051d459053744` (`2019_SML.csv`, 69,042,812 bytes) |
| Source file SHA-256 | `985e3141c3da4ad56ccc9dd0721eadf540b363d6929fdbc6d8fbf5417433ac0e` |
| Bundled file | 576,804 bytes, SHA-256 `07bc20e77cd15a65dec28b195f6eeef672a1336fd8b4f03a04bd1d3e3efa9b77` (the builder reproduces it byte for byte) |
| Publisher | Organisation for Economic Co-operation and Development (OECD) |
| Source page | <http://oe.cd/icio> |
| Unit | current million USD, basic prices |
| Loaded by | `puremacro.trade.data.load_icio_data()` (`source="oecd_isic"`, the default) |

### Citation and terms

Cite the data as the OECD asks in the ICIO ReadMe:

> OECD (2023), OECD Inter-Country Input-Output Tables, http://oe.cd/icio

The OECD Terms and Conditions (<https://www.oecd.org/en/about/terms-conditions.html>,
section 3, "Data") permit anyone to "extract from, download, copy, adapt, print,
distribute, share and embed Data for any purpose, even for commercial use", with
appropriate credit to the OECD. When you share or license work built on this
table, the same terms ask you to pass on the acknowledgment requirement to your
sub-licensees.

This is an adaptation of an original work by the OECD. The opinions expressed
and arguments employed in this adaptation should not be reported as
representing the official views of the OECD or of its Member countries. The
OECD is not affiliated with puremacro and does not endorse it.

The OECD's statistical data for Israel are supplied by and under the
responsibility of the relevant Israeli authorities; their use by the OECD is
without prejudice to the status of the Golan Heights, East Jerusalem and
Israeli settlements in the West Bank under the terms of international law.

### Countries

Rule: the 19 G20 member countries keep their own rows and columns (the European
Union and the African Union, members as blocs, are not separate economies); the
other 57 economies of the OECD tables and the OECD's own rest of the world are
summed into `ROW`. Order: alphabetical, `ROW` last.

Country codes (20): `ARG AUS BRA CAN CHN DEU FRA GBR IDN IND ITA JPN KOR MEX RUS SAU TUR USA ZAF ROW`

### Sectors

Grouping: the sections of the International Standard Industrial
Classification of All Economic Activities, Revision 4 (United Nations, 2008).
Each of the 45 OECD activities belongs to one section, whose letter starts the
OECD code; the OECD tables have no section U.

Sector codes (20): `A B C D E F G H I J K L M N O P Q R S T`

| Section | Title | OECD activities |
|---|---|---|
| A | Agriculture, forestry and fishing | A01_02, A03 |
| B | Mining and quarrying | B05_06, B07_08, B09 |
| C | Manufacturing | C10T12, C13T15, C16, C17_18, C19, C20, C21, C22, C23, C24, C25, C26, C27, C28, C29, C30, C31T33 |
| D | Electricity, gas, steam and air conditioning supply | D |
| E | Water supply; sewerage, waste management and remediation activities | E |
| F | Construction | F |
| G | Wholesale and retail trade; repair of motor vehicles and motorcycles | G |
| H | Transportation and storage | H49, H50, H51, H52, H53 |
| I | Accommodation and food service activities | I |
| J | Information and communication | J58T60, J61, J62_63 |
| K | Financial and insurance activities | K |
| L | Real estate activities | L |
| M | Professional, scientific and technical activities | M |
| N | Administrative and support service activities | N |
| O | Public administration and defence; compulsory social security | O |
| P | Education | P |
| Q | Human health and social work activities | Q |
| R | Arts, entertainment and recreation | R |
| S | Other service activities | S |
| T | Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use | T |

### Final uses

`C` = HFCE + NPISH + GGFC (final consumption), `I` = GFCF + INVNT (gross
capital formation, which receives the foreign-balance closure), `Cx` = DPABR
(residents' direct purchases abroad).

### How it is built

```
python tools/build_icio_isic.py --download
```

`--download` fetches `2016-2020_SML.zip` from the OECD site and keeps it in
`$PUREMACRO_OECD_ICIO_DIR` (default `~/.cache/puremacro/oecd_icio`); a copy
already on disk can be passed instead with `--source /path/to/2016-2020_SML.zip`
or `--source /path/to/2019_SML.csv`. The builder refuses a file whose MD5 is
not the one above, reads it with the strict native reader, and then:

1. groups final uses as above and sums countries and sections by the two rules
   above (every total is conserved);
2. nets negative final-use cells (inventory draw-downs larger than the other
   final sales of the same product): in such a row the negative cells become
   zero and the positive cells are scaled to the row's total, so each
   product's total final use and output are unchanged; a row with a negative
   total stops the build;
3. runs the library's own table pipeline: output set to row sales, the
   residual net-tax closure, value-added floors on near-empty nodes, and value
   added split 2/3 labour, 1/3 capital. This split is a modelling convention of
   the library, not OECD data: the OECD tables report value added without a
   labour/capital split.

The build record inside the file (`metadata_json`) lists the netted cells, the
floored and phantom nodes and the world totals. For this file: 4 negative
final-use cells netted (0.41 million USD in all); no value-added floors; three
phantom nodes with zero output (`AUS_T`, `CHN_T`, `JPN_T`: the OECD reports no
household-employer activity there); world value added 81.59 trillion USD and
world gross output 170.74 trillion USD.
