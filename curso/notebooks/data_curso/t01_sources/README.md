# Sources for the T01 core sequence

> **Superseded on 2026-10-07.** These files fed the previous pipeline. The panels are now rebuilt by `T01_A_data_panels` from the puremacro blocks in `../t01_blocks/`; nothing here is read by a notebook or a test, and `t01_manifest.json` does not list these files. They are kept, not erased.

These are provider responses used to construct the new teaching snapshots.
They are source data, not generated notebook outputs. Normalized panels,
country-variable coverage tables and `t01_manifest.json` are in the parent
directory. The manifest records hashes and source/retrieval dates. An assembly
date does not imply a common information set across releases.

| Files | Source |
|---|---|
| `countries.json.gz`, WDI indicator JSON files | [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581); country metadata excludes aggregates while retaining reporting territories |
| `ip.csv.gz` | OECD `DSD_STES@DF_INDSERV,4.3`, monthly industrial production, industry excluding construction, calendar/seasonally adjusted |
| `unemployment.csv.gz` | OECD `DSD_LFS@DF_IALFS_UNE_M,1.0`, monthly unemployment, total sex, ages 15+, seasonally adjusted |
| `pwt110.xlsx` | [Penn World Table 11.0](https://doi.org/10.34894/FABVLR), original workbook; selected columns and its variable dictionary are stored beside the panels |

PWT attribution: Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015),
“The Next Generation of the Penn World Table,” *American Economic Review*,
105(10), 3150–3182. PWT 11.0 is distributed under CC BY 4.0; see
[the provider documentation](https://www.rug.nl/ggdc/productivity/pwt/).

The existing course bundle supplies the quarterly national accounts and the
separate OECD vintage archive; they were not overwritten. FRED's `UNRATE`,
`SAHMCURRENT` and `SAHMREALTIME` snapshots are stored in the parent directory
with a `t01_` prefix. Their URLs and hashes are in the manifest.
