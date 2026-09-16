> 🇬🇧 English · 🇪🇸 [Español](es/real_time_data.md)

# Real-time data (vintages)

A **vintage** is one published edition of a series. Statistical offices
revise, so every reference quarter has a sequence of editions, and the
difference between the first and the last is the revision that
news-vs-noise tests are about.

```python
from puremacro.fetch import vintage_panel

rev = vintage_panel(["USA", "DEU", "ESP", "MEX"], series="B1GQ", freq="Q")

rev.coverage()            # what actually came back, per country
rev.revisions("DEU")      # preliminary / final / r_t per quarter
rev.news_or_noise("DEU")  # the Mankiw-Shapiro test pair
rev.news_or_noise_panel() # every country, one tidy table
```

`news_or_noise_panel()` is the cross-section behind "GDP revisions are
noise, not news": one row per country with `beta_on_preliminary` and
its standard error, so the claim stops being a statement about the
United States.

## Providers

| Provider | Countries | Editions | Vintage date is… |
|---|---|---|---|
| `oecd_stes` *(default)* | 42 | monthly, from 1999-02 | the OECD's snapshot month |
| `alfred` | 35 | per release | the source's release date, with fallbacks |
| `bundesbank` | DE | 111 (+42 legacy to 1995) | **the real release date** |
| `ons` | UK | 746, back to 1961 | publication month + release stage |
| `statcan` | CA | 55, from 2012-11 | **the real release date** (Daily) |
| `ecb_rtd` | EA, JP, US | version history from 2001 | ECB dissemination timestamp |
| `banxico` | MX | one per capture day | the local snapshot date |
| `inegi` | MX | one per capture day | the local snapshot date |
| `bcb` | BR | one per capture day | the local snapshot date |
| `bcch` | CL | one per capture day | the local snapshot date |

```python
from puremacro.fetch import vintage_catalog, available_providers
available_providers()
vintage_catalog("oecd_stes").head()
```

Besides the national-accounts aggregates, the catalogue carries two
monthly-or-daily variables the Latin American connectors are built
around: `policy_rate` (aliases `tpm`, `selic`, `tasa_objetivo`,
`interest_rate`, `target_rate`, ...) and `activity`, the monthly
activity index (aliases `igae`, `imacec`, `ibc_br`). `canonical_variable`
maps any accepted spelling onto the canonical name.

### The four Latin American connectors

`banxico`, `inegi`, `bcb` and `bcch` are **snapshot** connectors, not
archives. Each service publishes only the edition that is current when
you ask, so a vintage date here is the day *this machine* captured the
series and wrote it to the `realtime_vintages` table of the cache DB.
Revision history therefore accumulates only across repeated captures on
different days: one fetch gives one vintage, and a panel assembled on a
single afternoon has nothing to run a revision test on.

| Provider | Country | Series | Credential |
|---|---|---|---|
| `banxico` | MX | `policy_rate` SF61745, `cpi` SP1, `activity` SR17631 | token, sent as the `Bmx-Token` header |
| `inegi` | MX | `gdp_real` 735848, `cpi` 628197, `activity` 736184 | token, embedded in the URL path |
| `bcb` | BR | `gdp_real` 22099, `cpi` 433, `policy_rate` 432, `activity` 24363 | none (open endpoint) |
| `bcch` | CL | `gdp_real` F032.PIB.FLU.R.CLP.EP18.Z.Z.0.T, `cpi` F074.IPC.IND.Z.Z.C.M, `policy_rate` F022.TPM.TPO.D001.NO.Z.D, `activity` F032.IMC.IND.Z.Z.EP18.Z.Z.0.M | user **and** password, sent in the query string |

Catalogue entries whose identifier could not be checked against the live
service carry the marker `VERIFY ONLINE` in their `note` column, so a
guessed id is visible rather than silently authoritative:

```python
from puremacro.fetch import vintage_catalog
vintage_catalog("inegi")[["variable", "series_id", "freq", "note"]]
```

Credentials are documented in [`docs/CREDENTIALS.md`](CREDENTIALS.md),
the snapshot table in [`docs/CACHE_DB.md`](CACHE_DB.md), and the whole
workflow — capture, cartridge, revision test — in
[`docs/real_time_latam.md`](real_time_latam.md).

### Finding the OECD archive

It is easy to conclude the OECD vintage archive did not survive the
move to the Data Explorer, because the obvious identifier really is
dead — `OECD.SDD.STES,DSD_STES@DF_VINTAGES` returns nothing for any
country. The archive is alive under a different id:

```
OECD.SDD.STES,DSD_STES_REVISIONS@DF_STES_REVISIONS,4.0
```

42 economies, monthly editions from February 1999, including a genuine
GDP deflator (`B1GQ_D`) alongside real and nominal GDP and the
expenditure components.

## Vintage dates are not interchangeable

This is the subtlety that quietly corrupts cross-country work, so the
library carries it in the data rather than leaving it to memory:

```python
rev.vintage_semantics()   # per provider, what its vintage date means
```

Ordering editions — first release, second, latest — is safe with every
provider. Dating a release **event**, say against a policy
announcement, is only safe where the vintage is a national release
date. `vintage_panel` warns when a panel mixes providers.

A few specifics worth knowing:

- **ALFRED** applies a documented three-step fallback: the source's own
  release date when it supplies one, otherwise a provider date,
  otherwise the date the series first appeared in FRED. The payload
  does not say which branch produced any given date.
- **StatCan**'s WDS JSON has a `releaseTime` field that is *not* the
  vintage — it is the cube's last-refresh timestamp and is identical
  across editions. The vintage lives in the `Release` dimension.
- **ONS** identifies an edition by publication month plus stage (`1st`,
  `M1`, `M2`, `QNA`) with no day. The day component you see is assigned
  by this library to keep same-month editions ordered.
- **ECB** silently ignores the SDMX `asOf` parameter, and returns only
  the current vintage unless `includeHistory=true` is set.
- **Banxico, INEGI, BCB and BCCh** stamp a **snapshot date**, not a
  release date: the day this machine fetched the series. Ordering
  editions is still safe, but the first edition of a reference period is
  the first one *you captured*, not the office's first estimate, and no
  edition predates your first fetch.

## Some archive editions are not seasonally adjusted

The OECD STES archive has six dimensions and none of them is seasonal
adjustment. It carries whatever the OECD ingested at the time, and for
twelve reference areas the early editions are the **raw** series — the
archive switched over area by area between 2000 and 2007 without
recording it anywhere.

Nothing downstream survives that. Sweden's first estimate of 2002Q4 is
**+16.09%** quarterly growth against **+0.07%** today; Turkey's is
**−25.16%** against **+1.91%**. A news-vs-noise test run over those is
regressing a seasonal factor, and duly reports that essentially all of
the first release is measurement error. It is not. It is December.

Because the archive does not record it, it is detected from the data:

```python
from puremacro.fetch.realtime.seasonal import seasonal_signature
seasonal_signature(panel).query("unadjusted")
```

`vintage_panel` drops those editions **by default** and reports what it
removed. Pass `drop_unadjusted=False` to keep them.

Affected: AUT, CZE, HUN, IRL, ISL, LUX, MEX, POL, PRT, SVK, SWE, TUR.

## Units, levels, and why growth rates

Revision tests here default to growth rates, and transforms are applied
**within** each vintage column. Both choices are load-bearing.

*Within-vintage* matters because the growth rate a forecaster saw in
edition *v* is built from *v*'s own levels. Mixing a numerator from one
edition with a denominator from another manufactures revisions nobody
published.

*Growth rates* matter twice over. Regressing a revision on an I(1)
level is a spurious regression driven by the common trend. And levels
are not comparable across editions that straddle a benchmark revision
or rebasing — OECD, StatCan and ONS levels all shift. A rescaling of a
whole edition cancels out of that edition's growth rates, because
`log(k·x_t) − log(k·x_{t−1})` does not depend on `k`.

The catalogue also records each series' **units**, because FRED carries
both level and growth-rate series for the same concept and nothing in
the identifier says which. With `transform=None` (the default), each
series gets the transform its units imply, so a series already
published as a growth rate is not differenced twice.

## The test itself

```python
from puremacro.vintages import mankiw_shapiro, revision_frame

frame = revision_frame(long_panel)                 # preliminary / final / r_t
res = mankiw_shapiro(frame["preliminary"], frame["final"], hac_lags="auto")
res.verdict, res.beta_on_preliminary, res.noise_share
```

Both regressions are run, because failing to reject one is only
informative alongside the other:

- **news** — regress `r` on the *preliminary* value. If the office
  published an efficient forecast, the revision is a forecast error and
  is orthogonal to it, so β = 0.
- **noise** — regress `r` on the *final* value. If the published number
  was truth plus independent measurement error, `r = −u` is orthogonal
  to the truth, so β = 0.

A common reading error is to run only the first regression and call a
significantly negative β "noise". Under pure noise that coefficient is
−σ²_u/(σ²_f + σ²_u), which lies strictly inside (−1, 0) and equals −1
only in the degenerate case σ²_f = 0. Its magnitude identifies the
noise *share*, not the hypothesis — which is what `noise_share`
reports. The verdict comes from the pair of tests.

Standard errors are White by default; `hac_lags="auto"` applies the
Newey-West plug-in bandwidth, which is the safer choice on long samples
because annual benchmark revisions induce serial correlation in `r`.

## When a country comes back empty

That is reported, never silent:

```python
rev.metadata["missing"]   # countries that yielded nothing
rev.metadata["failed"]    # and why — HTTP status, parse error, ...
rev.coverage()            # n_vintages per country: 1 means no revision test
```

A country absent from a panel is **not** evidence that it publishes no
vintages. The most common cause is a rate limit — the OECD endpoint
answers a burst of requests with HTTP 429 — which the providers retry
with backoff and then report honestly.

The catalogue is also self-auditing, because identifiers rot. This
package previously resolved every non-US country to FRED's OECD-MEI
codes, which stopped updating in January 2024, and no offline test
noticed:

```bash
pytest -m network tests/test_realtime_providers/test_catalog_live.py
```

That walks every catalogued series against the live endpoints and
reports which no longer return two or more editions — the difference
between "this country publishes no vintages" and "this table went
stale".

Countries deliberately not covered by a provider, and why, are in
`puremacro.fetch.realtime.catalog.known_gaps()`, keyed `provider:ISO3`.
Most entries record what ALFRED does not archive for a country — a
discontinued series, or only a year-on-year growth rate where a level is
needed. Spain is the entry about a national office rather than a mirror:
neither INE nor the Banco de España publishes a machine-readable vintage
archive for quarterly GDP, so `oecd_stes` is the historical source.

Mexico is the case worth stating carefully, because it is two answers
rather than one. Neither Banxico's SIE nor INEGI's BIE retains
previously published editions — both overwrite in place — so the
`banxico` and `inegi` connectors are snapshot connectors: they give
today's edition plus whatever history your own captures have
accumulated. For *historical* Mexican editions the OECD archive remains
the source, and `providers_for("MEX", "gdp_real")` accordingly returns
`['alfred', 'inegi', 'oecd_stes']`. The same split applies to Brazil
(`bcb`) and Chile (`bcch`).
