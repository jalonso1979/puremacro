# Main Street Uncertainty — Phase 2 Findings

> **Status (2026-07-21): Phases 3 AND 4 are done — results and framing
> live in [`DRAFT.md`](DRAFT.md). Phase 4 (leave-own-district-out horse
> race, paired placebo closure, split-state border contrasts on county
> LAUS with a bank-sourced county→district crosswalk) reframed the
> headline: the robust signal is *national* uncertainty × exposure; the
> own-district increment survives at half strength at the state level
> and is absent at the county boundary. Tools:
> `tools/run_main_street_phase4.py`,
> `tools/run_main_street_phase4_border.py`,
> `tools/build_fed_county_crosswalk.py` (outputs in `output/`:
> `irf_loo_horserace.csv`, `placebo_shuffle_loo.csv`,
> `county_district_crosswalk.csv`, `irf_border_pairs.csv`,
> `phase4_summary.json`, `phase4b_summary.json`, …). This file remains
> the Phase-2 record; where the two disagree, `DRAFT.md` supersedes.
> Plans: `docs/plans/2026-07-20-main-street-phase3-plan.md`,
> `docs/plans/2026-07-21-puremacro-batch4-plan.md`.

*2026-07-19. Every number below is produced by
`tools/build_bbui_district_panel.py` (panel) and
`tools/run_main_street_lp.py` (merge + LPs; seed-deterministic, outputs
in `output/`). Nothing is hand-computed. Phase 1 shipped the district
crosswalk, `bbui(level='district')`, and the federalreserve.gov
connector; this phase extends the panel to its full digital history and
runs the first state-level local projections.*

## 1. Full-history district BBUI panel (1983Q3–2025Q4)

`data/processed/bbui_district_panel.csv` now covers **170 consecutive
quarters × 12 districts = 2,040 district-quarter cells, every cell
backed by parsed documents** (plus the National summary series). The
index is the raw LUI sentence-cooccurrence score — the fraction of a
district's Beige Book sentences that pair a labor-market term with an
uncertainty-tone term — averaged over the district's canonical sections
and the (usually two) releases per quarter.

Coverage by decade (`output/bbui_coverage_by_decade.csv`):

| decade | span | quarters covered | quarters with all 12 districts | district-quarter cells | section-records parsed | mean records/cell |
|---|---|---|---|---|---|---|
| 1980s | 1983Q3–1989Q4 | 26/26 | 26 | 312 | 2,471 | 7.9 |
| 1990s | 1990Q1–1999Q4 | 40/40 | 40 | 480 | 4,749 | 9.9 |
| 2000s | 2000Q1–2009Q4 | 40/40 | 40 | 480 | 6,491 | 13.5 |
| 2010s | 2010Q1–2019Q4 | 40/40 | 40 | 480 | 6,830 | 14.2 |
| 2020s | 2020Q1–2025Q4 | 24/24 | 24 | 288 | 5,097 | 17.7 |

Total: 27,684 (release, district, section) records across 342 releases.
**There are no quarterly gaps**: every quarter from 1983Q3 to 2025Q4 has
all 12 districts.

What made this possible: the live federalreserve.gov archive serves four
distinct layouts, two of which the Phase-1 connector did not reach. The
Phase-1 "pre-2017 URL enumeration" (`beigebook{yyyymm}-{slug}.htm`)
turned out not to exist on the live site — those requests 404 — so
1996–2016 initially parsed **zero** records. Two additive fallbacks
fixed this (`puremacro/narrative/sources/beige_book.py`; all 31 existing
district tests plus the 36 connector tests stay green):

* **(d) ~2011–2016**: `/monetarypolicy/beigebook/beigebook{yyyymm}.htm`
  — full report on one page, `<h2>` ordinal district headers
  ("First District--Boston"), `<p><strong>` section leaders. Parsed with
  a DOM walk; pre-2011 the same URL serves a TOC stub, so the parse is
  only accepted when it is substantive (≥6 districts, ≥5k chars).
* **(e) 1996–2010**: `/fomc/beigebook/{year}/{yyyymmdd}/FullReport.htm`,
  enumerated from the per-year archive index pages *and* the FOMC
  historical materials pages (the two listings are inconsistently
  maintained — the 2003 year index omits the September 3, 2003 release
  that `fomchistorical2003.htm` lists). These pages have unclosed-tag
  table markup that defeats a DOM walk, so the *linear text* is parsed
  with the same line state machine as the 1983–1995 FOMC PDFs.
* Months found under neither (notably January–September 1996, PDF-only)
  fall back to the FOMC-historical PDF backend.

Pre-1983 issues are not digitally available on federalreserve.gov (the
connector's documented floor), so 1983Q3 is the hard start of the panel.

Era-heterogeneity note: mean index levels are broadly comparable across
source eras (0.030–0.045) but the 1983–1995 PDF era is noisier
(sd 0.043 vs ~0.02–0.03 later) — layout-driven text extraction is
cleanest post-1996. The LP shock construction below (AR(2) purge +
within-district z-scoring) removes slow level drift, but era-specific
measurement noise remains a caveat.

## 2. State labor panel and merge

* **Outcomes** (`output/state_labor_quarterly.csv`): monthly SA state
  unemployment rate (FRED `{ST}UR`, BLS LAUS mirror; 1976Q1+) and total
  nonfarm employment (FRED `{ST}NA`, BLS CES mirror; 1990Q1+), fetched
  key-free via the public fredgraph CSV endpoint
  (`puremacro.fetch.bls_state_panel`), averaged to quarters. 50 states
  + DC. No API keys are configured on this machine
  (`puremacro.credentials.status()` shows all missing) — nothing here
  needs one. Current-vintage series, not real-time vintages.
* **Crosswalk**: state → primary Fed district via `STATE_TO_DISTRICT`
  (population-majority rule for the 14 split states; e.g. MO →
  St. Louis). Split-state assignment error is a known source of
  attenuation — the Bartik-style refinement is Phase 3.
* **Merged LP input** (`output/state_district_panel_quarterly.csv`):
  51 states × 170 quarters = 8,670 state-quarter rows, 1983Q3–2025Q4
  (employment enters 1990Q1+), with the district BBUI level and its
  innovation attached to every member state.

## 3. First local projections

**Shock.** Within-district AR(2) innovation of the quarterly district
BBUI, z-scored within district
(`puremacro.uncertainty.lp_long.derive_innovation_shock`). Unit = one
district-specific standard deviation of unexpected Beige Book labor
uncertainty.

**Estimator.** Pooled panel Jordà LP (horizons 0–12 quarters), LHS
`y_{t+h} − y_{t-1}`, 4 lags of shock and outcome,
**two-way (entity + time) fixed effects** with **Driscoll–Kraay (1998)
HAC standard errors** (`puremacro.lp.panel_dk.panel_lp_dk`). The time
fixed effects absorb the common national component of Beige Book
uncertainty each quarter, so the coefficients are identified purely off
**relative cross-district variation** — this is the honest object at
this stage, not a national uncertainty multiplier.

**Simultaneous band.** 90% **sup-t** band across the 13 horizons —
Montiel Olea & Plagborg-Møller (2019, *Journal of Applied
Econometrics* 34(1), 1–17), Algorithm 2, via
`puremacro.inference.supt.supt_band` — computed on 300 district-cluster
bootstrap draws (districts resampled with replacement; the district is
the level of the shock). All bands cited below are this sup-t band
unless labelled pointwise.

**Headline result (unemployment rate).** A one-s.d. relative district
uncertainty innovation is followed by a slow-building **relative
increase in member-state unemployment of ~0.04 pp, peaking 8–11
quarters out**:

| h | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| β (pp) | 0.001 | 0.018 | 0.017 | 0.027 | 0.034 | 0.042 | 0.038 | 0.037 |
| DK se | 0.009 | 0.005 | 0.007 | 0.009 | 0.014 | 0.016 | 0.014 | 0.014 |

The 90% sup-t band excludes zero at **8 of 13 horizons**
(h ∈ {1, 4, 5, 6, 7, 8, 10, 11}); the whole-path statement "the
unemployment response is positive over years 1–3" survives simultaneous
inference. n = 8,568 state-quarters.

**Employment.** 100·log nonfarm employment declines slowly (−0.03% at
h = 8, −0.055% at h = 12; pointwise DK bands exclude zero only at the
longest horizons and the **sup-t band never does**). Consistent in sign
with the unemployment result but not simultaneous-inference robust —
partly the shorter sample (1990+).

**Magnitudes are honest-small.** 0.04 pp of relative unemployment per
1 s.d. of relative narrative uncertainty is an economically modest
descriptive correlation. The Beige Book is itself a filtered, edited
narrative — attenuation from both measurement and the split-state
crosswalk pushes this toward zero.

**Per-district IRFs** (`fig_district_irf_grid`,
equal-weighted member-state unemployment, own-district shock,
time-series Jordà LP with HAC bandwidth h+1, T = 168 quarters each):
positive in 10 of 12 districts at h = 8, largest in New York
(+0.35 pp, se 0.09), St. Louis (+0.26, se 0.09), Chicago (+0.19),
essentially zero in Kansas City and negative in Dallas (−0.09, se 0.12
— energy-cycle composition is the obvious suspect, deferred to
Phase 3). These district-level paths are 5–10× the pooled two-way-FE
path because they include the national component of district
uncertainty that the pooled specification's time FE absorb —
the comparison itself is informative: **most of the raw
uncertainty–unemployment correlation is national, not district-
idiosyncratic.**

## 4. Caveats (read before quoting)

1. **No identification claims.** These are descriptive dynamic
   correlations. The AR(2) purge removes forecastable persistence, not
   endogeneity: districts whose economies are deteriorating for other
   reasons plausibly generate uncertain Beige Book language first.
2. **Relative, not aggregate.** Time FE absorb the national uncertainty
   component; general-equilibrium effects common to all districts are
   invisible here by construction.
3. **12 clusters.** The shock varies at the district level, so the
   bootstrap resamples 12 districts — small-cluster distortions are
   possible; wild-cluster refinements belong to Phase 3.
4. **Crosswalk noise.** 14 split states are assigned by population
   majority; district aggregates in the small-multiples figure weight
   member states equally.
5. **Measurement era-heterogeneity.** 1983–1995 comes from PDF text
   extraction (noisier); section attribution differs across layout
   eras. Within-district normalization mitigates level effects only.
6. **Current-vintage outcomes.** LAUS/CES state series are revised;
   real-time vintages are out of scope here.

## 5. Phase-3 plan

1. **Bartik exposure interaction.** Interact the district (and
   national) BBUI innovation with pre-period state industry shares
   (`puremacro.fetch.bea_industry_shares` / `state_industry_panel`) —
   shift-share exposure turns the national component into usable
   cross-state variation and addresses the Dallas/energy composition
   issue directly.
2. **WARN events.** The `puremacro.narrative.sources.us_warn` connector
   gives state-level mass-layoff notices — an outcome margin (advance
   notices respond faster than the unemployment stock) and a validation
   that BBUI innovations lead administrative distress measures.
3. **Identification.** Candidate designs: (i) granular-IV flavor —
   leave-own-district-out national BBUI as an instrument for district
   exposure; (ii) narrative timing within quarters (Beige Book
   collection windows end before most state data releases); (iii)
   split-state border-pair contrasts as a sharper within-labor-market
   comparison. Plus wild-cluster/randomization inference for the
   12-cluster problem.
