# Main Street Uncertainty: Five Decades of Beige Book Labor-Market Uncertainty and State Labor Outcomes

> **Status (2026-07-23): corpus extended to 1970 and all phases re-run.**
> The BBUI corpus now reaches back through the pre-1983 "Redbook"
> predecessor to 1970Q2 (482 releases, 30,449 records, 223 quarters /
> 2,676 district-quarter cells). The state-outcome merge starts 1976Q1
> (LAUS availability), so the descriptive LP is 1976–2025; the
> exposure-design phases still estimate from 1992Q1 (frozen 1990–91 base),
> so they shift only through the re-standardized shock. **Border results
> are reported seasonally adjusted (X-13) only.** The compiled manuscript
> `docs/paper/main_street_uncertainty/` carries the fully propagated
> numbers and is the current source of truth; headline figures in this
> skeleton's abstract are updated below, but some in-body numbers may lag.

*Draft skeleton — Phases 1–3 (2026-07-20), Phase 4 (2026-07-21). Every
number below is produced by `tools/build_bbui_district_panel.py` (corpus +
index), `tools/run_main_street_lp.py` (Phase-2 merge + pooled LPs),
`tools/run_main_street_phase3.py` (exposure design, WARN/MLS outcomes,
placebos), `tools/run_main_street_phase4.py` (leave-own-district-out horse
race; frozen-input deterministic, replication gate 1e-16 against Phase 3)
`tools/run_main_street_phase4_border.py` (split-state border contrasts
on county LAUS, county→district crosswalk built by
`tools/build_fed_county_crosswalk.py` from the Reserve Banks' own
published county lists; `--sa` re-runs on X-13ARIMA-SEATS-adjusted
county U/E) and `tools/run_main_street_phase5_realtime.py` (ALFRED
initial-release outcomes, Phase 5 2026-07-21). Nothing is hand-computed;
nothing is quoted from the literature as if it were ours.*

## Abstract

We build a quarterly panel of *labor-market uncertainty as the Federal
Reserve's own regional narrative records it*: a sentence-cooccurrence
index (BBUI) of labor terms and uncertainty tone, parsed from every
Beige Book release on federalreserve.gov and its pre-1983 Redbook
predecessor — 482 releases, 30,449
(release, district, section) records, covering **all 12 Federal Reserve
districts in every quarter from 1970Q2 to 2025Q4** (2,676
district-quarter cells, no gaps). Merged with state labor outcomes
(from 1976), the
descriptive fact is a slow-building association: after a one-standard-
deviation district BBUI innovation, member-state unemployment rises
about **0.06 pp over 2–3 years** relative to other districts (two-way
FE panel LP, Driscoll-Kraay SEs; the 90% sup-t band excludes zero at 6
of 13 horizons), while per-district IRFs are 5–10 times larger —
most of the raw uncertainty–unemployment correlation is national, not
district-idiosyncratic. We then push toward identification with an
exposure-differential design: interacting the district shock with each
state's *pre-determined 1990–91 manufacturing employment share*, under
state and **district-by-quarter** fixed effects that absorb the district
shock itself. High-manufacturing states do lose more: the differential
peaks at **+0.041 pp** of unemployment per (1 s.d. exposure) × (1 s.d.
shock) at nine quarters (Driscoll-Kraay 90% CI [0.010, 0.073]) with the
right-signed employment mirror (−0.12% at h = 12), and the path is
stable to dropping any of the 12 districts (jackknife range
[+0.032, +0.045]). But we report the fragility as prominently as the
point estimate: wild-cluster bootstrap-t inference over the 12 districts
(Cameron-Gelbach-Miller 2008) yields p = 0.14 at the peak (p < 0.10 only
at h = 10–12); a shuffle-district placebo retains about half the signal
(mean +0.019, randomization p = 0.10) because district uncertainty
co-moves across districts (mean pairwise shock correlation +0.10); and
the h = −2 lead is nonzero (−0.015, wild-cluster p = 0.04), consistent
with Beige Book language partly *responding to* recent relative
deterioration. Phase 4 turns those two symptoms into a test, and the
verdict reframes the paper. In a horse race against the
leave-own-district-out national innovation (the mean of the other eleven
districts' shocks, interacted with the same exposure), the own-district
differential at the frozen peak drops from +0.039 to +0.022 — 56%,
surviving our pre-registered half-magnitude rule, barely (wild-cluster
p = 0.12 at h = 9; its new peak is earlier and sharper, +0.025 at h = 5,
p = 0.048) — while the leave-one-out *national* interaction is the most
robust coefficient in the paper (+0.035 at h = 12, wild-cluster
p = 0.007). With the national control in place the shuffle placebo
collapses from +0.019 to exactly +0.000 (same 200 derangements, paired)
and the h = −2 lead loses significance (wild-cluster p 0.04 → 0.13). A
second, sharper design uses the 14 states that straddle two districts:
with county LAUS outcomes and state-by-quarter fixed effects — every
state-level policy and demand shock absorbed — counties on the
high-uncertainty side of a within-state district boundary do *no worse*
than their same-state neighbors: peak estimates are −0.014 to −0.015 pp
(wrong-signed) in both the 1,009-county and 179-border-county samples,
no positive estimate exceeds +0.010 pp at any horizon, and the
state-level 2–3-year build-up is entirely absent. The honest synthesis: manufacturing-exposed states suffer when
*national* labor-market uncertainty rises; the incremental,
own-district-specific component of Beige Book uncertainty language moves
state outcomes weakly at best, and county outcomes not at all. A final
sharpening: with the full 11-supersector 1990-91 exposure vector in
place of the single manufacturing share, composition exposure matters
jointly at every horizon (Driscoll-Kraay Wald p ≤ 0.001, h = 0-11) but
no single division — manufacturing included — is individually
separable, so "manufacturing exposure" throughout this paper should be
read as the manufacturing loading of a broader industrial-composition
gradient. Administrative mass-layoff margins show no positive
differential (BLS MLS events 1998–2013: −0.018 log points at h = 1,
wild-cluster p = 0.05, wrong-signed; WARN notices are too thinly covered
for the design — 17 fully-covered states spanning only 6 districts with
two or more states). A mining-share exposure yields a robustly
*negative* unemployment differential (−0.038 pp at h = 8, wild-cluster
p = 0.03 at every horizon) — energy-state cycles run against their
districts' uncertainty narrative. The corpus is, we argue, the paper's
durable contribution; the exposure evidence is suggestive, honestly
fragile, and quantifies exactly how far 12 clusters of narrative
uncertainty can take identification.

## 1. Introduction

Two literatures meet here. The uncertainty literature measures economic
uncertainty from newspapers (Baker, Bloom and Davis 2016, *QJE*),
forecast-error dispersion (Jurado, Ludvigson and Ng 2015, *AER*), or
market volatility (Bloom 2009, *Econometrica*), and asks what activity
does after uncertainty rises — overwhelmingly at the national level.
A smaller regional branch asks the subnational question: Baker, Bloom
and Davis's state EPU indices are newspaper-based and available for
recent decades; Mumtaz, Sunder-Plassmann and Theophilopoulou (2018,
*JMCB*) estimate state-level effects of uncertainty shocks and find
substantial heterogeneity. On the other side, the regional-cycle
literature (e.g., Hamilton and Owyang 2012, *REStat*) documents that US
states share national cycles imperfectly, and the granularity
literature (Gabaix 2011, *Econometrica*) insists that regional or
sectoral composition turns aggregate shocks into cross-sectional
variation — the logic that shift-share designs exploit
(Goldsmith-Pinkham, Sorkin and Swift 2020, *AER*; Borusyak, Hull and
Jaravel 2022, *REStud*).

Our measurement contribution sits before either question: the Beige
Book is the only continuously-published, institutionally-stable,
*regionally-disaggregated* narrative record of US economic conditions,
written eight times a year since 1970 by the twelve Reserve Banks and
digitally available from 1983. Prior work has used it mainly as a
national sentiment signal (e.g., Armesto, Hernández-Murillo, Owyang and
Piger 2009, *JMCB*, on its information content). We read it as a
*district-level uncertainty panel*: the fraction of a district's
sentences that pair a labor-market term with an uncertainty-tone term,
release by release, section by section. Phase 1–2 engineering (four
distinct archive layouts, PDF and HTML parsers, documented in the
pipeline) yields a panel with **no missing district-quarters for 42
years** — to our knowledge the longest subnational uncertainty series
for the US by roughly two decades (state EPU starts in the mid-1980s
for a few states and is newspaper-dependent; the WUI's subnational
analogues do not exist for US states).

The identification contribution is deliberately modest and honestly
reported. District uncertainty is not randomly assigned; deteriorating
regions plausibly generate uncertain language first. We therefore (i)
purge forecastable persistence (AR(2) innovations, z-scored within
district), (ii) absorb the national component with time fixed effects
(Phase 2), and (iii) in this phase absorb *the district shock itself*
with district-by-quarter fixed effects, identifying only the
*differential* response of member states with higher pre-determined
manufacturing exposure — a Bartik-discipline interaction with 1990–91
shares, estimated from 1992Q1 onward so the base period strictly
precedes the sample. Every headline is accompanied by the three
diagnostics that few-cluster narrative designs owe the reader:
wild-cluster bootstrap-t p-values over the 12 districts, a
shuffle-district placebo read as randomization inference, and lead
outcomes.

## 2. Data

**The BBUI district panel** (`data/processed/bbui_district_panel.csv`;
coverage table `output/bbui_coverage_by_decade.csv`). 1983Q3–2025Q4,
170 quarters × 12 districts, every cell backed by parsed documents:
27,684 (release, district, section) records from 342 releases, 7.9
records per cell in the 1980s rising to 17.7 in the 2020s. The index is
the share of sentences co-occurring a labor term and an uncertainty
term, averaged over the district's canonical sections and the (usually
two) releases per quarter. Era heterogeneity is documented: 1983–1995
comes from PDF text extraction and is noisier (sd 0.043 vs ~0.02–0.03
later); within-district z-scoring of the shock removes level drift but
not era-specific measurement noise.

**State outcomes** (`output/state_labor_quarterly.csv`). Monthly SA
state unemployment rates (LAUS) and nonfarm employment (CES), fetched
key-free from the FRED fredgraph CSV mirrors (`{ST}UR`, `{ST}NA`),
averaged to quarters; 50 states + DC; current vintage. Employment
starts 1990.

**Exposure** (`output/exposure_state.csv`). Manufacturing share =
state CES manufacturing employment (`{ST}MFG`; DC exists only NSA as
`DCMFGN`) over total nonfarm, averaged over 1990Q1–1991Q4 — the
earliest quarters the CES state mirrors exist — and frozen. Range:
0.010 (DC) to 0.261 (NC). Mining/energy share (`{ST}NRMN`, 45 states;
robustness) tops out at 0.082 (WY). Both are z-scored across states.

**Mass-layoff outcomes.** (i) BLS Mass Layoff Statistics state
layoff-event counts (administrative; program 1995–2013; quarterly sums
1998Q1–2013Q1 for all 51 units; months absent from the BLS flat files
are zero-filled — the smallest reported count is 3, so the error is
bounded and concentrated in small states). (ii) WARN notices from
state-DOL scrapers and archival collections
(`puremacro.narrative.sources.us_warn`): 47,622 filings across 45
states (`output/warn_state_quarterly.csv`), but coverage is
state-heterogeneous (`output/warn_coverage.csv`): only 17 states are
continuously covered over 2015Q1–2021Q4, spanning just 6 districts with
≥2 covered states — we run the design on it, label it fragile, and do
not lean on it.

**Crosswalk.** State → primary Fed district (population-majority for
the 14 split states). Split-state assignment error attenuates.

## 3. The exposure-differential design

For state *s* in district *d(s)*:

    y_{s,t+h} − y_{s,t−1} = β_h (expo_s × shock_{d(s),t})
                            + α_s + λ_{d(s),t}
                            + Σ_{k=1..4} [γ_k (expo_s × shock_{d(s),t−k})
                                          + δ_k y_{s,t−k}] + ε_{s,t+h}

The district-by-quarter fixed effects λ_{d,t} absorb the district shock
itself and *any* district-level disturbance that quarter; β_h is
identified purely off within-district, cross-state differences in
pre-determined exposure. This kills the Phase-2 objection that district
uncertainty language responds to district conditions — any such reverse
causality common to the district's states is absorbed. What survives as
a threat is *state-specific* reverse causality correlated with
exposure: the Beige Book writer worrying specifically about the
high-manufacturing member states because they are already
deteriorating. The lead-outcome diagnostic below speaks to exactly
that, and is not entirely clean — we flag it rather than hide it.

**Inference, side by side at every horizon** (all in
`output/irf_exposure.csv`): Driscoll-Kraay (1998, *REStat*) HAC SEs on
the within-projected design (house bandwidth, as in
`puremacro.lp.panel_dk`); and wild-cluster bootstrap-t p-values with
Rademacher weights at the district level, null imposed, B = 999
(Cameron, Gelbach and Miller 2008, *REStat*), re-applying the FE
projection to every bootstrap outcome. Twelve clusters is few; we cite
the MacKinnon-Webb (2017, *JAE*) caveat rather than claim to solve it,
and add randomization-flavored evidence: a placebo that reassigns every
state the shock of a wrong district (derangements of the 12 labels; the
district-quarter partition is permutation-invariant, so only the
regressor moves).

## 4. Results

**Phase-2 baseline (relative, two-way FE)** (`output/irf_pooled.csv`,
`output/fig_pooled_irf.pdf`). A one-s.d. district BBUI innovation is
followed by a relative member-state unemployment increase of 0.026 pp
at h = 4, 0.042 pp at h = 8, peaking 0.042 pp at h = 11; the 90% sup-t
band (Montiel Olea and Plagborg-Møller 2019, *JAE*; 300
district-cluster bootstrap draws) excludes zero at 8 of 13 horizons.
Employment mirrors with a lag (−0.03% at h = 8) but never clears the
sup-t band. Per-district IRFs
(`output/fig_district_irf_grid.pdf`) are 5–10× the pooled path (New
York +0.35 pp at h = 8, St. Louis +0.26, Dallas −0.09): most of the raw
correlation is the national component that the time FE absorb.

**The exposure differential** (`output/fig_exposure_irf.pdf`; Figure 1).
Manufacturing exposure × shock on unemployment builds from ~0 on impact
to **+0.039 pp at h = 9** (DK se 0.018, 90% CI [0.008, 0.069];
n = 6,273 state-quarters, 51 states, 1992Q1+), settling at +0.025–0.028
pp over h = 10–12. Wild-cluster p-values tell the honest version: 0.141
at the peak, below 0.10 only at h = 10–12. Employment gives the mirror
image — a slow decline reaching **−0.107%** at h = 12 (DK se 0.039, WC
p = 0.146, never below 0.10). Signs, shapes and timing all match the
Phase-2 pooled path; the differential magnitude at the peak is
comparable to the pooled relative effect, meaning a one-s.d.-more-
manufacturing state bears roughly *double* the average relative
response.

**Mass-layoff margins do not confirm** (Figure 1, right panel). On BLS
MLS layoff events (1998Q1–2013Q1, all 51 units) the differential is
never positive at any horizon; its extreme value is **−0.018 log
points at h = 1** (WC p = 0.054) — wrong-signed for an
uncertainty-driven layoff story and consistent with mass-layoff *events*
(≥50 workers) being dominated by plant-level idiosyncrasies. The WARN
exposure LP (17 states, 2015Q1–2021Q4, 6 effective districts) is a
noisy null (h = 0: −0.21, WC p = 0.30) and is labeled FRAGILE in every
output; coverage, not economics, is the binding constraint
(`output/warn_coverage.csv`).

**The mining anomaly** (`irf_exposure.csv`, spec `urate_mining`).
Mining-share exposure produces a flat, *negative*, wild-cluster-robust
unemployment differential: −0.022 to −0.038 pp at every horizon (h = 8:
−0.038, DK se 0.010, WC p = 0.026; p < 0.10 at all 13 horizons). The
Phase-2 Dallas suspicion generalizes: energy-state labor markets
systematically *improve* relative to their districts when district
labor-uncertainty language spikes — consistent with oil-price shocks
that raise both district-wide uncertainty talk and energy-state
activity. This is a warning against reading any single-exposure
interaction as "the" uncertainty effect.

**Placebos and robustness** (`output/fig_phase3_robustness.pdf`;
Figure 2). (i) *Shuffle-district placebo*: across 200 derangements the
placebo β at h = 9 averages +0.019 (sd 0.014) — half the true +0.039,
not zero, exactly as the cross-district shock correlation (+0.10 mean
pairwise) implies; the randomization p is 0.10. The design does not
distinguish "own-district uncertainty" from "correlated national
uncertainty loading on manufacturing states" as sharply as we would
like, and we say so. (ii) *Leads*: h = −4/−3 are small and insignificant
(−0.019/−0.014, WC p ≈ 0.26), but h = −2 is −0.015 with WC p = 0.04 —
high-exposure states were already deteriorating slightly in the two
quarters before the shock. Signed magnitude is ~40% of the peak; it
does not overturn the post-shock build-up but it does mean the strict
exogeneity reading is not available. MLS leads are clean (p ≥ 0.18).
(iii) *Jackknife*: dropping any one district leaves β(h = 9) in
[+0.032, +0.045], always positive; San Francisco is the most
influential single district.

## 5. Phase 4: what exactly is "district" about the differential?

Phase 3 left two symptoms on the table — the half-alive shuffle placebo
and the h = −2 lead — and both pointed the same way: district BBUI
innovations share a national component (mean pairwise correlation +0.10;
correlation between a district's innovation and the mean of the other
eleven, +0.30). Phase 4 makes that component an explicit regressor and
then removes the state dimension entirely.

**The leave-own-district-out horse race**
(`tools/run_main_street_phase4.py`; `output/fig_phase4_loo.pdf`). Define
loo\_{d,t} as the mean of the other eleven districts' AR(2)-purged
innovations (purge first, then average; z-scored on the 1992Q1+ cells,
s.d. 0.453 before scaling). The tool re-estimates the Phase-3 design
importing the Phase-3 estimator and ASSERTS reproduction of the frozen
`irf_exposure.csv` to 1e-16 before running anything new; the horse race
then adds expo × loo (and its four lags) to the baseline expo × own
specification. Three results. (i) At the frozen Phase-3 peak (h = 9) the
own-district coefficient falls from +0.0385 (DK se 0.018) to +0.0217 (DK
se 0.015, wild-cluster p = 0.12) — a ratio of 0.56 against our
pre-registered survives-at-half rule, recorded in the batch-4 plan before
the regression was run. Its peak relocates to h = 5 (+0.0252, wild-cluster
p = 0.048), the only wild-cluster-significant own-district estimate in the
paper. (ii) The national interaction is stronger and more robust than the
own one wherever they compete: alone, expo × loo peaks at +0.0349 at
h = 12 (wild-cluster p = 0.007); in the horse race it keeps +0.0337
(p = 0.031). (iii) The employment mirror thins similarly (−0.107 → −0.062
at its horse-race peak, p = 0.28). The decomposition of the Phase-3
differential is therefore roughly half own-district, half correlated
national uncertainty — with only the national half robust at conventional
levels.

**Placebo and lead closure.** Re-running the Phase-3 derangement placebo
*with the LOO control held at its true value* — same seed, hence the same
200 derangements, a paired comparison — collapses the placebo mean from
+0.0187 to +0.0000: the half-alive placebo was exactly the mechanical
national-component contamination the LOO term removes, which validates
reading the horse race as that decomposition. The h = −2 lead keeps its
magnitude (−0.0142) but loses significance under the LOO control
(wild-cluster p 0.04 → 0.13); leads at h = −4/−3 shrink by ~40% (p ≥
0.47). Exposed states' pre-shock deterioration loads mostly on the
national component, not on what the own district's Beige Book is about
to say.

**Split-state border contrasts**
(`tools/run_main_street_phase4_border.py`;
`output/fig_phase4_border.pdf`). Fourteen states straddle two districts.
The package's state→district crosswalk is state-level, so we built the
county-level assignment from the Reserve Banks' own published county
lists (St. Louis 8dmap/FRED categories; Cleveland via the Board's 1998
county-by-county district description on FRASER; Minneapolis, Dallas and
New York from their official territory pages; every source file with
URLs and transcription notes ships in `output/crosswalk_sources/`, and
`tools/build_fed_county_crosswalk.py` hard-fails on any unmatched county
name — 1,009 counties, 443 listed + 566 complement, every split matching
the banks' published counts, Kentucky partitioning exactly 64 + 56 with
neither gap nor overlap). County unemployment is LAUS-derived
(U/(U+E), NSA, quarterly means; all 1,009 counties have usable series
from 1990). The design regresses the county long-difference on the
district shock under county and state-× -quarter fixed effects — within
a state-quarter, only the boundary contrast identifies — with 4 lags,
district-level wild cluster (G = 11; San Francisco has no split state)
and Driscoll-Kraay side by side. One estimand caveat is mechanical: the
leave-one-out national difference across a within-state boundary equals
−1/11 of the own-shock difference, so no horse race is possible here —
the border design estimates own-relative-to-neighbor-district
uncertainty, which is exactly the component the horse race left in
doubt. The result is a null with the wrong sign at every horizon that
matters for the uncertainty story. In the all-counties sample the path
oscillates within ±0.014 pp and no h ≥ 6 estimate approaches
significance (peak −0.0143 at h = 12, wild-cluster p = 0.26); in the
border-counties-only sample (179 counties with a same-state
Census-adjacent neighbor in the other district) every h ≥ 6 estimate is
*negative*, two of them nominally significant (−0.0149 at h = 8,
p = 0.085; −0.0111 at h = 12, p = 0.015) — though both dissolve under
X-13 seasonal adjustment (§ 6), so we do not lean on them. No positive
estimate exceeds +0.010 pp anywhere. Two isolated short-horizon coefficients in the all-counties
sample are wild-cluster significant with opposite signs (−0.006 at
h = 0, +0.008 at h = 2) — quarter-scale sign flips of this size under
NSA county outcomes and G = 11 clusters read as seasonal measurement
noise, not economics, and we say so rather than cherry-pick either one.
Leads are flat (p ≥ 0.16). Counties whose district's Beige Book
uncertainty language spikes do not shed jobs relative to same-state
neighbors across the district line — the 2–3-year unemployment build-up
that defines the state-level result never appears.

**Synthesis.** The three Phase-4 facts are mutually consistent: exposed
states respond robustly to national narrative uncertainty; the
own-district increment survives the state-level horse race only at half
strength and marginal significance; and at the county level, where the
design is sharpest, the own-district increment is absent. The paper's
framing changes accordingly — from "district uncertainty hurts exposed
member states" to "the Beige Book corpus prices national labor-market
uncertainty into every district's language, and exposed states bear it;
the district-idiosyncratic residual carries little additional signal for
outcomes."

## 6. Data-quality stress tests (Phase 5)

**Outcomes as first published (ALFRED).**
(`tools/run_main_street_phase5_realtime.py`;
`output/fig_phase5_realtime.pdf`, `irf_realtime.csv`,
`realtime_coverage.csv`.) Beige Book text is never revised, so only the
outcome side needs vintages. The ALFRED initial-release archive
(``output_type=4``; observations kept only when first published within
120 days of the reference month — publication lags run 47–66 days, so
essentially everything qualifies) supports state LAUS/CES outcomes from
roughly 2005–2007 onward; the comparison window is 2005Q3–2025Q2 (3,779
state-quarters, 51 states, states entering as their archives begin).
Three estimates per anchor separate the two confounds: the frozen
full-sample Phase-4 number, the current-vintage estimate on the matched
window (window effect), and the first-release estimate (vintage
effect). Long differences use first-release values at both ends
(Croushore-Stark first-release diagonals). The verdict splits cleanly.
*Magnitudes survive first-release data essentially intact*: at the h = 5
own-district anchor, +0.062 (current vintage, matched window) →
+0.055 first-release (89%); at h = 9, +0.062 → +0.064 (104%); the
LOO-national h = 12 anchor keeps 87% (+0.041 → +0.036) and is the one
estimate whose wild-cluster inference *strengthens* on first-release
data (p = 0.003). Benchmark revisions do not manufacture the paper's
results. Two honest qualifications come with that. First, the
own-district differential's significance does not survive the noisier
unrevised outcomes (WC p = 0.18–0.19 at both anchors, from 0.03–0.09
current-vintage). Second, the matched-window comparison exposes a large
*window* effect: on the 2005+ era — dominated by the GFC and COVID —
the own-district differential is roughly three times its full-sample
size (+0.062 vs +0.025 at h = 5), i.e. the state-level differential is
concentrated in the crisis era, a sample-dependence the full-sample
numbers average away. First-release paths are also visibly more
volatile: the first-release LOO path swings negative at h = 6 (−0.047,
WC p = 0.06) before its strongly positive h = 12 — we read this as
first-release measurement noise, and flag rather than interpret it.

**X-13 seasonal adjustment of the county outcomes.**
(`tools/run_main_street_phase4_border.py --sa`;
`output/irf_border_pairs_sa.csv`, `fig_phase4_border_sa.pdf`,
`county_ue_monthly_sa.csv.gz`.) The border design ran on NSA county
LAUS, and § 5 flagged its two isolated short-horizon significants as
probable seasonal noise. We now adjust every county's monthly U and E
levels with genuine X-13ARIMA-SEATS (v1.1.57; airline model, automatic
log/level, no outlier regressors; 2,017 of 2,018 series adjusted by
the real binary, one STL fallback; the tool hard-fails if more than 5%
of series fall back, so "X-13 adjusted" cannot silently mean "STL
adjusted") and re-run both border samples at B = 999. The verdict is
cleaner than the flag: under seasonal adjustment **no horizon in
either sample is wild-cluster significant at 10%** — the h = 0 and
h = 2 short-horizon coefficients collapse (WC p 0.019 → 0.49 and
0.004 → 0.39), and the NSA border-only medium-run negatives dissolve
as well (h = 8: −0.0149, p = 0.085 → −0.0087, p = 0.34; h = 12:
−0.0111, p = 0.015 → −0.0033, p = 0.35). Peaks shrink toward zero
(all-counties −0.0143 → −0.0087; border-only −0.0149 → −0.0133,
p = 0.11) and no positive estimate exceeds +0.007 pp anywhere. Every
wild-cluster-significant coefficient the NSA border regressions
produced — of either sign — was a seasonal artifact; the border
contrast's honest summary is a complete null, which is precisely what
§ 5's synthesis requires of it.

## 7. The full exposure vector (Phase 6)

Phases 3-5 froze a SINGLE 1990-91 industry share and flagged the full
shift-share vector as future work. No keyed source was needed: the
same key-free CES state mirrors carry the entire 11-supersector NAICS
partition from 1990-01 ({ST}CONS, {ST}TRAD, {ST}INFO, {ST}FIRE,
{ST}PBSV, {ST}EDUH, {ST}LEIH, {ST}SRVO, {ST}GOVT alongside the
familiar {ST}MFG and {ST}NRMN; SA-then-NSA fallback, verified live).
`tools/run_main_street_phase6.py` freezes the 51-state share table
(`ces_supersector_shares_9091.csv`), reproduces the phase-3
manufacturing share to 9e-17 as a gate, and re-runs the phase-4
design with TEN z-scored share interactions jointly (other services
omitted as the reference division; 45 states survive full supersector
coverage; lags of every interaction per the house convention).

The verdict sharpens the paper's reading once more. Jointly, exposure
matters overwhelmingly: the Driscoll-Kraay Wald that all ten
own-district interaction coefficients are zero rejects at p <= 0.001
for h = 0-11 (p = 0.066 at h = 12). Individually, NOTHING is
separable: manufacturing's differential at the h = 5 anchor is +0.12
per (s.d. share x s.d. shock) with a DK standard error of 0.16 —
nine times the phase-3 standard error, the price of ten correlated
share interactions — and a wild-cluster p of 0.43; across all 260
(model x division x horizon) cells exactly two clear WC 10%
(financial activities at h = 6, both models), fewer than the 26
chance would predict, and we read them as noise. The phase-3/4/5
"manufacturing exposure" result should therefore be understood as
industrial-composition exposure summarized by its manufacturing
loading — the single-share design detects the composition gradient
efficiently but cannot, and now demonstrably does not, isolate a
manufacturing-specific mechanism. Honest caveats: the ten z-scored
shares are near-collinear by construction (they nearly sum to a
constant), so individual attribution was always going to be
underpowered at G = 12; and the joint Wald leans on DK asymptotics
with df = 10, which few-cluster caution applies to as well.

## 8. Limitations

Twelve clusters is the binding statistical constraint: the shock varies
at the district level, DK and cluster asymptotics are questionable at
G = 12, and our wild-cluster p-values — the honest ones — do not clear
0.10 at the unemployment peak. The Beige Book is an edited, filtered
narrative; measurement error is era-heterogeneous (PDF-era noise) and
attenuates. Outcomes are current-vintage LAUS/CES, not real-time.
Exposure is a single industry share fixed at 1990–91 (the CES state
floor), not a full shift-share vector, and 1990–91 is *early-sample-*
not *pre-sample-*fixed relative to the 1983 start of the corpus (the
estimation window starts 1992Q1 for exactly that reason). The h = −2
lead and the half-alive placebo mean the exposure differential should
be read as a sharpened correlation with a defensible absorbing
structure — not a natural experiment. WARN coverage is an honest data
gap: the scraped record supports a national event count from ~2014 but
not a 12-district exposure design. Phase 4 resolved the two top items of
the old candidate list (leave-own-district-out control, border
contrasts) and added its own caveats: the LOO shock is
purge-first-then-average (a documented construction choice); county
LAUS urate is NSA, so county-idiosyncratic seasonality survives the
state-×-quarter absorption (regressions are unweighted, and small-county
measurement noise attenuates toward zero — a bias *against* the border
null being informative, which we accept because the all-counties and
border-only samples agree); and the border design's estimand is
own-relative-to-neighbor uncertainty, not the level effect, by
construction. Phase 5 resolved the vintage question (§ 6): magnitudes
survive first-release outcomes; own-district *significance* does not,
and the matched-window comparison revealed that the state-level
differential is roughly 3× larger on the 2005+ crisis-era sample than
over the full 1992+ window — sample-era dependence is now a documented
caveat in its own right, not a suspicion. Phase 6 resolved the
shift-share item (§ 7): composition exposure matters jointly at every
horizon but no single division — manufacturing included — is
separable, so the paper's exposure language is now
"industrial-composition gradient," not a manufacturing mechanism.
Remaining candidates: a pre-1983 archival extension of the corpus
(which would also dilute the crisis-era concentration question), and
sharper exposure instruments that break the share collinearity
(e.g. QCEW county-industry detail via the available BEA/BLS keys).

## References

*(Method-level citations live in the module and pipeline docstrings;
positioning literature cited in the text: Bloom 2009; Baker, Bloom and
Davis 2016; Jurado, Ludvigson and Ng 2015; Mumtaz, Sunder-Plassmann and
Theophilopoulou 2018; Hamilton and Owyang 2012; Gabaix 2011;
Goldsmith-Pinkham, Sorkin and Swift 2020; Borusyak, Hull and Jaravel
2022; Armesto, Hernández-Murillo, Owyang and Piger 2009; Jordà 2005;
Driscoll and Kraay 1998; Cameron, Gelbach and Miller 2008; MacKinnon
and Webb 2017; Montiel Olea and Plagborg-Møller 2019.)*
