# How Much of the Uncertainty-Shock Literature Is Identification Choice?

*Draft skeleton — Phase 1 (2026-07-19); Phase 2 — narrative sign restrictions
as a tenth scheme and the Giacomini-Kitagawa identified-set overlay — merged
2026-07-19; Phase 3 — AD-RR Type II/III historical-decomposition dominance
restrictions (`--narrative-hd`) and the event-set sweep (`--event-sweep`) —
merged 2026-07-21 (all Phase-1/2 cells reproduce bit-for-bit; the flags are
additive). Every number below is produced by
`tools/run_uncertainty_ident_spec_curve.py` (full grid, seed-deterministic;
outputs in `output/`). Nothing is quoted from the literature as if it were
ours; nothing here is hand-computed.*

## Abstract

The negative response of real activity to an uncertainty shock is one of the
most replicated findings in empirical macroeconomics, yet nearly every paper
in the literature commits to a single identification scheme. We run eight
identification families — recursive (both orderings), sign restrictions
(traditional and, following Antolín-Díaz and Rubio-Ramírez 2018, sharpened
with narrative restrictions on four dated events), external instruments,
max-share, identification-through-heteroskedasticity (Rigobon and
Magnusson-Mavroeidis), non-Gaussian ICA, and lag-augmented local
projections — on the *same* reduced-form monthly US VAR (uncertainty proxy,
industrial production, employment, federal funds rate; 1954–2025 at the
longest), and then sweep proxy choice (EPU, WUI, JLN, VIX), sample window
(full, post-1985, pre-2020) and detrending (linear trend vs. first
differences) through a 288-cell specification curve. Identification choice
alone moves the peak response of industrial production to a one-standard-
deviation uncertainty shock from −3.3% to +1.4% on the identical baseline
dataset — a spread wider than most published point-estimate differences.
Across the 207 estimable specifications the median peak response is −2.1%
(bootstrap p-value for a zero median: < 0.001) and 95% of specifications
are negative — though the pooled median is composition-dependent by
construction: on the Phase-2 grid, before the two narrative-dominance
variants were added, it was −1.4% (p = 0.012). The qualitative finding is
robust; but the sign does *not* survive every scheme: every positive
estimate comes from statistical identifications (heteroskedasticity and
non-Gaussianity), and heteroskedasticity-identified responses on the
baseline dataset are positive (+1.4% at five months). Magnitudes are even
less stable than signs: family medians range from −0.9% (local
projections) to −8.7% (external instruments), and the external-instrument
median collapses from −9.7% to −2.0% once weak-instrument cells
(Olea-Pflueger F < 10) are set aside. Requiring the sign-identified shock
to have been *positive* in October 1987, September 2001, September 2008
and March 2020 moderates the sign family rather than amplifying it: the
narrative variant is less negative than plain sign restrictions in 10 of
11 matched cells and halves the family's spread (range 7.3 vs 13.2
percentage points; medians −3.2% vs −3.9%). Adding Antolín-Díaz and
Rubio-Ramírez *dominance* restrictions — the uncertainty shock was the
most important driver of the proxy itself in September 2008 and March 2020
(Type II), or overwhelming at COVID (Type III) — tightens the baseline
h = 12 band by a further 31–38% (Type II: [−3.66, −1.20] vs Type I's
[−4.13, −0.56]) and finally makes the importance weights informative: the
Kish effective sample size falls to 45–48% of the surviving-draw count,
against exactly 100% under Type-I restrictions alone. And the four event
months themselves turn out not to drive the result: across all nine
one-event and leave-one-out configurations the h = 12 band excludes zero,
with October 1987 — not COVID — the single most informative event (its
removal loosens the band most; dropping March 2020 slightly tightens it).
Giacomini-Kitagawa robust bands cut the other way: at twelve months the
sign scheme's 90% posterior percentile band excludes zero, but the
identified set does not — its upper bound is positive at every horizon
beyond two months — so the apparent significance of sign-restricted
uncertainty IRFs at business-cycle horizons is a property of the Haar
prior, not of the restrictions. In a variance-decomposition of the curve,
identification-family dummies alone explain 25% of the cross-specification
variance in peak estimates — two-fifths of the 62% that identification,
proxy, sample and detrending explain jointly. Identification choice is not
most of the story, but it is the single choice that can flip the answer.

## 1. Introduction

Since Bloom (2009, *Econometrica*) identified uncertainty shocks recursively
in a monthly VAR — volatility ordered after the stock market and before the
policy and activity block — the literature has accumulated a menu of
alternatives: Baker, Bloom and Davis (2016, *QJE*) popularised the news-based
EPU index inside small recursive VARs; Jurado, Ludvigson and Ng (2015, *AER*)
replaced proxies with an estimated common factor of forecast-error
volatility; Ludvigson, Ma and Ng (2021, *AEJ: Macro*) argued, using shock-
based external constraints, that financial uncertainty is a plausibly
exogenous impulse while macro uncertainty largely responds endogenously; and
Carriero, Clark and Marcellino (2018, *REStat*) estimated uncertainty and its
effects jointly in a large stochastic-volatility-in-mean VAR. Each paper
reports one identification scheme (occasionally two) on one dataset. When
estimates disagree — and published peak activity responses range over an
order of magnitude — it is impossible to tell from the papers alone whether
the disagreement comes from the proxy, the sample, the transformation, or the
identifying assumptions themselves.

This paper holds everything else fixed and varies exactly those four things.
Our contribution is deliberately modest in econometrics and aggressive in
bookkeeping: we implement the identification menu — two recursive orderings,
Rubio-Ramirez-Waggoner-Zha sign restrictions, Antolín-Díaz &
Rubio-Ramírez (2018) narrative sign restrictions anchored to four famous
uncertainty events, Mertens-Ravn external
instruments, Faust-Uhlig max-share, Rigobon (2003) and Magnusson-Mavroeidis
(2014) heteroskedasticity identification, Lanne-Meitz-Saikkonen (2017)
non-Gaussian ICA, and Plagborg-Moller-Wolf (2021) lag-augmented local
projections — as interchangeable estimators applied to the same reduced-form
VAR, and read the distribution of estimates as a specification curve
(Simonsohn, Simmons and Nelson 2020, *Nature Human Behaviour*). For the two
rotation-based schemes we additionally confront the reported percentile
bands with their Giacomini-Kitagawa (2021) identified sets. The exercise
answers a question every referee of an uncertainty paper implicitly asks:
*if this author had chosen a different identification, would the headline
number have moved?* Our answer: the sign usually survives, the magnitude
does not, and two specific choices — trusting a second uncertainty proxy as
an external instrument, and trusting variance regimes to label shocks — are
where the literature's disagreements concentrate.

## 2. Data and methods

**Dataset.** One frozen monthly US panel (`output/panel_monthly.csv`,
sha256 in `output/panel_manifest.json`): an uncertainty proxy, 100·log
industrial production (FRED INDPRO), 100·log nonfarm payrolls (PAYEMS) and
the federal funds rate (FEDFUNDS), a reduced four-variable version of the
Bloom (2009) / Baker-Bloom-Davis (2016) systems (no stock-price block; see
the pipeline docstring for every simplification). Proxies: BBD news-based
EPU (modern series ratio-spliced to the 1900 historical index over their
1985–2014 overlap), the World Uncertainty Index for the USA (quarterly,
forward-filled), JLN macro uncertainty at h=1, and the VIX (monthly mean,
1990+). All proxies are z-scored on their full history.

**Reduced form and normalisation.** VAR(6), constant, horizon 24 months, on
each dataset cell; every VAR-based scheme consumes the identical reduced
form. IRFs are normalised to a shock that raises the uncertainty proxy by
one (full-sample) standard deviation on impact; local projections estimate
the same unit-effect quantity by construction. Statistical identifications
(Rigobon, MagMav, ICA) do not name shocks, so the uncertainty shock is
labelled as the column with the largest absolute impact loading on the
proxy — a documented rule, applied uniformly.

**The menu.** Recursive with uncertainty first and last (90% residual-
bootstrap bands); sign restrictions (uncertainty up, IP down, horizons 0–2;
accepted-draw median and quantile bands); narrative sign restrictions
(Antolín-Díaz & Rubio-Ramírez 2018): the same traditional pattern plus four
Type-I event restrictions — the identified uncertainty shock was *positive*
in 1987-10 (Black Monday), 2001-09 (September 11), 2008-09 (Lehman) and
2020-03 (COVID onset) — with events outside a cell's estimation window
dropped and recorded, AD-RR importance-weighted percentile bands, and the
Kish effective sample size (ESS) reported (with only Type-I restrictions
the importance weights are constant, so ESS equals the surviving-draw
count by construction); proxy-SVAR with AR(6) innovations
of a *different* uncertainty proxy as the instrument (EPU↔JLN partner map,
zero-filled outside availability per the censored-proxy convention of
Stock-Watson 2018), reporting the Olea-Pflueger effective F; max-share
(uncertainty's forecast-error variance over 12 months); Rigobon with the
high-volatility regime defined as Great Recession + COVID months
(`puremacro.regime_dates`); Magnusson-Mavroeidis with sup-Wald/BIC
endogenous breaks; FastICA non-Gaussian identification; and lag-augmented LP
with Eicker-Huber-White bands. Bands for Rigobon and ICA come from a
regime-preserving residual-resimulation bootstrap implemented in the
pipeline.

**The identified-set overlay.** For the sign and narrative schemes on the
baseline dataset we also compute Giacomini-Kitagawa (2021) robust bands
(`var.identify.gk_robust_bands`): the min/max of the IP response over *all*
admissible rotations — the identified set — rather than percentiles across
a Haar sample of them. Two conventions keep the comparison honest: the
reduced form is fixed at the OLS estimate for both objects (`n_var_draws=1`;
the percentile bands also condition on it), and the overlay is drawn on the
raw structural scale (a one-standard-deviation structural shock) because
set bounds are per-rotation extremes that cannot be unit-normalised after
aggregation. `gk_robust_bands` imposes the traditional pattern only; the
narrative-truncated identified set (weakly smaller) is discussed, not
plotted (Figure `output/fig_gk_overlay.pdf`, data in `output/gk_overlay.csv`).

**The grid.** 4 proxies × 3 samples (full, post-1985, pre-2020) × 2
detrendings (within-sample linear trend on log levels vs. log first
differences with cumulated IRFs) × 10 schemes = 240 cells; 189 estimable
(20 duplicate windows, 11 LP cells that are detrend-invariant by
construction, 11 narrative cells outside the baseline detrending — a
documented Phase-2 scope cap, skipped and logged — and 9 MagMav
non-convergences). The curve, family
medians and the bootstrap median test come from
`puremacro.inference.spec_curve`.

## 3. Results

**The menu on one dataset** (`output/headline_menu.csv`; EPU, 1954-07 to
2025-11, linear-detrended, T=857). Identification choice alone spans
−3.3% to +1.4%:

| scheme | peak (90% band) | h=12 |
|---|---|---|
| Cholesky, U first | −1.23 [−1.78, −0.36] | −0.87 |
| Cholesky, U last | −1.11 [−1.51, −0.35] | −0.74 |
| Sign restrictions | −3.31 [−5.19, −0.41] | −3.07 |
| Narrative sign (4 events; ESS 1087) | −2.72 [−4.18, −0.69] | −2.46 |
| Proxy (JLN-innovation IV, F=6.0) | −3.12 [−2.28, +2.45] | −2.95 |
| Max-share (FEV share 0.81) | −1.37 [−1.83, −0.52] | −1.03 |
| Rigobon (var-ratio 1.77) | **+1.43** [−2.91, +1.89] | +0.93 |
| MagMav (4 breaks) | −1.25 [−2.23, +1.61] | −0.90 |
| Non-Gaussian ICA (kurt 25.1) | −1.15 [−1.56, −0.42] | −0.80 |
| Lag-augmented LP | −0.80 [−1.48, −0.12] | −0.66 |

Recursive, max-share, ICA and LP agree on a peak decline of roughly 0.8–1.4%
— the ordering (first vs. last) barely matters. Sign restrictions and the
proxy scheme double-to-triple the magnitude; the proxy bands include zero
(its first stage is weak, F=6). The narrative variant sits between the sign
scheme and the agnostic block: requiring the shock to have been positive in
the four event months discards 24% of the traditionally-accepted rotations
(1,087 of 1,436 survive; all restrictions are Type I, so the importance
weights are constant and ESS equals the surviving count) and shifts both
the median and the band toward zero — peak −2.72 vs −3.31, with a band 27%
narrower at the peak. Rigobon *flips the sign*: the shock loading
most on EPU in the high-volatility regime raises IP by 1.4% at five months —
an LMN-flavoured reminder that variance-regime identification can be picking
up the endogenous-response component of measured uncertainty.

**The curve** (`output/fig_spec_curve.pdf`, data in
`output/spec_curve_results.csv`). Across 189 specifications: median peak
−1.44%, 94.7% negative, bootstrap p-value for a zero median 0.012 (B=2000).
The h=12 response gives the same picture (median −1.21%, 96.8% negative).
The full range is [−46.8, +3.4]; the 5th–95th percentile range is
[−14.8, 0.0]. Only 28.6% of specifications peak at the h=24 boundary, so
the peak statistic is not merely an end-of-horizon artifact.

**Family medians** (`output/table_family_medians.md`):

| family | n | median peak | share negative |
|---|---|---|---|
| local-projection | 11 | −0.90 | 100% |
| non-Gaussian | 22 | −1.04 | 86% |
| recursive | 44 | −1.05 | 100% |
| heteroskedasticity | 35 | −1.19 | 80% |
| max-share | 22 | −1.29 | 100% |
| narrative-sign | 11 | −3.18 | 100% |
| sign | 22 | −3.90 | 100% |
| proxy | 22 | −8.68 | 100% |

Three regularities. First, the "agnostic-computation" families — LP,
recursive, ICA, max-share — cluster tightly around −1%: for these, proxy,
sample and detrending move the estimate far less than the literature's
headline disagreements. Second, sign restrictions are systematically the
most negative point-identified family (accepted-draw medians inherit the
imposed sign at short horizons). Third, the proxy family's median (−8.7%) is
an artifact of weak first stages: cells with Olea-Pflueger F ≥ 10 have a
median of −2.0%, cells with F < 10 a median of −9.7%, and every estimate
below −20% comes from a proxy cell with effective F between 3.7 and 13.6 —
all below the ≈23 threshold Montiel Olea and Pflueger recommend for 10%
worst-case bias (the four clipped markers in the figure). Under
unit-effect normalisation a weak first stage mechanically
inflates the activity response — a spec-curve rendering of the Lewis /
Montiel-Olea-Stock-Watson weak-proxy warnings. Finally, every one of the ten
positive peak estimates belongs to a statistical identification (Rigobon 4,
MagMav 3, ICA 3), concentrated in EPU-full/post-1985 and the short VIX and
JLN pre-2020 windows.

**Narrative restrictions tighten — and moderate — the sign family.** The
narrative scheme runs on the 11 linear-trend cells (events outside a cell's
window are dropped: Black Monday for the VIX cells, the COVID onset for the
pre-2020 ones). Compared with the plain sign scheme on the *same* 11 cells,
the narrative estimate is less negative in 10 of 11 (mean shift +1.8
percentage points), the family median moves from −3.9% to −3.2%, and the
family's spread halves — range [−8.8, −1.5] (7.3pp) against the matched
sign range [−15.2, −2.0] (13.2pp). The tightening is concentrated exactly
where the sign family misbehaves: the three JLN cells that produce the sign
family's −11.8 to −15.2 tail are pulled in to −8.7/−8.8. Between 23% and
83% of traditionally-accepted rotations survive the event restrictions per
cell (grid ESS 59–222 at 1,200 draws). Reading AD-RR through the spec-curve
lens: dated event information does not amplify the uncertainty effect — it
trims the rotations that made the sign family the most negative
point-identified family, without ever flipping a cell positive.

**The identified set versus the percentile band (GK).**
(`output/fig_gk_overlay.pdf`; raw structural scale, OLS reduced form for
both objects.) At h=12 the sign scheme's 90% percentile band is
[−1.26, −0.06] — zero excluded, the "significantly negative" reading —
while the Giacomini-Kitagawa identified set is [−1.36, +0.26]: 1.35×
wider, and it *includes* zero and mildly positive responses. The set's
upper bound turns positive at h=3 and stays positive at every horizon
thereafter (88% of horizons overall — impact horizons 0–2 are negative
because the traditional restrictions directly impose the sign there).
So the robust-Bayes critique bites in exactly one place, but the
important one: nothing in the data plus the sign restrictions rules out
a zero or slightly positive activity response beyond the restricted
horizons; the percentile band's apparent significance is contributed by
the uniform Haar prior over rotations. What the set does *not* do is
overturn the magnitude ranking — even the robust lower bound (−1.36 at
h=12) is far from the proxy family's −8 to −47 tail, and the posterior
mass and the set both sit predominantly below zero. The narrative panel
shows the AD-RR information working *within* an unchanged traditional
set: the weighted band shrinks to 0.89× the sign band's width at h=12
([−1.23, −0.17]) and its upper edge pulls away from zero. The
narrative-truncated identified set itself would be weakly smaller than
the plotted one (the event restrictions delete admissible rotations);
computing it is flagged as package work in the plan.

**Dominance restrictions (Phase 3: AD-RR Types II and III).**
(`--narrative-hd`; `output/headline_menu.csv` rows `narrative_t2`,
`narrative_t3`.) Requiring in addition that the uncertainty shock was the
*most important* contributor to the historical decomposition of the proxy
itself in September 2008 and March 2020 (Type II), or Type II at Lehman
plus *overwhelming* (Type III) at COVID, does two things on the baseline
dataset. First, it tightens: the h = 12 band goes from Type I's
[−4.13, −0.56] to [−3.66, −1.20] (Type II) and [−3.31, −1.10] (Type III)
— 31% and 38% narrower, with the upper edge pulled well away from zero —
while the medians barely move (−2.46 → −2.68 → −2.45), i.e. the dominance
information trims tails rather than relocating the estimate. Second, it
finally makes the AD-RR importance weights earn their keep: the Kish ESS
is 206 on 463 surviving draws (Type II) and 181 on 380 (Type III) — 45%
and 48% — against exactly 100% under Type-I restrictions, where the
weights are constant by construction. In the grid, the dominance variants
add 18 estimable cells (11 Type II, 7 Type III — Type III requires COVID
in-window and is skipped, with a logged reason, on pre-2020 samples);
their cell medians (−3.9%, −3.3%) sit at or below the plain narrative's
−3.2%, and every cell is negative.

**The event sweep (Phase 3).** (`--event-sweep`;
`output/event_sweep.csv`, `output/fig_event_sweep.pdf`.) The four event
months are a specification choice, so we sweep all nine one-event and
leave-one-out Type-I configurations on the baseline dataset. Every
configuration excludes zero at h = 12 — the narrative scheme's
zero-exclusion is not an artifact of any single dated judgment call. The
informative event is October 1987, not the obvious candidates: alone it
gives the tightest single-event band (width 3.95 of the four singles'
3.95–4.59), and removing it loosens the all-events band more than any
other deletion (3.57 → 4.11); dropping March 2020 actually *tightens* the
band slightly (3.57 → 3.53). The intuition is the one AD-RR emphasize:
an informative event is one where the shock's sign is unambiguous
*conditional on the model*, and Black Monday — a huge uncertainty spike
with no contemporaneous real-activity collapse — pins the rotation better
than COVID, where every shock moved at once.

**Answer to the title question.** On the same data the identification menu
spans −3.3% to +1.4% (menu) and family medians span −0.9% to −8.7% (curve).
Regressing the 207 peak estimates on dummies (`summary.json`,
`variance_decomposition`): identification family alone explains 24% of the
variance, the data dimensions (proxy × sample × detrending) 38%, and all
four jointly 62% (25/38/62 on the Phase-2 grid — the composition shift
from 18 added narrative-dominance cells barely moves the decomposition).
Identification is therefore not the largest single source
of spread in magnitudes — but it is decisive for the *sign*: no choice of
proxy, sample or detrending produces a positive peak under recursive,
sign-restriction, narrative-sign, proxy, max-share or LP identification
(those families are
100% negative), while every positive estimate sits inside
the heteroskedasticity and non-Gaussian families.

## 4. Limitations

Identification schemes we did not run, and why: stochastic-
volatility-in-mean joint estimation (Carriero-Clark-Marcellino 2018) requires
an MCMC budget outside this pipeline's Pyodide-pure runtime target; and
Ludvigson-Ma-Ng's full shock-restriction machinery (inequality constraints
on the shocks around events plus correlation bounds) remains out of scope,
although the AD-RR Type-I event restrictions we now impose are a close
cousin of its event-constraint component. Phase-2 additions carry their own
caveats: the narrative scheme fixes the reduced form at the OLS estimate
(only rotation uncertainty is sampled, matching the plain sign scheme) —
this carries over to the Phase-3 dominance variants; in the grid the
narrative schemes run only under the baseline linear-trend detrending
(first-difference cells skipped and logged). Two Phase-2 caveats are now
resolved rather than open: the ESS diagnostic is informative under the
Type II/III restrictions (45–48% of the surviving-draw count, vs
identically 100% under Type I), and the event-month specification choice
has been swept (all nine configurations exclude zero at h = 12; § 3).
What remains: the dominance events (Lehman, COVID) are themselves a
judgment call we imposed rather than swept — a Type-II sweep would need
a defensible dominance story per candidate month, which is exactly the
judgment the sweep cannot automate; and the window is fixed at the single
event month (window = 0), so persistence of dominance is untested. The GK overlay imposes the traditional sign pattern only (the
narrative-truncated set is weakly smaller but not computable from the
current `gk_robust_bands` API), switches off reduced-form uncertainty
(`n_var_draws=1`), and lives on the raw structural scale — set bounds
cannot be unit-normalised after aggregation, so its panel is not directly
comparable to the unit-effect numbers elsewhere in the paper. Within the
menu: MagMav failed to converge in 9 of its 11
first-difference cells (its variance-break identification is fragile after
differencing) and its estimates enter the curve only where its own
convergence flags pass; proxy-SVAR exogeneity of a second uncertainty
measure is assumed, not tested (the LMN critique applies with full force —
we report first-stage strength, not validity); the unit-effect normalisation
makes weak-loading schemes look explosive by construction (we flag rather
than truncate them, and report the F-split); the four-variable system omits
the stock-market block of Bloom (2009), so recursive cells are not exact
replications. The lag order is no longer held fixed: the Phase-4 sweep
(`--ph-sweep`; `output/ph_sweep.csv`, `fig_ph_sweep.pdf`) re-runs the
recursive, sign, proxy and max-share schemes on the baseline dataset at
p ∈ {3, 6, 12} and every one of the 15 cells stays negative, with
per-scheme peak ranges of at most 0.72 pp (recursive orderings move by
under 0.18) — the headline is insensitive to the lag-order choice. The
IRF horizon remains fixed at H = 24 with the peak searched within it
(peak_h is reported per cell, so a separate H truncation sweep would
add nothing the peak search does not already reveal); monthly
frequency is a maintained choice. The historical-EPU splice and the
WUI forward-fill are documented data choices frozen in the panel manifest;
rerunning `tools/run_uncertainty_ident_spec_curve.py` from the frozen CSV
reproduces every number in this draft bit-for-bit (the pipeline asserts
this on every run).

## References

*(Scheme-level citations live in the corresponding `puremacro` module
docstrings; the positioning literature is cited in the text: Bloom 2009;
Baker-Bloom-Davis 2016; Jurado-Ludvigson-Ng 2015; Ludvigson-Ma-Ng 2021;
Carriero-Clark-Marcellino 2018; Rigobon 2003; Mertens-Ravn 2013;
Stock-Watson 2018; Magnusson-Mavroeidis 2014; Lanne-Meitz-Saikkonen 2017;
Plagborg-Moller-Wolf 2021; Antolín-Díaz & Rubio-Ramírez 2018; Giacomini &
Kitagawa 2021; Simonsohn-Simmons-Nelson 2020.)*
