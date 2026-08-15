# Main Street Uncertainty Paper Plan — Phase 3 (identification) + Phase 4 (drafting)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase-2 descriptive fact (state unemployment rises ~0.04 pp over 2–3 years after a district Beige Book labor-uncertainty innovation) into an exposure-differential design with honest few-cluster inference, add mass-layoff outcome margins, and produce the paper skeleton (`DRAFT.md`) with pipeline-produced numbers only.

**Architecture:** One self-contained tool (`tools/run_main_street_phase3.py`) reading the Phase-2 outputs (`output/state_district_panel_quarterly.csv`, `data/processed/bbui_district_panel.csv`), fetching exposure series through the key-free FRED fredgraph CSV via the on-disk HTTP cache (resumable), and estimating interacted panel LPs with state FE + district-x-quarter FE, Driscoll-Kraay and wild-cluster (CGM 2008, Rademacher over the 12 districts, null imposed, FE reprojected per draw) inference side by side. No package modules touched; nothing under `puremacro/examples` or `notebooks/` (owned by parallel streams).

**Tech Stack:** numpy / scipy / pandas / matplotlib + `puremacro` internals only. Tools are not gated by the public-API snapshot; no new public symbols.

---

## File map

### New files (Phase 3 — DONE 2026-07-20)
- `tools/run_main_street_phase3.py` — the pipeline (argparse: `--wc-draws`, `--placebo`, `--seed`, `--fast`).
- `docs/research/main_street_uncertainty/DRAFT.md` — paper skeleton, honest numbers.
- `docs/research/main_street_uncertainty/output/` adds: `exposure_state.csv`, `warn_state_quarterly.csv`, `warn_coverage.csv`, `mls_state_quarterly.csv`, `irf_exposure.csv`, `pretrends.csv`, `placebo_shuffle.csv`, `jackknife_districts.csv`, `fig_exposure_irf.{png,pdf}`, `fig_phase3_robustness.{png,pdf}`, `phase3_manifest.json`, `phase3_summary.json`, `run_log_phase3.txt`.
- `docs/plans/2026-07-20-main-street-phase3-plan.md` — this plan.

### Modified files
- `docs/research/main_street_uncertainty/FINDINGS.md` — status header pointing at `DRAFT.md`.
- (Hard rules respected: no package modules, no CHANGELOG, no version bump, no snapshot, no notebooks.)

### Working assumptions (verified 2026-07-20 via probes + live runs)
- FRED fredgraph key-free serves `{ST}MFG` for 50 states 1990-01+ (DC only as `DCMFGN`) and `{ST}NRMN` for 45 states (missing DC/DE/FL/HI/MD/NE). `puremacro.fetch.bea_industry_shares` prefers a *synthetic* teaching bundle and `state_industry_panel.STATE_INDUSTRY_SHARES_2005` is an approximated snapshot — both rejected for research use; exposure is built from real CES mirrors instead.
- `puremacro.narrative.sources.us_warn.iter_us_warn` reaches 45 states through cached state-DOL scrapes + BLN(-GCS) + WARNTracker parquets (`../data/processed/warn_*.parquet`, `../notebooks/data_cache/warn_{ca,ny}*.parquet`); coverage spans are state-heterogeneous (GA 1989+, NY 2023+ only).
- `../data/processed/warn_bls_mls.parquet` = BLS Mass Layoff Statistics `MLUMS<SS>NN0001003` (built by `uncertainty_examples/tools/scrape_bls_mls.py`); flat files omit zero-event months (min reported count is 3) → zero-filled inside the program window.
- `puremacro.inference.wild_bootstrap` is a time-series residual wild bootstrap, not a cluster one → in-script wild-cluster implementation citing Cameron-Gelbach-Miller 2008.
- District-x-quarter FE + state FE via sparse alternating-projections demeaning (in-script; the house `two_way_fe_within` hardwires entity/time and entity-cluster SEs).

---

## Phase 1 — corpus + connector (DONE, prior stream)
- [x] Fed-district crosswalk, `bbui(level='district')`, federalreserve.gov Beige Book connector.

## Phase 2 — full-history panel + first LPs (DONE 2026-07-19)
- [x] 1983Q3–2025Q4 x 12 districts, all 2,040 cells document-backed; `tools/run_main_street_lp.py`; pooled two-way-FE LP (+0.042 pp urate at h=8, DK SEs, sup-t bands); `FINDINGS.md`.

## Phase 3 — identification (DONE 2026-07-20)
- [x] **Task 1: exposure interaction.** 1990–91 manufacturing (+ mining robustness) shares, frozen; interacted LP with state FE + district-x-quarter FE, estimation from 1992Q1; DK + wild-cluster side by side. Headline: +0.039 pp at h=9, DK 90% CI [0.008, 0.069], WC p=0.14 (p<0.10 only h=10–12); employment mirror −0.107% at h=12; mining differential robustly negative.
- [x] **Task 2: WARN + MLS events.** Full WARN quarterly panel (45 states, 47,622 filings) + coverage table; exposure LP only on the 17-state fully-covered 2015–2021 subset, labeled FRAGILE (6 effective districts, null). BLS MLS 1998–2013 (51 states) exposure LP: no positive layoff-event differential (−0.018 at h=1, WC p=0.054, wrong-signed).
- [x] **Task 3: placebos + robustness.** Shuffle-district derangement placebo (mean +0.019 = half signal, RI p=0.10, explained by +0.10 cross-district shock correlation); leads h=−4..−2 (h=−2: −0.015, WC p=0.04 — flagged, not hidden); drop-one-district jackknife ([+0.032, +0.045], all positive).
- [x] **Task 4: paper skeleton.** `DRAFT.md` (abstract + intro positioning vs BBD state EPU / Mumtaz et al. / granular-shift-share literature + data + design + results + limitations); `FINDINGS.md` status header; this plan file.

## Phase 4 — drafting + submission targets (NEXT)
- [x] **Split-state border-pair design:** DONE 2026-07-21 (batch-4 plan) — county→district crosswalk built from the banks' own county lists (`tools/build_fed_county_crosswalk.py`), all-county + border-only LPs in `tools/run_main_street_phase4_border.py`. Result: clean wrong-signed null — no own-vs-neighbor differential at the county level.
- [x] **Leave-own-district-out national BBUI** DONE 2026-07-21 — horse race in `tools/run_main_street_phase4.py`: own survives at ratio 0.56 (pre-registered rule), LOO-national is the robust term (+0.035 at h=12, WC p=0.007); placebo closes to +0.000 paired.
- [ ] **Corpus extension pre-1983** (archival FRASER scans; OCR budget) and real-time vintage outcomes (ALFRED).
- [ ] **Full shift-share exposure vector** (10 CES supersectors from real BEA/QCEW data — requires a keyed or bulk source; the bundled BEA fallback is synthetic and must not be used).
- [ ] **Write-up:** expand DRAFT.md into full text; venue candidates in order: *Journal of Applied Econometrics* (measurement + honest inference), *Journal of Money, Credit and Banking* (Fed-institutional angle), *Regional Science and Urban Economics* (subnational focus). Workshop target: SEA/Midwest Macro. Data/code release via the existing artifact conventions.
