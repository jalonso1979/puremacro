# puremacro Batch 7 — BEA shift-share exposure, X-11 v2 ends, tightness-regime LP

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal (three legs, in value order):**
(A) Main Street phase 6 — replace the single 1990-91 manufacturing share with a FULL industry-share exposure vector from real BEA data (the batch-4 limitation; a BEA key exists in the workspace `.env`). (B) X-11 v2 — close the documented left-boundary gap vs the binary (no-backcast asymmetric ends: Musgrave Henderson ends from the closed-form revision-minimizing formula, golden-tested; seasonal-MA end handling validated empirically against new pinned goldens — any constant that cannot be verified against binary output does NOT ship). (C) Notebook 18 extension or sibling analysis: state-dependent LP with labor-market tightness as the regime (connects the Beveridge notebook to `puremacro.lp` state-dependent machinery).

## Leg A — BEA shift-share exposure (Main Street phase 6)

Working design (probe before building):
- Source: BEA Regional API, SIC-era state employment by industry (SAEMP25S covers 1969-2000 — the 1990-91 base period the design freezes). `credentials` service `bea` (check `puremacro/credentials.py`; key in `uncertainty_examples/.env` as BEA_API_KEY, 36 chars).
- Fetcher `puremacro/fetch/bea_regional.py` (or extend existing bea_* modules — check `bea_cainc.py` conventions first): state × SIC-division employment shares, 1990-91 mean, all 51 states. SIC divisions: farm; ag services/forestry/fishing; mining; construction; manufacturing; transport & public utilities; wholesale; retail; FIRE; services; government.
- Exposure vector: z-scored shares for ~8 aggregated divisions (drop farm/ag-services tiny ones into 'other'). CRITICAL honesty check from batch 4: the bundled `bea_industry_shares` teaching fallback is SYNTHETIC — this leg must use the live keyed API and freeze the result (`docs/research/main_street_uncertainty/output/bea_sic_shares_9091.csv` + provenance manifest).
- `tools/run_main_street_phase6.py`: phase-4 horse-race spec generalized to K simultaneous exposure interactions (own + LOO variants); joint Wald (DK) + per-industry WC p; comparison row: does manufacturing survive controlling for the other divisions' exposures? Outputs `irf_shiftshare.csv`, `fig_phase6_shiftshare.*`, `phase6_manifest/summary`, DRAFT.md §7 + limitations update (the shift-share item resolves).

- [x] A1: DONE 2026-07-22 — BEA key installed to credentials.toml and probed, but the SIC-era SAEMP25 tables are RETIRED from the modern Regional API; pivoted to the key-free CES state supersector mirrors (same source/denominator/base as phase 3 — methodologically better). 11-supersector partition verified live ({ST}CONS/TRAD/INFO/FIRE/PBSV/EDUH/LEIH/SRVO/GOVT + MFG/NRMN; SA→NSA fallback, e.g. CAINFO only as CAINFON).
- [x] A2: DONE — `ces_supersector_shares_9091.csv` frozen (51 states; 45 with full 11-sector coverage); phase-3 mfg share reproduced to 9e-17 as a hard gate.
- [x] A3: DONE — `tools/run_main_street_phase6.py` full B=999 run. Verdict: joint DK Wald (df=10) p≤0.001 at h=0–11; NO individual division separable (mfg h=5: +0.12, DK se 0.16 = 9× phase-3, WC p=0.43; 2/260 cells at WC<0.10 = below chance). DRAFT §7 + limitations updated: exposure language is now "industrial-composition gradient," not a manufacturing mechanism.

## Leg B — X-11 v2 left-boundary fidelity

- [x] B1: `maxback=0` goldens generated (`--left-end` flag; `tests/fixtures/sa_goldens_lb/`, 9 series). Finding: the binary's own output moves up to 11.4% at the series start between maxback modes — backcasts DO enter some internal stages despite the B2 table's late start.
- [x] B2+B3 RESOLVED BY MEASUREMENT, Musgrave ends deprioritized: the native engine is CLOSER to the binary's default no-backcast mode than to the pinned maxback=60 spec on wild series (Keweenaw boundary 9.6% vs 14.3%), boundaries on clean series are already sub-1% (INDPRO 0.59%, BELOW its interior max), and the residual noisy-series gaps appear at boundary AND interior alike — i.e. they are extreme-replacement (B17/B20 weight-cascade) differences, not end-filter differences. Implementing Musgrave ends would not close them; no unverifiable constants were shipped. Evidence recorded in `test_x11_vs_binary_default_left_end`.
- [x] B4 (rescoped): both-mode distances tested and frozen; the fallback rewire stays deferred until the B17/B20 cascade — now confirmed as THE v2 lever — is implemented (its own future leg). [DONE in batch 8, 2026-07-22: interior maxima halved; fallback rewire remains deferred pending the 0.93 public-port decision.]

## Leg C — tightness-regime state-dependent LP

- [x] C1: DONE 2026-07-22 — new §"Does uncertainty bite harder in a slack labor market?" in Notebook 18 (EN+ES, rebuilt): `lp_state_dep` with log-tightness as the logistic state, EPU from the nb17 panel, IP + urate outcomes. Result: a TEACHABLE NULL (no horizon significant in either regime; IP points lean opposite to the bites-harder-in-slack prior); the honest-read cell describes exactly this data and ties it to nb07's inference caveats.

## Status
- Batches 4, 5a, 5b, 6 v1 all committed (3ab52c17, 664acec8, 144bcb4a, d6cb7d23). This plan created 2026-07-22 after "go ahead with the natural moves".
