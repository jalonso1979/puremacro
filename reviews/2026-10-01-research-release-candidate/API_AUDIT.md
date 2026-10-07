# Additive API reconciliation

Nine stale snapshot entries describe earlier, already implemented worktree
changes. This review checked their definitions, defaults and behavioral
regressions before updating those entries. The update did not regenerate the
whole fixture or change any implementation, assertion, reference value, or
known-failure whitelist. Newly added research APIs are updated separately by
their implementers.

| Snapshot entry | Additions and meaning | Existing regression evidence |
| --- | --- | --- |
| `puremacro.dsge.__all__` | `StructuralSingularityWarning` and `allow_structural_singularity`; public warning/context-manager access to existing singularity handling | `test_fix_I1_dsge_core.py`, `test_fix_STEADY_structural_singularity.py` |
| `ShiftShareIVResult` | Shock-construction mode and sector-cluster count | `test_fix_AKM_shift_share_projection.py` |
| `CallawaySantannaResult` | Overall uncertainty, aggregation identity, confidence level, cohort/aggregation tables and event-study covariance | `test_did/test_fix_did_aggregation.py` |
| `SunAbrahamResult` | Overall uncertainty, aggregation identity, confidence level, aggregation table and event-study covariance | `test_did/test_fix_did_aggregation.py` |
| `SyntheticDiDResult` | Variance-estimator identity, confidence level and actual replication count | `test_did/test_fix_SDID_inference_and_weights.py` |
| `GMMResult` | Tuple of diagnostic notes about instrument construction | `test_dynpanel/test_fix_dynpanel_instruments.py` |
| `PITUniformityResult` | Stored test significance level | `test_fix_NOWCAST_plots_and_revision_stats.py` |
| `RealtimeNowcastResult` | Model result, observed panel and previous-vintage identity | `test_fix_NOWCAST_plots_and_revision_stats.py`, `test_fix_RTNOW_realtime_asof.py` |
| `MSVARResult` | Initial regime probabilities, log-likelihood path and covariance-floor diagnostics | `test_fix_MSAR_ms_var_em.py` |

All additions to result dataclasses have defaults. No previous public export or
field is removed. This is a structural compatibility check; changed estimator
behavior in earlier fixes still requires its own scientific validation.

Validation: **286 passed**, six optional-reference skips and four deselections
across the relevant regression files and release-tool tests. See
[api-regressions.log](api-regressions.log) (275 passed) and
[gmm-notes-regressions.log](gmm-notes-regressions.log) (11 passed).
[api-reconciliation.json](api-reconciliation.json) records the exact before/after
field lists, SHA-256 values and update scope. The reproducible update script is
[reconcile_api.py](reconcile_api.py).

The initial checkout contained 541 status entries and continued to change while
other research work was implemented. That preflight inventory is not the final
candidate manifest. The final wheel and release gate use their own explicit
source freeze.
