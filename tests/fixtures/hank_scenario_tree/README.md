# Finite-tree HANK evidence

This is the completed 2026-11-01 production-solver study. `record.json`
authenticates the numerical implementation and records every completed or
failed case. `retention.json` authenticates the committed records and paths.
Large household NPZ arrays are explicitly omitted from this compact fixture;
their original checksums remain recorded. The runners regenerate those arrays.

The independent deterministic solver comparison has maximum path difference
7.10e-10, below the 3e-8 tolerance. Stochastic comparisons use the first six
dates of conditional C/Y/pi responses to an unresolved, equiprobable +/-0.005
monetary innovation. The tolerance is 1% of the finer conditional response's
peak, with a level floor of 1e-6. Each grid recalibrates beta and labor
disutility to common bond and effective-labor targets.

These final comparisons pass for all three variables:

- Asset bound 300->600, with 480 asset nodes, 15 income states and 48 dates.
- Income states 21->31, with 480 asset nodes, bound 600 and 48 dates.
- Asset nodes 480->960, with 31 income states, bound 600 and 48 dates.
- Horizon 48->96, with 480 asset nodes, 31 income states and bound 600.

The income comparison's largest fraction of tolerance is about 98.3%. These
results bound the stated comparisons; they do not establish the infinite-grid
limit or precise small uncertainty corrections. Earlier failures are retained:
12->24 dates, 5->7 and 11->15 and 15->21 income states; the narrow 11-income
grid exceeds the policy-exit mass threshold, and the narrow 15-income grid
fails steady distribution construction. Consequently `all_refinements_passed`
remains false. No failed case is silently used as an accepted comparison.

Reproduce in new directories, using the committed source and the recorded
environment or a separately recorded compatible environment:

```sh
python tools/reference_validation/check_hank_scenario_tree.py --output tree-base
python tools/reference_validation/extend_hank_scenario_tree.py \
  --previous tree-base --output tree-extended \
  --case horizon48:240:7:48 --case narrow11:240:11:48 --case narrow15:240:15:48 \
  --case bound11:240:11:48:300 --case dense11:480:11:48:300 \
  --case dense15:480:15:48:300 --case horizon96:480:15:96:300 \
  --case bound600:480:15:48:600 --case income21:480:21:48:600 \
  --case income31:480:31:48:600 --case assets31:960:31:48:600 \
  --case horizon31:480:31:96:600 \
  --compare income:horizon48 --compare horizon48:narrow11 \
  --compare narrow11:narrow15 --compare narrow11:bound11 \
  --compare bound11:dense11 --compare dense11:dense15 \
  --compare dense15:horizon96 --compare dense15:bound600 \
  --compare bound600:income21 --compare income21:income31 \
  --compare income31:assets31 --compare income31:horizon31
python tools/reference_validation/freeze_hank_tree_evidence.py \
  --source tree-extended --output tree-compact
```

The model assumes risk-neutral nominal firm valuation and steady terminal
marginal values. There is no independent stochastic author-code replication,
stationary aggregate-risk equilibrium or infinite-horizon welfare certificate.
