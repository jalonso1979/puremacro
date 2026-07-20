> 🇬🇧 English · 🇪🇸 [Español](es/SIGNAL_CONTRACT.md)

# Signal contract

> Status: **Slice 1 shipped in 0.65.0** (sparsity-only quality).
> Slice 2 (draws + propagation) and Slice 3 (calibration) are planned.

The signal contract is the per-`RiskIndex` data shape that lets a
downstream LP / SVAR estimator know how reliable an index reading is
and (later) propagate that reliability into IRF bands.

## Schema (Slice 1)

`puremacro.narrative.types.RiskIndex` carries two optional fields:

```python
@dataclass
class RiskIndex:
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict
    quality: SignalQualityReport | None = None   # 0.65.0+
    draws:   pd.DataFrame | None          = None  # 0.66.0+
```

Both default to `None` — every pre-0.65.0 caller behaves identically.

## Opt-in (Slice 1)

```python
from puremacro.narrative.indices import lui

ri = lui(records, country="USA", language="en", with_quality=True)
ri.quality.summary()      # one-row DataFrame: mean_n_docs, mean_doc_length, n_coverage_gaps, ...
```

The same `with_quality=False` kwarg sits on every canonical index in
`puremacro.narrative.indices.__all__`: `epu`, `mpu`, `gpr`, `tone`,
`wui`, `lui`, `ltui`, `ltui_up`, `ltui_down`, `lwui`, `lwui_wage`,
`bbui`, `cboui`, `ep_ui`, `erpui`, `eurlex_ui`, `sotuui`, `bluesky_ui`.

## `SignalQualityReport` (Slice 1 fields populated)

| Field                 | Populated in | Description                                          |
|-----------------------|--------------|------------------------------------------------------|
| `n_docs_per_period`   | 0.65.0       | docs per quarter that fed the kernel                 |
| `avg_doc_length`      | 0.65.0       | mean tokens per doc per quarter                      |
| `coverage_gaps`       | 0.65.0       | quarters with zero docs inside the date range        |
| `kernel_agreement`    | 0.66.0       | mean pairwise corr across kernel draws (Slice 2)     |
| `multilingual_parity` | 0.66.0       | corr between language subsets (Slice 2)              |
| `doc_bootstrap_sd`    | 0.66.0       | per-period sd across doc-bootstrap draws (Slice 2)   |
| `corpus_loo_max_swing`| 0.66.0       | max \|Δ\| across leave-one-corpus draws (Slice 2)    |
| `benchmark_scores`    | 0.67.0       | per-key Pearson/Spearman/RMSE vs. canonical (S3)     |
| `event_panel`         | 0.67.0       | rank-corr + top-decile hit + AUC vs. event panel     |
| `survey_scores`       | 0.67.0       | per-key Pearson/RMSE vs. survey series               |

## Validation

`RiskIndex.__post_init__` validates `draws` (when set):
- `draws.index` must equal `series.index`.
- `draws.columns` must be a 2-level `pd.MultiIndex` named `['source', 'draw_id']`.
- Every value of the `source` level must be in `{'kernel', 'lexicon', 'doc', 'corpus'}`.
