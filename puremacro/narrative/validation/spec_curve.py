"""NarrativeSpecificationCurve: LP-IV across instrument-construction choices."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd


_SOURCE_SUBSETS = ["canonical_only", "canonical+llm", "all"]
_AGGREGATIONS   = ["sum", "sum_sign", "count"]
_CONFIDENCE_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
_TARGETS        = ["investment", "consumption", "both"]
_TIME_WINDOWS   = ["full", "post1980", "post1990", "post2000"]
_SURPRISE_FLAGS = [True, False]
_HARMONIZE_FLAGS = [True, False]
_IV_KINDS       = ["announcement", "implementation"]

_WINDOW_STARTS = {
    "full": None,
    "post1980": "1980-01-01",
    "post1990": "1990-01-01",
    "post2000": "2000-01-01",
}


@dataclass
class NarrativeSpecificationCurve:
    """Run LP-IV over a cross-product of instrument construction choices.

    Parameters
    ----------
    panel : HomogeneousFiscalPanel (already quality-filtered if desired).
    df : macro DataFrame indexed by quarterly dates. Must contain columns
        named by ``y``, ``x_invest``, ``x_consume``.
    y : outcome variable column name.
    x_invest : endogenous variable for investment specs.
    x_consume : endogenous variable for consumption specs.
    horizon : max LP horizon (default 8).
    n_boot : reserved for future bootstrap infrastructure; ``lp_iv`` uses
        HAC SEs. Pass ``0`` for forward compatibility.
    max_specs : if set, randomly subsample this many specs (useful for speed).
    seed : random seed for subsample draw.

    Robustness dimensions
    ---------------------
    ``iv_kind`` ∈ {"announcement", "implementation"} is a per-spec dimension:
    each spec is run with both values, doubling the spec count.  The
    "announcement" variant places the full signed magnitude in the quarter the
    policy was announced; "implementation" distributes it over the event's
    ``effective_profile`` (timing-weighted).

    ``within_year_rule`` is **panel-level**: it controls how
    ``HomogeneousFiscalPanel.from_canonical`` spreads intra-year events across
    quarters.  Varying it requires re-running the panel construction upstream
    and passing a fresh panel into this class.  It is exposed in
    ``dimensions()`` for documentation / sensitivity bookkeeping, but does NOT
    add extra specs within a single curve run.
    """

    _DIMENSIONS: ClassVar[dict[str, list]] = {
        "source_subset":    _SOURCE_SUBSETS,
        "aggregation":      _AGGREGATIONS,
        "confidence_thr":   _CONFIDENCE_THRESHOLDS,
        "target":           _TARGETS,
        "time_window":      _TIME_WINDOWS,
        "surprise_filter":  _SURPRISE_FLAGS,
        "harmonized":       _HARMONIZE_FLAGS,
        "iv_kind":          _IV_KINDS,
        # Panel-level only — documented here but does not multiply spec count.
        "within_year_rule": ["uniform", "front_loaded", "fiscal_year"],
    }

    @classmethod
    def dimensions(cls) -> dict[str, list]:
        """Return the spec grid as a dict of dimension → candidate values.

        All dimensions except ``within_year_rule`` are enumerated per spec.
        ``within_year_rule`` is panel-level: varying it requires re-running
        ``HomogeneousFiscalPanel.from_canonical(within_year_rule=...)``
        upstream and passing a per-rule panel into a separate curve instance.
        """
        return {k: list(v) for k, v in cls._DIMENSIONS.items()}

    panel: Any
    df: pd.DataFrame
    y: str
    x_invest: str
    x_consume: str
    horizon: int = 8
    n_boot: int = 500
    max_specs: int | None = None
    seed: int = 42
    _results: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def _build_specs(self) -> list[dict]:
        keys = ["source_subset", "aggregation", "confidence_thr",
                "target", "time_window", "surprise_filter", "harmonized",
                "iv_kind"]
        combos = list(itertools.product(
            _SOURCE_SUBSETS, _AGGREGATIONS, _CONFIDENCE_THRESHOLDS,
            _TARGETS, _TIME_WINDOWS, _SURPRISE_FLAGS, _HARMONIZE_FLAGS,
            _IV_KINDS,
        ))
        specs = [dict(zip(keys, c)) for c in combos]
        if self.max_specs is not None and len(specs) > self.max_specs:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(specs), size=self.max_specs, replace=False)
            specs = [specs[i] for i in idx]
        return specs

    def _instrument_for_spec(self, spec: dict) -> pd.Series | None:
        """Build a quarterly instrument series for one spec. Returns None if unusable."""
        target = spec["target"]
        conf_thr = spec["confidence_thr"]

        events = list(self.panel.events)

        # Source subset filter.
        if spec["source_subset"] == "canonical_only":
            events = [e for e in events if e.metadata.get("replication")]
        elif spec["source_subset"] == "canonical+llm":
            events = [e for e in events
                      if e.metadata.get("replication") or e.scoring_method == "llm"]
        # "all" -> no filter

        # Confidence filter.
        events = [e for e in events if e.confidence >= conf_thr]
        if not events:
            return None

        # Apply surprise decay and magnitude harmonization per spec flags.
        if spec["surprise_filter"]:
            from ..quality.surprise_filter import SurpriseFilter
            events = SurpriseFilter(
                decay_factor=0.25, confidence_floor=conf_thr,
            ).apply(events)
            if not events:
                return None

        if spec["harmonized"]:
            from ..quality.magnitude_harmonizer import MagnitudeHarmonizer
            events = MagnitudeHarmonizer().apply(events)

        from dataclasses import replace as dc_replace
        from ..aggregate import events_to_quarterly

        aggregation = spec["aggregation"]
        iv_kind = spec.get("iv_kind", "implementation")
        if aggregation == "sum_sign":
            # Each event contributes its sign only (±1), not magnitude.
            events = [dc_replace(e, magnitude=1.0) for e in events]
            z = events_to_quarterly(
                events,
                target_filter=None if target == "both" else target,
                aggregation="sum",
                sign_weighted=True,
                confidence_threshold=conf_thr,
                iv_kind=iv_kind,
            )
        elif aggregation == "count":
            z = events_to_quarterly(
                events,
                target_filter=None if target == "both" else target,
                aggregation="sum",
                sign_weighted=False,
                confidence_threshold=conf_thr,
                iv_kind=iv_kind,
            )
            z = (z != 0).astype(float)
        else:  # "sum"
            z = events_to_quarterly(
                events,
                target_filter=None if target == "both" else target,
                aggregation="sum",
                sign_weighted=True,
                confidence_threshold=conf_thr,
                iv_kind=iv_kind,
            )

        return z

    def _run_one(self, spec: dict, spec_id: str) -> list[dict]:
        from ..validate import weak_iv_summary
        from ...lp.iv import lp_iv

        target = spec["target"]
        x_col  = self.x_invest if target in ("investment", "both") else self.x_consume

        z = self._instrument_for_spec(spec)
        if z is None or z.empty:
            return []

        # Time window.
        start = _WINDOW_STARTS[spec["time_window"]]
        df = self.df.copy()
        if start:
            df = df[df.index >= pd.Timestamp(start)]

        # Align instrument to df index.
        z_aligned = z.reindex(df.index).fillna(0.0)
        n_nonzero = int((z_aligned != 0).sum())
        if n_nonzero < 8 or len(df) < 40:
            return []

        # LP-IV at horizon.
        df2 = df.copy()
        df2["_z"] = z_aligned.values
        try:
            lp_res = lp_iv(df2, y=self.y, x=x_col, z="_z",
                           horizons=[self.horizon // 2, self.horizon],
                           n_lags=2, alpha=0.10)
        except Exception:
            return []

        h4 = self.horizon // 2
        h8 = self.horizon
        beta_h4 = float(lp_res.loc[lp_res["h"] == h4, "beta"].iloc[0]) if h4 in lp_res["h"].values else np.nan
        beta_h8 = float(lp_res.loc[lp_res["h"] == h8, "beta"].iloc[0]) if h8 in lp_res["h"].values else np.nan
        se_val  = float(lp_res.loc[lp_res["h"] == h8, "se"].iloc[0]) if h8 in lp_res["h"].values else np.nan

        try:
            wiv = weak_iv_summary(
                z_aligned.values,
                df[x_col].values,
                df[self.y].values,
            )
        except Exception:
            wiv = {"n_obs": 0, "first_stage_f_cd": np.nan,
                   "first_stage_f_kp": np.nan, "AR_p_value_at_2sls": np.nan,
                   "beta_2sls": np.nan}

        return [{
            "spec_id":             spec_id,
            "source_subset":       spec["source_subset"],
            "aggregation":         spec["aggregation"],
            "confidence_thr":      spec["confidence_thr"],
            "target":              target,
            "time_window":         spec["time_window"],
            "surprise_filter":     spec["surprise_filter"],
            "harmonized":          spec["harmonized"],
            "iv_kind":             spec.get("iv_kind", "implementation"),
            "n_obs":               wiv["n_obs"],
            "n_nonzero_quarters":  n_nonzero,
            "first_stage_f_cd":    wiv["first_stage_f_cd"],
            "first_stage_f_kp":    wiv["first_stage_f_kp"],
            "beta_2sls_h4":        beta_h4,
            "beta_2sls_h8":        beta_h8,
            "AR_p_value_at_2sls":  wiv["AR_p_value_at_2sls"],
            "se_2sls":             se_val,
        }]

    def run(self) -> pd.DataFrame:
        """Execute all specs and return results DataFrame."""
        specs = self._build_specs()
        rows: list[dict] = []
        for i, spec in enumerate(specs):
            spec_id = f"spec_{i:04d}"
            rows.extend(self._run_one(spec, spec_id))
        if not rows:
            self._results = pd.DataFrame()
        else:
            self._results = pd.DataFrame(rows).reset_index(drop=True)
        return self._results

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Call .run() first.")
        return self._results


__all__ = ["NarrativeSpecificationCurve"]
