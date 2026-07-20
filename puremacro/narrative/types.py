"""Core data structures: NarrativeEvent and NarrativeInstrument."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..instruments import Instrument

import numpy as np
import pandas as pd


VALID_KINDS = {"fiscal", "monetary", "macropru", "fx", "structural"}

VALID_TARGETS_BY_KIND = {
    "fiscal":     {"investment", "consumption", "both"},
    "monetary":   {"policy_rate", "asset_purchase", "forward_guidance",
                   "fx_intervention", "lending_facility"},
    "macropru":   {"capital_buffer", "ltv_dsti", "sector_limit",
                   "reserve_requirement"},
    "fx":         {"intervention", "peg_change"},
    "structural": {"labor", "product_market", "trade", "tax_admin"},
}

# Backwards compat: VALID_TARGETS retains its original meaning (fiscal kinds).
VALID_TARGETS = VALID_TARGETS_BY_KIND["fiscal"]
VALID_SIGNS = {-1, 0, 1}
VALID_SCORERS = {"keyword", "llm", "manual"}


@dataclass
class NarrativeEvent:
    """A single discrete fiscal announcement.

    Attributes
    ----------
    date : pd.Timestamp
        The announcement date (NOT the implementation date).
    country : str
        ISO3 country code; lowered to support the eventual c-scope.
    magnitude : float
        Magnitude in the unit indicated by ``magnitude_unit``. Always a
        non-negative number; the sign of the shock is carried by ``sign``.
    magnitude_unit : str
        ``"USD_bn"``, ``"pct_gdp"``, ``"z"`` (z-score), or any
        domain-specific unit.
    target : str
        ``"investment"``, ``"consumption"``, or ``"both"``.
    subtarget : str | None
        ``"defense"``, ``"infra"``, ``"transfers"``, ``"wages"``, etc.
    sign : int
        ``+1`` expansionary, ``-1`` contractionary, ``0`` ambiguous.
    confidence : float
        In ``[0, 1]``. Backends may use this to filter out low-quality
        events at aggregation time.
    source_text : str
        Excerpt for audit (truncate long excerpts to the relevant span).
    source_url : str
        Link to the original document.
    scoring_method : str
        ``"keyword"``, ``"llm"``, or ``"manual"``.
    metadata : dict
        Free-form extra fields (act_id, fiscal_year, agency, …).
    implementation_profile : list[tuple[pd.Timestamp, float]]
        Optional schedule of when the fiscal flow lands. Each entry is
        ``(quarter_start, weight)``; weights must sum to 1.0. An empty
        list is the canonical "use default" sentinel — ``effective_profile``
        then returns ``[(self.date, 1.0)]``.
    """
    date: pd.Timestamp
    country: str
    magnitude: float
    magnitude_unit: str
    target: str
    subtarget: str | None
    sign: int
    confidence: float
    source_text: str
    source_url: str
    scoring_method: str
    metadata: dict[str, Any] = field(default_factory=dict)
    implementation_profile: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    kind: str = "fiscal"
    language: str = "en"

    def __post_init__(self):
        # Coerce date.
        if not isinstance(self.date, pd.Timestamp):
            self.date = pd.Timestamp(self.date)
        # Validate kind / target / sign / scoring_method.
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"kind {self.kind!r} not in {VALID_KINDS}"
            )
        valid_targets = VALID_TARGETS_BY_KIND[self.kind]
        if self.target not in valid_targets:
            raise ValueError(
                f"target {self.target!r} not in {valid_targets} "
                f"(for kind={self.kind!r})"
            )
        if int(self.sign) not in VALID_SIGNS:
            raise ValueError(f"sign {self.sign!r} not in {VALID_SIGNS}")
        self.sign = int(self.sign)
        if self.scoring_method not in VALID_SCORERS:
            raise ValueError(
                f"scoring_method {self.scoring_method!r} not in {VALID_SCORERS}"
            )
        # Force magnitude non-negative.
        self.magnitude = float(abs(self.magnitude))
        # Clamp confidence to [0, 1].
        self.confidence = float(min(max(self.confidence, 0.0), 1.0))
        # Coerce + validate implementation_profile.
        if self.implementation_profile:
            coerced: list[tuple[pd.Timestamp, float]] = []
            for d, w in self.implementation_profile:
                coerced.append((pd.Timestamp(d), float(w)))
            self.implementation_profile = coerced
            total = sum(w for _, w in coerced)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"implementation_profile weights must sum to 1.0; "
                    f"got {total!r}"
                )
            slack = pd.Timedelta(days=7)
            for d, _ in coerced:
                if d < self.date - slack:
                    raise ValueError(
                        f"implementation_profile date {d} cannot predate "
                        f"announcement date {self.date} by more than 7 days"
                    )

    @property
    def effective_profile(self) -> list[tuple[pd.Timestamp, float]]:
        """Profile with the default rule applied: empty profile means
        single quarter at ``self.date`` with weight 1.0."""
        if not self.implementation_profile:
            return [(self.date, 1.0)]
        return self.implementation_profile

    @property
    def delay_quarters(self) -> int:
        """Offset (in quarters) from announcement quarter to the first
        implementation quarter. Returns 0 for trivially-defaulted events."""
        prof = self.effective_profile
        first_impl = min(d for d, _ in prof)
        ann_q = pd.Period(self.date, freq="Q")
        impl_q = pd.Period(first_impl, freq="Q")
        return int((impl_q.year - ann_q.year) * 4 + (impl_q.quarter - ann_q.quarter))

    @property
    def signed_magnitude(self) -> float:
        return self.magnitude * self.sign

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict() turns Timestamps inside the profile list into Timestamps still;
        # serialize them as ISO strings for cross-format round-trip.
        d["implementation_profile"] = [
            (pd.Timestamp(date).isoformat(), float(w))
            for date, w in self.implementation_profile
        ]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NarrativeEvent":
        # Drop unknown keys (backwards/forwards compat).
        known = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in d.items() if k in known}
        prof = clean.get("implementation_profile") or []
        clean["implementation_profile"] = [
            (pd.Timestamp(date), float(w)) for date, w in prof
        ]
        return cls(**clean)


@dataclass
class NarrativeInstrument:
    """A quarterly instrumental-variable series built from a list of events.

    Holds *two* parallel series — ``announcement_series`` puts the full
    signed magnitude on each event's announcement quarter (legacy
    behavior), ``implementation_series`` distributes signed magnitude
    over each event's ``effective_profile``. ``quarterly`` aliases the
    implementation series; pass ``iv_kind="announcement"`` to the LP-IV
    / Proxy-SVAR helpers when the legacy series is wanted.

    Use :meth:`from_events` to construct.
    """
    events: list[NarrativeEvent]
    target: str | None
    announcement_series: pd.Series
    implementation_series: pd.Series
    aggregation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quarterly(self) -> pd.Series:
        """Default quarterly IV: the implementation series."""
        return self.implementation_series

    @classmethod
    def from_events(
        cls,
        events: list[NarrativeEvent],
        *,
        target: str | None = None,
        aggregation: str = "sum",
        pct_gdp: pd.DataFrame | None = None,
        freq: str = "QS",
        sign_weighted: bool = True,
    ) -> "NarrativeInstrument":
        from .aggregate import events_to_quarterly
        ann = events_to_quarterly(
            events, target_filter=target, aggregation=aggregation,
            sign_weighted=sign_weighted, pct_gdp=pct_gdp, freq=freq,
            iv_kind="announcement",
        )
        impl = events_to_quarterly(
            events, target_filter=target, aggregation=aggregation,
            sign_weighted=sign_weighted, pct_gdp=pct_gdp, freq=freq,
            iv_kind="implementation",
        )
        # Align both on a shared index so diagnostics correlation works
        # cleanly without reindex juggling at every consumer.
        if not ann.empty and not impl.empty:
            shared = ann.index.union(impl.index)
            ann = ann.reindex(shared, fill_value=0.0)
            impl = impl.reindex(shared, fill_value=0.0)
            ann.index.name = "date"
            impl.index.name = "date"
            ann.name = "narrative_iv"
            impl.name = "narrative_iv"
        return cls(
            events=list(events), target=target,
            announcement_series=ann,
            implementation_series=impl,
            aggregation=aggregation,
            metadata={"sign_weighted": sign_weighted,
                      "pct_gdp": pct_gdp is not None},
        )

    def to_proxy_svar(self, Y, *, p: int, horizon: int,
                      z=None,
                      iv_kind: str = "implementation",
                      n_boot: int = 500, ci: float = 0.9, seed: int = 0):
        """Run :func:`puremacro.var.identify.proxy.proxy_svar` using this
        instrument as the external proxy.

        ``iv_kind`` selects which internal series to feed in
        (``"implementation"`` default, ``"announcement"`` legacy).
        Pass ``z`` explicitly to override.
        """
        from ..var.identify.proxy import proxy_svar
        if z is None:
            z = self._series_for(iv_kind).values
        z = np.asarray(z, dtype=float)
        return proxy_svar(Y, p=p, horizon=horizon,
                          instrument_series=z, n_boot=n_boot, ci=ci, seed=seed)

    def to_lp_iv(self, df, *, y, x, iv_kind: str = "implementation", **kwargs):
        """Wrap :func:`puremacro.lp.iv.lp_iv` using this instrument as ``z``."""
        from ..lp.iv import lp_iv
        z_col = "_narrative_iv_z"
        df2 = df.copy()
        df2[z_col] = self._series_for(iv_kind).reindex(df2.index)
        return lp_iv(df2, y=y, x=x, z=z_col, **kwargs)

    def _series_for(self, iv_kind: str) -> pd.Series:
        if iv_kind == "implementation":
            return self.implementation_series
        if iv_kind == "announcement":
            return self.announcement_series
        raise ValueError(
            f"iv_kind must be 'announcement' or 'implementation'; got {iv_kind!r}"
        )

    def diagnostics(self) -> dict:
        """Calendar density, summary stats, profile coverage, two-series
        correlation."""
        from .validate import event_density
        impl = self.implementation_series
        ann = self.announcement_series
        n_with_profile = sum(
            1 for e in self.events if len(e.implementation_profile) > 1
        )
        max_span = 0
        for e in self.events:
            prof = e.effective_profile
            if len(prof) > 1:
                first = pd.Period(min(d for d, _ in prof), freq="Q")
                last = pd.Period(max(d for d, _ in prof), freq="Q")
                span = (last.year - first.year) * 4 + (last.quarter - first.quarter) + 1
                max_span = max(max_span, span)
        delays = [e.delay_quarters for e in self.events]
        if ann.empty or impl.empty:
            corr = float("nan")
        else:
            joined = pd.concat([ann.rename("a"), impl.rename("i")], axis=1).dropna()
            if joined.shape[0] < 2 or joined["a"].std() == 0 or joined["i"].std() == 0:
                corr = 1.0 if joined["a"].equals(joined["i"]) else float("nan")
            else:
                corr = float(joined["a"].corr(joined["i"]))
        nz_impl = impl.dropna()
        return {
            "n_events": len(self.events),
            "density": event_density(self.events),
            "n_quarters": int(nz_impl.shape[0]),
            "mean": float(nz_impl.mean()) if nz_impl.shape[0] else 0.0,
            "std": float(nz_impl.std()) if nz_impl.shape[0] else 0.0,
            "first_date": str(nz_impl.index.min()) if nz_impl.shape[0] else None,
            "last_date": str(nz_impl.index.max()) if nz_impl.shape[0] else None,
            "n_events_with_profile": n_with_profile,
            "max_profile_span_quarters": max_span,
            "median_delay_quarters": float(np.median(delays)) if delays else 0.0,
            "announcement_vs_implementation_corr": corr,
        }

    def validate_against(self, benchmark: pd.Series) -> dict:
        from .validate import compare_to
        return compare_to(self.quarterly, benchmark)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.events])

    def as_instrument(self, iv_kind: str = "implementation") -> Instrument:
        """Wrap as an :class:`puremacro.instruments.Instrument`.

        ``iv_kind`` selects which underlying series to wrap (default
        ``"implementation"``).
        """
        from ..instruments import Instrument
        is_replication = any(
            "replication" in (e.metadata or {}) for e in self.events
        )
        category = "narrative_replication" if is_replication else "narrative_connector"
        series = self._series_for(iv_kind)
        kinds = sorted({e.kind for e in self.events})
        return Instrument(
            series=series,
            name=self.metadata.get("registry_key", "narrative_instrument"),
            source=self.metadata.get("source", "narrative aggregation"),
            category=category,
            frequency="Q",
            metadata={
                **self.metadata,
                "n_events": len(self.events),
                "target": self.target,
                "aggregation": self.aggregation,
                "iv_kind": iv_kind,
                "kinds": kinds,
            },
        )


VALID_RISKINDEX_METHODS = {"keyword_count", "llm_prob", "tone_dispersion", "hybrid", "sentence_cooccurrence", "length_normalized_count", "triple_cooccurrence_paragraph"}
VALID_RISKINDEX_NORMALIZATION = {"raw", "zscore", "bbd_100"}
_VALID_DRAW_SOURCES = frozenset({"kernel", "lexicon", "doc", "corpus"})


@dataclass
class RiskIndex:
    """A continuous text-derived risk / uncertainty / tone index.

    Attributes
    ----------
    name : str
        Short identifier (e.g., ``"epu_us_news"``, ``"mpu_ecb_speeches"``).
    country : str
        ISO3 code.
    series : pd.Series
        Quarterly index, indexed by quarter-start dates.
    method : str
        ``keyword_count`` | ``llm_prob`` | ``tone_dispersion`` | ``hybrid``.
    corpus : str
        Free-form label of the underlying corpus (e.g., ``"fed_speeches"``).
    language : str
        ISO-639-1 language code of the corpus.
    normalization : str
        ``raw`` | ``zscore`` | ``bbd_100``.
    metadata : dict
        Free-form (e.g., n_docs, vocab_size, source_urls_sample).
    """
    name: str
    country: str
    series: pd.Series
    method: str
    corpus: str
    language: str
    normalization: str
    metadata: dict[str, Any] = field(default_factory=dict)
    quality: "SignalQualityReport | None" = None
    draws: pd.DataFrame | None = None

    def __post_init__(self):
        if self.method not in VALID_RISKINDEX_METHODS:
            raise ValueError(
                f"method {self.method!r} not in {VALID_RISKINDEX_METHODS}"
            )
        if self.normalization not in VALID_RISKINDEX_NORMALIZATION:
            raise ValueError(
                f"normalization {self.normalization!r} not in "
                f"{VALID_RISKINDEX_NORMALIZATION}"
            )
        if self.draws is not None:
            if not self.draws.index.equals(self.series.index):
                raise ValueError(
                    "RiskIndex.draws.index must equal series.index "
                    f"(got draws.index of length {len(self.draws.index)}, "
                    f"series.index of length {len(self.series.index)})"
                )
            cols = self.draws.columns
            if not isinstance(cols, pd.MultiIndex) or cols.nlevels != 2 \
                    or list(cols.names) != ["source", "draw_id"]:
                raise ValueError(
                    "RiskIndex.draws.columns must be a 2-level MultiIndex "
                    "with names ['source', 'draw_id']; got "
                    f"{cols!r} (names={getattr(cols, 'names', None)})"
                )
            bad = set(cols.get_level_values("source")) - _VALID_DRAW_SOURCES
            if bad:
                raise ValueError(
                    f"RiskIndex.draws: source tags {sorted(bad)} not in "
                    f"{sorted(_VALID_DRAW_SOURCES)}"
                )

    def diagnostics(self) -> dict:
        s = self.series.dropna()
        return {
            "n_quarters": int(s.shape[0]),
            "mean": float(s.mean()) if s.shape[0] else float("nan"),
            "std": float(s.std()) if s.shape[0] > 1 else float("nan"),
            "first_date": str(s.index.min()) if s.shape[0] else None,
            "last_date": str(s.index.max()) if s.shape[0] else None,
        }

    def to_frame(self) -> pd.DataFrame:
        df = self.series.rename("value").reset_index()
        df.columns = ["qdate", "value"]
        df["country"] = self.country
        df["name"] = self.name
        return df

    def as_instrument(self) -> Instrument:
        from ..instruments import Instrument
        return Instrument(
            series=self.series,
            name=self.name,
            source=f"narrative.indices.{self.method}",
            category="text_index",
            frequency="Q",
            metadata={
                "corpus": self.corpus,
                "language": self.language,
                "method": self.method,
                "normalization": self.normalization,
                **self.metadata,
            },
        )


# --- Signal-quality report (Slice 1: sparsity-only fields populated; the
# kernel_agreement / multilingual_parity / *_sd / corpus_loo_max_swing
# fields wire up in Slice 2 — 0.66.0. The calibration fields use Any
# placeholders here; the BenchmarkScore / EventPanelScore / SurveyScore
# dataclasses arrive in Slice 3 — 0.67.0.) ---
BenchmarkScore = Any   # placeholder, replaced in 0.67.0
EventPanelScore = Any  # placeholder, replaced in 0.67.0
SurveyScore = Any      # placeholder, replaced in 0.67.0


@dataclass
class SignalQualityReport:
    """Signal-quality companion to RiskIndex.

    Slice 1 (0.65.0): sparsity / coverage fields only. Stability and
    calibration fields are declared in the schema (default-None / empty
    dict / None) so Slice 2 (draws) and Slice 3 (calibration) can fill
    them in-place without re-versioning the dataclass.
    """
    n_docs_per_period: pd.Series                       # date -> int
    avg_doc_length:    pd.Series                       # date -> float (tokens)
    coverage_gaps:     list[pd.Period]
    kernel_agreement:       pd.Series | None = None
    multilingual_parity:    pd.Series | None = None
    doc_bootstrap_sd:       pd.Series | None = None
    corpus_loo_max_swing:   pd.Series | None = None
    benchmark_scores:       dict[str, Any] = field(default_factory=dict)
    event_panel:            Any | None     = None
    survey_scores:          dict[str, Any] = field(default_factory=dict)
    metadata:               dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Flatten the report to a single-row DataFrame for cross-index tables.

        Columns:
            mean_n_docs, mean_doc_length, n_coverage_gaps — descriptive stats
                on the sparsity fields (mean across periods; mean_doc_length
                is unweighted across periods, not document-weighted).
            has_kernel_draws, has_multiling, has_doc_boot, has_corpus_loo —
                boolean presence indicators for the Slice-2 stability fields
                (`kernel_agreement`, `multilingual_parity`, `doc_bootstrap_sd`,
                `corpus_loo_max_swing`). The column names are intentional
                meta-presence markers, not the underlying field names.
            n_benchmark_scores, has_event_panel, n_survey_scores — counts /
                presence for the Slice-3 calibration fields.
        """
        return pd.DataFrame([{
            "mean_n_docs":         float(self.n_docs_per_period.mean())
                                    if not self.n_docs_per_period.empty else float("nan"),
            "mean_doc_length":     float(self.avg_doc_length.mean())
                                    if not self.avg_doc_length.empty else float("nan"),
            "n_coverage_gaps":     len(self.coverage_gaps),
            "has_kernel_draws":    self.kernel_agreement is not None,
            "has_multiling":       self.multilingual_parity is not None,
            "has_doc_boot":        self.doc_bootstrap_sd is not None,
            "has_corpus_loo":      self.corpus_loo_max_swing is not None,
            "n_benchmark_scores":  len(self.benchmark_scores),
            "has_event_panel":     self.event_panel is not None,
            "n_survey_scores":     len(self.survey_scores),
        }])


__all__ = ["NarrativeEvent", "NarrativeInstrument", "RiskIndex",
           "SignalQualityReport",
           "BenchmarkScore", "EventPanelScore", "SurveyScore",
           "VALID_KINDS", "VALID_TARGETS_BY_KIND",
           "VALID_TARGETS", "VALID_SIGNS", "VALID_SCORERS",
           "VALID_RISKINDEX_METHODS", "VALID_RISKINDEX_NORMALIZATION"]
