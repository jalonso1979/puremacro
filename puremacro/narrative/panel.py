"""Build the homogeneous cross-country fiscal narrative-IV panel.

Every event in the panel obeys the same conventions:

    * date           : quarterly timestamp (start-of-quarter)
    * country        : ISO3
    * magnitude_unit : ``"pct_gdp"`` (absolute size in % of GDP)
    * sign           : ``+1`` expansionary / ``-1`` contractionary
    * target         : ``investment`` / ``consumption`` / ``both``
    * subtarget      : ``defense`` / ``infra`` / ``transfers`` / ``tax``
                          / ``expenditure`` / ``rd`` / ``general``
    * provenance     : ``replication`` field in metadata identifies the
                       source dataset (DGLP / RR / MR / Cloyne / Ramey / …)

Stacking the canonical replication series produces the deterministic
17-country baseline. The optional ``add_llm_scored`` step augments it
with LLM-scored events extracted from IMF Article IV abstracts, OECD
fiscal Working Papers, and national press feeds — multiplying coverage
to the full 33-country wedges panel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .aggregate import events_to_quarterly
from .dedup import deduplicate
from .types import NarrativeEvent, NarrativeInstrument


# Default subset: the 17 DGLP advanced economies (the maximal panel
# that has at least one canonical narrative dataset per country).
DGLP_17 = [
    "AUS", "AUT", "BEL", "CAN", "DNK", "FIN", "FRA", "DEU", "IRL",
    "ITA", "JPN", "NLD", "PRT", "ESP", "SWE", "GBR", "USA",
]


@dataclass
class HomogeneousFiscalPanel:
    """A harmonised cross-country fiscal-narrative-IV panel.

    Construct with :meth:`from_canonical` (deterministic, no API keys)
    and optionally augment with :meth:`add_llm_scored`.
    """
    events: list[NarrativeEvent] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    audit: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------
    @classmethod
    def from_canonical(
        cls,
        countries: Iterable[str] | None = None,
        *,
        include_adler_2024: bool = True,
        include_dglp: bool = True,
        include_rr2017: bool = True,
        include_rr2010_us: bool = True,
        include_mr2013_us: bool = True,
        include_cloyne_uk: bool = True,
        include_ramey_us: bool = True,
        include_devries: bool = True,
        include_guajardo: bool = True,
        include_mitchell: bool = False,
        include_eu_nms_2023: bool = True,
        include_imf_covid_2022: bool = True,
        date_tol_days: int = 90,
        canonical_csv_paths: dict[str, str] | None = None,
        within_year_rule: str = "uniform",
    ) -> "HomogeneousFiscalPanel":
        """Stack the canonical replication series and deduplicate.

        Parameters
        ----------
        countries : ISO3 codes; defaults to the 17 DGLP economies.
        include_* : on/off switches for each canonical dataset. The
            default (everything on) gives the maximally-covered panel.
        canonical_csv_paths : optional ``{name: csv_path}`` overrides
            for the loaders, e.g. ``{"dglp_2011": "/path/to/dglp.csv"}``.
            Without overrides, the loaders will try their public mirror
            and fall back to a clear error pointing at the author's
            replication archive.
        """
        from .replication import (
            load_adler_2024,
            load_dglp_2011, load_romer_romer_2017,
            load_romer_romer_2010, load_mertens_ravn_2013,
            load_cloyne_2013_uk, load_ramey_2011_defense,
            load_devries_2011, load_guajardo_2011,
            load_mitchell_historical,
            load_eu_nms_2023,
            load_imf_covid_2022,
        )
        codes = [c.upper() for c in (countries or DGLP_17)]
        csvs = canonical_csv_paths or {}

        all_events: list[NarrativeEvent] = []
        load_summary: dict[str, dict] = {}

        def _try(name: str, loader):
            try:
                got = loader()
                load_summary[name] = {"ok": True}
                return got
            except Exception as e:
                load_summary[name] = {"ok": False, "error": str(e)[:120]}
                return None

        # ---- 0. Adler 2024 panel: OECD-17 + LATAM/Caribbean-14, supersedes
        # DGLP within the canonical tier (priority-1 — loaded first so dedup
        # representative() picks Adler over DGLP on overlap).
        if include_adler_2024:
            adler = _try("adler_2024", lambda: load_adler_2024(
                countries=codes,
                csv_path=csvs.get("adler_2024"),
            ))
            if adler:
                for c, inst in adler.items():
                    all_events.extend(inst.events)

        # ---- 1. DGLP 2011 panel: 17 countries, tax + expenditure cuts.
        if include_dglp:
            dglp = _try("dglp_2011", lambda: load_dglp_2011(
                countries=codes,
                csv_path=csvs.get("dglp_2011"),
                within_year_rule=within_year_rule,
            ))
            if dglp:
                for c, inst in dglp.items():
                    all_events.extend(inst.events)

        # ---- 2. Romer-Romer 2017 cross-country tax shocks.
        if include_rr2017:
            rr2017 = _try("rr_2017", lambda: load_romer_romer_2017(
                countries=codes,
                csv_path=csvs.get("rr_2017"),
            ))
            if rr2017:
                for c, inst in rr2017.items():
                    all_events.extend(inst.events)

        # ---- 3. US-specific canonicals (RR2010, MR2013, Ramey 2011).
        if "USA" in codes:
            if include_rr2010_us:
                rr2010 = _try("rr_2010", lambda: load_romer_romer_2010(
                    csv_path=csvs.get("rr_2010"),
                ))
                if rr2010:
                    all_events.extend(rr2010.events)
            if include_mr2013_us:
                mr2013 = _try("mr_2013", lambda: load_mertens_ravn_2013(
                    kind="unanticipated",
                    csv_path=csvs.get("mr_2013"),
                ))
                if mr2013:
                    all_events.extend(mr2013.events)
            if include_ramey_us:
                ramey = _try("ramey_2011", lambda: load_ramey_2011_defense(
                    csv_path=csvs.get("ramey_2011"),
                ))
                if ramey:
                    all_events.extend(ramey.events)

        # ---- 4. UK Cloyne 2013.
        if include_cloyne_uk and "GBR" in codes:
            cloyne = _try("cloyne_2013", lambda: load_cloyne_2013_uk(
                csv_path=csvs.get("cloyne_2013"),
            ))
            if cloyne:
                all_events.extend(cloyne.events)

        # ---- 5. Devries 2011 long-format.
        if include_devries:
            devries = _try("devries_2011", lambda: load_devries_2011(
                countries=codes,
                csv_path=csvs.get("devries_2011"),
                within_year_rule=within_year_rule,
            ))
            if devries:
                for c, inst in devries.items():
                    all_events.extend(inst.events)

        # ---- 6. Guajardo 2011 (advanced + 11 EMs).
        if include_guajardo:
            guajardo = _try("guajardo_2011", lambda: load_guajardo_2011(
                countries=codes,
                csv_path=csvs.get("guajardo_2011"),
                within_year_rule=within_year_rule,
            ))
            if guajardo:
                for c, inst in guajardo.items():
                    all_events.extend(inst.events)

        # ---- 7. Mitchell historical (pre-event annual anchors).
        if include_mitchell:
            mitchell = _try("mitchell_2013", lambda: load_mitchell_historical(
                countries=codes,
                csv_path=csvs.get("mitchell_2013"),
            ))
            if mitchell:
                for c, inst in mitchell.items():
                    all_events.extend(inst.events)

        # ---- 9. EU NMS 2023: 11 EU New Member States, 2004-2019.
        # No overlap with DGLP-OECD; loaded last in the canonical tier.
        if include_eu_nms_2023:
            eu_nms = _try("eu_nms_2023", lambda: load_eu_nms_2023(
                countries=codes,
                csv_path=csvs.get("eu_nms_2023"),
            ))
            if eu_nms:
                for c, inst in eu_nms.items():
                    all_events.extend(inst.events)

        # ---- 10. IMF COVID 2022 announcement-level database.
        # Different time period (2020) — no overlap with DGLP/Adler/EU NMS.
        if include_imf_covid_2022:
            covid = _try("imf_covid_2022", lambda: load_imf_covid_2022(
                countries=codes,
                csv_path=csvs.get("imf_covid_2022"),
            ))
            if covid:
                for c, inst in covid.items():
                    all_events.extend(inst.events)

        # ---- 8. Deduplicate across overlapping datasets.
        reps, audit = deduplicate(all_events, date_tol_days=date_tol_days)

        return cls(
            events=reps,
            countries=sorted({ev.country for ev in reps}),
            audit=audit,
            metadata={
                "load_summary": load_summary,
                "n_raw_events": len(all_events),
                "n_after_dedup": len(reps),
                "date_tol_days": date_tol_days,
            },
        )

    # ---------------------------------------------------------------
    # Optional LLM augmentation
    # ---------------------------------------------------------------
    def add_llm_scored(
        self,
        corpus_iter,
        *,
        backend,
        country: str,
        date_tol_days: int = 90,
    ) -> "HomogeneousFiscalPanel":
        """Augment the panel with LLM-scored events from a text corpus.

        The new events are appended, then the entire panel is
        re-deduplicated so canonical anchors take precedence.

        Parameters
        ----------
        corpus_iter : iterable of ``(date, text, source_url)`` tuples;
            typically a source connector from :mod:`puremacro.narrative.sources`.
        backend : an :class:`AnthropicBackend` or :class:`OpenAIBackend`
            instance from :mod:`puremacro.narrative.scoring.llm`.
        country : ISO3 the events should be tagged with.
        date_tol_days : passed through to deduplication.
        """
        from .scoring.llm import score_llm
        new_events = score_llm(corpus_iter, backend=backend, country=country)
        all_events = self.events + new_events
        reps, audit = deduplicate(all_events, date_tol_days=date_tol_days)
        return HomogeneousFiscalPanel(
            events=reps,
            countries=sorted({e.country for e in reps}),
            audit=audit,
            metadata={
                **self.metadata,
                "n_llm_added": len(new_events),
                "n_after_llm_dedup": len(reps),
            },
        )

    # ---------------------------------------------------------------
    # Quality pipeline
    # ---------------------------------------------------------------
    def apply_quality_pipeline(
        self,
        *,
        surprise_filter: bool = True,
        harmonize_magnitudes: bool = True,
        audit_targets: bool = True,
        llm_backend=None,
        confidence_floor: float = 0.5,
        decay_factor: float = 0.25,
        surprise_confidence_floor: float = 0.1,
    ) -> "HomogeneousFiscalPanel":
        """Run quality filters and return a new panel.

        Parameters
        ----------
        surprise_filter : apply budget-calendar and phase-year decay.
        harmonize_magnitudes : convert to real 2015 % of GDP.
        audit_targets : LLM re-classify ``target="both"`` events (requires
            ``llm_backend``).
        llm_backend : ``AnthropicBackend`` or ``OpenAIBackend`` instance;
            required only when ``audit_targets=True``.
        confidence_floor : drop events with confidence below this after all
            filters.
        decay_factor : multiplier for anticipated-event confidence decay.
        """
        from .quality.pipeline import run_pipeline
        filtered, qmeta, audit_df = run_pipeline(
            list(self.events),
            surprise_filter=surprise_filter,
            harmonize_magnitudes=harmonize_magnitudes,
            audit_targets=audit_targets,
            llm_backend=llm_backend,
            confidence_floor=confidence_floor,
            decay_factor=decay_factor,
            surprise_confidence_floor=surprise_confidence_floor,
        )
        # Quality pipeline produces its own audit schema (target reclassifications).
        # Don't concat with the dedup audit — they have incompatible schemas.
        new_audit = audit_df if not audit_df.empty else self.audit
        return HomogeneousFiscalPanel(
            events=filtered,
            countries=sorted({e.country for e in filtered}),
            audit=new_audit if (new_audit is not None and not new_audit.empty) else None,
            metadata={**self.metadata, "quality_pipeline": qmeta},
        )

    def specification_curve(
        self,
        df,
        *,
        y: str,
        x_invest: str,
        x_consume: str,
        controls=None,
        horizon: int = 8,
        n_boot: int = 500,
        seed: int = 42,
        max_specs=None,
    ):
        """Build a NarrativeSpecificationCurve for this panel.

        Parameters
        ----------
        df : quarterly macro DataFrame indexed by dates.
        y : outcome variable column name.
        x_invest : endogenous variable for investment instrument specs.
        x_consume : endogenous variable for consumption instrument specs.
        horizon : LP-IV max horizon.
        n_boot : bootstrap draws (0 = skip, HAC SE only).
        max_specs : subsample size (None = all 1728 specs).

        Returns
        -------
        NarrativeSpecificationCurve (call .run() to execute).
        """
        from .validation.spec_curve import NarrativeSpecificationCurve
        return NarrativeSpecificationCurve(
            panel=self,
            df=df,
            y=y,
            x_invest=x_invest,
            x_consume=x_consume,
            horizon=horizon,
            n_boot=n_boot,
            max_specs=max_specs,
            seed=seed,
        )

    # ---------------------------------------------------------------
    # Output / exporters
    # ---------------------------------------------------------------
    def to_long(self) -> pd.DataFrame:
        """Long-form DataFrame, one row per event."""
        rows = []
        for e in self.events:
            prof = e.effective_profile
            first_impl = min(d for d, _ in prof)
            last_impl = max(d for d, _ in prof)
            r = {
                "country":         e.country,
                "date":            e.date,
                "magnitude":       e.magnitude,
                "magnitude_unit":  e.magnitude_unit,
                "sign":            e.sign,
                "target":          e.target,
                "subtarget":       e.subtarget,
                "confidence":      e.confidence,
                "scoring_method":  e.scoring_method,
                "source_url":      e.source_url,
                "replication":     e.metadata.get("replication", ""),
                "implementation_first_quarter": first_impl,
                "implementation_last_quarter":  last_impl,
                "delay_quarters":  e.delay_quarters,
            }
            rows.append(r)
        if not rows:
            return pd.DataFrame(columns=[
                "country", "date", "magnitude", "magnitude_unit", "sign",
                "target", "subtarget", "confidence", "scoring_method",
                "source_url", "replication",
                "implementation_first_quarter", "implementation_last_quarter",
                "delay_quarters",
            ])
        return pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)

    def to_quarterly_panel(
        self,
        *,
        target: str | None = None,
        aggregation: str = "sum",
        sign_weighted: bool = True,
        freq: str = "QS",
        iv_kind: str = "implementation",
    ) -> pd.DataFrame:
        """Wide (date × country) quarterly panel of signed magnitudes
        in % of GDP. Drops zero-only countries.
        """
        per_country: dict[str, pd.Series] = {}
        for code in self.countries:
            sub_events = [e for e in self.events if e.country == code]
            if target is not None:
                sub_events = [
                    e for e in sub_events
                    if e.target == target or e.target == "both"
                ]
            if not sub_events:
                continue
            q = events_to_quarterly(
                sub_events, target_filter=None, aggregation=aggregation,
                sign_weighted=sign_weighted, freq=freq, iv_kind=iv_kind,
            )
            per_country[code] = q
        if not per_country:
            return pd.DataFrame()
        # Align on common date index.
        df = pd.concat(per_country, axis=1).fillna(0.0)
        df.columns = list(per_country)
        df.index.name = "date"
        return df

    def summary(self) -> pd.DataFrame:
        """Per-country event counts + date span + per-source breakdown."""
        long = self.to_long()
        if long.empty:
            return pd.DataFrame()
        rows = []
        for code in sorted(long["country"].unique()):
            sub = long[long["country"] == code]
            rows.append({
                "country": code,
                "n_events": len(sub),
                "first_date": sub["date"].min(),
                "last_date":  sub["date"].max(),
                "by_replication": dict(sub["replication"].value_counts()),
                "by_target": dict(sub["target"].value_counts()),
                "by_sign":   dict(sub["sign"].value_counts()),
                "median_magnitude_pct_gdp": float(sub["magnitude"].median()),
            })
        return pd.DataFrame(rows)

    def cross_correlations(
        self,
        *,
        target: str | None = None,
    ) -> pd.DataFrame:
        """Pairwise cross-country correlations of the quarterly panel.

        Useful sanity check: countries with simultaneous euro-area
        austerity should comove (DEU/FRA/ITA/ESP after 2010).
        """
        wide = self.to_quarterly_panel(target=target)
        if wide.empty:
            return pd.DataFrame()
        return wide.corr()

    def save(self, out_dir: str | Path = ".") -> dict[str, Path]:
        """Persist the panel to disk in three formats."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        long = self.to_long()
        wide_all = self.to_quarterly_panel()
        wide_inv = self.to_quarterly_panel(target="investment")
        wide_con = self.to_quarterly_panel(target="consumption")
        long_path = out_dir / "narrative_iv_panel_long.csv"
        wide_path = out_dir / "narrative_iv_panel_quarterly.csv"
        wide_inv_path = out_dir / "narrative_iv_panel_investment.csv"
        wide_con_path = out_dir / "narrative_iv_panel_consumption.csv"
        long.to_csv(long_path, index=False)
        wide_all.to_csv(wide_path)
        if not wide_inv.empty:
            wide_inv.to_csv(wide_inv_path)
        if not wide_con.empty:
            wide_con.to_csv(wide_con_path)
        if self.audit is not None and not self.audit.empty:
            audit_path = out_dir / "narrative_iv_panel_dedup_audit.csv"
            self.audit.to_csv(audit_path, index=False)
        return {
            "long": long_path,
            "wide_all": wide_path,
            "wide_investment": wide_inv_path,
            "wide_consumption": wide_con_path,
        }


__all__ = ["HomogeneousFiscalPanel", "DGLP_17"]
