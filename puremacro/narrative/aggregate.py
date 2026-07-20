"""Event-level data → quarterly time series.

Handles target filtering, sign convention, GDP normalisation, and
multiple aggregation rules (sum / max / first / mean) over each
(country, quarter, target) bucket.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .types import NarrativeEvent, VALID_KINDS


_AGG_RULE_BY_KIND = {
    "fiscal": "magnitude_sum",
    "monetary": "magnitude_sum",
    "macropru": "signed_count",
    "fx": "signed_count",
    "structural": "indicator",
}


def events_to_quarterly(
    events: Iterable[NarrativeEvent],
    *,
    target_filter: str | None = None,
    kind_filter: str | None = None,
    aggregation: str = "sum",
    sign_weighted: bool = True,
    pct_gdp: pd.DataFrame | None = None,
    freq: str = "QS",
    confidence_threshold: float = 0.0,
    iv_kind: str = "implementation",
) -> pd.Series:
    """Aggregate narrative events into a quarterly IV.

    Parameters
    ----------
    events : iterable of ``NarrativeEvent``.
    target_filter : keep events whose ``target`` matches; ``None`` =
        keep all (events with ``target="both"`` always kept regardless).
    kind_filter : str | None
        Filter events by ``kind`` (fiscal | monetary | macropru | fx |
        structural). If ``None`` and the event list contains multiple
        kinds, ``ValueError`` is raised. ``None`` on a single-kind list
        is fine (backward compatible with all-fiscal callers).
    aggregation : ``"sum"`` (default) | ``"max"`` (largest |signed
        magnitude|) | ``"first"`` (earliest in quarter) | ``"mean"``.
    sign_weighted : if True (default), use ``magnitude × sign`` so
        contractionary events enter negatively.
    pct_gdp : optional ``(date, country) → gdp_value`` DataFrame for
        normalising magnitudes to percent of GDP. If supplied, must
        cover the events' date range.
    freq : pandas frequency string for the output index.
    confidence_threshold : drop events with ``confidence`` strictly less
        than this (default 0 = keep everything).
    iv_kind : ``"implementation"`` (default) distributes signed magnitude
        over each event's ``effective_profile``. ``"announcement"`` puts
        the full signed magnitude in the announcement quarter (legacy
        behavior); profiles are ignored in this mode.

    Aggregation rule depends on the surviving kind:
      - fiscal, monetary: sum of signed magnitudes (existing rule).
      - macropru, fx:    signed count of actions per quarter.
      - structural:      sign of the dominant event in the quarter.

    Returns
    -------
    pd.Series indexed by quarter-start dates, name ``"narrative_iv"``.
    Quarters with no events get 0.
    """
    if iv_kind not in {"announcement", "implementation"}:
        raise ValueError(
            f"iv_kind must be 'announcement' or 'implementation'; "
            f"got {iv_kind!r}"
        )
    if kind_filter is not None and kind_filter not in VALID_KINDS:
        raise ValueError(
            f"kind_filter {kind_filter!r} not in {VALID_KINDS}"
        )

    events_list = list(events)
    if not events_list:
        return pd.Series(dtype=float, name="narrative_iv")

    kinds_present = {e.kind for e in events_list}
    if kind_filter is None and len(kinds_present) > 1:
        raise ValueError(
            f"events_to_quarterly: events have multiple kinds "
            f"{sorted(kinds_present)}; pass kind_filter= to disambiguate"
        )
    if kind_filter is not None:
        events_list = [e for e in events_list if e.kind == kind_filter]
        if not events_list:
            return pd.Series(dtype=float, name="narrative_iv")
        kinds_present = {kind_filter}

    surviving_kind = next(iter(kinds_present))
    agg_rule = _AGG_RULE_BY_KIND[surviving_kind]

    def _keep(e: NarrativeEvent) -> bool:
        if e.confidence < confidence_threshold:
            return False
        if target_filter is None:
            return True
        return e.target == target_filter or e.target == "both"

    filtered = [e for e in events_list if _keep(e)]
    if not filtered:
        return pd.Series(dtype=float, name="narrative_iv")

    rows = []
    for e in filtered:
        if agg_rule == "magnitude_sum":
            v = e.signed_magnitude if sign_weighted else e.magnitude
        elif agg_rule == "signed_count":
            v = float(e.sign)
        elif agg_rule == "indicator":
            v = float(e.sign)
        else:  # pragma: no cover
            raise AssertionError(agg_rule)
        if iv_kind == "announcement" or agg_rule != "magnitude_sum":
            rows.append({"date": e.date, "country": e.country, "value": float(v)})
        else:
            for d, w in e.effective_profile:
                rows.append({"date": d, "country": e.country,
                             "value": float(v) * float(w)})

    df = pd.DataFrame(rows)

    if pct_gdp is not None and agg_rule == "magnitude_sum":
        df["q_date"] = df["date"].dt.to_period("Q").dt.to_timestamp()
        gdp_long = pct_gdp.stack().rename("gdp").reset_index()
        gdp_long.columns = ["q_date", "country", "gdp"]
        df = df.merge(gdp_long, on=["q_date", "country"], how="left")
        df["value"] = df["value"] / df["gdp"]

    df["q_date"] = df["date"].dt.to_period("Q").dt.to_timestamp()

    if agg_rule == "indicator":
        idx = df.groupby("q_date")["value"].apply(lambda s: s.abs().idxmax())
        out = df.loc[idx].set_index("q_date")["value"].apply(np.sign)
    elif agg_rule == "signed_count":
        out = df.groupby("q_date")["value"].sum()
    elif aggregation == "sum":
        out = df.groupby("q_date")["value"].sum()
    elif aggregation == "mean":
        out = df.groupby("q_date")["value"].mean()
    elif aggregation == "max":
        idx = df.groupby("q_date")["value"].apply(lambda s: s.abs().idxmax())
        out = df.loc[idx].set_index("q_date")["value"]
    elif aggregation == "first":
        df_sorted = df.sort_values("date")
        out = df_sorted.groupby("q_date")["value"].first()
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")

    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=freq)
    out = out.reindex(full_idx, fill_value=0.0)
    out.index.name = "date"
    out.name = "narrative_iv"
    return out


from .types import RiskIndex
from ._signal_quality import compute_sparsity_report


def index_to_quarterly(
    records,
    *,
    kernel,
    country: str,
    language: str,
    name: str,
    method: str,
    corpus: str,
    normalization: str,
    freq: str = "QS",
    agg: str = "mean",
    weight_by: str | None = None,
    metadata: dict | None = None,
    with_quality: bool = False,
) -> RiskIndex:
    """Aggregate per-document kernel scores into a quarterly RiskIndex.

    Parameters
    ----------
    records : iterable of records (4-tuple or 5-tuple — see
        :mod:`puremacro.narrative.sources._schema`).
    kernel  : callable(records) → iterable of ``(pd.Timestamp, float)`` points.
    country, language, name, method, corpus, normalization : passed to RiskIndex.
    freq    : pandas frequency string for the output index (default "QS").
    agg     : "mean" | "max" | "dispersion" | "sum_weighted".
    weight_by : when ``agg="sum_weighted"``, must be ``"magnitude"``. Each
        record's 5th-slot ``magnitude`` (or 1.0 if absent / None) is used
        as the per-record weight on the kernel's score.
    metadata : optional extra metadata. ``n_docs`` is always added.
    with_quality : if True, compute a sparsity-only ``SignalQualityReport``
        from ``records_list`` and attach it to the returned ``RiskIndex``
        as ``ri.quality``. Default False preserves the 0.64.0 behaviour
        (``ri.quality is None``).

    Returns
    -------
    RiskIndex with quarterly-indexed series.
    """
    if agg not in {"mean", "max", "dispersion", "sum_weighted"}:
        raise ValueError(f"agg must be mean|max|dispersion|sum_weighted; got {agg!r}")
    if agg == "sum_weighted" and weight_by != "magnitude":
        raise ValueError(
            "agg='sum_weighted' requires weight_by='magnitude'; "
            f"got weight_by={weight_by!r}"
        )

    # Materialise records once so we can read magnitudes alongside kernel output.
    records_list = list(records)

    # Build a per-record magnitude vector (only used when weight_by='magnitude').
    magnitudes: list[float] = []
    if weight_by == "magnitude":
        for r in records_list:
            if len(r) >= 5 and r[4] is not None:
                magnitudes.append(float(r[4]))
            else:
                magnitudes.append(1.0)

    points = list(kernel(records_list))
    if not points:
        raise ValueError(
            f"index_to_quarterly: no documents in corpus={corpus!r} "
            f"for country={country!r}"
        )

    if weight_by == "magnitude" and len(magnitudes) != len(points):
        # Kernels may drop docs; fall back to uniform weight when lengths
        # don't align.
        magnitudes = [1.0] * len(points)

    df = pd.DataFrame(points, columns=["date", "value"])
    if weight_by == "magnitude":
        df["weight"] = magnitudes
    df["q_date"] = df["date"].dt.to_period("Q").dt.to_timestamp()

    if agg == "mean":
        out = df.groupby("q_date")["value"].mean()
    elif agg == "max":
        out = df.groupby("q_date")["value"].max()
    elif agg == "dispersion":
        out = df.groupby("q_date")["value"].std().fillna(0.0)
    else:  # sum_weighted
        out = df.groupby("q_date").apply(
            lambda g: (g["value"] * g["weight"]).sum()
        )

    full_idx = pd.date_range(out.index.min(), out.index.max(), freq=freq)
    out = out.reindex(full_idx)
    out.index.name = "date"
    out.name = name

    full_metadata = {"n_docs": len(points)}
    if metadata:
        full_metadata.update(metadata)

    if normalization != "raw":
        from .indices._kernels import normalize_series
        bp = (metadata or {}).get("base_period") if metadata else None
        out = normalize_series(out, normalization, base_period=bp)

    quality = compute_sparsity_report(records_list) if with_quality else None

    return RiskIndex(
        name=name, country=country, series=out,
        method=method, corpus=corpus, language=language,
        normalization=normalization, metadata=full_metadata,
        quality=quality,
    )


__all__ = ["events_to_quarterly", "index_to_quarterly"]
