"""Ratio-splicing: joining national-accounts vintages without inventing data.

A long quarterly panel is built from segments that no statistical office
ever published together — Spain's base-1986, base-1995 and current
Contabilidad Nacional Trimestral; Japan's 68SNA, 93SNA and 2008SNA. The
segments overlap, and in the overlap they disagree, because each is a
different vintage of the same economy on a different base.

WHAT A SPLICE MAY AND MAY NOT DO
--------------------------------
The one thing worth preserving from an old vintage is its **growth
rates**. Its levels are expressed in a base and a methodology that were
later abandoned, so carrying them over unchanged would put a step in
the series at the seam. So the older segment is rescaled by the ratio
between the two vintages over their overlap, and only then used to
extend the newer one backwards. Growth rates of the older segment
survive untouched; its levels do not, and are not meant to.

THE RATIO'S *STABILITY* IS THE TEST
-----------------------------------
If two vintages differ by a constant factor over the overlap, they
agree about growth and disagree only about level: the splice is sound
and the choice of anchor quarter does not matter. If the ratio drifts,
they disagree about growth itself, and no rescaling can reconcile them
— the spliced level then depends materially on which quarter you
anchored to, which is a result about your arbitrary choice rather than
about the economy.

So :func:`ratio_splice` measures that drift and reports it, and warns
when it exceeds :data:`RATIO_DRIFT_WARN`. This is not decoration. It is
what separates Spain's base-1995 join, whose ratio is near-constant
over 40 quarters, from CEPALSTAT's Mexican join, whose ratio wanders
from 0.75 to 0.84 across the overlap — the latter cannot support a
splice and this module says so instead of returning a number.

WHAT THIS MODULE DELIBERATELY WILL NOT DO
-----------------------------------------
It will not splice across a change in what the country *is*. German
national accounts before 1991 cover West Germany; a ratio-splice onto
unified Germany would silently fabricate an East German economy back to
1970. That is a definitional break, not a revision, and
:func:`ratio_splice` refuses it unless the caller passes
``allow_definitional_break=True`` and thereby takes responsibility.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


#: Coefficient of variation of the overlap ratio above which a splice is
#: reported as unstable. 2% is generous for a pure rebasing (which is
#: exact up to rounding) and tight enough to catch a vintage join whose
#: two sides disagree about growth.
RATIO_DRIFT_WARN = 0.02

#: Minimum overlapping quarters before a ratio is considered estimable.
#: One quarter gives a ratio with no way to tell whether it is stable.
MIN_OVERLAP = 4


@dataclass(frozen=True)
class Seam:
    """One join between two segments, and how well it is determined."""
    date: pd.Timestamp
    older: str
    newer: str
    overlap_n: int
    ratio: float
    ratio_drift: float
    ratio_min: float
    ratio_max: float
    stable: bool
    note: str = ""

    def __str__(self) -> str:                            # pragma: no cover
        flag = "" if self.stable else "  UNSTABLE"
        return (f"{self.older} -> {self.newer} at {self.date.date()}: "
                f"ratio {self.ratio:.4f} over {self.overlap_n}q, "
                f"drift {self.ratio_drift:.3%}{flag}")


@dataclass
class SpliceResult:
    """A spliced series, with the provenance and seams that produced it."""
    series: pd.Series
    provenance: pd.Series
    seams: list[Seam] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        """True when every seam's ratio held steady across its overlap."""
        return all(s.stable for s in self.seams)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.series,
                             "source": self.provenance})


def overlap_ratio(
    older: pd.Series, newer: pd.Series, *, min_overlap: int = MIN_OVERLAP,
) -> tuple[float, int, float, float, float]:
    """Ratio of ``newer`` to ``older`` across their common index.

    Returns ``(ratio, n, drift, lo, hi)`` where ``ratio`` is the mean of
    the pointwise ratios, ``drift`` their coefficient of variation, and
    ``lo``/``hi`` their range. The mean is used rather than a single
    anchor quarter precisely so that one revised quarter cannot move the
    whole backcast.
    """
    common = older.index.intersection(newer.index)
    if len(common) == 0:
        return float("nan"), 0, float("nan"), float("nan"), float("nan")
    a = pd.to_numeric(older.reindex(common), errors="coerce")
    b = pd.to_numeric(newer.reindex(common), errors="coerce")
    both = pd.concat([a, b], axis=1).dropna()
    both = both[both.iloc[:, 0] != 0]
    n = int(len(both))
    if n == 0:
        return float("nan"), 0, float("nan"), float("nan"), float("nan")
    r = (both.iloc[:, 1] / both.iloc[:, 0]).to_numpy(dtype=float)
    mean = float(np.mean(r))
    drift = float(np.std(r, ddof=1) / abs(mean)) if n > 1 and mean != 0 else 0.0
    return mean, n, drift, float(np.min(r)), float(np.max(r))


def ratio_splice(
    segments: list[tuple[str, pd.Series]],
    *,
    min_overlap: int = MIN_OVERLAP,
    drift_warn: float = RATIO_DRIFT_WARN,
    definitional_breaks: dict[str, str] | None = None,
    allow_definitional_break: bool = False,
) -> SpliceResult:
    """Splice segments onto the newest one, preserving its levels.

    Parameters
    ----------
    segments : ``[(label, series), ...]`` ordered **newest first**. The
        first is the spine: its levels are kept exactly as published,
        and every older segment is rescaled onto it.
    min_overlap : refuse to estimate a ratio from fewer overlapping
        quarters than this. A splice with no overlap is not a splice; it
        is two series concatenated at a step.
    definitional_breaks : ``{label: reason}``. A segment named here
        describes a different entity from its successor (West Germany
        vs unified Germany, say). Splicing it is refused unless
        ``allow_definitional_break``.

    Returns
    -------
    SpliceResult

    Raises
    ------
    ValueError
        If a segment cannot be joined — no overlap, too little overlap,
        or a definitional break the caller has not accepted. Returning a
        silently concatenated series would be far worse: it looks like
        history and contains a step nobody published.
    """
    segments = [(lab, s) for lab, s in segments
                if s is not None and len(s.dropna())]
    if not segments:
        return SpliceResult(pd.Series(dtype=float),
                            pd.Series(dtype=object), [])

    breaks = definitional_breaks or {}
    spine_label, spine = segments[0]
    spine = spine.dropna().sort_index()
    out = spine.copy()
    prov = pd.Series(spine_label, index=spine.index, dtype=object)
    seams: list[Seam] = []

    for older_label, older_raw in segments[1:]:
        older = older_raw.dropna().sort_index()
        if older.empty:
            continue
        ratio, n, drift, lo, hi = overlap_ratio(older, out,
                                                min_overlap=min_overlap)
        if n == 0:
            raise ValueError(
                f"cannot splice {older_label!r} onto {spine_label!r}: the "
                "two segments do not overlap at all, so there is nothing "
                "to estimate a level ratio from. Concatenating them would "
                "put an unmeasured step in the series."
            )
        if n < min_overlap:
            raise ValueError(
                f"cannot splice {older_label!r}: only {n} overlapping "
                f"quarter(s), need {min_overlap}. With so few, the ratio "
                "cannot be distinguished from a one-off revision."
            )
        if older_label in breaks and not allow_definitional_break:
            raise ValueError(
                f"refusing to splice {older_label!r}: {breaks[older_label]} "
                "This is a change in what is being measured, not a "
                "revision of it, so rescaling would fabricate history for "
                "an entity that did not exist. Pass "
                "allow_definitional_break=True to override, and say so in "
                "whatever you publish."
            )

        stable = bool(np.isfinite(drift) and drift <= drift_warn)
        if not stable:
            warnings.warn(
                f"splice {older_label!r} -> {spine_label!r}: the vintage "
                f"ratio drifts {drift:.2%} across {n} overlapping quarters "
                f"(range {lo:.4f}-{hi:.4f}). The two vintages disagree "
                "about growth, not just level, so the spliced level "
                "depends on the anchor. Treat the older segment as "
                "indicative.",
                UserWarning, stacklevel=2,
            )

        rescaled = older * ratio
        new_index = rescaled.index.difference(out.index)
        if len(new_index):
            out = pd.concat([rescaled.reindex(new_index), out]).sort_index()
            prov = pd.concat([
                pd.Series(older_label, index=new_index, dtype=object), prov,
            ]).sort_index()
            seam_date = out.index[out.index.get_indexer([new_index.max()])[0] + 1] \
                if new_index.max() < out.index.max() else new_index.max()
        else:
            seam_date = older.index.max()

        seams.append(Seam(
            date=pd.Timestamp(seam_date), older=older_label,
            newer=spine_label, overlap_n=n, ratio=ratio, ratio_drift=drift,
            ratio_min=lo, ratio_max=hi, stable=stable,
            note="" if stable else "ratio drifts across the overlap",
        ))

    return SpliceResult(series=out.sort_index(),
                        provenance=prov.sort_index(), seams=seams)


def splice_frame(
    segments: list[tuple[str, pd.DataFrame]],
    *,
    columns: list[str] | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[Seam]]]:
    """Column-wise :func:`ratio_splice` over aligned frames.

    Every column is spliced **independently**, each keeping its own
    source's growth rates. They are deliberately not forced to add up:
    see :func:`expenditure_residual`, which measures how badly the
    accounting identity fails rather than hiding it.

    Returns ``(values, provenance, seams_by_column)``. A column that
    cannot be spliced is carried from the spine alone and its failure is
    recorded under its name in ``seams_by_column`` as an empty list.
    """
    if not segments:
        return pd.DataFrame(), pd.DataFrame(), {}
    cols = columns or list(segments[0][1].columns)
    values, prov, seams = {}, {}, {}
    for col in cols:
        parts = [(lab, df[col]) for lab, df in segments if col in df.columns]
        if not parts:
            continue
        try:
            res = ratio_splice(parts, **kwargs)
        except ValueError as exc:
            spine_label, spine_df = segments[0]
            values[col] = spine_df[col]
            prov[col] = pd.Series(spine_label, index=spine_df.index,
                                  dtype=object)
            seams[col] = []
            warnings.warn(
                f"column {col!r} kept from {spine_label!r} only: {exc}",
                UserWarning, stacklevel=2,
            )
            continue
        values[col] = res.series
        prov[col] = res.provenance
        seams[col] = res.seams
    return (pd.DataFrame(values).sort_index(),
            pd.DataFrame(prov).sort_index(), seams)


def expenditure_residual(
    frame: pd.DataFrame,
    *,
    gdp: str = "gdp",
    plus: tuple[str, ...] = ("cons_hh", "cons_gov", "capform", "exports"),
    minus: tuple[str, ...] = ("imports",),
) -> pd.Series:
    """``gdp - (C + G + I + X - M)``, per period.

    Because the columns are spliced independently, this does **not**
    come back as zero, and it is not supposed to. Its size is the honest
    measure of how much the splice — and the underlying source data —
    fail to add up. A residual that is small relative to GDP before the
    seam and jumps after it is telling you the splice hurt.
    """
    missing = [c for c in (gdp, *plus, *minus) if c not in frame.columns]
    if missing:
        raise ValueError(
            f"cannot form the expenditure residual: missing {missing}. "
            f"Frame has {sorted(frame.columns)}"
        )
    total = frame[list(plus)].sum(axis=1, min_count=len(plus))
    for c in minus:
        total = total - frame[c]
    return frame[gdp] - total


__all__ = [
    "RATIO_DRIFT_WARN",
    "MIN_OVERLAP",
    "Seam",
    "SpliceResult",
    "overlap_ratio",
    "ratio_splice",
    "splice_frame",
    "expenditure_residual",
]
