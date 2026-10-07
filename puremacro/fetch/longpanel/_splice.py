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

MANY SOURCES AT ONCE
--------------------
:func:`splice_sources` applies the same discipline to a whole long panel
fed by several providers: per ``(code, variable)`` the best source with
enough history is kept as published, and lower-ranked sources may only
extend it backwards through a measured seam (a ratio for levels, a
bounded gap for rates). Every seam it tries, accepted or rejected, comes
back as a row, so precedence is an auditable table rather than a
``combine_first`` that nobody can reconstruct. :func:`to_long` turns the
fetchers' wide ``(code, date)`` frames into its input.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings
from dataclasses import asdict, dataclass, field

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


def _ratio_moments(r: np.ndarray) -> tuple[float, int, float, float, float]:
    """``(mean, n, drift, lo, hi)`` of pointwise ratios ``r`` (non-empty).

    Shared by :func:`overlap_ratio` and :func:`splice_sources`, so a seam
    measured on one series and on a whole panel is the same number.
    """
    n = int(r.size)
    mean = float(np.mean(r))
    drift = float(np.std(r, ddof=1) / abs(mean)) if n > 1 and mean != 0 else 0.0
    return mean, n, drift, float(np.min(r)), float(np.max(r))


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
    return _ratio_moments(r)


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
    values: dict[str, Any] = {}
    prov: dict[str, Any] = {}
    seams: dict[str, Any] = {}
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


# ---------------------------------------------------------------------------
# Many sources, one panel: precedence plus backward extension
# ---------------------------------------------------------------------------

#: Kinds whose older segments are rescaled by the overlap ratio: levels in
#: some unit or base, where a vintage change moves the level but should
#: leave growth rates alone.
_RATIO_KINDS = frozenset({"flow", "stock", "count", "deflator", "index"})

#: Kinds joined in levels: rates and ratios already share a unit (percent,
#: percent of GDP), so a gap between sources is a definitional difference
#: to be bounded, not a base to be divided out.
_LEVEL_KINDS = frozenset({"rate", "ratio"})

#: Kinds never extended: they are computed downstream from spliced inputs,
#: and splicing them as well would let the two disagree.
_NO_SPLICE_KINDS = frozenset({"derived"})

_SOURCES_COLUMNS = ("code", "date", "variable", "value", "source")
_PANEL_COLUMNS = ("code", "date", "variable", "value", "source")
_PROVENANCE_COLUMNS = (
    "freq", "code", "variable", "source", "first", "last", "n",
    "alternatives", "extended_from", "n_extended", "ref_year",
)
_SEAM_COLUMNS = (
    "freq", "code", "variable", "date", "older", "newer", "overlap_n",
    "ratio", "ratio_drift", "ratio_min", "ratio_max", "stable", "pow10",
    "method", "note",
)


def _factor(col: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes in lexicographic order of the labels, plus the labels.

    Lexicographic codes let every later sort and tie-break run on
    integers while still meaning "alphabetical" — which is what makes the
    output independent of row order and of categorical category order.
    """
    if col.isna().any():
        raise ValueError(f"column {name!r} has missing labels; every row "
                         "needs a code, a variable and a source")
    if isinstance(col.dtype, pd.CategoricalDtype):
        # Unused categories (filtered-out blocks, all-NaN sources) must not
        # reach the rank/kinds checks: drop them before reading the labels.
        col = col.cat.remove_unused_categories()
        cats = np.asarray(col.cat.categories.astype(str), dtype=object)
        order = np.argsort(cats, kind="stable")
        remap = np.empty(len(cats), dtype=np.int64)
        remap[order] = np.arange(len(cats))
        return remap[col.cat.codes.to_numpy()], cats[order]
    codes, uniques = pd.factorize(col.astype(str), sort=True)
    return codes.astype(np.int64), np.asarray(uniques, dtype=object)


def _infer_freq(dates_ns: np.ndarray) -> str:
    """``"Y"``, ``"Q"``, ``"M"`` or ``""`` from the commonest date spacing."""
    u = np.unique(dates_ns)
    if u.size < 2:
        return ""
    days = np.diff(u) // (86_400 * 10**9)
    vals, counts = np.unique(days[days > 0], return_counts=True)
    if vals.size == 0:
        return ""
    step = int(vals[np.argmax(counts)])
    if 28 <= step <= 31:
        return "M"
    if 89 <= step <= 92:
        return "Q"
    if 365 <= step <= 366:
        return "Y"
    return ""


def _fmt_periods(dates_ns: np.ndarray, freq: str) -> np.ndarray:
    """Short period labels for ``alternatives``: 1960, 1995Q1, 1995-01."""
    idx = pd.DatetimeIndex(dates_ns.astype("datetime64[ns]"))
    year = idx.year.astype(str)
    if freq == "Y":
        out = year
    elif freq == "Q":
        out = year + "Q" + idx.quarter.astype(str)
    elif freq == "M":
        out = year + "-" + pd.Index(idx.month).map("{:02d}".format)
    else:
        out = idx.strftime("%Y-%m-%d")
    return np.asarray(out, dtype=object)


def splice_sources(
    long: pd.DataFrame,
    rank: Mapping[str, int],
    *,
    prefer: Mapping[str, Sequence[str]] | None = None,
    kinds: Mapping[str, str] | None = None,
    min_obs: int = 8,
    min_overlap: int = 3,
    drift_warn: float = RATIO_DRIFT_WARN,
    rate_tol: float = 0.25,
    pow10_tol: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One value per ``(code, date, variable)`` from many sources.

    For every ``(code, variable)`` the sources are put in order, the best
    one with enough history becomes the **primary** and is kept exactly
    as published — gaps included — and lower-ranked sources may only
    extend it **backwards**, before its first date, through a measured
    seam. Nothing is ever overwritten and no interior gap is ever filled:
    a source that is better in 1990 but missing in 1991 leaves 1991
    missing rather than borrowing a number from a different statistical
    concept.

    Parameters
    ----------
    long : DataFrame
        Columns ``code, date, variable, value, source`` (others ignored).
        Labels may be categorical. Rows with a missing or non-finite
        ``value`` are dropped first. A source must not report the same
        ``(code, date, variable)`` twice.
    rank : mapping
        ``{source: int}``, lower is better. Every source in ``long`` must
        appear.
    prefer : mapping, optional
        ``{variable: [source, ...]}``. For those variables the listed
        sources come first, in the listed order, ahead of every unlisted
        source (which then follow by ``rank``).
    kinds : mapping, optional
        ``{variable: kind}``. ``flow``, ``stock``, ``count``, ``deflator``
        and ``index`` are joined by ratio; ``rate`` and ``ratio`` in
        levels; ``derived`` is never extended. ``None`` treats every
        variable as a ratio kind. When given, it must cover every
        variable in ``long``.
    min_obs : int
        Fewest observations a source needs to be the primary. A group in
        which no source reaches it is kept whole from its best source and
        not extended.
    min_overlap : int
        Common dates used to measure a seam: the **first** ``min_overlap``
        dates both sides report, i.e. the stretch nearest the older
        source's contribution. Fewer gives ``method="rejected"``,
        ``note="overlap"``.
    drift_warn : float
        Coefficient of variation of the overlap ratio above which a ratio
        seam is flagged ``stable=False``. It is still applied, as in
        :func:`ratio_splice`; one summary warning counts such seams.
    rate_tol : float
        Largest absolute mean gap (in the variable's units, percentage
        points for rates) at which a level seam is accepted.
    pow10_tol : float
        A ratio within this distance of ``10**k`` in log10 terms (``k``
        a nonzero integer) is recorded as ``pow10 = k``: a units change
        (thousands vs millions), absorbed by the ratio like any other.

    Returns
    -------
    panel : DataFrame
        Long ``code, date, variable, value, source``, sorted by
        ``(code, date, variable)`` and unique on it. Every row keeps the
        source it came from; extended rows carry rescaled values.
    provenance : DataFrame
        One row per ``(code, variable)`` with columns ``freq, code,
        variable, source, first, last, n, alternatives, extended_from,
        n_extended, ref_year``. ``source`` is the primary; ``first``,
        ``last``, ``n`` describe the spliced series; ``alternatives``
        lists every other source as ``"src:first-last"`` in precedence
        order; ``extended_from`` the sources whose seams were accepted,
        newest first. ``freq`` is inferred from the date spacing and
        ``ref_year`` is left empty for the caller to fill from the
        source metadata.
    seams : DataFrame
        One row per attempted extension: the :class:`Seam` fields plus
        ``freq, code, variable, pow10, method``. ``method`` is ``ratio``,
        ``level`` (where ``ratio``/``ratio_min``/``ratio_max`` hold the
        mean, minimum and maximum gap and ``ratio_drift`` its standard
        deviation) or ``rejected``. ``date`` is the first date of the
        newer side at the seam.

    Notes
    -----
    Order within a ``(code, variable)``: ``prefer`` position, then
    ``rank``, then more observations, then the later last date, then the
    source name alphabetically — so the result never depends on row
    order. Extension walks the remaining sources in that order, once
    each: a source with nothing before the current first date is skipped
    silently, a rejected one is recorded and the walk moves on, an
    accepted one moves the first date back and the next source is
    measured against the spliced series, so chains are allowed.

    Primaries are selected with array operations over all groups; the
    Python loop runs only over ``(code, variable)`` pairs that have an
    extension candidate.

    Examples
    --------
    >>> panel, prov, seams = splice_sources(
    ...     long, {"y_oecd_ana": 0, "y_wdi": 1},
    ...     kinds={"gdp": "flow", "urate": "rate"})  # doctest: +SKIP
    """
    missing = [c for c in _SOURCES_COLUMNS if c not in long.columns]
    if missing:
        raise ValueError(f"splice_sources needs columns {list(_SOURCES_COLUMNS)}; "
                         f"missing {missing}")
    if min_obs < 1 or min_overlap < 1:
        raise ValueError("min_obs and min_overlap must be at least 1")

    value_all = pd.to_numeric(long["value"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(value_all)
    df = long.loc[keep, ["code", "date", "variable", "source"]]
    val = value_all[keep]
    dates = pd.to_datetime(df["date"])
    if dates.isna().any():
        raise ValueError("splice_sources: some dates could not be parsed")
    d = dates.to_numpy(dtype="datetime64[ns]").view(np.int64)
    ci, codes_u = _factor(df["code"], "code")
    vi, vars_u = _factor(df["variable"], "variable")
    si, srcs_u = _factor(df["source"], "source")
    freq = _infer_freq(d)

    unranked = [s for s in srcs_u if s not in rank]
    if unranked:
        raise ValueError(f"sources missing from rank: {unranked}; rank has "
                         f"{sorted(rank)}")
    rank_arr = np.array([float(rank[s]) for s in srcs_u], dtype=float)

    # 0 = ratio, 1 = level, 2 = never extended
    if kinds is None:
        method_of_var = np.zeros(len(vars_u), dtype=np.int8)
    else:
        absent = [v for v in vars_u if v not in kinds]
        if absent:
            raise ValueError(f"kinds has no entry for variables {absent}")
        valid = _RATIO_KINDS | _LEVEL_KINDS | _NO_SPLICE_KINDS
        bad = sorted({kinds[v] for v in vars_u} - valid)
        if bad:
            raise ValueError(f"unknown kinds {bad}; valid are {sorted(valid)}")
        method_of_var = np.array(
            [0 if kinds[v] in _RATIO_KINDS else 1 if kinds[v] in _LEVEL_KINDS
             else 2 for v in vars_u], dtype=np.int8)

    empty_panel = pd.DataFrame({c: pd.Series(dtype=object) for c in _PANEL_COLUMNS})
    if val.size == 0:
        empty_panel["date"] = pd.Series(dtype="datetime64[ns]")
        empty_panel["value"] = pd.Series(dtype=float)
        return (empty_panel,
                pd.DataFrame(columns=list(_PROVENANCE_COLUMNS)),
                pd.DataFrame(columns=list(_SEAM_COLUMNS)))

    # Rows sorted by (code, variable, source, date): each source's series
    # within a pair is then one contiguous, date-sorted slice.
    order = np.lexsort((d, si, vi, ci))
    ci, vi, si, d, val = ci[order], vi[order], si[order], d[order], val[order]
    new_group = np.empty(ci.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = (ci[1:] != ci[:-1]) | (vi[1:] != vi[:-1]) | (si[1:] != si[:-1])
    dup = ~new_group[1:] & (d[1:] == d[:-1])
    if dup.any():
        k = int(np.flatnonzero(dup)[0]) + 1
        raise ValueError(
            "a source reports the same (code, date, variable) twice, e.g. "
            f"{srcs_u[si[k]]!r} for {codes_u[ci[k]]!r}, {vars_u[vi[k]]!r} at "
            f"{pd.Timestamp(d[k]).date()}; deduplicate before splicing")

    gstart = np.flatnonzero(new_group)
    gn = np.diff(np.append(gstart, ci.size))
    gc, gv, gs = ci[gstart], vi[gstart], si[gstart]
    gfirst = d[gstart]
    glast = d[gstart + gn - 1]
    G = gstart.size

    # Precedence key per (code, variable, source) group.
    pref_pos = np.full(G, np.iinfo(np.int64).max, dtype=np.int64)
    if prefer:
        var_id = {v: i for i, v in enumerate(vars_u)}
        src_id = {s: i for i, s in enumerate(srcs_u)}
        nS = len(srcs_u)
        pk, pp = [], []
        for var, srcs in prefer.items():
            if var not in var_id:
                continue
            for pos, s in enumerate(srcs):
                if s in src_id:
                    pk.append(var_id[var] * nS + src_id[s])
                    pp.append(pos)
        if pk:
            pk_a = np.asarray(pk, dtype=np.int64)
            pp_a = np.asarray(pp, dtype=np.int64)
            srt = np.argsort(pk_a, kind="stable")
            pk_a, pp_a = pk_a[srt], pp_a[srt]
            gkey = gv.astype(np.int64) * nS + gs
            loc = np.clip(np.searchsorted(pk_a, gkey), 0, pk_a.size - 1)
            hit = pk_a[loc] == gkey
            pref_pos[hit] = pp_a[loc[hit]]
    go = np.lexsort((gs, -glast, -gn, rank_arr[gs], pref_pos, gv, gc))

    # Pairs (code, variable) over the precedence-ordered groups.
    oc, ov = gc[go], gv[go]
    new_pair = np.empty(G, dtype=bool)
    new_pair[0] = True
    new_pair[1:] = (oc[1:] != oc[:-1]) | (ov[1:] != ov[:-1])
    pstart = np.flatnonzero(new_pair)
    P = pstart.size
    pair_of = np.cumsum(new_pair) - 1
    pos_in_pair = np.arange(G) - pstart[pair_of]

    big = np.iinfo(np.int64).max
    qual_pos = np.where(gn[go] >= min_obs, pos_in_pair, big)
    first_q = np.minimum.reduceat(qual_pos, pstart)
    has_primary = first_q != big
    prim = go[pstart + np.where(has_primary, first_q, 0)]   # group index

    # A pair needs the Python loop only if a later source starts earlier.
    prim_pos = np.where(has_primary, first_q, 0)
    later_earlier = ((pos_in_pair > prim_pos[pair_of])
                     & (gfirst[go] < gfirst[prim][pair_of]))
    cand = (np.logical_or.reduceat(later_earlier, pstart)
            & has_primary & (method_of_var[ov[pstart]] != 2))

    is_primary = np.zeros(G, dtype=bool)
    is_primary[prim] = True
    row_group = np.repeat(np.arange(G), gn)
    prim_rows = np.flatnonzero(is_primary[row_group])

    p_first = gfirst[prim].copy()
    p_n = gn[prim].copy()
    p_ext_from = np.full(P, "", dtype=object)
    p_n_ext = np.zeros(P, dtype=np.int64)

    ext_rows: list[np.ndarray] = []
    ext_scale: list[np.ndarray] = []
    seam_rows: list[dict] = []
    n_unstable = 0

    for p in np.flatnonzero(cand):
        g0 = prim[p]
        cur_d = d[gstart[g0]:gstart[g0] + gn[g0]]
        cur_v = val[gstart[g0]:gstart[g0] + gn[g0]]
        front = srcs_u[gs[g0]]
        kind_method = method_of_var[gv[g0]]
        accepted: list[str] = []
        lo_g = pstart[p]
        hi_g = pstart[p + 1] if p + 1 < P else G
        for j in range(lo_g + prim_pos[p] + 1, hi_g):
            h = go[j]
            cur_first = cur_d[0]
            if gfirst[h] >= cur_first:
                continue
            sl = slice(gstart[h], gstart[h] + gn[h])
            od, oval = d[sl], val[sl]
            _, ia, ib = np.intersect1d(od, cur_d, assume_unique=True,
                                       return_indices=True)
            a, b = oval[ia], cur_v[ib]
            if kind_method == 0:
                nz = a != 0
                a, b = a[nz], b[nz]
            a, b = a[:min_overlap], b[:min_overlap]
            older = srcs_u[gs[h]]
            base = dict(freq=freq, code=codes_u[gc[g0]], variable=vars_u[gv[g0]])
            seam_date = pd.Timestamp(cur_first)
            if a.size < min_overlap:
                seam = Seam(seam_date, older, front, int(a.size), np.nan,
                            np.nan, np.nan, np.nan, False, "overlap")
                seam_rows.append({**base, **asdict(seam), "pow10": 0,
                                  "method": "rejected"})
                continue
            if kind_method == 0:
                ratio, n, drift, lo, hi = _ratio_moments(b / a)
                if not (np.isfinite(ratio) and ratio > 0 and lo > 0):
                    seam = Seam(seam_date, older, front, n, ratio, drift, lo,
                                hi, False, "ratio not positive")
                    seam_rows.append({**base, **asdict(seam), "pow10": 0,
                                      "method": "rejected"})
                    continue
                stable = bool(np.isfinite(drift) and drift <= drift_warn)
                lg = float(np.log10(ratio))
                k = int(round(lg))
                pow10 = k if (k != 0 and abs(lg - k) <= pow10_tol) else 0
                notes = []
                if pow10:
                    notes.append(f"units differ by 10^{pow10}; residual "
                                 f"x{ratio / 10.0 ** pow10:.4f}")
                if not stable:
                    notes.append("ratio drifts across the overlap")
                    n_unstable += 1
                seam = Seam(seam_date, older, front, n, ratio, drift, lo, hi,
                            stable, "; ".join(notes))
                method, scale = "ratio", ratio
            else:
                gaps = b - a
                n = int(gaps.size)
                gap = float(np.mean(gaps))
                sd = float(np.std(gaps, ddof=1)) if n > 1 else 0.0
                ok = abs(gap) <= rate_tol
                seam = Seam(seam_date, older, front, n, gap, sd,
                            float(np.min(gaps)), float(np.max(gaps)), ok,
                            "" if ok else f"level gap {gap:+.3f} exceeds rate_tol")
                pow10 = 0
                if not ok:
                    seam_rows.append({**base, **asdict(seam), "pow10": 0,
                                      "method": "rejected"})
                    continue
                method, scale = "level", 1.0
            seam_rows.append({**base, **asdict(seam), "pow10": pow10,
                              "method": method})
            m = int(np.searchsorted(od, cur_first))          # rows before
            rows = np.arange(gstart[h], gstart[h] + m)
            ext_rows.append(rows)
            ext_scale.append(np.full(m, scale))
            cur_d = np.concatenate([od[:m], cur_d])
            cur_v = np.concatenate([oval[:m] * scale, cur_v])
            front = older
            accepted.append(older)
            p_n_ext[p] += m
        if accepted:
            p_first[p] = cur_d[0]
            p_n[p] += p_n_ext[p]
            p_ext_from[p] = ";".join(accepted)

    if n_unstable:
        warnings.warn(
            f"splice_sources: {n_unstable} ratio seam(s) drift more than "
            f"{drift_warn:.1%} across the overlap; see seams.stable",
            UserWarning, stacklevel=2)

    # Panel: primaries (vectorised) plus extension rows.
    if ext_rows:
        er = np.concatenate(ext_rows)
        es = np.concatenate(ext_scale)
    else:
        er = np.empty(0, dtype=np.int64)
        es = np.empty(0, dtype=float)
    rows = np.concatenate([prim_rows, er])
    out_val = np.concatenate([val[prim_rows], val[er] * es])
    oci, ovi, osi, od_ = ci[rows], vi[rows], si[rows], d[rows]
    srt = np.lexsort((ovi, od_, oci))
    panel = pd.DataFrame({
        "code": codes_u[oci[srt]],
        "date": od_[srt].astype("datetime64[ns]"),
        "variable": vars_u[ovi[srt]],
        "value": out_val[srt],
        "source": srcs_u[osi[srt]],
    })

    # Provenance: one row per pair, alternatives in precedence order.
    labels = (pd.Series(srcs_u[gs[go]]) + ":" + _fmt_periods(gfirst[go], freq)
              + "-" + _fmt_periods(glast[go], freq))
    not_prim = ~is_primary[go]
    alts = (labels[not_prim].groupby(pair_of[not_prim], sort=True)
            .agg(";".join).reindex(np.arange(P), fill_value=""))
    provenance = pd.DataFrame({
        "freq": freq,
        "code": codes_u[gc[prim]],
        "variable": vars_u[gv[prim]],
        "source": srcs_u[gs[prim]],
        "first": p_first.astype("datetime64[ns]"),
        "last": glast[prim].astype("datetime64[ns]"),
        "n": p_n,
        "alternatives": alts.to_numpy(dtype=object),
        "extended_from": p_ext_from,
        "n_extended": p_n_ext,
        "ref_year": "",
    }, columns=list(_PROVENANCE_COLUMNS))

    seams = pd.DataFrame(seam_rows, columns=list(_SEAM_COLUMNS))
    seams["overlap_n"] = seams["overlap_n"].astype(np.int64)
    seams["pow10"] = seams["pow10"].astype(np.int64)
    seams["stable"] = seams["stable"].astype(bool)
    return panel, provenance, seams


def to_long(wide: pd.DataFrame, *, keep_attrs: bool = True) -> pd.DataFrame:
    """``(code, date)`` wide frame to long ``code, date, variable, value``.

    The inverse of the wide layout the national-accounts fetchers return,
    and the shape :func:`splice_sources` reads once a ``source`` column is
    added. Only numeric columns are kept (provenance columns such as
    ``src_*`` in :func:`qna_long_panel` are labels, not values), missing
    values are dropped, and rows come sorted by ``(code, date,
    variable)``. An integer second level, as in ``(code, year)`` frames,
    becomes the 1 January of that year, and a ``PeriodIndex`` level
    becomes the start of each period, so every frequency shares one
    date column.

    Parameters
    ----------
    wide : DataFrame
        Two-level index ``(code, date)`` (names are not checked).
    keep_attrs : bool
        Copy ``wide.attrs`` onto the result, so ``qna_meta`` and friends
        still work on it; ``False`` returns empty ``attrs``.

    Examples
    --------
    >>> to_long(qna_panel(["ESP"]))  # doctest: +SKIP
    """
    if not isinstance(wide.index, pd.MultiIndex) or wide.index.nlevels != 2:
        raise ValueError("to_long expects a two-level (code, date) index; got "
                         f"{wide.index.nlevels} level(s)")
    num = wide.select_dtypes(include="number")
    vals = num.to_numpy(dtype=float)
    r, k = np.nonzero(np.isfinite(vals))
    codes = np.asarray(wide.index.get_level_values(0).astype(str), dtype=object)
    lvl = wide.index.get_level_values(1)
    if pd.api.types.is_integer_dtype(lvl):
        dates = pd.to_datetime(pd.Series(lvl).astype(str) + "-01-01").to_numpy()
    elif isinstance(lvl.dtype, pd.PeriodDtype):
        dates = lvl.to_timestamp(how="start").to_numpy()
    else:
        dates = pd.to_datetime(lvl).to_numpy()
    cols = np.asarray([str(c) for c in num.columns], dtype=object)
    out = pd.DataFrame({
        "code": codes[r],
        "date": dates[r],
        "variable": cols[k],
        "value": vals[r, k],
    })
    out = out.sort_values(["code", "date", "variable"], kind="stable",
                          ignore_index=True)
    out.attrs = dict(wide.attrs) if keep_attrs else {}
    return out


__all__ = [
    "RATIO_DRIFT_WARN",
    "MIN_OVERLAP",
    "Seam",
    "SpliceResult",
    "overlap_ratio",
    "ratio_splice",
    "splice_frame",
    "expenditure_residual",
    "splice_sources",
    "to_long",
]
