"""Combine multiple :class:`Instrument`s into a single composite series.

Use cases
---------
- Sum monetary, fiscal, and uncertainty proxies into a "macro shock index".
- Average two financial-conditions indices (NFCI + STLFSI4) into one.
- Concatenate Bloom 2009 events with a continuous uncertainty index for a
  longer-history series.

All inputs must share ``.frequency``; resampling is the caller's responsibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import Instrument


_VALID_OPS = ("sum", "mean", "weighted_mean", "concat")
_VALID_ALIGN = ("inner", "outer")


def _validate_inputs(
    instruments: list[Instrument],
    op: str,
    weights: list[float] | None,
    align: str,
) -> None:
    if not instruments:
        raise ValueError("compose() received an empty instruments list")
    if op not in _VALID_OPS:
        raise ValueError(f"op {op!r} not in {_VALID_OPS}")
    if align not in _VALID_ALIGN:
        raise ValueError(f"align {align!r} not in {_VALID_ALIGN}")
    freqs = {inst.frequency for inst in instruments}
    if len(freqs) > 1:
        raise ValueError(
            f"compose() requires all instruments to share a frequency; "
            f"got {sorted(freqs)}. Resample first."
        )
    if op == "weighted_mean":
        if weights is None:
            raise ValueError("op='weighted_mean' requires weights= kwarg")
        if len(weights) != len(instruments):
            raise ValueError(
                f"weights length {len(weights)} does not match number of "
                f"instruments {len(instruments)}"
            )


def _align_series(
    instruments: list[Instrument], align: str,
) -> pd.DataFrame:
    """Build a DataFrame with one column per instrument, indexed by the
    chosen alignment of the input series indexes.

    For `inner`, only dates present in all inputs are kept. For `outer`,
    the union is kept and missing values are NaN.
    """
    join = "inner" if align == "inner" else "outer"
    cols = {f"_inst_{i}": inst.series for i, inst in enumerate(instruments)}
    df = pd.concat(cols, axis=1, join=join)
    return df


def _apply_op(
    df: pd.DataFrame,
    op: str,
    weights: list[float] | None,
    skipna: bool,
) -> pd.Series:
    if op == "sum":
        return df.sum(axis=1, skipna=skipna)
    if op == "mean":
        return df.mean(axis=1, skipna=skipna)
    if op == "weighted_mean":
        # Pointwise weighted mean. With skipna=True, dynamically renormalize
        # weights per row to exclude NaN columns.
        w = np.asarray(weights, dtype=float)
        if skipna:
            mask = df.notna().to_numpy()  # (T, n)
            vals = df.fillna(0.0).to_numpy()  # (T, n)
            num = (vals * w).sum(axis=1)
            den = (mask * w).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(den > 0, num / den, np.nan)
            return pd.Series(out, index=df.index)
        return pd.Series((df.to_numpy() * w).sum(axis=1) / w.sum(),
                         index=df.index)
    if op == "concat":
        # Chronological concat: for each timestamp, the LAST instrument
        # with a non-NaN value wins. NaN values in later instruments do
        # NOT overwrite real values from earlier ones — useful for
        # splicing a historical series with a more recent one.
        union_idx = df.index.sort_values().unique()
        out = pd.Series(np.nan, index=union_idx, dtype=float)
        for col in df.columns:
            non_nan = df[col].dropna()
            out.loc[non_nan.index] = non_nan.values
        return out
    raise ValueError(f"unreachable: op={op!r}")  # validated upstream


def compose(
    instruments: list[Instrument],
    *,
    op: str = "sum",
    weights: list[float] | None = None,
    name: str | None = None,
    source: str | None = None,
    align: str = "inner",
    skipna: bool = False,
) -> Instrument:
    """Combine multiple :class:`Instrument` series into one.

    Parameters
    ----------
    instruments : list of Instrument
        Input series. All must share ``.frequency``.
    op : {"sum", "mean", "weighted_mean", "concat"}, default "sum"
        Combination operation. For ``"concat"``: at each timestamp the
        last instrument with a *non-NaN* value is used; NaN values in
        later instruments do not overwrite real values from earlier ones.
    weights : list of float, optional
        Required when ``op="weighted_mean"``; must have the same length
        as ``instruments``. Need not sum to 1 (will be normalized).
    name : str, optional
        Result Instrument name. Auto-generated if None.
    source : str, optional
        Result Instrument source. Auto-generated if None.
    align : {"inner", "outer"}, default "inner"
        Date-index alignment. ``"inner"`` keeps only dates present in
        every input; ``"outer"`` keeps the union and fills NaN where
        missing. **Ignored when ``op="concat"``** (concat always uses
        the union of input dates so non-overlapping series can be spliced).
    skipna : bool, default False
        For ``op`` in {sum, mean, weighted_mean}: whether to ignore
        NaN values per timestamp when combining. Has no effect on
        ``op="concat"`` (which always skips NaNs by construction).

    Returns
    -------
    Instrument
        A new Instrument with category ``"composite"`` and metadata
        recording the source instrument names, operation, weights,
        and alignment mode.
    """
    _validate_inputs(instruments, op, weights, align)

    # Single-instrument case: return a copy with composite category.
    if len(instruments) == 1:
        inst = instruments[0]
        result_name = name or f"compose_{op}_{inst.name}"
        series_copy = inst.series.copy()
        series_copy.name = result_name
        return Instrument(
            series=series_copy,
            name=result_name,
            source=source or f"compose({op}: {inst.name})",
            category="composite",
            frequency=inst.frequency,
            metadata={
                "source_instruments": [inst.name],
                "composition_op": op,
                "composition_weights": list(weights) if weights is not None else None,
                "composition_align": align,
            },
        )

    # concat always needs the outer union so non-overlapping indexes merge correctly.
    effective_align = "outer" if op == "concat" else align
    df = _align_series(instruments, effective_align)
    series = _apply_op(df, op, weights, skipna)

    src_names = [inst.name for inst in instruments]
    auto_name = f"compose_{op}_{len(instruments)}_inst"
    auto_source = f"compose({op}: {', '.join(src_names)})"

    return Instrument(
        series=pd.Series(
            series.values,
            index=series.index,
            name=name or auto_name,
        ),
        name=name or auto_name,
        source=source or auto_source,
        category="composite",
        frequency=instruments[0].frequency,
        metadata={
            "source_instruments": src_names,
            "composition_op": op,
            "composition_weights": list(weights) if weights is not None else None,
            "composition_align": align,
        },
    )


__all__ = ["compose"]
