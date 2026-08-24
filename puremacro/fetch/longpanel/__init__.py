"""The longest quarterly national accounts panel we can defensibly build.

``qna_long_panel`` extends :func:`puremacro.fetch.qna_panel`'s OECD
spine backwards, per country, by ratio-splicing archived national
vintages onto it — and reports how well each join is determined instead
of hiding it.

Only two countries are extended, because only two archived sources were
measured to reach further back than the OECD already does: Spain to
1970Q1 (+100 quarters) and Japan to 1955Q2 (+155). Everything else that
was checked and rejected is recorded, with the reason, in
:data:`~puremacro.fetch.longpanel.panel.KNOWN_GAPS`.
"""
from __future__ import annotations

from ._splice import (
    MIN_OVERLAP,
    RATIO_DRIFT_WARN,
    Seam,
    SpliceResult,
    expenditure_residual,
    overlap_ratio,
    ratio_splice,
    splice_frame,
)
from .panel import (
    KNOWN_GAPS,
    LONG_PANEL_COLUMNS,
    LONG_PANEL_SOURCES,
    long_panel_residual,
    qna_long_panel,
)

__all__ = [
    "qna_long_panel", "long_panel_residual",
    "LONG_PANEL_COLUMNS", "LONG_PANEL_SOURCES", "KNOWN_GAPS",
    "ratio_splice", "splice_frame", "overlap_ratio", "expenditure_residual",
    "Seam", "SpliceResult", "RATIO_DRIFT_WARN", "MIN_OVERLAP",
]
