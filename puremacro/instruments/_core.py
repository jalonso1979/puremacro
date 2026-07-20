"""Core types for puremacro.instruments.

Defines the canonical ``Instrument`` wrapper (a frozen dataclass) and
the ``InstrumentLike`` Protocol that any class can satisfy by exposing
an ``as_instrument()`` method. Downstream consumers (proxy_svar,
lp_iv, future SVAR-IV variants) accept ``Instrument`` and dispatch
uniformly; upstream classes (``NarrativeInstrument``, ``JKResult``)
provide adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..var.identify._results import ProxySVARResult


VALID_CATEGORIES = {
    "narrative_replication",
    "narrative_connector",
    "monetary_hfi",
    "literature",
    "external_csv",
    "composite",
    "text_index",
}


@dataclass(frozen=True)
class Instrument:
    """A single identified shock or instrument series with provenance.

    Constructed via ``as_instrument()`` adapters on existing classes
    (:class:`puremacro.narrative.NarrativeInstrument`,
    :class:`puremacro.hfi.JKResult`) or via
    :func:`puremacro.instruments.load`.

    Attributes
    ----------
    series : pd.Series
        Date-indexed proxy/shock values. Any frequency.
    name : str
        Short identifier. Matches the registry key when loaded via
        :func:`load`.
    source : str
        Human-readable provenance, e.g. ``"Ramey 2011 defense buildup events"``.
    category : str
        One of ``"narrative_replication"``, ``"narrative_connector"``,
        ``"monetary_hfi"``, ``"literature"``, ``"external_csv"``,
        ``"composite"``, ``"text_index"``. See :data:`VALID_CATEGORIES`.
    frequency : str
        Pandas-style frequency code: ``"M"``, ``"Q"``, ``"A"``.
    metadata : dict
        Free-form additional fields (e.g. country, target, reference).
    """

    series: pd.Series
    name: str
    source: str
    category: str
    frequency: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Note: __post_init__ deviates from the result-object standard's
    # "no __post_init__" rule because Instrument is a publicly-
    # constructable interface type (users create them directly, not
    # only via factory methods). Constructor-level category validation
    # gives early errors instead of silent downstream confusion.
    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category {self.category!r} not in {sorted(VALID_CATEGORIES)}"
            )

    def diagnostics(self) -> dict[str, Any]:
        """Sample-size, central-tendency, and date-coverage statistics.

        Returns ``{"n_obs": 0, "mean": nan, "std": nan, "first_date": None,
        "last_date": None}`` if ``series.dropna()`` is empty.
        """
        s = self.series.dropna()
        return {
            "n_obs": int(s.shape[0]),
            "mean": float(s.mean()) if s.shape[0] else float("nan"),
            "std": float(s.std()) if s.shape[0] else float("nan"),
            "first_date": str(s.index.min()) if s.shape[0] else None,
            "last_date": str(s.index.max()) if s.shape[0] else None,
        }

    def validate_against(self, benchmark: pd.Series) -> dict[str, Any]:
        """Correlation + overlap diagnostics against a benchmark series.

        Returns ``{"correlation": nan, "n_overlap": 0}`` when the date
        ranges of ``self.series`` and ``benchmark`` do not overlap after
        an inner join.
        """
        joined = pd.concat([self.series, benchmark], axis=1, join="inner").dropna()
        if joined.empty:
            return {"correlation": float("nan"), "n_overlap": 0}
        return {
            "correlation": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
            "n_overlap": int(joined.shape[0]),
        }

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        d = self.diagnostics()
        return (
            f"Instrument: {self.name}\n"
            f"  source            : {self.source}\n"
            f"  category          : {self.category}\n"
            f"  frequency         : {self.frequency}\n"
            f"  n_obs             : {d['n_obs']}\n"
            f"  mean (std)        : {d['mean']:+.4f} ({d['std']:.4f})\n"
            f"  date range        : {d['first_date']} → {d['last_date']}\n"
        )

    def to_proxy_svar(
        self,
        Y: np.ndarray,
        *,
        p: int,
        horizon: int,
        shock_target_idx: int = 0,
        n_boot: int = 500,
        ci: float = 0.9,
        seed: int = 0,
    ) -> "ProxySVARResult":
        """Run :func:`puremacro.var.identify.proxy.proxy_svar` with this
        instrument as the external proxy.

        Returns
        -------
        :class:`puremacro.var.identify._results.ProxySVARResult`
        """
        from ..var.identify.proxy import proxy_svar
        return proxy_svar(
            Y, p=p, horizon=horizon,
            instrument_series=np.asarray(self.series.values, dtype=float),
            shock_target_idx=shock_target_idx,
            n_boot=n_boot, ci=ci, seed=seed,
        )

    def to_lp_iv(self, df: pd.DataFrame, *, y: str, x: str, **kwargs) -> pd.DataFrame:
        """Run :func:`puremacro.lp.iv.lp_iv` with this instrument as ``z``.

        The instrument series is reindexed onto ``df.index`` and added as
        a column with a unique name; this column is then passed as ``z=``
        to ``lp_iv``. Any extra ``kwargs`` are forwarded.
        """
        from ..lp.iv import lp_iv
        z_col = "_instrument_z"
        if z_col in df.columns:
            raise ValueError(
                f"DataFrame already contains column {z_col!r}, which is reserved "
                f"for the instrument series in to_lp_iv()."
            )
        df2 = df.copy()
        df2[z_col] = self.series.reindex(df2.index)
        return lp_iv(df2, y=y, x=x, z=z_col, **kwargs)

    def compose(
        self,
        *others: "Instrument",
        op: str = "sum",
        weights: list[float] | None = None,
        name: str | None = None,
        source: str | None = None,
        align: str = "inner",
        skipna: bool = False,
    ) -> "Instrument":
        """Compose this instrument with zero or more others.

        Thin method wrapper for :func:`puremacro.instruments.compose`.
        Equivalent to ``compose([self, *others], op=op, ...)``.
        """
        from ._compose import compose as _compose
        return _compose(
            [self, *others],
            op=op,
            weights=weights,
            name=name,
            source=source,
            align=align,
            skipna=skipna,
        )


@runtime_checkable
class InstrumentLike(Protocol):
    """Anything that knows how to expose itself as an :class:`Instrument`.

    A single-method protocol. Implementations may require keyword arguments
    (e.g. :class:`puremacro.hfi.JKResult` requires ``index=`` because it
    carries no datetime info). The runtime ``isinstance`` check verifies
    only that ``as_instrument`` exists; call sites must consult each
    class's own ``as_instrument()`` signature for required kwargs.

    Forward-compatible with future shock types.
    """

    def as_instrument(self) -> Instrument: ...


__all__ = ["Instrument", "InstrumentLike", "VALID_CATEGORIES"]
