"""Frozen-dataclass result objects for puremacro.hfi."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JKResult:
    """Result of :func:`puremacro.hfi.jk2020.jk_poor_man` and
    :func:`puremacro.hfi.jk2020.jk_median_target`.

    Attributes
    ----------
    mp_shock : ndarray, shape (T,)
        Identified monetary-policy shock series. Zero where the info-shock
        category fired (poor-man variant) or projected onto the MP rotation
        column (median-target variant).
    info_shock : ndarray, shape (T,)
        Identified central-bank-information shock series.
    rotation : ndarray of shape (2, 2) or None
        Rotation matrix used by the median-target variant. None for poor-man.
    n_admissible : int or None
        Number of admissible rotations searched (median-target). None for poor-man.
    method : str
        Either ``"poor_man"`` or ``"median_target"``.
    """

    mp_shock: np.ndarray
    info_shock: np.ndarray
    rotation: np.ndarray | None
    n_admissible: int | None
    method: str

    def summary(self) -> str:
        T = self.mp_shock.shape[0]
        lines = [
            f"Jarociński-Karadi (2020) decomposition",
            f"  method            : {self.method}",
            f"  observations      : {T}",
            f"  MP shock var      : {float(np.var(self.mp_shock)):.4f}",
            f"  Info shock var    : {float(np.var(self.info_shock)):.4f}",
        ]
        if self.method == "median_target":
            lines.append(f"  admissible rots   : {self.n_admissible}")
        return "\n".join(lines) + "\n"

    def as_instrument(
        self,
        *,
        component: str = "mp",
        index: pd.DatetimeIndex,
    ):
        """Wrap one component of the decomposition as an
        :class:`puremacro.instruments.Instrument`.

        Parameters
        ----------
        component : ``"mp"`` (default) or ``"info"`` — which shock series.
        index : pd.DatetimeIndex — dates for the shock array (required;
            ``JKResult`` deliberately carries no datetime info).

        Notes
        -----
        The returned ``Instrument.series`` wraps the same underlying numpy
        buffer as ``self.mp_shock`` / ``self.info_shock`` (zero-copy).
        In-place mutation of the Series will propagate back to this JKResult.
        """
        from ..instruments import Instrument
        if component not in ("mp", "info"):
            raise ValueError(f"component must be 'mp' or 'info', got {component!r}")
        arr = self.mp_shock if component == "mp" else self.info_shock
        if len(index) != len(arr):
            raise ValueError(
                f"index length {len(index)} does not match shock array length {len(arr)}"
            )
        return Instrument(
            series=pd.Series(arr, index=index, name=f"jk_{component}_shock"),
            name=f"jk2020_{component}_shock",
            source=f"Jarociński-Karadi 2020 {component} component ({self.method})",
            category="monetary_hfi",
            frequency="M",
            metadata={
                "method": self.method,
                "n_admissible": self.n_admissible,
                "rotation": self.rotation,
            },
        )
