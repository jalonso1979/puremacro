"""Result classes for puremacro.lp."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class LPResult(pd.DataFrame):
    """Result of local projection estimators.

    Subclasses :class:`pandas.DataFrame` so that all existing DataFrame operations
    (column access, `.loc`, `.iloc`, slicing) continue to work seamlessly, while
    providing convenient properties (`.point`, `.se_arr`, `.ci_lower`, `.ci_upper`),
    plotting (`.plot()`), and formatted summary tables (`.summary()`).
    """

    _metadata = ["y_name", "x_name", "method"]

    @property
    def _constructor(self):
        return LPResult

    @property
    def point(self) -> np.ndarray:
        """Point estimate array across horizons (β_h)."""
        return self["beta"].to_numpy()

    @property
    def se(self) -> np.ndarray:
        """Standard errors across horizons (alias for se_arr)."""
        return self["se"].to_numpy() if "se" in self.columns else np.array([])

    @property
    def se_arr(self) -> np.ndarray:
        """Standard errors across horizons."""
        return self.se

    @property
    def ci_lower(self) -> np.ndarray | None:
        """Lower confidence band."""
        return self["lo"].to_numpy() if "lo" in self.columns else None

    @property
    def ci_upper(self) -> np.ndarray | None:
        """Upper confidence band."""
        return self["hi"].to_numpy() if "hi" in self.columns else None

    @property
    def t_stat(self) -> np.ndarray:
        """t-statistics across horizons."""
        if "t" in self.columns:
            return self["t"].to_numpy()
        se = self.se
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se > 0, self.point / se, np.nan)

    @property
    def horizons(self) -> np.ndarray:
        """Array of horizon values."""
        if "h" in self.columns:
            return self["h"].to_numpy()
        return np.asarray(self.index)

    @property
    def frame(self) -> pd.DataFrame:
        """Return as a standard pandas DataFrame."""
        return pd.DataFrame(self)

    @property
    def metadata(self) -> dict[str, Any]:
        """Estimation metadata dictionary."""
        return {
            "y_name": getattr(self, "y_name", None),
            "x_name": getattr(self, "x_name", None),
            "method": getattr(self, "method", "LP"),
        }

    def to_frame(self) -> pd.DataFrame:
        """Convert to a standard pandas.DataFrame."""
        return pd.DataFrame(self)

    def plot(
        self,
        *,
        title: str = "",
        ylabel: str = "Response",
        scale: float = 1.0,
        ax=None,
    ):
        """Plot impulse response function with error bands.

        Lazily delegates to puremacro.plot.plot_irf_single.
        """
        from ..plot import plot_irf_single

        return plot_irf_single(self, title=title, ylabel=ylabel, scale=scale, ax=ax)

    def summary(self) -> str:
        """Formatted text summary table of local projection estimates."""
        method = getattr(self, "method", "LP")
        lines = [f"Local Projection Result (method: {method})"]
        lines.append(f"{'h':>4}  {'beta':>10}  {'se':>10}  {'lo':>10}  {'hi':>10}")
        lines.append("-" * 52)
        for _, row in self.iterrows():
            h_val = int(row["h"]) if "h" in row and not pd.isna(row["h"]) else 0
            b_val = row.get("beta", np.nan)
            se_val = row.get("se", np.nan)
            lo_val = row.get("lo", np.nan)
            hi_val = row.get("hi", np.nan)
            lines.append(
                f"{h_val:>4}  {b_val:>10.4f}  {se_val:>10.4f}  "
                f"{lo_val:>10.4f}  {hi_val:>10.4f}"
            )
        return "\n".join(lines)

    def to_markdown(self, index: bool = False) -> str:
        """Render LP table as GitHub-flavored Markdown."""
        from ..reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), index=index)

    def to_latex(self, index: bool = False) -> str:
        """Render LP table as LaTeX tabular environment."""
        from ..reports import _df_to_latex

        return _df_to_latex(self.to_frame(), index=index)

    def to_typst(self, index: bool = False) -> str:
        """Render LP table as Typst table function."""
        from ..reports import _df_to_typst

        return _df_to_typst(self.to_frame(), index=index)
