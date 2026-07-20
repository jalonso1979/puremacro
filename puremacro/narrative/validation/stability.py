"""α.2 stability report.

Report sensitivity of an index's headline correlation across a
(kernel × lexicon_variant × base_period × leave_one_CB_out)
perturbation grid. Used by NB44 to render a per-index stability
table and by NB28/31/32/34 builders to stamp ``meta["stability"]``
in each index's ``*_meta.json``.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


_AXES = ("kernel", "lexicon_variant", "base_period", "leave_out")


@dataclass
class NarrativeStabilityReport:
    """Run a perturbation grid against a single index.

    Parameters
    ----------
    index_fn
        Callable returning a :class:`RiskIndex`. Called with positional
        ``records`` plus ``country, language, kernel, lexicon, base_period``
        keyword arguments (the report constructs the records iterator
        fresh from ``text_iter_factory`` for each cell).
    benchmark_series
        Quarterly-aligned series to correlate the index against. One ρ per
        cell.
    benchmark_name
        Human-readable label stamped in the meta JSON.
    perturbations
        Mapping ``axis -> [values]``. The four required axes are
        ``"kernel"``, ``"lexicon_variant"``, ``"base_period"``,
        ``"leave_out"``. Each axis needs at least one value.
    text_iter_factory
        Zero-argument callable returning a fresh iterator over source
        records on each call (iterators are single-use; we replay).
    country, language
        Forwarded to ``index_fn``.
    max_cells
        Cap on grid size; cells beyond this are random-subsampled with
        ``seed``.
    seed
        Random seed for the subsample draw.
    """
    index_fn: Callable
    benchmark_series: pd.Series
    benchmark_name: str
    perturbations: dict[str, list]
    text_iter_factory: Callable
    country: str
    language: str = "en"
    max_cells: int = 64
    seed: int = 0

    _run_results: pd.DataFrame | None = field(default=None, init=False)

    def _grid(self) -> list[dict[str, Any]]:
        """Return list of perturbation cells."""
        missing = set(_AXES) - set(self.perturbations.keys())
        if missing:
            raise ValueError(f"perturbations missing axes: {missing}")
        values = [self.perturbations[a] for a in _AXES]
        cells = [dict(zip(_AXES, combo)) for combo in itertools.product(*values)]
        if len(cells) > self.max_cells:
            rng = random.Random(self.seed)
            cells = rng.sample(cells, self.max_cells)
        return cells

    def run(self) -> pd.DataFrame:
        """Execute every cell; return DataFrame with one row per cell.

        Columns: ``cell_id, kernel, lexicon_variant, base_period, leave_out, rho, n``.
        Failed cells get ``rho=NaN, n=0`` rather than raising.
        Stores result in ``self._run_results``.
        """
        rows = []
        for i, cell in enumerate(self._grid()):
            try:
                records_iter = self.text_iter_factory()
                if cell["leave_out"] is not None:
                    records_iter = (
                        r for r in records_iter
                        if (len(r) >= 4 and isinstance(r[3], dict)
                            and (r[3].get("bank_code") or r[3].get("source") or "")
                                != cell["leave_out"])
                    )
                idx = self.index_fn(
                    records_iter,
                    country=self.country,
                    language=self.language,
                    kernel=cell["kernel"],
                    lexicon=cell["lexicon_variant"],
                    base_period=cell["base_period"],
                )
                # idx is a RiskIndex; align its series with the benchmark.
                idx_series = idx.series.dropna()
                aligned_idx, aligned_bench = idx_series.align(
                    self.benchmark_series.dropna(), join="inner",
                )
                if len(aligned_idx) >= 8:
                    rho = float(aligned_idx.corr(aligned_bench))
                else:
                    rho = float("nan")
                n = int(len(aligned_idx))
            except Exception:
                rho = float("nan")
                n = 0
            rows.append({"cell_id": i, **cell, "rho": rho, "n": n})
        self._run_results = pd.DataFrame(rows)
        return self._run_results

    def summary(self) -> dict[str, Any]:
        """One-dict summary suitable for ``meta["stability"]`` stamping."""
        if self._run_results is None:
            raise RuntimeError("call run() before summary()")
        rho = self._run_results["rho"].dropna()
        if rho.empty:
            cv, min_rho, max_rho = float("nan"), float("nan"), float("nan")
        else:
            mean_rho = rho.mean()
            if mean_rho == 0:
                cv = float("inf")
            else:
                cv = float(rho.std() / abs(mean_rho))
            min_rho = float(rho.min())
            max_rho = float(rho.max())
        return {
            "cv": cv,
            "min_rho": min_rho,
            "max_rho": max_rho,
            "n_specs": int(len(self._run_results)),
            "benchmark": self.benchmark_name,
            "computed_at": dt.datetime.utcnow().isoformat() + "Z",
        }


__all__ = ["NarrativeStabilityReport"]
