"""The two bundled Smets-Wouters (2007) datasets.

``"fred"`` is puremacro's rebuild of the seven observables from current FRED
vintages following the paper's data appendix (``_sw07_data.csv``, 1966Q1-2004Q4,
built by ``tools/build_sw07_data.py data``). ``"authors"`` is the authors' own
series from the AER replication files (``_sw07_usmodel_data.csv``,
1947Q3-2004Q4, converted verbatim by ``tools/build_sw07_data.py authors`` from
``usmodel_data.mat`` as redistributed in Johannes Pfeifer's DSGE_mod). Over
1966Q1-2004Q4 the two agree closely (correlations 0.99 or above except wage
growth at 0.87, the federal funds rate identical); the hours series differ by a
demeaning constant that the estimated ``constelab`` absorbs.
"""
from __future__ import annotations

from importlib import resources

import pandas as pd

from puremacro.dsge.sw07_observation import OBSERVED_VARS

SW07_DATASETS: dict[str, str] = {
    "fred": "_sw07_data.csv",
    "authors": "_sw07_usmodel_data.csv",
}
"""Bundled SW07 observable files by dataset name."""

__all__ = ["SW07_DATASETS", "load_sw07_data"]


def load_sw07_data(
    dataset: str = "fred",
    *,
    first_obs: str | None = None,
    last_obs: str | None = None,
) -> pd.DataFrame:
    """Return one bundled SW07 dataset, optionally sliced to ``[first_obs, last_obs]``.

    Parameters
    ----------
    dataset : {"fred", "authors"}
        See the module docstring.
    first_obs, last_obs : str, optional
        Inclusive quarter labels such as ``"1956Q1"``; the index is a string
        column ``date`` with that format, so comparisons are lexicographic and
        exact.

    Returns
    -------
    DataFrame
        Columns :data:`puremacro.dsge.sw07_observation.OBSERVED_VARS`, index
        ``date``.
    """
    if dataset not in SW07_DATASETS:
        raise ValueError(f"unknown SW07 dataset {dataset!r}; expected one of {sorted(SW07_DATASETS)}")
    res = resources.files("puremacro.dsge").joinpath(SW07_DATASETS[dataset])
    with res.open("r", encoding="utf-8") as fh:
        df = pd.read_csv(fh, comment="#", index_col="date")
    df = df[list(OBSERVED_VARS)]
    if first_obs is not None:
        if first_obs < df.index[0] or first_obs > df.index[-1]:
            raise ValueError(f"first_obs={first_obs!r} outside {df.index[0]}..{df.index[-1]} of {dataset!r}")
        df = df.loc[df.index >= first_obs]
    if last_obs is not None:
        if last_obs < df.index[0] or last_obs > df.index[-1]:
            raise ValueError(f"last_obs={last_obs!r} outside the sample of {dataset!r}")
        df = df.loc[df.index <= last_obs]
    return df.copy()
