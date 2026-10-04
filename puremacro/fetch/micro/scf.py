"""Federal Reserve Survey of Consumer Finances (SCF), summary extract.

The SCF is the reference source for the US wealth distribution and the
natural calibration target for heterogeneous-agent models
(:mod:`puremacro.vfi`). It is not an API: the Board publishes stable zip
files, freely and without registration. This module downloads them
through the shared HTTP cache and assembles one :class:`MicroFrame`.

WHAT COMES BACK
---------------
One row per household per implicate (the SCF multiply imputes missing
answers five times), with the Board's summary-extract variables
(``networth``, ``income``, ``asset``, ``debt``, ``age``, ``edcl``, ...),
the sampling weight ``wgt``, an ``implicate`` column (1-5), and — unless
``replicate_weights=False`` — the 999 bootstrap replicate weights.

Within one implicate, ``wgt`` sums to the number of US households, so
estimates are computed per implicate and combined with Rubin's rules;
never sum ``wgt`` across the stacked file.

REPLICATE WEIGHTS
-----------------
The replicate file carries ``wt1b1..wt1b999`` and multiplicity factors
``mm1..mm999``; the usable replicate weight is their product (missing
values are zero). Variance is ``(1/998) · Σ_r (θ_r − θ)²`` per implicate,
the convention of the CRAN ``scf`` package's design objects. Replicates
are keyed by household (``yy1``) and apply identically to every implicate.

Dollar values are in the survey year's dollars.

Files (2001 onward)::

    https://www.federalreserve.gov/econres/files/scfp{YEAR}s.zip   -> rscfp{YEAR}.dta
    https://www.federalreserve.gov/econres/files/scf{YEAR}rw1s.zip -> p{YY}_rw1.dta

Earlier waves live under ``/econresdata/scf/files/`` with two-digit
names; :func:`scf_urls` knows both.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from .. import _http
from ._design import MicroFrame, SurveyDesign

FIRST_WAVE = 1989
_NEW = "https://www.federalreserve.gov/econres/files/"
_OLD = "https://www.federalreserve.gov/econresdata/scf/files/"
_IDS = ("yy1", "y1", "implicate")


def available_years(through: int = 2022) -> list[int]:
    """Survey waves (triennial since 1989) up to ``through``."""
    return list(range(FIRST_WAVE, through + 1, 3))


def scf_urls(year: int) -> dict[str, str]:
    """Summary-extract and replicate-weight zip URLs for ``year``."""
    if year < FIRST_WAVE or (year - FIRST_WAVE) % 3:
        raise ValueError(f"no SCF wave in {year}; waves are {FIRST_WAVE}, "
                         f"{FIRST_WAVE + 3}, ...")
    yy = f"{year % 100:02d}"
    if year <= 1998:
        return {"summary": f"{_OLD}scfp{year}s.zip",
                "replicates": f"{_OLD}scf{yy}rw1s.zip"}
    return {"summary": f"{_NEW}scfp{year}s.zip",
            "replicates": f"{_NEW}scf{year}rw1s.zip"}


def _read_dta_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".dta")]
        if len(members) != 1:
            raise ValueError(f"expected one .dta file in the archive, found {members}")
        with zf.open(members[0]) as fh:
            df = pd.read_stata(io.BytesIO(fh.read()), convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    # The 1989 wave uses x1/xx1 for the case identifiers.
    return df.rename(columns={"x1": "y1", "xx1": "yy1"})


def prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``implicate`` (last digit of ``y1``) and check the layout."""
    for c in ("y1", "yy1", "wgt"):
        if c not in df.columns:
            raise ValueError(f"SCF summary extract lacks {c!r}")
    out = df.copy()
    out["y1"] = out["y1"].astype(np.int64)
    out["yy1"] = out["yy1"].astype(np.int64)
    out["implicate"] = (out["y1"] % 10).astype(np.int64)
    counts = out["implicate"].value_counts()
    if sorted(counts.index) != [1, 2, 3, 4, 5] or counts.nunique() != 1:
        raise ValueError(f"expected 5 equal-sized implicates, got {counts.to_dict()}")
    return out


def prepare_replicates(rw: pd.DataFrame) -> pd.DataFrame:
    """``wt1b_k x mm_k`` for k = 1..999, one row per household (``yy1``)."""
    rw = rw.rename(columns=str.lower)
    if "yy1" not in rw.columns:
        raise ValueError("SCF replicate file lacks 'yy1'")
    reps = [f"wt1b{k}" for k in range(1, 1000)]
    missing = [c for c in reps if c not in rw.columns]
    if missing:
        raise ValueError(f"SCF replicate file lacks {missing[:3]}...")
    wt = rw[reps].fillna(0.0).to_numpy(dtype=float)
    mm_cols = [f"mm{k}" for k in range(1, 1000)]
    if all(c in rw.columns for c in mm_cols):
        wt = wt * rw[mm_cols].fillna(0.0).to_numpy(dtype=float)
    out = pd.DataFrame(wt, columns=reps)
    out.insert(0, "yy1", rw["yy1"].astype(np.int64).to_numpy())
    if out["yy1"].duplicated().any():
        raise ValueError("SCF replicate file has duplicate households")
    return out


def assemble(summary: pd.DataFrame, replicates: pd.DataFrame | None,
             variables=None) -> pd.DataFrame:
    """Join the prepared summary extract and replicate weights."""
    keep = list(_IDS) + ["wgt"]
    if variables is not None:
        absent = [v for v in variables if v.lower() not in summary.columns]
        if absent:
            raise KeyError(f"not in the SCF summary extract: {absent}")
        keep += [v.lower() for v in variables if v.lower() not in keep]
    else:
        keep += [c for c in summary.columns if c not in keep]
    df = summary[keep]
    if replicates is None:
        return df.reset_index(drop=True)
    joined = df.merge(replicates, on="yy1", how="left", validate="many_to_one")
    if joined["wt1b1"].isna().any():
        n = joined.loc[joined["wt1b1"].isna(), "yy1"].nunique()
        raise ValueError(f"{n} households have no replicate weights")
    return joined


def fetch_scf(
    year: int = 2022,
    variables=None,
    *,
    replicate_weights: bool = True,
    refresh: bool = False,
    timeout: int = 300,
) -> MicroFrame:
    """SCF summary extract for one wave, with its variance design.

    Parameters
    ----------
    year : survey wave (1989, 1992, ..., 2022).
    variables : summary-extract columns to keep (case-insensitive); all
        of them when ``None``.
    replicate_weights : also download and attach the 999 replicates.

    Examples
    --------
    >>> scf = fetch_scf(2022, ["networth", "income", "age"])   # doctest: +SKIP
    >>> scf.quantile("networth", 0.5)                           # doctest: +SKIP
    """
    urls = scf_urls(year)
    summary = prepare_summary(_read_dta_zip(
        _http.cached_get(urls["summary"], refresh=refresh, timeout=timeout)))
    reps = None
    if replicate_weights:
        reps = prepare_replicates(_read_dta_zip(
            _http.cached_get(urls["replicates"], refresh=refresh, timeout=timeout)))
    data = assemble(summary, reps, variables)
    design = (SurveyDesign.scf() if replicate_weights else
              SurveyDesign(weight="wgt", implicate="implicate",
                           note="replicate weights not requested"))
    return MicroFrame(data=data, design=design, unit="household",
                      source="fed.scf", vintage=str(year),
                      query=urls["summary"])


__all__ = ["fetch_scf", "scf_urls", "available_years", "prepare_summary",
           "prepare_replicates", "assemble"]
