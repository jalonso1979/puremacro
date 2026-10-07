"""Original-vintage Romer--Romer (2010) Figure 4 fiscal-shock replication.

This is the paper's no-controls distributed-lag regression, not a local
projection: quarterly GDP growth on a constant and exogenous tax changes at
lags zero through twelve, over 1950Q1--2007Q4. Conventional OLS covariance is
retained to reproduce the paper. The frozen statsmodels reference is generated
by a standalone exporter that never imports puremacro.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import io
import json
from typing import Any

import numpy as np
import pandas as pd

from puremacro.regress.ols import ols

__all__ = ["RR2010Result", "load_rr2010_data", "load_rr2010_reference", "estimate_rr2010_baseline"]

_ARCHIVE_SHA256 = "c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c"
_METADATA_SHA256 = "06cf9cffee4213684b0abd9d05a158e00e880cc6260cfab632cf6a2793ee13d7"
_COLUMNS = ("gdp", "nomgdp", "defic", "longr")
_PERIODS = pd.period_range("1947Q1", "2007Q4", freq="Q-DEC", name="quarter")


def _metadata() -> dict[str, Any]:
    root = resources.files("puremacro.replication.data")
    content = root.joinpath("rr2010_original_metadata.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != _METADATA_SHA256:
        raise ValueError("Romer--Romer checksum mismatch: rr2010_original_metadata.json")
    metadata = json.loads(content)
    if metadata.get("archive_sha256") != _ARCHIVE_SHA256:
        raise ValueError("Romer--Romer source archive differs from the pinned original-vintage reference")
    return metadata


def _authenticated(name: str, key: str, metadata: dict) -> bytes:
    content = resources.files("puremacro.replication.data").joinpath(name).read_bytes()
    expected = metadata.get(key)
    if not isinstance(expected, str) or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Romer--Romer checksum mismatch: {name}")
    return content


def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(data.index, pd.PeriodIndex) or not data.index.equals(_PERIODS):
        raise ValueError("data must contain each quarter 1947Q1--2007Q4 exactly once, in order, with a Q-DEC PeriodIndex")
    if data.columns.has_duplicates or not set(_COLUMNS).issubset(data.columns):
        raise ValueError("data requires unique columns gdp, nomgdp, defic and longr")
    values = data.loc[:, list(_COLUMNS)].to_numpy()
    if np.iscomplexobj(values):
        raise ValueError("Romer--Romer inputs must be real")
    try:
        values = values.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Romer--Romer inputs must be numeric") from exc
    if not np.isfinite(values).all() or np.any(values[:, :2] <= 0):
        raise ValueError("Romer--Romer inputs must be finite; real and nominal GDP must be positive")
    result = pd.DataFrame(values, index=_PERIODS.copy(), columns=_COLUMNS)
    result.attrs.update(data.attrs)
    return result


def load_rr2010_data() -> pd.DataFrame:
    """Load authenticated, original-vintage quarterly inputs from the authors.

    Index: each quarter 1947Q1--2007Q4. Columns ``gdp`` (real GDP quantity
    index), ``nomgdp`` (nominal GDP), ``defic`` and ``longr`` (nonretroactive
    tax-liability changes motivated by inherited deficits and long-run goals).
    Nominal values use the authors' common annualized dollar scale; the
    regression uses ``100*(defic+longr)/nomgdp`` in percentage points of GDP.
    DataFrame attributes retain source URLs, hashes, vintage and extraction.
    No contemporary GDP series or alternative shock vintage is substituted.
    Metadata records a small published t-statistic rounding discrepancy, also
    present with the original program's RATS databank values.
    """
    metadata = _metadata()
    content = _authenticated("rr2010_original.csv", "csv_sha256", metadata)
    frame = pd.read_csv(io.BytesIO(content), float_precision="round_trip")
    if "quarter" not in frame:
        raise ValueError("Romer--Romer CSV lacks its quarterly date column")
    frame.index = pd.PeriodIndex(frame.pop("quarter"), freq="Q-DEC", name="quarter")
    frame.attrs.update(metadata)
    return _validate_data(frame)


def load_rr2010_reference() -> dict[str, Any]:
    """Read authenticated independent statsmodels results; no estimation occurs.

    Arrays are JSON lists. Keys include full coefficient and IRF covariance
    matrices, point estimates, standard errors, t-statistics, sample counts and
    reference environment/source metadata. Published rounded values are
    documented separately from these unrounded software outputs.
    """
    metadata = _metadata()
    content = _authenticated("rr2010_statsmodels_reference.json", "reference_sha256", metadata)
    reference = json.loads(content)
    shapes = {"coefficients": (14,), "coefficient_covariance": (14, 14),
              "irf": (13,), "irf_covariance": (13, 13), "standard_errors": (13,),
              "t_statistics": (13,)}
    for name, shape in shapes.items():
        values = np.asarray(reference.get(name), dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Romer--Romer reference has missing, malformed or nonfinite {name}")
    if reference.get("nobs") != 232 or reference.get("df_resid") != 218:
        raise ValueError("Romer--Romer reference uses a different sample or degrees of freedom")
    ref_metadata = reference.get("metadata", {})
    if (ref_metadata.get("archive_sha256") != _ARCHIVE_SHA256
            or ref_metadata.get("csv_sha256") != metadata["csv_sha256"]
            or ref_metadata.get("runtime_imported_puremacro") is not False):
        raise ValueError("Romer--Romer reference lacks matching independent provenance")
    return reference


@dataclass(frozen=True)
class RR2010Result:
    """The paper's Figure 4 estimates and their full joint covariance.

    Coefficient order is constant followed by tax lags 0--12. IRF order is
    horizons 0--12, in percent real GDP per tax increase of one percent of GDP.
    Covariances describe estimated moments, with no additional sample scaling.
    Arrays are copied and read-only; metadata is an ordinary JSON-ready dict.
    """

    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    irf: np.ndarray
    irf_covariance: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    nobs: int
    df_resid: int
    metadata: dict[str, Any]

    def __post_init__(self):
        for name in ("coefficients", "coefficient_covariance", "irf", "irf_covariance",
                     "standard_errors", "t_statistics"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    def to_frame(self) -> pd.DataFrame:
        """Return the thirteen cumulative responses and conventional OLS bands."""
        frame = pd.DataFrame({"h": np.arange(13), "response": self.irf,
                              "se": self.standard_errors, "t": self.t_statistics,
                              "lower_1se": self.irf-self.standard_errors,
                              "upper_1se": self.irf+self.standard_errors})
        frame.attrs.update(self.metadata)
        return frame

    def to_moment_targets(self):
        """Preserve the full covariance and fiscal-shock convention for fitting.

        Conventional OLS covariance is inherited from the published study;
        this adapter does not establish exogeneity or robust coverage.
        """
        from puremacro.structural import MomentTargets
        return MomentTargets.from_irf(self.irf, covariance=self.irf_covariance,
                                      response="real_gdp", shock="exogenous_tax_increase",
                                      units="percent GDP per tax increase of one percent of GDP",
                                      metadata=self.metadata)


def estimate_rr2010_baseline(data: pd.DataFrame | None = None) -> RR2010Result:
    """Reproduce RR2010 equation (6), Figure 4 using original-vintage data.

    Estimate 100*Δlog(GDP) on a constant and thirteen exogenous tax-change
    regressors (current through lag twelve) for 1950Q1--2007Q4. The cumulative
    IRF sums tax coefficients through each horizon. Conventional homoskedastic
    covariance uses SSR/(232-14), matching the original RATS LINREG block.
    The complete IRF covariance is ``C Cov(beta) C'``, not cumulative squared
    marginal standard errors. No GDP-growth controls, HAC correction, LP
    transformation, retroactive tax changes or updated GDP vintage are used.

    A supplied frame must contain the complete original quarterly grid and
    the four named primitive columns. Its values may differ for an explicit
    sensitivity experiment, which is labelled as caller-supplied in metadata.
    Missing observations or unavailable presample lags raise rather than drop.
    """
    supplied = data is not None
    frame = _validate_data(load_rr2010_data() if data is None else data)
    shocks = 100*(frame["defic"]+frame["longr"])/frame["nomgdp"]
    growth = 100*np.log(frame["gdp"]).diff()
    design = pd.DataFrame({"constant": 1.}, index=frame.index)
    for lag in range(13):
        design[f"tax_lag_{lag}"] = shocks.shift(lag)
    sample = slice("1950Q1", "2007Q4")
    x = design.loc[sample].to_numpy()
    y = growth.loc[sample].to_numpy()
    if x.shape != (232, 14) or y.shape != (232,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("The exact published sample/design cannot be constructed")
    fitted = ols(y, x, cov_type="nonrobust")
    if int(fitted.nobs) != 232 or int(fitted.df_resid) != 218:
        raise ValueError("The exact published regression requires 232 observations and 14 full-rank regressors")
    coefficients = np.asarray(fitted.params)
    coefficient_covariance = np.asarray(fitted.cov_params())
    accumulation = np.column_stack((np.zeros(13), np.tril(np.ones((13, 13)))))
    irf = accumulation@coefficients
    covariance = accumulation@coefficient_covariance@accumulation.T
    variances = np.diag(covariance)
    if not np.isfinite(covariance).all() or not np.isfinite(irf).all() or np.any(variances <= 0):
        raise ValueError("Published-study inference produced invalid cumulative-response covariance")
    se = np.sqrt(variances)
    metadata = dict(frame.attrs)
    metadata.update({"study": "Romer and Romer (2010), AER 100(3), equation (6), Figure 4",
                     "paper_url": "https://eml.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf",
                     "sample_start": "1950Q1", "sample_end": "2007Q4", "frequency": "quarterly",
                     "horizons": list(range(13)), "coefficient_labels": list(design.columns),
                     "shock_definition": "nonretroactive deficit-motivated plus long-run-motivated tax changes",
                     "shock_normalization": "positive tax increase of one percent of GDP",
                     "shock_size": 1., "shock_unit": "percentage points of nominal GDP",
                     "response_unit": "percent real GDP", "transformation": "cumulative distributed-lag response of 100*log GDP",
                     "covariance_type": "conventional homoskedastic OLS, SSR/(n-k)",
                     "covariance_scale": "covariance of estimated responses; no extra sample-size multiplier",
                     "input_origin": "caller-supplied sensitivity data" if supplied else "authenticated original author archive",
                     "is_synthetic": frame.attrs.get("is_synthetic") if supplied else False,
                     "limitations": ["Replication inherits the authors' narrative-shock exogeneity assumption",
                                     "Conventional OLS covariance does not provide HAC-robust coverage",
                                     "Thirteen-quarter distributed-lag response is not a fully specified structural fiscal model",
                                     "Caller-supplied sensitivity inputs are not automatically published-result replications"]})
    # JSON serialization is also a deliberate check on the exposed metadata contract.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    return RR2010Result(coefficients, coefficient_covariance, irf, covariance, se, irf/se,
                        int(fitted.nobs), int(fitted.df_resid), metadata)
