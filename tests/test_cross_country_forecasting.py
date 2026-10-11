"""Independent forecast oracles, time separation and portable study evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from puremacro import pocket
from puremacro.examples import cross_country_forecasting as study


def _levels():
    """Three nondegenerate growth histories; no package estimator generates them."""
    years = np.arange(1960, 2025)
    frames = []
    rng = np.random.default_rng(91023)
    for j, code in enumerate(study.COUNTRIES):
        growth = (2 + j + .07 * np.arange(len(years) - 1)
                  + rng.normal(size=len(years) - 1))
        values = (100 + j) * np.exp(np.r_[0., np.cumsum(growth)] / 100)
        frames.append(pd.DataFrame({"code": code, "date": pd.to_datetime(years, format="%Y"),
                                    "gdp_real": values}))
    return pd.concat(frames, ignore_index=True)


def test_forecasts_match_independent_scalar_ols_and_loss_oracles():
    data = _levels()
    result = study.evaluate_forecasts(data)
    forecasts = result["forecasts"]
    assert len(forecasts) == 3 * 3 * 25
    assert (forecasts.target_year - forecasts.origin_year).eq(1).all()
    assert forecasts.training_end_year.eq(forecasts.origin_year).all()
    assert forecasts.training_start_year.eq(1961).all()
    for code, levels in data.groupby("code"):
        # Direct ratios are independent of the implementation's log differencing.
        values = levels.gdp_real.to_numpy()
        growth = 100 * np.log(values[1:] / values[:-1])
        for target in range(2000, 2025):
            training = growth[:target - 1961]
            x, y = training[:-1], training[1:]
            slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
            intercept = y.mean() - slope * x.mean()
            expected = {"zero_growth": 0., "historical_mean": training.mean(),
                        "ar1": intercept + slope * training[-1]}
            block = forecasts[(forecasts.code == code) & (forecasts.target_year == target)].set_index("model")
            assert block.training_growth_observations.eq(len(training)).all()
            for model, prediction in expected.items():
                assert_allclose(block.loc[model, "forecast"], prediction, rtol=2e-12, atol=3e-12)
            assert_allclose(block.actual, growth[target - 1961], rtol=2e-12, atol=3e-12)
    for row in result["accuracy"].itertuples():
        values = forecasts[(forecasts.code == row.code) & (forecasts.model == row.model)]
        errors = values.actual.to_numpy() - values.forecast.to_numpy()
        assert row.n_forecasts == 25
        assert_allclose([row.rmse, row.mae, row.mean_error],
                        [np.linalg.norm(errors) / 5, np.abs(errors).sum() / 25, errors.sum() / 25])
    for row in result["pooled_accuracy"].itertuples():
        country = result["accuracy"].query("model == @row.model")
        assert row.n_countries == 3 and row.n_forecasts == 75
        assert_allclose(row.rmse, np.sqrt(np.sum(country.rmse ** 2) / 3))
    assert result["coverage"].initial_training_growth_observations.eq(39).all()
    assert len(result["exclusions"]) == 3 * 40
    assert not result["exclusions"].year.between(2000, 2024).any()


def test_future_observation_perturbation_cannot_change_earlier_forecasts():
    data = _levels()
    original = study.evaluate_forecasts(data)["forecasts"]
    changed = data.copy()
    future = changed.date.dt.year >= 2010
    changed.loc[future, "gdp_real"] *= np.exp(.3 * (changed.loc[future, "date"].dt.year - 2009))
    perturbed = study.evaluate_forecasts(changed)["forecasts"]
    columns = ["code", "model", "origin_year", "target_year", "forecast", "training_growth_observations"]
    earlier = original.target_year <= 2010
    assert_frame_equal(original.loc[earlier, columns], perturbed.loc[earlier, columns])
    # This perturbation does matter once the altered observation enters training.
    later = (original.target_year == 2011) & (original.model == "historical_mean")
    assert not np.allclose(original.loc[later, "forecast"], perturbed.loc[later, "forecast"])
    assert not np.allclose(original.loc[original.target_year == 2010, "actual"],
                           perturbed.loc[perturbed.target_year == 2010, "actual"])


def test_expanding_window_evaluation_is_prefix_invariant_and_order_independent():
    data = _levels()
    full = study.evaluate_forecasts(data)["forecasts"]
    short = study.evaluate_forecasts(data[data.date.dt.year <= 2010], evaluation_end=2010)["forecasts"]
    assert_frame_equal(full[full.target_year <= 2010].reset_index(drop=True), short)
    reordered = study.evaluate_forecasts(data.sample(frac=1, random_state=21))["forecasts"]
    assert_frame_equal(full, reordered)


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
def test_nonpositive_or_nonfinite_levels_are_rejected(bad):
    data = _levels()
    data.loc[7, "gdp_real"] = bad
    with pytest.raises(ValueError, match="finite and strictly positive"):
        study.evaluate_forecasts(data)


def test_missing_years_duplicates_and_bad_annual_dates_are_rejected():
    data = _levels()
    with pytest.raises(ValueError, match="missing calendar years"):
        study.evaluate_forecasts(data.drop(index=10))
    with pytest.raises(ValueError, match="Duplicate"):
        study.evaluate_forecasts(pd.concat([data, data.iloc[[10]]]))
    data.loc[10, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="January 1"):
        study.evaluate_forecasts(data)


def test_incomplete_evaluation_or_short_training_is_rejected():
    data = _levels()
    with pytest.raises(ValueError, match="complete evaluation years"):
        study.evaluate_forecasts(data[data.date.dt.year < 2024])
    with pytest.raises(ValueError, match="fewer than 20"):
        study.evaluate_forecasts(data[data.date.dt.year >= 1990])


def test_unidentified_ar_design_is_not_silently_accepted():
    data = _levels()
    data["gdp_real"] = 100.
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        study.evaluate_forecasts(data)


@pytest.mark.parametrize("kwargs", [{"min_train": 3}, {"min_train": True},
                                    {"evaluation_start": 2000.5},
                                    {"evaluation_end": 1999}])
def test_invalid_evaluation_design_is_rejected(kwargs):
    with pytest.raises(ValueError):
        study.evaluate_forecasts(_levels(), **kwargs)


def _write_fixture(directory, data):
    destination = directory / "data"
    destination.mkdir(exist_ok=True)
    raw = data.to_csv(index=False, date_format="%Y-%m-%d").encode()
    (destination / "cross_country_gdp.csv").write_bytes(raw)
    metadata = {"schema_version": 1, "countries": list(study.COUNTRIES),
                "start_year": 1960, "end_year": 2024,
                "csv_sha256": hashlib.sha256(raw).hexdigest()}
    (destination / "cross_country_gdp_metadata.json").write_text(json.dumps(metadata))


def test_loader_authenticates_bytes_and_declared_complete_grid(tmp_path, monkeypatch):
    _write_fixture(tmp_path, _levels())
    monkeypatch.setattr(study, "files", lambda name: tmp_path)
    observed = study.load_data()
    assert len(observed) == 195 and observed.attrs["schema_version"] == 1
    path = tmp_path / "data" / "cross_country_gdp.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        study.load_data()
    _write_fixture(tmp_path, _levels().query("code != 'BRA'"))
    with pytest.raises(ValueError, match="countries"):
        study.load_data()
    _write_fixture(tmp_path, _levels().loc[lambda x: x.date.dt.year >= 1961])
    with pytest.raises(ValueError, match="every year"):
        study.load_data()


@pytest.fixture(scope="module")
def exported_study(tmp_path_factory):
    output = tmp_path_factory.mktemp("cross_country_forecasts")
    return output, study.run_application(output)


def test_packaged_observations_and_exported_artifact_evidence(exported_study):
    output, result = exported_study
    data = study.load_data()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(data) == 195 and set(data.code) == set(study.COUNTRIES)
    assert data.attrs["source"]["indicator"] == "NY.GDP.MKTP.KN"
    assert manifest["design"] == study.DESIGN
    assert manifest["is_real_time"] is False and manifest["is_causal"] is False
    assert manifest["observed_data"] == data.attrs
    assert (output / "forecast_accuracy.png").stat().st_size > 1000
    assert "publication lags" in (output / "report.md").read_text()
    for name, record in manifest["artifacts"].items():
        raw = (output / name).read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((Path(study.__file__).parents[1] / name).read_bytes()).hexdigest() == digest
    cartridge = pocket.load(output / "cross_country_forecasting.pmz")
    assert cartridge.verify()
    assert cartridge["levels"].attrs == data.attrs
    assert_frame_equal(cartridge["levels"], result["levels"])
    assert_frame_equal(cartridge["forecasts"], result["forecasts"])
    assert len(cartridge) == 7


def test_numerical_tables_are_deterministic(exported_study, tmp_path):
    output, result = exported_study
    repeated = study.run_application(tmp_path)
    for name in ("levels", "growth", "forecasts", "accuracy", "pooled_accuracy", "coverage", "exclusions"):
        assert_frame_equal(result[name], repeated[name])
        assert (output / f"{name}.csv").read_bytes() == (tmp_path / f"{name}.csv").read_bytes()
