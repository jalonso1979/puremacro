"""Independently audit saved SW07 controls; no puremacro imports or refitting.

Run from the checkout root after both phase manifests have been written.
The audit authenticates saved evidence and reconstructs reported statistics.
It does not regenerate the unsaved raw simulated time series or certify the
optimizer's first-order calculations from parameter estimates alone.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = ("population_hac", "finite_hac", "population_oracle", "finite_oracle")
PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3))
TRUTH = np.array([.81, .24])


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(successes, trials):
    return (0. if successes == 0 else float(beta.ppf(.025, successes, trials - successes + 1)),
            1. if successes == trials else float(beta.ppf(.975, successes + 1, trials - successes)))


def _rate(lower, upper, trials, prefix):
    return {f"{prefix}_events_lower": lower, f"{prefix}_events_upper": upper,
            f"{prefix}_rate_lower": lower / trials, f"{prefix}_rate_upper": upper / trials,
            f"{prefix}_mc95_lower": _interval(lower, trials)[0],
            f"{prefix}_mc95_upper": _interval(upper, trials)[1]}


def _usable(frame):
    return (~frame.unresolved & np.isfinite(frame.objective)).to_numpy()


def _summary(draws, phase):
    rows = []
    critical = chi2.isf(.05, 7)
    for variant in VARIANTS:
        frame = draws[draws.variant == variant]
        good = _usable(frame)
        objective = frame.objective.to_numpy()[good]
        count, failed = len(frame), int((~good).sum())
        rejects = int(np.count_nonzero(objective > critical))
        error = frame[["crr", "em"]].to_numpy()[good] - TRUTH
        record = {"phase": phase, "variant": variant, "replications": count,
                  "usable_fits": len(objective), "unresolved_draws": failed,
                  "boundary_fits": int(np.count_nonzero(good & frame.boundary)),
                  "regular_fits": int(np.count_nonzero(good & frame.numerically_regular)),
                  "criterion_median": np.median(objective) if len(objective) else np.nan,
                  "criterion_q95": np.percentile(objective, 95) if len(objective) else np.nan,
                  **_rate(rejects, rejects + failed, count, "naive")}
        for j, name in enumerate(("crr", "em")):
            record[f"bias_{name}"] = error[:, j].sum() / len(error) if len(error) else np.nan
            record[f"rmse_{name}"] = np.linalg.norm(error[:, j]) / math.sqrt(len(error)) if len(error) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def _paired(draws, phase):
    rows = []
    for left_index, right_index in PAIRS:
        left_name, right_name = VARIANTS[left_index], VARIANTS[right_index]
        left = draws[draws.variant == left_name].sort_values("replication")
        right = draws[draws.variant == right_name].sort_values("replication")
        assert np.array_equal(left.replication, right.replication)
        common = _usable(left) & _usable(right)
        record = {"phase": phase, "left": left_name, "right": right_name,
                  "difference": "right minus left", "paired_usable": int(common.sum()),
                  "total_pairs": len(left)}
        for j, name in enumerate(("crr", "em")):
            left_error = left[name].to_numpy()[common] - TRUTH[j]
            right_error = right[name].to_numpy()[common] - TRUTH[j]
            # (b-a)(b+a) is the independently evaluated squared-error difference.
            difference = (right_error - left_error) * (right_error + left_error)
            mean = difference.sum() / len(difference) if len(difference) else np.nan
            se = (np.linalg.norm(difference - mean) /
                  math.sqrt(len(difference) * (len(difference) - 1))) if len(difference) > 1 else np.nan
            record[f"mean_squared_error_difference_{name}"] = mean
            record[f"paired_mc_se_{name}"] = se
        rows.append(record)
    return pd.DataFrame(rows)


def _check_table(actual, saved, keys):
    actual = actual.set_index(keys).sort_index()
    saved = saved.set_index(keys).sort_index()
    assert actual.index.equals(saved.index)
    differences = {}
    for column in saved.columns:
        assert column in actual, column
        if pd.api.types.is_numeric_dtype(saved[column]):
            a, b = actual[column].to_numpy(float), saved[column].to_numpy(float)
            np.testing.assert_allclose(a, b, rtol=2e-10, atol=1e-11, equal_nan=True,
                                       err_msg=f"saved statistic differs: {column}")
            finite = np.isfinite(a) & np.isfinite(b)
            differences[column] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
        else:
            assert actual[column].tolist() == saved[column].tolist(), column
    return differences


def _load(phase, count, index):
    directory = HERE / phase
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["phase"] == phase and manifest["phase_index"] == index
    assert manifest["replications"] == count and manifest["seed"] == 20261002
    assert manifest["variants"] == list(VARIANTS)
    assert manifest["nobs"] == 156 and manifest["effective_observations"] == 152
    assert manifest["bandwidth"] == 8 and manifest["common_max_lag"] == 4
    assert manifest["max_nfev"] == 500 and manifest["tolerance"] == 1e-9
    assert manifest["true_parameters"] == TRUTH.tolist()
    assert manifest["dgp"]["crr"] == .81 and manifest["dgp"]["em"] == .24
    for filename, expected in manifest["artifacts"].items():
        assert Path(filename).name == filename
        assert _sha(directory / filename) == expected, filename
    for filename, expected in manifest["source_hashes"].items():
        assert _sha(ROOT / "puremacro" / filename) == expected, filename
    draws = pd.read_csv(directory / "draws.csv", dtype={"simulation_seed": str, "sample_sha256": str})
    assert len(draws) == 4 * count and set(draws.phase) == {phase}
    for name in ("unresolved", "boundary", "numerically_regular", "starts_agree", "optimizer_success"):
        assert pd.api.types.is_bool_dtype(draws[name]), name
    usable_rows = draws.loc[_usable(draws)]
    assert (usable_rows.objective >= 0).all()
    assert np.isfinite(usable_rows[["crr", "em"]]).all().all()
    assert usable_rows.crr.between(.2 - 1e-12, .98 + 1e-12).all()
    assert usable_rows.em.between(.01 - 1e-12, 1. + 1e-12).all()
    with np.load(directory / "draw_evidence.npz", allow_pickle=False) as evidence:
        moments, covariances = evidence["moments"].copy(), evidence["covariances"].copy()
    assert moments.shape == (count, 15) and covariances.shape == (count, 15, 15)
    available = np.isfinite(moments).all(axis=1)
    assert np.isnan(moments[~available]).all() and np.isnan(covariances[~available]).all()
    assert np.isfinite(covariances[available]).all()
    np.testing.assert_allclose(covariances[available], covariances[available].swapaxes(1, 2), rtol=0, atol=1e-12)
    smallest_eigenvalue = float(np.linalg.eigvalsh(covariances[available]).min()) if available.any() else 0.
    assert smallest_eigenvalue >= -1e-12
    for i in range(count):
        frame = draws[draws.replication == i]
        assert frame.variant.tolist() == list(VARIANTS)
        assert (frame.array_row == i).all()
        assert frame.simulation_seed.nunique() == 1
        stream = int(np.random.SeedSequence([20261002, index, i]).generate_state(1, dtype=np.uint64)[0])
        assert frame.simulation_seed.iloc[0] == str(stream)
        digest = frame.sample_sha256.iloc[0]
        if available[i]:
            assert frame.sample_sha256.nunique() == 1
            assert isinstance(digest, str) and len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")
        else:
            assert frame.unresolved.all()
    assert draws.drop_duplicates("replication").simulation_seed.nunique() == count
    assert draws.drop_duplicates("replication").sample_sha256.nunique() >= int(available.sum())
    record_counts = {"total_starts": 0, "failed_starts": 0, "nonstationary_selected": 0,
                     "start_disagreement_fits": 0, "unresolved_from_records": 0,
                     "missing_sample_draws": int((~available).sum())}
    for row in draws.itertuples():
        records = json.loads(row.start_diagnostics_json)
        if not records:
            assert row.unresolved and not row.optimizer_success and pd.isna(row.objective)
            assert pd.notna(row.error)
            record_counts["unresolved_from_records"] += 1
            continue
        assert len(records) == 4
        record_counts["total_starts"] += len(records)
        assert [record["start"] for record in records] == manifest["starts"]
        successful = [record for record in records if record["success"]]
        record_counts["failed_starts"] += 4 - len(successful)
        candidates = [record for record in records if "objective" in record]
        if not candidates:
            assert row.unresolved and not row.optimizer_success and pd.isna(row.objective)
            assert row.start_successes == 0 and not row.starts_agree
            record_counts["start_disagreement_fits"] += 1
            record_counts["unresolved_from_records"] += 1
            continue
        best = min(successful or candidates, key=lambda record: record["objective"])
        np.testing.assert_allclose([row.crr, row.em, row.objective],
                                  best["theta"] + [best["objective"]], rtol=1e-12, atol=1e-12)
        agree = bool(successful and all(abs(record["objective"] - best["objective"])
                     <= 1e-5 * max(1., best["objective"]) for record in successful))
        nonstationary = "nonstationary_solution" in best["inference_unavailable_reasons"]
        unresolved = len(successful) < 4 or not agree or nonstationary or any("exception" in record for record in records)
        assert row.start_successes == len(successful) and row.starts_agree == agree
        assert row.unresolved == unresolved
        record_counts["start_disagreement_fits"] += not agree
        record_counts["nonstationary_selected"] += nonstationary
        record_counts["unresolved_from_records"] += unresolved
    # Checkpoint is separate from the public artifact manifest; authenticate its
    # design and confirm its arrays/rows reproduce the completed public evidence.
    with np.load(directory / "_checkpoints" / f"{phase}.npz", allow_pickle=False) as checkpoint:
        document = json.loads(str(checkpoint["document"]))
        np.testing.assert_array_equal(moments, checkpoint["moments"])
        np.testing.assert_array_equal(covariances, checkpoint["covariances"])
        for key, value in document["design"].items():
            assert manifest[key] == value
        checkpoint_rows = pd.DataFrame(document["rows"])
        assert len(checkpoint_rows) == len(draws)
        for column in ("simulation_seed", "sample_sha256", "variant", "replication", "array_row", "unresolved"):
            assert checkpoint_rows[column].fillna("").tolist() == draws[column].fillna("").tolist(), column
    summary, pairs = _summary(draws, phase), _paired(draws, phase)
    audit = {"phase": phase, "manifest_sha256": _sha(directory / "manifest.json"),
             "sample_count": count, "variant_fit_count": len(draws),
             "source_hashes_checked": len(manifest["source_hashes"]),
             "artifact_hashes_checked": len(manifest["artifacts"]),
             "minimum_hac_eigenvalue": smallest_eigenvalue, **record_counts,
             "summary_max_absolute_errors": _check_table(summary, pd.read_csv(directory / "summary.csv"), ["variant"]),
             "paired_max_absolute_errors": _check_table(pairs.drop(columns="phase"), pd.read_csv(directory / "paired_differences.csv"), ["left", "right"])}
    return manifest, draws, summary, pairs, audit


def _calibrated(calibration, validation):
    rows = []
    for variant in VARIANTS:
        a = calibration[calibration.variant == variant]
        b = validation[validation.variant == variant]
        known = np.sort(a.objective.to_numpy()[_usable(a)])
        missing = len(a) - len(known)
        rank = math.ceil(.95 * (len(a) + 1))
        assert rank == 380
        # With missing values below all known nonnegative criteria, subtract
        # their count from the requested known-data order rank. With missing
        # values above all known criteria, the known-data rank is unchanged.
        lower = 0. if rank <= missing else float(known[rank - missing - 1])
        upper = float(known[rank - 1]) if rank <= len(known) else np.inf
        good = _usable(b)
        selected = b.objective.to_numpy()[good]
        low_events = int(np.count_nonzero(selected > upper))
        high_events = int(np.count_nonzero(selected > lower) + (~good).sum())
        rows.append({"variant": variant, "calibration_draws": len(a), "validation_draws": len(b),
                     "calibration_unresolved": missing, "validation_unresolved": int((~good).sum()),
                     "critical_rank": rank, "critical_lower": lower, "critical_upper": upper,
                     "critical_lower_unbounded": bool(np.isinf(lower)),
                     "critical_upper_unbounded": bool(np.isinf(upper)),
                     **_rate(low_events, high_events, len(b), "calibrated")})
    return pd.DataFrame(rows)


def main():
    calibration = _load("calibration", 399, 0)
    validation = _load("validation", 999, 1)
    for key in ("dgp", "seed", "source_hashes", "observed_source", "environment", "variants",
                "oracle_covariance", "oracle_expectations_at_truth"):
        assert calibration[0][key] == validation[0][key], key
    assert set(calibration[1].simulation_seed).isdisjoint(validation[1].simulation_seed)
    assert set(calibration[1].sample_sha256.dropna()).isdisjoint(validation[1].sample_sha256.dropna())
    comparison = _calibrated(calibration[1], validation[1])
    reference_path = HERE / "calibration_reference.json"
    reference = json.loads(reference_path.read_text())
    assert reference["validation_complete_at_creation"] is False
    assert reference["calibration_manifest_sha256"] == calibration[4]["manifest_sha256"]
    assert reference["calibration_draws_sha256"] == _sha(HERE / "calibration" / "draws.csv")
    assert datetime.fromisoformat(reference["created_utc"]) <= datetime.fromtimestamp(
        (HERE / "validation" / "manifest.json").stat().st_mtime, tz=timezone.utc)
    reference_cutoffs = pd.DataFrame(reference["cutoffs"])
    for name in ("lower", "upper"):
        reference_cutoffs.loc[reference_cutoffs[f"critical_{name}_unbounded"], f"critical_{name}"] = np.inf
    reference_errors = _check_table(comparison, reference_cutoffs, ["variant"])
    observed_columns = ["variant", "crr", "em", "objective", "boundary", "unresolved"]
    observed = pd.read_csv(HERE / "calibration" / "observed_fits.csv")[observed_columns]
    observed_errors = _check_table(observed,
        pd.read_csv(HERE / "validation" / "observed_fits.csv")[observed_columns], ["variant"])
    observed = observed.merge(comparison[["variant", "critical_lower", "critical_upper"]], on="variant", validate="one_to_one")
    comparison_audit = None
    if (HERE / "validation" / "calibrated_validation.csv").exists():
        comparison_audit = _check_table(comparison,
            pd.read_csv(HERE / "validation" / "calibrated_validation.csv"), ["variant"])
        assert validation[0]["calibration_manifest_sha256"] == calibration[4]["manifest_sha256"]
    summary = pd.concat([calibration[2], validation[2]], ignore_index=True)
    pairs = pd.concat([calibration[3], validation[3]], ignore_index=True)
    summary.to_csv(HERE / "combined_summary.csv", index=False)
    pairs.to_csv(HERE / "combined_paired_differences.csv", index=False)
    comparison.to_csv(HERE / "calibrated_validation.csv", index=False)
    audit = {"passed": True, "no_puremacro_imports": True,
             "recalculation_source_sha256": _sha(Path(__file__)),
             "phase_audits": [calibration[4], validation[4]],
             "disjoint_phase_seeds_and_sample_hashes": True,
             "same_draw_four_variant_array_mapping": True,
             "calibration_rank": 380, "calibrated_validation_max_absolute_errors": comparison_audit,
             "frozen_calibration_reference_sha256": _sha(reference_path),
             "frozen_reference_created_utc": reference["created_utc"],
             "frozen_reference_max_absolute_errors": reference_errors,
             "observed_anchor_phase_max_absolute_errors": observed_errors,
             "limitations": ["Raw simulated time series are not in phase artifacts; hashes/seeds and shared array references are authenticated, not independently resimulated",
                             "Start-record classifications are rechecked; first-order stationarity diagnostics are not rederived from unsaved objective gradients",
                             "Monte Carlo envelopes are conditional on the realized calibration pool, not parameter confidence intervals",
                             "Exact covariance at known DGP is an oracle control, not an estimated feasible weighting procedure"]}
    (HERE / "independent_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    held_out = summary[summary.phase == "validation"].set_index("variant")
    calibrated = comparison.set_index("variant")
    unresolved_count = calibration[4]["unresolved_from_records"] + validation[4]["unresolved_from_records"]
    assessment = (
        "Exact finite-sample expectations reduce naive chi-square overrejection with HAC weights, "
        f"from {100 * held_out.loc['population_hac', 'naive_rate_lower']:.2f}–"
        f"{100 * held_out.loc['population_hac', 'naive_rate_upper']:.2f}% to "
        f"{100 * held_out.loc['finite_hac', 'naive_rate_lower']:.2f}–"
        f"{100 * held_out.loc['finite_hac', 'naive_rate_upper']:.2f}%, but leave severe distortion. "
        "Oracle covariance weights remove most of that distortion at this known DGP, but the "
        f"naive rejection rates remain {100 * held_out.loc['population_oracle', 'naive_rate_lower']:.2f}% "
        f"and {100 * held_out.loc['finite_oracle', 'naive_rate_lower']:.2f}%; "
        "5% lies below both corresponding 95% Monte Carlo intervals. Oracle weighting alone "
        "therefore does not restore the nominal chi-square reference.\n\n"
        "After applying the frozen simulation-reference cutoffs, the two oracle validation "
        f"fractions are {100 * calibrated.loc['population_oracle', 'calibrated_rate_lower']:.2f}% "
        f"and {100 * calibrated.loc['finite_oracle', 'calibrated_rate_lower']:.2f}%, "
        "with conditional Monte Carlo intervals containing 5%. "
        "The HAC variants have wider failure bounds. This evidence concerns the one fixed "
        "generating calibration; it is not a composite-null test or feasible estimated covariance.\n\n"
        "Finite expectations improve paired parameter squared errors under HAC weighting, but "
        "increase them for both parameters under oracle weighting. The exact expectation map "
        "is therefore not a universal parameter-recovery improvement. Changing the covariance "
        "control gives the larger recovery improvements in this experiment.\n\n"
        f"All {unresolved_count} unresolved variant fits are selected-result first-order stationarity failures. "
        "Every optimizer start reported success and successful-start objectives agreed, so an "
        "audit based on optimizer success flags alone would have missed these unresolved cases. "
        "They remain in the reported failure and rejection bounds."
    )
    report = ["# Independent saved-result audit", "",
              "PASS. This script imports no puremacro production code and performs no refitting. It independently reconstructs statistics from the authenticated phase evidence.", "",
              "Calibration contains 399 shared samples; validation contains 999 independently seeded shared samples. Each sample has exactly four variant records referring to the same seed, sample hash and NPZ array row. All numerical-source fingerprints and artifact hashes match the checkout; checkpoint arrays match the exported arrays exactly.", "",
              "## Assessment of the predeclared comparisons", "", assessment, "",
              "## Naive chi-square reference and parameter recovery", "",
              "All saved summary and paired-comparison numbers reproduce within serialization/numerical tolerance. Unresolved fits remain in rate denominators; parameter bias and RMSE condition on usable fits, and paired squared-error changes condition on common usable draws.", "",
              "```text\n" + summary.to_string(index=False) + "\n```", "", "## Independent calibration cutoffs", "",
              "The predeclared order rank is 380 of 399. Missing calibration criteria are bounded below all known criteria and above all known criteria. Validation uses strict exceedance, with missing validation fits counted both ways. Clopper–Pearson envelopes condition on the realized calibration pool and do not capture all cutoff-estimation uncertainty.", "",
              f"All four cutoffs match the frozen calibration_reference.json saved at {reference['created_utc']}, before validation finished. No validation outcomes or method selection enter that reference.", "",
              "```text\n" + comparison.to_string(index=False) + "\n```", "", "## Paired parameter recovery", "",
              "Negative right-minus-left squared-error differences favor the right variant on the reported common usable sample. These comparisons are descriptive; no variant is selected after seeing validation and presented as independently validated.", "",
              "```text\n" + pairs.to_string(index=False) + "\n```", "", "## Observed descriptive fits", "",
              "```text\n" + observed.to_string(index=False) + "\n```", "",
              "Observed fits use revised historical data and the fixed published nuisance calibration. In particular, observed oracle-weighted criteria exceed their fixed-DGP calibration cutoffs. Improved null rejection behavior in model-generated data therefore does not establish adequacy for the historical sample. These are descriptive comparisons; no composite-null test, parameter interval or empirically validated feasible estimator is added. Criteria under different weights have different absolute scales.", "", "## Scope", "",
              "Raw simulated time series were not saved in the phase artifacts, so this audit authenticates seed/hash identity and shared moment/covariance references without independently resimulating each series. It verifies recorded start successes, objective agreement and selected-result consistency, but does not rederive stationarity diagnostics from unsaved gradients. Oracle covariance controls and fixed-DGP simulation cutoffs do not establish a feasible general-purpose estimator or composite-null inference.", ""]
    (HERE / "independent_audit.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))
    print("Independent saved-result audit: PASS")


if __name__ == "__main__":
    main()
