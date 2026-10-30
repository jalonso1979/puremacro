"""A convergence verdict requires the registered experiment and complete data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.reference_validation.check_sovereign_sensitive_grids import (
    BASELINE, DESIGN, EXPLICIT_GRID_EQUIVALENCE, EXTENDED_GRIDS, GRIDS, MONOTONE_EQUIVALENCE,
    MONOTONE_SOURCE, SCENARIOS, TOLERANCES, assess, combine, validate_prior_sources,
)


@pytest.fixture
def record():
    path = Path(__file__).parent / 'fixtures/sovereign_brief/sweep.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    result = dict(schema=1, baseline=deepcopy(BASELINE), design=deepcopy(DESIGN),
                  tolerances=dict(TOLERANCES), grids=[list(g) for g in GRIDS], scenarios={})
    # Reuse real per-seed measurements at every grid to isolate the assessment
    # contract. These synthetic records are never evidence of grid convergence.
    for name, overrides in SCENARIOS.items():
        cases = {}
        for b, y in GRIDS:
            case = deepcopy(frozen['scenarios'][name]['cases']['6401x641'])
            case.update(n_b=b, n_y=y)
            case['parameters'].update(n_b=b, n_y=y)
            cases[f'{b}x{y}'] = case
        result['scenarios'][name] = dict(overrides=dict(overrides), cases=cases)
    return result


def test_registered_zero_grid_change_passes(record):
    assert assess(record)['passed']


@pytest.mark.parametrize('change', ['empty', 'baseline', 'missing_grid', 'overrides',
                                   'parameters', 'seeds', 'truncated', 'nonfinite'])
def test_incompatible_or_incomplete_study_cannot_pass(record, change):
    scenario = record['scenarios']['theta=0.5']
    case = scenario['cases']['12801x1281']
    if change == 'empty':
        record['scenarios'].clear()
    elif change == 'baseline':
        record['baseline']['beta'] = .9
    elif change == 'missing_grid':
        del scenario['cases']['12801x1281']
    elif change == 'overrides':
        scenario['overrides']['theta'] = .15
    elif change == 'parameters':
        case['parameters']['theta'] = .15
    elif change == 'seeds':
        case['simulation']['seeds'][0] = -1
    else:
        values = case['simulation']['per_seed'][next(iter(TOLERANCES))]
        if change == 'truncated':
            values.pop()
        else:
            values[0] = float('nan')
    with pytest.raises(ValueError):
        assess(record)


def test_failed_numerical_solve_prevents_convergence_verdict(record):
    record['scenarios']['theta=0.5']['cases']['12801x1281']['solver']['converged'] = False
    assert not assess(record)['passed']


def test_empty_record_list_cannot_be_combined():
    with pytest.raises(ValueError, match='completed record'):
        combine([])


def test_extended_ladder_requires_income_refinement_at_the_largest_debt_grid(record):
    record['grids'] = [list(g) for g in EXTENDED_GRIDS]
    for scenario in record['scenarios'].values():
        for b, y in EXTENDED_GRIDS[len(GRIDS):]:
            case = deepcopy(scenario['cases']['12801x1281'])
            case.update(n_b=b, n_y=y)
            case['parameters'].update(n_b=b, n_y=y)
            scenario['cases'][f'{b}x{y}'] = case
    assert assess(record)['passed']
    coarse = record['scenarios']['theta=0.5']['cases']['25601x641']
    coarse['simulation']['per_seed']['spread_sd_pp'] = [10.] * len(DESIGN['seeds'])
    result = assess(record)
    checks = result['scenarios']['theta=0.5']['comparisons']
    assert checks['final_asset_refinement']['passed']
    assert not checks['final_income_refinement']['passed']
    assert not result['passed']


def test_historical_provenance_accepts_only_the_checked_batching_change():
    old, new = MONOTONE_EQUIVALENCE
    validate_prior_sources({MONOTONE_SOURCE: old, 'unchanged': 'same'},
                           {MONOTONE_SOURCE: new, 'unchanged': 'same'})
    with pytest.raises(ValueError, match='unsupported historical source'):
        validate_prior_sources({MONOTONE_SOURCE: old}, {MONOTONE_SOURCE: 'unknown'})
    with pytest.raises(ValueError, match='unsupported historical source'):
        validate_prior_sources({MONOTONE_SOURCE: old, 'other': 'old'},
                               {MONOTONE_SOURCE: new, 'other': 'new'})
    with pytest.raises(ValueError, match='source keys'):
        validate_prior_sources({MONOTONE_SOURCE: old}, {})


def test_historical_uniform_grid_sources_accept_the_checked_optional_grid_branch():
    old, new = EXPLICIT_GRID_EQUIVALENCE
    path = 'puremacro/sovereign/arellano.py'
    validate_prior_sources({path: old}, {path: new})
    with pytest.raises(ValueError, match='unsupported historical source'):
        validate_prior_sources({path: old}, {path: 'unknown'})


def test_frozen_initial_ladder_preserves_failed_debt_refinements():
    from tools.reference_validation.check_sovereign_sensitive_grids import source_hashes

    path = Path(__file__).parent / 'fixtures/sovereign_brief/sensitive_grid_initial.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    validate_prior_sources(frozen['source_sha256'], source_hashes())
    checked = assess(deepcopy(frozen))
    assert not checked['passed']
    for name, scenario in checked['scenarios'].items():
        expected = frozen['scenarios'][name]['comparisons']
        for label, comparison in scenario['comparisons'].items():
            assert comparison['passed'] == expected[label]['passed']
            for moment, values in comparison['moments'].items():
                saved = expected[label]['moments'][moment]
                assert values['status'] == saved['status']
                for key in ('change', 'ci95', 'absolute_change_bound', 'tolerance'):
                    assert values[key] == pytest.approx(saved[key], abs=1e-12, rel=0)
        assert scenario['comparisons']['final_income_refinement']['passed']
        assert not scenario['comparisons']['final_asset_refinement']['passed']


def test_extension_retains_each_historical_case_and_its_source_hashes(tmp_path, monkeypatch):
    from tools.reference_validation import check_sovereign_sensitive_grids as study

    path = Path(__file__).parent / 'fixtures/sovereign_brief/sensitive_grid_initial.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    name = 'theta=0.5'
    generated = deepcopy(frozen)
    generated['source_sha256'] = study.source_hashes()
    generated['scenarios'] = {name: dict(overrides=SCENARIOS[name], cases={})}
    for b, y in EXTENDED_GRIDS[len(GRIDS):]:
        case = deepcopy(frozen['scenarios'][name]['cases']['12801x1281'])
        case.update(n_b=b, n_y=y)
        case['parameters'].update(n_b=b, n_y=y)
        generated['scenarios'][name]['cases'][f'{b}x{y}'] = case
    monkeypatch.setattr(study, 'build_record', lambda *a, **k: generated)
    # The real record may have been generated on a different CI interpreter.
    monkeypatch.setattr(study, '_versions', lambda: frozen['versions'])
    result = study.extend(tmp_path, path, {name: SCENARIOS[name]})
    assert result['initial_record_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key, case in frozen['scenarios'][name]['cases'].items():
        retained = deepcopy(result['scenarios'][name]['cases'][key])
        assert retained.pop('source_sha256') == frozen['source_sha256']
        assert retained == case
    assert not result['scenarios'][name]['comparisons']['initial_final_asset_refinement']['passed']
    assert result['scenarios'][name]['comparisons']['final_asset_refinement']['passed']
