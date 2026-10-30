"""Fixed-design refinement of the two grid-sensitive sovereign brief cases."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from puremacro.examples.first_brief_sovereign import (
    BASELINE, DESIGN, TOLERANCES, _package_version, _versions,
    build_record, paired_change, source_hashes,
)

SCENARIOS = {'theta=0.5': {'theta': .5}, 'rho_y=0.97': {'rho_y': .97}}
GRIDS = [(6401, 641), (12801, 641), (6401, 1281), (12801, 1281)]
EXTENDED_GRIDS = GRIDS + [(25601, 641), (25601, 1281)]
MONOTONE_SOURCE = 'puremacro/sovereign/_monotone.py'
# Income batching preserves every column's arithmetic and tie-breaking. The
# pinned original, dense search and batched search are compared in tests.
MONOTONE_EQUIVALENCE = (
    'fb04bb672f59c89ec40bd51179df3a249df09e0ddeb7cd9102d6288e7b251464',
    'cb0655d94f4f5dc01c1f71979e8bb71551d98ed5ac90e18a79d4f420b3c7e467',
)


def assess(record):
    """Recompute the unchanged registered decision from complete per-seed data."""
    if record.get('schema') != 1 or record.get('baseline') != BASELINE:
        raise ValueError('record does not match the registered schema and baseline')
    if not record.get('scenarios'):
        raise ValueError('at least one registered scenario is required')
    if record['design'] != DESIGN or record['tolerances'] != TOLERANCES:
        raise ValueError('record does not match the registered design and tolerances')
    extended = record['grids'] == [list(g) for g in EXTENDED_GRIDS]
    grids = EXTENDED_GRIDS if extended else GRIDS
    if record['grids'] != [list(g) for g in grids]:
        raise ValueError('record does not contain the registered crossed grid ladder')
    passed = True
    for name, scenario in record['scenarios'].items():
        if name not in SCENARIOS or scenario['overrides'] != SCENARIOS[name]:
            raise ValueError('record has an unregistered scenario')
        cases = scenario['cases']
        if set(cases) != {f'{b}x{y}' for b, y in grids}:
            raise ValueError(f'{name} is missing one or more required grids')
        for b, y in grids:
            case = cases[f'{b}x{y}']
            if 'source_sha256' in case:
                validate_prior_sources(case['source_sha256'], record['source_sha256'])
            if (case['n_b'], case['n_y']) != (b, y) or case['simulation']['seeds'] != DESIGN['seeds']:
                raise ValueError('grid or seed identity differs from its registered case')
            expected = {**BASELINE, **scenario['overrides'], 'n_b': b, 'n_y': y}
            if any(case['parameters'].get(key) != value for key, value in expected.items()):
                raise ValueError('case parameters differ from the registered calibration')
            for moment in TOLERANCES:
                values = np.asarray(case['simulation']['per_seed'][moment], dtype=float)
                if values.shape != (len(DESIGN['seeds']),) or not np.all(np.isfinite(values)):
                    raise ValueError('each moment requires one finite observation per registered seed')
        # build_record's two-grid display is not the validation criterion for
        # this crossed ladder. Require both final refinements explicitly.
        scenario.pop('grid_change', None)
        checks = {}
        comparisons = [
            ('assets_at_641_income_states', '6401x641', '12801x641'),
        ]
        if extended:
            comparisons += [
                ('initial_final_asset_refinement', '6401x1281', '12801x1281'),
                ('initial_final_income_refinement', '12801x641', '12801x1281'),
                ('extended_assets_at_641_income_states', '12801x641', '25601x641'),
                ('final_asset_refinement', '12801x1281', '25601x1281'),
                ('final_income_refinement', '25601x641', '25601x1281'),
            ]
        else:
            comparisons += [
                ('final_asset_refinement', '6401x1281', '12801x1281'),
                ('final_income_refinement', '12801x641', '12801x1281'),
            ]
        for label, coarse, fine in comparisons:
            checks[label] = dict(coarse=coarse, fine=fine, **paired_change(cases[coarse], cases[fine]))
        scenario['comparisons'] = checks
        scenario['passed'] = all(checks[k]['passed'] for k in ('final_asset_refinement', 'final_income_refinement'))
        for case in cases.values():
            solver = case['solver']
            scenario['passed'] &= (solver['converged'] and solver['monotone_certification_gap'] <= 1e-11
                                    and solver['exhaustive_bellman_residual'] <= 1e-8)
        passed &= scenario['passed']
        print(name, {k: v['passed'] for k, v in checks.items()}, flush=True)
    record['passed'] = bool(passed)
    record['status'] = 'passed' if passed else 'convergence_not_established'
    record['assessment_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def validate_prior_sources(prior, current):
    """Permit only the explicitly checked batching change when extending a run."""
    if set(prior) != set(current):
        raise ValueError('historical source keys differ')
    for path in current:
        if prior[path] == current[path]:
            continue
        if path != MONOTONE_SOURCE or (prior[path], current[path]) != MONOTONE_EQUIVALENCE:
            raise ValueError(f'unsupported historical source change: {path}')


def extend(output: Path, initial_path: Path, scenarios=None):
    """Keep the failed first ladder and solve only the additional debt grids.

    Historical cases retain their actual source hashes; they are never relabeled
    as generated by the newer batching implementation.
    """
    initial = json.loads(initial_path.read_text(encoding='utf-8'))
    if initial['grids'] != [list(g) for g in GRIDS]:
        raise ValueError('extension requires a completed initial crossed ladder')
    assess(initial)
    current_sources = source_hashes()
    validate_prior_sources(initial['source_sha256'], current_sources)
    if initial['versions'] != _versions() or initial['package_version'] != _package_version():
        raise ValueError('extension environment differs from the historical run')
    scenarios = SCENARIOS if scenarios is None else scenarios
    if not set(scenarios) <= set(initial['scenarios']):
        raise ValueError('initial record lacks a requested scenario')
    output.mkdir(parents=True, exist_ok=True)
    initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    protocol = dict(design=DESIGN, scenarios=scenarios, grids=EXTENDED_GRIDS,
                    tolerances=TOLERANCES, source_sha256=current_sources,
                    initial_record_sha256=initial_sha,
                    rule='abs(paired mean change) + 95% t half-width <= tolerance')
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(protocol)):
        raise ValueError('existing protocol differs; use a new output directory')
    path.write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    record = build_record(EXTENDED_GRIDS[len(GRIDS):], design=DESIGN, scenarios=scenarios,
                          partial=output / 'extension.partial.json',
                          log=lambda message: print(message, flush=True))
    for key in ('versions', 'package_version', 'baseline', 'design', 'tolerances'):
        if record[key] != initial[key]:
            raise ValueError(f'extension changed {key}')
    for name, scenario in record['scenarios'].items():
        old_cases = copy.deepcopy(initial['scenarios'][name]['cases'])
        for case in old_cases.values():
            case['source_sha256'] = dict(initial['source_sha256'])
        for case in scenario['cases'].values():
            case['source_sha256'] = dict(current_sources)
        scenario['cases'] = {**old_cases, **scenario['cases']}
    record['grids'] = [list(g) for g in EXTENDED_GRIDS]
    record['initial_record_sha256'] = initial_sha
    old_loads = {key: value for key, value in initial['load_averages'].items()
                 if any(key.startswith(name+' ') for name in scenarios)}
    record['load_averages'] = {**old_loads, **record['load_averages']}
    peaks = [v for v in (initial['peak_rss_gb'], record['peak_rss_gb']) if v is not None]
    record['peak_rss_gb'] = max(peaks) if peaks else None
    record = assess(record)
    (output / 'sensitive_grid_convergence.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return record


def combine(paths):
    """Combine separately run scenarios without rewriting historical provenance."""
    if not paths:
        raise ValueError('at least one completed record is required')
    records = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    result = copy.deepcopy(records[0])
    result['scenarios'] = {}
    result['load_averages'] = {}
    result['input_record_sha256'] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    for record in records:
        if record.get('initial_record_sha256') != result.get('initial_record_sha256'):
            raise ValueError('cannot combine different initial study records')
        for key in ('schema', 'source_sha256', 'design', 'baseline', 'tolerances', 'grids', 'versions', 'package_version'):
            if record[key] != result[key]:
                raise ValueError(f'cannot combine incompatible {key}')
        for name, scenario in record['scenarios'].items():
            # A first process may have started another scenario before stopping.
            # Empty entries contain no observations and carry no evidentiary value.
            if not scenario['cases']:
                continue
            if name in result['scenarios']:
                raise ValueError(f'duplicate scenario {name}; provide disjoint complete scenario records')
            result['scenarios'][name] = copy.deepcopy(scenario)
        result['load_averages'].update(record['load_averages'])
    if set(result['scenarios']) != set(SCENARIOS):
        raise ValueError('both registered scenarios are required')
    peaks = [r['peak_rss_gb'] for r in records if r['peak_rss_gb'] is not None]
    result['peak_rss_gb'] = max(peaks) if peaks else None
    return assess(result)


def run(output: Path, scenarios=None):
    output.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS if scenarios is None else scenarios
    protocol = dict(design=DESIGN, scenarios=scenarios, grids=GRIDS,
                     tolerances=TOLERANCES, source_sha256=source_hashes(),
                     rule='abs(paired mean change) + 95% t half-width <= tolerance')
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(protocol)):
        raise ValueError('existing protocol differs; use a new output directory')
    path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    record = assess(build_record(GRIDS, design=DESIGN, scenarios=scenarios,
                                 partial=output / 'sensitive_grid.partial.json',
                                 log=lambda message: print(message, flush=True)))
    (output / 'sensitive_grid_convergence.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=list(SCENARIOS))
    parser.add_argument('--combine', nargs='+', type=Path,
                        help='combine disjoint completed scenario records into --output directory')
    parser.add_argument('--extend-from', type=Path,
                        help='retain an initial record and refine to 25601 debt nodes')
    args = parser.parse_args()
    selected = None if args.scenario is None else {args.scenario: SCENARIOS[args.scenario]}
    if args.combine and args.extend_from:
        parser.error('--combine and --extend-from are mutually exclusive')
    if args.combine:
        result = combine(args.combine)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'sensitive_grid_convergence.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    elif args.extend_from:
        result = extend(args.output, args.extend_from, selected)
    else:
        result = run(args.output, selected)
    raise SystemExit(0 if result['passed'] else 1)
