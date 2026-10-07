"""Installed-wheel verification, run with the checkout absent from sys.path."""
from pathlib import Path
import json
import socket
import sys

BASE = Path('/tmp/puremacro-research-wheel-final')
INSTALLED = BASE / 'installed'
REPOSITORY = Path('/Users/jalonso/Documents/RESEARCH/puremacro')
assert str(REPOSITORY) not in sys.path

class OfflineSocket(socket.socket):
    def connect(self, *args, **kwargs):
        raise RuntimeError('Network forbidden in installed-wheel smoke')
    def connect_ex(self, *args, **kwargs):
        raise RuntimeError('Network forbidden in installed-wheel smoke')

socket.socket = OfflineSocket
# Positive control: the offline boundary must actually block a connect call.
try:
    with socket.socket() as control:
        control.connect(('127.0.0.1', 9))
except RuntimeError as exc:
    assert 'Network forbidden' in str(exc)
else:
    raise AssertionError('Offline mechanism did not engage')

import numpy as np
import pandas as pd
import puremacro
from puremacro.datasets import load_enigh2024_deciles
from puremacro.structural import MomentTargets, StructuralFitResult, fit_structural, lp_moment_targets
from puremacro.trade import prepare_household_groups, compute_distributional_welfare
from puremacro.validation import (
    ResearchBenchmark, BenchmarkResult, ResearchBenchmarkReport,
    research_benchmarks, run_research_benchmarks,
)

assert Path(puremacro.__file__).is_relative_to(INSTALLED)
data = load_enigh2024_deciles()
assert len(data) == 10 and not data.attrs['is_synthetic']
assert data['households'].sum() == 38_830_230

report = run_research_benchmarks()
assert report.passed, [(r.id, r.error) for r in report.results if not r.passed]
assert len(report.results) == 6
report.write(BASE / 'installed-evidence')

rng = np.random.default_rng(916)
shock = rng.normal(size=500)
noise = rng.normal(size=500)
output = np.zeros(500)
for t in range(1, 500):
    output[t] = .6 * output[t - 1] + .8 * shock[t] + .2 * noise[t]
frame = pd.DataFrame({'output': output, 'innovation': shock},
                     index=pd.period_range('1900Q1', periods=500, freq='Q'))
targets = lp_moment_targets(frame, responses=('output',), shock='innovation',
                            horizons=(0, 1, 2, 3), response_units={'output': 'log points'},
                            shock_unit='unit innovation', frequency='Q', lags=2, bandwidth=4)
fit = fit_structural(lambda theta: theta[0] * theta[1] ** np.arange(4),
                     targets, [.7, .5], parameter_names=('impact', 'persistence'),
                     moment_labels=targets.labels, bounds=[(0., None), (0., .99)])
assert fit.success and fit.inference_valid
assert np.all(np.isfinite(fit.covariance))
assert np.allclose(fit.theta, [.8, .6], atol=.12)

categories = list(data.attrs['categories'])
groups = prepare_household_groups(data[categories], data['households'],
                                  monetary_unit='MXN', period='quarter', provenance=data.attrs)
base_prices = pd.Series(1., index=categories)
prices = base_prices.copy()
prices['food'] = 1.1
incidence = compute_distributional_welfare(groups, base_prices, prices, rule='cobb_douglas')
assert len(incidence.groups) == 10
assert np.all(incidence.groups['ev'] < 0.)
assert incidence.aggregate['total_population'] == 38_830_230
assert np.isclose(incidence.aggregate['total_ev'], data.households @ incidence.groups.ev)

# Every loaded puremacro module must originate in the isolated wheel directory.
for name, module in list(sys.modules.items()):
    if name == 'puremacro' or name.startswith('puremacro.'):
        file = getattr(module, '__file__', None)
        if file:
            assert Path(file).is_relative_to(INSTALLED), (name, file)

result = {'passed': True, 'package_file': puremacro.__file__,
          'benchmark_count': len(report.results), 'benchmark_passes': sum(r.passed for r in report.results),
          'enigh_deciles': len(data), 'expanded_households': int(data.households.sum()),
          'lp_parameters': fit.theta.tolist(), 'lp_inference_valid': fit.inference_valid,
          'distributional_total_ev': float(incidence.aggregate['total_ev']),
          'network_blocked_with_positive_control': True,
          'repository_source_absent_from_sys_path': str(REPOSITORY) not in sys.path,
          'note': 'Built from current working tree, including pre-existing uncommitted work; no package published.'}
(BASE / 'installed_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
