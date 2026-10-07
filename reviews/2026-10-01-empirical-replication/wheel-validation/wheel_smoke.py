"""Installed-wheel verification, run with the checkout absent from sys.path."""
from pathlib import Path
import json
import socket
import sys

BASE = Path('/tmp/puremacro-empirical-wheel-final')
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
assert not report.passed
assert [r.id for r in report.results if not r.passed] == ['rr2010_published_peak']
assert all(not r.error for r in report.results)
peak = next(r for r in report.results if r.id == 'rr2010_published_peak')
assert peak.provenance['atol'] == .005 and peak.provenance['rtol'] == 0
assert [name for name, value in peak.metrics.items() if not value['passed']] == ['t_h10']
assert len(report.results) == 8
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

# Authentic observed-data applications, executed without network or checkout.
from puremacro.structural import fit_empirical_sw07, load_empirical_sw07_data
from puremacro.replication import estimate_rr2010_baseline, load_rr2010_data, load_rr2010_reference
from puremacro.examples.empirical_sw07_matching import run_application as run_sw07
from puremacro.examples.romer_romer_2010_replication import run_application as run_rr2010
observed_sw, sw_source = load_empirical_sw07_data()
assert observed_sw.shape == (156, 7) and not sw_source['is_synthetic']
sw_study = run_sw07(BASE / 'installed-applications' / 'sw07')
assert sw_study.metadata['status'] == 'nonregular_fit'
assert not sw_study.fit.inference_valid and np.isnan(sw_study.fit.standard_errors).all()
assert np.allclose(sw_study.fit.theta, [.98, .01], atol=2e-5)
comparison = sw_study.calibration_sensitivity.set_index('calibration')
assert comparison.loc['exploratory_published_sw07_mode', 'conditional_null_j_pvalue'] < .01
sw_manifest = json.loads((BASE / 'installed-applications' / 'sw07' / 'manifest.json').read_text(),
                         parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)))
assert sw_manifest['conditional_null_j_pvalue'] is None
rr_data = load_rr2010_data()
rr_reference = load_rr2010_reference()
rr_fit = estimate_rr2010_baseline()
assert rr_fit.nobs == 232 and rr_fit.df_resid == 218
assert abs(rr_fit.irf[10] + 3.08) <= .005
# Literal published rounding mismatch is evidence, never widened to pass.
assert abs(rr_fit.t_statistics[10] + 3.53) > .005
assert abs(rr_fit.t_statistics[10] - (-3.524996075)) < 1e-8
np.testing.assert_allclose(rr_fit.irf_covariance, rr_reference['irf_covariance'], rtol=1e-9, atol=1e-11)
rr_application, rr_report = run_rr2010(BASE / 'installed-applications' / 'rr2010')
assert not rr_report.passed and len(rr_report.results) == 2
assert [case.id for case in rr_report.results if not case.passed] == ['rr2010_published_peak']
rr_targets = rr_fit.to_moment_targets()
np.testing.assert_allclose(rr_targets.covariance, rr_fit.irf_covariance, rtol=1e-14, atol=1e-15)
assert 'statsmodels' not in sys.modules
assert all(case.negative_control_rejected for case in report.results)

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
          'sw07_status': sw_study.metadata['status'], 'sw07_conditional_inference_withheld': not sw_study.fit.inference_valid,
          'rr2010_h10': float(rr_fit.irf[10]), 'rr2010_t10': float(rr_fit.t_statistics[10]),
          'rr2010_replication_passed': rr_report.passed, 'statsmodels_runtime_imported': 'statsmodels' in sys.modules,
          'network_blocked_with_positive_control': True,
          'repository_source_absent_from_sys_path': str(REPOSITORY) not in sys.path,
          'expected_benchmark_failure': 'rr2010_published_peak: published t statistic outside unchanged rounding tolerance',
          'note': 'Installed-wheel verification passed while preserving expected scientific failures; built from current working tree including pre-existing uncommitted work; no package published.'}
(BASE / 'installed_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
