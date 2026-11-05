"""Run representative workflows from a wheel in a fresh, isolated environment.

Usage: python tools/check_installed_examples.py dist/puremacro-*.whl
The base dependencies are installed; dev extras and the source checkout are not.
This is a packaging/workflow check, not the full notebook or reference suite.
"""
from __future__ import annotations

import argparse
import email
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


SMOKE = r'''
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import sys
import numpy as np
import puremacro

# -I and the working directory isolate the source checkout; also prove origin.
package = Path(puremacro.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), package
assert puremacro.__version__ == sys.argv[1] == importlib.metadata.version('puremacro')

def offline(*args, **kwargs):
    raise AssertionError('installed examples must not use the network')
socket.create_connection = offline
socket.socket.connect = offline
socket.getaddrinfo = offline

from puremacro.examples.exact_diffuse_kalman import run_demo
kalman = run_demo()
assert np.isfinite(kalman['exact']['loglik'])
assert np.asarray(kalman['exact']['a_smooth']).shape[0] == len(kalman['y'])
print('PASS: exact-diffuse Kalman example', flush=True)

from puremacro.inference import gmm
sample = np.array([1., 3., 2., 4.])
fit = gmm(lambda theta, data: data[:, None] - theta, [0.], sample, hac_lags=0)
np.testing.assert_allclose(fit.coefs, [2.5], rtol=0, atol=1e-9)
assert fit.converged and np.all(np.isfinite(fit.se))
from puremacro.asset_pricing import value_at_risk
np.testing.assert_allclose(value_at_risk([.1, -.1, .1], horizon=2), .01, atol=1e-12)
print('PASS: GMM and compounded risk workflow', flush=True)

from puremacro.examples.first_brief_tariffs import run_brief
output = Path('tariff_brief')
manifest = run_brief(output, lang='es')['manifest']
assert manifest['lang'] == 'es'
for state in manifest['convergence'].values():
    assert state['converged']
for name, artifact in manifest['artifacts'].items():
    data = (output / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == artifact['sha256']
    assert len(data) == artifact['bytes']
record = json.loads((output / 'headline.json').read_text(encoding='utf-8'))
assert np.isfinite(record['national']['ev_pct_consumption'])
assert (output / 'first_brief.png').stat().st_size > 1000
print('PASS: bundled-data tariff brief, UTF-8 outputs and manifest', flush=True)
from puremacro.examples.first_brief_fiscal import run_brief as run_fiscal_brief
fiscal_output = Path('fiscal_brief')
fiscal = run_fiscal_brief(fiscal_output, lang='es')
fiscal_manifest = fiscal['manifest']
assert fiscal_manifest['lang'] == 'es'
for name, artifact in fiscal_manifest['artifacts'].items():
    data = (fiscal_output / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == artifact['sha256']
    assert len(data) == artifact['bytes']
assert np.isfinite(fiscal['record']['full']['horizons']['7']['linear']['multiplier'])
assert (fiscal_output / 'multipliers.png').stat().st_size > 1000
print('PASS: bundled-data fiscal brief, UTF-8 outputs and manifest', flush=True)
print('Installed wheel:', package, flush=True)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError('expected exactly one wheel METADATA file')
        metadata = email.message_from_bytes(archive.read(names[0]))
        if metadata['Name'] != 'puremacro':
            raise ValueError('expected a puremacro wheel')
        version = metadata['Version']
    with tempfile.TemporaryDirectory(prefix='puremacro-wheel-examples-') as tmp:
        root = Path(tmp)
        env_path = root / 'env'
        venv.EnvBuilder(with_pip=True).create(env_path)
        python = env_path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(root / 'mpl'),
                   XDG_CACHE_HOME=str(root / 'cache'), PUREMACRO_LLM_MODE='off')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        subprocess.run([str(python), '-I', '-m', 'pip', 'install',
                        '--disable-pip-version-check', str(wheel)],
                       check=True, cwd=root, env=env, timeout=600)
        subprocess.run([str(python), '-I', '-c', SMOKE, version],
                       check=True, cwd=root, env=env, timeout=300)
    return 0


if __name__ == '__main__':
    sys.exit(main())
