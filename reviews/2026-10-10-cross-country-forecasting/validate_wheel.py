"""Run and compare the installed wheel outside the repository, with network blocked."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-dir", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args()
    review = Path(__file__).resolve().parent
    installed = args.installed_dir.resolve()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        required = ["puremacro/examples/cross_country_forecasting.py",
                    "puremacro/datasets/data/cross_country_gdp.csv",
                    "puremacro/datasets/data/cross_country_gdp_metadata.json"]
        assert all(name in archive.namelist() for name in required)
    code = '''
import json, pathlib, socket, sys
from unittest.mock import patch
import puremacro
from puremacro import pocket
from puremacro.examples.cross_country_forecasting import run_application
installed, output = map(pathlib.Path, sys.argv[1:])
assert pathlib.Path(puremacro.__file__).is_relative_to(installed)
def blocked(*args, **kwargs):
    raise AssertionError("network disabled for wheel validation")
with patch.object(socket, "create_connection", side_effect=blocked):
    try:
        socket.create_connection(("example.invalid", 80))
    except AssertionError:
        pass
    else:
        raise AssertionError("network blocker did not activate")
    result = run_application(output)
cart = pocket.load(output / "cross_country_forecasting.pmz")
assert cart.verify()
assert cart["levels"].attrs == result["manifest"]["observed_data"]
print(json.dumps({"package_file":puremacro.__file__, "version":puremacro.__version__,
                  "forecasts":len(result["forecasts"]), "cartridge_verified":True}))
'''
    with tempfile.TemporaryDirectory(prefix="puremacro-wheel-replay-") as directory:
        work = Path(directory)
        output = work / "study"
        env = dict(os.environ, PYTHONPATH=str(installed), MPLCONFIGDIR=str(work / "mpl"),
                   XDG_CACHE_HOME=str(work / "cache"))
        run = subprocess.run([sys.executable, "-c", code, str(installed), str(output)],
                             cwd=work, env=env, capture_output=True, text=True, check=True)
        result = json.loads(run.stdout)
        tables = {}
        for name in ("levels", "growth", "forecasts", "accuracy", "pooled_accuracy", "coverage", "exclusions"):
            actual = (output / f"{name}.csv").read_bytes()
            expected = (review / "study" / f"{name}.csv").read_bytes()
            if actual != expected:
                raise AssertionError(f"Installed-wheel table differs from source: {name}")
            tables[name] = hashlib.sha256(actual).hexdigest()
        manifest = json.loads((output / "manifest.json").read_text())
        source = json.loads((review / "study/manifest.json").read_text())
        for key in ("source_sha256", "observed_data", "design"):
            assert manifest[key] == source[key], key
        result.update(passed=True, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      wheel_bytes=wheel.stat().st_size, table_sha256=tables,
                      validation_scope="Installed wheel in separate working directory, existing base dependencies; HTTP network blocked")
        (review / "wheel-validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
