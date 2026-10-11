#!/usr/bin/env node
// Narrow installed-wheel check; the full gallery is outside this review.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const repo = path.resolve(__dirname, "../..");
const { loadPyodide } = require(path.join(repo, "tools/pyodide/node_modules/pyodide"));

async function main() {
    const wheel = process.argv[2];
    if (!wheel || !fs.existsSync(wheel)) {
        throw new Error("usage: node pyodide_check.cjs /absolute/path/puremacro-VERSION-py3-none-any.whl");
    }
    const started = Date.now();
    const result = {
        schema_version: 1,
        checked_at: new Date().toISOString(),
        wheel_path: path.resolve(wheel),
        wheel_sha256: crypto.createHash("sha256").update(fs.readFileSync(wheel)).digest("hex"),
        scope: "Installed wheel; provenance tests and complete cross-country forecasting application in Node Pyodide",
    };
    try {
        const pyodide = await loadPyodide({
            stdout: msg => console.error("[pyodide]", msg),
            stderr: msg => console.error("[pyodide-err]", msg),
        });
        result.pyodide_version = pyodide.version;
        await pyodide.loadPackage(["pytest", "micropip"], {
            messageCallback: msg => console.error("[package]", msg),
        });
        pyodide.mountNodeFS("/mnt/tests", path.join(repo, "tests"));
        pyodide.mountNodeFS("/reference", path.join(__dirname, "study"));
        const wheelName = path.basename(wheel);
        pyodide.FS.writeFile(`/tmp/${wheelName}`, fs.readFileSync(wheel));
        pyodide.globals.set("review_wheel_name", wheelName);
        const evidence = await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/tmp/" + review_wheel_name)
pypi_dependencies = sorted(
    name for name, pkg in micropip.list().items()
    if str(getattr(pkg, "source", "")).lower() == "pypi"
)
assert not pypi_dependencies, pypi_dependencies

import hashlib, io, json, platform, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import puremacro
from puremacro import pocket
from puremacro.examples import cross_country_forecasting as example
import pytest
import pytz
import zoneinfo

# CPython normally obtains named-zone data from the operating system. Browser
# Python has no host zone database; use the one already bundled with pandas'
# pytz dependency for the ZoneInfo DST-fold regression case.
zoneinfo.reset_tzpath([str(Path(pytz.__file__).parent / "zoneinfo")])
buffer = io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = buffer
try:
    rc = pytest.main(["/mnt/tests/test_runtime_store_attrs.py", "-q", "--tb=short"])
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr
test_output = buffer.getvalue()

output = Path("/tmp/cross-country-study")
study = example.run_application(output)
reference = pd.read_csv("/reference/accuracy.csv", float_precision="round_trip")
actual = study["accuracy"]
pd.testing.assert_frame_equal(actual, reference, check_exact=False, atol=1e-10, rtol=1e-10)
metrics = ["rmse", "mae", "mean_error"]
max_difference = float(np.max(np.abs(actual[metrics].to_numpy() - reference[metrics].to_numpy())))
cart = pocket.load(output / "cross_country_forecasting.pmz")
assert cart.verify()
assert cart["levels"].attrs == example.load_data().attrs
for filename, metadata in study["manifest"]["artifacts"].items():
    assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == metadata["sha256"]
assert (output / "forecast_accuracy.png").stat().st_size > 1000
json.dumps({
    "python_version": platform.python_version(), "platform": sys.platform,
    "package_version": puremacro.__version__, "package_file": puremacro.__file__,
    "numpy_version": np.__version__, "pandas_version": pd.__version__,
    "pypi_dependencies": pypi_dependencies,
    "timezone_database": "pytz distribution's bundled zoneinfo",
    "pytest_returncode": int(rc),
    "pytest_passed": int(re.search(r"(\\d+) passed", test_output).group(1)) if re.search(r"(\\d+) passed", test_output) else 0,
    "pytest_output": test_output,
    "full_application_completed": True, "cartridge_verified": True,
    "provenance_preserved": True, "artifact_hashes_verified": True,
    "plot_bytes": (output / "forecast_accuracy.png").stat().st_size,
    "forecast_rows": len(study["forecasts"]), "accuracy_rows": len(actual),
    "accuracy_atol": 1e-10, "accuracy_rtol": 1e-10,
    "accuracy_max_absolute_difference": max_difference,
    "source_sha256": study["manifest"]["source_sha256"],
    "input_csv_sha256": example.load_data().attrs["csv_sha256"],
})
        `);
        Object.assign(result, JSON.parse(evidence));
        result.passed = result.pytest_returncode === 0;
    } catch (error) {
        result.passed = false;
        result.error = String(error.stack || error.message || error);
    }
    result.runtime_seconds = (Date.now() - started) / 1000;
    fs.writeFileSync(path.join(__dirname, "pyodide-result.json"), JSON.stringify(result, null, 2) + "\n");
    console.log(JSON.stringify(result, null, 2));
    if (!result.passed) process.exitCode = 1;
}

main().catch(error => { console.error(error); process.exitCode = 1; });
