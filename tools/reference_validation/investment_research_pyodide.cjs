/* Source-tree execution of the investment tests inside real Pyodide/WASM.
 * Usage: node tools/reference_validation/investment_research_pyodide.cjs [pyodide-module-dir]
 * Default dependency installation: cd tools/pyodide && npm install.
 * This does not build/install the wheel or validate a browser/device UI.
 */
const path = require("path");
const repo = path.resolve(__dirname, "../..");
const moduleDir = process.argv[2] || path.join(repo, "tools/pyodide/node_modules/pyodide");
const { loadPyodide } = require(moduleDir);

async function main() {
    const pyodide = await loadPyodide({
        stdout: text => console.error(text),
        stderr: text => console.error(text),
    });
    await pyodide.loadPackage(["numpy", "scipy", "pandas", "matplotlib", "requests", "pytest", "tzdata"]);
    pyodide.mountNodeFS("/work", repo);
    const result = await pyodide.runPythonAsync(`
import io, json, os, sys
from contextlib import redirect_stdout, redirect_stderr
sys.path.insert(0, "/work")
sys.path.insert(0, "/work/tests")
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import puremacro, pytest
assert puremacro.__file__.startswith("/work/")
files = ["backtest", "allocation", "technical", "fundamentals", "beliefs", "research_integration"]
capture = io.StringIO()
with redirect_stdout(capture), redirect_stderr(capture):
    code = pytest.main(["-q", "--tb=short", "--rootdir=/work", "-c", "/work/pyproject.toml"] +
                       ["/work/tests/test_asset_pricing/test_" + name + ".py" for name in files])
json.dumps({"runtime": sys.version, "puremacro_version": puremacro.__version__,
            "source_tree": True, "pytest_returncode": int(code), "output": capture.getvalue()})
`);
    const report = { pyodide_version: pyodide.version, ...JSON.parse(result) };
    console.log(JSON.stringify(report, null, 2));
    process.exitCode = report.pytest_returncode;
}

main().catch(error => {
    console.error(error.stack || error);
    process.exitCode = 1;
});
