#!/usr/bin/env node
// runner.js — headless Pyodide harness for puremacro Gate 6.
//
// Argv: --wheel <absolute-path-to-puremacro-*.whl>
//
// Stdout: exactly one JSON document with the schema in README.md.
// Stderr: human-readable progress lines.
// Exit code: 0 if JSON envelope emitted; non-zero if Pyodide failed to boot.

const { loadPyodide } = require("pyodide");
const path = require("path");
const fs = require("fs");

function parseArgs() {
    const argv = process.argv.slice(2);
    let wheel = null;
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === "--wheel" && i + 1 < argv.length) {
            wheel = argv[i + 1];
            i++;
        }
    }
    if (!wheel) {
        console.error("usage: node runner.js --wheel <path-to-wheel.whl>");
        process.exit(2);
    }
    if (!fs.existsSync(wheel)) {
        console.error(`error: wheel not found at ${wheel}`);
        process.exit(2);
    }
    return { wheel: path.resolve(wheel) };
}

async function main() {
    const t_start = Date.now();
    const { wheel } = parseArgs();

    // tests directory to mount: <repo>/puremacro/tests
    // Resolve relative to this script's location.
    const here = __dirname;  // .../puremacro/tools/pyodide
    const repo_subproject = path.resolve(here, "..", "..");  // .../puremacro
    const tests_dir = path.join(repo_subproject, "tests");
    if (!fs.existsSync(tests_dir)) {
        console.error(`error: tests dir not found at ${tests_dir}`);
        process.exit(2);
    }

    console.error("loading Pyodide ...");
    const pyodide = await loadPyodide({
        stdout: (msg) => console.error("[pyodide]", msg),
        stderr: (msg) => console.error("[pyodide-err]", msg),
    });
    const pyodide_version = pyodide.version;
    const loaded_at = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
    console.error(`Pyodide ${pyodide_version} loaded`);

    console.error("loading numpy / scipy / pandas / matplotlib / pytest / micropip ...");
    await pyodide.loadPackage(
        ["numpy", "scipy", "pandas", "matplotlib", "pytest", "micropip"],
        { messageCallback: (msg) => console.error("[pkg]", msg) }
    );

    // Mount the host tests/ dir into Pyodide's FS at /mnt/tests (readonly).
    console.error(`mounting ${tests_dir} -> /mnt/tests`);
    pyodide.mountNodeFS("/mnt/tests", tests_dir);

    // Copy the wheel bytes into Pyodide's FS, then install via micropip.
    console.error("installing puremacro wheel via micropip ...");
    const wheel_basename = path.basename(wheel);
    const wheel_bytes = fs.readFileSync(wheel);
    pyodide.FS.writeFile(`/tmp/${wheel_basename}`, wheel_bytes);

    let wheel_installed = false;
    try {
        await pyodide.runPythonAsync(`
import micropip
# deps=False is mandatory here: since 0.94.0 puremacro declares six base
# dependencies, and one of them (pyarrow) has no Pyodide wheel, so a
# dependency-resolving install can never succeed under Pyodide and would leave
# this gate reporting wheel_installed=false. numpy / scipy / pandas /
# matplotlib are already provided by the loadPackage() call above.
await micropip.install("emfs:/tmp/${wheel_basename}", deps=False)
import puremacro
_ = puremacro.__version__  # touch the attribute to confirm import worked
        `);
        wheel_installed = true;
    } catch (e) {
        console.error("wheel install failed:", e.message);
    }

    // `requests` is the other base dependency skipped by deps=False. It is pure
    // Python and does install under Pyodide, and puremacro.fetch.* / the
    // narrative sources import it at module level. Best-effort: it needs
    // network access, and a failure here must not flip wheel_installed.
    if (wheel_installed) {
        try {
            await pyodide.runPythonAsync(`
import micropip
await micropip.install("requests")
            `);
        } catch (e) {
            console.error("optional 'requests' install skipped:", e.message);
        }
    }

    // Run the marked pytest subset.
    console.error("running pytest -m pyodide_smoke ...");
    let pytest_returncode = -1;
    let passed = 0;
    let failed = 0;
    let skipped = 0;
    let stdout_tail = "";

    if (wheel_installed) {
        const py_out = await pyodide.runPythonAsync(`
import io, sys, os, glob
import pytest
# Only collect files that contain the pyodide_smoke mark to avoid
# importing modules unavailable in Pyodide (e.g. requests, boto3, etc.)
import re as _re
_pat = _re.compile(r"@pytest\.mark\.pyodide_smoke")
smoke_files = [
    f for f in glob.glob("/mnt/tests/**/*.py", recursive=True)
    if _pat.search(open(f).read())
]
buf = io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = buf
try:
    rc = pytest.main(smoke_files + ["-m", "pyodide_smoke", "--tb=short", "-q"])
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr
out = buf.getvalue()
import re
m = re.search(r"(\\d+) passed", out)
p = int(m.group(1)) if m else 0
m = re.search(r"(\\d+) failed", out)
f = int(m.group(1)) if m else 0
m = re.search(r"(\\d+) skipped", out)
s = int(m.group(1)) if m else 0
tail_lines = out.strip().splitlines()[-3:]
tail = "\\n".join(tail_lines)
[int(rc), p, f, s, tail]
        `);
        const arr = py_out.toJs();
        pytest_returncode = arr[0];
        passed = arr[1];
        failed = arr[2];
        skipped = arr[3];
        stdout_tail = arr[4];
        py_out.destroy();
    }

    const runtime_s = (Date.now() - t_start) / 1000;
    const envelope = {
        schema_version: 1,
        pyodide_version,
        loaded_at,
        wheel_installed,
        wheel_path: wheel,
        pytest_returncode,
        passed,
        failed,
        skipped,
        runtime_s: Math.round(runtime_s * 10) / 10,
        stdout_tail,
    };
    console.log(JSON.stringify(envelope));
}

main().catch((e) => {
    console.error("runner.js fatal:", e.stack || e.message);
    process.exit(1);
});
