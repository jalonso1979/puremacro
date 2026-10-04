#!/usr/bin/env node
// gallery_runner.js — run puremacro's validation gallery inside headless Pyodide.
//
// Argv: --wheel <path-to-puremacro-*.whl>
//       [--out <json path>]      also write the envelope, indented, to this file
//       [--pyodide <dir>]        an alternative Pyodide npm package directory
//                                (default: the pinned one in ./node_modules)
//       [--preload a,b,...]      Pyodide packages to load before the wheel, for
//                                diagnosis only (e.g. the unvendored stdlib module
//                                sqlite3); the playground preloads nothing
//
// Boots Pyodide, preloads only micropip, copies the wheel into the Emscripten
// file system and installs it the way the JupyterLite playground does: a
// dependency-resolving micropip install with PyPI fallback disabled, which fails
// if any dependency came from PyPI (same check as runner.js). It then loads
// gallery_cases.py into the interpreter and runs the cases one at a time, each
// timed with time.perf_counter() inside Python.
//
// Stdout: exactly one JSON envelope:
//   {schema_version, runtime: "pyodide", pyodide_version, node_version,
//    python_version, loaded_at, wheel, wheel_path, wheel_sha256, wheel_bytes,
//    puremacro_version, packages, preloaded, preload_error, boot_s, install_s,
//    import_s, wheel_installed, install_error, n_cases, n_passed, n_failed,
//    gallery_s, runtime_s, cases}
//   where cases[i] = {id, subsystem, mechanism, tol, passed, max_margin, error,
//   seconds} (max_margin is null when the case raised).
// Stderr: human-readable progress lines.
// Exit code: 0 if the envelope was emitted (even when the install or cases
// failed); non-zero only if Pyodide failed to boot or this script crashed.

const path = require("path");
const fs = require("fs");
const crypto = require("crypto");

function usage(code) {
    console.error("usage: node gallery_runner.js --wheel <path-to-wheel.whl> [--out <json>] [--pyodide <dir>] [--preload a,b]");
    process.exit(code);
}

function parseArgs() {
    const argv = process.argv.slice(2);
    let wheel = null;
    let out = null;
    let pyodideDir = null;
    let preload = [];
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === "--wheel" && i + 1 < argv.length) {
            wheel = argv[++i];
        } else if (argv[i] === "--out" && i + 1 < argv.length) {
            out = argv[++i];
        } else if (argv[i] === "--pyodide" && i + 1 < argv.length) {
            pyodideDir = argv[++i];
        } else if (argv[i] === "--preload" && i + 1 < argv.length) {
            preload = argv[++i].split(",").map((s) => s.trim()).filter(Boolean);
        } else {
            usage(2);
        }
    }
    if (!wheel) usage(2);
    if (!fs.existsSync(wheel)) {
        console.error(`error: wheel not found at ${wheel}`);
        process.exit(2);
    }
    if (pyodideDir && !fs.existsSync(pyodideDir)) {
        console.error(`error: Pyodide directory not found at ${pyodideDir}`);
        process.exit(2);
    }
    return {
        wheel: path.resolve(wheel),
        out: out ? path.resolve(out) : null,
        pyodideDir: pyodideDir ? path.resolve(pyodideDir) : null,
        preload,
    };
}

async function main() {
    const t_start = Date.now();
    const { wheel, out, pyodideDir, preload } = parseArgs();
    const { loadPyodide } = require(pyodideDir || "pyodide");

    const casesSource = fs.readFileSync(path.join(__dirname, "gallery_cases.py"), "utf-8");
    const wheel_basename = path.basename(wheel);
    const wheel_bytes = fs.readFileSync(wheel);
    const wheel_sha256 = crypto.createHash("sha256").update(wheel_bytes).digest("hex");

    console.error(`loading Pyodide${pyodideDir ? " from " + pyodideDir : ""} ...`);
    const t_boot = Date.now();
    const pyodide = await loadPyodide({
        stdout: (msg) => console.error("[pyodide]", msg),
        stderr: (msg) => console.error("[pyodide-err]", msg),
    });
    const pyodide_version = pyodide.version;
    const loaded_at = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
    console.error(`Pyodide ${pyodide_version} loaded`);

    // Only the installer is preloaded; puremacro's dependencies must arrive
    // through its own install below, as in the playground.
    console.error("loading micropip ...");
    await pyodide.loadPackage(["micropip"], {
        messageCallback: (msg) => console.error("[pkg]", msg),
    });
    let preload_error = "";
    if (preload.length) {
        console.error(`preloading ${preload.join(", ")} (diagnostic; the playground does not) ...`);
        try {
            await pyodide.loadPackage(preload, {
                messageCallback: (msg) => console.error("[pkg]", msg),
            });
        } catch (e) {
            // e.g. a module that this Pyodide ships inside its stdlib and has no
            // package for; the run goes on and the envelope records the reason.
            preload_error = e.message;
            console.error("preload failed:", e.message);
        }
    }
    const boot_s = (Date.now() - t_boot) / 1000;

    console.error(`installing ${wheel_basename} via micropip (resolving dependencies) ...`);
    pyodide.FS.writeFile(`/tmp/${wheel_basename}`, wheel_bytes);
    let wheel_installed = false;
    let install_error = "";
    const t_install = Date.now();
    try {
        await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/tmp/${wheel_basename}")
_from_pypi = sorted(
    name for name, pkg in micropip.list().items()
    if str(getattr(pkg, "source", "")).lower() == "pypi"
)
if _from_pypi:
    raise RuntimeError(
        "dependencies resolved from PyPI, which the playground cannot reach: "
        + ", ".join(_from_pypi)
    )
        `);
        wheel_installed = true;
    } catch (e) {
        install_error = e.message;
        console.error("wheel install failed:", e.message);
    }
    const install_s = (Date.now() - t_install) / 1000;

    let python_version = pyodide.runPython("import sys; sys.version.split()[0]");
    let puremacro_version = null;
    let packages = {};
    let import_s = null;
    let n_cases = 0;
    const cases = [];

    if (wheel_installed) {
        // Which distribution each core dependency came from (lock file vs the wheel).
        const sources = pyodide.runPython(`
import json, micropip
json.dumps({name: {"version": pkg.version, "source": str(pkg.source)}
            for name, pkg in micropip.list().items()
            if name in ("numpy", "scipy", "pandas", "matplotlib", "requests", "puremacro")})
        `);
        packages = JSON.parse(sources);

        console.error("importing puremacro and discovering cases ...");
        const t_import = Date.now();
        pyodide.runPython(casesSource);
        const discover = pyodide.globals.get("discover");
        const run_index = pyodide.globals.get("run_index");
        const versions = pyodide.globals.get("versions");
        n_cases = discover();
        import_s = (Date.now() - t_import) / 1000;
        const info = JSON.parse(versions());
        puremacro_version = info.puremacro_version;
        python_version = info.python_version;
        for (const [name, version] of Object.entries(info.packages)) {
            packages[name] = Object.assign({}, packages[name] || {}, { version });
        }
        console.error(`puremacro ${puremacro_version}: ${n_cases} cases`);

        for (let i = 0; i < n_cases; i++) {
            const rec = JSON.parse(run_index(i));
            cases.push(rec);
            let line = `[${i + 1}/${n_cases}] ${rec.passed ? "PASS" : "FAIL"} ${rec.id} (${rec.seconds.toFixed(3)} s)`;
            if (rec.error) line += ` -- ${rec.error}`;
            console.error(line);
        }
        discover.destroy();
        run_index.destroy();
        versions.destroy();
    }

    const n_passed = cases.filter((c) => c.passed).length;
    const round3 = (x) => (x === null ? null : Math.round(x * 1000) / 1000);
    const envelope = {
        schema_version: 1,
        runtime: "pyodide",
        pyodide_version,
        node_version: process.version,
        python_version,
        loaded_at,
        wheel: wheel_basename,
        wheel_path: wheel,
        wheel_sha256,
        wheel_bytes: wheel_bytes.length,
        puremacro_version,
        packages,
        preloaded: preload,
        preload_error,
        boot_s: round3(boot_s),
        install_s: round3(install_s),
        import_s: round3(import_s),
        wheel_installed,
        install_error,
        n_cases,
        n_passed,
        n_failed: n_cases - n_passed,
        gallery_s: round3(cases.reduce((s, c) => s + c.seconds, 0)),
        runtime_s: round3((Date.now() - t_start) / 1000),
        cases,
    };
    if (out) {
        fs.writeFileSync(out, JSON.stringify(envelope, null, 2) + "\n");
        console.error(`envelope written to ${out}`);
    }
    console.log(JSON.stringify(envelope));
}

main().catch((e) => {
    console.error("gallery_runner.js fatal:", e.stack || e.message);
    process.exit(1);
});
