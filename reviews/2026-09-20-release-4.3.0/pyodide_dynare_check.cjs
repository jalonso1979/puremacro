// Run from any directory: node pyodide_dynare_check.cjs /absolute/path/to/wheel.whl
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..', '..');
const { loadPyodide } = require(path.join(root, 'tools/pyodide/node_modules/pyodide'));
(async () => {
  const wheel = path.resolve(process.argv[2]);
  const name = path.basename(wheel);
  let output = [];
  const py = await loadPyodide({stdout: s => output.push(s), stderr: s => process.stderr.write(s+'\n')});
  await py.loadPackage('micropip', {messageCallback: s => process.stderr.write(s+'\n')});
  py.FS.writeFile('/tmp/'+name, fs.readFileSync(wheel));
  await py.runPythonAsync(`import micropip\nawait micropip.install('emfs:/tmp/${name}')`);
  py.mountNodeFS('/mnt/repo', root);
  output = [];
  await py.runPythonAsync(`
import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location('reference_check', '/mnt/repo/tools/reference_validation/validate_dynare.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
manifest = json.loads((mod.FIXTURES / 'manifest.json').read_text())
cases = {name: mod.compare_case(name) for name in manifest['cases']}
print(json.dumps({'passed': all(c['passed'] for c in cases.values()), 'cases': cases}))
`);
  const result = JSON.parse(output.join('\n'));
  result.pyodide_version = py.version;
  result.runtime = 'WebAssembly via Node';
  process.stdout.write(JSON.stringify(result, null, 2)+'\n');
  if (!result.passed) process.exitCode = 1;
})().catch(e => {process.stderr.write(String(e.stack || e)+'\n'); process.exitCode=1;});
