// node pyodide_policy_check.cjs /absolute/path/to/puremacro-4.3.0-py3-none-any.whl
const path = require('path');
const root = path.resolve(__dirname, '..', '..');
const { loadPyodide } = require(path.join(root, 'tools/pyodide/node_modules/pyodide'));
const fs = require('fs');
(async () => {
  let out = [];
  const py = await loadPyodide({stdout: s => out.push(s), stderr: s => process.stderr.write(s+'\n')});
  await py.loadPackage('micropip', {messageCallback: s => process.stderr.write(s+'\n')});
  const name = 'puremacro-4.3.0-py3-none-any.whl';
  py.FS.writeFile('/tmp/'+name, fs.readFileSync(path.resolve(process.argv[2])));
  await py.runPythonAsync(`import micropip\nawait micropip.install('emfs:/tmp/${name}')`);
  out = [];
  await py.runPythonAsync(fs.readFileSync(path.join(__dirname,'installed_wheel_smoke.py'),'utf8'));
  const result = JSON.parse(out.join('\n'));
  result.pyodide_version = py.version;
  result.runtime = 'WebAssembly via Node';
  process.stdout.write(JSON.stringify(result,null,2)+'\n');
})().catch(e => {process.stderr.write(String(e.stack || e)+'\n'); process.exitCode=1;});
