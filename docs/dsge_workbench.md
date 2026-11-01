> 🇬🇧 English · 🇪🇸 [Español](es/dsge_workbench.md)

# A workbench for DSGE experiments

`DSGEWorkbench` keeps a model, its draft calibration and its saved runs together.
It runs the existing first-order IRF and deterministic perfect-foresight engines.
Notebook displays show the run state, exact inputs and separate accuracy checks;
they do not select a different mathematical problem automatically.

## Solve, save and compare

```python
import numpy as np
from puremacro.dsge import DSGEWorkbench, DSGERun

source = """
var x; varexo e; parameters rho; rho=.8;
model; [name='Persistence'] x=rho*x(-1)+e; end;
initval; x=0; end;
"""
bench = DSGEWorkbench.from_mod(source, name="Persistence experiment")
baseline = bench.irf("e", horizon=20, size=.01, label="Baseline")
alternative = bench.irf("e", horizon=20, size=.01,
                         params={"rho": .5}, label="Lower persistence")
assert baseline.status == alternative.status == "succeeded"
np.testing.assert_allclose(baseline.table().x, .01 * .8 ** np.arange(21))
comparison = baseline.compare(alternative)
print(comparison.to_frame())
print(comparison.settings().query("changed"))
```

Display `bench`, `baseline`, `baseline.accuracy` or `comparison` in a notebook
for HTML tables with expandable evidence. `comparison.plot()` overlays the two
saved paths. `baseline.plot()` and `baseline.accuracy.plot()` produce Matplotlib
figures. These are views of captured data; editing parameters later does not
change them. `to_dict()`, `table()` and the evidence accessors return copies.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    directory = Path(directory)
    path = baseline.save(directory / "baseline.json")
    saved = DSGERun.load(path)  # verifies checksums; executes no model
    replayed = saved.replay()  # explicitly solves the frozen recipe again
    np.testing.assert_allclose(replayed.table(), baseline.table())
    (directory / "baseline.py").write_text(saved.to_python(), encoding="utf-8")
    (directory / "baseline.html").write_text(saved.to_html(), encoding="utf-8")
    (directory / "comparison.html").write_text(comparison.to_html(), encoding="utf-8")
```

Archives retain the expanded `.mod` source, hashes of its original source files,
calibration, shock specification, solver options, units, timing, run identifier,
environment fingerprint, numeric tables and accuracy evidence. Included source
files may subsequently change without changing the saved recipe. Existing files
require `overwrite=True`; loading rejects damaged checksums and archives larger
than 32 MiB. Checksums detect accidental corruption, not author identity.

Replay checks the recorded Python/package versions and a source fingerprint of
PureMacro. A changed environment requires `allow_environment_change=True` and
produces a new run. The exported Python is an editable recipe with recorded
environment comments; executing it does **not** enforce this environment guard.
Loading an archive never executes its recipe. Callback models and explicitly
captured external results are snapshots without portable replay or Python export.

Comparisons require successful runs with the same ordered variables, declared
units, timing and dates. Different dates require `alignment="inner"`, which
reports the common window and both original lengths. Different shocks or boundary
conditions require `allow_experiment_change=True`. Calibration and method changes
remain visible in `settings()`. A difference between two runs is not, by itself,
a convergence or welfare test.

## Apply edits explicitly

```python
widget = bench.interactive_irf(parameters={"rho": (.1, .95)}, shocks="e",
                               horizon=20, size=.01)
widget.update({"rho": .65})
assert widget.run_state == "edited"
assert widget.to_frame().isna().all().all()
widget.apply()
assert widget.is_current
snapshot = widget.capture_run(label="Selected slider experiment")
np.testing.assert_allclose(snapshot.table().x, .01 * .65 ** np.arange(21))
widget.disconnect()
```

The workbench opens the existing Matplotlib interface in Apply mode. Slider
changes mark it edited and clear the active curve until Apply succeeds. Set
`auto_solve=True` to solve each edit; direct `model.interactive_irf(...)` retains
that default for compatibility. A batched `update({...})` solves once. Reset
restores the initial slider settings and follows the selected Apply mode.

The static baseline remains labeled as a reference. Failed solves clear active
curves and exports; `last_success_model` retains the previous success explicitly.
Errors distinguish steady states, determinacy, input/domain and numerical failures
and suggest the next diagnostic. A failed initial override is visible too.
`capture_run()` freezes selected slider outcomes as snapshots; it does not append
every mouse movement to workbench history or invent a portable model recipe.

Live dragging and buttons require an interactive Matplotlib backend, such as a
supported notebook widget backend or a desktop GUI. Inline/Agg output is static;
the Python methods still work. The HTML reports are inspectable saved reports,
not a background execution service.

## Inspect each accuracy dimension

`run.status == "succeeded"` means that the selected engine completed with its
convergence criterion. `run.accuracy` separately reports equations, determinacy,
feasibility, household grids, horizon, quadrature, compression, branching and
derivatives. There is deliberately no overall accuracy pass flag.

| Evidence state | Meaning |
|---|---|
| `passed` / `failed` | A named check evaluated against its displayed criterion |
| `reported` | A value or diagnostic was retained without a new acceptance test |
| `not_checked` | No evidence for this dimension was supplied or evaluated |

The dashboard reuses `audit_result` where supported. First-order equation checks
do not certify global stochastic accuracy. Higher-order captures audit only their
embedded first-order model. Terminal distance is reported separately from a
horizon comparison. Global policy results retain their pointwise policy audit;
they do not gain a grid-refinement certificate from iteration convergence.

For HANK results, the adapter labels existing solver diagnostics as recorded
evidence. Euler, compression, mass and budget metrics need explicit tolerances
to become numerical passes. Missing diagnostics remain unchecked. Capturing an
external result requires the caller to supply its experiment metadata; the
workbench cannot recover arbitrary callback code or omitted calibration details.

```python
from puremacro.dsge import solution_accuracy

checks = solution_accuracy(None)
assert set(checks.to_frame().status) == {"not_checked"}
```

For an existing HANK result, use
`solution_accuracy(result, tolerances={"euler": 1e-3, "compression": 1e-4})`
with thresholds justified by your experiment. Capture it with
`DSGERun.capture(result, parameters=..., configuration=..., units=..., timing=...)`,
or add it to history with `bench.record(...)`. The thresholds above illustrate
the API and are not recommended universal accuracy requirements.

## Progress, cancellation and recovery

```python
from puremacro.dsge import DSGERunControl

events = []
control = DSGERunControl(max_seconds=30, on_progress=events.append)
path = bench.perfect_foresight(periods=5, y_init=[0.], y_end=[0.],
                               shocks=[[.01], [0.], [0.], [0.], [0.]],
                               linear_solver="krylov", control=control)
assert path.status == "succeeded"
assert len(events) > 0
```

`control.cancel()` requests cooperative cancellation. Krylov solves check at
accepted Newton steps and existing inner progress hooks; other operations check
at stage boundaries. Calls remain synchronous, and `max_seconds` is a cooperative
solver budget, not a hard deadline. The final evidence capture may take additional
time. A UI running the call on its main thread cannot process a cancel button
until that thread yields; use a separate execution context or a progress callback.

Cancellation and budget exhaustion retain distinct states (`cancelled` and
`budget_exhausted`). If a Krylov step was accepted, the archive retains its path
checkpoint and matching input fingerprint. `stopped_run.replay(resume=True)` uses
that path as an initial guess and reruns all solver checks. It does not restore
Krylov internals or certify an interrupted path as solved. A new control can set
a different budget for the resumed run. Without a compatible accepted checkpoint,
resume is rejected.

The runnable example `python -m puremacro.examples.dsge_workbench` compares two
analytically known AR(1) responses. Add `--out DIRECTORY` to export its archives,
Python recipe, HTML reports and plots. See [DSGE workflows](dsge_workflows.md)
for the mathematical scope of the other solvers.
