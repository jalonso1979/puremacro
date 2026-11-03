"""Saved DSGE experiments, explicit run state and scoped accuracy evidence.

The workbench orchestrates existing solvers. Loading an archive never executes
code. Replay is explicit and available only for captured declarative recipes;
arbitrary Python callbacks are not serialized or guessed from their repr.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy

from puremacro.validation._audit_report import _jsonable
from ._workbench_accuracy import DSGEAccuracyDashboard, solution_accuracy
from ._workbench_state import DSGEFailure, DSGERunControl, describe_failure, _Cancelled, _BudgetExpired
from ._workbench_html import badge, document, panel
from .build import LinearModel
from .model import NonlinearModel, read_mod

__all__ = ["DSGEWorkbench", "DSGERun", "DSGEComparison", "DSGEAccuracyDashboard",
           "DSGEFailure", "DSGERunControl", "solution_accuracy"]
_SCHEMA = "puremacro.dsge.run/1"
_STATES = {"succeeded", "failed", "cancelled", "budget_exhausted", "edited", "recorded"}


def _dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _digest(value: Any) -> str:
    return sha256(_dumps(value).encode("utf-8")).hexdigest()


def _input(value: Any) -> Any:
    """Only portable finite inputs; no callable/pickle/object serialization."""
    if isinstance(value, np.ndarray):
        return _input(value.tolist())
    if isinstance(value, np.generic):
        return _input(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("Saved input mappings require string keys")
        return {k: _input(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_input(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Saved numerical inputs must be finite")
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    raise TypeError(f"Input type {type(value).__name__} is not portable; capture the result separately")


@lru_cache(maxsize=1)
def _environment_json() -> str:
    import puremacro
    root = Path(puremacro.__file__).parent
    hashes = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*.py"))}
    return _dumps(dict(python=platform.python_version(), numpy=np.__version__,
                       scipy=scipy.__version__, pandas=pd.__version__,
                       puremacro=puremacro.__version__, package_sha256=_digest(hashes)))


def _frame(frame: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError("Run tables must be DataFrames with unique row/column labels")
    if not all(isinstance(c, str) for c in frame.columns):
        raise TypeError("Run table columns must be strings")
    values = frame.to_numpy(dtype=float)
    index = _input(frame.index.tolist())
    return dict(columns=list(frame.columns), index=index, index_name=frame.index.name,
                values=_jsonable(values))


def _unframe(data: Mapping[str, Any]) -> pd.DataFrame:
    # float converts the explicit NaN/Infinity strings retained in failed paths.
    values = np.asarray(data["values"], dtype=float)
    if not len(data["index"]):
        values = values.reshape(0, len(data["columns"]))
    return pd.DataFrame(values, index=pd.Index(data["index"], name=data["index_name"]), columns=data["columns"])


def _result_tables(result: Any) -> dict[str, pd.DataFrame]:
    for name in ("path", "paths", "transition_paths"):
        table = getattr(result, name, None)
        if isinstance(table, pd.DataFrame):
            return {"path": table}
    tree = getattr(result, "tree", None)
    tree_paths = getattr(tree, "paths", None)
    if isinstance(tree_paths, pd.DataFrame):
        return {"path": tree_paths}
    return {}


def _result_status(result: Any) -> str:
    if isinstance(result, LinearModel):
        return "succeeded" if result.is_determinate else "failed"
    value = getattr(result, "converged", None)
    if type(result).__module__.startswith(("puremacro.dsge.", "puremacro.models.hank")) and isinstance(value, (bool, np.bool_)):
        return "succeeded" if value else "failed"
    return "recorded"


@dataclass(frozen=True)
class DSGERun:
    """Immutable result snapshot with exact inputs and an optional replay recipe.

    Numeric tables and evidence are snapshots, not a serialized live solver.
    All accessors return independent copies. A run may have succeeded while
    accuracy checks fail or remain unchecked; inspect ``accuracy`` separately.
    """

    _json: str

    @classmethod
    def capture(cls, result: Any, *, label: str = "DSGE run", status: str | None = None,
                parameters: Mapping[str, float] | None = None,
                configuration: Mapping[str, Any] | None = None,
                tables: Mapping[str, pd.DataFrame] | None = None,
                units: Mapping[str, str] | None = None, timing: str = "unspecified",
                failure: DSGEFailure | None = None,
                accuracy: DSGEAccuracyDashboard | None = None,
                _recipe: Mapping[str, Any] | None = None,
                _execution: Mapping[str, Any] | None = None) -> DSGERun:
        """Capture an existing result without re-solving it.

        External captures have no executable recipe. Declare units and timing
        for later comparisons. The accuracy adapter identifies unsupported
        checks explicitly; arbitrary result attributes cannot certify accuracy.
        """
        state = status or _result_status(result)
        if state not in _STATES:
            raise ValueError(f"Unknown run state: {state}")
        pars = dict(parameters if parameters is not None else getattr(result, "_params", {}) or {})
        config = _input(dict(configuration or {}))
        snapshots = {str(k): _frame(v) for k, v in (tables if tables is not None else _result_tables(result)).items()}
        unit_map = dict(units if units is not None else getattr(result, "units", {}) or {})
        recipe = _input(_recipe) if _recipe is not None else None
        inputs = dict(parameters=_input(pars), configuration=config, recipe=recipe,
                      units=_input(unit_map), timing=str(timing))
        evidence = accuracy or solution_accuracy(result, failure=failure)
        payload = dict(schema=_SCHEMA, run_id=uuid4().hex, label=str(label), status=state,
                       created_utc=datetime.now(timezone.utc).isoformat(),
                       input_sha256=_digest(inputs), inputs=inputs, tables=snapshots,
                       accuracy=evidence.to_dict(), failure=None if failure is None else failure.to_dict(),
                       execution=_jsonable(dict(_execution or {})),
                       environment=json.loads(_environment_json()),
                       replayable=recipe is not None,
                       replay_scope="declarative recipe" if recipe is not None else "snapshot only; no portable solver recipe supplied")
        return cls(_dumps(payload))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def run_id(self) -> str:
        return self.to_dict()["run_id"]

    @property
    def label(self) -> str:
        return self.to_dict()["label"]

    @property
    def status(self) -> str:
        return self.to_dict()["status"]

    @property
    def parameters(self) -> dict[str, float]:
        return self.to_dict()["inputs"]["parameters"]

    @property
    def accuracy(self) -> DSGEAccuracyDashboard:
        return DSGEAccuracyDashboard(_dumps(self.to_dict()["accuracy"]))

    @property
    def failure(self) -> DSGEFailure | None:
        data = self.to_dict()["failure"]
        return None if data is None else DSGEFailure(**data)

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(self.to_dict()["tables"])

    def table(self, name: str | None = None) -> pd.DataFrame:
        tables = self.to_dict()["tables"]
        if name is None:
            if len(tables) != 1:
                raise ValueError(f"Select a table explicitly from {list(tables)}")
            name = next(iter(tables))
        if name not in tables:
            raise KeyError(f"No table {name!r}; available: {list(tables)}")
        table = _unframe(tables[name])
        table.attrs.update(run_id=self.run_id, run_status=self.status, parameters=self.parameters)
        return table

    def summary(self) -> str:
        d = self.to_dict()
        return (f"{d['label']} [{d['run_id'][:8]}]: {d['status']}; "
                f"{d['replay_scope']}. {self.accuracy.summary()}")

    def _repr_html_(self) -> str:
        return panel(self._html_body())

    def _html_body(self) -> str:
        d = self.to_dict()
        failure = d["failure"]
        note = "" if failure is None else '<p>' + escape(failure["message"] + " " + failure["next_step"]) + '</p>'
        inputs = escape(json.dumps(d["inputs"], indent=2, ensure_ascii=False))
        return ('<section class="pm-run"><h2>' + escape(d["label"]) + '</h2><p>'
                + badge(d['status']) + ' <span class="pm-meta">'
                + escape(f"Run {d['run_id'][:8]} · {d['created_utc']}")
                + '</span></p><p class="pm-meta">' + escape(d["replay_scope"]) + '</p>' + note
                + self.accuracy._html_body() + '<details><summary>Inputs and reproducibility</summary><pre>'
                + inputs + '</pre></details></section>')

    def to_html(self) -> str:
        """Standalone report, including a bounded preview of every saved table."""
        previews = []
        for name in self.table_names:
            table = self.table(name)
            previews.append('<h3>' + escape(name) + '</h3><p class="pm-meta">'
                            + escape(f"Showing {min(len(table), 20)} of {len(table)} rows; full data are in the run archive.")
                            + '</p><div class="pm-scroll">' + table.head(20).to_html(escape=True, border=0) + '</div>')
        return document(self.label, panel(self._html_body() + ''.join(previews)))

    def to_python(self) -> str:
        """Export the explicit recipe as Python; snapshots never invent code."""
        from pprint import pformat
        data = self.to_dict()
        recipe = data["inputs"]["recipe"]
        if recipe is None:
            raise ValueError("This snapshot has no portable Python recipe")
        options = dict(data["inputs"]["configuration"])
        operation = options.pop("operation")
        if operation not in {"irf", "perfect_foresight"}:
            raise ValueError(f"Unsupported operation: {operation}")
        return (f"# Input SHA256: {data['input_sha256']}\n"
                f"# Recorded environment: {_dumps(data['environment'])}\n"
                "from puremacro.dsge import DSGEWorkbench\n\n"
                f"source = {recipe['source']!r}\n"
                f"workbench = DSGEWorkbench.from_mod(source, name={data['label']!r})\n"
                f"parameters = {pformat(data['inputs']['parameters'])}\n"
                f"options = {pformat(options)}\n"
                f"run = workbench.{operation}(params=parameters, **options)\n"
                "print(run.summary())\n")

    def plot(self, table: str | None = None, *, variables=None, **kwargs):
        frame = self.table(table)
        if variables is not None:
            frame = frame.loc[:, list(variables)]
        axes = frame.plot(subplots=True, title=f"{self.label} [{self.run_id[:8]}] · {self.status}", **kwargs)
        units = self.to_dict()["inputs"]["units"]
        for ax, name in zip(np.asarray(axes).ravel(), frame.columns):
            ax.set_ylabel(units.get(name, "units unspecified"))
        return axes

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write a strict JSON archive; existing files are protected by default."""
        target = Path(path)
        payload = self.to_dict()
        envelope = dict(schema=_SCHEMA, sha256=_digest(payload), payload=payload)
        with target.open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> DSGERun:
        """Verify archive integrity and load data only; never run saved code."""
        target = Path(path)
        if target.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("DSGE run archives are limited to 32 MiB")
        data = json.loads(target.read_text(encoding="utf-8"),
                          parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Nonstandard JSON: {token}")))
        if not isinstance(data, dict) or data.get("schema") != _SCHEMA or not isinstance(data.get("payload"), dict):
            raise ValueError("Unsupported DSGE run archive schema")
        payload = data["payload"]
        if data.get("sha256") != _digest(payload):
            raise ValueError("DSGE run archive checksum mismatch")
        if payload.get("schema") != _SCHEMA or payload.get("status") not in _STATES:
            raise ValueError("Invalid DSGE run payload")
        required = {"run_id", "label", "created_utc", "inputs", "tables", "accuracy", "failure",
                    "execution", "environment", "replayable", "replay_scope"}
        if not required.issubset(payload):
            raise ValueError("Incomplete DSGE run payload")
        if payload.get("input_sha256") != _digest(payload["inputs"]):
            raise ValueError("DSGE run input checksum mismatch")
        try:
            if not {"parameters", "configuration", "recipe", "units", "timing"}.issubset(payload["inputs"]):
                raise ValueError("Incomplete run inputs")
            for table in payload["tables"].values():
                _frame(_unframe(table))
            if not isinstance(payload["accuracy"]["checks"], list):
                raise ValueError("Invalid accuracy evidence")
            if payload["failure"] is not None:
                DSGEFailure(**payload["failure"])
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed DSGE run payload") from exc
        return cls(_dumps(payload))

    def replay(self, *, allow_environment_change: bool = False,
               resume: bool = False, control: DSGERunControl | None = None) -> DSGERun:
        """Explicitly rerun a saved .mod recipe, optionally from its checkpoint.

        Environment differences require explicit opt-in. Resuming reuses only
        an accepted path as an initial guess; all solver checks run again.
        """
        data = self.to_dict()
        recipe = data["inputs"]["recipe"]
        if not recipe:
            raise ValueError("Snapshot has no portable replay recipe; call the original solver explicitly")
        if data["environment"] != json.loads(_environment_json()) and not allow_environment_change:
            raise ValueError("Saved numerical environment differs; inspect it or use allow_environment_change=True")
        if sha256(recipe["source"].encode("utf-8")).hexdigest() != recipe["source_sha256"]:
            raise ValueError("Saved model source checksum mismatch")
        bench = DSGEWorkbench.from_mod(recipe["source"], name=data["label"] + " replay")
        options = dict(data["inputs"]["configuration"])
        operation = options.pop("operation")
        if resume:
            checkpoint = data["execution"].get("checkpoint")
            if operation != "perfect_foresight" or checkpoint is None:
                raise ValueError("This run has no compatible perfect-foresight checkpoint")
            expected = _digest(dict(recipe=recipe, parameters=data["inputs"]["parameters"],
                                    configuration=data["inputs"]["configuration"]))
            if checkpoint["input_sha256"] != expected or data["execution"]["solver_input_sha256"] != expected:
                raise ValueError("Checkpoint does not match this experiment")
            options["initial_path"] = checkpoint["path"]
        if operation == "irf":
            shock = options.pop("shock")
            return bench.irf(shock, params=data["inputs"]["parameters"], control=control, **options)
        if operation == "perfect_foresight":
            return bench.perfect_foresight(params=data["inputs"]["parameters"], control=control, **options)
        raise ValueError(f"Unsupported saved operation: {operation}")

    def compare(self, other: DSGERun, *, table: str | None = None,
                alignment: str = "exact", allow_experiment_change: bool = False) -> DSGEComparison:
        return DSGEComparison.create(self, other, table=table, alignment=alignment,
                                     allow_experiment_change=allow_experiment_change)


@dataclass(frozen=True)
class DSGEComparison:
    """Comparison of two immutable runs, with exact settings differences."""

    baseline: DSGERun
    candidate: DSGERun
    table_name: str
    alignment: str
    allow_experiment_change: bool = False

    @classmethod
    def create(cls, baseline, candidate, *, table=None, alignment="exact", allow_experiment_change=False):
        if baseline.status != "succeeded" or candidate.status != "succeeded":
            raise ValueError("Comparisons require two successful runs; failed/partial paths remain inspectable separately")
        if alignment not in {"exact", "inner"}:
            raise ValueError("alignment must be 'exact' or explicitly 'inner'")
        if table is None:
            if len(baseline.table_names) != 1 or baseline.table_names != candidate.table_names:
                raise ValueError("Select a common table explicitly")
            table = baseline.table_names[0]
        a, b = baseline.table(table), candidate.table(table)
        x, y = baseline.to_dict()["inputs"], candidate.to_dict()["inputs"]
        if x["timing"] == "unspecified" or x["timing"] != y["timing"]:
            raise ValueError("Comparison requires identical, explicitly declared timing")
        if list(a.columns) != list(b.columns):
            raise ValueError("Comparison requires the same ordered variables")
        for name in a.columns:
            if name not in x["units"] or x["units"][name] != y["units"].get(name):
                raise ValueError(f"Comparison requires identical explicit units for {name}")
        if alignment == "exact" and not a.index.equals(b.index):
            raise ValueError("Run horizons/indices differ; use alignment='inner' to declare a common window")
        if not len(a.index.intersection(b.index)):
            raise ValueError("The runs have no common dates")
        if not np.isfinite(a.to_numpy()).all() or not np.isfinite(b.to_numpy()).all():
            raise ValueError("Successful-run comparison requires finite tables")
        if not allow_experiment_change:
            for name in ("operation", "shock", "size", "shocks", "exogenous_path", "y_init", "y_end", "y_ss"):
                if x["configuration"].get(name) != y["configuration"].get(name):
                    raise ValueError(f"Experiment differs at {name}; inspect settings and explicitly allow_experiment_change=True")
        return cls(baseline, candidate, str(table), alignment, allow_experiment_change)

    def paths(self) -> pd.DataFrame:
        a, b = self.baseline.table(self.table_name), self.candidate.table(self.table_name)
        if self.alignment == "inner":
            common = a.index.intersection(b.index, sort=False)
            a, b = a.loc[common], b.loc[common]
        return pd.concat({"baseline": a, "candidate": b, "difference": b - a}, axis=1)

    def to_frame(self) -> pd.DataFrame:
        paths = self.paths()
        diff = paths["difference"]
        result = pd.DataFrame({"max_abs_difference": diff.abs().max(), "rms_difference": np.sqrt((diff ** 2).mean()),
                               "units": {name: self.baseline.to_dict()["inputs"]["units"][name] for name in diff.columns}})
        result.attrs.update(alignment=self.alignment, compared_dates=len(diff),
                            baseline_dates=len(self.baseline.table(self.table_name)),
                            candidate_dates=len(self.candidate.table(self.table_name)))
        return result

    def settings(self) -> pd.DataFrame:
        def flatten(value, prefix=""):
            out = {}
            for key, item in value.items():
                name = prefix + key
                if isinstance(item, dict):
                    out.update(flatten(item, name + "."))
                else:
                    out[name] = item
            return out
        a, b = (flatten(run.to_dict()["inputs"]) for run in (self.baseline, self.candidate))
        # Source text is available on each run; the difference table uses its digest.
        keys = sorted((set(a) | set(b)) - {"recipe.source"})
        return pd.DataFrame([dict(setting=k, baseline=a.get(k), candidate=b.get(k),
                                  changed=a.get(k) != b.get(k)) for k in keys]).set_index("setting")

    def plot(self, *, variables=None):
        import matplotlib.pyplot as plt
        paths = self.paths()
        names = list(variables or paths["baseline"].columns)
        fig, axes = plt.subplots(len(names), 1, squeeze=False, figsize=(8, max(3, 2.3 * len(names))))
        for ax, name in zip(axes.ravel(), names):
            ax.plot(paths.index, paths["baseline"][name], label=f"{self.baseline.label} [{self.baseline.run_id[:8]}]")
            ax.plot(paths.index, paths["candidate"][name], "--", label=f"{self.candidate.label} [{self.candidate.run_id[:8]}]")
            ax.set_ylabel(name + " (" + self.baseline.to_dict()["inputs"]["units"][name] + ")")
            ax.legend()
        fig.tight_layout()
        return fig

    def _repr_html_(self) -> str:
        settings = self.settings()
        changed = settings.loc[settings.changed]
        return panel('<section><h3>' + escape(self.baseline.label + " → " + self.candidate.label)
                + '</h3><p>' + escape(f"Alignment: {self.alignment}; compared dates: {len(self.paths())}")
                + '</p>' + self.to_frame().to_html(escape=True, border=0)
                + '<h4>Changed inputs</h4>' + changed.to_html(escape=True, border=0) + '</section>')

    def to_html(self) -> str:
        return document("DSGE run comparison", self._repr_html_())


class DSGEWorkbench:
    """A frozen model source, editable calibration and an append-only run history.

    ``from_mod`` supports portable replay. A supplied ``NonlinearModel`` or
    ``LinearModel`` can also be used, but callback-based runs are snapshots
    only. Numerical methods remain explicit: ``irf`` is local first order;
    ``perfect_foresight`` is a finite deterministic path.
    """

    def __init__(self, model: NonlinearModel | LinearModel, *, name: str = "DSGE experiment"):
        if not isinstance(model, (NonlinearModel, LinearModel)):
            raise TypeError("DSGEWorkbench requires a NonlinearModel or LinearModel; use from_mod for text/files")
        from copy import deepcopy
        self._model = deepcopy(model)
        self._source = getattr(model, "_mod_source_text", None)
        self._parameters = dict(getattr(model, "_params", {}) or {})
        self._source_hashes = dict(getattr(model, "source_hashes", {}) or {})
        self.name = str(name)
        self._runs: list[DSGERun] = []
        self._running = False

    @classmethod
    def from_mod(cls, source: str | Path, *, name: str = "DSGE experiment") -> DSGEWorkbench:
        """Read and freeze expanded source without running model commands."""
        return cls(read_mod(source), name=name)

    @property
    def parameters(self) -> dict[str, float]:
        return self._parameters.copy()

    def set_parameters(self, updates: Mapping[str, float]) -> None:
        if self._running:
            raise RuntimeError("Cannot edit this workbench during a solve")
        unknown = set(updates) - set(self._parameters)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        validated = {name: float(value) for name, value in updates.items()}
        if not all(np.isfinite(value) for value in validated.values()):
            raise ValueError("Parameters must be finite")
        self._parameters.update(validated)

    @property
    def runs(self) -> tuple[DSGERun, ...]:
        return tuple(self._runs)

    @property
    def last_attempt(self) -> DSGERun | None:
        return self._runs[-1] if self._runs else None

    @property
    def last_success(self) -> DSGERun | None:
        return next((run for run in reversed(self._runs) if run.status == "succeeded"), None)

    @property
    def state(self) -> str:
        if self._running:
            return "running"
        last = self.last_attempt
        if last is None or last.parameters != self.parameters:
            return "edited"
        return last.status

    def _fresh_model(self):
        if self._source is not None:
            return read_mod(self._source, params=self._parameters)
        from copy import deepcopy
        candidate = deepcopy(self._model)
        if isinstance(candidate, NonlinearModel):
            candidate.parameter_values = self.parameters
            return candidate
        from .widgets import _fast_resolve
        return _fast_resolve(candidate, self.parameters, strict=True)

    def irf(self, shock: str, *, horizon: int = 20, size: float = 1.,
            params: Mapping[str, float] | None = None, label: str | None = None,
            solver_options: Mapping[str, Any] | None = None,
            control: DSGERunControl | None = None) -> DSGERun:
        """Run a first-order local IRF; failures return a retained failed run."""
        options = dict(solver_options or {})
        if set(options) & {"params", "order", "shock_cov", "variables", "shocks", "states"}:
            raise ValueError("Use the model declaration for structural options; workbench IRFs are explicitly first order")
        config = dict(operation="irf", shock=shock, horizon=horizon, size=size, solver_options=options)
        return self._execute(config, params=params, label=label, control=control)

    def perfect_foresight(self, *, params: Mapping[str, float] | None = None,
                         label: str | None = None, control: DSGERunControl | None = None,
                         **options: Any) -> DSGERun:
        """Run a deterministic path with the existing perfect-foresight engine.

        Krylov runs retain progress and accepted-path checkpoints. Cancellation
        is cooperative. Callable options are intentionally not serializable.
        """
        if options.get("surprise"):
            raise ValueError("Use the explicit announced/surprise solver and capture its result; this recipe is perfect foresight")
        return self._execute(dict(operation="perfect_foresight", **options), params=params, label=label, control=control)

    def _execute(self, config, *, params, label, control):
        if self._running:
            raise RuntimeError("This workbench already has a running solve")
        if control is not None and not isinstance(control, DSGERunControl):
            raise TypeError("control must be a DSGERunControl")
        ctl = control or DSGERunControl()
        if ctl._started is not None:
            raise ValueError("A DSGERunControl belongs to one run; create a fresh control")
        config = _input(config)
        if params is not None:
            self.set_parameters(params)
        parameters = self.parameters
        recipe = None if self._source is None else dict(
            source=self._source, source_sha256=sha256(self._source.encode("utf-8")).hexdigest(),
            original_source_hashes=self._source_hashes)
        solver_hash = _digest(dict(recipe=recipe, parameters=parameters, configuration=config))
        start = perf_counter()
        result: Any = None
        tables: dict[str, pd.DataFrame] = {}
        units: dict[str, str] = {}
        timing = "unspecified"
        failure = None
        status = "failed"
        self._running = True
        try:
            ctl._start()
            ctl._progress({"phase": "preparing", "operation": config["operation"]})
            model = self._fresh_model()
            ctl._check()
            options = {k: v for k, v in config.items() if k != "operation"}
            if config["operation"] == "irf":
                if isinstance(model, NonlinearModel):
                    result = model.solve(order=1, **options["solver_options"])
                else:
                    if options["solver_options"]:
                        raise ValueError("Solver options require a NonlinearModel or .mod source")
                    result = model
                if not result.is_determinate:
                    failure = DSGEFailure("determinacy", "BlanchardKahnError", "No unique stable first-order policy",
                                          "Inspect variable timing, roots and policy-rule parameters.")
                else:
                    tables = {"irf": result.irf(options["shock"], horizon=options["horizon"], size=options["size"])}
                units, timing = dict(result.units), str(result.timing)
            else:
                from .perfect_foresight import solve_perfect_foresight
                if options.get("linear_solver", "direct") == "krylov":
                    krylov = dict(options.get("krylov_options") or {})
                    if set(krylov) & {"progress", "checkpoint"}:
                        raise ValueError("Use DSGERunControl for progress and checkpoints")
                    krylov.update(progress=ctl._progress, checkpoint=ctl._save_checkpoint)
                    options["krylov_options"] = krylov
                result = solve_perfect_foresight(model, **options)
                tables = _result_tables(result)
                units = {name: "level" for name in model.variables}
                timing = "perfect_foresight: entering state at t=0; reported dates t=1..T"
            ctl._progress({"phase": "solved"})
            status = _result_status(result)
            if status == "failed" and failure is None:
                diagnostics = getattr(result, "solver_diagnostics", {})
                failure = DSGEFailure("numerical_solver", "SolverNotConverged",
                                      str(diagnostics.get("converged_by", "The solver did not converge")),
                                      "Inspect residuals, constraints and the retained solver history.")
            if status == "succeeded" and any(not np.isfinite(table.to_numpy()).all() for table in tables.values()):
                raise FloatingPointError("A reported successful result contains nonfinite output")
            ctl._progress({"phase": "auditing"})
        except Exception as exc:
            failure = describe_failure(exc)
            status = "cancelled" if isinstance(exc, _Cancelled) else "budget_exhausted" if isinstance(exc, _BudgetExpired) else "failed"
        try:
            evidence = solution_accuracy(result, failure=failure)
            execution: dict[str, Any] = dict(elapsed_seconds=perf_counter() - start, events=ctl.events,
                                            solver_input_sha256=solver_hash, max_seconds=ctl.max_seconds,
                                            cancellation_scope="cooperative Krylov hooks or stage boundaries")
            if ctl._checkpoint is not None:
                names = tuple(self._model.variables)
                execution["checkpoint"] = dict(input_sha256=solver_hash,
                    path=ctl._checkpoint.reshape(-1, len(names)).tolist(), record=ctl._checkpoint_record)
            run = DSGERun.capture(result, label=label or self.name, status=status, parameters=parameters,
                                 configuration=config, tables=tables, units=units, timing=timing,
                                 failure=failure, accuracy=evidence, _recipe=recipe, _execution=execution)
            self._runs.append(run)
            return run
        finally:
            self._running = False

    def record(self, result: Any, **kwargs: Any) -> DSGERun:
        """Add an explicitly described external solver result to this history."""
        if self._running:
            raise RuntimeError("Cannot append a result during this workbench's solve")
        run = DSGERun.capture(result, **kwargs)
        self._runs.append(run)
        return run

    def history(self) -> pd.DataFrame:
        return pd.DataFrame([dict(run_id=r.run_id, label=r.label, status=r.status,
                                  current_parameters=r.parameters == self.parameters,
                                  replayable=r.to_dict()["replayable"]) for r in self._runs])

    def interactive_irf(self, **kwargs):
        """Open the existing slider UI, with Apply mode by default.

        Capture chosen slider outcomes explicitly using ``capture_run``.
        Slider motion does not silently append hundreds of history records.
        """
        model = self._fresh_model()
        if isinstance(model, NonlinearModel):
            model = model.solve(order=1)
        kwargs.setdefault("auto_solve", False)
        return model.interactive_irf(**kwargs)

    def _repr_html_(self) -> str:
        last = self.last_attempt
        return panel('<section><h2>' + escape(self.name) + '</h2><p>Current workbench state: '
                + badge(self.state) + '</p><p class="pm-meta">Model → Experiment → Solve → Audit → Compare</p>'
                + '<details><summary>Current draft parameters</summary><pre>'
                + escape(json.dumps(self.parameters, indent=2)) + '</pre></details>'
                + self.history().to_html(index=False, escape=True, border=0)
                + (last._html_body() if last is not None else '<p>No run has been solved.</p>') + '</section>')
