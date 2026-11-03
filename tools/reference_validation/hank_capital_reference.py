"""Freeze the authors' productive-capital blocks, without importing PureMacro.

The pinned, MIT-licensed source is retained beside the output. Only four
functions are selected from its AST. Decorators are removed and SSJ's
``x.apply(f)`` is evaluated as ``f(x)``; the economic expressions are unchanged.
This validates an aggregate block, not the authors' household/full HANK model.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

COMMIT = "eb4ac4c3157357a4cacbea4b5a7a281f5b48cab9"
SOURCE_URL = f"https://github.com/shade-econ/sequence-jacobian/blob/{COMMIT}/src/sequence_jacobian/examples/two_asset.py"


class Dated(float):
    """A current scalar with explicitly supplied adjacent-date values."""

    def __new__(cls, current, *, before=None, after=None):
        value = super().__new__(cls, current)
        value.before = float(current if before is None else before)
        value.after = float(current if after is None else after)
        return value

    def __call__(self, offset):
        if offset == -1:
            return self.before
        if offset == 1:
            return self.after
        if offset == 0:
            return float(self)
        raise ValueError("only adjacent-date values are supplied")


class Apply(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "apply":
            if len(node.args) != 1 or node.keywords:
                raise ValueError("unexpected SSJ apply syntax")
            return ast.copy_location(ast.Call(func=node.args[0], args=[node.func.value], keywords=[]), node)
        return node


def author_functions(source):
    """Load only the reviewed pure numerical functions; do not import SSJ."""
    parsed = ast.parse(source)
    names = {"labor", "investment", "dividend", "partial_ss"}
    functions = []
    calibration = None
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            functions.append(Apply().visit(node))
        if isinstance(node, ast.FunctionDef) and node.name == "dag":
            for statement in node.body:
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == "calibration"):
                    calibration = ast.literal_eval(statement.value)
    if {f.name for f in functions} != names or calibration is None:
        raise ValueError("pinned author block definitions or calibration are missing")
    namespace = {"np": np}
    selected = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(selected, SOURCE_URL, "exec"), namespace)
    return namespace, calibration


def generate(folder: Path, output: Path):
    if output.exists():
        raise ValueError("output already exists; preserve earlier reference evidence")
    source = folder / "ssj_two_asset.py.txt"
    license_path = folder / "SSJ_LICENSE.txt"
    functions, published = author_functions(source.read_text())
    base = {k: published[k] for k in ("Y", "N", "K", "r", "tot_wealth", "Bg", "delta")}
    partial = dict(zip(("p", "mc", "mu", "alpha", "Z", "w"), functions["partial_ss"](**base)))
    parameters = dict(alpha=partial["alpha"], delta=base["delta"], epsI=published["epsI"],
                      mu=partial["mu"], kappa=published["kappap"])
    rng = np.random.default_rng(20261101)
    cases = []
    for index in range(25):
        scale = 0. if index == 0 else 1.
        values = dict(Y=base["Y"]*np.exp(scale*rng.normal(0, .03)),
                      w=partial["w"]*np.exp(scale*rng.normal(0, .03)),
                      K=base["K"]*np.exp(scale*rng.normal(0, .10)),
                      q=np.exp(scale*rng.normal(0, .025)),
                      lag_q=np.exp(scale*rng.normal(0, .02)),
                      Z=partial["Z"]*np.exp(scale*rng.normal(0, .01)),
                      pi=scale*rng.normal(0, .003), **parameters)
        K, q = values["K"], values["q"]
        def inv_residual(end_capital):
            inv, _ = functions["investment"](Dated(q), Dated(end_capital, before=K),
                Dated(base["r"]), Dated(1.), Dated(partial["mc"]), Dated(values["Z"]),
                parameters["delta"], parameters["epsI"], parameters["alpha"])
            return inv
        Knext = brentq(inv_residual, .5*K, 1.5*K, xtol=1e-13)
        N, mc = functions["labor"](values["Y"], values["w"],
                                    Dated(Knext, before=K), values["Z"], parameters["alpha"])
        adj, investment_goods, dividend = functions["dividend"](values["Y"], values["w"], N,
            Dated(Knext, before=K), values["pi"], parameters["mu"], parameters["kappa"],
            parameters["delta"], parameters["epsI"])
        def value_residual(real_return):
            # Evaluate the authors' t-1 valuation equation using current
            # production/investment, hence Q(t-1)=lag_q and K(t-1)=K.
            _, val = functions["investment"](Dated(values["lag_q"], after=q),
                Dated(K, after=Knext), Dated(base["r"], after=real_return),
                Dated(N), Dated(mc), Dated(values["Z"]), parameters["delta"],
                parameters["epsI"], parameters["alpha"])
            return val
        ra = brentq(value_residual, -.5, .5, xtol=1e-14)
        cases.append(dict(name="published_capital_steady" if index == 0 else f"off_steady_{index:02d}",
            inputs=values, reference=dict(N=float(N), mc=float(mc), Knext=float(Knext),
                r_a=float(ra), adj=float(adj), investment_goods=float(investment_goods),
                dividend=float(dividend), investment_foc=float(inv_residual(Knext)),
                valuation_foc=float(value_residual(ra)))))
    record = dict(schema_version=1, reference_commit=COMMIT, source_url=SOURCE_URL,
        scope="author capital/labor/dividend functions only; no household, portfolio, wage-setting or full equilibrium replication",
        evaluation="selected original AST; decorator removal, apply(f) to f(x), explicit adjacent-date scalars",
        seed=20261101, published_calibration=published, published_partial_ss=partial,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, license_path)},
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases)
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(f"Wrote {len(cases)} independent author capital-block cases to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source_folder, args.output)
