"""Freeze reviewed argument contracts in addition to exported symbol names."""
import importlib
import inspect
import json
from pathlib import Path

MODULES = (
    "puremacro.core.equilibrium", "puremacro.core.continuation",
    "puremacro.vfi.equilibrium_linearization", "puremacro.structural.simulation",
    "puremacro.structural.robust", "puremacro.expectations.gaussian",
    "puremacro.models.information_nk", "puremacro.models.hank_reduction", "puremacro.trade.reallocation",
    "puremacro.sovereign.banking", "puremacro.sovereign.adaptation",
)


def collect_research_api():
    contracts = {}

    def record(name, function):
        contracts[name] = [dict(name=p.name, kind=p.kind.name,
            default=None if p.default is inspect.Parameter.empty else repr(p.default))
            for p in inspect.signature(function).parameters.values()]

    for name in MODULES:
        module = importlib.import_module(name)
        parent = importlib.import_module(name.rsplit(".", 1)[0])
        for symbol in module.__all__:
            value = getattr(module, symbol)
            assert getattr(parent, symbol) is value
            record(name+"."+symbol, value)
            if inspect.isclass(value):
                for method_name, method in inspect.getmembers(value):
                    if not method_name.startswith("_") and (inspect.isfunction(method) or inspect.ismethod(method)):
                        record(name+"."+symbol+"."+method_name, method)
    return contracts


def test_research_public_arguments_match_reviewed_contracts():
    expected = json.loads((Path(__file__).parent/"fixtures/research_api_snapshot.json").read_text())
    assert collect_research_api() == expected
