"""What tests/conftest.py does for parallel runs (``pytest -n``, as CI runs).

Two pieces: the slowest files start first inside an xdist worker, and the
socket guard's blocked attempts travel from the workers to the controller,
which prints the summary.
"""
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tests_conftest(request):
    """tests/conftest.py as pytest loaded it (other folders have their own)."""
    target = (_ROOT / "tests" / "conftest.py").resolve()
    for plugin in request.config.pluginmanager.get_plugins():
        path = getattr(plugin, "__file__", None)
        if path and Path(path).resolve() == target:
            return plugin
    pytest.fail("tests/conftest.py is not loaded as a plugin")


class _Item:
    def __init__(self, nodeid):
        self.nodeid = nodeid


def test_every_heavy_first_file_exists(tests_conftest):
    heavy = tests_conftest._HEAVY_FIRST
    assert len(heavy) == len(set(heavy)) > 0
    assert [p for p in heavy if not (_ROOT / p).is_file()] == []


def test_heavy_files_sort_first_in_list_order(tests_conftest):
    first, second = tests_conftest._HEAVY_FIRST[:2]
    nodeids = [
        "tests/test_a.py::test_one",
        f"{second}::test_x",
        "tests/test_b.py::test_two",
        f"{first}::TestCase::test_y",
        f"{second}::test_z",
    ]
    assert sorted(nodeids, key=tests_conftest._heavy_first_rank) == [
        f"{first}::TestCase::test_y",
        f"{second}::test_x",
        f"{second}::test_z",
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
    ]


def test_only_an_xdist_worker_reorders(tests_conftest, monkeypatch):
    heavy = tests_conftest._HEAVY_FIRST[0]
    items = [_Item("tests/test_a.py::t"), _Item(f"{heavy}::t")]

    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    tests_conftest.pytest_collection_modifyitems(None, items)
    assert [i.nodeid for i in items] == ["tests/test_a.py::t", f"{heavy}::t"]

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    tests_conftest.pytest_collection_modifyitems(None, items)
    assert [i.nodeid for i in items] == [f"{heavy}::t", "tests/test_a.py::t"]


class _Config:
    pass


class _Session:
    def __init__(self, config):
        self.config = config


class _Node:
    def __init__(self, workeroutput):
        self.workeroutput = workeroutput


def test_guard_hits_travel_from_worker_to_controller(tests_conftest, monkeypatch):
    hit = ("tests/test_x.py::test_y", "connect", "192.0.2.1")

    # Worker: its own hits go into workeroutput, as lists (what execnet sends).
    monkeypatch.setattr(tests_conftest, "_SOCKET_GUARD_HITS", [hit])
    worker = _Config()
    worker.workeroutput = {}
    tests_conftest.pytest_sessionfinish(_Session(worker))
    assert worker.workeroutput["socket_guard_hits"] == [list(hit)]

    # Controller: the finished worker's hits join the list the summary prints.
    monkeypatch.setattr(tests_conftest, "_SOCKET_GUARD_HITS", [])
    tests_conftest.pytest_testnodedown(_Node(worker.workeroutput), None)
    assert tests_conftest._SOCKET_GUARD_HITS == [hit]


def test_a_serial_run_and_a_crashed_worker_add_nothing(tests_conftest, monkeypatch):
    monkeypatch.setattr(tests_conftest, "_SOCKET_GUARD_HITS", [])
    tests_conftest.pytest_sessionfinish(_Session(_Config()))  # no workeroutput: serial
    tests_conftest.pytest_testnodedown(_Node(None), RuntimeError("worker died"))
    tests_conftest.pytest_testnodedown(_Node({}), None)
    assert tests_conftest._SOCKET_GUARD_HITS == []
