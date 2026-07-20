"""Schema test for tests/known_failures.json — locks the contract used by tools/release_check.py."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATH = REPO_ROOT / "tests" / "known_failures.json"


def test_known_failures_file_exists():
    assert PATH.exists(), f"{PATH} must exist (tools/release_check.py reads it)"


def test_known_failures_top_level_shape():
    data = json.loads(PATH.read_text())
    assert isinstance(data, dict)
    assert data.get("schema_version") == 1
    assert isinstance(data.get("entries"), list)


def test_known_failures_entry_fields():
    data = json.loads(PATH.read_text())
    required = {"nodeid", "reason", "since_version", "owner_note"}
    for i, entry in enumerate(data["entries"]):
        missing = required - set(entry.keys())
        assert not missing, f"entry {i} missing fields: {missing}"
        assert isinstance(entry["nodeid"], str) and entry["nodeid"]
        assert isinstance(entry["reason"], str) and entry["reason"]
        assert isinstance(entry["since_version"], str) and entry["since_version"]
        assert isinstance(entry["owner_note"], str) and entry["owner_note"]


def test_known_failures_nodeids_unique():
    data = json.loads(PATH.read_text())
    nodeids = [e["nodeid"] for e in data["entries"]]
    assert len(nodeids) == len(set(nodeids)), "duplicate nodeid in known_failures.json"
