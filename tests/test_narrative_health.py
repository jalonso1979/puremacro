"""Sprint-1 / D1: connector health-check (offline, mocked)."""
from __future__ import annotations

from datetime import datetime
from unittest import mock

import pytest


def _stub_ok(records):
    def _iter(**_):
        yield from records
    return _iter


def test_probe_classifies_ok_when_records_returned():
    from puremacro.narrative.sources import health
    fn = _stub_ok([
        (datetime(2024, 1, 1), "text", "url", {}),
        (datetime(2024, 4, 1), "text", "url", {}),
    ])
    s = health._probe("iter_demo", fn)
    assert s.status == "ok"
    assert s.n_records == 2
    assert s.first_date == "2024-01-01"
    assert s.last_date == "2024-04-01"
    assert s.error is None


def test_probe_classifies_empty_when_no_records():
    from puremacro.narrative.sources import health
    fn = _stub_ok([])
    s = health._probe("iter_demo", fn)
    assert s.status == "empty"
    assert s.n_records == 0


def test_probe_classifies_error_on_exception():
    from puremacro.narrative.sources import health
    def raises(**_):
        raise RuntimeError("404 Not Found")
        yield  # pragma: no cover - unreachable
    s = health._probe("iter_demo", raises)
    assert s.status == "error"
    assert "404" in (s.error or "")


def test_probe_classifies_skipped_for_needs_args_connectors():
    from puremacro.narrative.sources import health
    fn = _stub_ok([(datetime(2024, 1, 1), "t", "u", {})])
    # iter_google_news is in _NEEDS_ARGS — must be reported as skipped.
    s = health._probe("iter_google_news", fn)
    assert s.status == "skipped"
    assert "args" in (s.error or "")


def test_format_table_renders_summary_line():
    from puremacro.narrative.sources import health
    results = [
        health.ConnectorStatus("iter_a", "ok",      5, "2024-01-01", "2024-04-01", None, 0.5),
        health.ConnectorStatus("iter_b", "empty",   0, None, None, None, 0.5),
        health.ConnectorStatus("iter_c", "error",   0, None, None, "BadURL", 0.1),
        health.ConnectorStatus("iter_d", "skipped", 0, None, None, "needs args", 0.0),
    ]
    out = health.format_table(results)
    assert "iter_a" in out
    assert "iter_c" in out
    assert "BadURL" in out
    assert "summary:" in out
    assert "ok=1" in out
    assert "error=1" in out


def test_main_returns_zero_when_at_least_one_ok_and_no_errors(capsys):
    from puremacro.narrative.sources import health
    fake = [
        health.ConnectorStatus("iter_a", "ok",    5, "2024-01-01", "2024-04-01", None, 0.5),
        health.ConnectorStatus("iter_b", "empty", 0, None, None, None, 0.5),
    ]
    with mock.patch.object(health, "run", return_value=fake):
        rc = health.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "iter_a" in captured.out


def test_main_returns_one_when_any_error(capsys):
    from puremacro.narrative.sources import health
    fake = [
        health.ConnectorStatus("iter_a", "ok",    5, "2024-01-01", "2024-04-01", None, 0.5),
        health.ConnectorStatus("iter_b", "error", 0, None, None, "ouch", 0.1),
    ]
    with mock.patch.object(health, "run", return_value=fake):
        rc = health.main([])
    assert rc == 1


def test_main_json_output_is_machine_readable(capsys):
    import json as jsonlib
    from puremacro.narrative.sources import health
    fake = [
        health.ConnectorStatus("iter_a", "ok", 3, "2024-01-01", "2024-03-01", None, 0.5),
    ]
    with mock.patch.object(health, "run", return_value=fake):
        health.main(["--json"])
    captured = capsys.readouterr()
    parsed = jsonlib.loads(captured.out)
    assert parsed["n_connectors"] == 1
    assert parsed["results"][0]["status"] == "ok"
    assert parsed["results"][0]["n_records"] == 3


def test_main_filter_narrows_results(capsys):
    from puremacro.narrative.sources import health
    fake = [
        health.ConnectorStatus("iter_fed_x", "ok",    1, "2024-01-01", "2024-01-01", None, 0.1),
        health.ConnectorStatus("iter_ecb_y", "ok",    1, "2024-01-01", "2024-01-01", None, 0.1),
    ]
    with mock.patch.object(health, "run", return_value=fake):
        health.main(["--filter", "fed"])
    out = capsys.readouterr().out
    assert "iter_fed_x" in out
    assert "iter_ecb_y" not in out
