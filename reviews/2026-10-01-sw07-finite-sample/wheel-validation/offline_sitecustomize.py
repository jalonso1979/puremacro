"""Validation-only network and optional-engine denial, copied as sitecustomize.py."""
import importlib.abc
import json
import os
from pathlib import Path
import socket
import sys


def denied(*args, **kwargs):
    raise RuntimeError("Network forbidden in installed-wheel validation")


class OfflineSocket(socket.socket):
    connect = denied
    connect_ex = denied


class NoStatsmodels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "statsmodels" or fullname.startswith("statsmodels."):
            raise RuntimeError("Statsmodels forbidden in installed-wheel validation")
        return None


socket.socket = OfflineSocket
socket.create_connection = denied
socket.getaddrinfo = denied
sys.meta_path.insert(0, NoStatsmodels())

# Both controls must fail before any puremacro import. Neither reaches a network.
try:
    with socket.socket() as control:
        control.connect(("127.0.0.1", 9))
except RuntimeError as exc:
    assert "Network forbidden" in str(exc)
else:
    raise AssertionError("Network denial positive control failed")
try:
    __import__("statsmodels")
except RuntimeError as exc:
    assert "Statsmodels forbidden" in str(exc)
else:
    raise AssertionError("Optional-engine denial positive control failed")

marker = Path(os.environ["PUREMACRO_VALIDATION_BASE"]) / "offline_controls"
marker.mkdir(parents=True, exist_ok=True)
(marker / f"{os.getpid()}.json").write_text(json.dumps({
    "pid": os.getpid(), "network_blocked_with_positive_control": True,
    "statsmodels_blocked_with_positive_control": True,
}) + "\n")
