"""Private run-record helpers shared by the policy briefs (environment, hashes, atomic JSON)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_average() -> list[float] | None:
    """``os.getloadavg()`` rounded to 2 decimals, or None where missing (Windows) or failing."""
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return None
    try:
        return [round(float(v), 2) for v in getloadavg()]
    except OSError:
        return None


def _peak_rss_gb() -> float | None:
    """Peak resident memory of this process in GB, or None where unavailable (Windows)."""
    if sys.platform == "win32":
        return None
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0  # bytes on macOS, KiB elsewhere
    return round(peak * scale / 1e9, 3)


def environment() -> dict[str, Any]:
    """Interpreter, platform, package versions, load average and peak RSS; no paths or user names."""
    import matplotlib
    import numpy
    import pandas
    import scipy

    import puremacro

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "puremacro": puremacro.__version__,
        "load_average": _load_average(),
        "peak_rss_gb": _peak_rss_gb(),
    }


def sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """Write UTF-8 JSON to a temp file beside ``path`` and ``os.replace`` it into place."""
    target = Path(path)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         delete=False, suffix=".tmp") as handle:
            tmp_name = handle.name
            json.dump(obj, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_name, target)
    except BaseException:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
