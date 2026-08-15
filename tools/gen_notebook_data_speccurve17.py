#!/usr/bin/env python
"""Freeze the Notebook-17 dataset: the uncertainty-identification panel.

Notebook 17 ("One shock, ten identifications") runs the identification
menu on the SAME frozen monthly panel as the research pipeline
``tools/run_uncertainty_ident_spec_curve.py``. The pipeline's freeze is
``docs/research/uncertainty_identification/output/panel_monthly.csv``
(assembled once from key-free sources, sha in ``panel_manifest.json``);
notebooks can only read package data, so this tool copies that freeze
into ``puremacro/replication/data/speccurve17_panel.csv`` byte-for-byte
and records the source hash. Rerun only if the pipeline re-freezes its
panel (--refresh-data there); the two files must stay identical — the
point of Notebook 17 is "same data, different assumptions".
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = (REPO_ROOT / "docs" / "research" / "uncertainty_identification"
       / "output" / "panel_monthly.csv")
DST = (REPO_ROOT / "puremacro" / "replication" / "data"
       / "speccurve17_panel.csv")


def main() -> int:
    if not SRC.exists():
        print(f"source freeze missing: {SRC}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(SRC.read_bytes()).hexdigest()
    shutil.copyfile(SRC, DST)
    if hashlib.sha256(DST.read_bytes()).hexdigest() != digest:
        print("copy verification failed", file=sys.stderr)
        return 1
    print(f"wrote {DST}")
    print(f"sha256 {digest} (== pipeline panel_monthly.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
