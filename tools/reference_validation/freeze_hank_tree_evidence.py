"""Freeze a completed tree study's records and paths without large household arrays.

Every source artifact is authenticated before selection. Omitted NPZ arrays
remain explicitly listed with their original checksums; a compact fixture is
not represented as the complete run directory. Failed comparisons are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = (args.source / "record.json").read_bytes()
    record = json.loads(payload)
    if not record.get("extension_complete", False):
        raise ValueError("only a completed extension record can be frozen")
    for name, digest in record["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("artifacts must be local filenames")
        if hashlib.sha256((args.source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    kept, omitted = {}, {}
    for name, digest in record["artifacts"].items():
        if name.endswith(".npz"):
            omitted[name] = digest
        else:
            shutil.copyfile(args.source/name, args.output/name)
            kept[name] = digest
    (args.output / "record.json").write_bytes(payload)
    kept["record.json"] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": 1, "retained": kept, "omitted_household_arrays": omitted,
        "scope": "completed numerical record and all conditional/expected paths; regenerate the full arrays with the study runners",
    }
    (args.output / "retention.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
