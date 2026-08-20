"""Portable, self-verifying data bundles — the ``.pmz`` cartridge.

The gap between "puremacro imports on an iPad" and "puremacro is useful
on an iPad" is data. The fetchers need sockets and CORS-friendly
endpoints; the cached panels are parquet, which needs pyarrow; and both
assumptions fail on a tablet. A cartridge closes that gap by moving the
fetch to the machine that can do it:

    # workstation, online
    panel = build_panel(["USA", "MEX", "ESP"], 1990, 2026)
    pocket.pack(panel, "g7.pmz", source="OECD QNA", vintage="2026-08-19")

    # iPad, airplane mode
    cart = pocket.load("g7.pmz")
    cart.verify()                 # sha256 per frame
    panel = cart.frame()
    cart.provenance.vintage       # '2026-08-19'

A ``.pmz`` is a plain zip: a JSON manifest plus one npz per frame
(:mod:`puremacro.runtime.store`). Both halves are stdlib or numpy, so a
cartridge reads anywhere numpy does — no pyarrow, no puremacro even, if
someone wants to open it with a script.

**Provenance is the point.** A CSV emailed to a co-author is a number
with no history. A cartridge records what produced it — the source, the
vintage, the machine, the puremacro version, and (for
:func:`~puremacro.pocket.snapshot`) the exact call — so the panel you
re-open in six months can still answer "where did this come from?".
Cartridges are a transport format, not a trust boundary: the checksums
detect corruption in transit, and nothing more.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from puremacro.runtime import store

__all__ = [
    "FORMAT_VERSION",
    "CartridgeError",
    "FrameRecord",
    "Provenance",
    "Cartridge",
    "pack",
    "packs",
    "load",
    "loads",
    "inspect_cartridge",
    "snapshot",
    "to_base64",
    "from_base64",
]

FORMAT_VERSION = 1

_MANIFEST_NAME = "manifest.json"
_FRAME_DIR = "frames"


class CartridgeError(ValueError):
    """A cartridge is malformed, corrupt, or from an unsupported version."""


@dataclass(frozen=True)
class FrameRecord:
    """What the manifest records about one frame.

    Attributes
    ----------
    name : str
        Key the frame is stored under.
    n_rows, n_cols : int
        Shape at pack time.
    columns : tuple[str, ...]
        Column labels, stringified.
    index : str
        Repr of the index type, e.g. ``"PeriodIndex[Q-DEC]"``.
    sha256 : str
        Digest of the stored npz payload; checked by
        :meth:`Cartridge.verify`.
    n_bytes : int
        Compressed payload size.
    """

    name: str
    n_rows: int
    n_cols: int
    columns: tuple
    index: str
    sha256: str
    n_bytes: int


@dataclass(frozen=True)
class Provenance:
    """Where a cartridge came from.

    Attributes
    ----------
    created : str
        UTC ISO-8601 timestamp of packing.
    puremacro_version : str
        Version that wrote the cartridge.
    source : str | None
        Free text naming the upstream, e.g. ``"OECD QNA"``.
    vintage : str | None
        Data vintage / release date, distinct from ``created``.
    notes : str | None
        Anything the packer wants their future self to read.
    call : str | None
        The reproducing call, when packed by :func:`snapshot`.
    host : dict
        Capability snapshot of the packing machine.
    """

    created: str
    puremacro_version: str
    source: str | None = None
    vintage: str | None = None
    notes: str | None = None
    call: str | None = None
    host: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Cartridge:
    """A loaded cartridge: frames plus the provenance that explains them."""

    frames: dict
    provenance: Provenance
    records: tuple

    def __getitem__(self, name: str) -> pd.DataFrame:
        try:
            return self.frames[name]
        except KeyError:
            raise KeyError(
                f"no frame {name!r} in this cartridge; it holds "
                f"{sorted(self.frames)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.frames

    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, name: str | None = None) -> pd.DataFrame:
        """The named frame, or the only one if the cartridge holds one."""
        if name is not None:
            return self[name]
        if len(self.frames) != 1:
            raise CartridgeError(
                f"this cartridge holds {len(self.frames)} frames "
                f"({sorted(self.frames)}); name the one you want"
            )
        return next(iter(self.frames.values()))

    def verify(self) -> bool:
        """Re-encode every frame and compare digests to the manifest.

        Returns True, or raises :class:`CartridgeError` naming the frames
        that do not match.

        Note this re-encodes rather than re-reading the file, so it also
        catches a frame mutated in memory since load — which is the
        common case when a notebook has been running for a while.
        """
        bad = []
        for rec in self.records:
            payload = store.dumps_frame(self.frames[rec.name])
            if _digest(payload) != rec.sha256:
                bad.append(rec.name)
        if bad:
            raise CartridgeError(
                f"checksum mismatch for {bad}: the frame(s) changed since "
                f"the cartridge was packed, or the file is corrupt"
            )
        return True

    def summary(self) -> str:
        """A short human-readable description."""
        p = self.provenance
        lines = [
            f"cartridge · {len(self.frames)} frame(s) · puremacro {p.puremacro_version}",
            f"  created : {p.created}",
        ]
        if p.source:
            lines.append(f"  source  : {p.source}")
        if p.vintage:
            lines.append(f"  vintage : {p.vintage}")
        if p.call:
            lines.append(f"  call    : {p.call}")
        if p.notes:
            lines.append(f"  notes   : {p.notes}")
        for rec in self.records:
            lines.append(
                f"  [{rec.name}] {rec.n_rows}x{rec.n_cols} on {rec.index} "
                f"({rec.n_bytes / 1024:.0f} KB)"
            )
        return "\n".join(lines)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _index_repr(index: pd.Index) -> str:
    if isinstance(index, pd.MultiIndex):
        inner = ", ".join(
            _index_repr(index.get_level_values(j)) for j in range(index.nlevels)
        )
        return f"MultiIndex[{inner}]"
    if isinstance(index.dtype, pd.PeriodDtype):
        return f"PeriodIndex[{str(index.dtype)[len('period['):-1]}]"
    return f"{type(index).__name__}[{index.dtype}]"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_mapping(data) -> dict:
    if isinstance(data, pd.DataFrame):
        return {"data": data}
    if isinstance(data, Mapping):
        if not data:
            raise CartridgeError("nothing to pack: the mapping is empty")
        for name, frame in data.items():
            if not isinstance(name, str):
                raise CartridgeError(
                    f"frame names must be strings; got {type(name).__name__}"
                )
            if not isinstance(frame, pd.DataFrame):
                raise CartridgeError(
                    f"frame {name!r} is a {type(frame).__name__}, not a DataFrame"
                )
        return dict(data)
    raise CartridgeError(
        f"pack() takes a DataFrame or a {{name: DataFrame}} mapping, got "
        f"{type(data).__name__}"
    )


def packs(data, *, source: str | None = None, vintage: str | None = None,
          notes: str | None = None, call: str | None = None,
          host: dict | None = None) -> bytes:
    """Build a cartridge in memory and return its bytes.

    Same arguments as :func:`pack`, minus the path.
    """
    from puremacro import __version__
    from puremacro.runtime import capabilities

    frames = _as_mapping(data)
    if host is None:
        caps = capabilities()
        host = {"host": caps.host, "device": caps.device, "python": caps.python}

    prov = Provenance(
        created=_now(), puremacro_version=__version__, source=source,
        vintage=vintage, notes=notes, call=call, host=host,
    )

    records, payloads = [], {}
    for name, frame in frames.items():
        payload = store.dumps_frame(frame)
        payloads[name] = payload
        records.append(FrameRecord(
            name=name,
            n_rows=int(len(frame)),
            n_cols=int(frame.shape[1]),
            columns=tuple(str(c) for c in frame.columns),
            index=_index_repr(frame.index),
            sha256=_digest(payload),
            n_bytes=len(payload),
        ))

    manifest = {
        "format": "puremacro-cartridge",
        "version": FORMAT_VERSION,
        "provenance": {
            "created": prov.created,
            "puremacro_version": prov.puremacro_version,
            "source": prov.source,
            "vintage": prov.vintage,
            "notes": prov.notes,
            "call": prov.call,
            "host": prov.host,
        },
        "frames": [
            {
                "name": r.name, "n_rows": r.n_rows, "n_cols": r.n_cols,
                "columns": list(r.columns), "index": r.index,
                "sha256": r.sha256, "n_bytes": r.n_bytes,
            }
            for r in records
        ],
    }

    buf = io.BytesIO()
    # ZIP_STORED: the npz payloads are already deflated, so a second pass
    # costs time and saves nothing.
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True))
        for name, payload in payloads.items():
            zf.writestr(f"{_FRAME_DIR}/{name}.npz", payload)
    return buf.getvalue()


def pack(data, path, *, source: str | None = None, vintage: str | None = None,
         notes: str | None = None, call: str | None = None) -> Path:
    """Write ``data`` to ``path`` as a ``.pmz`` cartridge.

    Parameters
    ----------
    data : DataFrame | Mapping[str, DataFrame]
        A single frame (stored as ``"data"``) or several, named.
    path : str | Path
        Destination. ``.pmz`` is conventional but not enforced.
    source, vintage, notes : str, optional
        Provenance, recorded verbatim in the manifest.
    call : str, optional
        The call that reproduces this data. Usually left to
        :func:`snapshot`.

    Returns
    -------
    Path
        The path written.
    """
    target = Path(path)
    target.write_bytes(packs(data, source=source, vintage=vintage,
                             notes=notes, call=call))
    return target


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        raw = zf.read(_MANIFEST_NAME)
    except KeyError:
        raise CartridgeError(
            "not a puremacro cartridge: no manifest.json in the archive"
        ) from None
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("format") != "puremacro-cartridge":
        raise CartridgeError("not a puremacro cartridge (wrong format tag)")
    version = manifest.get("version")
    if version != FORMAT_VERSION:
        raise CartridgeError(
            f"cartridge format version {version} is not readable by this "
            f"puremacro (supports {FORMAT_VERSION})"
        )
    return manifest


def _from_zip(zf: zipfile.ZipFile, *, verify: bool) -> Cartridge:
    manifest = _read_manifest(zf)
    records = tuple(
        FrameRecord(
            name=f["name"], n_rows=f["n_rows"], n_cols=f["n_cols"],
            columns=tuple(f["columns"]), index=f["index"],
            sha256=f["sha256"], n_bytes=f["n_bytes"],
        )
        for f in manifest["frames"]
    )
    frames, mismatched = {}, []
    payloads = {}
    for rec in records:
        payload = zf.read(f"{_FRAME_DIR}/{rec.name}.npz")
        if verify and _digest(payload) != rec.sha256:
            mismatched.append(rec.name)
            continue
        payloads[rec.name] = payload
    if mismatched:
        raise CartridgeError(
            f"corrupt cartridge: stored checksum does not match the payload "
            f"for {mismatched}"
        )
    for name, payload in payloads.items():
        frames[name] = store.loads_frame(payload)
    prov = Provenance(**manifest["provenance"])
    return Cartridge(frames=frames, provenance=prov, records=records)


def loads(payload: bytes, *, verify: bool = True) -> Cartridge:
    """Load a cartridge from bytes."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise CartridgeError("not a cartridge: the bytes are not a zip") from exc
    with zf:
        return _from_zip(zf, verify=verify)


def load(path, *, verify: bool = True) -> Cartridge:
    """Load a cartridge from ``path``.

    Parameters
    ----------
    verify : bool, default True
        Check each frame's sha256 while reading. Turn it off only for a
        very large cartridge you already trust.
    """
    try:
        zf = zipfile.ZipFile(Path(path))
    except zipfile.BadZipFile as exc:
        raise CartridgeError(f"{path} is not a cartridge (not a zip)") from exc
    with zf:
        return _from_zip(zf, verify=verify)


def inspect_cartridge(path) -> dict:
    """Read the manifest without decoding any data.

    Cheap enough to run over a directory of cartridges to find the one
    with the vintage you want.
    """
    with zipfile.ZipFile(Path(path)) as zf:
        return _read_manifest(zf)


def snapshot(func, *args, path=None, source: str | None = None,
             vintage: str | None = None, notes: str | None = None, **kwargs):
    """Call ``func(*args, **kwargs)`` and pack the result, recording the call.

    The provenance ends up carrying the reproducing expression, so the
    cartridge documents itself::

        pocket.snapshot(build_panel, ["USA", "MEX"], 1990, 2026,
                           path="panel.pmz", source="OECD QNA")
        # provenance.call == "build_panel(['USA', 'MEX'], 1990, 2026)"

    Returns the cartridge bytes when ``path`` is None, else the path.
    """
    result = func(*args, **kwargs)
    parts = [repr(a) for a in args]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    call = f"{getattr(func, '__name__', repr(func))}({', '.join(parts)})"
    payload = packs(result, source=source, vintage=vintage, notes=notes,
                    call=call)
    if path is None:
        return payload
    target = Path(path)
    target.write_bytes(payload)
    return target


def to_base64(source, *, width: int = 76) -> str:
    """Render a cartridge as base64 text, for moving it by clipboard.

    Getting a file onto an iPad can be more friction than the analysis
    itself. This turns a cartridge into text you can paste into a
    notebook cell, an email, or a message to yourself::

        blob = pocket.to_base64("g7.pmz")     # workstation
        pocket.from_base64(blob, "g7.pmz")    # iPad

    Parameters
    ----------
    source : str | Path | bytes
        A cartridge path, or cartridge bytes from :func:`packs`.
    width : int, default 76
        Line width; 0 for one unbroken line.
    """
    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    text = base64.b64encode(payload).decode("ascii")
    if width <= 0:
        return text
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


def from_base64(text: str, path=None):
    """Inverse of :func:`to_base64`. Whitespace in ``text`` is ignored.

    Returns a :class:`Cartridge` when ``path`` is None, else writes the
    file and returns its path.
    """
    compact = "".join(text.split())
    try:
        payload = base64.b64decode(compact, validate=True)
    except Exception as exc:
        raise CartridgeError(
            "the text is not valid base64 — check it survived the paste "
            "intact (some clients rewrite long lines)"
        ) from exc
    if path is None:
        return loads(payload)
    target = Path(path)
    target.write_bytes(payload)
    return target
