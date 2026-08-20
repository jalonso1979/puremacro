"""Portable DataFrame storage without pyarrow.

``pyarrow`` is a base dependency of this package and has no Pyodide
wheel, so every parquet path — ``cache``, ``fetch.labor*``,
``shock_atlas``, ``build_panel``, the bundled teaching datasets — is
unreachable on an iPad. ``numpy``'s own ``.npz`` container has no such
problem: it is zlib plus a header, implemented in numpy itself, and it
loads anywhere numpy does.

This module is a DataFrame ⇄ npz codec built on that: one array per
column, one JSON schema recording dtypes, index structure and column
names, so a frame survives the round trip with its index intact rather
than arriving as anonymous columns.

    >>> import pandas as pd
    >>> from puremacro.runtime import store
    >>> df = pd.DataFrame({"gdp": [1.0, 2.0]},
    ...                   index=pd.period_range("2020Q1", periods=2, freq="Q"))
    >>> store.save_frame(df, "/tmp/gdp.npz")          # doctest: +SKIP
    >>> store.load_frame("/tmp/gdp.npz").equals(df)   # doctest: +SKIP
    True

Supported: every numpy dtype, ``datetime64`` (tz-aware and naive),
``PeriodIndex`` / period columns, ``Categorical``, pandas nullable
extension dtypes (``Int64``, ``boolean``, ``string``), object columns of
strings, and ``MultiIndex`` of any of the above. Object columns holding
arbitrary Python objects are rejected loudly rather than pickled —
``allow_pickle`` archives are neither portable nor safe to load.

The format is versioned (:data:`SCHEMA_VERSION`) and self-describing:
:func:`describe` reads the schema without materialising the data.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd

__all__ = [
    "SCHEMA_VERSION",
    "StoreError",
    "dumps_frame",
    "loads_frame",
    "save_frame",
    "load_frame",
    "describe",
]

SCHEMA_VERSION = 1

# Key under which the JSON schema is stored inside the archive.
_SCHEMA_KEY = "__puremacro_schema__"


class StoreError(ValueError):
    """A frame could not be encoded, or an archive could not be decoded."""


# ---------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------

def _encode_values(values: pd.Series | pd.Index, key: str, out: dict,
                   label: str | None = None) -> dict:
    """Encode one column/level into ``out``; return its schema fragment.

    ``label`` is the user-facing name used in error messages; ``key`` is
    the internal archive key.
    """
    label = key if label is None else label
    dtype = values.dtype

    if isinstance(dtype, pd.PeriodDtype):
        # `.array` is the one accessor a Series and an Index share; a
        # period *column* has no `.asi8` of its own.
        out[key] = np.asarray(values.array.asi8, dtype=np.int64)
        # `str(dtype)` is "period[Q-DEC]"; its inner token is the only
        # form that round-trips. `dtype.freq.freqstr` gives "QE-DEC" on
        # pandas >= 2.2, which PeriodDtype then refuses to parse back.
        return {"kind": "period", "freq": str(dtype)[len("period["):-1]}

    if isinstance(dtype, pd.DatetimeTZDtype):
        as_utc = pd.DatetimeIndex(values.array).tz_convert("UTC")
        out[key] = np.asarray(as_utc.asi8, dtype=np.int64)
        return {"kind": "datetime_tz", "tz": str(dtype.tz), "unit": dtype.unit,
                "freq": _index_freq(values)}

    if isinstance(dtype, pd.CategoricalDtype):
        cat = values if isinstance(values, pd.Series) else pd.Series(values)
        cat = cat.cat
        out[key] = np.asarray(cat.codes, dtype=np.int64)
        sub = _encode_values(pd.Series(cat.categories), f"{key}__cats", out, label)
        return {"kind": "categorical", "ordered": bool(cat.ordered),
                "categories": sub}

    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        # Nullable Int64 / boolean / string: store the mask separately and
        # the payload as its numpy-native equivalent.
        arr = pd.array(values)
        mask = np.asarray(arr.isna(), dtype=bool)
        filled = pd.Series(arr).fillna(_fill_for(dtype))
        try:
            payload = filled.to_numpy(dtype=_numpy_dtype_for(dtype))
        except (TypeError, ValueError) as exc:
            raise StoreError(
                f"column {label!r}: cannot store extension dtype {dtype!r} "
                f"without pickling"
            ) from exc
        out[key] = payload
        out[f"{key}__mask"] = mask
        return {"kind": "nullable", "pandas_dtype": str(dtype)}

    arr = np.asarray(values)
    if arr.dtype == object:
        # Only strings (plus nulls) are portable without pickle.
        flat = pd.Series(arr)
        mask = np.asarray(flat.isna(), dtype=bool)
        non_null = flat[~mask]
        if not all(isinstance(v, str) for v in non_null):
            bad = next(
                (type(v).__name__ for v in non_null if not isinstance(v, str)),
                "object",
            )
            raise StoreError(
                f"column {label!r} holds {bad} objects. The npz format stores "
                f"arrays, not pickles — convert the column to a string, a "
                f"number, or a datetime first."
            )
        out[key] = np.asarray(flat.fillna("").astype(str).to_numpy(), dtype=np.str_)
        out[f"{key}__mask"] = mask
        return {"kind": "string"}

    if arr.dtype.kind == "M":
        out[key] = arr.view(np.int64)
        # A DatetimeIndex built by date_range carries a freq that is part
        # of its identity (assert_frame_equal compares it); a datetime
        # *column* never has one.
        return {"kind": "datetime", "dtype": str(arr.dtype),
                "freq": _index_freq(values)}

    if arr.dtype.kind == "m":
        out[key] = arr.view(np.int64)
        return {"kind": "timedelta", "dtype": str(arr.dtype)}

    out[key] = arr
    return {"kind": "plain", "dtype": str(arr.dtype)}


def _index_freq(values) -> str | None:
    """The frequency string of a DatetimeIndex, or None for anything else."""
    if isinstance(values, pd.DatetimeIndex):
        return values.freqstr
    return None


def _is_string_dtype(dtype) -> bool:
    """True for every pandas string extension dtype.

    Matching on ``"string" in str(dtype)`` is not enough. pandas 2 spells
    the nullable string dtype ``"string"``, but pandas 3 makes
    ``StringDtype(na_value=nan)`` the dtype of a plain string column and
    spells it ``"str"`` — which that test misses, sending every string
    column down the integer path to ``int('MEX')``.
    """
    if isinstance(dtype, pd.StringDtype):
        return True
    name = str(dtype).lower()
    # "string[pyarrow]" / "large_string[pyarrow]" reach here as ArrowDtype.
    return name == "str" or "string" in name


def _is_bool_dtype(dtype) -> bool:
    return isinstance(dtype, pd.BooleanDtype) or "bool" in str(dtype).lower()


def _fill_for(dtype):
    if _is_string_dtype(dtype):
        return ""
    if _is_bool_dtype(dtype):
        return False
    return 0


def _numpy_dtype_for(dtype):
    if _is_string_dtype(dtype):
        return np.str_
    if _is_bool_dtype(dtype):
        return bool
    # Masked numeric dtypes (Int64, UInt32, Float64, and their Arrow
    # equivalents) carry the numpy dtype they widen; ask them rather than
    # parsing their name.
    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    if numpy_dtype is not None:
        return numpy_dtype
    name = str(dtype).lower()
    if name.startswith("u"):
        return np.uint64
    if "float" in name:
        return np.float64
    return np.int64


def _encode_frame(df: pd.DataFrame) -> tuple[dict, dict]:
    if not isinstance(df, pd.DataFrame):
        raise StoreError(f"expected a DataFrame, got {type(df).__name__}")
    if df.columns.has_duplicates:
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise StoreError(f"duplicate column labels are not storable: {dupes}")

    arrays: dict = {}
    columns = []
    for i, name in enumerate(df.columns):
        schema = _encode_values(df[name], f"c{i}", arrays, str(name))
        schema["name"] = name
        schema["name_type"] = type(name).__name__
        columns.append(schema)

    index = df.index
    levels = []
    if isinstance(index, pd.MultiIndex):
        for j in range(index.nlevels):
            schema = _encode_values(index.get_level_values(j), f"i{j}", arrays,
                                    f"index level {index.names[j]!r}")
            schema["name"] = index.names[j]
            levels.append(schema)
    else:
        schema = _encode_values(index, "i0", arrays, f"index {index.name!r}")
        schema["name"] = index.name
        levels.append(schema)

    meta = {
        "version": SCHEMA_VERSION,
        "n_rows": int(len(df)),
        "columns": columns,
        "index": {"levels": levels, "multi": isinstance(index, pd.MultiIndex)},
        "columns_name": df.columns.name,
    }
    return arrays, meta


# ---------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------

def _decode_values(schema: dict, key: str, arrays) -> pd.Series | pd.Index:
    kind = schema["kind"]
    if kind == "period":
        return pd.PeriodIndex.from_ordinals(
            np.asarray(arrays[key]), freq=schema["freq"],
        )
    if kind == "datetime_tz":
        # The payload is `asi8`, which counts in the dtype's OWN unit, so it
        # has to be read back in that unit. pandas 2 made every timestamp
        # nanosecond; pandas 3 gives `date_range` microsecond resolution, and
        # reading a microsecond count as nanoseconds lands the whole index in
        # 1970 with its spacing destroyed. Archives written before the unit
        # was recorded are nanosecond by construction.
        unit = schema.get("unit") or "ns"
        idx = pd.DatetimeIndex(
            np.asarray(arrays[key]).view(f"datetime64[{unit}]"), tz="UTC",
        ).tz_convert(schema["tz"])
        if schema.get("freq"):
            idx.freq = schema["freq"]
        return idx
    if kind == "categorical":
        cats = _decode_values(schema["categories"], f"{key}__cats", arrays)
        return pd.Categorical.from_codes(
            np.asarray(arrays[key]), categories=pd.Index(cats),
            ordered=schema["ordered"],
        )
    if kind == "nullable":
        values = np.asarray(arrays[key])
        mask = np.asarray(arrays[f"{key}__mask"])
        series = pd.Series(values).astype(schema["pandas_dtype"])
        # ``None`` lands as whatever that dtype calls missing: pd.NA for the
        # nullable dtypes, NaN for pandas 3's default ``str``, whose na_value
        # is NaN and which stores a literal pd.NA as an object instead.
        series[mask] = None
        return series
    if kind == "string":
        values = np.asarray(arrays[key]).astype(object)
        mask = np.asarray(arrays[f"{key}__mask"])
        values[mask] = None
        return pd.Series(values, dtype=object)
    if kind == "datetime":
        values = np.asarray(arrays[key]).view(schema["dtype"])
        if schema.get("freq"):
            return pd.DatetimeIndex(values, freq=schema["freq"])
        return pd.Series(values)
    if kind == "timedelta":
        return pd.Series(np.asarray(arrays[key]).view(schema["dtype"]))
    if kind == "plain":
        return pd.Series(np.asarray(arrays[key]).astype(schema["dtype"], copy=False))
    raise StoreError(f"unknown column kind {kind!r} (archive from a newer puremacro?)")


def _decode_frame(arrays, meta: dict) -> pd.DataFrame:
    version = meta.get("version")
    if version != SCHEMA_VERSION:
        raise StoreError(
            f"archive schema version {version} != {SCHEMA_VERSION} supported "
            f"by this puremacro"
        )

    levels = [
        pd.Index(_decode_values(s, f"i{j}", arrays), name=s["name"])
        for j, s in enumerate(meta["index"]["levels"])
    ]
    if meta["index"]["multi"]:
        index = pd.MultiIndex.from_arrays(levels, names=[s["name"] for s in meta["index"]["levels"]])
    else:
        index = levels[0]

    data = {}
    names = []
    for i, schema in enumerate(meta["columns"]):
        name = schema["name"]
        if schema.get("name_type") == "tuple" and isinstance(name, list):
            name = tuple(name)
        values = _decode_values(schema, f"c{i}", arrays)
        # Keep the pandas array, never np.asarray: that would flatten an
        # Int64/string/Categorical column back to float64/object.
        payload = values.array if isinstance(values, (pd.Series, pd.Index)) else values
        data[i] = pd.Series(payload, index=index, copy=False)
        names.append(name)

    df = pd.DataFrame(data, index=index, copy=False)
    df.columns = pd.Index(names, name=meta.get("columns_name"))
    return df


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------

def dumps_frame(df: pd.DataFrame, *, compress: bool = True) -> bytes:
    """Encode ``df`` as an in-memory npz archive."""
    arrays, meta = _encode_frame(df)
    arrays = dict(arrays)
    arrays[_SCHEMA_KEY] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8,
    )
    buf = io.BytesIO()
    saver = np.savez_compressed if compress else np.savez
    saver(buf, **arrays)
    return buf.getvalue()


def loads_frame(payload: bytes) -> pd.DataFrame:
    """Decode bytes produced by :func:`dumps_frame`."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return _decode_frame(archive, _read_schema(archive))


def save_frame(df: pd.DataFrame, path, *, compress: bool = True) -> None:
    """Write ``df`` to ``path`` as npz. Works with no pyarrow installed."""
    from pathlib import Path

    Path(path).write_bytes(dumps_frame(df, compress=compress))


def load_frame(path) -> pd.DataFrame:
    """Read a frame written by :func:`save_frame`."""
    with np.load(path, allow_pickle=False) as archive:
        return _decode_frame(archive, _read_schema(archive))


def describe(path) -> dict:
    """The archive's schema — shape, dtypes, index — without decoding data."""
    with np.load(path, allow_pickle=False) as archive:
        return _read_schema(archive)


def _read_schema(archive) -> dict:
    if _SCHEMA_KEY not in archive:
        raise StoreError(
            "not a puremacro frame archive (no schema record). Plain npz "
            "files can be read with numpy.load."
        )
    return json.loads(bytes(archive[_SCHEMA_KEY]).decode("utf-8"))
