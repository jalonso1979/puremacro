"""Portable DataFrame storage without pyarrow.

``pyarrow`` is an optional file-format dependency. ``numpy``'s own
``.npz`` container needs no additional engine: it is zlib plus a header,
implemented in numpy itself, and loads anywhere numpy does.

This module is a DataFrame ⇄ npz codec built on that: one array per
column, one JSON schema recording dtypes, index structure and column
names and ``attrs``, so a frame keeps both its index and its provenance.

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
Version 2 preserves nested dict/list/tuple metadata, scalar strings, numbers,
booleans, missing values, and Python/NumPy/pandas date and time scalars.
Timestamp metadata retains recognized named timezones and ``datetime.timezone``
fixed offsets; unsupported timezone implementations are rejected before writing.
Tuple dictionary keys are supported. Arbitrary objects, arrays and cycles
raise :class:`StoreError`, naming the metadata path; nothing is pickled.
Version-1 archives remain readable, with empty attrs as originally stored.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math

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

SCHEMA_VERSION = 2

# Key under which the JSON schema is stored inside the archive.
_SCHEMA_KEY = "__puremacro_schema__"


class StoreError(ValueError):
    """A frame could not be encoded, or an archive could not be decoded."""


def _encode_timestamp_timezone(stamp, path):
    """Use a reconstructible zone, not a timezone implementation's repr."""
    zone = stamp.tz
    if zone is None:
        return None
    if isinstance(zone, dt.timezone):
        offset = zone.utcoffset(None)
        name = zone.tzname(None)
        # Keep existing v2 strings for UTC and ordinary fixed offsets. A custom
        # name is not necessarily a timezone identifier (even if it resembles
        # one), so preserve both its fixed offset and its display name.
        if name == dt.timezone(offset).tzname(None):
            candidate = str(zone)
        else:
            micros = ((offset.days * 86400 + offset.seconds) * 1_000_000
                      + offset.microseconds)
            return ["fixed_offset", micros, name]
    else:
        candidate = getattr(zone, "key", None) or getattr(zone, "zone", None)
        if candidate is None:
            # dateutil tzfile objects carry the source zoneinfo filename.
            # Its repr, e.g. tzfile('/usr/share/zoneinfo/Europe/Paris'), cannot
            # be passed back to pandas as a zone name. Bundled dateutil files
            # may already carry a relative IANA name instead of an absolute path.
            filename = getattr(zone, "_filename", None)
            if isinstance(filename, str):
                candidate = filename.split("/zoneinfo/", 1)[-1]
    try:
        if not isinstance(candidate, str):
            raise ValueError("no portable timezone identifier")
        restored = stamp.tz_convert(candidate)
        if restored.utcoffset() != stamp.utcoffset() or restored.tzname() != stamp.tzname():
            raise ValueError("named timezone changes the stored offset or name")
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise StoreError(f"{path}: unsupported timestamp timezone {zone!r}: {exc}") from exc
    return candidate


def _decode_timestamp_timezone(value):
    # Strings are the original v2 representation and remain readable unchanged.
    if value is None or isinstance(value, str):
        return value
    if (isinstance(value, list) and len(value) == 3 and value[0] == "fixed_offset"
            and type(value[1]) is int and isinstance(value[2], str)):
        return dt.timezone(dt.timedelta(microseconds=value[1]), value[2])
    raise StoreError("attrs: invalid timestamp timezone payload")


def _encode_attrs(value, path="attrs", active=None, depth=0):
    """A tagged JSON tree, never object reconstruction or pickle.

    Containers are tagged too, so user dictionaries cannot collide with
    type tags. Mapping entries preserve non-string keys and insertion order.
    """
    if depth > 100:
        raise StoreError(f"{path}: metadata nesting exceeds 100 levels")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else ["float", str(value)]
    if value is pd.NA:
        return ["missing", "NA"]
    if value is pd.NaT:
        return ["missing", "NaT"]
    if isinstance(value, np.generic) and value.dtype.kind in "biufcUSMm":
        # Empty NumPy strings have a zero-width dtype, but their scalar
        # tobytes() includes a null terminator. Keep the wire payload empty.
        payload = value.tobytes() if value.dtype.itemsize else b""
        return ["numpy", [value.dtype.str, payload.hex()]]
    if isinstance(value, pd.Timestamp):
        return ["timestamp", [_encode_attrs(value.asm8),
                               _encode_timestamp_timezone(value, path)]]
    if isinstance(value, pd.Timedelta):
        return ["timedelta", _encode_attrs(value.asm8)]
    if isinstance(value, pd.Period):
        return ["period", [value.ordinal, value.freqstr]]
    if type(value) is dt.datetime:
        return ["datetime", [value.isoformat(), value.fold,
                              getattr(value.tzinfo, "key", None)]]
    if type(value) is dt.date:
        return ["date", value.isoformat()]
    if type(value) is dt.timedelta:
        return ["duration", [value.days, value.seconds, value.microseconds]]
    if type(value) in (dict, list, tuple):
        active = set() if active is None else active
        if id(value) in active:
            raise StoreError(f"{path}: cyclic metadata is not storable")
        active.add(id(value))
        try:
            if isinstance(value, dict):
                items = [[_encode_attrs(k, f"{path}.key[{i}]", active, depth + 1),
                          _encode_attrs(v, f"{path}[{k!r}]", active, depth + 1)]
                         for i, (k, v) in enumerate(value.items())]
            else:
                items = [_encode_attrs(v, f"{path}[{i}]", active, depth + 1)
                         for i, v in enumerate(value)]
            return [type(value).__name__, items]
        finally:
            active.remove(id(value))
    raise StoreError(f"{path}: unsupported metadata type {type(value).__name__}; "
                     "use scalar values and dict/list/tuple containers")


def _decode_attrs(value, depth=0):
    if depth > 100:
        raise StoreError("attrs: metadata nesting exceeds 100 levels")
    if value is None or type(value) in (str, bool, int, float):
        return value
    if not isinstance(value, list) or len(value) != 2:
        raise StoreError("attrs: malformed metadata tag")
    tag, data = value
    if not isinstance(tag, str):
        raise StoreError("attrs: metadata tag must be a string")
    arities = {"numpy": 2, "timestamp": 2, "period": 2,
               "datetime": 3, "duration": 3}
    if tag in arities and (not isinstance(data, list) or len(data) != arities[tag]):
        raise StoreError(f"attrs: invalid {tag!r} metadata payload")
    if tag in ("dict", "list", "tuple") and not isinstance(data, list):
        raise StoreError(f"attrs: invalid {tag!r} metadata container")
    try:
        if tag == "float" and data in ("nan", "inf", "-inf"):
            return float(data)
        if tag == "missing" and data in ("NA", "NaT"):
            return pd.NA if data == "NA" else pd.NaT
        if tag == "numpy":
            if not all(isinstance(part, str) for part in data):
                raise StoreError("attrs: invalid numpy scalar payload")
            dtype = np.dtype(data[0])
            payload = bytes.fromhex(data[1])
            if dtype.kind not in "biufcUSMm" or dtype.itemsize != len(payload):
                raise StoreError("attrs: invalid numpy scalar payload")
            # Array scalar extraction strips trailing null characters from
            # fixed-width strings; constructing from the bytes keeps them.
            if dtype.kind == "S":
                return np.bytes_(payload)
            if dtype.kind == "U":
                encoding = "utf-32-le" if dtype.str[0] == "<" else "utf-32-be"
                return np.str_(payload.decode(encoding, errors="surrogatepass"))
            return np.frombuffer(payload, dtype=dtype, count=1)[0]
        if tag == "timestamp":
            scalar = _decode_attrs(data[0], depth + 1)
            if not isinstance(scalar, np.datetime64):
                raise StoreError("attrs: invalid timestamp payload")
            zone = _decode_timestamp_timezone(data[1])
            stamp = pd.Timestamp(scalar)
            return stamp if zone is None else stamp.tz_localize("UTC").tz_convert(zone)
        if tag == "timedelta":
            scalar = _decode_attrs(data, depth + 1)
            if not isinstance(scalar, np.timedelta64):
                raise StoreError("attrs: invalid timedelta payload")
            return pd.Timedelta(scalar)
        if tag == "period":
            return pd.Period(ordinal=data[0], freq=data[1])
        if tag == "datetime":
            result = dt.datetime.fromisoformat(data[0]).replace(fold=data[1])
            if data[2] is not None:
                from zoneinfo import ZoneInfo
                result = result.astimezone(ZoneInfo(data[2]))
            return result
        if tag == "date":
            return dt.date.fromisoformat(data)
        if tag == "duration":
            return dt.timedelta(days=data[0], seconds=data[1], microseconds=data[2])
        if tag in ("list", "tuple"):
            items = [_decode_attrs(v, depth + 1) for v in data]
            return tuple(items) if tag == "tuple" else items
        if tag == "dict":
            result = {}
            for entry in data:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise StoreError("attrs: malformed metadata mapping entry")
                key, item = entry
                key = _decode_attrs(key, depth + 1)
                if key in result:
                    raise StoreError("attrs: duplicate metadata key")
                result[key] = _decode_attrs(item, depth + 1)
            return result
    except StoreError:
        raise
    except (TypeError, ValueError, IndexError, KeyError, OverflowError) as exc:
        raise StoreError(f"attrs: invalid {tag!r} metadata: {exc}") from exc
    raise StoreError(f"attrs: unsupported metadata tag {tag!r}")


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


def _encode_frame(df: pd.DataFrame, *, schema_version=SCHEMA_VERSION) -> tuple[dict, dict]:
    if not isinstance(df, pd.DataFrame):
        raise StoreError(f"expected a DataFrame, got {type(df).__name__}")
    if df.columns.has_duplicates:
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise StoreError(f"duplicate column labels are not storable: {dupes}")

    attrs = _encode_attrs(df.attrs) if schema_version >= 2 else None

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
        "version": schema_version,
        "n_rows": int(len(df)),
        "columns": columns,
        "index": {"levels": levels, "multi": isinstance(index, pd.MultiIndex)},
        "columns_name": df.columns.name,
    }
    if schema_version >= 2:
        meta["attrs"] = attrs
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
    if type(version) is not int or version not in (1, SCHEMA_VERSION):
        raise StoreError(
            f"archive schema version {version} is not supported "
            f"by this puremacro (supports 1 and {SCHEMA_VERSION})"
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
        # Int64/string/Categorical column back to float64/object. State the
        # dtype too: pandas 3 infers `str` from an object array of strings,
        # so a column stored as object would silently come back as `str`.
        payload = values.array if isinstance(values, (pd.Series, pd.Index)) else values
        data[i] = pd.Series(payload, index=index, copy=False,
                            dtype=getattr(payload, "dtype", None))
        names.append(name)

    df = pd.DataFrame(data, index=index, copy=False)
    df.columns = pd.Index(names, name=meta.get("columns_name"))
    if version >= 2:
        if "attrs" not in meta:
            raise StoreError("attrs: version-2 archive lacks metadata")
        attrs = _decode_attrs(meta["attrs"])
        if not isinstance(attrs, dict):
            raise StoreError("attrs: frame metadata must be a dict")
        df.attrs = attrs
    return df


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------

def dumps_frame(df: pd.DataFrame, *, compress: bool = True) -> bytes:
    """Encode ``df`` and its supported ``attrs`` as an in-memory npz archive."""
    return _dumps_frame(df, compress=compress)


def _dumps_frame(df, *, compress=True, schema_version=SCHEMA_VERSION):
    # The private version override is only for verifying old cartridges:
    # their original digests cover the v1 encoding, which had no attrs.
    arrays, meta = _encode_frame(df, schema_version=schema_version)
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
