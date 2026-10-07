"""Authenticated, offline ENIGH income-decile baskets for incidence analysis."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import numpy as np
import pandas as pd


def load_enigh2024_deciles() -> pd.DataFrame:
    """Return observed Mexican household expenditures and cash wage exposures.

    Each row is a national household income decile. Monetary columns are MXN
    per household per quarter. ``households`` contains expansion counts for
    *all* households in the decile; zero-spending households are included.
    ``frame.attrs`` carries source hashes, exact transformations and limitations.
    ``outward_transfers`` is excluded from the eight consumption categories.

    This is a set of observed group means, not microdata or an estimated trade
    model. Sampling covariance, import exposure and policy responses are absent.
    """
    directory = files("puremacro.datasets").joinpath("data")
    metadata = json.loads(directory.joinpath("enigh2024_deciles_metadata.json").read_text(encoding="utf-8"))
    content = directory.joinpath("enigh2024_deciles.csv").read_bytes()
    if hashlib.sha256(content).hexdigest() != metadata["csv_sha256"]:
        raise ValueError("Bundled ENIGH CSV checksum differs from its provenance record")
    from io import BytesIO
    frame = pd.read_csv(BytesIO(content)).set_index("decile")
    if not np.isfinite(frame.to_numpy(dtype=float)).all() or (frame < 0).any().any():
        raise ValueError("Bundled ENIGH data contain invalid amounts")
    frame.attrs.update(metadata)
    return frame


__all__ = ["load_enigh2024_deciles"]
