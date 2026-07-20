"""Replications of canonical narrative-shock datasets.

Each module exposes:

* a top-level ``load(...)`` that returns a fully-populated
  :class:`~puremacro.narrative.NarrativeInstrument` (fetches the
  mirrored CSV when no local path is given);
* a ``<dataset>_csv_to_events`` helper that converts an already-loaded
  pandas DataFrame into a list of :class:`NarrativeEvent` — useful when
  the caller has the canonical CSV in hand or wants to merge several
  datasets into a single homogeneous panel.
"""
from .ramey_2011_defense import (
    load as load_ramey_2011_defense,
    ramey_csv_to_events,
)
from .romer_romer_2010 import (
    load as load_romer_romer_2010,
    romer_romer_2010_csv_to_events,
)
from .mertens_ravn_2013 import (
    load as load_mertens_ravn_2013,
    mertens_ravn_csv_to_events,
)
from .cloyne_2013_uk import (
    load as load_cloyne_2013_uk,
    cloyne_csv_to_events,
)
from .romer_romer_2017 import (
    load as load_romer_romer_2017,
    romer_romer_2017_csv_to_events,
)
from .dglp_2011_consolidations import (
    load as load_dglp_2011,
    dglp_csv_to_events,
)
from .devries_2011_consolidations import (
    load as load_devries_2011,
    devries_csv_to_events,
)
from .guajardo_2011_aeipf import (
    load as load_guajardo_2011,
    guajardo_csv_to_events,
)
from .mitchell_historical import (
    load as load_mitchell_historical,
    mitchell_csv_to_events,
)
from .adler_2024_consolidations import (
    load as load_adler_2024,
    adler_csv_to_events,
)
from .eu_nms_2023_consolidations import (
    load as load_eu_nms_2023,
    eu_nms_csv_to_events,
)
from .imf_covid_2022_announcements import (
    load as load_imf_covid_2022,
    imf_covid_2022_csv_to_events,
)

__all__ = [
    # Loaders
    "load_ramey_2011_defense", "load_romer_romer_2010",
    "load_mertens_ravn_2013", "load_cloyne_2013_uk",
    "load_romer_romer_2017", "load_dglp_2011",
    "load_devries_2011", "load_guajardo_2011", "load_mitchell_historical",
    "load_adler_2024", "load_eu_nms_2023", "load_imf_covid_2022",
    # CSV → events helpers
    "ramey_csv_to_events", "romer_romer_2010_csv_to_events",
    "mertens_ravn_csv_to_events", "cloyne_csv_to_events",
    "romer_romer_2017_csv_to_events", "dglp_csv_to_events",
    "devries_csv_to_events", "guajardo_csv_to_events", "mitchell_csv_to_events",
    "adler_csv_to_events", "eu_nms_csv_to_events", "imf_covid_2022_csv_to_events",
]
