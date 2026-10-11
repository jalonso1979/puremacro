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
from .adler_2024_consolidations import (
    adler_csv_to_events,
)
from .adler_2024_consolidations import (
    load as load_adler_2024,
)
from .cloyne_2013_uk import (
    cloyne_csv_to_events,
)
from .cloyne_2013_uk import (
    load as load_cloyne_2013_uk,
)
from .devries_2011_consolidations import (
    devries_csv_to_events,
)
from .devries_2011_consolidations import (
    load as load_devries_2011,
)
from .dglp_2011_consolidations import (
    dglp_csv_to_events,
)
from .dglp_2011_consolidations import (
    load as load_dglp_2011,
)
from .eu_nms_2023_consolidations import (
    eu_nms_csv_to_events,
)
from .eu_nms_2023_consolidations import (
    load as load_eu_nms_2023,
)
from .guajardo_2011_aeipf import (
    guajardo_csv_to_events,
)
from .guajardo_2011_aeipf import (
    load as load_guajardo_2011,
)
from .imf_covid_2022_announcements import (
    imf_covid_2022_csv_to_events,
)
from .imf_covid_2022_announcements import (
    load as load_imf_covid_2022,
)
from .mertens_ravn_2013 import (
    load as load_mertens_ravn_2013,
)
from .mertens_ravn_2013 import (
    mertens_ravn_csv_to_events,
)
from .mitchell_historical import (
    load as load_mitchell_historical,
)
from .mitchell_historical import (
    mitchell_csv_to_events,
)
from .ramey_2011_defense import (
    load as load_ramey_2011_defense,
)
from .ramey_2011_defense import (
    ramey_csv_to_events,
)
from .romer_romer_2010 import (
    load as load_romer_romer_2010,
)
from .romer_romer_2010 import (
    romer_romer_2010_csv_to_events,
)
from .romer_romer_2017 import (
    load as load_romer_romer_2017,
)
from .romer_romer_2017 import (
    romer_romer_2017_csv_to_events,
)

__all__ = [
    "adler_csv_to_events",
    "cloyne_csv_to_events",
    "devries_csv_to_events",
    "dglp_csv_to_events",
    "eu_nms_csv_to_events",
    "guajardo_csv_to_events",
    "imf_covid_2022_csv_to_events",
    "load_adler_2024",
    "load_cloyne_2013_uk",
    "load_devries_2011",
    "load_dglp_2011",
    "load_eu_nms_2023",
    "load_guajardo_2011",
    "load_imf_covid_2022",
    "load_mertens_ravn_2013",
    "load_mitchell_historical",
    # Loaders
    "load_ramey_2011_defense",
    "load_romer_romer_2010",
    "load_romer_romer_2017",
    "mertens_ravn_csv_to_events",
    "mitchell_csv_to_events",
    # CSV → events helpers
    "ramey_csv_to_events",
    "romer_romer_2010_csv_to_events",
    "romer_romer_2017_csv_to_events",
]
