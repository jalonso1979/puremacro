"""Mexico (Banxico / INEGI) — why there is no native vintage provider.

This module deliberately registers **no** vintage source. It exists so
that the determination behind that decision is recorded in the code
rather than lost, and so ``known_gaps("banxico")`` can explain itself.

THE FINDING
-----------
Neither Banco de México's SIE API nor INEGI's Banco de Indicadores /
BIE API publishes historical vintages of quarterly GDP. Both serve a
single current series that is **overwritten in place** at each
revision, and neither payload schema carries an edition, vintage,
release or as-of field of any kind.

INEGI says as much in its own quarterly GDP bulletin: incorporating
newly available source data means "se identifican diferencias en los
niveles de los valores, índices y variaciones que se publicaron
oportunamente" — the revised numbers replace the ones published
earlier, and the earlier ones are not retained as a product.

WHAT TO USE INSTEAD
-------------------
The OECD STES revisions archive is a genuine third-party vintage
archive of INEGI's series and is the right source::

    vintage_panel(["MEX"])          # provider "oecd_stes" by default

That gives 329 monthly editions from February 1999 — which is what
replaces a hand-made spreadsheet dump. The same archive covers Spain
(``ESP``), so Banco de España / INE need no separate connector either.

The one caveat, inherited from the archive rather than from Mexico:
``EDITION`` is the month of the OECD's snapshot, not INEGI's release
date, and the OECD ingests with a lag. Ordering and revision
magnitudes are sound; release-event dating is not.

BANXICO'S API, FOR COMPLETENESS
-------------------------------
Banxico SIE is live at ``https://www.banxico.org.mx/SieAPIRest/
service/v1/`` and needs a free token (a CAPTCHA form, no account),
sent as the ``Bmx-Token`` header or a ``?token=`` parameter. INEGI's
API needs a free registered token. Both serve current data only, so
neither is wired up here.
"""
from __future__ import annotations

from .catalog import register_catalog


#: Why Mexico has no native vintage provider, quoted by ``known_gaps``.
MEXICO_VINTAGE_NOTE = (
    "Neither Banxico SIE nor INEGI BIE retains previously published "
    "editions of quarterly GDP — both overwrite in place and their "
    "payloads carry no vintage field. Use provider 'oecd_stes', which "
    "archives INEGI's series with 329 monthly editions from 1999-02."
)

#: Same determination for Spain's national statistics office.
SPAIN_VINTAGE_NOTE = (
    "INE / Banco de España publish no machine-readable vintage archive "
    "for quarterly GDP. Use provider 'oecd_stes' (331 editions from "
    "1999-02)."
)

BANXICO_SIE_BASE = "https://www.banxico.org.mx/SieAPIRest/service/v1/"
INEGI_BIE_BASE = (
    "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/"
)


def _register() -> None:
    """Record the gaps. No provider is registered on purpose."""
    register_catalog(
        "banxico", {},
        gaps={"MEX": MEXICO_VINTAGE_NOTE, "ESP": SPAIN_VINTAGE_NOTE},
    )


__all__ = [
    "MEXICO_VINTAGE_NOTE",
    "SPAIN_VINTAGE_NOTE",
    "BANXICO_SIE_BASE",
    "INEGI_BIE_BASE",
]
