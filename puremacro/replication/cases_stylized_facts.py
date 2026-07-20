"""Replication cases — macro stylized facts reproduced from vendored FRED data.

Okun (1962): output and unemployment move strongly in opposite directions. We
read a frozen FRED snapshot (industrial production + unemployment) and check the
year-over-year IP growth is strongly *negatively* correlated with the
year-over-year change in unemployment. Offline + deterministic (the snapshot is a
fixed vintage; the FRED key is used only by the dev generator).
"""
from __future__ import annotations

import numpy as np

from ._model import ReplicationCase, TargetKind, Tol


def _okun_neg_corr() -> dict:
    from ._data import load_csv

    d = load_csv("okun").dropna()
    ip = d["indpro"].to_numpy(dtype=float)
    ur = d["unrate"].to_numpy(dtype=float)
    ip_growth = 100.0 * (np.log(ip[12:]) - np.log(ip[:-12]))   # YoY IP growth
    d_unemp = ur[12:] - ur[:-12]                               # YoY change in unemployment
    corr = float(np.corrcoef(ip_growth, d_unemp)[0, 1])
    # CORRELATION kind = "metric >= threshold"; -corr is large+positive iff corr is
    # strongly negative (Okun's law).
    return {"neg_okun_corr": -corr}


CASES: list[ReplicationCase] = [
    ReplicationCase(
        id="stylized_facts.okun_law_fred",
        family="stylized_facts",
        paper="Okun (1962), 'Potential GNP: Its Measurement and Significance', ASA Proceedings of the Business and Economics Statistics Section",
        title="Okun's law: output growth and unemployment change move strongly opposite",
        title_es="Ley de Okun: el crecimiento del producto y el cambio del desempleo se mueven fuertemente en sentido opuesto",
        source="snapshot:okun.csv",
        estimate=_okun_neg_corr,
        target={"neg_okun_corr": 0.5},
        target_kind=TargetKind.CORRELATION,
        tol=Tol.COARSE,
        citation="Okun (1962): the negative output-unemployment relationship; on FRED INDPRO/UNRATE, corr(IP growth, dUnemp) is about -0.77.",
    ),
]
