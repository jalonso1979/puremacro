"""Live checks of the microdata providers. Opt-in: ``pytest -m network``.

The SCF check is a real oracle: the Board's 2022 Bulletin reports median
family net worth of $192,900 and mean of $1,063,700 (2022 dollars).
Reproducing both from the summary extract confirms the download, the
implicate handling and the weights in one go.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.network


def test_scf_2022_reproduces_published_net_worth():
    from puremacro.fetch.micro import fetch_scf
    try:
        scf = fetch_scf(2022, ["networth"])
    except OSError as exc:                       # unreachable host: not a code failure
        pytest.skip(f"SCF download failed: {exc}")
    med = scf.quantile("networth", 0.5).iloc[0]
    mean = scf.mean("networth").iloc[0]
    assert med["estimate"] == pytest.approx(192_900, rel=0.01)
    assert mean["estimate"] == pytest.approx(1_063_700, rel=0.01)
    assert 0 < med["se"] < 0.1 * med["estimate"]


def test_acs_pums_small_state_pull():
    from puremacro import credentials
    if not credentials.get("census"):
        pytest.skip("no Census API key configured")
    from puremacro.fetch.micro import fetch_acs_pums
    acs = fetch_acs_pums(2023, ["AGEP", "WAGP", "ADJINC"], geography="state:11")
    assert len(acs) > 1000
    assert len(acs.design.replicate_weights) == 80
    res = acs.mean("AGEP")
    assert 30 < res.loc[0, "estimate"] < 45 and np.isfinite(res.loc[0, "se"])
