"""CI-only drift-guard: recompute the LIVE esda (PySAL) reference and assert the
frozen golden in goldens/spatial.json is still faithful. This is the single
place esda is touched for the spatial subsystem. Run via:
    pytest -m reference
"""
import numpy as np
import pytest

pytestmark = pytest.mark.reference

esda = pytest.importorskip("esda")
libpysal = pytest.importorskip("libpysal")


def test_spatial_goldens_match_live_esda():
    from tools.gen_validation_goldens_spatial import _live_esda_stats

    from puremacro.validation._goldens import load_golden

    live = _live_esda_stats()
    # If this fails, esda changed its output — regenerate the golden:
    #   python tools/gen_validation_goldens_spatial.py
    for case, stats in live.items():
        golden = load_golden(f"spatial:{case}")
        for key, values in stats.items():
            np.testing.assert_allclose(np.asarray(values), np.asarray(golden[key]), rtol=1e-8, atol=1e-12)
