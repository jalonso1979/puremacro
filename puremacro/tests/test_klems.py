import pandas as pd
import numpy as np

from puremacro.klems import load_klems_panel, _compute_p_equip_index

def test_compute_p_equip_index():
    # Setup test dataframe
    df = pd.DataFrame({
        'Ip_OMach': [1.0, 1.2, np.nan],
        'Ip_TraEq': [1.0, 1.1, np.nan],
        'K_OMach': [100.0, 150.0, 0.0],
        'K_TraEq': [50.0, 50.0, 0.0]
    })

    ip_cols = ('Ip_OMach', 'Ip_TraEq')
    k_cols = ('K_OMach', 'K_TraEq')

    result = _compute_p_equip_index(df, ip_cols, k_cols)

    assert isinstance(result, pd.Series)
    assert len(result) == 3

    # 1st row: log(1)*100/150 + log(1)*50/150 = 0 -> exp(0) = 1.0
    assert np.isclose(result[0], 1.0)

    # 3rd row should be NaN
    assert pd.isna(result[2])

def test_load_klems_panel_empty_cache(tmp_path):
    # Should gracefully return empty dataframe if cache missing
    df = load_klems_panel(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
