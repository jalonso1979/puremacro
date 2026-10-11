import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from puremacro.build_subnational_panel import attach_structural_covariates, _compute_county_structural

def test_attach_structural_covariates():
    # Setup test data
    county_df = pd.DataFrame({
        "county_fips": ["01001", "06037", "99999"],
        "state_fips": ["01", "06", "99"],  # 01 is RTW, 06 is not, 99 is invalid/not RTW
        "other_col": [1, 2, 3]
    })

    shares = pd.DataFrame({
        "county_fips": ["01001", "01001", "06037", "06037"],
        "naics": ["3111", "5411", "2121", "4411"], # 31 is manuf/tradable, 54 is neither, 21 is tradable, 44 is neither
        "share": [0.4, 0.6, 0.3, 0.7]
    })

    result = attach_structural_covariates(county_df, shares)

    # Assert structural merges worked
    np.testing.assert_allclose(result.loc[result["county_fips"] == "01001", "tradable_share_c"].iloc[0], 0.4)
    np.testing.assert_allclose(result.loc[result["county_fips"] == "01001", "manuf_share_c"].iloc[0], 0.4)

    np.testing.assert_allclose(result.loc[result["county_fips"] == "06037", "tradable_share_c"].iloc[0], 0.3)
    np.testing.assert_allclose(result.loc[result["county_fips"] == "06037", "manuf_share_c"].iloc[0], 0.0)

    # Assert RTW worked (01 is RTW, 06 is not, 99 is not)
    assert result.loc[result["county_fips"] == "01001", "right_to_work_state"].iloc[0] == 1
    assert result.loc[result["county_fips"] == "06037", "right_to_work_state"].iloc[0] == 0
    assert result.loc[result["county_fips"] == "99999", "right_to_work_state"].iloc[0] == 0

    # Assert missing structural defaults to 0.0
    np.testing.assert_allclose(result.loc[result["county_fips"] == "99999", "tradable_share_c"].iloc[0], 0.0)
    np.testing.assert_allclose(result.loc[result["county_fips"] == "99999", "manuf_share_c"].iloc[0], 0.0)

    # Ensure original column is kept
    assert "other_col" in result.columns

def test_compute_county_structural():
    shares = pd.DataFrame({
        "county_fips": ["01001", "01001", "01001"],
        "naics": ["3111", "3311", "2111"], # 31 and 33 are manuf, 21 is tradable but not manuf
        "share": [0.2, 0.1, 0.3]
    })

    result = _compute_county_structural(shares)

    np.testing.assert_allclose(result.loc[result["county_fips"] == "01001", "tradable_share_c"].iloc[0], 0.6) # 0.2 + 0.1 + 0.3
    np.testing.assert_allclose(result.loc[result["county_fips"] == "01001", "manuf_share_c"].iloc[0], 0.3) # 0.2 + 0.1
