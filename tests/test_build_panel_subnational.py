import pytest
from unittest.mock import patch

from puremacro.build_panel import build_all_subnational

@patch("puremacro.build_panel._build_subnational")
def test_build_all_subnational(mock_build_subnational):
    mock_build_subnational.return_value = {"state": "path/to/state", "county": "path/to/county", "industry": "path/to/industry"}

    result = build_all_subnational(refresh=True, data_dir="test_data")

    mock_build_subnational.assert_called_once_with(data_dir="test_data", refresh=True)
    assert result == {"state": "path/to/state", "county": "path/to/county", "industry": "path/to/industry"}

@patch("puremacro.build_panel._build_subnational")
def test_build_all_subnational_defaults(mock_build_subnational):
    mock_build_subnational.return_value = {"state": "path/to/state"}

    result = build_all_subnational()

    mock_build_subnational.assert_called_once_with(data_dir="data", refresh=False)
    assert result == {"state": "path/to/state"}
