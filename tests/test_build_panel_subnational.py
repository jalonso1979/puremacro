import pytest
from unittest.mock import patch

from puremacro.build_panel import build_all_subnational

def test_build_all_subnational_default_args():
    """Test build_all_subnational with default arguments."""
    with patch("puremacro.build_panel._build_subnational") as mock_build:
        mock_build.return_value = {"state": "path1", "county": "path2"}

        result = build_all_subnational()

        mock_build.assert_called_once_with(data_dir="data", refresh=False)
        assert result == {"state": "path1", "county": "path2"}

def test_build_all_subnational_custom_args():
    """Test build_all_subnational with custom arguments."""
    with patch("puremacro.build_panel._build_subnational") as mock_build:
        mock_build.return_value = {"state": "path3", "county": "path4"}

        result = build_all_subnational(refresh=True, data_dir="/custom/dir")

        mock_build.assert_called_once_with(data_dir="/custom/dir", refresh=True)
        assert result == {"state": "path3", "county": "path4"}

def test_build_all_subnational_error():
    """Test build_all_subnational when underlying function raises an exception."""
    with patch("puremacro.build_panel._build_subnational") as mock_build:
        mock_build.side_effect = RuntimeError("Failed to build subnational panel")

        with pytest.raises(RuntimeError, match="Failed to build subnational panel"):
            build_all_subnational()
