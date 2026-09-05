"""Tests for puremacro.runtime.colab bridge."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from puremacro.runtime.colab import colab_auth_guide, generate_colab_notebook


def test_colab_auth_guide():
    guide = colab_auth_guide()
    assert isinstance(guide, str)
    assert "auth.authenticate_user()" in guide
    assert "Google Colab" in guide


def test_generate_colab_notebook(tmp_path):
    out_file = tmp_path / "test_task.ipynb"
    code = "import puremacro\nprint(puremacro.__version__)"
    
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    arr = np.array([10.0, 20.0, 30.0])

    nb = generate_colab_notebook(
        code=code,
        data_payloads={"my_df": df, "my_arr": arr},
        title="Test Colab Task",
        save_path=out_file,
        mount_drive=True,
    )

    assert out_file.exists()
    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["nbformat"] == 4
    assert len(content["cells"]) >= 4

    # Check that code and data are in notebook cells
    sources = [cell["source"] for cell in content["cells"]]
    flattened = "".join("".join(s) for s in sources)
    assert "puremacro" in flattened
    assert "auth.authenticate_user()" in flattened
    assert "my_df" in flattened
    assert "my_arr" in flattened


def test_colab_auth_snippet():
    from puremacro.runtime.colab import colab_auth_snippet

    snippet = colab_auth_snippet(
        mount_drive=True,
        drive_folder="my_macro_folder",
        require_secrets=["FRED_API_KEY"],
    )
    assert "auth.authenticate_user()" in snippet
    assert "drive.mount('/content/drive')" in snippet
    assert "my_macro_folder" in snippet
    assert "FRED_API_KEY" in snippet
    assert "userdata.get('FRED_API_KEY')" in snippet


def test_colab_badge():
    from puremacro.runtime.colab import colab_badge

    badge1 = colab_badge("https://github.com/puremacro/puremacro/blob/main/notebooks/demo.ipynb")
    assert "[![Open In Colab]" in badge1
    assert "https://colab.research.google.com/github/puremacro/puremacro/blob/main/notebooks/demo.ipynb" in badge1

    badge2 = colab_badge("puremacro/puremacro/blob/main/demo.ipynb")
    assert "https://colab.research.google.com/github/puremacro/puremacro/blob/main/demo.ipynb" in badge2


def test_colab_result_roundtrip(tmp_path):
    from puremacro.runtime.colab import load_colab_result
    from puremacro.runtime.store import save_frame

    df = pd.DataFrame({"x": [10.0, 20.0, 30.0], "y": [1.0, 2.0, 3.0]})
    res_path = tmp_path / "colab_results.pmz"
    save_frame(df, res_path)

    loaded = load_colab_result(res_path)
    assert isinstance(loaded, pd.DataFrame)
    pd.testing.assert_frame_equal(loaded, df)

    # Test bytes input
    raw_bytes = res_path.read_bytes()
    loaded_bytes = load_colab_result(raw_bytes)
    assert isinstance(loaded_bytes, pd.DataFrame)
    pd.testing.assert_frame_equal(loaded_bytes, df)


def test_show_colab_offload_dialog(tmp_path):
    from puremacro.runtime.colab import show_colab_offload_dialog

    dummy_nb = tmp_path / "offloaded.ipynb"
    dummy_nb.write_text("{}", encoding="utf-8")

    res = show_colab_offload_dialog(dummy_nb, title="Test Offload Dialog")
    if hasattr(res, "data"):
        assert "Test Offload Dialog" in res.data
        assert "offloaded.ipynb" in res.data
    else:
        assert "Test Offload Dialog" in str(res)
