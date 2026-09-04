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
