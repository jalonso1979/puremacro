from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from puremacro.models.nested_dmp import backend as bk


def test_numpy_always_available_and_is_namespace():
    assert bk.backend_available("numpy") is True
    xp = bk.get_array_namespace("numpy")
    assert xp is np


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        bk.backend_available("tensorflow")
    with pytest.raises(ValueError, match="Unknown backend"):
        bk.get_array_namespace("tensorflow")


def test_numba_is_not_a_namespace():
    # numba uses compiled kernels, not an array module.
    with pytest.raises(ValueError, match="compiled kernels"):
        bk.get_array_namespace("numba")


def test_to_numpy_roundtrips_numpy():
    a = np.arange(5.0)
    np.testing.assert_array_equal(bk.to_numpy(a), a)


def test_available_backends_includes_numpy():
    avail = bk.available_backends()
    assert "numpy" in avail
    assert set(avail).issubset({"numpy", "numba", "mlx", "cupy"})


@pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None, reason="mlx not installed"
)
def test_mlx_namespace_and_roundtrip():
    mx = bk.get_array_namespace("mlx")
    a = mx.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(bk.to_numpy(a), [1.0, 2.0, 3.0])
