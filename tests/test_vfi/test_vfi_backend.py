from __future__ import annotations

import numpy as np
import pytest

from puremacro import _backend as bk


def test_supported_is_four_backends():
    assert bk.SUPPORTED == ("numpy", "numba", "mlx", "cupy")


def test_numpy_always_available_and_is_namespace():
    assert bk.backend_available("numpy") is True
    assert bk.get_array_namespace("numpy") is np


def test_cupy_is_recognized_but_never_errors_when_absent():
    assert "cupy" in bk.SUPPORTED
    _ = bk.backend_available("cupy")  # False on a non-CUDA host; must not raise


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        bk.backend_available("tensorflow")
    with pytest.raises(ValueError, match="Unknown backend"):
        bk.get_array_namespace("tensorflow")


def test_numba_is_not_a_namespace():
    with pytest.raises(ValueError, match="compiled kernels"):
        bk.get_array_namespace("numba")


def test_available_backends_subset_of_supported():
    avail = bk.available_backends()
    assert "numpy" in avail
    assert set(avail).issubset(set(bk.SUPPORTED))


def test_nested_dmp_backend_shim_reexports_shared_objects():
    from puremacro.models.nested_dmp import backend as nbk

    assert nbk.backend_available is bk.backend_available
    assert nbk.get_array_namespace is bk.get_array_namespace
    assert nbk.SUPPORTED == bk.SUPPORTED


def test_to_numpy_roundtrips_numpy():
    a = np.arange(5.0)
    np.testing.assert_array_equal(bk.to_numpy(a), a)


@pytest.mark.skipif(bk.backend_available("cupy"), reason="cupy is installed")
def test_cupy_namespace_raises_importerror_when_absent():
    with pytest.raises(ImportError, match="cupy not installed"):
        bk.get_array_namespace("cupy")
