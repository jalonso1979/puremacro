def test_inference_public_imports():
    # Robust against cross-test namespace pollution: importing an inference
    # *submodule* elsewhere in the suite (pkgutil sweeps / transitive imports)
    # rebinds a package attribute from the re-exported function to the submodule
    # (e.g. ``puremacro.inference.newey_west`` -> the module), which surfaces
    # order-dependently. Reload so we verify the canonical ``__init__`` exports
    # regardless of suite ordering.
    import importlib

    import puremacro.inference
    importlib.reload(puremacro.inference)
    from puremacro.inference import (
        newey_west,
        block_bootstrap,
        moving_block_bootstrap,
        wild_bootstrap,
        wild_bootstrap_var,
        cragg_donald_f,
        kleibergen_paap_f,
        pesaran_cd,
        swamy_test,
    )
    for fn in (newey_west, block_bootstrap, moving_block_bootstrap,
               wild_bootstrap, wild_bootstrap_var,
               cragg_donald_f, kleibergen_paap_f,
               pesaran_cd, swamy_test):
        assert callable(fn)


def test_weak_iv_module_has_all():
    """weak_iv.py must expose __all__ — added in iteration N+8 step 3."""
    from puremacro.inference import weak_iv

    assert hasattr(weak_iv, "__all__")
    expected = {
        "anderson_rubin_band", "cragg_donald_f", "kleibergen_paap_f",
        "olea_pflueger_f", "anderson_rubin_test", "msw_bands",
        "ARTestResult",
    }
    assert expected.issubset(set(weak_iv.__all__))
