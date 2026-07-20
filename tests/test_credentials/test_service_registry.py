"""F2.0 — Verify the SERVICES registry shape."""
from __future__ import annotations


def test_services_registry_has_expected_keys():
    from puremacro.credentials import SERVICES

    expected = {"fred", "bea", "anthropic", "openai", "census"}
    assert expected.issubset(set(SERVICES.keys())), (
        f"missing services: {expected - set(SERVICES.keys())}"
    )


def test_every_service_has_required_fields():
    from puremacro.credentials import SERVICES, ServiceCredentialSpec

    for name, spec in SERVICES.items():
        assert isinstance(spec, ServiceCredentialSpec), name
        assert spec.name == name, f"{name}: spec.name mismatch ({spec.name!r})"
        assert isinstance(spec.env_vars, tuple) and len(spec.env_vars) >= 1, name
        assert all(isinstance(v, str) and v for v in spec.env_vars), name
        assert spec.signup_url.startswith("https://"), f"{name}: insecure URL"
        assert spec.description and isinstance(spec.description, str), name


def test_known_env_var_aliases():
    from puremacro.credentials import SERVICES

    # Pin specific aliases — researchers' shells are full of these names.
    assert "FRED_API_KEY" in SERVICES["fred"].env_vars
    assert "BEA_API_KEY" in SERVICES["bea"].env_vars
    assert "ANTHROPIC_API_KEY" in SERVICES["anthropic"].env_vars
    assert "OPENAI_API_KEY" in SERVICES["openai"].env_vars
    assert "CENSUS_API_KEY" in SERVICES["census"].env_vars


def test_puremacro_version_is_wellformed():
    import re

    import puremacro
    assert isinstance(puremacro.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", puremacro.__version__), puremacro.__version__
