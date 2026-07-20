"""Tests confirming puremacro._http is the canonical home of HTTP helpers,
and that the legacy narrative.sources._http path keeps working as a shim."""
from __future__ import annotations

import pytest


def test_top_level_http_module_imports():
    """puremacro._http must expose the 5 canonical names."""
    from puremacro._http import (
        safe_get_bytes, safe_get_text, safe_get_json,
        USER_AGENT, DEFAULT_TIMEOUT,
    )
    assert callable(safe_get_bytes)
    assert callable(safe_get_text)
    assert callable(safe_get_json)
    assert isinstance(USER_AGENT, str)
    assert isinstance(DEFAULT_TIMEOUT, float)


def test_legacy_narrative_path_still_imports():
    """puremacro.narrative.sources._http must remain importable."""
    from puremacro.narrative.sources._http import (
        safe_get_bytes, safe_get_text, safe_get_json,
        USER_AGENT, DEFAULT_TIMEOUT,
    )
    assert callable(safe_get_bytes)


def test_legacy_path_returns_same_objects():
    """The shim must re-export the same function objects, not redefine."""
    from puremacro._http import safe_get_bytes as canonical
    from puremacro.narrative.sources._http import safe_get_bytes as shim
    assert canonical is shim


def test_user_agent_override_kwarg_present():
    """Verify the keyword-only `user_agent=` override survives the move."""
    import inspect
    from puremacro._http import safe_get_bytes
    sig = inspect.signature(safe_get_bytes)
    params = sig.parameters
    assert "user_agent" in params
    assert params["user_agent"].kind == inspect.Parameter.KEYWORD_ONLY
