"""Shared toy datasets for puremacro tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def toy_var2(rng):
    """Stationary 2-variable VAR(1): y_t = A y_{t-1} + u_t."""
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t-1] + L @ rng.standard_normal(n)
    return pd.DataFrame(Y, columns=["y1", "y2"])


@pytest.fixture
def toy_panel(rng):
    """Toy panel: 5 entities x 60 periods, single LP-friendly outcome."""
    rows = []
    for c in ("A", "B", "C", "D", "E"):
        x = rng.standard_normal(60)
        y = 0.3 * np.roll(x, 1) + 0.5 * rng.standard_normal(60)
        for t, (xv, yv) in enumerate(zip(x, y)):
            rows.append({"code": c, "t": t, "x": xv, "y": yv})
    df = pd.DataFrame(rows)
    return df.set_index(["code", "t"]).sort_index()


# ---------------------------------------------------------------------------
# Shared HTTP mock for CB-connector tests (Slice 1 + Slice 3)
# ---------------------------------------------------------------------------
import importlib


# Modules whose `safe_get_bytes` / `safe_get_text` we patch in offline
# CB-connector tests. Slice 1 + Slice 3 banks combined.
_CB_PATCH_TARGETS = [
    "puremacro.narrative.sources._rss",
    "puremacro.narrative.sources._ratedoc",
    "puremacro.narrative.sources._speeches",
    "puremacro.narrative.sources.fed_decision",
    "puremacro.narrative.sources.fed_minutes",
    "puremacro.narrative.sources.fed_press_conf",
    "puremacro.narrative.sources.fed_speeches",
    "puremacro.narrative.sources.ecb_decision",
    "puremacro.narrative.sources.ecb_minutes",
    "puremacro.narrative.sources.ecb_press_conf",
    "puremacro.narrative.sources.ecb_speeches",
    "puremacro.narrative.sources.boe_decision",
    "puremacro.narrative.sources.boe_minutes",
    "puremacro.narrative.sources.boe_speeches",
    "puremacro.narrative.sources.boj_decision",
    "puremacro.narrative.sources.boj_speeches",
    # Slice 3
    "puremacro.narrative.sources.banxico",
    "puremacro.narrative.sources.bcb",
    "puremacro.narrative.sources.bccl",
    "puremacro.narrative.sources.bcra",
    "puremacro.narrative.sources.banrep",
    "puremacro.narrative.sources.rba",
    "puremacro.narrative.sources.rbnz",
    "puremacro.narrative.sources.riksbank",
    "puremacro.narrative.sources.norges",
    "puremacro.narrative.sources.sarb",
    "puremacro.narrative.sources.pboc",
    "puremacro.narrative.sources.rbi",
    "puremacro.narrative.sources.bok",
    "puremacro.narrative.sources.mas",
    "puremacro.narrative.sources.bot",
    "puremacro.narrative.sources.bis_speeches",
    # BLS state-panel fetchers (Task 2 — Notebook 29)
    "puremacro.fetch.bls_state_panel",
    # Beige Book connector (Stream 4 — BBUI district-level)
    "puremacro.narrative.sources.beige_book",
]


@pytest.fixture
def mock_http(monkeypatch):
    """Per-test in-memory HTTP mock. Use ``register(bytes_=..., text=...)``
    to register URL → payload mappings before invoking the connector.
    Unregistered URLs raise ``LookupError`` (loud failure on missing mocks).
    """
    by_url_bytes: dict[str, bytes] = {}
    by_url_text: dict[str, str] = {}

    def _fake_bytes(url, **_kw):
        if url in by_url_bytes:
            return by_url_bytes[url]
        raise LookupError(f"mock_http: no bytes registered for {url}")

    def _fake_text(url, **_kw):
        if url in by_url_text:
            return by_url_text[url]
        raise LookupError(f"mock_http: no text registered for {url}")

    for modname in _CB_PATCH_TARGETS:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, "safe_get_bytes"):
            monkeypatch.setattr(mod, "safe_get_bytes", _fake_bytes)
        if hasattr(mod, "safe_get_text"):
            monkeypatch.setattr(mod, "safe_get_text", _fake_text)

    def register(*, bytes_=None, text=None):
        if bytes_:
            by_url_bytes.update(bytes_)
        if text:
            by_url_text.update(text)

    return register
