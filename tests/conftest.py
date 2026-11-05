"""Shared toy datasets for puremacro tests."""
import uuid

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_llm_cache(monkeypatch, tmp_path_factory):
    """Keep every test off the user's real LLM caches, mode, model and server.

    There are two caches in two files. ``puremacro.llm`` reads and writes the
    completion cache ``$PUREMACRO_LLM_CACHE`` (default
    ``~/.cache/puremacro/llm_cache.sqlite``); ``llm_prob_kernel`` reads and
    writes the narrative score cache ``$PUREMACRO_LLM_SCORES_CACHE`` (default
    ``~/.cache/puremacro/llm_scores.sqlite``). Each gets a file of its own,
    different for every test. Without this a test that completes a prompt or
    scores a document would leave rows in a real cache, and a cache hit left
    there by an earlier run would stand in for the engine a later test means
    to exercise.

    Both variables are set whatever the shell exports. A developer who points
    one at a cache worth keeping (to build one to ship, say) would otherwise
    get a test's fake completions or scores in it, and later tests would find
    them there. ``monkeypatch`` puts the developer's values back when the test
    ends.

    The files sit under the session's base temp directory but not in
    ``tmp_path``. ``tmp_path`` makes a numbered directory for every test, and
    pytest scans the whole base directory for the next number each time, so
    asking for it here would make the cost of a run grow with the square of
    the number of tests. A random name keeps each test's files its own, and no
    directory is made: both caches create their parent directory when they
    first open their file (``CompletionCache`` in ``_connect``, the score
    kernel just before it connects), so only a test that uses a cache touches
    the disk.

    The other three variables are removed: ``PUREMACRO_LLM_MODE`` (auto, cache
    or off), ``PUREMACRO_LLM_MODEL`` (what the ``default`` model means) and
    ``PUREMACRO_LLM_BASE_URL`` (the server the HTTP engines talk to). A test
    that wants one sets it with ``monkeypatch.setenv``; a value exported in the
    developer's shell would otherwise change which model, engine or server a
    test exercises.
    """
    base = tmp_path_factory.getbasetemp() / "llm_cache"
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(base / f"{uuid.uuid4().hex}.sqlite"))
    monkeypatch.setenv("PUREMACRO_LLM_SCORES_CACHE", str(base / f"{uuid.uuid4().hex}_scores.sqlite"))
    monkeypatch.delenv("PUREMACRO_LLM_FALLBACK_CACHE", raising=False)
    monkeypatch.delenv("PUREMACRO_LLM_MODE", raising=False)
    monkeypatch.delenv("PUREMACRO_LLM_MODEL", raising=False)
    monkeypatch.delenv("PUREMACRO_LLM_BASE_URL", raising=False)


# ---------------------------------------------------------------------------
# Default network and credentials isolation
# ---------------------------------------------------------------------------
import ipaddress
import socket
import sys

#: Tests carrying one of these marks may reach the network and see the
#: developer's real credentials. All other tests get neither.
_NETWORK_MARKS = ("network", "replication", "local_llm")

#: ``(nodeid, call, host)`` for every attempt the socket guard refused this
#: session; listed in the terminal summary so a test whose library swallowed
#: the error still shows up.
_SOCKET_GUARD_HITS: list[tuple[str, str, str]] = []

_LOCAL_NAMES = {"", "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


class SocketGuardError(OSError):
    """A test that is not marked network tried to connect off this machine."""


class SocketGuardResolveError(socket.gaierror, SocketGuardError):
    """A test that is not marked network tried to resolve a public host name."""


def _is_local_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    host = str(host).strip().lower().rstrip(".")
    if host in _LOCAL_NAMES or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0].strip("[]"))
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def _guard_message(call: str, host, nodeid: str) -> str:
    return (f"socket guard: {call}({host!r}) blocked in {nodeid}. Tests run offline "
            "by default; patch the fetch, or mark the test "
            "@pytest.mark.network if it really needs the internet.")


@pytest.fixture(autouse=True)
def _default_offline(request, monkeypatch, tmp_path_factory):
    """Keep every unmarked test off the internet and off the developer's keys.

    Socket guard. ``socket.socket.connect``/``connect_ex`` to an address that is
    not loopback, and ``socket.getaddrinfo`` of a host that is not local, raise
    :class:`SocketGuardError` (an ``OSError``; a ``socket.gaierror`` for the
    lookup) for the length of the test. Loopback (127.0.0.0/8, ::1,
    ``localhost``) and unix-domain sockets are untouched, so the stub LLM
    servers and anything else on 127.0.0.1 keep working. Because the error is
    an ``OSError``, library code that degrades gracefully on a network failure
    still does so; every refused attempt is also recorded and listed at the end
    of the run, so a test whose library swallowed the error is still visible.

    Credentials. ``PUREMACRO_CREDENTIALS_FILE`` points at a file that does not
    exist, the config that ``puremacro.credentials`` caches in ``_CONFIG_CACHE``
    is dropped (only if the module is already imported; importing it here would
    change what tests of import side effects see), and every variable in
    ``credentials.SERVICES`` and ``credentials.PASSWORD_ENV_VARS`` is removed.
    Without this a key in the developer's ``~/.puremacro/credentials.toml`` or
    shell reaches tests, and a config cached by one test decides what a later
    test's ``default_config_path`` patch can change.

    Tests marked ``network``, ``replication`` or ``local_llm`` get neither
    guard: they talk to real services with the developer's real keys.
    """
    if any(request.node.get_closest_marker(m) is not None for m in _NETWORK_MARKS):
        return

    nodeid = request.node.nodeid
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _check_address(sock, address, call):
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_local_host(host):
            _SOCKET_GUARD_HITS.append((nodeid, call, str(host)))
            raise SocketGuardError(_guard_message(call, host, nodeid))

    def guarded_connect(self, address):
        _check_address(self, address, "connect")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check_address(self, address, "connect_ex")
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_local_host(host):
            _SOCKET_GUARD_HITS.append((nodeid, "getaddrinfo", str(host)))
            raise SocketGuardResolveError(socket.EAI_NONAME,
                                          _guard_message("getaddrinfo", host, nodeid))
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    # Credentials isolation (see the docstring).
    base = tmp_path_factory.getbasetemp() / "credentials"
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE",
                       str(base / f"{uuid.uuid4().hex}_absent.toml"))
    names = set(_CREDENTIAL_ENV_VARS)
    credentials = sys.modules.get("puremacro.credentials")
    if credentials is not None:
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)
        names.update(v for spec in credentials.SERVICES.values() for v in spec.env_vars)
        names.update(v for vs in credentials.PASSWORD_ENV_VARS.values() for v in vs)
    for name in sorted(names):
        monkeypatch.delenv(name, raising=False)


#: The key variables of ``credentials.SERVICES`` and
#: ``credentials.PASSWORD_ENV_VARS``, for tests that run before
#: ``puremacro.credentials`` is first imported (once it is, the fixture reads
#: the module's own lists as well). tests/test_socket_guard_mechanism.py fails
#: if this falls out of step with the module.
_CREDENTIAL_ENV_VARS = (
    "FRED_API_KEY", "PUREMACRO_FRED_API_KEY",
    "BEA_API_KEY", "PUREMACRO_BEA_API_KEY",
    "ANTHROPIC_API_KEY", "PUREMACRO_ANTHROPIC_API_KEY",
    "OPENAI_API_KEY", "PUREMACRO_OPENAI_API_KEY",
    "CENSUS_API_KEY", "PUREMACRO_CENSUS_API_KEY",
    "BANXICO_API_KEY", "BMX_TOKEN", "PUREMACRO_BANXICO_API_KEY",
    "INEGI_API_KEY", "PUREMACRO_INEGI_API_KEY",
    "BCCH_API_USER", "PUREMACRO_BCCH_API_USER",
    "BCCH_API_PASS", "PUREMACRO_BCCH_API_PASS",
)


def pytest_sessionfinish(session):
    """Hand this worker's socket-guard hits to the pytest-xdist controller.

    Under ``pytest -n`` every worker records its own ``_SOCKET_GUARD_HITS``,
    but only the controller prints the terminal summary. ``workeroutput``
    exists only on a worker's config, and xdist sends it to the controller
    after this hook has run (see ``pytest_testnodedown``).
    """
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput["socket_guard_hits"] = [list(hit) for hit in _SOCKET_GUARD_HITS]


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """On the pytest-xdist controller, keep what a finished worker blocked."""
    hits = getattr(node, "workeroutput", None) or {}
    _SOCKET_GUARD_HITS.extend(tuple(hit) for hit in hits.get("socket_guard_hits", ()))


def pytest_terminal_summary(terminalreporter):
    if not _SOCKET_GUARD_HITS:
        return
    terminalreporter.section("socket guard: network attempts blocked")
    seen = set()
    for nodeid, call, host in _SOCKET_GUARD_HITS:
        if (nodeid, call, host) in seen:
            continue
        seen.add((nodeid, call, host))
        terminalreporter.write_line(f"{nodeid}  {call}  {host}")


# ---------------------------------------------------------------------------
# Parallel runs (pytest-xdist, as CI runs the suite)
# ---------------------------------------------------------------------------
import os

#: The slowest test files, slowest first (local test times of 2026-11-05).
#: CI runs ``pytest -n auto --dist loadfile --no-loadscope-reorder``: each
#: file goes whole to one worker, in collection order, so a long file that is
#: collected late sets the length of the whole run (test_vfi_krusell_smith.py
#: alone takes about 13 minutes on a laptop). Inside a worker these files are
#: moved to the front; only the order in which files start changes, and a
#: serial run is untouched. tests/test_parallel_runs.py fails if one of them
#: is renamed or removed.
_HEAVY_FIRST = (
    "tests/test_vfi/test_vfi_krusell_smith.py",
    "tests/test_trade_tier5_adversarial_probing.py",
    "tests/test_docs_code_blocks.py",
    "tests/test_benchmarks/test_benchmark_continuous_vfi.py",
    "tests/test_notebooks_trade_tutorials.py",
    "tests/test_46_state_strikes_replicates.py",
    "tests/test_spatial/test_models.py",
    "tests/test_smc_inference.py",
    "tests/test_m5_empirical_challenger2_icio.py",
    "tests/test_gar/test_qar_skewt_fci.py",
    "tests/replication/test_replication_cases.py",
    "tests/e2e/test_v290_tier3_e2e.py",
    "tests/test_e2e_upgrade.py",
    "tests/test_vfi/test_vfi_huggett.py",
    "tests/test_fix_MSAR_ms_var_em.py",
    "tests/test_notebooks/test_m5_trade_tutorials_tier5_stress.py",
)
_HEAVY_RANK = {path: rank for rank, path in enumerate(_HEAVY_FIRST)}


def _heavy_first_rank(nodeid: str) -> int:
    """Position of the test's file in ``_HEAVY_FIRST``; every other file ties last."""
    return _HEAVY_RANK.get(nodeid.split("::", 1)[0], len(_HEAVY_FIRST))


def pytest_collection_modifyitems(config, items):
    """Inside a pytest-xdist worker, start the slowest files first."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        items.sort(key=lambda item: _heavy_first_rank(item.nodeid))


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


# ---------------------------------------------------------------------------
# Stub local-LLM servers for the HTTP engine of puremacro.llm
# ---------------------------------------------------------------------------
import http.server
import json
import threading


class _LLMStub:
    """What a test sees of the server of ``llm_stub_server``.

    The server answers on 127.0.0.1 like Ollama and like an OpenAI-compatible
    server: ``GET /api/tags`` and ``GET /v1/models`` list one model,
    ``POST /api/chat`` replies with ``chat_content`` and
    ``POST /v1/chat/completions`` with ``score_content``; any other path gets a
    404. A test changes the attributes to change what the server does.

    Attributes
    ----------
    url : str
        ``http://127.0.0.1:<port>``, with no trailing slash.
    last_body : object
        The decoded JSON body of the last POST; None before the first.
    chat_content, score_content : str or None
        The ``content`` of the two chat replies (None is sent as JSON null).
        They start as ``"[]"`` and ``"0.7"``: an empty list of events and a score.
    probe_reply : tuple of (int, str, bytes) or None
        When set, the ``(status, content type, body)`` both model lists answer
        with instead: a server that is up but is not an LLM server.
    chat_reply : tuple of (int, str, bytes) or None
        When set, the ``(status, content type, body)`` both chat endpoints
        answer with instead: an error such as Ollama's 404 for a model that
        was never pulled.
    delay : float
        Seconds every request waits before it is answered, for a probe that
        has to give up. Teardown releases a request that is still waiting.
    """

    def __init__(self):
        self.url = ""
        self.last_body = None
        self.chat_content = "[]"
        self.score_content = "0.7"
        self.probe_reply = None
        self.chat_reply = None
        self.delay = 0.0
        self.release = threading.Event()


class _StubHTTPServer(http.server.ThreadingHTTPServer):
    """One thread per request, none waited for at teardown, and no traceback
    when a client gives up on a request (a probe that timed out)."""

    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):
        pass


def _llm_stub_handler(stub):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._answer(200, "application/json", json.dumps(obj).encode("utf-8"))

        def do_GET(self):
            stub.release.wait(stub.delay)
            if self.path not in ("/api/tags", "/v1/models"):
                self.send_error(404)
            elif stub.probe_reply is not None:
                self._answer(*stub.probe_reply)
            elif self.path == "/api/tags":
                self._json({"models": [{"name": "qwen3:4b"}]})
            else:
                self._json({"object": "list", "data": [{"id": "qwen3-4b"}]})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            stub.last_body = json.loads(raw) if raw else None
            stub.release.wait(stub.delay)
            if self.path not in ("/api/chat", "/v1/chat/completions"):
                self.send_error(404)
            elif stub.chat_reply is not None:
                self._answer(*stub.chat_reply)
            elif self.path == "/api/chat":
                self._json({"message": {"role": "assistant", "content": stub.chat_content}})
            else:
                self._json({"choices": [{"message": {"role": "assistant",
                                                     "content": stub.score_content}}]})

    return Handler


class _NotFoundHandler(http.server.BaseHTTPRequestHandler):
    """A 404 for every request: an open port that is not an LLM server."""

    def log_message(self, *args):
        pass

    def _not_found(self):
        self.send_error(404)

    do_GET = do_POST = do_HEAD = _not_found


def _serve(handler):
    """Serve ``handler`` on a free port of 127.0.0.1; return the server, its thread, its URL."""
    server = _StubHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def _stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
def llm_stub_server():
    """A stub Ollama and OpenAI-compatible server; ``_LLMStub`` says what it answers."""
    stub = _LLMStub()
    server, thread, stub.url = _serve(_llm_stub_handler(stub))
    try:
        yield stub
    finally:
        stub.release.set()
        _stop(server, thread)


@pytest.fixture
def llm_stub_404_server():
    """The URL of a server that answers every request with a 404."""
    server, thread, url = _serve(_NotFoundHandler)
    try:
        yield url
    finally:
        _stop(server, thread)
