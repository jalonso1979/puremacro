# tests/test_local_http_post.py
"""post_json: urllib JSON POST used by the local-LLM HTTP engine."""
import http.server
import json
import threading

import pytest

from puremacro._http import post_json


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        sent = json.loads(self.rfile.read(n) or b"{}")
        out = json.dumps({"echo": sent, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def echo_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_post_json_roundtrip(echo_server):
    out = post_json(f"{echo_server}/x", {"a": 1, "b": "hi"}, timeout=5)
    assert out["path"] == "/x"
    assert out["echo"] == {"a": 1, "b": "hi"}
