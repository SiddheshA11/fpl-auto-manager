"""
github_api.py against a real socket.

Every other test hands GitHubRepo a fake opener, so the real urllib transport,
the payload shapes, the base64 and the sha-on-update path were exercised only
by the live ledger self-test. Served from a local http.server here, as
test_notify.py does for Telegram: don't mock what you can run.
"""
from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import github_api


class _Api:
    """A scripted GitHub. `routes` maps (method, path) to a list of (status, body)."""

    def __init__(self):
        self.routes: dict[tuple[str, str], list[tuple[int, dict]]] = {}
        self.requests: list[dict] = []

    def on(self, method, path, *responses):
        self.routes[(method, path)] = list(responses)


@pytest.fixture
def api():
    state = _Api()

    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            state.requests.append({
                "method": self.command, "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "json": json.loads(raw) if raw else None,
            })
            queue = state.routes.get((self.command, self.path))
            status, body = queue.pop(0) if queue else (404, {"message": "Not Found"})
            payload = json.dumps(body).encode() if body is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PATCH = do_PUT = _handle

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.repo = github_api.GitHubRepo(
        "o/r", "TOKEN", api_base=f"http://127.0.0.1:{server.server_port}")
    yield state
    server.shutdown()
    thread.join(timeout=5)


VAR = f"/repos/o/r/actions/variables/{github_api.MARKER_VARIABLE}"


def test_every_call_carries_the_token_and_api_version(api):
    api.on("GET", VAR, (200, {"value": "7"}))
    api.repo.get_variable(github_api.MARKER_VARIABLE)
    h = api.requests[0]["headers"]  # urllib re-cases header names
    assert h["authorization"] == "Bearer TOKEN"
    assert h["accept"] == "application/vnd.github+json"
    assert h["x-github-api-version"] == "2022-11-28"


def test_a_variable_reads_its_value(api):
    api.on("GET", VAR, (200, {"name": github_api.MARKER_VARIABLE, "value": "7"}))
    assert api.repo.read_variable(github_api.MARKER_VARIABLE) == (True, "7")


def test_an_absent_variable_is_known_to_be_absent(api):
    assert api.repo.read_variable(github_api.MARKER_VARIABLE) == (True, None)


@pytest.mark.parametrize("status", [500, 502, 403, 401])
def test_a_failed_read_is_not_mistaken_for_an_absent_variable(api, status):
    """get_variable reports both as None; read_variable must not."""
    api.on("GET", VAR, (status, {"message": "boom"}))
    assert api.repo.read_variable(github_api.MARKER_VARIABLE) == (False, None)


def test_an_unreachable_api_is_unknown_not_absent():
    repo = github_api.GitHubRepo("o/r", "T", api_base="http://127.0.0.1:9")
    assert repo.read_variable(github_api.MARKER_VARIABLE) == (False, None)


def test_set_variable_patches_then_creates_when_missing(api):
    api.on("PATCH", VAR, (404, {"message": "Not Found"}))
    api.on("POST", "/repos/o/r/actions/variables", (201, {}))
    assert api.repo.set_variable(github_api.MARKER_VARIABLE, "8") is True
    patch, post = api.requests
    assert patch["method"] == "PATCH" and post["method"] == "POST"
    assert post["json"] == {"name": github_api.MARKER_VARIABLE, "value": "8"}


def test_set_variable_reports_a_refused_write(api):
    api.on("PATCH", VAR, (403, {"message": "Resource not accessible"}))
    assert api.repo.set_variable(github_api.MARKER_VARIABLE, "8") is False


def test_a_new_file_is_created_without_a_sha_and_base64_encoded(api):
    body = b'{"event": 8}\n'
    api.on("PUT", "/repos/o/r/contents/predictions/gw08.json", (201, {}))
    assert api.repo.put_file("predictions/gw08.json", body, "predictions", "msg") is True
    get, put = api.requests
    assert get["path"] == "/repos/o/r/contents/predictions/gw08.json?ref=predictions"
    assert "sha" not in put["json"]
    assert base64.b64decode(put["json"]["content"]) == body
    assert put["json"]["branch"] == "predictions" and put["json"]["message"] == "msg"


def test_an_existing_file_is_updated_with_its_sha(api):
    """Without the sha the Contents API answers 409, so every GW after the first would fail."""
    api.on("GET", "/repos/o/r/contents/predictions/gw08.json?ref=predictions",
           (200, {"content": "e30=", "sha": "abc123"}))
    api.on("PUT", "/repos/o/r/contents/predictions/gw08.json", (200, {}))
    assert api.repo.put_file("predictions/gw08.json", b"{}", "predictions", "m") is True
    assert api.requests[1]["json"]["sha"] == "abc123"


def test_a_refused_put_returns_false(api):
    api.on("PUT", "/repos/o/r/contents/p.json", (409, {"message": "sha mismatch"}))
    assert api.repo.put_file("p.json", b"{}", "predictions", "m") is False


def test_ensure_branch_creates_it_at_the_tip_of_main(api):
    api.on("GET", "/repos/o/r/git/ref/heads/main", (200, {"object": {"sha": "deadbeef"}}))
    api.on("POST", "/repos/o/r/git/refs", (201, {}))
    assert api.repo.ensure_branch("predictions") is True
    assert api.requests[-1]["json"] == {"ref": "refs/heads/predictions", "sha": "deadbeef"}


def test_ensure_branch_leaves_an_existing_branch_alone(api):
    api.on("GET", "/repos/o/r/git/ref/heads/predictions", (200, {"object": {"sha": "x"}}))
    assert api.repo.ensure_branch("predictions") is True
    assert [r["method"] for r in api.requests] == ["GET"]


def test_dispatch_sends_ref_and_inputs(api):
    api.on("POST", "/repos/o/r/actions/workflows/weekly_run.yml/dispatches", (204, None))
    assert api.repo.dispatch_workflow("weekly_run.yml", inputs={"dry_run": "false"}) is True
    assert api.requests[0]["json"] == {"ref": "main", "inputs": {"dry_run": "false"}}
