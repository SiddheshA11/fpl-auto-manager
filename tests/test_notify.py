"""
The notifier must be impossible to crash and impossible to silence quietly.

Every test here uses a fake transport. Reaching the real Telegram API from the
suite would be flaky and would leak the token into CI logs on failure.
"""
from __future__ import annotations

import pytest

import notify


class _Resp:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def _recorder(*responses):
    """A transport that returns the given responses in order, recording calls."""
    calls = []
    seq = list(responses)

    def post(url, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})
        r = seq.pop(0) if seq else _Resp(200)
        if isinstance(r, Exception):
            raise r
        return r

    post.calls = calls
    return post


def _notifier(transport):
    slept = []
    n = notify.TelegramNotifier("TOK", "CHAT", transport=transport, sleep=slept.append)
    return n, slept


def test_a_send_hits_the_right_url_and_chat():
    post = _recorder(_Resp(200))
    n, _ = _notifier(post)
    assert n.send("deadline in 3h") is True
    assert len(post.calls) == 1
    assert post.calls[0]["url"] == "https://api.telegram.org/botTOK/sendMessage"
    assert post.calls[0]["json"]["chat_id"] == "CHAT"
    assert post.calls[0]["json"]["text"] == "deadline in 3h"
    assert post.calls[0]["timeout"] == notify.REQUEST_TIMEOUT


def test_a_transient_failure_is_retried_with_backoff():
    post = _recorder(_Resp(500), _Resp(502), _Resp(200))
    n, slept = _notifier(post)
    assert n.send("hello") is True
    assert len(post.calls) == 3
    assert slept == [1.0, 2.0], "backoff should grow, not hammer"


def test_a_network_exception_does_not_escape():
    post = _recorder(ConnectionError("no route"), _Resp(200))
    n, _ = _notifier(post)
    assert n.send("hello") is True
    assert len(post.calls) == 2


def test_it_gives_up_rather_than_looping_forever():
    post = _recorder(_Resp(500), _Resp(500), _Resp(500))
    n, _ = _notifier(post)
    assert n.send("hello") is False
    assert len(post.calls) == notify.MAX_ATTEMPTS


def test_a_bad_token_is_not_retried():
    """401 will not improve on a retry; retrying it just delays the real alert."""
    post = _recorder(_Resp(401, "Unauthorized"))
    n, _ = _notifier(post)
    assert n.send("hello") is False
    assert len(post.calls) == 1


def test_rate_limiting_is_retried_even_though_it_is_4xx():
    post = _recorder(_Resp(429, "Too Many Requests"), _Resp(200))
    n, _ = _notifier(post)
    assert n.send("hello") is True
    assert len(post.calls) == 2


def test_an_over_long_message_is_truncated_not_dropped():
    post = _recorder(_Resp(200))
    n, _ = _notifier(post)
    assert n.send("x" * 9000) is True
    sent = post.calls[0]["json"]["text"]
    assert len(sent) == notify.MAX_MESSAGE_CHARS
    assert sent.endswith("...")


def test_missing_credentials_give_a_null_notifier_not_a_crash():
    n = notify.from_env({})
    assert isinstance(n, notify.NullNotifier)
    assert "TELEGRAM_BOT_TOKEN" in n.reason and "TELEGRAM_CHAT_ID" in n.reason
    assert n.send("anything") is False
    assert n.sent == ["anything"]


def test_blank_credentials_count_as_missing():
    """An unset GitHub secret interpolates to an empty string, not an absent key."""
    n = notify.from_env({"TELEGRAM_BOT_TOKEN": "   ", "TELEGRAM_CHAT_ID": ""})
    assert isinstance(n, notify.NullNotifier)


def test_complete_credentials_give_a_real_notifier():
    """Guards the other direction: a NullNotifier in production alerts nobody."""
    n = notify.from_env({"TELEGRAM_BOT_TOKEN": "TOK", "TELEGRAM_CHAT_ID": "123"})
    assert isinstance(n, notify.TelegramNotifier)


# ------------------------------------------------- the transport, for real
#
# Every test above passes a fake transport, so the real one was never executed
# and `import requests` sat at the top of this module unnoticed. The deadline
# watchdog installs no dependencies, so on its first scheduled run it died with
# ModuleNotFoundError - and its own failure-reporting step died the same way,
# which made the failure silent. Exactly what the watchdog exists to prevent.


def test_the_watchdog_import_graph_needs_nothing_outside_the_standard_library():
    """
    The watchdog installs no dependencies at all, by design - it has to survive
    the failures it reports on, including a broken dependency install. So every
    module it reaches must be stdlib.

    This is the assertion that would have caught the outage: notify.py imported
    `requests`, the watchdog died with ModuleNotFoundError on its first
    scheduled run, and its own failure-reporting step died the same way.
    """
    import ast
    import pathlib
    import sys

    repo = pathlib.Path(notify.__file__).parent
    local = {"deadline_state", "notify", "github_api", "watchdog", "config"}
    offenders = {}

    for name in ("watchdog.py", "notify.py", "deadline_state.py", "github_api.py"):
        imported = set()
        for node in ast.walk(ast.parse((repo / name).read_text())):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        third_party = imported - set(sys.stdlib_module_names) - local
        if third_party:
            offenders[name] = sorted(third_party)

    assert not offenders, (
        f"the watchdog reaches third-party imports {offenders}, which its "
        "workflow does not install; it will die of ModuleNotFoundError on the "
        "runner and take its own failure alert down with it"
    )


def test_the_real_transport_posts_json_and_reads_the_status():
    """
    Runs the actual urllib transport against a real socket. Don't mock what you
    can run - a fake transport is what hid the bug in the first place.
    """
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received["path"] = self.path
            received["json"] = _json.loads(body)
            received["content_type"] = self.headers["Content-Type"]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/botTOK/sendMessage"

    resp = notify._urllib_post(url, {"chat_id": "1", "text": "hi"}, timeout=5)

    assert resp.status_code == 200
    assert received["json"] == {"chat_id": "1", "text": "hi"}
    assert received["content_type"] == "application/json"
    srv.server_close()


def test_the_real_transport_returns_http_errors_rather_than_raising():
    """
    A 401 must come back as a response so the retry policy can tell it from a
    transient 500 and stop instead of hammering.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"Unauthorized")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.handle_request, daemon=True).start()

    resp = notify._urllib_post(f"http://127.0.0.1:{srv.server_port}/x", {"a": 1}, timeout=5)

    assert resp.status_code == 401
    assert "Unauthorized" in resp.text
    srv.server_close()
