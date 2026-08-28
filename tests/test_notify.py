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
