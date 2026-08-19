"""
Tests for refresh-token rotation.

Rotation is what makes the bot run unattended. FPL's auth server invalidates a
refresh token the moment it issues a replacement, so a run that gets a new
token and fails to store it will itself succeed while locking out every run
after it. That failure is invisible for a week, then presents as "the bot
stopped working" — which is exactly how the previous season's automation died.
"""
from __future__ import annotations

import logging

import pytest

import fpl_client
from fpl_client import FPLClient


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(fpl_client, "FPL_REFRESH_TOKEN", "old-token")
    monkeypatch.setattr(fpl_client, "FPL_TEAM_ID", 123)
    c = FPLClient()
    # The post-refresh validation call should look successful.
    monkeypatch.setattr(c.session, "get", lambda *a, **k: _Response(200))
    return c


def _token_response(monkeypatch, *, new_token="new-token"):
    payload = {"access_token": "access", "refresh_token": new_token}
    monkeypatch.setattr(fpl_client.requests, "post", lambda *a, **k: _Response(200, payload))


def test_successful_rotation_persists_the_new_token(client, monkeypatch):
    _token_response(monkeypatch)
    saved = {}

    def _save(name, value):
        saved[name] = value
        return True

    monkeypatch.setattr(client, "_update_github_secret", _save)

    assert client._refresh_access_token() is True
    assert saved == {"FPL_REFRESH_TOKEN": "new-token"}
    assert client.rotation_failed is False


def test_failed_rotation_is_flagged_and_logged_critical(client, monkeypatch, caplog):
    """
    Regression: the return value of _update_github_secret was discarded, so a
    failure to save the rotated token passed silently. The old token is already
    spent at that point, so this must be loud.
    """
    _token_response(monkeypatch)
    monkeypatch.setattr(client, "_update_github_secret", lambda name, value: False)

    with caplog.at_level(logging.CRITICAL, logger="fpl_auto"):
        result = client._refresh_access_token()

    # The run itself still authenticated - it holds a valid access token.
    assert result is True
    assert client.rotation_failed is True, "caller must be able to see rotation failed"
    assert any(r.levelno >= logging.CRITICAL for r in caplog.records), "must log at CRITICAL"
    assert "ROTATION FAILED" in caplog.text


def test_no_rotation_offered_is_not_a_failure(client, monkeypatch):
    """If the server returns no new refresh token, the existing one still works."""
    _token_response(monkeypatch, new_token=None)
    monkeypatch.setattr(
        client, "_update_github_secret",
        lambda *a: pytest.fail("must not try to save when no new token was issued"),
    )

    assert client._refresh_access_token() is True
    assert client.rotation_failed is False


def test_rotation_requires_a_github_token(monkeypatch, caplog):
    """
    Missing GH_PAT must be an error, not a warning. deadline_check.yml shipped
    without it, so that run would consume a token and be unable to save the
    replacement.
    """
    monkeypatch.delenv("GH_PAT", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    with caplog.at_level(logging.ERROR, logger="fpl_auto"):
        assert FPLClient()._update_github_secret("FPL_REFRESH_TOKEN", "x") is False
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_every_outbound_call_sets_a_timeout(monkeypatch):
    """A hung request in an unattended run holds the job open past the deadline."""
    monkeypatch.setenv("GH_PAT", "pat")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    seen = []

    def _get(*args, **kwargs):
        seen.append(kwargs.get("timeout"))
        return _Response(404, text="nope")

    monkeypatch.setattr(fpl_client.requests, "get", _get)

    FPLClient()._update_github_secret("FPL_REFRESH_TOKEN", "x")
    assert seen and all(t is not None for t in seen), "requests must pass a timeout"
