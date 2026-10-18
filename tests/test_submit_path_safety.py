"""
An unreadable submission marker was read as "no marker" and the run went on
to submit. On a dispatch that read is the only idempotency there is.
"""
from __future__ import annotations

import sys

import pytest

import github_api
import manager
from test_exit_codes import _StubClient


class _Repo:
    """A marker that answers with the given (known, value) pairs in turn."""

    def __init__(self, *reads):
        self.reads = list(reads)
        self.read_calls = 0
        self.written = {}

    def read_variable(self, name):
        self.read_calls += 1
        return self.reads.pop(0) if self.reads else (True, None)

    def get_variable(self, name):
        return self.read_variable(name)[1]

    def set_variable(self, name, value):
        self.written[name] = value
        return True

    def ensure_branch(self, *a, **k):
        return True

    def get_file(self, *a):
        return None, None

    def put_file(self, *a):
        return True


class _Sent:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return True


@pytest.fixture
def sent(monkeypatch):
    s = _Sent()
    monkeypatch.setattr(manager.notify, "from_env", lambda: s)
    return s


# ---------------------------------------------------------------- the marker

@pytest.mark.parametrize("reads, expected", [
    ([(True, "7")], 7),
    ([(True, None)], None),
    ([(True, "  ")], None),
    ([(False, None), (True, "7")], 7),
    ([(False, None), (False, None), (True, None)], None),
])
def test_a_marker_github_can_answer_is_read(monkeypatch, reads, expected):
    repo = _Repo(*reads)
    monkeypatch.setattr(manager, "_github", lambda: repo)
    assert manager._marker_gameweek(sleep=lambda s: None) == expected


def test_a_marker_that_stays_unreadable_is_unknown_not_absent(monkeypatch):
    repo = _Repo(*[(False, None)] * 5)
    slept = []
    monkeypatch.setattr(manager, "_github", lambda: repo)
    assert manager._marker_gameweek(sleep=slept.append) is manager.MARKER_UNKNOWN
    assert repo.read_calls == manager.MARKER_READ_ATTEMPTS
    assert slept == [manager.MARKER_RETRY_SECONDS * i
                     for i in range(1, manager.MARKER_READ_ATTEMPTS)], "backoff should grow"


def test_a_marker_that_is_not_a_gameweek_is_unknown(monkeypatch):
    monkeypatch.setattr(manager, "_github", lambda: _Repo((True, "GW7")))
    assert manager._marker_gameweek(sleep=lambda s: None) is manager.MARKER_UNKNOWN


def test_a_local_run_without_credentials_is_not_blocked(monkeypatch):
    monkeypatch.setattr(manager, "_github", lambda: None)
    assert manager._marker_gameweek() is None


def _dispatch(monkeypatch, repo, argv=("manager.py",)):
    stub = _StubClient(lead_hours=5.0)
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    monkeypatch.setattr(manager, "_github", lambda: repo)
    monkeypatch.setattr(manager, "MARKER_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(sys, "argv", list(argv))
    return manager.main(), stub


def test_an_unreadable_marker_refuses_to_submit_and_fails_the_run(monkeypatch, sent):
    code, stub = _dispatch(monkeypatch, _Repo(*[(False, None)] * 3))
    assert stub.my_team_calls == 0, "the run went on to fetch the squad, i.e. towards submitting"
    assert code == 1, "a refusal must fail the job so it is seen"
    assert any(github_api.MARKER_VARIABLE in m for m in sent.messages)


def test_force_still_overrides_an_unreadable_marker(monkeypatch, sent):
    repo = _Repo(*[(False, None)] * 3)
    _, stub = _dispatch(monkeypatch, repo, ("manager.py", "--force"))
    assert repo.read_calls == 0 and stub.my_team_calls == 1


def test_an_absent_marker_still_proceeds(monkeypatch, sent):
    """The other direction: the first submission of a season has no marker."""
    _, stub = _dispatch(monkeypatch, _Repo((True, None)))
    assert stub.my_team_calls == 1
    assert sent.messages == []
