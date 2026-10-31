"""
Three ways the submit path reported success, or acted, when it should not.

- An unreadable submission marker was read as "no marker" and the run went on
  to submit. On a dispatch that read is the only idempotency there is.
- A failed token rotation set `rotation_failed` and nothing read it, so the run
  exited 0 on a spent token and the next one could not log in.
- A failed set_lineup exiting 1 had no test: the exit-code stub returns None
  from get_my_team and never reaches STEP 6, so reverting it stayed green.
"""
from __future__ import annotations

import sys

import pytest

import github_api
import manager
from test_exit_codes import _StubClient
from test_pipeline import StubClient, _snapshot
from test_recovery_paths import _squad_and_costs


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


# ------------------------------------------------------------- the rotation

class _RotationFailed(_StubClient):
    """The flag is set by logging in, as FPLClient sets it - not before."""

    def login(self):
        self.rotation_failed = True
        return super().login()


def test_a_failed_rotation_alerts_and_fails_a_run_that_submitted(game_state, monkeypatch, sent):
    """
    The run still submits with the access token it holds, then fails the job.
    Driven through a real submission so the exit code is the rotation's, not
    an abort's. (It used to be shown on an out-of-window skip; a skip no
    longer logs in, so it can no longer spend or fail a rotation.)
    """
    bootstrap, fixtures = game_state
    owned, costs = _squad_and_costs(bootstrap, fixtures)

    class RotatingClient(StubClient):
        def login(self):
            self.rotation_failed = True
            return super().login()

    client = RotatingClient(bootstrap, fixtures, owned, costs)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)
    monkeypatch.setattr(manager, "_github", lambda: _Repo((True, None)))
    monkeypatch.setattr(sys, "argv", ["manager.py"])
    assert manager.main() == 1, "a spent token exited 0 green"
    assert client.submitted_lineup is not None, "the team must still go in"
    assert any("ROTATION FAILED" in m for m in sent.messages)


def test_an_already_submitted_dispatch_never_logs_in(monkeypatch, sent):
    """
    31 Oct, the backup run: dispatched (no window), GW9 already marked. It
    logged in with a token rotated a minute earlier, failed, and alerted
    over a submitted team. Now it stops at the marker and exits 0.
    """
    stub = _StubClient(lead_hours=2.7)
    calls = []
    stub.login = lambda: calls.append(1) or True
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    monkeypatch.setattr(manager, "_github", lambda: _Repo((True, "3")))
    monkeypatch.setattr(sys, "argv", ["manager.py"])
    assert manager.main() == 0
    assert calls == [], "a run with nothing to do logged in"
    assert sent.messages == []


def test_an_unreadable_marker_refuses_before_logging_in(monkeypatch, sent):
    stub = _StubClient(lead_hours=2.7)
    calls = []
    stub.login = lambda: calls.append(1) or True
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    monkeypatch.setattr(manager, "_github", lambda: _Repo(*[(False, None)] * 3))
    monkeypatch.setattr(manager, "MARKER_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(sys, "argv", ["manager.py"])
    assert manager.main() == 1
    assert calls == []


def test_a_failed_rotation_is_reported_on_a_refused_login_too(monkeypatch, sent):
    stub = _RotationFailed(lead_hours=5.0, login_ok=False)
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    monkeypatch.setattr(sys, "argv", ["manager.py"])
    assert manager.main() == 1
    assert any("ROTATION FAILED" in m for m in sent.messages)


def test_a_clean_rotation_stays_quiet(monkeypatch, sent):
    stub = _StubClient(lead_hours=164.5)
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    monkeypatch.setattr(sys, "argv", ["manager.py", "--respect-window"])
    assert manager.main() == 0
    assert sent.messages == []


# ------------------------------------------------------------- the lineup

@pytest.fixture(scope="module")
def game_state():
    return _snapshot("bootstrap-static"), _snapshot("fixtures")


def test_a_refused_lineup_exits_one_and_leaves_the_marker_unset(game_state, monkeypatch, sent):
    """
    Driven to STEP 6 on the real model. Recording the marker here would stand
    the watchdog down over a team that never went in.
    """
    bootstrap, fixtures = game_state
    owned, costs = _squad_and_costs(bootstrap, fixtures)

    class RefusingClient(StubClient):
        def set_lineup(self, picks, chip=None):
            self.lineup_attempts = getattr(self, "lineup_attempts", 0) + 1
            return None

    client = RefusingClient(bootstrap, fixtures, owned, costs)
    repo = _Repo((True, None))
    monkeypatch.setattr(manager, "FPLClient", lambda: client)
    monkeypatch.setattr(manager, "_github", lambda: repo)
    monkeypatch.setattr(sys, "argv", ["manager.py"])

    assert manager.main() == 1
    assert client.lineup_attempts == 1, "the run must actually have reached STEP 6"
    assert github_api.MARKER_VARIABLE not in repo.written
