"""
A run that correctly decides to do nothing must not report failure.

`manager.py` returned None both for "the deadline is a week away, nothing to
do" and for "authentication failed", and main() mapped None to exit 1. On a
daily cron that marks roughly six runs in seven as failures, which is not a
cosmetic problem: the whole point of the deadline watchdog is to ask "was
there a successful Weekly Run since the last deadline", and that question is
unanswerable while a healthy skip and a dead client are the same exit code.

The live instance was run 33210780881 on 2026-08-28: token rotated, game state
loaded, "GW3 deadline is 164.5h away, outside the 2-26h window; nothing to do",
exit 1.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pytest

import manager


class _StubClient:
    """Enough client to reach the window check and no further."""

    def __init__(self, lead_hours: float | None, login_ok: bool = True):
        self._lead = lead_hours
        self._login_ok = login_ok
        self.my_team_calls = 0

    def login(self):
        return self._login_ok

    def get_bootstrap(self):
        return {"teams": [], "events": []}

    def get_fixtures(self):
        return []

    def get_next_event(self):
        if self._lead is None:
            return None
        deadline = datetime.now(timezone.utc) + timedelta(hours=self._lead)
        return {"id": 3, "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ")}

    def get_my_team(self):
        # Reaching here means the window check let the run through. Returning
        # None stops the test short of the model without pretending to be a
        # failure the test cares about.
        self.my_team_calls += 1
        return None


def _run(monkeypatch, argv, lead_hours, login_ok=True):
    stub = _StubClient(lead_hours, login_ok)
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    monkeypatch.setattr(sys, "argv", argv)
    return manager.main(), stub


def test_a_skip_outside_the_window_exits_zero(monkeypatch, capsys):
    """
    164.5 hours out, exactly the live failure. This is a healthy no-op and must
    be reported as one.
    """
    code, stub = _run(monkeypatch, ["manager.py", "--respect-window"], lead_hours=164.5)
    assert code == 0, "an out-of-window skip was reported as a failed run"
    assert stub.my_team_calls == 0, "the run should have stopped at the window check"


def test_the_skip_is_distinguishable_from_a_submission(monkeypatch):
    """
    Exit 0 alone is not enough - the watchdog has to tell 'nothing to do' from
    'squad submitted', or it will treat a season of skips as a season of
    successful runs and never fire.
    """
    stub = _StubClient(164.5)
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    out = manager.run_weekly_cycle(respect_window=True)
    assert out is not None
    assert out["status"] == "skipped"
    assert out["event_id"] == 3
    assert out["lead_hours"] == pytest.approx(164.5, abs=0.5)


def test_a_real_failure_still_exits_one(monkeypatch):
    """The fix must not turn genuine breakage green."""
    code, _ = _run(monkeypatch, ["manager.py", "--respect-window"],
                   lead_hours=164.5, login_ok=False)
    assert code == 1, "authentication failure must still be a failed run"


def test_no_upcoming_gameweek_still_exits_one(monkeypatch):
    """End of season is not a skip; it means the client told us nothing."""
    code, _ = _run(monkeypatch, ["manager.py", "--respect-window"], lead_hours=None)
    assert code == 1


def test_in_window_the_run_proceeds_past_the_check(monkeypatch):
    """
    Guards the other direction: a change that made everything 'skip' would pass
    the tests above while never submitting a team again.
    """
    stub = _StubClient(5.0)
    monkeypatch.setattr(manager, "FPLClient", lambda *a, **k: stub)
    out = manager.run_weekly_cycle(respect_window=True)
    assert stub.my_team_calls == 1, "an in-window run must get as far as fetching the squad"
    assert out is None  # the stub squad is None, which is a real abort
