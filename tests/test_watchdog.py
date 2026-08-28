"""
The watchdog has to be right about one thing: whether a team has been submitted
for the gameweek in front of us. Everything else is delivery.

Nothing here touches the network. The watchdog exists to work when the rest is
broken, so its own tests must not depend on anything being up.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import deadline_state as ds
import notify
import watchdog


def _at(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _events(*specs):
    """(id, iso_deadline) pairs -> bootstrap-shaped events."""
    return [{"id": i, "deadline_time": iso} for i, iso in specs]


SEASON = _events(
    (1, "2026-08-21T17:30:00Z"),
    (2, "2026-08-28T17:30:00Z"),
    (3, "2026-09-04T17:30:00Z"),
)


class _FakeGitHub:
    def __init__(self, variables=None, dispatch_ok=True):
        self.variables = dict(variables or {})
        self.dispatched = []
        self._dispatch_ok = dispatch_ok

    def get_variable(self, name):
        return self.variables.get(name)

    def set_variable(self, name, value):
        self.variables[name] = value
        return True

    def dispatch_workflow(self, workflow, ref="main", inputs=None):
        self.dispatched.append((workflow, ref, inputs))
        return self._dispatch_ok


class _SpyNotifier:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return True


def _fetcher(events):
    return lambda url, timeout=30: {"events": events}


# ---------------------------------------------------------------- the verdict

def test_an_unsubmitted_gameweek_near_the_deadline_is_overdue():
    """The GW2 failure, exactly: deadline imminent, nothing submitted."""
    v = watchdog.assess(SEASON, last_submitted_gw=1, now=_at("2026-09-04T15:00:00"))
    assert v["state"] == "overdue"
    assert v["alert"] is True and v["trigger"] is True
    assert v["event_id"] == 3


def test_a_submitted_gameweek_is_quiet():
    v = watchdog.assess(SEASON, last_submitted_gw=3, now=_at("2026-09-04T15:00:00"))
    assert v["state"] == "submitted"
    assert v["alert"] is False and v["trigger"] is False


def test_an_unsubmitted_gameweek_days_out_is_not_an_emergency():
    """Alerting a week early would train the alert to be ignored."""
    v = watchdog.assess(SEASON, last_submitted_gw=2, now=_at("2026-08-30T10:00:00"))
    assert v["state"] == "waiting"
    assert v["alert"] is False and v["trigger"] is False


def test_the_marker_must_match_the_gameweek_in_front_not_merely_exist():
    """
    A stale marker from last week must not read as 'submitted'. This is the
    whole difference between watching the outcome and watching the cron.
    """
    v = watchdog.assess(SEASON, last_submitted_gw=2, now=_at("2026-09-04T15:00:00"))
    assert v["state"] == "overdue"


def test_no_marker_at_all_near_a_deadline_is_overdue():
    v = watchdog.assess(SEASON, last_submitted_gw=None, now=_at("2026-09-04T15:00:00"))
    assert v["state"] == "overdue"


def test_an_unreadable_deadline_alerts_but_does_not_blind_fire():
    broken = [{"id": 3, "deadline_time": "not a date"}]
    v = watchdog.assess(broken, None, now=_at("2026-09-04T15:00:00"))
    # next_event cannot place an unparseable deadline, so this reads as
    # 'no gameweek' - which must still be visible rather than a silent pass.
    assert v["alert"] is True or v["state"] == "no-gameweek"


def test_end_of_season_does_not_alert_forever():
    v = watchdog.assess(SEASON, last_submitted_gw=3, now=_at("2027-07-01T00:00:00"))
    assert v["state"] == "no-gameweek"
    assert v["alert"] is False


def test_the_alert_threshold_sits_above_the_weekly_runs_last_chance():
    """
    If the watchdog waited until inside the run's own window minimum it could
    not remediate: it would dispatch a run that manager.py refuses to act on.
    """
    assert watchdog.ALERT_LEAD_HOURS > ds.DEADLINE_WINDOW_MIN


def test_the_run_acts_before_the_watchdog_calls_it_overdue():
    """
    The scheduled run must get its turn first, or the watchdog becomes the
    primary path and every gameweek arrives as an emergency.
    """
    assert watchdog.ALERT_LEAD_HOURS < ds.DEADLINE_WINDOW_MAX


# ------------------------------------------------------------- the whole run

def test_an_overdue_gameweek_alerts_and_dispatches(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2"})
    spy = _SpyNotifier()
    v = watchdog.run(now=_at("2026-09-04T15:00:00"), gh=gh, notifier=spy,
                     fetcher=_fetcher(SEASON))
    assert v["state"] == "overdue"
    assert gh.dispatched == [("weekly_run.yml", "main", {"dry_run": "false"})]
    assert len(spy.sent) == 1
    assert "GW3" in spy.sent[0]
    assert "Dispatched the weekly run automatically" in spy.sent[0]


def test_a_failed_dispatch_says_so_instead_of_claiming_success(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2"}, dispatch_ok=False)
    spy = _SpyNotifier()
    watchdog.run(now=_at("2026-09-04T15:00:00"), gh=gh, notifier=spy,
                 fetcher=_fetcher(SEASON))
    assert "Could not dispatch" in spy.sent[0]
    assert "by hand" in spy.sent[0]


def test_dry_run_dispatches_nothing_and_sends_nothing(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2"})
    spy = _SpyNotifier()
    watchdog.run(dry_run=True, now=_at("2026-09-04T15:00:00"), gh=gh, notifier=spy,
                 fetcher=_fetcher(SEASON))
    assert gh.dispatched == []
    assert spy.sent == []


def test_a_healthy_week_stays_silent_once_the_heartbeat_is_recent(monkeypatch):
    """Alert fatigue is a real failure mode; a quiet week must be quiet."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    recent = _at("2026-08-30T08:00:00").isoformat()
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2",
                      watchdog.HEARTBEAT_VARIABLE: recent})
    spy = _SpyNotifier()
    watchdog.run(now=_at("2026-08-30T10:00:00"), gh=gh, notifier=spy,
                 fetcher=_fetcher(SEASON))
    assert spy.sent == []


def test_every_run_stamps_its_heartbeat(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2"})
    now = _at("2026-08-30T10:00:00")
    watchdog.run(now=now, gh=gh, notifier=_SpyNotifier(), fetcher=_fetcher(SEASON))
    assert gh.variables[watchdog.HEARTBEAT_VARIABLE] == now.isoformat()


def test_the_watchdog_reports_its_own_lateness(monkeypatch):
    """
    GitHub dropped the weekly run's schedule; it can drop this one too. A
    watchdog that is silently not running is worse than none, because it is
    trusted.
    """
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    stale = _at("2026-08-29T10:00:00").isoformat()
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2",
                      watchdog.HEARTBEAT_VARIABLE: stale})
    spy = _SpyNotifier()
    v = watchdog.run(now=_at("2026-08-30T10:00:00"), gh=gh, notifier=spy,
                     fetcher=_fetcher(SEASON))
    assert v["late_by_hours"] == pytest.approx(24.0)
    assert any("after its previous run" in m for m in spy.sent)


def test_a_punctual_watchdog_does_not_cry_late():
    on_time = _at("2026-08-30T08:00:00")
    assert watchdog.check_own_lateness(on_time, _at("2026-08-30T10:00:00")) is None


def test_a_first_ever_run_has_no_lateness_to_report():
    assert watchdog.check_own_lateness(None, _at("2026-08-30T10:00:00")) is None


def test_the_periodic_all_clear_is_sent_so_silence_is_meaningful(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2"})  # no heartbeat yet
    spy = _SpyNotifier()
    watchdog.run(now=_at("2026-08-30T10:00:00"), gh=gh, notifier=spy,
                 fetcher=_fetcher(SEASON))
    assert any("All clear" in m for m in spy.sent)


def test_a_missing_token_alerts_rather_than_reporting_all_clear(monkeypatch):
    """
    Without the API there is no marker, so 'no problem found' would be a lie.
    """
    monkeypatch.delenv("GH_PAT", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    spy = _SpyNotifier()
    v = watchdog.run(now=_at("2026-09-04T15:00:00"), notifier=spy,
                     fetcher=_fetcher(SEASON))
    assert v["state"] == "misconfigured"
    assert v["alert"] is True
    assert len(spy.sent) == 1


def test_an_unreachable_fpl_api_alerts_rather_than_passing(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")

    def boom(url, timeout=30):
        raise ConnectionError("dns")

    spy = _SpyNotifier()
    v = watchdog.run(now=_at("2026-09-04T15:00:00"), gh=_FakeGitHub(), notifier=spy,
                     fetcher=boom)
    assert v["state"] == "fpl-unreachable"
    assert len(spy.sent) == 1


def test_a_null_notifier_does_not_break_the_watchdog(monkeypatch):
    """Missing Telegram secrets must degrade delivery, not the decision."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    gh = _FakeGitHub({watchdog.MARKER_VARIABLE: "2"})
    v = watchdog.run(now=_at("2026-09-04T15:00:00"), gh=gh,
                     notifier=notify.NullNotifier(), fetcher=_fetcher(SEASON))
    assert v["state"] == "overdue"
    assert gh.dispatched, "remediation must still happen when alerting is mute"
