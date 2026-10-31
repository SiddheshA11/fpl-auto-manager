"""
A chip expiring unplayed must not be silent.

Nothing in the weekly run reports a chip it never picks, and on 2026-27's
numbers all four first-half chips would have lapsed at GW19 without a line in
any log. The watchdog already runs hourly on public endpoints, so it reports
what is still unplayed in every all-clear and alerts once a window has closed
over one. No network: the fetcher is stubbed, as in test_watchdog.py.
"""
from __future__ import annotations

from datetime import timedelta

import watchdog
from github_api import MARKER_VARIABLE
from test_watchdog import _at, _events, _FakeGitHub, _SpyNotifier

EVENTS = _events(
    (18, "2026-12-29T18:00:00Z"),
    (19, "2027-01-01T18:30:00Z"),
    (20, "2027-01-08T18:30:00Z"),
)
WINDOWS = [
    {"name": n, "start_event": 1 if n in ("bboost", "3xc") else 2, "stop_event": 19}
    for n in ("wildcard", "freehit", "bboost", "3xc")
] + [{"name": n, "start_event": 20, "stop_event": 38}
     for n in ("wildcard", "freehit", "bboost", "3xc")]
HISTORY = {"chips": [{"name": "wildcard", "event": 11}, {"name": "3xc", "event": 17}]}


def _fetcher(history=HISTORY, history_error=None):
    calls = []

    def fetch(url, timeout=30):
        calls.append(url)
        if "/history/" in url:
            if history_error:
                raise history_error
            return history
        return {"events": EVENTS, "chips": WINDOWS}
    fetch.calls = calls
    return fetch


def test_the_open_window_reports_what_is_left():
    got = watchdog.assess_chips(WINDOWS, EVENTS, [("wildcard", 11), ("3xc", 17)],
                                _at("2026-12-30T12:00:00"))
    assert got["open"] == {"unplayed": ["bboost", "freehit"], "closes": 19, "left": 1}
    assert got["lapsed"] == []


def test_a_closed_window_with_chips_left_is_a_lapse():
    after = _at("2027-01-01T18:30:00") + timedelta(hours=3)
    got = watchdog.assess_chips(WINDOWS, EVENTS, [("wildcard", 11), ("3xc", 17)], after)
    assert got["lapsed"] == [("bboost", 19), ("freehit", 19)]
    # The second-half chips are the open window now, and all of them are fresh.
    assert got["open"]["closes"] == 38 and len(got["open"]["unplayed"]) == 4


def test_a_chip_played_on_the_final_gameweek_gets_its_grace_period():
    """History may lag the deadline; an alert inside the grace would be false."""
    just_after = _at("2027-01-01T18:30:00") + timedelta(minutes=30)
    got = watchdog.assess_chips(WINDOWS, EVENTS, [], just_after)
    assert got["lapsed"] == []


def test_a_lapse_stops_alerting_once_it_is_old():
    late = _at("2027-01-01T18:30:00") + timedelta(hours=watchdog.MISS_ALERT_WINDOW_HOURS + 1)
    assert watchdog.assess_chips(WINDOWS, EVENTS, [], late)["lapsed"] == []


def test_every_chip_played_is_quiet():
    played = [(n, 5 + i) for i, n in enumerate(("wildcard", "freehit", "bboost", "3xc"))]
    after = _at("2027-01-01T18:30:00") + timedelta(hours=3)
    assert watchdog.assess_chips(WINDOWS, EVENTS, played, after)["lapsed"] == []


def test_a_lapse_alerts_and_withholds_the_all_clear(monkeypatch):
    monkeypatch.setenv("FPL_TEAM_ID", "5413589")
    now = _at("2027-01-01T18:30:00") + timedelta(hours=3)
    gh = _FakeGitHub({MARKER_VARIABLE: "19"})
    spy = _SpyNotifier()
    verdict = watchdog.run(now=now, gh=gh, notifier=spy, fetcher=_fetcher())

    assert verdict["chips"]["lapsed"] == [("bboost", 19), ("freehit", 19)]
    lapse = [m for m in spy.sent if "unplayed" in m and "closed" in m]
    assert lapse and "bboost (GW19)" in lapse[0] and "freehit (GW19)" in lapse[0]
    assert not any("All clear" in m for m in spy.sent)


def test_the_all_clear_carries_the_chip_status(monkeypatch):
    monkeypatch.setenv("FPL_TEAM_ID", "5413589")
    spy = _SpyNotifier()
    fetch = _fetcher()
    watchdog.run(now=_at("2026-12-26T12:00:00"), gh=_FakeGitHub({MARKER_VARIABLE: "17"}),
                 notifier=spy, fetcher=fetch)
    clear = [m for m in spy.sent if "All clear" in m]
    assert clear and "bboost, freehit unplayed" in clear[0] and "closes GW19" in clear[0]
    assert sum("bootstrap-static" in u for u in fetch.calls) == 1, "bootstrap fetched twice"


def test_an_unreadable_history_is_reported_not_fatal(monkeypatch):
    """The deadline check is the job; chip status failing must not take it down."""
    monkeypatch.setenv("FPL_TEAM_ID", "5413589")
    spy = _SpyNotifier()
    verdict = watchdog.run(now=_at("2026-12-26T12:00:00"), gh=_FakeGitHub({MARKER_VARIABLE: "17"}),
                           notifier=spy, fetcher=_fetcher(history_error=RuntimeError("503")))
    assert verdict["state"] == "waiting"
    assert verdict["chips"] == {"error": "RuntimeError: 503"}
    assert any("Chips: status unavailable (RuntimeError: 503)" in m for m in spy.sent)


def test_no_team_id_says_so_rather_than_reporting_nothing_unplayed(monkeypatch):
    monkeypatch.delenv("FPL_TEAM_ID", raising=False)
    spy = _SpyNotifier()
    fetch = _fetcher()
    watchdog.run(now=_at("2026-12-26T12:00:00"), gh=_FakeGitHub({MARKER_VARIABLE: "17"}),
                 notifier=spy, fetcher=fetch)
    assert any("Chips: status unavailable (FPL_TEAM_ID unset)" in m for m in spy.sent)
    assert not any("/history/" in u for u in fetch.calls)


def test_the_watchdog_job_is_given_the_team_id():
    from pathlib import Path
    wf = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "watchdog.yml").read_text()
    assert "FPL_TEAM_ID: ${{ secrets.FPL_TEAM_ID }}" in wf
