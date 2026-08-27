"""
The daily cron must act on exactly one day per gameweek.

A fixed Friday cron missed five gameweeks of 2026-27 outright - GW13, 18, 20,
25 and 28, all Wednesday midweek rounds with no Friday between the previous
deadline and their own - and fired up to three times for others, once 362 hours
out. Deadlines fall on Sat 26 times, Fri 5, Wed 5 and Sun 2.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import manager


def _at(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_hours_to_deadline_reads_fpls_format():
    now = _at("2026-12-01T10:00:00")
    got = manager.hours_to_deadline({"deadline_time": "2026-12-02T18:30:00Z"}, now)
    assert got == 32.5


def test_a_missing_or_unparseable_deadline_is_none():
    """None must mean 'unknown' so the caller can proceed rather than skip."""
    assert manager.hours_to_deadline({}) is None
    assert manager.hours_to_deadline({"deadline_time": None}) is None
    assert manager.hours_to_deadline({"deadline_time": "not a date"}) is None


def test_the_window_is_at_most_a_day_wide():
    """
    Wider than 24h and two consecutive daily runs both qualify, which is the
    double submission the disabled scheduler was killed for.
    """
    assert manager.DEADLINE_WINDOW_MAX - manager.DEADLINE_WINDOW_MIN <= 24.0


def test_every_deadline_shape_gets_exactly_one_run():
    """
    Replay a daily 10:00 cron against the real 2026-27 deadline shapes - the
    Wednesday midweek rounds are the ones a Friday cron dropped.
    """
    deadlines = [
        "2026-08-21T17:30:00",   # Fri
        "2026-09-12T12:30:00",   # Sat lunchtime
        "2026-10-10T10:00:00",   # Sat, same hour as the cron
        "2026-11-28T13:30:00",   # Sat afternoon
        "2026-12-02T18:30:00",   # Wed midweek - missed by a Friday cron
        "2027-01-06T18:30:00",   # Wed midweek
        "2027-05-23T15:00:00",   # Sun
    ]
    for iso in deadlines:
        deadline = _at(iso)
        fires = []
        t = deadline - timedelta(days=6)
        t = t.replace(hour=10, minute=0, second=0, microsecond=0)
        while t < deadline:
            lead = (deadline - t).total_seconds() / 3600.0
            if manager.DEADLINE_WINDOW_MIN <= lead < manager.DEADLINE_WINDOW_MAX:
                fires.append(round(lead, 1))
            t += timedelta(days=1)
        assert len(fires) == 1, f"{iso} fired {len(fires)} times: {fires}"
        assert 2.0 <= fires[0] < 26.0
