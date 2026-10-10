"""
The schedule must give every gameweek several chances, all of them late.

History, because both halves were learned the hard way:

  - A fixed Friday cron missed five gameweeks of 2026-27 outright - GW13, 18,
    20, 25 and 28, the Wednesday midweek rounds with no Friday between the
    previous deadline and their own.
  - Replacing it with a daily 10:00 cron on a 2-26h window fixed coverage and
    left a subtler hole: a 24h-wide window sampled every 24h admits exactly one
    tick per gameweek, for all 38. There was no retry. On 2026-08-28 that one
    tick did not fire, a run appeared at 21:02 - three and a half hours after
    the GW2 deadline - and the gameweek went in on a stale squad.
  - Widening naively trades the problem rather than fixing it. A 6-hourly cron
    on the old 2-26h band yields four ticks per gameweek, but the *first* one
    acts, at a lead of 22.0-25.5h. The bot would decide every week on day-old
    team news.

So: a narrow late band, sampled hourly. Several qualifying ticks, all close to
the deadline, with the submission marker making the spares idempotent.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import deadline_state as ds
import manager

# The real 2026-27 deadline shapes: Sat 26, Fri 5, Wed 5, Sun 2.
DEADLINE_SHAPES = [
    "2026-08-21T17:30:00",   # Fri evening
    "2026-09-12T12:30:00",   # Sat lunchtime - the shortest lead of the season
    "2026-10-10T10:00:00",   # Sat morning
    "2026-11-28T13:30:00",   # Sat afternoon
    "2026-12-02T18:30:00",   # Wed midweek - dropped entirely by a Friday cron
    "2027-01-06T18:30:00",   # Wed midweek
    "2027-05-23T15:00:00",   # Sun
]

MIN_TICKS_PER_GAMEWEEK = 2   # one to act, at least one held back as a retry


def _at(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _qualifying_ticks(deadline: datetime, step_hours: int = 1) -> list[float]:
    """Leads at which an hourly cron would find itself inside the window."""
    leads = []
    t = (deadline - timedelta(days=3)).replace(minute=0, second=0, microsecond=0)
    while t < deadline:
        lead = (deadline - t).total_seconds() / 3600.0
        if ds.in_acting_window(lead):
            leads.append(round(lead, 1))
        t += timedelta(hours=step_hours)
    return leads


def test_hours_to_deadline_reads_fpls_format():
    now = _at("2026-12-01T10:00:00")
    assert manager.hours_to_deadline({"deadline_time": "2026-12-02T18:30:00Z"}, now) == 32.5


def test_a_missing_or_unparseable_deadline_is_none():
    """None must mean 'unknown' so the caller can proceed rather than skip."""
    assert manager.hours_to_deadline({}) is None
    assert manager.hours_to_deadline({"deadline_time": None}) is None
    assert manager.hours_to_deadline({"deadline_time": "not a date"}) is None


def test_an_unknown_deadline_runs_rather_than_stands_down():
    """A schema change must make the bot noisy, not quietly stop playing."""
    assert ds.in_acting_window(None) is True


def test_every_deadline_shape_gets_a_retry_not_just_a_chance():
    """
    The single-tick schedule is what lost GW2. One qualifying tick per gameweek
    is not a schedule, it is a coin flip on GitHub's scheduler.
    """
    for iso in DEADLINE_SHAPES:
        ticks = _qualifying_ticks(_at(iso))
        assert len(ticks) >= MIN_TICKS_PER_GAMEWEEK, (
            f"{iso} gets {len(ticks)} qualifying tick(s): {ticks}. "
            "A dropped run would miss the deadline outright."
        )


def test_the_run_acts_on_fresh_news_not_day_old_news():
    """
    Guards against 'fixing' coverage by widening the window: the tick that
    acts is the earliest qualifying one, so a wide band means a stale decision.
    """
    for iso in DEADLINE_SHAPES:
        ticks = _qualifying_ticks(_at(iso))
        acting_lead = max(ticks)   # the earliest tick in the window acts first
        assert acting_lead <= 12.0, (
            f"{iso} would be decided {acting_lead}h out, on stale team news"
        )


def test_no_submission_is_attempted_inside_the_dead_zone():
    """Too close to the deadline and a submission may not land."""
    assert ds.in_acting_window(ds.DEADLINE_WINDOW_MIN - 0.1) is False
    assert ds.in_acting_window(0.5) is False


def test_the_window_is_narrow_enough_to_stay_late():
    assert ds.DEADLINE_WINDOW_MAX - ds.DEADLINE_WINDOW_MIN <= 12.0


def test_manager_still_exposes_the_window_it_enforces():
    """The run re-checks the window itself; it does not trust the gate job."""
    assert manager.DEADLINE_WINDOW_MIN == ds.DEADLINE_WINDOW_MIN
    assert manager.DEADLINE_WINDOW_MAX == ds.DEADLINE_WINDOW_MAX


def test_repeat_ticks_are_made_safe_by_the_marker():
    """
    Extra ticks only stop being a double-submission risk because should_act
    consults the marker. If that link ever breaks, the schedule becomes the
    hazard the old single-tick design existed to avoid.
    """
    event = {"id": 7, "deadline_time": "2026-10-17T10:00:00Z"}
    now = _at("2026-10-17T05:00:00")   # 5h out, inside the window

    act, why = ds.should_act(event, last_submitted_gw=None, now=now)
    assert act is True, why

    act, why = ds.should_act(event, last_submitted_gw=7, now=now)
    assert act is False and "already submitted" in why


def test_a_marker_from_last_week_does_not_block_this_week():
    event = {"id": 7, "deadline_time": "2026-10-17T10:00:00Z"}
    now = _at("2026-10-17T05:00:00")
    act, _ = ds.should_act(event, last_submitted_gw=6, now=now)
    assert act is True
