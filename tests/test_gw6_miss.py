"""
The GW6 2026-27 miss, as a regression test.

The deadline was 2026-10-10T10:00:00Z. Three scheduled runs arrived that day:

    01:29:16Z   lead  8.51h   too early for the old [2.0, 6.0) band
    08:03:33Z   lead  1.94h   3.6 MINUTES short of the old 2.0h floor
    15:00:46Z   lead -5.01h   after the deadline

The middle one skipped, the deadline passed with nothing submitted, and the
watchdog - 3-hourly, triggering at 4h - arrived at 6.7h out (correctly silent)
and then 1.6h after the deadline.

Measured cause, not a guess: GitHub throttles this repo's hourly cron to ~5.5
arrivals a day, median gap 4.24h, max 9.57h. Bootstrapping that distribution
over all 38 real deadlines, the old band was hit 81.3% of the time and the
old band plus old watchdog submitted 94.8% of gameweeks. The new settings
measure 100.0%.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import deadline_state as ds
import watchdog

GW6_DEADLINE = datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc)
GW6_EVENT = {"id": 6, "deadline_time": "2026-10-10T10:00:00Z"}

# The three arrivals, exactly as GitHub delivered them.
GW6_ARRIVALS = [
    datetime(2026, 10, 10, 1, 29, 16, tzinfo=timezone.utc),
    datetime(2026, 10, 10, 8, 3, 33, tzinfo=timezone.utc),
    datetime(2026, 10, 10, 15, 0, 46, tzinfo=timezone.utc),
]


def test_the_run_that_missed_gw6_by_three_minutes_now_acts():
    """08:03:33Z had 1.94h in hand and was rejected by an unjustified floor."""
    arrival = GW6_ARRIVALS[1]
    lead = (GW6_DEADLINE - arrival).total_seconds() / 3600
    assert 1.9 < lead < 2.0, "precondition: this arrival really was just under 2h"

    act, why = ds.should_act(GW6_EVENT, last_submitted_gw=5, now=arrival)
    assert act is True, (
        f"the 08:03Z run had {lead:.2f}h before the deadline and still skipped: {why}"
    )


def test_at_least_one_of_gw6s_real_arrivals_would_have_submitted():
    """The whole day, replayed. Exactly the failure, end to end."""
    acted = [
        a for a in GW6_ARRIVALS
        if ds.should_act(GW6_EVENT, last_submitted_gw=5, now=a)[0]
    ]
    assert acted, "no arrival on GW6 day qualifies; the deadline is missed again"
    first = acted[0]
    lead = (GW6_DEADLINE - first).total_seconds() / 3600
    assert lead > 0, "acted after the deadline, which is not acting"


def test_a_post_deadline_arrival_never_acts_on_the_passed_gameweek():
    """Guards the other direction: 0.5h is a floor, not an invitation."""
    after = GW6_ARRIVALS[2]
    assert ds.in_acting_window((GW6_DEADLINE - after).total_seconds() / 3600) is False


def test_the_floor_still_leaves_time_for_a_run_to_finish():
    """
    A run takes about 90 seconds. The floor protects against submitting into a
    closed deadline, nothing more - but it must not be zero either.
    """
    assert 0.25 <= ds.DEADLINE_WINDOW_MIN <= 1.0


def test_the_band_stays_tight_enough_to_act_on_fresh_news():
    """
    The FIRST qualifying tick acts, so the upper bound IS the worst-case
    acting lead. Widening it to buy coverage is the trap: a 6-hourly cron on a
    2-26h band covers everything and decides on 22-25h-old team news.
    """
    assert ds.DEADLINE_WINDOW_MAX <= 12.0


def test_the_watchdog_triggers_while_it_can_still_remediate():
    """
    At 4h with ~5h arrival gaps it could not reliably arrive in time. It must
    fire early enough that one of its own sparse arrivals lands in the zone.
    """
    assert watchdog.ALERT_LEAD_HOURS >= 6.0
    v = watchdog.assess([GW6_EVENT], last_submitted_gw=5,
                        now=GW6_DEADLINE - timedelta(hours=5))
    assert v["state"] == "overdue"
    assert v["alert"] is True and v["trigger"] is True


def test_the_watchdog_does_not_pre_empt_the_scheduled_run_pointlessly():
    """
    It must stay quiet while the run still has qualifying ticks ahead of it,
    or every gameweek arrives as an emergency.
    """
    v = watchdog.assess([GW6_EVENT], last_submitted_gw=5,
                        now=GW6_DEADLINE - timedelta(hours=20))
    assert v["state"] == "waiting"
    assert v["alert"] is False and v["trigger"] is False


@pytest.mark.parametrize("gw,marker", [(6, 6), (7, 7)])
def test_a_submitted_gameweek_is_still_left_alone(gw, marker):
    ev = {"id": gw, "deadline_time": "2026-10-10T10:00:00Z"}
    act, why = ds.should_act(ev, last_submitted_gw=marker,
                             now=GW6_DEADLINE - timedelta(hours=3))
    assert act is False and "already submitted" in why
