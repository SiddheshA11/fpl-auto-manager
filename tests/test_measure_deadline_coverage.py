"""
The coverage harness, on streams whose answer is known.

The point of replaying the real stream is that it keeps runs of slow gaps
together; the i.i.d. bootstrap breaks them up. A stream built to have that
structure has to show the difference, or the harness is not measuring it.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

import measure_deadline_coverage as m

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
DEADLINES = [T0.replace(hour=h, minute=30) for h in (10, 12, 17)]


def _stream(gaps_h):
    t, out = T0, []
    for g in gaps_h:
        t += timedelta(hours=g)
        out.append(t)
    return out


def test_an_arrival_exactly_at_the_band_edge_counts_once():
    t = T0 + timedelta(hours=10)
    s = [t - timedelta(hours=2)]
    assert m.any_in(s, t - timedelta(hours=6), t - timedelta(hours=2)) is False
    assert m.any_in(s, t - timedelta(hours=6), t - timedelta(hours=1.99)) is True


def test_a_punctual_hourly_cron_covers_any_band_an_hour_wide():
    s = _stream([1.0] * 24 * 30)
    rows = m.replay(DEADLINES, s[0], s[-1], m.band_hit(s, (2.0, 3.0)))
    cov, n, worst = m.summarise(rows)
    assert n > 60 and cov == 1.0 and worst == 1.0
    assert m.lag1([1.0, 2.0] * 50) == pytest.approx(-1.0)


def test_gap_order_alone_does_not_change_coverage():
    """
    A measured negative result, pinned. Alternating fast days (hourly) and
    slow days (9h gaps) have strong lag-1 correlation, and the i.i.d. shuffle
    of the same gaps covers a narrow band just as often: one window's coverage
    depends on how much time sits inside long gaps, not on their order. The
    17 Oct audit blamed autocorrelation for the optimistic figures; this is
    why that was the wrong mechanism.
    """
    gaps = []
    for day in range(60):
        gaps += [1.0] * 24 if day % 2 == 0 else [9.0] * 3
    s = _stream(gaps)
    assert m.lag1(m.gaps_h(s)) > 0.5

    band = (2.0, 6.0)
    cov, _, _ = m.summarise(m.replay(DEADLINES, s[0], s[-1], m.band_hit(s, band)))
    iid = m.iid_bootstrap(m.gaps_h(s), band, 20000, random.Random(0))
    assert cov == pytest.approx(iid, abs=0.03)


def test_arrivals_thin_before_the_deadline_hour_cover_less_than_iid_suggests():
    """
    The mechanism that is real: same gaps every day, but the slow stretch
    falls just before the deadline's time of day. The i.i.d. bootstrap cannot
    see when in the day a gap falls, so it is optimistic.
    """
    # Hourly from 13:00 to 04:00, then nothing until 13:00 (a 9h hole), daily.
    day = [T0.replace(hour=0) + timedelta(hours=h) for h in [*range(0, 5), *range(13, 24)]]
    s = [t + timedelta(days=k) for k in range(30) for t in day]
    deadline = [T0.replace(hour=12, minute=30)]
    band = (2.0, 6.0)
    cov, _, _ = m.summarise(m.replay(deadline, s[0], s[-1], m.band_hit(s, band)))
    iid = m.iid_bootstrap(m.gaps_h(s), band, 20000, random.Random(0))
    assert cov == 0.0, "06:30-10:30 has no arrival on any day"
    assert iid > 0.5


def test_a_watchdog_tick_rescues_a_deadline_the_weekly_run_missed():
    d = T0.replace(hour=10)
    weekly = [d - timedelta(hours=9)]                   # outside every band
    watchdog = [d - timedelta(hours=3)]
    assert m.band_hit(weekly, m.NEW_BAND)(d) is False
    assert m.joint_hit(weekly, watchdog, m.NEW_BAND, m.NEW_ALERT)(d) is True
    # ...but not if it arrives too late for a dispatched run to finish.
    late = [d - timedelta(hours=0.1)]
    assert m.joint_hit(weekly, late, m.NEW_BAND, m.NEW_ALERT)(d) is False


def test_the_span_edge_is_never_scored_as_a_miss():
    """A deadline whose look-back starts before the stream does is not replayed."""
    s = _stream([1.0] * 30)
    rows = m.replay([T0.replace(hour=3)], s[0], s[-1], m.band_hit(s, (0.5, 8.0)))
    assert all(target - timedelta(hours=12) >= s[0] for _, target, _ in rows)
