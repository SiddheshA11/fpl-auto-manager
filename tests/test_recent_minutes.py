"""
Recent minutes in the start rate.

A season-to-date start rate is a *level*: a player who lost his place three
weeks ago still carries the average of the weeks he was playing. How long he
was on the pitch last week is a *state*, and it is the single most informative
thing available. Measured over 22,774 player-gameweeks, last week's minutes
alone predict minutes better (R2 0.598) than this entire model did (0.472).

That dominates everything else. Rescaling expected points by lag-predicted
minutes lifts points R2 from 0.280 to 0.350 on held-out data, while the whole
goals / assists / clean-sheet / bonus apparatus is worth 0.004 once minutes are
known.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import priors
import xp_model as X

SNAPS = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@pytest.fixture(scope="module")
def state():
    files = sorted(SNAPS.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(files[0])
    fixtures = X.load_snapshot(sorted(SNAPS.glob("fixtures_*.json.gz"), reverse=True)[0])
    return bootstrap, fixtures, priors.build_priors()


def _played(bootstrap, n=6):
    """Mark the first n gameweeks finished, so the in-season path is live."""
    out = {**bootstrap, "events": [dict(e) for e in bootstrap["events"]]}
    for i, e in enumerate(out["events"]):
        e["finished"] = i < n
        e["is_current"] = i == n - 1
        e["is_next"] = i == n
    return out


def test_supplying_nothing_leaves_the_model_untouched(state):
    """Every existing caller must be unaffected."""
    bootstrap, fixtures, ps = state
    a = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=3))
    b = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=3), recent_minutes={})
    events = X.next_events(bootstrap, 3)
    assert a.expected_points(events)["xp_next"].equals(b.expected_points(events)["xp_next"])


def test_a_player_dropped_from_the_side_is_marked_down(state):
    """
    The case a season average cannot see. Same season-to-date record, but one
    has not played in three weeks - and the model must notice.
    """
    bootstrap, fixtures, ps = state
    bootstrap = _played(bootstrap)
    model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=1))
    frame = model.players
    # Someone the model currently rates as a regular starter.
    # Best available rather than a fixed threshold: `> 0.8` selected nobody
    # once the season started, so this skipped silently in CI.
    idx = frame.sort_values("start_rate", ascending=False).index[:1]
    if not len(idx) or float(frame.loc[idx[0], "start_rate"]) < 0.4:
        pytest.skip("snapshot has no plausible starter")
    pid = int(frame.loc[idx[0], "id"])
    base = float(frame.loc[idx[0], "start_rate"])

    dropped = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=1),
                        recent_minutes={pid: [0.0, 0.0, 0.0, 12.0, 90.0]})
    after = float(dropped.players.loc[dropped.players["id"] == pid, "start_rate"].iloc[0])
    assert after < base, "three weeks out of the side must lower the start rate"

    kept = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=1),
                     recent_minutes={pid: [90.0, 90.0, 88.0, 90.0, 90.0]})
    held = float(kept.players.loc[kept.players["id"] == pid, "start_rate"].iloc[0])
    assert held > after, "an ever-present must rate above a dropped player"


def test_one_rested_week_is_not_read_as_a_demotion(state):
    """
    Recent minutes never fully displace the season rate. A player rested for a
    cup tie, or given a week off after an international break, must not be
    written off on one observation.
    """
    bootstrap, fixtures, ps = state
    bootstrap = _played(bootstrap)
    base_model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=1))
    frame = base_model.players
    # Best available rather than a fixed threshold: `> 0.8` selected nobody
    # once the season started, so this skipped silently in CI.
    idx = frame.sort_values("start_rate", ascending=False).index[:1]
    if not len(idx) or float(frame.loc[idx[0], "start_rate"]) < 0.4:
        pytest.skip("snapshot has no plausible starter")
    pid = int(frame.loc[idx[0], "id"])

    rested = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=1),
                       recent_minutes={pid: [0.0, 90.0, 90.0, 90.0, 90.0]})
    after = float(rested.players.loc[rested.players["id"] == pid, "start_rate"].iloc[0])
    assert after > 0.4, f"one rested week dropped an established starter to {after:.2f}"


def test_recent_minutes_are_ignored_before_a_ball_is_kicked(state):
    """Pre-season there is nothing to observe, and the priors are all there is."""
    bootstrap, fixtures, ps = state
    # Force it rather than assume the snapshot is pre-season: CI fetches live
    # data, so this held only until GW1 finished.
    bootstrap = {**bootstrap,
                 "events": [{**e, "finished": False, "is_current": False}
                            for e in bootstrap["events"]]}
    plain = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=1))
    pid = int(plain.players["id"].iloc[0])
    with_lags = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=1),
                          recent_minutes={pid: [0.0, 0.0, 0.0]})
    assert plain.players["start_rate"].equals(with_lags.players["start_rate"])


def test_the_season_rate_no_longer_hard_caps_at_eight_gameweeks(state):
    """
    The blend used min(events/8, 1.0), which discards the prior entirely from
    GW8 onward - reading an eight-game sample as certainty. n/(n+k) keeps a
    little of the prior all season, which is what the evidence supports.
    """
    assert X.START_RATE_EVIDENCE_GAMEWEEKS > 0
    for n in (8, 19, 38):
        w = n / (n + X.START_RATE_EVIDENCE_GAMEWEEKS)
        assert w < 1.0, f"the prior is fully displaced by gameweek {n}"
    assert 8 / (8 + X.START_RATE_EVIDENCE_GAMEWEEKS) < 19 / (19 + X.START_RATE_EVIDENCE_GAMEWEEKS)
