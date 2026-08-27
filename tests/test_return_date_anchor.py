"""
A return date is judged against the fixture, not the deadline.

Availability was anchored on the gameweek DEADLINE - a Friday - while FPL
writes "expected back" dates against a match, and a gameweek runs Friday to
Monday. Every player whose return fell inside his own gameweek but after that
Friday was held out of a match he started.

On the snapshot this was found with, 8 of the 11 players carrying a parsed
return date lost a whole gameweek: Baleba, Garner and Bajcetic were all marked
unavailable for a GW1 they played 90 minutes of. The three that escaped did so
only because their gameweek happened to have deadline == kickoff.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import priors
import xp_model as X

SNAPS = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@pytest.fixture(scope="module")
def model():
    files = sorted(SNAPS.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(files[0])
    fixtures = X.load_snapshot(sorted(SNAPS.glob("fixtures_*.json.gz"), reverse=True)[0])
    return X.XPModel(bootstrap, fixtures, priors.build_priors(), X.ModelConfig(horizon=5))


def test_kickoffs_are_later_than_the_deadline(model):
    """The premise. If these were equal the bug could not exist."""
    events = X.next_events(model.bootstrap, 5)
    later = 0
    for ev in events:
        deadline = model._event_date(ev)
        for when in model._return_dates_by_team(ev).values():
            if when > deadline:
                later += 1
    assert later > 0, "expected some fixtures after their deadline"


def test_every_team_in_a_gameweek_gets_a_date(model):
    events = X.next_events(model.bootstrap, 5)
    by_team = model._return_dates_by_team(events[0])
    playing = set(model.fixtures[model.fixtures["event"] == events[0]]["team_h"]) | \
              set(model.fixtures[model.fixtures["event"] == events[0]]["team_a"])
    assert set(by_team) == {int(t) for t in playing}


def test_a_double_gameweek_takes_the_later_kickoff(model):
    """
    A player back between the two fixtures still features in the gameweek, so
    the later date is the one that decides whether he is available at all.
    """
    fx = model.fixtures
    for ev in sorted(fx["event"].dropna().unique()):
        rows = fx[fx["event"] == ev]
        counts = list(rows["team_h"]) + list(rows["team_a"])
        doubled = [t for t in set(counts) if counts.count(t) > 1]
        if not doubled:
            continue
        team = int(doubled[0])
        theirs = rows[(rows["team_h"] == team) | (rows["team_a"] == team)]
        latest = max(X.datetime.fromisoformat(str(k).replace("Z", "+00:00")).date()
                     for k in theirs["kickoff_time"])
        assert model._return_dates_by_team(int(ev))[team] == latest
        return
    pytest.skip("no double gameweek in the fixture list")


def test_a_player_back_inside_the_gameweek_is_available_for_it(model):
    """
    The regression, through minutes_model rather than the helper. Every player
    with a return date must be available in the first gameweek whose fixture
    for HIS team falls on or after it.
    """
    events = X.next_events(model.bootstrap, 5)
    checked = 0
    for pos, pid in enumerate(model.players["id"].astype(int)):
        info = model.news.get(int(pid))
        if info is None or info.returns_on is None:
            continue
        team = int(model.players["team"].iloc[pos])
        expected = None
        for ev in events:
            kick = model._return_dates_by_team(ev).get(team)
            if kick is not None and kick >= info.returns_on:
                expected = ev
                break
        if expected is None:
            continue
        checked += 1
        avail = float(model.minutes_model(expected).iloc[pos]["availability"])
        assert avail > 0.01, (
            f"{model.players['web_name'].iloc[pos]} returns {info.returns_on} and his "
            f"team plays GW{expected} on {model._return_dates_by_team(expected)[team]}, "
            f"but availability is {avail:.3f}"
        )
    if not checked:
        pytest.skip("no dated returns inside the horizon")
