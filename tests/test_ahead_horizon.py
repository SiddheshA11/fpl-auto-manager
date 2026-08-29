"""
`ahead` must mean "how many gameweeks from now", not "how far into the
optimiser's horizon".

xp_model._availability derived it from `next_events(bootstrap, config.horizon)`
- five gameweeks - while production scores `max(HORIZON, PLANNING_HORIZON)`,
which is ten. Any event past the fifth was not in that list, so `ahead` fell to
its 0 default and both things keyed on it silently stopped:

  - decay_doubt is guarded by `ahead > 0`, so a knock healed through GW5 and
    then un-healed. Measured on the committed snapshot with 25 doubtful
    players: mean availability 0.7300 at GW1, rising to 0.9831 by GW5, then
    back to 0.7300 for GW6 through GW10 - understated by ~0.27.
  - _suspension_risk returns early on `ahead <= 0`, so an accumulating card
    ban was priced at zero for exactly the gameweeks a chip planner looks at.

Only xp_gw6..gw10 are affected, and only chips.value_by_gameweek reads those,
so this distorts chip timing rather than squad selection. The last test here
pins that boundary: the fix must not move the squad.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import priors
import recommend as R
import xp_model as X

REPO = Path(__file__).resolve().parent.parent

PLANNING_EVENTS = 10
DOUBTFUL_CHANCE = 50.0


@pytest.fixture(scope="module")
def game_state():
    bootstrap, fixtures = R.load_game_state(offline=True)
    return bootstrap, fixtures


def _model(bootstrap, fixtures, horizon):
    tc = {t["code"]: t["name"] for t in bootstrap["teams"]}
    ps = priors.build_priors(current_team_codes=tc)
    m = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=horizon))
    # A doubt we control, so the test does not depend on who happens to be
    # injured in whatever snapshot CI fetched. News is cleared so the date
    # logic does not overwrite the decay this test is about.
    m.players = m.players.copy()
    m.players.loc[m.players.index[0], "chance_of_playing_next_round"] = DOUBTFUL_CHANCE
    m.news = {}
    m._minutes_cache = {}
    return m


@pytest.fixture(scope="module")
def doubtful(game_state):
    bootstrap, fixtures = game_state
    m = _model(bootstrap, fixtures, 5)
    return m.players.index[0]


def test_a_doubt_does_not_un_heal_after_the_fifth_gameweek(game_state, doubtful):
    """
    The headline symptom. Availability rose for five gameweeks and then
    collapsed back to its starting value, which no injury does.
    """
    bootstrap, fixtures = game_state
    m = _model(bootstrap, fixtures, 5)
    events = X.next_events(bootstrap, PLANNING_EVENTS)
    assert len(events) >= 6, "need at least six gameweeks ahead to exercise this"

    avail = [float(m._availability(ev).loc[doubtful]) for ev in events]

    assert avail[0] < 1.0, "precondition: the injected doubt must actually be a doubt"
    for i in range(1, len(avail)):
        assert avail[i] >= avail[i - 1] - 1e-9, (
            f"availability fell from {avail[i-1]:.4f} at GW{events[i-1]} to "
            f"{avail[i]:.4f} at GW{events[i]}; a doubt heals, it does not relapse"
        )


def test_ahead_does_not_depend_on_the_optimisers_horizon(game_state, doubtful):
    """
    `ahead` is a fact about the calendar. Two models differing only in the
    horizon they optimise over must agree about how far away a gameweek is.
    """
    bootstrap, fixtures = game_state
    events = X.next_events(bootstrap, PLANNING_EVENTS)
    m5 = _model(bootstrap, fixtures, 5)
    m10 = _model(bootstrap, fixtures, PLANNING_EVENTS)

    for ev in events:
        a5 = float(m5._availability(ev).loc[doubtful])
        a10 = float(m10._availability(ev).loc[doubtful])
        assert a5 == pytest.approx(a10, abs=1e-9), (
            f"GW{ev}: horizon 5 says {a5:.4f}, horizon {PLANNING_EVENTS} says "
            f"{a10:.4f}. config.horizon is leaking into the calendar."
        )


def test_the_later_gameweeks_actually_heal(game_state, doubtful):
    """
    Guards against 'fixing' this by freezing every gameweek at the base value,
    which would satisfy both tests above and model nothing.
    """
    bootstrap, fixtures = game_state
    m = _model(bootstrap, fixtures, 5)
    events = X.next_events(bootstrap, PLANNING_EVENTS)

    first = float(m._availability(events[0]).loc[doubtful])
    sixth = float(m._availability(events[5]).loc[doubtful])
    assert sixth > first + 0.05, (
        f"GW{events[5]} availability {sixth:.4f} is barely above GW{events[0]}'s "
        f"{first:.4f}; decay_doubt is still not firing past the horizon"
    )


def test_the_squad_objective_is_untouched(game_state):
    """
    xp_horizon sums only the first config.horizon gameweeks, so widening the
    calendar view must not change what the optimiser maximises. If this ever
    fails, the fix has quietly changed squad selection.
    """
    bootstrap, fixtures = game_state
    m = _model(bootstrap, fixtures, 5)
    events = X.next_events(bootstrap, PLANNING_EVENTS)

    short = m.expected_points(events[:5])
    m._minutes_cache = {}
    long = m.expected_points(events)

    a = short.set_index("id")["xp_horizon"]
    b = long.set_index("id")["xp_horizon"].reindex(a.index)
    assert (a - b).abs().max() < 1e-9, (
        "scoring ten gameweeks changed xp_horizon, so the squad the optimiser "
        "picks now depends on how far the chip planner looks"
    )
