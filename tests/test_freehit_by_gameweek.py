"""
Free hit has to be valued in every gameweek the chip planner can see.

manager.py passed the engine a free-hit value for the current gameweek only,
so the planner could never hold the chip for a blank or a double: it was
played this week or not at all. This records what ChipEngine.evaluate was
actually handed through a real `run_weekly_cycle` dry run - a helper that
computes the right numbers proves nothing if the run never passes them.

The committed snapshot is pre-season and the free-hit window opens at GW2, so
the calendar is advanced to GW3 here; on GW1 the chip is correctly unvalued.
"""
from __future__ import annotations

import copy

import pytest

import chips
import manager
from test_pipeline import StubClient, game_state, starting_squad  # noqa: F401 - fixtures

NEXT = 3


@pytest.fixture
def recorded(monkeypatch):
    seen: dict = {}
    real = chips.ChipEngine

    class Recording(real):
        def evaluate(self, event, my_team, **kw):
            seen["kwargs"] = kw
            out = super().evaluate(event, my_team, **kw)
            seen["plan"] = self.last_plan
            return out

    monkeypatch.setattr(manager.chips, "ChipEngine", Recording)
    return seen


def _stub(game_state, starting_squad, monkeypatch, chips_played=None):  # noqa: F811
    bootstrap, fixtures = game_state
    bootstrap = copy.deepcopy(bootstrap)
    for e in bootstrap["events"]:
        e["finished"] = e["id"] < NEXT
        e["is_current"] = e["id"] == NEXT - 1
        e["is_next"] = e["id"] == NEXT
    costs = dict(zip(starting_squad.squad["id"].astype(int), starting_squad.squad["cost"]))
    client = StubClient(bootstrap, fixtures, [int(i) for i in starting_squad.squad["id"]], costs,
                        chips_played=chips_played)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)
    return client


def test_free_hit_is_valued_beyond_this_gameweek(game_state, starting_squad, monkeypatch, recorded):  # noqa: F811
    _stub(game_state, starting_squad, monkeypatch)
    assert manager.run_weekly_cycle(dry_run=True) is not None

    best = recorded["kwargs"].get("best_xi_xp")
    assert best, "manager.py handed the chip engine no per-gameweek free-hit values"
    assert NEXT in best and max(best) > NEXT, f"only valued {sorted(best)}"

    plan = recorded["plan"]
    future = [gw for (chip, gw) in plan.values if chip == "freehit" and gw > NEXT]
    assert future, "the planner still sees free hit in the current gameweek alone"


def test_no_free_hit_solves_once_the_chip_is_spent(game_state, starting_squad, monkeypatch, recorded):  # noqa: F811
    """Ten extra solves buy nothing when there is no free hit to place."""
    _stub(game_state, starting_squad, monkeypatch,
          chips_played=[{"name": "freehit", "event": NEXT - 1}])
    assert manager.run_weekly_cycle(dry_run=True) is not None
    assert recorded["kwargs"].get("best_xi_xp") is None
