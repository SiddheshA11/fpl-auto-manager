"""
Transfers before the first deadline of the season.

FPL charges nothing for any number of transfers until Gameweek 1 kicks off.
That is the single week the whole squad can be rebuilt for free, and it is also
the week the bot got wrong: it read `(limit or 1) - made` off a transfers block
that carries no numeric limit while transfers are unlimited, concluded it had
one free transfer, and would have made one move and rolled the rest - leaving
the season's largest free gain on the table with no way to recover it.
"""
from __future__ import annotations

import copy

import pytest

import manager
import optimizer as O


def _events(first_id: int = 1, **overrides) -> dict:
    """A bootstrap carrying just the events the season-state check reads."""
    events = [
        {"id": first_id, "name": f"Gameweek {first_id}", "finished": False,
         "is_current": False, "is_next": True},
        {"id": first_id + 1, "name": f"Gameweek {first_id + 1}", "finished": False,
         "is_current": False, "is_next": False},
    ]
    events[0].update(overrides)
    return {"events": events}


def test_before_the_opening_kickoff_transfers_are_unlimited():
    assert manager.season_not_started(_events(), event_id=1) is True


def test_once_the_opening_gameweek_is_live_they_are_not():
    assert manager.season_not_started(_events(is_current=True), event_id=1) is False


def test_after_the_opening_gameweek_finishes_they_are_not():
    assert manager.season_not_started(_events(finished=True), event_id=1) is False


def test_no_later_gameweek_counts_as_pre_season():
    """The whole point is that this fires exactly once per season."""
    assert manager.season_not_started(_events(), event_id=2) is False


def test_an_empty_bootstrap_is_not_treated_as_pre_season():
    """Missing data must not unlock a fifteen-transfer rebuild."""
    assert manager.season_not_started({}, event_id=1) is False
    assert manager.season_not_started({"events": []}, event_id=1) is False


def test_the_check_does_not_assume_the_season_starts_at_one():
    """`is_next` and id 1 are not the same claim; the first event id is read."""
    assert manager.season_not_started(_events(first_id=1), event_id=1) is True
    assert manager.season_not_started(_events(first_id=1), event_id=2) is False


class _Pool:
    """A pool big enough to build two clearly different legal squads from."""

    @staticmethod
    def frame():
        import pandas as pd
        rows = []
        pid = 0
        for position, count in [(1, 6), (2, 16), (3, 16), (4, 10)]:
            for k in range(count):
                rows.append({
                    "id": pid,
                    "web_name": f"p{pid}",
                    "position": position,
                    "team": pid % 20 + 1,
                    "cost": 4.5,
                    # A clean split: the first half of each position is poor,
                    # the second half is good, so the optimal move is to
                    # replace the entire owned squad rather than part of it.
                    "xp_horizon": 1.0 if k < count // 2 else 10.0,
                    "xp_next": 0.2 if k < count // 2 else 2.0,
                    "selected_by_percent": 10.0,
                })
                pid += 1
        return pd.DataFrame(rows)


def test_free_transfers_are_not_charged_option_value_when_they_are_free():
    """
    The option value of a free transfer exists because using one spends
    flexibility that rolls over. Before the first deadline nothing rolls and
    nothing is spent, so charging it taxes a fifteen-player rebuild 4.5 xP it
    does not owe and biases the solver toward standing pat.
    """
    pool = _Pool.frame()
    bad = [int(i) for i in pool[pool["xp_horizon"] == 1.0]
           .groupby("position").head(5).head(15)["id"]]
    # Round the owned squad out to a legal 15 by position.
    owned = []
    for position, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        owned += [int(i) for i in pool[(pool["position"] == position)
                                       & (pool["xp_horizon"] == 1.0)].head(count)["id"]]
    assert len(owned) == 15

    opt = O.SquadOptimizer(pool, "xp_horizon", "xp_next")
    charged = opt.optimise_transfers(owned, bank=0.0, free_transfers=15, max_hits=0)
    free = opt.optimise_transfers(owned, bank=0.0, free_transfers=15, max_hits=0,
                                  free_transfer_value=0.0)

    assert len(free.transfers_in) >= len(charged.transfers_in)
    assert free.squad_xp >= charged.squad_xp
    assert free.hits == 0, "nothing is free about a rebuild that takes hits"


def test_preseason_run_rebuilds_the_squad_rather_than_making_one_move(monkeypatch):
    """
    End to end: the weekly run, executed before the opening kickoff against a
    squad of deliberately poor players, must replace far more than one of them.
    """
    from test_pipeline import StubClient, _snapshot

    bootstrap, fixtures = _snapshot("bootstrap-static"), _snapshot("fixtures")
    bootstrap = copy.deepcopy(bootstrap)

    # Force the snapshot into a pre-season state: nothing played, GW1 next.
    first = min(int(e["id"]) for e in bootstrap["events"])
    for e in bootstrap["events"]:
        e["finished"] = False
        e["is_current"] = False
        e["is_next"] = int(e["id"]) == first
        e["data_checked"] = False

    import pandas as pd
    import priors
    import xp_model as X

    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    scored = X.XPModel(bootstrap, fixtures, ps).expected_points(X.next_events(bootstrap, 5))
    available = scored[scored["status"].isin(["a", "d"])]

    # An owned squad drawn from the weakest legal players money can buy, so a
    # single transfer cannot possibly be the right answer.
    owned, costs = [], {}
    for position, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        worst = available[available["position"] == position].nsmallest(count * 3, "xp_horizon")
        for _, p in worst.head(count).iterrows():
            owned.append(int(p["id"]))
            costs[int(p["id"])] = float(p["cost"])
    assert len(owned) == 15

    client = StubClient(bootstrap, fixtures, owned, costs, bank=0.0, free_transfers=1)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)

    result = manager.run_weekly_cycle(dry_run=True)
    assert result is not None
    assert result["hits"] == 0, "pre-season transfers cost nothing"
    assert len(result["transfers_in"]) > 3, (
        f"only {len(result['transfers_in'])} transfers proposed before the first "
        f"deadline, when every transfer is free and the squad is deliberately awful"
    )
