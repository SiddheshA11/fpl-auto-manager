"""
Choosing the eleven who start.

Which fifteen to own is a horizon decision - you keep them for weeks. Which
eleven to start is a decision about one gameweek. The joint solve answered both
with the horizon column, so a defender worth 3.95 points this Saturday sat on
the bench behind a midfielder worth 3.46 who was worth more across the next
five. Half a point a gameweek, thirty-eight gameweeks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optimizer as O


def _squad(next_gw: dict[int, float] | None = None,
           horizon: dict[int, float] | None = None) -> pd.DataFrame:
    """
    A legal owned fifteen. Value descends with id inside each position, and
    `xp_next` mirrors `xp_horizon` unless overridden, so a caller can make one
    player better this week than his horizon rank suggests.

    Both columns are overridable because the positions hold different numbers
    of players - five defenders but only three forwards - so the formula alone
    cannot make a chosen player the worst outfielder in the squad.
    """
    rows = []
    pid = 0
    for position, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        for k in range(count):
            base = 20.0 - k
            rows.append({
                "id": pid,
                "web_name": f"p{pid}",
                "position": position,
                "team": pid + 1,
                "cost": 5.0,
                "selected_by_percent": 10.0,
                "xp_horizon": base,
                "xp_next": base / 5.0,
            })
            pid += 1
    df = pd.DataFrame(rows)
    for pid_, value in (horizon or {}).items():
        df.loc[df["id"] == pid_, ["xp_horizon", "xp_next"]] = [value, value / 5.0]
    for pid_, value in (next_gw or {}).items():
        df.loc[df["id"] == pid_, "xp_next"] = value
    return df


def test_lineup_is_chosen_on_the_coming_gameweek_not_the_horizon():
    """
    Player 14 is the worst forward over five gameweeks and would be benched,
    but is the best player in the squad this week. He has to start.
    """
    squad = _squad(horizon={14: 1.0}, next_gw={14: 99.0})

    # captain_col is the horizon too, so the boosted week value cannot leak
    # into the horizon solve through the armband and drag him into the XI.
    horizon_pick = O.SquadOptimizer(squad, "xp_horizon", "xp_horizon").build_squad(budget=75.0)
    week_pick = O.SquadOptimizer(squad, "xp_next", "xp_next").optimise_lineup()

    assert 14 not in set(horizon_pick.xi["id"]), "setup is wrong: he should be benched on horizon"
    assert 14 in set(week_pick.xi["id"])
    assert week_pick.captain == 14, "the best player this week should also take the armband"
    assert week_pick.xi["xp_next"].sum() > horizon_pick.xi["xp_next"].sum()


def test_lineup_solve_keeps_the_squad_intact():
    """It reorders the fifteen; it must never change who they are."""
    squad = _squad(horizon={14: 1.0}, next_gw={14: 99.0})
    sol = O.SquadOptimizer(squad, "xp_next", "xp_next").optimise_lineup()
    assert set(sol.squad["id"]) == set(squad["id"])
    assert len(sol.xi) == O.XI_SIZE
    assert len(sol.bench) == O.SQUAD_SIZE - O.XI_SIZE


def test_lineup_respects_the_formation_rules():
    squad = _squad(horizon={14: 1.0}, next_gw={14: 99.0})
    sol = O.SquadOptimizer(squad, "xp_next", "xp_next").optimise_lineup()
    counts = sol.xi["position"].value_counts().to_dict()
    for position, (lo, hi) in O.XI_BOUNDS.items():
        assert lo <= counts.get(position, 0) <= hi
    assert sol.bench.iloc[0]["position"] == 1, "the reserve keeper leads the bench"


def test_lineup_solve_refuses_a_squad_that_is_not_fifteen():
    """
    Guards against being handed a filtered pool by mistake, which would
    silently return a 'lineup' drawn from the whole league.
    """
    squad = _squad()
    with pytest.raises(ValueError, match="exactly the owned"):
        O.SquadOptimizer(squad.head(14), "xp_next", "xp_next").optimise_lineup()


def test_captain_is_the_best_player_of_the_week():
    squad = _squad(next_gw={7: 50.0})
    sol = O.SquadOptimizer(squad, "xp_next", "xp_next").optimise_lineup()
    assert sol.captain == 7
    assert sol.vice_captain != sol.captain
    assert sol.vice_captain in set(sol.xi["id"]), "the vice must be someone who is starting"
