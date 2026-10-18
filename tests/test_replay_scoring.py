"""
simulate.pick_xi against FPL's own scoring of real gameweeks.

Every decision-replay number rests on pick_xi - autosubs, formation legality,
captain and vice - and it was checked only against hand-written cases, which
share whatever misreading of the rules the code has. These are 569 real
gameweeks (my team and my mini-league rivals, GW1-6 of 2026-27, anonymised)
with the points FPL itself awarded. See fixtures/fetch_scoring_cases.py.

Bench boost and triple captain are excluded because the replay plays no
chips; wildcard and free hit score normally and stay in.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import simulate

CASES = json.loads(gzip.decompress(
    (Path(__file__).parent / "fixtures" / "scoring_cases.json.gz").read_bytes()))
SCORED = [c for c in CASES["cases"] if c["active_chip"] not in ("bboost", "3xc")]


def _named_lineup(case) -> list[dict]:
    """
    The lineup as the manager named it at the deadline. FPL's picks endpoint
    returns it after autosubs, with the player who came on in the starter's
    shirt, so a fixture read naively hands pick_xi a lineup whose subs are
    already made and tests nothing. Undo FPL's swaps, last first.
    """
    shirt = {p["element"]: p["position"] for p in case["picks"]}
    for came_on, went_off in reversed(case["automatic_subs"]):
        shirt[came_on], shirt[went_off] = shirt[went_off], shirt[came_on]
    return sorted(({**p, "position": shirt[p["element"]]} for p in case["picks"]),
                  key=lambda p: p["position"])


def _replay(case):
    """(points, the players pick_xi brought on)."""
    live = CASES["live"][str(case["event"])]
    picks = _named_lineup(case)
    stat = {p["element"]: live[str(p["element"])] for p in picks}
    # pick_xi names its XI, bench order and armband from an xP ranking, so
    # encode the manager's real choices as one: captain, vice, the other
    # starters in shirt order, then the bench in shirt order.
    xp = {}
    for p in picks:
        if p["is_captain"]:
            xp[p["element"]] = 100.0
        elif p["is_vice_captain"]:
            xp[p["element"]] = 99.0
        elif p["position"] <= 11:
            xp[p["element"]] = 50.0 - p["position"]
        else:
            xp[p["element"]] = 16.0 - p["position"]
    squad = [p["element"] for p in picks]
    xi, gross, _ = simulate.pick_xi(
        squad,
        {i: s["position"] for i, s in stat.items()},
        {i: s["total_points"] for i, s in stat.items()},
        {i: s["minutes"] for i, s in stat.items()},
        xp,
    )
    named_xi = {p["element"] for p in picks if p["position"] <= 11}
    return gross, set(xi) - named_xi


def test_the_fixture_exercises_the_rules_it_is_for():
    assert len(SCORED) > 400
    assert sum(bool(c["automatic_subs"]) for c in SCORED) > 150
    assert sum(len(c["automatic_subs"]) >= 2 for c in SCORED) > 20


def test_undoing_fpls_swaps_puts_the_benched_starter_back_in_the_xi():
    case = next(c for c in SCORED if c["automatic_subs"])
    came_on, went_off = case["automatic_subs"][0]
    shirt = {p["element"]: p["position"] for p in _named_lineup(case)}
    assert shirt[went_off] <= 11 < shirt[came_on]


def test_every_real_gameweek_scores_what_fpl_awarded():
    wrong = [(c["event"], len(c["automatic_subs"]), _replay(c)[0], c["fpl_points"])
             for c in SCORED if _replay(c)[0] != c["fpl_points"]]
    assert not wrong, f"{len(wrong)}/{len(SCORED)} differ (gw, autosubs, ours, fpl): {wrong[:10]}"


def test_pick_xi_makes_exactly_the_substitutions_fpl_made():
    wrong = [(c["event"], sorted(_replay(c)[1]), sorted(i for i, _ in c["automatic_subs"]))
             for c in SCORED if _replay(c)[1] != {i for i, _ in c["automatic_subs"]}]
    assert not wrong, f"{len(wrong)}/{len(SCORED)} differ (gw, ours, fpl): {wrong[:10]}"
