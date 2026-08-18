"""
Choosing which gameweek to play a chip in.

A threshold answers the wrong question. "Is this bench worth 13 points" cannot
decide anything on its own, because a chip is a one-shot resource with an
expiry - what matters is whether this gameweek is the best remaining one for
it. The old code compared each chip against a fixed bar in isolation and would
happily spend bench boost on an ordinary week while a double gameweek sat two
weeks away.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

import chips
import priors
import xp_model as X

SNAPS = Path(__file__).resolve().parent.parent / "data" / "snapshots"
SQUAD_NAMES = ["Sels", "Tarkowski", "Thiaw", "Collins", "B.Fernandes", "Mbeumo",
               "Rice", "Gibbs-White", "Ndiaye", "Watkins", "Evanilson",
               "Sánchez", "Virgil", "Milenković", "Woltemade"]


def _state(fixtures=None, scoring=None):
    bs_files = sorted(SNAPS.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not bs_files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(bs_files[0])
    fx = fixtures if fixtures is not None else X.load_snapshot(
        sorted(SNAPS.glob("fixtures_*.json.gz"), reverse=True)[0])
    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    model = X.XPModel(bootstrap, fx, ps, X.ModelConfig(horizon=chips.PLANNING_HORIZON))
    events = X.next_events(bootstrap, scoring or chips.PLANNING_HORIZON)
    scored = model.expected_points(events)
    ids = [int(scored.loc[scored["web_name"] == n, "id"].iloc[0]) for n in SQUAD_NAMES
           if (scored["web_name"] == n).any()]
    return bootstrap, fx, scored, events, ids


def _with_double(fixtures, event):
    """Duplicate one gameweek's fixtures, making it a double."""
    out = copy.deepcopy(fixtures)
    extra = [f for f in out if f.get("event") == event]
    nid = max(f["id"] for f in out) + 1
    for f in extra:
        d = copy.deepcopy(f)
        d["id"] = nid
        nid += 1
        out.append(d)
    return out


def test_a_chip_is_held_for_a_double_gameweek_ahead_of_it():
    """The behaviour the whole planner exists to produce."""
    base_bs, base_fx, *_ = _state()
    doubled = _with_double(base_fx, 6)
    bootstrap, fx, scored, events, ids = _state(doubled)
    assert len(ids) == 15, "snapshot no longer contains the reference squad"

    engine = chips.ChipEngine(bootstrap, fx, scored)
    horizon = chips.plan_horizon(bootstrap, events[0], {"bboost", "3xc"})
    values = engine.value_by_gameweek(events[0], {"bboost", "3xc"}, ids, horizon)

    assert values[("bboost", 6)] > 1.5 * values[("bboost", events[0])], (
        "a double gameweek must be worth substantially more than an ordinary one"
    )

    decision = engine.evaluate(events[0], {"chips": []}, xi_ids=ids[:11],
                               bench_ids=ids[11:], captain_id=ids[4], squad_ids=ids)
    assert decision.chip is None, "must not spend the chip this week"
    assert "GW6" in decision.reason and "holding" in decision.reason


def test_an_ordinary_run_of_gameweeks_plays_nothing():
    """Without a double in view, no chip is worth committing."""
    bootstrap, fx, scored, events, ids = _state()
    engine = chips.ChipEngine(bootstrap, fx, scored)
    decision = engine.evaluate(events[0], {"chips": []}, xi_ids=ids[:11],
                               bench_ids=ids[11:], captain_id=ids[4], squad_ids=ids)
    assert decision.chip is None


def test_chips_compete_for_gameweeks_rather_than_being_chosen_alone():
    """
    FPL allows one chip per gameweek, so triple captain on the single best
    week displaces bench boost from it. Choosing each chip independently would
    put both on the same gameweek and silently drop one.
    """
    values = {
        ("bboost", 5): 30.0, ("bboost", 6): 12.0,
        ("3xc", 5): 25.0, ("3xc", 6): 4.0,
    }
    assignment = chips.solve_assignment(values, ["bboost", "3xc"], [5, 6])
    assert len(set(assignment.values())) == len(assignment), "two chips in one gameweek"

    # Both chips want GW5, so one must yield - and it is decided by opportunity
    # cost, not by which raw value is larger. Bench boost is worth more there
    # in absolute terms (30 vs 25), but it loses less by moving: 30 -> 12 costs
    # 18, while 3xc dropping to GW6 costs 21. Total is maximised at 25 + 12 =
    # 37 rather than 30 + 4 = 34. Choosing each chip by its own best gameweek
    # would put both on GW5 and silently drop one.
    assert assignment["3xc"] == 5 and assignment["bboost"] == 6
    total = sum(values[(c, gw)] for c, gw in assignment.items())
    assert total == pytest.approx(37.0)


def test_a_chip_nobody_should_play_is_left_unassigned():
    """Skipping must be a legal branch, not the least-bad slot."""
    assert chips.solve_assignment({("bboost", 5): -3.0}, ["bboost"], [5, 6]) == {}
    assert chips.solve_assignment({}, ["bboost", "3xc"], [5, 6]) == {}


def test_the_commit_floor_falls_to_nothing_at_the_end_of_the_window():
    """
    The floor exists because the window runs past the planning horizon and an
    unseen double beats the best mediocre week in view. On the final gameweek
    there is no unseen future left, so holding stops being an option.
    """
    bootstrap, *_ = _state()
    early = chips.effective_threshold(bootstrap, "bboost", 1)
    late = chips.effective_threshold(bootstrap, "bboost", 19)
    assert early > late == pytest.approx(chips.THRESHOLDS["bboost"])


def test_scoring_further_ahead_does_not_change_the_squad_objective():
    """
    The chip planner needs to see ten gameweeks; the squad is chosen on five.
    Scoring the longer list must not widen `xp_horizon`, or the optimiser
    silently starts maximising something else - a squad built for ten
    gameweeks, chosen by a function whose comments all say five.
    """
    bootstrap, fx, _, _, _ = _state()
    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    model = X.XPModel(bootstrap, fx, ps, X.ModelConfig(horizon=5))

    short = model.expected_points(X.next_events(bootstrap, 5))
    long = X.XPModel(bootstrap, fx, ps, X.ModelConfig(horizon=5)).expected_points(
        X.next_events(bootstrap, 10))

    a = short.set_index("id")["xp_horizon"]
    b = long.set_index("id")["xp_horizon"].reindex(a.index)
    assert (a - b).abs().max() < 1e-9, "xp_horizon changed when more gameweeks were scored"

    # ...and the longer run really does carry the extra gameweeks.
    assert "xp_gw" + str(X.next_events(bootstrap, 10)[-1]) in long.columns
    assert "xp_gw" + str(X.next_events(bootstrap, 10)[-1]) not in short.columns


def test_the_planner_sees_the_whole_planning_horizon():
    """PLANNING_HORIZON claimed ten gameweeks while only five were scored."""
    bootstrap, fx, scored, events, ids = _state()
    engine = chips.ChipEngine(bootstrap, fx, scored)
    horizon = chips.plan_horizon(bootstrap, events[0], {"bboost", "3xc"})
    values = engine.value_by_gameweek(events[0], {"bboost", "3xc"}, ids, horizon)
    valued = {gw for (_, gw) in values}
    assert len(valued) >= min(chips.PLANNING_HORIZON, len(horizon)), (
        f"only {len(valued)} of {len(horizon)} gameweeks in the horizon were valued"
    )
