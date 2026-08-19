"""
Multi-gameweek transfer sequencing.

The point of a sequencer is that it makes a decision this week that a
one-step optimiser cannot: rolling a transfer now so two can be made next
week, or timing a hit to land in the gameweek that pays for it. So the test
that matters is a constructed scenario where the greedy optimiser demonstrably
takes the worse option and the sequencer does not - not a test that the
free-transfer arithmetic computes 2 when handed 1, which would pass on a
sequencer nothing ever called.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optimizer as O
import sequence as S

EVENTS = [10, 11, 12]


def _pool(per_gw: dict[int, dict[int, float]] | None = None,
          base: float = 2.0, cost: float = 4.5) -> pd.DataFrame:
    """
    A pool large enough for a legal 15, with per-gameweek xP overridable per
    player. `per_gw` maps player id -> {event: xp}.

    Costs are uniform so the budget never binds and every difference the tests
    observe is caused by the sequencing decision rather than by affordability.
    """
    per_gw = per_gw or {}
    rows = []
    pid = 0
    for position, count in [(1, 6), (2, 16), (3, 16), (4, 10)]:
        for _ in range(count):
            row = {
                "id": pid, "web_name": f"p{pid}", "position": position,
                "team": pid % 20 + 1, "cost": cost, "status": "a",
                "selected_by_percent": 10.0,
            }
            for ev in EVENTS:
                row[f"xp_gw{ev}"] = per_gw.get(pid, {}).get(ev, base)
            row["xp_next"] = row[f"xp_gw{EVENTS[0]}"]
            row["xp_horizon"] = sum(row[f"xp_gw{ev}"] for ev in EVENTS)
            rows.append(row)
            pid += 1
    return pd.DataFrame(rows)


def _legal_squad(pool: pd.DataFrame) -> list[int]:
    return [int(i) for i in O.SquadOptimizer(pool, "xp_horizon", "xp_next")
            .build_squad(100.0).squad["id"]]


# ─────────────────────── the free-transfer arithmetic ───────────────────────


@pytest.mark.parametrize("ft,used,expected", [
    (1, 0, 2),        # roll one, hold two
    (1, 1, 1),        # spend it, accrue the next
    (2, 2, 1),
    (5, 0, 5),        # the cap holds; a sixth is never banked
    (5, 1, 5),
    (5, 5, 1),
    (1, 3, 1),        # two of those three were hits; the allowance still ticks up
])
def test_banking_follows_the_fpl_rule(ft, used, expected):
    assert S._next_ft(ft, used) == expected


def test_restrict_pool_never_drops_an_owned_player():
    """
    Trimming an owned player makes him unkeepable, so the solver is forced to
    transfer him out - and enough of those at once make the free-transfer floor
    unsatisfiable. That aborted the first full season simulation at GW34.
    """
    pool = _pool()
    squad = _legal_squad(pool)
    # Make the owned squad the *worst* players in the pool, so any trim that
    # ranks purely on value would discard all fifteen.
    pool.loc[pool["id"].isin(squad), "xp_horizon"] = -99.0
    trimmed = S.restrict_pool(pool, squad, keep=20)
    assert set(squad) <= set(trimmed["id"]), "an owned player was trimmed from the pool"
    assert len(trimmed) >= 20


# ──────────────────────────── banking a transfer ────────────────────────────


def _banking_scenario():
    """
    A pool where rolling this week is strictly better than moving.

    Nothing on offer improves the squad in GW10. In GW11 two players become
    worth a great deal - more than the horizon can recover by buying one of
    them early, because they are worth nothing at all until then. So:

      greedy      buys one now (or churns), holds 1 FT into GW11, and can only
                  take the second by paying -4.
      sequenced   rolls, holds 2 FT into GW11, and takes both for free.
    """
    pool = _pool()
    squad = _legal_squad(pool)
    outsiders = [int(i) for i in pool["id"] if int(i) not in squad]
    # Two midfielders outside the squad, worthless now and enormous later.
    targets = [p for p in outsiders if pool.loc[pool["id"] == p, "position"].iloc[0] == 3][:2]
    for t in targets:
        pool.loc[pool["id"] == t, "xp_gw10"] = 0.0
        pool.loc[pool["id"] == t, "xp_gw11"] = 30.0
        pool.loc[pool["id"] == t, "xp_gw12"] = 30.0
        pool.loc[pool["id"] == t, "xp_next"] = 0.0
        pool.loc[pool["id"] == t, "xp_horizon"] = 60.0
    return pool, squad, targets


def test_sequencer_rolls_a_transfer_to_fund_a_double_move():
    pool, squad, targets = _banking_scenario()
    selling = {p: 4.5 for p in squad}

    plan = S.plan_by_enumeration(pool, squad, bank=0.0, free_transfers=1,
                                 selling=selling, events=EVENTS, decay=1.0, max_hits=2)

    assert plan.schedule[0] == 0, (
        f"sequencer moved in GW10 (schedule {plan.schedule}); rolling is worth more, "
        "because both targets are worthless until GW11 and two free transfers take both"
    )
    assert plan.banked, "a roll that funds a later move must be reported as banking"
    assert sum(plan.schedule[1:]) >= 2, f"never spent the banked transfer: {plan.schedule}"
    assert not plan.solution.transfers_in, "a rolled gameweek must submit no transfers"


def test_greedy_does_not_roll_in_the_same_scenario():
    """
    The control. Without this the banking test could pass on a sequencer that
    simply never transfers, and would prove nothing about sequencing.
    """
    pool, squad, targets = _banking_scenario()
    selling = {p: 4.5 for p in squad}
    greedy = O.SquadOptimizer(pool, "xp_horizon", "xp_next").optimise_transfers(
        squad, bank=0.0, free_transfers=1, selling_prices=selling,
        max_hits=2, free_transfer_value=0.0, horizon_weight=1.0,
    )
    assert greedy.transfers_in, (
        "greedy held too - the scenario does not separate the two strategies, "
        "so the banking test above proves nothing"
    )


def test_joint_solver_also_rolls_in_the_banking_scenario():
    pool, squad, targets = _banking_scenario()
    selling = {p: 4.5 for p in squad}
    plan = S.plan_jointly(pool, squad, bank=0.0, free_transfers=1,
                          selling=selling, events=EVENTS, decay=1.0, max_hits=2)
    assert plan.schedule[0] == 0, f"joint solver moved in GW10: {plan.schedule}"
    assert not plan.solution.transfers_in


# ───────────────────────────── the plan is legal ─────────────────────────────


@pytest.mark.parametrize("solver", ["enumerate", "joint"])
def test_plan_returns_a_legal_fifteen(solver):
    pool = _pool()
    squad = _legal_squad(pool)
    selling = {p: 4.5 for p in squad}
    fn = S.plan_by_enumeration if solver == "enumerate" else S.plan_jointly
    plan = fn(pool, squad, bank=0.0, free_transfers=1, selling=selling,
              events=EVENTS, decay=0.84, max_hits=2)
    sq = plan.solution.squad
    assert len(sq) == O.SQUAD_SIZE
    assert sq["position"].value_counts().to_dict() == {2: 5, 3: 5, 4: 3, 1: 2}
    assert sq["team"].value_counts().max() <= O.MAX_PER_CLUB
    assert len(plan.solution.xi) == O.XI_SIZE
    assert len(plan.schedule) == len(EVENTS)


@pytest.mark.parametrize("solver", ["enumerate", "joint"])
def test_transfers_in_and_out_balance(solver):
    pool = _pool()
    squad = _legal_squad(pool)
    selling = {p: 4.5 for p in squad}
    fn = S.plan_by_enumeration if solver == "enumerate" else S.plan_jointly
    plan = fn(pool, squad, bank=0.0, free_transfers=2, selling=selling,
              events=EVENTS, decay=0.84, max_hits=2)
    sol = plan.solution
    assert len(sol.transfers_in) == len(sol.transfers_out)
    assert not set(sol.transfers_in) & set(squad)
    assert set(sol.transfers_out) <= set(squad)


def test_a_hit_is_only_taken_when_it_pays():
    """
    Sequencing must not become a licence to churn. With one free transfer and
    nothing worth buying, neither solver may take a hit.
    """
    pool = _pool()
    squad = _legal_squad(pool)
    selling = {p: 4.5 for p in squad}
    for fn in (S.plan_by_enumeration, S.plan_jointly):
        plan = fn(pool, squad, bank=0.0, free_transfers=1, selling=selling,
                  events=EVENTS, decay=0.84, max_hits=2)
        assert plan.solution.hits == 0, f"{fn.__name__} took a hit for nothing"
