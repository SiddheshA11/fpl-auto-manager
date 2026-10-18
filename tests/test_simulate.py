"""
The season simulator's own arithmetic.

Every strategy comparison is measured through this file, so a bug here does
not produce a wrong answer - it produces a confident wrong answer about which
optimiser to ship. The three rules worth pinning are the ones that decide how
many points a squad is credited with: what a player raises when sold, which
eleven actually scored after autosubs, and how the free-transfer allowance
accrues.
"""
from __future__ import annotations

import pandas as pd
import pytest

import simulate as S


# ─────────────────────────── selling prices ───────────────────────────


@pytest.mark.parametrize("purchase,current,expected", [
    (5.0, 5.0, 5.0),      # unchanged
    (5.0, 4.5, 4.5),      # a fall is taken in full
    (5.0, 5.2, 5.1),      # half of a rise, rounded down
    (5.0, 5.1, 5.0),      # a single-tenth rise returns nothing
    (5.0, 5.4, 5.2),
    (5.0, 5.5, 5.2),      # 0.5 rise -> 0.25 -> rounds down to 0.2
    (10.0, 11.0, 10.5),
])
def test_selling_price_returns_half_the_rise(purchase, current, expected):
    assert S.selling_price(purchase, current) == pytest.approx(expected, abs=1e-9)


def test_selling_price_is_never_generous():
    """
    Overstating what a sale raises funds transfers with money that does not
    exist, and flatters whichever strategy churns the squad hardest - which is
    exactly the comparison this harness is built to make.
    """
    for purchase in (4.0, 5.5, 9.0, 14.0):
        for tenths in range(-20, 21):
            current = round(purchase + tenths / 10.0, 1)
            if current <= 0:
                continue
            assert S.selling_price(purchase, current) <= max(purchase, current) + 1e-9
            assert S.selling_price(purchase, current) <= current + 1e-9


# ──────────────────────────────── autosubs ────────────────────────────────


def _squad(positions: dict[int, int]) -> list[int]:
    return list(positions)


POSITIONS = {
    1: 1, 2: 1,                          # keepers
    3: 2, 4: 2, 5: 2, 6: 2, 7: 2,        # defenders
    8: 3, 9: 3, 10: 3, 11: 3, 12: 3,     # midfielders
    13: 4, 14: 4, 15: 4,                 # forwards
}
SQUAD = list(POSITIONS)


def test_a_starter_who_blanks_is_replaced_from_the_bench():
    xp = {p: 10.0 - p * 0.1 for p in SQUAD}      # ranks 1,2,3,... descending
    minutes = {p: 90 for p in SQUAD}
    points = {p: 2 for p in SQUAD}
    # The lowest-ranked starter fails to appear.
    xi_before, _, _ = S.pick_xi(SQUAD, POSITIONS, points, minutes, xp)
    victim = xi_before[-1]
    minutes[victim] = 0
    points[victim] = 0

    xi, gross, _ = S.pick_xi(SQUAD, POSITIONS, points, minutes, xp)
    assert victim not in xi, "a player who did not appear was left in the scoring eleven"
    assert all(minutes[p] > 0 for p in xi)


def test_autosub_never_produces_an_illegal_shape():
    """A sub that would leave two keepers or two defenders must not be made."""
    xp = {p: 10.0 - p * 0.1 for p in SQUAD}
    minutes = {p: 90 for p in SQUAD}
    points = {p: 2 for p in SQUAD}
    # Every defender in the eleven blanks; the bench cannot legally cover them all.
    for p in (3, 4, 5, 6, 7):
        minutes[p] = 0
        points[p] = 0
    xi, gross, _ = S.pick_xi(SQUAD, POSITIONS, points, minutes, xp)
    counts = {t: sum(1 for p in xi if POSITIONS[p] == t) for t in (1, 2, 3, 4)}
    assert len(xi) == 11
    assert counts[1] == 1, "the eleven must contain exactly one keeper"
    assert counts[2] >= 3, "fewer than three defenders is not a legal shape"
    assert counts[4] >= 1


def test_the_captain_is_doubled_and_the_vice_takes_over_on_a_blank():
    xp = {p: 1.0 for p in SQUAD}
    xp[8] = 9.0      # captain
    xp[9] = 8.0      # vice
    minutes = {p: 90 for p in SQUAD}
    points = {p: 2 for p in SQUAD}
    points[8], points[9] = 10, 6

    xi, gross, armband = S.pick_xi(SQUAD, POSITIONS, points, minutes, xp)
    assert armband == 8
    assert gross == sum(points[p] for p in xi) + 10

    minutes[8], points[8] = 0, 0
    xi, gross, armband = S.pick_xi(SQUAD, POSITIONS, points, minutes, xp)
    assert armband == 9, "the vice must take the armband when the captain does not play"
    assert gross == sum(points[p] for p in xi) + 6


def test_nobody_is_doubled_when_captain_and_vice_both_blank():
    """
    FPL's rule: the named captain, else the named vice, else no armband. The
    armband used to go to the best starter AFTER autosubs, so with both
    blanking and subbed off it doubled whoever ranked third - points FPL
    never awards.
    """
    xp = {p: 1.0 for p in SQUAD}
    xp[8], xp[9], xp[10] = 9.0, 8.0, 7.0
    minutes = {p: 90 for p in SQUAD}
    points = {p: 2 for p in SQUAD}
    minutes[8] = minutes[9] = 0
    points[8] = points[9] = 0
    points[10] = 12

    xi, gross, armband = S.pick_xi(SQUAD, POSITIONS, points, minutes, xp)
    assert 8 not in xi and 9 not in xi, "both blanks should have been subbed off"
    assert gross == sum(points[p] for p in xi), "a third player was doubled"


# ─────────────────────── the harness actually runs ───────────────────────


def test_a_short_season_runs_and_scores(tmp_path):
    """
    End-to-end. A harness that imports but cannot complete a gameweek is worth
    nothing, and the first full run died at GW34 on a squad the solver could
    not keep - which no unit test above would have caught.
    """
    if not (S.HISTORY_DIR / "2025-26").exists():
        pytest.skip("no 2025-26 history on disk")
    res = S.simulate_season("2025-26", 10, 13, strategy="greedy", horizon=3)
    f = res.frame
    assert len(f) == 4
    assert (f["gross"] > 0).all(), "a gameweek scored nothing, which means no eleven was picked"
    assert (f["transfers"] >= 0).all()
    assert (f["ft"].between(0, S.MAX_BANKED_TRANSFERS)).all(), "free transfers escaped the cap"
    assert res.total == int(f["net"].sum())


# ------------------------------------------------- forking a replay mid-season
#
# The decision-replay harness runs a candidate arm from a copy of the
# baseline's state at some gameweek. That is only sound if the state is
# complete - nothing a later gameweek depends on lives outside SquadState - and
# if play_gameweek never mutates the state it was handed. Both are checked on
# a real season rather than a stub.

FORK_SEASON = "2025-26"


@pytest.fixture(scope="module")
def season_data():
    import priors
    if FORK_SEASON not in priors.available_seasons():
        pytest.skip(f"{FORK_SEASON} not on disk")
    return S.load_season(FORK_SEASON)


def _keys(rows):
    return [{k: r[k] for k in ("gw", "net", "hits", "transfers", "bank", "ft",
                               "captain", "ins", "outs", "xi")} for r in rows]


def test_a_fork_continues_exactly_as_the_unforked_run(season_data):
    scored = {gw: S.score_gameweek(season_data, gw) for gw in range(2, 9)}

    straight, state = [], S.SquadState()
    for gw in range(2, 9):
        state, row = S.play_gameweek(season_data, state, gw, scored[gw])
        straight.append(row)

    forked, state = [], S.SquadState()
    for gw in range(2, 9):
        if gw == 5:
            state = state.copy()
        state, row = S.play_gameweek(season_data, state, gw, scored[gw])
        forked.append(row)

    assert _keys(straight) == _keys(forked)
    assert sum(r["transfers"] for r in straight[1:]) > 0, "the window must exercise transfers"


def test_play_gameweek_leaves_the_state_it_was_handed_untouched(season_data):
    scored = {gw: S.score_gameweek(season_data, gw) for gw in (2, 3)}
    state, _ = S.play_gameweek(season_data, S.SquadState(), 2, scored[2])
    before = state.copy()
    S.play_gameweek(season_data, state, 3, scored[3])
    assert state == before


def test_a_state_copy_carries_every_field_and_shares_nothing():
    """
    A fork is only as good as the copy. Checked field by field from the
    dataclass itself, so a field added later is covered without editing this.
    """
    import dataclasses

    state = S.SquadState(squad=[1, 2], purchase={1: 5.0}, last_price={1: 5.5, 3: 4.0},
                         bank=1.3, free_transfers=3)
    clone = state.copy()
    for f in dataclasses.fields(S.SquadState):
        assert getattr(clone, f.name) == getattr(state, f.name), f.name
        if isinstance(getattr(state, f.name), (list, dict)):
            assert getattr(clone, f.name) is not getattr(state, f.name), f"{f.name} is shared"
