"""
Reading the prediction ledger back against what actually happened.

The anchor test uses GW4 of 2026-27, recovered from the real run log before it
expired: the model gave Davis 2.81 xP, benched him, and he scored 14. Those
figures were computed by hand during the post-mortem - XI 70, best legal XI 82,
lineup loss 12, captain loss 5 - and this asserts the reader reproduces them
independently.

No test here touches the network. The gameweek's actuals are frozen in the
fixture, so the numbers cannot drift underneath the assertions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import calibrate

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gw04_ledger.json"


@pytest.fixture(scope="module")
def gw4():
    data = json.loads(FIXTURE.read_text())
    return data["record"], {int(k): v for k, v in data["actuals"].items()}


# ------------------------------------------------------- the real gameweek

def test_it_reproduces_the_gw4_post_mortem(gw4):
    record, act = gw4
    r = calibrate.assess(record, act)

    assert r["event"] == 4
    assert r["xi_actual"] == 70
    assert r["best_xi"] == 82
    assert r["lineup_loss"] == 12
    assert r["bench_points"] == 15
    assert r["captain_points"] == 9
    assert r["captain_loss"] == 5


def test_it_names_the_benched_player_who_returned(gw4):
    """
    The whole purpose: not "the model was wrong" but "it gave Davis 2.81 and
    benched him, and he scored 14".
    """
    record, act = gw4
    r = calibrate.assess(record, act)

    assert r["misranked"], "a 14-point benched defender must be surfaced"
    scored, name, xp, _p_start = r["misranked"][0]
    assert name == "Davis" and scored == 14
    assert xp == pytest.approx(2.81)


def test_the_best_legal_xi_respects_the_formation(gw4):
    """
    A hindsight XI that fields two keepers or six defenders is not a benchmark,
    it is a bug that would make every lineup look terrible.
    """
    record, act = gw4
    pts = {int(p["id"]): act.get(int(p["id"]), {}).get("total_points", 0)
           for p in record["players"]}
    total, xi = calibrate.best_legal_xi(record["players"], pts)

    assert len(xi) == 11
    pos = {int(p["id"]): int(p["position"]) for p in record["players"]}
    counts = {k: sum(1 for i in xi if pos[i] == k) for k in (1, 2, 3, 4)}
    assert counts[1] == 1, "exactly one goalkeeper"
    assert 3 <= counts[2] <= 5 and 2 <= counts[3] <= 5 and 1 <= counts[4] <= 3
    assert total == 82


def test_the_best_xi_is_never_worse_than_the_one_that_was_picked(gw4):
    record, act = gw4
    r = calibrate.assess(record, act)
    assert r["best_xi"] >= r["xi_actual"]
    assert r["lineup_loss"] >= 0


# --------------------------------------------------------------- arithmetic

def _rec(players, captain, event=7):
    return {"schema": 1, "event": event, "captain": captain, "vice_captain": None,
            "chip": None, "hits": 0, "transfers_in": [], "transfers_out": [],
            "players": players}


def _squad(points_by_id, started_ids, captain):
    """A legal 15: 2 GK, 5 DEF, 5 MID, 3 FWD."""
    layout = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
    players = []
    for i, pos in enumerate(layout, start=1):
        players.append({"id": i, "name": f"p{i}", "position": pos, "xp_next": 4.0,
                        "p_start": 0.9, "started": i in started_ids,
                        "is_captain": i == captain, "is_vice_captain": False})
    act = {i: {"total_points": points_by_id.get(i, 0), "minutes": 90} for i in range(1, 16)}
    return _rec(players, captain), act


def test_a_perfect_lineup_loses_nothing():
    # Start the top scorers in a legal shape: GK 1, DEF 3-7, MID 8-12, FWD 13-15.
    pts = {1: 6, 3: 5, 4: 5, 5: 5, 8: 5, 9: 5, 13: 5, 14: 5, 10: 5, 6: 5, 11: 5}
    started = {1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14}
    record, act = _squad(pts, started, captain=1)
    r = calibrate.assess(record, act)
    assert r["lineup_loss"] == 0


def test_benching_the_top_scorer_shows_up_as_lineup_loss():
    started = {1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14}
    # 13, 14 and 15 are all forwards and three is a legal maximum, so the best
    # XI drops a defender (5 on the books, 3 required) and fields all three
    # rather than swapping 15 for 14: 21 against the 1 actually scored.
    record, act = _squad({15: 20, 14: 1}, started, captain=1)
    r = calibrate.assess(record, act)
    assert r["xi_actual"] == 1 and r["best_xi"] == 21
    assert r["lineup_loss"] == 20
    assert any(name == "p15" for _s, name, _x, _p in r["misranked"])


def test_captain_loss_is_measured_against_the_xi_actually_fielded():
    """
    Not against the whole squad: you cannot captain a benched player, so
    charging the model for that would overstate the loss every week.
    """
    started = {1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14}
    record, act = _squad({9: 15, 1: 2, 15: 99}, started, captain=1)
    r = calibrate.assess(record, act)
    assert r["captain_loss"] == 13, "15 (best in XI) - 2 (actual captain)"


def test_zero_minutes_starters_are_surfaced_separately():
    """
    Starting someone who did not play is a p_start failure, and minutes are
    the dominant source of model error - worth more than goals, assists and
    clean sheets combined once known.
    """
    started = {1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14}
    record, act = _squad({}, started, captain=1)
    act[9]["minutes"] = 0
    r = calibrate.assess(record, act)
    assert "p9" in r["started_blanks"]


# ------------------------------------------------------------- loading

def test_records_load_from_a_local_directory(tmp_path):
    (tmp_path / "gw07.json").write_text(json.dumps({"event": 7, "players": []}))
    (tmp_path / "gw08.json").write_text(json.dumps({"event": 8, "players": []}))
    recs = calibrate.load_records(directory=str(tmp_path))
    assert [r["event"] for r in recs] == [7, 8]
    assert [r["event"] for r in calibrate.load_records([8], directory=str(tmp_path))] == [8]


def test_a_missing_gameweek_is_skipped_not_fatal():
    """The branch has no record for gameweeks played before the ledger existed."""
    calls = []

    def fetcher(url, timeout=30):
        calls.append(url)
        return None

    assert calibrate.load_records([1, 2, 3], fetcher=fetcher) == []
    assert len(calls) == 3


def test_an_empty_ledger_says_so_rather_than_printing_nothing(capsys):
    calibrate.report([])
    out = capsys.readouterr().out
    assert "empty" in out.lower()


def test_the_best_xi_cannot_field_two_goalkeepers():
    """
    GW4's real data cannot catch a missing formation constraint: there, the
    unconstrained top eleven happens to be legal already, so dropping the
    check changes nothing and the test passes. This forces the issue by making
    both keepers high scorers - an unconstrained solver takes both, and a
    benchmark that fields two keepers makes every real lineup look terrible.
    """
    started = {1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14}
    record, act = _squad({1: 15, 2: 15}, started, captain=1)

    pts = {int(p["id"]): act[int(p["id"])]["total_points"] for p in record["players"]}
    total, xi = calibrate.best_legal_xi(record["players"], pts)

    pos = {int(p["id"]): int(p["position"]) for p in record["players"]}
    assert sum(1 for i in xi if pos[i] == 1) == 1, "two keepers in the benchmark XI"
    assert total == 15, "only one of the two 15-point keepers may count"
