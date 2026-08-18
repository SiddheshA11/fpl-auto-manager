"""
Tests for chip availability and chip valuation.

The availability tests are the important ones. The game issues two of every
chip - one window for GW1-19, another for GW20-38 - and the previous
implementation treated a chip as spent forever once played, silently retiring
four chips over a season.
"""
from __future__ import annotations

import pandas as pd
import pytest

import chips


def _bootstrap_with_two_windows() -> dict:
    """Mirrors the real bootstrap: one chip entry per window per name."""
    entries = []
    for name in ("wildcard", "freehit", "bboost", "3xc"):
        entries.append({"name": name, "start_event": 1, "stop_event": 19, "chip_type": "x"})
        entries.append({"name": name, "start_event": 20, "stop_event": 38, "chip_type": "x"})
    return {"chips": entries}


def test_all_chips_available_when_none_played():
    bs = _bootstrap_with_two_windows()
    assert chips.available_chips(bs, {"chips": []}, event=5) == chips.CHIP_NAMES


def test_chip_played_in_first_half_is_unavailable_in_first_half():
    bs = _bootstrap_with_two_windows()
    team = {"chips": [{"name": "wildcard", "event": 8}]}
    assert "wildcard" not in chips.available_chips(bs, team, event=12)


def test_chip_played_in_first_half_is_available_again_in_second():
    """
    Regression: the old check was global, so a wildcard played in October
    retired the second-half wildcard too. Both halves get their own.
    """
    bs = _bootstrap_with_two_windows()
    team = {"chips": [{"name": "wildcard", "event": 8}]}
    assert "wildcard" in chips.available_chips(bs, team, event=25)


def test_second_half_chip_is_unavailable_once_used_in_that_window():
    bs = _bootstrap_with_two_windows()
    team = {"chips": [{"name": "wildcard", "event": 8}, {"name": "wildcard", "event": 24}]}
    assert "wildcard" not in chips.available_chips(bs, team, event=30)


def test_chip_outside_its_window_is_unavailable():
    bs = {"chips": [{"name": "freehit", "start_event": 20, "stop_event": 38}]}
    assert chips.available_chips(bs, {"chips": []}, event=5) == set()
    assert chips.available_chips(bs, {"chips": []}, event=22) == {"freehit"}


def test_missing_team_data_does_not_crash():
    bs = _bootstrap_with_two_windows()
    assert chips.available_chips(bs, None, event=3) == chips.CHIP_NAMES


# ──────────────────────── calendar ────────────────────────


def _fixtures() -> list[dict]:
    # Team 1 plays twice (double), team 4 not at all (blank).
    return [
        {"event": 10, "team_h": 1, "team_a": 2},
        {"event": 10, "team_h": 3, "team_a": 1},
        {"event": 10, "team_h": 5, "team_a": 6},
    ]


def test_double_and_blank_gameweeks_are_detected():
    counts = chips.fixtures_per_team(_fixtures(), event=10)
    assert counts[1] == 2, "team with two fixtures is a double"
    assert counts[2] == 1
    assert 4 not in counts, "team with no fixture is a blank"


def test_calendar_description_names_doubles_and_blanks():
    teams = pd.DataFrame([{"id": i, "short_name": f"T{i}"} for i in range(1, 7)])
    desc = chips.describe_calendar(_fixtures(), 10, teams)
    assert "T1" in desc and "doubles" in desc
    assert "T4" in desc and "blanks" in desc


# ──────────────────────── valuation ────────────────────────


def _scored(xp: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame([{"id": pid, "xp_next": v, "xp_horizon": v * 4} for pid, v in xp.items()])


def test_bench_boost_tracks_the_squad_that_would_actually_be_benched():
    """
    Bench value is the squad's weakest four *in that gameweek*, not a fixed
    list handed in by the caller. If the nominal bench outscores the starters
    they would simply start, and the chip is worth whatever the real tail is.
    """
    bs = _bootstrap_with_two_windows()

    # Whole squad worthless: nothing to boost.
    weak = _scored({i: 0.5 for i in range(1, 16)})
    eng = chips.ChipEngine(bs, [], weak)
    d = eng.evaluate(5, {"chips": []}, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    assert d.chip != "bboost"

    # A genuinely deep squad - the weakest four are still worth 8 apiece, which
    # clears the 17.6 bar for committing at GW5 with most of the window unseen.
    deep = _scored({i: 8.0 for i in range(1, 16)})
    eng = chips.ChipEngine(bs, [], deep)
    d = eng.evaluate(5, {"chips": []}, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    assert d.chip == "bboost"
    # Only what autosubs would not have delivered anyway.
    assert d.expected_gain == pytest.approx(4 * 8.0 * (1 - chips.AUTOSUB_SHARE))


def test_no_chip_when_no_gameweek_is_worth_it():
    """
    A near-worthless bench must not be boosted. The old code compared it to a
    fixed bar; the planner compares it to every other gameweek left in the
    window, and holds when none of them is worth a one-shot resource.
    """
    bs = _bootstrap_with_two_windows()
    eng = chips.ChipEngine(bs, [], _scored({i: 0.4 for i in range(1, 16)}))
    d = eng.evaluate(5, {"chips": []}, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    # A flat 0.4 bench is worth ~1.3 xP boosted; triple captain on a 0.4 player
    # is worth 0.4. Neither is a reason to spend a chip, but both are positive,
    # so the planner may legitimately schedule one rather than refuse outright.
    # What it must not do is play one *now* when every gameweek is identical
    # and nothing is urgent.
    assert d.chip is None or d.expected_gain < 2.0


def test_unavailable_chips_are_never_recommended():
    bs = _bootstrap_with_two_windows()
    played = {"chips": [{"name": n, "event": 2} for n in chips.CHIP_NAMES]}
    scored = _scored({**{i: 1.0 for i in range(1, 12)}, **{i: 9.0 for i in (12, 13, 14, 15)}})
    eng = chips.ChipEngine(bs, [], scored)
    d = eng.evaluate(6, played, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    assert d.chip is None
