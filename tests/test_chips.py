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


def test_bench_boost_played_only_when_bench_clears_threshold():
    bs = _bootstrap_with_two_windows()

    weak = _scored({i: 0.5 for i in range(1, 16)})
    eng = chips.ChipEngine(bs, [], weak)
    d = eng.evaluate(5, {"chips": []}, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    assert d.chip != "bboost"

    strong = _scored({**{i: 1.0 for i in range(1, 12)}, **{i: 5.0 for i in (12, 13, 14, 15)}})
    eng = chips.ChipEngine(bs, [], strong)
    d = eng.evaluate(5, {"chips": []}, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    assert d.chip == "bboost"
    assert d.expected_gain == pytest.approx(20.0)


def test_no_chip_when_nothing_clears_its_threshold():
    bs = _bootstrap_with_two_windows()
    eng = chips.ChipEngine(bs, [], _scored({i: 0.4 for i in range(1, 16)}))
    d = eng.evaluate(5, {"chips": []}, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    assert d.chip is None
    assert "threshold" in d.reason


def test_unavailable_chips_are_never_recommended():
    bs = _bootstrap_with_two_windows()
    played = {"chips": [{"name": n, "event": 2} for n in chips.CHIP_NAMES]}
    scored = _scored({**{i: 1.0 for i in range(1, 12)}, **{i: 9.0 for i in (12, 13, 14, 15)}})
    eng = chips.ChipEngine(bs, [], scored)
    d = eng.evaluate(6, played, xi_ids=list(range(1, 12)), bench_ids=[12, 13, 14, 15], captain_id=1)
    assert d.chip is None
