"""
Chip state, against a real authenticated payload.

`tests/fixtures/my_team.json` is a genuine `/my-team/` response for entry
5413589, captured 2026-08-17 and redacted. It exists because the rest of the
chip tests construct the shape the code *assumes*, which proves the code agrees
with itself rather than with FPL - and the assumption was wrong.

The code read `chip["event"]`. That key does not exist on `/my-team/`; the
endpoint reports `played_by_entry`, a list of gameweeks. So every chip looked
unplayed forever and the engine would have re-submitted one it had spent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import chips

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SNAPSHOTS = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@pytest.fixture(scope="module")
def my_team():
    return json.loads((FIXTURES / "my_team.json").read_text())["my_team"]


@pytest.fixture(scope="module")
def bootstrap():
    import xp_model as X
    files = sorted(SNAPSHOTS.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not files:
        pytest.skip("no bootstrap snapshot committed")
    return X.load_snapshot(files[0])


def test_the_fixture_is_the_shape_we_were_not_expecting(my_team):
    """Pins the discovery, so a future refactor cannot quietly re-guess."""
    chip = my_team["chips"][0]
    assert "event" not in chip, "the key the old code read must be absent"
    assert "played_by_entry" in chip and isinstance(chip["played_by_entry"], list)
    assert "status_for_entry" in chip
    assert {"name", "start_event", "stop_event", "chip_type", "is_pending"} <= set(chip)


def test_transfers_block_reports_no_limit_before_the_first_deadline(my_team):
    """
    The other half of the same discovery. `limit` is null while transfers are
    unlimited, so `(limit or 1) - made` yields one free transfer during the one
    week the whole squad can be rebuilt for nothing.
    """
    transfers = my_team["transfers"]
    assert transfers["limit"] is None
    assert transfers["status"] == "unlimited"
    assert (transfers.get("limit") or 1) - (transfers.get("made") or 0) == 1, (
        "this is the wrong answer the old arithmetic produced; it is pinned so "
        "the pre-season guard in manager.py cannot be removed unnoticed"
    )


def test_nothing_is_played_on_a_fresh_entry(my_team):
    assert chips.played_chips(my_team) == []
    assert chips.pending_chips(my_team) == set()


def test_only_the_team_chips_are_playable_in_gameweek_one(my_team, bootstrap):
    """Wildcard and free hit open at GW2; bench boost and triple captain at GW1."""
    assert chips.available_chips(bootstrap, my_team, 1) == {"bboost", "3xc"}
    assert chips.available_chips(bootstrap, my_team, 2) == {"bboost", "3xc", "freehit", "wildcard"}


def test_a_played_chip_is_detected_in_the_real_shape(my_team, bootstrap):
    played = json.loads(json.dumps(my_team))
    played["chips"][0]["played_by_entry"] = [3]
    played["chips"][0]["status_for_entry"] = "played"

    assert chips.played_chips(played) == [("bboost", 3)]
    assert "bboost" not in chips.available_chips(bootstrap, played, 5)


def test_the_second_half_chip_survives_the_first_being_spent(my_team, bootstrap):
    """The game issues two of everything; spending one must not retire both."""
    played = json.loads(json.dumps(my_team))
    played["chips"][0]["played_by_entry"] = [3]
    assert "bboost" not in chips.available_chips(bootstrap, played, 5)
    assert "bboost" in chips.available_chips(bootstrap, played, 25)


def test_the_history_endpoint_shape_is_still_understood():
    """`/entry/{id}/history/` uses {name, event}, and is still read."""
    history_shaped = {"chips": [{"name": "wildcard", "event": 8}]}
    assert chips.played_chips(history_shaped) == [("wildcard", 8)]


def test_a_pending_chip_blocks_a_second_one(my_team, bootstrap):
    """
    A chip switched on for a deadline that has not passed is not yet in
    `played_by_entry`. Without honouring `is_pending`, a second run in the same
    gameweek sees it as unplayed and stacks another chip on top.
    """
    pending = json.loads(json.dumps(my_team))
    pending["chips"][0]["is_pending"] = True
    assert chips.pending_chips(pending) == {"bboost"}
    assert "bboost" not in chips.available_chips(bootstrap, pending, 1)
    assert "3xc" in chips.available_chips(bootstrap, pending, 1)


def test_the_bar_falls_as_the_window_closes(bootstrap):
    """
    A chip is one-shot with an expiry, so the question is never "is this worth
    13 points" but "is this better than the best chance left before it expires".
    Early in a window there are many chances and the bar is high; on the final
    gameweek it is play-it-or-lose-it and the base threshold is the whole test.
    """
    bars = [chips.effective_threshold(bootstrap, "bboost", ev) for ev in (1, 5, 10, 15, 19)]
    assert bars == sorted(bars, reverse=True), "the bar must fall monotonically"
    assert bars[-1] == pytest.approx(chips.THRESHOLDS["bboost"]), (
        "on the last gameweek of the window the base threshold is the whole test"
    )
    assert bars[0] > bars[-1]


def test_the_bar_resets_for_the_second_half_window(bootstrap):
    """The game issues two of everything; the second gets its own full window."""
    assert chips.effective_threshold(bootstrap, "bboost", 20) > \
           chips.effective_threshold(bootstrap, "bboost", 19)
    assert chips.effective_threshold(bootstrap, "bboost", 38) == pytest.approx(
        chips.THRESHOLDS["bboost"])


def test_a_merely_decent_bench_is_not_boosted_in_gameweek_one(bootstrap, my_team):
    """
    Regression, from a real dry run. The engine proposed bench boost in GW1 for
    13.3 xP against a flat 12.0 bar - in a gameweek where every player appears
    exactly once, with every double gameweek of the half-season still ahead.
    """
    assert 13.3 < chips.effective_threshold(bootstrap, "bboost", 1)
    # ...and the same bench would still be boosted at the window's deadline.
    assert 13.3 > chips.effective_threshold(bootstrap, "bboost", 19)


def test_an_exceptional_bench_still_triggers_early(bootstrap):
    """The premium holds an ordinary week, not a genuinely outstanding one."""
    assert 25.0 > chips.effective_threshold(bootstrap, "bboost", 1)
