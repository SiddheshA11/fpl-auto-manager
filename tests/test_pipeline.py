"""
End-to-end test of the weekly run against a stub API client.

The weekly run executes unattended on a schedule and submits real transfers,
so "it imports" is not evidence that it works. This drives the whole pipeline
- scoring, chip valuation, transfer optimisation, lineup construction - on the
committed snapshot, with the network replaced by a stub that records what
would have been submitted.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pytest

import manager
import optimizer as O
import priors
import xp_model as X

REPO = Path(__file__).resolve().parent.parent
SNAPSHOTS = REPO / "data" / "snapshots"


def _snapshot(name: str):
    files = sorted(SNAPSHOTS.glob(f"{name}_*.json.gz"), reverse=True)
    if not files:
        pytest.skip(f"no {name} snapshot committed")
    return X.load_snapshot(files[0])


@pytest.fixture(scope="module")
def game_state():
    return _snapshot("bootstrap-static"), _snapshot("fixtures")


@pytest.fixture(scope="module")
def starting_squad(game_state):
    """A legal 15 built from the snapshot, standing in for an owned team."""
    bootstrap, fixtures = game_state
    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    scored = X.XPModel(bootstrap, fixtures, ps).expected_points(X.next_events(bootstrap, 5))
    pool = scored[scored["status"].isin(["a", "d"])]
    sol = O.SquadOptimizer(pool, "xp_horizon", "xp_next").build_squad(100.0)
    return sol


class StubClient:
    """Replaces FPLClient. Records submissions instead of making them."""

    def __init__(self, bootstrap, fixtures, squad_ids, costs, *, bank=0.5, free_transfers=1, chips_played=None):
        self._bootstrap = bootstrap
        self._fixtures = fixtures
        self._squad_ids = list(squad_ids)
        self._costs = costs
        self._bank = bank
        self._ft = free_transfers
        self._chips_played = chips_played or []
        self.submitted_lineup = None
        self.submitted_transfers = None
        self.logged_in = False

    def login(self):
        self.logged_in = True
        return True

    def get_bootstrap(self):
        return self._bootstrap

    def get_fixtures(self):
        return pd.DataFrame(self._fixtures)

    def get_next_event(self):
        return next((e for e in self._bootstrap["events"] if not e.get("finished")), None)

    def get_my_team(self):
        return {
            "picks": [
                {
                    "element": pid,
                    "position": i + 1,
                    "selling_price": int(self._costs[pid] * 10),
                    "is_captain": i == 0,
                    "is_vice_captain": i == 1,
                }
                for i, pid in enumerate(self._squad_ids)
            ],
            "transfers": {"bank": int(self._bank * 10), "limit": self._ft},
            "chips": self._chips_played,
        }

    def set_lineup(self, picks, chip=None):
        self.submitted_lineup = {"picks": picks, "chip": chip}
        return {"ok": True}

    def make_transfers(self, **kwargs):
        self.submitted_transfers = kwargs
        return {"ok": True}


@pytest.fixture
def stub(game_state, starting_squad, monkeypatch):
    bootstrap, fixtures = game_state
    costs = dict(zip(starting_squad.squad["id"].astype(int), starting_squad.squad["cost"]))
    client = StubClient(bootstrap, fixtures, [int(i) for i in starting_squad.squad["id"]], costs)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)
    return client


def test_dry_run_completes_and_submits_nothing(stub):
    result = manager.run_weekly_cycle(dry_run=True)

    assert result is not None, "pipeline must complete"
    assert stub.logged_in
    assert stub.submitted_lineup is None, "dry run must not submit a lineup"
    assert stub.submitted_transfers is None, "dry run must not submit transfers"
    assert result["event_id"] >= 1


def test_live_run_submits_a_legal_lineup(stub):
    manager.run_weekly_cycle(dry_run=False)

    assert stub.submitted_lineup is not None
    picks = stub.submitted_lineup["picks"]

    assert len(picks) == 15
    assert [p["position"] for p in picks] == list(range(1, 16)), "positions must be 1..15 in order"
    assert sum(p["is_captain"] for p in picks) == 1
    assert sum(p["is_vice_captain"] for p in picks) == 1

    captain = next(p for p in picks if p["is_captain"])
    vice = next(p for p in picks if p["is_vice_captain"])
    assert captain["position"] <= 11, "captain must be in the starting XI"
    assert vice["position"] <= 11, "vice-captain must be in the starting XI"
    assert captain["element"] != vice["element"]

    assert len({p["element"] for p in picks}) == 15, "no duplicate players"


def test_bench_keeper_is_first_on_the_bench(stub):
    """
    A benched keeper can only ever replace the keeper, so he must occupy
    position 12. Any other order silently breaks autosubs.
    """
    manager.run_weekly_cycle(dry_run=False)
    picks = stub.submitted_lineup["picks"]

    bootstrap = stub.get_bootstrap()
    position_of = {int(e["id"]): int(e["element_type"]) for e in bootstrap["elements"]}

    bench = [p for p in picks if p["position"] > 11]
    keepers = [p for p in bench if position_of[p["element"]] == 1]
    assert len(keepers) == 1
    assert keepers[0]["position"] == 12


def test_transfers_are_reported_and_within_the_hit_cap(stub):
    result = manager.run_weekly_cycle(dry_run=True, max_hits=1)
    assert len(result["transfers_in"]) == len(result["transfers_out"])
    assert result["hits"] <= 1


def test_run_survives_a_squad_containing_an_unavailable_player(game_state, starting_squad, monkeypatch):
    """
    An injured player already owned must still be scored, so the optimiser can
    value selling him. Dropping him from the frame entirely used to leave the
    squad short and the lineup unbuildable.
    """
    bootstrap, fixtures = game_state
    bootstrap = {**bootstrap, "elements": [dict(e) for e in bootstrap["elements"]]}

    owned = [int(i) for i in starting_squad.squad["id"]]
    for e in bootstrap["elements"]:
        if int(e["id"]) == owned[0]:
            e["status"] = "i"
            e["chance_of_playing_next_round"] = 0

    costs = dict(zip(starting_squad.squad["id"].astype(int), starting_squad.squad["cost"]))
    client = StubClient(bootstrap, fixtures, owned, costs)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)

    result = manager.run_weekly_cycle(dry_run=False)
    assert result is not None
    assert len(client.submitted_lineup["picks"]) == 15


def test_aborts_cleanly_when_authentication_fails(game_state, monkeypatch):
    class DeadClient(StubClient):
        def login(self):
            return False

    bootstrap, fixtures = game_state
    monkeypatch.setattr(manager, "FPLClient", lambda: DeadClient(bootstrap, fixtures, [], {}))

    assert manager.run_weekly_cycle(dry_run=True) is None, "must abort, not crash"


# ---------------------------------------------------------- submission marker
#
# FPL_LAST_SUBMITTED_GW is what the watchdog reads to decide whether a gameweek
# was actually handled, and what makes the hourly schedule idempotent. Both
# depend on it being written by the real submit path and nowhere else.
#
# These are end-to-end on purpose. This repo has already shipped four bugs of
# the form "the helper is correct and nobody calls it", one of them in exactly
# this shape: recommend.py passed no ownership_weight, so the preview tool
# optimised at 0.0 while production used +0.20, with the helper's own tests
# green throughout.


@pytest.fixture
def marker(monkeypatch):
    """Captures what the run records, without a GitHub API."""
    written = []
    monkeypatch.setattr(manager, "_record_submission", lambda gw: written.append(gw))
    return written


def test_a_live_run_records_the_gameweek_it_submitted(stub, marker):
    result = manager.run_weekly_cycle(dry_run=False)

    assert stub.submitted_lineup is not None, "precondition: the run must have submitted"
    assert marker == [result["event_id"]], (
        "a submitted gameweek must be recorded, or the watchdog will report a "
        "false alarm and the next hourly tick will submit again"
    )
    assert result["status"] == "submitted"


def test_a_dry_run_records_nothing(stub, marker):
    """
    Recording a dry run would tell the watchdog to stand down over a team that
    was never submitted - the precise silence this machinery exists to remove.
    """
    manager.run_weekly_cycle(dry_run=True)

    assert stub.submitted_lineup is None
    assert marker == []


def test_a_failed_lineup_submission_records_nothing_and_fails(stub, marker, monkeypatch):
    """
    set_lineup returning None used to be logged and otherwise ignored: the run
    exited 0 having submitted nothing. That is the failure mode the watchdog
    cannot see unless the marker stays unset and the run goes red.
    """
    monkeypatch.setattr(stub, "set_lineup", lambda picks, chip=None: None)

    result = manager.run_weekly_cycle(dry_run=False)

    assert marker == [], "nothing was submitted, so nothing may be recorded"
    assert result["status"] == "lineup-failed"


def test_an_already_submitted_gameweek_is_not_submitted_again(stub, marker, monkeypatch):
    """
    The hourly schedule gives each gameweek four qualifying ticks so a dropped
    run has retries. Without this check those retries each resubmit, which is
    the double submission the old single-tick schedule existed to prevent.
    """
    event_id = stub.get_next_event()["id"]
    monkeypatch.setattr(manager, "_marker_gameweek", lambda: int(event_id))

    result = manager.run_weekly_cycle(dry_run=False, respect_window=True)

    assert result["status"] == "already-submitted"
    assert stub.submitted_lineup is None, "a second tick must not resubmit"
    assert stub.submitted_transfers is None
    assert marker == []


def test_a_marker_from_a_previous_gameweek_does_not_block_this_one(stub, marker, monkeypatch):
    """
    Guards the direction that fails silently: a stale marker that blocked every
    subsequent gameweek would look exactly like a healthy quiet season.
    """
    event_id = int(stub.get_next_event()["id"])
    monkeypatch.setattr(manager, "_marker_gameweek", lambda: event_id - 1)
    monkeypatch.setattr(manager, "hours_to_deadline", lambda *a, **k: 5.0)

    result = manager.run_weekly_cycle(dry_run=False, respect_window=True)

    assert result["status"] == "submitted"
    assert stub.submitted_lineup is not None
    assert marker == [event_id]
