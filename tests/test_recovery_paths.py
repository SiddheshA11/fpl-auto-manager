"""
What happens when a submission is refused.

These paths run unattended, minutes before a deadline, and every one of them
was reachable in production while invisible to the suite: deleting the fix and
running all 113 tests stayed green. A lineup that never gets submitted costs
the entire gameweek, so the recovery path matters more than the happy one.
"""
from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

import fpl_client
import manager
import news
import priors
import xp_model as X
from test_pipeline import StubClient, _snapshot


@pytest.fixture(scope="module")
def game_state():
    return _snapshot("bootstrap-static"), _snapshot("fixtures")


def _squad_and_costs(bootstrap, fixtures):
    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    scored = X.XPModel(bootstrap, fixtures, ps).expected_points(X.next_events(bootstrap, 5))
    pool = scored[scored["status"].isin(["a", "d"])]
    # Respect the three-per-club cap: FPL never permits a squad that breaks it,
    # and an illegal one makes the solver infeasible for reasons that have
    # nothing to do with what these tests are checking.
    owned, costs, per_club = [], {}, {}
    for position, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        taken = 0
        for _, p in pool[pool["position"] == position].sort_values("xp_horizon").iterrows():
            club = int(p["team"])
            if per_club.get(club, 0) >= 3:
                continue
            owned.append(int(p["id"]))
            costs[int(p["id"])] = float(p["cost"])
            per_club[club] = per_club.get(club, 0) + 1
            taken += 1
            if taken == count:
                break
    assert len(owned) == 15
    return owned, costs


def test_a_rejected_pairing_still_submits_a_lineup(game_state, monkeypatch):
    """
    The guard in make_transfers raises so a bad pairing fails at the point of
    the mistake. Letting that escape skipped STEP 6 altogether and lost the
    gameweek - strictly worse than the 400 it was written to pre-empt.
    """
    bootstrap, fixtures = game_state
    owned, costs = _squad_and_costs(bootstrap, fixtures)

    class RejectingClient(StubClient):
        def make_transfers(self, **kwargs):
            raise ValueError("3 transfer pair(s) swap position, which FPL rejects "
                             "with transfer_element_type_mismatch: [(1, 2)]")

    client = RejectingClient(bootstrap, fixtures, owned, costs)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)

    result = manager.run_weekly_cycle(dry_run=False)
    assert result is not None, "the run must not die on a rejected pairing"
    assert client.submitted_lineup is not None, "a lineup must still be submitted"
    assert len(client.submitted_lineup["picks"]) == 15


def test_a_failed_transfer_post_still_submits_a_lineup(game_state, monkeypatch):
    """The same guarantee for FPL returning an error rather than us raising."""
    bootstrap, fixtures = game_state
    owned, costs = _squad_and_costs(bootstrap, fixtures)

    class FailingClient(StubClient):
        def make_transfers(self, **kwargs):
            return None

    client = FailingClient(bootstrap, fixtures, owned, costs)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)

    result = manager.run_weekly_cycle(dry_run=False)
    assert result is not None
    assert client.submitted_lineup is not None
    # The lineup must be built from the squad actually owned, not the one the
    # failed transfer would have produced.
    submitted = {p["element"] for p in client.submitted_lineup["picks"]}
    assert submitted == set(owned)


def test_the_lineup_is_reoptimised_on_the_coming_gameweek(game_state, monkeypatch):
    """
    Covers the call site, not just the helper. Deleting
    `plan = _reoptimise_lineup(...)` from manager.py left all 113 tests green -
    the same call-site-not-covered hole that shipped the transfer-pairing bug.
    """
    bootstrap, fixtures = game_state
    owned, costs = _squad_and_costs(bootstrap, fixtures)
    client = StubClient(bootstrap, fixtures, owned, costs)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)

    calls = []
    original = manager._reoptimise_lineup
    monkeypatch.setattr(
        manager, "_reoptimise_lineup",
        lambda plan, scored, tilt: calls.append(1) or original(plan, scored, tilt))

    result = manager.run_weekly_cycle(dry_run=False)
    assert result is not None
    assert calls, "the weekly run must re-optimise the lineup before submitting"

    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    scored = X.XPModel(bootstrap, fixtures, ps).expected_points(X.next_events(bootstrap, 5))
    xp_next = scored.set_index("id")["xp_next"].to_dict()
    submitted = {p["element"] for p in client.submitted_lineup["picks"]}
    xi = {p["element"] for p in client.submitted_lineup["picks"] if p["position"] <= 11}
    bench = submitted - xi
    # No benched outfielder may outscore a starter of the same position.
    position = scored.set_index("id")["position"].to_dict()
    for b in bench:
        for x in xi:
            if position[b] == position[x] and position[b] != 1:
                # 1e-3, not 1e-6. The lineup is a MILP, and two players within
                # a few thousandths of a point are a tie it may break either
                # way - this fired on 1.3512 against 1.3506, both printing as
                # 1.35. A solver tolerance is not a lineup bug.
                assert xp_next[b] <= xp_next[x] + 1e-3, (
                    f"benched {b} ({xp_next[b]:.2f}) beats starter {x} ({xp_next[x]:.2f})"
                )


def test_reoptimise_falls_back_when_the_solve_fails(game_state, monkeypatch):
    """A worse-ordered lineup costs a fraction of a point; none costs the week."""
    bootstrap, fixtures = game_state
    owned, costs = _squad_and_costs(bootstrap, fixtures)
    client = StubClient(bootstrap, fixtures, owned, costs)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)

    def explode(self):
        raise RuntimeError("solver unavailable")
    monkeypatch.setattr(manager.SquadOptimizer, "optimise_lineup", explode)

    result = manager.run_weekly_cycle(dry_run=False)
    assert result is not None, "a failed re-optimisation must not lose the gameweek"
    assert client.submitted_lineup is not None
    assert len(client.submitted_lineup["picks"]) == 15


class TestIndefiniteAbsencesNeverRecover:
    """
    FPL routinely publishes a percentage alongside an open-ended injury. A base
    above zero was enough for decay_doubt to lift such a player to 0.95
    availability by gameweek five - restoring someone with no return date to
    nearly fit, which is precisely what the guard exists to stop.
    """

    def test_a_percentage_does_not_resurrect_an_open_ended_injury(self, game_state):
        bootstrap, fixtures = game_state
        bootstrap = copy.deepcopy(bootstrap)
        target = next(e for e in bootstrap["elements"]
                      if "Unknown return" in (e.get("news") or "") and e["status"] == "i")
        target["chance_of_playing_next_round"] = 25

        ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
        model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=5))
        events = X.next_events(bootstrap, 5)
        idx = model.players.index[model.players["id"] == target["id"]][0]

        for ev in events:
            avail = model._availability(ev)[idx]
            assert avail <= 0.25 + 1e-9, (
                f"{target['web_name']} has no return date but reached {avail:.2f} "
                f"availability in GW{ev}"
            )

    def test_a_genuine_doubt_still_recovers(self, game_state):
        """The guard must not also freeze ordinary knocks."""
        bootstrap, fixtures = game_state
        ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
        model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=5))
        events = X.next_events(bootstrap, 5)
        frame = model.players
        doubts = frame.index[(frame["status"] == "d")
                             & frame["chance_of_playing_next_round"].between(1, 99)]
        assert len(doubts), "snapshot has no ordinary doubts to check"
        first, last = model._availability(events[0]), model._availability(events[-1])
        assert (last[doubts] > first[doubts]).all()


def test_a_short_absence_is_not_damped_like_a_long_one(game_state):
    """
    `_ramp` never supplied absence_days, so the carve-out in ramp_multiplier
    was unreachable and a one-match suspension was damped to 0.55 exactly like
    a three-month injury. tests/test_news.py asserted the carve-out by passing
    absence_days by hand, testing behaviour production never exhibited.
    """
    bootstrap, fixtures = game_state
    bootstrap = copy.deepcopy(bootstrap)
    events = X.next_events(bootstrap, 5)
    deadline = next(e for e in bootstrap["events"] if int(e["id"]) == events[0])["deadline_time"][:10]
    back = date.fromisoformat(deadline)

    target = bootstrap["elements"][0]
    target["status"] = "s"
    target["news"] = f"Suspended until {back.day} {back.strftime('%b')}"
    target["news_added"] = f"{back.isoformat()}T00:00:00Z"   # suspended days, not months

    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=5))
    idx = model.players.index[model.players["id"] == target["id"]][0]
    assert model._ramp(events[0])[idx] == pytest.approx(1.0), (
        "a short absence costs no match fitness and must not be damped"
    )
