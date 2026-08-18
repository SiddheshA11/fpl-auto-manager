"""
Pairing transfers in with transfers out.

FPL validates every pair individually and rejects the whole POST if any one of
them swaps position:

    {"non_field_errors": [{"message": "Element in and element out must be of
      the same type", "code": "transfer_element_type_mismatch"}]}

The lists were sorted by element id and zipped, which orders them by nothing
that matters. A real 12-transfer submission against the live account on
2026-08-18 had 8 of its 12 pairs rejected and the entire rebuild refused. A
single transfer pairs correctly by luck whenever the squad shape is unchanged,
which is why this survived every test that came before it.
"""
from __future__ import annotations

import pandas as pd
import pytest

import fpl_client
import manager

# 1 GK, 2 DEF, 3 MID, 4 FWD.
#
# Ids deliberately do not run parallel to positions. FPL assigns element ids by
# club in alphabetical order, so sorting a transfer list by id shuffles
# positions arbitrarily - which is the entire cause of the bug. A fixture whose
# ids ascend with position lines the two sides up by coincidence and tests
# nothing, which is what the first draft of this file did.
POOL = pd.DataFrame([
    {"id": 11, "position": 3, "web_name": "mid_in"},
    {"id": 14, "position": 1, "web_name": "gk_out"},
    {"id": 25, "position": 1, "web_name": "gk_in"},
    {"id": 28, "position": 4, "web_name": "fwd_out"},
    {"id": 33, "position": 4, "web_name": "fwd_in"},
    {"id": 36, "position": 2, "web_name": "def_out"},
    {"id": 42, "position": 3, "web_name": "mid_out"},
    {"id": 47, "position": 2, "web_name": "def_in"},
])
POSITIONS = POOL.set_index("id")["position"].to_dict()


def _pairs(ins, outs):
    return list(zip(ins, outs))


def test_every_pair_matches_position_after_pairing():
    transfers_in = [11, 25, 33, 47]     # MID, GK, FWD, DEF
    transfers_out = [14, 28, 36, 42]    # GK, FWD, DEF, MID
    naive = _pairs(sorted(transfers_in), sorted(transfers_out))
    assert any(POSITIONS[i] != POSITIONS[o] for i, o in naive), (
        "setup must be broken under the old id-sorted zip, or it proves nothing"
    )

    ins, outs = manager._pair_by_position(transfers_in, transfers_out, POOL)
    for i, o in zip(ins, outs):
        assert POSITIONS[i] == POSITIONS[o], f"pair {i}->{o} swaps position"
    assert set(ins) == set(transfers_in) and set(outs) == set(transfers_out)


def test_pairing_survives_an_adversarial_id_order():
    """
    The failure needs ids whose sort order cuts across position, which is the
    normal case: element ids are assigned alphabetically by club.
    """
    # The two sides must run through the positions in *different* orders,
    # otherwise the id-sorted zip lines them up by coincidence and the test
    # proves nothing - which is exactly what the first draft of it did.
    pool = pd.DataFrame([
        {"id": 1, "position": 1}, {"id": 2, "position": 2},
        {"id": 3, "position": 3}, {"id": 4, "position": 4},
        {"id": 5, "position": 4}, {"id": 6, "position": 3},
        {"id": 7, "position": 2}, {"id": 8, "position": 1},
    ])
    positions = pool.set_index("id")["position"].to_dict()
    transfers_in, transfers_out = [1, 2, 3, 4], [5, 6, 7, 8]

    bad = [(i, o) for i, o in zip(sorted(transfers_in), sorted(transfers_out))
           if positions[i] != positions[o]]
    assert bad, "setup must actually be broken under the old id-sorted zip"

    ins, outs = manager._pair_by_position(transfers_in, transfers_out, pool)
    assert all(positions[i] == positions[o] for i, o in zip(ins, outs))


def test_a_single_transfer_is_unaffected():
    """The case that always worked must keep working."""
    ins, outs = manager._pair_by_position([11], [42], POOL)   # both MID
    assert ins == [11] and outs == [42]


def test_multiple_swaps_within_one_position():
    pool = pd.DataFrame([{"id": i, "position": 3} for i in (1, 2, 3, 4)])
    ins, outs = manager._pair_by_position([1, 2], [3, 4], pool)
    assert sorted(ins) == [1, 2] and sorted(outs) == [3, 4]
    assert len(ins) == len(outs) == 2


def test_mismatched_position_multisets_are_reported_not_hidden(caplog):
    """
    Impossible for a squad the optimiser built, since it fixes composition at
    2/5/5/3. If it ever happens the submission is doomed, so it must be loud.
    """
    import logging
    with caplog.at_level(logging.ERROR, logger="fpl_auto"):
        ins, outs = manager._pair_by_position([11, 25], [36, 28], POOL)
    assert "positions do not match" in caplog.text.lower(), (
        "an unpairable transfer set must be logged, not silently submitted"
    )
    # Returned unchanged so the caller still sees what it asked for.
    assert ins == [11, 25] and outs == [36, 28]


class TestClientRefusesBadPairs:
    """The API boundary is the last place to catch this before FPL does."""

    def test_a_mismatched_pair_raises_before_any_request(self):
        client = fpl_client.FPLClient()
        client.authenticated = True
        with pytest.raises(ValueError, match="transfer_element_type_mismatch"):
            client.make_transfers(
                transfers_in=[11], transfers_out=[14],
                prices_in=[50], prices_out=[50],
                positions=POSITIONS,
            )

    def test_matched_pairs_pass_validation(self, monkeypatch):
        client = fpl_client.FPLClient()
        client.authenticated = True
        monkeypatch.setattr(client, "get_next_event", lambda: {"id": 1})
        sent = {}
        monkeypatch.setattr(client, "_post", lambda url, payload: sent.update(payload) or {"ok": True})
        client.make_transfers(
            transfers_in=[11, 25], transfers_out=[42, 14],
            prices_in=[50, 50], prices_out=[50, 50],
            positions=POSITIONS,
        )
        assert len(sent["transfers"]) == 2

    def test_validation_is_skipped_when_positions_are_not_supplied(self, monkeypatch):
        """Optional only so older callers keep working; it must not crash."""
        client = fpl_client.FPLClient()
        client.authenticated = True
        monkeypatch.setattr(client, "get_next_event", lambda: {"id": 1})
        monkeypatch.setattr(client, "_post", lambda url, payload: {"ok": True})
        assert client.make_transfers([11], [14], [50], [50]) == {"ok": True}


def test_the_weekly_run_submits_only_like_for_like_pairs(monkeypatch):
    """
    Integration, and the one that counts.

    The unit tests above exercise the pairing helper directly, so they keep
    passing if the helper is never called - which is precisely the bug that
    reached production. This drives the real weekly run against a squad that
    needs a full rebuild and inspects what the client was actually handed.
    """
    import copy
    from pathlib import Path

    import priors
    import xp_model as X
    from test_pipeline import StubClient, _snapshot

    bootstrap, fixtures = copy.deepcopy(_snapshot("bootstrap-static")), _snapshot("fixtures")
    first = min(int(e["id"]) for e in bootstrap["events"])
    for e in bootstrap["events"]:
        e["finished"] = False
        e["is_current"] = False
        e["is_next"] = int(e["id"]) == first

    ps = priors.build_priors(current_team_codes={t["code"]: t["name"] for t in bootstrap["teams"]})
    scored = X.XPModel(bootstrap, fixtures, ps).expected_points(X.next_events(bootstrap, 5))
    available = scored[scored["status"].isin(["a", "d"])]

    # A deliberately awful squad, so the run proposes many transfers at once.
    owned, costs = [], {}
    for position, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        worst = available[available["position"] == position].nsmallest(count, "xp_horizon")
        for _, p in worst.iterrows():
            owned.append(int(p["id"]))
            costs[int(p["id"])] = float(p["cost"])

    client = StubClient(bootstrap, fixtures, owned, costs, bank=0.0, free_transfers=1)
    monkeypatch.setattr(manager, "FPLClient", lambda: client)
    result = manager.run_weekly_cycle(dry_run=False)

    assert result is not None
    submitted = client.submitted_transfers
    assert submitted, "the run made no transfers, so this proves nothing"
    assert len(submitted["transfers_in"]) > 3, "need a multi-transfer submission to be meaningful"

    position = scored.set_index("id")["position"].to_dict()
    mismatched = [
        (i, o) for i, o in zip(submitted["transfers_in"], submitted["transfers_out"])
        if position[i] != position[o]
    ]
    assert not mismatched, (
        f"{len(mismatched)} pair(s) swap position; FPL rejects the whole POST with "
        f"transfer_element_type_mismatch: {mismatched}"
    )
