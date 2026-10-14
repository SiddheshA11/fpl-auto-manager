"""
The prediction ledger.

Five gameweeks were submitted before anyone asked why the engine was losing,
and answering that needed archaeology through 30-day run artifacts - GW1 to
GW3 had already expired. The only committed snapshot is pre-season and there
is no 2026-27 history on disk, so backtest.py cannot score the live season
either. The engine was unobservable precisely where observability matters.

Two properties are load-bearing and both are tested here: the record actually
captures what the decision was, and nothing in this module can fail a run that
has already submitted a team.
"""
from __future__ import annotations

import base64
import json

import pandas as pd
import pytest

import ledger


class _Plan:
    def __init__(self, squad, xi, captain, vice, transfers_in=(), transfers_out=(), hits=0):
        self.squad = squad
        self.xi = xi
        self.bench = squad[~squad["id"].isin(xi["id"])]
        self.captain = captain
        self.vice_captain = vice
        self.transfers_in = list(transfers_in)
        self.transfers_out = list(transfers_out)
        self.hits = hits


@pytest.fixture
def plan_and_scored():
    scored = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "web_name": ["Raya", "Davis", "Haaland", "Dubravka"],
        "team_name": ["ARS", "IPS", "MCI", "TOT"],
        # FPL element_type, as the real scored frame carries it. This fixture
        # used to say "GK"/"DEF", a shape production never produces.
        "position": [1, 2, 4, 1],
        "cost": [6.0, 4.0, 15.5, 4.0],
        "xp_next": [3.85, 2.81, 5.87, 0.18],
        "xp_horizon": [14.0, 10.2, 21.3, 0.6],
        "p_start": [0.95, 0.60, 0.98, 0.05],
        "availability": [1.0, 1.0, 1.0, 1.0],
        "exp_minutes": [88.0, 55.0, 89.0, 4.0],
        "sd_next": [2.1, 2.4, 3.9, 0.3],
    })
    squad = scored.copy()
    xi = scored[scored["id"].isin([1, 2, 3])]
    return _Plan(squad, xi, captain=3, vice=1, transfers_in=[2], transfers_out=[9], hits=0), scored


def _rec(plan, scored, **kw):
    base = dict(event_id=4, plan=plan, scored=scored, captain=plan.captain,
                vice=plan.vice_captain, chip=None,
                transfers_in=plan.transfers_in, transfers_out=plan.transfers_out,
                hits=plan.hits)
    base.update(kw)
    return ledger.build_record(**base)


def test_the_record_captures_the_decision(plan_and_scored):
    plan, scored = plan_and_scored
    r = _rec(plan, scored)

    assert r["event"] == 4 and r["captain"] == 3 and r["vice_captain"] == 1
    assert r["transfers_in"] == [2] and r["transfers_out"] == [9]
    assert {p["id"] for p in r["players"]} == {1, 2, 3, 4}
    assert r["schema"] == ledger.SCHEMA_VERSION


def test_it_records_the_xp_that_justified_each_bench_decision(plan_and_scored):
    """
    The whole point. Davis was benched at 2.81 xP and scored 14 in GW4, and
    that number was only recoverable from a run log that expires.
    """
    plan, scored = plan_and_scored
    players = {p["id"]: p for p in _rec(plan, scored)["players"]}

    davis = players[2]
    assert davis["started"] is True
    assert davis["xp_next"] == pytest.approx(2.81)
    assert davis["p_start"] == pytest.approx(0.60)

    benched = players[4]
    assert benched["started"] is False
    assert benched["xp_next"] == pytest.approx(0.18)


def test_the_captain_is_identifiable_without_cross_referencing(plan_and_scored):
    plan, scored = plan_and_scored
    players = {p["id"]: p for p in _rec(plan, scored)["players"]}
    assert players[3]["is_captain"] is True
    assert sum(p["is_captain"] for p in players.values()) == 1
    assert sum(p["is_vice_captain"] for p in players.values()) == 1


def test_a_record_is_json_serialisable(plan_and_scored):
    """numpy floats are not, and this is written straight to the API."""
    plan, scored = plan_and_scored
    json.dumps(_rec(plan, scored))


# ------------------------------------------------------ it may never fail a run

class _FakeRepo:
    def __init__(self, branch_exists=True, put_ok=True):
        self.branch_exists = branch_exists
        self.put_ok = put_ok
        self.written = []
        self.branches_created = []

    def ensure_branch(self, branch, from_branch="main"):
        if not self.branch_exists:
            self.branches_created.append(branch)
        return True

    def get_file(self, path, branch):
        return None, None

    def put_file(self, path, content, branch, message):
        self.written.append((path, content, branch, message))
        return self.put_ok


def test_a_successful_write_goes_to_its_own_branch(plan_and_scored):
    plan, scored = plan_and_scored
    gh = _FakeRepo()
    assert ledger.record_submission(gh, event_id=4, plan=plan, scored=scored,
                                    captain=3, vice=1, chip=None,
                                    transfers_in=[2], transfers_out=[9], hits=0) is True
    path, content, branch, msg = gh.written[0]
    assert branch == ledger.LEDGER_BRANCH != "main", "the ledger must not land on main"
    assert path == "predictions/gw04.json"
    assert json.loads(content)["event"] == 4
    assert "GW4" in msg


def test_a_broken_record_build_does_not_raise(plan_and_scored):
    """
    The build used to be evaluated as an argument to write(), outside any
    protection, so one unexpected column could raise AFTER the team had been
    submitted.
    """
    plan, scored = plan_and_scored
    gh = _FakeRepo()
    assert ledger.record_submission(gh, event_id=4, plan=object(), scored=scored,
                                    captain=3, vice=1, chip=None,
                                    transfers_in=[], transfers_out=[], hits=0) is False
    assert gh.written == []


def test_a_failed_api_write_does_not_raise(plan_and_scored):
    plan, scored = plan_and_scored
    gh = _FakeRepo(put_ok=False)
    assert ledger.record_submission(gh, event_id=4, plan=plan, scored=scored,
                                    captain=3, vice=1, chip=None,
                                    transfers_in=[], transfers_out=[], hits=0) is False


def test_no_credentials_is_a_quiet_no_op(plan_and_scored):
    plan, scored = plan_and_scored
    assert ledger.record_submission(None, event_id=4, plan=plan, scored=scored,
                                    captain=3, vice=1, chip=None,
                                    transfers_in=[], transfers_out=[], hits=0) is False


def test_an_exploding_repo_client_does_not_escape(plan_and_scored):
    class Boom:
        def ensure_branch(self, *a, **k): raise RuntimeError("network gone")
    plan, scored = plan_and_scored
    assert ledger.record_submission(Boom(), event_id=4, plan=plan, scored=scored,
                                    captain=3, vice=1, chip=None,
                                    transfers_in=[], transfers_out=[], hits=0) is False


# ------------------------------------------------- the self-test
#
# The ledger's first production write would otherwise land ninety seconds
# before a deadline, on a path no test covers against the live API, with a
# silent failure mode that looks exactly like the normal empty state.


def test_the_selftest_passes_when_the_api_cooperates(capsys):
    class _Ok:
        def __init__(self): self.writes = []
        def ensure_branch(self, b, from_branch="main"): return True
        def put_file(self, path, content, branch, message):
            self.writes.append(path); return True
        def get_file(self, path, branch): return ("e30=", "abc123")

    gh = _Ok()
    assert ledger.selftest(gh) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert len(gh.writes) == 2, "must exercise create AND overwrite"


def test_the_selftest_fails_loudly_when_the_branch_cannot_be_created(capsys):
    """
    The real failure: a GH_PAT with Actions and Secrets but not Contents write
    403s here, and in production that is one log warning and a green run.
    """
    class _NoBranch:
        def ensure_branch(self, b, from_branch="main"): return False
        def put_file(self, *a, **k): raise AssertionError("should not be reached")
        def get_file(self, *a, **k): return (None, None)

    assert ledger.selftest(_NoBranch()) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "Contents: write" in out


def test_the_selftest_catches_a_broken_overwrite(capsys):
    """
    The sha-on-update path is the one the unit-test fake can never exercise,
    because it always reports no existing sha. Without it every gameweek after
    the first would 409 forever.
    """
    class _FirstWriteOnly:
        def __init__(self): self.n = 0
        def ensure_branch(self, b, from_branch="main"): return True
        def get_file(self, *a, **k): return ("e30=", "abc123")
        def put_file(self, *a, **k):
            self.n += 1
            return self.n == 1

    assert ledger.selftest(_FirstWriteOnly()) == 1
    assert "OVERWRITE" in capsys.readouterr().out


def test_the_probe_cannot_collide_with_a_real_gameweek(capsys):
    """Event 0 is not a gameweek; a probe must never overwrite real data."""
    written = {}

    class _Rec:
        def ensure_branch(self, b, from_branch="main"): return True
        def get_file(self, *a, **k): return ("e30=", "s")
        def put_file(self, path, content, branch, message):
            written[path] = content; return True

    ledger.selftest(_Rec())
    assert list(written) == ["predictions/gw00.json"]
    assert all("gw0" in p for p in written)
