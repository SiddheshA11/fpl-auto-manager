"""
What the weekly run actually hands the ledger, through a real run.

`build_record` accepted `tilt_source` from the day it was written, and
manager.py never passed it: every record said null, including the one field
that would show a gameweek silently tilting on the template after the rival
fan-out failed. Unit tests of build_record could not see that, because they
are the ones passing the argument.
"""
from __future__ import annotations

import json

import pytest

import ledger
import manager
from test_pipeline import StubClient, _snapshot
from test_recovery_paths import _squad_and_costs


class _RecordingRepo:
    """Stands in for github_api.GitHubRepo. No marker set, every write lands."""

    def __init__(self):
        self.files = {}

    def get_variable(self, name):
        return None

    def read_variable(self, name):
        return True, None

    def set_variable(self, name, value):
        return True

    def ensure_branch(self, branch, from_branch="main"):
        return True

    def get_file(self, path, branch):
        return None, None

    def put_file(self, path, content, branch, message):
        self.files[path] = content
        return True


@pytest.fixture(scope="module")
def recorded():
    bootstrap, fixtures = _snapshot("bootstrap-static"), _snapshot("fixtures")
    owned, costs = _squad_and_costs(bootstrap, fixtures)
    client = StubClient(bootstrap, fixtures, owned, costs)
    repo = _RecordingRepo()
    mp = pytest.MonkeyPatch()
    mp.setattr(manager, "FPLClient", lambda: client)
    mp.setattr(manager, "_github", lambda: repo)
    try:
        result = manager.run_weekly_cycle(dry_run=False)
    finally:
        mp.undo()
    assert result is not None and result["status"] == "submitted"
    (path, body), = repo.files.items()
    assert path.startswith(f"{ledger.LEDGER_DIR}/gw")
    return bootstrap, client, json.loads(body)


def test_the_run_records_whose_ownership_it_tilted_on(recorded):
    _, _, record = recorded
    # conftest blocks the rival fan-out, so this run falls back to the template.
    assert record["tilt_source"] == "template"


def test_the_pool_is_every_player_fpl_lists(recorded):
    """
    Checked against bootstrap's own element list rather than against the
    scored frame, which is what the ledger read from and so cannot disagree
    with it.
    """
    bootstrap, _, record = recorded
    pool = record["pool"]
    assert set(pool["id"]) == {int(e["id"]) for e in bootstrap["elements"]}
    for col in ledger.POOL_COLUMNS:
        assert len(pool[col]) == len(pool["id"]), col
    positions = {int(e["id"]): int(e["element_type"]) for e in bootstrap["elements"]}
    assert all(positions[i] == p for i, p in zip(pool["id"], pool["position"]))


def test_the_pool_reaches_below_the_squad(recorded):
    """The point of the pool is the players the model did not choose."""
    _, _, record = recorded
    xp = [v for v in record["pool"]["xp_next"] if v is not None]
    squad_min = min(p["xp_next"] for p in record["players"])
    assert sum(v < squad_min for v in xp) > 100
    assert min(xp) < 0.1


def test_the_pool_agrees_with_the_squad_rows(recorded):
    _, client, record = recorded
    pool = dict(zip(record["pool"]["id"], record["pool"]["xp_next"]))
    for p in record["players"]:
        assert pool[p["id"]] == pytest.approx(p["xp_next"], abs=1e-4)
    submitted = {p["element"] for p in client.submitted_lineup["picks"]}
    assert submitted == {p["id"] for p in record["players"]}


def test_the_record_is_strict_json_and_small(recorded):
    _, _, record = recorded
    body = json.dumps(record, allow_nan=False)
    assert len(body) < 120_000
