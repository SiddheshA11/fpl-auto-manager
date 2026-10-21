"""
The preview tool must score on the same model as the weekly run.

`manager.py` hands XPModel `recent_minutes` from `event/{gw}/live/`; HANDOFF
credits those lags with the largest single accuracy gain the model has had
(points R2 0.269 -> 0.319). `recommend.py` built XPModel without them, so the
plan a human previews before a deadline was scored on a weaker model than the
one that gets submitted - the same shape as its earlier ownership_weight and
horizon_weight bugs.

So, as in test_recommend_tilt.py, this records what XPModel was actually
*handed* through a real `recommend.main()` run. The HTTP layer is stubbed; the
committed snapshot is pre-season, so the stub marks GW1-3 finished, otherwise
an empty dict would be the correct answer and the test would prove nothing.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import recommend as R
import xp_model as X
from config import ENDPOINTS
from fetch_data import LIVE_ENDPOINTS

REPO = Path(__file__).resolve().parent.parent
SNAPSHOTS = REPO / "data" / "snapshots"

FINISHED = 3


@pytest.fixture(autouse=True)
def _snapshot_exists():
    if not sorted(SNAPSHOTS.glob("bootstrap-static_*.json.gz")):
        pytest.skip("no bootstrap snapshot committed")


@pytest.fixture(scope="module")
def game_state():
    bootstrap = copy.deepcopy(X.load_snapshot(X.latest_snapshot(SNAPSHOTS, "bootstrap-static")))
    fixtures = X.load_snapshot(X.latest_snapshot(SNAPSHOTS, "fixtures"))
    for e in bootstrap["events"]:
        e["finished"] = e["id"] <= FINISHED
        e["is_current"] = e["id"] == FINISHED
        e["is_next"] = e["id"] == FINISHED + 1
    return bootstrap, fixtures


@pytest.fixture(scope="module")
def live_minutes(game_state):
    """Distinct minutes per player per gameweek, so order and source both show."""
    pids = [int(p["id"]) for p in game_state[0]["elements"][:4]]
    return {gw: {pid: float(10 * gw + i) for i, pid in enumerate(pids)}
            for gw in range(1, FINISHED + 1)}


@pytest.fixture
def recorded(monkeypatch):
    seen: list[dict] = []
    real = X.XPModel

    class Recording(real):
        def __init__(self, *a, **kw):
            seen.append(dict(kw))
            super().__init__(*a, **kw)

    monkeypatch.setattr(R.X, "XPModel", Recording)
    return seen


def _stub_http(monkeypatch, game_state, live_minutes, bootstrap_fails=False):
    bootstrap, fixtures = game_state
    urls: list[str] = []
    live_urls = {ENDPOINTS["event_live"].format(event_id=gw): gw for gw in live_minutes}

    def fake_get(url):
        urls.append(url)
        if url == LIVE_ENDPOINTS["bootstrap-static"]:
            if bootstrap_fails:
                raise RuntimeError("stubbed outage")
            return json.dumps(bootstrap).encode()
        if url == LIVE_ENDPOINTS["fixtures"]:
            return json.dumps(fixtures).encode()
        if url in live_urls:
            mins = live_minutes[live_urls[url]]
            return json.dumps({"elements": [
                {"id": pid, "stats": {"minutes": m}} for pid, m in mins.items()
            ]}).encode()
        raise AssertionError(f"recommend.py fetched an unexpected URL: {url}")

    monkeypatch.setattr(R, "_get", fake_get)
    return urls


def test_recommend_hands_the_model_its_recent_minutes(monkeypatch, recorded, game_state, live_minutes):
    urls = _stub_http(monkeypatch, game_state, live_minutes)
    monkeypatch.setattr("sys.argv", ["recommend.py", "--build", "--horizon", "1"])

    assert R.main() == 0
    assert recorded, "recommend.py never constructed an XPModel"
    got = recorded[0].get("recent_minutes")
    assert got, (
        "recommend.py built XPModel without recent_minutes while manager.py "
        "passes them, so the preview scores on a different model from the run"
    )
    expected = {pid: [live_minutes[gw][pid] for gw in range(FINISHED, 0, -1)]
                for pid in live_minutes[1]}
    assert got == expected, "lags must be most recent first, from event/{gw}/live/"
    assert [u for u in urls if "/live/" in u] == [
        ENDPOINTS["event_live"].format(event_id=gw) for gw in range(FINISHED, 0, -1)
    ]


def test_offline_degrades_to_no_recent_minutes(monkeypatch, recorded, game_state, live_minutes):
    urls = _stub_http(monkeypatch, game_state, live_minutes)
    monkeypatch.setattr("sys.argv", ["recommend.py", "--offline", "--build", "--horizon", "1"])

    assert R.main() == 0
    assert recorded[0].get("recent_minutes") == {}
    assert urls == [], "--offline must not touch the network"


def test_a_failed_live_fetch_does_not_go_on_to_fetch_minutes(monkeypatch, recorded, game_state, live_minutes):
    """
    Once the bootstrap has fallen back to the snapshot the API is down; five
    more gameweeks of retries against it would only delay the same answer.
    """
    urls = _stub_http(monkeypatch, game_state, live_minutes, bootstrap_fails=True)
    monkeypatch.setattr("sys.argv", ["recommend.py", "--build", "--horizon", "1"])

    assert R.main() == 0
    assert recorded[0].get("recent_minutes") == {}
    assert not [u for u in urls if "/live/" in u]
