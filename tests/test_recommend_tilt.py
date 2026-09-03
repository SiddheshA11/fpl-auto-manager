"""
The preview tool must optimise on the same objective as the weekly run.

`recommend.py` is what a human reaches for to check the plan before a deadline.
It passed no `ownership_weight`, so it silently ran at optimizer.py's module
default of 0.0 while manager.py ran at config.OWNERSHIP_WEIGHT. At the current
+0.20 the two disagree on the squad itself: the preview omitted Haaland and
came in 0.56 xP short of what would actually be submitted.

`deadline_check.py` carries a comment about having had this exact bug; this
caller was missed. So the test asserts what the optimiser was *handed*, not
that a helper computes the right number - a test of the latter passed
throughout the period the bug was live.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import optimizer as O
import recommend as R
import xp_model as X

REPO = Path(__file__).resolve().parent.parent
SNAPSHOTS = REPO / "data" / "snapshots"

# Deliberately not equal to optimizer.OWNERSHIP_WEIGHT (0.0), and not equal to
# config's current +0.20 either, so the assertion can only pass if the value
# actually flows from the module attribute into the solver.
SENTINEL = 0.37


class _Recorder:
    """Stands in for SquadOptimizer and records the kwargs it was constructed with."""

    seen: list[dict] = []

    def __init__(self, players, value_col=None, captain_col=None, **kwargs):
        _Recorder.seen.append(dict(kwargs))
        self._inner = O.SquadOptimizer(players, value_col, captain_col, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(autouse=True)
def _snapshot_exists():
    if not sorted(SNAPSHOTS.glob("bootstrap-static_*.json.gz")):
        pytest.skip("no bootstrap snapshot committed")


def test_recommend_hands_the_configured_tilt_to_the_optimiser(monkeypatch, capsys):
    _Recorder.seen = []
    monkeypatch.setattr(R, "SquadOptimizer", _Recorder)
    monkeypatch.setattr(R, "OWNERSHIP_WEIGHT", SENTINEL, raising=False)
    monkeypatch.setattr("sys.argv", ["recommend.py", "--offline", "--build", "--horizon", "1"])

    assert R.main() == 0

    assert _Recorder.seen, "recommend.py never constructed a SquadOptimizer"
    got = _Recorder.seen[0]
    assert "ownership_weight" in got, (
        "recommend.py passed no ownership_weight, so it silently optimises at "
        f"optimizer.OWNERSHIP_WEIGHT ({O.OWNERSHIP_WEIGHT}) while manager.py uses "
        "config.OWNERSHIP_WEIGHT - the preview and the submission disagree"
    )
    assert got["ownership_weight"] == pytest.approx(SENTINEL)


def test_recommend_and_manager_read_the_same_setting():
    """Both must read config, not optimizer's neutral module default."""
    import config
    import manager

    assert R.OWNERSHIP_WEIGHT is config.OWNERSHIP_WEIGHT
    assert manager.OWNERSHIP_WEIGHT is config.OWNERSHIP_WEIGHT


# ------------------------------------------------------------- horizon_weight
#
# The same class of bug, in the same file, one argument along. `recommend.py`
# passed no `horizon_weight`, so it used optimizer.py's default of 1.0 while
# manager.py passes sum(decay**i) ~= 3.64. The -4 for a hit was compared
# against `xp_horizon`, a decay-weighted sum over five gameweeks, so hits were
# priced roughly 3.6x too cheap and the preview recommended moves production
# refuses.
#
# Measured on the committed snapshot over six suboptimal squads x {0, 1} free
# transfers: 12 of 12 scenarios diverge, the preview taking the maximum two
# hits in every one where production takes none.


class _TransferRecorder:
    """Records the kwargs optimise_transfers was actually called with."""

    seen: list[dict] = []

    def __init__(self, players, value_col=None, captain_col=None, **kwargs):
        self._inner = O.SquadOptimizer(players, value_col, captain_col, **kwargs)

    def optimise_transfers(self, current_squad_ids, **kwargs):
        _TransferRecorder.seen.append(dict(kwargs))
        return self._inner.optimise_transfers(current_squad_ids, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def owned_squad():
    """
    A legal 15 from the committed snapshot. It has to be legal: an arbitrary
    fifteen ids violates the position and per-club constraints and the solver
    reports infeasible before any argument reaches it.
    """
    bootstrap, fixtures = R.load_game_state(offline=True)
    scored, _ = R.build_model(bootstrap, fixtures, 5)
    pool = R._exclude_unavailable(scored)
    sol = O.SquadOptimizer(pool, "xp_horizon", "xp_next").build_squad(100.0)
    return ",".join(str(int(i)) for i in sol.squad["id"])


def test_recommend_prices_hits_on_the_same_scale_as_the_weekly_run(monkeypatch, owned_squad):
    _TransferRecorder.seen = []
    monkeypatch.setattr(R, "SquadOptimizer", _TransferRecorder)
    squad = owned_squad
    monkeypatch.setattr("sys.argv", [
        "recommend.py", "--offline", "--transfer", "--squad", squad, "--horizon", "5",
    ])

    assert R.main() == 0
    assert _TransferRecorder.seen, "recommend.py never called optimise_transfers"
    got = _TransferRecorder.seen[0]

    assert "horizon_weight" in got, (
        "recommend.py passed no horizon_weight, so hits are priced against "
        "optimizer.py's default of 1.0 while the value column sums five "
        "decayed gameweeks - about 3.6x too cheap, and the preview recommends "
        "hits production will refuse"
    )
    expected = sum(X.ModelConfig(horizon=5).horizon_decay ** i for i in range(5))
    assert got["horizon_weight"] == pytest.approx(expected)
    assert got["horizon_weight"] > 3.0, "a weight near 1.0 means the bug is back"


def test_the_hit_price_tracks_the_requested_horizon(monkeypatch, owned_squad):
    """
    recommend.py takes --horizon as an argument where manager.py has a
    constant, so a hardcoded 3.64 would be wrong at every other horizon.
    """
    _TransferRecorder.seen = []
    monkeypatch.setattr(R, "SquadOptimizer", _TransferRecorder)
    monkeypatch.setattr("sys.argv", [
        "recommend.py", "--offline", "--transfer", "--squad", owned_squad, "--horizon", "3",
    ])

    assert R.main() == 0
    got = _TransferRecorder.seen[0]
    expected = sum(X.ModelConfig(horizon=3).horizon_decay ** i for i in range(3))
    assert got["horizon_weight"] == pytest.approx(expected)
    assert got["horizon_weight"] < 3.0, "horizon 3 must weigh less than horizon 5"
