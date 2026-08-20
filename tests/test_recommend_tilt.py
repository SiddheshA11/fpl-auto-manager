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
