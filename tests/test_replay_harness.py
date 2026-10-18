"""
The decision-replay harness's own correctness.

Its outputs are differences between arms, and a difference is only worth
reading if an arm that changes nothing reproduces the baseline exactly. These
pin the pieces that claim: the cached scoring equals simulate's, a derived
horizon equals scoring at that horizon, the placebo is reproducible and
exchangeable, and
a full run of null arms gives exactly zero in every window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import priors
import replay_harness as R
import simulate as S

SEASON = "2025-26"

# Captured before anything here scores. Comparing against R._production()
# would be circular: it reads the very global a leak would have changed.
import xp_model as _X  # noqa: E402
PRODUCTION_ALPHA = _X.OUTFIELD_COMPETITION_ALPHA


@pytest.fixture(scope="module")
def data():
    if SEASON not in priors.available_seasons():
        pytest.skip(f"{SEASON} not on disk")
    return S.load_season(SEASON)


# ------------------------------------------------------------------ scoring

def test_the_harness_scores_exactly_as_simulate_does(data):
    """score_with_order duplicates score_gameweek to get the order out; pin it."""
    scored, order = R.score_with_order(data, 10, R.SCORE_HORIZON)
    pd.testing.assert_frame_equal(scored, S.score_gameweek(data, 10, R.SCORE_HORIZON))
    assert sorted(order) == sorted(scored["id"])


@pytest.mark.parametrize("horizon", [3, 5, 7])
def test_a_derived_horizon_is_bit_identical_to_scoring_at_it(data, horizon):
    """
    Including row order: the model's sort is not stable and row order reaches
    the solver. An id-sorted reconstruction got the order wrong on every
    gameweek tried.
    """
    gw = 12
    scored, order = R.score_with_order(data, gw, R.SCORE_HORIZON)
    derived = R.derive_horizon(scored, order, gw, horizon)
    direct = S.score_gameweek(data, gw, horizon)
    assert list(derived["id"]) == list(direct["id"])
    assert np.array_equal(derived["xp_horizon"].to_numpy(), direct["xp_horizon"].to_numpy(),
                          equal_nan=True)
    assert np.array_equal(derived["xp_next"].to_numpy(), direct["xp_next"].to_numpy(),
                          equal_nan=True)


def test_a_horizon_beyond_the_cache_is_refused(data):
    scored, order = R.score_with_order(data, 12, R.SCORE_HORIZON)
    with pytest.raises(ValueError):
        R.derive_horizon(scored, order, 12, R.SCORE_HORIZON + 1)


# -------------------------------------------------------------------- arms

def test_arm_specs_parse_onto_production():
    prod = R._production()
    assert R.parse_arm("same").resolved() == R.Arm(name="same").resolved()
    assert R.parse_arm("alpha+1e-6").resolved().alpha == pytest.approx(prod["alpha"] + 1e-6)
    assert R.parse_arm("horizon=6").resolved().horizon == 6
    assert R.parse_arm("ftv=0.0").resolved().free_transfer_value == 0.0
    noise = R.parse_arm("noise=0.02@3")
    assert (noise.noise_eps, noise.noise_seed) == (0.02, 3)
    with pytest.raises(SystemExit):
        R.parse_arm("nonsense=1")


def test_an_arm_key_identifies_the_config_not_the_name():
    assert R.Arm(name="a").key() == R.Arm(name="b").key()
    assert R.Arm(horizon=6).key() != R.Arm().key()
    assert R.Arm(noise_eps=0.02, noise_seed=1).key() != R.Arm(noise_eps=0.02, noise_seed=2).key()


# ------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def short_run(tmp_path_factory, data):
    """
    A real run on GW2-9 of one season, in-process, into a throwaway cache:
    the identity arm, alpha +/- 1e-6, and two placebo seeds.
    """
    mp = pytest.MonkeyPatch()
    mp.setattr(R, "CACHE_DIR", tmp_path_factory.mktemp("replay_cache"))
    mp.setattr(R, "LAST_GW", 9)
    R._arm_frame.cache_clear()
    try:
        arms = [R.parse_arm(a) for a in
                ("same", "alpha+1e-6", "alpha-1e-6", "noise=0.05@1", "noise=0.05@2")]
        df = R.run(arms, [SEASON], L=3, T=2, workers=1)
    finally:
        R._arm_frame.cache_clear()
        mp.undo()
    return df


def test_every_arm_gets_every_window(short_run):
    # Eight played gameweeks; the first is the squad build, and a window of
    # three must fit, so starts are the 2nd through 6th: five windows.
    assert set(short_run.groupby("arm").size()) == {5}


@pytest.mark.parametrize("arm", ["same", "alpha+1e-6", "alpha-1e-6"])
def test_a_null_arm_reproduces_the_baseline_in_every_window(short_run, arm):
    g = short_run[short_run["arm"] == arm]
    assert (g["diff"] == 0).all() and (g["diff_terminal"] == 0).all()
    assert not g["flipped"].any()


def test_the_production_alpha_is_restored_after_scoring(short_run):
    import xp_model as X
    assert X.OUTFIELD_COMPETITION_ALPHA == PRODUCTION_ALPHA


def test_the_placebo_is_reproducible(data, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "CACHE_DIR", tmp_path)
    R._arm_frame.cache_clear()
    try:
        R._score_task((SEASON, 4, R._production()["alpha"]))
        a = R.arm_frame(R.parse_arm("noise=0.05@1"), SEASON, 4)
        R._arm_frame.cache_clear()
        b = R.arm_frame(R.parse_arm("noise=0.05@1"), SEASON, 4)
        c = R.arm_frame(R.parse_arm("noise=0.05@2"), SEASON, 4)
        base = R.arm_frame(R.parse_arm("same"), SEASON, 4)
    finally:
        R._arm_frame.cache_clear()
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.set_index("id")["xp_next"], c.set_index("id")["xp_next"])
    shift = (a.set_index("id")["xp_next"] - base.set_index("id")["xp_next"]).dropna()
    assert abs(shift.mean()) < 0.01 and 0.04 < shift.std() < 0.06
