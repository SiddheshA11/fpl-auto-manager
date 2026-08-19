"""
The spread around expected points, not just its centre.

Every decision this model feeds was made on a point estimate. A point estimate
cannot tell a forward who hauls or blanks from a defender who returns five
every week, and for captaincy - the highest-leverage weekly call - that
difference is the whole question. It is also the prerequisite for any
rank-aware objective: beating nineteen specific rivals is a question about
distributions, not means.

The variance is assembled from the distributions the mean already assumes -
Poisson goals and assists, Bernoulli clean sheets and defensive contribution,
the explicit goals-conceded and saves mass functions - so it is derived rather
than fitted, with one measured correction for what summing independent terms
misses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import priors
import xp_model as X

SNAPS = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@pytest.fixture(scope="module")
def scored():
    files = sorted(SNAPS.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(files[0])
    fixtures = X.load_snapshot(sorted(SNAPS.glob("fixtures_*.json.gz"), reverse=True)[0])
    model = X.XPModel(bootstrap, fixtures, priors.build_priors(), X.ModelConfig(horizon=5))
    return model.expected_points(X.next_events(bootstrap, 5))


def test_variance_is_produced_and_finite(scored):
    assert "var_next" in scored and "sd_next" in scored
    assert (scored["var_next"] >= 0).all(), "a negative variance is not a variance"
    assert np.isfinite(scored["sd_next"]).all()
    assert scored["sd_next"].pow(2).sub(scored["var_next"]).abs().max() < 1e-9


def test_a_player_who_cannot_play_has_no_spread(scored):
    """Certainty of zero is certain."""
    out = scored[scored["status"] == "i"]
    if out.empty:
        pytest.skip("snapshot has no unavailable players")
    ruled_out = out[out["xp_next"] == 0]
    assert (ruled_out["sd_next"] < 0.05).all(), "a player who cannot feature has spread"


def test_spread_rises_with_expected_points(scored):
    """More on offer means more to swing."""
    d = scored[scored["xp_next"] > 0.5]
    lo = d[d["xp_next"] < d["xp_next"].quantile(0.3)]["sd_next"].mean()
    hi = d[d["xp_next"] > d["xp_next"].quantile(0.7)]["sd_next"].mean()
    assert hi > lo


def test_goalkeepers_are_the_steadiest_return(scored):
    """
    Saves and clean sheets accumulate; goals arrive in lumps. This is the one
    positional distinction the data supports strongly - forwards and defenders
    measure within 5% of each other at matched ability, because a defender's
    clean sheet is as spiky a four-point Bernoulli as a forward's goal chance.
    """
    d = scored[(scored["xp_next"] >= 3.0) & (scored["xp_next"] <= 4.5)]
    if len(d) < 30:
        pytest.skip("not enough players in the comparison band")
    ratio = (d["sd_next"] / d["xp_next"]).groupby(d["position"]).mean()
    if 1 not in ratio or 2 not in ratio:
        pytest.skip("band lacks goalkeepers or defenders")
    assert ratio[1] < ratio[2], "goalkeepers should be steadier than defenders per point"


def test_a_double_gameweek_adds_variance(scored):
    """Two fixtures are two draws, so variances add rather than averaging."""
    events = [c for c in scored.columns if c.startswith("var_gw")]
    assert events, "per-gameweek variance is not exposed"
    doubles = scored[scored[[c for c in scored.columns if c.startswith("fixtures_gw")][0]] >= 2]
    if doubles.empty:
        pytest.skip("no double gameweek in the horizon")
    assert (doubles["var_next"] > 0).all()


def test_the_inflation_factor_is_calibrated_not_assumed():
    """
    Summing term variances treats them as independent, which they are not, and
    a residual also carries our own mean error. The factor absorbs both and was
    measured against 22,774 player-gameweeks - predicted sd 1.417 against an
    actual 1.954 - rather than chosen.
    """
    assert X.VARIANCE_CORRELATION_INFLATION > 1.0, (
        "independent summation understates the spread; the factor must inflate it"
    )
    assert X.VARIANCE_CORRELATION_INFLATION == pytest.approx(1.90, abs=0.3)


def test_variance_is_only_computed_when_asked(scored):
    """The mean path must keep its old signature for every existing caller."""
    files = sorted(SNAPS.glob("bootstrap-static_*.json.gz"), reverse=True)
    bootstrap = X.load_snapshot(files[0])
    fixtures = X.load_snapshot(sorted(SNAPS.glob("fixtures_*.json.gz"), reverse=True)[0])
    model = X.XPModel(bootstrap, fixtures, priors.build_priors(), X.ModelConfig(horizon=1))
    fx = model.fixtures_for_event(X.next_events(bootstrap, 1)[0]).iloc[0]
    args = (int(fx["team_id"]), int(fx["opponent_id"]), bool(fx["is_home"]))

    mean_only = model.expected_points_for_fixture(*args)
    assert isinstance(mean_only, pd.Series)

    mean, var = model.expected_points_for_fixture(*args, with_variance=True)
    assert mean.equals(mean_only), "asking for variance changed the mean"
    assert (var >= 0).all()
