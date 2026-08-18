"""
Defensive contribution across a club transfer, and stale pre-season totals.

Defensive contribution is the one scoring term that belongs as much to the side
as to the player: it counts tackles, interceptions, clearances and recoveries,
and a team that keeps the ball has fewer of them to make. Measured over 2025-26
the spread runs 0.86 to 1.13 relative to the league mean. Because the award is a
threshold - ten actions for a defender, twelve for a midfielder - a rate sitting
just above the line is much more fragile than that spread suggests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import priors
import xp_model as X


@pytest.fixture(scope="module")
def prior_set():
    seasons = priors.available_seasons()
    if not seasons:
        pytest.skip("no history on disk")
    return priors.build_priors()


def test_club_factors_are_centred_and_bounded(prior_set):
    """
    Factors are relative to the league mean, so they must straddle 1.0 and stay
    within touching distance of it. A factor of 3.0 would mean a club whose
    players make three times the league's defensive actions, which is not a
    thing that happens - it would mean the sample was too small.
    """
    factors = prior_set.team_dc_factor
    assert factors, "no DC club factors were built"
    values = np.array(list(factors.values()))
    assert values.min() < 1.0 < values.max()
    assert values.min() > 0.5 and values.max() < 2.0
    assert 0.9 < values.mean() < 1.1


def test_every_player_prior_is_attributed_to_a_club(prior_set):
    team = prior_set.player_dc_team
    assert team is not None and not team.empty
    assert team.notna().all()
    assert team.index.is_unique, "a player's rate cannot come from two clubs at once"


def _model(bootstrap, fixtures, prior_set):
    return X.XPModel(bootstrap, fixtures, prior_set, X.ModelConfig(horizon=1))


def test_a_player_who_stayed_put_is_left_alone(prior_set):
    """The correction must be inert for everyone who did not move."""
    codes = list(prior_set.player_dc_team.index[:3])
    df = pd.DataFrame({
        "code": codes,
        "dc90": [10.0, 8.0, 6.0],
        # Same club his rate was earned at.
        "team_code": [int(prior_set.player_dc_team[c]) for c in codes],
    })
    model = X.XPModel.__new__(X.XPModel)
    model.priors = prior_set
    out = model._adjust_dc_for_club_move(df, pd.Series(0.0, index=df.index))
    assert np.allclose(out.to_numpy(), df["dc90"].to_numpy())


def test_moving_to_a_lower_volume_club_lowers_the_rate(prior_set):
    """Direction is the whole point: possession up, defensive actions down."""
    factors = prior_set.team_dc_factor
    busiest = max(factors, key=factors.get)
    quietest = min(factors, key=factors.get)
    code = int(prior_set.player_dc_team.index[0])

    model = X.XPModel.__new__(X.XPModel)
    model.priors = prior_set
    # Pin the player's prior club to the busiest side, then move him to the
    # quietest one, so the expected direction is unambiguous.
    model.priors = priors.PriorSet(
        players=prior_set.players,
        teams=prior_set.teams,
        league_mean_goals=prior_set.league_mean_goals,
        positional=prior_set.positional,
        team_dc_factor=factors,
        player_dc_team=pd.Series({code: busiest}),
    )
    df = pd.DataFrame({"code": [code], "dc90": [12.0], "team_code": [quietest]})
    out = model._adjust_dc_for_club_move(df, pd.Series(0.0, index=df.index))
    assert out.iloc[0] < 12.0

    df_up = pd.DataFrame({"code": [code], "dc90": [12.0], "team_code": [busiest]})
    same = model._adjust_dc_for_club_move(df_up, pd.Series(0.0, index=df_up.index))
    assert same.iloc[0] == pytest.approx(12.0)


def test_the_correction_fades_as_the_new_club_supplies_evidence(prior_set):
    """
    Minutes played for the new club need no correction - they were earned in
    the style being corrected for - so the rescaling applies only to the prior
    share of the blend.
    """
    factors = prior_set.team_dc_factor
    busiest, quietest = max(factors, key=factors.get), min(factors, key=factors.get)
    code = int(prior_set.player_dc_team.index[0])
    model = X.XPModel.__new__(X.XPModel)
    model.priors = priors.PriorSet(
        players=prior_set.players, teams=prior_set.teams,
        league_mean_goals=prior_set.league_mean_goals, positional=prior_set.positional,
        team_dc_factor=factors, player_dc_team=pd.Series({code: busiest}),
    )
    df = pd.DataFrame({"code": [code], "dc90": [12.0], "team_code": [quietest]})

    all_prior = model._adjust_dc_for_club_move(df, pd.Series([0.0])).iloc[0]
    half = model._adjust_dc_for_club_move(df, pd.Series([0.5])).iloc[0]
    all_current = model._adjust_dc_for_club_move(df, pd.Series([1.0])).iloc[0]

    assert all_prior < half < all_current
    assert all_current == pytest.approx(12.0), "a full season of new-club evidence needs no correction"


def test_stale_preseason_totals_are_ignored():
    """
    Before a ball is kicked the bootstrap still carries last season's totals -
    400 players with up to 3420 minutes and no gameweek finished. Blending them
    in as current-season evidence weights them at 86% for an ever-present,
    double-counting a season the priors already hold and routing around the
    shrinkage that exists to stop a raw rate being trusted whole.
    """
    from pathlib import Path
    snapshots = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    files = sorted(snapshots.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not files:
        pytest.skip("no bootstrap snapshot committed")
    bootstrap = X.load_snapshot(files[0])
    fixtures = X.load_snapshot(sorted(snapshots.glob("fixtures_*.json.gz"), reverse=True)[0])

    ps = priors.build_priors()
    played = [e for e in bootstrap["elements"] if (e.get("minutes") or 0) > 0]
    assert played, "snapshot has no minutes to be stale about"

    # Force the pre-season state, then confirm the model takes rates from the
    # priors rather than from those totals.
    pre = {**bootstrap, "events": [{**e, "finished": False, "is_current": False}
                                   for e in bootstrap["events"]]}
    frame = X.XPModel(pre, fixtures, ps, X.ModelConfig(horizon=1))._build_player_frame()

    joined = frame.join(ps.players[["xg90"]], on="code", rsuffix="_prior")
    both = joined.dropna(subset=["xg90_prior"])
    both = both[both["minutes"] > 500]
    assert len(both) > 50, "not enough overlap to make the comparison meaningful"
    # dc90 is rescaled for movers, so xg90 is the clean column to compare on.
    assert np.allclose(both["xg90"], both["xg90_prior"], atol=1e-9), (
        "pre-season rates must come from the priors alone"
    )


@pytest.mark.parametrize("frame,expected", [
    ({"position": ["GK", "DEF", "MID"]}, [1, 2, 3]),
    ({"position": [1, 2, 3]}, [1, 2, 3]),
    ({"element_type": [1, 2, 3]}, [1, 2, 3]),
    ({"element_type": [1, 2, 3], "position": ["GK", "DEF", "MID"]}, [1, 2, 3]),
])
def test_position_schema_variants_are_all_understood(frame, expected):
    """
    Regression. The historical dataset is third-party and its schema drifts:
    `position` is a name in some seasons, an integer in others, and
    `element_type` is present directly in others again. Assuming one shape
    crashed a CI run on a freshly fetched copy while passing against the copy
    already on disk - the exact failure a live submission must never hit.
    """
    out = priors._with_element_type(pd.DataFrame(frame))
    assert out is not None
    assert list(out["element_type"]) == expected


@pytest.mark.parametrize("frame", [
    {"minutes": [1, 2, 3]},                       # no position column at all
    {"position": [None, None]},                   # present but empty
    {"position": ["nonsense", "rubbish"]},        # unmappable names
])
def test_an_unusable_season_is_skipped_not_raised(frame):
    """A schema we cannot read must degrade the priors, never break the run."""
    assert priors._with_element_type(pd.DataFrame(frame)) is None
