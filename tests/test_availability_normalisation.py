"""
Availability has to survive the shirt normaliser.

`_normalise_starts_within_team` divides a fixed number of shirts by relative
standing. That makes it purely relative, so folding availability into the
weights before allocating means a dominant first choice barely moves when he is
flagged - his share of the group is unchanged. `rate**3.0` makes every settled
goalkeeper dominant, and the result was that five of eleven first-choice
keepers in the committed snapshot retained EXACTLY 0.000 of a 25% availability
cut: Pickford, Leno, Verbruggen, Henderson and Sels could not be marked down at
all.

That is not a goalkeeper quirk. It silently disabled every availability signal
those players could receive - injury flags, chance_of_playing, news decay and
card bans alike.

Fix: allocate on ability, scale by availability, redistribute the freed shirts
to team-mates who can play.
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
def model():
    files = sorted(SNAPS.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(files[0])
    fixtures = X.load_snapshot(sorted(SNAPS.glob("fixtures_*.json.gz"), reverse=True)[0])
    return X.XPModel(bootstrap, fixtures, priors.build_priors(), X.ModelConfig(horizon=1))


def _rate(model):
    return model.players["start_rate"].clip(0.0, 1.0).fillna(0.0)


def _regulars(model, position=None):
    df = model.players
    sel = _rate(model) > 0.7
    if position is not None:
        sel &= df["position"] == position
    return list(df.index[sel])


def test_a_flagged_goalkeeper_can_actually_be_marked_down(model):
    """The regression. Retention was 0.000 for five of eleven first choices."""
    rate = _rate(model)
    ones = pd.Series(1.0, index=rate.index)
    base = model._normalise_starts_within_team(rate, ones)

    keepers = _regulars(model, position=1)
    if not keepers:
        pytest.skip("snapshot has no established keepers")

    for i in keepers:
        avail = ones.copy()
        avail.loc[i] = 0.75
        cut = float(model._normalise_starts_within_team(rate, avail).loc[i])
        full = float(base.loc[i])
        retained = (1.0 - cut / full) / 0.25
        assert retained > 0.75, (
            f"{model.players.loc[i, 'web_name']} retained only {retained:.3f} "
            f"of a 25% availability cut ({full:.4f} -> {cut:.4f})"
        )


def test_every_position_retains_an_availability_cut(model):
    rate = _rate(model)
    ones = pd.Series(1.0, index=rate.index)
    base = model._normalise_starts_within_team(rate, ones)
    for i in _regulars(model):
        avail = ones.copy()
        avail.loc[i] = 0.5
        cut = float(model._normalise_starts_within_team(rate, avail).loc[i])
        retained = (1.0 - cut / float(base.loc[i])) / 0.5
        assert 0.75 < retained < 1.25, (
            f"{model.players.loc[i, 'web_name']} retained {retained:.3f}"
        )


def test_teams_still_field_exactly_eleven(model):
    """The property the normaliser exists for, which the fix must not break."""
    rate = _rate(model)
    ones = pd.Series(1.0, index=rate.index)
    out = model._normalise_starts_within_team(rate, ones)
    df = model.players
    totals = out.groupby(df["team"]).sum()
    assert totals.min() == pytest.approx(11.0, abs=1e-6)
    assert totals.max() == pytest.approx(11.0, abs=1e-6)
    keepers = out[df["position"] == 1].groupby(df["team"]).sum()
    assert keepers.min() == pytest.approx(1.0, abs=1e-6)


def test_an_unavailable_first_choice_promotes_his_deputy(model):
    """
    The behaviour the old ordering got right, and the reason availability was
    folded into the weights in the first place. It must survive.
    """
    rate = _rate(model)
    ones = pd.Series(1.0, index=rate.index)
    base = model._normalise_starts_within_team(rate, ones)
    df = model.players

    keepers = _regulars(model, position=1)
    if not keepers:
        pytest.skip("snapshot has no established keepers")
    first = keepers[0]
    team = df.loc[first, "team"]
    deputies = df.index[(df["team"] == team) & (df["position"] == 1) & (df.index != first)]
    if not len(deputies):
        pytest.skip("no deputy keeper on the books")

    avail = ones.copy()
    avail.loc[first] = 0.0
    after = model._normalise_starts_within_team(rate, avail)

    assert float(after.loc[first]) == pytest.approx(0.0, abs=1e-9)
    assert float(after.loc[deputies].max()) > float(base.loc[deputies].max()) + 0.3, \
        "the shirt must go to the understudy, not evaporate"
    group = after[(df["position"] == 1) & (df["team"] == team)].sum()
    assert float(group) == pytest.approx(1.0, abs=1e-6)


def test_nobody_starts_more_often_than_he_is_available(model):
    rate = _rate(model)
    avail = pd.Series(0.4, index=rate.index)
    out = model._normalise_starts_within_team(rate, avail)
    assert (out <= avail + 1e-9).all(), "start probability exceeded availability"


# ---------------------------------------------------------------------------
# Edge cases on the redistribution helper itself. These are unit-level on
# purpose: the tests above already prove the helper is reached from production,
# so these are free to hammer the arithmetic at its boundaries.
# ---------------------------------------------------------------------------


def test_a_wholly_unavailable_group_allocates_nothing():
    out = X._redistribute_unavailable(pd.Series([0.9, 0.1]), pd.Series([0.0, 0.0]), 1.0)
    assert (out == 0.0).all()


def test_a_lone_player_is_capped_at_his_own_availability():
    out = X._redistribute_unavailable(pd.Series([1.0]), pd.Series([0.5]), 1.0)
    assert out.iloc[0] <= 0.5 + 1e-9


def test_a_squad_thinner_than_its_shirts_does_not_diverge():
    """Three players cannot fill ten shirts; the loop must stop, not spin."""
    out = X._redistribute_unavailable(pd.Series([0.98] * 3), pd.Series([1.0] * 3), 10.0)
    assert np.isfinite(out).all()
    assert (out <= 0.98 + 1e-9).all()


def test_a_fully_fit_squad_is_left_alone():
    pecking = pd.Series([0.5, 0.3, 0.2])
    out = X._redistribute_unavailable(pecking, pd.Series([1.0] * 3), 1.0)
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)


def test_an_empty_group_is_handled():
    out = X._redistribute_unavailable(pd.Series(dtype=float), pd.Series(dtype=float), 1.0)
    assert len(out) == 0


def test_fuzz_never_hands_out_more_than_availability():
    """
    The one invariant that must hold whatever the inputs: nobody starts more
    often than he is available to. Randomised because the redistribution loop
    has several branches and hand-picked cases miss them.
    """
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(1, 12))
        pecking = pd.Series(rng.random(n))
        shirts = min(10.0, n * 0.98)
        pecking = pecking / pecking.sum() * shirts
        avail = pd.Series(rng.random(n))
        out = X._redistribute_unavailable(pecking, avail, shirts)
        assert (out <= avail + 1e-9).all()
        assert (out >= -1e-12).all()
        assert np.isfinite(out).all()
