"""
Yellow-card suspension risk.

Five bookings inside the first 19 matches is a one-match ban. The gameweek
after the fifth card already works in production without any model: FPL sets
status='s' and STATUS_AVAILABILITY zeroes the player. What no flag can cover is
the card that has not been shown yet - a player on four bookings is banned
somewhere inside a five-gameweek horizon 50.7% of the time (measured over
2020-26, n=1048 starting player-gameweeks on exactly four), and was priced at
full minutes for all five.

These assert through minutes_model and expected_points rather than against
_suspension_risk, because a test that exercises a helper does not prove the
helper is reached - four separate fixes have been deletable from production
with the whole suite green.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import priors
import xp_model as X

SNAPS = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@pytest.fixture(scope="module")
def state():
    files = sorted(SNAPS.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(files[0])
    fixtures = X.load_snapshot(sorted(SNAPS.glob("fixtures_*.json.gz"), reverse=True)[0])
    return bootstrap, fixtures, priors.build_priors()


def _played(bootstrap, n=8):
    """Mark the first n gameweeks finished, so the in-season path is live."""
    out = {**bootstrap, "events": [dict(e) for e in bootstrap["events"]]}
    for i, e in enumerate(out["events"]):
        e["finished"] = i < n
        e["is_current"] = i == n - 1
        e["is_next"] = i == n
    return out


def _with_cards(bootstrap, pid, cards, minutes=720):
    """Give one player a card count and the minutes to have earned it."""
    out = {**bootstrap, "elements": [dict(e) for e in bootstrap["elements"]]}
    for e in out["elements"]:
        if int(e["id"]) == pid:
            e["yellow_cards"] = cards
            e["minutes"] = minutes
            e["status"] = "a"
            e["chance_of_playing_next_round"] = None
    return out


def _a_regular(bootstrap, fixtures, ps):
    """
    An established OUTFIELD starter.

    Goalkeepers are deliberately excluded. Their group has one shirt and
    GK_COMPETITION_ALPHA = 3.0, so _normalise_starts_within_team allocates on
    relative standing alone and a first-choice keeper's availability cut is
    largely destroyed - Pickford and Leno retain 0.00 of a 25% cut. That is a
    pre-existing defect in the normaliser, not in this model, and it damps
    every availability signal a keeper receives, injury flags included.
    Picking a keeper here would measure that bug instead of this feature.
    """
    model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=5))
    frame = model.players
    idx = frame.index[(frame["start_rate"] > 0.8) & (frame["position"] != 1)]
    if not len(idx):
        pytest.skip("snapshot has no established outfield starters")
    return int(frame.loc[idx[0], "id"])


def _horizon_minutes(bootstrap, fixtures, ps, pid, horizon=5):
    """Expected minutes summed across the planning horizon, for one player."""
    model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=horizon))
    events = X.next_events(bootstrap, horizon)
    rows = model.players.index[model.players["id"] == pid]
    return sum(float(model.minutes_model(ev).loc[rows[0], "exp_minutes"]) for ev in events)


def test_four_yellows_costs_horizon_minutes(state):
    """The whole point: an unearned ban must be priced before it happens."""
    bootstrap, fixtures, ps = state
    bootstrap = _played(bootstrap)
    pid = _a_regular(bootstrap, fixtures, ps)

    clean = _horizon_minutes(_with_cards(bootstrap, pid, 0), fixtures, ps, pid)
    risky = _horizon_minutes(_with_cards(bootstrap, pid, 4), fixtures, ps, pid)

    assert risky < clean, "a player one booking from a ban must lose horizon minutes"
    # 50.7% of a one-match ban across five gameweeks, on an ~80-minute starter,
    # is tens of minutes - not a rounding difference. Measured at 52-57 across
    # the outfield regulars in the committed snapshot.
    assert clean - risky > 20.0, f"effect implausibly small: {clean - risky:.2f} minutes"


def test_risk_rises_with_the_card_count(state):
    bootstrap, fixtures, ps = state
    bootstrap = _played(bootstrap)
    pid = _a_regular(bootstrap, fixtures, ps)
    mins = [_horizon_minutes(_with_cards(bootstrap, pid, c), fixtures, ps, pid)
            for c in (0, 2, 4)]
    assert mins[0] >= mins[1] > mins[2], f"not monotone in card count: {mins}"


def test_a_player_already_banned_is_not_charged_twice(state):
    """
    status='s' already zeroes him. Adding an unearned-ban penalty on top would
    price one absence twice, and the five-card rule is spent once passed.
    """
    bootstrap, fixtures, ps = state
    bootstrap = _played(bootstrap)
    pid = _a_regular(bootstrap, fixtures, ps)
    # Five cards: the threshold is reached, so it can no longer be *crossed*.
    at5 = _horizon_minutes(_with_cards(bootstrap, pid, 5), fixtures, ps, pid)
    at4 = _horizon_minutes(_with_cards(bootstrap, pid, 4), fixtures, ps, pid)
    assert at5 > at4, "a spent threshold must stop costing minutes"


def test_preseason_does_not_read_last_seasons_cards(state):
    """
    Before a ball is kicked the bootstrap's `yellow_cards` still holds LAST
    season's totals. Ungated, that suspends half the league in GW1 - the same
    trap that halved the squad's expected points via a dtype check.
    """
    bootstrap, fixtures, ps = state
    assert not any(e.get("finished") for e in bootstrap["events"]), \
        "fixture is expected to be a pre-season snapshot"
    pid = _a_regular(bootstrap, fixtures, ps)

    clean = _horizon_minutes(_with_cards(bootstrap, pid, 0), fixtures, ps, pid)
    carded = _horizon_minutes(_with_cards(bootstrap, pid, 9), fixtures, ps, pid)
    assert carded == pytest.approx(clean), \
        "pre-season must ignore last season's card totals"


def test_the_next_gameweek_is_left_to_fpls_own_flag(state):
    """
    `ahead == 0` is the gameweek FPL has already judged. Modelling a ban there
    would fight the status flag rather than complement it.
    """
    bootstrap, fixtures, ps = state
    bootstrap = _played(bootstrap)
    pid = _a_regular(bootstrap, fixtures, ps)
    events = X.next_events(bootstrap, 5)

    def first_gw_minutes(cards):
        bs = _with_cards(bootstrap, pid, cards)
        model = X.XPModel(bs, fixtures, ps, X.ModelConfig(horizon=5))
        row = model.players.index[model.players["id"] == pid][0]
        return float(model.minutes_model(events[0]).loc[row, "exp_minutes"])

    assert first_gw_minutes(4) == pytest.approx(first_gw_minutes(0))


# ---------------------------------------------------------------------------
# The Poisson tail the risk model is built on. Unit-level on purpose: the tests
# above already prove the model is reached from production, so these are free
# to hammer the arithmetic where it would misbehave quietly.
# ---------------------------------------------------------------------------


def test_poisson_tail_matches_scipy():
    """Verified against an independent implementation, not against itself."""
    scipy_stats = pytest.importorskip("scipy.stats")
    need = pd.Series([1.0, 2.0, 3.0, 1.0, 5.0])
    lam = pd.Series([0.16, 0.16, 0.30, 0.0, 0.5])
    got = X.XPModel._poisson_at_least(need, lam)
    want = [1.0 - scipy_stats.poisson.cdf(n - 1, l) if l > 0 else 0.0
            for n, l in zip(need, lam)]
    assert got.tolist() == pytest.approx(want, abs=1e-12)


def test_no_exposure_means_no_risk():
    assert float(X.XPModel._poisson_at_least(pd.Series([1.0]), pd.Series([0.0])).iloc[0]) == 0.0


def test_a_distant_threshold_stays_in_bounds():
    """Fifteen cards at a low rate underflows; it must not go negative."""
    v = float(X.XPModel._poisson_at_least(pd.Series([15.0]), pd.Series([0.2])).iloc[0])
    assert 0.0 <= v < 1e-10


def test_a_spent_threshold_is_clipped_not_inverted():
    v = float(X.XPModel._poisson_at_least(pd.Series([-3.0]), pd.Series([0.2])).iloc[0])
    assert 0.0 <= v <= 1.0


def test_cumulative_risk_is_monotone_in_the_horizon():
    vals = [float(X.XPModel._poisson_at_least(pd.Series([1.0]), pd.Series([0.15 * h])).iloc[0])
            for h in range(1, 6)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals


def test_an_empty_frame_is_handled():
    assert len(X.XPModel._poisson_at_least(pd.Series(dtype=float), pd.Series(dtype=float))) == 0
