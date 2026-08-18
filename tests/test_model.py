"""
Tests for the priors, expected-points model and optimiser.

Weighted toward regression tests for bugs found during the rebuild. Every one
of them produced plausible-looking output rather than an error - a backup
keeper rated as a starter, a defensive rate silently halved - which is the
class of bug that survives eyeballing and needs pinning down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optimizer as O
import priors
import xp_model as X


# ──────────────────────── fixtures ────────────────────────


def _gw_rows(code: int, position: str, *, gws: int, starts: int, minutes_each: int, **stats) -> pd.DataFrame:
    """Synthesise a season of per-gameweek rows for one player."""
    rows = []
    for gw in range(1, gws + 1):
        started = gw <= starts
        row = {
            "element": code, "code": code, "GW": gw, "position": position,
            "minutes": minutes_each if started else 0,
            "starts": 1 if started else 0,
            "total_points": 2 if started else 0,
        }
        for k, v in stats.items():
            row[k] = v if started else 0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_season() -> pd.DataFrame:
    """A nailed starter and a never-used backup, both registered all season."""
    starter = _gw_rows(1, "GK", gws=38, starts=38, minutes_each=90,
                       saves=3, bps=20, expected_goals=0.0, expected_assists=0.0,
                       defensive_contribution=0, goals_conceded=1, yellow_cards=0)
    backup = _gw_rows(2, "GK", gws=38, starts=0, minutes_each=0,
                      saves=0, bps=0, expected_goals=0.0, expected_assists=0.0,
                      defensive_contribution=0, goals_conceded=0, yellow_cards=0)
    return pd.concat([starter, backup], ignore_index=True)


def _player_pool(n_per_pos: int = 8) -> pd.DataFrame:
    """A legal, solvable player pool spread across enough clubs."""
    rows = []
    pid = 1
    for pos, count in [(1, n_per_pos), (2, n_per_pos * 3), (3, n_per_pos * 3), (4, n_per_pos * 2)]:
        for i in range(count):
            rows.append({
                "id": pid,
                "web_name": f"p{pid}",
                "position": pos,
                "team": (pid % 12) + 1,
                "cost": 4.0 + (i % 8) * 0.5,
                "xp_horizon": 1.0 + (i % 8) * 0.7,
                "xp_next": 0.5 + (i % 8) * 0.3,
            })
            pid += 1
    return pd.DataFrame(rows)


# ──────────────────────── priors ────────────────────────


def test_start_rate_uses_gameweeks_not_appearances(synthetic_season):
    """
    Regression: start_rate divided by appearances, which conditions on having
    played. A keeper with 0 starts in 38 registered gameweeks came out as a
    nailed starter because his only "appearances" were none at all.
    """
    agg = priors._aggregate_player_season(synthetic_season)
    per90 = priors._to_per90(agg)

    assert per90.loc[1, "start_rate"] == pytest.approx(1.0)
    assert per90.loc[2, "start_rate"] == pytest.approx(0.0)
    assert per90.loc[2, "squad_gameweeks"] == 38


def test_missing_stat_column_is_nan_not_zero():
    """
    Regression: a stat absent from a season (defensive_contribution predates
    2025-26) was defaulted to 0.0 and averaged in, halving every player's rate.
    A season that predates a stat carries no information about it.
    """
    season = _gw_rows(1, "DEF", gws=10, starts=10, minutes_each=90,
                      expected_goals=0.1, expected_assists=0.1, saves=0,
                      bps=20, goals_conceded=1, yellow_cards=0)
    season = season.drop(columns=["defensive_contribution"], errors="ignore")

    agg = priors._aggregate_player_season(season)
    assert np.isnan(agg.loc[1, "dc"]), "absent stat must aggregate to NaN, not 0.0"


def test_positional_start_rate_prior_is_not_conditioned_on_playing(synthetic_season):
    """
    The positional prior for start rate must be taken over every registered
    player. Restricting it to those with minutes drove the goalkeeper prior to
    ~1.0 and rated every backup as a starter.
    """
    per90 = priors._to_per90(priors._aggregate_player_season(synthetic_season))
    pos = priors._positional_means(per90)
    # One starter, one backup, so the marginal rate is well under 1.
    assert 0.0 < pos.loc[1, "start_rate"] < 0.75


def test_promoted_teams_get_explicit_prior():
    """A club with no history must not silently default to league average."""
    teams, _ = priors.build_team_strength(seasons=[], current_team_codes={999: "Newly Promoted"})
    t = teams[999]
    assert t.is_promoted
    assert t.attack < 1.0 and t.defence > 1.0


# ──────────────────────── expected points ────────────────────────


def _model(monkeypatch_players: pd.DataFrame | None = None) -> X.XPModel:
    bootstrap = {
        "game_config": {"scoring": {
            "long_play": 2, "short_play": 1, "saves": 1, "bonus": 1, "assists": 3,
            "yellow_cards": -1,
            "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
            "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
            "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
            "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
        }},
        "teams": [{"id": 1, "code": 11, "short_name": "AAA"}, {"id": 2, "code": 22, "short_name": "BBB"}],
        "elements": [],
        "events": [],
    }
    ps = priors.PriorSet(players=pd.DataFrame(), teams={}, league_mean_goals=1.4)
    m = X.XPModel.__new__(X.XPModel)
    m.bootstrap = bootstrap
    m.fixtures = pd.DataFrame()
    m.priors = ps
    m.config = X.ModelConfig()
    m.scoring = bootstrap["game_config"]["scoring"]
    m.teams = pd.DataFrame(bootstrap["teams"])
    m._team_id_to_code = {1: 11, 2: 22}
    m._team_id_to_name = {1: "AAA", 2: "BBB"}
    m.players = monkeypatch_players if monkeypatch_players is not None else pd.DataFrame()
    return m


def test_defensive_contribution_is_a_threshold_not_a_rate():
    """
    The award fires at 10 (DEF) / 12 (MID, FWD), so P(award) must come from the
    tail of a count distribution. A linear term on the rate would make the
    award probability proportional to the rate, which it is not: it is convex
    below the threshold and saturating above it.
    """
    m = _model(pd.DataFrame({"position": [2, 2, 2]}))
    dc90 = pd.Series([4.0, 8.0, 16.0])
    nailed = pd.Series([1.0, 1.0, 1.0])
    no_sub = pd.Series([0.0, 0.0, 0.0])
    p = m._p_dc_award(dc90, pd.Series([2, 2, 2]), nailed, no_sub)

    assert (p.diff().dropna() > 0).all(), "must increase with rate"
    assert p.iloc[0] < 0.10, "well below threshold should rarely fire"
    assert p.iloc[2] > 0.85, "well above threshold should usually fire"
    # Convexity: doubling the rate from 4 to 8 must more than double P.
    assert p.iloc[1] > 2 * p.iloc[0]


def test_goalkeepers_never_earn_defensive_contribution():
    m = _model(pd.DataFrame({"position": [1]}))
    p = m._p_dc_award(pd.Series([20.0]), pd.Series([1]), pd.Series([1.0]), pd.Series([0.0]))
    assert p.iloc[0] == 0.0


def test_defensive_contribution_uses_the_minutes_mixture_not_mean_minutes():
    """
    Regression: the tail was evaluated once at average minutes. It is convex in
    the rate, so by Jensen that sits far below the mean of the probabilities -
    a player starting half the time was priced at roughly a fifth of his true
    chance. Every previous test used a full 90 minutes, which is precisely the
    case where the bug disappears.
    """
    m = _model(pd.DataFrame({"position": [2]}))
    rate, thr = pd.Series([11.0]), 10

    rotated = m._p_dc_award(rate, pd.Series([2]), pd.Series([0.5]), pd.Series([0.0]))
    nailed = m._p_dc_award(rate, pd.Series([2]), pd.Series([1.0]), pd.Series([0.0]))

    # Half the starts must be worth half the nailed probability, not a fifth.
    assert rotated.iloc[0] == pytest.approx(0.5 * nailed.iloc[0], rel=1e-6)

    # And well above what evaluating the tail at mean minutes would have given.
    at_mean_minutes = m._tail_probability(np.array([11.0 * (0.5 * 82.0) / 90.0]), thr)[0]
    assert rotated.iloc[0] > 2 * at_mean_minutes


def test_start_probability_normalised_to_eleven_shirts():
    """
    Regression: start probability was estimated per player in isolation, so a
    club could field two keepers each 68% likely to start. Each team has
    exactly one goalkeeping shirt and ten outfield ones.
    """
    players = pd.DataFrame({
        "team": [1, 1, 1, 1] + [1] * 12,
        "position": [1, 1, 1, 1] + [3] * 12,
    })
    m = _model(players)
    raw = pd.Series([0.9, 0.6, 0.5, 0.4] + [0.8] * 12)
    out = m._normalise_starts_within_team(raw)

    gk = out[players["position"] == 1]
    outfield = out[players["position"] != 1]
    assert gk.sum() == pytest.approx(1.0, abs=1e-6)
    assert outfield.sum() == pytest.approx(10.0, abs=1e-6)
    assert (out <= 1.0).all()


def test_clear_first_choice_keeper_is_not_diluted_by_backups():
    """
    Regression: proportional allocation punished a settled first choice for
    having credible cover, dividing an 0.87 keeper down to 0.65 because his
    club carried two plausible deputies.
    """
    players = pd.DataFrame({"team": [1, 1, 1], "position": [1, 1, 1]})
    m = _model(players)
    out = m._normalise_starts_within_team(pd.Series([0.87, 0.32, 0.15]))
    assert out.iloc[0] > 0.85, "clear first choice should stay near-certain"
    assert out.iloc[1] < 0.12


def test_injured_first_choice_promotes_the_understudy():
    """Availability of zero must hand the shirt to the backup, not vanish."""
    players = pd.DataFrame({"team": [1, 1], "position": [1, 1]})
    m = _model(players)
    out = m._normalise_starts_within_team(pd.Series([0.0, 0.20]))
    assert out.iloc[0] == 0.0
    assert out.iloc[1] > 0.9


def test_bonus_curve_is_monotone_and_convex():
    """A linear BPS coefficient underprices premiums; the curve must bend up."""
    m = _model(pd.DataFrame({"position": [3, 3, 3]}))
    bonus = m._expected_bonus(pd.Series([8.0, 16.0, 24.0]), pd.Series([1.0, 1.0, 1.0]))
    assert bonus.is_monotonic_increasing
    assert (bonus.iloc[2] - bonus.iloc[1]) > (bonus.iloc[1] - bonus.iloc[0])


# ──────────────────────── optimiser ────────────────────────


def test_built_squad_is_legal():
    sol = O.SquadOptimizer(_player_pool(), "xp_horizon", "xp_next").build_squad(100.0)

    assert len(sol.squad) == O.SQUAD_SIZE
    assert len(sol.xi) == O.XI_SIZE
    assert len(sol.bench) == O.SQUAD_SIZE - O.XI_SIZE
    assert sol.total_cost <= 100.0 + 1e-6
    assert sol.squad["team"].value_counts().max() <= O.MAX_PER_CLUB

    counts = sol.squad["position"].value_counts().to_dict()
    assert counts == O.SQUAD_BY_POSITION

    xi_counts = sol.xi["position"].value_counts().to_dict()
    for pos, (lo, hi) in O.XI_BOUNDS.items():
        assert lo <= xi_counts.get(pos, 0) <= hi


def test_captain_is_a_starter_and_distinct_from_vice():
    sol = O.SquadOptimizer(_player_pool(), "xp_horizon", "xp_next").build_squad(100.0)
    xi_ids = set(sol.xi["id"].astype(int))
    assert sol.captain in xi_ids
    assert sol.vice_captain in xi_ids
    assert sol.captain != sol.vice_captain


def test_greedy_position_fill_would_bust_budget_but_milp_does_not():
    """
    Regression for the old WildcardOptimizer: filling GK -> DEF -> MID -> FWD
    greedily by score spent the budget before reaching the forwards and
    returned a squad that was short, over budget, or both.
    """
    sol = O.SquadOptimizer(_player_pool(), "xp_horizon", "xp_next").build_squad(100.0)
    assert len(sol.squad) == 15 and sol.total_cost <= 100.0 + 1e-6
    assert (sol.squad["position"] == 4).sum() == 3, "forwards must not be starved by earlier positions"


def test_transfers_respect_the_hit_cap():
    pool = _player_pool()
    opt = O.SquadOptimizer(pool, "xp_horizon", "xp_next")
    base = opt.build_squad(100.0)
    ids = [int(i) for i in base.squad["id"]]

    sol = opt.optimise_transfers(ids, bank=0.0, free_transfers=1, max_hits=0)
    assert len(sol.transfers_in) <= 1
    assert sol.hits == 0
    assert len(sol.squad) == 15


def test_transfer_keeps_squad_legal_under_budget_pressure():
    pool = _player_pool()
    opt = O.SquadOptimizer(pool, "xp_horizon", "xp_next")
    base = opt.build_squad(100.0)
    ids = [int(i) for i in base.squad["id"]]

    sol = opt.optimise_transfers(ids, bank=0.0, free_transfers=2, max_hits=2)
    assert sol.squad["team"].value_counts().max() <= O.MAX_PER_CLUB
    assert sol.squad["position"].value_counts().to_dict() == O.SQUAD_BY_POSITION


def test_zero_gain_transfer_is_declined():
    """
    An unused free transfer rolls over, so a move that gains nothing is a real
    loss. Without an option value on the free transfer the solver was
    indifferent and churned the squad for no reason.
    """
    pool = _player_pool()
    opt = O.SquadOptimizer(pool, "xp_horizon", "xp_next")
    base = opt.build_squad(100.0)
    ids = [int(i) for i in base.squad["id"]]

    # Same pool, same budget: there is nothing to gain.
    sol = opt.optimise_transfers(ids, bank=0.0, free_transfers=1, max_hits=0)
    assert len(sol.transfers_in) == 0, "should stand pat when no transfer improves the squad"


def test_every_team_starts_exactly_eleven_players():
    """
    Eleven shirts are worn in every match, so the start probabilities in a team
    must sum to eleven. The allocation clipped each player at 0.98 *after*
    scaling, which discarded the mass above the cap instead of giving it to
    somebody else: teams were allocated 10.75 shirts and 937 of the 990 minutes
    a side actually plays. A shortfall in minutes scales every per-90 term in
    the model, so it surfaced as a flat negative bias in expected points.
    """
    from pathlib import Path

    import priors
    import xp_model as X

    snaps = Path(__file__).resolve().parent.parent / "data" / "snapshots"
    bs_files = sorted(snaps.glob("bootstrap-static_*.json.gz"), reverse=True)
    if not bs_files:
        pytest.skip("no snapshot committed")
    bootstrap = X.load_snapshot(bs_files[0])
    fixtures = X.load_snapshot(sorted(snaps.glob("fixtures_*.json.gz"), reverse=True)[0])
    model = X.XPModel(bootstrap, fixtures, priors.build_priors(), X.ModelConfig(horizon=5))

    mm = model.minutes_model(X.next_events(bootstrap, 5)[0])
    per_team = pd.DataFrame({"team": model.players["team"], "p": mm["p_start"]}).groupby("team")["p"].sum()
    assert per_team.min() > 10.9, f"a team was allocated only {per_team.min():.2f} shirts"
    assert per_team.max() < 11.1
    assert per_team.mean() == pytest.approx(11.0, abs=0.05)


def test_shirt_allocation_redistributes_rather_than_discarding():
    """The unit behind the above: surplus above the cap goes to someone else."""
    import xp_model as X

    # One overwhelming favourite plus three fringe players, for one shirt.
    out = X._allocate_shirts(pd.Series([20.0, 1.0, 1.0, 1.0]), shirts=1.0, cap=0.98)
    assert out.max() <= 0.98 + 1e-9, "the cap must still bind"
    assert out.sum() == pytest.approx(1.0, abs=1e-6), "the shirt must go to somebody"

    # An unavailable player has zero weight and must stay at zero. The first
    # version redistributed to anyone under the cap and duly resurrected
    # injured players, who began scoring points while ruled out.
    with_injury = X._allocate_shirts(pd.Series([5.0, 5.0, 0.0]), shirts=2.0, cap=0.98)
    assert with_injury.iloc[2] == 0.0, "a player with no chance of playing was given one"

    # A squad thinner than the shirts available cannot fill them; terminate.
    capped = X._allocate_shirts(pd.Series([1.0, 1.0]), shirts=10.0, cap=0.98)
    assert capped.sum() == pytest.approx(1.96, abs=1e-6)
