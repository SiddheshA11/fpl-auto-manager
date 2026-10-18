"""
augment_history rebuilds `team`, `position`, teams.csv and fixtures.csv for
the seasons that lack them. Checked against the seasons that HAVE them:
run the same reconstruction, ignore the real columns, require an exact
match. Measured on 2020-21 to 2025-26: 100% of rows and fixtures.
"""
from __future__ import annotations

import pandas as pd
import pytest

import augment_history as A
import priors

CHECKED = [s for s in ("2024-25", "2025-26") if (priors.HISTORY_DIR / s / "teams.csv").exists()]


@pytest.fixture(scope="module", params=CHECKED or [pytest.param(None, marks=pytest.mark.skip)])
def season(request):
    d = priors.HISTORY_DIR / request.param
    return {k: priors.read_season_csv(d / f"{k}.csv")
            for k in ("merged_gw", "players_raw", "teams", "fixtures")}


def test_every_row_gets_the_club_fpl_recorded(season):
    gw, teams = season["merged_gw"], season["teams"]
    rebuilt = A.team_ids(gw).map(teams.set_index("id")["name"])
    assert (rebuilt == gw["team"]).all()


def test_positions_match_where_fpl_recorded_one(season):
    gw, raw = season["merged_gw"], season["players_raw"]
    squad = gw["position"].isin(["GK", "GKP", "DEF", "MID", "FWD"])
    rebuilt = gw["element"].map(raw.set_index("id")["element_type"]).map(A.POSITION_LABEL)
    assert (rebuilt[squad] == gw.loc[squad, "position"].replace({"GKP": "GK"})).all()


def test_rebuilt_teams_carry_the_real_codes(season):
    built = A.build_teams(season["players_raw"], A.club_names()).set_index("id")
    real = season["teams"].set_index("id")
    assert (built["code"] == real.loc[built.index, "code"]).all()
    assert len(built) == 20


def test_rebuilt_fixtures_match_every_fixture(season):
    built = A.build_fixtures(season["merged_gw"]).set_index("id")
    real = season["fixtures"].set_index("id")
    assert len(built) == len(real) == 380
    for col in ("team_h", "team_a", "event"):
        assert (built[col].astype(int) == real.loc[built.index, col].astype(int)).all(), col


def test_a_restart_season_is_renumbered_onto_38():
    gw = pd.DataFrame({"GW": [1, 29, 39, 47], "element": [1, 1, 1, 1]})
    fx = pd.DataFrame({"event": [1, 29, 39, 47]})
    gw = pd.concat([gw, pd.DataFrame({"GW": list(range(2, 29)) + list(range(40, 47)),
                                      "element": 1})], ignore_index=True)
    out_gw, out_fx = A.renumber_restart(gw, fx)
    assert sorted(out_gw["GW"].unique()) == list(range(1, 39))
    assert list(out_fx["event"]) == [1, 29, 30, 38]


def test_a_normal_season_is_left_alone():
    gw = pd.DataFrame({"GW": list(range(1, 39)), "element": 1})
    fx = pd.DataFrame({"event": list(range(1, 39))})
    out_gw, out_fx = A.renumber_restart(gw, fx)
    assert out_gw is gw and out_fx is fx
