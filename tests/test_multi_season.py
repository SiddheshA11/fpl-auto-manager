"""
Reading a decade of history rather than three seasons.

vaastav's dataset thins out going backwards, and every one of these was a hard
failure that stopped the harness dead rather than degrading it:

  2016-17 .. 2018-19   merged_gw.csv is latin-1, so pandas raises on the first
                       accented name
  2016-17 .. 2018-19   no teams.csv, and 2016-17/2017-18 no fixtures.csv
  2019-20              every file present, but no `team` column, which
                       build_player_dc_team reads directly
  2019-20 .. 2021-22   no expected_goals / expected_assists / starts
  2022-23 .. 2024-25   no defensive_contribution

`available_seasons` tested only that merged_gw.csv existed, so all of it was
let through and "fetch more history" became a KeyError deep in the prior build.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
import priors


# ─────────────────────────── encoding ───────────────────────────


def test_read_season_csv_falls_back_to_latin1(tmp_path):
    p = tmp_path / "merged_gw.csv"
    p.write_bytes("name,GW\nJos\xe9 Fonte,1\n".encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):
        pd.read_csv(p)
    df = priors.read_season_csv(p)
    assert len(df) == 1 and df["GW"].iloc[0] == 1


def test_read_season_csv_leaves_utf8_alone(tmp_path):
    p = tmp_path / "merged_gw.csv"
    p.write_text("name,GW\nJosé Fonte,1\n", encoding="utf-8")
    assert priors.read_season_csv(p)["name"].iloc[0] == "José Fonte"


# ──────────────────────── which seasons are usable ────────────────────────


def _season(dirpath, *, files=priors.SEASON_REQUIRED_FILES, columns=("team", "element", "GW", "total_points", "minutes")):
    dirpath.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f == "merged_gw.csv":
            pd.DataFrame({c: [1] for c in columns}).to_csv(dirpath / f, index=False)
        else:
            pd.DataFrame({"id": [1]}).to_csv(dirpath / f, index=False)
    return dirpath


def test_a_season_missing_a_file_is_not_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(priors, "HISTORY_DIR", tmp_path)
    _season(tmp_path / "2020-21")
    _season(tmp_path / "2018-19", files=[f for f in priors.SEASON_REQUIRED_FILES if f != "teams.csv"])
    assert priors.season_is_usable("2020-21")
    assert not priors.season_is_usable("2018-19"), "a season with no teams.csv cannot build priors"
    assert priors.available_seasons() == ["2020-21"]


def test_a_season_missing_the_team_column_is_not_usable(tmp_path, monkeypatch):
    """
    2019-20 has all four files and still breaks the prior build. A file-only
    check passes it, which is exactly how this got through the first time.
    """
    monkeypatch.setattr(priors, "HISTORY_DIR", tmp_path)
    _season(tmp_path / "2019-20", columns=("element", "GW", "total_points", "minutes"))
    assert not priors.season_is_usable("2019-20")
    assert priors.available_seasons() == []


def test_the_real_history_on_disk_is_all_usable():
    """Whatever is actually fetched must be usable, or the run is built on sand."""
    seasons = priors.available_seasons()
    if not seasons:
        pytest.skip("no history on disk")
    for s in seasons:
        assert priors.season_is_usable(s), f"{s} was reported available but is not usable"


# ─────────────────── a stat a season never recorded ───────────────────


def _gw_frame(columns: list[str]) -> pd.DataFrame:
    base = {"GW": [1, 1, 2, 2], "element": [1, 2, 1, 2], "value": [50, 60, 50, 60]}
    for c in columns:
        base[c] = [1.0, 2.0, 3.0, 4.0]
    return pd.DataFrame(base)


def test_a_stat_absent_from_a_season_is_nan_not_zero():
    """
    0.0 expected goals says "this player never threatens". A season that did
    not measure it says nothing at all, and the two must not be confused - it
    is the rule priors.py already follows.
    """
    present = [c for c in backtest.CUMULATIVE if c != "expected_goals"]
    gw = _gw_frame(present)
    raw = pd.DataFrame({"id": [1, 2], "code": [11, 22], "web_name": ["a", "b"],
                        "element_type": [3, 4], "team": [1, 2], "now_cost": [50, 60]})
    teams = pd.DataFrame({"id": [1, 2], "code": [1, 2], "short_name": ["A", "B"], "name": ["A", "B"]})

    # Only GW1 is in the past at upto_gw=2, where element 1 logged 1.0 and
    # element 2 logged 2.0.
    state = backtest.build_state(gw, raw, teams, 2, {})
    by_id = {e["id"]: e for e in state["elements"]}
    for el in state["elements"]:
        assert np.isnan(el["expected_goals"]), "an unmeasured stat was reported as zero"
    assert by_id[1]["minutes"] == 1.0, "a measured stat must still accumulate"
    assert by_id[2]["minutes"] == 2.0


def test_a_player_with_no_rows_yet_has_zero_not_nan():
    """A player who has not featured has genuinely accumulated nothing."""
    gw = _gw_frame(list(backtest.CUMULATIVE))
    raw = pd.DataFrame({"id": [1, 2, 99], "code": [11, 22, 99], "web_name": ["a", "b", "new"],
                        "element_type": [3, 4, 3], "team": [1, 2, 1], "now_cost": [50, 60, 45]})
    teams = pd.DataFrame({"id": [1, 2], "code": [1, 2], "short_name": ["A", "B"], "name": ["A", "B"]})
    state = backtest.build_state(gw, raw, teams, 2, {})
    newcomer = next(e for e in state["elements"] if e["id"] == 99)
    assert newcomer["minutes"] == 0.0
    assert newcomer["total_points"] == 0.0


@pytest.mark.parametrize("season", ["2022-23", "2023-24", "2024-25", "2025-26"])
def test_build_state_runs_on_every_fetched_season(season):
    """
    Before this, build_state raised KeyError on every season but the newest -
    so backtest.py had only ever been run on one, and there was no way to
    measure anything across seasons.
    """
    d = backtest.HISTORY_DIR / season
    if not d.exists():
        pytest.skip(f"no {season} on disk")
    gw = priors.read_season_csv(d / "merged_gw.csv")
    raw = priors.read_season_csv(d / "players_raw.csv")
    teams = priors.read_season_csv(d / "teams.csv")
    state = backtest.build_state(gw, raw, teams, 10, {})
    assert len(state["elements"]) > 300
    assert all(e["now_cost"] > 0 for e in state["elements"])


def test_the_in_progress_season_is_excluded_exactly_as_before(tmp_path, monkeypatch):
    """
    CI fetches the current season alongside the two before it, and before a
    ball is kicked vaastav has nothing for it - so `fetch_data` creates the
    directory and every file 404s. The old check excluded it because
    merged_gw.csv was absent; the new one must exclude it for the same reason
    and no other, or tightening this quietly changes which seasons the weekly
    run builds its priors from.
    """
    monkeypatch.setattr(priors, "HISTORY_DIR", tmp_path)
    _season(tmp_path / "2024-25")
    _season(tmp_path / "2025-26")
    (tmp_path / "2026-27").mkdir()                     # created, never populated
    assert priors.available_seasons() == ["2024-25", "2025-26"]

    # And a *partial* in-progress season, which the old check would have let
    # through: files present but merged_gw.csv missing the column the prior
    # build reads. Excluding it is the safe direction - including it raised
    # KeyError mid-run.
    _season(tmp_path / "2026-27", columns=("element", "GW", "total_points", "minutes"))
    assert priors.available_seasons() == ["2024-25", "2025-26"]
