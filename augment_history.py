"""
Make 2016-17 to 2019-20 usable: rebuild the columns their files lack.

The decision-replay harness is limited by its number of seasons - inference
is season-clustered, and five clusters give t4 and an MDE of ~85-90 points
a season. Four more seasons are on disk and unusable only because
merged_gw.csv has no `team` column before 2020-21 (and 2016-18 also lack
teams.csv and fixtures.csv). Everything needed to rebuild them is there:

  team      every row carries its fixture id, was_home and opponent_team, so
            a player's club in a gameweek is the opponent_team of the rows on
            the other side of the same fixture. Exact, and right for a player
            who moved mid-season, which a season-level lookup would not be.
  position  players_raw's element_type.
  teams     players_raw pairs each season's team id with the stable club
            `code`; names come from any season's teams.csv carrying that code.
  fixtures  one row per fixture id from merged_gw: event, kickoff, the two
            clubs and the score.

2019-20's restart gameweeks are numbered 39-47 in both files, with 30-38
empty, and the pipeline assumes 38 throughout; they are renumbered 30-38.

Validated by running the same reconstruction on the six seasons that DO
carry these columns, ignoring them, and requiring an exact match
(tests/test_augment_history.py). Originals are kept as *.orig.csv and are
what a re-run reads, so this is idempotent.

    python3 augment_history.py            # every season that needs it
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import priors

logger = logging.getLogger("fpl_auto.augment")

POSITION_LABEL = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
LAST_GW = 38


def fixture_sides(gw_df: pd.DataFrame) -> pd.DataFrame:
    """
    fixture -> (team_h, team_a). The home side is the opponent of anyone who
    played away in it, and the reverse.
    """
    home = gw_df[~gw_df["was_home"].astype(bool)].groupby("fixture")["opponent_team"].agg(pd.Series.mode)
    away = gw_df[gw_df["was_home"].astype(bool)].groupby("fixture")["opponent_team"].agg(pd.Series.mode)
    sides = pd.DataFrame({"team_h": home, "team_a": away}).dropna()
    return sides.astype(int)


def team_ids(gw_df: pd.DataFrame) -> pd.Series:
    """The club id each row played for."""
    sides = fixture_sides(gw_df)
    h = gw_df["fixture"].map(sides["team_h"])
    a = gw_df["fixture"].map(sides["team_a"])
    return h.where(gw_df["was_home"].astype(bool), a)


def build_teams(raw_df: pd.DataFrame, names: dict[int, tuple[str, str]]) -> pd.DataFrame:
    pairs = raw_df[["team", "team_code"]].drop_duplicates().astype(int).sort_values("team")
    if pairs["team"].duplicated().any():
        raise ValueError("players_raw maps one team id to several codes")
    return pd.DataFrame({
        "id": pairs["team"].to_numpy(), "code": pairs["team_code"].to_numpy(),
        "name": [names.get(c, (f"Club {c}", f"C{c}"))[0] for c in pairs["team_code"]],
        "short_name": [names.get(c, (f"Club {c}", f"C{c}"))[1] for c in pairs["team_code"]],
    })


def build_fixtures(gw_df: pd.DataFrame) -> pd.DataFrame:
    sides = fixture_sides(gw_df)
    g = gw_df.groupby("fixture")
    out = pd.DataFrame({
        "id": sides.index.astype(int),
        "event": g["GW"].agg(lambda s: int(s.mode().iloc[0])).reindex(sides.index).to_numpy(),
        "kickoff_time": g["kickoff_time"].first().reindex(sides.index).to_numpy(),
        "team_h": sides["team_h"].to_numpy(), "team_a": sides["team_a"].to_numpy(),
        "team_h_score": g["team_h_score"].first().reindex(sides.index).to_numpy(),
        "team_a_score": g["team_a_score"].first().reindex(sides.index).to_numpy(),
        "finished": True,
    })
    return out.sort_values(["event", "id"]).reset_index(drop=True)


def renumber_restart(gw_df: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2019-20: gameweeks 39-47 become 30-38, in both files, by rank of the non-empty ones."""
    used = sorted(set(gw_df["GW"].astype(int)) | set(fixtures["event"].dropna().astype(int)))
    if max(used) <= LAST_GW:
        return gw_df, fixtures
    if len(used) != LAST_GW:
        raise ValueError(f"{len(used)} non-empty gameweeks; cannot renumber onto 1-{LAST_GW}")
    rank = {gw: i + 1 for i, gw in enumerate(used)}
    gw_df = gw_df.assign(GW=gw_df["GW"].astype(int).map(rank))
    if "round" in gw_df.columns:
        gw_df["round"] = gw_df["GW"]
    fixtures = fixtures.assign(event=fixtures["event"].map(lambda e: rank.get(int(e), e)
                                                           if pd.notna(e) else e))
    return gw_df, fixtures


def club_names() -> dict[int, tuple[str, str]]:
    """code -> (name, short_name), from every season that has a teams.csv."""
    out = {}
    for d in sorted(priors.HISTORY_DIR.iterdir()):
        for name in ("teams.orig.csv", "teams.csv"):
            p = d / name
            if p.exists():
                t = priors.read_season_csv(p)
                out.update({int(r["code"]): (r["name"], r["short_name"]) for _, r in t.iterrows()})
                break
    return out


def _read(d, stem: str) -> pd.DataFrame | None:
    for name in (f"{stem}.orig.csv", f"{stem}.csv"):
        if (d / name).exists():
            return priors.read_season_csv(d / name)
    return None


def _write(d, stem: str, frame: pd.DataFrame) -> None:
    orig, canon = d / f"{stem}.orig.csv", d / f"{stem}.csv"
    if canon.exists() and not orig.exists():
        canon.rename(orig)
    frame.to_csv(canon, index=False)


def augment(season: str, names: dict[int, tuple[str, str]]) -> list[str]:
    """Rebuild what `season` is missing. Returns what was rebuilt."""
    d = priors.HISTORY_DIR / season
    gw = _read(d, "merged_gw")
    raw = _read(d, "players_raw")
    teams = _read(d, "teams")
    fixtures = _read(d, "fixtures")
    done = []
    if teams is None:
        teams = build_teams(raw, names)
        done.append("teams")
    if fixtures is None:
        fixtures = build_fixtures(gw)
        done.append("fixtures")
    renumbered_gw, renumbered_fx = renumber_restart(gw, fixtures)
    if not renumbered_gw["GW"].equals(gw["GW"]):
        done.append("restart renumbered")
    gw, fixtures = renumbered_gw, renumbered_fx
    if "team" not in gw.columns:
        id_to_name = teams.set_index("id")["name"]
        gw["team"] = team_ids(gw).map(id_to_name)
        if gw["team"].isna().any():
            raise ValueError(f"{season}: {int(gw['team'].isna().sum())} rows got no team")
        done.append("team")
    if "position" not in gw.columns:
        gw["position"] = gw["element"].map(raw.set_index("id")["element_type"]).map(POSITION_LABEL)
        done.append("position")
    if done:
        _write(d, "merged_gw", gw)
        _write(d, "teams", teams)
        _write(d, "fixtures", fixtures)
    return done


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2016-17,2017-18,2018-19,2019-20")
    args = ap.parse_args()
    names = club_names()
    for season in args.seasons.split(","):
        done = augment(season, names)
        print(f"{season}: {', '.join(done) if done else 'nothing to do'}")
    print("usable now:", ", ".join(priors.available_seasons()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
