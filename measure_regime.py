"""
Is a team's start rate worth reweighting when its selection regime shifts?

The standing lead was "a manager change resets a start rate": after a new
manager arrives, the previous twenty gameweeks of team selection are much
weaker evidence, and the start rate currently shrinks on a gameweek scale that
knows nothing about it.

That claim cannot be tested directly against the data on disk. FPL carried
managers as assets only in 2024-25, and only from GW23, so exactly one manager
change is observable in a decade of history. Testing against a list written
from memory would measure the list.

So this tests the channel the fix would have to act through, which needs no
external list. Production blends a recent-form start rate into the season-long
one at a fixed weight:

    p = w * recent + (1 - w) * season        RECENT_MINUTES_MAX_WEIGHT = 0.70

A manager change - or any regime shift - can only help by making w larger for
that team. So: does the Brier-minimising w rise with how much a team's recent
selection has diverged from its season-long pattern?

The circularity trap this is built to avoid: churn is DEFINED as the gap
between the two predictors, so comparing the two head to head must show the
recent one winning more as churn rises, whether or not anything real is
happening. Only the blend weight is a fair question.

CONTROL: the same scan against a shuffled churn label. A gradient that also
appears under shuffling is an artefact of binning, not an effect.

Run: python measure_regime.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import priors

HISTORY_DIR = Path(__file__).parent / "data" / "history"

# Seasons carrying a `starts` column, which is what "did he start" should read.
# Earlier seasons force a minutes>=60 proxy that conflates a start with a long
# substitute appearance.
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

RECENT_WINDOW = 5     # gameweeks in the "recent" view, matching the model's
MIN_GW = 10           # season-to-date evidence needed before asking
MIN_SQUAD = 8         # players needed either side to compute a churn figure
PRODUCTION_W = 0.70   # xp_model.RECENT_MINUTES_MAX_WEIGHT


def load_season(season: str) -> pd.DataFrame:
    d = priors.read_season_csv(HISTORY_DIR / season / "merged_gw.csv")
    # 2024-25 carries manager/assistant-manager rows; they are not footballers
    # and have no start rate to model.
    if "position" in d.columns:
        d = d[d["position"] != "AM"]
    # A double gameweek gives a player two rows; he started the gameweek if he
    # started either fixture.
    g = d.groupby(["element", "GW"], as_index=False).agg(
        starts=("starts", "sum"), team=("team", "first"))
    g["started"] = (g["starts"] > 0).astype(float)
    g["season"] = season
    return g


def build_rows(seasons: list[str]) -> pd.DataFrame:
    """One row per player-gameweek: the two competing predictors and the truth."""
    g = pd.concat([load_season(s) for s in seasons], ignore_index=True)
    out = []
    for (season, team), tg in g.groupby(["season", "team"]):
        for target in sorted(tg["GW"].unique()):
            if target < MIN_GW:
                continue
            hist = tg[tg["GW"] < target]
            recent = tg[(tg["GW"] < target) & (tg["GW"] >= target - RECENT_WINDOW)]
            if hist["GW"].nunique() < MIN_GW - 1 or recent.empty:
                continue
            season_rate = hist.groupby("element")["started"].mean()
            recent_rate = recent.groupby("element")["started"].mean()

            shared = season_rate.index.intersection(recent_rate.index)
            pool = shared[(season_rate[shared] > 0.1) | (recent_rate[shared] > 0.1)]
            if len(pool) < MIN_SQUAD:
                continue
            churn = float(np.abs(season_rate[pool] - recent_rate[pool]).mean())

            now = tg[tg["GW"] == target]
            truth = now.set_index("element")["started"]
            idx = truth.index.intersection(season_rate.index)
            if len(idx) < MIN_SQUAD:
                continue
            out.append(pd.DataFrame({
                "season": season, "team": team, "gw": target, "churn": churn,
                # clipped away from 0/1: a player who has never started is not
                # a certainty, and an unclipped Brier rewards overconfidence
                "p_season": season_rate[idx].clip(0.02, 0.98).to_numpy(),
                "p_recent": recent_rate.reindex(idx).fillna(season_rate[idx])
                                       .clip(0.02, 0.98).to_numpy(),
                "started": truth[idx].to_numpy(),
            }))
    return pd.concat(out, ignore_index=True)


GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)


def scan(sub: pd.DataFrame) -> tuple[float, float, float]:
    """Brier-minimising blend weight, its score, and the score at production's."""
    season = sub["p_season"].to_numpy()
    recent = sub["p_recent"].to_numpy()
    y = sub["started"].to_numpy()
    scores = np.array([(((w * recent + (1 - w) * season) - y) ** 2).mean() for w in GRID])
    i = int(scores.argmin())
    at_prod = float(scores[int(np.abs(GRID - PRODUCTION_W).argmin())])
    return float(GRID[i]), float(scores[i]), at_prod


def report(rows: pd.DataFrame, band_col: str, label: str) -> None:
    print(f"\n{label}")
    print(f"{'band':<16}{'rows':>9}{'churn':>9}{'w*':>7}{'Brier@w*':>11}"
          f"{'Brier@0.70':>12}{'gain':>9}")
    print("-" * 74)
    for band, sub in rows.groupby(band_col, observed=True):
        w, best, at_prod = scan(sub)
        print(f"{str(band):<16}{len(sub):>9}{sub['churn'].mean():>9.3f}{w:>7.2f}"
              f"{best:>11.4f}{at_prod:>12.4f}{at_prod - best:>9.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="control shuffle seed")
    args = ap.parse_args()

    rows = build_rows(SEASONS)
    teams = rows.groupby(["season", "team", "gw"]).ngroups
    print(f"{len(rows):,} player-gameweeks across {teams:,} team-gameweeks, "
          f"{len(SEASONS)} seasons")

    labels = ["Q1 stable", "Q2", "Q3", "Q4", "Q5 churning"]
    rows["band"] = pd.qcut(rows["churn"], 5, labels=labels)
    report(rows, "band", "BY REAL CHURN")

    rng = np.random.default_rng(args.seed)
    keys = rows[["season", "team", "gw", "churn"]].drop_duplicates()
    keys["shuffled"] = rng.permutation(keys["churn"].to_numpy())
    rows = rows.merge(keys[["season", "team", "gw", "shuffled"]],
                      on=["season", "team", "gw"], how="left")
    rows["band_shuffled"] = pd.qcut(rows["shuffled"], 5, labels=labels)
    report(rows, "band_shuffled", "CONTROL - SHUFFLED CHURN (must be flat)")

    print("\nIs w even identified? Brier across the blend grid:\n")
    coarse = np.round(np.arange(0.0, 1.001, 0.1), 1)
    print(f"{'band':<15}" + "".join(f"{w:>8.1f}" for w in coarse))
    print("-" * (15 + 8 * len(coarse)))
    for band, sub in rows.groupby("band", observed=True):
        season = sub["p_season"].to_numpy()
        recent = sub["p_recent"].to_numpy()
        y = sub["started"].to_numpy()
        line = [(((w * recent + (1 - w) * season) - y) ** 2).mean() for w in coarse]
        print(f"{str(band):<15}" + "".join(f"{s:>8.4f}" for s in line))

    w, best, at_prod = scan(rows)
    print(f"\npooled: w* = {w:.2f} (Brier {best:.4f}) against production's "
          f"{PRODUCTION_W:.2f} (Brier {at_prod:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
