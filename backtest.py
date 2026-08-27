"""
Replay a past season to find out whether the model is worth trusting.

For each gameweek it reconstructs the game state as it stood *before* that
gameweek - cumulative stats only up to the previous week - runs the real
XPModel against it, and scores the prediction against what actually happened.

Priors are restricted to seasons strictly before the one under test, so
nothing about the season being predicted leaks into the model that predicts
it.

Baselines it has to beat:

  ppg         points per game so far this season. What a human eyeballing the
              table would do, and the benchmark that matters.
  price       cost alone. The market's opinion, and a surprisingly hard
              baseline early in a season.

FPL's own published xP is deliberately *not* used as a baseline: the column
exists in the dataset but is 83% zeros and constant within most gameweeks, so
correlations against it are meaningless.

Correlation is reported over three populations, because the global figure is
misleading. It is dominated by correctly ordering the hundreds of players who
never appear - a task points-per-game wins trivially, since a player who never
plays averages zero and duly scores zero. Squad decisions only ever concern
the top of the ranking, so that is measured separately.

Two known handicaps, both of which make the results conservative rather than
flattering: historical injury flags are not in the dataset, so the model runs
blind to availability, and team strength does not update within the season.

Run: python backtest.py --season 2025-26
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import priors
import xp_model as X

logger = logging.getLogger("fpl_auto.backtest")

HISTORY_DIR = Path(__file__).parent / "data" / "history"
SNAPSHOT_DIR = Path(__file__).parent / "data" / "snapshots"

POSITION_IDS = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

# Stats accumulated from the start of the season up to the gameweek being
# predicted. These mirror the fields XPModel reads out of bootstrap elements.
CUMULATIVE = [
    "minutes", "starts", "expected_goals", "expected_assists", "saves", "bps",
    "yellow_cards", "goals_conceded", "defensive_contribution", "total_points",
]


def _scoring_config() -> dict:
    """Reuse a real game_config so scoring values are not reinvented here."""
    snap = X.latest_snapshot(SNAPSHOT_DIR, "bootstrap-static")
    if snap is None:
        raise SystemExit("need a bootstrap snapshot for the scoring config; run fetch_data.py")
    return X.load_snapshot(snap)["game_config"]


def build_state(
    gw_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    upto_gw: int,
    game_config: dict,
) -> dict:
    """
    Reconstruct a bootstrap-shaped snapshot as of just before `upto_gw`.

    Only rows from gameweeks strictly earlier than upto_gw are summed, so the
    model sees exactly what it would have seen at the deadline.
    """
    # A stat that a season never recorded is not zero. `defensive_contribution`
    # arrives in 2025-26 and `starts`/`expected_goals`/`expected_assists` in
    # 2022-23, so summing CUMULATIVE blindly raised KeyError on every season but
    # the newest - which is why this harness had only ever been run on one.
    # Absent columns aggregate to NaN, the same rule priors.py follows: a player
    # with 0.0 expected goals looks like a player who never threatens, and that
    # is a different claim from "this season did not measure it".
    present = [c for c in CUMULATIVE if c in gw_df.columns]
    absent = [c for c in CUMULATIVE if c not in gw_df.columns]
    if absent:
        logger.info("season lacks %s; those totals will be NaN", ", ".join(absent))
    past = gw_df[gw_df["GW"] < upto_gw]
    totals = past.groupby("element")[present].sum() if len(past) else pd.DataFrame(columns=present)
    for c in absent:
        totals[c] = np.nan

    # Price as at the gameweek being predicted, not season end. Deduplicated
    # because a double gameweek gives a player two rows with the same price.
    at_gw = gw_df[gw_df["GW"] == upto_gw].drop_duplicates(subset="element").set_index("element")
    price = at_gw["value"] if "value" in at_gw.columns else None

    elements = []
    for _, p in raw_df.iterrows():
        eid = int(p["id"])
        # A player with no rows yet has genuinely accumulated nothing, which is
        # 0.0 - but only for stats the season actually records.
        if eid in totals.index:
            t = totals.loc[eid]
        else:
            t = pd.Series({c: (np.nan if c in absent else 0.0) for c in CUMULATIVE})
        now_cost = float(price.get(eid, p["now_cost"])) if price is not None else float(p["now_cost"])
        elements.append({
            "id": eid,
            "code": int(p["code"]),
            "web_name": p["web_name"],
            "element_type": int(p["element_type"]),
            "team": int(p["team"]),
            "now_cost": now_cost,
            # Historical availability flags are not in the dataset. Treating
            # everyone as fit handicaps the model, which is the safe direction
            # for a validation run.
            "status": "a",
            "chance_of_playing_next_round": None,
            "selected_by_percent": 0.0,
            **{c: float(t.get(c, 0.0)) for c in CUMULATIVE},
        })

    events = [{"id": g, "finished": g < upto_gw} for g in range(1, 39)]
    teams = [
        {"id": int(r["id"]), "code": int(r["code"]), "short_name": r["short_name"], "name": r["name"]}
        for _, r in teams_df.iterrows()
    ]
    return {"elements": elements, "events": events, "teams": teams, "game_config": game_config}


def recent_minutes_for(gw_df: pd.DataFrame, upto_gw: int, depth: int = 5) -> dict[int, list[float]]:
    """
    Minutes each player logged in the `depth` gameweeks before `upto_gw`, most
    recent first - the offline equivalent of `FPLClient.get_recent_minutes`.

    This was missing entirely, and it is not a small omission: `manager.py` is
    the only caller that ever supplied `recent_minutes`, so `backtest.py` and
    `simulate.py` both scored a model with `_blend_recent_minutes` returning at
    its first line. Every offline measurement in the repo therefore evaluated a
    model without the input HANDOFF credits with the largest single improvement
    ever made, and `python backtest.py --season 2025-26` could not reproduce the
    R2 0.319 it quotes.

    Two details have to match the live client or the backtest measures a
    different model again:

    - the live endpoint returns a row for every player, so someone who did not
      feature scores a real 0.0 rather than being absent. A missing row here
      means the same thing and must become 0.0, not be skipped - dropping it
      would shorten that player's window and quietly reweight the lag average.
    - a double gameweek is one entry, summed, because the model's window is in
      gameweeks and not in fixtures.
    """
    if upto_gw <= 1:
        return {}
    window = [gw for gw in range(upto_gw - 1, max(0, upto_gw - 1 - depth), -1)]
    if not window:
        return {}

    played = gw_df[gw_df["GW"].isin(window)]
    # sum within a gameweek (doubles), then a dense element x gameweek grid so
    # a non-appearance is 0.0 rather than a hole
    grid = (played.groupby(["element", "GW"])["minutes"].sum()
                  .unstack("GW")
                  .reindex(columns=window)
                  .fillna(0.0))
    return {int(pid): [float(v) for v in row] for pid, row in grid.iterrows()}


def run_backtest(season: str, start_gw: int, end_gw: int, horizon: int = 1) -> pd.DataFrame:
    season_dir = HISTORY_DIR / season
    gw_df = priors.read_season_csv(season_dir / "merged_gw.csv")
    raw_df = priors.read_season_csv(season_dir / "players_raw.csv")
    teams_df = priors.read_season_csv(season_dir / "teams.csv")
    fixtures = priors.read_season_csv(season_dir / "fixtures.csv").to_dict("records")
    game_config = _scoring_config()

    # Strictly earlier seasons only. Including the season under test would let
    # the model learn the answer it is being asked to predict.
    earlier = [s for s in priors.available_seasons() if s < season]
    if not earlier:
        raise SystemExit(f"no seasons before {season} on disk; cannot build clean priors")
    logger.info("priors from %s (excluding %s to avoid leakage)", earlier, season)

    team_codes = {int(r["code"]): r["name"] for _, r in teams_df.iterrows()}
    prior_set = priors.build_priors(seasons=earlier, current_team_codes=team_codes)

    rows = []
    for gw in range(start_gw, end_gw + 1):
        gw_rows = gw_df[gw_df["GW"] == gw]
        if gw_rows.empty:
            continue
        # A double gameweek gives a player two rows. The model predicts the
        # gameweek total across both fixtures, so the truth must be summed to
        # match, otherwise DGW players are scored against half their return.
        agg = {"total_points": "sum", "minutes": "sum"}
        if "xP" in gw_rows.columns:
            agg["xP"] = "sum"
        actual = gw_rows.groupby("element", as_index=False).agg(agg)

        state = build_state(gw_df, raw_df, teams_df, gw, game_config)
        model = X.XPModel(state, fixtures, prior_set, X.ModelConfig(horizon=horizon),
                          recent_minutes=recent_minutes_for(gw_df, gw))
        pred = model.expected_points([gw])[["id", "xp_next", "cost", "position"]]

        merged = actual.merge(pred, left_on="element", right_on="id", how="inner")
        if merged.empty:
            continue

        # Points so far this season, as a naive baseline.
        past = gw_df[gw_df["GW"] < gw]
        ppg = (
            past.groupby("element")["total_points"].mean()
            if len(past) else pd.Series(dtype=float)
        )
        merged["ppg"] = merged["element"].map(ppg).fillna(0.0)

        played = merged[merged["minutes"] > 0]
        ownable = merged[merged["cost"] >= 5.0]

        record = {
            "gw": gw,
            "n": len(merged),
            "all_model": _rank_corr(merged["xp_next"], merged["total_points"]),
            "all_ppg": _rank_corr(merged["ppg"], merged["total_points"]),
            "all_price": _rank_corr(merged["cost"], merged["total_points"]),
            "played_model": _rank_corr(played["xp_next"], played["total_points"]),
            "played_ppg": _rank_corr(played["ppg"], played["total_points"]),
            "ownable_model": _rank_corr(ownable["xp_next"], ownable["total_points"]),
            "ownable_ppg": _rank_corr(ownable["ppg"], ownable["total_points"]),
            "model_mae": float(np.mean(np.abs(merged["xp_next"] - merged["total_points"]))),
        }
        # Absolute accuracy, on the population comparable published work uses.
        #
        # Rank correlation says whether the ordering is right; it cannot say
        # whether the numbers are. Those numbers are what the optimiser
        # actually consumes - a -4 hit is compared against them directly - so
        # a model can rank well and still price every decision wrong.
        #
        # Published figures for next-gameweek FPL points, from Valouxis (NTUA,
        # 2023) over 6,900 samples of 2022-23 GW26-38, several of them paid
        # products: MAE 1.29-1.42, RMSE 2.27-2.38, R2 0.29-0.35. Population
        # definitions differ, so treat these as a band to sit inside rather
        # than a leaderboard to win.
        for label, frame in (("all", merged), ("played", played), ("ownable", ownable)):
            err = frame["xp_next"] - frame["total_points"]
            ss_res = float((err ** 2).sum())
            ss_tot = float(((frame["total_points"] - frame["total_points"].mean()) ** 2).sum())
            record[f"{label}_mae"] = float(err.abs().mean())
            record[f"{label}_rmse"] = float(np.sqrt((err ** 2).mean()))
            record[f"{label}_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            # Calibration: a model can rank well and still be wrong in level,
            # and the level is what the optimiser prices decisions against.
            record[f"{label}_pred"] = float(frame["xp_next"].mean())
            record[f"{label}_actual"] = float(frame["total_points"].mean())
        for n in (20, 30, 50):
            record[f"model_top{n}"] = _top_n_actual(merged, "xp_next", n)
            record[f"ppg_top{n}"] = _top_n_actual(merged, "ppg", n)
        rows.append(record)

        logger.info("GW%-2d rho(played) model=%.3f ppg=%.3f | top30 model=%.2f ppg=%.2f",
                    gw, record["played_model"], record["played_ppg"],
                    record["model_top30"], record["ppg_top30"])

    return pd.DataFrame(rows)


def _rank_corr(pred: pd.Series, actual: pd.Series) -> float:
    if pred.nunique() < 2 or actual.nunique() < 2:
        return float("nan")
    rho, _ = spearmanr(pred, actual)
    return float(rho)


def _top_n_actual(df: pd.DataFrame, col: str, n: int) -> float:
    """Mean points actually scored by the n players the metric liked most."""
    return float(df.nlargest(n, col)["total_points"].mean())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--start-gw", type=int, default=2)
    ap.add_argument("--end-gw", type=int, default=38)
    args = ap.parse_args()

    res = run_backtest(args.season, args.start_gw, args.end_gw)
    if res.empty:
        print("no gameweeks evaluated")
        return 1

    print("\n" + "=" * 68)
    print(f"BACKTEST {args.season}  (GW{args.start_gw}-{args.end_gw}, {len(res)} gameweeks)")
    print("=" * 68)

    print(f"\nRank correlation with actual points   {'model':>8}{'ppg':>8}{'delta':>9}")
    for label, m, b in [
        ("  all players", "all_model", "all_ppg"),
        ("  players who appeared", "played_model", "played_ppg"),
        ("  price >= £5.0m", "ownable_model", "ownable_ppg"),
    ]:
        d = res[m].mean() - res[b].mean()
        print(f"{label:<38}{res[m].mean():>8.3f}{res[b].mean():>8.3f}{d:>+9.3f}")
    print(f"{'  (price-only baseline)':<38}{res['all_price'].mean():>8.3f}")

    print(f"\nAbsolute accuracy on next-gameweek points")
    print(f"{'  population':<38}{'MAE':>8}{'RMSE':>8}{'R2':>9}")
    for label, name in (("all", "  all players"), ("played", "  players who appeared"),
                        ("ownable", "  price >= £5.0m")):
        print(f"{name:<38}{res[f'{label}_mae'].mean():>8.4f}"
              f"{res[f'{label}_rmse'].mean():>8.4f}{res[f'{label}_r2'].mean():>9.4f}")
    print("  published band (Valouxis 2023, n=6900)   1.29-1.42   2.27-2.38   0.29-0.35")
    print("  note: population definitions differ; a band to sit inside, not a leaderboard")
    print("  * selects on the outcome and is biased against the model by construction:")
    print("    xp_next is unconditional, so it prices in the chance he does not play,")
    print("    while the population is filtered to players who did. Diagnostic only.")

    print(f"\nCalibration (level, not ordering)")
    print(f"{'  population':<38}{'predicted':>10}{'actual':>9}{'bias':>9}")
    for label, name in (("all", "  all players"), ("played", "  players who appeared"),
                        ("ownable", "  price >= £5.0m")):
        pr, ac = res[f"{label}_pred"].mean(), res[f"{label}_actual"].mean()
        print(f"{name:<38}{pr:>10.3f}{ac:>9.3f}{pr - ac:>+9.3f}")

    print(f"\nMean actual points of top-N picks     {'model':>8}{'ppg':>8}{'delta':>9}")
    for n in (20, 30, 50):
        d = res[f"model_top{n}"].mean() - res[f"ppg_top{n}"].mean()
        print(f"{'  top ' + str(n):<38}{res[f'model_top{n}'].mean():>8.3f}"
              f"{res[f'ppg_top{n}'].mean():>8.3f}{d:>+9.3f}")

    print(f"\nMean absolute error: {res['model_mae'].mean():.3f} points/player/gameweek")

    beat = (res["model_top30"] > res["ppg_top30"]).mean()
    print(f"Model's top 30 outscored the baseline's in {beat:.0%} of gameweeks.")

    if res["all_model"].mean() < res["all_ppg"].mean():
        print(
            "\nNote: the model trails on the all-players population. That gap is\n"
            "ordering players who never appear, where points-per-game wins by\n"
            "construction. It matters for transfer suggestions, not squad picks."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
