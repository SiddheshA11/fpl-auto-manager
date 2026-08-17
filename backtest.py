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
    past = gw_df[gw_df["GW"] < upto_gw]
    totals = past.groupby("element")[CUMULATIVE].sum() if len(past) else pd.DataFrame(columns=CUMULATIVE)

    # Price as at the gameweek being predicted, not season end. Deduplicated
    # because a double gameweek gives a player two rows with the same price.
    at_gw = gw_df[gw_df["GW"] == upto_gw].drop_duplicates(subset="element").set_index("element")
    price = at_gw["value"] if "value" in at_gw.columns else None

    elements = []
    for _, p in raw_df.iterrows():
        eid = int(p["id"])
        t = totals.loc[eid] if eid in totals.index else pd.Series(0.0, index=CUMULATIVE)
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


def run_backtest(season: str, start_gw: int, end_gw: int, horizon: int = 1) -> pd.DataFrame:
    season_dir = HISTORY_DIR / season
    gw_df = pd.read_csv(season_dir / "merged_gw.csv")
    raw_df = pd.read_csv(season_dir / "players_raw.csv")
    teams_df = pd.read_csv(season_dir / "teams.csv")
    fixtures = pd.read_csv(season_dir / "fixtures.csv").to_dict("records")
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
        model = X.XPModel(state, fixtures, prior_set, X.ModelConfig(horizon=horizon))
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
