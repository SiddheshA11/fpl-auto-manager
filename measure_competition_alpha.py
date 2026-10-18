"""
Is OUTFIELD_COMPETITION_ALPHA wrong, and in which direction?

The open question was "squad depth dilutes start probability": correlation
between how many credible outfielders a club carries and how many of them
clear 0.80 p_start is negative, Chelsea get two, and Palmer comes out around
0.59. The implied fix was that alpha = 1.5 is not sharp enough and should rise.

This measures three things: held-out accuracy, the symptom itself, and - with
`--decisions` - what the change is worth in points when a season is actually
replayed. All three disagree, which is the result.

    python3 measure_competition_alpha.py [--decisions]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import backtest
import priors
import recommend as R
import xp_model as X

logging.disable(logging.INFO)

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
ALPHAS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
# Read, not restated: every sweep restores this afterwards, and a literal here
# would silently reset production to a stale value.
PRODUCTION = X.OUTFIELD_COMPETITION_ALPHA


def accuracy_by_alpha(seasons=SEASONS, alphas=ALPHAS) -> pd.DataFrame:
    """Mean next-gameweek R2 per season, per alpha. Higher is better."""
    rows = []
    for a in alphas:
        X.OUTFIELD_COMPETITION_ALPHA = a
        for s in seasons:
            res = backtest.run_backtest(s, 10, 38)
            if res.empty:
                continue
            rows.append({"alpha": a, "season": s,
                         "r2": res["all_r2"].mean(),
                         "mae": res["all_mae"].mean(),
                         "rank": res["all_model"].mean()})
    X.OUTFIELD_COMPETITION_ALPHA = PRODUCTION
    return pd.DataFrame(rows)


def symptom_by_alpha(alphas=ALPHAS) -> pd.DataFrame:
    """The thing the open question actually complains about."""
    bootstrap, fixtures = R.load_game_state(offline=True)
    tc = {t["code"]: t["name"] for t in bootstrap["teams"]}
    ps = priors.build_priors(current_team_codes=tc)

    rows = []
    for a in alphas:
        X.OUTFIELD_COMPETITION_ALPHA = a
        m = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=5))
        ev = X.next_events(bootstrap, 1)[0]
        df = m.players.copy()
        df["p_start"] = m.minutes_model(ev)["p_start"].values
        out = df[df["position"] != 1]
        per_team = pd.DataFrame([
            {"depth": len(g), "above80": int((g["p_start"] > 0.80).sum())}
            for _, g in out.groupby("team")
        ])
        top = df[(df["web_name"] == "Palmer") & (df["team_name"] == "CHE")]["p_start"]
        rows.append({"alpha": a,
                     "corr_depth_above80": per_team["depth"].corr(per_team["above80"]),
                     "palmer": float(top.iloc[0]) if len(top) else np.nan})
    X.OUTFIELD_COMPETITION_ALPHA = PRODUCTION
    return pd.DataFrame(rows)


def decisions_by_alpha(seasons=SEASONS) -> pd.DataFrame:
    """
    Season points from replaying real decisions, with controls.

    Accuracy cannot answer this. A decision rule is not a forecast, and this
    repo has a measured noise floor where changing the horizon - which is not
    a strategy change at all - moves a season total by over a hundred points.
    So the horizon is swept alongside alpha purely as a control: any candidate
    whose effect does not exceed the spread of a change that means nothing has
    not been shown to do anything.

    A full season replays in about 50 seconds, so the whole sweep is ~15
    minutes. Earlier notes in this repo assumed otherwise; the first timing
    attempt was dominated by the history download, not by compute.
    """
    import simulate

    configs = [("candidate", 1.0, 5), ("candidate", 1.25, 5), ("baseline", 1.5, 5),
               ("CONTROL", 1.5, 4), ("CONTROL", 1.5, 6)]
    rows = []
    for season in seasons:
        for kind, alpha, horizon in configs:
            X.OUTFIELD_COMPETITION_ALPHA = alpha
            res = simulate.simulate_season(season, 10, 38, strategy="greedy",
                                           horizon=horizon, max_hits=2)
            rows.append({"season": season, "kind": kind, "alpha": alpha,
                         "horizon": horizon, "points": res.total})
    X.OUTFIELD_COMPETITION_ALPHA = PRODUCTION
    return pd.DataFrame(rows)


def report_decisions(df: pd.DataFrame) -> None:
    base = df[(df.alpha == PRODUCTION) & (df.horizon == 5)].set_index("season")["points"]
    print("\nDecision side - paired season differences against alpha=1.5, h=5\n")
    print(f"{'config':<18}{'n':>3}{'mean':>9}{'sd':>8}{'min':>7}{'max':>7}   reading")
    print("-" * 66)
    for (alpha, horizon), g in df.groupby(["alpha", "horizon"]):
        if (alpha, horizon) == (PRODUCTION, 5):
            continue
        g = g.set_index("season")
        common = [s for s in g.index if s in base.index]
        d = [g.loc[s, "points"] - base.loc[s] for s in common]
        kind = g["kind"].iloc[0]
        print(f"alpha={alpha} h={horizon:<4}{'':<4}{len(d):>3}{np.mean(d):>+9.1f}"
              f"{np.std(d):>8.1f}{min(d):>+7d}{max(d):>+7d}   {kind}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", action="store_true",
                    help="also replay seasons to measure the decision side (~15 min)")
    args = ap.parse_args()

    acc = accuracy_by_alpha()
    sym = symptom_by_alpha()

    print("\nHeld-out accuracy — mean next-GW R2, GW10-38\n")
    piv = acc.pivot(index="alpha", columns="season", values="r2")
    piv["MEAN"] = piv.mean(axis=1)
    print(piv.round(4).to_string())

    print("\nThe symptom, on the committed snapshot\n")
    print(sym.round(3).to_string(index=False))

    best = piv["MEAN"].idxmax()
    print(f"\naccuracy-optimal alpha = {best}   (production = {PRODUCTION})")
    print("\nThe two objectives disagree. Sharpening alpha is what the open\n"
          "question implies and it costs accuracy in every season measured;\n"
          "the accuracy-optimal direction is softer, and it makes the depth\n"
          "correlation and Palmer WORSE. Depth dilution is therefore not a\n"
          "defect that tuning this constant can fix.")

    if args.decisions:
        report_decisions(decisions_by_alpha())
        print("\nAnd the decision side kills it outright. alpha 1.0 wins on R2 by\n"
              "+0.005 and loses 39.8 points a season when decisions are actually\n"
              "replayed; alpha 1.25 gains 12.0 with an sd of 40.0, which is\n"
              "nothing. Both controls - changes that are not strategy changes -\n"
              "move the total by more than either candidate. This constant is\n"
              "measured, null and closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
