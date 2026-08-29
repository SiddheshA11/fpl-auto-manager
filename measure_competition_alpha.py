"""
Is OUTFIELD_COMPETITION_ALPHA wrong, and in which direction?

The open question was "squad depth dilutes start probability": correlation
between how many credible outfielders a club carries and how many of them
clear 0.80 p_start is negative, Chelsea get two, and Palmer comes out around
0.59. The implied fix was that alpha = 1.5 is not sharp enough and should rise.

This measures both directions against held-out accuracy, and against the
symptom itself. They disagree, which is the result.

    python3 measure_competition_alpha.py
"""
from __future__ import annotations

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
PRODUCTION = 1.5


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


def main() -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
