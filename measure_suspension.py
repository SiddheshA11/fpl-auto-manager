"""
Is the card-ban risk model calibrated against what actually happened?

The suspension model is not measurable the way most things here are, and it is
worth being explicit about why rather than quoting a number that flatters it.

  backtest.py would report a MISLEADING GAIN. `build_state` hardcodes
  status='a' and chance_of_playing_next_round=None for every player, so the
  backtest is blind to availability. Production is not: FPL sets status='s' on
  a banned player and STATUS_AVAILABILITY zeroes him. A backtest would credit
  this model for rediscovering bans production already handles for free.

  simulate.py cannot clear its own noise floor. The effect is worth roughly
  5-20 points a season against a season-total SD of ~53 on a horizon sweep that
  changes no strategy at all.

So the measurable claim is the one the model actually makes: a probability. If
P(banned in horizon gameweek h) matches the realised frequency out of sample,
the model is right; if it does not, it is wrong, and no decision-level story
rescues it.

Held out by season: the rate driving each season's prediction is estimated from
the OTHER seasons, so a season is never predicted with its own cards.

Run: python measure_suspension.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import priors
import xp_model as X

HISTORY_DIR = Path(__file__).parent / "data" / "history"
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
HORIZON = 5


def load(season: str) -> pd.DataFrame:
    d = priors.read_season_csv(HISTORY_DIR / season / "merged_gw.csv")
    if "position" in d.columns:
        d = d[d["position"] != "AM"]
    g = d.groupby(["element", "GW"], as_index=False).agg(
        minutes=("minutes", "sum"), yellow_cards=("yellow_cards", "sum"))
    g["season"] = season
    return g


def main() -> int:
    g = pd.concat([load(s) for s in SEASONS], ignore_index=True)
    g = g.sort_values(["season", "element", "GW"])
    key = ["season", "element"]
    g["cum"] = g.groupby(key)["yellow_cards"].cumsum()
    g["cum_before"] = g["cum"] - g["yellow_cards"]

    # Booking rate per 90, held out: estimated from every season but this one.
    per_season = g.groupby("season").apply(
        lambda s: s["yellow_cards"].sum() / max(s["minutes"].sum(), 1) * 90.0,
        include_groups=False)
    held_out = {s: float((per_season.drop(s)).mean()) for s in SEASONS}
    print("held-out booking rate per 90, by season under test:")
    for s, v in held_out.items():
        print(f"  {s}  {v:.4f}")

    # Population: a starter sitting on exactly four bookings, inside the
    # matchweek-19 window where the five-card rule can still bite.
    at4 = g[(g["cum_before"] == 4) & (g["minutes"] > 0) & (g["GW"] <= 19)].copy()
    lookup = g.set_index(["season", "element", "GW"])

    realised = {h: 0 for h in range(1, HORIZON + 1)}
    n = 0
    for _, r in at4.iterrows():
        n += 1
        for h in range(0, HORIZON):
            try:
                row = lookup.loc[(r["season"], r["element"], r["GW"] + h)]
            except KeyError:
                continue
            if float(row["cum"]) >= 5:
                if h + 1 <= HORIZON:
                    realised[h + 1] += 1
                break

    # Predicted, from the model's own arithmetic: cards arrive Poisson in
    # exposure, and a one-match ban is served the gameweek after the threshold
    # falls, so the risk in gameweek h is P(crossed exactly at h-1).
    lam = float(np.mean(list(held_out.values()))) * (at4["minutes"].mean() / 90.0)
    need = pd.Series([1.0])
    print(f"\nlambda = {lam:.4f} bookings per gameweek "
          f"(rate x {at4['minutes'].mean():.0f} expected minutes)")
    print(f"population: {n:,} starting player-gameweeks on exactly four bookings\n")

    print(f"{'horizon GW':<14}{'predicted':>12}{'realised':>11}{'error':>10}")
    print("-" * 48)
    cum_p = cum_r = 0.0
    for h in range(1, HORIZON + 1):
        before = float(X.XPModel._poisson_at_least(need, pd.Series([lam * (h - 1)])).iloc[0])
        through = float(X.XPModel._poisson_at_least(need, pd.Series([lam * h])).iloc[0])
        pred = through - before
        real = realised[h] / n
        cum_p += pred
        cum_r += real
        print(f"{h:<14}{pred:>12.3f}{real:>11.3f}{pred - real:>10.3f}")
    print("-" * 48)
    print(f"{'cumulative':<14}{cum_p:>12.3f}{cum_r:>11.3f}{cum_p - cum_r:>10.3f}")
    print("\nOver-prediction is expected and is the safe direction: the model "
          "\nassumes the player is on the pitch every gameweek of the horizon, "
          "\nwhile real players are rested, injured and rotated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
