"""
Should the ownership tilt run on the mini-league field instead of the template,
and at what weight?

The template understates the field for every top-owned player, always in the
same direction. That makes it the wrong target - but it also makes the field a
*wider* distribution, so carrying the current weight across unchanged would
tilt harder as well as more accurately, and the two effects would be
impossible to tell apart afterwards.

So this reports three things:

  1. the gap, per player;
  2. the **control** - how much of any change is just a stronger tilt. The
     objective multiplies `weight * (1 - 2*ownership)`, so the tilt's strength
     scales with the spread of that term. Matching spread gives the weight
     that swaps the target without changing the force;
  3. what actually changes in the squad, at the naive weight and at the
     strength-matched one.

Reads public endpoints only; no credentials, no token spent.

    python3 measure_rival_tilt.py [--event N]
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

import optimizer as O
import priors
import recommend as R
import rivals
import xp_model as X
from config import FPL_TEAM_ID, OWNERSHIP_WEIGHT

logging.disable(logging.INFO)

DEFAULT_ENTRY = 5413589
TEMPLATE_COL = "selected_by_percent"


def tilt_term(ownership_pct: pd.Series) -> pd.Series:
    """The quantity the weight multiplies in the objective."""
    return 1.0 - 2.0 * (pd.to_numeric(ownership_pct, errors="coerce").fillna(0.0) / 100.0).clip(0, 1)


def strength_matched_weight(scored: pd.DataFrame, weight: float = OWNERSHIP_WEIGHT) -> tuple[float, float]:
    """
    The field weight that tilts exactly as hard as `weight` does on the
    template, plus the spread ratio it came from.
    """
    sd_t = tilt_term(scored[TEMPLATE_COL]).std()
    sd_f = tilt_term(scored[rivals.FIELD_OWNERSHIP_COL]).std()
    ratio = float(sd_f / sd_t)
    return weight / ratio, ratio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, default=FPL_TEAM_ID or DEFAULT_ENTRY)
    ap.add_argument("--event", type=int, default=None,
                    help="gameweek to read rival picks from (default: latest available)")
    args = ap.parse_args()

    bootstrap, fixtures = R.load_game_state(offline=False)
    events = X.next_events(bootstrap, 5)
    event = args.event or (events[0] - 1)

    fo = rivals.field_ownership_or_none(args.entry, event)
    if fo is None:
        print(f"no rival picks available for GW{event}; nothing to measure")
        return 1

    tc = {t["code"]: t["name"] for t in bootstrap["teams"]}
    ps = priors.build_priors(current_team_codes=tc)
    model = X.XPModel(bootstrap, fixtures, ps, X.ModelConfig(horizon=5))
    scored = rivals.attach(model.expected_points(events), fo)
    pool = scored[scored["status"].isin(["a", "d"])].copy()

    print(f"\nGW{event} field: {fo.managers} rivals across their mini leagues\n")

    top = pool.sort_values(rivals.FIELD_OWNERSHIP_COL, ascending=False).head(10)
    print(f"{'player':<18}{'field':>8}{'template':>10}{'gap pp':>9}{'captained':>11}")
    print("-" * 56)
    for _, r in top.iterrows():
        print(f"{r['web_name']:<18}{r[rivals.FIELD_OWNERSHIP_COL]:>7.1f}%"
              f"{float(r[TEMPLATE_COL]):>9.1f}%"
              f"{r[rivals.FIELD_OWNERSHIP_COL] - float(r[TEMPLATE_COL]):>+9.1f}"
              f"{r[rivals.FIELD_CAPTAINCY_COL]:>10.1f}%")

    matched, ratio = strength_matched_weight(pool)
    print("\nCONTROL - tilt strength, not accuracy")
    print(f"  sd(1-2*own) template {tilt_term(pool[TEMPLATE_COL]).std():.3f}"
          f"   field {tilt_term(pool[rivals.FIELD_OWNERSHIP_COL]).std():.3f}"
          f"   ratio {ratio:.3f}")
    print(f"  carrying +{OWNERSHIP_WEIGHT:.2f} across unchanged tilts at an effective "
          f"+{OWNERSHIP_WEIGHT * ratio:.3f}")
    print(f"  strength-matched field weight = +{matched:.3f}")

    print("\nWhat the squad actually does\n")
    print(f"{'configuration':<34}{'XI xP':>9}{'mean field EO':>15}{'captain':>16}")
    print("-" * 74)
    configs = [
        ("template  @ +%.2f  (production)" % OWNERSHIP_WEIGHT, TEMPLATE_COL, OWNERSHIP_WEIGHT),
        ("field     @ +%.2f  (naive swap)" % OWNERSHIP_WEIGHT, rivals.FIELD_OWNERSHIP_COL, OWNERSHIP_WEIGHT),
        ("field     @ +%.3f (matched)" % matched, rivals.FIELD_OWNERSHIP_COL, matched),
        ("no tilt   @  0.00 (control)", TEMPLATE_COL, 0.0),
    ]
    for label, col, w in configs:
        sol = O.SquadOptimizer(pool, value_col="xp_horizon", captain_col="xp_next",
                               ownership_weight=w, ownership_col=col).build_squad(100.0)
        xi = sol.xi
        eo = pool.set_index("id")[rivals.FIELD_OWNERSHIP_COL]
        cap = pool.loc[pool["id"] == sol.captain, "web_name"]
        print(f"{label:<34}{xi['xp_next'].sum():>9.2f}"
              f"{eo.reindex(xi['id'].astype(int)).mean():>14.1f}%"
              f"{(cap.iloc[0] if len(cap) else '?'):>16}")

    print("\nCaptaincy is a separate distribution and is NOT tilted here:")
    caps = pool[pool[rivals.FIELD_CAPTAINCY_COL] > 0].sort_values(
        rivals.FIELD_CAPTAINCY_COL, ascending=False)
    for _, r in caps.iterrows():
        print(f"  {r['web_name']:<16} owned {r[rivals.FIELD_OWNERSHIP_COL]:>5.1f}%"
              f"   captained {r[rivals.FIELD_CAPTAINCY_COL]:>5.1f}%"
              f"   xp_next {r['xp_next']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
