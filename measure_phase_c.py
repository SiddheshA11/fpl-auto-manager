"""
Phase C: the hand-picked constants, decided by rules fixed before the run.

The decision-replay harness has ~85-90 pts/season MDE at 80% power, so it
cannot settle a constant by itself. It is used as a VETO: it can see a
change that costs ~90+ points a season, and nothing finer. Accuracy decides
what accuracy can see, which is the model's constants, not the optimiser's.

PRE-REGISTERED RULES (committed before any number below was produced)

  A. OUTFIELD_COMPETITION_ALPHA - a model constant, so accuracy can judge it.
     Grid {0.75, 1.0, 1.25} against production 1.5. Next-gameweek backtest,
     GW10-38 of each replayable season (2021-22 to 2025-26), paired per
     gameweek on all-player R2. Re-measured here from the data, not taken
     from the earlier t=7.9, which was this repo's own arithmetic.
     A value is ELIGIBLE if all of:
       A1. season-clustered t4 test of the per-gameweek R2 difference favours
           it at p < 0.05/3 (three candidates);
       A2. it is ahead on R2 in at least 4 of the 5 seasons;
       A3. all-player MAE is not significantly worse (same test, p < 0.05);
       A4. the harness does not veto it: its window effect is not negative
           at p < 0.05 (two-sided, season-clustered t4, plain window points).
     If any are eligible, the one with the largest mean R2 gain is adopted.
     Otherwise production stays.

  B. FREE_TRANSFER_VALUE {0.0, 0.6}, DEFAULT_BENCH_WEIGHT {0.05, 0.3},
     HORIZON {3, 4, 6, 7} - optimiser constants; accuracy says nothing about
     them. A candidate replaces production only if the harness finds it
     BETTER at p < 0.05/8. Otherwise production stays and the measured
     effect, CI and flip decomposition are recorded.

  C. Ownership tilt (OWNERSHIP_WEIGHT / FIELD_OWNERSHIP_WEIGHT 0.2). Its
     purpose is rank, which neither tool measures, and the field version
     cannot be replayed. Only the points cost of the template tilt
     (own=0.0 against 0.2) is measured, and reported as a fact. No change.

  D. Any production change lands only with CI green twice, and not between
     09:30 and 17:00 UTC on a deadline day.

    python3 measure_phase_c.py            # everything, ~40 minutes
    python3 measure_phase_c.py --accuracy # rule A's accuracy half only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

import replay_harness as R
import replay_stats as RS

SEASONS = R.SEASONS
ALPHA_GRID = [0.75, 1.0, 1.25]
PRODUCTION_ALPHA = 1.5
ACCURACY_GWS = (10, 38)
OPTIMISER_ARMS = ["ftv=0.0", "ftv=0.6", "bench=0.05", "bench=0.3",
                  "horizon=3", "horizon=4", "horizon=6", "horizon=7"]
TILT_ARMS = ["own=0.0"]


def _backtest(args) -> pd.DataFrame:
    season, alpha = args
    logging.disable(logging.WARNING)
    import backtest
    import xp_model as X
    X.OUTFIELD_COMPETITION_ALPHA = alpha
    res = backtest.run_backtest(season, *ACCURACY_GWS)
    return res.assign(season=season, alpha=alpha)[["season", "alpha", "gw", "all_r2", "all_mae"]]


def accuracy_table(workers: int) -> pd.DataFrame:
    tasks = [(s, a) for a in [PRODUCTION_ALPHA, *ALPHA_GRID] for s in SEASONS]
    with ProcessPoolExecutor(workers) as ex:
        return pd.concat(list(ex.map(_backtest, tasks)), ignore_index=True)


def clustered(diff: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """(mean, t, two-sided p), season-clustered t4."""
    mean, se, p, _ = RS.test_stat(diff, col)
    return mean, (mean / se if se > 0 else 0.0), p


def judge_accuracy(acc: pd.DataFrame) -> pd.DataFrame:
    base = acc[acc["alpha"] == PRODUCTION_ALPHA].set_index(["season", "gw"])
    rows = []
    for a in ALPHA_GRID:
        cand = acc[acc["alpha"] == a].set_index(["season", "gw"])
        j = cand.join(base, rsuffix="_base", how="inner").reset_index()
        d = pd.DataFrame({"season": j["season"],
                          "r2": j["all_r2"] - j["all_r2_base"],
                          "mae": j["all_mae"] - j["all_mae_base"]})
        r2_mean, r2_t, r2_p = clustered(d, "r2")
        mae_mean, mae_t, mae_p = clustered(d, "mae")
        per_season = d.groupby("season")["r2"].mean()
        rows.append({
            "alpha": a, "n_gw": len(d), "r2_gain": r2_mean, "r2_t": r2_t, "r2_p": r2_p,
            "gw_ahead": f"{int((d['r2'] > 0).sum())}/{len(d)}",
            "seasons_ahead": int((per_season > 0).sum()),
            "naive_t": float(d["r2"].mean() / (d["r2"].std(ddof=1) / np.sqrt(len(d)))),
            "mae_change": mae_mean, "mae_p": mae_p,
            "A1": r2_mean > 0 and r2_p < 0.05 / len(ALPHA_GRID),
            "A2": int((per_season > 0).sum()) >= 4,
            "A3": not (mae_mean > 0 and mae_p < 0.05),
        })
    return pd.DataFrame(rows)


def harness(arms: list[str], workers: int) -> pd.DataFrame:
    df = R.run([R.parse_arm(a) for a in arms], workers=workers)
    out = []
    for name, d in df.groupby("arm", sort=False):
        mean, t, p = clustered(d, "diff")
        f = RS.flip_decomposition(d, "diff")
        se = RS.test_stat(d, "diff")[1]
        crit = stats.t.ppf(0.975, 4)
        out.append({"arm": name, "n": len(d), "mean": mean, "lo": mean - crit * se,
                    "hi": mean + crit * se, "t": t, "p": p,
                    "per_season": RS.per_gw(mean, int(d["L"].iloc[0])) * RS.GW_PER_SEASON,
                    "p_flip": f["p_flip"], "mean_given_flip": f["mean_given_flip"],
                    "step": f["abs_step_single_flip"]})
    return pd.DataFrame(out)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accuracy", action="store_true", help="rule A's accuracy half only")
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4) - 1)
    args = ap.parse_args()

    acc = judge_accuracy(accuracy_table(args.workers))
    pd.set_option("display.width", 200)
    print("RULE A - accuracy, paired per gameweek against alpha 1.5")
    print(acc.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    if args.accuracy:
        return 0

    arms = [f"alpha={a}" for a in ALPHA_GRID] + OPTIMISER_ARMS + TILT_ARMS
    h = harness(arms, args.workers)
    print("\nHARNESS - window points against production, L=8, season-clustered t4")
    print(h.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    print("\nDECISIONS")
    hv = h.set_index("arm")
    eligible = []
    for _, r in acc.iterrows():
        arm = f"alpha={r['alpha']}"
        a4 = not (hv.loc[arm, "mean"] < 0 and hv.loc[arm, "p"] < 0.05)
        ok = r["A1"] and r["A2"] and r["A3"] and a4
        print(f"  alpha {r['alpha']}: A1 {r['A1']} A2 {r['A2']} A3 {r['A3']} A4 {a4}"
              f" -> {'ELIGIBLE' if ok else 'not eligible'}")
        if ok:
            eligible.append((r["r2_gain"], r["alpha"]))
    if eligible:
        print(f"  RULE A: adopt OUTFIELD_COMPETITION_ALPHA = {max(eligible)[1]}")
    else:
        print("  RULE A: production alpha 1.5 stays")
    k = len(OPTIMISER_ARMS)
    better = [a for a in OPTIMISER_ARMS if hv.loc[a, "mean"] > 0 and hv.loc[a, "p"] < 0.05 / k]
    print(f"  RULE B: {'replace with ' + ', '.join(better) if better else 'every optimiser constant stays'}")
    # The own=0.0 arm is (no tilt) minus (production's 0.2 tilt), so the
    # tilt's effect is its negation. This line first printed that effect as a
    # "cost", which read a 62-point gain as a 62-point loss.
    r = hv.loc["own=0.0"]
    print(f"  RULE C: the 0.2 template tilt scores {-r['per_season']:+.1f} pts/season against no tilt"
          f" [CI {-r['hi'] / 8 * 38:+.0f}, {-r['lo'] / 8 * 38:+.0f}], p={r['p']:.3f}; no change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
