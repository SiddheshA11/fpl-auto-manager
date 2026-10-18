"""
Statistics for the decision-replay harness, with the analysis fixed in advance.

The rules below were committed before any candidate arm's points were read.
The only candidate data used to set them is alpha 1.0's flip RATE, which
tunes the placebo; its points were never printed. Changing a rule after
seeing a result is exactly the forking path this file exists to close.

PRE-REGISTERED RULES

  1. Window length L = 8 is primary. L = 6 and 12 are sensitivity checks.

  2. Terminal value. The primary outcome is diff_terminal (window points plus
     the frozen final squads' next 4 gameweeks) unless it inflates the
     placebo's window-level SD by more than 25% over plain diff, in which
     case it is plain diff. Terminal value removes a bias - a candidate that
     buys well late in a window is otherwise not credited - so it is kept
     unless its variance cost is large.

  3. The placebo. Two noise arms with independent seeds are exchangeable, so
     their difference has expected value exactly zero, and they make real
     decision flips. eps is set so the noise arm flips about as often as
     alpha 1.0 does. A replicate is one seed pair, scored on every window.

     Measured flip profile, L=8, 144 windows (alpha 1.0: flips only, its
     points never read):

         arm          P(flip)  flipped GWs/window  first flip
         alpha=1.0      98.6%                6.67        1.02
         noise 0.1      95.1%                2.97        1.87
         noise 0.2     100.0%                5.07        1.04
         noise 0.3     100.0%                6.38        0.68
         noise 0.5     100.0%                7.28        0.28

     eps = 0.3 is primary (closest on intensity); 0.2 is the sensitivity
     check (closest on onset). Together they bracket the candidate. 20
     replicates at 0.3, 10 at 0.2. alpha 1.0 is not a near-tie nudge: it
     changes decisions in most gameweeks of almost every window.

  4. Standard error: season-clustered, G = 5, t with 4 degrees of freedom.
     Calibrated on the placebo replicates: their p-values should be
     Uniform(0, 1). Reported as the KS distance from uniform and the
     5%-level rejection rate. ddof = 1 throughout.

     The plan also proposed a block bootstrap within season. Measured on
     synthetic windows with known dependence (tests/test_replay_stats.py), it
     came out at 0.21-0.66 of the true SE: resampling inside a season keeps
     every season's level, so between-season variation is invisible to it,
     and with season-length blocks its variance collapses to zero. The
     cluster SE was 0.95-1.01 of the truth in every structure tried. The
     draft criterion was also wrong - "t-statistics with SD closest to 1" -
     because t on 4 dof has SD sqrt(2) when perfectly calibrated. Both were
     fixed before any candidate was read.

  5. Power is stated before any candidate is tested: the MDE at 80% power,
     two-sided 5%, from the calibrated SE on the placebo, in points per
     gameweek and per 38-gameweek season. If it cannot resolve the effects
     the open questions need, that is the result.

  6. A candidate is reported as: mean effect with CI, the t-statistic under
     the chosen SE, and the flip decomposition - P(flip), E[diff | flip],
     the single-flip step size - never as one season total.

    python3 replay_stats.py placebo  data/replay/placebo_*.csv
    python3 replay_stats.py arm      data/replay/<arm>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TERMINAL_SD_INFLATION_LIMIT = 1.25
GW_PER_SEASON = 38


def placebo_pairs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    One frame per replicate: seed 2r against seed 2r+1, window by window.
    Each noise arm's diff is against the same baseline, so their difference
    is the difference between the two arms.
    """
    noise = df[df["arm"].str.startswith("noise=")].copy()
    noise["seed"] = noise["arm"].str.split("@").str[1].astype(int)
    pairs = []
    for r in sorted(set(noise["seed"] // 2)):
        a = noise[noise["seed"] == 2 * r].set_index(["season", "start"])
        b = noise[noise["seed"] == 2 * r + 1].set_index(["season", "start"])
        if a.empty or b.empty:
            continue
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        pairs.append(pd.DataFrame({
            "season": j.index.get_level_values(0), "start": j.index.get_level_values(1),
            "diff": (j["diff_a"] - j["diff_b"]).to_numpy(),
            "diff_terminal": (j["diff_terminal_a"] - j["diff_terminal_b"]).to_numpy(),
            "flipped": (j["flipped_a"] | j["flipped_b"]).to_numpy(),
        }))
    return pairs


def se_cluster(d: pd.DataFrame, col: str) -> tuple[float, int]:
    """Cluster-robust SE of the mean, clusters = seasons. Returns (se, dof)."""
    x = d[col].to_numpy(dtype=float)
    mean = x.mean()
    resid = d.assign(_r=x - mean).groupby("season")["_r"].sum().to_numpy()
    g, n = len(resid), len(x)
    se = np.sqrt(g / (g - 1) * np.sum(resid ** 2)) / n
    return float(se), g - 1


def test_stat(d: pd.DataFrame, col: str) -> tuple[float, float, float, int]:
    """(mean, se, two-sided p, dof), season-clustered."""
    mean = float(d[col].mean())
    se, dof = se_cluster(d, col)
    p = 2 * stats.t.sf(abs(mean / se), dof) if se > 0 else 1.0
    return mean, se, float(p), dof


def calibrate(pairs: list[pd.DataFrame], col: str) -> dict:
    """How the test behaves on replicates whose true effect is exactly zero."""
    results = [test_stat(d, col) for d in pairs]
    ps = np.array([r[2] for r in results])
    ts = np.array([r[0] / r[1] if r[1] > 0 else 0.0 for r in results])
    dof = results[0][3]
    return {"ks": float(stats.kstest(ps, "uniform").statistic),
            "ks_p": float(stats.kstest(ps, "uniform").pvalue),
            "reject": float((ps < 0.05).mean()), "n": len(pairs),
            "t_sd": float(ts.std(ddof=1)), "t_sd_expected": float(np.sqrt(dof / (dof - 2)))}


def choose_outcome(pairs: list[pd.DataFrame]) -> tuple[str, float]:
    """Rule 2."""
    sd_plain = np.mean([d["diff"].std(ddof=1) for d in pairs])
    sd_term = np.mean([d["diff_terminal"].std(ddof=1) for d in pairs])
    ratio = sd_term / sd_plain if sd_plain > 0 else np.inf
    return ("diff_terminal" if ratio <= TERMINAL_SD_INFLATION_LIMIT else "diff"), float(ratio)


def mde(se: float, dof: int | None = None) -> float:
    """Two-sided 5%, 80% power."""
    if dof:
        return float((stats.t.ppf(0.975, dof) + stats.t.ppf(0.80, dof)) * se)
    return float((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se)


def per_gw(x: float, L: int) -> float:
    """
    Per decision-gameweek. A window holds L gameweeks of decisions; terminal
    gameweeks only collect the payoff of those same decisions, so dividing by
    L + T would understate the effect of playing the rule for one more week.
    """
    return x / L


def flip_decomposition(d: pd.DataFrame, col: str) -> dict:
    flipped = d[d["flipped"]]
    single = d[d["flip_gws"] == 1] if "flip_gws" in d.columns else flipped.iloc[0:0]
    return {
        "p_flip": float(d["flipped"].mean()),
        "unflipped_all_zero": bool((d.loc[~d["flipped"], col] == 0).all()),
        "mean_given_flip": float(flipped[col].mean()) if len(flipped) else float("nan"),
        "sd_given_flip": float(flipped[col].std(ddof=1)) if len(flipped) > 1 else float("nan"),
        "abs_step_single_flip": float(single[col].abs().median()) if len(single) else float("nan"),
        "n_flipped": int(len(flipped)),
    }


def report_placebo(paths: list[str]) -> int:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    L = int(df["L"].iloc[0])
    pairs = placebo_pairs(df)
    if len(pairs) < 2:
        print("need at least two placebo replicates")
        return 1
    col, ratio = choose_outcome(pairs)
    cal = calibrate(pairs, col)
    print(f"placebo: {len(pairs)} replicates x {len(pairs[0])} windows, L={L}\n")
    print(f"rule 2: terminal SD / plain SD = {ratio:.3f} (limit {TERMINAL_SD_INFLATION_LIMIT})"
          f" -> outcome = {col}")
    print(f"P(flip) per window, placebo pair: {np.mean([d['flipped'].mean() for d in pairs]):.1%}\n")
    print(f"rule 4: season-clustered test on {cal['n']} null replicates ({col})")
    print(f"  p-values vs Uniform(0,1): KS {cal['ks']:.3f} (p {cal['ks_p']:.2f}); "
          f"rejected at 5%: {cal['reject']:.1%}; "
          f"t sd {cal['t_sd']:.2f} (calibrated: {cal['t_sd_expected']:.2f})")

    se = float(np.median([test_stat(d, col)[1] for d in pairs]))
    m = mde(se, 4)
    print(f"\nrule 5: power (median placebo SE {se:.2f} pts/window)")
    print(f"  MDE {m:.2f} pts/window = {per_gw(m, L):.3f} pts/GW"
          f" = {per_gw(m, L) * GW_PER_SEASON:.1f} pts/season")
    return 0


def report_arm(path: str, col: str) -> int:
    df = pd.read_csv(path)
    L = int(df["L"].iloc[0])
    for name, d in df.groupby("arm", sort=False):
        mean, se, p, dof = test_stat(d, col)
        crit = stats.t.ppf(0.975, dof)
        f = flip_decomposition(d, col)
        g = per_gw(1.0, L) * GW_PER_SEASON
        print(f"{name}: n={len(d)}  mean {mean:+.2f} pts/window "
              f"[{mean - crit * se:+.2f}, {mean + crit * se:+.2f}]  t={mean / se:+.2f}  p={p:.3f}"
              f"  = {mean * g:+.1f} pts/season")
        print(f"   P(flip) {f['p_flip']:.1%}; given a flip {f['mean_given_flip']:+.2f} "
              f"(sd {f['sd_given_flip']:.1f}, n={f['n_flipped']}); single-flip |step| median "
              f"{f['abs_step_single_flip']:.1f}; unflipped windows all zero: {f['unflipped_all_zero']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("placebo")
    p.add_argument("paths", nargs="+")
    a = sub.add_parser("arm")
    a.add_argument("path")
    a.add_argument("--outcome", choices=["diff", "diff_terminal"], required=True,
                   help="the outcome rule 2 chose on the placebo")
    args = ap.parse_args()
    if args.cmd == "placebo":
        return report_placebo(args.paths)
    return report_arm(args.path, args.outcome)


if __name__ == "__main__":
    sys.exit(main())
