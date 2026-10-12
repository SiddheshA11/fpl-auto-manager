"""
What the engine predicted, against what happened.

The engine ran five gameweeks without recording a single prediction, so
diagnosing a decline meant archaeology through run logs that expire in 30
days - and GW1 to GW3 were already gone by the time anyone asked. `ledger.py`
fixed the recording. This reads it back.

Two different questions, kept apart on purpose:

  - **calibration**: does xp_next match what players went on to do? Reported
    with a caveat, because it is computed over the fifteen players the model
    chose, whose xP all sit in a narrow band. Range restriction crushes the
    correlation regardless of model quality, so read the bias and MAE and
    treat r as near-meaningless on a single gameweek.
  - **decision loss**: how many points a better *lineup* or *armband* would
    have won from the same fifteen. This is the actionable number. Measured
    over GW1-5 by hand it came to 7.4 and 4.6 points a gameweek.

Hindsight-optimal is not a target - nobody achieves it - but the gap's trend
and its composition are what tell you where to look.

    python3 calibrate.py                 # every gameweek in the ledger
    python3 calibrate.py --gw 7
    python3 calibrate.py --dir ./local    # read records from disk instead

Reads public endpoints and a public branch; no credentials, no token spent.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics as st
import urllib.error
import urllib.request
from itertools import combinations

logger = logging.getLogger("fpl_auto.calibrate")

DEFAULT_REPO = "SiddheshA11/fpl-auto-manager"
LEDGER_BRANCH = "predictions"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/predictions/gw{gw:02d}.json"
LIVE = "https://fantasy.premierleague.com/api/event/{gw}/live/"
BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

# FPL's legal XI shape, by element_type.
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}


def _fetch(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "fpl-auto-manager"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def load_records(gws=None, repo=DEFAULT_REPO, directory=None, fetcher=_fetch) -> list[dict]:
    """Ledger records, from the predictions branch or a local directory."""
    out = []
    if directory:
        for path in sorted(pathlib.Path(directory).glob("gw*.json")):
            out.append(json.loads(path.read_text()))
        if gws:
            out = [r for r in out if int(r["event"]) in set(gws)]
        return out

    for gw in (gws or range(1, 39)):
        rec = fetcher(RAW.format(repo=repo, branch=LEDGER_BRANCH, gw=int(gw)))
        if rec:
            out.append(rec)
    return out


def actuals(gw: int, fetcher=_fetch) -> dict[int, dict]:
    data = fetcher(LIVE.format(gw=int(gw)))
    if not data:
        return {}
    return {int(e["id"]): e.get("stats", {}) for e in data.get("elements", [])}


def _legal(positions: list[int]) -> bool:
    counts = {k: positions.count(k) for k in XI_MIN}
    return all(XI_MIN[k] <= counts[k] <= XI_MAX[k] for k in XI_MIN)


def best_legal_xi(players: list[dict], points: dict[int, int]) -> tuple[int, list[int]]:
    """The highest-scoring legal eleven from these fifteen, in hindsight."""
    ids = [int(p["id"]) for p in players]
    pos = {int(p["id"]): int(p.get("position", 0)) for p in players}
    best = (None, [])
    for xi in combinations(ids, 11):
        if not _legal([pos[i] for i in xi]):
            continue
        total = sum(points.get(i, 0) for i in xi)
        if best[0] is None or total > best[0]:
            best = (total, list(xi))
    return best


def assess(record: dict, act: dict[int, dict]) -> dict:
    players = record.get("players", [])
    pts = {int(p["id"]): int(act.get(int(p["id"]), {}).get("total_points", 0)) for p in players}
    mins = {int(p["id"]): int(act.get(int(p["id"]), {}).get("minutes", 0)) for p in players}

    started = [p for p in players if p.get("started")]
    benched = [p for p in players if not p.get("started")]
    cap = next((int(p["id"]) for p in players if p.get("is_captain")), None)

    xi_actual = sum(pts[int(p["id"])] for p in started)
    best_total, _ = best_legal_xi(players, pts)
    best_cap = max(started, key=lambda p: pts[int(p["id"])], default=None)

    predicted = [float(p["xp_next"]) for p in players if "xp_next" in p]
    observed = [pts[int(p["id"])] for p in players if "xp_next" in p]

    misranked = sorted(
        [(pts[int(b["id"])], b.get("name", b["id"]), float(b.get("xp_next", 0)),
          float(b.get("p_start", 0)))
         for b in benched if pts[int(b["id"])] >= 5],
        reverse=True,
    )
    return {
        "event": int(record["event"]),
        "xi_actual": xi_actual,
        "best_xi": best_total,
        "lineup_loss": (best_total - xi_actual) if best_total is not None else None,
        "bench_points": sum(pts[int(p["id"])] for p in benched),
        "captain": cap,
        "captain_points": pts.get(cap, 0) if cap else 0,
        "captain_loss": (pts[int(best_cap["id"])] - pts.get(cap, 0)) if best_cap and cap else 0,
        "predicted": predicted,
        "observed": observed,
        "misranked": misranked,
        "started_blanks": [p.get("name", p["id"]) for p in started if mins[int(p["id"])] == 0],
        "chip": record.get("chip"),
        "tilt_source": record.get("tilt_source"),
    }


def report(rows: list[dict]) -> None:
    if not rows:
        print("The ledger is empty. It starts recording on the next submitting run;\n"
              "nothing before that gameweek was ever written down.")
        return

    print(f"\n{'GW':>3}{'XI':>6}{'best XI':>9}{'lineup loss':>13}{'bench':>7}"
          f"{'cap':>6}{'cap loss':>10}")
    print("-" * 56)
    for r in rows:
        print(f"{r['event']:>3}{r['xi_actual']:>6}{r['best_xi']:>9}{r['lineup_loss']:>13}"
              f"{r['bench_points']:>7}{r['captain_points']:>6}{r['captain_loss']:>10}")

    n = len(rows)
    ll = [r["lineup_loss"] for r in rows]
    cl = [r["captain_loss"] for r in rows]
    print(f"\n{n} gameweek(s).  lineup loss {sum(ll)} total, {sum(ll)/n:.1f}/GW"
          f"   |   captaincy loss {sum(cl)} total, {sum(cl)/n:.1f}/GW")
    print("Hindsight-optimal is not a target. Watch the trend and the composition.")

    pred = [x for r in rows for x in r["predicted"]]
    obs = [x for r in rows for x in r["observed"]]
    if len(pred) >= 2:
        bias = st.mean(obs) - st.mean(pred)
        mae = st.mean(abs(a - b) for a, b in zip(pred, obs))
        print(f"\ncalibration over {len(pred)} player-gameweeks: "
              f"mean xP {st.mean(pred):.2f}, mean actual {st.mean(obs):.2f}, "
              f"bias {bias:+.2f}, MAE {mae:.2f}")
        if len(pred) >= 3 and st.pstdev(pred) > 0 and st.pstdev(obs) > 0:
            cov = sum((a - st.mean(pred)) * (b - st.mean(obs)) for a, b in zip(pred, obs)) / len(pred)
            r = cov / (st.pstdev(pred) * st.pstdev(obs))
            print(f"  r = {r:.3f} - computed over the fifteen the model chose, whose xP sit in "
                  f"a narrow band,\n  so range restriction depresses this badly. Do not read it "
                  f"as the model's R2.")

    print("\nWhere the lineup went wrong - benched players who returned 5+:")
    any_bad = False
    for r in rows:
        for scored_pts, name, xp, p_start in r["misranked"]:
            any_bad = True
            print(f"  GW{r['event']}  {name:<16} scored {scored_pts:>2}  "
                  f"predicted {xp:.2f} xP, p_start {p_start:.2f}")
    if not any_bad:
        print("  none - no benched player returned 5 or more.")

    blanks = [(r["event"], n_) for r in rows for n_ in r["started_blanks"]]
    if blanks:
        print("\nStarted but played no minutes (a p_start failure, the dominant error):")
        for gw, name in blanks:
            print(f"  GW{gw}  {name}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gw", type=int, action="append",
                    help="limit to these gameweeks (repeatable)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dir", dest="directory",
                    help="read records from a local directory instead of the branch")
    args = ap.parse_args()

    records = load_records(args.gw, repo=args.repo, directory=args.directory)
    rows = []
    for rec in sorted(records, key=lambda r: int(r["event"])):
        gw = int(rec["event"])
        act = actuals(gw)
        if not act:
            logger.info("GW%d has no live data yet; skipping", gw)
            continue
        rows.append(assess(rec, act))
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
