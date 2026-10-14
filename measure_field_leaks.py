"""
How much do your rivals leak to hindsight, and how much do you?

A leak figure on its own is not evidence. The GW1-5 post-mortem measured this
team's lineup leak at 7.4 points a gameweek and its captaincy leak at 4.6, and
read both as defects - without ever asking what a normal leak looks like. That
was the central error of the diagnosis: the comparison group was available from
the same public endpoints the tilt already reads.

This computes the identical metric for every rival in the same mini leagues
over the same gameweeks, using `calibrate.assess` so the arithmetic is
literally the same code, and reports where this team sits in that distribution.

Public endpoints only; no credentials, no token spent. ~5 requests per rival
per gameweek, so a few minutes for a full field.

    python3 measure_field_leaks.py [--through 5]
"""
from __future__ import annotations

import argparse
import logging
import statistics as st

import calibrate
import rivals
from config import FPL_TEAM_ID

logging.disable(logging.INFO)

DEFAULT_ENTRY = 5413589


def _positions(client: rivals.PublicFPL) -> dict[int, int]:
    bs = client.get("/bootstrap-static/")
    return {int(e["id"]): int(e["element_type"]) for e in bs["elements"]}


def record_from_picks(picks: dict, event: int, positions: dict[int, int]) -> dict | None:
    """A rival's team in the ledger's own record shape, so assess() can score it."""
    rows = picks.get("picks") or []
    if len(rows) < 15:
        return None
    players = []
    captain = None
    for pk in rows:
        pid = int(pk["element"])
        is_cap = int(pk.get("multiplier", 1)) > 1 and pk.get("is_captain", False)
        if is_cap:
            captain = pid
        players.append({
            "id": pid,
            "name": str(pid),
            "position": positions.get(pid, 0),
            "started": int(pk["position"]) <= 11,
            "is_captain": bool(pk.get("is_captain")),
            "is_vice_captain": bool(pk.get("is_vice_captain")),
        })
    return {"schema": 1, "event": event, "captain": captain,
            "chip": picks.get("active_chip"), "players": players}


def leaks_for(entry_id: int, events: list[int], positions: dict[int, int],
              client: rivals.PublicFPL, actuals: dict[int, dict]) -> dict | None:
    lineup = captain = total = 0
    seen = 0
    for gw in events:
        picks = client.get(f"/entry/{entry_id}/event/{gw}/picks/")
        if not picks:
            continue
        rec = record_from_picks(picks, gw, positions)
        if rec is None:
            continue
        r = calibrate.assess(rec, actuals[gw])
        lineup += r["lineup_loss"] or 0
        captain += r["captain_loss"] or 0
        total += r["gw_score"]
        seen += 1
    if seen != len(events):
        return None
    return {"entry": entry_id, "lineup": lineup, "captain": captain,
            "combined": lineup + captain, "total": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, default=FPL_TEAM_ID or DEFAULT_ENTRY)
    ap.add_argument("--through", type=int, default=5, help="last finished gameweek")
    args = ap.parse_args()

    events = list(range(1, args.through + 1))
    client = rivals.PublicFPL(retries=2, delay=0.05)
    positions = _positions(client)

    league_ids = rivals.mini_league_ids(client, args.entry)
    ids: set[int] = set()
    for lid in league_ids:
        ids.update(rivals.rival_ids(client, lid, args.entry))
    ids.discard(args.entry)
    print(f"{len(ids)} rivals across {len(league_ids)} mini leagues, GW1-{args.through}")

    actuals = {gw: calibrate.actuals(gw) for gw in events}

    mine = leaks_for(args.entry, events, positions, client, actuals)
    rows = [r for r in (leaks_for(i, events, positions, client, actuals) for i in sorted(ids)) if r]
    print(f"{len(rows)} rivals with a complete record\n")
    if not rows or not mine:
        print("not enough data")
        return 1

    def pct(values, mine_v, lower_is_better=True):
        if lower_is_better:
            return 100.0 * sum(1 for v in values if v < mine_v) / len(values)
        return 100.0 * sum(1 for v in values if v < mine_v) / len(values)

    print(f"{'metric':<12}{'mine':>7}{'field mean':>12}{'median':>8}{'sd':>7}"
          f"{'p90':>7}{'my pctile':>11}")
    print("-" * 64)
    for key, label in (("lineup", "lineup leak"), ("captain", "captain leak"),
                       ("combined", "combined")):
        vals = sorted(r[key] for r in rows)
        p90 = vals[min(len(vals) - 1, int(0.9 * len(vals)))]
        print(f"{label:<12}{mine[key]:>7}{st.mean(vals):>12.1f}{st.median(vals):>8.1f}"
              f"{st.pstdev(vals):>7.1f}{p90:>7}{pct(vals, mine[key]):>10.0f}th")

    totals = sorted(r["total"] for r in rows)
    tp = 100.0 * sum(1 for v in totals if v < mine["total"]) / len(totals)
    print(f"\n{'points':<12}{mine['total']:>7}{st.mean(totals):>12.1f}"
          f"{st.median(totals):>8.1f}{st.pstdev(totals):>7.1f}{'':>7}{tp:>10.0f}th")
    print("\nA leak percentile ABOVE 50 means leaking more than the field; BELOW 50\n"
          "means leaking less. Read the captaincy row before calling it a defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
