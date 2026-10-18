"""
Build tests/fixtures/scoring_cases.json.gz: real gameweeks as FPL scored them.

simulate.pick_xi decides what a squad scored - autosubs, formation legality,
captain and vice. Every decision-replay number rests on it, and until now it
was checked only against hand-written cases. This captures the answer from
FPL itself: for my team and every rival in my mini leagues, each finished
gameweek's picks, the autosubs FPL made, the players' live points and
minutes, and the points FPL awarded. test_replay_scoring.py replays them through pick_xi.

Public endpoints only. The repo is public, so nothing identifying is kept:
no entry ids, no names, and the cases are shuffled.

Run: python3 tests/fixtures/fetch_scoring_cases.py
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import rivals  # noqa: E402
from config import FPL_TEAM_ID  # noqa: E402

OUT = Path(__file__).parent / "scoring_cases.json.gz"
DEFAULT_ENTRY = 5413589


def main() -> int:
    client = rivals.PublicFPL(delay=0.2)
    entry = int(FPL_TEAM_ID or DEFAULT_ENTRY)
    boot = client.get("/bootstrap-static/")
    position = {e["id"]: e["element_type"] for e in boot["elements"]}
    done = [e["id"] for e in boot["events"] if e["finished"] and e["data_checked"]]

    entries = {entry}
    for league in rivals.mini_league_ids(client, entry):
        entries.update(rivals.rival_ids(client, league, entry))

    cases, live = [], {}
    for gw in done:
        stats = {e["id"]: e["stats"] for e in client.get(f"/event/{gw}/live/")["elements"]}
        used = set()
        for e in sorted(entries):
            try:
                p = client.get(f"/entry/{e}/event/{gw}/picks/")
            except Exception:  # noqa: BLE001 - a manager who joined late has no picks
                continue
            if not p or not p.get("picks"):
                continue
            cases.append({
                "event": gw,
                "active_chip": p.get("active_chip"),
                "picks": [{k: x[k] for k in ("element", "position", "multiplier",
                                             "is_captain", "is_vice_captain")}
                          for x in p["picks"]],
                "fpl_points": p["entry_history"]["points"],
                # The picks endpoint returns the lineup AFTER autosubs: the
                # player who came on sits in the starter's shirt. These pairs,
                # in FPL's order, are what undoes that.
                "automatic_subs": [[s["element_in"], s["element_out"]]
                                   for s in p.get("automatic_subs") or []],
            })
            used.update(x["element"] for x in p["picks"])
        live[str(gw)] = {str(i): {"position": position[i],
                                  "total_points": stats[i]["total_points"],
                                  "minutes": stats[i]["minutes"]} for i in sorted(used)}

    random.Random(0).shuffle(cases)
    payload = {"source": "fantasy.premierleague.com public picks + event live",
               "events": done, "cases": cases, "live": live}
    OUT.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
    print(f"{len(cases)} cases over GW{done[0]}-{done[-1]}, "
          f"{sum(bool(c['automatic_subs']) for c in cases)} with autosubs, "
          f"{sum(bool(c['active_chip']) for c in cases)} with a chip -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
