"""
A durable record of what the engine predicted, every gameweek.

The engine submitted five gameweeks before anyone asked why it was losing, and
answering that took archaeology through 30-day run artifacts - by which point
GW1 to GW3 had already expired. The only committed snapshot was from
pre-season, and there is no 2026-27 history on disk, so `backtest.py` cannot
score the live season either. The engine was unobservable in exactly the
situation where observability is the whole point.

So each submitting run writes one JSON file per gameweek: the squad it chose,
the expected points it assigned every player, who it captained, what it
transferred and why it thought that was right. Joining those against the
public `event/{gw}/live/` endpoint afterwards gives real in-season calibration
at any time, for any gameweek, without needing the run logs to still exist.

Written to a separate branch rather than main. The ledger is data the bot
appends on a schedule; mixing it into the code history would make every diff
noisy and every revert dangerous.

Nothing here may fail a run. The squad is already submitted by the time this
is called, and losing a prediction record is worth far less than the
submission it describes.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("fpl_auto")

LEDGER_BRANCH = "predictions"
LEDGER_DIR = "predictions"
# 2 added `pool`: the model's prediction for every scored player, not only the
# fifteen it chose. Additive, so a reader of schema 1 still works on 2.
SCHEMA_VERSION = 2

# Every scored player, as parallel columns keyed by `id`. The fifteen chosen
# players are a sample selected on xP, so their in-season r cannot tell range
# restriction from a model that has degraded; that needs the population the
# selection was made from. A top-100 cut would not do: on the pre-season
# snapshot it keeps an xP sd of 0.55 against 1.20 for the pool, and drops the
# near-zero players who are the other half of the gap. The whole pool is ~17KB.
POOL_COLUMNS = ("position", "xp_next", "p_start", "exp_minutes", "availability")


def build_record(*, event_id: int, plan, scored, captain: int, vice: int,
                 chip: str | None, transfers_in: list[int], transfers_out: list[int],
                 hits: int, tilt_source: str | None = None,
                 now: datetime | None = None) -> dict:
    """
    The record, as pure data so it can be tested without a network or a model.

    `scored` is the frame the optimiser ran on. `players` carries the squad's
    rows in full, to audit the decision; `pool` carries a few columns for every
    scored player, to calibrate the model on a population it did not select.
    """
    now = now or datetime.now(timezone.utc)
    ids = [int(i) for i in plan.squad["id"]]
    xi = {int(i) for i in plan.xi["id"]}

    by_id = scored.set_index("id")
    players = []
    for pos, pid in enumerate(ids):
        row = by_id.loc[pid] if pid in by_id.index else None
        rec = {
            "id": pid,
            "started": pid in xi,
            "is_captain": pid == int(captain),
            "is_vice_captain": pid == int(vice),
        }
        if row is not None:
            # position is FPL's element_type and stays an int; everything else
            # numeric is a float. Without the split it serialised as 2.0, which
            # every reader then has to coerce back.
            if "position" in by_id.columns:
                rec["position"] = int(row["position"])
            for col, key in (("web_name", "name"), ("team_name", "team")):
                if col in by_id.columns:
                    rec[key] = str(row[col])
            for col, key in (("cost", "cost"),
                             ("xp_next", "xp_next"), ("xp_horizon", "xp_horizon"),
                             ("p_start", "p_start"), ("availability", "availability"),
                             ("exp_minutes", "exp_minutes"), ("sd_next", "sd_next")):
                if col in by_id.columns:
                    rec[key] = float(row[col])
        players.append(rec)

    return {
        "schema": SCHEMA_VERSION,
        "event": int(event_id),
        "recorded_at": now.isoformat(),
        "captain": int(captain),
        "vice_captain": int(vice),
        "chip": chip,
        "hits": int(hits),
        "transfers_in": [int(i) for i in transfers_in],
        "transfers_out": [int(i) for i in transfers_out],
        "tilt_source": tilt_source,
        "xi_xp": float(plan.xi["xp_next"].sum()) if "xp_next" in plan.xi.columns else None,
        "players": players,
        "pool": _pool(scored),
    }


def _pool(scored) -> dict:
    """
    Columnar, sorted by id. NaN becomes null, because json.dumps would
    otherwise write a bare NaN that strict JSON readers reject.
    """
    import math

    frame = scored.sort_values("id")
    out: dict = {"id": [int(i) for i in frame["id"]]}
    for col in POOL_COLUMNS:
        if col not in frame.columns:
            continue
        if col == "position":
            out[col] = [int(v) for v in frame[col]]
            continue
        out[col] = [None if v is None or math.isnan(float(v)) else round(float(v), 4)
                    for v in frame[col]]
    return out


def record_submission(gh, **kwargs) -> bool:
    """
    Build and persist a record in one protected step.

    The build is inside the try on purpose. Called as
    `write(gh, build_record(...))` the build would be evaluated first, outside
    any protection, so one unexpected column could raise *after* the team was
    already submitted - turning an observability nicety into a failed run.
    """
    try:
        record = build_record(**kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not build the prediction record (%s: %s)", type(e).__name__, e)
        return False
    return write(gh, record)


def write(gh, record: dict) -> bool:
    """
    Persist one gameweek's record. Never raises.

    `gh` is a github_api.GitHubRepo, passed in rather than constructed so the
    caller owns credential handling and tests need no network.
    """
    if gh is None:
        logger.info("no GitHub credentials; not writing the prediction ledger")
        return False
    try:
        if not gh.ensure_branch(LEDGER_BRANCH):
            return False
        path = f"{LEDGER_DIR}/gw{int(record['event']):02d}.json"
        body = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ok = gh.put_file(path, body, LEDGER_BRANCH,
                         f"ledger: GW{record['event']} predictions")
        if ok:
            logger.info("recorded GW%s predictions to %s on %s",
                        record["event"], path, LEDGER_BRANCH)
        return ok
    except Exception as e:  # noqa: BLE001 - a ledger may not fail a submitted run
        logger.warning("could not write the prediction ledger (%s: %s)", type(e).__name__, e)
        return False


def selftest(gh=None) -> int:
    """
    Exercise the real write path against the real API, before a deadline needs it.

    The ledger's first production write would otherwise happen ninety seconds
    before a deadline, on a code path no test covers against the live API - and
    its failure mode is silent and indistinguishable from success. A GH_PAT
    provisioned with Actions and Secrets but not `Contents: write` returns 403
    on branch creation, `write()` logs one warning, the run exits 0 green, and
    `calibrate.py` reports "the ledger is empty" - which reads as "nothing has
    submitted yet" rather than "every write has 403ed for a season".

    Writes a probe record under a reserved event id so it cannot collide with a
    real gameweek, then reads it back.
    """
    import os

    import github_api

    if gh is None:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        token = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN", "")
        if not repo or not token:
            print("FAIL: GITHUB_REPOSITORY or GH_PAT is not set")
            return 1
        gh = github_api.GitHubRepo(repo, token)

    probe = {"schema": SCHEMA_VERSION, "event": 0, "players": [],
             "recorded_at": datetime.now(timezone.utc).isoformat(),
             "note": "selftest probe, not a gameweek"}

    if not gh.ensure_branch(LEDGER_BRANCH):
        print(f"FAIL: could not create or find the '{LEDGER_BRANCH}' branch. "
              "GH_PAT most likely lacks Contents: write.")
        return 1
    print(f"ok: branch '{LEDGER_BRANCH}' exists")

    if not write(gh, probe):
        print("FAIL: could not write the probe record. GH_PAT most likely lacks "
              "Contents: write.")
        return 1
    print(f"ok: wrote {LEDGER_DIR}/gw00.json")

    content, _sha = gh.get_file(f"{LEDGER_DIR}/gw00.json", LEDGER_BRANCH)
    if not content:
        print("FAIL: wrote the probe but could not read it back")
        return 1
    print("ok: read it back")

    # The update path is separate: it needs the existing sha, and the fake in
    # the unit tests always reports no sha, so this is the only exercise it gets.
    probe["note"] = "selftest probe, rewritten"
    if not write(gh, probe):
        print("FAIL: could not OVERWRITE the probe. The sha-on-update path is "
              "broken, so every gameweek after the first would 409.")
        return 1
    print("ok: overwrote it (sha-on-update path works)")
    print("\nPASS: the prediction ledger can write. GW7 will record.")
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(selftest() if "--selftest" in sys.argv else
                     (print("usage: python3 ledger.py --selftest") or 2))
