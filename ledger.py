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
SCHEMA_VERSION = 1


def build_record(*, event_id: int, plan, scored, captain: int, vice: int,
                 chip: str | None, transfers_in: list[int], transfers_out: list[int],
                 hits: int, tilt_source: str | None = None,
                 now: datetime | None = None) -> dict:
    """
    The record, as pure data so it can be tested without a network or a model.

    `scored` is the frame the optimiser ran on; only the squad's rows are kept,
    because the point is to audit the decision, not to archive 600 players a
    week.
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
            for col, key in (("web_name", "name"), ("team_name", "team"),
                             ("position", "position"), ("cost", "cost"),
                             ("xp_next", "xp_next"), ("xp_horizon", "xp_horizon"),
                             ("p_start", "p_start"), ("availability", "availability"),
                             ("exp_minutes", "exp_minutes"), ("sd_next", "sd_next")):
                if col in by_id.columns:
                    v = row[col]
                    rec[key] = float(v) if hasattr(v, "__float__") and not isinstance(v, str) else str(v)
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
    }


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
