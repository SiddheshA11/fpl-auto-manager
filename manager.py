"""
FPL Auto Manager - weekly run.

Pipeline:
    1. authenticate
    2. load game state and priors
    3. score every player in expected points
    4. value the chips
    5. optimise transfers (or rebuild, under wildcard / free hit)
    6. submit lineup, captain and bench order

The research steps this used to run are gone. They scraped Premier Injuries,
Rotowire and Understat for injury flags and xG - data the FPL API now returns
directly as chance_of_playing_next_round and expected_goals_per_90 - and the
sites had begun blocking the scrapers anyway, so the config had already been
emptied of sources. Roughly 900 lines of code that could only degrade the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import chips
import optimizer as optimizer_mod
import priors
import xp_model as X
from config import FPL_TEAM_ID, OWNERSHIP_WEIGHT
from fpl_client import FPLClient
from optimizer import SquadOptimizer, format_squad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"fpl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("fpl_auto")

HORIZON = 5


def season_not_started(bootstrap: dict, event_id: int) -> bool:
    """
    True before the first deadline of the season, when transfers are free.

    FPL charges nothing for any number of transfers until the Gameweek 1
    deadline passes, which is the one week the entire squad can be rebuilt for
    nothing. `manager.py` used to read `(limit or 1) - made` off the transfers
    block, and FPL reports no numeric limit while transfers are unlimited - so
    the bot concluded it had exactly one free transfer during the only week it
    had unlimited ones, and would have made a single move and rolled the rest.

    Deliberately decided from the fixture calendar rather than from the
    transfers payload. The payload's shape in this state is precisely what has
    never been observed, so keying the decision on it would be guessing again;
    "the first gameweek of the season has not kicked off yet" is a fact the
    bootstrap states plainly and cannot be wrong about.
    """
    events = bootstrap.get("events") or []
    if not events:
        return False
    first = min(int(e["id"]) for e in events)
    if event_id != first:
        return False
    opening = next((e for e in events if int(e["id"]) == first), None)
    return bool(opening) and not opening.get("finished") and not opening.get("is_current")


def _picks_payload(sol, chip: str | None) -> list[dict]:
    """
    Build the picks list FPL expects: 1-11 starting, 12-15 benched in order,
    with the bench keeper first because he can only ever replace the keeper.
    """
    payload = []
    position = 1
    for _, p in sol.xi.iterrows():
        payload.append({
            "element": int(p["id"]),
            "position": position,
            "is_captain": int(p["id"]) == sol.captain,
            "is_vice_captain": int(p["id"]) == sol.vice_captain,
        })
        position += 1
    for _, p in sol.bench.iterrows():
        payload.append({
            "element": int(p["id"]),
            "position": position,
            "is_captain": False,
            "is_vice_captain": False,
        })
        position += 1
    return payload


def run_weekly_cycle(dry_run: bool = False, max_hits: int = 2) -> dict | None:
    logger.info("=" * 60)
    logger.info("FPL Auto Manager - weekly run (%s)", datetime.now(timezone.utc).isoformat())
    logger.info("Team ID: %s | dry run: %s", FPL_TEAM_ID, dry_run)
    logger.info("=" * 60)

    logger.info("STEP 1: authenticating")
    client = FPLClient()
    if not client.login():
        logger.critical("authentication failed; aborting")
        return None

    logger.info("STEP 2: loading game state")
    bootstrap = client.get_bootstrap()
    fixtures_df = client.get_fixtures()
    fixtures = fixtures_df.to_dict("records") if isinstance(fixtures_df, pd.DataFrame) else fixtures_df

    next_event = client.get_next_event()
    if not next_event:
        logger.warning("no upcoming gameweek; season may be over")
        return None
    event_id = int(next_event["id"])

    my_team = client.get_my_team()
    if not my_team:
        logger.critical("could not fetch current squad; aborting")
        return None

    bank = my_team["transfers"].get("bank", 0) / 10.0
    # Subtract transfers already made this gameweek. The workflow can fire more
    # than once per deadline (the scheduler's dedup file lives in /tmp on an
    # ephemeral runner, and there is a separate cron fallback), and a second run
    # that reads `limit` alone believes an already-spent transfer is still free
    # - then takes what it prices as a free move for an actual -4.
    _transfers = my_team["transfers"]
    free_transfers = max(0, (_transfers.get("limit") or 1) - (_transfers.get("made") or 0))
    preseason = season_not_started(bootstrap, event_id)
    if preseason:
        # Every transfer is free until the deadline, so the ceiling is simply
        # replacing all fifteen. Logged verbatim because this is the state whose
        # payload shape has never been captured.
        logger.info("pre-season: transfers are unlimited and free until the GW%d deadline", event_id)
        logger.info("transfers block as reported by FPL: %s", _transfers)
        free_transfers = 15
    elif free_transfers == 0:
        logger.info("no free transfers left this gameweek; any move would cost a hit")
    squad_ids = [int(p["element"]) for p in my_team["picks"]]
    selling = {int(p["element"]): p["selling_price"] / 10.0 for p in my_team["picks"]}
    logger.info("GW%d | bank £%.1fm | %d free transfer(s)", event_id, bank, free_transfers)

    teams_df = pd.DataFrame(bootstrap["teams"])
    logger.info(chips.describe_calendar(fixtures, event_id, teams_df))

    logger.info("STEP 3: scoring players")
    team_codes = {t["code"]: t["name"] for t in bootstrap["teams"]}
    prior_set = priors.build_priors(current_team_codes=team_codes)
    try:
        prior_set.validate()
    except RuntimeError as e:
        logger.critical("%s", e)
        return None
    model = X.XPModel(bootstrap, fixtures, prior_set, X.ModelConfig(horizon=HORIZON))
    events = X.next_events(bootstrap, HORIZON)
    scored = model.expected_points(events)

    # Unavailable players are dropped from the buy pool but kept if already
    # owned, so an injured player still gets valued (near zero) for selling.
    pool = scored[scored["status"].isin(["a", "d"]) | scored["id"].isin(squad_ids)].copy()
    logger.info("ownership tilt: %+.2f (%s)", OWNERSHIP_WEIGHT,
                "differentials" if OWNERSHIP_WEIGHT < 0
                else "template" if OWNERSHIP_WEIGHT > 0 else "pure expected points")
    opt = SquadOptimizer(pool, value_col="xp_horizon", captain_col="xp_next",
                         ownership_weight=OWNERSHIP_WEIGHT)

    logger.info("STEP 4: evaluating chips")
    budget = bank + sum(selling.values())

    current = opt.optimise_transfers(
        squad_ids, bank=bank, free_transfers=free_transfers, selling_prices=selling,
        max_hits=max_hits,
        free_transfer_value=0.0 if preseason else optimizer_mod.FREE_TRANSFER_VALUE,
    )

    # Wildcard is a horizon decision: it buys a squad you keep. Free hit is a
    # one-week squad that reverts, so it must be optimised on the single
    # gameweek it is played in - scoring a horizon-optimal squad on xp_next
    # understates the achievable XI badly on exactly the double gameweek where
    # a free hit is worth playing, so the chip would never trigger.
    wildcard_squad = opt.build_squad(budget=budget)
    free_hit_squad = SquadOptimizer(
        pool, value_col="xp_next", captain_col="xp_next", ownership_weight=OWNERSHIP_WEIGHT
    ).build_squad(budget=budget)

    # Baseline the wildcard against the post-transfer squad, not the current
    # one: the gain a free transfer would have captured anyway is not a reason
    # to spend a wildcard.
    wildcard_gain = wildcard_squad.squad_xp - current.squad_xp
    chip_engine = chips.ChipEngine(bootstrap, fixtures, scored)
    decision = chip_engine.evaluate(
        event_id,
        my_team,
        xi_ids=[int(i) for i in current.xi["id"]],
        bench_ids=[int(i) for i in current.bench["id"]],
        captain_id=current.captain,
        free_hit_xi_xp=float(free_hit_squad.xi["xp_next"].sum()),
        wildcard_gain=wildcard_gain,
    )
    logger.info("chip decision: %s (%s)", decision.chip or "none", decision.reason)

    logger.info("STEP 5: planning transfers")
    if decision.chip in ("wildcard", "freehit"):
        # Both give unlimited transfers, so the plan is simply the best squad -
        # but built on different objectives, since a wildcard squad is kept and
        # a free hit squad is discarded after one gameweek.
        plan = wildcard_squad if decision.chip == "wildcard" else free_hit_squad
        plan.transfers_in = sorted(set(int(i) for i in plan.squad["id"]) - set(squad_ids))
        plan.transfers_out = sorted(set(squad_ids) - set(int(i) for i in plan.squad["id"]))
        plan.hits = 0
    else:
        plan = current

    names = scored.set_index("id")["web_name"].to_dict()
    if plan.transfers_in:
        for tin, tout in zip(plan.transfers_in, plan.transfers_out):
            logger.info("  OUT %s -> IN %s", names.get(tout, tout), names.get(tin, tin))
        if plan.hits:
            logger.info("  taking %d hit(s) = -%d pts", plan.hits, plan.hits * 4)
    else:
        logger.info("  no transfer improves the squad; rolling the free transfer")

    if not dry_run and plan.transfers_in:
        ok = client.make_transfers(
            transfers_in=plan.transfers_in,
            transfers_out=plan.transfers_out,
            prices_in=[int(scored.loc[scored["id"] == i, "cost"].iloc[0] * 10) for i in plan.transfers_in],
            prices_out=[int(selling.get(i, 0) * 10) for i in plan.transfers_out],
            wildcard=decision.chip == "wildcard",
            free_hit=decision.chip == "freehit",
        )
        if ok is None:
            # The squad is still the old one, so the planned XI refers to
            # players we do not own and FPL would reject it - leaving last
            # week's lineup and captain in place, including any player who has
            # since been injured. Re-optimise the eleven over what we actually
            # hold instead.
            logger.error("transfer submission failed; re-optimising the lineup over the current squad")
            owned = scored[scored["id"].isin(squad_ids)].copy()
            try:
                plan = SquadOptimizer(
                    owned, value_col="xp_next", captain_col="xp_next",
                    ownership_weight=OWNERSHIP_WEIGHT,
                ).build_squad(budget=float(owned["cost"].sum()) + 0.1)
            except (RuntimeError, ValueError) as e:
                logger.critical("could not rebuild a lineup from the owned squad: %s", e)
                return None
    elif dry_run:
        logger.info("[dry run] not submitting transfers")

    logger.info("STEP 6: submitting lineup")
    lineup_chip = decision.chip if decision.chip in ("bboost", "3xc") else None
    picks = _picks_payload(plan, lineup_chip)

    if not dry_run:
        if client.set_lineup(picks, chip=lineup_chip) is not None:
            logger.info("lineup submitted")
        else:
            logger.error("lineup submission failed")
    else:
        logger.info("[dry run] not submitting lineup")

    logger.info("\n%s", format_squad(plan))
    logger.info("=" * 60)
    logger.info("run complete | GW%d | transfers %d | hits -%d | chip %s",
                event_id, len(plan.transfers_in), plan.hits * 4, decision.chip or "none")
    logger.info("=" * 60)

    return {
        "event_id": event_id,
        "transfers_in": plan.transfers_in,
        "transfers_out": plan.transfers_out,
        "hits": plan.hits,
        "chip": decision.chip,
        "captain": plan.captain,
        "xi_xp": plan.xi_xp,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FPL Auto Manager")
    ap.add_argument("--dry-run", action="store_true", help="compute everything, submit nothing")
    ap.add_argument("--max-hits", type=int, default=2, help="most hits the optimiser may take")
    args = ap.parse_args()

    result = run_weekly_cycle(dry_run=args.dry_run, max_hits=args.max_hits)
    if result is None:
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
