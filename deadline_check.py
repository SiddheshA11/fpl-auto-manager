"""
Pre-deadline safety check.

Runs a couple of hours before the deadline, after the press conferences that
usually produce late injury news. It does not make transfers - that decision
was already taken and paid for earlier in the week. It only re-reads
availability and repairs the lineup: pull anyone who has been flagged since,
promote a fit replacement from the bench, and move the armband off a player
who is no longer starting.

Losing a captain to a late fitness doubt is the single most expensive
unforced error in FPL, and it is entirely avoidable.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import priors
import rivals
import xp_model as X
from config import FPL_TEAM_ID
from fpl_client import FPLClient
from optimizer import SquadOptimizer, format_squad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"fpl_deadline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("fpl_auto")

# A player this unlikely to feature should not be starting, whatever his
# expected points look like on paper.
BENCH_BELOW_AVAILABILITY = 0.5


def run_deadline_check(dry_run: bool = False) -> dict | None:
    logger.info("=" * 60)
    logger.info("pre-deadline check (%s)", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    client = FPLClient()
    if not client.login():
        logger.critical("authentication failed; aborting")
        return None

    bootstrap = client.get_bootstrap()
    fixtures_df = client.get_fixtures()
    fixtures = fixtures_df.to_dict("records") if isinstance(fixtures_df, pd.DataFrame) else fixtures_df

    next_event = client.get_next_event()
    if not next_event:
        logger.warning("no upcoming gameweek")
        return None
    event_id = int(next_event["id"])

    my_team = client.get_my_team()
    if not my_team:
        logger.critical("could not fetch squad; aborting")
        return None
    squad_ids = [int(p["element"]) for p in my_team["picks"]]

    team_codes = {t["code"]: t["name"] for t in bootstrap["teams"]}
    prior_set = priors.build_priors(current_team_codes=team_codes)
    try:
        prior_set.validate()
    except RuntimeError as e:
        logger.critical("%s", e)
        return None
    model = X.XPModel(bootstrap, fixtures, prior_set, X.ModelConfig(horizon=1))
    scored = model.expected_points([event_id])

    squad = scored[scored["id"].isin(squad_ids)].copy()
    if len(squad) < 15:
        logger.warning("only %d of 15 squad players found in the model frame", len(squad))

    flagged = squad[squad["availability"] < BENCH_BELOW_AVAILABILITY]
    if len(flagged):
        logger.warning("late doubts in the squad:")
        for _, p in flagged.iterrows():
            logger.warning("  %s (%s) availability %.0f%% - status '%s'",
                           p["web_name"], p["team_name"], p["availability"] * 100, p["status"])
    else:
        logger.info("no new availability concerns in the squad")

    # Re-pick the XI from the squad we already own.
    #
    # Two things here must match the weekly run exactly, because this job's
    # entire purpose is to make a *safety* adjustment to what that run
    # submitted. Any difference in objective shows up as an unexplained lineup
    # change hours before the deadline, indistinguishable from a real response
    # to late news:
    #
    #   - the ownership tilt, which was omitted here and so ran at 0.0 while
    #     the weekly run used config.OWNERSHIP_WEIGHT. Latent at the current
    #     -0.3, but FPL_OWNERSHIP_WEIGHT is an environment variable and at
    #     |w| >= 0.5 the two disagree on the eleven for no reason at all.
    #   - optimise_lineup rather than build_squad. The squad is fixed, so this
    #     is an ordering problem; build_squad re-solves a selection problem
    #     against a budget equal to the squad's own cost, which is feasible
    #     only by a hair and reports "infeasible" when it is not.
    tilt = rivals.tilt_inputs(squad, FPL_TEAM_ID, event_id - 1)
    squad = tilt.scored
    opt = SquadOptimizer(squad, value_col="xp_next", captain_col="xp_next",
                         ownership_weight=tilt.ownership_weight,
                         ownership_col=tilt.ownership_col,
                         captain_ownership_weight=tilt.captain_weight)
    try:
        sol = opt.optimise_lineup()
    except (RuntimeError, ValueError) as e:
        logger.error("could not re-optimise the lineup: %s", e)
        return None

    picks = []
    position = 1
    for _, p in sol.xi.iterrows():
        picks.append({
            "element": int(p["id"]),
            "position": position,
            "is_captain": int(p["id"]) == sol.captain,
            "is_vice_captain": int(p["id"]) == sol.vice_captain,
        })
        position += 1
    for _, p in sol.bench.iterrows():
        picks.append({
            "element": int(p["id"]), "position": position,
            "is_captain": False, "is_vice_captain": False,
        })
        position += 1

    previous_xi = {int(p["element"]) for p in my_team["picks"] if p["position"] <= 11}
    new_xi = set(sol.xi["id"].astype(int))
    changed = previous_xi != new_xi

    old_captain = next((int(p["element"]) for p in my_team["picks"] if p.get("is_captain")), None)
    captain_changed = old_captain != sol.captain

    if changed or captain_changed:
        names = scored.set_index("id")["web_name"].to_dict()
        if changed:
            logger.info("lineup changes: in %s | out %s",
                        [names.get(i, i) for i in new_xi - previous_xi],
                        [names.get(i, i) for i in previous_xi - new_xi])
        if captain_changed:
            logger.info("captain moved: %s -> %s", names.get(old_captain, old_captain), names.get(sol.captain))
    else:
        logger.info("lineup unchanged; nothing to submit")

    logger.info("\n%s", format_squad(sol))

    if not dry_run and (changed or captain_changed):
        # No chip is passed: chip decisions belong to the weekly run, and
        # re-sending one here would either be rejected or double-apply it.
        if client.set_lineup(picks, chip=None) is not None:
            logger.info("corrected lineup submitted")
        else:
            logger.error("lineup submission failed")
    elif dry_run:
        logger.info("[dry run] not submitting")

    return {
        "event_id": event_id,
        "flagged": [str(n) for n in flagged["web_name"].tolist()],
        "lineup_changed": bool(changed),
        "captain_changed": bool(captain_changed),
        "captain": sol.captain,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FPL pre-deadline safety check")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return 0 if run_deadline_check(dry_run=args.dry_run) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
