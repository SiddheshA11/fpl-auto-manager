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
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import chips
import deadline_state
import github_api
import notify
import optimizer as optimizer_mod
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


def _reoptimise_lineup(plan, scored, tilt: pd.DataFrame):
    """
    Re-pick the XI, captain and bench order over the squad in `plan`.

    Returns the plan unchanged if anything goes wrong. A worse-ordered lineup
    is a fraction of a point; no lineup at all is the whole gameweek, and this
    runs unattended minutes before a deadline.
    """
    squad_ids = [int(i) for i in plan.squad["id"]]
    owned = scored[scored["id"].isin(squad_ids)].copy()
    if len(owned) != len(squad_ids):
        logger.warning("could not score every owned player; keeping the horizon lineup")
        return plan

    try:
        lineup = SquadOptimizer(
            owned, value_col="xp_next", captain_col="xp_next",
            ownership_weight=tilt.ownership_weight, ownership_col=tilt.ownership_col,
            captain_ownership_weight=tilt.captain_weight,
        ).optimise_lineup()
    except (RuntimeError, ValueError) as e:
        logger.warning("lineup re-optimisation failed (%s); keeping the horizon lineup", e)
        return plan

    before = float(plan.xi["xp_next"].sum()) if "xp_next" in plan.xi.columns else None
    after = float(lineup.xi["xp_next"].sum())
    if before is not None:
        if after < before - 1e-6:
            # Cannot happen for a correct solve over the same fifteen, so if it
            # does the solve is wrong rather than merely unlucky.
            logger.warning(
                "lineup re-optimisation scored worse (%.2f vs %.2f); keeping the horizon lineup",
                after, before,
            )
            return plan
        logger.info("lineup re-optimised on GW xP: %.2f -> %.2f (%+.2f)", before, after, after - before)

    plan.xi = lineup.xi
    plan.bench = lineup.bench
    plan.captain = lineup.captain
    plan.vice_captain = lineup.vice_captain
    plan.xi_xp = float(lineup.xi["xp_next"].sum())
    return plan


def _pair_by_position(transfers_in: list[int], transfers_out: list[int],
                      scored: pd.DataFrame) -> tuple[list[int], list[int]]:
    """
    Order the two lists so that each pair is a like-for-like swap.

    FPL validates every transfer pair individually:

        {"non_field_errors": [{"message": "Element in and element out must be
          of the same type", "code": "transfer_element_type_mismatch"}]}

    and rejects the whole POST if any pair fails. Both lists were previously
    sorted by element id and zipped, which pairs players in an order that has
    nothing to do with position - a real 12-transfer submission had 8 of its 12
    pairs rejected. A single transfer happens to work whenever the squad shape
    is unchanged, which is why this survived every earlier test.

    The position multisets on both sides always match, because squad
    composition is fixed at 2/5/5/3, so a valid pairing always exists.
    """
    position = scored.set_index("id")["position"].to_dict()

    by_position_in: dict[int, list[int]] = {}
    by_position_out: dict[int, list[int]] = {}
    for pid in transfers_in:
        by_position_in.setdefault(int(position.get(pid, -1)), []).append(pid)
    for pid in transfers_out:
        by_position_out.setdefault(int(position.get(pid, -1)), []).append(pid)

    if sorted(by_position_in) != sorted(by_position_out) or any(
        len(by_position_in[k]) != len(by_position_out.get(k, [])) for k in by_position_in
    ):
        # Cannot happen for a squad the optimiser built, since it constrains
        # positions exactly. Surfaced rather than swallowed: submitting an
        # unpairable set guarantees a 400 at the deadline.
        logger.error(
            "transfer positions do not match: in=%s out=%s; submitting unpaired",
            {k: len(v) for k, v in by_position_in.items()},
            {k: len(v) for k, v in by_position_out.items()},
        )
        return transfers_in, transfers_out

    paired_in: list[int] = []
    paired_out: list[int] = []
    for pos in sorted(by_position_in):
        paired_in.extend(by_position_in[pos])
        paired_out.extend(by_position_out[pos])
    return paired_in, paired_out


# The acting window and the calendar arithmetic live in deadline_state, so the
# hourly gate job can evaluate them without importing pandas, priors or this
# module. Re-exported here because callers and tests reference manager.*, and
# because this run re-checks the window itself rather than trusting the gate.
DEADLINE_WINDOW_MIN = deadline_state.DEADLINE_WINDOW_MIN
DEADLINE_WINDOW_MAX = deadline_state.DEADLINE_WINDOW_MAX


def hours_to_deadline(event: dict, now: datetime | None = None) -> float | None:
    """Hours from `now` until this gameweek's deadline, or None if unknown."""
    return deadline_state.hours_to_deadline(event, now)


def _github() -> github_api.GitHubRepo | None:
    """The API client, or None when this is not running in CI."""
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN", "")
    if not repo or not token:
        return None
    return github_api.GitHubRepo(repo, token)


def _marker_gameweek() -> int | None:
    """The gameweek last submitted for, per the repo variable."""
    gh = _github()
    if gh is None:
        # A local run has no marker and must not be blocked by its absence.
        return None
    try:
        return int(str(gh.get_variable(github_api.MARKER_VARIABLE)).strip())
    except (TypeError, ValueError):
        return None


def _record_submission(event_id: int) -> None:
    """
    Record that this gameweek is done. Failure to record is logged, never
    raised: the squad is already in, and losing the marker costs at worst a
    redundant run and a spurious watchdog alert - both far cheaper than
    failing a run that already succeeded.
    """
    gh = _github()
    if gh is None:
        logger.info("no GitHub credentials; not recording the submission marker")
        return
    if gh.set_variable(github_api.MARKER_VARIABLE, str(event_id)):
        logger.info("recorded GW%d as submitted", event_id)
        return

    # This is worse than it looks and must not pass as a warning in a log
    # nobody reads. The marker is the only thing that tells the watchdog this
    # gameweek is handled. Unset, the watchdog will conclude nothing was
    # submitted and dispatch the weekly run again a few hours from now - and
    # that run authenticates, finds the free transfers already spent, and may
    # take a -4 to do something with them. The usual cause is a GH_PAT without
    # permission to write repository variables.
    logger.error("could not record the GW%d submission marker. The team IS submitted. "
                 "The watchdog will not know that and may resubmit; check GH_PAT can "
                 "write repository variables.", event_id)
    notify.from_env().send(
        f"*FPL weekly run*\n\nGW{event_id} was submitted, but the "
        f"`{github_api.MARKER_VARIABLE}` marker could not be written. The watchdog "
        "cannot see that this gameweek is done and may dispatch a second run, "
        "which could take a hit. Check GH_PAT can write repository variables."
    )


def run_weekly_cycle(dry_run: bool = False, respect_window: bool = False, max_hits: int = 2) -> dict | None:
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

    if respect_window:
        # The marker makes the hourly schedule idempotent. Four ticks qualify
        # per gameweek so that a dropped or delayed run has three retries
        # behind it; without this check those retries would each submit again.
        # A second run is not harmless: it re-optimises with the free transfers
        # already spent, so it can only improve the squad by taking a hit.
        already = _marker_gameweek()
        if already is not None and already == event_id:
            logger.info("GW%d has already been submitted; nothing to do", event_id)
            return {"status": "already-submitted", "event_id": event_id}

        lead = hours_to_deadline(next_event)
        if lead is None:
            logger.warning("GW%d has no deadline_time; proceeding rather than skipping", event_id)
        elif not (DEADLINE_WINDOW_MIN <= lead < DEADLINE_WINDOW_MAX):
            logger.info(
                "GW%d deadline is %.1fh away, outside the %g-%gh window; nothing to do",
                event_id, lead, DEADLINE_WINDOW_MIN, DEADLINE_WINDOW_MAX,
            )
            # A status dict, not None. None means "this run broke" and exits 1;
            # deciding correctly that today is not the day is a healthy no-op.
            # Conflating the two marked six runs in seven as failures and made
            # "was there a successful run before the deadline" unanswerable,
            # which is the exact question the watchdog has to answer.
            return {"status": "skipped", "event_id": event_id, "lead_hours": lead}
        else:
            logger.info("GW%d deadline is %.1fh away; proceeding", event_id, lead)

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
    # Recent minutes are the most valuable input the model takes: folding them
    # into the start rate lifts points R2 from 0.269 to 0.319 in backtest,
    # against 0.004 for the entire scoring apparatus once minutes are known.
    events_done = sum(1 for e in bootstrap.get("events", []) if e.get("finished"))
    try:
        recent_minutes = client.get_recent_minutes(events_done)
    except Exception as e:                      # noqa: BLE001 - never fatal
        logger.warning("could not fetch recent minutes (%s); using season rates alone", e)
        recent_minutes = {}

    model = X.XPModel(bootstrap, fixtures, prior_set, X.ModelConfig(horizon=HORIZON),
                      recent_minutes=recent_minutes)
    # Score further ahead than the squad objective looks. The squad is chosen
    # on HORIZON gameweeks, but the chip planner needs to see a double or blank
    # further out to decide whether this week is the right one to spend a chip
    # on - and PLANNING_HORIZON was claiming ten gameweeks of visibility while
    # only five were ever scored, so half its horizon was empty.
    events = X.next_events(bootstrap, max(HORIZON, chips.PLANNING_HORIZON))
    scored = model.expected_points(events)

    # Unavailable players are dropped from the buy pool but kept if already
    # owned, so an injured player still gets valued (near zero) for selling.
    # Whose ownership the tilt aims at. One helper decides it for the weekly
    # run, the pre-deadline check and the preview tool alike; see rivals.py for
    # why that is not merely tidiness. Reads public endpoints and falls back to
    # the template, so it cannot fail the run.
    tilt = rivals.tilt_inputs(scored, FPL_TEAM_ID, event_id - 1)
    scored = tilt.scored
    pool = scored[scored["status"].isin(["a", "d"]) | scored["id"].isin(squad_ids)].copy()
    opt = SquadOptimizer(pool, value_col="xp_horizon", captain_col="xp_next",
                         ownership_weight=tilt.ownership_weight,
                         ownership_col=tilt.ownership_col,
                         captain_ownership_weight=tilt.captain_weight)

    logger.info("STEP 4: evaluating chips")
    budget = bank + sum(selling.values())

    current = opt.optimise_transfers(
        squad_ids, bank=bank, free_transfers=free_transfers, selling_prices=selling,
        max_hits=max_hits,
        free_transfer_value=0.0 if preseason else optimizer_mod.FREE_TRANSFER_VALUE,
        # The value column sums HORIZON gameweeks under decay, so the -4 has to
        # be expressed on the same scale or hits look ~3.6x cheaper than they are.
        horizon_weight=sum(X.ModelConfig(horizon=HORIZON).horizon_decay**i for i in range(HORIZON)),
    )

    # Wildcard is a horizon decision: it buys a squad you keep. Free hit is a
    # one-week squad that reverts, so it must be optimised on the single
    # gameweek it is played in - scoring a horizon-optimal squad on xp_next
    # understates the achievable XI badly on exactly the double gameweek where
    # a free hit is worth playing, so the chip would never trigger.
    wildcard_squad = opt.build_squad(budget=budget)
    free_hit_squad = SquadOptimizer(
        pool, value_col="xp_next", captain_col="xp_next",
        ownership_weight=tilt.ownership_weight, ownership_col=tilt.ownership_col,
        captain_ownership_weight=tilt.captain_weight,
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
        # Lets the planner value bench boost and triple captain in every
        # gameweek of the window, not just this one, so a double gameweek two
        # weeks out beats an ordinary week now.
        squad_ids=[int(i) for i in current.squad["id"]],
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
        plan.transfers_in, plan.transfers_out = _pair_by_position(
            plan.transfers_in, plan.transfers_out, scored)
        for tin, tout in zip(plan.transfers_in, plan.transfers_out):
            logger.info("  OUT %s -> IN %s", names.get(tout, tout), names.get(tin, tin))
        if plan.hits:
            logger.info("  taking %d hit(s) = -%d pts", plan.hits, plan.hits * 4)
    else:
        logger.info("  no transfer improves the squad; rolling the free transfer")

    if not dry_run and plan.transfers_in:
        # make_transfers raises on a pairing FPL would reject. That guard exists
        # to fail loudly at the point of the mistake - but letting it escape
        # here skips STEP 6 entirely and costs the whole gameweek, which is
        # strictly worse than the 400 it was written to pre-empt. Funnelled into
        # the same recovery path as a failed submission: keep the squad, rebuild
        # a legal lineup over it, and still submit that.
        try:
            ok = client.make_transfers(
                transfers_in=plan.transfers_in,
                transfers_out=plan.transfers_out,
                prices_in=[int(scored.loc[scored["id"] == i, "cost"].iloc[0] * 10) for i in plan.transfers_in],
                prices_out=[int(selling.get(i, 0) * 10) for i in plan.transfers_out],
                wildcard=decision.chip == "wildcard",
                free_hit=decision.chip == "freehit",
                positions=scored.set_index("id")["position"].astype(int).to_dict(),
            )
        except ValueError as e:
            logger.error("transfers rejected before submission: %s", e)
            ok = None
        if ok is None:
            # The squad is still the old one, so the planned XI refers to
            # players we do not own and FPL would reject it - leaving last
            # week's lineup and captain in place, including any player who has
            # since been injured. Re-optimise the eleven over what we actually
            # hold instead.
            logger.error("transfer submission failed; re-optimising the lineup over the current squad")
            owned = scored[scored["id"].isin(squad_ids)].copy()
            # optimise_lineup, not build_squad. The squad here is fixed - these
            # are the fifteen we hold - so the job is to order them, and
            # optimise_lineup says exactly that: it asserts the frame IS the
            # owned fifteen and needs no budget. build_squad instead re-solved a
            # selection problem against a budget equal to the squad's own cost,
            # which is feasible only by a hair and returned "problem is
            # infeasible" the first time this path was exercised - turning a
            # failed transfer into a lost gameweek.
            try:
                plan = SquadOptimizer(
                    owned, value_col="xp_next", captain_col="xp_next",
                    ownership_weight=tilt.ownership_weight,
                    ownership_col=tilt.ownership_col,
                    captain_ownership_weight=tilt.captain_weight,
                ).optimise_lineup()
            except (RuntimeError, ValueError) as e:
                logger.critical("could not rebuild a lineup from the owned squad: %s", e)
                return None
    elif dry_run:
        logger.info("[dry run] not submitting transfers")

    logger.info("STEP 6: submitting lineup")

    # Which fifteen to own is a horizon decision; which eleven to start is a
    # decision about this Saturday. The squad solve answers both with the
    # horizon column, which benches players who are worth more *this* gameweek
    # than the starters ahead of them. Re-solve the eleven over the squad we
    # now hold, scored on the coming gameweek alone.
    plan = _reoptimise_lineup(plan, scored, tilt)

    lineup_chip = decision.chip if decision.chip in ("bboost", "3xc") else None
    picks = _picks_payload(plan, lineup_chip)

    lineup_ok = True
    if not dry_run:
        if client.set_lineup(picks, chip=lineup_chip) is not None:
            logger.info("lineup submitted")
            # Only now is the gameweek genuinely done. Recording the marker on
            # a failed submission would tell the watchdog to stand down over a
            # team that never went in - the exact silence this all exists to
            # remove. Leaving it unset makes the next hourly tick retry, and
            # the watchdog alert if those run out.
            _record_submission(event_id)
        else:
            lineup_ok = False
            logger.error("lineup submission failed")
    else:
        logger.info("[dry run] not submitting lineup")

    logger.info("\n%s", format_squad(plan))
    logger.info("=" * 60)
    logger.info("run complete | GW%d | transfers %d | hits -%d | chip %s",
                event_id, len(plan.transfers_in), plan.hits * 4, decision.chip or "none")
    logger.info("=" * 60)

    return {
        "status": "dry-run" if dry_run else ("submitted" if lineup_ok else "lineup-failed"),
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
    # Opt-in, so every existing caller and test keeps working unchanged and
    # only the scheduled run enforces it.
    ap.add_argument("--respect-window", action="store_true",
                    help="skip unless the deadline is inside the daily cron's window")
    args = ap.parse_args()

    result = run_weekly_cycle(dry_run=args.dry_run, max_hits=args.max_hits,
                              respect_window=args.respect_window)
    if result is None:
        return 1
    print(json.dumps(result, indent=2, default=str))
    # A lineup that did not go in is a failed run, even though everything
    # before it worked. It previously logged an error and exited 0.
    return 1 if result.get("status") == "lineup-failed" else 0


if __name__ == "__main__":
    sys.exit(main())
