"""
The deadline watchdog.

It answers one question on its own schedule: *is the next deadline close, and
has anything actually submitted a team for it?* If the answer is no, it says so
on Telegram and dispatches the weekly run itself.

Three properties matter more than anything it does:

1. **It watches the outcome, not the cron.** A checker that asks "did the
   scheduled job fire" dies the same silent death `deadline_check.yml` did -
   that one was `workflow_dispatch:` only, triggered by a `scheduler.yml` that
   has been `disabled_inactivity` since April, and nothing noticed for months.
   This asks whether a squad was submitted for the gameweek in front of us.
   That is true or false regardless of which mechanism was supposed to do it.

2. **It runs on its own schedule and depends on almost nothing.** Public
   endpoints and the GitHub API. No FPL credentials, no priors, no pandas. It
   must survive the failures it exists to report.

3. **It watches itself.** Every run stamps its own heartbeat, and the next run
   reports how late it was. GitHub drops scheduled runs; a watchdog that is
   itself dropped and says nothing is worse than no watchdog, because it is
   trusted. It cannot detect its own death in real time - nothing inside the
   same system can - so it also sends a periodic all-clear, and the absence of
   those is the signal a human can act on.

The marker it reads, `FPL_LAST_SUBMITTED_GW`, is a repository Actions variable
written by `manager.py` after a real submission. The same variable is what
makes the hourly weekly-run schedule idempotent.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import deadline_state as ds
import notify
from github_api import HEARTBEAT_VARIABLE, MARKER_VARIABLE, GitHubRepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fpl_auto")

WORKFLOW_TO_TRIGGER = "weekly_run.yml"

# How close the deadline has to be before an unsubmitted gameweek is an
# emergency rather than a normal state of affairs. The weekly run's own acting
# window closes at DEADLINE_WINDOW_MIN (2h), and it acts at a lead of 5.0-5.5h
# with three retries behind it. So if nothing has submitted with 4h left, every
# scheduled attempt has already failed and this is the last line.
ALERT_LEAD_HOURS = 4.0

# The watchdog's own cadence. Used only to judge its own lateness.
EXPECTED_INTERVAL_HOURS = 3.0
LATE_FACTOR = 2.5

# Send an unprompted all-clear at most this often, so a silent watchdog is
# distinguishable from a quiet week.
HEARTBEAT_EVERY_HOURS = 72.0


def _parse_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def assess(events: list[dict], last_submitted_gw: int | None,
           now: datetime | None = None) -> dict:
    """
    The judgement, as pure data. No network, no side effects, so the decision
    can be tested exhaustively without mocking a transport.
    """
    now = now or datetime.now(timezone.utc)
    event = ds.next_event(events, now)

    if event is None:
        return {"state": "no-gameweek", "event_id": None, "lead_hours": None,
                "alert": False, "trigger": False,
                "detail": "no upcoming gameweek; season over or calendar unpublished"}

    event_id = int(event["id"])
    lead = ds.hours_to_deadline(event, now)
    submitted = last_submitted_gw is not None and last_submitted_gw == event_id

    if submitted:
        return {"state": "submitted", "event_id": event_id, "lead_hours": lead,
                "alert": False, "trigger": False,
                "detail": f"GW{event_id} already submitted"}

    if lead is None:
        return {"state": "unreadable", "event_id": event_id, "lead_hours": None,
                "alert": True, "trigger": False,
                "detail": f"GW{event_id} has no readable deadline_time; "
                          "the payload shape may have changed"}

    if lead <= ALERT_LEAD_HOURS:
        return {"state": "overdue", "event_id": event_id, "lead_hours": lead,
                "alert": True, "trigger": True,
                "detail": f"GW{event_id} deadline is {lead:.1f}h away and nothing "
                          "has submitted a team for it"}

    return {"state": "waiting", "event_id": event_id, "lead_hours": lead,
            "alert": False, "trigger": False,
            "detail": f"GW{event_id} deadline is {lead:.1f}h away; "
                      "not yet due to be submitted"}


def check_own_lateness(last_heartbeat: datetime | None, now: datetime) -> float | None:
    """Hours late, or None if the watchdog is running roughly on time."""
    if last_heartbeat is None:
        return None
    gap = (now - last_heartbeat).total_seconds() / 3600.0
    if gap > EXPECTED_INTERVAL_HOURS * LATE_FACTOR:
        return gap
    return None


def _format_alert(verdict: dict, repo: str, triggered: bool | None) -> str:
    lines = ["*FPL deadline watchdog*", "", verdict["detail"]]
    if verdict["state"] == "overdue":
        if triggered is True:
            lines.append("\nDispatched the weekly run automatically. "
                         "Check it landed before the deadline.")
        elif triggered is False:
            lines.append("\n*Could not dispatch the weekly run.* Run it by hand now.")
    lines.append(f"\nhttps://github.com/{repo}/actions/workflows/{WORKFLOW_TO_TRIGGER}")
    return "\n".join(lines)


def run(dry_run: bool = False, now: datetime | None = None,
        gh: GitHubRepo | None = None, notifier=None,
        fetcher=ds._fetch) -> dict:
    now = now or datetime.now(timezone.utc)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN", "")

    notifier = notifier or notify.from_env()
    if gh is None:
        if not repo or not token:
            # Without the API there is no marker to read, so any verdict would
            # be a guess. Say so loudly rather than report a false all-clear.
            msg = "watchdog cannot reach the GitHub API (GITHUB_REPOSITORY or GH_PAT unset)"
            logger.error(msg)
            notifier.send(f"*FPL deadline watchdog*\n\n{msg}")
            return {"state": "misconfigured", "alert": True, "trigger": False, "detail": msg}
        gh = GitHubRepo(repo, token)

    try:
        events = ds.fetch_events(fetcher)
    except Exception as e:  # noqa: BLE001
        msg = f"watchdog could not read the FPL calendar: {type(e).__name__}: {e}"
        logger.error(msg)
        notifier.send(f"*FPL deadline watchdog*\n\n{msg}")
        return {"state": "fpl-unreachable", "alert": True, "trigger": False, "detail": msg}

    last_submitted = _parse_int(gh.get_variable(MARKER_VARIABLE))
    verdict = assess(events, last_submitted, now)
    logger.info("watchdog: %s (%s)", verdict["state"], verdict["detail"])

    last_heartbeat = _parse_ts(gh.get_variable(HEARTBEAT_VARIABLE))
    late_by = check_own_lateness(last_heartbeat, now)
    if late_by is not None:
        logger.warning("watchdog ran %.1fh after its previous run (expected every %.0fh)",
                       late_by, EXPECTED_INTERVAL_HOURS)

    triggered = None
    if verdict["trigger"] and not dry_run:
        triggered = gh.dispatch_workflow(WORKFLOW_TO_TRIGGER, inputs={"dry_run": "false"})
    elif verdict["trigger"] and dry_run:
        logger.info("[dry run] would dispatch %s", WORKFLOW_TO_TRIGGER)

    messages = []
    if verdict["alert"]:
        messages.append(_format_alert(verdict, repo, triggered))
    if late_by is not None:
        messages.append(f"*FPL deadline watchdog*\n\nThe watchdog itself ran {late_by:.1f}h "
                        f"after its previous run; it is scheduled every {EXPECTED_INTERVAL_HOURS:.0f}h. "
                        "GitHub dropped or delayed its schedule.")

    # The all-clear. Absence of these is the only externally visible sign that
    # the watchdog has stopped entirely, so it is sent even on a quiet week.
    due_heartbeat = (last_heartbeat is None or
                     (now - last_heartbeat).total_seconds() / 3600.0 >= HEARTBEAT_EVERY_HOURS)
    if due_heartbeat and not verdict["alert"]:
        messages.append(f"*FPL deadline watchdog*\n\nAll clear. {verdict['detail']}.")

    if not dry_run:
        for m in messages:
            notifier.send(m)
        # Stamped last and unconditionally: a heartbeat that only survives the
        # happy path would make a broken watchdog look like a dead one.
        gh.set_variable(HEARTBEAT_VARIABLE, now.isoformat())
    else:
        for m in messages:
            logger.info("[dry run] would send:\n%s", m)

    verdict["triggered"] = triggered
    verdict["late_by_hours"] = late_by
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description="FPL deadline watchdog")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log, but send nothing and dispatch nothing")
    args = ap.parse_args()
    verdict = run(dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in verdict.items()}, indent=2, default=str))
    # An alerting state is not a failed run: the watchdog did its job. Only a
    # watchdog that could not reach what it needs should go red.
    return 1 if verdict["state"] in ("misconfigured", "fpl-unreachable") else 0


if __name__ == "__main__":
    sys.exit(main())
