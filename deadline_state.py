"""
Where we are relative to the next deadline, and whether anything has acted on it.

Deliberately stdlib-only and free of pandas, priors and credentials. Three
callers need this and two of them must stay cheap:

  - the gate job on the weekly run, which runs hourly and has to decide in a
    few seconds, before any dependency install, whether the expensive job is
    worth starting. Crucially it decides *before authentication*, because the
    FPL refresh token is single-use and rotates: an hourly cron that logs in
    to discover it has nothing to do would burn 24 rotations a day.
  - the watchdog, which must keep working precisely when the rest is broken,
    so it depends on as little of the rest as possible.
  - manager.py, which re-checks the same window itself rather than trusting
    the gate.

`bootstrap-static` is public and needs no auth.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger("fpl_auto")

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
REQUEST_TIMEOUT = 30

# The band in which the weekly run is allowed to act.
#
# These numbers are measured, not argued. GitHub throttles this repo's hourly
# cron to ~5.5 arrivals a day with a median gap of 4.24h and a maximum of
# 9.57h, so a narrow band is simply not hit often enough.
#
# GW6 of 2026-27 is the proof. Three runs arrived that day: 8.51h out (too
# early for the old band), 1.94h out, and after the deadline. The middle one
# was 3.6 MINUTES short of the old 2.0h floor, so it skipped, and the deadline
# passed with no submission at all.
#
# Bootstrapping the observed gap distribution over all 38 real deadlines:
#
#     band          P(a tick lands in it)
#     [2.0, 6.0)          81.3%     <- what lost GW6
#     [0.5, 6.0)          94.5%
#     [0.5, 8.0)          99.8%
#     [0.25, 12.0)       100.0%
#
# MIN is 0.5 rather than 2.0 because the old floor protected nothing. FPL
# accepts changes until the deadline itself; half an hour is ample for a run
# that takes about ninety seconds, and the rejected GW6 run had nearly two
# hours in hand.
#
# MAX stays as tight as coverage allows, because the FIRST qualifying tick is
# the one that acts - so a wide upper bound means deciding on stale team news.
# At 8.0h the acting lead is bounded by 8h; widening to 24h would buy nothing
# (coverage is already 99.8%) and would cost a day of freshness.
DEADLINE_WINDOW_MIN = 0.5    # a submission still lands comfortably
DEADLINE_WINDOW_MAX = 8.0


def _fetch(url: str, timeout: int = REQUEST_TIMEOUT) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "fpl-auto-manager"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read().decode("utf-8"))


def fetch_events(fetcher=_fetch) -> list[dict]:
    """Every event carrying a deadline, in id order. `fetcher` is the test seam."""
    data = fetcher(BOOTSTRAP_URL)
    return [e for e in data.get("events", []) if e.get("deadline_time")]


def parse_deadline(event: dict) -> datetime | None:
    raw = event.get("deadline_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def hours_to_deadline(event: dict, now: datetime | None = None) -> float | None:
    """Hours from `now` until this gameweek's deadline, or None if unknown."""
    deadline = parse_deadline(event)
    if deadline is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (deadline - now).total_seconds() / 3600.0


def next_event(events: list[dict], now: datetime | None = None) -> dict | None:
    """The soonest gameweek whose deadline has not passed."""
    now = now or datetime.now(timezone.utc)
    upcoming = [(parse_deadline(e), e) for e in events]
    upcoming = [(d, e) for d, e in upcoming if d is not None and d > now]
    if not upcoming:
        return None
    return min(upcoming, key=lambda pair: pair[0])[1]


def previous_deadline(events: list[dict], now: datetime | None = None) -> datetime | None:
    """The most recent deadline already passed, or None before the season starts."""
    now = now or datetime.now(timezone.utc)
    past = [d for d in (parse_deadline(e) for e in events) if d is not None and d <= now]
    return max(past) if past else None


def in_acting_window(lead_hours: float | None) -> bool:
    """
    None means the deadline is unreadable. Treated as in-window on purpose: a
    schema change should make the bot run and be noisy, not quietly stand down.
    """
    if lead_hours is None:
        return True
    return DEADLINE_WINDOW_MIN <= lead_hours < DEADLINE_WINDOW_MAX


def should_act(event: dict | None, last_submitted_gw: int | None,
               now: datetime | None = None) -> tuple[bool, str]:
    """
    The whole gate, as one decision plus the reason for it.

    The `last_submitted_gw` arm is what makes extra ticks safe. Without it,
    sampling the window more than once per gameweek reintroduces the double
    submission the old single-tick schedule existed to prevent.
    """
    if event is None:
        return False, "no upcoming gameweek"

    event_id = int(event["id"])
    if last_submitted_gw is not None and last_submitted_gw == event_id:
        return False, f"GW{event_id} already submitted"

    lead = hours_to_deadline(event, now)
    if lead is None:
        return True, f"GW{event_id} deadline unreadable; running rather than skipping"
    if not in_acting_window(lead):
        return False, (f"GW{event_id} deadline is {lead:.1f}h away, outside the "
                       f"{DEADLINE_WINDOW_MIN:g}-{DEADLINE_WINDOW_MAX:g}h window")
    return True, f"GW{event_id} deadline is {lead:.1f}h away"


def _gate() -> int:
    """
    The hourly gate, as a CLI. Writes `should_run` to $GITHUB_OUTPUT so the
    expensive job can be skipped entirely - no dependency install, no priors
    fetch, and above all no FPL login, because the refresh token is single-use.

    Reads the marker from the environment rather than the API: GitHub already
    interpolates repository variables into the workflow, so this needs no token
    and cannot fail on one.
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description="decide whether the weekly run should act")
    ap.add_argument("--last-submitted-gw", default=os.environ.get("LAST_SUBMITTED_GW", ""))
    args = ap.parse_args()

    try:
        last = int(str(args.last_submitted_gw).strip())
    except (TypeError, ValueError):
        last = None

    try:
        events = fetch_events()
    except Exception as e:  # noqa: BLE001
        # Unreachable FPL API is not a reason to stand down: let the real run
        # start and fail loudly where it can be seen, rather than skipping
        # quietly on the one day it mattered.
        print(f"could not read the calendar ({type(e).__name__}: {e}); running anyway")
        act, reason = True, "calendar unreadable"
    else:
        act, reason = should_act(next_event(events), last)

    print(f"gate: should_run={act} ({reason})")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"should_run={'true' if act else 'false'}\n")
            fh.write(f"reason={reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_gate())
