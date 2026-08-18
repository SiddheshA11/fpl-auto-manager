"""
Capture a real authenticated /my-team/ response as a test fixture.

The chip, transfer and lineup payload shapes were inferred from documentation
and from what the code needed them to be, never observed. The existing tests
construct the *assumed* shape, so they prove the code agrees with itself rather
than with FPL - and every one of those assumptions fails at a deadline, which
is the one moment nobody is watching.

This runs in CI rather than locally on purpose. PingOne rotates the refresh
token on use, so a local run consumes the stored token and receives a
replacement that never reaches the GitHub secret: the local run reports success
and every subsequent CI run is locked out. Running here means the rotation path
writes the new token back to the secret exactly as the weekly run does.

Read-only. It issues no POST and cannot alter the team.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path

from config import FPL_TEAM_ID
from fpl_client import FPLClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fpl_auto.dump")

# Nothing in /my-team/ should carry personal data, but the file is destined for
# a public repository, so anything name- or contact-shaped is stripped rather
# than trusted. Squad, chip and transfer state - the whole point of the
# fixture - is unaffected.
REDACT_KEYS = {
    "email", "player_first_name", "player_last_name", "name", "entry_email",
    "player_region_name", "player_region_iso_code_long", "player_region_iso_code_short",
}

# `name` means two different things in this payload: the manager's name, which
# must go, and the chip's name ("bboost", "3xc"), which is the single most
# important field in the fixture. Redacting by key alone destroyed the latter,
# so these subtrees are exempt.
REDACT_EXEMPT_PARENTS = {"chips"}


def redact(obj, parent: str | None = None):
    """Recursively blank anything identifying, preserving structure and types."""
    if isinstance(obj, dict):
        return {
            k: ("<redacted>"
                if k in REDACT_KEYS and obj[k] is not None and parent not in REDACT_EXEMPT_PARENTS
                else redact(v, k))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v, parent) for v in obj]
    return obj


def describe(my_team: dict) -> str:
    """
    Summarise the three shapes that were unverified, for the run log.

    The log is the fast path: it answers the questions without anyone having to
    download and read the artifact.
    """
    lines = []
    transfers = my_team.get("transfers") or {}
    lines.append(f"transfers block keys : {sorted(transfers)}")
    lines.append(f"transfers block      : {json.dumps(transfers)}")
    lines.append("")
    lines.append(
        "  ^ `limit` is the one that matters before the GW1 deadline. Transfers are "
        "unlimited until then, and manager.py reads `(limit or 1) - made`, so a null "
        "or zero here means the bot believes it has one free transfer during the one "
        "week it can rebuild the entire squad for nothing."
    )
    lines.append("")

    chips = my_team.get("chips")
    lines.append(f"chips type           : {type(chips).__name__}")
    if isinstance(chips, list) and chips:
        lines.append(f"chips entry keys     : {sorted(chips[0])}")
        lines.append(f"chips[0]             : {json.dumps(chips[0])}")
        lines.append("")
        lines.append(
            "  ^ chips.py:74-79 expects {'name', 'event'} - the /entry/{id}/history/ "
            "shape. If these carry `status_for_entry` / `played_by_entry` instead, "
            "played chips are never detected and the engine re-submits a spent one."
        )
    else:
        lines.append(f"chips                : {json.dumps(chips)}")
        lines.append("  ^ empty; shape is still undetermined until a chip has been played.")
    lines.append("")

    picks = my_team.get("picks") or []
    lines.append(f"picks                : {len(picks)} entries")
    if picks:
        lines.append(f"pick entry keys      : {sorted(picks[0])}")
        lines.append(f"picks[0]             : {json.dumps(picks[0])}")
    lines.append("")
    lines.append(f"top-level keys       : {sorted(my_team)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tests/fixtures/my_team.json",
                    help="where to write the redacted fixture")
    args = ap.parse_args()

    # Check rotation works *before* spending the token, not after.
    #
    # This script was first dispatched against `main`, which still carries the
    # old client - one that looks for GITHUB_TOKEN and never GH_PAT. PingOne
    # rotates the refresh token on use, so the run authenticated, received a
    # replacement, failed to store it, and left the stored secret spent. The
    # workflow file has to live on the default branch for GitHub to register
    # workflow_dispatch, but the *code* comes from whichever ref is dispatched,
    # and those were not the same branch.
    #
    # A refresh token is a single-use resource. Refusing to start costs nothing;
    # starting and failing costs the token and locks out every later run.
    if not os.environ.get("GH_PAT") and not os.environ.get("GITHUB_TOKEN"):
        logger.critical("no GH_PAT/GITHUB_TOKEN; refusing to spend the refresh token")
        return 1
    try:
        rotation_source = inspect.getsource(FPLClient._update_github_secret)
    except (OSError, TypeError):
        rotation_source = ""
    if "GH_PAT" not in rotation_source:
        logger.critical(
            "this checkout's FPLClient cannot read GH_PAT, so a rotated token would be "
            "discarded and the stored secret left spent. Dispatch against a ref whose "
            "fpl_client.py supports GH_PAT (rebuild/xp-model or later)."
        )
        return 1

    client = FPLClient()
    if not client.login():
        logger.critical("authentication failed")
        return 1

    my_team = client.get_my_team()
    if not my_team:
        logger.critical("could not fetch /my-team/")
        return 1

    print("\n" + "=" * 72)
    print(f"/my-team/ for entry {FPL_TEAM_ID}")
    print("=" * 72)
    print(describe(my_team))
    print("=" * 72)

    # The history endpoint is where the {'name', 'event'} chip shape is
    # documented. Capturing both side by side is what actually settles whether
    # the two endpoints agree, which is the open question.
    history = client.get_my_history()
    if history is not None:
        hist_chips = history.get("chips")
        print(f"/entry/{FPL_TEAM_ID}/history/ chips: {json.dumps(hist_chips)}")
        if isinstance(hist_chips, list) and hist_chips:
            print(f"  history chip entry keys: {sorted(hist_chips[0])}")
        print("=" * 72)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "my_team": redact(my_team),
        "entry_history_chips": redact((history or {}).get("chips")),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", out)

    if getattr(client, "rotation_failed", False):
        logger.critical("token rotation failed; the stored refresh token is now spent")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
