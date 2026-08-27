"""
What the nineteen people you are actually playing against own, and captain.

The ownership tilt runs on `selected_by_percent` - the eleven-million-player
template - while the league that decides your season is a couple of dozen
specific managers. Measured on GW1 of 2026-27 those are not the same
distribution and the gap is not small:

    Szoboszlai      field 72.2%   template 43.0%   +29.2pp
    Calvert-Lewin   field 80.0%   template 29.1%   +50.9pp
    Kinsky          field 80.0%   template 22.9%   +57.1pp

A tilt aimed at the template is therefore aimed at the wrong target for
anything except the handful of players where the two happen to agree.

Squad ownership is also not captaincy ownership, which is the sharper of the
two decisions: in league 1178688, João Pedro was owned by 77.8% of the field
and captained by 16.7% of it.

Everything here reads **public** endpoints. `LeagueAnalyzer` goes through the
authenticated client for `get_my_leagues()`, which spends a single-use refresh
token; `/entry/{id}/` publishes the same league list to anybody, so none of
this needs to authenticate and none of it can lock the account out.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger("fpl_auto.rivals")

API = "https://fantasy.premierleague.com/api"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
TIMEOUT = 30
RETRIES = 3

# Leagues everybody is in by construction - "Overall", "Gameweek 1", a country,
# a favourite club. They carry millions of entries, so they are the template
# under another name and tell you nothing about your rivals. The real mini
# leagues are small.
MAX_LEAGUE_SIZE = 2000


@dataclass
class FieldOwnership:
    """How a specific field of managers is set up, for one gameweek."""

    event: int
    managers: int
    squad: dict[int, float] = field(default_factory=dict)      # element -> share
    captain: dict[int, float] = field(default_factory=dict)    # element -> share

    def squad_share(self, element: int) -> float:
        return self.squad.get(int(element), 0.0)

    def captain_share(self, element: int) -> float:
        return self.captain.get(int(element), 0.0)


class PublicFPL:
    """Read-only FPL access. No credentials, so no token can be spent."""

    def __init__(self, opener=None):
        self._opener = opener or self._fetch

    @staticmethod
    def _fetch(url: str):
        last = None
        for attempt in range(RETRIES):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    return json.load(r)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                # A rival who has never set a squad 404s; that is data, not a
                # failure, and must not take the weekly run down with it.
                if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                    return None
                time.sleep(2**attempt)
        logger.warning("giving up on %s (%s)", url, last)
        return None

    def get(self, path: str):
        return self._opener(f"{API}{path}")


def mini_league_ids(client: PublicFPL, entry_id: int) -> list[int]:
    """Classic leagues small enough to be a real field rather than the template."""
    entry = client.get(f"/entry/{entry_id}/")
    if not entry:
        return []
    out = []
    for lg in entry.get("leagues", {}).get("classic", []):
        size = lg.get("rank_count") or 0
        if 0 < size <= MAX_LEAGUE_SIZE:
            out.append(int(lg["id"]))
        else:
            logger.debug("skipping league %s (%s entries)", lg.get("name"), size)
    return out


def rival_ids(client: PublicFPL, league_id: int, entry_id: int,
              max_rivals: int = 50) -> list[int]:
    st = client.get(f"/leagues-classic/{league_id}/standings/")
    if not st or "standings" not in st:
        return []
    return [int(r["entry"]) for r in st["standings"].get("results", [])[:max_rivals]
            if int(r["entry"]) != int(entry_id)]


def field_ownership(client: PublicFPL, entry_id: int, event: int,
                    league_ids: list[int] | None = None) -> FieldOwnership:
    """
    Squad and captain shares across every rival in your mini leagues.

    Returns zero managers rather than raising when the gameweek has not
    finished - `/entry/{id}/event/{gw}/picks/` publishes nothing before then,
    and a caller must be able to fall back to the template instead of crashing
    the weekly run.
    """
    if league_ids is None:
        league_ids = mini_league_ids(client, entry_id)

    seen: set[int] = set()
    for lid in league_ids:
        seen.update(rival_ids(client, lid, entry_id))

    squad: Counter[int] = Counter()
    captain: Counter[int] = Counter()
    n = 0
    for rid in sorted(seen):
        picks = client.get(f"/entry/{rid}/event/{event}/picks/")
        if not picks or not picks.get("picks"):
            continue
        n += 1
        for pk in picks["picks"]:
            el = int(pk["element"])
            squad[el] += 1
            if int(pk.get("multiplier", 1)) > 1:
                captain[el] += 1

    if not n:
        logger.info("no rival picks available for GW%s; caller should fall back", event)
        return FieldOwnership(event=event, managers=0)

    logger.info("field ownership from %d rivals across %d leagues", n, len(league_ids))
    return FieldOwnership(
        event=event,
        managers=n,
        squad={e: c / n for e, c in squad.items()},
        captain={e: c / n for e, c in captain.items()},
    )
