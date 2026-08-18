"""
Chip availability and chip decisions.

Replaces chip_strategy.py, which had two problems. It treated a chip as spent
forever once played, and the game awards two of everything: the bootstrap chip
list carries separate windows for gameweeks 1-19 and 20-38, so a manager who
plays a wildcard in October still has one for the second half. The old code
would have believed it had no chips left for an entire half-season.

The second problem was that decisions ran on confidence heuristics - play a
chip at "60% confidence", trigger a wildcard on a rank drop - which cannot be
compared against each other or against the cost of doing nothing. Here every
chip is valued in expected points, so bench boost and triple captain compete
on the same scale and a chip is played when it clears a points threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("fpl_auto.chips")

CHIP_NAMES = {"wildcard", "freehit", "bboost", "3xc"}

# Minimum expected gain, in points, to justify burning a chip. A chip is a
# one-shot resource, so the bar is well above zero: playing bench boost for two
# points wastes the half-season's allocation.
THRESHOLDS = {
    "bboost": 12.0,
    "3xc": 6.0,      # gain over a normal captain, not the captain's total
    "freehit": 15.0,
    "wildcard": 20.0,
}

# How much higher the bar sits at the very start of a chip's window than at its
# end. A chip is a one-shot resource with an expiry, so "is this worth 13
# points" is the wrong question - the right one is "is this better than the
# best chance I expect before the window closes". With eighteen gameweeks still
# to come those chances are plentiful and the bar should be high; on the last
# gameweek of the window it is play-it-or-lose-it and the base threshold is the
# whole test.
#
# This came out of a real GW1 dry run proposing bench boost for 13.3 xP against
# a flat 12.0 bar - in a gameweek where every player appears exactly once, with
# every double gameweek of the half-season still ahead. A static threshold
# cannot express that, and spending the chip there would have been a clear loss
# against playing the same chip on a double.
#
# 0.6 is a judgement, not a measurement: it makes the opening bar 1.6x the
# closing one, enough to hold a chip through an ordinary gameweek but not so
# high that a genuinely exceptional early week is refused. Replacing it with a
# real option value - the expected maximum over the remaining window, which
# needs a distribution of weekly chip values - is the honest version and wants
# a backtest behind it.
EARLY_WINDOW_PREMIUM = 0.6


def window_for(bootstrap: dict, name: str, event: int) -> dict | None:
    """The window of chip `name` that covers `event`, if any."""
    for w in chip_windows(bootstrap):
        start, stop = w["start_event"], w["stop_event"]
        if start is not None and stop is not None and start <= event <= stop:
            if w["name"] == name:
                return w
    return None


def effective_threshold(bootstrap: dict, name: str, event: int) -> float:
    """
    The bar chip `name` must clear to be worth playing in `event`.

    Scales from EARLY_WINDOW_PREMIUM above the base threshold at the opening of
    the window down to the base threshold on its final gameweek.
    """
    base = THRESHOLDS.get(name, 0.0)
    w = window_for(bootstrap, name, event)
    if not w:
        return base
    start, stop = int(w["start_event"]), int(w["stop_event"])
    span = max(1, stop - start)
    remaining = max(0, stop - event) / span      # 1.0 at the opening, 0.0 at the close
    return base * (1.0 + EARLY_WINDOW_PREMIUM * remaining)


@dataclass
class ChipDecision:
    chip: str | None
    expected_gain: float
    reason: str


def chip_windows(bootstrap: dict) -> list[dict]:
    """
    Chip windows as published by the game.

    Each entry is one usable chip: name plus the gameweek range it is valid
    for. Two entries per chip name in the current format, one per half-season.
    """
    windows = []
    for c in bootstrap.get("chips", []):
        name = c.get("name")
        if name not in CHIP_NAMES:
            continue
        windows.append({
            "name": name,
            "start_event": c.get("start_event"),
            "stop_event": c.get("stop_event"),
        })
    return windows


def played_chips(my_team: dict | None) -> list[tuple[str, int]]:
    """
    Every chip this entry has played, as (name, gameweek).

    Two endpoints describe this and they do not agree, which is why this is a
    function rather than a comprehension. Verified against a real authenticated
    response on 2026-08-17, saved at `tests/fixtures/my_team.json`:

    - `/my-team/` returns entries keyed
      `{name, id, number, chip_type, start_event, stop_event,
        status_for_entry, played_by_entry, is_pending}`
      where `played_by_entry` is a list of gameweeks.
    - `/entry/{id}/history/` returns `{name, event}`.

    This code previously read `c["event"]` against the `/my-team/` payload,
    where that key does not exist. Every chip therefore looked unplayed
    forever, and the engine would have re-submitted a chip it had already
    spent - silently, since FPL simply ignores the second attempt.

    Both shapes are accepted because both endpoints are still read.
    """
    out: list[tuple[str, int]] = []
    for c in (my_team or {}).get("chips") or []:
        name = c.get("name")
        if not name:
            continue
        for ev in c.get("played_by_entry") or []:
            out.append((name, int(ev)))
        ev = c.get("event")
        if ev is not None:
            out.append((name, int(ev)))
    return out


def pending_chips(my_team: dict | None) -> set[str]:
    """
    Chips already activated for the coming gameweek but not yet locked in.

    `is_pending` is how `/my-team/` reports a chip that has been switched on
    for a deadline that has not passed. It is not in `played_by_entry` yet, so
    without this a second run in the same gameweek sees the chip as unplayed
    and tries to play another one on top of it.
    """
    return {
        c["name"] for c in (my_team or {}).get("chips") or []
        if c.get("name") and c.get("is_pending")
    }


def available_chips(bootstrap: dict, my_team: dict | None, event: int) -> set[str]:
    """
    Which chips can be played in `event`.

    A chip is available when some window covers this gameweek and no chip of
    that name has already been played *inside that same window*. Checking
    globally instead - as the old code did - retires the second-half chip the
    moment the first-half one is used.
    """
    played = played_chips(my_team)
    pending = pending_chips(my_team)

    out = set()
    for w in chip_windows(bootstrap):
        start, stop = w["start_event"], w["stop_event"]
        if start is None or stop is None:
            continue
        if not (start <= event <= stop):
            continue
        used_in_window = any(n == w["name"] and start <= e <= stop for n, e in played)
        if not used_in_window and w["name"] not in pending:
            out.add(w["name"])

    if pending:
        logger.info("chips already pending for this gameweek: %s", ", ".join(sorted(pending)))
    return out


def fixtures_per_team(fixtures: list[dict] | pd.DataFrame, event: int) -> dict[int, int]:
    """How many fixtures each team has in a gameweek. 2 = double, 0 = blank."""
    fx = pd.DataFrame(fixtures) if not isinstance(fixtures, pd.DataFrame) else fixtures
    if fx.empty or "event" not in fx.columns:
        return {}
    gw = fx[fx["event"] == event]
    counts: dict[int, int] = {}
    for _, f in gw.iterrows():
        for side in ("team_h", "team_a"):
            t = int(f[side])
            counts[t] = counts.get(t, 0) + 1
    return counts


def describe_calendar(fixtures: list[dict] | pd.DataFrame, event: int, teams: pd.DataFrame) -> str:
    """One-line summary of doubles and blanks, for the run report."""
    counts = fixtures_per_team(fixtures, event)
    if not counts:
        return f"GW{event}: no fixture data"
    names = teams.set_index("id")["short_name"].to_dict() if "short_name" in teams.columns else {}
    doubles = [names.get(t, t) for t, c in counts.items() if c >= 2]
    all_ids = set(teams["id"].astype(int)) if "id" in teams.columns else set()
    blanks = [names.get(t, t) for t in all_ids - set(counts)]
    bits = []
    if doubles:
        bits.append(f"doubles: {', '.join(map(str, sorted(doubles)))}")
    if blanks:
        bits.append(f"blanks: {', '.join(map(str, sorted(blanks)))}")
    return f"GW{event}: " + ("; ".join(bits) if bits else "standard gameweek")


class ChipEngine:
    """Values each available chip in expected points and picks the best one."""

    def __init__(self, bootstrap: dict, fixtures: list[dict] | pd.DataFrame, scored: pd.DataFrame):
        self.bootstrap = bootstrap
        self.fixtures = fixtures
        self.scored = scored.set_index("id", drop=False)

    def _xp(self, player_ids: list[int], col: str = "xp_next") -> float:
        rows = self.scored.reindex([i for i in player_ids if i in self.scored.index])
        return float(rows[col].sum()) if len(rows) else 0.0

    def evaluate(
        self,
        event: int,
        my_team: dict | None,
        xi_ids: list[int],
        bench_ids: list[int],
        captain_id: int,
        free_hit_xi_xp: float | None = None,
        wildcard_gain: float | None = None,
    ) -> ChipDecision:
        """
        Pick a chip for this gameweek, or none.

        free_hit_xi_xp and wildcard_gain come from the optimiser, since valuing
        those two requires solving for a different squad. When they are not
        supplied those chips are simply not considered - the caller decides
        whether the extra solve is worth it.
        """
        avail = available_chips(self.bootstrap, my_team, event)
        if not avail:
            return ChipDecision(None, 0.0, "no chips available in this window")

        candidates: list[ChipDecision] = []

        if "bboost" in avail:
            gain = self._xp(bench_ids)
            candidates.append(ChipDecision("bboost", gain, f"bench worth {gain:.1f} xP"))

        if "3xc" in avail:
            # Triple captain pays one extra multiple of the captain, on top of
            # the double he already earns.
            gain = self._xp([captain_id])
            candidates.append(ChipDecision("3xc", gain, f"captain worth {gain:.1f} xP extra"))

        if "freehit" in avail and free_hit_xi_xp is not None:
            gain = free_hit_xi_xp - self._xp(xi_ids)
            candidates.append(ChipDecision("freehit", gain, f"free hit XI beats current by {gain:.1f} xP"))

        if "wildcard" in avail and wildcard_gain is not None:
            candidates.append(ChipDecision("wildcard", wildcard_gain, f"rebuild gains {wildcard_gain:.1f} xP over the horizon"))

        bars = {c.chip: effective_threshold(self.bootstrap, c.chip, event) for c in candidates}
        viable = [c for c in candidates if c.expected_gain >= bars[c.chip]]
        if not viable:
            best = max(candidates, key=lambda c: c.expected_gain - bars[c.chip], default=None)
            detail = (
                f"best was {best.chip} at {best.expected_gain:.1f} xP against a "
                f"{bars[best.chip]:.1f} bar" if best else "nothing to evaluate"
            )
            return ChipDecision(None, 0.0, f"no chip clears its threshold ({detail})")

        # Rank by how far each clears its own bar, so chips with different
        # thresholds are compared fairly rather than by raw points.
        chosen = max(viable, key=lambda c: c.expected_gain - bars[c.chip])
        logger.info("chip: playing %s (%s)", chosen.chip, chosen.reason)
        return chosen
