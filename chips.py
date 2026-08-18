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

# Starting eleven, so a bench can be identified for a future gameweek.
XI_SIZE = 11

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


# ──────────────── chip planning over the whole window ────────────────

# How far ahead chip opportunities are compared. Beyond this the fixture list
# is still known but form is not, so a valuation stops being informative -
# and it does not need to be, because this is re-solved every gameweek. A
# rolling horizon reaches a distant double gameweek by walking toward it.
PLANNING_HORIZON = 10

# Share of a bench's expected points that autosubs already deliver without the
# chip. Bench boost pays only for what you would not otherwise have received.
# Measured on 2024-25 and 2025-26: for players who start >= 80% of gameweeks -
# the kind an XI is made of - P(zero minutes) is 0.0595, so an eleven blanks
# 0.65 times a gameweek on average, spread across four bench slots.
AUTOSUB_SHARE = 0.18


@dataclass
class ChipPlan:
    """Which gameweek each available chip should be played in."""

    assignments: dict[str, int]          # chip name -> gameweek
    values: dict[tuple[str, int], float]  # (chip, gameweek) -> expected gain
    horizon: list[int]

    def gameweek_for(self, chip: str) -> int | None:
        return self.assignments.get(chip)

    def describe(self, event: int) -> str:
        if not self.assignments:
            return "no chip is worth playing inside the planning horizon"
        bits = []
        for chip, gw in sorted(self.assignments.items(), key=lambda kv: kv[1]):
            gain = self.values.get((chip, gw), 0.0)
            when = "now" if gw == event else f"GW{gw}"
            bits.append(f"{chip} {when} ({gain:+.1f} xP)")
        return "plan: " + ", ".join(bits)


def plan_horizon(bootstrap: dict, event: int, chip_names: set[str]) -> list[int]:
    """Gameweeks worth comparing: the horizon, clipped to the widest window."""
    stops = [w["stop_event"] for w in chip_windows(bootstrap)
             if w["name"] in chip_names and w["start_event"] is not None
             and w["stop_event"] is not None and w["start_event"] <= event <= w["stop_event"]]
    if not stops:
        return [event]
    return list(range(event, min(max(stops), event + PLANNING_HORIZON - 1) + 1))


def solve_assignment(values: dict[tuple[str, int], float],
                     chips_to_place: list[str],
                     horizon: list[int]) -> dict[str, int]:
    """
    Best gameweek for each chip, at most one chip per gameweek.

    FPL permits only one chip per gameweek, so the chips compete for slots and
    cannot be chosen independently - triple captain on the one double gameweek
    displaces bench boost from it. Small enough to solve exactly by
    enumeration: four chips over ten gameweeks is a few thousand combinations.

    A chip is left unassigned when every slot is worth less than nothing.
    """
    best: tuple[float, dict[str, int]] = (0.0, {})

    def recurse(remaining: list[str], used: set[int], acc: dict[str, int], total: float) -> None:
        nonlocal best
        if total > best[0]:
            best = (total, dict(acc))
        if not remaining:
            return
        chip, rest = remaining[0], remaining[1:]
        # Skipping this chip entirely is a legal branch: a chip nobody should
        # play must not be forced into the least-bad gameweek.
        recurse(rest, used, acc, total)
        for gw in horizon:
            if gw in used:
                continue
            gain = values.get((chip, gw))
            if gain is None or gain <= 0:
                continue
            acc[chip] = gw
            recurse(rest, used | {gw}, acc, total + gain)
            acc.pop(chip)

    recurse(sorted(chips_to_place), set(), {}, 0.0)
    return best[1]


class ChipEngine:
    """Values each available chip in expected points and picks the best one."""

    def __init__(self, bootstrap: dict, fixtures: list[dict] | pd.DataFrame, scored: pd.DataFrame):
        self.bootstrap = bootstrap
        self.fixtures = fixtures
        self.scored = scored.set_index("id", drop=False)

    def _xp(self, player_ids: list[int], col: str = "xp_next") -> float:
        rows = self.scored.reindex([i for i in player_ids if i in self.scored.index])
        return float(rows[col].sum()) if len(rows) else 0.0

    def value_by_gameweek(
        self,
        event: int,
        available: set[str],
        squad_ids: list[int],
        horizon: list[int],
        best_xi_xp: dict[int, float] | None = None,
        wildcard_gain: dict[int, float] | None = None,
    ) -> dict[tuple[str, int], float]:
        """
        Expected gain from playing each available chip in each candidate
        gameweek, using the squad currently owned.

        Valuing a future gameweek on today's squad is an approximation, and a
        deliberate one: the alternative is a joint optimisation over transfers
        and chips together, and the dominant term here is the fixture calendar
        - doubles and blanks - which does not depend on the squad at all. The
        plan is re-solved every gameweek, so the squad it assumes is never more
        than a week stale.

        `best_xi_xp` and `wildcard_gain` are supplied by the caller for the
        transfer chips, since valuing those needs the optimiser.
        """
        values: dict[tuple[str, int], float] = {}
        squad = self.scored.reindex([i for i in squad_ids if i in self.scored.index])

        for gw in horizon:
            col = f"xp_gw{gw}"
            if col not in self.scored.columns:
                # A caller that scored only the coming gameweek - deadline_check,
                # or any frame without the per-gameweek columns - can still have
                # this gameweek valued. Without the fallback the planner sees no
                # opportunities at all and silently never plays a chip, which is
                # a far worse failure than a short horizon.
                if gw == event and "xp_next" in self.scored.columns:
                    col = "xp_next"
                else:
                    continue

            playing = squad[squad.get(f"fixtures_gw{gw}", 1) > 0] if f"fixtures_gw{gw}" in squad else squad

            if "bboost" in available and self._window_covers("bboost", gw):
                # Rank the squad for THAT gameweek: who sits on the bench in a
                # double gameweek is not who sits on it today.
                ranked = playing.sort_values(col, ascending=False)
                bench = ranked.iloc[XI_SIZE:] if len(ranked) > XI_SIZE else ranked.iloc[0:0]
                # Autosubs already deliver part of a bench without the chip, so
                # bench boost only pays for the remainder.
                values[("bboost", gw)] = float(bench[col].sum()) * (1.0 - AUTOSUB_SHARE)

            if "3xc" in available and self._window_covers("3xc", gw):
                # Triple captain pays one extra multiple beyond the double the
                # armband already earns.
                if len(playing):
                    values[("3xc", gw)] = float(playing[col].max())

            if "freehit" in available and self._window_covers("freehit", gw) and best_xi_xp:
                if gw in best_xi_xp:
                    ranked = playing.sort_values(col, ascending=False)
                    current_xi = float(ranked.iloc[:XI_SIZE][col].sum())
                    values[("freehit", gw)] = best_xi_xp[gw] - current_xi

            if "wildcard" in available and self._window_covers("wildcard", gw) and wildcard_gain:
                if gw in wildcard_gain:
                    values[("wildcard", gw)] = wildcard_gain[gw]

        return values

    def _window_covers(self, chip: str, gw: int) -> bool:
        for w in chip_windows(self.bootstrap):
            if w["name"] != chip:
                continue
            start, stop = w["start_event"], w["stop_event"]
            if start is not None and stop is not None and start <= gw <= stop:
                return True
        return False

    def evaluate(
        self,
        event: int,
        my_team: dict | None,
        xi_ids: list[int],
        bench_ids: list[int],
        captain_id: int,
        free_hit_xi_xp: float | None = None,
        wildcard_gain: float | None = None,
        squad_ids: list[int] | None = None,
        best_xi_xp: dict[int, float] | None = None,
        wildcard_gain_by_gw: dict[int, float] | None = None,
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

        # A threshold answers the wrong question. "Is this bench worth 13
        # points" cannot decide anything on its own; the chip is one-shot with
        # an expiry, so what matters is whether this gameweek is the *best*
        # remaining one for it. Compare the opportunities directly instead.
        horizon = plan_horizon(self.bootstrap, event, avail)
        values = self.value_by_gameweek(
            event, avail, squad_ids or (xi_ids + bench_ids), horizon,
            best_xi_xp=best_xi_xp, wildcard_gain=wildcard_gain_by_gw,
        )
        # Fall back to this gameweek's own numbers for anything the caller
        # could only value for now - the transfer chips, without a per-gameweek
        # optimiser solve.
        for chip, gain in (("freehit", free_hit_xi_xp), ("wildcard", wildcard_gain)):
            if chip in avail and (chip, event) not in values and gain is not None:
                values[(chip, event)] = (
                    gain - self._xp(xi_ids) if chip == "freehit" else gain
                )

        # The floor belongs inside the assignment, not after it.
        #
        # The plan answers *which* gameweek among those it can see; it cannot
        # answer *whether* to commit, because the window runs past the horizon
        # and an unseen double is worth more than the best mediocre week in
        # view. Left ungated the planner spends a one-shot chip on any positive
        # value at all - it proposed bench boost for 1.6 xP on a flat bench.
        #
        # Applying that as a veto *after* solving was worse still: with several
        # chips in play the solver would put a marginal one in this gameweek,
        # fail the bar, and abandon the whole plan - including a far more
        # valuable chip scheduled elsewhere. Filtering the options first means
        # every assignment the solver can reach is one worth committing to, and
        # it maximises total gain across them.
        committable = {
            (chip, gw): gain for (chip, gw), gain in values.items()
            if gain >= effective_threshold(self.bootstrap, chip, gw)
        }

        plan = ChipPlan(
            assignments=solve_assignment(committable, sorted(avail), horizon),
            values=values,
            horizon=horizon,
        )
        self.last_plan = plan
        logger.info("chip %s", plan.describe(event))

        playing_now = [c for c, gw in plan.assignments.items() if gw == event]
        if not playing_now:
            later = sorted(plan.assignments.items(), key=lambda kv: kv[1])
            if later:
                chip, gw = later[0]
                return ChipDecision(
                    None, 0.0,
                    f"holding {chip} for GW{gw} ({plan.values.get((chip, gw), 0.0):+.1f} xP "
                    f"vs {values.get((chip, event), 0.0):+.1f} now)",
                )
            best = max(values.items(), key=lambda kv: kv[1], default=None)
            if best:
                bar = effective_threshold(self.bootstrap, best[0][0], best[0][1])
                detail = (f"best was {best[0][0]} at {best[1]:.1f} xP in GW{best[0][1]}, "
                          f"under its {bar:.1f} bar")
            else:
                detail = "nothing to evaluate"
            return ChipDecision(None, 0.0, f"no chip is worth committing yet ({detail})")

        # One chip per gameweek is a hard rule, so this list has one entry.
        chip = playing_now[0]
        gain = values.get((chip, event), 0.0)
        chosen = ChipDecision(chip, gain, f"best gameweek in the window for it ({gain:+.1f} xP)")
        logger.info("chip: playing %s (%s)", chosen.chip, chosen.reason)
        return chosen
