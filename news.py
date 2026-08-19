"""
Read FPL's own injury and suspension text into structured availability.

The model used to compute one availability number per player and apply it
unchanged to every gameweek of the horizon. That writes off anyone who is out
now but back soon: a player carrying "Expected back 22 Aug" scored zero for all
five gameweeks when he is fit from the second, and a suspension counted against
gameweeks served long after it ended. Those players are systematically cheap
and almost unowned - the returning-player discount is one of the few edges the
game hands out for free - and the model could not buy any of them.

The `news` field is the right source for this rather than anything scraped. It
is maintained by the game itself, it is what the game's own flags are derived
from, and every one of the twelve GW1 absences it listed matched independent
reporting exactly. It is also close to structured already: the strings follow a
small number of templates.

Nothing here decides how good a player is. It decides only *when he can play*,
which is then multiplied into the minutes model.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

logger = logging.getLogger("fpl_auto.news")

# "Expected back 22 Aug", "Suspended until 29 Aug", "Expected back 6 Sep".
# The year is never given, which is handled in `_resolve_year`.
RETURN_PATTERNS = [
    re.compile(r"expected back\s+(\d{1,2})\s+([A-Za-z]{3,})", re.I),
    re.compile(r"suspended until\s+(\d{1,2})\s+([A-Za-z]{3,})", re.I),
    re.compile(r"(?:back|available)\s+(?:on\s+)?(\d{1,2})\s+([A-Za-z]{3,})", re.I),
]

MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# Phrases that mean "out, with no date attached". Distinguished from an
# unparsed string so that a genuinely new template shows up in the logs as
# unparsed rather than being silently read as an indefinite absence.
INDEFINITE = re.compile(r"unknown return|no return date|indefinit", re.I)

# Templates that are understood and deliberately carry no return date. Without
# these the unparsed log fills with 53 "75% chance of playing" strings per run -
# all of them already handled by chance_of_playing_next_round - and a genuinely
# new template FPL introduces would be lost in the noise. That log is the only
# thing that will tell us this module has stopped working.
KNOWN_NO_DATE = re.compile(
    r"\d+%\s+chance of playing"      # already carried by chance_of_playing_next_round
    r"|has joined|has returned to|on loan|permanently"  # left the league; status 'u' covers it
    r"|self-isolat|personal reasons|lack of match fitness|international duty",
    re.I,
)

# A player back from a long absence does not walk straight into ninety minutes.
# The damping is applied to his start probability for the first gameweeks after
# his return date, and is deliberately coarse: the point is to stop the model
# treating "fit again" as "immediately nailed", not to model a fitness curve
# nobody has published.
RAMP_UP = [0.55, 0.80]          # first and second gameweek back
LONG_ABSENCE_DAYS = 28          # below this, a return needs no ramp

# A knock resolves. `chance_of_playing_next_round` describes the coming
# gameweek only, so carrying 75% across a five-gameweek horizon prices a
# player as permanently doubtful. Doubt decays toward fully fit.
DOUBT_RECOVERY = 0.5            # share of the shortfall recovered per gameweek


@dataclass(frozen=True)
class Availability:
    """When a player can next feature, and why he cannot before then."""

    returns_on: date | None      # first date he could play; None = no date given
    indefinite: bool             # out, with no return date published
    reason: str                  # the raw news string, kept for the audit trail

    @property
    def parsed(self) -> bool:
        return self.returns_on is not None or self.indefinite


def _resolve_year(day: int, month: int, today: date) -> date:
    """
    Attach a year to a bare "22 Aug".

    A football season straddles new year, so the nearest interpretation is the
    right one: a date that lands far in the past means next year's, and one far
    in the future means last year's. Six months either side splits it cleanly.
    """
    for year in (today.year, today.year + 1, today.year - 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue            # 29 Feb in a non-leap year
        if -180 <= (candidate - today).days <= 185:
            return candidate
    return date(today.year, month, min(day, 28))


def parse(news: str | None, today: date | None = None) -> Availability | None:
    """
    Read one `news` string. Returns None when it says nothing about a return.

    A string that carries no date and matches no known "out indefinitely"
    phrasing returns None rather than guessing, so the caller falls back to the
    status flag exactly as before.
    """
    if not news or not news.strip():
        return None
    today = today or date.today()
    text = news.strip()

    for pattern in RETURN_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        day = int(m.group(1))
        month = MONTHS.get(m.group(2)[:3].lower())
        if not month or not 1 <= day <= 31:
            continue
        return Availability(returns_on=_resolve_year(day, month, today),
                            indefinite=False, reason=text)

    if INDEFINITE.search(text):
        return Availability(returns_on=None, indefinite=True, reason=text)

    return None


def is_known_template(news: str | None) -> bool:
    """True for strings we understand but which carry no return date."""
    return bool(news and KNOWN_NO_DATE.search(news))


def availability_on(
    info: Availability | None,
    when: date,
    *,
    base: float,
    status: str,
) -> float:
    """
    P(this player is selectable for a fixture on `when`.)

    `base` is the availability the caller would otherwise have used - the
    published chance of playing, or the status-flag fallback. This function
    only ever moves it in the direction the news supports:

    - no parsed news        -> base, unchanged
    - out indefinitely      -> base, unchanged (the status flag already says 0)
    - returns after `when`  -> 0.0, he cannot play
    - returns on or before  -> 1.0, the absence has ended

    The last case is the whole point. A suspension that ends before a gameweek
    should not depress that gameweek, and `base` is derived from
    chance_of_playing_next_round, which describes only the coming one.
    """
    if info is None or info.indefinite:
        return base
    if info.returns_on is None:
        return base
    if when < info.returns_on:
        return 0.0
    # Back in time for this fixture. The status flag still reads 'i' or 's'
    # because FPL has not cleared it yet, so base would be 0 - the news is the
    # better evidence and overrides it.
    return 1.0


def ramp_multiplier(info: Availability | None, when: date, absence_days: float | None = None) -> float:
    """
    How much of his usual starting role a returning player gets back, and when.

    Applied on top of availability: a player can be fit to play and still be
    eased in. Only absences long enough to cost match fitness are damped, so a
    one-match suspension returns at full strength.
    """
    if info is None or info.returns_on is None or when < info.returns_on:
        return 1.0
    if absence_days is not None and absence_days < LONG_ABSENCE_DAYS:
        return 1.0
    weeks_back = max(0, (when - info.returns_on).days // 7)
    if weeks_back < len(RAMP_UP):
        return RAMP_UP[weeks_back]
    return 1.0


def decay_doubt(base: float, gameweeks_ahead: int) -> float:
    """
    Let a short-term doubt fade over the horizon.

    `chance_of_playing_next_round` is a statement about the next gameweek. A
    player at 75% with a knock is not 75% likely to be fit in five weeks' time;
    he is nearly certain to be. Holding the figure flat across the horizon
    prices every knock as permanent and undervalues anyone carrying one.
    """
    if gameweeks_ahead <= 0 or base >= 1.0:
        return base
    recovered = 1.0 - (1.0 - DOUBT_RECOVERY) ** gameweeks_ahead
    return float(min(1.0, base + (1.0 - base) * recovered))


def parse_all(elements: list[dict], today: date | None = None) -> dict[int, Availability]:
    """Parse every element's news, logging anything that looks like a new template."""
    out: dict[int, Availability] = {}
    unparsed: list[str] = []
    for e in elements:
        news = (e.get("news") or "").strip()
        if not news:
            continue
        info = parse(news, today=today)
        if info is None:
            if e.get("status") not in ("a", None) and not is_known_template(news):
                unparsed.append(news)
            continue
        out[int(e["id"])] = info

    dated = sum(1 for a in out.values() if a.returns_on)
    logger.info("news: %d players flagged, %d with a return date, %d unrecognised",
                len(out) + len(unparsed), dated, len(unparsed))
    if unparsed:
        # Surfaced rather than swallowed: a template FPL changes is how this
        # quietly stops working.
        logger.info("news templates not recognised: %s", "; ".join(sorted(set(unparsed))[:5]))
    return out
