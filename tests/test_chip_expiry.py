"""
A chip that expires unplayed is worth nothing, so near the close of its window
any positive gain must beat holding it.

All four first-half chips lapse at GW19. The commit bar only ever fell to its
base (12 bench boost, 6 triple captain, 15 free hit, 20 wildcard), so on a
season with no first-half doubles - 2026-27 has none in GW9-19 - an ordinary
bench (~8 xP) never clears 12 and the chip is silently thrown away. The old
test of this behaviour was named "the commit floor falls to nothing at the end
of the window" and asserted it fell to 12.

Synthetic state rather than a snapshot: the failure is about the window
arithmetic, and every value here sits deliberately below its base bar.
"""
from __future__ import annotations

import pandas as pd
import pytest

import chips

FIRST_HALF = [
    {"name": "bboost", "start_event": 1, "stop_event": 19},
    {"name": "3xc", "start_event": 1, "stop_event": 19},
    {"name": "wildcard", "start_event": 2, "stop_event": 19},
    {"name": "freehit", "start_event": 2, "stop_event": 19},
]
SECOND_HALF = [{**w, "start_event": 20, "stop_event": 38} for w in FIRST_HALF]
BOOTSTRAP = {"chips": FIRST_HALF + SECOND_HALF}

XI = list(range(1, 12))
BENCH = list(range(12, 16))
XI_XP, BENCH_XP = 5.0, 2.0
WILDCARD_GAIN = 8.0     # under 20
FREE_HIT_GAIN = 5.0     # under 15


def _scored() -> pd.DataFrame:
    rows = []
    for pid in XI + BENCH:
        xp = XI_XP if pid in XI else BENCH_XP
        row = {"id": pid, "xp_next": xp}
        row.update({f"xp_gw{gw}": xp for gw in range(1, 39)})
        rows.append(row)
    return pd.DataFrame(rows)


def _decide(event: int, played: list[tuple[str, int]]) -> chips.ChipDecision:
    engine = chips.ChipEngine(BOOTSTRAP, [], _scored())
    my_team = {"chips": [{"name": n, "event": gw} for n, gw in played]}
    return engine.evaluate(
        event, my_team, xi_ids=XI, bench_ids=BENCH, captain_id=XI[0],
        free_hit_xi_xp=len(XI) * XI_XP + FREE_HIT_GAIN,
        wildcard_gain=WILDCARD_GAIN, squad_ids=XI + BENCH,
    )


def test_every_value_here_is_under_its_base_bar():
    """Otherwise the tests below would pass on the old rule too."""
    bench = len(BENCH) * BENCH_XP * (1 - chips.AUTOSUB_SHARE)
    assert bench < chips.THRESHOLDS["bboost"]
    assert XI_XP < chips.THRESHOLDS["3xc"]
    assert FREE_HIT_GAIN < chips.THRESHOLDS["freehit"]
    assert WILDCARD_GAIN < chips.THRESHOLDS["wildcard"]


def test_no_first_half_chip_expires_while_it_is_worth_something():
    """Walk the close of the window week by week, playing whatever is chosen."""
    played: list[tuple[str, int]] = []
    for gw in range(9, 20):
        d = _decide(gw, played)
        if d.chip:
            played.append((d.chip, gw))

    assert sorted(n for n, _ in played) == sorted(w["name"] for w in FIRST_HALF), (
        f"chips lapsed unplayed at GW19: played only {played}"
    )
    assert len({gw for _, gw in played}) == len(played), "two chips in one gameweek"


def test_mid_window_the_bar_still_holds():
    """Forcing is for the close only; GW8 with the same values plays nothing."""
    assert _decide(8, []).chip is None


def test_forcing_starts_once_slots_run_short():
    assert chips.chips_at_risk_of_expiry(BOOTSTRAP, {w["name"] for w in FIRST_HALF}, 14) == set()
    names = {w["name"] for w in FIRST_HALF}
    # Four chips, GW15-19 is five slots: four plus the one spare.
    assert chips.chips_at_risk_of_expiry(BOOTSTRAP, names, 15) == names
    assert chips.chips_at_risk_of_expiry(BOOTSTRAP, {"bboost"}, 17) == set()
    assert chips.chips_at_risk_of_expiry(BOOTSTRAP, {"bboost"}, 18) == {"bboost"}


def test_a_worthless_chip_is_still_not_forced():
    """Expiry beats a positive gain, not a negative one."""
    engine = chips.ChipEngine(BOOTSTRAP, [], _scored())
    my_team = {"chips": [{"name": n, "event": 10} for n in ("bboost", "3xc", "freehit")]}
    d = engine.evaluate(19, my_team, xi_ids=XI, bench_ids=BENCH, captain_id=XI[0],
                        wildcard_gain=-1.0, squad_ids=XI + BENCH)
    assert d.chip is None


def test_the_second_half_window_is_untouched_at_gw19():
    """Forcing the first-half chips must not reach into the second half's."""
    d = _decide(20, [(w["name"], 10 + i) for i, w in enumerate(FIRST_HALF)])
    assert d.chip is None
