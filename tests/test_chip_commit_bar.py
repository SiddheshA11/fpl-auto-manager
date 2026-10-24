"""
Every week in view is judged against THIS week's bar.

The decision being made is whether to commit now. Judging a future week
against its own, lower bar - the bar decays toward the close of the window -
let a later week enter the plan on a value that would not justify committing
today. GW7 of 2026-27 logged "holding 3xc for GW16 (+7.6 xP vs +7.7 now)": the
held-for week was worth less than the week it was held from, and only made the
plan because GW16's bar was 6.6 against GW7's 8.4. The decision itself (no
chip) was right; the plan it reported was not.

Synthetic state, reusing test_chip_expiry's: the point is the bar arithmetic.
"""
from __future__ import annotations

import pandas as pd
import pytest

import chips
from test_chip_expiry import BENCH, BOOTSTRAP, XI

OTHERS_SPENT = [{"name": n, "event": 3} for n in ("bboost", "wildcard", "freehit")]


def _engine(captain_by_gw: dict[int, float]) -> chips.ChipEngine:
    rows = []
    for pid in XI + BENCH:
        row = {"id": pid, "xp_next": 2.0}
        for gw in range(1, 39):
            row[f"xp_gw{gw}"] = captain_by_gw.get(gw, 4.0) if pid == XI[0] else 2.0
        rows.append(row)
    return chips.ChipEngine(BOOTSTRAP, [], pd.DataFrame(rows))


def _decide(engine, event):
    return engine.evaluate(event, {"chips": OTHERS_SPENT}, xi_ids=XI, bench_ids=BENCH,
                           captain_id=XI[0], squad_ids=XI + BENCH)


def test_the_gw7_case_is_not_reported_as_a_hold():
    assert chips.effective_threshold(BOOTSTRAP, "3xc", 7) == pytest.approx(8.4)
    assert chips.effective_threshold(BOOTSTRAP, "3xc", 16) == pytest.approx(6.6)

    engine = _engine({7: 7.7, 16: 7.6})
    d = _decide(engine, 7)
    assert d.chip is None
    assert "holding" not in d.reason, (
        f"{d.reason!r}: GW16 entered the plan on a bar GW7 does not face"
    )
    assert engine.last_plan.assignments == {}
    assert "under its 8.4 bar" in d.reason, "the bar quoted must be this week's"


def test_a_week_that_clears_todays_bar_is_still_held_for():
    """The real hold - a better week ahead that is worth committing to - survives."""
    engine = _engine({7: 7.7, 12: 11.0})
    d = _decide(engine, 7)
    assert d.chip is None and "holding 3xc for GW12" in d.reason


def test_this_week_still_plays_when_it_clears_its_bar():
    engine = _engine({7: 9.0, 16: 7.6})
    assert _decide(engine, 7).chip == "3xc"
