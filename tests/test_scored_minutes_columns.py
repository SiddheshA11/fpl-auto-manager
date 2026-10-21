"""
The minutes columns of expected_points describe the gameweek xp_next does.

They were built from minutes_model() with no event, which returns the base
status flag and skips everything event-specific - dated news, doubt decay,
card risk. The xP itself was right, because the per-fixture path passes the
event; only the reported columns were wrong. Those columns are what the
ledger records and what calibrate.py and deadline_check.py read.

Reproduced on the snapshot: a doubtful player whose news says he is out
until January scored xp_next 0.0, while the frame - and so the GW8 ledger -
said availability 0.75, p_start 0.72, exp_minutes 59.
"""
from __future__ import annotations

import copy

import pytest

import priors
import xp_model as X
from test_pipeline import _snapshot


@pytest.fixture(scope="module")
def out_until_january():
    b = copy.deepcopy(_snapshot("bootstrap-static"))
    f = _snapshot("fixtures")
    el = max((e for e in b["elements"] if e["element_type"] == 3),
             key=lambda e: float(e.get("minutes") or 0))
    el.update(status="d", chance_of_playing_next_round=75, chance_of_playing_this_round=75,
              news="Hamstring injury - Expected back 01 Jan", news_added="2026-08-10T12:00:00Z")
    tc = {t["code"]: t["name"] for t in b["teams"]}
    model = X.XPModel(b, f, priors.build_priors(current_team_codes=tc), X.ModelConfig(horizon=5))
    events = X.next_events(b, 5)
    return model, events, el["id"], model.expected_points(events).set_index("id")


def test_the_columns_are_the_next_gameweeks_not_the_base_flags(out_until_january):
    model, events, pid, out = out_until_january
    row = model.players.index[model.players["id"] == pid][0]
    scored_with = model.minutes_model(events[0]).loc[row]
    for col in ("availability", "p_start", "exp_minutes"):
        assert out.loc[pid, col] == pytest.approx(float(scored_with[col])), col


def test_a_player_the_model_scores_at_zero_is_not_recorded_as_playing(out_until_january):
    _, _, pid, out = out_until_january
    assert out.loc[pid, "xp_next"] == pytest.approx(0.0)
    assert out.loc[pid, "availability"] == 0.0
    assert out.loc[pid, "exp_minutes"] == 0.0


def test_the_case_discriminates(out_until_january):
    """The event-less figures really do differ here, so the tests above can fail."""
    model, events, pid, _ = out_until_january
    row = model.players.index[model.players["id"] == pid][0]
    assert model.minutes_model().loc[row, "availability"] == pytest.approx(0.75)
