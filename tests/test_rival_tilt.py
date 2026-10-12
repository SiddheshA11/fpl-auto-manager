"""
Field ownership and captaincy as optimiser inputs.

Two claims are load-bearing here and both are measured, not assumed:

  - the mini-league field is a *wider* distribution than the template, so the
    same weight applied to it is a stronger tilt. Measured on GW2 of 2026-27
    over 45 rivals, sd(1 - 2*ownership) is 0.171 against the template's 0.132,
    a ratio of 1.295. Carrying +0.20 across unchanged would silently tilt at
    an effective +0.26.
  - captaincy is a third distribution, not a scaled squad ownership. Joao
    Pedro: owned by 86.7% of the field, captained by 8.9%. Six players took
    the armband across 45 managers.

No test here touches the network. rivals.py is credential-free by design, but a
suite that reached the FPL API would be flaky and would hammer 45 endpoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optimizer as O
import rivals


def _pool(n=30):
    """A legal, solvable pool: enough of each position and a spread of clubs."""
    rows = []
    pid = 1
    for pos, count in ((1, 4), (2, 10), (3, 10), (4, 6)):
        for k in range(count):
            rows.append({
                "id": pid, "web_name": f"p{pid}", "position": pos,
                "team": pid % 8, "team_name": f"t{pid % 8}",
                "cost": 4.5 + (k % 5), "xp": 3.0 + (k % 7) * 0.4,
                "selected_by_percent": float((pid * 7) % 60),
                "field_ownership": float((pid * 11) % 90),
                "field_captaincy": 0.0,
                "status": "a",
            })
            pid += 1
    return pd.DataFrame(rows)


class _Ownership(rivals.FieldOwnership):
    pass


def test_attach_adds_both_columns_as_percentages():
    fo = rivals.FieldOwnership(event=2, managers=45,
                               squad={1: 0.867, 2: 0.60}, captain={2: 0.289})
    scored = pd.DataFrame({"id": [1, 2, 3], "web_name": ["a", "b", "c"]})

    out = rivals.attach(scored, fo)

    assert out[rivals.FIELD_OWNERSHIP_COL].tolist() == pytest.approx([86.7, 60.0, 0.0])
    assert out[rivals.FIELD_CAPTAINCY_COL].tolist() == pytest.approx([0.0, 28.9, 0.0])


def test_attach_does_not_mutate_the_frame_it_was_given():
    fo = rivals.FieldOwnership(event=2, managers=45, squad={1: 0.5}, captain={})
    scored = pd.DataFrame({"id": [1], "web_name": ["a"]})
    rivals.attach(scored, fo)
    assert rivals.FIELD_OWNERSHIP_COL not in scored.columns


def test_a_player_no_rival_owns_is_zero_not_missing():
    """
    In a 45-manager league most of the 622 players are owned by nobody. That
    is the honest number and it is what makes this distribution wide.
    """
    fo = rivals.FieldOwnership(event=2, managers=45, squad={1: 1.0}, captain={})
    out = rivals.attach(pd.DataFrame({"id": [1, 2]}), fo)
    assert out[rivals.FIELD_OWNERSHIP_COL].tolist() == pytest.approx([100.0, 0.0])
    assert not out[rivals.FIELD_OWNERSHIP_COL].isna().any()


# ------------------------------------------------------------- captain tilt

def test_captaincy_is_untilted_by_default():
    """
    The machinery landing must not change what the bot does. The weight is
    0.0 until somebody measures the right value.
    """
    opt = O.SquadOptimizer(_pool(), value_col="xp", captain_col="xp")
    assert opt.captain_ownership_weight == 0.0
    assert np.allclose(opt.captain_risk_multiplier, 1.0)


def test_the_captain_tilt_reads_captaincy_not_squad_ownership():
    """
    The whole point. Squad ownership is a proxy that is wrong in a known
    direction, and using it here is the bug this replaces.
    """
    pool = _pool()
    pool["field_ownership"] = 90.0      # everyone heavily owned
    pool["field_captaincy"] = 0.0       # nobody captained
    pool.loc[pool.index[0], "field_captaincy"] = 80.0

    opt = O.SquadOptimizer(pool, value_col="xp", captain_col="xp",
                           captain_ownership_weight=0.30)

    # Only the one heavily-captained player is lifted; the rest are pushed down.
    assert opt.captain_risk_multiplier[0] > 1.0
    assert np.all(opt.captain_risk_multiplier[1:] < 1.0)


def test_a_missing_captaincy_column_disables_the_tilt_loudly(caplog):
    pool = _pool().drop(columns=["field_captaincy"])
    opt = O.SquadOptimizer(pool, value_col="xp", captain_col="xp",
                           captain_ownership_weight=0.30)
    assert opt.captain_ownership_weight == 0.0
    assert np.allclose(opt.captain_risk_multiplier, 1.0)
    assert any("captain_ownership_weight" in r.getMessage() for r in caplog.records), (
        "a requested tilt that cannot be applied must say so; silently "
        "optimising at 0.0 is how the ownership_weight bug stayed live"
    )


def test_the_squad_tilt_and_the_captain_tilt_are_independent():
    """
    They are different distributions, so one must not move when the other is
    set. A shared multiplier would reintroduce the proxy silently.
    """
    pool = _pool()
    a = O.SquadOptimizer(pool, value_col="xp", captain_col="xp",
                         ownership_weight=0.20, captain_ownership_weight=0.0)
    b = O.SquadOptimizer(pool, value_col="xp", captain_col="xp",
                         ownership_weight=0.20, captain_ownership_weight=0.40)

    assert np.allclose(a.risk_multiplier, b.risk_multiplier), "squad tilt moved"
    assert not np.allclose(a.captain_risk_multiplier, b.captain_risk_multiplier)


def test_the_captain_tilt_can_change_the_armband():
    """
    Guards against shipping an inert knob: at a large enough weight the tilt
    must actually move the pick, or it is decoration.
    """
    pool = _pool()
    pool["field_captaincy"] = 0.0
    # Make the second-best player the field's captain by a wide margin.
    order = pool.sort_values("xp", ascending=False).index
    best, runner_up = order[0], order[1]
    pool.loc[runner_up, "field_captaincy"] = 95.0

    untilted = O.SquadOptimizer(pool, value_col="xp", captain_col="xp").build_squad(200.0)
    tilted = O.SquadOptimizer(pool, value_col="xp", captain_col="xp",
                              captain_ownership_weight=0.90).build_squad(200.0)

    assert untilted.captain != tilted.captain, (
        "a 0.90 captain tilt toward a 95%-captained player changed nothing; "
        "the weight is not reaching the objective"
    )
    assert tilted.captain == int(pool.loc[runner_up, "id"])


# ---------------------------------------------------- falling back safely
#
# tilt_inputs sits on the weekly run's main path and makes one request per
# rival. It must never be able to fail the run: a squad submitted on the
# template is worth vastly more than no squad at all.


def test_an_unreachable_api_falls_back_to_the_template(monkeypatch):
    """The conftest guard already makes every fetch raise; this asserts on it."""
    scored = pd.DataFrame({"id": [1, 2], "selected_by_percent": [10.0, 20.0]})

    tilt = rivals.tilt_inputs(scored, entry_id=5413589, event=2)

    assert tilt.source == "template"
    assert tilt.ownership_col == "selected_by_percent"
    assert tilt.captain_weight == 0.0, "no captain rates means no captain tilt"


def test_a_gameweek_with_no_rival_picks_falls_back(monkeypatch):
    """
    /entry/{id}/event/{gw}/picks/ publishes nothing before a deadline passes,
    so the first run of a season has no field to aim at.
    """
    monkeypatch.setattr(rivals, "field_ownership",
                        lambda *a, **k: rivals.FieldOwnership(event=1, managers=0))
    tilt = rivals.tilt_inputs(pd.DataFrame({"id": [1], "selected_by_percent": [5.0]}),
                              entry_id=1, event=1)
    assert tilt.source == "template"


def test_field_ownership_is_used_when_it_is_available(monkeypatch):
    """Guards the other direction: a permanently-falling-back tilt is inert."""
    monkeypatch.setattr(
        rivals, "field_ownership",
        lambda *a, **k: rivals.FieldOwnership(event=2, managers=45,
                                              squad={1: 0.844}, captain={1: 0.533}),
    )
    scored = pd.DataFrame({"id": [1, 2], "selected_by_percent": [68.5, 10.0]})

    tilt = rivals.tilt_inputs(scored, entry_id=5413589, event=2)

    assert tilt.source == "field"
    assert tilt.ownership_col == rivals.FIELD_OWNERSHIP_COL
    assert tilt.scored[rivals.FIELD_OWNERSHIP_COL].tolist() == pytest.approx([84.4, 0.0])
    assert tilt.scored[rivals.FIELD_CAPTAINCY_COL].tolist() == pytest.approx([53.3, 0.0])


def test_the_field_weight_is_not_the_template_weight(monkeypatch):
    """
    The trap this whole exercise exists to avoid. The field is the wider
    distribution - measured spread ratio 1.299 (GW1) and 1.301 (GW2) - so
    carrying +0.20 across would tilt at an effective +0.26, harder rather than
    merely better aimed.
    """
    import config

    assert config.FIELD_OWNERSHIP_WEIGHT < config.OWNERSHIP_WEIGHT, (
        "the field weight must be softer than the template weight to tilt with "
        "the same force; see measure_rival_tilt.py"
    )
    assert config.FIELD_OWNERSHIP_WEIGHT == pytest.approx(0.154, abs=0.02)
