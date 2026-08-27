"""
The offline harnesses must score the model production actually runs.

`recent_minutes` was supplied by `manager.py` alone. `backtest.py` and
`simulate.py` both built XPModel without it, so `_blend_recent_minutes`
returned at its first line and the start rate was never touched - meaning every
offline measurement in the repo evaluated a model missing the input HANDOFF
credits with the largest improvement ever made. Measured on 2025-26 GW10-38,
the harness understated the model by 0.05 R2 (0.2570 against 0.3093).

These assert the model is HANDED the lags through a real `run_backtest` call,
rather than that the helper works in isolation. A test that exercises a helper
does not prove the helper is reached; that has already made four separate fixes
deletable from production with the suite green.
"""
from __future__ import annotations

import pandas as pd
import pytest

import backtest
import simulate


def test_the_lag_window_is_dense_and_most_recent_first():
    """
    `event/{gw}/live/` returns a row for every player, so a non-appearance is a
    real 0.0. A missing row here means the same and must not be dropped -
    shortening one player's window silently reweights his lag average.
    """
    df = pd.DataFrame({
        "element": [1, 1, 1, 2, 1],
        "GW":      [7, 8, 9, 7, 9],       # player 2 appears only in GW7
        "minutes": [10, 20, 30, 45, 15],  # player 1 has a double in GW9
    })
    # window is GW9, GW8, GW7 - most recent first
    got = backtest.recent_minutes_for(df, upto_gw=10, depth=3)
    assert got[1] == [45.0, 20.0, 10.0], "most recent first; double summed"
    assert got[2] == [0.0, 0.0, 45.0], "non-appearances are 0.0, not missing"
    assert len(got[1]) == len(got[2]) == 3


def test_no_history_before_the_first_gameweek():
    df = pd.DataFrame({"element": [1], "GW": [1], "minutes": [90]})
    assert backtest.recent_minutes_for(df, upto_gw=1) == {}


def test_run_backtest_hands_the_model_its_recent_minutes(monkeypatch):
    """
    The regression itself, recorded through a real run_backtest call: whatever
    else changes, the offline harness must not construct XPModel without lags.
    """
    seen = []
    real = backtest.X.XPModel

    class Recording(real):
        def __init__(self, *a, **kw):
            seen.append(kw.get("recent_minutes"))
            super().__init__(*a, **kw)

    monkeypatch.setattr(backtest.X, "XPModel", Recording)
    backtest.run_backtest("2025-26", start_gw=12, end_gw=13, horizon=1)

    assert seen, "run_backtest built no model"
    for got in seen:
        assert got, "XPModel was constructed without recent_minutes"
        window = next(iter(got.values()))
        assert len(window) > 1, f"lag window too shallow: {window}"


def test_simulate_uses_the_same_lags_as_the_backtest():
    """Both harnesses must agree, or they measure two different models."""
    assert simulate.recent_minutes_for is backtest.recent_minutes_for
