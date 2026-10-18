"""
The harness statistics, checked against Monte Carlo rather than against the
formulas they implement.

An SE is right if it matches the actual spread of the mean over repeated
draws from a process with the dependence it claims to handle. Each test
builds that process - overlapping windows that share shocks, seasons with
their own level - and compares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import replay_stats as RS


def _windows(rng, seasons=5, per_season=28, overlap=8, season_sd=0.0, noise_sd=10.0):
    """
    Windows as the harness makes them: window i of a season sums `overlap`
    consecutive per-gameweek shocks, so neighbours share most of theirs.
    """
    rows = []
    for s in range(seasons):
        level = rng.normal(0, season_sd)
        shocks = rng.normal(0, noise_sd, per_season + overlap)
        for i in range(per_season):
            rows.append({"season": f"s{s}", "start": i + 2,
                         "diff": level + shocks[i:i + overlap].sum() / np.sqrt(overlap)})
    return pd.DataFrame(rows)


def _true_sd_of_mean(make, reps=400, seed=1):
    rng = np.random.default_rng(seed)
    return float(np.std([make(rng)["diff"].mean() for _ in range(reps)], ddof=1))


def test_overlapping_windows_make_an_iid_se_badly_optimistic():
    """Why the SE must account for overlap at all."""
    truth = _true_sd_of_mean(lambda r: _windows(r))
    d = _windows(np.random.default_rng(7))
    assert d["diff"].std(ddof=1) / np.sqrt(len(d)) < 0.6 * truth


@pytest.mark.parametrize("kw", [{}, {"season_sd": 3.0}, {"season_sd": 6.0, "overlap": 1}])
def test_the_cluster_se_matches_the_true_spread(kw):
    """
    Overlapping windows, season levels, both. The within-season block
    bootstrap the plan proposed measured 0.21-0.66 of the truth on these same
    processes and was dropped; this is the method that survived.
    """
    make = lambda r: _windows(r, **kw)  # noqa: E731
    truth = _true_sd_of_mean(make)
    ses = [RS.se_cluster(make(np.random.default_rng(k)), "diff")[0] for k in range(60)]
    # With five clusters any one estimate is noisy; its typical size is not.
    assert np.sqrt(np.mean(np.square(ses))) == pytest.approx(truth, rel=0.2)


def test_the_cluster_test_has_four_degrees_of_freedom():
    assert RS.se_cluster(_windows(np.random.default_rng(0)), "diff")[1] == 4


def test_calibration_finds_uniform_p_values_on_a_true_null():
    rng = np.random.default_rng(3)
    pairs = [_windows(rng).assign(diff_terminal=lambda x: x["diff"], flipped=True)
             for _ in range(60)]
    cal = RS.calibrate(pairs, "diff")
    assert cal["ks"] < 0.2
    assert cal["reject"] < 0.15
    assert cal["t_sd_expected"] == pytest.approx(np.sqrt(2))


def test_calibration_catches_a_test_that_ignores_overlap():
    """The check must be able to fail: halve the SE and it should."""
    rng = np.random.default_rng(4)
    pairs = [_windows(rng) for _ in range(60)]
    honest = RS.calibrate(pairs, "diff")
    real = RS.se_cluster
    try:
        RS.se_cluster = lambda d, col: (real(d, col)[0] / 2, real(d, col)[1])
        broken = RS.calibrate(pairs, "diff")
    finally:
        RS.se_cluster = real
    assert broken["reject"] > honest["reject"] + 0.15
    assert broken["ks"] > 0.2


def test_placebo_pairs_subtract_seed_from_seed_window_by_window():
    rows = []
    for seed, shift in ((0, 3.0), (1, 1.0), (2, 0.0), (3, 0.0)):
        for start in (2, 3):
            rows.append({"arm": f"noise=0.02@{seed}", "season": "s", "start": start,
                         "diff": shift + start, "diff_terminal": 2 * shift,
                         "flipped": seed == 0})
    pairs = RS.placebo_pairs(pd.DataFrame(rows))
    assert len(pairs) == 2
    assert list(pairs[0]["diff"]) == [2.0, 2.0]
    assert list(pairs[0]["diff_terminal"]) == [4.0, 4.0]
    assert pairs[0]["flipped"].all() and not pairs[1]["flipped"].any()


def test_rule_two_keeps_terminal_value_unless_it_costs_too_much_variance():
    base = pd.DataFrame({"season": ["a"] * 4, "diff": [1.0, -1.0, 1.0, -1.0]})
    cheap = [base.assign(diff_terminal=base["diff"] * 1.2)]
    dear = [base.assign(diff_terminal=base["diff"] * 1.3)]
    assert RS.choose_outcome(cheap)[0] == "diff_terminal"
    assert RS.choose_outcome(dear)[0] == "diff"


def test_flip_decomposition_separates_how_often_from_how_much():
    d = pd.DataFrame({"diff": [0, 0, 0, 10, -4, 6], "flipped": [False] * 3 + [True] * 3,
                      "flip_gws": [0, 0, 0, 1, 1, 3]})
    f = RS.flip_decomposition(d, "diff")
    assert f["p_flip"] == 0.5 and f["unflipped_all_zero"]
    assert f["mean_given_flip"] == pytest.approx(4.0)
    assert f["abs_step_single_flip"] == pytest.approx(7.0)
