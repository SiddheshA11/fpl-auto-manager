"""
The ownership tilt on the squad optimiser.

The tilt exists to trade expected points for rank variance, so the tests that
matter are the ones proving it only does that: that it is exactly inert at zero,
that its sign means what the docstring claims, and above all that it re-weights
players without ever inverting their order on points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optimizer as O


def _pool(ownerships: list[float], values: list[float]) -> pd.DataFrame:
    """
    A pool large enough to build a legal 15 from, with ownership and value set
    per player. Teams are spread so the three-per-club cap never binds.
    """
    rows = []
    pid = 0
    # Enough of every position that the solver always has a choice to make.
    for position, count in [(1, 8), (2, 20), (3, 20), (4, 12)]:
        for k in range(count):
            rows.append({
                "id": pid,
                "web_name": f"p{pid}",
                "position": position,
                "team": pid % 20 + 1,
                "cost": 4.5,
                "selected_by_percent": ownerships[pid % len(ownerships)],
                "xp_horizon": values[pid % len(values)],
                "xp_next": values[pid % len(values)] / 5.0,
            })
            pid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def pool():
    # Ownership and value are deliberately uncorrelated, so any shift in the
    # chosen squad's mean ownership is caused by the tilt and nothing else.
    rng = np.random.default_rng(0)
    n = 60
    return _pool(
        ownerships=list(rng.uniform(0.5, 70.0, n)),
        values=list(rng.uniform(2.0, 20.0, n)),
    )


def test_zero_weight_is_exactly_the_pure_points_objective(pool):
    """The default must not perturb the existing behaviour at all."""
    plain = O.SquadOptimizer(pool, "xp_horizon", "xp_next").build_squad(100.0)
    tilted = O.SquadOptimizer(pool, "xp_horizon", "xp_next", ownership_weight=0.0).build_squad(100.0)
    assert set(plain.squad["id"]) == set(tilted.squad["id"])
    assert plain.squad_xp == pytest.approx(tilted.squad_xp)


def test_sign_controls_the_direction_of_the_tilt(pool):
    """Positive tracks the field, negative buys differentials."""
    def mean_ownership(weight: float) -> float:
        sol = O.SquadOptimizer(pool, "xp_horizon", "xp_next", ownership_weight=weight).build_squad(100.0)
        return pd.to_numeric(sol.squad["selected_by_percent"]).mean()

    assert mean_ownership(-0.4) < mean_ownership(0.0) < mean_ownership(0.4)


def test_tilt_never_inverts_the_ordering_on_points(pool):
    """
    Regression. The tilt was first written as a penalty added to the squad
    block. Points reach that block scaled by `bench_weight`, so at any weight
    above roughly 0.15 the penalty outgrew the value it was adjusting and the
    coefficient flipped sign - making a higher-scoring player *worse* to own
    than a lower-scoring one at the same ownership. The solver duly filled its
    bench with near-worthless players. Owning a better player must never cost
    the objective, whatever the tilt.
    """
    for weight in (-0.6, -0.3, 0.3, 0.6, 0.9):
        opt = O.SquadOptimizer(pool, "xp_horizon", "xp_next", ownership_weight=weight)
        objective, _ = opt._base_program(budget=100.0)
        n = opt.n
        for block in (objective[:n], objective[n : 2 * n], objective[2 * n :]):
            assert np.all(block <= 1e-9), (
                f"weight {weight}: owning a player carries a positive cost, so the "
                f"solver is being paid to pick worse players"
            )

        # Same statement from the other side: among players who share an
        # ownership level, the objective must still rank them by points.
        for eo in np.unique(np.round(opt.ownership, 6)):
            same = np.isclose(opt.ownership, eo)
            if same.sum() < 2:
                continue
            order_by_value = np.argsort(opt.value[same])
            coeffs = objective[:n][same][order_by_value]
            assert np.all(np.diff(coeffs) <= 1e-9), (
                f"weight {weight}: at ownership {eo:.3f} the objective prefers the "
                f"lower-scoring player"
            )


def test_reported_points_stay_untilted(pool):
    """
    The tilt belongs to the objective, not to the reporting. `squad_xp` is what
    gets logged, compared against chip thresholds and sent to the owner, so it
    has to remain honest expected points.
    """
    sol = O.SquadOptimizer(pool, "xp_horizon", "xp_next", ownership_weight=0.5).build_squad(100.0)
    assert sol.squad_xp == pytest.approx(sol.squad["xp_horizon"].sum())
    assert sol.xi_xp == pytest.approx(sol.xi["xp_horizon"].sum())


def test_missing_ownership_column_falls_back_to_pure_points(pool):
    """A pool without ownership data must still solve, not raise."""
    bare = pool.drop(columns=["selected_by_percent"])
    opt = O.SquadOptimizer(bare, "xp_horizon", "xp_next", ownership_weight=0.5)
    assert np.allclose(opt.risk_multiplier, 1.0), "no ownership data means no tilt"
    assert len(opt.build_squad(100.0).squad) == O.SQUAD_SIZE


def test_extreme_weight_cannot_make_value_negative(pool):
    """
    A weight above 1.0 would otherwise flip low-ownership players to negative
    value, which reads as "prefer players who score fewer points" - never the
    intended meaning of any setting of this knob.
    """
    opt = O.SquadOptimizer(pool, "xp_horizon", "xp_next", ownership_weight=3.0)
    assert np.all(opt.risk_multiplier >= 0.0)


def test_captaincy_is_never_tilted_by_ownership(pool):
    """
    The armband is decided on points alone, at every setting of the knob.

    The exposure that matters for a captain is *captaincy* effective ownership,
    which FPL does not publish. Squad ownership is a proxy that is wrong in a
    known direction - widely-owned defenders are rarely captained - and it
    once picked a 5.12 xP defender over a 5.49 xP midfielder on that basis.
    """
    for weight in (-0.9, -0.3, 0.0, 0.3, 0.9):
        sol = O.SquadOptimizer(pool, "xp_horizon", "xp_next",
                               ownership_weight=weight).build_squad(100.0)
        chosen = sol.xi.loc[sol.xi["id"] == sol.captain, "xp_next"].iloc[0]
        assert chosen == pytest.approx(sol.xi["xp_next"].max()), (
            f"weight {weight}: captained a player who is not the best starter on points"
        )
        vice = sol.xi.loc[sol.xi["id"] == sol.vice_captain, "xp_next"].iloc[0]
        assert vice <= chosen, "the vice must not outscore the captain"
        assert sol.vice_captain != sol.captain
