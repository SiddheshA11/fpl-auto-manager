"""
Squad and transfer optimisation as a mixed-integer program.

The thing the old greedy optimiser could not do is trade off across the whole
squad at once. Selling two mid-price midfielders to fund one premium is where
a lot of real transfer gain lives, and a loop that walks the weakest player and
swaps him one-for-one can never find it. Nor can a position-by-position greedy
fill build a legal squad: filling goalkeepers first at the top of the price
list spends the budget before it reaches the forwards.

Both problems are the same integer program, solved here with HiGHS via
scipy.optimize.milp - already a dependency through scipy, so this adds nothing
to the install.

Three decisions are made jointly, because they are not separable: which 15 to
own, which 11 to start, and who to captain. Picking a squad on total xP and
only then choosing an XI overrates squads whose points sit on the bench.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

logger = logging.getLogger("fpl_auto.optimizer")

SQUAD_SIZE = 15
XI_SIZE = 11
SQUAD_BY_POSITION = {1: 2, 2: 5, 3: 5, 4: 3}
# Legal starting XI ranges. One keeper always; the rest bounded by FPL's rules.
XI_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}
MAX_PER_CLUB = 3
TRANSFER_HIT = 4.0

# What a bench place is worth, as a fraction of the player's own xP. Bench
# points only materialise through autosubs, so this is deliberately low: it is
# enough to break ties toward a bench that can cover, without tempting the
# solver to spend real money on players who will not start.
DEFAULT_BENCH_WEIGHT = 0.15

# Option value of *not* using a free transfer. Unused transfers roll over (the
# game allows banking up to five), so a move that gains nothing is a real loss:
# it spends flexibility that could fund a two-transfer swing later. Without this
# the solver is indifferent to a zero-gain transfer and will happily churn the
# squad for nothing.
FREE_TRANSFER_VALUE = 0.3

# Ownership tilt. Zero reproduces the pure expected-points objective exactly.
#
# Adding ownership as a *points* term is a mistake worth naming, because it is
# the obvious thing to reach for: rank is driven by (my points - field points),
# and the field's expected score is a constant, so it drops straight out of the
# argmax. Maximising the mean differential is algebraically identical to
# maximising points. Ownership only bites through variance.
#
# Exposure to a player is (owned - EO). Squaring it and using owned ∈ {0,1}:
#
#     exposure² = EO² + owned·(1 - 2·EO)
#
# which is linear in the decision variable, so it drops into the MILP without
# any extra columns. The EO² term is a constant and is dropped.
#
# It enters as a multiplier on the player's expected points rather than as a
# separate additive penalty:
#
#     adjusted value = value · (1 - weight·(1 - 2·EO))
#
# The additive form is the tempting one and it is wrong. Points reach the
# objective at full rate for a starter and at `bench_weight` for a substitute,
# so a penalty applied at full rate to squad membership exceeds a bench
# player's entire contribution - and the solver responds by benching worthless
# players to dodge it, which is how it produced a bench of 1.3-xP filler.
# Scaling keeps the tilt proportional to the points actually at stake, so it
# can never dominate them.
#
# Positive weight minimises exposure: it rewards owning a highly-owned player
# and penalises owning a differential, tracking the field to protect rank.
# Negative weight does the reverse and buys variance, which is what overtaking
# a small league actually requires. See `ownership_weight` on SquadOptimizer.
OWNERSHIP_WEIGHT = 0.0


@dataclass
class SquadSolution:
    squad: pd.DataFrame                  # the 15
    xi: pd.DataFrame                     # the 11
    bench: pd.DataFrame                  # ordered
    captain: int                         # element id
    vice_captain: int
    total_cost: float
    xi_xp: float
    squad_xp: float
    objective: float
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hits: int = 0
    status: str = "optimal"


class SquadOptimizer:
    """
    Build or improve a squad against an expected-points column.

    `players` must carry: id, position, team, cost, and the objective column.
    """

    def __init__(
        self,
        players: pd.DataFrame,
        value_col: str = "xp_horizon",
        captain_col: str | None = None,
        bench_weight: float = DEFAULT_BENCH_WEIGHT,
        ownership_weight: float = OWNERSHIP_WEIGHT,
        ownership_col: str = "selected_by_percent",
    ):
        required = {"id", "position", "team", "cost", value_col}
        missing = required - set(players.columns)
        if missing:
            raise ValueError(f"players frame missing columns: {sorted(missing)}")

        # Anyone who cannot be selected is dropped before the solve rather than
        # constrained to zero: it keeps the program smaller and avoids the
        # solver wasting effort on players the game will not accept.
        self.players = players[players["cost"] > 0].reset_index(drop=True).copy()
        self.value_col = value_col
        # The captain choice should be driven by a single gameweek even when the
        # squad is built over a horizon; captaincy does not carry forward.
        self.captain_col = captain_col if captain_col in self.players.columns else value_col
        self.bench_weight = bench_weight

        self.n = len(self.players)
        self.value = self.players[value_col].fillna(0.0).to_numpy(dtype=float)
        self.captain_value = self.players[self.captain_col].fillna(0.0).to_numpy(dtype=float)
        self.cost = self.players["cost"].to_numpy(dtype=float)
        self.position = self.players["position"].to_numpy(dtype=int)
        self.team = self.players["team"].to_numpy(dtype=int)
        self.ids = self.players["id"].to_numpy(dtype=int)

        # `selected_by_percent` arrives from the API as a string, and is a
        # percentage rather than a fraction. A missing column is not an error:
        # it just means no tilt is available, so the objective stays pure xP.
        self.ownership_weight = float(ownership_weight)
        if ownership_col in self.players.columns:
            eo = pd.to_numeric(self.players[ownership_col], errors="coerce")
            self.ownership = (eo.fillna(0.0) / 100.0).clip(0.0, 1.0).to_numpy(dtype=float)
        else:
            if self.ownership_weight:
                logger.warning(
                    "ownership_weight=%.2f requested but column %r is absent; "
                    "falling back to a pure expected-points objective",
                    self.ownership_weight, ownership_col,
                )
            self.ownership = np.zeros(self.n, dtype=float)
            self.ownership_weight = 0.0

        # Clipped at zero so a weight above 1.0 cannot invert the sign of a
        # player's value and turn the objective into "prefer players who score
        # fewer points", which is never what any setting of this knob means.
        self.risk_multiplier = np.clip(
            1.0 - self.ownership_weight * (1.0 - 2.0 * self.ownership), 0.0, None
        )

    # ---------- program construction ----------
    #
    # Variable layout, all binary: [ squad(n) | start(n) | captain(n) ]

    def _base_program(self, budget: float, include_budget: bool = True) -> tuple[np.ndarray, list[LinearConstraint]]:
        n = self.n
        cons: list[LinearConstraint] = []

        def block(squad=None, start=None, cap=None) -> np.ndarray:
            row = np.zeros(3 * n)
            if squad is not None:
                row[:n] = squad
            if start is not None:
                row[n : 2 * n] = start
            if cap is not None:
                row[2 * n :] = cap
            return row

        ones = np.ones(n)

        cons.append(LinearConstraint(block(squad=ones), SQUAD_SIZE, SQUAD_SIZE))
        cons.append(LinearConstraint(block(start=ones), XI_SIZE, XI_SIZE))
        cons.append(LinearConstraint(block(cap=ones), 1, 1))

        for pos, count in SQUAD_BY_POSITION.items():
            cons.append(LinearConstraint(block(squad=(self.position == pos).astype(float)), count, count))

        for pos, (lo, hi) in XI_BOUNDS.items():
            cons.append(LinearConstraint(block(start=(self.position == pos).astype(float)), lo, hi))

        # A starter must be owned, and a captain must be starting. Expressed in
        # one matrix each rather than n separate constraints, which keeps the
        # problem small enough for HiGHS to solve in well under a second.
        eye = np.eye(n)
        start_implies_squad = np.hstack([-eye, eye, np.zeros((n, n))])
        cons.append(LinearConstraint(start_implies_squad, -np.inf, np.zeros(n)))

        cap_implies_start = np.hstack([np.zeros((n, n)), -eye, eye])
        cons.append(LinearConstraint(cap_implies_start, -np.inf, np.zeros(n)))

        for club in np.unique(self.team):
            cons.append(LinearConstraint(block(squad=(self.team == club).astype(float)), 0, MAX_PER_CLUB))

        # Omitted for transfers, where the cost coefficients differ per player:
        # an owned player releases only his selling price, and a kept one costs
        # nothing at all.
        if include_budget:
            cons.append(LinearConstraint(block(squad=self.cost), 0, budget))

        # Maximise: starters at full value, bench at a fraction, captain doubled.
        # The ownership tilt scales squad and XI membership identically, so it
        # re-weights players against each other without disturbing the balance
        # between starting and benching any one of them.
        #
        # Captaincy is deliberately left untilted. The exposure that matters for
        # an armband is *captaincy* effective ownership, not squad ownership,
        # and the two come apart badly: a defender owned by 29% of managers may
        # be captained by almost none of them, while a midfielder owned by 49%
        # takes the armband from a quarter of the field. FPL publishes the
        # former and not the latter, so tilting the captain on squad ownership
        # applies a proxy that is wrong in a known direction - it read Gabriel
        # at 5.12 xP as a better captain than Bruno Fernandes at 5.49 purely
        # because fewer people own him. Captaincy doubles a score and is the
        # highest-leverage call of the week; it gets made on points.
        value = self.value * self.risk_multiplier
        objective = np.concatenate([
            -value * self.bench_weight,                           # owned (bench share)
            -value * (1.0 - self.bench_weight),                   # promoted to XI
            -self.captain_value,                                  # captaincy doubles the score
        ])
        return objective, cons

    def _solve(self, objective: np.ndarray, cons: list[LinearConstraint], extra_cols: int = 0):
        n = self.n
        size = 3 * n + extra_cols
        integrality = np.ones(size)
        lower = np.zeros(size)
        upper = np.ones(size)
        if extra_cols:
            # Trailing columns are the transfer-count surplus: integer, unbounded.
            upper[3 * n :] = np.inf

        res = milp(
            c=objective,
            constraints=cons,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            options={"time_limit": 60, "mip_rel_gap": 1e-4},
        )
        return res

    # ---------- results ----------

    def _build_solution(self, res, budget: float) -> SquadSolution:
        n = self.n
        x = np.round(res.x[:n]).astype(bool)
        y = np.round(res.x[n : 2 * n]).astype(bool)
        c = np.round(res.x[2 * n : 3 * n]).astype(bool)

        squad = self.players[x].copy()
        xi = self.players[y].copy()
        bench = self.players[x & ~y].copy()

        captain_idx = int(np.argmax(c))
        captain_id = int(self.ids[captain_idx])

        # Vice-captain is not a solver decision: it only pays out if the captain
        # does not play, so it is simply the best remaining starter.
        vice_pool = xi[xi["id"] != captain_id].sort_values(self.captain_col, ascending=False)
        vice_id = int(vice_pool.iloc[0]["id"]) if len(vice_pool) else captain_id

        # Bench order: keeper first (he can only replace the keeper), then by
        # descending chance of actually returning points.
        bench_gk = bench[bench["position"] == 1]
        bench_out = bench[bench["position"] != 1].sort_values(self.value_col, ascending=False)
        bench = pd.concat([bench_gk, bench_out])

        return SquadSolution(
            squad=squad.sort_values(["position", self.value_col], ascending=[True, False]),
            xi=xi.sort_values(["position", self.value_col], ascending=[True, False]),
            bench=bench,
            captain=captain_id,
            vice_captain=vice_id,
            total_cost=float(squad["cost"].sum()),
            xi_xp=float(xi[self.value_col].sum()),
            squad_xp=float(squad[self.value_col].sum()),
            objective=float(-res.fun),
            status=res.message,
        )

    # ---------- public entry points ----------

    def build_squad(self, budget: float = 100.0) -> SquadSolution:
        """Best legal 15 from scratch, for a new season or a wildcard."""
        objective, cons = self._base_program(budget)
        res = self._solve(objective, cons)
        if not res.success:
            raise RuntimeError(f"squad optimisation failed: {res.message}")
        sol = self._build_solution(res, budget)
        logger.info(
            "built squad: £%.1fm, XI xP %.2f, captain %s",
            sol.total_cost, sol.xi_xp,
            self.players.loc[self.players["id"] == sol.captain, "web_name"].iloc[0]
            if "web_name" in self.players.columns else sol.captain,
        )
        return sol

    def optimise_lineup(self) -> SquadSolution:
        """
        Best XI, captain and bench order over a squad that is already owned.

        Which fifteen to own is a horizon decision - you keep them for weeks -
        but which eleven to start is a decision about one gameweek, and the
        joint solve answered both with the same column. That is how a defender
        worth 3.95 points this Saturday ended up on the bench behind a
        midfielder worth 3.46, because the midfielder was worth more across the
        next five. Roughly half a point a gameweek, every gameweek.

        So the squad keeps its horizon objective and the lineup is re-solved
        here on the coming gameweek alone. Construct the optimiser with
        `value_col` set to the next-gameweek column to use it.
        """
        if self.n != SQUAD_SIZE:
            raise ValueError(
                f"optimise_lineup needs exactly the owned {SQUAD_SIZE}, got {self.n}"
            )
        # Every player is already owned, so the budget cannot bind. It is still
        # passed because the squad constraints are shared with build_squad.
        budget = float(self.cost.sum())
        objective, cons = self._base_program(budget)
        res = self._solve(objective, cons)
        if not res.success:
            raise RuntimeError(f"lineup optimisation failed: {res.message}")
        return self._build_solution(res, budget)

    def optimise_transfers(
        self,
        current_squad_ids: list[int],
        bank: float,
        free_transfers: int = 1,
        selling_prices: dict[int, float] | None = None,
        max_hits: int = 2,
        free_transfer_value: float = FREE_TRANSFER_VALUE,
        horizon_weight: float = 1.0,
    ) -> SquadSolution:
        """
        Best squad reachable from the current one, with hits priced honestly.

        The -4 enters the objective as an actual -4, so "is this hit worth it"
        becomes a real comparison against expected points rather than a
        threshold on an arbitrary scale.
        """
        n = self.n
        owned = np.isin(self.ids, list(current_squad_ids))
        if owned.sum() == 0:
            logger.warning("none of the current squad is in the player pool; building from scratch")
            return self.build_squad(budget=bank)

        # Selling price can differ from current price: FPL takes half of any
        # rise, so the budget must use what the player actually releases.
        sell = self.cost.copy()
        if selling_prices:
            for i, pid in enumerate(self.ids):
                if pid in selling_prices:
                    sell[i] = selling_prices[pid]

        # Money available if the entire current squad were liquidated. Kept
        # players are charged their selling price on both sides of the
        # inequality below, so they net to zero as they should.
        budget = float(bank + sell[owned].sum())
        objective, cons = self._base_program(budget, include_budget=False)

        # Charge every transfer the option value of the free transfer it
        # consumes, so a move has to actually gain something to be worth making.
        # Owned players get a negative coefficient (keeping them avoids the
        # charge); the constant offset this introduces does not affect argmax.
        #
        # Callers set this to zero when transfers are genuinely free - before
        # the first deadline of the season, or under a wildcard - where no free
        # transfer is being consumed and there is no option value to protect.
        # Left at the default it would charge a fifteen-player rebuild 4.5 xP
        # it does not owe, and bias the solver toward standing pat.
        objective[:n] += np.where(owned, -free_transfer_value, 0.0)

        # One extra integer column: hits taken beyond the free allowance.
        #
        # TRANSFER_HIT is 4 real points, but the value column it is weighed
        # against is `xp_horizon` - a decay-weighted sum over the horizon whose
        # weights total ~3.64 at the default settings. Comparing a scalar
        # against that sum priced hits about 3.6x too cheap: a move gaining
        # 1.1 points per gameweek cleared a bar meant to require 4.
        #
        # Scaling by the same weight restores the units. It is also the right
        # decision rule: a transfer that could instead be made next week with
        # the free allowance only buys you one gameweek of its own gain, so the
        # hit has to beat four points in that single week, not across five.
        objective = np.concatenate([objective, [TRANSFER_HIT * horizon_weight]])
        cons = [
            LinearConstraint(
                np.hstack([np.atleast_2d(c.A), np.zeros((np.atleast_2d(c.A).shape[0], 1))]),
                c.lb, c.ub,
            )
            for c in cons
        ]

        # Budget: an owned player enters at his selling price, an unowned one at
        # his purchase price. Charging owned players full current price would
        # penalise holding anyone whose value has risen, since FPL only returns
        # half of a rise on sale.
        effective_cost = np.where(owned, sell, self.cost)
        budget_row = np.zeros(3 * n + 1)
        budget_row[:n] = effective_cost
        cons.append(LinearConstraint(budget_row, 0, budget))

        # transfers_in = number of newly owned players = 15 - (owned kept).
        # hits >= transfers_in - free_transfers, hits >= 0.
        keep_row = np.zeros(3 * n + 1)
        keep_row[:n] = owned.astype(float)
        keep_row[-1] = 1.0
        # sum(owned kept) + hits >= 15 - free_transfers
        cons.append(LinearConstraint(keep_row, SQUAD_SIZE - free_transfers, np.inf))

        # Cap total transfers so the solver cannot propose a wildcard-sized
        # rebuild while paying for it in hits.
        max_transfers = free_transfers + max_hits
        cap_row = np.zeros(3 * n + 1)
        cap_row[:n] = owned.astype(float)
        cons.append(LinearConstraint(cap_row, SQUAD_SIZE - max_transfers, np.inf))

        res = self._solve(objective, cons, extra_cols=1)
        if not res.success:
            raise RuntimeError(f"transfer optimisation failed: {res.message}")

        sol = self._build_solution(res, budget)
        new_ids = set(sol.squad["id"].astype(int))
        old_ids = set(int(i) for i in current_squad_ids)
        sol.transfers_in = sorted(new_ids - old_ids)
        sol.transfers_out = sorted(old_ids - new_ids)
        sol.hits = max(0, len(sol.transfers_in) - free_transfers)

        logger.info(
            "transfer plan: %d in, %d out, %d hit(s) = -%d pts",
            len(sol.transfers_in), len(sol.transfers_out), sol.hits, sol.hits * int(TRANSFER_HIT),
        )
        return sol


def format_squad(sol: SquadSolution, name_col: str = "web_name") -> str:
    """Human-readable squad for logs and Telegram."""
    pos_names = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    lines = [f"Squad £{sol.total_cost:.1f}m | XI xP {sol.xi_xp:.2f}"]
    lines.append("\nStarting XI:")
    for _, p in sol.xi.iterrows():
        tag = ""
        if int(p["id"]) == sol.captain:
            tag = " (C)"
        elif int(p["id"]) == sol.vice_captain:
            tag = " (V)"
        lines.append(
            f"  {pos_names[int(p['position'])]:3s} {str(p.get(name_col, p['id'])):16s}"
            f" £{p['cost']:.1f}m  {p.get('xp_next', float('nan')):.2f}{tag}"
        )
    lines.append("Bench:")
    for i, (_, p) in enumerate(sol.bench.iterrows(), 1):
        lines.append(
            f"  {i}. {pos_names[int(p['position'])]:3s} {str(p.get(name_col, p['id'])):16s}"
            f" £{p['cost']:.1f}m  {p.get('xp_next', float('nan')):.2f}"
        )
    return "\n".join(lines)
