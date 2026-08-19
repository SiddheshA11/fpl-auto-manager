"""
Multi-gameweek transfer sequencing.

The transfer optimiser is one-step greedy: it picks this week's move against a
five-gameweek horizon, but it never plans a *sequence*. It cannot decide to
bank a transfer now so that two can be made next week, and it cannot decide
that a hit is worth taking this week rather than next because the fixture
swing arrives on Saturday. Chips already work as a joint assignment over a
rolling horizon (`chips.solve_assignment`); transfers did not.

Two solvers live here, because they trade correctness against risk differently
and the season simulator was built to decide between them rather than argue:

  `plan_by_enumeration`  enumerates how many transfers to make in each
        gameweek of the horizon and evaluates each plan by chaining the
        existing, tested single-gameweek optimiser. Captures banking and
        hit-timing. Cannot capture "buy X now because he enables Y later",
        since each step is still solved greedily given its budget.

  `plan_jointly`  one MILP over the whole horizon, with ownership binaries per
        player per gameweek and the free-transfer arithmetic carried in
        integer state variables. This is the formulation that can reason about
        a player bought now for a move made later.

Both return only the *first* step. A plan made five gameweeks out is a way of
valuing today's decision, not a commitment - next week's run re-plans from
whatever actually happened.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimizer import (DEFAULT_BENCH_WEIGHT, SQUAD_SIZE, TRANSFER_HIT,
                       SquadOptimizer, SquadSolution)

logger = logging.getLogger("fpl_auto.sequence")

MAX_BANKED_TRANSFERS = 5

# Pure tie-break, not a tuned cost: see the objective in plan_jointly.
BUY_TIEBREAK = 1e-4


@dataclass
class SequencedPlan:
    """The first step of a horizon plan, plus the plan it came from."""

    solution: SquadSolution           # the squad to own THIS gameweek
    schedule: list[int]               # transfers intended in each horizon gameweek
    horizon_xp: float                 # decayed xP of the whole path
    banked: bool                      # true when this week's move is a deliberate roll

    def __str__(self) -> str:
        return (f"plan {self.schedule} -> {self.horizon_xp:.2f} xP over "
                f"{len(self.schedule)} GW" + (" (banking)" if self.banked else ""))


def _next_ft(free_transfers: int, used: int) -> int:
    """FPL's banking rule: spend, then accrue one, capped at five."""
    return min(MAX_BANKED_TRANSFERS, max(0, free_transfers - used) + 1)


def _solve_capped(opt: SquadOptimizer, squad: list[int], bank: float, cap: int,
                  selling: dict[int, float], horizon_weight: float) -> SquadSolution | None:
    """
    Best squad reachable with **at most** `cap` transfers, hits priced at zero.

    The cap is imposed through `free_transfers`, with `max_hits=0` so the
    optimiser's own transfer ceiling becomes exactly `cap`. Hits are charged by
    the caller instead, because in a sequence the cost of a transfer depends on
    which gameweek it lands in - which is the entire question being asked, and
    is not visible from inside a single-gameweek solve.
    """
    if cap <= 0:
        return None
    try:
        return opt.optimise_transfers(
            squad, bank=bank, free_transfers=cap, selling_prices=selling,
            max_hits=0, free_transfer_value=0.0, horizon_weight=horizon_weight,
        )
    except (RuntimeError, ValueError) as e:
        logger.debug("capped solve at %d transfers failed: %s", cap, e)
        return None


def restrict_pool(pool: pd.DataFrame, squad: list[int], keep: int,
                  value_col: str = "xp_horizon") -> pd.DataFrame:
    """
    The `keep` best candidates plus everyone already owned.

    A sequencer solves the same program dozens of times, so pool size is the
    only lever that matters for its running time. Measured at GW15 of 2025-26,
    trimming 759 players to 150 cut a solve from 0.09s to 0.02s and returned an
    identical squad - the tail of the pool is players no optimiser would buy at
    any point in a five-gameweek horizon.

    Owned players are never trimmed. Dropping one makes him unkeepable and
    forces a transfer the plan never asked for; the same mistake aborted the
    first full season simulation.
    """
    if keep <= 0 or len(pool) <= keep:
        return pool
    col = value_col if value_col in pool.columns else "xp_next"
    return pd.concat([
        pool.nlargest(keep, col),
        pool[pool["id"].isin(squad)],
    ]).drop_duplicates(subset="id").reset_index(drop=True)


def plan_by_enumeration(
    pool: pd.DataFrame,
    squad: list[int],
    bank: float,
    free_transfers: int,
    selling: dict[int, float],
    events: list[int],
    *,
    decay: float = 0.84,
    max_hits: int = 2,
    ownership_weight: float = 0.0,
    beam_width: int = 8,
    pool_size: int = 150,
) -> SequencedPlan:
    """
    Choose this gameweek's transfers by searching over transfer *schedules*.

    A schedule says how many transfers to make in each gameweek of the horizon.
    Each is walked forward with the existing single-gameweek optimiser, scored
    on that gameweek's own xP column, charged for any move beyond the free
    allowance, and carried into the next gameweek with the squad and
    free-transfer count it produced. The best decayed total wins.

    Searched with a beam rather than exhaustively. Five gameweeks at up to
    three moves each is 4^5 = 1024 schedules, and walking them all took over
    ten minutes per gameweek - unusable inside a season simulation, let alone a
    deadline. A beam keeps the search proportional to the horizon instead of
    exponential in it, and the schedules it discards are ones already behind on
    accumulated points with the same number of gameweeks left to spend.

    Only the first step is executed. The rest exists to price it: next week's
    run re-plans against whatever actually happened.
    """
    horizon = len(events)
    if horizon == 0:
        raise ValueError("sequencing needs at least one gameweek")

    pool = restrict_pool(pool, squad, pool_size)
    cost = pool.set_index("id")["cost"].to_dict()
    max_per_gw = min(free_transfers + max_hits, MAX_BANKED_TRANSFERS + max_hits)

    optimisers: dict[int, SquadOptimizer] = {}
    solve_cache: dict[tuple[int, tuple[int, ...], int], SquadSolution | None] = {}
    hold_cache: dict[tuple[int, tuple[int, ...]], tuple[SquadSolution | None, float]] = {}

    def column_for(idx: int) -> str:
        col = f"xp_gw{events[idx]}"
        return col if col in pool.columns else "xp_next"

    def optimiser_for(idx: int) -> SquadOptimizer:
        if idx not in optimisers:
            col = column_for(idx)
            optimisers[idx] = SquadOptimizer(pool, value_col=col, captain_col=col,
                                             ownership_weight=ownership_weight)
        return optimisers[idx]

    def hold(idx: int, current: list[int]) -> tuple[SquadSolution | None, float]:
        """Score a squad kept unchanged, on this gameweek's own column."""
        key = (idx, tuple(sorted(current)))
        if key in hold_cache:
            return hold_cache[key]
        col = column_for(idx)
        held = pool[pool["id"].isin(current)]
        result: tuple[SquadSolution | None, float]
        if len(held) != SQUAD_SIZE:
            logger.warning("holding a squad of %d, not %d", len(held), SQUAD_SIZE)
            result = (None, float(held.nlargest(11, col)[col].sum()))
        else:
            try:
                sol = SquadOptimizer(held, value_col=col, captain_col=col,
                                     ownership_weight=ownership_weight).optimise_lineup()
                sol.transfers_in, sol.transfers_out, sol.hits = [], [], 0
                result = (sol, float(sol.xi_xp))
            except (RuntimeError, ValueError) as e:
                logger.debug("lineup solve while holding failed: %s", e)
                result = (None, float(held.nlargest(11, col)[col].sum()))
        hold_cache[key] = result
        return result

    def step(idx: int, current: list[int], cap: int, bank_now: float
             ) -> tuple[SquadSolution | None, float, list[int], float]:
        """Apply at most `cap` transfers at horizon index `idx`."""
        # Bank belongs in the key. Two paths can arrive at the same squad
        # holding different money - one bought and re-sold a player the other
        # never touched - and reusing a solve across them would let the second
        # spend money it does not have.
        key = (idx, tuple(sorted(current)), cap, int(round(bank_now * 10)))
        if key in solve_cache:
            sol = solve_cache[key]
        else:
            sol = (_solve_capped(optimiser_for(idx), current, bank_now, cap, selling, 1.0)
                   if cap > 0 else None)
            solve_cache[key] = sol
        if sol is None:
            _, xp = hold(idx, current)
            return None, xp, current, bank_now
        new = [int(i) for i in sol.squad["id"]]
        spent = sum(cost.get(p, 0.0) for p in new if p not in current)
        # A player bought earlier in the plan is not in `selling`, which only
        # covers the fifteen actually owned today. Defaulting him to 0.0 valued
        # him at nothing on the way out, so any path that bought and later sold
        # anyone looked ruinous and the beam discarded it - silently disabling
        # exactly the multi-step reshuffles this exists to find. He resells at
        # cost, which is what the optimiser itself assumes for an unlisted
        # player and is right while prices are held flat across the horizon.
        freed = sum(selling.get(p, cost.get(p, 0.0)) for p in current if p not in new)
        return sol, float(sol.xi_xp), new, bank_now + freed - spent

    @dataclass
    class _Path:
        squad: list[int]
        bank: float
        ft: int
        total: float
        schedule: list[int]
        first: SquadSolution | None

    beam = [_Path(list(squad), bank, free_transfers, 0.0, [], None)]
    for idx in range(horizon):
        nxt: list[_Path] = []
        for path in beam:
            for want in range(0, max_per_gw + 1):
                sol, xp, new_squad, new_bank = step(idx, path.squad, want, path.bank)
                made = 0 if sol is None else len(sol.transfers_in)
                hits = max(0, made - path.ft)
                nxt.append(_Path(
                    squad=new_squad, bank=new_bank, ft=_next_ft(path.ft, made),
                    total=path.total + (decay ** idx) * (xp - hits * TRANSFER_HIT),
                    schedule=path.schedule + [made],
                    first=path.first if idx else sol,
                ))
        # Distinct schedules can converge on the same squad; keeping both wastes
        # beam slots on paths that are identical from here on.
        seen: set[tuple[int, ...]] = set()
        beam = []
        for p in sorted(nxt, key=lambda p: -p.total):
            sig = tuple(sorted(p.squad))
            if sig in seen:
                continue
            seen.add(sig)
            beam.append(p)
            if len(beam) >= beam_width:
                break

    winner = beam[0]
    sol0 = winner.first
    if sol0 is None:
        sol0, _ = hold(0, squad)
        if sol0 is None:
            raise RuntimeError("no feasible transfer schedule")
    else:
        sol0.hits = max(0, len(sol0.transfers_in) - free_transfers)

    plan = SequencedPlan(
        solution=sol0, schedule=winner.schedule, horizon_xp=winner.total,
        banked=(winner.schedule[0] == 0 and any(t > 0 for t in winner.schedule[1:])),
    )
    logger.info("sequenced %s", plan)
    return plan


# ──────────────────────── the joint formulation ────────────────────────


def plan_jointly(
    pool: pd.DataFrame,
    squad: list[int],
    bank: float,
    free_transfers: int,
    selling: dict[int, float],
    events: list[int],
    *,
    decay: float = 0.84,
    max_hits: int = 2,
    ownership_weight: float = 0.0,
    pool_size: int = 150,
    time_limit: float = 60.0,
) -> SequencedPlan:
    """
    One MILP over the whole horizon.

    The enumeration above still solves each gameweek greedily given its budget,
    so it cannot buy a player this week *because* of a move it enables next
    week. This can: ownership is a decision variable in every gameweek at once,
    and the free-transfer arithmetic is carried in integer state variables
    rather than replayed outside the solver.

    Layout, per horizon index g over n candidates:

        own[g]    binary    in the fifteen
        start[g]  binary    in the eleven
        cap[g]    [0,1]     the armband
        buy[g]    [0,1]     newly owned this gameweek
        hits[g]   integer   moves beyond the free allowance
        ft[g]     integer   free transfers held going into g

    `buy` and `cap` are left continuous deliberately. Both are driven to an
    integral value by the objective - `buy` is pushed down to its lower bound
    max(0, own[g] - own[g-1]) because hits are penalised, and `cap` concentrates
    on the highest-scoring starter because the objective is linear in it - so
    declaring them binary would double the integer count for nothing.

    Two approximations, both stated rather than hidden:

      - prices are held at today's. Modelling a rise would require knowing when
        a player was bought, which is a decision the program is making. The
        greedy optimiser makes the same approximation past gameweek one.
      - the budget binds on current prices in every gameweek, so the plan
        cannot fund a later move out of a later price rise. This is the
        conservative direction.
    """
    horizon = len(events)
    if horizon == 0:
        raise ValueError("sequencing needs at least one gameweek")

    from scipy.optimize import Bounds, LinearConstraint, milp
    from optimizer import MAX_PER_CLUB, SQUAD_BY_POSITION, XI_BOUNDS, XI_SIZE

    pool = restrict_pool(pool, squad, pool_size).reset_index(drop=True)
    pool = pool[pool["cost"] > 0].reset_index(drop=True)
    n = len(pool)
    ids = pool["id"].to_numpy(dtype=int)
    position = pool["position"].to_numpy(dtype=int)
    team = pool["team"].to_numpy(dtype=int)
    cost = pool["cost"].to_numpy(dtype=float)
    owned_now = np.isin(ids, list(squad)).astype(float)

    sell = cost.copy()
    for i, pid in enumerate(ids):
        if pid in selling:
            sell[i] = selling[pid]

    # An owned player is already paid for, so the budget the squad must fit
    # inside is the money in the bank plus what the current fifteen would raise.
    budget = bank + float((owned_now * sell).sum())
    effective_cost = np.where(owned_now.astype(bool), sell, cost)

    xp = np.zeros((horizon, n), dtype=float)
    for g, ev in enumerate(events):
        col = f"xp_gw{ev}"
        xp[g] = pool[col].to_numpy(dtype=float) if col in pool.columns else pool["xp_next"].to_numpy(dtype=float)
    xp = np.nan_to_num(xp)

    if ownership_weight and "selected_by_percent" in pool.columns:
        eo = pd.to_numeric(pool["selected_by_percent"], errors="coerce").fillna(0.0)
        eo = (eo / 100.0).clip(0.0, 1.0).to_numpy(dtype=float)
        tilt = np.clip(1.0 - ownership_weight * (1.0 - 2.0 * eo), 0.0, None)
    else:
        tilt = np.ones(n)

    # Column offsets.
    OWN, START, CAP, BUY = 0, 1, 2, 3
    per_gw = 4 * n
    size = horizon * per_gw + 2 * horizon          # + hits[g], ft[g]

    def col(block: int, g: int) -> slice:
        base = g * per_gw + block * n
        return slice(base, base + n)

    def hits_col(g: int) -> int:
        return horizon * per_gw + g

    def ft_col(g: int) -> int:
        return horizon * per_gw + horizon + g

    cons: list[LinearConstraint] = []

    def row() -> np.ndarray:
        return np.zeros(size)

    for g in range(horizon):
        r = row(); r[col(OWN, g)] = 1.0
        cons.append(LinearConstraint(r, SQUAD_SIZE, SQUAD_SIZE))
        r = row(); r[col(START, g)] = 1.0
        cons.append(LinearConstraint(r, XI_SIZE, XI_SIZE))
        r = row(); r[col(CAP, g)] = 1.0
        cons.append(LinearConstraint(r, 1, 1))

        for pos, count in SQUAD_BY_POSITION.items():
            r = row(); r[col(OWN, g)] = (position == pos).astype(float)
            cons.append(LinearConstraint(r, count, count))
        for pos, (lo, hi) in XI_BOUNDS.items():
            r = row(); r[col(START, g)] = (position == pos).astype(float)
            cons.append(LinearConstraint(r, lo, hi))
        for club in np.unique(team):
            r = row(); r[col(OWN, g)] = (team == club).astype(float)
            cons.append(LinearConstraint(r, 0, MAX_PER_CLUB))

        # start <= own, cap <= start.
        A = np.zeros((n, size))
        A[:, col(OWN, g)] = -np.eye(n)
        A[:, col(START, g)] = np.eye(n)
        cons.append(LinearConstraint(A, -np.inf, np.zeros(n)))
        A = np.zeros((n, size))
        A[:, col(START, g)] = -np.eye(n)
        A[:, col(CAP, g)] = np.eye(n)
        cons.append(LinearConstraint(A, -np.inf, np.zeros(n)))

        # Budget. An owned player is charged his selling price, not his
        # current price - the same convention optimise_transfers uses, and for
        # the same reason: FPL returns only half of a rise, so charging full
        # price for a player you already hold makes keeping a riser look like
        # it costs money and quietly biases the plan toward selling him.
        r = row(); r[col(OWN, g)] = effective_cost
        cons.append(LinearConstraint(r, 0, budget))

        # buy[g] >= own[g] - own[g-1]. At g == 0 the previous squad is the one
        # actually held, which enters as a constant on the right-hand side.
        A = np.zeros((n, size))
        A[:, col(BUY, g)] = np.eye(n)
        A[:, col(OWN, g)] = -np.eye(n)
        if g == 0:
            cons.append(LinearConstraint(A, -owned_now, np.inf))
        else:
            A[:, col(OWN, g - 1)] = np.eye(n)
            cons.append(LinearConstraint(A, np.zeros(n), np.inf))

        # hits[g] >= transfers[g] - ft[g].
        r = row(); r[col(BUY, g)] = -1.0; r[hits_col(g)] = 1.0; r[ft_col(g)] = 1.0
        cons.append(LinearConstraint(r, 0, np.inf))

        # Cap moves per gameweek, mirroring the greedy optimiser's ceiling.
        r = row(); r[col(BUY, g)] = 1.0
        cons.append(LinearConstraint(r, 0, MAX_BANKED_TRANSFERS + max_hits))

        if g == 0:
            r = row(); r[ft_col(0)] = 1.0
            cons.append(LinearConstraint(r, free_transfers, free_transfers))
        else:
            # ft[g] <= ft[g-1] - (transfers[g-1] - hits[g-1]) + 1, and <= 5.
            # Stated as an upper bound only: the objective wants free transfers,
            # so the solver drives it to the bound, and an equality would need a
            # second binary to linearise the min().
            r = row()
            r[ft_col(g)] = 1.0
            r[ft_col(g - 1)] = -1.0
            r[col(BUY, g - 1)] = 1.0
            r[hits_col(g - 1)] = -1.0
            cons.append(LinearConstraint(r, -np.inf, 1.0))

    # Maximise decayed points, less hits. scipy.milp minimises.
    objective = np.zeros(size)
    for g in range(horizon):
        w = decay ** g
        objective[col(OWN, g)] = -w * xp[g] * tilt * DEFAULT_BENCH_WEIGHT
        objective[col(START, g)] = -w * xp[g] * tilt * (1.0 - DEFAULT_BENCH_WEIGHT)
        objective[col(CAP, g)] = -w * xp[g]
        objective[hits_col(g)] = w * TRANSFER_HIT
        # `buy` is otherwise unconstrained from above whenever the free
        # allowance covers it, so the solver is indifferent to reporting more
        # transfers than it makes and the schedule it hands back is not
        # trustworthy. A tie-break epsilon - far too small to outrank any real
        # points difference - pins it to its lower bound, max(0, own[g] -
        # own[g-1]), which is the number of transfers actually made.
        objective[col(BUY, g)] = BUY_TIEBREAK

    integrality = np.zeros(size)
    lower = np.zeros(size)
    upper = np.ones(size)
    for g in range(horizon):
        integrality[col(OWN, g)] = 1
        integrality[col(START, g)] = 1
        integrality[hits_col(g)] = 1
        integrality[ft_col(g)] = 1
        upper[hits_col(g)] = max_hits + MAX_BANKED_TRANSFERS
        upper[ft_col(g)] = MAX_BANKED_TRANSFERS

    res = milp(c=objective, constraints=cons, integrality=integrality,
               bounds=Bounds(lower, upper),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4})
    if not res.success or res.x is None:
        raise RuntimeError(f"joint transfer optimisation failed: {res.message}")

    own0 = np.round(res.x[col(OWN, 0)]).astype(bool)
    schedule = [int(round(float(res.x[col(BUY, g)].sum()))) for g in range(horizon)]

    # Hand the first gameweek back through the ordinary single-gameweek
    # machinery, so the caller receives a SquadSolution built exactly like
    # every other one - same lineup rules, same bench order, same vice pick.
    first_ids = [int(i) for i in ids[own0]]
    col0 = f"xp_gw{events[0]}" if f"xp_gw{events[0]}" in pool.columns else "xp_next"
    held = pool[pool["id"].isin(first_ids)]
    sol = SquadOptimizer(held, value_col=col0, captain_col=col0,
                         ownership_weight=ownership_weight).optimise_lineup()
    sol.transfers_in = sorted(set(first_ids) - set(squad))
    sol.transfers_out = sorted(set(squad) - set(first_ids))
    sol.hits = max(0, len(sol.transfers_in) - free_transfers)

    plan = SequencedPlan(
        solution=sol, schedule=schedule, horizon_xp=float(-res.fun),
        banked=(schedule[0] == 0 and any(t > 0 for t in schedule[1:])),
    )
    logger.info("jointly %s", plan)
    return plan
