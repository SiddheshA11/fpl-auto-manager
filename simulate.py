"""
Replay a season making real decisions, to find out whether one optimiser beats
another.

`backtest.py` scores *predictions* - it asks whether xp_next matches what a
player went on to do. That cannot answer whether a change to the *optimiser*
is worth shipping, because a decision rule is not a forecast. Every optimiser
constant in this repo was therefore set by argument rather than measurement,
FREE_TRANSFER_VALUE included, and that one is a hand-picked stand-in for
exactly the option value multi-gameweek sequencing would compute properly.

So this walks a real season one gameweek at a time. At each deadline it
reconstructs the state as it stood, runs the real SquadOptimizer against it,
applies the transfers, and then scores the resulting eleven against what the
players actually did - with autosubs, captaincy, real per-gameweek prices and
FPL's sell-at-half-the-rise rule. The output is a season points total, which
is the only number a decision rule can honestly be judged on.

What it deliberately does not model, each of which makes the absolute total
conservative rather than flattering:

  - chips. Every strategy plays none, so the comparison is unaffected, but a
    real season scores higher than anything printed here.
  - injury flags. The historical dataset has no `status`, so build_state marks
    everyone fit; the model buys players it would have been warned off.
  - the ownership tilt. build_state reports selected_by_percent as 0.0, so
    every strategy runs at an effective tilt of 0. Comparisons between
    strategies are unaffected; do not read the totals as a rank simulation.

Run: python simulate.py --season 2025-26 --strategy greedy
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import priors
import xp_model as X
from backtest import HISTORY_DIR, _scoring_config, build_state
from optimizer import SQUAD_SIZE, SquadOptimizer

logger = logging.getLogger("fpl_auto.simulate")

# FPL's transfer rules, as at 2025-26.
MAX_BANKED_TRANSFERS = 5
HIT_COST = 4

# Legal XI shapes: at least one keeper, three defenders, one forward.
MIN_XI = {1: 1, 2: 3, 3: 0, 4: 1}
MAX_XI = {1: 1, 2: 5, 3: 5, 4: 3}


@dataclass
class SeasonResult:
    """Everything a strategy comparison needs, per gameweek and in total."""

    rows: list[dict] = field(default_factory=list)

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    @property
    def total(self) -> int:
        return int(self.frame["net"].sum()) if self.rows else 0

    def summary(self, label: str) -> str:
        f = self.frame
        return (
            f"{label:<24} {int(f['net'].sum()):>6} pts  "
            f"({int(f['gross'].sum())} gross - {int(f['hits'].sum()) * HIT_COST} hits)  "
            f"| {int(f['transfers'].sum()):>3} transfers, {int(f['hits'].sum()):>2} hits, "
            f"{f['net'].mean():.1f} pts/gw"
        )


def _actuals(gw_df: pd.DataFrame, gw: int) -> tuple[dict[int, int], dict[int, int]]:
    """Points and minutes actually returned in `gw`, summed across a double."""
    rows = gw_df[gw_df["GW"] == gw]
    if rows.empty:
        return {}, {}
    agg = rows.groupby("element").agg({"total_points": "sum", "minutes": "sum"})
    return (
        {int(k): int(v) for k, v in agg["total_points"].items()},
        {int(k): int(v) for k, v in agg["minutes"].items()},
    )


def _prices(gw_df: pd.DataFrame, gw: int) -> dict[int, float]:
    """Price in £m as at `gw`, deduplicated across a double gameweek."""
    rows = gw_df[gw_df["GW"] == gw].drop_duplicates(subset="element")
    return {int(r["element"]): float(r["value"]) / 10.0 for _, r in rows.iterrows()}


def selling_price(purchase: float, current: float) -> float:
    """
    FPL returns the purchase price plus half of any rise, rounded down to
    £0.1m. A fall is taken in full.

    Getting this wrong in the optimistic direction lets a simulated manager
    fund transfers with money that would not exist, which flatters exactly the
    strategies that churn the squad most - so it is not a detail the comparison
    can be indifferent to.
    """
    if current <= purchase:
        return current
    return purchase + np.floor((current - purchase) * 10.0 / 2.0) / 10.0


def pick_xi(squad: list[int], positions: dict[int, int], points: dict[int, int],
            minutes: dict[int, int], xp: dict[int, float]) -> tuple[list[int], int, int]:
    """
    The eleven that actually scored, after autosubs.

    The manager names an XI before the deadline on expected points; players who
    then fail to appear are replaced from the bench in order, but only where
    the substitution leaves a legal shape. Skipping this would charge every
    strategy for blanks the real game absorbs, and would charge the
    squad-churning strategies most, since they carry the most uncertainty.
    """
    ordered = sorted(squad, key=lambda p: -xp.get(p, 0.0))
    keepers = [p for p in ordered if positions.get(p) == 1]
    outfield = [p for p in ordered if positions.get(p) != 1]

    xi = keepers[:1] + outfield[:10]
    bench = keepers[1:] + outfield[10:]

    def legal(sel: list[int]) -> bool:
        counts = {t: sum(1 for p in sel if positions.get(p) == t) for t in (1, 2, 3, 4)}
        return (
            len(sel) == 11
            and all(counts[t] >= MIN_XI[t] for t in MIN_XI)
            and all(counts[t] <= MAX_XI[t] for t in MAX_XI)
        )

    for slot, player in enumerate(list(xi)):
        if minutes.get(player, 0) > 0:
            continue
        for cand in list(bench):
            if minutes.get(cand, 0) <= 0:
                continue
            trial = list(xi)
            trial[slot] = cand
            if legal(trial):
                xi = trial
                bench.remove(cand)
                break

    # Captain the highest-xP starter; the vice takes the armband if he blanks,
    # which is the rule that makes a captain pick recoverable.
    by_xp = sorted(xi, key=lambda p: -xp.get(p, 0.0))
    captain = by_xp[0]
    vice = by_xp[1] if len(by_xp) > 1 else captain
    armband = captain if minutes.get(captain, 0) > 0 else vice

    gross = sum(points.get(p, 0) for p in xi) + points.get(armband, 0)
    return xi, gross, armband


def simulate_season(
    season: str,
    start_gw: int,
    end_gw: int,
    strategy: str = "greedy",
    horizon: int = 5,
    max_hits: int = 2,
    free_transfer_value: float | None = None,
    budget: float = 100.0,
) -> SeasonResult:
    d = HISTORY_DIR / season
    gw_df = pd.read_csv(d / "merged_gw.csv")
    raw_df = pd.read_csv(d / "players_raw.csv")
    teams_df = pd.read_csv(d / "teams.csv")
    fixtures = pd.read_csv(d / "fixtures.csv").to_dict("records")
    game_config = _scoring_config()

    earlier = [s for s in priors.available_seasons() if s < season]
    if not earlier:
        raise SystemExit(f"no seasons before {season} on disk; cannot build clean priors")
    team_codes = {int(r["code"]): r["name"] for _, r in teams_df.iterrows()}
    prior_set = priors.build_priors(seasons=earlier, current_team_codes=team_codes)
    logger.info("priors from %s (excluding %s)", earlier, season)

    import optimizer as optimizer_mod
    ftv = optimizer_mod.FREE_TRANSFER_VALUE if free_transfer_value is None else free_transfer_value
    decay = X.ModelConfig(horizon=horizon).horizon_decay
    horizon_weight = sum(decay ** i for i in range(horizon))

    squad: list[int] = []
    purchase: dict[int, float] = {}
    last_price: dict[int, float] = {}
    bank = 0.0
    free_transfers = 1
    result = SeasonResult()

    for gw in range(start_gw, end_gw + 1):
        if gw_df[gw_df["GW"] == gw].empty:
            continue
        state = build_state(gw_df, raw_df, teams_df, gw, game_config)
        events = [g for g in range(gw, min(gw + horizon, 39))]
        model = X.XPModel(state, fixtures, prior_set, X.ModelConfig(horizon=horizon))
        scored = model.expected_points(events)

        last_price.update(_prices(gw_df, gw))
        price_now = dict(last_price)
        # An unowned player with no row this gameweek cannot be bought, but an
        # OWNED one must stay in the pool whatever happens. Dropping him makes
        # him unkeepable, so the solver is forced to transfer him out - and
        # when several go at once the free-transfer floor
        # (sum(kept) + hits >= 15 - free_transfers) becomes unsatisfiable and
        # the whole season aborts. That is what killed the first full run, at
        # GW34. manager.py keeps owned players for the same reason, though
        # there the trigger is an injury flag rather than a missing row.
        tradeable = set(_prices(gw_df, gw))
        pool = scored[scored["id"].isin(tradeable) | scored["id"].isin(squad)].copy()
        if pool.empty:
            continue
        # Price the pool at what things actually cost this gameweek. build_state
        # falls back to the season-start price for a player with no row, which
        # would let the optimiser buy at a stale valuation.
        pool["cost"] = pool["id"].map(price_now).fillna(pool["cost"])

        positions = pool.set_index("id")["position"].astype(int).to_dict()
        xp_next = pool.set_index("id")["xp_next"].to_dict()
        opt = SquadOptimizer(pool, value_col="xp_horizon", captain_col="xp_next",
                             ownership_weight=0.0)

        if not squad:
            sol = opt.build_squad(budget=budget)
            squad = [int(i) for i in sol.squad["id"]]
            purchase = {p: price_now.get(p, 0.0) for p in squad}
            bank = budget - sum(purchase.values())
            hits, made = 0, 0
        else:
            selling = {p: selling_price(purchase.get(p, price_now.get(p, 0.0)),
                                        price_now.get(p, purchase.get(p, 0.0)))
                       for p in squad}
            sol = _plan_transfers(
                strategy, opt, pool, squad, bank, free_transfers, selling,
                max_hits, ftv, horizon_weight, events, model,
            )
            new = [int(i) for i in sol.squad["id"]]
            ins = [p for p in new if p not in squad]
            outs = [p for p in squad if p not in new]
            proceeds = sum(selling.get(p, price_now.get(p, 0.0)) for p in outs)
            spend = sum(price_now.get(p, 0.0) for p in ins)
            bank = bank + proceeds - spend
            for p in outs:
                purchase.pop(p, None)
            for p in ins:
                purchase[p] = price_now.get(p, 0.0)
            if len(new) != SQUAD_SIZE:
                raise RuntimeError(f"GW{gw}: solver returned {len(new)} players, not {SQUAD_SIZE}")
            squad = new
            made = len(ins)
            hits = max(0, made - free_transfers)
            free_transfers = min(MAX_BANKED_TRANSFERS, max(1, free_transfers - made + 1))

        points, minutes = _actuals(gw_df, gw)
        xi, gross, armband = pick_xi(squad, positions, points, minutes, xp_next)
        net = gross - hits * HIT_COST
        result.rows.append({
            "gw": gw, "gross": gross, "hits": hits, "net": net,
            "transfers": made, "bank": round(bank, 1), "ft": free_transfers,
            "captain": armband, "squad_value": round(sum(price_now.get(p, 0.0) for p in squad), 1),
        })
        logger.info("GW%-2d %3d pts (gross %3d, %d hit) | %d transfer(s), bank £%.1fm, %d FT",
                    gw, net, gross, hits, made, bank, free_transfers)

    return result


def _plan_transfers(strategy, opt, pool, squad, bank, free_transfers, selling,
                    max_hits, ftv, horizon_weight, events, model):
    """Dispatch to a transfer strategy. Greedy is the incumbent, one-step."""
    if strategy == "greedy":
        return opt.optimise_transfers(
            squad, bank=bank, free_transfers=free_transfers, selling_prices=selling,
            max_hits=max_hits, free_transfer_value=ftv, horizon_weight=horizon_weight,
        )
    if strategy == "enumerate":
        import sequence
        return sequence.plan_by_enumeration(
            pool, squad, bank, free_transfers, selling, events,
            decay=X.ModelConfig().horizon_decay, max_hits=max_hits,
            ownership_weight=0.0,
        ).solution
    if strategy == "joint":
        import sequence
        return sequence.plan_jointly(
            pool, squad, bank, free_transfers, selling, events,
            decay=X.ModelConfig().horizon_decay, max_hits=max_hits,
            ownership_weight=0.0,
        ).solution
    raise SystemExit(f"unknown strategy {strategy!r}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fpl_auto.optimizer").setLevel(logging.WARNING)
    logging.getLogger("fpl_auto.xp_model").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--start-gw", type=int, default=2)
    ap.add_argument("--end-gw", type=int, default=38)
    ap.add_argument("--strategy", default="greedy")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--max-hits", type=int, default=2)
    ap.add_argument("--free-transfer-value", type=float, default=None)
    args = ap.parse_args()

    res = simulate_season(
        args.season, args.start_gw, args.end_gw, strategy=args.strategy,
        horizon=args.horizon, max_hits=args.max_hits,
        free_transfer_value=args.free_transfer_value,
    )
    if not res.rows:
        logger.error("no gameweeks simulated")
        return 1
    print()
    print(res.summary(f"{args.strategy} h={args.horizon}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
