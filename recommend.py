"""
Produce a squad recommendation from live FPL data.

Two modes:

  --build    the best legal 15 from scratch, for entering a new season or
             planning a wildcard. FPL has no API for creating an initial
             squad, so this prints a list to enter on the site.

  --transfer the best moves from the squad you already own, with hits priced
             against expected points.

Run with --offline to work from the most recent snapshot instead of the API.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import fpl_client
import priors
import rivals
import xp_model as X
from fetch_data import LIVE_ENDPOINTS, _get
from config import ENDPOINTS, FPL_TEAM_ID
from optimizer import SquadOptimizer, format_squad

logger = logging.getLogger("fpl_auto.recommend")

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_game_state(offline: bool) -> tuple[dict, list[dict]]:
    """Live API by default; most recent snapshot when offline or on failure."""
    bootstrap, fixtures, _ = _load_game_state(offline)
    return bootstrap, fixtures


def _load_game_state(offline: bool) -> tuple[dict, list[dict], bool]:
    """As load_game_state, plus whether the data actually came from the live API."""
    if not offline:
        try:
            import json
            bootstrap = json.loads(_get(LIVE_ENDPOINTS["bootstrap-static"]))
            fixtures = json.loads(_get(LIVE_ENDPOINTS["fixtures"]))
            logger.info("loaded live FPL data")
            return bootstrap, fixtures, True
        except (RuntimeError, ValueError) as e:
            logger.warning("live fetch failed (%s); falling back to snapshot", e)

    bs_path = X.latest_snapshot(SNAPSHOT_DIR, "bootstrap-static")
    fx_path = X.latest_snapshot(SNAPSHOT_DIR, "fixtures")
    if not bs_path or not fx_path:
        raise SystemExit("no snapshot available and live fetch failed; run fetch_data.py")
    logger.info("using snapshot %s", bs_path.name)
    return X.load_snapshot(bs_path), X.load_snapshot(fx_path), False


def load_recent_minutes(bootstrap: dict, live: bool) -> dict[int, list[float]]:
    """
    The same recent-minutes lags manager.py hands the model, from the public
    `event/{gw}/live/` endpoint and through the same parser.

    Never an authenticated FPLClient: the refresh token is single-use and
    rotates, so a preview that logged in would spend the weekly run's secret.
    On a snapshot - --offline, or the live fetch already failed - there is
    nothing to read them from; that degrades to season rates, as manager.py
    does when the fetch fails, and says so.
    """
    if not live:
        logger.warning("snapshot data: no recent minutes; using season rates alone, "
                       "so this preview scores on a weaker model than the weekly run")
        return {}

    def get_live(gw: int) -> dict | None:
        try:
            return json.loads(_get(ENDPOINTS["event_live"].format(event_id=gw)))
        except (RuntimeError, ValueError) as e:
            logger.warning("GW%d live fetch failed (%s)", gw, e)
            return None

    events_done = sum(1 for e in bootstrap.get("events", []) if e.get("finished"))
    try:
        return fpl_client.recent_minutes_from(get_live, events_done)
    except Exception as e:                      # noqa: BLE001 - never fatal, as in manager.py
        logger.warning("could not fetch recent minutes (%s); using season rates alone", e)
        return {}


def build_model(bootstrap: dict, fixtures: list[dict], horizon: int,
                recent_minutes: dict[int, list[float]] | None = None) -> pd.DataFrame:
    team_codes = {t["code"]: t["name"] for t in bootstrap["teams"]}
    prior_set = priors.build_priors(current_team_codes=team_codes)
    model = X.XPModel(bootstrap, fixtures, prior_set, X.ModelConfig(horizon=horizon),
                      recent_minutes=recent_minutes)
    events = X.next_events(bootstrap, horizon)
    if not events:
        raise SystemExit("no upcoming gameweeks found")
    logger.info("modelling gameweeks %s", events)
    return model.expected_points(events), events


def _exclude_unavailable(scored: pd.DataFrame, keep_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Drop players who cannot play at all, except any already owned.

    The model already prices doubt through availability, but a suspended or
    long-term-injured player with a strong prior can still surface with a
    non-trivial horizon score, and there is no reason to buy him.

    Owned players must survive the filter regardless. The optimiser funds
    transfers from the selling prices of the squad it can see, so dropping an
    injured player you own deletes his sale money from the budget along with
    him - and quietly returns a worse plan than the one actually available.
    """
    before = len(scored)
    keep = scored["status"].isin(["a", "d"])
    if keep_ids:
        keep = keep | scored["id"].isin(keep_ids)
    out = scored[keep].copy()
    dropped = before - len(out)
    if dropped:
        logger.info("excluded %d unavailable players (injured/suspended/left)", dropped)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="best 15 from scratch")
    ap.add_argument("--transfer", action="store_true", help="best moves from an owned squad")
    ap.add_argument("--squad", help="comma-separated element ids you currently own")
    ap.add_argument("--bank", type=float, default=0.0, help="money in the bank, £m")
    ap.add_argument("--free-transfers", type=int, default=1)
    ap.add_argument("--max-hits", type=int, default=2)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--horizon", type=int, default=5, help="gameweeks to optimise over")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if not args.build and not args.transfer:
        args.build = True

    bootstrap, fixtures, live = _load_game_state(args.offline)
    # manager.py folds these into the start rate, the largest single accuracy
    # gain the model has (R2 0.269 -> 0.319). Omitted, this tool previewed a
    # plan scored on a different model from the one the weekly run submits.
    recent_minutes = load_recent_minutes(bootstrap, live)
    scored, events = build_model(bootstrap, fixtures, args.horizon, recent_minutes)

    owned = [int(s) for s in args.squad.split(",") if s.strip()] if args.squad else []
    pool = _exclude_unavailable(scored, keep_ids=owned)

    # The tilt has to match the weekly run. This tool exists to preview what
    # the bot will do, and it defaulted to optimizer.OWNERSHIP_WEIGHT (0.0)
    # while manager.py used config.OWNERSHIP_WEIGHT - so at the current +0.20
    # it previewed a squad without Haaland and 0.56 xP short of the one that
    # would actually be submitted. deadline_check.py had the identical bug and
    # carries a note about it; this caller was missed.
    # Same helper the weekly run uses, so the preview cannot aim at a
    # different field from the submission. Two separate bugs in this file came
    # from passing a tilt argument the weekly run did not.
    tilt = rivals.tilt_inputs(pool, FPL_TEAM_ID, events[0] - 1)
    pool = tilt.scored
    logger.info("ownership tilt: %+.3f on the %s", tilt.ownership_weight, tilt.source)
    opt = SquadOptimizer(pool, value_col="xp_horizon", captain_col="xp_next",
                         ownership_weight=tilt.ownership_weight,
                         ownership_col=tilt.ownership_col,
                         captain_ownership_weight=tilt.captain_weight)

    if args.transfer:
        if not owned:
            raise SystemExit("--transfer needs --squad with your current element ids")
        # The -4 for a hit is compared against `xp_horizon`, a decay-weighted
        # sum over args.horizon gameweeks, so it has to be expressed on the
        # same scale. Omitted, it fell back to optimizer.py's default of 1.0
        # and priced hits ~3.6x too cheap: measured over six suboptimal squads
        # x {0, 1} free transfers, all twelve diverged, the preview taking the
        # maximum two hits in every case where the weekly run takes none.
        #
        # This is the second argument in this call to have gone missing the
        # same way; see the note on ownership_weight above. Computed from
        # args.horizon rather than hardcoded, because unlike manager.py's
        # constant this tool takes the horizon as an argument.
        horizon_weight = sum(
            X.ModelConfig(horizon=args.horizon).horizon_decay ** i
            for i in range(args.horizon)
        )
        sol = opt.optimise_transfers(
            owned, bank=args.bank, free_transfers=args.free_transfers,
            max_hits=args.max_hits, horizon_weight=horizon_weight,
        )
        names = pool.set_index("id")["web_name"].to_dict()
        print("\n=== TRANSFERS ===")
        if not sol.transfers_in:
            print("  No transfer improves the squad. Roll the free transfer.")
        for tin, tout in zip(sol.transfers_in, sol.transfers_out):
            print(f"  OUT {names.get(tout, tout):18s}  ->  IN {names.get(tin, tin)}")
        if sol.hits:
            print(f"  Hits: {sol.hits} (-{sol.hits * 4} pts), taken because the gain exceeds the cost")
    else:
        sol = opt.build_squad(args.budget)

    print()
    print(format_squad(sol))

    gw = events[0]
    print(f"\nGameweek {gw} | squad £{sol.total_cost:.1f}m | £{args.budget - sol.total_cost:.1f}m remaining")
    cap = pool.loc[pool["id"] == sol.captain]
    vice = pool.loc[pool["id"] == sol.vice_captain]
    if len(cap):
        print(f"Captain: {cap.iloc[0]['web_name']} ({cap.iloc[0]['xp_next']:.2f} xP this GW)")
    if len(vice):
        print(f"Vice:    {vice.iloc[0]['web_name']} ({vice.iloc[0]['xp_next']:.2f} xP this GW)")

    flagged = sol.squad[sol.squad["status"] == "d"]
    if len(flagged):
        print("\nFlagged as doubtful - check the news before the deadline:")
        for _, p in flagged.iterrows():
            print(f"  {p['web_name']} ({p['team_name']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
