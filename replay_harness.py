"""
The decision-replay harness: does an optimiser change win, measured with power.

simulate.py replays whole seasons, and four of them could not resolve
anything: the sign test floors at p=0.125, the MDE is ~135 points a season,
and the horizon "control" was biased. This keeps simulate.py's rules - it
calls the same play_gameweek - and changes the design:

  FORKED WINDOWS. Each season is played once under the production config,
  recording the state before every deadline. A candidate arm then forks from
  the baseline's exact state at gameweek s and plays L gameweeks. The
  window's outcome is the candidate's points minus the baseline's over the
  same weeks. Every start gameweek of every replayable season is a window:
  ~150 of them, against 4 seasons.

  TERMINAL VALUE. A window can end with the two arms holding different
  squads, and that difference has not been scored yet. Optionally both final
  squads are frozen and scored on what they actually returned over the next
  T gameweeks - no transfers, so no model judges its own squad.

  NULLS. An arm identical to the baseline must reproduce it exactly in every
  window, and so must alpha +/- 1e-6. Both are determinism checks. Neither
  makes a decision flip, so neither says anything about noise; the placebo
  that does is in replay_stats.py.

The baseline is production: alpha, horizon, free-transfer value and bench
weight from the code, the ownership tilt at config.OWNERSHIP_WEIGHT on the
template ownership reconstructed per gameweek (simulate.template_ownership).
Field (mini-league) ownership and the captain tilt cannot be replayed - no
rival data exists for past seasons - and chips and injury flags are not
modelled, as in simulate.py.

Scoring is the expensive step (~1.5s a gameweek against ~0.25s to solve), so
every gameweek is scored once per alpha at SCORE_HORIZON and cached; shorter
horizons are derived from the per-gameweek columns, which a test pins
bit-identical to scoring directly.

    python3 replay_harness.py --arms same,alpha+1e-6,alpha-1e-6
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import logging
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import simulate as S

logger = logging.getLogger("fpl_auto.replay")

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "replay_cache"
OUT_DIR = ROOT / "data" / "replay"

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SCORE_HORIZON = 8
DEFAULT_L = 8
DEFAULT_TERMINAL = 4
LAST_GW = 38


def _production() -> dict:
    import config
    import manager
    import optimizer
    import xp_model as X
    return {"alpha": X.OUTFIELD_COMPETITION_ALPHA, "horizon": manager.HORIZON,
            "free_transfer_value": optimizer.FREE_TRANSFER_VALUE,
            "bench_weight": optimizer.DEFAULT_BENCH_WEIGHT,
            "ownership_weight": config.OWNERSHIP_WEIGHT, "max_hits": 2}


@dataclass(frozen=True)
class Arm:
    """One configuration of the decision rule. Defaults are production."""

    name: str = "baseline"
    alpha: float | None = None
    horizon: int | None = None
    free_transfer_value: float | None = None
    bench_weight: float | None = None
    ownership_weight: float | None = None
    max_hits: int | None = None
    # The placebo: N(0, noise_eps) added to every player's xP, drawn from
    # noise_seed. Two arms differing only in seed are exchangeable, so their
    # expected difference is exactly zero.
    noise_eps: float = 0.0
    noise_seed: int = 0

    def resolved(self) -> "Arm":
        prod = _production()
        return dataclasses.replace(self, **{k: (getattr(self, k) if getattr(self, k) is not None
                                                else v) for k, v in prod.items()})

    def key(self) -> str:
        r = dataclasses.asdict(self.resolved())
        r.pop("name")
        return hashlib.sha1(repr(sorted(r.items())).encode()).hexdigest()[:12]


def parse_arm(spec: str) -> Arm:
    """
    'same', 'alpha+1e-6', 'alpha=1.0', 'horizon=6', 'ftv=0.0', 'bench=0.1',
    'own=0.0', 'noise=0.02@3' (eps 0.02, seed 3). Comma-joined in --arms.
    """
    prod = _production()
    if spec in ("same", "baseline"):
        return Arm(name=spec)
    if spec.startswith("alpha+") or spec.startswith("alpha-"):
        delta = float(spec[5:])
        return Arm(name=spec, alpha=prod["alpha"] + delta)
    if spec.startswith("noise="):
        eps, _, seed = spec[6:].partition("@")
        return Arm(name=spec, noise_eps=float(eps), noise_seed=int(seed or 0))
    k, _, v = spec.partition("=")
    field_for = {"alpha": "alpha", "horizon": "horizon", "ftv": "free_transfer_value",
                 "bench": "bench_weight", "own": "ownership_weight", "hits": "max_hits"}
    if k not in field_for:
        raise SystemExit(f"unknown arm {spec!r}")
    cast = int if k in ("horizon", "hits") else float
    return Arm(name=spec, **{field_for[k]: cast(v)})


# ------------------------------------------------------------------ scoring


def _cache_path(season: str, gw: int, alpha: float) -> Path:
    return CACHE_DIR / f"{season}_gw{gw:02d}_alpha{alpha!r}_h{SCORE_HORIZON}.pkl"


def score_with_order(data: S.SeasonData, gw: int, horizon: int) -> tuple[pd.DataFrame, list[int]]:
    """
    S.score_gameweek, plus the model's pre-sort player order. The model sorts
    its output by xp_horizon with a sort that is not stable, so ties land in
    an order fixed by its internal player frame - and row order reaches the
    solver. Deriving another horizon's frame exactly needs that order back.
    A test pins the frame here to S.score_gameweek's.
    """
    import xp_model as X
    from backtest import build_state, recent_minutes_for

    state = build_state(data.gw_df, data.raw_df, data.teams_df, gw, data.game_config)
    events = [g for g in range(gw, min(gw + horizon, 39))]
    model = X.XPModel(state, data.fixtures, data.prior_set, X.ModelConfig(horizon=horizon),
                      recent_minutes=recent_minutes_for(data.gw_df, gw))
    return model.expected_points(events), [int(i) for i in model.players["id"]]


def _score_task(args) -> str:
    season, gw, alpha = args
    path = _cache_path(season, gw, alpha)
    if path.exists():
        return str(path)
    import xp_model as X
    production = X.OUTFIELD_COMPETITION_ALPHA
    X.OUTFIELD_COMPETITION_ALPHA = alpha
    try:
        scored, order = score_with_order(_season(season, with_priors=True), gw, SCORE_HORIZON)
    finally:
        # A module global: left set, it leaks into whatever this process
        # scores next - including the rest of a test session.
        X.OUTFIELD_COMPETITION_ALPHA = production
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps({"scored": scored, "order": order},
                                 protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(path)
    return str(path)


def load_scored(season: str, gw: int, alpha: float) -> tuple[pd.DataFrame, list[int]]:
    c = pickle.loads(_cache_path(season, gw, alpha).read_bytes())
    return c["scored"], c["order"]


_SEASONS: dict[tuple[str, bool], S.SeasonData] = {}


def _season(season: str, with_priors: bool = False) -> S.SeasonData:
    """Per process. Priors are only needed to score, so arms skip them."""
    key = (season, with_priors)
    if key not in _SEASONS:
        if with_priors:
            _SEASONS[key] = S.load_season(season)
        else:
            d = S.HISTORY_DIR / season
            _SEASONS[key] = S.SeasonData(
                season, S.priors.read_season_csv(d / "merged_gw.csv"),
                S.priors.read_season_csv(d / "players_raw.csv"),
                S.priors.read_season_csv(d / "teams.csv"),
                S.priors.read_season_csv(d / "fixtures.csv").to_dict("records"),
                S._scoring_config(), None)
    return _SEASONS[key]


def gameweeks(season: str) -> list[int]:
    data = _season(season)
    return [gw for gw in range(2, LAST_GW + 1) if S.has_gameweek(data, gw)]


def _map(fn, items, workers: int) -> list:
    """Across processes, or in this one when workers <= 1 (tests, debugging)."""
    if workers <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(workers, initializer=_quiet) as ex:
        return list(ex.map(fn, items, chunksize=1))


def fill_cache(seasons: list[str], alphas: list[float], workers: int) -> None:
    todo = [(s, gw, a) for a in alphas for s in seasons for gw in gameweeks(s)
            if not _cache_path(s, gw, a).exists()]
    if not todo:
        return
    logger.warning("scoring %d gameweeks into %s", len(todo), CACHE_DIR)
    _map(_score_task, todo, workers)


def derive_horizon(scored: pd.DataFrame, order: list[int], gw: int, horizon: int) -> pd.DataFrame:
    """
    The frame score_gameweek(horizon) would have produced, from one scored at
    SCORE_HORIZON. xp_horizon is the decayed sum of the first `horizon`
    per-gameweek columns, accumulated in the model's own order; the frame is
    then sorted exactly as the model sorts it, starting from the model's
    pre-sort order, because the sort is not stable and row order reaches the
    solver.
    """
    import xp_model as X

    if horizon > SCORE_HORIZON:
        raise ValueError(f"horizon {horizon} exceeds the cached {SCORE_HORIZON}")
    decay = X.ModelConfig(horizon=horizon).horizon_decay
    out = scored.set_index("id").loc[order].reset_index()[scored.columns]
    total = pd.Series(0.0, index=out.index)
    for i, ev in enumerate(range(gw, min(gw + SCORE_HORIZON, 39))):
        if i < horizon:
            total += out[f"xp_gw{ev}"] * (decay ** i)
    out["xp_horizon"] = total
    out["value"] = out["xp_horizon"] / out["cost"].clip(lower=3.5)
    return out.sort_values("xp_horizon", ascending=False).reset_index(drop=True)


def arm_frame(arm: Arm, season: str, gw: int) -> pd.DataFrame:
    """What this arm's optimiser sees at this deadline. A copy; safe to mutate."""
    return _arm_frame(arm.resolved(), season, gw).copy()


@functools.lru_cache(maxsize=4096)
def _arm_frame(arm: Arm, season: str, gw: int) -> pd.DataFrame:
    scored, order = load_scored(season, gw, arm.alpha)
    frame = derive_horizon(scored, order, gw, arm.horizon)
    own = S.template_ownership(_season(season).gw_df, gw)
    frame["selected_by_percent"] = frame["id"].map(own).fillna(0.0)
    if arm.noise_eps:
        # Seeded per (seed, season, gameweek) and assigned in id order, so a
        # draw does not depend on row order or on which process runs it.
        rng = np.random.default_rng([arm.noise_seed, zlib.crc32(season.encode()), gw])
        order = frame["id"].argsort().to_numpy()
        z = rng.standard_normal((2, len(frame)))
        frame.loc[frame.index[order], "xp_horizon"] += arm.noise_eps * z[0]
        frame.loc[frame.index[order], "xp_next"] += arm.noise_eps * z[1]
    return frame


# ---------------------------------------------------------------- replaying


def _play(arm: Arm, season: str, state: S.SquadState, gw: int):
    r = arm.resolved()
    return S.play_gameweek(
        _season(season), state, gw, arm_frame(arm, season, gw),
        horizon=r.horizon, max_hits=r.max_hits, free_transfer_value=r.free_transfer_value,
        ownership_weight=r.ownership_weight, bench_weight=r.bench_weight)


def _frozen_points(arm: Arm, season: str, squad: list[int], gws: list[int]) -> int:
    """What a squad scores over `gws` with no transfers, XI picked on the arm's xP."""
    data = _season(season)
    total = 0
    for gw in gws:
        frame = arm_frame(arm, season, gw)
        positions = frame.set_index("id")["position"].astype(int).to_dict()
        xp = frame.set_index("id")["xp_next"].to_dict()
        points, minutes = S._actuals(data.gw_df, gw)
        total += S.pick_xi(squad, positions, points, minutes, xp)[1]
    return total


def _baseline_path(season: str) -> Path:
    return CACHE_DIR / f"baseline_{season}_{Arm().key()}.pkl"


def _baseline_task(season: str) -> str:
    """
    The production trajectory: the state before every deadline, and each
    gameweek's row. Every arm forks from these states.
    """
    path = _baseline_path(season)
    if path.exists():
        return str(path)
    arm, state = Arm(), S.SquadState()
    states, rows = {}, {}
    for gw in gameweeks(season):
        states[gw] = state.copy()
        state, row = _play(arm, season, state, gw)
        if row is not None:
            rows[gw] = row
    states["end"] = state.copy()
    path.write_bytes(pickle.dumps({"states": states, "rows": rows}))
    return str(path)


def _decision(row: dict) -> tuple:
    return tuple(row["ins"]), tuple(row["outs"]), tuple(row["xi"]), row["captain"]


def _windows_task(args) -> list[dict]:
    season, arm, L, T = args
    base = pickle.loads(_baseline_path(season).read_bytes())
    played = [gw for gw in gameweeks(season) if gw in base["rows"]]
    out = []
    for i, start in enumerate(played):
        # Start at the second played gameweek: the first is the squad build,
        # which every arm makes from nothing, and there is nothing to fork.
        if i == 0 or i + L > len(played):
            continue
        window = played[i:i + L]
        after = played[i + L:i + L + T]
        state = base["states"][start]
        rows = []
        for gw in window:
            state, row = _play(arm, season, state, gw)
            rows.append(row)
        base_rows = [base["rows"][gw] for gw in window]
        flips = [gw for r, b, gw in zip(rows, base_rows, window) if _decision(r) != _decision(b)]
        base_end = base["states"][after[0]] if after else base["states"]["end"]
        out.append({
            "season": season, "start": start, "L": L, "arm": arm.name, "arm_key": arm.key(),
            "arm_net": sum(r["net"] for r in rows),
            "base_net": sum(b["net"] for b in base_rows),
            "arm_terminal": _frozen_points(arm, season, state.squad, after),
            "base_terminal": _frozen_points(Arm(), season, base_end.squad, after),
            "terminal_gws": len(after),
            "flipped": bool(flips),
            "first_flip": (window.index(flips[0]) if flips else -1),
            "flip_gws": len(flips),
            "arm_hits": sum(r["hits"] for r in rows),
            "base_hits": sum(b["hits"] for b in base_rows),
        })
    return out


def _quiet():
    logging.disable(logging.WARNING)


def run(arms: list[Arm], seasons=SEASONS, L: int = DEFAULT_L, T: int = DEFAULT_TERMINAL,
        workers: int = os.cpu_count() or 4) -> pd.DataFrame:
    fill_cache(seasons, sorted({a.resolved().alpha for a in arms} | {Arm().resolved().alpha}),
               workers)
    _map(_baseline_task, seasons, min(workers, len(seasons)))
    tasks = [(s, a, L, T) for a in arms for s in seasons]
    df = pd.concat([pd.DataFrame(rows) for rows in _map(_windows_task, tasks, workers)],
                   ignore_index=True)
    df["diff"] = df["arm_net"] - df["base_net"]
    df["diff_terminal"] = df["diff"] + df["arm_terminal"] - df["base_terminal"]
    return df


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="same,alpha+1e-6,alpha-1e-6")
    ap.add_argument("--L", type=int, default=DEFAULT_L, help="window length in gameweeks")
    ap.add_argument("--T", type=int, default=DEFAULT_TERMINAL, help="terminal gameweeks")
    ap.add_argument("--seasons", default=",".join(SEASONS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=None, help="CSV path (default data/replay/<arms>_L<L>.csv)")
    args = ap.parse_args()

    arms = [parse_arm(a) for a in args.arms.split(",")]
    df = run(arms, args.seasons.split(","), args.L, args.T, args.workers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"{'_'.join(a.name for a in arms)}_L{args.L}.csv"
    df.to_csv(out, index=False)

    print(f"{len(df)} windows -> {out}\n")
    print(f"  {'arm':<16}{'n':>5}{'flipped':>9}{'mean diff':>11}{'+terminal':>11}{'exactly 0':>11}")
    for name, g in df.groupby("arm", sort=False):
        print(f"  {name:<16}{len(g):>5}{g['flipped'].mean():>9.1%}{g['diff'].mean():>+11.2f}"
              f"{g['diff_terminal'].mean():>+11.2f}{(g['diff_terminal'] == 0).mean():>11.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
