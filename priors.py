"""
Season priors for the expected-points model.

Two things live here, both of which exist to answer the same question: what do
we believe about a player or a team *before* the current season has produced
any evidence?

  Player priors  per-90 rates (xG, xA, defensive contribution, saves, cards,
                 BPS) shrunk toward a positional mean. Shrinkage is the whole
                 point: a player with 200 minutes of history should be pulled
                 hard toward his position's average, a player with 3000 should
                 barely move. Without this the model spends August ranking
                 pre-season cameos above established starters.

  Team strength  attack and defence multipliers relative to the league mean,
                 derived from actual match results. Promoted sides have no
                 Premier League record at all, so they get an explicit prior
                 rather than a silent league-average default that would rate
                 them as mid-table.

Player and team identity is keyed on `code`, not `id`. FPL reassigns `id`
every season; `code` is stable across seasons and is the only safe join key.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("fpl_auto.priors")

HISTORY_DIR = Path(__file__).parent / "data" / "history"

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_IDS = {v: k for k, v in POSITION_NAMES.items()}

# Rate columns modelled per 90 minutes. Everything downstream of the xP model
# reads these names, so they are the contract between priors and xp_model.
RATE_COLUMNS = [
    "xg90",         # expected goals
    "xa90",         # expected assists
    "dc90",         # defensive contribution stat count (CBIT / CBIRT)
    "saves90",
    "bps90",
    "yellow90",
    "goals_conceded90",
]

# Pseudo-minutes of prior weight. 900 = ten full matches: a player needs about
# ten games before his own rates outweigh his position's average. Tuned so that
# a mid-season regular is essentially unshrunk while a fringe player is not
# promoted on the strength of one good cameo.
SHRINKAGE_MINUTES = 900.0

# Prior weight for start rate, measured in gameweeks rather than minutes.
# Eight games: enough that a handful of rotations does not swing the estimate,
# few enough that a settled starter or a permanent benchwarmer is recognised
# well inside a season.
SHRINKAGE_GAMEWEEKS = 8.0

# Weight applied to each season going backwards from the most recent. Last
# season dominates; the one before informs but does not drive.
SEASON_DECAY = 0.45

# Promoted sides, having no Premier League record, are assigned these
# multipliers. Derived from the last several seasons of promoted-team
# performance: they score well below and concede well above the league mean.
PROMOTED_ATTACK = 0.78
PROMOTED_DEFENCE = 1.28

# Home advantage as a multiplier on expected goals. ~1.12/0.89 is the standard
# split and is stable season to season.
HOME_ATTACK_BOOST = 1.12
AWAY_ATTACK_BOOST = 0.89


@dataclass
class TeamStrength:
    """Attack and defence multipliers relative to the league mean (1.0 = average)."""

    code: int
    name: str
    attack: float
    defence: float
    matches: int
    is_promoted: bool = False

    def expected_goals_for(self, opponent: "TeamStrength", *, home: bool, league_mean: float) -> float:
        boost = HOME_ATTACK_BOOST if home else AWAY_ATTACK_BOOST
        return league_mean * self.attack * opponent.defence * boost

    def expected_goals_against(self, opponent: "TeamStrength", *, home: bool, league_mean: float) -> float:
        return opponent.expected_goals_for(self, home=not home, league_mean=league_mean)


@dataclass
class PriorSet:
    """Everything the xP model needs that is not in the live bootstrap."""

    players: pd.DataFrame                      # indexed by player `code`
    teams: dict[int, TeamStrength] = field(default_factory=dict)   # keyed by team `code`
    league_mean_goals: float = 1.45            # goals per team per match
    positional: pd.DataFrame | None = None     # per-position mean rates
    # Defensive-contribution volume by club, and the club each player's own
    # rate was earned at, so a summer transfer can be corrected for style.
    team_dc_factor: dict[int, float] = field(default_factory=dict)
    player_dc_team: pd.Series | None = None

    def team_by_code(self, code: int) -> TeamStrength | None:
        return self.teams.get(code)

    def validate(self, min_players: int = 200, min_teams: int = 15) -> None:
        """
        Refuse to run on priors too thin to model with.

        Without this the failure is silent and expensive: an empty history
        directory - the default on a fresh CI checkout, since the dataset is
        gitignored - yields empty priors, every per-90 rate falls back to zero,
        and the optimiser cheerfully submits a squad chosen by nothing at all.
        Better to abort the run and keep last week's team.
        """
        problems = []
        if self.players.empty or len(self.players) < min_players:
            problems.append(f"only {len(self.players)} player priors (need {min_players})")
        if self.positional is None or self.positional.empty:
            problems.append("no positional means")
        if len(self.teams) < min_teams:
            problems.append(f"only {len(self.teams)} team strengths (need {min_teams})")

        if problems:
            raise RuntimeError(
                "priors are unusable: " + "; ".join(problems)
                + ". Run `python fetch_data.py` to populate data/history/."
            )


# ──────────────────────── loading ────────────────────────


def _season_dir(season: str) -> Path:
    return HISTORY_DIR / season


def available_seasons() -> list[str]:
    """Seasons present on disk, oldest first."""
    if not HISTORY_DIR.exists():
        return []
    return sorted(d.name for d in HISTORY_DIR.iterdir() if d.is_dir() and (d / "merged_gw.csv").exists())


def load_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Load one season's per-gameweek rows, player metadata and fixtures.

    Returns None rather than raising when a season is missing: history is a
    third-party dataset that lags the live game, and a missing season must
    degrade the priors, never break the run.
    """
    sd = _season_dir(season)
    try:
        gw = pd.read_csv(sd / "merged_gw.csv")
        raw = pd.read_csv(sd / "players_raw.csv")
        fixtures = pd.read_csv(sd / "fixtures.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.warning("season %s unavailable (%s); skipping", season, e)
        return None

    if "code" not in raw.columns or "id" not in raw.columns:
        logger.warning("season %s players_raw missing code/id; skipping", season)
        return None

    # merged_gw keys players by that season's element id; remap onto stable code.
    id_to_code = raw.set_index("id")["code"].to_dict()
    gw = gw.copy()
    gw["code"] = gw["element"].map(id_to_code)
    unmapped = gw["code"].isna().sum()
    if unmapped:
        logger.warning("season %s: %d/%d gw rows had no code mapping", season, unmapped, len(gw))
        gw = gw.dropna(subset=["code"])
    gw["code"] = gw["code"].astype(int)

    return gw, raw, fixtures


# ──────────────────────── player priors ────────────────────────


def _aggregate_player_season(gw: pd.DataFrame) -> pd.DataFrame:
    """Sum a season's per-gameweek rows into per-player totals."""
    gw = gw.copy()

    # Position arrives as a string in merged_gw; normalise to the FPL integer.
    if "position" in gw.columns and gw["position"].dtype == object:
        gw["element_type"] = gw["position"].map(POSITION_IDS)
    gw["element_type"] = gw.get("element_type", pd.Series(np.nan, index=gw.index))

    # Columns the dataset gained partway through its life (defensive_contribution
    # only exists from 2025-26, expected_* from 2022-23). A season that predates
    # a stat carries no information about it, so it must aggregate to NaN and be
    # skipped by the weighted blend. Defaulting to 0.0 instead would treat
    # "not recorded" as "recorded as none" and halve the rate for every player.
    missing: list[str] = []
    for col in ["defensive_contribution", "expected_goals", "expected_assists", "starts", "saves", "bps", "yellow_cards", "goals_conceded"]:
        if col not in gw.columns:
            gw[col] = np.nan
            missing.append(col)
    if missing:
        logger.info("season lacks %s; those rates will be omitted from the blend", ", ".join(missing))

    # min_count=1 so a stat that is NaN for the whole season aggregates to NaN
    # rather than 0.0, which is what keeps a missing stat out of the blend.
    def _sum(series: pd.Series) -> float:
        return series.sum(min_count=1)

    agg = gw.groupby("code").agg(
        minutes=("minutes", "sum"),
        appearances=("minutes", lambda s: int((s > 0).sum())),
        # Gameweeks the player was registered for, including ones he did not
        # play. This is the denominator for start rate: the dataset carries a
        # row per squad player per gameweek, so dividing starts by appearances
        # would condition on having played and rate an unused backup keeper as
        # a nailed starter.
        squad_gameweeks=("minutes", "size"),
        starts=("starts", _sum),
        xg=("expected_goals", _sum),
        xa=("expected_assists", _sum),
        dc=("defensive_contribution", _sum),
        saves=("saves", _sum),
        bps=("bps", _sum),
        yellows=("yellow_cards", _sum),
        goals_conceded=("goals_conceded", _sum),
        total_points=("total_points", _sum),
        element_type=("element_type", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan),
    )
    return agg


def _to_per90(agg: pd.DataFrame) -> pd.DataFrame:
    """Convert season totals to per-90 rates. Zero-minute players yield NaN, not inf."""
    per90 = np.where(agg["minutes"] > 0, 90.0 / agg["minutes"].replace(0, np.nan), np.nan)
    out = pd.DataFrame(index=agg.index)
    out["minutes"] = agg["minutes"]
    out["appearances"] = agg["appearances"]
    out["squad_gameweeks"] = agg["squad_gameweeks"]
    out["starts"] = agg["starts"]
    out["element_type"] = agg["element_type"]
    out["xg90"] = agg["xg"] * per90
    out["xa90"] = agg["xa"] * per90
    out["dc90"] = agg["dc"] * per90
    out["saves90"] = agg["saves"] * per90
    out["bps90"] = agg["bps"] * per90
    out["yellow90"] = agg["yellows"] * per90
    out["goals_conceded90"] = agg["goals_conceded"] * per90
    # Marginal P(start in a given gameweek), not P(start | played).
    out["start_rate"] = np.where(
        agg["squad_gameweeks"] > 0,
        agg["starts"] / agg["squad_gameweeks"].replace(0, np.nan),
        np.nan,
    )
    return out


def _positional_means(per90: pd.DataFrame) -> pd.DataFrame:
    """
    Minutes-weighted mean rate per position, over players with a real sample.

    Minutes-weighted rather than a flat mean so that a handful of 20-minute
    cameos with freak rates cannot drag a position's prior around.
    """
    sample = per90[per90["minutes"] >= 450]
    rows = {}
    for pos_id in POSITION_NAMES:
        sub = sample[sample["element_type"] == pos_id]
        if sub.empty:
            sub = per90[per90["element_type"] == pos_id]
        if sub.empty:
            rows[pos_id] = {c: 0.0 for c in RATE_COLUMNS} | {"start_rate": 0.5}
            continue
        w = sub["minutes"].to_numpy(dtype=float)
        entry = {}
        for col in RATE_COLUMNS:
            vals = sub[col].to_numpy(dtype=float)
            ok = np.isfinite(vals)
            entry[col] = float(np.average(vals[ok], weights=w[ok])) if ok.any() and w[ok].sum() > 0 else 0.0

        # Start rate is deliberately taken over *every* registered player, not
        # the 450-minute sample used for the rate columns. Restricting it to
        # players who accumulated minutes would condition on being a starter
        # and produce a positional prior near 1.0 for goalkeepers.
        full = per90[per90["element_type"] == pos_id]
        sr = full["start_rate"].to_numpy(dtype=float)
        n = full["squad_gameweeks"].to_numpy(dtype=float)
        ok = np.isfinite(sr) & np.isfinite(n) & (n > 0)
        entry["start_rate"] = float(np.average(sr[ok], weights=n[ok])) if ok.any() else 0.5
        rows[pos_id] = entry

    return pd.DataFrame.from_dict(rows, orient="index")


def build_player_priors(seasons: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build shrunk per-90 priors keyed by stable player `code`.

    Seasons are combined with exponential decay so the most recent dominates,
    then each player's blended rate is shrunk toward his positional mean in
    proportion to how few minutes back it.

    Returns (player_priors, positional_means).
    """
    seasons = seasons or available_seasons()
    if not seasons:
        logger.warning("no history seasons on disk; priors will be empty")
        return pd.DataFrame(), pd.DataFrame()

    # Most recent first, so decay index 0 is the freshest season.
    ordered = sorted(seasons, reverse=True)
    frames: list[tuple[float, pd.DataFrame]] = []

    for i, season in enumerate(ordered):
        loaded = load_season(season)
        if loaded is None:
            continue
        gw, _raw, _fx = loaded
        per90 = _to_per90(_aggregate_player_season(gw))
        weight = SEASON_DECAY**i
        frames.append((weight, per90))
        logger.info("priors: %s contributed %d players (weight %.2f)", season, len(per90), weight)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    # Effective minutes carry the season weight, so an old season counts for
    # proportionally less evidence rather than being averaged in as an equal.
    combined = []
    for weight, per90 in frames:
        f = per90.copy()
        f["weight"] = weight
        f["eff_minutes"] = f["minutes"] * weight
        combined.append(f)
    allf = pd.concat(combined)

    positional = _positional_means(pd.concat([f for _, f in frames]))

    def _blend(group: pd.DataFrame) -> pd.Series:
        w = group["eff_minutes"].to_numpy(dtype=float)
        out = {
            "minutes": float(group["minutes"].sum()),
            "eff_minutes": float(w.sum()),
            "appearances": float(group["appearances"].sum()),
            "eff_gameweeks": float((group["squad_gameweeks"] * group["weight"]).sum()),
            "element_type": group["element_type"].dropna().mode().iloc[0] if group["element_type"].notna().any() else np.nan,
        }
        for col in RATE_COLUMNS:
            vals = group[col].to_numpy(dtype=float)
            ok = np.isfinite(vals) & (w > 0)
            out[col] = float(np.average(vals[ok], weights=w[ok])) if ok.any() else np.nan

        # Start rate is weighted by gameweeks registered, not minutes played,
        # for the same reason its denominator is: a player who never starts has
        # no minutes but plenty of evidence.
        gwv = (group["squad_gameweeks"] * group["weight"]).to_numpy(dtype=float)
        sr = group["start_rate"].to_numpy(dtype=float)
        ok = np.isfinite(sr) & (gwv > 0)
        out["start_rate"] = float(np.average(sr[ok], weights=gwv[ok])) if ok.any() else np.nan
        return pd.Series(out)

    blended = allf.groupby(level=0).apply(_blend)

    # Shrink toward the positional mean. K pseudo-minutes of prior evidence.
    shrunk = blended.copy()
    for col in RATE_COLUMNS:
        pos_mean = shrunk["element_type"].map(positional[col]).astype(float)
        own = shrunk[col].astype(float)
        m = shrunk["eff_minutes"].astype(float).clip(lower=0.0)
        # Where a player has no rate at all, fall back entirely to the prior.
        own = own.fillna(pos_mean)
        shrunk[col] = (own * m + pos_mean * SHRINKAGE_MINUTES) / (m + SHRINKAGE_MINUTES)

    # Start rate shrinks on a games scale: SHRINKAGE_GAMEWEEKS of prior weight.
    sr_pos_mean = shrunk["element_type"].map(positional["start_rate"]).astype(float)
    sr_own = shrunk["start_rate"].astype(float).fillna(sr_pos_mean)
    sr_n = shrunk["eff_gameweeks"].astype(float).clip(lower=0.0)
    shrunk["start_rate"] = (
        (sr_own * sr_n + sr_pos_mean * SHRINKAGE_GAMEWEEKS) / (sr_n + SHRINKAGE_GAMEWEEKS)
    ).clip(0.0, 1.0)

    shrunk["prior_confidence"] = (shrunk["eff_minutes"] / (shrunk["eff_minutes"] + SHRINKAGE_MINUTES)).clip(0, 1)
    # Confidence in the start-rate estimate specifically, counted in gameweeks
    # registered. Kept separate from prior_confidence because a player can have
    # abundant evidence that he does not start (many gameweeks, no minutes),
    # which a minutes-based confidence would score as no evidence at all.
    shrunk["start_confidence"] = (shrunk["eff_gameweeks"] / (shrunk["eff_gameweeks"] + SHRINKAGE_GAMEWEEKS)).clip(0, 1)
    logger.info("priors: built %d player priors from %d seasons", len(shrunk), len(frames))
    return shrunk, positional


def _with_element_type(gw: pd.DataFrame) -> pd.DataFrame | None:
    """
    Return the frame with a usable integer `element_type`, or None.

    The historical dataset is third-party and its schema drifts: `position` is
    a name in some seasons, an integer in others, and `element_type` is present
    directly in others again. Assuming one of those shapes crashed a CI run on
    a freshly fetched copy while passing against the copy already on disk, so
    all three are handled and an unrecognisable season is skipped rather than
    raising.
    """
    df = gw.copy()
    if "element_type" in df.columns and df["element_type"].notna().any():
        df["element_type"] = pd.to_numeric(df["element_type"], errors="coerce")
        return df
    if "position" not in df.columns:
        return None
    if df["position"].dtype == object:
        df["element_type"] = df["position"].map(POSITION_IDS)
    else:
        df["element_type"] = pd.to_numeric(df["position"], errors="coerce")
    if df["element_type"].isna().all():
        return None
    return df


# ──────────────── defensive contribution, by club ────────────────

# Minutes a player needs in a season before his defensive rate says anything
# about his club's style rather than about a handful of cameo appearances.
DC_TEAM_MIN_MINUTES = 900

# How far a club move is allowed to move a player's defensive rate. The
# measured spread between clubs is real but modest - roughly 7 to 9 actions per
# 90 across the league - so a ratio far outside this band means one side of it
# was estimated from too little data, and the safe response is to damp it.
DC_FACTOR_BOUNDS = (0.75, 1.35)


def build_team_dc_factors(seasons: list[str] | None = None) -> dict[int, float]:
    """
    Per-club defensive-contribution volume, relative to the league mean.

    Defensive contribution is the one scoring term that is as much a property
    of the club as of the player: it counts tackles, interceptions, clearances
    and recoveries, and a side that keeps the ball simply has fewer of them to
    make. Measured over 2025-26, the spread runs from 6.9 actions per 90 at
    Liverpool to 9.1 at Everton - a 32% difference between the extremes that
    has nothing to do with the individuals involved.

    Returned as a multiplier keyed by team `code`, where 1.0 is league average.
    Only defenders and midfielders count: goalkeepers score no defensive
    contribution points, and forwards make too few actions for the average to
    be stable.
    """
    seasons = seasons or available_seasons()
    if not seasons:
        return {}

    ordered = sorted(seasons, reverse=True)
    totals: dict[int, float] = {}
    weights: dict[int, float] = {}

    for i, season in enumerate(ordered):
        loaded = load_season(season)
        if loaded is None:
            continue
        gw, _raw, _fx = loaded
        if "defensive_contribution" not in gw.columns or gw["defensive_contribution"].isna().all():
            # Seasons before 2025-26 predate the stat entirely.
            continue
        try:
            teams = pd.read_csv(_season_dir(season) / "teams.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            logger.warning("season %s missing teams.csv; skipped for DC factors", season)
            continue
        if "name" not in teams.columns or "code" not in teams.columns:
            continue
        name_to_code = teams.set_index("name")["code"].to_dict()

        df = _with_element_type(gw)
        if df is None:
            logger.warning("season %s has no usable position column; skipped for DC factors", season)
            continue
        df = df[df["element_type"].isin([2, 3])]
        if "team" not in df.columns:
            logger.warning("season %s merged_gw has no team column; skipped for DC factors", season)
            continue
        df = df.copy()
        df["team_code"] = df["team"].map(name_to_code)
        df = df.dropna(subset=["team_code"])

        per_player = df.groupby(["team_code", "element"]).agg(
            minutes=("minutes", "sum"),
            dc=("defensive_contribution", "sum"),
        ).reset_index()
        per_player = per_player[per_player["minutes"] >= DC_TEAM_MIN_MINUTES]
        if per_player.empty:
            continue
        per_player["dc90"] = per_player["dc"] * 90.0 / per_player["minutes"]

        league_mean = float(per_player["dc90"].mean())
        if not np.isfinite(league_mean) or league_mean <= 0:
            continue

        weight = SEASON_DECAY**i
        for code, group in per_player.groupby("team_code"):
            factor = float(group["dc90"].mean()) / league_mean
            code = int(code)
            totals[code] = totals.get(code, 0.0) + factor * weight
            weights[code] = weights.get(code, 0.0) + weight

    factors = {code: totals[code] / weights[code] for code in totals if weights[code] > 0}
    if factors:
        logger.info(
            "DC club factors: %d teams, range %.2f-%.2f",
            len(factors), min(factors.values()), max(factors.values()),
        )
    return factors


def build_player_dc_team(seasons: list[str] | None = None) -> pd.Series:
    """
    The club each player's defensive rate was actually earned at, by `code`.

    Needed because the rate is only meaningful alongside the side that produced
    it. A player who moves clubs carries his old club's volume into the new
    one's style, and defensive contribution is a *threshold* award - two points
    at ten actions for a defender, twelve for a midfielder - so a rate sitting
    just above the line is far more fragile than the size of the shift implies.
    """
    seasons = seasons or available_seasons()
    if not seasons:
        return pd.Series(dtype=float)

    ordered = sorted(seasons, reverse=True)
    rows = []
    for i, season in enumerate(ordered):
        loaded = load_season(season)
        if loaded is None:
            continue
        gw, raw, _fx = loaded
        try:
            teams = pd.read_csv(_season_dir(season) / "teams.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if "name" not in teams.columns or "code" not in teams.columns:
            continue
        name_to_code = teams.set_index("name")["code"].to_dict()

        df = gw.copy()
        df["team_code"] = df["team"].map(name_to_code)
        df = df.dropna(subset=["team_code", "code"])
        agg = df.groupby(["code", "team_code"])["minutes"].sum().reset_index()
        agg["weighted"] = agg["minutes"] * (SEASON_DECAY**i)
        rows.append(agg[["code", "team_code", "weighted"]])

    if not rows:
        return pd.Series(dtype=float)

    allr = pd.concat(rows).groupby(["code", "team_code"])["weighted"].sum().reset_index()
    # The club he played the most weighted minutes for is the one his rate
    # describes; a mid-season mover is attributed to wherever he played more.
    best = allr.sort_values("weighted").groupby("code").tail(1)
    return best.set_index("code")["team_code"].astype(int)


# ──────────────────────── team strength ────────────────────────


def build_team_strength(
    seasons: list[str] | None = None,
    current_team_codes: dict[int, str] | None = None,
) -> tuple[dict[int, TeamStrength], float]:
    """
    Derive attack/defence multipliers from completed matches.

    current_team_codes maps team `code` -> name for the teams in the *current*
    season. Any of those with no match history is treated as promoted and given
    an explicit prior; without this they would silently default to league
    average and the model would rate a promoted side's defenders like a
    mid-table side's.
    """
    seasons = seasons or available_seasons()
    ordered = sorted(seasons, reverse=True)

    goals_for: dict[int, float] = {}
    goals_against: dict[int, float] = {}
    matches: dict[int, float] = {}
    names: dict[int, str] = {}
    total_goals = 0.0
    total_matches = 0.0

    for i, season in enumerate(ordered):
        loaded = load_season(season)
        if loaded is None:
            continue
        _gw, _raw, fixtures = loaded
        try:
            teams = pd.read_csv(_season_dir(season) / "teams.csv")
        except FileNotFoundError:
            logger.warning("season %s missing teams.csv; skipping for team strength", season)
            continue

        id_to_code = teams.set_index("id")["code"].to_dict()
        id_to_name = teams.set_index("id")["name"].to_dict()
        weight = SEASON_DECAY**i

        played = fixtures.dropna(subset=["team_h_score", "team_a_score"])
        for _, fx in played.iterrows():
            h, a = int(fx["team_h"]), int(fx["team_a"])
            hc, ac = id_to_code.get(h), id_to_code.get(a)
            if hc is None or ac is None:
                continue
            hs, as_ = float(fx["team_h_score"]), float(fx["team_a_score"])
            names.setdefault(hc, id_to_name.get(h, str(hc)))
            names.setdefault(ac, id_to_name.get(a, str(ac)))

            goals_for[hc] = goals_for.get(hc, 0.0) + hs * weight
            goals_against[hc] = goals_against.get(hc, 0.0) + as_ * weight
            matches[hc] = matches.get(hc, 0.0) + weight

            goals_for[ac] = goals_for.get(ac, 0.0) + as_ * weight
            goals_against[ac] = goals_against.get(ac, 0.0) + hs * weight
            matches[ac] = matches.get(ac, 0.0) + weight

            total_goals += (hs + as_) * weight
            total_matches += 2.0 * weight

    if total_matches == 0:
        logger.warning("no completed matches found; team strength falls back to neutral")
        league_mean = 1.45
    else:
        league_mean = total_goals / total_matches

    strengths: dict[int, TeamStrength] = {}
    for code, m in matches.items():
        if m < 5:  # too little evidence to trust
            continue
        strengths[code] = TeamStrength(
            code=code,
            name=names.get(code, str(code)),
            attack=(goals_for[code] / m) / league_mean if league_mean > 0 else 1.0,
            defence=(goals_against[code] / m) / league_mean if league_mean > 0 else 1.0,
            matches=int(m),
        )

    # Anyone in the current season without history is promoted.
    if current_team_codes:
        for code, name in current_team_codes.items():
            if code not in strengths:
                strengths[code] = TeamStrength(
                    code=code,
                    name=name,
                    attack=PROMOTED_ATTACK,
                    defence=PROMOTED_DEFENCE,
                    matches=0,
                    is_promoted=True,
                )
                logger.info("team %s (%s) has no PL history; applying promoted prior", name, code)

    logger.info(
        "team strength: %d teams, league mean %.2f goals/team/match",
        len(strengths), league_mean,
    )
    return strengths, league_mean


def build_priors(seasons: list[str] | None = None, current_team_codes: dict[int, str] | None = None) -> PriorSet:
    """Build the full prior set. This is the entry point the xP model calls."""
    players, positional = build_player_priors(seasons)
    teams, league_mean = build_team_strength(seasons, current_team_codes)
    return PriorSet(
        players=players,
        teams=teams,
        league_mean_goals=league_mean,
        positional=positional,
        team_dc_factor=build_team_dc_factors(seasons),
        player_dc_team=build_player_dc_team(seasons),
    )
