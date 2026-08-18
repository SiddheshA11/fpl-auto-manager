"""
Expected points model.

Produces, for every player and every upcoming gameweek, an estimate in *actual
FPL points*. That unit matters: it is what lets the optimiser compare a
transfer against the -4 hit it costs, and lets the chip engine compare a bench
boost against a triple captain. The previous scoring engine emitted a unitless
0-1 composite and then multiplied it by 10 to pretend it was points, which made
every threshold in the system arbitrary.

Structure of an estimate, per player per fixture:

    xP = appearance + goals + assists + clean sheet + defensive contribution
         + saves - goals conceded - cards + bonus

Every term is conditioned on expected minutes, because a 12-point ceiling on a
player who starts 40% of the time is not worth 12 points.

Scoring values are read from the live `game_config.scoring` block rather than
hardcoded, so a mid-season rule change does not silently rot the model.
"""
from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from priors import POSITION_NAMES, PriorSet

logger = logging.getLogger("fpl_auto.xp")

# FPL's short position codes, as used as keys inside game_config.scoring.
POSITION_CODES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Defensive contribution thresholds. The point *value* is in game_config, but
# the threshold is not published there, so it lives here. Both were verified
# against 2025-26 per-gameweek data: for DEF the stat is clearances_blocks_
# interceptions + tackles, for MID/FWD it additionally includes recoveries,
# and the award fires at these counts (100% agreement across 29,757 rows).
DC_THRESHOLD = {1: None, 2: 10, 3: 12, 4: 12}

# How far a club move may rescale a defensive rate. The measured spread is
# real but modest, so a ratio outside this band means one side of it came
# from too small a sample and is damped rather than trusted.
DC_FACTOR_BOUNDS = (0.75, 1.35)

# Dispersion of the defensive-contribution count around a player's own rate.
# Population variance/mean is ~2.0, but that is mostly *between* players; once
# conditioned on a player's own rate the residual is ~1.3, and a plain Poisson
# calibrated better against held-out data than a negative binomial fitted to
# the population figure (MAE 0.025 vs 0.037 for DEF). Left tunable because it
# is the single least certain constant in this file.
DC_DISPERSION = 1.0

# Empirical map from BPS per 90 to bonus points per 90, from 673 players with
# 8+ appearances across 2024-25 and 2025-26. Bonus is a rank-within-match award
# so the relationship is strongly convex; a linear coefficient underprices
# premium players badly. Interpolated, then held flat outside the range.
BONUS_CURVE_BPS = [3.8, 7.06, 9.65, 12.66, 15.41, 18.34, 21.11, 24.51, 27.88]
BONUS_CURVE_PTS = [0.0286, 0.0497, 0.0999, 0.1582, 0.2839, 0.4288, 0.6485, 0.799, 1.1664]

# Minutes model constants.
# Sharpness of the within-team competition for a starting shirt. Goalkeeper is
# effectively winner-take-all, so probability concentrates hard on the first
# choice; outfield places are genuinely rotated, so the contest is softer.
GK_COMPETITION_ALPHA = 3.0
OUTFIELD_COMPETITION_ALPHA = 1.5

STARTER_MINUTES = 82.0      # mean minutes for a player who starts
SUB_MINUTES = 18.0          # mean minutes for a player who comes off the bench
P60_GIVEN_START = 0.92      # a starter reaching the 60-minute appearance bonus

# How much current-season evidence is needed before it outweighs the prior.
# Lower than the priors' own figure because in-season form is more relevant to
# the next fixture than last season's aggregate.
CURRENT_SEASON_WEIGHT_MINUTES = 540.0

# Availability by FPL status flag, used when chance_of_playing is not published.
STATUS_AVAILABILITY = {"a": 1.0, "d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}

# P(start) implied by price alone, per position, from 1,416 player-seasons over
# 2024-25 and 2025-26. FPL prices a player according to the role it expects him
# to have, which makes price the best available signal for someone with no
# Premier League history at all - a new signing from abroad, or a promoted
# club's squad. The £4.0-4.5 tier is where backup keepers and bench fodder sit.
# Values are season-long marginal start rates; the elite reach 0.92-1.0, so
# they need no correction for the availability multiplier applied separately.
PRICE_START_PRIOR = {
    1: ([4.3, 4.85, 5.6], [0.157, 0.676, 0.707]),
    2: ([4.3, 4.85, 5.6], [0.304, 0.469, 0.543]),
    3: ([4.3, 4.85, 5.6, 6.85, 9.0], [0.042, 0.332, 0.378, 0.513, 0.604]),
    4: ([4.3, 4.85, 5.6, 6.85, 9.0], [0.018, 0.069, 0.350, 0.388, 0.729]),
}


@dataclass
class ModelConfig:
    horizon: int = 5                 # gameweeks to look ahead
    dc_dispersion: float = DC_DISPERSION
    # Discount applied to gameweek n of the horizon. Fixtures further out are
    # less certain (injuries, rotation, price changes) and are worth less to a
    # decision made today.
    horizon_decay: float = 0.84


# ──────────────────────── data loading ────────────────────────


def load_snapshot(path: Path | str) -> dict:
    """Load a gzipped or plain JSON snapshot."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def latest_snapshot(data_dir: Path | str, name: str) -> Path | None:
    """Most recent snapshot for an endpoint, or None."""
    files = sorted(Path(data_dir).glob(f"{name}_*.json.gz"), reverse=True)
    return files[0] if files else None


# ──────────────────────── the model ────────────────────────


class XPModel:
    """Expected points for every player over a horizon of gameweeks."""

    def __init__(
        self,
        bootstrap: dict,
        fixtures: list[dict],
        priors: PriorSet,
        config: ModelConfig | None = None,
    ):
        self.bootstrap = bootstrap
        self.fixtures = pd.DataFrame(fixtures)
        self.priors = priors
        self.config = config or ModelConfig()

        self.scoring = bootstrap["game_config"]["scoring"]
        self.teams = pd.DataFrame(bootstrap["teams"])
        self._team_id_to_code = self.teams.set_index("id")["code"].to_dict()
        self._team_id_to_name = self.teams.set_index("id")["short_name"].to_dict()

        self.players = self._build_player_frame()

    # ---------- player frame ----------

    def _build_player_frame(self) -> pd.DataFrame:
        """Merge live bootstrap rows with shrunk priors, keyed on stable code."""
        df = pd.DataFrame(self.bootstrap["elements"])

        numeric = [
            "minutes", "starts", "expected_goals", "expected_assists", "saves", "bps",
            "yellow_cards", "goals_conceded", "defensive_contribution", "now_cost",
            "selected_by_percent", "chance_of_playing_next_round", "total_points",
        ]
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

        df["cost"] = df["now_cost"] / 10.0
        df["position"] = df["element_type"]
        df["team_code"] = df["team"].map(self._team_id_to_code)
        df["team_name"] = df["team"].map(self._team_id_to_name)

        # Current-season per-90 rates. Early in a season `minutes` is ~0 for
        # everyone, so these are almost entirely NaN and the prior carries the
        # whole estimate. That is the intended behaviour, not a degenerate case.
        mins = df["minutes"].fillna(0.0)
        per90 = np.where(mins > 0, 90.0 / mins.replace(0, np.nan), np.nan)
        current = pd.DataFrame(index=df.index)
        current["xg90"] = df["expected_goals"] * per90
        current["xa90"] = df["expected_assists"] * per90
        current["dc90"] = df["defensive_contribution"] * per90
        current["saves90"] = df["saves"] * per90
        current["bps90"] = df["bps"] * per90
        current["yellow90"] = df["yellow_cards"] * per90
        current["goals_conceded90"] = df["goals_conceded"] * per90

        # Blend current season into the prior, weighted by minutes played.
        pri = self.priors.players
        joined = df.join(pri, on="code", rsuffix="_prior")
        w_cur = (mins / (mins + CURRENT_SEASON_WEIGHT_MINUTES)).fillna(0.0)

        # Before a ball is kicked, `minutes` and every counting stat on the
        # bootstrap still hold *last* season's totals - 400 players carrying up
        # to 3420 minutes with no gameweek finished. Blending those in as
        # current-season evidence weights them at 86% for an ever-present, which
        # double-counts a season the priors already contain and routes around
        # the shrinkage in priors.py that exists precisely to stop a raw rate
        # being trusted whole. Until a gameweek has actually been played, the
        # priors are the only evidence there is.
        if not any(e.get("finished") for e in self.bootstrap.get("events", [])):
            played = int((mins > 0).sum())
            if played:
                logger.info(
                    "no gameweek finished yet; ignoring last season's totals on %d players "
                    "and taking rates from the priors alone", played,
                )
            w_cur = pd.Series(0.0, index=df.index)

        for col in ["xg90", "xa90", "dc90", "saves90", "bps90", "yellow90", "goals_conceded90"]:
            prior_vals = joined[col] if col in joined.columns else pd.Series(np.nan, index=df.index)
            prior_vals = pd.to_numeric(prior_vals, errors="coerce")
            # Fall back to the positional mean for anyone with no prior at all
            # (new signings, promoted-club players, academy debutants).
            if self.priors.positional is not None:
                pos_default = df["position"].map(self.priors.positional[col]).astype(float)
            else:
                pos_default = pd.Series(0.0, index=df.index)
            prior_vals = prior_vals.fillna(pos_default)
            cur_vals = current[col].fillna(prior_vals)
            df[col] = w_cur * cur_vals + (1.0 - w_cur) * prior_vals

        df["dc90"] = self._adjust_dc_for_club_move(df, w_cur)

        # Start rate. Two things differ from the rate columns above.
        #
        # First, the denominator is gameweeks elapsed, not games' worth of
        # minutes: dividing starts by (minutes/90) conditions on having played
        # and rates an unused substitute as a nailed starter.
        #
        # Second, a player with no history falls back to what his price
        # implies rather than to the positional mean, because the positional
        # mean is the same number for a £4.0 backup and a £9.0 signing.
        prior_sr = pd.to_numeric(joined.get("start_rate"), errors="coerce")
        price_sr = self._price_implied_start_rate(df)
        prior_confidence = pd.to_numeric(joined.get("prior_confidence"), errors="coerce").fillna(0.0)
        # Gameweek-scaled, so "registered all season, never started" counts as
        # the strong evidence it is rather than as no evidence.
        start_confidence = pd.to_numeric(joined.get("start_confidence"), errors="coerce").fillna(0.0)

        # Where the prior is thin, lean on price; where it is rich, trust it.
        prior_sr = prior_sr.fillna(price_sr)
        blended_prior = start_confidence * prior_sr + (1.0 - start_confidence) * price_sr

        events_done = sum(1 for e in self.bootstrap.get("events", []) if e.get("finished"))
        starts = df["starts"].fillna(0.0).clip(lower=0.0)
        if events_done > 0:
            cur_sr = (starts / events_done).clip(0.0, 1.0)
            w_sr = min(events_done / 8.0, 1.0)  # eight games before form fully displaces the prior
            df["start_rate"] = w_sr * cur_sr + (1.0 - w_sr) * blended_prior
        else:
            df["start_rate"] = blended_prior

        df["prior_confidence"] = prior_confidence

        return df

    def _price_implied_start_rate(self, df: pd.DataFrame) -> pd.Series:
        """P(start) implied by price alone, per position."""
        out = pd.Series(0.3, index=df.index)
        for pos_id, (xs, ys) in PRICE_START_PRIOR.items():
            mask = df["position"] == pos_id
            if mask.any():
                out.loc[mask] = np.interp(df.loc[mask, "cost"], xs, ys, left=ys[0], right=ys[-1])
        return out

    # ---------- minutes ----------

    def _availability(self) -> pd.Series:
        """
        P(player is fit and selectable).

        chance_of_playing_next_round is authoritative when FPL publishes it;
        otherwise fall back to the status flag. A player flagged 'd' with no
        published percentage is a genuine doubt, not a certainty.
        """
        df = self.players
        chance = df["chance_of_playing_next_round"] / 100.0
        fallback = df["status"].map(STATUS_AVAILABILITY).fillna(0.5)
        return chance.where(chance.notna(), fallback).clip(0.0, 1.0)

    def _normalise_starts_within_team(self, p_start: pd.Series) -> pd.Series:
        """
        Force each team's start probabilities to sum to the eleven shirts that
        actually exist: one goalkeeper and ten outfielders.

        Without this, start probability is estimated per player in isolation and
        nothing stops a club's two £5.0 keepers from both being 68% likely to
        start. It also produces the right behaviour when a first choice is
        injured: his probability drops to zero and the freed mass is
        redistributed to the understudy, who becomes a near-certain starter.

        Allocation within a group is proportional to rate**alpha rather than to
        the rate itself. Straight proportional scaling punishes a settled first
        choice for merely having credible cover on the books - a keeper rated
        0.87 behind two plausible deputies gets divided down to 0.65, which is
        not how team selection works. Raising to a power concentrates the
        probability on the likeliest starter. Goalkeeper is close to a binary
        contest so it gets the sharper exponent; outfield rotation is real, so
        it gets a gentler one.
        """
        df = self.players
        out = p_start.copy().astype(float)

        for team_id, idx in df.groupby("team").groups.items():
            for shirts, alpha, mask in (
                (1.0, GK_COMPETITION_ALPHA, df.loc[idx, "position"] == 1),
                (10.0, OUTFIELD_COMPETITION_ALPHA, df.loc[idx, "position"] != 1),
            ):
                sel = pd.Index(idx)[mask.to_numpy()]
                if len(sel) == 0:
                    continue
                raw = out.loc[sel].clip(lower=0.0)
                weights = raw**alpha
                total = float(weights.sum())
                if total <= 1e-12:
                    # Whole group unavailable (or no data): spread evenly rather
                    # than divide by zero. Rare, and self-corrects once FPL
                    # publishes availability.
                    out.loc[sel] = min(shirts / len(sel), 1.0)
                    continue
                out.loc[sel] = (weights * (shirts / total)).clip(upper=0.98)

        return out.clip(0.0, 1.0)

    def minutes_model(self) -> pd.DataFrame:
        """P(appear), P(60+ minutes) and expected minutes for each player."""
        df = self.players
        avail = self._availability()
        start_rate = df["start_rate"].clip(0.0, 1.0).fillna(0.0)

        # Non-starters who still appear: a fixed share of the remaining
        # probability mass. Bench cameos are frequent but low-minute.
        sub_rate = 0.35

        p_start = avail * start_rate
        p_start = self._normalise_starts_within_team(p_start)
        p_sub = avail * (1.0 - start_rate) * sub_rate
        p_appear = (p_start + p_sub).clip(0.0, 1.0)
        p60 = (p_start * P60_GIVEN_START).clip(0.0, 1.0)
        exp_minutes = p_start * STARTER_MINUTES + p_sub * SUB_MINUTES

        return pd.DataFrame({
            "availability": avail,
            "p_start": p_start,
            "p_appear": p_appear,
            "p60": p60,
            "exp_minutes": exp_minutes,
        }, index=df.index)

    # ---------- fixtures ----------

    def fixtures_for_event(self, event: int) -> pd.DataFrame:
        """Fixtures in a gameweek, one row per team (so doubles appear twice)."""
        fx = self.fixtures
        if fx.empty or "event" not in fx.columns:
            return pd.DataFrame(columns=["team_id", "opponent_id", "is_home", "event"])

        gw = fx[fx["event"] == event]
        rows = []
        for _, f in gw.iterrows():
            rows.append({"team_id": int(f["team_h"]), "opponent_id": int(f["team_a"]), "is_home": True, "event": event})
            rows.append({"team_id": int(f["team_a"]), "opponent_id": int(f["team_h"]), "is_home": False, "event": event})
        return pd.DataFrame(rows)

    def _team_strength(self, team_id: int):
        code = self._team_id_to_code.get(team_id)
        return self.priors.team_by_code(code) if code is not None else None

    def _goal_expectations(self, team_id: int, opponent_id: int, is_home: bool) -> tuple[float, float]:
        """(expected goals for, expected goals against) for a team in one fixture."""
        lm = self.priors.league_mean_goals
        t = self._team_strength(team_id)
        o = self._team_strength(opponent_id)
        if t is None or o is None:
            # No strength data: fall back to the league average fixture rather
            # than dropping the fixture, so the player still gets an estimate.
            return lm, lm
        return (
            t.expected_goals_for(o, home=is_home, league_mean=lm),
            t.expected_goals_against(o, home=is_home, league_mean=lm),
        )

    # ---------- scoring terms ----------

    def _pts(self, key: str, position: pd.Series) -> pd.Series:
        """Look up a positional point value from the live scoring config."""
        val = self.scoring.get(key)
        if isinstance(val, dict):
            return position.map({pid: float(val.get(code, 0.0)) for pid, code in POSITION_CODES.items()})
        return pd.Series(float(val or 0.0), index=position.index)

    def _adjust_dc_for_club_move(self, df: pd.DataFrame, w_cur: pd.Series) -> pd.Series:
        """
        Rescale a transferred player's defensive rate to his new club's style.

        Defensive contribution is the one scoring term that is as much a
        property of the side as of the player: it counts tackles,
        interceptions, clearances and recoveries, and a team that keeps the
        ball has fewer of them to make. The measured spread across 2025-26 runs
        0.86 to 1.13 relative to the league mean.

        That is a modest correction, but the award is a *threshold* - two
        points at ten actions for a defender, twelve for a midfielder - so it
        matters most exactly where it is most likely to bite: a player whose
        rate sits just above the line at a low-possession club can drop below
        it at a high-possession one on a shift far smaller than the raw
        percentage suggests.

        Only the prior is adjusted. Minutes played for the current club need no
        correction - they were earned in the style being corrected for - so the
        rescaling fades out as the season supplies real evidence.
        """
        dc90 = pd.to_numeric(df["dc90"], errors="coerce")
        factors = self.priors.team_dc_factor
        prior_team = self.priors.player_dc_team
        if not factors or prior_team is None or prior_team.empty:
            return dc90

        old_code = df["code"].map(prior_team)
        old_factor = old_code.map(factors)
        new_factor = df["team_code"].map(factors)

        # A player with no history, or at a club with no measured factor, is
        # left alone rather than pushed toward an average he was never near.
        ratio = (new_factor / old_factor).astype(float)
        ratio = ratio.where(np.isfinite(ratio), 1.0).fillna(1.0)
        ratio = ratio.clip(*DC_FACTOR_BOUNDS)

        # Only the prior share of the blend is rescaled.
        prior_share = (1.0 - pd.to_numeric(w_cur, errors="coerce").fillna(0.0))
        effective = 1.0 + (ratio - 1.0) * prior_share

        moved = (old_code.notna() & df["team_code"].notna() & (old_code != df["team_code"]))
        adjusted = dc90.where(~moved, dc90 * effective)

        n_moved = int((moved & (effective - 1.0).abs().gt(0.01)).sum())
        if n_moved:
            logger.info("defensive contribution rescaled for %d players who changed club", n_moved)
        return adjusted

    def _expected_bonus(self, bps90: pd.Series, minutes_share: pd.Series) -> pd.Series:
        """Interpolate the empirical BPS-to-bonus curve, scaled by minutes."""
        b = np.interp(bps90.fillna(0.0), BONUS_CURVE_BPS, BONUS_CURVE_PTS,
                      left=BONUS_CURVE_PTS[0], right=BONUS_CURVE_PTS[-1])
        return pd.Series(b, index=bps90.index) * minutes_share

    def _tail_probability(self, mu: np.ndarray, threshold: int) -> np.ndarray:
        """P(X >= threshold) for a count with mean mu."""
        k = self.config.dc_dispersion
        mu = np.maximum(mu, 1e-9)
        if k <= 1.0:
            return 1.0 - poisson.cdf(threshold - 1, mu)
        # Negative binomial with the requested variance/mean ratio.
        from scipy.stats import nbinom
        r = mu / (k - 1.0)
        return 1.0 - nbinom.cdf(threshold - 1, r, r / (r + mu))

    def _p_dc_award(
        self, dc90: pd.Series, position: pd.Series, p_start: pd.Series, p_sub: pd.Series
    ) -> pd.Series:
        """
        P(hits the defensive contribution threshold).

        A threshold event, so this needs the tail of a count distribution, not
        a linear term on the rate. Modelling it linearly - the common shortcut -
        overvalues players who accumulate steadily below the line and
        undervalues those who spike past it.

        The tail is evaluated separately for starting and substitute
        appearances rather than once at average minutes. The tail is convex in
        the rate, so by Jensen the probability at mean minutes is far below the
        mean of the probabilities: a player who starts half the time was scored
        at roughly a fifth of his true chance, silently underpricing every
        rotation-risk defender in the pool.
        """
        rate = dc90.fillna(0.0).clip(lower=0.0)
        out = pd.Series(0.0, index=dc90.index)

        for pos_id, thr in DC_THRESHOLD.items():
            if thr is None:
                continue
            mask = position == pos_id
            if not mask.any():
                continue
            r = rate[mask].to_numpy(dtype=float)
            p_started = self._tail_probability(r * (STARTER_MINUTES / 90.0), thr)
            p_benched = self._tail_probability(r * (SUB_MINUTES / 90.0), thr)
            out.loc[mask] = (
                p_start[mask].to_numpy(dtype=float) * p_started
                + p_sub[mask].clip(lower=0.0).to_numpy(dtype=float) * p_benched
            )

        return out.clip(0.0, 1.0)

    # ---------- the estimate ----------

    def expected_points_for_fixture(self, team_id: int, opponent_id: int, is_home: bool) -> pd.Series:
        """xP for every player, were their team to play this single fixture."""
        df = self.players
        mm = self.minutes_model()
        pos = df["position"]

        gf, ga = self._goal_expectations(team_id, opponent_id, is_home)

        # A player's historical per-90 rates were accumulated in his own team's
        # attacking context, so his club's quality is *already inside the rate*.
        # The fixture adjustment must therefore carry only what is new: the
        # opponent's defence and the venue. Dividing gf by the league mean
        # alone leaves own_attack in the multiplier and applies it twice -
        # inflating every Man City attacker by ~41% and deflating a good
        # forward at a poor club by the same logic.
        lm = max(self.priors.league_mean_goals, 1e-6)
        t = self._team_strength(team_id)
        own_attack = t.attack if t else 1.0
        own_defence = t.defence if t else 1.0

        attack_mult = gf / (lm * max(own_attack, 1e-6))
        minutes_share = mm["exp_minutes"] / 90.0

        # Appearance: 2 for 60+, 1 for anything less.
        long_play = float(self.scoring.get("long_play", 2))
        short_play = float(self.scoring.get("short_play", 1))
        appearance = mm["p60"] * long_play + (mm["p_appear"] - mm["p60"]).clip(lower=0) * short_play

        goals = df["xg90"].fillna(0.0) * minutes_share * attack_mult * self._pts("goals_scored", pos)
        assists = df["xa90"].fillna(0.0) * minutes_share * attack_mult * self._pts("assists", pos)

        # Clean sheet requires 60 minutes, so it is gated on p60 not p_appear.
        p_cs = float(np.exp(-ga))
        clean_sheet = mm["p60"] * p_cs * self._pts("clean_sheets", pos)

        # Goals conceded: -1 per 2 conceded, for GK and DEF only. Expectation
        # of floor(X/2) under Poisson(ga), truncated at a sane upper bound.
        gc_pts = self._pts("goals_conceded", pos)
        ks = np.arange(0, 11)
        pmf = poisson.pmf(ks, ga)
        e_floor_half = float(np.sum(pmf * (ks // 2)))
        goals_conceded = mm["p60"] * e_floor_half * gc_pts.abs() * -1.0

        # Saves scale with how much shooting the opponent does. Same correction
        # as the attack multiplier: saves90 already embeds his own defence, so
        # only the opponent's attack and the venue may be applied here.
        saves_mult = ga / (lm * max(own_defence, 1e-6))
        saves_rate = df["saves90"].fillna(0.0) * minutes_share * saves_mult
        saves = saves_rate / 3.0 * float(self.scoring.get("saves", 1))

        dc = self._p_dc_award(df["dc90"], pos, mm["p_start"], mm["p_appear"] - mm["p60"]) * self._pts(
            "defensive_contribution", pos
        )

        cards = df["yellow90"].fillna(0.0) * minutes_share * float(self.scoring.get("yellow_cards", -1))

        bonus = self._expected_bonus(df["bps90"], minutes_share) * float(self.scoring.get("bonus", 1))

        total = appearance + goals + assists + clean_sheet + goals_conceded + saves + dc + cards + bonus
        return total.clip(lower=0.0)

    def expected_points(self, events: list[int]) -> pd.DataFrame:
        """
        xP per player per gameweek, plus a decayed horizon total.

        A player whose team blanks in a gameweek scores 0 for it; a player with
        a double plays twice and both fixtures are summed. This is what makes
        the optimiser handle DGWs and BGWs without special-casing them.
        """
        df = self.players
        out = pd.DataFrame(index=df.index)
        out["id"] = df["id"]
        out["code"] = df["code"]
        out["web_name"] = df["web_name"]
        out["position"] = df["position"]
        out["team"] = df["team"]
        out["team_name"] = df["team_name"]
        out["cost"] = df["cost"]
        out["selected_by_percent"] = df["selected_by_percent"]
        out["status"] = df["status"]

        mm = self.minutes_model()
        out["exp_minutes"] = mm["exp_minutes"]
        out["availability"] = mm["availability"]
        out["p_start"] = mm["p_start"]

        horizon_total = pd.Series(0.0, index=df.index)

        for i, ev in enumerate(events):
            fx = self.fixtures_for_event(ev)
            gw_points = pd.Series(0.0, index=df.index)
            fixture_count = pd.Series(0, index=df.index)

            for _, f in fx.iterrows():
                mask = df["team"] == f["team_id"]
                if not mask.any():
                    continue
                xp = self.expected_points_for_fixture(int(f["team_id"]), int(f["opponent_id"]), bool(f["is_home"]))
                gw_points[mask] += xp[mask]
                fixture_count[mask] += 1

            out[f"xp_gw{ev}"] = gw_points
            out[f"fixtures_gw{ev}"] = fixture_count
            horizon_total += gw_points * (self.config.horizon_decay**i)

        out["xp_next"] = out[f"xp_gw{events[0]}"] if events else 0.0
        out["xp_horizon"] = horizon_total
        out["value"] = out["xp_horizon"] / out["cost"].clip(lower=3.5)

        return out.sort_values("xp_horizon", ascending=False).reset_index(drop=True)


def next_events(bootstrap: dict, n: int) -> list[int]:
    """
    The next n gameweeks to plan for.

    Anchored on the gameweek FPL marks `is_next`, not on the first unfinished
    one. A gameweek in progress is still unfinished, so during a midweek round
    - or on any run that fires before the previous week is settled - filtering
    on `finished` alone would return the *current* gameweek. Every downstream
    quantity, captaincy and chip valuation included, would then quietly
    describe a week whose deadline has already passed.
    """
    events = sorted(bootstrap["events"], key=lambda e: e["id"])
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    if nxt is None:
        nxt = next((e["id"] for e in events if not e.get("finished")), None)
    if nxt is None:
        return []
    return [e["id"] for e in events if e["id"] >= nxt][:n]
