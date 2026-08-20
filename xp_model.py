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
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

import news

from priors import PENALTY_XG90, POSITION_NAMES, PriorSet

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

# Penalty saves per goalkeeper appearance, measured across 2024-25 and
# 2025-26. Worth ~3 points a season and previously not modelled at all.
PENALTY_SAVE_RATE = 0.0163

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

# Per-position correction to that curve.
#
# Bonus is a *rank-within-match* award, so its expectation depends on the
# dispersion of a player's match BPS, not only on its mean. A forward's BPS is
# goal-spiked and lands on the podium or nowhere; a midfielder's accumulates
# steadily just below it. At matched bps90 a forward earns roughly 1.65x a
# midfielder's bonus - a single mean-to-mean curve cannot represent that and
# sits between the two, accurate for neither.
#
# Measured as the ratio of actual bonus/90 to what the curve above predicts,
# over 660 player-seasons at 900+ minutes. Fitting on 2024-25 and testing on
# 2025-26 reproduces every multiplier within 0.11, so this is a stable property
# of the scoring system rather than a one-season artefact:
#
#   fitted 24-25 -> observed 25-26:  GK 1.02 -> 0.91, DEF 0.94 -> 0.83,
#                                    MID 0.68 -> 0.70, FWD 1.38 -> 1.24
#
# An NTUA thesis on ML for FPL (Valouxis, 2023) reaches the same conclusion
# from the other direction, including `position` as an explicit feature in its
# bonus model because "each position class is rewarded bonus points a little
# bit differently".
BONUS_POSITION_MULTIPLIER = {1: 0.96, 2: 0.87, 3: 0.69, 4: 1.31}

# Minutes model constants.
# Sharpness of the within-team competition for a starting shirt. Goalkeeper is
# effectively winner-take-all, so probability concentrates hard on the first
# choice; outfield places are genuinely rotated, so the contest is softer.
GK_COMPETITION_ALPHA = 3.0
OUTFIELD_COMPETITION_ALPHA = 1.5

STARTER_MINUTES = 82.0      # mean minutes for a player who starts
SUB_MINUTES = 18.0          # mean minutes for a player who comes off the bench
# Minutes a starter plays, and his chance of reaching the 60-minute appearance
# bonus, BY POSITION. Both were single pooled numbers - 82.0 and 0.92 - which
# are accurate as averages and wrong for everyone: the error is the pooling,
# not the measurement.
#
# Measured over 2024-25 and 2025-26 separately; the two seasons agree to a
# tenth of a minute, so this is a property of how the positions are used
# rather than one season's noise.
#
#   GK 89.6' / 0.993    DEF 85.1' / 0.944    MID 79.9' / 0.908    FWD 78.8' / 0.919
#
# It matters more than one constant should, because starter minutes feed the
# appearance points, the clean-sheet gate through p60, the defensive
# contribution tail and the saves term all at once. The DEF award rate for
# defensive contribution is 26.4% per start; the Poisson tail at 82 minutes
# gives 20.2% and at the true 85.1 gives 25.0% - so most of what looked like a
# mis-shaped tail was this constant.
STARTER_MINUTES_BY_POSITION = {1: 89.6, 2: 85.1, 3: 79.9, 4: 78.8}
P60_BY_POSITION = {1: 0.993, 2: 0.944, 3: 0.908, 4: 0.919}

P60_GIVEN_START = 0.92      # pooled fallback, when position is unknown

# P(appears | did not start), as a function of how likely he was to start.
# Measured over 57,362 player-gameweeks across 2024-25 and 2025-26; the
# marginal rate across all of them is 0.156, not the 0.35 this replaced.
SUB_RATE_BY_START = (
    [0.00, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.00],
    [0.03, 0.034, 0.231, 0.330, 0.361, 0.436, 0.398, 0.35],
)

# ...and that curve was fitted pooled across positions, so it fits the average
# and therefore nobody. Measured P(appear | did not start): DEF 0.116, MID
# 0.206, FWD 0.234, against a pooled outfield rate of 0.176. A forward is a
# third more likely to come off the bench than the pool implies and a defender
# a third less, which is most of the model's underprediction of forwards.
SUB_RATE_POSITION_MULTIPLIER = {1: 1.0, 2: 0.659, 3: 1.169, 4: 1.329}

# FPL assists are not Opta expected assists, and the gap is large, positional
# and stable. FPL credits an assist for rebounds, deflections and won
# penalties, so its definition is simply broader than the one xA models.
# Measured actual-assists / xA:
#
#   season     DEF    MID    FWD    all
#   2024-25   1.107  1.371  2.077  1.374
#   2025-26   1.292  1.325  2.105  1.379
#
# The league-wide figure reproduces to three decimals across two seasons, and
# forwards earn over twice the assists their xA implies. Pricing assists at
# 3 * xA therefore understates every attacker, forwards worst.
ASSIST_PER_XA = {1: 1.20, 2: 1.20, 3: 1.35, 4: 2.09}
# Keepers essentially never appear off the bench: measured 0.0036 over 4,776
# non-start goalkeeper gameweeks. A substitute keeper means an injury or a red
# card to the first choice.
GK_SUB_RATE = 0.0036

# Gameweeks of evidence before a season-to-date start rate outweighs the prior,
# in the n/(n+k) sense. The old code capped at eight and then trusted form
# completely, which reads an eight-game sample as certainty.
START_RATE_EVIDENCE_GAMEWEEKS = 10.0

# Weights on the last few gameweeks' minutes, most recent first. Last week
# alone predicts minutes better than the whole model, so it leads.
RECENT_MINUTES_WEIGHTS = [0.5, 0.25, 0.15, 0.06, 0.04]

# How far recent minutes may displace the season rate once enough gameweeks
# have been observed. Short of 1.0 deliberately: one rotated week is not a
# demotion, and a player rested for a cup tie should not be written off.
RECENT_MINUTES_MAX_WEIGHT = 0.7

# Minutes at which an appearance counts as a start rather than a cameo.
STARTER_MINUTES_THRESHOLD = 60.0

# Points a bonus award is worth when it lands. Bonus pays one, two or three,
# and treating it as a ~2-point award occurring at the rate its mean implies
# reproduces its realised spread; treating it as deterministic understates
# every attacking player's variance.
BONUS_TYPICAL_AWARD = 2.0

# What the summed term variances miss, measured rather than argued.
#
# Two things are folded in here and it is worth being clear that they are not
# separable from outside:
#
#   - the terms are summed as though independent, and they are not. A goal
#     drags bonus with it; a clean sheet forbids conceding.
#   - our own mean is wrong by some amount, and a residual cannot tell "this
#     player is genuinely volatile" from "we mispriced him".
#
# Calibrated against realised residuals over 22,774 player-gameweeks of
# 2025-26: predicted sd 1.417 against actual 1.954, a ratio of 1.379 in sd and
# so 1.9 in variance. That makes this total *predictive* uncertainty rather
# than the player's own variance - which is the quantity a decision wants
# anyway, since being unsure is a real cost whatever its source.
#
# It is not a constant across the range: the ratio runs from 1.12 in the
# quietest bucket to 1.81 in the middle, so this gets the overall level right
# and leaves shape on the table. See tests/test_variance.py.
VARIANCE_CORRELATION_INFLATION = 1.90

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


def _allocate_shirts(weights: pd.Series, shirts: float, cap: float = 0.98) -> pd.Series:
    """
    Distribute `shirts` across a group in proportion to `weights`, capped.

    Clipping after scaling silently destroys the mass above the cap instead of
    giving it to somebody else, and a shirt handed to nobody is a shirt nobody
    plays in. Measured across the pool that leaked 0.25 of every team's eleven
    and 5.4% of its minutes - 937 against the 990 a side actually plays - which
    scales every per-90 term in the model and surfaces as a flat negative bias
    in expected points.

    Surplus is therefore redistributed, and repeated, since redistribution can
    push a second player over the cap. Two constraints make this narrower than
    it first appears:

    - only players with a positive weight may receive any. A zero means
      unavailable - injured, suspended, sold - and handing a shirt to a player
      who cannot play resurrects him. That is not hypothetical: the first
      version of this did exactly that, and injured players started scoring.
    - the loop is bounded, because a squad thinner than the shirts available
      genuinely cannot fill them and there is nothing to converge to.
    """
    weights = weights.astype(float).clip(lower=0.0)
    eligible = weights > 0.0
    total = float(weights.sum())
    if total <= 1e-12 or not eligible.any():
        return pd.Series(0.0, index=weights.index)

    out = weights * (shirts / total)
    for _ in range(8):
        out = out.clip(upper=cap)
        deficit = shirts - float(out.sum())
        if deficit <= 1e-9:
            break
        headroom = eligible & (out < cap)
        room = cap - out[headroom]
        if not headroom.any() or float(room.sum()) <= 1e-12:
            break
        out.loc[headroom] += room * min(1.0, deficit / float(room.sum()))
    return out.where(eligible, 0.0)


def _redistribute_unavailable(
    pecking: pd.Series, avail: pd.Series, shirts: float, cap: float = 0.98
) -> pd.Series:
    """
    Scale a pecking order by availability, then hand the freed shirts on.

    `pecking` is the allocation a fully fit squad would produce and already
    sums to `shirts`. Multiplying by availability takes mass out of the team
    total, and a shirt taken off an injured player is still a shirt somebody
    plays in - the same leak `_allocate_shirts` exists to stop, arriving by a
    different route.

    The rule that matters: mass released by a player goes to the OTHERS, in
    proportion to their own availability-weighted standing. Both of the obvious
    shortcuts are wrong, and both were tried here first:

    - sharing it by remaining headroom favours the deputy, because he is the
      one furthest below his ceiling. A first choice whose understudy carried a
      75% flag then finished *worse off* than if the understudy were fit -
      United's keeper fell 0.902 to 0.807 for no reason at all.
    - sharing it by standing hands it straight back to the player who released
      it, since he has the highest standing. A flagged first-choice keeper
      recovered his own mass and the flag did nothing, which is the bug this
      whole function exists to fix.

    Excluding the donor is what makes a 75%-fit first choice land near 0.75 of
    his fit probability with the remainder going to his deputy, which is what
    those two numbers are supposed to mean.
    """
    kept = (pecking * avail).clip(lower=0.0)
    released = (pecking * (1.0 - avail)).clip(lower=0.0)
    ceiling = pd.Series(np.minimum(cap, avail.to_numpy()), index=avail.index)

    total = float(kept.sum())
    if released.sum() > 1e-12 and total > 1e-12:
        # gain_j = w_j * sum over donors i != j of released_i / (total - w_i)
        denom = (total - kept).clip(lower=1e-12)
        per_donor = released / denom
        gain = kept * (float(per_donor.sum()) - per_donor)
        out = kept + gain
    else:
        out = kept

    # Ceilings can be breached by the hand-on, and filling the last of the
    # shirts can breach a second player's, so settle it iteratively. Bounded:
    # a squad whose ceilings sum below `shirts` genuinely cannot fill them.
    for _ in range(8):
        out = pd.Series(np.minimum(out.to_numpy(), ceiling.to_numpy()), index=out.index)
        deficit = shirts - float(out.sum())
        if deficit <= 1e-9:
            break
        room = (ceiling - out).clip(lower=0.0)
        total_room = float(room.sum())
        if total_room <= 1e-12:
            break
        out = out + room * min(1.0, deficit / total_room)
    return out.clip(lower=0.0)


class XPModel:
    """Expected points for every player over a horizon of gameweeks."""

    def __init__(
        self,
        bootstrap: dict,
        fixtures: list[dict],
        priors: PriorSet,
        config: ModelConfig | None = None,
        recent_minutes: dict[int, list[float]] | None = None,
    ):
        self.bootstrap = bootstrap
        self.fixtures = pd.DataFrame(fixtures)
        self.priors = priors
        self.config = config or ModelConfig()

        self.scoring = bootstrap["game_config"]["scoring"]
        # When each player can next feature, read from FPL's own news text.
        self.news = news.parse_all(bootstrap.get("elements") or [])
        self.recent_minutes: dict[int, list[float]] = dict(recent_minutes or {})
        self._news_added: dict[int, date] = {}
        for e in bootstrap.get("elements") or []:
            raw = e.get("news_added")
            if raw:
                try:
                    self._news_added[int(e["id"])] = datetime.fromisoformat(
                        str(raw).replace("Z", "+00:00")).date()
                except ValueError:
                    pass
        self._minutes_cache: dict[int | None, pd.DataFrame] = {}
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
            "penalties_order",
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
        df["xg90"] = self._adjust_xg_for_penalty_duty(df, w_cur)

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
            # n/(n+k) rather than a hard cap at eight gameweeks. The old form
            # displaced the prior entirely from GW8 onward, treating an
            # eight-game sample as certainty; regressing next season's start
            # rate on this one implies k of roughly 10 within a season.
            w_sr = events_done / (events_done + START_RATE_EVIDENCE_GAMEWEEKS)
            df["start_rate"] = w_sr * cur_sr + (1.0 - w_sr) * blended_prior
        else:
            df["start_rate"] = blended_prior

        # Recent minutes, which beat every rate above.
        #
        # A season-to-date start rate is a *level*: a player who lost his place
        # three weeks ago still carries the average of the weeks he was
        # playing. How long he was on the pitch last week is a *state*, and
        # measured over 22,774 player-gameweeks it predicts minutes far better
        # on its own (R2 0.598) than this entire model does (0.472). Combining
        # lags of one, three and five gameweeks reaches 0.633.
        #
        # That matters more than any scoring term: rescaling expected points by
        # lag-predicted minutes lifts points R2 from 0.280 to 0.350 on held-out
        # data, while the whole goals/assists/clean-sheet/bonus apparatus is
        # worth 0.004 once minutes are known.
        df["start_rate"] = self._blend_recent_minutes(df, events_done)

        df["prior_confidence"] = prior_confidence

        return df

    def _blend_recent_minutes(self, df: pd.DataFrame, events_done: int) -> pd.Series:
        """
        Fold recent observed minutes into the start rate.

        `recent_minutes` maps player id -> minutes in each of the last few
        gameweeks, most recent first. Supplied by the caller because the source
        differs: the weekly run reads `event/{id}/live/`, the backtest reads
        the gameweek history it is already replaying.

        Returns the start rate unchanged when nothing is supplied, so the
        pre-season path and any caller that cannot provide it are unaffected.
        """
        start_rate = df["start_rate"]
        if not self.recent_minutes or events_done <= 0:
            return start_rate

        # Weighted toward the most recent, since last week outperforms any
        # longer window on its own.
        weights = np.array(RECENT_MINUTES_WEIGHTS, dtype=float)
        observed = pd.Series(np.nan, index=df.index, dtype=float)
        for pos, pid in enumerate(df["id"].astype(int)):
            mins = self.recent_minutes.get(int(pid))
            if not mins:
                continue
            m = np.array(mins[: len(weights)], dtype=float)
            w = weights[: len(m)]
            if w.sum() <= 0:
                continue
            # Expressed as a start-equivalent: a full match is a start, a
            # cameo is not, and the threshold sits where an autosub stops
            # being one.
            observed.iloc[pos] = float((m >= STARTER_MINUTES_THRESHOLD) @ w / w.sum())

        # The lag view earns weight as gameweeks accumulate, and never fully
        # displaces the season rate - a single rotated week is not a demotion.
        depth = min(len(weights), events_done)
        w_recent = RECENT_MINUTES_MAX_WEIGHT * depth / len(weights)
        return start_rate.where(observed.isna(),
                                w_recent * observed + (1.0 - w_recent) * start_rate)

    def _price_implied_start_rate(self, df: pd.DataFrame) -> pd.Series:
        """P(start) implied by price alone, per position."""
        out = pd.Series(0.3, index=df.index)
        for pos_id, (xs, ys) in PRICE_START_PRIOR.items():
            mask = df["position"] == pos_id
            if mask.any():
                out.loc[mask] = np.interp(df.loc[mask, "cost"], xs, ys, left=ys[0], right=ys[-1])
        return out

    # ---------- minutes ----------

    def _event_date(self, event: int) -> date | None:
        """The deadline date of a gameweek - when availability is judged."""
        for e in self.bootstrap.get("events", []):
            if int(e["id"]) == int(event):
                raw = e.get("deadline_time")
                if not raw:
                    return None
                return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        return None

    def _availability(self, event: int | None = None) -> pd.Series:
        """
        P(player is fit and selectable), for `event` specifically.

        chance_of_playing_next_round is authoritative when FPL publishes it;
        otherwise fall back to the status flag. A player flagged 'd' with no
        published percentage is a genuine doubt, not a certainty.

        Both of those describe the *next* gameweek only, and the model plans
        over five. Holding them flat across the horizon has two costs, and they
        pull in opposite directions: a player out now but back in a fortnight
        scores zero for gameweeks he will play, and a player carrying a knock is
        priced as permanently 75% fit. Passing an event applies the news to that
        gameweek's date and lets a short-term doubt decay.
        """
        df = self.players
        chance = df["chance_of_playing_next_round"] / 100.0
        fallback = df["status"].map(STATUS_AVAILABILITY).fillna(0.5)
        base = chance.where(chance.notna(), fallback).clip(0.0, 1.0)
        if event is None:
            return base

        when = self._event_date(event)
        if when is None:
            return base

        events = next_events(self.bootstrap, self.config.horizon)
        ahead = events.index(event) if event in events else 0

        out = base.copy()
        # A doubt fades; an absence does not. Decaying from a base of zero would
        # quietly return a long-term injury to near-full fitness by gameweek
        # four, so only genuine doubts - anyone with something left to recover -
        # are allowed to improve with time.
        # Keyed on the news, not only on the base value. FPL routinely publishes
        # a percentage alongside an open-ended injury ("Unknown return date",
        # chance 25), and a base above zero was enough to let decay_doubt lift
        # that player to 0.95 availability by gameweek five - restoring someone
        # with no return date to nearly fit, which is the exact failure this
        # guard was written to prevent.
        indefinite = df["id"].astype(int).map(
            lambda pid: bool(getattr(self.news.get(int(pid)), "indefinite", False))
        )
        doubtful = (base > 0.0) & (base < 1.0) & ~indefinite.to_numpy()
        if doubtful.any() and ahead > 0:
            out.loc[doubtful] = [news.decay_doubt(float(v), ahead) for v in base[doubtful]]

        # The news is better evidence than the flag for anyone with a date
        # attached, in both directions: still out for this gameweek, or back.
        if self.news:
            ids = df["id"].astype(int)
            for pos, pid in enumerate(ids):
                info = self.news.get(int(pid))
                if info is None:
                    continue
                out.iloc[pos] = news.availability_on(
                    info, when, base=float(out.iloc[pos]), status=str(df["status"].iloc[pos]),
                )
        return out.clip(0.0, 1.0)

    def _absence_days(self, pid: int, info) -> float | None:
        """
        How long the player has been out, from `news_added` to his return date.

        Used only to decide whether the absence was long enough to cost match
        fitness. Returns None when FPL published no timestamp, which leaves
        ramp_multiplier on its default behaviour of damping.
        """
        added = self._news_added.get(pid)
        if added is None or info.returns_on is None:
            return None
        return float((info.returns_on - added).days)

    def _ramp(self, event: int | None) -> pd.Series:
        """Start-probability damping for players easing back from an absence."""
        df = self.players
        if event is None or not self.news:
            return pd.Series(1.0, index=df.index)
        when = self._event_date(event)
        if when is None:
            return pd.Series(1.0, index=df.index)
        out = pd.Series(1.0, index=df.index)
        for pos, pid in enumerate(df["id"].astype(int)):
            info = self.news.get(int(pid))
            if info is None:
                continue
            # absence_days was never supplied, which made the short-absence
            # carve-out in news.ramp_multiplier unreachable: a one-match
            # suspension was damped to 0.55 exactly like a three-month injury.
            # `news_added` is when FPL posted the flag, which is the best
            # available proxy for when the absence began.
            out.iloc[pos] = news.ramp_multiplier(
                info, when, absence_days=self._absence_days(int(pid), info))
        return out

    def _normalise_starts_within_team(
        self, start_rate: pd.Series, avail: pd.Series
    ) -> pd.Series:
        """
        Force each team's start probabilities to sum to the eleven shirts that
        actually exist: one goalkeeper and ten outfielders.

        Without this, start probability is estimated per player in isolation and
        nothing stops a club's two £5.0 keepers from both being 68% likely to
        start.

        Allocation within a group is proportional to rate**alpha rather than to
        the rate itself. Straight proportional scaling punishes a settled first
        choice for merely having credible cover on the books - a keeper rated
        0.87 behind two plausible deputies gets divided down to 0.65, which is
        not how team selection works. Raising to a power concentrates the
        probability on the likeliest starter. Goalkeeper is close to a binary
        contest so it gets the sharper exponent; outfield rotation is real, so
        it gets a gentler one.

        Availability is applied AFTER the pecking order rather than folded into
        the weights, and that ordering is the whole point of this rewrite.
        Allocation is purely relative: it divides a fixed number of shirts by
        relative standing, so scaling one player's weight down changes only his
        share. For a dominant first choice - and `rate**3.0` makes every settled
        keeper dominant - the share barely moves and the cut is destroyed.
        Measured on the committed snapshot, a 25% availability cut applied to
        each first-choice keeper in turn was retained at a mean of 0.243, and
        **five of eleven retained exactly 0.000**: Pickford, Leno, Verbruggen,
        Henderson and Sels were all completely immune to being flagged. That
        silently disabled every availability signal a keeper can receive -
        injury flags, chance_of_playing, news decay and card bans alike.

        So: allocate shirts on ability, scale each player by his own
        availability, then redistribute the freed mass to team-mates who can
        actually play. A 75%-fit first choice now lands near 0.75 of his fit
        probability and his deputy picks up the rest, which is what those two
        numbers are supposed to mean.

        The behaviour the old ordering got right is preserved: a player at zero
        availability is dropped from the pecking order outright, so his shirt
        goes to the understudy rather than being shared back to him.
        """
        df = self.players
        out = pd.Series(0.0, index=start_rate.index, dtype=float)
        rate = start_rate.clip(lower=0.0).fillna(0.0)
        av = avail.clip(0.0, 1.0).fillna(0.0)

        for team_id, idx in df.groupby("team").groups.items():
            for shirts, alpha, mask in (
                (1.0, GK_COMPETITION_ALPHA, df.loc[idx, "position"] == 1),
                (10.0, OUTFIELD_COMPETITION_ALPHA, df.loc[idx, "position"] != 1),
            ):
                sel = pd.Index(idx)[mask.to_numpy()]
                if len(sel) == 0:
                    continue
                r, a = rate.loc[sel], av.loc[sel]
                # Anyone certain to miss is out of the contest entirely, so his
                # shirt is handed on instead of shared back to him.
                weights = (r**alpha).where(a > 0.0, 0.0)
                if float(weights.sum()) <= 1e-12:
                    # Whole group unavailable (or no data): spread evenly rather
                    # than divide by zero. Rare, and self-corrects once FPL
                    # publishes availability.
                    out.loc[sel] = min(shirts / len(sel), 1.0)
                    continue
                pecking = _allocate_shirts(weights, shirts)
                out.loc[sel] = _redistribute_unavailable(pecking, a, shirts)

        return out.clip(0.0, 1.0)

    def minutes_model(self, event: int | None = None) -> pd.DataFrame:
        """
        P(appear), P(60+ minutes) and expected minutes for each player.

        Cached per gameweek: this is called once per fixture and the
        within-team start normalisation is the expensive part of the model.
        """
        cached = self._minutes_cache.get(event)
        if cached is not None:
            return cached

        df = self.players
        avail = self._availability(event)
        start_rate = (df["start_rate"].clip(0.0, 1.0).fillna(0.0) * self._ramp(event))

        p_start = self._normalise_starts_within_team(start_rate, avail)

        # Substitute appearances, conditional on the player's actual standing in
        # the side rather than a flat share.
        #
        # This was `0.35` for everyone. Measured over 57,000 player-gameweeks the
        # marginal rate is 0.156, and it is strongly monotone in how often the
        # player starts: 0.034 for someone who almost never starts, rising to
        # ~0.43 for a regular who happens to be benched. The flat figure was
        # therefore ~10x too high for exactly the population it was applied to
        # most - two thirds of all non-start rows are players in the bottom
        # band - which is the mechanism behind the model's known weakness at
        # ranking players who never appear.
        #
        # Goalkeepers are the extreme: measured P(appear | did not start) is
        # 0.0036. At 0.35 the model credited a keeper it had just assigned a
        # 0.000 start probability with six expected minutes and 0.45 xP, and
        # that fabricated value was the entire basis for ranking the bench
        # keeper slot.
        #
        # Keyed on post-normalisation p_start, not the raw rate: the normaliser
        # has already decided who is behind whom in the pecking order, and the
        # old code threw that verdict away by using the pre-normalisation value.
        sub_rate = pd.Series(
            np.interp(p_start, SUB_RATE_BY_START[0], SUB_RATE_BY_START[1]),
            index=df.index,
        ) * df["position"].map(SUB_RATE_POSITION_MULTIPLIER).fillna(1.0)
        sub_rate = sub_rate.where(df["position"] != 1, GK_SUB_RATE)
        p_sub = avail * (1.0 - p_start) * sub_rate
        p_appear = (p_start + p_sub).clip(0.0, 1.0)
        starter_minutes = df["position"].map(STARTER_MINUTES_BY_POSITION).fillna(STARTER_MINUTES)
        p60_given_start = df["position"].map(P60_BY_POSITION).fillna(P60_GIVEN_START)
        p60 = (p_start * p60_given_start).clip(0.0, 1.0)
        exp_minutes = p_start * starter_minutes + p_sub * SUB_MINUTES

        frame = pd.DataFrame({
            "availability": avail,
            "p_start": p_start,
            "p_appear": p_appear,
            "p60": p60,
            "exp_minutes": exp_minutes,
        }, index=df.index)
        self._minutes_cache[event] = frame
        return frame

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

        The rescaling fades out as the season supplies real evidence, since
        minutes played for the new club were earned in the style being corrected
        for.

        Note this is an approximation, not an exact decomposition. It scales the
        *blended* rate by (1 + (ratio-1)*prior_share); exact would be
        w*current + (1-w)*prior*ratio. The two agree at both endpoints and
        differ by at most w*(1-w)*(ratio-1)*(current-prior) in between - about
        5% at mid-season, always in the direction of over-applying. Left as is
        because the blend's components are not retained separately by this
        point, and the error is well inside the uncertainty on `ratio` itself.
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

    def _adjust_xg_for_penalty_duty(self, df: pd.DataFrame, w_cur: pd.Series) -> pd.Series:
        """
        Correct expected goals when a player has gained or lost the penalties.

        A taker's own xG history already contains his penalties - Opta's model
        counts them at about 0.79 each - so a player who keeps the job needs no
        adjustment. The correction is for the turnover, which is heavy: of 16
        takers in 2024-25 only 7 still held it a season later. A prior earned
        with the duty overstates a player who has since lost it, and one earned
        without it understates a player who has just been given it.

        Only the prior share is corrected, on the same reasoning as the club-move
        adjustment: minutes played this season already reflect the current duty.
        """
        xg90 = pd.to_numeric(df["xg90"], errors="coerce")
        duty = self.priors.player_penalty_duty
        if not duty or "penalties_order" not in df.columns:
            return xg90

        order = pd.to_numeric(df["penalties_order"], errors="coerce")
        now_taker = order == 1
        mapped = df["code"].map(duty)
        known = mapped.notna()
        # Built as a plain bool array rather than fillna(...).astype(bool) on an
        # object column, which pandas deprecates and which would start emitting
        # a downcasting warning - the kind of drift that already cost us once.
        was_taker = pd.Series(mapped.to_numpy() == True, index=df.index)  # noqa: E712

        prior_share = 1.0 - pd.to_numeric(w_cur, errors="coerce").fillna(0.0)
        delta = pd.Series(0.0, index=df.index)
        delta[known & now_taker & ~was_taker] = PENALTY_XG90
        delta[known & ~now_taker & was_taker] = -PENALTY_XG90

        moved = int((delta != 0).sum())
        if moved:
            logger.info("expected goals adjusted for %d players whose penalty duty changed", moved)
        return (xg90 + delta * prior_share).clip(lower=0.0)

    def _expected_bonus(self, bps90: pd.Series, minutes_share: pd.Series) -> pd.Series:
        """
        Interpolate the empirical BPS-to-bonus curve, scaled by minutes and
        corrected per position.

        See BONUS_POSITION_MULTIPLIER: the curve maps a mean to a mean, while
        bonus is a rank award whose expectation depends on BPS dispersion, and
        that dispersion differs sharply by position.
        """
        b = np.interp(bps90.fillna(0.0), BONUS_CURVE_BPS, BONUS_CURVE_PTS,
                      left=BONUS_CURVE_PTS[0], right=BONUS_CURVE_PTS[-1])
        out = pd.Series(b, index=bps90.index) * minutes_share
        return out * self.players["position"].map(BONUS_POSITION_MULTIPLIER).fillna(1.0)

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
            starter_mins = STARTER_MINUTES_BY_POSITION.get(pos_id, STARTER_MINUTES)
            p_started = self._tail_probability(r * (starter_mins / 90.0), thr)
            p_benched = self._tail_probability(r * (SUB_MINUTES / 90.0), thr)
            out.loc[mask] = (
                p_start[mask].to_numpy(dtype=float) * p_started
                + p_sub[mask].clip(lower=0.0).to_numpy(dtype=float) * p_benched
            )

        return out.clip(0.0, 1.0)

    # ---------- the estimate ----------

    @staticmethod
    def _second_moment_gate(second_moment, mean, gate) -> pd.Series:
        """
        Variance of a quantity that only occurs if the player is on the pitch.

        The term is a mixture: zero with probability (1 - gate), and the
        modelled distribution otherwise. Var = gate * E[X^2] - (gate * E[X])^2,
        and `mean` already carries the gate.
        """
        sm = pd.Series(second_moment, index=mean.index)
        return (gate * sm - mean ** 2).clip(lower=0.0)

    def expected_points_for_fixture(self, team_id: int, opponent_id: int, is_home: bool,
                                    event: int | None = None,
                                    with_variance: bool = False):
        """
        xP for every player, were their team to play this single fixture.

        With `with_variance`, returns (mean, variance) instead of the mean
        alone. The variance is assembled from the same distributions the mean
        already uses - Poisson goals and assists, Bernoulli clean sheets and
        defensive contribution, the explicit goals-conceded and saves mass
        functions - so it is derived rather than fitted.

        Every decision this model feeds was made on a point estimate, which
        cannot distinguish a forward who hauls or blanks from a defender who
        returns five every week. For captaincy, the highest-variance call of
        the week, that difference is the entire question.
        """
        df = self.players
        mm = self.minutes_model(event)
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
        # xA counts chances created as a goal-probability; FPL counts assists,
        # and its definition is broader (rebounds, deflections, won penalties).
        # Measured ratio is 1.20 for defenders, 1.35 for midfielders and 2.09
        # for forwards, reproducing across both seasons.
        assist_rate = df["xa90"].fillna(0.0) * pos.map(ASSIST_PER_XA).fillna(1.0)
        assists = assist_rate * minutes_share * attack_mult * self._pts("assists", pos)

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
        # E[floor(S/3)], not E[S]/3. The goals-conceded term eight lines below
        # already does this correctly; the saves term did the naive thing and
        # overpriced every keeper by +0.34 points per start (~13 a season,
        # measured over 2,700 goalkeeper appearances of 60+ minutes). Because
        # the bias is near-constant across keepers it never reordered them,
        # which is why it survived - but it inflated every reported xi_xp and
        # the bench keeper's contribution to the bench-boost valuation.
        save_ks = np.arange(0, 16)
        save_pmf = poisson.pmf(save_ks[None, :],
                               np.maximum(saves_rate.to_numpy()[:, None], 1e-9))
        saves = pd.Series((save_pmf * (save_ks // 3)).sum(axis=1), index=saves_rate.index)
        saves = saves * float(self.scoring.get("saves", 1))

        # Penalty saves were missing entirely: measured 0.0163 per goalkeeper
        # appearance at 5 points each, ~+3 a season. Small, but it offsets part
        # of the correction above, so the two belong together.
        pen_save_pts = float(self.scoring.get("penalties_saved", 5))
        saves = saves + (mm["p_appear"] * PENALTY_SAVE_RATE * pen_save_pts).where(pos == 1, 0.0)

        dc = self._p_dc_award(df["dc90"], pos, mm["p_start"], mm["p_appear"] - mm["p60"]) * self._pts(
            "defensive_contribution", pos
        )

        cards = df["yellow90"].fillna(0.0) * minutes_share * float(self.scoring.get("yellow_cards", -1))

        bonus = self._expected_bonus(df["bps90"], minutes_share) * float(self.scoring.get("bonus", 1))

        total = appearance + goals + assists + clean_sheet + goals_conceded + saves + dc + cards + bonus
        if not with_variance:
            return total

        # Variance, term by term, from the distribution each term already
        # assumes. Summed as though the terms were independent, which they are
        # not - a goal drags bonus with it, and a clean sheet forbids conceding
        # - so the sum is corrected by VARIANCE_CORRELATION_INFLATION, measured
        # against realised variance rather than argued for.
        g_pts = self._pts("goals_scored", pos)
        a_pts = self._pts("assists", pos)
        cs_pts = self._pts("clean_sheets", pos)
        dc_pts = self._pts("defensive_contribution", pos)

        # Appearance takes 0, 1 or 2; compute its second moment directly.
        p_two = mm["p60"]
        p_one = (mm["p_appear"] - mm["p60"]).clip(lower=0)
        appear_sq = (long_play ** 2) * p_two + (short_play ** 2) * p_one
        var_appearance = appear_sq - appearance ** 2

        # Poisson counts: variance equals the mean, scaled by points squared.
        var_goals = g_pts * goals
        var_assists = a_pts * assists

        # Bernoulli awards.
        q_cs = (mm["p60"] * p_cs).clip(0.0, 1.0)
        var_cs = (cs_pts ** 2) * q_cs * (1.0 - q_cs)
        p_dc_award = (dc / dc_pts.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)
        var_dc = (dc_pts ** 2) * p_dc_award * (1.0 - p_dc_award)

        # Terms whose second moment needs the mass function that produced them.
        var_conceded = self._second_moment_gate(
            (pmf * ((ks // 2) * gc_pts.abs().to_numpy()[:, None]) ** 2).sum(axis=1),
            goals_conceded, mm["p60"])
        var_saves = self._second_moment_gate(
            (save_pmf * ((save_ks // 3) * float(self.scoring.get("saves", 1))) ** 2).sum(axis=1),
            saves, mm["p_appear"])

        # Bonus is one to three points, rarely. Treated as an award of about
        # two points at the rate implied by its mean, which reproduces its
        # realised spread far better than assuming it is deterministic.
        var_bonus = (BONUS_TYPICAL_AWARD * bonus - bonus ** 2).clip(lower=0.0)
        var_cards = (float(self.scoring.get("yellow_cards", -1)) ** 2) * cards.abs()

        variance = (var_appearance + var_goals + var_assists + var_cs + var_dc
                    + var_conceded + var_saves + var_bonus + var_cards)
        return total, (variance * VARIANCE_CORRELATION_INFLATION).clip(lower=0.0)
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
            gw_var = pd.Series(0.0, index=df.index)
            fixture_count = pd.Series(0, index=df.index)

            for _, f in fx.iterrows():
                mask = df["team"] == f["team_id"]
                if not mask.any():
                    continue
                xp, var = self.expected_points_for_fixture(
                    int(f["team_id"]), int(f["opponent_id"]), bool(f["is_home"]),
                    event=ev, with_variance=True)
                gw_points[mask] += xp[mask]
                # Fixtures within a double gameweek are near enough independent
                # - different opponents, days apart - so variances add.
                gw_var[mask] += var[mask]
                fixture_count[mask] += 1

            out[f"xp_gw{ev}"] = gw_points
            out[f"var_gw{ev}"] = gw_var
            out[f"fixtures_gw{ev}"] = fixture_count
            # Only the first `horizon` gameweeks form the squad-selection
            # objective, however many are scored. Callers pass a longer event
            # list when something else needs the extra visibility - the chip
            # planner compares opportunities ten gameweeks out - and widening
            # xp_horizon to match would silently change what the optimiser is
            # maximising.
            if i < self.config.horizon:
                horizon_total += gw_points * (self.config.horizon_decay**i)

        out["xp_next"] = out[f"xp_gw{events[0]}"] if events else 0.0
        out["var_next"] = out[f"var_gw{events[0]}"] if events else 0.0
        out["sd_next"] = np.sqrt(out["var_next"].clip(lower=0.0))
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
