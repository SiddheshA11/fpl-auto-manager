"""
FPL Auto Manager - Player Scoring Engine
Computes a composite score for every player to rank them for selection and transfers.
Now integrates live news research and deep web analytics for better decisions.
"""
import logging
import numpy as np
import pandas as pd
from fpl_client import FPLClient
from config import STRATEGY

logger = logging.getLogger("fpl_auto")


class PlayerScorer:
    """Evaluate and rank every FPL player using a weighted multi-factor model."""

    def __init__(self, client: FPLClient):
        self.client = client
        self._news_profiles = None       # set by inject_news_research()
        self._web_research_data = None   # set by inject_web_research()

    def inject_news_research(self, profiles: dict):
        """
        Inject player availability profiles from the NewsResearcher.
        Call this before score_players() to use live injury/news data.
        """
        self._news_profiles = profiles
        logger.info(f"News research injected: {len(profiles)} player profiles.")

    def inject_web_research(self, research_data: dict):
        """
        Inject deep web research data (xG, form trends, team news).
        Call this before score_players() to use web analytics.
        """
        self._web_research_data = research_data
        logger.info(f"Web research injected: {len(research_data)} data points.")

    # ──────────────── Fixture Difficulty ────────────────

    def _fixture_difficulty_scores(self, players: pd.DataFrame) -> pd.Series:
        """
        For each player, compute an average fixture difficulty rating
        over the next N gameweeks (lower difficulty = higher score).
        """
        fixtures = self.client.get_fixtures()
        next_event = self.client.get_next_event()
        if next_event is None or fixtures.empty:
            return pd.Series(0.5, index=players.index)

        lookahead = STRATEGY["fixture_lookahead"]
        gw_start = next_event["id"]
        gw_end = gw_start + lookahead

        upcoming = fixtures[
            (fixtures["event"] >= gw_start) & (fixtures["event"] < gw_end)
        ].copy()

        if upcoming.empty:
            return pd.Series(0.5, index=players.index)

        # Build a mapping: team_id -> average opponent difficulty
        team_difficulties = {}
        for team_id in players["team"].unique():
            home_games = upcoming[upcoming["team_h"] == team_id]["team_h_difficulty"]
            away_games = upcoming[upcoming["team_a"] == team_id]["team_a_difficulty"]
            all_diffs = pd.concat([home_games, away_games])
            if len(all_diffs) > 0:
                avg_diff = all_diffs.mean()
                team_difficulties[team_id] = 1.0 - (avg_diff - 1) / 4.0
            else:
                team_difficulties[team_id] = 0.5

        return players["team"].map(team_difficulties).fillna(0.5)

    def _count_double_gameweeks(self, players: pd.DataFrame, event_id: int) -> pd.Series:
        """Count how many fixtures each player's team has in a given gameweek."""
        fixtures = self.client.get_fixtures()
        gw_fixtures = fixtures[fixtures["event"] == event_id]

        team_fixture_count = {}
        for team_id in players["team"].unique():
            count = len(gw_fixtures[
                (gw_fixtures["team_h"] == team_id) | (gw_fixtures["team_a"] == team_id)
            ])
            team_fixture_count[team_id] = count

        return players["team"].map(team_fixture_count).fillna(1).astype(int)

    # ──────────────── xGI Estimation ────────────────

    def _estimate_xgi(self, players: pd.DataFrame) -> pd.Series:
        """
        Estimate expected goal involvement per 90 minutes.
        Uses goals, assists, and ICT creativity/threat as proxies.
        Enhanced with web-sourced xG data when available.
        """
        minutes = players["minutes"].replace(0, 1)
        per90 = 90.0 / minutes

        goals_per90 = players["goals_scored"] * per90
        assists_per90 = players["assists"] * per90
        xgi = goals_per90 + assists_per90 * 0.8

        # Enhance with web research xG data if available
        if self._web_research_data and "xg_data" in self._web_research_data:
            xg_map = self._web_research_data["xg_data"]
            for idx, row in players.iterrows():
                pid = row["id"]
                web_name = row.get("web_name", "")
                if pid in xg_map:
                    web_xgi = xg_map[pid].get("xgi_per90", None)
                    if web_xgi is not None:
                        # Blend: 40% FPL-derived, 60% web xG (more reliable)
                        xgi.at[idx] = 0.4 * xgi.at[idx] + 0.6 * web_xgi

        max_xgi = xgi.quantile(0.99) if xgi.max() > 0 else 1
        return (xgi / max(max_xgi, 0.01)).clip(0, 1)

    # ──────────────── Research-Enhanced Availability ────────────────

    def _compute_availability(self, players: pd.DataFrame) -> pd.Series:
        """
        Compute player availability using all available intelligence:
        1. FPL API status flags (baseline)
        2. News research profiles (injuries, press conferences)
        3. Web research (team news, predicted lineups)
        """
        # Baseline from FPL status
        availability_map = {"a": 1.0, "d": 0.6, "i": 0.0, "u": 0.0, "s": 0.0, "n": 0.0}
        baseline = players["status"].map(availability_map).fillna(0.5)

        if self._news_profiles is None:
            logger.info("No news research available; using FPL status only.")
            return baseline

        # Override with researched availability (much more nuanced)
        result = baseline.copy()
        upgraded = 0
        downgraded = 0

        for idx, row in players.iterrows():
            pid = row["id"]
            profile = self._news_profiles.get(pid)
            if profile is None:
                continue

            researched = profile["final_availability"]
            fpl_baseline = baseline.at[idx]

            # Use the researched value — it already blends FPL + web sources
            result.at[idx] = researched

            if researched > fpl_baseline + 0.1:
                upgraded += 1
            elif researched < fpl_baseline - 0.1:
                downgraded += 1

        logger.info(
            f"Availability: {upgraded} players upgraded, {downgraded} downgraded by research "
            f"(vs FPL status alone)."
        )
        return result

    # ──────────────── Web Research Scoring Boosts ────────────────

    def _apply_web_research_boosts(self, players: pd.DataFrame) -> pd.Series:
        """
        Apply scoring adjustments from deep web research:
        - xG overperformers/underperformers
        - Recent form trends (last 3 GW momentum)
        - Team tactical changes
        - Set piece taker changes
        """
        boosts = pd.Series(0.0, index=players.index)

        if not self._web_research_data:
            return boosts

        # Form momentum (accelerating or decelerating)
        form_momentum = self._web_research_data.get("form_momentum", {})
        for idx, row in players.iterrows():
            pid = row["id"]
            if pid in form_momentum:
                # Momentum is -1 (crashing) to +1 (soaring)
                boosts.at[idx] += form_momentum[pid] * 0.05

        # Set piece takers get a boost
        set_piece_takers = self._web_research_data.get("set_piece_takers", set())
        for idx, row in players.iterrows():
            if row["id"] in set_piece_takers:
                boosts.at[idx] += 0.03

        # xG overperformance penalty (regression likely) / underperformance boost
        xg_regression = self._web_research_data.get("xg_regression", {})
        for idx, row in players.iterrows():
            pid = row["id"]
            if pid in xg_regression:
                # Positive = underperforming xG (due a rise), Negative = overperforming (due a drop)
                boosts.at[idx] += xg_regression[pid] * 0.04

        # Home/away splits
        ha_boosts = self._web_research_data.get("home_away_boosts", {})
        for idx, row in players.iterrows():
            pid = row["id"]
            if pid in ha_boosts:
                boosts.at[idx] += ha_boosts[pid] * 0.02

        total_boosted = (boosts.abs() > 0.01).sum()
        if total_boosted:
            logger.info(f"Web research boosts applied to {total_boosted} players.")

        return boosts

    # ──────────────── Composite Score ────────────────

    def score_players(self) -> pd.DataFrame:
        """
        Compute a composite score for every player and return the full DataFrame
        sorted by score descending. Integrates news research and web analytics
        when available.
        """
        players = self.client.get_players_df()
        weights = STRATEGY["weights"]

        # 1. Form (already 0-10 scale from FPL, normalize)
        max_form = players["form_numeric"].quantile(0.99)
        players["score_form"] = (players["form_numeric"] / max(max_form, 0.01)).clip(0, 1)

        # 2. Fixture difficulty
        players["score_fixtures"] = self._fixture_difficulty_scores(players)

        # 3. xGI per 90 (enhanced with web xG data)
        players["score_xgi"] = self._estimate_xgi(players)

        # 4. Points per game
        max_ppg = players["ppg_numeric"].quantile(0.99)
        players["score_ppg"] = (players["ppg_numeric"] / max(max_ppg, 0.01)).clip(0, 1)

        # 5. ICT Index
        max_ict = players["ict_numeric"].quantile(0.99)
        players["score_ict"] = (players["ict_numeric"] / max(max_ict, 0.01)).clip(0, 1)

        # 6. Minutes reliability
        players["score_minutes"] = players["minutes_frac"].clip(0, 1)

        # Weighted composite
        players["composite_score"] = (
            weights["form"] * players["score_form"]
            + weights["fixture_difficulty"] * players["score_fixtures"]
            + weights["xgi_per90"] * players["score_xgi"]
            + weights["points_per_game"] * players["score_ppg"]
            + weights["ict_index"] * players["score_ict"]
            + weights["minutes_reliability"] * players["score_minutes"]
        )

        # Apply web research boosts (form momentum, set pieces, xG regression, etc.)
        players["web_boost"] = self._apply_web_research_boosts(players)
        players["composite_score"] = (players["composite_score"] + players["web_boost"]).clip(lower=0)

        # Value score = composite / cost (bang for buck)
        players["value_score"] = players["composite_score"] / players["now_cost_m"].clip(lower=3.5)

        # Research-enhanced availability (uses news + web sources)
        players["availability"] = self._compute_availability(players)
        players["adjusted_score"] = players["composite_score"] * players["availability"]

        players = players.sort_values("adjusted_score", ascending=False).reset_index(drop=True)

        logger.info(f"Scored {len(players)} players. Top 5: "
                     f"{players[['web_name', 'adjusted_score', 'now_cost_m']].head().to_string(index=False)}")
        return players

    def get_top_by_position(self, scored_df: pd.DataFrame, position: int, n: int = 20) -> pd.DataFrame:
        """Return top N players for a given position (1=GK, 2=DEF, 3=MID, 4=FWD)."""
        return scored_df[
            (scored_df["element_type"] == position) & (scored_df["availability"] > 0)
        ].head(n)

    def get_captain_candidates(self, scored_df: pd.DataFrame, squad_ids: list[int], n: int = 5) -> pd.DataFrame:
        """Return top N captain candidates from the current squad."""
        squad = scored_df[scored_df["id"].isin(squad_ids)].copy()
        squad = squad.sort_values("adjusted_score", ascending=False)
        return squad.head(n)
