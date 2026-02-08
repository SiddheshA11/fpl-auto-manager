"""
FPL Auto Manager - Chip Strategy Engine
Decides when to play Wildcard, Bench Boost, Triple Captain, and Free Hit
using the aggressive strategy profile.
"""
import logging
import pandas as pd
from fpl_client import FPLClient
from player_scorer import PlayerScorer
from config import STRATEGY

logger = logging.getLogger("fpl_auto")


class ChipStrategyEngine:
    """
    Analyze the current game state and decide whether to activate a chip.

    Chip types:
        - wildcard: unlimited free transfers for one gameweek
        - freehit: temporary squad for one gameweek, reverts next week
        - bboost: bench players score points for one gameweek
        - 3xc: captain's points are tripled for one gameweek
    """

    def __init__(self, client: FPLClient, scorer: PlayerScorer):
        self.client = client
        self.scorer = scorer

    def get_chip_recommendation(self) -> dict:
        """
        Evaluate all conditions and return a chip recommendation.

        Returns:
            {
                "chip": str or None,   # 'wildcard', 'freehit', 'bboost', '3xc', or None
                "reason": str,
                "confidence": float,   # 0.0 - 1.0
            }
        """
        chips_status = self.client.get_chips_status()
        available_chips = [name for name, info in chips_status.items() if not info["played"]]

        if not available_chips:
            return {"chip": None, "reason": "All chips already used.", "confidence": 1.0}

        next_event = self.client.get_next_event()
        if not next_event:
            return {"chip": None, "reason": "No upcoming gameweek.", "confidence": 1.0}

        event_id = next_event["id"]
        scored_df = self.scorer.score_players()

        # Evaluate each chip independently
        recommendations = []

        if "bboost" in available_chips:
            bb = self._evaluate_bench_boost(event_id, scored_df)
            if bb:
                recommendations.append(bb)

        if "3xc" in available_chips:
            tc = self._evaluate_triple_captain(event_id, scored_df)
            if tc:
                recommendations.append(tc)

        if "freehit" in available_chips:
            fh = self._evaluate_free_hit(event_id, scored_df)
            if fh:
                recommendations.append(fh)

        if "wildcard" in available_chips:
            wc = self._evaluate_wildcard(scored_df)
            if wc:
                recommendations.append(wc)

        if not recommendations:
            return {"chip": None, "reason": "No chip activation warranted this week.", "confidence": 0.7}

        # Pick highest confidence recommendation
        best = max(recommendations, key=lambda r: r["confidence"])
        logger.info(f"Chip recommendation: {best['chip']} (confidence={best['confidence']:.2f}) - {best['reason']}")
        return best

    # ──────────────── Bench Boost ────────────────

    def _evaluate_bench_boost(self, event_id: int, scored_df: pd.DataFrame) -> dict | None:
        """
        Bench Boost is best during Double Gameweeks when bench players also have
        two fixtures.
        """
        fixtures = self.client.get_fixtures()
        gw_fixtures = fixtures[fixtures["event"] == event_id]

        # Count fixtures per team in this GW
        team_fixture_count = {}
        for _, row in gw_fixtures.iterrows():
            team_fixture_count[row["team_h"]] = team_fixture_count.get(row["team_h"], 0) + 1
            team_fixture_count[row["team_a"]] = team_fixture_count.get(row["team_a"], 0) + 1

        # Check how many of our squad have double fixtures
        team_data = self.client.get_my_team()
        if not team_data:
            return None

        squad_ids = [p["element"] for p in team_data["picks"]]
        squad_info = scored_df[scored_df["id"].isin(squad_ids)]

        dgw_count = 0
        for _, player in squad_info.iterrows():
            if team_fixture_count.get(int(player["team"]), 1) >= 2:
                dgw_count += 1

        threshold = STRATEGY["bench_boost_min_dgw_players"]
        if dgw_count >= threshold:
            confidence = min(1.0, dgw_count / 15.0)
            return {
                "chip": "bboost",
                "reason": f"Double GW detected: {dgw_count}/15 squad players have 2 fixtures.",
                "confidence": confidence,
            }

        # Also consider if bench is very strong
        bench_picks = [p for p in team_data["picks"] if p["position"] > 11]
        bench_ids = [p["element"] for p in bench_picks]
        bench_scores = scored_df[scored_df["id"].isin(bench_ids)]["adjusted_score"]

        if not bench_scores.empty and bench_scores.mean() > 0.5:
            return {
                "chip": "bboost",
                "reason": f"Strong bench (avg score {bench_scores.mean():.2f}) with favorable fixtures.",
                "confidence": 0.5,
            }

        return None

    # ──────────────── Triple Captain ────────────────

    def _evaluate_triple_captain(self, event_id: int, scored_df: pd.DataFrame) -> dict | None:
        """
        Triple Captain is best when the top captain pick has a double gameweek
        against weak opponents.
        """
        fixtures = self.client.get_fixtures()
        gw_fixtures = fixtures[fixtures["event"] == event_id]

        team_fixture_count = {}
        team_avg_difficulty = {}
        for _, row in gw_fixtures.iterrows():
            for team_key, diff_key in [("team_h", "team_h_difficulty"), ("team_a", "team_a_difficulty")]:
                tid = row[team_key]
                team_fixture_count[tid] = team_fixture_count.get(tid, 0) + 1
                if tid not in team_avg_difficulty:
                    team_avg_difficulty[tid] = []
                team_avg_difficulty[tid].append(row[diff_key])

        # Average difficulties
        for tid in team_avg_difficulty:
            team_avg_difficulty[tid] = sum(team_avg_difficulty[tid]) / len(team_avg_difficulty[tid])

        # Get top scorer in squad
        team_data = self.client.get_my_team()
        if not team_data:
            return None

        squad_ids = [p["element"] for p in team_data["picks"]]
        captain_candidates = scored_df[scored_df["id"].isin(squad_ids)].sort_values(
            "adjusted_score", ascending=False
        )

        if captain_candidates.empty:
            return None

        top_player = captain_candidates.iloc[0]
        player_team = int(top_player["team"])
        num_fixtures = team_fixture_count.get(player_team, 1)
        avg_diff = team_avg_difficulty.get(player_team, 3)

        expected_points = top_player["adjusted_score"] * 10 * num_fixtures

        threshold = STRATEGY["triple_captain_min_xp"]
        if num_fixtures >= 2 and expected_points >= threshold:
            confidence = min(1.0, expected_points / (threshold * 2))
            return {
                "chip": "3xc",
                "reason": (
                    f"TC candidate: {top_player['web_name']} with {num_fixtures} fixtures "
                    f"(avg difficulty {avg_diff:.1f}), expected ~{expected_points:.0f} pts."
                ),
                "confidence": confidence,
            }

        # Even in single GW, very high-scoring player warrants consideration
        if expected_points >= threshold * 1.5:
            return {
                "chip": "3xc",
                "reason": f"Strong TC pick: {top_player['web_name']} expected ~{expected_points:.0f} pts.",
                "confidence": 0.45,
            }

        return None

    # ──────────────── Free Hit ────────────────

    def _evaluate_free_hit(self, event_id: int, scored_df: pd.DataFrame) -> dict | None:
        """
        Free Hit is best during blank gameweeks (many teams not playing)
        or when the squad is badly positioned for a specific week.
        """
        fixtures = self.client.get_fixtures()
        gw_fixtures = fixtures[fixtures["event"] == event_id]

        teams_playing = set(gw_fixtures["team_h"].tolist() + gw_fixtures["team_a"].tolist())
        all_teams = set(scored_df["team"].unique())
        teams_not_playing = all_teams - teams_playing

        # If many teams blank, free hit is very valuable
        if len(teams_not_playing) >= 6:
            # Check how many of our players are affected
            team_data = self.client.get_my_team()
            if not team_data:
                return None

            squad_ids = [p["element"] for p in team_data["picks"]]
            squad_info = scored_df[scored_df["id"].isin(squad_ids)]
            blanking = squad_info[squad_info["team"].isin(teams_not_playing)]

            if len(blanking) >= 3:
                confidence = min(1.0, len(blanking) / 8.0)
                return {
                    "chip": "freehit",
                    "reason": (
                        f"Blank GW: {len(teams_not_playing)} teams not playing, "
                        f"{len(blanking)} of your players affected."
                    ),
                    "confidence": confidence,
                }

        # Check for recent rank drop
        history = self.client.get_my_history()
        if history and "current" in history:
            recent = history["current"][-3:]
            if len(recent) >= 2:
                rank_change = recent[-1].get("overall_rank", 0) - recent[0].get("overall_rank", 0)
                if rank_change > STRATEGY["free_hit_rank_drop_trigger"]:
                    return {
                        "chip": "freehit",
                        "reason": f"Rank dropped by {rank_change:,} over recent weeks. Emergency free hit.",
                        "confidence": 0.5,
                    }

        return None

    # ──────────────── Wildcard ────────────────

    def _evaluate_wildcard(self, scored_df: pd.DataFrame) -> dict | None:
        """
        Wildcard when the squad is fundamentally misaligned with form/fixtures
        over multiple weeks.
        """
        team_data = self.client.get_my_team()
        if not team_data:
            return None

        squad_ids = [p["element"] for p in team_data["picks"]]
        squad_scores = scored_df[scored_df["id"].isin(squad_ids)]["adjusted_score"]

        # Compare squad quality to what's available
        top_15_available = scored_df.head(30)["adjusted_score"].mean()
        squad_avg = squad_scores.mean() if not squad_scores.empty else 0

        quality_gap = top_15_available - squad_avg

        if quality_gap > 0.15:
            confidence = min(1.0, quality_gap / 0.3)
            return {
                "chip": "wildcard",
                "reason": (
                    f"Squad quality gap: avg score {squad_avg:.3f} vs top available {top_15_available:.3f}. "
                    f"Major overhaul needed."
                ),
                "confidence": confidence,
            }

        # Check declining form over multiple weeks
        history = self.client.get_my_history()
        if history and "current" in history:
            recent = history["current"][-(STRATEGY["wildcard_form_drop_weeks"]):]
            if len(recent) >= STRATEGY["wildcard_form_drop_weeks"]:
                ranks = [gw.get("overall_rank", 0) for gw in recent]
                # If rank has been consistently dropping
                if all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1)):
                    return {
                        "chip": "wildcard",
                        "reason": f"Rank declining for {len(ranks)} consecutive weeks: {ranks}",
                        "confidence": 0.55,
                    }

        return None
