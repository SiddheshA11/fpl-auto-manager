"""
FPL Auto Manager - Transfer Optimizer
Finds the best transfers using the aggressive strategy profile.
Uses a greedy knapsack-style approach to maximize expected points gain.
"""
import logging
import pandas as pd
import numpy as np
from itertools import combinations
from fpl_client import FPLClient
from player_scorer import PlayerScorer
from config import STRATEGY, FPL_TEAM_ID

logger = logging.getLogger("fpl_auto")


class TransferOptimizer:
    """Find and execute the optimal transfers for the upcoming gameweek."""

    def __init__(self, client: FPLClient, scorer: PlayerScorer):
        self.client = client
        self.scorer = scorer

    def _get_current_squad(self) -> list[dict]:
        """Return current squad picks from FPL API."""
        team_data = self.client.get_my_team()
        if not team_data:
            raise RuntimeError("Could not fetch current team data.")
        return team_data["picks"]

    def _get_selling_prices(self) -> dict[int, float]:
        """Map player_id -> selling price in £m."""
        team_data = self.client.get_my_team()
        if not team_data:
            return {}
        return {p["element"]: p["selling_price"] / 10.0 for p in team_data["picks"]}

    def _build_transfer_candidates(
        self, scored_df: pd.DataFrame, squad_ids: set[int], budget: float
    ) -> pd.DataFrame:
        """Filter scored players to those not already in the squad and available."""
        candidates = scored_df[
            (~scored_df["id"].isin(squad_ids))
            & (scored_df["availability"] > 0)
            & (scored_df["now_cost_m"] <= budget + 5.0)  # loose upper bound, refined later
        ].copy()
        return candidates

    def find_best_transfers(self) -> dict:
        """
        Determine the optimal set of transfers.

        Returns dict with:
            - transfers_in: list of player IDs to buy
            - transfers_out: list of player IDs to sell
            - expected_gain: estimated point gain from the moves
            - hit_cost: transfer hit points (if any)
            - use_wildcard: bool
            - use_free_hit: bool
        """
        scored_df = self.scorer.score_players()
        squad_picks = self._get_current_squad()
        squad_ids = {p["element"] for p in squad_picks}
        selling_prices = self._get_selling_prices()

        team_data = self.client.get_my_team()
        bank = (team_data["transfers"]["bank"] / 10.0) if team_data else 0.0
        free_transfers = team_data["transfers"].get("limit", 1) if team_data else 1

        # Score current squad members
        squad_scores = scored_df[scored_df["id"].isin(squad_ids)].set_index("id")

        # Find weakest links in the squad
        squad_ranked = squad_scores.sort_values("adjusted_score", ascending=True)

        transfers_in = []
        transfers_out = []
        total_gain = 0.0
        budget_remaining = bank

        # Try up to max transfers (aggressive: willing to take hits)
        max_transfers_to_try = min(
            len(squad_ids),
            free_transfers + (STRATEGY["max_hit_points"] // 4)
        )

        used_teams = {}  # track team counts for 3-per-team rule
        for pick in squad_picks:
            pid = pick["element"]
            if pid in scored_df["id"].values:
                team_id = scored_df.loc[scored_df["id"] == pid, "team"].values[0]
                used_teams[team_id] = used_teams.get(team_id, 0) + 1

        candidates = self._build_transfer_candidates(scored_df, squad_ids, bank + 15.0)

        for i, (_, weak_player) in enumerate(squad_ranked.iterrows()):
            if i >= max_transfers_to_try:
                break

            player_id = weak_player.name  # this is the index = element id
            sell_price = selling_prices.get(player_id, weak_player["now_cost_m"] if "now_cost_m" in weak_player.index else 4.0)
            position = int(weak_player["element_type"])
            player_team = int(weak_player["team"])

            # Budget if we sell this player
            available_budget = budget_remaining + sell_price

            # Find best replacement of same position
            pos_candidates = candidates[
                (candidates["element_type"] == position)
                & (candidates["now_cost_m"] <= available_budget)
            ].copy()

            # Enforce max 3 per team
            for team_id, count in used_teams.items():
                if count >= STRATEGY["max_per_team"] and team_id != player_team:
                    pos_candidates = pos_candidates[pos_candidates["team"] != team_id]

            if pos_candidates.empty:
                continue

            best_replacement = pos_candidates.iloc[0]  # already sorted by adjusted_score desc
            score_gain = best_replacement["adjusted_score"] - weak_player["adjusted_score"]

            # Apply hit cost: each transfer beyond free costs 4 points
            is_free = i < free_transfers
            hit_cost = 0 if is_free else 4
            net_gain = score_gain * 10 - hit_cost  # scale score to approximate point impact

            if net_gain < STRATEGY["min_transfer_gain"] and not is_free:
                continue  # not worth the hit

            if score_gain <= 0.02 and is_free:
                continue  # marginal free transfer not worth the disruption

            # Execute this transfer
            transfers_out.append(player_id)
            transfers_in.append(int(best_replacement["id"]))
            total_gain += net_gain
            budget_remaining = available_budget - best_replacement["now_cost_m"]

            # Update team counts
            used_teams[player_team] = max(0, used_teams.get(player_team, 0) - 1)
            new_team = int(best_replacement["team"])
            used_teams[new_team] = used_teams.get(new_team, 0) + 1

            # Remove this candidate so we don't pick them again
            candidates = candidates[candidates["id"] != best_replacement["id"]]
            squad_ids.discard(player_id)
            squad_ids.add(int(best_replacement["id"]))

        hit_points = max(0, (len(transfers_in) - free_transfers)) * 4

        result = {
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "expected_gain": total_gain,
            "hit_cost": hit_points,
            "num_transfers": len(transfers_in),
            "free_transfers": free_transfers,
            "use_wildcard": False,
            "use_free_hit": False,
        }

        logger.info(
            f"Transfer plan: {len(transfers_in)} transfers, "
            f"expected gain={total_gain:.1f}, hit cost={hit_points}"
        )
        self._log_transfer_details(result, scored_df)
        return result

    def _log_transfer_details(self, result: dict, scored_df: pd.DataFrame):
        """Log human-readable transfer details."""
        name_map = scored_df.set_index("id")["web_name"].to_dict()
        for t_in, t_out in zip(result["transfers_in"], result["transfers_out"]):
            logger.info(f"  OUT: {name_map.get(t_out, t_out)} -> IN: {name_map.get(t_in, t_in)}")

    def execute_transfers(self, transfer_plan: dict) -> bool:
        """Submit the transfers to FPL."""
        if not transfer_plan["transfers_in"]:
            logger.info("No transfers to execute.")
            return True

        resp = self.client.make_transfers(
            transfers_in=transfer_plan["transfers_in"],
            transfers_out=transfer_plan["transfers_out"],
            wildcard=transfer_plan["use_wildcard"],
            free_hit=transfer_plan["use_free_hit"],
        )

        if resp is not None:
            logger.info("Transfers executed successfully!")
            return True
        else:
            logger.error("Transfer execution failed.")
            return False


class WildcardOptimizer:
    """
    Build the best possible 15-man squad from scratch (for wildcard / free hit).
    Uses a constrained optimization approach.
    """

    def __init__(self, client: FPLClient, scorer: PlayerScorer):
        self.client = client
        self.scorer = scorer

    def build_optimal_squad(self, budget: float = 100.0) -> list[dict]:
        """
        Build the highest-scoring squad within budget and positional constraints.
        Returns list of picks dicts.
        """
        scored_df = self.scorer.score_players()
        available = scored_df[scored_df["availability"] > 0].copy()

        squad = []
        remaining_budget = budget
        team_counts = {}
        position_limits = STRATEGY["position_limits"]

        # Greedy fill: for each position, pick top players within constraints
        for pos, (min_count, max_count) in sorted(position_limits.items()):
            pos_pool = available[available["element_type"] == pos].copy()
            picked = 0

            for _, player in pos_pool.iterrows():
                if picked >= max_count:
                    break
                team_id = int(player["team"])
                cost = player["now_cost_m"]

                if cost > remaining_budget:
                    continue
                if team_counts.get(team_id, 0) >= STRATEGY["max_per_team"]:
                    continue

                squad.append({
                    "element": int(player["id"]),
                    "position": 0,  # will be assigned later
                    "is_captain": False,
                    "is_vice_captain": False,
                    "element_type": pos,
                    "adjusted_score": player["adjusted_score"],
                    "web_name": player["web_name"],
                    "now_cost_m": cost,
                })
                remaining_budget -= cost
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                picked += 1

                # Remove from pool
                available = available[available["id"] != player["id"]]

        logger.info(f"Built wildcard squad: {len(squad)} players, "
                     f"£{budget - remaining_budget:.1f}m spent, £{remaining_budget:.1f}m remaining")

        return squad
