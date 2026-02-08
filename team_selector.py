"""
FPL Auto Manager - Team Selection & Captain Optimizer
Picks the best starting XI, bench order, captain, and vice-captain
for the upcoming gameweek.
"""
import logging
import pandas as pd
from itertools import combinations
from fpl_client import FPLClient
from player_scorer import PlayerScorer
from config import STRATEGY

logger = logging.getLogger("fpl_auto")


class TeamSelector:
    """Optimize the starting XI, bench order, and captaincy."""

    def __init__(self, client: FPLClient, scorer: PlayerScorer):
        self.client = client
        self.scorer = scorer

    def _get_squad_with_scores(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Merge current squad picks with player scores."""
        team_data = self.client.get_my_team()
        if not team_data:
            raise RuntimeError("Could not fetch team data.")

        picks = team_data["picks"]
        squad_ids = [p["element"] for p in picks]

        squad = scored_df[scored_df["id"].isin(squad_ids)].copy()

        # Attach current position info
        pick_map = {p["element"]: p for p in picks}
        squad["current_position"] = squad["id"].map(lambda x: pick_map[x]["position"])
        squad["is_captain"] = squad["id"].map(lambda x: pick_map[x]["is_captain"])
        squad["is_vice_captain"] = squad["id"].map(lambda x: pick_map[x]["is_vice_captain"])

        return squad

    def select_best_xi(self, scored_df: pd.DataFrame) -> list[dict]:
        """
        Choose the optimal starting XI and bench from the 15-man squad.
        Tries all valid formations and picks the one with highest total score.

        Returns list of 15 pick dicts ready for the FPL API.
        """
        squad = self._get_squad_with_scores(scored_df)

        if len(squad) < 15:
            logger.warning(f"Squad has only {len(squad)} players (expected 15).")

        # Group players by position
        by_pos = {
            1: squad[squad["element_type"] == 1].sort_values("adjusted_score", ascending=False),
            2: squad[squad["element_type"] == 2].sort_values("adjusted_score", ascending=False),
            3: squad[squad["element_type"] == 3].sort_values("adjusted_score", ascending=False),
            4: squad[squad["element_type"] == 4].sort_values("adjusted_score", ascending=False),
        }

        best_score = -1
        best_xi = None
        best_bench = None

        for formation in STRATEGY["formation_options"]:
            gk_count, def_count, mid_count, fwd_count = formation

            # Check we have enough players
            if (len(by_pos[1]) < gk_count or len(by_pos[2]) < def_count or
                    len(by_pos[3]) < mid_count or len(by_pos[4]) < fwd_count):
                continue

            xi_players = []
            bench_players = []

            for pos, count in [(1, gk_count), (2, def_count), (3, mid_count), (4, fwd_count)]:
                pos_players = by_pos[pos]
                starters = pos_players.head(count)
                benched = pos_players.iloc[count:]
                xi_players.append(starters)
                bench_players.append(benched)

            xi_df = pd.concat(xi_players)
            bench_df = pd.concat(bench_players).sort_values("adjusted_score", ascending=False)

            total_score = xi_df["adjusted_score"].sum()

            if total_score > best_score:
                best_score = total_score
                best_xi = xi_df
                best_bench = bench_df

        if best_xi is None:
            logger.error("Could not find a valid formation!")
            return []

        logger.info(f"Best formation score: {best_score:.3f}")
        logger.info(f"Starting XI: {', '.join(best_xi['web_name'].tolist())}")
        logger.info(f"Bench: {', '.join(best_bench['web_name'].tolist())}")

        # Build picks list
        picks = []
        position = 1
        for _, player in best_xi.iterrows():
            picks.append({
                "element": int(player["id"]),
                "position": position,
                "is_captain": False,
                "is_vice_captain": False,
            })
            position += 1

        # Bench: GK first (position 12), then outfield by score
        bench_gk = best_bench[best_bench["element_type"] == 1]
        bench_outfield = best_bench[best_bench["element_type"] != 1].sort_values(
            "adjusted_score", ascending=False
        )

        for _, player in bench_gk.iterrows():
            picks.append({
                "element": int(player["id"]),
                "position": position,
                "is_captain": False,
                "is_vice_captain": False,
            })
            position += 1

        for _, player in bench_outfield.iterrows():
            picks.append({
                "element": int(player["id"]),
                "position": position,
                "is_captain": False,
                "is_vice_captain": False,
            })
            position += 1

        return picks

    def pick_captain(
        self, picks: list[dict], scored_df: pd.DataFrame, league_report: dict | None = None
    ) -> list[dict]:
        """
        Select captain and vice-captain from the starting XI.
        Aggressive strategy: willing to pick differentials.
        Uses league analysis to identify high-upside differential captains.
        """
        xi_picks = [p for p in picks if p["position"] <= 11]
        xi_ids = [p["element"] for p in xi_picks]
        xi_scores = scored_df[scored_df["id"].isin(xi_ids)].sort_values(
            "adjusted_score", ascending=False
        )

        if xi_scores.empty:
            return picks

        # Default: top scorer is captain
        captain_id = int(xi_scores.iloc[0]["id"])
        vice_captain_id = int(xi_scores.iloc[1]["id"]) if len(xi_scores) > 1 else captain_id

        # Aggressive: consider differential captain
        if STRATEGY["captain_aggressive"] and league_report:
            popular_captains = league_report.get("differentials", {}).get("popular_captains", [])
            popular_captain_ids = {c["element"] for c in popular_captains}

            # If our top pick is also everyone else's top pick, consider 2nd option
            # if it has high upside and low rival captaincy rate
            if captain_id in popular_captain_ids and len(xi_scores) > 1:
                second = xi_scores.iloc[1]
                second_id = int(second["id"])

                # Check if second choice is a genuine differential
                rival_captain_pct = 0
                for c in popular_captains:
                    if c["element"] == second_id:
                        rival_captain_pct = c["rival_captain_pct"]
                        break

                score_gap = xi_scores.iloc[0]["adjusted_score"] - second["adjusted_score"]

                # Switch if gap is small and second choice is differential
                if (score_gap < 0.05 and
                        rival_captain_pct < STRATEGY["captain_differential_threshold"]):
                    logger.info(
                        f"Differential captain pick: {second['web_name']} "
                        f"(rival captaincy {rival_captain_pct}%) over "
                        f"{xi_scores.iloc[0]['web_name']}"
                    )
                    vice_captain_id = captain_id
                    captain_id = second_id

        # Apply to picks
        for pick in picks:
            pick["is_captain"] = pick["element"] == captain_id
            pick["is_vice_captain"] = pick["element"] == vice_captain_id

        captain_name = scored_df.loc[scored_df["id"] == captain_id, "web_name"].values
        vc_name = scored_df.loc[scored_df["id"] == vice_captain_id, "web_name"].values
        logger.info(f"Captain: {captain_name[0] if len(captain_name) else captain_id}")
        logger.info(f"Vice-captain: {vc_name[0] if len(vc_name) else vice_captain_id}")

        return picks

    def apply_lineup(self, picks: list[dict], chip: str | None = None) -> bool:
        """Submit the lineup to FPL."""
        result = self.client.set_lineup(picks, chip=chip)
        if result is not None:
            logger.info(f"Lineup submitted successfully. Chip: {chip or 'none'}")
            return True
        else:
            logger.error("Failed to submit lineup.")
            return False
