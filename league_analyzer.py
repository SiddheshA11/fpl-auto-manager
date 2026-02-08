"""
FPL Auto Manager - League Rival Analysis
Analyzes the teams of managers in your mini-leagues to find differentials,
popular picks, and strategic edges.
"""
import logging
from collections import Counter
import pandas as pd
from fpl_client import FPLClient
from config import FPL_TEAM_ID

logger = logging.getLogger("fpl_auto")


class LeagueAnalyzer:
    """Analyze rival teams in all your classic leagues."""

    def __init__(self, client: FPLClient):
        self.client = client

    def get_all_rival_ids(self, max_per_league: int = 50) -> list[int]:
        """Collect manager IDs from all classic leagues (excluding self)."""
        leagues = self.client.get_my_leagues()
        rival_ids = set()

        for league in leagues:
            league_id = league["id"]
            standings = self.client.get_league_standings(league_id)
            if not standings or "standings" not in standings:
                continue

            results = standings["standings"].get("results", [])
            for entry in results[:max_per_league]:
                eid = entry["entry"]
                if eid != FPL_TEAM_ID:
                    rival_ids.add(eid)

        logger.info(f"Found {len(rival_ids)} unique rivals across {len(leagues)} leagues.")
        return list(rival_ids)

    def get_rival_picks(self, rival_ids: list[int], event_id: int) -> list[dict]:
        """Fetch picks for each rival in the given gameweek."""
        all_picks = []
        for rid in rival_ids:
            picks_data = self.client.get_entry_picks(rid, event_id)
            if picks_data and "picks" in picks_data:
                all_picks.append({
                    "manager_id": rid,
                    "picks": picks_data["picks"],
                    "active_chip": picks_data.get("active_chip"),
                })
        logger.info(f"Fetched picks for {len(all_picks)}/{len(rival_ids)} rivals.")
        return all_picks

    def analyze_ownership(self, rival_picks: list[dict], players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ownership percentages and captain rates among rivals.

        Returns DataFrame with columns:
            - element: player id
            - web_name: player name
            - rival_ownership_pct: % of rivals who own this player
            - rival_captain_pct: % of rivals who captained this player
            - starting_pct: % of rivals who have this player in starting XI
        """
        total_rivals = len(rival_picks)
        if total_rivals == 0:
            return pd.DataFrame()

        ownership = Counter()
        captain_counts = Counter()
        starting_counts = Counter()

        for rival in rival_picks:
            for pick in rival["picks"]:
                pid = pick["element"]
                ownership[pid] += 1
                if pick["is_captain"]:
                    captain_counts[pid] += 1
                if pick["position"] <= 11:
                    starting_counts[pid] += 1

        rows = []
        for pid, count in ownership.items():
            player_info = players_df[players_df["id"] == pid]
            name = player_info["web_name"].values[0] if not player_info.empty else f"Player {pid}"
            rows.append({
                "element": pid,
                "web_name": name,
                "rival_ownership_pct": round(100 * count / total_rivals, 1),
                "rival_captain_pct": round(100 * captain_counts.get(pid, 0) / total_rivals, 1),
                "starting_pct": round(100 * starting_counts.get(pid, 0) / total_rivals, 1),
            })

        df = pd.DataFrame(rows).sort_values("rival_ownership_pct", ascending=False)
        return df

    def find_differentials(
        self, ownership_df: pd.DataFrame, my_squad_ids: list[int], threshold: float = 25.0
    ) -> dict:
        """
        Identify:
        1. Players you own that rivals DON'T (your differentials)
        2. Players rivals own that you DON'T (threats to monitor)
        3. Popular captain picks among rivals

        threshold: ownership % below which a player is a differential
        """
        if ownership_df.empty:
            return {"my_differentials": [], "rival_threats": [], "popular_captains": []}

        # My differentials: I own them, low rival ownership
        my_owned = ownership_df[ownership_df["element"].isin(my_squad_ids)]
        my_differentials = my_owned[my_owned["rival_ownership_pct"] < threshold]

        # Rival threats: high rival ownership but I don't own
        not_owned = ownership_df[~ownership_df["element"].isin(my_squad_ids)]
        rival_threats = not_owned[not_owned["rival_ownership_pct"] >= 50.0]

        # Popular captains
        popular_captains = ownership_df[ownership_df["rival_captain_pct"] >= 10.0].sort_values(
            "rival_captain_pct", ascending=False
        )

        result = {
            "my_differentials": my_differentials.to_dict("records"),
            "rival_threats": rival_threats.head(10).to_dict("records"),
            "popular_captains": popular_captains.head(5).to_dict("records"),
        }

        logger.info(
            f"Differentials: {len(result['my_differentials'])} unique picks, "
            f"{len(result['rival_threats'])} threats, "
            f"{len(result['popular_captains'])} popular captains"
        )
        return result

    def generate_league_report(self, event_id: int, players_df: pd.DataFrame) -> dict:
        """
        Full league analysis report for the current gameweek.
        """
        rival_ids = self.get_all_rival_ids()
        if not rival_ids:
            logger.warning("No rivals found in any league.")
            return {}

        rival_picks = self.get_rival_picks(rival_ids, event_id)
        ownership = self.analyze_ownership(rival_picks, players_df)

        my_team = self.client.get_my_team()
        my_squad_ids = [p["element"] for p in my_team["picks"]] if my_team else []

        differentials = self.find_differentials(ownership, my_squad_ids)

        # Chip usage among rivals
        chip_usage = Counter()
        for rival in rival_picks:
            if rival["active_chip"]:
                chip_usage[rival["active_chip"]] += 1

        report = {
            "event_id": event_id,
            "total_rivals": len(rival_ids),
            "rivals_analyzed": len(rival_picks),
            "top_owned_players": ownership.head(15).to_dict("records"),
            "differentials": differentials,
            "rival_chip_usage": dict(chip_usage),
        }

        logger.info(f"League report generated for GW{event_id}: {len(rival_ids)} rivals analyzed.")
        return report
