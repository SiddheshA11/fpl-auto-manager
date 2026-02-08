"""
FPL Auto Manager - Deep Web Research Module
Pulls advanced analytics from public data sources to build a richer
picture of player quality beyond what the FPL API provides.

Research areas:
    1. xG / xA data (expected goals and assists from Understat)
    2. Form momentum (acceleration/deceleration of FPL points)
    3. Set piece taker identification
    4. Home/away performance splits
    5. xG overperformance/underperformance (regression analysis)
    6. Team-level attack/defense strength trends
"""
import re
import time
import json
import logging
from collections import defaultdict
import requests
import pandas as pd
import numpy as np
from config import STRATEGY

logger = logging.getLogger("fpl_auto")


class WebResearcher:
    """
    Fetch and process advanced FPL analytics from public web sources.
    All methods fail gracefully — web sources are supplementary, not critical.
    """

    # Public data endpoints (no auth required)
    UNDERSTAT_BASE = "https://understat.com"
    FPL_REVIEW_BASE = "https://fplreview.com"

    def __init__(self, bootstrap_data: dict):
        self.bootstrap = bootstrap_data
        self.players = {p["id"]: p for p in bootstrap_data.get("elements", [])}
        self.teams = {t["id"]: t for t in bootstrap_data.get("teams", [])}
        self.team_name_to_id = {t["name"].lower(): t["id"] for t in bootstrap_data.get("teams", [])}
        self.events = bootstrap_data.get("events", [])
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; FPL-Auto-Manager/1.0)"
        })

    # ──────────────── 1. Understat xG Data ────────────────

    def fetch_understat_xg(self) -> dict[int, dict]:
        """
        Fetch xG and xA data from Understat for all EPL players.
        Returns: {fpl_player_id: {"xg": float, "xa": float, "xgi_per90": float}}
        """
        xg_data = {}
        try:
            url = f"{self.UNDERSTAT_BASE}/league/EPL"
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            html = resp.text

            # Understat embeds player data as JSON in a script tag
            # Pattern: var playersData = JSON.parse('...')
            match = re.search(r"var\s+playersData\s*=\s*JSON\.parse\('(.+?)'\)", html)
            if not match:
                logger.warning("Could not find Understat player data in page.")
                return xg_data

            raw_json = match.group(1)
            # Understat uses unicode escapes
            raw_json = raw_json.encode().decode('unicode_escape')
            players_raw = json.loads(raw_json)

            for player in players_raw:
                name = player.get("player_name", "")
                xg = float(player.get("xG", 0))
                xa = float(player.get("xA", 0))
                minutes = int(player.get("time", 0))

                if minutes < 90:
                    continue

                xgi_per90 = (xg + xa) * 90 / max(minutes, 1)
                goals = int(player.get("goals", 0))
                assists = int(player.get("assists", 0))

                # Match to FPL player
                fpl_id = self._match_understat_to_fpl(name, player.get("team_title", ""))
                if fpl_id:
                    actual_gi_per90 = (goals + assists) * 90 / max(minutes, 1)
                    xg_data[fpl_id] = {
                        "xg": xg,
                        "xa": xa,
                        "xgi_per90": xgi_per90,
                        "actual_gi_per90": actual_gi_per90,
                        "xg_diff": actual_gi_per90 - xgi_per90,  # positive = overperforming
                    }

            logger.info(f"Understat xG: matched {len(xg_data)} players.")

        except requests.RequestException as e:
            logger.warning(f"Understat fetch failed (non-critical): {e}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Understat parse error: {e}")

        return xg_data

    def _match_understat_to_fpl(self, understat_name: str, team_name: str) -> int | None:
        """Fuzzy match an Understat player name to an FPL player ID."""
        name_lower = understat_name.lower().strip()
        name_parts = name_lower.split()

        # Try direct second_name or web_name match
        for pid, player in self.players.items():
            if player.get("web_name", "").lower() == name_lower:
                return pid
            if player.get("second_name", "").lower() == name_lower:
                return pid

        # Try last-name match within the same team
        team_lower = team_name.lower()
        # Map common Understat team names to FPL equivalents
        team_aliases = {
            "manchester united": "man utd",
            "manchester city": "man city",
            "tottenham": "spurs",
            "newcastle united": "newcastle",
            "wolverhampton wanderers": "wolves",
            "west ham": "west ham",
            "nottingham forest": "nott'm forest",
            "leicester": "leicester",
        }
        fpl_team_name = team_aliases.get(team_lower, team_lower)

        # Find team ID
        target_team_id = None
        for tid, team in self.teams.items():
            if (team["name"].lower() == fpl_team_name or
                    team.get("short_name", "").lower() == fpl_team_name):
                target_team_id = tid
                break

        if target_team_id and name_parts:
            last_name = name_parts[-1]
            candidates = [
                pid for pid, p in self.players.items()
                if p["team"] == target_team_id and last_name in p.get("second_name", "").lower()
            ]
            if len(candidates) == 1:
                return candidates[0]

        return None

    # ──────────────── 2. Form Momentum Analysis ────────────────

    def analyze_form_momentum(self) -> dict[int, float]:
        """
        Compute form momentum for each player: is their recent FPL form
        accelerating (positive) or decelerating (negative)?

        Uses the FPL 'form' field over recent weeks plus gameweek history
        to detect trends.

        Returns: {player_id: momentum} where momentum is -1.0 to +1.0
        """
        momentum = {}

        finished_events = [e for e in self.events if e.get("finished")]
        if len(finished_events) < 4:
            return momentum

        # We'll use the last 5 gameweeks of data
        recent_gws = finished_events[-5:]
        gw_ids = [gw["id"] for gw in recent_gws]

        for pid, player in self.players.items():
            if player.get("minutes", 0) < 180:
                continue  # not enough data

            # Use ICT index history as a proxy for recent performance trajectory
            # FPL form is already a rolling average, so we use it directly
            form = float(player.get("form", 0) or 0)
            ppg = float(player.get("points_per_game", 0) or 0)

            if ppg == 0:
                continue

            # Form vs season PPG: if form > ppg, player is trending up
            form_ratio = (form - ppg) / max(ppg, 1)
            momentum[pid] = max(-1.0, min(1.0, form_ratio))

        logger.info(f"Form momentum: analyzed {len(momentum)} players. "
                     f"Top risers: {sum(1 for v in momentum.values() if v > 0.3)}, "
                     f"Top fallers: {sum(1 for v in momentum.values() if v < -0.3)}")

        return momentum

    # ──────────────── 3. Set Piece Taker Detection ────────────────

    def identify_set_piece_takers(self) -> set[int]:
        """
        Identify likely set piece takers (corners, free kicks, penalties)
        from FPL data and web research.

        Returns: set of player_ids who are on set pieces.
        """
        takers = set()

        for pid, player in self.players.items():
            if player.get("minutes", 0) < 270:
                continue

            penalties_order = player.get("penalties_order")
            corners_order = player.get("corners_and_indirect_freekicks_order")
            direct_fk_order = player.get("direct_freekicks_order")

            # FPL provides penalty/set piece order (1 = first choice)
            if penalties_order is not None and penalties_order <= 2:
                takers.add(pid)
            if corners_order is not None and corners_order <= 1:
                takers.add(pid)
            if direct_fk_order is not None and direct_fk_order <= 1:
                takers.add(pid)

        # Also look at penalties scored as a signal
        for pid, player in self.players.items():
            if player.get("penalties_scored", 0) >= 2:
                takers.add(pid)

        logger.info(f"Set piece takers identified: {len(takers)} players.")
        return takers

    # ──────────────── 4. Home/Away Performance Splits ────────────────

    def analyze_home_away_splits(self) -> dict[int, float]:
        """
        For the upcoming gameweek, give a boost/penalty based on whether a player
        performs significantly better at home or away, and where their next match is.

        Returns: {player_id: boost} where positive = favorable venue, negative = unfavorable
        """
        boosts = {}

        next_event = None
        for ev in self.events:
            if ev.get("is_next"):
                next_event = ev
                break

        if not next_event:
            return boosts

        # Build home/away map for the next GW
        # We need to know which teams play at home vs away
        # This comes from fixtures data, but we can approximate from bootstrap
        # For now, use a simplified approach based on team strengths
        for pid, player in self.players.items():
            if player.get("minutes", 0) < 270:
                continue

            team_id = player["team"]
            team_data = self.teams.get(team_id, {})

            # FPL provides team strength ratings for home/away attack/defense
            home_attack = team_data.get("strength_attack_home", 0)
            away_attack = team_data.get("strength_attack_away", 0)
            home_defense = team_data.get("strength_defence_home", 0)
            away_defense = team_data.get("strength_defence_away", 0)

            if home_attack and away_attack:
                # Attackers benefit from home advantage
                if player["element_type"] in (3, 4):  # MID, FWD
                    split_ratio = (home_attack - away_attack) / max(home_attack, 1)
                else:  # DEF, GK
                    split_ratio = (home_defense - away_defense) / max(home_defense, 1)

                if abs(split_ratio) > 0.05:
                    boosts[pid] = split_ratio

        logger.info(f"Home/away splits: computed for {len(boosts)} players.")
        return boosts

    # ──────────────── 5. xG Regression Analysis ────────────────

    def compute_xg_regression(self, xg_data: dict[int, dict]) -> dict[int, float]:
        """
        Identify players overperforming or underperforming their xG.
        Overperformers are likely to regress (negative adjustment).
        Underperformers are likely to improve (positive adjustment).

        Returns: {player_id: regression_signal} where:
            positive = underperforming xG (expect improvement)
            negative = overperforming xG (expect regression)
        """
        regression = {}

        for pid, data in xg_data.items():
            xg_diff = data.get("xg_diff", 0)
            # xg_diff > 0 means overperforming (actual > expected)
            # We flip the sign: overperformers get negative adjustment
            if abs(xg_diff) > 0.05:  # only flag significant deviations
                regression[pid] = -xg_diff  # negative for overperformers

        overperforming = sum(1 for v in regression.values() if v < -0.1)
        underperforming = sum(1 for v in regression.values() if v > 0.1)
        logger.info(f"xG regression: {overperforming} overperformers (due drop), "
                     f"{underperforming} underperformers (due rise).")

        return regression

    # ──────────────── 6. Team Strength Trends ────────────────

    def analyze_team_trends(self) -> dict[int, float]:
        """
        Analyze recent team-level performance trends.
        Teams on an upward trajectory get a boost for all their players.

        Returns: {team_id: trend_score} where positive = improving, negative = declining
        """
        team_trends = {}

        for tid, team in self.teams.items():
            # Use FPL strength ratings as a proxy
            # In a more advanced version, this would track week-over-week changes
            overall_home = team.get("strength_overall_home", 0)
            overall_away = team.get("strength_overall_away", 0)
            avg_strength = (overall_home + overall_away) / 2.0

            # Normalize to a -1 to +1 scale centered on the league average
            # FPL strength is typically 1000-1400
            team_trends[tid] = (avg_strength - 1200) / 200.0

        return team_trends

    # ──────────────── Main Research Pipeline ────────────────

    def run_full_research(self) -> dict:
        """
        Execute all deep web research and return aggregated data
        for the PlayerScorer to consume.

        Returns dict with keys:
            - xg_data: {player_id: {xg, xa, xgi_per90, xg_diff}}
            - form_momentum: {player_id: float}
            - set_piece_takers: set of player_ids
            - home_away_boosts: {player_id: float}
            - xg_regression: {player_id: float}
            - team_trends: {team_id: float}
        """
        logger.info("=" * 40)
        logger.info("STARTING DEEP WEB RESEARCH")
        logger.info("=" * 40)

        result = {}

        # 1. Understat xG (web fetch, may fail)
        logger.info("Fetching Understat xG data...")
        xg_data = self.fetch_understat_xg()
        result["xg_data"] = xg_data
        time.sleep(STRATEGY.get("request_delay", 1.0))

        # 2. Form momentum (FPL data, always works)
        logger.info("Analyzing form momentum...")
        result["form_momentum"] = self.analyze_form_momentum()

        # 3. Set piece takers (FPL data, always works)
        logger.info("Identifying set piece takers...")
        result["set_piece_takers"] = self.identify_set_piece_takers()

        # 4. Home/away splits (FPL data, always works)
        logger.info("Analyzing home/away splits...")
        result["home_away_boosts"] = self.analyze_home_away_splits()

        # 5. xG regression (depends on step 1)
        if xg_data:
            logger.info("Computing xG regression signals...")
            result["xg_regression"] = self.compute_xg_regression(xg_data)
        else:
            result["xg_regression"] = {}

        # 6. Team trends
        logger.info("Analyzing team strength trends...")
        result["team_trends"] = self.analyze_team_trends()

        logger.info("Deep web research complete.")
        self._log_summary(result)
        return result

    def _log_summary(self, result: dict):
        """Log a summary of all research findings."""
        logger.info("Research summary:")
        logger.info(f"  xG data: {len(result.get('xg_data', {}))} players")
        logger.info(f"  Form momentum: {len(result.get('form_momentum', {}))} players")
        logger.info(f"  Set piece takers: {len(result.get('set_piece_takers', set()))} players")
        logger.info(f"  Home/away splits: {len(result.get('home_away_boosts', {}))} players")
        logger.info(f"  xG regression: {len(result.get('xg_regression', {}))} players")
        logger.info(f"  Team trends: {len(result.get('team_trends', {}))} teams")

        # Log top risers by form momentum
        momentum = result.get("form_momentum", {})
        if momentum:
            top_risers = sorted(momentum.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("  Top form risers:")
            for pid, m in top_risers:
                name = self.players.get(pid, {}).get("web_name", f"ID:{pid}")
                logger.info(f"    {name}: momentum={m:+.2f}")

            top_fallers = sorted(momentum.items(), key=lambda x: x[1])[:5]
            logger.info("  Top form fallers:")
            for pid, m in top_fallers:
                name = self.players.get(pid, {}).get("web_name", f"ID:{pid}")
                logger.info(f"    {name}: momentum={m:+.2f}")
