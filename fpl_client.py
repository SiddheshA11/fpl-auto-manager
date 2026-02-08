"""
FPL Auto Manager - API Client
Uses the `fpl` package (amosbastian/fpl) for authentication and data fetching.
Wraps async operations in sync functions for easier integration.
"""
import asyncio
import logging
import aiohttp
import pandas as pd
from fpl import FPL
from config import (
    FPL_EMAIL, FPL_PASSWORD, FPL_TEAM_ID,
    ENDPOINTS, STRATEGY,
)

logger = logging.getLogger("fpl_auto")


class FPLClient:
    """Authenticated client for the Fantasy Premier League API using fpl package."""

    def __init__(self):
        self.authenticated = False
        self._session = None
        self._fpl = None
        self._bootstrap_cache = None
        self._fixtures_cache = None
        self._my_team_cache = None

    # ──────────────── Async Context Management ────────────────

    async def _init_session(self):
        """Initialize aiohttp session and FPL client."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._fpl = FPL(self._session)
        return self._fpl

    async def _close_session(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
            self._fpl = None

    # ──────────────── Authentication ────────────────

    async def _login_async(self) -> bool:
        """Authenticate with FPL using the fpl package."""
        try:
            fpl = await self._init_session()
            await fpl.login(email=FPL_EMAIL, password=FPL_PASSWORD)
            self.authenticated = True
            logger.info("Successfully authenticated with FPL.")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def login(self) -> bool:
        """Synchronous wrapper for login."""
        return asyncio.get_event_loop().run_until_complete(self._login_async())

    # ──────────────── Public Data ────────────────

    async def _get_bootstrap_async(self) -> dict:
        """Fetch the master data blob (players, teams, gameweeks)."""
        if self._bootstrap_cache is None:
            fpl = await self._init_session()
            # The fpl package loads bootstrap data on init
            self._bootstrap_cache = {
                "elements": list(fpl.elements.values()),
                "teams": list(fpl.teams.values()),
                "events": list(fpl.events.values()),
                "element_types": list(fpl.element_types.values()),
            }
        return self._bootstrap_cache

    def get_bootstrap(self) -> dict:
        """Synchronous wrapper for bootstrap data."""
        return asyncio.get_event_loop().run_until_complete(self._get_bootstrap_async())

    def get_players_df(self) -> pd.DataFrame:
        """Return all players as a DataFrame with useful columns."""
        data = self.get_bootstrap()
        players = pd.DataFrame(data["elements"])
        # Convert cost from tenths to millions
        players["now_cost_m"] = players["now_cost"] / 10.0
        # Numeric form
        players["form_numeric"] = pd.to_numeric(players["form"], errors="coerce").fillna(0)
        players["ppg_numeric"] = pd.to_numeric(players["points_per_game"], errors="coerce").fillna(0)
        players["ict_numeric"] = pd.to_numeric(players["ict_index"], errors="coerce").fillna(0)
        # Minutes reliability (fraction of possible minutes played)
        events = data["events"]
        max_minutes = max((e.get("id", 0) for e in events), default=1) * 90
        players["minutes_frac"] = players["minutes"].clip(upper=max_minutes) / max(max_minutes, 1)
        return players

    def get_teams_df(self) -> pd.DataFrame:
        data = self.get_bootstrap()
        return pd.DataFrame(data["teams"])

    def get_events(self) -> list[dict]:
        return self.get_bootstrap()["events"]

    def get_current_event(self) -> dict | None:
        """Return the current (or next upcoming) gameweek."""
        events = self.get_events()
        for ev in events:
            if ev.get("is_current"):
                return ev
        # If none is current, find the next one
        for ev in events:
            if ev.get("is_next"):
                return ev
        return events[-1] if events else None

    def get_next_event(self) -> dict | None:
        """Return the next gameweek (the one transfers apply to)."""
        events = self.get_events()
        for ev in events:
            if ev.get("is_next"):
                return ev
        # If season is over
        return None

    async def _get_fixtures_async(self) -> pd.DataFrame:
        """All fixtures for the season."""
        if self._fixtures_cache is None:
            fpl = await self._init_session()
            fixtures = await fpl.get_fixtures(return_json=True)
            self._fixtures_cache = pd.DataFrame(fixtures) if fixtures else pd.DataFrame()
        return self._fixtures_cache

    def get_fixtures(self) -> pd.DataFrame:
        """Synchronous wrapper for fixtures."""
        return asyncio.get_event_loop().run_until_complete(self._get_fixtures_async())

    def get_fixtures_for_event(self, event_id: int) -> pd.DataFrame:
        """Fixtures for a specific gameweek."""
        all_fixtures = self.get_fixtures()
        if all_fixtures.empty:
            return all_fixtures
        return all_fixtures[all_fixtures["event"] == event_id]

    async def _get_player_detail_async(self, element_id: int) -> dict | None:
        """Detailed player history + upcoming fixtures."""
        try:
            fpl = await self._init_session()
            summary = await fpl.get_player_summary(element_id, return_json=True)
            return summary
        except Exception as e:
            logger.error(f"Failed to get player detail for {element_id}: {e}")
            return None

    def get_player_detail(self, element_id: int) -> dict | None:
        """Synchronous wrapper for player detail."""
        return asyncio.get_event_loop().run_until_complete(
            self._get_player_detail_async(element_id)
        )

    async def _get_live_event_async(self, event_id: int) -> dict | None:
        """Live stats for a gameweek."""
        try:
            fpl = await self._init_session()
            gameweeks = await fpl.get_gameweeks([event_id], return_json=True)
            return gameweeks[0] if gameweeks else None
        except Exception as e:
            logger.error(f"Failed to get live event {event_id}: {e}")
            return None

    def get_live_event(self, event_id: int) -> dict | None:
        """Synchronous wrapper for live event data."""
        return asyncio.get_event_loop().run_until_complete(
            self._get_live_event_async(event_id)
        )

    # ──────────────── Authenticated - My Team ────────────────

    async def _get_my_team_async(self) -> dict | None:
        """Get current squad, picks, budget, chips."""
        if not self.authenticated:
            raise RuntimeError("Authentication required. Call login() first.")
        try:
            fpl = await self._init_session()
            user = await fpl.get_user(FPL_TEAM_ID)
            team = await user.get_team()
            
            # Get transfers info
            transfers_info = await user.get_transfers_status()
            
            # Build response similar to API format
            self._my_team_cache = {
                "picks": [{"element": p.id, "position": i+1} for i, p in enumerate(team)],
                "transfers": {
                    "bank": transfers_info.get("bank", 0),
                    "limit": transfers_info.get("limit", 1),
                },
                "chips": await user.get_chips_status() if hasattr(user, 'get_chips_status') else [],
            }
            return self._my_team_cache
        except Exception as e:
            logger.error(f"Failed to get team data: {e}")
            return None

    def get_my_team(self) -> dict | None:
        """Synchronous wrapper for my team."""
        return asyncio.get_event_loop().run_until_complete(self._get_my_team_async())

    async def _get_my_history_async(self) -> dict | None:
        """Season history for the authenticated manager."""
        try:
            fpl = await self._init_session()
            user = await fpl.get_user(FPL_TEAM_ID)
            history = await user.get_history()
            return history
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return None

    def get_my_history(self) -> dict | None:
        """Synchronous wrapper for my history."""
        return asyncio.get_event_loop().run_until_complete(self._get_my_history_async())

    async def _get_my_entry_async(self) -> dict | None:
        """Public entry info (leagues, overall rank, etc.)."""
        try:
            fpl = await self._init_session()
            user = await fpl.get_user(FPL_TEAM_ID, return_json=True)
            return user
        except Exception as e:
            logger.error(f"Failed to get entry: {e}")
            return None

    def get_my_entry(self) -> dict | None:
        """Synchronous wrapper for my entry."""
        return asyncio.get_event_loop().run_until_complete(self._get_my_entry_async())

    async def _get_my_transfers_async(self) -> list | None:
        """All transfers made this season."""
        try:
            fpl = await self._init_session()
            user = await fpl.get_user(FPL_TEAM_ID)
            transfers = await user.get_transfers()
            return transfers
        except Exception as e:
            logger.error(f"Failed to get transfers: {e}")
            return None

    def get_my_transfers(self) -> list | None:
        """Synchronous wrapper for my transfers."""
        return asyncio.get_event_loop().run_until_complete(self._get_my_transfers_async())

    def get_remaining_budget(self) -> float:
        """Remaining transfer budget in £m."""
        team_data = self.get_my_team()
        if team_data and "transfers" in team_data:
            return team_data["transfers"].get("bank", 0) / 10.0
        return 0.0

    def get_free_transfers(self) -> int:
        """Number of free transfers available."""
        team_data = self.get_my_team()
        if team_data and "transfers" in team_data:
            return team_data["transfers"].get("limit", 1)
        return 1

    def get_chips_status(self) -> dict:
        """Return which chips are available / played."""
        team_data = self.get_my_team()
        chips_played = {}
        if team_data and "chips" in team_data:
            for chip in team_data["chips"]:
                if isinstance(chip, dict):
                    chips_played[chip.get("name", "")] = chip.get("event")

        all_chips = ["wildcard", "freehit", "bboost", "3xc"]
        return {c: {"played": c in chips_played, "event": chips_played.get(c)} for c in all_chips}

    # ──────────────── Team Modifications ────────────────

    async def _set_lineup_async(self, picks: list[dict], chip: str | None = None) -> dict | None:
        """
        Submit lineup (and optionally activate a chip).
        Note: The fpl package may have limited support for this.
        """
        if not self.authenticated:
            raise RuntimeError("Authentication required. Call login() first.")
        
        try:
            fpl = await self._init_session()
            user = await fpl.get_user(FPL_TEAM_ID)
            
            # Use the user's method to set lineup if available
            # The fpl package's exact method may vary
            result = await user.set_lineup(picks, chip=chip)
            logger.info(f"Set lineup with chip={chip}, {len(picks)} picks.")
            return result
        except AttributeError:
            logger.warning("set_lineup not directly supported by fpl package, using direct API call")
            # Fall back to direct API call
            return await self._direct_api_set_lineup(picks, chip)
        except Exception as e:
            logger.error(f"Failed to set lineup: {e}")
            return None

    async def _direct_api_set_lineup(self, picks: list[dict], chip: str | None = None) -> dict | None:
        """Direct API call to set lineup as fallback."""
        url = f"https://fantasy.premierleague.com/api/my-team/{FPL_TEAM_ID}/"
        payload = {"chip": chip, "picks": picks}
        
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Set lineup failed: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Direct API set lineup failed: {e}")
            return None

    def set_lineup(self, picks: list[dict], chip: str | None = None) -> dict | None:
        """Synchronous wrapper for set lineup."""
        logger.info(f"Setting lineup with chip={chip}, {len(picks)} picks.")
        return asyncio.get_event_loop().run_until_complete(
            self._set_lineup_async(picks, chip)
        )

    async def _make_transfers_async(
        self, 
        transfers_in: list[int], 
        transfers_out: list[int],
        wildcard: bool = False, 
        free_hit: bool = False
    ) -> dict | None:
        """Execute transfers."""
        if not self.authenticated:
            raise RuntimeError("Authentication required. Call login() first.")
            
        if len(transfers_in) != len(transfers_out):
            raise ValueError("transfers_in and transfers_out must be the same length.")

        next_event = self.get_next_event()
        if not next_event:
            logger.error("No upcoming event found. Season may be over.")
            return None

        chip = None
        if wildcard:
            chip = "wildcard"
        elif free_hit:
            chip = "freehit"

        try:
            fpl = await self._init_session()
            user = await fpl.get_user(FPL_TEAM_ID)
            
            # Try to use the user's transfer method if available
            result = await user.make_transfers(
                transfers_in, transfers_out, 
                chip=chip, event=next_event["id"]
            )
            logger.info(f"Made {len(transfers_in)} transfer(s). Chip={chip}")
            return result
        except AttributeError:
            logger.warning("make_transfers not directly supported, using direct API call")
            return await self._direct_api_transfers(transfers_in, transfers_out, chip, next_event["id"])
        except Exception as e:
            logger.error(f"Failed to make transfers: {e}")
            return None

    async def _direct_api_transfers(
        self, 
        transfers_in: list[int], 
        transfers_out: list[int], 
        chip: str | None,
        event_id: int
    ) -> dict | None:
        """Direct API call to make transfers as fallback."""
        url = "https://fantasy.premierleague.com/api/transfers/"
        payload = {
            "chip": chip,
            "entry": FPL_TEAM_ID,
            "event": event_id,
            "transfers": [
                {"element_in": t_in, "element_out": t_out, "purchase_price": 0, "selling_price": 0}
                for t_in, t_out in zip(transfers_in, transfers_out)
            ],
        }
        
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json() if resp.content_length else {"status": "ok"}
                else:
                    text = await resp.text()
                    logger.error(f"Transfers failed: {resp.status} - {text}")
                    return None
        except Exception as e:
            logger.error(f"Direct API transfers failed: {e}")
            return None

    def make_transfers(
        self, 
        transfers_in: list[int], 
        transfers_out: list[int],
        wildcard: bool = False, 
        free_hit: bool = False
    ) -> dict | None:
        """Synchronous wrapper for make transfers."""
        logger.info(f"Making {len(transfers_in)} transfer(s). Wildcard={wildcard}, FreeHit={free_hit}")
        return asyncio.get_event_loop().run_until_complete(
            self._make_transfers_async(transfers_in, transfers_out, wildcard, free_hit)
        )

    # ──────────────── League Data ────────────────

    def get_my_leagues(self) -> list[dict]:
        """Get list of classic leagues the manager is in."""
        entry = self.get_my_entry()
        if not entry:
            return []
        leagues = entry.get("leagues", {}).get("classic", [])
        return leagues

    async def _get_league_standings_async(self, league_id: int) -> dict | None:
        """Get standings for a classic league."""
        try:
            fpl = await self._init_session()
            league = await fpl.get_classic_league(league_id, return_json=True)
            return league
        except Exception as e:
            logger.error(f"Failed to get league standings: {e}")
            return None

    def get_league_standings(self, league_id: int, page: int = 1) -> dict | None:
        """Synchronous wrapper for league standings."""
        return asyncio.get_event_loop().run_until_complete(
            self._get_league_standings_async(league_id)
        )

    async def _get_entry_picks_async(self, manager_id: int, event_id: int) -> dict | None:
        """Get another manager's picks for a specific gameweek."""
        try:
            fpl = await self._init_session()
            user = await fpl.get_user(manager_id)
            picks = await user.get_picks(event_id)
            return picks
        except Exception as e:
            logger.error(f"Failed to get entry picks: {e}")
            return None

    def get_entry_picks(self, manager_id: int, event_id: int) -> dict | None:
        """Synchronous wrapper for entry picks."""
        return asyncio.get_event_loop().run_until_complete(
            self._get_entry_picks_async(manager_id, event_id)
        )

    # ──────────────── Cleanup ────────────────

    def close(self):
        """Close the session."""
        asyncio.get_event_loop().run_until_complete(self._close_session())

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self._session:
                self.close()
        except Exception:
            pass
