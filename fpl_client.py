"""
FPL Auto Manager - API Client
Handles authentication, data fetching, and team modifications via the FPL API.
"""
import time
import logging
import requests
import pandas as pd
from config import (
    FPL_EMAIL, FPL_PASSWORD, FPL_TEAM_ID,
    LOGIN_URL, ENDPOINTS, STRATEGY,
)

logger = logging.getLogger("fpl_auto")


class FPLClient:
    """Authenticated client for the Fantasy Premier League API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "FPL-Auto-Manager/1.0",
        })
        self.authenticated = False
        self._bootstrap_cache = None
        self._fixtures_cache = None

    # ──────────────── Authentication ────────────────

    def login(self) -> bool:
        """Authenticate with FPL and store session cookies."""
        # Step 1: Visit the login page to get initial cookies (CSRF token)
        try:
            login_page = self.session.get(LOGIN_URL)
            login_page.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to load login page: {e}")
            return False

        # Extract CSRF token from cookies
        csrf_token = self.session.cookies.get("csrftoken", "")
        if not csrf_token:
            logger.warning("No CSRF token found in cookies, attempting login anyway...")

        # Step 2: Submit login credentials
        payload = {
            "login": FPL_EMAIL,
            "password": FPL_PASSWORD,
            "redirect_uri": "https://fantasy.premierleague.com/",
            "app": "plfpl-web",
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": LOGIN_URL,
            "Origin": "https://users.premierleague.com",
        }
        if csrf_token:
            headers["X-CSRFToken"] = csrf_token

        # Step 3: POST login and follow redirects to capture all cookies
        resp = self.session.post(LOGIN_URL, data=payload, headers=headers, allow_redirects=True)
        
        # Log cookies for debugging
        cookies = self.session.cookies.get_dict()
        logger.debug(f"Cookies after login: {list(cookies.keys())}")
        
        # Check for authentication cookies
        auth_indicators = ["pl_profile", "sessionid", "datadome"]
        has_auth_cookie = any(cookie in cookies for cookie in auth_indicators)
        
        if has_auth_cookie:
            # Step 4: Verify by making an authenticated API call
            verify_url = ENDPOINTS["my_team"].format(manager_id=FPL_TEAM_ID)
            try:
                verify_resp = self.session.get(verify_url)
                if verify_resp.status_code == 200:
                    self.authenticated = True
                    logger.info("Successfully authenticated with FPL.")
                    return True
                else:
                    logger.warning(f"Auth cookies present but API returned {verify_resp.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Auth verification request failed: {e}")
        
        # Alternative: Check if we're logged in by looking for specific cookie patterns
        if "pl_profile" in cookies:
            self.authenticated = True
            logger.info("Authenticated (pl_profile cookie present).")
            return True
            
        # Check response for error messages
        if "Incorrect email or password" in resp.text or "errors" in resp.text.lower():
            logger.error("Authentication failed: Invalid credentials")
        else:
            logger.error(f"Authentication failed. Status: {resp.status_code}, Cookies: {list(cookies.keys())}")
        
        return False

    def _get(self, url: str, auth_required: bool = False) -> dict | list | None:
        """GET request with throttling and error handling."""
        if auth_required and not self.authenticated:
            raise RuntimeError("This endpoint requires authentication. Call login() first.")
        time.sleep(STRATEGY["request_delay"])
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"GET {url} failed: {e}")
            return None

    def _post(self, url: str, payload: dict) -> dict | None:
        """POST request (always requires auth)."""
        if not self.authenticated:
            raise RuntimeError("POST requires authentication. Call login() first.")
        time.sleep(STRATEGY["request_delay"])

        # We need the csrftoken for POST requests
        csrf = self.session.cookies.get("csrftoken", "")
        headers = {
            "Content-Type": "application/json",
            "X-CSRFToken": csrf,
            "Referer": "https://fantasy.premierleague.com/transfers",
        }
        try:
            resp = self.session.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json() if resp.content else {"status": "ok"}
        except requests.RequestException as e:
            logger.error(f"POST {url} failed: {e}")
            logger.error(f"Response body: {getattr(e.response, 'text', 'N/A')}")
            return None

    # ──────────────── Public Data ────────────────

    def get_bootstrap(self) -> dict:
        """Fetch the master data blob (players, teams, gameweeks)."""
        if self._bootstrap_cache is None:
            self._bootstrap_cache = self._get(ENDPOINTS["bootstrap"])
        return self._bootstrap_cache

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
        max_minutes = data["events"][-1]["id"] * 90 if data["events"] else 1
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
            if ev["is_current"]:
                return ev
        # If none is current, find the next one
        for ev in events:
            if ev["is_next"]:
                return ev
        return events[-1] if events else None

    def get_next_event(self) -> dict | None:
        """Return the next gameweek (the one transfers apply to)."""
        events = self.get_events()
        for ev in events:
            if ev["is_next"]:
                return ev
        # If season is over
        return None

    def get_fixtures(self) -> pd.DataFrame:
        """All fixtures for the season."""
        if self._fixtures_cache is None:
            data = self._get(ENDPOINTS["fixtures"])
            self._fixtures_cache = pd.DataFrame(data) if data else pd.DataFrame()
        return self._fixtures_cache

    def get_fixtures_for_event(self, event_id: int) -> pd.DataFrame:
        """Fixtures for a specific gameweek."""
        all_fixtures = self.get_fixtures()
        if all_fixtures.empty:
            return all_fixtures
        return all_fixtures[all_fixtures["event"] == event_id]

    def get_player_detail(self, element_id: int) -> dict | None:
        """Detailed player history + upcoming fixtures."""
        url = ENDPOINTS["element_summary"].format(element_id=element_id)
        return self._get(url)

    def get_live_event(self, event_id: int) -> dict | None:
        """Live stats for a gameweek."""
        url = ENDPOINTS["event_live"].format(event_id=event_id)
        return self._get(url)

    # ──────────────── Authenticated - My Team ────────────────

    def get_my_team(self) -> dict | None:
        """Get current squad, picks, budget, chips."""
        url = ENDPOINTS["my_team"].format(manager_id=FPL_TEAM_ID)
        return self._get(url, auth_required=True)

    def get_my_history(self) -> dict | None:
        """Season history for the authenticated manager."""
        url = ENDPOINTS["entry_history"].format(manager_id=FPL_TEAM_ID)
        return self._get(url)

    def get_my_entry(self) -> dict | None:
        """Public entry info (leagues, overall rank, etc.)."""
        url = ENDPOINTS["entry"].format(manager_id=FPL_TEAM_ID)
        return self._get(url)

    def get_my_transfers(self) -> list | None:
        """All transfers made this season."""
        url = ENDPOINTS["entry_transfers"].format(manager_id=FPL_TEAM_ID)
        return self._get(url)

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
                chips_played[chip["name"]] = chip["event"]

        all_chips = ["wildcard", "freehit", "bboost", "3xc"]
        return {c: {"played": c in chips_played, "event": chips_played.get(c)} for c in all_chips}

    # ──────────────── Team Modifications ────────────────

    def set_lineup(self, picks: list[dict], chip: str | None = None) -> dict | None:
        """
        Submit lineup (and optionally activate a chip).

        picks: list of dicts, each with keys:
            - element (int): player id
            - position (int): 1-11 starting, 12-15 bench
            - is_captain (bool)
            - is_vice_captain (bool)
        chip: one of 'wildcard', 'freehit', 'bboost', '3xc', or None
        """
        url = ENDPOINTS["my_team"].format(manager_id=FPL_TEAM_ID)
        payload = {"chip": chip, "picks": picks}
        logger.info(f"Setting lineup with chip={chip}, {len(picks)} picks.")
        return self._post(url, payload)

    def make_transfers(self, transfers_in: list[int], transfers_out: list[int],
                       wildcard: bool = False, free_hit: bool = False) -> dict | None:
        """
        Execute transfers.

        transfers_in:  list of player IDs to buy
        transfers_out: list of player IDs to sell (same length)
        wildcard: activate wildcard chip
        free_hit: activate free hit chip
        """
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

        payload = {
            "chip": chip,
            "entry": FPL_TEAM_ID,
            "event": next_event["id"],
            "transfers": [
                {"element_in": t_in, "element_out": t_out, "purchase_price": 0, "selling_price": 0}
                for t_in, t_out in zip(transfers_in, transfers_out)
            ],
        }

        url = ENDPOINTS["transfers"]
        logger.info(f"Making {len(transfers_in)} transfer(s). Chip={chip}")
        return self._post(url, payload)

    # ──────────────── League Data ────────────────

    def get_my_leagues(self) -> list[dict]:
        """Get list of classic leagues the manager is in."""
        entry = self.get_my_entry()
        if not entry:
            return []
        leagues = entry.get("leagues", {}).get("classic", [])
        return leagues

    def get_league_standings(self, league_id: int, page: int = 1) -> dict | None:
        """Get standings for a classic league."""
        url = ENDPOINTS["league_standings"].format(league_id=league_id)
        url += f"?page_standings={page}&page_new_entries=1&phase=1"
        return self._get(url)

    def get_entry_picks(self, manager_id: int, event_id: int) -> dict | None:
        """Get another manager's picks for a specific gameweek."""
        url = ENDPOINTS["entry_picks"].format(manager_id=manager_id, event_id=event_id)
        return self._get(url)
