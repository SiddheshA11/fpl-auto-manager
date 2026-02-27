"""
FPL Auto Manager - News & Injury Researcher
Gathers real-time injury, suspension, and fitness data from multiple sources
each week to ensure the scoring engine has the latest information.

Sources:
    1. FPL API official player news/flags (chance_of_playing fields)
    2. FPL API player news text (press conference quotes, injury reports)
    3. Premier Injuries website (comprehensive injury table)
    4. Rotowire EPL lineups (predicted lineups and availability)
    5. Premier League official news RSS
"""
import re
import time
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import requests
import pandas as pd
from config import STRATEGY

logger = logging.getLogger("fpl_auto")


@dataclass
class PlayerNewsItem:
    """A single piece of news/intelligence about a player."""
    player_id: int
    player_name: str
    source: str
    news_type: str          # 'injury', 'suspension', 'doubt', 'fit', 'dropped', 'returning'
    detail: str
    severity: float         # 0.0 (out) to 1.0 (fully fit), how likely to play
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NewsResearcher:
    """
    Aggregates player news from multiple sources to produce an availability
    and fitness profile for every relevant player.
    """

    def __init__(self, bootstrap_data: dict):
        """
        Args:
            bootstrap_data: The full bootstrap-static response from FPL API.
        """
        self.bootstrap = bootstrap_data
        self.players = {p["id"]: p for p in bootstrap_data.get("elements", [])}
        self.teams = {t["id"]: t["name"] for t in bootstrap_data.get("teams", [])}
        self.news_items: list[PlayerNewsItem] = []
        self.player_profiles: dict[int, dict] = {}  # player_id -> aggregated profile
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })

    # ──────────────── Source 1: FPL API Official Data ────────────────

    def research_fpl_api_news(self):
        """
        Extract player news, flags, and chance-of-playing data
        from the FPL bootstrap response.
        """
        stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=STRATEGY["news_stale_hours"])
        flagged_count = 0
        news_count = 0

        for pid, player in self.players.items():
            status = player.get("status", "a")
            news_text = player.get("news", "") or ""
            news_added = player.get("news_added")
            chance_this = player.get("chance_of_playing_this_round")
            chance_next = player.get("chance_of_playing_next_round")
            name = f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()

            # Parse chance_of_playing (comes as int 0-100 or None)
            effective_chance = self._compute_effective_chance(status, chance_this, chance_next)

            # Determine news type from status
            news_type = self._status_to_news_type(status)

            # Check if news is stale
            is_fresh = True
            if news_added:
                try:
                    added_dt = datetime.fromisoformat(news_added.replace("Z", "+00:00"))
                    is_fresh = added_dt > stale_cutoff
                except (ValueError, TypeError):
                    is_fresh = True

            # Create news item if player is flagged or has news
            if status != "a" or news_text:
                item = PlayerNewsItem(
                    player_id=pid,
                    player_name=name,
                    source="fpl_api",
                    news_type=news_type,
                    detail=news_text if news_text else f"Status: {status}",
                    severity=effective_chance,
                )
                self.news_items.append(item)
                flagged_count += 1

                if news_text and is_fresh:
                    news_count += 1

            # Build base profile
            self.player_profiles[pid] = {
                "player_id": pid,
                "player_name": name,
                "fpl_status": status,
                "fpl_news": news_text,
                "fpl_news_fresh": is_fresh,
                "chance_this_round": chance_this,
                "chance_next_round": chance_next,
                "effective_chance": effective_chance,
                "web_sources": [],
                "final_availability": effective_chance,
            }

        logger.info(f"FPL API research: {flagged_count} flagged players, {news_count} with fresh news.")

    def _compute_effective_chance(self, status: str, chance_this, chance_next) -> float:
        """Compute a 0-1 availability score from FPL status and chance fields."""
        # Status-based baseline
        status_scores = {
            "a": 1.0,   # available
            "d": 0.50,  # doubtful (75% flag in FPL = ~50% real chance)
            "i": 0.0,   # injured
            "u": 0.0,   # unavailable
            "s": 0.0,   # suspended
            "n": 0.0,   # not in squad / loaned out
        }
        base = status_scores.get(status, 0.5)

        # If FPL provides explicit percentages, use those (more granular)
        if chance_next is not None:
            # FPL uses 0, 25, 50, 75, 100
            fpl_pct = chance_next / 100.0
            # Weight: 60% FPL percentage, 40% status-based
            return 0.6 * fpl_pct + 0.4 * base

        return base

    def _status_to_news_type(self, status: str) -> str:
        mapping = {
            "a": "fit",
            "d": "doubt",
            "i": "injury",
            "u": "injury",
            "s": "suspension",
            "n": "dropped",
        }
        return mapping.get(status, "doubt")

    # ──────────────── Source 2: Premier Injuries ────────────────

    def research_premier_injuries(self):
        """
        Scrape premierinjuries.com for the latest injury table.
        Falls back gracefully if the site is unavailable.
        """
        sources = STRATEGY.get("news_sources", [])
        url = sources[0] if len(sources) > 0 else ""
        if not url:
            return

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            html = resp.text
            injuries = self._parse_premier_injuries_html(html)
            logger.info(f"Premier Injuries: parsed {len(injuries)} injury entries.")

            for inj in injuries:
                self._merge_web_news(inj)

        except requests.RequestException as e:
            logger.warning(f"Premier Injuries fetch failed (non-critical): {e}")
        except Exception as e:
            logger.warning(f"Premier Injuries parse error: {e}")

    def _parse_premier_injuries_html(self, html: str) -> list[dict]:
        """
        Parse injury data from the HTML. Looks for player names, injury types,
        and expected return dates.
        """
        injuries = []

        # Pattern: find rows with player injury info
        # The site uses table rows with player name, injury type, and status
        # We use regex since we don't want to add beautifulsoup as a dependency
        row_pattern = re.compile(
            r'<tr[^>]*>.*?'
            r'class=["\']player["\'][^>]*>([^<]+)</.*?'  # player name
            r'class=["\']injury["\'][^>]*>([^<]+)</.*?'  # injury type
            r'class=["\']status["\'][^>]*>([^<]+)<',     # status
            re.DOTALL | re.IGNORECASE
        )

        # Broader fallback: look for injury-related text near player names
        # Many injury sites list: "Player Name - Injury Description - Expected Return"
        broad_pattern = re.compile(
            r'(?:injury|injured|doubt|suspended|illness|knock|hamstring|ankle|knee|groin|'
            r'thigh|calf|muscle|back|hip|shoulder|concussion|illness|covid|sick|strain|'
            r'fracture|surgery|torn|ligament|tendon)',
            re.IGNORECASE
        )

        # Try structured parsing first
        for match in row_pattern.finditer(html):
            name, injury_type, status = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            severity = self._injury_text_to_severity(status, injury_type)
            injuries.append({
                "player_name": name,
                "injury_type": injury_type,
                "status_text": status,
                "severity": severity,
                "source": "premierinjuries",
            })

        # If structured parsing found nothing, try a simpler line-by-line approach
        if not injuries:
            lines = html.split("\n")
            for line in lines:
                if broad_pattern.search(line):
                    # Try to extract a player name (capitalized words before the injury keyword)
                    name_match = re.search(r'([A-Z][a-z]+ (?:[A-Z][a-z]+)?)', line)
                    if name_match:
                        injuries.append({
                            "player_name": name_match.group(1).strip(),
                            "injury_type": "unknown",
                            "status_text": line.strip()[:100],
                            "severity": 0.3,
                            "source": "premierinjuries",
                        })

        return injuries

    def _injury_text_to_severity(self, status_text: str, injury_type: str) -> float:
        """Convert injury status text to a 0-1 severity score."""
        text = f"{status_text} {injury_type}".lower()

        if any(w in text for w in ["out", "ruled out", "surgery", "long-term", "season", "months"]):
            return 0.0
        if any(w in text for w in ["serious", "torn", "fracture", "ligament", "acl", "mcl"]):
            return 0.05
        if any(w in text for w in ["weeks", "2-3 weeks", "3-4 weeks"]):
            return 0.10
        if any(w in text for w in ["doubt", "uncertain", "assessment", "scan"]):
            return 0.30
        if any(w in text for w in ["knock", "minor", "day-to-day", "slight"]):
            return 0.50
        if any(w in text for w in ["training", "returned to training", "light training"]):
            return 0.65
        if any(w in text for w in ["full training", "fit", "available", "back in contention"]):
            return 0.85
        if any(w in text for w in ["expected to start", "nailed", "confirmed"]):
            return 0.95

        return 0.40  # unknown = cautious

    # ──────────────── Source 3: Rotowire Lineups ────────────────

    def research_rotowire_lineups(self):
        """
        Fetch predicted lineups from Rotowire.
        Players in predicted lineups get a slight boost;
        players explicitly listed as out get penalized.
        """
        sources = STRATEGY.get("news_sources", [])
        url = sources[1] if len(sources) > 1 else ""
        if not url:
            return

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            html = resp.text
            lineup_data = self._parse_rotowire_html(html)
            logger.info(f"Rotowire: parsed {len(lineup_data)} lineup entries.")

            for entry in lineup_data:
                self._merge_web_news(entry)

        except requests.RequestException as e:
            logger.warning(f"Rotowire fetch failed (non-critical): {e}")
        except Exception as e:
            logger.warning(f"Rotowire parse error: {e}")

    def _parse_rotowire_html(self, html: str) -> list[dict]:
        """Parse predicted lineup data from Rotowire HTML."""
        entries = []

        # Rotowire lists players with status indicators
        # Look for "confirmed" lineup players and "out" players
        out_pattern = re.compile(
            r'class=["\'](?:lineup-out|is-out)["\'][^>]*>.*?'
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
            re.DOTALL | re.IGNORECASE
        )

        confirmed_pattern = re.compile(
            r'class=["\'](?:lineup-confirmed|is-confirmed)["\'][^>]*>.*?'
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
            re.DOTALL | re.IGNORECASE
        )

        for match in out_pattern.finditer(html):
            entries.append({
                "player_name": match.group(1).strip(),
                "injury_type": "lineup_out",
                "status_text": "Not in predicted lineup",
                "severity": 0.10,
                "source": "rotowire",
            })

        for match in confirmed_pattern.finditer(html):
            entries.append({
                "player_name": match.group(1).strip(),
                "injury_type": "lineup_confirmed",
                "status_text": "In predicted lineup",
                "severity": 0.95,
                "source": "rotowire",
            })

        return entries

    # ──────────────── Source 4: FPL Player Detailed History ────────────────

    def research_minutes_trends(self):
        """
        Analyze recent minutes patterns to detect rotation risks.
        A player who's been benched 2 of the last 3 games is a rotation risk
        even if technically 'available'.
        """
        flagged_rotation = 0

        for pid, profile in self.player_profiles.items():
            player = self.players.get(pid)
            if not player or profile["effective_chance"] < 0.5:
                continue  # already flagged, skip

            # Use minutes from recent GWs (from bootstrap history summary)
            recent_minutes = player.get("minutes", 0)
            total_events = len([e for e in self.bootstrap.get("events", []) if e.get("finished")])

            if total_events > 3:
                avg_minutes = recent_minutes / total_events
                # If playing less than 60 min on average, flag rotation risk
                if avg_minutes < 60 and player.get("status") == "a":
                    penalty = max(0.5, avg_minutes / 90.0)
                    profile["rotation_risk"] = True
                    profile["rotation_penalty"] = penalty
                    profile["final_availability"] = min(
                        profile["final_availability"],
                        profile["effective_chance"] * penalty
                    )
                    flagged_rotation += 1

        if flagged_rotation:
            logger.info(f"Rotation analysis: {flagged_rotation} players flagged as rotation risks.")

    # ──────────────── News Merging ────────────────

    def _merge_web_news(self, entry: dict):
        """
        Merge a web-sourced news entry into player profiles.
        Fuzzy-matches player names to FPL player IDs.
        """
        name = entry["player_name"]
        matched_id = self._fuzzy_match_player(name)

        if matched_id is None:
            return  # couldn't match to an FPL player

        profile = self.player_profiles.get(matched_id)
        if profile is None:
            return

        profile["web_sources"].append({
            "source": entry["source"],
            "detail": entry["status_text"],
            "severity": entry["severity"],
        })

        # Update final availability: take the MINIMUM of all sources
        # (most conservative approach — if any source says they're out, trust it)
        web_severities = [s["severity"] for s in profile["web_sources"]]
        min_web = min(web_severities) if web_severities else 1.0

        # Blend: 50% FPL API, 50% worst web source (conservative)
        profile["final_availability"] = min(
            profile["final_availability"],
            0.5 * profile["effective_chance"] + 0.5 * min_web
        )

        # If web says fit but FPL says doubt, give a small boost
        if min_web >= 0.85 and profile["fpl_status"] == "d":
            profile["final_availability"] = max(
                profile["final_availability"],
                0.65  # upgrade from ~0.5 to 0.65
            )
            profile["web_upgraded"] = True

    def _fuzzy_match_player(self, name: str) -> int | None:
        """
        Match a name string to an FPL player ID.
        Tries exact second_name match first, then fuzzy partial matching.
        """
        name_lower = name.lower().strip()
        name_parts = name_lower.split()

        # Pass 1: exact second_name match
        for pid, player in self.players.items():
            if player.get("second_name", "").lower() == name_lower:
                return pid
            if player.get("web_name", "").lower() == name_lower:
                return pid

        # Pass 2: last word in input matches second_name
        if name_parts:
            last_name = name_parts[-1]
            matches = []
            for pid, player in self.players.items():
                sn = player.get("second_name", "").lower()
                wn = player.get("web_name", "").lower()
                if last_name == sn or last_name == wn:
                    matches.append(pid)
            if len(matches) == 1:
                return matches[0]

        # Pass 3: partial match (both first and last name present)
        if len(name_parts) >= 2:
            for pid, player in self.players.items():
                fn = player.get("first_name", "").lower()
                sn = player.get("second_name", "").lower()
                if name_parts[0] in fn and name_parts[-1] in sn:
                    return pid

        return None

    # ──────────────── Main Research Pipeline ────────────────

    def run_full_research(self) -> dict[int, dict]:
        """
        Execute all research sources and return the aggregated player profiles.

        Returns:
            dict mapping player_id -> profile dict with:
                - player_name: str
                - fpl_status: str (a/d/i/u/s/n)
                - fpl_news: str
                - chance_next_round: int or None
                - effective_chance: float (0-1, from FPL data only)
                - web_sources: list of dicts
                - final_availability: float (0-1, blended from all sources)
                - rotation_risk: bool (if detected)
        """
        logger.info("=" * 40)
        logger.info("STARTING NEWS RESEARCH")
        logger.info("=" * 40)

        # 1. FPL API (always available, most reliable)
        logger.info("Researching FPL API player news...")
        self.research_fpl_api_news()

        # 2. Web sources (fail gracefully)
        if STRATEGY.get("news_research_enabled", True):
            logger.info("Researching Premier Injuries...")
            self.research_premier_injuries()
            time.sleep(STRATEGY.get("request_delay", 1.0))

            logger.info("Researching Rotowire lineups...")
            self.research_rotowire_lineups()
            time.sleep(STRATEGY.get("request_delay", 1.0))

        # 3. Minutes/rotation analysis
        logger.info("Analyzing rotation risks...")
        self.research_minutes_trends()

        # Summary
        flagged = [p for p in self.player_profiles.values() if p["final_availability"] < 0.8]
        logger.info(f"Research complete: {len(flagged)} players with availability concerns.")
        self._log_key_findings(flagged)

        return self.player_profiles

    def _log_key_findings(self, flagged: list[dict]):
        """Log the most impactful findings."""
        # Sort by how many FPL points they've scored (most impactful first)
        for profile in sorted(flagged, key=lambda p: p["final_availability"])[:15]:
            pid = profile["player_id"]
            player = self.players.get(pid, {})
            total_pts = player.get("total_points", 0)
            sources = ", ".join(s["source"] for s in profile["web_sources"]) or "fpl_api only"

            logger.info(
                f"  {profile['player_name']:25s} | "
                f"status={profile['fpl_status']} | "
                f"chance={profile['final_availability']:.0%} | "
                f"pts={total_pts} | "
                f"sources: {sources}"
            )
            if profile.get("fpl_news"):
                logger.info(f"    └─ FPL news: {profile['fpl_news'][:80]}")
            if profile.get("rotation_risk"):
                logger.info(f"    └─ Rotation risk detected (penalty: {profile.get('rotation_penalty', 0):.2f})")

    # ──────────────── Utility: Get Availability Series ────────────────

    def get_availability_series(self) -> pd.Series:
        """
        Return a pandas Series mapping player_id -> final_availability (0-1).
        For use by the PlayerScorer to adjust scores.
        """
        data = {pid: p["final_availability"] for pid, p in self.player_profiles.items()}
        return pd.Series(data, name="researched_availability")

    def get_flagged_players(self, threshold: float = 0.7) -> list[dict]:
        """Return profiles of players below the availability threshold."""
        return [
            p for p in self.player_profiles.values()
            if p["final_availability"] < threshold
        ]

    def get_fit_boosts(self) -> dict[int, float]:
        """
        Return players who web sources say are fitter than FPL suggests.
        These get a scoring boost.
        """
        boosts = {}
        for pid, profile in self.player_profiles.items():
            if profile.get("web_upgraded"):
                boosts[pid] = STRATEGY.get("press_conference_boost", 0.05)
        return boosts
