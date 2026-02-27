"""
FPL Auto Manager - Configuration
All settings and strategy parameters live here.
"""
import os

# ─── FPL Credentials (from environment / GitHub Secrets) ───
FPL_EMAIL = os.environ.get("FPL_EMAIL", "")
FPL_PASSWORD = os.environ.get("FPL_PASSWORD", "")
FPL_TEAM_ID = int(os.environ.get("FPL_TEAM_ID", "0"))
# Cookie-based auth (workaround for FPL's 2024 auth changes)
# Extract the full cookie string from your browser after logging in
FPL_COOKIE = os.environ.get("FPL_COOKIE", "")
# Refresh token for automatic access token renewal (lasts ~30 days)
FPL_REFRESH_TOKEN = os.environ.get("FPL_REFRESH_TOKEN", "")
# GitHub token for triggering workflows dynamically
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "")

# ─── OAuth/OIDC Authentication (for token refresh) ───
# FPL uses PingOne/PingFederate behind account.premierleague.com
OAUTH_CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
OAUTH_TOKEN_URL = "https://account.premierleague.com/as/token"

# Legacy PingOne config (kept for backward compatibility)
PINGONE_ENV_ID = "68340de1-dfb9-412e-937c-20172986d129"
PINGONE_CLIENT_ID = OAUTH_CLIENT_ID
PINGONE_TOKEN_URL = OAUTH_TOKEN_URL

# ─── API Endpoints ───
BASE_URL = "https://fantasy.premierleague.com/api"
LOGIN_URL = "https://users.premierleague.com/accounts/login/"

ENDPOINTS = {
    "bootstrap": f"{BASE_URL}/bootstrap-static/",
    "fixtures": f"{BASE_URL}/fixtures/",
    "element_summary": f"{BASE_URL}/element-summary/{{element_id}}/",
    "my_team": f"{BASE_URL}/my-team/{{manager_id}}/",
    "entry": f"{BASE_URL}/entry/{{manager_id}}/",
    "entry_history": f"{BASE_URL}/entry/{{manager_id}}/history/",
    "entry_picks": f"{BASE_URL}/entry/{{manager_id}}/event/{{event_id}}/picks/",
    "entry_transfers": f"{BASE_URL}/entry/{{manager_id}}/transfers/",
    "league_standings": f"{BASE_URL}/leagues-classic/{{league_id}}/standings/",
    "event_live": f"{BASE_URL}/event/{{event_id}}/live/",
    "transfers": f"{BASE_URL}/transfers/",
}

# ─── Strategy Parameters (AGGRESSIVE profile) ───
STRATEGY = {
    # Transfer behaviour
    "max_hit_points": 8,            # willing to take up to -8 in hits
    "min_transfer_gain": 3.0,       # minimum expected point gain to justify a transfer
    "enable_multi_transfer": True,   # allow multiple transfers per week

    # Captain selection
    "captain_aggressive": True,      # pick high-ceiling captain even if risky
    "captain_differential_threshold": 15.0,  # % ownership below which a captain is "differential"

    # Chip thresholds
    "bench_boost_min_dgw_players": 10,  # min players with double fixtures to use BB
    "triple_captain_min_xp": 12.0,      # min expected points for TC pick
    "free_hit_rank_drop_trigger": 100_000,  # if rank drops by this much, consider FH
    "wildcard_form_drop_weeks": 3,      # weeks of declining rank before WC

    # Scoring weights for player evaluation
    "weights": {
        "form": 0.30,
        "fixture_difficulty": 0.20,
        "xgi_per90": 0.15,         # expected goal involvement
        "points_per_game": 0.15,
        "ict_index": 0.10,
        "minutes_reliability": 0.10,
    },

    # Position constraints (FPL rules)
    "squad_size": 15,
    "playing_xi": 11,
    "max_per_team": 3,
    "formation_options": [
        (1, 3, 5, 2), (1, 3, 4, 3), (1, 4, 4, 2),
        (1, 4, 3, 3), (1, 4, 5, 1), (1, 5, 4, 1),
        (1, 5, 3, 2), (1, 5, 2, 3),
    ],
    "position_limits": {
        1: (2, 2),   # GK: exactly 2
        2: (5, 5),   # DEF: exactly 5
        3: (5, 5),   # MID: exactly 5
        4: (3, 3),   # FWD: exactly 3
    },

    # Fixture look-ahead (how many gameweeks to consider)
    "fixture_lookahead": 5,

    # Request throttle (seconds between API calls)
    "request_delay": 1.0,

    # News research settings
    "news_research_enabled": True,
    "news_sources": [
        # Note: PremierInjuries and Rotowire are actively blocking automated scrapers.
        # Removing them from the default config to gracefully fallback to FPL native API flags.
    ],
    "news_stale_hours": 48,           # ignore FPL news older than this
    "news_weight_in_score": 0.20,     # how much news/availability affects final score
    "press_conference_boost": 0.05,   # boost for players confirmed fit in pressers
    "flagged_player_penalty": 0.40,   # penalty multiplier for 75% chance players
}
