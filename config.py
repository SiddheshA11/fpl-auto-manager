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

# ─── Runtime settings ───
# The strategy parameters that used to live here - scoring weights, hit
# thresholds, captain aggression, news-research toggles - are gone. They tuned
# a heuristic scorer that has been replaced by an expected-points model, where
# the equivalent decisions fall out of the points arithmetic instead of being
# dialled in. What remains is genuinely runtime configuration.
#
# Model constants now live next to the code that uses them, each with the
# evidence behind it: priors.py for shrinkage and team strength, xp_model.py
# for the scoring terms, optimizer.py for squad rules, chips.py for chip
# thresholds.
STRATEGY = {
    # Seconds between API calls. FPL tolerates a steady trickle; bursts get
    # rate limited and the run dies mid-transfer.
    "request_delay": 1.0,
}

# How far to tilt the squad away from pure expected points, toward or against
# the field. This one *is* runtime configuration rather than a model constant,
# because it encodes a goal the model cannot infer: what winning means here.
#
#   0.0   maximise expected points, indifferent to what everyone else owns
#   > 0   track the template, trading points for a lower chance of a bad week
#   < 0   buy differentials, trading points for a higher chance of a great one
#
# Set negative because the target is a private league of roughly twenty people.
# Beating a small, known field means overtaking it, and a squad that mirrors
# the template finishes wherever captaincy luck puts it - the differentials are
# the only thing that can move you up. Protecting overall rank among millions
# is the opposite problem and would want this positive.
#
# At -0.3 the GW1 squad gives up 2.8 xP over five gameweeks against the
# points-optimal build (176.3 vs 179.1) and drops mean ownership from 19.0% to
# 12.4%. -0.6 costs 11.2 xP for a further 2 points of ownership, which is a bad
# trade: it starts dropping genuinely best-in-class assets rather than swapping
# between near-equals.
OWNERSHIP_WEIGHT = float(os.environ.get("FPL_OWNERSHIP_WEIGHT", "-0.3"))
