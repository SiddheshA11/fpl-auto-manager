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
# the field. This is runtime configuration rather than a model constant,
# because it encodes a goal the model cannot infer: what winning means here.
#
#   0.0   maximise expected points, indifferent to what everyone else owns
#   > 0   track the template, trading points for a lower chance of a bad week
#   < 0   buy differentials, trading points for a higher chance of a great one
#
# Set POSITIVE, which is the opposite of where this started, because the
# measurement contradicted the reasoning. The argument for negative was that a
# twenty-person league is won by overtaking and therefore wants variance. It is
# a good argument and it was wrong here, for a reason no amount of theory would
# have surfaced. Measured on the GW1 pool:
#
#     tilt      XI xP (5 GW)    mean EO    owns Haaland
#    -0.30          163.95        11.0%          no
#     0.00          167.28        18.2%          no
#    +0.20          167.84        20.1%         YES
#    +0.30          165.79        23.2%         YES
#
# +0.20 is the highest-scoring setting *and* the one that owns the 71%-owned
# premium. -0.30 gave up 3.9 xP over five gameweeks AND carried the rank risk
# of not owning him: worse on both axes at once.
#
# The mechanism is worth recording, because it is not obvious. Haaland is the
# highest-xP player in the pool, and forcing him in *improves* the XI by 0.27
# xP. The optimiser declines him anyway because funding him guts the bench, and
# the objective values a bench place at 0.15 - which is correct in isolation.
# So a 0.27 xP bench preference was quietly deciding a 71%-ownership question.
# A small positive tilt is what stops that trade being made blind.
#
# This is the one number here worth re-measuring as the season moves: it
# depends on the price and ownership landscape, not on anything structural.
OWNERSHIP_WEIGHT = float(os.environ.get("FPL_OWNERSHIP_WEIGHT", "0.2"))
