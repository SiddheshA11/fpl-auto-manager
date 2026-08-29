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


# ─── Whose ownership the tilt aims at ───
#
# OWNERSHIP_WEIGHT above tilts toward `selected_by_percent`, the eleven-million
# player template. The league that decides the season is a couple of dozen
# specific managers, and those are measurably not the same distribution.
# Measured on GW2 2026-27 across 45 rivals, the template understates the field
# for every top-owned player and always in the same direction:
#
#     Kinsky       field 55.6%   template 22.4%   +33.2pp
#     Calafiori    field 68.9%   template 41.7%   +27.2pp
#     Szoboszlai   field 66.7%   template 43.2%   +23.5pp
#     Haaland      field 84.4%   template 68.5%   +15.9pp
#
# The swap cannot carry the weight across unchanged, which is the trap. The
# objective multiplies `weight * (1 - 2*ownership)`, so its force scales with
# that term's spread - and the field is the wider distribution, because in a
# 45-manager league most players are owned by nobody at all. Measured spread
# ratio 1.299 (GW1) and 1.301 (GW2), so +0.20 on the field would tilt at an
# effective +0.26: harder, not merely better aimed, and afterwards the two
# would be impossible to tell apart.
#
# +0.154 is the strength-matched weight, and at it the swap dominates the
# template on both axes rather than trading between them:
#
#     configuration                   XI xP   mean field EO
#     template  @ +0.20  production    48.93        33.7%
#     field     @ +0.154 matched       49.87        34.1%
#     no tilt   @  0.00  control       50.94        23.2%
#
# Same protection against the field, about a point more expected per gameweek,
# because the tilt is finally aimed at the distribution it was always meant to
# be aimed at. Reproduce with `python3 measure_rival_tilt.py`.
#
# Reads public endpoints only and spends no refresh token. Falls back to the
# template if the rival picks cannot be read, so the weekly run cannot fail on
# it.
USE_FIELD_OWNERSHIP = os.environ.get("FPL_USE_FIELD_OWNERSHIP", "1") not in ("0", "false", "False")
FIELD_OWNERSHIP_WEIGHT = float(os.environ.get("FPL_FIELD_OWNERSHIP_WEIGHT", "0.154"))

# The armband, tilted on captaincy ownership - a third distribution again, and
# far more concentrated than squad ownership. Across 45 rivals in GW2 only six
# players took the armband at all, and Joao Pedro was owned by 86.7% while
# being captained by 8.9%.
#
# Deliberately 0.0. The machinery is what was missing, not the conviction: the
# optimiser could not tilt captaincy at all before rivals.py existed, because
# FPL publishes squad ownership and not captain rates. Setting this above zero
# needs a season of evidence, and captaincy is the highest-leverage call of the
# week - the wrong sign here is expensive.
CAPTAIN_OWNERSHIP_WEIGHT = float(os.environ.get("FPL_CAPTAIN_OWNERSHIP_WEIGHT", "0.0"))
