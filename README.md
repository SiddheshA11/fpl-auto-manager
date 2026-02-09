# FPL Auto Manager

Fully automated Fantasy Premier League team manager. Runs via GitHub Actions — handles transfers, lineup selection, captaincy, bench order, chip strategy, and league rival analysis without any manual intervention.

## What It Does

**Dynamic scheduling**: Automatically detects each gameweek's actual deadline (Friday, Saturday, or Tuesday for double GWs) and runs at the right time.

1. **Researches injuries and news online** — scrapes Premier Injuries, Rotowire predicted lineups, and FPL press conference updates
2. **Pulls advanced analytics** — fetches xG/xA data from Understat, computes form momentum, identifies set piece takers
3. **Scores every player** using a weighted model enhanced by research (form, fixtures, xGI, PPG, ICT, minutes)
4. **Analyzes your league rivals** — ownership %, captain picks, differentials
5. **Decides on chip usage** — Wildcard, Free Hit, Bench Boost, Triple Captain
6. **Makes transfers** — aggressive strategy, willing to take hits for high-ceiling moves
7. **Selects the best XI** — tries all valid formations, optimizes bench order
8. **Picks captain & vice-captain** — considers differential captaincy
9. **Submits everything** to FPL automatically

Plus a safety run 2 hours before deadline to catch last-minute news.

## Research Sources

| Source | What it provides |
|--------|-----------------|
| **FPL API** | Official player flags, chance_of_playing %, news |
| **Premier Injuries** | Injury table with return dates |
| **Rotowire** | Predicted starting lineups |
| **Understat** | xG, xA, expected vs actual |

All web sources fail gracefully — if a site is down, the bot continues with FPL API data.

## Setup (5 minutes)

### 1. Fork this repo
Click **Fork** on GitHub.

### 2. Find your FPL Team ID
Go to [fantasy.premierleague.com](https://fantasy.premierleague.com), click **Points**, grab the number from URL:
```
https://fantasy.premierleague.com/entry/1234567/event/1
                                         ^^^^^^^ Team ID
```

### 3. Get your FPL Refresh Token (Recommended - lasts 30 days)

1. Log in to [fantasy.premierleague.com](https://fantasy.premierleague.com)
2. Open DevTools (F12) → **Application** tab → **Cookies** → `fantasy.premierleague.com`
3. Find `refresh_token` and copy its value

**Or** get FPL Cookie (shorter lifespan):
1. Go to **Network** tab → refresh page → click any request
2. Copy the full **Cookie** header value

### 4. Add GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name | Required | Description |
|-------------|----------|-------------|
| `FPL_REFRESH_TOKEN` | ✅ Recommended | Refresh token (lasts ~30 days) |
| `FPL_TEAM_ID` | ✅ Required | Your Team ID |
| `FPL_COOKIE` | Optional | Full cookie string (fallback) |
| `FPL_EMAIL` | Optional | FPL email (legacy fallback) |
| `FPL_PASSWORD` | Optional | FPL password (legacy fallback) |

> **💡 Tip**: Use `FPL_REFRESH_TOKEN` — it lasts 30 days vs 8 hours for cookies. You'll get a GitHub issue reminder monthly to refresh it.

### 5. Enable GitHub Actions
Go to **Actions** tab → click **I understand my workflows, go ahead and enable them**.

**That's it!** The scheduler runs every 2 hours, automatically triggering runs before each deadline.


### 6. (Optional) Test with a dry run

Go to **Actions** → **FPL Auto Manager - Weekly Run** → **Run workflow** → set `dry_run` to `true` → **Run workflow**.

This will simulate everything without making actual changes to your FPL team.

## Strategy Profile: Aggressive

The bot uses an aggressive strategy tuned for maximizing upside:

- **Transfers**: willing to take up to -8 in hits if the expected gain justifies it
- **Research-driven**: player scores are adjusted by live injury data, xG analytics, and form momentum
- **Captaincy**: considers differential captains when the gap to the safe pick is small
- **Chips**: plays chips at 60%+ confidence — targets DGWs for Bench Boost/TC, blank GWs for Free Hit
- **Wildcards**: triggers on sustained rank decline or large squad quality gap
- **xG regression**: penalizes overperformers and boosts underperformers who are due a correction

You can tune all parameters in `config.py` under the `STRATEGY` dict.

## Project Structure

```
fpl-auto-manager/
├── .github/workflows/
│   ├── weekly_run.yml        # Main Friday automation
│   └── deadline_check.yml    # Saturday safety net
├── config.py                 # All settings & strategy params
├── fpl_client.py             # FPL API client (auth + data)
├── news_researcher.py        # Injury & news research (web scraping)
├── web_research.py           # Deep analytics (xG, form, set pieces)
├── player_scorer.py          # Research-enhanced player evaluation
├── transfer_optimizer.py     # Transfer & wildcard optimizer
├── chip_strategy.py          # Chip decision engine
├── league_analyzer.py        # Rival team analysis
├── team_selector.py          # XI, bench, captain selection
├── manager.py                # Main orchestrator (8-step pipeline)
├── deadline_check.py         # Pre-deadline safety check with fresh research
├── requirements.txt
└── .env.example
```

## Local Development

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fpl-auto-manager.git
cd fpl-auto-manager

# Install deps
pip install -r requirements.txt

# Set up credentials
cp .env.example .env
# Edit .env with your details

# Dry run
python manager.py --dry-run

# Live run (makes real changes!)
python manager.py
```

## Logs

Every run uploads logs as GitHub Actions artifacts (retained for 30 days). You can review:

- Research findings (flagged players, xG data, form momentum)
- Transfer decisions and reasoning
- Captain picks
- Chip evaluations with confidence scores
- League rival insights
- Any errors or warnings

## Customization

Edit `config.py` to change:

- `max_hit_points` — max transfer hit budget per week (default: 8)
- `captain_aggressive` — toggle differential captaincy
- `bench_boost_min_dgw_players` — DGW threshold for BB
- `fixture_lookahead` — how many GWs to look ahead
- `weights` — adjust the scoring model factors
- `news_research_enabled` — toggle web scraping for injuries/news
- `news_stale_hours` — how old FPL news can be before it's ignored
- Schedule: edit cron times in `.github/workflows/weekly_run.yml`
