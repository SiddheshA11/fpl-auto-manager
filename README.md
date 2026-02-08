# FPL Auto Manager

Fully automated Fantasy Premier League team manager. Runs weekly via GitHub Actions — handles transfers, lineup selection, captaincy, bench order, chip strategy, and league rival analysis without any manual intervention.

## What It Does

Every week (Friday evening + Saturday morning safety check), the bot:

1. **Researches injuries and news online** — scrapes Premier Injuries, Rotowire predicted lineups, and FPL press conference updates to get the latest availability picture
2. **Pulls advanced analytics** — fetches xG/xA data from Understat, computes form momentum, identifies set piece takers, analyzes home/away splits, and flags xG over/underperformers for regression
3. **Scores every player** using a weighted model enhanced by the research above (form, fixtures, xGI, PPG, ICT, minutes, research boosts)
4. **Analyzes your league rivals** — ownership %, captain picks, differentials, threats
5. **Decides on chip usage** — Wildcard, Free Hit, Bench Boost, Triple Captain with confidence thresholds
6. **Makes transfers** — aggressive strategy, willing to take hits up to -8 for high-ceiling moves
7. **Selects the best XI** — tries all valid formations, optimizes bench order
8. **Picks captain & vice-captain** — considers differential captaincy for upside
9. **Submits everything** to FPL automatically

The Saturday deadline check re-runs all research to catch any last-minute injury news or lineup leaks before the gameweek locks.

## Research Sources

The bot gathers intelligence from multiple sources each week:

| Source | What it provides | Reliability |
|--------|-----------------|-------------|
| **FPL API** | Official player flags, chance_of_playing %, news text | Always available |
| **Premier Injuries** | Comprehensive injury table with return dates | Web scrape, graceful fallback |
| **Rotowire** | Predicted starting lineups, confirmed outs | Web scrape, graceful fallback |
| **Understat** | xG, xA, actual vs expected goal involvement | Web scrape, graceful fallback |
| **FPL Bootstrap** | Form momentum, set piece duties, rotation patterns | Always available |

All web sources fail gracefully — if a site is down, the bot continues with FPL API data only.

## Setup (5 minutes)

### 1. Fork this repo

Click **Fork** on GitHub.

### 2. Find your FPL Team ID

Go to [fantasy.premierleague.com](https://fantasy.premierleague.com), click **Points**, and grab the number from the URL:
```
https://fantasy.premierleague.com/entry/1234567/event/1
                                         ^^^^^^^
                                         this is your Team ID
```

### 3. Get your FPL Cookie (Required for authentication)

Due to FPL's 2024 authentication changes, you need to extract cookies from your browser:

1. Log in to [fantasy.premierleague.com](https://fantasy.premierleague.com) in your browser
2. Open DevTools (F12 or right-click → Inspect)
3. Go to the **Network** tab
4. Refresh the page
5. Click on any request to `fantasy.premierleague.com`
6. In the **Headers** tab, find **Cookie** under Request Headers
7. Copy the entire cookie string (it's long!)

### 4. Add GitHub Secrets

Go to your forked repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name    | Value                                    |
|----------------|------------------------------------------|
| `FPL_COOKIE`   | **Required** - Cookie string from step 3 |
| `FPL_TEAM_ID`  | Your Team ID (e.g. 1234567)             |
| `FPL_EMAIL`    | Your FPL email (optional, for fallback)  |
| `FPL_PASSWORD` | Your FPL password (optional, for fallback)|

> **Note**: The `FPL_COOKIE` is the primary authentication method. Email/password may not work due to FPL's 2024 authentication changes.

### 5. Enable GitHub Actions

Go to **Actions** tab → click **I understand my workflows, go ahead and enable them**.

That's it. The bot runs automatically every Friday at 18:00 UTC and Saturday at 09:30 UTC.

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
