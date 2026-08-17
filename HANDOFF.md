# Handoff — FPL Auto Manager rebuild

State as of 2026-08-17. Branch `rebuild/xp-model`, 10 commits, not pushed,
working tree clean, 34 tests passing.

## What this project is

Fully automated Fantasy Premier League manager. Runs on GitHub Actions, picks
transfers, lineup, captain, bench order and chips, submits them to FPL without
intervention. Owner wants to win his league and does not want to check in
weekly.

## What was rebuilt

The heuristic core was replaced. It used to score players on a unitless 0-1
weighted composite, then multiply that by 10 and compare it against a -4
transfer hit — so every threshold in the system was arbitrary.

Now:

| file | role |
|---|---|
| `priors.py` | Per-90 priors with position-aware shrinkage; team attack/defence strength; promoted-club priors |
| `xp_model.py` | Expected points in real FPL points. Minutes model, threshold-based defensive contribution, empirical bonus curve, scoring read live from `game_config` |
| `optimizer.py` | MILP over squad + starting XI + captain jointly, via scipy HiGHS. Hits priced as actual -4 |
| `chips.py` | Chip availability across both half-season windows; chip value in expected points; DGW/BGW detection |
| `manager.py` | Weekly run orchestrator (unattended, submits real transfers) |
| `deadline_check.py` | Pre-deadline safety: re-reads availability, repairs lineup only, makes no transfers |
| `backtest.py` | Replays a past season, priors restricted to strictly earlier seasons |
| `recommend.py` | CLI: `--build` for a squad from scratch, `--transfer` for moves |
| `fetch_data.py` | Live API snapshots + the vaastav historical dataset |

Deleted (~1,400 lines): `web_research.py`, `news_researcher.py`,
`player_scorer.py`, `transfer_optimizer.py`, `team_selector.py`,
`chip_strategy.py`. The scrapers pulled xG and injury flags the FPL API now
returns directly, and the sites had begun blocking them.

## Validation

Backtested over 33 gameweeks of 2025-26 against a points-per-game baseline:

- top-20 picks: **+0.120** actual points per player per gameweek
- top-30 picks: **+0.128**, winning in 58% of gameweeks
- rank correlation, players who appeared: 0.315 vs 0.292

**Known weakness:** the model *loses* on all-players rank correlation (0.630 vs
0.690). That gap is ordering players who never appear, which points-per-game
wins by construction. It matters for transfer suggestions more than squad
picks, and the minutes model is where to fix it.

Metric noise is large: per-gameweek standard error on top-20 is ~0.18, so
differences below that are not resolvable with one season.

**The backtest was run on 2025-26, which used the old Bonus Points System.**
FPL rebalanced BPS for 2026/27 (see section 4b). The measured edge therefore
carries an unquantified assumption that it transfers across the rules change,
and it partly will not: bonus is one of the model's scoring terms and it is now
mis-weighted by position. Re-run the backtest against 2026/27 once enough
gameweeks exist to be worth it.

## Account

The project uses **siddheshagarwal10@gmail.com**. That is the owner's decision
and the account every secret should correspond to.

There are two Premier League accounts, and this caused real confusion on
2026-08-17 — worth knowing about if authentication misbehaves:

- Premier League mail going back to 2018 (account activation, email
  confirmation, a password reset on 2026-02-08 minutes before `FPL_EMAIL` and
  `FPL_PASSWORD` were created) lands in **siddheshagarwal10@gmail.com**.
- A browser session captured that same day produced an id_token whose `email`
  and `preferred_username` claims were **siddheshrox123@gmail.com**, with a
  `sub` matching the `global_sso_id` cookie. A squad existed on that account.

**The invariant that matters: `FPL_REFRESH_TOKEN` and `FPL_TEAM_ID` must come
from the same account.** A token minted by one account against the other's team
id returns 403 on `/api/my-team/{id}/`, which presents as an expired-token
error and is easy to misdiagnose. If auth fails after a token refresh, check
this first: log in, hit `/api/me/`, and confirm the `entry` matches
`FPL_TEAM_ID`.

`FPL_TEAM_ID` was last set 2026-02-08 and has not been verified against the
current account.

## Running locally without breaking CI

Refresh tokens rotate on use. Any local run — including `--dry-run` — consumes
the stored token and receives a replacement, so a local run with only
`FPL_REFRESH_TOKEN` set will leave the GitHub secret holding a spent token and
CI locked out, while the local run itself reports success.

Export `GH_PAT` and `GITHUB_REPOSITORY` alongside it so the rotated token is
written back:

```bash
export FPL_TEAM_ID=5413589
export GITHUB_REPOSITORY=SiddheshA11/fpl-auto-manager
read -rs FPL_REFRESH_TOKEN && export FPL_REFRESH_TOKEN && echo
read -rs GH_PAT && export GH_PAT && echo
python3 manager.py --dry-run
```

`read -rs` keeps the values out of shell history and off the screen. If a run
logs `ROTATION FAILED`, the stored token is spent: re-pull one from
localStorage (`account.premierleague.com` session) and re-set the secret.

There is no `.env` in a fresh clone; it is gitignored.

## Open work, highest priority first

### 0. Four of five workflows are disabled — nothing runs until they are back on

GitHub auto-disables scheduled workflows after 60 days of repo inactivity, and
that happened over the summer:

```
FPL Auto Manager - Deadline Check    active
FPL Auto Manager - Scheduler         disabled_inactivity
Token Refresh - Automated Browser    disabled_inactivity
FPL Auto Manager - Token Reminder    disabled_inactivity
FPL Auto Manager - Weekly Run        disabled_inactivity
```

A fresh token does not fix this: the Weekly Run itself is off. Re-enable with
`gh workflow enable "<name>" --repo SiddheshA11/fpl-auto-manager`, but only
*after* item 1 below — the first live run would otherwise also be the first
real test of three unverified API assumptions, on a real team, at a deadline.

### 1. Verify three API-shape assumptions against a live authenticated response

These were flagged in review and never confirmed. Each is a deadline-time
submission failure if the guess is wrong. Needs a valid token.

- `chips.py:74-79` expects `my_team["chips"]` entries shaped `{"name", "event"}`.
  That is the `/entry/{id}/history/` format. The `/my-team/` endpoint may
  instead return `status_for_entry` / `played_by_entry`. If so, played chips are
  never detected, a spent wildcard looks available all window, and the engine
  re-submits it.
- `deadline_check.py:142` posts `{"chip": None}`. If the weekly run activated
  bboost or 3xc, re-posting with null may *deactivate* it — the my-team POST
  sets chip state rather than merging.
- `manager.py` + `fpl_client.py:521` zip id-sorted in/out lists into transfer
  pairs. Position multisets always match, but individual pairs can be
  DEF-out/MID-in. If FPL validates per-pair, multi-transfer submissions fail.

Settle all three with one authenticated `GET /my-team/{id}/` and record the
real response as a test fixture. The current tests construct the *assumed*
shape, so they prove the code agrees with itself, not with FPL.

### 2. Duplicate weekly runs

`deadline_scheduler.py:31` writes its dedup marker to `/tmp` on an ephemeral
Actions runner, so it never persists. Combined with the independent cron in
`weekly_run.yml`, the weekly run can fire up to five times per deadline. A
defensive `limit - made` guard was added in `manager.py`, but the root cause is
unfixed. Consider an idempotence check via `get_my_transfers` (exists, unused).

### 3. Minutes model

The known correlation gap. The model assigns non-zero xP to players who never
appear. `p_sub` in `xp_model.py` also uses the pre-normalisation start rate, so
a demoted backup keeper still carries ~0.25 sub-appearance probability.

### 4. Telegram reporting and the scheduled judgment layer

Never built. Owner chose **full auto with notification after the fact**: the bot
acts on its own and sends a writeup of what it did and why. The intended design
is two layers — the deterministic optimiser on the Actions cron, plus a
scheduled Claude routine before the deadline that reads press-conference news,
sanity-checks the proposed moves, and flags or approves. Weekly review should
cover rank delta, xP vs actual, captain hit rate, and which decisions cost
points.

### 4b. 2026/27 rule changes — two of them hit the model

From FPL's own "What's New" (fantasy.premierleague.com/en/help/new). Most of
the list is UI and does not concern us: real-time ranks, the autopick tutorial,
the new pitch/list views, and the Rookie League need no code. Two do.

**Rebalanced Bonus Points System — the important one.** FPL has changed the
BPS weights to give full-backs, goalkeepers and attacking players a better
chance of bonus. `xp_model.py` prices bonus by interpolating an empirical
curve (`BONUS_CURVE_BPS` / `BONUS_CURVE_PTS`) fitted to 673 player-seasons from
**2024-25 and 2025-26 — both under the old weights**. That makes it stale in
two compounding ways:

1. The curve maps bps-per-90 to bonus-per-90. If every full-back's bps rises,
   a full-back on 20 bps/90 is now less exceptional than the curve assumes, so
   bonus is over-predicted for exactly the positions that were buffed.
2. The `bps90` fed *into* the curve comes from priors built on old-weight
   seasons, so the input is stale as well as the mapping.

The curve is also position-agnostic, while this change is explicitly
positional. It should become per-position. It cannot be recalibrated until
2026/27 gameweeks accrue — budget a refit around GW8-10, and treat bonus as the
least trustworthy term in the model until then. Note the BPS *weights* are not
published in `game_config` (it carries `"bps": 0`, meaning bps awards no points
directly), so there is nothing to read live; only the resulting `bps` stat is
exposed, and recalibration is the only route.

**Prices now change daily at midnight UK time.** Two consequences:

- *Scheduling.* A run before midnight UK prices a squad that may be repriced
  before the deadline. The weekly run should fire after the daily price change,
  not before. Worth auditing the cron times in `.github/workflows/` against
  this, along with the duplicate-run problem in section 2.
- *An unused signal.* `price_change_percent` is present on all 590 elements and
  tracks how close a player is to a rise or fall — the field behind the new
  price-prediction page. It is a **string** ("0" pre-season, populates once
  transfer activity starts), so parse it. The optimiser ignores price movement
  entirely: it neither protects squad value by buying ahead of rises nor avoids
  players about to drop. Team value compounds over a season and is worth having.

Also newly visible and unused: `scout_news_link` (populated for 26 players with
club injury articles) and `scout_risks` (present, empty pre-season). Both are
candidates for the pre-deadline judgment layer in section 4 — an official
per-player injury link is a far better input than the scrapers that were
deleted.

Real-time ranks matter only for reporting: provisional bonus lands once a match
passes 20 minutes, so a live rank tracker is now feasible via
`event/{id}/live/` if the weekly review is ever extended.

### 5. Smaller model gaps

- Team strength has no shrinkage toward 1.0 and uses raw goals, not xG. One
  extreme season propagates straight into every clean-sheet term.
- Saves use `E[S]/3` rather than `E[floor(S/3)]`, overpricing keepers.
- Missing scoring terms: red cards, own goals, penalty saves.
- `backtest.py` uses season-end team and position per player, so mid-season
  transfers are misattributed in early gameweeks.

### 6. README is stale

Eight references to deleted modules and the scraper architecture.

## Design decisions worth not re-litigating

- Scoring values are read from live `game_config`, never hardcoded.
- Priors exclude the season under test in backtests, to avoid leakage.
- A stat absent from a season aggregates to NaN, never 0.0.
- Start rate is measured per gameweek registered, not per appearance, and
  shrinks on a gameweek scale.
- Start probability is normalised within each team to 11 shirts, allocated by
  `rate**alpha`, so a settled first choice is not diluted by his deputies.
- Poisson beat negative binomial for the defensive-contribution tail on
  held-out data; population overdispersion is between-player, not within.
- `data/history/` is gitignored and fetched in CI; `PriorSet.validate()` aborts
  the run rather than let empty priors silently produce a squad chosen by
  nothing.

## The open judgment call

The corrected model drops Haaland from the optimal squad — it rates him poor
value (5.48 xP at £15.5m vs Mbeumo 5.08 at £8.0m) rather than mis-rating him
(his implied per-90 is 93% of actual). But it has no concept of **template
risk**: Haaland is 71% owned, and pure xP maximisation is the right objective
for total points and the wrong one for *rank*. Nothing in the model represents
playing against other managers. Unresolved, and the owner's call.
