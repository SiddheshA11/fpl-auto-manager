# Handoff — FPL Auto Manager rebuild

Written 2026-08-17, updated the same day. Branch `rebuild/xp-model`, 20 commits,
**pushed to origin**, working tree clean, 65 tests passing.

## Read this first: GW1 deadline is Fri 21 Aug 2026, 17:30 UTC

Transfers are **unlimited and free until that deadline**, which resolves the
wildcard question below: the ~21 xP rebuild is free right now, so take it and
keep both wildcards. Wildcard 1 covers GW2-19, wildcard 2 GW20-38.

No player's price has moved from its starting value (0 of 590), so the budget
is exactly £100.0m regardless of what is currently owned.

### Decisions taken

- **Ownership**: added as a signed tilt, `OWNERSHIP_WEIGHT` in `config.py`,
  set to **-0.3** (differentials) because the target is a private league of
  ~20. Positive would track the template and suit overall rank instead. The
  derivation is in the `OWNERSHIP_WEIGHT` comment in `optimizer.py`; note that
  a *mean* differential term is algebraically inert, so this acts on variance.
- **Wildcard**: not played. Superseded by the free pre-deadline rebuild.

### Four bugs found and fixed since the original handoff

1. **Pre-season transfers.** `manager.py` read `(limit or 1) - made`, and FPL
   publishes no numeric limit while transfers are unlimited, so the bot saw one
   free transfer during the only week it had fifteen. Measured on the snapshot:
   3 transfers and 2 hits (-8 pts) where the whole squad was free to replace.
2. **Stale pre-season totals.** Before a ball is kicked the bootstrap still
   carries *last* season's totals - 400 players with up to 3420 minutes, zero
   gameweeks finished. These were blended in as current-season evidence at 86%
   weight for an ever-present, double-counting a season the priors already hold
   and bypassing the shrinkage in `priors.py`. This was the largest of the four.
3. **XI chosen on the horizon.** Squad membership is a horizon decision; the
   starting eleven is a one-week decision. The joint solve used the horizon
   column for both, benching players worth more *this* gameweek. +0.5 xP per
   gameweek, ~19 over a season.
4. **Defensive contribution across club moves.** DC is as much a club property
   as a player one; the 2025-26 spread runs 0.86-1.13 relative to the league
   mean. Transferred players carried their old club's volume into their new
   club's style, and because DC is a *threshold* award a rate just above the
   line is fragile. Press coverage of Anderson's move to City implies possession
   guts his returns; the measured effect is 10%, not 50%.

Also removed the ownership tilt from the captaincy decision - captaincy
effective ownership is not squad ownership, FPL does not publish it, and the
proxy had started captaining a 5.12 xP defender over a 5.49 xP midfielder.

## What this project is

Fully automated Fantasy Premier League manager. Runs on GitHub Actions, picks
transfers, lineup, captain, bench order and chips, and submits them without
intervention. The owner wants to win his league and does not want to check in
weekly. He has chosen **full auto with notification after the fact**: the bot
acts on its own and reports what it did and why.

## Status: working, tested against the live account, not yet deployed

Verified on 2026-08-17 by dispatching the weekly run in dry-run mode against
the real account (run `32073377269`, 47s, passed):

- Authentication succeeds; the refresh token and team id pair correctly.
- **Token rotation succeeds** — the run consumed a token, received a
  replacement and wrote it back to the GitHub secret. This was silently broken
  for the whole of last season and is why the old bot died.
- The full pipeline runs in CI: priors → xP model → MILP optimiser → chip
  valuation → lineup construction.
- Chip logic correctly declined bench boost (7.8 xP against a 12.0 threshold)
  and correctly treated wildcard and free hit as unavailable in GW1.

**Nothing is deployed.** `main` is still the old heuristic bot. GitHub Actions
runs `main`, so merging the branch is what makes any of this live.

## Account and secrets — all current

| secret | state |
|---|---|
| `FPL_REFRESH_TOKEN` | rotates automatically every run; do not set by hand unless rotation breaks |
| `FPL_TEAM_ID` | `5413589` — "Siddhesh's Team", verified against the live API |
| `GH_PAT` | required for rotation; set 2026-02-09, **check it has not expired** |
| `FPL_EMAIL` / `FPL_PASSWORD` | only used by `token_refresh.py`, which is disabled |

`FPL_COOKIE` was deleted — it was four months stale and `login()` tried it
first, wasting a failed attempt every run.

Two Premier League accounts exist and caused confusion during setup. Premier
League mail lands in `siddheshagarwal10@gmail.com`; a browser session captured
the same day carried an id_token for `siddheshrox123@gmail.com`. This is now
moot — team `5413589` authenticates successfully with the stored token, which
is the only thing that matters. **The invariant to remember:**
`FPL_REFRESH_TOKEN` and `FPL_TEAM_ID` must belong to the same account, because
a mismatch returns 403 and reads exactly like an expired token.

## Workflow state

```
Deadline Check    active              (workflow_dispatch only — no schedule)
Weekly Run        disabled_manually   ← enable to go live
Scheduler         disabled_inactivity
Token Refresh     disabled_manually   ← leave off, see below
Token Reminder    disabled_inactivity
```

GitHub auto-disabled these after 60 days of repo inactivity over the summer.
`Weekly Run` was briefly enabled on 2026-08-17 to run the dry-run test, then
disabled again.

## What was rebuilt

The heuristic core was replaced. It scored players on a unitless 0-1 weighted
composite, then multiplied by 10 and compared that against a -4 transfer hit —
so every threshold in the system was arbitrary.

| file | role |
|---|---|
| `priors.py` | Per-90 priors with position-aware shrinkage; team strength; promoted-club priors |
| `xp_model.py` | Expected points in real FPL points. Minutes model, threshold-based defensive contribution, empirical bonus curve, scoring read live from `game_config` |
| `optimizer.py` | MILP over squad + XI + captain jointly, via scipy HiGHS. Hits priced as an actual -4 |
| `chips.py` | Chip availability across both half-season windows; chip value in expected points; DGW/BGW detection |
| `manager.py` | Weekly run orchestrator (unattended, submits real transfers) |
| `deadline_check.py` | Pre-deadline safety: re-reads availability, repairs lineup only, makes no transfers |
| `backtest.py` | Replays a past season with priors restricted to strictly earlier seasons |
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

Two caveats on those numbers:

- Per-gameweek standard error on top-20 is ~0.18, so differences below that
  are not resolvable with one season.
- **The backtest ran on 2025-26, under the old Bonus Points System.** FPL
  rebalanced BPS for 26/27, so the measured edge assumes it transfers across a
  rules change, and partly it will not. See item 4 below.

## Open work, highest priority first

### 1. Verify three API payload shapes, then merge

**In progress.** `dump_my_team.py` plus `.github/workflows/dump_my_team.yml`
capture a real authenticated `/my-team/` alongside `/entry/{id}/history/`
chips, redacted, as a committed fixture. Read-only; it issues no POST.

It runs in CI rather than locally because PingOne rotates the refresh token on
use, so a local run spends the stored token and the replacement never reaches
the GitHub secret. **Blocked on the workflow file reaching `main`** - GitHub
only registers `workflow_dispatch` for workflows on the default branch. The
commit exists locally on `main`; it needs pushing.

Still unverified. The dry run proved the code does not crash, but each of these
needs a *live* interaction to settle, and each is a deadline-time submission
failure if the guess is wrong.

- `chips.py:74-79` expects `my_team["chips"]` entries shaped `{"name", "event"}`
  — the `/entry/{id}/history/` format. `/my-team/` may instead return
  `status_for_entry` / `played_by_entry`. Nothing has been played yet, so both
  shapes look identical (empty) and the dry run could not distinguish them. If
  wrong, played chips are never detected and the engine re-submits a spent one.
- `deadline_check.py` posts `{"chip": None}`. If the weekly run activated
  bboost or 3xc, re-posting null may *deactivate* it — the my-team POST sets
  chip state rather than merging.
- `manager.py` + `fpl_client.py:521` zip id-sorted in/out lists into transfer
  pairs. Position multisets always match, but individual pairs can be
  DEF-out/MID-in. If FPL validates per-pair, multi-transfer submissions fail.

Record a real authenticated `/my-team/` response as a test fixture. The current
tests construct the *assumed* shape, so they prove the code agrees with itself,
not with FPL.

Then merge `rebuild/xp-model` to `main` and enable Weekly Run. Nothing is live
until that merge.

### 2. Duplicate weekly runs

`deadline_scheduler.py:31` writes its dedup marker to `/tmp` on an ephemeral
Actions runner, so it never persists. Combined with the independent cron in
`weekly_run.yml`, the weekly run can fire up to five times per deadline. A
defensive `limit - made` guard exists in `manager.py`, but the root cause is
unfixed. Consider an idempotence check via `get_my_transfers` (exists, unused).

### 3. Telegram reporting and the scheduled judgment layer

Never built, and the owner's main outstanding ask. Two layers: the
deterministic optimiser on the Actions cron, plus a scheduled Claude routine
before the deadline that reads press-conference news, sanity-checks the
proposed moves, and flags or approves. Weekly review should cover rank delta,
xP vs actual, captain hit rate, and which decisions cost points.

Newly available inputs that beat the deleted scrapers: `scout_news_link`
(official club injury articles, populated for ~26 players) and `scout_risks`,
both on the bootstrap element.

### 4. 2026/27 rule changes

**Rebalanced BPS — the one that costs points.** FPL changed BPS weights to
favour full-backs, goalkeepers and attackers. `xp_model.py` interpolates a
curve (`BONUS_CURVE_BPS` / `BONUS_CURVE_PTS`) fitted to 673 player-seasons from
2024-25 and 2025-26, both old-weights. Stale twice over: the mapping
over-predicts bonus for exactly the buffed positions, and the `bps90` fed into
it also comes from old-weight priors. It is also position-agnostic while the
change is positional. BPS weights are not published in `game_config`, so
recalibration is the only route — **refit around GW8-10** and treat bonus as
the least trustworthy term until then.

**Daily midnight price changes.** Audit the cron times so the weekly run fires
*after* the daily price change, not before. Also `price_change_percent` is on
every element (a **string**, "0" pre-season) and tracks proximity to a rise or
fall. The optimiser ignores price movement entirely — it neither buys ahead of
rises nor avoids falls, and team value compounds over a season.

Real-time ranks and provisional bonus matter only for reporting: a live rank
tracker is feasible via `event/{id}/live/`.

### 5. Minutes model

The known all-players correlation gap. The model assigns non-zero xP to players
who never appear. `p_sub` in `xp_model.py` also uses the pre-normalisation
start rate, so a demoted backup keeper still carries ~0.25 sub-appearance
probability.

### 6. Smaller model gaps

- Team strength has no shrinkage toward 1.0 and uses raw goals, not xG. One
  extreme season propagates into every clean-sheet term.
- Saves use `E[S]/3` rather than `E[floor(S/3)]`, overpricing keepers.
- Missing scoring terms: red cards, own goals, penalty saves.
- `backtest.py` uses season-end team and position per player, so mid-season
  transfers are misattributed in early gameweeks.

### 7. README is stale

Eight references to deleted modules and the scraper architecture.

## Strategic note for the owner

At GW1 the current squad's XI scores **112.4 xP** over five gameweeks; an
unconstrained rebuild scores **133.8**. A ~21 xP gap is large. Wildcard unlocks
at GW2 — worth considering early rather than banking it.

Also unresolved: the optimiser maximises expected points, which is the right
objective for total score and the wrong one for **rank**. It has no concept of
template risk — it will happily leave a 70%-owned premium out of the squad,
which is correct for points and dangerous for league position. Worth deciding
whether to add an ownership term.

## Running locally without breaking CI

Refresh tokens rotate on use. Any local run — including `--dry-run` — consumes
the stored token and receives a replacement, so a local run with only
`FPL_REFRESH_TOKEN` set leaves the GitHub secret spent and CI locked out, while
the local run reports success.

```bash
export FPL_TEAM_ID=5413589
export GITHUB_REPOSITORY=SiddheshA11/fpl-auto-manager
read -rs FPL_REFRESH_TOKEN && export FPL_REFRESH_TOKEN && echo
read -rs GH_PAT && export GH_PAT && echo
python3 manager.py --dry-run
```

`read -rs` keeps values out of shell history. If a run logs `ROTATION FAILED`,
the stored token is spent: pull a fresh one from localStorage on an
`account.premierleague.com` session and re-set the secret.

There is no `.env` in a fresh clone; it is gitignored. Easier alternative to
all of the above: dispatch the workflow in dry-run mode, which uses the repo
secrets and rotates correctly:

```bash
gh workflow enable 231935805 --repo SiddheshA11/fpl-auto-manager
gh workflow run 231935805 --repo SiddheshA11/fpl-auto-manager --ref rebuild/xp-model -f dry_run=true
# Wait for it to FINISH before disabling. `gh workflow run` only queues the
# dispatch and returns immediately, so chaining the disable onto it with && kills
# the run before GitHub schedules a job for it: the run sticks in `queued` with
# zero jobs, forever, and re-enabling afterwards does not revive it. That has
# already silently swallowed one live submission.
RID=$(gh run list --repo SiddheshA11/fpl-auto-manager --workflow 231935805 \
      --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RID" --repo SiddheshA11/fpl-auto-manager
gh workflow disable 231935805 --repo SiddheshA11/fpl-auto-manager
```

`--ref rebuild/xp-model` is not optional. The workflow file must exist on the
default branch for GitHub to register `workflow_dispatch`, but the *code* that
runs comes from the ref you dispatch - so omitting it runs `main`, which is
still the old heuristic bot. The same trap spent a refresh token: dispatching
against `main` ran a client that reads only `GITHUB_TOKEN`, so the rotated
token was discarded and the stored secret left dead.

**The scheduled cron always runs the default branch**, whatever you dispatch.
Weekly Run's cron is `0 10 * * 5` - Friday 10:00 UTC - so leaving it enabled
before the merge means `main`'s old bot overwrites the squad on Friday morning.
Disable it again as soon as a manual run finishes.

## Design decisions worth not re-litigating

- Scoring values are read from live `game_config`, never hardcoded.
- Priors exclude the season under test in backtests, to avoid leakage.
- A stat absent from a season aggregates to NaN, never 0.0.
- Start rate is measured per gameweek registered, not per appearance, and
  shrinks on a gameweek scale.
- Start probability is normalised within each team to 11 shirts, allocated by
  `rate**alpha`, so a settled first choice is not diluted by his deputies.
- A player's per-90 rate already embeds his own club's quality; the fixture
  adjustment carries only opponent and venue.
- The defensive-contribution tail is evaluated per minutes-outcome and mixed,
  not once at mean minutes (the tail is convex).
- Poisson beat negative binomial for that tail on held-out data; population
  overdispersion is between-player, not within.
- `data/history/` is gitignored and fetched in CI; `PriorSet.validate()` aborts
  the run rather than let empty priors produce a squad chosen by nothing.
- `token_refresh.py` (Playwright password login) is disabled and should
  probably be deleted once rotation has held for a few weeks. It broke on a
  page redesign, needs the password stored, and rotation makes it redundant.
