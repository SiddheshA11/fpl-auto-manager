# Handoff — FPL Auto Manager

Updated 2026-08-19. **Deployed**: `main` runs the expected-points model and
Weekly Run is enabled. 146 tests.

## State

| | |
|---|---|
| team | 5413589, GW1 squad submitted and verified against `/my-team/` |
| GW1 deadline | Fri 21 Aug 2026 17:30 UTC |
| Weekly Run | **active**, cron `0 10 * * 5` — Fri 10:00 UTC, 7.5h before the deadline |
| Scheduler | deliberately **disabled** — its dedup marker writes to `/tmp` on an ephemeral runner so it never persists, and alongside the cron it can fire five times per deadline |
| Tests | run in CI on the resolved dependency versions (`tests.yml`) |
| accuracy | MAE 0.97, RMSE 1.96, R² 0.319 on GW10-38 of 2025-26 |

Benchmark for context: MAE 1.29-1.42, RMSE 2.27-2.38, R² 0.29-0.35 (Valouxis,
NTUA 2023, n=6900, several compared models being paid products). **Those are
2022/23 numbers and three seasons stale** — treat as a floor, not a target. A
naive points-per-game baseline scores R² 0.257 on the same data, so the honest
read is that we are just inside a dated band, not at the frontier.

## The single most important fact about this model

Decomposing 22,774 player-gameweeks:

```
our model                                        R2 0.271
perfect appearance only (played at all, yes/no)     0.363
perfect minutes, scaling our xP                     0.440
actual minutes alone, no model of ability at all    0.491
```

**Minutes is not one lever among several. It is the lever.** The entire
goals / assists / clean-sheet / bonus / defensive-contribution apparatus is
worth 0.004 R² once minutes are known. Any future work that is not about
minutes should justify itself against that number.

Acting on it — folding recent minutes into the start rate — moved R² from
0.269 to 0.319, the largest single improvement made.

## Open work

### 1. Multi-gameweek transfer sequencing — the largest remaining gap

Transfers are one-step greedy: this week's move optimised against a 5-gameweek
horizon, with no plan for a *sequence*. Bank a transfer now to make a double
move next week, route toward a wildcard, take a hit early to catch a fixture
swing — none of it is modelled. This is what FPL Copilot claims as its
advantage, and chips already work this way (`chips.solve_assignment`); transfers
do not.

### 2. Rank-aware objective

The optimiser maximises expected points. The goal is winning a ~20-person
league, and those diverge: with a lead you want to track the field, trailing in
April you need variance the mean objective refuses to buy.

Everything needed exists and is wired to nothing: `league_analyzer.py` computes
rival ownership and **rival captain rates**, and is imported by no file. The
ownership tilt currently runs on `selected_by_percent`, the 11-million-player
template, when the field is 19 specific people. Captaincy is where this bites
hardest.

Blocked until a gameweek completes: `/entry/{id}/event/{gw}/picks/` returns
nothing before then.

### 3. Team value and price changes

`price_change_percent` is unused. This was measured as **worthless** at the old
-0.3 tilt, because the budget constraint was not binding — the model refused to
spend past £102.5m. At the current +0.20 it spends the full £100m, so it now
matters again. Caveat from the data: across two seasons there were 41-55 risers
against 450-524 fallers, and the average established player *loses* £0.11m, so
"build team value" is a thinner edge than folklore suggests.

### 4. Refit the bonus curve, per position, around GW8-10

`BONUS_CURVE_BPS`/`BONUS_CURVE_PTS` are fitted to old-BPS seasons.
`BONUS_POSITION_MULTIPLIER` corrects the positional error (measured: forwards
earn 1.65x a midfielder's bonus at matched BPS) but the curve itself is stale.
Note the 2026/27 BPS rebalance was measured as worth **under 1 point a season**
for defenders — much smaller than feared.

### 5. Variance is calibrated in level but not in shape

`sd_next` exists and is calibrated overall (ratio 1.000), but per-bucket ratios
run 0.82-1.32. Good enough to rank players by volatility; not yet good enough
to price a tail.

## Things that will bite you

- **`--ref` selects the code, not the workflow registration.** A workflow must
  exist on the default branch for `workflow_dispatch` to register, but the code
  that runs comes from the ref you dispatch. Dispatching against `main` when
  the fix was on a branch ran the old client, discarded a rotated refresh token
  and locked the account out. `dump_my_team.py` now refuses to start if the
  checkout cannot persist a rotated token.
- **Never chain `gh workflow disable` onto `gh workflow run`.** The run is only
  queued, so the disable lands first and the run sticks in `queued` with zero
  jobs, permanently. Re-enabling does not revive it. This silently swallowed a
  live submission.
- **Scheduled runs always use the default branch**, whatever you dispatch.
- **A refresh token is single-use.** It rotates on use; a run that authenticates
  and fails to persist the replacement leaves the secret spent. Set it with
  `gh secret set FPL_REFRESH_TOKEN` reading stdin, and strip the quotes — a
  value copied from DevTools carries them and PingOne answers "Failed to decode
  refresh token".
- **CI and local resolve different dependency versions.** pandas 3 gives text
  columns a dedicated string dtype, which silently falsified a
  `dtype == object` check, stripped the position off every player prior and
  halved the squad's expected points with all 113 tests green. Requirements now
  carry upper bounds and CI logs what it resolved.
- **Tests that exercise a helper do not prove the helper is called.** Three
  separate fixes could be deleted from production with the whole suite green.
  Prefer an end-to-end test that inspects what the client was actually handed.

## Verified API facts

Settled by live interaction, not documentation. Fixture: `tests/fixtures/my_team.json`.

- `/my-team/` chips carry `{name, id, number, chip_type, start_event,
  stop_event, status_for_entry, played_by_entry, is_pending}`. **There is no
  `event` key** — the old code read one, so played chips were never detected.
  `/entry/{id}/history/` really does use `{name, event}`; both are handled.
- Before the first deadline, `transfers` reports
  `{"status": "unlimited", "limit": null, "made": 0}`.
- **Transfer pairs must match by position.** FPL validates each pair and rejects
  the entire POST with `transfer_element_type_mismatch`. An id-sorted zip had 8
  of 12 pairs refused on a real submission.

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
