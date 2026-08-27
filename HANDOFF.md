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

## What to build next, in order

Ranked against the two measured facts that should govern everything: minutes
dominate accuracy, and the decision side has a noise floor that most changes
cannot clear. Anything not on this list should have to argue against these.

### Priority 1 — minutes, because the headroom is still enormous

The decomposition says our model scores R² 0.271 and *perfect minutes scaling
the same xP* scores 0.440. Our minutes model is at R² 0.609 after the
recent-minutes work. So closing the remaining minutes gap is worth up to
**+0.169 R²** — against 0.004 for the entire goals/assists/clean-sheet/bonus
apparatus. Nothing else on this page is in the same units.

Three concrete, unexercised leads, cheapest first:

1. **Yellow-card suspension risk — not modelled at all.** Five yellows is a
   ban, and `yellow_cards` is already aggregated in `priors.py` as a rate for
   the -1 point. Nobody converts a card count into a probability of missing
   the next match, which is a pure minutes effect and fully deterministic
   from data already on disk. Cheapest thing on this list by a distance.
2. **Manager change resets a start rate.** A new manager makes the previous
   twenty gameweeks of team selection much weaker evidence. Currently the
   start rate shrinks on a gameweek scale that knows nothing about this.
3. **Predicted lineups.** The single largest source, and what the paid
   services actually sell. Not in the FPL API, so it means an external feed
   and real fragility. Measure 1 and 2 before deciding this is worth the
   dependency.

Note what has already been *measured as worthless* and should not be retried:
rotation-from-congestion, team-effect-from-injuries.

### Priority 2 — the rank-aware objective, now unblocked

Different axis entirely: this is about winning a 20-person league, not about
accuracy, so the R² yardstick does not apply and neither does the noise floor
on total points.

`league_analyzer.py` computes rival ownership **and rival captain rates**, and
is imported by no file in the repo — verified again this session. `sd_next`
is computed on every production run and consumed by nothing. Those two plus
the existing tilt are the whole of a rank-aware captaincy rule, and captaincy
is the highest-leverage call of the week.

The tilt currently runs on `selected_by_percent` — the 11-million-player
template — when the field is 19 specific people. That proxy is known to be
wrong in a specific direction: squad ownership is not captaincy ownership.

Was blocked because `/entry/{id}/event/{gw}/picks/` returns nothing before a
gameweek completes. **After GW1 it is unblocked.**

### Priority 3 — measure the hand-picked optimiser constants

`simulate.py` now makes this possible for the first time. `FREE_TRANSFER_VALUE
= 0.3`, `DEFAULT_BENCH_WEIGHT = 0.15`, `HORIZON = 5` and `max_hits = 2` were
all set by argument, never measured.

Expect the answer to be "all inert". The horizon sweep already shows h=3 to
h=7 are indistinguishable, and that is the knob with the most obvious
mechanism. Run one batch, record the nulls, and stop touching them — a
measured null is worth more than a plausible story, and it closes the question
permanently.

**Use a control every time.** See the noise floor below.

### Priority 4 and below

Team value and price changes (now that the budget binds at +0.20), and the
bonus-curve refit. Both are small; the bonus curve is inside the 0.004.

## Open work

### 1. Multi-gameweek transfer sequencing — built, measured, DEAD

Both formulations exist on `feat/transfer-sequencing` and both work.
`sequence.plan_by_enumeration` (beam search over transfer-count schedules,
~4s) and `sequence.plan_jointly` (one MILP over the horizon, ~5s). They
demonstrably do the thing they were built for: over 2025-26 they hold 2+ free
transfers in 17 and 14 gameweeks against greedy's 6, and roll 11 and 10 times
against 7.

**Neither produces a measurable gain.** Measured over 4 seasons x 3 windows of
12 gameweeks = 12 paired samples, against greedy at horizon 5:

```
  config          mean      SE       t    won
  enumeration   -16.67   16.41   -1.02   6/12
  joint MILP     -3.25   13.61   -0.24   5/12
  ---- controls: changes that should mean nothing ----
  horizon 4      -1.25    9.71   -0.13   6/12
  horizon 6     +16.42    9.78   +1.68   8/12
```

Read the controls first. Changing the horizon from 5 to 6 is not a strategy
change, and it scores **+16.4 at t=+1.68** — a *larger* apparent effect than
either sequencer. That is the whole result: the apparatus manufactures effects
of this size out of nothing, and the sequencers do not clear it.

**Do not ship this, and do not "fix" it by tuning.** If you come back to it,
the honest framing is that the greedy objective's horizon-blended value column
is already doing the work — buying players who are good across five gameweeks
is a form of regularisation, and swapping it for a per-gameweek plan that
assumes you can correct later trades that robustness for a forecast that is
only R² ~0.15 at the horizon.

Do not read `horizon 6 +16.42` as a reason to retune `HORIZON`. It is noise,
and it is in the table specifically to demonstrate that.

### 1a. The noise floor — read this before measuring any optimiser change

`simulate.py` replays a season making real decisions and reports a points
total. It is the right tool. But it is chaotic: hold the strategy fixed at
greedy and vary **only the horizon**, and the 2025-26 season total moves
across a 141-point range.

```
  greedy h=1   1989   (4 hits taken — a myopic objective churns)
  greedy h=3   2114
  greedy h=4   2154
  greedy h=5   2062   <- production
  greedy h=6   2084
  greedy h=7   2013
```

SD across h=3..7 is about 53 points on a single season. With 4 seasons and 12
paired 12-gameweek windows the SE comes down to roughly 10-16 points per
window, which is still larger than the sequencers' effect.

Two earlier one-season designs contradicted each other outright, which is the
same fact seen twice — a single season cannot resolve this:

```
  8-GW windows, 8 independent      enumerate -32/season   joint -57/season
  staggered starts, all to GW38    enumerate +35/season   joint +36/season
```

**Always run a control.** A change that should not matter, measured the same
way, is the only thing that tells you what your apparatus's noise looks like.
Both control rows above exist for that reason and both earned their place.

### 1c. Availability was silently disabled for goalkeepers - FIXED

`_normalise_starts_within_team` divides a fixed number of shirts by relative
standing. That makes it purely relative, so folding availability into the
weights before allocating meant a dominant first choice barely moved when he
was flagged - his *share* of the group was unchanged. `rate**3.0` makes every
settled keeper dominant.

Measured on the committed snapshot, applying a 25% availability cut to each
first-choice keeper in turn:

```
  position   n   mean retained   median     min
  GK        11           0.243    0.070   0.000
  DEF       24           1.170    1.307   0.492
  MID       18           1.266    1.314   0.746
  FWD        2           1.280    1.280   1.235
```

**Five of eleven retained exactly 0.000**: Pickford, Leno, Verbruggen,
Henderson and Sels could not be marked down at all. This was never a card-ban
problem - it disabled *every* availability signal those players could receive:
injury flags, `chance_of_playing_next_round`, news decay, the lot. Outfielders
had the opposite, milder fault, over-applying a cut at 1.17-1.28.

Fix: allocate shirts on ability, scale each player by his own availability,
hand the freed shirts to team-mates who can play. After: GK 0.974, everyone
else 0.994-0.996.

**Two obvious shortcuts are wrong and both were tried here first.** Sharing the
freed mass by remaining *headroom* favours the deputy, since he is furthest
below his ceiling - a first choice whose understudy carried a 75% flag finished
worse off than if the understudy were fit (United's keeper fell 0.902 to 0.807
for no reason). Sharing it by *standing* hands it straight back to the player
who released it, so the flag does nothing again. It has to go to the others,
weighted by their availability-weighted standing.

Impact on the live snapshot: 316 of 590 players move, mean |change| 0.020 xP
over the horizon, max 0.678, and the **top 30 by horizon xP are unchanged**.
Doubtful outfielders correctly rise (the old code over-penalised them);
flagged first-choice keepers correctly fall.

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

## Where the minutes error actually is - re-measured on the real model

`measure_minutes_decomp.py`. The first version of this ran before the harness
fix below and was therefore on the crippled model; these are the numbers that
count. Both columns, same run, so the effect of the lags is visible directly:

```
                          without lags        with lags (production)
  minutes R2                  0.4742                     0.6085
  segment              n   share err   bias      share err   bias
  never plays      10621       0.043   +1.9          0.047   +1.4
  fringe <30        3915       0.253   -6.8          0.265   -7.1
  rotation 30-60    3976       0.387   -2.9          0.373   -4.1
  regular 60-80     2504       0.223   +1.1          0.215   -0.4
  ever-present 80+  1758       0.094   -3.5          0.100   -1.8
```

**0.6085 matches the 0.609 this document has always quoted, to three decimals.**
That figure was never unreproducible - it just needed the lags the harness was
not passing. An earlier session note in this file claiming otherwise was wrong.

Two things to carry:

**The composition survives.** Every segment's share of squared error moves by
0.03 or less. Rotation and fringe still carry ~64% of it between them, and
that is still not where a squad's players live. The ranking of what to work on
does not change.

**The addressable headroom is smaller than it looked.** Established regulars
who blank fall from 16.6% of all squared error to **13.9%**, and the model now
prices them at **49.3 expected minutes rather than 69.9** - the lag view
already catches much of a player losing his place. Rescaling the split in 1d by
0.139/0.166 gives roughly 7.9% injury absence, 4.6% unexplained rotation, 0.7%
card suspensions.

And the headline gap needs restating. Priority 1 sizes the prize as
`0.440 - 0.271 = +0.169` points R2. Our points R2 is now measured at **0.3093**,
so the gap to perfect minutes is **0.131**, not 0.169 - and the 0.440 itself was
measured on the crippled model and is due a re-run before anyone leans on it.

## The offline harnesses were scoring the wrong model - FIXED

`recent_minutes` was supplied by `manager.py` and nothing else. `backtest.py`
and `simulate.py` both built `XPModel` without it, so `_blend_recent_minutes`
returned at its first line and the start rate was never touched. **Every
offline measurement in this repo evaluated a model missing the input this
document credits with the largest single improvement ever made.**

Measured on 2025-26 GW10-38, same priors both sides:

```
                          MAE     RMSE       R2
  without recent_minutes  1.0615  2.0494   0.2570
  with recent_minutes     0.9791  1.9757   0.3093
                                          +0.052
```

The claim in the state table - "moved R2 from 0.269 to 0.319", a gain of 0.050
- reproduces at **+0.052**. Absolute levels sit ~0.01 lower because local disk
builds priors from six usable seasons; the *gain* is the reproducible part. So
the figure was always real, and `backtest.py` was understating the model by
0.05 R2 while reporting it.

Consequences worth carrying forward:

- Any measurement taken before this landed was on the crippled model. That
  includes the minutes decomposition in 1d - its **level** is wrong, though the
  compositional split should survive. Re-run before quoting the number.
- A regression in `_blend_recent_minutes` was invisible to every offline tool
  in the repo. `tests/test_offline_recent_minutes.py` now records what
  `run_backtest` actually hands the model, through a real call.
- The optimiser-constant sweeps in Priority 3 were measured on a model missing
  its dominant input and are worth nothing until re-run.

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
  A fourth was found this way: `recommend.py` passed no `ownership_weight`, so
  the tool used to preview a plan before a deadline optimised at 0.0 while the
  weekly run used +0.20 — different squad, no Haaland, 0.56 xP short.
  `deadline_check.py` had the identical bug and carries a comment about it.
  Now fixed, with a test that records the optimiser's constructor kwargs
  through a full `recommend.main()` run.
- **A season total is not a measurement.** See "the noise floor" above.

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
