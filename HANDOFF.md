# Handoff — FPL Auto Manager

Updated 2026-10-20. **Deployed**, season in progress, `main` green.

## State

| | |
|---|---|
| team | 5413589 |
| season to date | **469 pts vs a 416 average (+53)** through GW7, final. Overall rank 1,490,787. No hits taken all season. |
| per gameweek | GW1 53 (+3) / GW2 105 (+24) / GW3 49 (-2) / GW4 79 (+10) / GW5 43 (-5) / GW6 69 (+11) / GW7 71 (+12). |
| mini leagues | 5 leagues, 94 rivals. Mid-table everywhere. **Read "the decline diagnosis was wrong" below before concluding anything from that** - combined hindsight leak is the 53rd percentile of this field and captaincy the 24th. |
| deadlines missed | **GW2** (28 Aug, scheduled run never fired) and **GW6** (10 Oct, a run arrived with 1.94h in hand and was rejected by a 2.0h floor, 3.6 minutes short). Both squads rolled over. Nothing missed since the coverage fix, though GW7 did not exercise it - its single qualifying tick would have passed the old band too. |
| bench points | 74 through GW7, ~10.6/GW. Field comparison through GW7 (`measure_field_leaks.py --through 7`, 93 rivals): lineup leak 56 vs field 40.7 (75th pctile, p90 70), captaincy 32 vs 38.5 (33rd - better than the field), combined 69th, points 469 vs 445.5 (68th). About 10 of the lineup leak's GW6-7 rise is GW6, the missed deadline's stale lineup; GW7's 9 is mostly Gomez benched at p_start 0.98 for 8 points. |
| Weekly Run | cron **`0 * * * *`** (hourly), acting band **0.5-8h** before the deadline: 98.9-99.5% coverage, conditional on the cron firing (`measure_deadline_coverage.py`). `manager.py --respect-window` decides, and the `FPL_LAST_SUBMITTED_GW` marker makes repeats no-ops; an unreadable marker now refuses rather than proceeds. Cancellable with `gh run cancel`. |
| Deadline alerting | **LIVE.** `watchdog.yml` hourly; Telegram verified end to end 17 Oct. Alerts on an unsubmitted deadline inside 6h (and dispatches a run), a gameweek missed outright, a submitted team whose marker write failed, its own job failing, and an all-clear every 72h so silence means something. |
| Scheduler | `scheduler.yml`, `token_refresh.yml` and their scripts deleted 18 Oct. `deadline_check.yml` is dispatch-only and unreachable - a real gap, since nothing re-reads availability after Friday press conferences. |
| Tests | 462 in CI (4 data-conditional skips), green on main. CI's two passes take ~20 of the 30-minute limit. |
| accuracy | MAE 0.979, RMSE 1.976, **R2 0.3093** on GW10-38 of 2025-26, reproducible via `python backtest.py --season 2025-26` |

Benchmark for context: MAE 1.29-1.42, RMSE 2.27-2.38, R2 0.29-0.35 (Valouxis,
NTUA 2023, n=6900, several compared models being paid products). Those are
2022/23 numbers and three seasons stale - treat as a floor, not a target. A
naive points-per-game baseline scores R2 0.257 on the same data.

## THE ALERTING GAP - CLOSED 17 Oct 2026

Kept for the history, because the shape of the failure is worth remembering.

On 28 Aug the GW2 deadline passed with **no run at all**, and three layers
failed independently: the scheduled run did not fire, nothing checked that it
had, and nothing could have reported it - no workflow and no Python file
referenced Telegram despite the secrets being listed as required.

All three are now fixed and each was verified against the real thing rather
than inferred:

- **The run fires.** Hourly cron on a 0.5-8h acting band, chosen by replaying
  the real arrival stream (GitHub throttles this repo to ~5.5 arrivals/day,
  median gap 4.25h). The old 2-26h/daily combination admitted exactly one
  qualifying tick per gameweek for all 38, and GW6 died on a run that arrived
  with 1.94h in hand against an unjustified 2.0h floor - 3.6 minutes short.
- **Something checks the outcome.** `watchdog.yml`, hourly, on public endpoints
  and stdlib only, asking whether a team was submitted rather than whether a
  cron fired. It also looks BACKWARDS: `next_event` only returns future
  deadlines, so a missed gameweek was invisible and the watchdog reported
  `waiting` ninety minutes after GW6 was dropped.
- **It can tell you.** Telegram wired and verified by an actual delivered
  message on 17 Oct, not by reading logs - Telegram returns 200 for a
  well-formed request to a chat it cannot deliver to, so the logs cannot
  distinguish success from a wrong chat id.

Four things this cost that are worth not repeating:

1. Every one of those defects was live with a green test suite.
2. Two were found only because something independent re-checked a claim that
   had already been "validated" - and in two cases the validation was circular,
   comparing new code against earlier arithmetic that shared the same bug.
3. The ledger's first write 403ed on a PAT missing `Contents: write`, which
   logs one warning and exits 0. A self-test caught it; nothing else would
   have, for a season.
4. `calibrate.py` scored an unplayed gameweek and reported seven starters as
   p_start failures. Confidently formatted, silently wrong, in the one lever
   this repo already believes dominates - so it would have been believed.

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

**Read "Open after the 17 Oct audits" below first.** Item 1 there - the
decision-replay harness - is now the precondition for Priority 3 in this
section and for every "measured null" claim in this file. Priority 2 below is
DONE: the rank-aware objective shipped (field ownership, captain tilt
machinery), though the field-vs-template question is unresolved rather than
settled and the captain weight is deliberately 0.0.


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
2. ~~**Manager change resets a start rate.**~~ **MEASURED, DEAD.** See
   `measure_regime.py` and "the manager-change null" below.
1. ~~**Yellow-card suspension risk.**~~ **BUILT.** `_suspension_risk` in
   `xp_model.py`, `measure_suspension.py`, `tests/test_suspension.py`. Read
   "what the suspension model is and is not" below before extending it - the
   naive version of this feature is a no-op, and the obvious way to measure it
   reports a gain that is not real.
2. **Manager change resets a start rate.** A new manager makes the previous
   twenty gameweeks of team selection much weaker evidence. Currently the
   start rate shrinks on a gameweek scale that knows nothing about this.
3. **Predicted lineups.** The single largest source, and what the paid
   services actually sell. Not in the FPL API, so it means an external feed
   and real fragility. Measure 1 and 2 before deciding this is worth the
   dependency.

Note what has already been *measured as worthless* and should not be retried:
rotation-from-congestion, team-effect-from-injuries, manager-change start-rate
reweighting.

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

## Open after the 17 Oct audits - ranked, with what is blocked by what

Two independent audits (engineering, statistics) ran over the 12-17 Oct work.
The four with a live failure mode are fixed and merged. These are the rest,
and the ordering is not arbitrary: **item 1 is the precondition for every
accuracy question in this file.**

**18 Oct: Phase A is done** (#31-#35). Items 2 and 4 are closed, and every
item in 5 is fixed except the three marked open. Item 1 (the harness) is next;
item 3 still needs a decision. What changed in the reading of each is noted
inline - two of the audit's own claims did not survive measurement.

### 1. The decision-replay harness has no power. Fix this first.

**18 Oct: built and measured (#37, #38, follow-up). It has more power than
`simulate.py`, and still not enough for the open questions. Phase C stays
blocked on a decision.**

`replay_harness.py` forks every arm from the production trajectory's exact
state at every start gameweek: 144 windows of 8 gameweeks over the five
replayable seasons (2021-22 to 2025-26; earlier seasons lack the `team`
column the prior build needs, so the plan's "seven seasons" was wrong).
`replay_stats.py` holds the analysis rules, committed before any candidate's
points were read. What is validated, each against something independent:

- `pick_xi` scores 514 real 2026-27 gameweeks (mine and rivals', anonymised)
  exactly as FPL did, substitutions included - after undoing FPL's picks
  endpoint, which returns lineups *post*-autosub. Found on the way: the
  armband went to the best starter after autosubs; fixed (2 of 185 replayed
  gameweeks change, -24 points total).
- Template ownership is reconstructed exactly from `selected` (sum / 15 =
  squads) and matches FPL's own `selected_by_percent` to 0.03-0.05pp mean,
  so the baseline replays production's 0.2 tilt. Field ownership and the
  captain tilt cannot be replayed: no historical rival data.
- Determinism: an identity arm and alpha +/- 1e-6 give exactly 0 in all 432
  windows. Those make no decision flip, so they are not noise placebos. The
  placebo is a seed pair of N(0, 0.3) xP noise - expected difference exactly
  zero, and it flips decisions as often as alpha 1.0 does (alpha 1.0 changes
  decisions in 6.7 of 8 gameweeks of 98.6% of windows; it is not a near-tie
  nudge).
- The SE is season-clustered (t4). The proposed within-season block
  bootstrap measured 0.21-0.66 of the true SE on synthetic windows and was
  dropped before use.

**The result, measured by injecting known effects into 20 null replicates:**

```
  effect      per season   power
  1.0 pts/GW       38       20%
  1.5 pts/GW       57       55%
  2.0 pts/GW       76       70%
  2.5 pts/GW       95       85%
```

80% power is ~85-90 points a season, against ~135 for four full seasons. The
144 windows are worth ~33 independent ones (design effect 4.4), and the
binding constraint is G=5 seasons: inference is t4, and one season carries
60% of the variance on average. Terminal valuation inflated placebo SD by 29%
and was rejected by the pre-registered 25% rule; the outcome is plain window
points. L=6 and L=12 give the same MDE per gameweek (1.9-2.0 pts/GW).

**Phase C, 18 Oct - run under `measure_phase_c.py`, rules committed first.**
Option 2 below was taken: the harness as a veto, accuracy for what accuracy
can see.

```
  accuracy vs alpha 1.5, GW10-38, 5 seasons, season-clustered t4
  alpha   R2/GW    t      p     GW ahead  seasons  naive t   harness pts/season
  0.75   -0.0037  -0.98  0.38    60/145     2/5     -2.9       -43 [ns]
  1.0    +0.0034  +1.78  0.15   105/145     3/5     +5.2       -58 [p~0.05, VETO]
  1.25   +0.0025  +4.04  0.016  121/145     5/5    +10.0       -13 [ns]
```

- **`OUTFIELD_COMPETITION_ALPHA` is now 1.25.** The only value passing all
  four rules: better R2 in every season, MAE better, no harness harm. GW8's
  transfers, XI, captain and vice are identical under 1.25 and 1.5.
- **"alpha 1.0 at t=7.9" did not survive.** That was a naive pooled t over
  gameweeks; clustered by season 1.0 wins 3 of 5 (p=0.15) and the harness
  puts it at -58 pts/season.
- **Every optimiser constant stays.** FREE_TRANSFER_VALUE 0.0/0.6,
  DEFAULT_BENCH_WEIGHT 0.05/0.3, HORIZON 3/4/6/7: none better; every point
  estimate but bench 0.05 (+4, ns) is negative. Measured within ~90 points,
  not measured to be zero.
- **The template tilt at 0.2 scores ~+62 pts/season against no tilt in
  replay** (CI +15 to +109, p=0.022, uncorrected). Probably not transferable:
  the replay model is blind to injury flags (`build_state` sets every player
  fit) and ownership carries exactly that information. No change; the field
  tilt and the rank purpose remain unmeasurable.

So the harness can veto a change that costs ~90+ points a season and cannot
resolve anything smaller.

**Option 1 done, same day: nine seasons, MDE ~54 pts/season.**
`augment_history.py` rebuilt the `team` column (and teams/fixtures for
2016-18; 2019-20's restart renumbered onto 30-38), validated by reproducing
the real columns on the six seasons that have them - 100% of rows and
fixtures. The harness now replays 2017-18 to 2025-26: 260 windows,
effectively 63 independent (was 33), largest season's variance share 0.37
(was 0.60), and the t-test is now properly calibrated (placebo t sd 1.17
against 1.15 for t8; with five seasons it was 1.12 against 1.41).

```
  injected effect   per season   power (9 seasons)   power (5 seasons)
  0.5 pts/GW            19          20%                  0%
  1.0 pts/GW            38          50%                 20%
  1.5 pts/GW            57          85%                 55%
  2.0 pts/GW            76         100%                 70%
```

Phase C re-run on nine seasons as a sensitivity check (decisions stand as
made; nothing was re-decided):

- alpha 1.25 vs 1.5 on accuracy: ahead in **9 of 9 seasons**, 205/261
  gameweeks, clustered t=4.45 (p=0.002), MAE better (p=0.005). Confirmed.
- Harness vs production 1.25: reverting to 1.5 is +7.8 pts/season (ns);
  alpha 1.0 is -10.6 (ns) - **the five-season "veto" of 1.0 (-58, p~0.05)
  did not replicate**, so part of it was noise. 1.0 still fails accuracy (5
  of 9 seasons, p=0.22).
- **FREE_TRANSFER_VALUE is measured inert**, not merely unmeasured: at 0.0 it
  changes a decision in 1.9% of windows, effect CI +/-3 pts/season; 0.6 is
  +/-8. Bench weight 0.05/0.3: +/-15. HORIZON 4/6/7 within noise; 3 is -24
  pts/season (p=0.06). Tilt off: -24 (p=0.11), down from -62 at five seasons.

Two things this exposed that are decisions, not defaults:

- **Production builds priors from three seasons** (`fetch_data.HISTORY_SEASONS
  = 3`). Adding 2016-20 to the priors moved 2025-26 GW10-38 R2 from 0.3093 to
  0.3161 at alpha 1.5 - larger than the alpha change itself. Shipping that
  means fetching and augmenting older seasons in CI.
- **The harness's priors are "every earlier season"**, not production's
  three. Kept that way so adding seasons changed one thing at a time; a
  production-mirroring window would shift every harness number. The alpha 1.0 point estimate from the old harness
(-40/season) is inside its noise. **Do not run Phase C on it as if it could
settle the constants.** Options, which are a decision, not a default:

1. Make 2016-17 to 2019-20 replayable by reconstructing `team` per
   player-gameweek (and teams/fixtures for 2016-18). Projected ~50 pts/season
   MDE with nine seasons. Pre-2019 has no xG, so those priors are weaker.
2. Accept the harness as a veto for large harms only, and let accuracy
   (which does have power: alpha 1.0 t=7.9 over 50/58 gameweeks) decide
   anything the decision side cannot see.
3. Variance reduction with the model's own xP as a control variate. Cheap,
   but biased for any change to the model itself (alpha), so only valid for
   optimiser-only constants.

Original note:

`simulate.py` is the only honest judge of an optimiser change, and as used it
cannot resolve anything:

- **n = 4 seasons.** The two-sided sign test therefore floors at **p = 0.125** -
  no outcome could reject at 5%. MDE at 80% power is **~135 points/season** for
  the alpha-1.0 candidate and ~96 for alpha-1.25; resolving the observed effect
  sizes would need **21 and 116 seasons**.
- **The response surface is piecewise constant**, not smooth. In 2024-25 alone,
  horizon 6 scores -240 while horizon 7 scores +54. A 1% alpha change moves one
  season by 19 points: one arbitrary decision flip is worth ~20 points, so the
  measured -39.8 is consistent with 1-3 flips landing by chance. The
  paired-difference model (additive effect + mean-zero noise) is mis-specified
  for a step response.
- **The horizon "control" is not a null.** Pooled over h3/h4/h6/h7 x 4 seasons
  the mean is **-45.2, not 0**, and h3 alone is -78.8 at **t=-3.96, p=0.029,
  4/4 negative**. It is biased negative by construction because everything else
  in the model was tuned at h=5. Using its *spread* as a noise floor conflates
  a real mean shift with noise, so the rule "a candidate must clear the
  control's spread" fails on both halves.
- `np.std` (ddof=0) was used for 4-season sample spreads, understating sample
  sd by 13%.

What to build: get n into the hundreds - every (season x start-gameweek) as a
separate short replay is the cheap route - use a **provably null placebo**
(alpha +/- 1e-6 moves nothing; verified) instead of the horizon, and report
decision-flip step size separately from any variance estimate. A full season
replays in ~50s, so this is minutes of compute, not hours.

**Until this exists, do not retune any optimiser constant, and do not read
"measured null" anywhere in this file as "measured to be zero".** The alpha
question in particular is *unanswerable with the current harness*, not closed:
the accuracy side has real power (paired per-gameweek, alpha 1.0 beats 1.5 at
**t=7.9 pooled, 50/58 gameweeks**) and was overruled by a zero-power decision
test.

### 2. The ledger records only the 15 chosen players - FIXED 18 Oct (#31)

Schema 2 adds `pool`: id, position, xp_next, p_start, exp_minutes and
availability for **every** scored player, columnar, ~17KB a gameweek. GW8 is
the first gameweek recorded with it; GW1-7 cannot be backfilled. Not the
top-100 cut suggested below: on the snapshot the top 100 keep an xP sd of
0.547 against 1.197 for the pool, so a top-100 log is itself range
restricted and would not answer the question. `tilt_source` is now recorded
too (it was null in every record).

Next step once GW8 settles: compute r over the pool and over the squad from
the same gameweek, which separates range restriction from degradation
directly instead of through Thorndike.

Original note:

So the claim that in-season r=0.251 is range restriction rather than model
degradation **cannot be tested from the data the system records**. Measured
attenuation accounts for about half the drop (Thorndike Case II predicts
r=0.332 against 0.173 observed); the rest is population mismatch - the full
pool is ~780 players, most of whom do not play and are predicted near zero.
Note the comparison was also r-against-R2, which are different estimators; the
comparable full-pool figure is r=0.565.

Fix: log the top ~100 by xP, or a stratified sample across the xP range, each
gameweek. A few KB, and it makes the claim checkable in one gameweek.

### 3. Both deadline arms depend on the same scheduler

The coverage model's only failure mode is "no tick arrived in band", which is
not what caused either real miss: GW2 was a scheduled run that never fired,
and the watchdog's first scheduled run died of `ModuleNotFoundError`. Two
workflows in this repo are `disabled_inactivity` **right now**, and the two
"independent" arms correlate at **r=+0.54** on hour-of-day with watchdog
arrivals clustering nearer weekly arrivals than independent phase predicts
(p=0.029). An Actions outage, a quota stop or an inactivity disable kills both
at once. The season runs to 2027-05-30.

The watchdog's own design rule - watch the outcome, not the cron - is violated
at the only level that matters: its liveness depends on the scheduler it is
backstopping. Fix: an off-GitHub liveness check, and report coverage as a range
**conditional on the cron firing at all**.

### 4. The coverage figures have no committed harness - FIXED 18 Oct (#35)

`measure_deadline_coverage.py`, arrivals committed in `data/arrivals.json`.
Replay of the real stream: **69.8%** old band, **98.9%** new (the independent
re-derivation's 69.7 / 99.0 reproduce); the i.i.d. control reproduces the old
81.3 / 94.5 / 99.8, so the arithmetic was right and the method wrong. Old band
plus old watchdog was **83.2%**, not 94.8%. Replaying each past deadline on its
own day reproduces reality (GW6 missed on the old band, GW3-5 and 7 caught).

**The mechanism below is wrong.** Gap lag-1 r measures 0.507, not 0.814, and
it does not matter: the real stream replayed at uniform times of day matches
i.i.d. to 0.2pp. The whole correction is **time of day** - deadlines cluster
at 12:30-13:30 and 17:30-18:30 UTC, where GitHub's arrivals are thin. Whether
that persists past 44 days is unknown, so figures are quoted as a range. This
bears on item 3: hour-of-day structure is exactly what an off-GitHub check
would not share.

Original note:

Unique among this repo's claims, which otherwise all ship a `measure_*.py`. The
81.3% / 94.8% / 100% numbers exist only as prose. An independent re-derivation
got 69.7% for the old band and 99.0% for the new one by replaying the real
arrival stream instead of resampling i.i.d. gaps - **lag-1 autocorrelation is
+0.814**, so the i.i.d. bootstrap was invalid and optimistic by 11.6pp on the
old band. Commit `measure_deadline_coverage.py` and state the honest range.

### 5. Smaller, confirmed, each with a real failure mode

- FIXED (#32). **`always()` makes a weekly run uncancellable.** `gh run cancel` will not
  stop `manage-team` from submitting, so a bad decision spotted mid-run cannot
  be aborted. `!cancelled() && (...)` keeps fail-open on gate *failure* while
  restoring cancellability.
- FIXED (#32): 3 reads with backoff, then refuse + Telegram + exit 1. Fails closed on purpose. **`_marker_gameweek()` conflates "absent", "forbidden" and "network failed"
  into "proceed".** On the dispatch path - including the watchdog's own
  remediation - that single API call is the only idempotency in the system, so
  one transient 500 resubmits the gameweek. Return a sentinel for "could not
  determine" and refuse to proceed on it.
- FIXED (#31). **`tilt_source` is never passed**, so every ledger record says `null`. Given
  the tilt silently falls back to the template when the fan-out fails, this is
  the one field that would distinguish a field-tilted gameweek afterwards.
- FIXED (#32). **A failed `set_lineup` exiting 1 is untested.** Reverting it leaves the
  suite green; `test_exit_codes.py`'s stub returns `None` from `get_my_team`
  and never reaches STEP 6.
- FIXED (#32). **`github_api.py` has no behavioural tests.** The ledger self-test now
  exercises the live path, but nothing covers payload shape, base64 or
  sha-on-update locally. Use a local `http.server`, as `test_notify.py` does.
- FIXED (#32): alerts right after login, still submits, exits 1. **`rotation_failed` is set and never read by `manager.py`.** A failed token
  rotation logs CRITICAL, sets a flag nobody checks, and exits 0 green; the
  miss surfaces a gameweek later as an auth failure.
- FIXED (#33). **The stdlib-only test whitelists `config` without scanning it.** If anything
  in the watchdog graph grew `import config` and config grew a third-party
  import, the test passes and the watchdog dies of the exact outage it was
  written for. Walk the graph transitively instead.
- FIXED (#33). The clip threshold is 1/(1-2c) - **w = 1.25** at 10% captaincy, not ~20 - and reproduced, the arbitrary captain was a goalkeeper. Bit-identical at the production weight 0.0. **The captain tilt never tilts the vice-captain** (chosen outside the solver
  on raw `captain_col`), and at w >~ 20 every effective captain score clips to
  zero and `argmax` falls back to index order with no guard.
- OPEN. **`test_pipeline.py` exercises mutually exclusive branches** locally and in
  CI. The committed snapshot is pre-season, so locally the whole file runs the
  pre-season path (`free_transfers = 15`, zero free-transfer value) while CI
  runs in-season with real hit pricing. Neither environment covers both.
- OPEN. **Nothing asserts an upper bound on skipped tests.** ~20 data-conditional
  skip sites remain; `-rs` prints them but no test fails on a skip.
- DELETED (#34), with `token_refresh.yml`, and playwright is off the weekly run. **Dead code, confirmed:** `deadline_scheduler.py` (only referrer
  `scheduler.yml`, disabled - and if re-enabled it dispatches at a 20-28h lead,
  the stale-news behaviour the current design rejects), `league_analyzer.py`
  (superseded by `rivals.py`), `token_refresh.py` - which still pins
  **playwright** in `requirements.txt`, installed on every weekly run, a large
  dependency on the critical path 0.5-8h before a deadline for a disabled
  workflow.
- OPEN, needs a decision. **`deadline_check.yml` is a capability gap, not just cruft.** Dispatch-only
  and unreachable, and it is the only thing that re-reads availability and
  repairs the lineup after late team news. The watchdog does not do that.

### 6. What NOT to do

- Do not retune `OUTFIELD_COMPETITION_ALPHA`, `FIELD_OWNERSHIP_WEIGHT`,
  `FREE_TRANSFER_VALUE`, `DEFAULT_BENCH_WEIGHT` or `HORIZON` before item 1.
  (18 Oct: item 1 is built; alpha moved to 1.25 under pre-registered rules and
  the rest stayed. Re-measure through `measure_phase_c.py`, not by hand.)
  Every one of them was set by argument, and the harness that would settle them
  cannot currently resolve a 100-point effect.
- Do not derive a leak figure without the field distribution beside it. That
  single omission turned 24th-percentile captaincy into a reported 4.6 pts/GW
  defect.
- Do not treat a passing suite as evidence. Every defect found on 17 Oct was
  live with 350+ tests green, and two were found only because something
  independent re-checked a claim that had already been "validated" - in two
  cases circularly, against earlier arithmetic carrying the same bug.

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

### 1d. The minutes analysis, done properly

Everything above about minutes was measured on a model running **blind to
availability**. `backtest.py`'s `build_state` hardcodes `status='a'` and
`chance_of_playing_next_round=None` for every player, and `_ramp` only fires
when live news exists, so the backtest exercises neither injuries nor returns.
The headline minutes R2 of 0.474 is therefore a floor on a handicapped model,
not an estimate of production, and **9.5% of all squared minutes error is
injury absence that production sees and the backtest cannot**.

Per-gameweek availability is not reconstructible: `players_raw.csv` carries
`status`, `chance_of_playing_*` and `news`, but it is an end-of-season snapshot,
so it describes May and nothing else. The absence itself is observable though -
a run of zero-minute gameweeks - and that is enough for the two questions that
matter.

#### Returning from an absence: RAMP_UP is the wrong shape

`measure_ramp.py`. Production assumes `RAMP_UP = [0.55, 0.80]` then full, and
`LONG_ABSENCE_DAYS = 28` means anything under about four gameweeks gets no ramp
at all. Measured over six seasons, minutes on return as a ratio to the player's
own pre-absence average, **divided by a control of non-absent starters over the
same relative window**:

```
  absence          +1       +2       +3       +4
  1 GW          0.868    0.873    0.903    0.915
  2 GW          0.796    0.824    0.859    0.919
  3 GW          0.806    0.845    0.913    0.934
  4-5 GW        0.715    0.823    0.837    0.880
  6-8 GW        0.646    0.735    0.787    0.750
  9+ GW         0.602    0.719    0.826    0.736
```

**The control earns its place here.** Raw ratios look far more dramatic, because
minutes decay against a trailing five-gameweek baseline anyway - the control
runs 1.017, 0.920, 0.877, 0.848 with no absence at all. Most of the apparent
"ramp" in the raw numbers is regression to the mean.

Two errors in the current model, in opposite directions:

- **too harsh at +1**: 0.55 assumed against 0.60-0.72 measured, even for the
  longest absences.
- **far too generous from +3**: production returns to full, while a player back
  from 6+ gameweeks out is still at 0.75-0.83 in his third and fourth games.
  The ramp is **shallower but much longer** than modelled.

The gradient in absence length is also real and unmodelled - +1 runs from 0.868
after one week out to 0.602 after nine. A single `RAMP_UP` pair cannot express
it; the replacement is a 2-D table keyed on (absence length, weeks back).

Caveat that limits how far the short bands can be pushed: a run of zero-minute
gameweeks conflates injury with being dropped, and for 1-3 gameweek gaps most
are selection, not fitness. That is what the 28-day gate exists for, and those
rows are **not** an argument for ramping short absences - a dropped player's
minutes belong to the start-rate model. The evidence is clean for 4+ gameweeks.

#### International duty: measured, NULL

`measure_international.py`. Breaks are unlabelled but visible in the fixture
calendar as a fortnight between gameweeks instead of a week. Established
starters, compared within-player against their own trailing average:

```
                            n   mean minutes   ratio to base   P(blank)
  gameweek after break   3795           73.4           0.892      0.132
  CONTROL: ordinary GW  23844           74.3           0.899      0.113

  difference -0.0077   SE 0.0081   t = -0.95   (-0.88 minutes per starter)
```

Nothing there in aggregate. The split is now possible: `region` is an
alphabetical country index, and the ids in `measure_international.py` were each
confirmed by reading off a player whose nationality is not in doubt. Two early
probes matched the wrong player - region 21 resolves to Belgium via Amadou
Onana, not Cote d'Ivoire, and a search for "Erling" matched "Sterling" - so
every id is one checked by name. `region` appears only from 2024-25, but
nationality is static, so the map is built on `code` and applied backward.

```
  band                n after   ratio   n ctrl   ratio     diff       t
  CAF                     213   0.834     1366   0.913   -0.079   -2.24
  CONMEBOL                387   0.912     2620   0.913   -0.002   -0.07
  UEFA other             1250   0.908     8556   0.901   +0.008   +0.57
  far AFC/CONCACAF        105   0.957      595   0.894   +0.063   +1.37
  home nations           1204   0.900     7427   0.904   -0.005   -0.32
```

Only CAF moves, and the AFCON window says the same thing: CAF players in
January run 0.848 against 0.949 for UEFA, a difference of -0.077 (t = -1.95)
against a control of +0.008 for the same pairing outside January -
difference-in-differences **-0.085**.

**Do not ship this yet.** Two reasons. First, that is five bands times two
questions, and |t| of 2.24 and 1.95 does not survive a correction for ten
comparisons - Bonferroni wants 2.8. AFCON is at least a pre-specified
hypothesis rather than a fishing expedition, which is the only thing keeping it
alive. Second, CONMEBOL coming out at **exactly zero** is a problem for the
story: if the mechanism were long-haul travel, South America should be the
worst band, and it is flat. That points at AFCON the tournament rather than
flying as such, which is a narrower and more seasonal effect than "international
duty".

Impact on the current squad if it were shipped today: **none**. The GW1-5
horizon has gaps of 6-7 days throughout, no international break and no January,
so every multiplier would be 1.0.
### 1b. Manager-change start-rate reweighting - measured, DEAD

`measure_regime.py`. The lead was that a new manager makes the previous twenty
gameweeks of selection weak evidence, so the start rate should shrink harder.

It cannot be tested directly: FPL carried managers as assets only in 2024-25
and only from GW23, so **exactly one manager change is observable in a decade
of history on disk**. Testing against a list written from memory measures the
list, not the model.

So the test is on the channel the fix must act through. Production blends a
recent start rate into the season-long one at a fixed weight, and a regime
shift can only help by raising that weight for the affected team:

```
    p = w * recent + (1 - w) * season      RECENT_MINUTES_MAX_WEIGHT = 0.70

  band            churn     w*   Brier@w*  Brier@0.70    gain
  Q1 stable       0.131   0.75     0.1051      0.1051  0.0000
  Q2              0.180   0.70     0.0996      0.0996  0.0000
  Q3              0.216   0.65     0.1044      0.1045  0.0001
  Q4              0.259   0.70     0.0984      0.0984  0.0000
  Q5 churning     0.356   0.85     0.0967      0.0981  0.0014
  ---- CONTROL: shuffled churn, must be flat ----
  Q1..Q5       0.22-0.23   0.75      ~0.10       ~0.10  0.0000
```

`w*` wobbles 0.65-0.85 with no monotone trend and Q3 breaks the ordering. The
reason is that **w is barely identified**: across w in 0.50-1.00 the Brier
spread is 0.0011 for stable teams and 0.0088 even in the churning quintile.
Those `w*` figures are in-sample, so 0.0014 is an upper bound.

Note the shape of the trap this is built around. Churn is *defined* as the gap
between the two predictors, so comparing them head to head shows the recent one
winning ever more as churn rises - a mechanical result that looks like a
finding. Only the blend weight is a fair question. The first version of this
measurement fell into exactly that and reported t=23.

The detector also fails its one available validation. Southampton changed
manager around GW17 of 2024-25; churn over GW18-22 sits at the 0.60, 0.38,
0.39, 0.67 and 0.82 percentiles - never entering the top quintile. Caveat
honestly: n=1, and it is a weak case, since Southampton were bottom under both
managers and Jurić largely kept the XI.

**By-product worth keeping: `RECENT_MINUTES_MAX_WEIGHT = 0.70` is now
measured.** Pooled optimum is 0.75 with zero Brier difference, and the surface
is flat around it. That is one of the Priority 3 hand-picked constants retired
permanently.
### 1a. What the suspension model is, and is not

The obvious version of this feature is a **no-op**, and it is worth knowing why
before touching it. FPL sets `status='s'` on a banned player and
`STATUS_AVAILABILITY` maps that to 0.0, so **the gameweek after the fifth card
is already handled in production** and always was. Modelling it again there
prices one absence twice.

What no flag can cover is the card that has not been shown yet. Measured over
2020-26, a starter sitting on exactly four bookings is banned somewhere inside
a five-gameweek horizon **50.7% of the time** (n=1048), and was priced at full
minutes for all five. That is roughly 2 xP unpriced on one player, and the
optimiser makes transfer calls on far less. So the model runs only on
`ahead >= 1` and leaves the next gameweek to FPL.

Frequency, so the size is not oversold: ~3 affected starters league-wide at
GW8, 15 at GW12, 30 at GW16, 38 at GW19. Concentrated in the run-up to the
matchweek-19 cutoff, and worth **0.8% of total squared minutes error** - real,
cheap, and small. It will not move a backtest R2 and is not meant to.

**Do not measure this with `backtest.py`.** `build_state` hardcodes
`status='a'` and `chance_of_playing_next_round=None` for every player, so the
backtest is blind to availability in a way production is not. It would credit
this model for rediscovering bans the status flag already handles, and report a
gain that does not exist in production. `simulate.py` cannot help either: the
effect is worth 5-20 points a season against a ~53-point noise floor.

`measure_suspension.py` measures the claim the model actually makes - a
probability - held out by season:

```
  horizon GW       predicted   realised     error
  1                    0.138      0.158    -0.020
  2                    0.119      0.109     0.010
  3                    0.103      0.096     0.006
  4                    0.088      0.074     0.014
  5                    0.076      0.069     0.008
  cumulative           0.524      0.507     0.018
```

Over-prediction is the safe direction: the model assumes the player is on the
pitch every gameweek of the horizon, and real players are rested and rotated.

**Known limitation, and it is not in this model.** Goalkeepers barely respond.
Their group in `_normalise_starts_within_team` has one shirt and
`GK_COMPETITION_ALPHA = 3.0`, so allocation is on relative standing alone and a
first-choice keeper's availability cut is destroyed - Pickford and Leno retain
**0.00** of a 25% cut. That damps every availability signal a keeper gets,
injury flags included, not just this one. Tracked separately.
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

**UNBLOCKED and half-built.** GW1 completed 24 Aug 2026. `rivals.py` +
`tests/test_rivals.py` now compute field ownership and captain rates; wiring
them into the objective is the remaining work.

Two things changed the shape of this job:

**It needs no credentials.** `LeagueAnalyzer` was routed through the
authenticated client for `get_my_leagues()`, which spends a single-use refresh
token. `/entry/{id}/` publishes the same league list to anybody, so the whole
pipeline runs on public endpoints and cannot lock the account out. That removes
the only real hazard this feature had. `league_analyzer.py` is now redundant
and should be deleted rather than fixed.

**Global leagues are the template under another name.** "Overall" and "Gameweek
1" carry 8.9m entries each; treating them as a field would reintroduce the very
proxy this replaces. `MAX_LEAGUE_SIZE = 2000` keeps the four real mini leagues
(44 rivals).

Measured live on GW1, field ownership against the `selected_by_percent` the
tilt currently uses:

```
  player             field  template      gap  captained
  Haaland            84.1%     68.0%    +16.1      65.9%
  João Pedro         81.8%     67.3%    +14.5       9.1%
  Szoboszlai         68.2%     43.0%    +25.2       0.0%
  Calafiori          63.6%     41.5%    +22.1       0.0%
  B.Fernandes        61.4%     48.8%    +12.6      18.2%
  Kinsky             56.8%     22.9%    +33.9       0.0%
  Mbeumo             54.5%     36.2%    +18.3       2.3%
  Calvert-Lewin      45.5%     29.1%    +16.4       0.0%
```

**The template understates the field for every top-owned player, by 10 to 34
percentage points.** A mini league clusters far harder than eleven million
people do, so a tilt aimed at `selected_by_percent` is aimed at the wrong
target - and it is wrong in a consistent direction, which means it is
systematically under-tilting.

**Captaincy is a different distribution again, and far more concentrated.**
João Pedro is owned by 81.8% of the field and captained by 9.1% of it.
Szoboszlai and Calafiori are owned by two thirds and captained by nobody. Only
Haaland is genuinely a captaincy block at 65.9%. Squad ownership is not a proxy
for it, which is what the old plan assumed.

Practical note for the outstanding João Pedro question: the field owns him at
**81.8%**, not the 67.3% the template reports, so not owning him is a larger
differential than it looked.

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

## Audit findings, not yet fixed

### The preview tools read a DIFFERENT availability than the model scores with - FIXED 21 Oct

The columns are now built for the gameweek `xp_next` describes. It mattered
more than the note below says: the ledger's new `pool` records exactly these
columns, so GW8's minutes calibration would have been scored against numbers
the model never used - a player it priced at 0.0 xP logged as 75% available
and 59 expected minutes. No decision path reads them (optimiser, chips and
manager do not), so no squad changes. `tests/test_scored_minutes_columns.py`.


`expected_points` builds its output frame with `mm = self.minutes_model()` -
**no event** (xp_model.py:1103). `_availability(None)` returns the base flag
and skips everything event-specific: news-date availability
(`news.availability_on`), doubt decay across the horizon, and card-ban risk.
The xP itself is fine, because the per-fixture path calls
`self.minutes_model(event)` (xp_model.py:957).

So `out["availability"]`, `out["p_start"]` and `out["exp_minutes"]` are
event-less and disagree with the numbers the optimiser actually used.

That is not cosmetic. `deadline_check.py:84` is the pre-deadline safety net:

```python
flagged = squad[squad["availability"] < BENCH_BELOW_AVAILABILITY]
```

A player whose news says "out until 15 September" but whose status flag is
still 'd' reads 0.75 here and is not flagged, while the model scoring the next
gameweek correctly treats him as 0.0. **The tool that exists to catch an
unavailable player in the XI is looking at the wrong number.** This is the same
shape as the `recommend.py` ownership_weight bug: a preview disagreeing with
the run.

Fix is probably one line - pass the next event into that call - but it changes
three reported columns and wants a test that pins the preview and the run to
the same figure, so it was not done blind.

### Confirmed still dead

- `sd_next` (xp_model.py:1143) is produced and consumed by **tests only**.
- `league_analyzer.LeagueAnalyzer` is imported by **zero** non-test files.
- `price_change_percent` has no references at all.
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

## The cron missed five gameweeks a season - FIXED

`0 10 * * 5` assumed deadlines are weekly. They are not. Across 2026-27 they
fall on **Sat 26 times, Fri 5, Wed 5 and Sun 2**, and the five Wednesday
midweek rounds - **GW13, 18, 20, 25, 28** - have no Friday between the previous
deadline and their own:

```
  current: Fri 10:00 fixed
    covered 33/38   MISSED ENTIRELY: 13, 18, 20, 25, 28
    also fired up to 3x per gameweek, once with a lead of 362 hours

  now: daily 10:00, act only if 2 <= hours-to-deadline < 26
    covered 38/38   0 missed   0 doubles   leads 2.5-25.0h
```

Those five gameweeks would have gone in with a stale squad and lineup - no
transfers, no captain change, injured players left in the XI.

The cron is now daily and `manager.py` decides whether today is the day, from
the real `deadline_time`. **The window is 24 hours wide on purpose**: narrower
risks missing a deadline if a run fails, wider lets two consecutive daily runs
both qualify and reintroduces the double submission the scheduler was disabled
for. `--ignore-window` overrides it for manual dispatch.

## The GW1-5 decline diagnosis was wrong - read this before acting on it

Written up on 2026-10-07 as "losing in all our leagues", with lineup selection
leaking 7.4 points a gameweek and captaincy 4.6, measured against
hindsight-optimal. Both numbers were arithmetically right and the conclusion
was wrong, because no comparison group was ever collected. It was available
from the same public endpoints the ownership tilt already reads.

`measure_field_leaks.py` computes the identical metric for every rival in the
same mini leagues over the same gameweeks, using `calibrate.assess` so the
arithmetic is literally the same code. 92 rivals, GW1-5:

```
  metric         mine  field mean  median     sd    p90  my pctile
  lineup leak      37        30.3    28.5   17.7     57        66th
  captain leak     23        29.3    29.0    9.3     41        24th
  combined         60        59.5    57.5   19.6     87        53th
  points          329       317.0   319.5   28.8               59th
```

**Combined, this team leaks 60 against a field mean of 59.5 - the 53rd
percentile, dead average. Captaincy is the 24th percentile, better than three
quarters of the field: above-average captaincy was written up as a 4.6/GW
defect.** Only the lineup leak is mildly above the field, at the 66th
percentile, well inside p90.

Two further corrections to that write-up:

- **"+30 versus average" overstated the edge.** The global average is 299 but
  the mini-league field averages 317, so the real edge over the actual rivals
  is +12, not +30. The leagues are stronger than the global pool.
- **The 34-49 point gap to the league leaders needs no skill deficit.**
  Bootstrapping 20-manager leagues from the real rival-total distribution, the
  expected gap from a 59th-percentile manager to the leader after five
  gameweeks is ~37 points. The observed gap is the middle of that range. Five
  gameweeks cannot separate skill from variance here: rival-total sd is ~29
  over five gameweeks.

The one unambiguous loss this season was infrastructure, not modelling: GW6
passed unsubmitted because the acting window rejected a run with 1.94h in hand,
3.6 minutes short of an unjustified 2.0h floor.

**Do not re-derive a leak figure without the field distribution beside it.**

## Outstanding, verified but unfixed

Ranked. **1 and 2 are now FIXED; 3 is measured and should not be fixed.**

1. ~~**`recommend.py` prices hits 3.64x too cheap.**~~ **FIXED.** It passed no
   `horizon_weight`, so hits were priced against optimizer.py's default of 1.0
   while the value column sums five decayed gameweeks. Measured on the
   committed snapshot over six suboptimal squads x {0, 1} free transfers: all
   twelve scenarios diverged, the preview taking the maximum two hits in every
   case where the weekly run takes none. Computed from `args.horizon`, since
   unlike manager.py's constant this tool takes the horizon as an argument.
   Test asserts what the optimiser was *handed*, in `test_recommend_tilt.py`.
2. ~~**`ahead` is computed against `config.horizon`.**~~ **FIXED.** It is now
   read from the full remaining calendar, which is what "how many gameweeks
   away" means. The old curve had 25 doubtful players healing 0.7300 ->
   0.9831 through GW5 and then snapping back to 0.7300 for GW6-10, and
   `_suspension_risk` returning early on `ahead <= 0` for the same gameweeks.
   `xp_gw1..gw5` and `xp_horizon` are bit-identical, with a test pinning that.
   **Negative result: it changes no chip decision** on the current snapshot -
   triple captain stays GW7, bench boost stays GW1, only the valuation moves
   (6.57 -> 6.49). Right because the old curve modelled nothing, not because
   it moves a decision. See `tests/test_ahead_horizon.py`.
3. **Squad depth dilutes start probability — MEASURED, DO NOT "FIX".**
   `measure_competition_alpha.py` sweeps `OUTFIELD_COMPETITION_ALPHA` against
   held-out accuracy and against the symptom itself. They point in opposite
   directions:

```
   alpha   R2 22-23  23-24  24-25  25-26   MEAN | corr(depth,above80)  Palmer
   0.75      0.2608 0.2944 0.3811 0.3135 0.3125 |          -0.467      0.471
   1.00      0.2715 0.2961 0.3848 0.3216 0.3185 |          -0.611      0.516
   1.25      0.2732 0.2923 0.3812 0.3197 0.3166 |          -0.563      0.555
   1.50      0.2715 0.2890 0.3779 0.3161 0.3136 |          -0.330      0.588  <- production
   2.00      0.2663 0.2832 0.3725 0.3088 0.3077 |          -0.342      0.637
   3.00      0.2558 0.2727 0.3623 0.2943 0.2963 |          -0.257      0.687
```

   Sharpening alpha is what the open question implied. It fixes the symptom
   monotonically - Palmer rises to 0.687, the correlation weakens to -0.257 -
   and it **costs accuracy in all four seasons**, -0.006 R2 at 2.0 and -0.017
   at 3.0. The accuracy-optimal direction is the opposite, alpha ~1.0, worth
   **+0.005 R2** and better in every season measured - and it makes the
   symptom *worse*, driving the correlation to -0.611 and Palmer to 0.516.

   So the depth correlation is not a defect that this constant can repair. A
   model that concentrates start probability the way the symptom asks for is
   a less accurate model. Note also that the -0.621 in the original write-up
   does not reproduce at production alpha: it ranges -0.33 to -0.59 depending
   on how "credible" is defined, and turns *positive* at some thresholds.
   -0.611 is what alpha 1.0 produces, not alpha 1.5.

   **alpha 1.0 is MEASURED, DEAD.** It was a live candidate on +0.005 R2 - 
   comparable to the entire goals/assists/clean-sheet apparatus - and the
   decision side kills it. `measure_competition_alpha.py --decisions` replays
   four seasons with horizon sweeps as controls:

```
   config            n     mean      sd    min    max   reading
   alpha=1.0 h=5     4    -39.8    56.3   -127    +24   candidate
   alpha=1.25 h=5    4    +12.0    40.0    -52    +53   candidate
   alpha=1.5  h=4    4     -7.2    90.3   -120   +106   CONTROL
   alpha=1.5  h=6    4    -98.0    82.1   -240    -43   CONTROL
```

   The value that wins on R2 loses 39.8 points a season when decisions are
   actually replayed. alpha 1.25 gains 12.0 with an sd of 40.0, which is
   nothing. And both controls - changing the horizon, which is not a strategy
   change - move the total by more than either candidate does.

   So this constant is closed in all three directions: sharper costs accuracy,
   softer costs points, and the depth correlation it was supposed to fix moves
   the wrong way either side of production. Do not retune it.

   A note on cost, because an earlier version of this file was wrong about it:
   a full season replays in ~50 seconds and the whole sweep is ~15 minutes.
   The first timing attempt reported ~100 minutes per season, which was wall
   clock dominated by the history download, not compute - 27s of user CPU over
   78 minutes at 0% CPU. There is no need to move these measurements off a
   laptop.

## Process failures from 2026-08-28, worth not repeating

- **Merged to main with CI red.** PR CI from a week earlier was checked instead
  of re-running. This file already warned that CI and local diverge, because CI
  fetches a live snapshot while local uses the committed pre-season one. Three
  tests asserted the snapshot *was* pre-season and broke the moment GW1
  finished.
- **A new test skipped silently in CI.** Helpers selected `start_rate > 0.8`,
  which a week into a season is nobody, so four suspension tests passed by not
  running. That is the fifth instance of "a green suite proving nothing" in
  this repo. When adding a test that selects players by a threshold, check it
  still selects somebody in-season.
- **The submission path went unaudited for a whole session** while the model
  got five bug fixes worth fractions of a point. The alerting gap above was one
  `grep` away the entire time.

## Process note from 2026-10-18

- **Five PRs each green, the combination red.** #31 added a GitHubRepo test
  double; #32 added a method the run calls on it. Each PR's CI tested main
  plus itself and passed; the merge ref after #31 landed failed 5 tests.
  Caught because #32 was updated from main and CI'd again before merging.
  When PRs land in sequence, update each from main and re-run CI on the
  result - or push the full combination as a scratch branch and CI that.

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
- **`recommend.py` drifted from the weekly run a third time, one layer down.**
  After `ownership_weight` and `horizon_weight`, it also built `XPModel` with
  no `recent_minutes` - the input behind the largest accuracy gain the model
  has (R2 0.269 -> 0.319) - so the preview scored on a weaker model than the
  one submitted. Fixed 21 Oct: it reads the public `event/{gw}/live/` through
  `fpl_client.recent_minutes_from`, the same parser `get_recent_minutes` uses,
  and never logs in (the refresh token is single-use). On snapshot data
  (`--offline`, or a failed live fetch) it logs and runs without them.
  `tests/test_recommend_recent_minutes.py` records XPModel's kwargs through
  `recommend.main()`. **Anything new manager.py passes to XPModel or the
  optimiser must be passed here too.**
- **First-half chips would have lapsed at GW19 unplayed - FIXED 24 Oct.** The
  commit bar decays only to its base (12/6/15/20), and GW9-19 of 2026-27 has
  no doubles or blanks, so bench boost (~8 xP) and free hit could never
  clear it. Walking GW9-19 on sub-bar values played **zero** of four chips.
  Nothing alerted, because a chip that is never played raises nothing. The
  test meant to guard this was named "the commit floor falls to nothing at the
  end of the window" and asserted it fell to 12. `chips_at_risk_of_expiry`
  now drops the floor to zero once the gameweeks left are no more than the
  unplayed chips + 1 (GW15 for four chips); `tests/test_chip_expiry.py`.
  Follow-ups, same week: (a) every week in view is now held to *this* week's
  bar - GW7 had reported "holding 3xc for GW16 (+7.6 xP vs +7.7 now)" because
  GW16 was judged on its own lower bar (6.6 vs 8.4); `test_chip_commit_bar.py`.
  (b) free hit is valued per gameweek (#47); **wildcard deliberately stays
  current-week only**, since its future gain depends on a squad nothing here
  can know. (c) the watchdog's all-clear now lists unplayed chips and alerts
  if a window closes over one (#46). **Still open:** the weights behind all of
  it (`THRESHOLDS`, `EARLY_WINDOW_PREMIUM`, `END_OF_WINDOW_SLACK`) have never
  been backtested.
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
