"""How often does a scheduled run land in the acting band before a deadline?

The coverage figures in deadline_state.py and watchdog.py (81.3% for the old
[2, 6) band, 94.8% for old band plus old watchdog, 100% now) came from
resampling the observed arrival gaps independently, from a random phase.

This replays the real arrival stream instead. For every 2026-27 deadline it
takes the deadline's time of day and asks, on each day the stream was
observed, whether an arrival landed in the band before that time on that day.
That keeps the hour-of-day pattern, the gap order, and - for the joint figure
- the correlation between the weekly run and the watchdog, which ride the
same scheduler and are not independent.

The i.i.d. figures were optimistic, but NOT for the reason the 17 Oct audit
gave. Gap autocorrelation is real (lag-1 r ~0.5) and, measured, irrelevant:
replaying the real stream at uniformly spread times of day reproduces the
i.i.d. number to 0.2pp, because one window's coverage depends on how much
time sits inside long gaps, not on their order. The whole correction is time
of day. Deadlines cluster at 12:30-13:30 and 17:30-18:30 UTC, and GitHub's
throttled arrivals are thin in the hours before them. Whether that pattern is
stable beyond the 44 days observed is not known, so the honest statement is a
range: the "uniform time" column is the floor of the correction.

Everything here is CONDITIONAL ON THE CRON FIRING AT ALL. An Actions outage, a
quota stop or a `disabled_inactivity` kills both streams at once, and no
amount of band tuning covers it. Both real misses this season were of that
kind (GW2: the scheduled run never fired; the watchdog's first run died on an
import), not of the "no tick landed in band" kind this file measures.

CONTROLS: the i.i.d. gap bootstrap, and the real stream replayed at uniform
times of day, so both the size of the correction and its source are visible. CHECK: replaying each past deadline on
its own day must reproduce what actually happened - GW6 missed by the old
band, GW3-5 and GW7 caught.

Arrivals come from `gh run list` (scheduled runs only) and are written to
data/arrivals.json so the figures stay reproducible after GitHub's run
history ages out; pass --refresh to re-fetch.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import statistics
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import deadline_state as ds

ARRIVALS = Path(__file__).parent / "data" / "arrivals.json"
REPO = "SiddheshA11/fpl-auto-manager"

# When each cron was live on main, from the merge commits that changed it.
# Scheduled runs always use the default branch, so these are what matter.
WEEKLY_HOURLY_FROM = datetime(2026, 9, 3, 20, 24, 54, tzinfo=timezone.utc)      # 3b70417
WATCHDOG_3H_FROM = datetime(2026, 9, 3, 21, 42, 9, tzinfo=timezone.utc)         # 90e2a6b
WATCHDOG_HOURLY_FROM = datetime(2026, 10, 12, 2, 46, 26, tzinfo=timezone.utc)   # 4644105

OLD_BAND = (2.0, 6.0)
NEW_BAND = (ds.DEADLINE_WINDOW_MIN, ds.DEADLINE_WINDOW_MAX)
OLD_ALERT, NEW_ALERT = 4.0, 6.0

# Assumed, not measured: from a watchdog tick to a dispatched run having
# submitted. Too few watchdog-triggered dispatches exist to measure it; the
# sensitivity row shows how much it matters.
DISPATCH_LATENCY_H = 0.25


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fetch_arrivals() -> dict[str, list[str]]:
    out = {}
    for wf in ("weekly_run.yml", "watchdog.yml"):
        raw = subprocess.run(
            ["gh", "run", "list", "--repo", REPO, "--workflow", wf, "--event", "schedule",
             "-L", "5000", "--json", "createdAt", "-q", ".[].createdAt"],
            check=True, capture_output=True, text=True, timeout=120,
        ).stdout.split()
        out[wf] = sorted(raw)
    return out


def load_arrivals(refresh: bool) -> dict[str, list[datetime]]:
    if refresh or not ARRIVALS.exists():
        data = fetch_arrivals()
        data["_fetched"] = datetime.now(timezone.utc).isoformat()
        ARRIVALS.parent.mkdir(parents=True, exist_ok=True)
        ARRIVALS.write_text(json.dumps(data, indent=1) + "\n")
    data = json.loads(ARRIVALS.read_text())
    return {k: [_parse(t) for t in v] for k, v in data.items() if not k.startswith("_")}


def era(stream: list[datetime], start: datetime, end: datetime | None = None) -> list[datetime]:
    return [t for t in stream if t >= start and (end is None or t < end)]


def gaps_h(stream: list[datetime]) -> list[float]:
    return [(b - a).total_seconds() / 3600 for a, b in zip(stream, stream[1:])]


def lag1(xs: list[float]) -> float:
    """Pearson correlation of consecutive gaps."""
    a, b = xs[:-1], xs[1:]
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5


def any_in(stream: list[datetime], lo: datetime, hi: datetime) -> bool:
    """Is there an arrival in [lo, hi)?"""
    i = bisect.bisect_left(stream, lo)
    return i < len(stream) and stream[i] < hi


def replay(deadlines: list[datetime], start: datetime, end: datetime,
           hit, same_weekday: bool = False) -> list[tuple[datetime, datetime, bool]]:
    """
    Lay each deadline's time of day onto every observed day and evaluate
    `hit(target)`. Only days whose whole look-back window lies inside the
    observed span count, so a span edge is never scored as a miss.
    """
    rows = []
    lookback = timedelta(hours=12)
    for d in deadlines:
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        while day < end:
            target = day.replace(hour=d.hour, minute=d.minute, second=d.second)
            if (target - lookback >= start and target <= end
                    and (not same_weekday or target.weekday() == d.weekday())):
                rows.append((d, target, hit(target)))
            day += timedelta(days=1)
    return rows


def band_hit(stream, band):
    lo, hi = band
    return lambda t: any_in(stream, t - timedelta(hours=hi), t - timedelta(hours=lo))


def joint_hit(weekly, watchdog, band, alert, latency=DISPATCH_LATENCY_H):
    """Submitted if the weekly run lands in band, or the watchdog lands inside
    its alert lead with time left for a dispatched run to finish."""
    in_band = band_hit(weekly, band)
    return lambda t: in_band(t) or any_in(
        watchdog, t - timedelta(hours=alert), t - timedelta(hours=latency))


def iid_bootstrap(gaps: list[float], band, reps: int, rng: random.Random) -> float:
    """The method being replaced: i.i.d. gaps from a uniformly random phase."""
    lo, hi = band
    hits = 0
    for _ in range(reps):
        # Start far enough back that the first arrival's phase is random.
        t = -48.0 + rng.random() * rng.choice(gaps)
        while t < -hi:
            t += rng.choice(gaps)
        hits += t < -lo
    return hits / reps


def summarise(rows) -> tuple[float, int, float]:
    """(coverage, n day-replays, worst single deadline)."""
    if not rows:
        return float("nan"), 0, float("nan")
    by_deadline: dict[datetime, list[bool]] = {}
    for d, _, h in rows:
        by_deadline.setdefault(d, []).append(h)
    worst = min(sum(v) / len(v) for v in by_deadline.values())
    return sum(h for *_, h in rows) / len(rows), len(rows), worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-fetch arrivals with gh")
    ap.add_argument("--reps", type=int, default=20000, help="i.i.d. bootstrap draws")
    args = ap.parse_args()

    arrivals = load_arrivals(args.refresh)
    events = ds.fetch_events()
    deadlines = sorted(d for d in (ds.parse_deadline(e) for e in events) if d)
    now = max(max(v) for v in arrivals.values())

    weekly = era(arrivals["weekly_run.yml"], WEEKLY_HOURLY_FROM)
    wd_old = era(arrivals["watchdog.yml"], WATCHDOG_3H_FROM, WATCHDOG_HOURLY_FROM)
    wd_new = era(arrivals["watchdog.yml"], WATCHDOG_HOURLY_FROM)

    print(f"{len(deadlines)} deadlines; arrivals through {now.isoformat()[:16]}Z\n")
    print("arrival streams (scheduled runs, while each cron was live on main)")
    print(f"  {'stream':<26}{'n':>5}{'days':>7}{'per day':>9}{'median gap':>12}"
          f"{'max gap':>9}{'lag-1 r':>9}")
    for name, s in (("weekly run, hourly", weekly), ("watchdog, 3-hourly", wd_old),
                    ("watchdog, hourly", wd_new)):
        g = gaps_h(s)
        days = (s[-1] - s[0]).total_seconds() / 86400
        print(f"  {name:<26}{len(s):>5}{days:>7.1f}{len(s) / days:>9.2f}"
              f"{statistics.median(g):>11.2f}h{max(g):>8.2f}h{lag1(g):>9.3f}")

    rng = random.Random(0)
    wg = gaps_h(weekly)
    print("\nP(a weekly-run tick lands in the band), weekly run alone")
    # Every 10 minutes of the day: the real stream, with its gap order and
    # autocorrelation intact, but no alignment to when deadlines fall.
    uniform = [deadlines[0].replace(hour=h, minute=mm) for h in range(24) for mm in range(0, 60, 10)]
    print(f"  {'band':<14}{'iid gaps':>10}{'uniform time':>14}{'replay':>9}{'same wkday':>12}"
          f"{'worst GW':>10}{'n':>6}")
    for band in (OLD_BAND, (0.5, 6.0), NEW_BAND, (0.25, 12.0)):
        iid = iid_bootstrap(wg, band, args.reps, rng)
        uni, _, _ = summarise(replay(uniform, weekly[0], now, band_hit(weekly, band)))
        cov, n, worst = summarise(replay(deadlines, weekly[0], now, band_hit(weekly, band)))
        wk, _, _ = summarise(replay(deadlines, weekly[0], now, band_hit(weekly, band), True))
        label = f"[{band[0]:g}, {band[1]:g})"
        print(f"  {label:<14}{iid:>10.1%}{uni:>14.1%}{cov:>9.1%}{wk:>12.1%}{worst:>10.1%}{n:>6}")

    print("\nP(gameweek submitted): weekly run OR watchdog dispatch, replayed on the same day")
    print("(both streams from the same real day, so their correlation is kept)")
    print(f"  {'config':<44}{'replay':>8}{'worst GW':>10}{'n':>6}")
    configs = (
        ("old band + 3-hourly watchdog @4h", OLD_BAND, wd_old, OLD_ALERT, WATCHDOG_3H_FROM,
         WATCHDOG_HOURLY_FROM, DISPATCH_LATENCY_H),
        ("new band + 3-hourly watchdog @4h", NEW_BAND, wd_old, OLD_ALERT, WATCHDOG_3H_FROM,
         WATCHDOG_HOURLY_FROM, DISPATCH_LATENCY_H),
        ("new band + hourly watchdog @6h", NEW_BAND, wd_new, NEW_ALERT, WATCHDOG_HOURLY_FROM,
         now, DISPATCH_LATENCY_H),
        ("  same, dispatch latency 0.5h", NEW_BAND, wd_new, NEW_ALERT, WATCHDOG_HOURLY_FROM,
         now, 0.5),
    )
    for name, band, wd, alert, start, end, lat in configs:
        cov, n, worst = summarise(replay(deadlines, start, end,
                                         joint_hit(weekly, wd, band, alert, lat)))
        print(f"  {name:<44}{cov:>8.1%}{worst:>10.1%}{n:>6}")

    print("\nCHECK: each past deadline replayed on its own day, against what happened")
    print(f"  {'GW':<4}{'deadline':<18}{'old band':>10}{'new band':>10}  arrivals in the 12h before")
    for e in events:
        d = ds.parse_deadline(e)
        if not d or d < weekly[0] + timedelta(hours=12) or d > now:
            continue
        before = [t for t in weekly if d - timedelta(hours=12) <= t < d]
        leads = ", ".join(f"{(d - t).total_seconds() / 3600:.2f}h" for t in before)
        old = band_hit(weekly, OLD_BAND)(d)
        new = band_hit(weekly, NEW_BAND)(d)
        print(f"  {e['id']:<4}{d.isoformat()[:16]:<18}{'hit' if old else 'MISS':>10}"
              f"{'hit' if new else 'MISS':>10}  {leads}")

    print("\nAll of the above is conditional on the scheduler running. It measures")
    print("'no tick landed in band', which caused neither real miss this season.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
