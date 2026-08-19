"""
Fetch reference data the xP model needs.

Two sources, deliberately separated:

  snapshots/  live FPL API captures. Time-sensitive: FPL wipes last season's
              aggregates out of bootstrap-static at the season rollover, so a
              snapshot taken before rollover is the only place those numbers
              survive.
  history/    per-gameweek history from the vaastav/Fantasy-Premier-League
              community dataset. Third party, so it is fetched rather than
              vendored, and nothing in the model may hard-fail on its absence.

Run: python fetch_data.py [--seasons 2024-25,2025-26]
"""
import argparse
import gzip
import json
import logging
import shutil
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("fpl_auto.fetch")

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
HISTORY_DIR = DATA_DIR / "history"

FPL_API = "https://fantasy.premierleague.com/api"
LIVE_ENDPOINTS = {"bootstrap-static": f"{FPL_API}/bootstrap-static/", "fixtures": f"{FPL_API}/fixtures/"}

VAASTAV_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASON_FILES = ["players_raw.csv", "teams.csv", "fixtures.csv", "gws/merged_gw.csv"]
# Seasons are derived from the date, never hardcoded.
#
# They used to be a literal ["2024-25", "2025-26"]. In a project designed to
# run unattended for years that is a dated bomb: come August 2027 nothing
# fetches 2026-27, priors are built from data two and three years old,
# PriorSet.validate() passes because it only counts rows, and the bot quietly
# submits an opening squad priced off a season before last - with every summer
# signing invisible to it. Nothing fails; it just gets worse.
HISTORY_SEASONS = 3


def season_label(start_year: int) -> str:
    """FPL's own naming: the 2026-27 season starts in 2026."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_season_start_year(today: date | None = None) -> int:
    """
    The year the season now in progress began.

    A season runs August to May, so anything from July onward belongs to the
    season starting this year, and anything earlier to the one starting last.
    July rather than August because pre-season data appears before a ball is
    kicked.
    """
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def default_seasons(today: date | None = None) -> list[str]:
    """The current season and the two before it, oldest first."""
    start = current_season_start_year(today)
    return [season_label(y) for y in range(start - HISTORY_SEASONS + 1, start + 1)]

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 60
RETRIES = 3


def _get(url: str) -> bytes:
    """GET with backoff. Raises on final failure so callers decide what is fatal."""
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=TIMEOUT) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            if attempt < RETRIES - 1:
                backoff = 2**attempt
                logger.warning("GET %s failed (%s); retrying in %ss", url, e, backoff)
                time.sleep(backoff)
    raise RuntimeError(f"GET {url} failed after {RETRIES} attempts: {last_err}")


def snapshot_live() -> list[Path]:
    """
    Capture the live FPL API. Gzipped because bootstrap-static is ~1.4MB of JSON
    and these accumulate one set per run.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written = []

    for name, url in LIVE_ENDPOINTS.items():
        raw = _get(url)
        json.loads(raw)  # fail loudly here rather than at model-build time
        dest = SNAPSHOT_DIR / f"{name}_{stamp}.json.gz"
        with gzip.open(dest, "wb") as fh:
            fh.write(raw)
        logger.info("snapshot %s -> %s (%.0fKB gz)", name, dest.name, dest.stat().st_size / 1024)
        written.append(dest)

    return written


def fetch_history(seasons: list[str]) -> list[Path]:
    """
    Pull per-gameweek history for the given seasons. A missing season is logged
    and skipped: the dataset lags the live season, and an incomplete history
    degrades the priors rather than breaking them.
    """
    written = []
    for season in seasons:
        season_dir = HISTORY_DIR / season
        season_dir.mkdir(parents=True, exist_ok=True)

        # A finished season never changes, so it is fetched once. The season in
        # progress changes every week, and skipping it froze team strength at
        # last season's values for the whole campaign - by May the model was
        # still pricing every side on data a year old.
        in_progress = season == season_label(current_season_start_year())

        for rel in SEASON_FILES:
            dest = season_dir / Path(rel).name
            if dest.exists() and not in_progress:
                logger.info("have %s/%s, skipping", season, dest.name)
                written.append(dest)
                continue
            try:
                raw = _get(f"{VAASTAV_RAW}/{season}/{rel}")
            except RuntimeError as e:
                logger.warning("skipping %s/%s: %s", season, rel, e)
                continue
            dest.write_bytes(raw)
            logger.info("fetched %s/%s (%.0fKB)", season, dest.name, len(raw) / 1024)
            written.append(dest)

    return written


def prune_snapshots(keep: int = 8) -> None:
    """Keep the most recent N snapshots per endpoint; they are ~200KB each gzipped."""
    for name in LIVE_ENDPOINTS:
        files = sorted(SNAPSHOT_DIR.glob(f"{name}_*.json.gz"), reverse=True)
        for stale in files[keep:]:
            stale.unlink()
            logger.info("pruned %s", stale.name)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(default_seasons()))
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--skip-history", action="store_true")
    args = ap.parse_args()

    failed = False

    if not args.skip_live:
        try:
            snapshot_live()
            prune_snapshots()
        except RuntimeError as e:
            # The live snapshot is the irreplaceable half; surface it as a failure.
            logger.error("live snapshot failed: %s", e)
            failed = True

    if not args.skip_history:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
        if not fetch_history(seasons):
            logger.error("no history files fetched for %s", seasons)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
