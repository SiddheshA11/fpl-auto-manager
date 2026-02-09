"""
FPL Auto Manager - Deadline Scheduler
Runs periodically to check for upcoming deadlines and trigger the main workflow.
"""
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
import requests

from config import ENDPOINTS, GITHUB_TOKEN, GITHUB_REPO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fpl_scheduler")

# Hours before deadline to trigger different workflows
MAIN_RUN_HOURS_BEFORE = 24  # Run main workflow 24 hours before deadline
DEADLINE_CHECK_HOURS_BEFORE = 2  # Run deadline check 2 hours before


def get_next_deadline() -> tuple[datetime | None, int | None]:
    """Fetch next gameweek deadline from FPL API (no auth required)."""
    try:
        resp = requests.get(ENDPOINTS["bootstrap"], timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        now = datetime.now(timezone.utc)
        
        for event in data.get("events", []):
            deadline_str = event.get("deadline_time")
            if not deadline_str:
                continue
                
            deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            
            if deadline > now and not event.get("finished"):
                logger.info(f"Next deadline: GW{event['id']} at {deadline.isoformat()}")
                return deadline, event["id"]
        
        logger.warning("No upcoming deadline found")
        return None, None
        
    except Exception as e:
        logger.error(f"Failed to fetch deadline: {e}")
        return None, None


def trigger_workflow(workflow_name: str, inputs: dict = None) -> bool:
    """Trigger a GitHub Actions workflow via repository dispatch."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.error("GITHUB_TOKEN or GITHUB_REPOSITORY not set")
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_name}/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "ref": "main",
        "inputs": inputs or {},
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code in (200, 204):
            logger.info(f"Successfully triggered {workflow_name}")
            return True
        else:
            logger.error(f"Failed to trigger {workflow_name}: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Error triggering workflow: {e}")
        return False


def check_and_trigger(check_only: bool = False) -> None:
    """Check deadline and trigger appropriate workflows."""
    deadline, gw_id = get_next_deadline()
    
    if not deadline:
        logger.info("No upcoming deadline to process")
        return
    
    now = datetime.now(timezone.utc)
    hours_until = (deadline - now).total_seconds() / 3600
    
    logger.info(f"GW{gw_id} deadline in {hours_until:.1f} hours")
    
    if check_only:
        logger.info("Check-only mode, not triggering any workflows")
        return
    
    # Trigger main workflow ~24 hours before (within 2 hour window)
    if MAIN_RUN_HOURS_BEFORE - 1 <= hours_until <= MAIN_RUN_HOURS_BEFORE + 1:
        logger.info(f"Triggering main weekly run for GW{gw_id}")
        trigger_workflow("weekly_run.yml", {"dry_run": "false"})
    
    # Trigger deadline check ~2 hours before (within 30 min window)
    elif DEADLINE_CHECK_HOURS_BEFORE - 0.5 <= hours_until <= DEADLINE_CHECK_HOURS_BEFORE + 0.5:
        logger.info(f"Triggering deadline check for GW{gw_id}")
        trigger_workflow("deadline_check.yml")
    
    else:
        logger.info("No workflow trigger needed at this time")


def main():
    parser = argparse.ArgumentParser(description="FPL Deadline Scheduler")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check deadline, don't trigger workflows")
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("FPL Deadline Scheduler")
    logger.info("=" * 50)
    
    check_and_trigger(check_only=args.check_only)


if __name__ == "__main__":
    main()
