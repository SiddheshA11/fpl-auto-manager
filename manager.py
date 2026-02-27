"""
FPL Auto Manager - Main Orchestrator
Runs the full weekly management cycle:
    1. Authenticate
    2. Run online research (injuries, xG, form trends, news)
    3. Analyze league rivals
    4. Score all players (research-enhanced)
    5. Evaluate chips
    6. Make transfers
    7. Select best XI, captain, bench
    8. Submit everything
"""
import sys
import json
import logging
from datetime import datetime
import pandas as pd

from config import FPL_TEAM_ID, STRATEGY
from fpl_client import FPLClient
from player_scorer import PlayerScorer
from news_researcher import NewsResearcher
from web_research import WebResearcher
from transfer_optimizer import TransferOptimizer, WildcardOptimizer
from chip_strategy import ChipStrategyEngine
from league_analyzer import LeagueAnalyzer
from team_selector import TeamSelector

# ──────────────── Logging Setup ────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"fpl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("fpl_auto")


def run_weekly_cycle(dry_run: bool = False):
    """
    Execute the full weekly FPL management pipeline.

    Args:
        dry_run: If True, compute everything but don't submit to FPL.
    """
    logger.info("=" * 60)
    logger.info(f"FPL Auto Manager - Weekly Run ({datetime.now().isoformat()})")
    logger.info(f"Team ID: {FPL_TEAM_ID}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 60)

    # ─── Step 1: Authenticate ───
    logger.info("STEP 1: Authenticating...")
    client = FPLClient()
    if not client.login():
        logger.critical("Authentication failed. Aborting.")
        sys.exit(1)

    # ─── Step 2: Gather context ───
    logger.info("STEP 2: Gathering game data...")
    next_event = client.get_next_event()
    if not next_event:
        logger.warning("No upcoming gameweek found. Season may be over.")
        return

    event_id = next_event["id"]
    current_event = client.get_current_event()
    current_gw = current_event["id"] if current_event else event_id - 1
    logger.info(f"Current GW: {current_gw}, Next GW: {event_id}")

    my_team = client.get_my_team()
    if not my_team:
        logger.critical("Could not fetch team data. Aborting.")
        sys.exit(1)

    free_transfers = my_team["transfers"].get("limit", 1)
    bank = my_team["transfers"].get("bank", 0) / 10.0
    logger.info(f"Free transfers: {free_transfers}, Bank: £{bank:.1f}m")

    # ─── Step 3: Online Research (injuries, xG, form, news) ───
    logger.info("STEP 3: Running online research...")
    bootstrap_data = client.get_bootstrap()

    # 3a. News & injury research
    news_researcher = NewsResearcher(bootstrap_data)
    news_profiles = {}
    try:
        news_profiles = news_researcher.run_full_research()
        flagged = news_researcher.get_flagged_players(threshold=0.7)
        logger.info(f"News research: {len(flagged)} players with availability concerns.")
    except Exception as e:
        logger.warning(f"News research failed (non-critical): {e}")

    # 3b. Deep web research (xG, form momentum, set pieces, etc.)
    web_researcher = WebResearcher(bootstrap_data)
    web_research_data = {}
    try:
        web_research_data = web_researcher.run_full_research()
    except Exception as e:
        logger.warning(f"Web research failed (non-critical): {e}")

    # ─── Step 4: Initialize engines with research data ───
    logger.info("STEP 4: Initializing scoring engine with research data...")
    scorer = PlayerScorer(client)

    # Inject research into the scorer BEFORE scoring
    if news_profiles:
        scorer.inject_news_research(news_profiles)
    if web_research_data:
        scorer.inject_web_research(web_research_data)

    transfer_optimizer = TransferOptimizer(client, scorer)
    chip_engine = ChipStrategyEngine(client, scorer)
    league_analyzer = LeagueAnalyzer(client)
    team_selector = TeamSelector(client, scorer)

    # ─── Step 5: League rival analysis ───
    logger.info("STEP 5: Analyzing league rivals...")
    scored_df = scorer.score_players()
    league_report = {}
    try:
        league_report = league_analyzer.generate_league_report(current_gw, scored_df)
        if league_report:
            logger.info(f"League analysis complete: {league_report.get('rivals_analyzed', 0)} rivals.")
            _log_league_insights(league_report)
    except Exception as e:
        logger.warning(f"League analysis failed (non-critical): {e}")

    # ─── Step 6: Chip evaluation ───
    logger.info("STEP 6: Evaluating chip strategy...")
    chip_rec = chip_engine.get_chip_recommendation()
    logger.info(f"Chip recommendation: {chip_rec['chip'] or 'none'} "
                f"(confidence={chip_rec['confidence']:.2f}) - {chip_rec['reason']}")

    use_chip = chip_rec["chip"] if chip_rec["confidence"] >= 0.6 else None

    # ─── Step 7: Transfers ───
    logger.info("STEP 7: Optimizing transfers...")

    if use_chip == "wildcard":
        logger.info("WILDCARD ACTIVE - rebuilding entire squad.")
        wc_optimizer = WildcardOptimizer(client, scorer)
        optimal_squad = wc_optimizer.build_optimal_squad(budget=bank + _squad_value(my_team, scored_df))
        transfer_plan = {
            "transfers_in": [p["element"] for p in optimal_squad],
            "transfers_out": [p["element"] for p in my_team["picks"]],
            "prices_in": [int(p["now_cost_m"] * 10) for p in optimal_squad],
            "prices_out": [int(p.get("selling_price", 0)) for p in my_team["picks"]],
            "use_wildcard": True,
            "use_free_hit": False,
            "expected_gain": 0,
            "hit_cost": 0,
            "num_transfers": len(optimal_squad),
        }
    elif use_chip == "freehit":
        logger.info("FREE HIT ACTIVE - building temporary squad.")
        wc_optimizer = WildcardOptimizer(client, scorer)
        optimal_squad = wc_optimizer.build_optimal_squad(budget=bank + _squad_value(my_team, scored_df))
        transfer_plan = {
            "transfers_in": [p["element"] for p in optimal_squad],
            "transfers_out": [p["element"] for p in my_team["picks"]],
            "prices_in": [int(p["now_cost_m"] * 10) for p in optimal_squad],
            "prices_out": [int(p.get("selling_price", 0)) for p in my_team["picks"]],
            "use_wildcard": False,
            "use_free_hit": True,
            "expected_gain": 0,
            "hit_cost": 0,
            "num_transfers": len(optimal_squad),
        }
    else:
        transfer_plan = transfer_optimizer.find_best_transfers()

    if not dry_run and transfer_plan["transfers_in"]:
        logger.info("Executing transfers...")
        success = transfer_optimizer.execute_transfers(transfer_plan)
        if not success:
            logger.error("Transfer execution failed! Continuing with existing squad.")
    elif dry_run:
        logger.info("[DRY RUN] Skipping transfer execution.")

    # ─── Step 8: Team selection ───
    logger.info("STEP 8: Selecting optimal lineup...")

    # Re-score after transfers (cache invalidation, research data preserved)
    client._bootstrap_cache = None
    scored_df = scorer.score_players()

    picks = team_selector.select_best_xi(scored_df)

    if not picks:
        logger.error("Could not generate valid lineup!")
        return

    # Captain selection with league intelligence
    picks = team_selector.pick_captain(picks, scored_df, league_report)

    # Determine lineup chip (bench boost or triple captain, NOT wildcard/freehit)
    lineup_chip = None
    if use_chip in ("bboost", "3xc"):
        lineup_chip = use_chip

    if not dry_run:
        logger.info("Submitting lineup...")
        success = team_selector.apply_lineup(picks, chip=lineup_chip)
        if success:
            logger.info("LINEUP SUBMITTED SUCCESSFULLY!")
        else:
            logger.error("Lineup submission failed!")
    else:
        logger.info("[DRY RUN] Skipping lineup submission.")

    # ─── Summary ───
    logger.info("=" * 60)
    logger.info("WEEKLY CYCLE COMPLETE")
    logger.info(f"  Research: {len(news_profiles)} news profiles, {len(web_research_data)} web data categories")
    logger.info(f"  Transfers made: {transfer_plan['num_transfers']}")
    logger.info(f"  Hit cost: -{transfer_plan['hit_cost']} pts")
    logger.info(f"  Chip used: {use_chip or 'none'}")
    captain_pick = next((p for p in picks if p["is_captain"]), None)
    if captain_pick:
        cname = scored_df.loc[scored_df["id"] == captain_pick["element"], "web_name"].values
        logger.info(f"  Captain: {cname[0] if len(cname) else captain_pick['element']}")
    logger.info("=" * 60)

    return {
        "event_id": event_id,
        "transfers": transfer_plan,
        "chip": use_chip,
        "picks": picks,
        "league_report": league_report,
        "news_flagged": len(news_researcher.get_flagged_players()) if news_profiles else 0,
    }


def _squad_value(team_data: dict, scored_df: pd.DataFrame) -> float:
    """Calculate total selling value of current squad."""
    total = 0
    for pick in team_data["picks"]:
        total += pick.get("selling_price", 0)
    return total / 10.0


def _log_league_insights(report: dict):
    """Log key league analysis findings."""
    diffs = report.get("differentials", {})

    threats = diffs.get("rival_threats", [])
    if threats:
        logger.info("Rival threats (high ownership, not in your squad):")
        for t in threats[:5]:
            logger.info(f"  {t['web_name']}: {t['rival_ownership_pct']}% owned by rivals")

    captains = diffs.get("popular_captains", [])
    if captains:
        logger.info("Popular rival captains:")
        for c in captains[:3]:
            logger.info(f"  {c['web_name']}: captained by {c['rival_captain_pct']}% of rivals")

    chip_usage = report.get("rival_chip_usage", {})
    if chip_usage:
        logger.info(f"Rival chip usage this GW: {chip_usage}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FPL Auto Manager")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without making changes")
    args = parser.parse_args()

    run_weekly_cycle(dry_run=args.dry_run)
