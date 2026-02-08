"""
FPL Auto Manager - Deadline Check
Runs close to the GW deadline to catch last-minute changes:
  - Fresh online research for latest injury/team news
  - Swap out any newly injured/suspended starters
  - Update bench order based on latest news
  - Final captain check
Does NOT make transfers (those are done in the main Friday run).
"""
import sys
import logging
from datetime import datetime

from config import FPL_TEAM_ID
from fpl_client import FPLClient
from player_scorer import PlayerScorer
from news_researcher import NewsResearcher
from web_research import WebResearcher
from team_selector import TeamSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"fpl_run_deadline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("fpl_auto")


def run_deadline_check():
    """Last-minute lineup adjustments before the GW deadline."""
    logger.info("=" * 60)
    logger.info(f"FPL Deadline Check ({datetime.now().isoformat()})")
    logger.info("=" * 60)

    client = FPLClient()
    if not client.login():
        logger.critical("Auth failed. Aborting.")
        sys.exit(1)

    # ─── Fresh research to catch last-minute news ───
    logger.info("Running fresh research for deadline check...")
    bootstrap_data = client.get_bootstrap()

    # News research (injuries, press conferences)
    news_researcher = NewsResearcher(bootstrap_data)
    news_profiles = {}
    try:
        news_profiles = news_researcher.run_full_research()
        flagged = news_researcher.get_flagged_players(threshold=0.7)
        logger.info(f"Deadline research: {len(flagged)} flagged players.")
    except Exception as e:
        logger.warning(f"News research failed: {e}")

    # Web research (lineups, latest team news)
    web_researcher = WebResearcher(bootstrap_data)
    web_research_data = {}
    try:
        web_research_data = web_researcher.run_full_research()
    except Exception as e:
        logger.warning(f"Web research failed: {e}")

    # ─── Score with research data ───
    scorer = PlayerScorer(client)
    if news_profiles:
        scorer.inject_news_research(news_profiles)
    if web_research_data:
        scorer.inject_web_research(web_research_data)

    selector = TeamSelector(client, scorer)
    scored_df = scorer.score_players()

    # ─── Check for flagged players in starting XI ───
    my_team = client.get_my_team()
    if not my_team:
        logger.error("Cannot fetch team.")
        return

    current_picks = my_team["picks"]
    xi_ids = [p["element"] for p in current_picks if p["position"] <= 11]

    flagged = scored_df[
        (scored_df["id"].isin(xi_ids)) & (scored_df["availability"] < 0.8)
    ]

    if flagged.empty:
        logger.info("All starters look good. No changes needed.")
    else:
        logger.warning(f"Flagged starters found:")
        for _, player in flagged.iterrows():
            # Get detailed reason from news profiles
            profile = news_profiles.get(int(player["id"]), {})
            news = profile.get("fpl_news", "No detail")
            web_sources = profile.get("web_sources", [])
            source_detail = "; ".join(s.get("detail", "") for s in web_sources[:2])

            logger.warning(
                f"  {player['web_name']}: availability={player['availability']:.0%} "
                f"| FPL: {news[:60]} "
                f"| Web: {source_detail[:60] if source_detail else 'n/a'}"
            )

        logger.info("Re-optimizing lineup...")
        new_picks = selector.select_best_xi(scored_df)
        new_picks = selector.pick_captain(new_picks, scored_df)

        if new_picks:
            success = selector.apply_lineup(new_picks)
            if success:
                logger.info("Updated lineup submitted.")
            else:
                logger.error("Failed to submit updated lineup.")
        else:
            logger.error("Could not generate new lineup.")

    logger.info("Deadline check complete.")


if __name__ == "__main__":
    run_deadline_check()
