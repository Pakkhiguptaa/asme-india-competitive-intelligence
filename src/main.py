"""
main.py — CLI entrypoint for the ASME India Competitive Intelligence System.

Commands:
  python -m src.main run-daily     Run full pipeline: retrieve, classify, store, output
  python -m src.main run-weekly    Generate weekly digest from Notion data
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone

# Configure logging before any imports that use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_daily():
    """Full daily pipeline: retrieve → normalize → dedupe → classify → score → store → output."""
    from .config import Config
    from .tavily_client import TavilyClient
    from .retriever import retrieve_all
    from .normalizer import normalize
    from .deduper import compute_signal_hash, deduplicate_within_run, mark_new_vs_existing
    from .classifier import classify_signals
    from .scorer import rank_signals, top_new_signals
    from .notion_client import NotionClient
    from .digest import format_daily_output

    run_id = str(uuid.uuid4())[:8]
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(f"=== ASME Intel Daily Run | run_id={run_id} | {run_timestamp} ===")

    # --- Step 1: Retrieve ---
    logger.info("Step 1/6: Retrieving signals from Tavily...")
    tavily = TavilyClient(
        api_key=Config.TAVILY_API_KEY,
        max_results=Config.TAVILY_MAX_RESULTS,
        search_depth=Config.TAVILY_SEARCH_DEPTH,
    )
    raw_results = retrieve_all(tavily)

    if not raw_results:
        logger.warning("No raw results returned. Check Tavily API key or network.")
        print("\nNo results retrieved. Exiting.")
        sys.exit(0)

    logger.info(f"Retrieved {len(raw_results)} raw results.")

    # --- Step 2: Normalize ---
    logger.info("Step 2/6: Normalizing results...")
    signals = normalize(raw_results, run_id, run_timestamp)
    logger.info(f"Normalized to {len(signals)} signals.")

    if not signals:
        logger.warning("No valid signals after normalization.")
        print("\nNo valid signals after normalization. Exiting.")
        sys.exit(0)

    # --- Step 3: Deduplicate within run ---
    logger.info("Step 3/6: Deduplicating within run...")
    signals = deduplicate_within_run(signals)
    logger.info(f"{len(signals)} signals after within-run dedup.")

    # Attach signal_hash to each signal
    for signal in signals:
        signal["signal_hash"] = compute_signal_hash(
            signal.get("competitor", ""),
            signal.get("title", ""),
            signal.get("url", ""),
        )

    # --- Step 4: Classify ---
    logger.info(f"Step 4/6: Classifying {len(signals)} signals with LLM...")
    signals = classify_signals(
        signals,
        llm_api_key=Config.LLM_API_KEY,
        llm_model=Config.LLM_MODEL,
        llm_provider=Config.LLM_PROVIDER,
        llm_base_url=Config.LLM_BASE_URL,
    )

    # --- Step 5: Store in Notion + mark new vs existing ---
    logger.info("Step 5/6: Writing to Notion...")
    notion = NotionClient(
        api_key=Config.NOTION_API_KEY,
        database_id=Config.NOTION_DATABASE_ID,
    )

    # Get existing hashes BEFORE writing so we can mark is_new correctly
    existing_hashes = notion.get_existing_hashes()
    signals = mark_new_vs_existing(signals, existing_hashes)

    # Score and rank all signals (for composite_score before storage)
    signals = rank_signals(signals)

    written, skipped = notion.write_signals(signals)
    logger.info(f"Notion: {written} written, {skipped} skipped.")

    # --- Step 6: Output ---
    logger.info("Step 6/6: Generating output...")
    top = top_new_signals(signals, n=Config.TOP_SIGNALS_COUNT)
    output = format_daily_output(top, run_id, run_timestamp)
    print(output)

    # Summary line for logs
    new_count = sum(1 for s in signals if s.get("is_new"))
    logger.info(
        f"Daily run complete. "
        f"Total signals: {len(signals)}, "
        f"New: {new_count}, "
        f"Top displayed: {len(top)}, "
        f"Written to Notion: {written}"
    )


def run_weekly():
    """
    Weekly digest: pull recent signals from Notion, generate digest report.
    Queries all signals stored in the Notion database (no date filter —
    intended to be run after ~7 days of daily runs).
    """
    from .config import Config
    from .notion_client import NotionClient
    from .digest import format_weekly_digest

    logger.info("=== ASME Intel Weekly Digest ===")

    notion = NotionClient(
        api_key=Config.NOTION_API_KEY,
        database_id=Config.NOTION_DATABASE_ID,
    )

    # Fetch all signals from Notion
    signals = _fetch_all_signals_from_notion(notion)

    if not signals:
        print("\nNo signals found in Notion database. Run `run-daily` first.")
        sys.exit(0)

    logger.info(f"Loaded {len(signals)} signals from Notion for digest.")
    output = format_weekly_digest(signals)
    print(output)


def _fetch_all_signals_from_notion(notion) -> list[dict]:
    """
    Fetch all pages from the Notion database and reconstruct signal dicts
    for digest formatting.
    """
    import httpx

    signals = []
    has_more = True
    start_cursor = None

    while has_more:
        payload = {"page_size": 100}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        try:
            response = httpx.post(
                f"https://api.notion.com/v1/databases/{notion.database_id}/query",
                headers=notion.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            for page in data.get("results", []):
                props = page.get("properties", {})
                signal = _props_to_signal(props)
                signals.append(signal)

            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

        except Exception as e:
            logger.error(f"Error fetching signals from Notion: {e}")
            break

    return signals


def _props_to_signal(props: dict) -> dict:
    """Convert Notion page properties back to a signal dict for digest use."""

    def get_rt(key):
        try:
            rt = props.get(key, {}).get("rich_text", [])
            return rt[0]["text"]["content"] if rt else ""
        except Exception:
            return ""

    def get_title(key="title"):
        try:
            t = props.get(key, {}).get("title", [])
            return t[0]["text"]["content"] if t else ""
        except Exception:
            return ""

    def get_select(key):
        try:
            return props.get(key, {}).get("select", {}).get("name", "")
        except Exception:
            return ""

    def get_number(key):
        try:
            return props.get(key, {}).get("number", 0) or 0
        except Exception:
            return 0

    def get_url(key):
        try:
            return props.get(key, {}).get("url", "") or ""
        except Exception:
            return ""

    def get_checkbox(key):
        try:
            return props.get(key, {}).get("checkbox", False)
        except Exception:
            return False

    return {
        "title": get_title("title"),
        "competitor": get_select("competitor"),
        "competitor_full_name": get_rt("competitor_full_name"),
        "signal_type": get_select("signal_type"),
        "geography": get_select("geography"),
        "business_impact": get_select("business_impact"),
        "urgency": get_select("urgency"),
        "revenue_relevance": get_select("revenue_relevance"),
        "importance_score": get_number("importance_score"),
        "confidence_score": get_number("confidence_score"),
        "composite_score": get_number("composite_score"),
        "summary": get_rt("summary"),
        "why_it_matters": get_rt("why_it_matters"),
        "recommended_action": get_rt("recommended_action"),
        "url": get_url("url"),
        "source": get_rt("source"),
        "published_date": get_rt("published_date"),
        "run_id": get_rt("run_id"),
        "run_timestamp": get_rt("run_timestamp"),
        "signal_hash": get_rt("signal_hash"),
        "is_new": get_checkbox("is_new"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ASME India Competitive Intelligence System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["run-daily", "run-weekly"],
        help=(
            "run-daily  : Retrieve, classify, store, and output today's top signals\n"
            "run-weekly : Generate a weekly digest from stored Notion data"
        ),
    )
    args = parser.parse_args()

    if args.command == "run-daily":
        run_daily()
    elif args.command == "run-weekly":
        run_weekly()


if __name__ == "__main__":
    main()
