"""
retriever.py — Runs all hardcoded queries for all competitors.
Returns a flat list of raw (query_meta, result) tuples.
"""

import logging
from typing import Any

from .constants import COMPETITORS, SEARCH_QUERIES
from .tavily_client import TavilyClient

logger = logging.getLogger(__name__)


def retrieve_all(client: TavilyClient) -> list[dict[str, Any]]:
    """
    Run all queries for all competitors.
    Returns a flat list of raw result dicts, each augmented with
    competitor metadata and the query that produced it.
    """
    raw_results: list[dict[str, Any]] = []

    for competitor in COMPETITORS:
        name = competitor["name"]
        full_name = competitor["full_name"]
        queries = SEARCH_QUERIES.get(name, [])

        for query in queries:
            logger.info(f"[{name}] Searching: {query}")
            results = client.search(query)

            if not results:
                logger.warning(f"[{name}] No results for: {query}")
                continue

            for result in results:
                raw_results.append({
                    "_competitor": name,
                    "_competitor_full_name": full_name,
                    "_search_query": query,
                    **result,
                })

    logger.info(f"Retriever: collected {len(raw_results)} raw results total.")
    return raw_results
