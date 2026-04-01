"""
tavily_client.py — Thin wrapper around the Tavily search API.
Returns raw results without any transformation.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilyClient:
    def __init__(self, api_key: str, max_results: int = 5, search_depth: str = "advanced"):
        self.api_key = api_key
        self.max_results = max_results
        self.search_depth = search_depth

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Execute a single Tavily search query.
        Returns a list of result dicts, or [] on failure.
        """
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": True,
        }

        try:
            response = httpx.post(TAVILY_SEARCH_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            logger.debug(f"Tavily query '{query}' returned {len(results)} results.")
            return results
        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily HTTP error for query '{query}': {e.response.status_code} — {e.response.text}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Tavily request error for query '{query}': {e}")
            return []
        except Exception as e:
            logger.error(f"Tavily unexpected error for query '{query}': {e}")
            return []
