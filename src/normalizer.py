"""
normalizer.py — Converts raw Tavily results into a consistent schema.
No classification here — just clean structured data.
"""

import uuid
from datetime import datetime, timezone
from typing import Any


def normalize(raw_results: list[dict[str, Any]], run_id: str, run_timestamp: str) -> list[dict[str, Any]]:
    """
    Normalize a list of raw Tavily results into the standard signal schema.
    """
    normalized = []

    for raw in raw_results:
        # Tavily result fields
        title = (raw.get("title") or "").strip()
        url = (raw.get("url") or "").strip()
        snippet = (raw.get("content") or "").strip()
        extracted_text = (raw.get("raw_content") or "").strip()
        published_date = raw.get("published_date") or ""
        source = _extract_source(url)

        # Skip results with no URL or title — not actionable
        if not url or not title:
            continue

        normalized.append({
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "competitor": raw.get("_competitor", ""),
            "competitor_full_name": raw.get("_competitor_full_name", ""),
            "search_query": raw.get("_search_query", ""),
            "title": title,
            "url": url,
            "published_date": published_date,
            "source": source,
            "snippet": snippet[:1000],  # cap at 1000 chars for LLM efficiency
            "extracted_text": extracted_text[:3000],  # cap raw text
        })

    return normalized


def _extract_source(url: str) -> str:
    """Extract domain name from URL as the source label."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lstrip("www.")
    except Exception:
        return ""
