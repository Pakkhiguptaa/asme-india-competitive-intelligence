"""
notion_client.py — Notion API integration for persistent signal storage.

Handles:
- Writing new signals as database pages
- Querying existing signal_hashes to prevent duplicates
- Marking signals as new vs existing
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NOTION_API_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"


class NotionClient:
    def __init__(self, api_key: str, database_id: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        }
        self.database_id = database_id

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def get_existing_hashes(self) -> set[str]:
        """
        Query Notion for all existing signal_hash values.
        Used to detect duplicates before writing.
        """
        hashes: set[str] = set()
        has_more = True
        start_cursor = None

        while has_more:
            payload: dict[str, Any] = {
                "page_size": 100,
            }
            if start_cursor:
                payload["start_cursor"] = start_cursor

            try:
                response = httpx.post(
                    f"{NOTION_BASE_URL}/databases/{self.database_id}/query",
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                for page in data.get("results", []):
                    props = page.get("properties", {})
                    h = _extract_rich_text(props, "signal_hash")
                    if h:
                        hashes.add(h)

                has_more = data.get("has_more", False)
                start_cursor = data.get("next_cursor")

            except Exception as e:
                logger.error(f"Notion query error: {e}")
                break

        logger.info(f"Notion: found {len(hashes)} existing signal hashes.")
        return hashes

    def write_signals(self, signals: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Write signals to Notion. Skip signals whose hash already exists.
        Returns (written_count, skipped_count).
        """
        existing_hashes = self.get_existing_hashes()
        written = 0
        skipped = 0

        for signal in signals:
            signal_hash = signal.get("signal_hash", "")
            if signal_hash in existing_hashes:
                logger.debug(f"Skipping existing signal: {signal.get('title', '')[:60]}")
                skipped += 1
                continue

            success = self._create_page(signal)
            if success:
                written += 1
                existing_hashes.add(signal_hash)  # prevent duplicates within same batch
            else:
                logger.warning(f"Failed to write: {signal.get('title', '')[:60]}")

        logger.info(f"Notion write: {written} new, {skipped} skipped.")
        return written, skipped

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _create_page(self, signal: dict[str, Any]) -> bool:
        """Create a single Notion page for a signal."""
        properties = _build_properties(signal)
        payload = {
            "parent": {"database_id": self.database_id},
            "properties": properties,
        }

        try:
            response = httpx.post(
                f"{NOTION_BASE_URL}/pages",
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Notion page create HTTP error: {e.response.status_code} "
                f"— {e.response.text[:200]}"
            )
            return False
        except Exception as e:
            logger.error(f"Notion page create error: {e}")
            return False


# ------------------------------------------------------------------ #
# Property builders                                                    #
# ------------------------------------------------------------------ #

def _build_properties(s: dict[str, Any]) -> dict[str, Any]:
    """Map signal fields to Notion property format."""
    return {
        # Title property — required
        "title": _title(s.get("title", "(no title)")),

        # Identifiers
        "run_id": _rich_text(s.get("run_id", "")),
        "signal_hash": _rich_text(s.get("signal_hash", "")),
        "run_timestamp": _rich_text(s.get("run_timestamp", "")),

        # Competitor
        "competitor": _select(s.get("competitor", "")),
        "competitor_full_name": _rich_text(s.get("competitor_full_name", "")),

        # Signal metadata
        "signal_type": _select(s.get("signal_type", "other")),
        "geography": _select(s.get("geography", "unknown")),
        "business_impact": _select(s.get("business_impact", "")),
        "urgency": _select(s.get("urgency", "low")),
        "revenue_relevance": _select(s.get("revenue_relevance", "weak")),

        # Scores
        "importance_score": _number(s.get("importance_score", 0)),
        "confidence_score": _number(s.get("confidence_score", 0)),
        "composite_score": _number(s.get("composite_score", 0)),

        # Decision fields
        "summary": _rich_text(s.get("summary", "")),
        "why_it_matters": _rich_text(s.get("why_it_matters", "")),
        "recommended_action": _rich_text(s.get("recommended_action", "")),

        # Source
        "url": _url(s.get("url", "")),
        "source": _rich_text(s.get("source", "")),
        "published_date": _rich_text(s.get("published_date", "")),
        "search_query": _rich_text(s.get("search_query", "")),
        "snippet": _rich_text(s.get("snippet", "")[:2000]),

        # Status
        "is_new": _checkbox(s.get("is_new", True)),
    }


def _title(text: str) -> dict:
    return {"title": [{"text": {"content": str(text)[:2000]}}]}

def _rich_text(text: str) -> dict:
    return {"rich_text": [{"text": {"content": str(text)[:2000]}}]}

def _select(value: str) -> dict:
    return {"select": {"name": str(value)}}

def _number(value: Any) -> dict:
    try:
        return {"number": float(value)}
    except (TypeError, ValueError):
        return {"number": 0}

def _url(value: str) -> dict:
    return {"url": str(value) if value else None}

def _checkbox(value: bool) -> dict:
    return {"checkbox": bool(value)}

def _extract_rich_text(props: dict, key: str) -> str:
    """Extract text from a Notion rich_text property."""
    try:
        rt = props.get(key, {}).get("rich_text", [])
        return rt[0]["text"]["content"] if rt else ""
    except (IndexError, KeyError, TypeError):
        return ""
