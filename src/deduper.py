"""
deduper.py — Deduplication before classification and before Notion write.

Two layers:
1. Within-run: deduplicate the batch before sending to LLM.
2. Cross-run: compare against signal_hashes already stored in Notion.
"""

import hashlib
import logging
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)


def compute_signal_hash(competitor: str, title: str, url: str) -> str:
    """Stable hash: competitor + title + url (lowercased, stripped)."""
    raw = f"{competitor.lower().strip()}|{title.lower().strip()}|{url.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def deduplicate_within_run(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicates within a single run's result set.
    Deduplication order of priority:
    1. Exact URL match
    2. Title similarity >= 85% for the same competitor
    """
    seen_urls: set[str] = set()
    kept: list[dict[str, Any]] = []

    for signal in signals:
        url = signal.get("url", "").strip().lower()
        title = signal.get("title", "").strip().lower()
        competitor = signal.get("competitor", "")

        # Exact URL dedupe
        if url in seen_urls:
            logger.debug(f"Deduped (URL): {signal.get('title')}")
            continue

        # Title similarity dedupe (same competitor)
        is_near_duplicate = False
        for existing in kept:
            if existing.get("competitor") != competitor:
                continue
            existing_title = existing.get("title", "").strip().lower()
            ratio = SequenceMatcher(None, title, existing_title).ratio()
            if ratio >= 0.85:
                logger.debug(f"Deduped (title sim {ratio:.2f}): {signal.get('title')}")
                is_near_duplicate = True
                break

        if is_near_duplicate:
            continue

        seen_urls.add(url)
        kept.append(signal)

    removed = len(signals) - len(kept)
    if removed:
        logger.info(f"Within-run dedup: removed {removed} duplicates, kept {len(kept)}.")

    return kept


def mark_new_vs_existing(
    signals: list[dict[str, Any]],
    existing_hashes: set[str],
) -> list[dict[str, Any]]:
    """
    Attach 'is_new' flag to each signal based on whether its hash
    already exists in Notion.
    """
    for signal in signals:
        h = signal.get("signal_hash", "")
        signal["is_new"] = h not in existing_hashes
    return signals
