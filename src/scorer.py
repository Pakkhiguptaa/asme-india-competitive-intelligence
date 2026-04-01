"""
scorer.py — Post-classification scoring and ranking.

The LLM provides importance_score and confidence_score.
This module applies a final composite rank for output ordering.
"""

from typing import Any

# Urgency multiplier weights
URGENCY_WEIGHT = {"high": 1.0, "medium": 0.65, "low": 0.3}

# Revenue relevance multiplier weights
REVENUE_WEIGHT = {"direct": 1.0, "indirect": 0.6, "weak": 0.2}


def compute_composite_score(signal: dict[str, Any]) -> float:
    """
    Composite score = importance_score * urgency_weight * revenue_weight
    Scaled to 0–100.
    """
    importance = float(signal.get("importance_score", 0))
    urgency = URGENCY_WEIGHT.get(signal.get("urgency", "low"), 0.3)
    revenue = REVENUE_WEIGHT.get(signal.get("revenue_relevance", "weak"), 0.2)
    return round(importance * urgency * revenue, 2)


def rank_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Attach composite_score to each signal and return sorted descending.
    Only NEW signals are ranked for the daily digest.
    """
    for signal in signals:
        signal["composite_score"] = compute_composite_score(signal)

    return sorted(signals, key=lambda s: s["composite_score"], reverse=True)


def top_new_signals(signals: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:
    """Return the top N new signals by composite score."""
    new_signals = [s for s in signals if s.get("is_new", True)]
    ranked = rank_signals(new_signals)
    return ranked[:n]
