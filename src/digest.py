"""
digest.py — Formats output for daily run and weekly digest.

All outputs are decision-ready, not merely descriptive.
"""

from typing import Any
from datetime import datetime


# ------------------------------------------------------------------ #
# Daily run output                                                     #
# ------------------------------------------------------------------ #

def format_daily_output(top_signals: list[dict[str, Any]], run_id: str, run_timestamp: str) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("ASME INDIA — COMPETITIVE INTELLIGENCE DAILY RUN")
    lines.append(f"Run ID   : {run_id}")
    lines.append(f"Timestamp: {run_timestamp}")
    lines.append(f"Top signals shown: {len(top_signals)}")
    lines.append("=" * 72)

    if not top_signals:
        lines.append("\nNo new signals detected in this run.")
        return "\n".join(lines)

    for i, s in enumerate(top_signals, 1):
        lines.append(f"\n{'─' * 72}")
        lines.append(f"#{i}  [{s.get('competitor', '')}]  {s.get('title', '')}")
        lines.append(f"{'─' * 72}")

        lines.append(f"  Signal Type    : {s.get('signal_type', '')}")
        lines.append(f"  Business Impact: {s.get('business_impact', '')}")
        lines.append(f"  Urgency        : {_urgency_label(s.get('urgency', ''))}")
        lines.append(f"  Revenue Risk   : {s.get('revenue_relevance', '')}")
        lines.append(f"  Geography      : {s.get('geography', '')}")
        lines.append(f"  Importance     : {s.get('importance_score', 0)}/100  "
                     f"(composite: {s.get('composite_score', 0)})")
        lines.append(f"  Published      : {s.get('published_date', 'unknown')}")
        lines.append(f"  Source         : {s.get('source', '')}")
        lines.append(f"  URL            : {s.get('url', '')}")
        lines.append("")
        lines.append(f"  WHAT HAPPENED")
        lines.append(f"  {s.get('summary', '')}")
        lines.append("")
        lines.append(f"  WHY IT MATTERS TO ASME")
        lines.append(f"  {s.get('why_it_matters', '')}")
        lines.append("")
        lines.append(f"  RECOMMENDED ACTION")
        lines.append(f"  >> {s.get('recommended_action', '')}")

    lines.append(f"\n{'=' * 72}")
    lines.append(f"End of daily run. {len(top_signals)} actionable signals above.")
    lines.append("=" * 72)

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Weekly digest output                                                 #
# ------------------------------------------------------------------ #

def format_weekly_digest(signals: list[dict[str, Any]]) -> str:
    """
    Generate a weekly digest from all signals stored this week.
    Groups by competitor, surfaces patterns and revenue-risk signals.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("ASME INDIA — WEEKLY COMPETITIVE INTELLIGENCE DIGEST")
    lines.append(f"Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Total signals analysed: {len(signals)}")
    lines.append("=" * 72)

    if not signals:
        lines.append("\nNo signals found for this period.")
        return "\n".join(lines)

    # --- Section 1: Biggest moves per competitor ---
    lines.append("\n## BIGGEST COMPETITOR MOVES\n")
    by_competitor: dict[str, list[dict]] = {}
    for s in signals:
        c = s.get("competitor", "Unknown")
        by_competitor.setdefault(c, []).append(s)

    for competitor, comp_signals in by_competitor.items():
        top = sorted(comp_signals, key=lambda x: x.get("composite_score", 0), reverse=True)[:3]
        lines.append(f"  [{competitor}] — {len(comp_signals)} signals this week")
        for s in top:
            lines.append(f"    • {s.get('title', '')[:80]}")
            lines.append(f"      {s.get('summary', '')}")
            lines.append(f"      >> {s.get('recommended_action', '')}")
        lines.append("")

    # --- Section 2: Cross-competitor patterns ---
    lines.append("## PATTERNS ACROSS COMPETITORS\n")
    pattern_counts: dict[str, int] = {}
    for s in signals:
        st = s.get("signal_type", "other")
        pattern_counts[st] = pattern_counts.get(st, 0) + 1

    for signal_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {signal_type:<22} {count} signals")
    lines.append("")

    # --- Section 3: High revenue-risk signals ---
    lines.append("## HIGHEST REVENUE-RISK SIGNALS\n")
    high_risk = [
        s for s in signals
        if s.get("revenue_relevance") in ("direct", "indirect")
        and s.get("urgency") in ("high", "medium")
    ]
    high_risk_sorted = sorted(high_risk, key=lambda x: x.get("composite_score", 0), reverse=True)[:5]

    if high_risk_sorted:
        for s in high_risk_sorted:
            lines.append(f"  [{s.get('competitor', '')}] {s.get('title', '')[:70]}")
            lines.append(f"  Revenue Risk: {s.get('revenue_relevance', '')}  |  "
                         f"Urgency: {s.get('urgency', '')}  |  "
                         f"Impact: {s.get('business_impact', '')}")
            lines.append(f"  Why: {s.get('why_it_matters', '')}")
            lines.append(f"  Action: >> {s.get('recommended_action', '')}")
            lines.append(f"  URL: {s.get('url', '')}")
            lines.append("")
    else:
        lines.append("  No high revenue-risk signals detected this week.\n")

    # --- Section 4: Recommended actions summary ---
    lines.append("## RECOMMENDED ACTIONS FOR ASME INDIA\n")
    high_urgency = [s for s in signals if s.get("urgency") == "high"]
    if high_urgency:
        for s in sorted(high_urgency, key=lambda x: x.get("composite_score", 0), reverse=True)[:5]:
            lines.append(f"  [URGENT — {s.get('competitor', '')}]")
            lines.append(f"  >> {s.get('recommended_action', '')}")
            lines.append("")
    else:
        lines.append("  No high-urgency actions required this week.\n")

    lines.append("=" * 72)
    lines.append("End of weekly digest.")
    lines.append("=" * 72)

    return "\n".join(lines)


def _urgency_label(urgency: str) -> str:
    labels = {"high": "HIGH ⚠", "medium": "MEDIUM", "low": "low"}
    return labels.get(urgency, urgency)
