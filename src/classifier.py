"""
classifier.py — LLM-based classification of normalized signals.

Sends each signal to the LLM with a structured prompt.
Returns classification fields merged back into the signal dict.
"""

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a competitive intelligence analyst for ASME India.

ASME India competes in the following areas:
- Engineering standards and certification
- Professional membership and development
- Technical training programs
- Student chapter and pipeline development
- University and industrial partnerships

You will receive a news signal about a competitor. Classify it strictly based on evidence in the text.
Focus on business impact to ASME India, not surface-level description.

Prioritize classification weight toward:
- standards/regulatory shifts (highest threat)
- training/certification moves
- student pipeline growth
- industrial partnerships
- geographic expansion

If the signal is weak, generic, or off-topic, mark revenue_relevance as "weak" and urgency as "low".

You MUST return ONLY valid JSON — no explanation, no markdown, no extra text.

Classify the following signal:

Competitor: {competitor} ({competitor_full_name})
Title: {title}
Source: {source}
Published: {published_date}
Snippet: {snippet}

Return this exact JSON structure:
{{
  "signal_type": "<one of: regulatory|partnership|event|pricing|hiring|campaign|student_outreach|certification|expansion|other>",
  "geography": "<one of: India national|state|city|campus|unknown>",
  "business_impact": "<one of: standards|membership|training|student pipeline|brand|mixed>",
  "urgency": "<one of: low|medium|high>",
  "revenue_relevance": "<one of: direct|indirect|weak>",
  "summary": "<1-2 sentence factual description of what happened>",
  "why_it_matters": "<1-2 sentences on why this is relevant to ASME India's business>",
  "recommended_action": "<one clear, specific action ASME India should consider>",
  "confidence_score": <integer 0-100>,
  "importance_score": <integer 0-100>
}}"""


def classify_signals(
    signals: list[dict[str, Any]],
    llm_api_key: str,
    llm_model: str,
    llm_provider: str,
    llm_base_url: str = "",
) -> list[dict[str, Any]]:
    """
    Classify each signal using the LLM.
    Returns the signals list with classification fields merged in.
    Failed classifications get default low-confidence values.
    """
    client = _build_llm_client(llm_provider, llm_api_key, llm_model, llm_base_url)
    classified = []

    for i, signal in enumerate(signals):
        logger.info(f"Classifying [{i+1}/{len(signals)}]: {signal.get('title', '')[:60]}")
        classification = _classify_one(signal, client, llm_provider, llm_model)
        classified.append({**signal, **classification})

        # Respectful rate limiting — 1 call per second
        if i < len(signals) - 1:
            time.sleep(1)

    return classified


def _classify_one(
    signal: dict[str, Any],
    client: Any,
    provider: str,
    model: str,
) -> dict[str, Any]:
    """Classify a single signal. Returns classification dict."""
    prompt = CLASSIFICATION_PROMPT.format(
        competitor=signal.get("competitor", ""),
        competitor_full_name=signal.get("competitor_full_name", ""),
        title=signal.get("title", ""),
        source=signal.get("source", ""),
        published_date=signal.get("published_date", "unknown"),
        snippet=signal.get("snippet", "")[:800],
    )

    try:
        raw_text = _call_llm(client, provider, model, prompt)
        return _parse_classification(raw_text)
    except Exception as e:
        logger.error(f"Classification error for '{signal.get('title', '')}': {e}")
        return _default_classification()


def _call_llm(client: Any, provider: str, model: str, prompt: str) -> str:
    """Call the LLM and return the raw text response."""
    if provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    elif provider == "openai":
        response = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _parse_classification(raw_text: str) -> dict[str, Any]:
    """Parse LLM response into classification dict. Raises on parse failure."""
    # Strip markdown code fences if the model wraps the JSON
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    data = json.loads(text)

    # Validate required fields exist
    required = [
        "signal_type", "geography", "business_impact", "urgency",
        "revenue_relevance", "summary", "why_it_matters",
        "recommended_action", "confidence_score", "importance_score",
    ]
    for field in required:
        if field not in data:
            raise ValueError(f"LLM response missing field: {field}")

    return data


def _default_classification() -> dict[str, Any]:
    """Fallback classification when LLM fails."""
    return {
        "signal_type": "other",
        "geography": "unknown",
        "business_impact": "brand",
        "urgency": "low",
        "revenue_relevance": "weak",
        "summary": "Classification failed — review manually.",
        "why_it_matters": "Unknown.",
        "recommended_action": "Review signal manually.",
        "confidence_score": 0,
        "importance_score": 0,
    }


def _build_llm_client(provider: str, api_key: str, model: str, base_url: str = "") -> Any:
    """Build and return the appropriate LLM client."""
    if provider == "anthropic":
        try:
            import anthropic
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    elif provider == "openai":
        try:
            import openai
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return openai.OpenAI(**kwargs)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai in your .env"
        )
