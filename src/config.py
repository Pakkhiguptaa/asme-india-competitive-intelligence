"""
config.py — Load and validate all environment variables.
Fails clearly if any required key is missing.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"[CONFIG ERROR] Required environment variable '{key}' is missing or empty.\n"
            f"Set it in your .env file or shell environment before running."
        )
    return value


class Config:
    LLM_API_KEY: str = _require("LLM_API_KEY")
    TAVILY_API_KEY: str = _require("TAVILY_API_KEY")
    NOTION_API_KEY: str = _require("NOTION_API_KEY")
    NOTION_DATABASE_ID: str = _require("NOTION_DATABASE_ID")

    # LLM settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "anthropic" or "openai"
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")  # custom base URL (e.g. GitHub Models)

    # Tavily settings
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")

    # Daily run settings
    TOP_SIGNALS_COUNT: int = int(os.getenv("TOP_SIGNALS_COUNT", "10"))
