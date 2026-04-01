"""
constants.py — All hardcoded domain constants.
No business logic here — just data.
"""

COMPETITORS = [
    {"name": "BIS", "full_name": "Bureau of Indian Standards"},
    {"name": "IEI", "full_name": "Institution of Engineers India"},
    {"name": "IMechE", "full_name": "Institution of Mechanical Engineers"},
    {"name": "SAE India", "full_name": "SAE India"},
]

SIGNAL_THEMES = [
    "standards or regulatory shifts",
    "new training or certification pushes",
    "student chapter expansion",
    "university tie-ups",
    "industrial partnerships",
    "regional expansion activity",
]

# Hardcoded search queries per competitor
SEARCH_QUERIES: dict[str, list[str]] = {
    "BIS": [
        "Bureau of Indian Standards standards update India mechanical",
        "Bureau of Indian Standards regulatory update India manufacturing",
        "BIS pressure vessel standards India",
        "BIS industrial standards partnership India",
        "BIS regional office expansion India",
        "BIS certification training India",
    ],
    "IEI": [
        "Institution of Engineers India training certification mechanical",
        "Institution of Engineers India student chapter expansion",
        "Institution of Engineers India university partnership",
        "Institution of Engineers India industrial partnership",
        "Institution of Engineers India regional expansion",
        "Institution of Engineers India mechanical engineering event",
    ],
    "IMechE": [
        "Institution of Mechanical Engineers India training certification",
        "Institution of Mechanical Engineers India university partnership",
        "Institution of Mechanical Engineers India student chapter",
        "Institution of Mechanical Engineers India industrial partnership",
        "Institution of Mechanical Engineers India India expansion",
        "IMechE India mechanical engineering program",
    ],
    "SAE India": [
        "SAE India training certification",
        "SAE India student chapter expansion",
        "SAE India university partnership",
        "SAE India industrial partnership",
        "SAE India regional expansion",
        "SAE India automotive engineering program",
    ],
}

SIGNAL_TYPES = [
    "regulatory",
    "partnership",
    "event",
    "pricing",
    "hiring",
    "campaign",
    "student_outreach",
    "certification",
    "expansion",
    "other",
]

GEOGRAPHIES = [
    "India national",
    "state",
    "city",
    "campus",
    "unknown",
]

BUSINESS_IMPACTS = [
    "standards",
    "membership",
    "training",
    "student pipeline",
    "brand",
    "mixed",
]

URGENCY_LEVELS = ["low", "medium", "high"]
REVENUE_RELEVANCE = ["direct", "indirect", "weak"]
