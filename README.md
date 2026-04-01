# ASME India Competitive Intelligence System

A production-ready CLI tool that tracks competitor moves in India daily, classifies them using an LLM, stores results in Notion, and surfaces decision-ready outputs.

---

## What It Does

- Runs 24 hardcoded Tavily searches across 4 competitors (BIS, IEI, IMechE, SAE India)
- Normalizes results into a consistent schema
- Deduplicates within each run and against Notion history
- Classifies each signal using an LLM (urgency, business impact, revenue relevance, recommended action)
- Stores all signals in a Notion database
- Outputs the top 5–10 new signals with decision-ready analysis
- Generates a weekly digest of patterns, revenue risks, and recommended actions

---

## Setup

### 1. Clone / navigate to the project

```bash
cd asme-intel
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `anthropic` (default) or `openai` |
| `LLM_API_KEY` | Your Anthropic or OpenAI API key |
| `LLM_MODEL` | e.g. `claude-opus-4-6` or `gpt-4o` |
| `TAVILY_API_KEY` | Your Tavily API key (get one at tavily.com) |
| `NOTION_API_KEY` | Your Notion integration token |
| `NOTION_DATABASE_ID` | ID of your Notion signals database |

### 4. Set up the Notion database

Create a Notion integration at [notion.so/my-integrations](https://www.notion.so/my-integrations).

Create a new **full-page database** in Notion and connect your integration to it.

Add the following properties (exact names and types matter):

| Property Name | Type |
|---|---|
| `title` | Title (default) |
| `run_id` | Text |
| `signal_hash` | Text |
| `run_timestamp` | Text |
| `competitor` | Select |
| `competitor_full_name` | Text |
| `signal_type` | Select |
| `geography` | Select |
| `business_impact` | Select |
| `urgency` | Select |
| `revenue_relevance` | Select |
| `importance_score` | Number |
| `confidence_score` | Number |
| `composite_score` | Number |
| `summary` | Text |
| `why_it_matters` | Text |
| `recommended_action` | Text |
| `url` | URL |
| `source` | Text |
| `published_date` | Text |
| `search_query` | Text |
| `snippet` | Text |
| `is_new` | Checkbox |

**Get the database ID:**
Open your database as a full page in Notion. The URL looks like:
`https://www.notion.so/yourworkspace/abc123def456...?v=...`
The database ID is the 32-character string before the `?v=` — copy it into `NOTION_DATABASE_ID` in your `.env`.

---

## Running the System

### Daily run (retrieve, classify, store, output)

```bash
python -m src.main run-daily
```

This will:
1. Run all 24 Tavily queries
2. Normalize and deduplicate results
3. Classify each signal using the LLM
4. Write new signals to Notion
5. Print the top 10 new signals with recommended actions

### Weekly digest

```bash
python -m src.main run-weekly
```

This will:
1. Pull all signals from your Notion database
2. Analyse patterns across competitors
3. Surface highest revenue-risk signals
4. Print a structured digest with recommended actions for ASME India

---

## Scheduling Daily Runs

To run automatically each morning (macOS/Linux cron):

```bash
crontab -e
```

Add:
```
0 7 * * * cd /path/to/asme-intel && /path/to/venv/bin/python -m src.main run-daily >> logs/daily.log 2>&1
```

---

## Project Structure

```
asme-intel/
├── src/
│   ├── __init__.py
│   ├── main.py           # CLI entrypoint
│   ├── config.py         # Env var loading and validation
│   ├── constants.py      # Hardcoded competitors and queries
│   ├── tavily_client.py  # Tavily API wrapper
│   ├── retriever.py      # Runs all queries, returns raw results
│   ├── normalizer.py     # Converts raw → standard schema
│   ├── deduper.py        # Within-run and cross-run deduplication
│   ├── classifier.py     # LLM classification
│   ├── scorer.py         # Composite scoring and ranking
│   ├── notion_client.py  # Notion read/write integration
│   └── digest.py         # Daily and weekly output formatting
├── requirements.txt
├── .env.example
└── README.md
```

---

## Error Handling

| Error | Behaviour |
|---|---|
| Missing env var | Exits immediately with a clear message |
| Tavily API failure | Logs warning, skips that query, continues |
| LLM JSON parse error | Falls back to low-confidence default classification |
| Notion write failure | Logs error per signal, continues remaining writes |
| Empty results | Logs warning and exits cleanly |

---

## Notes

- LinkedIn is excluded from all searches by design.
- Deduplication uses a stable SHA-256 hash of `competitor + title + url`.
- The LLM is called once per signal — set `TAVILY_MAX_RESULTS=3` to reduce API costs during testing.
- Weekly digest reads from Notion — run `run-daily` at least once first.
