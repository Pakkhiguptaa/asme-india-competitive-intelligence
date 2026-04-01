"""
dashboard.py — Decision-first intelligence dashboard for ASME India.

Run: python -m src.dashboard
Opens at: http://localhost:5050
"""

import os
import uuid
import threading
import time
import httpx
import logging
from flask import Flask, jsonify, render_template_string, Response, stream_with_context, send_from_directory
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
app = Flask(__name__, static_folder=os.path.normpath(_STATIC_DIR), static_url_path="/static")

# ------------------------------------------------------------------ #
# Pipeline run state (in-memory, single-user local tool)              #
# ------------------------------------------------------------------ #

_run_state: dict = {
    "status": "idle",   # idle | running | done | error
    "steps": [],        # list of step message strings
    "run_id": None,
    "new_signals": 0,
    "total_signals": 0,
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_run_lock = threading.Lock()


def _push_step(msg: str):
    with _run_lock:
        _run_state["steps"].append(msg)
    logger.info(f"[pipeline] {msg}")


def _run_pipeline_thread():
    """Run the full daily pipeline in a background thread, pushing status updates."""
    with _run_lock:
        _run_state.update({
            "status": "running",
            "steps": [],
            "run_id": None,
            "new_signals": 0,
            "total_signals": 0,
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        })

    try:
        from .config import Config
        from .tavily_client import TavilyClient
        from .retriever import retrieve_all
        from .normalizer import normalize
        from .deduper import compute_signal_hash, deduplicate_within_run, mark_new_vs_existing
        from .classifier import classify_signals
        from .scorer import rank_signals, top_new_signals
        from .notion_client import NotionClient

        run_id = str(uuid.uuid4())[:8]
        from datetime import datetime, timezone
        run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with _run_lock:
            _run_state["run_id"] = run_id

        _push_step(f"Started run {run_id}")

        # Step 1: Retrieve
        _push_step("Searching Tavily — 24 queries across 4 competitors...")
        tavily = TavilyClient(
            api_key=Config.TAVILY_API_KEY,
            max_results=Config.TAVILY_MAX_RESULTS,
            search_depth=Config.TAVILY_SEARCH_DEPTH,
        )
        raw_results = retrieve_all(tavily)
        _push_step(f"Retrieved {len(raw_results)} raw results from Tavily")

        if not raw_results:
            _push_step("No results returned — check Tavily API key")
            with _run_lock:
                _run_state["status"] = "done"
                _run_state["finished_at"] = time.time()
            return

        # Step 2: Normalize
        _push_step("Normalizing results...")
        signals = normalize(raw_results, run_id, run_timestamp)
        _push_step(f"Normalized to {len(signals)} valid signals")

        # Step 3: Deduplicate
        _push_step("Deduplicating within run...")
        signals = deduplicate_within_run(signals)
        for signal in signals:
            signal["signal_hash"] = compute_signal_hash(
                signal.get("competitor", ""), signal.get("title", ""), signal.get("url", "")
            )
        _push_step(f"{len(signals)} unique signals after deduplication")

        # Step 4: Classify
        _push_step(f"Classifying {len(signals)} signals with LLM (this takes a few minutes)...")
        signals = classify_signals(
            signals,
            llm_api_key=Config.LLM_API_KEY,
            llm_model=Config.LLM_MODEL,
            llm_provider=Config.LLM_PROVIDER,
            llm_base_url=Config.LLM_BASE_URL,
        )
        _push_step("LLM classification complete")

        # Step 5: Store
        _push_step("Checking Notion for existing signals...")
        notion = NotionClient(api_key=Config.NOTION_API_KEY, database_id=Config.NOTION_DATABASE_ID)
        existing_hashes = notion.get_existing_hashes()
        signals = mark_new_vs_existing(signals, existing_hashes)
        signals = rank_signals(signals)

        new_count = sum(1 for s in signals if s.get("is_new"))
        _push_step(f"Writing {new_count} new signals to Notion...")
        written, skipped = notion.write_signals(signals)
        _push_step(f"Notion: {written} written, {skipped} already existed")

        with _run_lock:
            _run_state["status"] = "done"
            _run_state["new_signals"] = new_count
            _run_state["total_signals"] = len(signals)
            _run_state["finished_at"] = time.time()

        _push_step(f"Done — {new_count} new signals added. Refresh the feed to see them.")

    except Exception as e:
        logger.exception("Pipeline error")
        with _run_lock:
            _run_state["status"] = "error"
            _run_state["error"] = str(e)
            _run_state["finished_at"] = time.time()
        _push_step(f"Error: {e}")

NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# ------------------------------------------------------------------ #
# Data fetching                                                        #
# ------------------------------------------------------------------ #

def fetch_signals():
    signals = []
    has_more = True
    start_cursor = None

    while has_more:
        payload = {
            "page_size": 100,
            "sorts": [{"property": "composite_score", "direction": "descending"}],
        }
        if start_cursor:
            payload["start_cursor"] = start_cursor

        resp = httpx.post(
            f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query",
            headers=NOTION_HEADERS,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for page in data.get("results", []):
            signals.append(_parse_page(page))

        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")

    return signals


def _parse_page(page):
    p = page.get("properties", {})

    def rt(key):
        try:
            return p[key]["rich_text"][0]["text"]["content"]
        except Exception:
            return ""

    def ttl():
        try:
            return p["title"]["title"][0]["text"]["content"]
        except Exception:
            return ""

    def sel(key):
        try:
            return p[key]["select"]["name"]
        except Exception:
            return ""

    def num(key):
        try:
            return p[key]["number"] or 0
        except Exception:
            return 0

    def url_val():
        try:
            return p["url"]["url"] or ""
        except Exception:
            return ""

    def chk(key):
        try:
            return p[key]["checkbox"]
        except Exception:
            return False

    return {
        "title": ttl(),
        "competitor": sel("competitor"),
        "signal_type": sel("signal_type"),
        "business_impact": sel("business_impact"),
        "urgency": sel("urgency"),
        "revenue_relevance": sel("revenue_relevance"),
        "geography": sel("geography"),
        "importance_score": num("importance_score"),
        "confidence_score": num("confidence_score"),
        "composite_score": num("composite_score"),
        "summary": rt("summary"),
        "why_it_matters": rt("why_it_matters"),
        "recommended_action": rt("recommended_action"),
        "url": url_val(),
        "source": rt("source"),
        "published_date": rt("published_date"),
        "run_id": rt("run_id"),
        "run_timestamp": rt("run_timestamp"),
        "is_new": chk("is_new"),
    }


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/test")
def api_test():
    return jsonify({"ok": True, "notion_db": NOTION_DATABASE_ID[:8] + "...", "notion_key_set": bool(NOTION_API_KEY)})


@app.route("/api/signals")
def api_signals():
    try:
        signals = fetch_signals()
        return jsonify({"signals": signals, "total": len(signals)})
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        return jsonify({"error": str(e), "signals": [], "total": 0}), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start a fresh pipeline run in a background thread."""
    with _run_lock:
        if _run_state["status"] == "running":
            return jsonify({"error": "A run is already in progress"}), 409
    t = threading.Thread(target=_run_pipeline_thread, daemon=True)
    t.start()
    return jsonify({"started": True})


@app.route("/api/run/status")
def api_run_status():
    """SSE stream — pushes pipeline step messages to the browser in real time."""
    def generate():
        seen = 0
        while True:
            with _run_lock:
                state  = _run_state["status"]
                steps  = _run_state["steps"]
                error  = _run_state["error"]
                new_s  = _run_state["new_signals"]
                total  = _run_state["total_signals"]

            # Send any new steps
            while seen < len(steps):
                msg = steps[seen]
                yield f"data: {msg}\n\n"
                seen += 1

            if state == "done":
                yield f"event: done\ndata: {new_s} new signals | {total} total\n\n"
                break
            if state == "error":
                yield f"event: error\ndata: {error or 'Unknown error'}\n\n"
                break

            time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ #
# Dashboard HTML                                                       #
# ------------------------------------------------------------------ #

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ASME India — Competitive Intelligence</title>
<style>
/* ── Reset & tokens ─────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0e14;
  --surface:#13151f;
  --surface2:#1a1d2e;
  --surface3:#21253a;
  --border:#252840;
  --border2:#2e3355;
  --text:#eceef5;
  --text2:#9ba3c0;
  --text3:#5c6380;
  --accent:#5b8af0;
  --accent-dim:rgba(91,138,240,.12);

  --risk-high:#ef4444;
  --risk-high-dim:rgba(239,68,68,.1);
  --risk-med:#f59e0b;
  --risk-med-dim:rgba(245,158,11,.1);
  --risk-low:#10b981;
  --risk-low-dim:rgba(16,185,129,.1);

  --bis:#3b82f6;
  --iei:#10b981;
  --imech:#f59e0b;
  --sae:#8b5cf6;

  --radius:10px;
  --radius-sm:6px;
}
html{font-size:14px;-webkit-font-smoothing:antialiased}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif;min-height:100vh}
a{color:inherit;text-decoration:none}
button{cursor:pointer;font-family:inherit}

/* ── Layout ─────────────────────────────────────────────── */
.shell{display:grid;grid-template-columns:240px 1fr;min-height:100vh}
.sidebar{background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.main{overflow:hidden}

/* ── Sidebar ─────────────────────────────────────────────── */
.sidebar-logo{padding:20px 20px 16px;border-bottom:1px solid var(--border)}
.sidebar-logo .org{font-size:11px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--text3);margin-bottom:4px}
.sidebar-logo .product{font-size:15px;font-weight:700;color:var(--text)}

.sidebar-section{padding:20px 20px 0}
.sidebar-section-label{font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--text3);margin-bottom:12px}

/* Risk donut */
.donut-wrap{display:flex;align-items:center;gap:16px;padding-bottom:20px;border-bottom:1px solid var(--border)}
.donut-svg{flex-shrink:0}
.donut-legend{display:flex;flex-direction:column;gap:8px}
.legend-item{display:flex;align-items:center;gap:8px}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.legend-text{font-size:12px;color:var(--text2)}
.legend-count{margin-left:auto;font-size:12px;font-weight:600;color:var(--text)}

/* Competitor strip */
.competitor-list{display:flex;flex-direction:column;gap:4px;padding-bottom:20px;border-bottom:1px solid var(--border)}
.competitor-item{display:flex;align-items:center;gap:10px;padding:9px 10px;border-radius:var(--radius-sm);cursor:pointer;transition:background .15s;border:1px solid transparent}
.competitor-item:hover{background:var(--surface2)}
.competitor-item.active{background:var(--accent-dim);border-color:rgba(91,138,240,.2)}
.comp-logo{width:30px;height:30px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;letter-spacing:-.5px;flex-shrink:0}
.comp-info{flex:1;min-width:0}
.comp-name{font-size:12px;font-weight:600;color:var(--text)}
.comp-sub{font-size:11px;color:var(--text3);margin-top:1px}
.comp-badge{font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px}

/* Sidebar nav */
.sidebar-nav{padding:12px 12px;margin-top:auto;border-top:1px solid var(--border)}
.sidebar-nav .ts{font-size:11px;color:var(--text3);padding:8px 8px}
.refresh-btn{width:100%;padding:8px 12px;background:var(--surface2);border:1px solid var(--border2);border-radius:var(--radius-sm);color:var(--text2);font-size:12px;transition:all .15s;display:flex;align-items:center;gap:6px;justify-content:center}
.refresh-btn:hover{background:var(--surface3);color:var(--text)}
.run-btn{width:100%;padding:10px 12px;background:var(--accent);border:none;border-radius:var(--radius-sm);color:#fff;font-size:12px;font-weight:700;letter-spacing:.3px;transition:all .15s;display:flex;align-items:center;gap:6px;justify-content:center}
.run-btn:hover:not(:disabled){background:#4a79e0;transform:translateY(-1px)}
.run-btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
/* Progress panel */
.progress-header{display:flex;align-items:center;gap:8px;margin-top:12px;margin-bottom:6px}
.progress-pulse{width:8px;height:8px;border-radius:50%;background:var(--accent);flex-shrink:0;animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
#progress-title{font-size:11px;font-weight:600;color:var(--text2)}
.progress-log{max-height:160px;overflow-y:auto;display:flex;flex-direction:column;gap:3px}
.progress-step{font-size:11px;color:var(--text3);line-height:1.5;padding:2px 0;border-bottom:none}
.progress-step.latest{color:var(--text2)}
.progress-step.done-step{color:var(--risk-low)}
.progress-step.error-step{color:var(--risk-high)}
.view-new-btn{width:100%;margin-top:8px;padding:8px 12px;background:var(--risk-low);border:none;border-radius:var(--radius-sm);color:#fff;font-size:12px;font-weight:700;cursor:pointer;transition:opacity .15s}
.view-new-btn:hover{opacity:.85}

/* ── Main area ───────────────────────────────────────────── */
.main-header{padding:24px 32px 0;display:flex;align-items:flex-start;justify-content:space-between}
.main-header h1{font-size:20px;font-weight:700;letter-spacing:-.3px}
.main-header .date{font-size:12px;color:var(--text3);margin-top:4px}
.search-bar{position:relative}
.search-bar input{background:var(--surface);border:1px solid var(--border2);color:var(--text);padding:8px 12px 8px 34px;border-radius:var(--radius-sm);font-size:13px;outline:none;width:220px;transition:border .15s}
.search-bar input:focus{border-color:var(--accent)}
.search-bar .icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--text3);font-size:14px}

.main-body{padding:20px 32px 40px}

/* ── Top signals ─────────────────────────────────────────── */
.section-header{display:flex;align-items:baseline;gap:10px;margin-bottom:16px}
.section-title{font-size:13px;font-weight:700;letter-spacing:.2px;text-transform:uppercase;color:var(--text2)}
.section-count{font-size:11px;color:var(--text3)}

.top-signals-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:32px}
.signal-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;cursor:pointer;transition:border-color .15s,transform .1s;position:relative;overflow:hidden}
.signal-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--radius) var(--radius) 0 0}
.signal-card.high::before{background:var(--risk-high)}
.signal-card.medium::before{background:var(--risk-med)}
.signal-card.low::before{background:var(--risk-low)}
.signal-card:hover{border-color:var(--border2);transform:translateY(-1px)}
.card-meta{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.card-competitor{font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.card-score{font-size:11px;font-weight:700;color:var(--text3)}
.card-title{font-size:13px;font-weight:600;line-height:1.5;color:var(--text);margin-bottom:10px}
.card-why{font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:12px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card-action{background:var(--surface2);border-radius:var(--radius-sm);padding:10px 12px;font-size:11px;color:var(--accent);line-height:1.5;border-left:2px solid var(--accent)}
.card-action-label{font-size:10px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:var(--text3);margin-bottom:4px}

/* ── Strategic panel ─────────────────────────────────────── */
.strategic-panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:32px}
.strategic-header{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.strategic-icon{width:28px;height:28px;background:var(--accent-dim);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px}
.strategic-title{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.2px;color:var(--text2)}
.strategy-bullets{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.strategy-bullet{display:flex;align-items:flex-start;gap:10px;padding:12px 14px;background:var(--surface2);border-radius:var(--radius-sm);border:1px solid var(--border)}
.bullet-indicator{width:6px;height:6px;border-radius:50%;margin-top:5px;flex-shrink:0}
.bullet-text{font-size:12px;color:var(--text2);line-height:1.6}
.bullet-text strong{color:var(--text);font-weight:600}

/* ── Risk tabs + feed ────────────────────────────────────── */
.tabs{display:flex;gap:4px;margin-bottom:16px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:4px;width:fit-content}
.tab{padding:7px 18px;border-radius:5px;font-size:12px;font-weight:600;border:none;background:none;color:var(--text3);transition:all .15s;display:flex;align-items:center;gap:6px}
.tab:hover{color:var(--text)}
.tab.active{background:var(--surface3);color:var(--text)}
.tab .tab-count{font-size:10px;padding:1px 6px;border-radius:10px;font-weight:700}
.tab.active .tab-count{opacity:1}
.tab:not(.active) .tab-count{opacity:.6}
.tab-count.high{background:var(--risk-high-dim);color:var(--risk-high)}
.tab-count.med{background:var(--risk-med-dim);color:var(--risk-med)}
.tab-count.low{background:var(--risk-low-dim);color:var(--risk-low)}
.tab-count.all{background:var(--surface3);color:var(--text2)}

/* Feed */
.feed{display:flex;flex-direction:column;gap:1px;background:var(--border);border-radius:var(--radius);overflow:hidden;border:1px solid var(--border)}
.feed-item{background:var(--surface);padding:16px 20px;cursor:pointer;transition:background .12s;display:grid;grid-template-columns:36px 1fr auto;gap:14px;align-items:start}
.feed-item:hover{background:var(--surface2)}
.feed-item.expanded{background:var(--surface2)}

.feed-logo{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;margin-top:1px}

.feed-content{min-width:0}
.feed-top{display:flex;align-items:center;gap:8px;margin-bottom:5px;flex-wrap:wrap}
.feed-type{font-size:10px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:var(--text3);background:var(--surface3);padding:2px 7px;border-radius:4px}
.feed-comp{font-size:10px;font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.feed-title{font-size:13px;font-weight:600;color:var(--text);line-height:1.5;margin-bottom:6px}
.feed-why{font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:8px}
.feed-action{font-size:11px;color:var(--accent);padding:7px 10px;background:var(--accent-dim);border-radius:var(--radius-sm);line-height:1.5;margin-bottom:8px;border-left:2px solid var(--accent)}
.feed-footer{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.feed-footer-item{font-size:11px;color:var(--text3);display:flex;align-items:center;gap:4px}
.risk-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;display:inline-block}

.feed-right{display:flex;flex-direction:column;align-items:flex-end;gap:8px;flex-shrink:0;min-width:52px}
.feed-score{font-size:18px;font-weight:800;line-height:1}
.feed-score-label{font-size:9px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:var(--text3)}
.expand-icon{font-size:12px;color:var(--text3);transition:transform .2s;margin-top:4px}
.feed-item.expanded .expand-icon{transform:rotate(180deg)}

.feed-expanded{display:none;grid-column:2/-1;padding-top:8px;border-top:1px solid var(--border);margin-top:4px}
.feed-item.expanded .feed-expanded{display:block}
.expanded-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
.exp-item .lbl{font-size:10px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;color:var(--text3);margin-bottom:4px}
.exp-item .val{font-size:12px;color:var(--text2)}
.feed-source-link{font-size:11px;color:var(--accent);word-break:break-all}
.feed-source-link:hover{text-decoration:underline}

/* ── Loading / empty ─────────────────────────────────────── */
.loading-screen{display:flex;flex-direction:column;align-items:center;justify-content:center;height:400px;gap:16px;color:var(--text3)}
.spinner{width:28px;height:28px;border:2px solid var(--border2);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.empty-state{text-align:center;padding:60px;color:var(--text3)}

/* ── Detail drawer ───────────────────────────────────────── */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:30;backdrop-filter:blur(2px)}
.overlay.open{display:block}
.drawer{position:fixed;top:0;right:0;width:480px;max-width:95vw;height:100vh;background:var(--surface);border-left:1px solid var(--border);z-index:31;overflow-y:auto;transform:translateX(100%);transition:transform .22s cubic-bezier(.4,0,.2,1)}
.drawer.open{transform:none}
.drawer-header{padding:24px 24px 20px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--surface);z-index:1}
.drawer-close{float:right;background:none;border:none;color:var(--text3);font-size:18px;padding:2px;line-height:1;transition:color .15s}
.drawer-close:hover{color:var(--text)}
.drawer-comp{font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px}
.drawer-title{font-size:16px;font-weight:700;line-height:1.5;color:var(--text);padding-right:28px}
.drawer-body{padding:24px}
.drawer-section{margin-bottom:24px}
.drawer-section .lbl{font-size:10px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;color:var(--text3);margin-bottom:8px}
.drawer-section .val{font-size:13px;color:var(--text2);line-height:1.7}
.drawer-action{background:rgba(91,138,240,.08);border:1px solid rgba(91,138,240,.2);border-radius:var(--radius-sm);padding:14px 16px;font-size:13px;color:var(--accent);line-height:1.6}
.meta-chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}
.chip{padding:4px 10px;border-radius:5px;font-size:11px;font-weight:600}
.chip.high{background:var(--risk-high-dim);color:var(--risk-high)}
.chip.medium{background:var(--risk-med-dim);color:var(--risk-med)}
.chip.low{background:var(--risk-low-dim);color:var(--risk-low)}
.chip.direct{background:var(--risk-high-dim);color:var(--risk-high)}
.chip.indirect{background:var(--risk-med-dim);color:var(--risk-med)}
.chip.weak{background:var(--risk-low-dim);color:var(--risk-low)}
.chip.default{background:var(--surface2);color:var(--text2)}
.score-display{display:flex;align-items:baseline;gap:4px;margin-top:16px}
.score-big{font-size:36px;font-weight:800}
.score-denom{font-size:16px;color:var(--text3)}
.score-bar-track{height:6px;background:var(--surface3);border-radius:3px;margin-top:8px;overflow:hidden}
.score-bar-fill{height:100%;border-radius:3px;transition:width .4s ease}
</style>
</head>
<body>
<div class="shell">

<!-- ── Sidebar ──────────────────────────────────────────── -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <img src="/static/asme-logo-extracted.png" alt="ASME" style="width:140px;display:block"/>
  </div>

  <!-- Risk overview -->
  <div class="sidebar-section" style="padding-bottom:0">
    <div class="sidebar-section-label">Risk Overview</div>
    <div class="donut-wrap">
      <svg class="donut-svg" id="donut-svg" width="72" height="72" viewBox="0 0 72 72">
        <circle cx="36" cy="36" r="28" fill="none" stroke="var(--surface3)" stroke-width="10"/>
        <circle id="d-high" cx="36" cy="36" r="28" fill="none" stroke="var(--risk-high)" stroke-width="10" stroke-dasharray="0 176" stroke-linecap="round" transform="rotate(-90 36 36)" style="transition:stroke-dasharray .5s ease"/>
        <circle id="d-med" cx="36" cy="36" r="28" fill="none" stroke="var(--risk-med)" stroke-width="10" stroke-dasharray="0 176" stroke-linecap="round" transform="rotate(-90 36 36)" style="transition:stroke-dasharray .5s ease"/>
        <circle id="d-low" cx="36" cy="36" r="28" fill="none" stroke="var(--risk-low)" stroke-width="10" stroke-dasharray="0 176" stroke-linecap="round" transform="rotate(-90 36 36)" style="transition:stroke-dasharray .5s ease"/>
      </svg>
      <div class="donut-legend">
        <div class="legend-item">
          <div class="legend-dot" style="background:var(--risk-high)"></div>
          <div class="legend-text">High</div>
          <div class="legend-count" id="leg-high">—</div>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:var(--risk-med)"></div>
          <div class="legend-text">Medium</div>
          <div class="legend-count" id="leg-med">—</div>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:var(--risk-low)"></div>
          <div class="legend-text">Low</div>
          <div class="legend-count" id="leg-low">—</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Competitor strip -->
  <div class="sidebar-section" style="margin-top:20px">
    <div class="sidebar-section-label">Competitors</div>
    <div class="competitor-list" id="competitor-list">
      <div class="competitor-item active" data-comp="" onclick="filterCompetitor(this,'')">
        <div class="comp-logo" style="background:var(--surface3);color:var(--text2)">ALL</div>
        <div class="comp-info">
          <div class="comp-name">All Competitors</div>
          <div class="comp-sub" id="all-sub">— signals</div>
        </div>
      </div>
      <div class="competitor-item" data-comp="BIS" onclick="filterCompetitor(this,'BIS')">
        <div class="comp-logo" style="background:rgba(59,130,246,.15);color:var(--bis)">BIS</div>
        <div class="comp-info">
          <div class="comp-name">Bureau of Indian Standards</div>
          <div class="comp-sub" id="bis-sub">—</div>
        </div>
        <span class="comp-badge" id="bis-badge" style="background:var(--risk-high-dim);color:var(--risk-high)"></span>
      </div>
      <div class="competitor-item" data-comp="IEI" onclick="filterCompetitor(this,'IEI')">
        <div class="comp-logo" style="background:rgba(16,185,129,.15);color:var(--iei)">IEI</div>
        <div class="comp-info">
          <div class="comp-name">Institution of Engineers India</div>
          <div class="comp-sub" id="iei-sub">—</div>
        </div>
        <span class="comp-badge" id="iei-badge" style="background:var(--risk-med-dim);color:var(--risk-med)"></span>
      </div>
      <div class="competitor-item" data-comp="IMechE" onclick="filterCompetitor(this,'IMechE')">
        <div class="comp-logo" style="background:rgba(245,158,11,.15);color:var(--imech)">IMe</div>
        <div class="comp-info">
          <div class="comp-name">Institution of Mechanical Engineers</div>
          <div class="comp-sub" id="imech-sub">—</div>
        </div>
        <span class="comp-badge" id="imech-badge" style="background:var(--risk-med-dim);color:var(--risk-med)"></span>
      </div>
      <div class="competitor-item" data-comp="SAE India" onclick="filterCompetitor(this,'SAE India')">
        <div class="comp-logo" style="background:rgba(139,92,246,.15);color:var(--sae)">SAE</div>
        <div class="comp-info">
          <div class="comp-name">SAE India</div>
          <div class="comp-sub" id="sae-sub">—</div>
        </div>
        <span class="comp-badge" id="sae-badge" style="background:var(--risk-low-dim);color:var(--risk-low)"></span>
      </div>
    </div>
  </div>

  <div class="sidebar-nav">
    <div class="ts" id="ts-label">—</div>
    <!-- Reload from Notion -->
    <button class="refresh-btn" id="reload-btn" onclick="loadSignals()" style="margin-bottom:8px">
      ↻ Reload from Notion
    </button>
    <!-- Run full pipeline -->
    <button class="run-btn" id="run-btn" onclick="startRun()">
      <span id="run-btn-icon">⚡</span>
      <span id="run-btn-label">Fetch Fresh Signals</span>
    </button>
    <!-- Progress panel -->
    <div id="run-progress" style="display:none">
      <div class="progress-header">
        <div class="progress-pulse" id="progress-pulse"></div>
        <span id="progress-title">Running pipeline...</span>
      </div>
      <div class="progress-log" id="progress-log"></div>
    </div>
  </div>
</aside>

<!-- ── Main ──────────────────────────────────────────────── -->
<main class="main">
  <div class="main-header">
    <div>
      <h1 id="view-title">Intelligence Feed</h1>
      <div class="date" id="view-date">Loading...</div>
    </div>
    <div class="search-bar">
      <span class="icon">⌕</span>
      <input type="text" id="f-search" placeholder="Search signals..." oninput="applyAll()"/>
    </div>
  </div>

  <div class="main-body" id="main-body">
    <div class="loading-screen">
      <div class="spinner"></div>
      <div>Loading signals from Notion...</div>
    </div>
  </div>
</main>
</div>

<!-- ── Detail drawer ─────────────────────────────────────── -->
<div class="overlay" id="overlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">
    <button class="drawer-close" onclick="closeDrawer()">✕</button>
    <div class="drawer-comp" id="d-comp"></div>
    <div class="drawer-title" id="d-title"></div>
  </div>
  <div class="drawer-body">
    <div class="meta-chips" id="d-chips"></div>
    <div class="score-display">
      <div class="score-big" id="d-score-num"></div>
      <div class="score-denom">/100</div>
      <div style="margin-left:8px;font-size:11px;color:var(--text3)">composite score</div>
    </div>
    <div class="score-bar-track"><div class="score-bar-fill" id="d-score-fill"></div></div>

    <div style="margin-top:24px"></div>

    <div class="drawer-section">
      <div class="lbl">What Happened</div>
      <div class="val" id="d-summary"></div>
    </div>
    <div class="drawer-section">
      <div class="lbl">Why It Matters to ASME</div>
      <div class="val" id="d-why"></div>
    </div>
    <div class="drawer-section">
      <div class="lbl">Recommended Action</div>
      <div class="drawer-action" id="d-action"></div>
    </div>
    <div class="drawer-section">
      <div class="lbl">Signal Metadata</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div><div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px">Geography</div><div style="font-size:12px;color:var(--text2)" id="d-geo"></div></div>
        <div><div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px">Business Impact</div><div style="font-size:12px;color:var(--text2)" id="d-impact"></div></div>
        <div><div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px">Confidence</div><div style="font-size:12px;color:var(--text2)" id="d-conf"></div></div>
        <div><div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px">Published</div><div style="font-size:12px;color:var(--text2)" id="d-date"></div></div>
      </div>
    </div>
    <div class="drawer-section">
      <div class="lbl">Source</div>
      <a class="feed-source-link" id="d-url" href="" target="_blank"></a>
    </div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────
let allSignals = [];
let activeComp = '';
let activeTab = 'all';

// ── Colours ───────────────────────────────────────────────
const COMP_COLORS = {
  'BIS':      {bg:'rgba(59,130,246,.15)', text:'#3b82f6'},
  'IEI':      {bg:'rgba(16,185,129,.15)', text:'#10b981'},
  'IMechE':   {bg:'rgba(245,158,11,.15)', text:'#f59e0b'},
  'SAE India':{bg:'rgba(139,92,246,.15)', text:'#8b5cf6'},
};
const RISK_COLORS = {
  high:   {fg:'var(--risk-high)',   bg:'var(--risk-high-dim)'},
  medium: {fg:'var(--risk-med)',    bg:'var(--risk-med-dim)'},
  low:    {fg:'var(--risk-low)',    bg:'var(--risk-low-dim)'},
};
function compColor(c){ return COMP_COLORS[c] || {bg:'var(--surface3)',text:'var(--text2)'}; }
function scoreColor(v){ return v>=70?'var(--risk-high)':v>=40?'var(--risk-med)':'var(--risk-low)'; }

// ── Load ──────────────────────────────────────────────────
async function loadSignals(){
  document.getElementById('main-body').innerHTML = `
    <div class="loading-screen"><div class="spinner"></div><div>Loading signals from Notion...</div></div>`;
  try {
    const res = await fetch('/api/signals');
    if(!res.ok) throw new Error('API returned HTTP ' + res.status);
    const data = await res.json();
    if(data.error) throw new Error('API error: ' + data.error);
    allSignals = data.signals || [];

    // Safe date formatting — no locale dependency
    const now = new Date();
    document.getElementById('ts-label').textContent =
      'Updated ' + now.getHours().toString().padStart(2,'0') + ':' + now.getMinutes().toString().padStart(2,'0');
    document.getElementById('view-date').textContent =
      now.toDateString();

    try { updateSidebar(allSignals); } catch(e2){ console.error('updateSidebar failed:', e2); }
    try { renderMain(); } catch(e3){
      document.getElementById('main-body').innerHTML =
        `<div class="empty-state">Render error: ${e3.message}<br><small>${e3.stack}</small></div>`;
      return;
    }
  } catch(e) {
    document.getElementById('main-body').innerHTML =
      `<div class="empty-state" style="color:#f25757">
        <strong>Error loading signals</strong><br><br>${e.message}
      </div>`;
  }
}

// ── Sidebar ───────────────────────────────────────────────
function updateSidebar(signals) {
  const total = signals.length;
  const high  = signals.filter(s=>s.urgency==='high').length;
  const med   = signals.filter(s=>s.urgency==='medium').length;
  const low   = signals.filter(s=>s.urgency==='low').length;
  const circ  = 2 * Math.PI * 28; // circumference

  document.getElementById('leg-high').textContent = high;
  document.getElementById('leg-med').textContent  = med;
  document.getElementById('leg-low').textContent  = low;
  document.getElementById('all-sub').textContent  = total + ' signals';

  // Donut arcs
  const hFrac = total ? high/total : 0;
  const mFrac = total ? med/total  : 0;
  const lFrac = total ? low/total  : 0;
  const hArc  = hFrac * circ;
  const mArc  = mFrac * circ;
  const lArc  = lFrac * circ;

  const dHigh = document.getElementById('d-high');
  const dMed  = document.getElementById('d-med');
  const dLow  = document.getElementById('d-low');

  dHigh.style.strokeDasharray = `${hArc} ${circ}`;
  dHigh.style.strokeDashoffset = '0';

  dMed.style.strokeDasharray = `${mArc} ${circ}`;
  dMed.style.strokeDashoffset = `${-hArc}`;

  dLow.style.strokeDasharray = `${lArc} ${circ}`;
  dLow.style.strokeDashoffset = `${-(hArc+mArc)}`;

  // Competitor strips
  ['BIS','IEI','IMechE','SAE India'].forEach(c => {
    const cs = signals.filter(s=>s.competitor===c);
    const cHigh = cs.filter(s=>s.urgency==='high').length;
    const cMed  = cs.filter(s=>s.urgency==='medium').length;
    const topRisk = cHigh>0?'high': cMed>0?'medium':'low';
    const key = c.replace(' ','').toLowerCase();
    const badgeEl = document.getElementById(c==='SAE India'?'sae-badge':key+'-badge');
    const subEl   = document.getElementById(c==='SAE India'?'sae-sub':key+'-sub');
    if(badgeEl){ badgeEl.textContent = topRisk; }
    if(subEl){ subEl.textContent = cs.length + ' signals'; }
  });
}

// ── Main render ───────────────────────────────────────────
function renderMain(){
  const filtered = getFiltered();
  const topN = filtered.slice(0,5);

  document.getElementById('view-title').textContent =
    activeComp ? activeComp + ' — Signals' : 'Intelligence Feed';

  let html = '';

  // Top signals
  if(!activeComp && activeTab==='all'){
    const top = allSignals.slice(0,3);
    if(top.length){
      html += `<div class="section-header"><div class="section-title">Top Signals</div><div class="section-count">highest importance right now</div></div>`;
      html += `<div class="top-signals-grid">` + top.map((s,i)=>topCard(s,i)).join('') + `</div>`;
      html += renderStrategicPanel(allSignals);
    }
  }

  // Tabs
  const all = getFilteredByComp();
  const tHigh = all.filter(s=>s.urgency==='high').length;
  const tMed  = all.filter(s=>s.urgency==='medium').length;
  const tLow  = all.filter(s=>s.urgency==='low').length;

  html += `
  <div class="section-header" style="margin-top:4px">
    <div class="section-count">${filtered.length} signals</div>
  </div>
  <div class="tabs">
    <button class="tab ${activeTab==='all'?'active':''}" onclick="setTab('all')">
      All <span class="tab-count all">${all.length}</span>
    </button>
    <button class="tab ${activeTab==='high'?'active':''}" onclick="setTab('high')">
      High Risk <span class="tab-count high">${tHigh}</span>
    </button>
    <button class="tab ${activeTab==='medium'?'active':''}" onclick="setTab('medium')">
      Medium <span class="tab-count med">${tMed}</span>
    </button>
    <button class="tab ${activeTab==='low'?'active':''}" onclick="setTab('low')">
      Low <span class="tab-count low">${tLow}</span>
    </button>
  </div>`;

  if(filtered.length===0){
    html += `<div class="empty-state">No signals match the current filters.</div>`;
  } else {
    html += `<div class="feed">` + filtered.map((s,i)=>feedItem(s,i)).join('') + `</div>`;
  }

  document.getElementById('main-body').innerHTML = html;
}

// ── Top signal card ───────────────────────────────────────
function topCard(s, i){
  const cc = compColor(s.competitor);
  return `
  <div class="signal-card ${s.urgency}" onclick="openDrawer(${i})">
    <div class="card-meta">
      <span class="card-competitor" style="color:${cc.text}">${s.competitor}</span>
      <span class="card-score" style="color:${scoreColor(s.composite_score)}">${Math.round(s.composite_score)}</span>
    </div>
    <div class="card-title">${esc(s.title)}</div>
    <div class="card-why">${esc(s.why_it_matters)}</div>
    <div class="card-action">
      <div class="card-action-label">Action</div>
      ${esc(s.recommended_action)}
    </div>
  </div>`;
}

// ── Strategic panel ───────────────────────────────────────
function renderStrategicPanel(signals){
  const high = signals.filter(s=>s.urgency==='high' && s.revenue_relevance!=='weak');
  const byType = {};
  signals.forEach(s=>{ byType[s.signal_type]=(byType[s.signal_type]||0)+1; });
  const topType = Object.entries(byType).sort((a,b)=>b[1]-a[1])[0];
  const directCount = signals.filter(s=>s.revenue_relevance==='direct').length;

  const bullets = [
    high.length > 0 && `<strong>${high.length} high-urgency moves</strong> from competitors require immediate ASME response — review recommended actions in Top Signals above.`,
    topType && `<strong>${topType[0]}</strong> is the dominant signal type (${topType[1]} signals) — competitors are most active in this area.`,
    directCount > 0 && `<strong>${directCount} signals</strong> carry direct revenue risk to ASME India's core programs.`,
    signals.filter(s=>s.signal_type==='regulatory').length > 0 && `BIS regulatory moves are reshaping India's standards landscape — ASME should monitor certification alignment closely.`,
    signals.filter(s=>s.signal_type==='student_outreach'||s.signal_type==='expansion').length > 0 && `Student pipeline and geographic expansion activity detected — ASME chapter strategy should be reviewed.`,
  ].filter(Boolean).slice(0,4);

  if(!bullets.length) return '';
  return `
  <div class="strategic-panel">
    <div class="strategic-header">
      <div class="strategic-icon">◎</div>
      <div class="strategic-title">Strategic Signals for ASME India</div>
    </div>
    <div class="strategy-bullets">
      ${bullets.map(b=>`
        <div class="strategy-bullet">
          <div class="bullet-indicator" style="background:var(--accent)"></div>
          <div class="bullet-text">${b}</div>
        </div>`).join('')}
    </div>
  </div>`;
}

// ── Feed item ─────────────────────────────────────────────
function feedItem(s, i){
  const cc = compColor(s.competitor);
  const rc = RISK_COLORS[s.urgency] || RISK_COLORS.low;
  const initials = s.competitor.replace(' India','').substring(0,3).toUpperCase();
  return `
  <div class="feed-item" id="fi-${i}" onclick="toggleFeed(${i},event)">
    <div class="feed-logo" style="background:${cc.bg};color:${cc.text}">${initials}</div>
    <div class="feed-content">
      <div class="feed-top">
        <span class="feed-comp" style="color:${cc.text}">${s.competitor}</span>
        <span class="feed-type">${s.signal_type}</span>
      </div>
      <div class="feed-title">${esc(s.title)}</div>
      <div class="feed-why">${esc(s.why_it_matters)}</div>
      <div class="feed-action">${esc(s.recommended_action)}</div>
      <div class="feed-footer">
        <span class="feed-footer-item"><span class="risk-dot" style="background:${rc.fg}"></span>${s.urgency}</span>
        <span class="feed-footer-item" style="color:${s.revenue_relevance==='direct'?'var(--risk-high)':s.revenue_relevance==='indirect'?'var(--risk-med)':'var(--text3)'}">${s.revenue_relevance} revenue risk</span>
        ${s.source?`<span class="feed-footer-item">${esc(s.source)}</span>`:''}
        ${s.published_date?`<span class="feed-footer-item">${esc(s.published_date)}</span>`:''}
      </div>
      <div class="feed-expanded">
        <div class="expanded-grid">
          <div class="exp-item"><div class="lbl">Business Impact</div><div class="val">${esc(s.business_impact)}</div></div>
          <div class="exp-item"><div class="lbl">Geography</div><div class="val">${esc(s.geography)}</div></div>
          <div class="exp-item"><div class="lbl">Confidence</div><div class="val">${s.confidence_score}/100</div></div>
        </div>
        <a class="feed-source-link" href="${esc(s.url)}" target="_blank" onclick="event.stopPropagation()">${esc(s.url)}</a>
      </div>
    </div>
    <div class="feed-right">
      <div class="feed-score" style="color:${scoreColor(s.composite_score)}">${Math.round(s.composite_score)}</div>
      <div class="feed-score-label">score</div>
      <div class="expand-icon">▾</div>
    </div>
  </div>`;
}

// ── Interactions ──────────────────────────────────────────
function toggleFeed(i, e){
  // If clicking a link inside, let it pass
  if(e && e.target.tagName==='A') return;
  const el = document.getElementById('fi-'+i);
  if(!el) return;
  el.classList.toggle('expanded');
}

function filterCompetitor(el, comp){
  document.querySelectorAll('.competitor-item').forEach(e=>e.classList.remove('active'));
  el.classList.add('active');
  activeComp = comp;
  activeTab = 'all';
  renderMain();
}

function setTab(tab){
  activeTab = tab;
  renderMain();
}

function applyAll(){ renderMain(); }

function getFilteredByComp(){
  return activeComp ? allSignals.filter(s=>s.competitor===activeComp) : allSignals;
}

function getFiltered(){
  const q = (document.getElementById('f-search')||{}).value?.toLowerCase()||'';
  return getFilteredByComp().filter(s=>{
    const tabOk = activeTab==='all' || s.urgency===activeTab;
    const qOk = !q || [s.title,s.why_it_matters,s.recommended_action,s.summary,s.competitor]
      .join(' ').toLowerCase().includes(q);
    return tabOk && qOk;
  });
}

// ── Drawer ────────────────────────────────────────────────
function openDrawer(i){
  const s = allSignals[i];
  if(!s) return;
  const cc = compColor(s.competitor);
  document.getElementById('d-comp').textContent  = s.competitor;
  document.getElementById('d-comp').style.color  = cc.text;
  document.getElementById('d-title').textContent = s.title;
  document.getElementById('d-summary').textContent= s.summary;
  document.getElementById('d-why').textContent    = s.why_it_matters;
  document.getElementById('d-action').textContent = s.recommended_action;
  document.getElementById('d-geo').textContent    = s.geography;
  document.getElementById('d-impact').textContent = s.business_impact;
  document.getElementById('d-conf').textContent   = s.confidence_score + '/100';
  document.getElementById('d-date').textContent   = s.published_date || 'unknown';
  const urlEl = document.getElementById('d-url');
  urlEl.href = s.url; urlEl.textContent = s.url;

  const score = Math.round(s.composite_score);
  document.getElementById('d-score-num').textContent = score;
  document.getElementById('d-score-num').style.color = scoreColor(s.composite_score);
  const fill = document.getElementById('d-score-fill');
  fill.style.width = score + '%';
  fill.style.background = scoreColor(s.composite_score);

  const chips = [
    {label:s.urgency, cls:s.urgency},
    {label:s.revenue_relevance+' revenue', cls:s.revenue_relevance},
    {label:s.signal_type, cls:'default'},
    {label:s.business_impact, cls:'default'},
  ];
  document.getElementById('d-chips').innerHTML = chips
    .map(c=>`<span class="chip ${c.cls}">${esc(c.label)}</span>`).join('');

  document.getElementById('overlay').classList.add('open');
  document.getElementById('drawer').classList.add('open');
}

function closeDrawer(){
  document.getElementById('overlay').classList.remove('open');
  document.getElementById('drawer').classList.remove('open');
}

document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeDrawer(); });

function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Fresh signals pipeline ────────────────────────────────
async function startRun(){
  const btn      = document.getElementById('run-btn');
  const icon     = document.getElementById('run-btn-icon');
  const label    = document.getElementById('run-btn-label');
  const progress = document.getElementById('run-progress');
  const log      = document.getElementById('progress-log');
  const title    = document.getElementById('progress-title');
  const pulse    = document.getElementById('progress-pulse');

  // Start pipeline
  const res = await fetch('/api/run', {method:'POST'});
  if(!res.ok){
    const d = await res.json();
    if(res.status===409){
      title.textContent = 'Already running...';
      progress.style.display = 'block';
      return;
    }
    alert('Failed to start: ' + (d.error||'unknown error'));
    return;
  }

  // Lock button, show progress
  btn.disabled = true;
  icon.textContent = '⏳';
  label.textContent = 'Running...';
  progress.style.display = 'block';
  log.innerHTML = '';
  title.textContent = 'Pipeline running...';
  pulse.style.display = 'block';

  function addStep(text, cls=''){
    // De-bold previous latest
    log.querySelectorAll('.latest').forEach(e=>e.classList.remove('latest'));
    const div = document.createElement('div');
    div.className = 'progress-step latest ' + cls;
    div.textContent = '→ ' + text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  // SSE stream
  const es = new EventSource('/api/run/status');

  es.onmessage = e => { addStep(e.data); };

  es.addEventListener('done', e => {
    es.close();
    addStep(e.data, 'done-step');
    title.textContent = 'Complete';
    pulse.style.animation = 'none';
    pulse.style.background = 'var(--risk-low)';
    btn.disabled = false;
    icon.textContent = '⚡';
    label.textContent = 'Fetch Fresh Signals';
    // Show manual reload button
    const viewBtn = document.createElement('button');
    viewBtn.className = 'view-new-btn';
    viewBtn.textContent = '↻ View New Signals';
    viewBtn.onclick = () => { loadSignals(); viewBtn.remove(); };
    log.appendChild(viewBtn);
    log.scrollTop = log.scrollHeight;
  });

  es.addEventListener('error', e => {
    if(e.data){ addStep(e.data, 'error-step'); }
    es.close();
    title.textContent = 'Run failed';
    pulse.style.animation = 'none';
    pulse.style.background = 'var(--risk-high)';
    btn.disabled = false;
    icon.textContent = '⚡';
    label.textContent = 'Fetch Fresh Signals';
  });

  // SSE connection error (not pipeline error)
  es.onerror = () => {
    if(es.readyState === EventSource.CLOSED) return;
    es.close();
    title.textContent = 'Connection lost';
    btn.disabled = false;
  };
}

// ── Boot ─────────────────────────────────────────────────
loadSignals();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "5050")))
    host = "0.0.0.0"  # bind to all interfaces for Railway/cloud
    print(f"\n  ASME Intel Dashboard → http://localhost:{port}\n")
    app.run(host=host, port=port, debug=False)
