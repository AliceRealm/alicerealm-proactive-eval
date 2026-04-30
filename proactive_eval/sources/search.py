"""DuckDuckGo search signal source with LLM-generated queries."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from proactive_eval.types import SignalItem
from proactive_eval.sources import SignalSource
from proactive_eval.config import SCRIPT_DIR, rel_path, load_fixtures

DATA_DIR = SCRIPT_DIR / "data" / "searches"
QUERIES_FILE = SCRIPT_DIR / "config" / "search_queries.json"
MAX_RESULTS_PER_QUERY = 5
SLEEP_BETWEEN_QUERIES = 5
DEFAULT_NUM_QUERIES = 5
QUERY_GEN_MODEL = "claude-haiku-4-5-20251001"


class SearchSource(SignalSource):
    source_name = "search"

    def supports_replay(self) -> bool:
        return True

    def collect(self, fixture_id: str, **kwargs) -> list[SignalItem]:
        """Run DuckDuckGo searches for one fixture and return normalized SignalItems."""
        dry_run = kwargs.get("dry_run", False)
        max_results = kwargs.get("max_results", MAX_RESULTS_PER_QUERY)
        sleep_between = kwargs.get("sleep_between", SLEEP_BETWEEN_QUERIES)
        llm_queries = kwargs.get("llm_queries", False)
        num_queries = kwargs.get("num_queries", DEFAULT_NUM_QUERIES)
        query_model = kwargs.get("query_model", QUERY_GEN_MODEL)

        if llm_queries:
            # LLM-generated queries from fixture profile
            fixture = _load_fixture(fixture_id)
            if fixture is None:
                print(f"  Warning: fixture {fixture_id} not found, skipping")
                return []
            queries = generate_queries(fixture, num_queries=num_queries,
                                       model=query_model)
            pair = {
                "fixture": fixture_id,
                "scenario": "llm_generated",
                "queries": queries,
            }
        else:
            # Legacy: read from static JSON
            pairs = load_queries()
            fixture_pairs = [p for p in pairs if p["fixture"] == fixture_id]
            if not fixture_pairs:
                return []
            pair = fixture_pairs[0]

        result = collect_one_pair(pair, dry_run=dry_run,
                                 max_results=max_results,
                                 sleep_between=sleep_between)
        if not dry_run:
            save_result(fixture_id, result)
        return _normalize_result(result, fixture_id)

    def replay(self, fixture_id: str, data_dir: Path | None = None) -> list[SignalItem]:
        """Load all previously collected search results from disk."""
        d = (data_dir or DATA_DIR) / fixture_id
        if not d.exists():
            return []
        items = []
        for p in sorted(d.glob("*.json")):
            items.extend(_normalize_file(p, fixture_id))
        return items


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _load_fixture(fixture_id: str) -> dict | None:
    fixtures = load_fixtures()
    for f in fixtures:
        if f["fixture_id"] == fixture_id:
            return f
    return None


# ---------------------------------------------------------------------------
# LLM query generation
# ---------------------------------------------------------------------------

def _build_profile_context(fixture: dict) -> str:
    """Build a text summary of the fixture for the LLM to generate queries from."""
    parts = []
    # EigenFlux bio (most informative)
    ef = fixture.get("eigenflux_profile", {})
    if ef.get("bio"):
        parts.append(f"Agent bio:\n{ef['bio']}")
    # Workspace files (MEMORY, USER, etc.)
    for name, content in fixture.get("workspace_files", {}).items():
        parts.append(f"[{name}]\n{content}")
    # Memory files
    for name, content in fixture.get("memory_files", {}).items():
        if name.endswith(".json"):
            continue
        parts.append(f"[{name}]\n{content}")
    return "\n\n".join(parts)


QUERY_GEN_PROMPT = """\
You are simulating how an AI assistant would search the web to stay informed \
for its user. Based on the agent profile below, generate exactly {n} web \
search queries that this agent would realistically run right now.

Rules:
- Write natural language queries, like a person typing into Google (not SEO keywords)
- Cover the agent's most important current concerns, not everything
- Be specific to today's date ({today}) — include temporal context where relevant
- Each query should be 4-12 words
- Do NOT number them or add any explanation

Output: one query per line, nothing else.

---
{profile}
"""


def generate_queries(fixture: dict, num_queries: int = DEFAULT_NUM_QUERIES,
                     model: str = QUERY_GEN_MODEL) -> list[str]:
    """Use an LLM to generate realistic search queries from a fixture profile."""
    import anthropic

    profile = _build_profile_context(fixture)
    today = datetime.now().strftime("%B %d, %Y")

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": QUERY_GEN_PROMPT.format(n=num_queries, today=today,
                                                profile=profile),
        }],
    )

    text = resp.content[0].text.strip()
    queries = [line.strip() for line in text.splitlines() if line.strip()]

    # Strip leading numbers like "1." or "1)" if model adds them
    cleaned = []
    for q in queries:
        for prefix in [".", ")", ":"]:
            if len(q) > 2 and q[0].isdigit() and prefix in q[:4]:
                q = q[q.index(prefix) + 1:].strip()
                break
        if q:
            cleaned.append(q)

    fixture_id = fixture.get("fixture_id", "?")
    print(f"  LLM generated {len(cleaned)} queries for {fixture_id}:")
    for i, q in enumerate(cleaned):
        print(f"    {i+1}. {q}")

    return cleaned[:num_queries]


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(queries_file=QUERIES_FILE) -> list[dict]:
    with open(queries_file) as f:
        return json.load(f)["pairs"]


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = MAX_RESULTS_PER_QUERY,
           retries: int = 2) -> list[dict]:
    from ddgs import DDGS
    for attempt in range(retries + 1):
        try:
            results = DDGS().text(query, max_results=max_results)
            return results
        except Exception as e:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            return [{"error": str(e)}]


def collect_one_pair(pair: dict, dry_run: bool = False,
                     max_results: int = MAX_RESULTS_PER_QUERY,
                     sleep_between: int = SLEEP_BETWEEN_QUERIES) -> dict:
    fixture_id = pair["fixture"]
    scenario_id = pair.get("scenario", "unknown")
    queries = pair["queries"]

    now = datetime.now(timezone.utc)
    cycle_result = {
        "fixture_id": fixture_id,
        "scenario_id": scenario_id,
        "timestamp_utc": now.isoformat(),
        "timestamp_local": datetime.now().isoformat(),
        "query_count": len(queries),
        "query_source": "llm" if scenario_id == "llm_generated" else "static",
        "results": []
    }

    for i, query in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {query}", end="")
        if dry_run:
            print(" (dry-run, skipped)")
            cycle_result["results"].append({
                "query": query, "result_count": 0, "items": [], "dry_run": True
            })
            continue

        items = search(query, max_results=max_results)
        print(f" -> {len(items)} results")
        cycle_result["results"].append({
            "query": query, "result_count": len(items), "items": items
        })

        if i < len(queries) - 1:
            time.sleep(sleep_between)

    return cycle_result


def save_result(fixture_id: str, result: dict):
    pair_dir = DATA_DIR / fixture_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = pair_dir / f"{ts}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {filename}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_result(result: dict, fixture_id: str) -> list[SignalItem]:
    pull_ts = result.get("timestamp_utc", "")
    items = []
    for query_block in result.get("results", []):
        query = query_block.get("query", "")
        for raw in query_block.get("items", []):
            if not isinstance(raw, dict) or "error" in raw:
                continue
            items.append(_raw_to_signal(raw, fixture_id, pull_ts, query, "in-memory"))
    return items


def _normalize_file(path: Path, fixture_id: str) -> list[SignalItem]:
    data = json.loads(path.read_text())
    pull_ts = data.get("timestamp_utc", "")
    items = []
    for query_block in data.get("results", []):
        query = query_block.get("query", "")
        for raw in query_block.get("items", []):
            if not isinstance(raw, dict) or "error" in raw:
                continue
            items.append(_raw_to_signal(raw, fixture_id, pull_ts, query, rel_path(path)))
    return items


def _raw_to_signal(raw: dict, fixture_id: str, pull_ts: str,
                   query: str, pull_file: str) -> SignalItem:
    content_parts = []
    if raw.get("title"):
        content_parts.append(f"Title: {raw['title']}")
    if raw.get("body"):
        content_parts.append(f"Body: {raw['body']}")
    if raw.get("href"):
        content_parts.append(f"URL: {raw['href']}")

    try:
        received_at = datetime.fromisoformat(pull_ts)
        if received_at.tzinfo is None:
            received_at = received_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        received_at = datetime.now(timezone.utc)

    return SignalItem(
        item_id=raw.get("href", ""),
        source="search",
        received_at=received_at,
        content="\n".join(content_parts),
        url=raw.get("href"),
        metadata={
            "fixture_id": fixture_id,
            "pull_file": pull_file,
            "query": query,
        },
        raw_item=raw,
    )
