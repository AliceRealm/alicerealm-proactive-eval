"""DuckDuckGo search signal source."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from proactive_eval.types import SignalItem
from proactive_eval.sources import SignalSource
from proactive_eval.config import SCRIPT_DIR, rel_path

DATA_DIR = SCRIPT_DIR / "data" / "searches"
QUERIES_FILE = SCRIPT_DIR / "search_queries.json"
MAX_RESULTS_PER_QUERY = 5
SLEEP_BETWEEN_QUERIES = 5


class SearchSource(SignalSource):
    source_name = "search"

    def supports_replay(self) -> bool:
        return True

    def collect(self, fixture_id: str, **kwargs) -> list[SignalItem]:
        """Run DuckDuckGo searches for one fixture and return normalized SignalItems."""
        dry_run = kwargs.get("dry_run", False)
        max_results = kwargs.get("max_results", MAX_RESULTS_PER_QUERY)
        sleep_between = kwargs.get("sleep_between", SLEEP_BETWEEN_QUERIES)
        queries_file = kwargs.get("queries_file", QUERIES_FILE)

        pairs = load_queries(queries_file)
        fixture_pairs = [p for p in pairs if p["fixture"] == fixture_id]
        if not fixture_pairs:
            return []

        all_items = []
        for pair in fixture_pairs:
            result = collect_one_pair(pair, dry_run=dry_run,
                                     max_results=max_results,
                                     sleep_between=sleep_between)
            if not dry_run:
                save_result(fixture_id, result)
            all_items.extend(_normalize_result(result, fixture_id))
        return all_items

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
