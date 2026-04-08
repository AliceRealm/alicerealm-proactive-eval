#!/usr/bin/env python3
"""
Baseline search collector for proactive-eval.

Runs web searches for each fixture-episode pair and saves results.
Designed to be triggered every 30 minutes by cron over a 2-day window.

Usage:
    python3 collect_searches.py                # Run one cycle for all pairs
    python3 collect_searches.py --pair 0       # Run only the first pair
    python3 collect_searches.py --dry-run      # Show what would be searched, no actual queries
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

from ddgs import DDGS

SCRIPT_DIR = Path(__file__).parent
QUERIES_FILE = SCRIPT_DIR / "search_queries.json"
DATA_DIR = SCRIPT_DIR / "data" / "searches"
MAX_RESULTS_PER_QUERY = 5
SLEEP_BETWEEN_QUERIES = 5  # seconds, to avoid rate limiting


def load_queries():
    with open(QUERIES_FILE) as f:
        return json.load(f)["pairs"]


def search(query: str, max_results: int = MAX_RESULTS_PER_QUERY, retries: int = 2) -> list[dict]:
    """Run a single DuckDuckGo search and return results."""
    for attempt in range(retries + 1):
        try:
            results = DDGS().text(query, max_results=max_results)
            return results
        except Exception as e:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            return [{"error": str(e)}]


def collect_one_pair(pair: dict, dry_run: bool = False) -> dict:
    """Run all queries for one fixture-episode pair and return structured results."""
    fixture_id = pair["fixture"]
    episode_id = pair.get("episode", "unknown")
    queries = pair["queries"]

    now = datetime.now(timezone.utc)
    cycle_result = {
        "fixture_id": fixture_id,
        "episode_id": episode_id,
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
                "query": query,
                "result_count": 0,
                "items": [],
                "dry_run": True
            })
            continue

        items = search(query)
        print(f" -> {len(items)} results")

        cycle_result["results"].append({
            "query": query,
            "result_count": len(items),
            "items": items
        })

        if i < len(queries) - 1:
            time.sleep(SLEEP_BETWEEN_QUERIES)

    return cycle_result


def save_result(fixture_id: str, result: dict):
    """Save one cycle's results to a timestamped JSON file."""
    pair_dir = DATA_DIR / fixture_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = pair_dir / f"{ts}.json"

    with open(filename, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Saved -> {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Baseline search collector")
    parser.add_argument("--pair", type=int, default=None,
                        help="Run only the Nth pair (0-indexed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show queries without executing searches")
    args = parser.parse_args()

    pairs = load_queries()

    if args.pair is not None:
        if args.pair < 0 or args.pair >= len(pairs):
            print(f"Error: --pair must be 0-{len(pairs)-1}")
            sys.exit(1)
        pairs = [pairs[args.pair]]

    print(f"=== Baseline Search Collection ===")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Pairs: {len(pairs)}")
    print(f"Dry run: {args.dry_run}")
    print()

    for pair in pairs:
        fixture_id = pair["fixture"]
        print(f"--- {fixture_id} ({pair['description'][:60]}...) ---")

        result = collect_one_pair(pair, dry_run=args.dry_run)

        if not args.dry_run:
            save_result(fixture_id, result)

        print()

    print("Done.")


if __name__ == "__main__":
    main()
