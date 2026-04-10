#!/usr/bin/env python3
"""
Baseline search collector for proactive-eval.

Runs web searches for each fixture-scenario pair and saves results.
Loops every --interval minutes for --duration hours, then exits.

Usage:
    python3 collect_searches.py                          # Loop every 30min for 24h
    python3 collect_searches.py --interval 15 --duration 12  # Every 15min for 12h
    python3 collect_searches.py --once                   # Run one cycle and exit
    python3 collect_searches.py --pair 0                 # Only the first pair
    python3 collect_searches.py --dry-run                # Show queries, no searches
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
    """Run all queries for one fixture-scenario pair and return structured results."""
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


def run_one_cycle(pairs: list[dict], dry_run: bool = False):
    """Run one full collection cycle across all pairs."""
    print(f"=== Cycle @ {datetime.now().isoformat()} ===")
    for pair in pairs:
        fixture_id = pair["fixture"]
        print(f"--- {fixture_id} ({pair['description'][:60]}...) ---")
        result = collect_one_pair(pair, dry_run=dry_run)
        if not dry_run:
            save_result(fixture_id, result)
        print()


def main():
    parser = argparse.ArgumentParser(description="Baseline search collector")
    parser.add_argument("--pair", type=int, default=None,
                        help="Run only the Nth pair (0-indexed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show queries without executing searches")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutes between cycles (default: 5)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Hours to run (default: 0 = forever)")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    args = parser.parse_args()

    pairs = load_queries()

    if args.pair is not None:
        if args.pair < 0 or args.pair >= len(pairs):
            print(f"Error: --pair must be 0-{len(pairs)-1}")
            sys.exit(1)
        pairs = [pairs[args.pair]]

    if args.once:
        run_forever = False
        total_cycles = 1
    elif args.duration > 0:
        run_forever = False
        total_cycles = int(args.duration * 60 / args.interval)
    else:
        run_forever = True
        total_cycles = None

    print(f"=== Baseline Search Collection ===")
    print(f"Start: {datetime.now().isoformat()}")
    mode = "forever" if run_forever else (f"1 cycle" if args.once else f"{args.duration}h ({total_cycles} cycles)")
    print(f"Pairs: {len(pairs)} | Interval: {args.interval}min | Mode: {mode}")
    print(f"Dry run: {args.dry_run}")
    print()

    cycle = 0
    while True:
        cycle += 1
        label = f"[Cycle {cycle}]" if run_forever else f"[Cycle {cycle}/{total_cycles}]"
        print(label)
        run_one_cycle(pairs, dry_run=args.dry_run)

        if total_cycles is not None and cycle >= total_cycles:
            break

        next_time = datetime.now().timestamp() + args.interval * 60
        next_str = datetime.fromtimestamp(next_time).strftime("%H:%M:%S")
        print(f"Sleeping {args.interval}min, next cycle at {next_str} ...")
        time.sleep(args.interval * 60)

    print(f"=== All done. {cycle} cycles completed. ===")


if __name__ == "__main__":
    main()
