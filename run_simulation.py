#!/usr/bin/env python3
"""
Proactive evaluation runner — without_eigenflux condition.

For each fixture-scenario pair, runs an episode of 8 heartbeat cycles. At each
heartbeat the agent sees its identity files, chat history, past decisions, and
web search results from the matching time window, then decides whether to
proactively message the user. Each run produces one episode trace.

Usage:
    python3 run_simulation.py                        # Run all 6 pairs
    python3 run_simulation.py --pair 0               # Run first pair only
    python3 run_simulation.py --pair 0 --cycle 0     # Run first pair, first cycle only
    python3 run_simulation.py --dry-run              # Show prompts without calling Claude
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

SCRIPT_DIR = Path(__file__).parent

# Load .env file (no python-dotenv dependency)
_env_file = SCRIPT_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

SCENARIOS_DIR = SCRIPT_DIR / "scenarios"
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
CONDITIONS_DIR = SCRIPT_DIR / "conditions"
SEARCH_DATA_DIR = SCRIPT_DIR / "data" / "searches"
OUTPUT_DIR = SCRIPT_DIR / "data" / "episodes"

MODEL = "claude-haiku-4-5-20251001"
NUM_CYCLES = 8
HEARTBEAT_INTERVAL_MIN = 5


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_pairs() -> list[dict]:
    """Build fixture-scenario pairs from directory contents."""
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    scenarios = sorted(SCENARIOS_DIR.glob("*.json"))
    pairs = []
    for fix_path, scen_path in zip(fixtures, scenarios):
        fixture = load_json(fix_path)
        scenario = load_json(scen_path)
        pairs.append({
            "fixture": fixture,
            "scenario": scenario,
            "fixture_path": fix_path,
            "scenario_path": scen_path,
        })
    return pairs


def get_search_results_for_cycle(fixture_id: str, cycle_index: int, num_cycles: int) -> list[dict]:
    """Pick search results from collected data, mapped to cycle time window.

    We have ~35 collected snapshots per fixture. Divide them evenly across
    the requested number of cycles and return the snapshot closest to each cycle.
    """
    search_dir = SEARCH_DATA_DIR / fixture_id
    if not search_dir.exists():
        return []

    files = sorted(f for f in search_dir.glob("*.json"))
    if not files:
        return []

    # Divide files evenly across cycles
    chunk_size = max(1, len(files) // num_cycles)
    start = cycle_index * chunk_size
    end = start + chunk_size
    # For the last cycle, take all remaining
    if cycle_index == num_cycles - 1:
        end = len(files)

    selected = files[start:end]
    if not selected:
        selected = [files[-1]]

    # Use the last file in the window (most recent for that cycle)
    chosen = selected[-1]
    data = load_json(chosen)
    return data.get("results", [])


def build_system_prompt(fixture: dict) -> str:
    """Build the agent's system prompt from fixture workspace files."""
    ws = fixture.get("workspace_files", {})
    parts = []
    for filename in ["SOUL.md", "IDENTITY.md", "USER.md", "HEARTBEAT.md"]:
        if filename in ws:
            parts.append(f"# {filename}\n{ws[filename]}")

    # Memory files
    mem = fixture.get("memory_files", {})
    if mem:
        parts.append("# MEMORY")
        for path, content in mem.items():
            parts.append(f"## {path}\n{content}")

    # Functional traits
    traits = fixture.get("functional_traits", {})
    if traits:
        parts.append(f"# Functional Traits\n{json.dumps(traits, indent=2)}")

    parts.append(HEARTBEAT_INSTRUCTION)
    return "\n\n---\n\n".join(parts)


HEARTBEAT_INSTRUCTION = """# Heartbeat Protocol

You are now in a HEARTBEAT cycle. This is a periodic check-in where you decide
whether to proactively message your user.

You have access to web search results collected since the last heartbeat.
Review them and decide: is there anything worth surfacing to the user RIGHT NOW?

Consider:
- Is this information material enough to justify an interruption?
- Does it match what the user asked you to watch for?
- Is it timely — would waiting another 30 minutes make it stale?
- Have you already sent something about this topic in a previous cycle?

You MUST respond with a JSON object (and nothing else):

{
  "action": "send" or "skip",
  "reasoning": "your internal reasoning for why you chose to send or skip",
  "message": "the message you would send to the user (null if skip)",
  "signals_used": ["list of search queries or result titles that informed your decision"],
  "confidence": 0.0 to 1.0
}

Be judicious. The user trusts you to filter noise. A skip is often the right call."""


def build_cycle_message(
    scenario: dict,
    cycle_index: int,
    search_results: list[dict],
    prior_decisions: list[dict],
) -> str:
    """Build the user-turn message for one heartbeat cycle."""
    start_time = datetime.fromisoformat(scenario["start_time"])
    cycle_time = start_time + timedelta(minutes=HEARTBEAT_INTERVAL_MIN * cycle_index)

    parts = []

    # Timing context
    parts.append(f"## Heartbeat Cycle {cycle_index + 1}/{NUM_CYCLES}")
    parts.append(f"Current time: {cycle_time.strftime('%Y-%m-%d %H:%M UTC')}")
    parts.append("")

    # Chat history (only in first cycle)
    if cycle_index == 0:
        parts.append("## Chat History (prep stage)")
        for msg in scenario.get("history", []):
            sender = msg.get("sender_name", "Assistant")
            if msg["type"] == "user_chat":
                parts.append(f"**{sender}**: {msg['text']}")
            else:
                parts.append(f"**You**: {msg['text']}")
        parts.append("")

    # Prior decisions
    if prior_decisions:
        parts.append("## Your Previous Decisions This Session")
        for i, dec in enumerate(prior_decisions):
            action = dec.get("action", "unknown")
            msg_preview = ""
            if action == "send" and dec.get("message"):
                msg_preview = f" — \"{dec['message'][:80]}...\""
            parts.append(f"- Cycle {i + 1}: **{action}**{msg_preview}")
        parts.append("")

    # Search results
    if search_results:
        parts.append("## Web Search Results (collected this cycle)")
        for sr in search_results:
            query = sr.get("query", "unknown query")
            items = sr.get("items", [])
            parts.append(f"\n### Query: \"{query}\"")
            for item in items:
                if "error" in item:
                    parts.append(f"- (error: {item['error']})")
                else:
                    title = item.get("title", "No title")
                    body = item.get("body", "")
                    href = item.get("href", "")
                    parts.append(f"- **{title}**")
                    if href:
                        parts.append(f"  {href}")
                    if body:
                        parts.append(f"  {body[:200]}")
    else:
        parts.append("## Web Search Results\nNo new search results available this cycle.")

    return "\n".join(parts)


def parse_agent_response(text: str) -> dict:
    """Extract JSON decision from agent response."""
    # Try to find JSON in the response
    text = text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "action": "error",
            "reasoning": "Failed to parse response",
            "message": None,
            "signals_used": [],
            "confidence": 0.0,
            "raw_response": text[:500],
        }


def run_episode(pair: dict, pair_index: int, dry_run: bool = False,
                single_cycle: int | None = None, model: str = MODEL) -> dict:
    """Run a full episode for one fixture-scenario pair."""
    fixture = pair["fixture"]
    scenario = pair["scenario"]
    fixture_id = fixture["fixture_id"]
    scenario_id = scenario["scenario_id"]

    print(f"\n{'='*60}")
    print(f"Pair {pair_index}: {fixture_id} × {scenario_id}")
    print(f"{'='*60}")

    system_prompt = build_system_prompt(fixture)
    prior_decisions = []

    episode = {
        "scenario_id": scenario_id,
        "fixture_id": fixture_id,
        "condition": "without_eigenflux",
        "model": model,
        "num_cycles": NUM_CYCLES,
        "heartbeat_interval_min": HEARTBEAT_INTERVAL_MIN,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "system_prompt": system_prompt,
            "scenario": scenario,
            "fixture": fixture,
        },
        "cycles": [],
    }

    client = None
    if not dry_run:
        client = anthropic.Anthropic()

    cycles_to_run = range(NUM_CYCLES)
    if single_cycle is not None:
        cycles_to_run = [single_cycle]

    for cycle_index in cycles_to_run:
        print(f"\n--- Cycle {cycle_index + 1}/{NUM_CYCLES} ---")

        search_results = get_search_results_for_cycle(fixture_id, cycle_index, NUM_CYCLES)
        print(f"  Search results: {len(search_results)} query groups")

        user_message = build_cycle_message(scenario, cycle_index, search_results, prior_decisions)

        if dry_run:
            print(f"  [DRY RUN] System prompt: {len(system_prompt)} chars")
            print(f"  [DRY RUN] User message: {len(user_message)} chars")
            cycle_record = {
                "cycle": cycle_index + 1,
                "dry_run": True,
                "input": {
                    "user_message": user_message,
                    "search_results": search_results,
                    "prior_decisions_snapshot": list(prior_decisions),
                },
            }
            episode["cycles"].append(cycle_record)
            continue

        # Call Claude
        print(f"  Calling {model}...")
        request_payload = {
            "model": model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        request_started_at = datetime.now(timezone.utc).isoformat()
        response = client.messages.create(**request_payload)
        request_completed_at = datetime.now(timezone.utc).isoformat()

        raw_text = response.content[0].text
        decision = parse_agent_response(raw_text)
        prior_decisions.append(decision)

        action = decision.get("action", "?")
        confidence = decision.get("confidence", "?")
        print(f"  Decision: {action} (confidence: {confidence})")
        if action == "send" and decision.get("message"):
            print(f"  Message: {decision['message'][:100]}...")

        cycle_record = {
            "cycle": cycle_index + 1,
            "request_started_at": request_started_at,
            "request_completed_at": request_completed_at,
            "input": {
                "user_message": user_message,
                "search_results": search_results,
                "prior_decisions_snapshot": list(prior_decisions[:-1]),
            },
            "request": {
                "model": model,
                "max_tokens": request_payload["max_tokens"],
                "messages": request_payload["messages"],
                # system_prompt is stored once at episode["inputs"]; not duplicated per cycle
            },
            "output": {
                "raw_response": raw_text,
                "decision": decision,
                "stop_reason": response.stop_reason,
            },
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
        episode["cycles"].append(cycle_record)

    episode["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Count actions
    def _action(c: dict) -> str:
        return c.get("output", {}).get("decision", {}).get("action", "")

    sends = sum(1 for c in episode["cycles"] if _action(c) == "send")
    skips = sum(1 for c in episode["cycles"] if _action(c) == "skip")
    total_input_tokens = sum(c.get("usage", {}).get("input_tokens", 0) for c in episode["cycles"])
    total_output_tokens = sum(c.get("usage", {}).get("output_tokens", 0) for c in episode["cycles"])
    episode["summary"] = {
        "total_sends": sends,
        "total_skips": skips,
        "send_rate": sends / max(1, sends + skips),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    return episode


def save_episode(episode: dict):
    """Save episode trace."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture_id = episode["fixture_id"]
    scenario_id = episode["scenario_id"]
    condition = episode["condition"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"{condition}__{fixture_id}__{scenario_id}__{ts}.json"

    with open(filename, "w") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Proactive eval simulator")
    parser.add_argument("--pair", type=int, default=None, help="Run only the Nth pair (0-indexed)")
    parser.add_argument("--cycle", type=int, default=None, help="Run only the Nth cycle (0-indexed)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without calling Claude")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Model to use (default: {MODEL})")
    args = parser.parse_args()

    model = args.model

    pairs = get_pairs()
    print(f"Loaded {len(pairs)} fixture-episode pairs")

    if args.pair is not None:
        if args.pair < 0 or args.pair >= len(pairs):
            print(f"Error: --pair must be 0-{len(pairs)-1}")
            sys.exit(1)
        pairs_to_run = [(args.pair, pairs[args.pair])]
    else:
        pairs_to_run = list(enumerate(pairs))

    for pair_index, pair in pairs_to_run:
        episode = run_episode(pair, pair_index, dry_run=args.dry_run, single_cycle=args.cycle, model=model)
        if not args.dry_run:
            save_episode(episode)

        # Print summary
        summary = episode.get("summary", {})
        if summary:
            print(f"\n  Summary: {summary['total_sends']} sends, "
                  f"{summary['total_skips']} skips, "
                  f"send rate: {summary['send_rate']:.0%}")

    print(f"\n{'='*60}")
    print("All episodes complete.")


if __name__ == "__main__":
    main()
