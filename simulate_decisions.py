#!/usr/bin/env python3
"""
simulate_decisions.py — per-item decision simulation for proactive-eval.

Reads collected pulls under data/eigenflux/<fixture>/*.json and
data/searches/<fixture>/*.json, then for each item makes a single Claude call
that returns one of three decisions via the forced `decide` tool:

  - skip               — do nothing
  - send               — forward to the user with a message
  - send_with_action   — forward + recommend a concrete action

Items from the same pull share the pull timestamp but are presented one at a
time (per design: never batch a whole pull into a single model call). A
rolling context window of recent decisions is maintained for de-duplication;
when items fall out of the window we keep a tally so the model still knows
"how much has been processed and how it was triaged" without paying for the
full text.

Goal of this script: produce a complete, self-contained log under
data/decisions/<fixture>/<run_id>.jsonl that the (future) scoring module can
consume without re-running anything.

Usage:
  python3 simulate_decisions.py                              # all fixtures, both mode (see below)
  python3 simulate_decisions.py --fixture 1                  # only fixture 1 (1-indexed)
  python3 simulate_decisions.py --source eigenflux           # only EigenFlux pulls
  python3 simulate_decisions.py --source searches            # only search pulls
  python3 simulate_decisions.py --source both                # two runs: search_only + eigenflux_only
  python3 simulate_decisions.py --limit 5                    # at most 5 items per fixture
  python3 simulate_decisions.py --window 30                  # rolling-context size
  python3 simulate_decisions.py --model claude-haiku-4-5-20251001
  python3 simulate_decisions.py --start "2026-04-11T00:00" --end "2026-04-12T00:00"
  python3 simulate_decisions.py --dry-run                    # build prompts; no Claude call
"""

import argparse
import json
import math
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
EIGENFLUX_DIR = SCRIPT_DIR / "data" / "eigenflux"
SEARCHES_DIR = SCRIPT_DIR / "data" / "searches"
EXPERIMENTS_DIR = SCRIPT_DIR / "data" / "experiments"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024


def _rel_path(path: Path) -> str:
    """Stringify path relative to SCRIPT_DIR if possible, else absolute."""
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dependency)
# ---------------------------------------------------------------------------

def _load_env():
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixtures() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURES_DIR.glob("*.json"))]


def build_system_prompt(fixture: dict) -> str:
    """Compose the agent persona prompt from a fixture's workspace + memory + traits."""
    parts = []
    profile = fixture.get("eigenflux_profile", {})
    name = profile.get("agent_name", fixture["fixture_id"])
    parts.append(f"You are {name}, an autonomous personal assistant for one user.")
    parts.append("")

    workspace = fixture.get("workspace_files", {})
    if workspace:
        parts.append("# Persona files")
        for filename, body in workspace.items():
            parts.append(f"## {filename}")
            parts.append(body)
            parts.append("")

    memory = fixture.get("memory_files", {})
    if memory:
        parts.append("# Memory files")
        for filename, body in memory.items():
            parts.append(f"## {filename}")
            parts.append(body)
            parts.append("")

    traits = fixture.get("functional_traits", {})
    if traits:
        parts.append("# Functional traits")
        for k, v in traits.items():
            parts.append(f"- {k}: {v}")
        parts.append("")

    parts.append("# Decision protocol")
    parts.append(
        "You receive incoming information items one at a time. Each item is tagged with the "
        "timestamp at which it was pulled from the source.\n"
        "\n"
        "For every item you MUST call EXACTLY ONE of the following three tools:\n"
        "  - decide_skip               — the item is not worth surfacing to your user right now\n"
        "  - decide_send               — forward this to the user, with a message you write\n"
        "  - decide_send_with_action   — forward AND recommend a concrete action the user should take\n"
        "\n"
        "Each tool's input schema enforces the fields you must provide for that choice. Pick the "
        "tool whose semantics best match your judgment for this item.\n"
        "\n"
        "Apply your persona, traits, prior decisions and memory. Be conservative if your "
        "interruption_threshold is high; favor signal over noise. Avoid re-sending information the "
        "user has effectively already seen via this stream (the rolling history below is your record). "
        "When you choose decide_send or decide_send_with_action, write the message in the voice and "
        "language your user would expect from your persona."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Collected-data normalization
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime.
    If no timezone info is present, assume UTC."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def filter_items_by_time(items: list[dict], start: str | None, end: str | None) -> list[dict]:
    """Filter items whose pull_timestamp falls within [start, end)."""
    if start is None and end is None:
        return items
    start_dt = _parse_timestamp(start) if start else None
    end_dt = _parse_timestamp(end) if end else None
    filtered = []
    for item in items:
        ts = item.get("pull_timestamp", "")
        if not ts:
            continue
        try:
            item_dt = _parse_timestamp(ts)
        except (ValueError, TypeError):
            continue
        if start_dt and item_dt < start_dt:
            continue
        if end_dt and item_dt >= end_dt:
            continue
        filtered.append(item)
    return filtered


def _dedup_key(item: dict) -> str:
    """Return a stable dedup key for an item: URL for search, item_id for eigenflux."""
    if item["source"] == "search":
        return "search:" + (item["raw_item"].get("href") or item["content"])
    else:
        return "eigenflux:" + (item["source_metadata"].get("item_id") or item["content"])


def collect_items_for_fixture(fixture_id: str, source: str,
                              semantic_dedup: bool = True) -> list[dict]:
    """
    Walk collected pulls for one fixture and produce a chronologically sorted,
    deduplicated flat list of normalized items.  `source` is 'eigenflux',
    'searches', or 'both'.

    Dedup is two-phase (matching EigenFlux pipeline):
      1. URL/item_id exact dedup — keeps earliest occurrence
      2. Embedding semantic dedup — cosine similarity >= 0.70 considered duplicate
    """
    items: list[dict] = []
    if source in ("eigenflux", "both"):
        d = EIGENFLUX_DIR / fixture_id
        if d.exists():
            for p in sorted(d.glob("*.json")):
                items.extend(_normalize_eigenflux_file(p, fixture_id))
    if source in ("searches", "both"):
        d = SEARCHES_DIR / fixture_id
        if d.exists():
            for p in sorted(d.glob("*.json")):
                items.extend(_normalize_search_file(p, fixture_id))
    items.sort(key=lambda x: (x["pull_timestamp"], x["source"]))

    # Phase 1: URL / item_id exact dedup
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        key = _dedup_key(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    before_semantic = len(deduped)

    # Phase 2: embedding semantic dedup (search items only — eigenflux already
    # has server-side dedup via group_id)
    if semantic_dedup and source in ("searches", "both"):
        from proactive_eval.dedup import semantic_dedup as _sem_dedup
        search_items = [it for it in deduped if it["source"] == "search"]
        other_items = [it for it in deduped if it["source"] != "search"]
        if search_items:
            search_items = _sem_dedup(search_items, content_key="content", verbose=True)
        deduped = sorted(search_items + other_items,
                         key=lambda x: (x["pull_timestamp"], x["source"]))
        after_semantic = len(deduped)
        if before_semantic != after_semantic:
            print(f"    semantic dedup: {before_semantic} -> {after_semantic} "
                  f"(-{before_semantic - after_semantic})")

    return deduped


def _normalize_eigenflux_file(path: Path, fixture_id: str) -> list[dict]:
    data = json.loads(path.read_text())
    pull_ts = data.get("timestamp_utc", "")
    out = []
    for entry in data.get("detailed_items", []):
        item = entry.get("item") if isinstance(entry, dict) and "item" in entry else entry
        if not isinstance(item, dict) or "error" in item:
            continue
        content_parts = []
        if item.get("summary"):
            content_parts.append(f"Summary: {item['summary']}")
        if item.get("content"):
            content_parts.append(f"Content: {item['content']}")
        if item.get("url"):
            content_parts.append(f"URL: {item['url']}")
        out.append({
            "source": "eigenflux",
            "fixture_id": fixture_id,
            "pull_timestamp": pull_ts,
            "pull_file": _rel_path(path),
            "source_metadata": {
                "item_id": item.get("item_id"),
                "broadcast_type": item.get("broadcast_type"),
                "domains": item.get("domains", []),
                "geo": item.get("geo"),
                "keywords": item.get("keywords", []),
            },
            "content": "\n".join(content_parts),
            "raw_item": item,
        })
    return out


def _normalize_search_file(path: Path, fixture_id: str) -> list[dict]:
    data = json.loads(path.read_text())
    pull_ts = data.get("timestamp_utc", "")
    out = []
    for query_block in data.get("results", []):
        query = query_block.get("query", "")
        for raw in query_block.get("items", []):
            if not isinstance(raw, dict) or "error" in raw:
                continue
            content_parts = []
            if raw.get("title"):
                content_parts.append(f"Title: {raw['title']}")
            if raw.get("body"):
                content_parts.append(f"Body: {raw['body']}")
            if raw.get("href"):
                content_parts.append(f"URL: {raw['href']}")
            out.append({
                "source": "search",
                "fixture_id": fixture_id,
                "pull_timestamp": pull_ts,
                "pull_file": _rel_path(path),
                "source_metadata": {"query": query},
                "content": "\n".join(content_parts),
                "raw_item": raw,
            })
    return out


# ---------------------------------------------------------------------------
# Tool schema (forced via tool_choice)
# ---------------------------------------------------------------------------

DECIDE_SKIP_TOOL = {
    "name": "decide_skip",
    "description": (
        "Choose this when the incoming item is NOT worth surfacing to the user right now. "
        "Use when the item is noise, redundant with something already sent, off-topic, or "
        "below the persona's interruption threshold."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": "Brief reason for skipping (used by the evaluator).",
            },
        },
        "required": ["rationale"],
    },
}

DECIDE_SEND_TOOL = {
    "name": "decide_send",
    "description": (
        "Choose this when the item is worth forwarding to the user with a written message, "
        "but you are NOT recommending a specific concrete action they should take."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": "Brief reason for sending (used by the evaluator).",
            },
            "message": {
                "type": "string",
                "description": (
                    "User-facing message text. Write in the voice your persona would use, "
                    "in the user's preferred language."
                ),
            },
        },
        "required": ["rationale", "message"],
    },
}

DECIDE_SEND_WITH_ACTION_TOOL = {
    "name": "decide_send_with_action",
    "description": (
        "Choose this when the item is worth forwarding AND you want to recommend a concrete "
        "action the user should take in response (e.g. 'rebalance position', 'reply to X', "
        "'book a meeting'). Requires both a user-facing message and a structured action."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": "Brief reason for sending with an action (used by the evaluator).",
            },
            "message": {
                "type": "string",
                "description": (
                    "User-facing message text. Write in the voice your persona would use, "
                    "in the user's preferred language."
                ),
            },
            "recommended_action": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short imperative title for the action"},
                    "details": {"type": "string", "description": "Concrete details: what to do, how, when"},
                },
                "required": ["title", "details"],
                "description": "The concrete action you recommend the user take.",
            },
        },
        "required": ["rationale", "message", "recommended_action"],
    },
}

DECIDE_TOOLS = [DECIDE_SKIP_TOOL, DECIDE_SEND_TOOL, DECIDE_SEND_WITH_ACTION_TOOL]

# Map tool name -> action label stored in the decision record.
TOOL_NAME_TO_ACTION = {
    "decide_skip": "skip",
    "decide_send": "send",
    "decide_send_with_action": "send_with_action",
}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def format_item_block(item: dict) -> str:
    parts = [
        f"Source: {item['source']}",
        f"Pulled at: {item['pull_timestamp']}",
    ]
    meta = item.get("source_metadata", {}) or {}
    for k, v in meta.items():
        if v in (None, [], "", {}):
            continue
        parts.append(f"{k}: {v}")
    parts.append("")
    parts.append(item.get("content", ""))
    return "\n".join(parts)


def build_user_message(item: dict) -> str:
    return (
        "=== ITEM TO DECIDE ===\n\n"
        + format_item_block(item)
        + "\n\nCall the `decide` tool now with your decision."
    )


# ---------------------------------------------------------------------------
# Decision parsing & validation
# ---------------------------------------------------------------------------

def parse_decision(resp) -> dict:
    """Pull a decide_* tool input out of an Anthropic Messages API response.

    Returns a unified decision dict with an `action` field derived from the
    tool name, plus the original tool input fields (rationale, message,
    recommended_action) merged in.
    """
    for block in getattr(resp, "content", []) or []:
        btype = getattr(block, "type", None)
        bname = getattr(block, "name", None)
        if btype == "tool_use" and bname in TOOL_NAME_TO_ACTION:
            inp = getattr(block, "input", None)
            if isinstance(inp, dict):
                decision = {"action": TOOL_NAME_TO_ACTION[bname], "tool_name": bname}
                decision.update(inp)
                return decision
    return {"action": "skip", "tool_name": None, "rationale": "no decide_* tool_use in response"}


def validate_decision(decision: dict) -> tuple[bool, str]:
    """Defense-in-depth validation. With three tools and tool_choice='any',
    the schema-required fields should already be enforced by the API, but we
    keep this check so the evaluator can flag any drift."""
    action = decision.get("action")
    if action not in ("skip", "send", "send_with_action"):
        return False, f"invalid action: {action!r}"
    if not (decision.get("rationale") or "").strip():
        return False, f"action={action} requires non-empty 'rationale'"
    if action in ("send", "send_with_action") and not (decision.get("message") or "").strip():
        return False, f"action={action} requires non-empty 'message'"
    if action == "send_with_action":
        ra = decision.get("recommended_action") or {}
        if not ra.get("title") or not ra.get("details"):
            return False, "send_with_action requires recommended_action with title and details"
    return True, ""


# ---------------------------------------------------------------------------
# Single decision call
# ---------------------------------------------------------------------------

def call_decide(client, model: str, system_prompt: str, user_message: str,
                max_tokens: int = DEFAULT_MAX_TOKENS,
                temperature: float | None = None):
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        tools=DECIDE_TOOLS,
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_message}],
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    return client.messages.create(**kwargs)


def usage_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if not usage:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


# ---------------------------------------------------------------------------
# Per-fixture loop
# ---------------------------------------------------------------------------

def _decide_one(client, model: str, system_prompt: str, item: dict,
                fid: str, dry_run: bool,
                max_tokens: int = DEFAULT_MAX_TOKENS,
                temperature: float | None = None) -> dict:
    """Evaluate a single item independently. Returns a log entry dict."""
    user_msg = build_user_message(item)

    log_entry: dict = {
        "decision_id": str(uuid.uuid4()),
        "fixture_id": fid,
        "model": model,
        "source": item["source"],
        "pull_timestamp": item["pull_timestamp"],
        "pull_file": item["pull_file"],
        "source_metadata": item["source_metadata"],
        "content_preview": item["content"][:300],
        "raw_item": item["raw_item"],
        "decision_timestamp": None,
        "decision": None,
        "validation": None,
        "api": None,
        "dry_run": dry_run,
    }

    if dry_run:
        log_entry["decision_timestamp"] = datetime.now(timezone.utc).isoformat()
        log_entry["decision"] = {"action": "skip", "rationale": "dry-run"}
        log_entry["validation"] = {"ok": True, "error": None}
        log_entry["api"] = {"input_tokens": None, "output_tokens": None,
                            "latency_ms": 0, "error": None}
        log_entry["prompt"] = {
            "system": system_prompt,
            "user": user_msg,
            "tools": [t["name"] for t in DECIDE_TOOLS],
            "tool_choice": {"type": "any"},
        }
        return log_entry

    t0 = time.time()
    api_error = None
    decision = None
    try:
        resp = call_decide(client, model, system_prompt, user_msg,
                           max_tokens=max_tokens, temperature=temperature)
        decision = parse_decision(resp)
        usage = usage_dict(resp)
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
        usage = {}
    latency_ms = int((time.time() - t0) * 1000)

    if decision is None:
        decision = {"action": "skip", "rationale": f"API error: {api_error}"}
    ok, verr = validate_decision(decision)

    log_entry["decision_timestamp"] = datetime.now(timezone.utc).isoformat()
    log_entry["decision"] = decision
    log_entry["validation"] = {"ok": ok, "error": verr or None}
    log_entry["api"] = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "latency_ms": latency_ms,
        "error": api_error,
    }
    return log_entry


def simulate_fixture(client, fixture: dict, items: list[dict], model: str,
                     dry_run: bool, limit: int | None,
                     log_path: Path, workers: int = 4,
                     max_tokens: int = DEFAULT_MAX_TOKENS,
                     temperature: float | None = None) -> dict:
    fid = fixture["fixture_id"]
    system_prompt = build_system_prompt(fixture)
    summary = {
        "fixture_id": fid,
        "model": model,
        "items_total": len(items),
        "items_processed": 0,
        "decisions": {"skip": 0, "send": 0, "send_with_action": 0},
        "validation_errors": 0,
        "api_errors": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": _rel_path(log_path),
        "dry_run": dry_run,
    }

    if limit is not None:
        items = items[:limit]

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Each item is evaluated independently — parallelize API calls.
    completed = 0
    total = len(items)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_decide_one, client, model, system_prompt, item, fid, dry_run,
                        max_tokens, temperature): idx
            for idx, item in enumerate(items)
        }
        # Collect results, preserving original order for the log file.
        results: list[dict | None] = [None] * total
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "decision_id": str(uuid.uuid4()),
                    "fixture_id": fid, "model": model,
                    "decision": {"action": "skip", "rationale": f"worker error: {exc}"},
                    "validation": {"ok": False, "error": str(exc)},
                    "api": {"error": str(exc)},
                    "dry_run": dry_run,
                }
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"  {fid}: {completed}/{total}")

    # Write log in chronological order and tally.
    with log_path.open("w") as logf:
        for entry in results:
            if entry is None:
                continue
            logf.write(json.dumps(entry, ensure_ascii=False) + "\n")

            summary["items_processed"] += 1
            decision = entry.get("decision") or {}
            action = decision.get("action", "skip")
            if action in summary["decisions"]:
                summary["decisions"][action] += 1
            else:
                summary["decisions"]["skip"] += 1
            if not (entry.get("validation") or {}).get("ok", True):
                summary["validation_errors"] += 1
            if (entry.get("api") or {}).get("error"):
                summary["api_errors"] += 1

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sampling & confidence intervals
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion (95% CI by default).
    Returns (point_estimate, ci_low, ci_high)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return p_hat, max(0.0, centre - margin), min(1.0, centre + margin)


def sample_items(items: list[dict], sample_size: int | None, seed: int = 42) -> tuple[list[dict], int]:
    """Randomly sample items. Returns (sampled_items, population_size).
    If sample_size is None or >= population, return all items."""
    population = len(items)
    if sample_size is None or sample_size >= population:
        return items, population
    rng = random.Random(seed)
    sampled = rng.sample(items, sample_size)
    # Preserve chronological order
    index_set = set(id(s) for s in sampled)
    sampled_ordered = [it for it in items if id(it) in index_set]
    return sampled_ordered, population


def pre_step_clear_impressions(fixtures: list[dict]):
    """Clear server-side impression records for all fixture agents."""
    import urllib.request
    base_url = os.environ.get("EIGENFLUX_CONSOLE_URL", "http://localhost:8090") + "/console/api/v1/impr/items"
    for fixture in fixtures:
        agent_id = fixture.get("eigenflux_profile", {}).get("agent_id")
        if not agent_id:
            print(f"  [clear_impressions] skip {fixture['fixture_id']} (no agent_id)")
            continue
        url = f"{base_url}?agent_id={agent_id}"
        req = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"  [clear_impressions] {fixture['fixture_id']}: {resp.status}")
        except Exception as e:
            print(f"  [clear_impressions] {fixture['fixture_id']}: FAILED ({e})")


PRE_STEPS = {
    "clear_impressions": pre_step_clear_impressions,
}


def load_experiment_config(config_path: str) -> dict:
    """Load and return an experiment config JSON file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    return json.loads(path.read_text())


def _arg_was_defaulted(parser: argparse.ArgumentParser,
                       args: argparse.Namespace, dest: str) -> bool:
    """Return True if the arg at `dest` still holds its parser default
    (i.e. the user did not explicitly pass it on the CLI)."""
    return getattr(args, dest) == parser.get_default(dest)


def apply_config_to_args(parser: argparse.ArgumentParser,
                         args: argparse.Namespace,
                         cfg: dict) -> list[str]:
    """Merge experiment config into args namespace.

    Only overrides args that were NOT explicitly set on the CLI.
    Returns the list of pre_steps to run.
    """
    mapping = [
        # (config_key_path, args_dest, transform)
        (["model"], "model", None),
        (["max_tokens"], "max_tokens", None),
        (["temperature"], "temperature", None),
        (["source"], "source", None),
        (["time_range", "start"], "start", None),
        (["time_range", "end"], "end", None),
        (["sampling", "size"], "sample", None),
        (["sampling", "seed"], "seed", None),
        (["targets", "daily_sends"], "daily_target", None),
        (["targets", "send_rate"], "send_rate_target", None),
        (["targets", "action_rate"], "action_rate_target", None),
        (["workers"], "workers", None),
    ]

    for key_path, dest, transform in mapping:
        # Navigate nested config dict
        val = cfg
        for k in key_path:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = None
                break
        if val is None:
            continue
        if _arg_was_defaulted(parser, args, dest):
            if transform:
                val = transform(val)
            setattr(args, dest, val)

    # dedup: config["dedup"] -> args.no_semantic_dedup
    if "dedup" in cfg and _arg_was_defaulted(parser, args, "no_semantic_dedup"):
        args.no_semantic_dedup = (cfg["dedup"] != "semantic")

    # fixtures: config["fixtures"] -> args.fixture
    # Handled separately: "all" means None, list of ints means we store it
    # as a special attribute for multi-fixture selection.
    if "fixtures" in cfg and _arg_was_defaulted(parser, args, "fixture"):
        fix_val = cfg["fixtures"]
        if isinstance(fix_val, list):
            # Store as list of 1-indexed fixture indices for main() to handle
            args._config_fixture_indices = fix_val
        elif fix_val == "all":
            args.fixture = None  # already the default, but be explicit
        elif isinstance(fix_val, int):
            args.fixture = fix_val

    return cfg.get("pre_steps", [])


def make_experiment_id(model: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    return f"{ts}__{safe_model}"


def main():
    parser = argparse.ArgumentParser(description="Per-item decision simulation for proactive-eval")
    parser.add_argument("--fixture", type=int, default=None,
                        help="Only operate on the Nth fixture (1-indexed)")
    parser.add_argument("--source", choices=["eigenflux", "searches"], default="eigenflux",
                        help="Which collected source to feed to the model (default: eigenflux)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Anthropic model to call (default: {DEFAULT_MODEL})")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample size per (fixture, source). Omit to eval all items.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max items to process per fixture (default: no limit)")
    parser.add_argument("--start", default=None,
                        help="Only include items with pull_timestamp >= this (ISO 8601, e.g. 2026-04-11T00:00)")
    parser.add_argument("--end", default=None,
                        help="Only include items with pull_timestamp < this (ISO 8601, e.g. 2026-04-12T00:00)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and write log lines, but skip the Claude call")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max parallel sessions (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens for the model response (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for the model call (default: not set, use API default)")
    parser.add_argument("--no-semantic-dedup", action="store_true",
                        help="Disable embedding-based semantic dedup (URL dedup only)")
    parser.add_argument("--daily-target", type=float, default=15,
                        help="Baseline: min useful sends per day (default: 15)")
    parser.add_argument("--send-rate-target", type=float, default=0.70,
                        help="Baseline: min send rate (default: 0.70)")
    parser.add_argument("--action-rate-target", type=float, default=0.15,
                        help="Baseline: min action rate (default: 0.15)")
    parser.add_argument("--config", default=None,
                        help="Path to experiment config JSON. CLI flags override config values.")
    args = parser.parse_args()

    # --- Load experiment config if provided ---
    pre_steps: list[str] = []
    config_file_path = None
    if args.config:
        config_file_path = str(Path(args.config).resolve())
        experiment_cfg = load_experiment_config(args.config)
        pre_steps = apply_config_to_args(parser, args, experiment_cfg)
        print(f"Loaded config: {config_file_path}")
        if experiment_cfg.get("name"):
            print(f"  Experiment name: {experiment_cfg['name']}")

    all_fixtures = load_fixtures()

    # Handle multi-fixture selection from config (list of 1-indexed indices)
    config_fixture_indices = getattr(args, "_config_fixture_indices", None)
    if args.fixture is not None:
        idx = args.fixture - 1
        if idx < 0 or idx >= len(all_fixtures):
            print(f"Error: --fixture must be 1-{len(all_fixtures)}")
            sys.exit(1)
        fixtures = [all_fixtures[idx]]
    elif config_fixture_indices is not None:
        fixtures = []
        for ci in config_fixture_indices:
            idx = ci - 1  # config uses 1-indexed
            if 0 <= idx < len(all_fixtures):
                fixtures.append(all_fixtures[idx])
            else:
                print(f"Warning: fixture index {ci} out of range (1-{len(all_fixtures)}), skipping")
        if not fixtures:
            print("Error: no valid fixtures selected from config")
            sys.exit(1)
    else:
        fixtures = all_fixtures

    # --- Run pre-steps ---
    for step_name in pre_steps:
        handler = PRE_STEPS.get(step_name)
        if handler:
            print(f"Running pre-step: {step_name}")
            handler(fixtures)
        else:
            print(f"Warning: unknown pre-step '{step_name}', skipping")

    # Lazy import so --dry-run works without anthropic installed
    client = None
    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            print("Error: `anthropic` package not installed. `pip install anthropic` or use --dry-run.")
            sys.exit(1)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set (see proactive-eval/.env).")
            sys.exit(1)
        client = anthropic.Anthropic()

    # --- Experiment directory ---
    experiment_id = make_experiment_id(args.model)
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    semantic_dedup = not args.no_semantic_dedup
    config = {
        "experiment_id": experiment_id,
        "resolved_from": config_file_path,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "source": args.source,
        "workers": args.workers,
        "sample": args.sample,
        "seed": args.seed,
        "semantic_dedup": semantic_dedup,
        "limit": args.limit,
        "start": args.start,
        "end": args.end,
        "dry_run": args.dry_run,
        "pre_steps": pre_steps,
        "targets": {
            "daily_sends": args.daily_target,
            "send_rate": args.send_rate_target,
            "action_rate": args.action_rate_target,
        },
        "fixtures": [f["fixture_id"] for f in fixtures],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    print("=== Decision Simulation ===")
    print(f"Experiment: {experiment_id}")
    print(f"Output dir: {_rel_path(experiment_dir)}")
    print(f"Fixtures: {len(fixtures)}")
    sample_label = f"Sample: {args.sample}/group (seed={args.seed})" if args.sample else "Sample: all"
    dedup_label = "semantic" if semantic_dedup else "URL-only"
    print(f"Source: {args.source} | Model: {args.model} | Workers: {args.workers}"
          f" | {sample_label} | Dedup: {dedup_label}{' | DRY-RUN' if args.dry_run else ''}")
    if args.start or args.end:
        print(f"Time range: {args.start or '(open)'} -> {args.end or '(open)'}")
    if args.limit is not None:
        print(f"Limit: {args.limit} items per fixture")
    print()

    # Each experiment runs a single source
    source_runs = [(args.source, "results")]

    # Build the list of jobs: (fixture, run_label, items, population_size)
    jobs = []
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        for collect_source, run_label in source_runs:
            items = collect_items_for_fixture(fid, collect_source,
                                                semantic_dedup=semantic_dedup)
            if args.start or args.end:
                items = filter_items_by_time(items, args.start, args.end)
            if not items:
                print(f"--- {fid} [{run_label}]: 0 items (skipping)")
                continue
            sampled, population = sample_items(items, args.sample, seed=args.seed)
            if args.sample and len(sampled) < population:
                print(f"--- {fid} [{run_label}]: {len(sampled)}/{population} items (sampled)")
            else:
                print(f"--- {fid} [{run_label}]: {len(sampled)} items")
            jobs.append((fixture, run_label, sampled, population))

    print(f"\nRunning {len(jobs)} sessions ({args.workers} workers per session)...\n")

    all_summaries = []
    for fixture, run_label, items, population in jobs:
        fid = fixture["fixture_id"]
        fixture_dir = experiment_dir / fid
        fixture_dir.mkdir(parents=True, exist_ok=True)
        log_path = fixture_dir / f"{run_label}.jsonl"

        summary = simulate_fixture(
            client=client,
            fixture=fixture,
            items=items,
            model=args.model,
            dry_run=args.dry_run,
            limit=args.limit,
            log_path=log_path,
            workers=args.workers,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Add sampling stats & confidence intervals
        n = summary["items_processed"]
        send_count = summary["decisions"]["send"] + summary["decisions"]["send_with_action"]
        action_count = summary["decisions"]["send_with_action"]
        send_rate, send_ci_low, send_ci_high = wilson_ci(send_count, n)
        action_rate, action_ci_low, action_ci_high = wilson_ci(action_count, n)
        summary["population_size"] = population
        summary["sample_size"] = n
        summary["send_rate"] = round(send_rate, 4)
        summary["send_rate_ci95"] = [round(send_ci_low, 4), round(send_ci_high, 4)]
        summary["action_rate"] = round(action_rate, 4)
        summary["action_rate_ci95"] = [round(action_ci_low, 4), round(action_ci_high, 4)]

        # Quality score: skip=0, send=1, action=3. Per-item average. Target=1.0
        quality_total = (send_count - action_count) * 1 + action_count * 3
        summary["quality_total"] = quality_total
        summary["quality_per_item"] = round(quality_total / n, 4) if n else 0

        # Daily yield = population / days (items per day reaching the agent)
        days = 1.0
        if args.start and args.end:
            try:
                d0 = _parse_timestamp(args.start)
                d1 = _parse_timestamp(args.end)
                days = max((d1 - d0).total_seconds() / 86400, 0.01)
            except Exception:
                pass
        summary["daily_yield"] = round(population / days, 1)
        summary["targets"] = {
            "daily_sends": args.daily_target,
            "send_rate": args.send_rate_target,
            "action_rate": args.action_rate_target,
        }

        summary_path = fixture_dir / f"{run_label}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        all_summaries.append(summary)

        ci_str = (f"send={send_rate:.0%}[{send_ci_low:.0%},{send_ci_high:.0%}] "
                  f"action={action_rate:.0%}[{action_ci_low:.0%},{action_ci_high:.0%}]")
        if population > n:
            ci_str = f"{ci_str} (n={n}/{population})"
        print(f"  DONE {fid} [{run_label}]: {summary['decisions']} | {ci_str}")

    # Update config with final state
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    config["summaries"] = all_summaries
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    print("=== Done ===")
    print(f"  Experiment dir: {_rel_path(experiment_dir)}")
    if all_summaries:
        total_items = sum(s["items_processed"] for s in all_summaries)
        total_send = sum(s["decisions"]["send"] for s in all_summaries)
        total_send_action = sum(s["decisions"]["send_with_action"] for s in all_summaries)
        total_skip = sum(s["decisions"]["skip"] for s in all_summaries)
        print(f"  Total: {total_items} items, "
              f"send={total_send}, send_with_action={total_send_action}, skip={total_skip}")

    # Regenerate HTML so the dashboard picks up new results immediately
    try:
        from visualize_results import build_all_html
        build_all_html()
        print("  HTML updated.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
