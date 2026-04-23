"""Per-item decision harness: current behavior wrapped as a SimulationHarness.

Each signal item gets an independent LLM call with forced tool_choice.
Supports two backends:
  - "anthropic" (default): Anthropic Claude API via anthropic SDK
  - "openai": Any OpenAI-compatible API (Qwen, DeepSeek, etc.) via openai SDK

Maintains a RollingContext for dedup awareness.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from proactive_eval.types import SignalItem, Tick, Event
from proactive_eval.simulation import SimulationHarness
from proactive_eval.simulation.context import RollingContext


# ---------------------------------------------------------------------------
# Tool schema
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
        "action the user should take in response."
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

TOOL_NAME_TO_ACTION = {
    "decide_skip": "skip",
    "decide_send": "send",
    "decide_send_with_action": "send_with_action",
}

# OpenAI-compatible tool format (for Qwen, DeepSeek, etc.)
DECIDE_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in DECIDE_TOOLS
]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_system_prompt(fixture: dict) -> str:
    """Compose the agent persona prompt from a fixture."""
    parts = []
    profile = fixture.get("eigenflux_profile", {})
    name = profile.get("agent_name", fixture.get("fixture_id", "agent"))
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


def format_item_block(item: dict) -> str:
    parts = [
        f"Source: {item.get('source', '')}",
        f"Pulled at: {item.get('received_at', '')}",
    ]
    meta = item.get("metadata", {}) or {}
    for k, v in meta.items():
        if v in (None, [], "", {}) or k in ("fixture_id", "pull_file"):
            continue
        parts.append(f"{k}: {v}")
    parts.append("")
    parts.append(item.get("content", ""))
    return "\n".join(parts)


def build_user_message(item: dict, context_block: str = "") -> str:
    msg = ""
    if context_block:
        msg += context_block + "\n\n"
    msg += "=== ITEM TO DECIDE ===\n\n"
    msg += format_item_block(item)
    msg += "\n\nCall the `decide` tool now with your decision."
    return msg


# ---------------------------------------------------------------------------
# Decision parsing & validation
# ---------------------------------------------------------------------------

def parse_decision(resp) -> dict:
    """Parse Anthropic API response for tool_use decision."""
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


def parse_decision_openai(resp) -> dict:
    """Parse OpenAI-compatible API response for function call decision."""
    msg = resp.choices[0].message if resp.choices else None
    if not msg:
        return {"action": "skip", "tool_name": None, "rationale": "empty response"}

    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        fn = tc.function
        name = fn.name
        if name in TOOL_NAME_TO_ACTION:
            try:
                args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
            except (json.JSONDecodeError, TypeError):
                args = {}
            decision = {"action": TOOL_NAME_TO_ACTION[name], "tool_name": name}
            if isinstance(args, dict):
                decision.update(args)
            return decision

    # Fallback: try parsing text content as JSON (some models don't use tool_calls)
    content = getattr(msg, "content", "") or ""
    if content.strip():
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "action" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

    return {"action": "skip", "tool_name": None, "rationale": "no function call in response"}


def validate_decision(decision: dict) -> tuple[bool, str]:
    action = decision.get("action")
    if action not in ("skip", "send", "send_with_action"):
        return False, f"invalid action: {action!r}"
    if not (decision.get("rationale") or "").strip():
        return False, f"action={action} requires non-empty 'rationale'"
    if action in ("send", "send_with_action") and not (decision.get("message") or "").strip():
        return False, f"action={action} requires non-empty 'message'"
    if action == "send_with_action":
        ra = decision.get("recommended_action") or {}
        if isinstance(ra, str):
            # LLM returned a plain string instead of structured dict — accept it
            if not ra.strip():
                return False, "send_with_action requires non-empty recommended_action"
        elif not ra.get("title") or not ra.get("details"):
            return False, "send_with_action requires recommended_action with title and details"
    return True, ""


def usage_dict(resp, backend: str = "anthropic") -> dict:
    usage = getattr(resp, "usage", None)
    if not usage:
        return {}
    if backend == "openai":
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


# ---------------------------------------------------------------------------
# PerItemHarness
# ---------------------------------------------------------------------------

class PerItemHarness(SimulationHarness):
    """Per-item decision harness.

    Each signal item gets an independent LLM call with forced tool_choice.
    Maintains a RollingContext for dedup awareness.

    Supports backend="anthropic" (default) or backend="openai" (Qwen, DeepSeek, etc.).
    """

    def __init__(self, client=None, model: str = "claude-haiku-4-5-20251001",
                 max_tokens: int = 1024, temperature: float | None = None,
                 window_size: int = 30, dry_run: bool = False,
                 workers: int = 4, backend: str = "anthropic"):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.window_size = window_size
        self.dry_run = dry_run
        self.workers = workers
        self.backend = backend
        self.system_prompt = ""
        self.rolling_ctx = RollingContext(window_size=window_size)
        self.fixture_id = ""
        self._total_processed = 0
        self._decisions_tally: dict[str, int] = {"skip": 0, "send": 0, "send_with_action": 0}
        self._validation_errors = 0
        self._api_errors = 0

    def initialize(self, fixture: dict, scenario: dict | None, config: dict) -> None:
        self.fixture_id = fixture.get("fixture_id", "")
        self.system_prompt = build_system_prompt(fixture)
        self.rolling_ctx = RollingContext(window_size=self.window_size)

    def process_tick(self, tick_index: int, signals: list[SignalItem],
                     time_window: tuple[datetime, datetime]) -> Tick:
        """Process all signals in this tick.

        Uses batch-parallel execution: items within a batch run concurrently
        (same context snapshot), then context is updated before the next batch.
        Batch size = self.workers.
        """
        tick = Tick(
            tick_index=tick_index,
            time_window=time_window,
            signals_received=signals,
        )

        if not signals:
            return tick

        item_dicts = [_signal_to_dict(s) for s in signals]
        raw_outputs = []

        # Process in batches of `workers` for concurrent API calls
        for batch_start in range(0, len(item_dicts), max(self.workers, 1)):
            batch = item_dicts[batch_start:batch_start + self.workers]

            if self.workers > 1 and len(batch) > 1 and not self.dry_run:
                # Parallel: snapshot context, run batch concurrently
                entries = self._decide_batch_parallel(batch)
            else:
                # Sequential fallback
                entries = [self._decide_one(item) for item in batch]

            for entry in entries:
                raw_outputs.append(json.dumps(entry, ensure_ascii=False))

                decision = entry.get("decision", {})
                self.rolling_ctx.add(
                    # recover the item dict from entry metadata
                    {"content": entry.get("content_preview", ""),
                     "source": entry.get("source", ""),
                     "received_at": entry.get("pull_timestamp", ""),
                     "metadata": entry.get("source_metadata", {}),
                     "raw_item": entry.get("raw_item", {})},
                    decision,
                )

                self._total_processed += 1
                action = decision.get("action", "skip")
                if action in self._decisions_tally:
                    self._decisions_tally[action] += 1
                if not (entry.get("validation") or {}).get("ok", True):
                    self._validation_errors += 1
                if (entry.get("api") or {}).get("error"):
                    self._api_errors += 1

        tick.raw_output = "\n".join(raw_outputs)
        return tick

    def _decide_batch_parallel(self, batch: list[dict]) -> list[dict]:
        """Run a batch of items concurrently using ThreadPoolExecutor."""
        results: list[tuple[int, dict]] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._decide_one, item): i
                       for i, item in enumerate(batch)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results.append((idx, future.result()))
                except Exception as exc:
                    results.append((idx, {
                        "decision_id": str(uuid.uuid4()),
                        "fixture_id": self.fixture_id,
                        "decision": {"action": "skip",
                                     "rationale": f"ThreadPool error: {exc}"},
                        "validation": {"ok": False, "error": str(exc)},
                        "api": {"error": str(exc)},
                    }))
        # Return in original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _decide_one(self, item: dict) -> dict:
        """Evaluate a single item. Returns a log entry dict."""
        context_block = self.rolling_ctx.format_block()
        user_msg = build_user_message(item, context_block)

        log_entry: dict = {
            "decision_id": str(uuid.uuid4()),
            "fixture_id": self.fixture_id,
            "model": self.model,
            "backend": self.backend,
            "source": item.get("source", ""),
            "pull_timestamp": item.get("received_at", ""),
            "source_metadata": item.get("metadata", {}),
            "content_preview": item.get("content", "")[:300],
            "raw_item": item.get("raw_item", {}),
            "decision_timestamp": None,
            "decision": None,
            "validation": None,
            "api": None,
            "dry_run": self.dry_run,
            "context": {
                "history_size": len(self.rolling_ctx.history),
                "dropped_so_far": self.rolling_ctx.dropped_count,
                "dropped_summary": dict(self.rolling_ctx.dropped_summary),
            },
        }

        if self.dry_run:
            log_entry["decision_timestamp"] = datetime.now(timezone.utc).isoformat()
            log_entry["decision"] = {"action": "skip", "rationale": "dry-run"}
            log_entry["validation"] = {"ok": True, "error": None}
            log_entry["api"] = {"input_tokens": None, "output_tokens": None,
                                "latency_ms": 0, "error": None}
            log_entry["prompt"] = {
                "system": self.system_prompt,
                "user": user_msg,
                "tools": [t["name"] for t in DECIDE_TOOLS],
                "tool_choice": {"type": "any"},
            }
            return log_entry

        t0 = time.time()
        api_error = None
        decision = None
        usage = {}
        try:
            if self.backend == "openai":
                decision, usage = self._call_openai(user_msg)
            else:
                decision, usage = self._call_anthropic(user_msg)
        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"
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

    def _call_anthropic(self, user_msg: str) -> tuple[dict, dict]:
        """Call Anthropic Claude API."""
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            tools=DECIDE_TOOLS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        return parse_decision(resp), usage_dict(resp, "anthropic")

    def _call_openai(self, user_msg: str) -> tuple[dict, dict]:
        """Call OpenAI-compatible API (Qwen, DeepSeek, etc.).

        Qwen3 thinking mode doesn't support tool_choice=required,
        so we disable thinking via extra_body and use tool_choice=required.
        If that fails, fall back to tool_choice=auto.
        """
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=DECIDE_TOOLS_OPENAI,
            tool_choice="required",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            # Disable Qwen3 thinking mode to support tool_choice=required
            # and reduce token waste
            extra_body={"enable_thinking": False},
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        return parse_decision_openai(resp), usage_dict(resp, "openai")

    def finalize(self) -> dict:
        return {
            "items_processed": self._total_processed,
            "decisions": dict(self._decisions_tally),
            "validation_errors": self._validation_errors,
            "api_errors": self._api_errors,
        }


def _signal_to_dict(signal: SignalItem) -> dict:
    """Convert a SignalItem to the dict format expected by prompt building."""
    return {
        "source": signal.source,
        "received_at": signal.received_at.isoformat() if signal.received_at else "",
        "content": signal.content,
        "url": signal.url,
        "metadata": signal.metadata,
        "raw_item": signal.raw_item,
    }
