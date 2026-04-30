#!/usr/bin/env python3
"""
Tests for simulate_decisions.py.

These tests do NOT make any real Anthropic API calls. They mock the client so
the per-item flow, rolling-context behavior, decision validation, log-line
shape, and CLI semantics can all be verified offline.

Run as:

    cd proactive-eval
    python3 -m unittest tests/test_simulate_decisions.py -v

or:

    python3 tests/test_simulate_decisions.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "legacy"))

import simulate_decisions as sd  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — fake Anthropic responses
# ---------------------------------------------------------------------------

def make_tool_use_response(action: str, rationale: str = "ok",
                           message: str = "", recommended_action: dict | None = None,
                           usage_in: int = 100, usage_out: int = 50):
    """Build a SimpleNamespace mimicking an Anthropic Messages API response."""
    inp = {"action": action, "rationale": rationale}
    if message:
        inp["message"] = message
    if recommended_action:
        inp["recommended_action"] = recommended_action
    block = SimpleNamespace(type="tool_use", name="decide", input=inp, id="toolu_123")
    return SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(input_tokens=usage_in, output_tokens=usage_out),
    )


def make_text_only_response():
    """A pathological response: model returned text instead of tool_use."""
    block = SimpleNamespace(type="text", text="I'm sorry Dave...")
    return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def make_fixture(fid="test_fix"):
    return {
        "fixture_id": fid,
        "workspace_files": {
            "SOUL.md": "- Vibe: cautious",
            "USER.md": "- Timezone: UTC",
        },
        "memory_files": {"memory/note.md": "User likes brevity."},
        "functional_traits": {"interruption_threshold": "high"},
        "eigenflux_profile": {"agent_name": "tester", "bio": "test bio"},
    }


def make_item(idx: int, source: str = "eigenflux", pull_ts: str = "2026-04-10T16:00:00+00:00"):
    return {
        "source": source,
        "fixture_id": "test_fix",
        "pull_timestamp": pull_ts,
        "pull_file": f"data/{source}/test_fix/fake.json",
        "source_metadata": {"item_id": f"id-{idx}"} if source == "eigenflux"
                           else {"query": f"q-{idx}"},
        "content": f"Content of item {idx} " + ("x" * 100),
        "raw_item": {"item_id": f"id-{idx}"},
    }


# ---------------------------------------------------------------------------
# Fixture loading + system prompt
# ---------------------------------------------------------------------------

class FixtureLoadingTests(unittest.TestCase):
    def test_load_fixtures_returns_six(self):
        fixtures = sd.load_fixtures()
        self.assertEqual(len(fixtures), 6)

    def test_build_system_prompt_includes_persona_traits_and_protocol(self):
        fixture = make_fixture()
        prompt = sd.build_system_prompt(fixture)
        self.assertIn("tester", prompt)
        self.assertIn("SOUL.md", prompt)
        self.assertIn("- Vibe: cautious", prompt)
        self.assertIn("memory/note.md", prompt)
        self.assertIn("interruption_threshold", prompt)
        self.assertIn("Decision protocol", prompt)
        self.assertIn("skip", prompt)
        self.assertIn("send", prompt)
        self.assertIn("send_with_action", prompt)
        self.assertIn("decide", prompt)


# ---------------------------------------------------------------------------
# Data normalization — eigenflux + searches
# ---------------------------------------------------------------------------

class NormalizationTests(unittest.TestCase):
    def test_normalize_eigenflux_file_extracts_one_item_per_detailed_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "20260410_120000.json"
            f.write_text(json.dumps({
                "source": "eigenflux",
                "fixture_id": "fix",
                "timestamp_utc": "2026-04-10T12:00:00+00:00",
                "detailed_items": [
                    {"item": {"item_id": "a", "summary": "first", "content": "body1",
                              "url": "http://x/a", "domains": ["d1"], "broadcast_type": "info",
                              "geo": "US", "keywords": ["k1"]}},
                    {"item": {"item_id": "b", "summary": "second", "content": "body2",
                              "url": "http://x/b", "domains": ["d2"], "broadcast_type": "info",
                              "geo": "GLOBAL", "keywords": []}},
                    {"item_id": "c", "error": "fetch failed"},  # should be filtered out
                ],
            }))
            items = sd._normalize_eigenflux_file(f, "fix")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["source"], "eigenflux")
        self.assertEqual(items[0]["pull_timestamp"], "2026-04-10T12:00:00+00:00")
        self.assertEqual(items[0]["source_metadata"]["item_id"], "a")
        self.assertIn("Summary: first", items[0]["content"])
        self.assertIn("Content: body1", items[0]["content"])
        self.assertIn("URL: http://x/a", items[0]["content"])

    def test_normalize_search_file_creates_one_item_per_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "20260410_120000.json"
            f.write_text(json.dumps({
                "fixture_id": "fix",
                "timestamp_utc": "2026-04-10T12:00:00+00:00",
                "query_count": 2,
                "results": [
                    {"query": "q1", "items": [
                        {"title": "T1", "body": "B1", "href": "http://x/1"},
                        {"title": "T2", "body": "B2", "href": "http://x/2"},
                    ]},
                    {"query": "q2", "items": [
                        {"title": "T3", "body": "B3", "href": "http://x/3"},
                        {"error": "rate limited"},  # should be filtered out
                    ]},
                ],
            }))
            items = sd._normalize_search_file(f, "fix")
        # 2 + 1 (after dropping error) = 3 items
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["source"], "search")
        self.assertEqual(items[0]["source_metadata"]["query"], "q1")
        self.assertIn("Title: T1", items[0]["content"])
        # All share the pull timestamp
        for it in items:
            self.assertEqual(it["pull_timestamp"], "2026-04-10T12:00:00+00:00")

    def test_normalize_search_file_one_pull_many_items_share_timestamp(self):
        """Per spec: a pull with N items becomes N model calls all tagged with the pull time."""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "p.json"
            f.write_text(json.dumps({
                "fixture_id": "fix",
                "timestamp_utc": "2026-04-10T12:00:00+00:00",
                "results": [{"query": "q", "items": [
                    {"title": f"T{i}", "body": f"B{i}", "href": f"u{i}"}
                    for i in range(10)
                ]}],
            }))
            items = sd._normalize_search_file(f, "fix")
        self.assertEqual(len(items), 10)
        timestamps = {i["pull_timestamp"] for i in items}
        self.assertEqual(len(timestamps), 1, "all items from one pull must share the timestamp")


# ---------------------------------------------------------------------------
# Decision parsing & validation
# ---------------------------------------------------------------------------

class DecisionParsingTests(unittest.TestCase):
    def test_parse_decision_extracts_tool_use_input(self):
        resp = make_tool_use_response("send", "important news", message="hello user")
        decision = sd.parse_decision(resp)
        self.assertEqual(decision["action"], "send")
        self.assertEqual(decision["message"], "hello user")
        self.assertEqual(decision["rationale"], "important news")

    def test_parse_decision_falls_back_when_no_tool_use(self):
        resp = make_text_only_response()
        decision = sd.parse_decision(resp)
        self.assertEqual(decision["action"], "skip")
        self.assertIn("no decide tool_use", decision["rationale"])

    def test_validate_skip_is_valid(self):
        ok, err = sd.validate_decision({"action": "skip", "rationale": "noise"})
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_validate_send_requires_message(self):
        ok, err = sd.validate_decision({"action": "send", "rationale": "x"})
        self.assertFalse(ok)
        self.assertIn("message", err)
        ok, _ = sd.validate_decision({"action": "send", "rationale": "x", "message": "hi"})
        self.assertTrue(ok)

    def test_validate_send_with_action_requires_message_and_action(self):
        # Missing message
        ok, _ = sd.validate_decision({"action": "send_with_action", "rationale": "x",
                                      "recommended_action": {"title": "t", "details": "d"}})
        self.assertFalse(ok)
        # Missing recommended_action
        ok, _ = sd.validate_decision({"action": "send_with_action", "rationale": "x", "message": "m"})
        self.assertFalse(ok)
        # Recommended_action missing title
        ok, _ = sd.validate_decision({"action": "send_with_action", "rationale": "x", "message": "m",
                                      "recommended_action": {"details": "d"}})
        self.assertFalse(ok)
        # Valid
        ok, _ = sd.validate_decision({"action": "send_with_action", "rationale": "x", "message": "m",
                                      "recommended_action": {"title": "t", "details": "d"}})
        self.assertTrue(ok)

    def test_validate_unknown_action(self):
        ok, err = sd.validate_decision({"action": "delete_universe", "rationale": "x"})
        self.assertFalse(ok)
        self.assertIn("invalid action", err)


# ---------------------------------------------------------------------------
# Rolling context — window + token budget + drop tally
# ---------------------------------------------------------------------------

class RollingContextTests(unittest.TestCase):
    def test_window_drops_oldest_when_exceeded(self):
        ctx = sd.RollingContext(window_size=3)
        for i in range(5):
            ctx.add(make_item(i), {"action": "skip", "rationale": "n"})
        self.assertEqual(len(ctx.history), 3)
        self.assertEqual(ctx.dropped_count, 2)
        self.assertEqual(ctx.dropped_summary["skip"], 2)

    def test_drop_tally_tracks_action_types(self):
        ctx = sd.RollingContext(window_size=2)
        ctx.add(make_item(1), {"action": "send", "rationale": "n", "message": "m"})
        ctx.add(make_item(2), {"action": "send_with_action", "rationale": "n",
                               "message": "m", "recommended_action": {"title": "t", "details": "d"}})
        ctx.add(make_item(3), {"action": "skip", "rationale": "n"})  # forces drop of item 1
        ctx.add(make_item(4), {"action": "skip", "rationale": "n"})  # forces drop of item 2
        self.assertEqual(ctx.dropped_count, 2)
        self.assertEqual(ctx.dropped_summary["send"], 1)
        self.assertEqual(ctx.dropped_summary["send_with_action"], 1)

    def test_trim_to_token_budget_drops_when_over(self):
        # Tiny budget forces dropping
        ctx = sd.RollingContext(window_size=100)
        for i in range(20):
            ctx.add(make_item(i), {"action": "skip", "rationale": "n"})
        new_item = make_item(99)
        # Force a stupidly low budget; should trim hard
        trimmed = ctx.trim_to_token_budget("system " * 100, new_item, budget=200)
        self.assertGreater(trimmed, 0)
        # System + new item alone exceeds 200, so even history should be empty
        self.assertEqual(len(ctx.history), 0)
        self.assertEqual(ctx.dropped_count, 20)

    def test_history_block_includes_drop_note_when_dropped(self):
        ctx = sd.RollingContext(window_size=1)
        ctx.add(make_item(1), {"action": "send", "rationale": "n", "message": "hi"})
        ctx.add(make_item(2), {"action": "skip", "rationale": "n"})  # drops item 1
        block = sd.format_history_block(ctx.history, ctx.dropped_count, ctx.dropped_summary)
        self.assertIn("CONTEXT NOTE", block)
        self.assertIn("sent=1", block)
        self.assertIn("skipped=0", block)


# ---------------------------------------------------------------------------
# build_user_message — shape & invariants
# ---------------------------------------------------------------------------

class UserMessageTests(unittest.TestCase):
    def test_user_message_includes_new_item_section(self):
        ctx = sd.RollingContext(window_size=10)
        item = make_item(1)
        msg = sd.build_user_message(item, ctx.history, ctx.dropped_count, ctx.dropped_summary)
        self.assertIn("=== NEW ITEM TO DECIDE ===", msg)
        self.assertIn("Pulled at: 2026-04-10T16:00:00+00:00", msg)
        self.assertIn("Source: eigenflux", msg)
        self.assertIn("Call the `decide` tool now", msg)

    def test_user_message_history_grows_with_decisions(self):
        ctx = sd.RollingContext(window_size=10)
        ctx.add(make_item(1), {"action": "skip", "rationale": "n"})
        ctx.add(make_item(2), {"action": "send", "rationale": "n", "message": "hi"})
        msg = sd.build_user_message(make_item(3), ctx.history, ctx.dropped_count, ctx.dropped_summary)
        self.assertIn("SKIP", msg)
        self.assertIn("SEND", msg)
        self.assertIn("hi", msg)


# ---------------------------------------------------------------------------
# simulate_fixture — log shape, single API call per item, no batching
# ---------------------------------------------------------------------------

class SimulateFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = Path(self.tmpdir) / "run.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_one_call_per_item_not_batched(self):
        """Even if 5 items share a pull timestamp, the model is called 5 times."""
        client = MagicMock()
        client.messages.create.side_effect = [
            make_tool_use_response("skip"),
            make_tool_use_response("send", message="m"),
            make_tool_use_response("skip"),
            make_tool_use_response("send_with_action", message="m",
                                   recommended_action={"title": "t", "details": "d"}),
            make_tool_use_response("skip"),
        ]
        items = [make_item(i, pull_ts="2026-04-10T16:00:00+00:00") for i in range(5)]
        fixture = make_fixture()

        summary = sd.simulate_fixture(
            client=client, fixture=fixture, items=items, model="m",
            window=10, dry_run=False, limit=None, log_path=self.log_path,
        )

        self.assertEqual(client.messages.create.call_count, 5,
                         "must call API once per item, never batch")
        self.assertEqual(summary["items_processed"], 5)
        self.assertEqual(summary["decisions"]["skip"], 3)
        self.assertEqual(summary["decisions"]["send"], 1)
        self.assertEqual(summary["decisions"]["send_with_action"], 1)

    def test_log_line_shape_is_complete(self):
        client = MagicMock()
        client.messages.create.return_value = make_tool_use_response(
            "send_with_action", rationale="material catalyst",
            message="NVDA earnings beat by 8%",
            recommended_action={"title": "Review NVDA position",
                                "details": "Consider trimming if hits $850"},
        )
        fixture = make_fixture()
        items = [make_item(0)]

        sd.simulate_fixture(
            client=client, fixture=fixture, items=items, model="claude-test",
            window=10, dry_run=False, limit=None, log_path=self.log_path,
        )

        lines = self.log_path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])

        # Identity & traceability
        for field in ("decision_id", "fixture_id", "model", "source",
                      "pull_timestamp", "decision_timestamp", "pull_file",
                      "source_metadata", "content_preview", "raw_item",
                      "context", "decision", "validation", "api"):
            self.assertIn(field, entry)

        # Context block must capture rolling state
        self.assertIn("history_size", entry["context"])
        self.assertIn("dropped_so_far", entry["context"])
        self.assertIn("dropped_summary", entry["context"])
        self.assertIn("estimated_input_tokens", entry["context"])

        # Decision block fully captured
        self.assertEqual(entry["decision"]["action"], "send_with_action")
        self.assertEqual(entry["decision"]["message"], "NVDA earnings beat by 8%")
        self.assertEqual(entry["decision"]["recommended_action"]["title"], "Review NVDA position")

        # Validation passed
        self.assertTrue(entry["validation"]["ok"])

        # API usage captured
        self.assertEqual(entry["api"]["input_tokens"], 100)
        self.assertEqual(entry["api"]["output_tokens"], 50)
        self.assertIsNone(entry["api"]["error"])

    def test_api_error_is_logged_not_raised(self):
        """A model call that raises must be caught, logged, and the loop continues."""
        client = MagicMock()
        client.messages.create.side_effect = [
            RuntimeError("rate limited"),
            make_tool_use_response("skip"),
        ]
        items = [make_item(0), make_item(1)]
        fixture = make_fixture()

        summary = sd.simulate_fixture(
            client=client, fixture=fixture, items=items, model="m",
            window=10, dry_run=False, limit=None, log_path=self.log_path,
        )
        self.assertEqual(summary["items_processed"], 2)
        self.assertEqual(summary["api_errors"], 1)
        lines = [json.loads(l) for l in self.log_path.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertIn("rate limited", lines[0]["api"]["error"])
        self.assertEqual(lines[0]["decision"]["action"], "skip")  # fallback

    def test_dry_run_makes_no_api_calls(self):
        client = MagicMock()
        client.messages.create.side_effect = AssertionError("API must not be called in dry-run")
        items = [make_item(i) for i in range(3)]
        fixture = make_fixture()

        summary = sd.simulate_fixture(
            client=client, fixture=fixture, items=items, model="m",
            window=10, dry_run=True, limit=None, log_path=self.log_path,
        )
        self.assertEqual(summary["items_processed"], 3)
        self.assertEqual(client.messages.create.call_count, 0)
        # All entries marked dry_run + skip
        for line in self.log_path.read_text().splitlines():
            entry = json.loads(line)
            self.assertTrue(entry["dry_run"])
            self.assertEqual(entry["decision"]["action"], "skip")

    def test_limit_respects_max_items_per_fixture(self):
        client = MagicMock()
        client.messages.create.return_value = make_tool_use_response("skip")
        items = [make_item(i) for i in range(20)]
        fixture = make_fixture()

        summary = sd.simulate_fixture(
            client=client, fixture=fixture, items=items, model="m",
            window=10, dry_run=False, limit=5, log_path=self.log_path,
        )
        self.assertEqual(summary["items_processed"], 5)
        self.assertEqual(client.messages.create.call_count, 5)

    def test_rolling_history_visible_in_subsequent_calls(self):
        """Each call's user message must include the prior decisions."""
        client = MagicMock()
        client.messages.create.side_effect = [
            make_tool_use_response("send", message="first"),
            make_tool_use_response("skip"),
            make_tool_use_response("skip"),
        ]
        items = [make_item(i) for i in range(3)]
        fixture = make_fixture()

        sd.simulate_fixture(
            client=client, fixture=fixture, items=items, model="m",
            window=10, dry_run=False, limit=None, log_path=self.log_path,
        )

        # First call has empty history
        first_call_msgs = client.messages.create.call_args_list[0].kwargs["messages"]
        self.assertNotIn("SEND:", first_call_msgs[0]["content"])
        self.assertIn("(No prior items in this session.)", first_call_msgs[0]["content"])

        # Second call must show the first decision in history
        second_call_msgs = client.messages.create.call_args_list[1].kwargs["messages"]
        self.assertIn("SEND", second_call_msgs[0]["content"])
        self.assertIn("first", second_call_msgs[0]["content"])

    def test_window_overflow_logs_dropped_summary(self):
        """If a fixture has more items than the window, drops should appear in subsequent log entries."""
        client = MagicMock()
        client.messages.create.return_value = make_tool_use_response("skip")
        items = [make_item(i) for i in range(7)]
        fixture = make_fixture()

        sd.simulate_fixture(
            client=client, fixture=fixture, items=items, model="m",
            window=3, dry_run=False, limit=None, log_path=self.log_path,
        )
        entries = [json.loads(l) for l in self.log_path.read_text().splitlines()]
        # By the last entry, 7 items processed → 6 added before it → window=3 means
        # 3 in context, 3 already dropped (entry 7 itself isn't yet in ctx at log time)
        last = entries[-1]
        self.assertGreater(last["context"]["dropped_so_far"], 0)
        self.assertEqual(last["context"]["history_size"], 3)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class CliTests(unittest.TestCase):
    def test_cli_dry_run_does_not_import_or_use_anthropic(self):
        """`--dry-run` must work even if anthropic is unimportable."""
        argv = ["simulate_decisions.py", "--dry-run", "--fixture", "1", "--limit", "2"]
        with patch.object(sys, "argv", argv):
            # Should not raise even if we mock anthropic away — dry-run never imports it
            with patch.dict(sys.modules, {"anthropic": None}):
                sd.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
