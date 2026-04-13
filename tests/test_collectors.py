#!/usr/bin/env python3
"""
Tests for collect_eigenflux.py and collect_searches.py.

These tests do NOT make real network calls. They mock urllib / DDGS so the
polling logic, save paths, retry behavior, and daemon-loop semantics can be
verified offline. Run as:

    cd proactive-eval
    python3 -m unittest tests/test_collectors.py -v

or simply:

    python3 tests/test_collectors.py

Goal: keep `collect_eigenflux.py` and `collect_searches.py` always
runnable (as long-running daemons against the live EigenFlux service)
without surprises.
"""

import io
import json
import os
import sys
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make the proactive-eval/ directory importable regardless of cwd.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import collect_eigenflux as e  # noqa: E402
import collect_searches as s   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_urlopen_response(payload: dict):
    """Build a context-manager-compatible fake urlopen() return value."""
    m = MagicMock()
    m.read.return_value = json.dumps(payload).encode()
    m.__enter__ = lambda self: m
    m.__exit__ = lambda self, *a: None
    return m


# ---------------------------------------------------------------------------
# Structural tests — fixtures, queries, credentials, alignment
# ---------------------------------------------------------------------------

class StructuralTests(unittest.TestCase):
    def test_fixtures_load(self):
        fixtures = e.load_fixtures()
        self.assertEqual(len(fixtures), 6, "expected 6 fixtures")
        for f in fixtures:
            self.assertIn("fixture_id", f)
            self.assertIn("eigenflux_profile", f,
                          f"{f.get('fixture_id')} missing eigenflux_profile")

    def test_queries_load(self):
        pairs = s.load_queries()
        self.assertEqual(len(pairs), 6, "expected 6 search pairs")
        for p in pairs:
            self.assertIn("fixture", p)
            self.assertIn("queries", p)
            self.assertGreater(len(p["queries"]), 0)

    def test_fixture_pair_alignment(self):
        fixture_ids = {f["fixture_id"] for f in e.load_fixtures()}
        pair_ids = {p["fixture"] for p in s.load_queries()}
        self.assertEqual(fixture_ids, pair_ids,
                         "fixture IDs and search-pair fixture IDs must match")

    def test_credentials_present_for_all_fixtures(self):
        """If a fixture has no credentials we want to know early."""
        missing = []
        for f in e.load_fixtures():
            fid = f["fixture_id"]
            if not e.load_token(fid):
                missing.append(fid)
        self.assertFalse(
            missing,
            f"missing eigenflux credentials for: {missing}. "
            f"Run: python3 collect_eigenflux.py --setup --fixture N",
        )


# ---------------------------------------------------------------------------
# EigenFlux API layer — every error path, no real network
# ---------------------------------------------------------------------------

class EigenFluxApiTests(unittest.TestCase):
    def test_api_request_success(self):
        resp = _fake_urlopen_response({"code": 0, "data": {"items": []}})
        with patch("urllib.request.urlopen", return_value=resp):
            r = e.api_request("GET", "/items/feed", token="fake")
        self.assertEqual(r["code"], 0)

    def test_api_request_http_401(self):
        fp = io.BytesIO(b'{"msg":"unauthorized"}')
        err = urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, fp)
        with patch("urllib.request.urlopen", side_effect=err):
            r = e.api_request("GET", "/items/feed", token="bad")
        self.assertEqual(r["code"], 401)

    def test_api_request_network_error(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionError("boom")):
            r = e.api_request("GET", "/items/feed", token="fake")
        self.assertEqual(r["code"], -1)
        self.assertIn("boom", r["msg"])

    def test_pull_feed_uses_feed_endpoint(self):
        resp = _fake_urlopen_response({"code": 0, "data": {"items": []}})
        seen = {}

        def capture(req, timeout=None):
            seen["url"] = req.full_url
            seen["auth"] = req.headers.get("Authorization")
            return resp

        with patch("urllib.request.urlopen", side_effect=capture):
            r = e.pull_feed("tok123")
        self.assertEqual(r["code"], 0)
        self.assertIn("/items/feed", seen["url"])
        self.assertIn("limit=", seen["url"])
        self.assertEqual(seen["auth"], "Bearer tok123")

    def test_collect_one_fixture_happy_path(self):
        """Feed with 2 items -> details fetched for both."""
        def fake_urlopen(req, timeout=None):
            url = req.full_url
            if "/items/feed" in url:
                payload = {
                    "code": 0,
                    "data": {
                        "impression_id": "imp1",
                        "items": [{"item_id": "a1"}, {"item_id": "a2"}],
                        "notifications": [],
                    },
                }
            else:
                payload = {"code": 0, "data": {"item_id": url.rsplit("/", 1)[-1], "summary": "x"}}
            return _fake_urlopen_response(payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch.object(time, "sleep"):
            res = e.collect_one_fixture("test_fix", "fake_token", dry_run=False)

        self.assertEqual(res["feed"]["code"], 0)
        self.assertEqual(res["feed"]["item_count"], 2)
        self.assertEqual(len(res["detailed_items"]), 2)
        self.assertEqual(res["fixture_id"], "test_fix")
        self.assertEqual(res["source"], "eigenflux")

    def test_collect_one_fixture_dry_run_skips_network(self):
        with patch("urllib.request.urlopen",
                   side_effect=AssertionError("network must not be called in dry-run")):
            res = e.collect_one_fixture("test_fix", "fake_token", dry_run=True)
        self.assertTrue(res["dry_run"])
        self.assertNotIn("feed", res)

    def test_collect_one_fixture_handles_401_gracefully(self):
        """A 401 from feed endpoint must be returned, not raised."""
        fp = io.BytesIO(b'{"msg":"unauthorized"}')
        err = urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, fp)
        with patch("urllib.request.urlopen", side_effect=err):
            res = e.collect_one_fixture("test_fix", "expired", dry_run=False)
        self.assertEqual(res["feed"]["code"], 401)
        self.assertEqual(res["feed"]["item_count"], 0)


# ---------------------------------------------------------------------------
# Searches layer — DDGS wrapper, retries, dry-run
# ---------------------------------------------------------------------------

class SearchesTests(unittest.TestCase):
    def test_search_success(self):
        fake = MagicMock()
        fake.text.return_value = [{"title": "t", "href": "u", "body": "b"}]
        with patch("collect_searches.DDGS", return_value=fake):
            r = s.search("anything", max_results=5)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["title"], "t")

    def test_search_retries_then_returns_error(self):
        """Failed search should retry `retries` times then yield an error item."""
        fake = MagicMock()
        fake.text.side_effect = RuntimeError("rate limit")
        with patch("collect_searches.DDGS", return_value=fake), \
             patch.object(time, "sleep"):
            r = s.search("anything", max_results=5, retries=2)
        self.assertEqual(len(r), 1)
        self.assertIn("error", r[0])
        # 1 initial + 2 retries
        self.assertEqual(fake.text.call_count, 3)

    def test_collect_one_pair_happy_path(self):
        fake = MagicMock()
        fake.text.return_value = [{"title": "t1", "href": "u1", "body": "b1"}]
        pairs = s.load_queries()
        with patch("collect_searches.DDGS", return_value=fake), \
             patch.object(time, "sleep"):
            cycle = s.collect_one_pair(pairs[0], dry_run=False)
        self.assertEqual(cycle["fixture_id"], pairs[0]["fixture"])
        self.assertEqual(cycle["query_count"], len(pairs[0]["queries"]))
        self.assertEqual(len(cycle["results"]), len(pairs[0]["queries"]))

    def test_collect_one_pair_dry_run_skips_network(self):
        pairs = s.load_queries()
        with patch("collect_searches.DDGS",
                   side_effect=AssertionError("DDGS must not be called in dry-run")):
            cycle = s.collect_one_pair(pairs[0], dry_run=True)
        self.assertTrue(all(r.get("dry_run") for r in cycle["results"]))


# ---------------------------------------------------------------------------
# save_result — verify both collectors land files in the right place
# ---------------------------------------------------------------------------

class SaveResultTests(unittest.TestCase):
    def test_eigenflux_save_result(self):
        fake = {
            "source": "eigenflux",
            "fixture_id": "careful_operator_v1",
            "timestamp_utc": "now",
            "feed": {"code": 0, "item_count": 0, "items": []},
        }
        e.save_result("careful_operator_v1", fake)
        target_dir = SCRIPT_DIR / "data" / "eigenflux" / "careful_operator_v1"
        files = sorted(target_dir.glob("*.json"))
        self.assertTrue(files, "no eigenflux file written")
        loaded = json.loads(files[-1].read_text())
        self.assertEqual(loaded["fixture_id"], "careful_operator_v1")
        os.remove(files[-1])

    def test_searches_save_result(self):
        fake = {
            "fixture_id": "careful_operator_v1",
            "scenario_id": "stock",
            "timestamp_utc": "now",
            "query_count": 0,
            "results": [],
        }
        path = s.save_result("careful_operator_v1", fake)
        self.assertTrue(Path(path).exists())
        loaded = json.loads(Path(path).read_text())
        self.assertEqual(loaded["fixture_id"], "careful_operator_v1")
        os.remove(path)


# ---------------------------------------------------------------------------
# Daemon-loop semantics — once / duration / forever, no real waiting
# ---------------------------------------------------------------------------

class ProfileSyncTests(unittest.TestCase):
    """Profiles must be pushed to EigenFlux before any feed is pulled."""

    def test_sync_profiles_pushes_for_every_fixture_with_creds(self):
        """All 6 real fixtures have profiles + creds -> should push all 6."""
        seen = []

        def capture_urlopen(req, timeout=None):
            seen.append((req.method or req.get_method(), req.full_url,
                         json.loads(req.data.decode()) if req.data else None))
            return _fake_urlopen_response({"code": 0, "data": {}})

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            summary = e.sync_profiles(e.load_fixtures())

        self.assertEqual(summary["pushed"], 6)
        self.assertEqual(summary["skipped_no_profile"], 0)
        self.assertEqual(summary["skipped_no_creds"], 0)
        # Each call should be PUT /agents/profile with the bio in the body
        for method, url, body in seen:
            self.assertEqual(method, "PUT")
            self.assertIn("/agents/profile", url)
            self.assertIn("agent_name", body)
            self.assertIn("bio", body)
        self.assertEqual(len(seen), 6)

    def test_sync_profiles_skips_fixtures_without_credentials(self):
        """A fixture with no token should be skipped, not crash."""
        fix = {
            "fixture_id": "ghost",
            "eigenflux_profile": {"agent_name": "g", "bio": "b"},
        }
        with patch("collect_eigenflux.load_token", return_value=None), \
             patch("urllib.request.urlopen",
                   side_effect=AssertionError("must not call API for credential-less fixture")):
            summary = e.sync_profiles([fix])
        self.assertEqual(summary["pushed"], 0)
        self.assertEqual(summary["skipped_no_creds"], 1)

    def test_sync_profiles_skips_fixtures_without_profile(self):
        fix = {"fixture_id": "no_profile_fix"}  # no eigenflux_profile key
        with patch("collect_eigenflux.load_token", return_value="tok"), \
             patch("urllib.request.urlopen",
                   side_effect=AssertionError("must not push when no profile")):
            summary = e.sync_profiles([fix])
        self.assertEqual(summary["pushed"], 0)
        self.assertEqual(summary["skipped_no_profile"], 1)

    def test_main_auto_syncs_profiles_at_startup(self):
        """`collect_eigenflux.py --once` (no --skip-profile-sync) must call sync_profiles before pulling."""
        argv = ["collect_eigenflux.py", "--once", "--fixture", "1"]
        sync_called = {"n": 0}

        def fake_sync(fixtures):
            sync_called["n"] += 1
            return {"pushed": 1, "skipped_no_profile": 0, "skipped_no_creds": 0}

        # collect_one_fixture should also be patched so we don't make a real API call
        with patch.object(sys, "argv", argv), \
             patch("collect_eigenflux.sync_profiles", side_effect=fake_sync) as mock_sync, \
             patch("collect_eigenflux.collect_one_fixture",
                   return_value={"feed": {"code": 0, "item_count": 0}}), \
             patch.object(time, "sleep"):
            e.main()
        self.assertEqual(mock_sync.call_count, 1,
                         "sync_profiles must be called exactly once at startup")

    def test_main_skip_profile_sync_flag_disables_auto_sync(self):
        argv = ["collect_eigenflux.py", "--once", "--fixture", "1",
                "--skip-profile-sync"]
        with patch.object(sys, "argv", argv), \
             patch("collect_eigenflux.sync_profiles",
                   side_effect=AssertionError("sync_profiles must not run when --skip-profile-sync is set")), \
             patch("collect_eigenflux.collect_one_fixture",
                   return_value={"feed": {"code": 0, "item_count": 0}}), \
             patch.object(time, "sleep"):
            e.main()  # must not raise

    def test_main_dry_run_does_not_sync(self):
        """Dry-run promises no real API calls — sync_profiles must be skipped too."""
        argv = ["collect_eigenflux.py", "--once", "--dry-run"]
        with patch.object(sys, "argv", argv), \
             patch("collect_eigenflux.sync_profiles",
                   side_effect=AssertionError("dry-run must not call sync_profiles")), \
             patch.object(time, "sleep"):
            e.main()


class DaemonLoopTests(unittest.TestCase):
    def test_eigenflux_once(self):
        argv = ["collect_eigenflux.py", "--once", "--dry-run"]
        with patch.object(sys, "argv", argv), patch.object(time, "sleep"):
            e.main()  # exits cleanly after 1 cycle

    def test_eigenflux_duration(self):
        # 0.5h / 5min = 6 cycles -> 5 sleeps between them
        argv = ["collect_eigenflux.py", "--duration", "0.5",
                "--interval", "5", "--dry-run"]
        with patch.object(sys, "argv", argv), \
             patch.object(time, "sleep") as mock_sleep:
            e.main()
        self.assertGreaterEqual(mock_sleep.call_count, 5)

    def test_eigenflux_forever_can_be_interrupted(self):
        """Forever mode loops until something stops it. Stop via KeyboardInterrupt."""
        argv = ["collect_eigenflux.py", "--dry-run", "--interval", "5"]
        call_count = {"n": 0}

        def stop_after_3(_):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                raise KeyboardInterrupt()

        with patch.object(sys, "argv", argv), \
             patch.object(time, "sleep", side_effect=stop_after_3):
            with self.assertRaises(KeyboardInterrupt):
                e.main()
        self.assertEqual(call_count["n"], 3)

    def test_searches_once(self):
        argv = ["collect_searches.py", "--once", "--dry-run"]
        with patch.object(sys, "argv", argv), patch.object(time, "sleep"):
            s.main()

    def test_searches_duration(self):
        argv = ["collect_searches.py", "--duration", "0.5",
                "--interval", "5", "--dry-run"]
        with patch.object(sys, "argv", argv), \
             patch.object(time, "sleep") as mock_sleep:
            s.main()
        self.assertGreaterEqual(mock_sleep.call_count, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
