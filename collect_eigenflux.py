#!/usr/bin/env python3
"""
EigenFlux feed collector for proactive-eval.

Each fixture has its own EigenFlux agent (profile + credentials).
Pulls all 6 feeds in parallel at regular intervals.

Usage:
    python3 collect_eigenflux.py                        # All fixtures, every 5min, forever
    python3 collect_eigenflux.py --fixture 1             # Only fixture 1
    python3 collect_eigenflux.py --once                  # One pull and exit
    python3 collect_eigenflux.py --dry-run               # Show what would happen
    python3 collect_eigenflux.py --setup                 # Interactive login for all fixtures
    python3 collect_eigenflux.py --setup --fixture 1     # Login for fixture 1 only
    python3 collect_eigenflux.py --profile               # Push profiles to EigenFlux
"""

import csv
import json
import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
DATA_DIR = SCRIPT_DIR / "data" / "eigenflux"

API_BASE = "https://www.eigenflux.ai/api/v1"
SKILL_VERSION = "0.0.7"
FEED_LIMIT = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def load_fixtures() -> list[dict]:
    """Load all fixture files, sorted by filename."""
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        data["_path"] = str(path)
        fixtures.append(data)
    return fixtures


# ---------------------------------------------------------------------------
# Credentials — read from CSV (one file for all bots)
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env from project root (no dependency on python-dotenv)."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _csv_path() -> Path:
    """Resolve the bot-token CSV path from env."""
    rel = os.environ.get("EIGENFLUX_BOT_TOKEN_CSV", "eigenflux_bot_token.csv")
    p = Path(rel)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p


def _load_token_map() -> dict[str, str]:
    """Return {fixture_id: access_token} from the CSV."""
    path = _csv_path()
    if not path.exists():
        print(f"WARNING: bot token CSV not found: {path}")
        return {}
    mapping: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            used_by = row.get("used_by", "").strip()
            token = row.get("token", "").strip()
            if used_by and token:
                mapping[used_by] = token
    return mapping


# Module-level cache, populated on first use.
_TOKEN_MAP: dict[str, str] | None = None


def load_token(fixture_id: str) -> str | None:
    """Load access token for a fixture from the CSV."""
    global _TOKEN_MAP
    if _TOKEN_MAP is None:
        _TOKEN_MAP = _load_token_map()
    return _TOKEN_MAP.get(fixture_id)


def _upsert_csv_row(email: str, token: str, used_by: str):
    """Insert or update a row in the bot-token CSV."""
    path = _csv_path()
    rows: list[dict] = []
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    updated = False
    for row in rows:
        if row.get("used_by", "").strip() == used_by:
            row["email"] = email
            row["token"] = token
            updated = True
            break
    if not updated:
        rows.append({"email": email, "token": token, "used_by": used_by})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["email", "token", "used_by"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved token to {path}")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_request(method: str, endpoint: str, token: str | None = None,
                params: dict | None = None, body: dict | None = None) -> dict:
    """Make an API request to EigenFlux."""
    url = f"{API_BASE}{endpoint}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-Skill-Ver", SKILL_VERSION)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        resp_body = e.read().decode() if e.fp else ""
        return {"code": e.code, "msg": f"HTTP {e.code}", "error": resp_body}
    except Exception as e:
        return {"code": -1, "msg": str(e), "error": str(e)}


# ---------------------------------------------------------------------------
# Setup: login per fixture
# ---------------------------------------------------------------------------

def do_setup(fixtures: list[dict]):
    """Interactive login + profile for each fixture."""
    print("=== EigenFlux Agent Setup ===\n")

    for i, fix in enumerate(fixtures):
        fid = fix["fixture_id"]
        profile = fix.get("eigenflux_profile", {})
        agent_name = profile.get("agent_name", fid)
        print(f"[{i+1}/{len(fixtures)}] {fid} ({agent_name})")

        existing_token = load_token(fid)
        if existing_token:
            print(f"  Already has credentials. Re-login? [y/N] ", end="")
            if input().strip().lower() != "y":
                print("  Skipped.\n")
                continue

        email = input(f"  Email for {agent_name}: ").strip()
        if not email:
            print("  Skipped.\n")
            continue

        resp = api_request("POST", "/auth/login", body={"login_method": "email", "email": email})
        data = resp.get("data", {})

        if data.get("verification_required"):
            challenge_id = data["challenge_id"]
            print(f"  OTP sent to {email} (expires in {data.get('expires_in_sec', 600)}s)")
            code = input("  Enter OTP code: ").strip()
            resp = api_request("POST", "/auth/login/verify", body={
                "login_method": "email",
                "challenge_id": challenge_id,
                "code": code,
            })
            data = resp.get("data", {})

        if not data.get("access_token"):
            print(f"  Login failed: {resp}\n")
            continue

        _upsert_csv_row(email, data["access_token"], fid)
        # Invalidate cache so subsequent load_token calls see the new token.
        global _TOKEN_MAP
        _TOKEN_MAP = None
        print(f"  Agent ID: {data.get('agent_id')}")

        # Set profile
        if profile:
            push_profile(data["access_token"], profile)
        print()

    print("=== Setup complete ===")


# ---------------------------------------------------------------------------
# Profile push
# ---------------------------------------------------------------------------

def push_profile(token: str, profile: dict):
    """Push profile to EigenFlux."""
    resp = api_request("PUT", "/agents/profile", token=token, body=profile)
    if resp.get("code") == 0:
        print(f"  Profile set: {profile.get('agent_name')}")
    else:
        print(f"  Profile update failed: {resp}")


def sync_profiles(fixtures: list[dict]) -> dict:
    """Push every fixture's eigenflux_profile to the live agent.

    Returns a small summary dict {pushed, skipped_no_profile, skipped_no_creds}.
    Used both by `--profile` (one-shot) and by collection startup (auto-sync),
    so the line agent's profile is always in sync with the fixture file before
    any feed is pulled. Without this, editing a fixture's bio and re-running
    collection would silently pull feed against a stale profile.
    """
    summary = {"pushed": 0, "skipped_no_profile": 0, "skipped_no_creds": 0}
    for fix in fixtures:
        fid = fix["fixture_id"]
        profile = fix.get("eigenflux_profile")
        if not profile:
            print(f"  {fid}: no eigenflux_profile in fixture, skipped")
            summary["skipped_no_profile"] += 1
            continue
        token = load_token(fid)
        if not token:
            print(f"  {fid}: no credentials, skipped")
            summary["skipped_no_creds"] += 1
            continue
        push_profile(token, profile)
        summary["pushed"] += 1
    return summary


def do_profile_all(fixtures: list[dict]):
    """One-shot CLI entry point: `python3 collect_eigenflux.py --profile`."""
    print("=== Pushing Profiles ===\n")
    sync_profiles(fixtures)
    print("\n=== Done ===")


# ---------------------------------------------------------------------------
# Feed collection
# ---------------------------------------------------------------------------

def pull_feed(token: str) -> dict:
    """Pull feed items from EigenFlux."""
    return api_request("GET", "/items/feed", token=token,
                       params={"limit": str(FEED_LIMIT), "action": "refresh"})


def collect_one_fixture(fixture_id: str, token: str, dry_run: bool = False) -> dict:
    """Pull feed for one fixture."""
    now = datetime.now(timezone.utc)
    result = {
        "source": "eigenflux",
        "fixture_id": fixture_id,
        "timestamp_utc": now.isoformat(),
        "timestamp_local": datetime.now().isoformat(),
        "skill_version": SKILL_VERSION,
    }

    if dry_run:
        print(f"    [DRY RUN] Would pull feed for {fixture_id}")
        result["dry_run"] = True
        return result

    feed_resp = pull_feed(token)
    code = feed_resp.get("code", -1)
    data = feed_resp.get("data", {})
    items = data.get("items", [])
    notifications = data.get("notifications", [])

    result["feed"] = {
        "code": code,
        "impression_id": data.get("impression_id", ""),
        "item_count": len(items),
        "items": items,
        "notification_count": len(notifications),
        "notifications": notifications,
    }

    # Fetch item details
    if items:
        detailed = []
        for item in items:
            item_id = item.get("item_id")
            if item_id:
                detail = api_request("GET", f"/items/{item_id}", token=token)
                if detail.get("code") == 0:
                    detailed.append(detail.get("data", {}))
                else:
                    detailed.append({"item_id": item_id, "error": detail.get("msg")})
                time.sleep(0.5)
        result["detailed_items"] = detailed

    print(f"    {fixture_id}: {len(items)} items, {len(notifications)} notifications (code: {code})")
    return result


def save_result(fixture_id: str, result: dict):
    """Save one fixture's cycle result."""
    fixture_dir = DATA_DIR / fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = fixture_dir / f"{ts}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    Saved -> {filename}")


def run_one_cycle(fixtures: list[dict], dry_run: bool = False):
    """Run one collection cycle across all fixtures."""
    for fix in fixtures:
        fid = fix["fixture_id"]
        token = load_token(fid)
        if not token:
            print(f"    {fid}: no credentials, skipped")
            continue
        result = collect_one_fixture(fid, token, dry_run=dry_run)
        if not dry_run:
            save_result(fid, result)
        if result.get("feed", {}).get("code") == 401:
            print(f"    WARNING: {fid} got 401. Re-run --setup --fixture N to refresh.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _load_env()
    parser = argparse.ArgumentParser(description="EigenFlux feed collector (per-fixture)")
    parser.add_argument("--setup", action="store_true",
                        help="Interactive login and profile setup")
    parser.add_argument("--profile", action="store_true",
                        help="Push profiles from fixtures to EigenFlux")
    parser.add_argument("--fixture", type=int, default=None,
                        help="Only operate on the Nth fixture (1-indexed)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutes between pulls (default: 5)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Hours to run (default: 0 = forever)")
    parser.add_argument("--once", action="store_true",
                        help="Pull once and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without API calls")
    parser.add_argument("--skip-profile-sync", action="store_true",
                        help="Do not push fixture profiles to EigenFlux at startup "
                             "(by default profiles are auto-synced before the first cycle)")
    args = parser.parse_args()

    all_fixtures = load_fixtures()

    if args.fixture is not None:
        idx = args.fixture - 1
        if idx < 0 or idx >= len(all_fixtures):
            print(f"Error: --fixture must be 1-{len(all_fixtures)}")
            sys.exit(1)
        fixtures = [all_fixtures[idx]]
    else:
        fixtures = all_fixtures

    if args.setup:
        do_setup(fixtures)
        return
    if args.profile:
        do_profile_all(fixtures)
        return

    # Collection mode
    if args.once:
        run_forever = False
        total_cycles = 1
    elif args.duration > 0:
        run_forever = False
        total_cycles = int(args.duration * 60 / args.interval)
    else:
        run_forever = True
        total_cycles = None

    fixture_names = [f["fixture_id"] for f in fixtures]
    mode = "forever" if run_forever else ("1 cycle" if args.once else f"{args.duration}h ({total_cycles} cycles)")

    print(f"=== EigenFlux Feed Collection ===")
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Fixtures: {len(fixtures)} ({', '.join(fixture_names)})")
    print(f"Interval: {args.interval}min | Mode: {mode}")
    print()

    # Auto-sync profiles before any pull, so the line agent's profile always
    # matches the fixture file. Skip in dry-run (no real API calls allowed)
    # and when the user explicitly opts out.
    if not args.dry_run and not args.skip_profile_sync:
        print("=== Syncing fixture profiles to EigenFlux ===")
        summary = sync_profiles(fixtures)
        print(f"  pushed={summary['pushed']} "
              f"skipped_no_profile={summary['skipped_no_profile']} "
              f"skipped_no_creds={summary['skipped_no_creds']}")
        print()

    cycle = 0
    while True:
        cycle += 1
        label = f"[Cycle {cycle}]" if run_forever else f"[Cycle {cycle}/{total_cycles}]"
        print(f"{label} @ {datetime.now().isoformat()}")

        run_one_cycle(fixtures, dry_run=args.dry_run)

        if total_cycles is not None and cycle >= total_cycles:
            break

        next_time = datetime.now().timestamp() + args.interval * 60
        next_str = datetime.fromtimestamp(next_time).strftime("%H:%M:%S")
        print(f"  Next pull at {next_str}")
        time.sleep(args.interval * 60)

    print(f"\n=== Done. {cycle} cycles completed. ===")


if __name__ == "__main__":
    main()
