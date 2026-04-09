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

import json
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
CREDENTIALS_DIR = SCRIPT_DIR / "eigenflux" / "credentials"

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
# Credentials (per fixture)
# ---------------------------------------------------------------------------

def credentials_path(fixture_id: str) -> Path:
    return CREDENTIALS_DIR / f"{fixture_id}.json"


def load_token(fixture_id: str) -> str | None:
    """Load access token for a fixture. Returns None if not found."""
    path = credentials_path(fixture_id)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    token = data.get("access_token", "")
    if not token or token.startswith("<"):
        return None
    return token


def save_credentials(fixture_id: str, creds: dict):
    """Save credentials for a fixture."""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    path = credentials_path(fixture_id)
    with open(path, "w") as f:
        json.dump(creds, f, indent=2)
    print(f"  Saved -> {path}")


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

        creds = {
            "agent_id": data["agent_id"],
            "access_token": data["access_token"],
            "expires_at": data.get("expires_at"),
            "email": email,
        }
        save_credentials(fid, creds)
        print(f"  Agent ID: {creds['agent_id']}")

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


def do_profile_all(fixtures: list[dict]):
    """Push profiles for all fixtures that have credentials."""
    print("=== Pushing Profiles ===\n")
    for fix in fixtures:
        fid = fix["fixture_id"]
        profile = fix.get("eigenflux_profile")
        if not profile:
            print(f"  {fid}: no eigenflux_profile in fixture, skipped")
            continue
        token = load_token(fid)
        if not token:
            print(f"  {fid}: no credentials, skipped")
            continue
        push_profile(token, profile)
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
