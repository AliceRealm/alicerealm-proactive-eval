"""EigenFlux signal source: online collection + file replay."""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from proactive_eval.types import SignalItem
from proactive_eval.sources import SignalSource
from proactive_eval.config import SCRIPT_DIR, rel_path

DATA_DIR = SCRIPT_DIR / "data" / "eigenflux"

API_BASE = "https://www.eigenflux.ai/api/v1"
SKILL_VERSION = "0.0.7"
FEED_LIMIT = 50


class EigenFluxSource(SignalSource):
    source_name = "eigenflux"

    def supports_replay(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Online collection
    # ------------------------------------------------------------------

    def collect(self, fixture_id: str, **kwargs) -> list[SignalItem]:
        """Pull feed from EigenFlux API and return normalized SignalItems.
        Also saves raw JSON to data/eigenflux/<fixture_id>/."""
        dry_run = kwargs.get("dry_run", False)
        token = load_token(fixture_id)
        if not token:
            print(f"    {fixture_id}: no credentials, skipped")
            return []
        result = collect_one_fixture(fixture_id, token, dry_run=dry_run)
        if not dry_run:
            save_result(fixture_id, result)
        if result.get("feed", {}).get("code") == 401:
            print(f"    WARNING: {fixture_id} got 401. "
                  f"Re-run `python -m proactive_eval setup --source eigenflux` to refresh.")
        return _normalize_result(result, fixture_id)

    # ------------------------------------------------------------------
    # Replay from disk
    # ------------------------------------------------------------------

    def replay(self, fixture_id: str, data_dir: Path | None = None) -> list[SignalItem]:
        """Load all previously collected EigenFlux feeds from disk."""
        d = (data_dir or DATA_DIR) / fixture_id
        if not d.exists():
            return []
        items = []
        for p in sorted(d.glob("*.json")):
            items.extend(_normalize_file(p, fixture_id))
        return items

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, fixture: dict) -> None:
        """Interactive login for one fixture."""
        fid = fixture["fixture_id"]
        profile = fixture.get("eigenflux_profile", {})
        agent_name = profile.get("agent_name", fid)

        existing_token = load_token(fid)
        if existing_token:
            print(f"  Already has credentials. Re-login? [y/N] ", end="")
            if input().strip().lower() != "y":
                print("  Skipped.")
                return

        email = input(f"  Email for {agent_name}: ").strip()
        if not email:
            print("  Skipped.")
            return

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
            print(f"  Login failed: {resp}")
            return

        save_token(fid, email, data["access_token"])
        print(f"  Agent ID: {data['agent_id']}")

        if profile:
            push_profile(data["access_token"], profile)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_request(method: str, endpoint: str, token: str | None = None,
                params: dict | None = None, body: dict | None = None) -> dict:
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
# Credentials — CSV-based token store
# ---------------------------------------------------------------------------

def _csv_path() -> Path:
    """Resolve the bot-token CSV path from env or default."""
    rel = os.environ.get("EIGENFLUX_BOT_TOKEN_CSV", "eigenflux_bot_token.csv")
    p = Path(rel)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p


# Module-level cache, populated on first use.
_TOKEN_MAP: dict[str, str] | None = None


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


def load_token(fixture_id: str) -> str | None:
    """Load access token for a fixture from the CSV."""
    global _TOKEN_MAP
    if _TOKEN_MAP is None:
        _TOKEN_MAP = _load_token_map()
    return _TOKEN_MAP.get(fixture_id)


def save_token(fixture_id: str, email: str, token: str):
    """Insert or update a row in the bot-token CSV."""
    global _TOKEN_MAP
    path = _csv_path()
    rows: list[dict] = []
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    updated = False
    for row in rows:
        if row.get("used_by", "").strip() == fixture_id:
            row["email"] = email
            row["token"] = token
            updated = True
            break
    if not updated:
        rows.append({"email": email, "token": token, "used_by": fixture_id})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["email", "token", "used_by"])
        writer.writeheader()
        writer.writerows(rows)
    # Invalidate cache so subsequent load_token calls see the new token.
    _TOKEN_MAP = None
    print(f"  Saved token to {path}")


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

def push_profile(token: str, profile: dict):
    resp = api_request("PUT", "/agents/profile", token=token, body=profile)
    if resp.get("code") == 0:
        print(f"  Profile set: {profile.get('agent_name')}")
    else:
        print(f"  Profile update failed: {resp}")


def sync_profiles(fixtures: list[dict]) -> dict:
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


# ---------------------------------------------------------------------------
# Feed collection
# ---------------------------------------------------------------------------

def pull_feed(token: str) -> dict:
    return api_request("GET", "/items/feed", token=token,
                       params={"limit": str(FEED_LIMIT), "action": "refresh"})


def collect_one_fixture(fixture_id: str, token: str, dry_run: bool = False) -> dict:
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
    fixture_dir = DATA_DIR / fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = fixture_dir / f"{ts}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"    Saved -> {filename}")


# ---------------------------------------------------------------------------
# Normalization (raw JSON -> SignalItem)
# ---------------------------------------------------------------------------

def _normalize_result(result: dict, fixture_id: str) -> list[SignalItem]:
    """Normalize a single collection result dict (in-memory) to SignalItems."""
    pull_ts = result.get("timestamp_utc", "")
    items = []
    for entry in result.get("detailed_items", []):
        item = entry.get("item") if isinstance(entry, dict) and "item" in entry else entry
        if not isinstance(item, dict) or "error" in item:
            continue
        items.append(_item_dict_to_signal(item, fixture_id, pull_ts, "in-memory"))
    return items


def _normalize_file(path: Path, fixture_id: str) -> list[SignalItem]:
    """Normalize a stored JSON file to SignalItems."""
    data = json.loads(path.read_text())
    pull_ts = data.get("timestamp_utc", "")
    items = []
    for entry in data.get("detailed_items", []):
        item = entry.get("item") if isinstance(entry, dict) and "item" in entry else entry
        if not isinstance(item, dict) or "error" in item:
            continue
        items.append(_item_dict_to_signal(item, fixture_id, pull_ts, rel_path(path)))
    return items


def _item_dict_to_signal(item: dict, fixture_id: str,
                          pull_ts: str, pull_file: str) -> SignalItem:
    content_parts = []
    if item.get("summary"):
        content_parts.append(f"Summary: {item['summary']}")
    if item.get("content"):
        content_parts.append(f"Content: {item['content']}")
    if item.get("url"):
        content_parts.append(f"URL: {item['url']}")

    try:
        received_at = datetime.fromisoformat(pull_ts)
        if received_at.tzinfo is None:
            received_at = received_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        received_at = datetime.now(timezone.utc)

    return SignalItem(
        item_id=item.get("item_id", ""),
        source="eigenflux",
        received_at=received_at,
        content="\n".join(content_parts),
        url=item.get("url"),
        metadata={
            "fixture_id": fixture_id,
            "pull_file": pull_file,
            "broadcast_type": item.get("broadcast_type"),
            "domains": item.get("domains", []),
            "geo": item.get("geo"),
            "keywords": item.get("keywords", []),
        },
        raw_item=item,
    )
