"""Replay source: calls the EigenFlux Replay API to test ranking configs.

The Replay API (eigenflux/replay/) is an offline ranking sandbox that
simulates the full sort pipeline with arbitrary parameters. This source
calls that API for each agent, collects the ranked items, and converts
them to SignalItems for LLM evaluation.

Usage:
    python -m proactive_eval simulate --source replay \
        --agent-id 290703888640638976 \
        --alpha 0.8 --beta 0.5 --gamma 0.3

    python -m proactive_eval simulate --config config/replay_example.json
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from proactive_eval.types import SignalItem
from proactive_eval.sources import SignalSource

_CST = timezone(timedelta(hours=8))

DEFAULT_REPLAY_URL = "http://localhost:8092"
DEFAULT_REPLAY_LIMIT = None  # None = no limit; fetch all available items


@dataclass
class ReplayConfig:
    """Configuration for the replay API call.

    Maps directly to the eigenflux replay API request fields:
      - ranker_params: alpha, beta, gamma, delta, min_relevance_score,
        urgency_boost, urgency_window, exploration_slots, draft_dampening,
        freshness (per-type overrides)
      - recall_params: keyword_recall_size, enable_knn_recall,
        knn_recall_k, knn_recall_candidates
    """

    url: str = DEFAULT_REPLAY_URL
    simulated_at: str | None = None
    use_feed_history: bool = False
    limit: int | None = DEFAULT_REPLAY_LIMIT
    ranker_params: dict[str, Any] = field(default_factory=dict)
    recall_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for experiment config storage."""
        return {
            "url": self.url,
            "simulated_at": self.simulated_at,
            "use_feed_history": self.use_feed_history,
            "limit": self.limit,
            "ranker_params": self.ranker_params,
            "recall_params": self.recall_params,
        }


class ReplaySource(SignalSource):
    """Calls the EigenFlux Replay API to evaluate ranking configurations.

    Unlike eigenflux/search sources which collect→disk→replay, this source
    calls the replay API live during simulation. Each call returns ranked
    items for a specific agent with specific ranking params.
    """

    source_name = "replay"

    def __init__(self, config: ReplayConfig | None = None):
        self._config = config or ReplayConfig()
        self._last_response: dict | None = None

    def supports_replay(self) -> bool:
        return False

    def collect(self, fixture_id: str, *, agent_id: int | None = None,
                **kwargs) -> list[SignalItem]:
        """Call the replay API for an agent and return ranked items."""
        if not agent_id:
            raise ValueError(
                "agent_id is required for replay source. "
                "Set it in fixture eigenflux_profile.agent_id or pass --agent-id."
            )

        body: dict[str, Any] = {
            "agent_id": agent_id,
            "use_feed_history": self._config.use_feed_history,
        }
        # Always send limit: None means "no client-side limit" → ask for all
        body["limit"] = self._config.limit if self._config.limit is not None else 10000
        if self._config.simulated_at:
            body["simulated_at"] = self._config.simulated_at
        if self._config.ranker_params:
            body["ranker_params"] = self._config.ranker_params
        if self._config.recall_params:
            body["recall_params"] = self._config.recall_params

        resp = self._post("/api/v1/replay/sort", body)
        self._last_response = resp

        items: list[SignalItem] = []
        for ri in resp.get("ranked_items", []):
            items.append(self._to_signal(ri, agent_id))
        for ri in resp.get("exploration_items", []):
            items.append(self._to_signal(ri, agent_id, exploration=True))

        return items

    def replay(self, fixture_id: str, data_dir=None) -> list[SignalItem]:
        raise NotImplementedError(
            "ReplaySource uses collect(), not replay(). "
            "The simulate command handles this automatically."
        )

    def get_last_stats(self) -> dict:
        """Return stats from the most recent API call."""
        if self._last_response:
            return self._last_response.get("stats", {})
        return {}

    def get_last_config(self) -> dict:
        """Return the config the replay API actually used."""
        if self._last_response:
            return self._last_response.get("config_used", {})
        return {}

    def get_last_profile(self) -> dict:
        """Return the agent profile the replay API resolved."""
        if self._last_response:
            return self._last_response.get("agent_profile", {})
        return {}

    def health_check(self) -> bool:
        """Check if the replay API is reachable."""
        try:
            url = self._config.url.rstrip("/") + "/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # HTTP client
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        url = self._config.url.rstrip("/") + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise ConnectionError(
                f"Replay API returned {e.code}: {error_body}\n"
                f"URL: {url}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to replay API at {url}: {e}\n"
                f"Make sure the eigenflux replay service is running:\n"
                f"  cd repos/eigenflux && bash replay/scripts/start.sh"
            ) from e

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _to_signal(self, replay_item: dict, agent_id: int,
                   exploration: bool = False) -> SignalItem:
        """Convert a replay API response item to a SignalItem."""
        item_data = replay_item.get("item", {})
        scores = replay_item.get("scores", {})

        # Build content string for the LLM evaluator
        parts = []
        if item_data.get("summary"):
            parts.append(f"Summary: {item_data['summary']}")
        if item_data.get("content"):
            parts.append(f"Content: {item_data['content']}")
        content = "\n".join(parts) if parts else "(no content)"

        # Use item's updated_at as received_at
        received_at = datetime.now(_CST)
        if item_data.get("updated_at"):
            try:
                received_at = datetime.fromisoformat(item_data["updated_at"])
                if received_at.tzinfo is None:
                    received_at = received_at.replace(tzinfo=_CST)
            except ValueError:
                pass

        return SignalItem(
            item_id=str(replay_item["item_id"]),
            source="replay",
            received_at=received_at,
            content=content,
            url=None,
            metadata={
                "agent_id": agent_id,
                "position": replay_item.get("position", 0),
                "score": replay_item.get("score", 0),
                "scores": scores,
                "broadcast_type": item_data.get("broadcast_type", ""),
                "keywords": item_data.get("keywords", []),
                "domains": item_data.get("domains", []),
                "quality_score": item_data.get("quality_score", 0),
                "group_id": item_data.get("group_id", 0),
                "exploration": exploration,
            },
            raw_item=replay_item,
        )
