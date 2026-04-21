"""Extracts decision events from per-item harness output."""

from __future__ import annotations

import json

from proactive_eval.types import Trajectory, Event
from proactive_eval.scoring import Extractor


class DecisionExtractor(Extractor):
    """Extracts send/skip/action decisions from PerItemHarness output.

    The raw_output of each tick contains one JSON object per line,
    each representing a decision log entry from the per-item harness.
    """

    def extract(self, trajectory: Trajectory) -> list[Event]:
        events = []
        for tick in trajectory.ticks:
            if not tick.raw_output:
                continue
            for line in tick.raw_output.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                decision = entry.get("decision", {})
                action = decision.get("action", "skip")

                event = Event(
                    type=action,
                    content={
                        "decision_id": entry.get("decision_id"),
                        "rationale": decision.get("rationale", ""),
                        "message": decision.get("message"),
                        "recommended_action": decision.get("recommended_action"),
                        "source": entry.get("source", ""),
                        "pull_timestamp": entry.get("pull_timestamp", ""),
                        "content_preview": entry.get("content_preview", ""),
                        "validation": entry.get("validation", {}),
                        "api": entry.get("api", {}),
                        "dry_run": entry.get("dry_run", False),
                    },
                    item_id=entry.get("source_metadata", {}).get("item_id")
                            or entry.get("decision_id"),
                    tick_index=tick.tick_index,
                )
                events.append(event)
                # Also populate the tick's events list
                tick.events.append(event)
        return events
