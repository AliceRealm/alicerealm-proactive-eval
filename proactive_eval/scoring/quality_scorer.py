"""Quality scorer: computes existing metrics from extracted decision events."""

from __future__ import annotations

import math

from proactive_eval.types import Trajectory, Event
from proactive_eval.scoring import Scorer


class QualityScorer(Scorer):
    """Computes the existing scoring metrics:
    - send_rate + CI95
    - action_rate + CI95
    - quality_per_item (skip=0, send=1, action=3)
    - daily_yield
    """

    def __init__(self, targets: dict | None = None, days: float = 1.0):
        self.targets = targets or {
            "daily_sends": 15,
            "send_rate": 0.70,
            "action_rate": 0.15,
        }
        self.days = days

    def score(self, trajectory: Trajectory, events: list[Event]) -> dict:
        n = len(events)
        if n == 0:
            return {"scores": {}, "stats": {}, "details": {}}

        # Count actions
        skip_count = sum(1 for e in events if e.type == "skip")
        send_count = sum(1 for e in events if e.type == "send")
        action_count = sum(1 for e in events if e.type == "send_with_action")
        total_send = send_count + action_count

        # Rates with CI
        send_rate, send_ci_low, send_ci_high = wilson_ci(total_send, n)
        action_rate, action_ci_low, action_ci_high = wilson_ci(action_count, n)

        # Quality score
        quality_total = (send_count * 1) + (action_count * 3)
        quality_per_item = quality_total / n if n else 0

        # Daily yield
        population_size = n  # may be overridden by caller
        daily_yield = population_size / max(self.days, 0.01)

        stats = {
            "items_processed": n,
            "decisions": {
                "skip": skip_count,
                "send": send_count,
                "send_with_action": action_count,
            },
            "send_rate": round(send_rate, 4),
            "send_rate_ci95": [round(send_ci_low, 4), round(send_ci_high, 4)],
            "action_rate": round(action_rate, 4),
            "action_rate_ci95": [round(action_ci_low, 4), round(action_ci_high, 4)],
            "quality_total": quality_total,
            "quality_per_item": round(quality_per_item, 4),
            "daily_yield": round(daily_yield, 1),
            "population_size": population_size,
            "sample_size": n,
            "targets": dict(self.targets),
            "validation_errors": sum(
                1 for e in events
                if isinstance(e.content, dict)
                and not (e.content.get("validation") or {}).get("ok", True)
            ),
            "api_errors": sum(
                1 for e in events
                if isinstance(e.content, dict)
                and (e.content.get("api") or {}).get("error")
            ),
        }

        scores = {
            "send_rate": send_rate,
            "action_rate": action_rate,
            "quality_per_item": quality_per_item,
            "daily_yield": daily_yield,
        }

        return {"scores": scores, "stats": stats, "details": {}}


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
