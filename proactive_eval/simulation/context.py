"""Rolling context for dedup awareness across decisions."""

from __future__ import annotations


class RollingContext:
    """Sliding window of recent decisions.

    When items fall out of the window, keeps a tally summary.
    Satisfies the test expectations in test_simulate_decisions.py.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.history: list[dict] = []  # recent (item_summary, decision) pairs
        self.dropped_count: int = 0
        self.dropped_summary: dict[str, int] = {
            "skip": 0, "send": 0, "send_with_action": 0
        }

    def add(self, item: dict, decision: dict) -> None:
        """Add an item-decision pair to the rolling window."""
        self.history.append({"item": item, "decision": decision})
        while len(self.history) > self.window_size:
            dropped = self.history.pop(0)
            self.dropped_count += 1
            action = dropped["decision"].get("action", "skip")
            self.dropped_summary[action] = self.dropped_summary.get(action, 0) + 1

    def trim_to_token_budget(self, system_prompt: str, new_item: dict,
                             budget: int = 8000) -> int:
        """Trim history to fit within approximate token budget.
        Returns number of items trimmed."""
        # Approximate: 1 token ≈ 4 chars
        base_chars = len(system_prompt) + len(str(new_item))
        base_tokens = base_chars // 4
        available = budget - base_tokens
        if available <= 0:
            trimmed = len(self.history)
            for entry in self.history:
                self.dropped_count += 1
                action = entry["decision"].get("action", "skip")
                self.dropped_summary[action] = self.dropped_summary.get(action, 0) + 1
            self.history.clear()
            return trimmed

        trimmed = 0
        while self.history:
            history_chars = sum(len(str(e)) for e in self.history)
            history_tokens = history_chars // 4
            if history_tokens <= available:
                break
            dropped = self.history.pop(0)
            self.dropped_count += 1
            action = dropped["decision"].get("action", "skip")
            self.dropped_summary[action] = self.dropped_summary.get(action, 0) + 1
            trimmed += 1
        return trimmed

    def format_block(self) -> str:
        """Format the rolling context for inclusion in prompts."""
        if not self.history and self.dropped_count == 0:
            return ""

        parts = []
        if self.dropped_count > 0:
            parts.append(
                f"CONTEXT NOTE: {self.dropped_count} earlier items have been processed "
                f"and dropped from context. Summary: {self.dropped_summary}"
            )

        if self.history:
            parts.append(f"Recent decisions ({len(self.history)} items):")
            for entry in self.history:
                item = entry["item"]
                decision = entry["decision"]
                preview = (item.get("content", "") or "")[:100]
                action = decision.get("action", "skip")
                parts.append(f"  - [{action}] {preview}")

        return "\n".join(parts)
