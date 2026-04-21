"""Replay source: loads signals from any stored data directory."""

from __future__ import annotations

from pathlib import Path

from proactive_eval.types import SignalItem
from proactive_eval.sources import SignalSource


class ReplaySource(SignalSource):
    """Generic replay source — delegates to EigenFlux or Search replay
    based on what's available on disk."""

    source_name = "replay"

    def supports_replay(self) -> bool:
        return True

    def collect(self, fixture_id: str, **kwargs) -> list[SignalItem]:
        raise NotImplementedError("ReplaySource only supports replay(), not collect()")

    def replay(self, fixture_id: str, data_dir: Path | None = None) -> list[SignalItem]:
        from proactive_eval.sources.eigenflux import EigenFluxSource
        from proactive_eval.sources.search import SearchSource

        items = []
        items.extend(EigenFluxSource().replay(fixture_id, data_dir))
        items.extend(SearchSource().replay(fixture_id, data_dir))
        items.sort(key=lambda x: x.received_at)
        return items
