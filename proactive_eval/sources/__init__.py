"""Signal sources: pluggable data collection layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from proactive_eval.types import SignalItem


class SignalSource(ABC):
    """Abstract base class for all signal sources."""

    source_name: str = ""

    @abstractmethod
    def collect(self, fixture_id: str, **kwargs) -> list[SignalItem]:
        """Pull signals online for one fixture. Returns normalized SignalItems."""
        ...

    @abstractmethod
    def supports_replay(self) -> bool:
        """Whether this source can replay historical data from disk."""
        ...

    def replay(self, fixture_id: str, data_dir: Path | None = None) -> list[SignalItem]:
        """Load previously collected signals from stored files."""
        raise NotImplementedError(f"{self.source_name} does not support replay")

    def setup(self, fixture: dict) -> None:
        """Optional interactive setup (e.g. EigenFlux login)."""
        pass


# Source registry
_SOURCES: dict[str, type[SignalSource]] = {}


def register_source(name: str, cls: type[SignalSource]):
    _SOURCES[name] = cls


def get_source(name: str) -> SignalSource:
    if name not in _SOURCES:
        raise ValueError(f"Unknown signal source: {name}. Available: {list(_SOURCES.keys())}")
    return _SOURCES[name]()


def available_sources() -> list[str]:
    return list(_SOURCES.keys())


# Auto-register built-in sources
from proactive_eval.sources.eigenflux import EigenFluxSource  # noqa: E402
from proactive_eval.sources.search import SearchSource  # noqa: E402
from proactive_eval.sources.replay import ReplaySource  # noqa: E402

register_source("eigenflux", EigenFluxSource)
register_source("search", SearchSource)
register_source("replay", ReplaySource)
