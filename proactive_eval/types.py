"""Core data structures for the proactive-eval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SignalItem:
    """A single signal from any source, normalized to a common format."""

    item_id: str
    source: str  # "eigenflux", "search", etc.
    received_at: datetime
    content: str
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_item: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """A structured event extracted from simulation output by the scoring layer."""

    type: str  # "send", "skip", "action", "memory_update", "user_reply", ...
    content: Any = None
    item_id: str | None = None
    tick_index: int | None = None


@dataclass
class Tick:
    """One time-window's worth of simulation output."""

    tick_index: int
    time_window: tuple[datetime, datetime]
    signals_received: list[SignalItem] = field(default_factory=list)
    raw_output: str = ""
    events: list[Event] = field(default_factory=list)
    api_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete output of a simulation run."""

    fixture_id: str = ""
    run_label: str = ""  # e.g. "eigenflux", "search"
    model: str = ""
    ticks: list[Tick] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)
    harness_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredTrajectory:
    """A trajectory with scores attached by the scoring layer."""

    trajectory: Trajectory
    scores: dict[str, float] = field(default_factory=dict)
    extracted_events: list[Event] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
