"""Simulation layer: tick-based orchestration of agent harnesses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from proactive_eval.types import SignalItem, Tick, Trajectory


class SimulationHarness(ABC):
    """Black-box simulation harness.

    Receives batches of signals per tick, manages its own internal state,
    and produces raw output per tick. Does NOT do structured extraction —
    that happens in the scoring layer.
    """

    @abstractmethod
    def initialize(self, fixture: dict, scenario: dict | None, config: dict) -> None:
        """Set up the harness with agent persona, scenario context, and config."""
        ...

    @abstractmethod
    def process_tick(self, tick_index: int, signals: list[SignalItem],
                     time_window: tuple[datetime, datetime]) -> Tick:
        """Process one tick's worth of signals. Returns a Tick with raw_output populated."""
        ...

    def on_tick_end(self, tick: Tick) -> None:
        """Hook called after each tick completes.
        Override for dynamic behavior (e.g., adjusting search terms).
        Default: no-op."""
        pass

    def finalize(self) -> dict:
        """Return any harness-level summary data. Called after all ticks."""
        return {}


def group_signals_by_tick(signals: list[SignalItem],
                          tick_duration_minutes: int,
                          start_time: datetime | None = None,
                          end_time: datetime | None = None,
                          ) -> list[tuple[tuple[datetime, datetime], list[SignalItem]]]:
    """Group signals into time windows of tick_duration_minutes.

    Returns list of ((window_start, window_end), signals_in_window).
    Empty windows are included to maintain tick cadence.
    """
    if not signals:
        return []

    sorted_signals = sorted(signals, key=lambda s: s.received_at)
    first = start_time or sorted_signals[0].received_at
    last = end_time or sorted_signals[-1].received_at
    delta = timedelta(minutes=tick_duration_minutes)

    ticks = []
    window_start = first
    sig_idx = 0

    while window_start <= last:
        window_end = window_start + delta
        batch = []
        while sig_idx < len(sorted_signals) and sorted_signals[sig_idx].received_at < window_end:
            if sorted_signals[sig_idx].received_at >= window_start:
                batch.append(sorted_signals[sig_idx])
            sig_idx += 1
        ticks.append(((window_start, window_end), batch))
        window_start = window_end

    return ticks


def run_simulation(harness: SimulationHarness,
                   fixture: dict,
                   signals: list[SignalItem],
                   tick_duration_minutes: int = 30,
                   scenario: dict | None = None,
                   config: dict | None = None) -> Trajectory:
    """The tick loop: groups signals by time window, feeds each batch to the harness.

    This is the main orchestrator — it does NOT know what happens inside the harness.
    """
    config = config or {}
    harness.initialize(fixture, scenario, config)

    tick_groups = group_signals_by_tick(signals, tick_duration_minutes)

    trajectory = Trajectory(
        fixture_id=fixture.get("fixture_id", ""),
        model=config.get("model", ""),
        config=config,
        started_at=datetime.now(timezone.utc),
    )

    for tick_index, (time_window, tick_signals) in enumerate(tick_groups):
        tick = harness.process_tick(tick_index, tick_signals, time_window)
        harness.on_tick_end(tick)
        trajectory.ticks.append(tick)

    trajectory.finished_at = datetime.now(timezone.utc)
    trajectory.harness_summary = harness.finalize()
    return trajectory
