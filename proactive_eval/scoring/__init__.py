"""Scoring layer: extract structured events and compute scores from trajectories."""

from __future__ import annotations

from abc import ABC, abstractmethod

from proactive_eval.types import Trajectory, Event, ScoredTrajectory


class Extractor(ABC):
    """Extracts structured events from a trajectory's raw output."""

    @abstractmethod
    def extract(self, trajectory: Trajectory) -> list[Event]:
        ...


class Scorer(ABC):
    """Computes scores from extracted events."""

    @abstractmethod
    def score(self, trajectory: Trajectory, events: list[Event]) -> dict:
        """Returns {dimension_name: score, ...} plus breakdown details."""
        ...


def score_trajectory(trajectory: Trajectory,
                     extractor: Extractor,
                     scorer: Scorer) -> ScoredTrajectory:
    """Full scoring pipeline: extract events, then score."""
    events = extractor.extract(trajectory)
    result = scorer.score(trajectory, events)
    return ScoredTrajectory(
        trajectory=trajectory,
        scores=result.get("scores", {}),
        extracted_events=events,
        details=result.get("details", {}),
        stats=result.get("stats", {}),
    )
