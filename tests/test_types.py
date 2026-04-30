"""Tests for core data types."""

from datetime import datetime, timezone

from proactive_eval.types import SignalItem, Event, Tick, Trajectory


class TestSignalItem:

    def test_creation(self):
        item = SignalItem(
            item_id="test-1",
            source="search",
            received_at=datetime.now(timezone.utc),
            content="Breaking news about AI",
        )
        assert item.item_id == "test-1"
        assert item.source == "search"
        assert item.url is None
        assert item.metadata == {}

    def test_with_url(self):
        item = SignalItem(
            item_id="test-2",
            source="eigenflux",
            received_at=datetime.now(timezone.utc),
            content="Signal content",
            url="https://example.com/article",
        )
        assert item.url == "https://example.com/article"


class TestTrajectory:

    def test_empty_trajectory(self):
        t = Trajectory(fixture_id="test_fixture", run_label="search")
        assert t.ticks == []
        assert t.config == {}

    def test_trajectory_with_ticks(self):
        now = datetime.now(timezone.utc)
        tick = Tick(
            tick_index=0,
            time_window=(now, now),
            raw_output='{"decision": {"action": "skip"}}',
        )
        t = Trajectory(fixture_id="f1", run_label="ef", ticks=[tick])
        assert len(t.ticks) == 1
        assert t.ticks[0].tick_index == 0
