"""Tests for scoring logic."""

import math

from proactive_eval.scoring.quality_scorer import QualityScorer, wilson_ci
from proactive_eval.types import Trajectory, Event


class TestWilsonCI:

    def test_zero_trials(self):
        p, lo, hi = wilson_ci(0, 0)
        assert p == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_all_successes(self):
        p, lo, hi = wilson_ci(100, 100)
        assert p == 1.0
        assert lo > 0.9
        assert hi > 0.99

    def test_half(self):
        p, lo, hi = wilson_ci(50, 100)
        assert abs(p - 0.5) < 0.01
        assert lo < 0.5
        assert hi > 0.5

    def test_bounds(self):
        _, lo, hi = wilson_ci(3, 10)
        assert 0.0 <= lo <= hi <= 1.0


class TestQualityScorer:

    def test_empty_events(self):
        scorer = QualityScorer()
        t = Trajectory(fixture_id="test")
        result = scorer.score(t, [])
        assert result["scores"] == {}

    def test_all_skips(self):
        scorer = QualityScorer()
        events = [Event(type="skip") for _ in range(10)]
        result = scorer.score(Trajectory(), events)
        assert result["stats"]["send_rate"] == 0.0
        assert result["stats"]["action_rate"] == 0.0

    def test_mixed_decisions(self):
        scorer = QualityScorer()
        events = [
            Event(type="skip"),
            Event(type="send"),
            Event(type="send_with_action"),
            Event(type="skip"),
            Event(type="send"),
        ]
        result = scorer.score(Trajectory(), events)
        stats = result["stats"]
        assert stats["decisions"]["skip"] == 2
        assert stats["decisions"]["send"] == 2
        assert stats["decisions"]["send_with_action"] == 1
        assert stats["send_rate"] == 0.6  # 3/5
        assert stats["action_rate"] == 0.2  # 1/5
