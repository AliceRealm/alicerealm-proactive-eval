"""Tests for dedup logic (hash-based only; embedding tests need API key)."""

from datetime import datetime, timezone

from proactive_eval.dedup import content_hash, dedup_signal_items
from proactive_eval.types import SignalItem


def _make_item(item_id: str, content: str, source: str = "search") -> SignalItem:
    return SignalItem(
        item_id=item_id,
        source=source,
        received_at=datetime.now(timezone.utc),
        content=content,
    )


class TestContentHash:

    def test_identical_strings(self):
        assert content_hash("hello world") == content_hash("hello world")

    def test_whitespace_normalization(self):
        assert content_hash("hello   world") == content_hash("hello world")
        assert content_hash("  hello world  ") == content_hash("hello world")

    def test_case_normalization(self):
        assert content_hash("Hello World") == content_hash("hello world")

    def test_different_strings(self):
        assert content_hash("hello") != content_hash("world")


class TestDedupSignalItems:

    def test_no_duplicates(self):
        items = [_make_item("1", "aaa"), _make_item("2", "bbb")]
        result = dedup_signal_items(items, semantic=False)
        assert len(result) == 2

    def test_exact_id_dedup(self):
        items = [
            _make_item("1", "aaa"),
            _make_item("1", "aaa"),  # same id
        ]
        result = dedup_signal_items(items, semantic=False)
        assert len(result) == 1

    def test_eigenflux_dedup_by_item_id(self):
        items = [
            _make_item("ef-1", "content A", source="eigenflux"),
            _make_item("ef-1", "content A copy", source="eigenflux"),
        ]
        result = dedup_signal_items(items, semantic=False)
        assert len(result) == 1

    def test_mixed_sources_no_cross_dedup(self):
        items = [
            _make_item("1", "same content", source="search"),
            _make_item("1", "same content", source="eigenflux"),
        ]
        result = dedup_signal_items(items, semantic=False)
        assert len(result) == 2
