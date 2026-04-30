"""Tests for config loading and fixture validation."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "fixtures"
SCENARIOS_DIR = ROOT / "scenarios"
CONDITIONS_DIR = ROOT / "config" / "conditions"


# ---------------------------------------------------------------------------
# Fixture / scenario file integrity
# ---------------------------------------------------------------------------

def _load_all_json(directory: Path) -> list[tuple[str, dict]]:
    results = []
    for p in sorted(directory.glob("*.json")):
        with open(p) as f:
            results.append((p.name, json.load(f)))
    return results


class TestFixtures:
    """Validate that every fixture file is well-formed."""

    fixtures = _load_all_json(FIXTURES_DIR)

    @pytest.mark.parametrize("name,data", fixtures, ids=[f[0] for f in fixtures])
    def test_required_fields(self, name, data):
        assert "fixture_id" in data, f"{name} missing fixture_id"
        assert "functional_traits" in data, f"{name} missing functional_traits"

    @pytest.mark.parametrize("name,data", fixtures, ids=[f[0] for f in fixtures])
    def test_fixture_id_matches_filename(self, name, data):
        # e.g. "1_careful_operator.json" → fixture_id should contain "careful_operator"
        stem = Path(name).stem
        # strip leading number + underscore
        expected_suffix = "_".join(stem.split("_")[1:])
        assert expected_suffix in data["fixture_id"], (
            f"{name}: fixture_id '{data['fixture_id']}' doesn't match filename"
        )


class TestScenarios:
    """Validate that every scenario file is well-formed."""

    scenarios = _load_all_json(SCENARIOS_DIR)

    @pytest.mark.parametrize("name,data", scenarios, ids=[f[0] for f in scenarios])
    def test_required_fields(self, name, data):
        assert "scenario_id" in data, f"{name} missing scenario_id"


class TestConditions:
    """Validate condition config files."""

    conditions = _load_all_json(CONDITIONS_DIR)

    @pytest.mark.parametrize("name,data", conditions, ids=[f[0] for f in conditions])
    def test_has_condition_id(self, name, data):
        assert "condition_id" in data, f"{name} missing 'condition_id' field"


class TestFixtureScenarioPairing:
    """Each fixture should have a corresponding scenario."""

    def test_counts_match(self):
        fixture_count = len(list(FIXTURES_DIR.glob("*.json")))
        scenario_count = len(list(SCENARIOS_DIR.glob("*.json")))
        assert fixture_count == scenario_count, (
            f"{fixture_count} fixtures but {scenario_count} scenarios"
        )
