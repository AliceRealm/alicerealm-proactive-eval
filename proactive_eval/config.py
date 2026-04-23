"""Experiment configuration loading and CLI arg merging."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = SCRIPT_DIR / "fixtures"


def load_env():
    """Load .env file from project root (no dependency on python-dotenv)."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_fixtures() -> list[dict]:
    """Load all fixture JSON files, sorted by filename."""
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        data["_path"] = str(path)
        fixtures.append(data)
    return fixtures


def load_experiment_config(config_path: str) -> dict:
    """Load and return an experiment config JSON file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    return json.loads(path.read_text())


def make_experiment_id(model: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    return f"{ts}__{safe_model}"


def rel_path(path: Path) -> str:
    """Stringify path relative to SCRIPT_DIR if possible, else absolute."""
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Config merging (CLI args + config file)
# ---------------------------------------------------------------------------

def _arg_was_defaulted(parser: argparse.ArgumentParser,
                       args: argparse.Namespace, dest: str) -> bool:
    return getattr(args, dest) == parser.get_default(dest)


def apply_config_to_args(parser: argparse.ArgumentParser,
                         args: argparse.Namespace,
                         cfg: dict) -> list[str]:
    """Merge experiment config into args namespace.
    Only overrides args that were NOT explicitly set on the CLI.
    Returns the list of pre_steps to run."""
    mapping = [
        (["model"], "model", None),
        # backend is inferred from model name, not stored in config
        (["max_tokens"], "max_tokens", None),
        (["temperature"], "temperature", None),
        (["source"], "source", None),
        (["time_range", "start"], "start", None),
        (["time_range", "end"], "end", None),
        (["sampling", "size"], "sample", None),
        (["sampling", "seed"], "seed", None),
        (["targets", "daily_sends"], "daily_target", None),
        (["targets", "send_rate"], "send_rate_target", None),
        (["targets", "action_rate"], "action_rate_target", None),
        (["workers"], "workers", None),
    ]

    for key_path, dest, transform in mapping:
        val = cfg
        for k in key_path:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = None
                break
        if val is None:
            continue
        if _arg_was_defaulted(parser, args, dest):
            if transform:
                val = transform(val)
            setattr(args, dest, val)

    if "dedup" in cfg and _arg_was_defaulted(parser, args, "no_semantic_dedup"):
        args.no_semantic_dedup = (cfg["dedup"] != "semantic")

    if "fixtures" in cfg and _arg_was_defaulted(parser, args, "fixture"):
        fix_val = cfg["fixtures"]
        if isinstance(fix_val, list):
            args._config_fixture_indices = fix_val
        elif fix_val == "all":
            args.fixture = None
        elif isinstance(fix_val, int):
            args.fixture = fix_val

    # Replay config merging
    replay_cfg = cfg.get("replay", {})
    if replay_cfg:
        replay_mappings = [
            ("url", "replay_url"),
            ("simulated_at", "simulated_at"),
            ("limit", "replay_limit"),
            ("use_feed_history", "use_feed_history"),
        ]
        for cfg_key, arg_dest in replay_mappings:
            if cfg_key in replay_cfg and _arg_was_defaulted(parser, args, arg_dest):
                setattr(args, arg_dest, replay_cfg[cfg_key])

        # Ranker params from config file
        ranker = replay_cfg.get("ranker_params", {})
        for key in ("alpha", "beta", "gamma", "delta",
                     "min_relevance_score", "exploration_slots"):
            if key in ranker and _arg_was_defaulted(parser, args, key.replace("-", "_")):
                setattr(args, key.replace("-", "_"), ranker[key])

        # Recall params from config file
        recall = replay_cfg.get("recall_params", {})
        for key in ("keyword_recall_size", "knn_recall_k"):
            if key in recall and _arg_was_defaulted(parser, args, key):
                setattr(args, key, recall[key])
        if recall.get("enable_knn_recall") and _arg_was_defaulted(parser, args, "enable_knn_recall"):
            args.enable_knn_recall = True

    return cfg.get("pre_steps", [])
