"""Unified CLI entry point for proactive-eval.

Usage:
    python -m proactive_eval collect    --source eigenflux --once
    python -m proactive_eval simulate   --config config/experiments/sample.json
    python -m proactive_eval score      --experiment <id>
    python -m proactive_eval report     --experiment <id>
    python -m proactive_eval run        --config ...   # end-to-end pipeline
    python -m proactive_eval setup      --source eigenflux
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from proactive_eval.config import (
    SCRIPT_DIR, load_env, load_fixtures, load_experiment_config,
    make_experiment_id, rel_path, apply_config_to_args,
)
from proactive_eval.types import SignalItem

EXPERIMENTS_DIR = SCRIPT_DIR / "data" / "experiments"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def filter_by_time(items: list[SignalItem],
                   start: str | None, end: str | None) -> list[SignalItem]:
    if start is None and end is None:
        return items
    start_dt = _parse_timestamp(start) if start else None
    end_dt = _parse_timestamp(end) if end else None
    return [
        it for it in items
        if (not start_dt or it.received_at >= start_dt)
        and (not end_dt or it.received_at <= end_dt)
    ]


AUTO_SAMPLE_THRESHOLD = 500  # per-fixture: auto-sample when items exceed this
AUTO_SAMPLE_SIZE = 400       # target sample size when auto-sampling


def sample_items(items: list[SignalItem], sample_size: int | None,
                 seed: int = 42, auto: bool = True) -> tuple[list[SignalItem], int]:
    population = len(items)
    # Auto-sample: if no explicit sample_size and population exceeds threshold
    if sample_size is None and auto and population > AUTO_SAMPLE_THRESHOLD:
        sample_size = AUTO_SAMPLE_SIZE
        print(f"    [auto-sample] {population} items > {AUTO_SAMPLE_THRESHOLD} threshold, "
              f"sampling {sample_size}")
    if sample_size is None or sample_size >= population:
        return items, population
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(population), sample_size))
    return [items[i] for i in indices], population


# ---------------------------------------------------------------------------
# Pre-steps
# ---------------------------------------------------------------------------

def pre_step_clear_impressions(fixtures: list[dict]):
    import urllib.request
    base_url = os.environ.get("EIGENFLUX_CONSOLE_URL", "http://localhost:8090") + "/console/api/v1/impr/items"
    for fixture in fixtures:
        agent_id = fixture.get("eigenflux_profile", {}).get("agent_id")
        if not agent_id:
            continue
        url = f"{base_url}?agent_id={agent_id}"
        req = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"  [clear_impressions] {fixture['fixture_id']}: {resp.status}")
        except Exception as e:
            print(f"  [clear_impressions] {fixture['fixture_id']}: FAILED ({e})")


PRE_STEPS = {"clear_impressions": pre_step_clear_impressions}


# ---------------------------------------------------------------------------
# Subcommand: collect
# ---------------------------------------------------------------------------

def cmd_collect(args):
    """Collect signals from a source."""
    from proactive_eval.sources import get_source
    from proactive_eval.sources.eigenflux import sync_profiles

    all_fixtures = load_fixtures()
    fixtures = _select_fixtures(all_fixtures, args.fixture)

    source = get_source(args.source)

    # Auto-sync profiles for eigenflux
    if args.source == "eigenflux" and not args.dry_run and not args.skip_profile_sync:
        print("=== Syncing profiles ===")
        sync_profiles(fixtures)
        print()

    if args.once:
        total_cycles = 1
    elif args.duration > 0:
        total_cycles = int(args.duration * 60 / args.interval)
    else:
        total_cycles = None  # forever

    print(f"=== Collecting from {args.source} ===")
    print(f"Fixtures: {len(fixtures)} | Interval: {args.interval}min")
    print()

    cycle = 0
    while True:
        cycle += 1
        print(f"[Cycle {cycle}] @ {datetime.now().isoformat()}")
        for fix in fixtures:
            fid = fix["fixture_id"]
            collect_kwargs = {"dry_run": args.dry_run}
            if args.source == "search" and getattr(args, "llm_queries", False):
                collect_kwargs["llm_queries"] = True
                collect_kwargs["num_queries"] = args.num_queries
                collect_kwargs["query_model"] = args.query_model
            items = source.collect(fid, **collect_kwargs)
            print(f"  {fid}: {len(items)} items")

        if total_cycles is not None and cycle >= total_cycles:
            break
        print(f"  Sleeping {args.interval}min...")
        time.sleep(args.interval * 60)

    print(f"\n=== Done. {cycle} cycles. ===")


# ---------------------------------------------------------------------------
# Subcommand: simulate
# ---------------------------------------------------------------------------

def cmd_simulate(args):
    """Run simulation on collected signals."""
    from proactive_eval.dedup import dedup_signal_items
    from proactive_eval.simulation import run_simulation
    from proactive_eval.simulation.per_item import PerItemHarness
    from proactive_eval.scoring import score_trajectory
    from proactive_eval.scoring.decision_extractor import DecisionExtractor
    from proactive_eval.scoring.quality_scorer import QualityScorer
    from proactive_eval.report.html import generate_report

    # Load config file if provided
    pre_steps: list[str] = []
    config_file_path = None
    if args.config:
        config_file_path = str(Path(args.config).resolve())
        experiment_cfg = load_experiment_config(args.config)
        # Create a parser with defaults to merge config
        parser = _make_simulate_parser()
        pre_steps = apply_config_to_args(parser, args, experiment_cfg)
        print(f"Loaded config: {config_file_path}")

    all_fixtures = load_fixtures()
    fixtures = _select_fixtures(all_fixtures, args.fixture,
                                getattr(args, "_config_fixture_indices", None))

    # Run pre-steps
    for step_name in pre_steps:
        handler = PRE_STEPS.get(step_name)
        if handler:
            print(f"Running pre-step: {step_name}")
            handler(fixtures)

    # Set up Claude client
    client = None
    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            print("Error: `anthropic` not installed. Use --dry-run or pip install anthropic.")
            sys.exit(1)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        client = anthropic.Anthropic()

    # Create experiment directory
    experiment_id = make_experiment_id(args.model)
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    semantic_dedup = not args.no_semantic_dedup

    # Save experiment config
    config = {
        "experiment_id": experiment_id,
        "resolved_from": config_file_path,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "source": args.source,
        "workers": args.workers,
        "sample": args.sample,
        "seed": args.seed,
        "semantic_dedup": semantic_dedup,
        "limit": args.limit,
        "start": args.start,
        "end": args.end,
        "dry_run": args.dry_run,
        "pre_steps": pre_steps,
        "targets": {
            "daily_sends": args.daily_target,
            "send_rate": args.send_rate_target,
            "action_rate": args.action_rate_target,
        },
        "fixtures": [f["fixture_id"] for f in fixtures],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    # Each experiment runs a single source
    run_label = "results"

    # Load signals
    from proactive_eval.sources.eigenflux import EigenFluxSource
    from proactive_eval.sources.search import SearchSource

    if args.source == "eigenflux":
        src = EigenFluxSource()
    elif args.source == "searches":
        src = SearchSource()
    else:
        print(f"Error: unknown source '{args.source}'")
        sys.exit(1)

    # --- Pre-load all signals to auto-detect time range ---
    all_signals_by_fixture: dict[str, list] = {}
    for fixture in fixtures:
        fid = fixture["fixture_id"]
        signals = src.replay(fid)
        signals.sort(key=lambda s: s.received_at)
        signals = dedup_signal_items(signals, semantic=semantic_dedup, verbose=False)
        all_signals_by_fixture[fid] = signals

    # Auto-detect time range from data if not specified
    if not args.start or not args.end:
        all_timestamps = []
        for signals in all_signals_by_fixture.values():
            for s in signals:
                if s.received_at:
                    all_timestamps.append(s.received_at)
        if all_timestamps:
            data_start = min(all_timestamps)
            data_end = max(all_timestamps)
            if not args.start:
                args.start = data_start.isoformat()
            if not args.end:
                args.end = data_end.isoformat()
            print(f"  Data time range: {args.start[:19]} → {args.end[:19]}")

    # Update config with resolved time range
    config["start"] = args.start
    config["end"] = args.end

    # Compute days for daily_yield
    days = 1.0
    if args.start and args.end:
        try:
            d0 = _parse_timestamp(args.start)
            d1 = _parse_timestamp(args.end)
            days = max((d1 - d0).total_seconds() / 86400, 0.01)
        except Exception:
            pass
    config["data_days"] = round(days, 4)

    # Save updated config
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    print(f"=== Decision Simulation ===")
    print(f"Experiment: {experiment_id}")
    print(f"Fixtures: {len(fixtures)} | Source: {args.source} | Model: {args.model}")
    print(f"Time window: {args.start or '?'} → {args.end or '?'} ({days:.2f} days)")
    if args.dry_run:
        print("  DRY-RUN mode")
    print()

    all_scored = []

    for fixture in fixtures:
        fid = fixture["fixture_id"]
        signals = all_signals_by_fixture.get(fid, [])

        # Time filter
        if args.start or args.end:
            signals = filter_by_time(signals, args.start, args.end)

        if not signals:
            print(f"--- {fid}: 0 items (skipping)")
            continue

        # Sample (auto-samples when > threshold)
        signals, population = sample_items(signals, args.sample, seed=args.seed)
        if args.limit and len(signals) > args.limit:
            signals = signals[:args.limit]
        print(f"--- {fid}: {len(signals)}/{population} items")

        # Create harness and run simulation
        harness = PerItemHarness(
            client=client,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            window_size=getattr(args, "window", 30),
            dry_run=args.dry_run,
            workers=args.workers,
        )

        trajectory = run_simulation(
            harness=harness,
            fixture=fixture,
            signals=signals,
            tick_duration_minutes=99999,
            config=config,
        )
        trajectory.run_label = run_label

        # Score
        extractor = DecisionExtractor()
        scorer = QualityScorer(
            targets=config["targets"],
            days=days,
        )
        scored = score_trajectory(trajectory, extractor, scorer)
        scored.stats["population_size"] = population
        scored.stats["sample_size"] = len(signals)
        all_scored.append(scored)

        stats = scored.stats
        sr = stats.get("send_rate", 0)
        ar = stats.get("action_rate", 0)
        print(f"  DONE {fid}: {stats.get('decisions', {})} | "
              f"send={sr:.0%} action={ar:.0%}")

    # Generate report
    if all_scored:
        generate_report(all_scored, experiment_dir)

    # Update config
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    print(f"\n=== Done. Experiment: {experiment_id} ===")


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def cmd_compare(args):
    """Compare 2+ experiments side-by-side."""
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from visualize_results import load_experiment, generate_compare_html, EXPERIMENTS_DIR

    experiments = []
    for eid in args.experiments:
        exp_dir = EXPERIMENTS_DIR / eid
        if not exp_dir.exists():
            print(f"Error: experiment not found: {eid}")
            sys.exit(1)
        exp = load_experiment(exp_dir, include_decisions=True)
        if not exp:
            print(f"Error: failed to load experiment: {eid}")
            sys.exit(1)
        experiments.append(exp)

    html = generate_compare_html(experiments)

    # Write to a compare file
    ids_str = "_vs_".join(args.experiments[:3])
    out_path = EXPERIMENTS_DIR / f"compare_{ids_str}" / "index.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html)
    print(f"Compare page: {out_path}")

    if not args.no_serve:
        from visualize_results import serve
        serve(args.port, f"/compare_{ids_str}/index.html")


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def cmd_report(args):
    """Generate or serve reports."""
    from proactive_eval.report.html import (
        print_experiment_list, serve_dashboard, EXPERIMENTS_DIR,
    )

    if args.list:
        print_experiment_list()
        return

    if args.no_serve:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            from visualize_results import build_all_html
            build_all_html()
            print(f"HTML generated in {EXPERIMENTS_DIR}")
        except Exception as e:
            print(f"Error: {e}")
        return

    if args.experiment:
        exp_dir = EXPERIMENTS_DIR / args.experiment
        if not exp_dir.exists():
            print(f"Error: experiment not found: {args.experiment}")
            sys.exit(1)
        open_path = f"/{args.experiment}/index.html"
    else:
        # Find latest
        if EXPERIMENTS_DIR.exists():
            dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
            if dirs:
                open_path = f"/{dirs[-1].name}/index.html"
            else:
                open_path = "/"
        else:
            open_path = "/"

    serve_dashboard(args.port, open_path)


# ---------------------------------------------------------------------------
# Subcommand: setup
# ---------------------------------------------------------------------------

def cmd_setup(args):
    """Interactive setup for signal sources."""
    from proactive_eval.sources import get_source

    all_fixtures = load_fixtures()
    fixtures = _select_fixtures(all_fixtures, args.fixture)
    source = get_source(args.source)

    print(f"=== Setting up {args.source} ===\n")
    for i, fix in enumerate(fixtures):
        print(f"[{i+1}/{len(fixtures)}] {fix['fixture_id']}")
        source.setup(fix)
        print()
    print("=== Setup complete ===")


# ---------------------------------------------------------------------------
# Fixture selection helper
# ---------------------------------------------------------------------------

def _select_fixtures(all_fixtures: list[dict], fixture_idx: int | None,
                     config_indices: list[int] | None = None) -> list[dict]:
    if fixture_idx is not None:
        idx = fixture_idx - 1
        if idx < 0 or idx >= len(all_fixtures):
            print(f"Error: --fixture must be 1-{len(all_fixtures)}")
            sys.exit(1)
        return [all_fixtures[idx]]
    if config_indices is not None:
        fixtures = []
        for ci in config_indices:
            idx = ci - 1
            if 0 <= idx < len(all_fixtures):
                fixtures.append(all_fixtures[idx])
        if not fixtures:
            print("Error: no valid fixtures selected")
            sys.exit(1)
        return fixtures
    return all_fixtures


# ---------------------------------------------------------------------------
# Parser builders
# ---------------------------------------------------------------------------

def _make_simulate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=int, default=None)
    parser.add_argument("--source", choices=["eigenflux", "searches"], default="eigenflux")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no-semantic-dedup", action="store_true")
    parser.add_argument("--daily-target", type=float, default=15)
    parser.add_argument("--send-rate-target", type=float, default=0.70)
    parser.add_argument("--action-rate-target", type=float, default=0.15)
    parser.add_argument("--config", default=None)
    parser.add_argument("--window", type=int, default=30)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_env()

    parser = argparse.ArgumentParser(
        prog="proactive_eval",
        description="Evaluation framework for proactive AI agent behavior",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- collect ---
    p_collect = subparsers.add_parser("collect", help="Collect signals from a source")
    p_collect.add_argument("--source", required=True, choices=["eigenflux", "search"])
    p_collect.add_argument("--fixture", type=int, default=None)
    p_collect.add_argument("--once", action="store_true")
    p_collect.add_argument("--interval", type=int, default=5)
    p_collect.add_argument("--duration", type=float, default=0)
    p_collect.add_argument("--dry-run", action="store_true")
    p_collect.add_argument("--skip-profile-sync", action="store_true",
                           help="Do not push fixture profiles at startup")
    p_collect.add_argument("--llm-queries", action="store_true",
                           help="Generate search queries via LLM instead of static JSON")
    p_collect.add_argument("--num-queries", type=int, default=5,
                           help="Number of queries to generate per fixture (default: 5)")
    p_collect.add_argument("--query-model", default="claude-haiku-4-5-20251001",
                           help="Model for query generation (default: claude-haiku-4-5-20251001)")

    # --- simulate ---
    p_sim = subparsers.add_parser("simulate", help="Run simulation on collected signals")
    p_sim.add_argument("--fixture", type=int, default=None)
    p_sim.add_argument("--source", choices=["eigenflux", "searches"], default="eigenflux")
    p_sim.add_argument("--model", default=DEFAULT_MODEL)
    p_sim.add_argument("--sample", type=int, default=None)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--limit", type=int, default=None)
    p_sim.add_argument("--start", default=None)
    p_sim.add_argument("--end", default=None)
    p_sim.add_argument("--dry-run", action="store_true")
    p_sim.add_argument("--workers", type=int, default=4)
    p_sim.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_sim.add_argument("--temperature", type=float, default=None)
    p_sim.add_argument("--no-semantic-dedup", action="store_true")
    p_sim.add_argument("--daily-target", type=float, default=15)
    p_sim.add_argument("--send-rate-target", type=float, default=0.70)
    p_sim.add_argument("--action-rate-target", type=float, default=0.15)
    p_sim.add_argument("--config", default=None)
    p_sim.add_argument("--window", type=int, default=30)

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate or serve reports")
    p_report.add_argument("experiment", nargs="?", default=None)
    p_report.add_argument("--list", action="store_true")
    p_report.add_argument("--port", type=int, default=8420)
    p_report.add_argument("--no-serve", action="store_true")

    # --- compare ---
    p_compare = subparsers.add_parser("compare", help="Compare 2+ experiments")
    p_compare.add_argument("experiments", nargs="+", metavar="EXPERIMENT_ID",
                           help="Experiment IDs to compare")
    p_compare.add_argument("--port", type=int, default=8420)
    p_compare.add_argument("--no-serve", action="store_true")

    # --- setup ---
    p_setup = subparsers.add_parser("setup", help="Interactive source setup")
    p_setup.add_argument("--source", required=True, choices=["eigenflux"])
    p_setup.add_argument("--fixture", type=int, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "setup":
        cmd_setup(args)


if __name__ == "__main__":
    main()
