"""HTML dashboard generation.

This is a thin wrapper that delegates to the existing visualize_results.py
for HTML generation, while providing the new-pipeline interface.

The full HTML generation logic is complex (~700 lines of CSS/JS/SVG).
Rather than duplicating it, we import and reuse visualize_results.py
until the report layer is fully migrated.
"""

from __future__ import annotations

import json
from pathlib import Path

from proactive_eval.config import SCRIPT_DIR
from proactive_eval.types import ScoredTrajectory

EXPERIMENTS_DIR = SCRIPT_DIR / "data" / "experiments"


def generate_report(scored_trajectories: list[ScoredTrajectory],
                    experiment_dir: Path) -> Path:
    """Generate HTML report from scored trajectories.

    For now, writes summary JSONs in the format expected by
    visualize_results.py, then calls build_all_html().
    """
    for st in scored_trajectories:
        traj = st.trajectory
        fixture_dir = experiment_dir / traj.fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)

        # Write summary in old format
        summary = dict(st.stats)
        summary["fixture_id"] = traj.fixture_id
        summary["model"] = traj.model
        summary["started_at"] = traj.started_at.isoformat() if traj.started_at else None
        summary["finished_at"] = traj.finished_at.isoformat() if traj.finished_at else None
        summary["log_path"] = f"{traj.fixture_id}/results.jsonl"
        summary["dry_run"] = traj.config.get("dry_run", False)

        summary_path = fixture_dir / "results.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        # Write JSONL (decision log entries from raw_output)
        jsonl_path = fixture_dir / "results.jsonl"
        with jsonl_path.open("w") as f:
            for tick in traj.ticks:
                if tick.raw_output:
                    for line in tick.raw_output.strip().split("\n"):
                        if line.strip():
                            f.write(line.strip() + "\n")

    # Regenerate HTML using existing visualize_results
    try:
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        from visualize_results import build_all_html
        build_all_html()
        print("  HTML updated.")
    except Exception as e:
        print(f"  HTML generation failed: {e}")

    return experiment_dir


def list_experiments() -> list[dict]:
    """List all experiments from the data directory."""
    if not EXPERIMENTS_DIR.exists():
        return []
    experiments = []
    for d in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        if cfg_path.exists():
            experiments.append(json.loads(cfg_path.read_text()))
    return experiments


def print_experiment_list():
    """Print all experiments in terminal."""
    experiments = list_experiments()
    if not experiments:
        print("No experiments found.")
        return
    print(f"\n{'='*70}")
    print(f"{'ID':<45} {'model':<30} {'source':<8} {'range'}")
    print(f"{'-'*45} {'-'*30} {'-'*8} {'-'*30}")
    for e in experiments:
        eid = e.get("experiment_id", "?")
        model = e.get("model", "?")
        source = e.get("source", "?")
        r = f"{e.get('start') or '?'} -> {e.get('end') or '?'}"
        dry = " [dry]" if e.get("dry_run") else ""
        print(f"{eid:<45} {model:<30} {source:<8} {r}{dry}")
    print()


def serve_dashboard(port: int = 8420, open_path: str = "/"):
    """Start HTTP server for the dashboard."""
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from visualize_results import serve
    serve(port, open_path)
