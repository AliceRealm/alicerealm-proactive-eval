"""JSON report generation."""

from __future__ import annotations

import json
from pathlib import Path

from proactive_eval.types import ScoredTrajectory


def generate_json_report(scored_trajectories: list[ScoredTrajectory],
                         output_path: Path) -> Path:
    """Generate a machine-readable JSON report."""
    report = {
        "trajectories": [],
    }

    for st in scored_trajectories:
        traj = st.trajectory
        entry = {
            "fixture_id": traj.fixture_id,
            "run_label": traj.run_label,
            "model": traj.model,
            "started_at": traj.started_at.isoformat() if traj.started_at else None,
            "finished_at": traj.finished_at.isoformat() if traj.finished_at else None,
            "scores": st.scores,
            "stats": st.stats,
            "details": st.details,
            "tick_count": len(traj.ticks),
            "event_count": len(st.extracted_events),
        }
        report["trajectories"].append(entry)

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return output_path
