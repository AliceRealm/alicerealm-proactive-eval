#!/usr/bin/env python3
"""
visualize_results.py — serve experiment results as an interactive dashboard.

Usage:
  python3 visualize_results.py                                # latest experiment → open in browser
  python3 visualize_results.py <experiment_id>                # specific experiment
  python3 visualize_results.py --compare <id1> <id2> [id3..]  # compare N experiments
  python3 visualize_results.py --list                         # list all experiments (terminal)
  python3 visualize_results.py --port 9000                    # custom port (default: 8420)
  python3 visualize_results.py --no-serve                     # just generate HTML, don't serve
"""

import argparse
import json
import math
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR / "data" / "experiments"
DEFAULT_PORT = 8420

FIXTURE_NAMES = {
    "careful_operator_v1": "Careful Operator (Sable)",
    "executive_finisher_v1": "Executive Finisher (Flint)",
    "social_scout_v1": "Social Scout (Ember)",
    "research_librarian_v1": "Research Librarian (Mist)",
    "content_strategist_v1": "Content Strategist (Nova)",
    "learning_curator_v1": "Learning Curator (Lark)",
}

# Dynamic color palette — supports up to 8 experiments in comparison
COLOR_PALETTE = [
    {"bar": "#f97316", "dot": "#c2410c", "hex": "#f97316"},  # orange
    {"bar": "#3b82f6", "dot": "#2563eb", "hex": "#3b82f6"},  # blue
    {"bar": "#10b981", "dot": "#059669", "hex": "#10b981"},  # green
    {"bar": "#8b5cf6", "dot": "#7c3aed", "hex": "#8b5cf6"},  # purple
    {"bar": "#ec4899", "dot": "#db2777", "hex": "#ec4899"},  # pink
    {"bar": "#f59e0b", "dot": "#d97706", "hex": "#f59e0b"},  # amber
    {"bar": "#06b6d4", "dot": "#0891b2", "hex": "#06b6d4"},  # cyan
    {"bar": "#ef4444", "dot": "#dc2626", "hex": "#ef4444"},  # red
]


def _fixture_name(fid: str) -> str:
    return FIXTURE_NAMES.get(fid, fid.replace("_v1", "").replace("_", " ").title())


def _color_for(idx: int) -> dict:
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return p_hat, max(0.0, centre - margin), min(1.0, centre + margin)


def _backfill_summary(summary: dict, config: dict | None = None):
    """Ensure all derived fields exist, computing from decisions dict if needed."""
    n = summary.get("sample_size", summary.get("items_processed", 0))
    d = summary.get("decisions", {})
    send_count = d.get("send", 0) + d.get("send_with_action", 0)
    action_count = d.get("send_with_action", 0)

    if summary.get("send_rate") is None:
        sr, lo, hi = _wilson_ci(send_count, n)
        summary["send_rate"] = round(sr, 4)
        summary["send_rate_ci95"] = [round(lo, 4), round(hi, 4)]

    if summary.get("action_rate") is None:
        ar, lo, hi = _wilson_ci(action_count, n)
        summary["action_rate"] = round(ar, 4)
        summary["action_rate_ci95"] = [round(lo, 4), round(hi, 4)]

    if summary.get("population_size") is None:
        summary["population_size"] = n
    if summary.get("sample_size") is None:
        summary["sample_size"] = n

    if summary.get("quality_per_item") is None:
        q = (send_count - action_count) * 1 + action_count * 3
        summary["quality_total"] = q
        summary["quality_per_item"] = round(q / n, 4) if n else 0

    if summary.get("daily_yield") is None:
        pop = summary.get("population_size", n)
        days = 1.0
        if config:
            start = config.get("start")
            end = config.get("end")
            if start and end:
                try:
                    from datetime import datetime, timezone
                    d0 = datetime.fromisoformat(start)
                    d1 = datetime.fromisoformat(end)
                    if d0.tzinfo is None:
                        d0 = d0.replace(tzinfo=timezone.utc)
                    if d1.tzinfo is None:
                        d1 = d1.replace(tzinfo=timezone.utc)
                    days = max((d1 - d0).total_seconds() / 86400, 0.01)
                except Exception:
                    pass
        summary["daily_yield"] = round(pop / days, 1)


def load_experiment(exp_dir: Path, include_decisions: bool = True) -> dict:
    """Load config.json and all summary files from an experiment directory.
    Supports both new format (results.summary.json) and old format (*_only.summary.json)."""
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text())

    fixtures = {}
    for child in sorted(exp_dir.iterdir()):
        if not child.is_dir():
            continue
        fid = child.name
        fixtures[fid] = {}
        for summary_file in sorted(child.glob("*.summary.json")):
            run_label = summary_file.name.replace(".summary.json", "")
            summary = json.loads(summary_file.read_text())
            _backfill_summary(summary, config)
            decisions = []
            if include_decisions:
                jsonl_path = child / f"{run_label}.jsonl"
                if jsonl_path.exists():
                    with jsonl_path.open() as f:
                        for line in f:
                            decisions.append(json.loads(line))
            fixtures[fid][run_label] = {
                "summary": summary,
                "decisions": decisions,
            }

    config["_fixtures"] = fixtures
    return config


def list_experiments() -> list[dict]:
    if not EXPERIMENTS_DIR.exists():
        return []
    experiments = []
    for d in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if d.name.startswith("compare_"):
            continue
        cfg_path = d / "config.json"
        if cfg_path.exists():
            experiments.append(json.loads(cfg_path.read_text()))
    return experiments


def latest_experiment_dir() -> Path | None:
    if not EXPERIMENTS_DIR.exists():
        return None
    dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir()
                   if d.is_dir() and not d.name.startswith("compare_")])
    return dirs[-1] if dirs else None


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _short_ts_range(start: str | None, end: str | None) -> str:
    if not start and not end:
        return "(no time window)"
    from datetime import datetime
    parts = []
    for ts in [start, end]:
        if not ts:
            parts.append("open")
            continue
        try:
            dt = datetime.fromisoformat(ts)
            parts.append(dt.strftime("%b %d %H:%M"))
        except Exception:
            parts.append(ts[:16])
    return f"{parts[0]} → {parts[1]}"


def _format_duration(started: str | None, finished: str | None) -> tuple[str, bool]:
    if not started:
        return "—", False
    if not finished:
        return "running...", False
    try:
        from datetime import datetime, timezone
        s = datetime.fromisoformat(started)
        f = datetime.fromisoformat(finished)
        delta = f - s
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s", True
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s", True
        return f"{secs // 3600}h {(secs % 3600) // 60}m", True
    except Exception:
        return "—", bool(finished)


def _experiment_short_label(exp: dict) -> str:
    """Generate a short human-readable label for an experiment."""
    source = exp.get("source", "?")
    model = exp.get("model", "?").replace("claude-", "").replace("-20251001", "")
    return f"{source} / {model}"


# ---------------------------------------------------------------------------
# Terminal: --list
# ---------------------------------------------------------------------------

def print_experiment_list():
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


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg: #fff; --fg: #1a1a1a; --muted: #666; --border: #e0e0e0;
        --header-bg: #f7f7f7; --row-alt: #fafafa; --accent: #2563eb;
        --green: #16a34a; --red: #dc2626; --yellow: #ca8a04; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1a1a1a; --fg: #e5e5e5; --muted: #999; --border: #333;
          --header-bg: #252525; --row-alt: #222; --accent: #60a5fa;
          --green: #4ade80; --red: #f87171; --yellow: #fbbf24; }
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 2em 1.5em;
       background: var(--bg); color: var(--fg); line-height: 1.5; }
h1 { font-size: 1.3em; margin-bottom: 0.3em; }
h2 { font-size: 1.05em; margin-top: 2.5em; margin-bottom: 0.5em;
     padding-bottom: 0.3em; border-bottom: 1px solid var(--border); }
.meta { color: var(--muted); font-size: 0.85em; margin-bottom: 1.5em; }
.meta span { margin-right: 1.2em; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 0.85em; }
th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; white-space: nowrap; }
th { background: var(--header-bg); font-weight: 600; position: sticky; top: 0; }
tr:nth-child(even) { background: var(--row-alt); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pct { font-weight: 600; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.8em; font-weight: 600; }
.tag-send { background: #dbeafe; color: #1e40af; }
.tag-action { background: #fef3c7; color: #92400e; }
.tag-skip { background: #f3f4f6; color: #6b7280; }
.bar { display: inline-block; height: 14px; border-radius: 2px; vertical-align: middle; }
.bar-skip { background: #d1d5db; }
.bar-send { background: #60a5fa; }
.bar-action { background: #fbbf24; }
nav { margin-bottom: 1.5em; }
nav a { color: var(--accent); text-decoration: none; margin-right: 1em; font-size: 0.85em; }
nav a:hover { text-decoration: underline; }
.fixture-card { border: 1px solid var(--border); border-radius: 8px; margin: 1.2em 0; overflow: hidden; }
.fixture-header { background: var(--header-bg); padding: 8px 14px; font-weight: 600;
                  font-size: 0.95em; display: flex; justify-content: space-between; align-items: center; }
.fixture-col { padding: 12px 14px; }
.fixture-col + .fixture-col { border-left: 1px solid var(--border); }
.fixture-col h4 { margin: 0 0 8px 0; font-size: 0.85em; color: var(--muted); font-weight: 600; }
.stat-grid { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 0.85em; }
.stat-label { color: var(--muted); }
.stat-val { font-variant-numeric: tabular-nums; }
.stat-val.highlight { font-weight: 700; font-size: 1.05em; }
.delta-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; }
.delta-pos { background: #dcfce7; color: #166534; }
.delta-neg { background: #fef2f2; color: #991b1b; }
.delta-neutral { background: #f3f4f6; color: #6b7280; }
@media (prefers-color-scheme: dark) {
  .tag-send { background: #1e3a5f; color: #93c5fd; }
  .tag-action { background: #451a03; color: #fcd34d; }
  .tag-skip { background: #374151; color: #9ca3af; }
  .delta-pos { background: #052e16; color: #86efac; }
  .delta-neg { background: #450a0a; color: #fca5a5; }
  .delta-neutral { background: #374151; color: #9ca3af; }
}
"""

_JS = """
document.querySelectorAll('.send-row').forEach(row => {
  row.addEventListener('click', () => {
    const detail = row.nextElementSibling;
    if (detail && detail.classList.contains('detail')) {
      detail.style.display = detail.style.display === 'table-row' ? 'none' : 'table-row';
    }
  });
});
"""


def _bar_html(skip: int, send: int, send_action: int, width_px: int = 120) -> str:
    total = skip + send + send_action
    if total == 0:
        return ""
    parts = []
    for n, cls in [(skip, "bar-skip"), (send, "bar-send"), (send_action, "bar-action")]:
        if n > 0:
            w = max(1, int(n / total * width_px))
            parts.append(f'<span class="bar {cls}" style="width:{w}px" title="{n}"></span>')
    return "".join(parts)


def _gauge_html(val: float, target: float, max_val: float | None = None,
                label: str = "") -> str:
    if max_val is None:
        max_val = target * 2.5
    pct = min(val / max_val * 100, 100) if max_val else 0
    target_pct = min(target / max_val * 100, 100) if max_val else 0
    color = "var(--green)" if val >= target else "var(--yellow)" if val >= target * 0.7 else "var(--red)"
    return (
        f'<div style="position:relative;height:26px;background:var(--row-alt);'
        f'border-radius:4px;border:1px solid var(--border);overflow:hidden">'
        f'<div style="height:100%;width:{pct:.0f}%;background:{color};opacity:0.3"></div>'
        f'<div style="position:absolute;left:{target_pct:.0f}%;top:0;bottom:0;'
        f'width:2px;background:var(--fg);opacity:0.4" title="target: {target}"></div>'
        f'<div style="position:absolute;left:8px;top:4px;font-size:0.8em;font-weight:600">'
        f'{label}</div></div>'
    )


# ---------------------------------------------------------------------------
# Page: Single experiment detail
# ---------------------------------------------------------------------------

def generate_experiment_html(exp: dict) -> str:
    fixtures = exp.get("_fixtures", {})
    eid = exp.get("experiment_id", "?")
    targets = exp.get("targets", {})
    daily_target = targets.get("daily_sends", 15)
    send_rate_target = targets.get("send_rate", 0.70)
    action_rate_target = targets.get("action_rate", 0.15)

    # Collect all run labels (for backward compat with old multi-label experiments)
    all_labels = set()
    for runs in fixtures.values():
        all_labels.update(runs.keys())
    labels = sorted(all_labels)

    # --- Per-fixture cards ---
    def _col_html(s: dict, label: str) -> str:
        d = s["decisions"]
        total = s["items_processed"]
        send = d["send"]
        send_action = d["send_with_action"]
        skip = d["skip"]
        pop = s.get("population_size", total)
        n_str = f"{total} / {pop}" if pop > total else str(total)
        sr = s.get("send_rate", 0)
        sci = s.get("send_rate_ci95", [0, 0])
        daily = s.get("daily_yield", 0)
        bar = _bar_html(skip, send, send_action, width_px=140)
        ar = s.get("action_rate", 0)
        aci = s.get("action_rate_ci95", [0, 0])
        qpi = s.get("quality_per_item", 0)
        if not qpi and total:
            qpi = ((send - send_action) * 1 + send_action * 3) / total
        quality_baseline = (send_rate_target - action_rate_target) * 1 + action_rate_target * 3

        daily_gauge = _gauge_html(daily, daily_target,
                                  label=f'{daily:.1f}/day (baseline: {daily_target})')
        send_gauge = _gauge_html(sr * 100, send_rate_target * 100, max_val=100,
                                 label=f'{sr*100:.1f}% [{sci[0]*100:.0f}, {sci[1]*100:.0f}]')
        action_gauge = _gauge_html(ar * 100, action_rate_target * 100, max_val=50,
                                   label=f'{ar*100:.1f}% [{aci[0]*100:.0f}, {aci[1]*100:.0f}]')
        quality_gauge = _gauge_html(qpi, quality_baseline, max_val=3.0,
                                    label=f'{qpi:.2f} (baseline: {quality_baseline:.1f})')

        return (
            f'<div class="fixture-col">'
            f'<h4>{_esc(label)}</h4>'
            f'<div class="stat-grid">'
            f'<span class="stat-label">Sample (n/N)</span><span class="stat-val">{n_str}</span>'
            f'<span class="stat-label">Daily Yield</span><span>{daily_gauge}</span>'
            f'<span class="stat-label">Send Rate</span><span>{send_gauge}</span>'
            f'<span class="stat-label">Action Rate</span><span>{action_gauge}</span>'
            f'<span class="stat-label">Quality</span><span>{quality_gauge}</span>'
            f'<span class="stat-label">Skip / Send / Action</span>'
            f'<span class="stat-val">{skip} / {send} / {send_action}</span>'
            f'<span class="stat-label">Distribution</span><span>{bar}</span>'
            f'</div></div>'
        )

    fixture_cards = []
    for fid in sorted(fixtures.keys()):
        runs = fixtures[fid]
        if not runs:
            continue
        short = _fixture_name(fid)
        n_cols = len(labels)
        grid_cols = " ".join(["1fr"] * max(1, n_cols))
        cols = ""
        for l in labels:
            if l in runs:
                cols += _col_html(runs[l]["summary"], l)
            else:
                cols += f'<div class="fixture-col"><h4>{_esc(l)}</h4><span class="stat-label">No data</span></div>'

        fixture_cards.append(
            f'<div class="fixture-card">'
            f'<div class="fixture-header"><span>{_esc(short)}</span></div>'
            f'<div style="display:grid;grid-template-columns:{grid_cols};gap:0">{cols}</div>'
            f'</div>'
        )

    # --- Sent items ---
    total_sends = 0
    sent_sections = []
    for fid in sorted(fixtures.keys()):
        runs = fixtures[fid]
        has_sends = any(
            d.get("decision", {}).get("action") in ("send", "send_with_action")
            for l in labels if l in runs
            for d in runs[l].get("decisions", [])
        )
        if not has_sends:
            continue

        fixture_cols = []
        for l in labels:
            if l not in runs:
                continue
            all_decisions = runs[l].get("decisions", [])
            sends = [d for d in all_decisions
                     if d.get("decision", {}).get("action") in ("send", "send_with_action")]
            total_sends += len(sends)
            cards = []
            for d in sends:
                action = d["decision"]["action"]
                tag_cls = "tag-send" if action == "send" else "tag-action"
                tag_text = "SEND" if action == "send" else "ACTION"
                ts = d.get("pull_timestamp", "")
                short_ts = ts[5:16] if len(ts) > 16 else ts
                raw = d.get("raw_item", {})
                title = _esc(raw.get("title") or raw.get("summary", "")[:80] or "")
                body = _esc(raw.get("body") or raw.get("content", "")[:200] or d.get("content_preview", "")[:200])
                rationale = _esc(d["decision"].get("rationale", ""))
                message = _esc(d["decision"].get("message") or "")
                rec_html = ""
                if action == "send_with_action":
                    ra = d["decision"].get("recommended_action") or {}
                    if isinstance(ra, str):
                        ra = {"title": ra, "details": ""}
                    rec_html = (
                        f'<div style="margin-top:6px;padding:6px 8px;background:var(--header-bg);'
                        f'border-radius:4px;border-left:3px solid var(--yellow)">'
                        f'<div style="font-weight:600;font-size:0.85em">{_esc(ra.get("title", ""))}</div>'
                        f'<div style="font-size:0.8em;color:var(--muted)">{_esc(ra.get("details", ""))}</div>'
                        f'</div>'
                    )
                cards.append(
                    f'<div style="border:1px solid var(--border);border-radius:6px;margin-bottom:8px;overflow:hidden">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:4px 10px;background:var(--header-bg);border-bottom:1px solid var(--border)">'
                    f'<span class="tag {tag_cls}">{tag_text}</span>'
                    f'<span style="font-size:0.75em;color:var(--muted)">{short_ts}</span></div>'
                    f'<div style="padding:8px 10px">'
                    f'<div style="font-weight:600;font-size:0.9em;margin-bottom:4px">{title}</div>'
                    f'<div style="font-size:0.8em;color:var(--muted);margin-bottom:6px">{body}</div>'
                    f'<div style="padding:6px 8px;background:var(--row-alt);border-radius:4px;'
                    f'font-size:0.85em;margin-bottom:4px;border-left:3px solid var(--accent)">{message}</div>'
                    f'<details style="font-size:0.8em;color:var(--muted)">'
                    f'<summary style="cursor:pointer">Why this was sent</summary>'
                    f'<div style="margin-top:4px">{rationale}</div></details>'
                    f'{rec_html}'
                    f'</div></div>'
                )
            s = runs[l]["summary"]
            send_count = s["decisions"]["send"]
            action_count = s["decisions"]["send_with_action"]
            col_header = (f'<h4>{_esc(l)}'
                          f' <span style="font-weight:400;font-size:0.8em;color:var(--muted)">'
                          f'{send_count} send + {action_count} action</span></h4>')
            fixture_cols.append(
                f'<div class="fixture-col" style="max-height:600px;overflow-y:auto">'
                f'{col_header}'
                f'{"".join(cards) if cards else "<span class=stat-label>No sends</span>"}'
                f'</div>'
            )
        n_cols = len(fixture_cols)
        grid_cols = " ".join(["1fr"] * max(1, n_cols))
        sent_sections.append(
            f'<div class="fixture-card">'
            f'<div class="fixture-header"><span>{_esc(_fixture_name(fid))}</span></div>'
            f'<div style="display:grid;grid-template-columns:{grid_cols};gap:0">{"".join(fixture_cols)}</div>'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Experiment {_esc(eid)}</title>
<style>{_CSS}</style></head><body>

<nav><a href="../index.html" style="font-weight:600">&larr; All Experiments</a></nav>

<h1>Experiment: {_esc(eid)}</h1>
<p class="meta">
  <span>Model: <b>{_esc(str(exp.get('model')))}</b></span>
  <span>Source: <b>{_esc(str(exp.get('source')))}</b></span>
  <span>Data window: <b>{_esc(_short_ts_range(exp.get('start'), exp.get('end')))}</b></span>
  <span>Data span: <b>{exp.get('data_days', '?')} days</b></span>
  <span>Sample: <b>{exp.get('sample') or 'all'}</b> (seed={exp.get('seed', '—')})</span>
  <span>Dedup: <b>{"semantic" if exp.get("semantic_dedup", False) else "URL only"}</b></span>
  <span>Sim duration: <b>{_format_duration(exp.get('started_at'), exp.get('finished_at'))[0]}</b></span>
  {"<span><b>DRY RUN</b></span>" if exp.get('dry_run') else ""}
</p>

<h2>Per-Fixture Results</h2>
<p style="font-size:0.8em;color:var(--muted)">
  <span class="bar bar-skip" style="width:12px"></span> skip &nbsp;
  <span class="bar bar-send" style="width:12px"></span> send &nbsp;
  <span class="bar bar-action" style="width:12px"></span> send+action
</p>
{"".join(fixture_cards)}

<h2>Sent Items ({total_sends})</h2>
{"".join(sent_sections)}

<script>{_JS}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Page: Compare N experiments
# ---------------------------------------------------------------------------

def generate_compare_html(experiments: list[dict]) -> str:
    """Generate a comparison page for N experiments."""
    n_exp = len(experiments)
    if n_exp < 2:
        return "<html><body>Need at least 2 experiments to compare.</body></html>"

    # Assign colors and labels
    exp_labels = []
    for i, exp in enumerate(experiments):
        color = _color_for(i)
        source = exp.get("source", "?")
        model = exp.get("model", "?").replace("claude-", "").replace("-20251001", "")
        eid = exp.get("experiment_id", "?")
        exp_labels.append({
            "idx": i, "color": color, "source": source, "model": model,
            "eid": eid, "label": f"{source} / {model}",
            "exp": exp,
        })

    all_fixture_ids = sorted(set(
        fid for exp in experiments
        for fid in exp.get("_fixtures", {}).keys()
    ))

    # --- Executive Summary: aggregate stats per experiment ---
    agg = []
    for el in exp_labels:
        exp = el["exp"]
        fixtures = exp.get("_fixtures", {})
        total_items = total_sends = total_actions = 0
        send_rates = []
        total_cost = 0.0
        for fid in all_fixture_ids:
            if fid not in fixtures:
                continue
            for rl, rd in fixtures[fid].items():
                s = rd["summary"]
                n = s.get("items_processed", 0)
                total_items += n
                d = s.get("decisions", {})
                total_sends += d.get("send", 0) + d.get("send_with_action", 0)
                total_actions += d.get("send_with_action", 0)
                send_rates.append(s.get("send_rate", 0))
                for dec in rd.get("decisions", []):
                    api = dec.get("api", {})
                    total_cost += (api.get("input_tokens", 0) * 0.25 +
                                   api.get("output_tokens", 0) * 1.25) / 1_000_000
        avg_sr = sum(send_rates) / len(send_rates) if send_rates else 0
        agg.append({
            "items": total_items, "sends": total_sends, "actions": total_actions,
            "avg_sr": avg_sr, "cost": total_cost,
        })

    # Big number cards
    def _big_num_card(label: str, values: list, fmt: str = ".0f", unit: str = "") -> str:
        rows = ""
        for i, v in enumerate(values):
            c = _color_for(i)["hex"]
            el = exp_labels[i]
            rows += (f'<div style="display:flex;align-items:center;gap:8px">'
                     f'<span style="font-size:0.75em;font-weight:600;color:{c};min-width:70px">'
                     f'{_esc(el["source"])}</span>'
                     f'<span style="font-size:1.3em;font-weight:700;font-variant-numeric:tabular-nums">'
                     f'{v:{fmt}}{unit}</span></div>')
        return (
            f'<div style="border:1px solid var(--border);border-radius:8px;padding:14px 16px;background:var(--header-bg)">'
            f'<div style="font-size:0.8em;color:var(--muted);margin-bottom:8px;text-transform:uppercase;'
            f'letter-spacing:0.03em">{label}</div>'
            f'{rows}</div>'
        )

    exec_summary = (
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:1.2em 0">'
        f'{_big_num_card("Send Count", [a["sends"] for a in agg])}'
        f'{_big_num_card("Action Count", [a["actions"] for a in agg])}'
        f'{_big_num_card("Avg Send Rate", [a["avg_sr"] * 100 for a in agg], ".1f", "%")}'
        f'{_big_num_card("Total Items", [a["items"] for a in agg])}'
        f'{_big_num_card("Est. Cost", [a["cost"] * 100 for a in agg], ".2f", " cents")}'
        f'</div>'
    )

    # --- Per-Fixture Comparison Table ---
    # Dynamic columns: Fixture | (n, SR, AR) per experiment | Δ (if 2 experiments)
    header_cells = '<th>Fixture</th>'
    for el in exp_labels:
        c = el["color"]["hex"]
        header_cells += (f'<th style="border-bottom:3px solid {c}">n</th>'
                         f'<th style="border-bottom:3px solid {c}">Send</th>'
                         f'<th style="border-bottom:3px solid {c}">Action</th>'
                         f'<th style="border-bottom:3px solid {c}">SR</th>')
    if n_exp == 2:
        header_cells += '<th>&Delta; SR</th><th>&Delta; Send</th>'

    table_rows = []
    for fid in all_fixture_ids:
        fname = _fixture_name(fid)
        cells = f'<td style="font-weight:600">{_esc(fname)}</td>'
        sr_vals = []
        send_vals = []
        for el in exp_labels:
            exp = el["exp"]
            fixtures = exp.get("_fixtures", {})
            if fid in fixtures:
                for rl, rd in fixtures[fid].items():
                    s = rd["summary"]
                    items = s.get("items_processed", 0)
                    sr = s.get("send_rate", 0)
                    d = s.get("decisions", {})
                    send_n = d.get("send", 0) + d.get("send_with_action", 0)
                    action_n = d.get("send_with_action", 0)
                    sr_vals.append(sr)
                    send_vals.append(send_n)
                    cells += (f'<td class="num">{items}</td>'
                              f'<td class="num">{send_n}</td>'
                              f'<td class="num">{action_n}</td>'
                              f'<td class="num pct">{sr*100:.1f}%</td>')
                    break
            else:
                sr_vals.append(None)
                send_vals.append(None)
                cells += '<td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td>'

        if n_exp == 2 and sr_vals[0] is not None and sr_vals[1] is not None:
            delta = (sr_vals[0] - sr_vals[1]) * 100
            cls = "delta-pos" if delta > 2 else "delta-neg" if delta < -2 else "delta-neutral"
            sign = "+" if delta > 0 else ""
            cells += f'<td class="num"><span class="delta-badge {cls}">{sign}{delta:.1f}pp</span></td>'
            # Delta send count
            ds = send_vals[0] - send_vals[1]
            scls = "delta-pos" if ds > 0 else "delta-neg" if ds < 0 else "delta-neutral"
            ssign = "+" if ds > 0 else ""
            cells += f'<td class="num"><span class="delta-badge {scls}">{ssign}{ds}</span></td>'
        elif n_exp == 2:
            cells += '<td class="num">—</td><td class="num">—</td>'

        table_rows.append(f'<tr>{cells}</tr>')

    fixture_table = f'<table><tr>{header_cells}</tr>{"".join(table_rows)}</table>'

    # --- CI Chart (send rate comparison) ---
    ci_rows = []
    for fid in sorted(all_fixture_ids):
        short = _fixture_name(fid)
        for i, el in enumerate(exp_labels):
            exp = el["exp"]
            fixtures = exp.get("_fixtures", {})
            if fid not in fixtures:
                continue
            for rl, rd in fixtures[fid].items():
                s = rd["summary"]
                sr = s.get("send_rate", 0) or 0
                ci = s.get("send_rate_ci95", [sr, sr]) or [sr, sr]
                n = s.get("sample_size", s.get("items_processed", 0))
                pop = s.get("population_size", n)
                n_str = f"{n}/{pop}" if pop > n else str(n)
                c = _color_for(i)
                left_pct = ci[0] * 100
                width_pct = max(0.5, (ci[1] - ci[0]) * 100)
                dot_pct = sr * 100
                ci_rows.append(
                    f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:0.85em">'
                    f'<div style="width:180px;flex-shrink:0;text-align:right;padding-right:12px">'
                    f'{_esc(short)} / {_esc(el["source"])}</div>'
                    f'<div style="position:relative;flex:1;height:28px;background:var(--row-alt);'
                    f'border-radius:3px;border:1px solid var(--border)">'
                    f'<div style="position:absolute;top:6px;height:16px;border-radius:2px;opacity:0.7;'
                    f'background:{c["bar"]};left:{left_pct:.1f}%;width:{width_pct:.1f}%"></div>'
                    f'<div style="position:absolute;top:4px;width:4px;height:20px;border-radius:2px;'
                    f'background:{c["dot"]};left:{dot_pct:.1f}%"></div>'
                    f'<div style="position:absolute;top:5px;font-size:0.8em;font-weight:600;white-space:nowrap;'
                    f'left:{max(dot_pct+1.5, ci[1]*100+1):.1f}%">{sr*100:.0f}% (n={n_str})</div>'
                    f'</div></div>'
                )
                break

    ci_legend = ""
    for i, el in enumerate(exp_labels):
        c = _color_for(i)["hex"]
        ci_legend += (f'<span style="margin-right:1.5em">'
                      f'<span class="bar" style="width:12px;background:{c}"></span> '
                      f'{_esc(el["source"])}</span>')
    ci_html = (
        f'<div style="margin:1.5em 0">'
        f'<div style="font-size:0.8em;color:var(--muted);margin-bottom:1em">'
        f'{ci_legend} &nbsp; | &nbsp; bar = 95% CI, dot = point estimate</div>'
        f'{"".join(ci_rows)}'
        f'</div>'
    )

    # --- Side-by-side Sent Items ---
    sent_sections = []
    for fid in all_fixture_ids:
        cols = []
        for i, el in enumerate(exp_labels):
            exp = el["exp"]
            fixtures = exp.get("_fixtures", {})
            c = _color_for(i)["hex"]
            if fid not in fixtures:
                cols.append(f'<div class="fixture-col"><h4 style="color:{c}">{_esc(el["source"])}</h4>'
                            f'<span class="stat-label">No data</span></div>')
                continue
            cards = []
            for rl, rd in fixtures[fid].items():
                sends = [d for d in rd.get("decisions", [])
                         if d.get("decision", {}).get("action") in ("send", "send_with_action")]
                for d in sends:
                    action = d["decision"]["action"]
                    tag_cls = "tag-send" if action == "send" else "tag-action"
                    tag_text = "SEND" if action == "send" else "ACTION"
                    raw = d.get("raw_item", {})
                    title = _esc(raw.get("title") or raw.get("summary", "")[:80] or "")
                    body = _esc(raw.get("body") or raw.get("content", "")[:150] or d.get("content_preview", "")[:150])
                    message = _esc(d["decision"].get("message") or "")
                    rationale = _esc(d["decision"].get("rationale", ""))
                    rec_html = ""
                    if action == "send_with_action":
                        ra = d["decision"].get("recommended_action") or {}
                        if isinstance(ra, str):
                            ra = {"title": ra, "details": ""}
                        rec_html = (
                            f'<div style="margin-top:4px;padding:4px 6px;background:var(--header-bg);'
                            f'border-radius:3px;border-left:2px solid var(--yellow);font-size:0.8em">'
                            f'<b>{_esc(ra.get("title", ""))}</b> {_esc(ra.get("details", ""))}</div>'
                        )
                    cards.append(
                        f'<div style="border:1px solid var(--border);border-radius:5px;margin-bottom:6px;'
                        f'padding:8px;font-size:0.82em">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                        f'<span class="tag {tag_cls}">{tag_text}</span></div>'
                        f'<div style="font-weight:600;margin-bottom:3px">{title}</div>'
                        f'<div style="color:var(--muted);margin-bottom:4px;font-size:0.9em">{body}</div>'
                        f'<div style="padding:4px 6px;background:var(--row-alt);border-radius:3px;'
                        f'border-left:2px solid {c}">{message}</div>'
                        f'<details style="font-size:0.85em;color:var(--muted);margin-top:3px">'
                        f'<summary style="cursor:pointer">Rationale</summary>{rationale}</details>'
                        f'{rec_html}</div>'
                    )
            n_sends = len(cards)
            cols.append(
                f'<div class="fixture-col" style="max-height:500px;overflow-y:auto">'
                f'<h4 style="color:{c}">{_esc(el["source"])} ({n_sends} sent)</h4>'
                f'{"".join(cards) if cards else "<span class=stat-label>No sends</span>"}'
                f'</div>'
            )
        grid_cols = " ".join(["1fr"] * n_exp)
        sent_sections.append(
            f'<div class="fixture-card">'
            f'<div class="fixture-header"><span>{_esc(_fixture_name(fid))}</span></div>'
            f'<div style="display:grid;grid-template-columns:{grid_cols};gap:0">{"".join(cols)}</div>'
            f'</div>'
        )

    # --- Nav ---
    nav_links = " ".join(
        f'<a href="../{_esc(el["eid"])}/index.html" style="color:{el["color"]["hex"]}">'
        f'{_esc(el["source"])} detail</a>'
        for el in exp_labels
    )

    # --- Meta line ---
    meta_parts = []
    for el in exp_labels:
        c = el["color"]["hex"]
        exp = el["exp"]
        n_items = sum(
            rd["summary"].get("items_processed", 0)
            for fid_data in exp.get("_fixtures", {}).values()
            for rd in fid_data.values()
        )
        meta_parts.append(
            f'<span style="color:{c};font-weight:600">{_esc(el["source"])}</span>: '
            f'{_esc(el["eid"][:20])}… ({n_items} items)'
        )

    title_parts = " vs ".join(_esc(el["source"]) for el in exp_labels)

    compare_css = """
.exec-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
             gap: 12px; margin: 1.2em 0; }
"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Compare: {title_parts}</title>
<style>{_CSS}{compare_css}</style></head><body>

<nav><a href="../index.html" style="font-weight:600">&larr; All Experiments</a> {nav_links}</nav>

<h1>Compare: {title_parts}</h1>
<p class="meta">{"&nbsp;&nbsp;vs&nbsp;&nbsp;".join(meta_parts)}</p>

<h2>Executive Summary</h2>
{exec_summary}

<h2>Per-Fixture Comparison</h2>
<p style="font-size:0.8em;color:var(--muted)">
  <span class="bar bar-skip" style="width:12px"></span> skip &nbsp;
  <span class="bar bar-send" style="width:12px"></span> send &nbsp;
  <span class="bar bar-action" style="width:12px"></span> action
</p>
{fixture_table}

<h2>Sent Items (Side-by-Side)</h2>
<p style="font-size:0.8em;color:var(--muted)">All items that passed the filter, grouped by fixture.</p>
{"".join(sent_sections)}

<script>{_JS}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Page: Index (experiment list with multi-select compare)
# ---------------------------------------------------------------------------

def generate_index_html() -> str:
    all_exps = list_experiments()

    from collections import defaultdict
    date_groups = defaultdict(list)
    for e in all_exps:
        eid = e.get("experiment_id", "?")
        date_key = eid[:8] if len(eid) >= 8 else "unknown"
        if len(date_key) == 8 and date_key.isdigit():
            date_key = f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:8]}"
        date_groups[date_key].append(e)

    cards_html = []
    for date_str in sorted(date_groups.keys(), reverse=True):
        exps = date_groups[date_str]
        group_cards = []
        for e in exps:
            eid = e.get("experiment_id", "?")
            dry = ' <span style="color:var(--yellow);font-size:0.75em">[DRY]</span>' if e.get("dry_run") else ""
            source = e.get("source", "?")
            model = e.get("model", "?")
            model_short = model.replace("claude-", "").replace("-20251001", "")
            duration, complete = _format_duration(e.get("started_at"), e.get("finished_at"))
            status_color = "var(--green)" if complete else "var(--yellow)"
            n_fixtures = len(e.get("fixtures", []))

            # Count total items
            total_items = 0
            exp_dir = EXPERIMENTS_DIR / eid
            if exp_dir.exists():
                for fixture_dir in exp_dir.iterdir():
                    if fixture_dir.is_dir():
                        for sf in fixture_dir.glob("*.summary.json"):
                            try:
                                s = json.loads(sf.read_text())
                                total_items += s.get("items_processed", 0)
                            except Exception:
                                pass

            # Avg send rate
            avg_send_rate = None
            rates = []
            if exp_dir.exists():
                for fixture_dir in exp_dir.iterdir():
                    if fixture_dir.is_dir():
                        for sf in fixture_dir.glob("*.summary.json"):
                            try:
                                s = json.loads(sf.read_text())
                                sr = s.get("send_rate")
                                if sr is not None:
                                    rates.append(sr)
                            except Exception:
                                pass
            if rates:
                avg_send_rate = sum(rates) / len(rates)
            send_rate_str = f"{avg_send_rate*100:.0f}%" if avg_send_rate is not None else "—"

            # Dynamic badge color based on source string
            source_lower = source.lower()
            if "eigenflux" in source_lower:
                badge_color = COLOR_PALETTE[0]["hex"]
            elif "search" in source_lower:
                badge_color = COLOR_PALETTE[1]["hex"]
            else:
                badge_color = "var(--muted)"

            del_btn = (f'<button onclick="event.stopPropagation();deleteExp(\'{_esc(eid)}\')" '
                       f'style="font-size:0.7em;padding:2px 8px;cursor:pointer;'
                       f'border:1px solid var(--border);background:var(--bg);color:var(--red);'
                       f'border-radius:3px;opacity:0.6" title="Delete experiment">&times;</button>')

            group_cards.append(
                f'<div class="exp-card" onclick="window.location=\'{_esc(eid)}/index.html\'">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'<div style="flex:1">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'<input type="checkbox" class="compare-check" value="{_esc(eid)}" onclick="event.stopPropagation();updateCompare()">'
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{badge_color}"></span>'
                f'<span style="font-weight:600;font-size:0.9em">{_esc(source)}</span>'
                f'<span style="font-size:0.8em;color:var(--muted)">{_esc(model_short)}</span>'
                f'{dry}'
                f'</div>'
                f'<div style="display:flex;gap:16px;font-size:0.82em;color:var(--muted);flex-wrap:wrap">'
                f'<span title="Data time window">{_esc(_short_ts_range(e.get("start"), e.get("end")))}</span>'
                f'<span title="Total items processed">{total_items} items</span>'
                f'<span title="Average send rate">send: <b style="color:var(--fg)">{send_rate_str}</b></span>'
                f'<span title="Number of fixtures">{n_fixtures} fixtures</span>'
                f'<span title="Simulation duration" style="color:{status_color}">{duration}</span>'
                f'</div>'
                f'</div>'
                f'<div>{del_btn}</div>'
                f'</div>'
                f'<div style="font-size:0.72em;color:var(--muted);margin-top:4px;font-family:monospace">{_esc(eid)}</div>'
                f'</div>'
            )

        cards_html.append(
            f'<div class="date-group">'
            f'<div class="date-header">{_esc(date_str)}</div>'
            f'{"".join(group_cards)}'
            f'</div>'
        )

    index_css_extra = """
.exp-card { border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px;
            cursor: pointer; transition: border-color 0.15s, box-shadow 0.15s; }
.exp-card:hover { border-color: var(--accent); box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.date-group { margin-bottom: 1.5em; }
.date-header { font-size: 0.8em; font-weight: 600; color: var(--muted); margin-bottom: 8px;
               text-transform: uppercase; letter-spacing: 0.05em; }
.date-group .exp-card + .exp-card { margin-top: 8px; }
.compare-bar { position: sticky; bottom: 0; background: var(--bg); border-top: 1px solid var(--border);
               padding: 12px 0; display: none; align-items: center; gap: 12px; z-index: 10; }
.compare-bar.active { display: flex; }
.compare-bar button { background: var(--accent); color: #fff; border: none; padding: 8px 20px;
                      border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 0.85em; }
.compare-bar button:disabled { opacity: 0.4; cursor: not-allowed; }
"""

    # JS: support N-way comparison (min 2, no max)
    index_js = """
function updateCompare() {
  var checks = document.querySelectorAll('.compare-check:checked');
  var bar = document.getElementById('compare-bar');
  var btn = document.getElementById('compare-btn');
  var label = document.getElementById('compare-label');
  if (checks.length >= 1) {
    bar.classList.add('active');
    label.textContent = checks.length + ' selected';
    btn.disabled = checks.length < 2;
  } else {
    bar.classList.remove('active');
  }
}
function doCompare() {
  var checks = document.querySelectorAll('.compare-check:checked');
  if (checks.length < 2) return;
  var ids = Array.from(checks).map(c => c.value);
  var sorted = ids.slice().sort();
  var cachePath = 'compare_' + sorted.join('_vs_') + '/index.html';
  if (location.protocol === 'file:') {
    // file:// mode: try cached version, alert if missing
    var a = document.createElement('a');
    a.href = cachePath;
    // Can't detect 404 on file://, just navigate — browser shows error if missing
    window.location = cachePath;
  } else {
    // http mode: hit server which generates + caches + redirects
    var url = '/compare?' + ids.map(id => 'exp=' + encodeURIComponent(id)).join('&');
    window.location = url;
  }
}
function deleteExp(eid) {
  if (!confirm('Delete experiment ' + eid + '?')) return;
  if (location.protocol === 'file:') {
    alert('Delete requires the HTTP server. Run: python -m proactive_eval report');
    return;
  }
  fetch('/api/delete/' + eid, {method: 'DELETE'})
    .then(function(r) { if (r.ok) location.reload(); else alert('Failed: ' + r.status); })
    .catch(function(e) { alert('Error: ' + e); });
}
"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Proactive Eval — Experiments</title>
<style>{_CSS}{index_css_extra}</style></head><body>
<h1>Proactive Eval</h1>
<p class="meta">Experiment dashboard. Select experiments to compare.</p>

<h2 style="margin-top:0">Experiments</h2>
{"".join(cards_html)}

<div class="compare-bar" id="compare-bar">
  <span id="compare-label" style="font-size:0.85em;color:var(--muted)">0 selected</span>
  <button id="compare-btn" onclick="doCompare()" disabled>Compare Selected</button>
</div>

<script>{index_js}</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Build all HTML
# ---------------------------------------------------------------------------

def build_all_html():
    """Generate index.html for each experiment + top-level index."""
    if not EXPERIMENTS_DIR.exists():
        return
    for d in EXPERIMENTS_DIR.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("compare_"):
            continue
        if not (d / "config.json").exists():
            continue
        exp = load_experiment(d, include_decisions=True)
        html = generate_experiment_html(exp)
        (d / "index.html").write_text(html)

    # Top-level index (no auto-compare generation)
    (EXPERIMENTS_DIR / "index.html").write_text(generate_index_html())


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _Handler(SimpleHTTPRequestHandler):
    """Serve from EXPERIMENTS_DIR; handle compare + delete APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(EXPERIMENTS_DIR), **kwargs)

    def do_GET(self):
        if "?refresh" in self.path:
            build_all_html()
            self.send_response(302)
            self.send_header("Location", self.path.split("?")[0])
            self.end_headers()
            return

        # Handle /compare?exp=X&exp=Y&exp=Z (N-way comparison)
        if self.path.startswith("/compare?"):
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            exp_ids = params.get("exp", [])
            if len(exp_ids) >= 2:
                experiments = []
                for eid in exp_ids:
                    exp = load_experiment(EXPERIMENTS_DIR / eid, include_decisions=True)
                    if exp:
                        experiments.append(exp)
                if len(experiments) >= 2:
                    html = generate_compare_html(experiments)
                    # Cache to disk so file:// works next time
                    cache_name = "compare_" + "_vs_".join(sorted(exp_ids))
                    cache_dir = EXPERIMENTS_DIR / cache_name
                    cache_dir.mkdir(exist_ok=True)
                    (cache_dir / "index.html").write_text(html)
                    # Redirect to the cached file
                    self.send_response(302)
                    self.send_header("Location", f"/{cache_name}/index.html")
                    self.end_headers()
                    return
            self.send_error(400, "Need at least 2 valid exp= params")
            return

        super().do_GET()

    def do_DELETE(self):
        import shutil
        prefix = "/api/delete/"
        if not self.path.startswith(prefix):
            self.send_error(404)
            return
        eid = self.path[len(prefix):].strip("/")
        if not eid or ".." in eid or "/" in eid:
            self.send_error(400, "Invalid experiment ID")
            return
        exp_dir = EXPERIMENTS_DIR / eid
        if not exp_dir.exists() or not exp_dir.is_dir():
            self.send_error(404, "Experiment not found")
            return
        shutil.rmtree(exp_dir)
        build_all_html()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"deleted": eid}).encode())

    def address_string(self):
        return self.client_address[0]

    def log_message(self, fmt, *a):
        pass


def serve(port: int, open_path: str = "/"):
    build_all_html()

    import socketserver

    class _ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = _ThreadedServer(("0.0.0.0", port), _Handler)
    url = f"http://localhost:{port}{open_path}"

    print(f"Serving experiments at {url}")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize proactive-eval experiment results")
    parser.add_argument("experiment", nargs="?", default=None,
                        help="Experiment ID (default: latest)")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments in terminal")
    parser.add_argument("--compare", nargs="+", metavar="ID",
                        help="Compare N experiment IDs")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP server port (default: {DEFAULT_PORT})")
    parser.add_argument("--no-serve", action="store_true",
                        help="Only generate HTML files, don't start server")
    args = parser.parse_args()

    if args.list:
        print_experiment_list()
        return

    if args.no_serve:
        build_all_html()
        print(f"HTML generated in {EXPERIMENTS_DIR}")
        return

    if args.experiment:
        exp_dir = EXPERIMENTS_DIR / args.experiment
        if not exp_dir.exists():
            print(f"Error: experiment not found: {args.experiment}")
            sys.exit(1)
        open_path = f"/{args.experiment}/index.html"
    else:
        exp_dir = latest_experiment_dir()
        if exp_dir:
            open_path = f"/{exp_dir.name}/index.html"
        else:
            open_path = "/"

    serve(args.port, open_path)


if __name__ == "__main__":
    main()
