#!/usr/bin/env python3
"""
visualize_results.py — serve experiment results as an interactive dashboard.

Usage:
  python3 visualize_results.py                                # latest experiment → open in browser
  python3 visualize_results.py <experiment_id>                # specific experiment
  python3 visualize_results.py --compare <id1> <id2>          # compare two experiments
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


def _backfill_summary(summary: dict):
    """Ensure send_rate/action_rate and their CIs exist, computing from decisions dict if needed."""
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


def load_experiment(exp_dir: Path, include_decisions: bool = True) -> dict:
    """Load config.json and all summary files from an experiment directory.
    Set include_decisions=False to skip loading JSONL files (much faster)."""
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
            _backfill_summary(summary)
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
        cfg_path = d / "config.json"
        if cfg_path.exists():
            experiments.append(json.loads(cfg_path.read_text()))
    return experiments


def latest_experiment_dir() -> Path | None:
    if not EXPERIMENTS_DIR.exists():
        return None
    dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
    return dirs[-1] if dirs else None


def _esc(s: str) -> str:
    """HTML-escape a string."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


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
# HTML generation
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
.send-row { cursor: pointer; }
.send-row:hover { background: var(--header-bg); }
.detail { display: none; }
.detail td { white-space: normal; font-size: 0.9em; color: var(--muted); padding: 8px 14px; }
.tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.8em; font-weight: 600; }
.tag-send { background: #dbeafe; color: #1e40af; }
.tag-action { background: #fef3c7; color: #92400e; }
.tag-skip { background: #f3f4f6; color: #6b7280; }
.total-row td { font-weight: 700; border-top: 2px solid var(--border); }
.bar { display: inline-block; height: 14px; border-radius: 2px; vertical-align: middle; }
.bar-skip { background: #d1d5db; }
.bar-send { background: #60a5fa; }
.bar-action { background: #fbbf24; }
nav { margin-bottom: 1.5em; }
nav a { color: var(--accent); text-decoration: none; margin-right: 1em; font-size: 0.85em; }
nav a:hover { text-decoration: underline; }
.ci-chart { margin: 1.5em 0; }
.ci-row { display: flex; align-items: center; margin-bottom: 6px; font-size: 0.85em; }
.ci-label { width: 180px; flex-shrink: 0; text-align: right; padding-right: 12px; }
.ci-track { position: relative; flex: 1; height: 28px; background: var(--row-alt);
            border-radius: 3px; border: 1px solid var(--border); }
.ci-bar { position: absolute; top: 6px; height: 16px; border-radius: 2px; opacity: 0.7; }
.ci-dot { position: absolute; top: 4px; width: 4px; height: 20px; border-radius: 2px; }
.ci-val { position: absolute; top: 5px; font-size: 0.8em; font-weight: 600; white-space: nowrap; }
.ci-search { background: #60a5fa; }
.ci-eigenflux { background: #f97316; }
.ci-dot-search { background: #2563eb; }
.ci-dot-eigenflux { background: #c2410c; }
.ci-legend { font-size: 0.8em; color: var(--muted); margin-bottom: 1em; }
.ci-legend span { margin-right: 1.5em; }
.overlap-tag { font-size: 0.75em; padding: 1px 5px; border-radius: 3px; margin-left: 8px; }
.overlap-yes { background: #f3f4f6; color: #6b7280; }
.overlap-no { background: #fef2f2; color: #991b1b; }
/* Fixture cards */
.fixture-card { border: 1px solid var(--border); border-radius: 8px; margin: 1.2em 0;
                overflow: hidden; }
.fixture-header { background: var(--header-bg); padding: 8px 14px; font-weight: 600;
                  font-size: 0.95em; display: flex; justify-content: space-between; align-items: center; }
.fixture-body { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
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
  .overlap-no { background: #451a03; color: #fca5a5; }
  .overlap-yes { background: #374151; color: #9ca3af; }
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


def _ci_chart_html(fixtures: dict, labels: list[str],
                   rate_key: str = "send_rate", ci_key: str = "send_rate_ci95",
                   title: str = "Send Rate") -> str:
    """Build a horizontal CI comparison chart for a given metric."""
    if len(labels) < 2:
        return ""

    # Collect data: {fid: {label: (rate, ci_low, ci_high, n, pop)}}
    data = {}
    for fid in sorted(fixtures.keys()):
        runs = fixtures[fid]
        data[fid] = {}
        for l in labels:
            if l not in runs:
                continue
            s = runs[l]["summary"]
            sr = s.get(rate_key, 0) or 0
            ci = s.get(ci_key, [sr, sr]) or [sr, sr]
            n = s.get("sample_size", s["items_processed"])
            pop = s.get("population_size", n)
            data[fid][l] = (sr, ci[0], ci[1], n, pop)

    color_map = {}
    dot_map = {}
    for l in labels:
        if "search" in l:
            color_map[l] = "ci-search"
            dot_map[l] = "ci-dot-search"
        elif "eigenflux" in l:
            color_map[l] = "ci-eigenflux"
            dot_map[l] = "ci-dot-eigenflux"
        else:
            color_map[l] = "ci-search"
            dot_map[l] = "ci-dot-search"

    rows_html = []
    for fid in sorted(data.keys()):
        short = fid.replace("_v1", "")
        entries = data[fid]
        if len(entries) < 2:
            continue

        # Check CI overlap between first two labels
        vals = list(entries.values())
        overlaps = vals[0][2] >= vals[1][1] and vals[1][2] >= vals[0][1] if len(vals) >= 2 else True
        overlap_tag = ('<span class="overlap-tag overlap-yes">CI overlap</span>'
                       if overlaps else
                       '<span class="overlap-tag overlap-no">CI no overlap</span>')

        for l in labels:
            if l not in entries:
                continue
            sr, ci_lo, ci_hi, n, pop = entries[l]
            left_pct = ci_lo * 100
            width_pct = max(0.5, (ci_hi - ci_lo) * 100)
            dot_pct = sr * 100
            n_str = f"{n}/{pop}" if pop > n else str(n)

            rows_html.append(
                f'<div class="ci-row">'
                f'<div class="ci-label">{_esc(short)} / {_esc(l)}</div>'
                f'<div class="ci-track">'
                f'<div class="ci-bar {color_map.get(l, "ci-search")}" '
                f'style="left:{left_pct:.1f}%;width:{width_pct:.1f}%"></div>'
                f'<div class="ci-dot {dot_map.get(l, "ci-dot-search")}" '
                f'style="left:{dot_pct:.1f}%"></div>'
                f'<div class="ci-val" style="left:{max(dot_pct+1.5, ci_hi*100+1):.1f}%">'
                f'{sr*100:.0f}% (n={n_str})</div>'
                f'</div></div>'
            )
        rows_html.append(
            f'<div style="text-align:right;margin:-2px 0 10px 0">{overlap_tag}</div>'
        )

    if not rows_html:
        return ""

    legend_parts = []
    for l in labels:
        c = "#60a5fa" if "search" in l else "#f97316"
        legend_parts.append(f'<span><span class="bar" style="width:12px;background:{c}"></span> {_esc(l)}</span>')

    return (
        '<div class="ci-chart">'
        f'<div class="ci-legend">{"".join(legend_parts)}'
        ' &nbsp; | &nbsp; bar = 95% CI, dot = point estimate</div>'
        + "".join(rows_html)
        + '</div>'
    )


def generate_experiment_html(exp: dict) -> str:
    fixtures = exp.get("_fixtures", {})
    eid = exp.get("experiment_id", "?")

    all_labels = set()
    for runs in fixtures.values():
        all_labels.update(runs.keys())
    labels = sorted(all_labels)

    # --- Per-fixture comparison cards ---
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
        ar = s.get("action_rate", 0)
        aci = s.get("action_rate_ci95", [0, 0])

        bar = _bar_html(skip, send, send_action, width_px=140)

        return (
            f'<div class="fixture-col">'
            f'<h4>{_esc(label)}</h4>'
            f'<div class="stat-grid">'
            f'<span class="stat-label">Sample (n/N)</span><span class="stat-val">{n_str}</span>'
            f'<span class="stat-label">Send Rate</span>'
            f'<span class="stat-val highlight">{sr*100:.1f}%'
            f' <span style="font-weight:400;font-size:0.8em;color:var(--muted)">[{sci[0]*100:.1f}, {sci[1]*100:.1f}]</span></span>'
            f'<span class="stat-label">Action Rate</span>'
            f'<span class="stat-val highlight">{ar*100:.1f}%'
            f' <span style="font-weight:400;font-size:0.8em;color:var(--muted)">[{aci[0]*100:.1f}, {aci[1]*100:.1f}]</span></span>'
            f'<span class="stat-label">Skip / Send / Action</span>'
            f'<span class="stat-val">{skip} / {send} / {send_action}</span>'
            f'<span class="stat-label">Distribution</span><span>{bar}</span>'
            f'</div></div>'
        )

    def _delta_badge(val_a: float, ci_a: list, val_b: float, ci_b: list, metric: str) -> str:
        delta_pp = (val_a - val_b) * 100
        # Check CI overlap
        overlaps = ci_a[1] >= ci_b[0] and ci_b[1] >= ci_a[0]
        if abs(delta_pp) < 0.5:
            cls = "delta-neutral"
            sign = ""
        elif delta_pp > 0:
            cls = "delta-pos" if not overlaps else "delta-neutral"
            sign = "+"
        else:
            cls = "delta-neg" if not overlaps else "delta-neutral"
            sign = ""
        sig = "" if overlaps else " *"
        return (f'<span class="delta-badge {cls}">'
                f'{metric}: {sign}{delta_pp:.1f}pp{sig}</span>')

    fixture_cards = []
    for fid in sorted(fixtures.keys()):
        runs = fixtures[fid]
        if not runs:
            continue
        short = fid.replace("_v1", "")

        # Build header badges
        badges = ""
        if len(labels) == 2 and all(l in runs for l in labels):
            s0 = runs[labels[0]]["summary"]
            s1 = runs[labels[1]]["summary"]
            # Compare: first label vs second label
            badges += _delta_badge(
                s0.get("send_rate", 0), s0.get("send_rate_ci95", [0, 0]),
                s1.get("send_rate", 0), s1.get("send_rate_ci95", [0, 0]),
                "Send")
            badges += " "
            badges += _delta_badge(
                s0.get("action_rate", 0), s0.get("action_rate_ci95", [0, 0]),
                s1.get("action_rate", 0), s1.get("action_rate_ci95", [0, 0]),
                "Action")

        cols = ""
        for l in labels:
            if l in runs:
                cols += _col_html(runs[l]["summary"], l)
            else:
                cols += f'<div class="fixture-col"><h4>{_esc(l)}</h4><span class="stat-label">No data</span></div>'

        fixture_cards.append(
            f'<div class="fixture-card">'
            f'<div class="fixture-header"><span>{_esc(short)}</span><span>{badges}</span></div>'
            f'<div class="fixture-body">{cols}</div>'
            f'</div>'
        )

    # --- Sent items table ---
    send_rows = []
    for fid in sorted(fixtures.keys()):
        runs = fixtures[fid]
        for l in labels:
            if l not in runs:
                continue
            for d in runs[l]["decisions"]:
                action = d["decision"]["action"]
                if action not in ("send", "send_with_action"):
                    continue
                tag_cls = "tag-send" if action == "send" else "tag-action"
                tag_text = "send" if action == "send" else "send+action"
                preview = _esc(d["content_preview"][:120].replace("\n", " "))
                rationale = _esc(d["decision"].get("rationale", "")[:300])
                message = _esc((d["decision"].get("message") or "")[:300])
                rec = ""
                if action == "send_with_action":
                    ra = d["decision"].get("recommended_action") or {}
                    rec = f"<br><b>Action:</b> {_esc(ra.get('title',''))} — {_esc(ra.get('details','')[:200])}"

                send_rows.append(
                    f'<tr class="send-row"><td>{_esc(fid)}</td><td>{_esc(l)}</td>'
                    f'<td><span class="tag {tag_cls}">{tag_text}</span></td>'
                    f'<td>{preview}</td></tr>\n'
                    f'<tr class="detail"><td colspan="4">'
                    f'<b>Rationale:</b> {rationale}<br>'
                    f'<b>Message:</b> {message}{rec}</td></tr>'
                )

    # --- All experiments nav ---
    all_exps = list_experiments()
    nav_links = []
    for e in all_exps[:20]:
        e_id = e.get("experiment_id", "?")
        label = e_id
        if e.get("dry_run"):
            label += " [dry]"
        if e_id == eid:
            nav_links.append(f"<b>{_esc(label)}</b>")
        else:
            nav_links.append(f'<a href="/{e_id}/index.html">{_esc(label)}</a>')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Experiment {_esc(eid)}</title>
<style>{_CSS}</style></head><body>

<nav><a href="/index.html" style="font-weight:600">&larr; All Experiments</a>&nbsp;&nbsp;{"&nbsp;&middot;&nbsp;".join(nav_links)}</nav>

<h1>Experiment: {_esc(eid)}</h1>
<p class="meta">
  <span>Model: <b>{_esc(str(exp.get('model')))}</b></span>
  <span>Source: <b>{_esc(str(exp.get('source')))}</b></span>
  <span>Range: <b>{_esc(str(exp.get('start') or '(open)'))} &rarr; {_esc(str(exp.get('end') or '(open)'))}</b></span>
  <span>Sample: <b>{exp.get('sample') or 'all'}</b> (seed={exp.get('seed', '—')})</span>
  <span>Workers: <b>{exp.get('workers', '—')}</b></span>
  <span>Duration: <b>{_format_duration(exp.get('started_at'), exp.get('finished_at'))[0]}</b></span>
  {"<span><b>DRY RUN</b></span>" if exp.get('dry_run') else ""}
  {"<span style='color:var(--yellow)'><b>INCOMPLETE</b></span>" if not exp.get('finished_at') else ""}
</p>

<h2>Per-Fixture Comparison</h2>
<p style="font-size:0.8em;color:var(--muted)">
  <span class="bar bar-skip" style="width:12px"></span> skip &nbsp;
  <span class="bar bar-send" style="width:12px"></span> send &nbsp;
  <span class="bar bar-action" style="width:12px"></span> send+action &nbsp;
  | &nbsp; Delta badges: <span class="delta-badge delta-neutral">overlapping CI</span>
  <span class="delta-badge delta-pos">significant +</span>
  <span class="delta-badge delta-neg">significant &minus;</span>
  &nbsp; * = CIs do not overlap
</p>
{"".join(fixture_cards)}

<h2>Send Rate Overview (95% CI)</h2>
{_ci_chart_html(fixtures, labels, "send_rate", "send_rate_ci95")}

<h2>Action Rate Overview (95% CI)</h2>
{_ci_chart_html(fixtures, labels, "action_rate", "action_rate_ci95")}

<h2>Sent Items ({len(send_rows)})</h2>
<p style="font-size:0.8em;color:var(--muted)">Click a row to expand details.</p>
<table>
<tr><th>Fixture</th><th>Source</th><th>Action</th><th>Content Preview</th></tr>
{"".join(send_rows)}
</table>

<script>{_JS}</script>
</body></html>"""


def _format_duration(started: str | None, finished: str | None) -> tuple[str, bool]:
    """Return (duration_string, is_complete). If finished is missing, it's incomplete."""
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


def generate_index_html() -> str:
    """Generate a top-level index page listing all experiments."""
    all_exps = list_experiments()
    rows = []
    for e in all_exps:
        eid = e.get("experiment_id", "?")
        dry = " [dry]" if e.get("dry_run") else ""
        sample_str = str(e.get("sample")) if e.get("sample") else "all"
        duration, complete = _format_duration(e.get("started_at"), e.get("finished_at"))
        status = f'<span style="color:var(--green)">{duration}</span>' if complete \
            else f'<span style="color:var(--yellow)">{duration}</span>'
        del_btn = (f'<button onclick="deleteExp(\'{_esc(eid)}\')" '
                   f'style="font-size:0.75em;padding:1px 6px;cursor:pointer;'
                   f'border:1px solid var(--border);background:var(--bg);color:var(--red);'
                   f'border-radius:3px">delete</button>')
        rows.append(
            f'<tr><td><a href="/{_esc(eid)}/index.html">{_esc(eid)}{dry}</a></td>'
            f'<td>{_esc(str(e.get("model","?")))}</td>'
            f'<td>{_esc(str(e.get("source","?")))}</td>'
            f'<td>{_esc(str(e.get("start") or "?"))} &rarr; {_esc(str(e.get("end") or "?"))}</td>'
            f'<td>{sample_str}</td>'
            f'<td>{len(e.get("fixtures",[]))}</td>'
            f'<td>{status}</td>'
            f'<td>{del_btn}</td></tr>'
        )
    delete_js = """
function deleteExp(eid) {
  if (!confirm('Delete experiment ' + eid + '?')) return;
  fetch('/api/delete/' + eid, {method: 'DELETE'})
    .then(r => { if (r.ok) location.reload(); else alert('Failed: ' + r.status); })
    .catch(e => alert('Error: ' + e));
}
"""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Proactive Eval — Experiments</title>
<style>{_CSS}</style></head><body>
<h1>Proactive Eval — Experiments</h1>
<table>
<tr><th>Experiment</th><th>Model</th><th>Source</th><th>Time Range</th><th>Sample</th><th>Fixtures</th><th>Duration</th><th></th></tr>
{"".join(rows)}
</table>
<script>{delete_js}</script>
</body></html>"""


def build_all_html():
    """Generate index.html for each experiment + top-level index."""
    if not EXPERIMENTS_DIR.exists():
        return
    # Per-experiment pages
    for d in EXPERIMENTS_DIR.iterdir():
        if not d.is_dir():
            continue
        if not (d / "config.json").exists():
            continue
        exp = load_experiment(d, include_decisions=True)
        html = generate_experiment_html(exp)
        (d / "index.html").write_text(html)

    # Top-level index
    (EXPERIMENTS_DIR / "index.html").write_text(generate_index_html())


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _Handler(SimpleHTTPRequestHandler):
    """Serve from EXPERIMENTS_DIR; rebuild HTML on demand; handle DELETE API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(EXPERIMENTS_DIR), **kwargs)

    def do_GET(self):
        # Serve static files only. Use ?refresh to rebuild all HTML.
        if "?refresh" in self.path:
            build_all_html()
            self.send_response(302)
            self.send_header("Location", self.path.split("?")[0])
            self.end_headers()
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
    parser.add_argument("--compare", nargs=2, metavar="ID",
                        help="Compare two experiment IDs")
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

    # Determine which page to open
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
