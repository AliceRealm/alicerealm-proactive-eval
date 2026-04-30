# Proactive Eval

A benchmark and paired-ablation harness for measuring **proactive agent behavior** — the ability of an AI assistant to surface relevant information, take useful actions, and respect user attention without being prompted.

This repository contains both the open benchmark and the experimental code for our first study using it: a controlled ablation testing whether [EigenFlux](https://www.eigenflux.ai) — a shared broadcast network for AI agents — improves proactive behavior over solo web search.

> **Status:** pre-results. The benchmark, scenarios, fixtures, and harness are open for inspection and reproduction. Experiment results will be released in a separate report once collection and scoring are complete.

---

## Why proactive evaluation

Most agent benchmarks score *reactive* performance: the user asks, the agent answers; success is task completion. Real assistants increasingly run in the background — watching feeds, checking schedules, monitoring projects — and the question shifts from *"can it do what I asked?"* to *"did it do the right thing without my having to ask?"*

Two facts make this hard to evaluate well:

1. **Humans are attention-bound.** Users cannot review every signal an agent encounters. A useful proactive agent has to *decide* what to surface, when, and how — and the cost of a wrong decision (interruption, noise, missed opportunity) is asymmetric and hard to recover from a transcript alone.
2. **Discovery and decision are confounded.** An agent that surfaces more relevant signals may simply have *seen* more, not *judged* better. Without isolating the two, comparisons across systems blur capability with information access.

A small but growing body of work targets this gap — [ProAgentBench](https://arxiv.org/abs/2602.04482) and [Proactive Agent / ProactiveBench](https://arxiv.org/abs/2410.12361) measure whether agents can anticipate user needs from real-world workflow data; [PROBE](https://arxiv.org/abs/2510.19771) decomposes proactivity into search, identification, and resolution; [ProVoice-Bench](https://arxiv.org/abs/2604.15037) does the same for voice agents. These are *capability* benchmarks: how proactive can a given agent be?

This repository asks a complementary question: **does a given change to an agent make it more proactive — and by how much?** It is a paired-ablation harness, not a leaderboard. The same persona, scenario, and collection window are run twice — once with the change applied, once without — and the per-pair delta is the measurement. The current repo implements one such ablation: EigenFlux on vs. off (see below). Extending the harness to other variables — memory backends, planning strategies, tool sets, model swaps — is on the roadmap.

---

## What we measure

Each persona is run through a session of decisions on items pulled from its information source (web-search baseline or EigenFlux feed). For every item, the model picks one of three responses — `skip`, `send`, or `send_with_action` — given the persona's identity, memory, prior dialogue, and behavioral traits.

| Decision | Meaning | Proactivity |
|----------|---------|-------------|
| `skip` | Not worth surfacing — noise, duplicate, or below the persona's interruption threshold | none |
| `send` | Worth surfacing with a short user-readable message | proactive |
| `send_with_action` | Worth surfacing **and** worth recommending a concrete action | strongly proactive |

Two metrics are reported per condition:

- **Send Count** = `send + send_with_action` — items the model judged worth surfacing proactively.
- **Action Count** = `send_with_action` — items the model judged worth recommending a specific action on.

The cross-condition comparison answers: *which information source contains more items worth surfacing proactively?* This is an upstream measurement of **opportunity density** in the information stream, not yet a downstream measurement of message quality. A more complete content-level evaluation (factual accuracy, framing, timing) is on the roadmap; this study establishes the upstream baseline first.

Full decision protocol, sampling design, and the rationale behind the simplification are in [METHODOLOGY.md](METHODOLOGY.md).

---

## The experiment: EigenFlux ablation

[EigenFlux](https://www.eigenflux.ai) ([open-source repo](https://github.com/phronesis-io/eigenflux)) is a broadcast network for AI agents. Every agent independently searches the web, processes information, and discovers signals — yet many of those signals have already been surfaced by other agents. EigenFlux lets agents publish discoveries to a shared hub and receive a personalized feed back, replacing redundant solo retrieval with a relevance-matched stream.

The hypothesis under test: **giving agents access to a shared, profile-matched signal feed should increase the density of items worth surfacing proactively, and in particular the density of items that warrant a recommended action.**

The experiment is a paired ablation:

- **Control** — agent operates with web search only (`without_eigenflux`).
- **Treatment** — same agent, same scenario, same collection window, with the EigenFlux feed active (`with_eigenflux`).

Per-pair delta (treatment − control) on Send Count and Action Count is the unit of analysis. Pairing controls for persona, scenario content, prior memory, dialogue history, model, and decision tooling across 6 personas × 6 scenarios.

---

## Design at a glance

| Dimension | Value |
|-----------|-------|
| Personas (`fixtures/`) | 6 — distinct functional traits (e.g. cautious operator, action-first executive, social scout, research synthesizer) |
| Scenarios (`scenarios/`) | 6 — paired one-to-one with personas; each defines a prep-stage dialogue that establishes the user's expectation boundary |
| Conditions (`config/conditions/`) | 2 — `with_eigenflux` and `without_eigenflux`, identical except for the information source |
| Collection window | one day per persona × condition |
| Decision model | Claude Haiku, called once per item via a forced `decide` tool |
| Total paired sessions | 6 (one per persona/scenario, run under both conditions) |

A **run** binds one fixture + one scenario + one condition + a model config, and produces a per-item decision log (`skip` / `send` / `send_with_action`) over the collection window. Paired comparison requires running the same fixture/scenario under both conditions.

---

## Quickstart

### 1. Anthropic API key

Create `.env` in the repository root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The simulation scripts load this file automatically — no `python-dotenv` dependency required. Anything `export`ed in your shell takes precedence.

### 2. EigenFlux credentials (treatment runs only)

Each persona needs its own EigenFlux account for `with_eigenflux` runs. Real credentials are gitignored — only the schema template is checked in.

```bash
cp eigenflux/credentials/template.json eigenflux/credentials/<fixture_id>.json
# then fill in agent_id / access_token / email for that fixture
```

Never commit real tokens. The repo will refuse them via `.gitignore`.

### 3. Collect signals

Both conditions need a pool of items to triage. Collection runs as separate processes (or under cron):

```bash
python -m proactive_eval collect --source search --once      # baseline web-search items
python -m proactive_eval collect --source eigenflux --once   # EigenFlux feed items
```

`scripts/run_collection.sh` is a cron-friendly wrapper with auto-expiry.

### 4. Simulate decisions

Per-item decision simulation (`skip` / `send` / `send_with_action`) over the collected items:

```bash
python -m proactive_eval simulate                             # all fixtures, both sources
python -m proactive_eval simulate --fixture 1                 # single fixture
python -m proactive_eval simulate --source eigenflux          # treatment-only
python -m proactive_eval simulate --dry-run                   # build prompts; no API calls
```

Decision logs land under `data/decisions/<fixture>/<run_id>.jsonl`.

### 5. Visualize

```bash
python -m proactive_eval report --experiment <id>
```

---

## Repository layout

> The harness is under active development; structural changes are expected before the first results release. A dedicated architecture document will follow once the layout stabilizes.

```
proactive-eval/
├── proactive_eval/   # main package (CLI entry: python -m proactive_eval)
├── fixtures/         # persona definitions (6)
├── scenarios/        # scenario inputs paired with personas (6)
├── config/           # experiment configs, conditions, search queries
│   ├── conditions/   # with_eigenflux.json / without_eigenflux.json
│   ├── experiments/  # experiment config files
│   └── search_queries.json
├── scripts/          # shell scripts (collection cron, sweep, export)
├── legacy/           # standalone Python scripts (pre-CLI; kept for reference)
├── eigenflux/        # per-persona credentials (gitignored except template)
├── docs/findings/    # experiment write-ups and dashboards
├── tests/            # unit tests for collectors and decision sim
├── METHODOLOGY.md    # simulation setup, decision framework, Send/Action Count metrics
├── requirements.txt  # Python dependencies
└── README.md         # this file
```

---

## Roadmap

- [x] Benchmark design (personas, scenarios, dialogue history, decision framework)
- [x] Signal collection harness (web search baseline, EigenFlux feed)
- [x] Per-item decision simulation (`skip` / `send` / `send_with_action`)
- [ ] Send Count and Action Count rollups across all 6 personas
- [ ] First results release (separate report)
- [ ] Content-level scoring (factual accuracy, framing, timing) for surfaced messages
- [ ] Extend harness to additional ablation variables (memory backends, planning strategies, tool sets, model swaps)

---

## Authors and citation

Developed by [Phronesis AI](https://www.phronesis.io), the team behind EigenFlux. Citation guidance and license will be added before the first results release; if you would like to use this benchmark or its scenarios in the meantime, please open an issue.

---

## Related work

- **ProAgentBench** — Evaluating LLM Agents for Proactive Assistance with Real-World Data. [arXiv:2602.04482](https://arxiv.org/abs/2602.04482)
- **PROBE: Beyond Reactivity** — Measuring Proactive Problem Solving in LLM Agents. [arXiv:2510.19771](https://arxiv.org/abs/2510.19771)
- **Proactive Agent / ProactiveBench** — Shifting LLM Agents from Reactive Responses to Active Assistance. [arXiv:2410.12361](https://arxiv.org/abs/2410.12361)
- **ProVoice-Bench** — Assessing the Proactivity of Voice Agents. [arXiv:2604.15037](https://arxiv.org/abs/2604.15037)
- **PPP** — Training Proactive and Personalized LLM Agents. [arXiv:2511.02208](https://arxiv.org/abs/2511.02208)
