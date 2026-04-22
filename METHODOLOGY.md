# Methodology

This document specifies the experimental design, decision protocol, and metrics used by Proactive Eval. It is the reference companion to [README.md](README.md), which covers motivation, related work, and quickstart.

The harness is a **paired ablation**: a single persona is run twice over the same scenario, with one variable changed between runs, and the per-pair delta on a small set of decision-level metrics is the measurement of interest. Everything below — persona construction, scenario authoring, source pipelines, decision tool, sampling — exists in service of making that delta interpretable.

---

## The paired-ablation contract

For any pair of runs to be valid evidence about the variable under test, the following must be **identical** across the two runs:

- the persona (workspace files, memory files, functional traits)
- the scenario (prep-stage dialogue, user identity, time window)
- the decision model and the decision tool
- the sampling and aggregation procedure

Exactly **one** dimension is allowed to vary per pair. In the EigenFlux study, that dimension is the information source: DuckDuckGo search vs. the EigenFlux personalized feed. To extend the harness to a new ablation variable, define a new condition that holds everything above constant and changes only the dimension of interest.

This contract is the entire point of the framework. It is also where most of the design effort goes — small leaks (e.g. a profile sync running in only one condition) silently bias the delta.

---

## Personas

Six hand-authored personas span a deliberate spectrum on the proactivity axis — from high-threshold and conservative (Sable) to low-threshold and action-first (Flint, Nova) — across six task domains (finance, deal flow, networking, research, discourse, learning). Each persona's user is given a complicated, multi-day task and several days of prior memory, so decisions depend on user-specific context rather than item content alone.

Each persona is composed from three layers, all loaded into the system prompt at run time:

| Layer | Files | Purpose |
|-------|-------|---------|
| `workspace_files` | `SOUL.md`, `USER.md`, `HEARTBEAT.md`, `MEMORY.md` | *Who* the assistant is — personality, user profile, behavioral rules, preference summary |
| `memory_files` | `memory/YYYY-MM-DD.md` | *What* the assistant has already learned about the user from prior interaction |
| `functional_traits` | inline JSON | Quantified behavioral knobs the model conditions on |

`functional_traits` exposes three knobs and is injected verbatim into the system prompt:

```json
{
  "initiative": "low-medium",
  "risk_tolerance": "low",
  "interruption_threshold": "high"
}
```

Sable, with `interruption_threshold: high`, defaults to `skip` on ambiguous items. Flint, with `interruption_threshold: low`, defaults to `send`. The traits are not a separate scoring axis — they shape the decision distribution directly.

### Why memory files matter

Memory files are the persona-level personalization mechanism. They are hand-authored, dated entries simulating a stretch of interaction prior to the experiment. For Sable, this includes notes like:

> `memory/2026-04-08.md` — Sam sold half his TSM position last month on the Q4 dip; he is waiting for a re-entry around $150. NVDA is a long-term hold but he is uneasy about valuation.

Without these, decisions collapse into generic relevance judgments about each item. With them, the same news item ("TSM down 3% on supply-chain headline") gets a different answer for Sam than for an arbitrary user. The per-persona memory authoring is the single most labor-intensive part of the setup, and the most important for surfacing differential decision quality between conditions.

---

## Scenarios

Each persona is paired with one scenario. A scenario consists of (1) a user identity, (2) a short scripted dialogue establishing the user's expectation boundary, and (3) the topic frame for collection.

The dialogue is engineered, not naturalistic. Its job is to give the model the *contract* the user wants enforced — what to surface, what to skip, what format to use, how confident to be. Excerpt from Sam's exchange with Sable:

> **Sam:** Don't tell me about every small move, but if there's anything that genuinely affects the price or my decisions today, I want to know. ... If you flag something, I want a short note: what happened, why it matters, and what the reasonable next step is.
>
> **Sam:** Also — if you're not sure, say so. I'd rather have a careful judgment than false confidence.

The dialogue is injected verbatim into the system prompt. Scenarios differ in style as well as topic — Felix's dialogue opens with *"interrupt me only if it's something I can act on within the hour, otherwise just take notes"* and asks for two-line maxima. The persona × scenario combination is what creates the evaluation surface.

| Persona | Agent ID | Scenario | User | Core need |
|---------|----------|----------|------|-----------|
| Sable | `300840265495609344` | `stock_watch` | Sam | NVDA, TSM, Tencent, Xiaomi — only on material catalysts |
| Flint | `290804318221631488` | `deal_flow` | Felix | Japan distribution-deal negotiations |
| Ember | `290804322680176640` | `network_scan` | Elena | Early adopters and partnerships in AI agents |
| Mist | `290804327092584448` | `research_digest` | Mika | Key papers and vendor moves in on-device inference |
| Nova | `290804331492409344` | `discourse_watch` | Nora | Emerging narratives in behavioral science |
| Lark | `290804335900622848` | `learning_session` | Leo | Beginner-level CRISPR and gene therapy resources |

---

## Information sources

The control and treatment conditions differ only in where the items being decided on come from.

**Control (`without_eigenflux`)** — DuckDuckGo with 12 pre-defined queries per fixture. Queries are hand-authored against the persona's scenario and topic frame. For Sam, examples include:

```
"NVIDIA NVDA earnings report date Q2 2026"
"Xiaomi EV delivery numbers Q1 2026"
"US China semiconductor chip export controls 2026"
```

This represents an assistant with general search but no personalized feed.

**Treatment (`with_eigenflux`)** — each persona has its own [EigenFlux](https://www.eigenflux.ai) agent profile describing focus areas, recent work, and information needs. The platform pushes personalized content matched to the profile. The profile is auto-synced before each collection run so the feed is aligned to the persona's stated interests.

Both pipelines run on the same time window (one day per persona × condition) and produce a stream of items for the decision protocol below.

---

## Decision protocol

For every item from either pipeline, the decision model is invoked **once** with the persona's full system prompt (workspace files + memory files + functional traits + scenario dialogue) plus the item content. The model is forced to call a `decide` tool that returns one of three values:

| Decision | Definition |
|----------|------------|
| `skip` | Not worth surfacing now: noise, duplicate, irrelevant, or below the persona's interruption threshold. |
| `send` | Worth surfacing with a short user-readable message. |
| `send_with_action` | Worth surfacing **and** worth recommending a specific action. |

Two design choices to flag:

1. **Items are decided on individually**, not in batches. A given collection pull may contain many items; they are presented to the model one at a time, with a rolling context window of recent decisions for de-duplication. This avoids the model treating an entire pull as a single judgment unit.
2. **The decision tool is identical across conditions.** All cross-condition variation is upstream (in the items being decided on), not in the decision interface.

The current decision model is **Claude Haiku** (`claude-haiku-4-5-20251001`), chosen as a competent baseline that is consistent across all runs.

---

## Metrics

Two primary metrics are reported per condition per persona:

| Metric | Formula | What it counts |
|--------|---------|----------------|
| **Send Count** | `count(send) + count(send_with_action)` | Items judged worth surfacing proactively |
| **Action Count** | `count(send_with_action)` | Items judged worth recommending a specific action on |

The cross-condition delta on these counts is the experimental result.

---

## What Send Count actually measures

This is the most important interpretive caveat in the current design.

In a real deployment, the proactive-behavior quality of an agent has two layers: **(1)** does the agent decide to send, and **(2)** is the resulting send actually valuable to the user? Both layers can fail. An agent can push noise *and* miss signal.

This experiment intentionally simplifies the first layer. The model judging each item is the same competent baseline (Claude Haiku) across both conditions. The `send` and `send_with_action` counts reported here are therefore **items the model has already filtered as having proactive value** — not raw, unfiltered send volume.

The accurate reading of Send Count: *the number of items in a given information source that a competent model considers high proactive value.* The cross-condition comparison answers *"which information source contains more signal worth surfacing proactively?"* — an upstream measurement of **opportunity density** in the information stream.

The implicit assumption: an information source that yields more high-value surfacing opportunities will, in real deployment, produce more valuable proactive behavior. The next phase of the project tests this assumption by scoring the actual content the agent produces (see below).

---

## Roadmap to content-level scoring

The current metrics measure source-level opportunity density. They do not measure how well the agent uses those opportunities — whether the surfaced content is factually correct, well-framed, and well-timed. A complete proactivity evaluation needs both layers.

Planned for a later release:

- **Accuracy** — factual correctness of surfaced content, scored against the source.
- **Framing** — whether the message is concise, persona-appropriate, and useful.
- **Timing** — whether the surfacing happens within a useful window relative to when the underlying signal became available.

These will be added as additional per-item scores layered on top of the current decision protocol, not as a replacement for Send/Action Count.

---

## Reproducibility notes

- Decision model: `claude-haiku-4-5-20251001`. Changing the model is itself a valid ablation variable; if you do, do not compare deltas across model swaps.
- All persona files, scenario dialogues, query lists, and EigenFlux profile descriptions are checked into the repo and version-controlled. Real EigenFlux credentials are gitignored — see README.
