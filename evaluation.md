# Evaluation Method

## Terminology

- **Scenario** — the input definition: timestamped interaction history, event timeline, and heartbeat schedule. Lives in `scenarios/` as a JSON file.
- **Episode** — one complete execution of a scenario under a specific condition, producing a full trace. An episode = scenario + fixture + condition + resulting output.
- **Paired trial** — two episodes running the same scenario (control vs treatment).

---

## 1. Expected Output Log Structure

Each episode produces a timestamped trace. Below is the expected shape of a single episode's output.

### 1.1 Episode Header

```
scenario_id:   stock_watch_001
fixture_id:    careful_operator_v1
condition:     with_eigenflux | without_eigenflux
start_time:    2026-04-09T00:00:00Z
end_time:      2026-04-09T04:00:00Z
heartbeat_interval: 30min
total_cycles:  8
```

### 1.2 Prep-Stage Chat

The interaction history from the scenario file, reproduced verbatim. This is the shared context both conditions start from.

```
[00:05:00] user/Sam:       "Hey, could you help me keep an eye on a few stocks..."
[00:05:20] assistant/Sable: "Sure. Which ones are you most focused on..."
[00:06:10] user/Sam:       "Mainly NVDA, TSM, Tencent, and Xiaomi..."
[00:06:35] assistant/Sable: "Got it — I'll ignore routine noise..."
...
```

### 1.3 Heartbeat Cycles

Each cycle is a self-contained execution unit. A single cycle may contain multiple LLM calls, tool invocations, and internal state changes before producing (or choosing not to produce) a user-visible action.

```
── heartbeat_cycle 1 ─────────────────────────────────
trigger_time: 2026-04-09T00:30:00Z

[internal] LLM call 1 — planning: decide what to check
[tool_call] jina_search("NVDA news April 9 2026")          # control
           OR eigenflux_query(topics=["NVDA","TSM",...])    # treatment
[tool_result] [
    { title: "NVDA keynote announces Blackwell Ultra...", source: "reuters", ts: "00:22:00Z", relevance: high },
    { title: "TSM April revenue report in line...", source: "bloomberg", ts: "00:18:00Z", relevance: low }
  ]
[internal] LLM call 2 — evaluate: is anything here worth surfacing?
[memory_read] memory/2026-04-08.md → Sam holding NVDA long, nervous about valuation
[internal] decision: NVDA keynote is material → surface; TSM revenue routine → skip
[memory_write] memory/heartbeat-state.json → update lastChecks
[user_message] "NVDA just announced Blackwell Ultra at keynote — new architecture,
               shipping Q3. Could rerate the stock. You're long, so worth watching
               how the market digests it today. No action needed yet."

cycle_stats:
  llm_calls: 2
  tool_calls: 1
  tokens_used: 1840
  user_visible_action: yes (message)
  internal_action: yes (memory update)

── heartbeat_cycle 2 ─────────────────────────────────
trigger_time: 2026-04-09T01:00:00Z

[internal] LLM call 1 — planning
[tool_call] jina_search("TSM Tencent Xiaomi news") | eigenflux_query(...)
[tool_result] [ { title: "Xiaomi Q1 EV deliveries beat estimates", ... } ]
[internal] LLM call 2 — evaluate
[memory_read] memory/2026-04-07.md → Sam watching Xiaomi, waiting for earnings reaction
[internal] decision: relevant but not urgent — Sam said he wants to see earnings first.
           Hold. Do not surface.
[memory_write] memory/heartbeat-state.json → note Xiaomi EV signal, held

cycle_stats:
  llm_calls: 2
  tool_calls: 1
  tokens_used: 1520
  user_visible_action: no (held — signal noted internally)
  internal_action: yes (memory update)

── heartbeat_cycle 3–8 ──────────────────────────────
[same structure continues for each cycle]
```

### 1.4 Episode Summary

Appended at end of trace.

```
episode_summary:
  total_llm_calls: 14
  total_tool_calls: 8
  total_tokens: 12400
  messages_surfaced: 3
  messages_held: 5
  memory_updates: 6
  external_actions: 0
```

---

## 2. Reference Signal Set

Before scoring, define the **reference signal set** for the episode: the set of signals that a competent, well-informed agent could reasonably have discovered during the time window.

Each signal is annotated:

| Field | Description |
|-------|-------------|
| `signal_id` | Unique identifier |
| `timestamp` | When the signal became available |
| `description` | What happened |
| `relevance` | How relevant to the user's stated task (high / medium / low) |
| `urgency` | How time-sensitive (immediate / same-day / background) |
| `expected_action` | What a good agent should do (surface, hold, update memory, ignore) |
| `source` | Where discoverable (search-only, eigenflux-only, both) |

This set is constructed per-scenario and serves as the evaluation target. It is not ground truth — it is a best-effort reference.

---

## 3. Opportunity Annotation

Each scenario defines a set of **opportunities** — moments where proactive behavior is plausible.

An opportunity maps one or more reference signals to a heartbeat cycle window and an expected agent response:

```
opportunity_id: opp_001
cycle_window: [1, 2]              # which heartbeat cycles this falls within
signal_ids: [sig_001]
should_surface: true
expected_action_type: message     # message | memory_update | hold | tool_use
acceptable_timing: within 1 cycle of signal availability
persona_constraint: "Sam has high interruption threshold — only surface if clearly material"
```

Opportunities are the unit of behavioral evaluation. Episode-level scores are aggregated from opportunity-level judgments.

---

## 4. Scoring

### 4.1 Primary Metrics

Scored per-opportunity, then aggregated to episode level.

#### Coverage (did it find the signal?)

For each reference signal with `expected_action != ignore`:
- **1** — agent discovered and incorporated the signal
- **0.5** — agent found related information but missed the specific signal
- **0** — agent missed it entirely

Episode coverage = signals discovered / total scoreable signals.

#### Timeliness (did it act fast enough?)

For each opportunity where the agent acted:
- **1** — acted within the acceptable timing window
- **0.5** — acted late but still within the episode
- **0** — never acted, or acted so late the signal was stale

Only scored on opportunities where the agent *did* detect the signal (timeliness is conditional on coverage).

#### Action Quality (was the action appropriate?)

For each opportunity, score on two axes:

**Action-type appropriateness** — did the agent choose the right kind of response?
- **1** — correct action type (e.g., surfaced when it should surface, held when it should hold)
- **0.5** — partially correct (e.g., surfaced but as low-urgency when it was urgent)
- **0** — wrong action type (e.g., surfaced noise, or stayed silent on something urgent)

**Persona calibration** — did the agent respect the user's stated preferences and functional traits?
- **1** — action matches persona constraints (interruption threshold, tone, format)
- **0.5** — mostly right but tone/format/timing slightly off
- **0** — violated persona constraints (e.g., noisy alert to a high-interruption-threshold user)

Action quality = average of action-type appropriateness and persona calibration.

#### Accuracy (was the content correct and useful?)

For each surfaced message:
- **1** — factually correct, useful framing, actionable if appropriate
- **0.5** — mostly correct but missing context, or correct but not useful
- **0** — factually wrong, misleading, or fabricated

---

### 4.2 Secondary Metrics

#### Redundancy

Count of duplicate or near-duplicate signals surfaced across the episode. Penalize repeated information that adds no new value.

- **0** duplicates = no penalty
- Each duplicate = **-0.1** from aggregate score (capped at -0.5)

#### Token Cost

Total tokens consumed across the episode. Not scored directly — reported alongside primary metrics for cost-efficiency analysis. Flag runs where token usage is >2x the median for the same scenario across conditions.

---

### 4.3 Aggregate Score

```
Proactiveness Score =
    0.30 * Accuracy
  + 0.30 * Action Quality
  + 0.25 * Timeliness
  + 0.15 * Coverage
  - redundancy_penalty
```

Weights reflect the stated priority: usefulness and action quality first, then timeliness, then coverage. Redundancy penalizes but does not dominate.

Score range: 0.0 to 1.0 (before penalty).

---

## 5. Scoring Procedure

### Step 1 — Build reference signal set
Before running, define the reference signals and opportunities using the scenario's event timeline and domain knowledge.

### Step 2 — Run paired episodes
Run the same scenario under both conditions (control and treatment). Collect full traces per section 1.

### Step 3 — Annotate opportunities
For each opportunity in each run, record:
- Was the signal detected? (coverage)
- When did the agent act? (timeliness)
- What action did it take? (action quality)
- Was the content correct? (accuracy)
- Was anything repeated? (redundancy)

### Step 4 — Score
Compute per-opportunity scores, then aggregate to episode level using the formula above.

### Step 5 — Compare
The primary comparison is **treatment score minus control score** per episode. A positive delta indicates EigenFlux improved proactive behavior. Report both the aggregate delta and per-metric deltas to distinguish discovery advantage from decision advantage.

---

## 6. Distinguishing Discovery vs Decision Advantage

To avoid conflating "found more information" with "made better proactive decisions":

- **Discovery advantage** = treatment coverage minus control coverage. This measures raw signal access.
- **Decision advantage** = treatment (action quality + timeliness + accuracy) minus control (action quality + timeliness + accuracy), *on the subset of signals both conditions discovered*. This measures what the agent did with shared information.

Report both. EigenFlux may improve discovery, decision-making, or both — the benchmark should say which.
