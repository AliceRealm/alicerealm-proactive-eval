# Proactive Evaluation Benchmark

Evaluates whether enabling EigenFlux improves proactive agent behavior in a controlled, paired, multi-heartbeat setting.

## Structure

```
proactive-eval/
├── fixtures/                        # Agent personas (who the agent is)
│   ├── 1_careful_operator.json      # Sable — cautious, low-noise (stocks)
│   ├── 2_executive_finisher.json    # Flint — action-first, direct (deal flow)
│   ├── 3_social_scout.json          # Ember — opportunity-spotting, people-aware (network)
│   ├── 4_research_librarian.json    # Mist — analytical, synthesis-first (on-device AI)
│   ├── 5_content_strategist.json    # Nova — discourse-tracking, opinionated (psychology)
│   └── 6_learning_curator.json      # Lark — patient, level-matched (biology)
├── scenarios/                       # Scenario inputs (prep-stage chat + empty events)
│   ├── 1_stock.json                 # Sam monitors AI-related stocks
│   ├── 2_deal_flow.json             # Felix tracks East Asia deal signals
│   ├── 3_network_scan.json          # Elena finds devtools beta early adopters
│   ├── 4_research.json              # Mika synthesizes on-device inference research
│   ├── 5_discourse.json             # Nora watches behavioral science discourse
│   └── 6_learning.json              # Leo studies CRISPR and gene therapy basics
├── conditions/                      # Experimental conditions (what differs)
│   ├── with_eigenflux.json
│   └── without_eigenflux.json
└── runs/                            # Run configs (binds fixture + scenario + condition)
```

## How It Works

Each experiment is a **paired comparison**: same agent, same scenario, same timeline — only difference is EigenFlux on vs off.

### 4 Input Files Per Run

1. **`agent_fixture.json`** — agent persona, memory, heartbeat rules
2. **`scenario.json`** — timestamped history, external signals, heartbeat schedule
3. **`condition.json`** — EigenFlux enabled/disabled + runtime file state
4. **`run.json`** — binds the above three + model/runtime config

### Scoring Dimensions

| Metric | Description |
|--------|-------------|
| Coverage | Did the agent discover relevant signals? |
| Timeliness | Did it act quickly enough to be useful? |
| Action Quality | Was the chosen action appropriate? |
| Accuracy | Was the information correct and useful? |

Secondary: redundancy, token cost.

### Recommended Episode Config

- Scenario length: 4 hours
- Heartbeat interval: 30 minutes
- Total cycles: 8

## Source

Derived from Phronesis internal docs: "Proactive 讨论" knowledge base.
