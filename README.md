# Proactive Evaluation Benchmark

Evaluates whether enabling EigenFlux improves proactive agent behavior in a controlled, paired, multi-heartbeat setting.

## Structure

```
proactive-eval/
├── fixtures/                  # Agent personas (who the agent is)
│   ├── careful_operator.json  # Sable — cautious, low-noise
│   └── executive_finisher.json # Flint — action-first, direct
├── episodes/                  # Evaluation scenarios (what happens)
├── conditions/                # Experimental conditions (what differs)
│   ├── with_eigenflux.json
│   └── without_eigenflux.json
└── runs/                      # Run configs (binds fixture + episode + condition)
```

## How It Works

Each experiment is a **paired comparison**: same agent, same scenario, same timeline — only difference is EigenFlux on vs off.

### 4 Input Files Per Run

1. **`agent_fixture.json`** — agent persona, memory, heartbeat rules
2. **`episode.json`** — timestamped history, external signals, heartbeat schedule
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

- Episode length: 4 hours
- Heartbeat interval: 30 minutes
- Total cycles: 8

## Source

Derived from Phronesis internal docs: "Proactive 讨论" knowledge base.
