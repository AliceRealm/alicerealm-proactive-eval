# Legacy Scripts

These standalone Python scripts were the original entry points before the
`proactive_eval` package and its CLI (`python -m proactive_eval`) were built.

They are kept here for reference and backward compatibility with existing tests.
For new usage, prefer the unified CLI:

```bash
python -m proactive_eval collect   --source search --once
python -m proactive_eval collect   --source eigenflux --once
python -m proactive_eval simulate  --config config/experiments/sample.json
python -m proactive_eval report    --experiment <id>
python -m proactive_eval run       --config ...
```

| Legacy script              | CLI equivalent                                      |
|----------------------------|-----------------------------------------------------|
| `collect_searches.py`      | `python -m proactive_eval collect --source search`   |
| `collect_eigenflux.py`     | `python -m proactive_eval collect --source eigenflux`|
| `simulate_decisions.py`    | `python -m proactive_eval simulate`                  |
| `visualize_results.py`     | `python -m proactive_eval report`                    |
| `dedup_embeddings.py`      | Used internally by the package (`proactive_eval.dedup`) |
