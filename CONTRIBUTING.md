# Contributing to Proactive Eval

Thanks for your interest! This project is in early stages, so contributions of all kinds are welcome — bug fixes, new fixtures/scenarios, documentation improvements, or ideas for new ablation variables.

## Getting started

```bash
git clone https://github.com/phronesis-io/proactive-eval.git
cd proactive-eval
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests that require external API keys (e.g., embedding API) are skipped by default.

## Adding a new fixture or scenario

1. Create a fixture JSON in `fixtures/` following the existing naming convention (`N_name.json`)
2. Create a matching scenario in `scenarios/` with the same number prefix
3. Run `pytest tests/test_config.py` to validate the pairing

## Code style

- Python 3.10+
- No additional dependencies unless absolutely necessary — the project deliberately keeps a minimal dependency footprint
- Use type hints for function signatures

## Submitting changes

1. Fork the repo and create a feature branch
2. Make your changes with clear commit messages
3. Run `pytest tests/ -v` to ensure nothing breaks
4. Open a PR with a short description of what and why

## Security

Never commit API keys, tokens, or credentials. All secrets go in `.env` (gitignored). If you discover a leaked credential, please open an issue immediately.
