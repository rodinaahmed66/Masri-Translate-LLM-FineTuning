# AGENTS.md

Project: Masri-Translate — fine-tuning an LLM for Arabic↔English translation. Early stage: `pyproject.toml` and `params.yaml` are empty placeholders; there is no test suite, lint config, or CI yet.

## Structure

- `src/` — Python code. Only real file is `src/data.py` (HuggingFace `datasets`-based loading/filtering/splitting; expects dataset columns `Arabic` and `English`).
- `configs/` — reserved for training configs (currently empty).
- `models/lora-adapter/` — reserved for LoRA adapter output (currently empty).
- `params.yaml` — intended top-level params file (currently empty).

## Conventions

- Data filtering thresholds (test size, script-ratio 0.5, length ratio 0.3–3, max 600 chars, split seed 42) live as defaults in `src/data.py` — check there before assuming values in `params.yaml`.
- Current branch is `data-prep`; work so far is uncommitted data-prep scaffolding.

## Gotchas

- `pyproject.toml` declares no dependencies, but `src/data.py` requires `datasets`. Do not assume `pip install -e .` sets up the environment.
- No commands exist yet for lint/typecheck/test. If adding tooling, populate `pyproject.toml` rather than inventing scripts.
