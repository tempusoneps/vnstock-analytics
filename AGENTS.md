# AGENTS.md

## Project

This repository contains analytics and reporting tools for VN30F1M market datasets.

Primary workflow:
- Source datasets live in `datasets/`.
- Notebook helpers live in `notebooks/VN30F1M/`.
- The CLI report pipeline lives in `src/auto_report/` and is launched through `src/auto_report.py`.

## Data Files

Canonical CSV inputs are under `datasets/`:

- `datasets/VN30F1M_5m.csv`
- `datasets/VN30F1M_5m_features.csv`
- `datasets/VN30F1M_5m_labels.csv`

The `datasets/*.csv` files are ignored by git. Do not add generated dataset CSVs to git unless explicitly requested.

Notebook-local CSVs under `notebooks/VN30F1M/` may exist for Jupyter/Docker use, but prefer the canonical files in `datasets/` for code paths and verification.

## Report Pipeline

Run the CLI from the repository root:

```bash
src/.venv/bin/python src/auto_report.py
```

The CLI loads parameters from `src/auto_report/config.json` by default. Shared dataset/output settings live at the top level, and module-specific params live under `modules.statistics` and `modules.xgboost`; running a module should only read that module's config section plus shared settings. Use another config with:

```bash
src/.venv/bin/python src/auto_report.py --config /path/to/config.json
```

Use `src/auto_report/config.sample.json` as the template for all supported params.

Useful quick verification command:

```bash
src/.venv/bin/python src/auto_report.py --config /tmp/auto_report_smoke_config.json
```

Expected report outputs include:

- `xgboost_report.html`
- `xgboost_report.md`
- `xgboost_dataset_summary.json`
- `xgboost_label_metrics.csv`
- `xgboost_feature_importance_by_label.csv`
- `xgboost_feature_importance_matrix.csv`
- `statistics_report.html`
- `statistics_report.md`
- `statistics_feature_column_statistics.csv`
- `statistics_label_column_statistics.csv`

## Python Verification

At minimum, run syntax checks for touched Python files:

```bash
python3 -m py_compile src/auto_report.py src/auto_report/*.py notebooks/VN30F1M/utils.py
```

For behavior changes in `src/auto_report/`, also run the smoke command above.

## Notebook Helpers

`notebooks/VN30F1M/utils.py` provides notebook-facing helpers, including `load_analytics_dataset()`.

Important behavior:
- It should prefer the canonical `datasets/` directory when available.
- It merges `VN30F1M_5m_features.csv` with `VN30F1M_5m_labels.csv` to create/read `VN30F1M_5m_ready.csv`.
- If notebook behavior looks stale, restart the Jupyter kernel or rerun the import cell before debugging code.

## Git Hygiene

The worktree may contain user edits or notebook-generated changes. Do not revert unrelated files.

Common noisy files:
- `notebooks/**/*.ipynb`
- `notebooks/**/.ipynb_checkpoints/*`
- `docker-compose.yml`
- generated CSV/report outputs

Before editing, check:

```bash
git status --short
```

Only modify files needed for the current request.

## Style

- Prefer small, focused changes.
- Reuse existing helpers in `src/auto_report/data.py`, `reporting.py`, `multilabel.py`, and `visualization.py` before adding new abstractions.
- Keep generated report data in CSV/JSON/HTML/Markdown outputs rather than hardcoding notebook-only logic.
- Use ASCII in source files unless the surrounding file already requires otherwise.
