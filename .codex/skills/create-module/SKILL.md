---
name: create-module
description: Create or extend modules for the vnstock auto_report architecture. Use when Codex needs to add a new report module under src/auto_report/modules, scaffold reports and writers, wire it into config.json/config.sample.json and cli.py, prefix module outputs, and verify the module with compile and smoke tests.
---

# Create Module

## Goal

Add a new `auto_report` module that follows the existing module architecture:

```text
src/auto_report/
  config.py
  cli.py
  modules/
    base/
      pipeline.py
      writers/
    <module>/
      pipeline.py
      reports/
      writers/
```

Use the existing `statistics` and `xgboost` modules as the source of truth for local conventions.

## Workflow

1. Inspect the current module layout before editing:

```bash
find src/auto_report/modules -maxdepth 4 -type f
sed -n '1,220p' src/auto_report/config.py
sed -n '1,180p' src/auto_report/cli.py
```

2. Create a module folder:

```text
src/auto_report/modules/<module>/
  __init__.py
  pipeline.py
  reports/
    __init__.py
  writers/
    __init__.py
    csv.py
    markdown.py
    html.py
```

Only create report files that are actually needed for the module.

3. Implement `<ModuleName>Pipeline` in `pipeline.py`.

Required shape:

```python
class MyModulePipeline:
    name = "my_module"

    def run(self, context: PipelineContext) -> PipelineResult:
        ...
        return PipelineResult(name=self.name, output_files=output_files, summary=summary)
```

Use `context.config` for module settings, `context.bundle` for loaded data, and `context.output_dir` for outputs.

4. Prefix every output file with the module name.

Good:

```text
my_module_report.html
my_module_report.md
my_module_metrics.csv
my_module_summary.json
```

Avoid generic names such as:

```text
report.html
metrics.csv
dataset_summary.json
```

5. Put common dataset/output config at the top level of `config.json` and `config.sample.json`.

Shared top-level fields:

```json
{
  "module": "all",
  "data_dir": "datasets",
  "output_dir": "reports",
  "raw_file": "VN30F1M_5m.csv",
  "features_file": "VN30F1M_5m_features.csv",
  "labels_file": "VN30F1M_5m_labels.csv",
  "modules": {}
}
```

Only module-specific parameters go under `modules.<module>`.

6. Wire the module into `cli.py`.

Add the pipeline import and registration:

```python
from .modules.my_module.pipeline import MyModulePipeline

pipeline_by_module = {
    "statistics": StatisticsPipeline(),
    "xgboost": XGBoostPipeline(),
    "my_module": MyModulePipeline(),
}
```

7. Update `MODULE_NAMES` in `config.py` so the module can be selected.

8. Update `AGENTS.md` if commands, config shape, or expected outputs change.

## Data And Config Rules

- Load raw/features/labels through existing `data.py` helpers whenever possible.
- Do not add module-specific copies of `data_dir`, `output_dir`, or dataset filenames unless the user explicitly needs an override.
- If module-specific overrides are needed, preserve the top-level default merge behavior in `ReportConfigFile.module_config()`.
- Keep config names snake_case and JSON scalar/list/object only.

## Writer Rules

- Put module-specific CSV/Markdown/HTML output code under `modules/<module>/writers/`.
- Reuse base helpers from `modules/base/writers/` for simple file writing.
- Keep filenames stable and prefixed.
- Keep Markdown/HTML file lists in sync with actual generated outputs.

## Validation

At minimum, run:

```bash
python3 -m py_compile src/auto_report.py src/auto_report/*.py src/auto_report/modules/base/*.py src/auto_report/modules/base/writers/*.py src/auto_report/modules/*/*.py src/auto_report/modules/*/reports/*.py src/auto_report/modules/*/writers/*.py
python3 -m json.tool src/auto_report/config.json
python3 -m json.tool src/auto_report/config.sample.json
```

Then run a smoke config in `/tmp` for the new module. Keep it small when the module trains models or performs expensive work.

Example pattern:

```bash
src/.venv/bin/python src/auto_report.py --config /tmp/auto_report_<module>_smoke.json
```

Verify that generated files use the module prefix and that no generic output names are introduced.
