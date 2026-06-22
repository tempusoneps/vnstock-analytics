"""Configuration objects for auto report pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
MODULE_NAMES = {"all", "statistics", "xgboost"}
COMMON_CONFIG_KEYS = {
    "data_dir",
    "output_dir",
    "raw_file",
    "features_file",
    "labels_file",
}


@dataclass
class AutoReportConfig:
    module: str
    data_dir: Path
    output_dir: Path
    raw_file: str
    features_file: str
    labels_file: str
    targets: str
    target: str | None
    test_size: float
    val_size: float
    random_state: int
    max_rows: int
    min_rows_per_target: int
    include_leakage: bool
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    gpu: bool
    top_n: int

    @classmethod
    def from_json(cls, path: Path, module: str | None = None) -> "AutoReportConfig":
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {path}")
        data = json.loads(path.read_text())
        module_name = module or str(data.get("module", "all"))
        if module_name == "all":
            raise ValueError("AutoReportConfig.from_json requires a concrete module, not 'all'.")
        return cls.from_module_dict(module_name, data)

    @classmethod
    def from_module_dict(cls, module: str, data: dict[str, Any]) -> "AutoReportConfig":
        modules = data.get("modules", {})
        if not isinstance(modules, dict):
            raise ValueError("Config field 'modules' must be an object.")
        module_data = modules.get(module)
        if not isinstance(module_data, dict):
            raise ValueError(f"Missing config for module: {module}")
        common_data = {key: data[key] for key in COMMON_CONFIG_KEYS if key in data}
        return cls.from_dict(module, {**common_data, **module_data})

    @classmethod
    def from_dict(cls, module: str, data: dict[str, Any]) -> "AutoReportConfig":
        return cls(
            module=module,
            data_dir=Path(data.get("data_dir", "datasets")),
            output_dir=Path(data.get("output_dir", "notebooks/VN30F1M/predictability_reports")),
            raw_file=str(data.get("raw_file", "VN30F1M_5m.csv")),
            features_file=str(data.get("features_file", "VN30F1M_5m_features.csv")),
            labels_file=str(data.get("labels_file", "VN30F1M_5m_labels.csv")),
            targets=str(data.get("targets", "all")),
            target=data.get("target"),
            test_size=float(data.get("test_size", 0.15)),
            val_size=float(data.get("val_size", 0.15)),
            random_state=int(data.get("random_state", 42)),
            max_rows=int(data.get("max_rows", 0)),
            min_rows_per_target=int(data.get("min_rows_per_target", 500)),
            include_leakage=bool(data.get("include_leakage", False)),
            xgb_n_estimators=int(data.get("xgb_n_estimators", 200)),
            xgb_max_depth=int(data.get("xgb_max_depth", 4)),
            xgb_learning_rate=float(data.get("xgb_learning_rate", 0.05)),
            gpu=bool(data.get("gpu", False)),
            top_n=int(data.get("top_n", 30)),
        )

    @classmethod
    def from_namespace(cls, args: Any) -> "AutoReportConfig":
        return cls(
            module=getattr(args, "module", "xgboost"),
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            raw_file=args.raw_file,
            features_file=args.features_file,
            labels_file=args.labels_file,
            targets=args.targets,
            target=args.target,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            max_rows=args.max_rows,
            min_rows_per_target=args.min_rows_per_target,
            include_leakage=args.include_leakage,
            xgb_n_estimators=args.xgb_n_estimators,
            xgb_max_depth=args.xgb_max_depth,
            xgb_learning_rate=args.xgb_learning_rate,
            gpu=args.gpu,
            top_n=args.top_n,
        )


@dataclass
class ReportConfigFile:
    selected_module: str
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "ReportConfigFile":
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {path}")
        raw = json.loads(path.read_text())
        selected_module = str(raw.get("module", "all"))
        if selected_module not in MODULE_NAMES:
            raise ValueError("Config field 'module' must be one of: all, statistics, xgboost")
        return cls(selected_module=selected_module, raw=raw)

    def module_names(self) -> list[str]:
        if self.selected_module == "all":
            return ["statistics", "xgboost"]
        return [self.selected_module]

    def module_config(self, module: str) -> AutoReportConfig:
        return AutoReportConfig.from_module_dict(module, self.raw)
