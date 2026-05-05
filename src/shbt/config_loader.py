from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import yaml

from shbt.paths import ProjectPaths, resolve_resource_path


VALID_PARAMETER_CLASSIFICATIONS = frozenset({"Topological Necessity", "Empirical Matching Ansatz"})
DEFAULT_BENCHMARK_CONFIG_PATH = resolve_resource_path("config", "benchmark_v1.yaml")
DEFAULT_NUFIT_DATA_PATH = resolve_resource_path("data", "nufit_5_3.json")


class ConfigLoader:
    """Load benchmark configuration and provenance-tracked data sources."""

    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is not None:
            self.config_dir = Path(config_dir).expanduser().resolve()
            return
        self.config_dir = ProjectPaths.CONFIG.resolve()

    def _benchmark_config_path(self) -> Path:
        if self.config_dir == ProjectPaths.CONFIG.resolve():
            return DEFAULT_BENCHMARK_CONFIG_PATH
        return self.config_dir / DEFAULT_BENCHMARK_CONFIG_PATH.name

    def _resolve_config_path(self, config_path: str | Path) -> Path:
        path = Path(config_path)
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    @lru_cache(maxsize=128)
    def _load_yaml(self, filename: str | Path) -> dict[str, Any]:
        path = self._resolve_config_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Expected configuration file at {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Configuration file {path} must contain a top-level mapping")
        return data

    @lru_cache(maxsize=128)
    def _load_json(self, relative_path: str | Path) -> dict[str, Any]:
        path = self._resolve_relative_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Expected data file at {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise TypeError(f"Data file {path} must contain a top-level mapping")
        return data

    def _resolve_relative_path(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        config_relative_path = (self.config_dir / path).resolve()
        if config_relative_path.is_file():
            return config_relative_path
        repo_relative_path = (ProjectPaths.ROOT / path).resolve()
        if repo_relative_path.is_file():
            return repo_relative_path
        if path.name == DEFAULT_NUFIT_DATA_PATH.name:
            return DEFAULT_NUFIT_DATA_PATH
        return config_relative_path

    def _normalize_parameter_tree(
        self,
        node: Any,
        *,
        path: str = "",
        metadata: dict[str, str],
    ) -> Any:
        if isinstance(node, dict):
            if "value" in node:
                classification = node.get("classification")
                if classification not in VALID_PARAMETER_CLASSIFICATIONS:
                    raise TypeError(
                        f"Configuration parameter '{path}' must declare classification in {sorted(VALID_PARAMETER_CLASSIFICATIONS)}"
                    )
                metadata[path] = str(classification)
                return self._normalize_parameter_tree(node["value"], path=path, metadata=metadata)
            return {
                key: self._normalize_parameter_tree(
                    value,
                    path=f"{path}.{key}" if path else key,
                    metadata=metadata,
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [
                self._normalize_parameter_tree(value, path=f"{path}[{index}]", metadata=metadata)
                for index, value in enumerate(node)
            ]
        return node

    @lru_cache(maxsize=4)
    def _load_benchmark_bundle(self) -> tuple[dict[str, Any], dict[str, str]]:
        benchmark_path = self._benchmark_config_path()
        if not benchmark_path.is_file():
            legacy_physics = self._load_yaml("physics_constants.yaml")
            legacy_experimental = self._load_yaml("experimental_bounds.yaml")
            return ({**legacy_physics, "experimental": legacy_experimental}, {})
        raw_config = self._load_yaml(benchmark_path)
        metadata: dict[str, str] = {}
        normalized = self._normalize_parameter_tree(raw_config, metadata=metadata)
        if not isinstance(normalized, dict):
            raise TypeError("Benchmark configuration must contain a top-level mapping")
        return normalized, metadata

    def load_benchmark_config(self) -> dict[str, Any]:
        benchmark_config, _ = self._load_benchmark_bundle()
        return benchmark_config

    def load_parameter_classifications(self) -> dict[str, str]:
        _, metadata = self._load_benchmark_bundle()
        return dict(metadata)

    @lru_cache(maxsize=4)
    def load_physics_constants(self) -> dict[str, Any]:
        benchmark_path = self._benchmark_config_path()
        if not benchmark_path.is_file():
            return self._load_yaml("physics_constants.yaml")
        benchmark_config = self.load_benchmark_config()
        return {
            key: value
            for key, value in benchmark_config.items()
            if key != "experimental"
        }

    def load_experimental_bounds(self) -> dict[str, Any]:
        benchmark_path = self._benchmark_config_path()
        if not benchmark_path.is_file():
            return self._load_yaml("experimental_bounds.yaml")
        benchmark_config = self.load_benchmark_config()
        if "experimental" not in benchmark_config:
            return self._load_yaml("experimental_bounds.yaml")
        experimental = benchmark_config["experimental"]
        if not isinstance(experimental, dict):
            raise TypeError("Benchmark configuration section 'experimental' must be a mapping")
        releases = experimental.get("releases", {})
        data_sources = experimental.get("data_sources", {})
        if not isinstance(releases, dict) or not isinstance(data_sources, dict):
            raise TypeError("Benchmark configuration experimental releases/data_sources must be mappings")
        nufit_path = data_sources.get("nufit_normal_ordering", DEFAULT_NUFIT_DATA_PATH)
        if not isinstance(nufit_path, (str, Path)):
            raise TypeError("Benchmark configuration experimental.data_sources.nufit_normal_ordering must be a string path")
        nufit_dataset = self._load_json(nufit_path)
        return {
            "releases": {
                "nufit_release": str(releases.get("nufit_release", nufit_dataset.get("release", ""))),
                "nufit_reference": str(releases.get("nufit_reference", nufit_dataset.get("reference", ""))),
                "pdg_release": str(releases.get("pdg_release", "")),
                "pdg_reference": str(releases.get("pdg_reference", "")),
            },
            "lepton_1sigma": nufit_dataset["lepton_1sigma"],
            "quark_1sigma": experimental["quark_1sigma"],
            "ckm_gamma_experimental_input_deg": experimental["ckm_gamma_experimental_input_deg"],
            "normal_ordering_mass_splittings_ev2": nufit_dataset["normal_ordering_mass_splittings_ev2"],
            "nufit_3sigma_normal_ordering": nufit_dataset["nufit_3sigma_normal_ordering"],
        }


DEFAULT_CONFIG_LOADER = ConfigLoader()


__all__ = [
    "ConfigLoader",
    "DEFAULT_CONFIG_LOADER",
    "VALID_PARAMETER_CLASSIFICATIONS",
]
