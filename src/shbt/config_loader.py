from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import yaml

from shbt.paths import ProjectPaths, resolve_resource_path


VALID_PARAMETER_CLASSIFICATIONS = frozenset({"Topological Necessity", "Empirical Matching Ansatz"})
DEFAULT_BENCHMARK_CONFIG_PATH = resolve_resource_path("config", "benchmark_v1.yaml")
DEFAULT_NUFIT_DATA_PATH = resolve_resource_path("data", "nufit_5_3.json")
DEFAULT_UNIVERSAL_CONSTANTS_PATH = resolve_resource_path("data", "universal_constants.yaml")


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return value


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
            continue
        merged[key] = value
    return merged


class ConfigLoader:
    """Load benchmark configuration and provenance-tracked data sources."""

    def __init__(
        self,
        config_dir: Path | None = None,
        universal_constants_path: str | Path | None = None,
    ) -> None:
        if config_dir is not None:
            self.config_dir = Path(config_dir).expanduser().resolve()
        else:
            self.config_dir = ProjectPaths.CONFIG.resolve()
        self.universal_constants_path = (
            Path(universal_constants_path)
            if universal_constants_path is not None
            else DEFAULT_UNIVERSAL_CONSTANTS_PATH
        )

    def _benchmark_config_path(self) -> Path:
        if self.config_dir == ProjectPaths.CONFIG.resolve():
            return DEFAULT_BENCHMARK_CONFIG_PATH
        return self.config_dir / DEFAULT_BENCHMARK_CONFIG_PATH.name

    def _resolve_config_path(self, config_path: str | Path) -> Path:
        path = Path(config_path)
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    def _resolve_repo_or_config_path(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else self._resolve_relative_path(path)

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

    @lru_cache(maxsize=8)
    def _load_yaml_from_repo_or_config(self, filename: str | Path) -> dict[str, Any]:
        path = self._resolve_repo_or_config_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Expected configuration file at {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Configuration file {path} must contain a top-level mapping")
        return data

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
            benchmark_config: dict[str, Any] = {**legacy_physics, "experimental": legacy_experimental}
            metadata: dict[str, str] = {}
        else:
            raw_config = self._load_yaml(benchmark_path)
            metadata = {}
            normalized = self._normalize_parameter_tree(raw_config, metadata=metadata)
            if not isinstance(normalized, dict):
                raise TypeError("Benchmark configuration must contain a top-level mapping")
            benchmark_config = normalized

        universal_constants, universal_metadata = self._load_universal_constants_bundle()
        if universal_constants:
            benchmark_config = _deep_merge(
                benchmark_config,
                self._translate_universal_constants(universal_constants),
            )
            metadata.update(self._translate_universal_constant_metadata(universal_constants, universal_metadata))
        return benchmark_config, metadata

    @lru_cache(maxsize=4)
    def _load_universal_constants_bundle(self) -> tuple[dict[str, Any], dict[str, str]]:
        path = self._resolve_repo_or_config_path(self.universal_constants_path)
        if not path.is_file():
            return {}, {}
        raw_constants = self._load_yaml_from_repo_or_config(path)
        metadata: dict[str, str] = {}
        normalized = self._normalize_parameter_tree(raw_constants, metadata=metadata)
        if not isinstance(normalized, dict):
            raise TypeError("Universal constants file must contain a top-level mapping")
        return normalized, metadata

    def _translate_universal_constants(self, universal_constants: dict[str, Any]) -> dict[str, Any]:
        tier_1 = _require_mapping(universal_constants, "tier_1")
        tier_2 = _require_mapping(universal_constants, "tier_2")

        lepton_level = int(tier_1["lepton_level"])
        quark_level = int(tier_1["quark_level"])
        parent_level = int(tier_1["parent_level"])
        if lepton_level <= 0 or quark_level <= 0:
            raise ValueError("Tier 1 lepton_level and quark_level must be positive integers.")
        if parent_level % (2 * lepton_level) != 0:
            raise ValueError("Tier 1 constants must satisfy parent_level % (2 * lepton_level) == 0.")
        if parent_level % (3 * quark_level) != 0:
            raise ValueError("Tier 1 constants must satisfy parent_level % (3 * quark_level) == 0.")

        physical_constants = {key: value for key, value in tier_2.items()}
        if "planck2018_lambda_si_m2" not in physical_constants:
            hubble_si = float(physical_constants["planck2018_h0_km_s_mpc"]) * 1.0e3 / float(physical_constants["mpc_in_meters"])
            light_speed = float(physical_constants["light_speed_m_per_s"])
            physical_constants["planck2018_lambda_si_m2"] = (
                3.0 * float(physical_constants["planck2018_omega_lambda"]) * hubble_si * hubble_si / (light_speed * light_speed)
            )
        if "planck2018_lambda_fractional_sigma" not in physical_constants:
            physical_constants["planck2018_lambda_fractional_sigma"] = math.sqrt(
                (float(physical_constants["planck2018_omega_lambda_sigma"]) / float(physical_constants["planck2018_omega_lambda"])) ** 2
                + (2.0 * float(physical_constants["planck2018_h0_sigma_km_s_mpc"]) / float(physical_constants["planck2018_h0_km_s_mpc"])) ** 2
            )

        return {
            "model": {
                "parent_level": parent_level,
                "lepton_fixed_point_index": parent_level // (2 * lepton_level),
                "quark_fixed_point_index": parent_level // (3 * quark_level),
                "g_sm": int(tier_1["g_sm"]),
            },
            "physical_constants": physical_constants,
        }

    def _translate_universal_constant_metadata(
        self,
        universal_constants: dict[str, Any],
        metadata: dict[str, str],
    ) -> dict[str, str]:
        if not universal_constants:
            return {}
        translated = {
            "model.parent_level": metadata.get("tier_1.parent_level", "Topological Necessity"),
            "model.lepton_fixed_point_index": metadata.get("tier_1.lepton_level", "Topological Necessity"),
            "model.quark_fixed_point_index": metadata.get("tier_1.quark_level", "Topological Necessity"),
            "model.g_sm": metadata.get("tier_1.g_sm", "Topological Necessity"),
        }
        tier_2 = _require_mapping(universal_constants, "tier_2")
        for name in tier_2:
            translated[f"physical_constants.{name}"] = metadata.get(
                f"tier_2.{name}",
                "Empirical Matching Ansatz",
            )
        translated.setdefault(
            "physical_constants.planck2018_lambda_si_m2",
            metadata.get("tier_2.planck2018_lambda_si_m2", "Empirical Matching Ansatz"),
        )
        translated.setdefault(
            "physical_constants.planck2018_lambda_fractional_sigma",
            metadata.get("tier_2.planck2018_lambda_fractional_sigma", "Empirical Matching Ansatz"),
        )
        return translated

    def load_benchmark_config(self) -> dict[str, Any]:
        benchmark_config, _ = self._load_benchmark_bundle()
        return benchmark_config

    def load_universal_constants(self) -> dict[str, Any]:
        universal_constants, _ = self._load_universal_constants_bundle()
        return universal_constants

    def load_parameter_classifications(self) -> dict[str, str]:
        _, metadata = self._load_benchmark_bundle()
        return dict(metadata)

    @lru_cache(maxsize=4)
    def load_physics_constants(self) -> dict[str, Any]:
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
    "DEFAULT_BENCHMARK_CONFIG_PATH",
    "DEFAULT_CONFIG_LOADER",
    "DEFAULT_NUFIT_DATA_PATH",
    "DEFAULT_UNIVERSAL_CONSTANTS_PATH",
    "VALID_PARAMETER_CLASSIFICATIONS",
]
