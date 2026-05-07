from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import yaml

from shbt.paths import ProjectPaths, resolve_resource_path


COUNTER_UNIVERSAL_CLASSIFICATION = "Counter-Universal Scenario"
GEOMETRIC_EMERGENCE_CLASSIFICATION = "Geometric Emergence"
TOPOLOGICAL_EXTRACTION_CLASSIFICATION = "Topological Extraction"
VALID_PARAMETER_CLASSIFICATIONS = frozenset(
    {
        "Topological Necessity",
        "Empirical Matching Ansatz",
        COUNTER_UNIVERSAL_CLASSIFICATION,
        GEOMETRIC_EMERGENCE_CLASSIFICATION,
        TOPOLOGICAL_EXTRACTION_CLASSIFICATION,
    }
)
DEFAULT_BENCHMARK_CONFIG_PATH = resolve_resource_path("config", "benchmark_v1.yaml")
DEFAULT_COMPUTE_CLUSTER_RELATIVE_PATH = Path("compute") / "hpc_cluster.yaml"
DEFAULT_COMPUTE_CLUSTER_PATH = resolve_resource_path("config", str(DEFAULT_COMPUTE_CLUSTER_RELATIVE_PATH))
DEFAULT_NUFIT_DATA_PATH = resolve_resource_path("data", "nufit_5_3.json")
DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH = Path("physics_profiles") / "standard_model.yaml"
DEFAULT_PHYSICS_PROFILE_PATH = resolve_resource_path("config", str(DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH))
DEFAULT_UNIVERSAL_CONSTANTS_PATH = DEFAULT_PHYSICS_PROFILE_PATH


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


def _iter_leaf_paths(node: Any, *, path: str = "") -> tuple[str, ...]:
    if isinstance(node, dict):
        leaf_paths: list[str] = []
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            leaf_paths.extend(_iter_leaf_paths(value, path=child_path))
        return tuple(leaf_paths)
    if isinstance(node, list):
        leaf_paths = []
        for index, value in enumerate(node):
            child_path = f"{path}[{index}]"
            leaf_paths.extend(_iter_leaf_paths(value, path=child_path))
        return tuple(leaf_paths)
    return (path,) if path else ()


class ConfigLoader:
    """Load benchmark configuration and provenance-tracked data sources.

    Physics constants can be sourced from the repository's standard-model
    profile, an alternate on-disk profile, or directly injected mappings for
    counter-universal scenario testing.
    """

    def __init__(
        self,
        config_dir: Path | None = None,
        physics_profile_path: str | Path | None = None,
        universal_constants_path: str | Path | None = None,
        physics_profile: dict[str, Any] | None = None,
        physics_parameter_overrides: dict[str, Any] | None = None,
    ) -> None:
        if config_dir is not None:
            self.config_dir = Path(config_dir).expanduser().resolve()
        else:
            self.config_dir = ProjectPaths.CONFIG.resolve()
        resolved_profile_path = (
            physics_profile_path
            if physics_profile_path is not None
            else universal_constants_path
            if universal_constants_path is not None
            else DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH
        )
        self.physics_profile_path = Path(resolved_profile_path)
        self.universal_constants_path = self.physics_profile_path
        self.physics_profile = deepcopy(physics_profile) if physics_profile is not None else None
        self.physics_parameter_overrides = (
            deepcopy(physics_parameter_overrides) if physics_parameter_overrides is not None else {}
        )
        if self.physics_profile is not None and not isinstance(self.physics_profile, dict):
            raise TypeError("physics_profile must be a mapping when provided.")
        if not isinstance(self.physics_parameter_overrides, dict):
            raise TypeError("physics_parameter_overrides must be a mapping when provided.")

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
        return self._load_yaml_path(path)

    def _load_yaml_path(self, path: Path) -> dict[str, Any]:
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
        return self._load_yaml_path(path)

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

        physics_profile, physics_profile_metadata = self._load_physics_profile_bundle()
        if physics_profile:
            benchmark_config = _deep_merge(
                benchmark_config,
                self._translate_physics_profile(physics_profile),
            )
            metadata.update(self._translate_physics_profile_metadata(physics_profile, physics_profile_metadata))
        if self.physics_parameter_overrides:
            override_metadata: dict[str, str] = {}
            normalized_overrides = self._normalize_parameter_tree(
                deepcopy(self.physics_parameter_overrides),
                metadata=override_metadata,
            )
            if not isinstance(normalized_overrides, dict):
                raise TypeError("Injected physics_parameter_overrides must normalize to a mapping.")
            benchmark_config = _deep_merge(benchmark_config, normalized_overrides)
            for leaf_path in _iter_leaf_paths(normalized_overrides):
                override_metadata.setdefault(leaf_path, COUNTER_UNIVERSAL_CLASSIFICATION)
            metadata.update(override_metadata)
        benchmark_config, metadata = self._augment_with_geometric_emergence(benchmark_config, metadata)
        return benchmark_config, metadata

    @lru_cache(maxsize=4)
    def _load_physics_profile_bundle(self) -> tuple[dict[str, Any], dict[str, str]]:
        metadata: dict[str, str] = {}
        if self.physics_profile is not None:
            normalized = self._normalize_parameter_tree(deepcopy(self.physics_profile), metadata=metadata)
            if not isinstance(normalized, dict):
                raise TypeError("Injected physics_profile must contain a top-level mapping")
            for leaf_path in _iter_leaf_paths(normalized):
                metadata.setdefault(leaf_path, COUNTER_UNIVERSAL_CLASSIFICATION)
            return self._augment_profile_with_geometric_emergence(normalized, metadata)

        path = self._resolve_repo_or_config_path(self.physics_profile_path)
        if not path.is_file():
            return {}, {}
        raw_constants = self._load_yaml_from_repo_or_config(path)
        normalized = self._normalize_parameter_tree(raw_constants, metadata=metadata)
        if not isinstance(normalized, dict):
            raise TypeError("Physics profile file must contain a top-level mapping")
        return self._augment_profile_with_geometric_emergence(normalized, metadata)

    @lru_cache(maxsize=4)
    def _load_universal_constants_bundle(self) -> tuple[dict[str, Any], dict[str, str]]:
        return self._load_physics_profile_bundle()

    def _augment_profile_with_geometric_emergence(
        self,
        physics_profile: dict[str, Any],
        metadata: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not physics_profile or "tier_1" not in physics_profile:
            return physics_profile, metadata

        tier_1 = _require_mapping(physics_profile, "tier_1")
        tier_2 = physics_profile.setdefault("tier_2", {})
        if not isinstance(tier_2, dict):
            raise TypeError("Configuration section 'tier_2' must be a mapping when provided")

        from shbt.core.master_transport import build_geometry_origin_profile

        synthesized_values, synthesized_metadata = build_geometry_origin_profile(
            lepton_level=int(tier_1["lepton_level"]),
            quark_level=int(tier_1["quark_level"]),
            parent_level=int(tier_1["parent_level"]),
            generation_count=int(tier_1.get("g_sm", 15)),
        )

        for leaf_path, value in synthesized_values.items():
            if not leaf_path.startswith("physical_constants."):
                continue
            constant_name = leaf_path.removeprefix("physical_constants.")
            tier_2.setdefault(constant_name, value)
            metadata.setdefault(f"tier_2.{constant_name}", synthesized_metadata[leaf_path])
        metadata.setdefault("tier_3.geometric_kappa", synthesized_metadata["model.geometric_kappa"])
        return physics_profile, metadata

    def _augment_with_geometric_emergence(
        self,
        config: dict[str, Any],
        metadata: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        model = config.get("model")
        if not isinstance(model, dict):
            return config, metadata

        try:
            parent_level = int(model["parent_level"])
            lepton_fixed_point_index = int(model["lepton_fixed_point_index"])
            quark_fixed_point_index = int(model["quark_fixed_point_index"])
        except (KeyError, TypeError, ValueError):
            return config, metadata

        lepton_level = parent_level // (2 * lepton_fixed_point_index)
        quark_level = parent_level // (3 * quark_fixed_point_index)
        generation_count = int(model.get("g_sm", 15))

        from shbt.core.master_transport import build_geometry_origin_profile

        synthesized_values, synthesized_metadata = build_geometry_origin_profile(
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=parent_level,
            generation_count=generation_count,
        )

        model.setdefault("geometric_kappa", synthesized_values["model.geometric_kappa"])
        metadata.setdefault("model.geometric_kappa", synthesized_metadata["model.geometric_kappa"])

        physical_constants = config.get("physical_constants")
        if physical_constants is None:
            physical_constants = {}
            config["physical_constants"] = physical_constants
        if not isinstance(physical_constants, dict):
            raise TypeError("Benchmark configuration section 'physical_constants' must be a mapping")
        for leaf_path, value in synthesized_values.items():
            if not leaf_path.startswith("physical_constants."):
                continue
            constant_name = leaf_path.removeprefix("physical_constants.")
            physical_constants.setdefault(constant_name, value)
            metadata.setdefault(leaf_path, synthesized_metadata[leaf_path])
        return config, metadata

    def _translate_physics_profile(self, physics_profile: dict[str, Any]) -> dict[str, Any]:
        tier_1 = _require_mapping(physics_profile, "tier_1")
        tier_2 = physics_profile.get("tier_2", {})
        if not isinstance(tier_2, dict):
            raise TypeError("Configuration section 'tier_2' must be a mapping when provided")

        lepton_level = int(tier_1["lepton_level"])
        quark_level = int(tier_1["quark_level"])
        parent_level = int(tier_1["parent_level"])
        if lepton_level <= 0 or quark_level <= 0:
            raise ValueError("Tier 1 lepton_level and quark_level must be positive integers.")
        if parent_level % (2 * lepton_level) != 0:
            raise ValueError("Tier 1 constants must satisfy parent_level % (2 * lepton_level) == 0.")
        if parent_level % (3 * quark_level) != 0:
            raise ValueError("Tier 1 constants must satisfy parent_level % (3 * quark_level) == 0.")

        from shbt.core.master_transport import build_geometry_origin_profile

        synthesized_values, _ = build_geometry_origin_profile(
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=parent_level,
            generation_count=int(tier_1.get("g_sm", 15)),
        )
        physical_constants = {
            key.removeprefix("physical_constants."): value
            for key, value in synthesized_values.items()
            if key.startswith("physical_constants.")
        }
        physical_constants.update({key: value for key, value in tier_2.items()})
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
                "geometric_kappa": float(synthesized_values["model.geometric_kappa"]),
                "parent_level": parent_level,
                "lepton_fixed_point_index": parent_level // (2 * lepton_level),
                "quark_fixed_point_index": parent_level // (3 * quark_level),
                "g_sm": int(tier_1["g_sm"]),
            },
            "physical_constants": physical_constants,
        }

    def _translate_physics_profile_metadata(
        self,
        physics_profile: dict[str, Any],
        metadata: dict[str, str],
    ) -> dict[str, str]:
        if not physics_profile:
            return {}
        translated = {
            "model.geometric_kappa": metadata.get(
                "tier_3.geometric_kappa",
                GEOMETRIC_EMERGENCE_CLASSIFICATION,
            ),
            "model.parent_level": metadata.get("tier_1.parent_level", "Topological Necessity"),
            "model.lepton_fixed_point_index": metadata.get("tier_1.lepton_level", "Topological Necessity"),
            "model.quark_fixed_point_index": metadata.get("tier_1.quark_level", "Topological Necessity"),
            "model.g_sm": metadata.get("tier_1.g_sm", "Topological Necessity"),
        }
        tier_2 = physics_profile.get("tier_2", {})
        if not isinstance(tier_2, dict):
            raise TypeError("Configuration section 'tier_2' must be a mapping when provided")
        for name in tier_2:
            translated[f"physical_constants.{name}"] = metadata.get(
                f"tier_2.{name}",
                GEOMETRIC_EMERGENCE_CLASSIFICATION,
            )
        from shbt.core.master_transport import build_geometry_origin_profile

        synthesized_values, synthesized_metadata = build_geometry_origin_profile(
            lepton_level=int(_require_mapping(physics_profile, "tier_1")["lepton_level"]),
            quark_level=int(_require_mapping(physics_profile, "tier_1")["quark_level"]),
            parent_level=int(_require_mapping(physics_profile, "tier_1")["parent_level"]),
            generation_count=int(_require_mapping(physics_profile, "tier_1").get("g_sm", 15)),
        )
        for leaf_path in synthesized_values:
            translated.setdefault(leaf_path, synthesized_metadata[leaf_path])
        return translated

    def _translate_universal_constants(self, universal_constants: dict[str, Any]) -> dict[str, Any]:
        return self._translate_physics_profile(universal_constants)

    def _translate_universal_constant_metadata(
        self,
        universal_constants: dict[str, Any],
        metadata: dict[str, str],
    ) -> dict[str, str]:
        return self._translate_physics_profile_metadata(universal_constants, metadata)

    def load_benchmark_config(self) -> dict[str, Any]:
        benchmark_config, _ = self._load_benchmark_bundle()
        return benchmark_config

    def load_physics_profile(self) -> dict[str, Any]:
        physics_profile, _ = self._load_physics_profile_bundle()
        return physics_profile

    def load_universal_constants(self) -> dict[str, Any]:
        return self.load_physics_profile()

    def load_parameter_classifications(self) -> dict[str, str]:
        _, metadata = self._load_benchmark_bundle()
        return dict(metadata)

    def load_compute_cluster_config(
        self,
        relative_path: str | Path = DEFAULT_COMPUTE_CLUSTER_RELATIVE_PATH,
    ) -> dict[str, Any]:
        requested_path = Path(relative_path)
        if requested_path.is_absolute():
            return self._load_yaml_path(requested_path)

        candidate_paths = [
            (self.config_dir / requested_path).resolve(),
            (ProjectPaths.CONFIG / requested_path).resolve(),
        ]
        if requested_path == DEFAULT_COMPUTE_CLUSTER_RELATIVE_PATH:
            candidate_paths.insert(0, DEFAULT_COMPUTE_CLUSTER_PATH.resolve())

        unique_candidates: list[Path] = []
        for candidate in candidate_paths:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        for candidate in unique_candidates:
            if candidate.is_file():
                return self._load_yaml_path(candidate)

        raise FileNotFoundError(f"Expected configuration file at {unique_candidates[0]}")

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
    "COUNTER_UNIVERSAL_CLASSIFICATION",
    "ConfigLoader",
    "DEFAULT_BENCHMARK_CONFIG_PATH",
    "DEFAULT_COMPUTE_CLUSTER_PATH",
    "DEFAULT_COMPUTE_CLUSTER_RELATIVE_PATH",
    "DEFAULT_CONFIG_LOADER",
    "DEFAULT_NUFIT_DATA_PATH",
    "DEFAULT_PHYSICS_PROFILE_PATH",
    "DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH",
    "DEFAULT_UNIVERSAL_CONSTANTS_PATH",
    "GEOMETRIC_EMERGENCE_CLASSIFICATION",
    "VALID_PARAMETER_CLASSIFICATIONS",
]
