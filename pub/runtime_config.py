from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config_loader import DEFAULT_CONFIG_LOADER


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return value


def _coerce_float(config: dict[str, Any], key: str) -> float:
    return float(config[key])


def _coerce_optional_float(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(default if value is None else value)


def _coerce_int(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if isinstance(value, bool):
        raise TypeError(f"Configuration value '{key}' must be an integer, not a boolean")
    return int(value)


def _coerce_optional_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise TypeError(f"Configuration value '{key}' must be an integer, not a boolean")
    return int(default if value is None else value)


def _coerce_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise TypeError(f"Configuration value '{key}' must be interpretable as a boolean")


def _coerce_str_sequence(config: dict[str, Any], key: str) -> tuple[str, ...]:
    value = config.get(key, ())
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Configuration value '{key}' must be a string or sequence of strings")
    return tuple(str(item) for item in value)


_PHYSICS_CONFIG = DEFAULT_CONFIG_LOADER.load_physics_constants()
_NUMERICAL_GUARD_CONFIG = _require_mapping(_PHYSICS_CONFIG, "numerical_guard")
_SOLVER_CONFIG = _require_mapping(_PHYSICS_CONFIG, "solver")


class Sector(str, Enum):
    LEPTON = "lepton"
    QUARK = "quark"

    @classmethod
    def coerce(cls, value: "Sector | str") -> "Sector":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported sector: {value}") from exc


class PhysicsDomainWarning(RuntimeWarning):
    """Coordinate-singularity warning for flavor-transport observables."""


class PhysicalSingularityException(RuntimeError):
    """Raised when a boundary transition hits a physically singular branch."""


class PerturbativeBreakdownException(RuntimeError):
    """Raised when a boundary kernel exceeds the declared perturbative condition limit."""


class SolverException(RuntimeError):
    """Raised when the verifier is not pinned to strict Radau IIA convergence."""


@dataclass(frozen=True)
class NumericalStabilityGuard:
    singularity_threshold: float = _coerce_float(_NUMERICAL_GUARD_CONFIG, "singularity_threshold")
    zero_magnitude_threshold: float = _coerce_float(_NUMERICAL_GUARD_CONFIG, "zero_magnitude_threshold")
    norm_floor: float = _coerce_float(_NUMERICAL_GUARD_CONFIG, "norm_floor")
    perturbative_condition_limit: float = _coerce_float(_NUMERICAL_GUARD_CONFIG, "perturbative_condition_limit")
    raise_on_violation: bool = False
    logger_name: str = "pub.numerics"

    def _handle(self, message: str) -> None:
        if self.raise_on_violation:
            raise FloatingPointError(message)
        logging.getLogger(self.logger_name).warning(message)

    def _warn_physics_domain(self, message: str, *, stacklevel: int = 2) -> None:
        if self.raise_on_violation:
            raise FloatingPointError(message)
        logging.getLogger(self.logger_name).warning(message)
        warnings.warn(message, PhysicsDomainWarning, stacklevel=stacklevel)

    def warn_coordinate_singularity(
        self,
        *,
        coordinate: str,
        value: float,
        fallback: float,
        detail: str,
    ) -> float:
        message = (
            f"Coordinate singularity audit: {coordinate}={value:.6e} approached the PDG parameterization "
            f"singular surface; using fallback {fallback:.6e}. {detail}"
        )
        self._warn_physics_domain(message, stacklevel=2)
        return fallback

    def clamp_positive(self, value: float, *, coordinate: str, floor: float | None = None) -> float:
        threshold = self.singularity_threshold if floor is None else floor
        if value <= threshold:
            self._handle(
                f"Singularity logging: {coordinate}={value:.6e} approached a singular point; using floor {threshold:.6e}."
            )
            return threshold
        return value

    def clamp_nonzero_magnitude(
        self,
        value: complex | float,
        *,
        coordinate: str,
        fallback: complex | float,
        floor: float | None = None,
    ) -> complex | float:
        threshold = self.zero_magnitude_threshold if floor is None else floor
        magnitude = abs(value)
        if magnitude <= threshold:
            self._handle(
                f"Singularity logging: |{coordinate}|={magnitude:.6e} approached zero; using fallback {fallback!r}."
            )
            return fallback
        return value

    def clamp_unit_interval(self, value: float, *, coordinate: str) -> float:
        clamped = min(1.0, max(0.0, value))
        if clamped != value:
            deviation = abs(value - clamped)
            if deviation > self.singularity_threshold:
                self._warn_physics_domain(
                    f"Domain saturation audit: {coordinate}={value:.6e} left [0, 1]; using saturated value {clamped:.6e}.",
                    stacklevel=2,
                )
        return clamped

    def clamp_signed_unit_interval(self, value: float, *, coordinate: str) -> float:
        clamped = min(1.0, max(-1.0, value))
        if clamped != value:
            deviation = abs(value - clamped)
            if deviation > self.singularity_threshold:
                self._warn_physics_domain(
                    f"Domain saturation audit: {coordinate}={value:.6e} left [-1, 1]; using saturated value {clamped:.6e}.",
                    stacklevel=2,
                )
        return clamped

    def clamp_probability(self, value: float, *, coordinate: str) -> float:
        clamped = min(1.0, max(0.0, value))
        if clamped != value:
            deviation = abs(value - clamped)
            if deviation > self.singularity_threshold:
                self._warn_physics_domain(
                    f"Domain saturation audit: {coordinate}={value:.6e} left [0, 1]; using saturated value {clamped:.6e}.",
                    stacklevel=2,
                )
        return clamped

    def stable_phase(self, value: complex, *, coordinate: str) -> complex:
        safe_value = complex(
            self.clamp_nonzero_magnitude(
                value,
                coordinate=coordinate,
                fallback=1.0 + 0.0j,
            )
        )
        return safe_value / abs(safe_value)

    def stable_norm(self, value: float, *, coordinate: str, floor: float | None = None) -> float:
        threshold = self.norm_floor if floor is None else floor
        if value <= threshold:
            self._handle(
                f"Singularity logging: {coordinate}={value:.6e} approached zero norm; using floor {threshold:.6e}."
            )
            return threshold
        return value

    def require_nonzero_magnitude(
        self,
        value: complex | float,
        *,
        coordinate: str,
        detail: str,
        floor: float | None = None,
    ) -> complex | float:
        threshold = self.zero_magnitude_threshold if floor is None else floor
        magnitude = abs(value)
        if magnitude <= threshold:
            raise PhysicalSingularityException(
                f"Physically non-perturbative branch: |{coordinate}|={magnitude:.6e} crossed the exclusion threshold "
                f"{threshold:.6e}. {detail}"
            )
        return value

    def require_positive(self, value: float, *, coordinate: str, detail: str, floor: float | None = None) -> float:
        threshold = self.singularity_threshold if floor is None else floor
        if value <= threshold:
            raise PhysicalSingularityException(
                f"Physically non-perturbative branch: {coordinate}={value:.6e} crossed the positive threshold "
                f"{threshold:.6e}. {detail}"
            )
        return value

    def require_perturbative_condition_number(
        self,
        condition_number: float,
        *,
        coordinate: str,
        detail: str,
        limit: float | None = None,
    ) -> float:
        resolved_limit = self.perturbative_condition_limit if limit is None else limit
        if not math.isfinite(condition_number) or condition_number > resolved_limit:
            raise PerturbativeBreakdownException(
                f"Physically non-perturbative branch: {coordinate} has condition number {condition_number:.6e}, "
                f"above the perturbative limit {resolved_limit:.6e}. {detail}"
            )
        return condition_number


@dataclass(frozen=True)
class SolverConfig:
    rtol: float = _coerce_float(_SOLVER_CONFIG, "rtol")
    atol: float = _coerce_float(_SOLVER_CONFIG, "atol")
    method: str = str(_SOLVER_CONFIG["method"])
    fallback_methods: tuple[str, ...] = _coerce_str_sequence(_SOLVER_CONFIG, "fallback_methods") or ("BDF", "RK45")
    finite_diff_step: float = _coerce_float(_SOLVER_CONFIG, "finite_diff_step")
    jacobian_relative_step: float = _coerce_float(_SOLVER_CONFIG, "jacobian_relative_step")
    parametric_covariance_mc_samples: int = _coerce_int(_SOLVER_CONFIG, "parametric_covariance_mc_samples")
    quad_epsabs: float = _coerce_float(_SOLVER_CONFIG, "quad_epsabs")
    quad_epsrel: float = _coerce_float(_SOLVER_CONFIG, "quad_epsrel")
    linear_algebra_backend: str = str(_SOLVER_CONFIG.get("linear_algebra_backend", "auto"))
    backend_priority: tuple[str, ...] = _coerce_str_sequence(_SOLVER_CONFIG, "backend_priority") or ("cuda", "tensorcore", "cpu")
    deterministic_float64: bool = _coerce_bool(_SOLVER_CONFIG, "deterministic_float64", True)
    arbitrary_precision_dps: int = _coerce_optional_int(_SOLVER_CONFIG, "arbitrary_precision_dps", 250)
    unity_noise_floor_target: float = _coerce_optional_float(_SOLVER_CONFIG, "unity_noise_floor_target", 1.0e-123)
    stability_guard: NumericalStabilityGuard = field(default_factory=NumericalStabilityGuard)

    def __post_init__(self) -> None:
        normalized_method = str(self.method).strip()
        default_fallback_methods = ("BDF", "RK45")
        normalized_fallback_methods: list[str] = []
        for candidate in self.fallback_methods:
            normalized_candidate = str(candidate).strip()
            if not normalized_candidate or normalized_candidate == normalized_method:
                continue
            if normalized_candidate not in normalized_fallback_methods:
                normalized_fallback_methods.append(normalized_candidate)
        normalized_backend = str(self.linear_algebra_backend).strip().lower() or "auto"
        normalized_priority = tuple(str(entry).strip().lower() for entry in self.backend_priority if str(entry).strip())
        if normalized_method != "Radau":
            raise SolverException(f"SolverConfig is pinned to method='Radau'; got {self.method!r}.")
        if not math.isfinite(self.rtol) or self.rtol <= 0.0:
            raise SolverException(f"SolverConfig requires a positive finite rtol; got {self.rtol!r}.")
        resolved_atol_ceiling = 1.0e-10 if tuple(normalized_fallback_methods) != default_fallback_methods else 1.0e-12
        if not math.isfinite(self.atol) or self.atol <= 0.0 or self.atol > resolved_atol_ceiling:
            raise SolverException(
                f"SolverConfig requires a positive finite atol <= 1e-12 for strict Radau verification; got {self.atol!r}."
            )
        if not normalized_priority:
            raise SolverException("SolverConfig requires at least one linear-algebra backend priority entry.")
        if self.arbitrary_precision_dps < 250:
            raise SolverException(
                f"SolverConfig requires arbitrary_precision_dps >= 250 for theorem-level gravity checks; got {self.arbitrary_precision_dps!r}."
            )
        if not math.isfinite(self.unity_noise_floor_target) or self.unity_noise_floor_target <= 0.0 or self.unity_noise_floor_target > 1.0e-123:
            raise SolverException(
                "SolverConfig requires unity_noise_floor_target to be a positive finite floor no larger than 1e-123; "
                f"got {self.unity_noise_floor_target!r}."
            )
        object.__setattr__(self, "method", normalized_method)
        object.__setattr__(self, "fallback_methods", tuple(normalized_fallback_methods))
        object.__setattr__(self, "linear_algebra_backend", normalized_backend)
        object.__setattr__(self, "backend_priority", normalized_priority)

    @property
    def method_ladder(self) -> tuple[str, ...]:
        return (self.method, *self.fallback_methods)

    @property
    def prioritizes_accelerator(self) -> bool:
        accelerator_tokens = {"cuda", "gpu", "tensorcore", "tensor", "jax", "torch"}
        return any(token in accelerator_tokens for token in self.backend_priority)


DEFAULT_SOLVER_CONFIG = SolverConfig()


def solver_method_ladder(solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG) -> tuple[str, ...]:
    """Return the verifier's strictly pinned Radau IIA method tuple."""

    return (solver_config.method,)


def solver_isclose(
    left: float,
    right: float,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    rel_tol: float | None = 0.0,
    abs_tol: float | None = None,
) -> bool:
    """Compare scalars using the configured solver tolerance by default."""

    resolved_rel_tol = solver_config.rtol if rel_tol is None else rel_tol
    resolved_abs_tol = solver_config.atol if abs_tol is None else abs_tol
    return math.isclose(left, right, rel_tol=resolved_rel_tol, abs_tol=resolved_abs_tol)


__all__ = [
    "DEFAULT_SOLVER_CONFIG",
    "NumericalStabilityGuard",
    "PerturbativeBreakdownException",
    "PhysicalSingularityException",
    "PhysicsDomainWarning",
    "Sector",
    "SolverException",
    "SolverConfig",
    "solver_method_ladder",
    "solver_isclose",
]
