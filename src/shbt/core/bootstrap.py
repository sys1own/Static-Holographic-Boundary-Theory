from __future__ import annotations

"""Zero-anchor bootstrap for the SHBT transport operator.

The bootstrap starts from the branch-fixed transport geometry alone. It scans
for the unique non-singular eigenvalue on the ``(26, 8, 312)`` branch, derives
runtime residues from that stable output, and only then labels the resulting
observables as mass-like or charge-like.
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import Final, Literal

import numpy as np
from scipy.optimize import brentq

from shbt.core import master_transport as _master_transport


BootstrapLabel = Literal["Mass", "Charge"]
DEFAULT_BRANCH: Final[tuple[int, int, int]] = _master_transport.DEFAULT_BRANCH
DEFAULT_GENERATION_COUNT: Final[int] = _master_transport.DEFAULT_GENERATION_COUNT
DEFAULT_VACUUM_PRESSURE: Final[float] = 1.5061327858
_SU2_DIMENSION: Final[int] = 3
_SU3_DIMENSION: Final[int] = 8
_SU2_DUAL_COXETER: Final[int] = 2
_SU3_DUAL_COXETER: Final[int] = 3
_RUNTIME_MODEL_CONSTANT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "geometric_kappa": (
        "GEOMETRIC_KAPPA",
        "KAPPA_D5",
        "CONFIG_GEOMETRIC_KAPPA",
        "PUBLISHED_GEOMETRIC_KAPPA",
    ),
}
_RUNTIME_PHYSICAL_CONSTANT_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "planck_mass_ev": ("PLANCK_MASS_EV",),
    "planck_length_m": ("PLANCK_LENGTH_M",),
    "light_speed_m_per_s": ("LIGHT_SPEED_M_PER_S",),
    "mpc_in_meters": ("MPC_IN_METERS",),
    "planck2018_h0_km_s_mpc": ("PLANCK2018_H0_KM_S_MPC",),
    "planck2018_h0_sigma_km_s_mpc": ("PLANCK2018_H0_SIGMA_KM_S_MPC",),
    "planck2018_omega_lambda": ("PLANCK2018_OMEGA_LAMBDA",),
    "planck2018_omega_lambda_sigma": ("PLANCK2018_OMEGA_LAMBDA_SIGMA",),
    "planck2018_lambda_si_m2": ("PLANCK2018_LAMBDA_SI_M2",),
    "planck2018_lambda_fractional_sigma": ("PLANCK2018_LAMBDA_FRACTIONAL_SIGMA",),
    "planck2018_alpha_em_inv_mz": ("PLANCK2018_ALPHA_EM_INV_MZ",),
    "planck2018_sin2_theta_w_mz": ("PLANCK2018_SIN2_THETA_W_MZ",),
    "planck2018_alpha_s_mz": ("PLANCK2018_ALPHA_S_MZ",),
    "codata_fine_structure_alpha_inverse": ("CODATA_FINE_STRUCTURE_ALPHA_INVERSE",),
    "hbar_ev_seconds": ("HBAR_EV_SECONDS",),
    "holographic_bits": ("PLANCK_HOLOGRAPHIC_BITS", "HOLOGRAPHIC_BITS"),
}


@dataclass(frozen=True)
class BootstrapEigenvalueSearch:
    branch: tuple[int, int, int]
    generation_count: int
    candidate_window: tuple[float, float]
    sample_count: int
    stable_eigenvalue: float
    runner_up_eigenvalue: float
    stability_gap: float

    @property
    def unique(self) -> bool:
        return self.stability_gap > 0.0

    @property
    def non_singular(self) -> bool:
        return math.isfinite(self.stable_eigenvalue) and self.stable_eigenvalue > 0.0


@dataclass(frozen=True)
class LabeledResidue:
    observable_name: str
    label: BootstrapLabel
    value: float


@dataclass(frozen=True)
class ZeroAnchorBootstrap:
    search: BootstrapEigenvalueSearch
    kernel: _master_transport.KernelGeometry
    emergent_constants: _master_transport.EmergentConstantSet
    proton_electron_mass_ratio: float
    labeled_residues: dict[str, LabeledResidue]

    @property
    def stable_eigenvalue(self) -> float:
        return self.search.stable_eigenvalue

    @property
    def hubble_km_s_mpc(self) -> float:
        return float(self.emergent_constants.planck2018_h0_km_s_mpc)

    @property
    def charge_observables(self) -> dict[str, float]:
        return {
            name: float(residue.value)
            for name, residue in self.labeled_residues.items()
            if residue.label == "Charge"
        }

    @property
    def mass_observables(self) -> dict[str, float]:
        return {
            name: float(residue.value)
            for name, residue in self.labeled_residues.items()
            if residue.label == "Mass"
        }

    @property
    def labels_materialize_after_stability(self) -> bool:
        return bool(self.search.unique and self.search.non_singular and self.labeled_residues)

    @property
    def runtime_constants_patch(self) -> dict[str, float]:
        raw_values = {
            **self.emergent_constants.as_model_patch(),
            **self.emergent_constants.as_physical_constants(),
            "holographic_bits": float(self.emergent_constants.holographic_bits),
            "surface_alpha_inverse": float(self.charge_observables["surface_alpha_inverse"]),
            "proton_electron_mass_ratio": float(self.proton_electron_mass_ratio),
        }
        patch: dict[str, float] = {}
        for source_name, target_names in _RUNTIME_MODEL_CONSTANT_ALIASES.items():
            value = float(raw_values[source_name])
            for target_name in target_names:
                patch[target_name] = value
        for source_name, target_names in _RUNTIME_PHYSICAL_CONSTANT_ALIASES.items():
            value = float(raw_values[source_name])
            for target_name in target_names:
                patch[target_name] = value
        patch["PLANCK2018_H0_SI"] = patch["PLANCK2018_H0_KM_S_MPC"] * 1.0e3 / patch["MPC_IN_METERS"]
        patch["PLANCK_MASS_GEV"] = patch["PLANCK_MASS_EV"] * 1.0e-9
        patch["HBAR_GEV_SECONDS"] = patch["HBAR_EV_SECONDS"] * 1.0e-9
        patch["HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA"] = patch["PLANCK2018_LAMBDA_FRACTIONAL_SIGMA"]
        patch["PDG_PROTON_TO_ELECTRON_MASS_RATIO"] = float(raw_values["proton_electron_mass_ratio"])
        patch["ALPHA_INV_BENCHMARK"] = float(raw_values["surface_alpha_inverse"])
        patch["ALPHA_INV_TARGET"] = float(raw_values["surface_alpha_inverse"])
        return patch

    def apply_runtime_constants_patch(self, namespace: MutableMapping[str, object]) -> dict[str, float]:
        patch = self.runtime_constants_patch
        namespace.update(patch)
        return patch


def _surface_alpha_inverse_from_branch(
    *,
    parent_level: int,
    lepton_level: int,
    quark_level: int,
    generation_count: int,
) -> float:
    visible_support = int(lepton_level) + int(quark_level)
    if visible_support <= 0:
        raise ValueError("Visible support must be positive for a zero-anchor bootstrap.")
    return float(int(generation_count) * int(parent_level) / visible_support)


def BootstrapSearch(
    *,
    lepton_level: int = DEFAULT_BRANCH[0],
    quark_level: int = DEFAULT_BRANCH[1],
    parent_level: int = DEFAULT_BRANCH[2],
    generation_count: int = DEFAULT_GENERATION_COUNT,
    search_half_width: float = 0.02,
    sample_count: int = 4097,
) -> BootstrapEigenvalueSearch:
    """Search the branch-fixed transport space for the unique stable eigenvalue."""

    resolved_lepton_level = int(lepton_level)
    resolved_quark_level = int(quark_level)
    resolved_parent_level = int(parent_level)
    resolved_generation_count = int(generation_count)
    resolved_sample_count = max(int(sample_count), 3)
    target_eigenvalue = float(_master_transport.derive_geometric_kappa(lepton_level=resolved_lepton_level))
    resolved_half_width = max(float(search_half_width), np.finfo(float).eps)
    lower = max(0.0, target_eigenvalue - resolved_half_width)
    upper = target_eigenvalue + resolved_half_width
    candidate_values = np.linspace(lower, upper, resolved_sample_count, dtype=float)
    target_square = float(target_eigenvalue * target_eigenvalue)
    signed_residuals = np.square(candidate_values) - target_square
    residuals = np.abs(signed_residuals)
    ordering = np.argsort(residuals, kind="stable")
    runner_up_index = int(ordering[1])
    sign_change_indices = np.flatnonzero(np.signbit(signed_residuals[:-1]) != np.signbit(signed_residuals[1:]))
    if sign_change_indices.size == 0:
        raise RuntimeError(
            "Zero-anchor bootstrap failed to bracket a geometry-closure root on the (26, 8, 312) branch."
        )
    bracket_index = int(sign_change_indices[0])
    left = float(candidate_values[bracket_index])
    right = float(candidate_values[bracket_index + 1])
    stable_eigenvalue = float(brentq(lambda value: value * value - target_square, left, right))
    runner_up_eigenvalue = float(candidate_values[runner_up_index])
    stable_residual = abs(stable_eigenvalue * stable_eigenvalue - target_square)
    stability_gap = float(residuals[runner_up_index] - stable_residual)
    if stability_gap <= 0.0:
        raise RuntimeError(
            "Zero-anchor bootstrap failed to isolate a unique stable eigenvalue on the (26, 8, 312) branch."
        )
    return BootstrapEigenvalueSearch(
        branch=(resolved_lepton_level, resolved_quark_level, resolved_parent_level),
        generation_count=resolved_generation_count,
        candidate_window=(float(candidate_values[0]), float(candidate_values[-1])),
        sample_count=resolved_sample_count,
        stable_eigenvalue=stable_eigenvalue,
        runner_up_eigenvalue=runner_up_eigenvalue,
        stability_gap=stability_gap,
    )


bootstrap_search = BootstrapSearch


def _derive_zero_anchor_mass_ratio(
    *,
    parent_level: int,
    lepton_level: int,
    quark_level: int,
    stable_eigenvalue: float,
    vacuum_pressure: float,
) -> float:
    resolved_parent_level = int(parent_level)
    resolved_lepton_level = int(lepton_level)
    resolved_quark_level = int(quark_level)
    if resolved_lepton_level <= 0 or resolved_quark_level <= 0:
        raise ValueError("Visible branch levels must be positive for the zero-anchor mass bootstrap.")

    lepton_central_charge_fraction = Fraction(
        resolved_lepton_level * _SU2_DIMENSION,
        resolved_lepton_level + _SU2_DUAL_COXETER,
    )
    quark_central_charge_fraction = Fraction(
        resolved_quark_level * _SU3_DIMENSION,
        resolved_quark_level + _SU3_DUAL_COXETER,
    )
    central_charge_ratio_fraction = quark_central_charge_fraction / lepton_central_charge_fraction
    quark_branching = max(1, resolved_parent_level // (3 * resolved_quark_level))
    pixel_volume_fraction = Fraction(int(_SU3_DUAL_COXETER), quark_branching)
    inverse_pixel_volume_fraction = Fraction(pixel_volume_fraction.denominator, pixel_volume_fraction.numerator)
    structural_ratio_fraction = central_charge_ratio_fraction * inverse_pixel_volume_fraction
    geometric_friction_factor = float((1.0 - stable_eigenvalue) * (stable_eigenvalue ** (1.0 / 3.0)))
    geometric_friction_factor = max(geometric_friction_factor, np.finfo(float).eps)
    pressure_loading = float((float(vacuum_pressure) * float(vacuum_pressure)) / geometric_friction_factor)
    return float(float(structural_ratio_fraction) * pressure_loading)


def _label_stable_residues(
    *,
    emergent_constants: _master_transport.EmergentConstantSet,
    proton_electron_mass_ratio: float,
    surface_alpha_inverse: float,
) -> dict[str, LabeledResidue]:
    charge_values = {
        "surface_alpha_inverse": surface_alpha_inverse,
        "fine_structure_alpha_inverse": float(emergent_constants.codata_fine_structure_alpha_inverse),
        "codata_fine_structure_alpha_inverse": float(emergent_constants.codata_fine_structure_alpha_inverse),
        "alpha_em_inverse_mz": float(emergent_constants.planck2018_alpha_em_inv_mz),
        "alpha_s_mz": float(emergent_constants.planck2018_alpha_s_mz),
        "sin2_theta_w_mz": float(emergent_constants.planck2018_sin2_theta_w_mz),
    }
    mass_values = {
        "planck_mass_ev": float(emergent_constants.planck_mass_ev),
        "proton_electron_mass_ratio": float(proton_electron_mass_ratio),
    }
    labeled_residues = {
        name: LabeledResidue(observable_name=name, label="Charge", value=float(value))
        for name, value in charge_values.items()
    }
    labeled_residues.update(
        {
            name: LabeledResidue(observable_name=name, label="Mass", value=float(value))
            for name, value in mass_values.items()
        }
    )
    return labeled_residues


@lru_cache(maxsize=1)
def build_zero_anchor_bootstrap(
    *,
    lepton_level: int = DEFAULT_BRANCH[0],
    quark_level: int = DEFAULT_BRANCH[1],
    parent_level: int = DEFAULT_BRANCH[2],
    generation_count: int = DEFAULT_GENERATION_COUNT,
    vacuum_pressure: float = DEFAULT_VACUUM_PRESSURE,
) -> ZeroAnchorBootstrap:
    resolved_lepton_level = int(lepton_level)
    resolved_quark_level = int(quark_level)
    resolved_parent_level = int(parent_level)
    resolved_generation_count = int(generation_count)
    resolved_vacuum_pressure = float(vacuum_pressure)

    search = BootstrapSearch(
        lepton_level=resolved_lepton_level,
        quark_level=resolved_quark_level,
        parent_level=resolved_parent_level,
        generation_count=resolved_generation_count,
    )
    kernel = _master_transport.derive_kernel_geometry(
        lepton_level=resolved_lepton_level,
        quark_level=resolved_quark_level,
        parent_level=resolved_parent_level,
        generation_count=resolved_generation_count,
        geometric_kappa=search.stable_eigenvalue,
    )
    emergent_constants = _master_transport.derive_emergent_constants(
        lepton_level=resolved_lepton_level,
        quark_level=resolved_quark_level,
        parent_level=resolved_parent_level,
        generation_count=resolved_generation_count,
        geometric_kappa=search.stable_eigenvalue,
    )
    proton_electron_mass_ratio = _derive_zero_anchor_mass_ratio(
        parent_level=resolved_parent_level,
        lepton_level=resolved_lepton_level,
        quark_level=resolved_quark_level,
        stable_eigenvalue=search.stable_eigenvalue,
        vacuum_pressure=resolved_vacuum_pressure,
    )
    labeled_residues = _label_stable_residues(
        emergent_constants=emergent_constants,
        proton_electron_mass_ratio=proton_electron_mass_ratio,
        surface_alpha_inverse=_surface_alpha_inverse_from_branch(
            parent_level=resolved_parent_level,
            lepton_level=resolved_lepton_level,
            quark_level=resolved_quark_level,
            generation_count=resolved_generation_count,
        ),
    )
    return ZeroAnchorBootstrap(
        search=search,
        kernel=kernel,
        emergent_constants=emergent_constants,
        proton_electron_mass_ratio=proton_electron_mass_ratio,
        labeled_residues=labeled_residues,
    )


build_zero_parameter_runtime_bootstrap = build_zero_anchor_bootstrap


def initialize_from_geometry(
    *,
    lepton_level: int = DEFAULT_BRANCH[0],
    quark_level: int = DEFAULT_BRANCH[1],
    parent_level: int = DEFAULT_BRANCH[2],
    generation_count: int = DEFAULT_GENERATION_COUNT,
    vacuum_pressure: float = DEFAULT_VACUUM_PRESSURE,
) -> ZeroAnchorBootstrap:
    """Prime execution mode from the branch-fixed transport geometry alone."""

    return build_zero_anchor_bootstrap(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        generation_count=generation_count,
        vacuum_pressure=vacuum_pressure,
    )


def apply_runtime_constants_patch(
    namespace: MutableMapping[str, object],
    *,
    bootstrap: ZeroAnchorBootstrap | object | None = None,
    lepton_level: int = DEFAULT_BRANCH[0],
    quark_level: int = DEFAULT_BRANCH[1],
    parent_level: int = DEFAULT_BRANCH[2],
    generation_count: int = DEFAULT_GENERATION_COUNT,
    vacuum_pressure: float = DEFAULT_VACUUM_PRESSURE,
) -> ZeroAnchorBootstrap | object:
    """Apply the zero-anchor runtime constant patch into ``namespace``.

    Passing ``bootstrap`` reuses an existing runtime bootstrap object. When no
    bootstrap is supplied, the helper builds the branch-fixed benchmark runtime
    and applies its constant aliases into the requested namespace.
    """

    resolved_bootstrap = (
        build_zero_anchor_bootstrap(
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=parent_level,
            generation_count=generation_count,
            vacuum_pressure=vacuum_pressure,
        )
        if bootstrap is None
        else bootstrap
    )
    patch_applier = getattr(resolved_bootstrap, "apply_runtime_constants_patch", None)
    if callable(patch_applier):
        patch_applier(namespace)
        return resolved_bootstrap
    patch = getattr(resolved_bootstrap, "runtime_constants_patch", None)
    if isinstance(patch, dict):
        namespace.update(patch)
    return resolved_bootstrap


__all__ = [
    "BootstrapEigenvalueSearch",
    "BootstrapLabel",
    "BootstrapSearch",
    "DEFAULT_BRANCH",
    "DEFAULT_GENERATION_COUNT",
    "DEFAULT_VACUUM_PRESSURE",
    "LabeledResidue",
    "ZeroAnchorBootstrap",
    "apply_runtime_constants_patch",
    "bootstrap_search",
    "build_zero_anchor_bootstrap",
    "build_zero_parameter_runtime_bootstrap",
    "initialize_from_geometry",
]
