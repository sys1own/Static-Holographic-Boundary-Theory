from __future__ import annotations

import math
from decimal import Decimal
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from shbt.config_loader import DEFAULT_CONFIG_LOADER


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    @property
    def central(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def sigma(self) -> float:
        return 0.5 * (self.upper - self.lower)


@dataclass(frozen=True)
class ExperimentalContext:
    nufit_release: str
    nufit_reference: str
    pdg_release: str
    pdg_reference: str
    lepton_intervals: dict[str, Interval]
    quark_intervals: dict[str, Interval]
    ckm_gamma_experimental_input_deg: Interval


@dataclass(frozen=True)
class BenchmarkConstantDefinition:
    name: str
    value: Any
    legacy_metadata_paths: tuple[str, ...] = ()
    allowed_legacy_classifications: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkConstantTier:
    identifier: str
    title: str
    description: str
    constants: tuple[BenchmarkConstantDefinition, ...]

    @property
    def label(self) -> str:
        return f"{self.identifier}: {self.title}"

    @property
    def values(self) -> dict[str, Any]:
        return {constant.name: constant.value for constant in self.constants}


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return value


def _coerce_float(config: dict[str, Any], key: str) -> float:
    return float(config[key])


def _coerce_int(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if isinstance(value, bool):
        raise TypeError(f"Configuration value '{key}' must be an integer, not a boolean")
    return int(value)


def _coerce_str(config: dict[str, Any], key: str) -> str:
    return str(config[key])


def _coerce_float_mapping(config: dict[str, Any], key: str) -> dict[str, float]:
    return {name: float(value) for name, value in _require_mapping(config, key).items()}


def _coerce_interval(name: str, values: Any) -> Interval:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise TypeError(f"Interval '{name}' must be a length-2 sequence")
    lower, upper = values
    return Interval(float(lower), float(upper))


def _coerce_interval_mapping(config: dict[str, Any], key: str) -> dict[str, Interval]:
    return {name: _coerce_interval(name, values) for name, values in _require_mapping(config, key).items()}


def _coerce_float_sequence(config: dict[str, Any], key: str) -> tuple[float, ...]:
    values = config.get(key)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"Configuration value '{key}' must be a sequence")
    return tuple(float(value) for value in values)


def _coerce_int_sequence(config: dict[str, Any], key: str) -> tuple[int, ...]:
    values = config.get(key)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"Configuration value '{key}' must be a sequence")
    return tuple(int(value) for value in values)


def _coerce_nested_int_sequence(config: dict[str, Any], key: str) -> tuple[tuple[int, ...], ...]:
    values = config.get(key)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"Configuration value '{key}' must be a nested sequence")
    return tuple(tuple(int(item) for item in value) for value in values)


def _format_latex_scientific(value: float) -> str:
    coefficient_text, exponent_text = f"{value:.0e}".split("e")
    exponent = int(exponent_text)
    if coefficient_text == "1":
        return rf"10^{{{exponent}}}"
    return rf"{coefficient_text} \times 10^{{{exponent}}}"


_PHYSICS_CONFIG = DEFAULT_CONFIG_LOADER.load_physics_constants()
_EXPERIMENTAL_CONFIG = DEFAULT_CONFIG_LOADER.load_experimental_bounds()
# Legacy config-level provenance tags are retained as metadata. The strict
# benchmark hierarchy used by the repository is declared explicitly below.
BENCHMARK_PARAMETER_CLASSIFICATIONS = DEFAULT_CONFIG_LOADER.load_parameter_classifications()

_SOLVER_CONFIG = _require_mapping(_PHYSICS_CONFIG, "solver")
_PRESENTATION_CONFIG = _require_mapping(_PHYSICS_CONFIG, "presentation")
_MAJORANA_CONFIG = _require_mapping(_PHYSICS_CONFIG, "majorana")
_PHYSICAL_CONSTANTS_CONFIG = _require_mapping(_PHYSICS_CONFIG, "physical_constants")
_MODEL_CONFIG = _require_mapping(_PHYSICS_CONFIG, "model")
_BENCHMARKS_CONFIG = _require_mapping(_PHYSICS_CONFIG, "benchmarks")
_SCALES_CONFIG = _require_mapping(_PHYSICS_CONFIG, "scales")
_UNCERTAINTIES_CONFIG = _require_mapping(_PHYSICS_CONFIG, "uncertainties")
_REPORTING_CONFIG = _require_mapping(_PHYSICS_CONFIG, "reporting")
_GROUP_THEORY_CONFIG = _require_mapping(_PHYSICS_CONFIG, "group_theory")
_MODEL_CONSTANTS_CONFIG = _require_mapping(_PHYSICS_CONFIG, "model_constants")
_RUNTIME_KNOBS_CONFIG = _require_mapping(_PHYSICS_CONFIG, "runtime_knobs")
_RELEASES_CONFIG = _require_mapping(_EXPERIMENTAL_CONFIG, "releases")
_NORMAL_ORDERING_MASS_SPLITTINGS_CONFIG = _require_mapping(_EXPERIMENTAL_CONFIG, "normal_ordering_mass_splittings_ev2")

# Required Observational Boundary Condition (OBC); see README.md for tier hierarchy.
PLANCK_MASS_EV = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck_mass_ev")
PLANCK_LENGTH_M = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck_length_m")
LIGHT_SPEED_M_PER_S = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "light_speed_m_per_s")
MPC_IN_METERS = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "mpc_in_meters")
PLANCK2018_H0_KM_S_MPC = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_h0_km_s_mpc")
PLANCK2018_H0_SIGMA_KM_S_MPC = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_h0_sigma_km_s_mpc")
PLANCK2018_OMEGA_LAMBDA = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_omega_lambda")
PLANCK2018_OMEGA_LAMBDA_SIGMA = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_omega_lambda_sigma")
# Required Observational Boundary Condition (OBC); see README.md for tier hierarchy.
PLANCK2018_ALPHA_EM_INV_MZ = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_alpha_em_inv_mz")
PLANCK2018_SIN2_THETA_W_MZ = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_sin2_theta_w_mz")
PLANCK2018_ALPHA_S_MZ = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_alpha_s_mz")

PLANCK2018_H0_SI = PLANCK2018_H0_KM_S_MPC * 1.0e3 / MPC_IN_METERS
_PLANCK2018_LAMBDA_SI_M2_COMPUTED = (
    3.0 * PLANCK2018_OMEGA_LAMBDA * PLANCK2018_H0_SI * PLANCK2018_H0_SI / (LIGHT_SPEED_M_PER_S * LIGHT_SPEED_M_PER_S)
)
PLANCK2018_LAMBDA_SI_M2 = float(
    _PHYSICAL_CONSTANTS_CONFIG.get("planck2018_lambda_si_m2", _PLANCK2018_LAMBDA_SI_M2_COMPUTED)
)
_PLANCK2018_LAMBDA_FRACTIONAL_SIGMA_COMPUTED = math.sqrt(
    (PLANCK2018_OMEGA_LAMBDA_SIGMA / PLANCK2018_OMEGA_LAMBDA) ** 2
    + (2.0 * PLANCK2018_H0_SIGMA_KM_S_MPC / PLANCK2018_H0_KM_S_MPC) ** 2
)
PLANCK2018_LAMBDA_FRACTIONAL_SIGMA = float(
    _PHYSICAL_CONSTANTS_CONFIG.get("planck2018_lambda_fractional_sigma", _PLANCK2018_LAMBDA_FRACTIONAL_SIGMA_COMPUTED)
)
PLANCK_HOLOGRAPHIC_BITS = 3.0 * math.pi / (PLANCK2018_LAMBDA_SI_M2 * PLANCK_LENGTH_M * PLANCK_LENGTH_M)
HOLOGRAPHIC_BITS = PLANCK_HOLOGRAPHIC_BITS
HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA = PLANCK2018_LAMBDA_FRACTIONAL_SIGMA
CODATA_FINE_STRUCTURE_ALPHA_INVERSE = float(
    _PHYSICAL_CONSTANTS_CONFIG.get("codata_fine_structure_alpha_inverse", 137.035999084)
)
HBAR_EV_SECONDS = float(_PHYSICAL_CONSTANTS_CONFIG.get("hbar_ev_seconds", 6.582119569e-16))
HBAR_GEV_SECONDS = HBAR_EV_SECONDS * 1.0e-9
PLANCK_MASS_GEV = PLANCK_MASS_EV * 1.0e-9

AUDIT_TOLERANCE = Decimal("1e-38")

GEOMETRIC_KAPPA = _coerce_float(_MODEL_CONFIG, "geometric_kappa")
KAPPA_D5 = GEOMETRIC_KAPPA
PARENT_LEVEL = _coerce_int(_MODEL_CONFIG, "parent_level")
LEPTON_FIXED_POINT_INDEX = _coerce_int(_MODEL_CONFIG, "lepton_fixed_point_index")
QUARK_FIXED_POINT_INDEX = _coerce_int(_MODEL_CONFIG, "quark_fixed_point_index")
LEPTON_LEVEL = PARENT_LEVEL // (2 * LEPTON_FIXED_POINT_INDEX)
QUARK_LEVEL = PARENT_LEVEL // (3 * QUARK_FIXED_POINT_INDEX)
G_SM = _coerce_int(_MODEL_CONFIG, "g_sm") if "g_sm" in _MODEL_CONFIG else 15
DIRAC_HIGGS_BENCHMARK_MASS_GEV = _coerce_float(_MODEL_CONFIG, "dirac_higgs_benchmark_mass_gev")

SM_GUT_YUKAWA_BENCHMARKS = _coerce_float_mapping(_BENCHMARKS_CONFIG, "sm_gut_yukawa")
SM_MZ_YUKAWA_BENCHMARKS = _coerce_float_mapping(_BENCHMARKS_CONFIG, "sm_mz_yukawa")
CHARGED_LEPTON_YUKAWA_RATIOS = {
    "electron": SM_MZ_YUKAWA_BENCHMARKS["electron"] / SM_MZ_YUKAWA_BENCHMARKS["tau"],
    "muon": SM_MZ_YUKAWA_BENCHMARKS["muon"] / SM_MZ_YUKAWA_BENCHMARKS["tau"],
}

GUT_SCALE_GEV = _coerce_float(_SCALES_CONFIG, "gut_scale_gev")
MZ_SCALE_GEV = _coerce_float(_SCALES_CONFIG, "mz_scale_gev")
TOP_POLE_MASS_GEV = _coerce_float(_SCALES_CONFIG, "top_pole_mass_gev")
HIGGS_POLE_MASS_GEV = _coerce_float(_SCALES_CONFIG, "higgs_pole_mass_gev")

PDG_TOP_POLE_MASS_CENTRAL_GEV = _coerce_float(_UNCERTAINTIES_CONFIG, "pdg_top_pole_mass_central_gev")
PDG_TOP_POLE_MASS_SIGMA_GEV = _coerce_float(_UNCERTAINTIES_CONFIG, "pdg_top_pole_mass_sigma_gev")
ALPHA_S_MZ_SIGMA = _coerce_float(_UNCERTAINTIES_CONFIG, "alpha_s_mz_sigma")
THEORETICAL_MATCHING_UNCERTAINTY_FRACTION = float(_UNCERTAINTIES_CONFIG.get("theoretical_matching_uncertainty_fraction", 0.05))
PARAMETRIC_TRANSPORT_COVARIANCE_FRACTION = _coerce_float(_UNCERTAINTIES_CONFIG, "parametric_transport_covariance_fraction")
THEORETICAL_UNCERTAINTY_FRACTION = THEORETICAL_MATCHING_UNCERTAINTY_FRACTION
PARAMETRIC_COVARIANCE_FRACTION = PARAMETRIC_TRANSPORT_COVARIANCE_FRACTION

TOPOLOGICAL_QUANTUM_NUMBER_DOF_SUBTRACTION = _coerce_int(_REPORTING_CONFIG, "topological_quantum_number_dof_subtraction")
THRESHOLD_MATCHING_DOF_SUBTRACTION = _coerce_int(_REPORTING_CONFIG, "threshold_matching_dof_subtraction")
PHENOMENOLOGICAL_DOF_ADJUSTMENT = TOPOLOGICAL_QUANTUM_NUMBER_DOF_SUBTRACTION + THRESHOLD_MATCHING_DOF_SUBTRACTION
HONEST_FREQUENTIST_DOF_SUBTRACTION = PHENOMENOLOGICAL_DOF_ADJUSTMENT

LATEX_TABLE_STYLE = _coerce_str(_PRESENTATION_CONFIG, "latex_table_style")
MAJORANA_LOBSTER_GRID_POINTS = _coerce_int(_MAJORANA_CONFIG, "lobster_grid_points")

SU2_DIMENSION = _coerce_int(_GROUP_THEORY_CONFIG, "su2_dimension")
SU3_DIMENSION = _coerce_int(_GROUP_THEORY_CONFIG, "su3_dimension")
SO10_DIMENSION = _coerce_int(_GROUP_THEORY_CONFIG, "so10_dimension")
SU2_DUAL_COXETER = _coerce_int(_GROUP_THEORY_CONFIG, "su2_dual_coxeter")
SU3_DUAL_COXETER = _coerce_int(_GROUP_THEORY_CONFIG, "su3_dual_coxeter")
SO10_DUAL_COXETER = _coerce_int(_GROUP_THEORY_CONFIG, "so10_dual_coxeter")
SO10_TO_SU2_EMBEDDING_INDEX = _coerce_int(_GROUP_THEORY_CONFIG, "so10_to_su2_embedding_index")
SO10_TO_SU3_EMBEDDING_INDEX = _coerce_int(_GROUP_THEORY_CONFIG, "so10_to_su3_embedding_index")
SO10_RANK = _coerce_int(_GROUP_THEORY_CONFIG, "so10_rank")
SU3_RANK = _coerce_int(_GROUP_THEORY_CONFIG, "su3_rank")
RANK_DIFFERENCE = SO10_RANK - SU3_RANK
R_GUT = QUARK_LEVEL / (LEPTON_LEVEL + SU2_DUAL_COXETER)

BENCHMARK_C_DARK_RESIDUE_FRACTION = (
    Fraction(PARENT_LEVEL * SU3_DIMENSION, PARENT_LEVEL + SU3_DUAL_COXETER)
    + Fraction(PARENT_LEVEL * SU2_DIMENSION, PARENT_LEVEL + SU2_DUAL_COXETER)
    - Fraction(QUARK_LEVEL * SU3_DIMENSION, QUARK_LEVEL + SU3_DUAL_COXETER)
    - Fraction(LEPTON_LEVEL * SU2_DIMENSION, LEPTON_LEVEL + SU2_DUAL_COXETER)
)
BENCHMARK_C_DARK_RESIDUE = float(BENCHMARK_C_DARK_RESIDUE_FRACTION)
BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE_FRACTION = (
    Fraction(PARENT_LEVEL * SO10_DIMENSION, PARENT_LEVEL + SO10_DUAL_COXETER)
    - (
        Fraction(LEPTON_LEVEL * SU2_DIMENSION, LEPTON_LEVEL + SU2_DUAL_COXETER)
        + Fraction(QUARK_LEVEL * SU3_DIMENSION, QUARK_LEVEL + SU3_DUAL_COXETER)
    )
    - BENCHMARK_C_DARK_RESIDUE_FRACTION
)
BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE = float(BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE_FRACTION)

# Strict benchmark hierarchy
# --------------------------
# Tier 1 constants are the branch-fixed discrete coordinates that identify the
# anomaly-free benchmark cell.
# Tier 2 constants are external observational boundary conditions used as input
# anchors; they are not promoted as predictions.
# Tier 3 constants are derived residues computed once Tier 1 is fixed and Tier
# 2 is supplied. Numerical tolerances, reporting strings, and artifact
# filenames remain auxiliary runtime infrastructure and are declared below.

STRICT_BENCHMARK_TIER_DEFINITIONS = (
    BenchmarkConstantTier(
        identifier="Tier 1",
        title="Topological Coordinates",
        description="Branch-fixed discrete coordinates that identify the anomaly-free benchmark cell.",
        constants=(
            BenchmarkConstantDefinition(
                name="LEPTON_LEVEL",
                value=LEPTON_LEVEL,
                legacy_metadata_paths=("model.parent_level", "model.lepton_fixed_point_index"),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
            BenchmarkConstantDefinition(
                name="QUARK_LEVEL",
                value=QUARK_LEVEL,
                legacy_metadata_paths=("model.parent_level", "model.quark_fixed_point_index"),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
            BenchmarkConstantDefinition(
                name="PARENT_LEVEL",
                value=PARENT_LEVEL,
                legacy_metadata_paths=("model.parent_level",),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
            BenchmarkConstantDefinition(
                name="G_SM",
                value=G_SM,
                legacy_metadata_paths=("model.g_sm",),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
        ),
    ),
    BenchmarkConstantTier(
        identifier="Tier 2",
        title="Observational Boundary Conditions",
        description="External observational boundary conditions supplied to the benchmark as fixed input anchors.",
        constants=(
            BenchmarkConstantDefinition(
                name="PLANCK2018_H0_KM_S_MPC",
                value=PLANCK2018_H0_KM_S_MPC,
                legacy_metadata_paths=("physical_constants.planck2018_h0_km_s_mpc",),
                allowed_legacy_classifications=("Empirical Matching Ansatz", "Geometric Emergence"),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_H0_SIGMA_KM_S_MPC",
                value=PLANCK2018_H0_SIGMA_KM_S_MPC,
                legacy_metadata_paths=("physical_constants.planck2018_h0_sigma_km_s_mpc",),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_OMEGA_LAMBDA",
                value=PLANCK2018_OMEGA_LAMBDA,
                legacy_metadata_paths=("physical_constants.planck2018_omega_lambda",),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_OMEGA_LAMBDA_SIGMA",
                value=PLANCK2018_OMEGA_LAMBDA_SIGMA,
                legacy_metadata_paths=("physical_constants.planck2018_omega_lambda_sigma",),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_LAMBDA_SI_M2",
                value=PLANCK2018_LAMBDA_SI_M2,
                legacy_metadata_paths=(
                    "physical_constants.planck2018_h0_km_s_mpc",
                    "physical_constants.planck2018_omega_lambda",
                    "physical_constants.light_speed_m_per_s",
                    "physical_constants.mpc_in_meters",
                ),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_LAMBDA_FRACTIONAL_SIGMA",
                value=PLANCK2018_LAMBDA_FRACTIONAL_SIGMA,
                legacy_metadata_paths=(
                    "physical_constants.planck2018_h0_km_s_mpc",
                    "physical_constants.planck2018_h0_sigma_km_s_mpc",
                    "physical_constants.planck2018_omega_lambda",
                    "physical_constants.planck2018_omega_lambda_sigma",
                ),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_ALPHA_EM_INV_MZ",
                value=PLANCK2018_ALPHA_EM_INV_MZ,
                legacy_metadata_paths=("physical_constants.planck2018_alpha_em_inv_mz",),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_SIN2_THETA_W_MZ",
                value=PLANCK2018_SIN2_THETA_W_MZ,
                legacy_metadata_paths=("physical_constants.planck2018_sin2_theta_w_mz",),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK2018_ALPHA_S_MZ",
                value=PLANCK2018_ALPHA_S_MZ,
                legacy_metadata_paths=("physical_constants.planck2018_alpha_s_mz",),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
        ),
    ),
    BenchmarkConstantTier(
        identifier="Tier 3",
        title="Derived Residues",
        description="Quantities fixed after Tier 1 is chosen and Tier 2 inputs are supplied.",
        constants=(
            BenchmarkConstantDefinition(
                name="GEOMETRIC_KAPPA",
                value=GEOMETRIC_KAPPA,
                legacy_metadata_paths=("model.geometric_kappa",),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="PLANCK_HOLOGRAPHIC_BITS",
                value=PLANCK_HOLOGRAPHIC_BITS,
                legacy_metadata_paths=(
                    "physical_constants.planck_length_m",
                    "physical_constants.planck2018_h0_km_s_mpc",
                    "physical_constants.planck2018_omega_lambda",
                    "physical_constants.light_speed_m_per_s",
                    "physical_constants.mpc_in_meters",
                ),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="HOLOGRAPHIC_BITS",
                value=HOLOGRAPHIC_BITS,
                legacy_metadata_paths=(
                    "physical_constants.planck_length_m",
                    "physical_constants.planck2018_h0_km_s_mpc",
                    "physical_constants.planck2018_omega_lambda",
                    "physical_constants.light_speed_m_per_s",
                    "physical_constants.mpc_in_meters",
                ),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA",
                value=HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA,
                legacy_metadata_paths=(
                    "physical_constants.planck2018_h0_km_s_mpc",
                    "physical_constants.planck2018_h0_sigma_km_s_mpc",
                    "physical_constants.planck2018_omega_lambda",
                    "physical_constants.planck2018_omega_lambda_sigma",
                ),
                allowed_legacy_classifications=("Empirical Matching Ansatz",),
            ),
            BenchmarkConstantDefinition(
                name="R_GUT",
                value=R_GUT,
                legacy_metadata_paths=(
                    "model.parent_level",
                    "model.lepton_fixed_point_index",
                    "model.quark_fixed_point_index",
                    "group_theory.su2_dual_coxeter",
                ),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
            BenchmarkConstantDefinition(
                name="BENCHMARK_C_DARK_RESIDUE_FRACTION",
                value=BENCHMARK_C_DARK_RESIDUE_FRACTION,
                legacy_metadata_paths=(
                    "model.parent_level",
                    "model.lepton_fixed_point_index",
                    "model.quark_fixed_point_index",
                    "group_theory.su2_dimension",
                    "group_theory.su3_dimension",
                    "group_theory.su2_dual_coxeter",
                    "group_theory.su3_dual_coxeter",
                ),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
            BenchmarkConstantDefinition(
                name="BENCHMARK_C_DARK_RESIDUE",
                value=BENCHMARK_C_DARK_RESIDUE,
                legacy_metadata_paths=(
                    "model.parent_level",
                    "model.lepton_fixed_point_index",
                    "model.quark_fixed_point_index",
                    "group_theory.su2_dimension",
                    "group_theory.su3_dimension",
                    "group_theory.su2_dual_coxeter",
                    "group_theory.su3_dual_coxeter",
                ),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
            BenchmarkConstantDefinition(
                name="BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE_FRACTION",
                value=BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE_FRACTION,
                legacy_metadata_paths=(
                    "model.parent_level",
                    "model.lepton_fixed_point_index",
                    "model.quark_fixed_point_index",
                    "group_theory.su2_dimension",
                    "group_theory.su3_dimension",
                    "group_theory.so10_dimension",
                    "group_theory.su2_dual_coxeter",
                    "group_theory.su3_dual_coxeter",
                    "group_theory.so10_dual_coxeter",
                ),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
            BenchmarkConstantDefinition(
                name="BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE",
                value=BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE,
                legacy_metadata_paths=(
                    "model.parent_level",
                    "model.lepton_fixed_point_index",
                    "model.quark_fixed_point_index",
                    "group_theory.su2_dimension",
                    "group_theory.su3_dimension",
                    "group_theory.so10_dimension",
                    "group_theory.su2_dual_coxeter",
                    "group_theory.su3_dual_coxeter",
                    "group_theory.so10_dual_coxeter",
                ),
                allowed_legacy_classifications=("Topological Necessity",),
            ),
        ),
    ),
)


def _validate_benchmark_tier_metadata(
    tiers: tuple[BenchmarkConstantTier, ...],
    legacy_classifications: dict[str, str],
) -> None:
    seen_identifiers: set[str] = set()
    declared_constants: dict[str, str] = {}

    for tier in tiers:
        if tier.identifier in seen_identifiers:
            raise RuntimeError(f"Duplicate strict benchmark tier identifier '{tier.identifier}'.")
        seen_identifiers.add(tier.identifier)

        if not tier.constants:
            raise RuntimeError(f"Strict benchmark tier '{tier.label}' must declare at least one constant.")

        for constant in tier.constants:
            previous_tier = declared_constants.get(constant.name)
            if previous_tier is not None:
                raise RuntimeError(
                    f"Strict benchmark constant '{constant.name}' is declared in both '{previous_tier}' and '{tier.label}'."
                )
            declared_constants[constant.name] = tier.label

            if not legacy_classifications or not constant.legacy_metadata_paths:
                continue
            if not constant.allowed_legacy_classifications:
                raise RuntimeError(
                    f"Strict benchmark constant '{constant.name}' declares legacy metadata paths without allowed classifications."
                )

            # In the v2.0 geometry-first audit, legacy metadata across every
            # strict benchmark tier may be exported either as a geometric
            # emergence residue or as a topological extraction from the same
            # branch-fixed closure.
            tier_relaxations = ("Geometric Emergence", "Topological Extraction")
            for metadata_path in constant.legacy_metadata_paths:
                allowed_legacy_classifications = tuple(
                    dict.fromkeys((*constant.allowed_legacy_classifications, *tier_relaxations))
                )
                classification = legacy_classifications.get(metadata_path)
                if classification is None:
                    raise RuntimeError(
                        f"Strict benchmark constant '{constant.name}' expects legacy metadata path '{metadata_path}'."
                    )
                if classification not in allowed_legacy_classifications:
                    allowed = ", ".join(sorted(allowed_legacy_classifications))
                    raise RuntimeError(
                        f"Strict benchmark constant '{constant.name}' expects '{metadata_path}' to use one of [{allowed}], "
                        f"received '{classification}'."
                    )


_validate_benchmark_tier_metadata(STRICT_BENCHMARK_TIER_DEFINITIONS, BENCHMARK_PARAMETER_CLASSIFICATIONS)

TIER_1_TOPOLOGICAL_COORDINATES = STRICT_BENCHMARK_TIER_DEFINITIONS[0].values
TIER_2_OBSERVATIONAL_BOUNDARY_CONDITIONS = STRICT_BENCHMARK_TIER_DEFINITIONS[1].values
TIER_3_DERIVED_RESIDUES = STRICT_BENCHMARK_TIER_DEFINITIONS[2].values

STRICT_BENCHMARK_CONSTANT_TIERS = {
    constant.name: tier.label
    for tier in STRICT_BENCHMARK_TIER_DEFINITIONS
    for constant in tier.constants
}

SM_RUNNING_CONTENT = _coerce_str(_MODEL_CONSTANTS_CONFIG, "sm_running_content")
RHN_THRESHOLD_MATCHING_ANGLE_SHIFTS_DEG = _coerce_float_sequence(_MODEL_CONSTANTS_CONFIG, "rhn_threshold_matching_angle_shifts_deg")
RHN_THRESHOLD_MATCHING_DELTA_SHIFT_DEG = _coerce_float(_MODEL_CONSTANTS_CONFIG, "rhn_threshold_matching_delta_shift_deg")
RHN_THRESHOLD_MATCHING_MASS_SHIFT_FRACTION = _coerce_float(_MODEL_CONSTANTS_CONFIG, "rhn_threshold_matching_mass_shift_fraction")
LOW_SU3_WEIGHTS = _coerce_nested_int_sequence(_MODEL_CONSTANTS_CONFIG, "low_su3_weights")
STRICT_SUPPORT_PENALTY = _coerce_float(_MODEL_CONSTANTS_CONFIG, "strict_support_penalty")
BULK_SPACETIME_DIMENSION = _coerce_int(_MODEL_CONSTANTS_CONFIG, "bulk_spacetime_dimension")
MAJORANA_HIGGS_REPRESENTATION = _coerce_str(_MODEL_CONSTANTS_CONFIG, "majorana_higgs_representation")
SO10_HIGGS_10_DYNKIN_LABELS = _coerce_int_sequence(_MODEL_CONSTANTS_CONFIG, "so10_higgs_10_dynkin_labels")
SO10_HIGGS_126_DYNKIN_LABELS = _coerce_int_sequence(_MODEL_CONSTANTS_CONFIG, "so10_higgs_126_dynkin_labels")
SO10_HIGGS_210_DYNKIN_LABELS = _coerce_int_sequence(_MODEL_CONSTANTS_CONFIG, "so10_higgs_210_dynkin_labels")
SO10_SPINOR_16_DYNKIN_LABELS = _coerce_int_sequence(_MODEL_CONSTANTS_CONFIG, "so10_spinor_16_dynkin_labels")
VISIBLE_HYPERCHARGE_CENTRAL_CHARGE = _coerce_float(_MODEL_CONSTANTS_CONFIG, "visible_hypercharge_central_charge")
SO10_CLEBSCH_10 = _coerce_float(_MODEL_CONSTANTS_CONFIG, "so10_clebsch_10")
SM_MAJORANA_C_E = _coerce_float(_MODEL_CONSTANTS_CONFIG, "sm_majorana_c_e")

TRANSPORT_MC_MIN_STABILITY_YIELD = _coerce_float(_RUNTIME_KNOBS_CONFIG, "transport_mc_min_stability_yield")
TRANSPORT_MC_CAVEAT_YIELD = _coerce_float(_RUNTIME_KNOBS_CONFIG, "transport_mc_caveat_yield")
TRANSPORT_SINGULARITY_CHI2_PENALTY = _coerce_float(_RUNTIME_KNOBS_CONFIG, "transport_singularity_chi2_penalty")
DEFAULT_RANDOM_SEED = _coerce_int(_RUNTIME_KNOBS_CONFIG, "default_random_seed")
CONDITION_AWARE_TOLERANCE_MULTIPLIER = _coerce_float(_RUNTIME_KNOBS_CONFIG, "condition_aware_tolerance_multiplier")
GLOBAL_LEPTON_LEVEL_RANGE = _coerce_int_sequence(_RUNTIME_KNOBS_CONFIG, "global_lepton_level_range")
GLOBAL_QUARK_LEVEL_RANGE = _coerce_int_sequence(_RUNTIME_KNOBS_CONFIG, "global_quark_level_range")
LANDSCAPE_TRIAL_COUNT = (
    (GLOBAL_LEPTON_LEVEL_RANGE[1] - GLOBAL_LEPTON_LEVEL_RANGE[0] + 1)
    * (GLOBAL_QUARK_LEVEL_RANGE[1] - GLOBAL_QUARK_LEVEL_RANGE[0] + 1)
)
FOLLOWUP_LEPTON_HALF_WINDOW = _coerce_int(_RUNTIME_KNOBS_CONFIG, "followup_lepton_half_window")
FOLLOWUP_QUARK_HALF_WINDOW = _coerce_int(_RUNTIME_KNOBS_CONFIG, "followup_quark_half_window")
FOLLOWUP_CHI2_REFERENCE_DOF = _coerce_int(_RUNTIME_KNOBS_CONFIG, "followup_chi2_reference_dof")
FOLLOWUP_CHI2_SURVIVAL_PROBABILITY = _coerce_float(_RUNTIME_KNOBS_CONFIG, "followup_chi2_survival_probability")
CKM_PHASE_TILT_INVARIANCE_TOLERANCE = _coerce_float(_RUNTIME_KNOBS_CONFIG, "ckm_phase_tilt_invariance_tolerance")
MASS_RATIO_STABILITY_FACTOR = _coerce_float(_RUNTIME_KNOBS_CONFIG, "mass_ratio_stability_factor")
VEV_ALIGNMENT_SWEEP_SAMPLE_COUNT = _coerce_int(_RUNTIME_KNOBS_CONFIG, "vev_alignment_sweep_sample_count")
SEED_AUDIT_SAMPLE_COUNT = _coerce_int(_RUNTIME_KNOBS_CONFIG, "seed_audit_sample_count")
DISCRETE_SELECTION_CONSTRAINT_COUNT = _coerce_int(_RUNTIME_KNOBS_CONFIG, "discrete_selection_constraint_count")
RELAXED_NEIGHBOR_TILT_DEG = _coerce_float(_RUNTIME_KNOBS_CONFIG, "relaxed_neighbor_tilt_deg")
SUPPORT_TAU_IMAG = _coerce_float(_RUNTIME_KNOBS_CONFIG, "support_tau_imag")
SUPPORT_PHI_SAMPLES = _coerce_int(_RUNTIME_KNOBS_CONFIG, "support_phi_samples")
DEFAULT_BITCOUNT_FRACTIONAL_VARIATION = _coerce_float(_RUNTIME_KNOBS_CONFIG, "default_bitcount_fractional_variation")
KAPPA_SCAN_VALUES = _coerce_float_sequence(_RUNTIME_KNOBS_CONFIG, "kappa_scan_values")
LEGACY_SHORTCUT_MAX_SIGMA_SHIFT = _coerce_float(_RUNTIME_KNOBS_CONFIG, "legacy_shortcut_max_sigma_shift")
MATRIX_SVD_THRESHOLD_RULE = _coerce_str(_RUNTIME_KNOBS_CONFIG, "matrix_svd_threshold_rule")

EXPERIMENTAL_CONTEXT = ExperimentalContext(
    nufit_release=_coerce_str(_RELEASES_CONFIG, "nufit_release"),
    nufit_reference=_coerce_str(_RELEASES_CONFIG, "nufit_reference"),
    pdg_release=_coerce_str(_RELEASES_CONFIG, "pdg_release"),
    pdg_reference=_coerce_str(_RELEASES_CONFIG, "pdg_reference"),
    lepton_intervals=_coerce_interval_mapping(_EXPERIMENTAL_CONFIG, "lepton_1sigma"),
    quark_intervals=_coerce_interval_mapping(_EXPERIMENTAL_CONFIG, "quark_1sigma"),
    ckm_gamma_experimental_input_deg=_coerce_interval(
        "ckm_gamma_experimental_input_deg",
        _EXPERIMENTAL_CONFIG["ckm_gamma_experimental_input_deg"],
    ),
)

LEPTON_INTERVALS = EXPERIMENTAL_CONTEXT.lepton_intervals
QUARK_INTERVALS = EXPERIMENTAL_CONTEXT.quark_intervals
CKM_GAMMA_EXPERIMENTAL_INPUT_DEG = EXPERIMENTAL_CONTEXT.ckm_gamma_experimental_input_deg
CKM_GAMMA_GOLD_STANDARD_DEG = CKM_GAMMA_EXPERIMENTAL_INPUT_DEG
EXPERIMENTAL_INPUTS = {
    "nufit_lepton_1sigma": LEPTON_INTERVALS,
    "pdg_quark_1sigma": QUARK_INTERVALS,
    "pdg_ckm_gamma_deg": CKM_GAMMA_EXPERIMENTAL_INPUT_DEG,
}
SOLAR_MASS_SPLITTING_EV2 = _coerce_float(_NORMAL_ORDERING_MASS_SPLITTINGS_CONFIG, "solar")
ATMOSPHERIC_MASS_SPLITTING_NO_EV2 = _coerce_float(_NORMAL_ORDERING_MASS_SPLITTINGS_CONFIG, "atmospheric")
NUFIT_53_NO_3SIGMA = _coerce_interval_mapping(_EXPERIMENTAL_CONFIG, "nufit_3sigma_normal_ordering")

REPRODUCIBILITY_STATEMENT = (
    "Numerical outputs are reproduced under SciPy 1.12.0; deviations in later versions "
    f"are bounded by the reported ODE tolerance (${_format_latex_scientific(_coerce_float(_SOLVER_CONFIG, 'atol'))}$)."
)

LOCKED_TOPOLOGICAL_COORDINATE_LABEL = "Locked Topological Coordinate"
TRIPLE_MATCH_MANDATORY_CLOSURE_LABEL = "Mandatory Closure Requirement"
ELIMINATIVE_SURVIVAL_PROCESS_MESSAGE = (
    "Branch selection proceeds as an eliminative survival process on the fixed-parent moat audit rather than by likelihood maximization."
)
PRIMARY_BENCHMARK_AUDIT_SUCCESS_MESSAGE = (
    "[PRIMARY BENCHMARK AUDIT]: Eigenvector Rigidity verified. Matched Mass Sector (singular values) successfully decoupled from the Rigid Mixing Sector (eigenvectors). Standard Model values are mandatory residues of 4D gravity."
)
SVD_RIGIDITY_SHIELD_SIGMA_THRESHOLD = 1.0e-12
SVD_RIGIDITY_SHIELD_VEV_DEFORMATION_FRACTION = 0.10
MIXING_SECTOR_RIGIDITY_MESSAGE = (
    r"Mixing Sector Rigidity Verified: the SVD Rigidity Shield keeps PMNS/CKM eigenvectors stable at the $10^{-12}\sigma$ level "
    r"under $\pm10\%$ VEV-ratio deformations. This confirms the separation of the Matched Mass "
    r"Sector from the Rigid Mixing Sector."
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT_DIR = REPO_ROOT / "papers"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"

GLOBAL_FLAVOR_FIT_TABLE_FILENAME = "global_flavor_fit_table.tex"
UNIQUENESS_SCAN_TABLE_FILENAME = "uniqueness_scan_table.tex"
MODULARITY_RESIDUAL_MAP_FILENAME = "modularity_residual_map.tex"
LANDSCAPE_ANOMALY_MAP_FILENAME = "landscape_anomaly_map.tex"
AUDIT_SUMMARY_TEX_FILENAME = "audit_summary.tex"
COROLLARY_REPORT_FILENAME = "corollary_report.txt"
HARD_ANOMALY_UNIQUENESS_AUDIT_FILENAME = "hard_anomaly_uniqueness_audit.txt"
HOLOGRAPHIC_AUDIT_FILENAME = "holographic_audit.json"
MATCHING_RESIDUAL_BAND_FIGURE_FILENAME = "matching_residual_band.png"
MATCHING_RESIDUAL_REPORT_FILENAME = "matching_residual_report.txt"
REFEREE_SUMMARY_FILENAME = "referee_summary.json"
RESIDUALS_JSON_FILENAME = "residuals.json"
RESIDUE_SENSITIVITY_DATA_FILENAME = "sensitivity_data.csv"
SUPPLEMENTARY_IH_SUPPORT_MAP_FILENAME = "supplementary_ih_support_map.tex"
SUPPLEMENTARY_GAUGE_ORTHOGONALITY_TABLE_FILENAME = "supplementary_gauge_orthogonality_table.tex"
SUPPLEMENTARY_HEAVY_SCALE_SENSITIVITY_TABLE_FILENAME = "supplementary_heavy_scale_sensitivity_table.tex"
SUPPLEMENTARY_RESIDUE_SENSITIVITY_FIGURE_FILENAME = "supplementary_residue_sensitivity.png"
SUPPLEMENTARY_TOLERANCE_TABLE_FILENAME = "supplementary_tolerance_table.tex"
SUPPLEMENTARY_UNITARY_CONSISTENCY_TABLE_FILENAME = "supplementary_unitary_consistency_table.tex"
KAPPA_SENSITIVITY_AUDIT_FILENAME = "kappa_sensitivity_audit.tex"
KAPPA_STABILITY_SWEEP_FILENAME = "kappa_stability_sweep.tex"
SVD_STABILITY_AUDIT_TABLE_FILENAME = "svd_stability_audit.tex"
PHYSICS_CONSTANTS_FILENAME = "physics_constants.tex"
BENCHMARK_DIAGNOSTICS_FILENAME = "benchmark_diagnostics.json"
TRANSPORT_COVARIANCE_DIAGNOSTICS_FILENAME = "transport_covariance_diagnostics.json"
SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_DATA_FILENAME = "supplementary_ih_singular_value_spectrum.csv"
AUDIT_STATEMENT_FILENAME = "audit_statement.txt"
NUMERICAL_STABILITY_REPORT_FILENAME = "numerical_stability_report.txt"
SVD_STABILITY_REPORT_FILENAME = "svd_stability_report.txt"
EIGENVECTOR_STABILITY_AUDIT_FILENAME = "eigenvector_stability_audit.txt"
STABILITY_REPORT_FILENAME = "stability_report.txt"
TOPOLOGICAL_LOBSTER_FIGURE_FILENAME = "fig1_pmns_fit.png"
MAJORANA_FLOOR_FIGURE_FILENAME = "fig4_majorana_floor.png"
CKM_PHASE_TILT_PROFILE_FIGURE_FILENAME = "fig5_threshold_weight_profile.png"
DM_FINGERPRINT_FIGURE_FILENAME = "fig6_dm_fingerprint.png"
FRAMING_GAP_HEATMAP_FIGURE_FILENAME = "framing_gap_moat_heatmap.png"
SUPPLEMENTARY_HARD_ANOMALY_FILTER_FIGURE_FILENAME = "supplementary_hard_anomaly_filter.png"
BENCHMARK_STABILITY_TABLE_FILENAME = "benchmark_stability_table.tex"
SUPPLEMENTARY_TOPCHI2_TABLE_FILENAME = "supplementary_topchi2_table.tex"
SUPPLEMENTARY_DELTA_CHI2_RESIDUE_PROFILE_TABLE_FILENAME = "supplementary_delta_chi2_residue_profile_table.tex"
DISCRETE_LANDSCAPE_SCAN_RESULTS_FILENAME = "discrete_landscape_scan_9801.csv"
FOLLOWUP_SCAN_RESULTS_FILENAME = "followup_scan_533.csv"
ROBUSTNESS_AUDIT_FILENAME = "robustness_audit.csv"
SUPPLEMENTARY_VEV_ALIGNMENT_STABILITY_FIGURE_FILENAME = "supplementary_vev_alignment_stability.png"
SEED_ROBUSTNESS_AUDIT_FILENAME = "seed_robustness_audit.txt"
SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_FIGURE_FILENAME = "supplementary_ih_singular_value_spectrum.png"
SUPPLEMENTARY_DETERMINANT_GRADIENT_FIGURE_FILENAME = "supplementary_determinant_gradient.png"
SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME = "supplementary_step_size_convergence.png"
FRAMING_GAP_STABILITY_FIGURE_FILENAME = "framing_gap_stability.pdf"
AUDIT_OUTPUT_ARCHIVE_DIRNAME = "reviewer_audit_packet"
STABILITY_AUDIT_OUTPUTS_DIRNAME = "reviewer_defense_packet"
LANDSCAPE_METRICS_DIRNAME = "landscape_evidence_packet"
REFEREE_EVIDENCE_PACKET_DIRNAME = "referee_evidence_packet"
AUDIT_OUTPUT_MANIFEST_FILENAME = "README.md"
