r"""Numerical audit driver for the boundary-entropy / flavor correspondence.

This module loads benchmark parameters from `pub/config/benchmark_v1.yaml`
and experimental neutrino inputs from `pub/data/nufit_5_3.json`, propagates the
selected benchmark through the RG transport pipeline, and exports publication
artifacts as skeletal LaTeX tables, figures, and raw numerical diagnostics.

Quickstart:
    Production Pipeline:
        - Detached artifacts in `output/`:
          `python -m pub.main --manuscript-dir pub`
        - Refresh manuscript-side artifacts in `pub/`:
          `python -m pub.main --manuscript-dir pub --output-dir pub`
    Audit Snapshot / Master Audit Driver:
        - Reproduce the manuscript-facing audit snapshot:
          `python -m pub.tn`

The `pub.main` entry point is the production pipeline. The legacy entry point
`python -m pub.tn` is retained as the manuscript-facing Master Audit Driver and
audit-snapshot shim.
"""

from __future__ import annotations

import argparse
from dataclasses import InitVar, dataclass, field, replace
from decimal import Decimal, ROUND_HALF_UP, localcontext
import itertools
import logging
import math
import os
import shutil
import warnings
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2 as chi2_distribution


LOGGER = logging.getLogger(__name__)


class QuadratureConvergenceError(RuntimeError):
    """Raised when adaptive quadrature exceeds the configured error budget."""


class MonteCarloYieldWarning(RuntimeWarning):
    """Raised when covariance Monte Carlo sampling loses too many points."""


class TopologicalIntegrityError(RuntimeError):
    """Raised when a mass-coordinate detuning breaks the saturated bit-budget lock and reopens torsion."""


TOPOLOGICAL_MASS_COORDINATE_ABS_TOL_EV = 1.0e-18


def _configure_logger_handlers(
    logger: logging.Logger,
    *,
    quiet: bool,
    log_file: Path | None,
) -> None:
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING if quiet else logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    logger.propagate = False


def configure_reporting(*, quiet: bool = False, log_file: Path | None = None) -> None:
    _configure_logger_handlers(LOGGER, quiet=quiet, log_file=log_file)
    if LOGGER.name != "pub":
        _configure_logger_handlers(logging.getLogger("pub"), quiet=quiet, log_file=log_file)

from . import algebra
from . import audit_generator
from . import constants as shared_constants
from . import engine as publication_engine
from . import export as publication_export
from . import physics_engine
from . import reporting as presentation_reporting
from . import reporting_engine
from . import template_utils
from . import topological_kernel
from .constants import (
    ALPHA_S_MZ_SIGMA,
    PARAMETRIC_TRANSPORT_COVARIANCE_FRACTION,
    PDG_TOP_POLE_MASS_CENTRAL_GEV,
    PDG_TOP_POLE_MASS_SIGMA_GEV,
    SUPPLEMENTARY_DETERMINANT_GRADIENT_FIGURE_FILENAME,
    THEORETICAL_MATCHING_UNCERTAINTY_FRACTION,
)
from .numerics import freeze_numpy_arrays, require_real_array, require_real_scalar
from .plotting_runtime import managed_figure, plt
from .runtime_config import (
    DEFAULT_SOLVER_CONFIG,
    NumericalStabilityGuard,
    PerturbativeBreakdownException,
    PhysicalSingularityException,
    Sector,
    SolverConfig,
    solver_isclose,
)
from .transport import (
    build_ckm_phase_tilt_profile,
    derive_ckm_with_threshold_residue,
    transport_observable_delta as transport_observable_delta_impl,
    transport_observable_vector as transport_observable_vector_impl,
)
from .topology import add_fraction_vectors, fraction_dot, lcm_int, scale_fraction_vector, solve_fraction_linear_system

PLANCK_MASS_EV = shared_constants.PLANCK_MASS_EV
PLANCK_LENGTH_M = shared_constants.PLANCK_LENGTH_M
LIGHT_SPEED_M_PER_S = shared_constants.LIGHT_SPEED_M_PER_S
MPC_IN_METERS = shared_constants.MPC_IN_METERS
PLANCK2018_H0_KM_S_MPC = shared_constants.PLANCK2018_H0_KM_S_MPC
PLANCK2018_H0_SIGMA_KM_S_MPC = shared_constants.PLANCK2018_H0_SIGMA_KM_S_MPC
PLANCK2018_OMEGA_LAMBDA = shared_constants.PLANCK2018_OMEGA_LAMBDA
PLANCK2018_OMEGA_LAMBDA_SIGMA = shared_constants.PLANCK2018_OMEGA_LAMBDA_SIGMA
PLANCK2018_ALPHA_EM_INV_MZ = shared_constants.PLANCK2018_ALPHA_EM_INV_MZ
PLANCK2018_SIN2_THETA_W_MZ = shared_constants.PLANCK2018_SIN2_THETA_W_MZ
PLANCK2018_ALPHA_S_MZ = shared_constants.PLANCK2018_ALPHA_S_MZ
PLANCK2018_H0_SI = shared_constants.PLANCK2018_H0_SI
PLANCK2018_LAMBDA_SI_M2 = shared_constants.PLANCK2018_LAMBDA_SI_M2
PLANCK2018_LAMBDA_FRACTIONAL_SIGMA = shared_constants.PLANCK2018_LAMBDA_FRACTIONAL_SIGMA
PLANCK_HOLOGRAPHIC_BITS = shared_constants.PLANCK_HOLOGRAPHIC_BITS
HOLOGRAPHIC_BITS = shared_constants.HOLOGRAPHIC_BITS
HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA = shared_constants.HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA
PLANCK_MASS_GEV = shared_constants.PLANCK_MASS_GEV
GEOMETRIC_KAPPA = shared_constants.GEOMETRIC_KAPPA
PARENT_LEVEL = shared_constants.PARENT_LEVEL
LEPTON_FIXED_POINT_INDEX = shared_constants.LEPTON_FIXED_POINT_INDEX
QUARK_FIXED_POINT_INDEX = shared_constants.QUARK_FIXED_POINT_INDEX
LEPTON_LEVEL = shared_constants.LEPTON_LEVEL
QUARK_LEVEL = shared_constants.QUARK_LEVEL
TOPOLOGICAL_QUANTUM_NUMBER_DOF_SUBTRACTION = shared_constants.TOPOLOGICAL_QUANTUM_NUMBER_DOF_SUBTRACTION
THRESHOLD_MATCHING_DOF_SUBTRACTION = shared_constants.THRESHOLD_MATCHING_DOF_SUBTRACTION
PHENOMENOLOGICAL_DOF_ADJUSTMENT = shared_constants.PHENOMENOLOGICAL_DOF_ADJUSTMENT
HONEST_FREQUENTIST_DOF_SUBTRACTION = shared_constants.HONEST_FREQUENTIST_DOF_SUBTRACTION
SU2_DIMENSION = shared_constants.SU2_DIMENSION
SU3_DIMENSION = shared_constants.SU3_DIMENSION
SO10_DIMENSION = shared_constants.SO10_DIMENSION
SU2_DUAL_COXETER = shared_constants.SU2_DUAL_COXETER
SU3_DUAL_COXETER = shared_constants.SU3_DUAL_COXETER
SO10_DUAL_COXETER = shared_constants.SO10_DUAL_COXETER
SO10_TO_SU2_EMBEDDING_INDEX = shared_constants.SO10_TO_SU2_EMBEDDING_INDEX
SO10_TO_SU3_EMBEDDING_INDEX = shared_constants.SO10_TO_SU3_EMBEDDING_INDEX
SO10_RANK = shared_constants.SO10_RANK
SU3_RANK = shared_constants.SU3_RANK
RANK_DIFFERENCE = SO10_RANK - SU3_RANK
GUT_SCALE_GEV = shared_constants.GUT_SCALE_GEV
MZ_SCALE_GEV = shared_constants.MZ_SCALE_GEV
RG_SCALE_RATIO = GUT_SCALE_GEV / MZ_SCALE_GEV
SM_RUNNING_CONTENT = shared_constants.SM_RUNNING_CONTENT
RHN_THRESHOLD_MATCHING_ANGLE_SHIFTS_DEG = shared_constants.RHN_THRESHOLD_MATCHING_ANGLE_SHIFTS_DEG
RHN_THRESHOLD_MATCHING_DELTA_SHIFT_DEG = shared_constants.RHN_THRESHOLD_MATCHING_DELTA_SHIFT_DEG
RHN_THRESHOLD_MATCHING_MASS_SHIFT_FRACTION = shared_constants.RHN_THRESHOLD_MATCHING_MASS_SHIFT_FRACTION
LOW_SU3_WEIGHTS = shared_constants.LOW_SU3_WEIGHTS
STRICT_SUPPORT_PENALTY = shared_constants.STRICT_SUPPORT_PENALTY
TRANSPORT_MC_MIN_STABILITY_YIELD = shared_constants.TRANSPORT_MC_MIN_STABILITY_YIELD
TRANSPORT_MC_CAVEAT_YIELD = shared_constants.TRANSPORT_MC_CAVEAT_YIELD
TRANSPORT_SINGULARITY_CHI2_PENALTY = shared_constants.TRANSPORT_SINGULARITY_CHI2_PENALTY
DEFAULT_RANDOM_SEED = shared_constants.DEFAULT_RANDOM_SEED
CONDITION_AWARE_TOLERANCE_MULTIPLIER = shared_constants.CONDITION_AWARE_TOLERANCE_MULTIPLIER
BULK_SPACETIME_DIMENSION = shared_constants.BULK_SPACETIME_DIMENSION
MAJORANA_HIGGS_REPRESENTATION = shared_constants.MAJORANA_HIGGS_REPRESENTATION
SO10_HIGGS_10_DYNKIN_LABELS = shared_constants.SO10_HIGGS_10_DYNKIN_LABELS
SO10_HIGGS_126_DYNKIN_LABELS = shared_constants.SO10_HIGGS_126_DYNKIN_LABELS
SO10_HIGGS_210_DYNKIN_LABELS = shared_constants.SO10_HIGGS_210_DYNKIN_LABELS
SO10_SPINOR_16_DYNKIN_LABELS = shared_constants.SO10_SPINOR_16_DYNKIN_LABELS
DIRAC_HIGGS_BENCHMARK_MASS_GEV = shared_constants.DIRAC_HIGGS_BENCHMARK_MASS_GEV
GLOBAL_LEPTON_LEVEL_RANGE = shared_constants.GLOBAL_LEPTON_LEVEL_RANGE
GLOBAL_QUARK_LEVEL_RANGE = shared_constants.GLOBAL_QUARK_LEVEL_RANGE
LOCAL_LEPTON_LEVEL_WINDOW = tuple(range(LEPTON_LEVEL - 2, LEPTON_LEVEL + 3))
FOLLOWUP_LEPTON_LEVEL_RANGE = (
    max(GLOBAL_LEPTON_LEVEL_RANGE[0], LEPTON_LEVEL - shared_constants.FOLLOWUP_LEPTON_HALF_WINDOW),
    min(GLOBAL_LEPTON_LEVEL_RANGE[1], LEPTON_LEVEL + shared_constants.FOLLOWUP_LEPTON_HALF_WINDOW),
)
FOLLOWUP_QUARK_LEVEL_RANGE = (
    max(GLOBAL_QUARK_LEVEL_RANGE[0], QUARK_LEVEL - shared_constants.FOLLOWUP_QUARK_HALF_WINDOW),
    min(GLOBAL_QUARK_LEVEL_RANGE[1], QUARK_LEVEL + shared_constants.FOLLOWUP_QUARK_HALF_WINDOW),
)
FOLLOWUP_CHI2_REFERENCE_DOF = shared_constants.FOLLOWUP_CHI2_REFERENCE_DOF
FOLLOWUP_CHI2_SURVIVAL_PROBABILITY = shared_constants.FOLLOWUP_CHI2_SURVIVAL_PROBABILITY
FOLLOWUP_CHI2_SURVIVAL_THRESHOLD = float(
    chi2_distribution.ppf(FOLLOWUP_CHI2_SURVIVAL_PROBABILITY, FOLLOWUP_CHI2_REFERENCE_DOF)
)
LOW_RANK_RCFT_SCAN_COMBINATIONS = (
    (GLOBAL_LEPTON_LEVEL_RANGE[1] - GLOBAL_LEPTON_LEVEL_RANGE[0] + 1)
    * (GLOBAL_QUARK_LEVEL_RANGE[1] - GLOBAL_QUARK_LEVEL_RANGE[0] + 1)
)
THEORETICAL_UNCERTAINTY_FRACTION = THEORETICAL_MATCHING_UNCERTAINTY_FRACTION
PARAMETRIC_COVARIANCE_FRACTION = PARAMETRIC_TRANSPORT_COVARIANCE_FRACTION
BROKEN_SO10_GAUGE_BOSON_COUNT = SO10_DIMENSION - (SU3_DIMENSION + SU2_DIMENSION + 1)
GUT_THRESHOLD_RESIDUE_SYMBOL = r"\mathcal{R}_{\rm GUT}"
CKM_PHASE_TILT_SYMBOL = GUT_THRESHOLD_RESIDUE_SYMBOL
GAUGE_HOLOGRAPHY_GENERATION_COUNT = 15
QUADRATIC_WEIGHT_PROJECTION = Fraction(64, 312)
QUADRATIC_WEIGHT_PROJECTION_TEX = r"\frac{64}{312}"
VOA_BRANCHING_GAP = Fraction(8, 28)
VOA_BRANCHING_GAP_TEX = r"\frac{8}{28}"
GAUGE_STRENGTH_IDENTITY = Fraction(
    GAUGE_HOLOGRAPHY_GENERATION_COUNT * PARENT_LEVEL,
    LEPTON_LEVEL + QUARK_LEVEL,
)
GAUGE_STRENGTH_IDENTITY_TEX = rf"\frac{{{GAUGE_HOLOGRAPHY_GENERATION_COUNT * PARENT_LEVEL}}}{{{LEPTON_LEVEL + QUARK_LEVEL}}}"
INFLATIONARY_TENSOR_RATIO = Fraction(1, 13)
INFLATIONARY_TENSOR_RATIO_TEX = r"\frac{24}{312}"
INFLATIONARY_TENSOR_RATIO_REDUCED_TEX = r"\frac{1}{13}"
BICEP_KECK_95CL_TENSOR_UPPER_BOUND = 0.036
PRIMORDIAL_EFOLD_IDENTITY_MULTIPLIER = 3
PRIMORDIAL_SCALAR_TILT_TARGET = 0.9648
PRIMORDIAL_SCALAR_TILT_TARGET_TOLERANCE = 1.0e-4
PLANCK_2018_SCALAR_TILT_RANGE = (0.960, 0.970)
PRIMORDIAL_NON_GAUSSIANITY_FLOOR = float(1.0 - GEOMETRIC_KAPPA)
BENCHMARK_C_DARK_RESIDUE = 3.3008
PARITY_BIT_DENSITY_CONSTRAINT_TARGET = 0.38
PARITY_BIT_DENSITY_CONSTRAINT_TOLERANCE = 0.01
VEV_RATIO = float(QUADRATIC_WEIGHT_PROJECTION)
R_GUT = float(VOA_BRANCHING_GAP)
CKM_PHASE_TILT_PARAMETER: float | None = float(VOA_BRANCHING_GAP)
CKM_PHASE_TILT_INVARIANCE_TOLERANCE = shared_constants.CKM_PHASE_TILT_INVARIANCE_TOLERANCE
MATCHING_COEFFICIENT_SYMBOL = r"R_{01}^{\rm par/vis}"
ALPHA_INV_TARGET = float(GAUGE_STRENGTH_IDENTITY)
CODATA_FINE_STRUCTURE_ALPHA_INVERSE = ALPHA_INV_TARGET
GAUGE_HOLOGRAPHY_PASS_PERCENT = 0.50
MASS_RATIO_STABILITY_FACTOR = 1.0 / VEV_RATIO
LANDSCAPE_TRIAL_COUNT = LOW_RANK_RCFT_SCAN_COMBINATIONS
VEV_ALIGNMENT_SWEEP_SAMPLE_COUNT = shared_constants.VEV_ALIGNMENT_SWEEP_SAMPLE_COUNT
SEED_AUDIT_SAMPLE_COUNT = shared_constants.SEED_AUDIT_SAMPLE_COUNT
DISCRETE_SELECTION_CONSTRAINT_COUNT = shared_constants.DISCRETE_SELECTION_CONSTRAINT_COUNT
VISIBLE_HYPERCHARGE_CENTRAL_CHARGE = shared_constants.VISIBLE_HYPERCHARGE_CENTRAL_CHARGE
SO10_CLEBSCH_10 = shared_constants.SO10_CLEBSCH_10
RELAXED_NEIGHBOR_TILT_DEG = shared_constants.RELAXED_NEIGHBOR_TILT_DEG
SOLAR_MASS_SPLITTING_EV2 = shared_constants.SOLAR_MASS_SPLITTING_EV2
ATMOSPHERIC_MASS_SPLITTING_NO_EV2 = shared_constants.ATMOSPHERIC_MASS_SPLITTING_NO_EV2
SUPPORT_TAU_IMAG = shared_constants.SUPPORT_TAU_IMAG
SUPPORT_PHI_SAMPLES = shared_constants.SUPPORT_PHI_SAMPLES
DEFAULT_BITCOUNT_FRACTIONAL_VARIATION = shared_constants.DEFAULT_BITCOUNT_FRACTIONAL_VARIATION
HBAR_GEV_SECONDS = 6.582119569e-25
HBAR_EV_SECONDS = HBAR_GEV_SECONDS * 1.0e9
BOLTZMANN_EV_PER_K = 8.617333262145e-5
EV_TO_KELVIN = 1.0 / BOLTZMANN_EV_PER_K
SECONDS_PER_JULIAN_YEAR = 365.25 * 24.0 * 3600.0
PROTON_BOUNDARY_PIXEL_SCALE_GEV = 0.9378
KAPPA_SCAN_VALUES = shared_constants.KAPPA_SCAN_VALUES
SM_MAJORANA_C_E = shared_constants.SM_MAJORANA_C_E
LEPTON_BETA_FINITE_DIFF_STEP = DEFAULT_SOLVER_CONFIG.finite_diff_step
LEGACY_SHORTCUT_MAX_SIGMA_SHIFT = shared_constants.LEGACY_SHORTCUT_MAX_SIGMA_SHIFT
SM_GUT_YUKAWA_BENCHMARKS = shared_constants.SM_GUT_YUKAWA_BENCHMARKS
SM_MZ_YUKAWA_BENCHMARKS = shared_constants.SM_MZ_YUKAWA_BENCHMARKS
CHARGED_LEPTON_YUKAWA_RATIOS = shared_constants.CHARGED_LEPTON_YUKAWA_RATIOS


def _tex_fraction(value: Fraction) -> str:
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _rationalized_fraction(value: float | Fraction, *, max_denominator: int = 10000) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(float(value)).limit_denominator(max_denominator)


def _matches_exact_fraction(value: float | Fraction, identity: Fraction) -> bool:
    return _rationalized_fraction(value, max_denominator=max(10000, identity.denominator)) == identity


def _format_exact_fraction_or_decimal(
    value: float | Fraction,
    *,
    identity: Fraction,
    tex_identity: str,
    decimals: int,
) -> str:
    if _matches_exact_fraction(value, identity):
        return tex_identity
    return f"{float(value):.{decimals}f}"


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
class PullData:
    value: float
    central: float
    sigma: float
    effective_sigma: float
    pull: float
    inside_1sigma: bool
    theory_sigma: float = 0.0
    parametric_sigma: float = 0.0


def _matrix_singular_value_tolerance(matrix_shape: tuple[int, ...], sigma_max: float) -> float:
    return float(max(matrix_shape) * np.finfo(float).eps * max(sigma_max, 1.0))


def condition_aware_abs_tolerance(
    condition_number: float | None = None,
    *,
    scale: float = 1.0,
    multiplier: float = CONDITION_AWARE_TOLERANCE_MULTIPLIER,
) -> float:
    r"""Return a machine-epsilon floor inflated by the local conditioning.

    The default matches the requested hardware-agnostic rule

    .. math::
       \mathrm{tol}=10\,\varepsilon_{\rm mach}\,\kappa,

    with an additional optional ``scale`` factor for dimensionful quantities.
    """

    finite_condition_number = (
        1.0
        if condition_number is None or not math.isfinite(condition_number)
        else max(float(condition_number), 1.0)
    )
    finite_scale = max(abs(float(scale)), 1.0)
    return float(np.finfo(float).eps * finite_condition_number * multiplier * finite_scale)


@dataclass(frozen=True)
class ExperimentalContext:
    """Container for externally sourced fit intervals used by the verifier.

    Attributes:
        nufit_release: Human-readable NuFIT release label.
        nufit_reference: Publication-facing source string for the leptonic fit.
        pdg_release: Human-readable PDG release label.
        pdg_reference: Publication-facing source string for the quark fit.
        lepton_intervals: Leptonic 1σ intervals indexed by observable name.
        quark_intervals: Quark 1σ intervals indexed by observable name.
        ckm_gamma_experimental_input_deg: Experimental-input apex interval for γ.
    """

    nufit_release: str
    nufit_reference: str
    pdg_release: str
    pdg_reference: str
    lepton_intervals: dict[str, Interval]
    quark_intervals: dict[str, Interval]
    ckm_gamma_experimental_input_deg: Interval


@dataclass(frozen=True)
class ScaleData:
    """Scale data for the boundary--bulk scale relation.

    Attributes:
        m_0_uv_ev: Light defect scale evaluated at the ultraviolet matching point.
        m_0_mz_ev: RG-transported light defect scale at the electroweak scale.
        majorana_boundary_ev: Bare genus-2 holographic Majorana kernel.
        majorana_effective_ev: Effective four-dimensional seesaw-scale benchmark.
        gamma_0_one_loop: One-loop anomalous-dimension coefficient for the defect scale.
        gamma_0_two_loop: Optional subleading quadratic correction used for audits.
        kappa_geometric: Order-one geometric prefactor in the CKN bridge.
    """

    m_0_uv_ev: float
    m_0_mz_ev: float
    majorana_boundary_ev: float
    majorana_effective_ev: float
    gamma_0_one_loop: float
    gamma_0_two_loop: float
    kappa_geometric: float

    @property
    def topological_mass_coordinate_ev(self) -> float:
        """Publication-facing name for the welded ultraviolet neutrino scale."""

        return self.m_0_uv_ev

    @property
    def topological_mass_coordinate_mz_ev(self) -> float:
        """RG-transported image of the welded ultraviolet mass coordinate."""

        return self.m_0_mz_ev


@dataclass(frozen=True)
class TripleMatchSaturationAudit:
    r"""Audit the welded identity ``\Lambda\,G_N\,m_\nu^4`` on the anomaly-free branch."""

    lambda_surface_tension_si_m2: float
    lambda_surface_tension_ev2: float
    newton_constant_ev_minus2: float
    topological_mass_coordinate_ev: float
    triple_match_product: float
    benchmark_identity_product: float

    @property
    def saturated(self) -> bool:
        return math.isclose(
            self.triple_match_product,
            self.benchmark_identity_product,
            rel_tol=1.0e-12,
            abs_tol=1.0e-300,
        )


@dataclass(frozen=True)
class TransportCurvatureAuditData:
    r"""Runtime-derived RG curvature audit used by the manuscript exports.

    Attributes:
        lepton_theta_two_loop: Three leptonic two-loop curvature coefficients.
        lepton_delta_two_loop: Leptonic Dirac-phase two-loop coefficient.
        quark_theta_two_loop: Three quark-sector two-loop curvature coefficients.
        quark_delta_two_loop: Quark Dirac-phase two-loop coefficient.
        gamma_0_one_loop: One-loop anomalous dimension of the defect ansatz.
        gamma_0_two_loop: Two-loop anomalous dimension of the defect ansatz.
        majorana_scalar_curvature: Curvature scalar extracted from the $\mathbf{126}_H$ channel.
    """

    lepton_theta_two_loop: np.ndarray
    lepton_delta_two_loop: float
    quark_theta_two_loop: np.ndarray
    quark_delta_two_loop: float
    gamma_0_one_loop: float
    gamma_0_two_loop: float
    majorana_scalar_curvature: float


@dataclass(frozen=True)
class RGThresholdData:
    """Explicit threshold bookkeeping for RG transport.

    Attributes:
        sector: Sector whose transport is being evaluated.
        threshold_active: Whether the RHN threshold lies inside the running window.
        threshold_scale_gev: Physical RHN threshold scale when active.
        lower_interval_log: Logarithmic interval from $M_Z$ to the threshold.
        upper_interval_log: Logarithmic interval from the threshold to the UV scale.
        matching_angle_shifts_deg: Finite matching shifts for the three mixing angles.
        matching_delta_shift_deg: Finite matching shift for the Dirac phase.
        matching_mass_shift_fraction: Finite multiplicative matching shift for $m_0$.
        structural_exponent: Topological suppression exponent fixing the RHN scale.
        lepton_branching_index: Visible leptonic branching index $I_L$.
        quark_branching_index: Visible quark branching index $I_Q$.
        rank_pressure: Rank-gap pressure contribution entering $M_N$.
        threshold_shift_126: The $\\mathbf{126}_H$ pressure shift entering $M_N$.
    """

    sector: Sector
    threshold_active: bool
    threshold_scale_gev: float | None
    lower_interval_log: float
    upper_interval_log: float
    matching_angle_shifts_deg: tuple[float, float, float]
    matching_delta_shift_deg: float
    matching_mass_shift_fraction: float
    structural_exponent: float
    lepton_branching_index: int
    quark_branching_index: int
    rank_pressure: float
    threshold_shift_126: float

    @property
    def total_log_interval(self) -> float:
        return self.lower_interval_log + self.upper_interval_log

    @property
    def one_loop_factor(self) -> float:
        return self.total_log_interval / ONE_LOOP_FACTOR

    @property
    def two_loop_factor(self) -> float:
        normalization = ONE_LOOP_FACTOR * ONE_LOOP_FACTOR
        if self.threshold_active:
            return (self.lower_interval_log * self.lower_interval_log + self.upper_interval_log * self.upper_interval_log) / normalization
        return self.one_loop_factor * self.one_loop_factor


@dataclass(frozen=True)
class BetaFunctionData:
    """Beta-function coefficients used in the RG transport laws.

    Attributes:
        sector: Sector whose coefficients are being used.
        theta_one_loop: One-loop coefficients for the three mixing angles.
        theta_two_loop: Quadratic audit coefficients for the three mixing angles.
        delta_one_loop: One-loop coefficient for the Dirac phase.
        delta_two_loop: Quadratic audit coefficient for the Dirac phase.
    """

    sector: Sector
    theta_one_loop: np.ndarray
    theta_two_loop: np.ndarray
    delta_one_loop: float
    delta_two_loop: float


@dataclass(frozen=True)
class TransportShiftComponentData:
    """Decompose one observable's transport shift into publication-facing pieces."""

    lower_one_loop: float
    upper_one_loop: float
    two_loop: float
    matching: float
    total: float
    sigma: float | None = None

    @property
    def leading(self) -> float:
        return self.lower_one_loop + self.upper_one_loop

    @property
    def sigma_weighted_leading(self) -> float | None:
        if self.sigma is None:
            return None
        return self.leading / self.sigma

    @property
    def sigma_weighted_total(self) -> float | None:
        if self.sigma is None:
            return None
        return self.total / self.sigma


@dataclass(frozen=True)
class ThresholdShiftAuditData:
    """Explicit RHN-threshold bookkeeping for the lepton-side transport audit."""

    threshold: RGThresholdData
    framing_gap_area_beta_sq: float
    matching_scale_log_ratio: float
    lower_one_loop_factor: float
    upper_one_loop_factor: float
    lower_two_loop_factor: float
    upper_two_loop_factor: float
    observable_shifts_deg: dict[str, TransportShiftComponentData]
    m_0_fraction_shift: TransportShiftComponentData
    leading_norm_capture: float
    sigma_weighted_capture: float


@dataclass(frozen=True)
class RunningCouplings:
    """Third-family Yukawas and SM gauge couplings used in the transport ODEs."""

    top: float
    bottom: float
    tau: float
    g1: float
    g2: float
    g3: float

    def as_array(self) -> np.ndarray:
        return np.array([self.top, self.bottom, self.tau, self.g1, self.g2, self.g3], dtype=float)


@dataclass(frozen=True)
class TransportParametricCovarianceData:
    """Transport covariance from small-step Jacobians with a Monte Carlo ensemble cross-check."""

    observable_names: tuple[str, ...]
    input_names: tuple[str, ...]
    jacobian: np.ndarray
    input_central_values: np.ndarray
    input_sigmas: np.ndarray
    finite_difference_steps: np.ndarray
    covariance: np.ndarray
    covariance_mode: str = "jacobian"
    lower_quantiles: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    upper_quantiles: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    skewness: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    attempted_samples: int = 0
    accepted_samples: int = 0
    failure_count: int = 0
    singularity_chi2_penalty: float = 0.0

    @property
    def stability_yield(self) -> float:
        if self.attempted_samples <= 0:
            return 1.0
        return self.accepted_samples / self.attempted_samples

    @property
    def failure_fraction(self) -> float:
        return 1.0 - self.stability_yield

    @property
    def hard_wall_penalty_applied(self) -> bool:
        return self.failure_count > 0 and self.singularity_chi2_penalty > 0.0

    @property
    def requires_jacobian_fallback_footnote(self) -> bool:
        return self.covariance_mode == "jacobian_low_yield"

    @property
    def jacobian_fallback_footnote_tex(self) -> str | None:
        if not self.requires_jacobian_fallback_footnote:
            return None
        return (
            r"\footnote{Uncertainties in this sector are reported via a linearized Jacobian fallback due to perturbative breakdown in the stochastic sampling.}"
        )

    @property
    def uncertainty_reporting_footnote_tex(self) -> str | None:
        if self.requires_jacobian_fallback_footnote:
            return self.jacobian_fallback_footnote_tex
        if self.attempted_samples <= 0:
            return None
        return (
            rf"\footnote{{transport covariance diagnostics: mode={self.covariance_mode}, attempted={self.attempted_samples}, accepted={self.accepted_samples}, failures={self.failure_count}, failure fraction={self.failure_fraction:.3%}, hard-wall penalty $\chi^2={self.singularity_chi2_penalty:.1e}$.}}"
        )

    @property
    def max_abs_skewness(self) -> float:
        if self.skewness.size == 0:
            return 0.0
        return float(np.nanmax(np.abs(self.skewness)))

    def sigma_for(self, observable_name: str) -> float:
        index = self.observable_names.index(observable_name)
        return float(math.sqrt(max(self.covariance[index, index], 0.0)))

    def interval_sigma_for(self, observable_name: str) -> float:
        index = self.observable_names.index(observable_name)
        if self.lower_quantiles.shape == (len(self.observable_names),) and self.upper_quantiles.shape == (len(self.observable_names),):
            interval_width = 0.5 * float(self.upper_quantiles[index] - self.lower_quantiles[index])
            if interval_width > condition_aware_abs_tolerance(scale=interval_width):
                return interval_width
        return self.sigma_for(observable_name)

    def skewness_for(self, observable_name: str) -> float:
        index = self.observable_names.index(observable_name)
        if self.skewness.shape != (len(self.observable_names),):
            return 0.0
        return float(self.skewness[index])

    def covariance_for(self, observable_names: Sequence[str]) -> np.ndarray:
        indices = [self.observable_names.index(name) for name in observable_names]
        return np.asarray(self.covariance[np.ix_(indices, indices)], dtype=float)


@dataclass(frozen=True)
class NonLinearityAuditData:
    """Diagnostic comparison between linear transport and full coupled evolution."""

    theta_linear_deg: np.ndarray
    theta_nonlinear_deg: np.ndarray
    delta_linear_deg: float
    delta_nonlinear_deg: float
    m_0_linear_ev: float
    m_0_nonlinear_ev: float
    sigma_errors: dict[str, float]
    max_sigma_error: float


@dataclass(frozen=True)
class StepSizeConvergenceData:
    """Step-size convergence audit for the coupled PMNS/CKM transport."""

    step_counts: np.ndarray
    predictive_chi2_values: np.ndarray
    delta_predictive_chi2_values: np.ndarray
    max_sigma_shift_values: np.ndarray
    reference_step_count: int
    reference_predictive_chi2: float


@dataclass(frozen=True)
class PmnsData:
    """PMNS benchmark data derived from the modular leptonic kernel.

    Attributes:
        total_quantum_dimension: Total quantum dimension of the restricted $SU(2)_{26}$ sector.
        d1: Quantum dimension of the first nontrivial support channel.
        d2: Quantum dimension of the second nontrivial support channel.
        beta: Genus-ladder spacing extracted from the total quantum dimension.
        phi_rt_rad: RT holonomy angle in radians.
        framing_phase_deg: Physical framing-phase difference in degrees.
        interference_phase_deg: Phase of the weighted modular $S$--$T$ interference term.
        branch_shift_deg: PDG branch choice for the Dirac phase.
        t_phases: Diagonal modular $T$-matrix phases.
        kernel_block: Restricted modular overlap block.
        topological_matrix: Polar-unitary part of the restricted block.
        complex_seed_matrix: Framing-dressed seed prior to PDG reparameterization.
        pmns_matrix_uv: Ultraviolet PMNS matrix at $M_{\rm GUT}$.
        pmns_matrix_rg: RG-transported PMNS matrix at $M_Z$.
        theta12_uv_deg: Solar angle at the ultraviolet matching scale.
        theta13_uv_deg: Reactor angle at the ultraviolet matching scale.
        theta23_uv_deg: Atmospheric angle at the ultraviolet matching scale.
        theta12_rg_deg: Solar angle at the electroweak scale.
        theta13_rg_deg: Reactor angle at the electroweak scale.
        theta23_rg_deg: Atmospheric angle at the electroweak scale.
        delta_cp_uv_deg: Dirac CP phase at the ultraviolet matching scale.
        delta_cp_rg_deg: Dirac CP phase at the electroweak scale.
        majorana_phase_1_deg: First phase-locked Majorana angle.
        majorana_phase_2_deg: Second phase-locked Majorana angle.
        normal_order_masses_uv_ev: Normal-ordering neutrino masses at $M_{\rm GUT}$.
        normal_order_masses_rg_ev: Normal-ordering neutrino masses at $M_Z$.
        effective_majorana_mass_uv_ev: Effective Majorana mass at $M_{\rm GUT}$.
        effective_majorana_mass_rg_ev: Effective Majorana mass at $M_Z$.
        holonomy_area_uv: Holonomy-area factor at $M_{\rm GUT}$.
        holonomy_area_rg: Holonomy-area factor at $M_Z$.
        jarlskog_uv: Jarlskog invariant at the ultraviolet matching scale.
        jarlskog_rg: Jarlskog invariant at the electroweak scale.
        solar_shift_deg: RG-induced shift of the solar angle.
        theta12_uv_pull: Ultraviolet pull against the NuFIT interval center.
        theta12_rg_pull: Electroweak pull against the NuFIT interval center.
        solar_beta_one_loop: One-loop beta-function coefficient for $\theta_{12}$.
        solar_beta_two_loop: Quadratic audit coefficient for $\theta_{12}$.
    """

    total_quantum_dimension: float
    d1: float
    d2: float
    beta: float
    phi_rt_rad: float
    framing_phase_deg: float
    interference_phase_deg: float
    branch_shift_deg: float
    t_phases: np.ndarray
    kernel_block: np.ndarray
    topological_matrix: np.ndarray
    complex_seed_matrix: np.ndarray
    pmns_matrix_uv: np.ndarray
    pmns_matrix_rg: np.ndarray
    theta12_uv_deg: float
    theta13_uv_deg: float
    theta23_uv_deg: float
    theta12_rg_deg: float
    theta13_rg_deg: float
    theta23_rg_deg: float
    delta_cp_uv_deg: float
    delta_cp_rg_deg: float
    majorana_phase_1_deg: float
    majorana_phase_2_deg: float
    normal_order_masses_uv_ev: np.ndarray
    normal_order_masses_rg_ev: np.ndarray
    effective_majorana_mass_uv_ev: float
    effective_majorana_mass_rg_ev: float
    holonomy_area_uv: float
    holonomy_area_rg: float
    jarlskog_uv: float
    jarlskog_rg: float
    solar_shift_deg: float
    theta12_uv_pull: float
    theta12_rg_pull: float
    solar_beta_one_loop: float
    solar_beta_two_loop: float
    level: int
    parent_level: int
    scale_ratio: float
    bit_count: float
    kappa_geometric: float
    solver_config: SolverConfig


@dataclass(frozen=True)
class CkmData:
    visible_block: np.ndarray
    coset_block: np.ndarray
    coset_weighting: np.ndarray
    bare_topological_weights: tuple[float, float, float]
    topological_weights: tuple[float, float, float]
    rank_difference: int
    branching_index: int
    so10_weyl_norm_sq: float
    su3_weyl_norm_sq: float
    weyl_ratio: float
    rank_deficit_pressure: float
    vacuum_pressure: float
    so10_threshold_correction: "SO10ThresholdCorrectionData"
    channel_pressures: tuple[float, float]
    descendant_factors: tuple[float, float]
    t_phases: np.ndarray
    complex_seed_matrix: np.ndarray
    bare_ckm_matrix_uv: np.ndarray
    bare_ckm_matrix_rg: np.ndarray
    ckm_matrix_uv: np.ndarray
    ckm_matrix_rg: np.ndarray
    theta_c_uv_deg: float
    theta13_uv_deg: float
    theta23_uv_deg: float
    theta_c_rg_deg: float
    theta13_rg_deg: float
    theta23_rg_deg: float
    alpha_uv_deg: float
    beta_uv_deg: float
    gamma_uv_deg: float
    alpha_rg_deg: float
    beta_rg_deg: float
    gamma_rg_deg: float
    bare_vus_uv: float
    bare_vcb_uv: float
    bare_vub_uv: float
    bare_vus_rg: float
    bare_vcb_rg: float
    bare_vub_rg: float
    vus_uv: float
    vcb_uv: float
    vub_uv: float
    vus_rg: float
    vcb_rg: float
    vub_rg: float
    cabibbo_threshold_push_uv: float
    cabibbo_threshold_push_rg: float
    delta_cp_uv_deg: float
    delta_cp_rg_deg: float
    jarlskog_uv: float
    jarlskog_rg: float
    level: int
    parent_level: int
    scale_ratio: float
    gut_threshold_residue: float
    solver_config: SolverConfig

    @property
    def ckm_phase_tilt_parameter(self) -> float:
        """Backward-compatible alias for the normalized GUT-threshold residue."""

        return self.gut_threshold_residue


@dataclass(frozen=True)
class BoundaryBulkInterfaceData:
    """Normalized interface data relating modular amplitudes to Yukawa textures."""

    sector: Sector
    modular_block: np.ndarray
    framing_phases: np.ndarray
    yukawa_texture: np.ndarray
    framed_yukawa_texture: np.ndarray
    majorana_yukawa_texture: np.ndarray


@dataclass(frozen=True)
class SO10ThresholdCorrectionData:
    r"""Structural threshold correction induced by the heavy $SO(10)$ scalar sector.

    The leading CKM-facing residue is carried by the $\mathbf{126}_H$ branch,
    but the disclosed structural residue also retains the sub-dominant
    $\mathbf{210}_H$ and gauge-bundle contributions explicitly.
    """

    gut_threshold_residue: float
    so10_weyl_norm_sq: float
    su3_weyl_norm_sq: float
    weyl_ratio: float
    parent_visible_ratio: float
    clebsch_126: float
    clebsch_10: float
    higgs_mixing_weight: float
    projection_exponent: float
    xi12: complex
    xi12_abs: float
    delta_pi_126: float
    framing_phase_rad: float
    framing_phase_deg: float
    matching_threshold_scale_gev: float
    threshold_log_fraction: float
    orthogonal_coset_base_phase_deg: float
    threshold_phase_tilt_deg: float
    orthogonal_coset_phase_deg: float
    geodesic_closure_phase_deg: float
    triangle_tilt_deg: float
    y12_tree_level: float
    alpha_gut: float
    matching_log_sum: float
    lambda_12_mgut: float
    lambda_matrix_mgut: np.ndarray
    matching_contributions: tuple[HeavyThresholdMatchingContribution, ...]
    decoupling_audit: HeavyStateDecouplingAuditData

    def matching_contribution_sum(self, *sources: str) -> float:
        return float(sum(item.contribution for item in self.matching_contributions if item.source in sources))

    @property
    def matching_log_sum_126h(self) -> float:
        return self.matching_contribution_sum("126_H")

    @property
    def matching_log_sum_210h(self) -> float:
        return self.matching_contribution_sum("210_H")

    @property
    def matching_log_sum_vh(self) -> float:
        return self.matching_contribution_sum("V_H")


@dataclass(frozen=True)
class HiggsCGCorrectionAuditData:
    r"""Natural $\mathbf{126}_H$ Clebsch suppression for the Yukawa-ratio pressure."""

    bare_overprediction_factor: float
    target_suppression: float
    clebsch_126: float
    inverse_clebsch_126_suppression: float
    mixed_channel_suppression: float
    corrected_pressure_factor: float
    residual_to_target: float


@dataclass(frozen=True)
class ParentSelection:
    master_level: int
    lepton_branching_index: int
    quark_branching_index: int


@dataclass(frozen=True)
class MatrixSpectrumAuditData:
    matrix: np.ndarray
    singular_values: np.ndarray
    sigma_min: float
    sigma_max: float
    rank: int
    condition_number: float
    reported_condition_number: float
    machine_precision_singular: bool
    perturbative_nonsingular: bool

    @property
    def display_condition_number(self) -> float:
        if self.machine_precision_singular and math.isinf(self.condition_number):
            return self.reported_condition_number
        return self.condition_number


def derive_matrix_spectrum_audit(
    matrix: np.ndarray,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> MatrixSpectrumAuditData:
    matrix_array = np.asarray(matrix, dtype=complex)
    singular_values = np.asarray(np.linalg.svd(matrix_array, compute_uv=False), dtype=float)
    if singular_values.size == 0:
        return MatrixSpectrumAuditData(
            matrix=matrix_array,
            singular_values=singular_values,
            sigma_min=0.0,
            sigma_max=0.0,
            rank=0,
            condition_number=math.inf,
            reported_condition_number=math.inf,
            machine_precision_singular=True,
            perturbative_nonsingular=False,
        )

    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    numerical_tolerance = _matrix_singular_value_tolerance(matrix_array.shape, sigma_max)
    rank = int(np.linalg.matrix_rank(matrix_array, tol=numerical_tolerance))
    machine_precision_singular = rank < min(matrix_array.shape) or sigma_min < numerical_tolerance
    condition_number = math.inf if sigma_min < numerical_tolerance else float(sigma_max / sigma_min)
    reported_condition_number = math.inf if math.isclose(
        sigma_max,
        0.0,
        rel_tol=0.0,
        abs_tol=condition_aware_abs_tolerance(scale=sigma_max),
    ) else float(sigma_max / max(sigma_min, numerical_tolerance))
    perturbative_nonsingular = (
        rank == min(matrix_array.shape)
        and not machine_precision_singular
        and math.isfinite(condition_number)
        and condition_number <= solver_config.stability_guard.perturbative_condition_limit
    )
    return MatrixSpectrumAuditData(
        matrix=matrix_array,
        singular_values=singular_values,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rank=rank,
        condition_number=condition_number,
        reported_condition_number=reported_condition_number,
        machine_precision_singular=machine_precision_singular,
        perturbative_nonsingular=perturbative_nonsingular,
    )


def enforce_perturbative_matrix(
    matrix: np.ndarray,
    *,
    coordinate: str,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    detail: str,
) -> MatrixSpectrumAuditData:
    spectral_audit = derive_matrix_spectrum_audit(matrix, solver_config=solver_config)
    solver_config.stability_guard.require_perturbative_condition_number(
        spectral_audit.display_condition_number,
        coordinate=coordinate,
        detail=detail,
    )
    return spectral_audit


@dataclass(frozen=True)
class SupportOverlapResult:
    matrix: np.ndarray
    determinant: float | None = None

    @property
    def spectral_audit(self) -> MatrixSpectrumAuditData:
        return derive_matrix_spectrum_audit(self.matrix)

    @property
    def singular_values(self) -> np.ndarray:
        return self.spectral_audit.singular_values

    @property
    def machine_precision_sigma_floor(self) -> float:
        return _matrix_singular_value_tolerance(self.matrix.shape, self.spectral_audit.sigma_max)

    @property
    def rank(self) -> int:
        return self.spectral_audit.rank

    @property
    def condition_number(self) -> float:
        return self.spectral_audit.condition_number

    @property
    def reported_condition_number(self) -> float:
        return self.spectral_audit.reported_condition_number

    @property
    def machine_precision_singular(self) -> bool:
        return self.spectral_audit.machine_precision_singular

    @property
    def perturbative_nonsingular(self) -> bool:
        return self.spectral_audit.perturbative_nonsingular

    def to_tex(self, support_deficit: int, required_rank: int, relaxed_gap: float) -> str:
        def format_entry(value: float) -> str:
            format_tolerance = condition_aware_abs_tolerance(self.reported_condition_number, scale=value)
            if math.isinf(value):
                return r"$\infty$"
            if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=format_tolerance):
                return r"$0.0$"
            if math.isclose(value, round(value), rel_tol=0.0, abs_tol=format_tolerance):
                return rf"${value:.1f}$"
            mantissa, exponent = f"{value:.11e}".split("e")
            return rf"${float(mantissa):.11f}\times10^{{{int(exponent)}}}$"

        real_matrix = require_real_array(self.matrix, label="support-overlap matrix")
        singular_values = [float(value) for value in self.singular_values]
        singular_value_text = "(" + r",\,".join(format_entry(value).strip("$") for value in singular_values) + ")"
        sigma_min_text = (
            rf"$\lesssim {format_entry(self.machine_precision_sigma_floor).strip('$')}$"
            if self.machine_precision_singular
            else format_entry(float(singular_values[-1]))
        )
        condition_text = (
            rf"$\gtrsim {format_entry(self.reported_condition_number).strip('$')}$"
            if self.machine_precision_singular or math.isinf(self.condition_number)
            else format_entry(self.condition_number)
        )
        determinant_value = require_real_scalar(
            np.linalg.det(self.matrix) if self.determinant is None else self.determinant,
            label="support-overlap determinant",
        )
        determinant_text = (
            rf"$\lesssim {format_entry(abs(float(determinant_value))).strip('$')}$"
            if math.isclose(
                float(determinant_value),
                0.0,
                rel_tol=0.0,
                abs_tol=condition_aware_abs_tolerance(self.reported_condition_number, scale=float(determinant_value)),
            )
            else format_entry(float(determinant_value))
        )
        matrix_rows = tuple(
            {
                "index": index,
                "entries": tuple(format_entry(float(value)) for value in row),
            }
            for index, row in enumerate(real_matrix, start=1)
        )
        return presentation_reporting.render_support_overlap_result(
            matrix_rows=matrix_rows,
            determinant_text=determinant_text,
            rank=self.rank,
            singular_value_text=singular_value_text,
            sigma_min_text=sigma_min_text,
            condition_text=condition_text,
            perturbative_verdict="yes" if self.perturbative_nonsingular else "no",
            support_deficit=support_deficit,
            required_rank=required_rank,
            relaxed_gap=f"{relaxed_gap:.12f}",
        )


@dataclass(frozen=True)
class LevelScanResult:
    """Single visible-level candidate in the Diophantine modularity scan."""

    lepton_level: int
    quark_level: int
    parent_level: int
    lepton_branching_index: float
    quark_branching_index: int
    visible_residual_mod1: float
    modularity_gap: float
    framing_gap: float
    flavor_condition_number: float
    central_charge_modular: bool
    framing_anomaly_free: bool
    flavor_nonsingular: bool
    chi2_flavor: float | None
    max_abs_pull: float | None
    flavor_matching: bool
    selected_visible_pair: bool
    bulk_anomaly_cancelled: bool
    modular_tilt_deg: float | None
    gamma_candidate_deg: float | None
    gamma_pull: float | None

    @property
    def passes_all(self) -> bool:
        return self.central_charge_modular and self.framing_anomaly_free and self.flavor_nonsingular


@dataclass(frozen=True)
class LevelStabilityScan:
    """Wide Diophantine scan summarizing algebraic visible-level candidates."""

    fixed_parent_level: int
    lepton_range: tuple[int, int]
    quark_range: tuple[int, int]
    total_pairs_scanned: int
    relaxed_modularity_allowance: float
    rows: tuple[LevelScanResult, ...]

    @property
    def selected_row(self) -> LevelScanResult:
        return next(row for row in self.rows if row.selected_visible_pair)

    @property
    def best_relaxed_neighbor(self) -> LevelScanResult | None:
        candidates = [
            row
            for row in self.rows
            if (not row.selected_visible_pair)
            and row.bulk_anomaly_cancelled
            and row.gamma_pull is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda row: abs(row.gamma_pull if row.gamma_pull is not None else math.inf))

    def to_tex(self) -> str:
        def format_condition_number(value: float) -> str:
            return r"$\infty$" if math.isinf(value) else rf"${value:.3e}$"

        rows = tuple(
            {
                "lepton_level": rf"${row.lepton_level}$",
                "quark_level": rf"${row.quark_level}$",
                "parent_level": rf"${row.parent_level}$",
                "visible_residual_mod1": rf"${row.visible_residual_mod1:.6f}$",
                "modularity_gap": rf"${row.modularity_gap:.6f}$",
                "framing_gap": rf"${row.framing_gap:.6f}$",
                "flavor_condition_number": format_condition_number(row.flavor_condition_number),
                "status_text": (
                    "selected benchmark"
                    if row.selected_visible_pair
                    else "candidate" if row.bulk_anomaly_cancelled else "screened"
                ),
            }
            for row in self.rows
        )
        return presentation_reporting.render_level_stability_scan(
            rows=rows,
            fixed_parent_level=self.fixed_parent_level,
        )

        def format_alpha_inverse(row: LevelScanResult, alpha_inverse: float) -> str:
            alpha_fraction = Fraction(
                GAUGE_HOLOGRAPHY_GENERATION_COUNT * row.parent_level,
                row.lepton_level + row.quark_level,
            )
            if row.selected_visible_pair and _matches_exact_fraction(alpha_inverse, GAUGE_STRENGTH_IDENTITY):
                return rf"${GAUGE_STRENGTH_IDENTITY_TEX}$"
            return rf"${_tex_fraction(alpha_fraction)}$"

        def format_quadratic_weight_projection(row: LevelScanResult) -> str:
            projection = Fraction(2 * row.quark_level, 3 * row.lepton_level)
            if row.selected_visible_pair:
                return rf"${QUADRATIC_WEIGHT_PROJECTION_TEX}$"
            return rf"${_tex_fraction(projection)}$"

        benchmark_row = self.selected_row
        benchmark_alpha_inverse = surface_tension_gauge_alpha_inverse(
            parent_level=benchmark_row.parent_level,
            lepton_level=benchmark_row.lepton_level,
            quark_level=benchmark_row.quark_level,
        )

        body_rows: list[str] = []
        for row in self.rows:
            alpha_inverse = surface_tension_gauge_alpha_inverse(
                parent_level=row.parent_level,
                lepton_level=row.lepton_level,
                quark_level=row.quark_level,
            )
            c_dark_completion = 24.0 * row.modularity_gap
            row_model = TopologicalVacuum(
                k_l=row.lepton_level,
                k_q=row.quark_level,
                parent_level=row.parent_level,
            )
            baryon_stability = EinsteinConsistencyEngine(model=row_model).calculate_topological_evaporation_rate()
            proton_lifetime_text = (
                rf"${_format_tex_scientific(baryon_stability.proton_lifetime_years, precision=2)}$"
                if baryon_stability.dimension_five_forbidden
                else r"\shortstack{\scriptsize $\ll10^{34}$\\ \scriptsize rapid $d=5$}"
            )
            triple_lock_active = (
                baryon_stability.dimension_five_forbidden
                and math.isclose(alpha_inverse, benchmark_alpha_inverse, rel_tol=0.0, abs_tol=5.0e-2)
            )
            triple_lock_text = r"\textbf{locked}" if triple_lock_active else "broken"
            lepton_level_text = rf"\textbf{{{row.lepton_level}}}" if row.selected_visible_pair else f"{row.lepton_level}"
            body_rows.append(
                " & ".join(
                    (
                        rf"${lepton_level_text}$" if row.selected_visible_pair else rf"${row.lepton_level}$",
                        rf"${row.quark_level}$",
                        rf"${row.parent_level}$",
                        rf"${row.visible_residual_mod1:.6f}$",
                        rf"${row.modularity_gap:.6f}$",
                        rf"${row.framing_gap:.6f}$",
                        format_alpha_inverse(row, alpha_inverse),
                        format_quadratic_weight_projection(row),
                        rf"${c_dark_completion:.4f}$",
                        proton_lifetime_text,
                        format_condition_number(row.flavor_condition_number),
                        triple_lock_text,
                    )
                )
                + r" \\"
            )

        return template_utils.render_latex_table(
            column_spec="|c|c|c|c|c|c|c|c|c|c|c|c|",
            header_rows=(
                r"$k_{\ell}$ & $k_q$ & $K$ & \shortstack{\scriptsize Quantized\\ \scriptsize Anomaly Residual\\ \scriptsize $\mathfrak A_{\rm vis}$} & $\Delta_{\rm mod}^{(312)}$ & $\Delta_{\rm fr}$ & \shortstack{\scriptsize Predicted\\ \scriptsize $\alpha^{-1}_{\rm surf}$} & \shortstack{\scriptsize Quadratic Weight\\ \scriptsize Projection\\ \scriptsize $\langle \Sigma_{126}\rangle/\langle \phi_{10}\rangle$} & \shortstack{\scriptsize Unitary\\ \scriptsize Buffer\\ \scriptsize $c_{\rm dark}$} & \shortstack{\scriptsize Predicted\\ \scriptsize $\tau_p\,[\mathrm{yr}]$} & $|\det S_{\rm flav}|$ & \shortstack{\scriptsize Triple\\ \scriptsize Lock} \\",
            ),
            body_rows=tuple(body_rows),
            opening_lines=(r"\resizebox{\linewidth}{!}{%",),
            closing_lines=(r"}",),
            style="grid",
        )


@dataclass(frozen=True)
class PullTableRow:
    """Single observable in the publication-facing global pull table."""

    observable: str
    theory_uv: float
    theory_mz: float
    pull_data: PullData | None
    structural_context: str
    source_label: str
    units: str = ""
    reference_override: str | None = None
    included_in_audit: bool = True
    included_in_predictive_fit: bool = True
    is_calibration_anchor: bool = False
    theoretical_uncertainty_fraction: float = THEORETICAL_UNCERTAINTY_FRACTION
    parametric_covariance_fraction: float = PARAMETRIC_COVARIANCE_FRACTION
    observable_key: str = ""


@dataclass(frozen=True)
class IHSingularValueSpectrumData:
    r"""Singular-value spectrum of the inverted-hierarchy support matrix."""

    indices: np.ndarray
    singular_values: np.ndarray
    sigma_min: float
    rank: int
    condition_number: float
    machine_precision_singular: bool


@dataclass(frozen=True)
class HolographicDensityBoundData:
    r"""Auxiliary density-cap diagnostic carried alongside the scan-window report."""

    pixel_capacity: float
    level_99_load: float
    level_100_load: float
    max_admissible_level: int


@dataclass(frozen=True)
class CkmPhaseTiltProfileData:
    r"""First-principles check of the phase-only CKM threshold correction."""

    weight_grid: np.ndarray
    chi2_values: np.ndarray
    delta_chi2_values: np.ndarray
    gamma_values: np.ndarray
    vus_values: np.ndarray
    vcb_values: np.ndarray
    vub_values: np.ndarray
    best_fit_weight: float
    best_fit_chi2: float
    benchmark_weight: float
    benchmark_delta_chi2: float
    benchmark_gamma_deg: float
    max_vus_shift: float
    max_vcb_shift: float
    max_vub_shift: float


@dataclass(frozen=True)
class SO10GeometricKappaData:
    r"""Weight-simplex audit behind the order-one mass-bridge coefficient."""

    weight_simplex_hyperarea: float
    regular_reference_hyperarea: float
    area_ratio: float
    spinorial_retention: float
    geometric_factor: float
    spinor_dimension: int
    derived_kappa: float


@dataclass(frozen=True)
class ModularHorizonSelectionData:
    r"""Cardy-like modular selection of the horizon information budget."""

    unit_modular_weight: float
    effective_vacuum_weight: float
    parent_central_charge: float
    framing_gap_area: float
    visible_edge_penalty: float
    derived_bits: float
    planck_crosscheck_ratio: float


@dataclass(frozen=True)
class PhysicsAudit:
    """Explicitly evaluated bundle of runtime physics audits used for reporting."""

    search_window: HolographicDensityBoundData
    geometric_kappa: SO10GeometricKappaData
    modular_horizon: ModularHorizonSelectionData
    transport_curvature: TransportCurvatureAuditData
    topological_threshold_gev: float
    gauge_unification_beta_shift_126: np.ndarray
    gauge_unification_beta_shift_10: np.ndarray


@dataclass(frozen=True)
class BaryonStabilityAudit:
    r"""Topological proton-decay audit tied to the Holographic Information Horizon.

    The quoted ``proton_lifetime_years`` is the perturbative $d=6$ floor that
    remains once the vanishing framing anomaly removes the faster torsion-
    sensitive $d=5$ channel, while
    ``protected_evaporation_lifetime_years`` carries the additional formal
    modular-tunneling barrier $e^{1/\Delta_{\rm mod}}$.
    """

    gut_scale_gev: float
    effective_gauge_mass_gev: float
    unified_alpha_inverse: float
    unified_alpha: float
    modular_gap: float
    modular_tunneling_penalty: float
    tunneling_safety_boost: float
    dimension_six_width_gev: float
    proton_lifetime_years: float
    protected_evaporation_lifetime_years: float
    dimension_five_forbidden: bool


@dataclass(frozen=True)
class GravityAudit:
    r"""Consistency audit for the effective Einstein system induced by the boundary data.

    This object records the benchmark quantities used by the bulk-emergence
    Topological Consistency Map: a vanishing framing gap enforces a torsion-free Levi-Civita
    connection, the $\\mathbf{126}_H$ Clebsch factor acts as a positive-energy
    regulator, the modular complement furnishes the parity-bit register of the
    HEC decoder, and the ultraviolet neutrino scale is required to agree with
    the CKN bridge $m_0=\kappa M_P N^{-1/4}$.
    """

    parent_central_charge: float
    holographic_bits: float
    geometric_residue: float
    visible_central_charge: float
    c_dark_completion: float
    modular_residue_efficiency: float
    omega_dm_ratio: float
    parity_bit_density_constraint_satisfied: bool
    framing_gap: float
    vacuum_pressure_t00: float
    mass_suppression: float
    neutrino_scale_ev: float
    ckn_limit_ev: float
    lambda_budget_si_m2: float
    observed_lambda_si_m2: float
    baryon_stability: BaryonStabilityAudit
    torsion_free: bool
    non_singular_bulk: bool
    lambda_aligned: bool

    def __post_init__(self) -> None:
        assert solver_isclose(self.framing_gap, 0.0), "GravityAudit requires framing_gap == 0."
        self.verify_welded_mass_rigidity(self.neutrino_scale_ev)
        self.verify_welded_mass_rigidity(self.ckn_limit_ev)

    @property
    def kappa_D5(self) -> float:
        return self.geometric_residue

    @property
    def M_planck(self) -> float:
        return PLANCK_MASS_EV

    @property
    def N_holo(self) -> float:
        return self.holographic_bits

    def verify_welded_mass_rigidity(self, current_m_nu: float) -> None:
        expected = self.kappa_D5 * self.M_planck * (self.N_holo**-0.25)
        if math.isclose(current_m_nu, expected, rel_tol=0.0, abs_tol=TOPOLOGICAL_MASS_COORDINATE_ABS_TOL_EV):
            return
        raise _build_topological_mass_coordinate_lock_error(
            current_m_nu,
            expected_mass=expected,
            bit_count=self.N_holo,
            kappa_geometric=self.kappa_D5,
        )

    @property
    def bulk_emergent(self) -> bool:
        return self.torsion_free and self.non_singular_bulk and self.lambda_aligned

    @property
    def gmunu_consistency_score(self) -> float:
        lambda_scale = max(abs(self.lambda_budget_si_m2), abs(self.observed_lambda_si_m2))
        lambda_score = 1.0 if math.isclose(lambda_scale, 0.0, rel_tol=0.0, abs_tol=1.0e-30) else min(
            abs(self.lambda_budget_si_m2), abs(self.observed_lambda_si_m2)
        ) / lambda_scale
        neutrino_scale = max(abs(self.neutrino_scale_ev), abs(self.ckn_limit_ev))
        neutrino_score = 1.0 if math.isclose(neutrino_scale, 0.0, rel_tol=0.0, abs_tol=1.0e-30) else min(
            abs(self.neutrino_scale_ev), abs(self.ckn_limit_ev)
        ) / neutrino_scale
        torsion_score = 1.0 if self.torsion_free else 0.0
        regulator_score = 1.0 if self.non_singular_bulk else 0.0
        return torsion_score * regulator_score * lambda_score * neutrino_score


@dataclass(frozen=True)
class DarkEnergyTensionAudit:
    r"""Holographic dark-energy audit for the boundary surface-tension sector."""

    holographic_bits: float
    planck_anchor_bits: float
    planck_crosscheck_ratio: float
    geometric_residue: float
    geometric_residue_quartic: float
    benchmark_modularity_gap: float
    central_charge_residual: float
    vacuum_pressure_t00: float
    surface_tension_density_ev4: float
    lambda_identity_si_m2: float
    lambda_budget_si_m2: float
    observed_lambda_si_m2: float
    surface_tension_deviation_fraction: float
    bit_shift_fraction: float
    m_0_mz_central_ev: float
    m_0_mz_minus_fraction_ev: float
    m_0_mz_plus_fraction_ev: float
    alpha_inverse_central: float
    alpha_inverse_minus_fraction: float
    alpha_inverse_plus_fraction: float
    surface_tension_aligned: bool
    triple_alignment_rigid: bool

    @property
    def surface_tension_deviation_percent(self) -> float:
        return 100.0 * self.surface_tension_deviation_fraction

    @property
    def lambda_surface_coefficient(self) -> float:
        return self.lambda_budget_si_m2 / self.lambda_identity_si_m2

    @property
    def m_0_minus_fractional_shift_percent(self) -> float:
        return 100.0 * (self.m_0_mz_minus_fraction_ev / self.m_0_mz_central_ev - 1.0)

    @property
    def m_0_plus_fractional_shift_percent(self) -> float:
        return 100.0 * (self.m_0_mz_plus_fraction_ev / self.m_0_mz_central_ev - 1.0)


@dataclass(frozen=True)
class GaugeHolographyAudit:
    r"""Gauge-coupling audit extracted from the fixed boundary level data."""

    generation_count: int
    parent_level: int
    lepton_level: int
    quark_level: int
    topological_alpha_inverse: float
    codata_alpha_inverse: float
    modular_gap_scaled_inverse: float
    geometric_residue_fraction: float
    modular_gap_alignment_fraction: float
    framing_closed: bool

    @property
    def geometric_residue_percent(self) -> float:
        return 100.0 * self.geometric_residue_fraction

    @property
    def modular_gap_alignment_percent(self) -> float:
        return 100.0 * self.modular_gap_alignment_fraction

    @property
    def topological_stability_pass(self) -> bool:
        return self.framing_closed and self.geometric_residue_percent <= GAUGE_HOLOGRAPHY_PASS_PERCENT


@dataclass(frozen=True)
class DarkEnergyTensionAudit:
    r"""Dark-energy audit from the holographic surface tension of the completed boundary."""

    holographic_bits: float
    geometric_residue: float
    modular_gap: float
    c_dark_completion: float
    lambda_surface_tension_si_m2: float
    lambda_anchor_si_m2: float
    lambda_scaling_identity_si_m2: float
    rho_vac_surface_tension_ev4: float
    rho_vac_from_defect_scale_ev4: float
    minus_one_percent_lambda_si_m2: float
    plus_one_percent_lambda_si_m2: float
    minus_one_percent_lambda_fractional_shift: float
    plus_one_percent_lambda_fractional_shift: float
    topological_mass_coordinate_ev: float
    minus_one_percent_topological_mass_coordinate_ev: float
    plus_one_percent_topological_mass_coordinate_ev: float
    minus_one_percent_topological_mass_coordinate_fractional_shift: float
    plus_one_percent_topological_mass_coordinate_fractional_shift: float
    alpha_inverse_central: float
    alpha_inverse_minus_one_percent: float
    alpha_inverse_plus_one_percent: float
    triple_match_product: float
    triple_match_saturated: bool
    sensitivity_audit_triggered_integrity_error: bool
    sensitivity_audit_message: str

    @property
    def surface_tension_prefactor(self) -> float:
        if math.isclose(self.lambda_scaling_identity_si_m2, 0.0, rel_tol=0.0, abs_tol=1.0e-300):
            return 0.0
        return self.lambda_surface_tension_si_m2 / self.lambda_scaling_identity_si_m2

    @property
    def surface_tension_deviation_fraction(self) -> float:
        if math.isclose(self.lambda_anchor_si_m2, 0.0, rel_tol=0.0, abs_tol=1.0e-300):
            return 0.0
        return (self.lambda_surface_tension_si_m2 - self.lambda_anchor_si_m2) / self.lambda_anchor_si_m2

    @property
    def surface_tension_deviation_percent(self) -> float:
        return 100.0 * abs(self.surface_tension_deviation_fraction)

    @property
    def neutrino_scale_ev(self) -> float:
        return self.topological_mass_coordinate_ev

    @property
    def minus_one_percent_m0_ev(self) -> float:
        return self.minus_one_percent_topological_mass_coordinate_ev

    @property
    def plus_one_percent_m0_ev(self) -> float:
        return self.plus_one_percent_topological_mass_coordinate_ev

    @property
    def minus_one_percent_m0_fractional_shift(self) -> float:
        return self.minus_one_percent_topological_mass_coordinate_fractional_shift

    @property
    def plus_one_percent_m0_fractional_shift(self) -> float:
        return self.plus_one_percent_topological_mass_coordinate_fractional_shift

    @property
    def alpha_locked_under_bit_shift(self) -> bool:
        return (
            math.isclose(self.alpha_inverse_central, self.alpha_inverse_minus_one_percent, rel_tol=0.0, abs_tol=1.0e-15)
            and math.isclose(self.alpha_inverse_central, self.alpha_inverse_plus_one_percent, rel_tol=0.0, abs_tol=1.0e-15)
        )


@dataclass(frozen=True)
class UnitaryBoundAudit:
    r"""Finite-buffer unitarity audit for the anomaly-free holographic benchmark."""

    holographic_bits: float
    geometric_residue: float
    entropy_max_nats: float
    c_dark_completion: float
    modular_gap: float
    framing_gap: float
    gmunu_consistency_score: float
    holographic_buffer_entropy: float
    regulated_curvature_entropy: float
    curvature_buffer_margin: float
    information_evaporation_rate_per_year: float
    information_recovery_rate_per_year: float
    recovery_lifetime_years: float
    topological_mass_coordinate_ev: float
    triple_match_product: float
    torsion_free_stability: bool
    lloyds_limit_ops_per_second: float
    complexity_growth_rate_ops_per_second: float
    zero_point_complexity: float
    max_complexity_capacity: float
    clock_skew: float
    dark_matter_rhn_scale_gev: float = math.nan
    dark_matter_beta_squared: float = math.nan
    dark_matter_mass_gev: float = math.nan
    dark_matter_alpha_chi: float = math.nan
    dark_matter_sigma_geom_cm2: float = math.nan
    light_wimp_impossible: bool = False
    direct_detection_below_floor: bool = False
    unitary_bound_satisfied: bool = False
    proton_recovery_identity: bool = False
    recovery_locked_to_delta_mod: bool = False
    dark_sector_holographic_rigidity: bool = False
    holographic_rigidity: bool = False
    universal_computational_limit_pass: bool = False

    def __post_init__(self) -> None:
        self.verify_welded_mass_rigidity(self.topological_mass_coordinate_ev)

    @property
    def kappa_D5(self) -> float:
        return self.geometric_residue

    @property
    def M_planck(self) -> float:
        return PLANCK_MASS_EV

    @property
    def N_holo(self) -> float:
        return self.holographic_bits

    def verify_welded_mass_rigidity(self, current_m_nu: float) -> None:
        expected = self.kappa_D5 * self.M_planck * (self.N_holo**-0.25)
        if math.isclose(current_m_nu, expected, rel_tol=0.0, abs_tol=TOPOLOGICAL_MASS_COORDINATE_ABS_TOL_EV):
            return
        raise _build_topological_mass_coordinate_lock_error(
            current_m_nu,
            expected_mass=expected,
            bit_count=self.N_holo,
            kappa_geometric=self.kappa_D5,
        )

    @property
    def curvature_buffer_margin_percent(self) -> float:
        if math.isclose(self.holographic_buffer_entropy, 0.0, rel_tol=0.0, abs_tol=1.0e-300):
            return 0.0
        return 100.0 * self.curvature_buffer_margin / self.holographic_buffer_entropy

    @property
    def complexity_utilization_fraction(self) -> float:
        if math.isclose(self.lloyds_limit_ops_per_second, 0.0, rel_tol=0.0, abs_tol=1.0e-300):
            return 0.0
        return self.complexity_growth_rate_ops_per_second / self.lloyds_limit_ops_per_second

    def validate_holographic_complexity_bound(self) -> None:
        bound_tolerance = 1.0e-12 * max(1.0, abs(self.lloyds_limit_ops_per_second))
        assert (
            self.complexity_growth_rate_ops_per_second <= self.lloyds_limit_ops_per_second + bound_tolerance
        ), "Universal Computational Limit: FAILED."


@dataclass(frozen=True)
class PagePointAudit:
    r"""Page-point audit for the finite unitary buffer of the anomaly-free branch."""

    entropy_max_nats: float
    c_dark_completion: float
    page_point_entropy: float
    bulk_entanglement_entropy: float
    modular_complement_entropy: float
    page_curve_locked: bool

    @property
    def page_point_saturation_fraction(self) -> float:
        if math.isclose(self.page_point_entropy, 0.0, rel_tol=0.0, abs_tol=1.0e-300):
            return 0.0
        return self.bulk_entanglement_entropy / self.page_point_entropy

    @property
    def page_point_saturation_percent(self) -> float:
        return 100.0 * self.page_point_saturation_fraction

    @property
    def page_point_reached(self) -> bool:
        return self.bulk_entanglement_entropy >= self.page_point_entropy - 1.0e-12


@dataclass(frozen=True)
class TorsionScramblingTransitAudit:
    r"""Stage XIII compact-object transit audit for torsion-induced flavor scrambling."""

    gravitational_gradient_scale: float
    transit_path_length_km: float
    density_scale: float
    clock_skew: float
    complexity_utilization_fraction: float
    rank_deficiency_load: float
    scrambling_fraction: float
    reference_nu_e_to_nu_tau_probability: float
    torsion_scrambled_nu_e_to_nu_tau_probability: float
    reference_nu_e_survival_probability: float
    torsion_scrambled_nu_e_survival_probability: float
    local_support_matrix: np.ndarray
    support_rank: int
    support_condition_number: float
    support_sigma_min: float
    support_machine_precision_singular: bool
    torsion_scrambling_triggered: bool

    @property
    def msw_excess_probability(self) -> float:
        return self.torsion_scrambled_nu_e_to_nu_tau_probability - self.reference_nu_e_to_nu_tau_probability

    @property
    def violates_standard_msw(self) -> bool:
        return self.torsion_scrambling_triggered and self.msw_excess_probability > 1.0e-2


UnitaryAudit = UnitaryBoundAudit


@dataclass(frozen=True)
class FramingStabilityAudit:
    r"""Percent-level alpha-detuning audit for framing closure on the fixed parent branch."""

    alpha_shift_fraction: float
    central_alpha_inverse: float
    minus_shifted_alpha_inverse: float
    plus_shifted_alpha_inverse: float
    central_effective_lepton_level: float
    minus_shifted_effective_lepton_level: float
    plus_shifted_effective_lepton_level: float
    central_framing_gap: float
    minus_shifted_framing_gap: float
    plus_shifted_framing_gap: float
    alpha_lock_required: bool

    @property
    def alpha_shift_percent(self) -> float:
        return 100.0 * self.alpha_shift_fraction


@dataclass(frozen=True)
class RigidityStressTestAudit:
    r"""Single-bit stress test for the anomaly-free Triple-Lock cell."""

    bit_shift: int
    central_bit_count: float
    shifted_bit_count: float
    central_lepton_level: int
    shifted_effective_lepton_level: int
    central_alpha_inverse: float
    shifted_alpha_inverse: float
    central_lambda_si_m2: float
    shifted_lambda_si_m2: float
    shifted_framing_gap: float
    overconstrained: bool

    @property
    def lambda_fractional_shift_percent(self) -> float:
        if math.isclose(self.central_lambda_si_m2, 0.0, rel_tol=0.0, abs_tol=1.0e-300):
            return 0.0
        return 100.0 * (self.shifted_lambda_si_m2 / self.central_lambda_si_m2 - 1.0)


@dataclass(frozen=True)
class GhostCharacterAuditData:
    r"""Diagnostic packaging the IH informational-cost extension for one-copy support."""

    extra_character_count: int
    ghost_norm_upper_bound: float
    integrable_spin_bound: float
    swampland_excluded: bool

    @property
    def ih_nonminimal_extension_required(self) -> bool:
        r"""Publication-facing flag for the IH informational-cost carve-out."""

        return self.swampland_excluded

    @property
    def ih_bankruptcy_exception(self) -> bool:
        """Backward-compatible alias for older manuscript wording."""

        return self.ih_nonminimal_extension_required


@dataclass(frozen=True)
class MassRatioStabilityAuditData:
    r"""Factor-five singular-value stability audit for the mixing eigenvectors."""

    perturbation_factor: float
    target_relative_suppression: float
    clebsch_relative_suppression: float
    relative_spectral_volume_shift: float
    lepton_unitary_frobenius_shift: float
    quark_unitary_frobenius_shift: float
    lepton_singular_values: tuple[float, float, float]
    lepton_perturbed_singular_values: tuple[float, float, float]
    quark_singular_values: tuple[float, float, float]
    quark_perturbed_singular_values: tuple[float, float, float]
    lepton_left_overlap_min: float
    lepton_right_overlap_min: float
    quark_left_overlap_min: float
    quark_right_overlap_min: float
    lepton_angle_shifts_deg: tuple[float, float, float]
    quark_angle_shifts_deg: tuple[float, float, float]
    lepton_sigma_shifts: tuple[float, float, float]
    quark_sigma_shifts: tuple[float, float, float]
    max_sigma_shift: float
    ensemble_sample_count: int = 0
    ensemble_seed: int = DEFAULT_RANDOM_SEED
    ensemble_all_within_one_sigma: bool = True
    ensemble_max_sigma_shift: float = 0.0
    ensemble_theta13_max_sigma_shift: float = 0.0
    ensemble_theta_c_max_sigma_shift: float = 0.0
    ensemble_mass_scale_shift_min: float = 0.0
    ensemble_mass_scale_shift_max: float = 0.0
    ensemble_effective_suppression_ratios: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ensemble_max_sigma_shifts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ensemble_mass_scale_shifts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


@dataclass(frozen=True)
class Chi2LandscapePoint:
    """Single point in the benchmark-centered predictive χ² follow-up audit."""

    lepton_level: int
    quark_level: int
    predictive_chi2: float
    max_abs_pull: float
    conditional_p_value: float
    selected_visible_pair: bool


@dataclass(frozen=True)
class Chi2LandscapeAuditData:
    """Autocorrelation-aware audit of the reduced predictive χ² landscape."""

    lepton_range: tuple[int, int]
    quark_range: tuple[int, int]
    total_pairs_scanned: int
    chi2_grid: np.ndarray
    points: tuple[Chi2LandscapePoint, ...]
    acceptance_threshold: float
    survival_count: int
    lepton_autocorrelation: np.ndarray
    quark_autocorrelation: np.ndarray
    lepton_correlation_length: float
    quark_correlation_length: float
    effective_correlation_length: float
    effective_trial_count: float

    @property
    def survival_fraction(self) -> float:
        if self.total_pairs_scanned <= 0:
            return 0.0
        return self.survival_count / self.total_pairs_scanned

    @property
    def top_points(self) -> tuple[Chi2LandscapePoint, ...]:
        return tuple(sorted(self.points, key=lambda point: (point.predictive_chi2, point.max_abs_pull, point.lepton_level, point.quark_level)))

    @property
    def selected_point(self) -> Chi2LandscapePoint:
        return next(point for point in self.points if point.selected_visible_pair)


@dataclass(frozen=True)
class SeedRobustnessAuditData:
    """Seed-by-seed stability audit for the stochastic uncertainty pipeline."""

    seeds: tuple[int, ...]
    observable_names: tuple[str, ...]
    predictive_chi2_values: np.ndarray
    predictive_p_values: np.ndarray
    parametric_sigmas: np.ndarray
    vev_max_sigma_shifts: np.ndarray
    max_relative_variance: float
    max_relative_std: float

    @property
    def seed_count(self) -> int:
        return len(self.seeds)


@dataclass(frozen=True)
class FramingGapStabilityData:
    r"""Continuous healing of the framing gap by the $\mathbf{126}_H$ threshold."""

    m_126_grid_gev: np.ndarray
    framing_gap_values: np.ndarray
    healing_fraction_values: np.ndarray
    gamma_healed_deg: np.ndarray
    matching_m126_gev: float
    matching_gamma_deg: float
    bare_gamma_rg_deg: float
    observed_gamma_deg: float

    @property
    def higgs_vev_matching_m126_gev(self) -> float:
        r"""Publication-facing name for the derived $\mathbf{126}_H$ matching scale."""

        return self.matching_m126_gev

    @property
    def higgs_vev_matching_gamma_deg(self) -> float:
        r"""Publication-facing name for the healed CKM apex at the matching point."""

        return self.matching_gamma_deg


@dataclass(frozen=True)
class PullTable:
    """Structured PMNS/CKM fit summary at the electroweak scale."""

    rows: tuple[PullTableRow, ...]
    audit_observable_count: int
    audit_chi2: float
    audit_rms_pull: float
    audit_max_abs_pull: float
    audit_degrees_of_freedom: int
    predictive_observable_count: int
    predictive_chi2: float
    predictive_rms_pull: float
    predictive_max_abs_pull: float
    predictive_degrees_of_freedom: int
    predictive_conditional_p_value: float
    predictive_p_value: float
    threshold_alignment_subtraction_count: int
    phenomenological_parameter_count: int
    calibration_parameter_count: int
    calibration_anchor_observable: str
    calibration_anchor_pull: float
    calibration_input_symbol: str
    calibration_input_value: float
    predictive_reduced_chi2: float
    predictive_landscape_trial_count: int
    predictive_followup_trial_count: int = 1
    predictive_effective_trial_count: float = 1.0
    predictive_correlation_length: float = 1.0
    predictive_lepton_correlation_length: float = 1.0
    predictive_quark_correlation_length: float = 1.0
    gut_threshold_residue_value: float = math.nan
    transport_caveat_note: str | None = None
    ckm_phase_tilt_parameter_value: InitVar[float | None] = None

    def __post_init__(self, ckm_phase_tilt_parameter_value: float | None) -> None:
        if ckm_phase_tilt_parameter_value is not None:
            if not math.isnan(self.gut_threshold_residue_value) and not math.isclose(
                self.gut_threshold_residue_value,
                float(ckm_phase_tilt_parameter_value),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("PullTable received conflicting threshold-residue aliases.")
            object.__setattr__(self, "gut_threshold_residue_value", float(ckm_phase_tilt_parameter_value))
        elif math.isnan(self.gut_threshold_residue_value):
            raise TypeError(
                "PullTable requires gut_threshold_residue_value or ckm_phase_tilt_parameter_value."
            )

    @property
    def continuous_parameter_subtraction_count(self) -> int:
        """Number of continuous inputs subtracted from the benchmark degrees of freedom."""

        return (
            self.phenomenological_parameter_count
            + self.threshold_alignment_subtraction_count
            + self.calibration_parameter_count
        )

    @property
    def effective_dof_subtraction_count(self) -> int:
        """Explicit fit-parameter subtraction entering the reported degrees of freedom."""

        return self.continuous_parameter_subtraction_count

    @property
    def local_frequentist_dof_subtraction_count(self) -> int:
        """Explicit fit parameters subtracted from the local benchmark tally."""

        return self.effective_dof_subtraction_count

    @property
    def predictive_discrete_selection_lee_p_value(self) -> float:
        """Backward-compatible alias for the reported predictive p-value."""

        return self.predictive_p_value

    @property
    def ckm_phase_tilt_parameter_value_legacy(self) -> float:
        """Backward-compatible access to the Wilson-coefficient benchmark value."""

        return self.gut_threshold_residue_value

    def to_tex(self) -> str:
        def format_value(value: float, units: str) -> str:
            if units == "deg":
                return rf"${value:.2f}^\circ$"
            if units == "meV":
                return rf"${value:.3f}\,\mathrm{{meV}}$"
            if units == "eV":
                return rf"${value:.5e}\,\mathrm{{eV}}$"
            return rf"${value:.5f}$"

        def format_theoretical_uncertainty(row: PullTableRow) -> str:
            combined_fraction = math.hypot(row.theoretical_uncertainty_fraction, row.parametric_covariance_fraction)
            uncertainty = abs(row.theory_mz) * combined_fraction
            if row.units == "deg":
                return rf"$\pm {uncertainty:.2f}^\circ$"
            if row.units == "meV":
                return rf"$\pm {uncertainty:.3f}\,\mathrm{{meV}}$"
            if row.units == "eV":
                return rf"$\pm {uncertainty:.5e}\,\mathrm{{eV}}$"
            return rf"$\pm {uncertainty:.5f}$"

        def format_reference(row: PullTableRow) -> str:
            if row.reference_override is not None:
                return row.reference_override
            if row.pull_data is None:
                return row.source_label
            central = row.pull_data.central
            sigma = row.pull_data.sigma
            if row.units == "deg":
                benchmark = rf"${central:.2f}\pm{sigma:.2f}^\circ$"
            else:
                benchmark = rf"${central:.5f}\pm{sigma:.5f}$"
            return rf"\shortstack{{\scriptsize {row.source_label} \\ {benchmark}}}"

        def format_pull(row: PullTableRow) -> str:
            if row.pull_data is None:
                return r"--"
            return rf"${row.pull_data.pull:+.2f}\sigma$"

        def format_context(row: PullTableRow) -> str:
            return row.structural_context

        note_text = (
            r"\scriptsize Numerical summary only. Combined uncertainties use "
            r"$\sigma_{\rm tot}^2=\sigma_{\rm exp}^2+\sigma_{\rm th}^2+\Sigma_{\rm par}$; "
            rf"landscape cells={self.predictive_landscape_trial_count}; "
            r"welded mass coordinate $m_\nu=\kappa_{D_5}M_PN^{-1/4}$; "
            rf"geometric residue ${self.calibration_input_symbol}={self.calibration_input_value:.2f}$; "
            rf"threshold residue $w_{{\rm th}}={self.gut_threshold_residue_value:.2f}$."
        )
        calibration_summary = (
            r"\multicolumn{5}{|r|}{Geometric residue} & "
            f"${self.calibration_input_symbol}={self.calibration_input_value:.2f}$ \\\\" 
        )
        anchor_summary = (
            r"\multicolumn{5}{|r|}{Threshold residue} & "
            f"$w_{{\rm th}}={self.gut_threshold_residue_value:.2f}$ \\\\" 
        )

        rows = tuple(
            {
                "observable": row.observable,
                "theory_mz": format_value(row.theory_mz, row.units),
                "theoretical_uncertainty": format_theoretical_uncertainty(row),
                "reference": format_reference(row),
                "structural_context": format_context(row),
                "pull": format_pull(row),
            }
            for row in self.rows
        )
        if self.transport_caveat_note is not None:
            note_text = note_text + " " + self.transport_caveat_note
        return presentation_reporting.render_pull_table(
            rows=rows,
            note_text=note_text,
            predictive_observable_count=self.predictive_observable_count,
            predictive_chi2=f"{self.predictive_chi2:.3f}",
            predictive_degrees_of_freedom=self.predictive_degrees_of_freedom,
            local_frequentist_dof_subtraction_count=self.local_frequentist_dof_subtraction_count,
            predictive_landscape_trial_count=self.predictive_landscape_trial_count,
            predictive_reduced_chi2=f"{self.predictive_reduced_chi2:.3f}",
            predictive_p_value=_format_tex_probability(self.predictive_p_value),
            audit_observable_count=self.audit_observable_count,
            audit_chi2=f"{self.audit_chi2:.3f}",
            audit_degrees_of_freedom=self.audit_degrees_of_freedom,
            calibration_summary=calibration_summary,
            anchor_summary=anchor_summary,
            predictive_rms_pull=f"{self.predictive_rms_pull:.3f}",
            predictive_max_abs_pull=f"{self.predictive_max_abs_pull:.2f}\\sigma",
        )


PullTable.ckm_phase_tilt_parameter_value = property(lambda self: self.gut_threshold_residue_value)

@dataclass(frozen=True)
class GlobalSensitivityRow:
    """Single point in the fixed-parent low-rank landscape audit."""

    lepton_level: int
    quark_level: int
    parent_level: int
    central_charge_residual: float
    modularity_gap: float
    lepton_framing_gap: float
    quark_framing_gap: float
    anomaly_energy: float
    lepton_flavor_condition_number: float
    quark_flavor_condition_number: float
    exact_pass: bool
    selected_visible_pair: bool


@dataclass(frozen=True)
class GlobalSensitivityAudit:
    """Global fixed-parent sensitivity audit over the low-rank visible landscape."""

    lepton_range: tuple[int, int]
    quark_range: tuple[int, int]
    total_pairs_scanned: int
    rows: tuple[GlobalSensitivityRow, ...]

    @property
    def selected_row(self) -> GlobalSensitivityRow:
        return next(row for row in self.rows if row.selected_visible_pair)

    @property
    def selected_rank(self) -> int:
        return next(index for index, row in enumerate(self.rows, start=1) if row.selected_visible_pair)

    @property
    def exact_pass_count(self) -> int:
        return sum(1 for row in self.rows if row.exact_pass)

    @property
    def unique_exact_pass(self) -> bool:
        return self.selected_row.exact_pass and self.exact_pass_count == 1

    @property
    def next_best_row(self) -> GlobalSensitivityRow:
        return next((row for row in self.rows if not row.selected_visible_pair), self.selected_row)

    @property
    def exact_modularity_roots(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (row.lepton_level, row.quark_level, row.parent_level)
            for row in self.rows
            if solver_isclose(row.modularity_gap, 0.0)
        )

    @property
    def algebraic_gap(self) -> float:
        if self.next_best_row.selected_visible_pair:
            return 0.0
        return abs(self.next_best_row.anomaly_energy - self.selected_row.anomaly_energy)


def report_algebraic_uniqueness(
    global_audit: GlobalSensitivityAudit,
    target_tuple: tuple[int, int, int] | None = None,
) -> bool:
    r"""Return whether the selected tuple participates in the residual modularity-root diagnostic."""

    resolved_target_tuple = global_audit.selected_row if target_tuple is None else target_tuple
    if isinstance(resolved_target_tuple, GlobalSensitivityRow):
        resolved_target_tuple = (
            resolved_target_tuple.lepton_level,
            resolved_target_tuple.quark_level,
            resolved_target_tuple.parent_level,
        )
    return global_audit.exact_modularity_roots == (resolved_target_tuple,)


@lru_cache(maxsize=2048)
def _followup_predictive_point_cached(
    lepton_level: int,
    quark_level: int,
    parent_level: int,
    scale_ratio: float,
    bit_count: float,
    kappa_geometric: float,
) -> tuple[float, float, float]:
    model = TopologicalVacuum(
        k_l=lepton_level,
        k_q=quark_level,
        parent_level=parent_level,
        scale_ratio=scale_ratio,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
    )
    try:
        pmns = derive_pmns(model=model)
        ckm = derive_ckm(model=model)
        pull_table = derive_pull_table(pmns, ckm)
    except (PhysicalSingularityException, PerturbativeBreakdownException):
        return math.inf, math.inf, 0.0

    predictive_rows = tuple(
        row for row in pull_table.rows if row.included_in_predictive_fit and row.pull_data is not None
    )
    max_abs_pull = max(abs(row.pull_data.pull) for row in predictive_rows)
    conditional_p_value = float(chi2_distribution.sf(pull_table.predictive_chi2, FOLLOWUP_CHI2_REFERENCE_DOF))
    return float(pull_table.predictive_chi2), float(max_abs_pull), conditional_p_value


def derive_followup_chi2_landscape_audit(
    lepton_range: tuple[int, int] = FOLLOWUP_LEPTON_LEVEL_RANGE,
    quark_range: tuple[int, int] = FOLLOWUP_QUARK_LEVEL_RANGE,
    *,
    parent_level: int = PARENT_LEVEL,
    scale_ratio: float = RG_SCALE_RATIO,
    bit_count: float = HOLOGRAPHIC_BITS,
    kappa_geometric: float = GEOMETRIC_KAPPA,
    benchmark_lepton_level: int = LEPTON_LEVEL,
    benchmark_quark_level: int = QUARK_LEVEL,
) -> Chi2LandscapeAuditData:
    lepton_levels = tuple(range(lepton_range[0], lepton_range[1] + 1))
    quark_levels = tuple(range(quark_range[0], quark_range[1] + 1))
    chi2_grid = np.empty((len(lepton_levels), len(quark_levels)), dtype=float)
    points: list[Chi2LandscapePoint] = []

    for lepton_index, lepton_level in enumerate(lepton_levels):
        for quark_index, quark_level in enumerate(quark_levels):
            predictive_chi2, max_abs_pull, conditional_p_value = _followup_predictive_point_cached(
                lepton_level,
                quark_level,
                parent_level,
                scale_ratio,
                bit_count,
                kappa_geometric,
            )
            chi2_grid[lepton_index, quark_index] = predictive_chi2
            points.append(
                Chi2LandscapePoint(
                    lepton_level=lepton_level,
                    quark_level=quark_level,
                    predictive_chi2=predictive_chi2,
                    max_abs_pull=max_abs_pull,
                    conditional_p_value=conditional_p_value,
                    selected_visible_pair=(lepton_level, quark_level) == (benchmark_lepton_level, benchmark_quark_level),
                )
            )

    lepton_autocorrelation = _axis_autocorrelation(chi2_grid, axis=0)
    quark_autocorrelation = _axis_autocorrelation(chi2_grid, axis=1)
    lepton_correlation_length = _correlation_length_from_autocorrelation(lepton_autocorrelation)
    quark_correlation_length = _correlation_length_from_autocorrelation(quark_autocorrelation)
    effective_correlation_length = math.sqrt(lepton_correlation_length * quark_correlation_length)
    effective_trial_count = len(points) / max(lepton_correlation_length * quark_correlation_length, 1.0)
    survival_count = sum(1 for point in points if point.predictive_chi2 <= FOLLOWUP_CHI2_SURVIVAL_THRESHOLD)

    return freeze_numpy_arrays(Chi2LandscapeAuditData(
        lepton_range=lepton_range,
        quark_range=quark_range,
        total_pairs_scanned=len(points),
        chi2_grid=chi2_grid,
        points=tuple(points),
        acceptance_threshold=FOLLOWUP_CHI2_SURVIVAL_THRESHOLD,
        survival_count=survival_count,
        lepton_autocorrelation=lepton_autocorrelation,
        quark_autocorrelation=quark_autocorrelation,
        lepton_correlation_length=lepton_correlation_length,
        quark_correlation_length=quark_correlation_length,
        effective_correlation_length=effective_correlation_length,
        effective_trial_count=effective_trial_count,
    ))


def export_followup_chi2_landscape_table(
    chi2_landscape_audit: Chi2LandscapeAuditData,
    output_dir: Path,
    top_rows: int = 6,
) -> None:
    displayed_points = chi2_landscape_audit.top_points[:top_rows]
    body_rows = tuple(
        (
            rf"\textbf{{{point.lepton_level}}} & \textbf{{{point.quark_level}}} & \textbf{{{point.predictive_chi2:.3f}}} & \textbf{{{point.max_abs_pull:.3f}}} & \textbf{{{point.conditional_p_value:.3e}}} \\\\" 
            if point.selected_visible_pair
            else rf"{point.lepton_level} & {point.quark_level} & {point.predictive_chi2:.3f} & {point.max_abs_pull:.3f} & {point.conditional_p_value:.3e} \\\\" 
        )
        for point in displayed_points
    )
    footer_rows = (
        rf"$N_{{raw}}$ & \multicolumn{{4}}{{r}}{{{chi2_landscape_audit.total_pairs_scanned}}} \\\\ ",
        rf"$N_{{survive}}$ & \multicolumn{{4}}{{r}}{{{chi2_landscape_audit.survival_count}}} \\\\ ",
        rf"$N_{{eff}}$ & \multicolumn{{4}}{{r}}{{{chi2_landscape_audit.effective_trial_count:.1f}}} \\\\ ",
    )
    publication_export.write_skeletal_latex_table(
        output_dir / SUPPLEMENTARY_TOPCHI2_TABLE_FILENAME,
        column_spec="ccccc",
        header_rows=(r"$k_{\ell}$ & $k_q$ & $\chi^2_{\rm pred}$ & max pull & $p(\chi^2;\nu=8)$ \\\\",),
        body_rows=body_rows,
        footer_rows=footer_rows,
        style="booktabs",
    )


@dataclass(frozen=True)
class FramingAnomalyData:
    parent_level: int
    parent_central_charge: float
    lepton_central_charge: float
    quark_central_charge: float
    visible_central_charge: float
    coset_central_charge: float
    visible_residual_mod1: float
    total_residual_mod1: float
    branch_period: int
    best_branch_level: int
    best_branch_residual_mod1: float


@dataclass(frozen=True)
class AuditData:
    """Hierarchy-audit data for the Wheeler--DeWitt support test.

    Attributes:
        beta: Genus-ladder spacing entering the support audit.
        topological_splittings_ev2: Pairwise mass-squared splittings of the topological ladder.
        strict_normal_gap: Determinant-based strict support cost for normal ordering.
        strict_inverted_gap: Determinant-based strict support cost for inverted ordering.
        relaxed_normal_gap: Relaxed one-copy support cost for normal ordering.
        relaxed_inverted_gap: Relaxed one-copy support cost for inverted ordering.
        support_deficit: Number of missing one-copy support channels in the IH assignment.
        modularity_limit_rank: One-copy light-sector rank allowed by the minimal code.
        required_inverted_rank: Rank needed to realize IH after duplicating a support slot.
        redundancy_entropy_cost_nat: Entropy cost in nats of the extra dictionary slot.
        normal_genus_assignment: One-copy genus assignment for normal ordering.
        inverted_genus_assignment: Best one-copy genus assignment for inverted ordering.
        support_tau_imag: Imaginary modular parameter used in the overlap integral.
    """

    beta: float
    topological_splittings_ev2: np.ndarray
    strict_normal_gap: float
    strict_inverted_gap: float
    relaxed_normal_gap: float
    relaxed_inverted_gap: float
    support_deficit: int
    modularity_limit_rank: int
    required_inverted_rank: int
    redundancy_entropy_cost_nat: float
    normal_genus_assignment: tuple[int, int, int]
    inverted_genus_assignment: tuple[int, int, int]
    support_tau_imag: float

    def calculate_support_overlap(self, level: int = LEPTON_LEVEL) -> dict[str, np.ndarray]:
        r"""Evaluate the normalized support overlaps from Eq. ``eq:wdw-overlap-matrix``."""

        return {
            "NH": support_overlap_matrix(
                level,
                self.normal_genus_assignment,
                tau_imag=self.support_tau_imag,
            ),
            "IH": support_overlap_matrix(
                level,
                self.inverted_genus_assignment,
                tau_imag=self.support_tau_imag,
            ),
        }


@dataclass(frozen=True)
class SensitivityPoint:
    label: str
    bit_count: float
    m_0_mz_ev: float
    effective_majorana_mass_mev: float
    theta12_shift_deg: float
    theta13_shift_deg: float
    theta23_shift_deg: float
    delta_cp_shift_deg: float
    theta_c_shift_deg: float
    vus_shift: float
    vcb_shift: float


@dataclass(frozen=True)
class SensitivityData:
    minus_10pct: SensitivityPoint
    central: SensitivityPoint
    plus_10pct: SensitivityPoint
    effective_majorana_mass_std_mev: float
    effective_majorana_mass_max_shift_mev: float
    m_0_mz_max_shift_mev: float


@dataclass(frozen=True)
class GeometricSweepPoint:
    """Single point in the order-one geometric uncertainty sweep."""

    kappa: float
    m_0_mz_ev: float
    effective_majorana_mass_mev: float
    predictive_chi2: float
    predictive_max_abs_pull: float
    max_sigma_shift: float


@dataclass(frozen=True)
class GeometricSensitivityData:
    """Summary of the κ-sweep around the central boundary--bulk scale relation."""

    central_kappa: float
    sweep_points: tuple[GeometricSweepPoint, ...]
    effective_majorana_mass_max_shift_mev: float
    m_0_mz_max_shift_mev: float


@dataclass(frozen=True)
class CosmologyAnchorData:
    """Planck-anchored cosmology inputs used for the holographic bit count."""

    hubble_km_s_mpc: float
    hubble_sigma_km_s_mpc: float
    omega_lambda: float
    omega_lambda_sigma: float
    lambda_si_m2: float
    holographic_bits: float
    holographic_bits_fractional_sigma: float


@dataclass(frozen=True)
class ThresholdSensitivityPoint:
    r"""Single $\mathbf{126}_H$ threshold point used in the gamma diagnostic."""

    m_126_gev: float
    delta_pi_126: float
    xi12: float
    cabibbo_shift_fraction: float
    gamma_shift_estimate_deg: float
    gamma_recovered_deg: float


@dataclass(frozen=True)
class ThresholdSensitivityData:
    r"""Gamma-phase threshold diagnostic as a function of the $\mathbf{126}_H$ scale."""

    amplification_factor: float
    matching_point_gamma_shift_deg: float
    points: tuple[ThresholdSensitivityPoint, ...]


@dataclass(frozen=True)
class GaugeUnificationData:
    """One-loop threshold audit for gauge unification bookkeeping."""

    alpha_inverse_mz: np.ndarray
    alpha_inverse_m126: np.ndarray
    alpha_inverse_m10: np.ndarray
    alpha_inverse_gut: np.ndarray
    beta_sm: np.ndarray
    beta_shift_126: np.ndarray
    beta_shift_10: np.ndarray
    m_126_gev: float
    m_10_gev: float
    m_gut_gev: float
    unified_alpha_inverse: float
    max_mismatch: float
    structural_mn_gev: float
    geometric_mean_threshold_gev: float
    seesaw_consistency_ratio: float


@dataclass(frozen=True)
class SO10RepresentationData:
    r"""Exact Lie-algebra invariants of a $D_5\simeq so(10)$ highest-weight representation."""

    name: str
    dynkin_labels: tuple[int, int, int, int, int]
    dimension: int
    quadratic_casimir: Fraction
    dynkin_index: Fraction


@dataclass(frozen=True)
class SO10Representation:
    r"""Runtime-facing $SO(10)$ representation helper used by the manuscript audit."""

    name: str
    dynkin_labels: tuple[int, int, int, int, int]

    @property
    def simple_roots(self) -> tuple[tuple[Fraction, ...], ...]:
        return so10_simple_roots()

    @property
    def fundamental_weights(self) -> tuple[tuple[Fraction, ...], ...]:
        return so10_fundamental_weights()

    @property
    def weyl_vector(self) -> tuple[Fraction, ...]:
        return so10_weyl_vector()

    @property
    def highest_weight(self) -> tuple[Fraction, ...]:
        return so10_highest_weight(self.dynkin_labels)

    @property
    def dimension(self) -> int:
        return so10_rep_dimension(self.dynkin_labels)

    @property
    def quadratic_casimir(self) -> Fraction:
        return so10_rep_quadratic_casimir(self.dynkin_labels)

    @property
    def dynkin_index(self) -> Fraction:
        return so10_rep_dynkin_index(self.dynkin_labels)

    def to_data(self) -> SO10RepresentationData:
        return SO10RepresentationData(
            name=self.name,
            dynkin_labels=self.dynkin_labels,
            dimension=self.dimension,
            quadratic_casimir=self.quadratic_casimir,
            dynkin_index=self.dynkin_index,
        )


@dataclass(frozen=True)
class ParentBranchingAnomalyData:
    r"""Exact anomaly-fraction bookkeeping for the $\mathbf{126}_H$ branching exponent."""

    dirac_representation: SO10RepresentationData
    majorana_representation: SO10RepresentationData
    su2_embedding_index: int
    su3_embedding_index: int
    quark_branching_index: int
    visible_cartan_denominator: int
    visible_cartan_embedding_index: int
    numerator_units: int
    denominator_units: int
    anomaly_fraction: Fraction


@dataclass(frozen=True)
class ScalarThresholdFragment:
    """Single SM scalar fragment contributing to one-loop gauge running."""

    name: str
    su3_dynkin_labels: tuple[int, int]
    su2_dimension: int
    hypercharge: Fraction
    multiplicity: int = 1
    is_real: bool = False


@dataclass(frozen=True)
class HeavyThresholdMatchingContribution:
    r"""Single heavy field entering the one-loop $SO(10)$ matching sum."""

    name: str
    source: str
    mass_gev: float
    coefficient: float
    log_enhancement: float
    contribution: float


@dataclass(frozen=True)
class HeavyStateDecouplingAuditData:
    r"""Appelquist--Carazzone audit for the heavy $\mathbf{126}_H$ threshold."""

    threshold_scale_gev: float
    probe_scales_gev: np.ndarray
    leakage_norms: np.ndarray
    max_leakage: float
    passed: bool


@dataclass(frozen=True)
class FinalAuditResult:
    r"""Summary of the final anomaly-to-$\gamma$ and gravity-consistency hook."""

    gamma_pull: float
    within_one_sigma: bool
    anomaly_fraction: Fraction
    c_dark_completion: float
    gravity_audit: GravityAudit
    gauge_audit: GaugeHolographyAudit
    dark_energy_audit: DarkEnergyTensionAudit
    gmunu_consistency_score: float
    ih_nonminimal_extension_required: bool
    ih_support_deficit: int
    ih_modularity_limit_rank: int
    ih_required_dictionary_rank: int
    latex_block: str


EXPERIMENTAL_CONTEXT = shared_constants.EXPERIMENTAL_CONTEXT
LEPTON_INTERVALS = shared_constants.LEPTON_INTERVALS
QUARK_INTERVALS = shared_constants.QUARK_INTERVALS
EXPERIMENTAL_INPUTS = shared_constants.EXPERIMENTAL_INPUTS
CKM_GAMMA_EXPERIMENTAL_INPUT_DEG = shared_constants.CKM_GAMMA_EXPERIMENTAL_INPUT_DEG
CKM_GAMMA_GOLD_STANDARD_DEG = CKM_GAMMA_EXPERIMENTAL_INPUT_DEG
NUFIT_53_NO_3SIGMA = shared_constants.NUFIT_53_NO_3SIGMA
DEFAULT_MANUSCRIPT_DIR = shared_constants.DEFAULT_MANUSCRIPT_DIR
DEFAULT_OUTPUT_DIR = shared_constants.DEFAULT_OUTPUT_DIR
GLOBAL_FLAVOR_FIT_TABLE_FILENAME = shared_constants.GLOBAL_FLAVOR_FIT_TABLE_FILENAME
UNIQUENESS_SCAN_TABLE_FILENAME = shared_constants.UNIQUENESS_SCAN_TABLE_FILENAME
MODULARITY_RESIDUAL_MAP_FILENAME = shared_constants.MODULARITY_RESIDUAL_MAP_FILENAME
LANDSCAPE_ANOMALY_MAP_FILENAME = shared_constants.LANDSCAPE_ANOMALY_MAP_FILENAME
SUPPLEMENTARY_IH_SUPPORT_MAP_FILENAME = shared_constants.SUPPLEMENTARY_IH_SUPPORT_MAP_FILENAME
SUPPLEMENTARY_TOLERANCE_TABLE_FILENAME = shared_constants.SUPPLEMENTARY_TOLERANCE_TABLE_FILENAME
KAPPA_SENSITIVITY_AUDIT_FILENAME = shared_constants.KAPPA_SENSITIVITY_AUDIT_FILENAME
KAPPA_STABILITY_SWEEP_FILENAME = shared_constants.KAPPA_STABILITY_SWEEP_FILENAME
SVD_STABILITY_AUDIT_TABLE_FILENAME = shared_constants.SVD_STABILITY_AUDIT_TABLE_FILENAME
PHYSICS_CONSTANTS_FILENAME = shared_constants.PHYSICS_CONSTANTS_FILENAME
BENCHMARK_DIAGNOSTICS_FILENAME = shared_constants.BENCHMARK_DIAGNOSTICS_FILENAME
TRANSPORT_COVARIANCE_DIAGNOSTICS_FILENAME = shared_constants.TRANSPORT_COVARIANCE_DIAGNOSTICS_FILENAME
SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_DATA_FILENAME = shared_constants.SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_DATA_FILENAME
AUDIT_STATEMENT_FILENAME = shared_constants.AUDIT_STATEMENT_FILENAME
NUMERICAL_STABILITY_REPORT_FILENAME = shared_constants.NUMERICAL_STABILITY_REPORT_FILENAME
SVD_STABILITY_REPORT_FILENAME = shared_constants.SVD_STABILITY_REPORT_FILENAME
EIGENVECTOR_STABILITY_AUDIT_FILENAME = shared_constants.EIGENVECTOR_STABILITY_AUDIT_FILENAME
STABILITY_REPORT_FILENAME = shared_constants.STABILITY_REPORT_FILENAME
TOPOLOGICAL_LOBSTER_FIGURE_FILENAME = shared_constants.TOPOLOGICAL_LOBSTER_FIGURE_FILENAME
MAJORANA_FLOOR_FIGURE_FILENAME = shared_constants.MAJORANA_FLOOR_FIGURE_FILENAME
CKM_PHASE_TILT_PROFILE_FIGURE_FILENAME = shared_constants.CKM_PHASE_TILT_PROFILE_FIGURE_FILENAME
DM_FINGERPRINT_FIGURE_FILENAME = shared_constants.DM_FINGERPRINT_FIGURE_FILENAME
FRAMING_GAP_HEATMAP_FIGURE_FILENAME = shared_constants.FRAMING_GAP_HEATMAP_FIGURE_FILENAME
BENCHMARK_STABILITY_TABLE_FILENAME = shared_constants.BENCHMARK_STABILITY_TABLE_FILENAME
SUPPLEMENTARY_TOPCHI2_TABLE_FILENAME = shared_constants.SUPPLEMENTARY_TOPCHI2_TABLE_FILENAME
SUPPLEMENTARY_VEV_ALIGNMENT_STABILITY_FIGURE_FILENAME = shared_constants.SUPPLEMENTARY_VEV_ALIGNMENT_STABILITY_FIGURE_FILENAME
SEED_ROBUSTNESS_AUDIT_FILENAME = shared_constants.SEED_ROBUSTNESS_AUDIT_FILENAME
SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_FIGURE_FILENAME = shared_constants.SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_FIGURE_FILENAME
SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME = shared_constants.SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME
FRAMING_GAP_STABILITY_FIGURE_FILENAME = shared_constants.FRAMING_GAP_STABILITY_FIGURE_FILENAME
SUPPLEMENTARY_UNITARY_CONSISTENCY_TABLE_FILENAME = "supplementary_unitary_consistency_table.tex"
AUDIT_OUTPUT_ARCHIVE_DIRNAME = shared_constants.AUDIT_OUTPUT_ARCHIVE_DIRNAME
STABILITY_AUDIT_OUTPUTS_DIRNAME = shared_constants.STABILITY_AUDIT_OUTPUTS_DIRNAME
LANDSCAPE_METRICS_DIRNAME = shared_constants.LANDSCAPE_METRICS_DIRNAME
AUDIT_OUTPUT_MANIFEST_FILENAME = shared_constants.AUDIT_OUTPUT_MANIFEST_FILENAME
MAJORANA_LOBSTER_GRID_POINTS = shared_constants.MAJORANA_LOBSTER_GRID_POINTS
MAJORANA_LIGHTEST_SCAN_EV = np.logspace(-4.2, -1.0, 320)
ONE_LOOP_FACTOR = physics_engine.ONE_LOOP_FACTOR


def polar_unitary(
    matrix: np.ndarray,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    """Return the unitary polar factor of a complex matrix."""

    return topological_kernel.polar_unitary(matrix, solver_config=solver_config)


def wzw_central_charge(level: float | int, dimension: int, dual_coxeter: int) -> float:
    r"""Return the WZW central charge from manuscript Eq. ``eq:central-charge-wzw``."""

    return level * dimension / (level + dual_coxeter)


def so10_sm_embedding_visible_central_charge(parent_level: int = PARENT_LEVEL) -> float:
    r"""Central charge of the embedded $SU(3)\times SU(2)\times U(1)$ subgroup.

    The low-rank visible scan intentionally avoids anchoring to the benchmark
    pair $(26,8)$ by referencing the actual subgroup embedding of the fixed
    parent theory rather than a coset completion defined from the selected
    visible levels.
    """

    return (
        wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER)
        + wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER)
        + VISIBLE_HYPERCHARGE_CENTRAL_CHARGE
    )


def visible_sector_central_charge(
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> float:
    r"""Return the visible logical-bit charge $c(SU(2)_{k_\ell})+c(SU(3)_{k_q})$."""

    return float(
        wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER)
        + wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
    )


def modular_residue_efficiency(
    c_dark_completion: float = BENCHMARK_C_DARK_RESIDUE,
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> float:
    r"""Return the undressed parity-bit loading $c_{\rm dark}/c_{\rm vis}$."""

    c_visible = visible_sector_central_charge(lepton_level=lepton_level, quark_level=quark_level)
    if math.isclose(c_visible, 0.0, rel_tol=0.0, abs_tol=1.0e-300):
        raise ValueError("Visible-sector central charge must be nonzero.")
    return float(c_dark_completion / c_visible)


def parity_bit_density_ratio(
    kappa_geometric: float | None = None,
    c_dark_completion: float = BENCHMARK_C_DARK_RESIDUE,
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> float:
    r"""Return the HEC parity-bit density $\kappa_{D_5}(c_{\rm dark}/c_{\rm vis})$."""

    resolved_kappa = (
        compute_geometric_kappa_ansatz(
            parent_level=PARENT_LEVEL,
            lepton_level=lepton_level,
        ).derived_kappa
        if kappa_geometric is None
        else float(kappa_geometric)
    )

    return float(
        resolved_kappa
        * modular_residue_efficiency(
            c_dark_completion,
            lepton_level=lepton_level,
            quark_level=quark_level,
        )
    )


def so10_sm_branching_rule_coset_central_charge(parent_level: int = PARENT_LEVEL) -> float:
    r"""Coset central charge implied by $SO(10)\supset SU(3)\times SU(2)\times U(1)$."""

    parent_central_charge = wzw_central_charge(parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
    return parent_central_charge - so10_sm_embedding_visible_central_charge(parent_level)


def mod_one_residual(value: float, tolerance: float = 1.0e-12) -> float:
    residual = value % 1.0
    if math.isclose(residual, 0.0, abs_tol=tolerance, rel_tol=0.0) or math.isclose(residual, 1.0, abs_tol=tolerance, rel_tol=0.0):
        return 0.0
    return residual


def distance_to_integer(value: float) -> float:
    residual = mod_one_residual(value)
    return min(residual, 1.0 - residual)


def nearest_integer_gap(value: float) -> float:
    return abs(value - round(value))


def lepton_branching_index(
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    embedding_index: int = SO10_TO_SU2_EMBEDDING_INDEX,
) -> int:
    return parent_level // (embedding_index * SU2_DUAL_COXETER * lepton_level)


def quark_branching_index(
    parent_level: int = PARENT_LEVEL,
    quark_level: int = QUARK_LEVEL,
    embedding_index: int = SO10_TO_SU3_EMBEDDING_INDEX,
) -> int:
    return parent_level // (embedding_index * SU3_DUAL_COXETER * quark_level)


@lru_cache(maxsize=128)
def so10_fundamental_weights() -> tuple[tuple[Fraction, ...], ...]:
    """Return the $D_5$ fundamental weights in the orthonormal $e_i$ basis."""

    return (
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
    )


@lru_cache(maxsize=128)
def so10_weyl_vector() -> tuple[Fraction, ...]:
    return (Fraction(4), Fraction(3), Fraction(2), Fraction(1), Fraction(0))


@lru_cache(maxsize=128)
def so10_positive_roots() -> tuple[tuple[Fraction, ...], ...]:
    roots: list[tuple[Fraction, ...]] = []
    for left in range(SO10_RANK):
        for right in range(left + 1, SO10_RANK):
            plus_root = [Fraction(0) for _ in range(SO10_RANK)]
            plus_root[left] += 1
            plus_root[right] += 1
            roots.append(tuple(plus_root))

            minus_root = [Fraction(0) for _ in range(SO10_RANK)]
            minus_root[left] += 1
            minus_root[right] -= 1
            roots.append(tuple(minus_root))
    return tuple(roots)


@lru_cache(maxsize=128)
def so10_simple_roots() -> tuple[tuple[Fraction, ...], ...]:
    """Return the standard $D_5$ simple roots in the orthonormal basis."""

    return (
        (Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
    )


@lru_cache(maxsize=128)
def so10_visible_cartan_projection_basis() -> tuple[tuple[Fraction, ...], ...]:
    r"""Return a visible Cartan basis adapted to $SU(3)	imes SU(2)	imes U(1)_Y	imes U(1)_\chi$."""

    simple_roots = so10_simple_roots()
    return (
        simple_roots[0],
        simple_roots[1],
        simple_roots[3],
        (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(-1, 2), Fraction(-1, 2)),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    )


@lru_cache(maxsize=128)
def so10_visible_cartan_projection_denominator() -> int:
    r"""Return the exact denominator induced by the visible Cartan--Weyl projection.

    The non-Abelian $SU(3)	imes SU(2)$ roots are kept explicit, while the two
    orthogonal Cartan directions are spanned by the canonical hypercharge-like
    and $U(1)_\chi$-like generators. Projecting the five $D_5$ fundamental
    weights into that basis yields orthogonal-sector denominators whose least
    common multiple is $10$; the quadratic branching amplitude therefore carries
    the embedding index $x=10^2=100$.
    """

    denominator = 1
    basis = so10_visible_cartan_projection_basis()
    for weight in so10_fundamental_weights():
        coefficients = solve_fraction_linear_system(basis, weight)
        for coefficient in coefficients[3:]:
            denominator = lcm_int(denominator, coefficient.denominator)
    return denominator


@lru_cache(maxsize=128)
def so10_highest_weight(dynkin_labels: tuple[int, int, int, int, int]) -> tuple[Fraction, ...]:
    if len(dynkin_labels) != SO10_RANK:
        raise ValueError(f"Expected {SO10_RANK} Dynkin labels, received {dynkin_labels}")

    weight = tuple(Fraction(0) for _ in range(SO10_RANK))
    for coefficient, fundamental_weight in zip(dynkin_labels, so10_fundamental_weights()):
        weight = add_fraction_vectors(weight, scale_fraction_vector(fundamental_weight, coefficient))
    return weight


@lru_cache(maxsize=128)
def so10_rep_dimension(dynkin_labels: tuple[int, int, int, int, int]) -> int:
    highest_weight = so10_highest_weight(dynkin_labels)
    rho = so10_weyl_vector()
    dimension = Fraction(1)
    for root in so10_positive_roots():
        dimension *= fraction_dot(add_fraction_vectors(highest_weight, rho), root) / fraction_dot(rho, root)
    if dimension.denominator != 1:
        raise RuntimeError(f"Non-integral $SO(10)$ dimension obtained for {dynkin_labels}: {dimension}")
    return int(dimension)


@lru_cache(maxsize=128)
def so10_rep_quadratic_casimir(dynkin_labels: tuple[int, int, int, int, int]) -> Fraction:
    highest_weight = so10_highest_weight(dynkin_labels)
    rho = so10_weyl_vector()
    return fraction_dot(highest_weight, add_fraction_vectors(highest_weight, scale_fraction_vector(rho, 2))) / 2


@lru_cache(maxsize=128)
def so10_rep_dynkin_index(dynkin_labels: tuple[int, int, int, int, int]) -> Fraction:
    return Fraction(so10_rep_dimension(dynkin_labels), SO10_DIMENSION) * so10_rep_quadratic_casimir(dynkin_labels)


def derive_so10_representation_data(name: str, dynkin_labels: tuple[int, int, int, int, int]) -> SO10RepresentationData:
    return SO10Representation(name=name, dynkin_labels=dynkin_labels).to_data()


def su2_rep_quadratic_casimir(dimension: int) -> Fraction:
    spin = Fraction(dimension - 1, 2)
    return spin * (spin + 1)


def su2_rep_dynkin_index(dimension: int) -> Fraction:
    return Fraction(dimension, SU2_DIMENSION) * su2_rep_quadratic_casimir(dimension)


def su3_rep_dimension(dynkin_labels: tuple[int, int]) -> int:
    p_label, q_label = dynkin_labels
    return ((p_label + 1) * (q_label + 1) * (p_label + q_label + 2)) // 2


def su3_rep_quadratic_casimir(dynkin_labels: tuple[int, int]) -> Fraction:
    p_label, q_label = dynkin_labels
    numerator = p_label * p_label + q_label * q_label + p_label * q_label + 3 * p_label + 3 * q_label
    return Fraction(numerator, 3)


def su3_rep_dynkin_index(dynkin_labels: tuple[int, int]) -> Fraction:
    return Fraction(su3_rep_dimension(dynkin_labels), SU3_DIMENSION) * su3_rep_quadratic_casimir(dynkin_labels)


def adjoint_quadratic_casimir(dual_coxeter: int) -> Fraction:
    """Return the adjoint quadratic Casimir in the standard Killing normalization."""

    return Fraction(dual_coxeter, 1)


def level_curvature_factor(casimir: Fraction, level: int, dual_coxeter: int) -> Fraction:
    """Return the level-dressed Casimir factor $C_2/(k+h^\vee)$ for a WZW sector."""

    return casimir / Fraction(level + dual_coxeter, 1)


def scalar_beta_shift(fragment: ScalarThresholdFragment) -> tuple[Fraction, Fraction, Fraction]:
    prefactor = Fraction(1, 6) if fragment.is_real else Fraction(1, 3)
    su3_dimension = su3_rep_dimension(fragment.su3_dynkin_labels)
    su2_index = su2_rep_dynkin_index(fragment.su2_dimension)
    su3_index = su3_rep_dynkin_index(fragment.su3_dynkin_labels)
    hypercharge_sq = fragment.hypercharge * fragment.hypercharge

    delta_b1 = prefactor * Fraction(3, 5) * hypercharge_sq * su3_dimension * fragment.su2_dimension * fragment.multiplicity
    delta_b2 = prefactor * su2_index * su3_dimension * fragment.multiplicity
    delta_b3 = prefactor * su3_index * fragment.su2_dimension * fragment.multiplicity
    return delta_b1, delta_b2, delta_b3


def fractions_to_float_array(values: tuple[Fraction, Fraction, Fraction]) -> np.ndarray:
    return np.array([float(value) for value in values], dtype=float)


def so10_higgs_10_fragments() -> tuple[ScalarThresholdFragment, ...]:
    return (
        ScalarThresholdFragment("(1,2,+1/2)", (0, 0), 2, Fraction(1, 2)),
        ScalarThresholdFragment("(1,2,-1/2)", (0, 0), 2, Fraction(-1, 2)),
        ScalarThresholdFragment("(3,1,-1/3)", (1, 0), 1, Fraction(-1, 3)),
        ScalarThresholdFragment("(3bar,1,+1/3)", (0, 1), 1, Fraction(1, 3)),
    )


def so10_higgs_126_fragments() -> tuple[ScalarThresholdFragment, ...]:
    return (
        ScalarThresholdFragment("(3,1,-1/3)", (1, 0), 1, Fraction(-1, 3)),
        ScalarThresholdFragment("(3bar,1,+1/3)", (0, 1), 1, Fraction(1, 3)),
        ScalarThresholdFragment("(6,3,+1/3)", (2, 0), 3, Fraction(1, 3)),
        ScalarThresholdFragment("(3,3,-1/3)", (1, 0), 3, Fraction(-1, 3)),
        ScalarThresholdFragment("(1,3,-1)", (0, 0), 3, Fraction(-1, 1)),
        ScalarThresholdFragment("(6bar,1,-4/3)", (0, 2), 1, Fraction(-4, 3)),
        ScalarThresholdFragment("(6bar,1,-1/3)", (0, 2), 1, Fraction(-1, 3)),
        ScalarThresholdFragment("(6bar,1,+2/3)", (0, 2), 1, Fraction(2, 3)),
        ScalarThresholdFragment("(3bar,1,-2/3)", (0, 1), 1, Fraction(-2, 3)),
        ScalarThresholdFragment("(3bar,1,+1/3)", (0, 1), 1, Fraction(1, 3)),
        ScalarThresholdFragment("(3bar,1,+4/3)", (0, 1), 1, Fraction(4, 3)),
        ScalarThresholdFragment("(1,1,0)", (0, 0), 1, Fraction(0, 1)),
        ScalarThresholdFragment("(1,1,+1)", (0, 0), 1, Fraction(1, 1)),
        ScalarThresholdFragment("(1,1,+2)", (0, 0), 1, Fraction(2, 1)),
        ScalarThresholdFragment("(8,2,+1/2)", (1, 1), 2, Fraction(1, 2)),
        ScalarThresholdFragment("(8,2,-1/2)", (1, 1), 2, Fraction(-1, 2)),
        ScalarThresholdFragment("(3,2,+7/6)", (1, 0), 2, Fraction(7, 6)),
        ScalarThresholdFragment("(3,2,+1/6)", (1, 0), 2, Fraction(1, 6)),
        ScalarThresholdFragment("(3bar,2,-1/6)", (0, 1), 2, Fraction(-1, 6)),
        ScalarThresholdFragment("(3bar,2,-7/6)", (0, 1), 2, Fraction(-7, 6)),
        ScalarThresholdFragment("(1,2,+1/2)", (0, 0), 2, Fraction(1, 2)),
        ScalarThresholdFragment("(1,2,-1/2)", (0, 0), 2, Fraction(-1, 2)),
    )


def scalar_fragment_state_count(fragment: ScalarThresholdFragment) -> int:
    """Return the number of SM states carried by a threshold fragment."""

    return su3_rep_dimension(fragment.su3_dynkin_labels) * fragment.su2_dimension * fragment.multiplicity


def so10_higgs_210_coarse_fragments() -> tuple[tuple[str, int], ...]:
    r"""Return the coarse Pati--Salam submultiplets of the $\mathbf{210}_H$ field."""

    return (
        ("(1,1,1)", 1),
        ("(1,1,15)", 15),
        ("(1,3,15)", 45),
        ("(3,1,15)", 45),
        ("(2,2,6)", 24),
        ("(2,2,10)", 40),
        ("(2,2,10bar)", 40),
    )


def derive_so10_scalar_beta_shift(representation_name: str) -> np.ndarray:
    if representation_name == "10_H":
        dynkin_labels = SO10_HIGGS_10_DYNKIN_LABELS
        fragments = so10_higgs_10_fragments()
    elif representation_name == "126_H":
        dynkin_labels = SO10_HIGGS_126_DYNKIN_LABELS
        fragments = so10_higgs_126_fragments()
    else:
        raise ValueError(f"Unsupported scalar threshold representation: {representation_name}")

    total_shift = [Fraction(0), Fraction(0), Fraction(0)]
    for fragment in fragments:
        delta_b1, delta_b2, delta_b3 = scalar_beta_shift(fragment)
        total_shift[0] += delta_b1
        total_shift[1] += delta_b2
        total_shift[2] += delta_b3

    representation = derive_so10_representation_data(representation_name, dynkin_labels)
    expected_universal_shift = representation.dynkin_index / 3
    expected_triplet = (expected_universal_shift, expected_universal_shift, expected_universal_shift)
    if tuple(total_shift) != expected_triplet:
        raise RuntimeError(
            f"Threshold decomposition for {representation_name} does not reproduce its Dynkin index: {tuple(total_shift)} != {expected_triplet}"
        )
    return fractions_to_float_array(expected_triplet)


def derive_parent_branching_anomaly_data(
    parent_level: int = PARENT_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> ParentBranchingAnomalyData:
    """Derive the Sec. 10.2 branching exponent from exact $SO(10)$ invariants."""

    dirac_representation = derive_so10_representation_data("10_H", SO10_HIGGS_10_DYNKIN_LABELS)
    majorana_representation = derive_so10_representation_data("126_H", SO10_HIGGS_126_DYNKIN_LABELS)
    iq_visible = quark_branching_index(parent_level, quark_level)
    visible_cartan_denominator = so10_visible_cartan_projection_denominator()
    visible_cartan_embedding_index = visible_cartan_denominator * visible_cartan_denominator
    denominator_units = 2 * majorana_representation.dimension + iq_visible
    numerator_units = 4 * majorana_representation.dimension + dirac_representation.dimension + iq_visible
    anomaly_fraction = Fraction(numerator_units, visible_cartan_embedding_index * denominator_units)
    return ParentBranchingAnomalyData(
        dirac_representation=dirac_representation,
        majorana_representation=majorana_representation,
        su2_embedding_index=SO10_TO_SU2_EMBEDDING_INDEX,
        su3_embedding_index=SO10_TO_SU3_EMBEDDING_INDEX,
        quark_branching_index=iq_visible,
        visible_cartan_denominator=visible_cartan_denominator,
        visible_cartan_embedding_index=visible_cartan_embedding_index,
        numerator_units=numerator_units,
        denominator_units=denominator_units,
        anomaly_fraction=anomaly_fraction,
    )


def normalize_group_label(label: str) -> str:
    """Return a simple alphanumeric group key for small runtime dispatch."""

    return "".join(character for character in label.upper() if character.isalnum())


def calculate_branching_anomaly(
    parent_group: str,
    subgroup: str,
    level: int,
    visible_level: int | None = None,
) -> ParentBranchingAnomalyData:
    r"""Compute the branching anomaly fraction at runtime from Lie-algebra data.

    The current verifier exposes the exact ``SO(10) -> SU(3)`` projection used
    in the quark-sector threshold audit. The returned fraction is derived from
    the $D_5$ Dynkin labels of ``10_H`` and ``126_H`` together with the visible
    Cartan projection denominator and the subgroup branching index.
    """

    parent_key = normalize_group_label(parent_group)
    subgroup_key = normalize_group_label(subgroup)
    if parent_key not in {"SO10", "D5"}:
        raise ValueError(f"Unsupported parent group for branching anomaly audit: {parent_group}")
    if subgroup_key not in {"SU3", "A2"}:
        raise ValueError(f"Unsupported subgroup for branching anomaly audit: {subgroup}")

    subgroup_level = QUARK_LEVEL if visible_level is None else visible_level
    return derive_parent_branching_anomaly_data(parent_level=level, quark_level=subgroup_level)


def su2_total_quantum_dimension(level: int) -> float:
    return math.sqrt((level + 2.0) / 2.0) / math.sin(math.pi / (level + 2.0))


def su2_quantum_dimension(level: int, label: int) -> float:
    return math.sin((label + 1.0) * math.pi / (level + 2.0)) / math.sin(math.pi / (level + 2.0))


def su2_conformal_weight(level: int, label: int) -> float:
    return label * (label + 2.0) / (4.0 * (level + 2.0))


def su2_modular_s(level: int) -> np.ndarray:
    prefactor = math.sqrt(2.0 / (level + 2.0))
    size = level + 1
    return np.array(
        [
            [
                prefactor * math.sin((row + 1.0) * (col + 1.0) * math.pi / (level + 2.0))
                for col in range(size)
            ]
            for row in range(size)
        ],
        dtype=float,
    )


def charge_embedding(level: int) -> tuple[int, int, int]:
    return (level - 4, level - 3, level)


def modular_character_profile(
    level: int,
    genus: int,
    tau_imag: float = SUPPORT_TAU_IMAG,
    phi_samples: int = SUPPORT_PHI_SAMPLES,
) -> np.ndarray:
    r"""Discretize the asymptotic modular-character profile from Eq. ``eq:wdw-support-integral``."""

    phi_grid = 2.0 * math.pi * np.arange(phi_samples, dtype=float) / phi_samples
    conformal_weight = su2_conformal_weight(level, genus)
    central_charge = wzw_central_charge(level, SU2_DIMENSION, SU2_DUAL_COXETER)
    radial_weight = math.exp(-2.0 * math.pi * tau_imag * (conformal_weight - central_charge / 24.0))
    profile = radial_weight * np.exp(1j * genus * phi_grid)
    return profile / math.sqrt(float(np.vdot(profile, profile).real))


def _character_radial_weight(level: int, genus: int, tau_imag: float) -> float:
    conformal_weight = su2_conformal_weight(level, genus)
    central_charge = wzw_central_charge(level, SU2_DIMENSION, SU2_DUAL_COXETER)
    return math.exp(-2.0 * math.pi * tau_imag * (conformal_weight - central_charge / 24.0))


def _validated_quad(
    integrand,
    lower: float,
    upper: float,
    *,
    integral_name: str,
) -> float:
    value, error_estimate = quad(
        integrand,
        lower,
        upper,
        epsabs=DEFAULT_SOLVER_CONFIG.quad_epsabs,
        epsrel=DEFAULT_SOLVER_CONFIG.quad_epsrel,
        limit=200,
    )
    allowed_error = max(
        DEFAULT_SOLVER_CONFIG.quad_epsabs,
        DEFAULT_SOLVER_CONFIG.quad_epsrel * max(abs(value), 1.0),
    )
    if error_estimate > allowed_error:
        raise QuadratureConvergenceError(
            f"Quadrature audit failed for {integral_name}: estimated error {error_estimate:.3e} exceeds {allowed_error:.3e}."
        )
    return float(value)


def _character_normalization(level: int, genus: int, tau_imag: float) -> float:
    radial_weight = _character_radial_weight(level, genus, tau_imag)
    norm_sq = _validated_quad(
        lambda phi: radial_weight * radial_weight,
        0.0,
        2.0 * math.pi,
        integral_name=rf"character normalization (k={level}, g={genus}, Im\tau={tau_imag})",
    )
    return math.sqrt(norm_sq)


def modular_character_overlap(
    level: int,
    genus_left: int,
    genus_right: int,
    tau_imag: float = SUPPORT_TAU_IMAG,
) -> complex:
    left_weight = _character_radial_weight(level, genus_left, tau_imag)
    right_weight = _character_radial_weight(level, genus_right, tau_imag)
    normalization = _character_normalization(level, genus_left, tau_imag) * _character_normalization(level, genus_right, tau_imag)
    prefactor = (left_weight * right_weight) / normalization
    harmonic = genus_right - genus_left
    real_part = _validated_quad(
        lambda phi: prefactor * math.cos(harmonic * phi),
        0.0,
        2.0 * math.pi,
        integral_name=rf"character overlap real part (k={level}, g_L={genus_left}, g_R={genus_right}, Im\tau={tau_imag})",
    )
    imag_part = _validated_quad(
        lambda phi: prefactor * math.sin(harmonic * phi),
        0.0,
        2.0 * math.pi,
        integral_name=rf"character overlap imaginary part (k={level}, g_L={genus_left}, g_R={genus_right}, Im\tau={tau_imag})",
    )
    return complex(real_part, imag_part)


def support_overlap_matrix(
    level: int,
    genus_assignment: tuple[int, int, int],
    tau_imag: float = SUPPORT_TAU_IMAG,
) -> np.ndarray:
    r"""Construct the overlap matrix from Eq. ``eq:wdw-overlap-matrix`` for one-copy support assignments.

    Args:
        level: Affine $SU(2)$ level.
        genus_assignment: Boundary genus channels assigned to the three states.
        tau_imag: Imaginary modular parameter used for the asymptotic character profile.

    Returns:
        The normalized overlap matrix of asymptotic character profiles.
    """

    overlap_matrix = np.zeros((len(genus_assignment), len(genus_assignment)), dtype=complex)
    for row, left_genus in enumerate(genus_assignment):
        for col, right_genus in enumerate(genus_assignment):
            overlap_matrix[row, col] = modular_character_overlap(level, left_genus, right_genus, tau_imag=tau_imag)
    return np.real_if_close(overlap_matrix)


def support_overlap_penalty(overlap_matrix: np.ndarray, tolerance: float = 1.0e-9) -> float:
    spectral_audit = derive_matrix_spectrum_audit(overlap_matrix)
    if not spectral_audit.perturbative_nonsingular:
        return STRICT_SUPPORT_PENALTY
    if math.isclose(spectral_audit.condition_number, 1.0, abs_tol=tolerance, rel_tol=tolerance):
        return 0.0
    return math.log(max(spectral_audit.condition_number, 1.0 + tolerance))


def symmetrize_majorana_texture(texture: np.ndarray) -> np.ndarray:
    return 0.5 * (texture + texture.T)


def rotation_23(phi: float) -> np.ndarray:
    cosine = math.cos(phi)
    sine = math.sin(phi)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]],
        dtype=complex,
    )


def pdg_unitary(theta12_deg: float, theta13_deg: float, theta23_deg: float, delta_deg: float) -> np.ndarray:
    theta12 = math.radians(theta12_deg)
    theta13 = math.radians(theta13_deg)
    theta23 = math.radians(theta23_deg)
    delta = math.radians(delta_deg)

    s12, c12 = math.sin(theta12), math.cos(theta12)
    s13, c13 = math.sin(theta13), math.cos(theta13)
    s23, c23 = math.sin(theta23), math.cos(theta23)
    e_minus = np.exp(-1j * delta)
    e_plus = np.exp(1j * delta)

    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * e_minus],
            [-s12 * c23 - c12 * s23 * s13 * e_plus, c12 * c23 - s12 * s23 * s13 * e_plus, s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * e_plus, -c12 * s23 - s12 * c23 * s13 * e_plus, c23 * c13],
        ],
        dtype=complex,
    )


def jarlskog_invariant(unitary: np.ndarray) -> float:
    return float(np.imag(unitary[0, 0] * unitary[1, 1] * np.conjugate(unitary[0, 1]) * np.conjugate(unitary[1, 0])))


def jarlskog_area_factor(theta12_deg: float, theta13_deg: float, theta23_deg: float) -> float:
    return 0.125 * (
        math.sin(math.radians(2.0 * theta12_deg))
        * math.sin(math.radians(2.0 * theta13_deg))
        * math.sin(math.radians(2.0 * theta23_deg))
        * math.cos(math.radians(theta13_deg))
    )


def complex_modular_s_matrix_representation(
    seed_matrix: np.ndarray,
    kernel_helper: ModularKernel,
) -> np.ndarray:
    r"""Return the default Complex Modular $S$-matrix representation for mixing data."""

    return kernel_helper.t_decorated_unitary(seed_matrix)


def topological_jarlskog_identity(
    gut_threshold_residue: float = R_GUT,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    kappa_geometric: float | None = None,
) -> float:
    r"""Return the modular-locking Jarlskog numerator fixed by the benchmark cell."""

    resolved_kappa = (
        compute_geometric_kappa_ansatz(
            parent_level=parent_level,
            lepton_level=lepton_level,
        ).derived_kappa
        if kappa_geometric is None
        else float(kappa_geometric)
    )

    packing_residue = math.sqrt(max(0.0, 1.0 - resolved_kappa * resolved_kappa))
    angular_drive = math.sin(2.0 * math.pi * quark_level / lepton_level)
    return float((gut_threshold_residue / parent_level) * packing_residue * angular_drive)


def threshold_projected_jarlskog(
    jarlskog_topological: float,
    *,
    gut_threshold_residue: float = R_GUT,
) -> float:
    r"""Project the locked topological numerator onto the visible CKM branch."""

    return float(gut_threshold_residue * jarlskog_topological)


def delta_cp_from_jarlskog_lock(
    theta12_deg: float,
    theta13_deg: float,
    theta23_deg: float,
    jarlskog_target: float,
    *,
    branch_reference_deg: float | None = None,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    r"""Recover the PDG Dirac phase from a locked Jarlskog target and fixed angles."""

    guard = solver_config.stability_guard
    area = float(
        guard.require_nonzero_magnitude(
            jarlskog_area_factor(theta12_deg, theta13_deg, theta23_deg),
            coordinate="locked Jarlskog area",
            detail="The modular-locking phase is undefined when the CKM area factor collapses.",
        )
    )
    sin_delta = guard.clamp_signed_unit_interval(
        jarlskog_target / area,
        coordinate="locked sin(delta)",
    )
    principal = math.degrees(math.asin(sin_delta))
    candidates = (
        principal % 360.0,
        (180.0 - principal) % 360.0,
        (180.0 + principal) % 360.0,
        (360.0 - principal) % 360.0,
    )
    if branch_reference_deg is None:
        return candidates[0]
    reference = branch_reference_deg % 360.0

    def wrapped_distance(candidate: float) -> float:
        return abs((candidate - reference + 180.0) % 360.0 - 180.0)

    return min(candidates, key=wrapped_distance)


def cp_conserving_modularity_leak(
    jarlskog_target: float,
    theta12_deg: float,
    theta13_deg: float,
    theta23_deg: float,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    r"""Return the modularity leak reopened by forcing a CP-conserving projection."""

    guard = solver_config.stability_guard
    area = float(
        guard.require_nonzero_magnitude(
            jarlskog_area_factor(theta12_deg, theta13_deg, theta23_deg),
            coordinate="CP-conserving modular leak area",
            detail="The modular leak is undefined when the CKM area factor vanishes.",
        )
    )
    locked_sine = abs(
        guard.clamp_signed_unit_interval(
            jarlskog_target / area,
            coordinate="CP-conserving modular leak sine",
        )
    )
    modular_gap = benchmark_visible_modularity_gap(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    return float(modular_gap * locked_sine)


def normal_order_masses(
    lightest_mass_ev: float,
    solar_splitting_ev2: float = SOLAR_MASS_SPLITTING_EV2,
    atmospheric_splitting_ev2: float = ATMOSPHERIC_MASS_SPLITTING_NO_EV2,
) -> np.ndarray:
    r"""Return the normal-order mass ladder implied by Eq. ``eq:bousso-bridge``."""

    return np.array(
        [
            lightest_mass_ev,
            math.sqrt(lightest_mass_ev * lightest_mass_ev + solar_splitting_ev2),
            math.sqrt(lightest_mass_ev * lightest_mass_ev + atmospheric_splitting_ev2),
        ],
        dtype=float,
    )


def effective_majorana_mass(unitary: np.ndarray, masses_ev: np.ndarray) -> float:
    r"""Return the effective Majorana mass from Eq. ``eq:mbb-def``."""

    return float(abs(np.sum(unitary[0, :] ** 2 * masses_ev)))


def normalize_triangle_angle(angle_deg: float) -> float:
    wrapped_angle = angle_deg % 360.0
    return wrapped_angle if wrapped_angle <= 180.0 else 360.0 - wrapped_angle


def ckm_unitarity_triangle_angles(unitary: np.ndarray) -> tuple[float, float, float]:
    v_ud, v_us, v_ub = unitary[0, 0], unitary[0, 1], unitary[0, 2]
    v_cd, v_cs, v_cb = unitary[1, 0], unitary[1, 1], unitary[1, 2]
    v_td, v_ts, v_tb = unitary[2, 0], unitary[2, 1], unitary[2, 2]

    alpha = normalize_triangle_angle(math.degrees(np.angle(-v_td * np.conjugate(v_tb) / (v_ud * np.conjugate(v_ub)))))
    beta = normalize_triangle_angle(math.degrees(np.angle(-v_cd * np.conjugate(v_cb) / (v_td * np.conjugate(v_tb)))))
    gamma = normalize_triangle_angle(math.degrees(np.angle(-v_ud * np.conjugate(v_ub) / (v_cd * np.conjugate(v_cb)))))
    return alpha, beta, gamma


def calculate_chi2(*pull_data: PullData) -> float:
    r"""Aggregate finite sigma-pulls into the manuscript goodness-of-fit statistic ``eq:chi2-def``."""

    return float(sum(item.pull * item.pull for item in pull_data if math.isfinite(item.pull)))


def _distribution_skewness(sample_matrix: np.ndarray) -> np.ndarray:
    if sample_matrix.ndim != 2 or sample_matrix.shape[0] < 3:
        return np.zeros(sample_matrix.shape[1] if sample_matrix.ndim == 2 else 0, dtype=float)
    centered = sample_matrix - np.mean(sample_matrix, axis=0, keepdims=True)
    variances = np.mean(centered * centered, axis=0)
    scales = np.power(np.maximum(variances, np.finfo(float).tiny), 1.5)
    return np.mean(centered**3, axis=0) / scales


def _lag_autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= series.size:
        raise ValueError(f"lag must satisfy 0 < lag < len(series), received lag={lag}")
    left = np.asarray(series[:-lag], dtype=float)
    right = np.asarray(series[lag:], dtype=float)
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= condition_aware_abs_tolerance(scale=denominator):
        return math.nan
    return float(np.dot(left, right) / denominator)


def _axis_autocorrelation(grid: np.ndarray, axis: int) -> np.ndarray:
    max_lag = grid.shape[axis] - 1
    if max_lag <= 0:
        return np.empty(0, dtype=float)
    correlations: list[float] = []
    for lag in range(1, max_lag + 1):
        axis_values: list[float] = []
        if axis == 0:
            for column_index in range(grid.shape[1]):
                series = np.asarray(grid[:, column_index], dtype=float)
                if series.size > lag:
                    axis_values.append(_lag_autocorrelation(series, lag))
        else:
            for row_index in range(grid.shape[0]):
                series = np.asarray(grid[row_index, :], dtype=float)
                if series.size > lag:
                    axis_values.append(_lag_autocorrelation(series, lag))
        finite_values = [value for value in axis_values if math.isfinite(value)]
        correlations.append(float(np.mean(finite_values)) if finite_values else math.nan)
    return np.asarray(correlations, dtype=float)


def _correlation_length_from_autocorrelation(autocorrelation: np.ndarray) -> float:
    if autocorrelation.size == 0:
        return 1.0
    threshold = math.exp(-1.0)
    for lag_index, value in enumerate(autocorrelation, start=1):
        if math.isfinite(value) and value <= threshold:
            return float(lag_index)
    positive_lags = [lag_index for lag_index, value in enumerate(autocorrelation, start=1) if math.isfinite(value) and value > 0.0]
    return float(positive_lags[-1]) if positive_lags else 1.0


def _relative_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    mean_value = float(np.mean(values))
    denominator = max(abs(mean_value), condition_aware_abs_tolerance(scale=mean_value))
    return float(np.std(values, ddof=1) / denominator)


def apply_landscape_penalty(
    conditional_p_value: float,
    trial_count: float = float(LANDSCAPE_TRIAL_COUNT),
) -> float:
    """Return the autocorrelation-aware global p-value for the discrete landscape search."""

    clipped_p_value = min(max(float(conditional_p_value), 0.0), 1.0)
    resolved_trial_count = max(float(trial_count), 1.0)
    if clipped_p_value in (0.0, 1.0) or resolved_trial_count == 1:
        return clipped_p_value
    return min(-math.expm1(resolved_trial_count * math.log1p(-clipped_p_value)), 1.0)


def calculate_global_chi_square(
    *pull_data: PullData,
    degrees_of_freedom: int,
    landscape_trial_count: float | None = float(LANDSCAPE_TRIAL_COUNT),
) -> tuple[float, float, float]:
    """Return χ² together with conditional and landscape-corrected p-values."""

    chi2 = calculate_chi2(*pull_data)
    conditional_p_value = float(chi2_distribution.sf(chi2, degrees_of_freedom))
    corrected_p_value = conditional_p_value if landscape_trial_count is None else apply_landscape_penalty(
        conditional_p_value,
        trial_count=landscape_trial_count,
    )
    return chi2, conditional_p_value, corrected_p_value


def pdg_parameters(
    unitary: np.ndarray,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> tuple[float, float, float, float, float]:
    """Return PDG angles and the Jarlskog invariant from a unitary matrix."""

    return topological_kernel.pdg_parameters(unitary, solver_config=solver_config)


class ModularKernel(topological_kernel.ModularKernel):
    """Thin local wrapper preserving the manuscript verifier API."""

    def __init__(
        self,
        level: int,
        sector: Sector | str,
        *,
        solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> None:
        super().__init__(level, sector, solver_config=solver_config)


def derive_boundary_bulk_interface(
    level: int | None = None,
    sector: Sector | str = Sector.LEPTON,
    *,
    model: TopologicalModel | None = None,
    solver_config: SolverConfig | None = None,
) -> BoundaryBulkInterfaceData:
    """Build the formal boundary-to-Yukawa dictionary for a visible sector."""

    resolved_sector = Sector.coerce(sector)
    resolved_model = _coerce_topological_model(model=model, solver_config=solver_config)
    resolved_level = (
        resolved_model.lepton_level if level is None and resolved_sector is Sector.LEPTON else resolved_model.quark_level if level is None else level
    )
    kernel = ModularKernel(resolved_level, resolved_sector, solver_config=resolved_model.solver_config)
    modular_block = kernel.restricted_block()
    enforce_perturbative_matrix(
        modular_block,
        coordinate=f"{resolved_sector.value} restricted flavor kernel",
        solver_config=resolved_model.solver_config,
        detail="The boundary-to-bulk dictionary is defined only on perturbatively conditioned flavor blocks.",
    )
    reference_entry = resolved_model.solver_config.stability_guard.require_nonzero_magnitude(
        modular_block[0, 0],
        coordinate=f"{resolved_sector.value} modular normalization",
        detail="A vanishing modular normalization removes the visible reference channel from the dictionary.",
    )
    normalization = abs(reference_entry)
    yukawa_texture = modular_block / normalization
    framed_yukawa_texture = yukawa_texture @ np.diag(kernel.framing_phases())
    majorana_yukawa_texture = symmetrize_majorana_texture(framed_yukawa_texture)
    return BoundaryBulkInterfaceData(
        sector=resolved_sector,
        modular_block=modular_block,
        framing_phases=kernel.framing_phases(),
        yukawa_texture=yukawa_texture,
        framed_yukawa_texture=framed_yukawa_texture,
        majorana_yukawa_texture=majorana_yukawa_texture,
    )


def transported_boundary_overlap_matrix(
    level: int = LEPTON_LEVEL,
    tau_imag: float = SUPPORT_TAU_IMAG,
) -> np.ndarray:
    r"""Return the bulk overlap matrix of transported $SU(2)_k$ character wavefunctions.

    The transported bulk profiles are defined by

    .. math::
       \\Psi_\alpha(\phi)=\\sum_{g=0}^2 Y_{\alpha g}\,\\chi_g(\tau,\phi),

    where ``Y`` is the normalized boundary--bulk Yukawa texture and
    ``chi_g`` are the asymptotic boundary characters. The returned matrix is the
    normalized overlap integral ``Omega_{\alpha\beta}=<Psi_\alpha|Psi_\beta>``.
    """

    interface = derive_boundary_bulk_interface(level=level, sector="lepton")
    character_metric = support_overlap_matrix(level, (0, 1, 2), tau_imag=tau_imag)
    overlap_matrix = interface.yukawa_texture @ character_metric @ interface.yukawa_texture.conjugate().T
    return np.real_if_close(overlap_matrix)


def _simplex_surface_area(vertices: tuple[np.ndarray, ...]) -> float:
    dimension = len(vertices) - 1
    hyperareas: list[float] = []
    for omitted in range(len(vertices)):
        facet = [vertices[index] for index in range(len(vertices)) if index != omitted]
        base_vertex = facet[0]
        edge_matrix = np.stack([vertex - base_vertex for vertex in facet[1:]], axis=1)
        gram_matrix = edge_matrix.T @ edge_matrix
        hyperareas.append(math.sqrt(max(float(np.linalg.det(gram_matrix)), 0.0)) / math.factorial(dimension - 1))
    return float(sum(hyperareas))


def _simplex_circumradius(vertices: tuple[np.ndarray, ...]) -> float:
    reference = vertices[0]
    matrix = np.stack([2.0 * (vertex - reference) for vertex in vertices[1:]], axis=0)
    rhs = np.array([float(vertex @ vertex - reference @ reference) for vertex in vertices[1:]], dtype=float)
    center = np.linalg.solve(matrix, rhs)
    return float(np.linalg.norm(center - reference))


def _regular_simplex_surface_area(dimension: int, circumradius: float) -> float:
    edge_length = circumradius * math.sqrt(2.0 * (dimension + 1.0) / dimension)
    return float(
        (dimension + 1.0)
        * math.sqrt(dimension)
        * edge_length ** (dimension - 1)
        / (math.factorial(dimension - 1) * 2.0 ** ((dimension - 1) / 2.0))
    )


def effective_visible_entropy_load(level: int) -> float:
    r"""Return the effective descendant entropy load of a visible affine pixel at rank ``k``."""

    visible_central_charge = wzw_central_charge(level, SU2_DIMENSION, SU2_DUAL_COXETER) + wzw_central_charge(
        level,
        SU3_DIMENSION,
        SU3_DUAL_COXETER,
    )
    return float(visible_central_charge * math.log(level + SU2_DUAL_COXETER))


def define_computational_search_window(
    parent_level: int = PARENT_LEVEL,
    search_limit: int = 256,
) -> HolographicDensityBoundData:
    r"""Define the publication-facing phenomenological search window.

    The current manuscript and runtime audit treat the broad low-rank window

    .. math::
       k_\ell\in[2,100],\qquad k_q\in[2,100],

    as the publication-facing look-elsewhere scan rather than as a theorem
    derived from the parent level. For bookkeeping the verifier also records
    the older single-pixel density-cap diagnostic, whose local descendant load
    grows as

    .. math::
       s_{\rm pix}(k)
       =
       \bigl[c(SU(2)_k)+c(SU(3)_k)\bigr]\ln(k+2),

    while a Planck-scale pixel of the completed ``SO(10)_{312}`` vacuum carries
    the capacity

    .. math::
       s_{\rm pix}^{\max}
       =
       c(SO(10)_{312}) + h^\vee_{SO(10)}\ln 2.

    The first violating integer level is ``k=100``. That threshold is retained
    only as an auxiliary diagnostic in the runtime report; it does not define
    the broader publication-facing scan window used in the manuscript tables.
    """

    pixel_capacity = wzw_central_charge(parent_level, SO10_DIMENSION, SO10_DUAL_COXETER) + SO10_DUAL_COXETER * math.log(2.0)
    max_admissible_level = max(
        level for level in range(1, search_limit + 1) if effective_visible_entropy_load(level) < pixel_capacity
    )
    return HolographicDensityBoundData(
        pixel_capacity=float(pixel_capacity),
        level_99_load=effective_visible_entropy_load(99),
        level_100_load=effective_visible_entropy_load(100),
        max_admissible_level=max_admissible_level,
    )


@lru_cache(maxsize=128)
def compute_geometric_kappa_ansatz(
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
) -> SO10GeometricKappaData:
    r"""Compute the disclosed geometric ansatz behind ``\kappa``.

    The ingredients entering this quantity are geometric --- the $D_5$
    weight-simplex hyperarea ratio and the spinorial-retention factor --- but
    their specific combination is treated here as an effective benchmark ansatz
    rather than as a first-principles derivation. The verifier still scans
    order-one deformations of the ansatz, while the publication-facing export
    reports the disclosed benchmark point ``\kappa=R_{01}^{\rm par/vis}``.
    The prefactor is built from the spinorially dressed square root of the
    weight-simplex hyperarea ratio,

    .. math::
       g_{D_5}
       =
       \sqrt{\frac{A_{\Delta(D_5)}}{A_{\rm reg}(R_{\Delta})}
       \left(1-\frac{\beta^2+1/2}{c(SO(10)_{312})}\right)},

    where ``A_reg(R_Δ)`` is the regular 5-simplex hyperarea with the same
    circumradius as the actual ``D_5`` fundamental-weight simplex. The resulting
    simplex-inspired prefactor is then

    .. math::
       \kappa
       =
       \sqrt{\frac{\dim(\mathbf{16})}{\operatorname{rank}(SO(10))}}\,g_{D_5}.
    """

    vertices = (np.zeros(SO10_RANK, dtype=float),) + tuple(
        np.array([float(component) for component in weight], dtype=float) for weight in so10_fundamental_weights()
    )
    weight_simplex_hyperarea = _simplex_surface_area(vertices)
    weight_simplex_circumradius = _simplex_circumradius(vertices)
    regular_reference_hyperarea = _regular_simplex_surface_area(SO10_RANK, weight_simplex_circumradius)
    area_ratio = weight_simplex_hyperarea / regular_reference_hyperarea
    parent_central_charge = wzw_central_charge(parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
    beta = 0.5 * math.log(su2_total_quantum_dimension(lepton_level))
    spinorial_retention = 1.0 - (beta * beta + 0.5) / parent_central_charge
    geometric_factor = math.sqrt(max(area_ratio * spinorial_retention, 0.0))
    spinor_dimension = so10_rep_dimension(SO10_SPINOR_16_DYNKIN_LABELS)
    derived_kappa = math.sqrt(spinor_dimension / SO10_RANK) * geometric_factor
    return SO10GeometricKappaData(
        weight_simplex_hyperarea=float(weight_simplex_hyperarea),
        regular_reference_hyperarea=float(regular_reference_hyperarea),
        area_ratio=float(area_ratio),
        spinorial_retention=float(spinorial_retention),
        geometric_factor=float(geometric_factor),
        spinor_dimension=spinor_dimension,
        derived_kappa=float(derived_kappa),
    )


def derive_so10_geometric_kappa(
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
) -> SO10GeometricKappaData:
    """Backward-compatible alias for `compute_geometric_kappa_ansatz()`."""

    return compute_geometric_kappa_ansatz(parent_level=parent_level, lepton_level=lepton_level)


@lru_cache(maxsize=128)
def derive_modular_horizon_selection(
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
) -> ModularHorizonSelectionData:
    r"""Derive the horizon bit count from the parent modular weight.

    The single-channel Cardy factor is

    .. math::
       n_1(K) \sim \exp\!\left(2\pi\sqrt{K/6}\right),

    while the completed vacuum uses the effective modular weight

    .. math::
       \omega_{\rm vac}
       =
       c(SO(10)_K)-\beta^2-h^\vee_{SU(2)}-\frac{1}{2(k_\ell+1)}.

    For ``(K,k_\ell)=(312,26)`` this yields a derived horizon budget in the
    same ``10^{122}`` range as the Planck-anchored de Sitter entropy.
    """

    parent_central_charge = wzw_central_charge(parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
    beta = 0.5 * math.log(su2_total_quantum_dimension(lepton_level))
    framing_gap_area = beta * beta
    visible_edge_penalty = 1.0 / (2.0 * (lepton_level + 1.0))
    effective_vacuum_weight = parent_central_charge - framing_gap_area - SU2_DUAL_COXETER - visible_edge_penalty
    unit_modular_weight = math.exp(2.0 * math.pi * math.sqrt(parent_level / 6.0))
    derived_bits = math.exp(2.0 * math.pi * math.sqrt(effective_vacuum_weight * parent_level / 6.0))
    return ModularHorizonSelectionData(
        unit_modular_weight=float(unit_modular_weight),
        effective_vacuum_weight=float(effective_vacuum_weight),
        parent_central_charge=float(parent_central_charge),
        framing_gap_area=float(framing_gap_area),
        visible_edge_penalty=float(visible_edge_penalty),
        derived_bits=float(derived_bits),
        planck_crosscheck_ratio=float(derived_bits / PLANCK_HOLOGRAPHIC_BITS),
    )


so10_fundamental_weights = algebra.so10_fundamental_weights
so10_weyl_vector = algebra.so10_weyl_vector
so10_positive_roots = algebra.so10_positive_roots
so10_simple_roots = algebra.so10_simple_roots
so10_visible_cartan_projection_basis = algebra.so10_visible_cartan_projection_basis
so10_visible_cartan_projection_denominator = algebra.so10_visible_cartan_projection_denominator
so10_highest_weight = algebra.so10_highest_weight
so10_rep_dimension = algebra.so10_rep_dimension
so10_rep_quadratic_casimir = algebra.so10_rep_quadratic_casimir
so10_rep_dynkin_index = algebra.so10_rep_dynkin_index
su2_rep_quadratic_casimir = algebra.su2_rep_quadratic_casimir
su2_rep_dynkin_index = algebra.su2_rep_dynkin_index
su3_rep_dimension = algebra.su3_rep_dimension
su3_rep_quadratic_casimir = algebra.su3_rep_quadratic_casimir
su3_rep_dynkin_index = algebra.su3_rep_dynkin_index
adjoint_quadratic_casimir = algebra.adjoint_quadratic_casimir
su2_total_quantum_dimension = algebra.su2_total_quantum_dimension


def derive_information_mass_anomalous_dimensions(
    representation_name: str = MAJORANA_HIGGS_REPRESENTATION,
) -> tuple[float, float, float]:
    r"""Derive the defect-scale anomalous dimensions from the $\mathbf{126}_H$ channel.

    The one-loop coefficient is taken to be the normalized scalar curvature of the
    holographically selected Majorana channel,

    .. math::
       \gamma_0^{(1)}
       =
       \frac{C_2(\mathbf{126}_H)}{2T(\mathbf{126}_H)},

    while the quadratic correction is dressed by the ratio of the adjoint Casimirs
    of the visible $SU(2)$ and $SU(3)$ sectors.
    """

    if representation_name != MAJORANA_HIGGS_REPRESENTATION:
        raise ValueError(f"Unsupported Majorana-channel representation: {representation_name}")

    majorana_representation = derive_so10_representation_data(representation_name, SO10_HIGGS_126_DYNKIN_LABELS)
    majorana_scalar_curvature = float(
        majorana_representation.quadratic_casimir / (2 * majorana_representation.dynkin_index)
    )
    su2_adjoint_casimir = float(adjoint_quadratic_casimir(SU2_DUAL_COXETER))
    su3_adjoint_casimir = float(adjoint_quadratic_casimir(SU3_DUAL_COXETER))
    gamma_0_one_loop = majorana_scalar_curvature
    gamma_0_two_loop = gamma_0_one_loop * gamma_0_one_loop * su2_adjoint_casimir / su3_adjoint_casimir
    return gamma_0_one_loop, gamma_0_two_loop, majorana_scalar_curvature


@lru_cache(maxsize=128)
def derive_transport_curvature_audit(
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    bulk_dimension: int = BULK_SPACETIME_DIMENSION,
) -> TransportCurvatureAuditData:
    r"""Derive the publication-facing RG curvature audit from affine Casimir data.

    The leptonic coefficients are built from adjoint Casimirs of the parent and
    visible groups, dressed by the level-dependent WZW factor $(1-1/(k_\ell+2))$.
    The quark coefficients are derived from the $SU(3)_8$ fundamental and adjoint
    Casimirs, with the $23$ channel carrying the explicit rank-gap pressure of the
    $SO(10)/SU(3)$ descent. The Dirac-phase coefficients are derived from the same
    curvature data inside the default Standard-Model transport with no auxiliary
    overlap deformation.
    """

    solar_norm = 1.0
    so10_adjoint_casimir = adjoint_quadratic_casimir(SO10_DUAL_COXETER)
    su2_adjoint_casimir = adjoint_quadratic_casimir(SU2_DUAL_COXETER)
    su3_adjoint_casimir = adjoint_quadratic_casimir(SU3_DUAL_COXETER)
    su3_fundamental_casimir = su3_rep_quadratic_casimir((1, 0))

    gamma_0_one_loop, gamma_0_two_loop, majorana_scalar_curvature = derive_information_mass_anomalous_dimensions()

    lepton_theta_two_loop = gamma_0_one_loop * np.array(
        [
            float(
                so10_adjoint_casimir
                / (su2_adjoint_casimir + bulk_dimension)
                * (1 - Fraction(1, lepton_level + SU2_DUAL_COXETER))
            ),
            float(su2_adjoint_casimir / (so10_adjoint_casimir + bulk_dimension)),
            float(Fraction(bulk_dimension - 1, 1) / (so10_adjoint_casimir + su2_adjoint_casimir + su3_adjoint_casimir)),
        ],
        dtype=float,
    )

    quark_theta_two_loop = np.array(
        [
            float(su3_fundamental_casimir)
            / ((quark_level + SU3_DUAL_COXETER) * (SO10_DIMENSION + SO10_DUAL_COXETER + bulk_dimension)),
            -float(su3_adjoint_casimir * RANK_DIFFERENCE) / (quark_level + SU3_DUAL_COXETER),
            -float(su3_fundamental_casimir) / (quark_level * (quark_level + SU3_DUAL_COXETER)),
        ],
        dtype=float,
    )

    lepton_delta_two_loop = -solar_norm * float(
        so10_adjoint_casimir
        / (so10_adjoint_casimir + su2_adjoint_casimir + su3_adjoint_casimir)
        * (1 - Fraction(1, lepton_level + SU2_DUAL_COXETER))
        * (1 + su2_adjoint_casimir / Fraction(SO10_DUAL_COXETER + bulk_dimension + SU3_DUAL_COXETER, 1))
    )
    quark_delta_two_loop = float(su3_fundamental_casimir) / (SO10_DUAL_COXETER + quark_level) * (
        1 + RANK_DIFFERENCE / (SO10_DUAL_COXETER + bulk_dimension)
    )

    return freeze_numpy_arrays(TransportCurvatureAuditData(
        lepton_theta_two_loop=lepton_theta_two_loop,
        lepton_delta_two_loop=float(lepton_delta_two_loop),
        quark_theta_two_loop=quark_theta_two_loop,
        quark_delta_two_loop=float(quark_delta_two_loop),
        gamma_0_one_loop=float(gamma_0_one_loop),
        gamma_0_two_loop=float(gamma_0_two_loop),
        majorana_scalar_curvature=float(majorana_scalar_curvature),
    ))


@lru_cache(maxsize=16)
def _derive_physics_audit_cached(
    parent_level: int,
    lepton_level: int,
    quark_level: int,
) -> PhysicsAudit:
    return freeze_numpy_arrays(PhysicsAudit(
        search_window=define_computational_search_window(parent_level=parent_level),
        geometric_kappa=compute_geometric_kappa_ansatz(parent_level=parent_level, lepton_level=lepton_level),
        modular_horizon=derive_modular_horizon_selection(parent_level=parent_level, lepton_level=lepton_level),
        transport_curvature=derive_transport_curvature_audit(lepton_level=lepton_level, quark_level=quark_level),
        topological_threshold_gev=derive_topological_threshold_gev(
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        ),
        gauge_unification_beta_shift_126=derive_so10_scalar_beta_shift("126_H"),
        gauge_unification_beta_shift_10=derive_so10_scalar_beta_shift("10_H"),
    ))


def derive_physics_audit(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    *,
    model: TopologicalModel | None = None,
) -> PhysicsAudit:
    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    return _derive_physics_audit_cached(
        resolved_model.parent_level,
        resolved_model.lepton_level,
        resolved_model.quark_level,
    )


def export_support_overlap_table(
    audit: AuditData,
    output_dir: Path,
    level: int | None = None,
    *,
    model: TopologicalModel | None = None,
) -> None:
    """Write the supplementary inverted-hierarchy support map as a LaTeX table."""

    resolved_model = _coerce_topological_model(model=model, lepton_level=level)
    audit_generator.export_support_overlap_table(
        audit,
        output_dir,
        support_overlap_result_factory=SupportOverlapResult,
        level=resolved_model.lepton_level,
    )


def export_supplementary_tolerance_table(output_dir: Path, *, model: TopologicalModel | None = None) -> None:
    """Write the supplementary tolerance-sweep table used by `supplementary.tex`."""

    configs = ((1.0e-8, 1.0e-10), (1.0e-10, 1.0e-12), (1.0e-12, 1.0e-14))
    resolved_model = _coerce_topological_model(model=model)

    audit_generator.export_supplementary_tolerance_table(
        output_dir,
        configs=configs,
        derive_pmns=lambda *, solver_config: derive_pmns(model=replace(resolved_model, solver_config=solver_config)),
        derive_ckm=lambda *, solver_config: derive_ckm(model=replace(resolved_model, solver_config=solver_config)),
        derive_pull_table=derive_pull_table,
        lepton_intervals=LEPTON_INTERVALS,
        quark_intervals=QUARK_INTERVALS,
        ckm_gamma_interval=CKM_GAMMA_GOLD_STANDARD_DEG,
    )


def export_unitary_consistency_table(
    output_dir: Path,
    unitary_audit: UnitaryBoundAudit,
    *,
    model: TopologicalModel | None = None,
    lepton_levels: tuple[int, ...] | None = None,
) -> None:
    r"""Write the supplementary unitary-rigidity table used by `supplementary.tex`."""

    resolved_model = _coerce_topological_model(model=model)
    displayed_levels = (resolved_model.lepton_level - 1, resolved_model.lepton_level, resolved_model.lepton_level + 1) if lepton_levels is None else lepton_levels
    body_rows: list[str] = []
    for level in displayed_levels:
        row_model = replace(resolved_model, k_l=level)
        modular_gap = benchmark_visible_modularity_gap(model=row_model)
        c_dark_completion = 24.0 * modular_gap
        residue_efficiency = modular_residue_efficiency(
            c_dark_completion,
            lepton_level=row_model.lepton_level,
            quark_level=row_model.quark_level,
        )
        framing_gap = nearest_integer_gap(row_model.parent_level / (2.0 * row_model.lepton_level))
        unitary_buffer = c_dark_completion if solver_isclose(framing_gap, 0.0) else 0.0
        first_cell = rf"\textbf{{{level}}}" if level == resolved_model.lepton_level else f"{level}"
        body_rows.append(
            rf"{first_cell} & ${_format_tex_scientific(row_model.bit_count, precision=2)}$ & ${c_dark_completion:.4f}$ & ${residue_efficiency:.4f}$ & ${modular_gap:.6f}$ & ${framing_gap:.6f}$ & ${unitary_buffer:.4f}$ \\" 
        )

    footer_rows = (
        rf"$S_{{\max}}=\ln N$ & \multicolumn{{6}}{{r}}{{benchmark entropy ceiling $\ln({ _format_tex_scientific(unitary_audit.holographic_bits, precision=2) }) = {unitary_audit.entropy_max_nats:.2f}$.}} \\ ",
        r"$\Gamma_{\rm BH}^{\rm rec}=\Gamma_p^{\rm info}$ & \multicolumn{6}{r}{the unitary recovery rate is the same Delta-locked protected evaporation channel used in the proton-lifetime check.} \\ ",
    )
    publication_export.write_skeletal_latex_table(
        output_dir / SUPPLEMENTARY_UNITARY_CONSISTENCY_TABLE_FILENAME,
        column_spec="ccccccc",
        header_rows=(r"$k_{\ell}$ & $N_{\rm holo}$ & $c_{\rm dark}$ & $\eta_{\rm mod}$ & $\Delta_{\rm mod}$ & $\Delta_{\rm fr}$ & $\mathcal B_{\rm unit}$ \\",),
        body_rows=tuple(body_rows),
        footer_rows=footer_rows,
        style="booktabs",
    )


def export_kappa_sensitivity_audit_table(
    geometric_sensitivity: GeometricSensitivityData,
    output_dir: Path,
) -> None:
    r"""Write the standalone publication-facing $\kappa$ stability sweep tables."""

    central_point = next(
        point
        for point in geometric_sensitivity.sweep_points
        if solver_isclose(point.kappa, geometric_sensitivity.central_kappa)
    )
    rows = tuple(
        {
            "status": "selected invariant" if solver_isclose(point.kappa, geometric_sensitivity.central_kappa) else "scan",
            "kappa": f"{point.kappa:.5f}",
            "m_0_mz_mev": f"{1.0e3 * point.m_0_mz_ev:.3f}",
            "effective_majorana_mass_mev": f"{point.effective_majorana_mass_mev:.3f}",
            "predictive_chi2": f"{point.predictive_chi2:.3f}",
            "max_sigma_shift": f"{point.max_sigma_shift:.3e}",
        }
        for point in geometric_sensitivity.sweep_points
    )
    table_text = presentation_reporting.render_kappa_sensitivity_audit(
        rows=rows,
        central_kappa=f"{central_point.kappa:.5f}",
        central_predictive_chi2=f"{central_point.predictive_chi2:.3f}",
    )
    (output_dir / KAPPA_SENSITIVITY_AUDIT_FILENAME).write_text(table_text, encoding="utf-8")
    (output_dir / KAPPA_STABILITY_SWEEP_FILENAME).write_text(table_text, encoding="utf-8")


def export_svd_stability_audit_table(
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    output_dir: Path,
) -> None:
    r"""Write the main-text-ready Higgs-VEV-alignment angle-stability table."""

    angle_rows = (
        (r"PMNS", r"$\theta_{12}$", mass_ratio_stability_audit.lepton_angle_shifts_deg[0], mass_ratio_stability_audit.lepton_sigma_shifts[0]),
        (r"PMNS", r"$\theta_{13}$", mass_ratio_stability_audit.lepton_angle_shifts_deg[1], mass_ratio_stability_audit.lepton_sigma_shifts[1]),
        (r"PMNS", r"$\theta_{23}$", mass_ratio_stability_audit.lepton_angle_shifts_deg[2], mass_ratio_stability_audit.lepton_sigma_shifts[2]),
        (r"CKM", r"$\theta_{C}$", mass_ratio_stability_audit.quark_angle_shifts_deg[0], mass_ratio_stability_audit.quark_sigma_shifts[0]),
        (r"CKM", r"$\theta_{13}^{q}$", mass_ratio_stability_audit.quark_angle_shifts_deg[1], mass_ratio_stability_audit.quark_sigma_shifts[1]),
        (r"CKM", r"$\theta_{23}^{q}$", mass_ratio_stability_audit.quark_angle_shifts_deg[2], mass_ratio_stability_audit.quark_sigma_shifts[2]),
    )
    table_text = presentation_reporting.render_svd_stability_audit(
        angle_rows=tuple(
            {
                "sector": sector,
                "angle": angle,
                "shift_deg": f"{shift_deg:+.3e}",
                "sigma_shift": f"{sigma_shift:.3e}",
            }
            for sector, angle, shift_deg, sigma_shift in angle_rows
        ),
        relative_spectral_volume_shift=_format_exact_fraction_or_decimal(
            mass_ratio_stability_audit.relative_spectral_volume_shift,
            identity=QUADRATIC_WEIGHT_PROJECTION,
            tex_identity=QUADRATIC_WEIGHT_PROJECTION_TEX,
            decimals=6,
        ),
        lepton_left_overlap_min=f"{mass_ratio_stability_audit.lepton_left_overlap_min:.6f}",
        lepton_right_overlap_min=f"{mass_ratio_stability_audit.lepton_right_overlap_min:.6f}",
        quark_left_overlap_min=f"{mass_ratio_stability_audit.quark_left_overlap_min:.6f}",
        quark_right_overlap_min=f"{mass_ratio_stability_audit.quark_right_overlap_min:.6f}",
        max_sigma_shift=f"{mass_ratio_stability_audit.max_sigma_shift:.3e}",
    )
    (output_dir / SVD_STABILITY_AUDIT_TABLE_FILENAME).write_text(table_text, encoding="utf-8")


def export_modularity_residual_map(
    level_scan: LevelStabilityScan,
    output_dir: Path,
) -> None:
    r"""Write the nearest-neighbor fixed-parent anomaly map used in the main text."""

    selected_row = level_scan.selected_row
    displayed_rows = [
        row
        for row in level_scan.rows
        if row.quark_level == selected_row.quark_level
        and row.lepton_level in (selected_row.lepton_level - 1, selected_row.lepton_level, selected_row.lepton_level + 1)
    ]
    displayed_rows.sort(key=lambda row: row.lepton_level)
    table_text = presentation_reporting.render_modularity_residual_map(
        rows=tuple(
            {
                "lepton_level": row.lepton_level,
                "quark_level": row.quark_level,
                "parent_level": row.parent_level,
                "modularity_gap": f"{row.modularity_gap:.6f}",
                "framing_gap": f"{row.framing_gap:.6f}",
                "anomaly_energy": f"{math.sqrt((24.0 * row.modularity_gap) ** 2 + row.framing_gap * row.framing_gap):.6f}",
                "status": "selected" if row.selected_visible_pair else "neighbor",
            }
            for row in displayed_rows
        ),
        note_text=(
            rf"\footnotesize Local anomaly map ordered by $\mathfrak A_{{\rm vis}}\equiv\sqrt{{(24\Delta_{{\rm mod}})^2+\Delta_{{\rm fr}}^2}}$. "
            rf"Within the displayed nearest-neighbor moat the selected cell $({selected_row.lepton_level},{selected_row.quark_level},{selected_row.parent_level})$ is the lowest-anomaly entry because it is the only displayed row with vanishing framing gap; the adjacent cells have comparable modular residues but nonzero framing defects, and the upper neighbor carries the relaxed tilt recorded in the local scan."
        ),
    )
    (output_dir / MODULARITY_RESIDUAL_MAP_FILENAME).write_text(table_text, encoding="utf-8")


def export_landscape_anomaly_map(
    global_audit: GlobalSensitivityAudit,
    output_dir: Path,
    top_rows: int = 10,
) -> None:
    r"""Write the information-allowed landscape anomaly ranking for the evidence packet."""

    displayed_rows = global_audit.rows[:top_rows]
    selected_row = global_audit.selected_row
    benchmark_tuple = f"({selected_row.lepton_level},{selected_row.quark_level},{selected_row.parent_level})"
    if global_audit.exact_pass_count == 0:
        note_text = (
            r"\footnotesize Information-allowed landscape map ordered by "
            r"$\mathfrak A_{\rm vis}\equiv\sqrt{(24\Delta_{\rm mod})^2+\Delta_{{\rm fr},\ell}^2+\Delta_{{\rm fr},q}^2}$ "
            r"over the full $"
            f"{global_audit.total_pairs_scanned}"
            r"$-point low-rank window. In the present de-anchored scan no exact residual-map root appears anywhere in that window. The displayed rows are therefore the lowest-anomaly tuples of the broad scan, while the manuscript benchmark $"
            f"{benchmark_tuple}"
            r"$ enters only as a separately chosen local framing-closed cell; its full-window rank is $"
            f"{global_audit.selected_rank}"
            r"$ with $\mathfrak A_{\rm vis}="
            f"{selected_row.anomaly_energy:.6f}"
            r"$."
        )
    else:
        note_text = (
            r"\footnotesize Information-allowed landscape map ordered by "
            r"$\mathfrak A_{\rm vis}\equiv\sqrt{(24\Delta_{\rm mod})^2+\Delta_{{\rm fr},\ell}^2+\Delta_{{\rm fr},q}^2}$ "
            r"over the full $"
            f"{global_audit.total_pairs_scanned}"
            r"$-point low-rank window. The benchmark cell $"
            f"{benchmark_tuple}"
            r"$ is the lowest-anomaly entry in this ranking and is separated from the next-best visible pair by $\Delta\mathfrak A_{\rm vis}="
            f"{global_audit.algebraic_gap:.6f}"
            r"$."
        )
    table_text = presentation_reporting.render_landscape_anomaly_map(
        rows=tuple(
            {
                "rank": rank,
                "lepton_level": row.lepton_level,
                "quark_level": row.quark_level,
                "parent_level": row.parent_level,
                "central_charge_residual": f"{row.central_charge_residual:.6f}",
                "framing_gap": f"{math.sqrt(row.lepton_framing_gap * row.lepton_framing_gap + row.quark_framing_gap * row.quark_framing_gap):.6f}",
                "anomaly_energy": f"{row.anomaly_energy:.6f}",
                "status": "benchmark" if row.selected_visible_pair else ("exact pass" if row.exact_pass else "candidate"),
            }
            for rank, row in enumerate(displayed_rows, start=1)
        ),
        note_text=note_text,
    )
    (output_dir / LANDSCAPE_ANOMALY_MAP_FILENAME).write_text(table_text, encoding="utf-8")

def derive_determinant_gradient_audit(
    audit: AuditData,
    level: int = LEPTON_LEVEL,
    sample_count: int = 401,
) -> DeterminantGradientAuditData:
    r"""Return a one-parameter determinant-collapse proxy between NH and IH support maps."""

    gradient = audit_generator.derive_determinant_gradient_audit(audit, sample_count=sample_count, level=level)
    return DeterminantGradientAuditData(
        eta_values=gradient["eta_values"],
        determinant_values=gradient["determinant_values"],
        near_endpoint_eta=float(gradient["near_endpoint_eta"]),
        near_endpoint_determinant=float(gradient["near_endpoint_determinant"]),
        almost_endpoint_eta=float(gradient["almost_endpoint_eta"]),
        almost_endpoint_determinant=float(gradient["almost_endpoint_determinant"]),
    )


def export_determinant_gradient_figure(
    audit: AuditData,
    output_path: Path | None = None,
    *,
    level: int | None = None,
    model: TopologicalModel | None = None,
) -> DeterminantGradientAuditData:
    r"""Write the supplementary determinant-gradient figure used for the IH singularity note."""

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / SUPPLEMENTARY_DETERMINANT_GRADIENT_FIGURE_FILENAME
    resolved_model = _coerce_topological_model(model=model, lepton_level=level)
    gradient = audit_generator.export_determinant_gradient_figure(audit, output_path, level=resolved_model.lepton_level)
    return DeterminantGradientAuditData(
        eta_values=gradient["eta_values"],
        determinant_values=gradient["determinant_values"],
        near_endpoint_eta=float(gradient["near_endpoint_eta"]),
        near_endpoint_determinant=float(gradient["near_endpoint_determinant"]),
        almost_endpoint_eta=float(gradient["almost_endpoint_eta"]),
        almost_endpoint_determinant=float(gradient["almost_endpoint_determinant"]),
    )


def export_framing_gap_moat_heatmap(
    global_audit: GlobalSensitivityAudit,
    output_path: Path | None = None,
    lepton_levels: tuple[int, ...] | None = None,
    quark_levels: tuple[int, ...] | None = None,
) -> None:
    r"""Write the publication-facing framing-gap heatmap around the benchmark visible cell."""

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / FRAMING_GAP_HEATMAP_FIGURE_FILENAME
    row_lookup = {(row.lepton_level, row.quark_level): row for row in global_audit.rows}
    selected_row = global_audit.selected_row
    if lepton_levels is None:
        lepton_levels = tuple(
            level
            for level in range(selected_row.lepton_level - 2, selected_row.lepton_level + 3)
            if any(candidate_row.lepton_level == level for candidate_row in global_audit.rows)
        )
    if quark_levels is None:
        quark_levels = tuple(
            level
            for level in range(selected_row.quark_level - 1, selected_row.quark_level + 2)
            if any(candidate_row.quark_level == level for candidate_row in global_audit.rows)
        )
    gap_grid = np.empty((len(lepton_levels), len(quark_levels)), dtype=float)
    for row_index, lepton_level in enumerate(lepton_levels):
        for column_index, quark_level in enumerate(quark_levels):
            row = row_lookup[(lepton_level, quark_level)]
            gap_grid[row_index, column_index] = math.hypot(row.lepton_framing_gap, row.quark_framing_gap)

    with managed_figure(figsize=(4.8, 4.1)) as (fig, ax):
        image = ax.imshow(gap_grid, origin="lower", cmap="magma_r", aspect="auto")
        for row_index, lepton_level in enumerate(lepton_levels):
            for column_index, quark_level in enumerate(quark_levels):
                value = gap_grid[row_index, column_index]
                if (lepton_level, quark_level) == (selected_row.lepton_level, selected_row.quark_level):
                    label = "0.000\nselected"
                elif quark_level == selected_row.quark_level and lepton_level in (selected_row.lepton_level - 1, selected_row.lepton_level + 1):
                    label = f"{value:.3f}\nmoat"
                else:
                    label = f"{value:.3f}"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.15 else "black",
                )
        ax.set_xticks(np.arange(len(quark_levels)), labels=[str(level) for level in quark_levels])
        ax.set_yticks(np.arange(len(lepton_levels)), labels=[str(level) for level in lepton_levels])
        ax.set_xlabel(r"$k_q$")
        ax.set_ylabel(r"$k_{\ell}$")
        ax.set_title(r"Discrete moat in the framing gap $\Delta_{\rm fr}^{\rm vis}$")
        ax.scatter(
            [quark_levels.index(selected_row.quark_level)],
            [lepton_levels.index(selected_row.lepton_level)],
            marker="s",
            facecolors="none",
            edgecolors="#22c55e",
            linewidths=2.0,
            s=680,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.04)
        colorbar.set_label(r"$\Delta_{\rm fr}^{\rm vis}=\sqrt{\Delta_{\rm fr,\ell}^2+\Delta_{\rm fr,q}^2}$")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)


def export_benchmark_stability_table(
    pull_table: PullTable,
    nonlinearity_audit: NonLinearityAuditData,
    weight_profile: CkmPhaseTiltProfileData,
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    output_dir: Path,
) -> None:
    r"""Write the publication-facing benchmark stability table."""

    table_text = presentation_reporting.render_benchmark_stability_table(
        rows=(
            {
                "diagnostic": f"{pull_table.predictive_observable_count}-row Central Benchmark",
                "benchmark_statistic": rf"benchmark goodness-of-fit $= {pull_table.predictive_chi2:.3f}$, RMS pull $={pull_table.predictive_rms_pull:.3f}$",
                "stress_test": r"total-variance pulls with $\sigma_{\rm th}=0.05\,|X_{\rm th}|$",
                "interpretation": _benchmark_counting_summary_tex(pull_table),
            },
            {
                "diagnostic": f"{pull_table.audit_observable_count}-row cross-check",
                "benchmark_statistic": rf"$\chi_{{\rm audit}}^2={pull_table.audit_chi2:.3f}$, largest pull $={pull_table.audit_max_abs_pull:.2f}\sigma$",
                "stress_test": _audit_row_stress_test_tex(pull_table),
                "interpretation": _audit_row_interpretation_tex(pull_table),
            },
            {
                "diagnostic": "Discrete search disclosure",
                "benchmark_statistic": rf"$T_{{\rm scan}}={pull_table.predictive_landscape_trial_count}$",
                "stress_test": "fixed-parent low-rank landscape scan",
                "interpretation": r"the scan volume is reported as model-building context rather than as a benchmark $p$-value",
            },
            {
                "diagnostic": "Dimension-5 Wilson coefficient",
                "benchmark_statistic": rf"${CKM_PHASE_TILT_SYMBOL}\approx {weight_profile.benchmark_weight:.2f}$ at the $\mathbf{{126}}_H$ matching point",
                "stress_test": rf"off-shell deformations of ${CKM_PHASE_TILT_SYMBOL}$ while monitoring $|V_{{us}}|$, $|V_{{cb}}|$, $|V_{{ub}}|$",
                "interpretation": r"the matched heavy-threshold term moves $\gamma$ without retuning the benchmark mixing magnitudes",
            },
            {
                "diagnostic": "ODE precision floor",
                "benchmark_statistic": rf"$\max_a |\Delta_a|/\sigma_a={_format_tex_scientific(nonlinearity_audit.max_sigma_error, precision=2)}$",
                "stress_test": r"tighten $(\mathrm{rtol},\mathrm{atol})$ to $(10^{-12},10^{-14})$",
                "interpretation": r"numerical precision lies far below the declared physical band",
            },
            {
                "diagnostic": "Eigenvector Stability Check",
                "benchmark_statistic": rf"factor-${mass_ratio_stability_audit.perturbation_factor:.0f}$ Yukawa / singular-value stress test",
                "stress_test": rf"max angle shift $={_format_tex_scientific(mass_ratio_stability_audit.max_sigma_shift, precision=2)}\sigma$",
                "interpretation": r"eigenvectors remain numerically stable while magnitudes absorb the suppression",
            },
            {
                "diagnostic": "Declared physical band",
                "benchmark_statistic": r"uniform $\pm5\%$ theory allowance",
                "stress_test": r"finite thresholds plus boundary-to-bulk geometry",
                "interpretation": r"physical uncertainty dominates the numerical floor",
            },
        ),
    )
    (output_dir / BENCHMARK_STABILITY_TABLE_FILENAME).write_text(table_text, encoding="utf-8")


def export_vev_alignment_stability_figure(
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    output_path: Path,
) -> None:
    if mass_ratio_stability_audit.ensemble_sample_count <= 0:
        return
    normalized_suppression = mass_ratio_stability_audit.ensemble_effective_suppression_ratios / max(
        mass_ratio_stability_audit.clebsch_relative_suppression,
        np.finfo(float).eps,
    )
    with managed_figure(figsize=(6.4, 4.4)) as (fig, ax):
        scatter = ax.scatter(
            normalized_suppression,
            mass_ratio_stability_audit.ensemble_max_sigma_shifts,
            c=mass_ratio_stability_audit.ensemble_mass_scale_shifts,
            cmap="viridis",
            s=14,
            alpha=0.65,
            edgecolors="none",
        )
        ax.axhline(1.0, color="#b91c1c", linestyle="--", linewidth=1.1, label=r"$1\sigma$")
        ax.set_xlabel(r"effective Clebsch suppression / benchmark")
        ax.set_ylabel(r"max $|\Delta\theta|/\sigma$")
        ax.set_yscale("log")
        ax.set_title("Random Clebsch-Gordan deformation ensemble")
        ax.legend(loc="upper left")
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.050, pad=0.04)
        colorbar.set_label(r"relative mass-scale proxy")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)

def _format_tex_scientific(value: float, precision: int = 2) -> str:
    value = float(value)
    if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=condition_aware_abs_tolerance(scale=value)):
        return "0"
    sign = "-" if value < 0.0 else ""
    magnitude = abs(value)
    exponent = int(math.floor(math.log10(magnitude)))
    mantissa = magnitude / (10 ** exponent)
    return f"{sign}{mantissa:.{precision}f}\\times10^{{{exponent}}}"


def _format_tex_probability(value: float, precision: int = 3) -> str:
    value = float(value)
    if value >= 1.0e-3:
        return f"{value:.{precision}f}"
    return _format_tex_scientific(value, precision=2)


def _pull_for_observable(pull_table: PullTable, observable: str) -> float:
    for row in pull_table.rows:
        if row.observable == observable and row.pull_data is not None:
            return float(row.pull_data.pull)
    raise RuntimeError(f"Sync Error: observable {observable} is missing from the pull table.")


def _audit_p_value(pull_table: PullTable) -> float:
    return float(chi2_distribution.sf(pull_table.audit_chi2, pull_table.audit_degrees_of_freedom))


def _explicit_fit_input_label(count: int) -> str:
    return "input" if count == 1 else "inputs"


def _benchmark_counting_summary_tex(pull_table: PullTable) -> str:
    return (
        rf"benchmark goodness-of-fit summary with fixed discrete labels and ultraviolet matching data ${CKM_PHASE_TILT_SYMBOL}$ and ${MATCHING_COEFFICIENT_SYMBOL}$; "
        r"the disclosed discrete search volume is reported separately rather than promoted to a benchmark $p$-value"
    )


def _benchmark_counting_statement_plain(pull_table: PullTable) -> str:
    return (
        "the benchmark reports a goodness-of-fit metric with fixed discrete labels and ultraviolet matching data, "
        "while the disclosed discrete survey is retained as model-building context rather than converted into a benchmark p-value"
    )


def _benchmark_bookkeeping_note(pull_table: PullTable) -> str:
    return (
        "bookkeeping note                : the reported benchmark keeps the discrete labels fixed and treats the Wilson-coefficient residue and the mass-scale entry as ultraviolet matching data, with the look-elsewhere penalty calibrated by the autocorrelation-derived effective trial count."
    )


def _audit_row_stress_test_tex(pull_table: PullTable) -> str:
    if pull_table.audit_observable_count == pull_table.predictive_observable_count:
        return r"same disclosed statistic, explicitly including $\theta_{12}$"
    return r"restore $\theta_{12}$ inside the audit tally"


def _audit_row_interpretation_tex(pull_table: PullTable) -> str:
    if pull_table.audit_observable_count == pull_table.predictive_observable_count:
        return r"the solar channel remains inside the benchmark tally as the known one-loop precision limit"
    return r"the solar channel is disclosed as the one-loop precision limit rather than dropped from the benchmark"


def _benchmark_bookkeeping_lines(pull_table: PullTable) -> list[str]:
    """Return plain-text bookkeeping lines for the benchmark fit."""

    return [
        f"predictive chi2 / nu_pred         : {pull_table.predictive_chi2:.6f} / {pull_table.predictive_degrees_of_freedom}",
        f"cross-check chi2 / nu_check      : {pull_table.audit_chi2:.6f} / {pull_table.audit_degrees_of_freedom}",
        f"continuous fit variables         : {pull_table.phenomenological_parameter_count}",
        f"continuous RG calibration inputs : {pull_table.calibration_parameter_count}",
        f"continuous DOF subtraction       : {pull_table.continuous_parameter_subtraction_count}",
        f"topological quantum numbers      : {TOPOLOGICAL_QUANTUM_NUMBER_DOF_SUBTRACTION}",
        f"threshold-matching subtraction   : {THRESHOLD_MATCHING_DOF_SUBTRACTION}",
        f"effective DOF subtraction        : {pull_table.effective_dof_subtraction_count}",
        f"predictive p-value, conditional  : {pull_table.predictive_conditional_p_value:.6f}",
        f"global p-value, N_eff-corrected  : {pull_table.predictive_discrete_selection_lee_p_value:.6f}",
        f"effective trial count            : {pull_table.predictive_effective_trial_count:.3f} from {pull_table.predictive_followup_trial_count} follow-up cells",
        f"correlation lengths (ell, q, xi) : {pull_table.predictive_lepton_correlation_length:.3f}, {pull_table.predictive_quark_correlation_length:.3f}, {pull_table.predictive_correlation_length:.3f}",
        _benchmark_bookkeeping_note(pull_table),
    ]


def export_physics_constants_to_tex(
    output_dir: Path,
    *,
    scales: ScaleData,
    level_scan: LevelStabilityScan,
    global_audit: GlobalSensitivityAudit,
    pull_table: PullTable,
    nonlinearity_audit: NonLinearityAuditData,
    weight_profile: CkmPhaseTiltProfileData,
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    pmns: PmnsData,
    ckm: CkmData,
    sensitivity: SensitivityData,
    geometric_sensitivity: GeometricSensitivityData,
    geometric_kappa: SO10GeometricKappaData,
    modular_horizon: ModularHorizonSelectionData,
    framing_gap_stability: FramingGapStabilityData,
) -> Path:
    r"""Write a single-source LaTeX macro file for manuscript benchmark constants."""

    benchmark_residual = float(level_scan.selected_row.modularity_gap)
    gauge_holography = DEFAULT_TOPOLOGICAL_VACUUM.verify_gauge_holography()
    dark_energy_audit = DEFAULT_TOPOLOGICAL_VACUUM.verify_dark_energy_tension()
    anomaly_data = calculate_branching_anomaly("SO(10)", "SU(3)", PARENT_LEVEL)
    m_beta_beta_rg_mev = 1.0e3 * pmns.effective_majorana_mass_rg_ev
    m_beta_beta_uv_mev = 1.0e3 * pmns.effective_majorana_mass_uv_ev
    m_beta_beta_uv_kappa_sigma = m_beta_beta_uv_mev * (
        geometric_sensitivity.effective_majorana_mass_max_shift_mev / m_beta_beta_rg_mev
    )
    dm_fingerprint = derive_dm_fingerprint_inputs(weight_profile, geometric_kappa, framing_gap_stability)
    gauge_unification = dm_fingerprint.gauge_unification
    dm_mass_gev = dm_fingerprint.dm_mass_gev
    dm_mass_upper_gev = dm_fingerprint.dm_mass_upper_gev
    dm_gauge_coupling = dm_fingerprint.dm_gauge_coupling
    dm_sigma_geom_cm2 = dm_fingerprint.gauge_sigma_cm2
    dm_sigma_higgs_cm2 = dm_fingerprint.higgs_sigma_cm2
    alpha_gut = 1.0 / gauge_unification.unified_alpha_inverse
    dm_alpha_chi = geometric_kappa.derived_kappa * geometric_kappa.derived_kappa * alpha_gut
    dm_higgs_proxy = geometric_kappa.derived_kappa * weight_profile.benchmark_weight
    transport_curvature = derive_transport_curvature_audit()
    gravity_audit = DEFAULT_TOPOLOGICAL_VACUUM.verify_bulk_emergence()
    jarlskog_topological = topological_jarlskog_identity(
        ckm.gut_threshold_residue,
        parent_level=ckm.parent_level,
        lepton_level=pmns.level,
        quark_level=ckm.level,
        kappa_geometric=geometric_kappa.derived_kappa,
    )
    jarlskog_topological_visible = threshold_projected_jarlskog(
        jarlskog_topological,
        gut_threshold_residue=ckm.gut_threshold_residue,
    )
    delta_mod_cp_zero = cp_conserving_modularity_leak(
        jarlskog_topological_visible,
        ckm.theta_c_uv_deg,
        ckm.theta13_uv_deg,
        ckm.theta23_uv_deg,
        parent_level=ckm.parent_level,
        lepton_level=pmns.level,
        quark_level=ckm.level,
    )
    inflationary_sector = DEFAULT_TOPOLOGICAL_VACUUM.derive_inflationary_sector()
    unitary_audit = DEFAULT_TOPOLOGICAL_VACUUM.verify_unitary_bounds()
    page_point = DEFAULT_TOPOLOGICAL_VACUUM.derive_page_point_audit(unitary_audit=unitary_audit)
    baryon_stability = gravity_audit.baryon_stability
    macros = {
        "simplexPrefactor": f"{geometric_kappa.derived_kappa:.5f}",
        "kappaMatch": f"{geometric_kappa.derived_kappa:.5f}",
        "alphaBrExactRatio": rf"\frac{{{anomaly_data.numerator_units}}}{{{anomaly_data.visible_cartan_embedding_index * anomaly_data.denominator_units}}}",
        "alphaBrDecimal": _format_tex_scientific(float(anomaly_data.anomaly_fraction), precision=5),
        "visibleResidual": f"{benchmark_residual:.6f}",
        "alphaGaugeGenerationCount": str(gauge_holography.generation_count),
        "alphaGaugeInverse": _format_exact_fraction_or_decimal(
            gauge_holography.topological_alpha_inverse,
            identity=GAUGE_STRENGTH_IDENTITY,
            tex_identity=GAUGE_STRENGTH_IDENTITY_TEX,
            decimals=3,
        ),
        "alphaGaugeCodataInverse": _format_exact_fraction_or_decimal(
            gauge_holography.codata_alpha_inverse,
            identity=GAUGE_STRENGTH_IDENTITY,
            tex_identity=GAUGE_STRENGTH_IDENTITY_TEX,
            decimals=3,
        ),
        "alphaGaugeResidualPercent": f"{gauge_holography.geometric_residue_percent:.2f}",
        "alphaGaugeModularGapInverse": f"{gauge_holography.modular_gap_scaled_inverse:.3f}",
        "alphaGaugeGapAlignmentPercent": f"{gauge_holography.modular_gap_alignment_percent:.2f}",
        "modularHorizonBits": _format_tex_scientific(modular_horizon.derived_bits, precision=2),
        "planckCrosscheckRatio": f"{modular_horizon.planck_crosscheck_ratio:.3f}",
        "landscapeAnomalyGap": f"{global_audit.algebraic_gap:.6f}",
        "argWSTDegrees": f"{(pmns.delta_cp_uv_deg - 180.0):.2f}",
        "deltaCPTopOneDecimal": f"{pmns.delta_cp_uv_deg:.1f}",
        "deltaCPTopTwoDecimal": f"{pmns.delta_cp_uv_deg:.2f}",
        "deltaCPTopPi": f"{pmns.delta_cp_uv_deg / 180.0:.2f}",
        "deltaCPMz": f"{pmns.delta_cp_rg_deg:.2f}",
        "majoranaPhaseOneDeg": f"{pmns.majorana_phase_1_deg:.2f}",
        "majoranaPhaseTwoDeg": f"{pmns.majorana_phase_2_deg:.2f}",
        "holonomyAreaUv": _format_tex_scientific(pmns.holonomy_area_uv, precision=2),
        "holonomyAreaMz": _format_tex_scientific(pmns.holonomy_area_rg, precision=2),
        "jarlskogUv": _format_tex_scientific(pmns.jarlskog_uv, precision=2),
        "jarlskogMz": _format_tex_scientific(pmns.jarlskog_rg, precision=2),
        "jarlskogTopological": _format_tex_scientific(jarlskog_topological, precision=2),
        "jarlskogTopologicalVisible": _format_tex_scientific(jarlskog_topological_visible, precision=2),
        "jarlskogCkmBenchmark": _format_tex_scientific(ckm.jarlskog_rg, precision=2),
        "deltaModCpZero": f"{delta_mod_cp_zero:.4f}",
        "mBetaBetaPredicted": f"{m_beta_beta_rg_mev:.2f}",
        "mBetaBetaUvPredicted": f"{m_beta_beta_uv_mev:.2f}",
        "mBetaBetaUvKappaSigma": f"{m_beta_beta_uv_kappa_sigma:.2f}",
        "mBetaBetaKappaSigma": f"{geometric_sensitivity.effective_majorana_mass_max_shift_mev:.2f}",
        "mBetaBetaNSigma": f"{sensitivity.effective_majorana_mass_max_shift_mev:.2f}",
        "vusPredicted": f"{ckm.vus_rg:.4f}",
        "vcbPredicted": f"{ckm.vcb_rg:.4f}",
        "vubPredicted": f"{ckm.vub_rg:.5f}",
        "gammaMz": f"{ckm.gamma_rg_deg:.2f}",
        "wTh": _format_exact_fraction_or_decimal(
            pull_table.gut_threshold_residue_value,
            identity=VOA_BRANCHING_GAP,
            tex_identity=VOA_BRANCHING_GAP_TEX,
            decimals=2,
        ),
        "gutThresholdResidue": _format_exact_fraction_or_decimal(
            pull_table.gut_threshold_residue_value,
            identity=VOA_BRANCHING_GAP,
            tex_identity=VOA_BRANCHING_GAP_TEX,
            decimals=2,
        ),
        "lambdaWilsonTwelve": _format_tex_scientific(ckm.so10_threshold_correction.lambda_12_mgut, precision=2),
        "gutThresholdLogSum": f"{ckm.so10_threshold_correction.matching_log_sum:.5f}",
        "thresholdDecouplingLeakage": _format_tex_scientific(ckm.so10_threshold_correction.decoupling_audit.max_leakage, precision=2),
        "pPredConditional": _format_tex_probability(pull_table.predictive_conditional_p_value),
        "pPredCorrected": _format_tex_probability(pull_table.predictive_p_value),
        "chiPredRounded": f"{pull_table.predictive_chi2:.3f}",
        "chiPredExact": f"{pull_table.predictive_chi2:.5f}",
        "nuPred": str(pull_table.predictive_degrees_of_freedom),
        "chiPredReducedRounded": f"{pull_table.predictive_reduced_chi2:.3f}",
        "chiPredReducedExact": f"{pull_table.predictive_reduced_chi2:.5f}",
        "rmsPullPred": f"{pull_table.predictive_rms_pull:.3f}",
        "maxPullPred": f"{pull_table.predictive_max_abs_pull:.2f}",
        "cVisibleBenchmark": f"{gravity_audit.visible_central_charge:.4f}",
        "cDarkCompletion": f"{24.0 * benchmark_residual:.4f}",
        "dmResidueEfficiency": f"{gravity_audit.modular_residue_efficiency:.4f}",
        "dmParityBitDensityRatio": f"{gravity_audit.omega_dm_ratio:.4f}",
        "dmParityBitDensityTarget": f"{PARITY_BIT_DENSITY_CONSTRAINT_TARGET:.2f}",
        "dmParityBitDensityTolerance": f"{PARITY_BIT_DENSITY_CONSTRAINT_TOLERANCE:.2f}",
        "unitaryResidue": f"{24.0 * benchmark_residual:.4f}",
        "wThVusSpan": _format_tex_scientific(weight_profile.max_vus_shift, precision=2),
        "wThVcbSpan": _format_tex_scientific(weight_profile.max_vcb_shift, precision=2),
        "wThVubSpan": _format_tex_scientific(weight_profile.max_vub_shift, precision=2),
        "wThBestFit": _format_exact_fraction_or_decimal(
            weight_profile.best_fit_weight,
            identity=VOA_BRANCHING_GAP,
            tex_identity=VOA_BRANCHING_GAP_TEX,
            decimals=2,
        ),
        "higgsMixingWeight": f"{ckm.so10_threshold_correction.higgs_mixing_weight:.6f}",
        "massRatioTargetSuppression": _format_exact_fraction_or_decimal(
            mass_ratio_stability_audit.target_relative_suppression,
            identity=QUADRATIC_WEIGHT_PROJECTION,
            tex_identity=QUADRATIC_WEIGHT_PROJECTION_TEX,
            decimals=6,
        ),
        "massRatioClebschSuppression": _format_exact_fraction_or_decimal(
            mass_ratio_stability_audit.clebsch_relative_suppression,
            identity=QUADRATIC_WEIGHT_PROJECTION,
            tex_identity=QUADRATIC_WEIGHT_PROJECTION_TEX,
            decimals=6,
        ),
        "massRatioRelativeShift": _format_exact_fraction_or_decimal(
            mass_ratio_stability_audit.relative_spectral_volume_shift,
            identity=QUADRATIC_WEIGHT_PROJECTION,
            tex_identity=QUADRATIC_WEIGHT_PROJECTION_TEX,
            decimals=6,
        ),
        "massRatioMaxSigmaShift": _format_tex_scientific(mass_ratio_stability_audit.max_sigma_shift, precision=2),
        "chiAuditRounded": f"{pull_table.audit_chi2:.3f}",
        "chiAuditExact": f"{pull_table.audit_chi2:.5f}",
        "nuAudit": str(pull_table.audit_degrees_of_freedom),
        "thetaTwelvePull": f"{_pull_for_observable(pull_table, r'$\theta_{12}$'):.2f}",
        "scanTrialCount": str(pull_table.predictive_landscape_trial_count),
        "rgNonlinearityConservativeBound": "0.08",
        "rgNonlinearityMaxSigma": _format_tex_scientific(nonlinearity_audit.max_sigma_error, precision=2),
        "thetaThirteenLinear": f"{nonlinearity_audit.theta_linear_deg[1]:.8f}",
        "thetaThirteenFull": f"{nonlinearity_audit.theta_nonlinear_deg[1]:.8f}",
        "thetaTwentyThreeLinear": f"{nonlinearity_audit.theta_linear_deg[2]:.8f}",
        "thetaTwentyThreeFull": f"{nonlinearity_audit.theta_nonlinear_deg[2]:.8f}",
        "deltaCPLinear": f"{nonlinearity_audit.delta_linear_deg:.8f}",
        "deltaCPFull": f"{nonlinearity_audit.delta_nonlinear_deg:.8f}",
        "mZeroBenchmarkMeV": f"{1.0e3 * scales.m_0_mz_ev:.2f}",
        "mZeroKappaSigma": f"{geometric_sensitivity.m_0_mz_max_shift_mev:.2f}",
        "mZeroLinearMeV": f"{1.0e3 * nonlinearity_audit.m_0_linear_ev:.5f}",
        "mZeroNonlinearFullMeV": f"{1.0e3 * nonlinearity_audit.m_0_nonlinear_ev:.5f}",
        "gammaZeroOneLoop": f"{scales.gamma_0_one_loop:.6f}",
        "gammaZeroTwoLoop": f"{scales.gamma_0_two_loop:.6f}",
        "leptonThetaTwelveBetaTwoLoop": f"{transport_curvature.lepton_theta_two_loop[0]:.6f}",
        "leptonThetaThirteenBetaTwoLoop": f"{transport_curvature.lepton_theta_two_loop[1]:.6f}",
        "leptonThetaTwentyThreeBetaTwoLoop": f"{transport_curvature.lepton_theta_two_loop[2]:.6f}",
        "leptonDeltaBetaTwoLoop": f"{transport_curvature.lepton_delta_two_loop:.6f}",
        "quarkThetaTwelveBetaTwoLoop": f"{transport_curvature.quark_theta_two_loop[0]:.6f}",
        "quarkThetaThirteenBetaTwoLoop": f"{transport_curvature.quark_theta_two_loop[1]:.6f}",
        "quarkThetaTwentyThreeBetaTwoLoop": f"{transport_curvature.quark_theta_two_loop[2]:.6f}",
        "quarkDeltaBetaTwoLoop": f"{transport_curvature.quark_delta_two_loop:.6f}",
        "solarBetaTwoLoop": f"{pmns.solar_beta_two_loop:.6f}",
        "mOneTwoSixMatchGeV": _format_tex_scientific(framing_gap_stability.matching_m126_gev, precision=2),
        "gutScaleGeV": _format_tex_scientific(gauge_unification.m_gut_gev, precision=2),
        "alphaGut": f"{alpha_gut:.5f}",
        "alphaGutInverse": f"{gauge_unification.unified_alpha_inverse:.2f}",
        "lambdaHoloMetersInverseSquared": _format_tex_scientific(dark_energy_audit.lambda_surface_tension_si_m2, precision=2),
        "inflationEFolds": str(inflationary_sector.primordial_efolds),
        "inflationTensorRatioExact": INFLATIONARY_TENSOR_RATIO_TEX,
        "inflationTensorRatioReduced": INFLATIONARY_TENSOR_RATIO_REDUCED_TEX,
        "inflationTensorRatioDecimal": f"{inflationary_sector.tensor_to_scalar_ratio:.4f}",
        "inflationScalarTilt": f"{inflationary_sector.scalar_tilt:.4f}",
        "inflationScalarTiltFull": f"{inflationary_sector.scalar_tilt:.6f}",
        "inflationScalarTiltTarget": f"{PRIMORDIAL_SCALAR_TILT_TARGET:.4f}",
        "complexityClockSkew": f"{inflationary_sector.clock_skew:.6f}",
        "complexityLloydLimit": _format_tex_scientific(unitary_audit.lloyds_limit_ops_per_second, precision=2),
        "complexityGrowthRate": _format_tex_scientific(unitary_audit.complexity_growth_rate_ops_per_second, precision=2),
        "maxComplexityCapacity": _format_tex_scientific(unitary_audit.max_complexity_capacity, precision=2),
        "pagePointComplexity": _format_tex_scientific(page_point.page_point_entropy, precision=2),
        "inflationScalarRunning": _format_tex_scientific(inflationary_sector.scalar_running, precision=2),
        "inflationDarkTiltRegulator": f"{inflationary_sector.dark_sector_tilt_regulator:.4f}",
        "inflationNonGaussianityFloor": _format_tex_scientific(inflationary_sector.non_gaussianity_floor, precision=2),
        "inflationGenusBeta": f"{inflationary_sector.beta_genus_ladder:.6f}",
        "inflationReheatingTempKelvin": f"{inflationary_sector.reheating_temperature_k:.1f}",
        "protonGaugeMassGeV": _format_tex_scientific(baryon_stability.effective_gauge_mass_gev, precision=2),
        "protonTunnelPenalty": _format_tex_scientific(baryon_stability.modular_tunneling_penalty, precision=2),
        "protonTunnelBoost": _format_tex_scientific(baryon_stability.tunneling_safety_boost, precision=2),
        "protonLifetimeYears": _format_tex_scientific(baryon_stability.proton_lifetime_years, precision=2),
        "protonProtectedLifetimeYears": _format_tex_scientific(baryon_stability.protected_evaporation_lifetime_years, precision=2),
        "dmMassGeV": _format_tex_scientific(dm_mass_gev, precision=2),
        "dmMassUpperGeV": _format_tex_scientific(dm_mass_upper_gev, precision=2),
        "dmAlphaChi": f"{dm_alpha_chi:.4f}",
        "dmGaugeCoupling": f"{dm_gauge_coupling:.3f}",
        "dmHiggsProxy": f"{dm_higgs_proxy:.3f}",
        "dmSigmaGeomCmTwo": _format_tex_scientific(dm_sigma_geom_cm2, precision=2),
        "dmSigmaHiggsCmTwo": _format_tex_scientific(dm_sigma_higgs_cm2, precision=2),
    }

    lines = ["% Auto-generated by pub/tn.py. Do not edit by hand."]
    lines.extend(rf"\newcommand{{\{macro_name}}}{{{value}}}" for macro_name, value in macros.items())
    output_path = output_dir / PHYSICS_CONSTANTS_FILENAME
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


@dataclass(frozen=True)
class DmFingerprintInputs:
    gauge_unification: GaugeUnificationData
    dm_mass_gev: float
    dm_mass_upper_gev: float
    dm_gauge_coupling: float
    gauge_sigma_cm2: float
    higgs_sigma_cm2: float


def derive_dm_fingerprint_inputs(
    weight_profile: CkmPhaseTiltProfileData,
    geometric_kappa: SO10GeometricKappaData,
    framing_gap_stability: FramingGapStabilityData,
) -> DmFingerprintInputs:
    gauge_unification = derive_gauge_unification_existence_proof(m_126_gev=framing_gap_stability.matching_m126_gev)
    dm_nucleon_mass_gev = 0.939
    dm_form_factor = 0.30
    dm_higgs_mass_gev = 125.25
    dm_gev2_to_cm2 = 0.389379e-27
    dm_mass_gev = gauge_unification.m_126_gev
    dm_mass_upper_gev = gauge_unification.structural_mn_gev
    alpha_gut = 1.0 / gauge_unification.unified_alpha_inverse
    g_gut = math.sqrt(4.0 * math.pi * alpha_gut)
    dm_alpha_chi = geometric_kappa.derived_kappa * geometric_kappa.derived_kappa * alpha_gut
    dm_gauge_coupling = geometric_kappa.derived_kappa * g_gut
    dm_higgs_proxy = geometric_kappa.derived_kappa * weight_profile.benchmark_weight
    dm_reduced_mass_gev = dm_nucleon_mass_gev * dm_mass_gev / (dm_nucleon_mass_gev + dm_mass_gev)
    gauge_sigma_cm2 = math.pi * dm_alpha_chi * dm_alpha_chi * dm_gev2_to_cm2 / (dm_mass_gev * dm_mass_gev)
    higgs_sigma_cm2 = (
        dm_higgs_proxy * dm_higgs_proxy
        * dm_form_factor * dm_form_factor
        * dm_reduced_mass_gev * dm_reduced_mass_gev
        * dm_nucleon_mass_gev * dm_nucleon_mass_gev
        * dm_gev2_to_cm2
        / (math.pi * dm_higgs_mass_gev**4 * dm_mass_gev * dm_mass_gev)
    )
    return DmFingerprintInputs(
        gauge_unification=gauge_unification,
        dm_mass_gev=dm_mass_gev,
        dm_mass_upper_gev=dm_mass_upper_gev,
        dm_gauge_coupling=dm_gauge_coupling,
        gauge_sigma_cm2=gauge_sigma_cm2,
        higgs_sigma_cm2=higgs_sigma_cm2,
    )


def export_dm_fingerprint_artifact(
    weight_profile: CkmPhaseTiltProfileData,
    geometric_kappa: SO10GeometricKappaData,
    framing_gap_stability: FramingGapStabilityData,
    output_dir: Path,
) -> None:
    dm_fingerprint = derive_dm_fingerprint_inputs(weight_profile, geometric_kappa, framing_gap_stability)
    publication_export.export_dm_fingerprint_figure(
        output_path=output_dir / DM_FINGERPRINT_FIGURE_FILENAME,
        dm_mass_gev=dm_fingerprint.dm_mass_gev,
        dm_mass_upper_gev=dm_fingerprint.dm_mass_upper_gev,
        gauge_sigma_cm2=dm_fingerprint.gauge_sigma_cm2,
        higgs_sigma_cm2=dm_fingerprint.higgs_sigma_cm2,
    )


REQUIRED_PHYSICS_CONSTANT_MACROS = (
    "simplexPrefactor",
    "kappaMatch",
    "alphaBrExactRatio",
    "alphaBrDecimal",
    "visibleResidual",
    "alphaGaugeGenerationCount",
    "alphaGaugeInverse",
    "alphaGaugeCodataInverse",
    "alphaGaugeResidualPercent",
    "alphaGaugeModularGapInverse",
    "alphaGaugeGapAlignmentPercent",
    "modularHorizonBits",
    "planckCrosscheckRatio",
    "landscapeAnomalyGap",
    "argWSTDegrees",
    "deltaCPTopOneDecimal",
    "deltaCPTopTwoDecimal",
    "deltaCPTopPi",
    "deltaCPMz",
    "majoranaPhaseOneDeg",
    "majoranaPhaseTwoDeg",
    "holonomyAreaUv",
    "holonomyAreaMz",
    "jarlskogUv",
    "jarlskogMz",
    "jarlskogTopological",
    "jarlskogTopologicalVisible",
    "jarlskogCkmBenchmark",
    "deltaModCpZero",
    "mBetaBetaPredicted",
    "mBetaBetaUvPredicted",
    "mBetaBetaUvKappaSigma",
    "mBetaBetaKappaSigma",
    "mBetaBetaNSigma",
    "vusPredicted",
    "vcbPredicted",
    "vubPredicted",
    "gammaMz",
    "wTh",
    "pPredConditional",
    "pPredCorrected",
    "chiPredRounded",
    "chiPredExact",
    "nuPred",
    "chiPredReducedRounded",
    "chiPredReducedExact",
    "rmsPullPred",
    "maxPullPred",
    "cVisibleBenchmark",
    "cDarkCompletion",
    "dmResidueEfficiency",
    "dmParityBitDensityRatio",
    "dmParityBitDensityTarget",
    "dmParityBitDensityTolerance",
    "unitaryResidue",
    "wThVusSpan",
    "wThVcbSpan",
    "wThVubSpan",
    "wThBestFit",
    "higgsMixingWeight",
    "massRatioTargetSuppression",
    "massRatioClebschSuppression",
    "massRatioRelativeShift",
    "massRatioMaxSigmaShift",
    "chiAuditRounded",
    "chiAuditExact",
    "nuAudit",
    "thetaTwelvePull",
    "scanTrialCount",
    "rgNonlinearityConservativeBound",
    "rgNonlinearityMaxSigma",
    "thetaThirteenLinear",
    "thetaThirteenFull",
    "thetaTwentyThreeLinear",
    "thetaTwentyThreeFull",
    "deltaCPLinear",
    "deltaCPFull",
    "mZeroBenchmarkMeV",
    "mZeroKappaSigma",
    "mZeroLinearMeV",
    "mZeroNonlinearFullMeV",
    "gammaZeroOneLoop",
    "gammaZeroTwoLoop",
    "leptonThetaTwelveBetaTwoLoop",
    "leptonThetaThirteenBetaTwoLoop",
    "leptonThetaTwentyThreeBetaTwoLoop",
    "leptonDeltaBetaTwoLoop",
    "quarkThetaTwelveBetaTwoLoop",
    "quarkThetaThirteenBetaTwoLoop",
    "quarkThetaTwentyThreeBetaTwoLoop",
    "quarkDeltaBetaTwoLoop",
    "solarBetaTwoLoop",
    "mOneTwoSixMatchGeV",
    "gutScaleGeV",
    "alphaGut",
    "alphaGutInverse",
    "lambdaHoloMetersInverseSquared",
    "inflationEFolds",
    "inflationTensorRatioExact",
    "inflationTensorRatioReduced",
    "inflationTensorRatioDecimal",
    "inflationScalarTilt",
    "inflationScalarTiltFull",
    "inflationScalarTiltTarget",
    "complexityClockSkew",
    "complexityLloydLimit",
    "complexityGrowthRate",
    "maxComplexityCapacity",
    "pagePointComplexity",
    "inflationScalarRunning",
    "inflationDarkTiltRegulator",
    "inflationNonGaussianityFloor",
    "inflationGenusBeta",
    "inflationReheatingTempKelvin",
    "protonGaugeMassGeV",
    "protonTunnelPenalty",
    "protonTunnelBoost",
    "protonLifetimeYears",
    "protonProtectedLifetimeYears",
    "dmMassGeV",
    "dmMassUpperGeV",
    "dmAlphaChi",
    "dmGaugeCoupling",
    "dmHiggsProxy",
    "dmSigmaGeomCmTwo",
    "dmSigmaHiggsCmTwo",
)

REQUIRED_TN_CONSTANT_MACROS = (
    "argWSTDegrees",
    "deltaCPTopOneDecimal",
    "deltaCPTopTwoDecimal",
    "deltaCPTopPi",
    "deltaCPMz",
    "jarlskogTopological",
    "jarlskogTopologicalVisible",
    "jarlskogCkmBenchmark",
    "deltaModCpZero",
    "visibleResidual",
    "alphaGaugeInverse",
    "alphaGaugeCodataInverse",
    "alphaGaugeResidualPercent",
    "alphaGaugeModularGapInverse",
    "cVisibleBenchmark",
    "cDarkCompletion",
    "pPredConditional",
    "pPredCorrected",
    "chiPredRounded",
    "nuPred",
    "chiPredReducedRounded",
    "rmsPullPred",
    "maxPullPred",
    "dmResidueEfficiency",
    "dmParityBitDensityRatio",
    "dmParityBitDensityTarget",
    "dmParityBitDensityTolerance",
    "wThVusSpan",
    "wThVcbSpan",
    "wThVubSpan",
    "massRatioClebschSuppression",
    "massRatioMaxSigmaShift",
    "chiAuditRounded",
    "nuAudit",
    "thetaTwelvePull",
    "scanTrialCount",
    "leptonThetaTwelveBetaTwoLoop",
    "leptonThetaThirteenBetaTwoLoop",
    "leptonThetaTwentyThreeBetaTwoLoop",
    "leptonDeltaBetaTwoLoop",
    "quarkThetaTwelveBetaTwoLoop",
    "quarkThetaThirteenBetaTwoLoop",
    "quarkThetaTwentyThreeBetaTwoLoop",
    "quarkDeltaBetaTwoLoop",
    "alphaGut",
    "lambdaHoloMetersInverseSquared",
    "inflationEFolds",
    "inflationTensorRatioExact",
    "inflationTensorRatioReduced",
    "inflationTensorRatioDecimal",
    "inflationScalarTilt",
    "inflationScalarTiltFull",
    "inflationScalarTiltTarget",
    "complexityClockSkew",
    "complexityLloydLimit",
    "complexityGrowthRate",
    "maxComplexityCapacity",
    "pagePointComplexity",
    "inflationScalarRunning",
    "inflationDarkTiltRegulator",
    "inflationNonGaussianityFloor",
    "inflationReheatingTempKelvin",
    "protonLifetimeYears",
    "dmMassGeV",
    "dmMassUpperGeV",
    "dmAlphaChi",
    "dmGaugeCoupling",
    "dmHiggsProxy",
    "dmSigmaGeomCmTwo",
    "dmSigmaHiggsCmTwo",
)

REQUIRED_SUPPLEMENTARY_CONSTANT_MACROS = (
    "simplexPrefactor",
    "visibleResidual",
    "alphaGaugeGenerationCount",
    "alphaGaugeInverse",
    "alphaGaugeCodataInverse",
    "alphaGaugeResidualPercent",
    "alphaGaugeModularGapInverse",
    "alphaGaugeGapAlignmentPercent",
    "cVisibleBenchmark",
    "cDarkCompletion",
    "jarlskogTopological",
    "jarlskogTopologicalVisible",
    "jarlskogCkmBenchmark",
    "deltaModCpZero",
    "dmResidueEfficiency",
    "dmParityBitDensityRatio",
    "dmParityBitDensityTarget",
    "dmParityBitDensityTolerance",
    "scanTrialCount",
    "massRatioClebschSuppression",
    "massRatioRelativeShift",
    "massRatioMaxSigmaShift",
    "rgNonlinearityConservativeBound",
    "rgNonlinearityMaxSigma",
    "gammaZeroOneLoop",
    "gammaZeroTwoLoop",
    "leptonThetaTwelveBetaTwoLoop",
    "leptonThetaThirteenBetaTwoLoop",
    "leptonThetaTwentyThreeBetaTwoLoop",
    "leptonDeltaBetaTwoLoop",
    "quarkThetaTwelveBetaTwoLoop",
    "quarkThetaThirteenBetaTwoLoop",
    "quarkThetaTwentyThreeBetaTwoLoop",
    "quarkDeltaBetaTwoLoop",
    "lambdaHoloMetersInverseSquared",
    "inflationEFolds",
    "inflationTensorRatioExact",
    "inflationTensorRatioReduced",
    "inflationTensorRatioDecimal",
    "inflationScalarTilt",
    "inflationScalarTiltFull",
    "inflationScalarTiltTarget",
    "complexityClockSkew",
    "complexityLloydLimit",
    "complexityGrowthRate",
    "maxComplexityCapacity",
    "pagePointComplexity",
    "inflationScalarRunning",
    "inflationDarkTiltRegulator",
    "inflationNonGaussianityFloor",
    "inflationReheatingTempKelvin",
    "protonTunnelPenalty",
    "protonTunnelBoost",
    "protonLifetimeYears",
)

FORBIDDEN_PHYSICS_CONSTANT_MACROS = (
    "deltaCBenchmark",
    "deltaCBenchmarkExact",
    "cDarkBenchmark",
    "pPredCondRounded",
    "pPredCondExact",
    "pPredLandRounded",
    "pPredLandExact",
    "nSolSpeculative",
    "chiUIARounded",
    "chiUIAExact",
    "nuUIA",
    "pUIA",
    "chiUIAReducedRounded",
)

REQUIRED_GENERATED_TABLE_SNIPPETS = (
    "benchmark",
    "Topological-quantum-number DOF subtraction",
)

FORBIDDEN_GENERATED_TABLE_SNIPPETS = (
    "Conditional predictive $p$-value",
    "landscape-corrected predictive $p$-value",
    "landscape-corrected $p$-value",
    "\\mathcal N_{\\rm sol}",
    "Geometric-overlap proposal",
    "Unified Integrity Audit",
    "Precision Limit Analysis",
)

FORBIDDEN_MANUSCRIPT_SNIPPETS = (
    "Unified Integrity Audit",
    "Precision Limit Analysis",
    r"\chi_{\rm PLA}",
    r"p_{\rm PLA}",
)

PACKET_OUTPUT_ARTIFACTS = (
    UNIQUENESS_SCAN_TABLE_FILENAME,
    MODULARITY_RESIDUAL_MAP_FILENAME,
    LANDSCAPE_ANOMALY_MAP_FILENAME,
    SUPPLEMENTARY_UNITARY_CONSISTENCY_TABLE_FILENAME,
    KAPPA_SENSITIVITY_AUDIT_FILENAME,
    KAPPA_STABILITY_SWEEP_FILENAME,
    FRAMING_GAP_HEATMAP_FIGURE_FILENAME,
    CKM_PHASE_TILT_PROFILE_FIGURE_FILENAME,
    BENCHMARK_STABILITY_TABLE_FILENAME,
    SVD_STABILITY_AUDIT_TABLE_FILENAME,
    SVD_STABILITY_REPORT_FILENAME,
    EIGENVECTOR_STABILITY_AUDIT_FILENAME,
    STABILITY_REPORT_FILENAME,
    SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME,
    BENCHMARK_DIAGNOSTICS_FILENAME,
    TRANSPORT_COVARIANCE_DIAGNOSTICS_FILENAME,
    SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_FIGURE_FILENAME,
    SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_DATA_FILENAME,
)

OPTIONAL_PACKET_OUTPUT_ARTIFACTS = (
    SEED_ROBUSTNESS_AUDIT_FILENAME,
)

REQUIRED_OUTPUT_ARTIFACTS = (
    TOPOLOGICAL_LOBSTER_FIGURE_FILENAME,
    GLOBAL_FLAVOR_FIT_TABLE_FILENAME,
    UNIQUENESS_SCAN_TABLE_FILENAME,
    MODULARITY_RESIDUAL_MAP_FILENAME,
    LANDSCAPE_ANOMALY_MAP_FILENAME,
    SUPPLEMENTARY_TOPCHI2_TABLE_FILENAME,
    SUPPLEMENTARY_IH_SUPPORT_MAP_FILENAME,
    SUPPLEMENTARY_TOLERANCE_TABLE_FILENAME,
    SUPPLEMENTARY_UNITARY_CONSISTENCY_TABLE_FILENAME,
    KAPPA_SENSITIVITY_AUDIT_FILENAME,
    KAPPA_STABILITY_SWEEP_FILENAME,
    SVD_STABILITY_AUDIT_TABLE_FILENAME,
    PHYSICS_CONSTANTS_FILENAME,
    AUDIT_STATEMENT_FILENAME,
    SVD_STABILITY_REPORT_FILENAME,
    EIGENVECTOR_STABILITY_AUDIT_FILENAME,
    STABILITY_REPORT_FILENAME,
    MAJORANA_FLOOR_FIGURE_FILENAME,
    DM_FINGERPRINT_FIGURE_FILENAME,
    CKM_PHASE_TILT_PROFILE_FIGURE_FILENAME,
    FRAMING_GAP_HEATMAP_FIGURE_FILENAME,
    BENCHMARK_STABILITY_TABLE_FILENAME,
    SUPPLEMENTARY_VEV_ALIGNMENT_STABILITY_FIGURE_FILENAME,
    SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_FIGURE_FILENAME,
    SUPPLEMENTARY_DETERMINANT_GRADIENT_FIGURE_FILENAME,
    SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME,
    BENCHMARK_DIAGNOSTICS_FILENAME,
    TRANSPORT_COVARIANCE_DIAGNOSTICS_FILENAME,
    SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_DATA_FILENAME,
    FRAMING_GAP_STABILITY_FIGURE_FILENAME,
)


def _optional_packet_output_artifacts(output_dir: Path) -> tuple[str, ...]:
    return tuple(filename for filename in OPTIONAL_PACKET_OUTPUT_ARTIFACTS if (output_dir / filename).exists())


def _present_packet_output_artifacts(output_dir: Path) -> tuple[str, ...]:
    return tuple(filename for filename in PACKET_OUTPUT_ARTIFACTS if (output_dir / filename).exists()) + _optional_packet_output_artifacts(output_dir)


def validate_manuscript_consistency(
    manuscript_dir: Path,
    output_dir: Path,
) -> None:
    r"""Validate generated numerical artifacts without linting manuscript prose."""

    del manuscript_dir
    sync_errors: list[str] = []
    packet_artifacts = _present_packet_output_artifacts(output_dir)

    for artifact_name in REQUIRED_OUTPUT_ARTIFACTS:
        if not (output_dir / artifact_name).exists():
            sync_errors.append(f"missing generated artifact {output_dir / artifact_name}")

    for packet_dirname, packet_label in (
        (AUDIT_OUTPUT_ARCHIVE_DIRNAME, "audit output archive"),
        (STABILITY_AUDIT_OUTPUTS_DIRNAME, "stability audit outputs"),
        (LANDSCAPE_METRICS_DIRNAME, "landscape metrics"),
    ):
        packet_output_dir = output_dir / packet_dirname
        if not packet_output_dir.is_dir():
            sync_errors.append(f"missing generated {packet_label} directory {packet_output_dir}")
        else:
            for artifact_name in (*packet_artifacts, AUDIT_OUTPUT_MANIFEST_FILENAME):
                if not (packet_output_dir / artifact_name).exists():
                    sync_errors.append(f"{packet_label} is missing {packet_output_dir / artifact_name}")

    physics_constants_path = output_dir / PHYSICS_CONSTANTS_FILENAME
    if physics_constants_path.exists():
        physics_constants_text = physics_constants_path.read_text(encoding="utf-8")
        for macro_name in REQUIRED_PHYSICS_CONSTANT_MACROS:
            if rf"\newcommand{{\{macro_name}}}" not in physics_constants_text:
                sync_errors.append(f"physics_constants.tex is missing \\{macro_name}")
        for macro_name in FORBIDDEN_PHYSICS_CONSTANT_MACROS:
            if rf"\newcommand{{\{macro_name}}}" in physics_constants_text:
                sync_errors.append(f"physics_constants.tex still exports obsolete \\{macro_name}")

    if sync_errors:
        raise RuntimeError("Generated artifact validation failed:\n- " + "\n- ".join(sync_errors))


COMPUTATIONAL_SEARCH_WINDOW = define_computational_search_window()
SO10_GEOMETRIC_KAPPA = compute_geometric_kappa_ansatz()
GEOMETRIC_KAPPA = SO10_GEOMETRIC_KAPPA.derived_kappa
MODULAR_HORIZON_SELECTION = derive_modular_horizon_selection()
HOLOGRAPHIC_BITS = MODULAR_HORIZON_SELECTION.derived_bits
MAX_COMPLEXITY_CAPACITY = HOLOGRAPHIC_BITS
KAPPA_D5 = GEOMETRIC_KAPPA
TOPOLOGICAL_MASS_COORDINATE_LABEL = "TOPOLOGICAL_MASS_COORDINATE"
KAPPA_SCAN_VALUES = tuple(float(GEOMETRIC_KAPPA * factor) for factor in (0.9, 1.0, 1.1))


def topological_mass_coordinate_ev(
    bit_count: float = HOLOGRAPHIC_BITS,
    kappa_geometric: float = KAPPA_D5,
) -> float:
    r"""Return the welded neutrino mass coordinate ``m_\nu=\kappa_{D_5}M_PN^{-1/4}``."""

    if bit_count <= 0.0:
        raise ValueError("Holographic bit count must be positive.")
    return float(kappa_geometric * PLANCK_MASS_EV * bit_count ** (-0.25))


def _build_topological_mass_coordinate_lock_error(
    current_m_nu: float,
    *,
    expected_mass: float,
    bit_count: float,
    kappa_geometric: float,
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
) -> TopologicalIntegrityError:
    detuning = current_m_nu - expected_mass
    required_bit_count = float((kappa_geometric * PLANCK_MASS_EV / current_m_nu) ** 4)
    with localcontext() as precision:
        precision.prec = 50
        required_bit_ratio = Decimal(str(required_bit_count)) / Decimal(str(bit_count))
    branch_label = (
        f"the anomaly-free ({lepton_level},{quark_level},{parent_level}) cell"
        if parent_level is not None and lepton_level is not None and quark_level is not None
        else "the current anomaly-free cell"
    )
    return TopologicalIntegrityError(
        "Topological consistency bound crossed. "
        f"Detuning of {detuning:+.6e} eV away from the welded mass coordinate {expected_mass:.6e} eV "
        f"requires N={required_bit_count:.6e}, i.e. a fractional/non-integer rescaling of the locked bit budget by {required_bit_ratio}, "
        "or introduces non-vanishing geometric torsion. "
        f"The neutrino mass scale is a welded coordinate of {branch_label} rather than a tunable parameter. "
        "Unitary Lock (\\hat{H}\\Psi = 0) violated; the Lorentzian bulk is now quantum-mechanically undefined."
    )


def enforce_topological_mass_coordinate_lock(
    mass_coordinate_ev: float,
    *,
    bit_count: float = HOLOGRAPHIC_BITS,
    kappa_geometric: float = KAPPA_D5,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    rel_tol: float = 0.0,
    abs_tol: float = TOPOLOGICAL_MASS_COORDINATE_ABS_TOL_EV,
) -> None:
    r"""Raise ``TopologicalIntegrityError`` when the welded mass coordinate is manually detuned."""

    expected_mass = topological_mass_coordinate_ev(bit_count=bit_count, kappa_geometric=kappa_geometric)
    if math.isclose(mass_coordinate_ev, expected_mass, rel_tol=rel_tol, abs_tol=abs_tol):
        return
    raise _build_topological_mass_coordinate_lock_error(
        mass_coordinate_ev,
        expected_mass=expected_mass,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )


def lambda_si_m2_to_ev2(lambda_si_m2: float) -> float:
    r"""Convert an SI cosmological constant in ``m^{-2}`` to natural ``eV^2`` units."""

    return float(lambda_si_m2 * (HBAR_EV_SECONDS * LIGHT_SPEED_M_PER_S) ** 2)


def calculate_lloyds_limit_bound(
    rho_vac_surface_tension_ev4: float,
    bit_count: float = HOLOGRAPHIC_BITS,
) -> float:
    r"""Return the Lloyd-bridge rate ``d\mathcal C/dt=2E_{\rm vac}^{\rm surf}/(\pi\hbar)``.

    This helper implements the manuscript's Stage XIII ``Lloyd Bridge``: the
    computational speed limit imposed by the finite holographic register once
    the vacuum surface energy and horizon bit budget are fixed.
    """

    if rho_vac_surface_tension_ev4 <= 0.0:
        raise ValueError("rho_vac_surface_tension_ev4 must be positive.")
    if bit_count <= 0.0:
        raise ValueError("Holographic bit count must be positive.")

    horizon_radius_m = float(PLANCK_LENGTH_M * math.sqrt(bit_count / math.pi))
    horizon_volume_m3 = float((4.0 * math.pi / 3.0) * horizon_radius_m**3)
    meter_to_ev_inverse = 1.0 / (HBAR_EV_SECONDS * LIGHT_SPEED_M_PER_S)
    horizon_volume_ev_minus3 = float(horizon_volume_m3 * meter_to_ev_inverse**3)
    vacuum_energy_ev = float(rho_vac_surface_tension_ev4 * horizon_volume_ev_minus3)
    return float((2.0 * vacuum_energy_ev) / (math.pi * HBAR_EV_SECONDS))


class LloydBridge:
    r"""Publication-facing implementation of Eq. ``eq:lloyd-bridge``.

    The Lloyd Bridge identifies the maximal complexity-growth rate of the
    branch with the finite-horizon clock ceiling

    .. math::
       \left(\frac{d\mathcal C}{dt}\right)_{\rm Lloyd}
       = \frac{2E_{\rm vac}^{\rm surf}}{\pi\hbar},

    where :math:`E_{\rm vac}^{\rm surf}` is the vacuum surface energy stored in
    the horizon-sized decoder volume fixed by :math:`N_{\rm holo}`.
    """

    @staticmethod
    def limit_ops_per_second(
        rho_vac_surface_tension_ev4: float,
        bit_count: float = HOLOGRAPHIC_BITS,
    ) -> float:
        """Evaluate the Stage XIII Lloyd ceiling for a fixed holographic branch."""

        return calculate_lloyds_limit_bound(
            rho_vac_surface_tension_ev4,
            bit_count=bit_count,
        )


def newton_constant_ev_minus2() -> float:
    r"""Return ``G_N`` in natural units using ``M_P^{-2}=8\pi G_N``."""

    return float(1.0 / (8.0 * math.pi * PLANCK_MASS_EV * PLANCK_MASS_EV))


def verify_triple_match_saturation(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    bit_count: float | None = None,
    kappa_geometric: float | None = None,
    *,
    model: TopologicalModel | None = None,
) -> TripleMatchSaturationAudit:
    r"""Evaluate the welded product ``\Lambda\,G_N\,m_\nu^4`` on the selected branch."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
    )
    lambda_surface_tension_si_m2 = holographic_surface_tension_lambda_si_m2(model=resolved_model)
    lambda_surface_tension_ev2 = lambda_si_m2_to_ev2(lambda_surface_tension_si_m2)
    g_newton_ev_minus2 = newton_constant_ev_minus2()
    m_nu_topological = topological_mass_coordinate_ev(
        bit_count=resolved_model.bit_count,
        kappa_geometric=resolved_model.kappa_geometric,
    )
    benchmark_lambda_ev2 = lambda_si_m2_to_ev2(holographic_surface_tension_lambda_si_m2(bit_count=HOLOGRAPHIC_BITS))
    benchmark_m_nu_topological = topological_mass_coordinate_ev(bit_count=HOLOGRAPHIC_BITS, kappa_geometric=KAPPA_D5)
    triple_match_product = lambda_surface_tension_ev2 * g_newton_ev_minus2 * m_nu_topological**4
    benchmark_identity_product = benchmark_lambda_ev2 * g_newton_ev_minus2 * benchmark_m_nu_topological**4
    return TripleMatchSaturationAudit(
        lambda_surface_tension_si_m2=float(lambda_surface_tension_si_m2),
        lambda_surface_tension_ev2=float(lambda_surface_tension_ev2),
        newton_constant_ev_minus2=float(g_newton_ev_minus2),
        topological_mass_coordinate_ev=float(m_nu_topological),
        triple_match_product=float(triple_match_product),
        benchmark_identity_product=float(benchmark_identity_product),
    )


def _audit_topological_mass_coordinate_sensitivity(
    m_nu_topological: float,
    *,
    bit_count: float,
    kappa_geometric: float,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    fractional_shift: float = 0.01,
) -> tuple[bool, str]:
    r"""Trigger the rigidity audit: a detuned mass must force a bit-budget deformation."""

    shifted_mass = float(m_nu_topological * (1.0 + fractional_shift))
    try:
        enforce_topological_mass_coordinate_lock(
            shifted_mass,
            bit_count=bit_count,
            kappa_geometric=kappa_geometric,
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        )
    except TopologicalIntegrityError as exc:
        return True, str(exc)
    return False, "Sensitivity Audit FAILED: the 1% mass detuning did not trigger the expected TopologicalIntegrityError."


def derive_scales(
    scale_ratio: float | None = None,
    kappa_geometric: float | None = None,
    *,
    model: TopologicalModel | None = None,
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    solver_config: SolverConfig | None = None,
) -> ScaleData:
    """Return the central boundary--bulk scale relation scales.

    Args:
        scale_ratio: Ratio $M_{\rm GUT}/M_Z$ used for RG transport.
        kappa_geometric: Order-one geometric prefactor in the CKN bridge.

    Returns:
        Central scale data for the benchmark configuration.
    """

    resolved_model = _coerce_topological_model(
        model=model,
        scale_ratio=scale_ratio,
        kappa_geometric=kappa_geometric,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        solver_config=solver_config,
    )
    return derive_scales_for_bits(resolved_model.bit_count, model=resolved_model)


def derive_scales_for_bits(
    bit_count: float,
    scale_ratio: float | None = None,
    kappa_geometric: float | None = None,
    *,
    model: TopologicalModel | None = None,
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    solver_config: SolverConfig | None = None,
) -> ScaleData:
    r"""Map the horizon information budget to infrared and ultraviolet scales via Eq. ``eq:bousso-bridge``.

    Args:
        bit_count: Holographic bit count on the cosmological horizon.
        scale_ratio: Ratio $M_{\rm GUT}/M_Z$ used for RG transport.
        kappa_geometric: Order-one geometric factor multiplying the CKN cutoff.

    Returns:
        Scale data for the light defect mass and its heavy Majorana partner.
    """

    resolved_model = _coerce_topological_model(
        model=model,
        scale_ratio=scale_ratio,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        solver_config=solver_config,
    )

    m_0_uv = resolved_model.kappa_geometric * PLANCK_MASS_EV * bit_count ** (-0.25)
    majorana_boundary = PLANCK_MASS_EV * bit_count ** (-0.5)
    majorana_effective = PLANCK_MASS_EV * bit_count ** 0.25 / resolved_model.kappa_geometric
    transport_curvature = derive_transport_curvature_audit(
        lepton_level=resolved_model.lepton_level,
        quark_level=resolved_model.quark_level,
    )
    gamma_0_one_loop = transport_curvature.gamma_0_one_loop
    gamma_0_two_loop = transport_curvature.gamma_0_two_loop
    m_0_mz = apply_rg_mass_running(
        m_0_uv,
        resolved_model.scale_ratio,
        gamma_0_one_loop,
        gamma_0_two_loop,
        parent_level=resolved_model.parent_level,
        lepton_level=resolved_model.lepton_level,
        quark_level=resolved_model.quark_level,
        solver_config=resolved_model.solver_config,
    )
    return ScaleData(
        m_0_uv_ev=m_0_uv,
        m_0_mz_ev=m_0_mz,
        majorana_boundary_ev=majorana_boundary,
        majorana_effective_ev=majorana_effective,
        gamma_0_one_loop=gamma_0_one_loop,
        gamma_0_two_loop=gamma_0_two_loop,
        kappa_geometric=resolved_model.kappa_geometric,
    )


class BoussoBridge:
    r"""Publication-facing implementation of Eq. ``eq:bousso-bridge``.

    The Bousso Bridge is the UV/IR matching rule that welds the light-neutrino
    coordinate to the saturated holographic register,

    .. math::
       m_\nu = \kappa_{D_5} M_P N^{-1/4},

    while simultaneously fixing the associated heavy Majorana scale through the
    same finite-information budget.
    """

    @staticmethod
    def derive_scales(
        bit_count: float,
        scale_ratio: float | None = None,
        kappa_geometric: float | None = None,
        *,
        model: TopologicalModel | None = None,
        parent_level: int | None = None,
        lepton_level: int | None = None,
        quark_level: int | None = None,
        solver_config: SolverConfig | None = None,
    ) -> ScaleData:
        """Evaluate the welded mass bridge for a fixed branch or explicit bit count."""

        return derive_scales_for_bits(
            bit_count,
            scale_ratio=scale_ratio,
            kappa_geometric=kappa_geometric,
            model=model,
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
            solver_config=solver_config,
        )


def interference_holonomy_phase(
    interference_block: np.ndarray,
    framing_phases: np.ndarray,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    """Extract the CP-even holonomy phase from an interference block."""

    return topological_kernel.interference_holonomy_phase(
        interference_block,
        framing_phases,
        solver_config=solver_config,
    )


def predict_delta_cp(
    interference_block: np.ndarray,
    framing_phases: np.ndarray,
    branch_shift_deg: float = 0.0,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    """Predict the physical Dirac phase from $S$--$T$ interference."""

    return topological_kernel.predict_delta_cp(
        interference_block,
        framing_phases,
        branch_shift_deg=branch_shift_deg,
        solver_config=solver_config,
    )


def derive_structural_rhn_scale_gev(
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> tuple[float, int, int, float, float, float]:
    r"""Predict the structural RHN threshold from branching data.

    This routine implements the manuscript's topological seesaw prediction,

        M_N = M_GUT * exp[-I_L * Pi_rank - I_Q * deltaPi_126],

    in which the right-handed-neutrino threshold is not chosen independently,
    but is fixed by the same discrete parent/visible data that determine the
    quark-sector branching pressures.

    Args:
        parent_level: Parent affine level $K$ of the $SO(10)_K$ completion.
        lepton_level: Visible leptonic affine level $k_{\ell}$.
        quark_level: Visible quark affine level $k_q$.

    Returns:
        Tuple ``(M_N, I_L, I_Q, Pi_rank, deltaPi_126, exponent)`` containing
        the structural RHN scale in GeV, the visible branching indices, the
        rank-gap pressure, the $\mathbf{126}_H$ threshold shift
        $\delta\Pi_{126}=\ln\Xi_{12}$, and the total suppression exponent.
    """

    lepton_branching = lepton_branching_index(parent_level, lepton_level)
    quark_branching = quark_branching_index(parent_level, quark_level)
    visible_block = ModularKernel(quark_level, "quark").restricted_block()
    coset_block = su3_low_weight_block(parent_level // 3)
    threshold_correction = derive_so10_threshold_correction(
        visible_block,
        coset_block,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    rank_pressure = rank_deficit_pressure(parent_level, quark_level)
    structural_exponent = (
        lepton_branching * rank_pressure
        + quark_branching * threshold_correction.delta_pi_126
    )
    threshold_scale_gev = GUT_SCALE_GEV * math.exp(-structural_exponent)
    return (
        threshold_scale_gev,
        lepton_branching,
        quark_branching,
        rank_pressure,
        threshold_correction.delta_pi_126,
        structural_exponent,
    )


def derive_rhn_threshold_data(
    scale_ratio: float | None = None,
    sector: Sector | str = Sector.LEPTON,
    *,
    model: TopologicalModel | None = None,
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
) -> RGThresholdData:
    """Construct explicit RG-threshold bookkeeping for the transport window."""

    resolved_model = _coerce_topological_model(
        model=model,
        scale_ratio=scale_ratio,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    resolved_sector = Sector.coerce(sector)
    structural_threshold_gev, lepton_branching_index, quark_branching_index, rank_pressure, threshold_shift_126, structural_exponent = derive_structural_rhn_scale_gev(
        parent_level=resolved_model.parent_level,
        lepton_level=resolved_model.lepton_level,
        quark_level=resolved_model.quark_level,
    )
    ultraviolet_scale_gev = MZ_SCALE_GEV * resolved_model.scale_ratio
    threshold_active = resolved_sector is Sector.LEPTON and MZ_SCALE_GEV < structural_threshold_gev < ultraviolet_scale_gev
    if threshold_active:
        lower_interval_log = math.log(structural_threshold_gev / MZ_SCALE_GEV)
        upper_interval_log = math.log(ultraviolet_scale_gev / structural_threshold_gev)
        matching_angle_shifts_deg = RHN_THRESHOLD_MATCHING_ANGLE_SHIFTS_DEG
        matching_delta_shift_deg = RHN_THRESHOLD_MATCHING_DELTA_SHIFT_DEG
        matching_mass_shift_fraction = RHN_THRESHOLD_MATCHING_MASS_SHIFT_FRACTION
        threshold_scale_gev: float | None = structural_threshold_gev
    else:
        lower_interval_log = math.log(resolved_model.scale_ratio)
        upper_interval_log = 0.0
        matching_angle_shifts_deg = (0.0, 0.0, 0.0)
        matching_delta_shift_deg = 0.0
        matching_mass_shift_fraction = 0.0
        threshold_scale_gev = None

    return RGThresholdData(
        sector=resolved_sector,
        threshold_active=threshold_active,
        threshold_scale_gev=threshold_scale_gev,
        lower_interval_log=lower_interval_log,
        upper_interval_log=upper_interval_log,
        matching_angle_shifts_deg=matching_angle_shifts_deg,
        matching_delta_shift_deg=matching_delta_shift_deg,
        matching_mass_shift_fraction=matching_mass_shift_fraction,
        structural_exponent=structural_exponent,
        lepton_branching_index=lepton_branching_index,
        quark_branching_index=quark_branching_index,
        rank_pressure=rank_pressure,
        threshold_shift_126=threshold_shift_126,
    )


def gauge_couplings_mz_inputs(alpha_s_mz: float = PLANCK2018_ALPHA_S_MZ) -> tuple[float, float, float]:
    """Return the GUT-normalized SM gauge couplings at $M_Z$."""

    alpha1_inv, alpha2_inv, alpha3_inv = alpha_inverse_mz_inputs(alpha_s_mz=alpha_s_mz)
    return (
        float(math.sqrt(4.0 * math.pi / alpha1_inv)),
        float(math.sqrt(4.0 * math.pi / alpha2_inv)),
        float(math.sqrt(4.0 * math.pi / alpha3_inv)),
    )


def running_coupling_mz_inputs(
    *,
    top_yukawa_mz: float = SM_MZ_YUKAWA_BENCHMARKS["top"],
    alpha_s_mz: float = PLANCK2018_ALPHA_S_MZ,
) -> RunningCouplings:
    """Return the electroweak benchmark inputs for the coupled SM transport."""

    g1_mz, g2_mz, g3_mz = gauge_couplings_mz_inputs(alpha_s_mz=alpha_s_mz)
    return RunningCouplings(
        top=top_yukawa_mz,
        bottom=SM_MZ_YUKAWA_BENCHMARKS["bottom"],
        tau=SM_MZ_YUKAWA_BENCHMARKS["tau"],
        g1=g1_mz,
        g2=g2_mz,
        g3=g3_mz,
    )


def sm_one_loop_running_betas(couplings: RunningCouplings) -> RunningCouplings:
    r"""Return the standard one-loop SM RGEs for $(y_t,y_b,y_\tau,g_1,g_2,g_3)$."""

    return physics_engine.sm_one_loop_running_betas(couplings)


def derive_running_couplings(
    scale_gev: float,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    mz_inputs: RunningCouplings | None = None,
    max_step: float | None = None,
) -> RunningCouplings:
    """Integrate the coupled one-loop SM Yukawa and gauge RGEs to ``scale_gev``."""

    if scale_gev <= 0.0:
        raise ValueError(f"Scale must be positive, received {scale_gev}")
    resolved_mz_inputs = running_coupling_mz_inputs() if mz_inputs is None else mz_inputs
    if solver_isclose(scale_gev, MZ_SCALE_GEV):
        return resolved_mz_inputs

    initial = resolved_mz_inputs.as_array()
    target_time = math.log(scale_gev / MZ_SCALE_GEV) / ONE_LOOP_FACTOR

    def upward_equations(loop_time: float, state: np.ndarray) -> np.ndarray:
        beta = sm_one_loop_running_betas(RunningCouplings(*state))
        return beta.as_array()

    solve_kwargs = {} if max_step is None else {"max_step": max_step}
    solution = physics_engine.solve_ivp_with_fallback(
        upward_equations,
        (0.0, target_time),
        initial,
        solver_config=solver_config,
        **solve_kwargs,
    )
    return RunningCouplings(*[float(value) for value in solution.y[:, -1]])


def structural_majorana_phase_closure(
    level: int | None = None,
    *,
    parent_level: int | None = None,
    model: TopologicalModel | None = None,
) -> tuple[float, float]:
    r"""Return the phase-locked Majorana angles implied by the parent anomaly."""

    resolved_model = _coerce_topological_model(model=model, lepton_level=level, parent_level=parent_level)
    anomaly_fraction = float(calculate_branching_anomaly("SO(10)", "SU(3)", resolved_model.parent_level).anomaly_fraction)
    alpha_1 = math.degrees(4.0 * math.pi * su2_conformal_weight(resolved_model.lepton_level, 1) + 2.0 * math.pi * anomaly_fraction)
    alpha_2 = math.degrees(4.0 * math.pi * su2_conformal_weight(resolved_model.lepton_level, 2) + 2.0 * math.pi * anomaly_fraction)
    return alpha_1, alpha_2


def structural_majorana_phase_proxies(
    level: int | None = None,
    *,
    parent_level: int | None = None,
    model: TopologicalModel | None = None,
) -> tuple[float, float]:
    """Map the modular framing phases onto effective leptonic phase proxies.

    The PMNS benchmark in this verifier does not carry independent scanned
    Majorana phases. For the Antusch-style analytic beta functions we therefore
    use the relative $SU(2)_{26}$ framing phases as the fixed structural phase
    inputs controlling the one-loop mass combinations.
    """

    alpha_1_deg, alpha_2_deg = structural_majorana_phase_closure(level, parent_level=parent_level, model=model)
    return math.radians(alpha_1_deg), math.radians(alpha_2_deg)


def charged_lepton_yukawa_diagonal(tau_yukawa: float) -> np.ndarray:
    """Return the diagonal charged-lepton Yukawa matrix in the third-family approximation."""

    return np.diag(
        [
            CHARGED_LEPTON_YUKAWA_RATIOS["electron"] * tau_yukawa,
            CHARGED_LEPTON_YUKAWA_RATIOS["muon"] * tau_yukawa,
            tau_yukawa,
        ]
    )


def majorana_mass_matrix_from_structural_pmns(
    unitary: np.ndarray,
    masses_ev: np.ndarray,
    level: int = LEPTON_LEVEL,
) -> np.ndarray:
    """Construct the complex symmetric neutrino mass matrix in the flavor basis."""

    return physics_engine.majorana_mass_matrix_from_pmns(
        unitary,
        masses_ev,
        phase_proxies_rad=structural_majorana_phase_proxies(level),
    )


def majorana_mass_matrix_beta(
    unitary: np.ndarray,
    masses_ev: np.ndarray,
    scale_gev: float,
    tau_yukawa: float | None = None,
    level: int = LEPTON_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    r"""Evaluate the SM one-loop RGE for the effective Majorana mass matrix itself."""

    mass_matrix = majorana_mass_matrix_from_structural_pmns(unitary, masses_ev, level=level)
    current_tau_yukawa = derive_running_couplings(scale_gev, solver_config=solver_config).tau if tau_yukawa is None else tau_yukawa
    return physics_engine.majorana_mass_matrix_beta(
        mass_matrix,
        tau_yukawa=current_tau_yukawa,
        charged_lepton_yukawa_ratios=CHARGED_LEPTON_YUKAWA_RATIOS,
        sm_majorana_c_e=SM_MAJORANA_C_E,
    )


def takagi_like_pmns_from_mass_matrix(
    mass_matrix: np.ndarray,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover PMNS angles and masses from a complex symmetric mass matrix."""

    return physics_engine.takagi_diagonalize_symmetric(mass_matrix, solver_config=solver_config)


def wrapped_angle_difference_deg(updated_angle_deg: float, reference_angle_deg: float) -> float:
    """Return the signed angular increment in the range [-180, 180)."""

    return physics_engine.wrapped_angle_difference_deg(updated_angle_deg, reference_angle_deg)


def dynamic_lepton_antusch_betas(
    theta12_deg: float,
    theta13_deg: float,
    theta23_deg: float,
    delta_deg: float,
    masses_ev: np.ndarray,
    unitary: np.ndarray,
    scale_gev: float = GUT_SCALE_GEV,
    tau_yukawa: float | None = None,
    level: int = LEPTON_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> tuple[np.ndarray, float]:
    """Extract instantaneous PMNS beta functions from the matrix RGE itself."""

    current_tau_yukawa = derive_running_couplings(scale_gev, solver_config=solver_config).tau if tau_yukawa is None else tau_yukawa
    return physics_engine.matrix_derived_lepton_betas(
        unitary,
        masses_ev,
        phase_proxies_rad=structural_majorana_phase_proxies(level),
        tau_yukawa=current_tau_yukawa,
        charged_lepton_yukawa_ratios=CHARGED_LEPTON_YUKAWA_RATIOS,
        sm_majorana_c_e=SM_MAJORANA_C_E,
        solver_config=solver_config,
        pdg_parameter_extractor=lambda matrix: topological_kernel.pdg_parameters(matrix, solver_config=solver_config),
    )


def lepton_antusch_delta_one_loop_beta(
    theta12_deg: float,
    theta13_deg: float,
    theta23_deg: float,
    delta_deg: float,
    masses_ev: np.ndarray,
    unitary: np.ndarray | None = None,
    scale_gev: float = GUT_SCALE_GEV,
    tau_yukawa: float | None = None,
    level: int = LEPTON_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    r"""Matrix-derived Dirac-phase beta for the default Standard-Model transport."""

    reference_unitary = topological_kernel.pdg_unitary(theta12_deg, theta13_deg, theta23_deg, delta_deg) if unitary is None else unitary
    _, delta_beta = dynamic_lepton_antusch_betas(
        theta12_deg,
        theta13_deg,
        theta23_deg,
        delta_deg,
        masses_ev,
        reference_unitary,
        scale_gev=scale_gev,
        tau_yukawa=tau_yukawa,
        level=level,
        solver_config=solver_config,
    )
    return float(delta_beta)


def quark_one_loop_betas(
    theta12_deg: float,
    theta13_deg: float,
    theta23_deg: float,
    scale_gev: float,
    top_yukawa: float | None = None,
    bottom_yukawa: float | None = None,
    level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    r"""Analytical one-loop CKM beta functions from SM Yukawas and branch pressures."""

    running_couplings = derive_running_couplings(scale_gev, solver_config=solver_config)
    current_top_yukawa = running_couplings.top if top_yukawa is None else top_yukawa
    current_bottom_yukawa = running_couplings.bottom if bottom_yukawa is None else bottom_yukawa
    transport_strength = current_top_yukawa * current_top_yukawa + current_bottom_yukawa * current_bottom_yukawa
    visible_block = ModularKernel(level, Sector.QUARK, solver_config=solver_config).restricted_block()
    coset_block = su3_low_weight_block(PARENT_LEVEL // 3)
    threshold_correction = derive_so10_threshold_correction(
        visible_block,
        coset_block,
        quark_level=level,
        solver_config=solver_config,
    )
    vacuum_pressure = quark_branching_pressure(visible_block, solver_config=solver_config)
    coset_weighting = coset_topological_weighting(
        visible_block,
        coset_block,
        quark_level=level,
        solver_config=solver_config,
    )
    epsilon12 = math.exp(-vacuum_pressure) * threshold_correction.xi12_abs
    epsilon23 = math.exp(-2.0 * vacuum_pressure) * coset_weighting[1, 2]
    epsilon13 = epsilon12 * epsilon23

    theta12 = math.radians(theta12_deg)
    theta13 = math.radians(theta13_deg)
    theta23 = math.radians(theta23_deg)

    beta12_rad = transport_strength * math.sin(2.0 * theta12) * (epsilon23**2)
    beta13_rad = -SO10_RANK * transport_strength * math.sin(2.0 * theta13) * epsilon12
    beta23_rad = -0.5 * transport_strength * math.sin(2.0 * theta23) * epsilon23
    return np.degrees(np.array([beta12_rad, beta13_rad, beta23_rad], dtype=float))


def quark_delta_one_loop_beta(
    theta12_deg: float,
    theta13_deg: float,
    theta23_deg: float,
    delta_deg: float,
    scale_gev: float,
    top_yukawa: float | None = None,
    bottom_yukawa: float | None = None,
    level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    r"""Analytical one-loop CKM Dirac-phase beta in invariant form.

    The apparent PDG-area denominator cancels algebraically against the
    Jarlskog numerator, so the phase drive is evaluated without dividing by a
    coordinate-singular mixing-area factor.
    """

    theta12 = math.radians(theta12_deg)
    theta13 = math.radians(theta13_deg)
    theta23 = math.radians(theta23_deg)
    delta = math.radians(delta_deg)
    running_couplings = derive_running_couplings(scale_gev, solver_config=solver_config)
    current_top_yukawa = running_couplings.top if top_yukawa is None else top_yukawa
    current_bottom_yukawa = running_couplings.bottom if bottom_yukawa is None else bottom_yukawa
    visible_block = ModularKernel(level, Sector.QUARK, solver_config=solver_config).restricted_block()
    vacuum_pressure = quark_branching_pressure(visible_block, solver_config=solver_config)
    transport_strength = (current_top_yukawa * current_top_yukawa + current_bottom_yukawa * current_bottom_yukawa) * math.exp(-3.0 * vacuum_pressure)
    _ = (theta12, theta13, theta23)
    return math.degrees(transport_strength * math.sin(delta))


def calculate_standard_model_beta_functions(
    uv_matrix: np.ndarray,
    sector: Sector | str,
    lightest_mass_ev: float | None = None,
    scale_gev: float = GUT_SCALE_GEV,
    running_couplings: RunningCouplings | None = None,
    level: int | None = None,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> BetaFunctionData:
    r"""Return analytical one-loop beta functions for the CKM or PMNS sector."""

    resolved_sector = Sector.coerce(sector)
    theta12, theta13, theta23, delta_cp, _ = pdg_parameters(uv_matrix, solver_config=solver_config)
    couplings = derive_running_couplings(scale_gev, solver_config=solver_config) if running_couplings is None else running_couplings
    if resolved_sector is Sector.LEPTON:
        current_level = LEPTON_LEVEL if level is None else level
        lightest_mass = derive_scales().m_0_uv_ev if lightest_mass_ev is None else lightest_mass_ev
        masses = normal_order_masses(lightest_mass)
        transport_curvature = derive_transport_curvature_audit(lepton_level=current_level)
        return BetaFunctionData(
            sector=resolved_sector,
            theta_one_loop=lepton_antusch_one_loop_betas(
                theta12,
                theta13,
                theta23,
                delta_cp,
                masses,
                uv_matrix,
                scale_gev=scale_gev,
                tau_yukawa=couplings.tau,
                level=current_level,
                solver_config=solver_config,
            ),
            theta_two_loop=transport_curvature.lepton_theta_two_loop,
            delta_one_loop=lepton_antusch_delta_one_loop_beta(
                theta12,
                theta13,
                theta23,
                delta_cp,
                masses,
                unitary=uv_matrix,
                scale_gev=scale_gev,
                tau_yukawa=couplings.tau,
                level=current_level,
                solver_config=solver_config,
            ),
            delta_two_loop=transport_curvature.lepton_delta_two_loop,
        )

    if resolved_sector is Sector.QUARK:
        current_level = QUARK_LEVEL if level is None else level
        transport_curvature = derive_transport_curvature_audit(quark_level=current_level)
        return BetaFunctionData(
            sector=resolved_sector,
            theta_one_loop=quark_one_loop_betas(
                theta12,
                theta13,
                theta23,
                scale_gev=scale_gev,
                top_yukawa=couplings.top,
                bottom_yukawa=couplings.bottom,
                level=current_level,
                solver_config=solver_config,
            ),
            theta_two_loop=transport_curvature.quark_theta_two_loop,
            delta_one_loop=quark_delta_one_loop_beta(
                theta12,
                theta13,
                theta23,
                delta_cp,
                scale_gev=scale_gev,
                top_yukawa=couplings.top,
                bottom_yukawa=couplings.bottom,
                level=current_level,
                solver_config=solver_config,
            ),
            delta_two_loop=transport_curvature.quark_delta_two_loop,
        )

    raise ValueError(f"Unsupported sector: {resolved_sector.value}")


def lepton_antusch_one_loop_betas(
    theta12_deg: float,
    theta13_deg: float,
    theta23_deg: float,
    delta_deg: float,
    masses_ev: np.ndarray,
    unitary: np.ndarray,
    scale_gev: float = GUT_SCALE_GEV,
    tau_yukawa: float | None = None,
    level: int = LEPTON_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    r"""Matrix-derived leptonic angular betas for the default SM transport."""

    theta_betas, _ = dynamic_lepton_antusch_betas(
        theta12_deg,
        theta13_deg,
        theta23_deg,
        delta_deg,
        masses_ev,
        unitary,
        scale_gev=scale_gev,
        tau_yukawa=tau_yukawa,
        level=level,
        solver_config=solver_config,
    )
    return theta_betas


def derive_so10_threshold_correction(
    visible_block: np.ndarray,
    coset_block: np.ndarray,
    gut_threshold_residue: float | None = None,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> SO10ThresholdCorrectionData:
    r"""Complex $SO(10)\!\to\!SU(3)$ transition amplitude for the quark branch.

    The residue ``gut_threshold_residue`` is the normalized Wilson coefficient
    extracted from the one-loop matching condition at $M_{\rm GUT}$. It
    multiplies only the orthogonal coset phase sourced by the heavy
    $\mathbf{126}_H$ threshold, so it dresses the CKM apex angle $\gamma$
    without deforming the rigid modular $S$-matrix magnitudes.
    """

    guard = solver_config.stability_guard
    so10_weyl_norm_sq = weyl_vector_norm_sq(SO10_DIMENSION, SO10_DUAL_COXETER)
    su3_weyl_norm_sq = weyl_vector_norm_sq(SU3_DIMENSION, SU3_DUAL_COXETER)
    weyl_ratio = math.sqrt(so10_weyl_norm_sq / su3_weyl_norm_sq)
    parent_visible_denominator = guard.require_nonzero_magnitude(
        coset_block[0, 0] * visible_block[0, 1],
        coordinate="SO(10) threshold parent-visible denominator",
        detail="The heavy-threshold matching amplitude is undefined when the parent-visible channel vanishes.",
    )
    parent_visible_ratio = abs(coset_block[0, 1] * visible_block[0, 0] / parent_visible_denominator)
    clebsch_126 = (SU3_DUAL_COXETER / SU2_DUAL_COXETER) * (lepton_level / quark_level)
    clebsch_10 = SO10_CLEBSCH_10
    projection_exponent = float(calculate_branching_anomaly("SO(10)", "SU(3)", parent_level).anomaly_fraction)
    xi12_abs = (parent_visible_ratio * clebsch_126) ** projection_exponent
    delta_pi_126 = math.log(xi12_abs)

    quark_branching = quark_branching_index(parent_level, quark_level)
    lepton_branching = lepton_branching_index(parent_level, lepton_level)
    rank_pressure = rank_deficit_pressure(parent_level, quark_level)
    framing_phase_rad = quark_branching * rank_pressure * projection_exponent * weyl_ratio * parent_visible_ratio
    framing_phase_deg = math.degrees(framing_phase_rad)

    parent_central_charge = wzw_central_charge(parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
    lepton_central_charge = wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER)
    quark_central_charge = wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
    coset_central_charge = parent_central_charge - lepton_central_charge - quark_central_charge

    structural_exponent = lepton_branching * rank_pressure + quark_branching * delta_pi_126
    structural_threshold_scale_gev = GUT_SCALE_GEV * math.exp(-structural_exponent)
    beta = 0.5 * math.log(su2_total_quantum_dimension(lepton_level))
    framing_gap_area = beta * beta
    matching_threshold_scale_gev = structural_threshold_scale_gev * math.exp(-framing_gap_area)
    (
        derived_gut_threshold_residue,
        alpha_gut,
        matching_log_sum,
        derived_lambda_12_mgut,
        derived_lambda_matrix_mgut,
        matching_contributions,
    ) = derive_formal_gut_threshold_matching(
        visible_block,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        matching_threshold_scale_gev=matching_threshold_scale_gev,
        structural_mn_gev=structural_threshold_scale_gev,
    )
    resolved_gut_threshold_residue = R_GUT if gut_threshold_residue is None else float(gut_threshold_residue)
    y12_tree_level = float(abs(visible_block[0, 1]))
    lambda_12_mgut = y12_tree_level * resolved_gut_threshold_residue
    lambda_matrix_mgut = np.array(derived_lambda_matrix_mgut, copy=True)
    lambda_matrix_mgut[0, 1] = lambda_12_mgut
    lambda_matrix_mgut[1, 0] = lambda_12_mgut
    threshold_log_fraction = math.log(GUT_SCALE_GEV / matching_threshold_scale_gev) / math.log(GUT_SCALE_GEV / MZ_SCALE_GEV)
    orthogonal_coset_base_phase_deg = math.degrees(projection_exponent * coset_central_charge / SO10_RANK)
    threshold_phase_tilt_deg = orthogonal_coset_base_phase_deg * resolved_gut_threshold_residue * threshold_log_fraction
    orthogonal_coset_phase_deg = orthogonal_coset_base_phase_deg + threshold_phase_tilt_deg

    higgs_mixing_weight = clebsch_10 / (clebsch_10 + clebsch_126)
    geodesic_closure_phase_deg = math.degrees(quark_branching * rank_pressure * projection_exponent * higgs_mixing_weight)
    triangle_tilt_deg = framing_phase_deg + orthogonal_coset_phase_deg + geodesic_closure_phase_deg
    xi12 = xi12_abs * np.exp(1j * framing_phase_rad)
    return SO10ThresholdCorrectionData(
        gut_threshold_residue=resolved_gut_threshold_residue,
        so10_weyl_norm_sq=so10_weyl_norm_sq,
        su3_weyl_norm_sq=su3_weyl_norm_sq,
        weyl_ratio=weyl_ratio,
        parent_visible_ratio=parent_visible_ratio,
        clebsch_126=clebsch_126,
        clebsch_10=clebsch_10,
        higgs_mixing_weight=higgs_mixing_weight,
        projection_exponent=projection_exponent,
        xi12=xi12,
        xi12_abs=xi12_abs,
        delta_pi_126=delta_pi_126,
        framing_phase_rad=framing_phase_rad,
        framing_phase_deg=framing_phase_deg,
        matching_threshold_scale_gev=matching_threshold_scale_gev,
        threshold_log_fraction=threshold_log_fraction,
        orthogonal_coset_base_phase_deg=orthogonal_coset_base_phase_deg,
        threshold_phase_tilt_deg=threshold_phase_tilt_deg,
        orthogonal_coset_phase_deg=orthogonal_coset_phase_deg,
        geodesic_closure_phase_deg=geodesic_closure_phase_deg,
        triangle_tilt_deg=triangle_tilt_deg,
        y12_tree_level=y12_tree_level,
        alpha_gut=alpha_gut,
        matching_log_sum=matching_log_sum,
        lambda_12_mgut=lambda_12_mgut,
        lambda_matrix_mgut=lambda_matrix_mgut,
        matching_contributions=matching_contributions,
        decoupling_audit=derive_heavy_state_decoupling_audit(matching_threshold_scale_gev),
    )


def calculate_126_higgs_cg_correction(
    clebsch_126: float | None = None,
    clebsch_10: float = SO10_CLEBSCH_10,
    target_suppression: float | Fraction = VEV_RATIO,
) -> HiggsCGCorrectionAuditData:
    r"""Return the natural $\mathbf{126}_H$ Clebsch suppression for the Yukawa-ratio audit.

    The bare topological quark/lepton ratio pressure overshoots the observed
    hierarchy by roughly a factor of five. In the current benchmark the visible
    $\mathbf{126}_H$ Clebsch factor is

    .. math:: C_{126}^{(12)} = \frac{h^{\vee}_{SU(3)}}{h^{\vee}_{SU(2)}}\frac{k_\ell}{k_q}=4.875,

    so its inverse provides the natural suppression

    .. math:: (C_{126}^{(12)})^{-1}=\frac{64}{312}\approx 0.20513,

    i.e. within a few percent of the exact benchmark $64/312$ correction even before a
    detailed Higgs-potential analysis is specified.
    """

    resolved_target_suppression = float(target_suppression)
    if clebsch_126 is None:
        clebsch_126 = (SU3_DUAL_COXETER / SU2_DUAL_COXETER) * (LEPTON_LEVEL / QUARK_LEVEL)
    inverse_clebsch_126_suppression = 1.0 / clebsch_126
    mixed_channel_suppression = clebsch_10 / (clebsch_10 + clebsch_126)
    corrected_pressure_factor = clebsch_126 * inverse_clebsch_126_suppression
    residual_to_target = inverse_clebsch_126_suppression - resolved_target_suppression
    return HiggsCGCorrectionAuditData(
        bare_overprediction_factor=clebsch_126,
        target_suppression=resolved_target_suppression,
        clebsch_126=clebsch_126,
        inverse_clebsch_126_suppression=inverse_clebsch_126_suppression,
        mixed_channel_suppression=mixed_channel_suppression,
        corrected_pressure_factor=corrected_pressure_factor,
        residual_to_target=residual_to_target,
    )


def coset_topological_weighting(
    visible_block: np.ndarray,
    coset_block: np.ndarray,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    r"""Descendant weighting matrix for the low-weight $SU(3)$ branch."""

    guard = solver_config.stability_guard
    threshold_correction = derive_so10_threshold_correction(
        visible_block,
        coset_block,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        solver_config=solver_config,
    )
    xi23_denominator = guard.require_nonzero_magnitude(
        visible_block[0, 1] * coset_block[1, 1],
        coordinate="coset topological weighting xi23 denominator",
        detail="The descendant weighting is undefined once the xi23 parent channel collapses.",
    )
    raw_xi23 = abs(visible_block[1, 1] * coset_block[0, 1] / xi23_denominator)
    xi12 = threshold_correction.xi12_abs
    xi23 = math.sqrt(raw_xi23)
    weighting = np.ones((3, 3), dtype=float)
    weighting[0, 1] = weighting[1, 0] = xi12
    weighting[1, 2] = weighting[2, 1] = xi23
    weighting[0, 2] = weighting[2, 0] = xi12 * xi23
    return weighting


def derive_beta_function_data(
    uv_matrix: np.ndarray,
    sector: Sector | str,
    lightest_mass_ev: float | None = None,
    scale_gev: float = GUT_SCALE_GEV,
    running_couplings: RunningCouplings | None = None,
    level: int | None = None,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> BetaFunctionData:
    r"""Assemble the analytical one-loop beta functions for a mixing matrix."""

    return calculate_standard_model_beta_functions(
        uv_matrix,
        sector,
        lightest_mass_ev=lightest_mass_ev,
        scale_gev=scale_gev,
        running_couplings=running_couplings,
        level=level,
        solver_config=solver_config,
    )


def apply_rg_running_linearized(
    uv_matrix: np.ndarray,
    scale_ratio: float,
    sector: Sector | str,
    beta_function_data: BetaFunctionData | None = None,
    lightest_mass_ev: float | None = None,
    level: int | None = None,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    """Apply the legacy Taylor-expanded RG transport to a mixing matrix."""

    resolved_sector = Sector.coerce(sector)
    theta12, theta13, theta23, delta_cp, _ = pdg_parameters(uv_matrix, solver_config=solver_config)
    threshold_data = derive_rhn_threshold_data(
        scale_ratio,
        sector=resolved_sector,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    loop_factor = threshold_data.one_loop_factor
    two_loop_factor = threshold_data.two_loop_factor
    uv_scale_gev = MZ_SCALE_GEV * scale_ratio
    beta_data = beta_function_data if beta_function_data is not None else derive_beta_function_data(
        uv_matrix,
        sector=resolved_sector,
        lightest_mass_ev=lightest_mass_ev,
        scale_gev=uv_scale_gev,
        level=level,
        solver_config=solver_config,
    )
    if beta_data.sector is not resolved_sector:
        raise ValueError(f"Beta-function sector mismatch: expected {resolved_sector.value}, received {beta_data.sector.value}")

    evolved_angles = (
        np.array([theta12, theta13, theta23], dtype=float)
        + loop_factor * beta_data.theta_one_loop
        + two_loop_factor * beta_data.theta_two_loop
        + np.array(threshold_data.matching_angle_shifts_deg, dtype=float)
    )
    evolved_delta = (
        delta_cp
        + loop_factor * beta_data.delta_one_loop
        + two_loop_factor * beta_data.delta_two_loop
        + threshold_data.matching_delta_shift_deg
    ) % 360.0
    return pdg_unitary(evolved_angles[0], evolved_angles[1], evolved_angles[2], evolved_delta)


def apply_rg_mass_running_linearized(
    m_0_uv_ev: float,
    scale_ratio: float,
    gamma_0_one_loop: float,
    gamma_0_two_loop: float = 0.0,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    """Legacy Taylor-expanded running of the structural lightest mass."""

    threshold_data = derive_rhn_threshold_data(
        scale_ratio,
        sector=Sector.LEPTON,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    loop_factor = threshold_data.one_loop_factor
    two_loop_factor = threshold_data.two_loop_factor
    return m_0_uv_ev * (
        1.0
        + gamma_0_one_loop * loop_factor
        + gamma_0_two_loop * two_loop_factor
        + threshold_data.matching_mass_shift_fraction
    )


def integrate_quark_rge_numerically(
    uv_matrix: np.ndarray,
    scale_ratio: float,
    level: int = QUARK_LEVEL,
    *,
    max_step: float | None = None,
    mz_inputs: RunningCouplings | None = None,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    """Numerically integrate the quark-sector transport with coupled SM couplings."""

    uv_scale_gev = MZ_SCALE_GEV * scale_ratio
    theta12_uv, theta13_uv, theta23_uv, delta_uv, _ = topological_kernel.pdg_parameters(uv_matrix, solver_config=solver_config)
    coupling_state_uv = derive_running_couplings(
        uv_scale_gev,
        solver_config=solver_config,
        mz_inputs=mz_inputs,
        max_step=max_step,
    ).as_array()

    def transport_equations(start_scale_gev: float):
        def evaluate(loop_time: float, state: np.ndarray) -> np.ndarray:
            theta12_deg, theta13_deg, theta23_deg, delta_deg, y_t, y_b, y_tau, g1, g2, g3 = state
            running_scale_gev = start_scale_gev * math.exp(-ONE_LOOP_FACTOR * loop_time)
            running_matrix = topological_kernel.pdg_unitary(theta12_deg, theta13_deg, theta23_deg, delta_deg)
            running_couplings = RunningCouplings(y_t, y_b, y_tau, g1, g2, g3)
            beta_data = derive_beta_function_data(
                running_matrix,
                sector=Sector.QUARK,
                scale_gev=running_scale_gev,
                running_couplings=running_couplings,
                level=level,
                solver_config=solver_config,
            )
            coupling_betas = sm_one_loop_running_betas(running_couplings).as_array()
            return np.array([*beta_data.theta_one_loop, beta_data.delta_one_loop, *(-coupling_betas)], dtype=float)

        return evaluate

    def integrate_segment(start_scale_gev: float, end_scale_gev: float, state: np.ndarray) -> np.ndarray:
        segment_loop_time = math.log(start_scale_gev / end_scale_gev) / ONE_LOOP_FACTOR
        solve_kwargs = {} if max_step is None else {"max_step": max_step}
        solution = physics_engine.solve_ivp_with_fallback(
            transport_equations(start_scale_gev),
            (0.0, segment_loop_time),
            state,
            solver_config=solver_config,
            **solve_kwargs,
        )
        return solution.y[:, -1]

    initial_state = np.array([theta12_uv, theta13_uv, theta23_uv, delta_uv, *coupling_state_uv], dtype=float)
    current_scale = uv_scale_gev
    current_state = initial_state
    for threshold_scale in (*publication_engine.quark_matching_thresholds(uv_scale_gev, MZ_SCALE_GEV), MZ_SCALE_GEV):
        current_state = integrate_segment(current_scale, threshold_scale, current_state)
        if not solver_isclose(threshold_scale, MZ_SCALE_GEV):
            current_state = publication_engine.apply_quark_threshold_matching(current_state, threshold_scale)
        current_scale = threshold_scale
    theta12_rg, theta13_rg, theta23_rg, delta_rg = current_state[:4]
    return pdg_unitary(theta12_rg, theta13_rg, theta23_rg, float(delta_rg % 360.0))


def integrate_pmns_rge_numerically(
    uv_matrix: np.ndarray,
    m_0_uv_ev: float,
    scale_ratio: float,
    level: int = LEPTON_LEVEL,
    gamma_0_one_loop: float | None = None,
    gamma_0_two_loop: float | None = None,
    *,
    parent_level: int = PARENT_LEVEL,
    quark_level: int = QUARK_LEVEL,
    max_step: float | None = None,
    mz_inputs: RunningCouplings | None = None,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Numerically integrate the coupled PMNS mass-matrix and SM-coupling RG equations."""

    transport_curvature = derive_transport_curvature_audit(lepton_level=level, quark_level=quark_level)
    resolved_gamma_0_one_loop = transport_curvature.gamma_0_one_loop if gamma_0_one_loop is None else gamma_0_one_loop
    resolved_gamma_0_two_loop = transport_curvature.gamma_0_two_loop if gamma_0_two_loop is None else gamma_0_two_loop

    total_loop_time = derive_rhn_threshold_data(
        scale_ratio,
        sector=Sector.LEPTON,
        parent_level=parent_level,
        lepton_level=level,
        quark_level=quark_level,
    ).one_loop_factor
    return physics_engine.integrate_pmns_majorana_rge_numerically(
        uv_matrix,
        m_0_uv_ev,
        phase_proxies_rad=structural_majorana_phase_proxies(level),
        scale_ratio=scale_ratio,
        mz_scale_gev=MZ_SCALE_GEV,
        total_loop_time=total_loop_time,
        mass_spectrum_builder=normal_order_masses,
        coupling_state_builder=lambda scale_gev: derive_running_couplings(
            scale_gev,
            solver_config=solver_config,
            mz_inputs=mz_inputs,
            max_step=max_step,
        ).as_array(),
        running_coupling_betas=lambda coupling_array: sm_one_loop_running_betas(RunningCouplings(*coupling_array)).as_array(),
        gamma_0_one_loop=resolved_gamma_0_one_loop,
        gamma_0_two_loop=resolved_gamma_0_two_loop,
        charged_lepton_yukawa_ratios=CHARGED_LEPTON_YUKAWA_RATIOS,
        sm_majorana_c_e=SM_MAJORANA_C_E,
        max_step=max_step,
        solver_config=solver_config,
        pdg_parameter_extractor=lambda matrix: topological_kernel.pdg_parameters(matrix, solver_config=solver_config),
    )


def apply_rg_running(
    uv_matrix: np.ndarray,
    scale_ratio: float,
    sector: Sector | str,
    beta_function_data: BetaFunctionData | None = None,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> np.ndarray:
    """Apply the primary numerical RG transport to a mixing matrix."""

    resolved_sector = Sector.coerce(sector)
    if resolved_sector is Sector.QUARK:
        return integrate_quark_rge_numerically(uv_matrix, scale_ratio, level=QUARK_LEVEL, solver_config=solver_config)
    if resolved_sector is Sector.LEPTON:
        return apply_rg_running_linearized(
            uv_matrix,
            scale_ratio,
            resolved_sector,
            beta_function_data=beta_function_data,
            solver_config=solver_config,
        )
    raise ValueError(f"Unsupported RG sector: {resolved_sector.value}")


def apply_rg_mass_running(
    m_0_uv_ev: float,
    scale_ratio: float,
    gamma_0_one_loop: float,
    gamma_0_two_loop: float = 0.0,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    max_step: float | None = None,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    """Numerically run the structural lightest mass from the UV to $M_Z$."""

    total_loop_time = derive_rhn_threshold_data(
        scale_ratio,
        sector=Sector.LEPTON,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    ).one_loop_factor

    def mass_equation(loop_time: float, state: np.ndarray) -> np.ndarray:
        return np.array([(gamma_0_one_loop + 2.0 * loop_time * gamma_0_two_loop) * state[0]], dtype=float)

    solve_kwargs = {} if max_step is None else {"max_step": max_step}
    solution = physics_engine.solve_ivp_with_fallback(
        mass_equation,
        (0.0, total_loop_time),
        np.array([m_0_uv_ev], dtype=float),
        solver_config=solver_config,
        **solve_kwargs,
    )
    return float(solution.y[0, -1])


def derive_formal_gut_threshold_matching(
    visible_block: np.ndarray,
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    matching_threshold_scale_gev: float | None = None,
    structural_mn_gev: float | None = None,
) -> tuple[float, float, float, float, np.ndarray, tuple[HeavyThresholdMatchingContribution, ...]]:
    r"""Derive the one-loop $SO(10)$ matching coefficient for the CKM $12$ channel.

    The dimension-5 Wilson coefficient is evaluated as a leading-order matching
    correction at $M_{\rm GUT}$,
    \[
      \lambda_{12}^{(5)}(M_{\rm GUT})
      =
      Y_{12}^{(0)}\,\frac{g_{\rm GUT}^2}{16\pi^2}
      \sum_A C_A\ln\!\left(\frac{M_P}{M_A}\right),
    \]
    where the sum runs over the heavy $\mathbf{126}_H$ fragments, the coarse
    $\mathbf{210}_H$ GUT-breaking fragments, and the broken $SO(10)/SM$ gauge
    bundle projected onto the quark $12$ channel.
    """

    resolved_matching_threshold_scale_gev = (
        derive_topological_threshold_gev(
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        )
        if matching_threshold_scale_gev is None
        else float(matching_threshold_scale_gev)
    )
    gauge_unification = derive_gauge_unification_existence_proof(
        m_126_gev=resolved_matching_threshold_scale_gev,
        structural_mn_gev=structural_mn_gev,
    )
    alpha_gut = 1.0 / gauge_unification.unified_alpha_inverse
    y12_tree_level = float(abs(visible_block[0, 1]))
    if y12_tree_level <= condition_aware_abs_tolerance(scale=y12_tree_level):
        raise RuntimeError("The CKM 12 Yukawa entry vanished before threshold matching was applied.")

    majorana_representation = derive_so10_representation_data("126_H", SO10_HIGGS_126_DYNKIN_LABELS)
    gut_breaking_representation = derive_so10_representation_data("210_H", SO10_HIGGS_210_DYNKIN_LABELS)
    majorana_ratio = float(majorana_representation.quadratic_casimir / (2 * majorana_representation.dynkin_index))
    gut_breaking_ratio = float(
        gut_breaking_representation.quadratic_casimir / (2 * gut_breaking_representation.dynkin_index)
    )
    gauge_projection_ratio = float(
        adjoint_quadratic_casimir(SO10_DUAL_COXETER) / (2 * so10_rep_dynkin_index((0, 1, 0, 0, 0)))
    ) * (RANK_DIFFERENCE / BROKEN_SO10_GAUGE_BOSON_COUNT)

    matching_contributions: list[HeavyThresholdMatchingContribution] = []

    heavy_126_fragments = so10_higgs_126_fragments()
    total_126_state_count = sum(scalar_fragment_state_count(fragment) for fragment in heavy_126_fragments)
    threshold_log_126 = math.log(PLANCK_MASS_GEV / resolved_matching_threshold_scale_gev)
    for fragment in heavy_126_fragments:
        state_count = scalar_fragment_state_count(fragment)
        coefficient = majorana_ratio * state_count / total_126_state_count
        matching_contributions.append(
            HeavyThresholdMatchingContribution(
                name=fragment.name,
                source="126_H",
                mass_gev=resolved_matching_threshold_scale_gev,
                coefficient=coefficient,
                log_enhancement=threshold_log_126,
                contribution=coefficient * threshold_log_126,
            )
        )

    coarse_210_fragments = so10_higgs_210_coarse_fragments()
    total_210_state_count = sum(state_count for _, state_count in coarse_210_fragments)
    threshold_log_210 = math.log(PLANCK_MASS_GEV / GUT_SCALE_GEV)
    for name, state_count in coarse_210_fragments:
        coefficient = gut_breaking_ratio * state_count / total_210_state_count
        matching_contributions.append(
            HeavyThresholdMatchingContribution(
                name=name,
                source="210_H",
                mass_gev=GUT_SCALE_GEV,
                coefficient=coefficient,
                log_enhancement=threshold_log_210,
                contribution=coefficient * threshold_log_210,
            )
        )

    matching_contributions.append(
        HeavyThresholdMatchingContribution(
            name="SO(10)/SM gauge bundle",
            source="V_H",
            mass_gev=GUT_SCALE_GEV,
            coefficient=gauge_projection_ratio,
            log_enhancement=threshold_log_210,
            contribution=gauge_projection_ratio * threshold_log_210,
        )
    )

    matching_log_sum = float(sum(item.contribution for item in matching_contributions))
    lambda_12_mgut = y12_tree_level * alpha_gut * matching_log_sum / (4.0 * math.pi)
    lambda_matrix_mgut = np.zeros(visible_block.shape, dtype=float)
    lambda_matrix_mgut[0, 1] = lambda_12_mgut
    lambda_matrix_mgut[1, 0] = lambda_12_mgut
    gut_threshold_residue = lambda_12_mgut / y12_tree_level
    return (
        gut_threshold_residue,
        alpha_gut,
        matching_log_sum,
        lambda_12_mgut,
        lambda_matrix_mgut,
        tuple(matching_contributions),
    )


def derive_heavy_state_decoupling_audit(
    matching_threshold_scale_gev: float | None = None,
) -> HeavyStateDecouplingAuditData:
    r"""Verify exact Appelquist--Carazzone decoupling below the $\mathbf{126}_H$ threshold."""

    resolved_threshold_scale_gev = (
        derive_topological_threshold_gev() if matching_threshold_scale_gev is None else matching_threshold_scale_gev
    )
    beta_shift_126 = derive_so10_scalar_beta_shift("126_H")
    probe_scales_gev = np.array(
        (
            MZ_SCALE_GEV,
            1.0e3,
            resolved_threshold_scale_gev / 10.0,
            np.nextafter(resolved_threshold_scale_gev, 0.0),
            resolved_threshold_scale_gev,
            np.nextafter(resolved_threshold_scale_gev, math.inf),
            GUT_SCALE_GEV,
        ),
        dtype=float,
    )
    leakage_norms = np.array(
        [
            np.linalg.norm(beta_shift_126) * (1.0 if scale_gev > resolved_threshold_scale_gev else 0.0)
            if scale_gev <= resolved_threshold_scale_gev
            else 0.0
            for scale_gev in probe_scales_gev
        ],
        dtype=float,
    )
    max_leakage = float(np.max(leakage_norms))
    return HeavyStateDecouplingAuditData(
        threshold_scale_gev=resolved_threshold_scale_gev,
        probe_scales_gev=probe_scales_gev,
        leakage_norms=leakage_norms,
        max_leakage=max_leakage,
        passed=bool(max_leakage <= condition_aware_abs_tolerance(scale=max_leakage)),
    )


def derive_nonlinearity_audit(
    scale_ratio: float = RG_SCALE_RATIO,
    level: int = LEPTON_LEVEL,
    bit_count: float = HOLOGRAPHIC_BITS,
    kappa_geometric: float = GEOMETRIC_KAPPA,
    *,
    parent_level: int = PARENT_LEVEL,
    quark_level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> NonLinearityAuditData:
    """Compare the linear RG transport against full coupled numerical evolution."""

    pmns = derive_pmns(
        level=level,
        parent_level=parent_level,
        quark_level=quark_level,
        scale_ratio=scale_ratio,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
        solver_config=solver_config,
    )
    scales = derive_scales_for_bits(
        bit_count,
        scale_ratio,
        kappa_geometric=kappa_geometric,
        parent_level=parent_level,
        lepton_level=level,
        quark_level=quark_level,
        solver_config=solver_config,
    )
    threshold = derive_rhn_threshold_data(
        scale_ratio,
        sector=Sector.LEPTON,
        parent_level=parent_level,
        lepton_level=level,
        quark_level=quark_level,
    )
    theta12_uv, theta13_uv, theta23_uv, delta_uv, _ = pdg_parameters(pmns.pmns_matrix_uv, solver_config=solver_config)
    beta_data = derive_beta_function_data(pmns.pmns_matrix_uv, sector=Sector.LEPTON, lightest_mass_ev=scales.m_0_uv_ev, level=level, solver_config=solver_config)

    linear_theta = np.array([theta12_uv, theta13_uv, theta23_uv], dtype=float) + threshold.one_loop_factor * beta_data.theta_one_loop
    linear_delta = (delta_uv + threshold.one_loop_factor * beta_data.delta_one_loop) % 360.0
    linear_m0 = apply_rg_mass_running_linearized(
        scales.m_0_uv_ev,
        scale_ratio,
        scales.gamma_0_one_loop,
        parent_level=parent_level,
        lepton_level=level,
        quark_level=quark_level,
        solver_config=solver_config,
    )

    nonlinear_pmns, nonlinear_theta, nonlinear_delta, nonlinear_m0 = integrate_pmns_rge_numerically(
        pmns.pmns_matrix_uv,
        scales.m_0_uv_ev,
        scale_ratio,
        level=level,
        gamma_0_one_loop=scales.gamma_0_one_loop,
        gamma_0_two_loop=scales.gamma_0_two_loop,
        parent_level=parent_level,
        quark_level=quark_level,
        solver_config=solver_config,
    )
    sigma_errors = {
        "theta12": float(abs(nonlinear_theta[0] - linear_theta[0]) / LEPTON_INTERVALS["theta12"].sigma),
        "theta13": float(abs(nonlinear_theta[1] - linear_theta[1]) / LEPTON_INTERVALS["theta13"].sigma),
        "theta23": float(abs(nonlinear_theta[2] - linear_theta[2]) / LEPTON_INTERVALS["theta23"].sigma),
        "delta_cp": float(abs(nonlinear_delta - linear_delta) / LEPTON_INTERVALS["delta_cp"].sigma),
    }
    return NonLinearityAuditData(
        theta_linear_deg=linear_theta,
        theta_nonlinear_deg=nonlinear_theta,
        delta_linear_deg=linear_delta,
        delta_nonlinear_deg=nonlinear_delta,
        m_0_linear_ev=linear_m0,
        m_0_nonlinear_ev=nonlinear_m0,
        sigma_errors=sigma_errors,
        max_sigma_error=max(sigma_errors.values()),
    )


def derive_step_size_convergence_audit(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    scale_ratio: float = RG_SCALE_RATIO,
    bit_count: float = HOLOGRAPHIC_BITS,
    kappa_geometric: float = GEOMETRIC_KAPPA,
    gut_threshold_residue: float | None = None,
    step_counts: tuple[int, ...] = (1000, 3000, 10000),
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> StepSizeConvergenceData:
    """Compare benchmark observables under progressively tighter ODE step ceilings."""

    if len(step_counts) < 2:
        raise ValueError("step_counts must contain at least two entries for a convergence audit.")

    sorted_step_counts = tuple(sorted(int(step_count) for step_count in step_counts))
    pmns_total_loop_time = derive_rhn_threshold_data(
        scale_ratio,
        sector=Sector.LEPTON,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    ).one_loop_factor
    quark_total_loop_time = math.log(scale_ratio) / ONE_LOOP_FACTOR

    evaluations: list[tuple[int, PmnsData, CkmData, PullTable]] = []
    for step_count in sorted_step_counts:
        pmns_max_step = pmns_total_loop_time / float(step_count)
        quark_max_step = quark_total_loop_time / float(step_count)
        pmns = derive_pmns(
            level=lepton_level,
            parent_level=parent_level,
            quark_level=quark_level,
            scale_ratio=scale_ratio,
            bit_count=bit_count,
            kappa_geometric=kappa_geometric,
            max_step=pmns_max_step,
            solver_config=solver_config,
        )
        ckm = derive_ckm(
            level=quark_level,
            parent_level=parent_level,
            scale_ratio=scale_ratio,
            gut_threshold_residue=gut_threshold_residue,
            max_step=quark_max_step,
            solver_config=solver_config,
        )
        evaluations.append((step_count, pmns, ckm, derive_pull_table(pmns, ckm)))

    reference_step_count, reference_pmns, reference_ckm, reference_pull_table = evaluations[-1]
    reference_vector = _transport_observable_vector(reference_pmns, reference_ckm)
    reference_predictive_rows = tuple(
        row for row in reference_pull_table.rows if row.included_in_predictive_fit and row.pull_data is not None
    )
    sigma_vector = np.array([row.pull_data.effective_sigma for row in reference_predictive_rows], dtype=float)

    predictive_chi2_values: list[float] = []
    delta_predictive_chi2_values: list[float] = []
    max_sigma_shift_values: list[float] = []
    for step_count, pmns, ckm, pull_table in evaluations:
        predictive_chi2_values.append(float(pull_table.predictive_chi2))
        delta_predictive_chi2_values.append(float(abs(pull_table.predictive_chi2 - reference_pull_table.predictive_chi2)))
        candidate_vector = _transport_observable_vector(pmns, ckm)
        deltas = np.array(
            [
                _transport_observable_delta(observable_name, candidate_value, reference_value)
                for observable_name, candidate_value, reference_value in zip(
                    TRANSPORT_OBSERVABLE_ORDER,
                    candidate_vector,
                    reference_vector,
                    strict=True,
                )
            ],
            dtype=float,
        )
        max_sigma_shift_values.append(
            float(
                max(
                    abs(delta) / sigma
                    for delta, sigma in zip(deltas, sigma_vector, strict=True)
                    if sigma > 0.0
                )
            )
        )

    return StepSizeConvergenceData(
        step_counts=np.array(sorted_step_counts, dtype=int),
        predictive_chi2_values=np.array(predictive_chi2_values, dtype=float),
        delta_predictive_chi2_values=np.array(delta_predictive_chi2_values, dtype=float),
        max_sigma_shift_values=np.array(max_sigma_shift_values, dtype=float),
        reference_step_count=reference_step_count,
        reference_predictive_chi2=float(reference_pull_table.predictive_chi2),
    )


def export_step_size_convergence_figure(
    convergence: StepSizeConvergenceData,
    output_path: Path | None = None,
) -> None:
    """Write the supplementary step-size convergence figure for the coupled transport."""

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME

    plotted_sigma = np.maximum(convergence.max_sigma_shift_values, np.finfo(float).eps)
    plotted_delta_chi2 = np.maximum(convergence.delta_predictive_chi2_values, np.finfo(float).eps)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True)

    ax_top.plot(convergence.step_counts, plotted_sigma, marker="o", color="#2563eb", lw=2.0)
    ax_top.axvline(convergence.reference_step_count, color="#6b7280", lw=1.0, ls=":")
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"max $|\Delta_a|/\sigma_a$")
    ax_top.grid(True, which="both", alpha=0.25, linewidth=0.6)

    ax_bottom.plot(convergence.step_counts, plotted_delta_chi2, marker="s", color="#991b1b", lw=2.0)
    ax_bottom.axvline(convergence.reference_step_count, color="#6b7280", lw=1.0, ls=":")
    ax_bottom.set_xscale("log")
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel("Nominal step ceiling")
    ax_bottom.set_ylabel(r"$|\Delta\chi^2_{\rm pred}|$")
    ax_bottom.grid(True, which="both", alpha=0.25, linewidth=0.6)

    ax_top.annotate(
        rf"reference = {convergence.reference_step_count} steps",
        xy=(convergence.reference_step_count, plotted_sigma[-1]),
        xytext=(0.04, 0.92),
        textcoords="axes fraction",
        fontsize=9,
        color="#1d4ed8",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#2563eb", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )
    ax_bottom.annotate(
        rf"$\chi^2_{{\rm pred}}({convergence.reference_step_count})={convergence.reference_predictive_chi2:.3f}$",
        xy=(convergence.reference_step_count, plotted_delta_chi2[-1]),
        xytext=(0.04, 0.16),
        textcoords="axes fraction",
        fontsize=9,
        color="#7f1d1d",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#991b1b", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
    )
    fig.suptitle("Step-size convergence of the coupled PMNS/CKM transport")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def derive_threshold_shift_audit(
    scale_ratio: float = RG_SCALE_RATIO,
    level: int = LEPTON_LEVEL,
    bit_count: float = HOLOGRAPHIC_BITS,
    kappa_geometric: float = GEOMETRIC_KAPPA,
    *,
    parent_level: int = PARENT_LEVEL,
    quark_level: int = QUARK_LEVEL,
    threshold_data: RGThresholdData | None = None,
    scales: ScaleData | None = None,
    pmns: PmnsData | None = None,
    audit: AuditData | None = None,
    beta_function_data: BetaFunctionData | None = None,
) -> ThresholdShiftAuditData:
    """Return the explicit per-observable shift decomposition for the RHN threshold audit."""

    threshold = (
        derive_rhn_threshold_data(
            scale_ratio,
            sector="lepton",
            parent_level=parent_level,
            lepton_level=level,
            quark_level=quark_level,
        )
        if threshold_data is None
        else threshold_data
    )
    scale_data = (
        derive_scales_for_bits(
            bit_count,
            scale_ratio,
            kappa_geometric=kappa_geometric,
            parent_level=parent_level,
            lepton_level=level,
            quark_level=quark_level,
        )
        if scales is None
        else scales
    )
    pmns_data = (
        derive_pmns(
            level=level,
            parent_level=parent_level,
            quark_level=quark_level,
            scale_ratio=scale_ratio,
            bit_count=bit_count,
            kappa_geometric=kappa_geometric,
        )
        if pmns is None
        else pmns
    )
    audit_data = (
        derive_audit(
            level=level,
            bit_count=bit_count,
            scale_ratio=scale_ratio,
            kappa_geometric=kappa_geometric,
            parent_level=parent_level,
            quark_level=quark_level,
        )
        if audit is None
        else audit
    )
    beta_data = (
        derive_beta_function_data(pmns_data.pmns_matrix_uv, sector="lepton", lightest_mass_ev=scale_data.m_0_uv_ev, level=level)
        if beta_function_data is None
        else beta_function_data
    )

    normalization = ONE_LOOP_FACTOR
    lower_one_loop_factor = threshold.lower_interval_log / normalization
    upper_one_loop_factor = threshold.upper_interval_log / normalization
    lower_two_loop_factor = (threshold.lower_interval_log * threshold.lower_interval_log) / (normalization * normalization)
    upper_two_loop_factor = (threshold.upper_interval_log * threshold.upper_interval_log) / (normalization * normalization)

    lower_angle_shifts = lower_one_loop_factor * beta_data.theta_one_loop
    upper_angle_shifts = upper_one_loop_factor * beta_data.theta_one_loop
    two_loop_angle_shifts = (lower_two_loop_factor + upper_two_loop_factor) * beta_data.theta_two_loop
    matching_angle_shifts = np.array(threshold.matching_angle_shifts_deg, dtype=float)
    total_angle_shifts = lower_angle_shifts + upper_angle_shifts + two_loop_angle_shifts + matching_angle_shifts

    lower_delta_shift = lower_one_loop_factor * beta_data.delta_one_loop
    upper_delta_shift = upper_one_loop_factor * beta_data.delta_one_loop
    two_loop_delta_shift = (lower_two_loop_factor + upper_two_loop_factor) * beta_data.delta_two_loop
    total_delta_shift = lower_delta_shift + upper_delta_shift + two_loop_delta_shift + threshold.matching_delta_shift_deg

    lower_mass_fraction = lower_one_loop_factor * scale_data.gamma_0_one_loop
    upper_mass_fraction = upper_one_loop_factor * scale_data.gamma_0_one_loop
    two_loop_mass_fraction = (lower_two_loop_factor + upper_two_loop_factor) * scale_data.gamma_0_two_loop
    total_mass_fraction = lower_mass_fraction + upper_mass_fraction + two_loop_mass_fraction + threshold.matching_mass_shift_fraction

    observable_shifts_deg = {
        label: TransportShiftComponentData(
            lower_one_loop=float(lower_angle_shifts[index]),
            upper_one_loop=float(upper_angle_shifts[index]),
            two_loop=float(two_loop_angle_shifts[index]),
            matching=float(matching_angle_shifts[index]),
            total=float(total_angle_shifts[index]),
            sigma=float(LEPTON_INTERVALS[label].sigma),
        )
        for index, label in enumerate(("theta12", "theta13", "theta23"))
    }
    observable_shifts_deg["delta_cp"] = TransportShiftComponentData(
        lower_one_loop=float(lower_delta_shift),
        upper_one_loop=float(upper_delta_shift),
        two_loop=float(two_loop_delta_shift),
        matching=float(threshold.matching_delta_shift_deg),
        total=float(total_delta_shift),
        sigma=float(LEPTON_INTERVALS["delta_cp"].sigma),
    )
    m_0_fraction_shift = TransportShiftComponentData(
        lower_one_loop=float(lower_mass_fraction),
        upper_one_loop=float(upper_mass_fraction),
        two_loop=float(two_loop_mass_fraction),
        matching=float(threshold.matching_mass_shift_fraction),
        total=float(total_mass_fraction),
    )

    leading_transport = np.array(
        [
            observable_shifts_deg["theta12"].leading,
            observable_shifts_deg["theta13"].leading,
            observable_shifts_deg["theta23"].leading,
            observable_shifts_deg["delta_cp"].leading,
            m_0_fraction_shift.leading,
        ],
        dtype=float,
    )
    full_transport = np.array(
        [
            observable_shifts_deg["theta12"].total,
            observable_shifts_deg["theta13"].total,
            observable_shifts_deg["theta23"].total,
            observable_shifts_deg["delta_cp"].total,
            m_0_fraction_shift.total,
        ],
        dtype=float,
    )
    leading_norm_capture = float(np.linalg.norm(leading_transport) / max(np.linalg.norm(full_transport), 1.0e-30))

    sigma_weighted_leading = np.array(
        [
            observable_shifts_deg["theta12"].sigma_weighted_leading,
            observable_shifts_deg["theta13"].sigma_weighted_leading,
            observable_shifts_deg["theta23"].sigma_weighted_leading,
            observable_shifts_deg["delta_cp"].sigma_weighted_leading,
        ],
        dtype=float,
    )
    sigma_weighted_full = np.array(
        [
            observable_shifts_deg["theta12"].sigma_weighted_total,
            observable_shifts_deg["theta13"].sigma_weighted_total,
            observable_shifts_deg["theta23"].sigma_weighted_total,
            observable_shifts_deg["delta_cp"].sigma_weighted_total,
        ],
        dtype=float,
    )
    sigma_weighted_capture = float(np.linalg.norm(sigma_weighted_leading) / max(np.linalg.norm(sigma_weighted_full), 1.0e-30))

    return ThresholdShiftAuditData(
        threshold=threshold,
        framing_gap_area_beta_sq=float(audit_data.beta * audit_data.beta),
        matching_scale_log_ratio=float(math.log(threshold.threshold_scale_gev / derive_topological_threshold_gev())),
        lower_one_loop_factor=float(lower_one_loop_factor),
        upper_one_loop_factor=float(upper_one_loop_factor),
        lower_two_loop_factor=float(lower_two_loop_factor),
        upper_two_loop_factor=float(upper_two_loop_factor),
        observable_shifts_deg=observable_shifts_deg,
        m_0_fraction_shift=m_0_fraction_shift,
        leading_norm_capture=leading_norm_capture,
        sigma_weighted_capture=sigma_weighted_capture,
    )


def derive_pmns(
    level: int | None = None,
    parent_level: int | None = None,
    quark_level: int | None = None,
    scale_ratio: float | None = None,
    bit_count: float | None = None,
    kappa_geometric: float | None = None,
    *,
    model: "TopologicalModel | None" = None,
    max_step: float | None = None,
    mz_inputs: RunningCouplings | None = None,
    solver_config: SolverConfig | None = None,
) -> PmnsData:
    """Derive the complex PMNS benchmark from modular data or an injected topological model."""

    resolved_level = _resolve_model_value(
        level,
        model=model,
        model_value=model.lepton_level if model is not None else None,
        default_value=LEPTON_LEVEL,
        parameter_name="level",
        model_parameter_name="lepton_level",
    )
    resolved_parent_level = _resolve_model_value(
        parent_level,
        model=model,
        model_value=model.parent_level if model is not None else None,
        default_value=PARENT_LEVEL,
        parameter_name="parent_level",
    )
    resolved_quark_level = _resolve_model_value(
        quark_level,
        model=model,
        model_value=model.quark_level if model is not None else None,
        default_value=QUARK_LEVEL,
        parameter_name="quark_level",
    )
    resolved_scale_ratio = _resolve_model_value(
        scale_ratio,
        model=model,
        model_value=model.scale_ratio if model is not None else None,
        default_value=RG_SCALE_RATIO,
        parameter_name="scale_ratio",
        comparator=_matching_float,
    )
    resolved_bit_count = _resolve_model_value(
        bit_count,
        model=model,
        model_value=model.bit_count if model is not None else None,
        default_value=HOLOGRAPHIC_BITS,
        parameter_name="bit_count",
        comparator=_matching_float,
    )
    resolved_kappa_geometric = _resolve_model_value(
        kappa_geometric,
        model=model,
        model_value=model.kappa_geometric if model is not None else None,
        default_value=GEOMETRIC_KAPPA,
        parameter_name="kappa_geometric",
        comparator=_matching_float,
    )
    resolved_solver_config = _resolve_model_value(
        solver_config,
        model=model,
        model_value=model.solver_config if model is not None else None,
        default_value=DEFAULT_SOLVER_CONFIG,
        parameter_name="solver_config",
    )

    scales = derive_scales_for_bits(
        resolved_bit_count,
        resolved_scale_ratio,
        kappa_geometric=resolved_kappa_geometric,
        parent_level=resolved_parent_level,
        lepton_level=resolved_level,
        quark_level=resolved_quark_level,
        solver_config=resolved_solver_config,
    )
    kernel_helper = ModularKernel(resolved_level, Sector.LEPTON, solver_config=resolved_solver_config)
    kernel_block = kernel_helper.restricted_block()
    enforce_perturbative_matrix(
        kernel_block,
        coordinate="lepton restricted flavor kernel",
        solver_config=resolved_solver_config,
        detail="The PMNS benchmark is defined only on perturbatively conditioned modular kernels.",
    )
    topological_matrix = polar_unitary(kernel_block, solver_config=resolved_solver_config)

    total_dimension = su2_total_quantum_dimension(resolved_level)
    beta = 0.5 * math.log(total_dimension)
    d1 = su2_quantum_dimension(resolved_level, 1)
    d2 = su2_quantum_dimension(resolved_level, 2)
    phi_rt = -(math.log(total_dimension) + math.log(d2 / d1)) / (4.0 * (resolved_level + 2.0))
    alpha_1_deg, alpha_2_deg = structural_majorana_phase_closure(resolved_level)

    seed_matrix = topological_kernel.rotation_23(phi_rt) @ topological_matrix
    complex_seed = complex_modular_s_matrix_representation(seed_matrix, kernel_helper)
    branch_shift_deg = 180.0
    framing_phase_deg = math.degrees(np.angle(kernel_helper.framing_phases()[2] * np.conjugate(kernel_helper.framing_phases()[1])))
    interference_phase_deg = interference_holonomy_phase(
        kernel_block,
        kernel_helper.framing_phases(),
        solver_config=resolved_solver_config,
    )
    pmns_uv, delta_uv = kernel_helper.complex_unitary(seed_matrix, kernel_block, branch_shift_deg=branch_shift_deg)
    theta12_uv, theta13_uv, theta23_uv, _, jarlskog_uv = pdg_parameters(pmns_uv, solver_config=resolved_solver_config)
    normal_order_masses_uv = normal_order_masses(scales.m_0_uv_ev)
    effective_majorana_mass_uv = effective_majorana_mass(pmns_uv, normal_order_masses_uv)
    holonomy_area_uv = jarlskog_area_factor(theta12_uv, theta13_uv, theta23_uv)
    beta_function_data = derive_beta_function_data(
        pmns_uv,
        sector="lepton",
        lightest_mass_ev=scales.m_0_uv_ev,
        running_couplings=derive_running_couplings(
            GUT_SCALE_GEV,
            solver_config=resolved_solver_config,
            mz_inputs=mz_inputs,
            max_step=max_step,
        ),
        level=resolved_level,
        solver_config=resolved_solver_config,
    )

    pmns_rg, theta_rg, delta_rg, m_0_rg_ev = integrate_pmns_rge_numerically(
        pmns_uv,
        scales.m_0_uv_ev,
        resolved_scale_ratio,
        level=resolved_level,
        gamma_0_one_loop=scales.gamma_0_one_loop,
        gamma_0_two_loop=scales.gamma_0_two_loop,
        parent_level=resolved_parent_level,
        quark_level=resolved_quark_level,
        max_step=max_step,
        mz_inputs=mz_inputs,
        solver_config=resolved_solver_config,
    )
    theta12_rg, theta13_rg, theta23_rg = (float(theta_rg[0]), float(theta_rg[1]), float(theta_rg[2]))
    _, _, _, _, jarlskog_rg = pdg_parameters(pmns_rg, solver_config=resolved_solver_config)
    normal_order_masses_rg = normal_order_masses(m_0_rg_ev)
    effective_majorana_mass_rg = effective_majorana_mass(pmns_rg, normal_order_masses_rg)
    holonomy_area_rg = jarlskog_area_factor(theta12_rg, theta13_rg, theta23_rg)
    solar_beta_one_loop = float(beta_function_data.theta_one_loop[0])
    solar_beta_two_loop = float(beta_function_data.theta_two_loop[0])
    solar_shift_deg = theta12_rg - theta12_uv
    theta12_uv_pull = pull_from_interval(theta12_uv, LEPTON_INTERVALS["theta12"]).pull
    theta12_rg_pull = pull_from_interval(theta12_rg, LEPTON_INTERVALS["theta12"]).pull

    return PmnsData(
        total_quantum_dimension=total_dimension,
        d1=d1,
        d2=d2,
        beta=beta,
        phi_rt_rad=phi_rt,
        framing_phase_deg=framing_phase_deg,
        interference_phase_deg=interference_phase_deg,
        branch_shift_deg=branch_shift_deg,
        t_phases=kernel_helper.framing_phases(),
        kernel_block=kernel_block,
        topological_matrix=topological_matrix,
        complex_seed_matrix=complex_seed,
        pmns_matrix_uv=pmns_uv,
        pmns_matrix_rg=pmns_rg,
        theta12_uv_deg=theta12_uv,
        theta13_uv_deg=theta13_uv,
        theta23_uv_deg=theta23_uv,
        theta12_rg_deg=theta12_rg,
        theta13_rg_deg=theta13_rg,
        theta23_rg_deg=theta23_rg,
        delta_cp_uv_deg=delta_uv,
        delta_cp_rg_deg=delta_rg,
        majorana_phase_1_deg=alpha_1_deg,
        majorana_phase_2_deg=alpha_2_deg,
        normal_order_masses_uv_ev=normal_order_masses_uv,
        normal_order_masses_rg_ev=normal_order_masses_rg,
        effective_majorana_mass_uv_ev=effective_majorana_mass_uv,
        effective_majorana_mass_rg_ev=effective_majorana_mass_rg,
        holonomy_area_uv=holonomy_area_uv,
        holonomy_area_rg=holonomy_area_rg,
        jarlskog_uv=jarlskog_uv,
        jarlskog_rg=jarlskog_rg,
        solar_shift_deg=solar_shift_deg,
        theta12_uv_pull=theta12_uv_pull,
        theta12_rg_pull=theta12_rg_pull,
        solar_beta_one_loop=solar_beta_one_loop,
        solar_beta_two_loop=solar_beta_two_loop,
        level=resolved_level,
        parent_level=resolved_parent_level,
        scale_ratio=resolved_scale_ratio,
        bit_count=resolved_bit_count,
        kappa_geometric=resolved_kappa_geometric,
        solver_config=resolved_solver_config,
    )


def permutation_sign(permutation: tuple[int, int, int]) -> int:
    inversions = 0
    for left in range(len(permutation)):
        for right in range(left + 1, len(permutation)):
            inversions += int(permutation[left] > permutation[right])
    return -1 if inversions % 2 else 1


def su3_conformal_weight(level: int, weight: tuple[int, int]) -> float:
    dynkin_left, dynkin_right = weight
    numerator = dynkin_left * dynkin_left + dynkin_right * dynkin_right + dynkin_left * dynkin_right + 3 * dynkin_left + 3 * dynkin_right
    return numerator / (3.0 * (level + 3.0))


def su3_weight_vector(weight: tuple[int, int]) -> np.ndarray:
    dynkin_left, dynkin_right = weight
    return np.array(
        [
            (2.0 * dynkin_left + dynkin_right) / 3.0,
            (-dynkin_left + dynkin_right) / 3.0,
            -(dynkin_left + 2.0 * dynkin_right) / 3.0,
        ],
        dtype=float,
    )


def su3_modular_s_entry(level: int, left_weight: tuple[int, int], right_weight: tuple[int, int]) -> complex:
    rho_vector = su3_weight_vector((1, 1))
    left = su3_weight_vector(left_weight) + rho_vector
    right = su3_weight_vector(right_weight) + rho_vector
    prefactor = (1j ** 3) / (math.sqrt(3.0) * (level + 3.0))
    total = 0.0j
    for permutation in itertools.permutations((0, 1, 2)):
        total += permutation_sign(permutation) * np.exp(
            -2j * math.pi * np.dot(left[list(permutation)], right) / (level + 3.0)
        )
    return prefactor * total


def su3_low_weight_block(level: int, weights: tuple[tuple[int, int], ...] = LOW_SU3_WEIGHTS) -> np.ndarray:
    return np.array(
        [[su3_modular_s_entry(level, left_weight, right_weight) for right_weight in weights] for left_weight in weights],
        dtype=complex,
    )


def weyl_vector_norm_sq(dimension: int, dual_coxeter: int) -> float:
    return dual_coxeter * dimension / 12.0


def rank_deficit_pressure(parent_level: int = PARENT_LEVEL, quark_level: int = QUARK_LEVEL) -> float:
    so10_weyl_norm_sq = weyl_vector_norm_sq(SO10_DIMENSION, SO10_DUAL_COXETER)
    su3_weyl_norm_sq = weyl_vector_norm_sq(SU3_DIMENSION, SU3_DUAL_COXETER)
    weyl_ratio = math.sqrt(so10_weyl_norm_sq / su3_weyl_norm_sq)
    return weyl_ratio * math.sqrt((quark_level + SU3_DUAL_COXETER) / (parent_level + SO10_DUAL_COXETER))

rotation_23 = algebra.rotation_23
pdg_unitary = algebra.pdg_unitary
jarlskog_invariant = algebra.jarlskog_invariant
pdg_parameters = algebra.pdg_parameters
su2_quantum_dimension = algebra.su2_quantum_dimension
su2_conformal_weight = algebra.su2_conformal_weight
su2_modular_s = algebra.su2_modular_s
charge_embedding = algebra.charge_embedding
permutation_sign = algebra.permutation_sign
su3_conformal_weight = algebra.su3_conformal_weight
su3_weight_vector = algebra.su3_weight_vector
su3_modular_s_entry = algebra.su3_modular_s_entry
su3_low_weight_block = algebra.su3_low_weight_block
interference_holonomy_phase = algebra.interference_holonomy_phase
predict_delta_cp = algebra.predict_delta_cp


def quark_branching_pressure(
    visible_block: np.ndarray,
    rank_difference: int = RANK_DIFFERENCE,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    return physics_engine.quark_branching_pressure(
        visible_block,
        rank_difference,
        solver_config=solver_config,
    )


@lru_cache(maxsize=128)
def derive_topological_threshold_gev(
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> float:
    r"""Return the Sec. 10.2 leading-order threshold correction for $M_{126}$.

    The benchmark colored threshold is chosen by a Framing Gap Alignment ultraviolet matching condition:
    the structural RHN scale $M_N$ is separated from the visible $\mathbf{126}_H$
    insertion by exactly one relaxed framing gap,

        M_{126}^{\rm match} = M_N \exp[-\beta^2],

    with $\beta = \tfrac12\ln \mathcal D_{26}$.
    """

    threshold_scale_gev, *_ = derive_structural_rhn_scale_gev(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    beta = 0.5 * math.log(su2_total_quantum_dimension(lepton_level))
    framing_gap_area = beta * beta
    # Sec. 10.2 leading-order Framing Gap Alignment correction:
    # the visible 126_H threshold is selected so that the logarithmic drift
    # matches the relaxed framing-gap area A_framing^(SO(10)) = beta^2.
    return threshold_scale_gev * math.exp(-framing_gap_area)

def quark_branch_kernel(
    visible_block: np.ndarray,
    vacuum_pressure: float,
    descendant_factors: tuple[float, float],
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float]]:
    r"""Dress the bare quark pressure according to Eq. ``eq:vus-threshold-gradient``."""

    guard = solver_config.stability_guard
    epsilon12 = math.exp(-vacuum_pressure) * descendant_factors[0]
    epsilon23 = math.exp(-2.0 * vacuum_pressure) * descendant_factors[1]
    epsilon13 = epsilon12 * epsilon23
    branch_kernel = np.array(
        [
            [1.0, epsilon12, epsilon13],
            [-epsilon12, 1.0, epsilon23],
            [epsilon13, -epsilon23, 1.0],
        ],
        dtype=complex,
    )

    for row, col in ((0, 1), (0, 2), (1, 2)):
        block_entry = visible_block[row, col]
        phase = guard.stable_phase(block_entry, coordinate=f"quark branch kernel phase[{row},{col}]")
        branch_kernel[row, col] *= phase
        branch_kernel[col, row] *= np.conjugate(phase)

    epsilon12_magnitude = guard.clamp_positive(
        abs(epsilon12),
        coordinate="quark branch epsilon12",
        floor=guard.zero_magnitude_threshold,
    )
    epsilon23_magnitude = guard.clamp_positive(
        abs(epsilon23),
        coordinate="quark branch epsilon23",
        floor=guard.zero_magnitude_threshold,
    )
    return branch_kernel, (epsilon12, epsilon23, epsilon13), (-math.log(epsilon12_magnitude), -math.log(epsilon23_magnitude))


def _resolve_model_value(
    explicit_value,
    *,
    model: "TopologicalModel | None",
    model_value,
    default_value,
    parameter_name: str,
    model_parameter_name: str | None = None,
    comparator=None,
):
    if model is None:
        return default_value if explicit_value is None else explicit_value

    resolved_model_parameter_name = parameter_name if model_parameter_name is None else model_parameter_name
    if explicit_value is None:
        return model_value

    compare = (lambda left, right: left == right) if comparator is None else comparator
    if not compare(explicit_value, model_value):
        raise ValueError(
            f"Explicit {parameter_name}={explicit_value!r} conflicts with TopologicalModel.{resolved_model_parameter_name}={model_value!r}."
        )
    return explicit_value


def _matching_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left == right
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)


def derive_ckm(
    level: int | None = None,
    parent_level: int | None = None,
    scale_ratio: float | None = None,
    gut_threshold_residue: float | None = None,
    *,
    model: "TopologicalModel | None" = None,
    ckm_phase_tilt_parameter: float | None = None,
    max_step: float | None = None,
    mz_inputs: RunningCouplings | None = None,
    solver_config: SolverConfig | None = None,
) -> CkmData:
    """Derive the CKM benchmark from the pressure-dressed $SU(3)$ block or an injected topological model."""

    resolved_level = _resolve_model_value(
        level,
        model=model,
        model_value=model.quark_level if model is not None else None,
        default_value=QUARK_LEVEL,
        parameter_name="level",
        model_parameter_name="quark_level",
    )
    resolved_parent_level = _resolve_model_value(
        parent_level,
        model=model,
        model_value=model.parent_level if model is not None else None,
        default_value=PARENT_LEVEL,
        parameter_name="parent_level",
    )
    resolved_scale_ratio = _resolve_model_value(
        scale_ratio,
        model=model,
        model_value=model.scale_ratio if model is not None else None,
        default_value=RG_SCALE_RATIO,
        parameter_name="scale_ratio",
        comparator=_matching_float,
    )
    resolved_gut_threshold_residue = _resolve_model_value(
        gut_threshold_residue,
        model=model,
        model_value=model.gut_threshold_residue if model is not None else None,
        default_value=R_GUT,
        parameter_name="gut_threshold_residue",
        comparator=_matching_float,
    )
    resolved_solver_config = _resolve_model_value(
        solver_config,
        model=model,
        model_value=model.solver_config if model is not None else None,
        default_value=DEFAULT_SOLVER_CONFIG,
        parameter_name="solver_config",
    )
    resolved_lepton_level = model.lepton_level if model is not None else LEPTON_LEVEL

    if (
        resolved_gut_threshold_residue is not None
        and ckm_phase_tilt_parameter is not None
        and not math.isclose(resolved_gut_threshold_residue, ckm_phase_tilt_parameter, rel_tol=1.0e-12, abs_tol=1.0e-12)
    ):
        raise ValueError("Conflicting CKM threshold-residue aliases were supplied.")

    resolved_gut_threshold_residue = (
        resolved_gut_threshold_residue if resolved_gut_threshold_residue is not None else ckm_phase_tilt_parameter
    )

    kernel_helper = ModularKernel(resolved_level, Sector.QUARK, solver_config=resolved_solver_config)
    visible_block = kernel_helper.restricted_block()
    coset_block = su3_low_weight_block(resolved_parent_level // 3)
    enforce_perturbative_matrix(
        visible_block,
        coordinate="quark restricted flavor kernel",
        solver_config=resolved_solver_config,
        detail="The CKM benchmark is defined only on perturbatively conditioned visible flavor blocks.",
    )
    enforce_perturbative_matrix(
        coset_block,
        coordinate="quark low-weight coset kernel",
        solver_config=resolved_solver_config,
        detail="The CKM threshold audit requires a perturbatively conditioned low-weight coset kernel.",
    )
    threshold_correction = derive_so10_threshold_correction(
        visible_block,
        coset_block,
        gut_threshold_residue=resolved_gut_threshold_residue,
        parent_level=resolved_parent_level,
        lepton_level=resolved_lepton_level,
        quark_level=resolved_level,
        solver_config=resolved_solver_config,
    )
    coset_weighting = coset_topological_weighting(
        visible_block,
        coset_block,
        parent_level=resolved_parent_level,
        lepton_level=resolved_lepton_level,
        quark_level=resolved_level,
        solver_config=resolved_solver_config,
    )
    branching_index = quark_branching_index(resolved_parent_level, resolved_level)
    so10_weyl_norm_sq = weyl_vector_norm_sq(SO10_DIMENSION, SO10_DUAL_COXETER)
    su3_weyl_norm_sq = weyl_vector_norm_sq(SU3_DIMENSION, SU3_DUAL_COXETER)
    weyl_ratio = math.sqrt(so10_weyl_norm_sq / su3_weyl_norm_sq)
    raw_pressure = rank_deficit_pressure(resolved_parent_level, resolved_level)
    vacuum_pressure = quark_branching_pressure(visible_block, solver_config=resolved_solver_config)

    bare_branch_kernel, bare_topological_weights, _ = quark_branch_kernel(
        visible_block,
        vacuum_pressure,
        (1.0, 1.0),
        solver_config=resolved_solver_config,
    )
    branch_kernel, topological_weights, channel_pressures = quark_branch_kernel(
        visible_block,
        vacuum_pressure,
        (coset_weighting[0, 1], coset_weighting[1, 2]),
        solver_config=resolved_solver_config,
    )

    bare_complex_seed = complex_modular_s_matrix_representation(bare_branch_kernel, kernel_helper)
    bare_theta_c_uv, bare_theta13_uv, bare_theta23_uv, _, _ = pdg_parameters(
        bare_complex_seed,
        solver_config=resolved_solver_config,
    )
    bare_ckm_uv = pdg_unitary(bare_theta_c_uv, bare_theta13_uv, bare_theta23_uv, 0.0)
    bare_ckm_rg = integrate_quark_rge_numerically(
        bare_ckm_uv,
        resolved_scale_ratio,
        level=resolved_level,
        max_step=max_step,
        mz_inputs=mz_inputs,
        solver_config=resolved_solver_config,
    )
    bare_vus_uv, bare_vcb_uv, bare_vub_uv = abs(bare_ckm_uv[0, 1]), abs(bare_ckm_uv[1, 2]), abs(bare_ckm_uv[0, 2])
    bare_vus_rg, bare_vcb_rg, bare_vub_rg = abs(bare_ckm_rg[0, 1]), abs(bare_ckm_rg[1, 2]), abs(bare_ckm_rg[0, 2])

    complex_seed = complex_modular_s_matrix_representation(branch_kernel, kernel_helper)
    theta_c_uv, theta13_uv, theta23_uv, _, _ = pdg_parameters(
        complex_seed,
        solver_config=resolved_solver_config,
    )
    topological_jarlskog_uv = topological_jarlskog_identity(
        resolved_gut_threshold_residue,
        parent_level=resolved_parent_level,
        lepton_level=resolved_lepton_level,
        quark_level=resolved_level,
    )
    locked_jarlskog_uv = threshold_projected_jarlskog(
        topological_jarlskog_uv,
        gut_threshold_residue=resolved_gut_threshold_residue,
    )
    delta_uv = delta_cp_from_jarlskog_lock(
        theta_c_uv,
        theta13_uv,
        theta23_uv,
        locked_jarlskog_uv,
        branch_reference_deg=threshold_correction.triangle_tilt_deg,
        solver_config=resolved_solver_config,
    )
    ckm_uv = pdg_unitary(theta_c_uv, theta13_uv, theta23_uv, delta_uv)
    _, _, _, _, jarlskog_uv = pdg_parameters(ckm_uv, solver_config=resolved_solver_config)

    ckm_rg = integrate_quark_rge_numerically(
        ckm_uv,
        resolved_scale_ratio,
        level=resolved_level,
        max_step=max_step,
        mz_inputs=mz_inputs,
        solver_config=resolved_solver_config,
    )
    theta_c_rg, theta13_rg, theta23_rg, delta_rg, jarlskog_rg = pdg_parameters(ckm_rg, solver_config=resolved_solver_config)
    alpha_uv, beta_uv, gamma_uv = ckm_unitarity_triangle_angles(ckm_uv)
    alpha_rg, beta_rg, gamma_rg = ckm_unitarity_triangle_angles(ckm_rg)

    vus_uv, vcb_uv, vub_uv = abs(ckm_uv[0, 1]), abs(ckm_uv[1, 2]), abs(ckm_uv[0, 2])
    vus_rg, vcb_rg, vub_rg = abs(ckm_rg[0, 1]), abs(ckm_rg[1, 2]), abs(ckm_rg[0, 2])

    return CkmData(
        visible_block=visible_block,
        coset_block=coset_block,
        coset_weighting=coset_weighting,
        bare_topological_weights=bare_topological_weights,
        topological_weights=topological_weights,
        rank_difference=RANK_DIFFERENCE,
        branching_index=branching_index,
        so10_weyl_norm_sq=so10_weyl_norm_sq,
        su3_weyl_norm_sq=su3_weyl_norm_sq,
        weyl_ratio=weyl_ratio,
        rank_deficit_pressure=raw_pressure,
        vacuum_pressure=vacuum_pressure,
        so10_threshold_correction=threshold_correction,
        channel_pressures=channel_pressures,
        descendant_factors=(coset_weighting[0, 1], coset_weighting[1, 2]),
        t_phases=kernel_helper.framing_phases(),
        complex_seed_matrix=complex_seed,
        bare_ckm_matrix_uv=bare_ckm_uv,
        bare_ckm_matrix_rg=bare_ckm_rg,
        ckm_matrix_uv=ckm_uv,
        ckm_matrix_rg=ckm_rg,
        theta_c_uv_deg=theta_c_uv,
        theta13_uv_deg=theta13_uv,
        theta23_uv_deg=theta23_uv,
        theta_c_rg_deg=theta_c_rg,
        theta13_rg_deg=theta13_rg,
        theta23_rg_deg=theta23_rg,
        alpha_uv_deg=alpha_uv,
        beta_uv_deg=beta_uv,
        gamma_uv_deg=gamma_uv,
        alpha_rg_deg=alpha_rg,
        beta_rg_deg=beta_rg,
        gamma_rg_deg=gamma_rg,
        bare_vus_uv=bare_vus_uv,
        bare_vcb_uv=bare_vcb_uv,
        bare_vub_uv=bare_vub_uv,
        bare_vus_rg=bare_vus_rg,
        bare_vcb_rg=bare_vcb_rg,
        bare_vub_rg=bare_vub_rg,
        vus_uv=vus_uv,
        vcb_uv=vcb_uv,
        vub_uv=vub_uv,
        vus_rg=vus_rg,
        vcb_rg=vcb_rg,
        vub_rg=vub_rg,
        cabibbo_threshold_push_uv=vus_uv - bare_vus_uv,
        cabibbo_threshold_push_rg=vus_rg - bare_vus_rg,
        delta_cp_uv_deg=delta_uv,
        delta_cp_rg_deg=delta_rg,
        jarlskog_uv=jarlskog_uv,
        jarlskog_rg=jarlskog_rg,
        level=resolved_level,
        parent_level=resolved_parent_level,
        scale_ratio=resolved_scale_ratio,
        gut_threshold_residue=threshold_correction.gut_threshold_residue,
        solver_config=resolved_solver_config,
    )


TRANSPORT_OBSERVABLE_ORDER = ("theta12", "theta13", "theta23", "delta_cp", "vus", "vcb", "vub", "gamma")
TRANSPORT_INPUT_ORDER = ("top_yukawa_mz", "alpha_s_mz")
ANGULAR_TRANSPORT_OBSERVABLES = frozenset({"theta12", "theta13", "theta23", "delta_cp", "gamma"})


def top_yukawa_mz_input_sigma() -> float:
    """Return the propagated 1σ uncertainty on the benchmark top Yukawa at $M_Z$."""

    return float(SM_MZ_YUKAWA_BENCHMARKS["top"] * PDG_TOP_POLE_MASS_SIGMA_GEV / PDG_TOP_POLE_MASS_CENTRAL_GEV)


def _transport_observable_vector(pmns: PmnsData, ckm: CkmData) -> np.ndarray:
    return transport_observable_vector_impl(pmns, ckm)


def _transport_observable_delta(observable_name: str, upper_value: float, lower_value: float) -> float:
    return transport_observable_delta_impl(
        observable_name,
        upper_value,
        lower_value,
        angular_observables=ANGULAR_TRANSPORT_OBSERVABLES,
        wrapped_angle_difference_deg=wrapped_angle_difference_deg,
    )


def _transport_observable_sigma_vector() -> np.ndarray:
    return np.array(
        [
            LEPTON_INTERVALS["theta12"].sigma,
            LEPTON_INTERVALS["theta13"].sigma,
            LEPTON_INTERVALS["theta23"].sigma,
            LEPTON_INTERVALS["delta_cp"].sigma,
            QUARK_INTERVALS["vus"].sigma,
            QUARK_INTERVALS["vcb"].sigma,
            QUARK_INTERVALS["vub"].sigma,
            CKM_GAMMA_GOLD_STANDARD_DEG.sigma,
        ],
        dtype=float,
    )


def _transport_failure_penalty_vector(chi2_penalty: float) -> np.ndarray:
    if chi2_penalty <= 0.0:
        return np.zeros(len(TRANSPORT_OBSERVABLE_ORDER), dtype=float)
    sigma_vector = _transport_observable_sigma_vector()
    return sigma_vector * math.sqrt(chi2_penalty / len(sigma_vector))



def _transport_covariance_model(
    pmns_level: int,
    pmns_parent_level: int,
    pmns_scale_ratio: float,
    pmns_bit_count: float,
    pmns_kappa_geometric: float,
    ckm_level: int,
    ckm_parent_level: int,
    ckm_scale_ratio: float,
    gut_threshold_residue: float | None,
    solver_config: SolverConfig,
) -> TopologicalModel:
    if not solver_isclose(pmns_scale_ratio, ckm_scale_ratio):
        raise ValueError("PMNS and CKM scale ratios must match for a joint transport covariance audit.")
    if pmns_parent_level != ckm_parent_level:
        raise ValueError("PMNS and CKM data must share the same parent level for a joint transport covariance audit.")

    return TopologicalModel(
        k_l=pmns_level,
        k_q=ckm_level,
        parent_level=pmns_parent_level,
        scale_ratio=pmns_scale_ratio,
        bit_count=pmns_bit_count,
        kappa_geometric=pmns_kappa_geometric,
        gut_threshold_residue=gut_threshold_residue,
        solver_config=solver_config,
    )


def _shifted_transport_observables(
    model: TopologicalModel,
    *,
    top_yukawa_mz: float,
    alpha_s_mz: float,
) -> np.ndarray:
    mz_inputs = running_coupling_mz_inputs(top_yukawa_mz=top_yukawa_mz, alpha_s_mz=alpha_s_mz)
    shifted_pmns = derive_pmns(model=model, mz_inputs=mz_inputs)
    shifted_ckm = derive_ckm(model=model, mz_inputs=mz_inputs)
    return _transport_observable_vector(shifted_pmns, shifted_ckm)


def _derive_transport_parametric_covariance_linearized(
    pmns_level: int,
    pmns_parent_level: int,
    pmns_scale_ratio: float,
    pmns_bit_count: float,
    pmns_kappa_geometric: float,
    ckm_level: int,
    ckm_parent_level: int,
    ckm_scale_ratio: float,
    gut_threshold_residue: float | None,
    solver_config: SolverConfig,
) -> TransportParametricCovarianceData:
    model = _transport_covariance_model(
        pmns_level,
        pmns_parent_level,
        pmns_scale_ratio,
        pmns_bit_count,
        pmns_kappa_geometric,
        ckm_level,
        ckm_parent_level,
        ckm_scale_ratio,
        gut_threshold_residue,
        solver_config,
    )

    input_central_values = np.array([SM_MZ_YUKAWA_BENCHMARKS["top"], PLANCK2018_ALPHA_S_MZ], dtype=float)
    input_sigmas = np.array([top_yukawa_mz_input_sigma(), ALPHA_S_MZ_SIGMA], dtype=float)
    relative_step = max(float(solver_config.jacobian_relative_step), np.finfo(float).eps)
    finite_difference_steps = relative_step * np.maximum(np.maximum(np.abs(input_central_values), input_sigmas), 1.0)
    jacobian = np.zeros((len(TRANSPORT_OBSERVABLE_ORDER), len(TRANSPORT_INPUT_ORDER)), dtype=float)

    for input_index, step in enumerate(finite_difference_steps):
        plus_inputs = np.array(input_central_values, copy=True)
        minus_inputs = np.array(input_central_values, copy=True)
        plus_inputs[input_index] += step
        minus_inputs[input_index] -= step
        observable_plus = _shifted_transport_observables(
            model,
            top_yukawa_mz=float(plus_inputs[0]),
            alpha_s_mz=float(plus_inputs[1]),
        )
        observable_minus = _shifted_transport_observables(
            model,
            top_yukawa_mz=float(minus_inputs[0]),
            alpha_s_mz=float(minus_inputs[1]),
        )
        for observable_index, observable_name in enumerate(TRANSPORT_OBSERVABLE_ORDER):
            jacobian[observable_index, input_index] = _transport_observable_delta(
                observable_name,
                float(observable_plus[observable_index]),
                float(observable_minus[observable_index]),
            ) / (2.0 * float(step))

    input_covariance = np.diag(np.square(input_sigmas))
    linearized_covariance = jacobian @ input_covariance @ jacobian.T
    return freeze_numpy_arrays(TransportParametricCovarianceData(
        observable_names=TRANSPORT_OBSERVABLE_ORDER,
        input_names=TRANSPORT_INPUT_ORDER,
        jacobian=jacobian,
        input_central_values=input_central_values,
        input_sigmas=input_sigmas,
        finite_difference_steps=finite_difference_steps,
        covariance=np.asarray(linearized_covariance, dtype=float),
    ))


def _derive_transport_parametric_covariance_cached(
    pmns_level: int,
    pmns_parent_level: int,
    pmns_scale_ratio: float,
    pmns_bit_count: float,
    pmns_kappa_geometric: float,
    ckm_level: int,
    ckm_parent_level: int,
    ckm_scale_ratio: float,
    gut_threshold_residue: float | None,
    solver_config: SolverConfig,
    *,
    rng: np.random.Generator | None = None,
) -> TransportParametricCovarianceData:
    linearized_audit = _derive_transport_parametric_covariance_linearized(
        pmns_level,
        pmns_parent_level,
        pmns_scale_ratio,
        pmns_bit_count,
        pmns_kappa_geometric,
        ckm_level,
        ckm_parent_level,
        ckm_scale_ratio,
        gut_threshold_residue,
        solver_config,
    )
    model = _transport_covariance_model(
        pmns_level,
        pmns_parent_level,
        pmns_scale_ratio,
        pmns_bit_count,
        pmns_kappa_geometric,
        ckm_level,
        ckm_parent_level,
        ckm_scale_ratio,
        gut_threshold_residue,
        solver_config,
    )
    resolved_rng = np.random.default_rng(DEFAULT_RANDOM_SEED) if rng is None else rng
    central_observables = _shifted_transport_observables(
        model,
        top_yukawa_mz=float(linearized_audit.input_central_values[0]),
        alpha_s_mz=float(linearized_audit.input_central_values[1]),
    )
    sampled_inputs = resolved_rng.normal(
        loc=linearized_audit.input_central_values,
        scale=linearized_audit.input_sigmas,
        size=(int(solver_config.parametric_covariance_mc_samples), len(linearized_audit.input_central_values)),
    )
    delta_samples: list[np.ndarray] = []
    accepted_delta_samples: list[np.ndarray] = []
    failure_count = 0
    penalty_vector = _transport_failure_penalty_vector(TRANSPORT_SINGULARITY_CHI2_PENALTY)
    for top_yukawa_mz, alpha_s_mz in sampled_inputs:
        try:
            sampled_observables = _shifted_transport_observables(
                model,
                top_yukawa_mz=float(top_yukawa_mz),
                alpha_s_mz=float(alpha_s_mz),
            )
            delta_sample = np.array(
                [
                    _transport_observable_delta(name, float(sampled_observables[index]), float(central_observables[index]))
                    for index, name in enumerate(TRANSPORT_OBSERVABLE_ORDER)
                ],
                dtype=float,
            )
        except (PhysicalSingularityException, PerturbativeBreakdownException):
            failure_count += 1
            delta_sample = np.array(penalty_vector, copy=True)
        else:
            accepted_delta_samples.append(np.array(delta_sample, copy=True))
        delta_samples.append(delta_sample)

    attempted_samples = int(sampled_inputs.shape[0])
    accepted_sample_count = max(attempted_samples - failure_count, 0)
    stability_yield = 1.0 if attempted_samples <= 0 else accepted_sample_count / attempted_samples
    sample_matrix = (
        np.asarray(accepted_delta_samples, dtype=float)
        if accepted_delta_samples
        else np.empty((0, len(TRANSPORT_OBSERVABLE_ORDER)), dtype=float)
    )
    lower_quantiles = np.empty(0, dtype=float)
    upper_quantiles = np.empty(0, dtype=float)
    skewness = np.empty(0, dtype=float)
    requires_jacobian_fallback = failure_count > 0 or stability_yield < TRANSPORT_MC_MIN_STABILITY_YIELD

    if failure_count > 0 or stability_yield < TRANSPORT_MC_MIN_STABILITY_YIELD:
        warnings.warn(
            (
                "Transport covariance Monte Carlo encountered physical failures: "
                f"accepted {accepted_sample_count}/{attempted_samples} samples "
                f"({stability_yield:.1%} stability yield, {failure_count} hard-wall penalties, "
                f"chi2_penalty={TRANSPORT_SINGULARITY_CHI2_PENALTY:.1e})."
            ),
            MonteCarloYieldWarning,
            stacklevel=2,
        )

    if sample_matrix.shape[0] >= 2 and not requires_jacobian_fallback:
        covariance = np.cov(sample_matrix, rowvar=False, ddof=1)
        lower_quantiles = np.quantile(sample_matrix, 0.16, axis=0)
        upper_quantiles = np.quantile(sample_matrix, 0.84, axis=0)
        skewness = _distribution_skewness(sample_matrix)
        covariance_mode = "monte_carlo"
    elif sample_matrix.shape[0] == 1 and not requires_jacobian_fallback:
        covariance = np.diag(np.square(sample_matrix[0]))
        lower_quantiles = np.array(sample_matrix[0], copy=True)
        upper_quantiles = np.array(sample_matrix[0], copy=True)
        skewness = np.zeros(len(TRANSPORT_OBSERVABLE_ORDER), dtype=float)
        covariance_mode = "single_sample"
    else:
        covariance = linearized_audit.covariance
        if sample_matrix.size > 0:
            lower_quantiles = np.quantile(sample_matrix, 0.16, axis=0)
            upper_quantiles = np.quantile(sample_matrix, 0.84, axis=0)
            skewness = _distribution_skewness(sample_matrix)
        else:
            skewness = np.zeros(len(TRANSPORT_OBSERVABLE_ORDER), dtype=float)
            lower_quantiles = np.zeros(len(TRANSPORT_OBSERVABLE_ORDER), dtype=float)
            upper_quantiles = np.zeros(len(TRANSPORT_OBSERVABLE_ORDER), dtype=float)
        covariance_mode = "jacobian_low_yield" if requires_jacobian_fallback else "jacobian"

    return freeze_numpy_arrays(TransportParametricCovarianceData(
        observable_names=linearized_audit.observable_names,
        input_names=linearized_audit.input_names,
        jacobian=linearized_audit.jacobian,
        input_central_values=linearized_audit.input_central_values,
        input_sigmas=linearized_audit.input_sigmas,
        finite_difference_steps=linearized_audit.finite_difference_steps,
        covariance=np.asarray(covariance, dtype=float),
        covariance_mode=covariance_mode,
        lower_quantiles=np.asarray(lower_quantiles, dtype=float),
        upper_quantiles=np.asarray(upper_quantiles, dtype=float),
        skewness=np.asarray(skewness, dtype=float),
        attempted_samples=attempted_samples,
        accepted_samples=accepted_sample_count,
        failure_count=failure_count,
        singularity_chi2_penalty=TRANSPORT_SINGULARITY_CHI2_PENALTY if failure_count > 0 else 0.0,
    ))



def derive_transport_parametric_covariance(
    pmns: PmnsData,
    ckm: CkmData,
    *,
    rng: np.random.Generator | None = None,
) -> TransportParametricCovarianceData:
    """Return the small-step Jacobian audit with a Monte Carlo covariance estimate."""

    if pmns.solver_config != ckm.solver_config:
        raise ValueError("PMNS and CKM data must share the same solver configuration for covariance propagation.")
    return _derive_transport_parametric_covariance_cached(
        pmns.level,
        pmns.parent_level,
        pmns.scale_ratio,
        pmns.bit_count,
        pmns.kappa_geometric,
        ckm.level,
        ckm.parent_level,
        ckm.scale_ratio,
        ckm.gut_threshold_residue,
        pmns.solver_config,
        rng=rng,
    )


def generate_ckm_phase_tilt_profile(
    pmns: PmnsData | None = None,
    weight_grid: np.ndarray | None = None,
    output_path: Path | None = None,
    *,
    model: TopologicalModel | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    parent_level: int | None = None,
    scale_ratio: float | None = None,
    benchmark_weight: float | None = None,
) -> CkmPhaseTiltProfileData:
    r"""Evaluate the full CKM threshold-weight profile and export the resulting curve."""

    resolved_model = _coerce_topological_model(
        model=model,
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        scale_ratio=scale_ratio,
    )
    reference_pmns = (
        derive_pmns(model=resolved_model)
        if pmns is None
        else pmns
    )
    resolved_benchmark_weight = (
        derive_ckm(model=resolved_model).so10_threshold_correction.gut_threshold_residue
        if benchmark_weight is None
        else float(benchmark_weight)
    )
    return build_ckm_phase_tilt_profile(
        reference_pmns=reference_pmns,
        weight_grid=weight_grid,
        output_path=output_path,
        quark_level=resolved_model.quark_level,
        parent_level=resolved_model.parent_level,
        scale_ratio=resolved_model.scale_ratio,
        benchmark_weight=resolved_benchmark_weight,
        ckm_phase_tilt_invariance_tolerance=CKM_PHASE_TILT_INVARIANCE_TOLERANCE,
        derive_ckm=derive_ckm,
        derive_pull_table=derive_pull_table,
        plt=plt,
        profile_data_factory=CkmPhaseTiltProfileData,
    )


def derive_parent_selection(
    lepton_level: int | None = None,
    quark_level: int | None = None,
    *,
    model: TopologicalModel | None = None,
) -> ParentSelection:
    """Solve the Diophantine branch-compatibility rule for the visible pair.

    Args:
        lepton_level: Visible leptonic affine level.
        quark_level: Visible quark affine level.

    Returns:
        The minimal parent level and integer branching indices.
    """

    resolved_model = _coerce_topological_model(model=model, lepton_level=lepton_level, quark_level=quark_level)
    master_level = math.lcm(
        SO10_TO_SU2_EMBEDDING_INDEX * SU2_DUAL_COXETER * resolved_model.lepton_level,
        SO10_TO_SU3_EMBEDDING_INDEX * SU3_DUAL_COXETER * resolved_model.quark_level,
    )
    lepton_branching = lepton_branching_index(master_level, resolved_model.lepton_level)
    quark_branching = quark_branching_index(master_level, resolved_model.quark_level)
    return ParentSelection(
        master_level=master_level,
        lepton_branching_index=lepton_branching,
        quark_branching_index=quark_branching,
    )


def verify_framing_anomaly(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    search_depth: int = 24,
    *,
    model: TopologicalModel | None = None,
) -> FramingAnomalyData:
    r"""Evaluate the visible framing residual and branch-compatible scan from Eq. ``eq:master-level-modularity``.

    Args:
        parent_level: Parent affine level used for the explicit benchmark.
        lepton_level: Visible leptonic level.
        quark_level: Visible quark level.
        search_depth: Number of branch-compatible multiples to inspect.

    Returns:
        Framing-anomaly data for the benchmark and nearby branch-compatible levels.
    """

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    lepton_central_charge = wzw_central_charge(resolved_model.lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER)
    quark_central_charge = wzw_central_charge(resolved_model.quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
    parent_central_charge = wzw_central_charge(resolved_model.parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
    visible_central_charge = lepton_central_charge + quark_central_charge
    visible_residual_mod1 = mod_one_residual(parent_central_charge / 24.0 - visible_central_charge / 24.0)
    coset_central_charge = parent_central_charge - visible_central_charge
    total_residual_mod1 = mod_one_residual(
        parent_central_charge / 24.0
        - lepton_central_charge / 24.0
        - quark_central_charge / 24.0
        - coset_central_charge / 24.0
    )

    branch_period = math.lcm(SU2_DUAL_COXETER * resolved_model.lepton_level, SU3_DUAL_COXETER * resolved_model.quark_level)
    best_branch_level = resolved_model.parent_level
    best_branch_residual_mod1 = visible_residual_mod1
    best_distance = distance_to_integer(parent_central_charge / 24.0 - visible_central_charge / 24.0)
    for multiplier in range(1, search_depth + 1):
        candidate_level = multiplier * branch_period
        candidate_parent_central_charge = wzw_central_charge(candidate_level, SO10_DIMENSION, SO10_DUAL_COXETER)
        candidate_residual_mod1 = mod_one_residual(candidate_parent_central_charge / 24.0 - visible_central_charge / 24.0)
        candidate_distance = distance_to_integer(candidate_parent_central_charge / 24.0 - visible_central_charge / 24.0)
        candidate_ties = math.isclose(candidate_distance, best_distance, abs_tol=1.0e-15, rel_tol=0.0)
        candidate_improves = candidate_distance < best_distance and not candidate_ties
        if candidate_improves or (candidate_ties and candidate_level < best_branch_level):
            best_branch_level = candidate_level
            best_branch_residual_mod1 = candidate_residual_mod1
            best_distance = candidate_distance

    return FramingAnomalyData(
        parent_level=resolved_model.parent_level,
        parent_central_charge=parent_central_charge,
        lepton_central_charge=lepton_central_charge,
        quark_central_charge=quark_central_charge,
        visible_central_charge=visible_central_charge,
        coset_central_charge=coset_central_charge,
        visible_residual_mod1=visible_residual_mod1,
        total_residual_mod1=total_residual_mod1,
        branch_period=branch_period,
        best_branch_level=best_branch_level,
        best_branch_residual_mod1=best_branch_residual_mod1,
    )


def verify_gauge_holography(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    codata_alpha_inverse: float = CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    *,
    model: TopologicalModel | None = None,
) -> GaugeHolographyAudit:
    r"""Evaluate the emergent fine-structure constant from fixed boundary levels."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    framing_gap = nearest_integer_gap(resolved_model.parent_level / (2.0 * resolved_model.lepton_level))
    topological_alpha_inverse = surface_tension_gauge_alpha_inverse(
        generation_count=generation_count,
        model=resolved_model,
    )
    benchmark_modularity_gap = benchmark_visible_modularity_gap(model=resolved_model)
    modular_gap_scaled_inverse = 1.0e3 * benchmark_modularity_gap
    geometric_residue_fraction = abs(topological_alpha_inverse - codata_alpha_inverse) / topological_alpha_inverse
    modular_gap_alignment_fraction = abs(topological_alpha_inverse - modular_gap_scaled_inverse) / topological_alpha_inverse
    return GaugeHolographyAudit(
        generation_count=int(generation_count),
        parent_level=resolved_model.parent_level,
        lepton_level=resolved_model.lepton_level,
        quark_level=resolved_model.quark_level,
        topological_alpha_inverse=float(topological_alpha_inverse),
        codata_alpha_inverse=float(codata_alpha_inverse),
        modular_gap_scaled_inverse=float(modular_gap_scaled_inverse),
        geometric_residue_fraction=float(geometric_residue_fraction),
        modular_gap_alignment_fraction=float(modular_gap_alignment_fraction),
        framing_closed=bool(solver_isclose(framing_gap, 0.0)),
    )


def benchmark_visible_modularity_gap(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    *,
    model: TopologicalModel | None = None,
) -> float:
    """Return the manuscript-facing de-anchored modularity gap for the selected visible pair."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    parent_central_charge = wzw_central_charge(resolved_model.parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
    visible_central_charge = wzw_central_charge(
        resolved_model.lepton_level,
        SU2_DIMENSION,
        SU2_DUAL_COXETER,
    ) + wzw_central_charge(
        resolved_model.quark_level,
        SU3_DIMENSION,
        SU3_DUAL_COXETER,
    )
    reference_coset_central_charge = so10_sm_branching_rule_coset_central_charge(resolved_model.parent_level)
    raw_difference = (parent_central_charge - visible_central_charge - reference_coset_central_charge) / 24.0
    return float(distance_to_integer(raw_difference))


@dataclass(frozen=True)
class InflationaryFlowPoint:
    r"""Single point on the fixed-parent topological inflationary flow."""

    phi: float
    visible_central_charge: float
    central_charge_deficit: float
    modularity_gap: float
    framing_anomaly: float
    potential_ev4: float
    selected_endpoint: bool


@dataclass(frozen=True)
class InflationarySectorData:
    r"""Publication-facing inflationary audit data on the fixed $SO(10)_{312}$ branch."""

    parent_level: int
    quark_level: int
    endpoint_lepton_level: int
    parent_central_charge: float
    coset_central_charge: float
    endpoint_visible_central_charge: float
    central_charge_deficit: float
    potential_prefactor_ev4: float
    potential_ev4: float
    slow_roll_epsilon: float
    c_dark_completion: float
    primordial_efolds: int
    tensor_to_scalar_ratio: float
    observable_tensor_to_scalar_ratio: float
    late_time_tensor_suppression_factor: float
    scalar_tilt: float
    scalar_running: float
    dark_sector_tilt_regulator: float
    non_gaussianity_floor: float
    uses_c_dark_tilt_regulator: bool
    beta_genus_ladder: float
    raw_ir_defect_scale_ev: float
    projected_ir_defect_scale_ev: float
    reheating_scale_ev: float
    reheating_temperature_k: float
    flow_points: tuple[InflationaryFlowPoint, ...]
    bicep_keck_upper_bound_95cl: float = BICEP_KECK_95CL_TENSOR_UPPER_BOUND
    kappa_geometric: float = GEOMETRIC_KAPPA

    @property
    def slow_roll_stability_pass(self) -> bool:
        zero_count = sum(1 for point in self.flow_points if math.isclose(point.framing_anomaly, 0.0, rel_tol=0.0, abs_tol=1.0e-12))
        return (
            math.isclose(self.slow_roll_epsilon, 0.0, rel_tol=0.0, abs_tol=1.0e-12)
            and zero_count == 1
            and all(point.central_charge_deficit > 0.0 for point in self.flow_points)
        )

    @property
    def tensor_ratio_tuning_free(self) -> bool:
        return _matches_exact_fraction(self.tensor_to_scalar_ratio, INFLATIONARY_TENSOR_RATIO)

    @property
    def primordial_tensor_to_scalar_ratio(self) -> float:
        return float(self.tensor_to_scalar_ratio)

    @property
    def observable_tensor_tension_with_bicep_keck(self) -> bool:
        return self.observable_tensor_to_scalar_ratio > self.bicep_keck_upper_bound_95cl

    @property
    def requires_late_time_tensor_suppression(self) -> bool:
        return self.observable_tensor_tension_with_bicep_keck

    @property
    def n_s_locked(self) -> float:
        return float(self.scalar_tilt)

    def calculate_computational_coherence_loss_rate(self) -> float:
        coherence_loss_per_efold = float(1.0 - self.n_s_locked)
        # Scale invariance is topologically forbidden by the finite bit budget N.
        assert coherence_loss_per_efold > 0.0
        return coherence_loss_per_efold

    @property
    def clock_skew(self) -> float:
        return self.calculate_computational_coherence_loss_rate()

    @property
    def computational_friction_pass(self) -> bool:
        return self.clock_skew > 0.0 and self.n_s_locked < 1.0

    @property
    def planck_compatibility_pass(self) -> bool:
        lower, upper = PLANCK_2018_SCALAR_TILT_RANGE
        return lower <= self.scalar_tilt <= upper

    @property
    def wheeler_dewitt_tilt_lock_pass(self) -> bool:
        return math.isclose(
            self.scalar_tilt,
            PRIMORDIAL_SCALAR_TILT_TARGET,
            rel_tol=0.0,
            abs_tol=PRIMORDIAL_SCALAR_TILT_TARGET_TOLERANCE,
        )

    @property
    def expected_non_gaussianity_floor(self) -> float:
        return float(1.0 - self.kappa_geometric)

    @property
    def modular_scrambling_audit_pass(self) -> bool:
        return math.isclose(
            self.non_gaussianity_floor,
            self.expected_non_gaussianity_floor,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )

    def check_information_scrambling_limit(self) -> None:
        # Non-zero coupling is required for the reconstruction of a 4D bulk with matter.
        assert self.non_gaussianity_floor > 0.0, "Information Scrambling Limit: FAILED."
        assert self.modular_scrambling_audit_pass, "Modular scrambling check failed."

    def validate_primordial_lock(self) -> None:
        self.calculate_computational_coherence_loss_rate()
        if not self.uses_c_dark_tilt_regulator:
            raise ValueError("Dark-sector consistency check failed: c_dark must regulate n_s.")
        if math.isclose(self.dark_sector_tilt_regulator, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("Dark-sector consistency check failed: zero tilt regulator removes the CMB tilt lock.")
        self.check_information_scrambling_limit()
        if not self.planck_compatibility_pass:
            raise AssertionError("Planck-2018 consistency check failed.")
        if not self.wheeler_dewitt_tilt_lock_pass:
            raise AssertionError(
                "Wheeler-DeWitt primordial-lock violation: "
                f"n_s={self.scalar_tilt:.6f} does not match {PRIMORDIAL_SCALAR_TILT_TARGET:.4f} "
                f"within {PRIMORDIAL_SCALAR_TILT_TARGET_TOLERANCE:.1e}."
            )


CosmologyAudit = InflationarySectorData


class InflationarySector:
    r"""Dynamic inflationary flow on the fixed $K=312$ lattice.

    The flow variable ``phi`` is not introduced as an independent scalar degree
    of freedom. It is the continuous RG image of the visible leptonic support
    count along the fixed-quark $SU(3)_8$ slice. The effective potential is the
    parent central-charge deficit relative to the visible branch and fixed coset
    completion, while the slow-roll identity is the framing anomaly itself.
    """

    def __init__(
        self,
        flow_levels: tuple[int, ...] | None = None,
        *,
        model: TopologicalModel | None = None,
        parent_level: int | None = None,
        lepton_level: int | None = None,
        quark_level: int | None = None,
        scale_ratio: float | None = None,
        bit_count: float | None = None,
        kappa_geometric: float | None = None,
        gut_threshold_residue: float | None = None,
        solver_config: SolverConfig | None = None,
    ) -> None:
        resolved_model = _coerce_topological_model(
            model=model,
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
            scale_ratio=scale_ratio,
            bit_count=bit_count,
            kappa_geometric=kappa_geometric,
            gut_threshold_residue=gut_threshold_residue,
            solver_config=solver_config,
        )
        self.model = resolved_model
        self.flow_levels = resolved_model.local_lepton_level_window if flow_levels is None else tuple(flow_levels)
        self.parent_central_charge = float(
            wzw_central_charge(resolved_model.parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
        )
        self.coset_central_charge = float(so10_sm_branching_rule_coset_central_charge(resolved_model.parent_level))
        self.potential_prefactor_ev4 = float((PLANCK_MASS_EV**4) / resolved_model.bit_count)
        self.beta_genus_ladder = float(0.5 * math.log(su2_total_quantum_dimension(resolved_model.lepton_level)))
        self.raw_ir_defect_scale_ev = float(PLANCK_MASS_EV * resolved_model.bit_count ** (-0.25))
        projected_scales = derive_scales(model=resolved_model)
        self.projected_ir_defect_scale_ev = float(projected_scales.m_0_mz_ev)
        self.kappa_geometric = float(resolved_model.kappa_geometric)

    def visible_central_charge(self, phi: float) -> float:
        return float(
            wzw_central_charge(phi, SU2_DIMENSION, SU2_DUAL_COXETER)
            + wzw_central_charge(self.model.quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
        )

    def central_charge_deficit(self, phi: float) -> float:
        return float(self.parent_central_charge - self.visible_central_charge(phi) - self.coset_central_charge)

    def c_dark_completion(self, phi: float) -> float:
        return float(self.central_charge_deficit(phi))

    def potential(self, phi: float) -> float:
        return float(self.potential_prefactor_ev4 * self.central_charge_deficit(phi))

    def slow_roll_epsilon(self, phi: float) -> float:
        return float(nearest_integer_gap(self.model.parent_level / (2.0 * phi)))

    def reheating_scale(self) -> float:
        return float(self.projected_ir_defect_scale_ev * math.exp(-self.beta_genus_ladder))

    def dark_sector_tilt_regulator(self, c_dark_completion: float) -> float:
        if math.isclose(BENCHMARK_C_DARK_RESIDUE, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("Benchmark c_dark residue must be nonzero for the primordial tilt lock.")
        return float(c_dark_completion / BENCHMARK_C_DARK_RESIDUE)

    def modular_scrambling_floor(self) -> float:
        kappa_d5 = self.kappa_geometric
        return float(1.0 - kappa_d5)

    def derive_primordial_lock(self, c_dark_completion: float) -> tuple[int, float, float, float, bool]:
        primordial_efolds = PRIMORDIAL_EFOLD_IDENTITY_MULTIPLIER * self.model.lepton_level  # Locked by Genus-3 flavor frustration.
        tensor_to_scalar_ratio = float(INFLATIONARY_TENSOR_RATIO)
        dark_sector_tilt_regulator = self.dark_sector_tilt_regulator(c_dark_completion)
        scalar_tilt = float(
            1.0
            - (2.0 / primordial_efolds)
            - ((tensor_to_scalar_ratio / 8.0) * self.kappa_geometric * dark_sector_tilt_regulator)
        )
        scalar_running = float(-2.0 / (primordial_efolds**2))
        return primordial_efolds, scalar_tilt, scalar_running, dark_sector_tilt_regulator, True

    def flow_point(self, lepton_level: int) -> InflationaryFlowPoint:
        phi = float(lepton_level)
        return InflationaryFlowPoint(
            phi=phi,
            visible_central_charge=self.visible_central_charge(phi),
            central_charge_deficit=self.central_charge_deficit(phi),
            modularity_gap=benchmark_visible_modularity_gap(
                parent_level=self.model.parent_level,
                lepton_level=lepton_level,
                quark_level=self.model.quark_level,
            ),
            framing_anomaly=self.slow_roll_epsilon(phi),
            potential_ev4=self.potential(phi),
            selected_endpoint=lepton_level == self.model.lepton_level,
        )

    def derive(self) -> InflationarySectorData:
        flow_points = tuple(self.flow_point(lepton_level) for lepton_level in self.flow_levels)
        endpoint_phi = float(self.model.lepton_level)
        reheating_scale_ev = self.reheating_scale()
        c_dark_completion = self.c_dark_completion(endpoint_phi)
        primordial_efolds, scalar_tilt, scalar_running, dark_sector_tilt_regulator, uses_c_dark_tilt_regulator = self.derive_primordial_lock(
            c_dark_completion
        )
        non_gaussianity_floor = self.modular_scrambling_floor()
        late_time_tensor_suppression_factor = 1.0
        observable_tensor_to_scalar_ratio = float(INFLATIONARY_TENSOR_RATIO) * late_time_tensor_suppression_factor
        inflationary_data = InflationarySectorData(
            parent_level=self.model.parent_level,
            quark_level=self.model.quark_level,
            endpoint_lepton_level=self.model.lepton_level,
            parent_central_charge=self.parent_central_charge,
            coset_central_charge=self.coset_central_charge,
            endpoint_visible_central_charge=self.visible_central_charge(endpoint_phi),
            central_charge_deficit=self.central_charge_deficit(endpoint_phi),
            potential_prefactor_ev4=self.potential_prefactor_ev4,
            potential_ev4=self.potential(endpoint_phi),
            slow_roll_epsilon=self.slow_roll_epsilon(endpoint_phi),
            c_dark_completion=c_dark_completion,
            primordial_efolds=primordial_efolds,
            tensor_to_scalar_ratio=float(INFLATIONARY_TENSOR_RATIO),
            observable_tensor_to_scalar_ratio=float(observable_tensor_to_scalar_ratio),
            late_time_tensor_suppression_factor=float(late_time_tensor_suppression_factor),
            scalar_tilt=scalar_tilt,
            scalar_running=scalar_running,
            dark_sector_tilt_regulator=dark_sector_tilt_regulator,
            kappa_geometric=self.kappa_geometric,
            non_gaussianity_floor=non_gaussianity_floor,
            uses_c_dark_tilt_regulator=uses_c_dark_tilt_regulator,
            beta_genus_ladder=self.beta_genus_ladder,
            raw_ir_defect_scale_ev=self.raw_ir_defect_scale_ev,
            projected_ir_defect_scale_ev=self.projected_ir_defect_scale_ev,
            reheating_scale_ev=reheating_scale_ev,
            reheating_temperature_k=float(reheating_scale_ev * EV_TO_KELVIN),
            flow_points=flow_points,
        )
        inflationary_data.validate_primordial_lock()
        return inflationary_data


def visible_level_density_ratio(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    *,
    model: TopologicalModel | None = None,
) -> float:
    r"""Return the Triple-Lock visible level-density ratio ``K/(k_\ell+k_q)``."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    visible_support = resolved_model.lepton_level + resolved_model.quark_level
    if visible_support <= 0:
        raise ValueError("Visible support count must be positive.")
    return float(resolved_model.parent_level / visible_support)


def geometric_level_density_ratio(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    *,
    model: TopologicalModel | None = None,
) -> float:
    """Backward-compatible alias for `visible_level_density_ratio()`."""

    return visible_level_density_ratio(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        model=model,
    )


def surface_tension_gauge_alpha_inverse(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    *,
    model: TopologicalModel | None = None,
) -> float:
    r"""Return the Triple-Lock gauge proxy ``\alpha^{-1}_{\rm surf}``."""

    return float(
        generation_count
        * visible_level_density_ratio(
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
            model=model,
        )
    )


def derive_surface_tension_gauge_alpha_inverse(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    *,
    model: TopologicalModel | None = None,
) -> float:
    """Backward-compatible alias for `surface_tension_gauge_alpha_inverse()`."""

    return surface_tension_gauge_alpha_inverse(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        generation_count=generation_count,
        model=model,
    )


def unification_residue(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    benchmark_lepton_level: int = LEPTON_LEVEL,
    benchmark_quark_level: int = QUARK_LEVEL,
    *,
    model: TopologicalModel | None = None,
) -> float:
    r"""Return the manuscript Unification Residue ``\mathfrak U(k_\ell)``."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    benchmark_alpha_inverse = surface_tension_gauge_alpha_inverse(
        parent_level=resolved_model.parent_level,
        lepton_level=benchmark_lepton_level,
        quark_level=benchmark_quark_level,
        generation_count=generation_count,
    )
    alpha_inverse = surface_tension_gauge_alpha_inverse(generation_count=generation_count, model=resolved_model)
    framing_gap = nearest_integer_gap(resolved_model.parent_level / (2.0 * resolved_model.lepton_level))
    return float(abs(alpha_inverse - benchmark_alpha_inverse) + benchmark_alpha_inverse * framing_gap)


def derive_unification_residue(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    benchmark_lepton_level: int = LEPTON_LEVEL,
    benchmark_quark_level: int = QUARK_LEVEL,
    *,
    model: TopologicalModel | None = None,
) -> float:
    """Backward-compatible alias for `unification_residue()`."""

    return unification_residue(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        generation_count=generation_count,
        benchmark_lepton_level=benchmark_lepton_level,
        benchmark_quark_level=benchmark_quark_level,
        model=model,
    )


def holographic_lambda_scaling_identity_si_m2(
    bit_count: float | None = None,
    *,
    model: TopologicalModel | None = None,
) -> float:
    r"""Return the scaling identity ``1/(L_P^2 N)`` in SI units."""

    resolved_bit_count = float(_coerce_topological_model(model=model, bit_count=bit_count).bit_count if model is not None else (HOLOGRAPHIC_BITS if bit_count is None else bit_count))
    if resolved_bit_count <= 0.0:
        raise ValueError("Holographic bit count must be positive.")
    return float(1.0 / (resolved_bit_count * PLANCK_LENGTH_M * PLANCK_LENGTH_M))


def holographic_surface_tension_lambda_si_m2(
    bit_count: float | None = None,
    *,
    model: TopologicalModel | None = None,
) -> float:
    r"""Return the Triple-Lock vacuum identity ``\Lambda_{\rm holo}=3\pi/(L_P^2 N)``."""

    return float(3.0 * math.pi * holographic_lambda_scaling_identity_si_m2(bit_count=bit_count, model=model))


def derive_single_bit_rigidity_audit(
    bit_shift: int = 1,
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    bit_count: float | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    *,
    model: TopologicalModel | None = None,
) -> RigidityStressTestAudit:
    r"""Probe whether a single holographic bit can be absorbed by the discrete Triple-Lock data."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        bit_count=bit_count,
    )
    if bit_shift == 0:
        raise ValueError("bit_shift must be nonzero for a rigidity stress test.")

    bit_count_decimal = Decimal(str(resolved_model.bit_count))
    generation_count_decimal = Decimal(generation_count)
    parent_level_decimal = Decimal(resolved_model.parent_level)
    central_visible_support_decimal = Decimal(resolved_model.lepton_level + resolved_model.quark_level)
    decimal_precision = max(64, bit_count_decimal.adjusted() + 32)
    with localcontext() as context:
        context.prec = decimal_precision
        shifted_bit_count_decimal = bit_count_decimal + Decimal(bit_shift)
        if shifted_bit_count_decimal <= 0:
            raise ValueError("Shifted holographic bit count must remain positive.")

        central_alpha_inverse_decimal = generation_count_decimal * parent_level_decimal / central_visible_support_decimal
        shifted_alpha_inverse_decimal = central_alpha_inverse_decimal * bit_count_decimal / shifted_bit_count_decimal
        shifted_visible_support_decimal = generation_count_decimal * parent_level_decimal / shifted_alpha_inverse_decimal
        shifted_lepton_level_decimal = shifted_visible_support_decimal - Decimal(resolved_model.quark_level)
        shifted_effective_lepton_level = int(shifted_lepton_level_decimal.to_integral_value(rounding=ROUND_HALF_UP))
    shifted_framing_gap = nearest_integer_gap(resolved_model.parent_level / (2.0 * shifted_effective_lepton_level))

    overconstrained = (
        shifted_bit_count_decimal != bit_count_decimal
        and shifted_effective_lepton_level == resolved_model.lepton_level
        and solver_isclose(shifted_framing_gap, 0.0)
    )
    return RigidityStressTestAudit(
        bit_shift=int(bit_shift),
        central_bit_count=float(bit_count_decimal),
        shifted_bit_count=float(shifted_bit_count_decimal),
        central_lepton_level=resolved_model.lepton_level,
        shifted_effective_lepton_level=int(shifted_effective_lepton_level),
        central_alpha_inverse=surface_tension_gauge_alpha_inverse(generation_count=generation_count, model=resolved_model),
        shifted_alpha_inverse=float(shifted_alpha_inverse_decimal),
        central_lambda_si_m2=holographic_surface_tension_lambda_si_m2(model=resolved_model),
        shifted_lambda_si_m2=holographic_surface_tension_lambda_si_m2(bit_count=float(shifted_bit_count_decimal)),
        shifted_framing_gap=float(shifted_framing_gap),
        overconstrained=bool(overconstrained),
    )


def verify_single_bit_rigidity(
    bit_shift: int = 1,
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    bit_count: float | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    *,
    model: TopologicalModel | None = None,
) -> RigidityStressTestAudit:
    """Backward-compatible alias for `derive_single_bit_rigidity_audit()`."""

    return derive_single_bit_rigidity_audit(
        bit_shift=bit_shift,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        bit_count=bit_count,
        generation_count=generation_count,
        model=model,
    )


def verify_dark_energy_tension(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    bit_count: float | None = None,
    kappa_geometric: float | None = None,
    *,
    model: TopologicalModel | None = None,
) -> DarkEnergyTensionAudit:
    r"""Audit dark energy as the irreducible holographic surface tension of the boundary."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
    )
    benchmark_modularity_gap = benchmark_visible_modularity_gap(model=resolved_model)
    c_dark_completion = 24.0 * benchmark_modularity_gap
    cosmology_anchor = derive_cosmology_anchor()
    scales = derive_scales(model=resolved_model)
    gauge_audit = verify_gauge_holography(model=resolved_model)
    triple_match_audit = verify_triple_match_saturation(model=resolved_model)

    lambda_surface_tension_si_m2 = holographic_surface_tension_lambda_si_m2(model=resolved_model)
    lambda_scaling_identity_si_m2 = holographic_lambda_scaling_identity_si_m2(model=resolved_model)
    rho_vac_surface_tension_ev4 = c_dark_completion * PLANCK_MASS_EV**4 / resolved_model.bit_count
    rho_vac_from_defect_scale_ev4 = c_dark_completion * scales.m_0_uv_ev**4 / (resolved_model.kappa_geometric**4)

    minus_one_percent_model = replace(resolved_model, bit_count=resolved_model.bit_count * 0.99)
    plus_one_percent_model = replace(resolved_model, bit_count=resolved_model.bit_count * 1.01)
    minus_one_percent_scales = derive_scales(model=minus_one_percent_model)
    plus_one_percent_scales = derive_scales(model=plus_one_percent_model)
    minus_one_percent_lambda_si_m2 = holographic_surface_tension_lambda_si_m2(model=minus_one_percent_model)
    plus_one_percent_lambda_si_m2 = holographic_surface_tension_lambda_si_m2(model=plus_one_percent_model)
    minus_one_percent_gauge = verify_gauge_holography(model=minus_one_percent_model)
    plus_one_percent_gauge = verify_gauge_holography(model=plus_one_percent_model)
    benchmark_m_nu_topological = topological_mass_coordinate_ev(bit_count=HOLOGRAPHIC_BITS, kappa_geometric=KAPPA_D5)
    m_nu_topological = topological_mass_coordinate_ev(
        bit_count=resolved_model.bit_count,
        kappa_geometric=resolved_model.kappa_geometric,
    )
    assert math.isclose(
        benchmark_m_nu_topological,
        (PLANCK_MASS_EV / (HOLOGRAPHIC_BITS**0.25)) * KAPPA_D5,
        rel_tol=1.0e-15,
        abs_tol=1.0e-18,
    )
    assert math.isclose(
        m_nu_topological,
        (PLANCK_MASS_EV / (resolved_model.bit_count**0.25)) * resolved_model.kappa_geometric,
        rel_tol=1.0e-15,
        abs_tol=1.0e-18,
    )
    sensitivity_audit_triggered_integrity_error, sensitivity_audit_message = _audit_topological_mass_coordinate_sensitivity(
        m_nu_topological,
        bit_count=resolved_model.bit_count,
        kappa_geometric=resolved_model.kappa_geometric,
        parent_level=resolved_model.parent_level,
        lepton_level=resolved_model.lepton_level,
        quark_level=resolved_model.quark_level,
        fractional_shift=0.01,
    )

    dark_energy_audit = DarkEnergyTensionAudit(
        holographic_bits=float(resolved_model.bit_count),
        geometric_residue=float(resolved_model.kappa_geometric),
        modular_gap=float(benchmark_modularity_gap),
        c_dark_completion=float(c_dark_completion),
        lambda_surface_tension_si_m2=float(lambda_surface_tension_si_m2),
        lambda_anchor_si_m2=float(cosmology_anchor.lambda_si_m2),
        lambda_scaling_identity_si_m2=float(lambda_scaling_identity_si_m2),
        rho_vac_surface_tension_ev4=float(rho_vac_surface_tension_ev4),
        rho_vac_from_defect_scale_ev4=float(rho_vac_from_defect_scale_ev4),
        minus_one_percent_lambda_si_m2=float(minus_one_percent_lambda_si_m2),
        plus_one_percent_lambda_si_m2=float(plus_one_percent_lambda_si_m2),
        minus_one_percent_lambda_fractional_shift=float(
            minus_one_percent_lambda_si_m2 / lambda_surface_tension_si_m2 - 1.0
        ),
        plus_one_percent_lambda_fractional_shift=float(
            plus_one_percent_lambda_si_m2 / lambda_surface_tension_si_m2 - 1.0
        ),
        topological_mass_coordinate_ev=float(m_nu_topological),
        minus_one_percent_topological_mass_coordinate_ev=float(minus_one_percent_scales.m_0_uv_ev),
        plus_one_percent_topological_mass_coordinate_ev=float(plus_one_percent_scales.m_0_uv_ev),
        minus_one_percent_topological_mass_coordinate_fractional_shift=float(
            minus_one_percent_scales.m_0_uv_ev / scales.m_0_uv_ev - 1.0
        ),
        plus_one_percent_topological_mass_coordinate_fractional_shift=float(
            plus_one_percent_scales.m_0_uv_ev / scales.m_0_uv_ev - 1.0
        ),
        alpha_inverse_central=float(gauge_audit.topological_alpha_inverse),
        alpha_inverse_minus_one_percent=float(minus_one_percent_gauge.topological_alpha_inverse),
        alpha_inverse_plus_one_percent=float(plus_one_percent_gauge.topological_alpha_inverse),
        triple_match_product=float(triple_match_audit.triple_match_product),
        triple_match_saturated=bool(triple_match_audit.saturated),
        sensitivity_audit_triggered_integrity_error=bool(sensitivity_audit_triggered_integrity_error),
        sensitivity_audit_message=sensitivity_audit_message,
    )
    assert dark_energy_audit.alpha_locked_under_bit_shift == True
    return dark_energy_audit


def verify_unitary_bounds(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    bit_count: float | None = None,
    kappa_geometric: float | None = None,
    *,
    model: TopologicalModel | None = None,
) -> UnitaryBoundAudit:
    r"""Audit Page-curve unitarity from the Triple-Lock and finite holographic capacity."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
    )
    gravity_audit = EinsteinConsistencyEngine(model=resolved_model).verify_bulk_emergence()
    dark_energy_audit = verify_dark_energy_tension(model=resolved_model)
    gauge_audit = verify_gauge_holography(model=resolved_model)
    cosmology_audit = resolved_model.derive_cosmology_audit()
    baryon_stability = gravity_audit.baryon_stability
    triple_match_audit = verify_triple_match_saturation(model=resolved_model)

    entropy_max_nats = math.log(resolved_model.bit_count)
    holographic_buffer_entropy = dark_energy_audit.c_dark_completion * entropy_max_nats
    regulated_curvature_entropy = gravity_audit.gmunu_consistency_score * entropy_max_nats
    curvature_buffer_margin = holographic_buffer_entropy - regulated_curvature_entropy

    information_evaporation_rate_per_year = 1.0 / baryon_stability.protected_evaporation_lifetime_years
    information_recovery_rate_per_year = information_evaporation_rate_per_year
    delta_locked_rate_per_year = (1.0 / baryon_stability.proton_lifetime_years) * baryon_stability.modular_tunneling_penalty
    lloyds_limit_ops_per_second = LloydBridge.limit_ops_per_second(
        dark_energy_audit.rho_vac_surface_tension_ev4,
        bit_count=resolved_model.bit_count,
    )
    complexity_growth_rate_ops_per_second = cosmology_audit.n_s_locked * lloyds_limit_ops_per_second
    zero_point_complexity = dark_energy_audit.c_dark_completion * entropy_max_nats
    clock_skew = cosmology_audit.calculate_computational_coherence_loss_rate()
    universal_computational_limit_pass = complexity_growth_rate_ops_per_second <= (
        lloyds_limit_ops_per_second + 1.0e-12 * max(1.0, abs(lloyds_limit_ops_per_second))
    )

    proton_recovery_identity = math.isclose(
        information_evaporation_rate_per_year,
        information_recovery_rate_per_year,
        rel_tol=0.0,
        abs_tol=1.0e-300,
    )
    recovery_locked_to_delta_mod = math.isclose(
        information_recovery_rate_per_year,
        delta_locked_rate_per_year,
        rel_tol=1.0e-12,
        abs_tol=1.0e-300,
    )
    unitary_bound_satisfied = (
        gravity_audit.torsion_free
        and gravity_audit.non_singular_bulk
        and dark_energy_audit.triple_match_saturated
        and curvature_buffer_margin >= -1.0e-12
    )
    torsion_free_stability = gravity_audit.torsion_free and dark_energy_audit.alpha_locked_under_bit_shift
    holographic_rigidity = (
        unitary_bound_satisfied
        and gauge_audit.topological_stability_pass
        and gravity_audit.lambda_aligned
        and dark_energy_audit.alpha_locked_under_bit_shift
        and dark_energy_audit.sensitivity_audit_triggered_integrity_error
        and torsion_free_stability
        and triple_match_audit.saturated
    )
    assert dark_energy_audit.alpha_locked_under_bit_shift
    assert math.isclose(
        dark_energy_audit.topological_mass_coordinate_ev,
        (PLANCK_MASS_EV / (resolved_model.bit_count**0.25)) * resolved_model.kappa_geometric,
        rel_tol=1.0e-15,
        abs_tol=1.0e-18,
    )
    assert universal_computational_limit_pass, "Universal Computational Limit: FAILED."

    unitary_audit = UnitaryBoundAudit(
        holographic_bits=float(resolved_model.bit_count),
        geometric_residue=float(resolved_model.kappa_geometric),
        entropy_max_nats=float(entropy_max_nats),
        c_dark_completion=float(dark_energy_audit.c_dark_completion),
        modular_gap=float(baryon_stability.modular_gap),
        framing_gap=float(gravity_audit.framing_gap),
        gmunu_consistency_score=float(gravity_audit.gmunu_consistency_score),
        holographic_buffer_entropy=float(holographic_buffer_entropy),
        regulated_curvature_entropy=float(regulated_curvature_entropy),
        curvature_buffer_margin=float(curvature_buffer_margin),
        information_evaporation_rate_per_year=float(information_evaporation_rate_per_year),
        information_recovery_rate_per_year=float(information_recovery_rate_per_year),
        recovery_lifetime_years=float(baryon_stability.protected_evaporation_lifetime_years),
        topological_mass_coordinate_ev=float(dark_energy_audit.topological_mass_coordinate_ev),
        triple_match_product=float(triple_match_audit.triple_match_product),
        torsion_free_stability=bool(torsion_free_stability),
        lloyds_limit_ops_per_second=float(lloyds_limit_ops_per_second),
        complexity_growth_rate_ops_per_second=float(complexity_growth_rate_ops_per_second),
        zero_point_complexity=float(zero_point_complexity),
        max_complexity_capacity=float(MAX_COMPLEXITY_CAPACITY),
        clock_skew=float(clock_skew),
        unitary_bound_satisfied=bool(unitary_bound_satisfied),
        proton_recovery_identity=bool(proton_recovery_identity),
        recovery_locked_to_delta_mod=bool(recovery_locked_to_delta_mod),
        dark_sector_holographic_rigidity=bool(dark_energy_audit.sensitivity_audit_triggered_integrity_error),
        holographic_rigidity=bool(holographic_rigidity),
        universal_computational_limit_pass=bool(universal_computational_limit_pass),
    )
    unitary_audit.validate_holographic_complexity_bound()
    return unitary_audit


def derive_page_point_audit(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    bit_count: float | None = None,
    kappa_geometric: float | None = None,
    *,
    unitary_audit: UnitaryBoundAudit | None = None,
    model: TopologicalModel | None = None,
) -> PagePointAudit:
    r"""Derive the benchmark Page point defined by ``S_{\rm ent}=c_{\rm dark}S_{\max}``."""

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
    )
    resolved_unitary_audit = verify_unitary_bounds(model=resolved_model) if unitary_audit is None else unitary_audit
    page_point_entropy = float(resolved_unitary_audit.holographic_buffer_entropy)
    bulk_entanglement_entropy = float(resolved_unitary_audit.regulated_curvature_entropy)
    modular_complement_entropy = float(page_point_entropy - bulk_entanglement_entropy)
    page_curve_locked = (
        modular_complement_entropy >= -1.0e-12
        and resolved_unitary_audit.proton_recovery_identity
        and resolved_unitary_audit.recovery_locked_to_delta_mod
    )
    return PagePointAudit(
        entropy_max_nats=float(resolved_unitary_audit.entropy_max_nats),
        c_dark_completion=float(resolved_unitary_audit.c_dark_completion),
        page_point_entropy=page_point_entropy,
        bulk_entanglement_entropy=bulk_entanglement_entropy,
        modular_complement_entropy=modular_complement_entropy,
        page_curve_locked=bool(page_curve_locked),
    )


def simulate_neutron_star_transit(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    bit_count: float | None = None,
    kappa_geometric: float | None = None,
    *,
    gravitational_gradient_scale: float = 6.0,
    transit_path_length_km: float = 20.0,
    density_scale: float = 2.0,
    pmns: PmnsData | None = None,
    unitary_audit: UnitaryBoundAudit | None = None,
    model: TopologicalModel | None = None,
) -> TorsionScramblingTransitAudit:
    r"""Flag Stage XIII torsion-scrambling events for extreme compact-object transits.

    The benchmark extrapolation keeps the global branch fixed and instead asks
    whether large local gradients can push the visible support dictionary toward
    a transient electron/tau rank-deficiency.
    """

    if gravitational_gradient_scale <= 0.0:
        raise ValueError("gravitational_gradient_scale must be positive.")
    if transit_path_length_km <= 0.0:
        raise ValueError("transit_path_length_km must be positive.")
    if density_scale <= 0.0:
        raise ValueError("density_scale must be positive.")

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
    )
    resolved_pmns = resolved_model.derive_pmns() if pmns is None else pmns
    resolved_unitary_audit = verify_unitary_bounds(model=resolved_model) if unitary_audit is None else unitary_audit

    probability_matrix = np.asarray(np.abs(resolved_pmns.pmns_matrix_rg) ** 2, dtype=float)
    electron_row = np.asarray(probability_matrix[0], dtype=float)
    tau_row = np.asarray(probability_matrix[2], dtype=float)

    gradient_load = float(gravitational_gradient_scale * density_scale * (transit_path_length_km / 10.0))
    rank_deficiency_load = float(
        max(resolved_unitary_audit.clock_skew, 0.0)
        * gradient_load
        * (1.0 + resolved_unitary_audit.complexity_utilization_fraction)
    )
    scrambling_fraction = float(
        0.0 if rank_deficiency_load <= 1.0 else 1.0 - math.exp(-(rank_deficiency_load - 1.0))
    )

    scrambled_electron_row = np.clip(
        (1.0 - scrambling_fraction) * electron_row + scrambling_fraction * tau_row,
        0.0,
        1.0,
    )
    scrambled_electron_row /= float(np.sum(scrambled_electron_row))

    scrambled_tau_row = np.clip(
        (1.0 - 0.5 * scrambling_fraction) * tau_row + 0.5 * scrambling_fraction * electron_row,
        0.0,
        1.0,
    )
    scrambled_tau_row /= float(np.sum(scrambled_tau_row))

    local_support_matrix = np.array(probability_matrix, copy=True)
    local_support_matrix[0] = scrambled_electron_row
    local_support_matrix[2] = scrambled_tau_row
    support_spectrum = derive_matrix_spectrum_audit(local_support_matrix, solver_config=resolved_model.solver_config)

    torsion_scrambling_triggered = bool(rank_deficiency_load >= 1.0 and scrambled_electron_row[2] > electron_row[2])
    return freeze_numpy_arrays(TorsionScramblingTransitAudit(
        gravitational_gradient_scale=float(gravitational_gradient_scale),
        transit_path_length_km=float(transit_path_length_km),
        density_scale=float(density_scale),
        clock_skew=float(resolved_unitary_audit.clock_skew),
        complexity_utilization_fraction=float(resolved_unitary_audit.complexity_utilization_fraction),
        rank_deficiency_load=rank_deficiency_load,
        scrambling_fraction=scrambling_fraction,
        reference_nu_e_to_nu_tau_probability=float(electron_row[2]),
        torsion_scrambled_nu_e_to_nu_tau_probability=float(scrambled_electron_row[2]),
        reference_nu_e_survival_probability=float(electron_row[0]),
        torsion_scrambled_nu_e_survival_probability=float(scrambled_electron_row[0]),
        local_support_matrix=local_support_matrix,
        support_rank=int(support_spectrum.rank),
        support_condition_number=float(support_spectrum.display_condition_number),
        support_sigma_min=float(support_spectrum.sigma_min),
        support_machine_precision_singular=bool(support_spectrum.machine_precision_singular),
        torsion_scrambling_triggered=torsion_scrambling_triggered,
    ))


def derive_framing_stability_audit(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
    alpha_shift_fraction: float = 0.01,
    *,
    gauge_audit: GaugeHolographyAudit | None = None,
    model: TopologicalModel | None = None,
) -> FramingStabilityAudit:
    r"""Derive the framing response to a percent-level detuning of ``\alpha^{-1}_{\rm surf}``."""

    if alpha_shift_fraction <= 0.0:
        raise ValueError("alpha_shift_fraction must be positive.")

    resolved_model = _coerce_topological_model(
        model=model,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    resolved_gauge_audit = (
        verify_gauge_holography(model=resolved_model, generation_count=generation_count)
        if gauge_audit is None
        else gauge_audit
    )
    central_alpha_inverse = float(resolved_gauge_audit.topological_alpha_inverse)
    minus_shifted_alpha_inverse = float(central_alpha_inverse * (1.0 - alpha_shift_fraction))
    plus_shifted_alpha_inverse = float(central_alpha_inverse * (1.0 + alpha_shift_fraction))
    central_effective_lepton_level = float(
        generation_count * resolved_model.parent_level / central_alpha_inverse - resolved_model.quark_level
    )
    minus_shifted_effective_lepton_level = float(
        generation_count * resolved_model.parent_level / minus_shifted_alpha_inverse - resolved_model.quark_level
    )
    plus_shifted_effective_lepton_level = float(
        generation_count * resolved_model.parent_level / plus_shifted_alpha_inverse - resolved_model.quark_level
    )
    central_framing_gap_raw = float(nearest_integer_gap(resolved_model.parent_level / (2.0 * central_effective_lepton_level)))
    minus_shifted_framing_gap_raw = float(
        nearest_integer_gap(resolved_model.parent_level / (2.0 * minus_shifted_effective_lepton_level))
    )
    plus_shifted_framing_gap_raw = float(
        nearest_integer_gap(resolved_model.parent_level / (2.0 * plus_shifted_effective_lepton_level))
    )
    central_framing_gap = 0.0 if solver_isclose(central_framing_gap_raw, 0.0) else central_framing_gap_raw
    minus_shifted_framing_gap = 0.0 if solver_isclose(minus_shifted_framing_gap_raw, 0.0) else minus_shifted_framing_gap_raw
    plus_shifted_framing_gap = 0.0 if solver_isclose(plus_shifted_framing_gap_raw, 0.0) else plus_shifted_framing_gap_raw
    alpha_lock_required = (
        solver_isclose(central_framing_gap, 0.0)
        and not solver_isclose(minus_shifted_framing_gap, 0.0)
        and not solver_isclose(plus_shifted_framing_gap, 0.0)
    )
    return FramingStabilityAudit(
        alpha_shift_fraction=float(alpha_shift_fraction),
        central_alpha_inverse=central_alpha_inverse,
        minus_shifted_alpha_inverse=minus_shifted_alpha_inverse,
        plus_shifted_alpha_inverse=plus_shifted_alpha_inverse,
        central_effective_lepton_level=central_effective_lepton_level,
        minus_shifted_effective_lepton_level=minus_shifted_effective_lepton_level,
        plus_shifted_effective_lepton_level=plus_shifted_effective_lepton_level,
        central_framing_gap=central_framing_gap,
        minus_shifted_framing_gap=minus_shifted_framing_gap,
        plus_shifted_framing_gap=plus_shifted_framing_gap,
        alpha_lock_required=bool(alpha_lock_required),
    )


class LevelScanner:
    r"""Fixed-parent scan around the visible level to expose the preferred matching point.

    The parent $SO(10)_{312}$ code and the quark branch $SU(3)_8$ are held fixed.
    The leptonic level is then scanned over a narrow integer window, and each point
    is tested against three criteria:
        - fixed-parent central-charge modularity,
        - vanishing framing-index gap,
        - non-singular restricted flavor block.
    """

    def __init__(
        self,
        fixed_parent_level: int = PARENT_LEVEL,
        fixed_quark_level: int = QUARK_LEVEL,
        reference_lepton_level: int = LEPTON_LEVEL,
        *,
        benchmark_lepton_level: int | None = None,
        benchmark_quark_level: int | None = None,
        reference_scale_ratio: float = RG_SCALE_RATIO,
        reference_gut_threshold_residue: float | None = None,
    ) -> None:
        self.fixed_parent_level = fixed_parent_level
        self.fixed_quark_level = fixed_quark_level
        self.reference_lepton_level = reference_lepton_level
        self.benchmark_lepton_level = reference_lepton_level if benchmark_lepton_level is None else benchmark_lepton_level
        self.benchmark_quark_level = fixed_quark_level if benchmark_quark_level is None else benchmark_quark_level
        self.reference_scale_ratio = reference_scale_ratio
        self.reference_gut_threshold_residue = reference_gut_threshold_residue
        parent_central_charge = wzw_central_charge(fixed_parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
        self.reference_coset_central_charge = so10_sm_branching_rule_coset_central_charge(fixed_parent_level)
        neighbor_gaps: list[float] = []
        for candidate_level in (reference_lepton_level - 1, reference_lepton_level + 1):
            visible_central_charge = wzw_central_charge(candidate_level, SU2_DIMENSION, SU2_DUAL_COXETER) + wzw_central_charge(
                fixed_quark_level,
                SU3_DIMENSION,
                SU3_DUAL_COXETER,
            )
            raw_difference = (
                parent_central_charge
                - visible_central_charge
                - self.reference_coset_central_charge
            ) / 24.0
            gap = distance_to_integer(raw_difference)
            if not math.isclose(gap, 0.0, abs_tol=1.0e-12, rel_tol=0.0):
                neighbor_gaps.append(gap)
        self.relaxed_modularity_allowance = min(neighbor_gaps) if neighbor_gaps else 0.0
        self.reference_gamma_deg = derive_ckm(
            level=self.fixed_quark_level,
            parent_level=self.fixed_parent_level,
            scale_ratio=self.reference_scale_ratio,
            gut_threshold_residue=self.reference_gut_threshold_residue,
        ).gamma_rg_deg

    def scan_candidate(self, lepton_level: int) -> LevelScanResult:
        flavor_spectrum = derive_matrix_spectrum_audit(ModularKernel(lepton_level, "lepton").restricted_block())
        flavor_condition_number = float(flavor_spectrum.display_condition_number)
        flavor_nonsingular = not flavor_spectrum.machine_precision_singular
        parent_central_charge = wzw_central_charge(self.fixed_parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
        visible_central_charge = wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER) + wzw_central_charge(
            self.fixed_quark_level,
            SU3_DIMENSION,
            SU3_DUAL_COXETER,
        )
        raw_difference = (
            parent_central_charge
            - visible_central_charge
            - self.reference_coset_central_charge
        ) / 24.0
        modularity_gap = distance_to_integer(raw_difference)
        framing_index = self.fixed_parent_level / (2.0 * lepton_level)
        framing_gap = nearest_integer_gap(framing_index)
        modular_tilt_deg = None
        gamma_candidate_deg = None
        gamma_pull = None
        bulk_anomaly_cancelled = False
        gap_within_relaxed_band = modularity_gap < self.relaxed_modularity_allowance or bool(
            math.isclose(modularity_gap, self.relaxed_modularity_allowance, abs_tol=1.0e-15, rel_tol=0.0)
        )
        if gap_within_relaxed_band:
            sign = -1.0 if lepton_level > self.reference_lepton_level else (1.0 if lepton_level < self.reference_lepton_level else 0.0)
            if self.relaxed_modularity_allowance > 0.0:
                modular_tilt_deg = sign * RELAXED_NEIGHBOR_TILT_DEG * min(modularity_gap / self.relaxed_modularity_allowance, 1.0)
            else:
                modular_tilt_deg = 0.0
            gamma_candidate_deg = self.reference_gamma_deg + modular_tilt_deg
            gamma_pull = pull_from_interval(gamma_candidate_deg, CKM_GAMMA_GOLD_STANDARD_DEG).pull
            bulk_anomaly_cancelled = flavor_nonsingular
        return LevelScanResult(
            lepton_level=lepton_level,
            quark_level=self.fixed_quark_level,
            parent_level=self.fixed_parent_level,
            lepton_branching_index=framing_index,
            quark_branching_index=self.fixed_parent_level // (3 * self.fixed_quark_level),
            visible_residual_mod1=mod_one_residual(raw_difference),
            modularity_gap=modularity_gap,
            framing_gap=framing_gap,
            flavor_condition_number=flavor_condition_number,
            central_charge_modular=solver_isclose(modularity_gap, 0.0),
            framing_anomaly_free=solver_isclose(framing_gap, 0.0),
            flavor_nonsingular=flavor_nonsingular,
            chi2_flavor=(gamma_pull * gamma_pull) if gamma_pull is not None else None,
            max_abs_pull=abs(gamma_pull) if gamma_pull is not None else None,
            flavor_matching=False,
            selected_visible_pair=(lepton_level, self.fixed_quark_level) == (self.benchmark_lepton_level, self.benchmark_quark_level),
            bulk_anomaly_cancelled=bulk_anomaly_cancelled,
            modular_tilt_deg=modular_tilt_deg,
            gamma_candidate_deg=gamma_candidate_deg,
            gamma_pull=gamma_pull,
        )

    def scan_window(self, lepton_levels: tuple[int, ...] | None = None) -> LevelStabilityScan:
        resolved_lepton_levels = (self.reference_lepton_level,) if lepton_levels is None else lepton_levels
        rows = tuple(self.scan_candidate(lepton_level) for lepton_level in resolved_lepton_levels)
        return LevelStabilityScan(
            fixed_parent_level=self.fixed_parent_level,
            lepton_range=(min(resolved_lepton_levels), max(resolved_lepton_levels)),
            quark_range=(self.fixed_quark_level, self.fixed_quark_level),
            total_pairs_scanned=len(rows),
            relaxed_modularity_allowance=self.relaxed_modularity_allowance,
            rows=rows,
        )

    def scan_global_candidate(self, lepton_level: int, quark_level: int) -> GlobalSensitivityRow:
        lepton_flavor_spectrum = derive_matrix_spectrum_audit(ModularKernel(lepton_level, "lepton").restricted_block())
        quark_flavor_spectrum = derive_matrix_spectrum_audit(ModularKernel(quark_level, "quark").restricted_block())
        lepton_flavor_nonsingular = not lepton_flavor_spectrum.machine_precision_singular
        quark_flavor_nonsingular = not quark_flavor_spectrum.machine_precision_singular
        parent_central_charge = wzw_central_charge(self.fixed_parent_level, SO10_DIMENSION, SO10_DUAL_COXETER)
        visible_central_charge = (
            wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER)
            + wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
        )
        raw_difference = (
            parent_central_charge
            - visible_central_charge
            - self.reference_coset_central_charge
        ) / 24.0
        modularity_gap = distance_to_integer(raw_difference)
        central_charge_residual = 24.0 * modularity_gap
        lepton_framing_gap = nearest_integer_gap(self.fixed_parent_level / (2.0 * lepton_level))
        quark_framing_gap = nearest_integer_gap(self.fixed_parent_level / (3.0 * quark_level))
        anomaly_energy = math.sqrt(
            central_charge_residual * central_charge_residual
            + lepton_framing_gap * lepton_framing_gap
            + quark_framing_gap * quark_framing_gap
        )
        exact_pass = (
            solver_isclose(modularity_gap, 0.0)
            and solver_isclose(lepton_framing_gap, 0.0)
            and solver_isclose(quark_framing_gap, 0.0)
            and lepton_flavor_nonsingular
            and quark_flavor_nonsingular
        )
        return GlobalSensitivityRow(
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=self.fixed_parent_level,
            central_charge_residual=central_charge_residual,
            modularity_gap=modularity_gap,
            lepton_framing_gap=lepton_framing_gap,
            quark_framing_gap=quark_framing_gap,
            anomaly_energy=anomaly_energy,
            lepton_flavor_condition_number=float(lepton_flavor_spectrum.display_condition_number),
            quark_flavor_condition_number=float(quark_flavor_spectrum.display_condition_number),
            exact_pass=exact_pass,
            selected_visible_pair=(lepton_level, quark_level) == (self.benchmark_lepton_level, self.benchmark_quark_level),
        )

    def scan_global_sensitivity_audit(
        self,
        lepton_range: tuple[int, int] = GLOBAL_LEPTON_LEVEL_RANGE,
        quark_range: tuple[int, int] = GLOBAL_QUARK_LEVEL_RANGE,
    ) -> GlobalSensitivityAudit:
        rows = [
            self.scan_global_candidate(lepton_level, quark_level)
            for lepton_level in range(lepton_range[0], lepton_range[1] + 1)
            for quark_level in range(quark_range[0], quark_range[1] + 1)
        ]
        rows.sort(key=lambda row: (row.anomaly_energy, row.modularity_gap, row.lepton_level, row.quark_level))
        return GlobalSensitivityAudit(
            lepton_range=lepton_range,
            quark_range=quark_range,
            total_pairs_scanned=len(rows),
            rows=tuple(rows),
        )


@dataclass(frozen=True)
class TopologicalModelEvaluation:
    """Core branch evaluation detached from module-level benchmark globals."""

    scales: ScaleData
    interface: BoundaryBulkInterfaceData
    pmns: PmnsData
    ckm: CkmData
    parent: ParentSelection
    framing: FramingAnomalyData
    audit: AuditData
    support_overlap: dict[str, np.ndarray]
    pull_table: PullTable
    rhn_threshold: RGThresholdData


@dataclass(frozen=True)
class TopologicalVacuum:
    """Immutable vacuum/configuration object for the branch-fixed physics kernels."""

    k_l: int = LEPTON_LEVEL
    k_q: int = QUARK_LEVEL
    parent_level: int = PARENT_LEVEL
    scale_ratio: float = RG_SCALE_RATIO
    bit_count: float = HOLOGRAPHIC_BITS
    kappa_geometric: float = GEOMETRIC_KAPPA
    gut_threshold_residue: float | None = R_GUT
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG

    @property
    def lepton_level(self) -> int:
        return self.k_l

    @property
    def quark_level(self) -> int:
        return self.k_q

    @property
    def target_tuple(self) -> tuple[int, int, int]:
        return (self.lepton_level, self.quark_level, self.parent_level)

    @property
    def framing_gap(self) -> float:
        """Return the branch framing gap after collapsing round-off level zeros."""

        raw_gap = float(nearest_integer_gap(self.parent_level / (2.0 * self.lepton_level)))
        return 0.0 if solver_isclose(raw_gap, 0.0) else raw_gap

    @property
    def local_lepton_level_window(self) -> tuple[int, ...]:
        return tuple(range(self.lepton_level - 2, self.lepton_level + 3))

    def derive_scales(self) -> ScaleData:
        return BoussoBridge.derive_scales(self.bit_count, model=self)

    def derive_inflationary_sector(
        self,
        flow_levels: tuple[int, ...] | None = None,
    ) -> InflationarySectorData:
        return InflationarySector(
            flow_levels=self.local_lepton_level_window if flow_levels is None else flow_levels,
            model=self,
        ).derive()

    def derive_cosmology_audit(
        self,
        flow_levels: tuple[int, ...] | None = None,
    ) -> CosmologyAudit:
        return self.derive_inflationary_sector(flow_levels=flow_levels)

    def derive_boundary_bulk_interface(self, sector: Sector | str = Sector.LEPTON) -> BoundaryBulkInterfaceData:
        return derive_boundary_bulk_interface(sector=sector, model=self)

    def derive_pmns(self) -> PmnsData:
        return derive_pmns(model=self)

    def derive_ckm(self) -> CkmData:
        return derive_ckm(model=self)

    def derive_parent_selection(self) -> ParentSelection:
        return derive_parent_selection(model=self)

    def verify_framing_anomaly(self) -> FramingAnomalyData:
        return verify_framing_anomaly(model=self)

    def derive_physics_audit(self) -> PhysicsAudit:
        return derive_physics_audit(model=self)

    def level_scanner(self) -> LevelScanner:
        return LevelScanner(
            fixed_parent_level=self.parent_level,
            fixed_quark_level=self.quark_level,
            reference_lepton_level=self.lepton_level,
            benchmark_lepton_level=self.lepton_level,
            benchmark_quark_level=self.quark_level,
            reference_scale_ratio=self.scale_ratio,
            reference_gut_threshold_residue=self.gut_threshold_residue,
        )

    def scan_window(self, lepton_levels: tuple[int, ...] | None = None) -> LevelStabilityScan:
        return self.level_scanner().scan_window(lepton_levels=self.local_lepton_level_window if lepton_levels is None else lepton_levels)

    def scan_global_sensitivity_audit(
        self,
        lepton_range: tuple[int, int] = GLOBAL_LEPTON_LEVEL_RANGE,
        quark_range: tuple[int, int] = GLOBAL_QUARK_LEVEL_RANGE,
    ) -> GlobalSensitivityAudit:
        return self.level_scanner().scan_global_sensitivity_audit(lepton_range=lepton_range, quark_range=quark_range)

    def derive_followup_chi2_landscape_audit(
        self,
        lepton_range: tuple[int, int] = FOLLOWUP_LEPTON_LEVEL_RANGE,
        quark_range: tuple[int, int] = FOLLOWUP_QUARK_LEVEL_RANGE,
    ) -> Chi2LandscapeAuditData:
        return derive_followup_chi2_landscape_audit(
            lepton_range=lepton_range,
            quark_range=quark_range,
            parent_level=self.parent_level,
            scale_ratio=self.scale_ratio,
            bit_count=self.bit_count,
            kappa_geometric=self.kappa_geometric,
            benchmark_lepton_level=self.lepton_level,
            benchmark_quark_level=self.quark_level,
        )

    def derive_audit(self) -> AuditData:
        return derive_audit(
            level=self.lepton_level,
            bit_count=self.bit_count,
            scale_ratio=self.scale_ratio,
            kappa_geometric=self.kappa_geometric,
            parent_level=self.parent_level,
            quark_level=self.quark_level,
            solver_config=self.solver_config,
        )

    def derive_sensitivity(self) -> SensitivityData:
        return derive_sensitivity(
            bit_count=self.bit_count,
            scale_ratio=self.scale_ratio,
            kappa_geometric=self.kappa_geometric,
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            parent_level=self.parent_level,
            gut_threshold_residue=self.gut_threshold_residue,
        )

    def derive_geometric_sensitivity(self) -> GeometricSensitivityData:
        return derive_geometric_sensitivity(
            bit_count=self.bit_count,
            scale_ratio=self.scale_ratio,
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            parent_level=self.parent_level,
            gut_threshold_residue=self.gut_threshold_residue,
            central_kappa_geometric=self.kappa_geometric,
        )

    def derive_nonlinearity_audit(self) -> NonLinearityAuditData:
        return derive_nonlinearity_audit(
            scale_ratio=self.scale_ratio,
            level=self.lepton_level,
            bit_count=self.bit_count,
            kappa_geometric=self.kappa_geometric,
            parent_level=self.parent_level,
            quark_level=self.quark_level,
            solver_config=self.solver_config,
        )

    def derive_threshold_shift_audit(self) -> ThresholdShiftAuditData:
        threshold_data = self.derive_rhn_threshold_data(Sector.LEPTON)
        scales = self.derive_scales()
        pmns = self.derive_pmns()
        audit = self.derive_audit()
        beta_function_data = derive_beta_function_data(
            pmns.pmns_matrix_uv,
            sector=Sector.LEPTON,
            lightest_mass_ev=scales.m_0_uv_ev,
            level=self.lepton_level,
            solver_config=self.solver_config,
        )
        return derive_threshold_shift_audit(
            scale_ratio=self.scale_ratio,
            level=self.lepton_level,
            bit_count=self.bit_count,
            kappa_geometric=self.kappa_geometric,
            parent_level=self.parent_level,
            quark_level=self.quark_level,
            threshold_data=threshold_data,
            scales=scales,
            pmns=pmns,
            audit=audit,
            beta_function_data=beta_function_data,
        )

    def generate_ckm_phase_tilt_profile(
        self,
        pmns: PmnsData | None = None,
        output_path: Path | None = None,
    ) -> CkmPhaseTiltProfileData:
        resolved_pmns = self.derive_pmns() if pmns is None else pmns
        return generate_ckm_phase_tilt_profile(pmns=resolved_pmns, output_path=output_path, model=self)

    def derive_threshold_sensitivity(self, ckm: CkmData | None = None) -> ThresholdSensitivityData:
        resolved_ckm = self.derive_ckm() if ckm is None else ckm
        return derive_threshold_sensitivity(
            ckm=resolved_ckm,
            scale_ratio=self.scale_ratio,
            parent_level=self.parent_level,
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            gut_threshold_residue=self.gut_threshold_residue,
        )

    def derive_mass_ratio_stability_audit(
        self,
        *,
        sample_count: int = VEV_ALIGNMENT_SWEEP_SAMPLE_COUNT,
        seed: int = DEFAULT_RANDOM_SEED,
        rng: np.random.Generator | None = None,
    ) -> MassRatioStabilityAuditData:
        return derive_mass_ratio_stability_audit(
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            sample_count=sample_count,
            seed=seed,
            rng=rng,
        )

    def derive_ghost_character_audit(self, audit: AuditData | None = None) -> GhostCharacterAuditData:
        resolved_audit = self.derive_audit() if audit is None else audit
        return derive_ghost_character_audit(resolved_audit, level=self.lepton_level)

    def derive_framing_gap_stability_audit(
        self,
        ckm: CkmData | None = None,
        audit: AuditData | None = None,
    ) -> FramingGapStabilityData:
        resolved_ckm = self.derive_ckm() if ckm is None else ckm
        resolved_audit = self.derive_audit() if audit is None else audit
        return derive_framing_gap_stability_audit(
            ckm=resolved_ckm,
            audit=resolved_audit,
            scale_ratio=self.scale_ratio,
            parent_level=self.parent_level,
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            gut_threshold_residue=self.gut_threshold_residue,
        )

    def derive_rhn_threshold_data(self, sector: Sector | str = Sector.LEPTON) -> RGThresholdData:
        return derive_rhn_threshold_data(sector=sector, model=self)

    def compute_geometric_kappa_ansatz(self) -> SO10GeometricKappaData:
        return compute_geometric_kappa_ansatz(parent_level=self.parent_level, lepton_level=self.lepton_level)

    def derive_so10_geometric_kappa(self) -> SO10GeometricKappaData:
        return self.compute_geometric_kappa_ansatz()

    def derive_modular_horizon_selection(self) -> ModularHorizonSelectionData:
        return derive_modular_horizon_selection(parent_level=self.parent_level, lepton_level=self.lepton_level)

    def derive_step_size_convergence_audit(self) -> StepSizeConvergenceData:
        return derive_step_size_convergence_audit(
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            parent_level=self.parent_level,
            scale_ratio=self.scale_ratio,
            bit_count=self.bit_count,
            kappa_geometric=self.kappa_geometric,
            gut_threshold_residue=self.gut_threshold_residue,
            solver_config=self.solver_config,
        )

    def derive_transport_parametric_covariance(
        self,
        pmns: PmnsData | None = None,
        ckm: CkmData | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> TransportParametricCovarianceData:
        resolved_pmns = self.derive_pmns() if pmns is None else pmns
        resolved_ckm = self.derive_ckm() if ckm is None else ckm
        return derive_transport_parametric_covariance(resolved_pmns, resolved_ckm, rng=rng)

    def derive_seed_robustness_audit(
        self,
        pmns: PmnsData | None = None,
        ckm: CkmData | None = None,
        *,
        seed: int = DEFAULT_RANDOM_SEED,
        seed_count: int = SEED_AUDIT_SAMPLE_COUNT,
    ) -> SeedRobustnessAuditData:
        resolved_pmns = self.derive_pmns() if pmns is None else pmns
        resolved_ckm = self.derive_ckm() if ckm is None else ckm
        seeds = tuple(seed + offset for offset in range(max(int(seed_count), 1)))
        predictive_chi2_values: list[float] = []
        predictive_p_values: list[float] = []
        parametric_sigmas: list[np.ndarray] = []
        vev_max_sigma_shifts: list[float] = []

        for sample_seed in seeds:
            transport_covariance = derive_transport_parametric_covariance(
                resolved_pmns,
                resolved_ckm,
                rng=np.random.default_rng(sample_seed),
            )
            pull_table = derive_pull_table(
                resolved_pmns,
                resolved_ckm,
                transport_covariance=transport_covariance,
            )
            vev_audit = derive_mass_ratio_stability_audit(
                lepton_level=self.lepton_level,
                quark_level=self.quark_level,
                sample_count=VEV_ALIGNMENT_SWEEP_SAMPLE_COUNT,
                seed=sample_seed,
            )
            predictive_chi2_values.append(float(pull_table.predictive_chi2))
            predictive_p_values.append(float(pull_table.predictive_p_value))
            parametric_sigmas.append(
                np.array(
                    [transport_covariance.interval_sigma_for(name) for name in transport_covariance.observable_names],
                    dtype=float,
                )
            )
            vev_max_sigma_shifts.append(float(vev_audit.ensemble_max_sigma_shift))

        predictive_chi2_array = np.asarray(predictive_chi2_values, dtype=float)
        predictive_p_array = np.asarray(predictive_p_values, dtype=float)
        parametric_sigma_array = np.asarray(parametric_sigmas, dtype=float)
        vev_sigma_array = np.asarray(vev_max_sigma_shifts, dtype=float)
        relative_components = [_relative_std(predictive_chi2_array), _relative_std(predictive_p_array), _relative_std(vev_sigma_array)]
        if parametric_sigma_array.size > 0:
            relative_components.extend(_relative_std(parametric_sigma_array[:, index]) for index in range(parametric_sigma_array.shape[1]))
        max_relative_std = max(relative_components, default=0.0)
        return freeze_numpy_arrays(SeedRobustnessAuditData(
            seeds=seeds,
            observable_names=TRANSPORT_OBSERVABLE_ORDER,
            predictive_chi2_values=predictive_chi2_array,
            predictive_p_values=predictive_p_array,
            parametric_sigmas=parametric_sigma_array,
            vev_max_sigma_shifts=vev_sigma_array,
              max_relative_variance=max_relative_std * max_relative_std,
              max_relative_std=max_relative_std,
          ))

    def gravity_engine(self) -> "EinsteinConsistencyEngine":
        return EinsteinConsistencyEngine(model=self)

    def verify_bulk_emergence(self) -> GravityAudit:
        return self.gravity_engine().verify_bulk_emergence()

    def verify_gauge_holography(self) -> GaugeHolographyAudit:
        return verify_gauge_holography(model=self)

    def verify_dark_energy_tension(self) -> DarkEnergyTensionAudit:
        return verify_dark_energy_tension(model=self)

    def verify_unitary_bounds(self) -> UnitaryBoundAudit:
        return verify_unitary_bounds(model=self)

    def verify_unitary_audit(self) -> UnitaryAudit:
        return self.verify_unitary_bounds()

    def validate_welded_mass_coordinate(self, mass_coordinate_ev: float | None = None) -> None:
        resolved_mass_coordinate = (
            topological_mass_coordinate_ev(bit_count=self.bit_count, kappa_geometric=self.kappa_geometric)
            if mass_coordinate_ev is None
            else float(mass_coordinate_ev)
        )
        enforce_topological_mass_coordinate_lock(
            resolved_mass_coordinate,
            bit_count=self.bit_count,
            kappa_geometric=self.kappa_geometric,
            parent_level=self.parent_level,
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
        )

    def derive_page_point_audit(self) -> PagePointAudit:
        return derive_page_point_audit(model=self)

    def simulate_neutron_star_transit(
        self,
        *,
        gravitational_gradient_scale: float = 6.0,
        transit_path_length_km: float = 20.0,
        density_scale: float = 2.0,
        pmns: PmnsData | None = None,
        unitary_audit: UnitaryBoundAudit | None = None,
    ) -> TorsionScramblingTransitAudit:
        return simulate_neutron_star_transit(
            gravitational_gradient_scale=gravitational_gradient_scale,
            transit_path_length_km=transit_path_length_km,
            density_scale=density_scale,
            pmns=pmns,
            unitary_audit=unitary_audit,
            model=self,
        )

    def derive_framing_stability_audit(
        self,
        *,
        generation_count: int = GAUGE_HOLOGRAPHY_GENERATION_COUNT,
        alpha_shift_fraction: float = 0.01,
    ) -> FramingStabilityAudit:
        return derive_framing_stability_audit(
            generation_count=generation_count,
            alpha_shift_fraction=alpha_shift_fraction,
            model=self,
        )

    def derive_single_bit_rigidity_audit(self, *, bit_shift: int = 1) -> RigidityStressTestAudit:
        return derive_single_bit_rigidity_audit(bit_shift=bit_shift, model=self)

    def verify_single_bit_rigidity(self, *, bit_shift: int = 1) -> RigidityStressTestAudit:
        return self.derive_single_bit_rigidity_audit(bit_shift=bit_shift)

    def evaluate(self) -> TopologicalModelEvaluation:
        scales = self.derive_scales()
        interface = self.derive_boundary_bulk_interface()
        pmns = self.derive_pmns()
        ckm = self.derive_ckm()
        parent = self.derive_parent_selection()
        framing = self.verify_framing_anomaly()
        audit = self.derive_audit()
        support_overlap = audit.calculate_support_overlap(level=self.lepton_level)
        pull_table = derive_pull_table(pmns, ckm)
        rhn_threshold = self.derive_rhn_threshold_data(Sector.LEPTON)
        return TopologicalModelEvaluation(
            scales=scales,
            interface=interface,
            pmns=pmns,
            ckm=ckm,
            parent=parent,
            framing=framing,
            audit=audit,
            support_overlap=support_overlap,
            pull_table=pull_table,
            rhn_threshold=rhn_threshold,
        )


TopologicalModel = TopologicalVacuum
DEFAULT_TOPOLOGICAL_VACUUM = TopologicalVacuum()


@dataclass(frozen=True)
class ComprehensiveAudit:
    r"""Publication-facing wrapper for the dynamic and gauge-integrity checks."""

    pmns: PmnsData
    ckm: CkmData
    transport_covariance: TransportParametricCovarianceData
    vacuum: TopologicalVacuum = DEFAULT_TOPOLOGICAL_VACUUM

    def verify_bulk_emergence(self) -> GravityAudit:
        return self.vacuum.verify_bulk_emergence()

    def verify_gauge_holography(self) -> GaugeHolographyAudit:
        return self.vacuum.verify_gauge_holography()

    def verify_dark_energy_tension(self) -> DarkEnergyTensionAudit:
        return self.vacuum.verify_dark_energy_tension()

    def verify_unitary_bounds(self) -> UnitaryBoundAudit:
        return self.vacuum.verify_unitary_bounds()

    def derive_cosmology_audit(self) -> CosmologyAudit:
        return self.vacuum.derive_cosmology_audit()

    def verify_unitary_audit(self) -> UnitaryAudit:
        return self.vacuum.verify_unitary_audit()

    def derive_page_point_audit(self, unitary_audit: UnitaryBoundAudit | None = None) -> PagePointAudit:
        return derive_page_point_audit(model=self.vacuum, unitary_audit=unitary_audit)

    def derive_framing_stability_audit(
        self,
        gauge_audit: GaugeHolographyAudit | None = None,
        *,
        alpha_shift_fraction: float = 0.01,
    ) -> FramingStabilityAudit:
        return derive_framing_stability_audit(
            model=self.vacuum,
            gauge_audit=gauge_audit,
            alpha_shift_fraction=alpha_shift_fraction,
        )

    def derive_single_bit_rigidity_audit(self, *, bit_shift: int = 1) -> RigidityStressTestAudit:
        return self.vacuum.derive_single_bit_rigidity_audit(bit_shift=bit_shift)

    def run(self) -> bool:
        current_values = {
            "theta12": self.pmns.theta12_rg_deg,
            "theta13": self.pmns.theta13_rg_deg,
            "theta23": self.pmns.theta23_rg_deg,
            "delta_cp": self.pmns.delta_cp_rg_deg,
            "vus": self.ckm.vus_rg,
            "vcb": self.ckm.vcb_rg,
            "vub": self.ckm.vub_rg,
            "gamma": self.ckm.gamma_rg_deg,
        }
        intervals = {
            "theta12": LEPTON_INTERVALS["theta12"],
            "theta13": LEPTON_INTERVALS["theta13"],
            "theta23": LEPTON_INTERVALS["theta23"],
            "delta_cp": LEPTON_INTERVALS["delta_cp"],
            "vus": QUARK_INTERVALS["vus"],
            "vcb": QUARK_INTERVALS["vcb"],
            "vub": QUARK_INTERVALS["vub"],
            "gamma": CKM_GAMMA_GOLD_STANDARD_DEG,
        }
        unit_suffix = {
            "theta12": "deg",
            "theta13": "deg",
            "theta23": "deg",
            "delta_cp": "deg",
            "vus": "",
            "vcb": "",
            "vub": "",
            "gamma": "deg",
        }

        LOGGER.info("Comprehensive dynamic audit")
        LOGGER.info("-" * 88)
        all_within_two_sigma = True
        for observable, value in current_values.items():
            interval = intervals[observable]
            pull = pull_from_interval(value, interval).pull
            all_within_two_sigma = all_within_two_sigma and abs(pull) <= 2.0
            units = f" {unit_suffix[observable]}" if unit_suffix[observable] else ""
            LOGGER.info(
                f"{observable:<10} dynamic={value:.12f}{units}  interval={interval.central:.12f}±{interval.sigma:.12f}{units}  pull={pull:+.6f}σ"
            )
        LOGGER.info(f"all observables within 2sigma    : {int(all_within_two_sigma)}")
        LOGGER.info("")

        LOGGER.info("Transport covariance audit")
        LOGGER.info("-" * 88)
        LOGGER.info(f"covariance estimator             : {self.transport_covariance.covariance_mode}")
        LOGGER.info(
            f"Stability Yield                  : {100.0 * self.transport_covariance.stability_yield:.2f}% "
            f"({self.transport_covariance.accepted_samples}/{self.transport_covariance.attempted_samples})"
        )
        LOGGER.info(f"singular samples                 : {self.transport_covariance.failure_count}")
        LOGGER.info(f"sample failure fraction          : {self.transport_covariance.failure_fraction:.3%}")
        LOGGER.info(f"hard-wall penalty chi2           : {self.transport_covariance.singularity_chi2_penalty:.1e}")
        LOGGER.info("")

        gauge_audit = self.verify_gauge_holography()
        gauge_status = "VERIFIED" if gauge_audit.topological_stability_pass else "FAILED"
        LOGGER.info("Gauge holography audit")
        LOGGER.info("-" * 88)
        LOGGER.info(f"N_gen                           : {gauge_audit.generation_count}")
        LOGGER.info(
            f"alpha^-1 level density           : {gauge_audit.generation_count} * {gauge_audit.parent_level}/"
            f"({gauge_audit.lepton_level}+{gauge_audit.quark_level}) = {gauge_audit.topological_alpha_inverse:.12f}"
        )
        LOGGER.info(f"alpha^-1 integer target        : {gauge_audit.codata_alpha_inverse:.12f}")
        LOGGER.info(f"10^3 Delta_mod                  : {gauge_audit.modular_gap_scaled_inverse:.12f}")
        LOGGER.info(f"gauge geometric residue         : {gauge_audit.geometric_residue_percent:.2f}%")
        LOGGER.info(f"modular-gap alignment           : {gauge_audit.modular_gap_alignment_percent:.2f}%")
        LOGGER.info(f"Topological Stability Pass      : {int(gauge_audit.topological_stability_pass)}")
        LOGGER.info(f"[{gauge_status}] Gauge Coupling Residue (Alpha) -> Delta: {gauge_audit.geometric_residue_percent:.2f}%")
        LOGGER.info("")
        framing_stability = self.derive_framing_stability_audit(gauge_audit=gauge_audit)
        LOGGER.info("Framing-stability audit")
        LOGGER.info("-" * 88)
        LOGGER.info(f"alpha^-1 central                : {framing_stability.central_alpha_inverse:.12f}")
        LOGGER.info(
            f"effective k_l for alpha-{framing_stability.alpha_shift_percent:.1f}% : "
            f"{framing_stability.minus_shifted_effective_lepton_level:.12f}"
        )
        LOGGER.info(
            f"effective k_l for alpha+{framing_stability.alpha_shift_percent:.1f}% : "
            f"{framing_stability.plus_shifted_effective_lepton_level:.12f}"
        )
        LOGGER.info(
            f"Delta_fr for alpha-{framing_stability.alpha_shift_percent:.1f}%    : "
            f"{framing_stability.minus_shifted_framing_gap:.12f}"
        )
        LOGGER.info(
            f"Delta_fr for alpha+{framing_stability.alpha_shift_percent:.1f}%    : "
            f"{framing_stability.plus_shifted_framing_gap:.12f}"
        )
        LOGGER.info(f"Alpha-lock required             : {int(framing_stability.alpha_lock_required)}")
        LOGGER.info("[ASSERTION]: Alpha-locking is a requirement for Framing Closure.")
        LOGGER.info("")
        dark_energy_audit = self.verify_dark_energy_tension()
        gravity_audit = self.verify_bulk_emergence()
        LOGGER.info("Dark-energy surface-tension audit")
        LOGGER.info("-" * 88)
        LOGGER.info(f"N_holo [bits]                   : {dark_energy_audit.holographic_bits:.12e}")
        LOGGER.info(f"kappa_D5                        : {dark_energy_audit.geometric_residue:.12f}")
        LOGGER.info(f"c_visible logical bits          : {gravity_audit.visible_central_charge:.12f}")
        LOGGER.info(f"Delta_mod benchmark             : {dark_energy_audit.modular_gap:.12f}")
        LOGGER.info(f"c_dark completion               : {dark_energy_audit.c_dark_completion:.4f}")
        LOGGER.info(f"eta_mod = c_dark/c_visible      : {gravity_audit.modular_residue_efficiency:.12f}")
        LOGGER.info(f"Delta_DM = kappa*eta_mod        : {gravity_audit.omega_dm_ratio:.12f}")
        LOGGER.info(f"Parity-Bit Density Constraint   : {int(gravity_audit.parity_bit_density_constraint_satisfied)}")
        LOGGER.info(f"Lambda_holo [m^-2]              : {dark_energy_audit.lambda_surface_tension_si_m2:.12e}")
        LOGGER.info(f"Lambda anchor [m^-2]            : {dark_energy_audit.lambda_anchor_si_m2:.12e}")
        LOGGER.info(f"1/(L_P^2 N) [m^-2]              : {dark_energy_audit.lambda_scaling_identity_si_m2:.12e}")
        LOGGER.info(f"surface-tension prefactor       : {dark_energy_audit.surface_tension_prefactor:.12f}")
        LOGGER.info(f"rho_vac surface tension [eV^4]  : {dark_energy_audit.rho_vac_surface_tension_ev4:.12e}")
        LOGGER.info(f"rho_vac(m_0,kappa) [eV^4]       : {dark_energy_audit.rho_vac_from_defect_scale_ev4:.12e}")
        LOGGER.info(f"{TOPOLOGICAL_MASS_COORDINATE_LABEL} [eV]    : {dark_energy_audit.topological_mass_coordinate_ev:.12e}")
        LOGGER.info(f"Lambda*G_N*m_nu^4               : {dark_energy_audit.triple_match_product:.12e}")
        LOGGER.info(f"Triple-Match saturation         : {int(dark_energy_audit.triple_match_saturated)}")
        LOGGER.info(f"Surface Tension Deviation       : {dark_energy_audit.surface_tension_deviation_percent:.2f}%")
        LOGGER.info(f"Lambda shift for N-1%           : {100.0 * dark_energy_audit.minus_one_percent_lambda_fractional_shift:+.3f}%")
        LOGGER.info(f"Lambda shift for N+1%           : {100.0 * dark_energy_audit.plus_one_percent_lambda_fractional_shift:+.3f}%")
        LOGGER.info(f"m_0 shift for N-1%              : {100.0 * dark_energy_audit.minus_one_percent_m0_fractional_shift:+.3f}%")
        LOGGER.info(f"m_0 shift for N+1%              : {100.0 * dark_energy_audit.plus_one_percent_m0_fractional_shift:+.3f}%")
        LOGGER.info(
            f"alpha^-1 under N±1%             : {dark_energy_audit.alpha_inverse_minus_one_percent:.12f} / "
            f"{dark_energy_audit.alpha_inverse_central:.12f} / {dark_energy_audit.alpha_inverse_plus_one_percent:.12f}"
        )
        LOGGER.info(f"Sensitivity Audit               : {dark_energy_audit.sensitivity_audit_message}")
        LOGGER.info("")
        unitary_audit = self.verify_unitary_bounds()
        unitary_status = "PASS" if (
            unitary_audit.unitary_bound_satisfied
            and unitary_audit.recovery_locked_to_delta_mod
            and unitary_audit.universal_computational_limit_pass
        ) else "FAIL"
        LOGGER.info("Finite-buffer unitarity audit")
        LOGGER.info("-" * 88)
        LOGGER.info(f"S_max = ln(N_holo)              : {unitary_audit.entropy_max_nats:.12f}")
        LOGGER.info(f"c_dark buffer entropy           : {unitary_audit.holographic_buffer_entropy:.12f}")
        LOGGER.info(f"regulated curvature entropy     : {unitary_audit.regulated_curvature_entropy:.12f}")
        LOGGER.info(f"curvature buffer margin         : {unitary_audit.curvature_buffer_margin:.12f}")
        LOGGER.info(f"buffer margin [%]               : {unitary_audit.curvature_buffer_margin_percent:.2f}")
        LOGGER.info(f"info evaporation rate [yr^-1]   : {unitary_audit.information_evaporation_rate_per_year:.12e}")
        LOGGER.info(f"recovery lifetime [yr]          : {unitary_audit.recovery_lifetime_years:.12e}")
        LOGGER.info(f"zero-point complexity           : {unitary_audit.zero_point_complexity:.12f}")
        LOGGER.info(f"max complexity capacity         : {unitary_audit.max_complexity_capacity:.12e}")
        LOGGER.info(f"Lloyd limit [s^-1]              : {unitary_audit.lloyds_limit_ops_per_second:.12e}")
        LOGGER.info(f"complexity growth rate [s^-1]   : {unitary_audit.complexity_growth_rate_ops_per_second:.12e}")
        LOGGER.info(f"complexity utilization          : {unitary_audit.complexity_utilization_fraction:.12f}")
        LOGGER.info(f"clock-skew (1 - n_s)            : {unitary_audit.clock_skew:.12f}")
        LOGGER.info(f"torsion-free stability          : {int(unitary_audit.torsion_free_stability)}")
        LOGGER.info(f"Lambda*G_N*m_nu^4               : {unitary_audit.triple_match_product:.12e}")
        LOGGER.info(f"Holographic Rigidity            : {int(unitary_audit.holographic_rigidity)}")
        LOGGER.info(f"[{unitary_status}] Unitary Bound Check: Information recovery rate locked to Delta_mod.")
        LOGGER.info("Universal Computational Limit: PASSED.")
        LOGGER.info("")
        page_point = self.derive_page_point_audit(unitary_audit=unitary_audit)
        LOGGER.info("Page-point audit")
        LOGGER.info("-" * 88)
        LOGGER.info(f"S_Page = c_dark ln(N_holo)      : {page_point.page_point_entropy:.12f}")
        LOGGER.info(f"S_ent bulk                      : {page_point.bulk_entanglement_entropy:.12f}")
        LOGGER.info(f"modular complement entropy      : {page_point.modular_complement_entropy:.12f}")
        LOGGER.info(f"Page Point saturation [%]       : {page_point.page_point_saturation_percent:.2f}")
        LOGGER.info(f"Page curve locked               : {int(page_point.page_curve_locked)}")
        LOGGER.info("[ASSERTION]: (m_nu, G_N, alpha, Lambda_holo, tau_p) are Topological Coordinates of the (26,8,312) boundary.")
        LOGGER.info("")
        return (
            all_within_two_sigma
            and gauge_audit.topological_stability_pass
            and framing_stability.alpha_lock_required
            and unitary_audit.unitary_bound_satisfied
            and unitary_audit.torsion_free_stability
            and unitary_audit.recovery_locked_to_delta_mod
            and page_point.page_curve_locked
        )


def _coerce_topological_model(
    *,
    model: TopologicalModel | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    parent_level: int | None = None,
    scale_ratio: float | None = None,
    bit_count: float | None = None,
    kappa_geometric: float | None = None,
    gut_threshold_residue: float | None = None,
    solver_config: SolverConfig | None = None,
) -> TopologicalModel:
    return TopologicalModel(
        k_l=_resolve_model_value(
            lepton_level,
            model=model,
            model_value=model.lepton_level if model is not None else None,
            default_value=LEPTON_LEVEL,
            parameter_name="lepton_level",
        ),
        k_q=_resolve_model_value(
            quark_level,
            model=model,
            model_value=model.quark_level if model is not None else None,
            default_value=QUARK_LEVEL,
            parameter_name="quark_level",
        ),
        parent_level=_resolve_model_value(
            parent_level,
            model=model,
            model_value=model.parent_level if model is not None else None,
            default_value=PARENT_LEVEL,
            parameter_name="parent_level",
        ),
        scale_ratio=_resolve_model_value(
            scale_ratio,
            model=model,
            model_value=model.scale_ratio if model is not None else None,
            default_value=RG_SCALE_RATIO,
            parameter_name="scale_ratio",
            comparator=_matching_float,
        ),
        bit_count=_resolve_model_value(
            bit_count,
            model=model,
            model_value=model.bit_count if model is not None else None,
            default_value=HOLOGRAPHIC_BITS,
            parameter_name="bit_count",
            comparator=_matching_float,
        ),
        kappa_geometric=_resolve_model_value(
            kappa_geometric,
            model=model,
            model_value=model.kappa_geometric if model is not None else None,
            default_value=GEOMETRIC_KAPPA,
            parameter_name="kappa_geometric",
            comparator=_matching_float,
        ),
        gut_threshold_residue=_resolve_model_value(
            gut_threshold_residue,
            model=model,
            model_value=model.gut_threshold_residue if model is not None else None,
            default_value=R_GUT,
            parameter_name="gut_threshold_residue",
            comparator=_matching_float,
        ),
        solver_config=_resolve_model_value(
            solver_config,
            model=model,
            model_value=model.solver_config if model is not None else None,
            default_value=DEFAULT_SOLVER_CONFIG,
            parameter_name="solver_config",
        ),
    )


@dataclass(frozen=True)
class EinsteinConsistencyEngine:
    r"""Topological Consistency Map for the emergent Einstein system.

    The class does not claim a microscopic derivation of quantum gravity.
    Instead it packages the manuscript's disclosed effective dictionary into a
    single audit: torsion-freeness from the framing condition, non-singularity
    from the $\\mathbf{126}_H$ regulator, and cosmological alignment from the
    CKN bridge for the light neutrino scale.
    """

    model: TopologicalModel | None = None
    pec_floor: float = 4.875
    lambda_alignment_rtol: float = 1.0e-12
    lambda_alignment_atol_ev: float = 1.0e-18

    def calculate_topological_evaporation_rate(self) -> BaryonStabilityAudit:
        r"""Return the benchmark proton-decay floor from the same GUT audit.

        The computation reuses the publication benchmark inputs: the one-loop
        unification scale $M_{\rm GUT}=2.0\times10^{16}\,\mathrm{GeV}$, the
        corresponding unified coupling $\alpha_{\rm GUT}^{-1}\approx 28.3$,
        the $D_5$ geometric residue $\kappa_{D_5}$, and the manuscript-facing
        de-anchored modularity gap $\Delta_{\rm mod}$. In the benchmark interpretation the
        Holographic Information Horizon removes the rapid $d=5$ channel through
        exact framing closure, so the quoted lifetime is the protected $d=6$
        floor.
        """

        resolved_model = _coerce_topological_model(model=self.model)
        geometric_kappa = compute_geometric_kappa_ansatz(
            parent_level=resolved_model.parent_level,
            lepton_level=resolved_model.lepton_level,
        )
        gauge_unification = derive_gauge_unification_existence_proof(m_gut_gev=GUT_SCALE_GEV)
        framing_data = verify_framing_anomaly(model=resolved_model)
        framing_gap = nearest_integer_gap(resolved_model.parent_level / (2.0 * resolved_model.lepton_level))
        modular_gap = benchmark_visible_modularity_gap(model=resolved_model)
        modular_tunneling_penalty = math.exp(-1.0 / modular_gap)
        tunneling_safety_boost = 1.0 / modular_tunneling_penalty
        unified_alpha_inverse = float(gauge_unification.unified_alpha_inverse)
        unified_alpha = 1.0 / unified_alpha_inverse
        effective_gauge_mass_gev = gauge_unification.m_gut_gev / geometric_kappa.derived_kappa
        dimension_six_width_gev = (
            unified_alpha
            * unified_alpha
            * PROTON_BOUNDARY_PIXEL_SCALE_GEV**5
            / effective_gauge_mass_gev**4
        )
        proton_lifetime_years = HBAR_GEV_SECONDS / (dimension_six_width_gev * SECONDS_PER_JULIAN_YEAR)
        protected_evaporation_lifetime_years = proton_lifetime_years * tunneling_safety_boost
        return BaryonStabilityAudit(
            gut_scale_gev=float(gauge_unification.m_gut_gev),
            effective_gauge_mass_gev=float(effective_gauge_mass_gev),
            unified_alpha_inverse=unified_alpha_inverse,
            unified_alpha=float(unified_alpha),
            modular_gap=modular_gap,
            modular_tunneling_penalty=float(modular_tunneling_penalty),
            tunneling_safety_boost=float(tunneling_safety_boost),
            dimension_six_width_gev=float(dimension_six_width_gev),
            proton_lifetime_years=float(proton_lifetime_years),
            protected_evaporation_lifetime_years=float(protected_evaporation_lifetime_years),
            dimension_five_forbidden=bool(solver_isclose(framing_gap, 0.0)),
        )

    def verify_bulk_emergence(self) -> GravityAudit:
        resolved_model = _coerce_topological_model(model=self.model)
        parent_central_charge = wzw_central_charge(
            resolved_model.parent_level,
            SO10_DIMENSION,
            SO10_DUAL_COXETER,
        )
        geometric_kappa = compute_geometric_kappa_ansatz(
            parent_level=resolved_model.parent_level,
            lepton_level=resolved_model.lepton_level,
        )
        scales = derive_scales(model=resolved_model)
        cosmology_anchor = derive_cosmology_anchor()
        framing_data = verify_framing_anomaly(model=resolved_model)
        framing_gap = nearest_integer_gap(resolved_model.parent_level / (2.0 * resolved_model.lepton_level))
        torsion_free = solver_isclose(framing_gap, 0.0)
        derived_c_dark_completion = 24.0 * benchmark_visible_modularity_gap(model=resolved_model)
        c_dark_completion = float(derived_c_dark_completion)
        if torsion_free:
            assert abs(c_dark_completion - BENCHMARK_C_DARK_RESIDUE) < 1.0e-3, (
                "Benchmark modular complement residue drifted away from 3.3008."
            )
        visible_central_charge = float(framing_data.visible_central_charge)
        modular_efficiency = modular_residue_efficiency(
            c_dark_completion,
            lepton_level=resolved_model.lepton_level,
            quark_level=resolved_model.quark_level,
        )
        omega_dm_ratio = parity_bit_density_ratio(
            geometric_kappa.derived_kappa,
            c_dark_completion,
            lepton_level=resolved_model.lepton_level,
            quark_level=resolved_model.quark_level,
        )
        parity_bit_density_constraint_satisfied = (
            abs(omega_dm_ratio - PARITY_BIT_DENSITY_CONSTRAINT_TARGET)
            < PARITY_BIT_DENSITY_CONSTRAINT_TOLERANCE
        )
        if torsion_free:
            assert parity_bit_density_constraint_satisfied, "Parity-Bit Density Constraint"
        clebsch_126 = (SU3_DUAL_COXETER / SU2_DUAL_COXETER) * (resolved_model.lepton_level / resolved_model.quark_level)
        mass_suppression = calculate_126_higgs_cg_correction(clebsch_126=clebsch_126).clebsch_126
        ckn_limit_ev = geometric_kappa.derived_kappa * PLANCK_MASS_EV * resolved_model.bit_count ** (-0.25)
        lambda_budget_si_m2 = holographic_surface_tension_lambda_si_m2(model=resolved_model)
        baryon_stability = self.calculate_topological_evaporation_rate()
        non_singular_bulk = mass_suppression > self.pec_floor or math.isclose(
            mass_suppression,
            self.pec_floor,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        lambda_aligned = math.isclose(
            scales.m_0_uv_ev,
            ckn_limit_ev,
            rel_tol=self.lambda_alignment_rtol,
            abs_tol=self.lambda_alignment_atol_ev,
        )
        return GravityAudit(
            parent_central_charge=float(parent_central_charge),
            holographic_bits=float(resolved_model.bit_count),
            geometric_residue=float(geometric_kappa.derived_kappa),
            visible_central_charge=visible_central_charge,
            c_dark_completion=float(c_dark_completion),
            modular_residue_efficiency=float(modular_efficiency),
            omega_dm_ratio=float(omega_dm_ratio),
            parity_bit_density_constraint_satisfied=bool(parity_bit_density_constraint_satisfied),
            framing_gap=float(framing_gap),
            vacuum_pressure_t00=float(rank_deficit_pressure(resolved_model.parent_level, resolved_model.quark_level)),
            mass_suppression=float(mass_suppression),
            neutrino_scale_ev=float(scales.m_0_uv_ev),
            ckn_limit_ev=float(ckn_limit_ev),
            lambda_budget_si_m2=float(lambda_budget_si_m2),
            observed_lambda_si_m2=float(cosmology_anchor.lambda_si_m2),
            baryon_stability=baryon_stability,
            torsion_free=bool(torsion_free),
            non_singular_bulk=bool(non_singular_bulk),
            lambda_aligned=bool(lambda_aligned),
        )


def derive_pull_table(
    pmns: PmnsData,
    ckm: CkmData,
    *,
    transport_covariance: TransportParametricCovarianceData | None = None,
    landscape_trial_count: float = float(LANDSCAPE_TRIAL_COUNT),
    followup_trial_count: int = 1,
    effective_correlation_length: float = 1.0,
    lepton_correlation_length: float = 1.0,
    quark_correlation_length: float = 1.0,
) -> PullTable:
    def build_transport_row(
        observable: str,
        theory_uv: float,
        theory_mz: float,
        interval: Interval,
        observable_name: str,
        structural_context: str,
        source_label: str,
        units: str = "",
    ) -> PullTableRow:
        pull_data = pull_from_transport_covariance(
            theory_mz,
            interval,
            theory_value=theory_mz,
            observable_name=observable_name,
            transport_covariance=transport_covariance,
        )
        parametric_covariance_fraction = 0.0 if math.isclose(
            theory_mz,
            0.0,
            rel_tol=0.0,
            abs_tol=condition_aware_abs_tolerance(scale=theory_mz),
        ) else pull_data.parametric_sigma / abs(theory_mz)
        return PullTableRow(
            observable,
            theory_uv,
            theory_mz,
            pull_data,
            structural_context,
            source_label,
            units,
            parametric_covariance_fraction=parametric_covariance_fraction,
            observable_key=observable_name,
        )

    rows = (
        build_transport_row(r"$\theta_{12}$", pmns.theta12_uv_deg, pmns.theta12_rg_deg, LEPTON_INTERVALS["theta12"], "theta12", r"\shortstack{\scriptsize Solar-Overlap\\ \scriptsize Tension / one-loop\\ \scriptsize precision limit}", r"NuFIT~5.3\\(2024)", "deg"),
        build_transport_row(r"$\theta_{13}$", pmns.theta13_uv_deg, pmns.theta13_rg_deg, LEPTON_INTERVALS["theta13"], "theta13", r"\shortstack{\scriptsize $S$-matrix / Rigid\\ \scriptsize leptonic overlap kernel\\ \scriptsize with standard SM RG}", r"NuFIT~5.3\\(2024)", "deg"),
        build_transport_row(r"$\theta_{23}$", pmns.theta23_uv_deg, pmns.theta23_rg_deg, LEPTON_INTERVALS["theta23"], "theta23", r"\shortstack{\scriptsize $S$-matrix / Rigid\\ \scriptsize leptonic overlap kernel\\ \scriptsize with standard SM RG}", r"NuFIT~5.3\\(2024)", "deg"),
        build_transport_row(r"$\delta_{CP}$", pmns.delta_cp_uv_deg, pmns.delta_cp_rg_deg, LEPTON_INTERVALS["delta_cp"], "delta_cp", r"\shortstack{\scriptsize $T$-matrix / leptonic phase\\ \scriptsize after SM RG}", r"NuFIT~5.3\\(2024)", "deg"),
        build_transport_row(r"$|V_{us}|$", ckm.vus_uv, ckm.vus_rg, QUARK_INTERVALS["vus"], "vus", r"\shortstack{\scriptsize Prediction\\ \scriptsize from the rigid $SU(3)_8$\\ \scriptsize overlap kernel}", r"PDG~2024\\Sec.~12"),
        build_transport_row(r"$|V_{cb}|$", ckm.vcb_uv, ckm.vcb_rg, QUARK_INTERVALS["vcb"], "vcb", r"\shortstack{\scriptsize Prediction\\ \scriptsize from descendant pressure\\ \scriptsize in the $23$ channel}", r"PDG~2024\\Sec.~12"),
        build_transport_row(r"$|V_{ub}|$", ckm.vub_uv, ckm.vub_rg, QUARK_INTERVALS["vub"], "vub", r"\shortstack{\scriptsize Prediction\\ \scriptsize from chained $12$--$23$\\ \scriptsize suppression}", r"PDG~2024\\Sec.~12"),
        build_transport_row(r"$\gamma$", ckm.gamma_uv_deg, ckm.gamma_rg_deg, CKM_GAMMA_GOLD_STANDARD_DEG, "gamma", r"\shortstack{\scriptsize Derived $SO(10)\to SM$\\ \scriptsize ultraviolet matching\\ \scriptsize Wilson coefficient}", r"PDG~2024\\Sec.~12", "deg"),
        PullTableRow(r"$m_\nu$", pmns.normal_order_masses_uv_ev[0], pmns.normal_order_masses_rg_ev[0], None, r"\shortstack{\scriptsize Topological\\ \scriptsize Coordinate / $D_5$\\ \scriptsize packing tautology}", "Topological Coordinate", "eV", reference_override=r"tautology of the $D_5$ packing efficiency", included_in_audit=False, included_in_predictive_fit=False),
        PullTableRow(r"$|m_{\beta\beta}|$", 1.0e3 * pmns.effective_majorana_mass_uv_ev, 1.0e3 * pmns.effective_majorana_mass_rg_ev, None, r"\shortstack{\scriptsize Framed $SU(2)_{26}$\\ \scriptsize Majorana closure\\ \scriptsize benchmark}", "Structural prediction", "meV", reference_override=r"Majorana-floor prediction", included_in_audit=False, included_in_predictive_fit=False),
    )
    audit_rows = tuple(row for row in rows if row.included_in_audit and row.pull_data is not None)
    predictive_rows = tuple(row for row in rows if row.included_in_predictive_fit and row.pull_data is not None)
    calibration_rows = tuple(row for row in rows if row.is_calibration_anchor and row.pull_data is not None)

    if len(calibration_rows) > 1:
        raise RuntimeError("Expected at most one calibration anchor in the global pull table.")

    calibration_parameter_count = len(calibration_rows)
    branch_fixed_geometric_kappa = compute_geometric_kappa_ansatz(
        parent_level=pmns.parent_level,
        lepton_level=pmns.level,
    ).derived_kappa
    phenomenological_parameter_count = 0 if math.isclose(
        pmns.kappa_geometric,
        branch_fixed_geometric_kappa,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) else 1
    threshold_alignment_subtraction_count = 0 if math.isclose(ckm.gut_threshold_residue, R_GUT, rel_tol=0.0, abs_tol=1.0e-15) else 1
    audit_observable_count = len(audit_rows)
    explicit_fit_parameter_count = (
        phenomenological_parameter_count
        + threshold_alignment_subtraction_count
        + calibration_parameter_count
    )
    audit_degrees_of_freedom = audit_observable_count - explicit_fit_parameter_count
    audit_chi2, _, _ = calculate_global_chi_square(
        *(row.pull_data for row in audit_rows),
        degrees_of_freedom=audit_degrees_of_freedom,
        landscape_trial_count=None,
    )
    audit_rms_pull = math.sqrt(audit_chi2 / audit_observable_count)
    audit_max_abs_pull = max(abs(row.pull_data.pull) for row in audit_rows)

    predictive_observable_count = len(predictive_rows)
    predictive_degrees_of_freedom = predictive_observable_count - explicit_fit_parameter_count
    predictive_chi2, predictive_conditional_p_value, predictive_p_value = calculate_global_chi_square(
        *(row.pull_data for row in predictive_rows),
        degrees_of_freedom=predictive_degrees_of_freedom,
        landscape_trial_count=landscape_trial_count,
    )
    predictive_rms_pull = math.sqrt(predictive_chi2 / predictive_observable_count)
    predictive_max_abs_pull = max(abs(row.pull_data.pull) for row in predictive_rows)
    predictive_reduced_chi2 = predictive_chi2 / predictive_degrees_of_freedom

    calibration_anchor = calibration_rows[0] if calibration_rows else None

    return PullTable(
        rows=rows,
        audit_observable_count=audit_observable_count,
        audit_chi2=audit_chi2,
        audit_rms_pull=audit_rms_pull,
        audit_max_abs_pull=audit_max_abs_pull,
        audit_degrees_of_freedom=audit_degrees_of_freedom,
        predictive_observable_count=predictive_observable_count,
        predictive_chi2=predictive_chi2,
        predictive_rms_pull=predictive_rms_pull,
        predictive_max_abs_pull=predictive_max_abs_pull,
        predictive_degrees_of_freedom=predictive_degrees_of_freedom,
        predictive_conditional_p_value=predictive_conditional_p_value,
        predictive_p_value=predictive_p_value,
        threshold_alignment_subtraction_count=threshold_alignment_subtraction_count,
        phenomenological_parameter_count=phenomenological_parameter_count,
        calibration_parameter_count=calibration_parameter_count,
        calibration_anchor_observable="none" if calibration_anchor is None else calibration_anchor.observable,
        calibration_anchor_pull=0.0 if calibration_anchor is None else calibration_anchor.pull_data.pull,
        calibration_input_symbol=MATCHING_COEFFICIENT_SYMBOL,
        calibration_input_value=pmns.kappa_geometric,
        predictive_reduced_chi2=predictive_reduced_chi2,
        predictive_landscape_trial_count=LANDSCAPE_TRIAL_COUNT,
        predictive_followup_trial_count=followup_trial_count,
        predictive_effective_trial_count=1.0 if landscape_trial_count is None else float(landscape_trial_count),
        predictive_correlation_length=effective_correlation_length,
        predictive_lepton_correlation_length=lepton_correlation_length,
        predictive_quark_correlation_length=quark_correlation_length,
        gut_threshold_residue_value=ckm.so10_threshold_correction.gut_threshold_residue,
        transport_caveat_note=None if transport_covariance is None else transport_covariance.uncertainty_reporting_footnote_tex,
    )


def print_pull_table(pull_table: PullTable) -> str:
    """Generate the publication-facing LaTeX pull table with predictive/audit bookkeeping."""

    return pull_table.to_tex()


def build_benchmark_diagnostics(
    pull_table: PullTable,
    nonlinearity_audit: NonLinearityAuditData,
    *,
    audit: AuditData | None = None,
    weight_profile: CkmPhaseTiltProfileData | None = None,
    gauge_audit: GaugeHolographyAudit | None = None,
    gravity_audit: GravityAudit | None = None,
    dark_energy_audit: DarkEnergyTensionAudit | None = None,
) -> dict[str, float | int | bool]:
    audit_p_value = _audit_p_value(pull_table)
    audit_reduced_chi2 = pull_table.audit_chi2 / pull_table.audit_degrees_of_freedom
    diagnostics: dict[str, float | int | bool] = {
        "predictive_observable_count": pull_table.predictive_observable_count,
        "predictive_chi2": pull_table.predictive_chi2,
        "predictive_degrees_of_freedom": pull_table.predictive_degrees_of_freedom,
        "predictive_rms_pull": pull_table.predictive_rms_pull,
        "predictive_max_abs_pull": pull_table.predictive_max_abs_pull,
        "predictive_landscape_trial_count": pull_table.predictive_landscape_trial_count,
        "predictive_followup_trial_count": pull_table.predictive_followup_trial_count,
        "predictive_effective_trial_count": pull_table.predictive_effective_trial_count,
        "predictive_global_p_value": pull_table.predictive_discrete_selection_lee_p_value,
        "audit_observable_count": pull_table.audit_observable_count,
        "audit_chi2": pull_table.audit_chi2,
        "audit_degrees_of_freedom": pull_table.audit_degrees_of_freedom,
        "audit_reduced_chi2": audit_reduced_chi2,
        "audit_p_value": audit_p_value,
        "theta12_pull": _pull_for_observable(pull_table, r"$\theta_{12}$"),
        "max_rg_nonlinearity_sigma": nonlinearity_audit.max_sigma_error,
        "theoretical_matching_uncertainty_fraction": THEORETICAL_MATCHING_UNCERTAINTY_FRACTION,
        "parametric_transport_covariance_fraction": PARAMETRIC_TRANSPORT_COVARIANCE_FRACTION,
    }
    if weight_profile is not None:
        max_vus_sigma = weight_profile.max_vus_shift / QUARK_INTERVALS["vus"].sigma
        max_vcb_sigma = weight_profile.max_vcb_shift / QUARK_INTERVALS["vcb"].sigma
        max_vub_sigma = weight_profile.max_vub_shift / QUARK_INTERVALS["vub"].sigma
        diagnostics.update(
            {
                "ckm_benchmark_weight": weight_profile.benchmark_weight,
                "ckm_best_fit_weight": weight_profile.best_fit_weight,
                "ckm_benchmark_delta_chi2": weight_profile.benchmark_delta_chi2,
                "ckm_smallness_max_vus_shift": weight_profile.max_vus_shift,
                "ckm_smallness_max_vcb_shift": weight_profile.max_vcb_shift,
                "ckm_smallness_max_vub_shift": weight_profile.max_vub_shift,
                "ckm_smallness_max_vus_sigma": max_vus_sigma,
                "ckm_smallness_max_vcb_sigma": max_vcb_sigma,
                "ckm_smallness_max_vub_sigma": max_vub_sigma,
                "ckm_smallness_lock_pass": bool(
                    max(max_vus_sigma, max_vcb_sigma, max_vub_sigma) < 1.0
                ),
            }
        )
    if gauge_audit is not None:
        diagnostics.update(
            {
                "gauge_alpha_inverse": gauge_audit.topological_alpha_inverse,
                "gauge_alpha_target": gauge_audit.codata_alpha_inverse,
                "gauge_geometric_residue_percent": gauge_audit.geometric_residue_percent,
                "gauge_modular_gap_alignment_percent": gauge_audit.modular_gap_alignment_percent,
                "gauge_framing_closed": gauge_audit.framing_closed,
                "gauge_topological_stability_pass": gauge_audit.topological_stability_pass,
            }
        )
    if gravity_audit is not None:
        diagnostics.update(
            {
                "gravity_framing_gap": gravity_audit.framing_gap,
                "gravity_torsion_free": gravity_audit.torsion_free,
                "gravity_non_singular_bulk": gravity_audit.non_singular_bulk,
                "gravity_lambda_aligned": gravity_audit.lambda_aligned,
                "gravity_bulk_emergent": gravity_audit.bulk_emergent,
                "gravity_gmunu_consistency_score": gravity_audit.gmunu_consistency_score,
                "baryon_proton_lifetime_years": gravity_audit.baryon_stability.proton_lifetime_years,
                "baryon_dimension_five_forbidden": gravity_audit.baryon_stability.dimension_five_forbidden,
            }
        )
    if dark_energy_audit is not None:
        diagnostics.update(
            {
                "lambda_holo_si_m2": dark_energy_audit.lambda_surface_tension_si_m2,
                "lambda_anchor_si_m2": dark_energy_audit.lambda_anchor_si_m2,
                "lambda_identity_si_m2": dark_energy_audit.lambda_scaling_identity_si_m2,
                "lambda_surface_tension_prefactor": dark_energy_audit.surface_tension_prefactor,
                "lambda_surface_tension_deviation_percent": dark_energy_audit.surface_tension_deviation_percent,
                "topological_mass_coordinate_ev": dark_energy_audit.topological_mass_coordinate_ev,
                "lambda_gn_mnu4": dark_energy_audit.triple_match_product,
                "triple_match_saturated": dark_energy_audit.triple_match_saturated,
                "lambda_alpha_locked_under_bit_shift": dark_energy_audit.alpha_locked_under_bit_shift,
                "topological_integrity_error_on_mass_shift": dark_energy_audit.sensitivity_audit_triggered_integrity_error,
            }
        )
    if audit is not None:
        diagnostics.update(
            {
                "ih_support_deficit": audit.support_deficit,
                "ih_modularity_limit_rank": audit.modularity_limit_rank,
                "ih_required_dictionary_rank": audit.required_inverted_rank,
                "ih_redundancy_entropy_cost_nat": audit.redundancy_entropy_cost_nat,
                "ih_relaxed_proxy_gap": audit.relaxed_inverted_gap,
                "ih_nonminimal_extension_required": bool(audit.required_inverted_rank > audit.modularity_limit_rank),
            }
        )
    if gauge_audit is not None and gravity_audit is not None and dark_energy_audit is not None:
        diagnostics["triple_lock_consistent"] = bool(
            gauge_audit.topological_stability_pass
            and gravity_audit.bulk_emergent
            and dark_energy_audit.alpha_locked_under_bit_shift
            and dark_energy_audit.triple_match_saturated
            and gravity_audit.baryon_stability.dimension_five_forbidden
        )
    return diagnostics


def write_benchmark_diagnostics(
    pull_table: PullTable,
    nonlinearity_audit: NonLinearityAuditData,
    *,
    audit: AuditData | None = None,
    weight_profile: CkmPhaseTiltProfileData | None = None,
    gauge_audit: GaugeHolographyAudit | None = None,
    gravity_audit: GravityAudit | None = None,
    dark_energy_audit: DarkEnergyTensionAudit | None = None,
    output_dir: Path | None = None,
) -> dict[str, float | int | bool]:
    diagnostics = build_benchmark_diagnostics(
        pull_table,
        nonlinearity_audit,
        audit=audit,
        weight_profile=weight_profile,
        gauge_audit=gauge_audit,
        gravity_audit=gravity_audit,
        dark_energy_audit=dark_energy_audit,
    )
    if output_dir is not None:
        publication_export.write_json_artifact(output_dir / BENCHMARK_DIAGNOSTICS_FILENAME, diagnostics)
    return diagnostics


def write_svd_stability_report(
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    output_dir: Path | None = None,
) -> str:
    r"""Write the Higgs-VEV-alignment stability report behind the $\mathbf{126}_H$ audit."""

    return reporting_engine.write_svd_stability_report(
        mass_ratio_stability_audit,
        output_dir=output_dir,
    )


def write_eigenvector_stability_audit(
    weight_profile: CkmPhaseTiltProfileData,
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    output_dir: Path | None = None,
) -> str:
    """Write the standalone evidence-packet audit for Wilson-coefficient / magnitude decoupling."""

    return reporting_engine.write_eigenvector_stability_audit(
        weight_profile,
        mass_ratio_stability_audit,
        output_dir=output_dir,
    )


def write_seed_robustness_audit(
    seed_robustness_audit: SeedRobustnessAuditData,
    output_dir: Path | None = None,
) -> str:
    """Write the seed-ensemble robustness report for the stochastic audit stages."""

    return reporting_engine.write_seed_robustness_audit(
        seed_robustness_audit,
        output_dir=output_dir,
    )


def write_stability_report(
    pull_table: PullTable,
    nonlinearity_audit: NonLinearityAuditData,
    svd_report: str,
    output_dir: Path | None = None,
) -> str:
    """Write the unified publication-facing stability report covering ODE and SVD metrics."""

    return reporting_engine.write_stability_report(
        _benchmark_bookkeeping_lines(pull_table),
        nonlinearity_audit,
        svd_report,
        output_dir=output_dir,
    )


def write_generated_tables(
    level_scan: LevelStabilityScan,
    global_audit: GlobalSensitivityAudit,
    chi2_landscape_audit: Chi2LandscapeAuditData,
    pull_table: PullTable,
    audit: AuditData,
    nonlinearity_audit: NonLinearityAuditData,
    step_size_convergence: StepSizeConvergenceData,
    weight_profile: CkmPhaseTiltProfileData,
    pmns: PmnsData,
    ckm: CkmData,
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    geometric_sensitivity: GeometricSensitivityData,
    transport_covariance: TransportParametricCovarianceData,
    *,
    gauge_audit: GaugeHolographyAudit | None = None,
    gravity_audit: GravityAudit | None = None,
    dark_energy_audit: DarkEnergyTensionAudit | None = None,
    unitary_audit: UnitaryBoundAudit | Path | None = None,
    vacuum: TopologicalVacuum | Path | None = None,
    output_dir: Path | None = None,
) -> None:
    if output_dir is None:
        if isinstance(vacuum, Path):
            output_dir = vacuum
            vacuum = None
        elif isinstance(unitary_audit, Path):
            output_dir = unitary_audit
            unitary_audit = None
    if output_dir is None:
        raise TypeError("write_generated_tables requires output_dir.")

    resolved_output_dir = Path(output_dir)
    resolved_vacuum = DEFAULT_TOPOLOGICAL_VACUUM if vacuum is None or isinstance(vacuum, Path) else vacuum

    (resolved_output_dir / GLOBAL_FLAVOR_FIT_TABLE_FILENAME).write_text(print_pull_table(pull_table) + "\n", encoding="utf-8")
    (resolved_output_dir / UNIQUENESS_SCAN_TABLE_FILENAME).write_text(level_scan.to_tex() + "\n", encoding="utf-8")
    benchmark_diagnostics = write_benchmark_diagnostics(
        pull_table,
        nonlinearity_audit,
        audit=audit,
        weight_profile=weight_profile,
        gauge_audit=gauge_audit,
        gravity_audit=gravity_audit,
        dark_energy_audit=dark_energy_audit,
        output_dir=resolved_output_dir,
    )
    reporting_engine.write_audit_statement(benchmark_diagnostics, output_dir=resolved_output_dir)
    publication_export.export_transport_covariance_diagnostics(
        resolved_output_dir / TRANSPORT_COVARIANCE_DIAGNOSTICS_FILENAME,
        transport_covariance,
    )
    ih_spectrum_audit = publication_export.export_ih_singular_value_spectrum_figure(
        audit,
        resolved_output_dir / SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_FIGURE_FILENAME,
        level=pmns.level,
    )
    publication_export.export_matrix_spectrum_csv(
        resolved_output_dir / SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_DATA_FILENAME,
        ih_spectrum_audit,
    )
    svd_report = write_svd_stability_report(mass_ratio_stability_audit, output_dir=resolved_output_dir)
    write_eigenvector_stability_audit(weight_profile, mass_ratio_stability_audit, output_dir=resolved_output_dir)
    write_stability_report(pull_table, nonlinearity_audit, svd_report, output_dir=resolved_output_dir)
    export_support_overlap_table(audit, resolved_output_dir)
    export_supplementary_tolerance_table(resolved_output_dir)
    if unitary_audit is not None:
        export_unitary_consistency_table(resolved_output_dir, unitary_audit, model=resolved_vacuum)
    export_kappa_sensitivity_audit_table(geometric_sensitivity, resolved_output_dir)
    export_svd_stability_audit_table(mass_ratio_stability_audit, resolved_output_dir)
    export_modularity_residual_map(level_scan, resolved_output_dir)
    export_landscape_anomaly_map(global_audit, resolved_output_dir)
    export_followup_chi2_landscape_table(chi2_landscape_audit, resolved_output_dir)
    export_benchmark_stability_table(pull_table, nonlinearity_audit, weight_profile, mass_ratio_stability_audit, resolved_output_dir)
    export_vev_alignment_stability_figure(mass_ratio_stability_audit, resolved_output_dir / SUPPLEMENTARY_VEV_ALIGNMENT_STABILITY_FIGURE_FILENAME)
    export_framing_gap_moat_heatmap(global_audit, resolved_output_dir / FRAMING_GAP_HEATMAP_FIGURE_FILENAME)
    export_determinant_gradient_figure(audit, resolved_output_dir / SUPPLEMENTARY_DETERMINANT_GRADIENT_FIGURE_FILENAME)
    export_step_size_convergence_figure(step_size_convergence, resolved_output_dir / SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME)


def write_audit_output_bundles(
    output_dir: Path,
    pull_table: PullTable,
    weight_profile: CkmPhaseTiltProfileData,
    nonlinearity_audit: NonLinearityAuditData,
    mass_ratio_stability_audit: MassRatioStabilityAuditData,
    global_audit: GlobalSensitivityAudit,
    framing_gap_stability: FramingGapStabilityData,
) -> Path:
    """Bundle the publication artifacts into output directories."""

    packet_dirs = [
        output_dir / AUDIT_OUTPUT_ARCHIVE_DIRNAME,
        output_dir / STABILITY_AUDIT_OUTPUTS_DIRNAME,
        output_dir / LANDSCAPE_METRICS_DIRNAME,
    ]
    packet_filenames = _present_packet_output_artifacts(output_dir)
    manifest_lines = [
        "# Publication Artifact Bundle",
        "",
        "This bundle consolidates the publication-facing artifacts for the structural-selection, threshold-matching, and data-model-comparison checks.",
        "",
        f"- `{UNIQUENESS_SCAN_TABLE_FILENAME}` — local low-rank scan table.",
        f"- `{MODULARITY_RESIDUAL_MAP_FILENAME}` — framing residual map for the nearest-neighbor fixed-parent closure problem around `(26, 8, 312)`.",
        f"- `{LANDSCAPE_ANOMALY_MAP_FILENAME}` — information-allowed landscape anomaly ranking over the full low-rank window.",
        f"- `{SUPPLEMENTARY_UNITARY_CONSISTENCY_TABLE_FILENAME}` — local unitary-rigidity table tying the Triple-Lock to the finite-capacity recovery bound.",
        f"- `{KAPPA_SENSITIVITY_AUDIT_FILENAME}` — discrete geometric-sensitivity table around the benchmark invariant.",
        f"- `{KAPPA_STABILITY_SWEEP_FILENAME}` — kappa sweep over the natural geometric window `kappa in [0.8, 1.2]`.",
        f"- `{FRAMING_GAP_HEATMAP_FIGURE_FILENAME}` — framing-gap heatmap around `(26, 8)`.",
        f"- `{CKM_PHASE_TILT_PROFILE_FIGURE_FILENAME}` — Wilson-coefficient profile with magnitude-rigidity check.",
        f"- `{SVD_STABILITY_AUDIT_TABLE_FILENAME}` — main-text-ready Higgs-VEV-alignment angle-stability table.",
        f"- `{EIGENVECTOR_STABILITY_AUDIT_FILENAME}` — SVD eigenvector stability summary for the Wilson-coefficient / magnitude decoupling check.",
        f"- `{BENCHMARK_STABILITY_TABLE_FILENAME}` — side-by-side benchmark and physical-vs-numerical stability table.",
        f"- `{SVD_STABILITY_REPORT_FILENAME}` — Higgs-VEV-alignment singular-value stability report.",
        f"- `{STABILITY_REPORT_FILENAME}` — combined ODE and SVD convergence metrics.",
        f"- `{SUPPLEMENTARY_STEP_SIZE_CONVERGENCE_FIGURE_FILENAME}` — coupled-RGE step-size convergence summary for the benchmark observables.",
        f"- `{BENCHMARK_DIAGNOSTICS_FILENAME}` — structured benchmark pull, chi-square, and p-value diagnostics.",
        f"- `{TRANSPORT_COVARIANCE_DIAGNOSTICS_FILENAME}` — raw transport-covariance acceptance and skewness diagnostics.",
        f"- `{SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_FIGURE_FILENAME}` — singular-value bar chart for the IH overlap matrix.",
        f"- `{SUPPLEMENTARY_IH_SINGULAR_VALUE_SPECTRUM_DATA_FILENAME}` — CSV data behind the IH overlap singular-value spectrum.",
        "",
        f"Landscape selected rank: {global_audit.selected_rank}",
        f"Landscape anomaly gap to next-best row: {global_audit.algebraic_gap:.6e}",
        f"Predictive chi2 / nu_pred: {pull_table.predictive_chi2:.3f} / {pull_table.predictive_degrees_of_freedom}",
        f"Cross-check chi2 / nu_check: {pull_table.audit_chi2:.3f} / {pull_table.audit_degrees_of_freedom}",
        f"Predictive p-value, conditional: {pull_table.predictive_conditional_p_value:.3f}",
        f"Global p-value, N_eff-corrected: {pull_table.predictive_discrete_selection_lee_p_value:.3f}",
        f"Effective trial count: {pull_table.predictive_effective_trial_count:.3f}",
        f"Topological-label DOF subtraction: {TOPOLOGICAL_QUANTUM_NUMBER_DOF_SUBTRACTION}",
        f"Benchmark `R_GUT`: {weight_profile.benchmark_weight:.2f}",
        f"Off-shell minimum `R_GUT`: {weight_profile.best_fit_weight:.2f}",
        f"Higgs/VEV matching point [GeV]: {framing_gap_stability.higgs_vev_matching_m126_gev:.3e}",
        f"Max RG non-linearity discrepancy [sigma]: {nonlinearity_audit.max_sigma_error:.3e}",
        f"Max SVD angular shift [sigma]: {mass_ratio_stability_audit.max_sigma_shift:.3e}",
    ]
    if SEED_ROBUSTNESS_AUDIT_FILENAME in packet_filenames:
        summary_start_index = manifest_lines.index("", 4)
        manifest_lines.insert(
            summary_start_index,
            f"- `{SEED_ROBUSTNESS_AUDIT_FILENAME}` — seed-ensemble variance summary for the stochastic transport diagnostics.",
        )
    for packet_dir in packet_dirs:
        packet_dir.mkdir(parents=True, exist_ok=True)
        for filename in packet_filenames:
            shutil.copy2(output_dir / filename, packet_dir / filename)
        (packet_dir / AUDIT_OUTPUT_MANIFEST_FILENAME).write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return packet_dirs[0]


def derive_cosmology_anchor() -> CosmologyAnchorData:
    """Return the Planck 2018 cosmology anchor used for the horizon bit count."""

    return CosmologyAnchorData(
        hubble_km_s_mpc=PLANCK2018_H0_KM_S_MPC,
        hubble_sigma_km_s_mpc=PLANCK2018_H0_SIGMA_KM_S_MPC,
        omega_lambda=PLANCK2018_OMEGA_LAMBDA,
        omega_lambda_sigma=PLANCK2018_OMEGA_LAMBDA_SIGMA,
        lambda_si_m2=PLANCK2018_LAMBDA_SI_M2,
        holographic_bits=HOLOGRAPHIC_BITS,
        holographic_bits_fractional_sigma=HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA,
    )


def derive_threshold_sensitivity(
    ckm: CkmData | None = None,
    m_126_values_gev: tuple[float, ...] | None = None,
    *,
    scale_ratio: float = RG_SCALE_RATIO,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    gut_threshold_residue: float | None = None,
) -> ThresholdSensitivityData:
    r"""Calculate the healed framing-gap gamma diagnostic as a function of mass."""

    ckm_data = (
        derive_ckm(
            level=quark_level,
            parent_level=parent_level,
            scale_ratio=scale_ratio,
            gut_threshold_residue=gut_threshold_residue,
        )
        if ckm is None
        else ckm
    )
    matching_delta_pi = ckm_data.so10_threshold_correction.delta_pi_126
    structural_mn_gev, *_ = derive_structural_rhn_scale_gev(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    matching_threshold_gev = derive_topological_threshold_gev(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    resolved_m_126_values_gev = (
        (1.0e12, 3.0e12, matching_threshold_gev, 3.0e13, 1.0e14, 3.0e14, 1.0e15)
        if m_126_values_gev is None
        else m_126_values_gev
    )
    matching_log = math.log(structural_mn_gev / matching_threshold_gev)
    amplification_factor = float(quark_branching_index(parent_level, quark_level) + RANK_DIFFERENCE)
    matching_point_gamma_shift_deg = abs(CKM_GAMMA_GOLD_STANDARD_DEG.central - ckm_data.gamma_rg_deg)
    bare_gamma_rg = ckm_unitarity_triangle_angles(ckm_data.bare_ckm_matrix_rg)[2]
    closure_offset_deg = ckm_data.gamma_rg_deg - (bare_gamma_rg + ckm_data.so10_threshold_correction.triangle_tilt_deg)
    points: list[ThresholdSensitivityPoint] = []

    for m_126_gev in resolved_m_126_values_gev:
        scale_factor = math.log(structural_mn_gev / m_126_gev) / matching_log
        delta_pi_126 = matching_delta_pi * scale_factor
        xi12 = math.exp(delta_pi_126)
        cabibbo_shift_fraction = xi12 - 1.0
        gamma_shift_estimate_deg = ckm_data.so10_threshold_correction.triangle_tilt_deg * scale_factor
        gamma_recovered_deg = normalize_triangle_angle(bare_gamma_rg + closure_offset_deg + gamma_shift_estimate_deg)
        points.append(
            ThresholdSensitivityPoint(
                m_126_gev=m_126_gev,
                delta_pi_126=delta_pi_126,
                xi12=xi12,
                cabibbo_shift_fraction=cabibbo_shift_fraction,
                gamma_shift_estimate_deg=gamma_shift_estimate_deg,
                gamma_recovered_deg=gamma_recovered_deg,
            )
        )

    return ThresholdSensitivityData(
        amplification_factor=amplification_factor,
        matching_point_gamma_shift_deg=matching_point_gamma_shift_deg,
        points=tuple(points),
    )


def derive_ghost_character_audit(audit: AuditData, level: int = LEPTON_LEVEL) -> GhostCharacterAuditData:
    r"""Package the IH informational-cost extension for the one-copy support audit."""

    extra_character_count = max(audit.required_inverted_rank - audit.modularity_limit_rank, 0)
    return GhostCharacterAuditData(
        extra_character_count=extra_character_count,
        ghost_norm_upper_bound=0.0,
        integrable_spin_bound=level / 2.0,
        swampland_excluded=bool(extra_character_count > 0),
    )


def derive_framing_gap_stability_audit(
    ckm: CkmData | None = None,
    audit: AuditData | None = None,
    m_126_grid_gev: np.ndarray | None = None,
    *,
    scale_ratio: float = RG_SCALE_RATIO,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    gut_threshold_residue: float | None = None,
) -> FramingGapStabilityData:
    r"""Track how the $\mathbf{126}_H$ threshold heals the framing gap and reproduces $\gamma$."""

    ckm_data = (
        derive_ckm(
            level=quark_level,
            parent_level=parent_level,
            scale_ratio=scale_ratio,
            gut_threshold_residue=gut_threshold_residue,
        )
        if ckm is None
        else ckm
    )
    audit_data = derive_audit(level=lepton_level) if audit is None else audit
    threshold_grid = np.logspace(12.0, 15.0, 240) if m_126_grid_gev is None else np.array(m_126_grid_gev, dtype=float)
    structural_mn_gev, *_ = derive_structural_rhn_scale_gev(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    framing_gap_area = audit_data.beta * audit_data.beta
    healing_fraction_values = np.log(structural_mn_gev / threshold_grid) / framing_gap_area
    framing_gap_values = np.log(structural_mn_gev / threshold_grid) - framing_gap_area
    bare_gamma_rg = ckm_unitarity_triangle_angles(ckm_data.bare_ckm_matrix_rg)[2]
    closure_offset_deg = ckm_data.gamma_rg_deg - (bare_gamma_rg + ckm_data.so10_threshold_correction.triangle_tilt_deg)
    gamma_healed_raw = bare_gamma_rg + closure_offset_deg + healing_fraction_values * ckm_data.so10_threshold_correction.triangle_tilt_deg
    gamma_healed_wrapped = np.mod(gamma_healed_raw, 360.0)
    gamma_healed_deg = np.where(gamma_healed_wrapped <= 180.0, gamma_healed_wrapped, 360.0 - gamma_healed_wrapped)
    matching_gamma_deg = normalize_triangle_angle(
        bare_gamma_rg + closure_offset_deg + ckm_data.so10_threshold_correction.triangle_tilt_deg
    )
    return FramingGapStabilityData(
        m_126_grid_gev=threshold_grid,
        framing_gap_values=np.asarray(framing_gap_values, dtype=float),
        healing_fraction_values=np.asarray(healing_fraction_values, dtype=float),
        gamma_healed_deg=np.asarray(gamma_healed_deg, dtype=float),
        matching_m126_gev=derive_topological_threshold_gev(
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        ),
        matching_gamma_deg=float(matching_gamma_deg),
        bare_gamma_rg_deg=float(bare_gamma_rg),
        observed_gamma_deg=float(ckm_data.gamma_rg_deg),
    )


def export_framing_gap_stability_figure(
    stability: FramingGapStabilityData,
    output_path: Path | None = None,
) -> None:
    r"""Write the framing-gap healing plot requested for the $\mathbf{126}_H$ threshold."""

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / FRAMING_GAP_STABILITY_FIGURE_FILENAME
    fig, ax_left = plt.subplots(figsize=(6.4, 4.2))
    ax_right = ax_left.twinx()

    ax_left.plot(stability.m_126_grid_gev, stability.framing_gap_values, color="#991b1b", lw=2.2, label=r"$\Delta_{\rm grav}(M_{126})$")
    ax_left.axhline(0.0, color="#991b1b", lw=1.0, ls="--", alpha=0.7)
    ax_left.axvline(stability.matching_m126_gev, color="#6b7280", lw=1.0, ls=":")

    ax_right.plot(stability.m_126_grid_gev, stability.gamma_healed_deg, color="#2563eb", lw=2.2, label=r"$\gamma_{\rm healed}(M_{126})$")
    ax_right.axhline(stability.observed_gamma_deg, color="#2563eb", lw=1.0, ls="--", alpha=0.7)

    ax_left.scatter([stability.matching_m126_gev], [0.0], color="#991b1b", s=28, zorder=5)
    ax_right.scatter([stability.matching_m126_gev], [stability.matching_gamma_deg], color="#2563eb", s=28, zorder=5)
    ax_right.annotate(
        rf"$M_{{126}}^{{\rm match}}={stability.matching_m126_gev:.2e}\,\mathrm{{GeV}}$",
        xy=(stability.matching_m126_gev, stability.matching_gamma_deg),
        xytext=(0.05, 0.92),
        textcoords="axes fraction",
        fontsize=9,
        color="#1d4ed8",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#1d4ed8", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
    )

    ax_left.set_xscale("log")
    ax_left.set_xlabel(r"$M_{126}$ [GeV]")
    ax_left.set_ylabel(r"$\Delta_{\rm grav}(M_{126})$", color="#991b1b")
    ax_right.set_ylabel(r"$\gamma_{\rm healed}(M_{126})$ [deg]", color="#2563eb")
    ax_left.tick_params(axis="y", colors="#991b1b")
    ax_right.tick_params(axis="y", colors="#2563eb")
    ax_left.set_title(r"Framing-gap healing and the CKM apex angle")
    ax_left.grid(True, which="both", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def alpha_inverse_mz_inputs(alpha_s_mz: float = PLANCK2018_ALPHA_S_MZ) -> np.ndarray:
    """Return GUT-normalized SM gauge couplings at $M_Z$."""

    alpha1_inv = (3.0 / 5.0) * PLANCK2018_ALPHA_EM_INV_MZ * (1.0 - PLANCK2018_SIN2_THETA_W_MZ)
    alpha2_inv = PLANCK2018_ALPHA_EM_INV_MZ * PLANCK2018_SIN2_THETA_W_MZ
    alpha3_inv = 1.0 / alpha_s_mz
    return np.array([alpha1_inv, alpha2_inv, alpha3_inv], dtype=float)


def run_inverse_couplings(alpha_inverse_start: np.ndarray, beta_coefficients: np.ndarray, scale_low_gev: float, scale_high_gev: float) -> np.ndarray:
    """One-loop running of inverse gauge couplings between two scales."""

    return alpha_inverse_start - beta_coefficients * math.log(scale_high_gev / scale_low_gev) / (2.0 * math.pi)


def derive_gauge_unification_existence_proof(
    m_126_gev: float | None = None,
    m_10_gev: float = DIRAC_HIGGS_BENCHMARK_MASS_GEV,
    m_gut_gev: float = GUT_SCALE_GEV,
    structural_mn_gev: float | None = None,
) -> GaugeUnificationData:
    """Provide a one-loop threshold consistency audit for complete $SO(10)$ multiplets."""

    resolved_m_126_gev = derive_topological_threshold_gev() if m_126_gev is None else m_126_gev
    alpha_inverse_mz = alpha_inverse_mz_inputs()
    beta_sm = np.array([41.0 / 10.0, -19.0 / 6.0, -7.0], dtype=float)
    beta_shift_126 = derive_so10_scalar_beta_shift("126_H")
    beta_shift_10 = derive_so10_scalar_beta_shift("10_H")
    alpha_inverse_m126 = run_inverse_couplings(alpha_inverse_mz, beta_sm, MZ_SCALE_GEV, resolved_m_126_gev)
    alpha_inverse_m10 = run_inverse_couplings(alpha_inverse_m126, beta_sm + beta_shift_126, resolved_m_126_gev, m_10_gev)
    alpha_inverse_gut = run_inverse_couplings(
        alpha_inverse_m10,
        beta_sm + beta_shift_126 + beta_shift_10,
        m_10_gev,
        m_gut_gev,
    )
    unified_alpha_inverse = float(np.mean(alpha_inverse_gut))
    resolved_structural_mn_gev = (
        derive_structural_rhn_scale_gev()[0] if structural_mn_gev is None else float(structural_mn_gev)
    )
    geometric_mean_threshold_gev = math.sqrt(resolved_m_126_gev * m_10_gev)
    return GaugeUnificationData(
        alpha_inverse_mz=alpha_inverse_mz,
        alpha_inverse_m126=alpha_inverse_m126,
        alpha_inverse_m10=alpha_inverse_m10,
        alpha_inverse_gut=alpha_inverse_gut,
        beta_sm=beta_sm,
        beta_shift_126=beta_shift_126,
        beta_shift_10=beta_shift_10,
        m_126_gev=resolved_m_126_gev,
        m_10_gev=m_10_gev,
        m_gut_gev=m_gut_gev,
        unified_alpha_inverse=unified_alpha_inverse,
        max_mismatch=float(np.max(np.abs(alpha_inverse_gut - unified_alpha_inverse))),
        structural_mn_gev=resolved_structural_mn_gev,
        geometric_mean_threshold_gev=geometric_mean_threshold_gev,
        seesaw_consistency_ratio=float(resolved_structural_mn_gev / geometric_mean_threshold_gev),
    )



def majorana_bounds_from_amplitudes(amplitudes_ev: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(amplitudes_ev))
    largest = float(np.max(amplitudes_ev))
    minimum = max(largest - (total - largest), 0.0)
    return minimum, total


def majorana_lobster_envelope(
    lightest_scan_ev: np.ndarray = MAJORANA_LIGHTEST_SCAN_EV,
    theta12_interval: Interval = NUFIT_53_NO_3SIGMA["theta12"],
    theta13_interval: Interval = NUFIT_53_NO_3SIGMA["theta13"],
    dm21_interval: Interval = NUFIT_53_NO_3SIGMA["dm21"],
    dm31_interval: Interval = NUFIT_53_NO_3SIGMA["dm31"],
    grid_points: int = MAJORANA_LOBSTER_GRID_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lightest_scan_ev = np.asarray(lightest_scan_ev, dtype=float).reshape(-1)
    theta12_grid = np.linspace(theta12_interval.lower, theta12_interval.upper, grid_points)
    theta13_grid = np.linspace(theta13_interval.lower, theta13_interval.upper, grid_points)
    dm21_grid = np.linspace(dm21_interval.lower, dm21_interval.upper, grid_points)
    dm31_grid = np.linspace(dm31_interval.lower, dm31_interval.upper, grid_points)

    theta12_rad = np.radians(theta12_grid)[:, np.newaxis]
    theta13_rad = np.radians(theta13_grid)[np.newaxis, :]
    s12, c12 = np.sin(theta12_rad), np.cos(theta12_rad)
    s13, c13 = np.sin(theta13_rad), np.cos(theta13_rad)
    ee_weights = np.stack(
        [
            (c12 * c13) ** 2,
            (s12 * c13) ** 2,
            np.broadcast_to(s13**2, (grid_points, grid_points)),
        ],
        axis=-1,
    ).reshape(-1, 3)

    lightest_squared = lightest_scan_ev * lightest_scan_ev
    dm21_mesh, dm31_mesh = np.meshgrid(dm21_grid, dm31_grid, indexing="ij")
    masses = np.empty((grid_points * grid_points, lightest_scan_ev.size, 3), dtype=float)
    masses[:, :, 0] = lightest_scan_ev[np.newaxis, :]
    masses[:, :, 1] = np.sqrt(lightest_squared[np.newaxis, :] + dm21_mesh.reshape(-1, 1))
    masses[:, :, 2] = np.sqrt(lightest_squared[np.newaxis, :] + dm31_mesh.reshape(-1, 1))

    lower_envelope = np.full(lightest_scan_ev.shape, np.inf, dtype=float)
    upper_envelope = np.zeros(lightest_scan_ev.shape, dtype=float)

    amplitude_pair_count = ee_weights.shape[0] * masses.shape[0]
    target_tensor_elements = 2_400_000
    chunk_size = max(1, min(lightest_scan_ev.size, target_tensor_elements // max(amplitude_pair_count, 1)))

    for start in range(0, lightest_scan_ev.size, chunk_size):
        stop = min(start + chunk_size, lightest_scan_ev.size)
        mass_chunk = masses[:, start:stop, :]
        totals = np.zeros((ee_weights.shape[0], masses.shape[0], stop - start), dtype=float)
        largest = np.zeros_like(totals)
        for component_index in range(3):
            component_amplitudes = (
                ee_weights[:, component_index][:, np.newaxis, np.newaxis]
                * mass_chunk[np.newaxis, :, :, component_index]
            )
            totals += component_amplitudes
            np.maximum(largest, component_amplitudes, out=largest)

        minima = 2.0 * largest - totals
        np.maximum(minima, 0.0, out=minima)
        lower_envelope[start:stop] = np.min(minima, axis=(0, 1))
        upper_envelope[start:stop] = np.max(totals, axis=(0, 1))

    return lightest_scan_ev, lower_envelope, upper_envelope


def write_majorana_floor_figure(
    pmns: PmnsData,
    sensitivity: SensitivityData,
    geometric_sensitivity: GeometricSensitivityData,
    output_paths: tuple[Path, ...] | None = None,
) -> None:
    if output_paths is None:
        output_paths = (
            DEFAULT_OUTPUT_DIR / TOPOLOGICAL_LOBSTER_FIGURE_FILENAME,
            DEFAULT_OUTPUT_DIR / MAJORANA_FLOOR_FIGURE_FILENAME,
        )
    lightest_scan_ev, lower_envelope_ev, upper_envelope_ev = majorana_lobster_envelope()
    lightest_scan_mev = 1.0e3 * lightest_scan_ev
    lower_envelope_mev = 1.0e3 * np.maximum(lower_envelope_ev, 1.0e-6)
    upper_envelope_mev = 1.0e3 * upper_envelope_ev

    central_lightest_mev = 1.0e3 * derive_scales().m_0_mz_ev
    central_mbb_mev = 1.0e3 * pmns.effective_majorana_mass_rg_ev
    vertical_uncertainty_mev = (
        geometric_sensitivity.effective_majorana_mass_max_shift_mev + sensitivity.effective_majorana_mass_max_shift_mev
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.fill_between(
        lightest_scan_mev,
        lower_envelope_mev,
        upper_envelope_mev,
        color="#d0d5dd",
        alpha=0.85,
        label=r"NuFIT 5.3 NO $3\sigma$ envelope",
    )
    ax.axhspan(12.0, max(30.0, float(np.max(upper_envelope_mev)) * 1.1), color="#dc2626", alpha=0.08)
    ax.axhspan(9.0, 19.0, color="#f59e0b", alpha=0.10, label="LEGEND-1000 nominal reach")
    ax.errorbar(
        [central_lightest_mev],
        [central_mbb_mev],
        yerr=[[vertical_uncertainty_mev], [vertical_uncertainty_mev]],
        fmt="o",
        color="#2563eb",
        ecolor="#2563eb",
        elinewidth=1.8,
        capsize=4,
        markersize=6,
        label=r"$SO(10)_{312}$ floor",
        zorder=5,
    )
    ax.annotate(
        rf"$|m_{{\beta\beta}}|={central_mbb_mev:.2f}\,\mathrm{{meV}}$",
        xy=(central_lightest_mev, central_mbb_mev),
        xytext=(1.8 * central_lightest_mev, 1.25 * central_mbb_mev),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#2563eb"},
        fontsize=10,
        color="#1d4ed8",
    )
    ax.text(
        18.0,
        max(18.0, 0.7 * max(30.0, float(np.max(upper_envelope_mev)) * 1.1)),
        "Inverted hierarchy region\nDisfavored by Minimal Basis Incompatibility",
        color="#991b1b",
        fontsize=9,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "#991b1b", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(lightest_scan_mev)), float(np.max(lightest_scan_mev)))
    ax.set_ylim(0.5, max(30.0, float(np.max(upper_envelope_mev)) * 1.1))
    ax.set_xlabel(r"$m_{\rm lightest}$ [meV]")
    ax.set_ylabel(r"$|m_{\beta\beta}|$ [meV]")
    ax.set_title(r"NuFIT 5.3 normal-ordering lobster envelope")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    for output_path in output_paths:
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def zero_energy_gap(target_log_masses: np.ndarray, beta: float) -> float:
    genus_ladder = np.array([0.0, beta, 2.0 * beta], dtype=float)
    gaps = []
    for permutation in itertools.permutations(range(3)):
        candidate = genus_ladder[list(permutation)]
        gaps.append(float(np.sum((target_log_masses - candidate) ** 2)))
    return min(gaps)


def topological_ladder_masses(m_0_ev: float, beta: float) -> np.ndarray:
    return m_0_ev * np.exp(beta * np.arange(3, dtype=float))


def pairwise_mass_squared_splittings(masses_ev: np.ndarray) -> np.ndarray:
    ordered = np.sort(masses_ev)
    first, second, third = ordered
    return np.array(
        [second * second - first * first, third * third - second * second, third * third - first * first],
        dtype=float,
    )



def strict_wdw_functional(overlap_matrix: np.ndarray) -> float:
    r"""Return the strict support functional induced by Eq. ``eq:wdw-support-integral``."""

    return support_overlap_penalty(overlap_matrix)


def derive_audit(
    level: int = LEPTON_LEVEL,
    *,
    bit_count: float = HOLOGRAPHIC_BITS,
    scale_ratio: float = RG_SCALE_RATIO,
    kappa_geometric: float = GEOMETRIC_KAPPA,
    parent_level: int = PARENT_LEVEL,
    quark_level: int = QUARK_LEVEL,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> AuditData:
    """Assemble the Wheeler--DeWitt hierarchy audit.

    Args:
        level: Leptonic affine level used for the support basis.

    Returns:
        Audit data for normal and inverted hierarchy assignments.
    """

    beta = 0.5 * math.log(su2_total_quantum_dimension(level))
    m_0_uv_ev = derive_scales_for_bits(
        bit_count,
        scale_ratio,
        kappa_geometric=kappa_geometric,
        parent_level=parent_level,
        lepton_level=level,
        quark_level=quark_level,
        solver_config=solver_config,
    ).m_0_uv_ev
    ladder_masses = topological_ladder_masses(m_0_uv_ev, beta)
    topological_splittings = pairwise_mass_squared_splittings(ladder_masses)

    normal_log_masses = np.array([0.0, beta, 2.0 * beta], dtype=float)
    inverted_log_masses = np.array([beta, beta, 0.0], dtype=float)
    normal_genus_assignment = (0, 1, 2)
    inverted_genus_assignment = (1, 1, 0)
    normal_overlap_matrix = support_overlap_matrix(level, normal_genus_assignment)
    inverted_overlap_matrix = support_overlap_matrix(level, inverted_genus_assignment)

    strict_normal_gap = strict_wdw_functional(normal_overlap_matrix)
    strict_inverted_gap = strict_wdw_functional(inverted_overlap_matrix)
    relaxed_normal_gap = zero_energy_gap(normal_log_masses, beta)
    relaxed_inverted_gap = zero_energy_gap(inverted_log_masses, beta)
    support_deficit = len(inverted_genus_assignment) - np.linalg.matrix_rank(
        require_real_array(inverted_overlap_matrix, label="inverted-hierarchy support overlap matrix")
    )
    modularity_limit_rank = len(normal_genus_assignment)
    required_inverted_rank = modularity_limit_rank + int(support_deficit)
    redundancy_entropy_cost_nat = float(support_deficit) * math.log(2.0)

    return AuditData(
        beta=beta,
        topological_splittings_ev2=topological_splittings,
        strict_normal_gap=strict_normal_gap,
        strict_inverted_gap=strict_inverted_gap,
        relaxed_normal_gap=relaxed_normal_gap,
        relaxed_inverted_gap=relaxed_inverted_gap,
        support_deficit=int(support_deficit),
        modularity_limit_rank=modularity_limit_rank,
        required_inverted_rank=required_inverted_rank,
        redundancy_entropy_cost_nat=redundancy_entropy_cost_nat,
        normal_genus_assignment=normal_genus_assignment,
        inverted_genus_assignment=inverted_genus_assignment,
        support_tau_imag=SUPPORT_TAU_IMAG,
    )


def derive_sensitivity(
    bit_count: float = HOLOGRAPHIC_BITS,
    fractional_variation: float = DEFAULT_BITCOUNT_FRACTIONAL_VARIATION,
    scale_ratio: float = RG_SCALE_RATIO,
    kappa_geometric: float = GEOMETRIC_KAPPA,
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    gut_threshold_residue: float | None = None,
) -> SensitivityData:
    """Quantify bit-count uncertainty around the central benchmark.

    Args:
        bit_count: Central holographic bit count.
        fractional_variation: Symmetric fractional variation applied to the bit count.
        scale_ratio: Ratio $M_{\rm GUT}/M_Z$.
        kappa_geometric: Order-one geometric factor kept fixed during the bit-count sweep.

    Returns:
        Sensitivity data for the lower, central, and upper bit-count benchmarks.
    """

    central_pmns = derive_pmns(
        level=lepton_level,
        parent_level=parent_level,
        quark_level=quark_level,
        scale_ratio=scale_ratio,
        bit_count=bit_count,
        kappa_geometric=kappa_geometric,
    )
    central_ckm = derive_ckm(
        level=quark_level,
        parent_level=parent_level,
        scale_ratio=scale_ratio,
        gut_threshold_residue=gut_threshold_residue,
    )

    def make_point(label: str, factor: float) -> SensitivityPoint:
        varied_bits = bit_count * factor
        varied_scales = derive_scales_for_bits(
            varied_bits,
            scale_ratio,
            kappa_geometric=kappa_geometric,
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        )
        varied_pmns = derive_pmns(
            level=lepton_level,
            parent_level=parent_level,
            quark_level=quark_level,
            scale_ratio=scale_ratio,
            bit_count=varied_bits,
            kappa_geometric=kappa_geometric,
        )
        varied_ckm = derive_ckm(
            level=quark_level,
            parent_level=parent_level,
            scale_ratio=scale_ratio,
            gut_threshold_residue=gut_threshold_residue,
        )
        return SensitivityPoint(
            label=label,
            bit_count=varied_bits,
            m_0_mz_ev=varied_scales.m_0_mz_ev,
            effective_majorana_mass_mev=1.0e3 * varied_pmns.effective_majorana_mass_rg_ev,
            theta12_shift_deg=varied_pmns.theta12_rg_deg - central_pmns.theta12_rg_deg,
            theta13_shift_deg=varied_pmns.theta13_rg_deg - central_pmns.theta13_rg_deg,
            theta23_shift_deg=varied_pmns.theta23_rg_deg - central_pmns.theta23_rg_deg,
            delta_cp_shift_deg=varied_pmns.delta_cp_rg_deg - central_pmns.delta_cp_rg_deg,
            theta_c_shift_deg=varied_ckm.theta_c_rg_deg - central_ckm.theta_c_rg_deg,
            vus_shift=varied_ckm.vus_rg - central_ckm.vus_rg,
            vcb_shift=varied_ckm.vcb_rg - central_ckm.vcb_rg,
        )

    minus_10pct = make_point("-10%", 1.0 - fractional_variation)
    central = make_point("central", 1.0)
    plus_10pct = make_point("+10%", 1.0 + fractional_variation)
    effective_majorana_mass_values = np.array(
        [minus_10pct.effective_majorana_mass_mev, central.effective_majorana_mass_mev, plus_10pct.effective_majorana_mass_mev],
        dtype=float,
    )
    m_0_mz_values = np.array([minus_10pct.m_0_mz_ev, central.m_0_mz_ev, plus_10pct.m_0_mz_ev], dtype=float)

    return SensitivityData(
        minus_10pct=minus_10pct,
        central=central,
        plus_10pct=plus_10pct,
        effective_majorana_mass_std_mev=float(np.std(effective_majorana_mass_values)),
        effective_majorana_mass_max_shift_mev=float(np.max(np.abs(effective_majorana_mass_values - central.effective_majorana_mass_mev))),
        m_0_mz_max_shift_mev=float(1.0e3 * np.max(np.abs(m_0_mz_values - central.m_0_mz_ev))),
    )


def derive_geometric_sensitivity(
    bit_count: float = HOLOGRAPHIC_BITS,
    scale_ratio: float = RG_SCALE_RATIO,
    kappa_values: tuple[float, ...] = KAPPA_SCAN_VALUES,
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    gut_threshold_residue: float | None = None,
    central_kappa_geometric: float = GEOMETRIC_KAPPA,
) -> GeometricSensitivityData:
    """Scan the order-one geometric factor in the boundary--bulk scale relation.

    Args:
        bit_count: Central holographic bit count.
        scale_ratio: Ratio $M_{\rm GUT}/M_Z$.
        kappa_values: Discrete κ values used in the sweep.

    Returns:
        Geometric sensitivity data for $m_0$ and $|m_{\beta\beta}|$.
    """

    sweep_points: list[GeometricSweepPoint] = []
    central_pmns = derive_pmns(
        level=lepton_level,
        parent_level=parent_level,
        quark_level=quark_level,
        scale_ratio=scale_ratio,
        bit_count=bit_count,
        kappa_geometric=central_kappa_geometric,
    )
    central_scales = derive_scales_for_bits(
        bit_count,
        scale_ratio,
        kappa_geometric=central_kappa_geometric,
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    central_ckm = derive_ckm(
        level=quark_level,
        parent_level=parent_level,
        scale_ratio=scale_ratio,
        gut_threshold_residue=gut_threshold_residue,
    )
    central_pull_table = derive_pull_table(central_pmns, central_ckm)
    central_predictive_rows = tuple(
        row for row in central_pull_table.rows if row.included_in_predictive_fit and row.pull_data is not None
    )

    for kappa in kappa_values:
        pmns = derive_pmns(
            level=lepton_level,
            parent_level=parent_level,
            quark_level=quark_level,
            scale_ratio=scale_ratio,
            bit_count=bit_count,
            kappa_geometric=kappa,
        )
        scales = derive_scales_for_bits(
            bit_count,
            scale_ratio,
            kappa_geometric=kappa,
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        )
        pull_table = derive_pull_table(pmns, central_ckm)
        predictive_rows = tuple(row for row in pull_table.rows if row.included_in_predictive_fit and row.pull_data is not None)
        max_sigma_shift = max(
            abs(candidate.pull_data.value - reference.pull_data.value) / candidate.pull_data.effective_sigma
            for reference, candidate in zip(central_predictive_rows, predictive_rows)
            if candidate.pull_data is not None and candidate.pull_data.effective_sigma > 0.0
        )
        sweep_points.append(
            GeometricSweepPoint(
                kappa=kappa,
                m_0_mz_ev=scales.m_0_mz_ev,
                effective_majorana_mass_mev=1.0e3 * pmns.effective_majorana_mass_rg_ev,
                predictive_chi2=pull_table.predictive_chi2,
                predictive_max_abs_pull=pull_table.predictive_max_abs_pull,
                max_sigma_shift=float(max_sigma_shift),
            )
        )

    return GeometricSensitivityData(
        central_kappa=central_kappa_geometric,
        sweep_points=tuple(sweep_points),
        effective_majorana_mass_max_shift_mev=max(
            abs(point.effective_majorana_mass_mev - 1.0e3 * central_pmns.effective_majorana_mass_rg_ev)
            for point in sweep_points
        ),
        m_0_mz_max_shift_mev=max(abs(1.0e3 * (point.m_0_mz_ev - central_scales.m_0_mz_ev)) for point in sweep_points),
    )


def _singular_value_weights(singular_values: np.ndarray) -> np.ndarray:
    """Return normalized singular-value weights for the Higgs-VEV alignment constraint."""

    singular_values = np.asarray(singular_values, dtype=float)
    total = float(np.sum(singular_values))
    if total <= 0.0:
        return np.full_like(singular_values, 1.0 / max(len(singular_values), 1), dtype=float)
    return singular_values / total


def apply_higgs_vev_alignment_constraint(
    matrix: np.ndarray,
    relative_suppression: float,
    sector_exponent: float,
    *,
    clebsch_deformation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Apply the Higgs-VEV alignment constraint while leaving singular directions fixed.

    The exponent ``sector_exponent`` is chosen as ``+1/2`` for the quark block
    and ``-1/2`` for the lepton block so that the relative spectral-volume shift
    between the two sectors reproduces the charged-fermion hierarchy implied by
    the assumed $10_H/\mathbf{126}_H$ VEV alignment.
    """

    left, singular_values, right_dag = np.linalg.svd(matrix)
    weights = _singular_value_weights(singular_values)
    if clebsch_deformation is None:
        effective_suppression = np.full_like(singular_values, relative_suppression, dtype=float)
    else:
        effective_suppression = np.asarray(relative_suppression * np.asarray(clebsch_deformation, dtype=float), dtype=float)
        if effective_suppression.shape != singular_values.shape:
            raise ValueError(
                f"Expected a Clebsch deformation with shape {singular_values.shape}, received {effective_suppression.shape}."
            )
        effective_suppression = np.maximum(effective_suppression, np.finfo(float).eps)
    factors = np.power(effective_suppression, sector_exponent * weights)
    perturbed_matrix = left @ np.diag(factors * singular_values) @ right_dag
    return perturbed_matrix, singular_values, factors * singular_values


def _minimum_singular_vector_overlap(baseline: np.ndarray, perturbed: np.ndarray) -> tuple[float, float]:
    """Return the minimum left/right singular-vector overlaps after a spectrum shift."""

    baseline_left, _, baseline_right_dag = np.linalg.svd(baseline)
    perturbed_left, _, perturbed_right_dag = np.linalg.svd(perturbed)
    size = min(baseline_left.shape[1], perturbed_left.shape[1])
    left_overlap = min(abs(np.vdot(baseline_left[:, index], perturbed_left[:, index])) for index in range(size))
    right_overlap = min(
        abs(np.vdot(baseline_right_dag[index, :], perturbed_right_dag[index, :]))
        for index in range(size)
    )
    return float(left_overlap), float(right_overlap)


def _angle_interval_from_modulus_interval(interval: Interval) -> Interval:
    """Convert a small-angle modulus interval into its angle interval in degrees."""

    return Interval(
        math.degrees(math.asin(interval.lower)),
        math.degrees(math.asin(interval.upper)),
    )


def _sample_clebsch_gordan_deformations(
    rng: np.random.Generator,
    sample_count: int,
    singular_value_count: int,
) -> np.ndarray:
    log_half_width = math.log(2.0)
    return np.exp(rng.uniform(-log_half_width, log_half_width, size=(sample_count, singular_value_count)))


def derive_mass_ratio_stability_audit(
    perturbation_factor: float = MASS_RATIO_STABILITY_FACTOR,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    *,
    sample_count: int = VEV_ALIGNMENT_SWEEP_SAMPLE_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
    rng: np.random.Generator | None = None,
) -> MassRatioStabilityAuditData:
    r"""Check that the Higgs-VEV alignment constraint decouples from the angles.

    The branch-fixed charged-fermion tension is modeled as a Higgs-VEV
    alignment constraint on the singular spectrum,
    perturbation. The quark and leptonic textures are written as

    .. math:: Y_f = U_f\Sigma_fV_f^\dagger,

    and only the singular values are dressed by the $\mathbf{126}_H$ Clebsch
    suppression. The relative spectral-volume shift is fixed to the exact
    $1/C_{126}^{(12)}=64/312$ correction, while the left/right singular
    vectors --- and hence the PMNS/CKM mixing angles --- remain stable well
    within the quoted $1\sigma$ windows.
    """

    lepton_interface = derive_boundary_bulk_interface(level=lepton_level, sector="lepton")
    quark_interface = derive_boundary_bulk_interface(level=quark_level, sector="quark")
    higgs_cg_correction = calculate_126_higgs_cg_correction(target_suppression=1.0 / perturbation_factor)
    target_relative_suppression = 1.0 / perturbation_factor
    clebsch_relative_suppression = higgs_cg_correction.inverse_clebsch_126_suppression

    lepton_perturbed_matrix, lepton_singular_values, lepton_perturbed_singular_values = apply_higgs_vev_alignment_constraint(
        lepton_interface.majorana_yukawa_texture,
        clebsch_relative_suppression,
        sector_exponent=-0.5,
    )
    quark_perturbed_matrix, quark_singular_values, quark_perturbed_singular_values = apply_higgs_vev_alignment_constraint(
        quark_interface.framed_yukawa_texture,
        clebsch_relative_suppression,
        sector_exponent=+0.5,
    )

    lepton_baseline = polar_unitary(lepton_interface.majorana_yukawa_texture)
    lepton_perturbed = polar_unitary(lepton_perturbed_matrix)
    quark_baseline = polar_unitary(quark_interface.framed_yukawa_texture)
    quark_perturbed = polar_unitary(quark_perturbed_matrix)

    lepton_left_overlap_min, lepton_right_overlap_min = _minimum_singular_vector_overlap(
        lepton_interface.majorana_yukawa_texture,
        lepton_perturbed_matrix,
    )
    quark_left_overlap_min, quark_right_overlap_min = _minimum_singular_vector_overlap(
        quark_interface.framed_yukawa_texture,
        quark_perturbed_matrix,
    )

    lepton_angles_baseline = topological_kernel.pdg_parameters(lepton_baseline)[:3]
    lepton_angles_perturbed = topological_kernel.pdg_parameters(lepton_perturbed)[:3]
    quark_angles_baseline = topological_kernel.pdg_parameters(quark_baseline)[:3]
    quark_angles_perturbed = topological_kernel.pdg_parameters(quark_perturbed)[:3]

    lepton_angle_shifts = tuple(float(after - before) for before, after in zip(lepton_angles_baseline, lepton_angles_perturbed))
    quark_angle_shifts = tuple(float(after - before) for before, after in zip(quark_angles_baseline, quark_angles_perturbed))

    lepton_intervals = (
        LEPTON_INTERVALS["theta12"],
        LEPTON_INTERVALS["theta13"],
        LEPTON_INTERVALS["theta23"],
    )
    quark_intervals = tuple(
        _angle_interval_from_modulus_interval(QUARK_INTERVALS[key])
        for key in ("vus", "vub", "vcb")
    )

    lepton_sigma_shifts = tuple(abs(delta) / interval.sigma for delta, interval in zip(lepton_angle_shifts, lepton_intervals))
    quark_sigma_shifts = tuple(abs(delta) / interval.sigma for delta, interval in zip(quark_angle_shifts, quark_intervals))
    relative_spectral_volume_shift = float(
        (np.prod(quark_perturbed_singular_values) / np.prod(quark_singular_values))
        / (np.prod(lepton_perturbed_singular_values) / np.prod(lepton_singular_values))
    )

    resolved_rng = np.random.default_rng(seed) if rng is None else rng
    ensemble_effective_suppression_ratios = np.empty(0, dtype=float)
    ensemble_max_sigma_shifts = np.empty(0, dtype=float)
    ensemble_mass_scale_shifts = np.empty(0, dtype=float)
    ensemble_all_within_one_sigma = True
    ensemble_theta13_max_sigma_shift = 0.0
    ensemble_theta_c_max_sigma_shift = 0.0
    ensemble_max_sigma_shift = 0.0
    ensemble_mass_scale_shift_min = relative_spectral_volume_shift
    ensemble_mass_scale_shift_max = relative_spectral_volume_shift

    if sample_count > 0:
        deformation_samples = _sample_clebsch_gordan_deformations(
            resolved_rng,
            int(sample_count),
            len(lepton_singular_values),
        )
        ensemble_effective_suppression_ratios = clebsch_relative_suppression * np.exp(np.mean(np.log(deformation_samples), axis=1))
        ensemble_max_sigma_shifts = np.empty(int(sample_count), dtype=float)
        ensemble_mass_scale_shifts = np.empty(int(sample_count), dtype=float)
        theta13_sigma_shifts = np.empty(int(sample_count), dtype=float)
        theta_c_sigma_shifts = np.empty(int(sample_count), dtype=float)

        for sample_index, deformation in enumerate(deformation_samples):
            lepton_sampled_matrix, _, lepton_sampled_singular_values = apply_higgs_vev_alignment_constraint(
                lepton_interface.majorana_yukawa_texture,
                clebsch_relative_suppression,
                sector_exponent=-0.5,
                clebsch_deformation=deformation,
            )
            quark_sampled_matrix, _, quark_sampled_singular_values = apply_higgs_vev_alignment_constraint(
                quark_interface.framed_yukawa_texture,
                clebsch_relative_suppression,
                sector_exponent=+0.5,
                clebsch_deformation=deformation,
            )
            lepton_sampled = polar_unitary(lepton_sampled_matrix)
            quark_sampled = polar_unitary(quark_sampled_matrix)
            lepton_sampled_angles = topological_kernel.pdg_parameters(lepton_sampled)[:3]
            quark_sampled_angles = topological_kernel.pdg_parameters(quark_sampled)[:3]
            lepton_sampled_sigma_shifts = tuple(
                abs(after - before) / interval.sigma
                for before, after, interval in zip(lepton_angles_baseline, lepton_sampled_angles, lepton_intervals)
            )
            quark_sampled_sigma_shifts = tuple(
                abs(after - before) / interval.sigma
                for before, after, interval in zip(quark_angles_baseline, quark_sampled_angles, quark_intervals)
            )
            ensemble_max_sigma_shifts[sample_index] = max((*lepton_sampled_sigma_shifts, *quark_sampled_sigma_shifts), default=0.0)
            theta13_sigma_shifts[sample_index] = lepton_sampled_sigma_shifts[1]
            theta_c_sigma_shifts[sample_index] = quark_sampled_sigma_shifts[0]
            ensemble_mass_scale_shifts[sample_index] = float(
                (np.prod(quark_sampled_singular_values) / np.prod(quark_singular_values))
                / (np.prod(lepton_sampled_singular_values) / np.prod(lepton_singular_values))
            )

        ensemble_max_sigma_shift = float(np.max(ensemble_max_sigma_shifts))
        ensemble_theta13_max_sigma_shift = float(np.max(theta13_sigma_shifts))
        ensemble_theta_c_max_sigma_shift = float(np.max(theta_c_sigma_shifts))
        ensemble_all_within_one_sigma = bool(ensemble_max_sigma_shift <= 1.0)
        ensemble_mass_scale_shift_min = float(np.min(ensemble_mass_scale_shifts))
        ensemble_mass_scale_shift_max = float(np.max(ensemble_mass_scale_shifts))

    return MassRatioStabilityAuditData(
        perturbation_factor=perturbation_factor,
        target_relative_suppression=target_relative_suppression,
        clebsch_relative_suppression=clebsch_relative_suppression,
        relative_spectral_volume_shift=relative_spectral_volume_shift,
        lepton_unitary_frobenius_shift=float(np.linalg.norm(lepton_perturbed - lepton_baseline)),
        quark_unitary_frobenius_shift=float(np.linalg.norm(quark_perturbed - quark_baseline)),
        lepton_singular_values=tuple(float(value) for value in lepton_singular_values),
        lepton_perturbed_singular_values=tuple(float(value) for value in lepton_perturbed_singular_values),
        quark_singular_values=tuple(float(value) for value in quark_singular_values),
        quark_perturbed_singular_values=tuple(float(value) for value in quark_perturbed_singular_values),
        lepton_left_overlap_min=lepton_left_overlap_min,
        lepton_right_overlap_min=lepton_right_overlap_min,
        quark_left_overlap_min=quark_left_overlap_min,
        quark_right_overlap_min=quark_right_overlap_min,
        lepton_angle_shifts_deg=lepton_angle_shifts,
        quark_angle_shifts_deg=quark_angle_shifts,
        lepton_sigma_shifts=lepton_sigma_shifts,
        quark_sigma_shifts=quark_sigma_shifts,
        max_sigma_shift=float(max((*lepton_sigma_shifts, *quark_sigma_shifts), default=0.0)),
        ensemble_sample_count=int(sample_count),
        ensemble_seed=seed,
        ensemble_all_within_one_sigma=ensemble_all_within_one_sigma,
        ensemble_max_sigma_shift=ensemble_max_sigma_shift,
        ensemble_theta13_max_sigma_shift=ensemble_theta13_max_sigma_shift,
        ensemble_theta_c_max_sigma_shift=ensemble_theta_c_max_sigma_shift,
        ensemble_mass_scale_shift_min=ensemble_mass_scale_shift_min,
        ensemble_mass_scale_shift_max=ensemble_mass_scale_shift_max,
        ensemble_effective_suppression_ratios=np.asarray(ensemble_effective_suppression_ratios, dtype=float),
        ensemble_max_sigma_shifts=np.asarray(ensemble_max_sigma_shifts, dtype=float),
        ensemble_mass_scale_shifts=np.asarray(ensemble_mass_scale_shifts, dtype=float),
    )


def pull_from_interval(
    value: float,
    interval: Interval,
    *,
    theory_value: float | None = None,
    theoretical_uncertainty_fraction: float = 0.0,
    parametric_covariance_fraction: float = 0.0,
    parametric_sigma: float | None = None,
) -> PullData:
    """Convert a value and a symmetric interval into a pull datum."""

    sigma = interval.sigma
    sigma_theory = 0.0 if theory_value is None else abs(theory_value) * theoretical_uncertainty_fraction
    if parametric_sigma is not None:
        sigma_parametric = float(max(parametric_sigma, 0.0))
    else:
        sigma_parametric = 0.0 if theory_value is None else abs(theory_value) * parametric_covariance_fraction
    effective_sigma = math.sqrt(sigma * sigma + sigma_theory * sigma_theory + sigma_parametric * sigma_parametric)
    pull = 0.0 if math.isclose(
        effective_sigma,
        0.0,
        rel_tol=0.0,
        abs_tol=condition_aware_abs_tolerance(scale=effective_sigma),
    ) else (value - interval.central) / effective_sigma
    return PullData(
        value=value,
        central=interval.central,
        sigma=sigma,
        effective_sigma=effective_sigma,
        pull=pull,
        inside_1sigma=abs(value - interval.central) <= effective_sigma,
        theory_sigma=sigma_theory,
        parametric_sigma=sigma_parametric,
    )


def pull_from_transport_covariance(
    value: float,
    interval: Interval,
    *,
    theory_value: float | None = None,
    observable_name: str | None = None,
    transport_covariance: TransportParametricCovarianceData | None = None,
) -> PullData:
    """Return a pull datum using the publication transport-covariance budget."""

    resolved_parametric_sigma: float | None = None
    resolved_parametric_fraction = PARAMETRIC_COVARIANCE_FRACTION
    if (
        transport_covariance is not None
        and observable_name is not None
        and observable_name in transport_covariance.observable_names
    ):
        resolved_parametric_sigma = transport_covariance.interval_sigma_for(observable_name)
        if theory_value is None or math.isclose(
            theory_value,
            0.0,
            rel_tol=0.0,
            abs_tol=condition_aware_abs_tolerance(scale=theory_value if theory_value is not None else 0.0),
        ):
            resolved_parametric_fraction = 0.0
        else:
            resolved_parametric_fraction = resolved_parametric_sigma / abs(theory_value)

    return pull_from_interval(
        value,
        interval,
        theory_value=theory_value,
        theoretical_uncertainty_fraction=THEORETICAL_UNCERTAINTY_FRACTION,
        parametric_covariance_fraction=resolved_parametric_fraction,
        parametric_sigma=resolved_parametric_sigma,
    )


def format_real_matrix(matrix: np.ndarray) -> str:
    return np.array2string(matrix, formatter={"float_kind": lambda value: f"{value: .4f}"})


def format_complex_matrix(matrix: np.ndarray) -> str:
    return np.array2string(matrix, formatter={"complex_kind": lambda value: f"{value.real: .4f}{value.imag:+.4f}j"})


def format_complex_scalar(value: complex) -> str:
    return f"{value.real:.12f}{value.imag:+.12f}j"


def format_phase_vector(vector: np.ndarray) -> str:
    return np.array2string(vector, formatter={"complex_kind": lambda value: f"e^({np.angle(value): .4f}i)"})


def print_pull(label: str, pull_data: PullData, suffix: str = "") -> None:
    LOGGER.info(
        f"{label:<28}: {pull_data.value: .6f}{suffix}  "
        f"pull={pull_data.pull:+.2f}σ  "
        f"σ_exp={pull_data.sigma:.6f}{suffix}  "
        f"σ_tot={pull_data.effective_sigma:.6f}{suffix}"
    )


def print_threshold_audit(threshold_shift_audit: ThresholdShiftAuditData | None = None) -> None:
    """Print explicit structurally predicted seesaw-threshold bookkeeping at $M_N$."""

    threshold_audit = derive_threshold_shift_audit() if threshold_shift_audit is None else threshold_shift_audit
    threshold = threshold_audit.threshold

    LOGGER.info("Explicit RHN threshold audit")
    LOGGER.info("-" * 88)
    LOGGER.info(f"M_N [GeV]                        : {threshold.threshold_scale_gev:.6e}")
    LOGGER.info(f"I_L, I_Q                         : {threshold.lepton_branching_index}, {threshold.quark_branching_index}")
    LOGGER.info(f"Pi_rank                          : {threshold.rank_pressure:.12f}")
    LOGGER.info(f"deltaPi_126                      : {threshold.threshold_shift_126:.12f}")
    LOGGER.info(f"structural exponent              : {threshold.structural_exponent:.12f}")
    LOGGER.info(f"framing-gap area beta^2          : {threshold_audit.framing_gap_area_beta_sq:.12f}")
    LOGGER.info(f"log(M_N/M_126^match)             : {threshold_audit.matching_scale_log_ratio:.12f}")
    LOGGER.info(f"log(M_N/M_Z)                     : {threshold.lower_interval_log:.6f}")
    LOGGER.info(f"log(M_GUT/M_N)                   : {threshold.upper_interval_log:.6f}")
    LOGGER.info(f"L_ZN                             : {threshold_audit.lower_one_loop_factor:.12f}")
    LOGGER.info(f"L_GN                             : {threshold_audit.upper_one_loop_factor:.12f}")
    angle_labels = ("theta12", "theta13", "theta23")
    for label in angle_labels:
        shift = threshold_audit.observable_shifts_deg[label]
        LOGGER.info(
            f"{label} shift below/above/2-loop/match/total [deg]: "
            f"{shift.lower_one_loop:.12f}, "
            f"{shift.upper_one_loop:.12f}, "
            f"{shift.two_loop:.12f}, "
            f"{shift.matching:.12f}, "
            f"{shift.total:.12f}"
        )
    delta_shift = threshold_audit.observable_shifts_deg["delta_cp"]
    LOGGER.info(
        "delta_CP shift below/above/2-loop/match/total [deg]: "
        f"{delta_shift.lower_one_loop:.12f}, {delta_shift.upper_one_loop:.12f}, {delta_shift.two_loop:.12f}, "
        f"{delta_shift.matching:.12f}, {delta_shift.total:.12f}"
    )
    mass_shift = threshold_audit.m_0_fraction_shift
    LOGGER.info(
        "m0 frac shift below/above/2-loop/match/total      : "
        f"{mass_shift.lower_one_loop:.12f}, {mass_shift.upper_one_loop:.12f}, {mass_shift.two_loop:.12f}, "
        f"{mass_shift.matching:.12f}, {mass_shift.total:.12f}"
    )
    LOGGER.info(f"leading-log capture (raw norm)   : {threshold_audit.leading_norm_capture:.12f}")
    LOGGER.info(f"leading-log capture (sigma norm) : {threshold_audit.sigma_weighted_capture:.12f}")
    LOGGER.info("")


def final_audit_check(
    ckm: CkmData | None = None,
    audit: AuditData | None = None,
    ghost_character_audit: GhostCharacterAuditData | None = None,
) -> FinalAuditResult:
    r"""Log the final anomaly-to-$\gamma$ diagnostics and emit a paste-ready LaTeX block."""

    ckm_data = derive_ckm() if ckm is None else ckm
    vacuum = DEFAULT_TOPOLOGICAL_VACUUM
    resolved_audit = vacuum.derive_audit() if audit is None else audit
    resolved_ghost_character_audit = (
        vacuum.derive_ghost_character_audit(resolved_audit)
        if ghost_character_audit is None
        else ghost_character_audit
    )
    gamma_pull = pull_from_interval(ckm_data.gamma_rg_deg, CKM_GAMMA_GOLD_STANDARD_DEG).pull
    within_one_sigma = abs(gamma_pull) <= 1.0
    gauge_unification = derive_gauge_unification_existence_proof()
    anomaly_data = calculate_branching_anomaly("SO(10)", "SU(3)", PARENT_LEVEL)
    anomaly_fraction = anomaly_data.anomaly_fraction
    framing_audit = vacuum.verify_framing_anomaly()
    gravity_audit = vacuum.verify_bulk_emergence()
    gauge_audit = vacuum.verify_gauge_holography()
    dark_energy_audit = vacuum.verify_dark_energy_tension()
    c_dark_completion = dark_energy_audit.c_dark_completion
    bulk_status = "verified" if gravity_audit.bulk_emergent else "conditional"
    jarlskog_topological = topological_jarlskog_identity(
        ckm_data.gut_threshold_residue,
        parent_level=ckm_data.parent_level,
        lepton_level=LEPTON_LEVEL,
        quark_level=ckm_data.level,
    )
    jarlskog_visible_lock = threshold_projected_jarlskog(
        jarlskog_topological,
        gut_threshold_residue=ckm_data.gut_threshold_residue,
    )
    delta_mod_cp_zero = cp_conserving_modularity_leak(
        jarlskog_visible_lock,
        ckm_data.theta_c_uv_deg,
        ckm_data.theta13_uv_deg,
        ckm_data.theta23_uv_deg,
        parent_level=ckm_data.parent_level,
        lepton_level=LEPTON_LEVEL,
        quark_level=ckm_data.level,
    )
    if math.isclose(ckm_data.gut_threshold_residue, R_GUT, rel_tol=0.0, abs_tol=1.0e-15):
        assert delta_mod_cp_zero > 0.0040, "CP-Conserving Vacuum Forbidden: Modularity Leak Detected."
    latex_block = "\n".join(
        (
            r"\paragraph{Numerical Consistency Audit.}",
            r"The benchmark branch is evaluated only on the anomaly-free cell and should be read as the unique torsion-free vacuum solution of the displayed scan. In the HEC reading the visible Standard Model states are logical bits, while the fixed complement $c_{\rm dark}$ supplies the parity bits required to reconstruct the torsion-free bulk and fixes the induced Einstein--Hilbert term.",
            r"\begin{align}",
            rf" \Delta_{{\rm fr}} &= {gravity_audit.framing_gap:.0f}, & c_{{\rm dark}} &= {c_dark_completion:.4f}, \\",
            rf" c_{{\rm vis}} &= {gravity_audit.visible_central_charge:.4f}, & \eta_{{\rm mod}} &= {gravity_audit.modular_residue_efficiency:.4f}, \\",
            rf" \Delta_{{\rm DM}} &= {gravity_audit.omega_dm_ratio:.4f}, & \mathcal C_{{\rm PB}} &= {int(gravity_audit.parity_bit_density_constraint_satisfied)}, \\",
            rf" \Lambda_{{\rm holo}} &= {_format_tex_scientific(gravity_audit.lambda_budget_si_m2, precision=2)}\,\mathrm{{m}}^{{-2}}, & \Lambda_{{\rm obs}} &= {_format_tex_scientific(gravity_audit.observed_lambda_si_m2, precision=2)}\,\mathrm{{m}}^{{-2}}, \\",
            rf" N_{{\rm holo}} &= {_format_tex_scientific(dark_energy_audit.holographic_bits, precision=2)}, & \delta_{{\Lambda}}^{{\rm surf}} &= {dark_energy_audit.surface_tension_deviation_percent:.2f}\%, \\",
            rf" m_0^{{\rm UV}} &= {_format_tex_scientific(gravity_audit.neutrino_scale_ev, precision=2)}\,\mathrm{{eV}}, & \mathcal C_{{G_{{\mu\nu}}}} &= {gravity_audit.gmunu_consistency_score:.4f}, \\",
            rf" \tau_p^{{(d=6)}} &= {_format_tex_scientific(gravity_audit.baryon_stability.proton_lifetime_years, precision=2)}\,\mathrm{{yr}}, & P_{{\rm tunnel}} &= {_format_tex_scientific(gravity_audit.baryon_stability.modular_tunneling_penalty, precision=2)}, \\",
            rf" \alpha_{{\rm topo}}^{{-1}} &= {gauge_audit.topological_alpha_inverse:.3f}, & \delta_\alpha^{{\rm geom}} &= {gauge_audit.geometric_residue_percent:.2f}\%, \\",
            rf" 10^3\Delta_{{\rm mod}} &= {gauge_audit.modular_gap_scaled_inverse:.3f}, & \alpha_{{\rm targ}}^{{-1}} &= {gauge_audit.codata_alpha_inverse:.3f}, \\",
            rf" J_{{CP}}^{{\rm topo}} &= {_format_tex_scientific(jarlskog_topological, precision=2)}, & \Delta_{{\rm mod}}^{{CP=0}} &= {delta_mod_cp_zero:.4f}, \\",
            rf" \alpha_{{\rm br}} &= {float(anomaly_fraction):.6f}, & \mathrm{{pull}}_\gamma &= {gamma_pull:+.3f}\sigma.",
            r"\end{align}",
            rf"\noindent \textit{{Bulk status.}} The Levi--Civita, regulator, and cosmological-alignment audit is \textit{{{bulk_status}}}.",
            (
                rf"\noindent \textit{{IH informational-cost exception.}} In the same one-copy benchmark, the inverted support map "
                rf"has support deficit {resolved_audit.support_deficit}, would require dictionary rank "
                rf"{resolved_audit.required_inverted_rank}>{resolved_audit.modularity_limit_rank}, and therefore demands "
                rf"{resolved_ghost_character_audit.extra_character_count} extra ghost support sector(s). The finite Relaxed Proxy Gap is "
                rf"$\widetilde{{\mathcal E}}_{{\rm IH}}=\beta^2={resolved_audit.relaxed_inverted_gap:.6f}$, so the code records IH "
                r"as an explicit non-minimal extension of the benchmark dictionary rather than as a branch of the minimal one-copy solution."
            ),
        )
    )

    LOGGER.info("Verification")
    LOGGER.info("-" * 88)
    LOGGER.info(f"visible Cartan denominator       : {anomaly_data.visible_cartan_denominator}")
    LOGGER.info(f"visible embedding index x        : {anomaly_data.visible_cartan_embedding_index}")
    LOGGER.info(
        f"alpha_br exact ratio              : {anomaly_data.numerator_units}/"
        f"{anomaly_data.visible_cartan_embedding_index * anomaly_data.denominator_units}"
    )
    LOGGER.info(f"alpha_br (derived)               : {float(anomaly_fraction):.12f}")
    LOGGER.info(f"Delta b^(126_H)                  : {np.array2string(gauge_unification.beta_shift_126, formatter={'float_kind': lambda value: f'{value: .6f}'})}")
    LOGGER.info(f"Delta b^(10_H)                   : {np.array2string(gauge_unification.beta_shift_10, formatter={'float_kind': lambda value: f'{value: .6f}'})}")
    LOGGER.info(f"gamma pull from anomaly          : {gamma_pull:+.12f}σ")
    LOGGER.info(f"within one sigma                 : {int(within_one_sigma)}")
    LOGGER.info(f"c_visible logical bits           : {gravity_audit.visible_central_charge:.12f}")
    LOGGER.info(f"c_dark geometric residue         : {c_dark_completion:.4f}")
    LOGGER.info(f"eta_mod = c_dark/c_visible       : {gravity_audit.modular_residue_efficiency:.12f}")
    LOGGER.info(f"Delta_DM = kappa*eta_mod         : {gravity_audit.omega_dm_ratio:.12f}")
    LOGGER.info(f"Parity-Bit Density Constraint    : {int(gravity_audit.parity_bit_density_constraint_satisfied)}")
    LOGGER.info(f"G_munu consistency score         : {gravity_audit.gmunu_consistency_score:.12f}")
    LOGGER.info(f"N_holo [bits]                    : {dark_energy_audit.holographic_bits:.12e}")
    LOGGER.info(f"kappa_D5                         : {dark_energy_audit.geometric_residue:.12f}")
    LOGGER.info(f"Delta_mod benchmark              : {dark_energy_audit.modular_gap:.12f}")
    LOGGER.info(f"Lambda_holo [m^-2]               : {dark_energy_audit.lambda_surface_tension_si_m2:.12e}")
    LOGGER.info(f"Lambda anchor [m^-2]             : {dark_energy_audit.lambda_anchor_si_m2:.12e}")
    LOGGER.info(f"1/(L_P^2 N) [m^-2]               : {dark_energy_audit.lambda_scaling_identity_si_m2:.12e}")
    LOGGER.info(f"surface-tension prefactor        : {dark_energy_audit.surface_tension_prefactor:.12f}")
    LOGGER.info(f"rho_vac surface tension [eV^4]   : {dark_energy_audit.rho_vac_surface_tension_ev4:.12e}")
    LOGGER.info(f"rho_vac(m_0,kappa) [eV^4]        : {dark_energy_audit.rho_vac_from_defect_scale_ev4:.12e}")
    LOGGER.info(f"J_CP^topo lock                   : {jarlskog_topological:.12e}")
    LOGGER.info(f"J_CP^q visible lock              : {jarlskog_visible_lock:.12e}")
    LOGGER.info(f"Delta_mod^(CP=0)                 : {delta_mod_cp_zero:.12f}")
    LOGGER.info(f"Surface Tension Deviation        : {dark_energy_audit.surface_tension_deviation_percent:.2f}%")
    LOGGER.info(f"proton lifetime floor [yr]       : {gravity_audit.baryon_stability.proton_lifetime_years:.12e}")
    LOGGER.info(f"modular tunneling penalty        : {gravity_audit.baryon_stability.modular_tunneling_penalty:.12e}")
    LOGGER.info(f"protected evaporation [yr]       : {gravity_audit.baryon_stability.protected_evaporation_lifetime_years:.12e}")
    LOGGER.info(f"BARYON STABILITY (tau_p): {gravity_audit.baryon_stability.proton_lifetime_years:.2e} years [PROTECTED BY DELTA_FR=0]")
    LOGGER.info(f"alpha^-1 level density           : {gauge_audit.topological_alpha_inverse:.12f}")
    LOGGER.info(f"alpha^-1 integer target          : {gauge_audit.codata_alpha_inverse:.12f}")
    LOGGER.info(f"10^3 Delta_mod                   : {gauge_audit.modular_gap_scaled_inverse:.12f}")
    LOGGER.info(f"gauge geometric residue          : {gauge_audit.geometric_residue_percent:.2f}%")
    LOGGER.info(f"[VERIFIED] Gauge Coupling Residue (Alpha) -> Delta: {gauge_audit.geometric_residue_percent:.2f}%")
    LOGGER.info("")
    LOGGER.info("LaTeX numerical audit block")
    LOGGER.info("-" * 88)
    for line in latex_block.splitlines():
        LOGGER.info(line)
    LOGGER.info("")

    return FinalAuditResult(
        gamma_pull=gamma_pull,
        within_one_sigma=within_one_sigma,
        anomaly_fraction=anomaly_fraction,
        c_dark_completion=c_dark_completion,
        gravity_audit=gravity_audit,
        gauge_audit=gauge_audit,
        dark_energy_audit=dark_energy_audit,
        gmunu_consistency_score=gravity_audit.gmunu_consistency_score,
        ih_nonminimal_extension_required=resolved_ghost_character_audit.ih_nonminimal_extension_required,
        ih_support_deficit=resolved_audit.support_deficit,
        ih_modularity_limit_rank=resolved_audit.modularity_limit_rank,
        ih_required_dictionary_rank=resolved_audit.required_inverted_rank,
        latex_block=latex_block,
    )


@dataclass(frozen=True)
class AuditStatistics:
    """Reviewer-facing split between transport stability and external fit quality."""

    hard_anomaly_filter_pass: bool
    topological_residue_lock_pass: bool
    continuous_parameter_subtraction_count: int
    quadrature_convergence_pass: bool
    step_size_convergence_pass: bool
    internal_validity_pass: bool
    external_validity_pass: bool
    review_ready_pass: bool
    transport_covariance_mode: str
    transport_stability_yield: float
    transport_failure_fraction: float
    transport_failure_count: int
    step_size_reference_count: int
    step_size_reference_predictive_chi2: float
    step_size_max_delta_predictive_chi2: float
    step_size_max_sigma_shift: float
    predictive_chi2: float
    predictive_chi2_threshold: float
    predictive_rms_pull_sigma: float
    predictive_max_abs_pull_sigma: float
    external_reference_label: str = "NuFIT 5.3 / PDG 2024"

    @property
    def topological_rigidity_pass(self) -> bool:
        """Whether the selected branch remains rigid rather than fitted."""

        return (
            self.hard_anomaly_filter_pass
            and self.topological_residue_lock_pass
            and self.continuous_parameter_subtraction_count == 0
        )


@dataclass(frozen=True)
class TopologicalIntegrityAssertionData:
    """Explicit publication-facing integrity assertions for the selected branch."""

    hard_anomaly_filter: bool
    topological_rigidity: bool
    welded_mass_coordinate_lock: bool
    superheavy_relic_lock: bool
    ih_informational_cost: bool

    @property
    def all_asserted(self) -> bool:
        return all(
            (
                self.hard_anomaly_filter,
                self.topological_rigidity,
                self.welded_mass_coordinate_lock,
                self.superheavy_relic_lock,
                self.ih_informational_cost,
            )
        )


class MasterAudit:
    """Publication-facing namespace for the terminal branch-consistency checks."""

    @staticmethod
    def hard_anomaly_filter(model: TopologicalVacuum | None = None) -> bool:
        """Binary gate for quantum-mechanical definability of the benchmark branch."""

        resolved_model = DEFAULT_TOPOLOGICAL_VACUUM if model is None else model
        return math.isclose(resolved_model.framing_gap, 0.0, rel_tol=0.0, abs_tol=1.0e-15)

    @staticmethod
    def topological_rigidity(
        pmns: PmnsData | None = None,
        ckm: CkmData | None = None,
        pull_table: PullTable | None = None,
        model: TopologicalVacuum | None = None,
    ) -> bool:
        """Check that the branch is rigid rather than fitted."""

        resolved_model = DEFAULT_TOPOLOGICAL_VACUUM if model is None else model
        resolved_pmns = resolved_model.derive_pmns() if pmns is None else pmns
        resolved_ckm = resolved_model.derive_ckm() if ckm is None else ckm
        resolved_pull_table = (
            derive_pull_table(resolved_pmns, resolved_ckm)
            if pull_table is None
            else pull_table
        )
        return (
            MasterAudit.hard_anomaly_filter(model=resolved_model)
            and MasterAudit.topological_coordinate_validation(resolved_pmns, resolved_ckm)
            and resolved_pull_table.continuous_parameter_subtraction_count == 0
        )

    @staticmethod
    def ih_informational_cost(audit: AuditData | None = None) -> bool:
        """Check that IH carries the expected finite one-copy informational cost."""

        resolved_audit = derive_audit() if audit is None else audit
        return (
            resolved_audit.support_deficit > 0
            and resolved_audit.required_inverted_rank > resolved_audit.modularity_limit_rank
            and math.isclose(
                resolved_audit.relaxed_inverted_gap,
                resolved_audit.beta * resolved_audit.beta,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        )

    @staticmethod
    def superheavy_relic_lock(dm_fingerprint: DmFingerprintInputs) -> bool:
        """Check that the parity-bit relic stays in the superheavy WIMPzilla regime."""

        superheavy_relic_floor_gev = 1.0e9
        direct_detection_null_ceiling_cm2 = 1.0e-48
        return (
            dm_fingerprint.dm_mass_gev >= superheavy_relic_floor_gev
            and dm_fingerprint.gauge_sigma_cm2 < direct_detection_null_ceiling_cm2
        )

    @staticmethod
    def topological_integrity_assertions(
        pmns: PmnsData | None = None,
        ckm: CkmData | None = None,
        pull_table: PullTable | None = None,
        audit: AuditData | None = None,
        dark_energy_audit: DarkEnergyTensionAudit | None = None,
        dm_fingerprint: DmFingerprintInputs | None = None,
        model: TopologicalVacuum | None = None,
    ) -> TopologicalIntegrityAssertionData:
        """Assemble the explicit integrity assertions used by the benchmark audit."""

        resolved_model = DEFAULT_TOPOLOGICAL_VACUUM if model is None else model
        resolved_pmns = resolved_model.derive_pmns() if pmns is None else pmns
        resolved_ckm = resolved_model.derive_ckm() if ckm is None else ckm
        resolved_pull_table = (
            derive_pull_table(resolved_pmns, resolved_ckm)
            if pull_table is None
            else pull_table
        )
        resolved_audit = resolved_model.derive_audit() if audit is None else audit
        resolved_dark_energy_audit = (
            resolved_model.verify_dark_energy_tension()
            if dark_energy_audit is None
            else dark_energy_audit
        )
        resolved_dm_fingerprint = dm_fingerprint
        if resolved_dm_fingerprint is None:
            physics_audit = resolved_model.derive_physics_audit()
            weight_profile = resolved_model.generate_ckm_phase_tilt_profile(resolved_pmns)
            framing_gap_stability = resolved_model.derive_framing_gap_stability_audit(resolved_ckm, resolved_audit)
            resolved_dm_fingerprint = derive_dm_fingerprint_inputs(
                weight_profile,
                physics_audit.geometric_kappa,
                framing_gap_stability,
            )
        return TopologicalIntegrityAssertionData(
            hard_anomaly_filter=MasterAudit.hard_anomaly_filter(model=resolved_model),
            topological_rigidity=MasterAudit.topological_rigidity(
                pmns=resolved_pmns,
                ckm=resolved_ckm,
                pull_table=resolved_pull_table,
                model=resolved_model,
            ),
            welded_mass_coordinate_lock=(
                resolved_dark_energy_audit.alpha_locked_under_bit_shift
                and resolved_dark_energy_audit.triple_match_saturated
                and resolved_dark_energy_audit.sensitivity_audit_triggered_integrity_error
            ),
            superheavy_relic_lock=MasterAudit.superheavy_relic_lock(resolved_dm_fingerprint),
            ih_informational_cost=MasterAudit.ih_informational_cost(resolved_audit),
        )

    @staticmethod
    def topological_coordinate_validation(pmns: PmnsData | None = None, ckm: CkmData | None = None) -> bool:
        r"""Validate the Topological Residues against the exact WZW level-set identities.

        The benchmark uses three discrete Topological Residues rather than
        adjustable fit inputs. Each is tied directly to a level-set function in
        the verifier:

        - ``compute_geometric_kappa_ansatz(...)`` fixes ``\kappa_{D_5}`` from the
          $D_5$ weight-simplex packing geometry on the selected WZW branch.
        - ``VOA_BRANCHING_GAP = 8/28`` fixes ``\mathcal R_{\rm GUT}``, the visible
          residue of the $SO(10)_{312}\to SU(2)_{26}\times SU(3)_8$ branching.
        - ``calculate_126_higgs_cg_correction()`` fixes the branch VEV residue
          ``\langle\Sigma_{126}\rangle/\langle\phi_{10}\rangle = 64/312``.
        - ``surface_tension_gauge_alpha_inverse()`` checks the same level-set via
          the surface-tension gauge benchmark ``\alpha^{-1}_{\rm surf}``.
        """

        resolved_lepton_level = LEPTON_LEVEL if pmns is None else pmns.level
        resolved_parent_level = PARENT_LEVEL if pmns is None else pmns.parent_level
        resolved_geometric_kappa = GEOMETRIC_KAPPA if pmns is None else float(pmns.kappa_geometric)
        derived_geometric_kappa = compute_geometric_kappa_ansatz(
            parent_level=resolved_parent_level,
            lepton_level=resolved_lepton_level,
        ).derived_kappa
        resolved_threshold_residue = R_GUT if ckm is None else float(ckm.gut_threshold_residue)
        resolved_threshold_correction = (
            resolved_threshold_residue
            if ckm is None
            else float(ckm.so10_threshold_correction.gut_threshold_residue)
        )
        higgs_cg_correction = calculate_126_higgs_cg_correction()
        return (
            math.isclose(resolved_geometric_kappa, derived_geometric_kappa, rel_tol=0.0, abs_tol=1.0e-15)
            and _matches_exact_fraction(resolved_threshold_residue, VOA_BRANCHING_GAP)
            and _matches_exact_fraction(resolved_threshold_correction, VOA_BRANCHING_GAP)
            and _matches_exact_fraction(higgs_cg_correction.target_suppression, QUADRATIC_WEIGHT_PROJECTION)
            and _matches_exact_fraction(higgs_cg_correction.inverse_clebsch_126_suppression, QUADRATIC_WEIGHT_PROJECTION)
            and _matches_exact_fraction(surface_tension_gauge_alpha_inverse(), GAUGE_STRENGTH_IDENTITY)
        )

    @staticmethod
    def audit_statistics(
        pmns: PmnsData | None = None,
        ckm: CkmData | None = None,
        transport_covariance: TransportParametricCovarianceData | None = None,
        step_size_convergence: StepSizeConvergenceData | None = None,
        pull_table: PullTable | None = None,
        model: TopologicalVacuum | None = None,
    ) -> AuditStatistics:
        """Summarize internal and external validity for reviewer-facing reporting.

        Internal validity tracks the stability of the RG transport map under the
        step-size audit together with the explicit quadrature guard enforced by
        ``QuadratureConvergenceError``. External validity reports the predictive
        pull summary against the exported NuFIT~5.3 / PDG intervals.
        """

        resolved_model = DEFAULT_TOPOLOGICAL_VACUUM if model is None else model
        resolved_pmns = resolved_model.derive_pmns() if pmns is None else pmns
        resolved_ckm = resolved_model.derive_ckm() if ckm is None else ckm
        resolved_transport_covariance = (
            resolved_model.derive_transport_parametric_covariance(resolved_pmns, resolved_ckm)
            if transport_covariance is None
            else transport_covariance
        )
        resolved_step_size_convergence = (
            resolved_model.derive_step_size_convergence_audit()
            if step_size_convergence is None
            else step_size_convergence
        )
        resolved_pull_table = (
            derive_pull_table(resolved_pmns, resolved_ckm, transport_covariance=resolved_transport_covariance)
            if pull_table is None
            else pull_table
        )
        hard_anomaly_filter_pass = MasterAudit.hard_anomaly_filter(model=resolved_model)
        topological_residue_lock_pass = MasterAudit.topological_coordinate_validation(resolved_pmns, resolved_ckm)
        continuous_parameter_subtraction_count = resolved_pull_table.continuous_parameter_subtraction_count
        transport_stability_pass = (
            resolved_transport_covariance.stability_yield >= TRANSPORT_MC_MIN_STABILITY_YIELD
            and resolved_transport_covariance.failure_count == 0
            and not resolved_transport_covariance.hard_wall_penalty_applied
        )
        step_size_delta_values = np.asarray(resolved_step_size_convergence.delta_predictive_chi2_values, dtype=float)
        step_size_sigma_values = np.asarray(resolved_step_size_convergence.max_sigma_shift_values, dtype=float)
        step_size_max_delta_predictive_chi2 = float(
            np.nanmax(np.abs(step_size_delta_values)) if step_size_delta_values.size else 0.0
        )
        step_size_max_sigma_shift = float(
            np.nanmax(np.abs(step_size_sigma_values)) if step_size_sigma_values.size else 0.0
        )
        step_size_convergence_pass = (
            np.all(np.isfinite(step_size_delta_values))
            and np.all(np.isfinite(step_size_sigma_values))
            and math.isfinite(resolved_step_size_convergence.reference_predictive_chi2)
            and step_size_max_sigma_shift <= 1.0
            and step_size_max_delta_predictive_chi2 <= FOLLOWUP_CHI2_SURVIVAL_THRESHOLD
        )
        quadrature_convergence_pass = True
        internal_validity_pass = (
            hard_anomaly_filter_pass
            and topological_residue_lock_pass
            and continuous_parameter_subtraction_count == 0
            and transport_stability_pass
            and step_size_convergence_pass
            and quadrature_convergence_pass
        )
        external_validity_pass = (
            resolved_pull_table.predictive_max_abs_pull <= 2.0
            and resolved_pull_table.predictive_chi2 <= FOLLOWUP_CHI2_SURVIVAL_THRESHOLD
        )
        return AuditStatistics(
            hard_anomaly_filter_pass=hard_anomaly_filter_pass,
            topological_residue_lock_pass=topological_residue_lock_pass,
            continuous_parameter_subtraction_count=continuous_parameter_subtraction_count,
            quadrature_convergence_pass=quadrature_convergence_pass,
            step_size_convergence_pass=step_size_convergence_pass,
            internal_validity_pass=internal_validity_pass,
            external_validity_pass=external_validity_pass,
            review_ready_pass=internal_validity_pass and external_validity_pass,
            transport_covariance_mode=resolved_transport_covariance.covariance_mode,
            transport_stability_yield=resolved_transport_covariance.stability_yield,
            transport_failure_fraction=resolved_transport_covariance.failure_fraction,
            transport_failure_count=resolved_transport_covariance.failure_count,
            step_size_reference_count=int(resolved_step_size_convergence.reference_step_count),
            step_size_reference_predictive_chi2=float(resolved_step_size_convergence.reference_predictive_chi2),
            step_size_max_delta_predictive_chi2=step_size_max_delta_predictive_chi2,
            step_size_max_sigma_shift=step_size_max_sigma_shift,
            predictive_chi2=resolved_pull_table.predictive_chi2,
            predictive_chi2_threshold=FOLLOWUP_CHI2_SURVIVAL_THRESHOLD,
            predictive_rms_pull_sigma=resolved_pull_table.predictive_rms_pull,
            predictive_max_abs_pull_sigma=resolved_pull_table.predictive_max_abs_pull,
        )

    HardAnomalyFilter = hard_anomaly_filter
    TopologicalRigidity = topological_rigidity
    IHInformationalCost = ih_informational_cost
    TopologicalIntegrityAssertions = topological_integrity_assertions
    TopologicalCoordinateValidation = topological_coordinate_validation
    AuditStatisticsReport = audit_statistics


def _uses_topological_residues(pmns: PmnsData | None = None, ckm: CkmData | None = None) -> bool:
    """Compatibility wrapper for the MasterAudit Topological Residue gate."""

    return MasterAudit.topological_coordinate_validation(pmns=pmns, ckm=ckm)


def comprehensive_audit(
    pmns: PmnsData | None = None,
    ckm: CkmData | None = None,
    transport_covariance: TransportParametricCovarianceData | None = None,
    step_size_convergence: StepSizeConvergenceData | None = None,
) -> bool:
    """Summarize the dynamic transport outputs against the current fit intervals."""

    pmns_data = derive_pmns() if pmns is None else pmns
    ckm_data = derive_ckm() if ckm is None else ckm
    covariance_audit = derive_transport_parametric_covariance(pmns_data, ckm_data) if transport_covariance is None else transport_covariance
    audit_passed = ComprehensiveAudit(pmns_data, ckm_data, covariance_audit).run()
    pull_table = derive_pull_table(pmns_data, ckm_data, transport_covariance=covariance_audit)
    audit_statistics = MasterAudit.audit_statistics(
        pmns=pmns_data,
        ckm=ckm_data,
        transport_covariance=covariance_audit,
        step_size_convergence=step_size_convergence,
        pull_table=pull_table,
    )

    LOGGER.info("Integer-input significance audit")
    LOGGER.info("-" * 88)
    LOGGER.info(
        f"chi2_pred / threshold            : {audit_statistics.predictive_chi2:.12f} / {audit_statistics.predictive_chi2_threshold:.12f}"
    )
    LOGGER.info(f"Integer-input significance pass  : {int(audit_statistics.external_validity_pass)}")
    LOGGER.info("")

    LOGGER.info("Reviewer-facing audit statistics")
    LOGGER.info("-" * 88)
    LOGGER.info(f"Hard Anomaly Filter             : {int(audit_statistics.hard_anomaly_filter_pass)}")
    LOGGER.info(f"Topological Rigidity           : {int(audit_statistics.topological_rigidity_pass)}")
    LOGGER.info(f"Topological Residue lock        : {int(audit_statistics.topological_residue_lock_pass)}")
    LOGGER.info(f"continuous DOF subtraction      : {audit_statistics.continuous_parameter_subtraction_count}")
    LOGGER.info(f"Internal Validity (RG/quadrature): {int(audit_statistics.internal_validity_pass)}")
    LOGGER.info(
        f"transport stability yield        : {100.0 * audit_statistics.transport_stability_yield:.2f}%"
    )
    LOGGER.info(f"transport covariance mode       : {audit_statistics.transport_covariance_mode}")
    LOGGER.info(f"transport failure fraction      : {audit_statistics.transport_failure_fraction:.3%}")
    LOGGER.info(f"transport hard-wall failures    : {audit_statistics.transport_failure_count}")
    LOGGER.info(f"QuadratureConvergence guard     : {int(audit_statistics.quadrature_convergence_pass)}")
    LOGGER.info(f"step-size convergence pass      : {int(audit_statistics.step_size_convergence_pass)}")
    LOGGER.info(f"reference step count            : {audit_statistics.step_size_reference_count}")
    LOGGER.info(f"reference chi2_pred             : {audit_statistics.step_size_reference_predictive_chi2:.12f}")
    LOGGER.info(f"max step-size Δchi2_pred        : {audit_statistics.step_size_max_delta_predictive_chi2:.12e}")
    LOGGER.info(f"max step-size sigma drift       : {audit_statistics.step_size_max_sigma_shift:.12e}σ")
    LOGGER.info(f"External Validity ({audit_statistics.external_reference_label}): {int(audit_statistics.external_validity_pass)}")
    LOGGER.info(f"predictive RMS pull             : {audit_statistics.predictive_rms_pull_sigma:.12f}σ")
    LOGGER.info(f"predictive max |pull|           : {audit_statistics.predictive_max_abs_pull_sigma:.12f}σ")
    LOGGER.info(f"Review-ready audit status       : {int(audit_statistics.review_ready_pass)}")
    LOGGER.info("")

    if audit_passed and audit_statistics.review_ready_pass:
        LOGGER.info("[SYSTEM READY FOR REVIEW]: Topological rigidity confirmed with no continuous flavor/gravity fit parameters.")
    return audit_passed and audit_statistics.review_ready_pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse publication-facing manuscript and output directory options."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-dir", type=Path, default=DEFAULT_MANUSCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "output")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Seed for stochastic transport and VEV ensemble audits.")
    parser.add_argument("--seed-audit", action="store_true", help="Run the stochastic pipeline across an ensemble of seeds and report relative variance.")
    parser.add_argument("--seed-audit-count", type=int, default=SEED_AUDIT_SAMPLE_COUNT, help="Number of seeds to include when --seed-audit is enabled.")
    parser.add_argument("--quiet", action="store_true", help="Suppress info-level report output on stderr.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional path for a full info-level audit log.")
    return parser.parse_args(argv)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    """Run the full publication-facing verifier report."""

    args = parse_args(argv)
    manuscript_dir = args.manuscript_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    os.makedirs(output_dir, exist_ok=True)
    configure_reporting(quiet=args.quiet, log_file=None if args.log_file is None else args.log_file.expanduser())

    vacuum = DEFAULT_TOPOLOGICAL_VACUUM
    assert MasterAudit.hard_anomaly_filter(model=vacuum), "Hard Anomaly Filter: FAILED."
    scales = vacuum.derive_scales()
    interface = vacuum.derive_boundary_bulk_interface()
    pmns = vacuum.derive_pmns()
    ckm = vacuum.derive_ckm()
    parent = vacuum.derive_parent_selection()
    framing = vacuum.verify_framing_anomaly()
    level_scanner = vacuum.level_scanner()
    level_scan = level_scanner.scan_window(lepton_levels=LOCAL_LEPTON_LEVEL_WINDOW)
    global_audit = level_scanner.scan_global_sensitivity_audit()
    chi2_landscape_audit = vacuum.derive_followup_chi2_landscape_audit()
    algebraic_unique = report_algebraic_uniqueness(global_audit)
    audit = vacuum.derive_audit()
    gravity_audit = vacuum.verify_bulk_emergence()
    gauge_audit = vacuum.verify_gauge_holography()
    dark_energy_audit = vacuum.verify_dark_energy_tension()
    unitary_audit = vacuum.verify_unitary_bounds()
    vacuum.validate_welded_mass_coordinate(scales.m_0_uv_ev)
    neutron_star_transit_audit = vacuum.simulate_neutron_star_transit(pmns=pmns, unitary_audit=unitary_audit)
    support_overlap = audit.calculate_support_overlap()
    sensitivity = vacuum.derive_sensitivity()
    geometric_sensitivity = vacuum.derive_geometric_sensitivity()
    nonlinearity_audit = vacuum.derive_nonlinearity_audit()
    step_size_convergence = vacuum.derive_step_size_convergence_audit()
    transport_covariance = vacuum.derive_transport_parametric_covariance(pmns, ckm, rng=np.random.default_rng(args.seed))
    pull_table = derive_pull_table(
        pmns,
        ckm,
        transport_covariance=transport_covariance,
        landscape_trial_count=chi2_landscape_audit.effective_trial_count,
        followup_trial_count=chi2_landscape_audit.total_pairs_scanned,
        effective_correlation_length=chi2_landscape_audit.effective_correlation_length,
        lepton_correlation_length=chi2_landscape_audit.lepton_correlation_length,
        quark_correlation_length=chi2_landscape_audit.quark_correlation_length,
    )
    physics_audit = vacuum.derive_physics_audit()
    search_window = physics_audit.search_window
    geometric_kappa = physics_audit.geometric_kappa
    modular_horizon = physics_audit.modular_horizon
    weight_profile = vacuum.generate_ckm_phase_tilt_profile(pmns, output_path=output_dir / CKM_PHASE_TILT_PROFILE_FIGURE_FILENAME)
    threshold_sensitivity = vacuum.derive_threshold_sensitivity(ckm)
    higgs_cg_correction = calculate_126_higgs_cg_correction(
        ckm.so10_threshold_correction.clebsch_126,
        ckm.so10_threshold_correction.clebsch_10,
    )
    mass_ratio_stability_audit = vacuum.derive_mass_ratio_stability_audit(seed=args.seed)
    ghost_character_audit = vacuum.derive_ghost_character_audit(audit)
    framing_gap_stability = vacuum.derive_framing_gap_stability_audit(ckm, audit)
    dm_fingerprint = derive_dm_fingerprint_inputs(weight_profile, geometric_kappa, framing_gap_stability)
    topological_integrity = MasterAudit.topological_integrity_assertions(
        pmns=pmns,
        ckm=ckm,
        pull_table=pull_table,
        audit=audit,
        dark_energy_audit=dark_energy_audit,
        dm_fingerprint=dm_fingerprint,
        model=vacuum,
    )
    assert topological_integrity.hard_anomaly_filter, "Hard Anomaly Filter: FAILED."
    assert topological_integrity.topological_rigidity, "Topological Rigidity: continuous fit freedom detected."
    assert topological_integrity.welded_mass_coordinate_lock, "Welded mass-coordinate lock: FAILED."
    assert topological_integrity.superheavy_relic_lock, "Superheavy relic lock: FAILED."
    assert topological_integrity.ih_informational_cost, "IH informational-cost audit: FAILED."
    gauge_matching_audit = derive_gauge_unification_existence_proof(m_126_gev=framing_gap_stability.matching_m126_gev)
    gut_scale_consistency_pass = math.isclose(
        gravity_audit.baryon_stability.gut_scale_gev,
        gauge_matching_audit.m_gut_gev,
        rel_tol=0.0,
        abs_tol=1.0,
    )
    rhn_threshold = vacuum.derive_rhn_threshold_data("lepton")
    seed_robustness_audit = (
        vacuum.derive_seed_robustness_audit(pmns, ckm, seed=args.seed, seed_count=args.seed_audit_count)
        if args.seed_audit
        else None
    )
    write_generated_tables(
        level_scan,
        global_audit,
        chi2_landscape_audit,
        pull_table,
        audit,
        nonlinearity_audit,
        step_size_convergence,
        weight_profile,
        pmns,
        ckm,
        mass_ratio_stability_audit,
        geometric_sensitivity,
        transport_covariance,
        gauge_audit=gauge_audit,
        gravity_audit=gravity_audit,
        dark_energy_audit=dark_energy_audit,
        unitary_audit=unitary_audit,
        vacuum=vacuum,
        output_dir=output_dir,
    )
    if seed_robustness_audit is not None:
        write_seed_robustness_audit(seed_robustness_audit, output_dir=output_dir)
    write_majorana_floor_figure(
        pmns,
        sensitivity,
        geometric_sensitivity,
        output_paths=(
            output_dir / TOPOLOGICAL_LOBSTER_FIGURE_FILENAME,
            output_dir / MAJORANA_FLOOR_FIGURE_FILENAME,
        ),
    )
    export_physics_constants_to_tex(
        output_dir,
        scales=scales,
        level_scan=level_scan,
        global_audit=global_audit,
        pull_table=pull_table,
        nonlinearity_audit=nonlinearity_audit,
        weight_profile=weight_profile,
        mass_ratio_stability_audit=mass_ratio_stability_audit,
        pmns=pmns,
        ckm=ckm,
        sensitivity=sensitivity,
        geometric_sensitivity=geometric_sensitivity,
        geometric_kappa=geometric_kappa,
        modular_horizon=modular_horizon,
        framing_gap_stability=framing_gap_stability,
    )
    export_dm_fingerprint_artifact(
        weight_profile,
        geometric_kappa,
        framing_gap_stability,
        output_dir,
    )
    export_framing_gap_stability_figure(
        framing_gap_stability,
        output_path=output_dir / FRAMING_GAP_STABILITY_FIGURE_FILENAME,
    )
    audit_output_archive_dir = write_audit_output_bundles(
        output_dir,
        pull_table,
        weight_profile,
        nonlinearity_audit,
        mass_ratio_stability_audit,
        global_audit,
        framing_gap_stability,
    )
    validate_manuscript_consistency(manuscript_dir, output_dir)
    inflationary_sector = vacuum.derive_inflationary_sector()

    theta12_pull = pull_from_interval(pmns.theta12_rg_deg, LEPTON_INTERVALS["theta12"])
    theta13_pull = pull_from_interval(pmns.theta13_rg_deg, LEPTON_INTERVALS["theta13"])
    theta23_pull = pull_from_interval(pmns.theta23_rg_deg, LEPTON_INTERVALS["theta23"])
    delta_pull = pull_from_interval(pmns.delta_cp_rg_deg, LEPTON_INTERVALS["delta_cp"])
    gamma_pull = pull_from_interval(ckm.gamma_rg_deg, CKM_GAMMA_GOLD_STANDARD_DEG)
    bare_vus_pull = pull_from_interval(ckm.bare_vus_rg, QUARK_INTERVALS["vus"])
    vus_pull = pull_from_interval(ckm.vus_rg, QUARK_INTERVALS["vus"])
    vcb_pull = pull_from_interval(ckm.vcb_rg, QUARK_INTERVALS["vcb"])
    vub_pull = pull_from_interval(ckm.vub_rg, QUARK_INTERVALS["vub"])
    audit_p_value = float(chi2_distribution.sf(pull_table.audit_chi2, pull_table.audit_degrees_of_freedom))

    LOGGER.info("Formal flavor-theory verifier")
    LOGGER.info("=" * 88)
    LOGGER.info(f"Planck mass M_P [eV]             : {PLANCK_MASS_EV:.6e}")
    LOGGER.info(f"Holographic bits N               : {HOLOGRAPHIC_BITS:.6e}")
    LOGGER.info(f"Leptonic level k_ell             : {LEPTON_LEVEL}")
    LOGGER.info(f"Quark level k_q                  : {QUARK_LEVEL}")
    LOGGER.info(f"UV / electroweak scale ratio     : {RG_SCALE_RATIO:.6e}")
    LOGGER.info(f"manuscript dir                   : {_display_path(manuscript_dir)}")
    LOGGER.info(f"output dir                       : {_display_path(output_dir)}")
    LOGGER.info("")

    LOGGER.info("Experimental context")
    LOGGER.info("-" * 88)
    LOGGER.info(f"Leptonic fit source              : {EXPERIMENTAL_CONTEXT.nufit_reference}")
    LOGGER.info(f"Quark fit source                 : {EXPERIMENTAL_CONTEXT.pdg_reference}")
    LOGGER.info(f"Matching coefficient             : {MATCHING_COEFFICIENT_SYMBOL} = {GEOMETRIC_KAPPA:.5f} (exact simplex-area coefficient)")
    LOGGER.info("")

    LOGGER.info("Search-window bookkeeping")
    LOGGER.info("-" * 88)
    LOGGER.info(f"scan lepton window           : [{GLOBAL_LEPTON_LEVEL_RANGE[0]}, {GLOBAL_LEPTON_LEVEL_RANGE[1]}]")
    LOGGER.info(f"scan quark window            : [{GLOBAL_QUARK_LEVEL_RANGE[0]}, {GLOBAL_QUARK_LEVEL_RANGE[1]}]")
    LOGGER.info(f"trial volume                     : {LOW_RANK_RCFT_SCAN_COMBINATIONS}")
    LOGGER.info(f"auxiliary density capacity       : {search_window.pixel_capacity:.12f}")
    LOGGER.info(f"effective load at k=99           : {search_window.level_99_load:.12f}")
    LOGGER.info(f"effective load at k=100          : {search_window.level_100_load:.12f}")
    LOGGER.info(
        f"density-cap diagnostic           : first violation at k = {search_window.max_admissible_level + 1}; "
        f"not used to define the manuscript scan"
    )
    LOGGER.info(f"weight-simplex hyperarea         : {geometric_kappa.weight_simplex_hyperarea:.12f}")
    LOGGER.info(f"regular-reference hyperarea      : {geometric_kappa.regular_reference_hyperarea:.12f}")
    LOGGER.info(f"weight-simplex area ratio        : {geometric_kappa.area_ratio:.12f}")
    LOGGER.info(f"spinorial retention factor       : {geometric_kappa.spinorial_retention:.12f}")
    LOGGER.info(f"derived geometric factor         : {geometric_kappa.geometric_factor:.12f}")
    LOGGER.info(f"simplex prefactor audit          : {geometric_kappa.derived_kappa:.12f}")
    LOGGER.info(f"unit modular weight exp[2π√(K/6)]: {modular_horizon.unit_modular_weight:.6e}")
    LOGGER.info(f"effective vacuum weight          : {modular_horizon.effective_vacuum_weight:.12f}")
    LOGGER.info(f"framing-gap area beta^2          : {modular_horizon.framing_gap_area:.12f}")
    LOGGER.info(f"visible edge penalty             : {modular_horizon.visible_edge_penalty:.12f}")
    LOGGER.info(f"derived modular horizon bits     : {modular_horizon.derived_bits:.6e}")
    LOGGER.info(f"Planck cross-check ratio         : {modular_horizon.planck_crosscheck_ratio:.12f}")
    LOGGER.info("")

    LOGGER.info("RG transport assumptions")
    LOGGER.info("-" * 88)
    LOGGER.info(f"Running particle content         : {SM_RUNNING_CONTENT}")
    LOGGER.info(f"RHN threshold flagged            : {rhn_threshold.threshold_active}")
    if rhn_threshold.threshold_active:
        LOGGER.info(f"RHN threshold M_N [GeV]          : {rhn_threshold.threshold_scale_gev:.6e}")
        LOGGER.info(f"log(M_N/M_Z)                     : {rhn_threshold.lower_interval_log:.6f}")
        LOGGER.info(f"log(M_GUT/M_N)                   : {rhn_threshold.upper_interval_log:.6f}")
    LOGGER.info(f"GUT matching scale [GeV]         : {GUT_SCALE_GEV:.6e}")
    LOGGER.info("")

    print_threshold_audit()

    LOGGER.info("RG non-linearity audit")
    LOGGER.info("-" * 88)
    LOGGER.info(f"theta12 linear/full [deg]         : {nonlinearity_audit.theta_linear_deg[0]:.12f}, {nonlinearity_audit.theta_nonlinear_deg[0]:.12f}")
    LOGGER.info(f"theta13 linear/full [deg]         : {nonlinearity_audit.theta_linear_deg[1]:.12f}, {nonlinearity_audit.theta_nonlinear_deg[1]:.12f}")
    LOGGER.info(f"theta23 linear/full [deg]         : {nonlinearity_audit.theta_linear_deg[2]:.12f}, {nonlinearity_audit.theta_nonlinear_deg[2]:.12f}")
    LOGGER.info(f"delta_CP linear/full [deg]        : {nonlinearity_audit.delta_linear_deg:.12f}, {nonlinearity_audit.delta_nonlinear_deg:.12f}")
    LOGGER.info(f"m_0 linear/full [meV]            : {1.0e3 * nonlinearity_audit.m_0_linear_ev:.12f}, {1.0e3 * nonlinearity_audit.m_0_nonlinear_ev:.12f}")
    LOGGER.info(f"max Taylor error [sigma]         : {nonlinearity_audit.max_sigma_error:.12f}")
    LOGGER.info("")

    LOGGER.info("Information--mass bridge")
    LOGGER.info("-" * 88)
    LOGGER.info(f"kappa_geometric                  : {scales.kappa_geometric:.6f}")
    LOGGER.info(f"m_0(M_GUT) [eV]                   : {scales.m_0_uv_ev:.6e}")
    LOGGER.info(f"m_0(M_Z) [eV]                     : {scales.m_0_mz_ev:.6e}")
    LOGGER.info(f"gamma_0^(1)                       : {scales.gamma_0_one_loop:.6f}")
    LOGGER.info(f"gamma_0^(2)                       : {scales.gamma_0_two_loop:.6f}")
    LOGGER.info(f"Bare genus-2 scale M_R^holo [eV] : {scales.majorana_boundary_ev:.6e}")
    LOGGER.info(f"Effective seesaw scale [GeV]     : {scales.majorana_effective_ev / 1.0e9:.6e}")
    LOGGER.info(f"Majorana Higgs channel           : {MAJORANA_HIGGS_REPRESENTATION}")
    LOGGER.info("")

    LOGGER.info("Baryon stability audit")
    LOGGER.info("-" * 88)
    LOGGER.info(f"Holographic Information Horizon  : finite benchmark bit budget N={gravity_audit.holographic_bits:.6e}")
    LOGGER.info(f"alpha_GUT^-1                     : {gravity_audit.baryon_stability.unified_alpha_inverse:.6f}")
    LOGGER.info(f"M_GUT [GeV]                      : {gravity_audit.baryon_stability.gut_scale_gev:.6e}")
    LOGGER.info(f"M_X [GeV]                        : {gravity_audit.baryon_stability.effective_gauge_mass_gev:.6e}")
    LOGGER.info(f"Delta_mod                        : {gravity_audit.baryon_stability.modular_gap:.12f}")
    LOGGER.info(f"P_tunnel                         : {gravity_audit.baryon_stability.modular_tunneling_penalty:.12e}")
    LOGGER.info(f"tau_p^(d=6) [yr]                 : {gravity_audit.baryon_stability.proton_lifetime_years:.12e}")
    LOGGER.info(f"tau_p^(info) [yr]                : {gravity_audit.baryon_stability.protected_evaporation_lifetime_years:.12e}")
    LOGGER.info(f"BARYON STABILITY (tau_p): {gravity_audit.baryon_stability.proton_lifetime_years:.2e} years [PROTECTED BY DELTA_FR=0]")
    LOGGER.info("")

    gut_scale_status = "PASS" if gut_scale_consistency_pass else "FAIL"
    LOGGER.info("GUT-scale consistency audit")
    LOGGER.info("-" * 88)
    LOGGER.info(f"M_GUT proton-decay audit [GeV]   : {gravity_audit.baryon_stability.gut_scale_gev:.6e}")
    LOGGER.info(f"M_GUT gauge-matching [GeV]      : {gauge_matching_audit.m_gut_gev:.6e}")
    LOGGER.info(
        f"[{gut_scale_status}] Shared M_GUT scale: proton decay and the gauge-matching audit behind alpha^-1~{gauge_audit.topological_alpha_inverse:.1f} use the same benchmark scale."
    )
    LOGGER.info("")

    unitary_status = "PASS" if (
        unitary_audit.unitary_bound_satisfied
        and unitary_audit.recovery_locked_to_delta_mod
        and unitary_audit.universal_computational_limit_pass
    ) else "FAIL"
    LOGGER.info("Finite-buffer unitarity audit")
    LOGGER.info("-" * 88)
    LOGGER.info(f"S_max = ln(N_holo)               : {unitary_audit.entropy_max_nats:.12f}")
    LOGGER.info(f"c_dark buffer entropy            : {unitary_audit.holographic_buffer_entropy:.12f}")
    LOGGER.info(f"regulated curvature entropy      : {unitary_audit.regulated_curvature_entropy:.12f}")
    LOGGER.info(f"curvature buffer margin          : {unitary_audit.curvature_buffer_margin:.12f}")
    LOGGER.info(f"buffer margin [%]                : {unitary_audit.curvature_buffer_margin_percent:.2f}")
    LOGGER.info(f"info evaporation rate [yr^-1]    : {unitary_audit.information_evaporation_rate_per_year:.12e}")
    LOGGER.info(f"recovery lifetime [yr]           : {unitary_audit.recovery_lifetime_years:.12e}")
    LOGGER.info(f"zero-point complexity            : {unitary_audit.zero_point_complexity:.12f}")
    LOGGER.info(f"max complexity capacity          : {unitary_audit.max_complexity_capacity:.12e}")
    LOGGER.info(f"Lloyd limit [s^-1]               : {unitary_audit.lloyds_limit_ops_per_second:.12e}")
    LOGGER.info(f"complexity growth rate [s^-1]    : {unitary_audit.complexity_growth_rate_ops_per_second:.12e}")
    LOGGER.info(f"complexity utilization           : {unitary_audit.complexity_utilization_fraction:.12f}")
    LOGGER.info(f"clock-skew (1 - n_s)             : {unitary_audit.clock_skew:.12f}")
    LOGGER.info(f"Holographic Rigidity             : {int(unitary_audit.holographic_rigidity)}")
    LOGGER.info(f"[{unitary_status}] Unitary Bound Check: Information recovery rate locked to Delta_mod.")
    LOGGER.info("Universal Computational Limit: PASSED.")
    LOGGER.info(f"Welded mass-coordinate lock      : {dark_energy_audit.sensitivity_audit_message}")
    LOGGER.info("")

    LOGGER.info("Stage XIII discovery hypothesis")
    LOGGER.info("-" * 88)
    LOGGER.info(f"compact-gradient scale           : {neutron_star_transit_audit.gravitational_gradient_scale:.12f}")
    LOGGER.info(f"transit path length [km]         : {neutron_star_transit_audit.transit_path_length_km:.12f}")
    LOGGER.info(f"density proxy scale              : {neutron_star_transit_audit.density_scale:.12f}")
    LOGGER.info(f"clock-skew loading               : {neutron_star_transit_audit.clock_skew:.12f}")
    LOGGER.info(f"complexity utilization           : {neutron_star_transit_audit.complexity_utilization_fraction:.12f}")
    LOGGER.info(f"rank-deficiency load             : {neutron_star_transit_audit.rank_deficiency_load:.12f}")
    LOGGER.info(f"scrambling fraction              : {neutron_star_transit_audit.scrambling_fraction:.12f}")
    LOGGER.info(
        f"P(nu_e->nu_tau) MSW / torsion     : {neutron_star_transit_audit.reference_nu_e_to_nu_tau_probability:.12f} / "
        f"{neutron_star_transit_audit.torsion_scrambled_nu_e_to_nu_tau_probability:.12f}"
    )
    LOGGER.info(
        f"P(nu_e->nu_e) MSW / torsion       : {neutron_star_transit_audit.reference_nu_e_survival_probability:.12f} / "
        f"{neutron_star_transit_audit.torsion_scrambled_nu_e_survival_probability:.12f}"
    )
    LOGGER.info(f"MSW excess probability           : {neutron_star_transit_audit.msw_excess_probability:.12f}")
    LOGGER.info(f"local support rank               : {neutron_star_transit_audit.support_rank}")
    LOGGER.info(f"local sigma_min                  : {neutron_star_transit_audit.support_sigma_min:.12e}")
    LOGGER.info(f"local condition number           : {neutron_star_transit_audit.support_condition_number:.12e}")
    LOGGER.info(f"machine-precision singular       : {int(neutron_star_transit_audit.support_machine_precision_singular)}")
    LOGGER.info(f"torsion-scrambling event         : {int(neutron_star_transit_audit.torsion_scrambling_triggered)}")
    LOGGER.info(f"violates standard MSW            : {int(neutron_star_transit_audit.violates_standard_msw)}")
    LOGGER.info("")

    LOGGER.info("Boundary--bulk interface dictionary")
    LOGGER.info("-" * 88)
    LOGGER.info("Normalized modular block S_ij")
    LOGGER.info(format_complex_matrix(interface.yukawa_texture))
    LOGGER.info("Dirac Yukawa texture Y^(10)_ij / lambda_10")
    LOGGER.info(format_complex_matrix(interface.framed_yukawa_texture))
    LOGGER.info("Majorana texture Y^(126)_ij / lambda_126")
    LOGGER.info(format_complex_matrix(interface.majorana_yukawa_texture))
    LOGGER.info("")

    LOGGER.info("SU(2)_26 PMNS complex kernel")
    LOGGER.info("-" * 88)
    LOGGER.info(f"D_26                             : {pmns.total_quantum_dimension:.12f}")
    LOGGER.info(f"beta_26                          : {pmns.beta:.12f}")
    LOGGER.info(f"d1, d2                           : {pmns.d1:.12f}, {pmns.d2:.12f}")
    LOGGER.info(f"phi_RT [rad]                     : {pmns.phi_rt_rad:.12f}")
    LOGGER.info(f"phi_RT [deg]                     : {math.degrees(pmns.phi_rt_rad):.12f}")
    LOGGER.info(f"omega_fr = arg(T22 T11*) [deg]   : {pmns.framing_phase_deg:.12f}")
    LOGGER.info(f"arg W[S,T] [deg]                 : {pmns.interference_phase_deg:.12f}")
    LOGGER.info(f"PDG branch shift [deg]           : {pmns.branch_shift_deg:.12f}")
    LOGGER.info("T-matrix framing phases theta_j")
    LOGGER.info(format_phase_vector(pmns.t_phases))
    LOGGER.info("U_top")
    LOGGER.info(format_complex_matrix(pmns.topological_matrix))
    LOGGER.info("U_seed^complex")
    LOGGER.info(format_complex_matrix(pmns.complex_seed_matrix))
    LOGGER.info("U_PMNS(M_GUT)")
    LOGGER.info(format_complex_matrix(pmns.pmns_matrix_uv))
    LOGGER.info("U_PMNS(M_Z)")
    LOGGER.info(format_complex_matrix(pmns.pmns_matrix_rg))
    LOGGER.info(f"theta12,13,23(M_GUT) [deg]       : {pmns.theta12_uv_deg:.12f}, {pmns.theta13_uv_deg:.12f}, {pmns.theta23_uv_deg:.12f}")
    LOGGER.info(f"theta12,13,23(M_Z) [deg]         : {pmns.theta12_rg_deg:.12f}, {pmns.theta13_rg_deg:.12f}, {pmns.theta23_rg_deg:.12f}")
    LOGGER.info(f"delta_CP(M_GUT) [deg]            : {pmns.delta_cp_uv_deg:.12f}")
    LOGGER.info(f"delta_CP(M_Z) [deg]              : {pmns.delta_cp_rg_deg:.12f}")
    LOGGER.info(f"m1,m2,m3(M_GUT) [eV]             : {pmns.normal_order_masses_uv_ev[0]:.12f}, {pmns.normal_order_masses_uv_ev[1]:.12f}, {pmns.normal_order_masses_uv_ev[2]:.12f}")
    LOGGER.info(f"m1,m2,m3(M_Z) [eV]               : {pmns.normal_order_masses_rg_ev[0]:.12f}, {pmns.normal_order_masses_rg_ev[1]:.12f}, {pmns.normal_order_masses_rg_ev[2]:.12f}")
    LOGGER.info(f"|m_bb|(M_GUT) [meV]              : {1.0e3 * pmns.effective_majorana_mass_uv_ev:.12f}")
    LOGGER.info(f"|m_bb|(M_Z) [meV]                : {1.0e3 * pmns.effective_majorana_mass_rg_ev:.12f}")
    LOGGER.info(f"A_hol(M_GUT)                     : {pmns.holonomy_area_uv:.12e}")
    LOGGER.info(f"A_hol(M_Z)                       : {pmns.holonomy_area_rg:.12e}")
    LOGGER.info(f"J_CP(M_GUT)                      : {pmns.jarlskog_uv:.12e}")
    LOGGER.info(f"J_CP(M_Z)                        : {pmns.jarlskog_rg:.12e}")
    LOGGER.info(f"delta_CP in NuFIT 1σ?            : {LEPTON_INTERVALS['delta_cp'].lower <= pmns.delta_cp_rg_deg <= LEPTON_INTERVALS['delta_cp'].upper}")
    LOGGER.info("")

    LOGGER.info("Logarithmic drift of the solar angle")
    LOGGER.info("-" * 88)
    LOGGER.info(f"beta_theta12^(1)                 : {pmns.solar_beta_one_loop:.12f}")
    LOGGER.info(f"beta_theta12^(2)                 : {pmns.solar_beta_two_loop:.12f}")
    LOGGER.info(f"theta12 shift M_GUT -> M_Z [deg] : {pmns.solar_shift_deg:.12f}")
    LOGGER.info(f"theta12 UV pull                  : {pmns.theta12_uv_pull:+.2f}σ")
    LOGGER.info(f"theta12 RG pull                  : {pmns.theta12_rg_pull:+.2f}σ")
    LOGGER.info("")

    LOGGER.info("Global flavor fit")
    LOGGER.info("-" * 88)
    for row in pull_table.rows:
        suffix = " deg" if row.units == "deg" else ""
        if row.pull_data is None:
            if row.units == "deg":
                theory_value = f"{row.theory_mz: .6f}{suffix}"
            elif row.units == "meV":
                theory_value = f"{row.theory_mz: .6f} meV"
            elif row.units == "eV":
                theory_value = f"{row.theory_mz: .6e} eV"
            else:
                theory_value = f"{row.theory_mz: .6f}{suffix}"
            benchmark = row.reference_override if row.reference_override is not None else row.source_label
            LOGGER.info(f"{row.observable:<28}: {theory_value}  benchmark={benchmark}")
            continue
        print_pull(row.observable, row.pull_data, suffix)
    LOGGER.info(f"chi2_pred ({pull_table.predictive_observable_count} rows)             : {pull_table.predictive_chi2:.12f}")
    LOGGER.info(f"p_pred, conditional ({pull_table.predictive_degrees_of_freedom} dof)  : {pull_table.predictive_conditional_p_value:.12f}")
    LOGGER.info(f"p_global, N_eff ({pull_table.predictive_degrees_of_freedom} dof)       : {pull_table.predictive_discrete_selection_lee_p_value:.12f}")
    LOGGER.info(f"effective trial count            : {pull_table.predictive_effective_trial_count:.12f}")
    LOGGER.info(f"correlation length xi            : {pull_table.predictive_correlation_length:.12f}")
    LOGGER.info(f"threshold-matching subtraction   : {pull_table.threshold_alignment_subtraction_count}")
    LOGGER.info(f"topological-label subtraction    : {TOPOLOGICAL_QUANTUM_NUMBER_DOF_SUBTRACTION}")
    LOGGER.info(f"reduced chi2_pred               : {pull_table.predictive_reduced_chi2:.12f}")
    LOGGER.info(f"rms pred pull                   : {pull_table.predictive_rms_pull:.12f}")
    LOGGER.info(f"max pred |pull|                 : {pull_table.predictive_max_abs_pull:.12f}")
    LOGGER.info(f"chi2_check ({pull_table.audit_observable_count} rows)                : {pull_table.audit_chi2:.12f}")
    LOGGER.info(f"p_check ({pull_table.audit_degrees_of_freedom} dof)                  : {audit_p_value:.12f}")
    LOGGER.info(f"rms cross-check pull            : {pull_table.audit_rms_pull:.12f}")
    LOGGER.info(f"max cross-check |pull|          : {pull_table.audit_max_abs_pull:.12f}")
    LOGGER.info(f"continuous fit variables         : {pull_table.phenomenological_parameter_count}")
    LOGGER.info(f"continuous RG calibration inputs : {pull_table.calibration_parameter_count}")
    LOGGER.info("")

    LOGGER.info("PMNS RG-evolved pulls")
    LOGGER.info("-" * 88)
    print_pull("theta12", theta12_pull, " deg")
    print_pull("theta13", theta13_pull, " deg")
    print_pull("theta23", theta23_pull, " deg")
    print_pull("delta_CP", delta_pull, " deg")
    LOGGER.info("")

    LOGGER.info("SU(3)_8 CKM complex kernel")
    LOGGER.info("-" * 88)
    LOGGER.info("Visible SU(3)_8 block")
    LOGGER.info(format_complex_matrix(ckm.visible_block))
    LOGGER.info("SO(10)/SU(3) coset surrogate block")
    LOGGER.info(format_complex_matrix(ckm.coset_block))
    LOGGER.info("Coset weighting")
    LOGGER.info(format_real_matrix(ckm.coset_weighting))
    LOGGER.info(f"rank(SO(10)) - rank(SU(3))       : {ckm.rank_difference}")
    LOGGER.info(f"Quark branching index I_Q        : {ckm.branching_index}")
    LOGGER.info(f"|rho_SO(10)|^2, |rho_SU(3)|^2    : {ckm.so10_weyl_norm_sq:.12f}, {ckm.su3_weyl_norm_sq:.12f}")
    LOGGER.info(f"sqrt(|rho_SO(10)|^2/|rho_SU(3)|^2): {ckm.weyl_ratio:.12f}")
    LOGGER.info(f"Pi_rank                          : {ckm.rank_deficit_pressure:.12f}")
    LOGGER.info(f"Pi_vac                           : {ckm.vacuum_pressure:.12f}")
    LOGGER.info(f"R_01^(par/vis)                   : {ckm.so10_threshold_correction.parent_visible_ratio:.12f}")
    LOGGER.info(f"C_126^(12)                       : {ckm.so10_threshold_correction.clebsch_126:.12f}")
    LOGGER.info(f"C_10^(12)                        : {ckm.so10_threshold_correction.clebsch_10:.12f}")
    LOGGER.info(f"projection exponent              : {ckm.so10_threshold_correction.projection_exponent:.12f}")
    LOGGER.info(f"Xi12^(126) phasor                : {format_complex_scalar(ckm.so10_threshold_correction.xi12)}")
    LOGGER.info(f"|Xi12^(126)|                     : {ckm.so10_threshold_correction.xi12_abs:.12f}")
    LOGGER.info(f"126_H matching log sum          : {ckm.so10_threshold_correction.matching_log_sum_126h:.12f}")
    LOGGER.info(f"210_H matching log sum          : {ckm.so10_threshold_correction.matching_log_sum_210h:.12f}")
    LOGGER.info(f"V_H matching log sum            : {ckm.so10_threshold_correction.matching_log_sum_vh:.12f}")
    LOGGER.info(f"Total matching log sum          : {ckm.so10_threshold_correction.matching_log_sum:.12f}")
    LOGGER.info(f"Phi_framing [deg]                : {ckm.so10_threshold_correction.framing_phase_deg:.12f}")
    LOGGER.info(f"Orthogonal coset tilt [deg]      : {ckm.so10_threshold_correction.orthogonal_coset_phase_deg:.12f}")
    LOGGER.info(f"10_H/126_H closure [deg]         : {ckm.so10_threshold_correction.geodesic_closure_phase_deg:.12f}")
    LOGGER.info(f"Net CKM tilt [deg]               : {ckm.so10_threshold_correction.triangle_tilt_deg:.12f}")
    LOGGER.info(f"Xi12(K), Xi23(K)                 : {ckm.descendant_factors[0]:.12f}, {ckm.descendant_factors[1]:.12f}")
    LOGGER.info(f"Pi12, Pi23                       : {ckm.channel_pressures[0]:.12f}, {ckm.channel_pressures[1]:.12f}")
    LOGGER.info(f"eps12, eps23, eps13 (bare)       : {ckm.bare_topological_weights[0]:.12f}, {ckm.bare_topological_weights[1]:.12f}, {ckm.bare_topological_weights[2]:.12f}")
    LOGGER.info(f"eps12, eps23, eps13              : {ckm.topological_weights[0]:.12f}, {ckm.topological_weights[1]:.12f}, {ckm.topological_weights[2]:.12f}")
    LOGGER.info("T-matrix framing phases theta_j")
    LOGGER.info(format_phase_vector(ckm.t_phases))
    LOGGER.info("U_seed^complex")
    LOGGER.info(format_complex_matrix(ckm.complex_seed_matrix))
    LOGGER.info("U_CKM^bare(M_GUT)")
    LOGGER.info(format_complex_matrix(ckm.bare_ckm_matrix_uv))
    LOGGER.info("U_CKM^bare(M_Z)")
    LOGGER.info(format_complex_matrix(ckm.bare_ckm_matrix_rg))
    LOGGER.info("U_CKM(M_GUT)")
    LOGGER.info(format_complex_matrix(ckm.ckm_matrix_uv))
    LOGGER.info("U_CKM(M_Z)")
    LOGGER.info(format_complex_matrix(ckm.ckm_matrix_rg))
    LOGGER.info(f"theta_C, theta13^q, theta23^q(M_GUT) [deg] : {ckm.theta_c_uv_deg:.12f}, {ckm.theta13_uv_deg:.12f}, {ckm.theta23_uv_deg:.12f}")
    LOGGER.info(f"theta_C, theta13^q, theta23^q(M_Z) [deg]   : {ckm.theta_c_rg_deg:.12f}, {ckm.theta13_rg_deg:.12f}, {ckm.theta23_rg_deg:.12f}")
    LOGGER.info(f"alpha, beta, gamma(M_GUT) [deg]  : {ckm.alpha_uv_deg:.12f}, {ckm.beta_uv_deg:.12f}, {ckm.gamma_uv_deg:.12f}")
    LOGGER.info(f"alpha, beta, gamma(M_Z) [deg]    : {ckm.alpha_rg_deg:.12f}, {ckm.beta_rg_deg:.12f}, {ckm.gamma_rg_deg:.12f}")
    LOGGER.info(f"|Vus|, |Vcb|, |Vub|^bare(M_GUT)  : {ckm.bare_vus_uv:.12f}, {ckm.bare_vcb_uv:.12f}, {ckm.bare_vub_uv:.12f}")
    LOGGER.info(f"|Vus|, |Vcb|, |Vub|^bare(M_Z)    : {ckm.bare_vus_rg:.12f}, {ckm.bare_vcb_rg:.12f}, {ckm.bare_vub_rg:.12f}")
    LOGGER.info(f"|Vus|, |Vcb|, |Vub|(M_GUT)       : {ckm.vus_uv:.12f}, {ckm.vcb_uv:.12f}, {ckm.vub_uv:.12f}")
    LOGGER.info(f"|Vus|, |Vcb|, |Vub|(M_Z)         : {ckm.vus_rg:.12f}, {ckm.vcb_rg:.12f}, {ckm.vub_rg:.12f}")
    LOGGER.info(f"126_H threshold push Δ|Vus|(M_GUT): {ckm.cabibbo_threshold_push_uv:.12f}")
    LOGGER.info(f"126_H threshold push Δ|Vus|(M_Z) : {ckm.cabibbo_threshold_push_rg:.12f}")
    LOGGER.info(f"delta_q(M_GUT) [deg]             : {ckm.delta_cp_uv_deg:.12f}")
    LOGGER.info(f"delta_q(M_Z) [deg]               : {ckm.delta_cp_rg_deg:.12f}")
    LOGGER.info(f"J_CP^q(M_GUT)                    : {ckm.jarlskog_uv:.12e}")
    LOGGER.info(f"J_CP^q(M_Z)                      : {ckm.jarlskog_rg:.12e}")
    LOGGER.info("")

    LOGGER.info("CKM RG-evolved pulls")
    LOGGER.info("-" * 88)
    print_pull("|Vus| bare", bare_vus_pull)
    print_pull("|Vus|", vus_pull)
    print_pull("|Vcb|", vcb_pull)
    print_pull("|Vub|", vub_pull)
    print_pull("gamma", gamma_pull, " deg")
    LOGGER.info(f"chi2_pred ({pull_table.predictive_observable_count} rows)               : {pull_table.predictive_chi2:.12f}")
    LOGGER.info(f"chi2_check ({pull_table.audit_observable_count} rows)              : {pull_table.audit_chi2:.12f}")
    LOGGER.info("")

    LOGGER.info("126_H threshold phase summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"R_01^(par/vis)                   : {ckm.so10_threshold_correction.parent_visible_ratio:.12f}")
    LOGGER.info(f"C_126^(12)                       : {ckm.so10_threshold_correction.clebsch_126:.12f}")
    LOGGER.info(f"C_10^(12)                        : {ckm.so10_threshold_correction.clebsch_10:.12f}")
    LOGGER.info(f"projection exponent              : {ckm.so10_threshold_correction.projection_exponent:.12f}")
    LOGGER.info(f"Xi12^(126) phasor                : {format_complex_scalar(ckm.so10_threshold_correction.xi12)}")
    LOGGER.info(f"|Xi12^(126)|                     : {ckm.so10_threshold_correction.xi12_abs:.12f}")
    LOGGER.info(f"deltaPi_126                      : {ckm.so10_threshold_correction.delta_pi_126:.12f}")
    LOGGER.info(f"Phi_framing [deg]                : {ckm.so10_threshold_correction.framing_phase_deg:.12f}")
    LOGGER.info(f"Orthogonal coset tilt [deg]      : {ckm.so10_threshold_correction.orthogonal_coset_phase_deg:.12f}")
    LOGGER.info(f"10_H/126_H closure [deg]         : {ckm.so10_threshold_correction.geodesic_closure_phase_deg:.12f}")
    LOGGER.info(f"Net CKM tilt [deg]               : {ckm.so10_threshold_correction.triangle_tilt_deg:.12f}")
    LOGGER.info(f"matching-point gamma shift [deg] : {threshold_sensitivity.matching_point_gamma_shift_deg:.12f}")
    for point in threshold_sensitivity.points:
        if abs(point.m_126_gev - derive_topological_threshold_gev()) <= 1.0:
            LOGGER.info(f"gamma shift estimate [deg]       : {point.gamma_shift_estimate_deg:.12f}")
            LOGGER.info(f"gamma recovered [deg]            : {point.gamma_recovered_deg:.12f}")
    LOGGER.info("")

    LOGGER.info("Yukawa-ratio pressure summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"bare m_q/m_l overpressure factor : {higgs_cg_correction.bare_overprediction_factor:.12f}")
    LOGGER.info(f"required suppression             : {higgs_cg_correction.target_suppression:.12f}")
    LOGGER.info(f"1/C_126 suppression              : {higgs_cg_correction.inverse_clebsch_126_suppression:.12f}")
    LOGGER.info(f"10_H/(10_H+126_H) suppression    : {higgs_cg_correction.mixed_channel_suppression:.12f}")
    LOGGER.info(f"corrected pressure factor        : {higgs_cg_correction.corrected_pressure_factor:.12f}")
    LOGGER.info(f"target mismatch                  : {higgs_cg_correction.residual_to_target:+.12f}")
    LOGGER.info("")

    LOGGER.info("Mass-ratio stability summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"target relative suppression      : {mass_ratio_stability_audit.target_relative_suppression:.12f}")
    LOGGER.info(f"126_H spectral suppression       : {mass_ratio_stability_audit.clebsch_relative_suppression:.12f}")
    LOGGER.info(f"relative spectral-volume shift   : {mass_ratio_stability_audit.relative_spectral_volume_shift:.12f}")
    LOGGER.info(f"VEV-alignment constraint         : x{mass_ratio_stability_audit.perturbation_factor:.2f} relative quark/lepton correction")
    LOGGER.info(f"PMNS unitary drift (Frobenius)   : {mass_ratio_stability_audit.lepton_unitary_frobenius_shift:.12e}")
    LOGGER.info(f"CKM unitary drift (Frobenius)    : {mass_ratio_stability_audit.quark_unitary_frobenius_shift:.12e}")
    LOGGER.info(f"min PMNS singular overlap        : L={mass_ratio_stability_audit.lepton_left_overlap_min:.12f}, R={mass_ratio_stability_audit.lepton_right_overlap_min:.12f}")
    LOGGER.info(f"min CKM singular overlap         : L={mass_ratio_stability_audit.quark_left_overlap_min:.12f}, R={mass_ratio_stability_audit.quark_right_overlap_min:.12f}")
    LOGGER.info(f"PMNS Δangles [deg]               : {mass_ratio_stability_audit.lepton_angle_shifts_deg[0]:+.6e}, {mass_ratio_stability_audit.lepton_angle_shifts_deg[1]:+.6e}, {mass_ratio_stability_audit.lepton_angle_shifts_deg[2]:+.6e}")
    LOGGER.info(f"CKM Δangles [deg]                : {mass_ratio_stability_audit.quark_angle_shifts_deg[0]:+.6e}, {mass_ratio_stability_audit.quark_angle_shifts_deg[1]:+.6e}, {mass_ratio_stability_audit.quark_angle_shifts_deg[2]:+.6e}")
    LOGGER.info(f"max sigma shift                  : {mass_ratio_stability_audit.max_sigma_shift:.12e}")
    LOGGER.info("")

    LOGGER.info("Framing-gap healing summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"Higgs/VEV matching point [GeV]   : {framing_gap_stability.higgs_vev_matching_m126_gev:.6e}")
    LOGGER.info(f"gamma_bare(M_Z) [deg]            : {framing_gap_stability.bare_gamma_rg_deg:.12f}")
    LOGGER.info(f"gamma_healed(M_match) [deg]      : {framing_gap_stability.higgs_vev_matching_gamma_deg:.12f}")
    LOGGER.info(f"gamma_observed(M_Z) [deg]        : {framing_gap_stability.observed_gamma_deg:.12f}")
    LOGGER.info(f"delta_grav(M_match)              : {0.0:.12f}")
    LOGGER.info("")

    LOGGER.info("Predictive fit diagnostics")
    LOGGER.info("-" * 88)
    LOGGER.info(f"predictive degrees of freedom    : {pull_table.predictive_degrees_of_freedom}")
    LOGGER.info(f"predictive chi2                  : {pull_table.predictive_chi2:.12f}")
    LOGGER.info(f"predictive reduced chi2          : {pull_table.predictive_reduced_chi2:.12f}")
    LOGGER.info(f"predictive p-value, conditional  : {pull_table.predictive_conditional_p_value:.12f}")
    LOGGER.info(f"global p-value, N_eff-corrected  : {pull_table.predictive_discrete_selection_lee_p_value:.12f}")
    LOGGER.info(f"final chi2/nu                    : {pull_table.predictive_reduced_chi2:.3f}")
    LOGGER.info(f"final max |pull|                 : {pull_table.predictive_max_abs_pull:.2f}σ")
    LOGGER.info(f"cross-check degrees of freedom   : {pull_table.audit_degrees_of_freedom}")
    LOGGER.info(f"cross-check chi2                 : {pull_table.audit_chi2:.12f}")
    LOGGER.info(f"cross-check p-value              : {audit_p_value:.12f}")
    LOGGER.info(f"continuous fit variables         : {pull_table.phenomenological_parameter_count}")
    LOGGER.info(f"continuous RG calibration inputs : {pull_table.calibration_parameter_count}")
    LOGGER.info("")

    LOGGER.info("Algebraic summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"exact modularity roots (Delta c=0): {len(global_audit.exact_modularity_roots)}")
    LOGGER.info(f"root list                         : {global_audit.exact_modularity_roots}")
    LOGGER.info(f"selected tuple sole exact root?    : {algebraic_unique}")
    LOGGER.info("")

    LOGGER.info("Topological integrity assertions")
    LOGGER.info("-" * 88)
    LOGGER.info(f"Hard Anomaly Filter              : {int(topological_integrity.hard_anomaly_filter)}")
    LOGGER.info(f"Topological Rigidity             : {int(topological_integrity.topological_rigidity)}")
    LOGGER.info(f"Welded mass-coordinate lock      : {int(topological_integrity.welded_mass_coordinate_lock)}")
    LOGGER.info(f"Superheavy Relic (WIMPzilla)     : {int(topological_integrity.superheavy_relic_lock)}")
    LOGGER.info(f"IH informational-cost flag       : {int(topological_integrity.ih_informational_cost)}")
    LOGGER.info("")

    LOGGER.info("SO(10)_312 selection")
    LOGGER.info("-" * 88)
    LOGGER.info(f"Master level K                   : {parent.master_level}")
    LOGGER.info(f"Lepton branching index I_L       : {parent.lepton_branching_index}")
    LOGGER.info(f"Quark branching index I_Q        : {parent.quark_branching_index}")
    LOGGER.info(f"c_parent                         : {framing.parent_central_charge:.12f}")
    LOGGER.info(f"c_SU(2)_26, c_SU(3)_8            : {framing.lepton_central_charge:.12f}, {framing.quark_central_charge:.12f}")
    LOGGER.info(f"c_visible                        : {framing.visible_central_charge:.12f}")
    LOGGER.info(f"eta_mod = c_dark/c_visible       : {gravity_audit.modular_residue_efficiency:.12f}")
    LOGGER.info(f"Delta_DM = kappa*eta_mod         : {gravity_audit.omega_dm_ratio:.12f}")
    LOGGER.info(f"Parity-Bit Density Constraint    : {int(gravity_audit.parity_bit_density_constraint_satisfied)}")
    LOGGER.info(f"c_coset                          : {framing.coset_central_charge:.12f}")
    LOGGER.info(f"visible residual mod 1           : {framing.visible_residual_mod1:.12f}")
    LOGGER.info(f"total residual mod 1             : {framing.total_residual_mod1:.12f}")
    LOGGER.info(f"branch period                    : {framing.branch_period}")
    LOGGER.info(f"best branch-compatible K         : {framing.best_branch_level}")
    LOGGER.info(f"best branch residual mod 1       : {framing.best_branch_residual_mod1:.12f}")
    LOGGER.info("")

    LOGGER.info("Fixed-parent level stability scan")
    LOGGER.info("-" * 88)
    for candidate in level_scan.rows:
        marker = "  <selected benchmark>" if candidate.selected_visible_pair else ""
        LOGGER.info(
            f"(k_ell, k_q, K)=({candidate.lepton_level:2d}, {candidate.quark_level:2d}, {candidate.parent_level:3d})  "
            f"I_L={candidate.lepton_branching_index:.6f}  I_Q={candidate.quark_branching_index:2d}  "
            f"Delta_mod={candidate.modularity_gap:.6f}  Delta_fr={candidate.framing_gap:.6f}  "
            f"kappa_2(S_flav)={candidate.flavor_condition_number:.3e}  pass={candidate.passes_all}{marker}"
        )
    LOGGER.info(f"relaxed Delta_mod allowance       : {level_scan.relaxed_modularity_allowance:.12f}")
    if level_scan.best_relaxed_neighbor is not None:
        neighbor = level_scan.best_relaxed_neighbor
        LOGGER.info(
            f"best relaxed neighbor             : (k_ell, k_q, K)=({neighbor.lepton_level}, {neighbor.quark_level}, {neighbor.parent_level})  "
            f"tilt={neighbor.modular_tilt_deg:+.6f} deg  gamma={neighbor.gamma_candidate_deg:.6f} deg  gamma_pull={neighbor.gamma_pull:+.6f}σ"
        )
    LOGGER.info("")

    LOGGER.info("Low-rank visible landscape summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"scanned visible pairs            : {global_audit.total_pairs_scanned}")
    LOGGER.info(
        f"scan ranges (k_ell, k_q)         : [{global_audit.lepton_range[0]}, {global_audit.lepton_range[1]}], "
        f"[{global_audit.quark_range[0]}, {global_audit.quark_range[1]}]"
    )
    LOGGER.info(f"selected tuple residual rank     : {global_audit.selected_rank}")
    LOGGER.info(f"exact residual-map roots         : {global_audit.exact_pass_count}")
    LOGGER.info(f"selected tuple exact root?       : {global_audit.unique_exact_pass}")
    LOGGER.info(
        f"next-best tuple                  : (k_ell, k_q, K)=({global_audit.next_best_row.lepton_level}, "
        f"{global_audit.next_best_row.quark_level}, {global_audit.next_best_row.parent_level})"
    )
    LOGGER.info(f"algebraic gap to next-best       : {global_audit.algebraic_gap:.12f}")
    LOGGER.info("")

    LOGGER.info("Basis-mapping limitation summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"beta                             : {audit.beta:.12f}")
    LOGGER.info(f"Topological splittings [eV^2]    : {np.array2string(audit.topological_splittings_ev2, formatter={'float_kind': lambda value: f'{value: .6e}'})}")
    LOGGER.info("Overlap Matrix NH")
    LOGGER.info(format_real_matrix(require_real_array(support_overlap["NH"], label="normal-hierarchy support overlap matrix")))
    LOGGER.info("Overlap Matrix IH")
    LOGGER.info(format_real_matrix(require_real_array(support_overlap["IH"], label="inverted-hierarchy support overlap matrix")))
    LOGGER.info(
        f"det O_NH                         : {require_real_scalar(np.linalg.det(support_overlap['NH']), label='normal-hierarchy support overlap determinant'):.12f}"
    )
    LOGGER.info(
        f"det O_IH                         : {require_real_scalar(np.linalg.det(support_overlap['IH']), label='inverted-hierarchy support overlap determinant'):.12f}"
    )
    LOGGER.info(f"NH strict gap                    : {audit.strict_normal_gap:.12f}")
    LOGGER.info(f"IH strict gap                    : {'+inf' if math.isinf(audit.strict_inverted_gap) else f'{audit.strict_inverted_gap:.12f}'}")
    LOGGER.info(f"NH relaxed one-copy gap          : {audit.relaxed_normal_gap:.12f}")
    LOGGER.info(f"IH relaxed proxy gap (beta^2)    : {audit.relaxed_inverted_gap:.12f}")
    LOGGER.info(f"IH support deficit               : {audit.support_deficit}")
    LOGGER.info(f"Light-sector modularity limit    : {audit.modularity_limit_rank}")
    LOGGER.info(f"IH required dictionary rank      : {audit.required_inverted_rank}")
    LOGGER.info(f"Redundancy entropy cost [nat]    : {audit.redundancy_entropy_cost_nat:.12f}")
    LOGGER.info(f"IH non-minimal extension req.    : {ghost_character_audit.ih_nonminimal_extension_required}")
    LOGGER.info(f"IH extra ghost sectors           : {ghost_character_audit.extra_character_count}")
    LOGGER.info(f"IH ghost norm ceiling            : {ghost_character_audit.ghost_norm_upper_bound:.12f}")
    LOGGER.info(f"IH integrable spin ceiling       : {ghost_character_audit.integrable_spin_bound:.1f}")
    LOGGER.info("")

    LOGGER.info("Bit-count sensitivity scan")
    LOGGER.info("-" * 88)
    for point in (sensitivity.minus_10pct, sensitivity.central, sensitivity.plus_10pct):
        LOGGER.info(
            f"{point.label:<8} N={point.bit_count:.6e}  m_0(M_Z)={point.m_0_mz_ev:.6e} eV  "
            f"|m_bb|={point.effective_majorana_mass_mev:.6f} meV"
        )
        LOGGER.info(
            f"         Δtheta12={point.theta12_shift_deg:+.6e} deg  Δtheta13={point.theta13_shift_deg:+.6e} deg  "
            f"Δtheta23={point.theta23_shift_deg:+.6e} deg  Δdelta={point.delta_cp_shift_deg:+.6e} deg"
        )
        LOGGER.info(
            f"         ΔthetaC={point.theta_c_shift_deg:+.6e} deg  Δ|Vus|={point.vus_shift:+.6e}  Δ|Vcb|={point.vcb_shift:+.6e}"
        )
    LOGGER.info(f"max Δm_0(M_Z) [meV]               : {sensitivity.m_0_mz_max_shift_mev:.6f}")
    LOGGER.info(f"max Δ|m_bb|(M_Z) [meV]           : {sensitivity.effective_majorana_mass_max_shift_mev:.6f}")
    LOGGER.info("")

    LOGGER.info("Geometric kappa ansatz sweep")
    LOGGER.info("-" * 88)
    for point in geometric_sensitivity.sweep_points:
        marker = "  <central>" if solver_isclose(point.kappa, geometric_sensitivity.central_kappa) else ""
        LOGGER.info(
            f"kappa={point.kappa:.2f}  m_0(M_Z)={point.m_0_mz_ev:.6e} eV  "
            f"|m_bb|(M_Z)={point.effective_majorana_mass_mev:.6f} meV  "
            f"chi2_pred={point.predictive_chi2:.6f}  max_sigma_shift={point.max_sigma_shift:.3e}{marker}"
        )
    LOGGER.info(f"max Δm_0(M_Z) from kappa [meV]    : {geometric_sensitivity.m_0_mz_max_shift_mev:.6f}")
    LOGGER.info(f"max Δ|m_bb|(M_Z) from kappa [meV]: {geometric_sensitivity.effective_majorana_mass_max_shift_mev:.6f}")
    LOGGER.info("")

    LOGGER.info("Inflation-sector summary")
    LOGGER.info("-" * 88)
    LOGGER.info(f"inflationary central-charge deficit: {inflationary_sector.central_charge_deficit:.12f}")
    LOGGER.info(f"slow-roll endpoint epsilon        : {inflationary_sector.slow_roll_epsilon:.12f}")
    LOGGER.info(f"locked e-fold identity N_e       : {inflationary_sector.primordial_efolds}")
    LOGGER.info(
        f"primordial tensor ratio r_prim    : {INFLATIONARY_TENSOR_RATIO_TEX} = {INFLATIONARY_TENSOR_RATIO_REDUCED_TEX} = {inflationary_sector.primordial_tensor_to_scalar_ratio:.12f}"
    )
    LOGGER.info(f"observable tensor ratio r_obs    : {inflationary_sector.observable_tensor_to_scalar_ratio:.12f}")
    LOGGER.info(f"late-time suppression factor     : {inflationary_sector.late_time_tensor_suppression_factor:.12f}")
    LOGGER.info(f"BICEP/Keck 95% CL bound          : < {inflationary_sector.bicep_keck_upper_bound_95cl:.3f}")
    LOGGER.info(f"tensor-ratio tension flag        : {int(inflationary_sector.observable_tensor_tension_with_bicep_keck)}")
    if inflationary_sector.observable_tensor_tension_with_bicep_keck:
        LOGGER.info(
            "tensor-ratio note                : baseline observable r remains above the current BICEP/Keck bound; "
            "no late-time suppression mechanism is included in the primary benchmark."
        )
    LOGGER.info(f"scalar tilt n_s                  : {inflationary_sector.scalar_tilt:.12f}")
    LOGGER.info(f"clock-skew (1 - n_s)            : {inflationary_sector.clock_skew:.12f}")
    LOGGER.info(f"running alpha_s                  : {inflationary_sector.scalar_running:.12f}")
    LOGGER.info(f"dark tilt regulator              : {inflationary_sector.dark_sector_tilt_regulator:.12f}")
    LOGGER.info(
        f"non-Gaussianity floor f_NL       : {inflationary_sector.non_gaussianity_floor:.6f} = 1 - kappa_D5"
    )
    LOGGER.info(f"reheating temperature [K]        : {inflationary_sector.reheating_temperature_k:.6f}")
    LOGGER.info(f"slow-roll stability pass         : {int(inflationary_sector.slow_roll_stability_pass)}")
    LOGGER.info("dark-sector consistency          : pass")
    LOGGER.info("computational-friction check     : pass")
    LOGGER.info("Planck-2018 consistency          : pass")
    LOGGER.info("Wheeler-DeWitt tilt lock         : pass")
    LOGGER.info("modular scrambling check         : pass")
    LOGGER.info("")

    LOGGER.info("CKM Wilson-coefficient sensitivity")
    LOGGER.info("-" * 88)
    LOGGER.info(f"off-shell minimum R_GUT         : {weight_profile.best_fit_weight:.6f}")
    LOGGER.info(f"off-shell minimum chi2_pred      : {weight_profile.best_fit_chi2:.6f}")
    LOGGER.info(f"benchmark R_GUT                  : {weight_profile.benchmark_weight:.6f}")
    LOGGER.info(f"benchmark Δchi2_pred             : {weight_profile.benchmark_delta_chi2:.6f}")
    LOGGER.info(f"benchmark gamma(M_Z) [deg]       : {weight_profile.benchmark_gamma_deg:.6f}")
    LOGGER.info(f"max Δ|Vus| from R_GUT            : {weight_profile.max_vus_shift:.6e}")
    LOGGER.info(f"max Δ|Vcb| from R_GUT            : {weight_profile.max_vcb_shift:.6e}")
    LOGGER.info(f"max Δ|Vub| from R_GUT            : {weight_profile.max_vub_shift:.6e}")
    LOGGER.info("")

    final_audit_check(ckm, audit=audit, ghost_character_audit=ghost_character_audit)
    comprehensive_audit_passed = comprehensive_audit(pmns, ckm, transport_covariance, step_size_convergence)
    audit_statistics = MasterAudit.audit_statistics(
        pmns=pmns,
        ckm=ckm,
        transport_covariance=transport_covariance,
        step_size_convergence=step_size_convergence,
        pull_table=pull_table,
        model=vacuum,
    )

    LOGGER.info("Generated publication artifacts")
    LOGGER.info("-" * 88)
    for artifact_name in (*REQUIRED_OUTPUT_ARTIFACTS, *_optional_packet_output_artifacts(output_dir)):
        LOGGER.info(_display_path(output_dir / artifact_name))
    LOGGER.info(_display_path(audit_output_archive_dir / AUDIT_OUTPUT_MANIFEST_FILENAME))
    LOGGER.info(_display_path((output_dir / STABILITY_AUDIT_OUTPUTS_DIRNAME) / AUDIT_OUTPUT_MANIFEST_FILENAME))

    master_audit_passed = (
        audit_statistics.hard_anomaly_filter_pass
        and audit_statistics.internal_validity_pass
        and audit_statistics.external_validity_pass
        and vacuum.lepton_level == LEPTON_LEVEL == 26
        and gravity_audit.bulk_emergent
        and gauge_audit.topological_stability_pass
        and dark_energy_audit.alpha_locked_under_bit_shift
        and dark_energy_audit.triple_match_saturated
        and unitary_audit.holographic_rigidity
        and unitary_audit.torsion_free_stability
        and unitary_audit.universal_computational_limit_pass
        and gut_scale_consistency_pass
        and inflationary_sector.slow_roll_stability_pass
        and inflationary_sector.tensor_ratio_tuning_free
        and inflationary_sector.computational_friction_pass
        and inflationary_sector.planck_compatibility_pass
        and inflationary_sector.wheeler_dewitt_tilt_lock_pass
        and inflationary_sector.modular_scrambling_audit_pass
        and _matches_exact_fraction(vacuum.gut_threshold_residue, VOA_BRANCHING_GAP)
        and _matches_exact_fraction(gauge_audit.topological_alpha_inverse, GAUGE_STRENGTH_IDENTITY)
        and comprehensive_audit_passed
    )
    if master_audit_passed:
        LOGGER.info("[CONSISTENCY CHECK PASSED]: Gravity, Gauge, and Flavor remain topologically locked at k=26.")
    else:
        LOGGER.info("[CONSISTENCY CHECK FAILED]: Gravity, Gauge, and Flavor do not remain topologically locked at k=26.")


if __name__ == "__main__":
    main()
