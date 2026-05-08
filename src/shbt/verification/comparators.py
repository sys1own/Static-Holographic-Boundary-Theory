from __future__ import annotations

from dataclasses import dataclass
import math
from scipy.constants import electron_mass, proton_mass

from shbt.config_loader import DEFAULT_CONFIG_LOADER


@dataclass(frozen=True)
class EmpiricalComparator:
    label: str
    value: float
    sigma: float
    release: str
    reference: str | None = None


@dataclass(frozen=True)
class Tier2ComparatorBundle:
    codata_alpha_inverse: EmpiricalComparator
    codata_proton_electron_mass_ratio: EmpiricalComparator
    planck2018_h0_km_s_mpc: EmpiricalComparator
    planck2018_lambda_si_m2: EmpiricalComparator
    nufit_solar_mass_splitting_ev2: EmpiricalComparator
    nufit_atmospheric_mass_splitting_ev2: EmpiricalComparator


def _require_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return value


def _coerce_float(config: dict[str, object], key: str) -> float:
    return float(config[key])


def _coerce_str(config: dict[str, object], key: str) -> str:
    return str(config[key])


def _coerce_interval_sigma(config: dict[str, object], key: str) -> float:
    values = config.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise TypeError(f"Interval '{key}' must be a length-2 sequence")
    lower, upper = (float(value) for value in values)
    return 0.5 * abs(upper - lower) / 3.0


def _relative_sigma_floor(value: float, *, floor: float = 1.0e-3) -> float:
    return max(abs(float(value)) * floor, floor)


_PHYSICS_CONFIG = DEFAULT_CONFIG_LOADER.load_physics_constants()
_EXPERIMENTAL_CONFIG = DEFAULT_CONFIG_LOADER.load_experimental_bounds()
_RELEASES_CONFIG = _require_mapping(_EXPERIMENTAL_CONFIG, "releases")
_PHYSICAL_CONSTANTS_CONFIG = _require_mapping(_PHYSICS_CONFIG, "physical_constants")
_UNCERTAINTIES_CONFIG = _require_mapping(_PHYSICS_CONFIG, "uncertainties")
_NORMAL_ORDERING_MASS_SPLITTINGS_CONFIG = _require_mapping(_EXPERIMENTAL_CONFIG, "normal_ordering_mass_splittings_ev2")
_NUFIT_3SIGMA_NORMAL_ORDERING_CONFIG = _require_mapping(_EXPERIMENTAL_CONFIG, "nufit_3sigma_normal_ordering")

PLANCK2018_H0_KM_S_MPC = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_h0_km_s_mpc")
PLANCK2018_H0_SIGMA_KM_S_MPC = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_h0_sigma_km_s_mpc")
PLANCK2018_OMEGA_LAMBDA = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_omega_lambda")
PLANCK2018_OMEGA_LAMBDA_SIGMA = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_omega_lambda_sigma")
PLANCK2018_LAMBDA_SI_M2 = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_lambda_si_m2")
PLANCK2018_LAMBDA_FRACTIONAL_SIGMA = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_lambda_fractional_sigma")
PLANCK2018_ALPHA_EM_INV_MZ = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_alpha_em_inv_mz")
PLANCK2018_SIN2_THETA_W_MZ = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_sin2_theta_w_mz")
PLANCK2018_ALPHA_S_MZ = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "planck2018_alpha_s_mz")
PLANCK2018_ALPHA_S_MZ_SIGMA = _coerce_float(_UNCERTAINTIES_CONFIG, "alpha_s_mz_sigma")

CODATA_FINE_STRUCTURE_ALPHA_INVERSE = _coerce_float(_PHYSICAL_CONSTANTS_CONFIG, "codata_fine_structure_alpha_inverse")
CODATA_PROTON_TO_ELECTRON_MASS_RATIO = float(proton_mass / electron_mass)

SOLAR_MASS_SPLITTING_EV2 = _coerce_float(_NORMAL_ORDERING_MASS_SPLITTINGS_CONFIG, "solar")
ATMOSPHERIC_MASS_SPLITTING_NO_EV2 = _coerce_float(_NORMAL_ORDERING_MASS_SPLITTINGS_CONFIG, "atmospheric")
SOLAR_MASS_SPLITTING_SIGMA_EV2 = _coerce_interval_sigma(_NUFIT_3SIGMA_NORMAL_ORDERING_CONFIG, "dm21")
ATMOSPHERIC_MASS_SPLITTING_NO_SIGMA_EV2 = _coerce_interval_sigma(_NUFIT_3SIGMA_NORMAL_ORDERING_CONFIG, "dm31")

PLANCK2018_RELEASE = "Planck 2018 benchmark-config comparator"
CODATA_RELEASE = "CODATA benchmark-config comparator"
NUFIT_RELEASE = _coerce_str(_RELEASES_CONFIG, "nufit_release")
NUFIT_REFERENCE = _coerce_str(_RELEASES_CONFIG, "nufit_reference")

PLANCK2018_H0_COMPARATOR = EmpiricalComparator(
    label="Planck 2018 H0",
    value=PLANCK2018_H0_KM_S_MPC,
    sigma=PLANCK2018_H0_SIGMA_KM_S_MPC,
    release=PLANCK2018_RELEASE,
)
PLANCK2018_LAMBDA_COMPARATOR = EmpiricalComparator(
    label="Planck 2018 Lambda",
    value=PLANCK2018_LAMBDA_SI_M2,
    sigma=abs(PLANCK2018_LAMBDA_SI_M2) * PLANCK2018_LAMBDA_FRACTIONAL_SIGMA,
    release=PLANCK2018_RELEASE,
)
CODATA_ALPHA_INVERSE_COMPARATOR = EmpiricalComparator(
    label="CODATA alpha^-1",
    value=CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    sigma=_relative_sigma_floor(CODATA_FINE_STRUCTURE_ALPHA_INVERSE),
    release=CODATA_RELEASE,
)
CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR = EmpiricalComparator(
    label="CODATA m_p/m_e",
    value=CODATA_PROTON_TO_ELECTRON_MASS_RATIO,
    sigma=_relative_sigma_floor(CODATA_PROTON_TO_ELECTRON_MASS_RATIO),
    release=CODATA_RELEASE,
)
NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR = EmpiricalComparator(
    label="NuFIT Delta m21^2",
    value=SOLAR_MASS_SPLITTING_EV2,
    sigma=SOLAR_MASS_SPLITTING_SIGMA_EV2,
    release=NUFIT_RELEASE,
    reference=NUFIT_REFERENCE,
)
NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR = EmpiricalComparator(
    label="NuFIT Delta m31^2",
    value=ATMOSPHERIC_MASS_SPLITTING_NO_EV2,
    sigma=ATMOSPHERIC_MASS_SPLITTING_NO_SIGMA_EV2,
    release=NUFIT_RELEASE,
    reference=NUFIT_REFERENCE,
)

TIER2_COMPARATORS = Tier2ComparatorBundle(
    codata_alpha_inverse=CODATA_ALPHA_INVERSE_COMPARATOR,
    codata_proton_electron_mass_ratio=CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR,
    planck2018_h0_km_s_mpc=PLANCK2018_H0_COMPARATOR,
    planck2018_lambda_si_m2=PLANCK2018_LAMBDA_COMPARATOR,
    nufit_solar_mass_splitting_ev2=NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR,
    nufit_atmospheric_mass_splitting_ev2=NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR,
)


__all__ = [
    "ATMOSPHERIC_MASS_SPLITTING_NO_EV2",
    "ATMOSPHERIC_MASS_SPLITTING_NO_SIGMA_EV2",
    "CODATA_ALPHA_INVERSE_COMPARATOR",
    "CODATA_FINE_STRUCTURE_ALPHA_INVERSE",
    "CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR",
    "CODATA_PROTON_TO_ELECTRON_MASS_RATIO",
    "CODATA_RELEASE",
    "EmpiricalComparator",
    "NUFIT_ATMOSPHERIC_MASS_SPLITTING_COMPARATOR",
    "NUFIT_REFERENCE",
    "NUFIT_RELEASE",
    "NUFIT_SOLAR_MASS_SPLITTING_COMPARATOR",
    "PLANCK2018_ALPHA_EM_INV_MZ",
    "PLANCK2018_ALPHA_S_MZ",
    "PLANCK2018_ALPHA_S_MZ_SIGMA",
    "PLANCK2018_H0_COMPARATOR",
    "PLANCK2018_H0_KM_S_MPC",
    "PLANCK2018_H0_SIGMA_KM_S_MPC",
    "PLANCK2018_LAMBDA_COMPARATOR",
    "PLANCK2018_LAMBDA_FRACTIONAL_SIGMA",
    "PLANCK2018_LAMBDA_SI_M2",
    "PLANCK2018_OMEGA_LAMBDA",
    "PLANCK2018_OMEGA_LAMBDA_SIGMA",
    "PLANCK2018_RELEASE",
    "PLANCK2018_SIN2_THETA_W_MZ",
    "SOLAR_MASS_SPLITTING_EV2",
    "SOLAR_MASS_SPLITTING_SIGMA_EV2",
    "TIER2_COMPARATORS",
    "Tier2ComparatorBundle",
]
